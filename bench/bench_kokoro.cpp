// Kokoro-82M on the engine. --vocoder: the Metal vocoder alone on a reference case's own inputs (tools/parity/kokoro_reference.py), p50 over --iters runs; with
// --profile 1 also the GPU time per kernel kind (each dispatch in its own command buffer).
//
//   bench_kokoro --model DIR --vocoder 1 [--case K] [--warmup N] [--iters N] [--profile 1]
// Exit codes: 0 ok, 1 failed, 2 usage error.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "backend/metal/metal_kokoro_vocoder.h"
#include "core/stats.h"
#include "format/gguf.h"
#include "pipeline/kokoro_vocoder_loader.h"

using namespace dooms;

namespace {
constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

std::vector<float> tensor(const GgufFile & f, const std::string & name) {
    auto info = f.find_tensor(name);
    if (!info.is_ok()) return {};
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    return std::vector<float>(p, p + info.value()->shape.numel());
}
}  // namespace

int main(int argc, char ** argv) {
    std::string model;
    int vocoder = 0, kcase = 1, warmup = 3, iters = 20, profile = 0;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n", flag.c_str()); return kExitUsage; }
        if (flag == "--model") { model = argv[i + 1]; continue; }
        int * dst = flag == "--vocoder" ? &vocoder : flag == "--case" ? &kcase : flag == "--warmup" ? &warmup : flag == "--iters" ? &iters : flag == "--profile" ? &profile : nullptr;
        char * end = nullptr;
        const long v = std::strtol(argv[i + 1], &end, 10);
        if (!dst || end == argv[i + 1] || *end != '\0' || v < 0 || v > 100000) { std::fprintf(stderr, "error: bad option %s %s\n", flag.c_str(), argv[i + 1]); return kExitUsage; }
        *dst = static_cast<int>(v);
    }
    if (model.empty() || !vocoder || iters < 1) {
        std::fprintf(stderr, "usage: bench_kokoro --model DIR --vocoder 1 [--case K] [--warmup N] [--iters N>=1] [--profile 1]\n");
        return kExitUsage;
    }
    auto file = GgufFile::open(model + "/kokoro.gguf");
    auto ref = GgufFile::open(model + "/kokoro-reference.gguf");
    if (!file.is_ok() || !ref.is_ok()) { std::fprintf(stderr, "error: %s\n", (file.is_ok() ? ref.status() : file.status()).to_string().c_str()); return kExitFailed; }
    auto voc = load_kokoro_vocoder(file.value());
    if (!voc.is_ok()) { std::fprintf(stderr, "error: %s\n", voc.status().to_string().c_str()); return kExitFailed; }
    const std::string c = "case" + std::to_string(kcase) + ".";
    const GgufFile & r = ref.value();
    const std::vector<float> asr = tensor(r, c + "asr"), f0 = tensor(r, c + "f0"), n = tensor(r, c + "n"), s = tensor(r, c + "ref_s"), har = tensor(r, c + "har");
    if (f0.empty()) { std::fprintf(stderr, "error: no case %d in the reference\n", kcase); return kExitFailed; }
    const int32_t frames = static_cast<int32_t>(f0.size() / 2);
    std::vector<float> out;
    std::vector<double> ms;
    for (int it = 0; it < warmup + iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        if (Status st = voc.value()->run(asr.data(), frames, f0.data(), n.data(), s.data(), har.data(), &out); !st.is_ok()) {
            std::fprintf(stderr, "error: %s\n", st.to_string().c_str());
            return kExitFailed;
        }
        if (it >= warmup) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    const double audio_s = frames * 600.0 / 24000.0;
    std::printf("vocoder, case %d: %d frames (%.2f s of audio): p50 %.1f ms (p99 %.1f), RTF %.4f\n", kcase, frames, audio_s, median(ms), quantile(ms, 0.99), median(ms) / 1000.0 / audio_s);
    if (profile) {
        voc.value()->set_profiling(true);
        std::map<std::string, double> sum;
        std::vector<std::string> order;
        double total = 0;
        if (Status st = voc.value()->run(asr.data(), frames, f0.data(), n.data(), s.data(), har.data(), &out); !st.is_ok()) return kExitFailed;
        for (const auto & [kind, t] : voc.value()->profile()) {
            if (!sum.count(kind)) order.push_back(kind);
            sum[kind] += t;
            total += t;
        }
        std::printf("GPU time per kernel kind (each dispatch alone, %zu dispatches):\n", voc.value()->profile().size());
        for (const std::string & kind : order) std::printf("  %-22s %8.2f ms  (%4.1f%%)\n", kind.c_str(), sum[kind], 100.0 * sum[kind] / total);
        std::printf("  %-22s %8.2f ms\n", "total", total);
    }
    return kExitOk;
}
