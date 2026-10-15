// The engine's own Metal decoder on a llama-architecture GGUF (f16): prefill of a prompt given as token ids, then greedy decode of a fixed number of tokens
// (end-of-generation ignored), p50/p99 over --iters runs; --profile prints the GPU time of one decode step per kernel kind.
//
//   bench_metal_llama --model FILE.gguf [--prompt-len N] [--gen N] [--depth N] [--warmup N] [--iters N] [--profile 1] [--spin-wait 0|1] [--tile 0|1|2]
// The prompt is token ids 1..N (content does not change the work). Exit codes: 0 ok, 1 failed, 2 usage error.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/stats.h"
#include "pipeline/native_llama.h"

using namespace dooms;

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

struct Options {
    std::string model;
    int prompt_len = 82, gen = 64, depth = 1, warmup = 3, iters = 20, profile = 0, spin_wait = 1, tile = 0;
};

double ms_since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }

}  // namespace

int main(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n", flag.c_str()); return kExitUsage; }
        if (flag == "--model") { o.model = argv[i + 1]; continue; }
        int * dst = flag == "--prompt-len" ? &o.prompt_len : flag == "--gen" ? &o.gen : flag == "--depth" ? &o.depth : flag == "--warmup" ? &o.warmup
                  : flag == "--iters" ? &o.iters : flag == "--profile" ? &o.profile : flag == "--spin-wait" ? &o.spin_wait : flag == "--tile" ? &o.tile : nullptr;
        char * end = nullptr;
        const long v = std::strtol(argv[i + 1], &end, 10);
        if (!dst || end == argv[i + 1] || *end != '\0' || v < 0 || v > 100000) { std::fprintf(stderr, "error: bad option %s %s\n", flag.c_str(), argv[i + 1]); return kExitUsage; }
        *dst = static_cast<int>(v);
    }
    if (o.model.empty() || o.prompt_len < 1 || o.gen < 1 || o.depth < 1 || o.iters < 1) {
        std::fprintf(stderr, "usage: bench_metal_llama --model FILE.gguf [--prompt-len N>=1] [--gen N>=1] [--depth N>=1] [--warmup N] [--iters N>=1] [--profile 1]\n");
        return kExitUsage;
    }
    NativeLlamaOptions no;
    no.n_ctx = o.prompt_len + o.gen + 1;
    no.spin_wait = o.spin_wait != 0;
    no.prefill_tile = o.tile;
    auto dec = load_native_llama(o.model, no);
    if (!dec.is_ok()) { std::fprintf(stderr, "error: %s\n", dec.status().to_string().c_str()); return kExitFailed; }
    MetalLlamaDecoder & d = *dec.value();
    std::vector<int32_t> prompt;
    for (int i = 0; i < o.prompt_len; i++) prompt.push_back(1 + i % (d.config().n_vocab - 1));
    std::vector<double> prefill, per_token;
    for (int it = 0; it < o.warmup + o.iters; it++) {
        d.reset();
        auto t0 = std::chrono::steady_clock::now();
        if (Status st = d.eval_tokens(prompt); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
        double pre = 0;
        auto t1 = t0;
        int n = 0;
        Status st = d.generate(o.gen, o.depth, [&](int32_t) {
            if (n++ == 0) { pre = ms_since(t0); t1 = std::chrono::steady_clock::now(); }
            return true;
        });
        if (!st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
        if (it >= o.warmup) { prefill.push_back(pre); per_token.push_back(ms_since(t1) / (o.gen - 1 > 0 ? o.gen - 1 : 1)); }
    }
    std::printf("prefill (%d tokens)     p50 %8.2f  p99 %8.2f ms\n", o.prompt_len, median(prefill), quantile(prefill, 0.99));
    std::printf("decode                 p50 %8.2f  p99 %8.2f ms/token\n", median(per_token), quantile(per_token, 0.99));
    if (o.profile) {
        for (int rows : {1, o.prompt_len}) {
            d.reset();
            if (Status st = d.eval_tokens(prompt); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
            if (rows > 1) d.reset();
            auto prof = d.profile_pass(rows);
            if (!prof.is_ok()) { std::fprintf(stderr, "error: %s\n", prof.status().to_string().c_str()); return kExitFailed; }
            double total = 0;
            for (const auto & k : prof.value()) total += k.gpu_ms;
            std::printf("%s, each dispatch timed alone (GPU ms):\n", rows == 1 ? "one decode step" : "prefill");
            for (const auto & k : prof.value()) std::printf("  %-12s %4d dispatches %8.3f ms  %5.1f%%\n", k.kind, k.dispatches, k.gpu_ms, 100.0 * k.gpu_ms / total);
            std::printf("  %-12s %4s            %8.3f ms\n", "total", "", total);
        }
    }
    return kExitOk;
}
