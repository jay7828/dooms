// Latency of the SmolVLM pipeline (photo + question -> answer) on one photo, split into the stages that matter.
//
//   bench_vlm --models DIR --image photo.jpg [--prompt TEXT] [--max-tokens N] [--vision-units U] [--text FILE] [--threads N] [--warmup N] [--iters N] [--stream N] [--decode-only 1] [--device-sampling 0|1] [--text-engine auto|metal|llama.cpp] [--queue-depth N] [--pause-ms N] [--spin-wait 0|1]
// --pause-ms N sleeps N ms between runs (an idle gap like separate requests have; back to back, an engine that keeps CPU threads spinning starts the next
// run's CPU preprocessing on warmer cores).
// --decode-only 1 encodes the photo once and times only the language-model part (prefill + decode) on those embeddings, to separate it from the vision stage.
// --stream N additionally times N back-to-back requests (the same photo and question) two ways: one generate() after another, and generate_many() with the
// next image's preprocessing + Neural Engine work overlapping the GPU decoding of the current answer. Reports p50 / p99 of the total for each.
// Each run is a complete generate() call with greedy decoding (natural stopping, or N tokens). Reports p50 / p99 of: preprocessing, vision tower +
// connector, prefill (prompt + image through the language model), time to first token (preprocess + vision + prefill + first sample), decode ms per
// token. Exit codes: 0 ok, 1 failed, 2 usage error.
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/stats.h"
#include "pipeline/image_io.h"
#include "pipeline/vlm_pipeline.h"

using namespace dooms;

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

struct Options {
    std::string models, image, text, prompt = "Describe this image in one sentence.";
    VlmOptions vlm;
    int max_tokens = 64, warmup = 3, iters = 20, stream = 0, decode_only = 0, device_sampling = 0, pause_ms = 0, spin_wait = 1;
};

Result<long> parse_int(const std::string & flag, const char * text, long lo, long hi) {
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return invalid_argument(flag + ": '" + text + "' is not an integer");
    if (v < lo || v > hi) return invalid_argument(flag + ": " + std::string(text) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return v;
}

Result<Options> parse_args(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) return invalid_argument("missing value for " + flag);
        const char * value = argv[i + 1];
        if (flag == "--models") { o.models = value; continue; }
        if (flag == "--image") { o.image = value; continue; }
        if (flag == "--text") { o.text = value; continue; }  // another language-model file (a quantized one) instead of DIR/text-f16.gguf
        if (flag == "--prompt") { o.prompt = value; continue; }
        if (flag == "--vision-units") { o.vlm.vision_units = value; continue; }
        if (flag == "--text-engine") { o.vlm.text_engine = value; continue; }
        long lo = 0, hi = 100000;
        int * dst = nullptr;
        if (flag == "--max-tokens") { lo = 1; hi = 4096; dst = &o.max_tokens; }
        else if (flag == "--iters") { lo = 1; dst = &o.iters; }
        else if (flag == "--warmup") dst = &o.warmup;
        else if (flag == "--decode-only") { hi = 1; dst = &o.decode_only; }
        else if (flag == "--spin-wait") { hi = 1; dst = &o.spin_wait; }
        else if (flag == "--pause-ms") { hi = 10000; dst = &o.pause_ms; }
        else if (flag == "--queue-depth") { lo = 1; hi = 64; dst = &o.vlm.metal_queue_depth; }
        else if (flag == "--device-sampling") { hi = 1; dst = &o.device_sampling; }  // greedy token picked on the GPU (TextDecoderOptions::device_sampling)
        else if (flag == "--stream") { lo = 1; hi = 1000; dst = &o.stream; }
        else if (flag == "--threads") { hi = 1024; dst = &o.vlm.text.n_threads; }
        else return invalid_argument("unknown option '" + flag + "'");
        auto v = parse_int(flag, value, lo, hi);
        if (!v.is_ok()) return v.status();
        *dst = static_cast<int>(v.value());
    }
    if (o.models.empty() || o.image.empty()) return invalid_argument("--models and --image are required");
    o.vlm.text.device_sampling = o.device_sampling == 1;
    o.vlm.metal_spin_wait = o.spin_wait == 1;
    return o;
}

void report(const char * name, const std::vector<double> & v) { std::printf("%-22s p50 %8.2f  p99 %8.2f ms\n", name, median(v), quantile(v, 0.99)); }

}  // namespace

int main(int argc, char ** argv) {
    auto args = parse_args(argc, argv);
    if (!args.is_ok()) { std::fprintf(stderr, "error: %s\n", args.status().to_string().c_str()); return kExitUsage; }
    const Options & o = args.value();
    auto image = load_image(o.image);
    if (!image.is_ok()) { std::fprintf(stderr, "error: %s\n", image.status().to_string().c_str()); return kExitFailed; }
    auto pipe = VlmPipeline::create({o.models + "/vision-fp16.mlpackage", (o.text.empty() ? o.models + "/text-f16.gguf" : o.text), o.models + "/vlm-config.gguf"}, o.vlm);
    if (!pipe.is_ok()) { std::fprintf(stderr, "error: %s\n", pipe.status().to_string().c_str()); return kExitFailed; }
    std::vector<double> pre, vis, prefill, first, per_token;
    VlmTimings last;
    std::vector<float> fixed_embeddings;
    if (o.decode_only) {
        Status st = pipe.value().encode_image(image.value(), &fixed_embeddings);
        if (!st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
    }
    for (int i = 0; i < o.warmup + o.iters; i++) {
        if (o.pause_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(o.pause_ms));
        VlmTimings tm;
        auto r = o.decode_only ? pipe.value().generate_from_embeddings(fixed_embeddings, o.prompt, o.max_tokens, nullptr, &tm)
                               : pipe.value().generate(image.value(), o.prompt, o.max_tokens, nullptr, &tm);
        if (!r.is_ok()) { std::fprintf(stderr, "error: %s\n", r.status().to_string().c_str()); return kExitFailed; }
        if (i < o.warmup) continue;
        pre.push_back(tm.preprocess_ms);
        vis.push_back(tm.vision_ms);
        prefill.push_back(tm.prefill_ms);
        first.push_back(tm.preprocess_ms + tm.vision_ms + tm.first_token_ms);
        if (tm.generated_tokens > 1) per_token.push_back(tm.decode_ms_per_token());
        last = tm;
    }
    std::printf("VLM, %lld prompt tokens, %lld generated, %d warmup + %d timed runs\n", static_cast<long long>(last.prompt_tokens), static_cast<long long>(last.generated_tokens), o.warmup, o.iters);
    report("preprocess", pre);
    report("vision tower", vis);
    report("prefill", prefill);
    report("time to first token", first);
    if (!per_token.empty()) { std::printf("%-22s p50 %8.2f  p99 %8.2f ms/token\n", "decode", median(per_token), quantile(per_token, 0.99)); }
    if (o.stream > 0) {
        std::vector<VlmRequest> reqs(static_cast<size_t>(o.stream), VlmRequest{&image.value(), o.prompt});
        std::vector<double> serial, overlapped;
        for (int i = 0; i < o.warmup + o.iters; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            for (const VlmRequest & r : reqs) {
                auto one = pipe.value().generate(*r.image, r.prompt, o.max_tokens);
                if (!one.is_ok()) { std::fprintf(stderr, "error: %s\n", one.status().to_string().c_str()); return kExitFailed; }
            }
            const double serial_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            double wall = 0;
            auto many = pipe.value().generate_many(reqs, o.max_tokens, &wall);
            if (!many.is_ok()) { std::fprintf(stderr, "error: %s\n", many.status().to_string().c_str()); return kExitFailed; }
            if (i < o.warmup) continue;
            serial.push_back(serial_ms);
            overlapped.push_back(wall);
        }
        std::printf("%d back-to-back requests, %d runs:\n", o.stream, o.iters);
        report("one after another", serial);
        report("overlapped", overlapped);
        std::printf("speedup (p50) %.2fx\n", median(serial) / median(overlapped));
    }
    return kExitOk;
}
