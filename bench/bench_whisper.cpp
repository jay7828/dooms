// Whisper speech to text on the engine (Core ML encoder + the own Metal decoder): each --audio file is decoded once, then transcribed --warmup + --iters times;
// p50 per stage, the real-time factor (processing time / audio duration) and the text. Audio file decoding is not timed (other engines' harnesses do the same).
//
//   bench_whisper --model DIR --audio FILE [--audio FILE]... [--units cpu_gpu|cpu_ne|all|cpu] [--warmup N] [--iters N] [--profile 1]
//                 [--splits N] [--fused-ln -1|0|1]   decoder tuning (cross-attention position groups, 0 = automatic; LayerNorm placement, -1 = automatic)
// --profile 1: on the first file, the fixed cost of a window (cross keys/values + the prompt, wall time) and one decode step's GPU time per kernel kind
// (each dispatch in its own command buffer, summed over the layers, p50 of --iters steps).
// Exit codes: 0 ok, 1 failed, 2 usage error.
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "core/stats.h"
#include "backend/metal/metal_whisper_decoder.h"
#include "pipeline/audio_io.h"
#include "pipeline/whisper_pipeline.h"

using namespace dooms;

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

}  // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::vector<std::string> audio;
    WhisperOptions wo;
    int warmup = 2, iters = 10, profile = 0;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n", flag.c_str()); return kExitUsage; }
        const char * value = argv[i + 1];
        if (flag == "--model") model = value;
        else if (flag == "--audio") audio.emplace_back(value);
        else if (flag == "--units") wo.encoder_units = value;
        else if (flag == "--splits") wo.cross_splits = std::atoi(value);
        else if (flag == "--fused-ln") wo.fused_layernorm = std::atoi(value);
        else if (flag == "--warmup" || flag == "--iters" || flag == "--profile") {
            char * end = nullptr;
            const long v = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || v < 0 || v > 100000) { std::fprintf(stderr, "error: bad option %s %s\n", flag.c_str(), value); return kExitUsage; }
            (flag == "--warmup" ? warmup : flag == "--iters" ? iters : profile) = static_cast<int>(v);
        } else { std::fprintf(stderr, "error: unknown option %s\n", flag.c_str()); return kExitUsage; }
    }
    if (model.empty() || audio.empty() || iters < 1) {
        std::fprintf(stderr, "usage: bench_whisper --model DIR --audio FILE [--audio FILE]... [--units U] [--warmup N] [--iters N>=1]\n");
        return kExitUsage;
    }
    auto p = WhisperPipeline::create({model + "/encoder-fp16.mlpackage", model + "/decoder.gguf"}, wo);
    if (!p.is_ok()) { std::fprintf(stderr, "error: %s\n", p.status().to_string().c_str()); return kExitFailed; }
    double all_ms = 0, all_audio = 0;
    std::printf("encoder units %s, %d warm-up + %d timed runs per file\n", wo.encoder_units.c_str(), warmup, iters);
    for (const std::string & path : audio) {
        auto samples = load_audio(path, p.value().sample_rate());
        if (!samples.is_ok()) { std::fprintf(stderr, "error: %s: %s\n", path.c_str(), samples.status().to_string().c_str()); return kExitFailed; }
        std::vector<double> total, features, encoder, decode;
        WhisperTimings tm;
        std::string text;
        for (int it = 0; it < warmup + iters; it++) {
            auto r = p.value().transcribe(samples.value(), &tm);
            if (!r.is_ok()) { std::fprintf(stderr, "error: %s: %s\n", path.c_str(), r.status().to_string().c_str()); return kExitFailed; }
            if (it < warmup) continue;
            text = r.value().text;
            total.push_back(tm.total_ms);
            features.push_back(tm.features_ms);
            encoder.push_back(tm.encoder_ms);
            decode.push_back(tm.decode_ms);
        }
        const double t50 = median(total);
        all_ms += t50;
        all_audio += tm.audio_s;
        std::printf("%s: %.2f s audio, p50 %.1f ms (p99 %.1f), RTF %.4f | features %.1f, encoder %.1f, decoder %.1f ms | %lld windows, %lld tokens (%.2f ms/token incl. prompt)\n",
                    path.c_str(), tm.audio_s, t50, quantile(total, 0.99), t50 / 1000.0 / tm.audio_s, median(features), median(encoder), median(decode),
                    static_cast<long long>(tm.windows), static_cast<long long>(tm.tokens), tm.tokens > 0 ? median(decode) / static_cast<double>(tm.tokens) : 0.0);
        std::printf("  text:%s\n", text.c_str());
    }
    if (profile) {
        auto samples = load_audio(audio[0], p.value().sample_rate());
        if (!samples.is_ok()) { std::fprintf(stderr, "error: %s\n", samples.status().to_string().c_str()); return kExitFailed; }
        std::vector<float> mel, states;
        const size_t n = std::min<size_t>(samples.value().size(), static_cast<size_t>(p.value().window_samples()));
        if (Status st = p.value().features(samples.value().data(), n, &mel); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
        if (Status st = p.value().encode(mel, &states); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
        MetalWhisperDecoder & dec = p.value().decoder();
        const std::vector<int32_t> pr = p.value().prompt();
        std::vector<double> cross_ms, prompt_ms;
        for (int it = 0; it < warmup + iters; it++) {
            auto t0 = std::chrono::steady_clock::now();
            if (Status st = dec.set_audio(states.data()); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
            auto t1 = std::chrono::steady_clock::now();
            if (Status st = dec.eval_prompt(pr); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
            if (!dec.pending_token().is_ok()) return kExitFailed;
            auto t2 = std::chrono::steady_clock::now();
            if (it >= warmup) {
                cross_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                prompt_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
        }
        std::printf("fixed cost per window: set_audio (encode + host copy) %.2f ms, cross keys/values + prompt of %zu tokens to the first token %.2f ms (p50)\n",
                    median(cross_ms), pr.size(), median(prompt_ms));
        std::map<std::string, std::vector<double>> per_kind;
        std::vector<std::string> order;
        std::vector<double> totals;
        for (int it = 0; it < warmup + iters; it++) {
            auto prof = dec.profile_step();
            if (!prof.is_ok()) { std::fprintf(stderr, "error: %s\n", prof.status().to_string().c_str()); return kExitFailed; }
            if (it < warmup) continue;
            std::map<std::string, double> sum;
            double total = 0;
            for (const auto & [kind, ms] : prof.value()) {
                if (!sum.count(kind) && per_kind.find(kind) == per_kind.end() && std::find(order.begin(), order.end(), kind) == order.end()) order.push_back(kind);
                sum[kind] += ms;
                total += ms;
            }
            for (const auto & [kind, ms] : sum) per_kind[kind].push_back(ms);
            totals.push_back(total);
        }
        std::printf("one decode step, GPU time per kernel kind (each dispatch alone, summed over %d layers, p50 of %d):\n", dec.config().n_layer, iters);
        for (const std::string & kind : order) std::printf("  %-18s %7.3f ms  (%4.1f%%)\n", kind.c_str(), median(per_kind[kind]), 100.0 * median(per_kind[kind]) / median(totals));
        std::printf("  %-18s %7.3f ms\n", "total", median(totals));
    }
    if (audio.size() > 1) std::printf("all files: %.1f s audio in %.0f ms (sum of p50s), RTF %.4f\n", all_audio, all_ms, all_ms / 1000.0 / all_audio);
    return kExitOk;
}
