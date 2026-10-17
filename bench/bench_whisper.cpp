// Whisper speech to text on the engine (Core ML encoder + the own Metal decoder): each --audio file is decoded once, then transcribed --warmup + --iters times;
// p50 per stage, the real-time factor (processing time / audio duration) and the text. Audio file decoding is not timed (other engines' harnesses do the same).
//
//   bench_whisper --model DIR --audio FILE [--audio FILE]... [--units cpu_gpu|cpu_ne|all|cpu] [--warmup N] [--iters N]
// Exit codes: 0 ok, 1 failed, 2 usage error.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/stats.h"
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
    int warmup = 2, iters = 10;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n", flag.c_str()); return kExitUsage; }
        const char * value = argv[i + 1];
        if (flag == "--model") model = value;
        else if (flag == "--audio") audio.emplace_back(value);
        else if (flag == "--units") wo.encoder_units = value;
        else if (flag == "--warmup" || flag == "--iters") {
            char * end = nullptr;
            const long v = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || v < 0 || v > 100000) { std::fprintf(stderr, "error: bad option %s %s\n", flag.c_str(), value); return kExitUsage; }
            (flag == "--warmup" ? warmup : iters) = static_cast<int>(v);
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
    if (audio.size() > 1) std::printf("all files: %.1f s audio in %.0f ms (sum of p50s), RTF %.4f\n", all_audio, all_ms, all_ms / 1000.0 / all_audio);
    return kExitOk;
}
