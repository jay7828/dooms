#include "cli_transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pipeline/whisper_pipeline.h"

namespace dooms {

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;
constexpr char kUsage[] =
    "usage: dooms-cli transcribe --model DIR --audio FILE [--audio FILE]... [--language CODE] [--task transcribe|translate] [--units U] [--max-tokens N] [--stats]\n"
    "  --model DIR     encoder-fp16.mlpackage + decoder.gguf from tools/convert/convert_whisper.py\n"
    "  --audio FILE    any file Core Audio decodes (WAV, MP3, M4A, FLAC, ...); one line of text per file\n"
    "  --language      a code the model knows (default en)     --task  transcribe (default) or translate (into English)\n"
    "  --units U       the encoder's Core ML units: cpu_gpu (default), cpu_ne, all, cpu\n"
    "  --max-tokens N  text tokens per 30-s window (default 224)     --stats  print the timings to stderr\n";

struct Args {
    std::string model;
    std::vector<std::string> audio;
    bool stats = false;
    WhisperOptions options;
};

Result<Args> parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        const std::string flag = argv[i];
        if (flag == "--stats") { a.stats = true; continue; }
        if (i + 1 >= argc) return invalid_argument("missing value for " + flag);
        const char * value = argv[++i];
        if (flag == "--model") a.model = value;
        else if (flag == "--audio") a.audio.emplace_back(value);
        else if (flag == "--language") a.options.language = value;
        else if (flag == "--task") a.options.task = value;
        else if (flag == "--units") a.options.encoder_units = value;
        else if (flag == "--max-tokens") {
            char * end = nullptr;
            const long v = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || v < 1 || v > 1 << 20) return invalid_argument("--max-tokens: '" + std::string(value) + "' is not a positive integer");
            a.options.max_tokens_per_window = static_cast<int32_t>(v);
        } else return invalid_argument("unknown option '" + flag + "'");
    }
    if (a.model.empty() || a.audio.empty()) return invalid_argument("--model and --audio are required");
    return a;
}

}  // namespace

int run_transcribe_command(int argc, char ** argv) {
    auto args = parse(argc, argv);
    if (!args.is_ok()) {
        std::fprintf(stderr, "error: %s\n%s", args.status().to_string().c_str(), kUsage);
        return kExitUsage;
    }
    const Args & a = args.value();
    auto stt = WhisperPipeline::create({a.model + "/encoder-fp16.mlpackage", a.model + "/decoder.gguf"}, a.options);
    if (!stt.is_ok()) { std::fprintf(stderr, "error: %s\n", stt.status().to_string().c_str()); return kExitFailed; }
    for (const std::string & path : a.audio) {
        WhisperTimings tm;
        auto r = stt.value().transcribe_file(path, &tm);
        if (!r.is_ok()) { std::fprintf(stderr, "error: %s: %s\n", path.c_str(), r.status().to_string().c_str()); return kExitFailed; }
        const std::string & t = r.value().text;
        std::printf("%s\n", !t.empty() && t[0] == ' ' ? t.c_str() + 1 : t.c_str());   // Whisper's text starts with the space of its first word piece
        std::fflush(stdout);
        if (a.stats) {
            std::fprintf(stderr, "%s: %.1f s of audio in %.0f ms (RTF %.4f): features %.1f ms, encoder %.1f ms, decoder %.1f ms, %lld windows, %lld tokens\n", path.c_str(),
                         tm.audio_s, tm.total_ms, tm.rtf(), tm.features_ms, tm.encoder_ms, tm.decode_ms, static_cast<long long>(tm.windows), static_cast<long long>(tm.tokens));
        }
    }
    return kExitOk;
}

}  // namespace dooms
