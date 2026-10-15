#include "cli_vlm.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "pipeline/image_io.h"
#include "pipeline/vlm_pipeline.h"

namespace dooms {

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;
constexpr char kUsage[] =
    "usage: dooms-cli vlm --image FILE --prompt TEXT (--models DIR | --vision PKG --text GGUF --config GGUF) [--max-tokens N] [--vision-units all|cpu_gpu|cpu_ne|cpu] [--threads N] [--stats]\n"
    "  --models DIR      directory from tools/convert/convert_smolvlm.py (vision-fp16.mlpackage, text-f16.gguf, vlm-config.gguf)\n"
    "  --max-tokens N    answer length limit, 1..4096 (default 256)\n"
    "  --stats           print the stage timings to stderr\n";

struct Args {
    std::string image, prompt, models;
    VlmPaths paths;
    VlmOptions options;
    int max_tokens = 256;
    bool stats = false;
};

Result<long> parse_int(const std::string & flag, const char * text, long lo, long hi) {
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return invalid_argument(flag + ": '" + text + "' is not an integer");
    if (v < lo || v > hi) return invalid_argument(flag + ": " + std::string(text) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return v;
}

Result<Args> parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        const std::string flag = argv[i];
        if (flag == "--stats") { a.stats = true; continue; }
        if (i + 1 >= argc) return invalid_argument("missing value for " + flag);
        const char * value = argv[++i];
        if (flag == "--image") a.image = value;
        else if (flag == "--prompt") a.prompt = value;
        else if (flag == "--models") a.models = value;
        else if (flag == "--vision") a.paths.vision = value;
        else if (flag == "--text") a.paths.text = value;
        else if (flag == "--config") a.paths.config = value;
        else if (flag == "--vision-units") a.options.vision_units = value;
        else if (flag == "--max-tokens" || flag == "--threads") {
            const bool tokens = flag == "--max-tokens";
            auto v = parse_int(flag, value, tokens ? 1 : 0, tokens ? 4096 : 1024);
            if (!v.is_ok()) return v.status();
            if (tokens) a.max_tokens = static_cast<int>(v.value()); else a.options.text.n_threads = static_cast<int>(v.value());
        } else {
            return invalid_argument("unknown option '" + flag + "'");
        }
    }
    if (a.image.empty() || a.prompt.empty()) return invalid_argument("--image and --prompt are required");
    if (!a.models.empty()) {
        if (a.paths.vision.empty()) a.paths.vision = a.models + "/vision-fp16.mlpackage";
        if (a.paths.text.empty()) a.paths.text = a.models + "/text-f16.gguf";
        if (a.paths.config.empty()) a.paths.config = a.models + "/vlm-config.gguf";
    }
    if (a.paths.vision.empty() || a.paths.text.empty() || a.paths.config.empty()) return invalid_argument("give --models DIR, or all of --vision, --text and --config");
    return a;
}

}  // namespace

int run_vlm_command(int argc, char ** argv) {
    auto args = parse(argc, argv);
    if (!args.is_ok()) {
        std::fprintf(stderr, "error: %s\n%s", args.status().to_string().c_str(), kUsage);
        return kExitUsage;
    }
    const Args & a = args.value();
    auto image = load_image(a.image);
    if (!image.is_ok()) { std::fprintf(stderr, "error: %s\n", image.status().to_string().c_str()); return kExitFailed; }
    auto pipe = VlmPipeline::create(a.paths, a.options);
    if (!pipe.is_ok()) { std::fprintf(stderr, "error: %s\n", pipe.status().to_string().c_str()); return kExitFailed; }
    VlmTimings tm;
    auto out = pipe.value().generate(image.value(), a.prompt, a.max_tokens, [](const std::string & s) { std::fputs(s.c_str(), stdout); std::fflush(stdout); return true; }, &tm);
    std::printf("\n");
    if (!out.is_ok()) { std::fprintf(stderr, "error: %s\n", out.status().to_string().c_str()); return kExitFailed; }
    if (!out.value().stopped) std::fprintf(stderr, "note: stopped at --max-tokens %d before the model finished\n", a.max_tokens);
    if (a.stats) {
        std::fprintf(stderr, "preprocess %.1f ms, vision %.1f ms, prefill %.1f ms (%lld prompt tokens), first token %.1f ms, %lld tokens, %.2f ms/token\n", tm.preprocess_ms, tm.vision_ms, tm.prefill_ms,
                     static_cast<long long>(tm.prompt_tokens), tm.first_token_ms, static_cast<long long>(tm.generated_tokens), tm.decode_ms_per_token());
    }
    return kExitOk;
}

}  // namespace dooms
