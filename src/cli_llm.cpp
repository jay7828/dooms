#include "cli_llm.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pipeline/llm_pipeline.h"

namespace dooms {

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;
constexpr char kUsage[] =
    "usage: dooms-cli llm --model FILE.gguf --prompt TEXT [--system TEXT] [--raw] [--max-tokens N] [--ctx N] [--temp T] [--top-k K] [--top-p P] [--repeat-penalty R]\n"
    "                     [--seed N] [--stop STRING]... [--threads N] [--stats]\n"
    "  --prompt        the user message; it is wrapped in the model's chat template unless --raw (then it is continued as plain text)\n"
    "  --system        a system message (chat mode)\n"
    "  --max-tokens N  answer length limit (default 256)     --ctx N  context positions (default 4096)\n"
    "  --temp T        0 = greedy (default); above 0 samples with --top-k (40), --top-p (0.95), --repeat-penalty (1) and --seed (0)\n"
    "  --stop STRING   end the answer before this string (repeatable)     --stats  print the timings to stderr\n";

struct Args {
    std::string model, prompt, system;
    bool raw = false, stats = false;
    LlmOptions options;
    LlmGenerateOptions gen;
};

Result<double> parse_num(const std::string & flag, const char * text, double lo, double hi) {
    char * end = nullptr;
    const double v = std::strtod(text, &end);
    if (end == text || *end != '\0') return invalid_argument(flag + ": '" + text + "' is not a number");
    if (!(v >= lo && v <= hi)) return invalid_argument(flag + ": " + std::string(text) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return v;
}

Result<Args> parse(int argc, char ** argv) {
    Args a;
    a.gen.max_new_tokens = 256;
    for (int i = 1; i < argc; i++) {
        const std::string flag = argv[i];
        if (flag == "--raw") { a.raw = true; continue; }
        if (flag == "--stats") { a.stats = true; continue; }
        if (i + 1 >= argc) return invalid_argument("missing value for " + flag);
        const char * value = argv[++i];
        if (flag == "--model") a.model = value;
        else if (flag == "--prompt") a.prompt = value;
        else if (flag == "--system") a.system = value;
        else if (flag == "--stop") a.gen.stop.emplace_back(value);
        else {
            double lo = 0, hi = 0;
            if (flag == "--max-tokens") { lo = 1; hi = 1 << 20; }
            else if (flag == "--ctx") { lo = 16; hi = 1 << 20; }
            else if (flag == "--threads") { lo = 0; hi = 1024; }
            else if (flag == "--temp") { lo = 0; hi = 100; }
            else if (flag == "--top-k") { lo = 0; hi = 1 << 20; }
            else if (flag == "--top-p") { lo = 1e-6; hi = 1; }
            else if (flag == "--repeat-penalty") { lo = 1e-6; hi = 100; }
            else if (flag == "--seed") { lo = 0; hi = 4294967295.0; }
            else return invalid_argument("unknown option '" + flag + "'");
            auto v = parse_num(flag, value, lo, hi);
            if (!v.is_ok()) return v.status();
            const double x = v.value();
            if (flag == "--max-tokens") a.gen.max_new_tokens = static_cast<int32_t>(x);
            else if (flag == "--ctx") a.options.text.n_ctx = static_cast<int32_t>(x);
            else if (flag == "--threads") a.options.text.n_threads = static_cast<int32_t>(x);
            else if (flag == "--temp") a.gen.sampler.temperature = static_cast<float>(x);
            else if (flag == "--top-k") a.gen.sampler.top_k = static_cast<int32_t>(x);
            else if (flag == "--top-p") a.gen.sampler.top_p = static_cast<float>(x);
            else if (flag == "--repeat-penalty") a.gen.sampler.repeat_penalty = static_cast<float>(x);
            else if (flag == "--seed") a.gen.sampler.seed = static_cast<uint32_t>(x);
        }
    }
    if (a.model.empty() || a.prompt.empty()) return invalid_argument("--model and --prompt are required");
    if (a.options.text.n_batch > a.options.text.n_ctx) a.options.text.n_batch = a.options.text.n_ctx;
    return a;
}

}  // namespace

int run_llm_command(int argc, char ** argv) {
    auto args = parse(argc, argv);
    if (!args.is_ok()) {
        std::fprintf(stderr, "error: %s\n%s", args.status().to_string().c_str(), kUsage);
        return kExitUsage;
    }
    const Args & a = args.value();
    auto llm = Llm::load(a.model, a.options);
    if (!llm.is_ok()) { std::fprintf(stderr, "error: %s\n", llm.status().to_string().c_str()); return kExitFailed; }
    auto on_text = [](const std::string & s) { std::fputs(s.c_str(), stdout); std::fflush(stdout); return true; };
    LlmTimings tm;
    Result<LlmGeneration> out = a.raw ? llm.value().generate(a.prompt, a.gen, on_text, &tm) : [&] {
        std::vector<ChatMessage> msgs;
        if (!a.system.empty()) msgs.push_back({"system", a.system});
        msgs.push_back({"user", a.prompt});
        return llm.value().chat(msgs, a.gen, on_text, &tm);
    }();
    std::printf("\n");
    if (!out.is_ok()) { std::fprintf(stderr, "error: %s\n", out.status().to_string().c_str()); return kExitFailed; }
    if (!out.value().stopped) std::fprintf(stderr, "note: stopped at --max-tokens %d before the model finished\n", a.gen.max_new_tokens);
    if (a.stats) {
        std::fprintf(stderr, "prefill %.1f ms (%lld prompt tokens, %.0f tokens/s), first token %.1f ms, %lld tokens, %.2f ms/token (%.1f tokens/s)\n", tm.prefill_ms,
                     static_cast<long long>(tm.prompt_tokens), tm.prefill_ms > 0 ? 1000.0 * static_cast<double>(tm.prompt_tokens) / tm.prefill_ms : 0.0, tm.first_token_ms,
                     static_cast<long long>(tm.generated_tokens), tm.decode_ms_per_token(), tm.decode_ms_per_token() > 0 ? 1000.0 / tm.decode_ms_per_token() : 0.0);
    }
    return kExitOk;
}

}  // namespace dooms
