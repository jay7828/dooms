// Throughput of a text language model on the engine, measured like llama-bench: the prompt is processed in one batch (prefill, tokens/s) and then N tokens are
// generated one by one with greedy sampling that ignores end-of-generation, so the decode count is fixed (ms/token and tokens/s).
//
//   bench_llm --model FILE.gguf --prompt-ids FILE [--gen N] [--ctx N] [--warmup N] [--iters N] [--threads N]
// --prompt-ids: whitespace-separated token ids (the same file feeds the PyTorch and MLX baselines, tools: bench/python/llm_baselines.py).
// Reports p50 / p99 over --iters complete (prefill + decode) runs. Exit codes: 0 ok, 1 failed, 2 usage error.
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/stats.h"
#include "models/vlm/text_decoder.h"

using namespace dooms;

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

struct Options {
    std::string model, prompt_ids;
    int gen = 128, ctx = 0, warmup = 2, iters = 8, threads = 0;
};

Result<Options> parse_args(int argc, char ** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        const char * value = argv[i + 1];
        auto num = [&](int lo, int hi) -> Result<int> {
            char * end = nullptr;
            const long v = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || v < lo || v > hi) return invalid_argument(flag + ": expected an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
            return static_cast<int>(v);
        };
        if (flag == "--model") o.model = value;
        else if (flag == "--prompt-ids") o.prompt_ids = value;
        else if (flag == "--gen" || flag == "--ctx" || flag == "--warmup" || flag == "--iters" || flag == "--threads") {
            auto v = num(flag == "--warmup" || flag == "--threads" || flag == "--ctx" ? 0 : 1, 1 << 20);
            if (!v.is_ok()) return v.status();
            (flag == "--gen" ? o.gen : flag == "--ctx" ? o.ctx : flag == "--warmup" ? o.warmup : flag == "--iters" ? o.iters : o.threads) = v.value();
        } else {
            return invalid_argument("unknown option '" + flag + "'");
        }
    }
    if ((argc - 1) % 2 != 0) return invalid_argument("every option needs a value");
    if (o.model.empty() || o.prompt_ids.empty()) return invalid_argument("--model and --prompt-ids are required");
    return o;
}

double ms_since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }

}  // namespace

int main(int argc, char ** argv) {
    auto args = parse_args(argc, argv);
    if (!args.is_ok()) {
        std::fprintf(stderr, "error: %s\nusage: bench_llm --model FILE.gguf --prompt-ids FILE [--gen N] [--ctx N] [--warmup N] [--iters N] [--threads N]\n", args.status().to_string().c_str());
        return kExitUsage;
    }
    const Options & o = args.value();
    std::vector<int32_t> prompt;
    {
        std::ifstream in(o.prompt_ids);
        if (!in) { std::fprintf(stderr, "error: cannot open %s\n", o.prompt_ids.c_str()); return kExitFailed; }
        for (int32_t t; in >> t;) prompt.push_back(t);
    }
    if (prompt.empty()) { std::fprintf(stderr, "error: no token ids in %s\n", o.prompt_ids.c_str()); return kExitFailed; }
    TextDecoderOptions to;
    to.n_ctx = o.ctx > 0 ? o.ctx : static_cast<int32_t>(prompt.size()) + o.gen + 16;
    if (to.n_ctx < 16) to.n_ctx = 16;
    to.n_batch = static_cast<int32_t>(std::min<size_t>(prompt.size(), 2048));
    if (to.n_batch > to.n_ctx) to.n_batch = to.n_ctx;
    to.n_threads = o.threads;
    auto dec = TextDecoder::load(o.model, to);
    if (!dec.is_ok()) { std::fprintf(stderr, "error: %s\n", dec.status().to_string().c_str()); return kExitFailed; }
    TextDecoder & d = dec.value();

    std::vector<double> prefill_ms, decode_ms;
    for (int it = 0; it < o.warmup + o.iters; it++) {
        d.reset();
        auto t0 = std::chrono::steady_clock::now();
        if (Status st = d.eval_tokens(prompt); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
        auto next = d.greedy_next();
        if (!next.is_ok()) { std::fprintf(stderr, "error: %s\n", next.status().to_string().c_str()); return kExitFailed; }
        const double pre = ms_since(t0);
        int32_t tok = next.value();
        auto t1 = std::chrono::steady_clock::now();
        double eval_total = 0, pick_total = 0;
        for (int i = 0; i < o.gen; i++) {
            auto a = std::chrono::steady_clock::now();
            if (Status st = d.eval_tokens({tok}); !st.is_ok()) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; }
            auto b = std::chrono::steady_clock::now();
            auto n = d.greedy_next();  // end-of-generation is ignored on purpose: the decode length is fixed
            if (!n.is_ok()) { std::fprintf(stderr, "error: %s\n", n.status().to_string().c_str()); return kExitFailed; }
            tok = n.value();
            eval_total += std::chrono::duration<double, std::milli>(b - a).count();
            pick_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b).count();
        }
        if (it == o.warmup + o.iters - 1) std::printf("last run: eval_tokens %.3f ms/token, greedy_next %.3f ms/token\n", eval_total / o.gen, pick_total / o.gen);
        const double dec_ms = ms_since(t1);
        if (it >= o.warmup) { prefill_ms.push_back(pre); decode_ms.push_back(dec_ms / o.gen); }
    }
    const double p50 = median(prefill_ms), p99 = quantile(prefill_ms, 0.99), g50 = median(decode_ms), g99 = quantile(decode_ms, 0.99);
    std::printf("LLM %s: prompt %zu tokens, %d generated, %d warmup + %d timed runs\n", o.model.c_str(), prompt.size(), o.gen, o.warmup, o.iters);
    std::printf("prefill  p50 %9.2f ms  p99 %9.2f ms   %8.1f tokens/s\n", p50, p99, 1000.0 * static_cast<double>(prompt.size()) / p50);
    std::printf("decode   p50 %9.3f ms/token  p99 %9.3f   %8.1f tokens/s\n", g50, g99, 1000.0 / g50);
    return kExitOk;
}
