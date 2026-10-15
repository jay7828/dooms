#include "cli_vla.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "models/vlm/tokenizer.h"
#include "pipeline/image_io.h"
#include "pipeline/vla_pipeline.h"

namespace dooms {

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;
constexpr char kUsage[] =
    "usage: dooms-cli vla --models DIR --prompt TEXT --camera FILE [--camera FILE ...] --state v1,v2,... [options]\n"
    "  --models DIR        directory with vision-b1-fp16.mlpackage, prefix-fp16.mlpackage, expert-fp16.mlpackage, host.gguf, vocab.gguf (tools/convert/convert_smolvla_*.py)\n"
    "  --camera FILE       one per camera, in the policy's order (the model needs exactly as many as it was exported with)\n"
    "  --state LIST        the robot state, comma-separated (raw units when --state-mean/--state-std are given, else already normalized)\n"
    "  --state-mean LIST --state-std LIST --action-mean LIST --action-std LIST   the training dataset's statistics (all four or none); the actions are printed in robot units\n"
    "  --robot-dims N      action columns to print when no statistics are given (default: the state length)\n"
    "  --units U           all|cpu_gpu|cpu_ne|cpu for the three Core ML models (default cpu_ne)\n"
    "  --seed N            noise seed (default 0)     --stats   print the stage timings to stderr\n";

struct Args {
    std::string models, prompt;
    std::vector<std::string> cameras;
    std::vector<float> state, state_mean, state_std, action_mean, action_std;
    bool have_state = false;
    long robot_dims = 0;
    std::string units = "cpu_ne";
    long seed = 0;
    bool stats = false;
};

Result<std::vector<float>> parse_list(const std::string & flag, const char * text) {
    std::vector<float> out;
    std::string s = text;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        const std::string part = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        char * end = nullptr;
        const float v = std::strtof(part.c_str(), &end);
        if (part.empty() || end != part.c_str() + part.size()) return invalid_argument(flag + " expects comma-separated numbers, got '" + s + "'");
        out.push_back(v);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

Result<Args> parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        const std::string flag = argv[i];
        if (flag == "--stats") { a.stats = true; continue; }
        if (i + 1 >= argc) return invalid_argument("missing value for " + flag);
        const char * value = argv[++i];
        std::vector<float> * list = flag == "--state" ? &a.state : flag == "--state-mean" ? &a.state_mean : flag == "--state-std" ? &a.state_std : flag == "--action-mean" ? &a.action_mean
                                    : flag == "--action-std" ? &a.action_std : nullptr;
        if (list) {
            auto v = parse_list(flag, value);
            if (!v.is_ok()) return v.status();
            *list = v.value();
            if (flag == "--state") a.have_state = true;
        } else if (flag == "--models") a.models = value;
        else if (flag == "--prompt") a.prompt = value;
        else if (flag == "--camera") a.cameras.push_back(value);
        else if (flag == "--units") a.units = value;
        else if (flag == "--seed" || flag == "--robot-dims") {
            char * end = nullptr;
            const long v = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || v < 0 || v > 1000000) return invalid_argument(flag + " expects a non-negative integer, got '" + value + "'");
            (flag == "--seed" ? a.seed : a.robot_dims) = v;
        } else return invalid_argument("unknown option '" + flag + "'");
    }
    if (a.models.empty() || a.prompt.empty() || a.cameras.empty() || !a.have_state) return invalid_argument("--models, --prompt, --camera and --state are required");
    const int stats_given = !a.state_mean.empty() + !a.state_std.empty() + !a.action_mean.empty() + !a.action_std.empty();
    if (stats_given != 0 && stats_given != 4) return invalid_argument("give all four of --state-mean, --state-std, --action-mean, --action-std, or none");
    return a;
}

}  // namespace

int run_vla_command(int argc, char ** argv) {
    auto args = parse(argc, argv);
    if (!args.is_ok()) {
        std::fprintf(stderr, "error: %s\n%s", args.status().to_string().c_str(), kUsage);
        return kExitUsage;
    }
    Args a = std::move(args).value();
    auto fail = [](const Status & st) { std::fprintf(stderr, "error: %s\n", st.to_string().c_str()); return kExitFailed; };
    VlaOptions options;
    options.vision_units = options.prefix_units = options.expert_units = a.units;
    options.seed = static_cast<uint64_t>(a.seed);
    auto pipe = VlaPipeline::create({a.models + "/vision-b1-fp16.mlpackage", a.models + "/prefix-fp16.mlpackage", a.models + "/expert-fp16.mlpackage", a.models + "/host.gguf"}, options);
    if (!pipe.is_ok()) return fail(pipe.status());
    auto tok = Tokenizer::load(a.models + "/vocab.gguf");
    if (!tok.is_ok()) return fail(tok.status());
    VlaObservation obs;
    if (static_cast<int64_t>(a.cameras.size()) != pipe.value().n_cameras()) {
        std::fprintf(stderr, "error: the model needs %lld cameras, got %zu\n", static_cast<long long>(pipe.value().n_cameras()), a.cameras.size());
        return kExitUsage;
    }
    for (const std::string & path : a.cameras) {
        auto img = load_image(path);
        if (!img.is_ok()) return fail(img.status());
        auto cam = camera_from_rgb(img.value());
        if (!cam.is_ok()) return fail(cam.status());
        obs.cameras.push_back(std::move(cam).value());
    }
    auto tokens = tok.value().encode_task(a.prompt, pipe.value().tokenizer_max_length());
    if (!tokens.is_ok()) return fail(tokens.status());
    if (tokens.value().truncated) std::fprintf(stderr, "note: the prompt was truncated to %lld tokens\n", static_cast<long long>(pipe.value().tokenizer_max_length()));
    obs.tokens = tokens.value().tokens;
    obs.token_mask = tokens.value().mask;
    const bool stats = !a.state_mean.empty();
    obs.state = a.state;
    if (stats) {
        auto n = vla_normalize_state(a.state, a.state_mean, a.state_std);
        if (!n.is_ok()) return fail(n.status());
        obs.state = n.value();
    }
    VlaTimings tm;
    auto actions = pipe.value().infer(obs, nullptr, &tm);
    if (!actions.is_ok()) return fail(actions.status());
    std::vector<float> rows = actions.value();
    int64_t cols = pipe.value().action_dim();
    if (stats) {
        auto u = vla_unnormalize_actions(rows, cols, a.action_mean, a.action_std);
        if (!u.is_ok()) return fail(u.status());
        rows = u.value();
        cols = static_cast<int64_t>(a.action_mean.size());
    } else {
        const int64_t keep = a.robot_dims > 0 ? a.robot_dims : static_cast<int64_t>(a.state.size());
        if (keep > cols) { std::fprintf(stderr, "error: --robot-dims %lld exceeds the model's action width %lld\n", static_cast<long long>(keep), static_cast<long long>(cols)); return kExitUsage; }
        std::vector<float> cropped;
        for (size_t r = 0; r < rows.size() / static_cast<size_t>(cols); r++)
            for (int64_t j = 0; j < keep; j++) cropped.push_back(rows[r * static_cast<size_t>(cols) + static_cast<size_t>(j)]);
        rows = cropped;
        cols = keep;
    }
    for (size_t r = 0; r < rows.size() / static_cast<size_t>(cols); r++) {
        for (int64_t j = 0; j < cols; j++) std::printf("%s%.5f", j ? "," : "", rows[r * static_cast<size_t>(cols) + static_cast<size_t>(j)]);
        std::printf("\n");
    }
    if (a.stats) std::fprintf(stderr, "cameras %.1f ms, prefix %.1f ms, flow steps %.1f ms, total %.1f ms\n", tm.vision_ms, tm.prefix_ms, tm.expert_ms, tm.total_ms);
    return kExitOk;
}

}  // namespace dooms
