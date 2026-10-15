// Latency of the SmolVLA pipeline (3 cameras + prompt + state -> 50-step action chunk), split into its stages.
//
//   bench_vla --models DIR [--warmup N] [--iters N] [--units cpu_ne|cpu_gpu|all|cpu]
// DIR holds vision-b1-fp16.mlpackage, prefix-fp16.mlpackage, expert-fp16.mlpackage and host.gguf (tools/convert/convert_smolvla_*.py). The observation is synthetic
// (the timing does not depend on the values). Reports p50 / p99 of: the camera embedding (3 SigLIP towers), host work (embeddings, masks), the VLM prefix, the
// flow-matching steps, and the whole call. Exit codes: 0 ok, 1 failed, 2 usage error.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/stats.h"
#include "pipeline/vla_pipeline.h"

using namespace dooms;

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

Result<long> parse_int(const std::string & flag, const char * text, long lo, long hi) {
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return invalid_argument(flag + ": '" + text + "' is not an integer");
    if (v < lo || v > hi) return invalid_argument(flag + ": " + std::string(text) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return v;
}

void report(const char * name, const std::vector<double> & v) { std::printf("%-22s p50 %8.2f  p99 %8.2f ms\n", name, median(v), quantile(v, 0.99)); }

}  // namespace

int main(int argc, char ** argv) {
    std::string models;
    VlaOptions options;
    int warmup = 3, iters = 20;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n", flag.c_str()); return kExitUsage; }
        if (flag == "--models") { models = argv[i + 1]; continue; }
        if (flag == "--units") { options.vision_units = options.prefix_units = options.expert_units = argv[i + 1]; continue; }
        if (flag != "--warmup" && flag != "--iters") { std::fprintf(stderr, "error: unknown option '%s'\n", flag.c_str()); return kExitUsage; }
        auto v = parse_int(flag, argv[i + 1], flag == "--iters" ? 1 : 0, 100000);
        if (!v.is_ok()) { std::fprintf(stderr, "error: %s\n", v.status().to_string().c_str()); return kExitUsage; }
        (flag == "--iters" ? iters : warmup) = static_cast<int>(v.value());
    }
    if (models.empty()) { std::fprintf(stderr, "error: --models is required\n"); return kExitUsage; }
    auto pipe = VlaPipeline::create({models + "/vision-b1-fp16.mlpackage", models + "/prefix-fp16.mlpackage", models + "/expert-fp16.mlpackage", models + "/host.gguf"}, options);
    if (!pipe.is_ok()) { std::fprintf(stderr, "error: %s\n", pipe.status().to_string().c_str()); return kExitFailed; }
    VlaPipeline & v = pipe.value();
    VlaObservation obs;
    CameraImage cam;
    cam.width = cam.height = 256;
    cam.chw.assign(3 * 256 * 256, 0.5f);
    for (size_t i = 0; i < cam.chw.size(); i++) cam.chw[i] = static_cast<float>(i % 251) / 251.f;
    obs.cameras.assign(static_cast<size_t>(v.n_cameras()), cam);
    obs.tokens.assign(static_cast<size_t>(v.tokenizer_max_length()), 7);
    obs.token_mask.assign(obs.tokens.size(), 1);
    obs.state.assign(6, 0.1f);
    std::vector<double> vision, host, prefix, expert, total;
    for (int i = 0; i < warmup + iters; i++) {
        VlaTimings tm;
        auto r = v.infer(obs, nullptr, &tm);
        if (!r.is_ok()) { std::fprintf(stderr, "error: %s\n", r.status().to_string().c_str()); return kExitFailed; }
        if (i < warmup) continue;
        vision.push_back(tm.vision_ms); host.push_back(tm.host_ms); prefix.push_back(tm.prefix_ms); expert.push_back(tm.expert_ms); total.push_back(tm.total_ms);
    }
    std::printf("SmolVLA, %lld cameras, %lld prefix tokens, %lld flow steps, %d warmup + %d timed runs\n", static_cast<long long>(v.n_cameras()), static_cast<long long>(v.prefix_length()),
                static_cast<long long>(v.steps()), warmup, iters);
    report("camera embeddings", vision);
    report("host (masks, tables)", host);
    report("VLM prefix", prefix);
    report("expert (all steps)", expert);
    report("total", total);
    return kExitOk;
}
