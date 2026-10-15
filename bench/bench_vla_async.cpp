// A simulated robot control loop driven by SmolVLA, synchronous against asynchronous chunking (VlaChunkRunner).
//
//   bench_vla_async --models DIR [--hz 30] [--seconds 20] [--threshold 0.5] [--aggregate NAME|WEIGHT] [--mode both|sync|async] [--warmup 2] [--units cpu_ne|...]
//   bench_vla_async --fake-latency-ms MS [--fake-rows 50] [...]                              (no model: the inference sleeps MS and returns zeros)
// The loop ticks at --hz on the steady clock for --seconds after the first action. Each tick, the synchronous loop executes the next queued action or, with
// the queue empty, captures the observation and runs one inference on the control thread (the robot stands still meanwhile, as lerobot's synchronous
// record loop does); the asynchronous loop submits an observation when the runner wants one and takes next_action(now). The observation is fixed: lerobot's
// reference inputs from DIR/reference.gguf when present (tools/parity/smolvla_reference.py), synthetic otherwise; timing does not depend on the values.
// Reports per mode: effective control rate (actions / second), stalled control steps (ticks without an action after the first), the longest gap between two
// actions, inference latency p50 / p99 and the runner's splice counts. Exit codes: 0 ok, 1 failed, 2 usage error.
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/stats.h"
#include "format/gguf.h"
#include "pipeline/vla_chunk_runner.h"
#include "pipeline/vla_pipeline.h"

using namespace dooms;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;
// Defaults: lerobot's async inference (30 Hz, threshold 0.5, weighted_average) and a run long enough for ~10 chunks at SmolVLA's latency.
constexpr double kDefaultSeconds = 20;
constexpr int kDefaultWarmup = 2;
constexpr int64_t kDefaultFakeRows = 50;  // SmolVLA's chunk_size
constexpr int64_t kFakeCols = 32;         // SmolVLA's padded action width
constexpr double kMaxSeconds = 3600, kMaxLatencyMs = 60000;
constexpr int kMaxWarmup = 1000;
constexpr int64_t kMaxFakeRows = 4096;
// Synthetic observation when there is no reference file: the size bench_vla uses.
constexpr int64_t kSyntheticSide = 256, kSyntheticPattern = 251;
constexpr float kSyntheticState = 0.1f;
constexpr int32_t kSyntheticToken = 7;
constexpr int64_t kSyntheticStateDim = 6;
// How long to wait for the first chunk (the first Core ML call may compile the models).
constexpr double kFirstChunkWaitS = 600;

Result<double> parse_double(const std::string & flag, const char * text, double lo, double hi) {
    char * end = nullptr;
    const double v = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(v)) return invalid_argument(flag + ": '" + text + "' is not a number");
    if (v < lo || v > hi) return invalid_argument(flag + ": " + std::string(text) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return v;
}

std::string machine() {
    char buf[256] = {0};
    size_t n = sizeof buf - 1;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) != 0) return "unknown";
    return buf;
}

std::vector<float> tensor(const GgufFile & f, const std::string & name, Shape * shape = nullptr) {
    auto info = f.find_tensor(name);
    if (!info.is_ok()) return {};
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    if (shape) *shape = info.value()->shape;
    return std::vector<float>(p, p + info.value()->shape.numel());
}

Result<VlaObservation> reference_observation(const std::string & path) {
    auto ref = GgufFile::open(path);
    if (!ref.is_ok()) return ref.status();
    VlaObservation obs;
    for (const char * name : {"image_camera1", "image_camera2", "image_camera3"}) {
        Shape sh;
        CameraImage c;
        c.chw = tensor(ref.value(), name, &sh);
        if (c.chw.empty()) return not_found(path + " has no tensor " + name);
        c.height = sh.dim(1);
        c.width = sh.dim(2);
        obs.cameras.push_back(std::move(c));
    }
    for (float t : tensor(ref.value(), "tokens")) obs.tokens.push_back(static_cast<int32_t>(t));
    for (float t : tensor(ref.value(), "token_mask")) obs.token_mask.push_back(t != 0.f ? 1 : 0);
    obs.state = tensor(ref.value(), "state");
    return obs;
}

VlaObservation synthetic_observation(int64_t n_cameras, int64_t n_tokens) {
    VlaObservation obs;
    CameraImage cam;
    cam.width = cam.height = kSyntheticSide;
    cam.chw.resize(static_cast<size_t>(3 * kSyntheticSide * kSyntheticSide));
    for (size_t i = 0; i < cam.chw.size(); i++) cam.chw[i] = static_cast<float>(static_cast<int64_t>(i) % kSyntheticPattern) / static_cast<float>(kSyntheticPattern);
    obs.cameras.assign(static_cast<size_t>(n_cameras), cam);
    obs.tokens.assign(static_cast<size_t>(n_tokens), kSyntheticToken);
    obs.token_mask.assign(obs.tokens.size(), 1);
    obs.state.assign(static_cast<size_t>(kSyntheticStateDim), kSyntheticState);
    return obs;
}

// Wraps an inference function to record each call's latency (thread-safe: the asynchronous mode calls it from the worker).
struct Timed {
    VlaInferFn inner;
    std::mutex mu;
    std::vector<double> ms;
    Result<VlaChunk> operator()(const VlaObservation & o) {
        const auto t0 = Clock::now();
        auto r = inner(o);
        const double d = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::lock_guard<std::mutex> lock(mu);
        ms.push_back(d);
        return r;
    }
    std::vector<double> take() {
        std::lock_guard<std::mutex> lock(mu);
        std::vector<double> v;
        v.swap(ms);
        return v;
    }
};

struct LoopStats {
    int64_t ticks = 0, actions = 0, stalled = 0;
    double seconds = 0, longest_gap_ms = 0;
    std::vector<double> latency_ms;
    VlaChunkRunnerStats runner;  // asynchronous mode only
};

void note_action(Clock::time_point t, Clock::time_point * last, LoopStats * s) {
    if (s->actions > 0) s->longest_gap_ms = std::max(s->longest_gap_ms, std::chrono::duration<double, std::milli>(t - *last).count());
    *last = t;
    s->actions++;
}

Result<LoopStats> run_sync(Timed & timed, const VlaObservation & obs, double hz, double seconds) {
    LoopStats s;
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / hz));
    auto r = timed(obs);  // the robot waits for its first chunk; the window starts at the first action
    if (!r.is_ok()) return r.status();
    VlaChunk chunk = std::move(r).value();
    timed.take();
    int64_t next_row = 0;
    const auto start = Clock::now();
    const auto end = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto tick = start, last_action = start;
    while (tick < end) {
        std::this_thread::sleep_until(tick);
        s.ticks++;
        if (next_row < chunk.rows) {
            note_action(Clock::now(), &last_action, &s);
            next_row++;
            tick += period;
            continue;
        }
        // queue empty: observe and infer on the control thread; every tick that passes meanwhile is a stalled step
        auto c = timed(obs);
        if (!c.is_ok()) return c.status();
        chunk = std::move(c).value();
        next_row = 0;
        const auto now = Clock::now();
        int64_t missed = 1;
        while (tick + missed * period <= now) missed++;
        s.stalled += missed;
        s.ticks += missed - 1;
        tick += missed * period;
    }
    s.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    s.latency_ms = timed.take();
    return s;
}

Result<LoopStats> run_async(Timed & timed, const VlaObservation & obs, const VlaChunkRunnerOptions & options, double seconds) {
    LoopStats s;
    auto created = VlaChunkRunner::create([&timed](const VlaObservation & o) { return timed(o); }, options);
    if (!created.is_ok()) return created.status();
    VlaChunkRunner & runner = *created.value();
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / options.control_hz));
    const auto origin = Clock::now();
    auto secs = [&](Clock::time_point t) { return std::chrono::duration<double>(t - origin).count(); };
    auto st = runner.submit_observation(obs, secs(Clock::now()));
    if (!st.is_ok()) return st.status();
    if (!runner.wait_idle(kFirstChunkWaitS)) return internal_error("the first chunk did not arrive within " + std::to_string(kFirstChunkWaitS) + " s");
    if (runner.stats().chunks_received == 0) return runner.stats().last_error;
    timed.take();
    VlaAction a;
    const auto start = Clock::now();
    const auto end = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto tick = start, last_action = start;
    while (tick < end) {
        std::this_thread::sleep_until(tick);
        s.ticks++;
        const auto now = Clock::now();
        if (runner.wants_observation()) {
            auto sub = runner.submit_observation(obs, secs(now));
            if (!sub.is_ok()) return sub.status();
        }
        DOOMS_RETURN_IF_ERROR(runner.next_action(secs(now), &a));
        if (a.available) note_action(Clock::now(), &last_action, &s);
        tick += period;
    }
    s.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    runner.stop();
    s.runner = runner.stats();
    s.stalled = s.runner.stalled_calls;
    s.latency_ms = timed.take();
    if (s.runner.inferences_failed > 0) return Status(s.runner.last_error.code(), "an inference failed during the run: " + s.runner.last_error.message());
    return s;
}

void report(const char * mode, const LoopStats & s, double hz) {
    std::printf("%-6s %8.1f Hz of %.0f  %7lld actions  %6lld stalled steps  longest gap %8.1f ms  latency p50 %7.1f p99 %7.1f ms (%zu inferences)", mode,
                static_cast<double>(s.actions) / s.seconds, hz, static_cast<long long>(s.actions), static_cast<long long>(s.stalled), s.longest_gap_ms,
                median(s.latency_ms), quantile(s.latency_ms, 0.99), s.latency_ms.size());
    if (s.runner.chunks_received > 0)
        std::printf("  [stale %lld, blended %lld, late-skipped %lld]", static_cast<long long>(s.runner.stale_dropped), static_cast<long long>(s.runner.blended),
                    static_cast<long long>(s.runner.late_skipped));
    std::printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    std::string models, mode = "both";
    VlaOptions vla_options;
    VlaChunkRunnerOptions runner_options;
    double seconds = kDefaultSeconds, fake_latency_ms = -1;
    int warmup = kDefaultWarmup;
    int64_t fake_rows = kDefaultFakeRows;
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n", flag.c_str()); return kExitUsage; }
        const char * value = argv[i + 1];
        Result<double> v = 0.0;
        if (flag == "--models") { models = value; continue; }
        if (flag == "--units") { vla_options.vision_units = vla_options.prefix_units = vla_options.expert_units = value; continue; }
        if (flag == "--mode") {
            mode = value;
            if (mode != "both" && mode != "sync" && mode != "async") { std::fprintf(stderr, "error: --mode: expected both, sync or async\n"); return kExitUsage; }
            continue;
        }
        if (flag == "--aggregate") {
            auto named = vla_aggregate_weight(value);
            v = named.is_ok() ? named : parse_double(flag, value, 0, 1);
            if (!v.is_ok()) { std::fprintf(stderr, "error: --aggregate: expected a lerobot name or a weight in [0, 1] (%s)\n", named.status().message().c_str()); return kExitUsage; }
            runner_options.new_weight = v.value();
            continue;
        }
        if (flag == "--hz") v = parse_double(flag, value, 1, kVlaRunnerMaxControlHz);
        else if (flag == "--seconds") v = parse_double(flag, value, 0.1, kMaxSeconds);
        else if (flag == "--threshold") v = parse_double(flag, value, 0, 1);
        else if (flag == "--fake-latency-ms") v = parse_double(flag, value, 0, kMaxLatencyMs);
        else if (flag == "--warmup") v = parse_double(flag, value, 0, kMaxWarmup);
        else if (flag == "--fake-rows") v = parse_double(flag, value, 1, static_cast<double>(kMaxFakeRows));
        else { std::fprintf(stderr, "error: unknown option '%s'\n", flag.c_str()); return kExitUsage; }
        if (!v.is_ok()) { std::fprintf(stderr, "error: %s\n", v.status().message().c_str()); return kExitUsage; }
        if (flag == "--hz") runner_options.control_hz = v.value();
        else if (flag == "--seconds") seconds = v.value();
        else if (flag == "--threshold") runner_options.chunk_size_threshold = v.value();
        else if (flag == "--fake-latency-ms") fake_latency_ms = v.value();
        else if (flag == "--warmup") warmup = static_cast<int>(v.value());
        else fake_rows = static_cast<int64_t>(v.value());
    }
    if (models.empty() == (fake_latency_ms < 0)) { std::fprintf(stderr, "error: exactly one of --models DIR and --fake-latency-ms MS is required\n"); return kExitUsage; }

    std::unique_ptr<VlaPipeline> pipeline;
    VlaObservation obs;
    Timed timed;
    std::string source;
    if (!models.empty()) {
        auto p = VlaPipeline::create({models + "/vision-b1-fp16.mlpackage", models + "/prefix-fp16.mlpackage", models + "/expert-fp16.mlpackage", models + "/host.gguf"}, vla_options);
        if (!p.is_ok()) { std::fprintf(stderr, "error: %s\n", p.status().to_string().c_str()); return kExitFailed; }
        pipeline = std::make_unique<VlaPipeline>(std::move(p).value());
        auto ref = reference_observation(models + "/reference.gguf");
        obs = ref.is_ok() ? std::move(ref).value() : synthetic_observation(pipeline->n_cameras(), pipeline->tokenizer_max_length());
        source = ref.is_ok() ? "SmolVLA (" + models + "), lerobot's reference observation" : "SmolVLA (" + models + "), synthetic observation";
        timed.inner = vla_pipeline_infer_fn(pipeline.get());
        for (int i = 0; i < warmup; i++) {
            auto r = timed.inner(obs);
            if (!r.is_ok()) { std::fprintf(stderr, "error: %s\n", r.status().to_string().c_str()); return kExitFailed; }
        }
    } else {
        char text[128];
        std::snprintf(text, sizeof text, "fake model: %.1f ms per chunk of %lld actions", fake_latency_ms, static_cast<long long>(fake_rows));
        source = text;
        const auto latency = std::chrono::duration<double, std::milli>(fake_latency_ms);
        timed.inner = [latency, fake_rows](const VlaObservation &) -> Result<VlaChunk> {
            std::this_thread::sleep_for(latency);
            VlaChunk c;
            c.rows = fake_rows;
            c.cols = kFakeCols;
            c.actions.assign(static_cast<size_t>(fake_rows * kFakeCols), 0.f);
            return c;
        };
        obs = synthetic_observation(1, 1);
    }
    std::printf("%s\nmachine %s; %.0f Hz control loop for %.1f s after the first action; threshold %.2f, new_weight %.2f; %d warmup inferences\n", source.c_str(),
                machine().c_str(), runner_options.control_hz, seconds, runner_options.chunk_size_threshold, runner_options.new_weight, pipeline ? warmup : 0);
    if (mode != "async") {
        auto s = run_sync(timed, obs, runner_options.control_hz, seconds);
        if (!s.is_ok()) { std::fprintf(stderr, "error: sync: %s\n", s.status().to_string().c_str()); return kExitFailed; }
        report("sync", s.value(), runner_options.control_hz);
    }
    if (mode != "sync") {
        auto s = run_async(timed, obs, runner_options, seconds);
        if (!s.is_ok()) { std::fprintf(stderr, "error: async: %s\n", s.status().to_string().c_str()); return kExitFailed; }
        report("async", s.value(), runner_options.control_hz);
    }
    return kExitOk;
}
