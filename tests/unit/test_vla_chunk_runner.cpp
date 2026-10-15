// Tests VlaChunkRunner (asynchronous action chunking) with a fake model whose latency the test controls: each inference blocks on a
// gate until the test opens it, so "the chunk arrives after k control steps" is exact and no test sleeps. Expected actions are computed
// here from the chunk formula and the blending rule (double precision), never by the runner. With DOOMS_VLA_DIR (converted SmolVLA +
// reference.gguf) the runner also drives the real pipeline and its spliced action stream is checked against lerobot's chunk.
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "format/gguf.h"
#include "pipeline/vla_chunk_runner.h"

using namespace dooms;

namespace {

constexpr double kHz = 30.0;
constexpr double kT = 1.0 / kHz;
// Bounded waits for the worker thread; generous because a loaded machine (or a sanitizer build) can be slow. A correct runner never
// gets near them, a broken one fails instead of hanging.
constexpr double kWaitS = 30.0;
// float32 storage of blended values: |error| <= 2^-24 * |value|, values here are < 2^12.
constexpr double kBlendTol = 1e-3;

// Chunk value for observation tag `tag`, row i, column j (distinct for every tag, row and column).
double fake_value(int tag, int64_t i, int64_t j) { return tag * 1000.0 + static_cast<double>(i) + 0.25 * static_cast<double>(j); }

struct Fake {
    std::mutex mu;
    std::condition_variable cv;
    bool gated = false;      // when true, each call waits for `opened` to be incremented
    int opened = 0, entered = 0, finished = 0, active = 0, max_active = 0;
    int64_t rows = 10, cols = 2;
    int fail_call = -1;      // 0-based call index that returns an error
    Status fail_status = internal_error("fake model failure");
    int throw_call = -1;
    bool bad_shape = false, bad_value = false, change_cols = false;

    Result<VlaChunk> operator()(const VlaObservation & obs) {
        std::unique_lock<std::mutex> lock(mu);
        const int call = entered++;
        active++;
        max_active = std::max(max_active, active);
        cv.notify_all();
        cv.wait(lock, [&] { return !gated || opened > call; });
        active--;
        finished++;
        if (call == fail_call) return fail_status;
        if (call == throw_call) throw std::runtime_error("boom");
        const int tag = static_cast<int>(obs.state.at(0));
        VlaChunk c;
        c.rows = rows;
        c.cols = change_cols && call > 0 ? cols + 1 : cols;
        for (int64_t i = 0; i < c.rows; i++)
            for (int64_t j = 0; j < c.cols; j++) c.actions.push_back(static_cast<float>(fake_value(tag, i, j)));
        if (bad_shape) c.actions.pop_back();
        if (bad_value) c.actions[1] = NAN;
        return c;
    }
    void release_all() {  // lets every present and future call finish (an inference in flight blocks the runner's destruction)
        std::lock_guard<std::mutex> lock(mu);
        gated = false;
        cv.notify_all();
    }
    void open_gate() {
        std::lock_guard<std::mutex> lock(mu);
        opened++;
        cv.notify_all();
    }
    int read(int Fake::*field) {  // the counters are written by the worker thread
        std::lock_guard<std::mutex> lock(mu);
        return this->*field;
    }
    void wait_entered(int n) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait_for(lock, std::chrono::duration<double>(kWaitS), [&] { return entered >= n; });
    }
};

VlaObservation obs_with_tag(int tag) {
    VlaObservation o;
    o.state = {static_cast<float>(tag)};
    return o;
}

std::unique_ptr<VlaChunkRunner> make(const std::shared_ptr<Fake> & fake, double threshold, double new_weight) {
    VlaChunkRunnerOptions o;
    o.control_hz = kHz;
    o.chunk_size_threshold = threshold;
    o.new_weight = new_weight;
    auto r = VlaChunkRunner::create([fake](const VlaObservation & obs) { return (*fake)(obs); }, o);
    EXPECT(r.is_ok());
    return r.is_ok() ? std::move(r).value() : nullptr;
}

double t_of(int64_t k) { return static_cast<double>(k) * kT; }

bool row_equals(const VlaAction & a, int tag, int64_t row) {
    for (size_t j = 0; j < a.values.size(); j++)
        if (std::fabs(a.values[j] - fake_value(tag, row, static_cast<int64_t>(j))) > kBlendTol) return false;
    return !a.values.empty();
}

}  // namespace

TEST(create_validates_options) {
    auto fn = [](const VlaObservation &) -> Result<VlaChunk> { return internal_error("unused"); };
    EXPECT(!VlaChunkRunner::create(VlaInferFn()).is_ok());
    for (double hz : {0.0, -1.0, double(NAN), double(INFINITY), kVlaRunnerMaxControlHz * 2}) {
        VlaChunkRunnerOptions o;
        o.control_hz = hz;
        auto r = VlaChunkRunner::create(fn, o);
        EXPECT(!r.is_ok() && r.status().code() == Code::InvalidArgument);
    }
    for (double v : {-0.1, 1.1, double(NAN)}) {
        VlaChunkRunnerOptions o;
        o.chunk_size_threshold = v;
        EXPECT(!VlaChunkRunner::create(fn, o).is_ok());
        o = VlaChunkRunnerOptions();
        o.new_weight = v;
        EXPECT(!VlaChunkRunner::create(fn, o).is_ok());
    }
    for (double v : {0.0, 1.0}) {
        VlaChunkRunnerOptions o;
        o.chunk_size_threshold = o.new_weight = v;
        EXPECT(VlaChunkRunner::create(fn, o).is_ok());
    }
    EXPECT(VlaChunkRunner::create(fn).is_ok());
}

TEST(aggregate_names_follow_lerobot) {
    EXPECT(vla_aggregate_weight("weighted_average").value() == 0.7);
    EXPECT(vla_aggregate_weight("latest_only").value() == 1.0);
    EXPECT(vla_aggregate_weight("average").value() == 0.5);
    EXPECT(vla_aggregate_weight("conservative").value() == 0.3);
    auto bad = vla_aggregate_weight("median");
    EXPECT(!bad.is_ok() && bad.status().message().find("weighted_average") != std::string::npos);
}

TEST(nothing_before_the_first_chunk_then_row_zero) {
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    auto r = make(fake, 0.5, 1.0);
    VlaAction a;
    EXPECT(r->next_action(0, &a).is_ok() && !a.available && a.step == -1);
    EXPECT(r->wants_observation());
    EXPECT(r->submit_observation(obs_with_tag(1), 0.01).value());
    EXPECT(!r->wants_observation());
    EXPECT(!r->submit_observation(obs_with_tag(2), 0.02).value());  // one in flight: refused, not queued
    for (int k = 1; k <= 7; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && !a.available);  // the robot waits, steps do not advance
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    // the first chunk starts at its row 0 whatever the start-up took: nothing executed in between
    EXPECT(r->next_action(t_of(8), &a).is_ok() && a.available && a.step == 0 && a.skipped == 0 && row_equals(a, 1, 0));
    const VlaChunkRunnerStats s = r->stats();
    EXPECT_EQ(s.waiting_calls, int64_t{8});
    EXPECT_EQ(s.stalled_calls, int64_t{0});
    EXPECT_EQ(s.stale_dropped, int64_t{0});
    EXPECT_EQ(s.inferences_started, int64_t{1});
    EXPECT_EQ(fake->read(&Fake::entered), 1);
}

// The control loop of a robot, step by step: one next_action per period, an observation whenever the runner wants one, and the fake's
// inference released after exactly `latency` further steps. Returns the stats; `steps` receives every returned step index and `values`
// column 0 of each action.
struct LoopResult {
    VlaChunkRunnerStats stats;
    std::vector<int64_t> steps;
    std::vector<double> col0;
    std::vector<int64_t> obs_step;  // the step at which each observation was submitted (tag = index + 1)
    std::vector<int64_t> arrived_before;  // the step whose next_action call came first after the chunk was merged
};

LoopResult run_loop(int64_t rows, double threshold, double weight, int latency, int n_steps) {
    LoopResult out;
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    fake->rows = rows;
    auto r = make(fake, threshold, weight);
    int tag = 0, pending_for = -1;
    VlaAction a;
    // bootstrap: the first chunk before the loop starts (a robot waits for it once)
    EXPECT(r->submit_observation(obs_with_tag(++tag), 0).value());
    out.obs_step.push_back(0);
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    out.arrived_before.push_back(0);
    for (int64_t k = 0; k < n_steps; k++) {
        if (pending_for == 0) {
            fake->open_gate();
            EXPECT(r->wait_idle(kWaitS));
            out.arrived_before.push_back(static_cast<int64_t>(out.steps.size()));
            pending_for = -1;
        }
        EXPECT(r->next_action(t_of(k), &a).is_ok());
        if (a.available) {
            out.steps.push_back(a.step);
            out.col0.push_back(a.values[0]);
        }
        if (pending_for > 0) pending_for--;
        if (r->wants_observation()) {
            const int64_t at = a.available ? a.step : -1;
            EXPECT(r->submit_observation(obs_with_tag(++tag), t_of(k)).value());
            out.obs_step.push_back(at);
            pending_for = latency;
        }
    }
    out.stats = r->stats();
    EXPECT(fake->read(&Fake::max_active) == 1);
    fake->release_all();
    return out;
}

TEST(no_gap_when_latency_fits_in_the_threshold) {
    // rows 20, threshold 0.25: the next inference starts with 5 actions queued, so a chunk that arrives within 5 steps is in time.
    // (In steady state the merged chunk must also leave that many: rows - latency - 2 >= 5 here.)
    for (int latency = 0; latency <= 5; latency++) {
        const LoopResult res = run_loop(20, 0.25, 1.0, latency, 200);
        EXPECT_EQ(res.stats.stalled_calls, int64_t{0});
        EXPECT_EQ(static_cast<int64_t>(res.steps.size()), int64_t{200});
        bool consecutive = true;
        for (size_t i = 0; i < res.steps.size(); i++) consecutive = consecutive && res.steps[i] == static_cast<int64_t>(i);
        EXPECT(consecutive);
        // stale rows per chunk: row 0 (the step the observation was taken in) plus one per step of latency
        EXPECT_EQ(res.stats.stale_dropped, (res.stats.chunks_received - 1) * (latency + 1));
        EXPECT_EQ(res.stats.late_skipped, int64_t{0});
    }
    // one step slower than the threshold covers: exactly one stalled step per chunk
    const LoopResult slow = run_loop(20, 0.25, 1.0, 6, 200);
    EXPECT(slow.stats.stalled_calls > 0);
    EXPECT(slow.stats.stalled_calls <= slow.stats.chunks_received);
    // a threshold of 0 is the synchronous loop: every chunk costs `latency` stalled steps
    const LoopResult sync = run_loop(10, 0.0, 1.0, 3, 120);
    EXPECT(sync.stats.stalled_calls >= 3 * (sync.stats.chunks_received - 2));
}

TEST(replace_executes_the_newest_plan_in_order) {
    // Independent reference: with new_weight 1, step m runs row (m - s) of the newest chunk that covers m and arrived before m ran,
    // where s is that chunk's observation step.
    constexpr int64_t kRows = 20;
    for (int latency : {0, 2, 5, 7}) {
        const LoopResult res = run_loop(kRows, 0.25, 1.0, latency, 150);
        bool all = true;
        for (size_t idx = 0; idx < res.steps.size(); idx++) {
            const int64_t m = res.steps[idx];
            int best = -1;
            for (size_t c = 0; c < res.obs_step.size() && c < res.arrived_before.size(); c++) {
                const int64_t s = c == 0 ? 0 : res.obs_step[c];
                if (res.arrived_before[c] <= static_cast<int64_t>(idx) && m >= s && m < s + kRows) best = static_cast<int>(c);
            }
            const int64_t s = best <= 0 ? 0 : res.obs_step[static_cast<size_t>(best)];
            const double want = best < 0 ? NAN : fake_value(best + 1, m - s, 0);
            if (!(std::fabs(res.col0[idx] - want) < kBlendTol)) {
                if (all) std::fprintf(stderr, "  latency %d step %lld: got %.3f want %.3f\n", latency, static_cast<long long>(m), res.col0[idx], want);
                all = false;
            }
        }
        EXPECT(all);
    }
}

TEST(blend_matches_the_reference_and_drops_exactly_the_elapsed_steps) {
    // Chunk A (tag 1) from step 0; after step 4 runs (5 queued = 0.5 * 10) chunk B (tag 2) is requested with the observation of step 4,
    // and arrives after `latency` more steps. Steps 4 .. 4 + latency of B are stale; the rest of A's range is blended, the tail is B.
    for (double w : {0.7, 0.3, 1.0, 0.0}) {
        for (int latency = 0; latency <= 5; latency++) {
            auto fake = std::make_shared<Fake>();
            fake->gated = true;
            auto r = make(fake, 0.5, w);
            VlaAction a;
            EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
            fake->open_gate();
            EXPECT(r->wait_idle(kWaitS));
            int64_t k = 0;
            for (; k <= 4; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && a.step == k && row_equals(a, 1, k));
            EXPECT(r->wants_observation());
            EXPECT(r->submit_observation(obs_with_tag(2), t_of(4)).value());
            for (int i = 0; i < latency; i++, k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && a.step == k && row_equals(a, 1, k));
            fake->open_gate();
            EXPECT(r->wait_idle(kWaitS));
            const VlaChunkRunnerStats s = r->stats();
            EXPECT_EQ(s.stale_dropped, int64_t{latency + 1});
            EXPECT_EQ(s.blended, int64_t{10 - (5 + latency)});
            EXPECT_EQ(s.queued, int64_t{4 + 10 - (k)});
            bool ok = true;
            for (; k < 14; k++) {
                EXPECT(r->next_action(t_of(k), &a).is_ok());
                if (!a.available || a.step != k) { ok = false; continue; }
                for (int64_t j = 0; j < 2; j++) {
                    const double b = fake_value(2, k - 4, j);
                    const double want = k < 10 ? (1.0 - w) * fake_value(1, k, j) + w * b : b;
                    if (std::fabs(a.values[static_cast<size_t>(j)] - want) > kBlendTol) ok = false;
                }
            }
            EXPECT(ok);
            EXPECT(r->next_action(t_of(14), &a).is_ok() && !a.available);  // B ends at step 13
            EXPECT_EQ(r->stats().stalled_calls, int64_t{1});
        }
    }
}

TEST(a_chunk_later_than_its_own_length_is_dropped_whole) {
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    fake->rows = 4;
    auto r = make(fake, 1.0, 1.0);  // asks every step
    VlaAction a;
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    EXPECT(r->next_action(0, &a).is_ok() && a.step == 0);
    EXPECT(r->submit_observation(obs_with_tag(2), 0).value());  // observation of step 0: covers steps 0..3
    for (int64_t k = 1; k <= 3; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && row_equals(a, 1, k));
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    const VlaChunkRunnerStats s = r->stats();
    EXPECT_EQ(s.chunks_stale, int64_t{1});
    EXPECT_EQ(s.stale_dropped, int64_t{4});
    EXPECT_EQ(s.queued, int64_t{0});
}

TEST(an_observation_during_a_stall_starts_at_the_next_step) {
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    fake->rows = 3;
    auto r = make(fake, 0.0, 0.7);
    VlaAction a;
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    for (int64_t k = 0; k < 3; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && a.step == k);
    for (int64_t k = 3; k < 20; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && !a.available);  // stalled: the robot holds
    EXPECT(r->submit_observation(obs_with_tag(2), t_of(20)).value());
    for (int64_t k = 20; k < 25; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && !a.available);
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    // nothing ran while stalled, so nothing of the new chunk is stale and nothing was blended
    EXPECT(r->next_action(t_of(25), &a).is_ok() && a.available && a.step == 3 && a.skipped == 0 && row_equals(a, 2, 0));
    EXPECT(r->next_action(t_of(26), &a).is_ok() && a.step == 4 && row_equals(a, 2, 1));
    const VlaChunkRunnerStats s = r->stats();
    EXPECT_EQ(s.stale_dropped, int64_t{0});
    EXPECT_EQ(s.blended, int64_t{0});
    EXPECT_EQ(s.stalled_calls, int64_t{22});
}

TEST(late_calls_skip_actions_and_fast_calls_never_repeat) {
    auto fake = std::make_shared<Fake>();
    fake->rows = 30;
    auto r = make(fake, 0.0, 1.0);
    VlaAction a;
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    EXPECT(r->wait_idle(kWaitS));
    struct Call {
        double t;
        int64_t step, skipped;
    };
    // on time with jitter below half a period, then one late call (2 periods), one very late (3.6 -> 4), then a fast burst
    // (the burst runs ahead of the grid; the grid follows it to half a period ahead, so the gap after it counts from 9.4 periods)
    const Call calls[] = {{0.0, 0, 0},       {1.3 * kT, 1, 0},  {1.8 * kT, 2, 0},   {3.1 * kT, 3, 0},   {5.0 * kT, 5, 1},
                          {8.6 * kT, 9, 3},  {8.7 * kT, 10, 0}, {8.8 * kT, 11, 0},  {8.9 * kT, 12, 0},  {13.0 * kT, 16, 3}};
    for (const Call & c : calls) {
        EXPECT(r->next_action(c.t, &a).is_ok());
        EXPECT(a.available && a.step == c.step && a.skipped == c.skipped && row_equals(a, 1, c.step));
        if (a.step != c.step || a.skipped != c.skipped)
            std::fprintf(stderr, "  t=%.3f: step %lld skipped %lld, want %lld %lld\n", c.t, static_cast<long long>(a.step), static_cast<long long>(a.skipped),
                         static_cast<long long>(c.step), static_cast<long long>(c.skipped));
    }
    EXPECT_EQ(r->stats().late_skipped, int64_t{7});
    // later than the whole queue: the newest action, never one from the past
    EXPECT(r->next_action(100.0, &a).is_ok() && a.step == 29 && a.skipped == 12);
    EXPECT(r->next_action(100.0, &a).is_ok() && !a.available);
    // bad times
    EXPECT(r->next_action(99.0, &a).code() == Code::InvalidArgument);
    EXPECT(r->next_action(NAN, &a).code() == Code::InvalidArgument);
    EXPECT(r->next_action(101.0, nullptr).code() == Code::InvalidArgument);
    EXPECT(r->submit_observation(obs_with_tag(2), INFINITY).status().code() == Code::InvalidArgument);
}

TEST(a_steady_slow_loop_keeps_to_the_schedule) {
    // A loop calling every 1.5 periods alternates advances of 2 and 1 steps: 1.5 steps per call on average, no drift.
    auto fake = std::make_shared<Fake>();
    fake->rows = 400;
    auto r = make(fake, 0.0, 1.0);
    VlaAction a;
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    EXPECT(r->wait_idle(kWaitS));
    int64_t prev = -1;
    bool increasing = true;
    for (int i = 0; i <= 100; i++) {
        EXPECT(r->next_action(1.5 * kT * i, &a).is_ok() && a.available);
        increasing = increasing && a.step > prev;
        prev = a.step;
    }
    EXPECT(increasing);
    EXPECT(prev >= 149 && prev <= 151);  // 150 periods have passed
}

TEST(a_failed_inference_is_reported_and_the_chunk_keeps_running) {
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    fake->fail_call = 1;
    fake->fail_status = Status(Code::DataLoss, "camera frame truncated");
    auto r = make(fake, 0.5, 0.7);
    VlaAction a;
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    int64_t k = 0;
    for (; k <= 4; k++) EXPECT(r->next_action(t_of(k), &a).is_ok());
    EXPECT(r->submit_observation(obs_with_tag(2), t_of(4)).value());
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    VlaChunkRunnerStats s = r->stats();
    EXPECT_EQ(s.inferences_failed, int64_t{1});
    EXPECT(s.last_error.code() == Code::DataLoss && s.last_error.message() == "camera frame truncated");
    EXPECT_EQ(s.queued, int64_t{5});
    // the old chunk keeps running, untouched, and the runner asks again
    EXPECT(r->wants_observation());
    for (; k <= 6; k++) EXPECT(r->next_action(t_of(k), &a).is_ok() && a.step == k && row_equals(a, 1, k));
    EXPECT(r->submit_observation(obs_with_tag(3), t_of(6)).value());
    fake->open_gate();
    EXPECT(r->wait_idle(kWaitS));
    s = r->stats();
    EXPECT_EQ(s.chunks_received, int64_t{2});
    EXPECT(r->next_action(t_of(7), &a).is_ok() && a.step == 7);
}

TEST(malformed_chunks_and_exceptions_count_as_failures) {
    struct Case {
        const char * what;
        void (*setup)(Fake &);
        Code code;
    };
    const Case cases[] = {
        {"short", [](Fake & f) { f.bad_shape = true; }, Code::InvalidArgument},
        {"nan", [](Fake & f) { f.bad_value = true; }, Code::DataLoss},
        {"zero rows", [](Fake & f) { f.rows = 0; }, Code::InvalidArgument},
        {"too many rows", [](Fake & f) { f.rows = kVlaRunnerMaxChunkRows + 1; f.cols = 1; }, Code::InvalidArgument},
        {"throws", [](Fake & f) { f.throw_call = 0; }, Code::Internal},
    };
    for (const Case & c : cases) {
        auto fake = std::make_shared<Fake>();
        c.setup(*fake);
        auto r = make(fake, 0.5, 0.7);
        EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
        EXPECT(r->wait_idle(kWaitS));
        const VlaChunkRunnerStats s = r->stats();
        if (s.last_error.code() != c.code) std::fprintf(stderr, "  %s: %s\n", c.what, s.last_error.to_string().c_str());
        EXPECT(s.inferences_failed == 1 && s.last_error.code() == c.code && s.queued == 0 && s.chunks_received == 0);
    }
    // a chunk whose width differs from the earlier ones
    auto fake = std::make_shared<Fake>();
    fake->change_cols = true;
    auto r = make(fake, 1.0, 0.7);
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    EXPECT(r->wait_idle(kWaitS));
    EXPECT(r->submit_observation(obs_with_tag(2), 0).value());
    EXPECT(r->wait_idle(kWaitS));
    EXPECT(r->stats().inferences_failed == 1 && r->stats().last_error.code() == Code::InvalidArgument);
}

TEST(stop_waits_for_the_inference_in_flight_and_discards_it) {
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    auto r = make(fake, 0.5, 0.7);
    EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
    fake->wait_entered(1);
    std::atomic<bool> stopped{false};
    std::thread stopper([&] {
        r->stop();
        stopped = true;
    });
    // Wait until stop() has flagged the runner (a submit then reports Unsupported instead of "in flight"), so the inference finishes after it.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(kWaitS);
    while (r->submit_observation(obs_with_tag(9), 0).is_ok() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    // the gate is closed, so stop() cannot have returned
    EXPECT(!stopped.load());
    fake->open_gate();
    stopper.join();
    EXPECT(stopped.load());
    const VlaChunkRunnerStats s = r->stats();
    EXPECT(s.chunks_received == 0 && s.queued == 0 && !s.in_flight);
    EXPECT_EQ(fake->read(&Fake::finished), 1);
    EXPECT(!r->wants_observation());
    auto again = r->submit_observation(obs_with_tag(2), 0);
    EXPECT(!again.is_ok() && again.status().code() == Code::Unsupported);
    VlaAction a;
    EXPECT(r->next_action(0, &a).is_ok() && !a.available);
    r->stop();  // idempotent
    EXPECT(r->wait_idle(0));
}

TEST(destruction_with_an_inference_in_flight_joins_cleanly) {
    auto fake = std::make_shared<Fake>();
    fake->gated = true;
    std::thread opener;
    {
        auto r = make(fake, 0.5, 0.7);
        EXPECT(r->submit_observation(obs_with_tag(1), 0).value());
        fake->wait_entered(1);
        opener = std::thread([&] { fake->open_gate(); });
    }  // the destructor blocks until the gate opens
    opener.join();
    EXPECT_EQ(fake->read(&Fake::finished), 1);
}

TEST(control_and_submit_threads_against_the_worker) {
    // Real threads, no gate: the control thread steps with a synthetic clock while two other threads race to submit. At most one inference
    // runs at a time, steps never go backwards or repeat. Run under ThreadSanitizer (scripts/check-tsan.sh) to check for data races.
    auto fake = std::make_shared<Fake>();
    fake->rows = 20;
    auto r = make(fake, 0.5, 0.7);
    std::atomic<bool> done{false};
    std::atomic<int> tags{0};
    auto submitter = [&] {
        while (!done.load()) {
            if (r->wants_observation()) {
                auto s = r->submit_observation(obs_with_tag(++tags), 0.0);
                EXPECT(s.is_ok());
            }
            std::this_thread::yield();
        }
    };
    std::thread s1(submitter), s2(submitter);
    VlaAction a;
    int64_t prev = -1;
    bool ordered = true;
    constexpr int kIterations = 20000;
    for (int i = 0; i < kIterations; i++) {
        EXPECT(r->next_action(t_of(i), &a).is_ok());
        if (a.available) {
            ordered = ordered && a.step > prev;
            prev = a.step;
        }
        (void) r->stats();
    }
    done = true;
    s1.join();
    s2.join();
    r->stop();
    EXPECT(ordered);
    EXPECT_EQ(fake->read(&Fake::max_active), 1);
    const VlaChunkRunnerStats s = r->stats();
    EXPECT(s.chunks_received + s.inferences_failed <= s.inferences_started);
    std::printf("        %lld actions, %lld chunks, %lld stalled steps\n", static_cast<long long>(s.actions_returned), static_cast<long long>(s.chunks_received),
                static_cast<long long>(s.stalled_calls));
}

TEST(pipeline_function_rejects_a_null_pipeline) {
    auto r = vla_pipeline_infer_fn(nullptr)(VlaObservation());
    EXPECT(!r.is_ok() && r.status().code() == Code::InvalidArgument);
}

// ------------------------------------------------------------------------------------------------ real model (DOOMS_VLA_DIR)
namespace {

std::vector<float> tensor(const GgufFile & f, const std::string & name, Shape * shape = nullptr) {
    auto info = f.find_tensor(name);
    if (!info.is_ok()) {
        std::fprintf(stderr, "  %s\n", info.status().to_string().c_str());
        dooms_test::current_failures()++;
        return {};
    }
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    if (shape) *shape = info.value()->shape;
    return std::vector<float>(p, p + info.value()->shape.numel());
}

double rel_l2(const std::vector<double> & a, const std::vector<double> & b) {
    double d = 0, w = 0;
    for (size_t i = 0; i < a.size(); i++) { d += (a[i] - b[i]) * (a[i] - b[i]); w += b[i] * b[i]; }
    return std::sqrt(d / w);
}

}  // namespace

TEST(real_model_spliced_stream_matches_lerobot) {
    const char * env = std::getenv("DOOMS_VLA_DIR");
    if (!env) { std::printf("        (skipped: set DOOMS_VLA_DIR)\n"); return; }
    const std::string dir = env;
    auto ref = GgufFile::open(dir + "/reference.gguf");
    auto pipe = VlaPipeline::create({dir + "/vision-b1-fp16.mlpackage", dir + "/prefix-fp16.mlpackage", dir + "/expert-fp16.mlpackage", dir + "/host.gguf"});
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) return;
    const GgufFile & g = ref.value();
    VlaObservation obs;
    for (const char * name : {"image_camera1", "image_camera2", "image_camera3"}) {
        Shape sh;
        CameraImage c;
        c.chw = tensor(g, name, &sh);
        c.height = sh.dim(1);
        c.width = sh.dim(2);
        obs.cameras.push_back(std::move(c));
    }
    for (float t : tensor(g, "tokens")) obs.tokens.push_back(static_cast<int32_t>(t));
    for (float t : tensor(g, "token_mask")) obs.token_mask.push_back(t != 0.f ? 1 : 0);
    obs.state = tensor(g, "state");
    const std::vector<float> noise = tensor(g, "noise"), want = tensor(g, "actions");
    VlaPipeline & p = pipe.value();
    const int64_t n = p.chunk_size(), cols = p.action_dim(), robot = 6;
    // the reference noise every time, so every chunk is lerobot's chunk; the runner's splicing is what is tested
    VlaInferFn fn = [&p, &noise](const VlaObservation & o) -> Result<VlaChunk> {
        auto r = p.infer(o, &noise);
        if (!r.is_ok()) return r.status();
        VlaChunk c;
        c.actions = std::move(r).value();
        c.rows = p.chunk_size();
        c.cols = p.action_dim();
        return c;
    };
    constexpr double kWeight = 0.7, kThreshold = 0.5, kRealWaitS = 600;  // first call compiles the Core ML models
    constexpr double kTol = 5e-2;  // the pipeline's measured parity with lerobot (test_vla_pipeline: action_chunk_matches_lerobot)
    VlaChunkRunnerOptions o;
    o.new_weight = kWeight;
    o.chunk_size_threshold = kThreshold;
    auto created = VlaChunkRunner::create(fn, o);
    EXPECT(created.is_ok());
    if (!created.is_ok()) return;
    VlaChunkRunner & r = *created.value();
    VlaAction a;
    EXPECT(r.submit_observation(obs, 0).value());
    EXPECT(r.wait_idle(kRealWaitS));
    // Run until the runner asks (half the chunk left), submit at that step, then (inference finished before the next step) the rest.
    // Reference from lerobot's chunk W: steps < s + 1 use W[m]; the overlap uses 0.3 W[m] + 0.7 W[m - s]; the tail W[m - s].
    std::vector<double> got, ref_vals;
    int64_t k = 0, s = -1;
    for (; k < n + 20; k++) {  // the second chunk covers up to step s + n - 1 = 73
        EXPECT(r.next_action(t_of(k), &a).is_ok());
        EXPECT(a.available && a.step == k);
        if (!a.available) return;
        for (int64_t j = 0; j < robot; j++) {
            const auto at = [&](int64_t row) { return static_cast<double>(want[static_cast<size_t>(row * cols + j)]); };
            double w;
            if (s < 0 || k <= s) w = at(k);
            else w = k < n ? (1 - kWeight) * at(k) + kWeight * at(k - s) : at(k - s);
            got.push_back(a.values[static_cast<size_t>(j)]);
            ref_vals.push_back(w);
        }
        if (s < 0 && r.wants_observation()) {
            s = k;
            EXPECT(r.submit_observation(obs, t_of(k)).value());
            EXPECT(r.wait_idle(kRealWaitS));
        }
    }
    const VlaChunkRunnerStats st = r.stats();
    const double rel = rel_l2(got, ref_vals);
    std::printf("        observation 2 at step %lld; %lld steps, relative L2 vs lerobot-derived stream %.2e, blended %lld, stale %lld, last inference %.1f ms\n",
                static_cast<long long>(s), static_cast<long long>(k), rel, static_cast<long long>(st.blended), static_cast<long long>(st.stale_dropped), st.last_inference_ms);
    EXPECT(s == n - 1 - static_cast<int64_t>(kThreshold * n));
    EXPECT(rel < kTol);
    EXPECT_EQ(st.stalled_calls, int64_t{0});
    EXPECT_EQ(st.stale_dropped, int64_t{1});
    EXPECT_EQ(st.blended, n - s - 1);
}

int main() { return dooms_test::run_all(); }
