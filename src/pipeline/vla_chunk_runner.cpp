#include "pipeline/vla_chunk_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <system_error>
#include <utility>

#include "core/check.h"

namespace dooms {

namespace {

// lerobot 0.4.4 async_inference/configs.py AGGREGATE_FUNCTIONS, written as the weight of the new action.
struct NamedWeight {
    const char * name;
    double new_weight;
};
constexpr NamedWeight kAggregateFunctions[] = {
    {"weighted_average", 0.7},  // 0.3 * old + 0.7 * new
    {"latest_only", 1.0},
    {"average", 0.5},
    {"conservative", 0.3},      // 0.7 * old + 0.3 * new
};

Status validate_chunk(const VlaChunk & c, int64_t expected_cols) {
    if (c.rows < 1 || c.rows > kVlaRunnerMaxChunkRows || c.cols < 1)
        return invalid_argument("the inference returned a chunk of " + std::to_string(c.rows) + " x " + std::to_string(c.cols) + " actions; rows must be in [1, " +
                                std::to_string(kVlaRunnerMaxChunkRows) + "] and cols >= 1");
    if (static_cast<int64_t>(c.actions.size()) != c.rows * c.cols)
        return invalid_argument("the inference returned " + std::to_string(c.actions.size()) + " values for a " + std::to_string(c.rows) + " x " + std::to_string(c.cols) + " chunk");
    if (expected_cols != 0 && c.cols != expected_cols)
        return invalid_argument("the inference returned " + std::to_string(c.cols) + " values per action, earlier chunks had " + std::to_string(expected_cols));
    for (float v : c.actions)
        if (!std::isfinite(v)) return data_loss("the inference returned a non-finite action value");
    return Status();
}

}  // namespace

Result<double> vla_aggregate_weight(const std::string & name) {
    std::string names;
    for (const NamedWeight & w : kAggregateFunctions) {
        if (name == w.name) return w.new_weight;
        names += names.empty() ? "" : ", ";
        names += w.name;
    }
    return invalid_argument("unknown aggregate function '" + name + "' (expected one of: " + names + ")");
}

VlaInferFn vla_pipeline_infer_fn(VlaPipeline * pipeline) {
    return [pipeline](const VlaObservation & observation) -> Result<VlaChunk> {
        if (!pipeline) return invalid_argument("vla_pipeline_infer_fn was given a null pipeline");
        auto r = pipeline->infer(observation);
        if (!r.is_ok()) return r.status();
        VlaChunk c;
        c.actions = std::move(r).value();
        c.rows = pipeline->chunk_size();
        c.cols = pipeline->action_dim();
        return c;
    };
}

Result<std::unique_ptr<VlaChunkRunner>> VlaChunkRunner::create(VlaInferFn infer, const VlaChunkRunnerOptions & options) {
    if (!infer) return invalid_argument("VlaChunkRunner needs an inference function");
    if (!(options.control_hz > 0 && options.control_hz <= kVlaRunnerMaxControlHz))
        return invalid_argument("control_hz must be in (0, " + std::to_string(kVlaRunnerMaxControlHz) + "], got " + std::to_string(options.control_hz));
    if (!(options.chunk_size_threshold >= 0 && options.chunk_size_threshold <= 1))
        return invalid_argument("chunk_size_threshold must be in [0, 1] (a fraction of the chunk), got " + std::to_string(options.chunk_size_threshold));
    if (!(options.new_weight >= 0 && options.new_weight <= 1))
        return invalid_argument("new_weight must be in [0, 1] (1 replaces the queued actions), got " + std::to_string(options.new_weight));
    std::unique_ptr<VlaChunkRunner> runner(new VlaChunkRunner(std::move(infer), options));
    try {
        runner->worker_ = std::thread(&VlaChunkRunner::worker_loop, runner.get());
    } catch (const std::system_error & e) {
        return resource_exhausted(std::string("could not start the runner's worker thread: ") + e.what());
    }
    return runner;
}

VlaChunkRunner::VlaChunkRunner(VlaInferFn infer, const VlaChunkRunnerOptions & options)
    : infer_(std::move(infer)), options_(options), period_s_(1.0 / options.control_hz) {}

VlaChunkRunner::~VlaChunkRunner() { stop(); }

void VlaChunkRunner::stop() {
    std::lock_guard<std::mutex> join_lock(join_mu_);
    {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool VlaChunkRunner::wants_observation() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_ || in_flight_) return false;
    if (stats_.last_chunk_rows == 0) return true;
    return static_cast<double>(queued_locked()) <= options_.chunk_size_threshold * static_cast<double>(stats_.last_chunk_rows);
}

Result<bool> VlaChunkRunner::submit_observation(VlaObservation observation, double timestamp_s) {
    if (!std::isfinite(timestamp_s)) return invalid_argument("submit_observation: the timestamp must be finite");
    std::unique_lock<std::mutex> lock(mu_);
    if (stopping_) return unsupported("submit_observation: the runner has been stopped");
    if (in_flight_) return false;
    if (stats_.last_chunk_rows != 0 && static_cast<double>(queued_locked()) > options_.chunk_size_threshold * static_cast<double>(stats_.last_chunk_rows)) return false;
    // Row 0 of the chunk is the action for the step at which the observation was taken. While actions are flowing that step is read
    // off the control grid anchored at the last returned step; before the first action and during a stall the robot is holding
    // still and the next step to execute is step_ + 1. It never lies past the queue's end: the loop would stall there.
    int64_t obs_step = step_ + 1;
    if (step_ >= 0 && !stalled_) obs_step = std::min(step_ + static_cast<int64_t>(std::llround((timestamp_s - step_time_) / period_s_)), q_end_);
    job_ = Job{std::move(observation), obs_step};
    in_flight_ = true;
    stats_.inferences_started++;
    lock.unlock();
    cv_.notify_all();
    return true;
}

Status VlaChunkRunner::next_action(double now_s, VlaAction * out) {
    if (!out) return invalid_argument("next_action: out is null");
    if (!std::isfinite(now_s)) return invalid_argument("next_action: the time must be finite");
    std::lock_guard<std::mutex> lock(mu_);
    if (called_ && now_s < last_call_s_)
        return invalid_argument("next_action: time went backwards (" + std::to_string(now_s) + " s after " + std::to_string(last_call_s_) + " s); pass a monotonic clock");
    called_ = true;
    last_call_s_ = now_s;
    out->available = false;
    out->step = -1;
    out->skipped = 0;
    if (queued_locked() == 0) {
        if (step_ < 0) {
            stats_.waiting_calls++;
        } else {
            stats_.stalled_calls++;
            stalled_ = true;
        }
        return Status();
    }
    const bool resync = step_ < 0 || stalled_;
    // Whole control periods since the last step (at least one: every call is a step). Late calls skip the actions whose time passed.
    int64_t advance = 1;
    if (!resync) advance = std::max<int64_t>(1, std::llround((now_s - step_time_) / period_s_));
    // q_begin_ == step_ + 1 always; if the loop is later than the whole queue, the newest action is the one closest in time.
    const int64_t target = std::min(step_ + advance, q_end_ - 1);
    out->skipped = target - q_begin_;
    out->step = target;
    out->available = true;
    const float * row = queue_.data() + (target - q_base_) * cols_;
    out->values.assign(row, row + cols_);
    if (resync) {
        step_time_ = now_s;
    } else {
        // Keep the grid within half a period of the caller's clock: a loop that runs fast cannot bank time, a slow one cannot lag.
        step_time_ += static_cast<double>(target - step_) * period_s_;
        step_time_ = std::min(std::max(step_time_, now_s - 0.5 * period_s_), now_s + 0.5 * period_s_);
    }
    step_ = target;
    q_begin_ = target + 1;
    stalled_ = false;
    stats_.actions_returned++;
    stats_.late_skipped += out->skipped;
    return Status();
}

bool VlaChunkRunner::wait_idle(double timeout_s) const {
    std::unique_lock<std::mutex> lock(mu_);
    const auto timeout = std::chrono::duration<double>(std::max(0.0, timeout_s));
    return cv_.wait_for(lock, timeout, [&] { return !in_flight_; });
}

VlaChunkRunnerStats VlaChunkRunner::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    VlaChunkRunnerStats s = stats_;
    s.queued = queued_locked();
    s.in_flight = in_flight_;
    return s;
}

void VlaChunkRunner::merge_chunk(const VlaChunk & chunk, int64_t obs_step) {
    const int64_t first = step_ + 1;  // == q_begin_: steps before it have been executed
    const int64_t chunk_end = obs_step + chunk.rows;
    stats_.chunks_received++;
    stats_.last_chunk_rows = chunk.rows;
    stats_.stale_dropped += std::max<int64_t>(0, std::min(first, chunk_end) - obs_step);
    if (chunk_end <= first) {
        stats_.chunks_stale++;
        return;
    }
    // submit_observation clamps obs_step to the queue end of that moment, and the end only moves in merges (one in flight).
    DOOMS_CHECK(obs_step <= q_end_, "VlaChunkRunner: an observation step past the queue end would leave a hole");
    const int64_t cols = chunk.cols;
    const double w = options_.new_weight;
    scratch_.resize(static_cast<size_t>((chunk_end - first) * cols));
    for (int64_t k = first; k < chunk_end; k++) {
        float * dst = scratch_.data() + (k - first) * cols;
        const float * old_row = (k < q_end_) ? queue_.data() + (k - q_base_) * cols_ : nullptr;
        const float * new_row = (k >= obs_step) ? chunk.actions.data() + (k - obs_step) * cols : nullptr;
        if (new_row && old_row) {
            for (int64_t j = 0; j < cols; j++) dst[j] = static_cast<float>((1.0 - w) * old_row[j] + w * new_row[j]);
            stats_.blended++;
        } else {
            std::copy(new_row ? new_row : old_row, (new_row ? new_row : old_row) + cols, dst);
        }
    }
    std::swap(queue_, scratch_);
    q_base_ = q_begin_ = first;
    q_end_ = chunk_end;
    cols_ = cols;
}

void VlaChunkRunner::worker_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
        cv_.wait(lock, [&] { return stopping_ || job_.has_value(); });
        if (stopping_) {
            job_.reset();
            in_flight_ = false;
            cv_.notify_all();
            return;
        }
        Job job = std::move(*job_);
        job_.reset();
        lock.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        Result<VlaChunk> result = internal_error("the inference function did not run");
        try {
            result = infer_(job.observation);
        } catch (const std::exception & e) {
            result = internal_error(std::string("the inference function threw: ") + e.what());
        } catch (...) {
            result = internal_error("the inference function threw a non-standard exception");
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        lock.lock();
        in_flight_ = false;
        if (!stopping_) {
            stats_.last_inference_ms = ms;
            Status st = result.is_ok() ? validate_chunk(result.value(), cols_) : result.status();
            if (st.is_ok()) {
                merge_chunk(result.value(), job.obs_step);
            } else {
                stats_.inferences_failed++;
                stats_.last_error = std::move(st);
            }
        }
        cv_.notify_all();
    }
}

}  // namespace dooms
