#pragma once
// Asynchronous action chunking for a VLA policy in a robot control loop ("async inference" as in lerobot 0.4.4's
// async_inference/robot_client.py, run in-process instead of over gRPC).
//
// A synchronous loop executes a whole chunk, then stops for one inference (~270 ms for SmolVLA). The runner instead starts the
// next inference from a fresh observation while the current chunk is still being executed, on its own worker thread, and
// splices the new chunk in when it arrives:
//   - every action has a control step index; chunk row i of an observation tagged with step s is the action for step s + i;
//   - rows whose step has already been executed when the chunk arrives are dropped (exactly the elapsed steps);
//   - rows for steps the old chunk also covers are blended: (1 - new_weight) * old + new_weight * new (lerobot's aggregate_fn);
//   - rows past the old chunk's end are taken as they are; old rows past the new chunk's end are dropped (the newer plan wins).
// Time enters only through the timestamps the caller passes (seconds on any monotonic clock), so the behaviour is deterministic
// and testable without sleeping. See src/pipeline/CLAUDE.md for the step-tagging rules and docs/USING.md for a control loop.
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "core/status.h"
#include "pipeline/vla_pipeline.h"

namespace dooms {

// lerobot 0.4.4 async_inference defaults (constants.py DEFAULT_FPS, configs.py chunk_size_threshold and the
// "weighted_average" aggregate function 0.3 * old + 0.7 * new).
constexpr double kVlaRunnerDefaultControlHz = 30.0;
constexpr double kVlaRunnerDefaultChunkThreshold = 0.5;
constexpr double kVlaRunnerDefaultNewWeight = 0.7;
// Validation bounds: a control rate above this is not a robot arm loop (and makes one period shorter than a mutex hand-off).
constexpr double kVlaRunnerMaxControlHz = 10000.0;
// A chunk with more rows than this is rejected as a broken inference function (SmolVLA returns 50).
constexpr int64_t kVlaRunnerMaxChunkRows = 1 << 16;

struct VlaChunkRunnerOptions {
    double control_hz = kVlaRunnerDefaultControlHz;  // control steps per second; one next_action() call per step
    // Start the next inference when (queued actions) / (rows of the last chunk) <= this, as lerobot's chunk_size_threshold.
    // 0 waits until the queue is empty (synchronous behaviour, with a stall per chunk); 1 asks for an observation every step.
    double chunk_size_threshold = kVlaRunnerDefaultChunkThreshold;
    // Weight of the new chunk where it overlaps the queued one: 1 replaces ("latest_only"), 0.7 is lerobot's "weighted_average",
    // 0.5 "average", 0.3 "conservative".
    double new_weight = kVlaRunnerDefaultNewWeight;
};

// lerobot's aggregate_fn_name -> new_weight. InvalidArgument (listing the names) for anything else.
Result<double> vla_aggregate_weight(const std::string & name);

struct VlaChunk {
    std::vector<float> actions;  // rows x cols, row-major
    int64_t rows = 0, cols = 0;
};

// Computes one chunk from one observation. Called only on the runner's worker thread, one call at a time.
using VlaInferFn = std::function<Result<VlaChunk>(const VlaObservation &)>;

// Wires a pipeline into a runner. The pipeline must outlive the runner and must not be used by anyone else while the runner exists
// (VlaPipeline is not thread-safe and the worker thread calls it).
VlaInferFn vla_pipeline_infer_fn(VlaPipeline * pipeline);

struct VlaAction {
    bool available = false;    // false: no action for this step (before the first chunk, or the queue ran dry: a stall)
    int64_t step = -1;         // control step index of the returned action (-1 when !available)
    int64_t skipped = 0;       // queued actions dropped in this call because their time had passed (the loop was late)
    std::vector<float> values; // one row (the chunk's cols), reused across calls
};

struct VlaChunkRunnerStats {
    int64_t actions_returned = 0;
    int64_t waiting_calls = 0;      // next_action() calls without an action before the first one was ever returned
    int64_t stalled_calls = 0;      // ... after it: the loop ran out of actions (the number to drive to zero)
    int64_t late_skipped = 0;       // actions skipped because the control loop called late (sum of VlaAction::skipped)
    int64_t stale_dropped = 0;      // rows of arrived chunks whose step had already been executed
    int64_t blended = 0;            // rows aggregated with a queued action
    int64_t chunks_received = 0;    // successful inferences merged into the queue (including entirely stale ones)
    int64_t chunks_stale = 0;       // ... of which every row had already passed
    int64_t inferences_started = 0;
    int64_t inferences_failed = 0;
    int64_t queued = 0;             // actions waiting right now
    int64_t last_chunk_rows = 0;
    bool in_flight = false;
    double last_inference_ms = 0;   // wall time of the last completed inference (steady clock), for reporting only
    Status last_error;              // of the last failed inference (Ok if none failed yet)
};

// Thread safety: every public method may be called from any thread (one mutex guards the state); the intended use is one control
// thread calling wants_observation / submit_observation / next_action while the worker runs the inference. The inference function
// runs outside the lock. Destruction joins the worker and therefore waits for an inference in flight to finish (it cannot be
// interrupted); its result is discarded.
class VlaChunkRunner {
public:
    //   InvalidArgument  missing function, control_hz outside (0, kVlaRunnerMaxControlHz], threshold or new_weight outside [0, 1]
    static Result<std::unique_ptr<VlaChunkRunner>> create(VlaInferFn infer, const VlaChunkRunnerOptions & options = VlaChunkRunnerOptions());
    ~VlaChunkRunner();
    VlaChunkRunner(const VlaChunkRunner &) = delete;
    VlaChunkRunner & operator=(const VlaChunkRunner &) = delete;

    // True when submit_observation() would start an inference: none in flight and the queue is at or below the threshold
    // (always true before the first chunk). Cheap; call it every step and capture an observation only when it is true.
    bool wants_observation() const;

    // Starts an inference on `observation`, taken at `timestamp_s`, if wants_observation(); returns whether it did (an observation
    // that is not needed is dropped, never queued, so it cannot go stale).
    //   InvalidArgument  non-finite timestamp
    //   Unsupported      after stop()
    Result<bool> submit_observation(VlaObservation observation, double timestamp_s);

    // The action for the control step at `now_s`. Never waits for an inference. Advances one step per call, more if the loop was
    // late by whole periods (those actions are skipped, never executed late). out->available is false before the first chunk
    // and when the queue has run dry (the caller holds its last command).
    //   InvalidArgument  non-finite time, time earlier than the previous call, out == nullptr
    Status next_action(double now_s, VlaAction * out);

    // Blocks until no inference is in flight or `timeout_s` passes; returns whether it is idle. For start-up (wait for the first
    // chunk) and tests; a control loop never needs it.
    bool wait_idle(double timeout_s) const;

    VlaChunkRunnerStats stats() const;

    // Stops accepting observations and joins the worker (waiting for an inference in flight). Queued actions stay available
    // to next_action(). Idempotent.
    void stop();

private:
    VlaChunkRunner(VlaInferFn infer, const VlaChunkRunnerOptions & options);
    void worker_loop();
    // Both run with mu_ held.
    void merge_chunk(const VlaChunk & chunk, int64_t obs_step);
    int64_t queued_locked() const { return q_end_ - q_begin_; }

    const VlaInferFn infer_;
    const VlaChunkRunnerOptions options_;
    const double period_s_;

    mutable std::mutex mu_;
    mutable std::condition_variable cv_;  // job posted, job finished, stop requested
    // The queue holds the contiguous steps [q_begin_, q_end_); row k - q_base_ of queue_ is the action for step k.
    std::vector<float> queue_, scratch_;
    int64_t q_base_ = 0, q_begin_ = 0, q_end_ = 0, cols_ = 0;
    int64_t step_ = -1;           // step of the last returned action (-1: none yet)
    double step_time_ = 0;        // ideal time of step_ on the control grid, re-anchored when the loop drifts or stalls
    double last_call_s_ = 0;      // time of the previous next_action() call (monotonicity check)
    bool called_ = false, stalled_ = false;
    struct Job {
        VlaObservation observation;
        int64_t obs_step = 0;
    };
    std::optional<Job> job_;
    bool in_flight_ = false, stopping_ = false;
    VlaChunkRunnerStats stats_;

    std::mutex join_mu_;
    std::thread worker_;
};

}  // namespace dooms
