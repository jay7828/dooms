#pragma once
// A minimal fork-join loop for independent rows of work (integer or per-element float arithmetic whose result does not depend on how the rows are split).

#include <algorithm>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace dooms {

// Runs fn(begin, end) over [0, n) split into contiguous chunks on up to `max_threads` threads (the calling thread takes the first chunk). Jobs smaller than
// `min_work_per_thread` units of work (n * work_per_item in total) run on the calling thread alone.
inline void parallel_for(int64_t n, int64_t work_per_item, const std::function<void(int64_t, int64_t)> & fn, int max_threads = 8, int64_t min_work_per_thread = int64_t(1) << 18) {
    if (n <= 0) return;
    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int64_t total = n * std::max<int64_t>(work_per_item, 1);
    const int threads = static_cast<int>(std::min<int64_t>({max_threads, hw, n, total / std::max<int64_t>(min_work_per_thread, 1)}));
    if (threads <= 1) { fn(0, n); return; }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads - 1));
    for (int t = 1; t < threads; t++) pool.emplace_back([&fn, n, t, threads] { fn(n * t / threads, n * (t + 1) / threads); });
    fn(0, n / threads);
    for (auto & th : pool) th.join();
}

}  // namespace dooms
