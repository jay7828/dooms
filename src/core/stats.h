#pragma once
// Small statistics helpers shared by the profiler and the benchmarks.

#include <vector>

namespace dooms {

// The q-quantile (0 <= q <= 1) of `values` by the nearest-rank-below rule: sorted[floor(q * n)], clamped to
// the last element. So q = 0.5 of {1,2,3,4} is 3 (the upper median) and q = 0.5 of {1,2,3} is 2.
// Empty input returns 0. `values` is taken by value because it is sorted.
double quantile(std::vector<double> values, double q);
inline double median(std::vector<double> values) { return quantile(std::move(values), 0.5); }

}  // namespace dooms
