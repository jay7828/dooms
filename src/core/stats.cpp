#include "core/stats.h"

#include <algorithm>
#include <cmath>

#include "core/check.h"

namespace dooms {

double quantile(std::vector<double> values, double q) {
    DOOMS_CHECK(q >= 0.0 && q <= 1.0 && !std::isnan(q), "quantile: q must be in [0, 1]");
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t idx = std::min(values.size() - 1, static_cast<size_t>(q * static_cast<double>(values.size())));
    return values[idx];
}

}  // namespace dooms
