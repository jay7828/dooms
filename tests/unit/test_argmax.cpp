// argmax_f32 against the scalar definition (first index of the maximum; NaN never wins) on random data, ties, every length around the vector width, and extreme positions.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "check.h"
#include "core/argmax.h"

using namespace dooms;

namespace {

size_t reference(const std::vector<float> & v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[best] || (std::isnan(v[best]) && !std::isnan(v[i]))) best = i;
    return best;
}

}  // namespace

TEST(matches_the_scalar_loop_on_random_data) {
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    for (size_t n = 1; n <= 70; n++) {
        for (int rep = 0; rep < 20; rep++) {
            std::vector<float> v(n);
            for (float & x : v) x = nd(rng);
            EXPECT_EQ(argmax_f32(v.data(), n), reference(v));
        }
    }
    std::vector<float> big(151936);
    for (float & x : big) x = nd(rng);
    EXPECT_EQ(argmax_f32(big.data(), big.size()), reference(big));
}

TEST(ties_pick_the_first_index_and_the_maximum_may_sit_anywhere) {
    for (size_t n : {8u, 9u, 13u, 64u, 67u}) {
        for (size_t at = 0; at < n; at++) {
            std::vector<float> v(n, 0.f);
            v[at] = 5.f;
            EXPECT_EQ(argmax_f32(v.data(), n), at);
            if (at + 3 < n) {  // a second equal maximum later: the first wins
                v[at + 3] = 5.f;
                EXPECT_EQ(argmax_f32(v.data(), n), at);
            }
        }
        std::vector<float> flat(n, 1.f);
        EXPECT_EQ(argmax_f32(flat.data(), n), static_cast<size_t>(0));
    }
}

TEST(nan_and_infinities) {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    std::vector<float> v(20, 1.f);
    v[3] = nan;
    v[11] = 2.f;
    EXPECT_EQ(argmax_f32(v.data(), v.size()), static_cast<size_t>(11));
    v[17] = inf;
    EXPECT_EQ(argmax_f32(v.data(), v.size()), static_cast<size_t>(17));
    // NaN anywhere (each lane position), against the scalar definition
    for (size_t at = 0; at < 20; at++) {
        std::vector<float> w(20);
        for (size_t i = 0; i < w.size(); i++) w[i] = static_cast<float>((i * 7) % 11);
        w[at] = nan;
        EXPECT_EQ(argmax_f32(w.data(), w.size()), reference(w));
    }
    std::vector<float> all_nan(10, nan);
    EXPECT_EQ(argmax_f32(all_nan.data(), all_nan.size()), static_cast<size_t>(0));
    std::vector<float> neg(33, -inf);
    neg[30] = -1e30f;
    EXPECT_EQ(argmax_f32(neg.data(), neg.size()), static_cast<size_t>(30));
}

int main() { return dooms_test::run_all(); }
