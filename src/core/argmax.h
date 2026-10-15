#pragma once
// Index of the largest element of a float array (the first one on ties), NEON-vectorized: a language model's vocabulary is 50k-250k logits per generated token.
#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace dooms {

// n >= 1. NaN is never chosen over a number; an all-NaN array gives 0.
inline size_t argmax_f32(const float * v, size_t n) {
    size_t i = 0;
    size_t best = 0;
    float best_val = v[0];
#if defined(__ARM_NEON)
    if (n >= 8) {
        float32x4_t mx = vld1q_f32(v);
        uint32x4_t idx = {0, 1, 2, 3};
        uint32x4_t cur = idx;
        const uint32x4_t step = vdupq_n_u32(4);
        for (i = 4; i + 4 <= n; i += 4) {
            cur = vaddq_u32(cur, step);
            const float32x4_t x = vld1q_f32(v + i);
            // strictly greater (the earlier index wins ties within a lane), or the lane still holds a NaN and x is a number
            const uint32x4_t lane_nan = vmvnq_u32(vceqq_f32(mx, mx));
            const uint32x4_t x_num = vceqq_f32(x, x);
            const uint32x4_t gt = vorrq_u32(vcgtq_f32(x, mx), vandq_u32(lane_nan, x_num));
            mx = vbslq_f32(gt, x, mx);
            idx = vbslq_u32(gt, cur, idx);
        }
        float lanes[4];
        uint32_t ids[4];
        vst1q_f32(lanes, mx);
        vst1q_u32(ids, idx);
        best_val = lanes[0];
        best = ids[0];
        for (int k = 1; k < 4; k++) {
            const bool take = (best_val != best_val && lanes[k] == lanes[k]) || lanes[k] > best_val || (lanes[k] == best_val && ids[k] < best);
            if (take) { best_val = lanes[k]; best = ids[k]; }
        }
    } else {
        i = 1;
    }
#else
    i = 1;
#endif
    for (; i < n; i++) {
        if (v[i] > best_val || (best_val != best_val && v[i] == v[i])) { best_val = v[i]; best = i; }
    }
    return best;
}

}  // namespace dooms
