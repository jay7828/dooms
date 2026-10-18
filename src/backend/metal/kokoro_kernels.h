#pragma once
// Metal Shading Language source of the Kokoro vocoder kernels (metal_kokoro_vocoder.mm), compiled when a vocoder is created. Parameter structs mirror the C++
// ones in metal_kokoro_vocoder.mm field for field (all 32-bit); the host checks their sizes.

namespace dooms {

inline const char * kokoro_kernel_source() {
    return R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// One 1-D convolution as an implicit GEMM: Y[oc][t(q)] = epilogue(sum_k W[oc][k] * B[k][q]), k = ic * taps + j, with the input column gathered on the fly:
//   B[k][q] = X[ic][src],  src = q * in_stride + j * dil + in_off  (in the virtual input of length tin * up: nearest upsampling by `up`),
//   zero where src falls outside [0, tin * up), t(q) = q * out_stride + out_off. X is half: the activation before the convolution was applied once per element by
//   adain_prep / act_prep (applying it in the gather repeated it for every tap and every output tile: ~22x for the kernel-11 layers).
// Ordinary convolutions: in_stride = stride, dil = dilation, in_off = -padding, out_stride 1. A transposed convolution (stride s, kernel K, padding p) is s of these,
// one per output phase phi = (t + p) mod s: taps K / s, dil -1, in_off 0, out_stride s, out_off phi - p, the phase's weights packed by the host.
struct C1Params {
    int ic, oc, taps;
    int tin, up;          // input length (row stride of X) and nearest-upsampling factor
    int ncol;             // columns q computed
    int in_stride, dil, in_off;
    int out_stride, out_off, tout;   // output positions and row stride of Y (and of the residual)
    int K, Kp;            // ic * taps, rounded up to the tile depth (row length of the packed weights)
    float post_scale;     // the epilogue's scale
    int flags;            // bit 0: + bias[oc]; bit 1: + R[oc][t] (residual, before the scale); bit 2: accumulate into Y (Y += v)
};

constant int BN = 64;   // output columns per threadgroup
constant int BK = 32;   // reduction depth per step

// 4 simdgroups in a 2 x 2 layout; a TM x 64 tile per threadgroup, each simdgroup (TM / 2) x 32 as (TM / 16) x 4 8x8 matrices. TM = 128 covers 128 output channels
// at once (the input columns are gathered once instead of once per 64 channels, and each simdgroup does 32 products per 12 operand loads instead of 16 per 8).
// A thread loads 8 consecutive columns of one reduction index and TM / 16 halves of weight rows; the next step's operands are fetched while the current step runs.
// TAPMAJOR: the reduction index is k = j * ic_count + ic (weights packed that way by the host) and ic_count is a multiple of BK, so every step has one tap j
// and BK consecutive channels: no division per element, and an 8-column run that lies inside the input is loaded without bounds checks (the fast path).
template <int TM, bool TAPMAJOR>
kernel void conv1d(device const half * x [[buffer(0)]], device const half * w [[buffer(1)]], device const float * bias [[buffer(2)]],
                   device const float * res [[buffer(3)]], device float * y [[buffer(4)]], constant C1Params & p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int SM = TM / 2, MI = SM / 8;                // simdgroup rows, 8x8 matrices along them
    constexpr int AROWS = TM / 64;                         // weight rows a thread loads per step (8 halves each)
    threadgroup float smem[TM * BN];
    threadgroup half * tA = (threadgroup half *)smem;
    threadgroup half * tB = tA + TM * BK;
    threadgroup float * tC = smem;
    const int m0 = int(tg.y) * TM, n0 = int(tg.x) * BN;
    const int sgm = (int(sg) >> 1) * SM, sgn = (int(sg) & 1) * 32;
    constexpr int KR = BK / 16;            // reduction rows a thread loads per step (16 rows per pass of 128 threads)
    constexpr int A4 = BK / 8;             // half4 a thread loads per weight row per step (two threads per row)
    const int kk = int(tid) >> 3;          // first reduction index within the step (then + 16, ...)
    const int c0 = (int(tid) & 7) * 8;     // first of the thread's 8 columns
    const int vlen = p.tin * p.up;
    simdgroup_float8x8 acc[MI][4];
    for (int i = 0; i < MI; i++) for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.f);
    device const half * wrow[AROWS];
    threadgroup half * arow[AROWS];
    for (int r = 0; r < AROWS; r++) {
        const int row = (int(tid) >> 1) + r * 64;
        wrow[r] = w + long(m0 + row) * p.Kp + (int(tid) & 1) * (BK / 2);
        arow[r] = tA + row * BK + (int(tid) & 1) * (BK / 2);
    }
    half4 ra[AROWS][A4];
    half rb[KR][8];
    auto fetch = [&](int k0) {
        for (int r = 0; r < AROWS; r++) for (int a = 0; a < A4; a++) ra[r][a] = *(device const half4 *)(wrow[r] + k0 + 4 * a);
        if (TAPMAJOR) {
            const int j = k0 / p.ic, ic0 = k0 - j * p.ic;
            const int base = (n0 + c0) * p.in_stride + j * p.dil + p.in_off;
            const bool inside = base >= 0 && base + 7 * p.in_stride < vlen && n0 + c0 + 7 < p.ncol && p.up == 1;
            for (int i = 0; i < KR; i++) {
                device const half * xr = x + long(ic0 + kk + 16 * i) * p.tin;
                if (inside) {
                    for (int e = 0; e < 8; e++) rb[i][e] = xr[base + e * p.in_stride];
                } else {
                    for (int e = 0; e < 8; e++) {
                        const int q = n0 + c0 + e, src = base + e * p.in_stride;
                        rb[i][e] = (q < p.ncol && uint(src) < uint(vlen)) ? xr[p.up == 1 ? src : src / p.up] : 0.h;
                    }
                }
            }
            return;
        }
        for (int i = 0; i < KR; i++) {
            const int k = k0 + kk + 16 * i;
            if (k < p.K) {
                const int ic = k / p.taps, j = k - ic * p.taps;
                device const half * xr = x + long(ic) * p.tin;
                const int base = (n0 + c0) * p.in_stride + j * p.dil + p.in_off;
                for (int e = 0; e < 8; e++) {
                    half v = 0.h;
                    const int q = n0 + c0 + e, src = base + e * p.in_stride;
                    if (q < p.ncol && uint(src) < uint(vlen)) v = xr[p.up == 1 ? src : src / p.up];
                    rb[i][e] = v;
                }
            } else {
                for (int e = 0; e < 8; e++) rb[i][e] = 0.h;
            }
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < p.K; k0 += BK) {
        for (int r = 0; r < AROWS; r++) for (int a = 0; a < A4; a++) *(threadgroup half4 *)(arow[r] + 4 * a) = ra[r][a];
        for (int i = 0; i < KR; i++) {
            *(threadgroup half4 *)(tB + (kk + 16 * i) * BN + c0) = half4(rb[i][0], rb[i][1], rb[i][2], rb[i][3]);
            *(threadgroup half4 *)(tB + (kk + 16 * i) * BN + c0 + 4) = half4(rb[i][4], rb[i][5], rb[i][6], rb[i][7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < p.K) fetch(k0 + BK);
        for (int kf = 0; kf < BK; kf += 8) {
            simdgroup_half8x8 a[MI], b[4];
            for (int i = 0; i < MI; i++) simdgroup_load(a[i], tA + (sgm + i * 8) * BK + kf, BK);
            for (int j = 0; j < 4; j++) simdgroup_load(b[j], tB + kf * BN + sgn + j * 8, BN);
            for (int i = 0; i < MI; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < MI; i++) for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], tC + (sgm + i * 8) * BN + sgn + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < TM * BN; e += 128) {
        const int r = e >> 6, c = e & 63;
        const int oc = m0 + r, q = n0 + c;
        if (oc >= p.oc || q >= p.ncol) continue;
        const int t = q * p.out_stride + p.out_off;
        if (t < 0 || t >= p.tout) continue;
        const long o = long(oc) * p.tout + t;
        float v = tC[e];
        if (p.flags & 1) v += bias[oc];
        if (p.flags & 2) v += res[o];
        v *= p.post_scale;
        y[o] = (p.flags & 4) ? y[o] + v : v;
    }
}

#define CONV1D_INSTANCE(NAME, TM, TAP) \
    template [[host_name(NAME)]] kernel void conv1d<TM, TAP>(device const half *, device const half *, device const float *, device const float *, device float *, \
                                                             constant C1Params &, uint2, uint, uint);
CONV1D_INSTANCE("conv1d_64", 64, false)
CONV1D_INSTANCE("conv1d_64_tap", 64, true)
CONV1D_INSTANCE("conv1d_128", 128, false)

// The activation before a convolution: 0 none, 1 leaky ReLU (slope), 2 Snake x + sin(a x)^2 / a (a per channel).
static float activate(float v, int act, float slope, float a) {
    if (act == 1) return v >= 0.f ? v : v * slope;
    if (act == 2) { const float s = sin(a * v); return v + s * s / a; }
    return v;
}

struct NormParams {
    int channels, len;   // x is [channels][len]
    int gb_off;          // offset of this layer's (gamma, beta) in the style projections: gamma = gb[gb_off + c], beta = gb[gb_off + channels + c]
    float eps;
    int act;             // activation after the AdaIN (activate())
    float slope;
};

// AdaIN + activation of one channel row, written as half for the next convolution: InstanceNorm1d (biased variance, two passes; affine weight w and bias b),
// (1 + gamma) * norm + beta folded into one scale and shift, then the activation. One threadgroup (256 threads) per channel; the row is read three times.
kernel void adain_prep(device const float * x [[buffer(0)]], device const float * w [[buffer(1)]], device const float * b [[buffer(2)]],
                       device const float * gb [[buffer(3)]], device half * y [[buffer(4)]], constant NormParams & p [[buffer(5)]],
                       device const float * alpha [[buffer(6)]],
                        uint c [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    device const float * xr = x + long(c) * p.len;
    float s = 0.f;
    for (int t = int(tid); t < p.len; t += 256) s += xr[t];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.f;
    for (int i = 0; i < 8; i++) mean += red[i];
    mean /= float(p.len);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.f;
    for (int t = int(tid); t < p.len; t += 256) { const float d = xr[t] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.f;
    for (int i = 0; i < 8; i++) var += red[i];
    const float rstd = rsqrt(var / float(p.len) + p.eps);
    const float g1 = 1.f + gb[p.gb_off + c], beta = gb[p.gb_off + p.channels + c];
    const float scale = g1 * w[c] * rstd, shift = g1 * b[c] - mean * scale + beta;
    const float a = p.act == 2 ? alpha[c] : 1.f;
    device half * yr = y + long(c) * p.len;
    for (int t = int(tid); t < p.len; t += 256) yr[t] = half(activate(xr[t] * scale + shift, p.act, p.slope, a));
}

struct ActParams { int channels, len; int act; float slope; };

// y = half(activate(x)) over [channels][len]: the input of a convolution without an AdaIN in front.
kernel void act_prep(device const float * x [[buffer(0)]], device half * y [[buffer(1)]], constant ActParams & p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    const int t = int(g.x), c = int(g.y);
    if (t >= p.len || c >= p.channels) return;
    const long i = long(c) * p.len + t;
    y[i] = half(activate(x[i], p.act, p.slope, 1.f));
}

// The style projections of every AdaIN layer at once: gb[r] = W[r] . s + bias[r] (W [rows][dim] f32). One simdgroup per row.
kernel void style_proj(device const float * W [[buffer(0)]], device const float * bias [[buffer(1)]], device const float * s [[buffer(2)]],
                       device float * gb [[buffer(3)]], constant int2 & p [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (int(row) >= p.x) return;
    float acc = 0.f;
    for (int k = int(lane); k < p.y; k += 32) acc += W[long(row) * p.y + k] * s[k];
    acc = simd_sum(acc);
    if (lane == 0) gb[row] = acc + bias[row];
}

struct DwParams { int channels, tin, tout; };

// AdainResBlk1d's upsampling `pool`: a depthwise ConvTranspose1d (kernel 3, stride 2, padding 1, output padding 1) of the prepped input (AdaIN + leaky, half),
// y[c][t] = bias[c] + sum_{k: (t + 1 - k) even, 0 <= (t + 1 - k) / 2 < tin} w[c][k] * x[c][(t + 1 - k) / 2], written as half for the next convolution.
kernel void dwconvt2(device const half * x [[buffer(0)]], device const float * w [[buffer(1)]], device const float * bias [[buffer(2)]],
                     device half * y [[buffer(4)]], constant DwParams & p [[buffer(5)]], uint2 g [[thread_position_in_grid]]) {
    const int t = int(g.x), c = int(g.y);
    if (t >= p.tout || c >= p.channels) return;
    float v = bias[c];
    for (int k = 0; k < 3; k++) {
        const int u = t + 1 - k;
        if (u < 0 || (u & 1)) continue;
        const int src = u >> 1;
        if (src >= p.tin) continue;
        v += w[c * 3 + k] * float(x[long(c) * p.tin + src]);
    }
    y[long(c) * p.tout + t] = half(v);
}

// The generator's last upsampling is followed by ReflectionPad1d((1, 0)) and the add of the noise branch: the convolution wrote u[t - 1] + src[t] at t >= 1;
// column 0 is u[1] + src[0] = (y[2] - src[2]) + src[0].
kernel void reflect_first(device float * y [[buffer(0)]], device const float * src [[buffer(1)]], constant int2 & p [[buffer(2)]], uint c [[thread_position_in_grid]]) {
    if (int(c) >= p.x) return;
    const long r = long(c) * p.y;
    y[r] = (y[r + 2] - src[r + 2]) + src[r];
}
)MSL";
}

}  // namespace dooms
