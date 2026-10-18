#pragma once
// Metal Shading Language source of the Llama decoder kernels (metal_llama_decoder.mm), compiled when a decoder is created. Parameter structs mirror the C++ ones
// in metal_llama_decoder.mm field for field (all 32-bit); the host checks their sizes. Constants named k... below must equal the host's constants of the same
// names. Measurements behind the shapes: src/backend/metal/CLAUDE.md "Llama decoder".

namespace dooms {

inline const char * llama_kernel_source() {
    return R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint kSimdgroupsPerGroup = 4;      // simdgroups per threadgroup (128 threads) in every kernel but argmax
constant constexpr uint kGroupThreads = kSimdgroupsPerGroup * 32;
constant constexpr uint kDecodeRowsPerSimdgroup = 2;  // rows a simdgroup computes at once in matvec1 (independent loads in flight)
constant constexpr uint kGateRowsPerSimdgroup = 1;    // outputs a simdgroup computes at once in gate_up1 (each needs a gate row and an up row)
constant constexpr uint kMaxHeadDim = 256;
constant constexpr uint kAttnSimdgroups = 8;          // attention threadgroup: 256 threads (more positions in parallel for long contexts)
constant constexpr uint kAttnThreads = kAttnSimdgroups * 32;
// matvec1 / gate_up1 variant: true stages the input row in threadgroup memory (short rows); false reads it from device memory (long rows, where staging
// would take so much threadgroup memory that one threadgroup fills a core).
constant bool kStageInput [[function_constant(0)]];

struct MatvecParams {
    uint K;         // input width (a multiple of 4)
    uint N;         // output rows
    uint flags;     // bit 0: RMS-normalize the input with gamma first; bit 1: add into the output (residual) instead of overwriting it; bit 2: add bias[r];
                    // bit 3: LayerNorm the input instead (mean and variance, then gamma and beta; needs the staged input); bit 4: GELU (exact) on the output
    float eps;
};

// erf with a maximum absolute error of 1.5e-7 (Abramowitz and Stegun 7.1.26): the Metal standard library has none.
static float erf_approx(float x) {
    const float t = 1.f / (1.f + 0.3275911f * fabs(x));
    const float y = 1.f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}

static float gelu_erf(float x) { return 0.5f * x * (1.f + erf_approx(x * 0.70710678118654752f)); }

// LayerNorm of x[0..K) by the whole threadgroup, staged into xs as (x - mean) / sqrt(var + eps) * gamma + beta (two passes: mean, then variance).
static void stage_input_ln(device const float * x, device const float * gamma, device const float * beta, uint K, float eps, threadgroup float * xs,
                           threadgroup float * red, uint tid, uint lane, uint sg) {
    float s = 0.f;
    for (uint k = tid; k < K; k += kGroupThreads) s += x[k];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.f;
    for (uint i = 0; i < kSimdgroupsPerGroup; i++) mean += red[i];
    mean /= float(K);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.f;
    for (uint k = tid; k < K; k += kGroupThreads) { const float d = x[k] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.f;
    for (uint i = 0; i < kSimdgroupsPerGroup; i++) var += red[i];
    const float inv = rsqrt(var / float(K) + eps);
    for (uint k = tid; k < K; k += kGroupThreads) xs[k] = (x[k] - mean) * inv * gamma[k] + beta[k];
}

// ---- One token (decode) ----

// Sum of squares of x[0..K) by the whole threadgroup (staged as x * gamma, or x, into xs when kStageInput); returns 1 / rms (or 1) to every thread.
static float stage_input(device const float * x, device const float * gamma, uint K, bool norm, float eps, threadgroup float * xs, threadgroup float * red,
                         uint tid, uint lane, uint sg) {
    float ss = 0.f;
    for (uint k = tid; k < K; k += kGroupThreads) {
        const float v = x[k];
        ss += v * v;
        if (kStageInput) xs[k] = norm ? v * gamma[k] : v;
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint i = 0; i < kSimdgroupsPerGroup; i++) total += red[i];
    return norm ? rsqrt(total / float(K) + eps) : 1.f;
}

// Element k4 (4 values) of the input row as the projection sees it.
static float4 input4(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint k4) {
    if (kStageInput) return xs4[k4];
    const float4 v = ((device const float4 *)x)[k4];
    return norm ? v * ((device const float4 *)gamma)[k4] : v;
}

// y[r] (+)= W[r] . xin. The input is staged once per threadgroup (threadgroup memory sized by the host to K floats); each simdgroup runs
// kDecodeRowsPerSimdgroup rows together, lanes striding the rows in half4 steps.
kernel void matvec1(device const half * W [[buffer(0)]], device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                    device const float * gamma [[buffer(3)]], constant MatvecParams & p [[buffer(4)]], device const float * bias [[buffer(5)]],
                    device const float * beta [[buffer(6)]], threadgroup float4 * xs4 [[threadgroup(0)]],
                    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kSimdgroupsPerGroup];
    const bool layernorm = (p.flags & 8u) != 0;
    const bool norm = (p.flags & 1u) != 0 && !layernorm, accumulate = (p.flags & 2u) != 0;
    float scale = 1.f;
    if (layernorm) stage_input_ln(x, gamma, beta, p.K, p.eps, (threadgroup float *)xs4, red, tid, lane, sg);   // host guarantees kStageInput
    else scale = stage_input(x, gamma, p.K, norm, p.eps, (threadgroup float *)xs4, red, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (gid * kSimdgroupsPerGroup + sg) * kDecodeRowsPerSimdgroup;
    if (row0 >= p.N) return;
    const uint K4 = p.K / 4;
    float acc[kDecodeRowsPerSimdgroup];
    for (uint i = 0; i < kDecodeRowsPerSimdgroup; i++) acc[i] = 0.f;
    device const half4 * w[kDecodeRowsPerSimdgroup];
    for (uint i = 0; i < kDecodeRowsPerSimdgroup; i++) w[i] = (device const half4 *)(W + (ulong)min(row0 + i, p.N - 1) * p.K);
    for (uint k4 = lane; k4 < K4; k4 += 32) {
        const float4 xv = input4(xs4, x, gamma, norm, k4);
        for (uint i = 0; i < kDecodeRowsPerSimdgroup; i++) acc[i] += dot(float4(w[i][k4]), xv);
    }
    for (uint i = 0; i < kDecodeRowsPerSimdgroup; i++) {
        const uint r = row0 + i;
        float s = simd_sum(acc[i]) * scale + ((p.flags & 4u) && r < p.N ? bias[r] : 0.f);
        if (p.flags & 16u) s = gelu_erf(s);
        if (lane == 0 && r < p.N) y[r] = accumulate ? y[r] + s : s;
    }
}

// h[r] = silu(Wg[r] . xin) * (Wu[r] . xin), xin = RMS-normalized x * gamma.
kernel void gate_up1(device const half * Wg [[buffer(0)]], device const half * Wu [[buffer(1)]], device const float * x [[buffer(2)]],
                     device float * h [[buffer(3)]], device const float * gamma [[buffer(4)]], constant MatvecParams & p [[buffer(5)]],
                     threadgroup float4 * xs4 [[threadgroup(0)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kSimdgroupsPerGroup];
    const float scale = stage_input(x, gamma, p.K, true, p.eps, (threadgroup float *)xs4, red, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (gid * kSimdgroupsPerGroup + sg) * kGateRowsPerSimdgroup;
    if (row0 >= p.N) return;
    const uint K4 = p.K / 4;
    constexpr uint R = kGateRowsPerSimdgroup;
    float ag[R], au[R];
    for (uint i = 0; i < R; i++) { ag[i] = 0.f; au[i] = 0.f; }
    device const half4 * wg[R];
    device const half4 * wu[R];
    for (uint i = 0; i < R; i++) {
        const ulong r = min(row0 + i, p.N - 1);
        wg[i] = (device const half4 *)(Wg + r * p.K);
        wu[i] = (device const half4 *)(Wu + r * p.K);
    }
    for (uint k4 = lane; k4 < K4; k4 += 32) {
        const float4 xv = input4(xs4, x, gamma, true, k4);
        for (uint i = 0; i < R; i++) { ag[i] += dot(float4(wg[i][k4]), xv); au[i] += dot(float4(wu[i][k4]), xv); }
    }
    for (uint i = 0; i < R; i++) {
        const float a = simd_sum(ag[i]) * scale, b = simd_sum(au[i]) * scale;
        const uint r = row0 + i;
        if (lane == 0 && r < p.N) h[r] = a / (1.f + exp(-a)) * b;
    }
}

// ---- Several tokens (prefill): tiled matrix products on the simdgroup matrix units ----
// Templates on the tile (TM tokens x TN weight rows); 4 simdgroups in a 2 x 2 layout, each (TM/2) x (TN/2) as 8x8 blocks. Instances, picked by the host from
// the number of tokens (short prompts need many small tiles to fill the GPU, long ones reuse operands better in large tiles): matmul_16x16, matmul_32x32,
// matmul_64x32 and the gate_up equivalents.

constant constexpr uint kTileK = 32;  // reduction depth per step

struct GemmParams {
    uint K, N, M, x_stride, y_stride;
    uint flags;  // bit 0: input = RMS-normalized x * gamma; bit 1: add into y; bit 2: add bias[n]
    float eps;
};

// Per-token RMS scale of the tile's tokens (1 if not normalizing), each simdgroup taking every 4th token.
template <uint TM>
static void tile_scales(device const float * X, constant GemmParams & p, uint m0, bool norm, threadgroup float * scale, uint lane, uint sg) {
    for (uint r = sg; r < TM; r += kSimdgroupsPerGroup) {
        const uint m = m0 + r;
        float ss = 0.f;
        if (norm && m < p.M) for (uint k = lane; k < p.K; k += 32) { const float v = X[m * p.x_stride + k]; ss += v * v; }
        ss = simd_sum(ss);
        if (lane == 0) scale[r] = norm ? rsqrt(ss / float(p.K) + p.eps) : 1.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The A tile (tokens x reduction slice) as half: x, or x * gamma * scale.
template <uint TM>
static void stage_a(device const float * X, device const float * gamma, constant GemmParams & p, uint m0, uint k0, bool norm, threadgroup const float * scale,
                    threadgroup half4 * As4, uint tid) {
    constexpr uint K4 = kTileK / 4;
    for (uint idx = tid; idx < TM * K4; idx += kGroupThreads) {
        const uint r = idx / K4, k = k0 + (idx % K4) * 4, m = m0 + r;
        float4 v = 0.f;
        if (m < p.M && k < p.K) {
            v = *(device const float4 *)(X + m * p.x_stride + k);
            if (norm) v *= *(device const float4 *)(gamma + k) * scale[r];
        }
        As4[idx] = half4(v);
    }
}

template <uint TN>
static void stage_b(device const half * W, constant GemmParams & p, uint n0, uint k0, threadgroup half4 * Bs4, uint tid) {
    constexpr uint K4 = kTileK / 4;
    for (uint idx = tid; idx < TN * K4; idx += kGroupThreads) {
        const uint r = idx / K4, k = k0 + (idx % K4) * 4, n = n0 + r;
        Bs4[idx] = (n < p.N && k < p.K) ? *(device const half4 *)(W + (ulong)n * p.K + k) : half4(0.f);
    }
}

template <uint BM, uint BN>
static void tile_mma(threadgroup const half * As, threadgroup const half * Bs, uint sm, uint sn, thread simdgroup_float8x8 (&c)[BM][BN]) {
    for (uint kf = 0; kf < kTileK; kf += 8) {
        simdgroup_half8x8 a[BM], b[BN];
        for (uint i = 0; i < BM; i++) simdgroup_load(a[i], As + (sm + i * 8) * kTileK + kf, kTileK);
        for (uint j = 0; j < BN; j++) simdgroup_load(b[j], Bs + (sn + j * 8) * kTileK + kf, kTileK, ulong2(0, 0), true);
        for (uint i = 0; i < BM; i++) for (uint j = 0; j < BN; j++) simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
    }
}

// Y[m][n] (+)= sum_k Xin[m][k] * W[n][k] (+ bias[n]) for a TM x TN tile. Operand tiles are rounded to half in threadgroup memory and multiplied with f32
// accumulation (what llama.cpp's Metal matmul does). The RMS scales of the tile's tokens are computed here (no separate pass). K must be a multiple of 4.
template <uint TM, uint TN>
kernel void matmul(device const half * W [[buffer(0)]], device const float * X [[buffer(1)]], device float * Y [[buffer(2)]],
                   device const float * gamma [[buffer(3)]], constant GemmParams & p [[buffer(4)]], device const float * bias [[buffer(5)]],
                   uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = TM / 16, BN = TN / 16;
    threadgroup half4 As4[TM * kTileK / 4];
    threadgroup half4 Bs4[TN * kTileK / 4];
    threadgroup float Cs[TM * TN];
    threadgroup float scale[TM];
    const uint n0 = g.x * TN, m0 = g.y * TM;
    const uint sm = (sg / 2) * (TM / 2), sn = (sg % 2) * (TN / 2);
    const bool norm = (p.flags & 1u) != 0;
    tile_scales<TM>(X, p, m0, norm, scale, lane, sg);
    simdgroup_float8x8 c[BM][BN];
    for (uint i = 0; i < BM; i++) for (uint j = 0; j < BN; j++) c[i][j] = simdgroup_float8x8(0.f);
    for (uint k0 = 0; k0 < p.K; k0 += kTileK) {
        stage_a<TM>(X, gamma, p, m0, k0, norm, scale, As4, tid);
        stage_b<TN>(W, p, n0, k0, Bs4, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_mma<BM, BN>((threadgroup const half *)As4, (threadgroup const half *)Bs4, sm, sn, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < BM; i++) for (uint j = 0; j < BN; j++) simdgroup_store(c[i][j], Cs + (sm + i * 8) * TN + sn + j * 8, TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < TM * TN; idx += kGroupThreads) {
        const uint r = idx / TN, col = idx % TN, m = m0 + r, n = n0 + col;
        if (m >= p.M || n >= p.N) continue;
        const ulong o = (ulong)m * p.y_stride + n;
        const float v = Cs[idx] + ((p.flags & 4u) ? bias[n] : 0.f);
        Y[o] = (p.flags & 2u) ? Y[o] + v : v;
    }
}

// H[m][n] = silu(Xin . Wg[n]) * (Xin . Wu[n]), Xin = RMS-normalized x * gamma: the gate and up products share one staged A tile.
template <uint TM, uint TN>
kernel void matmul_gate_up(device const half * Wg [[buffer(0)]], device const half * Wu [[buffer(1)]], device const float * X [[buffer(2)]],
                           device float * H [[buffer(3)]], device const float * gamma [[buffer(4)]], constant GemmParams & p [[buffer(5)]],
                           uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = TM / 16, BN = TN / 16;
    threadgroup half4 As4[TM * kTileK / 4];
    threadgroup half4 Bg4[TN * kTileK / 4];
    threadgroup half4 Bu4[TN * kTileK / 4];
    threadgroup float Cg[TM * TN];
    threadgroup float Cu[TM * TN];
    threadgroup float scale[TM];
    const uint n0 = g.x * TN, m0 = g.y * TM;
    const uint sm = (sg / 2) * (TM / 2), sn = (sg % 2) * (TN / 2);
    tile_scales<TM>(X, p, m0, true, scale, lane, sg);
    simdgroup_float8x8 cg[BM][BN], cu[BM][BN];
    for (uint i = 0; i < BM; i++) for (uint j = 0; j < BN; j++) { cg[i][j] = simdgroup_float8x8(0.f); cu[i][j] = simdgroup_float8x8(0.f); }
    for (uint k0 = 0; k0 < p.K; k0 += kTileK) {
        stage_a<TM>(X, gamma, p, m0, k0, true, scale, As4, tid);
        stage_b<TN>(Wg, p, n0, k0, Bg4, tid);
        stage_b<TN>(Wu, p, n0, k0, Bu4, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_mma<BM, BN>((threadgroup const half *)As4, (threadgroup const half *)Bg4, sm, sn, cg);
        tile_mma<BM, BN>((threadgroup const half *)As4, (threadgroup const half *)Bu4, sm, sn, cu);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < BM; i++) for (uint j = 0; j < BN; j++) {
        simdgroup_store(cg[i][j], Cg + (sm + i * 8) * TN + sn + j * 8, TN);
        simdgroup_store(cu[i][j], Cu + (sm + i * 8) * TN + sn + j * 8, TN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < TM * TN; idx += kGroupThreads) {
        const uint r = idx / TN, col = idx % TN, m = m0 + r, n = n0 + col;
        if (m >= p.M || n >= p.N) continue;
        const float a = Cg[idx];
        H[(ulong)m * p.y_stride + n] = a / (1.f + exp(-a)) * Cu[idx];
    }
}

#define DOOMS_GEMM_INSTANCES(TM, TN) \
    template [[host_name("matmul_" #TM "x" #TN)]] kernel void matmul<TM, TN>(device const half *, device const float *, device float *, device const float *, \
        constant GemmParams &, device const float *, uint2, uint, uint, uint); \
    template [[host_name("matmul_gate_up_" #TM "x" #TN)]] kernel void matmul_gate_up<TM, TN>(device const half *, device const half *, device const float *, \
        device float *, device const float *, constant GemmParams &, uint2, uint, uint, uint);
DOOMS_GEMM_INSTANCES(16, 16)

// ---- Long prompts: 64 weight rows x 32 tokens per threadgroup ----
// The structure of llama.cpp's Metal kernel_mul_mm (ggml/src/ggml-metal/kernels/mul_mm.metal, MIT licence): operand tiles staged in threadgroup memory as
// 8x8 blocks so every simdgroup_load reads 64 contiguous halfs, each simdgroup computing a 16-token x 32-row block (8 accumulators: 2 token blocks x 4 row
// blocks) per reduction step, 6 KB of operands per threadgroup. Our additions: the RMS norm (per-token scale from rms_scale, gamma per column) applied while
// converting the activations to half, and epilogues (bias, residual add, SiLU(gate) x up) through threadgroup memory. K must be a multiple of 32.

constant constexpr uint kBigRows = 64, kBigTokens = 32, kBigK = 32;

// scale[t] = 1 / sqrt(mean(x[t]^2) + eps) for every token of the chunk. One threadgroup per token.
kernel void rms_scale(device const float * x [[buffer(0)]], device float * scale [[buffer(1)]], constant GemmParams & p [[buffer(2)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kSimdgroupsPerGroup];
    device const float * row = x + t * p.x_stride;
    float ss = 0.f;
    for (uint k = tid; k < p.K; k += kGroupThreads) ss += row[k] * row[k];
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.f;
        for (uint i = 0; i < kSimdgroupsPerGroup; i++) total += red[i];
        scale[t] = rsqrt(total / float(p.K) + p.eps);
    }
}

// The weight tile: each thread copies 16 consecutive values of one row into 8x8 blocks stored k-major (block (k-block, row-block), element [k][row]).
static void big_stage_w(device const half * W, constant GemmParams & p, uint n0, uint k0, threadgroup half * sa, ushort tid) {
    const ushort row = tid / 2, half_k = tid % 2;
    const uint n = min(n0 + row, p.N - 1);   // rows past the end repeat the last one; their outputs are never stored
    device const half4 * src = (device const half4 *)(W + (ulong)n * p.K + k0 + 16 * half_k);
    _Pragma("clang loop unroll(full)")
    for (ushort q = 0; q < 4; q++) {
        const half4 v = src[q];
        _Pragma("clang loop unroll(full)")
        for (ushort c = 0; c < 4; c++) {
            const ushort i = 4 * q + c;                          // 0..15 within this thread's slice
            const ushort kb = 2 * half_k + i / 8, rb = row / 8;  // k block 0..3, row block 0..7
            sa[64 * (8 * kb + rb) + 8 * (i % 8) + row % 8] = v[c];
        }
    }
}

// The activation tile: each thread converts 8 consecutive values of one token (normalized when asked) into blocks (k-block, token-block), element [token][k].
static void big_stage_x(device const float * X, device const float * gamma, device const float * scale, constant GemmParams & p, uint m0, uint k0, bool norm,
                        threadgroup half * sb, ushort tid) {
    const ushort tok = tid / 4, kb = tid % 4;
    const uint m = min(m0 + tok, p.M - 1);
    device const float4 * src = (device const float4 *)(X + (ulong)m * p.x_stride + k0 + 8 * kb);
    float4 lo = src[0], hi = src[1];
    if (norm) {
        device const float4 * g = (device const float4 *)(gamma + k0 + 8 * kb);
        const float s = scale[m];
        lo *= g[0] * s;
        hi *= g[1] * s;
    }
    threadgroup half4 * dst = (threadgroup half4 *)(sb + 64 * (4 * kb + tok / 8) + 8 * (tok % 8));
    dst[0] = half4(lo);
    dst[1] = half4(hi);
}

static void big_mma(threadgroup const half * sa, threadgroup const half * sb, ushort sg, thread simdgroup_float8x8 (&mc)[8]) {
    threadgroup const half * lsma = sa + 4 * 64 * (sg % 2);
    threadgroup const half * lsmb = sb + 2 * 64 * (sg / 2);
    // Fully unrolled, with scheduling barriers between the load and multiply phases, as llama.cpp's kernel_mul_mm does.
    _Pragma("clang loop unroll(full)")
    for (ushort ik = 0; ik < kBigK / 8; ik++) {
        simdgroup_half8x8 ma[4], mb[2];
        simdgroup_barrier(mem_flags::mem_none);
        _Pragma("clang loop unroll(full)")
        for (ushort i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
        simdgroup_barrier(mem_flags::mem_none);
        _Pragma("clang loop unroll(full)")
        for (ushort i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
        simdgroup_barrier(mem_flags::mem_none);
        _Pragma("clang loop unroll(full)")
        for (ushort i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
        lsma += 8 * 64;
        lsmb += 4 * 64;
    }
}

// Accumulators to threadgroup memory as [token][row] (stride kBigRows).
static void big_store_tile(thread simdgroup_float8x8 (&mc)[8], threadgroup float * tile, ushort sg) {
    threadgroup float * base = tile + 32 * (sg & 1) + 16 * (sg >> 1) * kBigRows;
    for (ushort i = 0; i < 8; i++) simdgroup_store(mc[i], base + 8 * (i % 4) + 8 * kBigRows * (i / 4), kBigRows, 0, false);
}

// Y[m][n] (+)= Xin[m] . W[n] (+ bias[n]); flags as GemmParams (bit 0 uses scale[] from rms_scale).
kernel void matmul_big(device const half * W [[buffer(0)]], device const float * X [[buffer(1)]], device float * Y [[buffer(2)]],
                       device const float * gamma [[buffer(3)]], constant GemmParams & p [[buffer(4)]], device const float * bias [[buffer(5)]],
                       device const float * scale [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[kBigRows * kBigTokens];   // 8 KB: the operand tiles (6 KB) during the loop, the output tile after it
    threadgroup half * sa = (threadgroup half *)buf;
    threadgroup half * sb = sa + kBigRows * kBigK;
    const uint n0 = g.y * kBigRows, m0 = g.x * kBigTokens;   // token tiles along x: neighbouring threadgroups share weight rows (L2 reuse)
    const bool norm = (p.flags & 1u) != 0;
    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.f);
    for (uint k0 = 0; k0 < p.K; k0 += kBigK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_stage_w(W, p, n0, k0, sa, tid);
        big_stage_x(X, gamma, scale, p, m0, k0, norm, sb, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_mma(sa, sb, sg, mc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    big_store_tile(mc, buf, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < kBigRows * kBigTokens; idx += kGroupThreads) {
        const uint t = idx / kBigRows, r = idx % kBigRows, m = m0 + t, n = n0 + r;
        if (m >= p.M || n >= p.N) continue;
        const ulong o = (ulong)m * p.y_stride + n;
        const float v = buf[idx] + ((p.flags & 4u) ? bias[n] : 0.f);
        Y[o] = (p.flags & 2u) ? Y[o] + v : v;
    }
}

// H[m][n] = silu(Xin . Wg[n]) * (Xin . Wu[n]), Xin = RMS-normalized x * gamma (scale from rms_scale).
kernel void matmul_gate_up_big(device const half * Wg [[buffer(0)]], device const half * Wu [[buffer(1)]], device const float * X [[buffer(2)]],
                               device float * H [[buffer(3)]], device const float * gamma [[buffer(4)]], constant GemmParams & p [[buffer(5)]],
                               device const float * scale [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[2 * kBigRows * kBigTokens];   // 16 KB: gate and up weight tiles + activations (10 KB) during the loop, both output tiles after it
    threadgroup half * sg_w = (threadgroup half *)buf;
    threadgroup half * su_w = sg_w + kBigRows * kBigK;
    threadgroup half * sb = su_w + kBigRows * kBigK;
    const uint n0 = g.y * kBigRows, m0 = g.x * kBigTokens;   // token tiles along x: neighbouring threadgroups share weight rows (L2 reuse)
    simdgroup_float8x8 cg[8], cu[8];
    for (ushort i = 0; i < 8; i++) { cg[i] = simdgroup_float8x8(0.f); cu[i] = simdgroup_float8x8(0.f); }
    for (uint k0 = 0; k0 < p.K; k0 += kBigK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_stage_w(Wg, p, n0, k0, sg_w, tid);
        big_stage_w(Wu, p, n0, k0, su_w, tid);
        big_stage_x(X, gamma, scale, p, m0, k0, true, sb, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_mma(sg_w, sb, sg, cg);
        big_mma(su_w, sb, sg, cu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    big_store_tile(cg, buf, sg);
    big_store_tile(cu, buf + kBigRows * kBigTokens, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < kBigRows * kBigTokens; idx += kGroupThreads) {
        const uint t = idx / kBigRows, r = idx % kBigRows, m = m0 + t, n = n0 + r;
        if (m >= p.M || n >= p.N) continue;
        const float a = buf[idx];
        H[(ulong)m * p.y_stride + n] = a / (1.f + exp(-a)) * buf[kBigRows * kBigTokens + idx];
    }
}

// ---- Attention ----

struct AttnParams {
    uint n_head, n_head_kv, head_dim;
    uint qkv_stride;  // floats between tokens in the qkv buffer (q | k | v)
    uint kv_stride;   // halfs between positions in a cache (n_head_kv * head_dim)
    uint out_stride;  // floats between tokens in the output
    uint p0;          // position of the first token
    uint fused_rope;  // 1: one token whose q, k, v are not rotated or cached yet (decode): rotate and cache them here; 0: rope_store ran (prefill)
    uint neox;        // rotary pair layout: 0 adjacent (2i, 2i+1), llama.cpp's "normal"; 1 split halves (i, i + head_dim/2), "neox" (qwen2);
                      // 2 none (positions come from a learned table, as in Whisper)
    float scale;      // 1 / sqrt(head_dim)
    float base;       // rotary frequency base
};

// The two elements of rotary pair i in a head.
static uint2 pair_index(uint i, uint hd, uint neox) { return neox == 1 ? uint2(i, i + hd / 2) : uint2(2 * i, 2 * i + 1); }

// Rotates the adjacent pair i of a head for position pos by pos * base^(-2i / head_dim).
static float2 rotate_pair(float2 v, uint i, uint hd, uint pos, float base, uint mode = 0) {
    if (mode == 2) return v;
    const float theta = float(pos) * pow(base, -2.f * float(i) / float(hd));
    const float c = cos(theta), s = sin(theta);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// One threadgroup per (head slot, token), head_dim / 2 threads (prefill only). Slots: query heads (rotated in place), key heads (rotated, stored), value heads.
kernel void rope_store(device float * qkv [[buffer(0)]], device half * kc [[buffer(1)]], device half * vc [[buffer(2)]], constant AttnParams & p [[buffer(3)]],
                       uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const uint slot = g.x, t = g.y, pos = p.p0 + t, hd = p.head_dim;
    const uint2 e = pair_index(i, hd, p.neox);
    device float * row = qkv + t * p.qkv_stride;
    if (slot < p.n_head) {
        device float * x = row + slot * hd;
        const float2 r = rotate_pair(float2(x[e.x], x[e.y]), i, hd, pos, p.base, p.neox);
        x[e.x] = r.x;
        x[e.y] = r.y;
    } else if (slot < p.n_head + p.n_head_kv) {
        const uint kvh = slot - p.n_head;
        device const float * x = row + (p.n_head + kvh) * hd;
        const float2 r = rotate_pair(float2(x[e.x], x[e.y]), i, hd, pos, p.base, p.neox);
        device half * dst = kc + (ulong)pos * p.kv_stride + kvh * hd;
        dst[e.x] = half(r.x);
        dst[e.y] = half(r.y);
    } else {
        const uint kvh = slot - p.n_head - p.n_head_kv;
        device const float * x = row + (p.n_head + p.n_head_kv + kvh) * hd;
        device half * dst = vc + (ulong)pos * p.kv_stride + kvh * hd;
        dst[e.x] = half(x[e.x]);
        dst[e.y] = half(x[e.y]);
    }
}

// Causal attention of one query head of one token against the cache (positions 0 .. p0 + t). One threadgroup per (head, token), kAttnThreads threads.
// Each thread scores whole positions (no cross-lane reduction per position); in the weighted sum over values each thread owns 4 dimensions of every
// (kAttnThreads / (head_dim / 4))-th position. With fused_rope (decode) the threadgroup rotates its query and its key/value head's new key itself, the new
// position's key and value come from threadgroup memory (rounded to half like the cache), and the first query head of each key/value head writes them to the
// cache: one kernel instead of rope_store + attention. head_dim is a power of two in [4, kMaxHeadDim] (the host checks).
kernel void attention(device const float * qkv [[buffer(0)]], device half * kc [[buffer(1)]], device half * vc [[buffer(2)]],
                      device float * out [[buffer(3)]], constant AttnParams & p [[buffer(4)]], threadgroup float * scores [[threadgroup(0)]],
                      uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kAttnSimdgroups];
    threadgroup float4 qs4[kMaxHeadDim / 4], kn4[kMaxHeadDim / 4], vn4[kMaxHeadDim / 4];
    threadgroup float4 partial4[kAttnThreads];
    const uint h = g.x, t = g.y, n = p.p0 + t + 1, hd = p.head_dim, hd4 = hd / 4, half_hd = hd / 2;
    const uint group = p.n_head / p.n_head_kv, kvh = h / group;
    const bool fused = p.fused_rope != 0;
    device const float * row = qkv + t * p.qkv_stride;
    threadgroup float * qs = (threadgroup float *)qs4;
    threadgroup float * kn = (threadgroup float *)kn4;
    threadgroup float * vn = (threadgroup float *)vn4;
    for (uint i = tid; i < half_hd; i += kAttnThreads) {
        const uint2 e = pair_index(i, hd, p.neox);
        device const float * q = row + h * hd;
        const float2 qr = fused ? rotate_pair(float2(q[e.x], q[e.y]), i, hd, n - 1, p.base, p.neox) : float2(q[e.x], q[e.y]);
        qs[e.x] = qr.x;
        qs[e.y] = qr.y;
        if (fused) {
            device const float * kx = row + (p.n_head + kvh) * hd;
            device const float * vx = row + (p.n_head + p.n_head_kv + kvh) * hd;
            const half2 k = half2(rotate_pair(float2(kx[e.x], kx[e.y]), i, hd, n - 1, p.base, p.neox));
            const half2 v = half2(vx[e.x], vx[e.y]);
            kn[e.x] = float(k.x); kn[e.y] = float(k.y);
            vn[e.x] = float(v.x); vn[e.y] = float(v.y);
            if (h % group == 0) {
                device half * kd = kc + (ulong)(n - 1) * p.kv_stride + kvh * hd;
                device half * vd = vc + (ulong)(n - 1) * p.kv_stride + kvh * hd;
                kd[e.x] = k.x; kd[e.y] = k.y;
                vd[e.x] = v.x; vd[e.y] = v.y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < n; j += kAttnThreads) {
        float s = 0.f;
        if (fused && j == n - 1) {
            for (uint i = 0; i < hd4; i++) s += dot(qs4[i], kn4[i]);
        } else {
            device const half4 * k = (device const half4 *)(kc + (ulong)j * p.kv_stride + kvh * hd);
            for (uint i = 0; i < hd4; i++) s += dot(qs4[i], float4(k[i]));
        }
        scores[j] = s * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint j = tid; j < n; j += kAttnThreads) mx = max(mx, scores[j]);
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint i = 1; i < kAttnSimdgroups; i++) mx = max(mx, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint j = tid; j < n; j += kAttnThreads) {
        const float e = exp(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.f;
    for (uint i = 0; i < kAttnSimdgroups; i++) sum += red[i];
    const uint groups = kAttnThreads / hd4, pg = tid / hd4, d4 = tid % hd4;
    float4 acc = 0.f;
    const uint cached = fused ? n - 1 : n;  // with fused_rope the new position is added from threadgroup memory below
    for (uint j = pg; j < cached; j += groups) acc += scores[j] * float4(((device const half4 *)(vc + (ulong)j * p.kv_stride + kvh * hd))[d4]);
    if (fused && pg == (n - 1) % groups) acc += scores[n - 1] * vn4[d4];
    partial4[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hd4) {
        float4 total = 0.f;
        for (uint i = 0; i < groups; i++) total += partial4[i * hd4 + tid];
        ((device float4 *)(out + t * p.out_stride + h * hd))[tid] = total / sum;
    }
}

// ---- Prefill attention: flash attention on the simdgroup matrix units ----
// One threadgroup per (query head, 32 queries); 4 simdgroups, 8 queries each. Keys and values stream through threadgroup memory 32 positions at a time:
// S = Q K^T (8 x 32 per simdgroup), then a running softmax per row (max, sum, rescale of the output by diag(exp(m_old - m_new))), then O += P V. Q, K, V and P are
// half, scores and the output accumulate in f32 (as llama.cpp's flash attention). Rows past the chunk and keys past the causal limit are zero/masked, so no
// uninitialized value can reach a product. HD (head_dim) 64 or 128.

constant constexpr uint kFlashQ = 32, kFlashK = 32;

template <uint HD>
kernel void attention_prefill(device const float * qkv [[buffer(0)]], device const half * kc [[buffer(1)]], device const half * vc [[buffer(2)]],
                              device float * out [[buffer(3)]], constant AttnParams & p [[buffer(4)]], constant uint & M [[buffer(5)]],
                              uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint DB = HD / 8;                       // 8-wide blocks along the head
    threadgroup half Qs[kFlashQ * HD];
    threadgroup half KV[2 * kFlashK * HD];            // keys then values; reused as the output staging area at the end
    threadgroup half * Ks = KV;
    threadgroup half * Vs = KV + kFlashK * HD;
    threadgroup float S[4][8 * kFlashK];
    threadgroup half P[4][8 * kFlashK];
    threadgroup float D[4][64];
    const uint h = g.x, t0 = g.y * kFlashQ;           // first query of the tile (index within the chunk)
    const uint kvh = h / (p.n_head / p.n_head_kv);
    const uint last_key = p.p0 + min(t0 + kFlashQ, M) - 1;   // the tile's last query attends up to here
    for (uint i = tid; i < kFlashQ * HD; i += kGroupThreads) {
        const uint t = t0 + i / HD, d = i % HD;
        Qs[i] = t < M ? half(qkv[t * p.qkv_stride + h * HD + d]) : half(0.f);
    }
    simdgroup_float8x8 O[DB];
    for (uint d = 0; d < DB; d++) O[d] = simdgroup_float8x8(0.f);
    // Row statistics of this lane's row (lanes 4r..4r+3 share row r).
    const ushort r = lane / 4, quarter = lane % 4;
    const uint qpos = p.p0 + t0 + 8 * sg + r;          // absolute position of the row's query
    float m_run = -INFINITY, l_run = 0.f;
    for (uint k0 = 0; k0 <= last_key; k0 += kFlashK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < kFlashK * HD; i += kGroupThreads) {
            const uint j = k0 + i / HD, d = i % HD;
            const bool ok = j <= last_key;
            Ks[i] = ok ? kc[(ulong)j * p.kv_stride + kvh * HD + d] : half(0.f);
            Vs[i] = ok ? vc[(ulong)j * p.kv_stride + kvh * HD + d] : half(0.f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q_sg (8 x HD) . K^T (HD x 32)
        for (uint kb = 0; kb < kFlashK / 8; kb++) {
            simdgroup_float8x8 acc(0.f);
            for (uint d = 0; d < DB; d++) {
                simdgroup_half8x8 qa, kb_t;
                simdgroup_load(qa, Qs + (8 * sg) * HD + 8 * d, HD);
                simdgroup_load(kb_t, Ks + (8 * kb) * HD + 8 * d, HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kb_t, acc);
            }
            simdgroup_store(acc, &S[sg][8 * kb], kFlashK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Running softmax of row r over this block's 32 keys: each of the 4 lanes takes 8 columns.
        float sv[8];
        float bmax = -INFINITY;
        for (uint c = 0; c < 8; c++) {
            const uint col = 8 * quarter + c, j = k0 + col;
            const float v = (j <= qpos) ? S[sg][r * kFlashK + col] * p.scale : -INFINITY;
            sv[c] = v;
            bmax = max(bmax, v);
        }
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 2));
        const float m_new = max(m_run, bmax);
        const float corr = (m_new == -INFINITY) ? 1.f : exp(m_run - m_new);
        float bsum = 0.f;
        for (uint c = 0; c < 8; c++) {
            const float e = (m_new == -INFINITY) ? 0.f : exp(sv[c] - m_new);
            bsum += e;
            P[sg][r * kFlashK + 8 * quarter + c] = half(e);
        }
        bsum += simd_shuffle_xor(bsum, 1);
        bsum += simd_shuffle_xor(bsum, 2);
        l_run = l_run * corr + bsum;
        m_run = m_new;
        // O = diag(corr) . O
        D[sg][2 * lane] = 0.f;
        D[sg][2 * lane + 1] = 0.f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (quarter == 0) D[sg][9 * r] = corr;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        simdgroup_load(dm, D[sg], 8);
        for (uint d = 0; d < DB; d++) { simdgroup_float8x8 t; simdgroup_multiply(t, dm, O[d]); O[d] = t; }
        // O += P (8 x 32) . V (32 x HD)
        for (uint kb = 0; kb < kFlashK / 8; kb++) {
            simdgroup_half8x8 pa;
            simdgroup_load(pa, P[sg] + 8 * kb, kFlashK);
            for (uint d = 0; d < DB; d++) {
                simdgroup_half8x8 vb;
                simdgroup_load(vb, Vs + (8 * kb) * HD + 8 * d, HD);
                simdgroup_multiply_accumulate(O[d], pa, vb, O[d]);
            }
        }
    }
    // O = diag(1 / l) . O, then the rows of the chunk to the output (via threadgroup memory: rows past the chunk are not written).
    simdgroup_barrier(mem_flags::mem_threadgroup);
    D[sg][2 * lane] = 0.f;
    D[sg][2 * lane + 1] = 0.f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (quarter == 0) D[sg][9 * r] = 1.f / l_run;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dl;
    simdgroup_load(dl, D[sg], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * ostage = (threadgroup float *)KV;   // 32 x HD floats = the 2 x 32 x HD halfs of KV
    for (uint d = 0; d < DB; d++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dl, O[d]);
        simdgroup_store(t, ostage + (8 * sg) * HD + 8 * d, HD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kFlashQ * HD; i += kGroupThreads) {
        const uint t = t0 + i / HD, d = i % HD;
        if (t < M) out[t * p.out_stride + h * HD + d] = ostage[i];
    }
}

template [[host_name("attention_prefill_64")]] kernel void attention_prefill<64>(device const float *, device const half *, device const half *, device float *,
    constant AttnParams &, constant uint &, uint2, ushort, ushort, ushort);
template [[host_name("attention_prefill_128")]] kernel void attention_prefill<128>(device const float *, device const half *, device const half *, device float *,
    constant AttnParams &, constant uint &, uint2, ushort, ushort, ushort);

// Float to half (the cross-attention keys and values of an audio window, computed once into a float scratch by matmul_big).
kernel void to_half(device const float * src [[buffer(0)]], device half * dst [[buffer(1)]], constant uint & n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = half(src[i]);
}

// ---- Whisper: LayerNorm once per site, several prompt tokens per weight read, cross-attention split over the audio positions ----

constant constexpr uint kMaxBatch = 8;   // tokens per matvecm call

// y[row] = LayerNorm(x[row]) * gamma + beta for row = threadgroup index (rows of K floats). One threadgroup of kGroupThreads per row, two passes (mean, variance).
kernel void layernorm(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], device const float * gamma [[buffer(2)]],
                      device const float * beta [[buffer(3)]], constant MatvecParams & p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kSimdgroupsPerGroup];
    device const float * xr = x + (ulong)row * p.K;
    device float * yr = y + (ulong)row * p.K;
    float s = 0.f;
    for (uint k = tid; k < p.K; k += kGroupThreads) s += xr[k];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.f;
    for (uint i = 0; i < kSimdgroupsPerGroup; i++) mean += red[i];
    mean /= float(p.K);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.f;
    for (uint k = tid; k < p.K; k += kGroupThreads) { const float d = xr[k] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.f;
    for (uint i = 0; i < kSimdgroupsPerGroup; i++) var += red[i];
    const float inv = rsqrt(var / float(p.K) + p.eps);
    for (uint k = tid; k < p.K; k += kGroupThreads) yr[k] = (xr[k] - mean) * inv * gamma[k] + beta[k];
}

// Y[m][r] (+)= W[r] . X[m] (+ bias[r], GELU) for m < M <= kMaxBatch: each weight row is read once for all M tokens (the prompt of a window). X rows (stride K)
// come from device memory; flags as matvec1 bits 1, 2 and 4 (no normalization: run layernorm first). kDecodeRowsPerSimdgroup rows per simdgroup.
kernel void matvecm(device const half * W [[buffer(0)]], device const float * X [[buffer(1)]], device float * Y [[buffer(2)]],
                    constant MatvecParams & p [[buffer(4)]], device const float * bias [[buffer(5)]], constant uint & M [[buffer(6)]],
                    uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint R = kDecodeRowsPerSimdgroup;
    const uint row0 = (gid * kSimdgroupsPerGroup + sg) * R;
    if (row0 >= p.N) return;
    const uint K4 = p.K / 4;
    float acc[R][kMaxBatch];
    for (uint i = 0; i < R; i++) for (uint m = 0; m < kMaxBatch; m++) acc[i][m] = 0.f;
    device const half4 * w[R];
    for (uint i = 0; i < R; i++) w[i] = (device const half4 *)(W + (ulong)min(row0 + i, p.N - 1) * p.K);
    for (uint k4 = lane; k4 < K4; k4 += 32) {
        float4 wv[R];
        for (uint i = 0; i < R; i++) wv[i] = float4(w[i][k4]);
        for (uint m = 0; m < kMaxBatch; m++) {
            if (m < M) {
                const float4 xv = ((device const float4 *)(X + (ulong)m * p.K))[k4];
                for (uint i = 0; i < R; i++) acc[i][m] += dot(wv[i], xv);
            }
        }
    }
    for (uint i = 0; i < R; i++) {
        const uint r = row0 + i;
        for (uint m = 0; m < kMaxBatch; m++) {
            if (m >= M) break;
            float s = simd_sum(acc[i][m]) + ((p.flags & 4u) && r < p.N ? bias[r] : 0.f);
            if (p.flags & 16u) s = gelu_erf(s);
            if (lane == 0 && r < p.N) {
                const ulong o = (ulong)m * p.N + r;
                Y[o] = (p.flags & 2u) ? Y[o] + s : s;
            }
        }
    }
}

struct SplitParams {
    uint n_head, head_dim;
    uint q_stride;    // floats between query tokens
    uint kv_stride;   // halfs between key (value) positions
    uint n_keys;      // positions attended (all of them: no causal mask)
    uint chunk;       // positions per split
    uint n_split;
    uint out_stride;  // floats between output tokens
    float scale;
};

// Attention without a causal mask over many keys (Whisper's cross-attention: 1500 audio positions), split over the positions so that every core has work:
// threadgroup (head, split, query token) scores its chunk, and writes (max, sum of exp, 2 unused, unnormalized value sum) to part (4 + head_dim floats per
// entry); attention_combine merges the splits. The query is pre-scaled here. head_dim 64 or 128 (the host checks).
kernel void attention_split(device const float * q [[buffer(0)]], device const half * kc [[buffer(1)]], device const half * vc [[buffer(2)]],
                            device float * part [[buffer(3)]], constant SplitParams & p [[buffer(4)]], threadgroup float * scores [[threadgroup(0)]],
                            uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kAttnSimdgroups];
    threadgroup float4 qs4[kMaxHeadDim / 4];
    threadgroup float4 partial4[kAttnThreads];
    const uint h = g.x, s = g.y, t = g.z, hd = p.head_dim, hd4 = hd / 4;
    const uint j0 = s * p.chunk, j1 = min(j0 + p.chunk, p.n_keys), n = j1 > j0 ? j1 - j0 : 0;
    device const float4 * qrow = (device const float4 *)(q + (ulong)t * p.q_stride + h * hd);
    for (uint i = tid; i < hd4; i += kAttnThreads) qs4[i] = qrow[i] * p.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint j = tid; j < n; j += kAttnThreads) {
        device const half4 * k = (device const half4 *)(kc + (ulong)(j0 + j) * p.kv_stride + h * hd);
        float sc = 0.f;
        for (uint i = 0; i < hd4; i++) sc += dot(qs4[i], float4(k[i]));
        scores[j] = sc;
        mx = max(mx, sc);
    }
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint i = 1; i < kAttnSimdgroups; i++) mx = max(mx, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint j = tid; j < n; j += kAttnThreads) {
        const float e = exp(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.f;
    for (uint i = 0; i < kAttnSimdgroups; i++) sum += red[i];
    const uint groups = kAttnThreads / hd4, pg = tid / hd4, d4 = tid % hd4;
    float4 acc = 0.f;
    for (uint j = pg; j < n; j += groups) acc += scores[j] * float4(((device const half4 *)(vc + (ulong)(j0 + j) * p.kv_stride + h * hd))[d4]);
    partial4[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * out = part + ((ulong)(t * p.n_head + h) * p.n_split + s) * (4 + hd);   // (max, sum, -, -, values): 16-byte aligned
    if (tid < hd4) {
        float4 total = 0.f;
        for (uint i = 0; i < groups; i++) total += partial4[i * hd4 + tid];
        ((device float4 *)(out + 4))[tid] = total;
    }
    if (tid == 0) { out[0] = mx; out[1] = sum; }
}

// out[t][h] = merge of the n_split partial results of (head h, token t): rescale each by exp(max_s - max), divide by the total sum. One threadgroup per
// (head, token), head_dim threads.
kernel void attention_combine(device const float * part [[buffer(0)]], device float * out [[buffer(1)]], constant SplitParams & p [[buffer(2)]],
                              uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const uint h = g.x, t = g.y, hd = p.head_dim;
    device const float * base = part + (ulong)(t * p.n_head + h) * p.n_split * (4 + hd);
    float mx = -INFINITY;
    for (uint s = 0; s < p.n_split; s++) if (base[s * (4 + hd) + 1] > 0.f) mx = max(mx, base[s * (4 + hd)]);
    float total = 0.f, acc = 0.f;
    for (uint s = 0; s < p.n_split; s++) {
        device const float * ps = base + s * (4 + hd);
        if (!(ps[1] > 0.f)) continue;   // an empty split
        const float w = exp(ps[0] - mx);
        total += ps[1] * w;
        acc += ps[4 + i] * w;
    }
    if (i < hd) out[(ulong)t * p.out_stride + h * hd + i] = acc / total;
}

// ---- Tokens ----

struct EmbedParams { uint n_embd; uint token_index; uint pos; uint use_pos; };

// x[row][:] = table[tokens[token_index + row]][:] (+ pos_table[pos + row][:] when use_pos: Whisper's learned positions). A decode step's input is the token
// the previous step's argmax wrote, read here without the host.
kernel void embed(device const half * table [[buffer(0)]], device const int * tokens [[buffer(1)]], device float * x [[buffer(2)]],
                  constant EmbedParams & p [[buffer(3)]], device const float * pos_table [[buffer(4)]], uint2 gp [[thread_position_in_grid]]) {
    const uint i = gp.x, row = gp.y;
    if (i >= p.n_embd) return;
    float v = float(table[(ulong)tokens[p.token_index + row] * p.n_embd + i]);
    if (p.use_pos) v += pos_table[(ulong)(p.pos + row) * p.n_embd + i];
    x[(ulong)row * p.n_embd + i] = v;
}

struct ArgmaxParams { uint n; uint out_index; uint use_mask; };

// tokens[out_index] = argmax(logits + mask) (the lowest index among equal maxima; mask: 0 or -inf per token, Whisper's suppressed tokens). One threadgroup.
kernel void argmax(device const float * logits [[buffer(0)]], device int * tokens [[buffer(1)]], constant ArgmaxParams & p [[buffer(2)]],
                   device const float * mask [[buffer(3)]],
                   uint tid [[thread_index_in_threadgroup]], uint nthreads [[threads_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    float best = -INFINITY;
    int idx = 0;
    for (uint i = tid; i < p.n; i += nthreads) {
        const float v = logits[i] + (p.use_mask ? mask[i] : 0.f);
        if (v > best) { best = v; idx = int(i); }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(best, off);
        const int oi = simd_shuffle_down(idx, off);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint nsg = (nthreads + 31) / 32;
        float b = bv[0];
        int k = bi[0];
        for (uint s = 1; s < nsg; s++) if (bv[s] > b || (bv[s] == b && bi[s] < k)) { b = bv[s]; k = bi[s]; }
        tokens[p.out_index] = k;
    }
}
)MSL";
}

}  // namespace dooms
