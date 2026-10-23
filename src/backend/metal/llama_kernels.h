#pragma once
// Metal Shading Language source of the Llama decoder kernels (metal_llama_decoder.mm), compiled when a decoder is created. Parameter structs mirror the C++ ones
// in metal_llama_decoder.mm field for field (all 32-bit); the host checks their sizes. Constants named k... below must equal the host's constants of the same
// names. Measurements behind the shapes: src/backend/metal/CLAUDE.md "Llama decoder".

#include <string>

namespace dooms {

// Two literals joined once (one literal may not exceed 65536 characters): the f16 kernels, then the quantized-format kernels.
inline const char * llama_kernel_source() {
    static const std::string source = std::string(R"MSL(
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
// Split into the device load (issued before the barrier that waits for the previous step's products, so the loads overlap it, as llama.cpp's kernel
// does) and the threadgroup store.
static void big_load_w(device const half * W, constant GemmParams & p, uint n0, uint k0, ushort tid, thread half4 (&v)[4]) {
    const ushort row = tid / 2, half_k = tid % 2;
    const uint n = min(n0 + row, p.N - 1);   // rows past the end repeat the last one; their outputs are never stored
    device const half4 * src = (device const half4 *)(W + (ulong)n * p.K + k0 + 16 * half_k);
    _Pragma("clang loop unroll(full)")
    for (ushort q = 0; q < 4; q++) v[q] = src[q];
}

static void big_put_w(thread const half4 (&v)[4], threadgroup half * sa, ushort tid) {
    const ushort row = tid / 2, half_k = tid % 2;
    _Pragma("clang loop unroll(full)")
    for (ushort q = 0; q < 4; q++) {
        _Pragma("clang loop unroll(full)")
        for (ushort c = 0; c < 4; c++) {
            const ushort i = 4 * q + c;                          // 0..15 within this thread's slice
            const ushort kb = 2 * half_k + i / 8, rb = row / 8;  // k block 0..3, row block 0..7
            sa[64 * (8 * kb + rb) + 8 * (i % 8) + row % 8] = v[q][c];
        }
    }
}

static void big_stage_w(device const half * W, constant GemmParams & p, uint n0, uint k0, threadgroup half * sa, ushort tid) {
    half4 v[4];
    big_load_w(W, p, n0, k0, tid, v);
    big_put_w(v, sa, tid);
}

// The activation tile: each thread converts 8 consecutive values of one token (normalized when asked) into blocks (k-block, token-block), element [token][k].
static void big_load_x(device const float * X, device const float * gamma, device const float * scale, constant GemmParams & p, uint m0, uint k0, bool norm,
                       ushort tid, thread half4 (&v)[2]) {
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
    v[0] = half4(lo);
    v[1] = half4(hi);
}

static void big_put_x(thread const half4 (&v)[2], threadgroup half * sb, ushort tid) {
    const ushort tok = tid / 4, kb = tid % 4;
    threadgroup half4 * dst = (threadgroup half4 *)(sb + 64 * (4 * kb + tok / 8) + 8 * (tok % 8));
    dst[0] = v[0];
    dst[1] = v[1];
}

static void big_stage_x(device const float * X, device const float * gamma, device const float * scale, constant GemmParams & p, uint m0, uint k0, bool norm,
                        threadgroup half * sb, ushort tid) {
    half4 v[2];
    big_load_x(X, gamma, scale, p, m0, k0, norm, tid, v);
    big_put_x(v, sb, tid);
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
        half4 wv[4], xv[2];
        big_load_w(W, p, n0, k0, tid, wv);
        big_load_x(X, gamma, scale, p, m0, k0, norm, tid, xv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_put_w(wv, sa, tid);
        big_put_x(xv, sb, tid);
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
    // 10 KB: gate and up weight tiles + activations during the loop; after it, one 8 KB output tile at a time (gate, then up), so more threadgroups fit
    // on a core than with both output tiles (16 KB).
    threadgroup float buf[(2 * kBigRows * kBigK + kBigTokens * kBigK) / 2];
    threadgroup half * sg_w = (threadgroup half *)buf;
    threadgroup half * su_w = sg_w + kBigRows * kBigK;
    threadgroup half * sb = su_w + kBigRows * kBigK;
    const uint n0 = g.y * kBigRows, m0 = g.x * kBigTokens;   // token tiles along x: neighbouring threadgroups share weight rows (L2 reuse)
    simdgroup_float8x8 cg[8], cu[8];
    for (ushort i = 0; i < 8; i++) { cg[i] = simdgroup_float8x8(0.f); cu[i] = simdgroup_float8x8(0.f); }
    for (uint k0 = 0; k0 < p.K; k0 += kBigK) {
        half4 gv[4], uv[4], xv[2];
        big_load_w(Wg, p, n0, k0, tid, gv);
        big_load_w(Wu, p, n0, k0, tid, uv);
        big_load_x(X, gamma, scale, p, m0, k0, true, tid, xv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_put_w(gv, sg_w, tid);
        big_put_w(uv, su_w, tid);
        big_put_x(xv, sb, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_mma(sg_w, sb, sg, cg);
        big_mma(su_w, sb, sg, cu);
    }
    // SiLU(gate) of this thread's 16 outputs kept in registers, then the up tile in the same threadgroup memory
    constexpr uint kPer = kBigRows * kBigTokens / kGroupThreads;
    float act[kPer];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    big_store_tile(cg, buf, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kPer; j++) { const float a = buf[tid + j * kGroupThreads]; act[j] = a / (1.f + exp(-a)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    big_store_tile(cu, buf, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kPer; j++) {
        const uint idx = tid + j * kGroupThreads, t = idx / kBigRows, r = idx % kBigRows, m = m0 + t, n = n0 + r;
        if (m < p.M && n < p.N) H[(ulong)m * p.y_stride + n] = act[j] * buf[idx];
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
    // keys then values; first the query tile (loaded into registers once), and the output staging area at the end. Without a separate query buffer
    // two threadgroups fit on a core.
    threadgroup half KV[2 * kFlashK * HD];
    threadgroup half * Qs = KV;
    threadgroup half * Ks = KV;
    threadgroup half * Vs = KV + kFlashK * HD;
    threadgroup float S[4][8 * kFlashK];
    threadgroup half P[4][8 * kFlashK];
    threadgroup float D[4][64];
    const uint h = g.x, t0 = g.y * kFlashQ;           // first query of the tile (index within the chunk)
    const uint kvh = h / (p.n_head / p.n_head_kv);
    const uint last_key = p.p0 + min(t0 + kFlashQ, M) - 1;   // the tile's last query attends up to here
    for (uint i = tid; i < kFlashQ * HD / 4; i += kGroupThreads) {
        const uint t = t0 + (4 * i) / HD, d = (4 * i) % HD;
        ((threadgroup half4 *)Qs)[i] = t < M ? half4(*(device const float4 *)(qkv + t * p.qkv_stride + h * HD + d)) : half4(0.f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[DB];   // this simdgroup's 8 queries, for every key block
    for (uint d = 0; d < DB; d++) simdgroup_load(qm[d], Qs + (8 * sg) * HD + 8 * d, HD);
    simdgroup_float8x8 O[DB];
    for (uint d = 0; d < DB; d++) O[d] = simdgroup_float8x8(0.f);
    // Row statistics of this lane's row (lanes 4r..4r+3 share row r).
    const ushort r = lane / 4, quarter = lane % 4;
    const uint qpos = p.p0 + t0 + 8 * sg + r;          // absolute position of the row's query
    float m_run = -INFINITY, l_run = 0.f;
    for (uint k0 = 0; k0 <= last_key; k0 += kFlashK) {
        // the block's keys and values, 4 halfs per load, fetched before the barrier (the previous block's products may still read the tile)
        constexpr uint KV4 = kFlashK * HD / 4 / kGroupThreads;
        half4 kr[KV4], vr[KV4];
        for (uint q = 0; q < KV4; q++) {
            const uint i = tid + q * kGroupThreads, j = k0 + (4 * i) / HD, d = (4 * i) % HD;
            const bool ok = j <= last_key;
            const ulong o = (ulong)j * p.kv_stride + kvh * HD + d;
            kr[q] = ok ? *(device const half4 *)(kc + o) : half4(0.h);
            vr[q] = ok ? *(device const half4 *)(vc + o) : half4(0.h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0; q < KV4; q++) {
            const uint i = tid + q * kGroupThreads;
            ((threadgroup half4 *)Ks)[i] = kr[q];
            ((threadgroup half4 *)Vs)[i] = vr[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q_sg (8 x HD) . K^T (HD x 32)
        for (uint kb = 0; kb < kFlashK / 8; kb++) {
            simdgroup_float8x8 acc(0.f);
            for (uint d = 0; d < DB; d++) {
                simdgroup_half8x8 kb_t;
                simdgroup_load(kb_t, Ks + (8 * kb) * HD + 8 * d, HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qm[d], kb_t, acc);
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
    uint group;       // query heads per key/value head (grouped-query attention; 1: one key/value head per query head)
};

// Attention without a causal mask over many keys (Whisper's cross-attention: 1500 audio positions; a long decode step's self-attention over the cache, whose
// positions are all visible), split over the positions so that every core has work:
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
    const uint kvh = h / p.group;
    for (uint j = tid; j < n; j += kAttnThreads) {
        device const half4 * k = (device const half4 *)(kc + (ulong)(j0 + j) * p.kv_stride + kvh * hd);
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
    for (uint j = pg; j < n; j += groups) acc += scores[j] * float4(((device const half4 *)(vc + (ulong)(j0 + j) * p.kv_stride + kvh * hd))[d4]);
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
)MSL") + R"MSL(
// ---- Quantized weights: ggml's block formats, read in place (one token: dot products per 8-element group; prefill: the tiles unpacked as they are staged) ----
// Layouts (ggml-common.h): q4_0 {half d; uchar qs[16]} 18 B / 32; q5_0 {half d; uchar qh[4]; uchar qs[16]} 22 B / 32; q8_0 {half d; char qs[32]} 34 B / 32;
// q4_K {half d, dmin; uchar scales[12]; uchar qs[128]} 144 B / 256; q6_K {uchar ql[128]; uchar qh[64]; char scales[16]; half d} 210 B / 256.
// Blocks are only 2-byte aligned (rows follow each other unpadded), so the reads are bytes and halves. The values equal ggml's dequantize_row_* exactly.

static float ld_half(device const uchar * p) { return float(*(device const half *)p); }

// The quantized one-token kernels split each row into groups of 8 elements, one group per lane per step (an 896-wide row is 3.5 steps, a 4864-wide one 19),
// grouped by each format's byte layout so a lane's bytes are contiguous and every byte it loads is used whole. The input values of a group are loaded once
// and shared by the rows a simdgroup computes, with the sums that fold the formats' offsets (q - 8, q - 16, the q4_K mins, q - 32) out of the per-element work.
struct XGroup {
    float4 a, b;   // the group's 8 input values in the format's order (below)
    float4 s;      // their partial sums the format needs
};

// The input as the projection sees it: 4 or 2 consecutive values from element k.
static float4 xin4(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint k) {
    return input4(xs4, x, gamma, norm, k / 4);
}
static float2 xin2(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint k) {
    if (kStageInput) { threadgroup const float * xs = (threadgroup const float *)xs4; return float2(xs[k], xs[k + 1]); }
    const float2 v = float2(x[k], x[k + 1]);
    return norm ? v * float2(gamma[k], gamma[k + 1]) : v;
}

static float hsum(float4 v) { return v.x + v.y + v.z + v.w; }
// Four bytes of a word as floats (the unpacking is done on whole words: nibbles with 0x0F0F0F0F masks, then one conversion).
static float4 bytes4(uint u) { return float4(as_type<uchar4>(u)); }

// Bit i of a 4-bit value moved to bit 4 of byte i: q5_0's fifth bits for four elements at once (shifts and a mask: a table indexed differently per lane
// serializes on the constant cache).
static uint spread4(uint n) { return ((n | (n << 7) | (n << 14) | (n << 21)) & 0x01010101u) << 4; }

// Per format: elem (embedding lookup), row16 (the prefill matmuls' weight tiles), load (group g's input values and sums) and dot (one row's group g).
// q4_0 / q5_0: group g = block g/4, bytes 4q..4q+3 (q = g%4) of qs: low nibbles are elements 4q.. (a), high nibbles 16+4q.. (b).
struct Q4_0 {
    enum { BLK = 32, BYTES = 18 };
    static float elem(device const uchar * row, uint i) {
        device const uchar * b = row + (i / 32) * 18;
        const uint j = i % 32, q = b[2 + (j & 15)];
        return ld_half(b) * (float(j < 16 ? (q & 15u) : (q >> 4)) - 8.f);
    }
    static void load(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint g, thread XGroup & xg) {
        const uint k = (g / 4) * 32 + (g % 4) * 4;
        xg.a = xin4(xs4, x, gamma, norm, k);
        xg.b = xin4(xs4, x, gamma, norm, k + 16);
        xg.s.x = hsum(xg.a) + hsum(xg.b);
    }
    static uint qword(device const uchar * p) { device const ushort * q = (device const ushort *)p; return uint(q[0]) | (uint(q[1]) << 16); }
    // The 16 weights of the row at columns c..c+15 (c a multiple of 16) as half, for the prefill matmuls' tile loads (the values elem gives).
    static void row16(device const uchar * row, uint c, thread half4 (&o)[4]) {
        device const uchar * b = row + (c / 32) * 18;
        const float d = ld_half(b);
        const bool hi = (c % 32) >= 16;
        for (uint i = 0; i < 4; i++) {
            const uint u = qword(b + 2 + 4 * i);
            o[i] = half4(d * (bytes4(hi ? (u >> 4) & 0x0F0F0F0Fu : u & 0x0F0F0F0Fu) - 8.f));
        }
    }
    static float dot(device const uchar * row, uint g, thread const XGroup & x) {
        device const uchar * b = row + (g / 4) * 18;
        const uint u = qword(b + 2 + (g % 4) * 4);
        return ld_half(b) * (metal::dot(bytes4(u & 0x0F0F0F0Fu), x.a) + metal::dot(bytes4((u >> 4) & 0x0F0F0F0Fu), x.b) - 8.f * x.s.x);
    }
};

struct Q5_0 {
    enum { BLK = 32, BYTES = 22 };
    static uint high_bits(device const uchar * b) { return Q4_0::qword(b + 2); }
    static float elem(device const uchar * row, uint i) {
        device const uchar * b = row + (i / 32) * 22;
        const uint j = i % 32, q = b[6 + (j & 15)], low = j < 16 ? (q & 15u) : (q >> 4);
        return ld_half(b) * (float(low | (((high_bits(b) >> j) & 1u) << 4)) - 16.f);
    }
    static void load(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint g, thread XGroup & xg) {
        Q4_0::load(xs4, x, gamma, norm, g, xg);   // the same element layout
    }
    static void row16(device const uchar * row, uint c, thread half4 (&o)[4]) {
        device const uchar * b = row + (c / 32) * 22;
        const float d = ld_half(b);
        const uint qh = high_bits(b);
        const bool hi = (c % 32) >= 16;
        for (uint i = 0; i < 4; i++) {
            const uint u = Q4_0::qword(b + 6 + 4 * i), j0 = 4 * i;
            const uint w = hi ? ((u >> 4) & 0x0F0F0F0Fu) | spread4((qh >> (j0 + 16)) & 15u) : (u & 0x0F0F0F0Fu) | spread4((qh >> j0) & 15u);
            o[i] = half4(d * (bytes4(w) - 16.f));
        }
    }
    static float dot(device const uchar * row, uint g, thread const XGroup & x) {
        device const uchar * b = row + (g / 4) * 22;
        const uint j0 = (g % 4) * 4, qh = high_bits(b), u = Q4_0::qword(b + 6 + j0);
        const uint lo = (u & 0x0F0F0F0Fu) | spread4((qh >> j0) & 15u), hi = ((u >> 4) & 0x0F0F0F0Fu) | spread4((qh >> (j0 + 16)) & 15u);
        return ld_half(b) * (metal::dot(bytes4(lo), x.a) + metal::dot(bytes4(hi), x.b) - 16.f * x.s.x);
    }
};

// q8_0: group g = block g/4, elements 8q..8q+7 (q = g%4).
struct Q8_0 {
    enum { BLK = 32, BYTES = 34 };
    static float elem(device const uchar * row, uint i) {
        device const uchar * b = row + (i / 32) * 34;
        return ld_half(b) * float(((device const char *)b)[2 + i % 32]);
    }
    static void load(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint g, thread XGroup & xg) {
        xg.a = xin4(xs4, x, gamma, norm, g * 8);
        xg.b = xin4(xs4, x, gamma, norm, g * 8 + 4);
    }
    static void row16(device const uchar * row, uint c, thread half4 (&o)[4]) {
        device const uchar * b = row + (c / 32) * 34;
        const float d = ld_half(b);
        device const uchar * q = b + 2 + (c % 32);
        for (uint i = 0; i < 4; i++) o[i] = half4(d * float4(as_type<char4>(Q4_0::qword(q + 4 * i))));
    }
    static float dot(device const uchar * row, uint g, thread const XGroup & x) {
        device const uchar * b = row + (g / 4) * 34;
        device const uchar * q = b + 2 + (g % 4) * 8;
        const float4 v0 = float4(as_type<char4>(Q4_0::qword(q))), v1 = float4(as_type<char4>(Q4_0::qword(q + 4)));
        return ld_half(b) * (metal::dot(v0, x.a) + metal::dot(v1, x.b));
    }
};

// q4_K's 6-bit scale and min of sub-block j (ggml's get_scale_min_k4).
static void scale_min_k4(uint j, device const uchar * q, thread float & d, thread float & m) {
    if (j < 4) { d = float(q[j] & 63u); m = float(q[j + 4] & 63u); }
    else { d = float((q[j + 4] & 15u) | ((q[j - 4] >> 6) << 4)); m = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

// q4_K: group g = block g/32, t = g%32: 4 bytes at qs[32p + l] (p = t/8, l = 4(t%8)); low nibbles are elements 64p+l.. (sub-block 2p, a), high ones
// 64p+32+l.. (sub-block 2p+1, b). The 144-byte blocks are 16-byte aligned: one 4-byte load.
struct Q4_K {
    enum { BLK = 256, BYTES = 144 };
    static float elem(device const uchar * row, uint i) {
        device const uchar * b = row + (i / 256) * 144;
        const uint j = i % 256, c = j / 32, l = j % 32;
        float sc, m;
        scale_min_k4(c, b + 4, sc, m);
        const uint q = b[16 + 32 * (c / 2) + l];
        return ld_half(b) * sc * float((c & 1u) ? (q >> 4) : (q & 15u)) - ld_half(b + 2) * m;
    }
    static void load(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint g, thread XGroup & xg) {
        const uint t = g % 32, k = (g / 32) * 256 + 64 * (t / 8) + 4 * (t % 8);
        xg.a = xin4(xs4, x, gamma, norm, k);
        xg.b = xin4(xs4, x, gamma, norm, k + 32);
        xg.s.x = hsum(xg.a);
        xg.s.y = hsum(xg.b);
    }
    static void row16(device const uchar * row, uint c, thread half4 (&o)[4]) {
        device const uchar * b = row + (c / 256) * 144;
        const uint j0 = c % 256, sub = j0 / 32, l0 = j0 % 32, shift = (sub & 1u) * 4;
        float sc, m;
        scale_min_k4(sub, b + 4, sc, m);
        const float d = ld_half(b), dmin = ld_half(b + 2);
        const uint4 w = *(device const uint4 *)(b + 16 + 32 * (sub / 2) + l0);   // 16-byte aligned (144-byte blocks)
        for (uint i = 0; i < 4; i++) o[i] = half4(d * sc * bytes4((w[i] >> shift) & 0x0F0F0F0Fu) - dmin * m);
    }
    static float dot(device const uchar * row, uint g, thread const XGroup & x) {
        device const uchar * b = row + (g / 32) * 144;
        const uint t = g % 32, p = t / 8;
        const uint u = *(device const uint *)(b + 16 + 32 * p + 4 * (t % 8));
        float sc0, m0, sc1, m1;
        scale_min_k4(2 * p, b + 4, sc0, m0);
        scale_min_k4(2 * p + 1, b + 4, sc1, m1);
        return ld_half(b) * (sc0 * metal::dot(bytes4(u & 0x0F0F0F0Fu), x.a) + sc1 * metal::dot(bytes4((u >> 4) & 0x0F0F0F0Fu), x.b)) - ld_half(b + 2) * (m0 * x.s.x + m1 * x.s.y);
    }
};

// q6_K: group g = block g/32, t = g%32: half h = t/16, positions l, l+1 (l = 2(t%16)) of its four 32-element chunks k: elements 128h + 32k + l + {0, 1}
// (a: chunks 0 and 1, b: chunks 2 and 3); two bytes each of ql[64h + l], ql[64h + 32 + l] and qh[32h + l] give all 8.
struct Q6_K {
    enum { BLK = 256, BYTES = 210 };
    static float elem(device const uchar * row, uint i) {
        device const uchar * b = row + (i / 256) * 210;
        const uint j = i % 256, h = j / 128, k = (j % 128) / 32, l = j % 32;
        const uint qlb = b[64 * h + (k & 1u) * 32 + l], low = k < 2 ? (qlb & 15u) : (qlb >> 4);
        const uint hb = (b[128 + 32 * h + l] >> (2 * k)) & 3u;
        const float sc = float(((device const char *)(b + 192))[8 * h + l / 16 + 2 * k]);
        return ld_half(b + 208) * sc * float(int(low | (hb << 4)) - 32);
    }
    static void load(threadgroup const float4 * xs4, device const float * x, device const float * gamma, bool norm, uint g, thread XGroup & xg) {
        const uint t = g % 32, k0 = (g / 32) * 256 + 128 * (t / 16) + 2 * (t % 16);
        const float2 v0 = xin2(xs4, x, gamma, norm, k0), v1 = xin2(xs4, x, gamma, norm, k0 + 32), v2 = xin2(xs4, x, gamma, norm, k0 + 64),
                     v3 = xin2(xs4, x, gamma, norm, k0 + 96);
        xg.a = float4(v0, v1);
        xg.b = float4(v2, v3);
        xg.s = float4(v0.x + v0.y, v1.x + v1.y, v2.x + v2.y, v3.x + v3.y);
    }
    static void row16(device const uchar * row, uint c, thread half4 (&o)[4]) {
        device const uchar * b = row + (c / 256) * 210;
        const uint j0 = c % 256, h = j0 / 128, k = (j0 % 128) / 32, l0 = j0 % 32;
        device const uchar * ql = b + 64 * h + (k & 1u) * 32 + l0;
        device const uchar * qh = b + 128 + 32 * h + l0;
        const float d = ld_half(b + 208), sc = float(((device const char *)(b + 192))[8 * h + l0 / 16 + 2 * k]);
        for (uint i = 0; i < 4; i++) {
            const uint lo = Q4_0::qword(ql + 4 * i), hb = Q4_0::qword(qh + 4 * i);
            const uint w = ((k < 2 ? lo : lo >> 4) & 0x0F0F0F0Fu) | (((hb >> (2 * k)) & 0x03030303u) << 4);
            o[i] = half4(d * sc * (bytes4(w) - 32.f));
        }
    }
    static float dot(device const uchar * row, uint g, thread const XGroup & x) {
        device const uchar * b = row + (g / 32) * 210;
        const uint t = g % 32, h = t / 16, l = 2 * (t % 16);
        const uint q0 = *(device const ushort *)(b + 64 * h + l), q1 = *(device const ushort *)(b + 64 * h + 32 + l), qh = *(device const ushort *)(b + 128 + 32 * h + l);
        // bytes (chunk 0 e0, e1, chunk 1 e0, e1) and (chunk 2 ..., chunk 3 ...): 4 low bits from ql, 2 high bits from qh
        const uint w01 = ((q0 & 0x0F0Fu) | ((q1 & 0x0F0Fu) << 16)) | (((qh & 0x0303u) | (((qh >> 2) & 0x0303u) << 16)) << 4);
        const uint w23 = (((q0 >> 4) & 0x0F0Fu) | (((q1 >> 4) & 0x0F0Fu) << 16)) | ((((qh >> 4) & 0x0303u) | (((qh >> 6) & 0x0303u) << 16)) << 4);
        const float4 p01 = bytes4(w01) * x.a, p23 = bytes4(w23) * x.b;
        device const char * sc = (device const char *)(b + 192) + 8 * h + l / 16;
        return ld_half(b + 208) * (float(sc[0]) * (p01.x + p01.y - 32.f * x.s.x) + float(sc[2]) * (p01.z + p01.w - 32.f * x.s.y) +
                                   float(sc[4]) * (p23.x + p23.y - 32.f * x.s.z) + float(sc[6]) * (p23.z + p23.w - 32.f * x.s.w));
    }
};

// Rows a simdgroup computes in the quantized kernels (the host's constants of the same names): more than the f16 kernels, because each threadgroup's input
// staging (and RMS) is a fixed cost and the quantized rows are cheap to read.
constant constexpr uint kQuantRows = 2, kQuantGateRows = 2;

// matvec1 / gate_up1 on quantized rows. The input comes normalized already (norm_vec) and is read from device memory (the kStageInput = false pipelines the
// host uses for every quantized matrix): no staging and no barrier before the products. Each row's 8-element groups are split over S simdgroups (S = 1, 2
// or 4 in flags bits 8..10, picked by the host from the row length so a lane gets several groups), which add their partial sums; a threadgroup computes
// kQuantRows rows (gate_up: kQuantGateRows gate/up pairs) per S simdgroups.
template <typename Q>
kernel void matvec1q(device const uchar * W [[buffer(0)]], device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                     device const float * gamma [[buffer(3)]], constant MatvecParams & p [[buffer(4)]], device const float * bias [[buffer(5)]],
                     device const float * beta [[buffer(6)]], threadgroup float4 * xs4 [[threadgroup(0)]],
                     uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[kSimdgroupsPerGroup][kQuantRows];
    const bool accumulate = (p.flags & 2u) != 0;
    const uint S = (p.flags >> 8) & 7u, slots = kSimdgroupsPerGroup / S, slot = sg / S, piece = sg % S;
    const uint row0 = (gid * slots + slot) * kQuantRows;
    const ulong RB = ulong(p.K / Q::BLK) * Q::BYTES;
    const uint NG = p.K / 8;
    float acc[kQuantRows];
    device const uchar * w[kQuantRows];
    for (uint i = 0; i < kQuantRows; i++) { acc[i] = 0.f; w[i] = W + (ulong)min(row0 + i, p.N - 1) * RB; }
    if (row0 < p.N) {
        XGroup xg;
        for (uint g = piece * 32 + lane; g < NG; g += S * 32) {
            Q::load(xs4, x, gamma, false, g, xg);
            for (uint i = 0; i < kQuantRows; i++) acc[i] += Q::dot(w[i], g, xg);
        }
    }
    for (uint i = 0; i < kQuantRows; i++) { const float s = simd_sum(acc[i]); if (lane == 0) part[sg][i] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < slots * kQuantRows) {
        const uint sl = tid / kQuantRows, i = tid % kQuantRows, r = (gid * slots + sl) * kQuantRows + i;
        if (r < p.N) {
            float s = 0.f;
            for (uint k = 0; k < S; k++) s += part[sl * S + k][i];
            s += (p.flags & 4u) ? bias[r] : 0.f;
            y[r] = accumulate ? y[r] + s : s;
        }
    }
}

template <typename Q>
kernel void gate_up1q(device const uchar * Wg [[buffer(0)]], device const uchar * Wu [[buffer(1)]], device const float * x [[buffer(2)]],
                      device float * h [[buffer(3)]], device const float * gamma [[buffer(4)]], constant MatvecParams & p [[buffer(5)]],
                      threadgroup float4 * xs4 [[threadgroup(0)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[kSimdgroupsPerGroup][2 * kQuantGateRows];
    const uint S = (p.flags >> 8) & 7u, slots = kSimdgroupsPerGroup / S, slot = sg / S, piece = sg % S;
    const uint row0 = (gid * slots + slot) * kQuantGateRows;
    const ulong RB = ulong(p.K / Q::BLK) * Q::BYTES;
    const uint NG = p.K / 8;
    float ag[kQuantGateRows], au[kQuantGateRows];
    device const uchar * wg[kQuantGateRows];
    device const uchar * wu[kQuantGateRows];
    for (uint i = 0; i < kQuantGateRows; i++) {
        const ulong r = min(row0 + i, p.N - 1);
        ag[i] = 0.f; au[i] = 0.f; wg[i] = Wg + r * RB; wu[i] = Wu + r * RB;
    }
    if (row0 < p.N) {
        XGroup xg;
        for (uint g = piece * 32 + lane; g < NG; g += S * 32) {
            Q::load(xs4, x, gamma, false, g, xg);
            for (uint i = 0; i < kQuantGateRows; i++) { ag[i] += Q::dot(wg[i], g, xg); au[i] += Q::dot(wu[i], g, xg); }
        }
    }
    for (uint i = 0; i < kQuantGateRows; i++) {
        const float a = simd_sum(ag[i]), u = simd_sum(au[i]);
        if (lane == 0) { part[sg][2 * i] = a; part[sg][2 * i + 1] = u; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < slots * kQuantGateRows) {
        const uint sl = tid / kQuantGateRows, i = tid % kQuantGateRows, r = (gid * slots + sl) * kQuantGateRows + i;
        if (r < p.N) {
            float a = 0.f, u = 0.f;
            for (uint k = 0; k < S; k++) { a += part[sl * S + k][2 * i]; u += part[sl * S + k][2 * i + 1]; }
            h[r] = a / (1.f + exp(-a)) * u;
        }
    }
}

// xn = x * rsqrt(mean(x^2) + eps) * gamma, one row (n a multiple of 4), one threadgroup: the normalized input of the quantized one-token kernels.
kernel void norm_vec(device const float * x [[buffer(0)]], device float * xn [[buffer(1)]], device const float * gamma [[buffer(2)]],
                     constant MatvecParams & p [[buffer(3)]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[kSimdgroupsPerGroup];
    device const float4 * x4 = (device const float4 *)x;
    float ss = 0.f;
    for (uint k = tid; k < p.K / 4; k += kGroupThreads) { const float4 v = x4[k]; ss += dot(v, v); }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint i = 0; i < kSimdgroupsPerGroup; i++) total += red[i];
    const float inv = rsqrt(total / float(p.K) + p.eps);
    for (uint k = tid; k < p.K / 4; k += kGroupThreads) ((device float4 *)xn)[k] = x4[k] * inv * ((device const float4 *)gamma)[k];
}

// x[row][:] = table[tokens[token_index + row]][:] from a quantized embedding table.
template <typename Q>
kernel void embedq(device const uchar * table [[buffer(0)]], device const int * tokens [[buffer(1)]], device float * x [[buffer(2)]],
                   constant EmbedParams & p [[buffer(3)]], device const float * pos_table [[buffer(4)]], uint2 gp [[thread_position_in_grid]]) {
    const uint i = gp.x, row = gp.y;
    if (i >= p.n_embd) return;
    const ulong RB = ulong(p.n_embd / Q::BLK) * Q::BYTES;
    x[(ulong)row * p.n_embd + i] = Q::elem(table + (ulong)tokens[p.token_index + row] * RB, i);
}



// ---- Prefill matmuls on quantized weights: the same kernels as matmul_16x16 / matmul_big, the weight tiles unpacked from the blocks as they are staged
// (row16: 16 weights per thread and step, the values elem gives, rounded to half as the f16 path's operands) instead of a dequantized copy first.

template <typename Q>
static void big_stage_wq(device const uchar * W, constant GemmParams & p, uint n0, uint k0, threadgroup half * sa, ushort tid) {
    const ushort row = tid / 2, half_k = tid % 2;
    const uint n = min(n0 + row, p.N - 1);   // rows past the end repeat the last one; their outputs are never stored
    half4 v[4];
    Q::row16(W + (ulong)n * (ulong(p.K / Q::BLK) * Q::BYTES), k0 + 16 * half_k, v);
    _Pragma("clang loop unroll(full)")
    for (ushort q = 0; q < 4; q++) {
        _Pragma("clang loop unroll(full)")
        for (ushort c = 0; c < 4; c++) {
            const ushort i = 4 * q + c;
            const ushort kb = 2 * half_k + i / 8, rb = row / 8;
            sa[64 * (8 * kb + rb) + 8 * (i % 8) + row % 8] = v[q][c];
        }
    }
}

template <typename Q>
kernel void matmul_bigq(device const uchar * W [[buffer(0)]], device const float * X [[buffer(1)]], device float * Y [[buffer(2)]],
                        device const float * gamma [[buffer(3)]], constant GemmParams & p [[buffer(4)]], device const float * bias [[buffer(5)]],
                        device const float * scale [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[kBigRows * kBigTokens];
    threadgroup half * sa = (threadgroup half *)buf;
    threadgroup half * sb = sa + kBigRows * kBigK;
    const uint n0 = g.y * kBigRows, m0 = g.x * kBigTokens;
    const bool norm = (p.flags & 1u) != 0;
    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.f);
    for (uint k0 = 0; k0 < p.K; k0 += kBigK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_stage_wq<Q>(W, p, n0, k0, sa, tid);
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

template <typename Q>
kernel void matmul_gate_up_bigq(device const uchar * Wg [[buffer(0)]], device const uchar * Wu [[buffer(1)]], device const float * X [[buffer(2)]],
                                device float * H [[buffer(3)]], device const float * gamma [[buffer(4)]], constant GemmParams & p [[buffer(5)]],
                                device const float * scale [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[2 * kBigRows * kBigTokens];
    threadgroup half * sg_w = (threadgroup half *)buf;
    threadgroup half * su_w = sg_w + kBigRows * kBigK;
    threadgroup half * sb = su_w + kBigRows * kBigK;
    const uint n0 = g.y * kBigRows, m0 = g.x * kBigTokens;
    simdgroup_float8x8 cg[8], cu[8];
    for (ushort i = 0; i < 8; i++) { cg[i] = simdgroup_float8x8(0.f); cu[i] = simdgroup_float8x8(0.f); }
    for (uint k0 = 0; k0 < p.K; k0 += kBigK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        big_stage_wq<Q>(Wg, p, n0, k0, sg_w, tid);
        big_stage_wq<Q>(Wu, p, n0, k0, su_w, tid);
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

// The small kernels' B tile (TN rows x kTileK) from quantized rows: one thread per 16 weights.
template <uint TN, typename Q>
static void stage_bq(device const uchar * W, constant GemmParams & p, uint n0, uint k0, threadgroup half4 * Bs4, uint tid) {
    constexpr uint K16 = kTileK / 16, K4 = kTileK / 4;
    const ulong RB = ulong(p.K / Q::BLK) * Q::BYTES;
    for (uint idx = tid; idx < TN * K16; idx += kGroupThreads) {
        const uint r = idx / K16, kq = idx % K16, n = n0 + r;
        half4 v[4] = {half4(0.h), half4(0.h), half4(0.h), half4(0.h)};
        if (n < p.N) Q::row16(W + (ulong)n * RB, k0 + 16 * kq, v);
        for (uint i = 0; i < 4; i++) Bs4[r * K4 + 4 * kq + i] = v[i];
    }
}

template <uint TM, uint TN, typename Q>
kernel void matmulq(device const uchar * W [[buffer(0)]], device const float * X [[buffer(1)]], device float * Y [[buffer(2)]],
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
        stage_bq<TN, Q>(W, p, n0, k0, Bs4, tid);
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

template <uint TM, uint TN, typename Q>
kernel void matmul_gate_upq(device const uchar * Wg [[buffer(0)]], device const uchar * Wu [[buffer(1)]], device const float * X [[buffer(2)]],
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
        stage_bq<TN, Q>(Wg, p, n0, k0, Bg4, tid);
        stage_bq<TN, Q>(Wu, p, n0, k0, Bu4, tid);
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

struct DequantParams { uint K; uint rows; };

// out[r][k] = half(W[r][k]) (rows x K elements, one thread each): long prompts dequantize each matrix once before the f16 matmul (cheaper than unpacking
// the tiles again for every 32 tokens, see the host's kFusedMaxTokens).
template <typename Q>
kernel void dequant_half(device const uchar * W [[buffer(0)]], device half * out [[buffer(1)]], constant DequantParams & p [[buffer(2)]],
                         uint2 gp [[thread_position_in_grid]]) {
    const uint k = gp.x, r = gp.y;
    if (k >= p.K || r >= p.rows) return;
    out[(ulong)r * p.K + k] = half(Q::elem(W + (ulong)r * (ulong(p.K / Q::BLK) * Q::BYTES), k));
}

#define DOOMS_QUANT_KERNELS(Q) \
    template [[host_name("matvec1_" #Q)]] kernel void matvec1q<Q>(device const uchar *, device const float *, device float *, device const float *, \
        constant MatvecParams &, device const float *, device const float *, threadgroup float4 *, uint, uint, uint, uint); \
    template [[host_name("gate_up1_" #Q)]] kernel void gate_up1q<Q>(device const uchar *, device const uchar *, device const float *, device float *, \
        device const float *, constant MatvecParams &, threadgroup float4 *, uint, uint, uint, uint); \
    template [[host_name("embed_" #Q)]] kernel void embedq<Q>(device const uchar *, device const int *, device float *, constant EmbedParams &, \
        device const float *, uint2); \
    template [[host_name("dequant_" #Q)]] kernel void dequant_half<Q>(device const uchar *, device half *, constant DequantParams &, uint2); \
    template [[host_name("matmul_big_" #Q)]] kernel void matmul_bigq<Q>(device const uchar *, device const float *, device float *, device const float *, \
        constant GemmParams &, device const float *, device const float *, uint2, ushort, ushort); \
    template [[host_name("matmul_gate_up_big_" #Q)]] kernel void matmul_gate_up_bigq<Q>(device const uchar *, device const uchar *, device const float *, \
        device float *, device const float *, constant GemmParams &, device const float *, uint2, ushort, ushort); \
    template [[host_name("matmul_16x16_" #Q)]] kernel void matmulq<16, 16, Q>(device const uchar *, device const float *, device float *, \
        device const float *, constant GemmParams &, device const float *, uint2, uint, uint, uint); \
    template [[host_name("matmul_gate_up_16x16_" #Q)]] kernel void matmul_gate_upq<16, 16, Q>(device const uchar *, device const uchar *, \
        device const float *, device float *, device const float *, constant GemmParams &, uint2, uint, uint, uint);
DOOMS_QUANT_KERNELS(Q4_0)
DOOMS_QUANT_KERNELS(Q5_0)
DOOMS_QUANT_KERNELS(Q8_0)
DOOMS_QUANT_KERNELS(Q4_K)
DOOMS_QUANT_KERNELS(Q6_K)
)MSL";
    return source.c_str();
}

}  // namespace dooms
