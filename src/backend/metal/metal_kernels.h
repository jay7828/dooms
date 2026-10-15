#pragma once
// Metal Shading Language source of the kernels, compiled when a backend is created (no offline metallib step). The parameter structs mirror the C++ ones in
// metal_backend.mm field for field (all 32-bit); the host checks their sizes.

namespace dooms {

inline const char * metal_kernel_source() {
    return R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct ConvParams {
    int ic, ih, iw, oc, oh, ow;
    int sh, sw;
    int K;      // ic * kh * kw
    int Kp;     // K rounded up to the tile depth: row length of the packed weights
    int flags;  // bit 0: add bias[oc]; bits 1-2: activation (0 none, 1 silu, 2 relu); bit 3: add res[...] after the activation
    int ksplit; // reduction slices computed by separate threadgroups (1: none). With ksplit > 1 the kernel writes raw partial sums and splitk_reduce finishes
    int kslice; // reduction length of one slice (a multiple of the tile depth)
};
struct ReduceParams { uint n; int ksplit, oc, ohw, flags; };
struct UnaryParams { uint n; int kind; float scale; };
struct BinaryParams { uint n; int kind; int d0, d1, d2, d3; int b0, b1, b2, b3; };
struct CopyParams { uint total; int count, inner, src_len, dst_len, src_off, dst_off; };
struct PermuteParams { uint n; int o0, o1, o2, o3; int s0, s1, s2, s3; };
struct PoolParams { uint total; int ih, iw, oh, ow, kh, kw, sh, sw, ph, pw; };
struct UpParams { uint total; int ih, iw, oh, ow, sh, sw; };

// Specialization: how the B (activation) tile is staged. true: a thread loads 8 consecutive pixels of one reduction index (one gather-table read per step; best for
// stride-1 layers with many input channels); false: a thread loads one pixel of 8 reduction indices, so the 32 lanes of a simdgroup read 32 neighbouring pixels at once
// (best for strided layers and few input channels). Chosen per layer by the host from measurements (bench/BASELINES.md).
constant bool kRunMapping [[function_constant(0)]];

constant int BM = 64;   // output channels per threadgroup
constant int BN = 64;   // output pixels per threadgroup
constant int BK = 16;   // reduction depth per step

// Implicit-GEMM convolution. C[oc][pixel] = sum_k W[oc][k] * X[k][pixel], with X gathered on the fly from the NCHW input through the table ktab[k] =
// (ic * ih * iw, ky * dilation_h - pad_h, kx * dilation_w - pad_w). One threadgroup of 4 simdgroups computes a 64 x 64 tile, each simdgroup 32 x 32 as 4 x 4
// 8 x 8 matrices. Weights are packed as half [ceil(oc/64) * 64][Kp] with zero padding, so only the activations need bounds checks.
kernel void conv_igemm(device const float * x [[buffer(0)]],
                       device const half * w [[buffer(1)]],
                       device const int4 * ktab [[buffer(2)]],
                       device const float * bias [[buffer(3)]],
                       device const float * res [[buffer(4)]],
                       device float * y [[buffer(5)]],
                       constant ConvParams & p [[buffer(6)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    // One 16 KB block of threadgroup memory serves as the operand tiles during the reduction and as the result staging area after it (two threadgroups fit a core).
    threadgroup float smem[BM * BN];
    threadgroup half * tA = (threadgroup half *)smem;
    threadgroup half * tB = tA + BM * BK;
    threadgroup float * tC = smem;
    const int ohw = p.oh * p.ow;
    const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
    const long batch = long(tg.z) / p.ksplit;
    const int slice = int(tg.z) % p.ksplit;
    const int kbeg = slice * p.kslice, kend = min(p.K, kbeg + p.kslice);
    device const float * xb = x + batch * long(p.ic) * p.ih * p.iw;
    const int sgm = (int(sg) >> 1) * 32, sgn = (int(sg) & 1) * 32;
    // run mapping: a thread owns 8 consecutive pixels of reduction index kk
    const int kk = int(tid) >> 3;
    const int c0 = (int(tid) & 7) * 8;
    int ox = 0, oy = 0;
    bool pix_ok0 = (n0 + c0) < ohw;
    if (kRunMapping && pix_ok0) { oy = (n0 + c0) / p.ow; ox = (n0 + c0) - oy * p.ow; }
    // column mapping: a thread owns pixel `col` and reduction indices kk0, kk0 + 2, ...
    const int col = int(tid) & 63;
    const int kk0 = int(tid) >> 6;
    const int pix = n0 + col;
    const bool pix_ok = pix < ohw;
    int coy = 0, cox = 0;
    if (!kRunMapping && pix_ok) { coy = pix / p.ow; cox = pix - coy * p.ow; }
    const int iy0 = coy * p.sh, ix0 = cox * p.sw;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.f);

    device const half * wrow = w + long(m0 + (int(tid) >> 1)) * p.Kp + (int(tid) & 1) * 8;
    threadgroup half * arow = tA + (int(tid) >> 1) * BK + (int(tid) & 1) * 8;
    // Software pipelining: the next step's operands are loaded into registers while the current step's matrix products run, then stored to threadgroup memory.
    half4 ra0, ra1;
    half rb[8];
    auto fetch = [&](int k0) {
        ra0 = *(device const half4 *)(wrow + k0);
        ra1 = *(device const half4 *)(wrow + k0 + 4);
        if (kRunMapping) {
            const int k = k0 + kk;
            if (k < p.K && pix_ok0) {
                const int4 t = ktab[k];
                int cy = oy, cx = ox;
                for (int j = 0; j < 8; j++) {
                    half v = 0.h;
                    if (n0 + c0 + j < ohw) {
                        const int iy = cy * p.sh + t.y, ix = cx * p.sw + t.z;
                        if (uint(iy) < uint(p.ih) && uint(ix) < uint(p.iw)) v = half(xb[t.x + iy * p.iw + ix]);
                    }
                    rb[j] = v;
                    if (++cx == p.ow) { cx = 0; cy++; }
                }
            } else {
                for (int j = 0; j < 8; j++) rb[j] = 0.h;
            }
        } else {
            for (int i = 0; i < 8; i++) {
                const int k = k0 + kk0 + 2 * i;
                half v = 0.h;
                if (pix_ok && k < p.K) {
                    const int4 t = ktab[k];
                    const int iy = iy0 + t.y, ix = ix0 + t.z;
                    if (uint(iy) < uint(p.ih) && uint(ix) < uint(p.iw)) v = half(xb[t.x + iy * p.iw + ix]);
                }
                rb[i] = v;
            }
        }
    };
    fetch(kbeg);
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        *(threadgroup half4 *)(arow) = ra0;
        *(threadgroup half4 *)(arow + 4) = ra1;
        if (kRunMapping) {
            *(threadgroup half4 *)(tB + kk * BN + c0) = half4(rb[0], rb[1], rb[2], rb[3]);
            *(threadgroup half4 *)(tB + kk * BN + c0 + 4) = half4(rb[4], rb[5], rb[6], rb[7]);
        } else {
            for (int i = 0; i < 8; i++) tB[(kk0 + 2 * i) * BN + col] = rb[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < kend) fetch(k0 + BK);
        for (int kf = 0; kf < BK; kf += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (int i = 0; i < 4; i++) simdgroup_load(a[i], tA + (sgm + i * 8) * BK + kf, BK);
            for (int j = 0; j < 4; j++) simdgroup_load(b[j], tB + kf * BN + sgn + j * 8, BN);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], tC + (sgm + i * 8) * BN + sgn + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.ksplit > 1) {  // raw partial sums of this slice; splitk_reduce adds the slices and applies the epilogue
        device float * part = y + long(tg.z) * p.oc * ohw;
        for (int e = int(tid); e < BM * BN; e += 128) {
            const int r = e >> 6, c = e & 63;
            const int oc = m0 + r, px = n0 + c;
            if (oc < p.oc && px < ohw) part[long(oc) * ohw + px] = tC[e];
        }
        return;
    }
    const int act = (p.flags >> 1) & 3;
    for (int e = int(tid); e < BM * BN; e += 128) {
        const int r = e >> 6, c = e & 63;
        const int oc = m0 + r, px = n0 + c;
        if (oc < p.oc && px < ohw) {
            float v = tC[e];
            if (p.flags & 1) v += bias[oc];
            if (act == 1) v = v / (1.f + exp(-v));
            else if (act == 2) v = max(v, 0.f);
            const long off = (batch * p.oc + oc) * long(ohw) + px;
            if (p.flags & 8) v += res[off];
            y[off] = v;
        }
    }
}

// Second half of a split-reduction convolution: out[n][oc][px] = epilogue(sum over slices of part[n * ksplit + s][oc][px]).
kernel void splitk_reduce(device const float * part [[buffer(0)]], device const float * bias [[buffer(1)]], device const float * res [[buffer(2)]], device float * y [[buffer(3)]],
                          constant ReduceParams & p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const long plane = long(p.oc) * p.ohw;
    const long n = long(i) / plane, rem = long(i) % plane;
    const int oc = int(rem / p.ohw);
    float v = 0.f;
    for (int s = 0; s < p.ksplit; s++) v += part[(n * p.ksplit + s) * plane + rem];
    if (p.flags & 1) v += bias[oc];
    const int act = (p.flags >> 1) & 3;
    if (act == 1) v = v / (1.f + exp(-v));
    else if (act == 2) v = max(v, 0.f);
    if (p.flags & 8) v += res[i];
    y[i] = v;
}

kernel void ew_unary(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], constant UnaryParams & p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float v = x[i];
    y[i] = p.kind == 0 ? v / (1.f + exp(-v)) : (p.kind == 1 ? max(v, 0.f) : v * p.scale);
}

kernel void ew_binary(device const float * a [[buffer(0)]], device const float * b [[buffer(1)]], device float * y [[buffer(2)]], constant BinaryParams & p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint r = i;
    const int i3 = int(r % uint(p.d3)); r /= uint(p.d3);
    const int i2 = int(r % uint(p.d2)); r /= uint(p.d2);
    const int i1 = int(r % uint(p.d1)); r /= uint(p.d1);
    const int i0 = int(r);
    const float bv = b[i0 * p.b0 + i1 * p.b1 + i2 * p.b2 + i3 * p.b3];
    y[i] = p.kind == 0 ? a[i] + bv : a[i] * bv;
}

// Copies `count` entries along an axis (concat: into an offset of the output; slice: from an offset of the input).
kernel void copy_axis(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], constant CopyParams & p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.total) return;
    const int inn = int(i % uint(p.inner));
    const uint t = i / uint(p.inner);
    const int a = int(t % uint(p.count));
    const long o = long(t / uint(p.count));
    y[(o * p.dst_len + p.dst_off + a) * p.inner + inn] = x[(o * p.src_len + p.src_off + a) * p.inner + inn];
}

kernel void permute4(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], constant PermuteParams & p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint r = i;
    const int i3 = int(r % uint(p.o3)); r /= uint(p.o3);
    const int i2 = int(r % uint(p.o2)); r /= uint(p.o2);
    const int i1 = int(r % uint(p.o1)); r /= uint(p.o1);
    const int i0 = int(r);
    y[i] = x[long(i0) * p.s0 + long(i1) * p.s1 + long(i2) * p.s2 + long(i3) * p.s3];
}

kernel void maxpool(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], constant PoolParams & p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.total) return;
    const int ohw = p.oh * p.ow;
    const uint plane = i / uint(ohw);
    const int pix = int(i % uint(ohw));
    const int oy = pix / p.ow, ox = pix - oy * p.ow;
    device const float * src = x + long(plane) * p.ih * p.iw;
    float m = -INFINITY;
    for (int ky = 0; ky < p.kh; ky++) {
        const int iy = oy * p.sh - p.ph + ky;
        if (uint(iy) >= uint(p.ih)) continue;
        for (int kx = 0; kx < p.kw; kx++) {
            const int ix = ox * p.sw - p.pw + kx;
            if (uint(ix) < uint(p.iw)) m = max(m, src[iy * p.iw + ix]);
        }
    }
    y[i] = m;
}

kernel void upsample_nearest(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], constant UpParams & p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.total) return;
    const int ohw = p.oh * p.ow;
    const uint plane = i / uint(ohw);
    const int pix = int(i % uint(ohw));
    const int oy = pix / p.ow, ox = pix - oy * p.ow;
    y[i] = x[long(plane) * p.ih * p.iw + (oy / p.sh) * p.iw + ox / p.sw];
}
)MSL";
}

}  // namespace dooms
