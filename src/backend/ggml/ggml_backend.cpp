#include "backend/ggml/ggml_backend.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>
#include <vector>
#include <unordered_map>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#define ACCELERATE_NEW_LAPACK 1  // the non-deprecated CBLAS declarations
#include <Accelerate/Accelerate.h>  // cblas_sgemm for the CPU attention (Apple Silicon only, ADR 0005)

#include "core/log.h"
#include "core/stats.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace dooms {

namespace {

static_assert(kMaxDims == GGML_MAX_DIMS, "IR rank limit must match ggml's");

// ggml aborts when its context is full, so the number of ggml tensors each IR node can
// create must be bounded exactly. Counts below are the worst case (non-f32 dtype, where
// every input is cast to f32 and the result cast back); tests exercise each worst case.
//   layer_norm: 3 casts + norm + mul + add + cast = 7  (largest of the fixed-size ops)
//   matmul: cast, transpose, cont, mul_mat, cast = 5;  permute: permute + cont = 2;  slice: view + cont = 2
//   conv2d: 2 casts + im2col + 2 reshape_2d + mul_mat + reshape_4d + permute + cont + cast = 10
//   silu / max_pool2d / upsample_nearest2d: cast + op + cast = 3
//   conv_transpose2d: 2 casts + conv_transpose + crop view + cont + cast = 6;  relu: cast + relu + cast = 3
//   concat of N inputs: N casts + (N-1) concats + 1 cast = 2N
//   self_attention: fused = 3 strided views + flash_attn + reshape = 5 (+2 casts on non-f32); on a GPU with f16 K/V
//     add 2 new tensors + 2 copies = 9 (+2). Unfused fallback = 24.
//     Worst case is a device that rejects the fused op after it was built: 3 views + 4 (f16 K/V) + flash_attn (8, never
//     part of the graph) + 24 fallback + 2 casts = 34.
constexpr size_t kFixedOpTensors = 7;
constexpr size_t kConv2dTensors = 10;
constexpr size_t kSelfAttentionTensors = 34;

size_t tensor_budget(const Node & n) {
    switch (n.op) {
        case OpKind::Conv2d: return kConv2dTensors;
        case OpKind::Concat: return 2 * n.inputs.size();
        case OpKind::SelfAttention: return kSelfAttentionTensors;
        default:             return kFixedOpTensors;
    }
}

// Frees the ggml backend once the backend object and every executable are gone.
struct BackendHandle {
    explicit BackendHandle(ggml_backend_t b, ggml_backend_t blas_backend = nullptr) : backend(b), blas(blas_backend) {}
    ~BackendHandle() {
        if (blas) ggml_backend_free(blas);
        if (backend) ggml_backend_free(backend);
    }
    BackendHandle(const BackendHandle &) = delete;
    BackendHandle & operator=(const BackendHandle &) = delete;
    ggml_backend_t backend;
    ggml_backend_t blas;  // optional: Accelerate matmuls for the CPU device; graphs then run through a scheduler [blas, cpu]
};

ggml_type to_ggml_type(DType t) {
    switch (t) {
        case DType::F32:  return GGML_TYPE_F32;
        case DType::F16:  return GGML_TYPE_F16;
        case DType::BF16: return GGML_TYPE_BF16;
        case DType::I32:  return GGML_TYPE_I32;
    }
    DOOMS_CHECK(false, "unknown DType");
    return GGML_TYPE_F32;
}

// IR shapes are row-major, outermost first. ggml lists the fastest-varying dimension first.
void to_ggml_ne(const Shape & s, int64_t ne[GGML_MAX_DIMS]) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) ne[i] = 1;
    for (int i = 0; i < s.rank(); i++) ne[i] = s.dim(s.rank() - 1 - i);
}

ggml_tensor * as_f32(ggml_context * c, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(c, t, GGML_TYPE_F32);
}

// Several ggml kernels only take f32 (mul_mat output, soft_max, norm). For other float
// dtypes we compute in f32 and cast back: correct, not fast (fp16 fast paths are M6).
ggml_tensor * back_to(ggml_context * c, ggml_tensor * t, DType dtype) {
    const ggml_type want = to_ggml_type(dtype);
    return t->type == want ? t : ggml_cast(c, t, want);
}

Status check_supported(const Graph & g) {
    for (size_t i = 0; i < g.nodes().size(); i++) {
        const Node & n = g.nodes()[i];
        const TypeInfo & out = g.values()[n.output].type;
        const std::string label = "node " + std::to_string(i) + " (" + op_name(n.op) + ")";
        if ((n.op == OpKind::Add || n.op == OpKind::Mul) && out.dtype == DType::I32) {
            return unsupported(label + ": ggml backend does not support i32 elementwise ops");
        }
        if (n.op == OpKind::ConvTranspose2d) {
            auto cp = conv_transpose2d_params(n.attrs);
            if (cp.is_ok() && cp.value().stride_h != cp.value().stride_w) {
                return unsupported(label + ": ggml's transposed convolution takes a single stride for both axes (got " + std::to_string(cp.value().stride_h) +
                                   " and " + std::to_string(cp.value().stride_w) + ")");
            }
        }
        if (n.op == OpKind::Softmax && n.attrs.axis != out.shape.rank() - 1) {
            return unsupported(label + ": ggml backend only supports softmax over the last axis (got axis " +
                               std::to_string(n.attrs.axis) + " of rank " + std::to_string(out.shape.rank()) + ")");
        }
    }
    return Status();
}

ggml_tensor * lower_reshape(ggml_context * c, ggml_tensor * x, const Shape & to) {
    int64_t ne[GGML_MAX_DIMS];
    to_ggml_ne(to, ne);
    if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
    switch (to.rank()) {
        case 0:
        case 1: return ggml_reshape_1d(c, x, ne[0]);
        case 2: return ggml_reshape_2d(c, x, ne[0], ne[1]);
        case 3: return ggml_reshape_3d(c, x, ne[0], ne[1], ne[2]);
        default: return ggml_reshape_4d(c, x, ne[0], ne[1], ne[2], ne[3]);
    }
}

// IR: output axis i is input axis perm[i] (row-major). ggml dim d = rank-1-axis, and
// ggml_permute(a, ax0..ax3) moves input dim k to output dim ax_k.
ggml_tensor * lower_permute(ggml_context * c, ggml_tensor * x, const std::vector<int64_t> & perm) {
    const int rank = static_cast<int>(perm.size());
    int axes[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; d++) axes[d] = d;
    for (int i = 0; i < rank; i++) axes[rank - 1 - perm[i]] = rank - 1 - i;
    // Materialize: downstream kernels (and reshape) expect contiguous inputs.
    return ggml_cont(c, ggml_permute(c, x, axes[0], axes[1], axes[2], axes[3]));
}

// x[..., start:end, ...] along IR axis `axis`: a strided view, then materialized.
ggml_tensor * lower_slice(ggml_context * c, ggml_tensor * x, int rank, int64_t axis, int64_t start, int64_t end) {
    const int d = rank - 1 - static_cast<int>(axis);
    int64_t ne[GGML_MAX_DIMS] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    ne[d] = end - start;
    ggml_tensor * view = ggml_view_4d(c, x, ne[0], ne[1], ne[2], ne[3], x->nb[1], x->nb[2], x->nb[3],
                                      static_cast<size_t>(start) * x->nb[d]);
    return ggml_cont(c, view);
}

enum class ConvAlgorithm { Im2colF32, Im2colF16, Direct };

// Same steps as ggml_conv_2d, but with an f32 im2col buffer: ggml_conv_2d builds it in f16,
// which would round the activations of an f32 graph.
ggml_tensor * lower_conv2d(ggml_context * c, ggml_tensor * x, ggml_tensor * w, const Conv2dParams & p, ConvAlgorithm algo) {
    if (algo == ConvAlgorithm::Direct) {
        return ggml_conv_2d_direct(c, w, x, static_cast<int>(p.stride_w), static_cast<int>(p.stride_h), static_cast<int>(p.pad_w), static_cast<int>(p.pad_h),
                                   static_cast<int>(p.dilation_w), static_cast<int>(p.dilation_h));
    }
    if (algo == ConvAlgorithm::Im2colF16) {
        return ggml_conv_2d(c, w, x, static_cast<int>(p.stride_w), static_cast<int>(p.stride_h), static_cast<int>(p.pad_w), static_cast<int>(p.pad_h),
                            static_cast<int>(p.dilation_w), static_cast<int>(p.dilation_h));
    }
    // ggml's "dimension 0" is width, so width parameters come first.
    ggml_tensor * im2col = ggml_im2col(c, w, x, static_cast<int>(p.stride_w), static_cast<int>(p.stride_h),
                                       static_cast<int>(p.pad_w), static_cast<int>(p.pad_h),
                                       static_cast<int>(p.dilation_w), static_cast<int>(p.dilation_h), true, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(c, ggml_reshape_2d(c, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                                   ggml_reshape_2d(c, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    r = ggml_reshape_4d(c, r, im2col->ne[1], im2col->ne[2], im2col->ne[3], w->ne[3]);  // [OW, OH, N, OC]
    return ggml_cont(c, ggml_permute(c, r, 0, 1, 3, 2));                               // [OW, OH, OC, N]
}

// ggml's transposed convolution has no padding: it returns the full (in-1)*stride + kernel output, so the padding is
// removed by cropping `pad` rows/columns from each side (view + copy). Kernel layout [KW, KH, OC, IC] is PyTorch's
// [IC, OC, KH, KW] reversed, as for every tensor.
ggml_tensor * lower_conv_transpose2d(ggml_context * c, ggml_tensor * x, ggml_tensor * w, const ConvTranspose2dParams & p) {
    ggml_tensor * full = ggml_conv_transpose_2d_p0(c, w, x, static_cast<int>(p.stride_w));
    if (p.pad_h == 0 && p.pad_w == 0) return full;
    const int64_t ow = full->ne[0] - 2 * p.pad_w, oh = full->ne[1] - 2 * p.pad_h;
    ggml_tensor * view = ggml_view_4d(c, full, ow, oh, full->ne[2], full->ne[3], full->nb[1], full->nb[2], full->nb[3],
                                      static_cast<size_t>(p.pad_h) * full->nb[1] + static_cast<size_t>(p.pad_w) * full->nb[0]);
    return ggml_cont(c, view);
}

// ---- exact-erf GELU on the CPU -------------------------------------------------------------------------------------------------------------------
// ggml's CPU gelu_erf evaluates erff() element by element (about 1.3 ns each: 40 ms of a 350 ms ViT-H backbone). This kernel computes the same function with NEON,
// four floats at a time: erf(a) = 1 - poly(t) * exp(-a^2) with t = 1 / (1 + 0.3275911 a) (Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7, about the rounding noise of f32)
// and a degree-7 exp with Cody-Waite range reduction (relative error ~5e-9). Used for contiguous f32 tensors on the CPU device only.
#if defined(__ARM_NEON)
inline float32x4_t exp_neg_f32x4(float32x4_t x) {  // exp(x) for x <= 0
    x = vmaxq_f32(x, vdupq_n_f32(-87.f));
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504089f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));
    r = vfmsq_f32(r, n, vdupq_n_f32(1.428606765330187e-06f));
    float32x4_t p = vdupq_n_f32(1.f / 5040.f);
    p = vfmaq_f32(vdupq_n_f32(1.f / 720.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 120.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 24.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 6.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

inline float32x4_t gelu_erf_f32x4(float32x4_t x) {
    const float32x4_t a = vabsq_f32(vmulq_f32(x, vdupq_n_f32(0.70710678118f)));
    const float32x4_t t = vdivq_f32(vdupq_n_f32(1.f), vfmaq_f32(vdupq_n_f32(1.f), a, vdupq_n_f32(0.3275911f)));
    float32x4_t poly = vdupq_n_f32(1.061405429f);
    poly = vfmaq_f32(vdupq_n_f32(-1.453152027f), poly, t);
    poly = vfmaq_f32(vdupq_n_f32(1.421413741f), poly, t);
    poly = vfmaq_f32(vdupq_n_f32(-0.284496736f), poly, t);
    poly = vfmaq_f32(vdupq_n_f32(0.254829592f), poly, t);
    poly = vmulq_f32(poly, t);
    const float32x4_t erf_abs = vfmsq_f32(vdupq_n_f32(1.f), poly, exp_neg_f32x4(vnegq_f32(vmulq_f32(a, a))));
    // erf(x) = sign(x) * erf(|x|); the sign of x * 0.7071 is the sign of x
    const uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000u));
    const float32x4_t erf_x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(erf_abs), sign));
    return vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), x), vaddq_f32(vdupq_n_f32(1.f), erf_x));
}
#endif

void gelu_erf_cpu_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * /*userdata*/) {
    const int64_t n = ggml_nelements(a);
    const int64_t chunk = (n + nth - 1) / nth;
    const int64_t begin = std::min<int64_t>(n, chunk * ith), end = std::min<int64_t>(n, begin + chunk);
    const float * x = static_cast<const float *>(a->data);
    float * y = static_cast<float *>(dst->data);
    int64_t i = begin;
#if defined(__ARM_NEON)
    for (; i + 4 <= end; i += 4) vst1q_f32(y + i, gelu_erf_f32x4(vld1q_f32(x + i)));
    if (i < end) {  // the tail goes through the same vector code so that every element uses one formula
        float in[4] = {0, 0, 0, 0}, out[4];
        for (int64_t k = i; k < end; k++) in[k - i] = x[k];
        vst1q_f32(out, gelu_erf_f32x4(vld1q_f32(in)));
        for (int64_t k = i; k < end; k++) y[k] = out[k - i];
        return;
    }
#else
    for (; i < end; i++) y[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.70710678118f));
#endif
}

// ---- self-attention on the CPU with Accelerate -----------------------------------------------------------------------------------------------
// ggml's CPU flash attention runs at 0.13 TFLOPS on the ViT-H shapes (head size 80, 193 tokens): 58 of the 350 ms backbone. This kernel handles one (batch, head) pair per
// task with three Accelerate calls on the AMX units: S = scale * Q K^T (cblas_sgemm reading the head's columns of the packed qkv in place), a NEON row softmax, and
// O = P V written straight into the head's columns of the output. Used for contiguous f32 packed qkv on the CPU device when Accelerate is enabled (cpu_blas).
struct CpuAttentionParams {
    int64_t heads;
    float scale;
};

const CpuAttentionParams * intern_attention_params(int64_t heads, float scale) {
    // ggml custom ops carry an opaque pointer that has to outlive the graph; the few distinct (heads, scale) pairs of a process live in a never-shrinking pool.
    static std::mutex m;
    static std::deque<CpuAttentionParams> pool;
    std::lock_guard<std::mutex> lock(m);
    for (const CpuAttentionParams & p : pool) if (p.heads == heads && p.scale == scale) return &p;
    pool.push_back({heads, scale});
    return &pool.back();
}

void softmax_row_inplace(float * row, int64_t n) {
    float m = -INFINITY;
    for (int64_t i = 0; i < n; i++) m = std::max(m, row[i]);
    float sum = 0.f;
    int64_t i = 0;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.f);
    const float32x4_t vm = vdupq_n_f32(m);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t e = exp_neg_f32x4(vsubq_f32(vld1q_f32(row + i), vm));
        vst1q_f32(row + i, e);
        acc = vaddq_f32(acc, e);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < n; i++) { row[i] = std::exp(row[i] - m); sum += row[i]; }
    const float inv = 1.f / sum;
    for (int64_t k = 0; k < n; k++) row[k] *= inv;
}

void attention_cpu_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const CpuAttentionParams & p = *static_cast<const CpuAttentionParams *>(userdata);
    const ggml_tensor * qkv = dst->src[0];
    const int64_t E = dst->ne[0], N = dst->ne[1], B = dst->ne[2];
    const int64_t D = E / p.heads, ld = 3 * E;
    const float * x = static_cast<const float *>(qkv->data);
    float * y = static_cast<float *>(dst->data);
    std::vector<float> scores(static_cast<size_t>(N * N));
    for (int64_t task = ith; task < B * p.heads; task += nth) {
        const int64_t b = task / p.heads, h = task % p.heads;
        const float * base = x + b * N * ld;
        const float * q = base + h * D;
        const float * k = base + E + h * D;
        const float * v = base + 2 * E + h * D;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(N), static_cast<int>(N), static_cast<int>(D), p.scale, q, static_cast<int>(ld), k, static_cast<int>(ld), 0.f,
                    scores.data(), static_cast<int>(N));
        for (int64_t i = 0; i < N; i++) softmax_row_inplace(scores.data() + i * N, N);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(N), static_cast<int>(D), static_cast<int>(N), 1.f, scores.data(), static_cast<int>(N), v, static_cast<int>(ld), 0.f,
                    y + b * N * E + h * D, static_cast<int>(E));
    }
}

struct LowerOptions {
    ConvAlgorithm conv = ConvAlgorithm::Im2colF32;
    ggml_backend_t backend = nullptr;
    bool fused_attention = true;
    bool attention_kv_f16 = true;
    bool cpu_attention = false;  // CPU device with Accelerate: self-attention through attention_cpu_op
};

// Fused: strided views of q, k, v straight out of the packed f32 qkv [3E, N, B] (ggml order), then flash attention.
// The result [D, H, N, B] has exactly the memory order of the IR's [B, N, E] output. Returns nullptr when the
// backend cannot run the op for this shape/type.
ggml_tensor * lower_attention_fused(ggml_context * c, ggml_tensor * qkv, int64_t heads, float scale, ggml_backend_t backend, bool kv_f16) {
    const int64_t N = qkv->ne[1], B = qkv->ne[2], E = qkv->ne[0] / 3, D = E / heads;
    auto part = [&](int p) {  // [D, N, H, B]: tokens stride 3E, heads stride D
        return ggml_view_4d(c, qkv, D, N, heads, B, qkv->nb[1], static_cast<size_t>(D) * sizeof(float), qkv->nb[2],
                            static_cast<size_t>(p) * static_cast<size_t>(E) * sizeof(float));
    };
    ggml_tensor * k = part(1), * v = part(2);
    if (kv_f16) {  // materialize K and V as contiguous f16 [D, N, H, B]
        k = ggml_cpy(c, k, ggml_new_tensor_4d(c, GGML_TYPE_F16, D, N, heads, B));
        v = ggml_cpy(c, v, ggml_new_tensor_4d(c, GGML_TYPE_F16, D, N, heads, B));
    }
    ggml_tensor * fa = ggml_flash_attn_ext(c, part(0), k, v, nullptr, scale, 0.f, 0.f);
    if (backend && !ggml_backend_supports_op(backend, fa)) return nullptr;
    return ggml_reshape_3d(c, fa, E, N, B);
}

// Unfused reference chain, the same operations as the IR composition slice/reshape/permute/matmul/scale/softmax/matmul.
ggml_tensor * lower_attention_unfused(ggml_context * c, ggml_tensor * qkv, int64_t heads, float scale) {
    const int64_t N = qkv->ne[1], B = qkv->ne[2], E = qkv->ne[0] / 3, D = E / heads;
    auto head_part = [&](int p) {  // [D, N, H, B], contiguous
        ggml_tensor * slice = ggml_cont(c, ggml_view_3d(c, qkv, E, N, B, qkv->nb[1], qkv->nb[2],
                                                        static_cast<size_t>(p) * static_cast<size_t>(E) * sizeof(float)));
        return ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, slice, D, heads, N, B), 0, 2, 1, 3));
    };
    ggml_tensor * q = head_part(0), * k = head_part(1), * v = head_part(2);
    ggml_tensor * probs = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, k, q), scale));  // [N, N, H, B]
    ggml_tensor * ctx = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, v)), probs);       // [D, N, H, B]
    ctx = ggml_cont(c, ggml_permute(c, ctx, 0, 2, 1, 3));                                  // [D, H, N, B]
    return ggml_reshape_3d(c, ctx, E, N, B);
}

Result<ggml_tensor *> lower_node(ggml_context * c, const Graph & g, const Node & n,
                                 const std::vector<ggml_tensor *> & t, const LowerOptions & opt) {
    const TypeInfo & out = g.values()[n.output].type;
    ggml_tensor * in0 = t[n.inputs[0]];
    switch (n.op) {
        case OpKind::Add:
            return back_to(c, ggml_add(c, as_f32(c, in0), as_f32(c, t[n.inputs[1]])), out.dtype);
        case OpKind::Mul:
            return back_to(c, ggml_mul(c, as_f32(c, in0), as_f32(c, t[n.inputs[1]])), out.dtype);
        case OpKind::MatMul: {
            // ggml_mul_mat(w, x) contracts dim 0 of both, so w must have K in dim 0, i.e. IR [N, K].
            ggml_tensor * w = t[n.inputs[1]];
            if (!n.attrs.transpose_b) w = ggml_cont(c, ggml_transpose(c, w));
            return back_to(c, ggml_mul_mat(c, w, as_f32(c, in0)), out.dtype);
        }
        case OpKind::Gelu: {
            ggml_tensor * x = as_f32(c, in0);
            if (opt.backend && ggml_backend_is_cpu(opt.backend) && ggml_is_contiguous(x)) return back_to(c, ggml_map_custom1(c, x, gelu_erf_cpu_op, GGML_N_TASKS_MAX, nullptr), out.dtype);
            return back_to(c, ggml_gelu_erf(c, x), out.dtype);
        }
        case OpKind::Softmax:
            return back_to(c, ggml_soft_max(c, as_f32(c, in0)), out.dtype);
        case OpKind::Scale:
            return back_to(c, ggml_scale(c, as_f32(c, in0), n.attrs.scalar), out.dtype);
        case OpKind::LayerNorm: {
            ggml_tensor * y = ggml_norm(c, as_f32(c, in0), n.attrs.eps);
            y = ggml_mul(c, y, as_f32(c, t[n.inputs[1]]));
            y = ggml_add(c, y, as_f32(c, t[n.inputs[2]]));
            return back_to(c, y, out.dtype);
        }
        case OpKind::Reshape:
            return lower_reshape(c, in0, out.shape);
        case OpKind::Permute:
            return lower_permute(c, in0, n.attrs.ints);
        case OpKind::Slice:
            return lower_slice(c, in0, out.shape.rank(), n.attrs.axis, n.attrs.ints[0], n.attrs.ints[1]);
        case OpKind::Concat: {
            const int dim = out.shape.rank() - 1 - static_cast<int>(n.attrs.axis);
            ggml_tensor * acc = as_f32(c, in0);
            for (size_t k = 1; k < n.inputs.size(); k++) acc = ggml_concat(c, acc, as_f32(c, t[n.inputs[k]]), dim);
            return back_to(c, acc, out.dtype);
        }
        case OpKind::SelfAttention: {
            auto params = self_attention_params(n.attrs);
            if (!params.is_ok()) return params.status();
            ggml_tensor * qkv = as_f32(c, in0);
            if (opt.cpu_attention && opt.fused_attention && ggml_is_contiguous(qkv) && qkv->ne[0] % (3 * params.value().heads) == 0 && qkv->ne[3] == 1) {
                // output [E, N, B]: the same memory order as the IR output [B, N, E]
                ggml_tensor * args[1] = {qkv};
                ggml_tensor * y = ggml_custom_4d(c, GGML_TYPE_F32, qkv->ne[0] / 3, qkv->ne[1], qkv->ne[2], 1, args, 1, attention_cpu_op, GGML_N_TASKS_MAX,
                                                 const_cast<CpuAttentionParams *>(intern_attention_params(params.value().heads, params.value().scale)));
                return back_to(c, y, out.dtype);
            }
            ggml_tensor * y = opt.fused_attention ? lower_attention_fused(c, qkv, params.value().heads, params.value().scale, opt.backend,
                                                                             opt.attention_kv_f16 && opt.backend && !ggml_backend_is_cpu(opt.backend)) : nullptr;
            if (!y) y = lower_attention_unfused(c, qkv, params.value().heads, params.value().scale);
            return back_to(c, y, out.dtype);
        }
        case OpKind::Relu:
            return back_to(c, ggml_relu(c, as_f32(c, in0)), out.dtype);
        case OpKind::Silu:
            return back_to(c, ggml_silu(c, as_f32(c, in0)), out.dtype);
        case OpKind::MaxPool2d: {
            auto params = max_pool2d_params(n.attrs);
            if (!params.is_ok()) return params.status();
            const MaxPool2dParams & q = params.value();
            // ggml's dimension 0 is the width; its pooling ignores padded positions like PyTorch's max pooling.
            return back_to(c, ggml_pool_2d(c, as_f32(c, in0), GGML_OP_POOL_MAX, static_cast<int>(q.kernel_w), static_cast<int>(q.kernel_h),
                                           static_cast<int>(q.stride_w), static_cast<int>(q.stride_h), static_cast<float>(q.pad_w), static_cast<float>(q.pad_h)), out.dtype);
        }
        case OpKind::UpsampleNearest2d:
            return back_to(c, ggml_interpolate(c, as_f32(c, in0), out.shape.dim(3), out.shape.dim(2), out.shape.dim(1), out.shape.dim(0), GGML_SCALE_MODE_NEAREST), out.dtype);
        case OpKind::ConvTranspose2d: {
            auto params = conv_transpose2d_params(n.attrs);
            if (!params.is_ok()) return params.status();
            return back_to(c, lower_conv_transpose2d(c, as_f32(c, in0), as_f32(c, t[n.inputs[1]]), params.value()), out.dtype);
        }
        case OpKind::Conv2d: {
            auto params = conv2d_params(n.attrs);
            if (!params.is_ok()) return params.status();
            return back_to(c, lower_conv2d(c, as_f32(c, in0), as_f32(c, t[n.inputs[1]]), params.value(), opt.conv), out.dtype);
        }
    }
    return internal_error("lower_node: unhandled op");
}

class GgmlExecutable : public Executable {
public:
    static Result<std::unique_ptr<Executable>> create(std::shared_ptr<BackendHandle> handle, const Graph & graph, bool fused_attention, bool kv_f16, ConvAlgorithm conv) {
        std::unique_ptr<GgmlExecutable> e(new GgmlExecutable(std::move(handle), graph));
        e->fused_attention_ = fused_attention;
        e->attention_kv_f16_ = kv_f16;
        e->conv_ = conv;
        Status st = e->build();
        if (!st.is_ok()) return st;
        return std::unique_ptr<Executable>(std::move(e));
    }

    ~GgmlExecutable() override {
        if (sched_) ggml_backend_sched_free(sched_);
        if (galloc_) ggml_gallocr_free(galloc_);
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_) ggml_free(ctx_);
    }

    const Graph & graph() const override { return graph_; }

    Status set_weight(std::string_view name, const TensorView & data) override {
        auto id = check_weight(graph_, name, data);
        if (!id.is_ok()) return id.status();
        ggml_tensor * t = tensors_[id.value()];
        if (static_cast<int64_t>(ggml_nbytes(t)) != data.span_bytes()) return internal_error("weight byte size mismatch");
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        weight_set_[id.value()] = true;
        return Status();
    }

    Status run(const std::vector<NamedTensor> & inputs, const std::vector<NamedTensor> & outputs) override {
        std::vector<ValueId> in_ids, out_ids;
        DOOMS_RETURN_IF_ERROR(check_bindings(graph_, inputs, outputs, &in_ids, &out_ids));
        for (size_t i = 0; i < graph_.values().size(); i++) {
            if (graph_.values()[i].kind == ValueKind::Weight && !weight_set_[i]) {
                return invalid_argument("weight '" + graph_.values()[i].name + "' was not set");
            }
        }
        for (size_t i = 0; i < inputs.size(); i++) {
            ggml_tensor * t = tensors_[in_ids[i]];
            if (static_cast<int64_t>(ggml_nbytes(t)) != inputs[i].tensor.span_bytes()) return internal_error("input byte size mismatch");
            ggml_backend_tensor_set(t, inputs[i].tensor.data(), 0, ggml_nbytes(t));
        }
        const ggml_status status = sched_ ? ggml_backend_sched_graph_compute(sched_, gf_) : ggml_backend_graph_compute(handle_->backend, gf_);
        if (status != GGML_STATUS_SUCCESS) {
            return internal_error("ggml graph compute failed with status " + std::to_string(static_cast<int>(status)));
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            ggml_tensor * t = tensors_[out_ids[i]];
            if (static_cast<int64_t>(ggml_nbytes(t)) != outputs[i].tensor.span_bytes()) return internal_error("output byte size mismatch");
            ggml_backend_tensor_get(t, outputs[i].tensor.data(), 0, ggml_nbytes(t));
        }
        return Status();
    }

    Result<ProfileReport> profile(const std::vector<NamedTensor> & inputs, int iterations) override {
        if (iterations < 1) return invalid_argument("profile: iterations must be >= 1, got " + std::to_string(iterations));
        std::vector<ValueId> in_ids;
        DOOMS_RETURN_IF_ERROR(check_input_bindings(graph_, inputs, &in_ids));
        for (size_t i = 0; i < graph_.values().size(); i++) {
            if (graph_.values()[i].kind == ValueKind::Weight && !weight_set_[i]) {
                return invalid_argument("weight '" + graph_.values()[i].name + "' was not set");
            }
        }
        for (size_t i = 0; i < inputs.size(); i++) {
            ggml_backend_tensor_set(tensors_[in_ids[i]], inputs[i].tensor.data(), 0, ggml_nbytes(tensors_[in_ids[i]]));
        }

        struct SchedDeleter { void operator()(ggml_backend_sched * s) const { ggml_backend_sched_free(s); } };
        struct BackendDeleter { void operator()(ggml_backend * b) const { ggml_backend_free(b); } };
        ggml_backend_t be = handle_->backend;
        // ggml's scheduler requires its last backend to be a CPU (the fallback device). Every operation of an
        // executable that runs normally is supported by `be`, so nothing is actually placed on the fallback.
        std::unique_ptr<ggml_backend, BackendDeleter> fallback;
        std::vector<ggml_backend_t> sched_backends = {be};
        if (handle_->blas) sched_backends.insert(sched_backends.begin(), handle_->blas);  // same placement as run(): large matmuls on Accelerate
        if (!ggml_backend_is_cpu(be)) {
            fallback.reset(ggml_backend_cpu_init());
            if (!fallback) return resource_exhausted("could not create the CPU fallback backend for profiling");
            sched_backends.push_back(fallback.get());
        }
        // With Accelerate matmuls the executable already owns the scheduler that placed its graph; profile through it so that the run being measured is the real one.
        std::unique_ptr<ggml_backend_sched, SchedDeleter> local_sched;
        ggml_backend_sched_t sched = sched_;
        if (!sched) {
            local_sched.reset(ggml_backend_sched_new(sched_backends.data(), nullptr, static_cast<int>(sched_backends.size()), graph_size_ + n_named_ + 1, false, false));
            if (!local_sched) return resource_exhausted("could not create a backend scheduler for profiling");
            if (!ggml_backend_sched_alloc_graph(local_sched.get(), gf_)) return resource_exhausted("scheduler could not allocate the graph");
            sched = local_sched.get();
        }

        const size_t n_nodes = graph_.nodes().size();
        struct Observer {
            const std::unordered_map<const ggml_tensor *, int> * owner;
            std::vector<double> per_node_ms;   // this iteration
            std::vector<double> view_ms;       // observed time of operations that do no work
            std::chrono::steady_clock::time_point last;
        } obs{&owner_, std::vector<double>(n_nodes, 0.0), {}, {}};
        auto callback = [](ggml_tensor * t, bool ask, void * user) -> bool {
            if (ask) return true;  // observe every operation
            auto * o = static_cast<Observer *>(user);
            const auto now = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - o->last).count();
            o->last = now;
            auto it = o->owner->find(t);
            if (it != o->owner->end()) {
                o->per_node_ms[static_cast<size_t>(it->second)] += ms;
                if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE) {
                    o->view_ms.push_back(ms);
                }
            }
            return true;
        };
        ggml_backend_sched_set_eval_callback(sched, callback, &obs);

        auto now_ms = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        std::vector<std::vector<double>> samples(n_nodes);
        std::vector<double> whole, all_view;
        for (int it = 0; it <= iterations; it++) {  // iteration 0 is a warmup and is discarded
            const double t0 = now_ms();
            ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);  // the unobserved run is the one run() performs: through the member scheduler if there is one
            if ((sched_ ? ggml_backend_sched_graph_compute(sched_, gf_) : ggml_backend_graph_compute(be, gf_)) != GGML_STATUS_SUCCESS) return internal_error("profile: unobserved run failed");
            ggml_backend_sched_set_eval_callback(sched, callback, &obs);
            const double whole_ms = now_ms() - t0;

            std::fill(obs.per_node_ms.begin(), obs.per_node_ms.end(), 0.0);
            obs.view_ms.clear();
            obs.last = std::chrono::steady_clock::now();
            if (ggml_backend_sched_graph_compute(sched, gf_) != GGML_STATUS_SUCCESS) return internal_error("profile: observed run failed");
            if (it == 0) continue;
            whole.push_back(whole_ms);
            for (size_t k = 0; k < n_nodes; k++) samples[k].push_back(obs.per_node_ms[k]);
            all_view.insert(all_view.end(), obs.view_ms.begin(), obs.view_ms.end());
        }

        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);  // do not leave a callback pointing at this stack frame in a scheduler that outlives it
        ProfileReport report;
        report.iterations = iterations;
        report.whole_graph_ms = median(whole);
        report.observation_overhead_ms = median(all_view);
        std::vector<int> ops_per_node(n_nodes, 0);
        for (int i = 0; i < ggml_graph_n_nodes(gf_); i++) {
            auto o = owner_.find(ggml_graph_node(gf_, i));
            if (o != owner_.end()) ops_per_node[static_cast<size_t>(o->second)]++;
        }
        for (size_t k = 0; k < n_nodes; k++) {
            NodeProfile np;
            np.node_index = static_cast<int>(k);
            np.op = graph_.nodes()[k].op;
            np.ms = median(samples[k]);
            np.backend_ops = ops_per_node[k];
            report.nodes.push_back(np);
        }
        return report;
    }

private:
    GgmlExecutable(std::shared_ptr<BackendHandle> handle, const Graph & graph)
        : handle_(std::move(handle)), graph_(graph) {}

    Status build() {
        const size_t n_values = graph_.values().size();
        // Exact bound: one tensor per input/weight, plus every node's budget (which already
        // includes that node's output tensor, so intermediate values are not counted again).
        size_t n_named = 0;
        for (const Value & v : graph_.values()) n_named += v.kind != ValueKind::Intermediate;
        size_t op_tensors = 0;
        for (const Node & n : graph_.nodes()) op_tensors += tensor_budget(n);
        const size_t max_tensors = n_named + op_tensors;
        const size_t graph_size = op_tensors + 1;

        ggml_init_params ip = {};
        ip.mem_size = ggml_tensor_overhead() * max_tensors + ggml_graph_overhead_custom(graph_size, false);
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;  // tensor data lives in backend buffers
        ctx_ = ggml_init(ip);
        if (!ctx_) return resource_exhausted("ggml_init failed");

        tensors_.assign(n_values, nullptr);
        weight_set_.assign(n_values, false);

        // Inputs and weights first, allocated together; everything else is planned by gallocr.
        for (size_t i = 0; i < n_values; i++) {
            const Value & v = graph_.values()[i];
            if (v.kind == ValueKind::Intermediate) continue;
            int64_t ne[GGML_MAX_DIMS];
            to_ggml_ne(v.type.shape, ne);
            ggml_tensor * t = ggml_new_tensor(ctx_, to_ggml_type(v.type.dtype), GGML_MAX_DIMS, ne);
            ggml_set_name(t, v.name.c_str());
            if (v.kind == ValueKind::Input) ggml_set_input(t);
            tensors_[i] = t;
        }
        if (n_named > 0) {
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, handle_->backend);
            if (!buffer_) return resource_exhausted("could not allocate device memory for inputs and weights");
        }

        LowerOptions lower_options;
        lower_options.backend = handle_->backend;
        lower_options.fused_attention = fused_attention_;
        lower_options.attention_kv_f16 = attention_kv_f16_;
        lower_options.conv = conv_;
        lower_options.cpu_attention = handle_->blas != nullptr && ggml_backend_is_cpu(handle_->backend);

        // Remember which IR node created each ggml tensor, so profiling can attribute time per IR node.
        ggml_tensor * last = nullptr;
        auto claim_new_tensors = [&](int owner) {
            ggml_tensor * t = last ? ggml_get_next_tensor(ctx_, last) : ggml_get_first_tensor(ctx_);
            for (; t; t = ggml_get_next_tensor(ctx_, t)) {
                if (owner >= 0) owner_[t] = owner;
                last = t;
            }
        };
        claim_new_tensors(-1);  // inputs and weights belong to no node
        int node_index = 0;
        for (const Node & n : graph_.nodes()) {
            auto lowered = lower_node(ctx_, graph_, n, tensors_, lower_options);
            if (!lowered.is_ok()) return lowered.status();
            tensors_[n.output] = lowered.value();
            claim_new_tensors(node_index++);
        }

        graph_size_ = graph_size;
        n_named_ = n_named;
        gf_ = ggml_new_graph_custom(ctx_, graph_size, false);
        for (const GraphOutput & o : graph_.outputs()) {
            ggml_set_output(tensors_[o.value]);
            ggml_build_forward_expand(gf_, tensors_[o.value]);
        }

        if (handle_->blas) {
            // the scheduler places each operation on the first backend that supports it (large matmuls on Accelerate), and plans the intermediates itself
            ggml_backend_t order[2] = {handle_->blas, handle_->backend};
            sched_ = ggml_backend_sched_new(order, nullptr, 2, graph_size + n_named + 1, false, false);  // nodes plus leaves (inputs and weights)
            if (!sched_ || !ggml_backend_sched_alloc_graph(sched_, gf_)) return resource_exhausted("could not allocate memory for the scheduled graph");
            return Status();
        }
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(handle_->backend));
        if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
            return resource_exhausted("could not allocate device memory for intermediate tensors");
        }
        return Status();
    }

    std::shared_ptr<BackendHandle> handle_;
    Graph graph_;
    std::unordered_map<const ggml_tensor *, int> owner_;  // ggml tensor -> index of the IR node that created it
    size_t graph_size_ = 0, n_named_ = 0;
    bool fused_attention_ = true;
    bool attention_kv_f16_ = true;
    ConvAlgorithm conv_ = ConvAlgorithm::Im2colF32;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    ggml_cgraph * gf_ = nullptr;
    std::vector<ggml_tensor *> tensors_;  // by ValueId
    std::vector<bool> weight_set_;        // by ValueId
};

class GgmlBackend : public Backend {
public:
    GgmlBackend(std::shared_ptr<BackendHandle> handle, std::string name, bool fused_attention, bool kv_f16, ConvAlgorithm conv)
        : handle_(std::move(handle)), name_(std::move(name)), fused_attention_(fused_attention), kv_f16_(kv_f16), conv_(conv) {}

    const char * name() const override { return name_.c_str(); }
    Status supports(const Graph & graph) const override { return check_supported(graph); }

    Result<std::unique_ptr<Executable>> compile(const Graph & graph) override {
        DOOMS_RETURN_IF_ERROR(graph.validate());
        DOOMS_RETURN_IF_ERROR(check_supported(graph));
        return GgmlExecutable::create(handle_, graph, fused_attention_, kv_f16_, conv_);
    }

private:
    std::shared_ptr<BackendHandle> handle_;
    std::string name_;
    bool fused_attention_;
    bool kv_f16_;
    ConvAlgorithm conv_;
};

}  // namespace

Result<std::unique_ptr<Backend>> create_ggml_backend(const GgmlBackendOptions & options) {
    if (options.n_threads < 0) return invalid_argument("n_threads must be >= 0");
    ConvAlgorithm conv;
    if (options.conv_algorithm == "im2col") conv = ConvAlgorithm::Im2colF32;
    else if (options.conv_algorithm == "im2col_f16") conv = ConvAlgorithm::Im2colF16;
    else if (options.conv_algorithm == "direct") conv = ConvAlgorithm::Direct;
    else return invalid_argument("unknown conv_algorithm '" + options.conv_algorithm + "' (expected im2col, im2col_f16 or direct)");

    static std::once_flag loaded;
    std::call_once(loaded, [] { ggml_backend_load_all(); });

    ggml_backend_t b = nullptr;
    if (options.device == "auto") {
        b = ggml_backend_init_best();
    } else if (options.device == "cpu") {
        b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else if (options.device == "gpu") {
        b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!b) b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    } else {
        return invalid_argument("unknown device '" + options.device + "' (expected auto, cpu or gpu)");
    }
    if (!b) return not_found("no ggml backend available for device '" + options.device + "'");

    if (options.n_threads > 0 && ggml_backend_is_cpu(b)) ggml_backend_cpu_set_n_threads(b, options.n_threads);
    ggml_backend_t blas = nullptr;
    if (options.cpu_blas && ggml_backend_is_cpu(b)) {
        if (ggml_backend_dev_t dev = ggml_backend_dev_by_name("BLAS")) blas = ggml_backend_dev_init(dev, nullptr);
        if (blas && options.n_threads > 0) {
            if (auto set_threads = reinterpret_cast<void (*)(ggml_backend_t, int)>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(blas)), "ggml_backend_blas_set_n_threads"))) set_threads(blas, options.n_threads);
        }
    }

    std::string name = ggml_backend_name(b);
    DOOMS_LOG_INFO("ggml backend '%s' (requested '%s')", name.c_str(), options.device.c_str());
    return std::unique_ptr<Backend>(new GgmlBackend(std::make_shared<BackendHandle>(b, blas), std::move(name), options.fused_attention, options.attention_kv_f16, conv));
}

}  // namespace dooms
