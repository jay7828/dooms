#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "core/status.h"

namespace dooms {

// Compute ops. Graph inputs and weights are values, not ops (see graph.h).
// Semantics (shapes are row-major, outermost first; see shape_infer.cpp):
//   Add, Mul   y = a (+|*) b, b broadcast to a (b right-aligned, each dim equal to a's or 1)
//   MatMul     y = a @ b, or a @ b^T with Attrs::transpose_b (PyTorch Linear weights are [out, in]). Both operands share a
//              floating dtype, except that b may be f16 or bf16 when a is f32 (low-precision weights, f32 activations);
//              the result has a's dtype.
//   Gelu       exact (erf) GELU, elementwise
//   Softmax    over Attrs::axis
//   LayerNorm  over the last dim, inputs (x, gamma, beta), Attrs::eps
//   Reshape    to Attrs::ints, same element count
//   Permute    dimensions reordered by Attrs::ints
//   Slice      x[..., start:end, ...] along Attrs::axis; Attrs::ints = {start, end}, 0 <= start < end <= dim
//   Concat     joins >= 2 inputs along Attrs::axis; all other dims must match
//   Relu       y = max(x, 0), elementwise, floating dtypes
//   ConvTranspose2d  NCHW input [N,IC,H,W] * weight [IC,OC,KH,KW] (PyTorch layout), groups = 1, dilation 1, no output padding, no bias:
//              out size (in-1)*stride - 2*pad + kernel per axis. Attrs from conv_transpose2d_attrs().
//   Silu       y = x * sigmoid(x), elementwise, floating dtypes
//   MaxPool2d  NCHW max pooling, Attrs from max_pool2d_attrs(); padded positions never win (they act as -inf), padding <= kernel / 2
//   UpsampleNearest2d  NCHW, every pixel repeated scale_h x scale_w times (integer factors >= 1), Attrs from upsample_nearest2d_attrs()
//   Scale      y = x * Attrs::scalar (any finite value), elementwise
//   SelfAttention  packed qkv [B,N,3E] -> [B,N,E]: per head, softmax(q.k^T * scale) . v with q|k|v the consecutive
//              thirds of the last dim and head h the h-th run of E/heads values; no mask. Attrs from self_attention_attrs().
//   Conv2d     NCHW input [N,C,H,W] * weight [OC,C,KH,KW] (PyTorch layout), groups = 1, no bias
//              (add one with Add). Attrs::ints encodes Conv2dParams, see conv2d_attrs().
enum class OpKind { Add, Mul, MatMul, Gelu, Softmax, LayerNorm, Reshape, Permute, Slice, Concat, Conv2d, Scale, SelfAttention, Relu, ConvTranspose2d, Silu, MaxPool2d, UpsampleNearest2d };

const char * op_name(OpKind op);  // "add", "mul", "matmul", ...
Result<OpKind> op_from_name(std::string_view name);
// Number of inputs, or kVariadic for ops that take any number >= kMinVariadicInputs.
constexpr int kVariadic = -1;
constexpr int kMinVariadicInputs = 2;
int op_arity(OpKind op);

// Op attributes. Which fields an op reads is documented above; others must stay default.
struct Attrs {
    int64_t axis = 0;
    float eps = 0.f;
    float scalar = 0.f;
    bool transpose_b = false;
    std::vector<int64_t> ints;

    bool operator==(const Attrs & o) const {
        return axis == o.axis && eps == o.eps && scalar == o.scalar && transpose_b == o.transpose_b && ints == o.ints;
    }
};

// Conv2d hyperparameters. Defaults describe a plain stride-1, unpadded, undilated convolution.
struct Conv2dParams {
    int64_t stride_h = 1, stride_w = 1;
    int64_t pad_h = 0, pad_w = 0;
    int64_t dilation_h = 1, dilation_w = 1;
};

// SelfAttention hyperparameters (stored in Attrs::ints[0] and Attrs::scalar).
struct SelfAttentionParams {
    int64_t heads = 1;
    float scale = 1.f;  // multiplies q.k^T before the softmax, typically head_dim^-0.5
};

Attrs self_attention_attrs(const SelfAttentionParams & p);
Result<SelfAttentionParams> self_attention_params(const Attrs & attrs);  // heads >= 1, finite scale

// ConvTranspose2d hyperparameters (Attrs::ints = stride_h, stride_w, pad_h, pad_w).
struct ConvTranspose2dParams {
    int64_t stride_h = 1, stride_w = 1;
    int64_t pad_h = 0, pad_w = 0;
};

Attrs conv_transpose2d_attrs(const ConvTranspose2dParams & p);
Result<ConvTranspose2dParams> conv_transpose2d_params(const Attrs & attrs);  // stride >= 1, pad >= 0, exactly 4 values

// MaxPool2d hyperparameters (Attrs::ints = kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w).
struct MaxPool2dParams {
    int64_t kernel_h = 1, kernel_w = 1;
    int64_t stride_h = 1, stride_w = 1;
    int64_t pad_h = 0, pad_w = 0;
};
Attrs max_pool2d_attrs(const MaxPool2dParams & p);
Result<MaxPool2dParams> max_pool2d_params(const Attrs & attrs);  // kernel >= 1, stride >= 1, 0 <= pad <= kernel / 2, exactly 6 values

// UpsampleNearest2d factors (Attrs::ints = scale_h, scale_w).
struct UpsampleParams {
    int64_t scale_h = 1, scale_w = 1;
};
Attrs upsample_nearest2d_attrs(const UpsampleParams & p);
Result<UpsampleParams> upsample_nearest2d_params(const Attrs & attrs);  // factors >= 1, exactly 2 values

Attrs conv2d_attrs(const Conv2dParams & p);
// Decodes and validates (stride >= 1, pad >= 0, dilation >= 1, exactly 6 values).
Result<Conv2dParams> conv2d_params(const Attrs & attrs);

}  // namespace dooms
