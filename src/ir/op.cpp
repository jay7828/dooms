#include "ir/op.h"

#include <cmath>
#include <string>

namespace dooms {

namespace {

struct OpInfo {
    OpKind op;
    const char * name;
    int arity;
};

constexpr OpInfo kOps[] = {
    {OpKind::Add,       "add",        2},
    {OpKind::Mul,       "mul",        2},
    {OpKind::MatMul,    "matmul",     2},
    {OpKind::Gelu,      "gelu",       1},
    {OpKind::Softmax,   "softmax",    1},
    {OpKind::LayerNorm, "layer_norm", 3},
    {OpKind::Reshape,   "reshape",    1},
    {OpKind::Permute,   "permute",    1},
    {OpKind::Slice,     "slice",      1},
    {OpKind::Concat,    "concat",     kVariadic},
    {OpKind::Conv2d,    "conv2d",     2},
    {OpKind::Scale,     "scale",      1},
    {OpKind::SelfAttention, "self_attention", 1},
    {OpKind::Relu,      "relu",       1},
    {OpKind::ConvTranspose2d, "conv_transpose2d", 2},
    {OpKind::Silu,      "silu",       1},
    {OpKind::MaxPool2d, "max_pool2d", 1},
    {OpKind::UpsampleNearest2d, "upsample_nearest2d", 1},
};

const OpInfo & info(OpKind op) {
    for (const OpInfo & i : kOps) {
        if (i.op == op) return i;
    }
    DOOMS_CHECK(false, "unknown OpKind value");
    return kOps[0];  // unreachable
}

}  // namespace

const char * op_name(OpKind op) { return info(op).name; }
int op_arity(OpKind op) { return info(op).arity; }

Result<OpKind> op_from_name(std::string_view name) {
    for (const OpInfo & i : kOps) {
        if (name == i.name) return i.op;
    }
    return invalid_argument("unknown op name '" + std::string(name) + "'");
}

// Order of the encoded values in Attrs::ints.
namespace {
constexpr size_t kConv2dAttrCount = 6;
}

Attrs self_attention_attrs(const SelfAttentionParams & p) {
    Attrs a;
    a.ints = {p.heads};
    a.scalar = p.scale;
    return a;
}

Result<SelfAttentionParams> self_attention_params(const Attrs & attrs) {
    if (attrs.ints.size() != 1) return invalid_argument("self_attention needs exactly one integer attribute (heads), got " + std::to_string(attrs.ints.size()));
    SelfAttentionParams p;
    p.heads = attrs.ints[0];
    p.scale = attrs.scalar;
    if (p.heads < 1) return invalid_argument("self_attention: heads must be >= 1, got " + std::to_string(p.heads));
    if (!std::isfinite(p.scale)) return invalid_argument("self_attention: scale must be finite");
    return p;
}

namespace {
constexpr size_t kConvTranspose2dAttrCount = 4;
}

Attrs conv_transpose2d_attrs(const ConvTranspose2dParams & p) {
    Attrs a;
    a.ints = {p.stride_h, p.stride_w, p.pad_h, p.pad_w};
    return a;
}

Result<ConvTranspose2dParams> conv_transpose2d_params(const Attrs & attrs) {
    if (attrs.ints.size() != kConvTranspose2dAttrCount) {
        return invalid_argument("conv_transpose2d needs " + std::to_string(kConvTranspose2dAttrCount) + " parameters, got " + std::to_string(attrs.ints.size()));
    }
    ConvTranspose2dParams p;
    p.stride_h = attrs.ints[0];
    p.stride_w = attrs.ints[1];
    p.pad_h = attrs.ints[2];
    p.pad_w = attrs.ints[3];
    if (p.stride_h < 1 || p.stride_w < 1) return invalid_argument("conv_transpose2d: strides must be >= 1");
    if (p.pad_h < 0 || p.pad_w < 0) return invalid_argument("conv_transpose2d: padding must be >= 0");
    return p;
}

Attrs max_pool2d_attrs(const MaxPool2dParams & p) {
    Attrs a;
    a.ints = {p.kernel_h, p.kernel_w, p.stride_h, p.stride_w, p.pad_h, p.pad_w};
    return a;
}

Result<MaxPool2dParams> max_pool2d_params(const Attrs & attrs) {
    if (attrs.ints.size() != 6) return invalid_argument("max_pool2d needs 6 parameters, got " + std::to_string(attrs.ints.size()));
    MaxPool2dParams p;
    p.kernel_h = attrs.ints[0]; p.kernel_w = attrs.ints[1];
    p.stride_h = attrs.ints[2]; p.stride_w = attrs.ints[3];
    p.pad_h = attrs.ints[4]; p.pad_w = attrs.ints[5];
    if (p.kernel_h < 1 || p.kernel_w < 1) return invalid_argument("max_pool2d: kernel must be >= 1");
    if (p.stride_h < 1 || p.stride_w < 1) return invalid_argument("max_pool2d: stride must be >= 1");
    if (p.pad_h < 0 || p.pad_w < 0 || p.pad_h > p.kernel_h / 2 || p.pad_w > p.kernel_w / 2) return invalid_argument("max_pool2d: padding must be in [0, kernel / 2]");
    return p;
}

Attrs upsample_nearest2d_attrs(const UpsampleParams & p) {
    Attrs a;
    a.ints = {p.scale_h, p.scale_w};
    return a;
}

Result<UpsampleParams> upsample_nearest2d_params(const Attrs & attrs) {
    if (attrs.ints.size() != 2) return invalid_argument("upsample_nearest2d needs 2 parameters, got " + std::to_string(attrs.ints.size()));
    UpsampleParams p;
    p.scale_h = attrs.ints[0];
    p.scale_w = attrs.ints[1];
    if (p.scale_h < 1 || p.scale_w < 1) return invalid_argument("upsample_nearest2d: factors must be >= 1");
    return p;
}

Attrs conv2d_attrs(const Conv2dParams & p) {
    Attrs a;
    a.ints = {p.stride_h, p.stride_w, p.pad_h, p.pad_w, p.dilation_h, p.dilation_w};
    return a;
}

Result<Conv2dParams> conv2d_params(const Attrs & attrs) {
    if (attrs.ints.size() != kConv2dAttrCount) {
        return invalid_argument("conv2d needs " + std::to_string(kConv2dAttrCount) + " parameters, got " +
                                std::to_string(attrs.ints.size()));
    }
    Conv2dParams p;
    p.stride_h = attrs.ints[0];
    p.stride_w = attrs.ints[1];
    p.pad_h = attrs.ints[2];
    p.pad_w = attrs.ints[3];
    p.dilation_h = attrs.ints[4];
    p.dilation_w = attrs.ints[5];
    if (p.stride_h < 1 || p.stride_w < 1) return invalid_argument("conv2d: strides must be >= 1");
    if (p.pad_h < 0 || p.pad_w < 0) return invalid_argument("conv2d: padding must be >= 0");
    if (p.dilation_h < 1 || p.dilation_w < 1) return invalid_argument("conv2d: dilations must be >= 1");
    return p;
}

}  // namespace dooms
