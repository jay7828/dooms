#include "ir/shape_infer.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace dooms {

namespace {

std::string describe(const TypeInfo & t) { return std::string(dtype_name(t.dtype)) + t.shape.to_string(); }

Result<TypeInfo> same_type(const TypeInfo & t) { return t; }

Result<TypeInfo> infer_binary_elementwise(OpKind op, const std::vector<TypeInfo> & in) {
    const TypeInfo & a = in[0];
    const TypeInfo & b = in[1];
    if (a.dtype != b.dtype) {
        return invalid_argument(std::string(op_name(op)) + ": dtype mismatch " + describe(a) + " vs " + describe(b));
    }
    if (b.shape.rank() > a.shape.rank()) {
        return invalid_argument(std::string(op_name(op)) + ": second operand " + describe(b) +
                                " has higher rank than first " + describe(a));
    }
    for (int i = 0; i < b.shape.rank(); i++) {
        const int64_t bd = b.shape.dim(b.shape.rank() - 1 - i);
        const int64_t ad = a.shape.dim(a.shape.rank() - 1 - i);
        if (bd != ad && bd != 1) {
            return invalid_argument(std::string(op_name(op)) + ": " + describe(b) + " does not broadcast to " + describe(a));
        }
    }
    return same_type(a);
}

Result<TypeInfo> infer_matmul(const std::vector<TypeInfo> & in, const Attrs & attrs) {
    const TypeInfo & a = in[0];
    const TypeInfo & b = in[1];
    // Same floating dtype, or low-precision weights (f16/bf16) under f32 activations.
    const bool same_dtype = a.dtype == b.dtype && dtype_is_float(a.dtype);
    const bool low_precision_weights = a.dtype == DType::F32 && (b.dtype == DType::F16 || b.dtype == DType::BF16);
    if (!same_dtype && !low_precision_weights) {
        return invalid_argument("matmul: operands must share a floating dtype or have f16/bf16 weights under f32 activations, got " +
                                describe(a) + " and " + describe(b));
    }
    const int ra = a.shape.rank();
    const int rb = b.shape.rank();
    if (ra < 2 || rb < 2) {
        return invalid_argument("matmul: operands need rank >= 2, got " + describe(a) + " and " + describe(b));
    }
    const int64_t k_a = a.shape.dim(ra - 1);
    const int64_t k_b = attrs.transpose_b ? b.shape.dim(rb - 1) : b.shape.dim(rb - 2);
    const int64_t n = attrs.transpose_b ? b.shape.dim(rb - 2) : b.shape.dim(rb - 1);
    if (k_a != k_b) {
        return invalid_argument("matmul: inner dimensions differ (" + std::to_string(k_a) + " vs " + std::to_string(k_b) +
                                ") for " + describe(a) + " @ " + describe(b) + (attrs.transpose_b ? "^T" : ""));
    }
    // b is either a plain matrix applied to every leading batch of a (weights), or has a's rank and batch dims.
    if (rb != 2) {
        if (rb != ra) {
            return invalid_argument("matmul: rank " + std::to_string(rb) + " second operand must be rank 2 or match first operand's rank " + std::to_string(ra));
        }
        for (int i = 0; i < ra - 2; i++) {
            if (a.shape.dim(i) != b.shape.dim(i)) {
                return invalid_argument("matmul: batch dimension " + std::to_string(i) + " differs: " + describe(a) + " vs " + describe(b));
            }
        }
    }
    int64_t dims[kMaxDims];
    for (int i = 0; i < ra - 1; i++) dims[i] = a.shape.dim(i);
    dims[ra - 1] = n;
    auto shape = Shape::create(dims, ra);
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), a.dtype};
}

Result<TypeInfo> infer_float_unary(OpKind op, const TypeInfo & x) {
    if (!dtype_is_float(x.dtype)) {
        return invalid_argument(std::string(op_name(op)) + ": needs a floating dtype, got " + describe(x));
    }
    return same_type(x);
}

Result<TypeInfo> infer_conv_transpose2d(const std::vector<TypeInfo> & in, const Attrs & attrs) {
    const TypeInfo & x = in[0];
    const TypeInfo & w = in[1];
    auto params = conv_transpose2d_params(attrs);
    if (!params.is_ok()) return params.status();
    const ConvTranspose2dParams & p = params.value();
    if (x.dtype != w.dtype || !dtype_is_float(x.dtype)) {
        return invalid_argument("conv_transpose2d: input and weight must share a floating dtype, got " + describe(x) + " and " + describe(w));
    }
    if (x.shape.rank() != 4 || w.shape.rank() != 4) {
        return invalid_argument("conv_transpose2d: needs NCHW input and [IC,OC,KH,KW] weight (rank 4), got " + describe(x) + " and " + describe(w));
    }
    if (x.shape.dim(1) != w.shape.dim(0)) {
        return invalid_argument("conv_transpose2d: input has " + std::to_string(x.shape.dim(1)) + " channels but the weight expects " + std::to_string(w.shape.dim(0)));
    }
    int64_t out_hw[2];
    const int64_t size[2] = {x.shape.dim(2), x.shape.dim(3)};
    const int64_t kernel[2] = {w.shape.dim(2), w.shape.dim(3)};
    const int64_t stride[2] = {p.stride_h, p.stride_w};
    const int64_t pad[2] = {p.pad_h, p.pad_w};
    for (int a = 0; a < 2; a++) {
        int64_t grown, trimmed;
        if (__builtin_mul_overflow(size[a] - 1, stride[a], &grown) || __builtin_add_overflow(grown, kernel[a], &grown) ||
            __builtin_mul_overflow(pad[a], int64_t(2), &trimmed)) {
            return resource_exhausted("conv_transpose2d: output size computation overflows int64");
        }
        out_hw[a] = grown - trimmed;
        if (out_hw[a] < 1) {
            return invalid_argument("conv_transpose2d: padding " + std::to_string(pad[a]) + " removes the whole output on spatial axis " + std::to_string(a) +
                                    " (full size " + std::to_string(grown) + ")");
        }
    }
    auto shape = Shape::create({x.shape.dim(0), w.shape.dim(1), out_hw[0], out_hw[1]});
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), x.dtype};
}

Result<TypeInfo> infer_max_pool2d(const TypeInfo & x, const Attrs & attrs) {
    auto params = max_pool2d_params(attrs);
    if (!params.is_ok()) return params.status();
    const MaxPool2dParams & p = params.value();
    if (!dtype_is_float(x.dtype) || x.shape.rank() != 4) return invalid_argument("max_pool2d: needs a floating NCHW input, got " + describe(x));
    const int64_t size[2] = {x.shape.dim(2), x.shape.dim(3)};
    const int64_t kernel[2] = {p.kernel_h, p.kernel_w}, stride[2] = {p.stride_h, p.stride_w}, pad[2] = {p.pad_h, p.pad_w};
    int64_t out[2];
    for (int a = 0; a < 2; a++) {
        const int64_t padded = size[a] + 2 * pad[a];
        if (padded < kernel[a]) return invalid_argument("max_pool2d: kernel " + std::to_string(kernel[a]) + " is larger than the padded input " + std::to_string(padded) + " on spatial axis " + std::to_string(a));
        out[a] = (padded - kernel[a]) / stride[a] + 1;
    }
    auto shape = Shape::create({x.shape.dim(0), x.shape.dim(1), out[0], out[1]});
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), x.dtype};
}

Result<TypeInfo> infer_upsample_nearest2d(const TypeInfo & x, const Attrs & attrs) {
    auto params = upsample_nearest2d_params(attrs);
    if (!params.is_ok()) return params.status();
    if (!dtype_is_float(x.dtype) || x.shape.rank() != 4) return invalid_argument("upsample_nearest2d: needs a floating NCHW input, got " + describe(x));
    int64_t h, w;
    if (__builtin_mul_overflow(x.shape.dim(2), params.value().scale_h, &h) || __builtin_mul_overflow(x.shape.dim(3), params.value().scale_w, &w)) {
        return resource_exhausted("upsample_nearest2d: output size overflows int64");
    }
    auto shape = Shape::create({x.shape.dim(0), x.shape.dim(1), h, w});
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), x.dtype};
}

Result<TypeInfo> infer_scale(const TypeInfo & x, const Attrs & attrs) {
    if (!std::isfinite(attrs.scalar)) return invalid_argument("scale: factor must be finite, got " + std::to_string(attrs.scalar));
    return infer_float_unary(OpKind::Scale, x);
}

Result<TypeInfo> infer_self_attention(const TypeInfo & qkv, const Attrs & attrs) {
    auto params = self_attention_params(attrs);
    if (!params.is_ok()) return params.status();
    auto r = infer_float_unary(OpKind::SelfAttention, qkv);
    if (!r.is_ok()) return r;
    if (qkv.shape.rank() != 3) return invalid_argument("self_attention: needs a packed [B, N, 3E] input, got " + describe(qkv));
    const int64_t last = qkv.shape.dim(2);
    if (last % 3 != 0) return invalid_argument("self_attention: last dimension " + std::to_string(last) + " is not a multiple of 3 (packed q|k|v)");
    const int64_t embed = last / 3;
    if (embed % params.value().heads != 0) {
        return invalid_argument("self_attention: embedding " + std::to_string(embed) + " is not divisible by " + std::to_string(params.value().heads) + " heads");
    }
    auto shape = Shape::create({qkv.shape.dim(0), qkv.shape.dim(1), embed});
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), qkv.dtype};
}

Result<TypeInfo> infer_softmax(const TypeInfo & x, const Attrs & attrs) {
    auto r = infer_float_unary(OpKind::Softmax, x);
    if (!r.is_ok()) return r;
    if (attrs.axis < 0 || attrs.axis >= x.shape.rank()) {
        return out_of_range("softmax: axis " + std::to_string(attrs.axis) + " outside [0, " +
                            std::to_string(x.shape.rank()) + ") for " + describe(x));
    }
    return r;
}

Result<TypeInfo> infer_layer_norm(const std::vector<TypeInfo> & in, const Attrs & attrs) {
    const TypeInfo & x = in[0];
    auto r = infer_float_unary(OpKind::LayerNorm, x);
    if (!r.is_ok()) return r;
    if (x.shape.rank() < 1) return invalid_argument("layer_norm: input must have rank >= 1");
    if (!(attrs.eps > 0.f) || !std::isfinite(attrs.eps)) {
        return invalid_argument("layer_norm: eps must be finite and > 0, got " + std::to_string(attrs.eps));
    }
    const int64_t last = x.shape.dim(x.shape.rank() - 1);
    for (int i = 1; i <= 2; i++) {
        const TypeInfo & p = in[i];
        if (p.dtype != x.dtype || p.shape.rank() != 1 || p.shape.dim(0) != last) {
            return invalid_argument(std::string("layer_norm: ") + (i == 1 ? "gamma" : "beta") + " must be " +
                                    dtype_name(x.dtype) + "[" + std::to_string(last) + "], got " + describe(p));
        }
    }
    return r;
}

Result<TypeInfo> infer_reshape(const TypeInfo & x, const Attrs & attrs) {
    auto shape = Shape::create(attrs.ints.data(), static_cast<int>(attrs.ints.size()));
    if (!shape.is_ok()) return invalid_argument("reshape: invalid target shape: " + shape.status().message());
    if (shape.value().numel() != x.shape.numel()) {
        return invalid_argument("reshape: " + describe(x) + " has " + std::to_string(x.shape.numel()) +
                                " elements, target " + shape.value().to_string() + " has " +
                                std::to_string(shape.value().numel()));
    }
    return TypeInfo{shape.value(), x.dtype};
}

Result<TypeInfo> infer_permute(const TypeInfo & x, const Attrs & attrs) {
    const int rank = x.shape.rank();
    if (rank < 1 || static_cast<int>(attrs.ints.size()) != rank) {
        return invalid_argument("permute: needs " + std::to_string(rank) + " indices for " + describe(x) +
                                ", got " + std::to_string(attrs.ints.size()));
    }
    bool seen[kMaxDims] = {};
    int64_t dims[kMaxDims];
    for (int i = 0; i < rank; i++) {
        const int64_t p = attrs.ints[i];
        if (p < 0 || p >= rank) return out_of_range("permute: index " + std::to_string(p) + " outside [0, " + std::to_string(rank) + ")");
        if (seen[p]) return invalid_argument("permute: index " + std::to_string(p) + " repeated");
        seen[p] = true;
        dims[i] = x.shape.dim(static_cast<int>(p));
    }
    auto shape = Shape::create(dims, rank);
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), x.dtype};
}

Result<TypeInfo> infer_slice(const TypeInfo & x, const Attrs & attrs) {
    const int rank = x.shape.rank();
    if (attrs.axis < 0 || attrs.axis >= rank) {
        return out_of_range("slice: axis " + std::to_string(attrs.axis) + " outside [0, " + std::to_string(rank) +
                            ") for " + describe(x));
    }
    if (attrs.ints.size() != 2) return invalid_argument("slice: ints must be {start, end}");
    const int64_t start = attrs.ints[0], end = attrs.ints[1], dim = x.shape.dim(static_cast<int>(attrs.axis));
    if (start < 0 || start >= end || end > dim) {
        return out_of_range("slice: range [" + std::to_string(start) + ", " + std::to_string(end) +
                            ") invalid for axis " + std::to_string(attrs.axis) + " of size " + std::to_string(dim));
    }
    int64_t dims[kMaxDims];
    for (int i = 0; i < rank; i++) dims[i] = x.shape.dim(i);
    dims[attrs.axis] = end - start;
    auto shape = Shape::create(dims, rank);
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), x.dtype};
}

Result<TypeInfo> infer_concat(const std::vector<TypeInfo> & in, const Attrs & attrs) {
    const TypeInfo & first = in[0];
    const int rank = first.shape.rank();
    if (rank < 1) return invalid_argument("concat: inputs must have rank >= 1");
    if (attrs.axis < 0 || attrs.axis >= rank) {
        return out_of_range("concat: axis " + std::to_string(attrs.axis) + " outside [0, " + std::to_string(rank) +
                            ") for " + describe(first));
    }
    int64_t dims[kMaxDims];
    for (int i = 0; i < rank; i++) dims[i] = first.shape.dim(i);
    for (size_t k = 1; k < in.size(); k++) {
        const TypeInfo & t = in[k];
        if (t.dtype != first.dtype || t.shape.rank() != rank) {
            return invalid_argument("concat: input " + std::to_string(k) + " is " + describe(t) + " but input 0 is " + describe(first));
        }
        for (int i = 0; i < rank; i++) {
            if (i == attrs.axis) continue;
            if (t.shape.dim(i) != first.shape.dim(i)) {
                return invalid_argument("concat: input " + std::to_string(k) + " " + describe(t) + " differs from " +
                                        describe(first) + " in non-concat dimension " + std::to_string(i));
            }
        }
        if (__builtin_add_overflow(dims[attrs.axis], t.shape.dim(static_cast<int>(attrs.axis)), &dims[attrs.axis])) {
            return resource_exhausted("concat: size along axis overflows int64");
        }
    }
    auto shape = Shape::create(dims, rank);
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), first.dtype};
}

Result<TypeInfo> infer_conv2d(const std::vector<TypeInfo> & in, const Attrs & attrs) {
    const TypeInfo & x = in[0];
    const TypeInfo & w = in[1];
    auto params = conv2d_params(attrs);
    if (!params.is_ok()) return params.status();
    const Conv2dParams & p = params.value();
    if (x.dtype != w.dtype || !dtype_is_float(x.dtype)) {
        return invalid_argument("conv2d: input and weight must share a floating dtype, got " + describe(x) + " and " + describe(w));
    }
    if (x.shape.rank() != 4 || w.shape.rank() != 4) {
        return invalid_argument("conv2d: needs NCHW input and OIHW weight (rank 4), got " + describe(x) + " and " + describe(w));
    }
    if (x.shape.dim(1) != w.shape.dim(1)) {
        return invalid_argument("conv2d: input has " + std::to_string(x.shape.dim(1)) + " channels but weight expects " +
                                std::to_string(w.shape.dim(1)) + " (groups are not supported)");
    }
    // out = (in + 2*pad - dilation*(k-1) - 1) / stride + 1, per spatial axis
    int64_t out_hw[2];
    const int64_t size[2] = {x.shape.dim(2), x.shape.dim(3)};
    const int64_t kernel[2] = {w.shape.dim(2), w.shape.dim(3)};
    const int64_t stride[2] = {p.stride_h, p.stride_w};
    const int64_t pad[2] = {p.pad_h, p.pad_w};
    const int64_t dil[2] = {p.dilation_h, p.dilation_w};
    for (int a = 0; a < 2; a++) {
        int64_t padded, span, reach;
        if (__builtin_mul_overflow(pad[a], int64_t(2), &padded) || __builtin_add_overflow(padded, size[a], &padded) ||
            __builtin_mul_overflow(dil[a], kernel[a] - 1, &reach) || __builtin_add_overflow(reach, int64_t(1), &span)) {
            return resource_exhausted("conv2d: output size computation overflows int64");
        }
        if (padded < span) {
            return invalid_argument("conv2d: kernel footprint " + std::to_string(span) + " exceeds padded input " +
                                    std::to_string(padded) + " on spatial axis " + std::to_string(a));
        }
        out_hw[a] = (padded - span) / stride[a] + 1;
    }
    auto shape = Shape::create({x.shape.dim(0), w.shape.dim(0), out_hw[0], out_hw[1]});
    if (!shape.is_ok()) return shape.status();
    return TypeInfo{shape.value(), x.dtype};
}

// An attribute an op does not read must stay default, so a typo'd or misplaced
// attribute is an error instead of being silently ignored.
Status check_unused_attrs(OpKind op, const Attrs & a) {
    const bool uses_axis = op == OpKind::Softmax || op == OpKind::Slice || op == OpKind::Concat;
    const bool uses_eps = op == OpKind::LayerNorm;
    const bool uses_scalar = op == OpKind::Scale || op == OpKind::SelfAttention;
    const bool uses_transpose = op == OpKind::MatMul;
    const bool uses_ints = op == OpKind::Reshape || op == OpKind::Permute || op == OpKind::Slice || op == OpKind::Conv2d || op == OpKind::SelfAttention ||
                           op == OpKind::ConvTranspose2d || op == OpKind::MaxPool2d || op == OpKind::UpsampleNearest2d;
    const char * bad = nullptr;
    if (!uses_axis && a.axis != 0) bad = "axis";
    else if (!uses_eps && a.eps != 0.f) bad = "eps";
    else if (!uses_scalar && a.scalar != 0.f) bad = "scalar";
    else if (!uses_transpose && a.transpose_b) bad = "transpose_b";
    else if (!uses_ints && !a.ints.empty()) bad = "ints";
    if (bad) return invalid_argument(std::string(op_name(op)) + " does not use attribute '" + bad + "'");
    return Status();
}

}  // namespace

Result<TypeInfo> infer_output(OpKind op, const std::vector<TypeInfo> & inputs, const Attrs & attrs) {
    DOOMS_RETURN_IF_ERROR(check_unused_attrs(op, attrs));
    const int arity = op_arity(op);
    if (arity == kVariadic) {
        if (static_cast<int>(inputs.size()) < kMinVariadicInputs) {
            return invalid_argument(std::string(op_name(op)) + " takes at least " + std::to_string(kMinVariadicInputs) +
                                    " inputs, got " + std::to_string(inputs.size()));
        }
    } else if (static_cast<int>(inputs.size()) != arity) {
        return invalid_argument(std::string(op_name(op)) + " takes " + std::to_string(arity) +
                                " inputs, got " + std::to_string(inputs.size()));
    }
    switch (op) {
        case OpKind::Add:
        case OpKind::Mul:       return infer_binary_elementwise(op, inputs);
        case OpKind::MatMul:    return infer_matmul(inputs, attrs);
        case OpKind::Gelu:      return infer_float_unary(op, inputs[0]);
        case OpKind::Softmax:   return infer_softmax(inputs[0], attrs);
        case OpKind::LayerNorm: return infer_layer_norm(inputs, attrs);
        case OpKind::Reshape:   return infer_reshape(inputs[0], attrs);
        case OpKind::Permute:   return infer_permute(inputs[0], attrs);
        case OpKind::Slice:     return infer_slice(inputs[0], attrs);
        case OpKind::Concat:    return infer_concat(inputs, attrs);
        case OpKind::Conv2d:    return infer_conv2d(inputs, attrs);
        case OpKind::Scale:     return infer_scale(inputs[0], attrs);
        case OpKind::SelfAttention: return infer_self_attention(inputs[0], attrs);
        case OpKind::Relu:      return infer_float_unary(op, inputs[0]);
        case OpKind::ConvTranspose2d: return infer_conv_transpose2d(inputs, attrs);
        case OpKind::Silu:      return infer_float_unary(op, inputs[0]);
        case OpKind::MaxPool2d: return infer_max_pool2d(inputs[0], attrs);
        case OpKind::UpsampleNearest2d: return infer_upsample_nearest2d(inputs[0], attrs);
    }
    return internal_error("infer_output: unhandled op");
}

}  // namespace dooms
