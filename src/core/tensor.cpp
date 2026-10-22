#include "core/tensor.h"

#include <limits>

namespace dooms {

namespace {

// Byte strides of a contiguous row-major layout, or an error if sizes overflow.
Result<std::array<int64_t, kMaxDims>> contiguous_strides(const Shape & shape, DType dtype) {
    std::array<int64_t, kMaxDims> st{};
    int64_t elem = static_cast<int64_t>(dtype_size(dtype));
    int64_t acc = elem;
    for (int i = shape.rank() - 1; i >= 0; i--) {
        st[i] = acc;
        if (acc > std::numeric_limits<int64_t>::max() / shape.dim(i)) {
            return resource_exhausted("tensor byte size overflows int64");
        }
        acc *= shape.dim(i);
    }
    return st;
}

}  // namespace

Result<TensorView> TensorView::create(void * data, const Shape & shape, DType dtype) {
    if (dtype_is_quantized(dtype)) return unsupported(std::string("tensor views take plain element types, not ") + dtype_name(dtype));
    auto st = contiguous_strides(shape, dtype);
    if (!st.is_ok()) return st.status();
    return create_strided(data, shape, dtype, st.value().data(), shape.rank());
}

Result<TensorView> TensorView::create_strided(void * data, const Shape & shape, DType dtype,
                                              const int64_t * strides_bytes, int n_strides) {
    if (data == nullptr) return invalid_argument("tensor data pointer is null");
    if (dtype_is_quantized(dtype)) return unsupported(std::string("tensor views take plain element types, not ") + dtype_name(dtype));
    if (n_strides != shape.rank()) {
        return invalid_argument("got " + std::to_string(n_strides) + " strides for rank " +
                                std::to_string(shape.rank()));
    }
    if (shape.rank() > 0 && strides_bytes == nullptr) {
        return invalid_argument("null strides pointer");
    }
    const int64_t elem = static_cast<int64_t>(dtype_size(dtype));
    TensorView v;
    v.data_ = data;
    v.shape_ = shape;
    v.dtype_ = dtype;
    int64_t extent = elem;  // offset of the end of the last element
    for (int i = 0; i < shape.rank(); i++) {
        const int64_t s = strides_bytes[i];
        if (s <= 0 || s % elem != 0) {
            return invalid_argument("stride " + std::to_string(i) + " is " + std::to_string(s) +
                                    " bytes; must be positive and a multiple of " +
                                    std::to_string(elem));
        }
        const int64_t steps = shape.dim(i) - 1;
        if (steps > 0 && s > (std::numeric_limits<int64_t>::max() - extent) / steps) {
            return resource_exhausted("tensor span overflows int64");
        }
        extent += steps * s;
        v.strides_[i] = s;
    }
    v.span_bytes_ = extent;
    return v;
}

int64_t TensorView::stride_bytes(int i) const {
    DOOMS_CHECK(i >= 0 && i < shape_.rank(), "TensorView::stride_bytes index out of range");
    return strides_[i];
}

bool TensorView::is_contiguous() const {
    auto st = contiguous_strides(shape_, dtype_);
    if (!st.is_ok()) return false;
    for (int i = 0; i < shape_.rank(); i++) {
        if (strides_[i] != st.value()[i]) return false;
    }
    return true;
}

}  // namespace dooms
