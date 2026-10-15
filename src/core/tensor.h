#pragma once

#include <array>
#include <cstdint>

#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"

namespace dooms {

// Non-owning view of memory as a tensor. The caller keeps the buffer alive for
// as long as the view is used. Strides are in bytes, positive, and multiples of
// the element size (no broadcasting strides).
class TensorView {
public:
    // Contiguous row-major view. data must be non-null.
    static Result<TensorView> create(void * data, const Shape & shape, DType dtype);
    // View with explicit byte strides, one per dimension.
    static Result<TensorView> create_strided(void * data, const Shape & shape, DType dtype,
                                             const int64_t * strides_bytes, int n_strides);

    void * data() const { return data_; }
    const Shape & shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    int64_t stride_bytes(int i) const;  // precondition: 0 <= i < rank
    bool is_contiguous() const;
    // Bytes of memory spanned by the view (last element's end offset).
    int64_t span_bytes() const { return span_bytes_; }

private:
    TensorView() = default;
    void * data_ = nullptr;
    Shape shape_;
    DType dtype_ = DType::F32;
    std::array<int64_t, kMaxDims> strides_{};
    int64_t span_bytes_ = 0;
};

}  // namespace dooms
