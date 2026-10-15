#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "core/status.h"

namespace dooms {

// Maximum tensor rank. ViT, transformer decoders and conv2d fit in 4
// (batch/channels/height/width or batch/heads/seq/dim).
constexpr int kMaxDims = 4;

// Row-major shape, outermost dimension first. Every dimension is >= 1 and the
// element count fits in int64_t; both are checked at creation, so a Shape
// obtained from create() is always valid. A rank-0 shape is a scalar (1 element).
class Shape {
public:
    Shape() = default;  // scalar

    static Result<Shape> create(std::initializer_list<int64_t> dims);
    static Result<Shape> create(const int64_t * dims, int rank);

    int rank() const { return rank_; }
    int64_t dim(int i) const;  // precondition: 0 <= i < rank()
    const int64_t * data() const { return dims_.data(); }
    int64_t numel() const { return numel_; }

    bool operator==(const Shape & o) const;
    bool operator!=(const Shape & o) const { return !(*this == o); }
    std::string to_string() const;  // "[2, 3, 4]", scalar is "[]"

private:
    std::array<int64_t, kMaxDims> dims_{};
    int rank_ = 0;
    int64_t numel_ = 1;
};

}  // namespace dooms
