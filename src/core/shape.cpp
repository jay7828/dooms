#include "core/shape.h"

#include <limits>

namespace dooms {

Result<Shape> Shape::create(std::initializer_list<int64_t> dims) {
    return create(dims.begin(), static_cast<int>(dims.size()));
}

Result<Shape> Shape::create(const int64_t * dims, int rank) {
    if (rank < 0 || rank > kMaxDims) {
        return invalid_argument("rank " + std::to_string(rank) + " outside [0, " +
                                std::to_string(kMaxDims) + "]");
    }
    if (rank > 0 && dims == nullptr) {
        return invalid_argument("null dims pointer with rank " + std::to_string(rank));
    }
    Shape s;
    s.rank_ = rank;
    int64_t n = 1;
    for (int i = 0; i < rank; i++) {
        if (dims[i] < 1) {
            return invalid_argument("dimension " + std::to_string(i) + " is " +
                                    std::to_string(dims[i]) + "; dimensions must be >= 1");
        }
        if (n > std::numeric_limits<int64_t>::max() / dims[i]) {
            return resource_exhausted("element count overflows int64 at dimension " + std::to_string(i));
        }
        n *= dims[i];
        s.dims_[i] = dims[i];
    }
    s.numel_ = n;
    return s;
}

int64_t Shape::dim(int i) const {
    DOOMS_CHECK(i >= 0 && i < rank_, "Shape::dim index out of range");
    return dims_[i];
}

bool Shape::operator==(const Shape & o) const {
    if (rank_ != o.rank_) return false;
    for (int i = 0; i < rank_; i++) {
        if (dims_[i] != o.dims_[i]) return false;
    }
    return true;
}

std::string Shape::to_string() const {
    std::string out = "[";
    for (int i = 0; i < rank_; i++) {
        if (i) out += ", ";
        out += std::to_string(dims_[i]);
    }
    return out + "]";
}

}  // namespace dooms
