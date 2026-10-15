#pragma once

#include <vector>

#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"
#include "ir/op.h"

namespace dooms {

struct TypeInfo {
    Shape shape;
    DType dtype = DType::F32;

    bool operator==(const TypeInfo & o) const { return dtype == o.dtype && shape == o.shape; }
    bool operator!=(const TypeInfo & o) const { return !(*this == o); }
};

// Output type of `op` given its input types and attributes, or the reason the
// combination is invalid. Pure function; the single source of truth for op typing.
Result<TypeInfo> infer_output(OpKind op, const std::vector<TypeInfo> & inputs, const Attrs & attrs);

}  // namespace dooms
