#pragma once

#include <cstddef>
#include <string_view>

#include "core/status.h"

namespace dooms {

// Element types. Quantized block types are added when a model needs them
// (see docs/PLAN.md M6), together with block-size metadata.
enum class DType { F32, F16, BF16, I32 };

size_t dtype_size(DType t);        // bytes per element
const char * dtype_name(DType t);  // "f32", "f16", "bf16", "i32"
bool dtype_is_float(DType t);
Result<DType> dtype_from_name(std::string_view name);  // inverse of dtype_name

}  // namespace dooms
