#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/status.h"

namespace dooms {

// Element types. The quantized ones are ggml's block formats (a block of `dtype_block_elems` elements in `dtype_block_bytes` bytes, rows a whole number
// of blocks), read from GGUF files for the Metal text decoder; tensors, IR graphs and the backends take the plain types only.
enum class DType { F32, F16, BF16, I32, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K };

size_t dtype_size(DType t);        // bytes per element; plain types only (a quantized type is a programming error)
bool dtype_is_quantized(DType t);
int64_t dtype_block_elems(DType t);   // 1 for the plain types
int64_t dtype_block_bytes(DType t);   // dtype_size for the plain types
// Bytes of n elements: InvalidArgument when n is not a whole number of blocks.
Result<uint64_t> dtype_bytes(DType t, int64_t n);
const char * dtype_name(DType t);  // "f32", "f16", "bf16", "i32"
bool dtype_is_float(DType t);
Result<DType> dtype_from_name(std::string_view name);  // inverse of dtype_name

}  // namespace dooms
