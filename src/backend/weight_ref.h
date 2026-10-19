#pragma once
// A weight tensor in host memory, as the fixed-model decoders take it (Metal and CPU alike).
#include <cstdint>

#include "core/dtype.h"

namespace dooms {

// A tensor in host memory (e.g. a mapped GGUF file); copied at creation, not referenced afterwards. Matrices are row-major [rows][cols] with cols the input width
// (ggml's ne0), so row r of a projection produces output r.
struct WeightRef {
    const void * data = nullptr;
    DType dtype = DType::F32;
    int64_t rows = 0, cols = 0;   // vectors: rows = 1
};

}  // namespace dooms
