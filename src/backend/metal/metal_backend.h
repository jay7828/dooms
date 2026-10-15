#pragma once
// Native Metal backend for convolutional graphs: the ops of the detector and RefineNet graphs (Conv2d, Add/Mul with broadcast, Relu, Silu, Scale, Reshape, Permute,
// Slice, Concat, MaxPool2d, UpsampleNearest2d) run on hand-written kernels, the convolution as an implicit GEMM on the GPU's simdgroup matrix units (no im2col buffer,
// activations f32 in memory, half-precision operand tiles with f32 accumulation like ggml's Metal matmul). Graphs with other ops are rejected by supports() with
// Unsupported; use the ggml backend for those. Apple Silicon only (ADR 0005). See CLAUDE.md in this directory for the design and measurements.

#include <memory>

#include "backend/backend.h"

namespace dooms {

struct MetalBackendOptions {
    // Fold "Conv2d -> Add(per-channel bias) -> Silu|Relu" into the convolution kernel's epilogue when the intermediate values have no other use.
    bool fuse_epilogue = true;
};

//   Unsupported  no Metal device
Result<std::unique_ptr<Backend>> create_metal_backend(const MetalBackendOptions & options = MetalBackendOptions());

}  // namespace dooms
