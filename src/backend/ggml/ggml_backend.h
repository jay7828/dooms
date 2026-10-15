#pragma once
// ggml backend (phase 1): lowers the IR to ggml graphs. Together with the llama
// bridge, the only code allowed to include ggml headers.

#include <memory>
#include <string>

#include "backend/backend.h"

namespace dooms {

struct GgmlBackendOptions {
    // "auto" = best available (GPU if present, else CPU), "cpu", or "gpu".
    std::string device = "auto";
    // CPU thread count; 0 keeps ggml's default. Ignored on non-CPU devices.
    int n_threads = 0;
    // Lower SelfAttention to ggml's fused flash attention when the device supports the shape (strided
    // views of the packed qkv, no score tensor). false always uses the unfused chain; used for A/B comparison.
    bool fused_attention = true;
    // Hand K and V to the fused attention as f16 (Q stays f32) on a GPU: a measured ~6% faster forward on the ViT-H
    // backbone (bench/PROFILE-2026-10-08.md, part 2) with no measurable change in accuracy
    // (the GPU matmul already rounds its operand tiles to half). Ignored by the CPU backend, which keeps f32.
    bool attention_kv_f16 = true;
    // How convolutions are computed:
    //   "im2col"      f32 im2col buffer + matmul (exact f32 inputs on the CPU; the default)
    //   "im2col_f16"  the im2col buffer in half precision (ggml's own conv_2d; the GPU matmul rounds its tiles to half anyway, so on a GPU only
    //                 the memory traffic changes; on the CPU the activations are rounded)
    //   "direct"      ggml's direct convolution kernel (no im2col buffer)
    // See bench/BASELINES.md for the measured trade-offs on the detector's shapes.
    std::string conv_algorithm = "im2col";
    // CPU device only: run the large matrix multiplications on Apple's Accelerate (the AMX matrix units) through ggml's BLAS device, everything else on the CPU. Measured
    // 1.0-1.4 TFLOPS on f32 weights against 0.28 for ggml's own CPU kernels (bench/BASELINES.md). Silently off when the BLAS device is not built in. Weights stay f32 for the
    // speed-up (f16 weights are converted to f32 for every call, which makes them slower than ggml's own f16 kernels).
    bool cpu_blas = true;
};

Result<std::unique_ptr<Backend>> create_ggml_backend(const GgmlBackendOptions & options);

}  // namespace dooms
