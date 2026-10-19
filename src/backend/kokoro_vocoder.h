#pragma once
// Kokoro-82M's vocoder, whichever device runs it: the interface the pipeline (src/pipeline/kokoro_pipeline.h) drives, and the tensors and config both
// implementations take. Implementations: backend/metal/metal_kokoro_vocoder.h (the GPU, the default) and backend/cpu/cpu_kokoro_vocoder.h (the CPU fallback).

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/status.h"

namespace dooms {

struct KokoroTensor {
    const void * data = nullptr;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;   // outermost first (PyTorch's order)
};
using KokoroTensors = std::map<std::string, KokoroTensor>;   // the vocoder's "dec.*" tensors of kokoro.gguf (docs/formats/kokoro-gguf.md)

struct KokoroVocoderConfig {
    int32_t hidden = 0;                       // the decoder's input width (asr features): 512
    int32_t style_dim = 0;                    // 128
    int32_t n_fft = 0;                        // the generator's iSTFT size: its output has n_fft + 2 channels (20 -> 22)
    std::vector<int32_t> upsample_rates;      // generator upsampling (10, 6)
    std::vector<int32_t> upsample_kernels;    // (20, 12)
    std::vector<int32_t> resblock_kernels;    // (3, 7, 11)
    int32_t initial_channel = 0;              // 512
    bool spin_wait = true;
    bool tap_major = true;                    // the tap-major fast path for convolutions whose input width is a multiple of the step depth (false: generic only)
};

class KokoroVocoder {
public:
    virtual ~KokoroVocoder() = default;
    virtual const char * device() const = 0;   // "gpu" or "cpu"
    // frames F >= 1. asr [hidden][F], f0 and n [2F], style [style_dim] (the voice's first half), har [n_fft + 2][har_len] with har_len = har_length(F)
    // (magnitudes then phases of the harmonic source's STFT). out: [n_fft + 2][har_len] row-major.
    //   InvalidArgument  sizes that do not fit, non-finite input     ResourceExhausted  buffers could not be allocated
    virtual Status run(const float * asr, int32_t frames, const float * f0, const float * n, const float * style, const float * har, std::vector<float> * out) = 0;
    virtual int32_t har_length(int32_t frames) const = 0;
    // GPU time per kernel kind of the last run with profiling on; empty where the device has none.
    virtual void set_profiling(bool) {}
    virtual const std::vector<std::pair<std::string, double>> & profile() const { static const std::vector<std::pair<std::string, double>> none; return none; }
};

}  // namespace dooms
