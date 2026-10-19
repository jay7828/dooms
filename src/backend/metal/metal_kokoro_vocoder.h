#pragma once
// Kokoro-82M's vocoder (StyleTTS 2 decoder + iSTFTNet generator) on the engine's own Metal kernels (kokoro_kernels.h): every convolution is one implicit-GEMM
// kernel (half operands, f32 accumulation) whose input loader applies the preceding AdaIN (instance-norm statistics folded with the style's gamma/beta into one
// per-channel affine map) and the activation (leaky ReLU or Snake), and whose epilogue adds the bias, the residual and the shortcut. Transposed convolutions run as
// one such convolution per output phase. Input: the frame-rate features, F0/energy curves, the style and the harmonic source's spectrum; output: the log-magnitude
// and phase-argument channels the host turns into audio (iSTFT). The model is fixed (not an IR graph): the layer list follows Kokoro's architecture, sizes come
// from the weights and are checked at create. GGUF loading lives in the pipeline layer (src/pipeline/kokoro_pipeline.h). Apple Silicon only (ADR 0005).

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/kokoro_vocoder.h"

namespace dooms {

// Not thread-safe.
class MetalKokoroVocoder : public KokoroVocoder {
public:
    //   Unsupported  no Metal device, an architecture this implementation does not cover     InvalidArgument  missing tensors or wrong shapes/dtypes
    static Result<std::unique_ptr<MetalKokoroVocoder>> create(const KokoroVocoderConfig & config, const KokoroTensors & tensors);
    ~MetalKokoroVocoder() override;
    const char * device() const override { return "gpu"; }
    MetalKokoroVocoder(const MetalKokoroVocoder &) = delete;
    MetalKokoroVocoder & operator=(const MetalKokoroVocoder &) = delete;

    // frames F >= 1. asr [hidden][F], f0 and n [2F], style [style_dim] (the voice's first half), har [n_fft + 2][har_len] with har_len = 2F * prod(rates) + 1
    // (magnitudes then phases of the harmonic source's STFT). out: [n_fft + 2][har_len] row-major.
    //   InvalidArgument  sizes that do not fit, non-finite input     ResourceExhausted  buffers could not be allocated
    Status run(const float * asr, int32_t frames, const float * f0, const float * n, const float * style, const float * har, std::vector<float> * out) override;

    int32_t har_length(int32_t frames) const override;
    // GPU time per kernel kind of the last run with profiling on (every dispatch in its own command buffer); empty otherwise.
    void set_profiling(bool on) override;
    const std::vector<std::pair<std::string, double>> & profile() const override;

private:
    struct Impl;
    explicit MetalKokoroVocoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
