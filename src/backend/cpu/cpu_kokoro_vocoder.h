#pragma once
// Kokoro-82M's vocoder on the CPU: the fallback when no Metal device can run it (or when asked for: KokoroOptions::device = "cpu"). The same layers and fused
// order as the Metal vocoder (backend/metal/metal_kokoro_vocoder.h): the AdaIN + activation in front of each convolution applied once per element, every
// convolution as column blocks times the weights with Accelerate's BLAS (transposed ones one output phase at a time), the residual / shortcut / average in the
// epilogue. f32 throughout (the f16 kernels are converted at creation).
#include <memory>

#include "backend/kokoro_vocoder.h"

namespace dooms {

class CpuKokoroVocoder : public KokoroVocoder {
public:
    //   Unsupported  an architecture this implementation does not cover     InvalidArgument  missing tensors or wrong shapes/dtypes
    static Result<std::unique_ptr<CpuKokoroVocoder>> create(const KokoroVocoderConfig & config, const KokoroTensors & tensors);
    ~CpuKokoroVocoder() override;
    CpuKokoroVocoder(const CpuKokoroVocoder &) = delete;
    CpuKokoroVocoder & operator=(const CpuKokoroVocoder &) = delete;

    const char * device() const override { return "cpu"; }
    Status run(const float * asr, int32_t frames, const float * f0, const float * n, const float * style, const float * har, std::vector<float> * out) override;
    int32_t har_length(int32_t frames) const override;

private:
    struct Impl;
    explicit CpuKokoroVocoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
