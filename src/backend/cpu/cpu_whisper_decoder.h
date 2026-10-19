#pragma once
// Whisper's text decoder on the CPU: the fallback when no Metal device can run it (or when asked for: WhisperOptions::device = "cpu"). Same interface and the
// same arithmetic as the GPU decoder, in f32 (weights converted from f16 at creation) with Accelerate's BLAS: per token a matrix-vector product per projection,
// self-attention over the cached positions, cross-attention over the window's keys/values (computed once per window as matrix products). One token per step.
#include <memory>

#include "backend/whisper_text_decoder.h"

namespace dooms {

class CpuWhisperDecoder : public WhisperTextDecoder {
public:
    //   InvalidArgument  weights of the wrong shape or dtype, an invalid config
    static Result<std::unique_ptr<CpuWhisperDecoder>> create(const WhisperDecoderConfig & config, const WhisperDecoderWeights & weights);
    ~CpuWhisperDecoder() override;
    CpuWhisperDecoder(const CpuWhisperDecoder &) = delete;
    CpuWhisperDecoder & operator=(const CpuWhisperDecoder &) = delete;

    const WhisperDecoderConfig & config() const override;
    const char * device() const override { return "cpu"; }
    int32_t position() const override;
    Status set_suppression(const std::vector<int32_t> & always, const std::vector<int32_t> & at_begin) override;
    Status set_audio(const float * encoder_states) override;
    Status eval_token(int32_t token, bool first_step) override;
    Status eval_prompt(const std::vector<int32_t> & tokens) override;
    Result<int32_t> pending_token() override;
    Result<std::vector<float>> logits() override;
    Status generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token) override;

private:
    struct Impl;
    explicit CpuWhisperDecoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
