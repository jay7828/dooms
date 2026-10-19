#pragma once
// Whisper's text decoder on the engine's Metal kernels (llama_kernels.h): per generated token, 8 kernels per layer (LayerNorm folded into the projection that
// follows it, biases and residual adds in the projection epilogues, GELU in fc1's epilogue, self-attention with the new key/value written to the cache by the
// attention kernel itself, cross-attention over the audio window), then the final LayerNorm + vocabulary projection and the masked argmax on the GPU. The
// cross-attention keys and values of an audio window are computed once, by the prompt matmul kernel. Learned positions (no rotary). Greedy decoding only;
// the suppressed tokens (Whisper's suppress lists) are a -inf mask applied inside the argmax. Activations f32, weight matrices f16, caches f16.
// The GGUF loading lives in the pipeline layer (src/pipeline/whisper_pipeline.h). Apple Silicon only (ADR 0005).

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/whisper_text_decoder.h"
#include "core/status.h"

namespace dooms {

// Not thread-safe; work queued by a call is complete before the next call touches state, except generate()'s steps past the last delivered token.
class MetalWhisperDecoder : public WhisperTextDecoder {
public:
    //   Unsupported  no Metal device, head_dim other than 64/128, widths the kernels cannot take     InvalidArgument  weights of the wrong shape or dtype
    static Result<std::unique_ptr<MetalWhisperDecoder>> create(const WhisperDecoderConfig & config, const WhisperDecoderWeights & weights);
    ~MetalWhisperDecoder() override;
    const char * device() const override { return "gpu"; }
    MetalWhisperDecoder(const MetalWhisperDecoder &) = delete;
    MetalWhisperDecoder & operator=(const MetalWhisperDecoder &) = delete;

    const WhisperDecoderConfig & config() const override;
    int32_t position() const override;   // text positions evaluated since the last set_audio()

    // Token ids that may never be generated, and those that may not be the first generated token (both stay suppressed until changed).
    //   InvalidArgument  an id outside the vocabulary
    Status set_suppression(const std::vector<int32_t> & always, const std::vector<int32_t> & at_begin) override;

    // A new audio window: encoder states (n_audio_ctx x n_state floats, row-major) become the cross-attention keys and values; the text position restarts at 0.
    //   InvalidArgument  null or non-finite input
    Status set_audio(const float * encoder_states) override;

    // Appends one token at the current position. `first_step`: its greedy successor is the first generated token (the begin suppression applies).
    //   InvalidArgument  a token outside the vocabulary     ResourceExhausted  the text context is full
    Status eval_token(int32_t token, bool first_step) override;

    // The prompt of a window: every token appended, the last one's greedy successor is the first generated token (eval_token over the tokens, with first_step on
    // the last). Up to 8 tokens per pass, each weight read once for all of them; the results equal that loop's to rounding.   InvalidArgument  empty, or a token outside the vocabulary     ResourceExhausted  the text context would overflow
    Status eval_prompt(const std::vector<int32_t> & tokens) override;

    Result<int32_t> pending_token() override;              // the masked greedy successor of the last evaluated token
    Result<std::vector<float>> logits() override;          // raw logits of the last evaluated position (before the mask), for parity checks

    // Greedy generation from the pending token, steps queued on the GPU like MetalLlamaDecoder::generate (same semantics; the always-mask applies).
    Status generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token) override;

    // One decode step after the pending token with every dispatch in its own command buffer: (kernel kind, GPU ms) in dispatch order. For bench_whisper --profile.
    Result<std::vector<std::pair<std::string, double>>> profile_step() override;

private:
    struct Impl;
    explicit MetalWhisperDecoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
