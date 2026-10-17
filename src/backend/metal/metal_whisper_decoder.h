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
#include <vector>

#include "backend/metal/metal_llama_decoder.h"   // WeightRef
#include "core/status.h"

namespace dooms {

struct WhisperDecoderConfig {
    int32_t n_layer = 0;
    int32_t n_state = 0;      // d_model
    int32_t n_head = 0;       // head_dim = n_state / n_head: 64 or 128
    int32_t n_ff = 0;
    int32_t n_vocab = 0;
    int32_t n_text_ctx = 0;   // positions of the learned position table (448 for every released Whisper)
    int32_t n_audio_ctx = 0;  // encoder positions per window (1500)
    float ln_eps = 0.f;
    bool spin_wait = true;    // as LlamaDecoderConfig::spin_wait
};

struct WhisperLayerWeights {
    WeightRef attn_ln_w, attn_ln_b, attn_q_w, attn_q_b, attn_k_w, attn_v_w, attn_v_b, attn_out_w, attn_out_b;
    WeightRef cross_ln_w, cross_ln_b, cross_q_w, cross_q_b, cross_k_w, cross_v_w, cross_v_b, cross_out_w, cross_out_b;
    WeightRef mlp_ln_w, mlp_ln_b, fc1_w, fc1_b, fc2_w, fc2_b;
};

struct WhisperDecoderWeights {
    WeightRef token_embd;     // f16 [n_vocab][n_state]; also the output projection (tied)
    WeightRef pos_embd;       // f32 [n_text_ctx][n_state]
    WeightRef ln_w, ln_b;     // final LayerNorm
    std::vector<WhisperLayerWeights> layers;
};

// Not thread-safe; work queued by a call is complete before the next call touches state, except generate()'s steps past the last delivered token.
class MetalWhisperDecoder {
public:
    //   Unsupported  no Metal device, head_dim other than 64/128, widths the kernels cannot take     InvalidArgument  weights of the wrong shape or dtype
    static Result<std::unique_ptr<MetalWhisperDecoder>> create(const WhisperDecoderConfig & config, const WhisperDecoderWeights & weights);
    ~MetalWhisperDecoder();
    MetalWhisperDecoder(const MetalWhisperDecoder &) = delete;
    MetalWhisperDecoder & operator=(const MetalWhisperDecoder &) = delete;

    const WhisperDecoderConfig & config() const;
    int32_t position() const;   // text positions evaluated since the last set_audio()

    // Token ids that may never be generated, and those that may not be the first generated token (both stay suppressed until changed).
    //   InvalidArgument  an id outside the vocabulary
    Status set_suppression(const std::vector<int32_t> & always, const std::vector<int32_t> & at_begin);

    // A new audio window: encoder states (n_audio_ctx x n_state floats, row-major) become the cross-attention keys and values; the text position restarts at 0.
    //   InvalidArgument  null or non-finite input
    Status set_audio(const float * encoder_states);

    // Appends one token at the current position. `first_step`: its greedy successor is the first generated token (the begin suppression applies).
    //   InvalidArgument  a token outside the vocabulary     ResourceExhausted  the text context is full
    Status eval_token(int32_t token, bool first_step);

    Result<int32_t> pending_token();              // the masked greedy successor of the last evaluated token
    Result<std::vector<float>> logits();          // raw logits of the last evaluated position (before the mask), for parity checks

    // Greedy generation from the pending token, steps queued on the GPU like MetalLlamaDecoder::generate (same semantics; the always-mask applies).
    Status generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token);

private:
    struct Impl;
    explicit MetalWhisperDecoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
