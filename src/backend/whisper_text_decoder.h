#pragma once
// Whisper's text decoder, whichever device runs it: the interface the pipeline (src/pipeline/whisper_pipeline.h) drives, with the configuration and weights both
// implementations take. Implementations: backend/metal/metal_whisper_decoder.h (the GPU, the default) and backend/cpu/cpu_whisper_decoder.h (the CPU fallback).

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "backend/weight_ref.h"
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
    int32_t cross_splits = 0;        // cross-attention: groups of audio positions run as separate threadgroups (0: about 64 threadgroups per token)
    // LayerNorm of a decode step's input: 1 inside every threadgroup of the projection that follows, 0 one layernorm dispatch first, -1 automatic (folded in for
    // rows up to 512 wide: measured faster for whisper-tiny's 384, slower for turbo's 1280; src/backend/metal/CLAUDE.md).
    int32_t fused_layernorm = -1;
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

// Greedy decoding with Whisper's suppression masks. Not thread-safe.
class WhisperTextDecoder {
public:
    virtual ~WhisperTextDecoder() = default;
    virtual const WhisperDecoderConfig & config() const = 0;
    virtual const char * device() const = 0;   // "gpu" or "cpu"
    virtual int32_t position() const = 0;      // text positions evaluated since the last set_audio()

    // Token ids that may never be generated, and those that may not be the first generated token (both stay suppressed until changed).
    //   InvalidArgument  an id outside the vocabulary
    virtual Status set_suppression(const std::vector<int32_t> & always, const std::vector<int32_t> & at_begin) = 0;
    // A new audio window: encoder states (n_audio_ctx x n_state floats, row-major) become the cross-attention keys and values; the text position restarts at 0.
    //   InvalidArgument  null or non-finite input
    virtual Status set_audio(const float * encoder_states) = 0;
    // Appends one token at the current position. `first_step`: its greedy successor is the first generated token (the begin suppression applies).
    //   InvalidArgument  a token outside the vocabulary     ResourceExhausted  the text context is full
    virtual Status eval_token(int32_t token, bool first_step) = 0;
    // The prompt of a window: every token appended, the last one's greedy successor is the first generated token (eval_token over the tokens, with first_step
    // on the last; same results). InvalidArgument  empty, or a token outside the vocabulary     ResourceExhausted  the text context would overflow
    virtual Status eval_prompt(const std::vector<int32_t> & tokens) = 0;
    virtual Result<int32_t> pending_token() = 0;              // the masked greedy successor of the last evaluated token
    virtual Result<std::vector<float>> logits() = 0;          // raw logits of the last evaluated position (before the mask), for parity checks
    // Greedy generation from the pending token: on_token gets each token (the pending one first); returning false stops. At most max_tokens tokens; `depth`
    // steps may be queued ahead (a hint: the CPU computes one at a time). The always-mask applies.
    virtual Status generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token) = 0;
    // One decode step's time per kernel kind (bench_whisper --profile); Unsupported where the device has no such breakdown.
    virtual Result<std::vector<std::pair<std::string, double>>> profile_step() { return unsupported(std::string("no kernel profile on the ") + device()); }
};

}  // namespace dooms
