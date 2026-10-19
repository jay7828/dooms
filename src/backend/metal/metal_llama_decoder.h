#pragma once
// A Llama-architecture text decoder on hand-written Metal kernels, built to cut the GPU calls per generated token: six fused kernels per layer
// (RMSNorm folded into the projections that follow it, the residual adds folded into the projections that produce them, SiLU(gate) x up in one kernel),
// greedy argmax on the GPU, and the next step's input token read from GPU memory, so several decode steps can be queued without waiting for the host.
// Activations are f32, weight matrices f16 (f32 accumulation), the KV cache f16 (what llama.cpp's Metal path does). Weights are copied into one GPU buffer at
// creation. This header exposes no Metal type; the GGUF loading lives in the pipeline layer (src/pipeline/native_llama.h). Apple Silicon only (ADR 0005).

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "backend/weight_ref.h"
#include "core/dtype.h"
#include "core/status.h"

namespace dooms {

// Hyperparameters, from the model file's metadata (never literals in model code).
struct LlamaDecoderConfig {
    int32_t n_layer = 0;
    int32_t n_embd = 0;      // residual width
    int32_t n_head = 0;      // query heads
    int32_t n_head_kv = 0;   // key/value heads (grouped-query attention: n_head % n_head_kv == 0)
    int32_t head_dim = 0;    // even, a multiple of 32, at most 256
    int32_t n_ff = 0;        // feed-forward width
    int32_t n_vocab = 0;
    float rope_base = 0.f;   // rotary frequency base
    bool rope_neox = false;  // rotary pairs: false = adjacent (2i, 2i+1), llama.cpp's "normal" (llama files); true = split halves (i, i + head_dim/2), "neox" (qwen2)
    float rms_eps = 0.f;
    int32_t n_ctx = 0;       // positions the KV cache holds
    int32_t n_batch = 0;     // largest number of positions one prefill call processes at once (longer inputs are split)
    // How generate() waits for each token: true polls the command buffer's status on the calling thread (one core busy while generating, like llama.cpp's
    // decode threads; no wake-up latency per token and the CPU stays clocked up); false sleeps in waitUntilCompleted. Not model metadata: a runtime choice.
    bool spin_wait = true;
    // Prefill matmul kernel: 0 picks by the number of tokens in the chunk; 1 forces the 16x16 tiles, 2 the 64-row x 32-token tiles (when the widths allow),
    // for measuring. Not model metadata.
    int32_t prefill_tile = 0;
};


struct LlamaLayerWeights {
    WeightRef attn_norm;          // [n_embd] f32
    WeightRef attn_q, attn_k, attn_v;  // f16 [n_head*head_dim | n_head_kv*head_dim][n_embd]; or attn_qkv set instead (rows q, then k, then v)
    WeightRef attn_qkv;
    WeightRef attn_q_bias, attn_k_bias, attn_v_bias;  // optional (qwen2): f32 vectors of the projections' output sizes, all three or none; or attn_qkv_bias
    WeightRef attn_qkv_bias;
    WeightRef attn_output;        // f16 [n_embd][n_head*head_dim]
    WeightRef ffn_norm;           // [n_embd] f32
    WeightRef ffn_gate, ffn_up;   // f16 [n_ff][n_embd]
    WeightRef ffn_down;           // f16 [n_embd][n_ff]
};

struct LlamaDecoderWeights {
    WeightRef token_embd;         // f16 [n_vocab][n_embd]
    WeightRef output_norm;        // [n_embd] f32
    WeightRef output;             // f16 [n_vocab][n_embd]; may be the same memory as token_embd (tied)
    std::vector<LlamaLayerWeights> layers;
};

// Not thread-safe: one caller at a time. Work queued by a call is complete before the next call reads or writes any state (each call starts by waiting),
// except that generate() returns without waiting for the decode steps it queued past the last delivered token.
class MetalLlamaDecoder {
public:
    //   Unsupported  no Metal device, a weight that is not f16 (matrices) / f32 or f16 (norms), a head_dim or context the kernels do not handle
    //   InvalidArgument  the weights do not have the shapes the config implies
    static Result<std::unique_ptr<MetalLlamaDecoder>> create(const LlamaDecoderConfig & config, const LlamaDecoderWeights & weights);
    ~MetalLlamaDecoder();
    MetalLlamaDecoder(const MetalLlamaDecoder &) = delete;
    MetalLlamaDecoder & operator=(const MetalLlamaDecoder &) = delete;

    const LlamaDecoderConfig & config() const;
    int32_t position() const;   // positions in the KV cache so far

    void reset();

    // Appends n positions given as input embeddings (n x n_embd floats, row-major). Afterwards the greedy token of the last position is known (pending_token)
    // and its logits can be read (logits).   InvalidArgument  null/empty/non-finite input     ResourceExhausted  the context would overflow
    Status eval_embeddings(const float * embeddings, int32_t n);
    // Appends tokens (looked up in the embedding table on the GPU). Same results as eval_embeddings of their embeddings.
    //   InvalidArgument  empty input or a token outside the vocabulary     ResourceExhausted  the context would overflow
    Status eval_tokens(const std::vector<int32_t> & tokens);

    // Greedy token of the last evaluated position.   InvalidArgument  nothing evaluated yet
    Result<int32_t> pending_token();
    // Logits of the last evaluated position (n_vocab floats), for parity checks.   InvalidArgument  nothing evaluated yet
    Result<std::vector<float>> logits();

    // Greedy generation from the pending token: on_token receives the pending token first, then each next one, in order; return false to stop. Generation also
    // stops after max_tokens tokens. Up to `depth` decode steps are queued on the GPU ahead of the token the host is reading (the step that consumes token i
    // reads it from GPU memory), so the host never stalls the GPU; steps queued past the stopping point are discarded. On return position() counts the tokens
    // that were fed back (every delivered token except the last), as if they had been evaluated one by one.
    //   InvalidArgument  nothing evaluated yet, max_tokens < 1, depth < 1     ResourceExhausted  max_tokens would overflow the context
    Status generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token);

    // GPU time of one forward pass over `rows` new positions (1: a decode step; more: a prefill chunk of token ids 0..rows-1) per kernel kind, each dispatch
    // in its own command buffer (so the sum exceeds a pipelined pass by the per-buffer overhead). Leaves the cache contents past position() undefined and
    // does not advance position().   InvalidArgument  nothing evaluated yet (rows == 1), rows outside [1, n_batch]     ResourceExhausted  the cache is too full
    struct KernelTime { const char * kind; int32_t dispatches; double gpu_ms; };
    Result<std::vector<KernelTime>> profile_pass(int32_t rows = 1);

private:
    struct Impl;
    explicit MetalLlamaDecoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
