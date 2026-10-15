#pragma once
// The language-model half of a vision-language model, run by llama.cpp (Metal GPU): a GGUF text decoder that can be fed tokens AND precomputed
// embeddings (the image tokens come from the vision tower, not from the vocabulary). Everything that touches llama.h lives in text_decoder.cpp.
// Evaluation appends to a KV cache; each call returns after the model has seen the new positions, with the logits of the last one available.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"

namespace dooms {

struct TextDecoderOptions {
    int32_t n_ctx = 2048;       // positions the KV cache can hold (prompt + image tokens + generated tokens)
    int32_t n_batch = 512;      // largest chunk sent to the model in one call (longer inputs are split)
    int32_t n_gpu_layers = -1;  // -1: every layer on the GPU; 0: CPU only
    int32_t n_threads = 0;      // CPU threads; 0 = llama.cpp's default
};

// How the next token is chosen from the logits. temperature 0 is greedy (deterministic); otherwise tokens are drawn with a seeded generator, so the same
// seed on the same inputs gives the same text.
struct SamplerOptions {
    float temperature = 0.f;       // >= 0
    int32_t top_k = 40;            // 0 = off (used when temperature > 0)
    float top_p = 0.95f;           // in (0, 1]; 1 = off (used when temperature > 0)
    float repeat_penalty = 1.f;    // > 0; 1 = off; applied to the last 64 tokens
    uint32_t seed = 0;
};

struct ChatMessage {
    std::string role;     // "system", "user" or "assistant"
    std::string content;
};

class TextDecoder {
public:
    //   NotFound  no such file     DataLoss  not a GGUF text model llama.cpp can load     InvalidArgument  bad options
    static Result<TextDecoder> load(const std::string & path, const TextDecoderOptions & options = TextDecoderOptions());

    TextDecoder(TextDecoder &&) noexcept;
    TextDecoder & operator=(TextDecoder &&) noexcept;
    ~TextDecoder();

    int64_t embedding_dim() const;   // width of the vectors eval_embeddings() takes
    int64_t vocab_size() const;
    int32_t position() const;        // positions in the KV cache so far
    int32_t context_size() const;

    // `parse_special`: strings like "<image>" or "<|im_start|>" become their special tokens. `add_bos` adds the vocabulary's BOS token (off for
    // chat prompts that already start with it). InvalidArgument for empty text.
    Result<std::vector<int32_t>> tokenize(const std::string & text, bool parse_special, bool add_bos = false) const;
    // The text a token stands for (special tokens included, as their names).
    std::string piece(int32_t token) const;
    bool is_end_of_generation(int32_t token) const;

    // Forget everything (new conversation).
    void reset();

    // Append tokens / embeddings at the current position. The logits of the LAST appended position become available to greedy_next() / logits().
    //   InvalidArgument  empty input, a token outside the vocabulary, non-finite embeddings
    //   ResourceExhausted  the KV cache would overflow     Internal  llama.cpp failed
    Status eval_tokens(const std::vector<int32_t> & tokens);
    // embeddings: n_positions x embedding_dim() floats, row-major.
    Status eval_embeddings(const float * embeddings, int64_t n_positions);

    // The model's input embeddings of `tokens` (what eval_tokens would feed), n_tokens x embedding_dim() floats, read from the GGUF's token embedding table. With them a prompt
    // that interleaves tokens and image embeddings goes through the model in ONE call (eval_embeddings) instead of one per segment.
    //   Unsupported  the table is not f32/f16/bf16 (quantized models: use eval_tokens)     InvalidArgument  empty input or a token outside the vocabulary
    Result<std::vector<float>> embed_tokens(const std::vector<int32_t> & tokens) const;

    // Starts a new sampling session (call before each generation: it clears the repetition history).  InvalidArgument for options out of range.
    Status start_sampling(const SamplerOptions & options);
    // The next token drawn from the last logits with the session's sampler; the token is recorded for the repetition penalty. InvalidArgument if nothing was
    // evaluated yet or start_sampling() was not called.
    Result<int32_t> sample_next();

    // The prompt text for a conversation, with the model's own chat template (from its GGUF). add_generation_prompt ends it with the assistant's turn header.
    //   Unsupported  the model has no chat template or llama.cpp does not know it     InvalidArgument  no messages or an empty role
    Result<std::string> apply_chat_template(const std::vector<ChatMessage> & messages, bool add_generation_prompt = true) const;

    // Argmax of the last logits. FailedPrecondition-like: InvalidArgument if nothing was evaluated yet.
    Result<int32_t> greedy_next() const;
    // Copy of the last position's logits (vocab_size() values), for parity checks.
    Result<std::vector<float>> logits() const;

private:
    struct Impl;
    explicit TextDecoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
