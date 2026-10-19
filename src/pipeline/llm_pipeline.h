#pragma once
// A text language model from a GGUF file: tokenization, the model's own chat template, seeded sampling, streaming and stop strings. The engine's own Metal
// decoder (backend/metal/metal_llama_decoder.h) runs llama/qwen2 files with f16 weights; everything else (other architectures, quantized files) runs on
// llama.cpp. llama.cpp always provides the vocabulary and the sampler (so sampling behaves the same on both). This is the engine's general LLM entry point.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"
#include "models/vlm/text_decoder.h"

namespace dooms {

struct LlmOptions {
    TextDecoderOptions text;  // n_ctx (prompt + answer), n_batch, n_gpu_layers, n_threads
    // "auto": the own Metal decoder when it can run the file, otherwise llama.cpp; "metal": the own decoder or an error; "llama.cpp".
    std::string text_engine = "auto";
    // "auto": the GPU (the own decoder or llama.cpp's Metal), the CPU without one; "gpu"; "cpu" (llama.cpp on the CPU, no GPU layers). DOOMS_DEVICE overrides
    // "auto" (pipeline/device.h). With "auto", text.n_gpu_layers == 0 also means the CPU (the own decoder is GPU-only).
    std::string device = "auto";
    int32_t metal_queue_depth = 1;   // decode steps the Metal decoder keeps queued (greedy generation)
    LlmOptions() { text.n_ctx = 4096; }
};

struct LlmGenerateOptions {
    int32_t max_new_tokens = 128;       // in [1, n_ctx)
    SamplerOptions sampler;             // greedy by default
    std::vector<std::string> stop;      // generation ends before the first occurrence of any of these strings (non-empty strings only)
};

struct LlmTimings {
    double prefill_ms = 0;              // prompt through the model, up to the first logits
    double first_token_ms = 0;          // from the start of generate() to the first generated token
    double decode_ms = 0;               // all later tokens
    int64_t prompt_tokens = 0, generated_tokens = 0;
    double decode_ms_per_token() const { return generated_tokens > 1 ? decode_ms / static_cast<double>(generated_tokens - 1) : 0.0; }
};

struct LlmGeneration {
    std::string text;                   // the answer (the end-of-generation token and anything from a stop string on are excluded)
    std::vector<int32_t> tokens;        // generated token ids (including a final end-of-generation token)
    bool stopped = false;               // the model emitted an end-of-generation token, or a stop string matched
    bool cancelled = false;             // the on_text callback returned false; none of stopped / cancelled: max_new_tokens was reached
};

// Not thread-safe. Every generate()/chat() is a fresh conversation (the KV cache is cleared first).
class Llm {
public:
    //   NotFound  no such file     DataLoss  not a GGUF model llama.cpp can load     InvalidArgument  bad options
    static Result<Llm> load(const std::string & path, const LlmOptions & options = LlmOptions());

    Llm(Llm &&) noexcept;
    Llm & operator=(Llm &&) noexcept;
    ~Llm();

    int64_t vocab_size() const;
    const char * text_engine_used() const;   // "metal" or "llama.cpp"
    int32_t context_size() const;
    bool has_chat_template() const;

    Result<std::vector<int32_t>> tokenize(const std::string & text, bool add_special = true, bool parse_special = false) const;
    // The exact prompt tokens chat() feeds the model for `messages` (chat template applied, generation prompt appended). Unsupported when the file has no template.
    Result<std::vector<int32_t>> chat_tokens(const std::vector<ChatMessage> & messages) const;

    // Continues `prompt` (raw text; the vocabulary's BOS is added when the model asks for it). `on_text` (optional) receives each new piece of text, always complete
    // UTF-8 (a character split over tokens is held back until it is whole); return false to stop.
    //   InvalidArgument  empty prompt, max_new_tokens or sampler out of range, empty stop string
    //   ResourceExhausted  prompt + max_new_tokens exceed the context
    Result<LlmGeneration> generate(const std::string & prompt, const LlmGenerateOptions & options = LlmGenerateOptions(),
                                   const std::function<bool(const std::string &)> & on_text = nullptr, LlmTimings * timings = nullptr);

    // Answers a conversation with the model's chat template (Unsupported when the file has none).
    Result<LlmGeneration> chat(const std::vector<ChatMessage> & messages, const LlmGenerateOptions & options = LlmGenerateOptions(),
                               const std::function<bool(const std::string &)> & on_text = nullptr, LlmTimings * timings = nullptr);

private:
    struct Impl;
    explicit Llm(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
