#pragma once
// A vision-language model (SmolVLM / Idefics3 layout, one image tile): photo -> Pillow-exact preprocessing -> vision tower + connector on Core ML (the
// Neural Engine by default) -> 64 image embeddings -> spliced into the prompt of the language model that llama.cpp runs on the GPU -> greedy generation.
// Files from tools/convert/convert_smolvlm.py: vision-fp16.mlpackage, text-f16.gguf (or any quantization of it), vlm-config.gguf.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "backend/coreml/coreml_model.h"
#include "core/status.h"
#include "models/vlm/text_decoder.h"
#include "pipeline/rgb_image.h"
#include "pipeline/vlm_preprocess.h"

namespace dooms {

struct VlmPaths {
    std::string vision;  // Core ML package
    std::string text;    // language-model GGUF
    std::string config;  // vlm-config.gguf
};

struct VlmOptions {
    std::string vision_units = "cpu_ne";  // "all" | "cpu_gpu" | "cpu_ne" | "cpu"
    TextDecoderOptions text;
    // Who runs the language model: "auto" (the engine's own Metal decoder when it can run the file, otherwise llama.cpp), "metal" (the own decoder,
    // backend/metal/metal_llama_decoder.h: f16 llama-architecture files only; anything else is refused at create()), or "llama.cpp". With the own decoder,
    // llama.cpp still provides the vocabulary (tokenizing, token text). text_engine_used() says which one runs.
    std::string text_engine = "auto";
    bool metal_spin_wait = true;     // LlamaDecoderConfig::spin_wait
    int32_t metal_queue_depth = 1;   // decode steps the Metal decoder keeps queued ahead of the token being read (>= 1)
};

struct VlmTimings {
    double preprocess_ms = 0, vision_ms = 0;      // photo -> pixel values; vision tower + connector
    double prefill_ms = 0;                        // prompt + image embeddings through the language model (up to the first logits)
    double first_token_ms = 0;                    // from the start of generate() to the first generated token
    double decode_ms = 0;                         // all later tokens
    int64_t prompt_tokens = 0, generated_tokens = 0;
    double decode_ms_per_token() const { return generated_tokens > 1 ? decode_ms / static_cast<double>(generated_tokens - 1) : 0.0; }
};

struct VlmGeneration {
    std::string text;                  // what the model said (the end-of-generation token excluded)
    std::vector<int32_t> tokens;       // generated token ids, including the final end-of-generation token when the model stopped on its own
    bool stopped = false;              // true: the model emitted an end-of-generation token
    bool cancelled = false;            // true: the on_text callback returned false; neither stopped nor cancelled: max_new_tokens was reached
};

struct VlmRequest {
    const RgbImage * image = nullptr;  // must outlive the call
    std::string prompt;
};

// Not thread-safe.
class VlmPipeline {
public:
    //   NotFound/InvalidArgument/DataLoss  as the loaders report them; InvalidArgument when the three files do not belong together
    static Result<VlmPipeline> create(const VlmPaths & paths, const VlmOptions & options = VlmOptions());

    VlmPipeline(VlmPipeline &&) noexcept;
    VlmPipeline & operator=(VlmPipeline &&) noexcept;
    ~VlmPipeline();

    int64_t n_image_tokens() const;
    const char * text_engine_used() const;   // "metal" or "llama.cpp"
    int64_t embedding_dim() const;
    const VlmImageOptions & image_options() const;

    // The image embeddings the language model receives: n_image_tokens() x embedding_dim() floats.
    Status encode_image(const RgbImage & image, std::vector<float> * embeddings, VlmTimings * timings = nullptr);

    // Answers `prompt` about `image` (a fresh conversation each call). `on_text` (optional) receives each new piece of text (raw token bytes: a multi-byte UTF-8 character may arrive split over two pieces) as it is generated; returning false stops the generation (the result then has cancelled = true).
    //   InvalidArgument  empty prompt/image, max_new_tokens outside [1, 4096]     ResourceExhausted  prompt + image + answer exceed the context
    Result<VlmGeneration> generate(const RgbImage & image, const std::string & prompt, int max_new_tokens, const std::function<bool(const std::string &)> & on_text = nullptr,
                                   VlmTimings * timings = nullptr);

    // Same, with the image embeddings already computed (encode_image) or taken from elsewhere.
    Result<VlmGeneration> generate_from_embeddings(const std::vector<float> & embeddings, const std::string & prompt, int max_new_tokens,
                                                   const std::function<bool(const std::string &)> & on_text = nullptr, VlmTimings * timings = nullptr);

    // Answers a batch of requests one after the other, with the vision encoder (preprocessing + Neural Engine) already working on request i+1 on a worker
    // thread while the GPU decodes the answer to request i. Results equal calling generate() for each request. Stops at the first failure.
    //   wall_ms (optional) receives the total time.
    Result<std::vector<VlmGeneration>> generate_many(const std::vector<VlmRequest> & requests, int max_new_tokens, double * wall_ms = nullptr);

private:
    struct Impl;
    explicit VlmPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
