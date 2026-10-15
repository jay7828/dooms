#pragma once
// Loads a llama- or qwen2-architecture GGUF text model (f16 weight matrices) into the engine's own Metal decoder (backend/metal/metal_llama_decoder.h). The file's
// metadata gives every hyperparameter; anything the decoder does not implement (another architecture, scaled or partial rotary embeddings, quantized or f32
// matrices) is refused with Unsupported so the caller can fall back to llama.cpp.

#include <cstdint>
#include <memory>
#include <string>

#include "backend/metal/metal_llama_decoder.h"
#include "core/status.h"

namespace dooms {

struct NativeLlamaOptions {
    int32_t n_ctx = 2048;    // positions the KV cache holds (at most the model's context length)
    int32_t n_batch = 512;   // largest prefill chunk
    bool spin_wait = true;    // LlamaDecoderConfig::spin_wait
    int32_t prefill_tile = 0; // LlamaDecoderConfig::prefill_tile
};

//   NotFound  no such file     DataLoss  not a readable GGUF     Unsupported  see above     InvalidArgument  bad options or inconsistent metadata
Result<std::unique_ptr<MetalLlamaDecoder>> load_native_llama(const std::string & path, const NativeLlamaOptions & options = NativeLlamaOptions());

}  // namespace dooms
