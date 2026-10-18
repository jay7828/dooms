#pragma once
// kokoro.gguf (tools/convert/convert_kokoro.py, docs/formats/kokoro-gguf.md) -> the Metal vocoder (backend/metal/metal_kokoro_vocoder.h).
#include <memory>

#include "backend/metal/metal_kokoro_vocoder.h"
#include "core/status.h"
#include "format/gguf.h"

namespace dooms {

// The vocoder's config from the file's metadata and its "dec.*" tensors (they must outlive the vocoder only during create: the weights are copied).
Result<std::unique_ptr<MetalKokoroVocoder>> load_kokoro_vocoder(const GgufFile & file, bool spin_wait = true);

}  // namespace dooms
