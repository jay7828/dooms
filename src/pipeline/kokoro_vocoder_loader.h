#pragma once
// kokoro.gguf (tools/convert/convert_kokoro.py, docs/formats/kokoro-gguf.md) -> the Metal vocoder (backend/metal/metal_kokoro_vocoder.h).
#include <memory>
#include <string>

#include "backend/kokoro_vocoder.h"
#include "core/status.h"
#include "format/gguf.h"

namespace dooms {

// The vocoder's config from the file's metadata and its "dec.*" tensors (they must outlive the vocoder only during create: the weights are copied), on
// `device` ("auto": the GPU, the CPU when no Metal device can run it; "gpu"; "cpu"; pipeline/device.h).
Result<std::unique_ptr<KokoroVocoder>> load_kokoro_vocoder(const GgufFile & file, bool spin_wait = true, const std::string & device = "auto");

}  // namespace dooms
