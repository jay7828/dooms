#pragma once
// Which device runs a model's own kernels (the Whisper decoder, the Kokoro vocoder): "auto" (default) is the GPU when it can, else the CPU; "gpu" and "cpu"
// force one. The environment variable DOOMS_DEVICE=cpu|gpu replaces "auto" (to run on the CPU without changing code, or in tests).
#include <string>

#include "core/status.h"

namespace dooms {

struct DeviceChoice {
    bool try_gpu = true;        // attempt the GPU implementation first
    bool allow_fallback = true; // on failure to create it, use the CPU (only for "auto")
};

//   InvalidArgument  anything but "auto", "gpu", "cpu" (in the option or in DOOMS_DEVICE)
Result<DeviceChoice> resolve_device(const std::string & requested);

}  // namespace dooms
