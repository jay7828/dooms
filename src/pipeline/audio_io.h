#pragma once
// Audio file decoding for the speech models: any format macOS's AudioToolbox reads (WAV, AIFF, FLAC, MP3, M4A/AAC, CAF, ...), converted on decode to mono float
// samples in [-1, 1] at the requested rate (AudioToolbox's own resampler). Everything that touches AudioToolbox lives in audio_io.cpp. Apple only (ADR 0005).

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"

namespace dooms {

//   NotFound  no such file     DataLoss  not audio AudioToolbox can decode, or empty     InvalidArgument  sample_rate outside [1000, 192000]
Result<std::vector<float>> load_audio(const std::string & path, int32_t sample_rate);

}  // namespace dooms
