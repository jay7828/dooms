#pragma once
// Whisper's log-mel spectrogram, as transformers' WhisperFeatureExtractor computes it on its torch path: the window of samples zero-padded to chunk_samples,
// reflect-padded by n_fft/2 (torch.stft center=True), a periodic Hann window, |STFT|^2 with the last frame dropped, the model's mel filter bank,
// log10(max(mel, 1e-10)), clamped to (max - 8), then (x + 4) / 4. The DFT runs as a matrix product on Accelerate.

#include <cstdint>
#include <vector>

#include "core/status.h"

namespace dooms {

struct WhisperFeatureConfig {
    int32_t n_fft = 0, hop = 0, n_mels = 0;
    int32_t chunk_samples = 0;              // samples of one window (30 s at 16 kHz = 480000)
    std::vector<float> mel_filters;         // n_mels x (n_fft/2 + 1), row-major (from the model file)
};

class WhisperFeatures {
public:
    //   InvalidArgument  sizes that do not fit together
    static Result<WhisperFeatures> create(const WhisperFeatureConfig & config);

    int32_t n_frames() const;   // frames per window (chunk_samples / hop)
    // Log-mel features of up to chunk_samples samples (shorter input is zero-padded): n_mels x n_frames floats, row-major.
    //   InvalidArgument  null samples, more than chunk_samples
    Status compute(const float * samples, size_t n, std::vector<float> * out) const;

private:
    WhisperFeatureConfig cfg_;
    std::vector<float> basis_;    // (2 * n_freq) x n_fft: rows cos(2 pi k t / n) * w[t], then -sin(...) * w[t]
    int32_t n_freq_ = 0, n_frames_ = 0;
};

}  // namespace dooms
