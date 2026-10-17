#include "pipeline/whisper_features.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>

namespace dooms {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr float kLogFloor = 1e-10f;       // log10(max(mel, 1e-10))
constexpr float kDynamicRange = 8.f;      // clamp to max - 8 (in log10 units)
constexpr float kOffset = 4.f, kScale = 4.f;   // (x + 4) / 4
}  // namespace

Result<WhisperFeatures> WhisperFeatures::create(const WhisperFeatureConfig & c) {
    if (c.n_fft < 2 || c.hop < 1 || c.n_mels < 1 || c.chunk_samples < c.n_fft || c.chunk_samples % c.hop != 0) {
        return invalid_argument("whisper features: n_fft >= 2, hop >= 1, n_mels >= 1, chunk_samples >= n_fft and a multiple of hop are required");
    }
    const int32_t n_freq = c.n_fft / 2 + 1;
    if (static_cast<int64_t>(c.mel_filters.size()) != static_cast<int64_t>(c.n_mels) * n_freq) {
        return invalid_argument("the mel filter bank has " + std::to_string(c.mel_filters.size()) + " values, expected " + std::to_string(c.n_mels * n_freq));
    }
    WhisperFeatures f;
    f.cfg_ = c;
    f.n_freq_ = n_freq;
    f.n_frames_ = c.chunk_samples / c.hop;
    // Periodic Hann window (torch.hann_window default) folded into the DFT basis.
    std::vector<double> w(static_cast<size_t>(c.n_fft));
    for (int32_t t = 0; t < c.n_fft; t++) w[static_cast<size_t>(t)] = 0.5 - 0.5 * std::cos(2.0 * kPi * t / c.n_fft);
    f.basis_.resize(static_cast<size_t>(2 * n_freq) * static_cast<size_t>(c.n_fft));
    for (int32_t k = 0; k < n_freq; k++) {
        for (int32_t t = 0; t < c.n_fft; t++) {
            const double a = 2.0 * kPi * static_cast<double>((static_cast<int64_t>(k) * t) % c.n_fft) / c.n_fft;
            f.basis_[static_cast<size_t>(k) * c.n_fft + t] = static_cast<float>(std::cos(a) * w[static_cast<size_t>(t)]);
            f.basis_[static_cast<size_t>(n_freq + k) * c.n_fft + t] = static_cast<float>(-std::sin(a) * w[static_cast<size_t>(t)]);
        }
    }
    return f;
}

int32_t WhisperFeatures::n_frames() const { return n_frames_; }

Status WhisperFeatures::compute(const float * samples, size_t n, std::vector<float> * out) const {
    if (!samples || !out) return invalid_argument("null samples or output");
    if (n > static_cast<size_t>(cfg_.chunk_samples)) return invalid_argument("more samples than one window (" + std::to_string(cfg_.chunk_samples) + ")");
    const int32_t nfft = cfg_.n_fft, pad = nfft / 2, L = cfg_.chunk_samples;
    // Window padded to L with zeros, then reflect-padded by n_fft/2 on both sides (index i of the padded signal = reflect(i - pad)).
    std::vector<float> sig(static_cast<size_t>(L + 2 * pad), 0.f);
    auto at = [&](int64_t i) -> float { return i < static_cast<int64_t>(n) ? samples[i] : 0.f; };
    for (int64_t i = 0; i < L + 2 * pad; i++) {
        int64_t j = i - pad;
        if (j < 0) j = -j;
        if (j >= L) j = 2 * (L - 1) - j;
        sig[static_cast<size_t>(i)] = at(j);
    }
    // Frames: the stft has L / hop + 1 frames; the last is dropped.
    const int32_t F = n_frames_;
    std::vector<float> frames(static_cast<size_t>(F) * nfft);
    for (int32_t f = 0; f < F; f++) std::copy_n(sig.data() + static_cast<size_t>(f) * cfg_.hop, nfft, frames.data() + static_cast<size_t>(f) * nfft);
    // spec[2*n_freq][F] = basis[2*n_freq][nfft] . frames^T
    std::vector<float> spec(static_cast<size_t>(2 * n_freq_) * F);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2 * n_freq_, F, nfft, 1.f, basis_.data(), nfft, frames.data(), nfft, 0.f, spec.data(), F);
    std::vector<float> power(static_cast<size_t>(n_freq_) * F);
    for (size_t i = 0; i < power.size(); i++) { const float re = spec[i], im = spec[power.size() + i]; power[i] = re * re + im * im; }
    out->assign(static_cast<size_t>(cfg_.n_mels) * F, 0.f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cfg_.n_mels, F, n_freq_, 1.f, cfg_.mel_filters.data(), n_freq_, power.data(), F, 0.f, out->data(), F);
    float mx = -INFINITY;
    for (float & v : *out) { v = std::log10(std::max(v, kLogFloor)); mx = std::max(mx, v); }
    for (float & v : *out) v = (std::max(v, mx - kDynamicRange) + kOffset) / kScale;
    return Status();
}

}  // namespace dooms
