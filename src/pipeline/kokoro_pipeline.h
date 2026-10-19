#pragma once
// Kokoro-82M text to speech (hexgrad, Apache-2.0): phonemes -> 24 kHz audio. The phonemes come from the caller (misaki, Kokoro's own G2P, in the Python
// binding): this layer does not convert text. Stages (tools/convert/kokoro_model.py is their reference):
//   phoneme ids -> prosody (Core ML: ALBERT + duration/prosody encoders) -> durations rounded to frames, features expanded per frame -> F0 / energy (Core ML)
//   -> the harmonic source (sine generator from F0, seeded noise) and its STFT -> the vocoder (the engine's own Metal kernels, backend/metal/metal_kokoro_vocoder.h)
//   -> iSTFT -> samples.
// Files from tools/convert/convert_kokoro.py: prosody-fp32.mlpackage, f0n-fp32.mlpackage, kokoro.gguf (docs/formats/kokoro-gguf.md).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"

namespace dooms {

struct KokoroPaths {
    std::string prosody;   // prosody-fp32.mlpackage
    std::string f0n;       // f0n-fp32.mlpackage
    std::string model;     // kokoro.gguf
};

struct KokoroOptions {
    std::string units = "cpu";   // the two Core ML packages: "cpu" (default: their LSTMs are fastest there, bench/BASELINES.md), "cpu_gpu", "all"
    std::string device = "auto"; // the vocoder: "auto" (the GPU, the CPU when no Metal device can run it; DOOMS_DEVICE overrides), "gpu", "cpu"
    bool spin_wait = true;
};

struct KokoroTimings {
    double prosody_ms = 0, f0n_ms = 0, source_ms = 0, vocoder_ms = 0, istft_ms = 0, total_ms = 0;
    int64_t tokens = 0, frames = 0;
    double audio_s = 0;
    double rtf() const { return audio_s > 0 ? total_ms / 1000.0 / audio_s : 0.0; }
};

// The sine generator's random inputs: initial phases of the overtones (index 0 is ignored: the fundamental starts at phase 0) and Gaussian noise
// [samples][harmonics]. Given explicitly for reproducing a reference run; otherwise drawn from the seed.
struct KokoroNoise {
    std::vector<float> rand_ini;
    std::vector<float> noise;
};

// Constants of the harmonic source (kokoro.gguf metadata).
struct KokoroSourceConfig {
    int32_t upsample = 300, harmonics = 9, sample_rate = 24000, n_fft = 20, hop = 5;
    float sine_amp = 0.1f, noise_std = 0.003f, voiced_threshold = 10.f;
};

// Host stages, exposed for the tests (each mirrors its numpy reference in tools/convert/kokoro_model.py).
// round(durations / speed), at least 1 -> frames per token; en [640][F] and asr [512][F] (d is [T][640], t_en [512][T]).
void kokoro_align(const std::vector<float> & durations, const std::vector<float> & d, int32_t d_width, const std::vector<float> & t_en, int32_t t_width, float speed,
                  std::vector<int32_t> * frames, std::vector<float> * en, std::vector<float> * asr);
// F0 [2F] -> the merged harmonic excitation [2F * upsample]: SineGen (linear-interpolated phase) * voiced + noise, tanh(w . s + b).
void kokoro_sine_source(const std::vector<float> & f0, const KokoroNoise & noise, const KokoroSourceConfig & c, const float * w, float b, std::vector<float> * out);
// STFT (center, reflect padding, periodic Hann): [n_fft + 2][frames] = magnitudes then phases.
void kokoro_stft(const std::vector<float> & x, int32_t n_fft, int32_t hop, std::vector<float> * out, int32_t * frames);
// The vocoder output [n_fft + 2][frames] -> samples: exp(log-magnitude), sin(phase argument), iSTFT (center, periodic Hann, window-square normalization).
void kokoro_istft(const std::vector<float> & x, int32_t frames, int32_t n_fft, int32_t hop, std::vector<float> * out);
// Seeded draws in SineGen's shapes (xoshiro256** + Box-Muller): reproducible for a seed, not torch's sequence.
KokoroNoise kokoro_draw_noise(int64_t samples, int32_t harmonics, uint64_t seed);

// Not thread-safe.
class KokoroPipeline {
public:
    //   NotFound / InvalidArgument / DataLoss as the loaders report them; Unsupported  a model the vocoder cannot run
    static Result<KokoroPipeline> create(const KokoroPaths & paths, const KokoroOptions & options = KokoroOptions());
    KokoroPipeline(KokoroPipeline &&) noexcept;
    KokoroPipeline & operator=(KokoroPipeline &&) noexcept;
    ~KokoroPipeline();

    int32_t sample_rate() const;
    const char * device() const;     // where the vocoder runs: "gpu" or "cpu"
    std::vector<std::string> voices() const;
    int32_t max_phonemes() const;   // phonemes per call (the voices' style table and ALBERT's positions)

    // phonemes: UTF-8, misaki's symbols (one Unicode code point each; unknown ones are skipped as Kokoro does). speed > 0 (1 = the model's pace).
    //   InvalidArgument  no known phoneme, too many, unknown voice, speed <= 0
    Result<std::vector<float>> synthesize(const std::string & phonemes, const std::string & voice, float speed = 1.f, uint64_t seed = 0, KokoroTimings * timings = nullptr);

    // Stages, for tests: the token ids (0, ids..., 0) and the style row Kokoro picks (voice row (phonemes - 1)), then the whole path from them with explicit noise.
    Result<std::vector<int32_t>> ids(const std::string & phonemes, int32_t * n_phonemes = nullptr) const;
    Result<std::vector<float>> style(const std::string & voice, int32_t n_phonemes) const;
    Result<std::vector<float>> synthesize_ids(const std::vector<int32_t> & ids, const std::vector<float> & style, float speed, const KokoroNoise * noise, uint64_t seed,
                                              KokoroTimings * timings = nullptr, std::vector<int32_t> * frames = nullptr);

private:
    struct Impl;
    explicit KokoroPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
