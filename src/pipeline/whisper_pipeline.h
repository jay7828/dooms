#pragma once
// Whisper speech-to-text: audio -> 30-second windows -> log-mel features (whisper_features.h) -> the encoder on Core ML (the GPU by default) ->
// the engine's own Metal decoder (backend/metal/metal_whisper_decoder.h), greedy, with Whisper's suppression lists -> text. Files from
// tools/convert/convert_whisper.py: encoder-fp16.mlpackage and decoder.gguf (docs/formats/whisper-gguf.md).
// Long audio is cut into consecutive 30-second windows (no timestamp-based seeking): a word that straddles a window boundary can be split.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"

namespace dooms {

class MetalWhisperDecoder;

struct WhisperPaths {
    std::string encoder;   // Core ML package
    std::string decoder;   // decoder.gguf
};

struct WhisperOptions {
    // "all" | "cpu_gpu" | "cpu_ne" | "cpu". The GPU is the default: more accurate AND faster than the Neural Engine on both models measured (bench/BASELINES.md;
    // large-v3-turbo's 32 fp16 layers on the Neural Engine are 0.26 relative L2 from PyTorch and change the text).
    std::string encoder_units = "cpu_gpu";
    std::string language = "en";            // a code the model knows ("en", "de", ...)
    std::string task = "transcribe";         // "transcribe" | "translate" (into English)
    int32_t max_tokens_per_window = 224;     // in [1, n_text_ctx / 2] (Whisper's own limit per 30-s window)
    int32_t queue_depth = 1;                 // decode steps queued on the GPU ahead of the token being read
    bool spin_wait = true;
    int32_t cross_splits = 0;                // decoder tuning (WhisperDecoderConfig): 0 = automatic
    int32_t fused_layernorm = -1;
};

struct WhisperTimings {
    double features_ms = 0, encoder_ms = 0, decode_ms = 0, total_ms = 0;
    int64_t windows = 0, tokens = 0;
    double audio_s = 0;
    double rtf() const { return audio_s > 0 ? total_ms / 1000.0 / audio_s : 0.0; }   // real-time factor: processing time / audio duration
};

struct WhisperResult {
    std::string text;                  // the transcript (windows joined)
    std::vector<int32_t> tokens;       // generated text tokens of every window (end-of-text tokens excluded)
};

// Not thread-safe.
class WhisperPipeline {
public:
    //   NotFound / DataLoss / InvalidArgument  as the loaders report them; Unsupported  a model the decoder cannot run; InvalidArgument  unknown language/task
    static Result<WhisperPipeline> create(const WhisperPaths & paths, const WhisperOptions & options = WhisperOptions());

    WhisperPipeline(WhisperPipeline &&) noexcept;
    WhisperPipeline & operator=(WhisperPipeline &&) noexcept;
    ~WhisperPipeline();

    int32_t sample_rate() const;
    int32_t window_samples() const;
    int32_t n_mels() const;
    int32_t n_frames() const;

    // 16 kHz mono samples in [-1, 1] -> text.   InvalidArgument  no samples
    Result<WhisperResult> transcribe(const std::vector<float> & samples, WhisperTimings * timings = nullptr);
    // Any audio file AudioToolbox decodes (see pipeline/audio_io.h).
    Result<WhisperResult> transcribe_file(const std::string & path, WhisperTimings * timings = nullptr);

    // Stages, for tests: one window's features (n_mels x n_frames) and encoder states (n_audio_ctx x n_state).
    Status features(const float * samples, size_t n, std::vector<float> * mel);
    Status encode(const std::vector<float> & mel, std::vector<float> * states);
    std::vector<int32_t> prompt() const;                       // <|startoftranscript|> <|lang|> <|task|> <|notimestamps|>
    std::string decode_text(const std::vector<int32_t> & tokens) const;
    MetalWhisperDecoder & decoder();
    bool is_end_of_text(int32_t token) const;

private:
    struct Impl;
    explicit WhisperPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
