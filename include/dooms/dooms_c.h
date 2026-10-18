#ifndef DOOMS_C_H
#define DOOMS_C_H
/* dooms.cpp C API: opaque handles and plain structs over the language-model (any GGUF), vision-language (SmolVLM) and vision-language-action (SmolVLA) pipelines.
 *
 *  - Every function returns a dooms_code; DOOMS_OK is 0. On failure dooms_last_error() returns a message for the calling thread (valid until the next call
 *    on that thread). No function throws or aborts on bad arguments.
 *  - Handles are not thread-safe: use one handle from one thread at a time (different handles may be used from different threads). Closing NULL is a no-op.
 *  - Images are interleaved 8-bit RGB, row-major and tightly packed (stride = 3 * width); the library copies what it needs.
 *  - Output pointers in result structs point into memory owned by the handle and stay valid until the next call that uses the handle, or until it is closed.
 *  - Struct layouts are append-only within a major version; zero-initialize structs and set `struct_size = sizeof(...)` so that a newer library can tell an older caller.
 *  - Apple Silicon only. Models are read from the files named in the paths structs (made by the scripts in tools/convert); nothing is downloaded.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DOOMS_C_API_VERSION 1
#define DOOMS_API __attribute__((visibility("default")))

typedef enum dooms_code {
    DOOMS_OK = 0,
    DOOMS_INVALID_ARGUMENT = 1,
    DOOMS_NOT_FOUND = 2,
    DOOMS_DATA_LOSS = 3,
    DOOMS_UNSUPPORTED = 4,
    DOOMS_RESOURCE_EXHAUSTED = 5, /* also: an output buffer that is too small (the needed size is reported where documented) */
    DOOMS_IO_ERROR = 6,
    DOOMS_INTERNAL = 7
} dooms_code;

DOOMS_API const char * dooms_version(void);         /* "major.minor.patch" of the library */
DOOMS_API int32_t dooms_c_api_version(void);        /* DOOMS_C_API_VERSION the library was built with */
DOOMS_API const char * dooms_last_error(void);      /* message of the last failure on this thread, "" if none */
DOOMS_API const char * dooms_code_name(dooms_code code);

typedef struct dooms_image {
    int32_t width, height;
    const uint8_t * rgb; /* height * width * 3 bytes */
} dooms_image;


/* ---------------------------------------------------------------- vision-language model (SmolVLM) */
typedef struct dooms_vlm dooms_vlm;

typedef struct dooms_vlm_paths {
    size_t struct_size;
    const char * vision; /* vision-fp16.mlpackage */
    const char * text;   /* text-f16.gguf */
    const char * config; /* vlm-config.gguf */
} dooms_vlm_paths;

typedef struct dooms_vlm_options {
    size_t struct_size;
    const char * vision_units; /* NULL: cpu_ne */
    int32_t n_threads;         /* text decoder CPU threads, 0: default */
} dooms_vlm_options;

DOOMS_API dooms_code dooms_vlm_open(const dooms_vlm_paths * paths, const dooms_vlm_options * options, dooms_vlm ** out);
DOOMS_API void dooms_vlm_close(dooms_vlm * vlm);
/* Greedy answer of at most max_new_tokens (1..4096) tokens, written as a NUL-terminated UTF-8 string. *needed (optional) receives the length including the NUL;
 * if `capacity` is smaller, nothing is written and DOOMS_RESOURCE_EXHAUSTED is returned. */
DOOMS_API dooms_code dooms_vlm_generate(dooms_vlm * vlm, const dooms_image * image, const char * prompt, int32_t max_new_tokens, char * out, size_t capacity, size_t * needed);

/* Streaming variant: `callback` receives each new piece of the answer as raw token bytes (a UTF-8 character may arrive split across two pieces) and returns 0 to
 * continue or non-zero to stop the generation early (still DOOMS_OK). The piece pointer is valid only during the call. */
typedef int32_t (*dooms_text_callback)(const char * piece, void * user);
DOOMS_API dooms_code dooms_vlm_generate_stream(dooms_vlm * vlm, const dooms_image * image, const char * prompt, int32_t max_new_tokens, dooms_text_callback callback, void * user);

/* ---------------------------------------------------------------- text language model (any GGUF llama.cpp can load) */
typedef struct dooms_llm dooms_llm;

typedef struct dooms_llm_options {
    size_t struct_size;
    int32_t n_ctx;        /* positions of the KV cache (prompt + answer); 0: 4096 */
    int32_t n_threads;    /* CPU threads, 0: default */
    int32_t n_gpu_layers; /* layers on the GPU; -1: all, 0: none */
} dooms_llm_options;

/* How the next token is chosen. temperature 0 is greedy. top_k 0 and top_p 1 switch those filters off; repeat_penalty 1 is off. The same seed on the same inputs
 * gives the same text. Pass NULL wherever a dooms_llm_sampling is taken for greedy decoding. */
typedef struct dooms_llm_sampling {
    size_t struct_size;
    float temperature;
    int32_t top_k;
    float top_p;
    float repeat_penalty;
    uint32_t seed;
} dooms_llm_sampling;

typedef struct dooms_chat_message {
    const char * role;    /* "system", "user" or "assistant" */
    const char * content;
} dooms_chat_message;

DOOMS_API dooms_code dooms_llm_open(const char * gguf_path, const dooms_llm_options * options, dooms_llm ** out);
DOOMS_API void dooms_llm_close(dooms_llm * llm);
DOOMS_API int32_t dooms_llm_context_size(const dooms_llm * llm);

/* Continue `prompt` with at most max_new_tokens (1..context size) tokens. `stop` (optional, n_stop strings) ends the text before the first match. The answer is
 * written as a NUL-terminated UTF-8 string; *needed (optional) receives its length including the NUL; if `capacity` is smaller nothing is written and
 * DOOMS_RESOURCE_EXHAUSTED is returned. Every call starts a fresh conversation. */
DOOMS_API dooms_code dooms_llm_generate(dooms_llm * llm, const char * prompt, int32_t max_new_tokens, const dooms_llm_sampling * sampling, const char * const * stop,
                                        int32_t n_stop, char * out, size_t capacity, size_t * needed);
/* Streaming: `callback` receives each new piece of text, always complete UTF-8, and returns 0 to continue or non-zero to stop early (still DOOMS_OK). */
DOOMS_API dooms_code dooms_llm_generate_stream(dooms_llm * llm, const char * prompt, int32_t max_new_tokens, const dooms_llm_sampling * sampling,
                                               const char * const * stop, int32_t n_stop, dooms_text_callback callback, void * user);
/* The same for a conversation, formatted with the model's own chat template (DOOMS_UNSUPPORTED if the file has none). */
DOOMS_API dooms_code dooms_llm_chat(dooms_llm * llm, const dooms_chat_message * messages, int32_t n_messages, int32_t max_new_tokens, const dooms_llm_sampling * sampling,
                                    const char * const * stop, int32_t n_stop, char * out, size_t capacity, size_t * needed);
DOOMS_API dooms_code dooms_llm_chat_stream(dooms_llm * llm, const dooms_chat_message * messages, int32_t n_messages, int32_t max_new_tokens, const dooms_llm_sampling * sampling,
                                           const char * const * stop, int32_t n_stop, dooms_text_callback callback, void * user);

/* How the last generate/chat call on a handle ended, with its exact token counts (for usage accounting and to tell a token limit from a natural stop). */
typedef enum dooms_finish_reason {
    DOOMS_FINISH_NONE = 0,      /* no call has finished on this handle, or the last one failed */
    DOOMS_FINISH_STOP = 1,      /* the model emitted an end-of-generation token, or a stop string matched */
    DOOMS_FINISH_LENGTH = 2,    /* max_new_tokens tokens were generated without a stop */
    DOOMS_FINISH_CANCELLED = 3  /* the streaming callback returned non-zero */
} dooms_finish_reason;

typedef struct dooms_llm_generation_info {
    size_t struct_size;
    int32_t prompt_tokens;      /* tokens fed to the model: the prompt as tokenized (BOS included) or the chat template's output */
    int32_t completion_tokens;  /* tokens the model sampled, including a final end-of-generation token and the tokens of a matched stop string */
    int32_t finish_reason;      /* a dooms_finish_reason */
} dooms_llm_generation_info;

/* Fills `out` (struct_size set by the caller) for the last dooms_llm_generate / _stream / dooms_llm_chat / _stream call on this handle. */
DOOMS_API dooms_code dooms_llm_last_generation(const dooms_llm * llm, dooms_llm_generation_info * out);
/* Number of tokens `text` becomes (add_special non-zero: with the BOS token the model asks for, as dooms_llm_generate tokenizes a prompt). An empty or NULL text is
 * DOOMS_INVALID_ARGUMENT, like an empty prompt. */
DOOMS_API dooms_code dooms_llm_count_tokens(dooms_llm * llm, const char * text, int32_t add_special, int32_t * count);
/* Number of prompt tokens dooms_llm_chat would feed the model for this conversation (chat template applied, generation prompt appended; DOOMS_UNSUPPORTED without one). */
DOOMS_API dooms_code dooms_llm_count_chat_tokens(dooms_llm * llm, const dooms_chat_message * messages, int32_t n_messages, int32_t * count);

/* ---------------------------------------------------------------- vision-language-action model (SmolVLA) */
typedef struct dooms_vla dooms_vla;

typedef struct dooms_vla_paths {
    size_t struct_size;
    const char * vision;  /* vision-b1-fp16.mlpackage */
    const char * prefix;  /* prefix-fp16.mlpackage */
    const char * expert;  /* expert-fp16.mlpackage */
    const char * host;    /* host.gguf */
    const char * vocab;   /* vocab.gguf (tokenizer) */
} dooms_vla_paths;

typedef struct dooms_vla_options {
    size_t struct_size;
    const char * units; /* all three Core ML models; NULL: cpu_ne */
    uint64_t seed;      /* seeds the flow's noise; each call uses seed + call index */
} dooms_vla_options;

typedef struct dooms_vla_info {
    int32_t n_cameras;
    int32_t chunk_size;  /* actions per call */
    int32_t action_dim;  /* padded width of each action row */
    int32_t state_dim;   /* the state is zero-padded to this width */
    int32_t max_task_tokens;
} dooms_vla_info;

DOOMS_API dooms_code dooms_vla_open(const dooms_vla_paths * paths, const dooms_vla_options * options, dooms_vla ** out);
DOOMS_API void dooms_vla_close(dooms_vla * vla);
DOOMS_API dooms_code dooms_vla_info_get(const dooms_vla * vla, dooms_vla_info * out);
/* cameras: exactly info.n_cameras images (any size). state: n_state (<= state_dim) already-normalized values. noise: NULL for random noise (seeded by the options), or
 * chunk_size * action_dim floats to start the flow from (reproducible runs, parity checks). actions: chunk_size * action_dim floats, row-major;
 * `capacity` is in floats. The robot's own action columns come first; denormalize with your dataset's statistics. */
DOOMS_API dooms_code dooms_vla_infer(dooms_vla * vla, const dooms_image * cameras, int32_t n_cameras, const char * task, const float * state, int32_t n_state, const float * noise,
                                     float * actions, size_t capacity);

/* ---------------------------------------------------------------- asynchronous action chunks for a control loop (SmolVLA)
 * A runner computes the next action chunk on its own thread while the robot executes the current one, and splices it in without a gap
 * (lerobot's async inference, in-process). Polling only: no callback ever runs on the runner's thread. Per control step:
 *     if (dooms_vla_runner_wants_observation(r)) { capture cameras + state; dooms_vla_runner_submit(r, ..., now, &started); }
 *     dooms_vla_runner_next_action(r, now, action, capacity, &info);   if (info.available) send `action` to the robot, else hold
 * Every action has a control step index. Row i of a chunk computed from an observation taken at step s is the action for step s + i; when
 * the chunk arrives, rows whose step has already run are dropped, rows overlapping the queued plan are blended
 * ((1 - new_weight) * queued + new_weight * new) and the rest are appended. Times are seconds on any monotonic clock, passed by the caller.
 *
 * While a runner is open on a dooms_vla, dooms_vla_infer on that handle fails (the model runs on the runner's thread) and no second runner
 * can be opened on it. The runner keeps the model alive: dooms_vla_close may come before dooms_vla_runner_close. A runner handle, like
 * every handle, is used from one thread at a time; closing it waits for an inference in flight to finish (it cannot be interrupted). */
typedef struct dooms_vla_runner dooms_vla_runner;

typedef struct dooms_vla_runner_options {
    size_t struct_size;
    double control_hz;           /* control steps per second, (0, 10000]; default 30 (lerobot) */
    double chunk_size_threshold; /* ask for the next observation when queued / chunk_size <= this, [0, 1]; default 0.5 (lerobot). 0: synchronous */
    double new_weight;           /* weight of the new chunk where it overlaps the queued one, [0, 1]; default 0.7 (lerobot "weighted_average"); 1 replaces */
} dooms_vla_runner_options;

typedef struct dooms_vla_action_info {
    size_t struct_size;
    int32_t available; /* 1: `action` was written; 0: no action this step (before the first chunk, or the queue ran dry): hold the last command */
    int32_t skipped;   /* queued actions dropped in this call because the loop was late by whole periods */
    int64_t step;      /* control step index of the action, -1 when none */
} dooms_vla_action_info;

typedef struct dooms_vla_runner_stats {
    size_t struct_size;
    int64_t actions_returned;
    int64_t waiting_calls;      /* next_action calls without an action before the first one */
    int64_t stalled_calls;      /* ... after it (the loop ran out of actions) */
    int64_t late_skipped;       /* sum of dooms_vla_action_info.skipped */
    int64_t stale_dropped;      /* rows of arrived chunks whose step had already run */
    int64_t blended;            /* rows aggregated with a queued action */
    int64_t chunks_received;
    int64_t chunks_stale;       /* chunks of which every row had already passed */
    int64_t inferences_started;
    int64_t inferences_failed;
    int64_t queued;             /* actions waiting now */
    int32_t in_flight;          /* 1 while an inference runs */
    int32_t last_error_code;    /* dooms_code of the last failed inference, DOOMS_OK if none failed */
    double last_inference_ms;   /* wall time of the last completed inference */
    const char * last_error;    /* its message ("" if none); valid until the next call on the runner */
} dooms_vla_runner_stats;

/* Fills *options with the defaults above (and struct_size). */
DOOMS_API dooms_code dooms_vla_runner_options_init(dooms_vla_runner_options * options);
/* lerobot's aggregate function names -> new_weight: "weighted_average" 0.7, "latest_only" 1, "average" 0.5, "conservative" 0.3. DOOMS_INVALID_ARGUMENT otherwise. */
DOOMS_API dooms_code dooms_vla_aggregate_weight(const char * name, double * new_weight);
/* options NULL: the defaults. */
DOOMS_API dooms_code dooms_vla_runner_open(dooms_vla * vla, const dooms_vla_runner_options * options, dooms_vla_runner ** out);
DOOMS_API void dooms_vla_runner_close(dooms_vla_runner * runner);
/* *wants = 1 when a submitted observation would start an inference (none in flight, queue at or below the threshold; always before the first chunk). */
DOOMS_API dooms_code dooms_vla_runner_wants_observation(dooms_vla_runner * runner, int32_t * wants);
/* The observation taken at `timestamp_s`, arguments as dooms_vla_infer. *started (optional) = 1 if an inference started, 0 if the observation was not
 * needed (it is dropped, never queued). The images are converted only when it is needed. */
DOOMS_API dooms_code dooms_vla_runner_submit(dooms_vla_runner * runner, const dooms_image * cameras, int32_t n_cameras, const char * task, const float * state,
                                             int32_t n_state, double timestamp_s, int32_t * started);
/* The action for the control step at `now_s` (non-decreasing across calls); never waits for an inference. `action` holds action_dim floats (capacity in
 * floats); it is written only when info->available. */
DOOMS_API dooms_code dooms_vla_runner_next_action(dooms_vla_runner * runner, double now_s, float * action, size_t capacity, dooms_vla_action_info * info);
/* Waits up to timeout_s for the inference in flight (start-up: wait for the first chunk); *idle (optional) = 1 if none is in flight. */
DOOMS_API dooms_code dooms_vla_runner_wait_idle(dooms_vla_runner * runner, double timeout_s, int32_t * idle);
DOOMS_API dooms_code dooms_vla_runner_stats_get(dooms_vla_runner * runner, dooms_vla_runner_stats * out);

/* ---------------------------------------------------------------- speech to text (Whisper: encoder on Core ML, the engine's own Metal decoder)
 * Greedy decoding without timestamps, 30-second windows one after another. Files from tools/convert/convert_whisper.py (docs/formats/whisper-gguf.md). */
typedef struct dooms_whisper dooms_whisper;

typedef struct dooms_whisper_paths {
    size_t struct_size;
    const char * encoder;  /* encoder-fp16.mlpackage */
    const char * decoder;  /* decoder.gguf */
} dooms_whisper_paths;

typedef struct dooms_whisper_options {
    size_t struct_size;
    const char * units;        /* the encoder's Core ML units: "all", "cpu_gpu", "cpu_ne", "cpu"; NULL: cpu_gpu (more accurate and faster than the Neural Engine here) */
    const char * language;     /* a code the model knows ("en", "de", ...); NULL: "en" */
    const char * task;         /* "transcribe" or "translate" (into English); NULL: "transcribe" */
    int32_t max_tokens;        /* text tokens per 30-s window, [1, 224]; 0: 224 */
} dooms_whisper_options;

typedef struct dooms_whisper_timings {
    size_t struct_size;
    double features_ms, encoder_ms, decode_ms, total_ms;  /* wall time of the last transcription, per stage and in all (audio decoding excluded) */
    double audio_s;                                       /* its duration */
    int32_t windows, tokens;                              /* 30-s windows and generated text tokens */
} dooms_whisper_timings;

DOOMS_API dooms_code dooms_whisper_open(const dooms_whisper_paths * paths, const dooms_whisper_options * options, dooms_whisper ** out);
DOOMS_API void dooms_whisper_close(dooms_whisper * whisper);
DOOMS_API int32_t dooms_whisper_sample_rate(const dooms_whisper * whisper);   /* 16000; 0 for NULL */
/* n mono samples in [-1, 1] at dooms_whisper_sample_rate -> NUL-terminated UTF-8 text, with *needed and DOOMS_RESOURCE_EXHAUSTED as dooms_llm_generate. A text that
 * did not fit stays in the handle: dooms_whisper_last_text fetches it without transcribing again. */
DOOMS_API dooms_code dooms_whisper_transcribe(dooms_whisper * whisper, const float * samples, size_t n, char * out, size_t capacity, size_t * needed);
/* Any audio file Core Audio decodes (WAV, AIFF, CAF, MP3, M4A/AAC, FLAC, ...), resampled to the model's rate and mixed to mono. */
DOOMS_API dooms_code dooms_whisper_transcribe_file(dooms_whisper * whisper, const char * path, char * out, size_t capacity, size_t * needed);
/* The text of the last successful transcription (DOOMS_NOT_FOUND before one). */
DOOMS_API dooms_code dooms_whisper_last_text(const dooms_whisper * whisper, char * out, size_t capacity, size_t * needed);
DOOMS_API dooms_code dooms_whisper_last_timings(const dooms_whisper * whisper, dooms_whisper_timings * out);

/* ---------------------------------------------------------------- text to speech (Kokoro-82M: prosody on Core ML, the vocoder on the engine's own Metal kernels)
 * The input is phonemes (misaki's symbols, UTF-8; the Python binding converts English text with misaki). Output: mono float samples in [-1, 1] at
 * dooms_tts_sample_rate. Files from tools/convert/convert_kokoro.py (docs/formats/kokoro-gguf.md). */
typedef struct dooms_tts dooms_tts;

typedef struct dooms_tts_paths {
    size_t struct_size;
    const char * prosody;  /* prosody-fp32.mlpackage */
    const char * f0n;      /* f0n-fp32.mlpackage */
    const char * model;    /* kokoro.gguf */
} dooms_tts_paths;

typedef struct dooms_tts_options {
    size_t struct_size;
    const char * units;    /* the Core ML packages' units: "cpu" (NULL: the default; fastest for their LSTMs), "cpu_gpu", "all" */
} dooms_tts_options;

typedef struct dooms_tts_timings {
    size_t struct_size;
    double prosody_ms, f0n_ms, source_ms, vocoder_ms, istft_ms, total_ms;   /* wall time of the last synthesis, per stage and in all */
    double audio_s;
    int32_t tokens, frames;
} dooms_tts_timings;

DOOMS_API dooms_code dooms_tts_open(const dooms_tts_paths * paths, const dooms_tts_options * options, dooms_tts ** out);
DOOMS_API void dooms_tts_close(dooms_tts * tts);
DOOMS_API int32_t dooms_tts_sample_rate(const dooms_tts * tts);    /* 24000; 0 for NULL */
DOOMS_API int32_t dooms_tts_max_phonemes(const dooms_tts * tts);   /* per call; 0 for NULL */
DOOMS_API int32_t dooms_tts_voice_count(const dooms_tts * tts);
DOOMS_API const char * dooms_tts_voice_name(const dooms_tts * tts, int32_t index);   /* NULL outside [0, count); valid while the handle lives */
/* phonemes -> samples (capacity in floats; *needed receives the sample count). A result that did not fit stays in the handle: dooms_tts_last_audio fetches
 * it. speed > 0 (1: the model's pace); seed: the sine source's noise (the same seed gives the same samples). */
DOOMS_API dooms_code dooms_tts_synthesize(dooms_tts * tts, const char * phonemes, const char * voice, float speed, uint64_t seed, float * out, size_t capacity, size_t * needed);
DOOMS_API dooms_code dooms_tts_last_audio(const dooms_tts * tts, float * out, size_t capacity, size_t * needed);
DOOMS_API dooms_code dooms_tts_last_timings(const dooms_tts * tts, dooms_tts_timings * out);

#ifdef __cplusplus
}
#endif
#endif /* DOOMS_C_H */
