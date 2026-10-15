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

#ifdef __cplusplus
}
#endif
#endif /* DOOMS_C_H */
