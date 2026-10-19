// Implementation of include/dooms/dooms_c.h: a thin, exception-free layer over the C++ pipelines. Every entry point validates its arguments, converts the
// Status of the C++ call to a dooms_code and stores the message for dooms_last_error().
#include "dooms/dooms_c.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/rgb_image.h"
#include "pipeline/vla_chunk_runner.h"
#include "pipeline/kokoro_pipeline.h"
#include "pipeline/whisper_pipeline.h"
#ifdef DOOMS_WITH_LLAMA
#include "models/vlm/tokenizer.h"
#include "pipeline/vla_pipeline.h"
#include "pipeline/llm_pipeline.h"
#include "pipeline/vlm_pipeline.h"
#endif

#ifndef DOOMS_VERSION_STRING
#define DOOMS_VERSION_STRING "0.1.0"
#endif

namespace {

thread_local std::string g_last_error;

dooms_code to_code(dooms::Code c) {
    switch (c) {
        case dooms::Code::Ok: return DOOMS_OK;
        case dooms::Code::InvalidArgument: case dooms::Code::OutOfRange: return DOOMS_INVALID_ARGUMENT;
        case dooms::Code::NotFound: return DOOMS_NOT_FOUND;
        case dooms::Code::Unsupported: return DOOMS_UNSUPPORTED;
        case dooms::Code::DataLoss: return DOOMS_DATA_LOSS;
        case dooms::Code::ResourceExhausted: return DOOMS_RESOURCE_EXHAUSTED;
        case dooms::Code::IoError: return DOOMS_IO_ERROR;
        default: return DOOMS_INTERNAL;
    }
}

dooms_code fail(dooms_code code, const std::string & message) {
    g_last_error = message;
    return code;
}

dooms_code from_status(const dooms::Status & st) {
    if (st.is_ok()) return DOOMS_OK;
    return fail(to_code(st.code()), st.to_string());
}

// Runs `body` so that no exception crosses the C boundary.
template <class F>
dooms_code guarded(F && body) {
    g_last_error.clear();
    try {
        return body();
    } catch (const std::exception & e) {
        return fail(DOOMS_INTERNAL, std::string("unexpected exception: ") + e.what());
    } catch (...) {
        return fail(DOOMS_INTERNAL, "unexpected exception");
    }
}

constexpr int64_t kMaxSide = 1 << 14;

// Copies a caller's image into the C++ type after validating it.
dooms_code to_rgb_image(const dooms_image * in, const char * what, dooms::RgbImage * out) {
    if (!in) return fail(DOOMS_INVALID_ARGUMENT, std::string(what) + " is NULL");
    if (in->width < 1 || in->height < 1 || in->width > kMaxSide || in->height > kMaxSide) return fail(DOOMS_INVALID_ARGUMENT, std::string(what) + " has a side outside [1, 16384]");
    if (!in->rgb) return fail(DOOMS_INVALID_ARGUMENT, std::string(what) + " has no pixel data");
    out->width = in->width;
    out->height = in->height;
    out->pixels.assign(in->rgb, in->rgb + static_cast<size_t>(in->width) * static_cast<size_t>(in->height) * 3);
    return DOOMS_OK;
}

// A NUL-terminated copy of `text` into a caller's buffer; *needed (optional) gets the size it takes. Too small: nothing written, DOOMS_RESOURCE_EXHAUSTED.
dooms_code write_text(const std::string & text, char * out, size_t capacity, size_t * needed) {
    if (needed) *needed = text.size() + 1;
    if (!out && capacity > 0) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
    if (capacity < text.size() + 1) return fail(DOOMS_RESOURCE_EXHAUSTED, "the output buffer holds " + std::to_string(capacity) + " bytes, " + std::to_string(text.size() + 1) + " are needed");
    std::memcpy(out, text.c_str(), text.size() + 1);
    return DOOMS_OK;
}

const char * or_default(const char * s, const char * fallback) { return (s && *s) ? s : fallback; }

// struct_size lets a newer library accept an older caller's smaller struct; zero (an uninitialized caller) is rejected.
template <class T>
bool struct_ok(const T * p, const char * name, dooms_code * code) {
    if (!p) { *code = fail(DOOMS_INVALID_ARGUMENT, std::string(name) + " is NULL"); return false; }
    if (p->struct_size < sizeof(size_t) * 2) { *code = fail(DOOMS_INVALID_ARGUMENT, std::string(name) + ".struct_size is not set"); return false; }
    return true;
}

}  // namespace

extern "C" {

DOOMS_API const char * dooms_version(void) { return DOOMS_VERSION_STRING; }
DOOMS_API int32_t dooms_c_api_version(void) { return DOOMS_C_API_VERSION; }
DOOMS_API const char * dooms_last_error(void) { return g_last_error.c_str(); }

DOOMS_API const char * dooms_code_name(dooms_code code) {
    switch (code) {
        case DOOMS_OK: return "Ok";
        case DOOMS_INVALID_ARGUMENT: return "InvalidArgument";
        case DOOMS_NOT_FOUND: return "NotFound";
        case DOOMS_DATA_LOSS: return "DataLoss";
        case DOOMS_UNSUPPORTED: return "Unsupported";
        case DOOMS_RESOURCE_EXHAUSTED: return "ResourceExhausted";
        case DOOMS_IO_ERROR: return "IoError";
        case DOOMS_INTERNAL: return "Internal";
    }
    return "Unknown";
}

// ------------------------------------------------------------------------------------------------ VLM
}  // extern "C"

#ifdef DOOMS_WITH_LLAMA
struct dooms_vlm {
    std::unique_ptr<dooms::VlmPipeline> pipeline;
};
struct dooms_llm {
    std::unique_ptr<dooms::Llm> llm;
    dooms_llm_generation_info last{};  // what dooms_llm_last_generation reports; struct_size unused here
};
// Shared by a dooms_vla and the runner opened on it, so that either may be closed first.
struct VlaModel {
    VlaModel(dooms::VlaPipeline p, dooms::Tokenizer t) : pipeline(std::move(p)), tokenizer(std::move(t)) {}
    dooms::VlaPipeline pipeline;
    dooms::Tokenizer tokenizer;
    std::atomic<bool> runner_open{false};  // while set, the pipeline belongs to the runner's worker thread
};
struct dooms_vla {
    std::shared_ptr<VlaModel> model;
};
struct dooms_vla_runner {
    std::shared_ptr<VlaModel> model;               // declared first: released after the runner has joined its thread
    std::unique_ptr<dooms::VlaChunkRunner> runner;
    std::string task;                              // the last task and its tokens, reused while the task does not change
    std::vector<int32_t> tokens;
    std::vector<uint8_t> mask;
    dooms::VlaAction action;
    std::string last_error;                        // storage behind dooms_vla_runner_stats.last_error
    ~dooms_vla_runner() {
        runner.reset();
        if (model) model->runner_open = false;
    }
};

namespace {

dooms_code vla_check_inputs(const dooms::VlaPipeline & p, const dooms_image * cameras, int32_t n_cameras, const char * task, const float * state, int32_t n_state) {
    if (n_cameras != p.n_cameras() || !cameras) return fail(DOOMS_INVALID_ARGUMENT, "expected " + std::to_string(p.n_cameras()) + " cameras");
    if (!task || !*task) return fail(DOOMS_INVALID_ARGUMENT, "task is NULL or empty");
    if (n_state < 0 || n_state > p.state_dim() || (n_state > 0 && !state)) return fail(DOOMS_INVALID_ARGUMENT, "n_state must be in [0, state_dim] and state non-NULL");
    return DOOMS_OK;
}

// Cameras and state into `obs` (the caller fills the tokens).
dooms_code vla_fill_observation(const dooms_image * cameras, int32_t n_cameras, const float * state, int32_t n_state, dooms::VlaObservation * obs) {
    for (int32_t i = 0; i < n_cameras; i++) {
        dooms::RgbImage img;
        dooms_code c = to_rgb_image(&cameras[i], "camera", &img);
        if (c != DOOMS_OK) return c;
        auto cam = dooms::camera_from_rgb(img);
        if (!cam.is_ok()) return from_status(cam.status());
        obs->cameras.push_back(std::move(cam).value());
    }
    if (n_state > 0) obs->state.assign(state, state + n_state);
    return DOOMS_OK;
}

}  // namespace
#else
struct dooms_vlm {};
struct dooms_llm {};
struct dooms_vla {};
struct dooms_vla_runner {};
#endif

extern "C" {

DOOMS_API dooms_code dooms_vlm_open(const dooms_vlm_paths * paths, const dooms_vlm_options * options, dooms_vlm ** out) {
    return guarded([&]() -> dooms_code {
        if (!out) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
#ifdef DOOMS_WITH_LLAMA
        dooms_code c;
        if (!struct_ok(paths, "paths", &c)) return c;
        if (!paths->vision || !paths->text || !paths->config) return fail(DOOMS_INVALID_ARGUMENT, "paths.vision, paths.text and paths.config are required");
        dooms::VlmOptions vo;
        if (options) {
            if (!struct_ok(options, "options", &c)) return c;
            if (options->n_threads < 0 || options->n_threads > 1024) return fail(DOOMS_INVALID_ARGUMENT, "options.n_threads is outside [0, 1024]");
            vo.vision_units = or_default(options->vision_units, "cpu_ne");
            vo.text.n_threads = options->n_threads;
        }
        auto p = dooms::VlmPipeline::create({paths->vision, paths->text, paths->config}, vo);
        if (!p.is_ok()) return from_status(p.status());
        auto h = std::make_unique<dooms_vlm>();
        h->pipeline = std::make_unique<dooms::VlmPipeline>(std::move(p).value());
        *out = h.release();
        return DOOMS_OK;
#else
        (void) paths; (void) options;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API void dooms_vlm_close(dooms_vlm * vlm) { delete vlm; }

DOOMS_API dooms_code dooms_vlm_generate(dooms_vlm * vlm, const dooms_image * image, const char * prompt, int32_t max_new_tokens, char * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!vlm) return fail(DOOMS_INVALID_ARGUMENT, "vlm is NULL");
        if (!prompt || !*prompt) return fail(DOOMS_INVALID_ARGUMENT, "prompt is NULL or empty");
        if (max_new_tokens < 1 || max_new_tokens > 4096) return fail(DOOMS_INVALID_ARGUMENT, "max_new_tokens must be in [1, 4096]");
        if (!out && capacity > 0) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        dooms::RgbImage img;
        dooms_code c = to_rgb_image(image, "image", &img);
        if (c != DOOMS_OK) return c;
        auto r = vlm->pipeline->generate(img, prompt, max_new_tokens);
        if (!r.is_ok()) return from_status(r.status());
        const std::string & text = r.value().text;
        if (needed) *needed = text.size() + 1;
        if (capacity < text.size() + 1) return fail(DOOMS_RESOURCE_EXHAUSTED, "the output buffer holds " + std::to_string(capacity) + " bytes, " + std::to_string(text.size() + 1) + " are needed");
        std::memcpy(out, text.c_str(), text.size() + 1);
        return DOOMS_OK;
#else
        (void) vlm; (void) image; (void) prompt; (void) max_new_tokens; (void) out; (void) capacity; (void) needed;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_vlm_generate_stream(dooms_vlm * vlm, const dooms_image * image, const char * prompt, int32_t max_new_tokens, dooms_text_callback callback, void * user) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!vlm) return fail(DOOMS_INVALID_ARGUMENT, "vlm is NULL");
        if (!callback) return fail(DOOMS_INVALID_ARGUMENT, "callback is NULL");
        if (!prompt || !*prompt) return fail(DOOMS_INVALID_ARGUMENT, "prompt is NULL or empty");
        if (max_new_tokens < 1 || max_new_tokens > 4096) return fail(DOOMS_INVALID_ARGUMENT, "max_new_tokens must be in [1, 4096]");
        dooms::RgbImage img;
        dooms_code c = to_rgb_image(image, "image", &img);
        if (c != DOOMS_OK) return c;
        auto r = vlm->pipeline->generate(img, prompt, max_new_tokens, [&](const std::string & piece) { return callback(piece.c_str(), user) == 0; });
        return r.is_ok() ? DOOMS_OK : from_status(r.status());
#else
        (void) vlm; (void) image; (void) prompt; (void) max_new_tokens; (void) callback; (void) user;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

// ------------------------------------------------------------------------------------------------ LLM
}  // extern "C"

#ifdef DOOMS_WITH_LLAMA
namespace {

dooms_code llm_generate_options(const dooms_llm_sampling * sampling, const char * const * stop, int32_t n_stop, int32_t max_new_tokens, dooms::LlmGenerateOptions * o) {
    if (n_stop < 0 || (n_stop > 0 && !stop)) return fail(DOOMS_INVALID_ARGUMENT, "n_stop is negative or stop is NULL");
    o->max_new_tokens = max_new_tokens;
    if (sampling) {
        dooms_code c;
        if (!struct_ok(sampling, "sampling", &c)) return c;
        o->sampler.temperature = sampling->temperature;
        o->sampler.top_k = sampling->top_k;
        o->sampler.top_p = sampling->top_p;
        o->sampler.repeat_penalty = sampling->repeat_penalty;
        o->sampler.seed = sampling->seed;
    }
    for (int32_t i = 0; i < n_stop; i++) {
        if (!stop[i] || !*stop[i]) return fail(DOOMS_INVALID_ARGUMENT, "a stop string is NULL or empty");
        o->stop.emplace_back(stop[i]);
    }
    return DOOMS_OK;
}

dooms_code llm_messages(const dooms_chat_message * messages, int32_t n, std::vector<dooms::ChatMessage> * out) {
    if (!messages || n < 1) return fail(DOOMS_INVALID_ARGUMENT, "messages is NULL or n_messages < 1");
    for (int32_t i = 0; i < n; i++) {
        if (!messages[i].role || !*messages[i].role || !messages[i].content) return fail(DOOMS_INVALID_ARGUMENT, "a message has no role or no content");
        out->push_back({messages[i].role, messages[i].content});
    }
    return DOOMS_OK;
}

// Runs one generation and records its counts and finish reason for dooms_llm_last_generation (the callers clear the record first, so a failed call leaves DOOMS_FINISH_NONE).
template <class F>
dooms_code llm_run(dooms_llm * llm, F && call, std::string * text) {
    dooms::LlmTimings tm;
    auto r = call(&tm);
    if (!r.is_ok()) return from_status(r.status());
    const dooms::LlmGeneration & g = r.value();
    llm->last.prompt_tokens = static_cast<int32_t>(tm.prompt_tokens);
    llm->last.completion_tokens = static_cast<int32_t>(tm.generated_tokens);
    llm->last.finish_reason = g.cancelled ? DOOMS_FINISH_CANCELLED : g.stopped ? DOOMS_FINISH_STOP : DOOMS_FINISH_LENGTH;
    if (text) *text = g.text;
    return DOOMS_OK;
}

dooms_code llm_write(const std::string & text, char * out, size_t capacity, size_t * needed) { return write_text(text, out, capacity, needed); }

}  // namespace
#endif

extern "C" {

DOOMS_API dooms_code dooms_llm_open(const char * gguf_path, const dooms_llm_options * options, dooms_llm ** out) {
    return guarded([&]() -> dooms_code {
        if (!out) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
#ifdef DOOMS_WITH_LLAMA
        if (!gguf_path || !*gguf_path) return fail(DOOMS_INVALID_ARGUMENT, "gguf_path is NULL or empty");
        dooms::LlmOptions lo;
        if (options) {
            dooms_code c;
            if (!struct_ok(options, "options", &c)) return c;
            if (options->n_threads < 0 || options->n_threads > 1024) return fail(DOOMS_INVALID_ARGUMENT, "options.n_threads is outside [0, 1024]");
            if (options->n_ctx < 0) return fail(DOOMS_INVALID_ARGUMENT, "options.n_ctx is negative");
            if (options->n_ctx > 0) lo.text.n_ctx = options->n_ctx;
            if (lo.text.n_batch > lo.text.n_ctx) lo.text.n_batch = lo.text.n_ctx;
            lo.text.n_threads = options->n_threads;
            lo.text.n_gpu_layers = options->n_gpu_layers;
            if (options->struct_size >= offsetof(dooms_llm_options, device) + sizeof(options->device)) lo.device = or_default(options->device, "auto");
        }
        auto m = dooms::Llm::load(gguf_path, lo);
        if (!m.is_ok()) return from_status(m.status());
        auto h = std::make_unique<dooms_llm>();
        h->llm = std::make_unique<dooms::Llm>(std::move(m).value());
        *out = h.release();
        return DOOMS_OK;
#else
        (void) gguf_path; (void) options;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API void dooms_llm_close(dooms_llm * llm) { delete llm; }

DOOMS_API int32_t dooms_llm_context_size(const dooms_llm * llm) {
#ifdef DOOMS_WITH_LLAMA
    return llm && llm->llm ? llm->llm->context_size() : 0;
#else
    (void) llm;
    return 0;
#endif
}

DOOMS_API dooms_code dooms_llm_generate(dooms_llm * llm, const char * prompt, int32_t max_new_tokens, const dooms_llm_sampling * sampling, const char * const * stop,
                                        int32_t n_stop, char * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
        llm->last = dooms_llm_generation_info{};  // a call rejected before it runs must not leave the previous record
        if (!prompt || !*prompt) return fail(DOOMS_INVALID_ARGUMENT, "prompt is NULL or empty");
        dooms::LlmGenerateOptions o;
        dooms_code c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        std::string text;
        c = llm_run(llm, [&](dooms::LlmTimings * tm) { return llm->llm->generate(prompt, o, nullptr, tm); }, &text);
        if (c != DOOMS_OK) return c;
        return llm_write(text, out, capacity, needed);
#else
        (void) llm; (void) prompt; (void) max_new_tokens; (void) sampling; (void) stop; (void) n_stop; (void) out; (void) capacity; (void) needed;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_llm_generate_stream(dooms_llm * llm, const char * prompt, int32_t max_new_tokens, const dooms_llm_sampling * sampling,
                                               const char * const * stop, int32_t n_stop, dooms_text_callback callback, void * user) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
        llm->last = dooms_llm_generation_info{};  // a call rejected before it runs must not leave the previous record
        if (!callback) return fail(DOOMS_INVALID_ARGUMENT, "callback is NULL");
        if (!prompt || !*prompt) return fail(DOOMS_INVALID_ARGUMENT, "prompt is NULL or empty");
        dooms::LlmGenerateOptions o;
        dooms_code c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        return llm_run(llm, [&](dooms::LlmTimings * tm) {
            return llm->llm->generate(prompt, o, [&](const std::string & piece) { return callback(piece.c_str(), user) == 0; }, tm);
        }, nullptr);
#else
        (void) llm; (void) prompt; (void) max_new_tokens; (void) sampling; (void) stop; (void) n_stop; (void) callback; (void) user;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_llm_chat(dooms_llm * llm, const dooms_chat_message * messages, int32_t n_messages, int32_t max_new_tokens, const dooms_llm_sampling * sampling,
                                    const char * const * stop, int32_t n_stop, char * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
        llm->last = dooms_llm_generation_info{};  // a call rejected before it runs must not leave the previous record
        std::vector<dooms::ChatMessage> msgs;
        dooms_code c = llm_messages(messages, n_messages, &msgs);
        if (c != DOOMS_OK) return c;
        dooms::LlmGenerateOptions o;
        c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        std::string text;
        c = llm_run(llm, [&](dooms::LlmTimings * tm) { return llm->llm->chat(msgs, o, nullptr, tm); }, &text);
        if (c != DOOMS_OK) return c;
        return llm_write(text, out, capacity, needed);
#else
        (void) llm; (void) messages; (void) n_messages; (void) max_new_tokens; (void) sampling; (void) stop; (void) n_stop; (void) out; (void) capacity; (void) needed;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_llm_chat_stream(dooms_llm * llm, const dooms_chat_message * messages, int32_t n_messages, int32_t max_new_tokens, const dooms_llm_sampling * sampling,
                                           const char * const * stop, int32_t n_stop, dooms_text_callback callback, void * user) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
        llm->last = dooms_llm_generation_info{};  // a call rejected before it runs must not leave the previous record
        if (!callback) return fail(DOOMS_INVALID_ARGUMENT, "callback is NULL");
        std::vector<dooms::ChatMessage> msgs;
        dooms_code c = llm_messages(messages, n_messages, &msgs);
        if (c != DOOMS_OK) return c;
        dooms::LlmGenerateOptions o;
        c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        return llm_run(llm, [&](dooms::LlmTimings * tm) {
            return llm->llm->chat(msgs, o, [&](const std::string & piece) { return callback(piece.c_str(), user) == 0; }, tm);
        }, nullptr);
#else
        (void) llm; (void) messages; (void) n_messages; (void) max_new_tokens; (void) sampling; (void) stop; (void) n_stop; (void) callback; (void) user;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_llm_last_generation(const dooms_llm * llm, dooms_llm_generation_info * out) {
    return guarded([&]() -> dooms_code {
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
        dooms_code c;
        if (!struct_ok(out, "out", &c)) return c;
#ifdef DOOMS_WITH_LLAMA
        dooms_llm_generation_info info = llm->last;
#else
        dooms_llm_generation_info info{};
#endif
        // copy only what the caller's (possibly older, smaller) struct holds, keeping its struct_size
        info.struct_size = out->struct_size;
        std::memcpy(out, &info, std::min(out->struct_size, sizeof info));
        return DOOMS_OK;
    });
}

DOOMS_API dooms_code dooms_llm_count_tokens(dooms_llm * llm, const char * text, int32_t add_special, int32_t * count) {
    return guarded([&]() -> dooms_code {
        if (!count) return fail(DOOMS_INVALID_ARGUMENT, "count is NULL");
        *count = 0;
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
        if (!text || !*text) return fail(DOOMS_INVALID_ARGUMENT, "text is NULL or empty");
#ifdef DOOMS_WITH_LLAMA
        auto t = llm->llm->tokenize(text, add_special != 0, /*parse_special=*/false);
        if (!t.is_ok()) return from_status(t.status());
        *count = static_cast<int32_t>(t.value().size());
        return DOOMS_OK;
#else
        (void) add_special;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_llm_count_chat_tokens(dooms_llm * llm, const dooms_chat_message * messages, int32_t n_messages, int32_t * count) {
    return guarded([&]() -> dooms_code {
        if (!count) return fail(DOOMS_INVALID_ARGUMENT, "count is NULL");
        *count = 0;
        if (!llm) return fail(DOOMS_INVALID_ARGUMENT, "llm is NULL");
#ifdef DOOMS_WITH_LLAMA
        std::vector<dooms::ChatMessage> msgs;
        dooms_code c = llm_messages(messages, n_messages, &msgs);
        if (c != DOOMS_OK) return c;
        auto t = llm->llm->chat_tokens(msgs);
        if (!t.is_ok()) return from_status(t.status());
        *count = static_cast<int32_t>(t.value().size());
        return DOOMS_OK;
#else
        (void) messages; (void) n_messages;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

// ------------------------------------------------------------------------------------------------ VLA
DOOMS_API dooms_code dooms_vla_open(const dooms_vla_paths * paths, const dooms_vla_options * options, dooms_vla ** out) {
    return guarded([&]() -> dooms_code {
        if (!out) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
#ifdef DOOMS_WITH_LLAMA
        dooms_code c;
        if (!struct_ok(paths, "paths", &c)) return c;
        if (!paths->vision || !paths->prefix || !paths->expert || !paths->host || !paths->vocab) return fail(DOOMS_INVALID_ARGUMENT, "paths.vision, prefix, expert, host and vocab are required");
        dooms::VlaOptions vo;
        if (options) {
            if (!struct_ok(options, "options", &c)) return c;
            vo.vision_units = vo.prefix_units = vo.expert_units = or_default(options->units, "cpu_ne");
            vo.seed = options->seed;
        }
        auto p = dooms::VlaPipeline::create({paths->vision, paths->prefix, paths->expert, paths->host}, vo);
        if (!p.is_ok()) return from_status(p.status());
        auto t = dooms::Tokenizer::load(paths->vocab);
        if (!t.is_ok()) return from_status(t.status());
        auto h = std::make_unique<dooms_vla>();
        h->model = std::make_shared<VlaModel>(std::move(p).value(), std::move(t).value());
        *out = h.release();
        return DOOMS_OK;
#else
        (void) paths; (void) options;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API void dooms_vla_close(dooms_vla * vla) { delete vla; }

DOOMS_API dooms_code dooms_vla_info_get(const dooms_vla * vla, dooms_vla_info * out) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!vla || !out) return fail(DOOMS_INVALID_ARGUMENT, "vla and out must be non-NULL");
        const dooms::VlaPipeline & p = vla->model->pipeline;
        out->n_cameras = static_cast<int32_t>(p.n_cameras());
        out->chunk_size = static_cast<int32_t>(p.chunk_size());
        out->action_dim = static_cast<int32_t>(p.action_dim());
        out->state_dim = static_cast<int32_t>(p.state_dim());
        out->max_task_tokens = static_cast<int32_t>(p.tokenizer_max_length());
        return DOOMS_OK;
#else
        (void) vla; (void) out;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_vla_infer(dooms_vla * vla, const dooms_image * cameras, int32_t n_cameras, const char * task, const float * state, int32_t n_state, const float * noise, float * actions, size_t capacity) {
    return guarded([&]() -> dooms_code {
#ifdef DOOMS_WITH_LLAMA
        if (!vla) return fail(DOOMS_INVALID_ARGUMENT, "vla is NULL");
        if (vla->model->runner_open)
            return fail(DOOMS_INVALID_ARGUMENT, "this VLA handle has an open runner (dooms_vla_runner_open), which owns the model: close the runner or open a second handle");
        dooms::VlaPipeline & p = vla->model->pipeline;
        dooms_code c = vla_check_inputs(p, cameras, n_cameras, task, state, n_state);
        if (c != DOOMS_OK) return c;
        const size_t need = static_cast<size_t>(p.chunk_size() * p.action_dim());
        if (!actions || capacity < need) return fail(DOOMS_RESOURCE_EXHAUSTED, "actions must hold chunk_size * action_dim = " + std::to_string(need) + " floats");
        dooms::VlaObservation obs;
        c = vla_fill_observation(cameras, n_cameras, state, n_state, &obs);
        if (c != DOOMS_OK) return c;
        auto tokens = vla->model->tokenizer.encode_task(task, p.tokenizer_max_length());
        if (!tokens.is_ok()) return from_status(tokens.status());
        obs.tokens = tokens.value().tokens;
        obs.token_mask = tokens.value().mask;
        std::vector<float> start;
        if (noise) start.assign(noise, noise + need);
        auto r = noise ? p.infer(obs, &start) : p.infer(obs);
        if (!r.is_ok()) return from_status(r.status());
        std::memcpy(actions, r.value().data(), need * sizeof(float));
        return DOOMS_OK;
#else
        (void) vla; (void) cameras; (void) n_cameras; (void) task; (void) state; (void) n_state; (void) noise; (void) actions; (void) capacity;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

// ------------------------------------------------------------------------------------------------ VLA runner (asynchronous chunks)
DOOMS_API dooms_code dooms_vla_runner_options_init(dooms_vla_runner_options * options) {
    return guarded([&]() -> dooms_code {
        if (!options) return fail(DOOMS_INVALID_ARGUMENT, "options is NULL");
        std::memset(options, 0, sizeof *options);
        options->struct_size = sizeof *options;
        options->control_hz = dooms::kVlaRunnerDefaultControlHz;
        options->chunk_size_threshold = dooms::kVlaRunnerDefaultChunkThreshold;
        options->new_weight = dooms::kVlaRunnerDefaultNewWeight;
        return DOOMS_OK;
    });
}

DOOMS_API dooms_code dooms_vla_aggregate_weight(const char * name, double * new_weight) {
    return guarded([&]() -> dooms_code {
        if (!name || !new_weight) return fail(DOOMS_INVALID_ARGUMENT, "name and new_weight must be non-NULL");
        auto w = dooms::vla_aggregate_weight(name);
        if (!w.is_ok()) return from_status(w.status());
        *new_weight = w.value();
        return DOOMS_OK;
    });
}

DOOMS_API dooms_code dooms_vla_runner_open(dooms_vla * vla, const dooms_vla_runner_options * options, dooms_vla_runner ** out) {
    return guarded([&]() -> dooms_code {
        if (!out) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
#ifdef DOOMS_WITH_LLAMA
        if (!vla) return fail(DOOMS_INVALID_ARGUMENT, "vla is NULL");
        dooms::VlaChunkRunnerOptions o;
        if (options) {
            dooms_code c;
            if (!struct_ok(options, "options", &c)) return c;
            if (options->struct_size < sizeof(dooms_vla_runner_options)) return fail(DOOMS_INVALID_ARGUMENT, "options.struct_size is smaller than dooms_vla_runner_options");
            o.control_hz = options->control_hz;
            o.chunk_size_threshold = options->chunk_size_threshold;
            o.new_weight = options->new_weight;
        }
        bool expected = false;
        if (!vla->model->runner_open.compare_exchange_strong(expected, true))
            return fail(DOOMS_INVALID_ARGUMENT, "this VLA handle already has an open runner; close it first or open a second handle");
        auto h = std::make_unique<dooms_vla_runner>();
        h->model = vla->model;  // from here on the handle's destructor clears runner_open, also on the error path below
        auto r = dooms::VlaChunkRunner::create(dooms::vla_pipeline_infer_fn(&h->model->pipeline), o);
        if (!r.is_ok()) return from_status(r.status());
        h->runner = std::move(r).value();
        *out = h.release();
        return DOOMS_OK;
#else
        (void) vla; (void) options;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API void dooms_vla_runner_close(dooms_vla_runner * runner) { delete runner; }

DOOMS_API dooms_code dooms_vla_runner_wants_observation(dooms_vla_runner * runner, int32_t * wants) {
    return guarded([&]() -> dooms_code {
        if (!runner || !wants) return fail(DOOMS_INVALID_ARGUMENT, "runner and wants must be non-NULL");
#ifdef DOOMS_WITH_LLAMA
        *wants = runner->runner->wants_observation() ? 1 : 0;
        return DOOMS_OK;
#else
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_vla_runner_submit(dooms_vla_runner * runner, const dooms_image * cameras, int32_t n_cameras, const char * task, const float * state,
                                             int32_t n_state, double timestamp_s, int32_t * started) {
    return guarded([&]() -> dooms_code {
        if (started) *started = 0;
        if (!runner) return fail(DOOMS_INVALID_ARGUMENT, "runner is NULL");
#ifdef DOOMS_WITH_LLAMA
        if (!std::isfinite(timestamp_s)) return fail(DOOMS_INVALID_ARGUMENT, "timestamp_s must be finite");
        const dooms::VlaPipeline & p = runner->model->pipeline;
        dooms_code c = vla_check_inputs(p, cameras, n_cameras, task, state, n_state);
        if (c != DOOMS_OK) return c;
        // Only this (the control) thread turns "wanted" from true to false, by submitting, so checking first skips converting an unwanted
        // observation without a race; submit_observation checks again.
        if (!runner->runner->wants_observation()) return DOOMS_OK;
        if (runner->task != task) {
            auto tokens = runner->model->tokenizer.encode_task(task, p.tokenizer_max_length());
            if (!tokens.is_ok()) return from_status(tokens.status());
            runner->task = task;
            runner->tokens = tokens.value().tokens;
            runner->mask = tokens.value().mask;
        }
        dooms::VlaObservation obs;
        c = vla_fill_observation(cameras, n_cameras, state, n_state, &obs);
        if (c != DOOMS_OK) return c;
        obs.tokens = runner->tokens;
        obs.token_mask = runner->mask;
        auto r = runner->runner->submit_observation(std::move(obs), timestamp_s);
        if (!r.is_ok()) return from_status(r.status());
        if (started) *started = r.value() ? 1 : 0;
        return DOOMS_OK;
#else
        (void) cameras; (void) n_cameras; (void) task; (void) state; (void) n_state; (void) timestamp_s;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_vla_runner_next_action(dooms_vla_runner * runner, double now_s, float * action, size_t capacity, dooms_vla_action_info * info) {
    return guarded([&]() -> dooms_code {
        if (!runner) return fail(DOOMS_INVALID_ARGUMENT, "runner is NULL");
        dooms_code c;
        if (!struct_ok(info, "info", &c)) return c;
        if (info->struct_size < sizeof(dooms_vla_action_info)) return fail(DOOMS_INVALID_ARGUMENT, "info.struct_size is smaller than dooms_vla_action_info");
#ifdef DOOMS_WITH_LLAMA
        const size_t need = static_cast<size_t>(runner->model->pipeline.action_dim());
        if (!action || capacity < need) return fail(DOOMS_RESOURCE_EXHAUSTED, "action must hold action_dim = " + std::to_string(need) + " floats");
        dooms::Status st = runner->runner->next_action(now_s, &runner->action);
        if (!st.is_ok()) return from_status(st);
        if (runner->action.available && runner->action.values.size() != need) return fail(DOOMS_INTERNAL, "the runner returned an action of the wrong width");
        info->available = runner->action.available ? 1 : 0;
        info->skipped = static_cast<int32_t>(std::min<int64_t>(runner->action.skipped, INT32_MAX));
        info->step = runner->action.step;
        if (runner->action.available) std::memcpy(action, runner->action.values.data(), need * sizeof(float));
        return DOOMS_OK;
#else
        (void) now_s; (void) action; (void) capacity;
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_vla_runner_wait_idle(dooms_vla_runner * runner, double timeout_s, int32_t * idle) {
    return guarded([&]() -> dooms_code {
        if (idle) *idle = 0;
        if (!runner) return fail(DOOMS_INVALID_ARGUMENT, "runner is NULL");
        if (!std::isfinite(timeout_s) || timeout_s < 0) return fail(DOOMS_INVALID_ARGUMENT, "timeout_s must be finite and >= 0");
#ifdef DOOMS_WITH_LLAMA
        const bool done = runner->runner->wait_idle(timeout_s);
        if (idle) *idle = done ? 1 : 0;
        return DOOMS_OK;
#else
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

DOOMS_API dooms_code dooms_vla_runner_stats_get(dooms_vla_runner * runner, dooms_vla_runner_stats * out) {
    return guarded([&]() -> dooms_code {
        if (!runner) return fail(DOOMS_INVALID_ARGUMENT, "runner is NULL");
        dooms_code c;
        if (!struct_ok(out, "out", &c)) return c;
        if (out->struct_size < sizeof(dooms_vla_runner_stats)) return fail(DOOMS_INVALID_ARGUMENT, "out.struct_size is smaller than dooms_vla_runner_stats");
#ifdef DOOMS_WITH_LLAMA
        const dooms::VlaChunkRunnerStats s = runner->runner->stats();
        out->actions_returned = s.actions_returned;
        out->waiting_calls = s.waiting_calls;
        out->stalled_calls = s.stalled_calls;
        out->late_skipped = s.late_skipped;
        out->stale_dropped = s.stale_dropped;
        out->blended = s.blended;
        out->chunks_received = s.chunks_received;
        out->chunks_stale = s.chunks_stale;
        out->inferences_started = s.inferences_started;
        out->inferences_failed = s.inferences_failed;
        out->queued = s.queued;
        out->in_flight = s.in_flight ? 1 : 0;
        out->last_error_code = to_code(s.last_error.code());
        out->last_inference_ms = s.last_inference_ms;
        runner->last_error = s.last_error.is_ok() ? "" : s.last_error.to_string();
        out->last_error = runner->last_error.c_str();
        return DOOMS_OK;
#else
        return fail(DOOMS_UNSUPPORTED, "this build of dooms.cpp was made without llama.cpp (DOOMS_WITH_LLAMA=OFF)");
#endif
    });
}

}  // extern "C"

// ------------------------------------------------------------------------------------------------ speech to text (Whisper)
struct dooms_whisper {
    std::unique_ptr<dooms::WhisperPipeline> pipeline;
    std::string text;            // last successful transcription
    bool has_text = false;
    dooms::WhisperTimings timings;
};

namespace {

// Runs one transcription, keeps its text and timings in the handle, writes the text.
template <class F>
dooms_code whisper_run(dooms_whisper * w, F && call, char * out, size_t capacity, size_t * needed) {
    dooms::WhisperTimings tm;
    auto r = call(&tm);
    if (!r.is_ok()) return from_status(r.status());
    w->text = std::move(r.value().text);
    w->has_text = true;
    w->timings = tm;
    return write_text(w->text, out, capacity, needed);
}

}  // namespace

extern "C" {

DOOMS_API dooms_code dooms_whisper_open(const dooms_whisper_paths * paths, const dooms_whisper_options * options, dooms_whisper ** out) {
    return guarded([&]() -> dooms_code {
        dooms_code c;
        if (!out) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
        if (!struct_ok(paths, "paths", &c)) return c;
        if (!paths->encoder || !*paths->encoder || !paths->decoder || !*paths->decoder) return fail(DOOMS_INVALID_ARGUMENT, "paths.encoder and paths.decoder are required");
        dooms::WhisperOptions o;
        if (options) {
            if (!struct_ok(options, "options", &c)) return c;
            o.encoder_units = or_default(options->units, o.encoder_units.c_str());
            o.language = or_default(options->language, o.language.c_str());
            o.task = or_default(options->task, o.task.c_str());
            if (options->max_tokens < 0) return fail(DOOMS_INVALID_ARGUMENT, "options.max_tokens is negative");
            if (options->max_tokens > 0) o.max_tokens_per_window = options->max_tokens;
            if (options->struct_size >= offsetof(dooms_whisper_options, device) + sizeof(options->device)) o.device = or_default(options->device, "auto");
        }
        auto p = dooms::WhisperPipeline::create({paths->encoder, paths->decoder}, o);
        if (!p.is_ok()) return from_status(p.status());
        auto h = std::make_unique<dooms_whisper>();
        h->pipeline = std::make_unique<dooms::WhisperPipeline>(std::move(p).value());
        *out = h.release();
        return DOOMS_OK;
    });
}

DOOMS_API void dooms_whisper_close(dooms_whisper * whisper) { delete whisper; }

DOOMS_API int32_t dooms_whisper_sample_rate(const dooms_whisper * whisper) { return whisper && whisper->pipeline ? whisper->pipeline->sample_rate() : 0; }
DOOMS_API const char * dooms_whisper_device(const dooms_whisper * whisper) { return whisper && whisper->pipeline ? whisper->pipeline->device() : nullptr; }

DOOMS_API dooms_code dooms_whisper_transcribe(dooms_whisper * whisper, const float * samples, size_t n, char * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
        if (!whisper) return fail(DOOMS_INVALID_ARGUMENT, "whisper is NULL");
        if (!samples || n == 0) return fail(DOOMS_INVALID_ARGUMENT, "no samples");
        const std::vector<float> audio(samples, samples + n);
        return whisper_run(whisper, [&](dooms::WhisperTimings * tm) { return whisper->pipeline->transcribe(audio, tm); }, out, capacity, needed);
    });
}

DOOMS_API dooms_code dooms_whisper_transcribe_file(dooms_whisper * whisper, const char * path, char * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
        if (!whisper) return fail(DOOMS_INVALID_ARGUMENT, "whisper is NULL");
        if (!path || !*path) return fail(DOOMS_INVALID_ARGUMENT, "path is NULL or empty");
        return whisper_run(whisper, [&](dooms::WhisperTimings * tm) { return whisper->pipeline->transcribe_file(path, tm); }, out, capacity, needed);
    });
}

DOOMS_API dooms_code dooms_whisper_last_text(const dooms_whisper * whisper, char * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
        if (!whisper) return fail(DOOMS_INVALID_ARGUMENT, "whisper is NULL");
        if (!whisper->has_text) return fail(DOOMS_NOT_FOUND, "no transcription has finished on this handle");
        return write_text(whisper->text, out, capacity, needed);
    });
}

DOOMS_API dooms_code dooms_whisper_last_timings(const dooms_whisper * whisper, dooms_whisper_timings * out) {
    return guarded([&]() -> dooms_code {
        dooms_code c;
        if (!whisper) return fail(DOOMS_INVALID_ARGUMENT, "whisper is NULL");
        if (!struct_ok(out, "out", &c)) return c;
        dooms_whisper_timings t{};
        const dooms::WhisperTimings & s = whisper->timings;
        t.features_ms = s.features_ms;
        t.encoder_ms = s.encoder_ms;
        t.decode_ms = s.decode_ms;
        t.total_ms = s.total_ms;
        t.audio_s = s.audio_s;
        t.windows = static_cast<int32_t>(s.windows);
        t.tokens = static_cast<int32_t>(s.tokens);
        const size_t n = std::min(out->struct_size, sizeof(t));
        t.struct_size = out->struct_size;
        std::memcpy(out, &t, n);
        return DOOMS_OK;
    });
}

}  // extern "C"

// ------------------------------------------------------------------------------------------------ text to speech (Kokoro)
struct dooms_tts {
    std::unique_ptr<dooms::KokoroPipeline> pipeline;
    std::vector<std::string> voices;
    std::vector<float> audio;    // last successful synthesis
    bool has_audio = false;
    dooms::KokoroTimings timings;
};

namespace {

dooms_code write_samples(const std::vector<float> & a, float * out, size_t capacity, size_t * needed) {
    if (needed) *needed = a.size();
    if (!out && capacity > 0) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
    if (capacity < a.size()) return fail(DOOMS_RESOURCE_EXHAUSTED, "the output buffer holds " + std::to_string(capacity) + " samples, " + std::to_string(a.size()) + " are needed");
    if (!a.empty()) std::memcpy(out, a.data(), a.size() * sizeof(float));
    return DOOMS_OK;
}

}  // namespace

extern "C" {

DOOMS_API dooms_code dooms_tts_open(const dooms_tts_paths * paths, const dooms_tts_options * options, dooms_tts ** out) {
    return guarded([&]() -> dooms_code {
        dooms_code c;
        if (!out) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
        if (!struct_ok(paths, "paths", &c)) return c;
        if (!paths->prosody || !*paths->prosody || !paths->f0n || !*paths->f0n || !paths->model || !*paths->model) {
            return fail(DOOMS_INVALID_ARGUMENT, "paths.prosody, paths.f0n and paths.model are required");
        }
        dooms::KokoroOptions o;
        if (options) {
            if (!struct_ok(options, "options", &c)) return c;
            o.units = or_default(options->units, o.units.c_str());
            if (options->struct_size >= offsetof(dooms_tts_options, device) + sizeof(options->device)) o.device = or_default(options->device, "auto");
        }
        auto p = dooms::KokoroPipeline::create({paths->prosody, paths->f0n, paths->model}, o);
        if (!p.is_ok()) return from_status(p.status());
        auto h = std::make_unique<dooms_tts>();
        h->pipeline = std::make_unique<dooms::KokoroPipeline>(std::move(p).value());
        h->voices = h->pipeline->voices();
        *out = h.release();
        return DOOMS_OK;
    });
}

DOOMS_API void dooms_tts_close(dooms_tts * tts) { delete tts; }
DOOMS_API int32_t dooms_tts_sample_rate(const dooms_tts * tts) { return tts && tts->pipeline ? tts->pipeline->sample_rate() : 0; }
DOOMS_API const char * dooms_tts_device(const dooms_tts * tts) { return tts && tts->pipeline ? tts->pipeline->device() : nullptr; }
DOOMS_API int32_t dooms_tts_max_phonemes(const dooms_tts * tts) { return tts && tts->pipeline ? tts->pipeline->max_phonemes() : 0; }
DOOMS_API int32_t dooms_tts_voice_count(const dooms_tts * tts) { return tts ? static_cast<int32_t>(tts->voices.size()) : 0; }
DOOMS_API const char * dooms_tts_voice_name(const dooms_tts * tts, int32_t index) {
    if (!tts || index < 0 || index >= static_cast<int32_t>(tts->voices.size())) return nullptr;
    return tts->voices[static_cast<size_t>(index)].c_str();
}

DOOMS_API dooms_code dooms_tts_synthesize(dooms_tts * tts, const char * phonemes, const char * voice, float speed, uint64_t seed, float * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
        if (!tts) return fail(DOOMS_INVALID_ARGUMENT, "tts is NULL");
        if (!phonemes || !*phonemes) return fail(DOOMS_INVALID_ARGUMENT, "phonemes is NULL or empty");
        if (!voice || !*voice) return fail(DOOMS_INVALID_ARGUMENT, "voice is NULL or empty");
        dooms::KokoroTimings tm;
        auto a = tts->pipeline->synthesize(phonemes, voice, speed, seed, &tm);
        if (!a.is_ok()) return from_status(a.status());
        tts->audio = std::move(a).value();
        tts->has_audio = true;
        tts->timings = tm;
        return write_samples(tts->audio, out, capacity, needed);
    });
}

DOOMS_API dooms_code dooms_tts_last_audio(const dooms_tts * tts, float * out, size_t capacity, size_t * needed) {
    return guarded([&]() -> dooms_code {
        if (!tts) return fail(DOOMS_INVALID_ARGUMENT, "tts is NULL");
        if (!tts->has_audio) return fail(DOOMS_NOT_FOUND, "no synthesis has finished on this handle");
        return write_samples(tts->audio, out, capacity, needed);
    });
}

DOOMS_API dooms_code dooms_tts_last_timings(const dooms_tts * tts, dooms_tts_timings * out) {
    return guarded([&]() -> dooms_code {
        dooms_code c;
        if (!tts) return fail(DOOMS_INVALID_ARGUMENT, "tts is NULL");
        if (!struct_ok(out, "out", &c)) return c;
        dooms_tts_timings t{};
        const dooms::KokoroTimings & s = tts->timings;
        t.prosody_ms = s.prosody_ms; t.f0n_ms = s.f0n_ms; t.source_ms = s.source_ms; t.vocoder_ms = s.vocoder_ms; t.istft_ms = s.istft_ms; t.total_ms = s.total_ms;
        t.audio_s = s.audio_s;
        t.tokens = static_cast<int32_t>(s.tokens);
        t.frames = static_cast<int32_t>(s.frames);
        const size_t n = std::min(out->struct_size, sizeof(t));
        t.struct_size = out->struct_size;
        std::memcpy(out, &t, n);
        return DOOMS_OK;
    });
}

}  // extern "C"
