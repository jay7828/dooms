// Implementation of include/dooms/dooms_c.h: a thin, exception-free layer over the C++ pipelines. Every entry point validates its arguments, converts the
// Status of the C++ call to a dooms_code and stores the message for dooms_last_error().
#include "dooms/dooms_c.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/rgb_image.h"
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
};
struct dooms_vla {
    std::unique_ptr<dooms::VlaPipeline> pipeline;
    std::unique_ptr<dooms::Tokenizer> tokenizer;
};
#else
struct dooms_vlm {};
struct dooms_llm {};
struct dooms_vla {};
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

dooms_code llm_write(const std::string & text, char * out, size_t capacity, size_t * needed) {
    if (needed) *needed = text.size() + 1;
    if (!out && capacity > 0) return fail(DOOMS_INVALID_ARGUMENT, "out is NULL");
    if (capacity < text.size() + 1) return fail(DOOMS_RESOURCE_EXHAUSTED, "the output buffer holds " + std::to_string(capacity) + " bytes, " + std::to_string(text.size() + 1) + " are needed");
    std::memcpy(out, text.c_str(), text.size() + 1);
    return DOOMS_OK;
}

}  // namespace
#endif

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
        if (!prompt || !*prompt) return fail(DOOMS_INVALID_ARGUMENT, "prompt is NULL or empty");
        dooms::LlmGenerateOptions o;
        dooms_code c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        auto r = llm->llm->generate(prompt, o);
        if (!r.is_ok()) return from_status(r.status());
        return llm_write(r.value().text, out, capacity, needed);
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
        if (!callback) return fail(DOOMS_INVALID_ARGUMENT, "callback is NULL");
        if (!prompt || !*prompt) return fail(DOOMS_INVALID_ARGUMENT, "prompt is NULL or empty");
        dooms::LlmGenerateOptions o;
        dooms_code c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        auto r = llm->llm->generate(prompt, o, [&](const std::string & piece) { return callback(piece.c_str(), user) == 0; });
        return r.is_ok() ? DOOMS_OK : from_status(r.status());
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
        std::vector<dooms::ChatMessage> msgs;
        dooms_code c = llm_messages(messages, n_messages, &msgs);
        if (c != DOOMS_OK) return c;
        dooms::LlmGenerateOptions o;
        c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        auto r = llm->llm->chat(msgs, o);
        if (!r.is_ok()) return from_status(r.status());
        return llm_write(r.value().text, out, capacity, needed);
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
        if (!callback) return fail(DOOMS_INVALID_ARGUMENT, "callback is NULL");
        std::vector<dooms::ChatMessage> msgs;
        dooms_code c = llm_messages(messages, n_messages, &msgs);
        if (c != DOOMS_OK) return c;
        dooms::LlmGenerateOptions o;
        c = llm_generate_options(sampling, stop, n_stop, max_new_tokens, &o);
        if (c != DOOMS_OK) return c;
        auto r = llm->llm->chat(msgs, o, [&](const std::string & piece) { return callback(piece.c_str(), user) == 0; });
        return r.is_ok() ? DOOMS_OK : from_status(r.status());
#else
        (void) llm; (void) messages; (void) n_messages; (void) max_new_tokens; (void) sampling; (void) stop; (void) n_stop; (void) callback; (void) user;
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
        h->pipeline = std::make_unique<dooms::VlaPipeline>(std::move(p).value());
        h->tokenizer = std::make_unique<dooms::Tokenizer>(std::move(t).value());
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
        const dooms::VlaPipeline & p = *vla->pipeline;
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
        dooms::VlaPipeline & p = *vla->pipeline;
        if (n_cameras != p.n_cameras() || !cameras) return fail(DOOMS_INVALID_ARGUMENT, "expected " + std::to_string(p.n_cameras()) + " cameras");
        if (!task || !*task) return fail(DOOMS_INVALID_ARGUMENT, "task is NULL or empty");
        if (n_state < 0 || n_state > p.state_dim() || (n_state > 0 && !state)) return fail(DOOMS_INVALID_ARGUMENT, "n_state must be in [0, state_dim] and state non-NULL");
        const size_t need = static_cast<size_t>(p.chunk_size() * p.action_dim());
        if (!actions || capacity < need) return fail(DOOMS_RESOURCE_EXHAUSTED, "actions must hold chunk_size * action_dim = " + std::to_string(need) + " floats");
        dooms::VlaObservation obs;
        for (int32_t i = 0; i < n_cameras; i++) {
            dooms::RgbImage img;
            dooms_code c = to_rgb_image(&cameras[i], "camera", &img);
            if (c != DOOMS_OK) return c;
            auto cam = dooms::camera_from_rgb(img);
            if (!cam.is_ok()) return from_status(cam.status());
            obs.cameras.push_back(std::move(cam).value());
        }
        auto tokens = vla->tokenizer->encode_task(task, p.tokenizer_max_length());
        if (!tokens.is_ok()) return from_status(tokens.status());
        obs.tokens = tokens.value().tokens;
        obs.token_mask = tokens.value().mask;
        obs.state.assign(state, state + n_state);
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

}  // extern "C"
