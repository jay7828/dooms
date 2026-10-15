// dooms.hpp: C++17 header-only RAII wrapper over the stable C API (dooms_c.h). Link against libdooms_c (CMake: find_package(dooms) and dooms::dooms).
//
//     #include <dooms/dooms.hpp>
//     dooms::Llm llm("qwen2.5-0.5b-instruct-q4_k_m.gguf");
//     std::string answer = llm.chat({{"user", "Name three colours."}}, 64);
//     llm.generate("Once upon a time", 100, {0.8f, 40, 0.95f, 1.0f, /*seed*/ 1}, {"\n\n"}, [](const std::string & piece) { std::cout << piece << std::flush; return true; });
//
// Errors are thrown as dooms::Error. Handles are move-only and not thread-safe. Apple Silicon macOS only.
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dooms_c.h"

namespace dooms {

class Error : public std::runtime_error {
public:
    Error(dooms_code code, const std::string & what) : std::runtime_error(std::string(dooms_code_name(code)) + ": " + what), code_(code) {}
    dooms_code code() const { return code_; }

private:
    dooms_code code_;
};

namespace detail {

inline void check(dooms_code c) {
    if (c != DOOMS_OK) throw Error(c, dooms_last_error());
}

using TextFn = std::function<bool(const std::string &)>;

inline int32_t trampoline(const char * piece, void * user) {
    TextFn & fn = *static_cast<TextFn *>(user);
    return fn(piece) ? 0 : 1;
}

}  // namespace detail

struct Sampling {
    float temperature = 0.f;     // 0: greedy
    int32_t top_k = 40;          // 0: off
    float top_p = 0.95f;         // 1: off
    float repeat_penalty = 1.f;  // 1: off
    uint32_t seed = 0;
};

struct Message {
    std::string role;     // "system", "user" or "assistant"
    std::string content;
};

// A text language model (any GGUF llama.cpp loads) on the Apple GPU.
class Llm {
public:
    explicit Llm(const std::string & gguf_path, int32_t n_ctx = 0, int32_t n_threads = 0, int32_t n_gpu_layers = -1) {
        dooms_llm_options o{sizeof(dooms_llm_options), n_ctx, n_threads, n_gpu_layers};
        detail::check(dooms_llm_open(gguf_path.c_str(), &o, &h_));
    }
    Llm(Llm && o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
    Llm & operator=(Llm && o) noexcept {
        if (this != &o) { dooms_llm_close(h_); h_ = std::exchange(o.h_, nullptr); }
        return *this;
    }
    Llm(const Llm &) = delete;
    Llm & operator=(const Llm &) = delete;
    ~Llm() { dooms_llm_close(h_); }

    int32_t context_size() const { return dooms_llm_context_size(h_); }

    // Continues `prompt`. `on_text` (optional) gets each new piece, always whole UTF-8; return false to stop early (the text so far is returned).
    std::string generate(const std::string & prompt, int32_t max_new_tokens = 128, const Sampling & s = Sampling(), const std::vector<std::string> & stop = {},
                         const detail::TextFn & on_text = nullptr) {
        auto samp = sampling(s);
        auto sp = stop_pointers(stop);
        if (on_text) return run_stream([&](dooms_text_callback cb, void * user) {
            return dooms_llm_generate_stream(h_, prompt.c_str(), max_new_tokens, &samp, sp.data(), static_cast<int32_t>(sp.size()), cb, user);
        }, on_text);
        return run_buffer([&](char * out, size_t cap, size_t * need) {
            return dooms_llm_generate(h_, prompt.c_str(), max_new_tokens, &samp, sp.data(), static_cast<int32_t>(sp.size()), out, cap, need);
        }, max_new_tokens);
    }

    // Answers a conversation formatted with the model's own chat template.
    std::string chat(const std::vector<Message> & messages, int32_t max_new_tokens = 128, const Sampling & s = Sampling(), const std::vector<std::string> & stop = {},
                     const detail::TextFn & on_text = nullptr) {
        auto samp = sampling(s);
        auto sp = stop_pointers(stop);
        std::vector<dooms_chat_message> msgs;
        for (const Message & m : messages) msgs.push_back({m.role.c_str(), m.content.c_str()});
        if (on_text) return run_stream([&](dooms_text_callback cb, void * user) {
            return dooms_llm_chat_stream(h_, msgs.data(), static_cast<int32_t>(msgs.size()), max_new_tokens, &samp, sp.data(), static_cast<int32_t>(sp.size()), cb, user);
        }, on_text);
        return run_buffer([&](char * out, size_t cap, size_t * need) {
            return dooms_llm_chat(h_, msgs.data(), static_cast<int32_t>(msgs.size()), max_new_tokens, &samp, sp.data(), static_cast<int32_t>(sp.size()), out, cap, need);
        }, max_new_tokens);
    }

private:
    static dooms_llm_sampling sampling(const Sampling & s) {
        return dooms_llm_sampling{sizeof(dooms_llm_sampling), s.temperature, s.top_k, s.top_p, s.repeat_penalty, s.seed};
    }
    static std::vector<const char *> stop_pointers(const std::vector<std::string> & stop) {
        std::vector<const char *> p;
        for (const std::string & s : stop) p.push_back(s.c_str());
        return p;
    }
    template <class F> static std::string run_buffer(F call, int32_t max_new_tokens) {
        size_t cap = 8 * static_cast<size_t>(max_new_tokens > 0 ? max_new_tokens : 1) + 64;
        for (;;) {
            std::string buf(cap, '\0');
            size_t need = 0;
            const dooms_code c = call(&buf[0], cap, &need);
            if (c == DOOMS_RESOURCE_EXHAUSTED && need > cap) { cap = need; continue; }
            detail::check(c);
            buf.resize(std::char_traits<char>::length(buf.c_str()));
            return buf;
        }
    }
    template <class F> static std::string run_stream(F call, const detail::TextFn & on_text) {
        std::string all;
        detail::TextFn fn = [&](const std::string & piece) { all += piece; return on_text(piece); };
        detail::check(call(&detail::trampoline, &fn));
        return all;
    }

    dooms_llm * h_ = nullptr;
};

// An image question-answering model (SmolVLM): vision tower on the Neural Engine, language model on the GPU.
class Vlm {
public:
    Vlm(const std::string & vision_mlpackage, const std::string & text_gguf, const std::string & config_gguf, const std::string & vision_units = "") {
        dooms_vlm_paths p{sizeof(dooms_vlm_paths), vision_mlpackage.c_str(), text_gguf.c_str(), config_gguf.c_str()};
        dooms_vlm_options o{sizeof(dooms_vlm_options), vision_units.empty() ? nullptr : vision_units.c_str(), 0};
        detail::check(dooms_vlm_open(&p, &o, &h_));
    }
    Vlm(Vlm && o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
    Vlm & operator=(Vlm && o) noexcept {
        if (this != &o) { dooms_vlm_close(h_); h_ = std::exchange(o.h_, nullptr); }
        return *this;
    }
    Vlm(const Vlm &) = delete;
    Vlm & operator=(const Vlm &) = delete;
    ~Vlm() { dooms_vlm_close(h_); }

    // `rgb`: height x width x 3 bytes, RGB, row-major.
    std::string generate(const uint8_t * rgb, int32_t width, int32_t height, const std::string & prompt, int32_t max_new_tokens = 128,
                         const detail::TextFn & on_text = nullptr) {
        dooms_image img{width, height, rgb};
        if (on_text) {
            std::string all;
            detail::TextFn fn = [&](const std::string & piece) { all += piece; return on_text(piece); };
            detail::check(dooms_vlm_generate_stream(h_, &img, prompt.c_str(), max_new_tokens, &detail::trampoline, &fn));
            return all;
        }
        size_t cap = 8 * static_cast<size_t>(max_new_tokens > 0 ? max_new_tokens : 1) + 64;
        for (;;) {
            std::string buf(cap, '\0');
            size_t need = 0;
            const dooms_code c = dooms_vlm_generate(h_, &img, prompt.c_str(), max_new_tokens, &buf[0], cap, &need);
            if (c == DOOMS_RESOURCE_EXHAUSTED && need > cap) { cap = need; continue; }
            detail::check(c);
            buf.resize(std::char_traits<char>::length(buf.c_str()));
            return buf;
        }
    }

private:
    dooms_vlm * h_ = nullptr;
};

}  // namespace dooms
