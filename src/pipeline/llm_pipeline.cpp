#include "pipeline/llm_pipeline.h"

#include <algorithm>
#include <chrono>

#include "pipeline/native_llama.h"

namespace dooms {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Length of the longest prefix of `s` that ends on a complete UTF-8 character (a trailing partial character is not emitted yet).
size_t complete_utf8_prefix(const std::string & s) {
    size_t i = s.size();
    for (size_t back = 1; back <= 3 && back <= s.size(); back++) {
        const unsigned char c = static_cast<unsigned char>(s[s.size() - back]);
        if ((c & 0xC0) == 0x80) continue;                 // continuation byte: look further back for the lead byte
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (need > back) i = s.size() - back;             // the character is not complete
        break;
    }
    return i;
}

}  // namespace

struct Llm::Impl {
    TextDecoder decoder;                          // the whole model (llama.cpp engine) or only the vocabulary and the sampler (Metal engine)
    std::unique_ptr<MetalLlamaDecoder> native;    // the own Metal decoder, when it runs this file
    int32_t queue_depth = 1;
    explicit Impl(TextDecoder d) : decoder(std::move(d)) {}
};

Llm::Llm(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Llm::Llm(Llm &&) noexcept = default;
Llm & Llm::operator=(Llm &&) noexcept = default;
Llm::~Llm() = default;

Result<Llm> Llm::load(const std::string & path, const LlmOptions & options) {
    if (options.text_engine != "auto" && options.text_engine != "metal" && options.text_engine != "llama.cpp") {
        return invalid_argument("text_engine must be \"auto\", \"metal\" or \"llama.cpp\", got \"" + options.text_engine + "\"");
    }
    if (options.metal_queue_depth < 1) return invalid_argument("metal_queue_depth must be >= 1");
    std::unique_ptr<MetalLlamaDecoder> native;
    TextDecoderOptions to = options.text;
    if (options.text_engine != "llama.cpp") {
        NativeLlamaOptions no;
        no.n_ctx = options.text.n_ctx;
        no.n_batch = options.text.n_batch;
        auto n = load_native_llama(path, no);
        // "auto" falls back to llama.cpp only for files the own decoder does not run; a missing or broken file is an error either way.
        if (!n.is_ok() && (options.text_engine == "metal" || n.status().code() != Code::Unsupported)) {
            return Status(n.status().code(), "text_engine \"" + options.text_engine + "\": " + n.status().message());
        }
        if (n.is_ok()) {
            native = std::move(n).value();
            to.vocab_only = true;   // llama.cpp only tokenizes and samples
        }
    }
    auto dec = TextDecoder::load(path, to);
    if (!dec.is_ok()) return dec.status();
    auto impl = std::make_unique<Impl>(std::move(dec.value()));
    impl->native = std::move(native);
    impl->queue_depth = options.metal_queue_depth;
    return Llm(std::move(impl));
}

const char * Llm::text_engine_used() const { return impl_->native ? "metal" : "llama.cpp"; }

int64_t Llm::vocab_size() const { return impl_->decoder.vocab_size(); }
int32_t Llm::context_size() const { return impl_->native ? impl_->native->config().n_ctx : impl_->decoder.context_size(); }
bool Llm::has_chat_template() const { return impl_->decoder.apply_chat_template({{"user", "x"}}, true).is_ok(); }

Result<std::vector<int32_t>> Llm::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    return impl_->decoder.tokenize(text, parse_special, add_special);
}

namespace {

// Turns generated tokens into text: holds back incomplete UTF-8 and possible starts of stop strings, stops on end-of-generation, a stop string or a cancel.
class TokenSink {
public:
    TokenSink(TextDecoder & vocab, const LlmGenerateOptions & o, const std::function<bool(const std::string &)> & on_text, LlmGeneration & out)
        : vocab_(vocab), o_(o), on_text_(on_text), out_(out) {
        for (const std::string & s : o.stop) max_stop_ = std::max(max_stop_, s.size());
    }
    // false: generation is over (stopped or cancelled).
    bool push(int32_t tok) {
        out_.tokens.push_back(tok);
        if (vocab_.is_end_of_generation(tok)) { out_.stopped = true; return false; }
        pending_ += vocab_.piece(tok);
        size_t hit = std::string::npos;
        for (const std::string & s : o_.stop) hit = std::min(hit, pending_.find(s));
        if (hit != std::string::npos) {
            pending_.resize(hit);
            if (!flush(pending_.size())) out_.cancelled = true;
            out_.stopped = !out_.cancelled;
            return false;
        }
        size_t safe = pending_.size() > max_stop_ ? pending_.size() - max_stop_ : 0;
        safe = std::min(safe, complete_utf8_prefix(pending_));
        if (max_stop_ == 0) safe = complete_utf8_prefix(pending_);
        if (!flush(safe)) { out_.cancelled = true; return false; }
        return true;
    }
    void finish() {
        if (!out_.stopped && !out_.cancelled) flush(pending_.size());
        else if (out_.stopped && !pending_.empty() && !out_.tokens.empty() && vocab_.is_end_of_generation(out_.tokens.back())) flush(pending_.size());
    }

private:
    bool flush(size_t upto) {
        if (upto == 0) return true;
        std::string piece = pending_.substr(0, upto);
        pending_.erase(0, upto);
        out_.text += piece;
        return !on_text_ || on_text_(piece);
    }
    TextDecoder & vocab_;
    const LlmGenerateOptions & o_;
    const std::function<bool(const std::string &)> & on_text_;
    LlmGeneration & out_;
    std::string pending_;
    size_t max_stop_ = 0;
};

Status check_request(const std::vector<int32_t> & prompt, const LlmGenerateOptions & o, int32_t n_ctx) {
    if (o.max_new_tokens < 1 || o.max_new_tokens >= n_ctx) return invalid_argument("max_new_tokens must be in [1, context size)");
    for (const std::string & s : o.stop) if (s.empty()) return invalid_argument("a stop string is empty");
    if (static_cast<int64_t>(prompt.size()) + o.max_new_tokens > n_ctx) {
        return resource_exhausted("prompt of " + std::to_string(prompt.size()) + " tokens + " + std::to_string(o.max_new_tokens) + " new tokens exceed the context of " +
                                  std::to_string(n_ctx));
    }
    return Status();
}

Result<LlmGeneration> run(TextDecoder & dec, const std::vector<int32_t> & prompt, const LlmGenerateOptions & o, const std::function<bool(const std::string &)> & on_text,
                          LlmTimings * timings) {
    if (Status st = check_request(prompt, o, dec.context_size()); !st.is_ok()) return st;
    const auto t0 = Clock::now();
    LlmTimings tm;
    tm.prompt_tokens = static_cast<int64_t>(prompt.size());
    dec.reset();
    if (Status st = dec.start_sampling(o.sampler); !st.is_ok()) return st;
    if (Status st = dec.eval_tokens(prompt); !st.is_ok()) return st;
    tm.prefill_ms = ms_since(t0);

    LlmGeneration out;
    TokenSink sink(dec, o, on_text, out);
    const auto t_decode = Clock::now();
    for (int32_t i = 0; i < o.max_new_tokens; i++) {
        auto next = dec.sample_next();
        if (!next.is_ok()) return next.status();
        tm.generated_tokens++;
        if (i == 0) tm.first_token_ms = ms_since(t0);
        if (!sink.push(next.value())) break;
        if (i + 1 < o.max_new_tokens) {
            if (Status st = dec.eval_tokens({next.value()}); !st.is_ok()) return st;
        }
    }
    sink.finish();
    tm.decode_ms = tm.generated_tokens > 1 ? ms_since(t_decode) - (tm.first_token_ms - tm.prefill_ms) : 0.0;
    if (timings) *timings = tm;
    return out;
}

// The own Metal decoder: greedy sessions decode with steps queued on the GPU (the token is picked there); other sessions take the logits of each step and
// draw the token with llama.cpp's sampler (`vocab`), then feed it back.
Result<LlmGeneration> run_native(MetalLlamaDecoder & native, int32_t depth, TextDecoder & vocab, const std::vector<int32_t> & prompt, const LlmGenerateOptions & o,
                                 const std::function<bool(const std::string &)> & on_text, LlmTimings * timings) {
    if (Status st = check_request(prompt, o, native.config().n_ctx); !st.is_ok()) return st;
    if (Status st = vocab.start_sampling(o.sampler); !st.is_ok()) return st;
    const auto t0 = Clock::now();
    LlmTimings tm;
    tm.prompt_tokens = static_cast<int64_t>(prompt.size());
    native.reset();
    if (Status st = native.eval_tokens(prompt); !st.is_ok()) return st;
    LlmGeneration out;
    TokenSink sink(vocab, o, on_text, out);
    auto t_decode = t0;
    auto first = [&]() {
        // The GPU runs asynchronously: the first token is ready once the prefill has run, so this is where its time shows.
        tm.prefill_ms = tm.first_token_ms = ms_since(t0);
        t_decode = Clock::now();
    };
    const bool greedy = o.sampler.temperature == 0.f && o.sampler.repeat_penalty == 1.f;
    if (greedy) {
        Status st = native.generate(o.max_new_tokens, depth, [&](int32_t tok) {
            if (tm.generated_tokens++ == 0) first();
            return sink.push(tok);
        });
        if (!st.is_ok()) return st;
    } else {
        for (int32_t i = 0; i < o.max_new_tokens; i++) {
            auto lg = native.logits();
            if (!lg.is_ok()) return lg.status();
            auto tok = vocab.sample_logits(lg.value().data());
            if (!tok.is_ok()) return tok.status();
            if (tm.generated_tokens++ == 0) first();
            if (!sink.push(tok.value())) break;
            if (i + 1 < o.max_new_tokens) {
                if (Status st = native.eval_tokens({tok.value()}); !st.is_ok()) return st;
            }
        }
    }
    sink.finish();
    tm.decode_ms = tm.generated_tokens > 1 ? ms_since(t_decode) : 0.0;
    if (timings) *timings = tm;
    return out;
}

}  // namespace

Result<LlmGeneration> Llm::generate(const std::string & prompt, const LlmGenerateOptions & options, const std::function<bool(const std::string &)> & on_text,
                                    LlmTimings * timings) {
    if (prompt.empty()) return invalid_argument("the prompt is empty");
    auto toks = impl_->decoder.tokenize(prompt, /*parse_special=*/false, /*add_bos=*/true);
    if (!toks.is_ok()) return toks.status();
    if (impl_->native) return run_native(*impl_->native, impl_->queue_depth, impl_->decoder, toks.value(), options, on_text, timings);
    return run(impl_->decoder, toks.value(), options, on_text, timings);
}

Result<std::vector<int32_t>> Llm::chat_tokens(const std::vector<ChatMessage> & messages) const {
    auto text = impl_->decoder.apply_chat_template(messages, true);
    if (!text.is_ok()) return text.status();
    // the template already carries every special token it needs, including BOS where the model wants one
    return impl_->decoder.tokenize(text.value(), /*parse_special=*/true, /*add_bos=*/false);
}

Result<LlmGeneration> Llm::chat(const std::vector<ChatMessage> & messages, const LlmGenerateOptions & options, const std::function<bool(const std::string &)> & on_text,
                                LlmTimings * timings) {
    auto toks = chat_tokens(messages);
    if (!toks.is_ok()) return toks.status();
    if (impl_->native) return run_native(*impl_->native, impl_->queue_depth, impl_->decoder, toks.value(), options, on_text, timings);
    return run(impl_->decoder, toks.value(), options, on_text, timings);
}

}  // namespace dooms
