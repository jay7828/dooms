#include "pipeline/llm_pipeline.h"

#include <algorithm>
#include <chrono>

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
    TextDecoder decoder;
    explicit Impl(TextDecoder d) : decoder(std::move(d)) {}
};

Llm::Llm(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Llm::Llm(Llm &&) noexcept = default;
Llm & Llm::operator=(Llm &&) noexcept = default;
Llm::~Llm() = default;

Result<Llm> Llm::load(const std::string & path, const LlmOptions & options) {
    auto dec = TextDecoder::load(path, options.text);
    if (!dec.is_ok()) return dec.status();
    return Llm(std::make_unique<Impl>(std::move(dec.value())));
}

int64_t Llm::vocab_size() const { return impl_->decoder.vocab_size(); }
int32_t Llm::context_size() const { return impl_->decoder.context_size(); }
bool Llm::has_chat_template() const { return impl_->decoder.apply_chat_template({{"user", "x"}}, true).is_ok(); }

Result<std::vector<int32_t>> Llm::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    return impl_->decoder.tokenize(text, parse_special, add_special);
}

namespace {

Result<LlmGeneration> run(TextDecoder & dec, const std::vector<int32_t> & prompt, const LlmGenerateOptions & o, const std::function<bool(const std::string &)> & on_text,
                          LlmTimings * timings) {
    if (o.max_new_tokens < 1 || o.max_new_tokens >= dec.context_size()) return invalid_argument("max_new_tokens must be in [1, context size)");
    for (const std::string & s : o.stop) if (s.empty()) return invalid_argument("a stop string is empty");
    if (static_cast<int64_t>(prompt.size()) + o.max_new_tokens > dec.context_size()) {
        return resource_exhausted("prompt of " + std::to_string(prompt.size()) + " tokens + " + std::to_string(o.max_new_tokens) + " new tokens exceed the context of " +
                                  std::to_string(dec.context_size()));
    }
    const auto t0 = Clock::now();
    LlmTimings tm;
    tm.prompt_tokens = static_cast<int64_t>(prompt.size());
    dec.reset();
    if (Status st = dec.start_sampling(o.sampler); !st.is_ok()) return st;
    if (Status st = dec.eval_tokens(prompt); !st.is_ok()) return st;
    tm.prefill_ms = ms_since(t0);

    LlmGeneration out;
    std::string pending;     // bytes not yet handed to on_text (an incomplete UTF-8 character, or a possible start of a stop string)
    size_t max_stop = 0;
    for (const std::string & s : o.stop) max_stop = std::max(max_stop, s.size());
    auto flush = [&](size_t upto) -> bool {
        if (upto == 0) return true;
        std::string piece = pending.substr(0, upto);
        pending.erase(0, upto);
        out.text += piece;
        return !on_text || on_text(piece);
    };
    const auto t_decode = Clock::now();
    for (int32_t i = 0; i < o.max_new_tokens; i++) {
        auto next = dec.sample_next();
        if (!next.is_ok()) return next.status();
        const int32_t tok = next.value();
        out.tokens.push_back(tok);
        tm.generated_tokens++;
        if (i == 0) tm.first_token_ms = ms_since(t0);
        if (dec.is_end_of_generation(tok)) { out.stopped = true; break; }
        pending += dec.piece(tok);
        // stop strings: look for the earliest match in what is pending
        size_t hit = std::string::npos;
        for (const std::string & s : o.stop) hit = std::min(hit, pending.find(s));
        if (hit != std::string::npos) {
            pending.resize(hit);
            if (!flush(pending.size())) out.cancelled = true;
            out.stopped = !out.cancelled;
            break;
        }
        // emit what cannot be the start of a stop string and is complete UTF-8
        size_t safe = pending.size() > max_stop ? pending.size() - max_stop : 0;
        safe = std::min(safe, complete_utf8_prefix(pending));
        if (max_stop == 0) safe = complete_utf8_prefix(pending);
        if (!flush(safe)) { out.cancelled = true; break; }
        if (i + 1 < o.max_new_tokens) {
            if (Status st = dec.eval_tokens({tok}); !st.is_ok()) return st;
        }
    }
    if (!out.stopped && !out.cancelled) flush(pending.size());
    else if (out.stopped && !pending.empty() && out.tokens.size() && dec.is_end_of_generation(out.tokens.back())) flush(pending.size());
    tm.decode_ms = tm.generated_tokens > 1 ? ms_since(t_decode) - (tm.first_token_ms - tm.prefill_ms) : 0.0;
    if (timings) *timings = tm;
    return out;
}

}  // namespace

Result<LlmGeneration> Llm::generate(const std::string & prompt, const LlmGenerateOptions & options, const std::function<bool(const std::string &)> & on_text,
                                    LlmTimings * timings) {
    if (prompt.empty()) return invalid_argument("the prompt is empty");
    auto toks = impl_->decoder.tokenize(prompt, /*parse_special=*/false, /*add_bos=*/true);
    if (!toks.is_ok()) return toks.status();
    return run(impl_->decoder, toks.value(), options, on_text, timings);
}

Result<LlmGeneration> Llm::chat(const std::vector<ChatMessage> & messages, const LlmGenerateOptions & options, const std::function<bool(const std::string &)> & on_text,
                                LlmTimings * timings) {
    auto text = impl_->decoder.apply_chat_template(messages, true);
    if (!text.is_ok()) return text.status();
    // the template already carries every special token it needs, including BOS where the model wants one
    auto toks = impl_->decoder.tokenize(text.value(), /*parse_special=*/true, /*add_bos=*/false);
    if (!toks.is_ok()) return toks.status();
    return run(impl_->decoder, toks.value(), options, on_text, timings);
}

}  // namespace dooms
