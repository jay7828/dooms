// Tests the Llm pipeline: argument validation (always), and with a real model (DOOMS_LLM_FILE, a chat-tuned GGUF; DOOMS_LLM_REFERENCE a text file with the first
// reference token ids of a greedy answer from transformers, one per line) streaming, stop strings, seeded sampling, the chat template and context limits.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "pipeline/llm_pipeline.h"

using namespace dooms;

namespace {

std::string env(const char * n) { const char * v = std::getenv(n); return v ? v : ""; }

}  // namespace

TEST(load_rejects_a_missing_file) {
    auto r = Llm::load("/nonexistent/model.gguf");
    EXPECT(!r.is_ok());
    EXPECT(r.status().code() == Code::NotFound);
}

TEST(generate_validates_arguments_and_streams_whole_characters) {
    const std::string file = env("DOOMS_LLM_FILE");
    if (file.empty()) { std::printf("        (skipped: set DOOMS_LLM_FILE)\n"); return; }
    auto llm = Llm::load(file);
    EXPECT(llm.is_ok());
    if (!llm.is_ok()) return;
    Llm & m = llm.value();
    EXPECT(!m.generate("").is_ok());
    LlmGenerateOptions o;
    o.max_new_tokens = 0;
    EXPECT(!m.generate("hi", o).is_ok());
    o.max_new_tokens = 100000;
    EXPECT(!m.generate("hi", o).is_ok());
    o.max_new_tokens = 8;
    o.stop = {""};
    EXPECT(!m.generate("hi", o).is_ok());
    o.stop.clear();
    o.sampler.temperature = -1.f;
    EXPECT(!m.generate("hi", o).is_ok());
    o.sampler = SamplerOptions();
    std::string long_prompt;
    for (int i = 0; i < 20000; i++) long_prompt += "word ";  // far more tokens than the 4096-position context
    EXPECT(m.generate(long_prompt, o).status().code() == Code::ResourceExhausted);

    // greedy, streamed: the pieces concatenate to the text, and every piece is whole UTF-8
    o.max_new_tokens = 24;
    std::string streamed;
    int pieces = 0;
    auto g = m.generate("The capital of France is", o, [&](const std::string & s) {
        streamed += s;
        pieces++;
        size_t i = 0;
        while (i < s.size()) {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            const size_t n = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
            EXPECT(n != 0 && i + n <= s.size());
            if (n == 0 || i + n > s.size()) break;
            i += n;
        }
        return true;
    });
    EXPECT(g.is_ok());
    if (!g.is_ok()) return;
    EXPECT_EQ(streamed, g.value().text);
    EXPECT(pieces > 0 && !g.value().tokens.empty());
    // greedy is deterministic
    auto g2 = m.generate("The capital of France is", o);
    EXPECT(g2.is_ok() && g2.value().tokens == g.value().tokens);
}

TEST(stop_strings_cancellation_and_seeded_sampling) {
    const std::string file = env("DOOMS_LLM_FILE");
    if (file.empty()) { std::printf("        (skipped: set DOOMS_LLM_FILE)\n"); return; }
    auto llm = Llm::load(file);
    EXPECT(llm.is_ok());
    if (!llm.is_ok()) return;
    Llm & m = llm.value();
    LlmGenerateOptions o;
    o.max_new_tokens = 40;
    auto full = m.generate("Count from one to ten: one, two, three,", o);
    EXPECT(full.is_ok());
    if (!full.is_ok()) return;
    const size_t at = full.value().text.find("six");
    EXPECT(at != std::string::npos);
    if (at != std::string::npos) {
        o.stop = {"six"};
        auto cut = m.generate("Count from one to ten: one, two, three,", o);
        EXPECT(cut.is_ok());
        if (cut.is_ok()) {
            EXPECT(cut.value().stopped);
            EXPECT_EQ(cut.value().text, full.value().text.substr(0, at));
        }
    }
    // a stop string that never matches holds text back while it could still start the match; reaching max_new_tokens must release all of it
    {
        LlmGenerateOptions h = o;
        h.stop = {"zzzzzzzzzzzzzzzzzzzz"};
        h.max_new_tokens = 6;
        auto held = m.generate("Count from one to ten: one, two, three,", h);
        LlmGenerateOptions p = o;
        p.stop.clear();
        p.max_new_tokens = 6;
        auto plain = m.generate("Count from one to ten: one, two, three,", p);
        EXPECT(held.is_ok() && plain.is_ok());
        if (held.is_ok() && plain.is_ok()) EXPECT_EQ(held.value().text, plain.value().text);
    }
    // cancellation after two pieces
    o.stop.clear();
    int n = 0;
    auto c = m.generate("Count from one to ten: one, two, three,", o, [&](const std::string &) { return ++n < 2; });
    EXPECT(c.is_ok() && c.value().cancelled && !c.value().stopped && n == 2);
    // sampling: same seed same text, a different seed (almost surely) another
    o.sampler.temperature = 1.2f;
    o.sampler.seed = 7;
    auto a = m.generate("Once upon a time", o);
    auto b = m.generate("Once upon a time", o);
    o.sampler.seed = 8;
    auto d = m.generate("Once upon a time", o);
    EXPECT(a.is_ok() && b.is_ok() && d.is_ok());
    if (a.is_ok() && b.is_ok() && d.is_ok()) {
        EXPECT(a.value().tokens == b.value().tokens);
        EXPECT(a.value().tokens != d.value().tokens);
    }
}

TEST(chat_matches_the_reference_answer) {
    const std::string file = env("DOOMS_LLM_FILE"), ref = env("DOOMS_LLM_REFERENCE");
    if (file.empty() || ref.empty()) { std::printf("        (skipped: set DOOMS_LLM_FILE and DOOMS_LLM_REFERENCE)\n"); return; }
    std::ifstream in(ref);
    std::string system, user;
    std::getline(in, system);
    std::getline(in, user);
    std::vector<int32_t> want;
    for (int32_t t; in >> t;) want.push_back(t);
    EXPECT(!user.empty() && !want.empty());
    auto llm = Llm::load(file);
    EXPECT(llm.is_ok());
    if (!llm.is_ok()) return;
    Llm & m = llm.value();
    EXPECT(m.has_chat_template());
    LlmGenerateOptions o;
    o.max_new_tokens = static_cast<int32_t>(want.size());
    LlmTimings tm;
    auto g = m.chat({{"system", system}, {"user", user}}, o, nullptr, &tm);
    EXPECT(g.is_ok());
    if (!g.is_ok()) return;
    size_t agree = 0;
    while (agree < want.size() && agree < g.value().tokens.size() && g.value().tokens[agree] == want[agree]) agree++;
    std::printf("        %zu of %zu tokens equal transformers; prefill %.1f ms (%lld tokens), first token %.1f ms, %.2f ms/token\n", agree, want.size(),
                tm.prefill_ms, static_cast<long long>(tm.prompt_tokens), tm.first_token_ms, tm.decode_ms_per_token());
    EXPECT(agree >= 8);
    EXPECT(tm.generated_tokens == static_cast<int64_t>(g.value().tokens.size()));
    // chat_tokens is exactly the prompt chat() fed the model
    auto ct = m.chat_tokens({{"system", system}, {"user", user}});
    EXPECT(ct.is_ok() && static_cast<int64_t>(ct.value().size()) == tm.prompt_tokens);
    EXPECT(!m.chat_tokens({}).is_ok());
    EXPECT(!m.chat({}, o).is_ok());
}

// The own Metal decoder (the default for f16 llama/qwen2 files) against llama.cpp on the same file: identical greedy answers, sampling through llama.cpp's
// sampler on the Metal logits (seeded: reproducible), and the engine choice reported.
TEST(metal_and_llama_cpp_engines_agree) {
    const std::string file = env("DOOMS_LLM_FILE");
    if (file.empty()) { std::printf("        (skipped: set DOOMS_LLM_FILE)\n"); return; }
    LlmOptions lo;
    lo.text_engine = "llama.cpp";
    auto metal = Llm::load(file);
    auto ref = Llm::load(file, lo);
    EXPECT(metal.is_ok() && ref.is_ok());
    if (!metal.is_ok() || !ref.is_ok()) return;
    EXPECT_EQ(std::string(ref.value().text_engine_used()), std::string("llama.cpp"));
    std::printf("        default engine for this file: %s\n", metal.value().text_engine_used());
    LlmGenerateOptions o;
    o.max_new_tokens = 48;
    for (const char * q : {"Name three colours.", "What is 12 times 7?"}) {
        auto a = metal.value().chat({{"user", q}}, o);
        auto b = ref.value().chat({{"user", q}}, o);
        EXPECT(a.is_ok() && b.is_ok());
        if (!a.is_ok() || !b.is_ok()) continue;
        std::printf("        \"%s\" -> \"%s\" (%s)\n", q, a.value().text.c_str(), a.value().text == b.value().text ? "same as llama.cpp" : "DIFFERENT from llama.cpp");
        EXPECT_EQ(a.value().text, b.value().text);
    }
    LlmGenerateOptions so;
    so.max_new_tokens = 24;
    so.sampler.temperature = 0.8f;
    so.sampler.seed = 11;
    auto s1 = metal.value().chat({{"user", "Tell me a fact."}}, so);
    auto s2 = metal.value().chat({{"user", "Tell me a fact."}}, so);
    EXPECT(s1.is_ok() && s2.is_ok());
    if (s1.is_ok() && s2.is_ok()) EXPECT(s1.value().tokens == s2.value().tokens);
    LlmOptions bad;
    bad.text_engine = "gpu";
    EXPECT(Llm::load(file, bad).status().code() == Code::InvalidArgument);
}

int main() { return dooms_test::run_all(); }
