// Tests the Whisper pipeline against transformers (tools/parity/whisper_reference.py, DOOMS_WHISPER_DIR with encoder-fp16.mlpackage, decoder.gguf and
// reference.gguf): each stage separately (log-mel features, the Core ML encoder, the Metal decoder fed transformers' own encoder states, teacher-forced
// greedy tokens) and the whole path, samples in -> text out. Without the directory only the argument checks run;
// DOOMS_WHISPER_UNITS picks the encoder's Core ML units.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend/metal/metal_whisper_decoder.h"
#include "check.h"
#include "format/gguf.h"
#include "pipeline/audio_io.h"
#include "pipeline/whisper_features.h"
#include "pipeline/whisper_pipeline.h"

using namespace dooms;

namespace {

std::string dir() {
    const char * d = std::getenv("DOOMS_WHISPER_DIR");
    return d ? d : "";
}

std::vector<float> tensor(const GgufFile & f, const std::string & name) {
    auto info = f.find_tensor(name);
    if (!info.is_ok()) { std::fprintf(stderr, "  missing %s\n", name.c_str()); dooms_test::current_failures()++; return {}; }
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    return std::vector<float>(p, p + info.value()->shape.numel());
}

std::vector<int32_t> ints(const std::vector<float> & v) {
    std::vector<int32_t> out;
    for (float x : v) out.push_back(static_cast<int32_t>(x));
    return out;
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0, n = 0;
    for (size_t i = 0; i < b.size() && i < a.size(); i++) { d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]); n += double(b[i]) * b[i]; }
    return std::sqrt(d / n);
}

}  // namespace

TEST(arguments_are_validated) {
    EXPECT(load_audio("/nonexistent.wav", 16000).status().code() == Code::NotFound);
    EXPECT(load_audio("/nonexistent.wav", 10).status().code() == Code::InvalidArgument);
    WhisperFeatureConfig fc;
    fc.n_fft = 400; fc.hop = 160; fc.n_mels = 2; fc.chunk_samples = 1600;
    fc.mel_filters.assign(2 * 201, 0.f);
    auto f = WhisperFeatures::create(fc);
    EXPECT(f.is_ok());
    if (f.is_ok()) {
        std::vector<float> out, tooLong(1601, 0.f);
        EXPECT(!f.value().compute(nullptr, 1, &out).is_ok());
        EXPECT(!f.value().compute(tooLong.data(), tooLong.size(), &out).is_ok());
        EXPECT(f.value().compute(tooLong.data(), 1600, &out).is_ok());
        EXPECT_EQ(out.size(), static_cast<size_t>(2 * 10));
    }
    fc.mel_filters.pop_back();
    EXPECT(!WhisperFeatures::create(fc).is_ok());
    EXPECT(!WhisperPipeline::create({"/nonexistent.mlpackage", "/nonexistent.gguf"}).is_ok());
}

TEST(every_stage_matches_transformers) {
    const std::string d = dir();
    if (d.empty()) { std::printf("        (skipped: set DOOMS_WHISPER_DIR)\n"); return; }
    auto ref = GgufFile::open(d + "/reference.gguf");
    WhisperOptions o;
    if (const char * u = std::getenv("DOOMS_WHISPER_UNITS")) o.encoder_units = u;   // default: the pipeline's
    auto pipe = WhisperPipeline::create({d + "/encoder-fp16.mlpackage", d + "/decoder.gguf"}, o);
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) { if (!pipe.is_ok()) std::printf("        %s\n", pipe.status().to_string().c_str()); return; }
    WhisperPipeline & p = pipe.value();
    const GgufFile & r = ref.value();
    const int64_t n_cases = static_cast<int64_t>(r.get_uint("whisper_reference.n_cases").value());
    for (int64_t k = 0; k < n_cases; k++) {
        const std::string c = "case" + std::to_string(k) + ".";
        const std::vector<float> audio = tensor(r, c + "audio"), want_mel = tensor(r, c + "mel"), want_enc = tensor(r, c + "encoder_out"), want_logits = tensor(r, c + "logits");
        const std::vector<int32_t> prompt = ints(tensor(r, c + "prompt")), generated = ints(tensor(r, c + "generated"));
        EXPECT(prompt == p.prompt());
        // 1. features
        std::vector<float> mel;
        EXPECT(p.features(audio.data(), audio.size(), &mel).is_ok());
        const double mel_err = rel_l2(mel, want_mel);
        double mel_max = 0;
        for (size_t i = 0; i < mel.size(); i++) mel_max = std::max(mel_max, std::fabs(double(mel[i]) - want_mel[i]));
        // 2. encoder (Core ML, fp16) on our features
        std::vector<float> states;
        EXPECT(p.encode(mel, &states).is_ok());
        const double enc_err = rel_l2(states, want_enc);
        // 3. decoder on transformers' own encoder states
        MetalWhisperDecoder & dec = p.decoder();
        EXPECT(dec.set_audio(want_enc.data()).is_ok());
        for (size_t i = 0; i < prompt.size(); i++) EXPECT(dec.eval_token(prompt[i], i + 1 == prompt.size()).is_ok());
        const double logit_err = rel_l2(dec.logits().value(), want_logits);
        // teacher-forced: each of transformers' tokens must be our (masked) greedy choice
        size_t agree = 0;
        bool agreeing = true;
        for (size_t i = 0; i < generated.size(); i++) {
            const int32_t ours = dec.pending_token().value();
            if (agreeing && ours == generated[i]) agree++; else agreeing = false;
            if (p.is_end_of_text(generated[i])) break;
            EXPECT(dec.eval_token(generated[i], false).is_ok());
        }
        // 4. the whole path
        WhisperTimings tm;
        auto res = p.transcribe(audio, &tm);
        EXPECT(res.is_ok());
        const std::string want_text = r.get_string(c + "text").value();
        const std::string got = res.is_ok() ? res.value().text : "";
        std::printf("        case %lld: mel rel %.1e (max %.1e), encoder rel %.1e, logits rel %.1e, greedy %zu/%zu, text %s (%.0f ms for %.1f s)\n",
                    static_cast<long long>(k), mel_err, mel_max, enc_err, logit_err, agree, generated.size(), got == want_text ? "identical" : "DIFFERENT", tm.total_ms, tm.audio_s);
        if (got != want_text) std::printf("          ours:         %s\n          transformers: %s\n", got.c_str(), want_text.c_str());
        EXPECT(mel_err < 1e-4);
        EXPECT(enc_err < 5e-2);
        EXPECT(logit_err < 1e-2);
        EXPECT(agree + 1 >= generated.size());   // every token, or all but a final near-tie
        EXPECT(got == want_text);
    }
}

// The suppression masks only matter where the model's own choice is suppressed, which never happens on the reference clips. So the lists are extended
// with the model's own unmasked choices (run 1) and the run repeated (run 2): the begin list must apply to the first generated token only, the other list
// to every token, and the GPU's masked argmax must equal the host's over the same logits.
TEST(suppression_masks_apply) {
    const std::string d = dir();
    if (d.empty()) { std::printf("        (skipped: set DOOMS_WHISPER_DIR)\n"); return; }
    auto file = GgufFile::open(d + "/decoder.gguf");
    auto ref = GgufFile::open(d + "/reference.gguf");
    auto pipe = WhisperPipeline::create({d + "/encoder-fp16.mlpackage", d + "/decoder.gguf"});
    EXPECT(file.is_ok() && ref.is_ok() && pipe.is_ok());
    if (!file.is_ok() || !ref.is_ok() || !pipe.is_ok()) return;
    auto list = [&](const char * key) {
        std::vector<int32_t> v;
        auto it = file.value().metadata().find(key);
        if (it != file.value().metadata().end()) {
            for (const GgufValue & x : it->second.items) v.push_back(static_cast<int32_t>(x.type == GgufValueType::I32 || x.type == GgufValueType::I64 ? x.i : static_cast<int64_t>(x.u)));
        }
        return v;
    };
    auto host_argmax = [](const std::vector<float> & logits, const std::vector<int32_t> & masked) {
        int32_t best = -1;
        for (int32_t t = 0; t < static_cast<int32_t>(logits.size()); t++) {
            if (std::find(masked.begin(), masked.end(), t) != masked.end()) continue;
            if (best < 0 || logits[static_cast<size_t>(t)] > logits[static_cast<size_t>(best)]) best = t;
        }
        return best;
    };
    auto join = [](std::vector<int32_t> a, const std::vector<int32_t> & b) { a.insert(a.end(), b.begin(), b.end()); return a; };
    std::vector<int32_t> always = list("whisper.suppress_tokens"), begin = list("whisper.begin_suppress_tokens");
    EXPECT(!always.empty() && !begin.empty());
    WhisperPipeline & p = pipe.value();
    MetalWhisperDecoder & dec = p.decoder();
    const std::vector<float> states = tensor(ref.value(), "case0.encoder_out");
    const std::vector<int32_t> pr = p.prompt();
    auto run_prompt = [&]() {
        EXPECT(dec.set_audio(states.data()).is_ok());
        for (size_t i = 0; i < pr.size(); i++) EXPECT(dec.eval_token(pr[i], i + 1 == pr.size()).is_ok());
    };
    // run 1: the model's lists
    run_prompt();
    const std::vector<float> l1 = dec.logits().value();
    const int32_t first = dec.pending_token().value(), a = host_argmax(l1, {});
    EXPECT(first == host_argmax(l1, join(always, begin)));
    EXPECT(dec.eval_token(first, false).is_ok());
    const std::vector<float> l2 = dec.logits().value();
    const int32_t c = host_argmax(l2, {});
    EXPECT(dec.pending_token().value() == host_argmax(l2, always));
    // run 2: a joins the begin list, c the other list; the same tokens are fed
    always.push_back(c);
    begin.push_back(a);
    EXPECT(dec.set_suppression(always, begin).is_ok());
    run_prompt();
    const int32_t first2 = dec.pending_token().value();
    EXPECT(first2 != a && first2 == host_argmax(l1, join(always, begin)));
    EXPECT(dec.eval_token(first, false).is_ok());
    const int32_t second2 = dec.pending_token().value();
    EXPECT(second2 != c && second2 == host_argmax(l2, always));
    std::printf("        masks: first token %d -> %d with %d begin-suppressed, second %d with %d suppressed\n", a, first2, a, second2, c);
    // run 3, the queued generate() path: tokens the model generates mid-sequence join the lists (g[3] the other list: never generated; g[5] the begin
    // list: no effect after the first token); generate() must equal stepping eval_token with the host's masked argmax.
    const std::vector<int32_t> base_always = list("whisper.suppress_tokens"), base_begin = list("whisper.begin_suppress_tokens");
    auto generate = [&](int32_t n) {
        std::vector<int32_t> out;
        run_prompt();
        EXPECT(dec.generate(n, 1, [&](int32_t t) { out.push_back(t); return true; }).is_ok());
        return out;
    };
    EXPECT(dec.set_suppression(base_always, base_begin).is_ok());
    const std::vector<int32_t> g = generate(8);
    EXPECT(g.size() == 8);
    if (g.size() != 8) return;
    const std::vector<int32_t> always3 = join(base_always, {g[3]}), begin3 = join(base_begin, {g[5]});
    EXPECT(dec.set_suppression(always3, begin3).is_ok());
    const std::vector<int32_t> h = generate(8);
    run_prompt();
    bool same = h.size() == 8;
    for (size_t i = 0; same && i < h.size(); i++) {
        const int32_t want = host_argmax(dec.logits().value(), i == 0 ? join(always3, begin3) : always3);
        same = dec.pending_token().value() == want && h[i] == want;
        EXPECT(dec.eval_token(h[i], false).is_ok());
    }
    EXPECT(same);
    EXPECT(std::find(h.begin(), h.end(), g[3]) == h.end() && h != g);
}

int main() { return dooms_test::run_all(); }
