// Tests the llama.cpp text decoder wrapper: argument validation (always), and on the real SmolVLM language model (DOOMS_VLM_DIR with text-f16.gguf,
// vlm-config.gguf and reference.gguf from tools/convert/convert_smolvlm.py + tools/parity/smolvlm_reference.py): the prompt tokenization against the
// processor's input ids, the logits and the greedy continuation after feeding transformers' own image embeddings.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "format/gguf.h"
#include "models/vlm/text_decoder.h"

using namespace dooms;

namespace {

std::string vlm_dir() {
    const char * d = std::getenv("DOOMS_VLM_DIR");
    return d ? d : "";
}

struct Reference {
    std::vector<float> data;
    Shape shape;
};

Reference get(const GgufFile & f, const std::string & name) {
    Reference r;
    auto info = f.find_tensor(name);
    if (!info.is_ok()) { std::fprintf(stderr, "  %s\n", info.status().to_string().c_str()); dooms_test::current_failures()++; return r; }
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    r.data.assign(p, p + info.value()->shape.numel());
    r.shape = info.value()->shape;
    return r;
}

std::vector<int32_t> ids_of(const Reference & r) {
    std::vector<int32_t> ids;
    for (float v : r.data) ids.push_back(static_cast<int32_t>(v));
    return ids;
}

struct Layout {
    std::string prefix, infix, suffix;
    int64_t n_image = 0, image_token_id = 0, embedding_dim = 0;
};

Layout read_layout(const GgufFile & f) {
    Layout l;
    l.prefix = f.get_string("vlm_config.prefix").value();
    l.infix = f.get_string("vlm_config.infix").value();
    l.suffix = f.get_string("vlm_config.suffix").value();
    l.n_image = static_cast<int64_t>(f.get_uint("vlm_config.n_image_tokens").value());
    l.image_token_id = static_cast<int64_t>(f.get_uint("vlm_config.image_token_id").value());
    l.embedding_dim = static_cast<int64_t>(f.get_uint("vlm_config.embedding_dim").value());
    return l;
}

}  // namespace

TEST(load_validates_options_and_the_file) {
    TextDecoderOptions bad;
    bad.n_ctx = 8;
    EXPECT(TextDecoder::load("/nonexistent/x.gguf", bad).status().code() == Code::InvalidArgument);
    bad = TextDecoderOptions();
    bad.n_batch = 0;
    EXPECT(TextDecoder::load("/nonexistent/x.gguf", bad).status().code() == Code::InvalidArgument);
    bad = TextDecoderOptions();
    bad.n_batch = bad.n_ctx + 1;
    EXPECT(TextDecoder::load("/nonexistent/x.gguf", bad).status().code() == Code::InvalidArgument);
    bad = TextDecoderOptions();
    bad.n_gpu_layers = -2;
    EXPECT(TextDecoder::load("/nonexistent/x.gguf", bad).status().code() == Code::InvalidArgument);
    EXPECT(TextDecoder::load("/nonexistent/dooms/none.gguf").status().code() == Code::NotFound);
    const std::string dir = vlm_dir();
    if (!dir.empty()) EXPECT(TextDecoder::load(dir + "/vlm-config.gguf").status().code() == Code::DataLoss);  // a GGUF, but not a model llama.cpp can run
}

TEST(tokenization_matches_the_processor_and_the_api_is_strict) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto dec = TextDecoder::load(dir + "/text-f16.gguf");
    auto cfg = GgufFile::open(dir + "/vlm-config.gguf");
    auto ref = GgufFile::open(dir + "/reference.gguf");
    EXPECT(dec.is_ok() && cfg.is_ok() && ref.is_ok());
    if (!dec.is_ok() || !cfg.is_ok() || !ref.is_ok()) { if (!dec.is_ok()) std::fprintf(stderr, "  %s\n", dec.status().to_string().c_str()); return; }
    const Layout l = read_layout(cfg.value());
    EXPECT_EQ(dec.value().embedding_dim(), l.embedding_dim);
    for (int k = 0; k < 4; k++) {
        const std::string p = "case" + std::to_string(k) + ".";
        const std::string prompt = ref.value().get_string(p + "prompt").value();
        auto before = dec.value().tokenize(l.prefix, true);
        auto after = dec.value().tokenize(l.infix + prompt + l.suffix, true);
        EXPECT(before.is_ok() && after.is_ok());
        if (!before.is_ok() || !after.is_ok()) continue;
        std::vector<int32_t> ids = before.value();
        ids.insert(ids.end(), static_cast<size_t>(l.n_image), static_cast<int32_t>(l.image_token_id));
        ids.insert(ids.end(), after.value().begin(), after.value().end());
        EXPECT(ids == ids_of(get(ref.value(), p + "input_ids")));
    }
    EXPECT(dec.value().tokenize("", true).status().code() == Code::InvalidArgument);
    // special-token strings are parsed only when asked to
    auto parsed = dec.value().tokenize("<image>", true), plain = dec.value().tokenize("<image>", false);
    EXPECT(parsed.is_ok() && plain.is_ok() && parsed.value().size() == 1 && plain.value().size() > 1 && parsed.value()[0] == l.image_token_id);
    EXPECT(dec.value().piece(static_cast<int32_t>(l.image_token_id)) == "<image>");
    EXPECT(dec.value().piece(-1).empty() && dec.value().piece(1 << 30).empty());
    // end of generation: the end-of-utterance token ends it, an ordinary token does not
    auto eou = dec.value().tokenize("<end_of_utterance>", true);
    EXPECT(eou.is_ok() && eou.value().size() == 1 && dec.value().is_end_of_generation(eou.value()[0]) && !dec.value().is_end_of_generation(parsed.value()[0]));
    EXPECT(!dec.value().is_end_of_generation(-3));
}

TEST(evaluation_arguments_are_validated_and_state_is_recoverable) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    TextDecoderOptions o;
    o.n_ctx = 32;
    o.n_batch = 32;
    auto dec = TextDecoder::load(dir + "/text-f16.gguf", o);
    EXPECT(dec.is_ok());
    if (!dec.is_ok()) { std::fprintf(stderr, "  %s\n", dec.status().to_string().c_str()); return; }
    TextDecoder & d = dec.value();
    EXPECT(d.greedy_next().status().code() == Code::InvalidArgument);  // nothing evaluated yet
    EXPECT(d.logits().status().code() == Code::InvalidArgument);
    EXPECT(d.eval_tokens({}).code() == Code::InvalidArgument);
    EXPECT(d.eval_tokens({-1}).code() == Code::InvalidArgument);
    EXPECT(d.eval_tokens({static_cast<int32_t>(d.vocab_size())}).code() == Code::InvalidArgument);
    EXPECT(d.eval_embeddings(nullptr, 3).code() == Code::InvalidArgument);
    std::vector<float> nan_emb(static_cast<size_t>(d.embedding_dim()), 0.f);
    nan_emb[3] = std::nanf("");
    EXPECT(d.eval_embeddings(nan_emb.data(), 1).code() == Code::InvalidArgument);
    EXPECT_EQ(d.position(), 0);  // none of the rejected calls advanced the cache
    auto ids = d.tokenize("Hello there, how are you doing today?", false);
    EXPECT(ids.is_ok());
    if (!ids.is_ok()) return;
    EXPECT(d.eval_tokens(ids.value()).is_ok());
    EXPECT_EQ(d.position(), static_cast<int32_t>(ids.value().size()));
    auto g1 = d.greedy_next();
    EXPECT(g1.is_ok() && g1.value() >= 0 && g1.value() < d.vocab_size());
    auto lg = d.logits();
    EXPECT(lg.is_ok() && static_cast<int64_t>(lg.value().size()) == d.vocab_size());
    // overflow is reported and leaves the evaluated part usable
    std::vector<int32_t> many(40, ids.value()[0]);
    EXPECT(d.eval_tokens(many).code() == Code::ResourceExhausted);
    EXPECT_EQ(d.position(), static_cast<int32_t>(ids.value().size()));
    // reset gives the same answer again (deterministic), and a different prompt a different one
    d.reset();
    EXPECT_EQ(d.position(), 0);
    EXPECT(d.greedy_next().status().code() == Code::InvalidArgument);
    EXPECT(d.eval_tokens(ids.value()).is_ok());
    auto g2 = d.greedy_next();
    EXPECT(g2.is_ok() && g2.value() == g1.value());
    auto lg2 = d.logits();
    EXPECT(lg2.is_ok() && lg2.value() == lg.value());
}

TEST(logits_and_greedy_tokens_match_transformers_given_its_image_embeddings) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto dec = TextDecoder::load(dir + "/text-f16.gguf");
    auto cfg = GgufFile::open(dir + "/vlm-config.gguf");
    auto ref = GgufFile::open(dir + "/reference.gguf");
    EXPECT(dec.is_ok() && cfg.is_ok() && ref.is_ok());
    if (!dec.is_ok() || !cfg.is_ok() || !ref.is_ok()) return;
    TextDecoder & d = dec.value();
    const Layout l = read_layout(cfg.value());
    for (int k = 0; k < 4; k++) {
        const std::string p = "case" + std::to_string(k) + ".";
        const std::string prompt = ref.value().get_string(p + "prompt").value();
        const Reference emb = get(ref.value(), p + "image_embeddings"), want_logits = get(ref.value(), p + "logits");
        const std::vector<int32_t> generated = ids_of(get(ref.value(), p + "generated"));
        d.reset();
        EXPECT(d.eval_tokens(d.tokenize(l.prefix, true).value()).is_ok());
        EXPECT(d.eval_embeddings(emb.data.data(), l.n_image).is_ok());
        EXPECT(d.eval_tokens(d.tokenize(l.infix + prompt + l.suffix, true).value()).is_ok());
        EXPECT_EQ(d.position(), static_cast<int32_t>(ids_of(get(ref.value(), p + "input_ids")).size()));
        auto lg = d.logits();
        EXPECT(lg.is_ok());
        if (!lg.is_ok()) continue;
        double dd = 0, ww = 0;
        int best_ours = 0, best_ref = 0;
        for (size_t i = 0; i < lg.value().size(); i++) {
            dd += (double(lg.value()[i]) - want_logits.data[i]) * (double(lg.value()[i]) - want_logits.data[i]);
            ww += double(want_logits.data[i]) * want_logits.data[i];
            if (lg.value()[i] > lg.value()[static_cast<size_t>(best_ours)]) best_ours = static_cast<int>(i);
            if (want_logits.data[i] > want_logits.data[static_cast<size_t>(best_ref)]) best_ref = static_cast<int>(i);
        }
        const double rel = std::sqrt(dd / ww);
        EXPECT(rel < 2e-2);  // f16 weights vs fp32 transformers
        EXPECT_EQ(best_ours, best_ref);
        // Teacher-forced greedy continuation: feed transformers' own tokens and require that each of them is (within a hair) our argmax. An exact
        // free-running match is not a fair test of f16 weights against fp32: two tokens whose logits differ by ~1e-2 swap places and everything after
        // diverges. A real error (wrong cache, wrong positions) would make transformers' token lose by a wide margin.
        size_t free_agree = 0;
        bool still_agreeing = true;
        double worst_gap = 0;
        std::vector<int32_t> free_tokens;
        for (size_t t = 0; t < generated.size(); t++) {
            auto cur = d.logits();
            EXPECT(cur.is_ok());
            if (!cur.is_ok()) break;
            const float top = *std::max_element(cur.value().begin(), cur.value().end());
            const double gap = static_cast<double>(top) - cur.value()[static_cast<size_t>(generated[t])];
            worst_gap = std::max(worst_gap, gap);
            auto ours = d.greedy_next();
            if (still_agreeing && ours.is_ok() && ours.value() == generated[t]) free_agree++; else still_agreeing = false;
            EXPECT(d.eval_tokens({generated[t]}).is_ok());
        }
        std::printf("        case %d: logits relative L2 %.2e, free-running greedy equal for %zu of %zu tokens, largest margin by which a transformers token loses: %.4f\n", k, rel, free_agree, generated.size(), worst_gap);
        EXPECT(worst_gap < 0.1);
        EXPECT(free_agree >= 3);
    }
}

TEST(embedded_tokens_equal_token_evaluation) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto dec = TextDecoder::load(dir + "/text-f16.gguf");
    EXPECT(dec.is_ok());
    if (!dec.is_ok()) return;
    TextDecoder & d = dec.value();
    auto toks = d.tokenize("The quick brown fox jumps over the lazy dog.", false, true);
    EXPECT(toks.is_ok());
    if (!toks.is_ok()) return;
    EXPECT(d.eval_tokens(toks.value()).is_ok());
    auto a = d.logits();
    auto emb = d.embed_tokens(toks.value());
    EXPECT(a.is_ok() && emb.is_ok());
    if (!a.is_ok() || !emb.is_ok()) return;
    EXPECT_EQ(emb.value().size(), toks.value().size() * static_cast<size_t>(d.embedding_dim()));
    d.reset();
    EXPECT(d.eval_embeddings(emb.value().data(), static_cast<int64_t>(toks.value().size())).is_ok());
    auto b = d.logits();
    EXPECT(b.is_ok());
    if (!b.is_ok()) return;
    double diff = 0, ref = 0;
    for (size_t i = 0; i < a.value().size(); i++) { diff += (double(a.value()[i]) - b.value()[i]) * (double(a.value()[i]) - b.value()[i]); ref += double(a.value()[i]) * a.value()[i]; }
    std::printf("        logits after embedding the tokens ourselves vs llama.cpp's own lookup: relative L2 %.2e\n", std::sqrt(diff / ref));
    EXPECT(std::sqrt(diff / ref) < 1e-5);
    EXPECT(!d.embed_tokens({}).is_ok());
    EXPECT(!d.embed_tokens({-1}).is_ok());
    EXPECT(!d.embed_tokens({static_cast<int32_t>(d.vocab_size())}).is_ok());
}

int main() { return dooms_test::run_all(); }
