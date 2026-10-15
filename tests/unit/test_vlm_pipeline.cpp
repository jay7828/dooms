// Tests the VLM pipeline: argument/file validation (always), and on the real SmolVLM files (DOOMS_VLM_DIR with vision-fp16.mlpackage, text-f16.gguf,
// vlm-config.gguf, reference.gguf) the image embeddings and the generated answer against transformers, photo in -> text out.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "format/gguf.h"
#include "pipeline/vlm_pipeline.h"

using namespace dooms;

namespace {

std::string vlm_dir() {
    const char * d = std::getenv("DOOMS_VLM_DIR");
    return d ? d : "";
}

struct Tensor {
    std::vector<float> data;
    Shape shape;
};

Tensor get(const GgufFile & f, const std::string & name) {
    Tensor t;
    auto info = f.find_tensor(name);
    if (!info.is_ok()) { std::fprintf(stderr, "  %s\n", info.status().to_string().c_str()); dooms_test::current_failures()++; return t; }
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    t.data.assign(p, p + info.value()->shape.numel());
    t.shape = info.value()->shape;
    return t;
}

}  // namespace

TEST(create_rejects_missing_files) {
    VlmPaths p{"/nonexistent/vision.mlpackage", "/nonexistent/text.gguf", "/nonexistent/config.gguf"};
    auto r = VlmPipeline::create(p);
    EXPECT(!r.is_ok());
}

TEST(create_rejects_a_config_that_is_not_a_vlm_config) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    // The text model is a valid GGUF but not a vlm-config.
    auto r = VlmPipeline::create({dir + "/vision-fp16.mlpackage", dir + "/text-f16.gguf", dir + "/text-f16.gguf"});
    EXPECT(!r.is_ok());
}

TEST(generate_validates_arguments) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto pipe = VlmPipeline::create({dir + "/vision-fp16.mlpackage", dir + "/text-f16.gguf", dir + "/vlm-config.gguf"});
    EXPECT(pipe.is_ok());
    if (!pipe.is_ok()) return;
    VlmPipeline & v = pipe.value();
    std::vector<float> emb(static_cast<size_t>(v.n_image_tokens() * v.embedding_dim()), 0.f);
    EXPECT(!v.generate_from_embeddings(emb, "", 8).is_ok());
    EXPECT(!v.generate_from_embeddings(emb, "hi", 0).is_ok());
    EXPECT(!v.generate_from_embeddings(emb, "hi", 100000).is_ok());
    EXPECT(!v.generate_from_embeddings(std::vector<float>(10), "hi", 8).is_ok());
    RgbImage empty;
    EXPECT(!v.generate(empty, "hi", 8).is_ok());
    // Prompt + answer larger than the context.
    EXPECT(v.generate_from_embeddings(emb, std::string(20000, 'a'), 8).status().code() == Code::ResourceExhausted);
}

TEST(callback_can_stop_the_generation) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto ref = GgufFile::open(dir + "/reference.gguf");
    auto pipe = VlmPipeline::create({dir + "/vision-fp16.mlpackage", dir + "/text-f16.gguf", dir + "/vlm-config.gguf"});
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) return;
    const Tensor photo = get(ref.value(), "case0.photo");
    RgbImage img;
    img.height = photo.shape.dim(0);
    img.width = photo.shape.dim(1);
    for (float x : photo.data) img.pixels.push_back(static_cast<uint8_t>(x));
    const std::string prompt = ref.value().get_string("case0.prompt").value();
    auto full = pipe.value().generate(img, prompt, 64);
    EXPECT(full.is_ok() && full.value().tokens.size() > 4);
    if (!full.is_ok()) return;
    int pieces = 0;
    auto cut = pipe.value().generate(img, prompt, 64, [&](const std::string &) { return ++pieces < 3; });
    EXPECT(cut.is_ok());
    if (!cut.is_ok()) return;
    EXPECT(cut.value().cancelled && !cut.value().stopped);
    EXPECT_EQ(pieces, 3);
    EXPECT_EQ(cut.value().tokens.size(), static_cast<size_t>(3));
    EXPECT(std::equal(cut.value().tokens.begin(), cut.value().tokens.end(), full.value().tokens.begin()));  // the same tokens as an uninterrupted run
    EXPECT(!full.value().cancelled);
}

TEST(photo_to_answer_matches_transformers) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto ref = GgufFile::open(dir + "/reference.gguf");
    auto pipe = VlmPipeline::create({dir + "/vision-fp16.mlpackage", dir + "/text-f16.gguf", dir + "/vlm-config.gguf"});
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) return;
    VlmPipeline & v = pipe.value();
    for (int k = 0; k < 4; k++) {
        const std::string p = "case" + std::to_string(k) + ".";
        const Tensor photo = get(ref.value(), p + "photo"), want_emb = get(ref.value(), p + "image_embeddings"), generated = get(ref.value(), p + "generated");
        const std::string prompt = ref.value().get_string(p + "prompt").value();
        RgbImage img;
        img.height = photo.shape.dim(0);
        img.width = photo.shape.dim(1);
        for (float x : photo.data) img.pixels.push_back(static_cast<uint8_t>(x));

        std::vector<float> emb;
        EXPECT(v.encode_image(img, &emb).is_ok());
        EXPECT_EQ(emb.size(), want_emb.data.size());
        if (emb.size() != want_emb.data.size()) continue;
        double dd = 0, ww = 0;
        for (size_t i = 0; i < emb.size(); i++) {
            dd += (double(emb[i]) - want_emb.data[i]) * (double(emb[i]) - want_emb.data[i]);
            ww += double(want_emb.data[i]) * want_emb.data[i];
        }
        const double rel = std::sqrt(dd / ww);
        EXPECT(rel < 5e-2);  // Neural Engine fp16 vs fp32 transformers

        VlmTimings tm;
        std::string streamed;
        auto out = v.generate(img, prompt, 64, [&](const std::string & s) { streamed += s; return true; }, &tm);
        EXPECT(out.is_ok());
        if (!out.is_ok()) continue;
        EXPECT_EQ(streamed, out.value().text);
        EXPECT(!out.value().tokens.empty());
        size_t agree = 0;
        while (agree < out.value().tokens.size() && agree < generated.data.size() && out.value().tokens[agree] == static_cast<int32_t>(generated.data[agree])) agree++;
        std::printf("        case %d: embeddings rel L2 %.2e, %zu of %zu tokens equal transformers, %lld new tokens, prefill %.1f ms, first token %.1f ms, %.1f ms/token\n", k, rel, agree,
                    generated.data.size(), static_cast<long long>(tm.generated_tokens), tm.prefill_ms, tm.first_token_ms, tm.decode_ms_per_token());
        EXPECT(tm.generated_tokens == static_cast<int64_t>(out.value().tokens.size()));
        EXPECT(agree >= 3);
    }
}

// text_engine: "auto" runs the own Metal decoder on the f16 file and falls back to llama.cpp for a quantized one; "metal" refuses what it cannot run;
// both engines answer the reference photo with transformers' first tokens.
TEST(text_engine_selection_and_fallback) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    const VlmPaths f16{dir + "/vision-fp16.mlpackage", dir + "/text-f16.gguf", dir + "/vlm-config.gguf"};
    VlmOptions o;
    o.text_engine = "bogus";
    EXPECT(VlmPipeline::create(f16, o).status().code() == Code::InvalidArgument);
    o.text_engine = "metal";
    o.metal_queue_depth = 0;
    EXPECT(VlmPipeline::create(f16, o).status().code() == Code::InvalidArgument);
    auto ref = GgufFile::open(dir + "/reference.gguf");
    EXPECT(ref.is_ok());
    if (!ref.is_ok()) return;
    const Tensor photo = get(ref.value(), "case0.photo"), generated = get(ref.value(), "case0.generated");
    RgbImage img;
    img.height = photo.shape.dim(0);
    img.width = photo.shape.dim(1);
    for (float x : photo.data) img.pixels.push_back(static_cast<uint8_t>(x));
    const std::string prompt = ref.value().get_string("case0.prompt").value();
    for (const char * engine : {"auto", "metal", "llama.cpp"}) {
        VlmOptions eo;
        eo.text_engine = engine;
        auto pipe = VlmPipeline::create(f16, eo);
        EXPECT(pipe.is_ok());
        if (!pipe.is_ok()) continue;
        EXPECT_EQ(std::string(pipe.value().text_engine_used()), std::string(std::string(engine) == "llama.cpp" ? "llama.cpp" : "metal"));
        auto out = pipe.value().generate(img, prompt, 32);
        EXPECT(out.is_ok());
        if (!out.is_ok()) continue;
        size_t agree = 0;
        while (agree < out.value().tokens.size() && agree < generated.data.size() && out.value().tokens[agree] == static_cast<int32_t>(generated.data[agree])) agree++;
        std::printf("        %s (%s): %zu of %zu tokens equal transformers\n", engine, pipe.value().text_engine_used(), agree, generated.data.size());
        EXPECT(agree >= 3);
    }
    const std::string q = dir + "/text-Q4_K_M.gguf";
    if (FILE * fp = std::fopen(q.c_str(), "rb")) {
        std::fclose(fp);
        const VlmPaths quant{f16.vision, q, f16.config};
        auto pipe = VlmPipeline::create(quant);
        EXPECT(pipe.is_ok());
        if (pipe.is_ok()) EXPECT_EQ(std::string(pipe.value().text_engine_used()), std::string("llama.cpp"));
        VlmOptions mo;
        mo.text_engine = "metal";
        EXPECT(VlmPipeline::create(quant, mo).status().code() == Code::Unsupported);
    }
}

TEST(generate_many_equals_generate_one_by_one) {
    const std::string dir = vlm_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    auto ref = GgufFile::open(dir + "/reference.gguf");
    auto pipe = VlmPipeline::create({dir + "/vision-fp16.mlpackage", dir + "/text-f16.gguf", dir + "/vlm-config.gguf"});
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) return;
    std::vector<RgbImage> images(4);
    std::vector<VlmRequest> requests;
    for (int k = 0; k < 4; k++) {
        const std::string p = "case" + std::to_string(k) + ".";
        const Tensor photo = get(ref.value(), p + "photo");
        images[static_cast<size_t>(k)].height = photo.shape.dim(0);
        images[static_cast<size_t>(k)].width = photo.shape.dim(1);
        for (float x : photo.data) images[static_cast<size_t>(k)].pixels.push_back(static_cast<uint8_t>(x));
        requests.push_back({&images[static_cast<size_t>(k)], ref.value().get_string(p + "prompt").value()});
    }
    double wall = 0, serial = 0;
    auto many = pipe.value().generate_many(requests, 24, &wall);
    EXPECT(many.is_ok());
    if (!many.is_ok()) return;
    EXPECT_EQ(many.value().size(), requests.size());
    for (size_t k = 0; k < requests.size(); k++) {
        VlmTimings tm;
        auto one = pipe.value().generate(*requests[k].image, requests[k].prompt, 24, nullptr, &tm);
        EXPECT(one.is_ok());
        if (!one.is_ok()) continue;
        serial += tm.preprocess_ms + tm.vision_ms + tm.first_token_ms + tm.decode_ms;
        EXPECT(many.value()[k].tokens == one.value().tokens);
    }
    std::printf("        4 requests: overlapped %.1f ms, one by one %.1f ms\n", wall, serial);
    EXPECT(!pipe.value().generate_many({{nullptr, "x"}}, 8).is_ok());
    EXPECT(!pipe.value().generate_many({{&images[0], ""}}, 8).is_ok());
    EXPECT(pipe.value().generate_many({}, 8).is_ok());
}

int main() { return dooms_test::run_all(); }
