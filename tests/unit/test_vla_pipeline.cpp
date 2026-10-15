// Tests the SmolVLA pipeline: camera preprocessing and argument validation (always), and on the converted model (DOOMS_VLA_DIR with vision-b1-fp16.mlpackage,
// prefix-fp16.mlpackage, expert-fp16.mlpackage, host.gguf and reference.gguf from tools/parity/smolvla_reference.py) the action chunk against lerobot's fp32 one.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "format/gguf.h"
#include "pipeline/vla_pipeline.h"

using namespace dooms;

namespace {

std::string vla_dir() {
    const char * d = std::getenv("DOOMS_VLA_DIR");
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

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0, w = 0;
    for (size_t i = 0; i < a.size(); i++) { d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]); w += double(b[i]) * b[i]; }
    return std::sqrt(d / w);
}

VlaPaths paths(const std::string & dir) { return {dir + "/vision-b1-fp16.mlpackage", dir + "/prefix-fp16.mlpackage", dir + "/expert-fp16.mlpackage", dir + "/host.gguf"}; }

CameraImage camera_of(const Tensor & t) {
    CameraImage c;
    c.height = t.shape.dim(1);
    c.width = t.shape.dim(2);
    c.chw = t.data;
    return c;
}

}  // namespace

TEST(camera_preprocessing_follows_resize_with_pad) {
    // 2x4 image (wider than tall) to a 8x8 square: ratio 0.5 -> resized 4x8, padded 4 rows on top with -1 (pad value 0 scaled).
    CameraImage c;
    c.width = 4;
    c.height = 2;
    c.chw.assign(3 * 2 * 4, 0.5f);
    std::vector<float> out(3 * 8 * 8);
    EXPECT(vla_prepare_camera(c, 8, out.data()).is_ok());
    for (int ch = 0; ch < 3; ch++) {
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++) EXPECT_EQ(out[ch * 64 + y * 8 + x], y < 4 ? -1.f : 0.f);  // constant 0.5 -> 0.5 * 2 - 1 = 0 in the image area
    }
    // a pure gradient keeps its order after the resize
    CameraImage g;
    g.width = g.height = 2;
    g.chw.assign(12, 0.f);
    for (int ch = 0; ch < 3; ch++) { g.chw[ch * 4 + 1] = 1.f; g.chw[ch * 4 + 3] = 1.f; }  // left column 0, right column 1
    std::vector<float> o2(3 * 4 * 4);
    EXPECT(vla_prepare_camera(g, 4, o2.data()).is_ok());
    EXPECT(o2[0] < o2[1] && o2[1] < o2[2] && o2[2] < o2[3]);
    EXPECT(std::fabs(o2[0] + 1.f) < 1e-6 && std::fabs(o2[3] - 1.f) < 1e-6);
}

TEST(camera_preprocessing_rejects_bad_images) {
    std::vector<float> out(3 * 8 * 8);
    CameraImage empty;
    EXPECT(!vla_prepare_camera(empty, 8, out.data()).is_ok());
    CameraImage wrong;
    wrong.width = wrong.height = 4;
    wrong.chw.assign(5, 0.f);
    EXPECT(!vla_prepare_camera(wrong, 8, out.data()).is_ok());
    CameraImage nan_img;
    nan_img.width = nan_img.height = 2;
    nan_img.chw.assign(12, 0.f);
    nan_img.chw[3] = std::nanf("");
    EXPECT(!vla_prepare_camera(nan_img, 8, out.data()).is_ok());
}

TEST(camera_from_rgb_converts_layout_and_range) {
    RgbImage f;
    f.width = 2;
    f.height = 1;
    f.pixels = {255, 0, 51, 0, 255, 102};  // two pixels, interleaved RGB
    auto c = camera_from_rgb(f);
    EXPECT(c.is_ok());
    if (c.is_ok()) {
        EXPECT_EQ(c.value().chw.size(), static_cast<size_t>(6));
        const std::vector<float> want = {1.f, 0.f, 0.f, 1.f, 0.2f, 0.4f};  // planes R, G, B
        for (size_t i = 0; i < 6; i++) EXPECT(std::fabs(c.value().chw[i] - want[i]) < 1e-6);
    }
    f.pixels.pop_back();
    EXPECT(!camera_from_rgb(f).is_ok());
    EXPECT(!camera_from_rgb(RgbImage()).is_ok());
}

TEST(normalization_follows_mean_std) {
    auto n = vla_normalize_state({3.f, 10.f}, {1.f, 10.f}, {2.f, 5.f});
    EXPECT(n.is_ok());
    if (n.is_ok()) EXPECT(std::fabs(n.value()[0] - 1.f) < 1e-6 && n.value()[1] == 0.f);
    EXPECT(!vla_normalize_state({1.f}, {1.f, 2.f}, {1.f, 1.f}).is_ok());
    EXPECT(!vla_normalize_state({1.f}, {1.f}, {-1.f}).is_ok());
    EXPECT(!vla_normalize_state({INFINITY}, {1.f}, {1.f}).is_ok());
    // two rows of width 4, statistics for the first 2 columns
    auto a = vla_unnormalize_actions({1.f, 2.f, 9.f, 9.f, -1.f, 0.f, 9.f, 9.f}, 4, {10.f, 20.f}, {2.f, 3.f});
    EXPECT(a.is_ok());
    if (a.is_ok()) {
        EXPECT_EQ(a.value().size(), static_cast<size_t>(4));
        EXPECT(a.value()[0] == 12.f && a.value()[1] == 26.f && a.value()[2] == 8.f && a.value()[3] == 20.f);
    }
    EXPECT(!vla_unnormalize_actions({1.f, 2.f, 3.f}, 2, {0.f}, {1.f}).is_ok());
    EXPECT(!vla_unnormalize_actions({1.f, 2.f}, 2, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}).is_ok());
}

TEST(create_rejects_missing_files) {
    EXPECT(!VlaPipeline::create({"/nonexistent/v.mlpackage", "/nonexistent/p.mlpackage", "/nonexistent/e.mlpackage", "/nonexistent/host.gguf"}).is_ok());
}

TEST(infer_validates_the_observation) {
    const std::string dir = vla_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLA_DIR)\n"); return; }
    auto pipe = VlaPipeline::create(paths(dir));
    EXPECT(pipe.is_ok());
    if (!pipe.is_ok()) return;
    VlaPipeline & v = pipe.value();
    VlaObservation ok;
    CameraImage cam;
    cam.width = cam.height = 8;
    cam.chw.assign(3 * 64, 0.5f);
    ok.cameras.assign(static_cast<size_t>(v.n_cameras()), cam);
    ok.tokens.assign(static_cast<size_t>(v.tokenizer_max_length()), 1);
    ok.token_mask.assign(ok.tokens.size(), 1);
    ok.state.assign(6, 0.f);
    VlaObservation bad = ok;
    bad.cameras.pop_back();
    EXPECT(!v.infer(bad).is_ok());
    bad = ok; bad.tokens.pop_back();
    EXPECT(!v.infer(bad).is_ok());
    bad = ok; bad.token_mask.pop_back();
    EXPECT(!v.infer(bad).is_ok());
    bad = ok; bad.state.assign(static_cast<size_t>(v.state_dim()) + 1, 0.f);
    EXPECT(!v.infer(bad).is_ok());
    bad = ok; bad.state[0] = INFINITY;
    EXPECT(!v.infer(bad).is_ok());
    bad = ok; bad.tokens[0] = -1;
    EXPECT(!v.infer(bad).is_ok());
    bad = ok; bad.tokens[0] = 1 << 30;
    EXPECT(!v.infer(bad).is_ok());
    std::vector<float> short_noise(5, 0.f);
    EXPECT(!v.infer(ok, &short_noise).is_ok());
}

TEST(action_chunk_matches_lerobot) {
    const std::string dir = vla_dir();
    if (dir.empty()) { std::printf("        (skipped: set DOOMS_VLA_DIR)\n"); return; }
    auto ref = GgufFile::open(dir + "/reference.gguf");
    auto pipe = VlaPipeline::create(paths(dir));
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) return;
    VlaPipeline & v = pipe.value();
    const GgufFile & r = ref.value();
    VlaObservation obs;
    for (const char * name : {"image_camera1", "image_camera2", "image_camera3"}) obs.cameras.push_back(camera_of(get(r, name)));
    for (float t : get(r, "tokens").data) obs.tokens.push_back(static_cast<int32_t>(t));
    for (float t : get(r, "token_mask").data) obs.token_mask.push_back(t != 0.f ? 1 : 0);
    obs.state = get(r, "state").data;
    const std::vector<float> noise = get(r, "noise").data, want = get(r, "actions").data;
    EXPECT_EQ(static_cast<int64_t>(obs.tokens.size()), v.tokenizer_max_length());

    // the camera preprocessing against lerobot's prepared images
    for (int k = 1; k <= 3; k++) {
        const Tensor prepared = get(r, "prepared_camera" + std::to_string(k));
        std::vector<float> out(prepared.data.size());
        EXPECT(vla_prepare_camera(obs.cameras[static_cast<size_t>(k - 1)], 512, out.data()).is_ok());
        double worst = 0;
        for (size_t i = 0; i < out.size(); i++) worst = std::max(worst, double(std::fabs(out[i] - prepared.data[i])));
        std::printf("        camera%d: largest difference to lerobot's prepared image %.2e\n", k, worst);
        EXPECT(worst < 1e-5);
    }

    VlaTimings tm;
    auto got = v.infer(obs, &noise, &tm);
    EXPECT(got.is_ok());
    if (!got.is_ok()) return;
    EXPECT_EQ(got.value().size(), want.size());
    const double rel = rel_l2(got.value(), want);
    std::vector<float> g6, w6;  // the robot's 6 action dimensions
    for (int64_t i = 0; i < v.chunk_size(); i++)
        for (int64_t j = 0; j < 6; j++) { g6.push_back(got.value()[static_cast<size_t>(i * v.action_dim() + j)]); w6.push_back(want[static_cast<size_t>(i * v.action_dim() + j)]); }
    const double rel6 = rel_l2(g6, w6);
    std::printf("        action chunk relative L2 %.2e (robot dimensions %.2e); vision %.1f ms, host %.1f ms, prefix %.1f ms, expert %.1f ms, total %.1f ms\n", rel, rel6, tm.vision_ms, tm.host_ms,
                tm.prefix_ms, tm.expert_ms, tm.total_ms);
    EXPECT(rel < 5e-2);
    EXPECT(rel6 < 5e-2);
    // a different noise gives a different (but finite) chunk; the same noise the same one
    auto again = v.infer(obs, &noise);
    EXPECT(again.is_ok() && rel_l2(again.value(), got.value()) < 1e-3);
    auto random = v.infer(obs);
    EXPECT(random.is_ok() && rel_l2(random.value(), got.value()) > 1e-2);
}

int main() { return dooms_test::run_all(); }
