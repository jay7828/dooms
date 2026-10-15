#include "pipeline/vla_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <random>

#include "format/gguf.h"

namespace dooms {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr float kMaskedBias = -30000.f;  // additive; exp() of it is exactly 0 in fp16 after the max subtraction (matches tools/convert/smolvla_model.py NEG)
constexpr int64_t kMaxSide = 16384;

Result<Shape> shape_of(std::initializer_list<int64_t> dims) { return Shape::create(dims); }

Result<TensorView> view(void * data, std::initializer_list<int64_t> dims, DType type = DType::F32) {
    auto s = shape_of(dims);
    if (!s.is_ok()) return s.status();
    return TensorView::create(data, s.value(), type);
}

// torch's F.interpolate(mode="bilinear", align_corners=False) on one channel plane, float32.
void resize_bilinear(const float * src, int64_t sh, int64_t sw, float * dst, int64_t dh, int64_t dw) {
    const float hs = static_cast<float>(sh) / static_cast<float>(dh), ws = static_cast<float>(sw) / static_cast<float>(dw);
    auto coords = [](int64_t out, int64_t in, float scale, std::vector<int64_t> * i0, std::vector<int64_t> * i1, std::vector<float> * lam) {
        i0->resize(static_cast<size_t>(out));
        i1->resize(static_cast<size_t>(out));
        lam->resize(static_cast<size_t>(out));
        for (int64_t o = 0; o < out; o++) {
            float s = scale * (static_cast<float>(o) + 0.5f) - 0.5f;
            if (s < 0.f) s = 0.f;
            const int64_t a = std::min<int64_t>(static_cast<int64_t>(s), in - 1);
            (*i0)[static_cast<size_t>(o)] = a;
            (*i1)[static_cast<size_t>(o)] = std::min<int64_t>(a + 1, in - 1);
            (*lam)[static_cast<size_t>(o)] = s - static_cast<float>(a);
        }
    };
    std::vector<int64_t> y0, y1, x0, x1;
    std::vector<float> ly, lx;
    coords(dh, sh, hs, &y0, &y1, &ly);
    coords(dw, sw, ws, &x0, &x1, &lx);
    for (int64_t y = 0; y < dh; y++) {
        const float * r0 = src + y0[static_cast<size_t>(y)] * sw;
        const float * r1 = src + y1[static_cast<size_t>(y)] * sw;
        const float wy = ly[static_cast<size_t>(y)];
        for (int64_t x = 0; x < dw; x++) {
            const size_t xi = static_cast<size_t>(x);
            const float top = r0[x0[xi]] * (1.f - lx[xi]) + r0[x1[xi]] * lx[xi];
            const float bot = r1[x0[xi]] * (1.f - lx[xi]) + r1[x1[xi]] * lx[xi];
            dst[y * dw + x] = top * (1.f - wy) + bot * wy;
        }
    }
}

// cos and sin of position / max_wavelength^(2 i / head_dim), float32, [len, head_dim / 2] (lerobot's apply_rope).
void rope_tables(const std::vector<int64_t> & positions, int64_t head_dim, double max_wavelength, float * cos_out, float * sin_out) {
    const int64_t half = head_dim / 2;
    std::vector<float> timescale(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; i++) timescale[static_cast<size_t>(i)] = std::pow(static_cast<float>(max_wavelength), (2.0f / static_cast<float>(head_dim)) * static_cast<float>(i));
    for (size_t p = 0; p < positions.size(); p++) {
        for (int64_t i = 0; i < half; i++) {
            const float r = static_cast<float>(positions[p]) / timescale[static_cast<size_t>(i)];
            cos_out[p * static_cast<size_t>(half) + static_cast<size_t>(i)] = std::cos(r);
            sin_out[p * static_cast<size_t>(half) + static_cast<size_t>(i)] = std::sin(r);
        }
    }
}

// lerobot's create_sinusoidal_pos_embedding for one time value, float64 inside.
void time_embedding(double t, int64_t dim, double min_period, double max_period, float * out) {
    const int64_t half = dim / 2;
    const double step = half > 1 ? 1.0 / static_cast<double>(half - 1) : 0.0;
    for (int64_t i = 0; i < half; i++) {
        const double fraction = (i == half - 1 && half > 1) ? 1.0 : static_cast<double>(i) * step;
        const double period = min_period * std::pow(max_period / min_period, fraction);
        const double arg = (1.0 / period * 2.0 * M_PI) * t;
        out[i] = static_cast<float>(std::sin(arg));
        out[half + i] = static_cast<float>(std::cos(arg));
    }
}

float bf16_to_float(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

}  // namespace

// lerobot's resize_with_pad(pad_value=0) followed by x * 2 - 1, into [3, size, size].
Status vla_prepare_camera(const CameraImage & img, int64_t size, float * out) {
    if (img.width < 1 || img.height < 1 || img.width > kMaxSide || img.height > kMaxSide) return invalid_argument("a camera image has a side outside [1, 16384]");
    if (static_cast<int64_t>(img.chw.size()) != 3 * img.width * img.height) return invalid_argument("a camera image buffer does not match 3 * height * width");
    for (float v : img.chw) if (!std::isfinite(v)) return invalid_argument("a camera image has non-finite values");
    const double ratio = std::max(static_cast<double>(img.width) / static_cast<double>(size), static_cast<double>(img.height) / static_cast<double>(size));
    const int64_t rh = std::max<int64_t>(1, static_cast<int64_t>(static_cast<double>(img.height) / ratio)), rw = std::max<int64_t>(1, static_cast<int64_t>(static_cast<double>(img.width) / ratio));
    const int64_t pad_h = std::max<int64_t>(0, size - rh), pad_w = std::max<int64_t>(0, size - rw);  // padding goes on the top and left
    const int64_t oh = rh + pad_h, ow = rw + pad_w;
    if (oh != size || ow != size) return invalid_argument("the resized camera image does not fit the model's square input");
    std::vector<float> plane(static_cast<size_t>(rh * rw));
    for (int c = 0; c < 3; c++) {
        float * dst = out + c * size * size;
        std::fill(dst, dst + size * size, -1.f);  // padding value 0 scaled to -1
        resize_bilinear(img.chw.data() + c * img.width * img.height, img.height, img.width, plane.data(), rh, rw);
        for (int64_t y = 0; y < rh; y++)
            for (int64_t x = 0; x < rw; x++) dst[(y + pad_h) * size + x + pad_w] = plane[static_cast<size_t>(y * rw + x)] * 2.f - 1.f;
    }
    return Status();
}

Result<CameraImage> camera_from_rgb(const RgbImage & frame) {
    if (frame.width < 1 || frame.height < 1 || frame.width > kMaxSide || frame.height > kMaxSide || static_cast<int64_t>(frame.pixels.size()) != frame.width * frame.height * 3) {
        return invalid_argument("the camera frame is empty or its buffer does not match width * height * 3");
    }
    CameraImage out;
    out.width = frame.width;
    out.height = frame.height;
    const int64_t plane = frame.width * frame.height;
    out.chw.resize(static_cast<size_t>(3 * plane));
    for (int64_t i = 0; i < plane; i++)
        for (int c = 0; c < 3; c++) out.chw[static_cast<size_t>(c * plane + i)] = static_cast<float>(frame.pixels[static_cast<size_t>(i * 3 + c)]) / 255.f;
    return out;
}

namespace {
Status check_stats(const std::vector<float> & mean, const std::vector<float> & stddev) {
    if (mean.empty() || mean.size() != stddev.size()) return invalid_argument("mean and std must be non-empty and of equal length");
    for (size_t i = 0; i < mean.size(); i++)
        if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] < 0.f) return invalid_argument("mean/std must be finite and std non-negative");
    return Status();
}
}  // namespace

Result<std::vector<float>> vla_normalize_state(const std::vector<float> & state, const std::vector<float> & mean, const std::vector<float> & stddev) {
    Status st = check_stats(mean, stddev);
    if (!st.is_ok()) return st;
    if (state.size() != mean.size()) return invalid_argument("the state has " + std::to_string(state.size()) + " values but the statistics have " + std::to_string(mean.size()));
    std::vector<float> out(state.size());
    for (size_t i = 0; i < state.size(); i++) {
        if (!std::isfinite(state[i])) return invalid_argument("the state has non-finite values");
        out[i] = (state[i] - mean[i]) / (stddev[i] + 1e-8f);
    }
    return out;
}

Result<std::vector<float>> vla_unnormalize_actions(const std::vector<float> & actions, int64_t action_dim, const std::vector<float> & mean, const std::vector<float> & stddev) {
    Status st = check_stats(mean, stddev);
    if (!st.is_ok()) return st;
    const int64_t d = static_cast<int64_t>(mean.size());
    if (action_dim < d || actions.empty() || static_cast<int64_t>(actions.size()) % action_dim != 0) return invalid_argument("actions must be whole rows of action_dim values, with action_dim >= the statistics' length");
    const int64_t rows = static_cast<int64_t>(actions.size()) / action_dim;
    std::vector<float> out(static_cast<size_t>(rows * d));
    for (int64_t r = 0; r < rows; r++)
        for (int64_t j = 0; j < d; j++) out[static_cast<size_t>(r * d + j)] = actions[static_cast<size_t>(r * action_dim + j)] * stddev[static_cast<size_t>(j)] + mean[static_cast<size_t>(j)];
    return out;
}

struct VlaPipeline::Impl {
    std::unique_ptr<CoreMlModel> vision, prefix, expert;
    GgufFile host;
    int64_t hidden = 0, head_dim = 0, kv_heads = 0, layers = 0, chunk = 0, action_dim = 0, state_dim = 0, steps = 0, expert_hidden = 0, max_tokens = 0, image_size = 0, cameras = 0, per_image = 0, prefix_len = 0;
    double embed_scale = 0, min_period = 0, max_period = 0, max_wavelength = 0;
    const uint16_t * embed = nullptr;  // [vocab, hidden] bf16 inside the mapped host file
    int64_t vocab = 0;
    const float * state_w = nullptr;  // [hidden, state_dim]
    const float * state_b = nullptr;
    uint64_t seed = 0, calls = 0;
    // working buffers (allocated once)
    std::vector<float> camera_input, camera_embed, prefix_embeds, prefix_cos, prefix_sin, prefix_bias;
    std::vector<uint16_t> kv;  // the prefix key/value cache in half precision, written by the prefix model and read by the expert model without conversion
    std::vector<float> actions, final_actions, time_embs, self_cos, self_sin, cross_cos, cross_sin, step_bias;
    explicit Impl(GgufFile h) : host(std::move(h)) {}
};

VlaPipeline::VlaPipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VlaPipeline::VlaPipeline(VlaPipeline &&) noexcept = default;
VlaPipeline & VlaPipeline::operator=(VlaPipeline &&) noexcept = default;
VlaPipeline::~VlaPipeline() = default;

int64_t VlaPipeline::n_cameras() const { return impl_->cameras; }
int64_t VlaPipeline::tokenizer_max_length() const { return impl_->max_tokens; }
int64_t VlaPipeline::state_dim() const { return impl_->state_dim; }
int64_t VlaPipeline::chunk_size() const { return impl_->chunk; }
int64_t VlaPipeline::action_dim() const { return impl_->action_dim; }
int64_t VlaPipeline::prefix_length() const { return impl_->prefix_len; }
int64_t VlaPipeline::steps() const { return impl_->steps; }

Result<VlaPipeline> VlaPipeline::create(const VlaPaths & paths, const VlaOptions & options) {
    auto host = GgufFile::open(paths.host);
    if (!host.is_ok()) return host.status();
    auto arch = host.value().get_string("general.architecture");
    if (!arch.is_ok()) return arch.status();
    if (arch.value() != "vla_config") return invalid_argument("'" + paths.host + "' is not a SmolVLA host file (architecture '" + arch.value() + "')");
    auto impl = std::make_unique<Impl>(std::move(host).value());
    Impl & m = *impl;
    const GgufFile & h = m.host;
    struct U { const char * key; int64_t * dst; int64_t lo, hi; };
    for (const U & u : {U{"hidden", &m.hidden, 1, 65536}, U{"head_dim", &m.head_dim, 2, 1024}, U{"kv_heads", &m.kv_heads, 1, 1024}, U{"layers", &m.layers, 1, 256}, U{"chunk", &m.chunk, 1, 4096},
                        U{"action_dim", &m.action_dim, 1, 4096}, U{"state_dim", &m.state_dim, 1, 4096}, U{"steps", &m.steps, 1, 1000}, U{"expert_hidden", &m.expert_hidden, 2, 65536},
                        U{"tokenizer_max_length", &m.max_tokens, 1, 4096}, U{"image_size", &m.image_size, 1, 8192}, U{"n_cameras", &m.cameras, 1, 16}, U{"tokens_per_image", &m.per_image, 1, 4096}}) {
        auto v = h.get_uint(std::string("vla_config.") + u.key);
        if (!v.is_ok()) return v.status();
        if (static_cast<int64_t>(v.value()) < u.lo || static_cast<int64_t>(v.value()) > u.hi) return data_loss(std::string("vla_config.") + u.key + " is outside its valid range");
        *u.dst = static_cast<int64_t>(v.value());
    }
    if (m.head_dim % 2 != 0 || m.expert_hidden % 2 != 0) return data_loss("head_dim and expert_hidden must be even");
    struct F { const char * key; double * dst; };
    for (const F & f : {F{"embed_scale", &m.embed_scale}, F{"min_period", &m.min_period}, F{"max_period", &m.max_period}, F{"max_wavelength", &m.max_wavelength}}) {
        auto v = h.get_float(std::string("vla_config.") + f.key);
        if (!v.is_ok()) return v.status();
        if (!std::isfinite(v.value()) || !(v.value() > 0.0)) return data_loss(std::string("vla_config.") + f.key + " must be positive and finite");
        *f.dst = v.value();
    }
    if (!(m.max_period > m.min_period)) return data_loss("max_period must exceed min_period");
    m.prefix_len = m.cameras * m.per_image + m.max_tokens + 1;
    m.seed = options.seed;

    auto emb = h.find_tensor("embed_tokens");
    auto sw = h.find_tensor("state_proj.weight");
    auto sb = h.find_tensor("state_proj.bias");
    if (!emb.is_ok()) return emb.status();
    if (!sw.is_ok()) return sw.status();
    if (!sb.is_ok()) return sb.status();
    if (emb.value()->dtype != DType::BF16 || emb.value()->shape.rank() != 2 || emb.value()->shape.dim(1) != m.hidden) return data_loss("embed_tokens must be a bf16 [vocab, hidden] tensor");
    if (sw.value()->dtype != DType::F32 || sw.value()->shape.rank() != 2 || sw.value()->shape.dim(0) != m.hidden || sw.value()->shape.dim(1) != m.state_dim) return data_loss("state_proj.weight must be f32 [hidden, state_dim]");
    if (sb.value()->dtype != DType::F32 || sb.value()->shape.numel() != m.hidden) return data_loss("state_proj.bias must be f32 [hidden]");
    m.vocab = emb.value()->shape.dim(0);
    m.embed = reinterpret_cast<const uint16_t *>(h.tensor_data(*emb.value()));
    m.state_w = reinterpret_cast<const float *>(h.tensor_data(*sw.value()));
    m.state_b = reinterpret_cast<const float *>(h.tensor_data(*sb.value()));

    auto load = [&](const std::string & path, const std::string & units_name, std::unique_ptr<CoreMlModel> * dst) -> Status {
        auto units = parse_coreml_units(units_name);
        if (!units.is_ok()) return units.status();
        auto model = CoreMlModel::load(path, units.value());
        if (!model.is_ok()) return model.status();
        *dst = std::make_unique<CoreMlModel>(std::move(model).value());
        return Status();
    };
    Status st = load(paths.vision, options.vision_units, &m.vision);
    if (st.is_ok()) st = load(paths.prefix, options.prefix_units, &m.prefix);
    if (st.is_ok()) st = load(paths.expert, options.expert_units, &m.expert);
    if (!st.is_ok()) return st;

    auto expect = [](const std::vector<CoreMlTensorInfo> & infos, const std::string & name, std::initializer_list<int64_t> dims) -> bool {
        auto s = Shape::create(dims);
        if (!s.is_ok()) return false;
        for (const auto & i : infos) if (i.name == name) return i.shape == s.value();
        return false;
    };
    const int64_t P = m.prefix_len;
    if (m.vision->inputs().size() != 1 || !expect(m.vision->inputs(), "pixel_values", {1, 3, m.image_size, m.image_size}) || m.vision->outputs().size() != 1 ||
        !expect(m.vision->outputs(), "image_embeddings", {1, m.per_image, m.hidden})) {
        return invalid_argument("the vision package must map pixel_values [1,3," + std::to_string(m.image_size) + "," + std::to_string(m.image_size) + "] to image_embeddings [1," + std::to_string(m.per_image) + "," + std::to_string(m.hidden) + "]");
    }
    const std::string kvs = "[1," + std::to_string(P) + "," + std::to_string(m.kv_heads) + "," + std::to_string(m.head_dim) + "]";
    bool ok = m.prefix->inputs().size() == 4 && expect(m.prefix->inputs(), "embeds", {1, P, m.hidden}) && expect(m.prefix->inputs(), "rope_cos", {1, P, 1, m.head_dim / 2}) &&
              expect(m.prefix->inputs(), "rope_sin", {1, P, 1, m.head_dim / 2}) && expect(m.prefix->inputs(), "mask_bias", {1, 1, P, P}) &&
              static_cast<int64_t>(m.prefix->outputs().size()) == 2 * m.layers;
    for (int64_t i = 0; ok && i < m.layers; i++) ok = expect(m.prefix->outputs(), "kv_k_" + std::to_string(i), {1, P, m.kv_heads, m.head_dim}) && expect(m.prefix->outputs(), "kv_v_" + std::to_string(i), {1, P, m.kv_heads, m.head_dim});
    if (!ok) return invalid_argument("the prefix package does not match the host file: expected inputs embeds/rope_cos/rope_sin/mask_bias for a prefix of " + std::to_string(P) + " tokens and " + std::to_string(2 * m.layers) + " outputs kv_k_i/kv_v_i of shape " + kvs);
    ok = static_cast<int64_t>(m.expert->inputs().size()) == 7 + 2 * m.layers && expect(m.expert->inputs(), "actions", {1, m.chunk, m.action_dim}) && expect(m.expert->inputs(), "time_embs", {m.steps, 1, 1, m.expert_hidden}) &&
         expect(m.expert->inputs(), "self_cos", {1, m.chunk, 1, m.head_dim / 2}) && expect(m.expert->inputs(), "self_sin", {1, m.chunk, 1, m.head_dim / 2}) &&
         expect(m.expert->inputs(), "cross_cos", {1, m.chunk, 1, m.head_dim / 2}) && expect(m.expert->inputs(), "cross_sin", {1, m.chunk, 1, m.head_dim / 2}) &&
         expect(m.expert->inputs(), "mask_bias", {1, 1, m.chunk, P + m.chunk}) && m.expert->outputs().size() == 1 && expect(m.expert->outputs(), "final_actions", {1, m.chunk, m.action_dim});
    for (int64_t i = 0; ok && i < m.layers; i++) ok = expect(m.expert->inputs(), "kv_k_" + std::to_string(i), {1, P, m.kv_heads, m.head_dim}) && expect(m.expert->inputs(), "kv_v_" + std::to_string(i), {1, P, m.kv_heads, m.head_dim});
    if (!ok) return invalid_argument("the expert package does not match the host file and the prefix package (it must run all " + std::to_string(m.steps) + " flow steps: time_embs [" + std::to_string(m.steps) + ",1,1," + std::to_string(m.expert_hidden) + "] -> final_actions)");

    const size_t plane = static_cast<size_t>(3 * m.image_size * m.image_size);
    m.camera_input.resize(plane);
    m.camera_embed.resize(static_cast<size_t>(m.per_image * m.hidden));
    m.prefix_embeds.resize(static_cast<size_t>(P * m.hidden));
    m.prefix_cos.resize(static_cast<size_t>(P * m.head_dim / 2));
    m.prefix_sin.resize(m.prefix_cos.size());
    m.prefix_bias.resize(static_cast<size_t>(P * P));
    m.kv.resize(static_cast<size_t>(2 * m.layers * P * m.kv_heads * m.head_dim));
    m.actions.resize(static_cast<size_t>(m.chunk * m.action_dim));
    m.final_actions.resize(m.actions.size());
    m.time_embs.resize(static_cast<size_t>(m.steps * m.expert_hidden));
    m.self_cos.resize(static_cast<size_t>(m.chunk * m.head_dim / 2));
    m.self_sin.resize(m.self_cos.size());
    m.cross_cos.resize(m.self_cos.size());
    m.cross_sin.resize(m.self_cos.size());
    m.step_bias.resize(static_cast<size_t>(m.chunk * (P + m.chunk)));
    return VlaPipeline(std::move(impl));
}

Result<std::vector<float>> VlaPipeline::infer(const VlaObservation & obs, const std::vector<float> * noise, VlaTimings * timings) {
    Impl & m = *impl_;
    const auto t_start = Clock::now();
    if (static_cast<int64_t>(obs.cameras.size()) != m.cameras) return invalid_argument("expected " + std::to_string(m.cameras) + " camera images, got " + std::to_string(obs.cameras.size()));
    if (static_cast<int64_t>(obs.tokens.size()) != m.max_tokens || obs.token_mask.size() != obs.tokens.size()) return invalid_argument("tokens and token_mask must both have " + std::to_string(m.max_tokens) + " entries");
    if (static_cast<int64_t>(obs.state.size()) > m.state_dim) return invalid_argument("the state has more than " + std::to_string(m.state_dim) + " values");
    for (float v : obs.state) if (!std::isfinite(v)) return invalid_argument("the state has non-finite values");
    for (int32_t t : obs.tokens) if (t < 0 || t >= m.vocab) return invalid_argument("a token id is outside the vocabulary");
    if (noise && noise->size() != m.actions.size()) return invalid_argument("noise must have chunk * action_dim = " + std::to_string(m.actions.size()) + " values");
    if (noise) for (float v : *noise) if (!std::isfinite(v)) return invalid_argument("noise has non-finite values");
    const int64_t P = m.prefix_len, H = m.hidden;

    // 1. cameras -> image embeddings, straight into the prefix buffer (scaled by sqrt(hidden) like the policy)
    double vision_ms = 0;
    for (int64_t c = 0; c < m.cameras; c++) {
        Status st = vla_prepare_camera(obs.cameras[static_cast<size_t>(c)], m.image_size, m.camera_input.data());
        if (!st.is_ok()) return st;
        const auto tv = Clock::now();
        auto in = view(m.camera_input.data(), {1, 3, m.image_size, m.image_size});
        auto out = view(m.camera_embed.data(), {1, m.per_image, H});
        if (!in.is_ok() || !out.is_ok()) return internal_error("could not bind the vision buffers");
        st = m.vision->predict({{"pixel_values", in.value()}}, {{"image_embeddings", out.value()}});
        if (!st.is_ok()) return st;
        vision_ms += ms_since(tv);
        float * dst = m.prefix_embeds.data() + c * m.per_image * H;
        for (int64_t i = 0; i < m.per_image * H; i++) dst[i] = m.camera_embed[static_cast<size_t>(i)] * static_cast<float>(m.embed_scale);
    }
    const auto t_host = Clock::now();
    // 2. language embeddings (bf16 table) and the state token
    float * lang = m.prefix_embeds.data() + m.cameras * m.per_image * H;
    for (int64_t t = 0; t < m.max_tokens; t++) {
        const uint16_t * row = m.embed + static_cast<int64_t>(obs.tokens[static_cast<size_t>(t)]) * H;
        for (int64_t i = 0; i < H; i++) lang[t * H + i] = bf16_to_float(row[i]) * static_cast<float>(m.embed_scale);
    }
    std::vector<float> state(static_cast<size_t>(m.state_dim), 0.f);
    std::copy(obs.state.begin(), obs.state.end(), state.begin());
    float * state_tok = m.prefix_embeds.data() + (P - 1) * H;
    for (int64_t i = 0; i < H; i++) {
        float acc = m.state_b[i];
        for (int64_t j = 0; j < m.state_dim; j++) acc += m.state_w[i * m.state_dim + j] * state[static_cast<size_t>(j)];
        state_tok[i] = acc;
    }
    // 3. masks and positions of the prefix: images and prompt see each other (not the padding, not the state); the state token sees everything valid
    std::vector<uint8_t> pad(static_cast<size_t>(P), 1), att(static_cast<size_t>(P), 0);
    for (int64_t t = 0; t < m.max_tokens; t++) pad[static_cast<size_t>(m.cameras * m.per_image + t)] = obs.token_mask[static_cast<size_t>(t)] ? 1 : 0;
    att[static_cast<size_t>(P - 1)] = 1;
    std::vector<int64_t> cum(static_cast<size_t>(P)), pos(static_cast<size_t>(P));
    int64_t running_att = 0, running_pad = 0;
    for (int64_t i = 0; i < P; i++) {
        running_att += att[static_cast<size_t>(i)];
        running_pad += pad[static_cast<size_t>(i)];
        cum[static_cast<size_t>(i)] = running_att;
        pos[static_cast<size_t>(i)] = running_pad - 1;
    }
    for (int64_t i = 0; i < P; i++)
        for (int64_t j = 0; j < P; j++) {
            const bool allowed = cum[static_cast<size_t>(j)] <= cum[static_cast<size_t>(i)] && pad[static_cast<size_t>(i)] && pad[static_cast<size_t>(j)];
            m.prefix_bias[static_cast<size_t>(i * P + j)] = allowed ? 0.f : kMaskedBias;
        }
    rope_tables(pos, m.head_dim, m.max_wavelength, m.prefix_cos.data(), m.prefix_sin.data());
    const int64_t valid = running_pad;
    const double host_a = ms_since(t_host);

    // 4. prefix through the VLM layers -> key/value cache (the buffers the expert reads directly)
    const auto t_prefix = Clock::now();
    {
        auto e = view(m.prefix_embeds.data(), {1, P, H});
        auto c = view(m.prefix_cos.data(), {1, P, 1, m.head_dim / 2});
        auto s = view(m.prefix_sin.data(), {1, P, 1, m.head_dim / 2});
        auto b = view(m.prefix_bias.data(), {1, 1, P, P});
        if (!e.is_ok() || !c.is_ok() || !s.is_ok() || !b.is_ok()) return internal_error("could not bind the prefix inputs");
        std::vector<NamedTensor> outs;
        const size_t per = static_cast<size_t>(P * m.kv_heads * m.head_dim);
        for (int64_t i = 0; i < m.layers; i++) {
            for (int kv = 0; kv < 2; kv++) {
                auto v = view(m.kv.data() + static_cast<size_t>(2 * i + kv) * per, {1, P, m.kv_heads, m.head_dim}, DType::F16);
                if (!v.is_ok()) return internal_error("could not bind a cache buffer");
                outs.push_back({std::string(kv == 0 ? "kv_k_" : "kv_v_") + std::to_string(i), v.value()});
            }
        }
        Status st = m.prefix->predict({{"embeds", e.value()}, {"rope_cos", c.value()}, {"rope_sin", s.value()}, {"mask_bias", b.value()}}, outs);
        if (!st.is_ok()) return st;
    }
    const double prefix_ms = ms_since(t_prefix);

    // 5. flow matching: tables that do not change between steps, then `steps` expert passes
    const auto t_expert = Clock::now();
    std::vector<int64_t> self_pos(static_cast<size_t>(m.chunk)), cross_pos(static_cast<size_t>(m.chunk));
    for (int64_t i = 0; i < m.chunk; i++) { self_pos[static_cast<size_t>(i)] = valid + i; cross_pos[static_cast<size_t>(i)] = i; }  // cross-attention queries restart at position 0
    rope_tables(self_pos, m.head_dim, m.max_wavelength, m.self_cos.data(), m.self_sin.data());
    rope_tables(cross_pos, m.head_dim, m.max_wavelength, m.cross_cos.data(), m.cross_sin.data());
    for (int64_t i = 0; i < m.chunk; i++)
        for (int64_t j = 0; j < P + m.chunk; j++) {
            const bool allowed = j < P ? pad[static_cast<size_t>(j)] != 0 : (j - P) <= i;  // the action tokens are causal among themselves
            m.step_bias[static_cast<size_t>(i * (P + m.chunk) + j)] = allowed ? 0.f : kMaskedBias;
        }
    if (noise) {
        m.actions = *noise;
    } else {
        std::mt19937_64 rng(m.seed + m.calls);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (float & v : m.actions) v = normal(rng);
    }
    m.calls++;
    const double dt = -1.0 / static_cast<double>(m.steps);
    const size_t per = static_cast<size_t>(P * m.kv_heads * m.head_dim);
    for (int64_t s = 0; s < m.steps; s++) time_embedding(1.0 + static_cast<double>(s) * dt, m.expert_hidden, m.min_period, m.max_period, m.time_embs.data() + s * m.expert_hidden);
    std::vector<NamedTensor> ins;
    auto push = [&](const std::string & name, void * data, std::initializer_list<int64_t> dims, DType type = DType::F32) -> Status {
        auto v = view(data, dims, type);
        if (!v.is_ok()) return v.status();
        ins.push_back({name, v.value()});
        return Status();
    };
    Status st = push("actions", m.actions.data(), {1, m.chunk, m.action_dim});
    if (st.is_ok()) st = push("time_embs", m.time_embs.data(), {m.steps, 1, 1, m.expert_hidden});
    if (st.is_ok()) st = push("self_cos", m.self_cos.data(), {1, m.chunk, 1, m.head_dim / 2});
    if (st.is_ok()) st = push("self_sin", m.self_sin.data(), {1, m.chunk, 1, m.head_dim / 2});
    if (st.is_ok()) st = push("cross_cos", m.cross_cos.data(), {1, m.chunk, 1, m.head_dim / 2});
    if (st.is_ok()) st = push("cross_sin", m.cross_sin.data(), {1, m.chunk, 1, m.head_dim / 2});
    if (st.is_ok()) st = push("mask_bias", m.step_bias.data(), {1, 1, m.chunk, P + m.chunk});
    for (int64_t i = 0; st.is_ok() && i < m.layers; i++)
        for (int kv = 0; st.is_ok() && kv < 2; kv++) st = push(std::string(kv == 0 ? "kv_k_" : "kv_v_") + std::to_string(i), m.kv.data() + static_cast<size_t>(2 * i + kv) * per, {1, P, m.kv_heads, m.head_dim}, DType::F16);
    if (!st.is_ok()) return st;
    auto out = view(m.final_actions.data(), {1, m.chunk, m.action_dim});
    if (!out.is_ok()) return out.status();
    st = m.expert->predict(ins, {{"final_actions", out.value()}});
    if (!st.is_ok()) return st;
    const double expert_ms = ms_since(t_expert);
    for (float v : m.final_actions) if (!std::isfinite(v)) return internal_error("the action chunk has non-finite values");
    if (timings) { timings->vision_ms = vision_ms; timings->host_ms = host_a; timings->prefix_ms = prefix_ms; timings->expert_ms = expert_ms; timings->total_ms = ms_since(t_start); }
    return m.final_actions;
}

}  // namespace dooms
