// Tests the native Metal backend by running IR graphs and comparing against independent double-precision references written here (not the code under test). The
// convolution multiplies half-rounded operands (like ggml's Metal matmul), so its reference rounds the inputs and weights to half first; a separate check bounds the
// distance to the unrounded result.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "backend/ggml/ggml_backend.h"
#include "backend/metal/metal_backend.h"
#include "check.h"

using namespace dooms;
using Vec = std::vector<float>;

static Shape S(std::initializer_list<int64_t> d) { return Shape::create(d).value(); }

static Vec rnd(size_t n, uint32_t seed, float scale = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    Vec v(n);
    for (float & x : v) x = dist(gen);
    return v;
}

static float half_round(float v) { return static_cast<float>(static_cast<__fp16>(v)); }

static double rel_err(const Vec & got, const std::vector<double> & want) {
    if (got.size() != want.size()) return 1e30;
    double d = 0, w = 0;
    for (size_t i = 0; i < got.size(); i++) { d += (got[i] - want[i]) * (got[i] - want[i]); w += want[i] * want[i]; }
    return std::sqrt(d / std::max(w, 1e-30));
}

static std::unique_ptr<Backend> metal(bool fuse = true) {
    MetalBackendOptions o;
    o.fuse_epilogue = fuse;
    auto b = create_metal_backend(o);
    if (!b.is_ok()) { std::fprintf(stderr, "no Metal backend: %s\n", b.status().to_string().c_str()); std::exit(1); }
    return std::move(b).value();
}

// Runs a graph with one output "y": inputs and weights by name.
static Vec run_graph(Backend & be, const Graph & g, const std::vector<std::pair<std::string, Vec>> & inputs, const std::vector<std::pair<std::string, Vec>> & weights) {
    auto exe = be.compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return {};
    std::vector<Vec> keep;
    for (const auto & w : weights) {
        const Value & v = g.values()[static_cast<size_t>(g.find_named(w.first).value())];
        Vec data = w.second;
        EXPECT(exe.value()->set_weight(w.first, TensorView::create(data.data(), v.type.shape, DType::F32).value()).is_ok());
    }
    std::vector<NamedTensor> in, out;
    std::vector<Vec> in_data;
    for (const auto & i : inputs) in_data.push_back(i.second);
    for (size_t k = 0; k < inputs.size(); k++) {
        const Value & v = g.values()[static_cast<size_t>(g.find_named(inputs[k].first).value())];
        in.push_back({inputs[k].first, TensorView::create(in_data[k].data(), v.type.shape, DType::F32).value()});
    }
    const Shape os = g.values()[static_cast<size_t>(g.outputs()[0].value)].type.shape;
    Vec y(static_cast<size_t>(os.numel()), -777.f);
    out.push_back({"y", TensorView::create(y.data(), os, DType::F32).value()});
    Status st = exe.value()->run(in, out);
    EXPECT(st.is_ok());
    if (!st.is_ok()) std::fprintf(stderr, "  %s\n", st.to_string().c_str());
    return y;
}

struct ConvCase { int64_t n, ic, h, w, oc, kh, kw, stride, pad, dil; };

// reference convolution in double on half-rounded operands
static std::vector<double> conv_ref(const ConvCase & c, const Vec & x, const Vec & w, bool round, int64_t * oh_out, int64_t * ow_out) {
    const int64_t oh = (c.h + 2 * c.pad - c.dil * (c.kh - 1) - 1) / c.stride + 1, ow = (c.w + 2 * c.pad - c.dil * (c.kw - 1) - 1) / c.stride + 1;
    *oh_out = oh; *ow_out = ow;
    std::vector<double> y(static_cast<size_t>(c.n * c.oc * oh * ow), 0.0);
    for (int64_t n = 0; n < c.n; n++)
        for (int64_t oc = 0; oc < c.oc; oc++)
            for (int64_t oy = 0; oy < oh; oy++)
                for (int64_t ox = 0; ox < ow; ox++) {
                    double acc = 0;
                    for (int64_t ic = 0; ic < c.ic; ic++)
                        for (int64_t ky = 0; ky < c.kh; ky++)
                            for (int64_t kx = 0; kx < c.kw; kx++) {
                                const int64_t iy = oy * c.stride - c.pad + ky * c.dil, ix = ox * c.stride - c.pad + kx * c.dil;
                                if (iy < 0 || iy >= c.h || ix < 0 || ix >= c.w) continue;
                                float xv = x[static_cast<size_t>(((n * c.ic + ic) * c.h + iy) * c.w + ix)], wv = w[static_cast<size_t>(((oc * c.ic + ic) * c.kh + ky) * c.kw + kx)];
                                if (round) { xv = half_round(xv); wv = half_round(wv); }
                                acc += static_cast<double>(xv) * wv;
                            }
                    y[static_cast<size_t>(((n * c.oc + oc) * oh + oy) * ow + ox)] = acc;
                }
    return y;
}

static Graph conv_graph(const ConvCase & c, const char * tail, int64_t * oh, int64_t * ow) {
    Graph g;
    ValueId x = g.add_input("x", S({c.n, c.ic, c.h, c.w}), DType::F32).value();
    ValueId w = g.add_weight("w", S({c.oc, c.ic, c.kh, c.kw}), DType::F32).value();
    Conv2dParams p;
    p.stride_h = p.stride_w = c.stride; p.pad_h = p.pad_w = c.pad; p.dilation_h = p.dilation_w = c.dil;
    ValueId y = g.add_conv2d(x, w, p).value();
    if (std::string(tail) != "none") {
        ValueId b = g.add_weight("b", S({c.oc}), DType::F32).value();
        y = g.add_add(y, g.add_reshape(b, {c.oc, 1, 1}).value()).value();
        if (std::string(tail) == "silu") y = g.add_silu(y).value();
        if (std::string(tail) == "relu") y = g.add_relu(y).value();
    }
    EXPECT(g.mark_output("y", y).is_ok());
    const Shape os = g.values()[static_cast<size_t>(y)].type.shape;
    *oh = os.dim(2); *ow = os.dim(3);
    return g;
}

TEST(conv2d_matches_a_double_precision_reference) {
    auto be = metal();
    const ConvCase cases[] = {
        {1, 3, 17, 19, 5, 3, 3, 1, 1, 1},    // odd sizes, fewer than one tile everywhere
        {2, 8, 16, 16, 70, 3, 3, 2, 1, 1},   // batch 2, stride 2, more than 64 output channels
        {1, 70, 9, 9, 64, 1, 1, 1, 0, 1},    // pointwise, reduction length 70 (not a multiple of 16)
        {1, 4, 12, 12, 8, 3, 3, 1, 2, 2},    // dilation 2
        {1, 3, 64, 64, 16, 3, 3, 2, 1, 1},   // several pixel tiles
        {1, 16, 7, 5, 9, 5, 5, 1, 2, 1},     // 5x5 kernel
        {1, 1, 1, 1, 1, 1, 1, 1, 0, 1},      // the smallest convolution
        {1, 64, 8, 8, 16, 3, 3, 1, 1, 1},    // long reduction, one output tile: split over several threadgroups
        {2, 128, 6, 6, 70, 3, 3, 1, 1, 1},   // split, batch 2, two channel tiles
        {1, 600, 3, 3, 8, 1, 1, 1, 0, 1},    // split with a reduction length that is not a multiple of the slice
    };
    for (const ConvCase & c : cases) {
        int64_t oh, ow, oh2, ow2;
        Graph g = conv_graph(c, "none", &oh, &ow);
        Vec x = rnd(static_cast<size_t>(c.n * c.ic * c.h * c.w), 11), w = rnd(static_cast<size_t>(c.oc * c.ic * c.kh * c.kw), 12, 0.5f);
        Vec y = run_graph(*be, g, {{"x", x}}, {{"w", w}});
        auto want = conv_ref(c, x, w, true, &oh2, &ow2);
        const double e = rel_err(y, want);
        auto exact = conv_ref(c, x, w, false, &oh2, &ow2);
        EXPECT(e < 1e-5);                       // same operands, only the summation order differs
        EXPECT(rel_err(y, exact) < 5e-3);       // half-rounded operands vs the exact result
        if (e >= 1e-5) std::fprintf(stderr, "  conv case n=%lld ic=%lld hw=%lldx%lld oc=%lld k=%lldx%lld s=%lld p=%lld d=%lld: rel err %.2e\n", (long long) c.n, (long long) c.ic, (long long) c.h, (long long) c.w, (long long) c.oc, (long long) c.kh, (long long) c.kw, (long long) c.stride, (long long) c.pad, (long long) c.dil, e);
    }
}

TEST(fused_epilogue_equals_separate_kernels_and_the_reference) {
    const ConvCase fused_cases[] = {{1, 12, 20, 20, 33, 3, 3, 1, 1, 1}, {1, 96, 7, 7, 40, 3, 3, 1, 1, 1}};  // the second one is split over the reduction
    for (const ConvCase & c : fused_cases) {
    for (const char * tail : {"bias", "silu", "relu"}) {
        int64_t oh, ow, oh2, ow2;
        Graph g = conv_graph(c, tail, &oh, &ow);
        Vec x = rnd(static_cast<size_t>(c.n * c.ic * c.h * c.w), 21), w = rnd(static_cast<size_t>(c.oc * c.ic * c.kh * c.kw), 22, 0.5f), b = rnd(static_cast<size_t>(c.oc), 23);
        auto fused = metal(true), plain = metal(false);
        Vec yf = run_graph(*fused, g, {{"x", x}}, {{"w", w}, {"b", b}});
        Vec yp = run_graph(*plain, g, {{"x", x}}, {{"w", w}, {"b", b}});
        auto ref = conv_ref(c, x, w, true, &oh2, &ow2);
        const int64_t ohw = oh * ow;
        for (size_t i = 0; i < ref.size(); i++) {
            double v = ref[i] + b[static_cast<size_t>(static_cast<int64_t>(i) / ohw % c.oc)];
            if (std::string(tail) == "silu") v = v / (1.0 + std::exp(-v));
            if (std::string(tail) == "relu") v = std::max(v, 0.0);
            ref[i] = v;
        }
        EXPECT(rel_err(yf, ref) < 1e-5);
        EXPECT(rel_err(yp, ref) < 1e-5);
    }
    }
}

TEST(a_bias_with_another_consumer_is_not_fused_away) {
    // the conv output feeds the bias add AND a second node: fusing would lose the plain conv result
    Graph g;
    ValueId x = g.add_input("x", S({1, 2, 6, 6}), DType::F32).value();
    ValueId w = g.add_weight("w", S({4, 2, 1, 1}), DType::F32).value();
    ValueId b = g.add_weight("b", S({4}), DType::F32).value();
    ValueId c = g.add_conv2d(x, w, Conv2dParams()).value();
    ValueId biased = g.add_add(c, g.add_reshape(b, {4, 1, 1}).value()).value();
    ValueId y = g.add_add(biased, c).value();  // reads both
    EXPECT(g.mark_output("y", y).is_ok());
    Vec xv = rnd(72, 31), wv = rnd(8, 32), bv = rnd(4, 33);
    Vec got = run_graph(*metal(), g, {{"x", xv}}, {{"w", wv}, {"b", bv}});
    const ConvCase cc{1, 2, 6, 6, 4, 1, 1, 1, 0, 1};
    int64_t oh, ow;
    auto want = conv_ref(cc, xv, wv, true, &oh, &ow);
    for (size_t i = 0; i < want.size(); i++) want[i] = 2 * want[i] + bv[i / 36];
    EXPECT(rel_err(got, want) < 1e-5);
}

TEST(elementwise_ops_with_broadcast) {
    auto be = metal();
    {
        Graph g;
        ValueId a = g.add_input("a", S({2, 3, 4, 5}), DType::F32).value();
        ValueId b = g.add_input("b", S({3, 1, 5}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_add(a, b).value()).is_ok());
        Vec av = rnd(120, 41), bv = rnd(15, 42);
        Vec y = run_graph(*be, g, {{"a", av}, {"b", bv}}, {});
        std::vector<double> want(120);
        for (int n = 0; n < 2; n++) for (int c = 0; c < 3; c++) for (int h = 0; h < 4; h++) for (int w = 0; w < 5; w++) {
            const size_t i = static_cast<size_t>(((n * 3 + c) * 4 + h) * 5 + w);
            want[i] = av[i] + bv[static_cast<size_t>(c * 5 + w)];
        }
        EXPECT(rel_err(y, want) < 1e-7);
    }
    {
        Graph g;
        ValueId a = g.add_input("a", S({1, 4, 2, 2}), DType::F32).value();
        ValueId b = g.add_input("b", S({1, 4, 1, 1}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_mul(a, b).value()).is_ok());
        Vec av = rnd(16, 43), bv = rnd(4, 44);
        Vec y = run_graph(*be, g, {{"a", av}, {"b", bv}}, {});
        std::vector<double> want(16);
        for (size_t i = 0; i < 16; i++) want[i] = double(av[i]) * bv[i / 4];
        EXPECT(rel_err(y, want) < 1e-7);
    }
    for (int kind = 0; kind < 3; kind++) {
        Graph g;
        ValueId a = g.add_input("a", S({3, 7}), DType::F32).value();
        ValueId r = kind == 0 ? g.add_silu(a).value() : (kind == 1 ? g.add_relu(a).value() : g.add_scale(a, -2.5f).value());
        EXPECT(g.mark_output("y", r).is_ok());
        Vec av = rnd(21, 45, 4.f);
        Vec y = run_graph(*be, g, {{"a", av}}, {});
        std::vector<double> want(21);
        for (size_t i = 0; i < 21; i++) want[i] = kind == 0 ? av[i] / (1.0 + std::exp(-double(av[i]))) : (kind == 1 ? std::max<double>(av[i], 0) : -2.5 * av[i]);
        EXPECT(rel_err(y, want) < 1e-6);
    }
}

TEST(layout_ops_permute_slice_concat_reshape) {
    auto be = metal();
    {
        Graph g;
        ValueId a = g.add_input("a", S({2, 3, 4, 5}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_permute(a, {2, 0, 3, 1}).value()).is_ok());
        Vec av = rnd(120, 51);
        Vec y = run_graph(*be, g, {{"a", av}}, {});
        std::vector<double> want(120);
        size_t o = 0;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) for (int k = 0; k < 5; k++) for (int l = 0; l < 3; l++) want[o++] = av[static_cast<size_t>(((j * 3 + l) * 4 + i) * 5 + k)];
        EXPECT(rel_err(y, want) < 1e-9);
    }
    {
        Graph g;
        ValueId a = g.add_input("a", S({2, 6, 3}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_slice(a, 1, 2, 5).value()).is_ok());
        Vec av = rnd(36, 52);
        Vec y = run_graph(*be, g, {{"a", av}}, {});
        std::vector<double> want;
        for (int n = 0; n < 2; n++) for (int c = 2; c < 5; c++) for (int w = 0; w < 3; w++) want.push_back(av[static_cast<size_t>((n * 6 + c) * 3 + w)]);
        EXPECT(rel_err(y, want) < 1e-9);
    }
    {
        Graph g;
        ValueId a = g.add_input("a", S({2, 2, 3}), DType::F32).value();
        ValueId b = g.add_input("b", S({2, 3, 3}), DType::F32).value();
        ValueId c = g.add_input("c", S({2, 1, 3}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_concat({a, b, c}, 1).value()).is_ok());
        Vec av = rnd(12, 53), bv = rnd(18, 54), cv = rnd(6, 55);
        Vec y = run_graph(*be, g, {{"a", av}, {"b", bv}, {"c", cv}}, {});
        std::vector<double> want;
        for (int n = 0; n < 2; n++) {
            for (int i = 0; i < 6; i++) want.push_back(av[static_cast<size_t>(n * 6 + i)]);
            for (int i = 0; i < 9; i++) want.push_back(bv[static_cast<size_t>(n * 9 + i)]);
            for (int i = 0; i < 3; i++) want.push_back(cv[static_cast<size_t>(n * 3 + i)]);
        }
        EXPECT(rel_err(y, want) < 1e-9);
    }
    {  // a reshape is an alias: the chain reshape -> slice -> reshape of an input
        Graph g;
        ValueId a = g.add_input("a", S({4, 6}), DType::F32).value();
        ValueId r = g.add_reshape(a, {2, 2, 6}).value();
        ValueId s = g.add_slice(r, 0, 1, 2).value();
        EXPECT(g.mark_output("y", g.add_reshape(s, {2, 6}).value()).is_ok());
        Vec av = rnd(24, 56);
        Vec y = run_graph(*be, g, {{"a", av}}, {});
        std::vector<double> want(av.begin() + 12, av.end());
        EXPECT(rel_err(y, want) < 1e-9);
    }
}

TEST(pooling_and_upsampling) {
    auto be = metal();
    {
        Graph g;
        ValueId a = g.add_input("a", S({1, 2, 9, 7}), DType::F32).value();
        MaxPool2dParams mp;
        mp.kernel_h = mp.kernel_w = 5; mp.stride_h = mp.stride_w = 1; mp.pad_h = mp.pad_w = 2;
        EXPECT(g.mark_output("y", g.add_max_pool2d(a, mp).value()).is_ok());
        Vec av = rnd(126, 61);
        Vec y = run_graph(*be, g, {{"a", av}}, {});
        std::vector<double> want(126);
        for (int c = 0; c < 2; c++) for (int oy = 0; oy < 9; oy++) for (int ox = 0; ox < 7; ox++) {
            double m = -1e30;
            for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                const int iy = oy + dy, ix = ox + dx;
                if (iy >= 0 && iy < 9 && ix >= 0 && ix < 7) m = std::max<double>(m, av[static_cast<size_t>((c * 9 + iy) * 7 + ix)]);
            }
            want[static_cast<size_t>((c * 9 + oy) * 7 + ox)] = m;
        }
        EXPECT(rel_err(y, want) < 1e-9);
    }
    {
        Graph g;
        ValueId a = g.add_input("a", S({1, 2, 3, 4}), DType::F32).value();
        UpsampleParams up;
        up.scale_h = 2; up.scale_w = 3;
        EXPECT(g.mark_output("y", g.add_upsample_nearest2d(a, up).value()).is_ok());
        Vec av = rnd(24, 62);
        Vec y = run_graph(*be, g, {{"a", av}}, {});
        std::vector<double> want;
        for (int c = 0; c < 2; c++) for (int oy = 0; oy < 6; oy++) for (int ox = 0; ox < 12; ox++) want.push_back(av[static_cast<size_t>((c * 3 + oy / 2) * 4 + ox / 3)]);
        EXPECT(rel_err(y, want) < 1e-9);
    }
}

TEST(memory_reuse_keeps_values_alive_until_their_last_read) {
    // Large transposes are not safe to run in place (threads would overwrite elements other threads still have to read), so a pool that recycled a value's buffer
    // for the output of a kernel that still reads it, because it freed the value one plan too soon, corrupts the result.
    constexpr int C = 4, N = 64;
    const size_t total = C * N * N;
    auto reference = [&](const Vec & xv, bool chain) {
        std::vector<double> want(total);
        auto silu = [&](size_t i) { return xv[i] / (1.0 + std::exp(-double(xv[i]))); };
        for (int c = 0; c < C; c++)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    const size_t idx = static_cast<size_t>((c * N + i) * N + j), tr = static_cast<size_t>((c * N + j) * N + i);
                    double v = silu(idx);
                    if (chain) { for (int m = 0; m < 6; m++) v *= 0.9 + 0.01 * m; } else { v *= 2.0; }
                    want[idx] = v + silu(tr);
                }
        return want;
    };
    Vec xv = rnd(total, 71, 2.f);
    {   // an early value read again at the end of a long chain
        Graph g;
        ValueId x = g.add_input("x", S({1, C, N, N}), DType::F32).value();
        ValueId early = g.add_silu(x).value();
        ValueId h = early;
        for (int i = 0; i < 6; i++) h = g.add_scale(h, 0.9f + 0.01f * static_cast<float>(i)).value();
        ValueId k = g.add_permute(early, {0, 1, 3, 2}).value();
        EXPECT(g.mark_output("y", g.add_add(h, k).value()).is_ok());
        EXPECT(rel_err(run_graph(*metal(), g, {{"x", xv}}, {}), reference(xv, true)) < 1e-6);
    }
    {   // a value read by two consecutive kernels: the second (a permute) must not get the first one's buffer as its output
        Graph g;
        ValueId x = g.add_input("x", S({1, C, N, N}), DType::F32).value();
        ValueId a = g.add_silu(x).value();
        ValueId b = g.add_scale(a, 2.f).value();
        ValueId c = g.add_permute(a, {0, 1, 3, 2}).value();
        EXPECT(g.mark_output("y", g.add_add(b, c).value()).is_ok());
        EXPECT(rel_err(run_graph(*metal(), g, {{"x", xv}}, {}), reference(xv, false)) < 1e-6);
    }
}

TEST(unsupported_graphs_and_bad_bindings_are_rejected) {
    auto be = metal();
    Graph g;
    ValueId a = g.add_input("a", S({4, 4}), DType::F32).value();
    ValueId b = g.add_input("b", S({4, 4}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_matmul(a, b).value()).is_ok());
    EXPECT(be->supports(g).code() == Code::Unsupported);
    EXPECT(!be->compile(g).is_ok());

    Graph h;
    ValueId x = h.add_input("x", S({1, 1, 4, 4}), DType::F32).value();
    ValueId w = h.add_weight("w", S({2, 1, 3, 3}), DType::F32).value();
    EXPECT(h.mark_output("y", h.add_conv2d(x, w, Conv2dParams()).value()).is_ok());
    auto exe = be->compile(h);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    Vec xv(16, 1.f), yv(2 * 2 * 2);
    std::vector<NamedTensor> in = {{"x", TensorView::create(xv.data(), S({1, 1, 4, 4}), DType::F32).value()}};
    std::vector<NamedTensor> out = {{"y", TensorView::create(yv.data(), S({1, 2, 2, 2}), DType::F32).value()}};
    EXPECT(!exe.value()->run(in, out).is_ok());  // the weight was not set
    Vec wv(18, 0.5f);
    EXPECT(exe.value()->set_weight("w", TensorView::create(wv.data(), S({2, 1, 3, 3}), DType::F32).value()).is_ok());
    EXPECT(exe.value()->run(in, out).is_ok());
    Vec bad(17, 0.f);
    EXPECT(!exe.value()->set_weight("w", TensorView::create(bad.data(), S({17}), DType::F32).value()).is_ok());
    EXPECT(!exe.value()->set_weight("nope", TensorView::create(wv.data(), S({2, 1, 3, 3}), DType::F32).value()).is_ok());
    wv[3] = INFINITY;
    EXPECT(!exe.value()->set_weight("w", TensorView::create(wv.data(), S({2, 1, 3, 3}), DType::F32).value()).is_ok());
}

TEST(matches_the_ggml_backend_on_a_small_cnn) {
    // conv -> bias -> silu -> maxpool -> upsample -> concat with the input, the ops of the detector's blocks together
    Graph g;
    ValueId x = g.add_input("x", S({1, 3, 24, 24}), DType::F32).value();
    ValueId w = g.add_weight("w", S({8, 3, 3, 3}), DType::F32).value();
    ValueId b = g.add_weight("b", S({8}), DType::F32).value();
    Conv2dParams cp;
    cp.pad_h = cp.pad_w = 1;
    ValueId y = g.add_silu(g.add_add(g.add_conv2d(x, w, cp).value(), g.add_reshape(b, {8, 1, 1}).value()).value()).value();
    MaxPool2dParams mp;
    mp.kernel_h = mp.kernel_w = 2; mp.stride_h = mp.stride_w = 2;
    ValueId p = g.add_max_pool2d(y, mp).value();
    UpsampleParams up;
    up.scale_h = up.scale_w = 2;
    ValueId u = g.add_upsample_nearest2d(p, up).value();
    EXPECT(g.mark_output("y", g.add_concat({u, y}, 1).value()).is_ok());
    Vec xv = rnd(3 * 24 * 24, 81), wv = rnd(8 * 27, 82, 0.5f), bv = rnd(8, 83);
    Vec got = run_graph(*metal(), g, {{"x", xv}}, {{"w", wv}, {"b", bv}});
    GgmlBackendOptions go;
    go.device = "cpu";
    auto cpu = create_ggml_backend(go);
    EXPECT(cpu.is_ok());
    if (!cpu.is_ok()) return;
    Vec want = run_graph(*cpu.value(), g, {{"x", xv}}, {{"w", wv}, {"b", bv}});
    std::vector<double> wd(want.begin(), want.end());
    EXPECT(rel_err(got, wd) < 3e-3);  // half-rounded convolution operands against the CPU backend's f32 im2col
}

int main() { return dooms_test::run_all(); }
