// Tests the ggml backend by running IR graphs and comparing against independent
// double-precision references written here (not the code under test).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "backend/ggml/ggml_backend.h"
#include "check.h"
#include "ggml.h"

using namespace dooms;
using Vec = std::vector<float>;
using Named = std::vector<std::pair<std::string, Vec>>;

// Tolerances: f32 differs from the double reference only by accumulation order;
// f16 inputs carry ~1e-3 relative rounding, and the result is rounded to f16 again.
constexpr double kF32Tol = 1e-4;
constexpr double kF16Tol = 2e-2;
// GPU vs CPU: different accumulation order and fast-math in kernels.
constexpr double kDeviceTol = 1e-3;

static Shape S(std::initializer_list<int64_t> d) { return Shape::create(d).value(); }

static Vec rnd(size_t n, uint32_t seed, float scale = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    Vec v(n);
    for (float & x : v) x = dist(gen);
    return v;
}

static double max_diff(const Vec & a, const std::vector<double> & b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static bool close(const Vec & got, const std::vector<double> & want, double tol, const char * what) {
    const double d = max_diff(got, want);
    if (d > tol) std::fprintf(stderr, "  %s: max abs diff %g exceeds %g (sizes %zu vs %zu)\n", what, d, tol, got.size(), want.size());
    return d <= tol;
}

static std::vector<double> widen(const Vec & v) { return std::vector<double>(v.begin(), v.end()); }

// ---- f16 helpers (shared by the reduced-precision tests) ----
static std::vector<uint16_t> to_f16(const Vec & v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = ggml_fp32_to_fp16(v[i]);
    return out;
}
static Vec from_f16(const std::vector<uint16_t> & v) {
    Vec out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = ggml_fp16_to_fp32(v[i]);
    return out;
}
static Vec round_f16(const Vec & v) { return from_f16(to_f16(v)); }

// ---- references ----

static std::vector<double> matmul_ref(const Vec & a, const Vec & b, int64_t batch, int64_t M, int64_t K, int64_t N,
                                      bool trans_b, bool b_batched) {
    std::vector<double> out(batch * M * N);
    for (int64_t bi = 0; bi < batch; bi++) {
        const int64_t bb = b_batched ? bi : 0;
        for (int64_t m = 0; m < M; m++) {
            for (int64_t n = 0; n < N; n++) {
                double s = 0;
                for (int64_t k = 0; k < K; k++) {
                    const double av = a[(bi * M + m) * K + k];
                    const double bv = trans_b ? b[(bb * N + n) * K + k] : b[(bb * K + k) * N + n];
                    s += av * bv;
                }
                out[(bi * M + m) * N + n] = s;
            }
        }
    }
    return out;
}

static double gelu_ref(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

static std::vector<double> softmax_ref(const Vec & x, int64_t rows, int64_t cols) {
    std::vector<double> out(x.size());
    for (int64_t r = 0; r < rows; r++) {
        double mx = -1e300, sum = 0;
        for (int64_t c = 0; c < cols; c++) mx = std::max<double>(mx, x[r * cols + c]);
        for (int64_t c = 0; c < cols; c++) sum += std::exp(x[r * cols + c] - mx);
        for (int64_t c = 0; c < cols; c++) out[r * cols + c] = std::exp(x[r * cols + c] - mx) / sum;
    }
    return out;
}

static std::vector<double> layer_norm_ref(const Vec & x, const Vec & g, const Vec & b, int64_t rows, int64_t cols, double eps) {
    std::vector<double> out(x.size());
    for (int64_t r = 0; r < rows; r++) {
        double mean = 0, var = 0;
        for (int64_t c = 0; c < cols; c++) mean += x[r * cols + c];
        mean /= cols;
        for (int64_t c = 0; c < cols; c++) var += (x[r * cols + c] - mean) * (x[r * cols + c] - mean);
        var /= cols;
        for (int64_t c = 0; c < cols; c++) out[r * cols + c] = (x[r * cols + c] - mean) / std::sqrt(var + eps) * g[c] + b[c];
    }
    return out;
}

static std::vector<double> permute_ref(const Vec & in, const std::vector<int64_t> & dims, const std::vector<int64_t> & perm) {
    const int r = static_cast<int>(dims.size());
    std::vector<int64_t> odims(r), istride(r), idx(r, 0);
    int64_t s = 1;
    for (int i = r - 1; i >= 0; i--) { istride[i] = s; s *= dims[i]; }
    for (int i = 0; i < r; i++) odims[i] = dims[perm[i]];
    std::vector<double> out(in.size());
    for (size_t o = 0; o < out.size(); o++) {
        int64_t src = 0;
        for (int i = 0; i < r; i++) src += idx[i] * istride[perm[i]];
        out[o] = in[src];
        for (int i = r - 1; i >= 0; i--) { if (++idx[i] < odims[i]) break; idx[i] = 0; }
    }
    return out;
}

static int64_t prod(const std::vector<int64_t> & d, size_t from, size_t to) {
    int64_t p = 1;
    for (size_t i = from; i < to; i++) p *= d[i];
    return p;
}

static std::vector<double> slice_ref(const Vec & in, const std::vector<int64_t> & dims, size_t axis, int64_t start, int64_t end) {
    const int64_t outer = prod(dims, 0, axis), inner = prod(dims, axis + 1, dims.size()), len = dims[axis];
    std::vector<double> out;
    for (int64_t o = 0; o < outer; o++)
        for (int64_t j = start; j < end; j++)
            for (int64_t i = 0; i < inner; i++) out.push_back(in[(o * len + j) * inner + i]);
    return out;
}

static std::vector<double> concat_ref(const std::vector<Vec> & parts, const std::vector<std::vector<int64_t>> & dims, size_t axis) {
    const int64_t outer = prod(dims[0], 0, axis), inner = prod(dims[0], axis + 1, dims[0].size());
    std::vector<double> out;
    for (int64_t o = 0; o < outer; o++)
        for (size_t p = 0; p < parts.size(); p++) {
            const int64_t len = dims[p][axis];
            for (int64_t j = 0; j < len; j++)
                for (int64_t i = 0; i < inner; i++) out.push_back(parts[p][(o * len + j) * inner + i]);
        }
    return out;
}

// Direct convolution, NCHW x OIHW, independent of any im2col trick.
static std::vector<double> conv2d_ref(const Vec & x, const Vec & w, int64_t N, int64_t C, int64_t H, int64_t W, int64_t OC,
                                      int64_t KH, int64_t KW, const Conv2dParams & p, int64_t * out_h, int64_t * out_w) {
    const int64_t OH = (H + 2 * p.pad_h - p.dilation_h * (KH - 1) - 1) / p.stride_h + 1;
    const int64_t OW = (W + 2 * p.pad_w - p.dilation_w * (KW - 1) - 1) / p.stride_w + 1;
    *out_h = OH;
    *out_w = OW;
    std::vector<double> out(N * OC * OH * OW, 0.0);
    for (int64_t n = 0; n < N; n++)
        for (int64_t oc = 0; oc < OC; oc++)
            for (int64_t oh = 0; oh < OH; oh++)
                for (int64_t ow = 0; ow < OW; ow++) {
                    double sum = 0;
                    for (int64_t c = 0; c < C; c++)
                        for (int64_t kh = 0; kh < KH; kh++)
                            for (int64_t kw = 0; kw < KW; kw++) {
                                const int64_t ih = oh * p.stride_h - p.pad_h + kh * p.dilation_h;
                                const int64_t iw = ow * p.stride_w - p.pad_w + kw * p.dilation_w;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                sum += static_cast<double>(x[((n * C + c) * H + ih) * W + iw]) * w[((oc * C + c) * KH + kh) * KW + kw];
                            }
                    out[((n * OC + oc) * OH + oh) * OW + ow] = sum;
                }
    return out;
}

// Transposed convolution by scattering: every input pixel adds x * w into the output window it covers; padding then
// crops. x [N,IC,H,W], w [IC,OC,KH,KW] (PyTorch layout). Independent of how the backend lowers it.
static std::vector<double> conv_transpose2d_ref(const Vec & x, const Vec & w, int64_t N, int64_t IC, int64_t H, int64_t W, int64_t OC,
                                                int64_t KH, int64_t KW, const ConvTranspose2dParams & p, int64_t * out_h, int64_t * out_w) {
    const int64_t OH = (H - 1) * p.stride_h - 2 * p.pad_h + KH, OW = (W - 1) * p.stride_w - 2 * p.pad_w + KW;
    *out_h = OH;
    *out_w = OW;
    std::vector<double> out(N * OC * OH * OW, 0.0);
    for (int64_t n = 0; n < N; n++)
        for (int64_t ic = 0; ic < IC; ic++)
            for (int64_t ih = 0; ih < H; ih++)
                for (int64_t iw = 0; iw < W; iw++)
                    for (int64_t oc = 0; oc < OC; oc++)
                        for (int64_t kh = 0; kh < KH; kh++)
                            for (int64_t kw = 0; kw < KW; kw++) {
                                const int64_t oh = ih * p.stride_h - p.pad_h + kh, ow = iw * p.stride_w - p.pad_w + kw;
                                if (oh < 0 || oh >= OH || ow < 0 || ow >= OW) continue;
                                out[((n * OC + oc) * OH + oh) * OW + ow] +=
                                    static_cast<double>(x[((n * IC + ic) * H + ih) * W + iw]) * w[((ic * OC + oc) * KH + kh) * KW + kw];
                            }
    return out;
}

// Max pooling NCHW; padded positions never win.
static std::vector<double> max_pool_ref(const Vec & x, int64_t N, int64_t C, int64_t H, int64_t W, const MaxPool2dParams & p, int64_t * out_h, int64_t * out_w) {
    const int64_t OH = (H + 2 * p.pad_h - p.kernel_h) / p.stride_h + 1, OW = (W + 2 * p.pad_w - p.kernel_w) / p.stride_w + 1;
    *out_h = OH;
    *out_w = OW;
    std::vector<double> out(N * C * OH * OW);
    for (int64_t nc = 0; nc < N * C; nc++)
        for (int64_t oh = 0; oh < OH; oh++)
            for (int64_t ow = 0; ow < OW; ow++) {
                double best = -1e300;
                for (int64_t kh = 0; kh < p.kernel_h; kh++)
                    for (int64_t kw = 0; kw < p.kernel_w; kw++) {
                        const int64_t ih = oh * p.stride_h - p.pad_h + kh, iw = ow * p.stride_w - p.pad_w + kw;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        best = std::max(best, static_cast<double>(x[(nc * H + ih) * W + iw]));
                    }
                out[(nc * OH + oh) * OW + ow] = best;
            }
    return out;
}

static std::vector<double> upsample_ref(const Vec & x, int64_t NC, int64_t H, int64_t W, int64_t sh, int64_t sw) {
    std::vector<double> out(NC * H * sh * W * sw);
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t y = 0; y < H * sh; y++)
            for (int64_t xx = 0; xx < W * sw; xx++) out[(nc * H * sh + y) * W * sw + xx] = x[(nc * H + y / sh) * W + xx / sw];
    return out;
}

static double silu_ref(double x) { return x / (1.0 + std::exp(-x)); }

// ---- runner ----

static std::unique_ptr<Backend> make_backend(const char * device, bool fused_attention = true, bool kv_f16 = true) {
    GgmlBackendOptions o;
    o.device = device;
    o.fused_attention = fused_attention;
    o.attention_kv_f16 = kv_f16;
    auto b = create_ggml_backend(o);
    if (!b.is_ok()) {
        std::fprintf(stderr, "  create_ggml_backend(%s): %s\n", device, b.status().to_string().c_str());
        return nullptr;
    }
    return std::move(b).value();
}

// Whether `device` (e.g. "auto") resolves to the CPU backend on this machine. Tolerances and expectations that depend on the
// device must follow what actually runs, not the name requested: a CPU-only build (such as the sanitizer one) maps "auto" to CPU.
static bool resolves_to_cpu(const char * device) {
    auto b = make_backend(device);
    return !b || std::string(b->name()).rfind("CPU", 0) == 0;
}

// Runs an all-f32 graph with one output; returns that output.
static Result<Vec> run_f32(const Graph & g, const Named & weights, const Named & inputs, const std::string & out_name,
                           const char * device = "cpu", bool fused_attention = true, bool kv_f16 = true) {
    auto backend = make_backend(device, fused_attention, kv_f16);
    if (!backend) return internal_error("no backend");
    auto exe = backend->compile(g);
    if (!exe.is_ok()) return exe.status();
    std::vector<Vec> keep = {};  // storage must outlive the views
    keep.reserve(weights.size() + inputs.size());
    for (auto & w : weights) {
        keep.push_back(w.second);
        auto id = g.find_named(w.first);
        if (!id.is_ok()) return id.status();
        auto tv = TensorView::create(keep.back().data(), g.values()[id.value()].type.shape, DType::F32);
        if (!tv.is_ok()) return tv.status();
        DOOMS_RETURN_IF_ERROR(exe.value()->set_weight(w.first, tv.value()));
    }
    std::vector<NamedTensor> in;
    for (auto & x : inputs) {
        keep.push_back(x.second);
        auto id = g.find_named(x.first);
        if (!id.is_ok()) return id.status();
        in.push_back({x.first, TensorView::create(keep.back().data(), g.values()[id.value()].type.shape, DType::F32).value()});
    }
    ValueId out_id = kInvalidValue;
    for (auto & o : g.outputs()) if (o.name == out_name) out_id = o.value;
    if (out_id == kInvalidValue) return not_found("output " + out_name);
    const TypeInfo & ot = g.values()[out_id].type;
    Vec out(ot.shape.numel());
    std::vector<NamedTensor> outs = {{out_name, TensorView::create(out.data(), ot.shape, DType::F32).value()}};
    DOOMS_RETURN_IF_ERROR(exe.value()->run(in, outs));
    return out;
}

// ---- options ----

TEST(backend_option_validation) {
    GgmlBackendOptions bad_threads;
    bad_threads.n_threads = -1;
    EXPECT(create_ggml_backend(bad_threads).status().code() == Code::InvalidArgument);
    GgmlBackendOptions bad_dev;
    bad_dev.device = "tpu";
    EXPECT(create_ggml_backend(bad_dev).status().code() == Code::InvalidArgument);
    GgmlBackendOptions cpu;
    cpu.device = "cpu";
    cpu.n_threads = 2;
    auto b = create_ggml_backend(cpu);
    EXPECT(b.is_ok());
}

// ---- per-op numerics (CPU) ----

TEST(op_add_and_mul_with_broadcast) {
    Graph g;
    auto x = g.add_input("x", S({2, 3, 4}), DType::F32).value();
    auto bias = g.add_weight("bias", S({4}), DType::F32).value();
    auto col = g.add_weight("col", S({3, 1}), DType::F32).value();
    auto s = g.add_add(x, bias).value();
    auto p = g.add_mul(s, col).value();
    EXPECT(g.mark_output("y", p).is_ok());
    Vec xv = rnd(24, 1), bv = rnd(4, 2), cv = rnd(3, 3);
    auto got = run_f32(g, {{"bias", bv}, {"col", cv}}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    std::vector<double> want(24);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 4; k++) {
        const int idx = (i * 3 + j) * 4 + k;
        want[idx] = (xv[idx] + bv[k]) * cv[j];
    }
    EXPECT(close(got.value(), want, kF32Tol, "add/mul broadcast"));
}

TEST(op_matmul_plain_and_transposed) {
    for (bool tr : {false, true}) {
        Graph g;
        auto a = g.add_input("a", S({5, 7}), DType::F32).value();
        auto b = g.add_weight("b", tr ? S({3, 7}) : S({7, 3}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_matmul(a, b, tr).value()).is_ok());
        Vec av = rnd(35, 4), bv = rnd(21, 5);
        auto got = run_f32(g, {{"b", bv}}, {{"a", av}}, "y");
        EXPECT(got.is_ok());
        EXPECT(close(got.value(), matmul_ref(av, bv, 1, 5, 7, 3, tr, false), kF32Tol, tr ? "matmul transposed" : "matmul plain"));
    }
}

TEST(cpu_blas_matmuls_match_and_are_faster) {
    // Large f32 matmuls on the CPU device go to Accelerate (AMX) through the scheduler; results equal the plain CPU kernels' and the double reference, profile() keeps
    // working on that executable, and the speed shows that the Accelerate path is really taken.
    constexpr int64_t M = 256, K = 1024, N = 1024;
    Graph g;
    auto a = g.add_input("a", S({M, K}), DType::F32).value();
    auto b = g.add_weight("b", S({N, K}), DType::F32).value();
    auto h = g.add_gelu(g.add_matmul(a, b, true).value()).value();
    EXPECT(g.mark_output("y", h).is_ok());
    Vec av = rnd(static_cast<size_t>(M * K), 90, 0.1f), bv = rnd(static_cast<size_t>(N * K), 91, 0.1f);
    double ms[2] = {0, 0};
    Vec results[2];
    for (int use = 0; use < 2; use++) {
        GgmlBackendOptions o;
        o.device = "cpu";
        o.cpu_blas = use == 1;
        auto be = create_ggml_backend(o);
        EXPECT(be.is_ok());
        if (!be.is_ok()) return;
        auto exe = be.value()->compile(g);
        EXPECT(exe.is_ok());
        if (!exe.is_ok()) return;
        Vec bcopy = bv, acopy = av, y(static_cast<size_t>(M * N));
        EXPECT(exe.value()->set_weight("b", TensorView::create(bcopy.data(), S({N, K}), DType::F32).value()).is_ok());
        std::vector<NamedTensor> in = {{"a", TensorView::create(acopy.data(), S({M, K}), DType::F32).value()}};
        std::vector<NamedTensor> out = {{"y", TensorView::create(y.data(), S({M, N}), DType::F32).value()}};
        EXPECT(exe.value()->run(in, out).is_ok());  // warm-up
        std::vector<double> times;
        for (int i = 0; i < 5; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            EXPECT(exe.value()->run(in, out).is_ok());
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(times.begin(), times.end());
        ms[use] = times[times.size() / 2];
        results[use] = y;
        auto rep = exe.value()->profile(in, 2);  // the profiling pass must leave the executable runnable
        EXPECT(rep.is_ok());
        EXPECT(exe.value()->run(in, out).is_ok());
        EXPECT(std::equal(y.begin(), y.end(), results[use].begin()));  // bitwise: the same executable, the same inputs
    }
    std::vector<double> want(static_cast<size_t>(M * N));
    for (int64_t i = 0; i < M; i += 37)  // spot rows of the double reference
        for (int64_t j = 0; j < N; j++) {
            double acc = 0;
            for (int64_t k = 0; k < K; k++) acc += static_cast<double>(av[static_cast<size_t>(i * K + k)]) * bv[static_cast<size_t>(j * K + k)];
            want[static_cast<size_t>(i * N + j)] = 0.5 * acc * (1.0 + std::erf(acc / std::sqrt(2.0)));
        }
    for (int use = 0; use < 2; use++)
        for (int64_t i = 0; i < M; i += 37)
            for (int64_t j = 0; j < N; j += 17) EXPECT(std::fabs(results[use][static_cast<size_t>(i * N + j)] - want[static_cast<size_t>(i * N + j)]) < 1e-4);
    std::printf("        %lldx%lldx%lld matmul + gelu on the CPU: %.2f ms with Accelerate, %.2f ms without (%.1fx)\n", (long long) M, (long long) K, (long long) N, ms[1], ms[0], ms[0] / ms[1]);
    EXPECT(ms[1] * 1.5 < ms[0]);
}

TEST(op_matmul_weight_broadcast_over_batch) {
    for (bool tr : {false, true}) {
        Graph g;
        auto a = g.add_input("a", S({2, 4, 6}), DType::F32).value();
        auto b = g.add_weight("b", tr ? S({5, 6}) : S({6, 5}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_matmul(a, b, tr).value()).is_ok());
        Vec av = rnd(48, 6), bv = rnd(30, 7);
        auto got = run_f32(g, {{"b", bv}}, {{"a", av}}, "y");
        EXPECT(got.is_ok());
        EXPECT(close(got.value(), matmul_ref(av, bv, 2, 4, 6, 5, tr, false), kF32Tol, "matmul broadcast weight"));
    }
}

TEST(op_matmul_equal_batch_dims) {
    for (bool tr : {false, true}) {
        Graph g;
        auto a = g.add_input("a", S({2, 3, 4, 6}), DType::F32).value();
        auto b = g.add_input("b", tr ? S({2, 3, 5, 6}) : S({2, 3, 6, 5}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_matmul(a, b, tr).value()).is_ok());
        Vec av = rnd(144, 8), bv = rnd(180, 9);
        auto got = run_f32(g, {}, {{"a", av}, {"b", bv}}, "y");
        EXPECT(got.is_ok());
        EXPECT(close(got.value(), matmul_ref(av, bv, 6, 4, 6, 5, tr, true), kF32Tol, "matmul batched"));
    }
}

TEST(op_gelu_is_exact_erf) {
    Graph g;
    auto x = g.add_input("x", S({4, 8}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_gelu(x).value()).is_ok());
    Vec xv = rnd(32, 10, 4.f);
    auto got = run_f32(g, {}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    std::vector<double> want(32);
    for (int i = 0; i < 32; i++) want[i] = gelu_ref(xv[i]);
    EXPECT(close(got.value(), want, kF32Tol, "gelu"));
}

TEST(cpu_gelu_kernel_is_accurate_over_the_whole_range) {
    // The CPU device computes GELU with a NEON kernel (Abramowitz-Stegun erf, series exp); compare it with the double-precision erf over a dense scan, the far tails
    // (where exp underflows), and a length that is not a multiple of the vector width.
    const int64_t n = 100003;
    Graph g;
    auto x = g.add_input("x", S({n}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_gelu(x).value()).is_ok());
    Vec xv(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) xv[static_cast<size_t>(i)] = -12.f + 24.f * static_cast<float>(i) / static_cast<float>(n - 1);
    xv[0] = -1000.f; xv[1] = 1000.f; xv[2] = 0.f; xv[3] = -0.f; xv[4] = 1e-30f; xv[5] = -1e-30f;
    auto got = run_f32(g, {}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    if (!got.is_ok()) return;
    double worst = 0;
    for (int64_t i = 0; i < n; i++) {
        const double want = gelu_ref(xv[static_cast<size_t>(i)]);
        worst = std::max(worst, std::fabs(got.value()[static_cast<size_t>(i)] - want) / std::max(1.0, std::fabs(want)));
    }
    std::printf("        CPU GELU kernel: largest error relative to max(1, |gelu|) over %lld values: %.2e\n", (long long) n, worst);
    EXPECT(worst < 5e-7);
    EXPECT(got.value()[0] == 0.f && got.value()[1] == 1000.f);
    // a size below one vector and an exact multiple of the vector width
    for (int64_t m : {1, 3, 4, 8}) {
        Graph h;
        auto y = h.add_input("x", S({m}), DType::F32).value();
        EXPECT(h.mark_output("y", h.add_gelu(y).value()).is_ok());
        Vec v(static_cast<size_t>(m));
        for (int64_t i = 0; i < m; i++) v[static_cast<size_t>(i)] = 0.7f * static_cast<float>(i) - 1.f;
        auto r = run_f32(h, {}, {{"x", v}}, "y");
        EXPECT(r.is_ok());
        if (r.is_ok()) for (int64_t i = 0; i < m; i++) EXPECT(std::fabs(r.value()[static_cast<size_t>(i)] - gelu_ref(v[static_cast<size_t>(i)])) < 1e-6);
    }
}

TEST(op_scale) {
    for (float factor : {0.125f, -2.5f, 0.f, 1.f, 1e4f}) {
        Graph g;
        auto x = g.add_input("x", S({3, 5}), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_scale(x, factor).value()).is_ok());
        Vec xv = rnd(15, 20);
        auto got = run_f32(g, {}, {{"x", xv}}, "y");
        EXPECT(got.is_ok());
        std::vector<double> want(15);
        for (int i = 0; i < 15; i++) want[i] = static_cast<double>(xv[i]) * factor;
        EXPECT(close(got.value(), want, kF32Tol * std::max(1.0, std::fabs(static_cast<double>(factor))), "scale"));
    }
}

TEST(op_softmax_last_axis) {
    Graph g;
    auto x = g.add_input("x", S({2, 3, 9}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_softmax(x, 2).value()).is_ok());
    Vec xv = rnd(54, 11, 3.f);
    auto got = run_f32(g, {}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    EXPECT(close(got.value(), softmax_ref(xv, 6, 9), kF32Tol, "softmax"));
}

TEST(op_layer_norm) {
    Graph g;
    auto x = g.add_input("x", S({2, 5, 16}), DType::F32).value();
    auto ga = g.add_weight("g", S({16}), DType::F32).value();
    auto be = g.add_weight("b", S({16}), DType::F32).value();
    constexpr float eps = 1e-6f;
    EXPECT(g.mark_output("y", g.add_layer_norm(x, ga, be, eps).value()).is_ok());
    Vec xv = rnd(160, 12, 2.f), gv = rnd(16, 13), bv = rnd(16, 14);
    auto got = run_f32(g, {{"g", gv}, {"b", bv}}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    EXPECT(close(got.value(), layer_norm_ref(xv, gv, bv, 10, 16, eps), kF32Tol, "layer_norm"));
}

TEST(op_reshape) {
    Graph g;
    auto x = g.add_input("x", S({2, 6, 4}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_reshape(x, {2, 3, 2, 4}).value()).is_ok());
    Vec xv = rnd(48, 15);
    auto got = run_f32(g, {}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    EXPECT(close(got.value(), widen(xv), 0.0, "reshape keeps row-major order"));
}

TEST(op_permute_various) {
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> cases = {
        {{2, 5, 3, 4}, {0, 2, 1, 3}},
        {{2, 5, 3, 4}, {3, 2, 1, 0}},
        {{2, 5, 3, 4}, {1, 0, 3, 2}},
        {{3, 4, 5}, {2, 0, 1}},
        {{3, 4, 5}, {1, 2, 0}},
        {{6, 7}, {1, 0}},
        {{5, 3, 4}, {0, 1, 2}},  // identity
    };
    for (auto & c : cases) {
        const auto & dims = c.first;
        Graph g;
        auto x = g.add_input("x", Shape::create(dims.data(), static_cast<int>(dims.size())).value(), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_permute(x, c.second).value()).is_ok());
        int64_t n = 1;
        for (int64_t d : dims) n *= d;
        Vec xv = rnd(n, 16);
        auto got = run_f32(g, {}, {{"x", xv}}, "y");
        EXPECT(got.is_ok());
        EXPECT(close(got.value(), permute_ref(xv, dims, c.second), 0.0, "permute"));
    }
}

TEST(op_reshape_after_permute_handles_noncontiguous) {
    Graph g;
    auto x = g.add_input("x", S({2, 3, 4, 5}), DType::F32).value();
    auto p = g.add_permute(x, {0, 2, 1, 3}).value();
    EXPECT(g.mark_output("y", g.add_reshape(p, {2, 12, 5}).value()).is_ok());
    Vec xv = rnd(120, 17);
    auto got = run_f32(g, {}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    EXPECT(close(got.value(), permute_ref(xv, {2, 3, 4, 5}, {0, 2, 1, 3}), 0.0, "permute+reshape"));
}

TEST(chain_attention_scores_then_softmax) {
    const int64_t B = 2, T = 5, H = 3, D = 4;
    Graph g;
    auto x = g.add_input("x", S({B, T, H, D}), DType::F32).value();
    auto q = g.add_permute(x, {0, 2, 1, 3}).value();                    // [B,H,T,D]
    auto s = g.add_matmul(q, q, true).value();                          // [B,H,T,T]
    EXPECT(g.mark_output("p", g.add_softmax(s, 3).value()).is_ok());
    Vec xv = rnd(B * T * H * D, 18);
    auto got = run_f32(g, {}, {{"x", xv}}, "p");
    EXPECT(got.is_ok());
    Vec qv;
    for (double d : permute_ref(xv, {B, T, H, D}, {0, 2, 1, 3})) qv.push_back(static_cast<float>(d));
    auto scores = matmul_ref(qv, qv, B * H, T, D, T, true, true);
    Vec sv(scores.begin(), scores.end());
    EXPECT(close(got.value(), softmax_ref(sv, B * H * T, T), kF32Tol, "attention chain"));
}

// ---- low-precision (f16) weights under f32 activations ----

// Max absolute error relative to the output scale, max(1, max|want|).
static double scaled_err(const Vec & got, const std::vector<double> & want) {
    if (got.size() != want.size()) return 1e30;
    double scale = 1.0, err = 0.0;
    for (double w : want) scale = std::max(scale, std::fabs(w));
    for (size_t i = 0; i < got.size(); i++) err = std::max(err, std::fabs(got[i] - want[i]));
    return err / scale;
}

namespace {
// Measured (ggml 46baf1f, M1 Pro) max-abs/scale of f16-weight matmuls vs a reference built from the rounded weights:
// CPU ~2e-3 (ggml's CPU f16 dot product also rounds the activations to f16), GPU ~2e-4 (its tiles are half anyway).
constexpr double kF16WeightsCpuTol = 8e-3, kF16WeightsGpuTol = 1e-3;

struct F16Run {
    Vec got;
    std::vector<double> want;
    int backend_ops = 0;
};

Result<F16Run> run_f16_weight_matmul(const char * device, std::vector<int64_t> a_dims, std::vector<int64_t> w_dims, bool transpose_b) {
    Graph g;
    auto a = g.add_input("a", Shape::create(a_dims.data(), static_cast<int>(a_dims.size())).value(), DType::F32).value();
    auto w = g.add_weight("w", Shape::create(w_dims.data(), static_cast<int>(w_dims.size())).value(), DType::F16).value();
    g.mark_output("y", g.add_matmul(a, w, transpose_b).value());
    auto backend = make_backend(device);
    auto exe = backend->compile(g);
    if (!exe.is_ok()) return exe.status();

    const int64_t wn = std::accumulate(w_dims.begin(), w_dims.end(), int64_t(1), std::multiplies<int64_t>());
    const int64_t an = std::accumulate(a_dims.begin(), a_dims.end(), int64_t(1), std::multiplies<int64_t>());
    Vec av = rnd(static_cast<size_t>(an), 800), wv = rnd(static_cast<size_t>(wn), 801, 0.3f);
    std::vector<uint16_t> w16 = to_f16(wv);
    DOOMS_RETURN_IF_ERROR(exe.value()->set_weight("w", TensorView::create(w16.data(), g.values()[w].type.shape, DType::F16).value()));
    const Shape & ys = g.values()[g.outputs()[0].value].type.shape;
    F16Run r;
    r.got.resize(static_cast<size_t>(ys.numel()));
    std::vector<NamedTensor> in = {{"a", TensorView::create(av.data(), g.values()[a].type.shape, DType::F32).value()}};
    DOOMS_RETURN_IF_ERROR(exe.value()->run(in, {{"y", TensorView::create(r.got.data(), ys, DType::F32).value()}}));

    const int ra = static_cast<int>(a_dims.size());
    const int64_t K = a_dims[ra - 1], M = a_dims[ra - 2];
    const int64_t N = transpose_b ? w_dims[w_dims.size() - 2] : w_dims[w_dims.size() - 1];
    const int64_t batch = an / (M * K);
    r.want = matmul_ref(av, round_f16(wv), batch, M, K, N, transpose_b, false);   // the reference uses the *rounded* weights
    auto rep = exe.value()->profile(in, 1);
    if (!rep.is_ok()) return rep.status();
    r.backend_ops = rep.value().nodes[0].backend_ops;
    return r;
}
}  // namespace

TEST(op_matmul_f16_weights_under_f32_activations) {
    struct Case { std::vector<int64_t> a, w; bool tr; int ops; };
    const Case cases[] = {
        {{5, 7}, {3, 7}, true, 1},                 // PyTorch [out, in] layout: a single backend operation, no casts
        {{2, 4, 6}, {5, 6}, true, 1},              // weight broadcast over the batch
        {{5, 7}, {7, 3}, false, 3},                // untransposed: transpose view + cont + mul_mat, still no cast
        {{210, 1280}, {3840, 1280}, true, 1},      // the ViT-H qkv shape
    };
    for (const char * device : {"cpu", "auto"}) {
        const bool cpu = resolves_to_cpu(device);   // not `device == "cpu"`: "auto" is the CPU on a build without a GPU
        for (const Case & c : cases) {
            auto r = run_f16_weight_matmul(device, c.a, c.w, c.tr);
            EXPECT(r.is_ok());
            if (!r.is_ok()) { std::fprintf(stderr, "  %s\n", r.status().to_string().c_str()); continue; }
            const double err = scaled_err(r.value().got, r.value().want);
            const double tol = cpu ? kF16WeightsCpuTol : kF16WeightsGpuTol;
            if (err > tol) std::fprintf(stderr, "  f16 weights device=%s a=%zu-d: scaled max error %g exceeds %g\n", device, c.a.size(), err, tol);
            EXPECT(err <= tol);
            EXPECT_EQ(r.value().backend_ops, c.ops);
        }
    }
}

// ---- self attention (fused flash attention and the unfused fallback) ----

// softmax(q.k^T * scale) . v per head on a packed [B, N, 3E] qkv (q|k|v thirds, head h = h-th run of D values).
static std::vector<double> attention_ref(const Vec & qkv, int64_t B, int64_t N, int64_t H, int64_t D, double scale) {
    const int64_t E = H * D;
    std::vector<double> out(static_cast<size_t>(B * N * E), 0.0), p(static_cast<size_t>(N));
    auto at = [&](int64_t b, int64_t n, int64_t part, int64_t h, int64_t d) {
        return static_cast<double>(qkv[static_cast<size_t>((b * N + n) * 3 * E + part * E + h * D + d)]);
    };
    for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < H; h++)
            for (int64_t i = 0; i < N; i++) {
                double mx = -1e300;
                for (int64_t j = 0; j < N; j++) {
                    double s = 0;
                    for (int64_t d = 0; d < D; d++) s += at(b, i, 0, h, d) * at(b, j, 1, h, d);
                    p[static_cast<size_t>(j)] = s * scale;
                    mx = std::max(mx, p[static_cast<size_t>(j)]);
                }
                double sum = 0;
                for (int64_t j = 0; j < N; j++) { p[static_cast<size_t>(j)] = std::exp(p[static_cast<size_t>(j)] - mx); sum += p[static_cast<size_t>(j)]; }
                for (int64_t d = 0; d < D; d++) {
                    double acc = 0;
                    for (int64_t j = 0; j < N; j++) acc += p[static_cast<size_t>(j)] / sum * at(b, j, 2, h, d);
                    out[static_cast<size_t>((b * N + i) * E + h * D + d)] = acc;
                }
            }
    return out;
}

struct AttnCase { int64_t B, N, H, D; };
// Includes the real ViT-H shape (210 tokens, 16 heads of 80), batch 2, one token, and head dim 6, which
// Metal's flash attention does not support (forcing the fallback).
static const AttnCase kAttnCases[] = {{1, 5, 2, 4}, {2, 7, 3, 8}, {1, 1, 1, 16}, {1, 17, 4, 64}, {1, 9, 2, 6}, {1, 210, 16, 80}};

static Graph attention_graph(const AttnCase & c, float scale, DType dtype = DType::F32) {
    Graph g;
    auto qkv = g.add_input("qkv", S({c.B, c.N, 3 * c.H * c.D}), dtype).value();
    SelfAttentionParams p;
    p.heads = c.H;
    p.scale = scale;
    g.mark_output("y", g.add_self_attention(qkv, p).value());
    return g;
}

TEST(op_self_attention_matches_reference_on_every_device_and_path) {
    // CPU f32 is the tight reference; Metal's flash attention rounds Q/K tiles to half, like its matmul.
    constexpr double kCpuTol = 2e-5, kGpuTol = 3e-3;
    struct Config { const char * device; bool fused; bool kv_f16; double tol; };
    const Config configs[] = {{"cpu", true, true, kCpuTol},   {"cpu", false, true, kCpuTol},
                              {"auto", true, true, kGpuTol},  {"auto", true, false, kGpuTol},   // K/V as f16 (default) and as f32
                              {"auto", false, true, kGpuTol}};
    for (const AttnCase & c : kAttnCases) {
        const double scale = std::pow(static_cast<double>(c.D), -0.5);
        Graph g = attention_graph(c, static_cast<float>(scale));
        Vec qkv = rnd(static_cast<size_t>(c.B * c.N * 3 * c.H * c.D), 700 + static_cast<uint32_t>(c.N), 1.f);
        const auto want = attention_ref(qkv, c.B, c.N, c.H, c.D, scale);
        for (const Config & cfg : configs) {
            auto got = run_f32(g, {}, {{"qkv", qkv}}, "y", cfg.device, cfg.fused, cfg.kv_f16);
            EXPECT(got.is_ok());
            if (!got.is_ok()) { std::fprintf(stderr, "  %s\n", got.status().to_string().c_str()); continue; }
            const std::string what = std::string("attention B=") + std::to_string(c.B) + " N=" + std::to_string(c.N) + " H=" + std::to_string(c.H) +
                                     " D=" + std::to_string(c.D) + " device=" + cfg.device + (cfg.fused ? " fused" : " unfused") + (cfg.kv_f16 ? " kv_f16" : " kv_f32");
            EXPECT(close(got.value(), want, cfg.tol, what.c_str()));
        }
    }
}

TEST(op_self_attention_scale_semantics) {
    const AttnCase c = {1, 6, 2, 8};
    Vec qkv = rnd(static_cast<size_t>(c.B * c.N * 3 * c.H * c.D), 710, 1.f);
    for (float scale : {0.f, 0.1f, 0.5f, 3.f}) {
        auto got = run_f32(attention_graph(c, scale), {}, {{"qkv", qkv}}, "y");
        EXPECT(got.is_ok());
        if (got.is_ok()) EXPECT(close(got.value(), attention_ref(qkv, c.B, c.N, c.H, c.D, scale), 2e-5, "attention scale"));
    }
    // scale 0 means uniform attention: every output row is the mean of v over the tokens
    auto zero = run_f32(attention_graph(c, 0.f), {}, {{"qkv", qkv}}, "y");
    EXPECT(zero.is_ok());
    if (zero.is_ok()) {
        const int64_t E = c.H * c.D;
        for (int64_t d = 0; d < E; d++) {
            double mean = 0;
            for (int64_t n = 0; n < c.N; n++) mean += qkv[static_cast<size_t>(n * 3 * E + 2 * E + d)];
            mean /= static_cast<double>(c.N);
            EXPECT(std::fabs(zero.value()[static_cast<size_t>(d)] - mean) < 2e-5);
        }
    }
}

TEST(op_self_attention_fused_and_unfused_agree) {
    const AttnCase c = {2, 11, 4, 16};
    Graph g = attention_graph(c, 0.25f);
    Vec qkv = rnd(static_cast<size_t>(c.B * c.N * 3 * c.H * c.D), 720, 1.f);
    auto fused = run_f32(g, {}, {{"qkv", qkv}}, "y", "cpu", true);
    auto unfused = run_f32(g, {}, {{"qkv", qkv}}, "y", "cpu", false);
    EXPECT(fused.is_ok() && unfused.is_ok());
    if (fused.is_ok() && unfused.is_ok()) EXPECT(close(fused.value(), widen(unfused.value()), 2e-5, "fused vs unfused"));
}

TEST(op_self_attention_f16_input_keeps_dtype) {
    const AttnCase c = {1, 6, 2, 8};
    Graph g = attention_graph(c, 0.35f, DType::F16);
    Vec qkv = rnd(static_cast<size_t>(c.B * c.N * 3 * c.H * c.D), 730, 1.f);
    auto q16 = to_f16(qkv);
    std::vector<uint16_t> y16(static_cast<size_t>(c.B * c.N * c.H * c.D));
    auto backend = make_backend("cpu");
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->run({{"qkv", TensorView::create(q16.data(), S({c.B, c.N, 3 * c.H * c.D}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({c.B, c.N, c.H * c.D}), DType::F16).value()}}).is_ok());
    EXPECT(close(from_f16(y16), attention_ref(round_f16(qkv), c.B, c.N, c.H, c.D, 0.35), kF16Tol, "f16 attention"));
}

namespace {
// Number of backend operations the (single) attention node was lowered to: 5 when fused, 24 when unfused.
int attention_backend_ops(const char * device, bool fused, const AttnCase & c, DType dtype = DType::F32, bool kv_f16 = true) {
    Graph g = attention_graph(c, 0.25f, dtype);
    auto backend = make_backend(device, fused, kv_f16);
    auto exe = backend->compile(g);
    if (!exe.is_ok()) return -1;
    Vec qkv = rnd(static_cast<size_t>(c.B * c.N * 3 * c.H * c.D), 740, 1.f);
    std::vector<uint16_t> q16 = to_f16(qkv);
    auto view = dtype == DType::F32 ? TensorView::create(qkv.data(), S({c.B, c.N, 3 * c.H * c.D}), DType::F32).value()
                                    : TensorView::create(q16.data(), S({c.B, c.N, 3 * c.H * c.D}), DType::F16).value();
    auto rep = exe.value()->profile({{"qkv", view}}, 1);
    return rep.is_ok() ? rep.value().nodes[0].backend_ops : -2;
}
constexpr int kAccelerateOps = 1, kFusedOps = 5, kFusedKvF16Ops = 7, kUnfusedOps = 24;  // Accelerate custom op: one operation  // fused: 3 views + flash + reshape (+2 copies for f16 K/V)
}  // namespace

// Whether "auto" resolves to a GPU on this machine (the expectations below differ: f16 K/V copies and the
// supported head dimensions are GPU properties; a CPU-only build, e.g. the sanitizer one, has neither).
static bool auto_is_gpu() { return !resolves_to_cpu("auto"); }

TEST(op_self_attention_path_selection) {
    EXPECT_EQ(attention_backend_ops("cpu", false, {1, 8, 2, 16}), kUnfusedOps);   // forced fallback
    EXPECT_EQ(attention_backend_ops("auto", false, {1, 8, 2, 16}), kUnfusedOps);
    const int cpu_default = attention_backend_ops("cpu", true, {1, 8, 2, 16});
    // On the CPU device with Accelerate (the default) attention is the custom Accelerate op; the CPU kernels' flash attention is what runs with cpu_blas off.
    EXPECT(cpu_default == kAccelerateOps || cpu_default == kFusedOps || cpu_default == kUnfusedOps);
    EXPECT_EQ(cpu_default, kAccelerateOps);
    if (cpu_default == kAccelerateOps) EXPECT_EQ(attention_backend_ops("cpu", true, {1, 210, 16, 80}), kAccelerateOps);

    if (auto_is_gpu()) {
        EXPECT_EQ(attention_backend_ops("auto", true, {1, 210, 16, 80}, DType::F32, false), kFusedOps);   // Metal supports head dim 80
        EXPECT_EQ(attention_backend_ops("auto", true, {1, 210, 16, 80}), kFusedKvF16Ops);                 // default: K/V copied to f16
        EXPECT_EQ(attention_backend_ops("auto", true, {1, 9, 2, 6}), kUnfusedOps);                         // head dim 6 unsupported: falls back
    } else {
        std::printf("        (no GPU here: skipped the GPU-specific path expectations)\n");
        EXPECT_EQ(attention_backend_ops("auto", true, {1, 210, 16, 80}), cpu_default);
    }
}

// Worst case for the tensor budget: a device that rejects the fused op after it was built (3 views + flash
// attention) and then lowers the unfused chain, with f16 casts. Only this node is in the graph.
TEST(op_self_attention_worst_case_budget) {
    EXPECT(attention_backend_ops("auto", true, {1, 9, 2, 6}, DType::F16) > 0);
    EXPECT(attention_backend_ops("cpu", false, {1, 9, 2, 6}, DType::F16) > 0);
    EXPECT(attention_backend_ops("auto", true, {1, 8, 2, 16}, DType::F16) > 0);
    EXPECT(attention_backend_ops("auto", true, {1, 210, 16, 80}, DType::F16) > 0);   // fused + f16 K/V + f16 casts
}

// ---- whole network: y = gelu(x W1^T) W2^T ----

static Result<Graph> mlp_graph() {
    Graph g;
    auto x = g.add_input("x", S({8, 256}), DType::F32).value();
    auto w1 = g.add_weight("w1", S({1024, 256}), DType::F32).value();
    auto w2 = g.add_weight("w2", S({64, 1024}), DType::F32).value();
    auto h = g.add_gelu(g.add_matmul(x, w1, true).value()).value();
    Status st = g.mark_output("y", g.add_matmul(h, w2, true).value());
    if (!st.is_ok()) return st;
    return g;
}

// Seeded Gaussian weights (mt19937(0), stddev 0.05): small enough that activations stay O(1).
static void mlp_weights(Vec * w1, Vec * w2) {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 0.05f);
    w1->resize(1024 * 256);
    for (float & v : *w1) v = nd(rng);
    w2->resize(64 * 1024);
    for (float & v : *w2) v = nd(rng);
}

TEST(mlp_matches_double_reference) {
    Graph g = mlp_graph().value();
    Vec w1, w2;
    mlp_weights(&w1, &w2);
    Vec x(8 * 256, 0.1f);
    auto got = run_f32(g, {{"w1", w1}, {"w2", w2}}, {{"x", x}}, "y");
    EXPECT(got.is_ok());
    auto h = matmul_ref(x, w1, 1, 8, 256, 1024, true, false);
    Vec hv(h.size());
    for (size_t i = 0; i < h.size(); i++) hv[i] = static_cast<float>(gelu_ref(h[i]));
    EXPECT(close(got.value(), matmul_ref(hv, w2, 1, 8, 1024, 64, true, false), kF32Tol, "mlp vs reference"));
}

TEST(mlp_gpu_matches_cpu) {
    Graph g = mlp_graph().value();
    Vec w1, w2;
    mlp_weights(&w1, &w2);
    Vec x(8 * 256, 0.1f);
    auto cpu = run_f32(g, {{"w1", w1}, {"w2", w2}}, {{"x", x}}, "y", "cpu");
    auto best = run_f32(g, {{"w1", w1}, {"w2", w2}}, {{"x", x}}, "y", "auto");
    EXPECT(cpu.is_ok());
    EXPECT(best.is_ok());
    if (cpu.is_ok() && best.is_ok()) EXPECT(close(best.value(), widen(cpu.value()), kDeviceTol, "auto device vs cpu"));
}

// ---- reduced precision paths (cast wrappers; also exercise the per-node tensor budget) ----


TEST(op_scale_f16_keeps_dtype) {
    Graph g;
    auto x = g.add_input("x", S({2, 4}), DType::F16).value();
    EXPECT(g.mark_output("y", g.add_scale(x, 0.25f).value()).is_ok());
    Vec xv = rnd(8, 21, 4.f);
    auto x16 = to_f16(xv);
    std::vector<uint16_t> y16(8);
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->run({{"x", TensorView::create(x16.data(), S({2, 4}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({2, 4}), DType::F16).value()}}).is_ok());
    std::vector<double> want;
    for (float v : round_f16(xv)) want.push_back(static_cast<double>(v) * 0.25);
    EXPECT(close(from_f16(y16), want, kF16Tol, "f16 scale"));
}

TEST(f16_matmul_non_transposed_worst_case_budget) {
    Graph g;
    auto a = g.add_input("a", S({4, 8}), DType::F16).value();
    auto w = g.add_weight("w", S({8, 6}), DType::F16).value();
    EXPECT(g.mark_output("y", g.add_matmul(a, w, false).value()).is_ok());
    Vec av = rnd(32, 19), wv = rnd(48, 20);
    auto a16 = to_f16(av), w16 = to_f16(wv);
    std::vector<uint16_t> y16(24);
    auto backend = make_backend("cpu");
    EXPECT(backend != nullptr);
    if (!backend) return;
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->set_weight("w", TensorView::create(w16.data(), S({8, 6}), DType::F16).value()).is_ok());
    EXPECT(exe.value()->run({{"a", TensorView::create(a16.data(), S({4, 8}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({4, 6}), DType::F16).value()}}).is_ok());
    EXPECT(close(from_f16(y16), matmul_ref(round_f16(av), round_f16(wv), 1, 4, 8, 6, false, false), kF16Tol, "f16 matmul"));
}

TEST(f16_layer_norm_worst_case_budget) {
    Graph g;
    auto x = g.add_input("x", S({2, 3, 8}), DType::F16).value();
    auto ga = g.add_weight("g", S({8}), DType::F16).value();
    auto be = g.add_weight("b", S({8}), DType::F16).value();
    constexpr float eps = 1e-5f;
    EXPECT(g.mark_output("y", g.add_layer_norm(x, ga, be, eps).value()).is_ok());
    Vec xv = rnd(48, 21, 2.f), gv = rnd(8, 22), bv = rnd(8, 23);
    auto x16 = to_f16(xv), g16 = to_f16(gv), b16 = to_f16(bv);
    std::vector<uint16_t> y16(48);
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->set_weight("g", TensorView::create(g16.data(), S({8}), DType::F16).value()).is_ok());
    EXPECT(exe.value()->set_weight("b", TensorView::create(b16.data(), S({8}), DType::F16).value()).is_ok());
    EXPECT(exe.value()->run({{"x", TensorView::create(x16.data(), S({2, 3, 8}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({2, 3, 8}), DType::F16).value()}}).is_ok());
    EXPECT(close(from_f16(y16), layer_norm_ref(round_f16(xv), round_f16(gv), round_f16(bv), 6, 8, eps), kF16Tol, "f16 layer_norm"));
}

// ---- slice / concat / conv2d ----

TEST(op_slice_along_each_axis) {
    const std::vector<int64_t> dims = {3, 6, 5};
    struct Case { size_t axis; int64_t start, end; };
    for (Case c : {Case{0, 1, 3}, Case{1, 1, 4}, Case{1, 0, 6}, Case{2, 3, 4}, Case{2, 0, 2}}) {
        Graph g;
        auto x = g.add_input("x", Shape::create(dims.data(), 3).value(), DType::F32).value();
        EXPECT(g.mark_output("y", g.add_slice(x, static_cast<int64_t>(c.axis), c.start, c.end).value()).is_ok());
        Vec xv = rnd(90, 50 + c.axis);
        auto got = run_f32(g, {}, {{"x", xv}}, "y");
        EXPECT(got.is_ok());
        EXPECT(close(got.value(), slice_ref(xv, dims, c.axis, c.start, c.end), 0.0, "slice"));
    }
}

TEST(op_concat_two_and_many_inputs) {
    struct Case { size_t axis; std::vector<std::vector<int64_t>> dims; };
    const std::vector<Case> cases = {
        {0, {{2, 3, 4}, {1, 3, 4}}},
        {1, {{2, 3, 4}, {2, 5, 4}, {2, 1, 4}}},
        {2, {{2, 3, 4}, {2, 3, 2}}},
        {1, {{2, 1, 3}, {2, 2, 3}, {2, 1, 3}, {2, 4, 3}, {2, 1, 3}}},
    };
    for (const Case & c : cases) {
        Graph g;
        std::vector<ValueId> ids;
        std::vector<Vec> parts;
        Named inputs;
        for (size_t i = 0; i < c.dims.size(); i++) {
            const auto & d = c.dims[i];
            const std::string name = "in" + std::to_string(i);
            ids.push_back(g.add_input(name, Shape::create(d.data(), static_cast<int>(d.size())).value(), DType::F32).value());
            parts.push_back(rnd(prod(d, 0, d.size()), 60 + i));
            inputs.push_back({name, parts.back()});
        }
        EXPECT(g.mark_output("y", g.add_concat(ids, static_cast<int64_t>(c.axis)).value()).is_ok());
        auto got = run_f32(g, {}, inputs, "y");
        EXPECT(got.is_ok());
        EXPECT(close(got.value(), concat_ref(parts, c.dims, c.axis), 0.0, "concat"));
    }
}

struct ConvCase {
    int64_t N, C, H, W, OC, KH, KW;
    Conv2dParams p;
};

static ConvCase conv_case(int64_t N, int64_t C, int64_t H, int64_t W, int64_t OC, int64_t KH, int64_t KW,
                          int64_t sh = 1, int64_t sw = 1, int64_t ph = 0, int64_t pw = 0, int64_t dh = 1, int64_t dw = 1) {
    ConvCase c{N, C, H, W, OC, KH, KW, Conv2dParams()};
    c.p.stride_h = sh; c.p.stride_w = sw; c.p.pad_h = ph; c.p.pad_w = pw; c.p.dilation_h = dh; c.p.dilation_w = dw;
    return c;
}

static void check_conv(const ConvCase & c, const char * device, double tol, const char * what) {
    Graph g;
    auto x = g.add_input("x", S({c.N, c.C, c.H, c.W}), DType::F32).value();
    auto w = g.add_weight("w", S({c.OC, c.C, c.KH, c.KW}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_conv2d(x, w, c.p).value()).is_ok());
    Vec xv = rnd(c.N * c.C * c.H * c.W, 70), wv = rnd(c.OC * c.C * c.KH * c.KW, 71);
    auto got = run_f32(g, {{"w", wv}}, {{"x", xv}}, "y", device);
    EXPECT(got.is_ok());
    int64_t oh, ow;
    auto want = conv2d_ref(xv, wv, c.N, c.C, c.H, c.W, c.OC, c.KH, c.KW, c.p, &oh, &ow);
    EXPECT_EQ(g.values()[g.outputs()[0].value].type.shape.to_string(), S({c.N, c.OC, oh, ow}).to_string());
    if (got.is_ok()) EXPECT(close(got.value(), want, tol, what));
}

TEST(op_conv2d_matches_direct_convolution) {
    check_conv(conv_case(1, 1, 5, 5, 1, 3, 3), "cpu", kF32Tol, "conv basic");
    check_conv(conv_case(2, 3, 7, 9, 4, 3, 3, 1, 1, 1, 1), "cpu", kF32Tol, "conv same padding, batch 2");
    check_conv(conv_case(1, 2, 8, 10, 3, 3, 3, 2, 3), "cpu", kF32Tol, "conv uneven strides");
    check_conv(conv_case(1, 2, 9, 9, 3, 3, 3, 1, 1, 0, 0, 2, 2), "cpu", kF32Tol, "conv dilation 2");
    check_conv(conv_case(2, 3, 10, 7, 4, 3, 2, 2, 1, 2, 1), "cpu", kF32Tol, "conv non-square kernel");
    check_conv(conv_case(1, 3, 32, 24, 8, 16, 16, 16, 16, 2, 2), "cpu", kF32Tol, "conv patch embed, padding 2 (as ViT-H)");
    check_conv(conv_case(1, 3, 32, 24, 8, 16, 16, 16, 16, 4, 4), "cpu", kF32Tol, "conv patch embed, padding 4");
}

TEST(op_conv2d_f32_input_is_not_rounded_to_f16) {
    // Values with bits below f16 precision: a conv that routed through f16 im2col would miss kF32Tol.
    Graph g;
    auto x = g.add_input("x", S({1, 1, 4, 4}), DType::F32).value();
    auto w = g.add_weight("w", S({1, 1, 1, 1}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_conv2d(x, w, Conv2dParams()).value()).is_ok());
    Vec xv(16);
    for (int i = 0; i < 16; i++) xv[i] = 1.0f + 1e-4f * static_cast<float>(i + 1);  // f16 spacing near 1 is ~1e-3
    auto got = run_f32(g, {{"w", Vec{1.0f}}}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    EXPECT(close(got.value(), widen(xv), 1e-6, "identity conv keeps f32 precision"));
}

TEST(chain_vit_front_end_patch_embed_to_tokens) {
    // image -> conv2d patchify (+bias) -> [N, tokens, dim] -> layer_norm, against a composed reference.
    const int64_t N = 2, C = 3, H = 20, W = 12, D = 8, K = 4;
    Conv2dParams p;
    p.stride_h = p.stride_w = K;
    p.pad_h = p.pad_w = 2;
    Graph g;
    auto img = g.add_input("img", S({N, C, H, W}), DType::F32).value();
    auto pw = g.add_weight("pw", S({D, C, K, K}), DType::F32).value();
    auto pb = g.add_weight("pb", S({D, 1, 1}), DType::F32).value();
    auto ga = g.add_weight("g", S({D}), DType::F32).value();
    auto be = g.add_weight("b", S({D}), DType::F32).value();
    auto f = g.add_add(g.add_conv2d(img, pw, p).value(), pb).value();          // [N, D, OH, OW]
    const Shape fs = g.values()[f].type.shape;
    const int64_t OH = fs.dim(2), OW = fs.dim(3);
    auto flat = g.add_reshape(f, {N, D, OH * OW}).value();
    auto tok = g.add_permute(flat, {0, 2, 1}).value();                          // [N, OH*OW, D]
    constexpr float eps = 1e-6f;
    EXPECT(g.mark_output("y", g.add_layer_norm(tok, ga, be, eps).value()).is_ok());

    Vec iv = rnd(N * C * H * W, 80), wv = rnd(D * C * K * K, 81, 0.3f), bv = rnd(D, 82), gv = rnd(D, 83), bev = rnd(D, 84);
    auto got = run_f32(g, {{"pw", wv}, {"pb", bv}, {"g", gv}, {"b", bev}}, {{"img", iv}}, "y");
    EXPECT(got.is_ok());

    int64_t oh, ow;
    auto conv = conv2d_ref(iv, wv, N, C, H, W, D, K, K, p, &oh, &ow);
    EXPECT_EQ(oh, OH);
    EXPECT_EQ(ow, OW);
    Vec tokens(N * OH * OW * D);
    for (int64_t n = 0; n < N; n++)
        for (int64_t d = 0; d < D; d++)
            for (int64_t t = 0; t < OH * OW; t++)
                tokens[(n * OH * OW + t) * D + d] = static_cast<float>(conv[(n * D + d) * OH * OW + t] + bv[d]);
    EXPECT(close(got.value(), layer_norm_ref(tokens, gv, bev, N * OH * OW, D, eps), kF32Tol, "vit front end"));
}

TEST(chain_qkv_split_attention) {
    // tokens -> fused qkv projection -> three slices -> heads -> q.k^T -> softmax
    const int64_t B = 2, T = 5, Dm = 12, H = 3, Dh = Dm / H;
    Graph g;
    auto x = g.add_input("x", S({B, T, Dm}), DType::F32).value();
    auto wqkv = g.add_weight("wqkv", S({3 * Dm, Dm}), DType::F32).value();
    auto qkv = g.add_matmul(x, wqkv, true).value();
    auto heads = [&](int64_t part) {
        auto sl = g.add_slice(qkv, 2, part * Dm, (part + 1) * Dm).value();
        auto r = g.add_reshape(sl, {B, T, H, Dh}).value();
        return g.add_permute(r, {0, 2, 1, 3}).value();                           // [B,H,T,Dh]
    };
    auto q = heads(0), k = heads(1);
    EXPECT(g.mark_output("p", g.add_softmax(g.add_matmul(q, k, true).value(), 3).value()).is_ok());

    Vec xv = rnd(B * T * Dm, 90), wv = rnd(3 * Dm * Dm, 91);
    auto got = run_f32(g, {{"wqkv", wv}}, {{"x", xv}}, "p");
    EXPECT(got.is_ok());

    auto qkv_ref = matmul_ref(xv, wv, 1, B * T, Dm, 3 * Dm, true, false);        // [B*T, 3*Dm]
    auto head_ref = [&](int64_t part) {
        Vec sl;  // [B, T, Dm] part, then heads
        for (int64_t r = 0; r < B * T; r++)
            for (int64_t c = 0; c < Dm; c++) sl.push_back(static_cast<float>(qkv_ref[r * 3 * Dm + part * Dm + c]));
        Vec out;
        for (double v : permute_ref(sl, {B, T, H, Dh}, {0, 2, 1, 3})) out.push_back(static_cast<float>(v));
        return out;
    };
    auto scores = matmul_ref(head_ref(0), head_ref(1), B * H, T, Dh, T, true, true);
    EXPECT(close(got.value(), softmax_ref(Vec(scores.begin(), scores.end()), B * H * T, T), kF32Tol, "qkv attention"));
}

TEST(new_ops_run_on_the_best_device_too) {
    // Catches ops the GPU backend cannot execute (e.g. unsupported im2col variants).
    check_conv(conv_case(2, 3, 10, 8, 4, 3, 3, 1, 1, 1, 1), "auto", kDeviceTol, "conv on auto device");
    check_conv(conv_case(1, 3, 32, 24, 8, 16, 16, 16, 16, 2, 2), "auto", kDeviceTol, "patch embed on auto device");

    Graph g;
    auto x = g.add_input("x", S({3, 6, 5}), DType::F32).value();
    auto y = g.add_input("y", S({3, 2, 5}), DType::F32).value();
    auto cat = g.add_concat({g.add_slice(x, 1, 1, 4).value(), y}, 1).value();
    EXPECT(g.mark_output("o", cat).is_ok());
    Vec xv = rnd(90, 100), yv = rnd(30, 101);
    auto got = run_f32(g, {}, {{"x", xv}, {"y", yv}}, "o", "auto");
    EXPECT(got.is_ok());
    auto sl = slice_ref(xv, {3, 6, 5}, 1, 1, 4);
    EXPECT(close(got.value(), concat_ref({Vec(sl.begin(), sl.end()), yv}, {{3, 3, 5}, {3, 2, 5}}, 1), 0.0, "slice+concat on auto device"));
}

// f16: exercises the cast wrappers and the worst-case per-node tensor budgets (ggml aborts if exceeded).
TEST(f16_conv2d_worst_case_budget) {
    Graph g;
    auto x = g.add_input("x", S({1, 2, 6, 6}), DType::F16).value();
    auto w = g.add_weight("w", S({3, 2, 3, 3}), DType::F16).value();
    Conv2dParams p;
    p.pad_h = p.pad_w = 1;
    EXPECT(g.mark_output("y", g.add_conv2d(x, w, p).value()).is_ok());
    Vec xv = rnd(72, 110), wv = rnd(54, 111, 0.5f);
    auto x16 = to_f16(xv), w16 = to_f16(wv);
    std::vector<uint16_t> y16(1 * 3 * 6 * 6);
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->set_weight("w", TensorView::create(w16.data(), S({3, 2, 3, 3}), DType::F16).value()).is_ok());
    EXPECT(exe.value()->run({{"x", TensorView::create(x16.data(), S({1, 2, 6, 6}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({1, 3, 6, 6}), DType::F16).value()}}).is_ok());
    int64_t oh, ow;
    EXPECT(close(from_f16(y16), conv2d_ref(round_f16(xv), round_f16(wv), 1, 2, 6, 6, 3, 3, 3, p, &oh, &ow), kF16Tol, "f16 conv2d"));
}

// ---- relu / conv_transpose2d ----

static ConvTranspose2dParams ct_params(int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    ConvTranspose2dParams p;
    p.stride_h = sh; p.stride_w = sw; p.pad_h = ph; p.pad_w = pw;
    return p;
}

TEST(op_relu) {
    Graph g;
    auto x = g.add_input("x", S({2, 3, 4}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_relu(x).value()).is_ok());
    Vec xv = rnd(24, 120);
    xv[0] = 0.0f;  // the boundary itself
    std::vector<double> want;
    for (float v : xv) want.push_back(v > 0 ? v : 0.0);
    bool has_neg = false, has_pos = false;
    for (float v : xv) { has_neg |= v < 0; has_pos |= v > 0; }
    EXPECT(has_neg && has_pos);
    auto got = run_f32(g, {}, {{"x", xv}}, "y");
    EXPECT(got.is_ok());
    if (got.is_ok()) EXPECT(close(got.value(), want, 0.0, "relu"));
}

static void check_conv_transpose(int64_t N, int64_t IC, int64_t H, int64_t W, int64_t OC, int64_t KH, int64_t KW,
                                 const ConvTranspose2dParams & p, const char * device, double tol, const char * what) {
    Graph g;
    auto x = g.add_input("x", S({N, IC, H, W}), DType::F32).value();
    auto w = g.add_weight("w", S({IC, OC, KH, KW}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_conv_transpose2d(x, w, p).value()).is_ok());
    Vec xv = rnd(N * IC * H * W, 130), wv = rnd(IC * OC * KH * KW, 131);
    auto got = run_f32(g, {{"w", wv}}, {{"x", xv}}, "y", device);
    EXPECT(got.is_ok());
    int64_t oh, ow;
    auto want = conv_transpose2d_ref(xv, wv, N, IC, H, W, OC, KH, KW, p, &oh, &ow);
    EXPECT_EQ(g.values()[g.outputs()[0].value].type.shape.to_string(), S({N, OC, oh, ow}).to_string());
    if (got.is_ok()) EXPECT(close(got.value(), want, tol, what));
}

TEST(op_conv_transpose2d_matches_scatter_reference) {
    check_conv_transpose(1, 1, 3, 3, 1, 2, 2, ct_params(1, 1, 0, 0), "cpu", kF32Tol, "convT basic");
    check_conv_transpose(2, 3, 4, 5, 2, 3, 3, ct_params(2, 2, 0, 0), "cpu", kF32Tol, "convT stride 2, batch 2");
    check_conv_transpose(1, 4, 6, 5, 3, 4, 4, ct_params(2, 2, 1, 1), "cpu", kF32Tol, "convT k4 s2 p1 (as RefineNet)");
    check_conv_transpose(1, 2, 5, 4, 3, 3, 5, ct_params(2, 2, 1, 2), "cpu", kF32Tol, "convT non-square kernel and padding");
    check_conv_transpose(1, 8, 16, 12, 4, 4, 4, ct_params(2, 2, 1, 1), "cpu", kF32Tol, "convT RefineNet first scale shape");
}

TEST(op_conv_transpose2d_doubles_resolution_like_refinenet) {
    Graph g;
    auto x = g.add_input("x", S({1, 2, 16, 12}), DType::F32).value();
    auto w = g.add_weight("w", S({2, 2, 4, 4}), DType::F32).value();
    auto y = g.add_conv_transpose2d(x, w, ct_params(2, 2, 1, 1)).value();
    EXPECT_EQ(g.values()[y].type.shape.to_string(), S({1, 2, 32, 24}).to_string());
}

TEST(op_conv_transpose2d_runs_on_the_best_device_too) {
    check_conv_transpose(1, 4, 6, 5, 3, 4, 4, ct_params(2, 2, 1, 1), "auto", kDeviceTol, "convT on auto device");
    Graph g;
    auto x = g.add_input("x", S({2, 3, 4}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_relu(x).value()).is_ok());
    Vec xv = rnd(24, 121);
    std::vector<double> want;
    for (float v : xv) want.push_back(v > 0 ? v : 0.0);
    auto got = run_f32(g, {}, {{"x", xv}}, "y", "auto");
    EXPECT(got.is_ok());
    if (got.is_ok()) EXPECT(close(got.value(), want, 0.0, "relu on auto device"));
}

TEST(op_conv_transpose2d_unequal_strides_are_unsupported) {
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    Graph g;
    auto x = g.add_input("x", S({1, 2, 4, 4}), DType::F32).value();
    auto w = g.add_weight("w", S({2, 2, 3, 3}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_conv_transpose2d(x, w, ct_params(2, 1, 0, 0)).value()).is_ok());
    EXPECT(backend->supports(g).code() == Code::Unsupported);
    EXPECT(backend->compile(g).status().code() == Code::Unsupported);
}

TEST(f16_conv_transpose2d_and_relu_worst_case_budget) {
    // Only the node under test in each graph, so spare budget cannot hide a bound that is one tensor too small.
    const ConvTranspose2dParams p = ct_params(2, 2, 1, 1);
    Graph g;
    auto x = g.add_input("x", S({1, 2, 4, 4}), DType::F16).value();
    auto w = g.add_weight("w", S({2, 3, 4, 4}), DType::F16).value();
    EXPECT(g.mark_output("y", g.add_conv_transpose2d(x, w, p).value()).is_ok());
    Vec xv = rnd(32, 140), wv = rnd(96, 141, 0.5f);
    auto x16 = to_f16(xv), w16 = to_f16(wv);
    std::vector<uint16_t> y16(1 * 3 * 8 * 8);
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->set_weight("w", TensorView::create(w16.data(), S({2, 3, 4, 4}), DType::F16).value()).is_ok());
    EXPECT(exe.value()->run({{"x", TensorView::create(x16.data(), S({1, 2, 4, 4}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({1, 3, 8, 8}), DType::F16).value()}}).is_ok());
    int64_t oh, ow;
    EXPECT(close(from_f16(y16), conv_transpose2d_ref(round_f16(xv), round_f16(wv), 1, 2, 4, 4, 3, 4, 4, p, &oh, &ow), kF16Tol, "f16 convT"));

    Graph r;
    auto rx = r.add_input("x", S({4, 5}), DType::F16).value();
    EXPECT(r.mark_output("y", r.add_relu(rx).value()).is_ok());
    Vec rv = rnd(20, 142);
    auto r16 = to_f16(rv);
    std::vector<uint16_t> ry(20);
    auto rexe = backend->compile(r);
    EXPECT(rexe.is_ok());
    if (!rexe.is_ok()) return;
    EXPECT(rexe.value()->run({{"x", TensorView::create(r16.data(), S({4, 5}), DType::F16).value()}},
                             {{"y", TensorView::create(ry.data(), S({4, 5}), DType::F16).value()}}).is_ok());
    std::vector<double> want;
    for (float v : round_f16(rv)) want.push_back(v > 0 ? v : 0.0);
    EXPECT(close(from_f16(ry), want, 0.0, "f16 relu"));
}

// ---- silu / max_pool2d / upsample_nearest2d ----

TEST(op_silu) {
    Graph g;
    auto x = g.add_input("x", S({3, 7}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_silu(x).value()).is_ok());
    Vec xv = rnd(21, 150, 6.f);
    std::vector<double> want;
    for (float v : xv) want.push_back(silu_ref(v));
    for (const char * dev : {"cpu", "auto"}) {
        auto got = run_f32(g, {}, {{"x", xv}}, "y", dev);
        EXPECT(got.is_ok());
        if (got.is_ok()) EXPECT(close(got.value(), want, resolves_to_cpu(dev) ? 1e-5 : 1e-3, "silu"));
    }
}

static void check_pool(int64_t N, int64_t C, int64_t H, int64_t W, const MaxPool2dParams & p, const char * device, const char * what) {
    Graph g;
    auto x = g.add_input("x", S({N, C, H, W}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_max_pool2d(x, p).value()).is_ok());
    Vec xv = rnd(N * C * H * W, 151, 4.f);
    auto got = run_f32(g, {}, {{"x", xv}}, "y", device);
    EXPECT(got.is_ok());
    int64_t oh, ow;
    auto want = max_pool_ref(xv, N, C, H, W, p, &oh, &ow);
    EXPECT_EQ(g.values()[g.outputs()[0].value].type.shape.to_string(), S({N, C, oh, ow}).to_string());
    if (got.is_ok()) EXPECT(close(got.value(), want, 0.0, what));  // max is exact
}

static MaxPool2dParams pool_p(int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    MaxPool2dParams p;
    p.kernel_h = kh; p.kernel_w = kw; p.stride_h = sh; p.stride_w = sw; p.pad_h = ph; p.pad_w = pw;
    return p;
}

TEST(op_max_pool2d_matches_direct_pooling) {
    for (const char * dev : {"cpu", "auto"}) {
        check_pool(1, 3, 8, 6, pool_p(5, 5, 1, 1, 2, 2), dev, "SPPF pooling k5 s1 p2 (size preserving, padded border)");
        check_pool(2, 2, 9, 7, pool_p(2, 2, 2, 2, 0, 0), dev, "k2 s2");
        check_pool(1, 4, 11, 13, pool_p(3, 3, 2, 2, 1, 1), dev, "k3 s2 p1");
        check_pool(1, 2, 10, 12, pool_p(3, 2, 1, 2, 1, 0), dev, "non-square window");
    }
}

TEST(op_max_pool2d_all_negative_input_is_not_polluted_by_padding) {
    // If padding acted as 0 instead of -inf the maxima at the border would be 0.
    Graph g;
    auto x = g.add_input("x", S({1, 1, 4, 4}), DType::F32).value();
    EXPECT(g.mark_output("y", g.add_max_pool2d(x, pool_p(3, 3, 1, 1, 1, 1)).value()).is_ok());
    Vec xv(16);
    for (int i = 0; i < 16; i++) xv[i] = -1.f - static_cast<float>(i);
    for (const char * dev : {"cpu", "auto"}) {
        auto got = run_f32(g, {}, {{"x", xv}}, "y", dev);
        EXPECT(got.is_ok());
        if (got.is_ok()) for (float v : got.value()) EXPECT(v < 0.f);
    }
}

TEST(op_upsample_nearest2d) {
    for (const char * dev : {"cpu", "auto"}) {
        for (auto sc : {std::pair<int64_t, int64_t>{2, 2}, {3, 1}, {1, 2}, {2, 3}}) {
            Graph g;
            auto x = g.add_input("x", S({2, 3, 5, 4}), DType::F32).value();
            UpsampleParams up;
            up.scale_h = sc.first; up.scale_w = sc.second;
            EXPECT(g.mark_output("y", g.add_upsample_nearest2d(x, up).value()).is_ok());
            Vec xv = rnd(2 * 3 * 5 * 4, 152);
            auto got = run_f32(g, {}, {{"x", xv}}, "y", dev);
            EXPECT(got.is_ok());
            if (got.is_ok()) EXPECT(close(got.value(), upsample_ref(xv, 6, 5, 4, sc.first, sc.second), 0.0, "nearest upsample"));
        }
    }
}

TEST(f16_detector_ops_keep_dtype_and_stay_within_budget) {
    auto run16 = [&](const Graph & g, const Vec & xv, size_t out_n) {
        auto backend = make_backend("cpu");
        std::vector<uint16_t> x16 = to_f16(xv), y16(out_n);
        auto exe = backend->compile(g);
        EXPECT(exe.is_ok());
        if (!exe.is_ok()) return Vec();
        EXPECT(exe.value()->run({{"x", TensorView::create(x16.data(), g.values()[g.find_named("x").value()].type.shape, DType::F16).value()}},
                                {{"y", TensorView::create(y16.data(), g.values()[g.outputs()[0].value].type.shape, DType::F16).value()}}).is_ok());
        return from_f16(y16);
    };
    Vec xv = rnd(1 * 2 * 6 * 6, 153, 3.f);
    {
        Graph g;
        auto x = g.add_input("x", S({1, 2, 6, 6}), DType::F16).value();
        EXPECT(g.mark_output("y", g.add_silu(x).value()).is_ok());
        std::vector<double> want;
        for (float v : round_f16(xv)) want.push_back(silu_ref(v));
        EXPECT(close(run16(g, xv, 72), want, kF16Tol, "f16 silu"));
    }
    {
        Graph g;
        auto x = g.add_input("x", S({1, 2, 6, 6}), DType::F16).value();
        EXPECT(g.mark_output("y", g.add_max_pool2d(x, pool_p(3, 3, 1, 1, 1, 1)).value()).is_ok());
        int64_t oh, ow;
        EXPECT(close(run16(g, xv, 72), max_pool_ref(round_f16(xv), 1, 2, 6, 6, pool_p(3, 3, 1, 1, 1, 1), &oh, &ow), kF16Tol, "f16 max pool"));
    }
    {
        Graph g;
        auto x = g.add_input("x", S({1, 2, 6, 6}), DType::F16).value();
        UpsampleParams up;
        up.scale_h = up.scale_w = 2;
        EXPECT(g.mark_output("y", g.add_upsample_nearest2d(x, up).value()).is_ok());
        EXPECT(close(run16(g, xv, 288), upsample_ref(round_f16(xv), 2, 6, 6, 2, 2), kF16Tol, "f16 upsample"));
    }
}

// The budget tests below contain ONLY the node under test: any other node's spare budget would
// hide a bound that is one tensor too small.
TEST(f16_concat_many_inputs_worst_case_budget) {
    const int n_parts = 5;
    Graph g;
    std::vector<ValueId> ids;
    std::vector<std::vector<uint16_t>> bufs;
    std::vector<Vec> parts;
    for (int i = 0; i < n_parts; i++) {
        ids.push_back(g.add_input("in" + std::to_string(i), S({2, 3}), DType::F16).value());
        parts.push_back(rnd(6, 120 + i));
        bufs.push_back(to_f16(parts.back()));
    }
    EXPECT(g.mark_output("cat", g.add_concat(ids, 1).value()).is_ok());          // [2, 15]
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    std::vector<NamedTensor> in;
    for (int i = 0; i < n_parts; i++) in.push_back({"in" + std::to_string(i), TensorView::create(bufs[i].data(), S({2, 3}), DType::F16).value()});
    std::vector<uint16_t> cat16(30);
    EXPECT(exe.value()->run(in, {{"cat", TensorView::create(cat16.data(), S({2, 15}), DType::F16).value()}}).is_ok());
    std::vector<Vec> rounded;
    for (auto & p : parts) rounded.push_back(round_f16(p));
    EXPECT(close(from_f16(cat16), concat_ref(rounded, std::vector<std::vector<int64_t>>(n_parts, {2, 3}), 1), 0.0, "f16 concat"));
}

TEST(f16_slice_keeps_dtype) {
    Graph g;
    auto x = g.add_input("x", S({2, 8}), DType::F16).value();
    EXPECT(g.mark_output("y", g.add_slice(x, 1, 3, 7).value()).is_ok());
    Vec xv = rnd(16, 130);
    auto x16 = to_f16(xv);
    std::vector<uint16_t> y16(8);
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    EXPECT(exe.value()->run({{"x", TensorView::create(x16.data(), S({2, 8}), DType::F16).value()}},
                            {{"y", TensorView::create(y16.data(), S({2, 4}), DType::F16).value()}}).is_ok());
    EXPECT(close(from_f16(y16), slice_ref(round_f16(xv), {2, 8}, 1, 3, 7), 0.0, "f16 slice"));
}

// ---- profiling ----

namespace {

// Compiles `g`, runs profile() with `iterations` on random inputs; weights (if any) get random data.
Result<ProfileReport> profile_graph(const Graph & g, int iterations = 3, const char * device = "cpu") {
    auto backend = make_backend(device);
    if (!backend) return internal_error("no backend");
    auto exe = backend->compile(g);
    if (!exe.is_ok()) return exe.status();
    std::vector<Vec> keep;
    keep.reserve(g.values().size());
    std::vector<NamedTensor> in;
    uint32_t seed = 500;
    for (const Value & v : g.values()) {
        if (v.kind == ValueKind::Intermediate) continue;
        keep.push_back(rnd(static_cast<size_t>(v.type.shape.numel()), seed++, 0.5f));
        auto view = TensorView::create(keep.back().data(), v.type.shape, DType::F32).value();
        if (v.kind == ValueKind::Weight) {
            DOOMS_RETURN_IF_ERROR(exe.value()->set_weight(v.name, view));
        } else {
            in.push_back({v.name, view});
        }
    }
    return exe.value()->profile(in, iterations);
}

int ops_of_single_node(const std::function<ValueId(Graph &, ValueId, ValueId)> & build) {
    Graph g;
    auto x = g.add_input("x", S({2, 4, 4, 4}), DType::F32).value();
    auto y = g.add_input("y", S({2, 4, 4, 4}), DType::F32).value();
    ValueId r = build(g, x, y);
    if (r == kInvalidValue) return -1;
    g.mark_output("o", r);
    auto rep = profile_graph(g, 1);
    if (!rep.is_ok() || rep.value().nodes.empty()) return -2;
    return rep.value().nodes.back().backend_ops;
}

}  // namespace

TEST(profile_reports_every_node_in_order) {
    Graph g;
    auto x = g.add_input("x", S({8, 16}), DType::F32).value();
    auto w1 = g.add_weight("w1", S({32, 16}), DType::F32).value();
    auto w2 = g.add_weight("w2", S({4, 32}), DType::F32).value();
    g.mark_output("y", g.add_matmul(g.add_gelu(g.add_matmul(x, w1, true).value()).value(), w2, true).value());
    auto rep = profile_graph(g, 5);
    EXPECT(rep.is_ok());
    if (!rep.is_ok()) { std::fprintf(stderr, "  %s\n", rep.status().to_string().c_str()); return; }
    EXPECT_EQ(rep.value().iterations, 5);
    EXPECT_EQ(rep.value().nodes.size(), size_t(3));
    const OpKind expect_ops[] = {OpKind::MatMul, OpKind::Gelu, OpKind::MatMul};
    for (size_t k = 0; k < rep.value().nodes.size() && k < 3; k++) {
        const NodeProfile & n = rep.value().nodes[k];
        EXPECT_EQ(n.node_index, static_cast<int>(k));
        EXPECT(n.op == expect_ops[k]);
        EXPECT(std::isfinite(n.ms) && n.ms >= 0.0);
        EXPECT_EQ(n.backend_ops, 1);  // transposed-weight matmul and gelu are one ggml operation each
    }
    EXPECT(rep.value().whole_graph_ms > 0.0);
    EXPECT(rep.value().observation_overhead_ms >= 0.0);
}

TEST(profile_attributes_exactly_the_backend_operations_each_node_lowers_to) {
    // Derived from the lowering (src/backend/ggml/ggml_backend.cpp), f32 so no cast operations appear:
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId) { return g.add_scale(x, 2.f).value(); }), 1);
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId y) { return g.add_add(x, y).value(); }), 1);
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId) { return g.add_gelu(x).value(); }), 1);
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId) { return g.add_softmax(x, 3).value(); }), 1);
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId) { return g.add_reshape(x, {2, 16, 4}).value(); }), 1);          // one view
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId) { return g.add_permute(x, {0, 2, 1, 3}).value(); }), 2);        // permute view + cont
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId) { return g.add_slice(x, 1, 1, 3).value(); }), 2);                // view + cont
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId y) { return g.add_matmul(x, y, true).value(); }), 1);             // mul_mat
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId y) { return g.add_matmul(x, y, false).value(); }), 3);            // transpose view + cont + mul_mat
    EXPECT_EQ(ops_of_single_node([](Graph & g, ValueId x, ValueId y) { return g.add_concat({x, y, x}, 1).value(); }), 2);          // 3 inputs -> 2 concats
}

TEST(profile_layer_norm_and_conv2d_operation_counts) {
    Graph ln;
    auto x = ln.add_input("x", S({4, 8}), DType::F32).value();
    auto ga = ln.add_input("g", S({8}), DType::F32).value();
    auto be = ln.add_input("b", S({8}), DType::F32).value();
    ln.mark_output("y", ln.add_layer_norm(x, ga, be, 1e-5f).value());
    auto r1 = profile_graph(ln, 1);
    EXPECT(r1.is_ok());
    if (r1.is_ok()) EXPECT_EQ(r1.value().nodes[0].backend_ops, 3);  // norm, mul, add

    Graph cv;
    auto img = cv.add_input("img", S({1, 2, 6, 6}), DType::F32).value();
    auto w = cv.add_input("w", S({3, 2, 3, 3}), DType::F32).value();
    cv.mark_output("y", cv.add_conv2d(img, w, Conv2dParams()).value());
    auto r2 = profile_graph(cv, 1);
    EXPECT(r2.is_ok());
    // im2col, reshape_2d (im2col), reshape_2d (kernel), mul_mat, reshape_4d, permute, cont
    if (r2.is_ok()) EXPECT_EQ(r2.value().nodes[0].backend_ops, 7);
}

TEST(profile_observation_overhead_is_measured_only_from_view_operations) {
    Graph with_view;
    auto x = with_view.add_input("x", S({4, 8}), DType::F32).value();
    with_view.mark_output("y", with_view.add_gelu(with_view.add_reshape(x, {8, 4}).value()).value());
    auto r1 = profile_graph(with_view, 5);
    EXPECT(r1.is_ok());
    if (r1.is_ok()) EXPECT(r1.value().observation_overhead_ms > 0.0);   // a reshape is observed and costs something

    Graph no_view;
    auto y = no_view.add_input("x", S({4, 8}), DType::F32).value();
    no_view.mark_output("y", no_view.add_gelu(y).value());
    auto r2 = profile_graph(no_view, 5);
    EXPECT(r2.is_ok());
    if (r2.is_ok()) EXPECT_EQ(r2.value().observation_overhead_ms, 0.0); // nothing to measure it from
}

TEST(profile_does_not_disturb_results) {
    Graph g;
    auto x = g.add_input("x", S({4, 8}), DType::F32).value();
    auto w = g.add_weight("w", S({6, 8}), DType::F32).value();
    g.mark_output("y", g.add_gelu(g.add_matmul(x, w, true).value()).value());
    auto backend = make_backend("cpu");
    auto exe = std::move(backend->compile(g)).value();
    Vec xv = rnd(32, 61), wv = rnd(48, 62), before(24);
    EXPECT(exe->set_weight("w", TensorView::create(wv.data(), S({6, 8}), DType::F32).value()).is_ok());
    auto in = std::vector<NamedTensor>{{"x", TensorView::create(xv.data(), S({4, 8}), DType::F32).value()}};
    EXPECT(exe->run(in, {{"y", TensorView::create(before.data(), S({4, 6}), DType::F32).value()}}).is_ok());
    EXPECT(exe->profile(in, 4).is_ok());
    Vec again(24);
    EXPECT(exe->run(in, {{"y", TensorView::create(again.data(), S({4, 6}), DType::F32).value()}}).is_ok());
    EXPECT(before == again);  // profiling left the executable's state (weights, memory plan) intact
}

TEST(profile_validates_its_arguments) {
    // Each check is isolated (everything else valid) and identified by its message, so that removing one
    // check cannot hide behind another that returns the same error code.
    Graph g;
    auto x = g.add_input("x", S({2, 3}), DType::F32).value();
    auto w = g.add_weight("w", S({4, 3}), DType::F32).value();
    g.mark_output("y", g.add_matmul(x, w, true).value());
    auto backend = make_backend("cpu");
    auto exe = std::move(backend->compile(g)).value();
    Vec xv = rnd(6, 1), wv = rnd(12, 2), bad(8);
    auto xin = std::vector<NamedTensor>{{"x", TensorView::create(xv.data(), S({2, 3}), DType::F32).value()}};
    auto has = [](const Result<ProfileReport> & r, Code code, const char * fragment) {
        return !r.is_ok() && r.status().code() == code && r.status().message().find(fragment) != std::string::npos;
    };

    EXPECT(has(exe->profile(xin, 2), Code::InvalidArgument, "weight 'w' was not set"));  // valid except for the weight
    EXPECT(exe->set_weight("w", TensorView::create(wv.data(), S({4, 3}), DType::F32).value()).is_ok());
    EXPECT(has(exe->profile(xin, 0), Code::InvalidArgument, "iterations"));
    EXPECT(has(exe->profile(xin, -3), Code::InvalidArgument, "iterations"));
    EXPECT(has(exe->profile({}, 2), Code::InvalidArgument, "input 'x' was not provided"));
    auto wrong_shape = std::vector<NamedTensor>{{"x", TensorView::create(bad.data(), S({2, 4}), DType::F32).value()}};
    EXPECT(has(exe->profile(wrong_shape, 2), Code::InvalidArgument, "input 'x'"));
    auto unknown = std::vector<NamedTensor>{{"nope", TensorView::create(xv.data(), S({2, 3}), DType::F32).value()}};
    EXPECT(has(exe->profile(unknown, 2), Code::NotFound, "nope"));
    auto twice = std::vector<NamedTensor>{xin[0], xin[0]};
    EXPECT(has(exe->profile(twice, 2), Code::InvalidArgument, "bound twice"));
    EXPECT(exe->profile(xin, 2).is_ok());
}

TEST(profile_works_on_the_best_device_too) {
    Graph g;
    auto x = g.add_input("x", S({8, 16}), DType::F32).value();
    auto w = g.add_input("w", S({32, 16}), DType::F32).value();
    g.mark_output("y", g.add_gelu(g.add_matmul(x, w, true).value()).value());
    auto rep = profile_graph(g, 3, "auto");
    EXPECT(rep.is_ok());
    if (rep.is_ok()) {
        EXPECT_EQ(rep.value().nodes.size(), size_t(2));
        for (const NodeProfile & n : rep.value().nodes) EXPECT(std::isfinite(n.ms) && n.ms >= 0.0);
    }
}

namespace {
// A backend that does not implement profiling uses the interface default.
class NoProfileExecutable : public Executable {
public:
    Status set_weight(std::string_view, const TensorView &) override { return Status(); }
    Status run(const std::vector<NamedTensor> &, const std::vector<NamedTensor> &) override { return Status(); }
    const Graph & graph() const override { return g_; }
private:
    Graph g_;
};
}  // namespace

TEST(profile_default_is_unsupported) {
    NoProfileExecutable e;
    EXPECT(e.profile({}, 1).status().code() == Code::Unsupported);
}

// ---- unsupported combinations ----

TEST(unsupported_graphs_are_reported_not_attempted) {
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }

    Graph soft;
    auto x = soft.add_input("x", S({3, 4}), DType::F32).value();
    EXPECT(soft.mark_output("y", soft.add_softmax(x, 0).value()).is_ok());
    EXPECT(backend->supports(soft).code() == Code::Unsupported);
    EXPECT(backend->compile(soft).status().code() == Code::Unsupported);

    Graph ints;
    auto a = ints.add_input("a", S({4}), DType::I32).value();
    EXPECT(ints.mark_output("y", ints.add_add(a, a).value()).is_ok());
    EXPECT(backend->supports(ints).code() == Code::Unsupported);

    Graph fine;
    auto f = fine.add_input("f", S({3, 4}), DType::F32).value();
    EXPECT(fine.mark_output("y", fine.add_softmax(f, 1).value()).is_ok());
    EXPECT(backend->supports(fine).is_ok());
}

TEST(compile_rejects_graph_without_outputs) {
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    Graph g;
    g.add_input("x", S({2}), DType::F32);
    EXPECT(backend->compile(g).status().code() == Code::InvalidArgument);
}

// ---- run() contract ----

namespace {
struct Fixture {
    Graph g;
    std::unique_ptr<Backend> backend;
    std::unique_ptr<Executable> exe;
    Vec x = rnd(6, 30), w = rnd(12, 31), y = Vec(8);
    Fixture() {
        auto xi = g.add_input("x", S({2, 3}), DType::F32).value();
        auto wi = g.add_weight("w", S({4, 3}), DType::F32).value();
        g.mark_output("y", g.add_matmul(xi, wi, true).value());
        backend = make_backend("cpu");
        exe = std::move(backend->compile(g)).value();
    }
    TensorView xv() { return TensorView::create(x.data(), S({2, 3}), DType::F32).value(); }
    TensorView wv() { return TensorView::create(w.data(), S({4, 3}), DType::F32).value(); }
    TensorView yv() { return TensorView::create(y.data(), S({2, 4}), DType::F32).value(); }
};
}  // namespace

TEST(run_requires_all_weights_set) {
    Fixture f;
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", f.yv()}}).code() == Code::InvalidArgument);
    EXPECT(f.exe->set_weight("w", f.wv()).is_ok());
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", f.yv()}}).is_ok());
}

TEST(run_validates_input_bindings) {
    Fixture f;
    EXPECT(f.exe->set_weight("w", f.wv()).is_ok());
    EXPECT(f.exe->run({}, {{"y", f.yv()}}).code() == Code::InvalidArgument);                              // missing input
    EXPECT(f.exe->run({{"nope", f.xv()}}, {{"y", f.yv()}}).code() == Code::NotFound);                      // unknown input
    EXPECT(f.exe->run({{"w", f.xv()}}, {{"y", f.yv()}}).code() == Code::NotFound);                         // weight is not an input
    EXPECT(f.exe->run({{"x", f.xv()}, {"x", f.xv()}}, {{"y", f.yv()}}).code() == Code::InvalidArgument);   // twice
    Vec wrong(8);
    auto bad_shape = TensorView::create(wrong.data(), S({2, 4}), DType::F32).value();
    EXPECT(f.exe->run({{"x", bad_shape}}, {{"y", f.yv()}}).code() == Code::InvalidArgument);
    std::vector<uint16_t> half(6);
    auto bad_dtype = TensorView::create(half.data(), S({2, 3}), DType::F16).value();
    EXPECT(f.exe->run({{"x", bad_dtype}}, {{"y", f.yv()}}).code() == Code::InvalidArgument);
    Vec wide(12);
    const int64_t strides[2] = {24, 4};  // rows of 6 floats, viewing 3 of them
    auto strided = TensorView::create_strided(wide.data(), S({2, 3}), DType::F32, strides, 2).value();
    EXPECT(f.exe->run({{"x", strided}}, {{"y", f.yv()}}).code() == Code::InvalidArgument);                 // non-contiguous
}

TEST(run_validates_output_bindings) {
    Fixture f;
    EXPECT(f.exe->set_weight("w", f.wv()).is_ok());
    EXPECT(f.exe->run({{"x", f.xv()}}, {}).code() == Code::InvalidArgument);                               // no destination
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"z", f.yv()}}).code() == Code::NotFound);                         // unknown output
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", f.yv()}, {"y", f.yv()}}).code() == Code::InvalidArgument);   // twice
    Vec small(4);
    auto bad = TensorView::create(small.data(), S({2, 2}), DType::F32).value();
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", bad}}).code() == Code::InvalidArgument);
}

TEST(set_weight_validation) {
    Fixture f;
    EXPECT(f.exe->set_weight("nope", f.wv()).code() == Code::NotFound);
    EXPECT(f.exe->set_weight("x", f.xv()).code() == Code::NotFound);  // an input, not a weight
    EXPECT(f.exe->set_weight("w", f.xv()).code() == Code::InvalidArgument);  // wrong shape
}

TEST(repeated_runs_follow_current_inputs) {
    Fixture f;
    EXPECT(f.exe->set_weight("w", f.wv()).is_ok());
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", f.yv()}}).is_ok());
    Vec first = f.y;
    EXPECT(close(first, matmul_ref(f.x, f.w, 1, 2, 3, 4, true, false), kF32Tol, "first run"));
    f.x = rnd(6, 99);
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", f.yv()}}).is_ok());
    EXPECT(close(f.y, matmul_ref(f.x, f.w, 1, 2, 3, 4, true, false), kF32Tol, "second run"));
    EXPECT(max_diff(first, widen(f.y)) > 1e-3);  // outputs really changed
}

TEST(executable_outlives_backend) {
    Fixture f;
    f.backend.reset();  // the executable keeps the device alive
    EXPECT(f.exe->set_weight("w", f.wv()).is_ok());
    EXPECT(f.exe->run({{"x", f.xv()}}, {{"y", f.yv()}}).is_ok());
}

TEST(graph_outputs_can_alias_inputs_and_be_multiple) {
    Graph g;
    auto x = g.add_input("x", S({2, 3}), DType::F32).value();
    EXPECT(g.mark_output("same", x).is_ok());
    EXPECT(g.mark_output("act", g.add_gelu(x).value()).is_ok());
    auto backend = make_backend("cpu");
    if (!backend) { EXPECT(false); return; }
    auto exe = backend->compile(g);
    EXPECT(exe.is_ok());
    if (!exe.is_ok()) return;
    Vec xv = rnd(6, 40), same(6), act(6);
    EXPECT(exe.value()->run({{"x", TensorView::create(xv.data(), S({2, 3}), DType::F32).value()}},
                            {{"same", TensorView::create(same.data(), S({2, 3}), DType::F32).value()},
                             {"act", TensorView::create(act.data(), S({2, 3}), DType::F32).value()}}).is_ok());
    EXPECT(close(same, widen(xv), 0.0, "passthrough output"));
    std::vector<double> want(6);
    for (int i = 0; i < 6; i++) want[i] = gelu_ref(xv[i]);
    EXPECT(close(act, want, kF32Tol, "second output"));
}

int main() { return dooms_test::run_all(); }
