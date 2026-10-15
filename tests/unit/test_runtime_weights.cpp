// Tests load_weights(): GGUF tensors bound into a compiled graph and run on the ggml backend.
// References are computed here in double precision, independent of the code under test.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "backend/ggml/ggml_backend.h"
#include "check.h"
#include "gguf_builder.h"
#include "runtime/weights.h"

using namespace dooms;
using namespace dooms_test_gguf;
using Vec = std::vector<float>;

namespace {

// f32 accumulation order differs from the double reference; the tolerance scales with output magnitude.
constexpr double kRelTol = 1e-4;

Shape S(std::initializer_list<int64_t> d) { return Shape::create(d).value(); }

Vec rnd(size_t n, uint32_t seed, float scale = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    Vec v(n);
    for (float & x : v) x = dist(gen);
    return v;
}

// A tensor as the converter stores it: `dims` is the row-major shape, the file lists it reversed.
TensorSpec f32_tensor(const std::string & name, std::vector<int64_t> dims, const Vec & data, int32_t type = kTypeF32) {
    std::vector<int64_t> file_dims(dims.rbegin(), dims.rend());
    Bytes bytes(data.size() * (type == kTypeF16 ? 2 : 4), 0);
    if (type == kTypeF32) std::memcpy(bytes.data(), data.data(), bytes.size());
    return {name, file_dims, type, bytes, std::nullopt};
}

struct Compiled {
    std::unique_ptr<Backend> backend;
    std::unique_ptr<Executable> exe;
};

Compiled compile_on(const Graph & g, const char * device = "cpu") {
    GgmlBackendOptions o;
    o.device = device;
    Compiled c;
    c.backend = std::move(create_ggml_backend(o)).value();
    c.exe = std::move(c.backend->compile(g)).value();
    return c;
}

// y = gelu(x W1^T) W2^T, x[2,8], W1[16,8], W2[4,16]
Graph mlp_graph() {
    Graph g;
    auto x = g.add_input("x", S({2, 8}), DType::F32).value();
    auto w1 = g.add_weight("w1", S({16, 8}), DType::F32).value();
    auto w2 = g.add_weight("w2", S({4, 16}), DType::F32).value();
    g.mark_output("y", g.add_matmul(g.add_gelu(g.add_matmul(x, w1, true).value()).value(), w2, true).value());
    return g;
}

double gelu_ref(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

std::vector<double> mlp_ref(const Vec & x, const Vec & w1, const Vec & w2) {
    std::vector<double> h(2 * 16), y(2 * 4);
    for (int b = 0; b < 2; b++)
        for (int j = 0; j < 16; j++) {
            double s = 0;
            for (int k = 0; k < 8; k++) s += static_cast<double>(x[b * 8 + k]) * w1[j * 8 + k];
            h[b * 16 + j] = gelu_ref(s);
        }
    for (int b = 0; b < 2; b++)
        for (int j = 0; j < 4; j++) {
            double s = 0;
            for (int k = 0; k < 16; k++) s += h[b * 16 + k] * w2[j * 16 + k];
            y[b * 4 + j] = s;
        }
    return y;
}

double max_rel_err(const Vec & got, const std::vector<double> & want) {
    if (got.size() != want.size()) return 1e30;
    double scale = 1.0, err = 0.0;
    for (double w : want) scale = std::max(scale, std::fabs(w));
    for (size_t i = 0; i < got.size(); i++) err = std::max(err, std::fabs(got[i] - want[i]));
    return err / scale;
}

Result<Vec> run_mlp(Executable & exe, const Vec & x) {
    Vec xv = x, y(2 * 4);
    DOOMS_RETURN_IF_ERROR(exe.run({{"x", TensorView::create(xv.data(), S({2, 8}), DType::F32).value()}},
                                  {{"y", TensorView::create(y.data(), S({2, 4}), DType::F32).value()}}));
    return y;
}

Bytes mlp_file(const Vec & w1, const Vec & w2, std::vector<TensorSpec> extra = {}) {
    Builder b;
    b.kvs.push_back(kv_string("general.architecture", "test"));
    b.tensors.push_back(f32_tensor("w1", {16, 8}, w1));
    b.tensors.push_back(f32_tensor("w2", {4, 16}, w2));
    for (auto & t : extra) b.tensors.push_back(std::move(t));
    return b.build();
}

}  // namespace

namespace {

Bytes one_tensor_file(const std::string & name, std::vector<int64_t> dims, const Vec & data, std::vector<TensorSpec> extra = {}) {
    Builder b;
    b.kvs.push_back(kv_string("general.architecture", "test"));
    b.tensors.push_back(f32_tensor(name, dims, data));
    for (auto & t : extra) b.tensors.push_back(std::move(t));
    return b.build();
}

}  // namespace

TEST(weights_can_come_from_several_files) {
    Vec w1 = rnd(16 * 8, 1), w2 = rnd(4 * 16, 2), x = rnd(2 * 8, 3);
    Bytes fa = one_tensor_file("w1", {16, 8}, w1, {f32_tensor("only_in_a", {2}, Vec{1, 2})});
    Bytes fb = one_tensor_file("w2", {4, 16}, w2, {f32_tensor("only_in_b", {3}, Vec{1, 2, 3})});
    auto a = GgufFile::parse(fa.data(), fa.size()).value();
    auto b = GgufFile::parse(fb.data(), fb.size()).value();
    Compiled c = compile_on(mlp_graph());
    auto report = load_weights({&a, &b}, *c.exe);
    EXPECT(report.is_ok());
    if (!report.is_ok()) { std::fprintf(stderr, "  %s\n", report.status().to_string().c_str()); return; }
    EXPECT_EQ(report.value().loaded, size_t(2));
    EXPECT_EQ(report.value().unused_file_tensors.size(), size_t(2));
    EXPECT(report.value().unused_file_tensors[0] == "only_in_a" && report.value().unused_file_tensors[1] == "only_in_b");
    auto y = run_mlp(*c.exe, x);
    EXPECT(y.is_ok());
    if (y.is_ok()) EXPECT(max_rel_err(y.value(), mlp_ref(x, w1, w2)) < kRelTol);
}

TEST(the_first_file_wins_and_the_loser_is_reported_unused) {
    Vec w1 = rnd(16 * 8, 1), w2 = rnd(4 * 16, 2), other_w2 = rnd(4 * 16, 9), x = rnd(2 * 8, 3);
    Bytes fa = mlp_file(w1, w2);
    Bytes fb = one_tensor_file("w2", {4, 16}, other_w2);
    auto a = GgufFile::parse(fa.data(), fa.size()).value();
    auto b = GgufFile::parse(fb.data(), fb.size()).value();
    Compiled c = compile_on(mlp_graph());
    auto report = load_weights({&a, &b}, *c.exe);
    EXPECT(report.is_ok());
    if (!report.is_ok()) return;
    EXPECT_EQ(report.value().unused_file_tensors.size(), size_t(1));  // b's w2
    auto y = run_mlp(*c.exe, x);
    EXPECT(y.is_ok());
    if (y.is_ok()) EXPECT(max_rel_err(y.value(), mlp_ref(x, w1, w2)) < kRelTol);  // a's weights, not b's
}

TEST(multi_file_loading_reports_missing_weights_and_bad_arguments) {
    Vec w1 = rnd(16 * 8, 1);
    Bytes fa = one_tensor_file("w1", {16, 8}, w1);
    auto a = GgufFile::parse(fa.data(), fa.size()).value();
    Compiled c = compile_on(mlp_graph());
    auto missing = load_weights({&a}, *c.exe);
    EXPECT(missing.status().code() == Code::NotFound);
    EXPECT(missing.status().message().find("w2") != std::string::npos);
    EXPECT(load_weights(std::vector<const GgufFile *>{}, *c.exe).status().code() == Code::InvalidArgument);
    EXPECT(load_weights({&a, nullptr}, *c.exe).status().code() == Code::InvalidArgument);
}

TEST(loads_weights_and_the_graph_computes_with_them) {
    Vec w1 = rnd(16 * 8, 1), w2 = rnd(4 * 16, 2), x = rnd(2 * 8, 3);
    Bytes file = mlp_file(w1, w2);
    auto gguf = GgufFile::parse(file.data(), file.size());
    EXPECT(gguf.is_ok());
    if (!gguf.is_ok()) return;
    Compiled c = compile_on(mlp_graph());
    auto report = load_weights(gguf.value(), *c.exe);
    EXPECT(report.is_ok());
    if (!report.is_ok()) return;
    EXPECT_EQ(report.value().loaded, size_t(2));
    EXPECT_EQ(report.value().bytes, uint64_t((16 * 8 + 4 * 16) * 4));
    EXPECT(report.value().unused_file_tensors.empty());
    auto y = run_mlp(*c.exe, x);
    EXPECT(y.is_ok());
    if (y.is_ok()) EXPECT(max_rel_err(y.value(), mlp_ref(x, w1, w2)) < kRelTol);
}

TEST(unused_file_tensors_are_reported_in_file_order) {
    Vec w1 = rnd(16 * 8, 1), w2 = rnd(4 * 16, 2);
    Bytes file = mlp_file(w1, w2, {f32_tensor("zeta", {2}, Vec{1, 2}), f32_tensor("alpha", {3}, Vec{1, 2, 3})});
    auto gguf = GgufFile::parse(file.data(), file.size()).value();
    Compiled c = compile_on(mlp_graph());
    auto report = load_weights(gguf, *c.exe);
    EXPECT(report.is_ok());
    if (!report.is_ok()) return;
    EXPECT_EQ(report.value().unused_file_tensors.size(), size_t(2));
    if (report.value().unused_file_tensors.size() == 2) {
        EXPECT_EQ(report.value().unused_file_tensors[0], std::string("zeta"));
        EXPECT_EQ(report.value().unused_file_tensors[1], std::string("alpha"));
    }
}

TEST(missing_weight_is_not_found_and_the_executable_refuses_to_run) {
    Builder b;
    b.tensors.push_back(f32_tensor("w1", {16, 8}, rnd(16 * 8, 1)));  // w2 absent
    Bytes file = b.build();
    auto gguf = GgufFile::parse(file.data(), file.size()).value();
    Compiled c = compile_on(mlp_graph());
    auto report = load_weights(gguf, *c.exe);
    EXPECT(!report.is_ok());
    if (!report.is_ok()) {
        EXPECT(report.status().code() == Code::NotFound);
        EXPECT(report.status().message().find("w2") != std::string::npos);
    }
    EXPECT(run_mlp(*c.exe, rnd(16, 3)).status().code() == Code::InvalidArgument);  // w2 was never set
}

TEST(transposed_weight_shape_is_rejected) {
    // The classic silent mistake: [in, out] stored where the graph expects [out, in].
    Builder b;
    b.tensors.push_back(f32_tensor("w1", {8, 16}, rnd(16 * 8, 1)));
    b.tensors.push_back(f32_tensor("w2", {4, 16}, rnd(4 * 16, 2)));
    Bytes file = b.build();
    auto gguf = GgufFile::parse(file.data(), file.size()).value();
    Compiled c = compile_on(mlp_graph());
    auto report = load_weights(gguf, *c.exe);
    EXPECT(!report.is_ok());
    if (!report.is_ok()) {
        EXPECT(report.status().code() == Code::InvalidArgument);
        EXPECT(report.status().message().find("w1") != std::string::npos);
    }
}

TEST(dtype_mismatch_is_rejected_not_converted) {
    Builder b;
    Bytes half(16 * 8 * 2, 0);
    b.tensors.push_back({"w1", {8, 16}, kTypeF16, half, std::nullopt});
    b.tensors.push_back(f32_tensor("w2", {4, 16}, rnd(4 * 16, 2)));
    Bytes file = b.build();
    auto gguf = GgufFile::parse(file.data(), file.size()).value();
    Compiled c = compile_on(mlp_graph());
    EXPECT(load_weights(gguf, *c.exe).status().code() == Code::InvalidArgument);
}

TEST(reloading_replaces_the_previous_weights) {
    Vec w1a = rnd(16 * 8, 1), w2a = rnd(4 * 16, 2), w1b = rnd(16 * 8, 11), w2b = rnd(4 * 16, 12), x = rnd(2 * 8, 3);
    Bytes fa = mlp_file(w1a, w2a), fb = mlp_file(w1b, w2b);
    auto ga = GgufFile::parse(fa.data(), fa.size()).value();
    auto gb = GgufFile::parse(fb.data(), fb.size()).value();
    Compiled c = compile_on(mlp_graph());
    EXPECT(load_weights(ga, *c.exe).is_ok());
    auto ya = run_mlp(*c.exe, x);
    EXPECT(load_weights(gb, *c.exe).is_ok());
    auto yb = run_mlp(*c.exe, x);
    EXPECT(ya.is_ok() && yb.is_ok());
    if (!ya.is_ok() || !yb.is_ok()) return;
    EXPECT(max_rel_err(ya.value(), mlp_ref(x, w1a, w2a)) < kRelTol);
    EXPECT(max_rel_err(yb.value(), mlp_ref(x, w1b, w2b)) < kRelTol);
    EXPECT(max_rel_err(ya.value(), std::vector<double>(yb.value().begin(), yb.value().end())) > 1e-2);  // really different
}

TEST(graph_without_weights_loads_nothing) {
    Graph g;
    auto x = g.add_input("x", S({2, 8}), DType::F32).value();
    g.mark_output("y", g.add_gelu(x).value());
    Bytes file = mlp_file(rnd(16 * 8, 1), rnd(4 * 16, 2));
    auto gguf = GgufFile::parse(file.data(), file.size()).value();
    Compiled c = compile_on(g);
    auto report = load_weights(gguf, *c.exe);
    EXPECT(report.is_ok());
    if (report.is_ok()) {
        EXPECT_EQ(report.value().loaded, size_t(0));
        EXPECT_EQ(report.value().unused_file_tensors.size(), size_t(2));
    }
}

TEST(weights_are_copied_so_the_file_can_go_away) {
    Vec w1 = rnd(16 * 8, 1), w2 = rnd(4 * 16, 2), x = rnd(2 * 8, 3);
    const auto path = std::filesystem::temp_directory_path() / "dooms_test_weights.gguf";
    {
        Bytes file = mlp_file(w1, w2);
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
    }
    Compiled c = compile_on(mlp_graph());
    {
        auto gguf = GgufFile::open(path.string());  // memory-mapped; unmapped when this scope ends
        EXPECT(gguf.is_ok());
        if (!gguf.is_ok()) return;
        EXPECT(load_weights(gguf.value(), *c.exe).is_ok());
    }
    std::filesystem::remove(path);
    auto y = run_mlp(*c.exe, x);
    EXPECT(y.is_ok());
    if (y.is_ok()) EXPECT(max_rel_err(y.value(), mlp_ref(x, w1, w2)) < kRelTol);
}

TEST(weights_load_on_the_best_device_too) {
    Vec w1 = rnd(16 * 8, 1), w2 = rnd(4 * 16, 2), x = rnd(2 * 8, 3);
    Bytes file = mlp_file(w1, w2);
    auto gguf = GgufFile::parse(file.data(), file.size()).value();
    Compiled c = compile_on(mlp_graph(), "auto");
    EXPECT(load_weights(gguf, *c.exe).is_ok());
    auto y = run_mlp(*c.exe, x);
    EXPECT(y.is_ok());
    if (y.is_ok()) EXPECT(max_rel_err(y.value(), mlp_ref(x, w1, w2)) < 1e-3);  // GPU accumulation order differs
}


int main() { return dooms_test::run_all(); }
