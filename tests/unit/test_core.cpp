#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "check.h"
#include "core/dtype.h"
#include "core/log.h"
#include "core/shape.h"
#include "core/stats.h"
#include "core/status.h"
#include "core/tensor.h"

using namespace dooms;

// ---- Status / Result ----

TEST(status_default_is_ok) {
    Status s;
    EXPECT(s.is_ok());
    EXPECT_EQ(s.to_string(), std::string("Ok"));
}

TEST(status_error_carries_code_and_message) {
    Status s = not_found("tensor x");
    EXPECT(!s.is_ok());
    EXPECT(s.code() == Code::NotFound);
    EXPECT_EQ(s.to_string(), std::string("NotFound: tensor x"));
}

TEST(status_ok_drops_message) {
    Status s(Code::Ok, "ignored");
    EXPECT(s.is_ok());
    EXPECT_EQ(s.message(), std::string());
}

TEST(result_holds_value_or_error) {
    Result<int> ok(7);
    EXPECT(ok.is_ok());
    EXPECT_EQ(ok.value(), 7);
    Result<int> bad(invalid_argument("nope"));
    EXPECT(!bad.is_ok());
    EXPECT(bad.status().code() == Code::InvalidArgument);
}

static Status propagate(const Status & in) {
    DOOMS_RETURN_IF_ERROR(in);
    return internal_error("reached end");
}

TEST(return_if_error_macro) {
    EXPECT(propagate(out_of_range("x")).code() == Code::OutOfRange);
    EXPECT(propagate(Status()).code() == Code::Internal);
    EXPECT(std::string(code_name(Code::IoError)) == "IoError");
    EXPECT(io_error("disk full").to_string() == "IoError: disk full");
}

// ---- DType ----

TEST(dtype_sizes_and_names_roundtrip) {
    const DType all[] = {DType::F32, DType::F16, DType::BF16, DType::I32};
    const size_t sizes[] = {4, 2, 2, 4};
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(dtype_size(all[i]), sizes[i]);
        auto back = dtype_from_name(dtype_name(all[i]));
        EXPECT(back.is_ok());
        EXPECT(back.value() == all[i]);
    }
}

TEST(dtype_float_classification) {
    EXPECT(dtype_is_float(DType::F32));
    EXPECT(dtype_is_float(DType::F16));
    EXPECT(dtype_is_float(DType::BF16));
    EXPECT(!dtype_is_float(DType::I32));
}

TEST(dtype_unknown_name_rejected) {
    EXPECT(dtype_from_name("f64").status().code() == Code::InvalidArgument);
    EXPECT(dtype_from_name("").status().code() == Code::InvalidArgument);
    EXPECT(dtype_from_name("F32").status().code() == Code::InvalidArgument);  // case-sensitive
}

// ---- Shape ----

TEST(shape_basic) {
    auto s = Shape::create({2, 3, 4});
    EXPECT(s.is_ok());
    EXPECT_EQ(s.value().rank(), 3);
    EXPECT_EQ(s.value().dim(1), int64_t(3));
    EXPECT_EQ(s.value().numel(), int64_t(24));
    EXPECT_EQ(s.value().to_string(), std::string("[2, 3, 4]"));
}

TEST(shape_scalar) {
    auto s = Shape::create({});
    EXPECT(s.is_ok());
    EXPECT_EQ(s.value().rank(), 0);
    EXPECT_EQ(s.value().numel(), int64_t(1));
    EXPECT_EQ(s.value().to_string(), std::string("[]"));
    EXPECT(Shape() == s.value());
}

TEST(shape_max_rank_accepted_and_over_rejected) {
    EXPECT(Shape::create({1, 2, 3, 4}).is_ok());
    EXPECT(Shape::create({1, 2, 3, 4, 5}).status().code() == Code::InvalidArgument);
    int64_t d[1] = {1};
    EXPECT(Shape::create(d, -1).status().code() == Code::InvalidArgument);
}

TEST(shape_rejects_nonpositive_dims) {
    EXPECT(Shape::create({2, 0}).status().code() == Code::InvalidArgument);
    EXPECT(Shape::create({-3}).status().code() == Code::InvalidArgument);
}

TEST(shape_rejects_null_dims_with_rank) {
    EXPECT(Shape::create(nullptr, 2).status().code() == Code::InvalidArgument);
    EXPECT(Shape::create(nullptr, 0).is_ok());
}

TEST(shape_numel_overflow_detected) {
    const int64_t big = std::numeric_limits<int64_t>::max() / 2 + 1;
    EXPECT(Shape::create({big, 2}).status().code() == Code::ResourceExhausted);
    EXPECT(Shape::create({big, 1}).is_ok());
}

TEST(shape_equality) {
    auto a = Shape::create({2, 3}).value();
    auto b = Shape::create({2, 3}).value();
    auto c = Shape::create({3, 2}).value();
    auto d = Shape::create({2, 3, 1}).value();
    EXPECT(a == b);
    EXPECT(a != c);
    EXPECT(a != d);
}

// ---- TensorView ----

TEST(tensor_contiguous_view) {
    std::vector<float> buf(2 * 3 * 4);
    auto shape = Shape::create({2, 3, 4}).value();
    auto t = TensorView::create(buf.data(), shape, DType::F32);
    EXPECT(t.is_ok());
    const TensorView & v = t.value();
    EXPECT_EQ(v.stride_bytes(0), int64_t(3 * 4 * 4));
    EXPECT_EQ(v.stride_bytes(1), int64_t(4 * 4));
    EXPECT_EQ(v.stride_bytes(2), int64_t(4));
    EXPECT(v.is_contiguous());
    EXPECT_EQ(v.span_bytes(), int64_t(2 * 3 * 4 * 4));
    EXPECT(v.data() == buf.data());
}

TEST(tensor_half_precision_strides) {
    std::vector<uint16_t> buf(6);
    auto shape = Shape::create({2, 3}).value();
    auto t = TensorView::create(buf.data(), shape, DType::F16).value();
    EXPECT_EQ(t.stride_bytes(0), int64_t(6));
    EXPECT_EQ(t.stride_bytes(1), int64_t(2));
}

TEST(tensor_scalar_view) {
    float x = 1.f;
    auto t = TensorView::create(&x, Shape(), DType::F32);
    EXPECT(t.is_ok());
    EXPECT_EQ(t.value().span_bytes(), int64_t(4));
    EXPECT(t.value().is_contiguous());
}

TEST(tensor_null_data_rejected) {
    auto shape = Shape::create({2}).value();
    EXPECT(TensorView::create(nullptr, shape, DType::F32).status().code() == Code::InvalidArgument);
}

TEST(tensor_strided_noncontiguous) {
    std::vector<float> buf(2 * 8);  // 2 rows of 8; view the first 4 columns
    auto shape = Shape::create({2, 4}).value();
    const int64_t strides[2] = {8 * 4, 4};
    auto t = TensorView::create_strided(buf.data(), shape, DType::F32, strides, 2);
    EXPECT(t.is_ok());
    EXPECT(!t.value().is_contiguous());
    EXPECT_EQ(t.value().span_bytes(), int64_t(32 + 3 * 4 + 4));
}

TEST(tensor_bad_strides_rejected) {
    std::vector<float> buf(8);
    auto shape = Shape::create({2, 4}).value();
    const int64_t zero[2] = {16, 0};
    const int64_t misaligned[2] = {16, 3};
    const int64_t ok[2] = {16, 4};
    EXPECT(TensorView::create_strided(buf.data(), shape, DType::F32, zero, 2).status().code() == Code::InvalidArgument);
    EXPECT(TensorView::create_strided(buf.data(), shape, DType::F32, misaligned, 2).status().code() == Code::InvalidArgument);
    EXPECT(TensorView::create_strided(buf.data(), shape, DType::F32, ok, 1).status().code() == Code::InvalidArgument);
    EXPECT(TensorView::create_strided(buf.data(), shape, DType::F32, nullptr, 2).status().code() == Code::InvalidArgument);
}

TEST(tensor_byte_size_overflow_detected) {
    float x = 0.f;
    const int64_t big = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto shape = Shape::create({big}).value();
    EXPECT(TensorView::create(&x, shape, DType::F32).status().code() == Code::ResourceExhausted);
}

// ---- Stats ----

TEST(median_and_quantile_known_values) {
    EXPECT_EQ(median({}), 0.0);
    EXPECT_EQ(median({7.0}), 7.0);
    EXPECT_EQ(median({3.0, 1.0, 2.0}), 2.0);             // unsorted input, odd count
    EXPECT_EQ(median({4.0, 1.0, 3.0, 2.0}), 3.0);        // even count: the upper median
    EXPECT_EQ(median({5.0, 5.0, 5.0, 1.0}), 5.0);        // duplicates
    EXPECT_EQ(quantile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 0.0), 1.0);
    EXPECT_EQ(quantile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 0.99), 10.0);
    EXPECT_EQ(quantile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 1.0), 10.0);   // q = 1 clamps to the last element
    EXPECT_EQ(quantile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 0.5), 6.0);
    EXPECT_EQ(quantile({-2.0, -9.0, 4.0}, 0.0), -9.0);
}

TEST(quantile_does_not_modify_its_input) {
    std::vector<double> v = {3.0, 1.0, 2.0};
    (void) quantile(v, 0.5);
    EXPECT(v == (std::vector<double>{3.0, 1.0, 2.0}));
}

// ---- Log ----

TEST(log_level_names_roundtrip) {
    const LogLevel all[] = {LogLevel::Debug, LogLevel::Info, LogLevel::Warn, LogLevel::Error, LogLevel::Off};
    for (LogLevel l : all) {
        auto back = parse_log_level(log_level_name(l));
        EXPECT(back.is_ok());
        EXPECT(back.value() == l);
    }
    EXPECT(!parse_log_level("verbose").is_ok());
}

TEST(log_filters_by_level_and_uses_sink) {
    std::vector<std::string> got;
    set_log_sink([&](LogLevel, const char * m) { got.push_back(m); });
    set_log_level(LogLevel::Warn);
    DOOMS_LOG_INFO("hidden %d", 1);
    DOOMS_LOG_WARN("shown %d", 2);
    DOOMS_LOG_ERROR("shown %s", "three");
    set_log_level(LogLevel::Off);
    DOOMS_LOG_ERROR("never");
    set_log_level(kDefaultLogLevel);
    set_log_sink(nullptr);
    EXPECT_EQ(got.size(), size_t(2));
    if (got.size() == 2) {
        EXPECT_EQ(got[0], std::string("shown 2"));
        EXPECT_EQ(got[1], std::string("shown three"));
    }
}

TEST(log_env_override) {
    set_log_level(kDefaultLogLevel);
    ::unsetenv(kLogLevelEnvVar);
    EXPECT(init_log_from_env().is_ok());
    EXPECT(log_level() == kDefaultLogLevel);

    ::setenv(kLogLevelEnvVar, "debug", 1);
    EXPECT(init_log_from_env().is_ok());
    EXPECT(log_level() == LogLevel::Debug);

    ::setenv(kLogLevelEnvVar, "bogus", 1);
    Status s = init_log_from_env();
    EXPECT(s.code() == Code::InvalidArgument);
    EXPECT(log_level() == LogLevel::Debug);  // unchanged on error

    ::unsetenv(kLogLevelEnvVar);
    set_log_level(kDefaultLogLevel);
}

int main() { return dooms_test::run_all(); }
