// Tests for the GGUF reader. Valid files come from three independent sources: a byte-level builder
// written here (also used to craft corrupt files), ggml's own GGUF writer, and ggml's reader as a
// differential oracle. Hostile-input behaviour (truncation at every byte, random corruption) is
// checked for "no crash, no out-of-bounds"; run scripts/check-asan.sh to enforce that with sanitizers.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "gguf_builder.h"
#include "format/gguf.h"
#include "ggml.h"
#include "gguf.h"

using namespace dooms;
using dooms_test_gguf::Bytes;

namespace {

using namespace dooms_test_gguf;

// A representative valid file: many metadata types, every supported dtype, scalar and 4-D tensors.
Builder base_builder() {
    Builder b;
    b.kvs.push_back(kv_string("general.architecture", "test"));
    b.kvs.push_back(kv<uint32_t>("test.u32", GgufValueType::U32, 7));
    b.kvs.push_back(kv<int32_t>("test.i32", GgufValueType::I32, -5));
    b.kvs.push_back(kv<float>("test.f32", GgufValueType::F32, 1.5f));
    b.kvs.push_back(kv<double>("test.f64", GgufValueType::F64, 2.25));
    b.kvs.push_back(kv<uint8_t>("test.bool", GgufValueType::Bool, 1));
    b.kvs.push_back(kv<uint64_t>("test.u64", GgufValueType::U64, 0xFFFFFFFFFFFFFFFFull));
    {
        Bytes a = kv_head("test.strs", GgufValueType::Array);
        put<int32_t>(a, static_cast<int32_t>(GgufValueType::String));
        put<uint64_t>(a, 2);
        put_str(a, "a");
        put_str(a, "bc");
        b.kvs.push_back(a);
    }
    {
        Bytes a = kv_head("test.nums", GgufValueType::Array);
        put<int32_t>(a, static_cast<int32_t>(GgufValueType::U32));
        put<uint64_t>(a, 3);
        for (uint32_t v : {1u, 2u, 3u}) put<uint32_t>(a, v);
        b.kvs.push_back(a);
    }
    b.tensors.push_back({"a", {3, 2}, kTypeF32, raw_bytes(std::vector<float>{0, 1, 2, 3, 4, 5}), std::nullopt});
    b.tensors.push_back({"h", {4}, kTypeF16, raw_bytes(std::vector<uint16_t>{1, 2, 3, 4}), std::nullopt});
    b.tensors.push_back({"i", {2, 2, 2}, kTypeI32, raw_bytes(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}), std::nullopt});
    b.tensors.push_back({"s", {}, kTypeF32, raw_bytes(std::vector<float>{42.f}), std::nullopt});
    b.tensors.push_back({"bf", {2}, kTypeBF16, raw_bytes(std::vector<uint16_t>{9, 10}), std::nullopt});
    std::vector<float> big(5 * 4 * 3 * 2);
    for (size_t i = 0; i < big.size(); i++) big[i] = static_cast<float>(i);
    b.tensors.push_back({"d4", {5, 4, 3, 2}, kTypeF32, raw_bytes(big), std::nullopt});
    return b;
}

Result<GgufFile> parse(const Bytes & b, const GgufLimits & lim = GgufLimits()) {
    return GgufFile::parse(b.data(), b.size(), lim);
}
Code code_of(const Bytes & b, const GgufLimits & lim = GgufLimits()) {
    auto r = parse(b, lim);
    return r.is_ok() ? Code::Ok : r.status().code();
}

std::filesystem::path temp_path(const char * name) { return std::filesystem::temp_directory_path() / name; }

void write_file(const std::filesystem::path & p, const Bytes & b) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size()));
}

}  // namespace

// ---- valid files ----

TEST(parses_metadata_and_tensors_of_a_valid_file) {
    Builder b = base_builder();
    Bytes bytes = b.build();
    auto r = parse(bytes);
    EXPECT(r.is_ok());
    if (!r.is_ok()) { std::fprintf(stderr, "  %s\n", r.status().to_string().c_str()); return; }
    const GgufFile & f = r.value();
    EXPECT_EQ(f.version(), uint32_t(3));
    EXPECT_EQ(f.alignment(), uint64_t(32));
    EXPECT_EQ(f.tensors().size(), size_t(6));
    EXPECT_EQ(f.metadata().size(), size_t(9));
    EXPECT_EQ(f.get_string("general.architecture").value(), std::string("test"));
}

TEST(file_dims_are_reversed_into_row_major_shapes) {
    auto f = parse(base_builder().build()).value();
    EXPECT_EQ(f.find_tensor("a").value()->shape.to_string(), std::string("[2, 3]"));      // file {3,2}
    EXPECT_EQ(f.find_tensor("i").value()->shape.to_string(), std::string("[2, 2, 2]"));
    EXPECT_EQ(f.find_tensor("d4").value()->shape.to_string(), std::string("[2, 3, 4, 5]")); // file {5,4,3,2}
    EXPECT_EQ(f.find_tensor("h").value()->shape.to_string(), std::string("[4]"));
}

TEST(scalar_tensor_has_rank_zero) {
    auto f = parse(base_builder().build()).value();
    const GgufTensorInfo * s = f.find_tensor("s").value();
    EXPECT_EQ(s->shape.rank(), 0);
    EXPECT_EQ(s->nbytes, uint64_t(4));
}

TEST(dtypes_and_byte_sizes) {
    auto f = parse(base_builder().build()).value();
    EXPECT(f.find_tensor("a").value()->dtype == DType::F32);
    EXPECT(f.find_tensor("h").value()->dtype == DType::F16);
    EXPECT(f.find_tensor("i").value()->dtype == DType::I32);
    EXPECT(f.find_tensor("bf").value()->dtype == DType::BF16);
    EXPECT_EQ(f.find_tensor("a").value()->nbytes, uint64_t(24));
    EXPECT_EQ(f.find_tensor("h").value()->nbytes, uint64_t(8));
    EXPECT_EQ(f.find_tensor("d4").value()->nbytes, uint64_t(5 * 4 * 3 * 2 * 4));
}

TEST(tensor_bytes_come_from_the_right_place) {
    Bytes bytes = base_builder().build();
    auto f = parse(bytes).value();
    const float * a = reinterpret_cast<const float *>(f.tensor_data(*f.find_tensor("a").value()));
    for (int i = 0; i < 6; i++) EXPECT_EQ(a[i], static_cast<float>(i));
    const float * d4 = reinterpret_cast<const float *>(f.tensor_data(*f.find_tensor("d4").value()));
    EXPECT_EQ(d4[0], 0.f);
    EXPECT_EQ(d4[119], 119.f);
    const int32_t * iv = reinterpret_cast<const int32_t *>(f.tensor_data(*f.find_tensor("i").value()));
    EXPECT_EQ(iv[7], 8);
    auto view = f.tensor_view("a");
    EXPECT(view.is_ok());
    EXPECT(view.value().shape() == Shape::create({2, 3}).value());
    EXPECT(view.value().dtype() == DType::F32);
    EXPECT(view.value().is_contiguous());
}

TEST(every_tensor_offset_is_aligned_and_in_bounds) {
    Bytes bytes = base_builder().build();
    auto f = parse(bytes).value();
    for (const GgufTensorInfo & t : f.tensors()) {
        EXPECT_EQ(t.offset % f.alignment(), uint64_t(0));
        EXPECT(t.offset + t.nbytes <= bytes.size());
    }
}

TEST(custom_alignment_is_honoured) {
    Builder b = base_builder();
    b.alignment = 128;
    Bytes bytes = b.build();
    auto f = parse(bytes);
    EXPECT(f.is_ok());
    if (!f.is_ok()) return;
    EXPECT_EQ(f.value().alignment(), uint64_t(128));
    for (const GgufTensorInfo & t : f.value().tensors()) EXPECT_EQ(t.offset % 128, uint64_t(0));
}

TEST(version_2_is_accepted) {
    Builder b = base_builder();
    b.version = 2;
    EXPECT(code_of(b.build()) == Code::Ok);
}

TEST(metadata_without_tensors_is_valid) {
    Builder b;
    b.kvs.push_back(kv_string("general.architecture", "x"));
    auto f = parse(b.build());
    EXPECT(f.is_ok());
    if (f.is_ok()) EXPECT_EQ(f.value().tensors().size(), size_t(0));
}

// ---- typed getters ----

TEST(typed_getters) {
    auto f = parse(base_builder().build()).value();
    EXPECT_EQ(f.get_uint("test.u32").value(), uint64_t(7));
    EXPECT_EQ(f.get_int("test.u32").value(), int64_t(7));
    EXPECT_EQ(f.get_int("test.i32").value(), int64_t(-5));
    EXPECT_EQ(f.get_float("test.f32").value(), 1.5);
    EXPECT_EQ(f.get_float("test.f64").value(), 2.25);
    EXPECT(f.get_bool("test.bool").value());
    EXPECT_EQ(f.get_uint("test.u64").value(), 0xFFFFFFFFFFFFFFFFull);
}

TEST(getter_errors) {
    auto f = parse(base_builder().build()).value();
    EXPECT(f.get_string("nope").status().code() == Code::NotFound);
    EXPECT(f.get_uint("nope").status().code() == Code::NotFound);
    EXPECT(f.get_uint("test.i32").status().code() == Code::OutOfRange);          // negative
    EXPECT(f.get_int("test.u64").status().code() == Code::OutOfRange);           // does not fit int64
    EXPECT(f.get_uint("general.architecture").status().code() == Code::InvalidArgument);
    EXPECT(f.get_string("test.u32").status().code() == Code::InvalidArgument);
    EXPECT(f.get_float("test.u32").status().code() == Code::InvalidArgument);    // no silent int->float
    EXPECT(f.get_bool("test.u32").status().code() == Code::InvalidArgument);
    EXPECT(f.get_uint("test.nums").status().code() == Code::InvalidArgument);    // arrays are not scalars
}

TEST(arrays_are_parsed) {
    auto f = parse(base_builder().build()).value();
    const GgufValue & strs = f.metadata().at("test.strs");
    EXPECT(strs.type == GgufValueType::Array);
    EXPECT(strs.element_type == GgufValueType::String);
    EXPECT_EQ(strs.items.size(), size_t(2));
    EXPECT_EQ(strs.items[1].s, std::string("bc"));
    const GgufValue & nums = f.metadata().at("test.nums");
    EXPECT_EQ(nums.items.size(), size_t(3));
    EXPECT_EQ(nums.items[2].u, uint64_t(3));
}

TEST(tensor_lookup_errors) {
    auto f = parse(base_builder().build()).value();
    EXPECT(f.find_tensor("missing").status().code() == Code::NotFound);
    EXPECT(f.tensor_view("missing").status().code() == Code::NotFound);
}

// ---- rejected headers ----

TEST(rejects_empty_input_and_bad_magic) {
    EXPECT(GgufFile::parse(nullptr, 0).status().code() == Code::InvalidArgument);
    uint8_t one = 0;
    EXPECT(GgufFile::parse(&one, 0).status().code() == Code::InvalidArgument);
    Bytes bytes = base_builder().build();
    bytes[0] = 'X';
    EXPECT(code_of(bytes) == Code::DataLoss);
    EXPECT(code_of(Bytes{'G', 'G'}) == Code::DataLoss);  // shorter than the magic
}

TEST(rejects_unsupported_versions) {
    for (uint32_t v : {0u, 1u, 4u, 0x03000000u, 0xFFFFFFFFu}) {
        Builder b = base_builder();
        b.version = v;
        EXPECT(code_of(b.build()) == Code::Unsupported);
    }
}

TEST(endianness_hint_in_version_error) {
    Builder b = base_builder();
    b.version = 0x03000000;  // version 3 byte-swapped
    auto r = parse(b.build());
    EXPECT(!r.is_ok());
    EXPECT(r.status().message().find("endianness") != std::string::npos);
}

// ---- rejected metadata ----

TEST(rejects_duplicate_and_empty_keys) {
    Builder b = base_builder();
    b.kvs.push_back(kv<uint32_t>("test.u32", GgufValueType::U32, 9));
    EXPECT(code_of(b.build()) == Code::DataLoss);
    Builder e;
    e.kvs.push_back(kv<uint32_t>("", GgufValueType::U32, 1));
    EXPECT(code_of(e.build()) == Code::DataLoss);
}

TEST(rejects_bad_values) {
    Builder b;
    b.kvs.push_back(kv<uint8_t>("flag", GgufValueType::Bool, 2));
    EXPECT(code_of(b.build()) == Code::DataLoss);                        // bool must be 0 or 1

    Builder t;
    Bytes k = kv_head("x", GgufValueType::U32);
    k.resize(k.size() - 4);
    Bytes raw = k;
    raw.resize(0);
    put_str(raw, "x");
    put<int32_t>(raw, 99);                                               // unknown value type
    t.kvs.push_back(raw);
    EXPECT(code_of(t.build()) == Code::Unsupported);
}

TEST(rejects_nested_arrays_and_hostile_array_sizes) {
    Builder nested;
    Bytes a = kv_head("n", GgufValueType::Array);
    put<int32_t>(a, static_cast<int32_t>(GgufValueType::Array));
    put<uint64_t>(a, 1);
    nested.kvs.push_back(a);
    EXPECT(code_of(nested.build()) == Code::Unsupported);

    Builder huge;
    Bytes h = kv_head("h", GgufValueType::Array);
    put<int32_t>(h, static_cast<int32_t>(GgufValueType::U32));
    put<uint64_t>(h, uint64_t(1) << 40);                                 // far above the limit
    huge.kvs.push_back(h);
    EXPECT(code_of(huge.build()) == Code::ResourceExhausted);

    Builder liar;
    Bytes l = kv_head("l", GgufValueType::Array);
    put<int32_t>(l, static_cast<int32_t>(GgufValueType::U32));
    put<uint64_t>(l, 1000);                                              // within the limit, but the file has no such data
    liar.kvs.push_back(l);
    EXPECT(code_of(liar.build()) == Code::DataLoss);
}

TEST(rejects_hostile_string_lengths) {
    Builder b;
    Bytes s = kv_head("s", GgufValueType::String);
    put<uint64_t>(s, uint64_t(1) << 60);
    b.kvs.push_back(s);
    EXPECT(code_of(b.build()) == Code::ResourceExhausted);

    Builder w;
    Bytes t = kv_head("t", GgufValueType::String);
    put<uint64_t>(t, 1000);                                              // under the limit, over the file
    w.kvs.push_back(t);
    EXPECT(code_of(w.build()) == Code::DataLoss);
}

TEST(rejects_hostile_counts) {
    Builder many_tensors = base_builder();
    many_tensors.n_tensors_override = uint64_t(1) << 40;
    EXPECT(code_of(many_tensors.build()) == Code::ResourceExhausted);
    Builder many_kv = base_builder();
    many_kv.n_kv_override = uint64_t(1) << 40;
    EXPECT(code_of(many_kv.build()) == Code::ResourceExhausted);
    Builder liar = base_builder();
    liar.n_tensors_override = 5000;                                      // within limits, more than the file holds
    EXPECT(code_of(liar.build()) == Code::DataLoss);
    Builder liar_kv = base_builder();
    liar_kv.n_kv_override = 5000;  // the reader then parses the tensor infos as metadata: any corruption-class error is right
    const Code c = code_of(liar_kv.build());
    EXPECT(c == Code::DataLoss || c == Code::Unsupported || c == Code::ResourceExhausted);
}

TEST(custom_limits_are_enforced) {
    Bytes bytes = base_builder().build();
    GgufLimits few_tensors;
    few_tensors.max_tensors = 2;
    EXPECT(code_of(bytes, few_tensors) == Code::ResourceExhausted);
    GgufLimits few_kv;
    few_kv.max_kv = 3;
    EXPECT(code_of(bytes, few_kv) == Code::ResourceExhausted);
    GgufLimits short_names;
    short_names.max_name_bytes = 4;                                      // "general.architecture" is longer
    EXPECT(code_of(bytes, short_names) == Code::ResourceExhausted);
    GgufLimits small_arrays;
    small_arrays.max_array_elements = 2;                                 // test.nums has 3
    EXPECT(code_of(bytes, small_arrays) == Code::ResourceExhausted);
    GgufLimits small_strings;
    small_strings.max_string_bytes = 3;                                  // "general.architecture" value "test" is 4
    EXPECT(code_of(bytes, small_strings) == Code::ResourceExhausted);
}

TEST(rejects_bad_alignment_values) {
    for (uint32_t a : {0u, 3u, 48u, 1u << 30}) {
        Builder b;  // metadata only: the alignment is validated before any tensor is read
        b.kvs.push_back(kv<uint32_t>("general.alignment", GgufValueType::U32, a));
        EXPECT(code_of(b.build()) == Code::DataLoss);
    }
    Builder wrong_type;
    wrong_type.kvs.push_back(kv<uint64_t>("general.alignment", GgufValueType::U64, 32));
    EXPECT(code_of(wrong_type.build()) == Code::DataLoss);
}

TEST(accepts_the_largest_allowed_alignment_and_small_ones) {
    for (uint32_t a : {1u, 2u, 16u, 4096u}) {
        Builder b = base_builder();
        b.alignment = a;
        EXPECT(code_of(b.build()) == Code::Ok);
    }
}

// ---- rejected tensors ----

static Bytes with_tensor(TensorSpec t) {
    Builder b;
    b.tensors.push_back(std::move(t));
    return b.build();
}

TEST(rejects_bad_tensor_names) {
    EXPECT(code_of(with_tensor({"", {1}, kTypeF32, raw_bytes(std::vector<float>{1}), std::nullopt})) == Code::DataLoss);
    Builder dup;
    dup.tensors.push_back({"w", {1}, kTypeF32, raw_bytes(std::vector<float>{1}), std::nullopt});
    dup.tensors.push_back({"w", {1}, kTypeF32, raw_bytes(std::vector<float>{2}), std::nullopt});
    EXPECT(code_of(dup.build()) == Code::DataLoss);
}

TEST(rejects_bad_tensor_shapes) {
    const Bytes four = raw_bytes(std::vector<float>(16, 0.f));
    EXPECT(code_of(with_tensor({"t", {1, 1, 1, 1, 1}, kTypeF32, four, std::nullopt})) == Code::DataLoss);   // rank 5
    EXPECT(code_of(with_tensor({"t", {4, 0}, kTypeF32, four, std::nullopt})) == Code::DataLoss);           // zero dim
    EXPECT(code_of(with_tensor({"t", {-3}, kTypeF32, four, std::nullopt})) == Code::DataLoss);              // negative dim
    const int64_t big = std::numeric_limits<int64_t>::max() / 2 + 1;
    EXPECT(code_of(with_tensor({"t", {big, 2}, kTypeF32, four, std::nullopt})) == Code::DataLoss);          // numel overflow
    EXPECT(code_of(with_tensor({"t", {big}, kTypeF32, four, std::nullopt})) == Code::DataLoss);             // bytes exceed file
}

TEST(rejects_unsupported_tensor_types) {
    const Bytes data = raw_bytes(std::vector<float>(32, 0.f));
    for (int32_t type : {3 /* q4_1 */, 7, 40, -1, 9999}) {
        auto r = parse(with_tensor({"t", {32}, type, data, std::nullopt}));
        EXPECT(!r.is_ok());
        if (!r.is_ok()) EXPECT(r.status().code() == Code::Unsupported);
    }
}

// The block types the Metal text decoder reads: sized by whole blocks per row; a row that is not a whole number of blocks is DataLoss.
TEST(quantized_tensors_are_sized_by_blocks) {
    const Bytes data(4096, 0);
    struct Case { int32_t type; int64_t row; uint64_t bytes; DType dtype; };
    for (const Case & k : {Case{kTypeQ4_0, 64, 36, DType::Q4_0}, Case{6, 32, 22, DType::Q5_0}, Case{8, 96, 102, DType::Q8_0}, Case{12, 256, 144, DType::Q4_K},
                           Case{14, 512, 420, DType::Q6_K}}) {
        auto r = parse(with_tensor({"t", {k.row, 2}, k.type, Bytes(data.begin(), data.begin() + static_cast<long>(2 * k.bytes)), std::nullopt}));
        EXPECT(r.is_ok());
        if (!r.is_ok()) { std::printf("        %s\n", r.status().to_string().c_str()); continue; }
        const GgufTensorInfo & t = r.value().tensors()[0];
        EXPECT(t.dtype == k.dtype && t.nbytes == 2 * k.bytes);
        EXPECT(dtype_is_quantized(t.dtype) && dtype_bytes(t.dtype, k.row).value() == k.bytes && !dtype_bytes(t.dtype, k.row + 1).is_ok());
    }
    EXPECT(code_of(with_tensor({"t", {33}, kTypeQ4_0, data, std::nullopt})) == Code::DataLoss);    // 33 is not a whole number of 32-element blocks
    EXPECT(code_of(with_tensor({"t", {128}, 12, data, std::nullopt})) == Code::DataLoss);         // q4_K blocks hold 256
    EXPECT(!dtype_is_quantized(DType::F16) && dtype_block_elems(DType::F32) == 1 && dtype_bytes(DType::F16, 3).value() == 6);
}

TEST(rejects_bad_tensor_offsets) {
    const Bytes data = raw_bytes(std::vector<float>(8, 1.f));
    // Unaligned but otherwise in range: extra bytes after the data keep the end-of-file check from firing,
    // so this isolates the alignment check.
    Bytes roomy = with_tensor({"t", {8}, kTypeF32, data, 4});
    roomy.resize(roomy.size() + 64, 0);
    EXPECT(code_of(roomy) == Code::DataLoss);
    EXPECT(code_of(with_tensor({"t", {8}, kTypeF32, data, 7})) == Code::DataLoss);                          // unaligned (and also too long)
    EXPECT(code_of(with_tensor({"t", {8}, kTypeF32, data, 4096})) == Code::DataLoss);                       // past the end
    EXPECT(code_of(with_tensor({"t", {8}, kTypeF32, data, std::numeric_limits<uint64_t>::max() - 31})) == Code::DataLoss);  // wraps
    EXPECT(code_of(with_tensor({"t", {8}, kTypeF32, data, 32})) == Code::DataLoss);                         // runs off the end
}

TEST(rejects_overlapping_tensors) {
    Builder b;
    b.tensors.push_back({"a", {16}, kTypeF32, raw_bytes(std::vector<float>(16, 1.f)), std::nullopt});
    b.tensors.push_back({"b", {16}, kTypeF32, raw_bytes(std::vector<float>(16, 2.f)), 32});  // starts inside a
    EXPECT(code_of(b.build()) == Code::DataLoss);
}

TEST(tensor_data_may_have_gaps_but_not_overlap) {
    Builder b;
    b.tensors.push_back({"a", {8}, kTypeF32, raw_bytes(std::vector<float>(8, 1.f)), 0});
    b.tensors.push_back({"b", {8}, kTypeF32, raw_bytes(std::vector<float>(8, 2.f)), 64});     // gap of 32 bytes
    Bytes bytes = b.build();
    bytes.resize(bytes.size() + 64, 0);
    EXPECT(code_of(bytes) == Code::Ok);
}

TEST(rejects_file_ending_before_the_data_section) {
    Builder b = base_builder();
    Bytes bytes = b.build();
    EXPECT(b.data_start > b.info_end);                        // there is padding to cut into
    EXPECT(code_of(Bytes(bytes.begin(), bytes.begin() + static_cast<long>(b.info_end))) == Code::DataLoss);
    EXPECT(code_of(Bytes(bytes.begin(), bytes.begin() + static_cast<long>(b.data_start) - 1)) == Code::DataLoss);
    // Exactly at the start of the data section there are still no tensor bytes.
    EXPECT(code_of(Bytes(bytes.begin(), bytes.begin() + static_cast<long>(b.data_start))) == Code::DataLoss);
    // The complete tensor bytes (without trailing padding) are enough.
    EXPECT(code_of(Bytes(bytes.begin(), bytes.begin() + static_cast<long>(b.data_end))) == Code::Ok);
}

// ---- hostile-input robustness ----

TEST(every_truncation_is_rejected_or_complete) {
    Builder b = base_builder();
    Bytes bytes = b.build();
    const uint64_t needed = b.data_end;  // everything up to the last tensor's final byte
    for (size_t len = 1; len <= bytes.size(); len++) {
        auto r = GgufFile::parse(bytes.data(), len);
        if (r.is_ok()) {
            if (len < needed) {
                std::fprintf(stderr, "  accepted a file truncated to %zu of %zu bytes (needs %llu)\n", len, bytes.size(),
                             static_cast<unsigned long long>(needed));
                dooms_test::current_failures()++;
                return;
            }
        } else {
            const Code c = r.status().code();
            EXPECT(c == Code::DataLoss || c == Code::InvalidArgument || c == Code::Unsupported || c == Code::ResourceExhausted);
        }
    }
}

TEST(random_corruption_never_escapes_the_file) {
    const Bytes original = base_builder().build();
    std::mt19937 rng(12345);
    constexpr int kIterations = 4000;
    for (int it = 0; it < kIterations; it++) {
        Bytes mutated = original;
        const int flips = 1 + static_cast<int>(rng() % 4);
        for (int f = 0; f < flips; f++) mutated[rng() % mutated.size()] = static_cast<uint8_t>(rng());
        auto r = GgufFile::parse(mutated.data(), mutated.size());
        if (!r.is_ok()) continue;
        // Anything accepted must be fully inside the buffer, with unique names.
        std::vector<std::string> names;
        for (const GgufTensorInfo & t : r.value().tensors()) {
            if (t.offset > mutated.size() || t.nbytes > mutated.size() - t.offset) {
                std::fprintf(stderr, "  iteration %d: accepted tensor '%s' outside the file\n", it, t.name.c_str());
                dooms_test::current_failures()++;
                return;
            }
            names.push_back(t.name);
        }
        std::sort(names.begin(), names.end());
        EXPECT(std::adjacent_find(names.begin(), names.end()) == names.end());
    }
}

TEST(random_garbage_is_rejected_cleanly) {
    std::mt19937 rng(777);
    for (int it = 0; it < 500; it++) {
        Bytes junk(1 + rng() % 600);
        for (auto & x : junk) x = static_cast<uint8_t>(rng());
        EXPECT(!GgufFile::parse(junk.data(), junk.size()).is_ok());      // practically never has valid magic
        if (junk.size() >= 8) {                                           // valid magic + version, random rest
            std::memcpy(junk.data(), "GGUF", 4);
            const uint32_t v = 3;
            std::memcpy(junk.data() + 4, &v, 4);
            (void) GgufFile::parse(junk.data(), junk.size());             // must not crash; outcome unconstrained
        }
    }
}

// ---- independent references: ggml's writer and reader ----

TEST(reads_a_file_written_by_ggml) {
    ggml_init_params ip = {};
    ip.mem_size = 1 << 20;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);  // ne0 = 3: shape [2, 3] row-major
    ggml_set_name(w, "layer.weight");
    for (int i = 0; i < 6; i++) static_cast<float *>(w->data)[i] = 10.f + static_cast<float>(i);
    ggml_tensor * v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 5);
    ggml_set_name(v, "vec");
    for (int i = 0; i < 5; i++) static_cast<ggml_fp16_t *>(v->data)[i] = ggml_fp32_to_fp16(static_cast<float>(i));

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "ggml-written");
    gguf_set_val_u32(g, "ggml.count", 12345);
    gguf_set_val_f32(g, "ggml.eps", 1e-6f);
    gguf_set_val_bool(g, "ggml.flag", true);
    gguf_add_tensor(g, w);
    gguf_add_tensor(g, v);
    const auto path = temp_path("dooms_test_ggml_written.gguf");
    EXPECT(gguf_write_to_file(g, path.string().c_str(), false));
    gguf_free(g);
    ggml_free(ctx);

    auto f = GgufFile::open(path.string());
    EXPECT(f.is_ok());
    if (f.is_ok()) {
        EXPECT_EQ(f.value().get_string("general.architecture").value(), std::string("ggml-written"));
        EXPECT_EQ(f.value().get_uint("ggml.count").value(), uint64_t(12345));
        EXPECT_EQ(f.value().get_float("ggml.eps").value(), static_cast<double>(1e-6f));
        EXPECT(f.value().get_bool("ggml.flag").value());
        const GgufTensorInfo * wi = f.value().find_tensor("layer.weight").value();
        EXPECT_EQ(wi->shape.to_string(), std::string("[2, 3]"));
        const float * wd = reinterpret_cast<const float *>(f.value().tensor_data(*wi));
        for (int i = 0; i < 6; i++) EXPECT_EQ(wd[i], 10.f + static_cast<float>(i));
        const GgufTensorInfo * vi = f.value().find_tensor("vec").value();
        EXPECT(vi->dtype == DType::F16);
        const uint16_t * vd = reinterpret_cast<const uint16_t *>(f.value().tensor_data(*vi));
        for (int i = 0; i < 5; i++) EXPECT_EQ(ggml_fp16_to_fp32(vd[i]), static_cast<float>(i));
    }
    std::filesystem::remove(path);
}

TEST(agrees_with_ggml_reader_on_valid_files) {
    for (uint32_t align : {0u, 64u}) {
        Builder b = base_builder();
        if (align) b.alignment = align;
        const Bytes bytes = b.build();
        auto ours = parse(bytes);
        EXPECT(ours.is_ok());
        ggml_context * data_ctx = nullptr;
        gguf_init_params gp = {true, &data_ctx};
        gguf_context * theirs = gguf_init_from_buffer(bytes.data(), bytes.size(), gp);
        EXPECT(theirs != nullptr);
        if (!ours.is_ok() || !theirs) { if (theirs) gguf_free(theirs); if (data_ctx) ggml_free(data_ctx); continue; }

        const GgufFile & f = ours.value();
        EXPECT_EQ(f.version(), gguf_get_version(theirs));
        EXPECT_EQ(f.alignment(), static_cast<uint64_t>(gguf_get_alignment(theirs)));
        EXPECT_EQ(static_cast<int64_t>(f.tensors().size()), gguf_get_n_tensors(theirs));
        EXPECT_EQ(static_cast<int64_t>(f.metadata().size()), gguf_get_n_kv(theirs));
        for (int64_t i = 0; i < gguf_get_n_tensors(theirs); i++) {
            const char * name = gguf_get_tensor_name(theirs, i);
            auto mine = f.find_tensor(name);
            EXPECT(mine.is_ok());
            if (!mine.is_ok()) continue;
            EXPECT_EQ(mine.value()->offset, static_cast<uint64_t>(gguf_get_data_offset(theirs) + gguf_get_tensor_offset(theirs, i)));
            EXPECT_EQ(mine.value()->nbytes, static_cast<uint64_t>(gguf_get_tensor_size(theirs, i)));
            const int64_t * ne = gguf_get_tensor_ne(theirs, i);
            const Shape & s = mine.value()->shape;
            for (int d = 0; d < s.rank(); d++) EXPECT_EQ(s.dim(d), ne[s.rank() - 1 - d]);  // ggml lists dims reversed
        }
        gguf_free(theirs);
        if (data_ctx) ggml_free(data_ctx);
    }
}

// ---- files on disk ----

TEST(open_maps_a_file_and_survives_moves) {
    const auto path = temp_path("dooms_test_open.gguf");
    write_file(path, base_builder().build());
    auto f = GgufFile::open(path.string());
    EXPECT(f.is_ok());
    if (!f.is_ok()) return;
    GgufFile moved = std::move(f).value();          // the mapping must follow the object
    GgufFile moved_again = std::move(moved);
    const float * a = reinterpret_cast<const float *>(moved_again.tensor_data(*moved_again.find_tensor("a").value()));
    EXPECT_EQ(a[5], 5.f);
    std::filesystem::remove(path);
}

TEST(open_reports_file_problems) {
    EXPECT(GgufFile::open("/nonexistent/dooms/none.gguf").status().code() == Code::NotFound);
    EXPECT(GgufFile::open(std::filesystem::temp_directory_path().string()).status().code() == Code::InvalidArgument);  // a directory
    const auto empty = temp_path("dooms_test_empty.gguf");
    write_file(empty, Bytes());
    EXPECT(GgufFile::open(empty.string()).status().code() == Code::DataLoss);
    std::filesystem::remove(empty);
    const auto junk = temp_path("dooms_test_junk.gguf");
    write_file(junk, Bytes{'n', 'o', 't', ' ', 'g', 'g', 'u', 'f', '!', '!'});
    auto r = GgufFile::open(junk.string());
    EXPECT(r.status().code() == Code::DataLoss);
    EXPECT(r.status().message().find(junk.string()) != std::string::npos);   // errors name the file
    std::filesystem::remove(junk);
}


int main() { return dooms_test::run_all(); }
