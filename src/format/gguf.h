#pragma once
// GGUF reader (format version 2 and 3, little-endian hosts). Treats the input as
// hostile: every read is bounds-checked, every count and size is capped, and any
// inconsistency is reported as a Status, never a crash. Layout reference: the header
// comment of ggml's include/gguf.h.
//
// Error codes: DataLoss (truncated/corrupt/inconsistent file), Unsupported (version,
// value or tensor type this reader does not handle), ResourceExhausted (a limit in
// GgufLimits would be exceeded), NotFound / InvalidArgument (typed getters).

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"
#include "core/tensor.h"
#include "format/mapped_file.h"

namespace dooms {

// Caps that bound the work and memory a file can demand. Defaults are far above
// real models and far below anything that could exhaust memory.
struct GgufLimits {
    uint64_t max_kv = uint64_t(1) << 20;
    uint64_t max_tensors = uint64_t(1) << 20;
    uint64_t max_name_bytes = 256;                     // keys and tensor names
    uint64_t max_string_bytes = uint64_t(64) << 20;    // string values
    uint64_t max_array_elements = uint64_t(1) << 22;
    uint64_t max_alignment = uint64_t(1) << 24;
};

enum class GgufValueType { U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
                           String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12 };

const char * gguf_value_type_name(GgufValueType t);

struct GgufValue {
    GgufValueType type = GgufValueType::U8;
    GgufValueType element_type = GgufValueType::U8;  // meaningful when type == Array
    uint64_t u = 0;                                  // U8/U16/U32/U64/Bool
    int64_t i = 0;                                   // I8/I16/I32/I64
    double f = 0;                                    // F32/F64
    std::string s;                                   // String
    std::vector<GgufValue> items;                    // Array elements
};

struct GgufTensorInfo {
    std::string name;
    DType dtype = DType::F32;
    Shape shape;           // row-major, outermost first (the file stores the reverse)
    uint64_t offset = 0;   // from the start of the file
    uint64_t nbytes = 0;
};

// An immutable parsed GGUF file. Tensor data is not copied: it is read from the
// underlying bytes (a memory map when created with open()).
class GgufFile {
public:
    // `data` must stay valid for the lifetime of the returned object.
    static Result<GgufFile> parse(const uint8_t * data, size_t size, const GgufLimits & limits = GgufLimits());
    // Memory-maps `path` and parses it; the mapping is owned by the returned object.
    static Result<GgufFile> open(const std::string & path, const GgufLimits & limits = GgufLimits());

    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }

    const std::map<std::string, GgufValue> & metadata() const { return kv_; }
    Result<std::string> get_string(std::string_view key) const;
    Result<uint64_t> get_uint(std::string_view key) const;   // any integer type holding a value >= 0
    Result<int64_t> get_int(std::string_view key) const;     // any integer type that fits int64
    Result<double> get_float(std::string_view key) const;    // F32 or F64
    Result<bool> get_bool(std::string_view key) const;

    const std::vector<GgufTensorInfo> & tensors() const { return tensors_; }
    Result<const GgufTensorInfo *> find_tensor(std::string_view name) const;
    // Pointer to the tensor's bytes inside the file (valid while this object lives).
    const uint8_t * tensor_data(const GgufTensorInfo & info) const;
    // Read-only view of a tensor's data. The view is typed non-const only because TensorView
    // is; callers must not write through it (set_weight() copies out of it).
    Result<TensorView> tensor_view(std::string_view name) const;

private:
    GgufFile() = default;
    const GgufValue * find_kv(std::string_view key) const;

    const uint8_t * data_ = nullptr;
    size_t size_ = 0;
    std::shared_ptr<const MappedFile> mapping_;  // set by open()
    uint32_t version_ = 0;
    uint64_t alignment_ = 0;
    std::map<std::string, GgufValue> kv_;
    std::vector<GgufTensorInfo> tensors_;
    std::map<std::string, size_t, std::less<>> tensor_index_;
};

}  // namespace dooms
