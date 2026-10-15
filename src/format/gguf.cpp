#include "format/gguf.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

namespace dooms {

namespace {

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the GGUF reader assumes a little-endian host");

constexpr char kMagic[4] = {'G', 'G', 'U', 'F'};
constexpr uint32_t kMinVersion = 2;  // v2 and v3 share this layout (64-bit counts); v1 used 32-bit counts
constexpr uint32_t kMaxVersion = 3;
constexpr uint64_t kDefaultAlignment = 32;
constexpr const char * kAlignmentKey = "general.alignment";
constexpr int32_t kMaxValueTypeId = static_cast<int32_t>(GgufValueType::F64);

// ggml_type ids this reader can map to a DType (see ggml.h enum ggml_type).
constexpr int32_t kGgmlTypeF32 = 0, kGgmlTypeF16 = 1, kGgmlTypeI32 = 26, kGgmlTypeBF16 = 30;

// Bounds-checked little-endian reader over a byte range.
class Cursor {
public:
    Cursor(const uint8_t * data, uint64_t size) : data_(data), size_(size) {}
    uint64_t pos() const { return pos_; }
    uint64_t remaining() const { return size_ - pos_; }

    template <class T>
    Status read(T * out, const char * what) {
        if (remaining() < sizeof(T)) return truncated(what, sizeof(T));
        std::memcpy(out, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return Status();
    }

    Status read_string(std::string * out, const char * what, uint64_t max_len) {
        uint64_t len = 0;
        DOOMS_RETURN_IF_ERROR(read(&len, what));
        if (len > max_len) {
            return resource_exhausted(std::string(what) + ": string of " + std::to_string(len) + " bytes at offset " +
                                      std::to_string(pos_) + " exceeds the limit of " + std::to_string(max_len));
        }
        if (remaining() < len) return truncated(what, len);
        out->assign(reinterpret_cast<const char *>(data_ + pos_), static_cast<size_t>(len));
        pos_ += len;
        return Status();
    }

private:
    Status truncated(const char * what, uint64_t need) const {
        return data_loss(std::string("file truncated at offset ") + std::to_string(pos_) + " while reading " + what +
                         " (needs " + std::to_string(need) + " bytes, " + std::to_string(remaining()) + " left)");
    }
    const uint8_t * data_;
    uint64_t size_;
    uint64_t pos_ = 0;
};

Status read_scalar(Cursor & c, GgufValueType t, GgufValue * v, const GgufLimits & lim, const char * what) {
    v->type = t;
    switch (t) {
        case GgufValueType::U8:  { uint8_t x;  DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->u = x; return Status(); }
        case GgufValueType::I8:  { int8_t x;   DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->i = x; return Status(); }
        case GgufValueType::U16: { uint16_t x; DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->u = x; return Status(); }
        case GgufValueType::I16: { int16_t x;  DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->i = x; return Status(); }
        case GgufValueType::U32: { uint32_t x; DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->u = x; return Status(); }
        case GgufValueType::I32: { int32_t x;  DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->i = x; return Status(); }
        case GgufValueType::U64: { uint64_t x; DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->u = x; return Status(); }
        case GgufValueType::I64: { int64_t x;  DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->i = x; return Status(); }
        case GgufValueType::F32: { float x;    DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->f = x; return Status(); }
        case GgufValueType::F64: { double x;   DOOMS_RETURN_IF_ERROR(c.read(&x, what)); v->f = x; return Status(); }
        case GgufValueType::Bool: {
            uint8_t x;
            DOOMS_RETURN_IF_ERROR(c.read(&x, what));
            if (x > 1) return data_loss(std::string(what) + ": bool value " + std::to_string(x) + " is not 0 or 1");
            v->u = x;
            return Status();
        }
        case GgufValueType::String:
            return c.read_string(&v->s, what, lim.max_string_bytes);
        case GgufValueType::Array:
            break;
    }
    return internal_error("read_scalar called with an array type");
}

Result<GgufValueType> to_value_type(int32_t id, const char * what) {
    if (id < 0 || id > kMaxValueTypeId) {
        return unsupported(std::string(what) + ": unknown value type " + std::to_string(id));
    }
    return static_cast<GgufValueType>(id);
}

Status read_value(Cursor & c, GgufValue * v, const GgufLimits & lim, const std::string & key) {
    const std::string what = "value of '" + key + "'";
    int32_t type_id = 0;
    DOOMS_RETURN_IF_ERROR(c.read(&type_id, what.c_str()));
    auto type = to_value_type(type_id, what.c_str());
    if (!type.is_ok()) return type.status();
    if (type.value() != GgufValueType::Array) return read_scalar(c, type.value(), v, lim, what.c_str());

    v->type = GgufValueType::Array;
    int32_t elem_id = 0;
    uint64_t count = 0;
    DOOMS_RETURN_IF_ERROR(c.read(&elem_id, what.c_str()));
    DOOMS_RETURN_IF_ERROR(c.read(&count, what.c_str()));
    auto elem = to_value_type(elem_id, what.c_str());
    if (!elem.is_ok()) return elem.status();
    if (elem.value() == GgufValueType::Array) return unsupported(what + ": nested arrays are not supported");
    v->element_type = elem.value();
    if (count > lim.max_array_elements) {
        return resource_exhausted(what + ": array of " + std::to_string(count) + " elements exceeds the limit of " +
                                  std::to_string(lim.max_array_elements));
    }
    if (count > c.remaining()) {  // every element takes at least one byte
        return data_loss(what + ": array claims " + std::to_string(count) + " elements but only " +
                         std::to_string(c.remaining()) + " bytes remain");
    }
    v->items.resize(static_cast<size_t>(count));
    for (GgufValue & item : v->items) DOOMS_RETURN_IF_ERROR(read_scalar(c, elem.value(), &item, lim, what.c_str()));
    return Status();
}

Result<DType> to_dtype(int32_t ggml_type, const std::string & tensor) {
    switch (ggml_type) {
        case kGgmlTypeF32:  return DType::F32;
        case kGgmlTypeF16:  return DType::F16;
        case kGgmlTypeBF16: return DType::BF16;
        case kGgmlTypeI32:  return DType::I32;
    }
    return unsupported("tensor '" + tensor + "' has ggml type " + std::to_string(ggml_type) +
                       ", which this reader does not support (supported: f32, f16, bf16, i32)");
}

bool is_integer(GgufValueType t) {
    switch (t) {
        case GgufValueType::U8: case GgufValueType::I8: case GgufValueType::U16: case GgufValueType::I16:
        case GgufValueType::U32: case GgufValueType::I32: case GgufValueType::U64: case GgufValueType::I64:
            return true;
        default:
            return false;
    }
}

bool is_signed(GgufValueType t) {
    return t == GgufValueType::I8 || t == GgufValueType::I16 || t == GgufValueType::I32 || t == GgufValueType::I64;
}

}  // namespace

const char * gguf_value_type_name(GgufValueType t) {
    switch (t) {
        case GgufValueType::U8:     return "u8";
        case GgufValueType::I8:     return "i8";
        case GgufValueType::U16:    return "u16";
        case GgufValueType::I16:    return "i16";
        case GgufValueType::U32:    return "u32";
        case GgufValueType::I32:    return "i32";
        case GgufValueType::F32:    return "f32";
        case GgufValueType::Bool:   return "bool";
        case GgufValueType::String: return "string";
        case GgufValueType::Array:  return "array";
        case GgufValueType::U64:    return "u64";
        case GgufValueType::I64:    return "i64";
        case GgufValueType::F64:    return "f64";
    }
    return "unknown";
}

Result<GgufFile> GgufFile::parse(const uint8_t * data, size_t size, const GgufLimits & lim) {
    if (data == nullptr || size == 0) return invalid_argument("no data to parse");
    Cursor c(data, size);

    char magic[4];
    DOOMS_RETURN_IF_ERROR(c.read(&magic, "magic"));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) return data_loss("not a GGUF file (bad magic)");

    GgufFile f;
    f.data_ = data;
    f.size_ = size;
    DOOMS_RETURN_IF_ERROR(c.read(&f.version_, "version"));
    if (f.version_ < kMinVersion || f.version_ > kMaxVersion) {
        std::string msg = "GGUF version " + std::to_string(f.version_) + " is not supported (supported: " +
                          std::to_string(kMinVersion) + " to " + std::to_string(kMaxVersion) + ")";
        if ((f.version_ & 0xFFFF) == 0) msg += "; the file may have been written on a machine with different endianness";
        return unsupported(msg);
    }

    uint64_t n_tensors = 0, n_kv = 0;
    DOOMS_RETURN_IF_ERROR(c.read(&n_tensors, "tensor count"));
    DOOMS_RETURN_IF_ERROR(c.read(&n_kv, "metadata count"));
    if (n_tensors > lim.max_tensors) {
        return resource_exhausted("tensor count " + std::to_string(n_tensors) + " exceeds the limit of " + std::to_string(lim.max_tensors));
    }
    if (n_kv > lim.max_kv) {
        return resource_exhausted("metadata count " + std::to_string(n_kv) + " exceeds the limit of " + std::to_string(lim.max_kv));
    }

    for (uint64_t k = 0; k < n_kv; k++) {
        std::string key;
        DOOMS_RETURN_IF_ERROR(c.read_string(&key, "metadata key", lim.max_name_bytes));
        if (key.empty()) return data_loss("metadata entry " + std::to_string(k) + " has an empty key");
        GgufValue v;
        DOOMS_RETURN_IF_ERROR(read_value(c, &v, lim, key));
        if (!f.kv_.emplace(key, std::move(v)).second) return data_loss("duplicate metadata key '" + key + "'");
    }

    f.alignment_ = kDefaultAlignment;
    auto al = f.kv_.find(kAlignmentKey);
    if (al != f.kv_.end()) {
        if (al->second.type != GgufValueType::U32) {
            return data_loss(std::string(kAlignmentKey) + " must be a u32, got " + gguf_value_type_name(al->second.type));
        }
        f.alignment_ = al->second.u;
        if (f.alignment_ == 0 || (f.alignment_ & (f.alignment_ - 1)) != 0 || f.alignment_ > lim.max_alignment) {
            return data_loss(std::string(kAlignmentKey) + " = " + std::to_string(f.alignment_) +
                             " must be a power of two no larger than " + std::to_string(lim.max_alignment));
        }
    }

    f.tensors_.reserve(static_cast<size_t>(std::min<uint64_t>(n_tensors, c.remaining())));
    for (uint64_t t = 0; t < n_tensors; t++) {
        GgufTensorInfo info;
        DOOMS_RETURN_IF_ERROR(c.read_string(&info.name, "tensor name", lim.max_name_bytes));
        if (info.name.empty()) return data_loss("tensor " + std::to_string(t) + " has an empty name");
        if (f.tensor_index_.count(info.name)) return data_loss("duplicate tensor name '" + info.name + "'");

        uint32_t n_dims = 0;
        DOOMS_RETURN_IF_ERROR(c.read(&n_dims, "tensor rank"));
        if (n_dims > static_cast<uint32_t>(kMaxDims)) {
            return data_loss("tensor '" + info.name + "' has " + std::to_string(n_dims) + " dimensions; the maximum is " + std::to_string(kMaxDims));
        }
        int64_t dims_row_major[kMaxDims];
        for (uint32_t d = 0; d < n_dims; d++) {  // the file lists the fastest-varying dimension first
            int64_t dim = 0;
            DOOMS_RETURN_IF_ERROR(c.read(&dim, "tensor dimension"));
            if (dim < 1) return data_loss("tensor '" + info.name + "' dimension " + std::to_string(d) + " is " + std::to_string(dim) + "; dimensions must be >= 1");
            dims_row_major[n_dims - 1 - d] = dim;
        }
        auto shape = Shape::create(dims_row_major, static_cast<int>(n_dims));
        if (!shape.is_ok()) return data_loss("tensor '" + info.name + "': " + shape.status().message());
        info.shape = shape.value();

        int32_t ggml_type = 0;
        DOOMS_RETURN_IF_ERROR(c.read(&ggml_type, "tensor type"));
        auto dtype = to_dtype(ggml_type, info.name);
        if (!dtype.is_ok()) return dtype.status();
        info.dtype = dtype.value();

        const uint64_t elem = dtype_size(info.dtype);
        if (static_cast<uint64_t>(info.shape.numel()) > std::numeric_limits<uint64_t>::max() / elem) {
            return data_loss("tensor '" + info.name + "' byte size overflows");
        }
        info.nbytes = static_cast<uint64_t>(info.shape.numel()) * elem;
        DOOMS_RETURN_IF_ERROR(c.read(&info.offset, "tensor data offset"));  // relative to the data section for now

        f.tensor_index_.emplace(info.name, f.tensors_.size());
        f.tensors_.push_back(std::move(info));
    }

    if (f.tensors_.empty()) return f;

    // Data section: starts at the next multiple of the alignment after the tensor infos.
    const uint64_t pos = c.pos();
    const uint64_t data_start = (pos + (f.alignment_ - 1)) & ~(f.alignment_ - 1);
    if (data_start > size) return data_loss("file ends at " + std::to_string(size) + " before the data section at " + std::to_string(data_start));
    const uint64_t data_size = size - data_start;

    for (GgufTensorInfo & t : f.tensors_) {
        if (t.offset % f.alignment_ != 0) {
            return data_loss("tensor '" + t.name + "' offset " + std::to_string(t.offset) + " is not a multiple of the alignment " + std::to_string(f.alignment_));
        }
        if (t.offset > data_size || t.nbytes > data_size - t.offset) {
            return data_loss("tensor '" + t.name + "' (" + std::to_string(t.nbytes) + " bytes at data offset " + std::to_string(t.offset) +
                             ") extends past the end of the file (data section is " + std::to_string(data_size) + " bytes)");
        }
    }
    // No two tensors may share bytes.
    std::vector<size_t> order(f.tensors_.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return f.tensors_[a].offset < f.tensors_[b].offset; });
    for (size_t i = 1; i < order.size(); i++) {
        const GgufTensorInfo & prev = f.tensors_[order[i - 1]];
        const GgufTensorInfo & cur = f.tensors_[order[i]];
        if (prev.offset + prev.nbytes > cur.offset) {
            return data_loss("tensors '" + prev.name + "' and '" + cur.name + "' overlap in the data section");
        }
    }
    for (GgufTensorInfo & t : f.tensors_) t.offset += data_start;  // now absolute
    return f;
}

Result<GgufFile> GgufFile::open(const std::string & path, const GgufLimits & lim) {
    auto mapped = MappedFile::open(path);
    if (!mapped.is_ok()) return mapped.status();
    auto holder = std::make_shared<const MappedFile>(std::move(mapped).value());
    auto parsed = parse(holder->data(), holder->size(), lim);
    if (!parsed.is_ok()) return Status(parsed.status().code(), path + ": " + parsed.status().message());
    GgufFile f = std::move(parsed).value();
    f.mapping_ = std::move(holder);
    return f;
}

const GgufValue * GgufFile::find_kv(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    return it == kv_.end() ? nullptr : &it->second;
}

namespace {
Status wrong_type(std::string_view key, GgufValueType got, const char * want) {
    return invalid_argument("metadata '" + std::string(key) + "' has type " + gguf_value_type_name(got) + ", expected " + want);
}
}  // namespace

Result<std::string> GgufFile::get_string(std::string_view key) const {
    const GgufValue * v = find_kv(key);
    if (!v) return not_found("no metadata key '" + std::string(key) + "'");
    if (v->type != GgufValueType::String) return wrong_type(key, v->type, "string");
    return v->s;
}

Result<uint64_t> GgufFile::get_uint(std::string_view key) const {
    const GgufValue * v = find_kv(key);
    if (!v) return not_found("no metadata key '" + std::string(key) + "'");
    if (!is_integer(v->type)) return wrong_type(key, v->type, "an integer");
    if (!is_signed(v->type)) return v->u;
    if (v->i < 0) return out_of_range("metadata '" + std::string(key) + "' is " + std::to_string(v->i) + ", expected >= 0");
    return static_cast<uint64_t>(v->i);
}

Result<int64_t> GgufFile::get_int(std::string_view key) const {
    const GgufValue * v = find_kv(key);
    if (!v) return not_found("no metadata key '" + std::string(key) + "'");
    if (!is_integer(v->type)) return wrong_type(key, v->type, "an integer");
    if (is_signed(v->type)) return v->i;
    if (v->u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return out_of_range("metadata '" + std::string(key) + "' is " + std::to_string(v->u) + ", which does not fit in int64");
    }
    return static_cast<int64_t>(v->u);
}

Result<double> GgufFile::get_float(std::string_view key) const {
    const GgufValue * v = find_kv(key);
    if (!v) return not_found("no metadata key '" + std::string(key) + "'");
    if (v->type != GgufValueType::F32 && v->type != GgufValueType::F64) return wrong_type(key, v->type, "a float");
    return v->f;
}

Result<bool> GgufFile::get_bool(std::string_view key) const {
    const GgufValue * v = find_kv(key);
    if (!v) return not_found("no metadata key '" + std::string(key) + "'");
    if (v->type != GgufValueType::Bool) return wrong_type(key, v->type, "bool");
    return v->u != 0;
}

Result<const GgufTensorInfo *> GgufFile::find_tensor(std::string_view name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return not_found("no tensor named '" + std::string(name) + "'");
    return &tensors_[it->second];
}

const uint8_t * GgufFile::tensor_data(const GgufTensorInfo & info) const {
    DOOMS_CHECK(info.offset <= size_ && info.nbytes <= size_ - info.offset, "tensor info does not belong to this file");
    return data_ + info.offset;
}

Result<TensorView> GgufFile::tensor_view(std::string_view name) const {
    auto info = find_tensor(name);
    if (!info.is_ok()) return info.status();
    return TensorView::create(const_cast<uint8_t *>(tensor_data(*info.value())), info.value()->shape, info.value()->dtype);
}

}  // namespace dooms
