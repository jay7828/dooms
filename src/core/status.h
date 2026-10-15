#pragma once
// Status / Result: how every fallible operation reports errors. No exceptions
// cross module boundaries.

#include <optional>
#include <string>
#include <utility>

#include "core/check.h"

namespace dooms {

enum class Code {
    Ok = 0,
    InvalidArgument,    // caller passed something malformed
    OutOfRange,         // index or value outside the permitted range
    NotFound,           // named thing (tensor, file, op) does not exist
    Unsupported,        // valid request this build/backend cannot do
    DataLoss,           // corrupt or truncated input data
    ResourceExhausted,  // allocation or size limit exceeded
    IoError,            // the operating system refused a file operation (open, write, rename)
    Internal,           // bug or broken invariant detected at runtime
};

const char * code_name(Code code);

class Status {
public:
    Status() = default;  // Ok
    // A non-Ok status carries a message; the message of an Ok status is dropped.
    Status(Code code, std::string message);

    bool is_ok() const { return code_ == Code::Ok; }
    Code code() const { return code_; }
    const std::string & message() const { return message_; }
    std::string to_string() const;  // "InvalidArgument: ..." or "Ok"

private:
    Code code_ = Code::Ok;
    std::string message_;
};

inline Status invalid_argument(std::string m) { return Status(Code::InvalidArgument, std::move(m)); }
inline Status out_of_range(std::string m)     { return Status(Code::OutOfRange, std::move(m)); }
inline Status not_found(std::string m)        { return Status(Code::NotFound, std::move(m)); }
inline Status unsupported(std::string m)      { return Status(Code::Unsupported, std::move(m)); }
inline Status data_loss(std::string m)        { return Status(Code::DataLoss, std::move(m)); }
inline Status resource_exhausted(std::string m) { return Status(Code::ResourceExhausted, std::move(m)); }
inline Status io_error(std::string m)         { return Status(Code::IoError, std::move(m)); }
inline Status internal_error(std::string m)   { return Status(Code::Internal, std::move(m)); }

// Either a value or a non-Ok Status. Calling value() on an error is a contract
// violation (aborts); check is_ok() first.
template <class T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    Result(Status status) : status_(std::move(status)) {
        DOOMS_CHECK(!status_.is_ok(), "Result constructed from an Ok status without a value");
    }

    bool is_ok() const { return value_.has_value(); }
    const Status & status() const { return status_; }

    T & value() & { DOOMS_CHECK(is_ok(), "value() on an error Result"); return *value_; }
    const T & value() const & { DOOMS_CHECK(is_ok(), "value() on an error Result"); return *value_; }
    T && value() && { DOOMS_CHECK(is_ok(), "value() on an error Result"); return std::move(*value_); }

private:
    std::optional<T> value_;
    Status status_;
};

}  // namespace dooms

// Propagate a non-Ok Status from a function that returns Status or Result<U>.
#define DOOMS_RETURN_IF_ERROR(expr)              \
    do {                                         \
        ::dooms::Status _dooms_st = (expr);      \
        if (!_dooms_st.is_ok()) return _dooms_st; \
    } while (0)
