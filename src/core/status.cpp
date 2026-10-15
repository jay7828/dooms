#include "core/status.h"

namespace dooms {

const char * code_name(Code code) {
    switch (code) {
        case Code::Ok:                return "Ok";
        case Code::InvalidArgument:   return "InvalidArgument";
        case Code::OutOfRange:        return "OutOfRange";
        case Code::NotFound:          return "NotFound";
        case Code::Unsupported:       return "Unsupported";
        case Code::DataLoss:          return "DataLoss";
        case Code::ResourceExhausted: return "ResourceExhausted";
        case Code::IoError:           return "IoError";
        case Code::Internal:          return "Internal";
    }
    return "Unknown";
}

Status::Status(Code code, std::string message) : code_(code) {
    if (code != Code::Ok) message_ = std::move(message);
}

std::string Status::to_string() const {
    if (is_ok()) return "Ok";
    return std::string(code_name(code_)) + ": " + message_;
}

}  // namespace dooms
