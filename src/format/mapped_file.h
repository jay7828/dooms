#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/status.h"

namespace dooms {

// Read-only memory mapping of a whole file (POSIX mmap). Move-only; unmaps on destruction.
// Pages are loaded lazily by the OS, so opening a multi-GB model is cheap.
class MappedFile {
public:
    // NotFound if the file cannot be opened, DataLoss if it is empty or the map fails.
    static Result<MappedFile> open(const std::string & path);

    MappedFile(MappedFile && other) noexcept;
    MappedFile & operator=(MappedFile && other) noexcept;
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    ~MappedFile();

    const uint8_t * data() const { return data_; }
    size_t size() const { return size_; }

private:
    MappedFile(const uint8_t * data, size_t size) : data_(data), size_(size) {}
    void reset();
    const uint8_t * data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace dooms
