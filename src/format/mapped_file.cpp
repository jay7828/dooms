#include "format/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace dooms {

Result<MappedFile> MappedFile::open(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return not_found("cannot open '" + path + "': " + std::strerror(errno));

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::close(fd);
        return data_loss("cannot stat '" + path + "': " + std::strerror(err));
    }
    if (!S_ISREG(st.st_mode)) {
        ::close(fd);
        return invalid_argument("'" + path + "' is not a regular file");
    }
    if (st.st_size <= 0) {
        ::close(fd);
        return data_loss("'" + path + "' is empty");
    }
    const size_t size = static_cast<size_t>(st.st_size);
    void * p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int map_err = errno;
    ::close(fd);  // the mapping stays valid after closing the descriptor
    if (p == MAP_FAILED) return resource_exhausted("cannot map '" + path + "': " + std::strerror(map_err));
    return MappedFile(static_cast<const uint8_t *>(p), size);
}

MappedFile::MappedFile(MappedFile && other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

MappedFile & MappedFile::operator=(MappedFile && other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

MappedFile::~MappedFile() { reset(); }

void MappedFile::reset() {
    if (data_) ::munmap(const_cast<uint8_t *>(data_), size_);
    data_ = nullptr;
    size_ = 0;
}

}  // namespace dooms
