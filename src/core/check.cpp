#include "core/check.h"

#include <cstdio>
#include <cstdlib>

namespace dooms {

void check_failed(const char * file, int line, const char * expr, const char * msg) {
    std::fprintf(stderr, "[dooms] check failed at %s:%d: (%s) %s\n", file, line, expr, msg);
    std::abort();
}

}  // namespace dooms
