#pragma once
// Internal-invariant checks. These guard programmer errors (violated
// preconditions), never bad user input: user input is validated and reported
// through Status. A failed check logs to stderr and aborts.

namespace dooms {

[[noreturn]] void check_failed(const char * file, int line, const char * expr, const char * msg);

}  // namespace dooms

#define DOOMS_CHECK(cond, msg)                                              \
    do {                                                                    \
        if (!(cond)) ::dooms::check_failed(__FILE__, __LINE__, #cond, msg); \
    } while (0)
