#pragma once
// Tiny dependency-free test framework shared by all unit tests.
//   TEST(name) { EXPECT(cond); EXPECT_EQ(a, b); }
//   int main() { return dooms_test::run_all(); }   // exit code = number of failed tests
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace dooms_test {

struct Case {
    const char * name;
    void (*fn)();
};

inline std::vector<Case> & cases() {
    static std::vector<Case> c;
    return c;
}
inline int & current_failures() {
    static int n = 0;
    return n;
}

struct Registrar {
    Registrar(const char * name, void (*fn)()) { cases().push_back({name, fn}); }
};

inline int run_all() {
    int failed_tests = 0;
    for (const Case & c : cases()) {
        current_failures() = 0;
        c.fn();
        if (current_failures() > 0) {
            std::fprintf(stderr, "FAILED  %s (%d check(s))\n", c.name, current_failures());
            failed_tests++;
        } else {
            std::printf("ok      %s\n", c.name);
        }
    }
    std::printf("%zu tests, %d failed\n", cases().size(), failed_tests);
    return failed_tests;
}

template <class A, class B>
inline void expect_eq(const A & a, const B & b, const char * sa, const char * sb, const char * file, int line) {
    if (!(a == b)) {
        std::ostringstream os;
        os << a << " vs " << b;
        std::fprintf(stderr, "  %s:%d: EXPECT_EQ(%s, %s) failed: %s\n", file, line, sa, sb, os.str().c_str());
        current_failures()++;
    }
}

}  // namespace dooms_test

#define TEST(name)                                                   \
    static void name();                                              \
    static ::dooms_test::Registrar registrar_##name(#name, name);    \
    static void name()

#define EXPECT(cond)                                                                      \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "  %s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
            ::dooms_test::current_failures()++;                                           \
        }                                                                                 \
    } while (0)

#define EXPECT_EQ(a, b) ::dooms_test::expect_eq((a), (b), #a, #b, __FILE__, __LINE__)
