#include "core/log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace dooms {

namespace {

std::atomic<int> g_level{static_cast<int>(kDefaultLogLevel)};
std::mutex g_sink_mutex;
LogSink g_sink;  // guarded by g_sink_mutex; empty = stderr

constexpr struct { LogLevel level; const char * name; } kLevels[] = {
    {LogLevel::Debug, "debug"},
    {LogLevel::Info,  "info"},
    {LogLevel::Warn,  "warn"},
    {LogLevel::Error, "error"},
    {LogLevel::Off,   "off"},
};

}  // namespace

const char * log_level_name(LogLevel level) {
    for (const auto & l : kLevels) {
        if (l.level == level) return l.name;
    }
    return "unknown";
}

Result<LogLevel> parse_log_level(std::string_view name) {
    for (const auto & l : kLevels) {
        if (name == l.name) return l.level;
    }
    return invalid_argument("unknown log level '" + std::string(name) + "'");
}

LogLevel log_level() { return static_cast<LogLevel>(g_level.load(std::memory_order_relaxed)); }

void set_log_level(LogLevel level) { g_level.store(static_cast<int>(level), std::memory_order_relaxed); }

Status init_log_from_env() {
    const char * v = std::getenv(kLogLevelEnvVar);
    if (v == nullptr) return Status();
    auto parsed = parse_log_level(v);
    if (!parsed.is_ok()) {
        return invalid_argument(std::string(kLogLevelEnvVar) + ": " + parsed.status().message());
    }
    set_log_level(parsed.value());
    return Status();
}

void set_log_sink(LogSink sink) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    g_sink = std::move(sink);
}

void log_message(LogLevel level, const char * fmt, ...) {
    if (level == LogLevel::Off || static_cast<int>(level) < g_level.load(std::memory_order_relaxed)) return;

    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::vector<char> buf(n > 0 ? static_cast<size_t>(n) + 1 : 1, '\0');
    if (n > 0) std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);

    std::lock_guard<std::mutex> lock(g_sink_mutex);
    if (g_sink) {
        g_sink(level, buf.data());
    } else {
        std::fprintf(stderr, "[dooms %s] %s\n", log_level_name(level), buf.data());
    }
}

}  // namespace dooms
