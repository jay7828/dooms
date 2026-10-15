#pragma once

#include <functional>
#include <string_view>

#include "core/status.h"

namespace dooms {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

// Default level and the environment variable that overrides it (single source of truth).
constexpr LogLevel kDefaultLogLevel = LogLevel::Warn;
constexpr const char * kLogLevelEnvVar = "DOOMS_LOG_LEVEL";

const char * log_level_name(LogLevel level);                // "debug", "info", "warn", "error", "off"
Result<LogLevel> parse_log_level(std::string_view name);    // case-sensitive inverse of the above

LogLevel log_level();
void set_log_level(LogLevel level);
// Apply DOOMS_LOG_LEVEL if set. Unset leaves the level alone; an invalid value
// returns InvalidArgument and leaves the level alone.
Status init_log_from_env();

// Receives each emitted message (already filtered by level). Pass an empty
// function to restore the default stderr sink. Thread-safe.
using LogSink = std::function<void(LogLevel, const char * message)>;
void set_log_sink(LogSink sink);

void log_message(LogLevel level, const char * fmt, ...) __attribute__((format(printf, 2, 3)));

}  // namespace dooms

#define DOOMS_LOG_DEBUG(...) ::dooms::log_message(::dooms::LogLevel::Debug, __VA_ARGS__)
#define DOOMS_LOG_INFO(...)  ::dooms::log_message(::dooms::LogLevel::Info, __VA_ARGS__)
#define DOOMS_LOG_WARN(...)  ::dooms::log_message(::dooms::LogLevel::Warn, __VA_ARGS__)
#define DOOMS_LOG_ERROR(...) ::dooms::log_message(::dooms::LogLevel::Error, __VA_ARGS__)
