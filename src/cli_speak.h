#pragma once
// `dooms-cli speak`: phonemes -> a WAV file with Kokoro-82M. Returns the process exit code: 0 ok, 1 failed, 2 usage error.
namespace dooms {
int run_speak_command(int argc, char ** argv);  // argv[0] is the sub-command name "speak"
}
