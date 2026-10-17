#pragma once
// `dooms-cli transcribe`: speech file -> text with Whisper. Returns the process exit code: 0 ok, 1 failed, 2 usage error.
namespace dooms {
int run_transcribe_command(int argc, char ** argv);  // argv[0] is the sub-command name "transcribe"
}
