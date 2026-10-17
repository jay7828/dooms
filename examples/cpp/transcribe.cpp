// Speech to text through the dooms C++ API:   transcribe MODEL_DIR audio.wav [more files...]
// MODEL_DIR holds encoder-fp16.mlpackage and decoder.gguf from tools/convert/convert_whisper.py.
#include <dooms/dooms.hpp>

#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " MODEL_DIR AUDIO [AUDIO...]\n";
        return 2;
    }
    try {
        dooms::Whisper stt(argv[1]);
        for (int i = 2; i < argc; i++) {
            std::cout << stt.transcribe_file(argv[i]) << "\n";
            const dooms_whisper_timings t = stt.last_timings();
            std::cerr << argv[i] << ": " << t.audio_s << " s of audio in " << t.total_ms << " ms\n";
        }
    } catch (const dooms::Error & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
