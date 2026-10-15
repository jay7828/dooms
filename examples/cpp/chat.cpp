// Chat with any GGUF model through the dooms C++ API:   chat MODEL.gguf "your question"
#include <dooms/dooms.hpp>

#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " MODEL.gguf \"question\"\n";
        return 2;
    }
    try {
        dooms::Llm llm(argv[1]);
        llm.chat({{"user", argv[2]}}, 128, {}, {}, [](const std::string & piece) {
            std::cout << piece << std::flush;
            return true;
        });
        std::cout << "\n";
    } catch (const dooms::Error & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
