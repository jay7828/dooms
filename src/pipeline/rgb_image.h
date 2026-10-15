#pragma once
// An 8-bit RGB image in memory: the input type of every image pipeline (vision, vision-language and vision-language-action).

#include <cstdint>
#include <vector>

namespace dooms {

struct RgbImage {
    int64_t width = 0, height = 0;
    std::vector<uint8_t> pixels;  // height * width * 3, interleaved RGB, row-major
};

}  // namespace dooms
