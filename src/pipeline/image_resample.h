#pragma once
// Pillow-exact Lanczos resampling of 8-bit RGB images (Image.resize(size, resample=LANCZOS)): the same support, coefficient normalization, 22-bit fixed-point
// arithmetic and 8-bit rounding between the horizontal and the vertical pass as Pillow 11's reference C code, so results are bit-identical to the Python processors
// that call it (transformers' image processors do). Generic image code.

#include <cstdint>

#include "core/status.h"
#include "pipeline/rgb_image.h"

namespace dooms {

// InvalidArgument: an empty or inconsistent image, or a target side outside [1, 65536].
Result<RgbImage> resize_lanczos(const RgbImage & image, int64_t width, int64_t height);

}  // namespace dooms
