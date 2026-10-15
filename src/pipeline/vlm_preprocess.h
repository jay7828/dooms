#pragma once
// Image preprocessing of Idefics3-family vision-language models (SmolVLM) for ONE tile with image splitting off, exactly as transformers'
// Idefics3ImageProcessor does it: the longest edge is first rescaled to `first_resize_side` (upscaling small photos), then the image is squashed to
// image_size x image_size, both with Pillow's Lanczos filter; then value / 255 (float64 product rounded to float32), (x - mean) / std in float32, CHW.

#include <cstdint>
#include <vector>

#include "core/status.h"
#include "pipeline/rgb_image.h"

namespace dooms {

struct VlmImageOptions {
    int64_t image_size = 0;          // the vision tower's square input
    int64_t first_resize_side = 2048;// processor `size.longest_edge`
    int64_t max_side = 4096;         // transformers' MAX_IMAGE_SIZE: the first resize never exceeds it
    double mean[3] = {0.5, 0.5, 0.5}, std[3] = {0.5, 0.5, 0.5};
};

// The size the first resize produces (transformers' _resize_output_size_rescale_to_max_len + _scale_below_upper_bound).
void idefics3_first_size(int64_t height, int64_t width, int64_t longest_edge, int64_t max_side, int64_t * out_h, int64_t * out_w);

// Returns [3, image_size, image_size] floats. InvalidArgument for bad images/options.
Result<std::vector<float>> idefics3_pixel_values(const RgbImage & image, const VlmImageOptions & options);

}  // namespace dooms
