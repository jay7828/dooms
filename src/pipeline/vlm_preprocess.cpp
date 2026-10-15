#include "pipeline/vlm_preprocess.h"

#include <algorithm>
#include <cmath>

#include "pipeline/image_resample.h"

namespace dooms {

void idefics3_first_size(int64_t height, int64_t width, int64_t longest_edge, int64_t max_side, int64_t * out_h, int64_t * out_w) {
    // _resize_output_size_rescale_to_max_len: the longer side becomes longest_edge, the other keeps the aspect ratio (truncated, then made even)
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    int64_t h = height, w = width;
    if (width >= height) {
        w = longest_edge;
        h = static_cast<int64_t>(static_cast<double>(w) / aspect);
        if (h % 2 != 0) h += 1;
    } else {
        h = longest_edge;
        w = static_cast<int64_t>(static_cast<double>(h) * aspect);
        if (w % 2 != 0) w += 1;
    }
    h = std::max<int64_t>(h, 1);
    w = std::max<int64_t>(w, 1);
    // _resize_output_size_scale_below_upper_bound
    const double aspect2 = static_cast<double>(w) / static_cast<double>(h);
    if (w >= h && w > max_side) {
        w = max_side;
        h = static_cast<int64_t>(static_cast<double>(w) / aspect2);
    } else if (h > w && h > max_side) {
        h = max_side;
        w = static_cast<int64_t>(static_cast<double>(h) * aspect2);
    }
    *out_h = std::max<int64_t>(h, 1);
    *out_w = std::max<int64_t>(w, 1);
}

Result<std::vector<float>> idefics3_pixel_values(const RgbImage & image, const VlmImageOptions & o) {
    if (o.image_size < 1 || o.image_size > 8192 || o.first_resize_side < 1 || o.first_resize_side > 65536 || o.max_side < 1) return invalid_argument("bad image options");
    for (int c = 0; c < 3; c++) if (!std::isfinite(o.mean[c]) || !(o.std[c] > 0.0) || !std::isfinite(o.std[c])) return invalid_argument("mean must be finite and std positive");
    if (image.width < 1 || image.height < 1) return invalid_argument("the image is empty");
    int64_t h1, w1;
    idefics3_first_size(image.height, image.width, o.first_resize_side, o.max_side, &h1, &w1);
    auto first = resize_lanczos(image, w1, h1);
    if (!first.is_ok()) return first.status();
    auto square = resize_lanczos(first.value(), o.image_size, o.image_size);
    if (!square.is_ok()) return square.status();
    const int64_t plane = o.image_size * o.image_size;
    std::vector<float> chw(static_cast<size_t>(3 * plane));
    const double rescale = 0.00392156862745098;  // transformers' 1/255 as a float64 literal
    for (int c = 0; c < 3; c++) {
        const float mean = static_cast<float>(o.mean[c]), sd = static_cast<float>(o.std[c]);
        for (int64_t i = 0; i < plane; i++) {
            const float v = static_cast<float>(static_cast<double>(square.value().pixels[static_cast<size_t>(i * 3 + c)]) * rescale);
            chw[static_cast<size_t>(c * plane + i)] = (v - mean) / sd;
        }
    }
    return chw;
}

}  // namespace dooms
