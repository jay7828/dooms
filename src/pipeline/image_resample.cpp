#include "pipeline/image_resample.h"

#include "core/parallel.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

namespace dooms {

namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;  // Pillow's PRECISION_BITS for 8-bit images
constexpr double kLanczosSupport = 3.0;
constexpr int64_t kMaxSide = 65536;
constexpr int64_t kMaxPixels = int64_t(1) << 28;

double sinc_filter(double x) {
    if (x == 0.0) return 1.0;
    x = x * M_PI;
    return std::sin(x) / x;
}

double lanczos_filter(double x) {
    if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3.0);
    return 0.0;
}

struct Coefficients {
    std::vector<int> bounds;   // per output: xmin, count
    std::vector<int32_t> kk;   // per output: ksize fixed-point weights
    int ksize = 0;
};

// Pillow's precompute_coeffs + normalize_coeffs_8bpc for the full input range.
Coefficients precompute(int in_size, int out_size) {
    Coefficients c;
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = kLanczosSupport * filterscale;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<size_t>(out_size) * 2);
    c.kk.assign(static_cast<size_t>(out_size) * static_cast<size_t>(c.ksize), 0);
    std::vector<double> k(static_cast<size_t>(c.ksize));
    for (int xx = 0; xx < out_size; xx++) {
        const double center = 0.0 + (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        std::fill(k.begin(), k.end(), 0.0);
        for (int x = 0; x < xmax; x++) {
            const double w = lanczos_filter((x + xmin - center + 0.5) * ss);
            k[static_cast<size_t>(x)] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; x++) if (ww != 0.0) k[static_cast<size_t>(x)] /= ww;
        c.bounds[static_cast<size_t>(xx) * 2] = xmin;
        c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
        for (int x = 0; x < c.ksize; x++) {
            const double v = k[static_cast<size_t>(x)];
            c.kk[static_cast<size_t>(xx) * static_cast<size_t>(c.ksize) + static_cast<size_t>(x)] =
                v < 0 ? static_cast<int32_t>(-0.5 + v * (1 << kPrecisionBits)) : static_cast<int32_t>(0.5 + v * (1 << kPrecisionBits));
        }
    }
    return c;
}

inline uint8_t clip8(int32_t sum) {
    const int32_t v = sum >> kPrecisionBits;  // arithmetic shift, as in Pillow
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

}  // namespace

Result<RgbImage> resize_lanczos(const RgbImage & image, int64_t width, int64_t height) {
    if (image.width < 1 || image.height < 1 || image.width * image.height > kMaxPixels || static_cast<int64_t>(image.pixels.size()) != image.width * image.height * 3) {
        return invalid_argument("the image is empty or its buffer does not match width * height * 3");
    }
    if (width < 1 || height < 1 || width > kMaxSide || height > kMaxSide || width * height > kMaxPixels) return invalid_argument("the target size is outside [1, 65536] per side or too large");
    if (width == image.width && height == image.height) return image;  // Pillow returns a copy

    const int iw = static_cast<int>(image.width), ih = static_cast<int>(image.height), ow = static_cast<int>(width), oh = static_cast<int>(height);
    const bool horizontal = ow != iw, vertical = oh != ih;
    // Horizontal pass first (Pillow's order), into an 8-bit intermediate image; then the vertical pass.
    std::vector<uint8_t> temp;
    const uint8_t * src = image.pixels.data();
    int tw = iw;
    if (horizontal) {
        const Coefficients c = precompute(iw, ow);
        temp.resize(static_cast<size_t>(ow) * static_cast<size_t>(ih) * 3);
        parallel_for(ih, static_cast<int64_t>(ow) * 3 * c.ksize, [&](int y0, int y1) {
            for (int y = y0; y < y1; y++) {
                const uint8_t * row = src + static_cast<size_t>(y) * static_cast<size_t>(iw) * 3;
                uint8_t * out = temp.data() + static_cast<size_t>(y) * static_cast<size_t>(ow) * 3;
                for (int x = 0; x < ow; x++) {
                    const int xmin = c.bounds[static_cast<size_t>(x) * 2], n = c.bounds[static_cast<size_t>(x) * 2 + 1];
                    const int32_t * k = c.kk.data() + static_cast<size_t>(x) * static_cast<size_t>(c.ksize);
                    int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
                    for (int i = 0; i < n; i++) {
                        const uint8_t * p = row + static_cast<size_t>(xmin + i) * 3;
                        s0 += p[0] * k[i];
                        s1 += p[1] * k[i];
                        s2 += p[2] * k[i];
                    }
                    out[static_cast<size_t>(x) * 3] = clip8(s0);
                    out[static_cast<size_t>(x) * 3 + 1] = clip8(s1);
                    out[static_cast<size_t>(x) * 3 + 2] = clip8(s2);
                }
            }
        });
        src = temp.data();
        tw = ow;
    }
    RgbImage out;
    out.width = ow;
    out.height = oh;
    if (vertical) {
        const Coefficients c = precompute(ih, oh);
        out.pixels.resize(static_cast<size_t>(tw) * static_cast<size_t>(oh) * 3);
        const size_t row_values = static_cast<size_t>(tw) * 3;
        parallel_for(oh, static_cast<int64_t>(row_values) * c.ksize, [&](int y0, int y1) {
            std::vector<int32_t> acc(row_values);
            for (int y = y0; y < y1; y++) {
                const int ymin = c.bounds[static_cast<size_t>(y) * 2], n = c.bounds[static_cast<size_t>(y) * 2 + 1];
                const int32_t * k = c.kk.data() + static_cast<size_t>(y) * static_cast<size_t>(c.ksize);
                std::fill(acc.begin(), acc.end(), 1 << (kPrecisionBits - 1));
                for (int i = 0; i < n; i++) {  // whole source rows at a time: contiguous reads, vectorizable
                    const uint8_t * srow = src + static_cast<size_t>(ymin + i) * row_values;
                    const int32_t w = k[i];
                    for (size_t x = 0; x < row_values; x++) acc[x] += srow[x] * w;
                }
                uint8_t * dst = out.pixels.data() + static_cast<size_t>(y) * row_values;
                for (size_t x = 0; x < row_values; x++) dst[x] = clip8(acc[x]);
            }
        });
    } else {
        out.pixels.assign(src, src + static_cast<size_t>(tw) * static_cast<size_t>(oh) * 3);
    }
    return out;
}

}  // namespace dooms
