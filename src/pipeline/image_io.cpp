#include "pipeline/image_io.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

namespace dooms {

namespace {

constexpr int64_t kMaxPixels = int64_t(1) << 28;

// Owns a CoreFoundation object.
struct CfDeleter { void operator()(const void * p) const { if (p) CFRelease(p); } };
template <class T> using CfPtr = std::unique_ptr<std::remove_pointer_t<T>, CfDeleter>;

}  // namespace

Status apply_exif_orientation(int orientation, RgbImage * img) {
    if (!img || img->width < 1 || img->height < 1 || static_cast<int64_t>(img->pixels.size()) != img->width * img->height * 3) {
        return invalid_argument("inconsistent image");
    }
    if (orientation < 1 || orientation > 8) return invalid_argument("EXIF orientation " + std::to_string(orientation) + " is outside 1..8");
    if (orientation == 1) return Status();
    const int64_t w = img->width, h = img->height;
    const bool swap = orientation >= 5;  // 5..8 transpose the axes
    const int64_t ow = swap ? h : w, oh = swap ? w : h;
    std::vector<uint8_t> out(img->pixels.size());
    for (int64_t y = 0; y < h; y++) {
        for (int64_t x = 0; x < w; x++) {
            int64_t nx, ny;  // destination of source pixel (x, y)
            switch (orientation) {
                case 2: nx = w - 1 - x; ny = y; break;            // mirrored horizontally
                case 3: nx = w - 1 - x; ny = h - 1 - y; break;    // rotated 180
                case 4: nx = x; ny = h - 1 - y; break;            // mirrored vertically
                case 5: nx = y; ny = x; break;                    // transposed
                case 6: nx = h - 1 - y; ny = x; break;            // rotated 90 clockwise
                case 7: nx = h - 1 - y; ny = w - 1 - x; break;    // transversed
                default: nx = y; ny = w - 1 - x; break;           // 8: rotated 90 counter-clockwise
            }
            for (int c = 0; c < 3; c++) out[static_cast<size_t>((ny * ow + nx) * 3 + c)] = img->pixels[static_cast<size_t>((y * w + x) * 3 + c)];
        }
    }
    img->pixels = std::move(out);
    img->width = ow;
    img->height = oh;
    return Status();
}

Result<RgbImage> load_image(const std::string & path) {
    std::FILE * probe = std::fopen(path.c_str(), "rb");
    if (!probe) return not_found("cannot open image '" + path + "'");
    std::fclose(probe);

    CfPtr<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(path.c_str()),
                                                                static_cast<CFIndex>(path.size()), false));
    if (!url) return invalid_argument("cannot make a URL from '" + path + "'");
    CfPtr<CGImageSourceRef> source(CGImageSourceCreateWithURL(url.get(), nullptr));
    if (!source || CGImageSourceGetCount(source.get()) < 1) return data_loss("'" + path + "' is not an image the system can decode");
    CfPtr<CGImageRef> image(CGImageSourceCreateImageAtIndex(source.get(), 0, nullptr));
    if (!image) return data_loss("'" + path + "': decoding failed");

    const int64_t w = static_cast<int64_t>(CGImageGetWidth(image.get())), h = static_cast<int64_t>(CGImageGetHeight(image.get()));
    if (w < 1 || h < 1) return data_loss("'" + path + "' has an empty image");
    if (w * h > kMaxPixels) return resource_exhausted("'" + path + "' has more than 2^28 pixels");

    int orientation = 1;
    CfPtr<CFDictionaryRef> props(CGImageSourceCopyPropertiesAtIndex(source.get(), 0, nullptr));
    if (props) {
        const void * v = CFDictionaryGetValue(props.get(), kCGImagePropertyOrientation);
        int value = 1;
        if (v && CFGetTypeID(v) == CFNumberGetTypeID() && CFNumberGetValue(static_cast<CFNumberRef>(v), kCFNumberIntType, &value) && value >= 1 && value <= 8) {
            orientation = value;
        }
    }

    // Draw into an sRGB RGBX buffer (alpha composited over black, wide-gamut and grayscale sources converted).
    CfPtr<CGColorSpaceRef> space(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    if (!space) return internal_error("sRGB color space unavailable");
    std::vector<uint8_t> rgbx(static_cast<size_t>(w * h * 4), 0);
    CfPtr<CGContextRef> ctx(CGBitmapContextCreate(rgbx.data(), static_cast<size_t>(w), static_cast<size_t>(h), 8, static_cast<size_t>(w * 4), space.get(),
                                                  kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault));
    if (!ctx) return internal_error("cannot create a bitmap context");
    CGContextDrawImage(ctx.get(), CGRectMake(0, 0, static_cast<CGFloat>(w), static_cast<CGFloat>(h)), image.get());

    RgbImage out;
    out.width = w;
    out.height = h;
    out.pixels.resize(static_cast<size_t>(w * h * 3));
    for (int64_t i = 0; i < w * h; i++) {
        out.pixels[static_cast<size_t>(3 * i)] = rgbx[static_cast<size_t>(4 * i)];
        out.pixels[static_cast<size_t>(3 * i + 1)] = rgbx[static_cast<size_t>(4 * i + 1)];
        out.pixels[static_cast<size_t>(3 * i + 2)] = rgbx[static_cast<size_t>(4 * i + 2)];
    }
    Status st = apply_exif_orientation(orientation, &out);
    if (!st.is_ok()) return st;
    return out;
}

}  // namespace dooms
