#pragma once
// Image decoding (macOS ImageIO: JPEG, PNG, WebP, HEIC, TIFF, ... whatever the system decodes).
// The result matches cv2.imread's default (IMREAD_COLOR): 8-bit RGB, EXIF orientation applied, alpha dropped, grayscale expanded.
// Differences from OpenCV that cannot be removed: JPEG decoders may differ by a few grey levels; non-sRGB profiles are converted
// to sRGB by ImageIO while OpenCV ignores them; semi-transparent pixels are composited over black here.

#include <string>

#include "core/status.h"
#include "pipeline/rgb_image.h"

namespace dooms {

// NotFound: no such file. DataLoss: the file is not an image the system can decode. ResourceExhausted: more than 2^28 pixels.
Result<RgbImage> load_image(const std::string & path);

// The 8 EXIF orientations applied to an RGB image (1 = unchanged ... 8 = rotated 90 degrees counter-clockwise on display).
// InvalidArgument for orientation outside 1..8 or an inconsistent image.
Status apply_exif_orientation(int orientation, RgbImage * image);

}  // namespace dooms
