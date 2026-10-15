#pragma once
// SmolVLA (vision-language-action, flow matching): camera images + task tokens + robot state -> a chunk of actions.
//   cameras -> resize with padding -> SigLIP towers (Core ML, Neural Engine) -> image embeddings
//   + language embeddings + state projection -> prefix tokens -> 16 VLM layers (Core ML) -> key/value cache of every layer
//   noise -> all flow-matching steps of the action expert in one Core ML call, reading the cache -> actions
// Files from tools/convert: vision-b1-fp16.mlpackage (convert_smolvla_vision.py), prefix-fp16.mlpackage and expert-fp16.mlpackage (convert_smolvla_coreml.py),
// host.gguf (convert_smolvla_host.py). The packages are fixed to the prefix length of the conversion (cameras x tokens + prompt length + state).
// Research/robotics use; SmolVLA and SmolVLM2 are Apache-2.0.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend/coreml/coreml_model.h"
#include "core/status.h"
#include "pipeline/rgb_image.h"

namespace dooms {

struct VlaPaths {
    std::string vision, prefix, expert, host;
};

struct VlaOptions {
    std::string vision_units = "cpu_ne", prefix_units = "cpu_ne", expert_units = "cpu_ne";  // "all" | "cpu_gpu" | "cpu_ne" | "cpu"
    uint64_t seed = 0;  // noise seed when the caller gives none
};

struct CameraImage {
    int64_t width = 0, height = 0;
    std::vector<float> chw;  // 3 * height * width, RGB, values in 0..1 (the policy scales to -1..1 itself)
};

struct VlaObservation {
    std::vector<CameraImage> cameras;  // exactly n_cameras(), in the order of the policy's image features
    std::vector<int32_t> tokens;       // the tokenized task, padded to tokenizer_max_length() (right padding)
    std::vector<uint8_t> token_mask;   // 1 for real tokens, same length as tokens
    std::vector<float> state;          // normalized robot state, at most state_dim() values (zero-padded here)
};

// The policy's camera preprocessing (lerobot's resize_with_pad with pad value 0, then x * 2 - 1): bilinear resize keeping the aspect ratio so that the longer side
// is `size`, padding on the top and left, into out[3 * size * size]. InvalidArgument for an empty/oversized/mismatched/non-finite image.
Status vla_prepare_camera(const CameraImage & image, int64_t size, float * out);

// A camera frame as the pipeline wants it: interleaved 8-bit RGB (from ImageIO, a camera driver, ...) -> planar float 0..1. InvalidArgument for an empty or mismatched frame.
Result<CameraImage> camera_from_rgb(const RgbImage & frame);

// The policy's MEAN_STD normalization with the statistics of the dataset it was trained on (given by the caller): (x - mean) / (std + 1e-8) for the state, x * std + mean for
// the actions. mean/std have one value per robot dimension; InvalidArgument for a size mismatch, non-finite values or a negative std.
Result<std::vector<float>> vla_normalize_state(const std::vector<float> & state, const std::vector<float> & mean, const std::vector<float> & stddev);
// actions: rows of `action_dim` values (the padded width); only the first mean.size() columns are returned, rows of mean.size() values.
Result<std::vector<float>> vla_unnormalize_actions(const std::vector<float> & actions, int64_t action_dim, const std::vector<float> & mean, const std::vector<float> & stddev);

struct VlaTimings {
    double vision_ms = 0, host_ms = 0, prefix_ms = 0, expert_ms = 0, total_ms = 0;
};

// Not thread-safe.
class VlaPipeline {
public:
    //   NotFound/InvalidArgument/DataLoss  as the loaders report them; InvalidArgument when the files do not belong together
    static Result<VlaPipeline> create(const VlaPaths & paths, const VlaOptions & options = VlaOptions());

    VlaPipeline(VlaPipeline &&) noexcept;
    VlaPipeline & operator=(VlaPipeline &&) noexcept;
    ~VlaPipeline();

    int64_t n_cameras() const;
    int64_t tokenizer_max_length() const;
    int64_t state_dim() const;
    int64_t chunk_size() const;
    int64_t action_dim() const;  // padded width of each action row
    int64_t prefix_length() const;
    int64_t steps() const;

    // Returns chunk_size() x action_dim() values (row-major); the robot's own action dimensions are the first columns.
    // `noise` (optional, chunk_size() * action_dim() values) replaces the random start of the flow.
    //   InvalidArgument  an observation that does not match the model (camera count, token count, mask, state size, non-finite values)
    Result<std::vector<float>> infer(const VlaObservation & observation, const std::vector<float> * noise = nullptr, VlaTimings * timings = nullptr);

private:
    struct Impl;
    explicit VlaPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
