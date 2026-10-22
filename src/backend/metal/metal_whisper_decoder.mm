#include "backend/metal/metal_whisper_decoder.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "backend/metal/llama_kernels.h"

namespace dooms {

namespace {

// Must equal the kernel source's constants (llama_kernels.h).
constexpr uint32_t kSimdgroupsPerGroup = 4;
constexpr uint32_t kGroupThreads = kSimdgroupsPerGroup * 32;
constexpr uint32_t kDecodeRowsPerGroup = 2 * kSimdgroupsPerGroup;   // matvec1: kDecodeRowsPerSimdgroup rows per simdgroup
constexpr uint32_t kMaxStagedInput = 2048;                           // matvec1 stages input rows up to this length (LayerNorm needs the staged path)
constexpr uint32_t kAttnThreads = 8 * 32;                           // attention: kAttnSimdgroups simdgroups
constexpr uint32_t kMaxHeadDim = 256;
constexpr uint32_t kAttentionStaticBytes = 3 * kMaxHeadDim * 4 + kAttnThreads * 16 + 64;
constexpr uint32_t kBigRows = 64, kBigTokens = 32, kBigK = 32;      // matmul_big tile
constexpr uint32_t kEmbedThreads = 256;
constexpr uint32_t kArgmaxThreads = 1024;
constexpr size_t kWeightAlign = 256;
constexpr uint32_t kRopeNone = 2;                                   // AttnParams::neox: learned positions

constexpr uint32_t kFlagBias = 4, kFlagAccumulate = 2, kFlagLayerNorm = 8, kFlagGelu = 16;
constexpr uint32_t kMaxBatch = 8;                                    // matvecm: prompt tokens per call
constexpr uint32_t kSplitTarget = 64;                                // cross-attention: aim for this many threadgroups per token (auto splits)
constexpr int32_t kFusedLayerNormMaxWidth = 512;                     // automatic LayerNorm placement: folded into the projections up to this width
constexpr uint32_t kMinSplitChunk = 64;                              // ... with at least this many positions each

struct MatvecParams { uint32_t K, N, flags; float eps; };
struct GemmParams { uint32_t K, N, M, x_stride, y_stride, flags; float eps; };
struct AttnParams { uint32_t n_head, n_head_kv, head_dim, qkv_stride, kv_stride, out_stride, p0, fused_rope, neox; float scale, base; };
struct EmbedParams { uint32_t n_embd, token_index, pos, use_pos; };
struct ArgmaxParams { uint32_t n, out_index, use_mask; };
struct SplitParams { uint32_t n_head, head_dim, q_stride, kv_stride, n_keys, chunk, n_split, out_stride; float scale; uint32_t group; };
static_assert(sizeof(MatvecParams) == 16 && sizeof(GemmParams) == 28 && sizeof(AttnParams) == 44 && sizeof(EmbedParams) == 16 && sizeof(ArgmaxParams) == 12 &&
                  sizeof(SplitParams) == 40,
              "parameter structs must match the kernel source field for field");

std::string describe(NSError * e) { return e ? std::string([[e localizedDescription] UTF8String]) : std::string("unknown error"); }
size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

enum class Mask { none, always, begin };

}  // namespace

struct WhisperLayerOffsets {
    size_t attn_ln_w, attn_ln_b, qkv, qkv_b, attn_out, attn_out_b;
    size_t cross_ln_w, cross_ln_b, cross_q, cross_q_b, cross_k, cross_v, cross_v_b, cross_out, cross_out_b;
    size_t mlp_ln_w, mlp_ln_b, fc1, fc1_b, fc2, fc2_b;
};

struct MetalWhisperDecoder::Impl {
    WhisperDecoderConfig cfg;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_matvec1, p_matvec1_direct, p_attention, p_embed, p_argmax, p_matmul_big, p_to_half, p_layernorm, p_matvecm, p_rope_store,
        p_attn_split, p_attn_combine;
    uint32_t argmax_threads = 0;
    id<MTLBuffer> weights, kc, vc, ck, cv, x, xn, qkv, q, attn, h, logits_buf, tokens, in_token, prompt_ids, mask_always, mask_begin, enc_in, scratch, part;
    uint32_t cross_splits = 1;
    bool fused_ln = false;
    size_t token_embd = 0, pos_embd = 0, ln_w = 0, ln_b = 0;
    std::vector<WhisperLayerOffsets> layers;
    uint32_t hd = 0;
    size_t self_layer_bytes = 0, cross_layer_bytes = 0;
    int32_t pos = 0;
    bool has_audio = false, has_pending = false;
    id<MTLCommandBuffer> last;
    id<MTLCommandBuffer> cb_;                  // the command buffer being encoded (begin / end)
    id<MTLComputeCommandEncoder> enc_;
    std::vector<std::pair<std::string, double>> * prof = nullptr;   // set by profile_step: every dispatch runs alone and its GPU time is recorded

    void begin() {
        cb_ = prof ? [queue commandBuffer] : [queue commandBufferWithUnretainedReferences];
        enc_ = [cb_ computeCommandEncoder];
    }
    id<MTLCommandBuffer> end() {
        [enc_ endEncoding];
        [cb_ commit];
        id<MTLCommandBuffer> cb = cb_;
        enc_ = nil;
        cb_ = nil;
        return cb;
    }
    // Profiling only: ends the command buffer after the dispatch just encoded, waits for it, records its GPU time under `kind`, starts a new one.
    void cut(const char * kind) {
        if (!prof) return;
        id<MTLCommandBuffer> cb = end();
        [cb waitUntilCompleted];
        prof->emplace_back(kind, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        begin();
    }

    Status wait_idle() {
        if (!last) return Status();
        [last waitUntilCompleted];
        const bool failed = last.status == MTLCommandBufferStatusError;
        NSError * err = last.error;
        last = nil;
        if (failed) return internal_error("a Metal command buffer failed: " + describe(err));
        return Status();
    }

    void matvec1(const char * kind, size_t w, id<MTLBuffer> xb, id<MTLBuffer> yb, uint32_t K, uint32_t N, uint32_t flags, size_t gamma = 0,
                 size_t beta = 0, size_t bias = 0, size_t x_off = 0) {
        MatvecParams p{K, N, flags, cfg.ln_eps};
        const bool stage = K <= kMaxStagedInput;
        [enc_ setComputePipelineState:stage ? p_matvec1 : p_matvec1_direct];
        [enc_ setBuffer:weights offset:w atIndex:0];
        [enc_ setBuffer:xb offset:x_off atIndex:1];
        [enc_ setBuffer:yb offset:0 atIndex:2];
        [enc_ setBuffer:weights offset:gamma atIndex:3];
        [enc_ setBytes:&p length:sizeof p atIndex:4];
        [enc_ setBuffer:weights offset:bias atIndex:5];
        [enc_ setBuffer:weights offset:beta atIndex:6];
        [enc_ setThreadgroupMemoryLength:align_up(static_cast<size_t>(stage ? K : 4) * sizeof(float), 16) atIndex:0];
        [enc_ dispatchThreadgroups:MTLSizeMake((N + kDecodeRowsPerGroup - 1) / kDecodeRowsPerGroup, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        cut(kind);
    }

    // yb[row] = LayerNorm(xb[row]) for M rows of n_state floats starting x_off bytes into xb.
    void layernorm(const char * kind, id<MTLBuffer> xb, size_t x_off, id<MTLBuffer> yb, uint32_t M, size_t gamma, size_t beta) {
        MatvecParams p{static_cast<uint32_t>(cfg.n_state), 0, 0, cfg.ln_eps};
        [enc_ setComputePipelineState:p_layernorm];
        [enc_ setBuffer:xb offset:x_off atIndex:0];
        [enc_ setBuffer:yb offset:0 atIndex:1];
        [enc_ setBuffer:weights offset:gamma atIndex:2];
        [enc_ setBuffer:weights offset:beta atIndex:3];
        [enc_ setBytes:&p length:sizeof p atIndex:4];
        [enc_ dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        cut(kind);
    }

    // A LayerNorm-ed projection of one token: the norm once (layernorm, then matvec1 on the result), or folded into every threadgroup of matvec1 (fused).
    void ln_matvec1(const char * kind, size_t w, uint32_t N, uint32_t flags, size_t gamma, size_t beta, size_t bias, id<MTLBuffer> yb, size_t x_off = 0) {
        const uint32_t d = static_cast<uint32_t>(cfg.n_state);
        if (fused_ln) { matvec1(kind, w, x, yb, d, N, flags | kFlagLayerNorm, gamma, beta, bias, x_off); return; }
        layernorm("layernorm", x, x_off, xn, 1, gamma, beta);
        matvec1(kind, w, xn, yb, d, N, flags, 0, 0, bias);
    }

    void matvecm(const char * kind, size_t w, id<MTLBuffer> xb, id<MTLBuffer> yb, uint32_t K, uint32_t N, uint32_t M, uint32_t flags, size_t bias) {
        MatvecParams p{K, N, flags, cfg.ln_eps};
        [enc_ setComputePipelineState:p_matvecm];
        [enc_ setBuffer:weights offset:w atIndex:0];
        [enc_ setBuffer:xb offset:0 atIndex:1];
        [enc_ setBuffer:yb offset:0 atIndex:2];
        [enc_ setBytes:&p length:sizeof p atIndex:4];
        [enc_ setBuffer:weights offset:bias atIndex:5];
        [enc_ setBytes:&M length:sizeof M atIndex:6];
        [enc_ dispatchThreadgroups:MTLSizeMake((N + kDecodeRowsPerGroup - 1) / kDecodeRowsPerGroup, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        cut(kind);
    }

    // Self-attention of one new token (the attention kernel rotates nothing, caches the token's key/value and attends positions 0 .. p).
    void self_attention(size_t cache_off, uint32_t p) {
        const uint32_t nh = static_cast<uint32_t>(cfg.n_head), d = static_cast<uint32_t>(cfg.n_state);
        AttnParams ap{nh, nh, hd, 3 * d, d, d, p, 1u, kRopeNone, 1.f / std::sqrt(static_cast<float>(hd)), 0.f};
        [enc_ setComputePipelineState:p_attention];
        [enc_ setBuffer:qkv offset:0 atIndex:0];
        [enc_ setBuffer:kc offset:cache_off atIndex:1];
        [enc_ setBuffer:vc offset:cache_off atIndex:2];
        [enc_ setBuffer:attn offset:0 atIndex:3];
        [enc_ setBytes:&ap length:sizeof ap atIndex:4];
        [enc_ setThreadgroupMemoryLength:align_up(static_cast<size_t>(p + 1) * sizeof(float), 16) atIndex:0];
        [enc_ dispatchThreadgroups:MTLSizeMake(nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(kAttnThreads, 1, 1)];
        cut("self attention");
    }

    // Self-attention of M prompt tokens at positions p0 .. p0 + M - 1: rope_store caches their keys/values (no rotation: learned positions), then the attention
    // kernel runs one threadgroup per (head, token) with the causal limit.
    void self_attention_prompt(size_t cache_off, uint32_t p0, uint32_t M) {
        const uint32_t nh = static_cast<uint32_t>(cfg.n_head), d = static_cast<uint32_t>(cfg.n_state);
        AttnParams ap{nh, nh, hd, 3 * d, d, d, p0, 0u, kRopeNone, 1.f / std::sqrt(static_cast<float>(hd)), 0.f};
        [enc_ setComputePipelineState:p_rope_store];
        [enc_ setBuffer:qkv offset:0 atIndex:0];
        [enc_ setBuffer:kc offset:cache_off atIndex:1];
        [enc_ setBuffer:vc offset:cache_off atIndex:2];
        [enc_ setBytes:&ap length:sizeof ap atIndex:3];
        [enc_ dispatchThreadgroups:MTLSizeMake(3 * nh, M, 1) threadsPerThreadgroup:MTLSizeMake(hd / 2, 1, 1)];
        cut("cache k/v");
        [enc_ setComputePipelineState:p_attention];
        [enc_ setBuffer:qkv offset:0 atIndex:0];
        [enc_ setBuffer:kc offset:cache_off atIndex:1];
        [enc_ setBuffer:vc offset:cache_off atIndex:2];
        [enc_ setBuffer:attn offset:0 atIndex:3];
        [enc_ setBytes:&ap length:sizeof ap atIndex:4];
        [enc_ setThreadgroupMemoryLength:align_up(static_cast<size_t>(p0 + M) * sizeof(float), 16) atIndex:0];
        [enc_ dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(kAttnThreads, 1, 1)];
        cut("self attention");
    }

    // Cross-attention of M query tokens (q, stride n_state) over the window's audio positions, split over cross_splits groups of positions, then merged.
    void cross_attention(size_t cache_off, uint32_t M) {
        const uint32_t nh = static_cast<uint32_t>(cfg.n_head), d = static_cast<uint32_t>(cfg.n_state), A = static_cast<uint32_t>(cfg.n_audio_ctx);
        const uint32_t chunk = (A + cross_splits - 1) / cross_splits;
        SplitParams sp{nh, hd, d, d, A, chunk, cross_splits, d, 1.f / std::sqrt(static_cast<float>(hd)), 1};
        [enc_ setComputePipelineState:p_attn_split];
        [enc_ setBuffer:q offset:0 atIndex:0];
        [enc_ setBuffer:ck offset:cache_off atIndex:1];
        [enc_ setBuffer:cv offset:cache_off atIndex:2];
        [enc_ setBuffer:part offset:0 atIndex:3];
        [enc_ setBytes:&sp length:sizeof sp atIndex:4];
        [enc_ setThreadgroupMemoryLength:align_up(static_cast<size_t>(chunk) * sizeof(float), 16) atIndex:0];
        [enc_ dispatchThreadgroups:MTLSizeMake(nh, cross_splits, M) threadsPerThreadgroup:MTLSizeMake(kAttnThreads, 1, 1)];
        cut("cross attention");
        [enc_ setComputePipelineState:p_attn_combine];
        [enc_ setBuffer:part offset:0 atIndex:0];
        [enc_ setBuffer:attn offset:0 atIndex:1];
        [enc_ setBytes:&sp length:sizeof sp atIndex:2];
        [enc_ dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
        cut("cross combine");
    }

    void embed(id<MTLBuffer> ids, uint32_t idx, uint32_t p, uint32_t M) {
        const uint32_t d = static_cast<uint32_t>(cfg.n_state);
        EmbedParams ep{d, idx, p, 1};
        [enc_ setComputePipelineState:p_embed];
        [enc_ setBuffer:weights offset:token_embd atIndex:0];
        [enc_ setBuffer:ids offset:0 atIndex:1];
        [enc_ setBuffer:x offset:0 atIndex:2];
        [enc_ setBytes:&ep length:sizeof ep atIndex:3];
        [enc_ setBuffer:weights offset:pos_embd atIndex:4];
        [enc_ dispatchThreadgroups:MTLSizeMake((d + kEmbedThreads - 1) / kEmbedThreads, M, 1) threadsPerThreadgroup:MTLSizeMake(kEmbedThreads, 1, 1)];
        cut("embed");
    }

    // The final LayerNorm + tied vocabulary projection of the residual row at x_off, then the masked argmax into tokens[out_index].
    void logits_argmax(size_t x_off, uint32_t out_index, Mask mask) {
        const uint32_t V = static_cast<uint32_t>(cfg.n_vocab);
        ln_matvec1("vocabulary", token_embd, V, 0, ln_w, ln_b, 0, logits_buf, x_off);
        ArgmaxParams ap{V, out_index, mask == Mask::none ? 0u : 1u};
        [enc_ setComputePipelineState:p_argmax];
        [enc_ setBuffer:logits_buf offset:0 atIndex:0];
        [enc_ setBuffer:tokens offset:0 atIndex:1];
        [enc_ setBytes:&ap length:sizeof ap atIndex:2];
        [enc_ setBuffer:(mask == Mask::begin ? mask_begin : mask_always) offset:0 atIndex:3];
        [enc_ dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(argmax_threads, 1, 1)];
        cut("argmax");
    }

    // One position: the input token (tokens buffer `ids`, index `idx`) at text position `p`, every layer, the vocabulary projection, the masked argmax into
    // tokens[out_index].
    void encode_step(id<MTLBuffer> ids, uint32_t idx, uint32_t p, uint32_t out_index, Mask mask) {
        const uint32_t d = static_cast<uint32_t>(cfg.n_state), F = static_cast<uint32_t>(cfg.n_ff);
        embed(ids, idx, p, 1);
        for (size_t l = 0; l < layers.size(); l++) {
            const WhisperLayerOffsets & L = layers[l];
            ln_matvec1("self q/k/v", L.qkv, 3 * d, kFlagBias, L.attn_ln_w, L.attn_ln_b, L.qkv_b, qkv);
            self_attention(l * self_layer_bytes, p);
            matvec1("self out", L.attn_out, attn, x, d, d, kFlagBias | kFlagAccumulate, 0, 0, L.attn_out_b);
            ln_matvec1("cross q", L.cross_q, d, kFlagBias, L.cross_ln_w, L.cross_ln_b, L.cross_q_b, q);
            cross_attention(l * cross_layer_bytes, 1);
            matvec1("cross out", L.cross_out, attn, x, d, d, kFlagBias | kFlagAccumulate, 0, 0, L.cross_out_b);
            ln_matvec1("fc1 + GELU", L.fc1, F, kFlagBias | kFlagGelu, L.mlp_ln_w, L.mlp_ln_b, L.fc1_b, h);
            matvec1("fc2", L.fc2, h, x, F, d, kFlagBias | kFlagAccumulate, 0, 0, L.fc2_b);
        }
        logits_argmax(0, out_index, mask);
    }

    // M <= kMaxBatch prompt tokens (prompt_ids[0 .. M)) at positions p0 .. p0 + M - 1 through every layer, each weight read once for all of them; with `logits`,
    // the last token's vocabulary projection and masked argmax into tokens[0].
    void encode_prompt(uint32_t p0, uint32_t M, bool logits, Mask mask) {
        const uint32_t d = static_cast<uint32_t>(cfg.n_state), F = static_cast<uint32_t>(cfg.n_ff);
        embed(prompt_ids, 0, p0, M);
        for (size_t l = 0; l < layers.size(); l++) {
            const WhisperLayerOffsets & L = layers[l];
            layernorm("layernorm", x, 0, xn, M, L.attn_ln_w, L.attn_ln_b);
            matvecm("self q/k/v", L.qkv, xn, qkv, d, 3 * d, M, kFlagBias, L.qkv_b);
            self_attention_prompt(l * self_layer_bytes, p0, M);
            matvecm("self out", L.attn_out, attn, x, d, d, M, kFlagBias | kFlagAccumulate, L.attn_out_b);
            layernorm("layernorm", x, 0, xn, M, L.cross_ln_w, L.cross_ln_b);
            matvecm("cross q", L.cross_q, xn, q, d, d, M, kFlagBias, L.cross_q_b);
            cross_attention(l * cross_layer_bytes, M);
            matvecm("cross out", L.cross_out, attn, x, d, d, M, kFlagBias | kFlagAccumulate, L.cross_out_b);
            layernorm("layernorm", x, 0, xn, M, L.mlp_ln_w, L.mlp_ln_b);
            matvecm("fc1 + GELU", L.fc1, xn, h, d, F, M, kFlagBias | kFlagGelu, L.fc1_b);
            matvecm("fc2", L.fc2, h, x, F, d, M, kFlagBias | kFlagAccumulate, L.fc2_b);
        }
        if (logits) logits_argmax(static_cast<size_t>(M - 1) * d * sizeof(float), 0, mask);
    }

    // K or V for every audio position: scratch = enc_in . W^T (+ bias) (matmul_big), then to half in the cross cache.
    void encode_cross(size_t w, size_t bias, id<MTLBuffer> cache, size_t cache_off) {
        const uint32_t d = static_cast<uint32_t>(cfg.n_state), M = static_cast<uint32_t>(cfg.n_audio_ctx);
        GemmParams gp{d, d, M, d, d, bias ? kFlagBias : 0u, cfg.ln_eps};
        [enc_ setComputePipelineState:p_matmul_big];
        [enc_ setBuffer:weights offset:w atIndex:0];
        [enc_ setBuffer:enc_in offset:0 atIndex:1];
        [enc_ setBuffer:scratch offset:0 atIndex:2];
        [enc_ setBuffer:weights offset:0 atIndex:3];
        [enc_ setBytes:&gp length:sizeof gp atIndex:4];
        [enc_ setBuffer:weights offset:bias atIndex:5];
        [enc_ setBuffer:scratch offset:0 atIndex:6];   // RMS scales: unused (no norm flag) but bound
        [enc_ dispatchThreadgroups:MTLSizeMake((M + kBigTokens - 1) / kBigTokens, (d + kBigRows - 1) / kBigRows, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        cut("cross k/v matmul");
        const uint32_t n = M * d;
        [enc_ setComputePipelineState:p_to_half];
        [enc_ setBuffer:scratch offset:0 atIndex:0];
        [enc_ setBuffer:cache offset:cache_off atIndex:1];
        [enc_ setBytes:&n length:sizeof n atIndex:2];
        [enc_ dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        cut("cross k/v to half");
    }

    Status wait_one(id<MTLCommandBuffer> cb) {
        if (!cb) return Status();
        if (cfg.spin_wait) {
            while (cb.status != MTLCommandBufferStatusCompleted && cb.status != MTLCommandBufferStatusError) __builtin_arm_yield();
        } else {
            [cb waitUntilCompleted];
        }
        if (cb.status == MTLCommandBufferStatusError) return internal_error("a Metal command buffer failed: " + describe(cb.error));
        return Status();
    }
};

MetalWhisperDecoder::MetalWhisperDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalWhisperDecoder::~MetalWhisperDecoder() {
    if (impl_) (void)impl_->wait_idle();
}
const WhisperDecoderConfig & MetalWhisperDecoder::config() const { return impl_->cfg; }
int32_t MetalWhisperDecoder::position() const { return impl_->pos; }

Result<std::unique_ptr<MetalWhisperDecoder>> MetalWhisperDecoder::create(const WhisperDecoderConfig & c, const WhisperDecoderWeights & w) {
    if (c.n_layer < 1 || c.n_state < 32 || c.n_head < 1 || c.n_ff < 32 || c.n_vocab < 1 || c.n_text_ctx < 1 || c.n_audio_ctx < 1 || !(c.ln_eps > 0.f)) {
        return invalid_argument("whisper decoder config: every size must be positive and ln_eps > 0");
    }
    if (c.n_state % c.n_head != 0) return invalid_argument("n_state is not a multiple of n_head");
    const int32_t hd = c.n_state / c.n_head;
    if (hd != 64 && hd != 128) return unsupported("head_dim " + std::to_string(hd) + ": the attention kernel takes 64 or 128 here");
    if (c.n_state % static_cast<int32_t>(kBigK) != 0 || c.n_ff % 4 != 0) return unsupported("n_state must be a multiple of 32 and n_ff of 4");
    if (c.n_state > static_cast<int32_t>(kMaxStagedInput)) return unsupported("n_state above " + std::to_string(kMaxStagedInput) + ": LayerNorm needs the staged input path");
    if (static_cast<int64_t>(w.layers.size()) != c.n_layer) return invalid_argument("expected " + std::to_string(c.n_layer) + " layers of weights");
    const int64_t d = c.n_state, F = c.n_ff, V = c.n_vocab;
    auto check = [](const WeightRef & r, int64_t rows, int64_t cols, bool matrix, const std::string & name) -> Status {
        if (!r.data) return invalid_argument("weight '" + name + "' is missing");
        if (r.rows != rows || r.cols != cols) {
            return invalid_argument("weight '" + name + "' is " + std::to_string(r.rows) + "x" + std::to_string(r.cols) + ", expected " + std::to_string(rows) + "x" + std::to_string(cols));
        }
        if (matrix && r.dtype != DType::F16) return invalid_argument("weight '" + name + "' must be f16");
        if (!matrix && r.dtype != DType::F32) return invalid_argument("vector '" + name + "' must be f32");
        return Status();
    };
    Status st = check(w.token_embd, V, d, true, "token_embd");
    if (st.is_ok()) st = check(w.pos_embd, c.n_text_ctx, d, false, "pos_embd");
    if (st.is_ok()) st = check(w.ln_w, 1, d, false, "ln.weight");
    if (st.is_ok()) st = check(w.ln_b, 1, d, false, "ln.bias");
    for (int32_t l = 0; l < c.n_layer && st.is_ok(); l++) {
        const WhisperLayerWeights & L = w.layers[static_cast<size_t>(l)];
        const std::string b = "blk." + std::to_string(l) + ".";
        for (auto [ref, name] : std::initializer_list<std::pair<const WeightRef *, const char *>>{
                 {&L.attn_ln_w, "attn_ln.weight"}, {&L.attn_ln_b, "attn_ln.bias"}, {&L.attn_q_b, "attn_q.bias"}, {&L.attn_v_b, "attn_v.bias"}, {&L.attn_out_b, "attn_out.bias"},
                 {&L.cross_ln_w, "cross_ln.weight"}, {&L.cross_ln_b, "cross_ln.bias"}, {&L.cross_q_b, "cross_q.bias"}, {&L.cross_v_b, "cross_v.bias"},
                 {&L.cross_out_b, "cross_out.bias"}, {&L.mlp_ln_w, "mlp_ln.weight"}, {&L.mlp_ln_b, "mlp_ln.bias"}, {&L.fc2_b, "fc2.bias"}}) {
            if (st.is_ok()) st = check(*ref, 1, d, false, b + name);
        }
        if (st.is_ok()) st = check(L.fc1_b, 1, F, false, b + "fc1.bias");
        for (auto [ref, name] : std::initializer_list<std::pair<const WeightRef *, const char *>>{
                 {&L.attn_q_w, "attn_q.weight"}, {&L.attn_k_w, "attn_k.weight"}, {&L.attn_v_w, "attn_v.weight"}, {&L.attn_out_w, "attn_out.weight"},
                 {&L.cross_q_w, "cross_q.weight"}, {&L.cross_k_w, "cross_k.weight"}, {&L.cross_v_w, "cross_v.weight"}, {&L.cross_out_w, "cross_out.weight"}}) {
            if (st.is_ok()) st = check(*ref, d, d, true, b + name);
        }
        if (st.is_ok()) st = check(L.fc1_w, F, d, true, b + "fc1.weight");
        if (st.is_ok()) st = check(L.fc2_w, d, F, true, b + "fc2.weight");
    }
    if (!st.is_ok()) return st;

    auto impl = std::make_unique<Impl>();
    Impl & m = *impl;
    m.cfg = c;
    m.hd = static_cast<uint32_t>(hd);
    @autoreleasepool {
        m.device = MTLCreateSystemDefaultDevice();
        if (!m.device) return unsupported("no Metal device");
        const size_t tg_needed = align_up(static_cast<size_t>(std::max(c.n_text_ctx, c.n_audio_ctx)) * sizeof(float), 16) + kAttentionStaticBytes;
        if (tg_needed > m.device.maxThreadgroupMemoryLength) return unsupported("the attention scores of " + std::to_string(std::max(c.n_text_ctx, c.n_audio_ctx)) + " positions do not fit in threadgroup memory");
        m.queue = [m.device newCommandQueue];
        if (!m.queue) return internal_error("could not create a Metal command queue");
        NSError * error = nil;
        id<MTLLibrary> lib = [m.device newLibraryWithSource:[NSString stringWithUTF8String:llama_kernel_source()] options:[MTLCompileOptions new] error:&error];
        if (!lib) return internal_error("could not compile the decoder kernels: " + describe(error));
        auto pipeline = [&](const char * name, bool stage, id<MTLComputePipelineState> __strong & out) -> Status {
            MTLFunctionConstantValues * fc = [MTLFunctionConstantValues new];
            [fc setConstantValue:&stage type:MTLDataTypeBool atIndex:0];
            NSError * fe = nil;
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:fc error:&fe];
            if (!fn) return internal_error(std::string("kernel '") + name + "' could not be specialized: " + describe(fe));
            NSError * e = nil;
            out = [m.device newComputePipelineStateWithFunction:fn error:&e];
            if (!out) return internal_error(std::string("could not build kernel '") + name + "': " + describe(e));
            return Status();
        };
        for (auto & [name, stage, slot] : std::initializer_list<std::tuple<const char *, bool, id<MTLComputePipelineState> __strong *>>{
                 {"matvec1", true, &m.p_matvec1}, {"matvec1", false, &m.p_matvec1_direct}, {"attention", true, &m.p_attention}, {"embed", true, &m.p_embed},
                 {"argmax", true, &m.p_argmax}, {"matmul_big", true, &m.p_matmul_big}, {"to_half", true, &m.p_to_half}, {"layernorm", true, &m.p_layernorm},
                 {"matvecm", true, &m.p_matvecm}, {"rope_store", true, &m.p_rope_store}, {"attention_split", true, &m.p_attn_split},
                 {"attention_combine", true, &m.p_attn_combine}}) {
            if (Status s = pipeline(name, stage, *slot); !s.is_ok()) return s;
        }
        if (m.p_attn_split.maxTotalThreadsPerThreadgroup < kAttnThreads) return unsupported("the split attention kernel cannot run " + std::to_string(kAttnThreads) + " threads per group");
        if (m.p_attention.maxTotalThreadsPerThreadgroup < kAttnThreads) return unsupported("the attention kernel cannot run " + std::to_string(kAttnThreads) + " threads per group");
        m.argmax_threads = static_cast<uint32_t>(std::min<NSUInteger>(kArgmaxThreads, m.p_argmax.maxTotalThreadsPerThreadgroup)) / 32 * 32;

        // Pack the weights: q, k, v rows concatenated (k has no bias: zeros in the packed bias), matrices f16, vectors f32.
        size_t total = 0;
        auto reserve = [&](size_t bytes) { const size_t off = total; total = align_up(total + bytes, kWeightAlign); return off; };
        const size_t vec = static_cast<size_t>(d) * 4, mat = static_cast<size_t>(d * d) * 2;
        m.token_embd = reserve(static_cast<size_t>(V * d) * 2);
        m.pos_embd = reserve(static_cast<size_t>(c.n_text_ctx) * vec);
        m.ln_w = reserve(vec);
        m.ln_b = reserve(vec);
        for (int32_t l = 0; l < c.n_layer; l++) {
            WhisperLayerOffsets o;
            o.attn_ln_w = reserve(vec); o.attn_ln_b = reserve(vec); o.qkv = reserve(3 * mat); o.qkv_b = reserve(3 * vec); o.attn_out = reserve(mat); o.attn_out_b = reserve(vec);
            o.cross_ln_w = reserve(vec); o.cross_ln_b = reserve(vec); o.cross_q = reserve(mat); o.cross_q_b = reserve(vec); o.cross_k = reserve(mat);
            o.cross_v = reserve(mat); o.cross_v_b = reserve(vec); o.cross_out = reserve(mat); o.cross_out_b = reserve(vec);
            o.mlp_ln_w = reserve(vec); o.mlp_ln_b = reserve(vec); o.fc1 = reserve(static_cast<size_t>(F * d) * 2); o.fc1_b = reserve(static_cast<size_t>(F) * 4);
            o.fc2 = reserve(static_cast<size_t>(d * F) * 2); o.fc2_b = reserve(vec);
            m.layers.push_back(o);
        }
        m.weights = [m.device newBufferWithLength:total options:MTLResourceStorageModeShared];
        if (!m.weights) return resource_exhausted("could not allocate " + std::to_string(total) + " bytes for the decoder weights");
        uint8_t * base = static_cast<uint8_t *>([m.weights contents]);
        auto put = [&](size_t off, const WeightRef & r) {
            const size_t bytes = static_cast<size_t>(r.rows * r.cols) * (r.dtype == DType::F16 ? 2 : 4);
            std::memcpy(base + off, r.data, bytes);
            return off + bytes;
        };
        put(m.token_embd, w.token_embd);
        put(m.pos_embd, w.pos_embd);
        put(m.ln_w, w.ln_w);
        put(m.ln_b, w.ln_b);
        for (size_t l = 0; l < w.layers.size(); l++) {
            const WhisperLayerWeights & L = w.layers[l];
            const WhisperLayerOffsets & o = m.layers[l];
            put(o.attn_ln_w, L.attn_ln_w); put(o.attn_ln_b, L.attn_ln_b);
            put(put(put(o.qkv, L.attn_q_w), L.attn_k_w), L.attn_v_w);
            float * qkv_b = reinterpret_cast<float *>(base + o.qkv_b);
            std::memcpy(qkv_b, L.attn_q_b.data, vec);
            std::memset(qkv_b + d, 0, vec);
            std::memcpy(qkv_b + 2 * d, L.attn_v_b.data, vec);
            put(o.attn_out, L.attn_out_w); put(o.attn_out_b, L.attn_out_b);
            put(o.cross_ln_w, L.cross_ln_w); put(o.cross_ln_b, L.cross_ln_b); put(o.cross_q, L.cross_q_w); put(o.cross_q_b, L.cross_q_b);
            put(o.cross_k, L.cross_k_w); put(o.cross_v, L.cross_v_w); put(o.cross_v_b, L.cross_v_b); put(o.cross_out, L.cross_out_w); put(o.cross_out_b, L.cross_out_b);
            put(o.mlp_ln_w, L.mlp_ln_w); put(o.mlp_ln_b, L.mlp_ln_b); put(o.fc1, L.fc1_w); put(o.fc1_b, L.fc1_b); put(o.fc2, L.fc2_w); put(o.fc2_b, L.fc2_b);
        }
        m.self_layer_bytes = static_cast<size_t>(c.n_text_ctx) * static_cast<size_t>(d) * 2;
        m.cross_layer_bytes = static_cast<size_t>(c.n_audio_ctx) * static_cast<size_t>(d) * 2;
        const size_t A = static_cast<size_t>(c.n_audio_ctx);
        if (c.cross_splits > 0) m.cross_splits = static_cast<uint32_t>(std::min<int32_t>(c.cross_splits, c.n_audio_ctx));
        else m.cross_splits = std::max<uint32_t>(1, std::min<uint32_t>((kSplitTarget + c.n_head - 1) / c.n_head, static_cast<uint32_t>(c.n_audio_ctx) / kMinSplitChunk));
        const size_t B = kMaxBatch;
        m.fused_ln = c.fused_layernorm < 0 ? c.n_state <= kFusedLayerNormMaxWidth : c.fused_layernorm != 0;
        auto buffer = [&](size_t bytes, id<MTLBuffer> __strong & out, const char * what) -> Status {
            out = [m.device newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
            if (!out) return resource_exhausted(std::string("could not allocate the decoder's ") + what + " (" + std::to_string(bytes) + " bytes)");
            return Status();
        };
        for (auto & [bytes, slot, what] : std::initializer_list<std::tuple<size_t, id<MTLBuffer> __strong *, const char *>>{
                 {static_cast<size_t>(c.n_layer) * m.self_layer_bytes, &m.kc, "self-attention key cache"}, {static_cast<size_t>(c.n_layer) * m.self_layer_bytes, &m.vc, "self-attention value cache"},
                 {static_cast<size_t>(c.n_layer) * m.cross_layer_bytes, &m.ck, "cross-attention keys"}, {static_cast<size_t>(c.n_layer) * m.cross_layer_bytes, &m.cv, "cross-attention values"},
                 {B * vec, &m.x, "residual stream"}, {B * vec, &m.xn, "normalized rows"}, {B * 3 * vec, &m.qkv, "q/k/v"}, {B * vec, &m.q, "cross query"},
                 {B * vec, &m.attn, "attention output"}, {B * static_cast<size_t>(F) * 4, &m.h, "fc1 output"}, {B * 4, &m.prompt_ids, "prompt ids"},
                 {B * c.n_head * m.cross_splits * (4 + static_cast<size_t>(hd)) * 4, &m.part, "cross-attention partial results"},
                 {static_cast<size_t>(V) * 4, &m.logits_buf, "logits"}, {(static_cast<size_t>(c.n_text_ctx) + 1) * 4, &m.tokens, "token ids"}, {16, &m.in_token, "input token"},
                 {static_cast<size_t>(V) * 4, &m.mask_always, "suppression mask"}, {static_cast<size_t>(V) * 4, &m.mask_begin, "begin suppression mask"},
                 {A * vec, &m.enc_in, "encoder states"}, {A * vec, &m.scratch, "cross key/value scratch"}}) {
            if (Status s = buffer(bytes, *slot, what); !s.is_ok()) return s;
        }
    }
    return std::unique_ptr<MetalWhisperDecoder>(new MetalWhisperDecoder(std::move(impl)));
}

Status MetalWhisperDecoder::set_suppression(const std::vector<int32_t> & always, const std::vector<int32_t> & at_begin) {
    Impl & m = *impl_;
    for (int32_t t : always) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("suppressed token " + std::to_string(t) + " is outside the vocabulary");
    for (int32_t t : at_begin) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("begin-suppressed token " + std::to_string(t) + " is outside the vocabulary");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    float * a = static_cast<float *>([m.mask_always contents]);
    float * b = static_cast<float *>([m.mask_begin contents]);
    std::fill(a, a + m.cfg.n_vocab, 0.f);
    for (int32_t t : always) a[t] = -std::numeric_limits<float>::infinity();
    std::copy(a, a + m.cfg.n_vocab, b);
    for (int32_t t : at_begin) b[t] = -std::numeric_limits<float>::infinity();
    return Status();
}

Status MetalWhisperDecoder::set_audio(const float * states) {
    Impl & m = *impl_;
    if (!states) return invalid_argument("no encoder states");
    const size_t n = static_cast<size_t>(m.cfg.n_audio_ctx) * static_cast<size_t>(m.cfg.n_state);
    for (size_t i = 0; i < n; i++) if (!std::isfinite(states[i])) return invalid_argument("the encoder states contain NaN or Inf");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    std::memcpy([m.enc_in contents], states, n * sizeof(float));
    @autoreleasepool {
        m.begin();
        for (size_t l = 0; l < m.layers.size(); l++) {
            m.encode_cross(m.layers[l].cross_k, 0, m.ck, l * m.cross_layer_bytes);
            m.encode_cross(m.layers[l].cross_v, m.layers[l].cross_v_b, m.cv, l * m.cross_layer_bytes);
        }
        id<MTLCommandBuffer> cb = m.end();
        m.last = cb;
    }
    m.pos = 0;
    m.has_audio = true;
    m.has_pending = false;
    return Status();
}

Status MetalWhisperDecoder::eval_token(int32_t token, bool first_step) {
    Impl & m = *impl_;
    if (!m.has_audio) return invalid_argument("set_audio() has not been called");
    if (token < 0 || token >= m.cfg.n_vocab) return invalid_argument("token " + std::to_string(token) + " is outside the vocabulary");
    if (m.pos >= m.cfg.n_text_ctx) return resource_exhausted("the text context of " + std::to_string(m.cfg.n_text_ctx) + " positions is full");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;   // in_token is rewritten by the host
    static_cast<int32_t *>([m.in_token contents])[0] = token;
    @autoreleasepool {
        m.begin();
        m.encode_step(m.in_token, 0, static_cast<uint32_t>(m.pos), 0, first_step ? Mask::begin : Mask::always);
        id<MTLCommandBuffer> cb = m.end();
        m.last = cb;
    }
    m.pos++;
    m.has_pending = true;
    return Status();
}

Status MetalWhisperDecoder::eval_prompt(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    if (!m.has_audio) return invalid_argument("set_audio() has not been called");
    if (tokens.empty()) return invalid_argument("empty prompt");
    for (int32_t t : tokens) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("token " + std::to_string(t) + " is outside the vocabulary");
    if (m.pos + static_cast<int64_t>(tokens.size()) > m.cfg.n_text_ctx) return resource_exhausted("the prompt does not fit in the text context");
    for (size_t i0 = 0; i0 < tokens.size(); i0 += kMaxBatch) {
        const uint32_t M = static_cast<uint32_t>(std::min<size_t>(kMaxBatch, tokens.size() - i0));
        const bool last_chunk = i0 + M == tokens.size();
        if (Status st = m.wait_idle(); !st.is_ok()) return st;   // prompt_ids is rewritten by the host
        std::memcpy([m.prompt_ids contents], tokens.data() + i0, M * sizeof(int32_t));
        @autoreleasepool {
            m.begin();
            m.encode_prompt(static_cast<uint32_t>(m.pos), M, last_chunk, Mask::begin);
            m.last = m.end();
        }
        m.pos += static_cast<int32_t>(M);
    }
    m.has_pending = true;
    return Status();
}

Result<int32_t> MetalWhisperDecoder::pending_token() {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    return static_cast<const int32_t *>([m.tokens contents])[0];
}

Result<std::vector<float>> MetalWhisperDecoder::logits() {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    const float * l = static_cast<const float *>([m.logits_buf contents]);
    return std::vector<float>(l, l + m.cfg.n_vocab);
}

Status MetalWhisperDecoder::generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token) {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (max_tokens < 1 || depth < 1) return invalid_argument("max_tokens and depth must be >= 1");
    const int32_t base = m.pos;
    if (static_cast<int64_t>(base) + max_tokens - 1 > m.cfg.n_text_ctx) return resource_exhausted("the text context of " + std::to_string(m.cfg.n_text_ctx) + " positions would overflow");
    std::deque<id<MTLCommandBuffer>> inflight;
    inflight.push_back(m.last);
    int32_t produced = 0, next = 1;
    Status st;
    while (true) {
        while (next < max_tokens && next - produced <= depth) {
            @autoreleasepool {
                m.begin();
                m.encode_step(m.tokens, static_cast<uint32_t>(next - 1), static_cast<uint32_t>(base + next - 1), static_cast<uint32_t>(next), Mask::always);
                id<MTLCommandBuffer> cb = m.end();
                m.last = cb;
                inflight.push_back(cb);
            }
            next++;
        }
        id<MTLCommandBuffer> cb = inflight.front();
        inflight.pop_front();
        st = m.wait_one(cb);
        if (!st.is_ok()) break;
        const int32_t token = static_cast<const int32_t *>([m.tokens contents])[produced];
        produced++;
        if (!on_token(token) || produced == max_tokens) break;
    }
    if (!st.is_ok()) (void)m.wait_idle();
    m.pos = base + produced - 1;
    m.has_pending = false;
    return st;
}

Result<std::vector<std::pair<std::string, double>>> MetalWhisperDecoder::profile_step() {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (m.pos >= m.cfg.n_text_ctx) return resource_exhausted("the text context is full");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    std::vector<std::pair<std::string, double>> out;
    @autoreleasepool {
        m.prof = &out;
        m.begin();
        m.encode_step(m.tokens, 0, static_cast<uint32_t>(m.pos), 0, Mask::always);   // the pending token at the next position (rewritten by the next real step)
        id<MTLCommandBuffer> cb = m.end();
        [cb waitUntilCompleted];
        m.prof = nullptr;
    }
    return out;
}

}  // namespace dooms
