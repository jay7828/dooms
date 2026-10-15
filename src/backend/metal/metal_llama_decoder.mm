#include "backend/metal/metal_llama_decoder.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <string>
#include <utility>

#include "backend/metal/llama_kernels.h"

namespace dooms {

namespace {

// Must equal the kernel source's constants (llama_kernels.h).
constexpr uint32_t kSimdgroupsPerGroup = 4;
constexpr uint32_t kGroupThreads = kSimdgroupsPerGroup * 32;
constexpr uint32_t kDecodeRowsPerSimdgroup = 2;        // matvec1: rows per simdgroup
constexpr uint32_t kGateRowsPerSimdgroup = 1;          // gate_up1: outputs per simdgroup
constexpr uint32_t kDecodeRowsPerGroup = kDecodeRowsPerSimdgroup * kSimdgroupsPerGroup;
constexpr uint32_t kGateRowsPerGroup = kGateRowsPerSimdgroup * kSimdgroupsPerGroup;
// Prefill matmuls: 16x16 tiles (matmul_16x16, many threadgroups for short chunks) or 64 rows x 32 tokens (matmul_big, llama.cpp's structure) from
// kBigMinTokens tokens on, when every reduction length is a multiple of kBigK. Crossover measured: src/backend/metal/CLAUDE.md "Llama decoder".
constexpr uint32_t kSmallTile = 16;
constexpr uint32_t kBigRows = 64, kBigTokens = 32, kBigK = 32;
constexpr uint32_t kBigMinTokens = 64;
constexpr uint32_t kFlashQ = 32;                       // attention_prefill: queries per threadgroup (llama_kernels.h)
constexpr uint32_t kMaxHeadDim = 256;
constexpr uint32_t kAttnThreads = 8 * 32;              // attention threadgroup (llama_kernels.h kAttnSimdgroups)
// Input rows longer than this are read from device memory by matvec1/gate_up1 instead of being staged in threadgroup memory (kStageInput = false): a
// staged row of n floats takes 4n bytes, and above ~8 KB fewer threadgroups fit on a core. Measured: src/backend/metal/CLAUDE.md "Llama decoder".
constexpr uint32_t kMaxStagedInput = 2048;
// The attention kernel's static threadgroup arrays: red[8] floats, qs4/kn4/vn4[kMaxHeadDim/4] float4, partial4[kAttnThreads] float4; rounded up.
constexpr uint32_t kAttentionStaticBytes = 3 * kMaxHeadDim * 4 + kAttnThreads * 16 + 64;
constexpr uint32_t kEmbedThreads = 256;
constexpr uint32_t kArgmaxThreads = 1024;              // upper bound; clamped to what the pipeline allows
constexpr size_t kWeightAlign = 256;                   // offset alignment of each tensor inside the weight buffer

constexpr uint32_t kFlagNorm = 1, kFlagAccumulate = 2, kFlagBias = 4;  // MatvecParams / GemmParams flags

struct MatvecParams { uint32_t K, N, flags; float eps; };
struct GemmParams { uint32_t K, N, M, x_stride, y_stride, flags; float eps; };
struct AttnParams { uint32_t n_head, n_head_kv, head_dim, qkv_stride, kv_stride, out_stride, p0, fused_rope, neox; float scale, base; };
struct EmbedParams { uint32_t n_embd, token_index; };
struct ArgmaxParams { uint32_t n, out_index; };
static_assert(sizeof(MatvecParams) == 16 && sizeof(GemmParams) == 28 && sizeof(AttnParams) == 44 && sizeof(EmbedParams) == 8 && sizeof(ArgmaxParams) == 8,
              "parameter structs must match the kernel source field for field");

std::string describe(NSError * e) { return e ? std::string([[e localizedDescription] UTF8String]) : std::string("unknown error"); }

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

struct LayerOffsets { size_t attn_norm, qkv, qkv_bias, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down; bool has_bias; };

struct MetalLlamaDecoder::Impl {
    LlamaDecoderConfig cfg;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_matvec1, p_gate_up1, p_rope, p_attention, p_embed, p_argmax;
    id<MTLComputePipelineState> p_matmul, p_matmul_gate_up, p_matmul_big, p_gate_up_big, p_rms_scale;
    id<MTLComputePipelineState> p_attn_prefill;   // flash attention for chunks of several tokens (head_dim 64 or 128), else nil
    bool big = false;        // the current encode_forward call uses the 64x32 kernels
    bool big_capable = false;  // every reduction length (n_embd, n_head*head_dim, n_ff) is a multiple of kBigK
    id<MTLComputePipelineState> p_matvec1_direct, p_gate_up1_direct;  // kStageInput = false variants (long input rows)
    uint32_t argmax_threads = 0;
    id<MTLBuffer> weights, kc, vc, x, qkv, attn, h, logits_buf, tokens, in_tokens, scale;  // scale: per-token RMS scales of a prefill chunk
    size_t token_embd = 0, output_norm = 0, output = 0;
    std::vector<LayerOffsets> layers;
    uint32_t qkv_width = 0, kv_width = 0, q_width = 0;
    size_t cache_layer_bytes = 0;
    int32_t pos = 0;
    bool has_pending = false;
    id<MTLCommandBuffer> last;   // the most recently committed command buffer (all earlier ones on the queue finish before it)
    // Profiling: when set, every dispatch ends the current encoder and command buffer, which is committed, waited for and timed under `kind`.
    std::vector<MetalLlamaDecoder::KernelTime> * profile = nullptr;
    id<MTLCommandBuffer> prof_cb;
    id<MTLComputeCommandEncoder> prof_enc;

    // Called after each dispatch: in profiling mode, closes the dispatch's own command buffer and records its GPU time.
    void after_dispatch(const char * kind) {
        if (!profile) return;
        [prof_enc endEncoding];
        [prof_cb commit];
        [prof_cb waitUntilCompleted];
        const double ms = (prof_cb.GPUEndTime - prof_cb.GPUStartTime) * 1e3;
        auto it = std::find_if(profile->begin(), profile->end(), [&](const MetalLlamaDecoder::KernelTime & k) { return std::string(k.kind) == kind; });
        if (it == profile->end()) profile->push_back({kind, 1, ms});
        else { it->dispatches++; it->gpu_ms += ms; }
        prof_cb = [queue commandBuffer];
        prof_enc = [prof_cb computeCommandEncoder];
    }
    id<MTLComputeCommandEncoder> enc_now(id<MTLComputeCommandEncoder> enc) const { return profile ? prof_enc : enc; }

    Status wait_idle() {
        if (!last) return Status();
        [last waitUntilCompleted];
        const bool failed = last.status == MTLCommandBufferStatusError;
        NSError * err = last.error;
        last = nil;
        if (failed) return internal_error("a Metal command buffer failed: " + describe(err));
        return Status();
    }

    // One token: y (+)= W . x (x optionally RMS-normalized with gamma).
    void matvec1(id<MTLComputeCommandEncoder> enc0, const char * kind, size_t w, id<MTLBuffer> xb, size_t x_off, id<MTLBuffer> yb, size_t gamma, uint32_t K, uint32_t N,
                 uint32_t flags, size_t bias = 0) {
        id<MTLComputeCommandEncoder> enc = enc_now(enc0);
        MatvecParams p{K, N, flags, cfg.rms_eps};
        const bool stage = K <= kMaxStagedInput;
        [enc setComputePipelineState:stage ? p_matvec1 : p_matvec1_direct];
        [enc setBuffer:weights offset:w atIndex:0];
        [enc setBuffer:xb offset:x_off atIndex:1];
        [enc setBuffer:yb offset:0 atIndex:2];
        [enc setBuffer:weights offset:gamma atIndex:3];
        [enc setBytes:&p length:sizeof p atIndex:4];
        [enc setBuffer:weights offset:bias atIndex:5];
        [enc setThreadgroupMemoryLength:align_up(static_cast<size_t>(stage ? K : 4) * sizeof(float), 16) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((N + kDecodeRowsPerGroup - 1) / kDecodeRowsPerGroup, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        after_dispatch(kind);
    }

    // Per-token RMS scales of the chunk's rows of x (for the 64x32 kernels).
    void rms_scale(id<MTLComputeCommandEncoder> enc0, uint32_t M) {
        id<MTLComputeCommandEncoder> enc = enc_now(enc0);
        GemmParams p{static_cast<uint32_t>(cfg.n_embd), 0, M, static_cast<uint32_t>(cfg.n_embd), 0, 0, cfg.rms_eps};
        [enc setComputePipelineState:p_rms_scale];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:scale offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof p atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        after_dispatch("rms_scale");
    }

    // Several tokens: Y[M][N] (+)= X . W^T (X optionally RMS-normalized with gamma).
    void matmul(id<MTLComputeCommandEncoder> enc0, const char * kind, size_t w, id<MTLBuffer> xb, id<MTLBuffer> yb, size_t gamma,
                uint32_t K, uint32_t N, uint32_t M, uint32_t x_stride, uint32_t y_stride, uint32_t flags, size_t bias = 0) {
        id<MTLComputeCommandEncoder> enc = enc_now(enc0);
        GemmParams p{K, N, M, x_stride, y_stride, flags, cfg.rms_eps};
        [enc setComputePipelineState:big ? p_matmul_big : p_matmul];
        [enc setBuffer:weights offset:w atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:yb offset:0 atIndex:2];
        [enc setBuffer:weights offset:gamma atIndex:3];
        [enc setBytes:&p length:sizeof p atIndex:4];
        [enc setBuffer:weights offset:bias atIndex:5];
        [enc setBuffer:scale offset:0 atIndex:6];
        const uint32_t tn = big ? kBigRows : kSmallTile, tm = big ? kBigTokens : kSmallTile;
        // The 64x32 kernels take token tiles along x (neighbouring threadgroups share weight rows); the 16x16 ones take row tiles along x.
        const MTLSize grid = big ? MTLSizeMake((M + tm - 1) / tm, (N + tn - 1) / tn, 1) : MTLSizeMake((N + tn - 1) / tn, (M + tm - 1) / tm, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
        after_dispatch(kind);
    }

    void attention(id<MTLComputeCommandEncoder> enc0, size_t cache_off, uint32_t M, uint32_t p0, bool fused_rope) {
        const uint32_t nh = static_cast<uint32_t>(cfg.n_head), nkv = static_cast<uint32_t>(cfg.n_head_kv), hd = static_cast<uint32_t>(cfg.head_dim);
        AttnParams ap{nh, nkv, hd, qkv_width, kv_width, q_width, p0, fused_rope ? 1u : 0u, cfg.rope_neox ? 1u : 0u, 1.f / std::sqrt(static_cast<float>(hd)), cfg.rope_base};
        id<MTLComputeCommandEncoder> enc = enc_now(enc0);
        if (!fused_rope) {
            [enc setComputePipelineState:p_rope];
            [enc setBuffer:qkv offset:0 atIndex:0];
            [enc setBuffer:kc offset:cache_off atIndex:1];
            [enc setBuffer:vc offset:cache_off atIndex:2];
            [enc setBytes:&ap length:sizeof ap atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(nh + 2 * nkv, M, 1) threadsPerThreadgroup:MTLSizeMake(hd / 2, 1, 1)];
            after_dispatch("rope_store");
            enc = enc_now(enc0);
        }
        if (!fused_rope && p_attn_prefill) {
            [enc setComputePipelineState:p_attn_prefill];
            [enc setBuffer:qkv offset:0 atIndex:0];
            [enc setBuffer:kc offset:cache_off atIndex:1];
            [enc setBuffer:vc offset:cache_off atIndex:2];
            [enc setBuffer:attn offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof ap atIndex:4];
            [enc setBytes:&M length:sizeof M atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(nh, (M + kFlashQ - 1) / kFlashQ, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
            after_dispatch("attention");
            return;
        }
        [enc setComputePipelineState:p_attention];
        [enc setBuffer:qkv offset:0 atIndex:0];
        [enc setBuffer:kc offset:cache_off atIndex:1];
        [enc setBuffer:vc offset:cache_off atIndex:2];
        [enc setBuffer:attn offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof ap atIndex:4];
        [enc setThreadgroupMemoryLength:align_up(static_cast<size_t>(p0 + M) * sizeof(float), 16) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(kAttnThreads, 1, 1)];
        after_dispatch("attention");
    }

    // All layers for M positions starting at p0, whose input embeddings are in x rows 0..M-1; then the logits and greedy token of the last row into
    // tokens[out_index]. One token: 5 kernels per layer (rotation and cache write happen inside attention). Several: 6 (rope_store first).
    void encode_forward(id<MTLComputeCommandEncoder> enc0, uint32_t M, uint32_t p0, uint32_t out_index) {
        const uint32_t E = static_cast<uint32_t>(cfg.n_embd), F = static_cast<uint32_t>(cfg.n_ff);
        big = big_capable && (cfg.prefill_tile == 2 || (cfg.prefill_tile == 0 && M >= kBigMinTokens));
        for (size_t l = 0; l < layers.size(); l++) {
            const LayerOffsets & L = layers[l];
            const size_t cache_off = l * cache_layer_bytes;
            if (M == 1) {
                matvec1(enc0, "qkv", L.qkv, x, 0, qkv, L.attn_norm, E, qkv_width, kFlagNorm | (L.has_bias ? kFlagBias : 0u), L.qkv_bias);
                attention(enc0, cache_off, 1, p0, true);
                matvec1(enc0, "attn_output", L.attn_output, attn, 0, x, 0, q_width, E, kFlagAccumulate);
                id<MTLComputeCommandEncoder> enc = enc_now(enc0);
                MatvecParams gp{E, F, kFlagNorm, cfg.rms_eps};
                const bool stage = E <= kMaxStagedInput;
                [enc setComputePipelineState:stage ? p_gate_up1 : p_gate_up1_direct];
                [enc setBuffer:weights offset:L.ffn_gate atIndex:0];
                [enc setBuffer:weights offset:L.ffn_up atIndex:1];
                [enc setBuffer:x offset:0 atIndex:2];
                [enc setBuffer:h offset:0 atIndex:3];
                [enc setBuffer:weights offset:L.ffn_norm atIndex:4];
                [enc setBytes:&gp length:sizeof gp atIndex:5];
                [enc setThreadgroupMemoryLength:align_up(static_cast<size_t>(stage ? E : 4) * sizeof(float), 16) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((F + kGateRowsPerGroup - 1) / kGateRowsPerGroup, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
                after_dispatch("gate_up");
                matvec1(enc0, "ffn_down", L.ffn_down, h, 0, x, 0, F, E, kFlagAccumulate);
            } else {
                if (big) rms_scale(enc0, M);
                matmul(enc0, "qkv", L.qkv, x, qkv, L.attn_norm, E, qkv_width, M, E, qkv_width, kFlagNorm | (L.has_bias ? kFlagBias : 0u), L.qkv_bias);
                attention(enc0, cache_off, M, p0, false);
                matmul(enc0, "attn_output", L.attn_output, attn, x, 0, q_width, E, M, q_width, E, kFlagAccumulate);
                if (big) rms_scale(enc0, M);
                id<MTLComputeCommandEncoder> enc = enc_now(enc0);
                GemmParams gp{E, F, M, E, F, kFlagNorm, cfg.rms_eps};
                [enc setComputePipelineState:big ? p_gate_up_big : p_matmul_gate_up];
                [enc setBuffer:weights offset:L.ffn_gate atIndex:0];
                [enc setBuffer:weights offset:L.ffn_up atIndex:1];
                [enc setBuffer:x offset:0 atIndex:2];
                [enc setBuffer:h offset:0 atIndex:3];
                [enc setBuffer:weights offset:L.ffn_norm atIndex:4];
                [enc setBytes:&gp length:sizeof gp atIndex:5];
                [enc setBuffer:scale offset:0 atIndex:6];
                const uint32_t tn = big ? kBigRows : kSmallTile, tm = big ? kBigTokens : kSmallTile;
                const MTLSize grid = big ? MTLSizeMake((M + tm - 1) / tm, (F + tn - 1) / tn, 1) : MTLSizeMake((F + tn - 1) / tn, (M + tm - 1) / tm, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(kGroupThreads, 1, 1)];
                after_dispatch("gate_up");
                matmul(enc0, "ffn_down", L.ffn_down, h, x, 0, F, E, M, F, E, kFlagAccumulate);
            }
        }
        const uint32_t V = static_cast<uint32_t>(cfg.n_vocab);
        matvec1(enc0, "output", output, x, static_cast<size_t>(M - 1) * E * sizeof(float), logits_buf, output_norm, E, V, kFlagNorm);
        id<MTLComputeCommandEncoder> enc = enc_now(enc0);
        ArgmaxParams ap{V, out_index};
        [enc setComputePipelineState:p_argmax];
        [enc setBuffer:logits_buf offset:0 atIndex:0];
        [enc setBuffer:tokens offset:0 atIndex:1];
        [enc setBytes:&ap length:sizeof ap atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(argmax_threads, 1, 1)];
        after_dispatch("argmax");
    }

    void encode_embed(id<MTLComputeCommandEncoder> enc0, id<MTLBuffer> ids, uint32_t first, uint32_t rows) {
        id<MTLComputeCommandEncoder> enc = enc_now(enc0);
        EmbedParams ep{static_cast<uint32_t>(cfg.n_embd), first};
        [enc setComputePipelineState:p_embed];
        [enc setBuffer:weights offset:token_embd atIndex:0];
        [enc setBuffer:ids offset:0 atIndex:1];
        [enc setBuffer:x offset:0 atIndex:2];
        [enc setBytes:&ep length:sizeof ep atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((static_cast<uint32_t>(cfg.n_embd) + kEmbedThreads - 1) / kEmbedThreads, rows, 1)
             threadsPerThreadgroup:MTLSizeMake(kEmbedThreads, 1, 1)];
        after_dispatch("embed");
    }

    // Runs one chunk (rows already in x, or token ids in in_tokens when from_tokens) and commits it. The chunk's last row's greedy token lands in tokens[0].
    void run_chunk(uint32_t rows, bool from_tokens) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (from_tokens) encode_embed(enc, in_tokens, 0, rows);
            encode_forward(enc, rows, static_cast<uint32_t>(pos), 0);
            [enc endEncoding];
            [cb commit];
            last = cb;
        }
        pos += static_cast<int32_t>(rows);
        has_pending = true;
    }
};

MetalLlamaDecoder::MetalLlamaDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalLlamaDecoder::~MetalLlamaDecoder() {
    if (impl_) (void)impl_->wait_idle();
}

const LlamaDecoderConfig & MetalLlamaDecoder::config() const { return impl_->cfg; }
int32_t MetalLlamaDecoder::position() const { return impl_->pos; }

Result<std::unique_ptr<MetalLlamaDecoder>> MetalLlamaDecoder::create(const LlamaDecoderConfig & c, const LlamaDecoderWeights & w) {
    if (c.n_layer < 1 || c.n_embd < 4 || c.n_head < 1 || c.n_head_kv < 1 || c.head_dim < 2 || c.n_ff < 4 || c.n_vocab < 1 || c.n_ctx < 1 || c.n_batch < 1 ||
        !(c.rope_base > 0.f) || !(c.rms_eps > 0.f)) {
        return invalid_argument("llama decoder config: every size must be positive, rope_base and rms_eps > 0");
    }
    if (c.n_head % c.n_head_kv != 0) return invalid_argument("n_head must be a multiple of n_head_kv");
    if (c.prefill_tile < 0 || c.prefill_tile > 2) return invalid_argument("prefill_tile must be 0 (automatic), 1 (16x16) or 2 (64x32)");
    // The attention kernel gives each thread 4 dimensions of a position and needs kGroupThreads to split evenly: a power of two in [4, kMaxHeadDim].
    if (c.head_dim < 4 || c.head_dim > static_cast<int32_t>(kMaxHeadDim) || (c.head_dim & (c.head_dim - 1)) != 0) {
        return unsupported("head_dim must be a power of two in [4, " + std::to_string(kMaxHeadDim) + "], got " + std::to_string(c.head_dim));
    }
    if (c.n_embd % 4 != 0 || c.n_ff % 4 != 0 || (c.n_head * c.head_dim) % 4 != 0) return unsupported("n_embd, n_ff and n_head*head_dim must be multiples of 4");
    if (static_cast<int64_t>(w.layers.size()) != c.n_layer) return invalid_argument("expected " + std::to_string(c.n_layer) + " layers of weights, got " + std::to_string(w.layers.size()));

    auto impl = std::make_unique<Impl>();
    Impl & m = *impl;
    m.cfg = c;
    m.q_width = static_cast<uint32_t>(c.n_head * c.head_dim);
    m.kv_width = static_cast<uint32_t>(c.n_head_kv * c.head_dim);
    m.qkv_width = m.q_width + 2 * m.kv_width;

    // Shapes and dtypes, then the packed layout of the weight buffer.
    auto check = [](const WeightRef & r, int64_t rows, int64_t cols, bool matrix, const std::string & name) -> Status {
        if (!r.data) return invalid_argument("weight '" + name + "' is missing");
        if (r.rows != rows || r.cols != cols) {
            return invalid_argument("weight '" + name + "' is " + std::to_string(r.rows) + "x" + std::to_string(r.cols) + ", expected " + std::to_string(rows) + "x" + std::to_string(cols));
        }
        if (matrix && r.dtype != DType::F16) return unsupported("weight '" + name + "' is " + dtype_name(r.dtype) + "; the Metal decoder takes f16 matrices");
        if (!matrix && r.dtype != DType::F32 && r.dtype != DType::F16) return unsupported("norm '" + name + "' is " + dtype_name(r.dtype) + "; expected f32 or f16");
        return Status();
    };
    const int64_t E = c.n_embd, F = c.n_ff, V = c.n_vocab;
    Status st = check(w.token_embd, V, E, true, "token_embd");
    if (st.is_ok()) st = check(w.output_norm, 1, E, false, "output_norm");
    if (st.is_ok()) st = check(w.output, V, E, true, "output");
    for (int32_t l = 0; l < c.n_layer && st.is_ok(); l++) {
        const LlamaLayerWeights & L = w.layers[static_cast<size_t>(l)];
        const std::string b = "blk." + std::to_string(l) + ".";
        st = check(L.attn_norm, 1, E, false, b + "attn_norm");
        if (st.is_ok()) {
            if (L.attn_qkv.data) st = check(L.attn_qkv, m.qkv_width, E, true, b + "attn_qkv");
            else {
                st = check(L.attn_q, m.q_width, E, true, b + "attn_q");
                if (st.is_ok()) st = check(L.attn_k, m.kv_width, E, true, b + "attn_k");
                if (st.is_ok()) st = check(L.attn_v, m.kv_width, E, true, b + "attn_v");
            }
        }
        if (st.is_ok()) {
            const bool split = L.attn_q_bias.data || L.attn_k_bias.data || L.attn_v_bias.data;
            if (split && L.attn_qkv_bias.data) st = invalid_argument(b + "attn: give attn_qkv_bias or attn_q/k/v_bias, not both");
            else if (L.attn_qkv_bias.data) st = check(L.attn_qkv_bias, 1, m.qkv_width, false, b + "attn_qkv_bias");
            else if (split) {
                st = check(L.attn_q_bias, 1, m.q_width, false, b + "attn_q_bias");
                if (st.is_ok()) st = check(L.attn_k_bias, 1, m.kv_width, false, b + "attn_k_bias");
                if (st.is_ok()) st = check(L.attn_v_bias, 1, m.kv_width, false, b + "attn_v_bias");
            }
        }
        if (st.is_ok()) st = check(L.attn_output, E, m.q_width, true, b + "attn_output");
        if (st.is_ok()) st = check(L.ffn_norm, 1, E, false, b + "ffn_norm");
        if (st.is_ok()) st = check(L.ffn_gate, F, E, true, b + "ffn_gate");
        if (st.is_ok()) st = check(L.ffn_up, F, E, true, b + "ffn_up");
        if (st.is_ok()) st = check(L.ffn_down, E, F, true, b + "ffn_down");
    }
    if (!st.is_ok()) return st;

    @autoreleasepool {
        m.device = MTLCreateSystemDefaultDevice();
        if (!m.device) return unsupported("no Metal device");
        // The scores of one attention row live in threadgroup memory.
        const size_t tg_needed = align_up(static_cast<size_t>(c.n_ctx) * sizeof(float), 16) + kAttentionStaticBytes;
        if (tg_needed > m.device.maxThreadgroupMemoryLength) {
            return unsupported("n_ctx " + std::to_string(c.n_ctx) + " needs " + std::to_string(tg_needed) + " bytes of threadgroup memory; this GPU has " +
                               std::to_string(m.device.maxThreadgroupMemoryLength));
        }
        static_assert(kMaxStagedInput * sizeof(float) + 64 <= 32768, "a staged input row must fit the smallest Apple GPU threadgroup memory");
        m.queue = [m.device newCommandQueue];
        if (!m.queue) return internal_error("could not create a Metal command queue");
        NSError * error = nil;
        id<MTLLibrary> lib = [m.device newLibraryWithSource:[NSString stringWithUTF8String:llama_kernel_source()] options:[MTLCompileOptions new] error:&error];
        if (!lib) return internal_error("could not compile the llama decoder kernels: " + describe(error));
        auto pipeline_with = [&](const char * name, bool stage, id<MTLComputePipelineState> __strong & out) -> Status {
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
        for (auto & [name, slot] : std::initializer_list<std::pair<const char *, id<MTLComputePipelineState> __strong *>>{
                 {"rope_store", &m.p_rope}, {"attention", &m.p_attention}, {"embed", &m.p_embed}, {"argmax", &m.p_argmax},
                 {"matvec1", &m.p_matvec1}, {"gate_up1", &m.p_gate_up1}}) {
            if (Status s = pipeline_with(name, true, *slot); !s.is_ok()) return s;
        }
        for (auto & [name, slot] : std::initializer_list<std::pair<const char *, id<MTLComputePipelineState> __strong *>>{
                 {"matmul_16x16", &m.p_matmul}, {"matmul_gate_up_16x16", &m.p_matmul_gate_up}, {"matmul_big", &m.p_matmul_big}, {"matmul_gate_up_big", &m.p_gate_up_big},
                 {"rms_scale", &m.p_rms_scale}}) {
            if (Status s = pipeline_with(name, true, *slot); !s.is_ok()) return s;
            if ((*slot).maxTotalThreadsPerThreadgroup < kGroupThreads) return unsupported(std::string("kernel ") + name + " cannot run " + std::to_string(kGroupThreads) + " threads per group on this GPU");
        }
        if (c.head_dim == 64 || c.head_dim == 128) {
            if (Status s = pipeline_with(c.head_dim == 64 ? "attention_prefill_64" : "attention_prefill_128", true, m.p_attn_prefill); !s.is_ok()) return s;
            if (m.p_attn_prefill.maxTotalThreadsPerThreadgroup < kGroupThreads) m.p_attn_prefill = nil;   // fall back to the per-query kernel
        }
        m.big_capable = c.n_embd % kBigK == 0 && c.n_ff % kBigK == 0 && (c.n_head * c.head_dim) % kBigK == 0;
        if (Status s = pipeline_with("matvec1", false, m.p_matvec1_direct); !s.is_ok()) return s;
        if (Status s = pipeline_with("gate_up1", false, m.p_gate_up1_direct); !s.is_ok()) return s;
        if (m.p_attention.maxTotalThreadsPerThreadgroup < kAttnThreads) return unsupported("the attention kernel cannot run " + std::to_string(kAttnThreads) + " threads per group on this GPU");
        for (id<MTLComputePipelineState> pso : {m.p_matvec1, m.p_gate_up1, m.p_matvec1_direct, m.p_gate_up1_direct}) {
            if (pso.maxTotalThreadsPerThreadgroup < kGroupThreads) return unsupported("a decoder kernel cannot run " + std::to_string(kGroupThreads) + " threads per group on this GPU");
        }
        if (m.p_rope.maxTotalThreadsPerThreadgroup < static_cast<NSUInteger>(c.head_dim / 2)) return unsupported("head_dim / 2 threads per group are not available for rope");
        m.argmax_threads = static_cast<uint32_t>(std::min<NSUInteger>(kArgmaxThreads, m.p_argmax.maxTotalThreadsPerThreadgroup)) / 32 * 32;

        // Pack the weights: f16 matrices as they are (q, k, v concatenated row-wise when separate), norms as f32.
        size_t total = 0;
        auto reserve = [&](size_t bytes) { const size_t off = total; total = align_up(total + bytes, kWeightAlign); return off; };
        const size_t norm_bytes = static_cast<size_t>(E) * sizeof(float);
        auto mat_bytes = [](const WeightRef & r) { return static_cast<size_t>(r.rows * r.cols) * 2; };
        m.token_embd = reserve(mat_bytes(w.token_embd));
        const bool tied = w.output.data == w.token_embd.data;
        m.output = tied ? m.token_embd : reserve(mat_bytes(w.output));
        m.output_norm = reserve(norm_bytes);
        for (const LlamaLayerWeights & L : w.layers) {
            LayerOffsets o;
            o.attn_norm = reserve(norm_bytes);
            o.qkv = reserve(static_cast<size_t>(m.qkv_width) * static_cast<size_t>(E) * 2);
            o.has_bias = L.attn_qkv_bias.data || L.attn_q_bias.data;
            o.qkv_bias = o.has_bias ? reserve(static_cast<size_t>(m.qkv_width) * sizeof(float)) : 0;
            o.attn_output = reserve(mat_bytes(L.attn_output));
            o.ffn_norm = reserve(norm_bytes);
            o.ffn_gate = reserve(mat_bytes(L.ffn_gate));
            o.ffn_up = reserve(mat_bytes(L.ffn_up));
            o.ffn_down = reserve(mat_bytes(L.ffn_down));
            m.layers.push_back(o);
        }
        m.weights = [m.device newBufferWithLength:total options:MTLResourceStorageModeShared];
        if (!m.weights) return resource_exhausted("could not allocate " + std::to_string(total) + " bytes for the decoder weights");
        uint8_t * base = static_cast<uint8_t *>([m.weights contents]);
        auto put_mat = [&](size_t off, const WeightRef & r) { std::memcpy(base + off, r.data, mat_bytes(r)); };
        // A vector (norm or bias) as f32 at base + off: r.cols values.
        auto put_vec = [&](size_t off, const WeightRef & r) {
            float * dst = reinterpret_cast<float *>(base + off);
            for (int64_t i = 0; i < r.cols; i++) {
                if (r.dtype == DType::F32) std::memcpy(&dst[i], static_cast<const uint8_t *>(r.data) + i * 4, 4);
                else { __fp16 hv; std::memcpy(&hv, static_cast<const uint8_t *>(r.data) + i * 2, 2); dst[i] = static_cast<float>(hv); }
            }
        };
        auto put_norm = put_vec;
        put_mat(m.token_embd, w.token_embd);
        if (!tied) put_mat(m.output, w.output);
        put_norm(m.output_norm, w.output_norm);
        for (size_t l = 0; l < w.layers.size(); l++) {
            const LlamaLayerWeights & L = w.layers[l];
            const LayerOffsets & o = m.layers[l];
            put_norm(o.attn_norm, L.attn_norm);
            if (L.attn_qkv.data) put_mat(o.qkv, L.attn_qkv);
            else {
                put_mat(o.qkv, L.attn_q);
                put_mat(o.qkv + mat_bytes(L.attn_q), L.attn_k);
                put_mat(o.qkv + mat_bytes(L.attn_q) + mat_bytes(L.attn_k), L.attn_v);
            }
            if (L.attn_qkv_bias.data) put_vec(o.qkv_bias, L.attn_qkv_bias);
            else if (L.attn_q_bias.data) {
                put_vec(o.qkv_bias, L.attn_q_bias);
                put_vec(o.qkv_bias + static_cast<size_t>(m.q_width) * sizeof(float), L.attn_k_bias);
                put_vec(o.qkv_bias + static_cast<size_t>(m.q_width + m.kv_width) * sizeof(float), L.attn_v_bias);
            }
            put_mat(o.attn_output, L.attn_output);
            put_norm(o.ffn_norm, L.ffn_norm);
            put_mat(o.ffn_gate, L.ffn_gate);
            put_mat(o.ffn_up, L.ffn_up);
            put_mat(o.ffn_down, L.ffn_down);
        }

        // Activations for the largest chunk, the KV cache, the logits and the token ids.
        const size_t B = static_cast<size_t>(c.n_batch);
        auto buffer = [&](size_t bytes, id<MTLBuffer> __strong & out, const char * what) -> Status {
            out = [m.device newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
            if (!out) return resource_exhausted(std::string("could not allocate the decoder's ") + what + " (" + std::to_string(bytes) + " bytes)");
            return Status();
        };
        m.cache_layer_bytes = static_cast<size_t>(c.n_ctx) * m.kv_width * 2;
        for (auto & [bytes, slot, what] : std::initializer_list<std::tuple<size_t, id<MTLBuffer> __strong *, const char *>>{
                 {static_cast<size_t>(c.n_layer) * m.cache_layer_bytes, &m.kc, "key cache"},
                 {static_cast<size_t>(c.n_layer) * m.cache_layer_bytes, &m.vc, "value cache"},
                 {B * static_cast<size_t>(E) * 4, &m.x, "residual stream"},
                 {B * m.qkv_width * 4, &m.qkv, "q/k/v activations"},
                 {B * m.q_width * 4, &m.attn, "attention output"},
                 {B * static_cast<size_t>(F) * 4, &m.h, "feed-forward activations"},
                 {static_cast<size_t>(V) * 4, &m.logits_buf, "logits"},
                 {(static_cast<size_t>(c.n_ctx) + 1) * 4, &m.tokens, "token ids"},
                 {B * 4, &m.in_tokens, "input token ids"},
                 {B * 4, &m.scale, "RMS scales"}}) {
            if (Status s = buffer(bytes, *slot, what); !s.is_ok()) return s;
        }
    }
    return std::unique_ptr<MetalLlamaDecoder>(new MetalLlamaDecoder(std::move(impl)));
}

void MetalLlamaDecoder::reset() {
    (void)impl_->wait_idle();
    impl_->pos = 0;
    impl_->has_pending = false;
}

Status MetalLlamaDecoder::eval_embeddings(const float * embeddings, int32_t n) {
    Impl & m = *impl_;
    if (!embeddings || n < 1) return invalid_argument("no embeddings to evaluate");
    const int64_t total = static_cast<int64_t>(n) * m.cfg.n_embd;
    for (int64_t i = 0; i < total; i++) if (!std::isfinite(embeddings[i])) return invalid_argument("the embeddings contain NaN or Inf");
    if (static_cast<int64_t>(m.pos) + n > m.cfg.n_ctx) return resource_exhausted("the context of " + std::to_string(m.cfg.n_ctx) + " positions would overflow");
    for (int32_t start = 0; start < n; start += m.cfg.n_batch) {
        if (Status st = m.wait_idle(); !st.is_ok()) return st;  // x is rewritten by the host
        const int32_t rows = std::min(m.cfg.n_batch, n - start);
        std::memcpy([m.x contents], embeddings + static_cast<int64_t>(start) * m.cfg.n_embd, static_cast<size_t>(rows) * static_cast<size_t>(m.cfg.n_embd) * sizeof(float));
        m.run_chunk(static_cast<uint32_t>(rows), false);
    }
    return Status();
}

Status MetalLlamaDecoder::eval_tokens(const std::vector<int32_t> & ids) {
    Impl & m = *impl_;
    if (ids.empty()) return invalid_argument("no tokens to evaluate");
    for (int32_t t : ids) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("token " + std::to_string(t) + " is outside the vocabulary [0, " + std::to_string(m.cfg.n_vocab) + ")");
    if (static_cast<int64_t>(m.pos) + static_cast<int64_t>(ids.size()) > m.cfg.n_ctx) return resource_exhausted("the context of " + std::to_string(m.cfg.n_ctx) + " positions would overflow");
    for (size_t start = 0; start < ids.size(); start += static_cast<size_t>(m.cfg.n_batch)) {
        if (Status st = m.wait_idle(); !st.is_ok()) return st;  // in_tokens is rewritten by the host
        const size_t rows = std::min<size_t>(static_cast<size_t>(m.cfg.n_batch), ids.size() - start);
        std::memcpy([m.in_tokens contents], ids.data() + start, rows * sizeof(int32_t));
        m.run_chunk(static_cast<uint32_t>(rows), true);
    }
    return Status();
}

Result<int32_t> MetalLlamaDecoder::pending_token() {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    return static_cast<const int32_t *>([m.tokens contents])[0];
}

Result<std::vector<float>> MetalLlamaDecoder::logits() {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    const float * l = static_cast<const float *>([m.logits_buf contents]);
    return std::vector<float>(l, l + m.cfg.n_vocab);
}

Status MetalLlamaDecoder::generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token) {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (max_tokens < 1 || depth < 1) return invalid_argument("max_tokens and depth must be >= 1");
    // Step i (1 <= i < max_tokens) writes the cache at position base + i - 1, and its token into tokens[i].
    const int32_t base = m.pos;
    if (static_cast<int64_t>(base) + max_tokens - 1 > m.cfg.n_ctx) return resource_exhausted("the context of " + std::to_string(m.cfg.n_ctx) + " positions would overflow");
    // Logits and tokens[0] of the last evaluated position come from the last committed buffer; the queue runs buffers in order.
    std::deque<id<MTLCommandBuffer>> inflight;  // inflight[k] produces tokens[produced + k]
    inflight.push_back(m.last);
    int32_t produced = 0, next = 1;
    Status st;
    while (true) {
        while (next < max_tokens && next - produced <= depth) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [m.queue commandBufferWithUnretainedReferences];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                m.encode_embed(enc, m.tokens, static_cast<uint32_t>(next - 1), 1);
                m.encode_forward(enc, 1, static_cast<uint32_t>(base + next - 1), static_cast<uint32_t>(next));
                [enc endEncoding];
                [cb commit];
                m.last = cb;
                inflight.push_back(cb);
            }
            next++;
        }
        id<MTLCommandBuffer> cb = inflight.front();
        inflight.pop_front();
        if (cb) {
            if (m.cfg.spin_wait) {
                while (cb.status != MTLCommandBufferStatusCompleted && cb.status != MTLCommandBufferStatusError) __builtin_arm_yield();
            } else {
                [cb waitUntilCompleted];
            }
            if (cb.status == MTLCommandBufferStatusError) { st = internal_error("a Metal command buffer failed: " + describe(cb.error)); break; }
        }
        const int32_t token = static_cast<const int32_t *>([m.tokens contents])[produced];
        produced++;
        if (!on_token(token) || produced == max_tokens) break;
    }
    // Steps queued past the stopping point still run, but nothing waits for them here: the answer is complete, and every later call (and the
    // destructor) starts with wait_idle(), so they never overlap with new work or host writes. On an error the queue is drained now.
    if (!st.is_ok()) (void)m.wait_idle();
    m.pos = base + produced - 1;
    m.has_pending = false;
    return st;
}

Result<std::vector<MetalLlamaDecoder::KernelTime>> MetalLlamaDecoder::profile_pass(int32_t rows) {
    Impl & m = *impl_;
    if (rows < 1 || rows > m.cfg.n_batch) return invalid_argument("rows must be in [1, n_batch]");
    if (rows == 1 && !m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (m.pos + rows > m.cfg.n_ctx) return resource_exhausted("the cache is too full");
    if (Status st = m.wait_idle(); !st.is_ok()) return st;
    if (rows > 1) {
        int32_t * ids = static_cast<int32_t *>([m.in_tokens contents]);
        for (int32_t i = 0; i < rows; i++) ids[i] = i % m.cfg.n_vocab;
    }
    std::vector<KernelTime> times;
    @autoreleasepool {
        m.profile = &times;
        m.prof_cb = [m.queue commandBuffer];
        m.prof_enc = [m.prof_cb computeCommandEncoder];
        m.encode_embed(nil, rows == 1 ? m.tokens : m.in_tokens, 0, static_cast<uint32_t>(rows));
        m.encode_forward(nil, static_cast<uint32_t>(rows), static_cast<uint32_t>(m.pos), 1);
        [m.prof_enc endEncoding];
        m.prof_enc = nil;
        m.prof_cb = nil;
        m.profile = nullptr;
    }
    return times;
}

}  // namespace dooms
