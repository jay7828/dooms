#include "pipeline/native_llama.h"

#include <fstream>

#include "format/gguf.h"

namespace dooms {

Result<std::unique_ptr<MetalLlamaDecoder>> load_native_llama(const std::string & path, const NativeLlamaOptions & o) {
    if (o.n_ctx < 1 || o.n_batch < 1) return invalid_argument("n_ctx and n_batch must be >= 1");
    if (!std::ifstream(path, std::ios::binary)) return not_found("cannot open '" + path + "'");
    auto file = GgufFile::open(path);
    if (!file.is_ok()) {
        // The reader knows f32/f16/bf16/i32 tensors only; a quantized file is a valid model the Metal decoder does not run.
        return unsupported("the Metal decoder cannot read '" + path + "': " + file.status().message());
    }
    const GgufFile & f = file.value();
    auto arch = f.get_string("general.architecture");
    if (!arch.is_ok()) return invalid_argument("'" + path + "' has no general.architecture");
    // llama: adjacent rotary pairs; qwen2: the same block with q/k/v biases and split-half rotary pairs (llama.cpp's LLAMA_ROPE_TYPE_NEOX for qwen2).
    if (arch.value() != "llama" && arch.value() != "qwen2") return unsupported("the Metal decoder runs llama and qwen2 models; '" + path + "' is " + arch.value());
    const std::string prefix = arch.value() + ".";
    auto u = [&](const char * key) -> Result<int64_t> {
        auto v = f.get_uint(prefix + key);
        if (!v.is_ok()) return invalid_argument("'" + path + "' lacks " + prefix + key);
        return static_cast<int64_t>(v.value());
    };
    if (auto scaling = f.get_string(prefix + "rope.scaling.type"); scaling.is_ok() && scaling.value() != "none") {
        return unsupported("rotary scaling '" + scaling.value() + "' is not implemented by the Metal decoder");
    }
    LlamaDecoderConfig c;
    for (auto [key, dst] : {std::pair<const char *, int32_t *>{"block_count", &c.n_layer}, {"embedding_length", &c.n_embd}, {"attention.head_count", &c.n_head},
                            {"attention.head_count_kv", &c.n_head_kv}, {"feed_forward_length", &c.n_ff}}) {
        auto v = u(key);
        if (!v.is_ok()) return v.status();
        *dst = static_cast<int32_t>(v.value());
    }
    if (c.n_head < 1 || c.n_embd % c.n_head != 0) return invalid_argument("embedding_length is not a multiple of head_count");
    c.head_dim = c.n_embd / c.n_head;
    for (const char * key : {"attention.key_length", "attention.value_length", "rope.dimension_count"}) {
        if (auto v = u(key); v.is_ok() && v.value() != c.head_dim) {
            return unsupported(prefix + key + " = " + std::to_string(v.value()) + " differs from the head size " + std::to_string(c.head_dim) + "; the Metal decoder rotates and attends over whole heads");
        }
    }
    auto base = f.get_float(prefix + "rope.freq_base");
    auto eps = f.get_float(prefix + "attention.layer_norm_rms_epsilon");
    if (!base.is_ok() || !eps.is_ok()) return invalid_argument("'" + path + "' lacks " + prefix + "rope.freq_base or " + prefix + "attention.layer_norm_rms_epsilon");
    c.rope_neox = arch.value() == "qwen2";
    c.rope_base = static_cast<float>(base.value());
    c.rms_eps = static_cast<float>(eps.value());
    auto ctx_len = u("context_length");
    if (!ctx_len.is_ok()) return ctx_len.status();
    if (o.n_ctx > ctx_len.value()) return invalid_argument("n_ctx " + std::to_string(o.n_ctx) + " exceeds the model's context length " + std::to_string(ctx_len.value()));
    c.n_ctx = o.n_ctx;
    c.n_batch = std::min(o.n_batch, o.n_ctx);
    c.spin_wait = o.spin_wait;
    c.prefill_tile = o.prefill_tile;

    auto ref = [&](const std::string & name, bool required, WeightRef * out) -> Status {
        auto t = f.find_tensor(name);
        if (!t.is_ok()) return required ? invalid_argument("'" + path + "' has no tensor " + name) : Status();
        const GgufTensorInfo & info = *t.value();
        out->data = f.tensor_data(info);
        out->dtype = info.dtype;
        if (info.shape.rank() == 1) { out->rows = 1; out->cols = info.shape.dim(0); }
        else if (info.shape.rank() == 2) { out->rows = info.shape.dim(0); out->cols = info.shape.dim(1); }
        else return invalid_argument("tensor " + name + " has rank " + std::to_string(info.shape.rank()));
        return Status();
    };
    LlamaDecoderWeights w;
    if (Status st = ref("token_embd.weight", true, &w.token_embd); !st.is_ok()) return st;
    if (Status st = ref("output_norm.weight", true, &w.output_norm); !st.is_ok()) return st;
    if (Status st = ref("output.weight", false, &w.output); !st.is_ok()) return st;
    if (!w.output.data) w.output = w.token_embd;  // tied embeddings
    c.n_vocab = static_cast<int32_t>(w.token_embd.rows);
    w.layers.resize(static_cast<size_t>(c.n_layer));
    for (int32_t l = 0; l < c.n_layer; l++) {
        LlamaLayerWeights & L = w.layers[static_cast<size_t>(l)];
        const std::string b = "blk." + std::to_string(l) + ".";
        Status st = ref(b + "attn_norm.weight", true, &L.attn_norm);
        if (st.is_ok()) st = ref(b + "attn_qkv.weight", false, &L.attn_qkv);
        if (st.is_ok() && !L.attn_qkv.data) {
            st = ref(b + "attn_q.weight", true, &L.attn_q);
            if (st.is_ok()) st = ref(b + "attn_k.weight", true, &L.attn_k);
            if (st.is_ok()) st = ref(b + "attn_v.weight", true, &L.attn_v);
        }
        // Biases: qwen2 has them on q, k and v (required there); llama files normally have none.
        const bool need_bias = arch.value() == "qwen2";
        if (st.is_ok()) st = ref(b + "attn_qkv.bias", false, &L.attn_qkv_bias);
        if (st.is_ok() && !L.attn_qkv_bias.data) {
            st = ref(b + "attn_q.bias", need_bias, &L.attn_q_bias);
            if (st.is_ok()) st = ref(b + "attn_k.bias", need_bias || L.attn_q_bias.data, &L.attn_k_bias);
            if (st.is_ok()) st = ref(b + "attn_v.bias", need_bias || L.attn_q_bias.data, &L.attn_v_bias);
        }
        for (auto [name, dst] : {std::pair<const char *, WeightRef *>{"attn_output", &L.attn_output}, {"ffn_norm", &L.ffn_norm}, {"ffn_gate", &L.ffn_gate},
                                 {"ffn_up", &L.ffn_up}, {"ffn_down", &L.ffn_down}}) {
            if (st.is_ok()) st = ref(b + name + ".weight", true, dst);
        }
        if (!st.is_ok()) return st;
    }
    return MetalLlamaDecoder::create(c, w);
}

}  // namespace dooms
