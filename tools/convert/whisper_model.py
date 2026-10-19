"""Whisper (OpenAI's speech-to-text transformer, Hugging Face layout) for the dooms.cpp engine: the encoder as a Core ML package for the Neural Engine, and
the decoder + vocabulary + feature-extraction constants as a GGUF the engine's Metal decoder reads (architecture "whisper"; layout in
docs/formats/whisper-gguf.md). Hyperparameters come from the live model and its processor, never typed in; anything the engine does not implement is refused.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

import convert_common as cc

ARCH = "whisper"
ENC_INPUT, ENC_OUTPUT = "input_features", "encoder_out"


def load(model_dir: str):
    """(model fp32 eval, processor, generation config) from a Hugging Face snapshot."""
    from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    processor = WhisperProcessor.from_pretrained(model_dir)
    gen = GenerationConfig.from_pretrained(model_dir)
    return model, processor, gen


def check_supported(model, processor) -> None:
    c = model.config
    cc._require(c.activation_function == "gelu", f"activation {c.activation_function!r}: the engine implements exact (erf) GELU only")
    cc._require(not getattr(c, "scale_embedding", False), "scale_embedding=True is not implemented")
    cc._require(c.d_model % c.decoder_attention_heads == 0, "d_model is not a multiple of the decoder heads")
    hd = c.d_model // c.decoder_attention_heads
    cc._require(hd in (64, 128), f"decoder head_dim {hd}: the engine's attention kernels take 64 or 128")
    cc._require(c.d_model % 32 == 0 and c.decoder_ffn_dim % 32 == 0, "d_model and decoder_ffn_dim must be multiples of 32")
    fe = processor.feature_extractor
    cc._require(fe.sampling_rate == 16000 and fe.n_fft == 400 and fe.hop_length == 160, "feature extractor is not Whisper's 16 kHz / n_fft 400 / hop 160")
    cc._require(fe.feature_size == c.num_mel_bins, "feature extractor and model disagree on the number of mel bins")
    cc._require(getattr(fe, "dither", 0.0) == 0.0, "dithering is not implemented")
    cc._require(c.max_source_positions * 2 == fe.nb_max_frames, "encoder context does not match 30 s of 10 ms frames")
    tied = model.proj_out.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr() or torch.equal(model.proj_out.weight, model.model.decoder.embed_tokens.weight)
    cc._require(tied, "proj_out is not tied to the token embedding")
    dec = model.model.decoder
    for i, L in enumerate(dec.layers):
        cc._require(L.self_attn.k_proj.bias is None and L.encoder_attn.k_proj.bias is None, f"decoder layer {i}: k_proj has a bias (Whisper's has none)")


class Encoder(torch.nn.Module):
    """input_features [1, n_mels, 3000] -> encoder output [1, 1500, d_model] (the model's own encoder, final layer norm included)."""

    def __init__(self, model):
        super().__init__()
        self.enc = model.model.encoder

    def forward(self, x):
        return self.enc(x).last_hidden_state


def convert_encoder(model, out_path: str, compute_units: str = "gpu", max_rel_l2: float = 2e-2, seed: int = 0, precision: str = "fp16") -> float:
    """Writes the Core ML package in `precision` (fp16, or fp32 for the CPU: the CPU computes an fp16 package in half precision, 0.18 relative L2 for
    large-v3-turbo's 32 layers) only if its output on `compute_units` is within max_rel_l2 of PyTorch fp32 on random features; returns the error."""
    import coremltools as ct  # noqa: PLC0415
    c = model.config
    torch.manual_seed(seed)
    x = torch.randn(1, c.num_mel_bins, 2 * c.max_source_positions) * 0.5
    enc = Encoder(model).eval()
    with torch.no_grad():
        ref = enc(x)
        traced = torch.jit.trace(enc, x)
    mlm = ct.convert(traced, inputs=[ct.TensorType(name=ENC_INPUT, shape=tuple(x.shape))], outputs=[ct.TensorType(name=ENC_OUTPUT)], convert_to="mlprogram",
                     minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32)
    tmp = out_path + ".tmp.mlpackage"
    if os.path.exists(tmp):
        import shutil  # noqa: PLC0415
        shutil.rmtree(tmp)
    mlm.save(tmp)
    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ne": ct.ComputeUnit.CPU_AND_NE}[compute_units]
    out = ct.models.MLModel(tmp, compute_units=units).predict({ENC_INPUT: x.numpy()})[ENC_OUTPUT]
    err = float(np.linalg.norm(out - ref.numpy()) / np.linalg.norm(ref.numpy()))
    cc._require(np.isfinite(err) and err <= max_rel_l2, f"Core ML encoder ({compute_units}) differs from PyTorch by {err:.2e} > {max_rel_l2}")
    import shutil  # noqa: PLC0415
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.rename(tmp, out_path)
    return err


def special_tokens(processor, gen) -> Dict[str, int]:
    tok = processor.tokenizer
    ids = {}
    # Whisper's unk token IS <|endoftext|>, so "not unk" is no test: a token exists when its id maps back to its own name.
    def find(name):
        i = tok.convert_tokens_to_ids(name)
        return int(i) if i is not None and tok.convert_ids_to_tokens(int(i)) == name else None

    for key, name in [("sot", "<|startoftranscript|>"), ("eot", "<|endoftext|>"), ("transcribe", "<|transcribe|>"), ("translate", "<|translate|>"),
                      ("notimestamps", "<|notimestamps|>"), ("nospeech", "<|nospeech|>")]:
        i = find(name)
        if key == "nospeech" and i is None:
            i = find("<|nocaptions|>")   # older vocabularies
        cc._require(i is not None, f"the vocabulary has no {name}")
        ids[key] = i
    ids["timestamp_begin"] = find("<|0.00|>")
    cc._require(ids["timestamp_begin"] is not None, "the vocabulary has no <|0.00|>")
    return ids


def languages(gen) -> Tuple[List[str], List[int]]:
    """Language codes ("en", ...) and their token ids, from the generation config's lang_to_id."""
    lang_to_id = getattr(gen, "lang_to_id", None) or {}
    cc._require(lang_to_id, "the generation config has no lang_to_id table")
    codes, ids = [], []
    for token, i in sorted(lang_to_id.items(), key=lambda kv: kv[1]):
        cc._require(token.startswith("<|") and token.endswith("|>"), f"unexpected language token {token!r}")
        codes.append(token[2:-2])
        ids.append(int(i))
    return codes, ids


def write_decoder(model, processor, gen, out_path: str) -> None:
    """The decoder (f16 matrices, f32 vectors), vocabulary, special tokens, suppression lists and the feature-extraction constants."""
    import gguf  # noqa: PLC0415
    check_supported(model, processor)
    c, fe, tok = model.config, processor.feature_extractor, processor.tokenizer
    dec = model.model.decoder
    w = gguf.GGUFWriter(out_path + ".tmp", ARCH)
    k = lambda name: f"{ARCH}.{name}"  # noqa: E731
    w.add_uint32(k("n_mels"), c.num_mel_bins)
    w.add_uint32(k("n_audio_ctx"), c.max_source_positions)
    w.add_uint32(k("n_audio_state"), c.d_model)
    w.add_uint32(k("n_text_ctx"), c.max_target_positions)
    w.add_uint32(k("n_text_state"), c.d_model)
    w.add_uint32(k("n_text_head"), c.decoder_attention_heads)
    w.add_uint32(k("n_text_layer"), c.decoder_layers)
    w.add_uint32(k("n_text_ff"), c.decoder_ffn_dim)
    w.add_uint32(k("n_vocab"), c.vocab_size)
    w.add_float32(k("layer_norm_eps"), float(dec.layer_norm.eps))
    w.add_uint32(k("sample_rate"), fe.sampling_rate)
    w.add_uint32(k("n_fft"), fe.n_fft)
    w.add_uint32(k("hop_length"), fe.hop_length)
    w.add_uint32(k("chunk_samples"), fe.n_samples)
    for name, i in special_tokens(processor, gen).items():
        w.add_uint32(k(f"token.{name}"), i)
    codes, lang_ids = languages(gen)
    w.add_array(k("language.codes"), codes)
    w.add_array(k("language.ids"), lang_ids)
    w.add_array(k("suppress_tokens"), [int(t) for t in (gen.suppress_tokens or [])])
    w.add_array(k("begin_suppress_tokens"), [int(t) for t in (gen.begin_suppress_tokens or [])])
    # id -> token string (byte-level BPE pieces, special tokens as their names)
    n_tok = len(tok)
    cc._require(n_tok <= c.vocab_size, f"the tokenizer has {n_tok} tokens, more than the model's vocab_size {c.vocab_size}")
    pieces = tok.convert_ids_to_tokens(list(range(n_tok)))
    pieces += [f"<|unused_{i}|>" for i in range(n_tok, c.vocab_size)]
    w.add_array(k("tokens"), pieces)
    w.add_uint32(k("n_text_tokens"), int(tok.convert_tokens_to_ids("<|endoftext|>")))   # ids below this are text pieces; the rest are special
    mel = np.asarray(fe.mel_filters, dtype=np.float32)          # [n_freq, n_mels]
    cc._require(mel.shape == (fe.n_fft // 2 + 1, c.num_mel_bins), f"mel filter bank has shape {mel.shape}")
    w.add_tensor("mel_filters", np.ascontiguousarray(mel.T))   # [n_mels][n_freq]

    def mat(name, t):
        w.add_tensor(name, t.detach().float().numpy().astype(np.float16))

    def vec(name, t):
        w.add_tensor(name, t.detach().float().numpy().astype(np.float32))

    mat("token_embd.weight", dec.embed_tokens.weight)
    vec("pos_embd.weight", dec.embed_positions.weight)
    for i, L in enumerate(dec.layers):
        b = f"blk.{i}."
        for part, ln in (("attn_ln", L.self_attn_layer_norm), ("cross_ln", L.encoder_attn_layer_norm), ("mlp_ln", L.final_layer_norm)):
            vec(b + part + ".weight", ln.weight)
            vec(b + part + ".bias", ln.bias)
        for part, att in (("attn", L.self_attn), ("cross", L.encoder_attn)):
            mat(b + part + "_q.weight", att.q_proj.weight)
            vec(b + part + "_q.bias", att.q_proj.bias)
            mat(b + part + "_k.weight", att.k_proj.weight)
            mat(b + part + "_v.weight", att.v_proj.weight)
            vec(b + part + "_v.bias", att.v_proj.bias)
            mat(b + part + "_out.weight", att.out_proj.weight)
            vec(b + part + "_out.bias", att.out_proj.bias)
        mat(b + "fc1.weight", L.fc1.weight)
        vec(b + "fc1.bias", L.fc1.bias)
        mat(b + "fc2.weight", L.fc2.weight)
        vec(b + "fc2.bias", L.fc2.bias)
    vec("ln.weight", dec.layer_norm.weight)
    vec("ln.bias", dec.layer_norm.bias)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    verify_written(out_path + ".tmp", model)
    os.replace(out_path + ".tmp", out_path)


def verify_written(path: str, model) -> None:
    """Re-reads the file with the reference gguf reader: every decoder tensor present, with the source's shape and (rounded) values."""
    import gguf  # noqa: PLC0415
    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}
    dec = model.model.decoder
    want = {"token_embd.weight": dec.embed_tokens.weight, "pos_embd.weight": dec.embed_positions.weight, "ln.weight": dec.layer_norm.weight}
    for i, L in enumerate(dec.layers):
        want[f"blk.{i}.fc2.weight"] = L.fc2.weight
        want[f"blk.{i}.cross_k.weight"] = L.encoder_attn.k_proj.weight
        want[f"blk.{i}.attn_q.bias"] = L.self_attn.q_proj.bias
    for name, src in want.items():
        cc._require(name in t, f"{name} missing from the written file")
        got = np.asarray(t[name].data, dtype=np.float32).reshape(tuple(src.shape))
        ref = src.detach().float().numpy()
        cc._require(np.allclose(got, ref, rtol=1e-3, atol=1e-3), f"{name} differs from the source after writing")
    n_layer_tensors = sum(1 for n in t if n.startswith("blk."))
    per_layer = 24   # 3 layer norms x 2, 2 attentions x 7 (q.w q.b k.w v.w v.b out.w out.b), fc1.w fc1.b fc2.w fc2.b
    cc._require(n_layer_tensors == per_layer * len(dec.layers), f"{n_layer_tensors} layer tensors written, expected {per_layer * len(dec.layers)}")
