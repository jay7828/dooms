"""SmolVLM (Idefics3: SigLIP vision tower + pixel-shuffle connector) image encoder as one traceable module, and its Core ML export.

For one full 512x512 tile (no image splitting) transformers' Idefics3 reduces to: patch convolution + the position rows picked by the model's own bucketizing rule (a fixed table for a full tile), the 12 encoder blocks without a mask, the post LayerNorm, pixel shuffle (scale 4) and the projection to the language
model's width: [1,3,512,512] -> [1,64,576]. `ImageEncoder` is exactly that, verified against `model.model.get_image_features` before anything is
exported. Everything (sizes, scale factor) is read from the live model.
"""
from __future__ import annotations

import os
import shutil
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

import convert_common as wb  # ConversionError / _require shared conventions
from convert_common import ConversionError, _require

INPUT = "pixel_values"
OUTPUT = "image_embeddings"
MAX_SPLIT_MISMATCH = 1e-5


def load_model(model_dir: str):
    from transformers import AutoModelForImageTextToText  # noqa: PLC0415  (Idefics3 and SmolVLM2 both load through it)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class ImageEncoder(nn.Module):
    def __init__(self, model, batch: int = 1):
        super().__init__()
        _require(batch >= 1, "batch must be at least 1")
        self.batch = batch
        vm = model.model.vision_model
        cfg = model.config.vision_config
        self.size, self.patch = int(cfg.image_size), int(cfg.patch_size)
        self.grid = self.size // self.patch
        _require(vm.embeddings.num_patches_per_side == self.grid, "the position table is not for the configured image size")
        self.patch_embedding = vm.embeddings.patch_embedding
        # Idefics3VisionEmbeddings picks position ids by bucketizing fractional coordinates scaled by (1 - 1e-6), which puts every row/column just below its
        # bucket boundary: row r gets id max(r - 1, 0). The model was trained with that code, so it is reproduced exactly (same torch calls, evaluated once).
        side = vm.embeddings.num_patches_per_side
        boundaries = torch.arange(1 / side, 1.0, 1 / side)
        fractional = torch.arange(self.grid, dtype=torch.float32) / self.grid * (1 - 1e-6)
        bucket = torch.bucketize(fractional, boundaries, right=True)
        position_ids = (bucket[:, None] * side + bucket).flatten()
        self.position = vm.embeddings.position_embedding.weight[position_ids].detach().clone()  # [grid*grid, width]
        self.layers, self.post_layernorm, self.connector = vm.encoder.layers, vm.post_layernorm, model.model.connector
        self.heads = int(cfg.num_attention_heads)
        _require(int(cfg.hidden_size) % self.heads == 0, "the hidden size is not a multiple of the head count")
        self.head_dim = int(cfg.hidden_size) // self.heads

    def attention(self, q, k, v):
        """softmax(q k^T / sqrt(d)) v for [1, heads, S, d] tensors, arranged for the Neural Engine (measured in bench/python/vision_ane_experiments.py,
        61 ms vs 78 ms for the textbook form): the scores are kept transposed ([keys, queries]) because the Neural Engine reduces over that axis faster, and
        the softmax denominator comes out of the value matmul for free as an extra column of ones, so the [S, S] probabilities are never normalized. Same
        function in exact arithmetic; the max subtraction stays so that fp16 cannot overflow."""
        scores = k @ (q * self.head_dim ** -0.5).transpose(-1, -2)  # [1, heads, keys, queries]
        weights = torch.exp(scores - scores.amax(dim=-2, keepdim=True))
        out = weights.transpose(-1, -2) @ torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        return out[..., :-1] / out[..., -1:]

    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2) + self.position[None]
        b, n = x.shape[0], x.shape[1]
        for layer in self.layers:
            a = layer.self_attn
            h = layer.layer_norm1(x)
            q, k, v = (p(h).reshape(b, n, self.heads, self.head_dim).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
            x = x + a.out_proj(self.attention(q, k, v).transpose(1, 2).reshape(b, n, -1))
            x = x + layer.mlp(layer.layer_norm2(x))
        return self.connector(self.post_layernorm(x))

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        return self.batch, 3, self.size, self.size


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def verify(model, enc: ImageEncoder, seed: int = 0) -> torch.Tensor:
    """The wrapper must reproduce transformers' own image features; returns the reference output for a random tile."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*enc.input_shape, generator=g) * 2 - 1  # the processor normalizes to [-1, 1]
    with torch.no_grad():
        want = model.model.get_image_features(pixel_values=x[None])
        want = want[0] if want.ndim == 3 and want.shape[0] != 1 else want
        got = enc(x)
    _require(got.shape == want.reshape(got.shape).shape, f"the wrapper returns {tuple(got.shape)} but transformers {tuple(want.shape)}")
    err = rel_l2(got.numpy(), want.reshape(got.shape).numpy())
    _require(err <= MAX_SPLIT_MISMATCH, f"the image encoder wrapper differs from get_image_features: relative L2 {err:.2e}")
    return got


def convert(model, out_path: str, precision: str = "fp16", max_rel_l2: float = None, compute_units: str = "gpu", batch: int = 1) -> float:
    """Writes the Core ML package (only if the checks pass); returns the relative L2 error of the converted model on `compute_units` ('cpu' or 'gpu')
    against the fp32 PyTorch encoder."""
    import coremltools as ct  # noqa: PLC0415
    _require(precision in ("fp16", "fp32"), f"precision must be fp16 or fp32, got '{precision}'")
    _require(out_path.endswith(".mlpackage"), "the output path must end in .mlpackage")
    single = ImageEncoder(model).eval()
    verify(model, single)  # the wrapper is checked against transformers on one image
    enc = single if batch == 1 else ImageEncoder(model, batch).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.rand(*enc.input_shape, generator=g) * 2 - 1
    with torch.no_grad():
        want = enc(x)
        if batch > 1:  # the layers are batch-independent: every row must equal the single-image result
            for i in range(batch):
                _require(rel_l2(want[i].numpy(), single(x[i:i + 1])[0].numpy()) <= MAX_SPLIT_MISMATCH, "a batched row differs from the single-image result")
    with torch.no_grad():
        traced = torch.jit.trace(enc, x, check_trace=False)
    mlm = ct.convert(traced, inputs=[ct.TensorType(name=INPUT, shape=tuple(x.shape))], outputs=[ct.TensorType(name=OUTPUT)], convert_to="mlprogram",
                     minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32)
    mlm.user_defined_metadata["dooms.format"] = "smolvlm-image-encoder/1"
    mlm.user_defined_metadata["dooms.precision"] = precision
    tmp = out_path + ".tmp.mlpackage"
    mlm.save(tmp)
    try:
        units = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU}[compute_units]
        out = ct.models.MLModel(tmp, compute_units=units).predict({INPUT: x.numpy()})
        err = rel_l2(np.asarray(out[OUTPUT]), want.numpy())
        if max_rel_l2 is not None:
            _require(err <= max_rel_l2, f"the converted model differs from PyTorch: relative L2 {err:.2e} > {max_rel_l2:.1e}")
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.rename(tmp, out_path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return err


# ---------------------------------------------------------------------------------------------
# Prompt layout and preprocessing parameters (read from the processor, not typed in)
# ---------------------------------------------------------------------------------------------

SENTINEL = "@@PROMPT@@"
ARCH = "vlm_config"


def prompt_layout(processor) -> Dict[str, object]:
    """How a user prompt becomes model input for one image tile, derived by rendering the chat template with a sentinel prompt:
    prefix (text before the image tokens), the image token and its count, infix (between the image tokens and the prompt), suffix (after the
    prompt). Verified by re-tokenizing the reassembled string and comparing with the processor's own input ids."""
    from PIL import Image  # noqa: PLC0415
    tok = processor.tokenizer
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SENTINEL}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    ids = processor(text=text, images=[Image.new("RGB", (64, 64))], return_tensors="pt")["input_ids"][0].tolist()
    image_token = processor.image_token
    image_id = tok.convert_tokens_to_ids(image_token)
    positions = [i for i, t in enumerate(ids) if t == image_id]
    _require(positions and positions == list(range(positions[0], positions[0] + len(positions))), "the image tokens are not one contiguous run")
    n_image = len(positions)
    before = tok.decode(ids[:positions[0]], skip_special_tokens=False)
    after_image = tok.decode(ids[positions[-1] + 1:], skip_special_tokens=False)
    _require(after_image.count(SENTINEL) == 1, "the sentinel prompt did not survive the chat template exactly once")
    infix, suffix = after_image.split(SENTINEL)
    rebuilt = tok(before + image_token * n_image + infix + SENTINEL + suffix, add_special_tokens=False)["input_ids"]
    _require(rebuilt == ids, "re-tokenizing the reassembled prompt does not reproduce the processor's token ids; the layout cannot be expressed as prefix/infix/suffix")
    return {"prefix": before, "image_token": image_token, "image_token_id": int(image_id), "n_image_tokens": n_image, "infix": infix, "suffix": suffix}


def write_config(path: str, model, processor, layout: Dict[str, object], text_gguf_sha256: str = None) -> None:
    import gguf  # noqa: PLC0415
    ip = processor.image_processor
    mean, std = [float(v) for v in ip.image_mean], [float(v) for v in ip.image_std]
    _require(len(mean) == 3 and len(std) == 3 and all(s > 0 for s in std), "image_mean / image_std must have 3 entries, std positive")
    _require(float(ip.rescale_factor) == 1.0 / 255.0, "the preprocessing assumes rescale_factor = 1/255")
    size = int(model.config.vision_config.image_size)
    w = gguf.GGUFWriter(path, ARCH)
    w.add_name("SmolVLM prompt layout and image preprocessing")
    w.add_string("general.license", "Apache-2.0 (SmolVLM, Hugging Face)")
    w.add_string(f"{ARCH}.family", "idefics3")
    w.add_uint32(f"{ARCH}.image_size", size)
    w.add_uint32(f"{ARCH}.n_image_tokens", int(layout["n_image_tokens"]))
    w.add_uint32(f"{ARCH}.image_token_id", int(layout["image_token_id"]))
    w.add_uint32(f"{ARCH}.embedding_dim", int(model.config.text_config.hidden_size))
    w.add_string(f"{ARCH}.image_token", str(layout["image_token"]))
    w.add_string(f"{ARCH}.prefix", str(layout["prefix"]))
    w.add_string(f"{ARCH}.infix", str(layout["infix"]))
    w.add_string(f"{ARCH}.suffix", str(layout["suffix"]))
    w.add_string(f"{ARCH}.resample", "lanczos")
    for c in range(3):
        w.add_float64(f"{ARCH}.image_mean_{c}", mean[c])
        w.add_float64(f"{ARCH}.image_std_{c}", std[c])
    if text_gguf_sha256:
        w.add_string(f"{ARCH}.text_model_sha256", text_gguf_sha256)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
