"""The host-side part of SmolVLA for the C++ pipeline: language-token embeddings, state projection and the sizes/patterns the pipeline needs.

    .venv/bin/python tools/convert/convert_smolvla_host.py --vlm HF_DIR --policy-dir DIR --out models/smolvla/host.gguf [--reference-npz models/smolvla/reference.npz --reference-out models/smolvla/reference.gguf]

host.gguf (architecture 'vla_config'): tensors embed_tokens [vocab, hidden] bf16 (the checkpoint's own values, unchanged), state_proj.weight [hidden, state_dim] / state_proj.bias f32; metadata hidden, head_dim, kv_heads, layers, chunk, action_dim (the padded
width), state_dim, steps, expert_hidden, tokenizer_max_length, prefix_length, image_size, n_cameras, tokens_per_image, embed_scale, min_period, max_period,
max_wavelength, self_attn_every_n_layers. With --reference-npz also writes the npz of tools/parity/smolvla_reference.py as a GGUF (float32 tensors, integers
as float32) for the C++ parity tests. Exit codes: 0 ok, 1 error.
"""
import argparse
import math
import os
import sys

import gguf
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import smolvla_model as sm  # noqa: E402
from convert_common import ConversionError, _require  # noqa: E402

ARCH = "vla_config"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-npz")
    ap.add_argument("--reference-out")
    ap.add_argument("--task", default="pick up the red cube and place it in the box", help="the prompt the reference tokens were made from (stored with the reference)")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--n-cameras", type=int, default=3)
    ap.add_argument("--tokens-per-image", type=int, default=64)
    a = ap.parse_args()
    try:
        cfg = sm.Config(a.vlm, os.path.join(a.policy_dir, "config.json"))
        t = sm.load_tensors(os.path.join(a.policy_dir, "model.safetensors"))
        emb = t["model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"]
        bf = emb.to(torch.bfloat16)
        _require(bool((bf.to(torch.float32) == emb).all()), "the embedding table is not exactly representable in bf16 (the checkpoint is expected to store bf16)")
        bf_bits = bf.view(torch.int16).numpy().view(np.uint16)
        sw, sb = t["model.state_proj.weight"], t["model.state_proj.bias"]
        import json  # noqa: PLC0415
        pc = json.load(open(os.path.join(a.policy_dir, "config.json")))
        w = gguf.GGUFWriter(a.out, ARCH)
        w.add_name("SmolVLA host tensors")
        w.add_string("general.license", "Apache-2.0 (SmolVLA, Hugging Face / lerobot)")
        for k, v in dict(hidden=cfg.hidden, head_dim=cfg.head_dim, kv_heads=cfg.kv_heads, layers=cfg.layers, chunk=cfg.chunk, action_dim=cfg.action_dim, state_dim=cfg.state_dim,
                         steps=cfg.steps, expert_hidden=cfg.expert_hidden, tokenizer_max_length=int(pc["tokenizer_max_length"]), image_size=a.image_size, n_cameras=a.n_cameras,
                         tokens_per_image=a.tokens_per_image, self_attn_every_n_layers=cfg.self_every).items():
            w.add_uint32(f"{ARCH}.{k}", int(v))
        for k, v in dict(embed_scale=math.sqrt(cfg.hidden), min_period=cfg.min_period, max_period=cfg.max_period, max_wavelength=cfg.max_wavelength).items():
            w.add_float64(f"{ARCH}.{k}", float(v))
        w.add_tensor("embed_tokens", bf_bits, raw_shape=tuple(bf.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
        w.add_tensor("state_proj.weight", sw.numpy())
        w.add_tensor("state_proj.bias", sb.numpy())
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        print(f"wrote {a.out}: embed_tokens {tuple(bf.shape)} bf16, state_proj {tuple(sw.shape)}")
        if a.reference_npz:
            _require(a.reference_out, "--reference-out is required with --reference-npz")
            ref = np.load(a.reference_npz)
            r = gguf.GGUFWriter(a.reference_out, "vla_reference")
            r.add_name("SmolVLA reference activations")
            r.add_string("task", a.task)
            for k in ref.files:
                r.add_tensor(k, np.ascontiguousarray(ref[k].astype(np.float32)))
            r.write_header_to_file(); r.write_kv_data_to_file(); r.write_tensors_to_file(); r.close()
            print(f"wrote {a.reference_out}: {len(ref.files)} tensors")
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
