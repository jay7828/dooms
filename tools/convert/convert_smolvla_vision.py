"""SmolVLA's camera encoders -> Core ML packages for the Neural Engine.

    .venv/bin/python tools/convert/convert_smolvla_vision.py --vlm HuggingFaceTB/SmolVLM2-500M-Video-Instruct --policy lerobot/smolvla_base --out-dir models/smolvla [--batches 1,3]

SmolVLA embeds each camera image with the SigLIP tower and the pixel-shuffle connector of SmolVLM2-500M (the same function as tools/convert/smolvlm_vision.py:
[B,3,512,512] -> [B,64,960]; the policy then multiplies by sqrt(960), which stays on the host). The policy checkpoint carries its own copy of those weights; they
are loaded over the base model's and the script reports whether they differ. Writes vision-b<B>-fp16.mlpackage for each requested batch size. Exit codes: 0 ok, 1 error.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import smolvlm_vision as sv  # noqa: E402
from convert_common import ConversionError, _require  # noqa: E402

PREFIX = "model.vlm_with_expert.vlm.model."


def overlay_policy_weights(model, policy: str) -> dict:
    """Loads the policy's vision tower and connector weights into `model`; returns {'tensors': n, 'max_abs_diff': largest change versus the base weights}."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    path = policy if os.path.isfile(policy) else hf_hub_download(policy, "model.safetensors")
    state = model.model.state_dict()
    n, worst = 0, 0.0
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if not key.startswith(PREFIX) or not (key[len(PREFIX):].startswith("vision_model.") or key[len(PREFIX):].startswith("connector.")):
                continue
            name = key[len(PREFIX):]
            _require(name in state, f"the policy has a tensor '{name}' the base model does not")
            t = f.get_tensor(key).to(torch.float32)
            _require(tuple(t.shape) == tuple(state[name].shape), f"'{name}': shape {tuple(t.shape)} in the policy, {tuple(state[name].shape)} in the model")
            worst = max(worst, float((t - state[name]).abs().max()))
            state[name].copy_(t)
            n += 1
    _require(n > 0, "no vision/connector tensors found in the policy checkpoint")
    return {"tensors": n, "max_abs_diff": worst}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", required=True, help="SmolVLM2 model directory or Hugging Face id")
    ap.add_argument("--policy", required=True, help="SmolVLA checkpoint (model.safetensors path or Hugging Face id)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batches", default="1,3", help="comma-separated batch sizes to export")
    ap.add_argument("--max-rel-l2", type=float, default=2e-2)
    a = ap.parse_args()
    try:
        batches = [int(b) for b in a.batches.split(",")]
        _require(batches and all(1 <= b <= 16 for b in batches), "--batches must be integers in [1, 16]")
        model = sv.load_model(a.vlm)
        info = overlay_policy_weights(model, a.policy)
        print(f"loaded {info['tensors']} vision/connector tensors from the policy; largest change versus the base model {info['max_abs_diff']:.3g}")
        os.makedirs(a.out_dir, exist_ok=True)
        for b in batches:
            out = os.path.join(a.out_dir, f"vision-b{b}-fp16.mlpackage")
            err = sv.convert(model, out, max_rel_l2=a.max_rel_l2, batch=b)
            print(f"wrote {out} (GPU-units error vs fp32 PyTorch {err:.2e})")
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
