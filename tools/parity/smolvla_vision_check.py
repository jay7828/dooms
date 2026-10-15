"""SmolVLA camera embeddings: the Core ML towers (tools/convert/convert_smolvla_vision.py) against lerobot's own `embed_image` on the policy's prepared images.

    .venv-lerobot/bin/python tools/parity/smolvla_vision_check.py --package models/smolvla/vision-b1-fp16.mlpackage [--units cpu_ne]
Prints the relative L2 error per camera. Exit codes: 0 ok (all below --max-rel-l2), 1 mismatch or error.
"""
import argparse
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--units", default="cpu_ne", choices=["cpu_ne", "cpu_gpu", "all", "cpu"])
    ap.add_argument("--max-rel-l2", type=float, default=5e-2)
    a = ap.parse_args()
    import coremltools as ct  # noqa: PLC0415
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
    policy = SmolVLAPolicy.from_pretrained(a.model).to("cpu").eval()
    units = {"cpu_ne": ct.ComputeUnit.CPU_AND_NE, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}[a.units]
    ml = ct.models.MLModel(a.package, compute_units=units)
    g = torch.Generator().manual_seed(0)
    batch = {k: torch.rand(1, *f.shape, generator=g) for k, f in policy.config.image_features.items()}
    images, _ = policy.prepare_images(batch)
    print(f"policy dtypes: vlm {policy.model.vlm_with_expert.vlm.dtype}, expert {next(policy.model.vlm_with_expert.lm_expert.parameters()).dtype}, state_proj {policy.model.state_proj.weight.dtype}")
    bad = 0
    for k, img in zip(batch, images):
        with torch.no_grad():
            want = policy.model.vlm_with_expert.embed_image(img.to(policy.model.vlm_with_expert.vlm.dtype)).float().numpy()
        got = np.asarray(ml.predict({"pixel_values": img.numpy()})["image_embeddings"])
        err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        print(f"{k}: {tuple(want.shape)} relative L2 {err:.2e}")
        bad += err > a.max_rel_l2
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
