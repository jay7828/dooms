"""MLX baseline for SmolVLA (the third-party `mlx-smolvla` package, github.com/daniiarabdiev/mlx-smolvla): one observation -> 50-step action chunk with the same
checkpoint, inputs and noise as the engine's `bench_vla` / lerobot reference, timed like vla_baseline.py.

    .venv-mlx/bin/python bench/python/mlx_vla_baseline.py [--reference models/smolvla/reference.npz] [--dtypes fp32,bf16,fp16] [--iters 10]
Needs mlx==0.32.2 and mlx-smolvla (the package requires MLX >= 0.32.0, < 0.32.3). The package assembles a fixed 177-token prefix (two cameras); this checkpoint has three
cameras (241 tokens), so the module constant `mlx_smolvla.language._PREFIX_LENGTH` is raised to 241 here, which the accuracy check below validates against lerobot's
fp32 chunk (the same one the engine is checked against). The first load downloads and converts the checkpoint (minutes). Reported: p50/p99 of
predict_action_chunk (eval'd on the GPU) and the relative L2 error of the robot's 6 action dimensions against lerobot fp32 with the same noise.
"""
import argparse
import os
import statistics
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(ROOT, "models", "smolvla", "reference.npz"))
    ap.add_argument("--dtypes", default="fp32,bf16,fp16")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--task", default="pick up the red cube and place it in the box")
    a = ap.parse_args()

    import mlx.core as mx
    import mlx_smolvla.language as language
    language._PREFIX_LENGTH = 241
    from mlx_smolvla import SmolVLAMLX

    ref = np.load(a.reference)
    obs = {f"observation.images.camera{k}": np.round(ref[f"image_camera{k}"] * 255).astype(np.uint8) for k in (1, 2, 3)}
    obs["observation.state"] = ref["state"].astype(np.float32)
    obs["task"] = a.task
    noise = ref["noise"][None].astype(np.float32)
    want = ref["actions"][:, :6]
    print(f"mlx {mx.__version__}; device {mx.default_device()}")
    print(f"{'case':<26}{'p50 ms':>10}{'p99 ms':>10}   relative L2 vs lerobot fp32 (6 robot dims)")
    dts = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    for dtn in a.dtypes.split(","):
        try:
            policy = SmolVLAMLX.from_pretrained("lerobot/smolvla_base", dtype=dts[dtn])
            out = policy.predict_action_chunk(obs, noise=noise)
            mx.eval(out)
            got = np.array(out.astype(mx.float32))[0]
            err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
            for _ in range(a.warmup):
                mx.eval(policy.predict_action_chunk(obs, noise=noise))
            ts = []
            for _ in range(a.iters):
                t = time.perf_counter()
                mx.eval(policy.predict_action_chunk(obs, noise=noise))
                ts.append((time.perf_counter() - t) * 1e3)
            ts.sort()
            print(f"{'vla mlx ' + dtn:<26}{statistics.median(ts):>10.1f}{ts[min(len(ts) - 1, int(0.99 * len(ts)))]:>10.1f}   {err:.2e}")
        except Exception as e:  # noqa: BLE001
            print(f"{'vla mlx ' + dtn:<26} failed: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
