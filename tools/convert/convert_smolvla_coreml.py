"""SmolVLA's language-model prefix and action expert -> Core ML packages (Neural Engine).

    .venv/bin/python tools/convert/convert_smolvla_coreml.py --vlm HF_DIR --policy-dir DIR_WITH_config.json_AND_model.safetensors --reference models/smolvla/reference.npz --out-dir models/smolvla

Writes prefix-fp16.mlpackage (embeds, rope_cos, rope_sin, mask_bias -> kv_k_<i>, kv_v_<i>, declared float16) and expert-fp16.mlpackage (all flow-matching steps in one graph: actions = the
noise, time_embs [steps,1,1,expert_hidden], self_cos, self_sin, cross_cos, cross_sin, mask_bias, kv_k_<i>, kv_v_<i> (float16) -> final_actions). The expert's cross-attention
projections of the prefix cache are computed once per call; one call replaces `steps` separate ones (58.8 ms instead of 80.6 ms for 10 steps). The prefix length is fixed by the reference observation (3 cameras x 64 image tokens + the padded
prompt + the state token). Each package is checked against lerobot's fp32 activations on the GPU units before it is kept. Exit codes: 0 ok, 1 error.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import smolvla_model as sm  # noqa: E402
from convert_common import ConversionError, _require  # noqa: E402


def convert_module(module, example, input_names, output_names, path, deployment="macOS14", half_inputs=(), half_outputs=()):
    """half_inputs / half_outputs: names declared as float16 (the runner then binds half-precision host buffers without a conversion pass)."""
    import coremltools as ct  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    mlm = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=tuple(x.shape), **({"dtype": np.float16} if n in half_inputs else {})) for n, x in zip(input_names, example)],
                     outputs=[ct.TensorType(name=n, **({"dtype": np.float16} if n in half_outputs else {})) for n in output_names],
                     convert_to="mlprogram", minimum_deployment_target=getattr(ct.target, deployment), compute_precision=ct.precision.FLOAT16)
    tmp = path + ".tmp.mlpackage"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    mlm.save(tmp)
    return tmp


def gpu_predict(tmp, inputs, half=()):
    import coremltools as ct  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return ct.models.MLModel(tmp, compute_units=ct.ComputeUnit.CPU_AND_GPU).predict({k: (v.numpy().astype(np.float16) if k in half else v.numpy()) for k, v in inputs.items()})


def keep(tmp, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-rel-l2", type=float, default=3e-2)
    a = ap.parse_args()
    try:
        cfg = sm.Config(a.vlm, os.path.join(a.policy_dir, "config.json"))
        tensors = sm.load_tensors(os.path.join(a.policy_dir, "model.safetensors"))
        ref = dict(np.load(a.reference))
        P = len(ref["prefix_pad"])
        os.makedirs(a.out_dir, exist_ok=True)

        prefix = sm.PrefixModel(cfg, tensors).eval()
        _require(sm.verify_prefix(cfg, prefix, ref) < 1e-4, "the torch prefix module does not reproduce lerobot")
        ex = sm.prefix_inputs(cfg, ref["prefix_embs"], ref["prefix_pad"], ref["prefix_att"], ref["prefix_pos"])
        names = ["embeds", "rope_cos", "rope_sin", "mask_bias"]
        outs = [f"kv_{kv}_{i}" for i in range(cfg.layers) for kv in "kv"]
        tmp = convert_module(prefix, ex, names, outs, os.path.join(a.out_dir, "prefix-fp16.mlpackage"), half_outputs=outs)  # the cache stays half precision from here on
        pred = gpu_predict(tmp, dict(zip(names, ex)))
        pad = ref["prefix_pad"]
        err = max(sm.rel_l2(np.asarray(pred[o])[0][pad], ref[o][pad]) for o in outs)
        _require(err <= a.max_rel_l2, f"prefix: Core ML differs from lerobot by {err:.2e} (> {a.max_rel_l2:.1e})")
        keep(tmp, os.path.join(a.out_dir, "prefix-fp16.mlpackage"))
        print(f"wrote prefix-fp16.mlpackage ({P} tokens, GPU-units worst cache error {err:.2e})")

        expert = sm.ExpertModel(cfg, tensors, P).eval()
        flow = sm.FlowModel(expert, cfg.steps).eval()
        _require(sm.verify_flow(cfg, flow, ref) < 1e-4, "the torch flow module does not reproduce lerobot")
        cache = []
        for i in range(cfg.layers):
            cache += [torch.from_numpy(ref[f"kv_k_{i}"])[None], torch.from_numpy(ref[f"kv_v_{i}"])[None]]
        tabs = sm.expert_tables(cfg, ref["prefix_pad"], 1.0)
        embs = torch.stack([torch.from_numpy(sm.time_embedding(1.0 - k / cfg.steps, cfg.expert_hidden, cfg.min_period, cfg.max_period))[None, None] for k in range(cfg.steps)])
        ex = (torch.from_numpy(ref["step_x_0"])[None], embs, *tabs[1:], *cache)
        names = ["actions", "time_embs", "self_cos", "self_sin", "cross_cos", "cross_sin", "mask_bias"] + outs
        tmp = convert_module(flow, ex, names, ["final_actions"], os.path.join(a.out_dir, "expert-fp16.mlpackage"), half_inputs=outs)
        pred = gpu_predict(tmp, dict(zip(names, ex)), half=outs)
        err = sm.rel_l2(np.asarray(pred["final_actions"])[0], ref["actions"])
        _require(err <= a.max_rel_l2, f"expert: Core ML differs from lerobot by {err:.2e} (> {a.max_rel_l2:.1e})")
        keep(tmp, os.path.join(a.out_dir, "expert-fp16.mlpackage"))
        print(f"wrote expert-fp16.mlpackage ({cfg.steps} flow steps in one graph, GPU-units action chunk error {err:.2e})")
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
