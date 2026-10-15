"""PyTorch (lerobot) baseline for SmolVLA, one observation -> one 50-step action chunk: the numbers the engine is compared with.

    .venv-lerobot/bin/python bench/python/vla_baseline.py [--model lerobot/smolvla_base] [--devices cpu,mps] [--dtypes bf16,fp32] [--iters 10] [--save-reference PATH.npz]
Needs the separate Python 3.11 environment with lerobot (`uv venv --python 3.11 .venv-lerobot && uv pip install --python .venv-lerobot/bin/python "lerobot[smolvla]"`).
Input: 3 cameras (256x256, resized with padding to 512x512), a 6-dim state, a task prompt; fixed noise so runs are comparable. Measured per device (batch 1, 10 flow-matching steps with the prefix key/value cache; `bf16` is how lerobot loads the policy, `fp32` is the accurate oracle):
  embed     images -> SigLIP -> connector, language embedding, state projection
  fill      the 16 VLM layers over the whole prefix (fills the key/value cache)
  denoise   the 10 expert passes (sum)
  total     predict_action_chunk-equivalent: embed + fill + denoise
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

PROMPT = "pick up the red cube and place it in the box"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def build_batch(policy, device, seed):
    g = torch.Generator().manual_seed(seed)
    batch = {}
    for key, feat in policy.config.image_features.items():
        batch[key] = torch.rand(1, *feat.shape, generator=g).to(device)
    state_dim = policy.config.robot_state_feature.shape[0]
    batch["observation.state"] = torch.randn(1, state_dim, generator=g).to(device)
    tok = policy.model.vlm_with_expert.processor.tokenizer
    enc = tok(PROMPT + "\n", return_tensors="pt", padding="max_length", max_length=policy.config.tokenizer_max_length, truncation=True)
    batch["observation.language.tokens"] = enc["input_ids"].to(device)
    batch["observation.language.attention_mask"] = enc["attention_mask"].bool().to(device)
    return batch


@torch.no_grad()
def run_once(policy, batch, noise, device, timings=None):
    """sample_actions with stage timing (same calls, same order as VLAFlowMatching.sample_actions)."""
    m = policy.model
    cfg = policy.config
    t0 = time.perf_counter()
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # noqa: PLC0415
    prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    sync(device)
    t1 = time.perf_counter()
    att2d = make_att_2d_masks(prefix_pad, prefix_att)
    pos = torch.cumsum(prefix_pad, dim=1) - 1
    _, kv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=cfg.use_cache, fill_kv_cache=True)
    sync(device)
    t2 = time.perf_counter()
    dt = -1.0 / cfg.num_steps
    x_t = noise
    for step in range(cfg.num_steps):
        tt = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=device).expand(1)
        v = m.denoise_step(x_t=x_t, prefix_pad_masks=prefix_pad, past_key_values=kv, timestep=tt)
        x_t = x_t + dt * v
    sync(device)
    t3 = time.perf_counter()
    if timings is not None:
        timings.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3))
    return x_t[:, :, : cfg.action_feature.shape[0]], prefix_embs.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--dtypes", default="bf16,fp32")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--save-reference", default=None, help="write the cpu fp32 inputs and the action chunk to this .npz for parity tests")
    a = ap.parse_args()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
    print(f"torch {torch.__version__}")
    print(f"{'device':<12}{'embed ms':>10}{'fill ms':>10}{'denoise ms':>12}{'total ms':>10}{'prefix tokens':>15}")
    for device, dtype in [(d, t) for d in a.devices.split(",") for t in a.dtypes.split(",")]:
        policy = SmolVLAPolicy.from_pretrained(a.model).to(device).eval()
        if dtype == "fp32":
            policy = policy.float()
        batch = build_batch(policy, device, seed=0)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=torch.Generator().manual_seed(1)).to(device)
        timings = []
        for i in range(a.warmup + a.iters):
            actions, n_prefix = run_once(policy, batch, noise, device, timings if i >= a.warmup else None)
        med = [statistics.median(t[k] for t in timings) for k in range(4)]
        print(f"{device + ' ' + dtype:<12}{med[0]:>10.1f}{med[1]:>10.1f}{med[2]:>12.1f}{med[3]:>10.1f}{n_prefix:>15}", flush=True)
        if a.save_reference and device == "cpu" and dtype == "fp32":
            np.savez(a.save_reference, noise=noise.cpu().numpy(), actions=actions.cpu().numpy(), state=batch["observation.state"].cpu().numpy(),
                     tokens=batch["observation.language.tokens"].cpu().numpy(), **{k.replace(".", "_"): v.cpu().numpy() for k, v in batch.items() if "images" in k})
        del policy


if __name__ == "__main__":
    main()
