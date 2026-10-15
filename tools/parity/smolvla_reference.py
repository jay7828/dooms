"""PyTorch (lerobot, fp32, CPU) activations of SmolVLA for one fixed observation, for the engine's stage-by-stage parity tests.

    .venv-lerobot/bin/python tools/parity/smolvla_reference.py --out models/smolvla/reference.npz [--model lerobot/smolvla_base] [--seed 0]
The task text is stored in the GGUF form by tools/convert/convert_smolvla_host.py (--task).
Writes (all float32 unless noted; B=1, P=prefix tokens):
  image_<k> [3,H,W] raw camera images in 0..1;  tokens [48] int, token_mask [48] bool;  state [S];  noise [50,32]
  prepared_<k> [3,512,512]   the images after resize-with-pad and scaling to -1..1
  image_emb_<k> [64,960]     embed_image output (before the sqrt(960) scaling)
  prefix_embs [P,960], prefix_pad [P] bool, prefix_att [P] bool, prefix_pos [P] int
  kv_k_<i>, kv_v_<i> [P,5,64]  the prefix key/value cache of layer i (keys after RoPE)
  step_x_<s> [50,32], step_v_<s> [50,32]  input and velocity of denoising step s;  actions [50,32] final chunk (before cropping to the action dimension)
Exit codes: 0 ok, 1 error.
"""
import argparse
import sys

import numpy as np
import torch

PROMPT = "pick up the red cube and place it in the box"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks  # noqa: PLC0415
    policy = SmolVLAPolicy.from_pretrained(a.model).to("cpu").float().eval()
    cfg, m = policy.config, policy.model
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    batch = {}
    for key, feat in cfg.image_features.items():
        batch[key] = torch.rand(1, *feat.shape, generator=g)
        out["image_" + key.split(".")[-1]] = batch[key][0].numpy()
    batch["observation.state"] = torch.randn(1, cfg.robot_state_feature.shape[0], generator=g)
    tok = m.vlm_with_expert.processor.tokenizer
    enc = tok(PROMPT + "\n", return_tensors="pt", padding="max_length", max_length=cfg.tokenizer_max_length, truncation=True)
    batch["observation.language.tokens"] = enc["input_ids"]
    batch["observation.language.attention_mask"] = enc["attention_mask"].bool()
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    out.update(tokens=enc["input_ids"][0].numpy().astype(np.int64), token_mask=enc["attention_mask"][0].numpy().astype(bool), state=batch["observation.state"][0].numpy(), noise=noise[0].numpy())
    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        for key, img in zip(cfg.image_features, images):
            name = key.split(".")[-1]
            out["prepared_" + name] = img[0].numpy()
            out["image_emb_" + name] = m.vlm_with_expert.embed_image(img)[0].numpy()
        state = policy.prepare_state(batch)
        embs, pad, att = m.embed_prefix(images, img_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state=state)
        pos = torch.cumsum(pad, dim=1) - 1
        out.update(prefix_embs=embs[0].numpy(), prefix_pad=pad[0].numpy(), prefix_att=att[0].numpy(), prefix_pos=pos[0].numpy().astype(np.int64))
        _, kv = m.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att), position_ids=pos, past_key_values=None, inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
        for i in sorted(kv):
            out[f"kv_k_{i}"] = kv[i]["key_states"][0].numpy()
            out[f"kv_v_{i}"] = kv[i]["value_states"][0].numpy()
        x = noise
        dt = -1.0 / cfg.num_steps
        for s in range(cfg.num_steps):
            t = torch.tensor(1.0 + s * dt, dtype=torch.float32).expand(1)
            v = m.denoise_step(x_t=x, prefix_pad_masks=pad, past_key_values=kv, timestep=t)
            out[f"step_x_{s}"], out[f"step_v_{s}"] = x[0].numpy(), v[0].numpy()
            x = x + dt * v
        out["actions"] = x[0].numpy()
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(out)} arrays, prefix {embs.shape[1]} tokens, {len(kv)} cache layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
