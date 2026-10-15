"""SmolVLA's language-model prefix and action expert as plain torch modules (no lerobot import), for Core ML export and for the host-side tables the C++ pipeline
must reproduce. Everything (sizes, layer pattern, epsilons) is read from the two configs and the checkpoint's tensor shapes.

Checked against lerobot's own fp32 activations (tools/parity/smolvla_reference.py -> reference.npz) by `verify_*`; see tools/convert/test_smolvla_model.py.

PrefixModel   (embeds, rope cos/sin, mask bias) -> keys and values of every VLM layer (keys after RoPE): the cache the expert reads.
ExpertModel   (noisy actions, time embedding, prefix cache, rope tables, mask bias) -> velocity, one flow-matching step.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from convert_common import ConversionError, _require

NEG = -30000.0  # additive mask value: representable in fp16, and exp() of it is exactly 0 after the max subtraction


# ---------------------------------------------------------------------------------------------
# Configuration and weights
# ---------------------------------------------------------------------------------------------

class Config:
    def __init__(self, vlm_dir: str, policy_config: str):
        vc = json.load(open(os.path.join(vlm_dir, "config.json")))["text_config"]
        pc = json.load(open(policy_config))
        for key in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps"):
            _require(vc.get(key) is not None, f"the text config has no '{key}'")
        _require(not vc.get("attention_bias"), "attention biases are not supported")
        self.hidden = int(vc["hidden_size"])
        self.heads, self.kv_heads, self.head_dim = int(vc["num_attention_heads"]), int(vc["num_key_value_heads"]), int(vc["head_dim"])
        self.eps = float(vc["rms_norm_eps"])
        self.layers = int(pc["num_vlm_layers"])
        self.self_every = int(pc["self_attn_every_n_layers"])
        _require(pc["attention_mode"] == "cross_attn" and int(pc["num_expert_layers"]) in (0, -1) or int(pc["num_expert_layers"]) == self.layers,
                 "only one expert layer per VLM layer in cross-attention mode is supported")
        self.expert_hidden = int(self.hidden * float(pc["expert_width_multiplier"]))
        self.chunk = int(pc["chunk_size"])
        self.action_dim = int(pc["max_action_dim"])
        self.state_dim = int(pc["max_state_dim"])
        self.steps = int(pc["num_steps"])
        self.min_period, self.max_period = float(pc["min_period"]), float(pc["max_period"])
        self.max_wavelength = 10_000.0  # lerobot's apply_rope default (not the text model's rope_theta)
        self.kv_dim = self.kv_heads * self.head_dim
        _require(self.heads % self.kv_heads == 0, "heads must be a multiple of the key/value heads")

    def self_layer(self, i: int) -> bool:
        """True: the expert layer attends over the prefix cache plus its own tokens; False: it attends over the cache only (cross-attention)."""
        return self.self_every > 0 and i % self.self_every == 0


def load_tensors(policy_path: str) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open  # noqa: PLC0415
    out = {}
    with safe_open(policy_path, "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).to(torch.float32)
    return out


# ---------------------------------------------------------------------------------------------
# Host-side tables (the C++ pipeline implements the same functions)
# ---------------------------------------------------------------------------------------------

def rope_tables(positions: np.ndarray, head_dim: int, max_wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
    """cos, sin of position / max_wavelength**(2 i / head_dim), shape [len(positions), head_dim/2], float32 (lerobot: float32 throughout)."""
    half = head_dim // 2
    exponents = (2.0 / head_dim) * np.arange(half, dtype=np.float32)
    timescale = np.float32(max_wavelength) ** exponents
    radians = positions.astype(np.float32)[:, None] / timescale[None, :]
    return np.cos(radians).astype(np.float32), np.sin(radians).astype(np.float32)


def prefix_mask(pad: np.ndarray, att: np.ndarray) -> np.ndarray:
    """lerobot's make_att_2d_masks for one example: bool [P, P], query row i may read key column j."""
    cum = np.cumsum(att.astype(np.int64))
    return (cum[None, :] <= cum[:, None]) & pad[None, :] & pad[:, None]


def time_embedding(t: float, dim: int, min_period: float, max_period: float) -> np.ndarray:
    """lerobot's create_sinusoidal_pos_embedding for one scalar time: [dim] float32 (computed in float64 like the original)."""
    _require(dim % 2 == 0, "the time embedding width must be even")
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    arg = (1.0 / period * 2 * math.pi) * t
    return np.concatenate([np.sin(arg), np.cos(arg)]).astype(np.float32)


def bias_of(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 0.0, NEG).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------------------------

def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rope(x, cos, sin):
    """x [1, L, H, D]; cos/sin [1, L, 1, D/2]: lerobot's apply_rope (halves are rotated pairwise)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attend(q, k, v, bias, groups, head_dim):
    """q [1, Lq, H, D]; k, v [1, Lk, Hkv, D]; bias [1, 1, Lq, Lk] -> [1, Lq, H*D] (grouped-query attention, softmax in the working precision)."""
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    scores = (q.transpose(1, 2) * head_dim ** -0.5) @ k.permute(0, 2, 3, 1) + bias
    out = torch.softmax(scores, dim=-1) @ v.transpose(1, 2)
    return out.transpose(1, 2).reshape(1, q.shape[1], -1)


class Layer(nn.Module):
    """A Llama decoder layer's tensors."""

    def __init__(self, t: Dict[str, torch.Tensor], prefix: str):
        super().__init__()
        g = lambda n: nn.Parameter(t[f"{prefix}.{n}"], requires_grad=False)  # noqa: E731
        self.ln1, self.ln2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        self.q, self.k, self.v, self.o = (g(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.gate, self.up, self.down = (g(f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))

    def mlp(self, x, eps):
        h = rms_norm(x, self.ln2, eps)
        return (torch.nn.functional.silu(h @ self.gate.T) * (h @ self.up.T)) @ self.down.T


class PrefixModel(nn.Module):
    def __init__(self, cfg: Config, t: Dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(Layer(t, f"model.vlm_with_expert.vlm.model.text_model.layers.{i}") for i in range(cfg.layers))

    def forward(self, embeds, cos, sin, bias):
        c = self.cfg
        x = embeds
        n = x.shape[1]
        outs: List[torch.Tensor] = []
        for i, l in enumerate(self.layers):
            h = rms_norm(x, l.ln1, c.eps)
            k = rope((h @ l.k.T).reshape(1, n, c.kv_heads, c.head_dim), cos, sin)
            v = (h @ l.v.T).reshape(1, n, c.kv_heads, c.head_dim)
            outs += [k, v]
            if i == len(self.layers) - 1:
                break  # the last layer's own output is never read: only its keys and values are cached
            q = rope((h @ l.q.T).reshape(1, n, c.heads, c.head_dim), cos, sin)
            x = x + attend(q, k, v, bias, c.heads // c.kv_heads, c.head_dim) @ l.o.T
            x = x + l.mlp(x, c.eps)
        return tuple(outs)


class ExpertModel(nn.Module):
    def __init__(self, cfg: Config, t: Dict[str, torch.Tensor], prefix_len: int):
        super().__init__()
        self.cfg, self.prefix_len = cfg, prefix_len
        self.layers = nn.ModuleList(Layer(t, f"model.vlm_with_expert.lm_expert.layers.{i}") for i in range(cfg.layers))
        for i, l in enumerate(self.layers):
            want = (cfg.kv_dim, cfg.expert_hidden if cfg.self_layer(i) else cfg.kv_dim)
            _require(tuple(l.k.shape) == want, f"expert layer {i}: k_proj is {tuple(l.k.shape)}, expected {want}")
        self.norm = nn.Parameter(t["model.vlm_with_expert.lm_expert.norm.weight"], requires_grad=False)
        p = lambda n: nn.Parameter(t[f"model.{n}"], requires_grad=False)  # noqa: E731
        self.in_w, self.in_b, self.out_w, self.out_b = p("action_in_proj.weight"), p("action_in_proj.bias"), p("action_out_proj.weight"), p("action_out_proj.bias")
        self.mi_w, self.mi_b, self.mo_w, self.mo_b = p("action_time_mlp_in.weight"), p("action_time_mlp_in.bias"), p("action_time_mlp_out.weight"), p("action_time_mlp_out.bias")

    def project_cache(self, cache):
        """The expert's cross-attention layers read the VLM cache through their own k/v projections; those do not depend on the flow step, so they are computed once.
        Returns per layer (k, v) [1, P, kv_heads, head_dim]: projected for cross layers, the cache itself for self layers."""
        c, P = self.cfg, self.prefix_len
        out = []
        for i, l in enumerate(self.layers):
            ck, cv = cache[2 * i], cache[2 * i + 1]
            if c.self_layer(i):
                out.append((ck, cv))
            else:
                out.append(((ck.reshape(1, P, -1) @ l.k.T).reshape(1, P, c.kv_heads, c.head_dim), (cv.reshape(1, P, -1) @ l.v.T).reshape(1, P, c.kv_heads, c.head_dim)))
        return out

    def step(self, actions, time_emb, self_cos, self_sin, cross_cos, cross_sin, bias, projected):
        """One velocity evaluation. actions [1, chunk, action_dim]; time_emb [1, 1, expert_hidden]; bias [1, 1, chunk, prefix + chunk]."""
        c, P = self.cfg, self.prefix_len
        n = actions.shape[1]
        a = actions @ self.in_w.T + self.in_b
        x = torch.cat([a, time_emb.expand(1, n, -1)], dim=-1) @ self.mi_w.T + self.mi_b
        x = torch.nn.functional.silu(x) @ self.mo_w.T + self.mo_b
        groups = c.heads // c.kv_heads
        for i, l in enumerate(self.layers):
            pk, pv = projected[i]
            h = rms_norm(x, l.ln1, c.eps)
            if c.self_layer(i):
                q = rope((h @ l.q.T).reshape(1, n, c.heads, c.head_dim), self_cos, self_sin)
                k = rope((h @ l.k.T).reshape(1, n, c.kv_heads, c.head_dim), self_cos, self_sin)
                v = (h @ l.v.T).reshape(1, n, c.kv_heads, c.head_dim)
                att = attend(q, torch.cat([pk, k], dim=1), torch.cat([pv, v], dim=1), bias, groups, c.head_dim)
            else:
                q = rope((h @ l.q.T).reshape(1, n, c.heads, c.head_dim), cross_cos, cross_sin)
                att = attend(q, pk, pv, bias[..., :P], groups, c.head_dim)
            x = x + att @ l.o.T
            x = x + l.mlp(x, c.eps)
        return rms_norm(x, self.norm, c.eps) @ self.out_w.T + self.out_b

    def forward(self, actions, time_emb, self_cos, self_sin, cross_cos, cross_sin, bias, *cache):
        """Single step, cache = (k_0, v_0, k_1, v_1, ...) each [1, prefix, kv_heads, head_dim] (the form exported to Core ML first)."""
        return self.step(actions, time_emb, self_cos, self_sin, cross_cos, cross_sin, bias, self.project_cache(cache))


class FlowModel(nn.Module):
    """`n_steps` Euler steps of the flow in one graph: x <- x + dt * velocity(x, t_s) with dt = -1 / total_steps; the cache projections are computed once.
    time_embs [n_steps, 1, 1, expert_hidden] holds the time embedding of every step in order."""

    def __init__(self, expert: ExpertModel, n_steps: int):
        super().__init__()
        self.expert, self.n_steps, self.dt = expert, n_steps, -1.0 / expert.cfg.steps

    def forward(self, actions, time_embs, self_cos, self_sin, cross_cos, cross_sin, bias, *cache):
        projected = self.expert.project_cache(cache)
        x = actions
        for s in range(self.n_steps):
            x = x + self.dt * self.expert.step(x, time_embs[s], self_cos, self_sin, cross_cos, cross_sin, bias, projected)
        return x


# ---------------------------------------------------------------------------------------------
# Host-side input construction from a reference observation (mirrors lerobot's masks and positions)
# ---------------------------------------------------------------------------------------------

def prefix_inputs(cfg: Config, embeds: np.ndarray, pad: np.ndarray, att: np.ndarray, positions: np.ndarray):
    cos, sin = rope_tables(positions, cfg.head_dim, cfg.max_wavelength)
    bias = bias_of(prefix_mask(pad, att))
    return (torch.from_numpy(embeds)[None], torch.from_numpy(cos)[None, :, None, :], torch.from_numpy(sin)[None, :, None, :], torch.from_numpy(bias)[None, None])


def expert_tables(cfg: Config, pad: np.ndarray, t: float):
    """time embedding, self/cross rope tables and the [chunk, P + chunk] mask bias for one step (positions as lerobot's denoise_step)."""
    P, n = len(pad), cfg.chunk
    valid = int(pad.sum())
    self_pos = valid + np.arange(n)
    s_cos, s_sin = rope_tables(self_pos, cfg.head_dim, cfg.max_wavelength)
    c_cos, c_sin = rope_tables(np.arange(n), cfg.head_dim, cfg.max_wavelength)  # cross layers: positions restart at 0
    suffix = np.tril(np.ones((n, n), dtype=bool))  # the action tokens form a causal block
    mask = np.concatenate([np.broadcast_to(pad[None, :], (n, P)), suffix], axis=1)
    f = lambda a: torch.from_numpy(a)[None, :, None, :]  # noqa: E731
    return (torch.from_numpy(time_embedding(t, cfg.expert_hidden, cfg.min_period, cfg.max_period))[None, None], f(s_cos), f(s_sin), f(c_cos), f(c_sin), torch.from_numpy(bias_of(mask))[None, None])


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def verify_prefix(cfg: Config, model: PrefixModel, ref) -> float:
    """Worst relative L2 error of the cached keys/values over the valid tokens against lerobot's reference."""
    pad = ref["prefix_pad"]
    with torch.no_grad():
        outs = model(*prefix_inputs(cfg, ref["prefix_embs"], pad, ref["prefix_att"], ref["prefix_pos"]))
    worst = 0.0
    for i in range(cfg.layers):
        for j, name in enumerate(("kv_k_", "kv_v_")):
            got, want = outs[2 * i + j][0].numpy()[pad], ref[f"{name}{i}"][pad]
            worst = max(worst, rel_l2(got, want))
    return worst


def verify_expert(cfg: Config, model: ExpertModel, ref, steps=(0, -1)) -> float:
    pad = ref["prefix_pad"]
    cache = []
    for i in range(cfg.layers):
        cache += [torch.from_numpy(ref[f"kv_k_{i}"])[None], torch.from_numpy(ref[f"kv_v_{i}"])[None]]
    worst = 0.0
    for s in steps:
        s = s % cfg.steps
        with torch.no_grad():
            v = model(torch.from_numpy(ref[f"step_x_{s}"])[None], *expert_tables(cfg, pad, 1.0 - s / cfg.steps), *cache)
        worst = max(worst, rel_l2(v[0].numpy(), ref[f"step_v_{s}"]))
    return worst


def verify_flow(cfg: Config, flow: FlowModel, ref) -> float:
    """Relative L2 error of `flow.n_steps` steps from the reference start against lerobot's x after those steps (all steps -> the final action chunk)."""
    pad = ref["prefix_pad"]
    cache = []
    for i in range(cfg.layers):
        cache += [torch.from_numpy(ref[f"kv_k_{i}"])[None], torch.from_numpy(ref[f"kv_v_{i}"])[None]]
    tabs = expert_tables(cfg, pad, 1.0)
    embs = torch.stack([torch.from_numpy(time_embedding(1.0 - s / cfg.steps, cfg.expert_hidden, cfg.min_period, cfg.max_period))[None, None] for s in range(flow.n_steps)])
    with torch.no_grad():
        x = flow(torch.from_numpy(ref["step_x_0"])[None], embs, *tabs[1:], *cache)
    want = ref["actions"] if flow.n_steps == cfg.steps else ref[f"step_x_{flow.n_steps}"]
    return rel_l2(x[0].numpy(), want)
