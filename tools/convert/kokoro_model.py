"""Kokoro-82M (hexgrad, Apache-2.0; StyleTTS 2 with an iSTFTNet vocoder) split for the dooms.cpp engine.

Three networks become Core ML packages, everything between them is small exact host work (the engine does it in C++; the numpy versions here are its
reference and are checked against the real model):

  prosody   phoneme ids [1, T] + style [1, 128]  -> durations [1, T] (sigmoid sums), prosody features d [1, T, 640], text features t_en [1, 512, T]
  host      round(durations / speed) clamped to >= 1 -> each phoneme repeated for its frames: en = d^T expanded [1, 640, F], asr = t_en expanded [1, 512, F]
  f0n       en [1, 640, F] + style [1, 128]      -> F0 [1, 2F], N [1, 2F]
  host      sine source from F0 (random initial phases of the 8 overtones and Gaussian noise: inputs, so runs are reproducible) -> tanh(linear) -> STFT
            (n_fft 20, hop 5, Hann, reflect padding) -> har [1, 22, 120F + 1] (magnitudes, then phases)
  vocoder   asr, F0, N, style (the voice's first 128 values), har -> x [1, 22, 120F + 1]
  host      spectrum exp(x[:11]), phase sin(x[11:]) -> iSTFT -> 600F samples at 24 kHz

The model's own code (kokoro 0.9.4) runs these steps inside KModel.forward_with_tokens; the wrappers below call its modules in the same order, with the LSTMs
run on the whole sequence (pack_padded_sequence of a single unpadded sequence is the identity) so that they trace.
"""
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 24000


def load(model_dir: str):
    """(KModel fp32 eval, config dict) from a directory with config.json and kokoro-v1_0.pth."""
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    from kokoro import KModel  # noqa: PLC0415
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg_path, model=os.path.join(model_dir, "kokoro-v1_0.pth")).eval()
    return model, cfg


def generator_constants(model) -> Dict[str, float]:
    g = model.decoder.generator
    sg = g.m_source.l_sin_gen
    return {"upsample": int(math.prod(g.f0_upsamp.scale_factor if isinstance(g.f0_upsamp.scale_factor, tuple) else (g.f0_upsamp.scale_factor,))),
            "harmonics": int(sg.dim), "sine_amp": float(sg.sine_amp), "noise_std": float(sg.noise_std), "voiced_threshold": float(sg.voiced_threshold),
            "sample_rate": int(sg.sampling_rate), "n_fft": int(g.post_n_fft), "hop": int(g.stft.hop_length)}


_INV_SQRT2 = float(np.float32(2.0 ** -0.5))   # torch.rsqrt(torch.tensor(2)) in float32


def _adain_res_forward(self, x, s):
    """AdainResBlk1d.forward with its residual scale as a float constant (the original takes rsqrt of an int tensor, which Core ML rejects); same value."""
    return (self._residual(x, s) + self._shortcut(x)) * _INV_SQRT2


def patch_for_export(module: nn.Module) -> None:
    from kokoro.istftnet import AdainResBlk1d  # noqa: PLC0415
    for m in module.modules():
        if isinstance(m, AdainResBlk1d):
            m.forward = _adain_res_forward.__get__(m)


def albert_forward(bert, ids):
    """transformers' AlbertModel.forward(ids).last_hidden_state for one unpadded sequence, written with its own submodules (the library's mask code uses ops
    Core ML cannot convert; with no padding the mask is a no-op)."""
    e = bert.embeddings
    T = ids.shape[1]
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    h = e.word_embeddings(ids) + e.token_type_embeddings.weight[0] + e.position_embeddings(pos)
    h = e.LayerNorm(h)
    enc = bert.encoder
    h = enc.embedding_hidden_mapping_in(h)
    cfg = bert.config
    for i in range(cfg.num_hidden_layers):
        group = enc.albert_layer_groups[int(i / (cfg.num_hidden_layers / cfg.num_hidden_groups))]
        for layer in group.albert_layers:
            a = layer.attention
            shape = (1, T, a.num_attention_heads, a.attention_head_size)
            q = a.query(h).view(shape).transpose(1, 2)
            k = a.key(h).view(shape).transpose(1, 2)
            v = a.value(h).view(shape).transpose(1, 2)
            w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * a.scaling, dim=-1)
            o = torch.matmul(w, v).transpose(1, 2).reshape(1, T, a.all_head_size)
            att = a.LayerNorm(h + a.dense(o))
            ff = layer.ffn_output(layer.activation(layer.ffn(att)))
            h = layer.full_layer_layer_norm(ff + att)
    return h


class Prosody(nn.Module):
    """input_ids [1, T] (float32 holding the integer ids, with the 0 pads at both ends: the engine's Core ML runner exchanges float tensors; ids < 2^24 are exact),
    style [1, 128] (the voice's last 128 values) -> (durations [1, T], d [1, T, 640], t_en [1, 512, T])."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, style):
        m = self.m
        ids = input_ids.long()
        bert = albert_forward(m.bert, ids)
        d_en = m.bert_encoder(bert).transpose(-1, -2)                      # [1, 512, T]
        de = m.predictor.text_encoder
        T = ids.shape[1]
        s_seq = style.unsqueeze(1).expand(-1, T, -1)                          # [1, T, 128]
        x = torch.cat([d_en.transpose(-1, -2), s_seq], dim=-1)               # [1, T, 640]
        for block in de.lstms:
            if isinstance(block, nn.LSTM):
                x, _ = block(x)                                               # [1, T, 512]
            else:                                                             # AdaLayerNorm over channels, then the style appended again
                x = block(x, style)
                x = torch.cat([x, s_seq], dim=-1)
        d = x                                                                 # [1, T, 640]
        h, _ = m.predictor.lstm(d)
        durations = torch.sigmoid(m.predictor.duration_proj(h)).sum(dim=-1)   # [1, T]
        te = m.text_encoder
        t = te.embedding(ids).transpose(1, 2)                                 # [1, 512, T]
        for c in te.cnn:
            t = c(t)
        t, _ = te.lstm(t.transpose(1, 2))
        t_en = t.transpose(1, 2)                                              # [1, 512, T]
        return durations, d, t_en


class F0N(nn.Module):
    """en [1, 640, F], style [1, 128] -> (F0 [1, 2F], N [1, 2F])."""

    def __init__(self, model):
        super().__init__()
        self.p = model.predictor
        patch_for_export(self.p)

    def forward(self, en, style):
        return self.p.F0Ntrain(en, style)


class Vocoder(nn.Module):
    """asr [1, 512, F], F0 [1, 2F], N [1, 2F], style [1, 128] (the voice's first 128 values), har [1, 22, 120F + 1] -> x [1, 22, 120F + 1] (before exp / sin)."""

    def __init__(self, model):
        super().__init__()
        self.d = model.decoder
        patch_for_export(self.d)

    def forward(self, asr, f0_curve, n_curve, style, har):
        dec = self.d
        f0 = dec.F0_conv(f0_curve.unsqueeze(1))
        n = dec.N_conv(n_curve.unsqueeze(1))
        x = torch.cat([asr, f0, n], dim=1)
        x = dec.encode(x, style)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n], dim=1)
            x = block(x, style)
            if block.upsample_type != "none":
                res = False
        g = dec.generator
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = g.noise_res[i](g.noise_convs[i](har), style)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(g.num_kernels):
                r = g.resblocks[i * g.num_kernels + j](x, style)
                xs = r if xs is None else xs + r
            x = xs / g.num_kernels
        x = F.leaky_relu(x)
        return g.conv_post(x)


# ------------------------------------------------------------------------------------------------ host stages (numpy reference of the engine's C++)

def align(durations: np.ndarray, d: np.ndarray, t_en: np.ndarray, speed: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """durations [T] (sigmoid sums), d [T, 640], t_en [512, T] -> (frames per phoneme [T] int, en [640, F], asr [512, F]): round(dur / speed), at least 1."""
    frames = np.maximum(np.round(np.asarray(durations, np.float32) / np.float32(speed)), 1).astype(np.int64)   # torch.round: half to even, as np.round
    idx = np.repeat(np.arange(len(frames)), frames)
    return frames, np.ascontiguousarray(d[idx].T), np.ascontiguousarray(t_en[:, idx])


def _interp_linear(x: np.ndarray, out_len: int, scale: float) -> np.ndarray:
    """torch F.interpolate(mode='linear', align_corners=False, scale_factor=1/scale given) along axis 0: source index (i + 0.5) * scale - 0.5, clamped at 0."""
    n = x.shape[0]
    src = (np.arange(out_len, dtype=np.float32) + np.float32(0.5)) * np.float32(scale) - np.float32(0.5)   # float32, as torch's CPU kernel
    src = np.maximum(src, np.float32(0.0))
    i0 = np.minimum(src.astype(np.int64), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    lam = (src - i0.astype(np.float32))[:, None]
    return (np.float32(1) - lam) * x[i0] + lam * x[i1]


def sine_source(f0: np.ndarray, rand_ini: np.ndarray, noise: np.ndarray, c: Dict[str, float], w: np.ndarray, b: float) -> np.ndarray:
    """f0 [2F] -> the harmonic excitation [L = 2F * upsample]: SineGen (fundamental + overtones, linear-interpolated phase) * uv + noise, then tanh(w . s + b).
    rand_ini [harmonics] (initial phase per overtone, index 0 forced to 0), noise [L, harmonics] standard normal."""
    up, H = c["upsample"], c["harmonics"]
    f0u = np.repeat(np.asarray(f0, np.float32), up)                                  # nearest upsampling
    fn = f0u[:, None] * np.arange(1, H + 1, dtype=np.float32)[None, :]                # [L, H]
    rad = np.mod(fn / np.float32(c["sample_rate"]), np.float32(1.0)).astype(np.float32)
    ini = np.asarray(rand_ini, np.float32).copy()
    ini[0] = 0
    rad[0, :] += ini
    L = rad.shape[0]
    down = _interp_linear(rad, L // up, float(up))                                    # scale_factor 1/up
    phase = np.cumsum(down, axis=0, dtype=np.float64).astype(np.float32) * np.float32(2) * np.float32(np.pi)   # torch: cumsum (in double on the CPU) * 2 * pi
    phase_up = _interp_linear(phase * np.float32(up), L, 1.0 / up)                    # scale_factor up
    sines = np.sin(phase_up).astype(np.float32) * np.float32(c["sine_amp"])
    uv = (f0u > c["voiced_threshold"]).astype(np.float32)[:, None]
    amp = uv * np.float32(c["noise_std"]) + (1 - uv) * np.float32(c["sine_amp"] / 3)
    s = sines * uv + amp * np.asarray(noise, np.float32)
    return np.tanh(s @ np.asarray(w, np.float32) + np.float32(b)).astype(np.float32)


def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float64)    # periodic


def stft_mag_phase(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """torch.stft(center=True, pad reflect, periodic Hann, onesided) -> [n_fft + 2, frames]: magnitudes then phases (atan2)."""
    pad = n_fft // 2
    xp = np.pad(np.asarray(x, np.float64), (pad, pad), mode="reflect")
    frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.fft.rfft(xp[idx] * _hann(n_fft)[None, :], axis=1).T                     # [bins, frames]
    return np.concatenate([np.abs(spec), np.angle(spec)], axis=0).astype(np.float32)


def istft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """The vocoder output [n_fft + 2, frames] -> samples: spectrum exp(x[:bins]), phase sin(x[bins:]), torch.istft(center=True, periodic Hann) semantics."""
    bins = n_fft // 2 + 1
    spec = np.exp(np.asarray(x[:bins], np.float64)) * np.exp(1j * np.sin(np.asarray(x[bins:], np.float64)))
    frames = spec.shape[1]
    win = _hann(n_fft)
    seg = np.fft.irfft(spec.T, n=n_fft, axis=1) * win[None, :]                        # [frames, n_fft]
    total = n_fft + hop * (frames - 1)
    out = np.zeros(total)
    env = np.zeros(total)
    for f in range(frames):
        out[f * hop:f * hop + n_fft] += seg[f]
        env[f * hop:f * hop + n_fft] += win ** 2
    pad = n_fft // 2
    length = hop * (frames - 1)
    return (out[pad:pad + length] / env[pad:pad + length]).astype(np.float32)


def draw_noise(f0_frames: int, c: Dict[str, float], generator: torch.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """The sine source's random inputs in the order SineGen draws them (torch.rand for the initial phases, then torch.randn for the noise)."""
    L = f0_frames * c["upsample"]
    rand_ini = torch.rand(1, c["harmonics"], generator=generator)
    noise = torch.randn(1, L, c["harmonics"], generator=generator)
    return rand_ini[0].numpy(), noise[0].numpy()


@torch.no_grad()
def synthesize_split(model, input_ids: np.ndarray, ref_s: np.ndarray, speed: float, rand_ini: np.ndarray, noise: np.ndarray, nets=None):
    """The whole split path with the given noise; `nets` (optional) replaces (prosody, f0n, vocoder) callables (e.g. Core ML) taking/returning numpy."""
    c = generator_constants(model)
    lin = model.decoder.generator.m_source.l_linear
    w, b = lin.weight.detach().numpy()[0], float(lin.bias.detach().numpy()[0])
    if nets is None:
        P, Fn, V = Prosody(model).eval(), F0N(model).eval(), Vocoder(model).eval()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        prosody = lambda ids, s: tuple(o.numpy() for o in P(t(ids), t(s)))  # noqa: E731
        f0n = lambda en, s: tuple(o.numpy() for o in Fn(t(en), t(s)))  # noqa: E731
        vocoder = lambda asr, f0, n, s, har: V(t(asr), t(f0), t(n), t(s), t(har)).numpy()  # noqa: E731
    else:
        prosody, f0n, vocoder = nets
    ref_s = np.asarray(ref_s, np.float32).reshape(1, -1)
    ids = np.asarray(input_ids, np.int32).reshape(1, -1)
    dur, d, t_en = prosody(ids, ref_s[:, 128:])
    frames, en, asr = align(dur[0], d[0], t_en[0], speed)
    f0, n = f0n(en[None], ref_s[:, 128:])
    har_src = sine_source(f0[0], rand_ini, noise, c, w, b)
    har = stft_mag_phase(har_src, c["n_fft"], c["hop"])[None]
    x = vocoder(asr[None], f0, n, ref_s[:, :128], har)
    return istft(x[0], c["n_fft"], c["hop"]), frames
