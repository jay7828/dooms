#!/usr/bin/env python3
"""Convert Kokoro-82M (hexgrad, Apache-2.0) for the dooms.cpp engine.

    .venv-kokoro/bin/python tools/convert/convert_kokoro.py --model models/kokoro/Kokoro-82M [--out-dir DIR] [--voices af_heart,am_michael,...]
Writes into --out-dir (default: <model>/dooms):
  prosody-fp32.mlpackage  phoneme ids + style -> durations, prosody features, text features (Core ML, flexible length; CPU)
  f0n-fp32.mlpackage      prosody features per frame + style -> F0 and energy curves (Core ML, flexible length; CPU)
  kokoro.gguf             the vocoder's weights (weight norm folded: f16 convolution kernels, f32 vectors), the sine source's merge layer, the vocabulary,
                          the voices and the constants (docs/formats/kokoro-gguf.md)
Every output is checked against the PyTorch model before it is moved into place. Exit codes: 0 ok, 1 failed (nothing left half-written), 2 bad arguments.
Needs the reference package (kokoro 0.9.4) and coremltools: the .venv-kokoro environment (bench/python/CLAUDE.md).
"""
import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert_common as cc  # noqa: E402
import kokoro_model as km  # noqa: E402

ARCH = "kokoro"
MAX_TOKENS = 512        # ALBERT's positions (KModel.context_length)
MAX_FRAMES = 16384      # frames per call the Core ML packages accept (~410 s of audio); the pipeline splits longer input


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def convert_package(module, example, inputs, outputs, path: str, max_rel: float, check_examples) -> float:
    """Traces `module`, converts it (fp32, macOS 14), checks it on the CPU against PyTorch on every example, moves it into place; returns the worst error."""
    import coremltools as ct  # noqa: PLC0415
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    mlm = ct.convert(traced, inputs=inputs, outputs=outputs, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                     compute_precision=ct.precision.FLOAT32)
    tmp = path + ".tmp.mlpackage"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    mlm.save(tmp)
    m = ct.models.MLModel(tmp, compute_units=ct.ComputeUnit.CPU_ONLY)
    worst = 0.0
    for ex in check_examples:
        with torch.no_grad():
            ref = module(*ex)
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = m.predict({i.name: e.numpy() for i, e in zip(inputs, ex)})
        for o, r in zip(outputs, ref):
            worst = max(worst, rel(got[o.name], r.numpy()))
    cc._require(np.isfinite(worst) and worst <= max_rel, f"{os.path.basename(path)}: Core ML differs from PyTorch by {worst:.2e} > {max_rel}")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)
    return worst


def fold_weight_norm(module: torch.nn.Module) -> torch.nn.Module:
    """Folds every weight norm of `module` into a plain weight, in place (what the forward pre-hook computes on each call); returns it. Weight-normed modules
    cannot be deep-copied once they have run, so callers fold a freshly loaded model."""
    for sub in module.modules():
        if hasattr(sub, "weight_g") and hasattr(sub, "weight_v"):
            torch.nn.utils.remove_weight_norm(sub)
    return module


def write_gguf(model, folded, cfg, voices: dict, out_path: str) -> int:
    import gguf  # noqa: PLC0415
    dec = folded.decoder
    c = km.generator_constants(model)
    w = gguf.GGUFWriter(out_path + ".tmp", ARCH)
    k = lambda name: f"{ARCH}.{name}"  # noqa: E731
    for name in ("sample_rate", "n_fft", "hop", "upsample", "harmonics"):
        w.add_uint32(k(name), int(c[name]))
    for name in ("sine_amp", "noise_std", "voiced_threshold"):
        w.add_float32(k(name), float(c[name]))
    w.add_uint32(k("style_dim"), int(cfg["style_dim"]))
    w.add_uint32(k("hidden_dim"), int(cfg["hidden_dim"]))
    w.add_uint32(k("max_tokens"), MAX_TOKENS)
    w.add_uint32(k("max_frames"), MAX_FRAMES)
    w.add_uint32(k("max_duration"), int(cfg["max_dur"]))
    ist = cfg["istftnet"]
    w.add_array(k("upsample_rates"), [int(x) for x in ist["upsample_rates"]])
    w.add_array(k("upsample_kernel_sizes"), [int(x) for x in ist["upsample_kernel_sizes"]])
    w.add_array(k("resblock_kernel_sizes"), [int(x) for x in ist["resblock_kernel_sizes"]])
    w.add_uint32(k("upsample_initial_channel"), int(ist["upsample_initial_channel"]))
    symbols = sorted(cfg["vocab"], key=lambda s: cfg["vocab"][s])
    w.add_array(k("vocab.symbols"), symbols)
    w.add_array(k("vocab.ids"), [int(cfg["vocab"][s]) for s in symbols])
    w.add_array(k("voices"), sorted(voices))
    n = 0
    for name, p in dec.state_dict().items():
        t = p.detach().float().numpy()
        # convolution kernels as f16 (the Metal kernels' operand type), everything else f32
        is_kernel = name.endswith(".weight") and t.ndim == 3
        w.add_tensor("dec." + name, t.astype(np.float16) if is_kernel else t.astype(np.float32))
        n += 1
    lin = model.decoder.generator.m_source.l_linear
    w.add_tensor("source.linear.weight", lin.weight.detach().float().numpy().astype(np.float32))
    w.add_tensor("source.linear.bias", lin.bias.detach().float().numpy().astype(np.float32))
    for vname, v in sorted(voices.items()):
        w.add_tensor(f"voice.{vname}", np.ascontiguousarray(v.reshape(v.shape[0], -1)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    verify_gguf(out_path + ".tmp", dec, voices)
    os.replace(out_path + ".tmp", out_path)
    return n


def verify_gguf(path: str, dec, voices: dict) -> None:
    import gguf  # noqa: PLC0415
    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}
    sd = dec.state_dict()
    cc._require(sum(1 for n in t if n.startswith("dec.")) == len(sd), "not every vocoder tensor was written")
    for name, p in sd.items():
        got = np.asarray(t["dec." + name].data, dtype=np.float32).reshape(tuple(p.shape))
        tol = 2e-3 if p.ndim == 3 else 0.0
        cc._require(np.allclose(got, p.detach().float().numpy(), rtol=tol, atol=tol * 1e-2 + 1e-12), f"dec.{name} differs from the folded model after writing")
    for vname, v in voices.items():
        got = np.asarray(t[f"voice.{vname}"].data, dtype=np.float32).reshape(v.shape[0], -1)
        cc._require(np.array_equal(got, v.reshape(v.shape[0], -1)), f"voice {vname} differs after writing")


def check_folded(model, folded) -> float:
    """The folded vocoder must compute what the weight-normed one does."""
    torch.manual_seed(0)
    Fr = 40
    ex = (torch.randn(1, 512, Fr) * 0.5, torch.rand(1, 2 * Fr) * 200, torch.randn(1, 2 * Fr), torch.randn(1, 128) * 0.3, torch.rand(1, 22, 120 * Fr + 1))
    V = km.Vocoder(model).eval()
    Vf = km.Vocoder(folded).eval()
    with torch.no_grad():
        return rel(Vf(*ex).numpy(), V(*ex).numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="directory with config.json, kokoro-v1_0.pth and voices/*.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--voices", default=None, help="comma-separated voice names (default: every voices/*.pt)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.model):
        print(f"error: {a.model} is not a directory", file=sys.stderr)
        return 2
    out = a.out_dir or os.path.join(a.model, "dooms")
    os.makedirs(out, exist_ok=True)
    try:
        import coremltools as ct  # noqa: PLC0415
        t0 = time.time()
        model, cfg = km.load(a.model)
        names = a.voices.split(",") if a.voices else sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(a.model, "voices", "*.pt")))
        voices = {}
        for n in names:
            p = os.path.join(a.model, "voices", n + ".pt")
            cc._require(os.path.isfile(p), f"no voice file {p}")
            voices[n] = torch.load(p, map_location="cpu", weights_only=True).float().numpy()
            cc._require(voices[n].ndim == 3 and voices[n].shape[1] == 1 and voices[n].shape[2] == 2 * cfg["style_dim"], f"voice {n} has shape {voices[n].shape}")
        cc._require(bool(voices), "no voices")
        torch.manual_seed(0)
        sd = cfg["style_dim"]
        # prosody: ids [1, T] + style [1, 128]
        P = km.Prosody(model).eval()
        ex = lambda T: (torch.randint(1, cfg["n_token"], (1, T)).float(), torch.randn(1, sd) * 0.3)  # noqa: E731
        err = convert_package(P, ex(64), [ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(3, MAX_TOKENS, default=64))),
                                          ct.TensorType(name="style", shape=(1, sd))],
                              [ct.TensorType(name="durations"), ct.TensorType(name="d"), ct.TensorType(name="t_en")],
                              os.path.join(out, "prosody-fp32.mlpackage"), 2e-3, [ex(9), ex(64), ex(300)])
        print(f"prosody-fp32.mlpackage written, relative L2 vs PyTorch (CPU) {err:.1e}")
        Fn = km.F0N(model).eval()
        hd = cfg["hidden_dim"] + sd
        exf = lambda Fr: (torch.randn(1, hd, Fr) * 0.5, torch.randn(1, sd) * 0.3)  # noqa: E731
        err = convert_package(Fn, exf(200), [ct.TensorType(name="en", shape=(1, hd, ct.RangeDim(1, MAX_FRAMES, default=200))), ct.TensorType(name="style", shape=(1, sd))],
                              [ct.TensorType(name="f0"), ct.TensorType(name="n")], os.path.join(out, "f0n-fp32.mlpackage"), 2e-3, [exf(7), exf(200), exf(1500)])
        print(f"f0n-fp32.mlpackage written, relative L2 vs PyTorch (CPU) {err:.1e}")
        folded, _ = km.load(a.model)
        fold_weight_norm(folded.decoder)
        err = check_folded(model, folded)
        cc._require(err <= 1e-5, f"folding the weight norms changed the vocoder's output by {err:.2e}")
        n = write_gguf(model, folded, cfg, voices, os.path.join(out, "kokoro.gguf"))
        print(f"kokoro.gguf written: {n} vocoder tensors (weight norm folded: {err:.1e}), {len(voices)} voices ({time.time() - t0:.0f} s in all)")
    except cc.ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
