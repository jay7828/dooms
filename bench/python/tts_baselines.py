#!/usr/bin/env python3
"""Text-to-speech baselines for Kokoro-82M: the same phonemes, voice and style rows for every engine, speed and intelligibility.

    .venv-kokoro/bin/python bench/python/tts_baselines.py prepare --out DIR             # misaki phonemes of the 73 LibriSpeech transcripts -> DIR/inputs.json
    .venv-kokoro/bin/python bench/python/tts_baselines.py all --inputs DIR/inputs.json --out DIR/raw.jsonl --wav-dir DIR/wav
    .venv/bin/python bench/python/tts_baselines.py score --inputs DIR/inputs.json --wav-dir DIR/wav --out DIR/scores.jsonl   # Whisper turbo (dooms.cpp) -> WER
    ... run --engine E --config C --inputs ... --out ... --wav-dir ...                    # one configuration in this interpreter

Engines and configurations (each `all` row runs in its own process with the interpreter of ENGINE_PYTHON):
  pytorch  cpu-fp32, mps-fp32       kokoro 0.9.4 KModel.forward_with_tokens (the reference implementation)
  mlx      gpu-fp32, gpu-fp16       mlx-audio's Kokoro, loaded from the same checkpoint (models/kokoro/mlx-fp32: the .pth as safetensors), fp16 = weights cast
  onnx     cpu-fp32                 kokoro-onnx 0.6.1 / ONNX Runtime, kokoro-v1.0.onnx, its _infer(tokens, style, speed) with our tokens and style
  dooms    default                  dooms.TTS.synthesize (prosody + F0/N on Core ML fp32 CPU, the vocoder on the engine's Metal kernels: f16 operands, f32 sums)
Every engine gets the same token ids and style row (voice row len(phonemes) - 1, as KModel picks it). Timed region: phonemes or ids -> float samples on the host
(model loaded, one warm-up sentence first); median of --repeats passes per sentence, summed over the sentences. The audio of the last pass is written to
<wav-dir>/<engine>_<config>/<k>.wav for `score`, which transcribes it with Whisper large-v3-turbo on dooms.cpp and computes the corpus WER against the text with
Whisper's English normalizer (bench/python/speech_baselines.py). Output: JSON lines. Exit codes: 0 ok, 1 a configuration failed, 2 usage, 3 unavailable.
"""
import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.abspath(__file__)
SAMPLE_RATE = 24000
KOKORO_DIR = os.path.join(REPO, "models", "kokoro", "Kokoro-82M")
MLX_DIR = os.path.join(REPO, "models", "kokoro", "mlx-fp32")
ONNX_DIR = os.path.join(REPO, "models", "kokoro", "onnx")
DATA = os.path.join(REPO, "models", "whisper", "data", "transcripts.tsv")
VOICE = "af_heart"
ENGINE_PYTHON = {"pytorch": os.path.join(REPO, ".venv-kokoro", "bin", "python"), "mlx": os.path.join(REPO, ".venv-mlx-audio", "bin", "python"),
                 "onnx": os.path.join(REPO, ".venv-kokoro-onnx", "bin", "python"), "dooms": os.path.join(REPO, ".venv-kokoro", "bin", "python")}
CONFIGS = [("pytorch", "cpu-fp32"), ("pytorch", "mps-fp32"), ("mlx", "gpu-fp32"), ("mlx", "gpu-fp16"), ("onnx", "cpu-fp32"), ("dooms", "default")]


class Unavailable(Exception):
    pass


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def write_wav(path, a):
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    a = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((a * 32767).round().astype("<i2").tobytes())


# ------------------------------------------------------------------------------------------------ engines: load(), synth(item) -> float32 samples

class PyTorchKokoro:
    def __init__(self, config):
        import torch  # noqa: PLC0415
        from kokoro import KModel  # noqa: PLC0415
        self.torch = torch
        self.dev = {"cpu-fp32": "cpu", "mps-fp32": "mps"}.get(config)
        if not self.dev:
            raise Unavailable(f"pytorch has no config {config}")
        if self.dev == "mps" and not torch.backends.mps.is_available():
            raise Unavailable("MPS not available")
        self.model = KModel(repo_id="hexgrad/Kokoro-82M", config=os.path.join(KOKORO_DIR, "config.json"), model=os.path.join(KOKORO_DIR, "kokoro-v1_0.pth")).eval().to(self.dev)

    def synth(self, item):
        t = self.torch
        with t.no_grad():
            audio, _ = self.model.forward_with_tokens(t.LongTensor([item["ids"]]).to(self.dev), t.tensor([item["ref_s"]], dtype=t.float32).to(self.dev), 1.0)
            return audio.float().cpu().numpy()

    def info(self):
        return {"version": f"kokoro {version('kokoro')}, torch {version('torch')}", "device": self.dev, "dtype": "fp32"}


class MlxKokoro:
    def __init__(self, config):
        import mlx.core as mx  # noqa: PLC0415
        from mlx_audio.tts.utils import load_model  # noqa: PLC0415
        self.mx = mx
        if config not in ("gpu-fp32", "gpu-fp16"):
            raise Unavailable(f"mlx has no config {config}")
        if not os.path.isfile(os.path.join(MLX_DIR, "model.safetensors")):
            raise Unavailable(f"{MLX_DIR} is missing (bench/python/CLAUDE.md: the checkpoint as safetensors)")
        self.model = load_model(MLX_DIR)
        self.dtype = mx.float16 if config == "gpu-fp16" else mx.float32
        if config == "gpu-fp16":
            self.model.set_dtype(mx.float16)

    def synth(self, item):
        mx = self.mx
        out = self.model(item["phonemes"], mx.array([item["ref_s"]], dtype=self.dtype), 1.0)
        mx.eval(out)
        import numpy as np  # noqa: PLC0415
        return np.array(out.astype(mx.float32)).reshape(-1)

    def info(self):
        return {"version": f"mlx-audio {version('mlx-audio')}, mlx {version('mlx')}", "device": "GPU", "dtype": "fp16" if self.dtype == self.mx.float16 else "fp32"}


class OnnxKokoro:
    def __init__(self, config):
        if config != "cpu-fp32":
            raise Unavailable(f"onnx has no config {config}")
        from kokoro_onnx import Kokoro  # noqa: PLC0415
        model = os.path.join(ONNX_DIR, "kokoro-v1.0.onnx")
        if not os.path.isfile(model):
            raise Unavailable(f"{model} is missing (bench/python/CLAUDE.md)")
        self.k = Kokoro(model, os.path.join(ONNX_DIR, "voices-v1.0.bin"))

    def synth(self, item):
        import numpy as np  # noqa: PLC0415
        audio, _ = self.k._infer(item["ids"][1:-1], np.asarray([item["ref_s"]], dtype=np.float32), 1.0)   # it adds the 0 pads itself
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def info(self):
        import onnxruntime  # noqa: PLC0415
        return {"version": f"kokoro-onnx {version('kokoro-onnx')}, onnxruntime {onnxruntime.__version__}", "device": "CPU", "dtype": "fp32",
                "providers": self.k.sess.get_providers()}


class DoomsKokoro:
    def __init__(self, config):
        if config != "default":
            raise Unavailable(f"dooms has no config {config}")
        sys.path.insert(0, os.path.join(REPO, "python"))
        os.environ.setdefault("DOOMS_LIB", os.path.join(REPO, "build", "libdooms_c.dylib"))
        import dooms  # noqa: PLC0415
        self.dooms = dooms
        self.tts = dooms.TTS(os.path.join(KOKORO_DIR, "dooms"))

    def synth(self, item):
        return self.tts.synthesize(item["phonemes"], VOICE, 1.0, 0)

    def info(self):
        return {"version": f"dooms {self.dooms.version()}", "device": "Core ML CPU (prosody, F0/N) + GPU (Metal vocoder)", "dtype": "fp32 prosody, f16 x f16 -> f32 vocoder"}


ENGINES = {"pytorch": PyTorchKokoro, "mlx": MlxKokoro, "onnx": OnnxKokoro, "dooms": DoomsKokoro}


# ------------------------------------------------------------------------------------------------ commands

def cmd_prepare(a):
    import torch  # noqa: PLC0415
    from kokoro import KPipeline  # noqa: PLC0415
    cfg = json.load(open(os.path.join(KOKORO_DIR, "config.json"), encoding="utf-8"))
    voice = torch.load(os.path.join(KOKORO_DIR, "voices", VOICE + ".pt"), weights_only=True)
    g2p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    items = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            name, text = line.rstrip("\n").split("\t", 1)
            text = text.lower()   # LibriSpeech is upper case: misaki would spell it letter by letter
            chunks = [r.phonemes for r in g2p(text, voice=voice)]
            ph = " ".join(chunks)
            if len(ph) > 510:
                continue
            ids = [0] + [cfg["vocab"][p] for p in ph if p in cfg["vocab"]] + [0]
            items.append({"name": name, "text": text, "phonemes": ph, "ids": ids, "ref_s": voice[len(ph) - 1][0].tolist()})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"voice": VOICE, "misaki": version("misaki"), "items": items}, open(os.path.join(a.out, "inputs.json"), "w", encoding="utf-8"))
    print(f"{len(items)} sentences, {sum(len(i['phonemes']) for i in items)} phonemes -> {a.out}/inputs.json")
    return 0


def cmd_run(a):
    data = json.load(open(a.inputs, encoding="utf-8"))
    items = data["items"][:a.limit] if a.limit else data["items"]
    try:
        eng = ENGINES[a.engine](a.config)
    except Unavailable as e:
        print(json.dumps({"engine": a.engine, "config": a.config, "status": "unavailable", "reason": str(e)}), flush=True)
        return 3
    row = {"engine": a.engine, "config": a.config, **eng.info(), "repeats": a.repeats, "loadavg_1m": os.getloadavg()[0]}
    eng.synth(items[0])   # warm-up (compilation, first allocations)
    per, audio_s, out_dir = [], 0.0, os.path.join(a.wav_dir, f"{a.engine}_{a.config}")
    os.makedirs(out_dir, exist_ok=True)
    for k, it in enumerate(items):
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            audio = eng.synth(it)
            ts.append(time.perf_counter() - t0)
        per.append(statistics.median(ts))
        audio_s += len(audio) / SAMPLE_RATE
        write_wav(os.path.join(out_dir, f"{k}.wav"), audio)
    total = sum(per)
    row.update({"status": "ok", "sentences": len(items), "audio_s": round(audio_s, 3), "total_s": round(total, 4), "rtf": round(total / audio_s, 5),
                "median_sentence_ms": round(1000 * statistics.median(per), 2), "first_sentence_ms": round(1000 * per[0], 2)})
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    return 0


def cmd_all(a):
    worst = 0
    for eng, cfg in CONFIGS:
        if a.only and not any(o in f"{eng}/{cfg}" for o in a.only.split(",")):
            continue
        py = ENGINE_PYTHON[eng]
        cmd = [py, SCRIPT, "run", "--engine", eng, "--config", cfg, "--inputs", a.inputs, "--wav-dir", a.wav_dir, "--repeats", str(a.repeats)]
        if a.out:
            cmd += ["--out", a.out]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        r = subprocess.run(cmd, cwd=REPO)
        worst = max(worst, 1 if r.returncode not in (0, 3) else 0)
    return worst


def cmd_score(a):
    sys.path.insert(0, os.path.join(REPO, "python"))
    sys.path.insert(0, os.path.dirname(SCRIPT))
    os.environ.setdefault("DOOMS_LIB", os.path.join(REPO, "build", "libdooms_c.dylib"))
    import dooms  # noqa: PLC0415
    import speech_baselines as sb  # noqa: PLC0415
    data = json.load(open(a.inputs, encoding="utf-8"))
    normalize = sb.make_normalizer(os.path.join(REPO, "models", "whisper", "whisper-large-v3-turbo"))
    stt = dooms.Whisper(os.path.join(REPO, "models", "whisper", "whisper-large-v3-turbo", "dooms"))
    rows = []
    for d in sorted(os.listdir(a.wav_dir)):
        files = sorted((int(f[:-4]), f) for f in os.listdir(os.path.join(a.wav_dir, d)) if f.endswith(".wav"))
        refs = [data["items"][k]["text"] for k, _ in files]
        hyps = [stt.transcribe(os.path.join(a.wav_dir, d, f)) for _, f in files]
        wer, edits, n = sb.corpus_wer(normalize, refs, hyps)
        row = {"config": d, "sentences": len(files), "wer": round(wer, 5), "edits": edits, "ref_words": n, "asr": "whisper-large-v3-turbo on dooms.cpp"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--out", required=True)
    for name in ("run", "all"):
        r = sub.add_parser(name)
        r.add_argument("--inputs", required=True)
        r.add_argument("--wav-dir", required=True)
        r.add_argument("--out", default=None)
        r.add_argument("--repeats", type=int, default=3)
        r.add_argument("--limit", type=int, default=0)
        if name == "run":
            r.add_argument("--engine", required=True, choices=sorted(ENGINES))
            r.add_argument("--config", required=True)
        else:
            r.add_argument("--only", default=None, help="comma-separated substrings of engine/config")
    sc = sub.add_parser("score")
    sc.add_argument("--inputs", required=True)
    sc.add_argument("--wav-dir", required=True)
    sc.add_argument("--out", default=None)
    a = p.parse_args(argv)
    return {"prepare": cmd_prepare, "run": cmd_run, "all": cmd_all, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
