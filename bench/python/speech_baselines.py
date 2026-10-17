#!/usr/bin/env python3
"""Speech-to-text baselines for the Whisper benchmark: the same clips, protocol and WER for every engine (whisper.cpp, mlx-whisper, PyTorch, Sapient,
WhisperKit, and a dooms.cpp hook), one JSON line per (configuration, set).

    .venv/bin/python bench/python/speech_baselines.py list
    .venv/bin/python bench/python/speech_baselines.py run --engine whispercpp --config metal --model tiny [--limit 3] [--no-long] [--repeats 1]
    .venv/bin/python bench/python/speech_baselines.py all [--models tiny,large-v3-turbo] [--only whispercpp] [--out results.jsonl]

`run` executes one configuration in this interpreter; `all` runs every configuration, each in its own process with the interpreter its engine needs
(ENGINE_PYTHON below, override with --python ENGINE=PATH), and refuses to time anything while a job matching --busy-pattern runs (--allow-busy to
override, e.g. for smoke tests). An engine that is missing or fails prints a line with "status": "unavailable" or "error" and a reason; the run goes on.

Protocol (every engine): English, task transcribe, greedy (beam 1, temperature 0), temperature fallback off where the engine exposes it, no timestamp
tokens, no conditioning on the previous window's text. Audio longer than 30 s is decoded as consecutive 30-s windows (whisper.cpp and mlx-whisper do this
themselves when timestamps are off; the PyTorch runner cuts the windows; Sapient's own long-form mode without --timestamps). Inputs are the 16 kHz mono
16-bit WAVs from speech_testset.py; every engine reads the WAV itself, so file reading is inside the timed region for all of them.

Timing: model load is reported separately (load_s). Then one warm-up transcription (the first clip), then --repeats passes over the clip set and over
long.wav; wall_s is the median pass. Per file the timed region is, by engine:
  in-process (mlx, pytorch, dooms): WAV read + log-mel + encoder + decoding + detokenization, model already loaded and warmed;
  whispercpp (whisper-server, model loaded once): HTTP upload of the WAV on localhost + WAV parse + mel + encoder + decoding + JSON reply;
  sapient (one process per file, it has no server for audio): from its "<model> ready" line on stderr to process exit, i.e. audio decode +
    inference + printing + teardown, but each process starts cold (first-call kernel setup is inside the timed region).
WER: Whisper's EnglishTextNormalizer (transformers, with the model's normalizer.json) on reference and hypothesis, word-level Levenshtein; corpus WER over
the clips (sum of edits / sum of reference words) and long.wav against the concatenated references.

Exit codes: 0 ok, 1 a configuration failed, 2 usage, 3 configuration unavailable.
"""
import argparse
import importlib.metadata
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.abspath(__file__)

# Defaults, all overridable on the command line.
DATA_DIR = os.path.join(REPO, "models", "whisper", "data")
HF_DIR = os.path.join(REPO, "models", "whisper")                    # holds whisper-<model>/ snapshots (normalizer.json, PyTorch weights)
WHISPERCPP_DIR = os.path.join(REPO, "third_party", "whisper.cpp")
DEFAULT_REPEATS = 3
DEFAULT_MODELS = "tiny,large-v3-turbo"
BUSY_PATTERN = "mlx_lm|grpo.py|run.py --agent"                       # the user's training jobs; timing next to them is meaningless
SERVER_START_TIMEOUT_S = 600.0                                       # large-v3-turbo loads in seconds; this only bounds a hung server
REQUEST_TIMEOUT_S = 3600.0
ENGINE_PYTHON = {                                                    # interpreter per engine for `all`
    "whispercpp": os.path.join(REPO, ".venv", "bin", "python"),
    "mlx": os.path.join(REPO, ".venv-mlx-whisper", "bin", "python"),
    "pytorch": os.path.join(REPO, ".venv", "bin", "python"),
    "sapient": os.path.join(REPO, ".venv", "bin", "python"),
    "whisperkit": os.path.join(REPO, ".venv", "bin", "python"),
    "dooms": os.path.join(REPO, ".venv", "bin", "python"),
}
SAMPLE_RATE = 16000        # Whisper's input rate
WINDOW_S = 30              # Whisper's fixed input window

# Model names per engine. "hf" is the snapshot directory under HF_DIR (also the source of normalizer.json).
MODELS = {
    "tiny": {"hf": "whisper-tiny", "ggml": "ggml-tiny.bin", "mlx": "mlx-community/whisper-tiny",
             "sapient": "openhorizon/whisper-tiny"},
    "large-v3-turbo": {"hf": "whisper-large-v3-turbo", "ggml": "ggml-large-v3-turbo.bin", "mlx": "mlx-community/whisper-large-v3-turbo",
                       "sapient": "openhorizon/whisper-large-v3-turbo"},
}


class Unavailable(Exception):
    """The configuration cannot run on this machine; the message says what is missing and how to get it."""


# ---------------------------------------------------------------- audio, references, WER

def read_wav(path):
    """Float32 samples in [-1, 1) of a 16 kHz mono 16-bit WAV (as written by speech_testset.py)."""
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2:
            raise ValueError(f"{path}: {w.getnchannels()} ch, {w.getframerate()} Hz, {8 * w.getsampwidth()} bit; expected mono {SAMPLE_RATE} Hz 16 bit "
                             "(rebuild with bench/python/speech_testset.py)")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def load_transcripts(data_dir):
    path = os.path.join(data_dir, "transcripts.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing: build the test set with bench/python/speech_testset.py")
    rows = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{n}: expected 'file<TAB>text', got {line!r}")
        wav = os.path.join(data_dir, parts[0])
        if not os.path.isfile(wav):
            raise FileNotFoundError(f"{path}:{n}: {wav} does not exist")
        rows.append((wav, parts[1]))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def make_normalizer(hf_model_dir):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    path = os.path.join(hf_model_dir, "normalizer.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing: it ships with the Hugging Face snapshot of the model")
    return EnglishTextNormalizer(json.load(open(path, encoding="utf-8")))


def word_edits(ref, hyp):
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def corpus_wer(normalize, refs, hyps):
    edits = words = 0
    for r, h in zip(refs, hyps):
        rw, hw = normalize(r).split(), normalize(h).split()
        edits += word_edits(rw, hw)
        words += len(rw)
    return edits / words if words else float("nan"), edits, words


# ---------------------------------------------------------------- engines
# Each engine: __init__(model, args) checks availability (raises Unavailable), load() loads the model, transcribe(wav_path) -> text,
# close() frees it, info() -> dict of versions/settings for the JSON line, notes: protocol deviations.

def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def multipart(fields, file_path):
    boundary = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(file_path)}"\r\n'
               f'Content-Type: audio/wav\r\n\r\n'.encode())
    out.append(open(file_path, "rb").read())
    out.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


class WhisperCpp:
    """whisper.cpp's whisper-server, model loaded once; configs: metal (build/), cpu (build/, -ng), coreml (build-coreml/, Core ML encoder)."""
    BUILD = {"metal": "build", "cpu": "build", "coreml": "build-coreml"}
    notes = ("greedy, temperature 0, --no-fallback, no timestamps, server default no_context (no previous-text prompt); "
             "long audio: whisper_full seeks by a full 30-s window when no timestamp tokens are produced")

    def __init__(self, model, a):
        if a.config not in self.BUILD:
            raise Unavailable(f"whispercpp has configs {sorted(self.BUILD)}, not {a.config!r}")
        self.root = a.whispercpp_dir
        self.server = os.path.join(self.root, self.BUILD[a.config], "bin", "whisper-server")
        if not os.access(self.server, os.X_OK):
            how = ("cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON" if a.config != "coreml" else
                   "cmake -B build-coreml -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DWHISPER_COREML=1")
            raise Unavailable(f"{self.server} not built: in {self.root} run `{how} && cmake --build {self.BUILD[a.config]} -j`")
        self.model_file = os.path.join(self.root, "models", MODELS[model]["ggml"])
        if not os.path.isfile(self.model_file):
            raise Unavailable(f"{self.model_file} missing: run `sh models/download-ggml-model.sh {model}` in {self.root}")
        if a.config == "coreml":
            enc = self.model_file[:-len(".bin")] + "-encoder.mlmodelc"
            if not os.path.isdir(enc):
                raise Unavailable(f"{enc} missing: run `models/generate-coreml-model.sh {model}` in {self.root} (needs coremltools, openai-whisper, "
                                  "ane_transformers)")
        self.config, self.threads, self.proc, self.log = a.config, a.threads, None, None

    def load(self):
        port = free_port()
        cmd = [self.server, "-m", self.model_file, "--host", "127.0.0.1", "--port", str(port), "-l", "en", "-nt", "-nf", "-bs", "1"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        if self.config == "cpu":
            cmd.append("-ng")
        self.log = open(os.devnull, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"whisper-server exited with {self.proc.returncode} while loading; run `{' '.join(cmd)}` to see why")
            try:
                urllib.request.urlopen(self.url + "/", timeout=1).read()
                return
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                time.sleep(0.05)
        raise RuntimeError(f"whisper-server did not answer within {SERVER_START_TIMEOUT_S} s")

    def transcribe(self, path):
        body, ctype = multipart({"language": "en", "temperature": "0.0", "temperature_inc": "0.0", "no_timestamps": "true",
                                 "response_format": "json"}, path)
        req = urllib.request.Request(self.url + "/inference", data=body, headers={"Content-Type": ctype})
        reply = json.loads(urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S).read())
        if "text" not in reply:
            raise RuntimeError(f"whisper-server reply without text: {reply}")
        return reply["text"]

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        if self.log:
            self.log.close()

    def info(self):
        tag = subprocess.run(["git", "-C", self.root, "describe", "--tags", "--always"], capture_output=True, text=True).stdout.strip()
        return {"version": f"whisper.cpp {tag}", "model_file": os.path.basename(self.model_file), "dtype": "f16",
                "device": {"metal": "GPU (Metal)", "cpu": "CPU", "coreml": "encoder on Core ML (ANE/GPU), decoder on Metal"}[self.config]}


class MlxWhisper:
    """mlx-whisper (Apple MLX), fp16 on the GPU."""
    notes = ("temperature=0.0 alone (no fallback list), without_timestamps, condition_on_previous_text=False; long audio: transcribe() seeks by the "
             "full window when no timestamps; its no_speech_threshold (0.6) skip is left at its default")

    def __init__(self, model, a):
        if a.config != "gpu-fp16":
            raise Unavailable(f"mlx has config 'gpu-fp16' only, not {a.config!r}")
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise Unavailable(f"mlx_whisper not importable ({e}): `uv pip install --python .venv-mlx-whisper/bin/python mlx-whisper`")
        from huggingface_hub import snapshot_download
        self.name = MODELS[model]["mlx"]
        try:   # resolve the local snapshot so a download never lands inside load_s
            self.repo = snapshot_download(self.name, local_files_only=True)
        except Exception as e:
            raise Unavailable(f"{self.name} is not in the Hugging Face cache ({type(e).__name__}): `hf download {self.name}`")

    def load(self):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.get_model(self.repo, mx.float16)

    def transcribe(self, path):
        import mlx_whisper
        out = mlx_whisper.transcribe(read_wav(path), path_or_hf_repo=self.repo, language="en", task="transcribe", temperature=0.0,
                                     without_timestamps=True, condition_on_previous_text=False, fp16=True, verbose=None, beam_size=None)
        return out["text"]

    def close(self):
        pass

    def info(self):
        return {"version": f"mlx-whisper {pkg_version('mlx-whisper')}, mlx {pkg_version('mlx')}", "model_file": f"{self.name} ({self.repo})", "dtype": "fp16",
                "device": "GPU (MLX)"}


class PyTorchWhisper:
    """transformers WhisperForConditionalGeneration from the local snapshot; configs mps-fp16, cpu-fp32."""
    CONFIGS = {"mps-fp16": ("mps", "float16"), "cpu-fp32": ("cpu", "float32")}
    # SDPA by default; eager only where SDPA was measured to break: torch 2.8 MPS fp16 SDPA turns large-v3-turbo's output into NaN (token 0 repeated
    # to the length limit), while for tiny it is correct and ~3x faster than eager.
    EAGER = {("mps-fp16", "large-v3-turbo")}
    notes = ("generate(num_beams=1, do_sample=False, return_timestamps=False), no temperature fallback (only with a temperature tuple); long audio: "
             "this script cuts consecutive 30-s windows and decodes each independently (transformers' own long-form mode requires timestamps); "
             "SDPA attention except large-v3-turbo on MPS fp16, which runs eager (SDPA gives NaN there on torch 2.8)")

    def __init__(self, model, a):
        if a.config not in self.CONFIGS:
            raise Unavailable(f"pytorch has configs {sorted(self.CONFIGS)}, not {a.config!r}")
        try:
            import torch
        except ImportError as e:
            raise Unavailable(f"torch not importable ({e}): run with .venv/bin/python")
        self.device, dtype = self.CONFIGS[a.config]
        self.attn = a.torch_attn or ("eager" if (a.config, model) in self.EAGER else "sdpa")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise Unavailable("torch.backends.mps is not available in this interpreter")
        self.dtype = getattr(torch, dtype)
        self.dir = os.path.join(a.hf_dir, MODELS[model]["hf"])
        if not os.path.isfile(os.path.join(self.dir, "model.safetensors")):
            raise Unavailable(f"{self.dir}/model.safetensors missing: download openai/{MODELS[model]['hf']} into {self.dir}")
        self.threads = a.threads

    def load(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if self.threads:
            torch.set_num_threads(self.threads)
        self.proc = WhisperProcessor.from_pretrained(self.dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.dir, dtype=self.dtype, attn_implementation=self.attn).to(self.device).eval()

    def transcribe(self, path):
        import torch
        audio = read_wav(path)
        win = WINDOW_S * SAMPLE_RATE
        texts = []
        for start in range(0, len(audio), win):
            feats = self.proc(audio[start:start + win], sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            with torch.no_grad():
                ids = self.model.generate(feats.to(self.device, self.dtype), language="en", task="transcribe", num_beams=1, do_sample=False,
                                          return_timestamps=False)
            texts.append(self.proc.batch_decode(ids.cpu(), skip_special_tokens=True)[0])   # .cpu() waits for the device
        return " ".join(t.strip() for t in texts)

    def close(self):
        self.model = None

    def info(self):
        import torch
        return {"version": f"torch {torch.__version__}, transformers {pkg_version('transformers')}", "model_file": self.dir,
                "dtype": str(self.dtype).replace("torch.", ""), "device": self.device, "attention": self.attn}


class Sapient:
    """`sapient transcribe`, one process per file; configs cpu (--sapient-bin, --backend cpu) and metal (--sapient-metal-bin, --backend metal)."""
    notes = ("--beam-size 1 --language en, no --timestamps; temperature and fallback are not exposed (undocumented); long audio: its own mode "
             "without timestamps (no re-seek); no server for audio, so every file is a cold process: load_s is its own 'ready (N s)' report, "
             "and first-call kernel setup falls inside wall_s")
    READY = " ready ("

    def __init__(self, model, a):
        binary = {"cpu": a.sapient_bin, "metal": a.sapient_metal_bin}.get(a.config)
        if a.config not in ("cpu", "metal"):
            raise Unavailable(f"sapient has configs ['cpu', 'metal'], not {a.config!r}")
        if not binary or not os.access(binary, os.X_OK):
            flag = "--sapient-bin / SAPIENT_BIN" if a.config == "cpu" else "--sapient-metal-bin / SAPIENT_METAL_BIN"
            raise Unavailable(f"no sapient binary for {a.config!r} ({binary!r}): pass {flag}")
        self.binary, self.backend, self.model = binary, a.config, MODELS[model]["sapient"]
        self.load_reports = []

    def _run(self, path):
        """Returns (text, seconds from the ready line to exit, the load time sapient reported)."""
        cmd = [self.binary, "transcribe", "--backend", self.backend, "--language", "en", "--beam-size", "1", self.model, path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(self.binary))  # metallib sits next to it
        ready = {}
        err_lines = []

        def watch():
            for raw in p.stderr:
                line = raw.decode(errors="replace")
                err_lines.append(line)
                if self.READY in line and "t" not in ready:
                    ready["t"] = time.perf_counter()
                    try:
                        ready["load"] = float(line.split(self.READY, 1)[1].split("s)", 1)[0])
                    except ValueError:
                        pass
        th = threading.Thread(target=watch)
        th.start()
        out = p.stdout.read().decode(errors="replace")
        p.wait()
        t_end = time.perf_counter()
        th.join()
        if p.returncode != 0:
            raise RuntimeError(f"`{' '.join(cmd)}` exited {p.returncode}: {''.join(err_lines)[-500:]}")
        if "t" not in ready:
            raise RuntimeError(f"sapient printed no '{self.READY.strip()}' line, cannot separate load time; stderr: {''.join(err_lines)[-500:]}")
        self.load_reports.append(ready.get("load"))
        return out.strip(), t_end - ready["t"], ready.get("load")

    def load(self):
        pass   # every transcription loads the model; load_s is taken from sapient's own report (see run_config)

    def transcribe(self, path):
        text, seconds, _ = self._run(path)
        self.last_seconds = seconds
        return text

    def close(self):
        pass

    def info(self):
        v = subprocess.run([self.binary, "--version"], capture_output=True, text=True).stdout.strip()
        return {"version": v, "model_file": self.model, "dtype": "unknown (not reported by sapient)",
                "device": "CPU" if self.backend == "cpu" else "GPU (Metal)"}


class WhisperKit:
    """Argmax WhisperKit CLI; only a prebuilt whisperkit-cli on PATH (or --whisperkit-cli) is used, never an Xcode build."""
    notes = "not wired up: no prebuilt binary on this machine"

    def __init__(self, model, a):
        cli = a.whisperkit_cli or shutil.which("whisperkit-cli")
        raise Unavailable("whisperkit-cli not found (Homebrew is not installed and argmaxinc/WhisperKit releases ship no binaries)" if not cli else
                          f"whisperkit-cli found at {cli} but this script has no runner for it yet")


class Dooms:
    """dooms.cpp's Python package (python/ of this checkout, the library in build/): dooms.Whisper(dir).transcribe(path). Greedy, no timestamps, consecutive 30-s
    windows, suppress / begin-suppress lists from the generation config, no temperature fallback (the engine has none). Configs: the encoder's Core ML units."""
    UNITS = {"gpu": "cpu_gpu", "ane": "cpu_ne"}
    notes = "encoder fp16 on Core ML, decoder the engine's own Metal kernels (f16 weights); timed region includes the file decode"

    def __init__(self, model, a):
        if a.config not in self.UNITS:
            raise Unavailable(f"dooms has configs {sorted(self.UNITS)}, not {a.config!r}")
        sys.path.insert(0, os.path.join(REPO, "python"))
        os.environ.setdefault("DOOMS_LIB", os.path.join(REPO, "build", "libdooms_c.dylib"))
        try:
            import dooms
        except ImportError as e:
            raise Unavailable(f"dooms not importable ({e}): build the checkout (docs/USING.md)")
        self.dooms, self.units = dooms, self.UNITS[a.config]
        self.models_dir = a.dooms_models or os.path.join(a.hf_dir, MODELS[model]["hf"], "dooms")
        if not os.path.isfile(os.path.join(self.models_dir, "decoder.gguf")):
            raise Unavailable(f"{self.models_dir} has no converted model: tools/convert/convert_whisper.py --model {os.path.dirname(self.models_dir)}")

    def load(self):
        self.w = self.dooms.Whisper(self.models_dir, units=self.units)

    def transcribe(self, path):
        return self.w.transcribe(path)

    def close(self):
        if self.w is not None:
            self.w.close()
        self.w = None

    def info(self):
        return {"version": f"dooms {self.dooms.version()}", "model_file": self.models_dir, "dtype": "fp16 weights (encoder fp16 compute, decoder f32 accumulation)",
                "device": "GPU (Core ML encoder + Metal decoder)" if self.units == "cpu_gpu" else "Neural Engine encoder + GPU (Metal) decoder"}


ENGINES = {"whispercpp": WhisperCpp, "mlx": MlxWhisper, "pytorch": PyTorchWhisper, "sapient": Sapient, "whisperkit": WhisperKit, "dooms": Dooms}
CONFIGS = [("whispercpp", "metal"), ("whispercpp", "coreml"), ("mlx", "gpu-fp16"), ("pytorch", "mps-fp16"), ("pytorch", "cpu-fp32"),
           ("sapient", "metal"), ("sapient", "cpu"), ("whisperkit", "default"), ("dooms", "gpu"), ("dooms", "ane")]


# ---------------------------------------------------------------- running

def busy_jobs(pattern):
    """'pid command' (command cut to 120 characters) of every process matching pattern, except this script and its parent (whose arguments
    contain the pattern itself)."""
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if int(p) not in (os.getpid(), os.getppid())]
    if not pids:
        return []
    ps = subprocess.run(["ps", "-o", "pid=,args=", "-p", ",".join(pids)], capture_output=True, text=True)
    return [l.strip()[:120] for l in ps.stdout.splitlines() if SCRIPT not in l]


def timed_pass(engine, paths):
    """Transcribes each path once; returns (texts, seconds). Sapient reports its own per-file transcription time (after the load)."""
    texts, total = [], 0.0
    for p in paths:
        t0 = time.perf_counter()
        texts.append(engine.transcribe(p))
        dt = time.perf_counter() - t0
        total += getattr(engine, "last_seconds", dt)
    return texts, total


def run_config(a):
    base = {"engine": a.engine, "config": a.config, "model": a.model}
    rows = load_transcripts(a.data_dir)[:a.limit or None]
    long_wav = os.path.join(a.data_dir, "long.wav")
    if not a.no_long and not os.path.isfile(long_wav):
        raise FileNotFoundError(f"{long_wav} missing: build the test set with bench/python/speech_testset.py")
    normalize = make_normalizer(os.path.join(a.hf_dir, MODELS[a.model]["hf"]))
    engine = ENGINES[a.engine](a.model, a)
    common = {"busy_jobs": busy_jobs(a.busy_pattern), "loadavg_1m": round(os.getloadavg()[0], 2), "repeats": a.repeats, "warmup": 1,
              "notes": engine.notes}
    try:
        t0 = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - t0
        engine.transcribe(rows[0][0])                                  # warm-up, not timed
        if a.engine == "sapient":
            load_s = engine.load_reports[-1]                            # its own report; each file reloads
        meta = dict(base, **engine.info(), load_s=load_s, **common)
        sets = [("clips", [p for p, _ in rows], [t for _, t in rows])]
        if not a.no_long:
            sets.append(("long", [long_wav], [" ".join(t for _, t in load_transcripts(a.data_dir))]))
        for name, paths, refs in sets:
            walls, texts = [], None
            for _ in range(a.repeats):
                texts, w = timed_pass(engine, paths)
                walls.append(w)
            audio = sum(wav_seconds(p) for p in paths)
            wer, edits, words = corpus_wer(normalize, refs, texts)
            wall = statistics.median(walls)
            line = dict(meta, status="ok", set=name, n_files=len(paths), total_audio_s=round(audio, 3), wall_s=round(wall, 4),
                        wall_s_repeats=[round(w, 4) for w in walls], rtf=round(wall / audio, 5), wer=round(wer, 5), wer_edits=edits,
                        wer_ref_words=words)
            print(json.dumps(line), flush=True)
            if a.hyp_dir:
                os.makedirs(a.hyp_dir, exist_ok=True)
                with open(os.path.join(a.hyp_dir, f"{a.engine}_{a.config}_{a.model}_{name}.tsv"), "w", encoding="utf-8") as f:
                    for p, r, h in zip(paths, refs, texts):
                        f.write(f"{os.path.basename(p)}\t{normalize(r)}\t{normalize(h)}\n")
    finally:
        engine.close()


def cmd_run(a):
    try:
        run_config(a)
        return 0
    except Unavailable as e:
        print(json.dumps({"engine": a.engine, "config": a.config, "model": a.model, "status": "unavailable", "reason": str(e)}), flush=True)
        return 3
    except Exception as e:  # report and let `all` carry on with the next configuration
        print(json.dumps({"engine": a.engine, "config": a.config, "model": a.model, "status": "error", "reason": f"{type(e).__name__}: {e}"}),
              flush=True)
        return 1


def cmd_all(a):
    jobs = busy_jobs(a.busy_pattern)
    if jobs and not a.allow_busy:
        print(f"error: busy jobs match {a.busy_pattern!r}: {jobs}; wait for them (timings next to training are meaningless) or pass --allow-busy",
              file=sys.stderr)
        return 2
    pythons = dict(ENGINE_PYTHON)
    for kv in a.python:
        k, _, v = kv.partition("=")
        if k not in pythons or not v:
            print(f"error: --python expects ENGINE=PATH with ENGINE in {sorted(pythons)}, got {kv!r}", file=sys.stderr)
            return 2
        pythons[k] = v
    models = [m for m in a.models.split(",") if m]
    bad = [m for m in models if m not in MODELS]
    if bad:
        print(f"error: unknown model(s) {bad}; known: {sorted(MODELS)}", file=sys.stderr)
        return 2
    out = open(a.out, "a") if a.out else None   # appended: several `all --only` calls can share one file
    worst = 0
    for model in models:
        for engine, config in CONFIGS:
            if a.only and a.only not in f"{engine}/{config}":
                continue
            py = pythons[engine]
            cmd = [py, SCRIPT, "run", "--engine", engine, "--config", config, "--model", model, "--repeats", str(a.repeats),
                   "--data-dir", a.data_dir, "--hf-dir", a.hf_dir, "--whispercpp-dir", a.whispercpp_dir, "--busy-pattern", a.busy_pattern]
            for flag, val in (("--limit", a.limit), ("--threads", a.threads), ("--sapient-bin", a.sapient_bin),
                              ("--sapient-metal-bin", a.sapient_metal_bin), ("--whisperkit-cli", a.whisperkit_cli), ("--torch-attn", a.torch_attn),
                              ("--dooms-models", a.dooms_models), ("--hyp-dir", a.hyp_dir)):
                if val:
                    cmd += [flag, str(val)]
            if a.no_long:
                cmd.append("--no-long")
            if not os.access(py, os.X_OK):
                lines = [json.dumps({"engine": engine, "config": config, "model": model, "status": "unavailable",
                                     "reason": f"interpreter {py} missing; pass --python {engine}=PATH"})]
                rc = 3
            else:
                r = subprocess.run(cmd, capture_output=True, text=True)
                lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
                rc = r.returncode
                if not lines:
                    lines = [json.dumps({"engine": engine, "config": config, "model": model, "status": "error",
                                         "reason": f"exit {rc}, no JSON; stderr tail: {r.stderr[-500:]}"})]
            for l in lines:
                print(l, flush=True)
                if out:
                    out.write(l + "\n")
                    out.flush()
            if rc == 1:
                worst = 1
    if out:
        out.close()
    return worst


def add_common(p):
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help=f"timed passes per set, median reported (default {DEFAULT_REPEATS})")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke tests); 0 = all")
    p.add_argument("--no-long", action="store_true", help="skip long.wav")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for whisper.cpp / PyTorch (0 = engine default)")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--hf-dir", default=HF_DIR, help="directory with the whisper-<model>/ Hugging Face snapshots")
    p.add_argument("--whispercpp-dir", default=WHISPERCPP_DIR)
    p.add_argument("--sapient-bin", default=os.environ.get("SAPIENT_BIN"), help="sapient default (CPU) build")
    p.add_argument("--sapient-metal-bin", default=os.environ.get("SAPIENT_METAL_BIN"), help="sapient Metal build (mlx.metallib next to it)")
    p.add_argument("--torch-attn", choices=["eager", "sdpa"], default=None, help="override the PyTorch attention implementation of the config")
    p.add_argument("--whisperkit-cli", default=None)
    p.add_argument("--dooms-models", default=None, help="models directory for dooms.Whisper (default: the Hugging Face snapshot)")
    p.add_argument("--hyp-dir", default=None, help="write normalized reference/hypothesis TSVs here")
    p.add_argument("--busy-pattern", default=BUSY_PATTERN)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the configurations")
    r = sub.add_parser("run", help="one configuration in this interpreter")
    r.add_argument("--engine", required=True, choices=sorted(ENGINES))
    r.add_argument("--config", required=True)
    r.add_argument("--model", required=True, choices=sorted(MODELS))
    add_common(r)
    al = sub.add_parser("all", help="every configuration, one process each")
    al.add_argument("--models", default=DEFAULT_MODELS)
    al.add_argument("--only", default="", help="substring of engine/config")
    al.add_argument("--python", action="append", default=[], metavar="ENGINE=PATH")
    al.add_argument("--out", default=None, help="also write the JSON lines here")
    al.add_argument("--allow-busy", action="store_true")
    add_common(al)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for e, c in CONFIGS:
            print(f"{e}/{c}\t{ENGINE_PYTHON[e]}")
        return 0
    if a.repeats < 1 or a.limit < 0 or a.threads < 0:
        print("error: --repeats must be >= 1, --limit and --threads >= 0", file=sys.stderr)
        return 2
    return cmd_run(a) if a.cmd == "run" else cmd_all(a)


if __name__ == "__main__":
    sys.exit(main())
