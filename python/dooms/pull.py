"""`dooms pull` / `dooms list`: download models from Hugging Face and verify them (standard library only).

    dooms pull qwen2.5-0.5b                                   # an alias of ALIASES: a GGUF from its publisher's own repository, pinned to a commit
    dooms pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf[@REVISION]
    dooms pull whisper-tiny | whisper-large-v3-turbo | kokoro   # converted files (PREBUILT), checked by pinned SHA-256; --convert converts the originals
    dooms pull smolvlm | smolvla                              # original weights + local conversion (source checkout only, see CONVERTED)

Every file is checked against the hash Hugging Face publishes for it (the SHA-256 of an LFS file, `lfs.oid` in the paths-info API; the git blob SHA-1 of a small
file, `oid`). Downloads go to `<file>.part`, resume with an HTTP Range request, and are renamed into place only after the check; a mismatch deletes the partial file.

Environment: DOOMS_MODELS (the models directory, default ~/.cache/dooms), HF_ENDPOINT (default https://huggingface.co), HF_TOKEN (gated or private repositories; only
sent to the endpoint's own host, never to the CDN a download redirects to), DOOMS_SOURCE_DIR / DOOMS_CONVERT_PYTHON / DOOMS_LEROBOT_PYTHON / DOOMS_LLAMA_CPP_DIR (conversion).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MODELS_ENV = "DOOMS_MODELS"
DEFAULT_MODELS_DIR = os.path.join("~", ".cache", "dooms")
HF_ENDPOINT_ENV = "HF_ENDPOINT"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HF_TOKEN_ENV = "HF_TOKEN"
CHUNK_BYTES = 1 << 20          # read/hash granularity
TIMEOUT_SECONDS = 60           # per socket operation (connect, each read), not for the whole download
PART_SUFFIX = ".part"
SOURCES_SUBDIR = "sources"     # original checkpoints downloaded for local conversion: <models>/sources/<owner>/<repo>
USER_AGENT = "dooms-pull"
DOWNLOAD_ATTEMPTS = 5          # a download that breaks off is resumed this many times before pull gives up
RETRY_WAIT_SECONDS = 2.0       # times the attempt number, between attempts


def models_dir(override: Optional[str] = None) -> str:
    """The models directory: `override`, else $DOOMS_MODELS, else ~/.cache/dooms."""
    return os.path.abspath(os.path.expanduser(override or os.environ.get(MODELS_ENV) or DEFAULT_MODELS_DIR))


@dataclass(frozen=True)
class GgufAlias:
    repo: str
    file: str
    revision: str   # a commit, so an alias always means the same bytes
    note: str


# Only repositories published by the model's owner (or ggml-org, llama.cpp's own organisation). Each was checked against the Hugging Face API (author, commit, file list).
ALIASES: Dict[str, GgufAlias] = {
    "qwen2.5-0.5b": GgufAlias("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "9217f5db79a29953eb74d5343926648285ec7e67", "Qwen2.5 0.5B Instruct Q4_K_M, Apache-2.0"),
    "qwen2.5-1.5b": GgufAlias("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "91cad51170dc346986eccefdc2dd33a9da36ead9", "Qwen2.5 1.5B Instruct Q4_K_M, Apache-2.0"),
    "qwen3-0.6b": GgufAlias("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "23749fefcc72300e3a2ad315e1317431b06b590a", "Qwen3 0.6B Q8_0, Apache-2.0"),
    "smollm2-1.7b": GgufAlias("HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF", "smollm2-1.7b-instruct-q4_k_m.gguf", "2d4a76a30b4af41ecd395c35725ac11688d4cfe4",
                              "SmolLM2 1.7B Instruct Q4_K_M, Apache-2.0"),
    "gemma-3-1b": GgufAlias("ggml-org/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf", "f9c28bcd85737ffc5aef028638d3341d49869c27", "Gemma 3 1B IT Q4_K_M, Gemma licence"),
}


@dataclass(frozen=True)
class Snapshot:
    repo: str
    revision: str
    exclude_prefixes: Tuple[str, ...] = ()   # files of the repository the converters do not read (ONNX exports, notebooks, ...)


# Models the engine runs only after conversion (Core ML packages + GGUF). ADR 0002: converted weights are never redistributed, so `pull` downloads the ORIGINAL weights
# from the official repositories and runs the repository's converters locally.
SMOLVLM = Snapshot("HuggingFaceTB/SmolVLM-256M-Instruct", "7e3e67edbbed1bf9888184d9df282b700a323964", ("onnx/",))
SMOLVLA_VLM = Snapshot("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467", ("onnx/",))
SMOLVLA_POLICY = Snapshot("lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6", ("Finetune_SmolVLA_notebook.ipynb", "collage_small.gif"))
# Whisper (MIT): the snapshot, and the encoder check's limit (convert_whisper.py --encoder-max-rel-l2; large-v3-turbo's 32 fp16 layers measure 2.28e-2 on random features,
# its real-speech parity is in tests/unit/test_whisper.cpp). Other framework exports in the repository are not read.
_WHISPER_SKIP = ("flax_model.msgpack", "tf_model.h5", "pytorch_model.bin")
WHISPER: Dict[str, Tuple[Snapshot, float]] = {
    "whisper-tiny": (Snapshot("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", _WHISPER_SKIP), 2e-2),
    "whisper-large-v3-turbo": (Snapshot("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", _WHISPER_SKIP), 3e-2),
}
# Kokoro-82M (Apache-2.0): the voices are part of the snapshot; its evaluation images and samples are not read. Converted by tools/convert/convert_kokoro.py in an
# environment with the reference package (kokoro 0.9.4: the converter checks every output against it).
KOKORO = Snapshot("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", ("eval/", "samples/"))
KOKORO_MODULES = ("torch", "kokoro", "coremltools", "gguf")
CONVERTED = ("smolvlm", "smolvla") + tuple(WHISPER) + ("kokoro",)


@dataclass(frozen=True)
class Prebuilt:
    file: str
    size: int
    sha256: str
    note: str


# Converted archives of the permissively licensed models (docs/adr/0002 addendum 2026-10-18; scripts/package-models.py), assets of the jay7828/dooms release
# PREBUILT_RELEASE. The hash is pinned here, so a changed asset is refused. `dooms pull NAME` downloads these; `--convert` converts the original weights instead.
PREBUILT_RELEASE = "models-v1"
PREBUILT_URL = "https://github.com/jay7828/dooms/releases/download/{release}/{file}"
PREBUILT: Dict[str, Prebuilt] = {
    "whisper-tiny": Prebuilt("dooms-whisper-tiny.tar.gz", 69470346, "535f30c77f4819d31143794195fd067bc91c428016d0bdd351bae71a0b695dbb", "Whisper tiny (OpenAI, MIT)"),
    "whisper-large-v3-turbo": Prebuilt("dooms-whisper-large-v3-turbo.tar.gz", 1492091280, "6e2d98a483854d127476d32c4011704104882ef63e8b813a3c8060e45de6d171",
                                       "Whisper large-v3-turbo (OpenAI, MIT)"),
    "kokoro": Prebuilt("dooms-kokoro.tar.gz", 237102426, "b5fa9026c663e2c75f141c892fb18fd4dddfef59c9955f6d05b842f2abaf2b80", "Kokoro-82M, 54 voices (hexgrad, Apache-2.0)"),
}
CONVERT_MODULES = ("torch", "coremltools", "transformers", "gguf", "safetensors", "huggingface_hub")   # what tools/convert imports
LEROBOT_MODULES = ("torch", "lerobot")                                                                  # tools/parity/smolvla_reference.py


class PullError(Exception):
    """A failure with a message that says what to do."""


class TransientError(PullError):
    """A network failure that a later attempt may not have (connection refused or reset, timeout, HTTP 429 / 5xx)."""


# ------------------------------------------------------------------------------------------------ HTTP

def hf_endpoint() -> str:
    return (os.environ.get(HF_ENDPOINT_ENV) or DEFAULT_HF_ENDPOINT).rstrip("/")


class _NoAuthAcrossHosts(urllib.request.HTTPRedirectHandler):
    """Follows redirects (Hugging Face sends file downloads to a CDN) but drops the Authorization header when the host changes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_NoAuthAcrossHosts)


def _open(url: str, what: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None):
    h = {"User-Agent": USER_AGENT}
    token = os.environ.get(HF_TOKEN_ENV)
    if token and urllib.parse.urlsplit(url).netloc == urllib.parse.urlsplit(hf_endpoint()).netloc:
        h["Authorization"] = f"Bearer {token}"
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if data is not None else "GET")
    try:
        return _opener.open(req, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            raise  # the caller handles an unsatisfiable Range
        detail = ""
        try:
            detail = json.loads(e.read().decode(errors="replace")).get("error", "")
        except (ValueError, AttributeError, OSError):
            pass
        if e.code == 404:
            raise PullError(f"{what}: not found on {hf_endpoint()} (HTTP 404{': ' + detail if detail else ''}); check the repository, file and revision names") from None
        if e.code in (401, 403):
            raise PullError(f"{what}: access denied (HTTP {e.code}{': ' + detail if detail else ''}). The repository does not exist, or it is private or gated: "
                            f"check the name, accept its licence on the website and set {HF_TOKEN_ENV}") from None
        cls = TransientError if e.code == 429 or e.code >= 500 else PullError
        raise cls(f"{what}: HTTP {e.code} from {urllib.parse.urlsplit(url).netloc}{': ' + detail if detail else ''}; try again later") from None
    except urllib.error.URLError as e:
        raise TransientError(f"{what}: cannot reach {urllib.parse.urlsplit(url).netloc} ({e.reason}); check the network connection (or {HF_ENDPOINT_ENV})") from None
    except (TimeoutError, OSError) as e:
        raise TransientError(f"{what}: network error ({e}); check the connection and run the command again (downloads resume)") from None


def _get_json(url: str, what: str, body: Optional[dict] = None):
    data = json.dumps(body).encode() if body is not None else None
    with _open(url, what, {"Content-Type": "application/json"} if data else None, data) as r:
        try:
            return json.loads(r.read().decode())
        except ValueError:
            raise PullError(f"{what}: the server's answer is not JSON (is {HF_ENDPOINT_ENV} a Hugging Face endpoint?)") from None


def _quote(path: str) -> str:
    return urllib.parse.quote(path, safe="/")


def resolve_revision(repo: str, revision: str) -> str:
    """The commit a branch, tag or commit names."""
    d = _get_json(f"{hf_endpoint()}/api/models/{_quote(repo)}/revision/{_quote(revision)}", f"{repo}@{revision}")
    sha = d.get("sha") if isinstance(d, dict) else None
    if not isinstance(sha, str) or len(sha) != 40:
        raise PullError(f"{repo}@{revision}: the API did not return a commit")
    return sha


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    algorithm: str   # "sha256" (LFS) or "git-sha1" (git blob of a small file)
    digest: str


def _remote_file(entry: dict, what: str) -> RemoteFile:
    try:
        lfs = entry.get("lfs")
        if lfs:
            return RemoteFile(entry["path"], int(lfs["size"]), "sha256", str(lfs["oid"]).lower())
        return RemoteFile(entry["path"], int(entry["size"]), "git-sha1", str(entry["oid"]).lower())
    except (KeyError, TypeError, ValueError):
        raise PullError(f"{what}: the API's file description has no size or hash; refusing to download unverified") from None


def file_info(repo: str, commit: str, path: str) -> RemoteFile:
    what = f"{repo}/{path}"
    entries = _get_json(f"{hf_endpoint()}/api/models/{_quote(repo)}/paths-info/{commit}", what, {"paths": [path]})
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and e.get("path") == path and e.get("type") == "file":
            return _remote_file(e, what)
    raise PullError(f"{what}: no such file in {repo} at {commit[:12]} (list the repository's files on the website)")


def list_files(repo: str, commit: str) -> List[RemoteFile]:
    """Every file of the repository at `commit` (following the API's pagination)."""
    url = f"{hf_endpoint()}/api/models/{_quote(repo)}/tree/{commit}?recursive=true"
    out = []
    while url:
        with _open(url, f"{repo} file list") as r:
            try:
                entries = json.loads(r.read().decode())
            except ValueError:
                raise PullError(f"{repo} file list: the server's answer is not JSON") from None
            link = r.headers.get("Link", "")
        out += [_remote_file(e, f"{repo}/{e.get('path')}") for e in entries if isinstance(e, dict) and e.get("type") == "file"]
        url = None
        for part in link.split(","):
            if 'rel="next"' in part and "<" in part:
                url = urllib.parse.urljoin(hf_endpoint() + "/", part[part.index("<") + 1: part.index(">")])
    return out


# ------------------------------------------------------------------------------------------------ verified, resumable download

def _hasher(f: RemoteFile):
    if f.algorithm == "sha256":
        return hashlib.sha256()
    h = hashlib.sha1()  # git blob: sha1("blob <size>\0" + content)
    h.update(b"blob %d\0" % f.size)
    return h


def _hash_file(path: str, h) -> int:
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK_BYTES)
            if not b:
                return n
            h.update(b)
            n += len(b)


def verify(path: str, f: RemoteFile) -> bool:
    h = _hasher(f)
    return _hash_file(path, h) == f.size and h.hexdigest() == f.digest


Progress = Callable[[str, int, int], None]


_progress_shown: Dict[str, int] = {}  # file name -> last 10% step printed (non-terminal stderr)


def stderr_progress(name: str, done: int, total: int) -> None:
    """Progress on stderr: an updating line on a terminal, a line per 10% otherwise."""
    pct = 100 * done // total if total else 100
    if sys.stderr.isatty():
        sys.stderr.write(f"\r  {name}: {done / 1e6:,.1f} / {total / 1e6:,.1f} MB ({pct}%)" + ("\n" if done >= total else ""))
        sys.stderr.flush()
    elif _progress_shown.get(name) != pct // 10:
        _progress_shown[name] = pct // 10
        print(f"  {name}: {pct}% of {total / 1e6:,.1f} MB", file=sys.stderr, flush=True)


def download(repo: str, commit: str, f: RemoteFile, dest: str, progress: Optional[Progress] = stderr_progress, attempts: Optional[int] = None,
             url: Optional[str] = None) -> str:
    """Downloads one file to `dest` and verifies it, resuming up to `attempts` times after a network failure. Returns "present" (already there and verified) or
    "downloaded". `url` (optional) replaces the Hugging Face address built from repo / commit / path (a release asset with a pinned hash)."""
    attempts = attempts or DOWNLOAD_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(repo, commit, f, dest, progress, url)
        except TransientError as e:
            if attempt == attempts:
                raise
            print(f"  {e}; retrying ({attempt + 1}/{attempts})", file=sys.stderr, flush=True)
            time.sleep(RETRY_WAIT_SECONDS * attempt)
    raise AssertionError("unreachable")


def _download_once(repo: str, commit: str, f: RemoteFile, dest: str, progress: Optional[Progress], direct_url: Optional[str] = None) -> str:
    name = os.path.basename(dest)
    if os.path.exists(dest):
        if verify(dest, f):
            return "present"
        raise PullError(f"{dest} exists but does not match {repo}/{f.path} ({f.algorithm} {f.digest[:16]}...); delete it to download it again")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + PART_SUFFIX
    h = _hasher(f)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > f.size:
        os.remove(part)
        have = 0
    if have:
        if _hash_file(part, h) != have:
            raise PullError(f"{part} changed while it was being read; run the command again")
    url = direct_url or f"{hf_endpoint()}/{_quote(repo)}/resolve/{commit}/{_quote(f.path)}"
    if have < f.size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            resp = _open(url, f"{repo}/{f.path}", headers)
        except urllib.error.HTTPError as e:  # 416: the server cannot serve that range; start over
            e.close()
            os.remove(part)
            return _download_once(repo, commit, f, dest, progress, direct_url)
        with resp:
            status = resp.status
            if have and status == 206:
                content_range = resp.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {have}-"):
                    raise PullError(f"{repo}/{f.path}: the server resumed at the wrong offset ({content_range!r}); delete {part} and run the command again")
                mode = "ab"
            elif status == 200:
                if have:  # the server ignored the Range header: the body is the whole file
                    h, have = _hasher(f), 0
                mode = "wb"
            else:
                raise PullError(f"{repo}/{f.path}: unexpected HTTP {status} for a download")
            with open(part, mode) as out:
                while True:
                    try:
                        b = resp.read(CHUNK_BYTES)
                    except (TimeoutError, OSError) as e:
                        raise TransientError(f"{repo}/{f.path}: the download broke off at {have} of {f.size} bytes ({e}); run the command again to resume") from None
                    if not b:
                        break
                    if have + len(b) > f.size:
                        out.close()
                        os.remove(part)
                        raise PullError(f"{repo}/{f.path}: the server sent more than the {f.size} bytes the API announced; the partial file was deleted")
                    out.write(b)
                    h.update(b)
                    have += len(b)
                    if progress:
                        progress(name, have, f.size)
                out.flush()
                os.fsync(out.fileno())
    if have != f.size:
        raise TransientError(f"{repo}/{f.path}: the download ended at {have} of {f.size} bytes; run the command again to resume")
    if h.hexdigest() != f.digest:
        os.remove(part)
        raise PullError(f"{repo}/{f.path}: checksum mismatch ({f.algorithm} {h.hexdigest()[:16]}..., expected {f.digest[:16]}...); the partial file was deleted, "
                        f"run the command again")
    os.replace(part, dest)  # atomic: `dest` only ever exists complete and verified
    return "downloaded"


# ------------------------------------------------------------------------------------------------ what a name means

def parse_hf(spec: str) -> Tuple[str, str, str]:
    """hf:OWNER/REPO/PATH[@REVISION] -> (repo, path, revision)."""
    body = spec[len("hf:"):]
    revision = "main"
    if "@" in body:
        body, revision = body.rsplit("@", 1)
    parts = body.split("/")
    if len(parts) < 3 or not all(parts) or not revision:
        raise PullError(f"'{spec}': expected hf:OWNER/REPO/FILE.gguf[@REVISION]")
    if not parts[-1].endswith(".gguf"):
        raise PullError(f"'{spec}': only .gguf files can be pulled this way (the engine loads GGUF language models)")
    return "/".join(parts[:2]), "/".join(parts[2:]), revision


def pull_gguf(repo: str, path: str, revision: str, target_dir: str, progress: Optional[Progress] = stderr_progress) -> str:
    commit = revision if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision) else resolve_revision(repo, revision)
    f = file_info(repo, commit, path)
    dest = os.path.join(target_dir, os.path.basename(path))
    state = download(repo, commit, f, dest, progress)
    print(f"{dest}: {'already present, ' if state == 'present' else ''}verified ({f.algorithm} {f.digest[:16]}..., {f.size} bytes, {repo}@{commit[:12]})",
          file=sys.stderr)
    return dest


def pull_snapshot(s: Snapshot, target_dir: str, progress: Optional[Progress] = stderr_progress) -> str:
    """Every file of a repository at a commit (minus exclude_prefixes) into <models>/sources/<owner>/<repo>, each verified."""
    dest_dir = os.path.join(target_dir, SOURCES_SUBDIR, *s.repo.split("/"))
    files = [f for f in list_files(s.repo, s.revision) if not f.path.startswith(s.exclude_prefixes)]
    if not files:
        raise PullError(f"{s.repo}@{s.revision[:12]}: the repository lists no files")
    print(f"{s.repo}@{s.revision[:12]}: {len(files)} files, {sum(f.size for f in files) / 1e6:,.1f} MB -> {dest_dir}", file=sys.stderr)
    for f in files:
        dest = os.path.join(dest_dir, *f.path.split("/"))
        if not os.path.abspath(dest).startswith(os.path.abspath(dest_dir) + os.sep):
            raise PullError(f"{s.repo}: refusing a file path that leaves the target directory: {f.path!r}")
        download(s.repo, s.revision, f, dest, progress)
    return dest_dir


# ------------------------------------------------------------------------------------------------ local conversion (smolvlm, smolvla, whisper-*)

def source_dir() -> Optional[str]:
    """The dooms.cpp checkout this package runs from (or $DOOMS_SOURCE_DIR), if it has the converters."""
    d = os.environ.get("DOOMS_SOURCE_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    return d if os.path.isfile(os.path.join(d, "tools", "convert", "convert_smolvlm.py")) else None


def _how_to_convert(name: str, problem: str) -> PullError:
    return PullError(
        f"cannot pull {name}: {problem}.\n"
        f"{name} runs from Core ML packages and GGUF files that are converted locally from the original weights (the converted files are never redistributed, "
        "docs/adr/0002). To convert:\n"
        "  1. git clone https://github.com/jay7828/dooms && cd dooms && build it (README: cmake -B build ... && cmake --build build)\n"
        + ("  2. python3.11 -m venv .venv-kokoro && .venv-kokoro/bin/pip install kokoro==0.9.4 'misaki[en]' torch==2.8.0 coremltools==9.0 gguf\n"
           if name == "kokoro" else
           "  2. python3 -m venv .venv && .venv/bin/pip install torch coremltools transformers gguf safetensors huggingface_hub numpy pillow\n")
        + ("  3. a second environment with LeRobot for the reference observation: python3 -m venv .venv-lerobot && .venv-lerobot/bin/pip install 'lerobot[smolvla]'\n"
           if name == "smolvla" else "")
        + f"  then, in the checkout: .venv/bin/python -m dooms pull {name}   (run from python/, or set PYTHONPATH=python; DOOMS_SOURCE_DIR, DOOMS_CONVERT_PYTHON, "
        "DOOMS_LEROBOT_PYTHON, DOOMS_KOKORO_PYTHON and DOOMS_LLAMA_CPP_DIR override the defaults)")


def _python_with(env_var: str, default: str, modules: Sequence[str], name: str, purpose: str) -> str:
    py = os.environ.get(env_var) or default
    if not os.path.isfile(py):
        raise _how_to_convert(name, f"no Python for {purpose} at {py} (set {env_var})")
    probe = subprocess.run([py, "-c", "import " + ", ".join(modules)], capture_output=True, text=True)
    if probe.returncode != 0:
        last = (probe.stderr.strip().splitlines() or ["?"])[-1]
        raise _how_to_convert(name, f"{py} cannot import {', '.join(modules)} ({last}); set {env_var} to a Python that can")
    return py


def _run(cmd: List[str], cwd: str) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise PullError(f"the converter failed (exit {r.returncode}): {' '.join(cmd)}; its output is above. Nothing was moved into place.")


def _llama_cpp_dir(root: str, name: str) -> str:
    d = os.environ.get("DOOMS_LLAMA_CPP_DIR") or os.path.join(root, "build", "_deps", "llama_cpp-src")
    if not os.path.isfile(os.path.join(d, "convert_hf_to_gguf.py")):
        raise _how_to_convert(name, f"llama.cpp's convert_hf_to_gguf.py is not in {d} (build the checkout once, which fetches llama.cpp, or set DOOMS_LLAMA_CPP_DIR)")
    return d


def _check_loads(name: str, out_dir: str) -> None:
    import numpy as np  # noqa: PLC0415

    import dooms  # noqa: PLC0415

    if name == "kokoro":
        with dooms.TTS(out_dir) as tts:
            audio = tts.synthesize("h\u0259l\u02c8O", tts.voices[0])
            seconds, voices = len(audio) / tts.sample_rate, len(tts.voices)
        print(f"{out_dir}: loads and speaks ({seconds:.2f} s for a word, {voices} voices)", file=sys.stderr)
    elif name in WHISPER:
        with dooms.Whisper(out_dir) as stt:
            text = stt.transcribe(np.zeros(stt.sample_rate, dtype=np.float32))
        print(f"{out_dir}: loads and transcribes one second of silence ({text!r})", file=sys.stderr)
    elif name == "smolvlm":
        with dooms.VLM.from_dir(out_dir) as vlm:
            text = vlm.generate(np.full((64, 64, 3), 128, dtype=np.uint8), "What colour is this image?", max_new_tokens=4)
        print(f"{out_dir}: loads and answers ({text!r})", file=sys.stderr)
    else:
        with dooms.VLA.from_dir(out_dir) as vla:
            info = (vla.n_cameras, vla.chunk_size, vla.action_dim)
        print(f"{out_dir}: loads ({info[0]} cameras, chunks of {info[1]} x {info[2]})", file=sys.stderr)


def pull_converted(name: str, target_dir: str, progress: Optional[Progress] = stderr_progress) -> str:
    root = source_dir()
    if root is None:
        raise _how_to_convert(name, "this installation has no converters (tools/convert is part of the source checkout, not of the wheel)")
    if name == "kokoro":
        convert_py = _python_with("DOOMS_KOKORO_PYTHON", os.path.join(root, ".venv-kokoro", "bin", "python"), KOKORO_MODULES, name, "the Kokoro converter")
    else:
        convert_py = _python_with("DOOMS_CONVERT_PYTHON", os.path.join(root, ".venv", "bin", "python"), CONVERT_MODULES, name, "the converters")
    llama_cpp = None if (name in WHISPER or name == "kokoro") else _llama_cpp_dir(root, name)
    lerobot_py = None
    if name == "smolvla":
        lerobot_py = _python_with("DOOMS_LEROBOT_PYTHON", os.path.join(root, ".venv-lerobot", "bin", "python"), LEROBOT_MODULES, name, "the LeRobot reference")
    out = os.path.join(target_dir, name)
    tmp = out + ".converting"
    os.makedirs(tmp, exist_ok=True)
    conv = lambda script: os.path.join(root, "tools", "convert", script)  # noqa: E731
    if name == "kokoro":
        model = pull_snapshot(KOKORO, target_dir, progress)
        _run([convert_py, conv("convert_kokoro.py"), "--model", model, "--out-dir", tmp], root)
    elif name in WHISPER:
        snapshot, max_rel_l2 = WHISPER[name]
        model = pull_snapshot(snapshot, target_dir, progress)
        _run([convert_py, conv("convert_whisper.py"), "--model", model, "--out-dir", tmp, "--encoder-max-rel-l2", str(max_rel_l2)], root)
    elif name == "smolvlm":
        vlm = pull_snapshot(SMOLVLM, target_dir, progress)
        _run([convert_py, conv("convert_smolvlm.py"), "--model", vlm, "--out-dir", tmp, "--llama-cpp", llama_cpp], root)
    else:
        vlm = pull_snapshot(SMOLVLA_VLM, target_dir, progress)
        policy = pull_snapshot(SMOLVLA_POLICY, target_dir, progress)
        reference = os.path.join(tmp, "reference.npz")
        _run([lerobot_py, os.path.join(root, "tools", "parity", "smolvla_reference.py"), "--out", reference, "--model", policy], root)
        _run([convert_py, conv("convert_smolvla_vision.py"), "--vlm", vlm, "--policy", os.path.join(policy, "model.safetensors"), "--out-dir", tmp, "--batches", "1"], root)
        _run([convert_py, conv("convert_smolvla_coreml.py"), "--vlm", vlm, "--policy-dir", policy, "--reference", reference, "--out-dir", tmp], root)
        _run([convert_py, conv("convert_smolvla_host.py"), "--vlm", vlm, "--policy-dir", policy, "--out", os.path.join(tmp, "host.gguf")], root)
        _run([convert_py, os.path.join(llama_cpp, "convert_hf_to_gguf.py"), "--vocab-only", vlm, "--outfile", os.path.join(tmp, "vocab.gguf")], root)
    _check_loads(name, tmp)
    if os.path.exists(out):
        backup = out + ".previous"
        if os.path.exists(backup):
            raise PullError(f"{backup} is in the way; delete it (or {out}) and run the command again")
        os.replace(out, backup)
        os.replace(tmp, out)
        print(f"the previous {out} was kept as {backup}", file=sys.stderr)
    else:
        os.replace(tmp, out)
    opener = "dooms.TTS" if name == "kokoro" else "dooms.Whisper" if name in WHISPER else f"dooms.{'VLM' if name == 'smolvlm' else 'VLA'}.from_dir"
    print(f"{out}: converted and verified (open it with {opener})", file=sys.stderr)
    return out


def _extract(archive: str, name: str, dest: str) -> None:
    """Unpacks `archive` (one top-level directory `name`) into `dest`. Only regular files and directories inside `name/` are accepted: no links, devices,
    absolute paths or `..` (the archive is verified by hash first; this guards the extraction itself)."""
    import tarfile  # noqa: PLC0415
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if m.name.startswith("/") or ".." in parts or parts[0] != name or not (m.isfile() or m.isdir()):
                raise PullError(f"{archive}: unexpected entry {m.name!r}; nothing was extracted into place")
            rel = "/".join(parts[1:])
            if not rel:
                continue
            target = os.path.realpath(os.path.join(dest, rel))
            if not target.startswith(root + os.sep):
                raise PullError(f"{archive}: entry {m.name!r} leaves the target directory")
            if m.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tar.extractfile(m)
            with open(target, "wb") as out:
                while True:
                    b = src.read(CHUNK_BYTES)
                    if not b:
                        break
                    out.write(b)


def pull_prebuilt(name: str, target_dir: str, progress: Optional[Progress] = stderr_progress, keep_archive: bool = False) -> str:
    """Downloads the converted archive of `name` (pinned size and SHA-256), unpacks it, checks that it loads, moves it into place."""
    p = PREBUILT[name]
    archive = os.path.join(target_dir, "downloads", p.file)
    url = PREBUILT_URL.format(release=PREBUILT_RELEASE, file=p.file)
    print(f"{name}: {p.note}, converted ({p.size / 1e6:,.1f} MB) -> {os.path.join(target_dir, name)}", file=sys.stderr)
    download(f"jay7828/dooms release {PREBUILT_RELEASE}", PREBUILT_RELEASE, RemoteFile(p.file, p.size, "sha256", p.sha256), archive, progress, url=url)
    out = os.path.join(target_dir, name)
    tmp = out + ".extracting"
    if os.path.exists(tmp):
        import shutil  # noqa: PLC0415
        shutil.rmtree(tmp)
    _extract(archive, name, tmp)
    _check_loads(name, tmp)
    if os.path.exists(out):
        backup = out + ".previous"
        if os.path.exists(backup):
            raise PullError(f"{backup} is in the way; delete it (or {out}) and run the command again")
        os.replace(out, backup)
        print(f"the previous {out} was kept as {backup}", file=sys.stderr)
    os.replace(tmp, out)
    if not keep_archive:
        os.remove(archive)
    print(f"{out}: ready (open it with {'dooms.TTS' if name == 'kokoro' else 'dooms.Whisper'})", file=sys.stderr)
    return out


def pull(name: str, target_dir: Optional[str] = None, progress: Optional[Progress] = stderr_progress, convert: bool = False) -> str:
    """Downloads `name`: a GGUF alias, a prebuilt archive (PREBUILT), or, for the CONVERTED models (all of them with convert=True), the original weights
    converted locally. Returns the path of the model file or directory."""
    d = models_dir(target_dir)
    if name in PREBUILT and not convert:
        return pull_prebuilt(name, d, progress)
    if name.startswith("hf:"):
        repo, path, revision = parse_hf(name)
        return pull_gguf(repo, path, revision, d, progress)
    if name in ALIASES:
        a = ALIASES[name]
        return pull_gguf(a.repo, a.file, a.revision, d, progress)
    if name in CONVERTED:
        return pull_converted(name, d, progress)
    raise PullError(f"unknown model '{name}'. Known: {', '.join(sorted(ALIASES) + list(CONVERTED))}, or hf:OWNER/REPO/FILE.gguf[@REVISION]")


def list_models(target_dir: Optional[str] = None) -> List[Tuple[str, str, int]]:
    """(kind, path, bytes) of what is in the models directory: GGUF files, converted model directories, original checkpoints, unfinished downloads."""
    d = models_dir(target_dir)
    if not os.path.isdir(d):
        return []
    out = []

    def size(p):
        if os.path.isfile(p):
            return os.path.getsize(p)
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)

    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        if e.endswith(".gguf") and os.path.isfile(p):
            out.append(("gguf", p, size(p)))
        elif e.endswith(PART_SUFFIX):
            out.append(("partial", p, size(p)))
        elif e in CONVERTED and os.path.isdir(p):
            out.append((e, p, size(p)))
        elif e == SOURCES_SUBDIR and os.path.isdir(p):
            for owner in sorted(os.listdir(p)):
                for repo in sorted(os.listdir(os.path.join(p, owner))) if os.path.isdir(os.path.join(p, owner)) else []:
                    out.append(("source", os.path.join(p, owner, repo), size(os.path.join(p, owner, repo))))
    return out
