#!/usr/bin/env python3
"""Archives of the converted, permissively licensed models for `dooms pull` (docs/adr/0002 addendum 2026-10-18: Whisper is MIT, Kokoro-82M Apache-2.0; nothing
else may be packaged). Each archive holds the files the engine opens, the model's licence and a README with the provenance (source repository and commit, the
converter, the dooms.cpp version); no test references, no Core ML compile caches.

    python3 scripts/package-models.py --name whisper-large-v3-turbo --src models/whisper/whisper-large-v3-turbo/dooms --out dist/models
Prints `name file size sha256` for python/dooms/pull.py's PREBUILT table. Deterministic (sorted entries, fixed times and owners). Exit codes: 0 ok, 1 failed, 2 usage.
"""
import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MIT_OPENAI = """MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

MODELS = {   # name -> (files the engine opens, licence text or a file under ROOT, source, converter)
    "whisper-tiny": (["encoder-fp16.mlpackage", "decoder.gguf"], MIT_OPENAI, "openai/whisper-tiny @ 169d4a4341b33bc18d8881c4b69c2e104e1cc0af (MIT)",
                     "tools/convert/convert_whisper.py"),
    "whisper-large-v3-turbo": (["encoder-fp16.mlpackage", "decoder.gguf"], MIT_OPENAI,
                               "openai/whisper-large-v3-turbo @ 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 (MIT)", "tools/convert/convert_whisper.py"),
    "kokoro": (["prosody-fp32.mlpackage", "f0n-fp32.mlpackage", "kokoro.gguf"], "@scripts/public/LICENSE",
               "hexgrad/Kokoro-82M @ f3ff3571791e39611d31c381e3a41a3af07b4987 (Apache-2.0; StyleTTS 2 / iSTFTNet architecture), all voices",
               "tools/convert/convert_kokoro.py"),
}


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = len(data), 0, 0o644, 0, 0, "", ""
    tar.addfile(info, io.BytesIO(data))


def add_tree(tar: tarfile.TarFile, src: str, arc: str) -> None:
    if os.path.isdir(src):
        info = tarfile.TarInfo(arc)
        info.type, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = tarfile.DIRTYPE, 0, 0o755, 0, 0, "", ""
        tar.addfile(info)
        for entry in sorted(os.listdir(src)):
            if entry.startswith("."):
                continue
            add_tree(tar, os.path.join(src, entry), f"{arc}/{entry}")
    elif os.path.isfile(src) and not os.path.islink(src):
        with open(src, "rb") as f:
            add_bytes(tar, arc, f.read())
    else:
        raise SystemExit(f"error: {src} is neither a regular file nor a directory")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, choices=sorted(MODELS))
    ap.add_argument("--src", required=True, help="the converted model directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files, licence, source, converter = MODELS[a.name]
    for f in files:
        if not os.path.exists(os.path.join(a.src, f)):
            print(f"error: {a.src} has no {f}", file=sys.stderr)
            return 1
    version = open(os.path.join(ROOT, "VERSION"), encoding="utf-8").read().strip()
    if licence.startswith("@"):
        licence = open(os.path.join(ROOT, licence[1:]), encoding="utf-8").read()
    readme = (f"# {a.name} for dooms.cpp\n\nConverted from {source} with {converter} (dooms.cpp {version}). Open the directory with dooms.Whisper / dooms.TTS, "
              f"or get it with `dooms pull {a.name}`.\nThe model's licence is in LICENSE; the conversion changes the file format, not the weights (Core ML packages "
              f"in fp16 or fp32, GGUF with f16 matrices).\n")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"dooms-{a.name}.tar.gz")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        add_bytes(tar, f"{a.name}/LICENSE", licence.encode())
        add_bytes(tar, f"{a.name}/README.md", readme.encode())
        for f in files:
            add_tree(tar, os.path.join(a.src, f), f"{a.name}/{f}")
    with open(path + ".tmp", "wb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=6) as gz:
            gz.write(raw.getvalue())
    os.replace(path + ".tmp", path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    print(a.name, os.path.basename(path), os.path.getsize(path), h.hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
