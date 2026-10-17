#!/usr/bin/env python3
"""Builds the speech test set the Whisper benchmarks and parity tests use, from hf-internal-testing/librispeech_asr_dummy (73 LibriSpeech validation clips
with their reference transcripts): one 16 kHz mono 16-bit WAV per clip, a `long.wav` that concatenates every clip (a few minutes, for real-time-factor
measurements), and `transcripts.tsv` (file name, reference text). Decoding uses macOS's own afconvert, so no engine under test decodes its own inputs.

    .venv-lerobot/bin/python bench/python/speech_testset.py [--src models/whisper/librispeech_dummy] [--out models/whisper/data]

Needs pyarrow (the parquet file). Exit codes: 0 ok, 1 failed, 2 usage.
"""
import argparse
import glob
import os
import struct
import subprocess
import sys
import tempfile
import wave

SAMPLE_RATE = 16000  # Whisper's input rate


def read_pcm16(path: str) -> bytes:
    """The sample bytes of a 16-bit mono WAV, whatever its header variant (afconvert writes WAVE_FORMAT_EXTENSIBLE, which Python's wave cannot read)."""
    data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    i, fmt, pcm = 12, None, None
    while i + 8 <= len(data):
        cid, size = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        body = data[i + 8:i + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            pcm = body
        i += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"{path} has no fmt or data chunk")
    _, channels, rate, _, _, bits = fmt
    if channels != 1 or rate != SAMPLE_RATE or bits != 16:
        raise ValueError(f"{path} is {channels} ch, {rate} Hz, {bits} bit; expected 1 ch, {SAMPLE_RATE} Hz, 16 bit")
    return pcm


def write_wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def to_wav(src_bytes: bytes, dst: str) -> bytes:
    """Decodes with afconvert to 16 kHz mono 16-bit, rewrites dst as a plain PCM WAV, returns the samples."""
    with tempfile.TemporaryDirectory() as d:
        src, tmp = os.path.join(d, "in.flac"), os.path.join(d, "out.wav")
        open(src, "wb").write(src_bytes)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", src, tmp], check=True, capture_output=True)
        pcm = read_pcm16(tmp)
    write_wav(dst, pcm)
    return pcm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="models/whisper/librispeech_dummy")
    ap.add_argument("--out", default="models/whisper/data")
    a = ap.parse_args(argv)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("error: needs pyarrow (use .venv-lerobot/bin/python)", file=sys.stderr)
        return 2
    files = sorted(glob.glob(os.path.join(a.src, "clean", "*.parquet")))
    if not files:
        print(f"error: no parquet file under {a.src}/clean", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    rows, pcms = [], []
    for path in files:
        t = pq.read_table(path).to_pylist()
        for r in t:
            name = r["id"] + ".wav"
            pcms.append(to_wav(r["audio"]["bytes"], os.path.join(a.out, name)))
            rows.append((name, r["text"]))
    write_wav(os.path.join(a.out, "long.wav"), b"".join(pcms))   # every clip back to back
    with open(os.path.join(a.out, "transcripts.tsv"), "w") as f:
        for name, text in rows:
            f.write(f"{name}\t{text}\n")
    with wave.open(os.path.join(a.out, "long.wav"), "rb") as w:
        seconds = w.getnframes() / SAMPLE_RATE
    print(f"{len(rows)} clips + long.wav ({seconds:.1f} s) in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
