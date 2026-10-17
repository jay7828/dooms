#!/usr/bin/env python3
"""Reference outputs of Whisper (transformers, fp32 CPU) for the engine's parity tests: per clip the 16 kHz samples, the log-mel features, the encoder states, the
decoder prompt, the logits after the prompt, transformers' greedy tokens (English, transcribe, no timestamps) and the text.

    .venv/bin/python tools/parity/whisper_reference.py --model models/whisper/whisper-tiny [--clips 4] [--data models/whisper/data] [--out <model>/dooms/reference.gguf]

Exit codes: 0 ok, 1 failed, 2 bad arguments.
"""
import argparse
import os
import sys
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convert"))

import whisper_model as wm  # noqa: E402


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def main(argv=None) -> int:
    import gguf  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="models/whisper/data")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.clips < 1:
        print("error: --clips must be >= 1", file=sys.stderr)
        return 2
    model, processor, gen = wm.load(a.model)
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(a.data, "transcripts.tsv"))][: a.clips]
    out = a.out or os.path.join(a.model, "dooms", "reference.gguf")
    w = gguf.GGUFWriter(out + ".tmp", "whisper_reference")
    w.add_uint32("whisper_reference.n_cases", len(rows))
    prompt = [gen.decoder_start_token_id] + [int(t) for t in processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
                                              for t in [t[1]]]
    for k, (name, text) in enumerate(rows):
        audio = read_wav(os.path.join(a.data, name))
        feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features      # [1, n_mels, 3000]
        with torch.no_grad():
            enc = model.model.encoder(feats).last_hidden_state                                  # [1, 1500, d]
            logits = model(input_features=feats, decoder_input_ids=torch.tensor([prompt])).logits[0, -1]
            ids = model.generate(feats, language="en", task="transcribe", return_timestamps=False, num_beams=1, do_sample=False, max_new_tokens=200)[0].tolist()
        gen_ids = ids[len(prompt):] if ids[: len(prompt)] == prompt else ids
        hyp = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        p = f"case{k}."
        w.add_string(p + "file", name)
        w.add_string(p + "reference_text", text)
        w.add_string(p + "text", hyp)
        w.add_tensor(p + "audio", audio)
        w.add_tensor(p + "mel", feats[0].numpy().astype(np.float32))
        w.add_tensor(p + "encoder_out", enc[0].numpy().astype(np.float32))
        w.add_tensor(p + "prompt", np.array(prompt, dtype=np.float32))
        w.add_tensor(p + "logits", logits.numpy().astype(np.float32))
        w.add_tensor(p + "generated", np.array(gen_ids, dtype=np.float32))
        print(f"case {k}: {name}: {hyp!r}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(out + ".tmp", out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
