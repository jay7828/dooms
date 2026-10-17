#!/usr/bin/env python3
"""Convert Whisper (Hugging Face layout, e.g. openai/whisper-large-v3-turbo) for the dooms.cpp engine (MIT-licensed model).

    .venv/bin/python tools/convert/convert_whisper.py --model models/whisper/whisper-large-v3-turbo [--out-dir DIR] [--no-encoder]
Writes into --out-dir (default: <model>/dooms): encoder-fp16.mlpackage (log-mel features -> encoder states, Core ML, for the Neural Engine) and decoder.gguf
(decoder, vocabulary, special tokens, suppression lists, mel filter bank; docs/formats/whisper-gguf.md). Exit codes: 0 ok, 1 failed (nothing left half-written),
2 bad arguments.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert_common as cc  # noqa: E402
import whisper_model as wm  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face snapshot directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-encoder", action="store_true", help="only the decoder GGUF")
    ap.add_argument("--encoder-check-units", default="gpu", choices=["cpu", "gpu", "ne"], help="compute units the encoder's accuracy check runs on")
    ap.add_argument("--encoder-max-rel-l2", type=float, default=2e-2,
                    help="largest relative L2 error of the fp16 encoder vs PyTorch fp32 on random features (whisper-large-v3-turbo's 32 layers: 2.3e-2 on gpu; "
                         "the real-speech criterion is tests/unit/test_whisper.cpp)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.model):
        print(f"error: {a.model} is not a directory", file=sys.stderr)
        return 2
    out = a.out_dir or os.path.join(a.model, "dooms")
    os.makedirs(out, exist_ok=True)
    try:
        t0 = time.time()
        model, processor, gen = wm.load(a.model)
        wm.check_supported(model, processor)
        wm.write_decoder(model, processor, gen, os.path.join(out, "decoder.gguf"))
        print(f"decoder.gguf written ({time.time() - t0:.0f} s)")
        if not a.no_encoder:
            t0 = time.time()
            err = wm.convert_encoder(model, os.path.join(out, "encoder-fp16.mlpackage"), compute_units=a.encoder_check_units, max_rel_l2=a.encoder_max_rel_l2)
            print(f"encoder-fp16.mlpackage written ({time.time() - t0:.0f} s), relative L2 vs PyTorch fp32 on {a.encoder_check_units}: {err:.2e}")
    except cc.ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
