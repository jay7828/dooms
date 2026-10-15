#!/usr/bin/env python3
"""Convert SmolVLM (Idefics3: SigLIP + connector + SmolLM2) for the dooms.cpp engine (Apache-2.0 model; derived files stay local anyway).

    .venv/bin/python tools/convert/convert_smolvlm.py --model models/smolvlm/256M-Instruct --out-dir models/smolvlm [--no-text]
Writes into --out-dir: vision-fp16.mlpackage (image -> 64 embeddings, Core ML), vlm-config.gguf (prompt layout + preprocessing), and text-f16.gguf (the language
model, via llama.cpp's own converter). Exit codes: 0 ok, 1 failed (partial outputs removed), 2 bad arguments.
"""
import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

import smolvlm_vision as sv  # noqa: E402
import convert_common as wb  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="Core ML precision of the vision tower")
    ap.add_argument("--max-rel-l2", type=float, default=2e-2, help="refuse to write the vision package if it differs from PyTorch on the GPU units by more")
    ap.add_argument("--no-fuse-qkv", action="store_true", help="keep separate q/k/v projections in the text GGUF (default: one fused matrix per layer, about 6%% faster decoding)")
    ap.add_argument("--no-text", action="store_true", help="skip the text GGUF (run llama.cpp's convert_hf_to_gguf.py yourself)")
    ap.add_argument("--llama-cpp", default=os.path.join(ROOT, "build", "_deps", "llama_cpp-src"), help="llama.cpp checkout containing convert_hf_to_gguf.py")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.model):
        print(f"error: model directory not found: {args.model}", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    text_path = os.path.join(args.out_dir, "text-f16.gguf")
    try:
        from transformers import AutoProcessor  # noqa: PLC0415
        model = sv.load_model(args.model)
        processor = AutoProcessor.from_pretrained(args.model, do_image_splitting=False)
        layout = sv.prompt_layout(processor)
        if not args.no_text:
            script = os.path.join(args.llama_cpp, "convert_hf_to_gguf.py")
            if not os.path.exists(script):
                print(f"error: llama.cpp converter not found: {script}", file=sys.stderr)
                return 1
            run = subprocess.run([sys.executable, script, args.model, "--outfile", text_path + ".tmp", "--outtype", "f16"], capture_output=True, text=True)
            if run.returncode != 0:
                print(f"error: llama.cpp's converter failed:\n{run.stderr[-800:]}", file=sys.stderr)
                return 1
            if args.no_fuse_qkv:
                os.replace(text_path + ".tmp", text_path)
            else:
                from fuse_qkv_gguf import fuse_qkv  # noqa: PLC0415
                n = fuse_qkv(text_path + ".tmp", text_path)
                os.remove(text_path + ".tmp")
                print(f"text model: q/k/v fused in {n} layers")
        digest = hashlib.sha256(open(text_path, "rb").read()).hexdigest() if os.path.exists(text_path) else None
        sv.write_config(os.path.join(args.out_dir, "vlm-config.gguf"), model, processor, layout, digest)
        err = sv.convert(model, os.path.join(args.out_dir, f"vision-{args.precision}.mlpackage"), args.precision, args.max_rel_l2, "gpu")
    except wb.ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.out_dir}: vision-{args.precision}.mlpackage (GPU-units error vs fp32 PyTorch {err:.1e}), vlm-config.gguf" + ("" if args.no_text else ", text-f16.gguf")
          + f"; prompt layout: {layout['n_image_tokens']} image tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
