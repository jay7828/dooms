#!/usr/bin/env python3
"""PyTorch (transformers, fp32 CPU) reference for the SmolVLM pipeline on photos: everything the engine reproduces, stage by stage.

    .venv/bin/python tools/parity/smolvlm_reference.py --out models/smolvlm/reference.gguf [--tokens 24]
Per case k (float32 tensors; ids as float32): `case<k>.photo` [H,W,3] RGB 0..255, `case<k>.pixel_values` [3,512,512] (the processor's output),
`case<k>.image_embeddings` [64,576], `case<k>.input_ids` [n] (the complete prompt), `case<k>.logits` [vocab] at the last prompt position,
`case<k>.generated` [<= tokens] the greedy continuation (including the end-of-utterance token when the model stopped); metadata `case<k>.prompt` (the user text). Exit codes: 0 ok, 1 error, 2 bad arguments.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "convert"))

import gguf  # noqa: E402
from PIL import Image  # noqa: E402

CASES = [("test1.jpg", "Describe this image in one sentence."), ("test4.jpg", "What is the person doing?"), ("test6.jpg", "How many hands are visible?"),
         ("test2.png", "What colors do you see?")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "smolvlm", "256M-Instruct"))
    ap.add_argument("--images", default=os.path.join(ROOT, "tests", "data"))
    ap.add_argument("--tokens", type=int, default=24)
    args = ap.parse_args(argv)
    if args.tokens < 1:
        ap.error("--tokens must be >= 1")
    from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: PLC0415
    processor = AutoProcessor.from_pretrained(args.model, do_image_splitting=False)
    model = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    w = gguf.GGUFWriter(args.out, "vlm_reference")
    for k, (name, prompt) in enumerate(CASES):
        path = os.path.join(args.images, name)
        if not os.path.exists(path):
            print(f"error: missing {path}", file=sys.stderr)
            return 1
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")
        with torch.no_grad():
            feats = model.model.get_image_features(pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs["pixel_attention_mask"])
            logits = model(**inputs).logits[0, -1]
            out = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False)  # stops at <end_of_utterance> like a user would see
        generated = out[0, inputs["input_ids"].shape[1]:]
        p = f"case{k}."
        w.add_tensor(p + "photo", np.asarray(image, dtype=np.float32))
        w.add_tensor(p + "pixel_values", inputs["pixel_values"][0, 0].numpy().astype(np.float32))
        w.add_tensor(p + "image_embeddings", feats.reshape(-1, feats.shape[-1]).numpy().astype(np.float32))
        w.add_tensor(p + "input_ids", inputs["input_ids"][0].numpy().astype(np.float32))
        w.add_tensor(p + "logits", logits.numpy().astype(np.float32))
        w.add_tensor(p + "generated", generated.numpy().astype(np.float32))
        w.add_string(p + "prompt", prompt)
        print(f"{name}: {inputs['input_ids'].shape[1]} prompt tokens -> {processor.tokenizer.decode(generated, skip_special_tokens=True)!r}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out} ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
