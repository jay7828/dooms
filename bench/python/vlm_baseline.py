"""PyTorch (transformers) baseline for a vision-language model on one image + prompt: the numbers the engine is compared with.

    .venv/bin/python bench/python/vlm_baseline.py [--model models/smolvlm/256M-Instruct] [--image PATH] [--devices cpu,mps] [--dtypes fp32,fp16]
Measured per (device, dtype), batch 1, greedy decoding, no image splitting (one image tile -> 64 image tokens for SmolVLM):
  vision      pixel values -> image embeddings in the language model's space (vision tower + connector)
  ttft        time to the first generated token (the image encoder, the prefill over image + prompt tokens, one decode step)
  decode      steady-state seconds per additional token = (T(N) - T(1)) / (N - 1)
and the generated text, so the engine's output can be compared token by token.
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "smolvlm", "256M-Instruct"))
    ap.add_argument("--image", default=os.path.join(ROOT, "tests", "data", "test1.jpg"))
    ap.add_argument("--prompt", default="Describe this image in one sentence.")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    if args.tokens < 2 or args.iters < 1 or args.warmup < 0:
        ap.error("--tokens must be >= 2, --iters >= 1, --warmup >= 0")
    from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: PLC0415
    processor = AutoProcessor.from_pretrained(args.model, do_image_splitting=False)
    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    base_inputs = processor(text=text, images=[image], return_tensors="pt")
    print(f"torch {torch.__version__}; prompt tokens {base_inputs['input_ids'].shape[1]}, image tokens {(base_inputs['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<image>')).sum().item()}")
    print(f"{'case':<24}{'vision ms':>11}{'ttft ms':>10}{'ms/token':>10}{'tok/s':>8}")
    dts = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    for dev in args.devices.split(","):
        if dev == "mps" and not torch.backends.mps.is_available():
            print(f"{'vlm ' + dev:<24} unavailable")
            continue
        for dtn in args.dtypes.split(","):
            label = f"vlm eager {dev} {dtn}"
            try:
                model = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=dts[dtn]).to(dev).eval()
                inputs = {k: (v.to(dev, dts[dtn]) if v.dtype.is_floating_point else v.to(dev)) for k, v in base_inputs.items()}

                def gen(n):
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False)
                    sync(dev)
                    return out

                def vision():
                    with torch.no_grad():
                        model.model.get_image_features(pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs.get("pixel_attention_mask"))
                    sync(dev)

                for _ in range(args.warmup):
                    gen(args.tokens); vision()
                tv, t1, tn = [], [], []
                for _ in range(args.iters):
                    t0 = time.perf_counter(); vision(); tv.append((time.perf_counter() - t0) * 1e3)
                    t0 = time.perf_counter(); gen(1); t1.append((time.perf_counter() - t0) * 1e3)
                    t0 = time.perf_counter(); out = gen(args.tokens); tn.append((time.perf_counter() - t0) * 1e3)
                v, a, b = statistics.median(tv), statistics.median(t1), statistics.median(tn)
                per_token = (b - a) / (args.tokens - 1)
                reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            except Exception as e:  # noqa: BLE001
                print(f"{label:<24} failed: {type(e).__name__}: {str(e)[:110]}")
                continue
            print(f"{label:<24}{v:>11.1f}{a:>10.1f}{per_token:>10.2f}{1000 / per_token:>8.1f}")
            print(f"    -> {reply!r}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
