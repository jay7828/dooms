"""MLX (mlx-vlm) baseline for SmolVLM-256M on one photo + prompt: the Apple-native array framework, measured like the PyTorch baseline (vlm_baseline.py).

    /tmp/mlx_env/bin/python bench/python/mlx_vlm_baseline.py [--model models/smolvlm/256M-Instruct] [--image PHOTO] [--dtypes bf16,fp16,fp32] [--splitting off,on] [--tokens 32] [--iters 10]
Needs `mlx` and `mlx-vlm` (uv venv --python 3.11 && uv pip install mlx mlx-vlm pillow). Inputs are prepared once (the HF processor's work is not timed, like the PyTorch
baseline); then, per (dtype, image splitting) case, batch 1, greedy, eager:
  vision   model.get_input_embeddings(...) evaluated (vision tower + connector + embedding splice)
  ttft     stream_generate() start -> first token (vision + prefill + first decode step)
  decode   median interval between the following tokens
Image splitting `off` gives one 512x512 tile (64 image tokens) like the engine and the transformers reference; `on` is mlx-vlm's / the model's default (the photo is
split into tiles: 13 for a 640x480 photo, ~830 image tokens), which is also what llama.cpp's mtmd does. A case that raises is reported as failed.
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "smolvlm", "256M-Instruct"))
    ap.add_argument("--image", default=os.path.join(ROOT, "tests", "data", "test1.jpg"))
    ap.add_argument("--prompt", default="Describe this image in one sentence.")
    ap.add_argument("--dtypes", default="bf16,fp16,fp32")
    ap.add_argument("--splitting", default="off,on")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate.dispatch import _prepare_generation_inputs, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"mlx {mx.__version__}; device {mx.default_device()}")
    print(f"{'case':<30}{'vision ms':>11}{'ttft ms':>10}{'ms/token':>10}{'tok/s':>8}{'prompt':>8}")
    dts = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    for dtn in a.dtypes.split(","):
        for split in a.splitting.split(","):
            label = f"mlx {dtn} splitting {split}"
            try:
                model, processor = load(a.model)
                from mlx.utils import tree_map
                model.update(tree_map(lambda p: p.astype(dts[dtn]) if p.dtype in (mx.bfloat16, mx.float16, mx.float32) else p, model.parameters()))
                mx.eval(model.parameters())
                ip = getattr(processor, "image_processor", None)
                if ip is not None and hasattr(ip, "do_image_splitting"):
                    ip.do_image_splitting = split == "on"
                config = load_config(a.model)
                prompt = apply_chat_template(processor, config, a.prompt, num_images=1)
                input_ids, pixel_values, mask, data_kwargs = _prepare_generation_inputs(model, processor, prompt, [a.image], None, None, {})
                n_prompt = int(input_ids.shape[-1])

                def vision():
                    out = model.get_input_embeddings(input_ids, pixel_values, **data_kwargs)
                    mx.eval(out.inputs_embeds)

                def generate():
                    t0 = time.perf_counter()
                    stamps = []
                    text = ""
                    for r in stream_generate(model, processor, prompt, input_ids=input_ids, pixel_values=pixel_values, mask=mask, max_tokens=a.tokens, **data_kwargs):
                        stamps.append(time.perf_counter())
                        text += r.text
                    return t0, stamps, text

                for _ in range(a.warmup):
                    vision()
                    generate()
                v, ttft, dec = [], [], []
                text = ""
                for _ in range(a.iters):
                    t = time.perf_counter()
                    vision()
                    v.append((time.perf_counter() - t) * 1e3)
                    t0, stamps, text = generate()
                    ttft.append((stamps[0] - t0) * 1e3)
                    if len(stamps) > 2:
                        dec.append(statistics.median((stamps[i] - stamps[i - 1]) * 1e3 for i in range(2, len(stamps))))
                per = statistics.median(dec)
                print(f"{label:<30}{statistics.median(v):>11.1f}{statistics.median(ttft):>10.1f}{per:>10.2f}{1000 / per:>8.1f}{n_prompt:>8}")
                print(f"    -> {text.strip()[:110]!r}")
            except Exception as e:  # noqa: BLE001 - reported, not skipped
                print(f"{label:<30} failed: {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
