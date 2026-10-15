"""Throughput baselines for a text LLM on the same fixed prompt as bench_llm: PyTorch eager (transformers, KV cache) and MLX-LM.

    .venv/bin/python bench/python/llm_baselines.py torch --model models/llm/Qwen2.5-0.5B-Instruct --prompt-ids models/llm/prompt512.txt --device mps --dtype fp16
    .venv-mlxlm/bin/python bench/python/llm_baselines.py mlx --model models/llm/mlx-fp16 --prompt-ids models/llm/prompt512.txt

Prints `llm <name> prefill p50 <ms> ... decode p50 <ms/token>`. Prefill: the prompt through the model up to the first logits (tokens/s = prompt length / that time).
Decode: --gen greedy tokens one at a time, end-of-generation ignored (a fixed count), per-token wall time. Same protocol as `bench_llm` and `llama-bench`.
"""
import argparse
import statistics
import sys
import time


def read_ids(path):
    return [int(x) for x in open(path).read().split()]


def stats(xs):
    xs = sorted(xs)
    return statistics.median(xs), xs[min(len(xs) - 1, int(0.99 * len(xs)))]


def run_torch(a):
    import torch
    from transformers import AutoModelForCausalLM

    dt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    ids = torch.tensor([read_ids(a.prompt_ids)], device=a.device)
    sync = torch.mps.synchronize if a.device == "mps" else (lambda: None)
    pre, dec = [], []
    with torch.no_grad():
        for it in range(a.warmup + a.iters):
            sync()
            t0 = time.perf_counter()
            out = model(ids, use_cache=True)
            tok = out.logits[:, -1].argmax(-1, keepdim=True)
            tok.item()
            sync()
            t1 = time.perf_counter()
            past = out.past_key_values
            for _ in range(a.gen):
                out = model(tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tok = out.logits[:, -1].argmax(-1, keepdim=True)
                tok.item()
            sync()
            t2 = time.perf_counter()
            if it >= a.warmup:
                pre.append((t1 - t0) * 1000)
                dec.append((t2 - t1) * 1000 / a.gen)
    return pre, dec, ids.shape[1]


def run_mlx(a):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, _ = load(a.model)
    prompt = mx.array(read_ids(a.prompt_ids))
    pre, dec = [], []
    for it in range(a.warmup + a.iters):
        t0 = time.perf_counter()
        stamps = []
        for n, (tok, _) in enumerate(generate_step(prompt, model, max_tokens=a.gen + 1)):
            stamps.append(time.perf_counter())
        # the first yield is the prompt processed and the first token chosen; each later one is one decode step
        if it >= a.warmup:
            pre.append((stamps[0] - t0) * 1000)
            dec.append((stamps[-1] - stamps[0]) * 1000 / (len(stamps) - 1))
    return pre, dec, prompt.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["torch", "mlx"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    pre, dec, n = (run_torch if a.backend == "torch" else run_mlx)(a)
    p50, p99 = stats(pre)
    d50, d99 = stats(dec)
    name = a.name or (f"pytorch {a.device} {a.dtype}" if a.backend == "torch" else f"mlx-lm {a.model.rsplit('/', 1)[-1]}")
    print(f"llm {name}: prompt {n} tokens, {a.gen} generated")
    print(f"llm {name} prefill p50 {p50:9.2f} ms p99 {p99:9.2f} ms {1000 * n / p50:9.1f} tokens/s")
    print(f"llm {name} decode p50 {d50:9.3f} ms/token p99 {d99:9.3f} {1000 / d50:9.1f} tokens/s")


if __name__ == "__main__":
    sys.exit(main())
