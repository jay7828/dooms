#!/usr/bin/env python3
"""Benchmarks a text LLM on the engine against llama.cpp (llama-bench), MLX-LM and PyTorch, same model weights, same 512-token prompt, 128 greedy tokens.

    .venv/bin/python bench/python/run_llm.py [--rounds 3] [--max-load 1.5] [--wait-minutes 60] [--out bench/reports/DATE-llm]

Interleaved rounds (every configuration meets the same background load), one process per configuration and round, median over rounds, spread = (max-min)/median.
Prefill = 512 prompt tokens in one batch (tokens/s); decode = tokens/s of 128 single-token steps after that prompt. Refuses to run on battery or with
macOS Low Power Mode on (both change the clocks).
"""
import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
B = os.path.join(ROOT, "build")
LB = os.path.join(ROOT, "build-llamacpp-ref", "bin", "llama-bench")
PY, PY_MLXLM = os.path.join(ROOT, ".venv", "bin", "python"), os.path.join(ROOT, ".venv-mlxlm", "bin", "python")
M = os.path.join(ROOT, "models", "llm")
IDS = os.path.join(M, "prompt512.txt")
HF = os.path.join(M, "Qwen2.5-0.5B-Instruct")
BASE = os.path.join(ROOT, "bench", "python", "llm_baselines.py")


def parse_engine(out):
    p = re.search(r"prefill\s+p50\s+([\d.]+) ms.*?([\d.]+) tokens/s", out)
    d = re.search(r"decode\s+p50\s+([\d.]+) ms/token.*?([\d.]+) tokens/s", out)
    return {"prefill_tps": float(p.group(2)), "decode_tps": float(d.group(2)), "decode_ms": float(d.group(1))}


def parse_baseline(out):
    p = re.search(r"prefill p50\s+([\d.]+) ms p99\s+[\d.]+ ms\s+([\d.]+) tokens/s", out)
    d = re.search(r"decode p50\s+([\d.]+) ms/token p99\s+[\d.]+\s+([\d.]+) tokens/s", out)
    return {"prefill_tps": float(p.group(2)), "decode_tps": float(d.group(2)), "decode_ms": float(d.group(1))}


def parse_llama_bench(out):
    rows = []
    for chunk in re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S):
        rows += json.loads(chunk)
    pre = [r["avg_ts"] for r in rows if r["n_prompt"] == 512 and r["n_gen"] == 0]
    dec = [r["avg_ts"] for r in rows if r["n_gen"] == 128]
    return {"prefill_tps": pre[0], "decode_tps": dec[0], "decode_ms": 1000.0 / dec[0]}


def configs():
    cs = []
    for tag, gguf in (("f16", "qwen2.5-0.5b-instruct-f16.gguf"), ("Q4_K_M", "qwen2.5-0.5b-instruct-q4_k_m.gguf")):
        path = os.path.join(M, gguf)
        # "dooms.cpp" is the shipped choice (--engine auto: the own Metal decoder for f16, llama.cpp for quantized files); the second row forces llama.cpp.
        cs.append((f"dooms.cpp {tag}", [[os.path.join(B, "bench_llm"), "--model", path, "--prompt-ids", IDS, "--gen", "128", "--iters", "5", "--warmup", "2"]], parse_engine))
        if tag == "f16":
            cs.append((f"dooms.cpp {tag}, llama.cpp inside", [[os.path.join(B, "bench_llm"), "--model", path, "--prompt-ids", IDS, "--gen", "128", "--iters", "5", "--warmup", "2",
                                                               "--engine", "llama.cpp"]], parse_engine))
        cs.append((f"llama.cpp (llama-bench) {tag}", [[LB, "-m", path, "-p", "512", "-n", "0", "-fa", "1", "-r", "3", "-o", "json"],
                                                      [LB, "-m", path, "-p", "0", "-n", "128", "-d", "512", "-fa", "1", "-r", "3", "-o", "json"]], parse_llama_bench))
    cs.append(("MLX-LM fp16", [[PY_MLXLM, BASE, "mlx", "--model", os.path.join(M, "mlx-fp16"), "--prompt-ids", IDS, "--name", "x"]], parse_baseline))
    cs.append(("MLX-LM 4-bit", [[PY_MLXLM, BASE, "mlx", "--model", os.path.join(M, "mlx-q4"), "--prompt-ids", IDS, "--name", "x"]], parse_baseline))
    cs.append(("PyTorch MPS fp16", [[PY, BASE, "torch", "--model", HF, "--prompt-ids", IDS, "--device", "mps", "--dtype", "fp16", "--name", "x"]], parse_baseline))
    cs.append(("PyTorch CPU fp32", [[PY, BASE, "torch", "--model", HF, "--prompt-ids", IDS, "--device", "cpu", "--dtype", "fp32", "--iters", "3", "--warmup", "1", "--name", "x"]], parse_baseline))
    return cs


def state():
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
    lpm = [l for l in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout.splitlines() if l.strip().startswith("lowpowermode")]
    return {"load": [round(x, 2) for x in os.getloadavg()], "power": "AC" if "AC Power" in batt else "battery", "low_power_mode": bool(lpm and lpm[0].split()[-1] == "1")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-load", type=float, default=0.0)
    ap.add_argument("--wait-minutes", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "reports", datetime.date.today().isoformat() + "-llm"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cs = configs()
    results = {name: [] for name, _, _ in cs}
    failures = []
    states = []
    for r in range(a.rounds):
        if a.max_load > 0:
            deadline = time.time() + a.wait_minutes * 60
            while os.getloadavg()[0] >= a.max_load:
                if time.time() > deadline:
                    sys.exit(f"the machine did not become idle (load {os.getloadavg()[0]:.1f} >= {a.max_load}) within {a.wait_minutes} minutes; nothing recorded for round {r + 1}")
                time.sleep(15)
        st = state()
        if a.max_load > 0 and (st["power"] != "AC" or st["low_power_mode"]):
            sys.exit("the machine is on battery or in Low Power Mode; nothing was measured")
        states.append(st)
        print(f"=== round {r + 1}/{a.rounds} {st}", flush=True)
        for name, cmds, parse in cs:
            t0 = time.time()
            try:
                out = ""
                for cmd in cmds:
                    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1800)
                    if p.returncode != 0:
                        raise RuntimeError(f"exit {p.returncode}: {p.stderr[-300:]}")
                    out += p.stdout + "\n"
                res = parse(out)
                results[name].append(res)
                print(f"  {name:36s} prefill {res['prefill_tps']:8.1f} tok/s   decode {res['decode_tps']:7.1f} tok/s ({res['decode_ms']:.2f} ms)  ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                failures.append({"round": r + 1, "config": name, "error": str(e)[:400]})
                print(f"  {name}: FAILED {str(e)[:200]}", flush=True)
    json.dump({"states": states, "results": results, "failures": failures}, open(os.path.join(a.out, "raw.json"), "w"), indent=1)

    def med(name, key):
        v = [x[key] for x in results[name]]
        return (statistics.median(v), (max(v) - min(v)) / statistics.median(v) * 100) if v else (None, None)

    lines = [f"# LLM throughput: Qwen2.5-0.5B-Instruct, 512-token prompt, 128 greedy tokens ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
             f"- Machine: {platform.platform()} (Apple Silicon), power {states[0]['power']}, Low Power Mode {'ON' if states[0]['low_power_mode'] else 'off'}, load at round starts: "
             + ", ".join(str(s['load'][0]) for s in states),
             f"- {a.rounds} interleaved rounds, median of per-round values; spread = (max - min) / median.",
             "- Prefill: 512 prompt tokens in one batch. Decode: 128 single-token steps after that prompt, end-of-generation ignored. Same token ids for every row.",
             "- dooms.cpp rows: the shipped text engine (the own Metal decoder for f16 llama/qwen2 files, llama.cpp for quantized ones); 'llama.cpp inside' forces llama.cpp.",
             "", "| configuration | prefill tokens/s | spread | decode tokens/s | spread | decode ms/token |", "|---|--:|--:|--:|--:|--:|"]
    for name, _, _ in cs:
        pf, ps = med(name, "prefill_tps")
        dc, ds = med(name, "decode_tps")
        dm, _ = med(name, "decode_ms")
        if pf is None:
            lines.append(f"| {name} | failed | | | | |")
        else:
            lines.append(f"| {name} | {pf:.0f} | {ps:.0f}% | {dc:.1f} | {ds:.0f}% | {dm:.2f} |")
    if failures:
        lines += ["", "## Failed runs", ""] + [f"- round {f['round']} {f['config']}: {f['error']}" for f in failures]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
