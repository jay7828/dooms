#!/bin/bash
# Reproduces the README's 4-bit table on your Mac: dooms.cpp's own decoder vs llama.cpp (the copy inside the engine) on the same pure Q4_0 file, and MLX-LM's
# 4-bit model of the same size; then checks that the engine picks llama.cpp's greedy tokens. Qwen2.5-0.5B-Instruct, a 512-token prompt, 128 tokens after it
# (end-of-text ignored, as llama-bench and MLX's own timing do), 2 interleaved rounds.
#
#   scripts/reproduce-4bit.sh [WORK_DIR]       # default: ./reproduce-4bit; needs ~2 GB of disk and network for the first run
#
# Environment: PYTHON (a Python 3.10+ for MLX-LM; default python3); DOOMS_ALLOW_BUSY_GPU=1 measures even when another app keeps the GPU busy. Run on AC power with nothing else busy; the numbers vary between Macs: please share yours.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/reproduce-4bit}"
PY="${PYTHON:-python3}"
mkdir -p "$WORK/models"
cd "$ROOT"

if [ "$(uname -m)" != "arm64" ] || [ "$(uname -s)" != "Darwin" ]; then echo "error: Apple Silicon macOS only"; exit 2; fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then echo "error: MLX-LM needs Python 3.10+ (set PYTHON=/path/to/python3.11)"; exit 2; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "error: on battery: plug in (the GPU clocks differ)"; exit 2; fi
# Another app holding the GPU (a video call, a browser tab with WebGL or video) makes every engine slow: sample its utilization before measuring.
gpu_busy() {
  local total=0 n=0 u
  for _ in 1 2 3 4 5; do
    u=$(ioreg -r -d 1 -c IOAccelerator | grep -oE '"Device Utilization %"=[0-9]+' | head -1 | cut -d= -f2)
    total=$((total + ${u:-0})); n=$((n + 1)); sleep 1
  done
  echo $((total / n))
}

echo "== 1/5 Python environment (cmake, ninja, gguf, huggingface_hub, mlx-lm)"
[ -x "$WORK/venv/bin/python" ] || "$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --upgrade pip
"$WORK/venv/bin/pip" install -q cmake ninja numpy gguf==0.18.0 mlx==0.32.4 mlx-lm==0.32.0 huggingface_hub
export PATH="$WORK/venv/bin:$PATH"

echo "== 2/5 build (bench_llm, test_metal_llama_decoder)"
[ -f build/build.ninja ] || cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target bench_llm test_metal_llama_decoder -j >/dev/null

echo "== 3/5 models (pinned commits)"
F16="$WORK/models/qwen2.5-0.5b-instruct-fp16.gguf"
Q40="$WORK/models/qwen2.5-0.5b-instruct-q4_0-pure.gguf"
MLX="$WORK/models/mlx-4bit"
[ -f "$F16" ] || hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf --revision 9217f5db79a29953eb74d5343926648285ec7e67 --local-dir "$WORK/models" >/dev/null
[ -f "$MLX/model.safetensors" ] || hf download mlx-community/Qwen2.5-0.5B-Instruct-4bit --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 --local-dir "$MLX" >/dev/null
# pure Q4_0 (every matrix, the embeddings too): the same bytes as llama.cpp's `llama-quantize --pure ... Q4_0`, and MLX-LM's 4-bit size
[ -f "$Q40" ] || python tools/convert/quantize_q4_0.py --in "$F16" --out "$Q40"
IDS="$WORK/prompt512.txt"   # the same 512 token ids for every engine (the content does not change the work)
python -c "print(' '.join(str(100 + (i * 7919) % 30000) for i in range(512)))" > "$IDS"

busy=$(gpu_busy)
if [ "$busy" -gt 15 ] && [ "${DOOMS_ALLOW_BUSY_GPU:-0}" != "1" ]; then
  echo "error: the GPU is ${busy}% busy before the benchmark: close what is using it (video calls, browser tabs, games), or set DOOMS_ALLOW_BUSY_GPU=1"; exit 2
fi
echo "== 4/5 speed: 2 interleaved rounds (GPU ${busy}% busy before)"
LOG="$WORK/results.txt"
: > "$LOG"
for round in 1 2; do
  for cfg in dooms llamacpp mlx; do
    case $cfg in
      dooms)    r=$(build/bench_llm --model "$Q40" --prompt-ids "$IDS" --gen 128 --iters 5 --warmup 2 2>&1) ;;
      llamacpp) r=$(build/bench_llm --model "$Q40" --prompt-ids "$IDS" --gen 128 --iters 5 --warmup 2 --engine llama.cpp 2>&1) ;;
      mlx)      r=$(python bench/python/llm_baselines.py mlx --model "$MLX" --prompt-ids "$IDS" --name mlx 2>&1) ;;
    esac
    r=$(echo "$r" | grep -E 'prefill|decode' | tr '\n' ' ')
    echo "$round $cfg $r" | tee -a "$LOG"
  done
done

echo "== 5/5 the engine's tokens against llama.cpp's on the same file"
DOOMS_QWEN_GGUF="$Q40" build/test_metal_llama_decoder 2>&1 | grep -E 'vs llama.cpp|qwen2_matches' || true

python - "$LOG" <<'EOF'
import re, statistics, sys
rows = {}
for line in open(sys.argv[1]):
    parts = line.split(maxsplit=2)
    if len(parts) < 3: continue
    p = re.search(r"prefill\s+p50\s+([\d.]+) ms", parts[2])
    d = re.search(r"decode\s+p50\s+([\d.]+) ms/token", parts[2])
    if d and p: rows.setdefault(parts[1], []).append((float(d.group(1)), float(p.group(1))))
names = {"dooms": "dooms.cpp, own decoder (Q4_0, pure)", "llamacpp": "llama.cpp inside the engine (same file)", "mlx": "MLX-LM 4-bit (group 64)"}
print("\n| 512-token prompt, 128 tokens | decode tokens/s | prompt tokens/s |\n|---|--:|--:|")
for k, label in names.items():
    if k in rows:
        dm = statistics.median(x[0] for x in rows[k]); pm = statistics.median(x[1] for x in rows[k])
        print(f"| {label} | {1000 / dm:.1f} | {512 * 1000 / pm:.0f} |")
EOF
echo
echo "Raw lines: $LOG. The README's numbers: Apple M1 Pro 16 GB (bench/BASELINES.md \"4-bit on the own decoder\")."
