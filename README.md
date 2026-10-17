# dooms.cpp

An inference engine for Apple Silicon that runs language, vision-language and vision-language-action models from C++, C and Python, on the GPU (Metal) and the
Neural Engine (Core ML).

- **LLMs**: its own Metal decoder for llama and Qwen2 models in f16 (fused kernels, flash-attention prompt processing, the next token picked on the GPU), and
  any other GGUF through [llama.cpp](https://github.com/ggml-org/llama.cpp) (every architecture and quantization). Chat templates, streaming, stop strings,
  seeded sampling, and an OpenAI-compatible server (`dooms serve`).
- **Vision-language** (SmolVLM): the vision tower on the Neural Engine, the language model on the engine's own decoder.
- **Speech to text** (Whisper, tiny to large-v3-turbo): the encoder on Core ML, the decoder on the engine's own Metal kernels; any audio file Core Audio decodes.
- **Vision-language-action** (SmolVLA): three cameras plus a prompt to a 50-step action chunk in about 270 ms, and an asynchronous runner that computes the next
  chunk while the robot executes the current one.
- **Packaging**: a self-contained Python wheel, a C/C++ SDK with `find_package(dooms)`, a command-line tool, `dooms pull` for verified model downloads.

Apple Silicon macOS only. The engine is an inference engine, not a trainer.

## Benchmarks

Apple M1 Pro (16 GB), macOS 26.6, plugged in, interleaved rounds so every runtime meets the same conditions, each baseline in its fastest configuration, the
same precision everywhere (f16 unless stated). Full tables, methods and history: [dooms-kappa.vercel.app](https://dooms-kappa.vercel.app) and
`bench/BASELINES.md`.

**Text LLM** (Qwen2.5-0.5B-Instruct, 512-token prompt, then 128 greedy tokens)

| | decode, tokens/s | prompt, tokens/s |
|---|--:|--:|
| **dooms.cpp f16** | **137.9** | 3999 |
| MLX-LM fp16 | 136.2 | 3864 |
| llama.cpp f16 | 130.7 | **4604** |
| PyTorch MPS fp16 | 37.6 | 2492 |

Fastest f16 decode; the prompt is 13% behind llama.cpp. At 4-bit, MLX-LM (211.6 tokens/s) is ahead of llama.cpp's Q4_K_M (164.9), which the engine uses for
quantized files (161.5).

**Vision-language** (SmolVLM-256M, one 512x512 image and a question)

| | decode, ms/token | first token, ms | vision tower, ms |
|---|--:|--:|--:|
| **dooms.cpp** | **2.70** | **82.5** | **62.2** |
| MLX fp16 | 3.44 | 97.0 | 68.6 |
| llama.cpp (text model alone) | 4.11 | | 86 per tile |
| PyTorch MPS | 27.3 | 139.4 | 106.2 |

**Vision-language-action** (SmolVLA, three cameras + prompt + state to 50 actions): dooms.cpp 266.6 ms, MLX 469.3 ms (1.76x), PyTorch MPS 603.8 ms (2.26x).
In a simulated 30 Hz control loop the asynchronous runner removes every stall (90 waited steps to 0).

**Speech to text** (Whisper, 73 LibriSpeech clips = 481 s of audio, greedy, f16; time for all of them)

| | large-v3-turbo | WER | tiny | WER |
|---|--:|--:|--:|--:|
| **dooms.cpp** | 54.8 s | 3.25% | 3.24 s | 12.2% |
| MLX-Whisper fp16 | **53.9 s** | 3.42% | **3.08 s** | 12.1% |
| whisper.cpp (Metal) | 64.5 s | 3.25% | 5.44 s | 12.0% |
| PyTorch MPS fp16 | 105.0 s | 3.25% | 15.1 s | 12.1% |
| Sapient (Metal) | 948 s | 3.34% | 68.6 s | 18.5% |

Level with MLX (on one 8-minute file dooms.cpp is ahead: turbo 16.2 s vs 16.3, tiny 1.66 s vs 1.80), ahead of everything else, with PyTorch's WER.

Tokens match transformers' on the reference prompts, photos and audio clips for every model above.

## Use it

```
pip install dooms-cpp                         # PyPI; the import and the command are both `dooms`
dooms pull qwen2.5-0.5b                       # downloads and verifies a GGUF (SHA-256)
dooms serve --model ~/.cache/dooms/qwen2.5-0.5b-instruct-q4_k_m.gguf   # OpenAI-compatible API on 127.0.0.1:8080
```

```python
import dooms
llm = dooms.LLM("qwen2.5-0.5b-instruct-f16.gguf")      # f16 llama/qwen2: the engine's own Metal decoder
print(llm.chat([{"role": "user", "content": "Name three colours."}], max_new_tokens=48))
```

```cpp
#include <dooms/dooms.hpp>
dooms::Llm llm("qwen2.5-0.5b-instruct-f16.gguf");
std::cout << llm.chat({{"user", "Name three colours."}}, 48) << "\n";
```

See [docs/USING.md](docs/USING.md) for installation and the full API (LLM, VLM, VLA and the asynchronous runner, the server, `dooms pull`). Models are not
included.

## Build
```
python3 -m venv .venv && .venv/bin/pip install cmake ninja numpy
export PATH="$PWD/.venv/bin:$PATH"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
scripts/package-python.sh     # dist/dooms_cpp-*.whl (release steps: docs/RELEASING.md)
scripts/package-sdk.sh        # dist/dooms-sdk-*.tar.gz
```
The first configure fetches llama.cpp (pinned in `CMakeLists.txt`).

## Scope
The engine's own Metal kernels run llama/Qwen2 text models in f16 and the convolutional graphs; Core ML runners serve the vision towers and SmolVLA on the
Neural Engine; llama.cpp runs every other GGUF (other architectures, quantized files). The prompt kernels borrow the structure of llama.cpp's Metal matmul
(MIT). Not included: training, non-Apple platforms, and the hand-mesh models (see `NOTICE`).

## Licence
Apache-2.0 (`LICENSE`). Third-party software and model licences: `NOTICE`.
