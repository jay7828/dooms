# dooms.cpp

An inference engine for Apple Silicon that runs language, vision-language and vision-language-action models from C++, C and Python, on the GPU (Metal) and the
Neural Engine (Core ML).

- **LLMs**: any text GGUF that [llama.cpp](https://github.com/ggml-org/llama.cpp) loads (all architectures, all quantizations) behind a small API: chat templates,
  streaming, stop strings, seeded sampling. Same speed as llama.cpp's own benchmark on the same file, plus bindings and packaging.
- **Vision-language** (SmolVLM): the vision tower on the Neural Engine, the language model on the GPU; 1.5-1.8x faster than PyTorch MPS in time to first token and vision.
- **Vision-language-action** (SmolVLA): three cameras plus a prompt to a 50-step action chunk in about 270 ms, 2.3x faster than PyTorch MPS.
- **Packaging**: a self-contained Python wheel, a C/C++ SDK with `find_package(dooms)`, a command-line tool.

Apple Silicon macOS only. The engine is an inference engine, not a trainer.

```python
import dooms
llm = dooms.LLM("qwen2.5-0.5b-instruct-q4_k_m.gguf")
print(llm.chat([{"role": "user", "content": "Name three colours."}], max_new_tokens=48))
```

```cpp
#include <dooms/dooms.hpp>
dooms::Llm llm("qwen2.5-0.5b-instruct-q4_k_m.gguf");
std::cout << llm.chat({{"user", "Name three colours."}}, 48) << "\n";
```

See [docs/USING.md](docs/USING.md) for installation and the full API, and [dooms-kappa.vercel.app](https://dooms-kappa.vercel.app) for the benchmarks and what changed along the way. Models are not included.

## Build
```
python3 -m venv .venv && .venv/bin/pip install cmake ninja numpy
export PATH="$PWD/.venv/bin:$PATH"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
scripts/package-python.sh     # dist/dooms-*.whl
scripts/package-sdk.sh        # dist/dooms-sdk-*.tar.gz
```
The first configure fetches llama.cpp (pinned in `CMakeLists.txt`).

## Scope
The language-model path is llama.cpp's Metal backend behind this API; the engine's own IR, Metal kernels and Core ML runners serve the vision and action models.
Not included: training, non-Apple platforms, and the hand-mesh models (see `NOTICE`).

## Licence
Apache-2.0 (`LICENSE`). Third-party software and model licences: `NOTICE`.
