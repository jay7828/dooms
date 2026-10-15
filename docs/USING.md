# Using dooms.cpp

Apple Silicon macOS only (ADR 0005). Models are never bundled: bring a GGUF for the LLM, or convert the others with `tools/convert` (docs/DEVELOPING.md).

## Python

```
pip install dooms-0.2.0-py3-none-macosx_11_0_arm64.whl      # from scripts/package-python.sh; needs numpy
```

```python
import dooms

# Any text GGUF that llama.cpp loads (every architecture, every quantization)
llm = dooms.LLM("qwen2.5-0.5b-instruct-q4_k_m.gguf", n_ctx=4096)
print(llm.generate("The capital of France is", max_new_tokens=32))
print(llm.chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name three colours."}], max_new_tokens=48))
for piece in llm.stream_chat([{"role": "user", "content": "Tell a short story."}], max_new_tokens=200, temperature=0.8, seed=1):
    print(piece, end="", flush=True)

vlm = dooms.VLM.from_dir("models/smolvlm");  vlm.generate(rgb_array, "Describe this image.")
vla = dooms.VLA.from_dir("models/smolvla");  vla.infer([cam1, cam2, cam3], "pick up the red cube", state)
```

`temperature=0` (default) is greedy and deterministic; above 0 tokens are drawn with a seeded generator (`seed`, `top_k`, `top_p`, `repeat_penalty`). `stop=[...]` ends the text before a
string. Each call is a fresh conversation. Errors raise `dooms.DoomsError` (`.code` is `NotFound`, `InvalidArgument`, ...). llama.cpp's loading log is silent; `DOOMS_LOG=info` shows it.

## C++

```
tar xzf dooms-sdk-0.2.0-macos-arm64.tar.gz           # from scripts/package-sdk.sh
cmake -S examples/cpp -B build -DCMAKE_PREFIX_PATH=$PWD/dooms-sdk-0.2.0-macos-arm64
```

```cpp
#include <dooms/dooms.hpp>                      // header-only RAII wrapper over the C API (dooms_c.h)
dooms::Llm llm("model-q4_k_m.gguf");
std::string answer = llm.chat({{"user", "Name three colours."}}, 64);
llm.generate("Once upon a time", 100, {0.8f, 40, 0.95f, 1.0f, /*seed*/ 1}, {"\n\n"},
             [](const std::string & piece) { std::cout << piece << std::flush; return true; });   // return false to stop
```

CMake: `find_package(dooms 0.2 REQUIRED)` then `target_link_libraries(app PRIVATE dooms::dooms)`. pkg-config: `dooms.pc`. Plain C: `dooms_c.h` (stable, append-only; `dooms_llm_*`,

## Command line

```
dooms-cli llm --model qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "Name three colours." --max-tokens 64 --stats
dooms-cli llm --model model.gguf --raw --prompt "Once upon a time" --temp 0.8 --seed 1 --stop "\n\n"
dooms-cli vlm --image photo.jpg --prompt "Describe this image." --models models/smolvlm
```

## Getting a model

```
python build/_deps/llama_cpp-src/convert_hf_to_gguf.py Qwen2.5-0.5B-Instruct --outfile qwen-f16.gguf --outtype f16
build-llamacpp-ref/bin/llama-quantize qwen-f16.gguf qwen-q4_k_m.gguf Q4_K_M       # llama.cpp's own quantizer
```

## What it is and is not
The language-model path is llama.cpp's Metal backend behind this API (ADR 0003): same kernels, same speed (`bench/reports/*-llm`), plus streaming, stop strings, seeded sampling,
chat templates, C/C++/Python bindings and a self-contained wheel/SDK. The engine's own kernels and the Neural Engine paths are for the vision and action models.
