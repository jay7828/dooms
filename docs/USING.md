# Using dooms.cpp

Apple Silicon macOS only (ADR 0005). Models are never bundled: bring a GGUF for the LLM, or convert the others with `tools/convert` (docs/DEVELOPING.md).

## Python

```
pip install dooms-cpp      # PyPI (macOS 14+, Apple Silicon); or the wheel attached to the GitHub release, or your own build: scripts/package-python.sh
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

# Speech to text (Whisper; dooms pull whisper-large-v3-turbo, or tools/convert/convert_whisper.py)
stt = dooms.Whisper("models/whisper-large-v3-turbo")
print(stt.transcribe("meeting.m4a"), stt.last_timings.rtf)        # a file Core Audio decodes, or 16 kHz mono float samples

# Text to speech (Kokoro-82M; dooms pull kokoro, or tools/convert/convert_kokoro.py; English text needs pip install "misaki[en]")
tts = dooms.TTS("models/kokoro")
dooms.save_wav("hello.wav", tts.speak("Hello from dooms.cpp.", voice="af_heart"), tts.sample_rate)
audio = tts.synthesize("həlˈO wˈɜɹld.", voice="bm_george", speed=1.1, seed=1)   # phonemes directly; the same seed gives the same samples
```

`temperature=0` (default) is greedy and deterministic; above 0 tokens are drawn with a seeded generator (`seed`, `top_k`, `top_p`, `repeat_penalty`). `stop=[...]` ends the text before a
string. Each call is a fresh conversation. Errors raise `dooms.DoomsError` (`.code` is `NotFound`, `InvalidArgument`, ...). llama.cpp's loading log is silent; `DOOMS_LOG=info` shows it.

## C++

```
curl -LO https://github.com/jay7828/dooms/releases/download/v0.5.0/dooms-sdk-0.5.0-macos-arm64.tar.gz && tar xzf dooms-sdk-0.5.0-macos-arm64.tar.gz   # or scripts/package-sdk.sh
cmake -S examples/cpp -B build -DCMAKE_PREFIX_PATH=$PWD/dooms-sdk-0.5.0-macos-arm64
```

```cpp
#include <dooms/dooms.hpp>                      // header-only RAII wrapper over the C API (dooms_c.h)
dooms::Llm llm("model-q4_k_m.gguf");
std::string answer = llm.chat({{"user", "Name three colours."}}, 64);
llm.generate("Once upon a time", 100, {0.8f, 40, 0.95f, 1.0f, /*seed*/ 1}, {"\n\n"},
             [](const std::string & piece) { std::cout << piece << std::flush; return true; });   // return false to stop
```

CMake: `find_package(dooms 0.5 REQUIRED)` then `target_link_libraries(app PRIVATE dooms::dooms)`. pkg-config: `dooms.pc`. Plain C: `dooms_c.h` (stable, append-only; `dooms_llm_*`,

## Command line

```
dooms-cli llm --model qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "Name three colours." --max-tokens 64 --stats
dooms-cli llm --model model.gguf --raw --prompt "Once upon a time" --temp 0.8 --seed 1 --stop "\n\n"
dooms-cli vlm --image photo.jpg --prompt "Describe this image." --models models/smolvlm
dooms-cli transcribe --model ~/.cache/dooms/whisper-large-v3-turbo --audio speech.wav --stats
dooms-cli speak --model ~/.cache/dooms/kokoro --phonemes "həlˈO wˈɜɹld." --voice af_heart --out hello.wav --stats
```

## OpenAI-compatible server

The wheel installs a `dooms` command (also `python -m dooms`). `dooms serve` puts one text GGUF behind OpenAI's HTTP API (standard library only, no extra dependency):

```
dooms serve --model qwen2.5-0.5b-instruct-q4_k_m.gguf [--host 127.0.0.1] [--port 8080] [--n-ctx 4096] [--model-name qwen]
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, with `"stream": true` as Server-Sent Events (`data: {...}` chunks, then
`data: [DONE]`; `"stream_options": {"include_usage": true}` adds a final usage chunk).

```
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Name three colours."}], "temperature": 0, "max_tokens": 48}'
curl -N http://127.0.0.1:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "prompt": "Once upon a time", "max_tokens": 64, "stream": true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")      # no authentication: keep --host on 127.0.0.1 unless the network is trusted
r = client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Name three colours."}], max_tokens=48)
print(r.choices[0].message.content, r.usage)
for chunk in client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Tell a story."}], stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

- Parameters: `max_tokens` / `max_completion_tokens` (default: the rest of the context), `temperature` (default 1, 0 is greedy), `top_p` (default 1), `seed` (reduced modulo
  2^32; without it each request draws a random one), `stop` (a string or up to 4). `frequency_penalty` / `presence_penalty` are accepted only at 0, their no-op value.
- Rejected with a 400 and OpenAI's error object (`{"error": {"message", "type", "param", "code"}}`): `n` > 1, tools / function calling, logprobs, images or audio, JSON
  response formats, `logit_bias`, `echo`, `suffix`, several prompts per request, unknown parameters. A `model` other than the served name is a 404; omitting it is allowed.
- `usage` holds the real token counts (`completion_tokens` includes the end-of-generation token); `finish_reason` is `"length"` only when the token limit was hit.
- One model, one request at a time (the handle is not thread-safe): concurrent requests wait their turn. A client that disconnects during a stream stops its generation.
- The model's own chat template formats `messages` (`developer` is treated as `system`; a list of text parts is joined with newlines).

## Getting a model

`dooms pull` downloads a model from Hugging Face and checks it against the hash Hugging Face publishes for the file (SHA-256 for LFS files). Downloads are resumable
(`<file>.part`, HTTP Range; a broken connection is resumed automatically, and running the command again continues where it stopped), the file appears under its
final name only once verified, and a checksum mismatch deletes the partial file. Progress goes to stderr, the path of the model to stdout.

```
dooms pull qwen2.5-0.5b                          # aliases, each pinned to a commit of the publisher's own repository: qwen2.5-0.5b, qwen2.5-1.5b, qwen3-0.6b,
                                                 #   smollm2-1.7b, gemma-3-1b (ggml-org; Gemma licence)   (dooms pull --help lists them)
dooms pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf[@REVISION]   # any GGUF file of any repository
dooms serve --model "$(dooms pull qwen2.5-0.5b)"
dooms list                                       # what is in the models directory
```

- The models directory is `--dir DIR`, else `$DOOMS_MODELS`, else `~/.cache/dooms`. `HF_TOKEN` is sent for gated or private repositories (to Hugging Face only, never to
  the CDN a download is redirected to); `HF_ENDPOINT` points at a mirror.
- `dooms pull smolvlm` / `dooms pull smolvla`: these run from Core ML packages and GGUF files converted on your machine, never redistributed (licensing, see the NOTICE).
  `pull` downloads the **original** weights from the official repositories (HuggingFaceTB/SmolVLM-256M-Instruct; HuggingFaceTB/SmolVLM2-500M-Video-Instruct and
  lerobot/smolvla_base), pinned to a commit and verified, into `<dir>/sources/`, runs the converters of `tools/convert` into `<dir>/smolvlm` (or `smolvla`), checks that
  the result loads, and only then moves it into place. This needs a source checkout (built once, which fetches llama.cpp) and its converter environment:
  `.venv` with torch, coremltools, transformers, gguf, safetensors, huggingface_hub (plus `.venv-lerobot` with LeRobot for SmolVLA); run
  `PYTHONPATH=python .venv/bin/python -m dooms pull smolvlm` there. Override with `DOOMS_SOURCE_DIR`, `DOOMS_CONVERT_PYTHON`, `DOOMS_LEROBOT_PYTHON`,
  `DOOMS_LLAMA_CPP_DIR`. From a wheel install it stops with these instructions.
- `dooms pull whisper-tiny` / `dooms pull whisper-large-v3-turbo`: the same, from openai/whisper-tiny or openai/whisper-large-v3-turbo (MIT) into `<dir>/whisper-*`
  (encoder-fp16.mlpackage + decoder.gguf; turbo's encoder conversion takes about 17 minutes). No llama.cpp needed.
- `dooms pull kokoro`: hexgrad/Kokoro-82M (Apache-2.0, all 54 voices) converted into `<dir>/kokoro` with the reference package installed in its own environment
  (`.venv-kokoro`: `kokoro==0.9.4 'misaki[en]' torch==2.8.0 coremltools==9.0 gguf`; `DOOMS_KOKORO_PYTHON` overrides it).
- By hand, any Hugging Face checkpoint llama.cpp supports:

```
python build/_deps/llama_cpp-src/convert_hf_to_gguf.py Qwen2.5-0.5B-Instruct --outfile qwen-f16.gguf --outtype f16
build-llamacpp-ref/bin/llama-quantize qwen-f16.gguf qwen-q4_k_m.gguf Q4_K_M       # llama.cpp's own quantizer
```

## What it is and is not
The language-model path is llama.cpp's Metal backend behind this API (ADR 0003): same kernels, same speed (`bench/reports/*-llm`), plus streaming, stop strings, seeded sampling,
chat templates, C/C++/Python bindings and a self-contained wheel/SDK. The engine's own kernels and the Neural Engine paths are for the vision and action models.
