# dooms-cpp

Python bindings and command-line tools for [dooms.cpp](https://github.com/jay7828/dooms), an inference engine for Apple Silicon that runs language,
vision-language, speech (Whisper, Kokoro text to speech) and vision-language-action models on the GPU (Metal) and the Neural Engine (Core ML).

**Apple Silicon macOS only** (arm64). The wheel bundles the engine's libraries (built with llama.cpp and ggml, MIT); it contains no model weights.

```
pip install dooms-cpp
dooms pull qwen2.5-0.5b                                   # downloads and verifies a GGUF from its publisher's Hugging Face repository
dooms serve --model ~/.cache/dooms/qwen2.5-0.5b-instruct-q4_k_m.gguf     # OpenAI-compatible API on http://127.0.0.1:8080/v1
dooms pull whisper-large-v3-turbo                         # speech to text, ready converted (whisper-tiny: 69 MB)
dooms pull kokoro                                         # text to speech, 54 voices
```

```python
import dooms
with dooms.LLM("qwen2.5-0.5b-instruct-q4_k_m.gguf") as llm:              # any text GGUF that llama.cpp loads
    print(llm.chat([{"role": "user", "content": "Name three colours."}], max_new_tokens=48))
    for piece in llm.stream_chat([{"role": "user", "content": "Tell a short story."}], max_new_tokens=200):
        print(piece, end="", flush=True)

stt = dooms.Whisper("~/.cache/dooms/whisper-large-v3-turbo")
print(stt.transcribe("meeting.m4a"))                                     # any file Core Audio decodes
tts = dooms.TTS("~/.cache/dooms/kokoro")                                 # pip install "misaki[en]" for English text
dooms.save_wav("hello.wav", tts.speak("Hello from dooms.cpp.", voice="af_heart"), tts.sample_rate)
```

- `dooms.LLM`: chat templates, streaming, stop strings, seeded sampling, exact token counts (`last_generation`, `count_tokens`).
- `dooms.VLM` (SmolVLM) and `dooms.VLA` (SmolVLA) open models converted locally with the repository's converters (`dooms pull smolvlm` in a source checkout).
- `dooms.Whisper` (speech to text) and `dooms.TTS` (Kokoro-82M) run the files `dooms pull` downloads (converted, SHA-256 pinned, with the models' licences).
- `dooms serve`: `/v1/chat/completions`, `/v1/completions` (with streaming) and `/v1/models`, usable from the official `openai` client.

The import name is `dooms`. Documentation: [docs/USING.md](https://github.com/jay7828/dooms/blob/main/docs/USING.md). Licence: Apache-2.0; each model keeps its own licence.
