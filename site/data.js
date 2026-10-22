// Every number on the site comes from this file, and every value here comes from a benchmark report in the repository.
// Models: bench/reports/2026-10-15-all-models/report.md (3 interleaved rounds, AC power, Low Power Mode off, load 1.3-1.5).
// SmolVLM decode and time to first token: bench/BASELINES.md "Own decoder for text LLMs too" (decode_shootout.sh, 3 interleaved rounds, the engine's own
// Metal decoder). LLM: bench/reports/2026-10-15-llm-metal/report.md (run_llm.py, 2 interleaved rounds).
// Speech: bench/BASELINES.md "Speech to text" (dooms.cpp, MLX, whisper.cpp: bench/reports/2026-10-18-speech after the decoder optimizations; PyTorch and Sapient:
// bench/reports/2026-10-17-speech). Lower median of two rounds per configuration.
// Machine: Apple M1 Pro (6 performance + 2 efficiency cores, 14-core GPU, 16 GB), macOS 26.6.

window.DOOMS = {
  machine: "Apple M1 Pro, 16 GB, macOS 26.6",
  measured: "15 to 18 October 2026",

  systems: {
    dooms: { label: "dooms.cpp", slot: 1 },
    pytorch: { label: "PyTorch", slot: 2 },
    mlx: { label: "MLX", slot: 3 },
    llama: { label: "llama.cpp", slot: 4 },
    whispercpp: { label: "whisper.cpp", slot: 4 },
    sapient: { label: "Sapient", slot: 4 },
    onnx: { label: "ONNX Runtime", slot: 4 },
  },

  // SmolVLA: three cameras + prompt + robot state -> a 50-step action chunk (10 flow-matching steps)
  vla: {
    totals: [
      { sys: "dooms", name: "dooms.cpp, Neural Engine", ms: 266.6 },
      { sys: "mlx", name: "MLX (mlx-smolvla) fp32", ms: 469.3 },
      { sys: "pytorch", name: "PyTorch MPS fp32", ms: 603.8 },
      { sys: "pytorch", name: "PyTorch CPU fp32", ms: 1301.7 },
    ],
    stages: {
      dooms: [
        { stage: "3 camera towers", ms: 185.8 },
        { stage: "VLM prefix", ms: 18.9 },
        { stage: "action expert, 10 steps", ms: 58.1 },
      ],
      pytorch: [
        { stage: "3 camera towers", ms: 327.9 },
        { stage: "VLM prefix", ms: 55.5 },
        { stage: "action expert, 10 steps", ms: 220.3 },
      ],
    },
  },

  // SmolVLM-256M: one 512x512 image + question -> answer
  vlm: {
    vision: [
      { sys: "dooms", name: "dooms.cpp, Neural Engine", ms: 62.2 },
      { sys: "mlx", name: "MLX fp16", ms: 68.6 },
      { sys: "pytorch", name: "PyTorch MPS fp32", ms: 106.2 },
    ],
    ttft: [
      { sys: "dooms", name: "dooms.cpp", ms: 82.5 },
      { sys: "mlx", name: "MLX fp16", ms: 97.0 },
      { sys: "pytorch", name: "PyTorch MPS fp16", ms: 139.4 },
    ],
    decode: [
      { sys: "dooms", name: "dooms.cpp, own Metal decoder", ms: 2.70 },
      { sys: "mlx", name: "MLX fp16", ms: 3.44 },
      { sys: "llama", name: "llama.cpp, text model alone", ms: 4.11 },
      { sys: "pytorch", name: "PyTorch MPS fp32", ms: 27.29 },
    ],
  },

  // Qwen2.5-0.5B-Instruct, 512-token prompt, 128 greedy tokens after it
  llm: {
    provisional: false,
    decode: [
      { sys: "dooms", name: "dooms.cpp f16, own Metal decoder", tps: 137.9 },
      { sys: "mlx", name: "MLX-LM fp16", tps: 136.2 },
      { sys: "llama", name: "llama.cpp f16", tps: 130.7 },
      { sys: "pytorch", name: "PyTorch MPS fp16", tps: 37.6 },
    ],
    // 4-bit rows are a different precision: shown apart, never ranked against f16
    // Q4_0 (pure, 271 MB) is MLX-LM 4-bit's size (mlx-community, 265 MB); Q4_K_M (379 MB) is llama.cpp's default mix (scripts/reproduce-4bit.sh, 2026-10-22)
    decode4: [
      { sys: "dooms", name: "dooms.cpp Q4_0, own decoder", tps: 256.1 },
      { sys: "llama", name: "llama.cpp Q4_0", tps: 238.7 },
      { sys: "mlx", name: "MLX-LM 4-bit (mlx-community)", tps: 224.3 },
      { sys: "dooms", name: "dooms.cpp Q4_K_M, own decoder", tps: 209.2 },
      { sys: "llama", name: "llama.cpp Q4_K_M", tps: 165.3 },
    ],
    prefill: [
      { sys: "llama", name: "llama.cpp f16", tps: 4604 },
      { sys: "dooms", name: "dooms.cpp f16, own Metal decoder", tps: 3999 },
      { sys: "mlx", name: "MLX-LM fp16", tps: 3864 },
      { sys: "pytorch", name: "PyTorch MPS fp16", tps: 2492 },
    ],
  },

  // Whisper speech to text: the 73 LibriSpeech clips (481 s of audio, one call per clip) and the same audio as one 8-minute file, greedy, f16; seconds for all of it
  speech: {
    turboClips: [
      { sys: "dooms", name: "dooms.cpp, own Metal decoder", s: 53.61 },
      { sys: "mlx", name: "MLX-Whisper fp16", s: 53.62 },
      { sys: "whispercpp", name: "whisper.cpp, Metal", s: 63.77 },
      { sys: "pytorch", name: "PyTorch MPS fp16", s: 104.97 },
      { sys: "sapient", name: "Sapient, Metal", s: 948.44 },
    ],
    turboLong: [
      { sys: "dooms", name: "dooms.cpp, own Metal decoder", s: 15.48 },
      { sys: "mlx", name: "MLX-Whisper fp16", s: 15.48 },
      { sys: "whispercpp", name: "whisper.cpp, Metal (one window loops)", s: 21.88 },
      { sys: "pytorch", name: "PyTorch MPS fp16", s: 41.91 },
      { sys: "sapient", name: "Sapient, Metal", s: 241.16 },
    ],
    tinyClips: [
      { sys: "dooms", name: "dooms.cpp, own Metal decoder", s: 2.81 },
      { sys: "mlx", name: "MLX-Whisper fp16", s: 3.03 },
      { sys: "whispercpp", name: "whisper.cpp, Metal", s: 5.05 },
      { sys: "pytorch", name: "PyTorch MPS fp16", s: 15.12 },
      { sys: "sapient", name: "Sapient, Metal", s: 68.57 },
    ],
  },

  // Kokoro-82M text to speech: 73 sentences (404 s of speech), same phonemes and voice for every engine; seconds for all of them (bench/reports/2026-10-18-tts)
  tts: [
    { sys: "dooms", name: "dooms.cpp, own Metal vocoder", s: 15.08 },
    { sys: "mlx", name: "MLX fp32 (mlx-audio)", s: 23.45 },
    { sys: "pytorch", name: "PyTorch MPS fp32", s: 33.27 },
    { sys: "pytorch", name: "PyTorch CPU fp32", s: 46.66 },
    { sys: "onnx", name: "ONNX Runtime CPU fp32", s: 85.96 },
  ],

  // A 631M-parameter ViT backbone (1280 wide, 32 blocks), one forward on the GPU only
  vit: [
    { sys: "dooms", name: "dooms.cpp f16 matrices", ms: 101.8 },
    { sys: "mlx", name: "MLX fp16", ms: 103.6 },
    { sys: "pytorch", name: "PyTorch MPS fp16", ms: 105.1 },
    { sys: "dooms", name: "dooms.cpp f32", ms: 106.8 },
    { sys: "pytorch", name: "PyTorch MPS fp32", ms: 123.5 },
  ],

  // what changed, measured before and after
  log: [
    {
      title: "Our own text decoder",
      body: "Hand-written Metal kernels for llama and Qwen2 models: five kernels per layer instead of about fifteen, the greedy token picked on the GPU, and the next step reading it from GPU memory, so decode steps queue without the CPU in between.",
      before: 4.58, after: 2.70, unit: "ms/token", what: "SmolVLM decode",
    },
    {
      title: "Flash attention for the prompt",
      body: "The prompt's attention ran one query at a time. A flash-attention kernel on the matrix units (32 queries per threadgroup, a running softmax) removed most of it.",
      before: 72.6, after: 13.9, unit: "ms", what: "attention, 512-token prompt",
    },
    {
      title: "Prompt matmuls like llama.cpp's",
      body: "Our first prefill kernel used 16x16 tiles. Borrowing the structure of llama.cpp's kernel_mul_mm (64x32 tiles, 8x8-blocked operands, fully unrolled) plus our own fused norms and epilogues more than halved the prompt.",
      before: 315, after: 128, unit: "ms", what: "Qwen2.5 512-token prefill",
    },
    {
      title: "A persistent CPU thread pool",
      body: "Without one, ggml starts and joins its worker threads for every graph it computes, which is every generated token. Attaching a pool to the llama.cpp context removed that.",
      before: 70.9, after: 119.4, unit: "tokens/s", what: "Qwen2.5-0.5B decode",
    },
    {
      title: "A vectorized argmax",
      body: "Greedy decoding scans 151,936 logits per token. The scalar loop cost about 0.6 ms; a NEON version with first-index tie breaking brought decode level with llama-bench.",
      before: 119.4, after: 129.0, unit: "tokens/s", what: "Qwen2.5-0.5B decode",
    },
    {
      title: "Fused q/k/v projections",
      body: "llama.cpp loads a single attn_qkv tensor when a GGUF has one. Merging the three matrices at conversion time removes 90 dispatches per token.",
      before: 4.72, after: 4.55, unit: "ms/token", what: "SmolVLM decode",
    },
    {
      title: "Ten flow steps in one Core ML call",
      body: "The action expert ran as ten separate Neural Engine calls. Unrolling the flow into one graph, with the cache projections hoisted, cut the per-call overhead.",
      before: 80.6, after: 59.0, unit: "ms", what: "SmolVLA action expert",
    },
    {
      title: "A hybrid we threw away",
      body: "Neural Engine for the linear layers, GPU for attention, cameras pipelined across both. Correct, and slower: 75 extra calls at 3 to 5 ms of fixed cost each.",
      before: 183, after: 308, unit: "ms", what: "three vision towers", worse: true,
    },
    {
      title: "The benchmark that lied",
      body: "macOS was set to Low Power Mode only on the power adapter. A whole report ran throttled and showed the GPU backbone 8% behind MLX. Rerun with it off, it was ahead. The harness now refuses to run in Low Power Mode.",
      before: 112.5, after: 101.8, unit: "ms", what: "ViT backbone, GPU",
    },
  ],
};
