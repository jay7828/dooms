// Every number on the site comes from this file, and every value here comes from a benchmark report in the repository.
// Models: bench/reports/2026-10-15-all-models/report.md (3 interleaved rounds, AC power, Low Power Mode off, load 1.3-1.5).
// LLM: bench/reports/2026-10-15-llm/report.md once the gated run lands; until then the single-round measurements of 2026-10-15 (marked `provisional`).
// Machine: Apple M1 Pro (6 performance + 2 efficiency cores, 14-core GPU, 16 GB), macOS 26.6.

window.DOOMS = {
  machine: "Apple M1 Pro, 16 GB, macOS 26.6",
  measured: "15 October 2026",

  systems: {
    dooms: { label: "dooms.cpp", slot: 1 },
    pytorch: { label: "PyTorch", slot: 2 },
    mlx: { label: "MLX", slot: 3 },
    llama: { label: "llama.cpp", slot: 4 },
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
      { sys: "dooms", name: "dooms.cpp", ms: 92.5 },
      { sys: "mlx", name: "MLX fp16", ms: 97.3 },
      { sys: "pytorch", name: "PyTorch MPS fp16", ms: 139.4 },
    ],
    decode: [
      { sys: "mlx", name: "MLX fp16", ms: 3.41 },
      { sys: "llama", name: "llama.cpp, text model alone", ms: 4.12 },
      { sys: "dooms", name: "dooms.cpp", ms: 4.58 },
      { sys: "pytorch", name: "PyTorch CPU fp32", ms: 29.66 },
    ],
  },

  // Qwen2.5-0.5B-Instruct, 512-token prompt, 128 greedy tokens after it
  llm: {
    provisional: true,
    decode: [
      { sys: "mlx", name: "MLX-LM 4-bit", tps: 212.4 },
      { sys: "llama", name: "llama.cpp Q4_K_M", tps: 164.2 },
      { sys: "mlx", name: "MLX-LM fp16", tps: 136.3 },
      { sys: "llama", name: "llama.cpp f16", tps: 130.4 },
      { sys: "dooms", name: "dooms.cpp f16", tps: 129.0 },
      { sys: "pytorch", name: "PyTorch MPS fp16", tps: 37.5 },
    ],
    prefill: [
      { sys: "llama", name: "llama.cpp f16", tps: 4606 },
      { sys: "dooms", name: "dooms.cpp f16", tps: 4493 },
      { sys: "mlx", name: "MLX-LM fp16", tps: 3881 },
      { sys: "pytorch", name: "PyTorch MPS fp16", tps: 2508 },
    ],
  },

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
