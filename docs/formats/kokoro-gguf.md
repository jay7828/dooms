# Kokoro-82M files for the engine

Written by `tools/convert/convert_kokoro.py` from the original checkpoint (hexgrad/Kokoro-82M: `config.json`, `kokoro-v1_0.pth`, `voices/*.pt`; Apache-2.0).
Three files in `<model>/dooms/`; the split and its numpy reference are in `tools/convert/kokoro_model.py`.

## prosody-fp32.mlpackage (Core ML, ML program, macOS 14+, fp32; run on the CPU)
- inputs `input_ids` float32 `[1, T]` (the integer token ids as floats, with the 0 pads at both ends; 3 <= T <= 512), `style` float32 `[1, 128]` (the voice row's second half)
- outputs `durations` `[1, T]` (sigmoid sums: frames per token before rounding), `d` `[1, T, 640]` (prosody features with the style appended), `t_en` `[1, 512, T]` (text features)
- ALBERT runs on its own submodules (one unpadded sequence: no mask). The converter checks it against PyTorch on 9, 64 and 300 tokens (2e-3 limit; 1.2e-5 measured).

## f0n-fp32.mlpackage (same format)
- inputs `en` float32 `[1, 640, F]` (d expanded to frames), `style` `[1, 128]`; outputs `f0`, `n` `[1, 2F]` (pitch and energy at twice the frame rate). Checked on 7, 200, 1500 frames.

## kokoro.gguf (`general.architecture = "kokoro"`)
Metadata (all `kokoro.` prefixed):

| key | type | meaning |
|---|---|---|
| `sample_rate`, `n_fft`, `hop`, `upsample`, `harmonics` | u32 | 24000; the generator's iSTFT (20, 5); F0 samples per curve value (300); fundamental + overtones of the sine source (9) |
| `sine_amp`, `noise_std`, `voiced_threshold` | f32 | the sine source (0.1, 0.003, 10 Hz) |
| `style_dim`, `hidden_dim`, `max_tokens`, `max_frames`, `max_duration` | u32 | 128, 512, 512 (ALBERT's positions), 16384 (the packages' frame limit), 50 |
| `upsample_rates`, `upsample_kernel_sizes`, `resblock_kernel_sizes` | i32[] | the generator (10, 6), (20, 12), (3, 7, 11) |
| `upsample_initial_channel` | u32 | 512 |
| `vocab.symbols`, `vocab.ids` | string[], i32[] | one phoneme symbol (one Unicode code point) -> token id |
| `voices` | string[] | the voices in the file |

Tensors:
- `dec.<module path>`: every tensor of the vocoder (`KModel.decoder`: the StyleTTS 2 decoder and the iSTFTNet generator) with its PyTorch name and layout, weight
  norm folded into plain weights; 3-D convolution kernels (`[out][in][k]`, transposed convolutions `[in][out][k]`) f16, everything else f32 (Snake alphas `[1][C][1]`).
- `source.linear.weight` f32 `[1][9]`, `source.linear.bias` `[1]`: the sine source's merge layer (applied on the host).
- `voice.<name>` f32 `[510][256]`: style rows; a call with n phonemes uses row n - 1 (first half: the vocoder's style, second half: prosody's).
