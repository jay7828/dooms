# Whisper files for the engine

Written by `tools/convert/convert_whisper.py` from a Hugging Face Whisper snapshot (e.g. `openai/whisper-large-v3-turbo`). Two files in `<model>/dooms/`:

## encoder-fp16.mlpackage (Core ML, ML program, macOS 14+)
- input `input_features`: float32 `[1, n_mels, 2 * n_audio_ctx]` (log-mel features of one 30-s window)
- output `encoder_out`: float32 `[1, n_audio_ctx, n_audio_state]` (the encoder's last hidden state, final layer norm included)
- weights and compute fp16; the converter refuses to write it when its output on the check units (default the GPU) differs from PyTorch fp32 on random features by more
  than `--encoder-max-rel-l2` (default 2e-2; large-v3-turbo measures 2.28e-2). The engine runs it on the GPU by default: on the Neural Engine large-v3-turbo's error is ~10x larger.

## decoder.gguf (`general.architecture = "whisper"`)
Metadata (all `whisper.` prefixed):

| key | type | meaning |
|---|---|---|
| `n_mels`, `n_audio_ctx`, `n_audio_state` | u32 | encoder input bins, positions per window (1500), width |
| `n_text_ctx`, `n_text_state`, `n_text_head`, `n_text_layer`, `n_text_ff`, `n_vocab` | u32 | decoder sizes (`n_text_ctx` = learned positions, 448) |
| `layer_norm_eps` | f32 | every LayerNorm |
| `sample_rate`, `n_fft`, `hop_length`, `chunk_samples` | u32 | 16000, 400, 160, 480000 |
| `token.sot`, `token.eot`, `token.transcribe`, `token.translate`, `token.notimestamps`, `token.nospeech`, `token.timestamp_begin` | u32 | special token ids |
| `n_text_tokens` | u32 | ids below are text pieces; ids from here on are special (no text) |
| `language.codes`, `language.ids` | string[], i32[] | `"en"` -> `<|en|>` id, ... (from the generation config's `lang_to_id`) |
| `suppress_tokens`, `begin_suppress_tokens` | i32[] | never generated; not the first generated token |
| `tokens` | string[n_vocab] | id -> piece (GPT-2 byte-level: each code point stands for one byte; special tokens as their names) |

Tensors (row-major, outermost first; matrices f16 `[out][in]`, vectors f32):
- `mel_filters` f32 `[n_mels][n_fft/2 + 1]` (Slaney mel filter bank, the feature extractor's own)
- `token_embd.weight` `[n_vocab][n_state]` (also the output projection: tied), `pos_embd.weight` f32 `[n_text_ctx][n_state]`, `ln.weight`, `ln.bias`
- per layer `blk.N.`: `attn_ln`, `cross_ln`, `mlp_ln` (`.weight`, `.bias`); `attn_q`, `attn_k`, `attn_v`, `attn_out`, `cross_q`, `cross_k`, `cross_v`, `cross_out`
  (`.weight`; `.bias` for q, v and out: Whisper's k projections have none); `fc1`, `fc2` (`.weight`, `.bias`). 24 tensors per layer.

## Decoding as the engine runs it
Prompt `<|startoftranscript|> <|lang|> <|task|> <|notimestamps|>`, greedy, `suppress_tokens` always masked, `begin_suppress_tokens` also masked for the first
generated token, up to 224 tokens or `<|endoftext|>` per 30-s window. Matches transformers' `generate(language=..., task=..., return_timestamps=False,
num_beams=1)` token for token on the reference clips (`tests/unit/test_whisper.cpp`).
