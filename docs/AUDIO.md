# rMLX Audio — Whisper STT + Qwen3-TTS

How `crates/rmlx-audio` transcribes and synthesizes. The HTTP routes, their
fields and their status codes are in `docs/SERVER.md`.

## Whisper STT

### Model

`WhisperModel::load` reads `config.json` and `weights.npz` from an
`mlx-community` Whisper snapshot (`crates/rmlx-audio/src/whisper.rs`). Every
dimension comes from `config.json`. The mel filterbank matches its `n_mels`:
80 or 128. The special-token ids are large-v3's, hard-coded (see "Token ids").

### Pipeline

`rmlx_audio::transcribe::Transcriber` is the one transcription path. The
`/v1/audio/transcriptions` and `/v1/audio/translations` routes and
`rmlx transcribe` all call it.

```
audio bytes → WavDecoder (mono f32, native rate) → resample_to_16k
→ per 30 s window: MelExtractor → encode_mel → greedy decode in timestamp mode
→ timestamp tokens → segments with cumulative times
→ seek to the last timestamp; previous text fed back after <|startofprev|>
```

Each window decodes with the openai-whisper logit filters (`DecodeFilters`).
Temperature is 0, so one input gives one output. A segment that opens in the
zero-padded tail of the last window is dropped.

Long audio is walked window by window. No voice-activity detection runs.

### Language detection

With `language` absent or `"auto"`, `WhisperModel::detect_language` runs on
the first window. It takes one SOT-only decoder step and the argmax over the
100 language tokens. On error it falls back to English (`TOK_EN`).

### Silero VAD

The Silero VAD v4 weights are vendored at
`crates/rmlx-audio/assets/silero_vad_16k.safetensors` (MIT, see `NOTICE`
beside them). `rmlx_audio::vad` is exported, and nothing calls it.
`crates/rmlx-audio/src/transcript.rs`, the one file that uses it, is not
declared in `lib.rs` and is not compiled.

### Token ids

From `crates/rmlx-audio/src/tokenizer.rs`:

| Constant | Id | Token |
|---|---|---|
| `TOK_EOT` | 50257 | `<\|endoftext\|>` |
| `TOK_SOT` | 50258 | `<\|startoftranscript\|>` |
| `TOK_EN` | 50259 | `<\|en\|>`, the first language token |
| `TOK_LANG_LAST` | 50358 | the last language token |
| `TOK_TRANSLATE` | 50359 | `<\|translate\|>` |
| `TOK_TRANSCRIBE` | 50360 | `<\|transcribe\|>` |
| `TOK_SOT_LM` | 50361 | `<\|startoflm\|>` |
| `TOK_SOT_PREV` | 50362 | `<\|startofprev\|>` |
| `TOK_NOSPEECH` | 50363 | `<\|nospeech\|>` |
| `TOK_NO_TIMESTAMPS` | 50364 | `<\|notimestamps\|>` |
| `TOK_TIMESTAMP_BEGIN` | 50365 | `<\|0.00\|>`; each step is 0.02 s |

The language tokens are 50259 to 50358 inclusive: 100 languages.

---

## Qwen3-TTS

`rmlx_audio::tts::synthesize(text, voice, model, tokenizer)` returns mono f32
PCM at 24 kHz (`crates/rmlx-audio/src/tts.rs`). `--tts-model-path` names the
talker snapshot (`mlx-community__Qwen3-TTS-*-CustomVoice-8bit`), which also
holds the text tokenizer. `--tts-tokenizer-path` names the codec decoder
snapshot (`Qwen__Qwen3-TTS-Tokenizer-12Hz`).

### Loading

`TtsModel::load_config` reads only the talker's `config.json`. The weights
load on the first `synthesize` call. The server keeps the model for the life
of the process.

Every quantized linear loads as affine 8-bit, group size 64. The loader does
not read the snapshot's quantization config.

### Talker

A Qwen3 transformer over the `talker.*` weights. Its dimensions come from
`config.json`.

- `text_projection` (fc1, fc2) maps text embeddings into the talker's hidden
  space.
- Attention: grouped-query, per-head Q/K RMSNorm, RoPE over sequential
  positions.
- The KV cache grows by concatenation each step, unquantized.
- `codec_head` predicts codec group 0. Generation stops at the codec EOS or
  after 2048 steps.

### CodePredictor

A 5-layer Qwen3 stack that predicts codec groups 1 to 15 from the talker's
last hidden state and group 0. Its hidden size and layer count come from
`config.json`. Heads (16), KV heads (8), head dim (128) and MLP width (3072)
are constants in `load_talker_weights`. Its KV cache resets every step.

### Codec decoder

The `decoder.*` weights of the codec snapshot turn codes into audio:

```
codes [1, 16, T] (16 groups × T steps)
→ SplitRVQ: 1 semantic + 15 acoustic codebooks, each 2048 × 256
→ sum + output_proj → [1, T, 512]
→ pre_conv (causal, k=3) → [1, T, 1024]
→ pre_transformer: input_proj → 8 layers (layer scale) → norm → output_proj
→ 2× (ConvTranspose1d stride 2 + ConvNeXt block)
→ initial_conv (causal, k=7, 1024 → 1536)
→ 4 decoder groups (strides 8, 5, 4, 3; SnakeBeta)
→ output_snake + output_conv (k=7) → tanh → 24 kHz mono f32
```

The codes run at 12.5 Hz. The total upsample is 2·2·8·5·4·3 = 1920, and
12.5 Hz × 1920 = 24 kHz.

- **SnakeBeta**: `x + (1 / exp(β)) · sin²(exp(α) · x)`.
- **Codebook**: `embedding_sum / max(cluster_usage, 1e-5)`, applied on load.
- **Conv weights** are stored in PyTorch layout and transposed on load:
  Conv1d `[out, in, k]` by axes `[0, 2, 1]`, ConvTranspose1d `[in, out, k]`
  by axes `[1, 2, 0]`, both to MLX `[out, k, in]`.

---

## WAV I/O

`WavDecoder::decode` is Symphonia-backed. It returns mono f32 at the input's
own sample rate; the caller resamples to 16 kHz for Whisper.

`WavEncoder::encode` writes a 44-byte RIFF PCM-16 LE header, then the samples.
Each f32 in `[-1, 1]` is clamped and rounded to `i16`.
