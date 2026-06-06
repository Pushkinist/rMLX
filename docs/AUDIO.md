# rMLX Audio — Whisper STT + Qwen3-TTS

Architecture and implementation notes for the audio subsystem (`crates/rmlx-audio`).
See `docs/SERVER.md` for HTTP API shape.

## Whisper STT (`crates/rmlx-audio/src/whisper.rs`)

### Architecture

Whisper large-v3 (`mlx-community` snapshot, weights in `weights.npz`).

| Config key | Value |
|---|---|
| `n_mels` | 128 |
| `n_audio_ctx` | 1500 |
| `n_audio_state` | 1280 |
| `n_audio_head` | 20 |
| `n_audio_layer` | 32 |
| `n_vocab` | 51866 |
| `n_text_ctx` | 448 |
| `n_text_state` | 1280 |

### Pipeline

```
audio bytes → WavDecoder (16 kHz mono f32) → MelExtractor (128 mel, 30 s window)
→ WhisperModel::encode_mel() → encoder output [1, 1500, 1280]
  [optional: detect_language()]
→ WhisperModel::greedy_decode() → token ids
→ WhisperTokenizer::decode() → text
```

### Language auto-detection

`WhisperModel::detect_language(encoder_out, device)`:
1. Runs a single SOT-only decoder step.
2. Takes argmax over language tokens 50259–50357 (99 languages).
3. Returns the winning language token id.
4. Falls back to 50259 (English) on any error.

Called when `language` field is absent or `"auto"` in the transcription request.
The returned token id is passed directly to `WhisperTokenizer::sot_sequence_from_tok()`.

### VAD chunking

Audio longer than 30 s (480 000 samples) is split using Silero VAD v4
(weights vendored at `crates/rmlx-audio/assets/silero_vad_16k.safetensors`).
Each voiced segment is transcribed independently; transcripts are joined with spaces.

---

## Qwen3-TTS (`crates/rmlx-audio/src/tts.rs`)

Full synthesis pipeline — text → talker → codec → 24 kHz PCM.
Entry point: `pub fn synthesize(text, voice, model, tokenizer) -> Result<(Vec<f32>, u32)>`.

### Stage 1: Talker

`mlx-community__Qwen3-TTS-*-CustomVoice-8bit` snapshot, `talker.*` weights.

| Hyperparameter | Value |
|---|---|
| Layers | 28 |
| Hidden | 2048 |
| Heads | 16 |
| KV heads | 8 |
| Head dim | 128 |
| MLP intermediate | 11008 |
| Quantization | Affine 8-bit, group_size=64 |

Key blocks:
- **text_projection**: fc1+fc2 — maps text embeddings into talker hidden space.
- **Attention**: GQA (n_groups=2), per-head Q/K RMSNorm, MRoPE (sequential positions for text-only input).
- **KV cache**: slice-concatenation per step (no quantization — short TTS sequences).
- **codec_head**: affine-8bit LM head → audio token vocabulary.

### Stage 1: CodePredictor

5-layer mini-Qwen3 (hidden=1024, 16 heads, head_dim=64). Generates codec groups 1..15 per step.

- Input: talker hidden last position + codec embedding.
- Output: argmax over 2048-token vocab for each of 15 codec groups.

### Stage 2: Codec decoder

`Qwen__Qwen3-TTS-Tokenizer-12Hz` snapshot, `decoder.*` weights.

Pipeline:

```
code tokens [1, 16, T] (16 groups × T steps)
→ SplitRVQ: 1 rvq_first (semantic) + 15 rvq_rest (acoustic) codebooks
  each codebook: 2048×256 embeddings, VQ-normalized
→ sum + output_proj → [1, T, 512]
→ pre_conv (causal, k=3) → [1, T, 1024]
→ pre_transformer: input_proj → 8-layer attn+MLP (layer-scale) → norm → output_proj
→ 2× ConvNeXt upsample (ConvTranspose1d stride=2 + ConvNeXt block)
→ initial_conv (causal, k=7, 1024→1536)
→ 4× ResNet decoder group (strides 8,5,4,3, SnakeBeta activations)
→ output_snake + output_conv (k=7)
→ tanh → 24 kHz mono f32
```

Token rate: 12.5 Hz. Total upsample: 2×2×8×5×4×3 = 1920×. Output: 12.5 Hz × 1920 = 24000 Hz.

### SnakeBeta

`x + (1/exp(β)) * sin²(exp(α)·x)` — learned periodic activation in ResNet and final output.

### VQ normalization

`embedding = embedding_sum / max(cluster_usage, 1.0)` — applied on load.

### Conv weight layout

| Layer type | PyTorch storage | MLX layout | Transpose applied |
|---|---|---|---|
| Conv1d | `[out, in, kernel]` | `[out, kernel, in]` | `axes [0, 2, 1]` |
| ConvTranspose1d | `[in, out, kernel]` | `[out, kernel, in]` | `axes [1, 2, 0]` |

### Lazy weight loading

`TtsModel::load_config()` reads only `config.json` (fast, called at server start).
Weights are loaded on the first `synthesize()` call. The server caches `Arc<Mutex<TtsModel>>`
and locks it for each synthesis call.

---

## WAV I/O

### Decoder (`WavDecoder::decode`)

Symphonia-backed. Returns `(Vec<f32>, sample_rate)`. Caller must validate `sample_rate == 16_000`.

### Encoder (`WavEncoder::encode`)

44-byte RIFF PCM-16 LE header + samples. `f32 [-1, 1]` → `i16` with clamping and rounding.

---

## Token constants (Whisper)

| Constant | Value | Meaning |
|---|---|---|
| `TOK_EOT` | 50257 | `<\|endoftext\|>` |
| `TOK_SOT` | 50258 | `<\|startoftranscript\|>` |
| `TOK_EN` | 50259 | `<\|en\|>` (first language token) |
| `TOK_TRANSLATE` | 50358 | `<\|translate\|>` |
| `TOK_TRANSCRIBE` | 50359 | `<\|transcribe\|>` |
| `TOK_NOSPEECH` | 50362 | `<\|nospeech\|>` |
| `TOK_NO_TIMESTAMPS` | 50363 | `<\|notimestamps\|>` |

Language tokens span `[50259, 50358)` — 99 languages in alphabetical order.
