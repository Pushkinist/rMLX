# rMLX Audio — Whisper STT + Qwen3-TTS

How `crates/rmlx-audio` transcribes and synthesizes. The speech-to-text
route's fields, bodies and status codes are in "HTTP route" below; its
admission, cancel and model cache, and the speech route, are in
`docs/SERVER_AUDIO.md`.

## Whisper STT

### Model

`WhisperModel::load` reads `config.json` and `weights.npz` from an
`mlx-community` Whisper snapshot (`crates/rmlx-audio/src/whisper.rs`). Every
dimension comes from `config.json`. The mel filterbank matches its `n_mels`:
80 or 128. The special-token ids are large-v3's (see "Token ids").

**Supported variants.** Only the large-v3 vocabulary loads. The load refuses
anything else, and the error names the file, the token or field, its value
and the value the decode expects:

| Variant | `n_vocab` | `n_mels` | Load |
|---|---|---|---|
| large-v3, large-v3-turbo, distil-large-v3 | 51866 | 128 | loads; only large-v3 has been decoded |
| tiny, base, small, medium, large-v1, large-v2 | 51865 | 80 | refused: `config.json: n_vocab is 51865` |
| tiny.en, base.en, small.en, medium.en | 51864 | 80 | refused: `config.json: n_vocab is 51864` |

The values are the `config.json` of each `mlx-community` snapshot. The
refused vocabularies have other special-token ids: v1, v2 and the smaller
multilingual models have 99 languages, so every token after `<|su|>` is one
id lower; the English-only models start at `<|endoftext|>` = 50256. Decoded
with the large-v3 ids, they would give wrong text and no error. Deriving the
ids from `tokenizer.json` would let them load; that is not built.

- `WhisperConfig::from_json` refuses an `n_vocab` other than 51866 and an
  `n_audio_ctx` other than 1500 (the encoder positions of one 30 s window).
  `WhisperModel::load` refuses before it reads `weights.npz`.
- `WhisperTokenizer::from_path` refuses a `tokenizer.json` in which one of the
  1609 special tokens (end of text, start of transcript, the 100 languages,
  the task and control tokens, the 1501 timestamps) is absent or has another
  id, or which holds another number of tokens than 51866. `rmlx transcribe`
  and the server load the tokenizer before the model, so this refusal costs
  no weight load.

The check reads ids and sizes only. A model or a tokenizer that states the
large-v3 ids but was trained on another vocabulary is not detected.

### Pipeline

`rmlx_audio::transcribe::Transcriber` is the one transcription path. The
`/v1/audio/transcriptions` and `/v1/audio/translations` routes and
`rmlx transcribe` all call it.

```
audio bytes → WavDecoder::decode_resampled (mono f32, resampled to 16 kHz as the packets decode; refuses a file that lost too much)
→ per 30 s window: MelExtractor → encode_mel → greedy decode in timestamp mode
→ timestamp tokens → segments with cumulative times
→ seek: the whole window, or the end of the last closed segment
```

With `condition_on_previous_text` set, the previous window's text is fed back
after `<|startofprev|>`, unless that text was a repetition loop (see
"Repetition guard"). The server and `rmlx transcribe` always set it.

Each window decodes with the openai-whisper logit filters (`DecodeFilters`).
Temperature is 0, so one input gives one output. The only filler guard drops
a segment that opens after the window's own length plus 0.5 s. The
openai-whisper reference has no such guard.

**Log-mel floor.** rMLX computes the log-mel of each 30 s window alone. Each
value is clamped to the window's largest value minus 8.0 (log10), then scaled
as `(x + 4) / 4`. openai-whisper, mlx-whisper, whisper.cpp and the sequential
path of faster-whisper compute the log-mel of the whole file once and clamp to
the file's largest value minus 8.0, so a quiet window loses more low-level
detail there. The batched path of faster-whisper clamps per voice segment. A
build with the file maximum was compared with the window maximum (large-v3,
greedy, one build twice per clip to show the noise: 0 words):

- Synthetic speech whose second part is 45 dB quieter than the first, 267
  words: 12 and 7 word errors with the file maximum (two input offsets), 3 and
  3 with the window maximum. The file floor removes the quiet part's weak
  sounds.
- A public meeting recording and three private recordings, 10 minutes each: the
  file maximum changed 13 to 357 words. With the window maximum, a 10 ms shift
  of the input changed 18 and 356 words on two of them. On these clips the
  effect is as large as that of any small change to the input. Against the
  machine transcripts of three of them, no clip had fewer errors with the file
  maximum.

The window maximum stays.

Long audio is walked window by window. No voice-activity detection runs.

**Seek rule.** It follows openai-whisper `transcribe()`, in samples:

- Tokens with a timestamp pair (two timestamps in a row) and a single
  timestamp at the end: no speech follows, so the whole real window is
  consumed.
- Tokens with a timestamp pair that do not end in a single timestamp: the
  unfinished segment is dropped, and the seek goes to the end of the last
  closed segment.
- Tokens with no timestamp pair: one segment from the window start, and the
  whole window is consumed.

The advance is never zero, and a window that is consumed whole advances by its
exact sample count. The walk stops when less than one mel hop (160 samples,
10 ms) of audio is left, so a few samples at the end of a clip open no window.
A window that ends in a timestamp pair still leaves real audio for the next
window, and that window can print filler.
No no-speech gate exists to skip a near-silent window. `temperature` has no
effect: every decode, the repetition retry below included, is greedy.

**Repetition guard.** Greedy decode under a previous-text prompt can loop: one
sentence repeated until the token budget, with no closing timestamp, so the
whole window is consumed and its speech is lost. The loop text then prompts the
next window, which loops too. A window's text whose zlib compression ratio is
above 2.4 (openai-whisper's `compression_ratio_threshold`) is a loop. When it
was decoded with a prompt, the window is decoded again with no prompt. Of the
two decodes, the one with the lower ratio is kept (the retry on a tie). If the
kept one is still a loop, it stays in the transcript, a `warn` event names the
window and both ratios, and it does not prompt the next window. The ratio is
the stripped text's bytes over a zlib stream, as openai-whisper's
`compression_ratio`; the encoder is miniz at level 6, not zlib, so a short text
near 2.4 can fall on the other side of the bound. openai-whisper re-decodes at temperatures 0.2 to 1.0 instead and drops
the prompt after a fallback above 0.5; whisper.cpp decodes its fallbacks at 0.5
and above with no prompt. rMLX has no sampler, so its one retry is greedy and
prompt-free.

**Cancel.** `Transcriber::transcribe` takes a cancel flag and reads it before
each window opens. Once the flag is set, no further window opens, and the call
returns `WhisperError::Cancelled` with the number and the `seek` of the first
window not opened. It returns no partial transcript. A flag set during the
last window has no effect.

Worst-case latency: the window in flight runs to its end, that is, one mel and
encode and at most two greedy decodes (the decode and the repetition retry),
each of at most `n_text_ctx / 2` tokens (224 for large-v3). Before window 0,
the caller's model load runs to its end. The audio decode before it asks the
caller before each packet and stops there (`WavError::Cancelled`); the server
passes the request's cancel flag to the decode and to the window walk.

The server sets the flag when the request is gone (see the speech-to-text
routes in `docs/SERVER_AUDIO.md`); `rmlx transcribe` never sets it.

### Language detection

With `language` absent or `"auto"`, `WhisperModel::detect_language` runs
once, on the first window. It takes one SOT-only decoder step and the argmax
over the 100 language tokens. An error of that step fails the request; there
is no fallback to English.

The language of the first window holds for the whole file, as in
openai-whisper and mlx-whisper, whose `transcribe()` also detects on the
first 30 s only. A later window in another language is decoded as the first
window's language, and its speech can come out translated into it. A first
window of silence, music or a short phrase in another language sets the
language of the file. Send `language` when it is known.

### Language forms

A request names a language as a code of the Whisper table (`fr`, `haw`,
`yue`) or as the English name (`french`), in any letter case, as `get_tokenizer` of openai-whisper and mlx-whisper
reads it. The twelve extra names of its `TO_LANGUAGE_CODE` (`castilian`, `flemish`,
`mandarin`, …) are taken too. Any other text is an error that names it: 422
on the route, a refusal before the audio is read in `rmlx transcribe`.

The reply names the language of the decode as the lower-case English name
(`french`), whether the request named it or the model detected it. This is the
form of the OpenAI reference (the `verbose_json` example has
`"language": "english"`) and of the `whisper.cpp` server
(`whisper_lang_str_full`). `openai-whisper` and `faster-whisper` give the
code instead; `vLLM` gives back the code of the request. The table of codes
and names is `tokenizer::LANGUAGES`, in the order of the vocabulary. The load
refuses every vocabulary except large-v3's, so one table serves every
checkpoint that loads.

With `auto` and audio shorter than one 10 ms mel hop, no window runs, so no
language is detected. The reply names `english` then, and a `warn` event is
logged.

### HTTP route

`POST /v1/audio/transcriptions` and `POST /v1/audio/translations` take the
multipart form of the OpenAI reference (`CreateTranscriptionRequest`);
`translations` translates to English.

| Field | Default | What the route does |
|---|---|---|
| `file` | required | WAV (PCM), MP3, M4A (AAC) or ADTS (AAC), at most 25 MiB. Downmixed to mono and resampled to 16 kHz. |
| `model` | `whisper-large-v3` | Logged only; a longer value is kept cut to 64 bytes. The configured snapshot serves every request. |
| `language` | `auto` | An ISO code or an English name, in any case, or `auto` to detect it on the first window (see "Language forms" and "Language detection"). Anything else is 422. |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt` or `vtt`. Anything else, `diarized_json` included, is 422. |
| `temperature` | `0.0` | A non-finite or negative value is 422. Any other value is taken and has no effect: every decode is greedy. |
| `timestamp_granularities[]` | `segment` | `segment` is taken: every response is timed per segment. `word` is 422: word timestamps need the alignment heads and a DTW pass, which are not built. The name without `[]` is read the same way. |
| `stream` | `false` | Read in any case. `false`, `0` or empty is taken. `true` or `1` is 422: the transcription is sent whole when it ends. Any other value is 422. |
| `prompt` | — | Taken and ignored. The decoder's prompt is the previous window's text only. |
| any other | — | Taken and ignored (`include[]`, `chunking_strategy`, …). |

A field the route reads, other than `file`, holds at most 64 bytes: a longer
value is 422, `<field> is over 64 bytes`, and the refusal does not echo it.
An ignored field is read and dropped as it arrives; it is never held. At
most one `warn` event per request, `form fields ignored`, gives their count
and the first eight names, each cut to 64 bytes; there is none when the form
is refused. A form of more than 64 parts, the file included, is 422.

OpenAI ignores `stream` for `whisper-1`; the route refuses it, so a client
that waits for events gets an error and not a body it cannot read as events.
`prompt` is ignored, not refused: it guides the style of the text, and a
client that sends it still gets a transcript.

| `response_format` | Body |
|---|---|
| `json` | `{"text": "..."}` |
| `text` | Plain text. |
| `verbose_json` | `{"task","language","duration","text","segments":[…]}`, no `words`. |
| `srt` | SRT, one cue per segment. |
| `vtt` | WebVTT, one cue per segment. |

A `verbose_json` segment holds eight of the ten keys the OpenAI
`TranscriptionSegment` schema requires, each with a value as openai-whisper
`transcribe()` gives it:

| Key | Value |
|---|---|
| `id`, `start`, `end`, `text` | The segment. |
| `seek` | The start of the 30 s window that holds the segment, in 10 ms mel frames. |
| `tokens` | The segment's slice of the window's tokens, its timestamp tokens included (large-v3 ids). |
| `temperature` | 0: every decode is greedy. |
| `compression_ratio` | The window's: the ratio of the decode the repetition guard kept, the same for each segment of the window. |

`avg_logprob` and `no_speech_prob` are left out: no log-probability is kept
and no no-speech gate runs. A `null` in their place would make a typed client
throw (openai-dotnet reads them as numbers). A client that requires the two
cannot read rMLX's `verbose_json` until they exist:

- `@ai-sdk/openai` (the Vercel AI SDK): it asks for `verbose_json` itself when
  the model is `whisper-1`, and its schema requires both as numbers;
- openai-python with `_strict_response_validation`.

Clients known to read it: openai-python by default (`None` for the two) and
openai-dotnet (0 for the two). `language` is the English name of the language
of the decode, for example `french` (see "Language forms").

An error body is `{"error": "…"}`, except the 408 and the 429, which are
the OpenAI envelope `{"error":{"message","type"}}`, and the 400, which is
axum's plain text.

| Status | When |
|---|---|
| 400 | A `multipart/form-data` request with no boundary: axum refuses it before the route runs, ``Invalid `boundary` for `multipart/form-data` request``. |
| 413 | The file is over 25 MiB (26,214,400 bytes): `audio file is <N> bytes, over the cap of 26214400 bytes (25 MiB)`. A request body over 26 MiB (27,262,976 bytes) stops while it is read: `request body is <N> bytes, over the limit of 27262976 bytes: …`, without `<N> bytes,` when the request sent no `Content-Length`. OpenAI answers 413 for a request over 26,214,400 bytes. |
| 422 | A malformed form, more than 64 parts, no `file` or an empty one, a field value over 64 bytes, a refused field value, or a file the decoder refuses (`audio decode: …`, see "WAV I/O"). |
| 408 | The request timeout expired (`docs/SERVER.md` § "Error responses"). |
| 429 | The admission queue is full. |
| 503 | The server's fault. With `x-should-retry: false`, which OpenAI's SDKs obey before their own retry of a 5xx: `--whisper-model-path` or `--whisper-tokenizer-path` is not set; the tokenizer, `config.json` or `weights.npz` does not load or is refused (see "Supported variants"); or the transcriber cannot be built on the loaded model (`transcriber init: …`). The body names the file and the fact, never a directory; the server log gives the path. Without that header: an engine failure during the transcription (`transcribe: …`), which a retry may pass. |
| 500 | The job panicked. |

A file that loses audio under the limit for its length is not refused, and
no response field shows it: the transcript has a gap or ends early, and only
the `warn` event `audio lost under the limit` in the server log gives the
seconds and the positions.

### Silero VAD

The Silero VAD v4 weights are vendored at
`crates/rmlx-audio/assets/silero_vad_16k.safetensors` (MIT, see `NOTICE`
beside them). `rmlx_audio::vad` is exported, and nothing calls it.

### Token ids

From `crates/rmlx-audio/src/tokenizer.rs`. Each id is the one after the
previous, in openai-whisper's `get_encoding` order, from `TOK_EOT` = 50257;
the vocabulary is `N_VOCAB` = 51866 tokens.

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

The language tokens are 50259 to 50358 inclusive: 100 languages, in the order
of `LANGUAGES`. Timestamps run to `<|30.00|>` = 51865.

---

## Qwen3-TTS

`rmlx_audio::tts::synthesize(text, voice, model, tokenizer, pick)` returns a
`Synthesis`: mono f32 PCM at 24 kHz, the sample rate, the number of codec
frames, and a truncation flag (`crates/rmlx-audio/src/tts.rs`). `pick` chooses
every codec token (see "Sampling"). `--tts-model-path` names the
talker snapshot (`mlx-community__Qwen3-TTS-*-CustomVoice-8bit`), which also
holds the text tokenizer. `--tts-tokenizer-path` names the codec decoder
snapshot (`Qwen__Qwen3-TTS-Tokenizer-12Hz`).

`synthesize` opens a span with `text_len` and, once `voice` names a speaker of
the model, `voice`. No span field and no event of `synthesize` or of a
function it calls holds the text of a request or a voice that names no
speaker. The speech route logs more: the voice of every request, cut to 64
bytes.

### Loading

`TtsModel::load_config` reads only the talker's `config.json`. The weights
load on the first `synthesize` call. When the server loads the model, it also
reads the talker snapshot's `generation_config.json` once, for the sampling
defaults. The server keeps both for the life of the process.

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
- `codec_head` predicts codec group 0.

### CodePredictor

A 5-layer Qwen3 stack that predicts codec groups 1 to 15 from the talker's
last hidden state and group 0. None of its dimensions comes from the
snapshot. Hidden size (1024) and layer count (5) are serde defaults of flat
`talker_config` keys; the nested `code_predictor_config` block is not read.
Heads (16), KV heads (8), head dim (128) and MLP width (3072) are constants in
`load_talker_weights`. Its KV cache resets every frame.

### Generation loop

`run_frames` calls `frame_step` once for each frame, at most `MAX_FRAMES`
(2048) times. 2048 frames are 163.84 s of audio. One frame:

1. The talker picks group 0 from its last position, under
   `talker_allow_mask`. The mask blocks the last 1024 ids of the codec
   vocabulary, except the codec end token (`codec_eos_token_id`). Before
   `MIN_FRAMES` (2) frames exist, the mask also blocks the end token. A
   pick that the mask blocks is an error.
2. If the talker picks the end token, the loop stops. The end token is not a
   frame.
3. Otherwise the code predictor picks groups 1 to 15. Each pick reads the
   previous code. These picks have no mask and no history. A pick outside
   the vocabulary of its row is an error.
4. The group-0 code goes into the history. The talker's repetition penalty
   reads the full history.

If the loop reaches the cap before the end token, `Synthesis::truncated` is
`true`, and the audio is cut off. The speech route then sets the
`x-rmlx-tts-truncated: true` header and logs a `warn` event with the frame
count and the cap (`docs/SERVER_AUDIO.md`, "Text-to-speech").

### Sampling

`synthesize` does not choose tokens. The caller gives a `TokenPick`. The
speech route gives `TtsPicker` (`crates/rmlx-server/src/tts_sampling.rs`),
which calls the shared sampler, `rmlx_models::sampler::select_token`
(`docs/SAMPLING.md`).

Each value resolves key by key: the request field, then the key in the talker
snapshot's `generation_config.json`, then the Qwen3-TTS reference default.

| Head | Value | Request field | File key | Reference default |
|---|---|---|---|---|
| Talker | temperature | `temperature` | `do_sample: false` gives 0; else `temperature` | 0.9 |
| Talker | top-k | `top_k` | `top_k` | 50 |
| Talker | top-p | — | `top_p` | 1.0 |
| Talker | repetition penalty | `repetition_penalty` | `repetition_penalty` | 1.05 |
| Code predictor | temperature | `temperature` | `subtalker_dosample: false` gives 0; else `subtalker_temperature` | 0.9 |
| Code predictor | top-k | `top_k` | `subtalker_top_k` | 50 |
| Code predictor | top-p | — | `subtalker_top_p` | 1.0 |
| Code predictor | repetition penalty | — | — | none (1.0) |

- A request `temperature` wins over `do_sample: false`.
- Temperature 0 is greedy decoding. A greedy talker keeps its repetition
  penalty. So with the checkpoint's file, `temperature: 0` alone is greedy
  with penalty 1.05. Greedy decoding with no penalty needs `temperature: 0`
  and `repetition_penalty: 1.0`.
- `seed` seeds one PCG32 stream for the request. Both heads draw from it in
  pick order, one draw for each sampled pick. With no `seed`, the stream
  starts at the shared sampler's default seed, `0xA7A7`. So a request
  without a seed gives the same audio each time for the same model, input,
  voice and fields.
- The reference default is used for a key that the file does not have, and
  when the snapshot has no `generation_config.json`.
- Bounds. `temperature` is 0 or in `[1e-6, 2]`. `repetition_penalty` is
  finite and at least `1e-6`. `top_p` is in `[0, 1]`. The upper ends are those
  of `/v1/chat/completions`. The floor of `1e-6` and the refusal of an
  infinite `repetition_penalty` belong to this route: the sampler divides a
  logit by the temperature, and divides or multiplies it by the penalty, and
  a smaller value takes the result out of the f32 range.
- A `temperature` that rounds to zero in f32 (`-0.0`, `1e-50`) is 0: greedy
  decoding.
- A request field out of bounds is 400 `invalid_request_error`, before the
  route loads anything.
- The same bounds apply to the values of `generation_config.json`. A file
  that does not parse, or that holds a value out of bounds, disables the
  speech route: each request fails with 500 and a fixed message. The server
  log names the file, and the key or the parse error.
- `top_p` is not a request field. A `top_p` in a speech request is ignored;
  the value comes from the file.

Differences from the references:

- The request `temperature` and `top_k` apply to both heads, as in
  mlx-audio. Qwen3-TTS has separate `subtalker_*` arguments for the code
  predictor.
- The shared sampler cuts top-p before top-k. Hugging Face and mlx-audio cut
  top-k first. At top-p 1.0, the checkpoint value, the order has no effect.
- The repetition penalty reads the full group-0 history, as Hugging Face
  does. mlx-audio reads the last 64 ids. The text routes read the last 20.

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

`WavDecoder::decode` is Symphonia-backed. It reads WAV (PCM), MP3, MP4/M4A
(AAC) and ADTS (AAC). FLAC, Ogg and the other Symphonia formats are not
compiled in, and their probe fails. It returns `DecodedAudio`: mono f32 at the
input's own sample rate, the length the file declares, the end of the last
packet read, how the reader stopped, and the audio lost (`AudioLoss`).
`WavDecoder::decode_resampled(bytes, rate, max_secs, cancelled)` returns the
same with the samples at `rate`; Whisper and the Gemma4 audio input take it
at 16 kHz. `max_secs` is `MAX_AUDIO_SECS` on the transcription routes, no cap
for `rmlx transcribe`, and on the chat `input_audio` path what the context
holds (see "Duration cap"). `cancelled` is asked before each packet; when it
says the caller is gone the decode stops with `WavError::Cancelled`.

**Resample and memory.** The channels are averaged and the result is
resampled by linear interpolation packet by packet. The samples are the same,
bit for bit, as a linear resample of the whole mono stream. More than two
channels are a plain average, each channel 1/n, the LFE included; `ffmpeg
-ac 1` uses a downmix matrix that drops the LFE (an LFE-only 5.1 file is
silence there and 1/6 of its level here; ffmpeg 9.0.2).

The decoder takes a file rate, and a resample rate, of 4 kHz to 768 kHz
(`SAMPLE_RATES`) and refuses any other (`WavError::SampleRate`). The floor
bounds a resample to 16 kHz at four outputs an input: an 8-bit mono WAV at
4 kHz gives 16 bytes of output a byte of file.

Beside the file's bytes, a decode holds its output and about one packet: no
copy of the bytes, no buffer of all channels, no buffer at the file's own
rate. The output is sized from the length the file declares or Symphonia
guesses, never past the frames the bytes can hold: one a sample's bytes for
a WAV (a WAV whose data length is a placeholder is sized from that alone),
256 a byte for a compressed stream (an AAC-LC frame of 1024 samples takes at
least 4 bytes). For a well-formed file the declared length is the smaller.
When the stream holds more, the buffer grows, and holds its old and its new
copy while it moves. The silence for dropped packets is at most 5 s
(`LOST_CAP_SECS`): see "Lost audio".

The output at 16 kHz is 64,000 bytes per second of audio: 3.84 MB a minute,
230.4 MB an hour.

**Duration cap.** The server's 25 MiB upload cap is a size, not a length,
and compressed audio of that size holds more than 7 hours. The duration cap,
`MAX_AUDIO_SECS`, bounds the audio itself on the server; `rmlx transcribe`
decodes a local file of any length:

| Bound | Value | What reaches it | Memory at the bound |
|---|---|---|---|
| Upload cap (server only) | 25 MiB (26,214,400 bytes) of file | 13.7 min of 16 kHz mono PCM-16 WAV, 2.3 min of 48 kHz stereo PCM-16 WAV, 27.3 min of MP3 at 128 kbit/s, 54.6 min at 64 kbit/s, 7.3 h at 8 kbit/s | the file |
| Duration cap | 3 h (10,800 s) of audio | every file the server decodes, of any size | 691.2 MB of 16 kHz f32 output, beside the file |

A file that declares more than the cap is refused before any packet decodes
(`WavError::TooLong`, `the file declares … s of audio, over the 10800 s the
decoder takes`; 422 `audio decode: …` on the server). A stream that declares
no length (ADTS, an MP3 with no LAME tag, a WAV through a pipe) is refused
before its output passes the cap. The end padding a LAME tag declares is read
before the trim and is not counted; the cap applies to what the trim leaves,
so an MP3 that ends before its padding is refused if it holds more than the
cap. A one-hour 16 kHz mono WAV is 115 MB and
is refused by the upload cap, not by the duration cap.

A chat `input_audio` clip (Gemma4) is capped lower: at the whole seconds
whose audio tokens fit the context left after the prompt and the two audio
markers, 25 tokens a second (640 samples a token on both audio branches),
and at most `MAX_AUDIO_SECS`. With an 8192-token context and a 100-token
prompt that is 323 s. A clip over it is refused before the mel.

The peak of a read and a decode to 16 kHz, counted on the
heap with the file's bytes: 3.0 times the file for a 16 kHz mono PCM-16 WAV
(the f32 output alone is twice the file), 1.33 times for a 48 kHz stereo
PCM-16 WAV, and the file plus 230.9 MB for an hour of 48 kHz stereo MP3 at
64 kbit/s.

**Declared length.** A WAV header, an MP4 track and the LAME tag of an MP3
declare the stream length; a length of 0 (a fragmented MP4) is no length.
Symphonia also fills a length for an ADTS stream and for an MP3 with no LAME
tag, as a guess from the bitrate; the decoder does not take a guess as a
declared length. A Xing or VBRI frame count with no LAME tag is exact, but
Symphonia does not say whether it read one or guessed, so it is not used
either.

A WAV data length of `0x7FFF0000` or more is a placeholder, not a length,
and the stream is not held to it. Writers that cannot seek back to patch the
header leave one: `0xFFFFFFFF` (ffmpeg to a pipe), `0x7FFFF000` rounded down
to the block (sox 14.4.2 to a pipe), `0x80000000` (arecord to stdout). Below
the bound, the length is held to: a WAV cut short of it loses the rest, and
the shortfall counts as lost audio.

Known limits:

- A real WAV with a data length of `0x7FFF0000` (2 GiB) or more that is cut
  decodes as a complete shorter file.
- The header shapes some writers leave on a pipe fail Symphonia's probe, as
  they did before this check: sox after 14.4.2 (RIFF length 36, data
  `0xFFFFFFFF`) and libsndfile (RIFF length 8, data 0).

**Trim.** The MP3 encoder delay (frames before time 0) is left out. The
padding a LAME tag declares is cut when the stream ends where the tag says.
When the stream holds more than the tag declares (two MP3 files joined, or a
frame count that is low), all of it is kept and a `warn` says so; `ffmpeg`
also decodes the frames past the count.

**Lost audio.** Two things lose audio:

- A packet that does not decode is dropped: a `DecodeError`, an `IoError` (the
  AAC decoder reads past the end of a mangled packet) or `Unsupported` (a
  feature bit set by damage). Its length of silence takes its place, so the
  audio after it stays at its time on the stream and the segment times hold.
- The stream ends short of its declared length.

`AudioLoss` gives the dropped packets, their seconds, the first and last
position, the first error and the shortfall. The limit scales with the
stream: `lost_limit_secs` is 1 % (`LOST_FRACTION`) of the stream's length
(the declared length, or the end reached when that is later), never below
0.2 s (`LOST_FLOOR_SECS`, more than one frame of any compiled codec at
8 kHz) and never above 5 s (`LOST_CAP_SECS`). A 5 s clip may lose 0.2 s, a
100 s file 1 s, a file of 500 s or more 5 s. Up to the limit the decode goes
on and a `warn` event, `audio lost under the limit`, carries `lost_secs`,
`limit_secs`, `dropped_packets`, `dropped_secs`, `short_secs` and
`first_drop_at_secs` as fields. Above it, the decode is refused. The limit
does not depend on where in the file the loss is: one bad frame at the end
and one in the middle are treated the same. The three numbers are a default
the owner may change.

Dropped packets that hold more than `LOST_CAP_SECS`, the largest limit, are
refused when that point is reached, before their silence is written; the
message then gives the limit of the declared length, or of the length read
so far when the file declares none.

**Refusals.** Each is a `WavError`; the message gives the time in seconds.

- `LostAudio`: more lost than the limit for the stream's length. For example
  an MP3 cut short of its LAME length, or a second MP3 at another sample rate
  appended to the first.
- `ReadFailed`: the container reader returns an error other than running out
  of bytes. An MP4 cut inside its media data, an ADTS header that is damaged.
- `PacketFailed`: the decoder fails in a way a damaged packet does not
  explain (it needs a reset, or it hit a limit), or no packet of the stream
  decodes (a feature the decoder does not support); the error is the
  decoder's on the first packet.
- `Empty`: the stream holds no packet.

A `warn` event comes before each refusal, with the declared and reached
seconds and the end. A `debug` event, `audio stream read`, gives the same for
every decode, with the end: `EndOfStream` (the reader reported it) or
`EndOfBytes` (the bytes ran out inside a packet). When the bytes run out
inside a packet of a stream that declares no length and is not a WAV, a
`warn` event, `the bytes end inside a packet; the file may be cut`, gives
`reached_secs`.

An M4A decodes 1000 to 3000 samples more than `ffmpeg` gives (measured 1152 to
2816): Symphonia does not read the MP4 edit list, so the AAC encoder's
priming at the start and its padding at the end stay in the samples.

**What no check sees.** A stream that declares no length ends where its bytes
end: a WAV written to a pipe, an MP3 with no LAME tag, an ADTS stream. Such a
file, cut short, decodes as a complete shorter file. The MP3 and ADTS readers
also skip bytes that are not a frame, with no error and no timestamp gap:
frames that garbage replaced in the middle of such a stream are missing from
the samples and from the count. With a LAME tag the missing frames show as a
shortfall at the end and count as lost.

`WavEncoder::encode` writes a 44-byte RIFF PCM-16 LE header, then the samples.
Each f32 in `[-1, 1]` is clamped and rounded to `i16`.
