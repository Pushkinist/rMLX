# rMLX Audio: Qwen3-TTS

Split out of [`AUDIO.md`](AUDIO.md) to keep that doc under the size cap.

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
- The KV cache is unquantized. Each layer keeps its keys and values in two
  buffers that grow by 256 rows (see "Memory").
- `codec_head` predicts codec group 0.

### CodePredictor

A 5-layer Qwen3 stack that predicts codec groups 1 to 15 from the talker's
last hidden state and group 0. None of its dimensions comes from the
snapshot. Hidden size (1024) and layer count (5) are serde defaults of flat
`talker_config` keys; the nested `code_predictor_config` block is not read.
Heads (16), KV heads (8), head dim (128) and MLP width (3072) are constants in
`load_talker_weights`. Its KV cache resets every frame and grows by
concatenation, one step at a time.

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

### Memory

Each talker layer keeps its keys and its values in a buffer of shape
`[1, kv_heads, capacity, head_dim]`. `KvSlot::update_and_fetch` writes the
rows of a step into the buffers (`slice_update`) and returns views of the live
rows. The attention reads the views with K and V at their own head count, so
nothing is copied for the query heads. When the rows of a step do not fit, the
capacity grows to the next multiple of `TALKER_KV_BLOCK_ROWS` (256) rows. The
slot drops its handle of the old buffer before the step is evaluated and no
other handle of it lives then, so MLX writes the rows in place. Rows of
another dtype than the buffer are an error: `slice_update` would cast them.

The buffers grow in blocks because of MLX's buffer cache. MLX serves a
request from a cached buffer only when that buffer is at least as large and
smaller than both twice the request and the request plus two pages (32 KiB).
While the capacity stays, a frame asks for the buffer sizes that the frame
before asked for, so the cache serves them and the frame leaves nothing
behind. An array that is one row longer in each frame is never served from the
cache: the freed ones pile up, the cache grows with the square of the frame
count, and the live arrays stay small.

A growth frees the buffers of the old capacity, and no later frame asks for
that size. `run_frames` releases the cache (`mlx_clear_cache`) after a frame
in which the capacity grew, once in 256 frames, and when the loop ends, by the
end token, the cap or an error. There is no other release in the loop. The
closure that `synthesize` gives it owns the key/value slots of the talker and
of the code predictor, and `run_frames` drops it before that last release, so
the codec decode starts with no key/value buffer and an empty cache. The live
keys and values still grow with the frames, in steps of 256 rows; the cache
does not. A growth logs a `debug` event with the new capacity and the live
rows, and a release logs one with its reason. A request of 421 frames grows
twice (to 256 rows at frame 0, to 512 rows at frame 248) and releases three
times. The call takes the allocator's lock and evaluates nothing, so it runs
on the calling thread. The cache is process-wide: a release also empties what
a text model on the same server had cached, so that model allocates fresh
buffers for its next steps instead of reusing those. The codec decode releases
it much more often: after each stage of each pass (about twenty times a pass).

The code predictor's slots start empty in every frame and hold at most 16
rows. They grow by one row (`PREDICTOR_KV_BLOCK_ROWS`), which is a
concatenation at each step, so their buffers have the same sizes in every
frame.

The codec decode bounds its memory by the window, not by the frame count; see
"Codec decoder".

The text embeddings are bounded by the frame cap, not by the length of
`input`. Frame `i` reads row `i` of the trailing text: the text tokens after
the first one, then the `tts_eos` row, then the `tts_pad` row. With at most
`MAX_FRAMES` frames, a row at index 2048 or later is never read, and
`text_rows` says which ids to embed. A text with up to 2047 trailing tokens is
embedded whole: up to 2056 token ids, the five tail tokens included, and
`tts_eos` follows its trailing rows. A longer text is cut to its first 2052
token ids (three role tokens, the first text token and 2048 trailing ones) and
has no `tts_eos` row, which no frame would reach. The chat text and the
tokenizer pass still grow with the length of `input`.

Measured on Metal (M5, 128 GB, `release-perf`,
Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit):

| Request | Peak footprint | Time |
|---|---|---|
| Default, 86 frames | 3.0 GiB | |
| 407 frames | | frame loop 6837 and 6887 ms; request 8.24 and 8.30 s |
| 421 frames | | frame loop 7032 and 7065 ms; request 7.78 and 7.84 s |
| To the cap, 2048 frames, temperature 0 | 4.34 and 4.36 GiB | frame loop 34.7 s in both; request 38.8 s in both; codec decode 3.4 s |
| To the cap, `input` of 1 KiB to 16 MiB, eight requests on one server | 5.68, 5.83, 4.62, 5.68, 6.33, 4.16, 5.40 and 4.73 GiB, no trend with the size | 38.8 to 41.8 s each |

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

**Windows.** `decode_in_chunks` decodes at most 300 codes (`DECODE_CHUNK_FRAMES`)
in one pass. A sequence of up to 300 frames is one pass over all its codes. A
longer one is decoded in windows of 300 frames. Each window after the first
starts 25 frames early (`DECODE_LEFT_CONTEXT_FRAMES`), as left context, and
the samples of those 25 frames are dropped, so the samples kept are those of
the window's own frames and the total is frames x 1920. This is the rule of
the reference, `chunked_decode` in mlx-audio `qwen3_tts/speech_tokenizer.py`,
which uses a chunk of 300 frames and a left context of 25. A window sees only
its own codes, so the audio of a sequence longer than 300 frames can differ
from a one-pass decode near the joins. Measured on Metal at 407 and 421 frames
(two passes each): the sample count and the transcription equal those of a
one-pass decode, and there is no step at the join (the sample step at the join
is 0.0017 of full scale, against a local median step of 0.0023). Before the
join the two decodes agree within 1 LSB of 16-bit audio; after it the
waveforms differ (rms of the difference 0.007 to 0.009, against a signal rms
of 0.05 over the first 25 frames, lower after). A request of up to 300 frames
gives the same samples as one pass. An empty sequence is an error.

**Memory.** Inside a window, `decode_window` evaluates each stage (the
pre-transformer, each upsample stage, the initial convolution, the decoder
groups' upsample and each of their residual blocks, the output snake) and
releases the MLX buffer cache before the next one, and the cache is released
after every window and after a failed one. The memory of a decode is that of
one window, not of the sequence. Evaluating an intermediate array does not
change the arithmetic. The evaluations and releases cost time: measured on
Metal at 407 and 421 frames, this decode takes 740 to 772 ms, and one pass
with no release takes 411 to 468 ms.
