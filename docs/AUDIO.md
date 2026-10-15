# rMLX Audio — Whisper STT + Qwen3-TTS

How `crates/rmlx-audio` transcribes and synthesizes. The speech-to-text
route's fields, bodies and status codes are in "HTTP route" below; its
admission, cancel and model cache, and the speech route, are in
`docs/SERVER_AUDIO.md`.

## Whisper STT

### Model

`WhisperModel::load` reads `config.json` and the weight file from an
`mlx-community` Whisper snapshot (`crates/rmlx-audio/src/whisper.rs`). The weight
file is `weights.safetensors` when the directory has it, else `weights.npz`; with
neither, the error names both files. A safetensors tensor of another
dtype than F16, BF16, F32, U8, I8, BOOL, U32, I32 or I64 is refused, and the
error names the tensor. Every
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
  `WhisperModel::load` refuses before it reads the weight file. It also
  refuses a `quantization` key: a quantized snapshot stores packed weights,
  and the decode does not unpack them.
- `WhisperTokenizer::from_path` refuses a `tokenizer.json` in which one of the
  1609 special tokens (end of text, start of transcript, the 100 languages,
  the task and control tokens, the 1501 timestamps) is absent or has another
  id, or which holds another number of tokens than 51866. `rmlx transcribe`
  and the server load the tokenizer before the model, so this refusal costs
  no weight load.

The check reads ids and sizes only. A model or a tokenizer that states the
large-v3 ids but was trained on another vocabulary is not detected.

**Alignment heads.** The `alignment_heads` entry of the weight file names the
decoder cross-attention heads that line text tokens up with audio frames, as
int32 or int64 (layer, head) pairs of shape `(N, 2)` (the large-v3 `.npz`:
`<i8`, shape `(10, 2)`; the turbo `.safetensors`: `I64`, shape `(6, 2)`). `WhisperModel::alignment_heads` gives the pairs. It
gives `None` for a snapshot with no entry, and for an entry that the reader
reads but that is not usable pairs: a dtype other than int32 or int64 (`<u4`,
float, a one-byte mask), another shape, a negative value, or a layer or head
outside the decoder. The load warns of that once, with the dtype and shape or
the pair, and goes on. An entry that the reader cannot read is an error of the
archive (next paragraph). Nothing reads the pairs yet; word timestamps are not
built.

**Members the reader reads.** A member whose name ends in `.npy` is a weight,
and it is read or the load fails. A directory member (its name ends in `/`) is
skipped. Any other member, such as a text file or a `__MACOSX/` member of a zip
tool, is not a weight: it is skipped with one debug event that names it, and it
has no key. So `weights.npz` loads whole or not at all.

**Errors of the archive.** Each of these fails the load with an error that
names the entry and the cause:
- a dtype that the reader does not know, or a multi-byte dtype with the `>`
  mark; the marks `<`, `=` and `|` read as little-endian (`=` is native order,
  little-endian on Apple Silicon), and a one-byte dtype loads with any mark;
- `'fortran_order': True`, a shape that does not parse, a byte count that
  overflows, a dimension beyond int32, data shorter than the shape, or an int64
  value outside int32;
- a `.npy` member of no bytes or with a bad header;
- an unsupported compression method, a deflate stream that fails or that is
  larger than its declared size, a record that is truncated or has a bad
  signature, a member name that is not UTF-8, and a size or offset that
  overflows.

### Pipeline

`rmlx_audio::transcribe::Transcriber` is the one transcription path. The
`/v1/audio/transcriptions` and `/v1/audio/translations` routes and
`rmlx transcribe` all call it.

```
audio bytes → WavDecoder::decode_resampled (mono f32, resampled to 16 kHz as the packets decode; refuses a file that lost too much)
→ per 30 s window: MelExtractor → encode_mel → no-speech gate → decode in timestamp mode (temperature fallback)
→ timestamp tokens → segments with cumulative times
→ seek: the whole window, or the end of the last closed segment
```

With `condition_on_previous_text` set, the previous window's text is fed back
after `<|startofprev|>`, unless the decode that window kept may not prompt
(see "Window decision"). The server and `rmlx transcribe` always set it.

Each window decodes with the openai-whisper logit filters (`DecodeFilters`).
No segment of a reply starts or ends after the duration of the file. The
decoder can put a timestamp in the zero pad after a short last window: a time
after the end of the file becomes the duration of the file, and a segment that
is left with no length is not in the reply (it opens at or after the end of
the real audio, where Whisper writes filler). openai-whisper bounds neither.
The seek, the next prompt and the scores of the window read its tokens, not
these times. The `tokens` of a segment keep each timestamp token as the
decoder gave it, so a closing timestamp token can be later than `end`.

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

**Short-window pad.** A window shorter than 30 s is a short file or the last
window of a long file. A window of `n` samples has `n / 160` content frames.
Such a window has two inputs, as in openai-whisper:

- The decode input, which the no-speech gate reads too: the frames after the
  content are 0.0 in each mel bin (`pad_after_content` in
  `crates/rmlx-audio/src/mel.rs`). The content frames are those of the audio
  with zero samples after it, clamped at the window maximum.
- The detection input: the mel of the audio padded with zero samples to 30 s
  (`MelExtractor::extract`). A frame reads 400 samples around its centre, so
  the frames from index `(n + 200) / 160`, rounded up, read zero samples only.
  Each of these has the floor of the window in each mel bin,
  `(max(window maximum - 8.0, -10.0) + 4.0) / 4.0`. The -10.0 is the `1e-10`
  clamp of the power: it gives -1.5 for digital silence. The two or three
  frames between the content and that index read samples of both.

The detection input costs a second encoder pass and one decoder step
(`detection_input_needed` in `crates/rmlx-audio/src/transcribe.rs`). The
count of encoder passes is for each window:

| Window | Encoder passes |
|---|---|
| 30 s | One |
| Shorter, the request states a language | One |
| Shorter, `auto`, a window of the file was decoded before it | One |
| Shorter, `auto`, no window of the file was decoded before it | Two |

A file shorter than 30 s can have a second window, when the first one ends at
its last closed segment. A skipped window is not a decoded window, so a short
window after it has two passes again. A window of exactly 30 s has one input:
the detection reads the decode input.

The siblings do not agree. Each row was read on GitHub at the commit in
brackets:

| Sibling | Frames after the content of a short window | Where |
|---|---|---|
| openai-whisper (`openai/whisper`, `86098128`) | Decode: 0.0, the mel of the file is cut at `n / 160` frames, then padded. Detection: the floor of the file, from 30 s of zero samples after the audio | `whisper/transcribe.py:283-286` and `:151`, `whisper/audio.py:65-87` |
| mlx-whisper (`ml-explore/mlx-examples`, `796f5b53`) | The same two inputs | `whisper/mlx_whisper/transcribe.py:289-294` and `:172` |
| faster-whisper (`SYSTRAN/faster-whisper`, `a52b026a`) | 0.0 for the decode and for the detection | `faster_whisper/transcribe.py:1178-1185` and `:1829-1832`, `faster_whisper/audio.py:117-129` |
| whisper.cpp (`ggml-org/whisper.cpp`, `0c7be696`) | The floor of the file for both: 30 s of zero samples are added to the audio, and the encoder reads the mel across them | `src/whisper.cpp:3252-3261`, `:3296-3310` and `:2434-2444` |
| WhisperKit (`argmaxinc/WhisperKit`, `9f99b5cb`) | Zero samples for both: each window is padded with zero samples, then the mel model runs on it | `Sources/WhisperKit/Core/TranscribeTask.swift:120-137`, `Sources/WhisperKit/Core/Audio/AudioProcessor.swift:188-208` |
| rMLX | Decode: 0.0. Detection: the floor of the window | `pad_after_content`, `MelExtractor::extract` |

rMLX differs from openai-whisper in three points. Its floor is that of the
window, not of the file (see "Log-mel floor"). Its detection reads the first
window that the gate does not skip, not the first 30 s of the file (see
"Language detection"). And it computes the mel of each window alone, so the
frames at the two edges of a window read a reflect pad of that window, where
openai-whisper makes one STFT of the file and reads the samples of the
adjacent window.

The model tests of `crates/rmlx-audio/tests/transcribe.rs` hold why a short
window has two inputs, on large-v3 and synthetic audio:

- With zero samples for the decode input too, a short window of noise is not
  skipped and gives a stock phrase.
  `a_short_window_of_noise_gives_no_segment` holds 5 windows of noise or of a
  hum from 2.8 s to 10 s, each alone and after a spoken window.
- With 0.0 for the detection input too, zeros of a few seconds in an `auto`
  request are not skipped and give text: the bounded decode of the gate
  writes the stock phrase of the language that the detection read, and that
  one is longer than the bound.
  `zeros_of_a_few_seconds_under_auto_give_no_segment` holds 4 lengths from
  2 s to 7 s.
- `a_file_of_a_few_spoken_words_keeps_its_words` holds that a file of one
  word and a file of three words keep their text, with `en` and with `auto`.

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

**No-speech gate.** Before a window is decoded, one decoder step of
`<|startoftranscript|>` alone gives its logit row with no token before it. The
no-speech probability is the softmax share of `<|nospeech|>` in that
unfiltered row. A window is skipped when both conditions hold:

1. The no-speech probability is above 0.6 (not at 0.6).
2. The greedy decode of the window with no prompt writes at most 8 text
   tokens: the model wrote no more than a stock phrase.

A skipped window gives no segment, the seek moves by the whole real window,
and the prompt stays. Condition 2 is read only for a window that has
condition 1. It is a decode that ends when its text passes the bound, so it
writes at most 9 text tokens, and it reads the cross-attention keys and
values of the window. A window that has condition 1 and a longer text is
decoded as each window is: with its prompt, the retry and the ladder.

Cost in decoder steps: one for each window. Above 0.6 the bounded decode adds
one step for its prefix and one for each token before its last pick. For a
skipped window that is 4 steps for ` you`, 6 for ` Thank you.`, 11 for 8 text
tokens in one segment, and at most 25 (8 segments of one token). For a window
with a long text it is 10 to 26 steps, before its normal decode starts.

openai-whisper reads the probability behind the previous-text prompt and skips
only when the average log-probability is not above -1.0 too. rMLX reads it
with no prompt and does not read the average, on purpose: the reference rule
skips none of 15 synthetic windows with no speech. Condition 2 has no
counterpart in openai-whisper. Behind a prompt they read
0.02 to 0.18, and with no prompt their average is -0.21 to -0.93, because the
model is sure of its stock text. In the 108 speech windows of a long private
recording the probability with no prompt is 0.31 or less.

Measured on large-v3 (greedy, `en`, no prompt, synthetic audio):

| Windows | No-speech | Text tokens |
|---|---|---|
| 15 with no speech: zeros (0.1 s to 30 s), noise at -80 to -10 dBFS (30 s), a hum, a tone | 0.71 to 0.92 | 1 or 3 (` you`, ` Thank you.`) |
| Speech: sentences, one word, quiet speech, speech in noise, the 5 windows of a long clip | 0.06 or less | 2 to 69 |
| Spoken repetitions of 7.3 s, 19.2 s and 22.4 s | 0.43, 0.65 and 0.72 | 2, 82 and 223 |
| 30 s of four spoken words in a loop | 0.92 | 223, the token budget |
| 30 s of one spoken word in a loop | 0.88 | 2 (` Hello?`) |

The last two rows are why condition 2 exists: with condition 1 alone, each
was skipped. The bound of 8 is above each window with no speech (3 at most)
and far below speech that the model transcribes (223), and it is the most
text that a skip can lose.

Known limits:

- A window of speech above 0.6 that the model itself decodes into no more
  than 8 text tokens is dropped. 30 s of one repeated word is such a window:
  the greedy decode of the model is one word, and the reply loses that word.
- Condition 1 does not find each window with no speech. In one measurement
  on generated white noise, made once with a harness that is not in the
  tree, 27 of 28 windows from 0.2 s to 29 s were skipped. The one of 29 s at
  -74 dBFS read 0.5617 and gave `Thank you.`, or a subtitle credit in
  Portuguese for an `auto` request.
- With `auto`, the bounded decode of condition 2 writes the stock phrase of
  the detected language, which can be longer than that of `en`. In the same
  measurement it was 5 or 6 text tokens for noise that was read as Norwegian
  Nynorsk, against 3 at most for `en`: the room to the bound of 8 is small.
- A spoken repetition can read above the bound when it is shorter than 30 s
  too: 0.65 for 19.2 s and 0.72 for 22.4 s (the clips of the two repetition
  tests). Condition 2 keeps each one, and its bounded decode runs for them.
- Not measured: a second model or language, music, room noise.

**Window decision.** A decode has two scores: the compression ratio of its
text (text bytes over a zlib stream; the encoder is miniz at level 6, not
zlib, so a text near 2.4 can fall on the other side) and openai-whisper's
`avg_logprob`. A ratio above 2.4 is a loop. An average below -1.0, or not
finite, is weak. Any other decode is accepted.

The decodes of a window go up a temperature ladder and stop at the first
accepted one: the request `temperature`, then steps of 0.2 up to 1.0 (one
decode for a request above 1.0). At 0 the pick is the largest logit. Above 0
it is one draw from `softmax(row / temperature)` by the shared sampler
(`rmlx_models::sampler::RowPicker`), one generator for each request at the
default seed: a repeated request gives the same tokens. No `best_of`, no beam.

Four rules differ from openai-whisper:

- A loop under the prompt is first decoded again at the same temperature with
  no prompt, which stays dropped for that window: a prompt can cause the
  loop. The reference keeps the prompt.
- When no decode is accepted, the best is kept: the highest average of those
  that are not loops, else the lowest ratio (the later of two equal ones),
  with a `warn` event. The reference keeps the last decode.
- A kept decode prompts the next window only when it is accepted at 0.5 or
  below. The reference has the temperature bound only.
- An average that is not finite is weak. The reference tests
  `avg_logprob < -1.0` only, which is false for a value that is not a number.

**Cancel.** `Transcriber::transcribe` takes a cancel flag and reads it before
each window opens. Once the flag is set, no further window opens, and the call
returns `WhisperError::Cancelled` with the number and the `seek` of the first
window not opened. It returns no partial transcript. A flag set during the
last window has no effect.

Worst-case latency: the window in flight runs to its end, that is, one mel and
encode (two encodes and one more decoder step for a short window whose
language is detected), the gate (its step and a bounded decode of at most 26
steps), and at
most seven decodes (six temperatures, one more with no prompt), each of at
most `n_text_ctx / 2` tokens (224 for large-v3). Before window 0,
the caller's model load runs to its end. The audio decode before it asks the
caller before each packet and stops there (`WavError::Cancelled`); the server
passes the request's cancel flag to the decode and to the window walk.

The server sets the flag when the request is gone (see the speech-to-text
routes in `docs/SERVER_AUDIO.md`); `rmlx transcribe` never sets it.

### Language detection

With `language` absent or `"auto"`, the language is the argmax over the 100
language tokens of the start row of a window. For a window of 30 s that is the
row that the no-speech gate reads, so detection adds no step. For a shorter
window it is the start row of a second input, which costs one encoder pass and
one decoder step (see "Short-window
pad"). It is read once, on the first window that the gate does not
skip. An error of that step fails the request; an error does not give
English.

That language holds for the whole file (openai-whisper and mlx-whisper detect
on the first 30 s, silent or not). A later window in another language is
decoded as that language and can come out translated into it. A first window
of music or of a short phrase in another language sets the language of the
file. Send `language` when it is known.

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

With `auto`, no language is detected when no window is decoded. The reply
names `english` then. This is so when the no-speech gate skips each window
(a `debug` event is logged) and when the audio is shorter than one 10 ms mel
hop, so that no window opens (a `warn` event is logged).

### HTTP route

`POST /v1/audio/transcriptions` and `POST /v1/audio/translations` take the
multipart form of the OpenAI reference (`CreateTranscriptionRequest`);
`translations` translates to English.

| Field | Default | What the route does |
|---|---|---|
| `file` | required | WAV (PCM), MP3, M4A (AAC) or ADTS (AAC), at most 25 MiB. Downmixed to mono and resampled to 16 kHz. |
| `model` | `whisper-large-v3` | Logged only; a longer value is kept cut to 64 bytes. The configured snapshot serves every request. |
| `language` | `auto` | An ISO code or an English name, in any case, or `auto` to detect it on the first window that is decoded (see "Language forms" and "Language detection"). Anything else is 422. |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt` or `vtt`. Anything else, `diarized_json` included, is 422. |
| `temperature` | `0.0` | A non-finite or negative value is 422, and so is a value above 0 and below `1e-6`: the sampler divides a logit by the temperature, and the result is not finite for a smaller one. Any other value is the temperature of the first decode of each window (see "Window decision"). |
| `timestamp_granularities[]` | `segment` | `segment` is taken: every response is timed per segment. `word` is 422: word timestamps need a DTW pass over the alignment heads, which is not built. The name without `[]` is read the same way. |
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
| `temperature`, `compression_ratio` | The window's: those of the decode the window kept, the same for each segment of the window. |

`avg_logprob` and `no_speech_prob` are left out: the engine reads both for its
window decision and no segment holds them. A `null` in their place would make a typed client
throw (openai-dotnet reads them as numbers). A client that requires the two
cannot read rMLX's `verbose_json` until they exist:

- `@ai-sdk/openai` (the Vercel AI SDK): it asks for `verbose_json` itself when
  the model is `whisper-1`, and its schema requires both as numbers;
- openai-python with `_strict_response_validation`.

Clients known to read it: openai-python by default (`None` for the two) and
openai-dotnet (0 for the two). `language` is the English name of the language
of the decode, for example `french` (see "Language forms").

Every error the route sends is the OpenAI envelope
`{"error":{"message","type"}}` (`docs/SERVER.md` § "Error responses").
Replies made before the route runs have no body, for example the 405 of the
router for a method other than POST, the 400 of the HTTP layer for a request
head that does not parse, and the 431 for a `Content-Length` of
18446744073709551615. When one request field is at fault,
the envelope also holds `"param"`, the name of the field: `language`,
`response_format`, `temperature`, `stream`, the array name the request used
(`timestamp_granularities` or `timestamp_granularities[]`), or `file` for a
missing, empty, oversize or undecodable file. A fault of the form as a whole
has no `param`. The `type` is `invalid_request_error` for 400, 413 and 422,
`service_unavailable` for 503, `internal_error` for 500, `rate_limit_error`
for 429 and `timeout` for 408.

| Status | When |
|---|---|
| 400 | A `multipart/form-data` request with no boundary: axum refuses it before the route reads the form, and the message is ``Invalid `boundary` for `multipart/form-data` request``. |
| 413 | The file is over 25 MiB (26,214,400 bytes): `audio file is <N> bytes, over the cap of 26214400 bytes (25 MiB)`. A request body over 26 MiB (27,262,976 bytes) stops while it is read: `request body is <N> bytes, over the limit of 27262976 bytes: …`, without `<N> bytes,` when the request sent no `Content-Length`. OpenAI answers 413 for a request over 26,214,400 bytes. |
| 422 | A malformed form, more than 64 parts, no `file` or an empty one, a field value over 64 bytes, a refused field value, or a file the decoder refuses (`audio decode: …`, see "WAV I/O"). |
| 408 | The request timeout expired (`docs/SERVER.md` § "Error responses"). |
| 429 | The admission queue is full. |
| 503 | The server's fault. With `x-should-retry: false`, which OpenAI's SDKs obey before their own retry of a 5xx: `--whisper-model-path` or `--whisper-tokenizer-path` is not set; the tokenizer, `config.json` or the weight file (`weights.safetensors` or `weights.npz`) does not load or is refused (see "Supported variants"); or the transcriber cannot be built on the loaded model (`transcriber init: …`). The body names the file and the fact, never a directory; the server log gives the path. On these routes, text that Whisper reports as an MLX error (it starts at `MLX op error:` or `MLX:`) is replaced by `MLX error; the server log has the cause`, because it can hold a path of the MLX install; the file or step before it stays. That covers mel extractor failures, an empty audio after processing and every engine error the audio crate converts, not only MLX ops. Without that header: an engine failure during the transcription (`transcribe: …`), which a retry may pass. |
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

See [`AUDIO_TTS.md`](AUDIO_TTS.md).

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
