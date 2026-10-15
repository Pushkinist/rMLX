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
80 or 128. The load refuses any other `n_mels` before it reads the weights,
and so does `MelExtractor::new`. The special-token ids are large-v3's (see
"Token ids").

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

- `WhisperConfig::from_json` refuses an `n_mels` other than 80 or 128, an
  `n_vocab` other than 51866 and an `n_audio_ctx` other than 1500 (the
  encoder positions of one 30 s window).
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
archive (next paragraph). The word timestamps read the pairs (see "Word
timestamps").

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
→ when words are asked: one alignment pass over the window's text → word times
→ seek: the whole window, or the end of the last closed segment
```

With `condition_on_previous_text` set, the previous window's text is fed back
after `<|startofprev|>`, unless the decode that window kept may not prompt
(see "Window decision"). The server and `rmlx transcribe` always set it.

**Prompt.** The `prompt` of a request is encoded as a space and the last 16 KiB
of its trimmed text; the trim removes the Unicode White_Space characters. The
decoder keeps 223 tokens at most, which the tail holds many times over, so a
longer prompt is not encoded whole. An empty or white-space prompt is no
prompt. The caller encodes it once, after the tokenizer loads and before the
audio is decoded and the model loads, and hands the tokens to the engine. The
prompt of a window is the last `n_text_ctx / 2 - 1` tokens (223 for large-v3)
of the prompt's tokens followed by the carried text, so a long carried text
cuts the prompt from the left. A window whose kept decode may not prompt the
next one drops both, a skipped window changes neither, and a request with no
`prompt` has the decoder prefix it had before the field was read. The
no-speech gate reads the window with no prompt. A prompt whose last 16 KiB
holds the text of a special token (`<|endoftext|>`, `<|en|>`, `<|0.00|>`, …)
is refused, because the tokenizer would turn it into a control token: 422 with
`param: prompt` on the route, an error from `rmlx transcribe`. The text before
the tail is not read, and no other check reads the text. With
`condition_on_previous_text` off, the prompt serves the first decoded window
only.

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
skips none of 15 synthetic windows with no speech. So the `no_speech_prob` of
a segment, which is this value, differs from the openai-whisper value for a
window that has a prompt; it is the value that explains a skip. Condition 2 has no
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

Seven rules differ from openai-whisper:

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
- The carried text is the text of the last window only, encoded again with no
  timestamp tokens. The reference carries the tokens of every kept segment
  since the last reset, timestamp tokens included.
- An empty or white-space prompt is no prompt. In openai-whisper `86098128` it
  is the one-token prompt `encode(" ")`.
- The prompt of the request stays at the head of the prompt of each window
  until a reset, and the carried text cuts it from the left. In openai-whisper
  (`carry_initial_prompt` off, its default) the initial prompt leaves the
  223-token tail as soon as the text since the last reset is longer.

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
most `n_text_ctx / 2` tokens (224 for large-v3), and, when words are asked,
one alignment forward of at most `n_text_ctx / 2 + 5` tokens and its CPU
steps (see "Word timestamps"). Before window 0,
the caller's model load runs to its end. The audio decode before it asks the
caller before each packet and stops there (`WavError::Cancelled`); the server
passes the request's cancel flag to the decode and to the window walk.

The server sets the flag when the request is gone (see the speech-to-text
routes in `docs/SERVER_AUDIO.md`); `rmlx transcribe` never sets it.

### Word timestamps

A request for words (`timestamp_granularities[]=word` on the route, `--words`
in `rmlx transcribe`) gets the start and the end of each word, by the method of
openai-whisper `86098128` (`whisper/timing.py`). After the segments of a
decoded window are made, one decoder forward runs over
`<|startoftranscript|>`, the language, the task, `<|notimestamps|>`, the text
tokens of all segments of the window and `<|endoftext|>`. It reads the
cross-attention keys and values of the window: no second encoder pass and no
second projection. It keeps the scores before the softmax of the alignment
heads (see "Alignment heads"), cut to the encoder frames of the real audio
(`n / 320` for a window of `n` samples). The other steps run on the CPU
(`crates/rmlx-audio/src/transcribe/words.rs`): a softmax over the frames, each
frame standardised over the tokens, a median filter of width 7, the mean over
the heads, DTW, the split into words (at spaces, or at each whole character
for `zh`, `ja`, `th`, `lo`, `my` and `yue`), the time at which the path enters
each token (50 frames a second), the rules that cut a long word at a sentence
mark and a long first word after a pause, and the merge of punctuation into
the next or the previous word. Times are rounded to 0.01 s. A skipped window
and a window with no text get no pass. A request with no words runs none of
this.

The words are added only: the text, the segments, their times and tokens,
and the seek are those of the same request with no words. Differences from
openai-whisper:

- openai-whisper moves the start or the end of a segment to its first or last
  word, and the seek to the end of the last word. rMLX moves none of them; a
  word time outside its segment is clamped into the segment.
- The last speech of the long-first-word rule is the end of the segment
  before, not the end of its last word.
- A frame with a standard deviation of 0 over the tokens becomes 0.0;
  openai-whisper divides by 0.
- The standardisation computes in f64 (torch: f32), and the rounding to
  0.01 s takes a half away from zero (Python `round(x, 2)` takes a half of
  the binary value to even: 0.125 gives 0.13 here and 0.12 there).
- A segment that rMLX drops (no length, or in the zero pad after the end of
  the file) is not aligned, and a window shorter than one encoder frame
  (0.02 s) gets no words.
- No word probability: the OpenAI word object has `word`, `start` and `end`
  only. No `hallucination_silence_threshold`, which is off by default in
  openai-whisper.

A model with no usable `alignment_heads` entry refuses a request for words:
422 with `param: timestamp_granularities` on the route, an error from
`rmlx transcribe`. A translation with words is taken, but, as openai-whisper
warns, word times of a translation may not be reliable. Alignment scores that
are not numbers fail the request with an engine error, as a logit row that is
not a number does.

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
| `timestamp_granularities[]` | `segment` | `segment` and `word`. Every response is timed per segment; `word` adds the times of each word to `verbose_json` (see "Word timestamps") and is 422 with any other `response_format`. Any other value is 422. The name without `[]` is read the same way. |
| `stream` | `false` | Read in any case. `false`, `0` or empty is taken. `true` or `1` sends a `json` reply as server-sent events, one event for each window that adds text, then a done event (`docs/SERVER_AUDIO.md` § "Streamed transcription"); with another `response_format` it is 422. Any other value is 422. |
| `prompt` | — | Text that prompts the decoder before the previous window's text (see "Prompt"). Read whole: the limit of the form bounds it, and its last 16 KiB is encoded. Text of a special token in that tail is 422. |
| any other | — | Taken and ignored (`include[]`, `chunking_strategy`, …). |

A field the route reads, other than `file` and `prompt`, holds at most 64
bytes: a longer value is 422, `<field> is over 64 bytes`, and the refusal does not echo it.
An ignored field is read and dropped as it arrives; it is never held. At
most one `warn` event per request, `form fields ignored`, gives their count
and the first eight names, each cut to 64 bytes; there is none when the form
is refused. A form of more than 64 parts, the file included, is 422.

OpenAI ignores `stream` for `whisper-1`, and its streamed models give `json`
only. The route streams `json` and refuses `stream` with any other format, so
a client that waits for events gets events or an error, and not a body it
cannot read as events.

| `response_format` | Body |
|---|---|
| `json` | `{"text": "..."}` |
| `text` | Plain text. |
| `verbose_json` | `{"task","language","duration","text","segments":[…]}`, and `"words":[…]` when `word` is asked. |
| `srt` | SRT, one cue per segment. |
| `vtt` | WebVTT, one cue per segment. |

A `verbose_json` segment holds the ten keys the OpenAI
`TranscriptionSegment` schema requires, each with a value as openai-whisper
`transcribe()` gives it, except `no_speech_prob`. For a window with no
accepted decode, the values come from the decode that the engine keeps, the
best one and not the last:

| Key | Value |
|---|---|
| `id`, `start`, `end`, `text` | The segment. |
| `seek` | The start of the 30 s window that holds the segment, in 10 ms mel frames. |
| `tokens` | The segment's slice of the window's tokens, its timestamp tokens included (large-v3 ids). |
| `temperature`, `avg_logprob`, `compression_ratio` | The window's: those of the decode the window kept, the same for each segment of the window. An `avg_logprob` below -1.0 is a decode that failed. |
| `no_speech_prob` | The window's: the value the no-speech gate read (see "No-speech gate"), the same for each segment of the window. |

A word is `{"word","start","end"}` (the OpenAI `TranscriptionWord` schema): its
text with the leading space removed and with the punctuation that the merge
joined to it, and its times in seconds, inside its segment.

`rmlx transcribe --format json` writes the same ten keys for each segment, and
with `--words` the same `words`, from the same code:
`rmlx_audio::transcribe::TranscriptionRecord`. `language` is the
English name of the language of the decode, for example `french` (see
"Language forms").

Every error the route sends is the OpenAI envelope
`{"error":{"message","type"}}` (`docs/SERVER.md` § "Error responses").
Replies made before the route runs have no body, for example the 405 of the
router for a method other than POST, the 400 of the HTTP layer for a request
head that does not parse, and the 431 for a `Content-Length` of
18446744073709551615. When one request field is at fault,
the envelope also holds `"param"`, the name of the field: `language`,
`response_format`, `temperature`, `stream`, the array name the request used
(`timestamp_granularities` or `timestamp_granularities[]`) for a value it
does not know, `timestamp_granularities` for words that the format or the
model cannot give, or `file` for a
missing, empty, oversize or undecodable file. A fault of the form as a whole
has no `param`. The `type` is `invalid_request_error` for 400, 413 and 422,
`service_unavailable` for 503, `internal_error` for 500, `rate_limit_error`
for 429 and `timeout` for 408.

| Status | When |
|---|---|
| 400 | A `multipart/form-data` request with no boundary: axum refuses it before the route reads the form, and the message is ``Invalid `boundary` for `multipart/form-data` request``. |
| 413 | The file is over 25 MiB (26,214,400 bytes): `audio file is <N> bytes, over the cap of 26214400 bytes (25 MiB)`. A request body over 26 MiB (27,262,976 bytes) stops while it is read: `request body is <N> bytes, over the limit of 27262976 bytes: …`, without `<N> bytes,` when the request sent no `Content-Length`. OpenAI answers 413 for a request over 26,214,400 bytes. |
| 422 | A malformed form, more than 64 parts, no `file` or an empty one, a field value over 64 bytes, a refused field value, words with a `response_format` other than `verbose_json` or from a model with no alignment heads, or a file the decoder refuses (`audio decode: …`, see [`AUDIO_WAV.md`](AUDIO_WAV.md)). |
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

See [`AUDIO_WAV.md`](AUDIO_WAV.md).
