# HTTP server: audio routes

Split out of [`SERVER.md`](SERVER.md) to keep that doc under the size cap.

## Audio

### Speech-to-text: `/v1/audio/transcriptions` and `/v1/audio/translations`

Both take the OpenAI multipart form; `translations` translates to English.
The fields, the bodies and each status are in [`AUDIO.md`](AUDIO.md) §
"HTTP route". `file` is at most 25 MiB (413 above it, naming the cap and
the size); `language`, `response_format` and `temperature` (the first step of
each window's temperature ladder) are read; `timestamp_granularities=word`
adds word times to `verbose_json` and is 422 with any other format
([`AUDIO.md`](AUDIO.md) § "Word timestamps"); `stream=true` sends a `json`
reply as server-sent events and is 422 with any other format (see "Streamed
transcription"); `prompt` is read ([`AUDIO.md`](AUDIO.md) § "Prompt"),
and any other field is ignored, counted in at most one `warn` event. A `verbose_json` segment has the ten keys OpenAI's schema
requires ([`AUDIO.md`](AUDIO.md) § "HTTP route").
A Whisper load failure is 503 with `x-should-retry: false` and names no
server directory; an engine failure during the transcription is 503.

The upload is one buffer, sized from `Content-Length` up to the cap; a
chunked upload, or one over the cap, grows it. The file is decoded
after admission, under the GPU permit and before the Whisper model loads:
decodes run one at a time, a waiting request holds only its upload, and a
file the decoder refuses (422 `audio decode: …`) loads no model. The decode
holds the file and its 16 kHz output, then drops the file.

`rmlx_audio::transcribe::Transcriber`, shared with `rmlx transcribe`, walks
the audio in 30 s windows. Each window decodes in timestamp mode with the
openai-whisper logit filters (`SuppressBlank`, `SuppressTokens`,
`ApplyTimestampRules`). The emitted timestamps become segments with real
times. The seek consumes the whole window when the tokens end in a single
timestamp, and otherwise goes to the end of the last closed segment
([`AUDIO.md`](AUDIO.md), "Seek rule"). A window is skipped when its
no-speech probability is above 0.6 and the model writes no more than a stock
phrase for it ([`AUDIO.md`](AUDIO.md), "No-speech gate"). The request's `prompt`, then the previous window's text, are fed back as a
`<|startofprev|>` prompt.
A decode that is a repetition loop or has a low average log-probability is
decoded again, first without the prompt for a loop, then up a temperature
ladder ([`AUDIO.md`](AUDIO.md), "Window decision"). No segment
starts or ends after the duration of the file, and a segment that opens at or
after the end of the real audio is not in the reply.

`language` is an ISO code or an English name, in any case; the reply names the
language as the English name. With `language` absent or `auto`, the language
is detected once, on the first window that is not skipped, and holds for the
whole file ([`AUDIO.md`](AUDIO.md), "Language forms" and "Language
detection").

The Whisper model and tokenizer load on the first request and stay for the life
of the process; changing the snapshot needs a restart. The tokenizer loads
first. A tokenizer or a `config.json` of another vocabulary than large-v3's, or
with an `n_mels` other than 80 or 128, or with a `quantization` key, is refused
before the weights load, and every request is 503 with the refusal text
(`docs/AUDIO.md` § "Supported variants"). Audio requests pass the same admission
gate as chat, and hold the GPU permit for the audio file decode, the encode and
the Whisper decode.

When the request is gone, because the 408 fired, the client closed the
connection or a server shutdown dropped it, the transcription stops before
its next 30 s window and releases the permit. An `info` event in the request's span,
`long-form transcription cancelled before the window opened`, gives the
window number, its `seek` in samples and the cause, `stop`: `flag` for the
flag of the request, `reader_gone` for a streamed reply whose client is gone.
The stop is not immediate:

- The window in flight runs to its end: the encode, the gate and at most
  seven decodes of `n_text_ctx / 2` tokens each (224 for large-v3). See
  "Worst-case latency" in [`AUDIO.md`](AUDIO.md).
- The audio decode stops before its next packet. The model load of the
  first request is not cut.
- Only a 408, a closed connection or a server shutdown stops the walk. After
  SIGINT or SIGTERM a transcription has 8 s to end and return its text. At
  8 s, or at a second signal, the request gets 503 in the OpenAI envelope,
  `the server is shutting down`, also while it waits for the GPU, and the
  walk stops. A window that still runs 2 s later ends with the process
  ([`SERVER_SHUTDOWN.md`](SERVER_SHUTDOWN.md)). With `--max-timeout-secs 0`,
  only a closed connection or the shutdown stops it.

#### Streamed transcription

`stream=true` (or `True`, or `1`) with `response_format` `json`, named or by
default, gives a reply of server-sent events (`text/event-stream`), on
`/v1/audio/transcriptions` and on `/v1/audio/translations`. With any other
`response_format` the request is 422 with `param: stream`: OpenAI's streamed
models give `json` only. `stream=false`, `0` and an empty value give the
reply of a request with no `stream` field.

Each event is one `data:` line that holds a JSON object with a `type`, as the
chat routes send their chunks. No event has an `event:` name.

| Event | When |
|---|---|
| `{"type":"transcript.text.delta","delta":"…"}` | A 30 s window closed and its segments hold text. A window with no text sends no event. |
| `{"type":"transcript.text.done","text":"…"}` | The transcription ended. This is the last event. |

The segments of a window are final when the window closes, so no later event
changes the text of a delta. A `delta` is the text of the segments of its
window, with each run of white space made one space. Each delta but the first
starts with one space. The deltas, joined with nothing, are the `text` of the
done event, and that is the `text` of the reply with no `stream`. The done
event holds no `usage`, and no `data: [DONE]` line follows it. The server
sends no keep-alive event, and one window can take seconds.

The reply starts when the prompt is encoded, the audio is decoded and the
model is loaded. A failure before that point, for example a file that the
decoder refuses, is the JSON error reply of a request with no `stream`, with
its status. A failure after the first event is one error event,
`{"error":{"message","type"}}`, with the message and the type of that error
reply. Then the stream ends with no done event, and no `[DONE]` line follows.
A panic after the first event ends the stream with no event; the server log
has it. A client must look for `transcript.text.done` to know that it has the whole
text.

A client that closes the connection stops the transcription before its next
window, as for a request with no `stream`: the job cannot send the next delta,
or it reads the flag of its request. The GPU permit is free when the window in
flight ends. A client that does not read can hold the permit: when the
connection's buffers are full and four events wait, the job waits until the
client reads, closes, or the server shuts down.

The request timeout (`--max-timeout-secs`, `X-Request-Timeout-Seconds`) bounds
the time to the start of the reply, that is the prompt, the audio decode and
the model load. It does not bound the stream after the start: then only a
closed connection, the server shutdown or the end of the audio ends it: the
timeout layer stops its clock when the route returns the response head.

### Text-to-speech: `/v1/audio/speech`

Qwen3-TTS synthesizes mono 24 kHz audio from a JSON body:

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | yes | — | Model identifier. |
| `input` | yes | — | Text to synthesize. An empty or whitespace-only input is 400. |
| `voice` | no | `serena` | `serena`, `vivian`, `ryan`, `aiden`, `eric`, `dylan`, `ono_anna`, `sohee` or `uncle_fu`. An unknown voice is 422; the reply lists the voices of the model and does not repeat the voice of the request. |
| `response_format` | no | `wav` | `wav` (RIFF, PCM-16 LE) as `audio/wav`, or `pcm` (raw f32 LE) as `audio/pcm`. |
| `speed` | no | `1.0` | Accepted and ignored. |
| `temperature` | no | snapshot | Temperature of the talker and the code predictor: 0, or in `[1e-6, 2]`. `0` is greedy decoding. |
| `top_k` | no | snapshot | Top-k cutoff of both heads. `0` disables it. |
| `repetition_penalty` | no | snapshot | Repetition penalty of the talker, finite and at least `1e-6`. The code predictor has none. `1.0` disables it. |
| `seed` | no | `0xA7A7` | Seed of the request's one RNG stream. One input, voice, seed and field set give the same audio each time. |

A field that the request does not set comes from the talker snapshot's
`generation_config.json`, then from the Qwen3-TTS reference default
(`docs/AUDIO_TTS.md`, "Sampling"). The checkpoint samples at temperature 0.9 with
top-k 50 and a talker repetition penalty of 1.05. `temperature: 0` alone keeps
that penalty; greedy decoding with no penalty needs `temperature: 0` and
`repetition_penalty: 1.0`. A sampling field of the wrong JSON type is refused
by the JSON body parser; a value out of bounds is 400 `invalid_request_error`
before the route loads anything. The upper bounds are those of
`/v1/chat/completions`; the floor of `1e-6` and the refusal of an infinite
`repetition_penalty` belong to this route. `top_p` is not a request field
and is ignored. A `generation_config.json` that does not parse, or that
holds a value out of bounds, makes every speech request 500.

A successful response carries two headers:

| Header | Value |
|---|---|
| `x-rmlx-tts-frames` | The number of codec frames generated (12.5 frames per second of audio). |
| `x-rmlx-tts-truncated` | `true` when generation stopped at the 2048-frame cap before the end token, so the audio is cut off; otherwise `false`. |

A truncated generation also logs a `warn` event with the fields `frames` and
`cap`.

Without `--tts-model-path` and `--tts-tokenizer-path` the route is 503 with
type `not_supported`. Another synthesis failure is 500 with a fixed message;
the cause, which can name a local path, is in the server log. The 400 of a
value out of bounds or an empty input, the 422 of an unknown voice and every
500 are `{"error":{"message","type"}}`. A body that the JSON parser refuses
is 400 or 422 with a plain-text message. The model and its
sampling defaults load on the first request, stay for the life of the
process, and the model holds the GPU permit while it synthesizes. A request
that times out, or whose client goes away, stops the synthesis: it reads the
stop before each talker frame and before each pass of the codec decode, and
the permit is free when the job returns (`docs/SERVER_SHUTDOWN.md` § "What
stops, and what does not").
