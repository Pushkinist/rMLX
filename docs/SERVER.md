# HTTP Server

Reference for the rMLX HTTP server (`crates/rmlx-server`): routes, the OpenAI
and Anthropic surfaces, streaming, tool calling, embeddings, audio, chat
templates, admission, model lifecycle, the claim file and the retry envelope.

---

## Overview

One binary serves two API surfaces:

- **OpenAI-compatible**: `POST /v1/chat/completions`, `GET /v1/models` and the
  model lifecycle routes.
- **Anthropic-compatible**: `POST /v1/messages`.

Both drive the same `Generator` token stream. They differ only in the wire
schema: field names, SSE event types and block shape.

---

## Architecture

SIGINT and SIGTERM shut the server down gracefully, so the claim file is
removed. A handler panic returns 500. The request body limit is 26 MiB.

### Compute placement

Inference is synchronous and runs in `tokio::task::spawn_blocking`. Async is
used only at the HTTP boundary and for file I/O.

### Concurrency model — single-stream by design

The server runs one generation at a time. The `gpu_queue` semaphore has one
permit. Requests take it in FIFO order and hold it for the whole prefill and
decode. There is no continuous batching and no interleaving across
sequences: request B makes no progress while request A runs. Per-arch prefill
chunking bounds one request's memory; it does not schedule other requests
between chunks.

This fits the target workload: one user or one agent, with one generation in
flight. Under concurrent load, aggregate throughput stays flat and a
request's latency grows with the queue ahead of it. `max_queue_depth` and the
adaptive admission controller shed load; neither adds parallelism. To serve
many users at once, run several rMLX processes behind a load balancer, one
Metal context each.

### GPU admission and queue depth

`engine::admit_request` refuses a request with HTTP 429 `rate_limit_error`
when `max_queue_depth` is non-zero and `gpu_pending` has reached it. Otherwise
it counts the request and waits for the permit. `--max-queue-depth` defaults
to 64. An RAII guard releases the permit on every exit path: success, error,
timeout and stream abort. Chat, messages and audio requests pass this gate.
`/v1/embeddings` does not: it never returns 429, and it waits instead on the
`gpu_gate` mutex that a running generation also holds.

### Adaptive admission controller

`--adaptive-admission` (off by default) adds `crate::admission`:

1. **Anticipatory 503.** Before the FIFO wait, a sliding-window OLS regressor
   predicts the request's admission-to-final-token time from
   `(prompt_tokens, kv_bytes)`. When the prediction exceeds twice
   `--step-target-ms` (default 500 ms; alias `--ttft-target-ms`), the request
   gets HTTP 503 `admission_sla_exceeded` with `Retry-After: 5`.
2. **Adaptive queue depth.** A tick every 5 s adjusts `max_queue_depth` from
   a predicted inter-token latency. It lowers the depth by 1 after 3
   consecutive ticks above `--itl-target-ms` (default 50 ms). It raises it by 1
   when the prediction is below 0.80 × the target. The depth stays in
   `[1, 256]`.
3. **Step metrics.** Each completed request feeds
   `(prompt_tokens, kv_bytes, step_ms)` back into the regressor.
4. **Events.** Each tick writes a `stage = "admission_ctrl"` row to the
   `events` table, with the `DecisionReason` as `op`.
5. **Adaptive prefill chunk.** `--adaptive-prefill-chunk` also moves the
   process-wide prefill chunk with the same deadband, within `[32, 2048]`
   tokens. Its reasons (`prefill_chunk_raise`, `prefill_chunk_lower`,
   `prefill_chunk_hold`) go to the log only, not to the `events` table.

The tick task lives in `AppState::admission_handle` and is aborted when the
last `AppState` clone drops. With the controller off, the FIFO gate above is
the only admission path.

---

## Routes

| Method | Path | Handler | Description |
|---|---|---|---|
| `GET` | `/health` | `health` | Liveness probe. Returns `{"ok":true}`. |
| `POST` | `/v1/chat/completions` | `openai::chat_completions` | OpenAI chat, streaming and non-streaming. |
| `GET` | `/v1/models` | `openai::list_models` | List registered models. |
| `POST` | `/v1/models/{id}/load` | `openai::load_model` | Load a model; returns when it is resident. |
| `POST` | `/v1/models/{id}/unload` | `openai::unload_model` | Unload a resident model. |
| `GET` | `/v1/models/{id}/status` | `openai::model_status` | Resident or not, with timestamps. |
| `POST` | `/v1/embeddings` | `embeddings::embeddings` | Text and image embeddings (jina-v4). |
| `POST` | `/v1/audio/transcriptions` | `audio::audio_transcriptions` | Whisper speech-to-text. |
| `POST` | `/v1/audio/translations` | `audio::audio_translations` | Whisper speech-to-text, translated to English. |
| `POST` | `/v1/audio/speech` | `audio::audio_speech` | Qwen3-TTS speech synthesis. |
| `POST` | `/v1/messages` | `anthropic::messages` | Anthropic Messages API, streaming and non-streaming. |
| `GET` | `/metrics/cache` | `openai::metrics_cache` | Prompt-cache, TTFT, ITL and error counters as JSON. |
| `GET` | `/metrics` | `openai::metrics_prometheus` | Prometheus text exposition v0.0.4. |
| `GET` | `/v1/metrics` | `openai::metrics_v1_summary` | Rolling request summary as JSON (mlx-vlm shape). |

The timeout middleware reads an optional `X-Request-Timeout-Seconds` header
and caps it at `--max-timeout-secs` (default 600). A cap of 0 disables the
timeout. A header that is not a positive integer gets 400. An expired request
gets HTTP 408 `timeout`.

`X-Request-Id`, when present, becomes the request's correlation id (trimmed,
printable ASCII, at most 128 characters). Otherwise the server makes a
`req-<uuid>`. It is returned in the `X-Request-Id` response header and tagged
on the request's tracing span. The response `id` is built from it:
`chatcmpl-<request-id>` on the OpenAI route, `msg_<request-id>` on
`/v1/messages`.

---

## OpenAI Compatibility

### `POST /v1/chat/completions`

**Request fields** (all optional unless noted):

| Field | Type | Notes |
|---|---|---|
| `model` | string (required) | Registry model id. |
| `messages` | array (required) | `role` and `content`; also `tool_calls`, `tool_call_id`, `name`. |
| `stream` | bool | Default false. |
| `temperature`, `top_p`, `top_k`, `min_p` | number | Resolution order in [`SAMPLING.md`](SAMPLING.md) § "Defaults and resolution order". |
| `repetition_penalty`, `frequency_penalty`, `presence_penalty` | f32 | Same. |
| `logit_bias` | object | Token id (string key) → finite bias. A bad key or a non-finite value is 400. |
| `seed` | u64 | |
| `max_tokens` | u32 | Default 512. Capped at `--max-tokens-cap`, itself bounded by 1 048 576. Over the cap is 400, never clamped. |
| `stop` | string or array | Stop sequences. |
| `tools` | array | OpenAI function specs. |
| `tool_choice` | string or object | `"auto"`, `"none"`, `"required"` or `{type:"function",function:{name:…}}`. |
| `response_format` | object | `{type:"text"}`, `{type:"json_object"}` or `{type:"json_schema",json_schema:{…}}`. |
| `logprobs` | bool | Return the chosen token's logprob for each content token. |
| `top_logprobs` | u32 | 0–20; requires `logprobs:true`. |
| `stream_options` | object | `{include_usage:true}` appends a usage chunk before `[DONE]`. |
| `enable_thinking` | bool | `false` selects the template's no-think branch. |
| `thinking_budget` | u32 | Caps the reasoning channel at N tokens. |
| `thinking_start_token`, `thinking_end_token` | string | Replace `<think>` / `</think>` for the splitter and the budget injection. |
| `kv_quant` | string | Per-request KV codec. See "Per-request KV config". |
| `max_ctx` | i32 | Per-request context ceiling. See "Per-request KV config". |
| `image_max_tokens` | u32 | Per-image soft-token budget for Gemma4-unified vision. Must be `> 0`; clamped to the model's upper bound. Order: request, `--image-max-tokens`, the snapshot's `processor_config.json`. A no-op elsewhere. |
| `echo` | bool | `echo:true` is 400; use `rmlx eval ppl` for prompt logprobs. |

A `functions` field is refused with 400. Any other unknown field is
debug-logged and ignored.

### Per-request KV config

`kv_quant` and `max_ctx` change the KV cache for one request without
reloading the weights. The cache is built per request, so a resident model
can sweep codec and context cells, or run a different KV policy per request.

- **`kv_quant`** takes the `--kv-quant` grammar (`"none"`, `"k8v4"`,
  `"planar"`, `"mixed_k<kb>g<kg>_v<vb>g<vg>"`, …). A malformed string is 400.
  It wins over the launch `--kv-quant`. `"auto"` names no codec, so the
  request runs what the server resolved at load. Absent means the launch
  default.
- **The prompt cache is partitioned by codec.** A prefix cached under one
  codec never serves a request on another; see `docs/PROMPT_CACHE.md`
  § "Codec namespacing".
- **`max_ctx`** must be `> 0`. The route resolves it through
  `rmlx_models::context::resolve_context`, the same function as the launch
  `--max-ctx`. A value above the checkpoint's positional capacity is 400
  `context_length_exceeded`, never clamped. The prompt-length guard then uses
  the resolved ceiling.
- `/v1/messages` does not take these fields; it uses the launch defaults.
- There is no per-request SSD-tier switch; see `docs/SSD_TIER.md`
  § "Live reconfiguration".

### Multimodal content parts — image + native audio input

A user message's `content` is a string or an array of parts. Image and audio
parts come from the last user message. The tower output is scattered into the
prompt at the placeholder positions, and decode runs from the fused
`inputs_embeds`.

| Part `type` | Shape | Tower | Architectures |
|---|---|---|---|
| `text` | `{type:"text", text:"…"}` | — | all |
| `image_url` | `{type:"image_url", image_url:{url:"<url\|data-URL>"}}` | SigLIP vision | Gemma4, Gemma3, Qwen3-VL-MoE |
| `input_image` | `{type:"input_image", image_url:"<url>"}` (mlx-vlm shape) | SigLIP vision | same |
| `input_audio` | `{type:"input_audio", input_audio:{data:"<base64>", format:"wav"}}` | Conformer or encoder-free | Gemma4 |

**Audio.** The `rmlx-audio` decoder (Symphonia: WAV, MP3, M4A and others)
downmixes to mono and resamples to 16 kHz. The Gemma4 Conformer checkpoints
run the USM log-mel front-end and the Conformer `audio_tower`. The unified
encoder-free checkpoint (`Gemma4UnifiedForConditionalGeneration`) cuts the
waveform into 640-sample frames and projects each with `embed_audio`, one soft
token per 40 ms; see [`MODELS.md`](MODELS.md). The prompt gets `<|audio>`,
N × `<|audio|>` and `<audio|>`, and the soft tokens land on the `<|audio|>`
positions.

- One clip per request; more is an error.
- Image and audio together in one request is an error.
- Audio sent to a model with no audio tower is 503 "this model does not
  accept audio input (no audio tower)". Images sent to a model with no vision
  tower get the matching "no vision tower" 503.
- Each `input_audio` part is capped at 16 MiB decoded
  (`bounds::MAX_INPUT_AUDIO_BYTES`); a larger clip is 400.

### Responses

**Non-streaming** (`stream:false`):

```json
{
  "id": "chatcmpl-<request-id>",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "my-model",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "…",
      "reasoning_content": "…",
      "tool_calls": [{"index":0,"id":"call_<hex>","type":"function","function":{"name":"…","arguments":"…"}}]
    },
    "finish_reason": "stop",
    "logprobs": {"content": [{"token":"…","logprob":-0.5,"bytes":[…],"top_logprobs":[…]}]}
  }],
  "usage": {"prompt_tokens":42,"completion_tokens":7,"total_tokens":49}
}
```

`reasoning_content` is omitted when the model produced no thinking text,
`tool_calls` when no call was parsed, and `logprobs` unless requested. When
present, `logprobs.content` has one entry per completion token. A prompt-cache
exact hit replays the first token's logprob stored with the entry, so a hit
returns as many entries as a miss.

A `response_format` request whose grammar never engaged returns 502
`constraint_not_engaged`; see [`SAMPLING.md`](SAMPLING.md) § "Non-enforcement
is reported".

**Streaming** (`stream:true`): each SSE event is a `data:` line holding a
`ChatCompletionChunk`:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null,"logprobs":{…}}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

- Thinking tokens arrive as `reasoning_content` deltas and answer tokens as
  `content`. One chunk never carries both.
- `stream_options.include_usage:true` adds a chunk with `choices:[]` and a
  `usage` object before `[DONE]`.
- `logprobs` rides on each content chunk. Chunks with no content token (role
  preamble, tool call, usage) omit it.

**Mid-stream failure.** The status is already 200, so the failure goes in the
payload. An error event, in the envelope the blocking route uses, replaces the
terminal `finish_reason` chunk:

```
data: {"error":{"message":"…","type":"service_unavailable"}}

data: [DONE]
```

A stream with no `finish_reason` chunk did not complete. Treat a missing
terminal `finish_reason`, or an `error` key, as a failure, not a short
answer. The engine emits no `finish_reason:"error"`.

### Stop-sequence truncation

`stop` (OpenAI) and `stop_sequences` (Anthropic) truncate the generated
content at the first match, on both surfaces, streaming or not:

- The matched string is excluded; output ends just before it.
- OpenAI sets `finish_reason:"stop"`. Anthropic sets
  `stop_reason:"stop_sequence"` and names the match in `stop_sequence`.
- With several strings, the earliest byte offset wins; a tie goes to the
  first string in the array.
- Matching runs on the detokenized text, so a stop string that straddles a
  token boundary is found.
- Streaming holds back a tail that could still grow into a stop string, so a
  straddling stop is never half-emitted, and nothing follows the stop.
- Only the content channel is matched; reasoning is not.
- A response carrying tool calls is not truncated.

`rmlx_server::stop_matcher` holds the matcher: `find_stop_match` for
non-streaming and `StopMatcher` for streaming. An empty stop string is
ignored.

### Anthropic `stop_reason` mapping

`map_stop_reason` (`crates/rmlx-server/src/anthropic/route.rs`) maps the
engine's `finish_reason`:

| Engine `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `"stop"` or none | `"end_turn"` |
| `"length"` | `"max_tokens"` |
| `"tool_calls"` | `"tool_use"` |
| anything else | `"error"` |

An unrecognised reason maps to `"error"`, never to `"end_turn"`.
`"stop_sequence"` comes only from the stop-matching path in `blocking.rs` and
`streaming.rs`, which bypasses `map_stop_reason`. A stream that dies mid-flight
ends with Anthropic's native `error` event and no `message_delta` or
`message_stop`:

```
event: error
data: {"type":"error","error":{"type":"service_unavailable_error","message":"…"}}
```

A `/v1/messages` stream with no `message_stop` did not complete.

### `GET /v1/models`

```json
{"object":"list","data":[{"id":"my-model","object":"model","created":0,"owned_by":"rmlx","loaded":false}]}
```

Every registry model is listed. A resident one also carries `loaded_at` and
`last_used` (Unix seconds), and, when the architecture exposes
`max_position_embeddings`, the two context numbers:

| Field | Meaning |
|---|---|
| `max_ctx` | The ceiling in force: what the prompt guard enforces and what the KV ring may grow to. |
| `positional_max` | What the checkpoint can address, RoPE scaling included; the highest a per-request `max_ctx` may ask for. |

Without `max_position_embeddings` the resolver accepts any `max_ctx`, so both
fields are omitted. The same pair is on the `slots: model loaded` log line.
See `docs/CLI.md` § "Context ceiling".

### Model lifecycle endpoints

- `POST /v1/models/{id}/load` calls `AppState::ensure_loaded` and returns
  once the model is resident: 200 `{"ok":true,"model":"<id>"}`. An id outside
  the registry is 404 `model_not_found`. A load failure is 503
  `service_unavailable`.

  An optional JSON body takes `keep_alive` (integer seconds, as in Ollama and
  LM Studio): negative pins the model, `0` unloads after the next request,
  positive sets the idle TTL. Absent keeps the slot's policy, which comes from
  `--idle-timeout-secs`, then the `projects.toml` profile, then 15 minutes.

  ```bash
  curl -X POST -d '{"keep_alive": -1}' http://127.0.0.1:8080/v1/models/gemma-4-e4b/load
  curl -X POST -d '{"keep_alive": 120}' http://127.0.0.1:8080/v1/models/gemma-4-e4b/load
  ```

- `POST /v1/models/{id}/unload` calls `AppState::unload`: 200 `{"ok":true}`
  when the model was resident, 404 `{"ok":false,"message":"…"}` when not.
- `GET /v1/models/{id}/status` returns 200
  `{"id","loaded","loaded_at","last_used","idle_secs"}`, with the last three
  `null` when not resident. An id outside the registry is 404
  `model_not_found`.

**Slots and eviction.** Up to `--max-loaded-models` (default 1) models stay
resident. When the slots are full, `ensure_loaded` for another model evicts
the least recently used one first. Any request can trigger this, not only
`/load`.

**Keep-alive on the compat routes.** `/v1/chat/completions` and
`/v1/messages` take no `keep_alive` field. Each successful `ensure_loaded`
resets the model's timer, so an active client keeps it resident. The
embedding and audio models live in their own process-lifetime slots
(`embed_slot`, `audio_model`, `tts_model`) with no keep-alive TTL.

**Decode lease.** Every generation holds a decode lease for the life of its
response. A streaming response holds it through `GuardedStream` until the
stream is consumed or the client disconnects. While a lease is held, the
keep-alive timer does not unload the model: it logs `keep_alive: decode in
flight — deferring unload` and re-arms. LRU eviction does not wait on the
timer.

### Error responses

Errors use the OpenAI envelope:

```json
{"error":{"message":"…","type":"<error_type>"}}
```

| HTTP | `type` | Condition |
|---|---|---|
| 400 | `invalid_request_error` | Bad field, out-of-range value, `echo:true`, `functions`, over-cap `max_tokens`. |
| 400 | `context_length_exceeded` | The prompt exceeds the effective `max_ctx`, or a per-request `max_ctx` exceeds the positional capacity. |
| 404 | `not_found_error` | Chat or embeddings model not in the registry (`model_not_found` on the lifecycle routes). |
| 408 | `timeout` | The request timeout expired. |
| 429 | `rate_limit_error` | The GPU admission queue is full. |
| 500 | `internal_error` | A handler panic, a prompt-pipeline task panic, or an embeddings preprocessor or compute failure. The audio routes send 500 as `{"error":"…"}`. |
| 502 | `constraint_not_engaged` | A non-streaming `response_format` request whose grammar never engaged. |
| 503 | `service_unavailable` | Any load failure, OOM while loading included, and any other engine error. With `--require-smoke-probe` (off by default) a failed smoke probe at load lands here too. Counter: `upstream`. |
| 503 | `admission_sla_exceeded` | The adaptive controller's anticipatory rejection, with `Retry-After: 5`. Counter: `admission_sla_503`. |

`/v1/messages` sends the same statuses. `context_length_exceeded`,
`admission_sla_exceeded`, `timeout`, `invalid_request_error`,
`not_found_error` and `rate_limit_error` keep the OpenAI strings; 500 is
`internal_server_error` and 503 is `service_unavailable_error`.
`GET /metrics/cache` exposes a process-lifetime counter per category under
`error_counts`.

### `X-Session-Id` header

A request with `X-Session-Id` registers `(model_id, session_id)` in the
session cache, which raises the prompt-cache slot count passed to the
generator. See [Session cache](#session-cache).

---

## Anthropic Compatibility

### `POST /v1/messages`

**Request fields**:

| Field | Type | Notes |
|---|---|---|
| `model` | string (required) | Registry model id. |
| `max_tokens` | u32 (required) | Missing is 400. |
| `messages` | array (required) | `role` and `content` (string or block array). `input_audio` blocks carry `source:{type:"base64",data:…}`. |
| `system` | string or array | System prompt, rendered through the chat template. |
| `temperature`, `top_p`, `top_k` | number | Same resolution order as the OpenAI route. |
| `stop_sequences` | array | Stop sequences. |
| `stream` | bool | Default false. |
| `tools` | array | `name`, `description`, `input_schema`. |
| `tool_choice` | object | `{type:"auto"\|"any"\|"tool", name?:"…"}`. |
| `metadata` | object | Accepted and ignored. |

**Non-streaming response**:

```json
{
  "id": "msg_<request-id>",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type":"thinking","thinking":"…"},
    {"type":"text","text":"…"},
    {"type":"tool_use","id":"call_<hex>","name":"…","input":{…}}
  ],
  "model": "my-model",
  "stop_reason": "end_turn",
  "usage": {"input_tokens":42,"output_tokens":7}
}
```

The `thinking` block appears only when the model produced reasoning text, the
`tool_use` block only for a parsed tool call. `input` is a JSON object, not the
JSON string OpenAI puts in `arguments`.

**Streaming** follows the Anthropic protocol: `message_start`, a `ping`, then
per block `content_block_start`, `content_block_delta` and
`content_block_stop`, then `message_delta` and `message_stop`. Thinking text
arrives in a `thinking` block with `thinking_delta` events, before the `text`
block. A tool call arrives as a `tool_use` block with `input_json_delta`.

---

## Embeddings

### `POST /v1/embeddings`

Serves jina-embeddings-v4 text and image embeddings.

| Field | Type | Notes |
|---|---|---|
| `model` | string | Registry id of a jina-v4 snapshot; another architecture is 400, an unknown id 404. |
| `input` | string, array, or image object | Text: `"a"` or `["a","b"]`. Image: `{"image":"<data-URI\|base64\|path>"}` or a list of them. |
| `encoding_format` | string | `"float"` (default) or `"base64"`; `base64` with `return_multivector` is 400. |
| `dimensions` | usize | Matryoshka truncation to one of the model's sizes. |
| `task` | string | LoRA task: `retrieval` (default), `text-matching` or `code`. |
| `prompt_name` | string | `query` (default) or `passage`. `text-matching` always uses `Query`. |
| `return_multivector` | bool | Per-token multi-vector output instead of one pooled vector. |

Text gets the prefix `"{Query|Passage}: {text}"` and no special tokens. One
request embeds all text or all images, not a mix.

```json
{
  "object": "list",
  "data": [{"object":"embedding","index":0,"embedding":[0.1,0.2,…]}],
  "model": "my-embed-model",
  "usage": {"prompt_tokens":10,"total_tokens":10}
}
```

`embedding` is `[f32]` for one vector, `[[f32]]` for multi-vector output, or
a base64 string. The embedding model lives in `AppState::embed_slot`, not in
the LLM slots, and loads on the first request. `apply_task` swaps the LoRA
adapter inside the GPU critical section.

---

## Tool Calling

### Parser architecture

`ToolCallStreamParser` parses tool calls out of the raw token stream. The
format is detected once at registry build from markers in
`chat_template.jinja`, with the architecture as the fallback, and cached in
`ModelEntry`.

| `ToolCallFormat` | Used by | Syntax |
|---|---|---|
| `Qwen3XmlFunction` | Qwen3.6 | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` |
| `Qwen3JsonToolCall` | `Qwen3ForCausalLM` (Bonsai) | `<tool_call>{"name":"…","arguments":{…}}</tool_call>` |
| `GemmaToolCall` | `Gemma4ForConditionalGeneration` | `<\|tool_call>call:NAME{key:val}<tool_call\|>` |

Gemma registers `<|tool_call>`, `<tool_call|>` and `<|"|>` as special tokens,
which `tokenizer.decode` strips. The engine rebuilds them from the token ids
before the parser sees them.

The parser is split-invariant: any BPE-aligned split of the stream parses the
same as the whole string. Several `<tool_call>` blocks may follow each other.

### Template support probe

At registry build, `probe_tools_supported` renders each template with one
tool and stores the result in `ModelEntry::tools_supported`. When it is
`false`, the request runs without tools instead of failing.

### Multi-turn tool loop

The client drives the loop. It sends `tools`; the model emits tool-call
blocks; the server returns them as `tool_calls` with
`finish_reason:"tool_calls"`; the client runs the tools and sends the results
as `tool` messages. The server renders the whole history through the chat
template each turn.

### `tool_choice=required` / `tool_choice=named` (constrained generation)

A `"required"` or named `tool_choice` engages the constraint engine to force
a valid call as bare JSON. `tool_choice_to_schema` builds the schema:

- **Named**, or **required with one tool**:
  `{"type":"object","properties":{"name":{"const":"<fn>"},"arguments":<fn-schema>},"required":["name","arguments"]}`.
- **Required with several tools**: `{"oneOf":[…]}`, one such branch per tool.

The `SchemaConstraint` runs with `EngagePolicy::Immediate`, so masking starts
at the first token. The output has no `<tool_call>` wrapper, so the marker
parser is bypassed: `bare_json_to_tool_call` turns the text into the
`tool_calls` envelope. Streaming buffers the JSON and emits one `tool_calls`
delta at the end. If the constraint cannot be built, for example because the
named tool is not in `tools`, the request runs unconstrained and returns no
error.

### EOF recovery

Streaming never completes a partial call. On the non-streaming path, a
Bonsai-style JSON call cut off mid-body (for example at `max_tokens`) is
repaired by closing its open strings and brackets; a truncated Gemma call is
dropped.

OpenAI `parameters` and Anthropic `input_schema` tools render identically.

---

## Chat Templates

Each model's `chat_template.jinja` renders every request. Tool specs render
byte for byte as HuggingFace `apply_chat_template` does. A request's
`enable_thinking` wins over `--enable-thinking`, which wins over the template
default. A template may ignore the flag, so the server reads the thinking
channel off the rendered prompt; see `docs/SAMPLING.md` § "Thinking-budget
enforcement". Streamed deltas never split a UTF-8 code point.

---

## Registry and Claim

### Model registry

`ModelRegistry` is the in-process catalog of snapshot directories. `rmlx
serve` builds it from `--model <path>` (the id is the directory name) or
`--registry <json>`:

```json
{"models":[{"id":"my-id","path":"/path/to/snapshot"},…]}
```

The `id` is optional and defaults to the directory name. Per snapshot:

- `config.json`: required; gives the architecture.
- `chat_template.jinja`: compiled into a `ChatTemplate`; the source is kept
  for tool-format detection.
- `tokenizer.json`: the tokenizer.
- `tokenizer_config.json`: `bos_token` and `eos_token`.
- `generation_config.json`: the model's sampling defaults.

Everything but `config.json` is best-effort: a missing file logs a `warn!`.
`list()` returns the entries in id order.

### Claim file

`claim::try_claim(port) -> Result<MetalClaim, ClaimError>` guards the Metal
claim for one port. The claim file is `/tmp/rmlx.<port>.claim`.

1. It creates the file with `O_CREAT | O_EXCL`.
2. It takes an exclusive non-blocking `flock` on it.
3. It writes its PID into it.

If the file exists, it reads the holder's PID and probes it with
`kill(pid, 0)`. A live holder gets `ClaimError::AlreadyHeld { port,
holder_pid }`. A dead holder (ESRCH) left a stale file. The claimer then
takes the `flock`, re-reads and re-probes the PID under the lock, and only
then reclaims the file, with a `warn`. Any doubt refuses rather than
reclaims. The `flock` is the real gate: a live holder keeps its fd open, so
the lock fails whatever the PID says.

`MetalClaim` is a RAII guard. Dropping it removes the file; the `flock` goes
with the fd. `rmlx serve` handles SIGINT and SIGTERM gracefully, so the guard
drops. After SIGKILL or a crash, the next claimer reclaims the stale file.

**The claim is per port, not per machine.** `rmlx serve` claims its
`--port`. The single-shot GPU commands (`rmlx info`, `rmlx chat`,
`rmlx baseline`) claim the sentinel port `0xCAFE` (51966). A `serve` and a
`baseline`, or two `serve`s on different ports, each hold their own claim and
can run on the GPU at once. The lock is advisory and does not see non-rMLX
MLX processes such as `mlx_lm.server`; the `ClaimError` message prints
unload and stop hints.

---

## Retry Envelope

### Purpose

A transient Metal error can kill a response mid-decode. The retry envelope
replays the request and delivers the rest of the stream without the client
seeing the fault.

### Classification

`classify` sorts an error into `RetryClass::Migratable` or `RetryClass::Fatal`
through `RmlxError::is_migratable` (`crates/rmlx-core/src/error.rs`):

| Class | Variants | Action |
|---|---|---|
| `Migratable` | `Mlx`, `Other` | Replay. |
| `Fatal` | every other variant: `Io`, `Config`, `Loader`, `Quant`, `Model`, `SmokeProbe`, `Oom`, `ArchUnsupported`, `KvStorageMismatch`, `SsdTierAlreadyInstalled`, `Unimplemented`, `KvHardCapExceeded`, `KvCeilingExceeded`, `ContextCeilingExceeded`, `SpeculativePairing` | Surface the error. |

The match has no wildcard arm and lives in `rmlx-core`, where `RmlxError` is
defined. A new variant fails the build until it is classified.

A NaN logit row at prefill is `Other`, so it is replayed; a deterministic one
surfaces after three attempts as 503 `service_unavailable`. An empty prompt is
`Model`, so it fails at once.

### Skip conditions

`retry::is_replayable` turns replay off when:

- `temperature > 0`: the continuation is not deterministic;
- `constraint` is set (`response_format` or a forced `tool_choice`): the
  grammar state would restart.

Without replay, the handler calls `generator.generate(req)` directly.

### Replay mechanism

`replay_stream(…)` runs the generator in a tokio task:

1. The first attempt runs the original `GenerationRequest` and holds the GPU
   admission permit.
2. Each delivered token id goes into `delivered`.
3. On a `Migratable` error, the task rebuilds the request from a
   `RequestPlan`: the original prompt, unchanged, with the original
   `max_tokens`. At `temperature = 0` the engine re-emits the delivered tokens
   first. The task skips `delivered.len()` tokens, asserting they match, and
   forwards the continuation. If they do not match, it surfaces the root error.
4. On a `Fatal` error, or once the attempts run out, it forwards the error.
5. If the client is gone (the channel send fails), the task exits quietly.

`DEFAULT_MAX_RETRIES` is 2, so 3 attempts in all. `RequestPlan` holds only
the clonable fields. It leaves out `constraint`, which already disables replay,
and `gpu_admission`, which the first attempt releases when its blocking task
exits. Dropping the `ReplayStream` (a client cancel) aborts the task at its
next `tx.send().await`.

---

## Session Cache

`SessionCache` tracks sessions keyed by `(model_id, session_id)`. For a
request with `X-Session-Id` on either chat route:

1. `touch(key, prompt_len)` records the session and its `last_used`.
2. `effective_prompt_cache_slots` passes `base_slots + active_count()` to
   the generator. The extra slots keep a live session's KV snapshot from being
   evicted before its next turn.

At `max_sessions` (`--session-cache-max-sessions`, default 64) a new session
evicts the one with the oldest `last_used`. The session cache holds only
timestamps and prompt lengths; the KV lives in the per-arch `PromptCache`.
Unloading a model drops its sessions.

---

## Metrics Endpoints

### `GET /metrics/cache` (JSON)

```json
{
  "models": [
    {
      "model_id": "gemma4-26b",
      "hits": 1, "misses": 1, "evictions": 0, "bytes": 1048576, "hit_rate": 0.5,
      "block_hits": 72, "block_misses": 72,
      "partial_hits": 0, "partial_hit_rate": 0.0, "ssd_hits": 0,
      "kv_cache_bytes": 134217728,
      "metal_peak_alloc_bytes": 4294967296,
      "load_phases": {
        "mmap_ms": 120, "dequant_ms": 340, "gpu_residency_ms": 80,
        "first_kernel_ready_ms": 25, "total_load_ms": 565
      }
    }
  ],
  "ttft": [{ "model_id": "gemma4-26b", "ttft_ms": 312 }],
  "itl": [{ "model_id": "gemma4-26b", "p50_ms": 15.19, "p95_ms": 23.2, "step_mean_ms": 15.8, "step_count": 163 }],
  "tokens_in": 37238,
  "tokens_out": 500,
  "error_counts": {
    "bad_request": 0, "context_overflow": 0, "not_found": 0,
    "oom_load": 0, "oom_kv_cache": 0, "oom_mid_stream": 0,
    "timeout": 0, "upstream": 0, "internal": 0, "rate_limit": 0,
    "admission_sla_503": 0
  }
}
```

- `models`: one entry per resident model. `hits`, `misses`, `evictions`,
  `bytes` and `hit_rate` are the prompt cache's lifetime counts and current
  size. `block_*` are block-level counts, `partial_*` partial-prefix reuse,
  `ssd_hits` RAM misses served from the SSD tier. `kv_cache_bytes` is the last
  request's KV size; `metal_peak_alloc_bytes` the Metal allocator peak.
  `load_phases` is the load timing, when the generator tracks it.
- `ttft`: the last 20 TTFT samples across models, oldest first.
- `itl`: the last 20 per-request inter-token latency aggregates.
- `tokens_in` / `tokens_out`: process-lifetime prompt and completion tokens.
- `error_counts`: process-lifetime counts per error category.

### `GET /metrics` (Prometheus)

The same data in Prometheus text exposition v0.0.4, plus the SSD tier's
spill and hydrate latency histograms (`HIST_BUCKETS_US`) and its
per-namespace byte and eviction series.

### `GET /v1/metrics` (JSON summary)

A rolling request summary in the mlx-vlm shape: `uptime_s`,
`requests_started`, `requests_completed`, `requests_failed`, `in_flight`,
`avg_decode_tok_s`, `avg_request_tok_s` and `last_error`.

---

## Audio

### Speech-to-text: `/v1/audio/transcriptions` and `/v1/audio/translations`

Both take a multipart form. `translations` translates to English.

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | yes | — | Audio in any Symphonia container (WAV, MP3, FLAC, OGG, AAC/`.m4a`, …). Downmixed to mono and resampled to 16 kHz. |
| `model` | no | `whisper-large-v3` | Logged only; the configured snapshot serves every request. |
| `language` | no | `auto` | A language code, or `auto` to detect it. An unknown code is 422. |
| `response_format` | no | `json` | `json`, `text`, `verbose_json`, `srt` or `vtt`. Anything else is 422. |
| `temperature` | no | `0.0` | In `[0.0, 1.0]`; anything else is 422. |
| `prompt` | no | — | Accepted and ignored. |

| `response_format` | Body |
|---|---|
| `json` | `{"text": "..."}` |
| `text` | Plain text. |
| `verbose_json` | `{"task","language","duration","text","segments":[{"id","start","end","text"},…]}` |
| `srt` | SRT, one cue per segment. |
| `vtt` | WebVTT, one cue per segment. |

- Without `--whisper-model-path` and `--whisper-tokenizer-path`, every request
  is 503 `{"error":"audio model not configured; set --whisper-model-path"}`.
- A file over 25 MiB is 422; the transport limit is 26 MiB.
- A malformed form is 422.
- No streaming and no word timestamps; timing is per segment.

`rmlx_audio::transcribe::Transcriber`, shared with `rmlx transcribe`, walks
the audio in 30 s windows. Each window decodes in timestamp mode with the
openai-whisper logit filters (`SuppressBlank`, `SuppressTokens`,
`ApplyTimestampRules`). The emitted timestamps become segments with real
times. The seek advances to the last timestamp, and the previous window's text
is fed back as a `<|startofprev|>` prompt. Filler in the zero-padded tail of
the last window is dropped.

With `language` absent or `auto`, `WhisperModel::detect_language()` runs one
SOT decoder step and takes the argmax over the language tokens. On error it
falls back to English.

The Whisper model and tokenizer load on the first request and stay for the
life of the process; changing the snapshot needs a restart. Audio requests
pass the same admission gate as chat, and hold the GPU permit for the encode
and decode.

### Text-to-speech: `/v1/audio/speech`

Qwen3-TTS synthesizes mono 24 kHz audio from a JSON body:

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | yes | — | Model identifier. |
| `input` | yes | — | Text to synthesize. |
| `voice` | no | `serena` | `serena`, `vivian`, `ryan`, `aiden`, `eric`, `dylan`, `ono_anna`, `sohee` or `uncle_fu`. An unknown voice is 422. |
| `response_format` | no | `wav` | `wav` (RIFF, PCM-16 LE) as `audio/wav`, or `pcm` (raw f32 LE) as `audio/pcm`. |
| `speed` | no | `1.0` | Accepted and ignored. |

Without `--tts-model-path` and `--tts-tokenizer-path` the route is 503 with
type `not_supported`. Another synthesis failure is 500. The model loads on the
first request, stays for the life of the process, and holds the GPU permit
while it synthesizes.

---

## See also

- `docs/CLI.md`: `rmlx serve` flags and the other subcommands.
- `docs/MODELS.md`: per-architecture facts, including the multimodal towers.
- `docs/SAMPLING.md`: sampling parameters, constrained decoding and the
  thinking budget.
- `docs/SPECULATIVE.md`: speculative decoding.
- `docs/PROMPT_CACHE.md`: the prompt cache behind `/metrics/cache`.
- `docs/METRICS_DB.md`: the metrics database the server writes.
