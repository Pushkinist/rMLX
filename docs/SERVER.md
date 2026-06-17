# HTTP Server

Reference for the rMLX HTTP server (`crates/rmlx-server`). Covers architecture,
routes, both API surfaces, streaming, tool calling, embeddings, chat templates,
the model registry, claim-file enforcement, and the retry envelope.

---

## Overview

rMLX exposes two API surfaces from a single binary:

- **OpenAI-compatible** — `POST /v1/chat/completions`, `GET /v1/models`, and
  related management routes.
- **Anthropic-compatible** — `POST /v1/messages`.

Both surfaces share one token stream produced by the `Generator` trait. The
wire schema differs (field names, SSE event types, block shape) but the
underlying engine call is identical.

---

## Architecture

### Stack

```
tokio (multi-thread runtime)
  └─ axum 0.8 Router
       ├─ middleware::timeout_mw  (per-request wall-clock timeout)
       └─ handlers
            ├─ openai.rs    POST /v1/chat/completions, GET /v1/models, …
            ├─ anthropic.rs POST /v1/messages
            └─ embeddings.rs POST /v1/embeddings
```

The server is built by `build_router(state: AppState) -> Router` in `lib.rs`.
`serve(state, host, port)` binds the TCP listener and calls `axum::serve`. All
accepted sockets have `TCP_NODELAY` set to prevent Nagle-induced latency on SSE
frames.

### AppState

`AppState` is cloned cheaply per request (all fields are `Arc`-wrapped). Key
fields:

| Field | Type | Purpose |
|---|---|---|
| `registry` | `Arc<ModelRegistry>` | In-process model catalog |
| `slots` | `Arc<RwLock<Vec<LoadedModel>>>` | Resident model slots (≤ `max_loaded_models`) |
| `embed_slot` | `Arc<RwLock<Option<JinaEmbedModel>>>` | Single resident embedding model |
| `gpu_gate` | `Arc<Mutex<()>>` | Process-wide GPU serialisation lock |
| `gpu_queue` | `Arc<Semaphore>` | FIFO admission gate (1 permit) |
| `gpu_pending` | `Arc<AtomicUsize>` | Count of admitted-and-in-flight requests |
| `max_queue_depth` | `usize` | HTTP 429 threshold; 0 = unlimited |
| `max_loaded_models` | `usize` | LRU eviction threshold; default 1 |
| `session_cache` | `Arc<Mutex<SessionCache>>` | Per-session KV-slot reservation |
| `ttft_store` | `TtftStore` | Rolling ring-buffer of TTFT samples |
| `itl_store` | `ItlStore` | Rolling ring-buffer of ITL aggregate samples |
| `metrics_drainer` | `Option<DrainerHandle>` | SPSC channel to the SQLite writer task |
| `error_counts` | `ApiErrorCounters` | Per-category atomic error counters |
| `tokens_in` / `tokens_out` | `Arc<AtomicU64>` | Process-lifetime token counters |
| `mm_cache` | `Arc<MultimodalCache>` | Shared encoder-output cache for vision towers + Whisper. Byte-budget LRU; sized by `--mm-cache-bytes` (default 512 MiB; `0` disables). Hit/miss/insert events are emitted into the `events` table with `op = "mm_cache_hit" / "mm_cache_miss" / "mm_cache_insert"` and `stage = "mm_cache"`. |

### Single-GPU enforcement

Apple Silicon Metal context is exclusive per process. Two mechanisms work in
tandem:

1. **Claim file** (`/tmp/rmlx.<port>.claim`) — POSIX advisory `flock` prevents
   two rMLX processes from starting on the same port. See the
   [Registry and claim](#registry-and-claim) section.
2. **`gpu_gate` / `gpu_queue`** — inside a process, the `gpu_queue` semaphore
   (1 permit) serialises all forward passes. Requests acquire the permit in
   strict FIFO order via `tokio::sync::Semaphore::acquire_owned` and hold it
   for the entire decode, so only one blocking inference thread runs at a time.

### GPU admission and queue depth

Before entering the semaphore wait, each request increments `gpu_pending` and
checks it against `max_queue_depth`. If `pending > max_queue_depth` (and the
cap is non-zero), the request is rejected with HTTP 429 `rate_limit_error`
immediately, without waiting for the semaphore. This bounds memory consumption
under sustained load. The acquired semaphore permit is held by an RAII guard
dropped on every completion path (success, error, timeout, stream abort).

### Adaptive admission controller

Enabled via `--adaptive-admission` (default OFF). When enabled:

1. **Anticipatory 503** — before the FIFO semaphore wait, the controller
   estimates the end-to-end step latency (admission→final-token wall clock) using
   a sliding-window 2D OLS regressor (`step_ms ≈ β₀ + β₁·prompt_tokens + β₂·kv_bytes`).
   If `est_step > 2 × --step-target-ms` (default 500 ms, alias `--ttft-target-ms`),
   the request is rejected immediately with HTTP 503 `service_unavailable` and
   `Retry-After: 5`.

2. **Adaptive queue depth** — a background tick loop (every 5 s) adjusts
   `max_queue_depth` based on a predicted ITL proxy derived from the same
   regressor:
   - If `est_itl > --itl-target-ms` (default 50 ms) for 3 consecutive ticks →
     depth decreases by 1 (scale-down with hold-ticks anti-thrash gate).
   - If `est_itl < 0.80 × --itl-target-ms` → depth increases by 1
     (scale-up, deadband prevents oscillation).
   - Depth is clamped to `[1, 256]`.

3. **StepMetrics** — after each request completes, the route layer records
   `(prompt_tokens, kv_bytes, step_ms)` into the regressor window.

4. **DB events** — every tick writes a `stage="admission_ctrl"` event to the
   `events` table with `op` set to the `DecisionReason` string (see
   `docs/METRICS_DB.md`). When OFF, the existing open-loop FIFO path is
   byte-identical to the pre-admission-controller behavior.

5. **Adaptive prefill chunk** (`--adaptive-prefill-chunk`, OFF by default) —
   requires `--adaptive-admission`. The controller also adjusts the
   process-wide prefill chunk size using the same deadband shape: raises when
   `est_itl < 0.80 × --itl-target-ms`; lowers after 3 consecutive overload
   ticks. Bounds: `[32, 2048]` tokens. `DecisionReason` values `prefill_chunk_raise`,
   `prefill_chunk_lower`, `prefill_chunk_hold` are emitted as DB events.

6. **Graceful shutdown** — the tick loop task is held as `AppState::admission_handle`
   (`Option<Arc<AdmissionHandle>>`). The `AdmissionHandle` aborts the task on
   `Drop` (fired when the last `AppState` clone is released on runtime teardown).
   The 503 admission error counter is `admission_sla_503`, distinct from the
   `upstream` catch-all engine-error counter.

### Compute placement

Inference is synchronous and runs in `tokio::task::spawn_blocking`. Async is
used only at the HTTP boundary (axum handlers, SSE channel receive) and for
file I/O. The GPU gate mutex inside each generator ensures at most one blocking
thread executes a forward pass at any instant.

---

## Routes

| Method | Path | Handler | Description |
|---|---|---|---|
| `GET` | `/health` | `health` | Liveness probe. Returns `{"ok":true}`. |
| `POST` | `/v1/chat/completions` | `openai::chat_completions` | OpenAI chat, streaming + non-streaming. |
| `GET` | `/v1/models` | `openai::list_models` | List registered models. |
| `POST` | `/v1/models/{id}/load` | `openai::load_model` | Load model synchronously; 200 when resident. |
| `POST` | `/v1/models/{id}/unload` | `openai::unload_model` | Unload resident model. |
| `GET` | `/v1/models/{id}/status` | `openai::model_status` | Resident / unloaded status. |
| `POST` | `/v1/embeddings` | `embeddings::embeddings` | Text and image embeddings (jina-v4). |
| `POST` | `/v1/audio/transcriptions` | `audio::audio_transcriptions` | Whisper STT — transcribe audio to text. |
| `POST` | `/v1/audio/translations` | `audio::audio_translations` | Whisper STT — transcribe audio to English. |
| `POST` | `/v1/audio/speech` | `audio::audio_speech` | Qwen3-TTS speech synthesis (returns 501 until codec decoder lands). |
| `POST` | `/v1/messages` | `anthropic::messages` | Anthropic Messages API, streaming + non-streaming. |
| `GET` | `/metrics/cache` | `openai::metrics_cache` | Prompt-cache hit/miss/bytes + TTFT ring-buffer — JSON. |
| `GET` | `/metrics` | `openai::metrics_prometheus` | Prometheus text exposition v0.0.4. |
| `GET` | `/v1/metrics` | `openai::metrics_v1_summary` | Rolling request-level JSON summary (mlx-vlm compatible). |

A per-request timeout middleware wraps every handler. It reads the optional
`X-Request-Timeout-Seconds` header and caps the effective timeout at
`AppState::max_timeout_secs` (default 600 s, configurable via
`--max-timeout-secs`). Setting the cap to 0 disables the timeout entirely.
Expired requests return HTTP 408 `timeout`.

---

## OpenAI Compatibility

### `POST /v1/chat/completions`

**Request fields** (all optional unless noted):

| Field | Type | Notes |
|---|---|---|
| `model` | string (required) | Registry model id. |
| `messages` | array (required) | `role` + `content`; also `tool_calls`, `tool_call_id`, `name`. |
| `stream` | bool | Default false. |
| `temperature` | f32 | Four-tier fallback: request → `--default-temperature` → `generation_config.json` → 1.0. |
| `max_tokens` | u32 | Capped at `--max-tokens-cap` (default unlimited). |
| `top_p` | f32 | Same four-tier fallback as temperature. |
| `top_k` | u32 | Falls back to `generation_config.json`. |
| `seed` | u64 | |
| `stop` | string or array | Stop sequences. |
| `tools` | array | OpenAI-shaped function specs. |
| `tool_choice` | string or object | `"auto"` \| `"none"` \| `"required"` or `{type:"function",function:{name:…}}`. |
| `response_format` | object | `{type:"text"}` \| `{type:"json_object"}` \| `{type:"json_schema",json_schema:{…}}`. |
| `logprobs` | bool | Return chosen-token logprob per content token. |
| `top_logprobs` | u32 | 0–20; requires `logprobs:true`. |
| `stream_options` | object | `{include_usage:true}` appends a usage chunk before `[DONE]`. |
| `enable_thinking` | bool | `false` suppresses the open `<think>` block on Qwen3-family models. |
| `thinking_budget` | u32 | Cap the reasoning channel at N tokens. |
| `kv_quant` | string | **Issue #26 — per-request KV-codec hot-swap.** Override the KV-cache codec for this request on the resident model, no weight reload. Accepts the same grammar as the `--kv-quant` CLI flag (`"none"`/`"bf16"`, `"k8v4"`, `"k8v8"`, `"planar"`, `"mixed"`, `"mixed_k<kb>g<kg>_v<vb>g<vg>"`, …). `"auto"` selects the per-arch/per-ctx default. Omitted → the server's launch `--kv-quant`. A malformed codec string returns HTTP 400 `invalid_request_error`. |
| `max_ctx` | i32 | **Issue #26 — per-request context-ceiling override.** Re-size the KV-ring virtual ceiling (lazy-grow, #25) for this request only; must be `> 0`. Omitted → the server's launch `--max-ctx`. No weight touch — a ring realloc only. |
| `logit_bias` | object | Token-id (string key) → logit bias (float). |
| `frequency_penalty` | f32 | |
| `presence_penalty` | f32 | |
| `repetition_penalty` | f32 | Falls back to `generation_config.json`. |
| `min_p` | f32 | |
| `echo` | bool | Parsed but rejected with HTTP 501; use `rmlx eval ppl` instead. |

Unknown fields are accepted, debug-logged, and discarded. Fields that indicate
injection intent are explicitly rejected.

### Per-request KV-config hot-swap (issue #26)

`kv_quant` and `max_ctx` change the **KV cache** for one request without
reloading the model weights. Weights are read-only during decode; the KV cache
(codec, ring size) is built per request, so a config switch only rebuilds the
cache — the resident weights stay put. This lets a single `rmlx serve` process
sweep KV codecs / context ceilings, or pick a KV policy per request (aggressive
quant for a 128k request, `none` for a short chat) with zero downtime.

- **Precedence.** A per-request `kv_quant` wins over the launch `--kv-quant`
  (explicit or `auto`) and over the per-ctx auto policy — exactly like a
  startup-explicit flag, but scoped to the one request. `"auto"` defers to the
  generator's per-arch/per-ctx default. Absent → launch default (byte-identical
  to pre-#26 behavior; zero regression).
- **Codec-partitioned prefix cache.** The prompt/prefix cache key is namespaced
  by KV codec, so a prefix cached under one codec **never** serves a request
  running a different codec (the cached K/V bytes are codec-specific). Two
  codecs for the same tokens occupy **distinct** cache slots and coexist — a
  codec switch is a clean cross-codec miss, not a thrash-eviction. See
  `docs/PROMPT_CACHE.md` § "Codec namespacing".
- **`max_ctx`** re-sizes the KV-ring virtual ceiling (#25 lazy-grow) for the
  request; the engine clamps it by the model's `max_position_embeddings` during
  cache build. The `context_length_exceeded` prompt-length guard uses the
  per-request ceiling when the override is present.
- **Single-MLX claim is unaffected** — one model stays resident throughout.
- **Anthropic `/v1/messages`** does not expose these fields (stricter wire
  spec); it always uses the launch default.
- **Deferred:** live SSD-tier reconfiguration (per-request `kv_ssd` toggle) is
  **not** implemented — see `docs/SSD_TIER.md` § "Live reconfiguration (deferred)".

### Multimodal content parts — image + native audio input

A user message's `content` may be a string (text) or an array of content parts.
Image and native-audio parts are extracted from the last user message and routed
through the model's multimodal towers. The tower output (soft tokens) is
scattered into the prompt at the corresponding placeholder positions, then decode
runs from the fused `inputs_embeds` (mirroring mlx-vlm `get_input_embeddings`).

| Part `type` | Shape | Tower | Supported arch |
|---|---|---|---|
| `text` | `{type:"text", text:"…"}` | — | all |
| `image_url` | `{type:"image_url", image_url:{url:"<url\|data-URL>"}}` | SigLIP vision | Gemma4, Gemma3, Qwen3-VL-MoE |
| `input_image` | `{type:"input_image", image_url:"<url>"}` (mlx-vlm shape) | SigLIP vision | same |
| `input_audio` | `{type:"input_audio", input_audio:{data:"<base64>", format:"wav"}}` | Conformer audio (USM) / unified encoder-free | **Gemma4** (e4b/26b Conformer; 12B unified) |

**Native audio (`input_audio`) — Gemma4.** The base64 payload is decoded
(`rmlx-audio` symphonia decoder — WAV/MP3/M4A/etc.), downmixed to mono and
resampled to 16 kHz. The downstream front-end then forks by architecture:

- **Conformer (e4b/26b).** The waveform runs through the Gemma4 USM log-mel
  front-end, then the Conformer `audio_tower` produces `T_sub` audio soft
  tokens. `T_sub` is derived from the encoder's SSCP downsample
  (`≈ mel_frames / 4`).
- **Unified encoder-free (12B `Gemma4UnifiedForConditionalGeneration`).** No mel
  front-end, no Conformer: the raw 16 kHz waveform is chunked into fixed-length
  640-sample frames (`extract_waveform_frames`) and each frame is projected by
  `embed_audio` (`RMSNorm → Linear`, 640→hidden). `num_soft_tokens =
  ceil(num_samples / 640)` (one soft token per 40 ms frame). See *Unified
  (encoder-free) audio* in `docs/MODELS.md`.

In both cases the prompt is spliced with `<|audio>` + `N`×`<|audio|>` +
`<audio|>` after the leading token, and the soft tokens are scattered at the
`<|audio|>` positions; the placeholder count always matches the front-end output
(scatter aligns by construction). One clip per request; >1 is rejected with a
clear error. Combined image+audio in one request is also rejected with a clear
error (on both the Conformer and unified arches), never a silent drop.

**Not-supported path.** Submitting `input_audio` to a model without an audio
tower (text-only, or a vision-only checkpoint) returns **HTTP 503**
`"this model does not accept audio input (no audio tower)"` — never a silent
drop. This mirrors the vision path's `503 no vision tower` rejection.

**Bounds.** Each `input_audio` part is capped at 16 MiB decoded
(`bounds::MAX_INPUT_AUDIO_BYTES`); larger clips return HTTP 400.

**Non-streaming response** (`stream:false`):

```json
{
  "id": "chatcmpl-<hex>",
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

`reasoning_content` is omitted when the model produced no thinking text.
`tool_calls` is omitted when no tool calls were parsed. `logprobs` is omitted
unless `logprobs:true` was requested. When present, `logprobs.content` holds
exactly one entry per emitted completion token — including the first token on a
**prompt-cache exact hit**: the cached first-token logprob is captured
at store time and replayed on the hit path, so a cache hit returns the same
number of `logprobs.content` entries as the equivalent cache miss (it previously
returned N-1).

**Streaming response** (`stream:true`):

Each SSE event carries a `data:` line with a `ChatCompletionChunk` JSON object:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null,"logprobs":{…}}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

When `enable_thinking` is active and the model emits reasoning text, the delta
carries `reasoning_content` instead of `content` for thinking tokens, and
`content` for answer tokens. The two fields are mutually exclusive within a
single chunk.

When `stream_options.include_usage:true`, an extra chunk with `choices:[]` and
a populated `usage` object is appended before `[DONE]`.

**Logprobs in streaming**: per-token `logprobs` appears on each `StreamChoice`
alongside the content delta. Chunks that carry no content token (role preamble,
tool_call, usage chunk) omit the field.

### Stop-sequence truncation

The `stop` parameter (OpenAI `stop`, Anthropic `stop_sequences`) truncates the
generated **content** at the first stop-string match. The contract is uniform
across both API surfaces and both streaming and non-streaming:

- The matched stop string is **excluded** from the returned content — output
  ends just before the match.
- Generation halts at the boundary; OpenAI sets `finish_reason:"stop"`,
  Anthropic sets `stop_reason:"stop_sequence"` and names the match in the
  `stop_sequence` field (non-streaming response body and the streaming
  `message_delta`).
- A multi-element `stop` array → the match at the **earliest byte offset**
  wins; ties break to the **first** string in the array.
- Matching is on the **detokenized text**, not raw token ids, so a stop string
  that **straddles token boundaries** (e.g. `"char" + "lie"` for stop
  `"charlie"`) is detected correctly.
- In **streaming**, the chunk containing the stop is truncated and no post-stop
  chunk is emitted. A partial-match tail (text that could still grow into a
  stop string) is **held back** until it is confirmed not to be a stop, so a
  straddling stop is never half-emitted.
- Stop matching applies to the **content / text channel only**. Reasoning
  (`reasoning_content` / Anthropic `thinking`) is a separate channel and is not
  truncated by `stop`.
- **Tool calls are not truncated by stop sequences.** When the model emits a
  tool call (i.e. `tool_calls` is populated), stop-truncation does not apply to
  that response — uniform across streaming and non-streaming paths.

The shared matcher lives in `rmlx_server::stop_matcher` (`find_stop_match` for
non-streaming, `StopMatcher` for streaming). Empty stop strings are ignored.

### Anthropic `stop_reason` mapping

The `/v1/messages` route maps the engine's OpenAI-style `finish_reason` to the
Anthropic `stop_reason` field via `map_stop_reason` in
`crates/rmlx-server/src/anthropic/route.rs`:

| Engine `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `"stop"` (natural EOS) | `"end_turn"` |
| `"length"` (token cap) | `"max_tokens"` |
| `"tool_calls"` | `"tool_use"` |
| anything else / `None` | `"end_turn"` |

`"stop_sequence"` is **never** produced by `map_stop_reason`. It is set
exclusively by the stop-matching path in `blocking.rs` /
`streaming.rs` when a `stop_sequences` entry actually matched — and that
path bypasses `map_stop_reason` entirely. This keeps the two cases cleanly
separated: real stop-string hit → `stop_reason:"stop_sequence"` +
`stop_sequence:"<matched>"` (and `null` on the normal path). Natural EOS
always yields `"end_turn"` with `stop_sequence:null`.

### `GET /v1/models`

Returns the OpenAI-shaped model list:

```json
{"object":"list","data":[{"id":"my-model","object":"model","owned_by":"rmlx","permission":[]}]}
```

All models in the registry are listed regardless of resident status.

### Model lifecycle endpoints

- `POST /v1/models/{id}/load` — calls `AppState::ensure_loaded`. Blocks until
  the model is resident. Returns 200 with `{"id":"…","status":"loaded"}` on
  success. Returns 404 if the id is not in the registry, 507 on OOM, 503 on
  loader failure.

  Accepts an optional JSON body with a
  `keep_alive` integer field (Ollama / LM-Studio compatible). Negative pins
  the model forever; `0` unloads after the next request finishes; positive
  is the idle TTL in seconds. Absent = inherit the slot's current policy
  (env `RMLX_KEEP_ALIVE` > `--idle-timeout-secs` flag > 15-min default).

  ```bash
  # Pin "gemma-4-e4b" forever (until the rmlx serve process exits).
  curl -X POST -d '{"keep_alive": -1}' http://127.0.0.1:8080/v1/models/gemma-4-e4b/load
  # Load + auto-unload after 2 minutes idle.
  curl -X POST -d '{"keep_alive": 120}' http://127.0.0.1:8080/v1/models/gemma-4-e4b/load
  ```

- `POST /v1/models/{id}/unload` — calls `AppState::unload`. Returns 200 with
  `{"id":"…","status":"unloaded"}` whether or not the model was resident.
- `GET /v1/models/{id}/status` — returns `{"id":"…","status":"loaded"|"unloaded"}`.

### Keep-alive on compat routes

The OpenAI-compatible chat completions route (`/v1/chat/completions`) and
the Anthropic (`/v1/messages`) route do **not** parse a per-request
`keep_alive` body field — matching the broader ecosystem (cf. ollama#11458).
They still **reset** the per-model keep-alive timer on every successful
`ensure_loaded`, so an active client keeps the model resident without any
explicit field.

`/v1/embeddings` (jina) and `/v1/audio/*` (Whisper STT, Qwen3-TTS) use a
separate process-lifetime cache (`embed_slot`, `audio_model`) that is **not**
subject to the keep-alive TTL today: those slots stay resident for the
lifetime of the `rmlx serve` process and do not feed into the per-model
keep-alive lifecycle. A follow-up may unify them with the main slot
lifecycle.

### Decode-lease semantics

Every generation path acquires an active-decode lease guard for the
duration of the response. While the lease is held, the keep-alive timer
**cannot** unload the model — when the TTL fires it observes the
non-zero lease count, logs `keep_alive: decode in flight — deferring
unload`, and reschedules a fresh TTL period. This guarantees:

1. Streaming responses always complete — the SSE stream owns the guard via
   the `GuardedStream` wrapper, so the guard drops only when the stream is
   fully consumed or the client disconnects.
2. Blocking responses hold the guard across the entire `.await`.
3. The cooperative same-process evict path (loading a different model when
   `max_loaded_models == 1`) still proceeds — it bypasses the TTL gate
   because the evicting load goes through `ensure_loaded`'s LRU branch
   rather than the timer. (A request that arrives mid-evict simply waits
   for the GPU admission semaphore.)

### Cooperative same-process evict

`POST /v1/models/{id}/load` for a *different* model id while the slot is
full and `max_loaded_models == 1` immediately unloads the resident model —
the LM-Studio "Auto-Evict" semantics. The cross-process claim file at
`/tmp/rmlx.<port>.claim` is **not** affected (the rMLX server still holds
it for its full lifetime); only the in-process slot is freed. Loading a
second model in *another* `rmlx serve` process is still blocked by the
claim file — that's a binary single-MLX-per-machine guarantee.

### Error responses

All errors follow the OpenAI error envelope:

```json
{"error":{"message":"…","type":"<error_type>"}}
```

Common `error_type` values:

| HTTP | `error_type` | Condition |
|---|---|---|
| 400 | `invalid_request_error` | Bad field, out-of-range param, unsupported feature. |
| 400 | `context_length_exceeded` | Prompt exceeds the model's effective max context. |
| 404 | `not_found_error` | Model id not in registry. |
| 408 | `timeout` | Per-request wall-clock timeout exceeded. |
| 429 | `rate_limit_error` | GPU admission queue full. |
| 500 | `internal_error` | NaN logits, smoke-probe failure, task panic. |
| 503 | `service_unavailable` | Loader failure or engine error (non-OOM). Counter label: `upstream`. |
| 503 | `admission_sla_exceeded` | Anticipatory SLA rejection (adaptive admission controller). Includes `Retry-After: 5` header. Counter label: `admission_sla_503`. |
| 507 | `oom_during_load` | Weight-load OOM. |
| 507 | `oom_kv_cache` | KV-cache allocation OOM. |
| 507 | `oom_mid_stream` | Mid-decode OOM. |

Process-lifetime counters for each category are exposed via `GET /metrics/cache`
under `error_counts`.

### `X-Session-Id` header

When a request includes an `X-Session-Id` header, the value is registered in
the session cache keyed by `(model_id, session_id)`. This increases the
effective prompt-cache slot count (`base_slots + active_count`) passed to the
generator, preventing FIFO eviction from clobbering a live session's KV
snapshot across turns. See [Session cache](#session-cache).

---

## Anthropic Compatibility

### `POST /v1/messages`

**Request fields**:

| Field | Type | Notes |
|---|---|---|
| `model` | string (required) | Registry model id. |
| `max_tokens` | u32 (required) | No default; missing field returns 400. |
| `messages` | array (required) | `role` + `content` (string or block array). |
| `system` | string or array | System prompt; injected before the first user turn via chat template. |
| `temperature` | f32 | Same four-tier fallback as the OpenAI route. |
| `top_p` | f32 | |
| `top_k` | u32 | |
| `stop_sequences` | array | Stop sequences. |
| `stream` | bool | Default false. |
| `tools` | array | Anthropic-shaped: `name`, `description`, `input_schema` (not `parameters`). |
| `tool_choice` | object | `{type:"auto"|"any"|"tool", name?:"…"}`. |
| `metadata` | object | Accepted and debug-logged; ignored. |

**Non-streaming response**:

```json
{
  "id": "msg-<hex>",
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

The `thinking` block is included only when the model produced reasoning text
(Qwen3-family with `enable_thinking` active). The `tool_use` block is included
when the model emitted a parseable tool call. In the Anthropic surface,
`input` is a JSON object — not a JSON-stringified string as in the OpenAI
`arguments` field.

**Streaming response**:

Events follow the Anthropic streaming protocol:

```
event: message_start
data: {"type":"message_start","message":{…}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
```

When the model produces thinking text, a `thinking` block precedes the `text`
block, with `thinking_delta` events carrying the reasoning fragments.

---

## Embeddings

### `POST /v1/embeddings`

Supports jina-embeddings-v4 text and image embeddings.

**Request**:

| Field | Type | Notes |
|---|---|---|
| `model` | string | Registry model id resolving to a `JinaEmbeddingsV4Model`. |
| `input` | string, array, or image object | Text: `"string"` or `["a","b"]`. Image: `{"image":"<data-URI|base64|path>"}` or a list of such objects. |
| `encoding_format` | string | `"float"` (default) or `"base64"`. |
| `dimensions` | usize | Matryoshka truncation. Must be one of `{128,256,512,1024,2048}`. |
| `task` | string | LoRA task: `retrieval` (default), `text-matching`, `code`. |
| `prompt_name` | string | `query` (default) or `passage`. `text-matching` always uses `Query` regardless of this field. |
| `return_multivector` | bool | Return per-token multi-vector embeddings (`[[f32;128];seq]`) instead of a single pooled vector. |

Text inputs are prepended with the task prefix `"{Query|Passage}: {text}"` per
the jina convention. No BOS token is added; `add_special_tokens=false` is used.

**Response**:

```json
{
  "object": "list",
  "data": [{"object":"embedding","index":0,"embedding":[0.1,0.2,…]}],
  "model": "my-embed-model",
  "usage": {"prompt_tokens":10,"total_tokens":10}
}
```

`embedding` is a `[f32]` array for single-vector float output, `[[f32]]` for
multi-vector output, or a base64 string when `encoding_format="base64"`.

**Model placement**: the embedding model is not a causal LM and is not placed
in `AppState::slots`. It resides in `AppState::embed_slot` and is loaded lazily
on the first embedding request. Per-request `apply_task` swaps the live LoRA
adapter inside the GPU critical section. Text and image inputs are disjoint —
a single request embeds either all text or all images, not a mix.

---

## Tool Calling

### Parser architecture

Tool-call output is parsed from the model's raw token stream by
`ToolCallStreamParser`. The parser is arch-specific; the format is detected
once at registry build time from the snapshot's `chat_template.jinja` source
and architecture string, then cached in `ModelEntry`.

Three formats are supported:

| `ToolCallFormat` | Architecture | Syntax |
|---|---|---|
| `Qwen3XmlFunction` | `Qwen3_5MoeForCausalLM` (Qwen3.6) | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` |
| `Qwen3JsonToolCall` | `Qwen3ForCausalLM` (Bonsai) | `<tool_call>{"name":"…","arguments":{…}}</tool_call>` |
| `GemmaToolCall` | `Gemma4ForConditionalGeneration` | `<|tool_call>call:NAME{key:val}<tool_call|>` |

For `GemmaToolCall`, the `<|tool_call>`, `<tool_call|>`, and `<|"|>` markers
are registered as special tokens and stripped by `tokenizer.decode`. The engine
reconstructs them from raw token ids before feeding the parser.

The parser is stream-friendly: token pieces fed in arbitrary BPE-aligned splits
produce the same parse result as the same string fed all at once. Multiple
`<tool_call>` blocks may appear in sequence.

### Template support probe

At registry build, each compiled `ChatTemplate` is probed with a minimal
one-tool render (`probe_tools_supported`). The result is stored in
`ModelEntry::tools_supported`. When `false`, tool injection is skipped and the
request proceeds without tools rather than returning 500.

### Multi-turn tool loop

The OpenAI client drives the multi-turn tool loop externally. On each turn:

1. The client sends a request with a `tools` array.
2. The model emits one or more `<tool_call>` blocks.
3. The server parses them and returns `tool_calls` in the response with
   `finish_reason:"tool_calls"`.
4. The client executes the tools, appends `tool`-role result messages, and
   sends the next request.
5. The server renders the full conversation history (including tool results)
   through the chat template on each turn.

### `tool_choice=required` / `tool_choice=named` (constrained generation)

When `tool_choice` is `"required"` or a named function (`{type:"function",function:{name:"…"}}`),
the server engages the **constraint engine** (tokenizer-aware JSON byte-FSM) to force the
model to emit a valid tool call as bare JSON, bypassing the marker-based stream parser.

**Schema synthesis** (`tool_choice_to_schema`):

- `Named` — builds a single-branch JSON Schema `{"type":"object","properties":{"name":{"const":"<fn>"},"arguments":<fn-schema>},"required":["name","arguments"]}`.
- `Required` with one tool — same single-branch shape (no `oneOf` wrapper needed).
- `Required` with multiple tools — synthesises `{"oneOf":[…branches…]}` where each branch is the same single-tool schema for one of the declared functions.

The constraint is loaded as a `SchemaConstraint` with `EngagePolicy::Immediate` so generation
begins in constrained mode from the first token.

**Bare-JSON mode**:

Because the constraint engine produces bare JSON output (no `<tool_call>` wrapper), the
marker-based `ToolCallStreamParser` is bypassed. After generation finishes, the accumulated
text is parsed by `bare_json_to_tool_call` and wrapped into the standard OpenAI
`tool_calls` envelope with `finish_reason:"tool_calls"`.

In streaming mode the accumulated bare JSON is buffered internally (not forwarded as content
chunks) and converted to a `tool_calls` delta at the done-token boundary.

### EOF recovery and the `allow_eof_recovery` invariant

`ToolCallStreamParser` carries an `allow_eof_recovery: bool` flag (default `false`).

**Invariant:**
- **Streaming path** — `allow_eof_recovery` stays `false`. Recovery logic is never triggered
  mid-stream to prevent false-positive completion from partial BPE tokens.
- **Non-streaming / finalize path** — the caller explicitly invokes `parser.finalize()` once
  all tokens have been consumed. `finalize()` flips `allow_eof_recovery=true` and runs
  `run_eof_recovery()` exactly once (idempotent on repeated calls).

**Truncation recovery** (`run_eof_recovery`):

- `Qwen3JsonToolCall` (Bonsai) — if generation ended mid-call (e.g. `max_tokens` hit), the
  pending JSON body is repaired by `balance_truncated_json`, which closes any open strings,
  braces, and brackets. The repaired JSON is then parsed as a Hermes call. This recovers
  tool calls that were silently dropped on EOS/length truncation.
- `Qwen3XmlFunction` — delegates to the existing `finalize_current_call` path.
- `GemmaToolCall` — truncated blocks are dropped; no recovery (Gemma marker syntax cannot be
  safely reconstructed from partial state).

`balance_truncated_json` returns `None` when the input is already well-formed (no repair
needed) and `Some(repaired)` otherwise. It handles mid-string truncation, dangling escape
sequences, and arbitrarily nested `{}`/`[]` containers.

### Tool normalization

Inbound OpenAI `tools` (with `parameters`) and Anthropic `tools` (with
`input_schema`) are normalized to a common `NormalizedTool` shape before
being passed to the chat template renderer. The `PythonCompatFormatter`
produces Python-compatible JSON spacing (`": "` and `", "`) so that rendered
tool specs are byte-identical to HuggingFace `apply_chat_template` output.

---

## Chat Templates

### Rendering

`ChatTemplate` wraps a minijinja environment compiled from
`<snapshot>/chat_template.jinja`. It is constructed once per model at registry
build and is reused across requests.

`ChatTemplate::render(messages, opts)` accepts:

- `messages` — a slice of `ChatMessageTpl` structs (`role`, `content`,
  optional `tool_calls`, `tool_call_id`, `name`).
- `opts` — `RenderOpts` containing `bos_token`, `eos_token`,
  `add_generation_prompt`, `tools`, and `enable_thinking`.

The `{% generation %}` / `{% endgeneration %}` markers used by HuggingFace for
loss-masking are stripped before compilation (replaced with empty Jinja
comments) since minijinja rejects unknown statements.

### Python-compatible JSON serialisation

Chat templates commonly pass tool specs through the `| tojson` filter. The
default minijinja `tojson` produces compact JSON (no spaces), while Python's
`json.dumps` uses `": "` (colon-space) and `", "` (comma-space). A custom
`PythonCompatFormatter` replaces the built-in `tojson` filter so rendered tool
specs are byte-identical to HuggingFace output.

### Thinking mode

`enable_thinking` in `RenderOpts` controls the Qwen3-family `<think>` block:

- `Some(false)` — injects `enable_thinking = false` into the Jinja context,
  triggering the template's no-think branch (emits a closed `<think></think>`
  block).
- `None` or `Some(true)` — leaves the variable undefined; the template falls
  through to its default (open `<think>\n` block, byte-identical to HuggingFace
  output).

The variable is never defined as `true` — defining it would not change
behavior relative to `None`, since the template tests `enable_thinking is
defined and enable_thinking is false`.

Per-request `enable_thinking` takes precedence over `AppState::default_enable_thinking`
(`--enable-thinking` startup flag), which in turn takes precedence over the
template default.

### Detokenizer and UTF-8 healing

`StreamingDetokenizer` in `detokenizer.rs` manages the streaming decode loop
for all architectures. It uses a full-prefix decode model (decode the growing
token-id prefix at every step, diff against the prior decoded string) rather
than the HuggingFace `DecodeStream` which has known cross-request state leakage
issues.

**UTF-8 healing**: byte-level BPE tokenizers (Qwen3.6 uses `ByteLevel` decoder)
may produce a replacement character U+FFFD (`\u{FFFD}`) when a multi-byte
codepoint's bytes straddle two token ids. The detokenizer withholds any
delta that would advance past a `\u{FFFD}`-terminated boundary, accumulating
further tokens until the codepoint completes. `finalize()` flushes the
remaining bytes lossy at true end-of-stream.

No leading-space stripping is applied for the current target models (Gemma3/4
and Qwen3/Qwen3.6), as neither uses the strict SentencePiece `Strip` decoder
variant that would require it.

---

## Registry and Claim

### Model registry

`ModelRegistry` is an in-process catalog of known snapshot directories. It is
built at startup from either:

- `--model <path>` — single snapshot, id derived from directory basename.
- `--registry <json>` — a JSON file of the form:
  ```json
  {"models":[{"id":"my-id","path":"/path/to/snapshot"},…]}
  ```
  The `id` field is optional; if absent, the basename is used.

For each snapshot, the registry loads (all best-effort — missing files produce
a `warn!` but do not skip the entry except for `config.json`):

- `config.json` — required; determines the architecture string.
- `chat_template.jinja` — compiled into a `ChatTemplate`. Raw source retained
  for tool-format detection.
- `tokenizer.json` — loaded into a `tokenizers::Tokenizer`.
- `tokenizer_config.json` — provides `bos_token` and `eos_token`.
- `generation_config.json` — provides per-model sampling defaults
  (`temperature`, `top_k`, `repetition_penalty`, etc.).

Entries are stored in a `BTreeMap` and returned alphabetically by `list()`.

### Claim file

`try_claim(port) -> Result<MetalClaim, ClaimError>` enforces the single-MLX-
process-per-Mac constraint.

On call:

1. Creates `/tmp/rmlx.<port>.claim` with `O_CREAT | O_EXCL`.
2. Acquires an exclusive non-blocking `flock` (POSIX advisory lock) on the file.
3. Writes the current PID as a decimal string into the file.

If the file already exists:

- The holder's PID is read from the file body.
- The PID is probed with `kill(pid, 0)`. If the holder is **alive**,
  `ClaimError::AlreadyHeld { port, holder_pid }` is returned — a live claim is
  never stolen (the single-MLX invariant).
- If the holder is **dead** (ESRCH), the claim is stale: it was left by a
  process that died without running `Drop` (SIGKILL, crash, power loss). A
  non-blocking `flock` confirms no live fd still holds the lock, the file is
  reclaimed (truncated, rewritten with our PID), and a `warn` is logged.

`MetalClaim` is a RAII guard. Dropping it removes the claim file and releases
the lock (the `flock` is released automatically when the fd closes). The HTTP
server installs a SIGINT/SIGTERM graceful-shutdown handler so a signalled
`rmlx serve` runs `Drop` and removes the claim proactively; SIGKILL/crash are
covered by the dead-PID reclaim above.

Non-server GPU CLI operations (`rmlx info`, `rmlx chat`, `rmlx baseline`) use
the sentinel port `0xCAFE` (51966) to represent "a single-shot GPU op in
progress."

The advisory lock prevents two rMLX processes from clobbering each other but
does not block Python `mlx_lm.server` or other non-rMLX processes. Unload/stop
hints are printed when `ClaimError` is returned.

---

## Retry Envelope

### Purpose

When a Metal-level error interrupts a streaming response mid-decode (GPU
watchdog kill, transient dispatch error), the client would otherwise receive a
truncated stream. The retry envelope transparently reconstructs the response
without client involvement.

### Classification

Errors are classified as `RetryClass::Migratable` or `RetryClass::Fatal`:

| Class | Conditions | Action |
|---|---|---|
| `Migratable` | `RmlxError::Mlx` (any Metal error), `RmlxError::Other` (engine panic) | Replay permitted. |
| `Fatal` | `SmokeProbe` (NaN logits), `Oom`, `Config`, `Loader`, `Quant`, `Model`, `Io` | No retry; surface error. |

### Skip conditions

Token-replay retry is disabled when any of the following hold:

- `temperature > 0` — decode is non-deterministic.
- `n > 1` — multiple choices requested.
- Guided decoding (`constraint` is `Some`) — the FSM resets on every request.

When retry is disabled, the handler calls `generator.generate(req)` directly.

### Replay mechanism

`replay_stream(…)` wraps the generator call in a tokio task:

1. Attempt 1 runs with the original `GenerationRequest` (holding the GPU
   admission permit).
2. Each delivered token id is appended to `delivered`.
3. On `Migratable` error: the request is rebuilt from a `RequestPlan` with
   `prompt_tokens = original_prompt_tokens ++ delivered`. At `temperature=0`
   the decode is deterministic, so the model reproduces the same continuation.
   The task skips the first `delivered.len()` tokens of the new attempt,
   asserting prefix identity, and forwards only the new continuation.
4. On `Fatal` error or attempt exhaustion: the error is forwarded to the caller.
5. On channel-send failure (client disconnect): the task exits silently — that
   is intentional cancellation, not a transient fault.

The default retry limit is 2 retries (3 total attempts).

`RequestPlan` holds only the clonable fields needed to reconstruct the request.
Non-clonable fields (`constraint`, `gpu_admission`) are excluded: `constraint`
already disqualifies retry via the skip-condition check; `gpu_admission` is
released when the first attempt's blocking task exits, and subsequent attempts
run without it.

The `ReplayStream` owns a `JoinHandle`. Dropping the stream (HTTP client
cancel) calls `handle.abort()`, stopping the spawned engine task at the next
`tx.send().await` yield point with no further engine work.

---

## Session Cache

`SessionCache` tracks active sessions keyed by `(model_id, session_id)`. On
each request carrying `X-Session-Id`:

1. `cache.touch(key, prompt_len)` is called, returning `true` (hit) or
   `false` (miss) and updating `last_used`.
2. `active_count()` is read and added to the base prompt-cache slot count:
   `effective_slots = base_slots + active_count`. This reserves headroom in the
   per-arch `PromptCache` so a live session's KV snapshot is not FIFO-evicted
   before the next turn arrives.

When `active_count` reaches `max_sessions` (default 64, configurable via
`RMLX_SESSION_CACHE_MAX_SESSIONS`), the entry with the oldest `last_used`
timestamp is evicted before inserting the new session.

KV tensors live inside the per-arch `PromptCache` global; the session cache
holds only timestamps and prompt lengths. Two sessions always produce distinct
`PromptCache` lookups keyed by different token sequences. When a model is
unloaded, all session-cache entries for that model are removed.

---

## Metrics Endpoints

### `GET /metrics/cache` (JSON)

Returns a JSON object with:

- `prompt_cache` — hit count, miss count, total bytes across all prompt-cache
  namespaces.
- `ttft` — rolling ring-buffer (last 20 samples) of per-model TTFT values in
  milliseconds.
- `last_itl` — last ITL aggregate per model (p50, p95, mean, step count).
- `error_counts` — process-lifetime per-category counts (`bad_request`,
  `context_overflow`, `not_found`, `oom_load`, `oom_kv_cache`,
  `oom_mid_stream`, `timeout`, `upstream`, `internal`, `rate_limit`,
  `admission_sla_503`). The `admission_sla_503` counter is incremented
  specifically by adaptive-admission anticipatory SLA rejections, distinct from the
  `upstream` catch-all.
- `tokens_in` / `tokens_out` — process-lifetime cumulative token counts.

### `GET /metrics` (Prometheus)

Exposes the same data in Prometheus text exposition format v0.0.4. Includes
SSD-tier histograms for spill and hydrate latency (buckets at 100, 500, 1 000,
5 000, 10 000, 50 000, 100 000, 500 000, 1 000 000 µs) and per-namespace
gauges for on-disk bytes and eviction counters.

### `GET /v1/metrics` (JSON summary)

Rolling request-level summary (mlx-vlm compatible). Includes `uptime_s`,
`requests_started`, `requests_completed`, `requests_failed`, and a short-window
decode TPS estimate.

---

## Audio

Audio I/O: Whisper speech-to-text and Qwen3-TTS text-to-speech.

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text in the original language. |
| `POST` | `/v1/audio/translations` | Transcribe audio and translate to English. |
| `POST` | `/v1/audio/speech` | Synthesize speech from text. Returns `audio/wav`. |

STT returns HTTP 503 when the server was started without `--whisper-model-path`.
TTS returns HTTP 503 when started without `--tts-model-path`.

### Multipart form fields

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | yes | — | Audio bytes. Any Symphonia-supported container (WAV, MP3, FLAC, OGG, AAC/`.m4a`, …). Stereo is downmixed to mono and any sample rate is resampled to 16 kHz internally. |
| `model` | no | `whisper-large-v3` | Model identifier (logged; routing is fixed to the configured snapshot). |
| `language` | no | `auto` | BCP-47 language code, or `auto` (default) to detect language automatically via a single SOT decoder step. Unknown explicit codes return 422. |
| `response_format` | no | `json` | `json` \| `text` \| `verbose_json` \| `srt` \| `vtt`. Unknown values return 422. |
| `temperature` | no | `0.0` | Decoding temperature in `[0.0, 1.0]`. Malformed or out-of-range values return 422. |
| `prompt` | no | — | Accepted, ignored at v1. |

### Response formats

Transcription is **long-form**: the engine walks the audio in 30 s windows and
emits real per-segment timestamps (not a single hardcoded block).

| `response_format` | Body |
|---|---|
| `json` (default) | `{"text": "..."}` |
| `text` | Plain text string. |
| `verbose_json` | `{"task":"transcribe","language":"en","duration":<seconds>,"text":"...","segments":[{"id","start","end","text"},…]}` |
| `srt` | SRT subtitle, one cue per segment with real `HH:MM:SS,mmm` times. |
| `vtt` | WebVTT, one cue per segment with real `HH:MM:SS.mmm` times. |

### Constraints

- Any sample rate / channel count is accepted — the server downmixes to mono and
  resamples to 16 kHz (linear) before mel extraction.
- Maximum audio file size: 25 MiB. Transport body limit is 26 MiB (25 MiB + 1 MiB multipart framing).
- No streaming (SSE timestamps) — deferred to v2.
- No word-level timestamps — segment-level timing only (real per-segment times).

### Model caching

The Whisper model + tokenizer are loaded on the **first** request and cached for
the server lifetime. Subsequent requests skip disk I/O. A server restart is
required to change the snapshot.

### Admission

Audio requests go through the same `admit_request` → `gpu_queue` FIFO semaphore
as LLM chat requests. They are counted toward `max_queue_depth` and receive HTTP
429 when the queue is full. The GPU permit is held for the duration of the Whisper
encode + decode and released on completion.

### 503-when-unset behaviour

If `--whisper-model-path` or `--whisper-tokenizer-path` is absent at startup,
every audio request returns:

```json
{"error": "audio model not configured; set --whisper-model-path"}
```

with HTTP 503.

### Long-form transcription

Audio of any length is transcribed by the shared long-form engine
(`rmlx_audio::transcribe::Transcriber`, also used by `rmlx transcribe` — one core,
not two). The engine:

1. Walks the audio in 30 s windows. Each window runs the Whisper decoder in
   **timestamp mode** with the full openai-whisper logit-filter chain
   (`SuppressBlank` + `SuppressTokens` + `ApplyTimestampRules`).
2. Parses the emitted timestamp tokens into segments with real cumulative times,
   advances the window seek by the last consumed timestamp, and feeds the previous
   window's text back as a `<|startofprev|>` prompt (previous-text conditioning).
3. Drops filler hallucinated in the 30 s zero-pad tail of the final short window.

Decoding is greedy at temperature 0 — output is deterministic across runs.

Silero VAD weights remain vendored at
`crates/rmlx-audio/assets/silero_vad_16k.safetensors` (MIT) for future
voice-activity gating; the current long-form path uses fixed 30 s windows with
timestamp-driven seek rather than VAD pre-segmentation.

### `POST /v1/audio/speech` — Qwen3-TTS

Synthesizes mono 24 kHz PCM from text. Returns `Content-Type: audio/wav` (default)
or `audio/pcm` when `response_format=pcm`.

JSON request body:

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | yes | — | Model identifier (e.g. `qwen3-tts`). |
| `input` | yes | — | Text to synthesize. |
| `voice` | no | `serena` | Voice name. Available: `serena`, `vivian`, `ryan`, `aiden`, `eric`, `dylan`, `ono_anna`, `sohee`, `uncle_fu`. |
| `response_format` | no | `wav` | Output format: `wav` (44-byte RIFF header + PCM-16 LE) or `pcm` (raw f32-LE). |
| `speed` | no | `1.0` | Accepted but not applied at v1 (codec speed is fixed). |

Returns HTTP 503 when `--tts-model-path` is absent. Unknown voice names return 422.

#### Language auto-detection (Whisper)

When `language` is absent or `"auto"` in `/v1/audio/transcriptions`, the handler
runs `WhisperModel::detect_language()` — a single SOT decoder step followed by
argmax over the 100 large-v3 language tokens (`<|en|>`=50259 … `<|yue|>`=50358) —
and uses the detected token to build the SOT sequence. Falls back to English
(50259) on error.

---

## See also

- `docs/CLI.md` — `rmlx serve` flags, `rmlx chat`, `rmlx info`, and other
  subcommands.
- `docs/MODELS.md` — supported architectures, quantization matrix, snapshot
  layout.
- `docs/SPECULATIVE.md` — Eagle3 speculative decoding, chunked prefill,
  restricted-vocab hot-path.
- `docs/KV_CACHE.md` — KV cache quantization presets and primitives.
- `docs/METRICS_DB.md` — SQLite metrics schema, ingest pipeline, operating
  rules.
