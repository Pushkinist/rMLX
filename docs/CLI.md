# rMLX CLI Reference

## Overview

`rmlx` is the single compiled binary produced by `cargo build --release`. It
serves an OpenAI-compatible HTTP API, runs one-off chat sessions, inspects
model snapshots, and manages the metrics database. All subcommands share a
common global flag (`--log`) for verbosity control.

```text
rmlx [--log <level>] <subcommand> [flags]
```

Subcommands:

| Subcommand | Purpose |
|---|---|
| `serve` | OpenAI + Anthropic-compatible HTTP inference server |
| `chat` | Interactive REPL for ad-hoc model testing |
| `info` | Print arch + quant metadata for a snapshot; no inference |
| `baseline` | Measure load time, TTFT, decode TPS, peak RSS |
| `healthcheck` | Shell-able readiness probe; JSON or plain text output |
| `metrics` | Metrics database management (schema, query, export) |
| `eval ppl` | Offline perplexity evaluation over a text corpus |
| `profile list` | List named `serve` launch profiles |

Additionally, `qwen36_diag` is a standalone diagnostic binary (built from
`crates/rmlx-cli/src/bin/qwen36_diag.rs`) for low-level Qwen3.6 forward-pass
verification.

---

## Global flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--log` | `info \| debug \| verbose` | `info` | Log verbosity preset. `RUST_LOG` overrides this when set. `info` keeps per-token and per-layer trace events off; `debug` enables per-step phase events; `verbose` enables per-token, per-FFI, and per-layer trace events. |

---

## Subcommands

### `serve`

Starts the HTTP inference server. Exposes an OpenAI-compatible API on
`http://<host>:<port>/v1/`.

```bash
rmlx serve --model /path/to/snapshot
rmlx serve --registry /path/to/registry.json --port 9000
rmlx serve --profile myrun
```

**Model source** — exactly one of `--model`, `--registry`, or the profile's
`model`/`registry` key must be provided. `--model` and `--registry` are
mutually exclusive.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | — | Path to a model snapshot directory. Mutually exclusive with `--registry`. |
| `--registry` | path | — | Path to a JSON registry file. Format: `{"models":[{"id":"name","path":"/abs/path"},…]}`. Mutually exclusive with `--model`. |
| `--profile` | string | — | Named launch profile from `<RMLX_HOME>/profiles.toml`. CLI flags override profile values. See `rmlx profile list`. |
| `--port` | u16 | 8080 | TCP port to listen on. |
| `--host` | string | `127.0.0.1` | Host or IP to bind. |
| `--device` | `cpu \| gpu` | `gpu` | Inference device. |
| `--kv-quant` | string | `auto` | KV cache quantization preset: `auto` (arch default), `bf16` / `none` (unquantized), `k8v4`, `k8v8`, `planar`, `planar3` (3-bit V PlanarQuant), `k8vturbo3` (q8_0 K + TurboQuant 3-bit Lloyd-Max V; auto default for Gemma4 small, opt-in for other archs), `tsym4` (symmetric WHT-4 K + tq4 V; rejected on Qwen MoE arch with exit 78), `planar_k` (K-axis PlanarQuant 4-bit, V=bf16; rejected on Qwen MoE arch via `QwenMoePlanarKRejected`), `tsym3` (WHT-3 K + WHT-3 V symmetric; rejected on Qwen MoE arch). Mutually exclusive with `--cache-type-k`/`--cache-type-v`. |
| `--kv-preset` | string | — | Named KV-cache preset. Resolves to a `KvQuant` by name. Mutually exclusive with `--kv-quant`, `--cache-type-k`, `--cache-type-v`, `--kv-bits`. See `docs/KV_QUANT.md §Presets`. Available: `fp16`, `q8`, `speed`, `quality`, `planar`, `planar3`, `k_only_planar`. |
| `--kv-quant` | string | `auto` | KV cache quantization preset: `auto` (arch default), `bf16` / `none` (unquantized), `k8v4`, `k8v8`, `planar`, `planar3`, `k8vturbo3`, `k8vturbo3tcq` (Viterbi trellis 3-bit V; reuses turbo3 Lloyd-Max codebook with 4-state Viterbi-optimal trellis assignment instead of nearest-centroid; decoder bit-identical to plain turbo3; CPU encode on hot path; MSL kernel parity-tested but parked as future-reference hook; distinct SSD layout tag prevents cross-codec hydrate; `--ctv` alias: `turbo3_tcq`), `k8vturbo2` (native 2-bit V, ships naïve — no outlier-mask, see `docs/KV_QUANT.md`), `k8vturbo2tcq` (Viterbi trellis 2-bit V; same 4-state trellis over the 2-bit Lloyd-Max codebook; decoder bit-identical to plain turbo2; CPU encode on hot path; MSL kernel parity-tested but parked as future-reference hook; distinct SSD layout tag prevents cross-codec hydrate; outlier-mask deferred; `--ctv` alias: `turbo2_tcq`), `iso3` (IsoQuant quaternion SO(4) 3-bit V; requires `head_dim % 4 == 0`; CPU-only V dequant), `iso4` (IsoQuant quaternion SO(4) 4-bit V; requires `head_dim % 4 == 0`; CPU-only — no MSL kernel; pairs only with K=`q8_g128`; alias: `--ctv iso4`), `rotor3` (Cl(3,0) Clifford rotor sandwich + 3-bit V; static per-(layer, head) rotor table loaded once + per-token codes/scales/norm; pairs with K=`q8_g128`; no head_dim divisibility constraint — `head_dim % 3` is tail-padded; CPU-only; alias `--ctv rotor3` or `--ctv rotor_v_3`), `rotor4` (Cl(3,0) Clifford rotor sandwich + 4-bit V; same structure as `rotor3` with 16-centroid Lloyd-Max codebook and dense 8-vals/u32 pack; ~10.7 bpe effective; CPU-only; alias `--ctv rotor4` or `--ctv rotor_v_4`), `iso3_sym` (symmetric IsoQuant 3-bit on **both** K and V; quaternion SO(4) rotation + 3-bit Lloyd-Max codebook applied identically per axis; CPU-only; rejected on Qwen3.5/3.6 MoE with `QwenMoeIsoKRejected`), `iso4_sym` (symmetric IsoQuant 4-bit K+V; same Qwen MoE arch guard as `iso3_sym`), `k_iso3` (K-only IsoQuant 3-bit; V stays bf16; pairs `--ctk iso_k_3 --ctv bf16`; same Qwen MoE guard), `k_iso4` (K-only IsoQuant 4-bit; V bf16; same arch guard), `rotor3_sym` (symmetric Clifford rotor3 K+V; K side carries optional 1-bit QJL residual sideband when `--rotor-qjl on` (default); rejected on Qwen MoE with `QwenMoeRotorKRejected`), `rotor4_sym` (symmetric Clifford rotor4 K+V; same QJL toggle + Qwen MoE guard), `k_rotor3` (K-only rotor3; V stays bf16; optional QJL; same Qwen MoE guard), `k_rotor4` (K-only rotor4; V bf16; optional QJL; same arch guard), `rotor_k_3_asym_v<vb>_g<vg>` (payload-bearing asymmetric: rotor3 K + TurboQuant V at `(v_bits, v_group_size)`; accepted V tuples: `(4,128)`, `(4,64)`, `(4,32)`, `(3,64)`, `(2,64)` (v_group_size is layout-tag-only — TurboQuant V uses GROUP_SIZE=32 regardless); compose via `--ctk k_rotor3 --ctv q4_g64` (or other affine `q*_g*`); same Qwen MoE guard), `rotor_k_4_asym_v<vb>_g<vg>` (rotor4 K + TurboQuant V; same compose / guard rules), `tsym3` (TurboSym3: WHT-3 K + WHT-3 V symmetric; storage `TurboSym3 { k: QuantKTurbo3, v: QuantVBits3 }`; 3 bits on both K and V sides using the WHT + Lloyd-Max codebook; CPU-only; rejected for Qwen3.5/3.6 MoE with `QwenMoeTurboSymKRejected`; opt-in only, never an auto baseline), `rot_k_tq4v`, `mixed_k<kb>g<kg>_v<vb>g<vg>`. Mutually exclusive with `--cache-type-k`/`--cache-type-v`. |
| `--kv-preset` | string | — | Named KV-cache preset. Resolves to a `KvQuant` by name. Mutually exclusive with `--kv-quant`, `--cache-type-k`, `--cache-type-v`, `--kv-bits`. See `docs/KV_QUANT.md §Presets`. Available: `fp16`, `q8`, `speed`, `quality`, `planar`, `planar3`. |
| `--cache-type-k` / `--ctk` | string | — | Per-side codec for the K (key) tensor. See `rmlx info --list-cache-types` for the full codec table. Mutually exclusive with `--kv-quant`. |
| `--cache-type-v` / `--ctv` | string | — | Per-side codec for the V (value) tensor. Mutually exclusive with `--kv-quant`. |
| `--kv-bits` | float | — | Bit-width alias (integer or fractional, e.g. `4`, `3.5`). Mutually exclusive with `--kv-quant` and `--cache-type-*`. See KV-bits mapping below. |
| `--kv-group-size` | usize | 64 | Group size for `--kv-bits`. Requires `--kv-bits`. |
| `--max-ctx` | u32 | (from model) | **Virtual ceiling** on context length, in tokens — NOT an eager allocation. The KV ring starts small (`KV_MAX_SEQ_DEFAULT = 4096`) and grows lazily up to this ceiling as the prompt fills; prompts over the ceiling are rejected. Short requests on a large-`--max-ctx` server thus decode at full speed (no long-context working-set tax — see `docs/KV_CACHE.md` §4.6). Derives from `max_position_embeddings` capped at 4096 when unset. Must be ≥ 256 when set. |
| `--idle-timeout-secs` | string | `15m` | Idle time before the model is unloaded. Accepts an integer count of seconds (`30`, `900`) OR a Go-style duration (`30s`, `15m`, `2h`, `24h`). Negative (`-1`) pins the model forever; `0` unloads after each response. Per-request override on **native** routes only (`POST /v1/models/{id}/load` body field `keep_alive`); OpenAI/Anthropic compat routes do not parse the field but still reset the timer on use. **Interaction with the single-MLX claim file:** the timer never bypasses the claim — when TTL fires it unloads the slot in-process; the cross-process claim file (`/tmp/rmlx.<port>.claim`) remains held for the lifetime of the `rmlx serve` process. |
| `--prompt-cache-slots` | usize | 4 | Number of prompt-cache slots for multi-slot prefix matching. Set to `1` for legacy single-slot exact-match behaviour. |
| `--draft-model` | path | — | Path to a draft model for speculative decoding. Requires `--draft-kind`. |
| `--draft-kind` | `mtp \| dflash \| eagle3` | — | Drafter architecture. Requires `--draft-model`. Env: `MLX_VLM_DRAFT_KIND`. |
| `--draft-block-size` | usize | 4 | Tokens proposed per speculative round. Env: `MLX_VLM_DRAFT_BLOCK_SIZE`. |
| `--max-tokens-cap` | u32 | `u32::MAX` | Per-request `max_tokens` ceiling. Requests exceeding this receive HTTP 400. |
| `--max-timeout-secs` | u64 | 600 | Server-startup wall-clock timeout cap per request in seconds. `0` disables. |
| `--turbo-flash` | `on` \| `off` \| `auto` | `auto` | TurboFlash MSL attention kernel. `auto` (default): hardware-gated. Apple ≤9 (M1/M2/M3/M4) → ON (validated: 32k NIAH × 3 models on M3, commit fcb2e894ccc4). Apple10 (M5+) → ON (re-validated the historical `head_dim = 256` hazard on M5 Max; no crash + cosine ≥ 0.997 vs bf16; the 0.997 residual is the V turbo-4 codec floor, confirmed via CPU baseline). Apple11+ → ON with a **`tracing::warn!`** noting the family has not been hw-validated yet (operator-visible at standard log level; squelch via `RUST_LOG` if unwanted). Unknown / non-Apple hosts → OFF (sysctl probe failed). `on` forces `RMLX_TURBO_FLASH=1`. `off` hard-overrides by removing `RMLX_TURBO_FLASH` from the environment so a stale `=1` cannot latch the OnceLock. **Risk (Apple11+ optimistic-ON policy):** untested future Apple gens default ON, not OFF. Justification: (a) Apple7–10 cleared validation; (b) the historical hazard mode is SIGSEGV (`EXC_BAD_ACCESS`), not silent corruption, so a regression on a new gen surfaces immediately via crash log; (c) operators on untested hw can hard-override via `--turbo-flash off`; (d) defaulting OFF would create perpetual silent fallback the moment Apple ships M6. If a future Apple gen exhibits the hazard, flip the per-gen arm accordingly. |
| `--turbo-flash-lock` | bool flag | off | Enable TurboFlash lock variant. Has no effect unless `--turbo-flash` or `RMLX_TURBO_FLASH=1` is also active. |
| `--planar-flash-decode` | `on` \| `off` \| `auto` | `auto` | PlanarK single-pass flash-decode MSL kernel. `auto` (default): resolves OFF on every host — validation confirmed the kernel is bit-for-bit identical to the fused chain (`update_and_sdpa_planar_k_fused` dispatch_delta>0 with output matching OFF byte-for-byte on Bonsai) but did not deliver a measurable decode-TPS gain (-0.19% mean at 4k canary; well below the ≥10% Auto-flip gate). A pre-existing PlanarK-on-Bonsai long-prompt chunked-prefill bug (`docs/KV_QUANT.md` §"Correctness gap") also prevented the NIAH correctness anchor from passing on the only reachable arch. `on` forces `RMLX_PLANAR_FLASH_DECODE=1` (opt-in ablation). `off` **hard-overrides** — removes any pre-existing `RMLX_PLANAR_FLASH_DECODE` from the env so a stale `=1` cannot latch the OnceLock. |
| `--require-smoke-probe` | bool flag | off | Run 8-token smoke probe on every model load; reject `BrokenPunctLoop` / `BrokenNan` results with HTTP 503. |
| `--max-loaded-models` | usize | 1 | Maximum models held resident in GPU memory. LRU eviction when exceeded. |
| `--max-queue-depth` | usize | 64 | FIFO admission queue depth. Requests beyond this limit receive HTTP 429. `0` = unlimited. |
| `--adaptive-admission` | bool flag | off | Enable the in-process adaptive admission controller. When set, the controller adjusts `max_queue_depth` dynamically based on SLA telemetry and rejects requests with HTTP 503 + `Retry-After: 5` when the end-to-end step estimate exceeds `2 × step-target-ms`. When absent, the static `--max-queue-depth` is used unchanged. |
| `--step-target-ms` | u64 | 500 | End-to-end step SLA target in milliseconds for the adaptive controller. Anticipatory 503 fires when `est_step > 2 × this`. Requires `--adaptive-admission`. `--ttft-target-ms` is accepted as a hidden alias for backward compatibility. |
| `--itl-target-ms` | u64 | 50 | ITL SLA target in milliseconds for the adaptive controller. Queue depth is lowered after `HOLD_TICKS` (3) consecutive ticks above target and raised when below `0.80 × target`. Requires `--adaptive-admission`. |
| `--adaptive-prefill-chunk` | bool flag | off | Enable adaptive prefill-chunk sizing. Requires `--adaptive-admission`. Adjusts the process-wide prefill chunk within `[32, 2048]` tokens using the same deadband shape: raises when `est_itl < 0.80 × --itl-target-ms`, lowers after 3 consecutive overload ticks. OFF by default — defaults are locked from p0b-ttft bench. |
| `--default-temperature` | f32 | — | Server-wide default temperature when a request omits the field. Must be in `[0.0, 2.0]`. |
| `--enable-thinking` | bool | — | Server-wide default for Qwen3-family thinking mode. Per-request `enable_thinking` overrides this. |
| `--whisper-model-path` | path | — | Path to a Whisper snapshot directory (contains `config.json` + `weights.npz`). Required for `POST /v1/audio/transcriptions` and `/v1/audio/translations`. Env: `RMLX_WHISPER_MODEL_PATH`. |
| `--whisper-tokenizer-path` | path | — | Path to a directory containing `tokenizer.json` (e.g. `openai/whisper-large-v3`). Required for audio endpoints; the mlx-community Whisper snapshot does not ship tokenizer files. Env: `RMLX_WHISPER_TOKENIZER_PATH`. |
| `--tts-model-path` | path | — | Path to a Qwen3-TTS model snapshot directory. Required for `POST /v1/audio/speech`. Codec decoder not yet implemented; returns 501 until then. Env: `RMLX_TTS_MODEL_PATH`. |
| `--tts-tokenizer-path` | path | — | Path to the Qwen3-TTS speech tokenizer snapshot directory. Used alongside `--tts-model-path`. Env: `RMLX_TTS_TOKENIZER_PATH`. |
| `--mm-cache-bytes` | usize | `536870912` (512 MiB) | Byte budget for the multimodal encoder-output cache. Vision-tower (and Whisper-encoder) outputs are cached keyed on the post-preprocess pixel/PCM content hash so repeated calls with identical inputs skip the encoder. `0` disables the cache. Env: `RMLX_MM_CACHE_BYTES`. |
| `--kv-ssd-cache-gb` | f64 | 0.0 | SSD prompt-cache tier budget in GiB per namespace. `0` = tier off (RAM-only). Blocks land in `<RMLX_HOME>/cache/kv/<namespace>/`. |
| `--project` | string | (model id) | SSD prompt-cache namespace name. Requires `--kv-ssd-cache-gb > 0`. |
| `--kv-ssd-global-gb` | f64 | 0.0 | Global SSD pool ceiling across all namespaces in GiB. `0` = no global cap. Effective per-namespace ceiling is `min(--kv-ssd-cache-gb, --kv-ssd-global-gb)` when global > 0. |
| `--prompt-cache-ram-gb` | f64 | 2.0 | RAM cap for the in-process prompt cache in GiB. |
| `--paged-kv` | bool flag | off | Route K8V4/K8V8/Planar caches through the block-table paged storage path. Incompatible with `bf16`/`none` and `rot_k*` cache types. |
| `--paged-kv-page-tokens` | i32 | 32 | Tokens per paged-KV block. Requires `--paged-kv`. |
| `--rotor-qjl` | `on \| off` | `on` | Toggle the K-side 1-bit QJL residual for the `rotor3_sym` / `rotor4_sym` / `k_rotor3` / `k_rotor4` / `rotor_k_{3,4}_asym_v*_g*` codecs. Default `on` (spec mandate). `--rotor-qjl off` disables the QJL sideband for ablation / bench. Env fallback: `RMLX_ROTOR_QJL=0`. No effect on non-rotor-K-side quant variants. |
| `--planar-fused-qk` | `on \| off` | `on` | Route pre-softmax QK over PlanarQuant-packed K (`KvStorage::PlanarK`) through the `planar_fused_qk` MSL kernel instead of dequant+SDPA. Decode-step only (prefill chunks fall through to the legacy path). No effect on any non-PlanarK cache. **No env fallback — CLI-only**; tests do not need an env lock. See `docs/KV_QUANT.md` §"Fused-QK kernels". |
| `--prefix-index` | `linear \| radix` | `linear` | Longest-prefix index strategy for the prompt cache. `linear` is O(slots × n\_blocks); `radix` is O(n\_blocks). |

> **Per-request override (issue #26).** `--kv-quant` and `--max-ctx` set the
> **launch defaults**. On a running server, the OpenAI route accepts per-request
> `kv_quant` / `max_ctx` fields that override them for one request without
> reloading weights — one resident model can serve multiple KV codecs / context
> ceilings. Requests that omit the fields use these launch defaults. See
> `docs/SERVER.md` § "Per-request KV-config hot-swap". `--kv-ssd-cache-gb` stays
> launch-fixed (SSD live reconfig deferred — `docs/SSD_TIER.md`).

**KV-bits mapping** (`--kv-bits` + `--kv-group-size`):

| `--kv-bits` | Result |
|---|---|
| `8` with group `128` | K8V8 (rMLX MSL q8\_0, both sides) |
| `4` with group `64` | Mixed K8/g64 + V4/g64 (mlx-lm default) |
| `3.5` with group `64` | Mixed K3/g64 + V4/g64 (TurboQuant floor/ceil) |
| `4.5` with group `64` | Mixed K4/g64 + V5/g64 |

Integer values always default K to 8-bit/g64. Fractional values map K to
`floor(bits)` and V to `ceil(bits)`.

#### Keep-alive & auto-unload TTL

Each loaded model carries its own keep-alive timer. After every successful
request, the timer is reset; on expiry the model is unloaded from GPU
memory. The timer never tears down a model that has an active decode
(streaming, blocking, or audio/embeddings) — instead it defers until the
decode-lease guard drops, then re-arms for another full TTL period.

Precedence (highest to lowest):

1. Per-request `keep_alive` body field on **native** routes only —
   `POST /v1/models/{id}/load` accepts `{"keep_alive": <int>}`.
2. `--idle-timeout-secs <DURATION>` CLI flag.
3. Default: `15m` (900 s).

Accepted duration syntax (CLI flag, env var, and request field):

| Value | Meaning |
|---|---|
| `-1`, `-1s`, `-30m` | Pin — never unload via TTL. |
| `0`, `0s` | Unload immediately after the next response finishes. |
| `30`, `30s` | Idle TTL = 30 s. |
| `15m` | Idle TTL = 15 min (the default). |
| `2h`, `24h` | Idle TTL in hours. |

Compatibility note — only `/v1/models/{id}/load` honours the per-request
`keep_alive` field. The OpenAI-compatible chat completions route
(`/v1/chat/completions`) and the Anthropic `/v1/messages` route
intentionally **ignore** the field (matching the upstream ecosystem;
cf. ollama#11458) but still reset the timer on each request.

`/v1/embeddings` (jina) and `/v1/audio/*` (Whisper STT, Qwen3-TTS) use a
separate process-lifetime cache (`embed_slot`, `audio_model`) that is **not**
subject to the keep-alive TTL today: those slots stay resident for the
lifetime of the `rmlx serve` process. A follow-up may unify them with the
main slot lifecycle.

Interaction with the single-MLX claim file: keep-alive operates entirely
**inside the rmlx serve process**. The cross-process claim file at
`/tmp/rmlx.<port>.claim` is held for the lifetime of the `rmlx serve`
process and is *not* released on TTL unload — only on `rmlx serve` exit.
Use `pkill -f "rmlx serve" && rm -f /tmp/rmlx.<port>.claim` to swap the
process itself.

Tracing events emitted at the four anchor points:

- `keep_alive_armed { model_id, ttl_secs }` — fresh timer on load.
- `keep_alive_reset { model_id, ttl_secs }` — request reset.
- `model_unload_idle { model_id, idle_secs }` — TTL fired.
- `model_unload_evict { model_id, requested, reason="cooperative_evict" }` —
  loading a different model evicted this one (LM-Studio Auto-Evict).

---

### `chat`

Interactive REPL for testing a model locally. Reads from stdin line by line;
type an empty line to send the accumulated buffer.

```bash
rmlx chat --model /path/to/snapshot
rmlx chat --model /path/to/snapshot --device cpu --kv-quant k8v4
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Path to the model snapshot directory. |
| `--device` | `cpu \| gpu` | `gpu` | Inference device. |
| `--kv-quant` | string | `auto` | KV cache quantization preset. |
| `--kv-preset` | string | — | Named KV-cache preset. Mutually exclusive with `--kv-quant`, `--cache-type-k`, `--cache-type-v`, `--kv-bits`. Values: `auto` (hardware-aware selector), `fp16`, `q8`, `speed`, `quality`, `planar`, `k_only_planar`. |
| `--cache-type-k` / `--ctk` | string | — | Per-side K codec. Mutually exclusive with `--kv-quant`. |
| `--cache-type-v` / `--ctv` | string | — | Per-side V codec. Mutually exclusive with `--kv-quant`. |
| `--kv-bits` | float | — | Bit-width alias. Mutually exclusive with `--kv-quant` and `--cache-type-*`. |
| `--kv-group-size` | usize | 64 | Group size for `--kv-bits`. Requires `--kv-bits`. |
| `--max-ctx` | u32 | (from model) | **Virtual ceiling** on context length, in tokens (not an eager allocation): the KV ring grows lazily up to it, prompts over it are rejected. See `docs/KV_CACHE.md` §4.6. Must be ≥ 256 when set. |

---

### `info`

Prints architecture and quantization metadata for a model snapshot without
running inference. Also provides the codec table and optional smoke/forward probes.

```bash
rmlx info --model /path/to/snapshot
rmlx info --list-cache-types
rmlx info --model /path/to/snapshot --probe-smoke
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | — | Path to the model snapshot directory. Required unless `--list-cache-types` is set. |
| `--device` | `cpu \| gpu` | `gpu` | Device for probe passes (`--probe-forward`, `--probe-smoke`). |
| `--probe-forward` | bool flag | off | Run a single-token forward pass and print the top-1 token + max logit. |
| `--probe-smoke` | bool flag | off | Run the 8-token smoke probe and classify the snapshot. Exit 1 on `BrokenPunctLoop` or `BrokenNan`. |
| `--kv-quant` | string | `auto` | KV cache quantization preset. |
| `--kv-preset` | string | — | Named KV-cache preset. Mutually exclusive with `--kv-quant`, `--cache-type-k`, `--cache-type-v`, `--kv-bits`. Values: `auto` (hardware-aware selector), `fp16`, `q8`, `speed`, `quality`, `planar`, `k_only_planar`. |
| `--cache-type-k` / `--ctk` | string | — | Per-side K codec. Mutually exclusive with `--kv-quant`. |
| `--cache-type-v` / `--ctv` | string | — | Per-side V codec. Mutually exclusive with `--kv-quant`. |
| `--kv-bits` | float | — | Bit-width alias. Mutually exclusive with `--kv-quant` and `--cache-type-*`. |
| `--kv-group-size` | usize | 64 | Group size for `--kv-bits`. |
| `--max-ctx` | u32 | (from model) | **Virtual ceiling** on context length, in tokens (not an eager allocation): the KV ring grows lazily up to it, prompts over it are rejected. See `docs/KV_CACHE.md` §4.6. Must be ≥ 256 when set. |
| `--list-cache-types` | bool flag | off | Print the full §D1 KV codec table and exit. No model load. |

---

### `baseline`

Measures load time, time-to-first-token (TTFT), decode tokens/sec, and peak
RSS for a model snapshot. Appends one row to the metrics buffer.

```bash
rmlx baseline --model /path/to/snapshot
rmlx baseline --model /path/to/snapshot --kv-quant k8v4 --max-tokens 64 --record
rmlx baseline --model /path/to/snapshot --prompt-tokens 4096 --label "8k-bench"
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Path to the model snapshot directory. |
| `--prompt` | path | bundled fixture | Path to a plain-text prompt file. Mutually exclusive with `--prompt-tokens`. |
| `--prompt-tokens` | u32 | — | Select a canonical bench prompt from `prompts/longctx_<N/1024>k.json`. Mutually exclusive with `--prompt`. |
| `--device` | `cpu \| gpu` | `gpu` | Inference device. |
| `--max-tokens` / `--gen-tokens` | u32 | 32 | Number of tokens to generate. |
| `--prompt-label` | string | (filename) | Short label for the `prompt` column in the metrics record. |
| `--kv-quant` | string | `auto` | KV cache quantization preset. |
| `--kv-preset` | string | — | Named KV-cache preset. Mutually exclusive with `--kv-quant`, `--cache-type-k`, `--cache-type-v`, `--kv-bits`. Values: `auto` (hardware-aware selector), `fp16`, `q8`, `speed`, `quality`, `planar`, `k_only_planar`. |
| `--cache-type-k` / `--ctk` | string | — | Per-side K codec. Mutually exclusive with `--kv-quant`. |
| `--cache-type-v` / `--ctv` | string | — | Per-side V codec. Mutually exclusive with `--kv-quant`. |
| `--kv-bits` | float | — | Bit-width alias. Mutually exclusive with `--kv-quant` and `--cache-type-*`. |
| `--kv-group-size` | usize | 64 | Group size for `--kv-bits`. |
| `--max-ctx` / `--ctx-max` | u32 | (from model) | KV cache buffer token capacity. Must be ≥ 256 when set. |
| `--label` | string | — | Free-form campaign label stamped into the metrics record's `notes` column. |
| `--record` | bool flag | off | Emit a §8.5 `RunRecord` to the metrics buffer and ingest into `runs.db` in-process. |

---

### `kv-calibrate`

CPU-only weight-norm calibration for TurboQuant KV cache. Walks K/V projection
weight tensors in all safetensors shards, computes per-head L2 norms, selects
top-K high-precision indices per head, and writes `kv_calib.json`. Output is
byte-identical with `multi-turboquant`'s `turboquant_kv.json` v1 schema.

**This subcommand acquires no Metal claim and performs no MLX allocation.** It
is safe to run alongside a running `rmlx serve` instance.

**Requires unquantized model weights** (dtype F32, BF16, or F16). Running on an
already weight-quantized snapshot (e.g. mxfp8 or 2-bit MLX affine) will
produce an error: run calibration on the base float model before quantizing.

```bash
rmlx kv-calibrate /path/to/snapshot
rmlx kv-calibrate /path/to/snapshot --recipe turbo2
rmlx kv-calibrate /path/to/snapshot --recipe turbo3 --out /path/to/kv_calib.json
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `MODEL` | path (positional) | required | Path to the model snapshot directory. |
| `--recipe` | string | `turbo3` | KV quantization recipe: `turbo2`, `turbo2_tcq`, `turbo3`, `turbo3_tcq`, `turbo4`, `head_budget`, `softmax_mass`, or `k_norm_proxy` (see below). |
| `--out` | path | `<MODEL>/kv_calib.json` (or `<MODEL>/head_budgets.json` for the head-budget family) | Output path for the calibration JSON file. |
| `--prompts` | path | softmax_mass: `prompts/calibration_long_context.json`; head_budget/k_norm_proxy: `prompts/calibration_default.json`; resolved by walking up from cwd | (head-budget family only) Calibration prompt JSON. Ignored by weight-norm recipes. |
| `--mass-threshold` | f32 | `0.95` | (head-budget family only) Cumulative softmax-mass coverage target in `[0.50, 1.00]`. Ignored by weight-norm recipes. |
| `--target-mass-budget-floor` | u32 | `16` | (`softmax_mass` recipe) Minimum per-(layer, head) budget. Guards against pathological single-mass distributions producing a 1-slot budget. Ignored by other recipes. |

**Recipe → internal recipe mapping:**

| User recipe | Internal recipe | Outlier ratio |
|---|---|---|
| `turbo2`, `turbo2_tcq` | `turboquant25` | 25% of `head_dim` |
| `turbo3`, `turbo3_tcq`, `turbo4` | `turboquant35` | 50% of `head_dim` |

The outlier count is `round(head_dim * ratio / 16) * 16` (aligned to group size 16, using
round-half-away-from-zero). For all standard head_dims (64, 128, 256) this matches mtq exactly;
divergence with Python's banker's rounding can only arise for non-standard head_dims at exact
midpoints (e.g. `head_dim=80` with `turboquant35`).

**Output schema** (`kv_calib.json`):

```json
{
  "version": 1,
  "recipe": "turboquant35",
  "head_size": 128,
  "model_name": "my-model",
  "transform_version": "structured_hadamard_v1",
  "codebook_version": "lloyd_beta_v1",
  "layers": {
    "model.layers.0.self_attn": {
      "key_high_precision_indices": [[3, 7, 11, ...], [1, 5, 9, ...]],
      "value_high_precision_indices": [[2, 6, 10, ...], [0, 4, 8, ...]]
    }
  },
  "calibration": {
    "method": "weight_norm",
    "objective": "l2_norm",
    "num_prompts": 0,
    ...
  }
}
```

#### `--recipe softmax_mass`

True softmax-mass head-budget calibration. Supersedes `head_budget`
(K-norm² proxy). Writes schema v2 `head_budgets.json`.

**Loads the model on GPU** to derive per-(layer, head) k-budgets for the
two-phase sparse-attention dispatcher. Acquires the single-MLX claim
(hard rule 8); refuses to start if another `rmlx serve` holds it.
Architecture gate: ships `Qwen3ForCausalLM` only (Bonsai is the smoke
target). Adding Gemma4 / Qwen3.5MoE / Qwen3VL is follow-up work.

```bash
rmlx kv-calibrate /path/to/qwen3-snapshot --recipe softmax_mass
rmlx kv-calibrate /path/to/qwen3-snapshot --recipe softmax_mass \
    --prompts ./prompts/calibration_long_context.json \
    --mass-threshold 0.95 \
    --target-mass-budget-floor 16 \
    --out ./head_budgets.json
```

Each prompt is tokenised, prefilled into a fresh bf16 KV cache
(`KvQuant::None`), and a `CalibrationSink` plumbed through
`Qwen3Text::forward_seq_with_cache_calibrated` captures the last-position
post-RoPE Q vector and the full accumulated K tensor per layer. The
sink computes real Q@K^T → softmax → cumulative-mass top-K per
kv-head (Q-head group-mean-folded for GQA), and max-aggregates across
prompts. The kv-head budget table is then GQA-expanded to fill the
Q-head rows of the v2 schema.

Output: `head_budgets.json` v2 with `recipe="softmax_mass"`,
`target_mass`, `target_mass_budget_floor`, and `prompts_provenance`
fields populated per
[`rmlx_loader::head_budgets::HeadBudgets`](../crates/rmlx-loader/src/head_budgets.rs).

The production code path is unchanged — `forward_seq_with_cache` (no
sink) takes the steady-state branch with zero per-call overhead. The
sink only fires when the calibration runtime installs `Some(sink)`.

#### `--recipe head_budget` / `--recipe k_norm_proxy` (legacy)

K-norm² proxy calibration. **Superseded by `softmax_mass`** but kept for
back-compat. Writes schema v1 `head_budgets.json` (no `recipe` field on
disk; the schema's `method = "softmax_mass"` label names the concept,
the implementation is the K-norm² stand-in). `k_norm_proxy` is an explicit
alias for the same recipe.

```bash
rmlx kv-calibrate /path/to/qwen3-snapshot --recipe head_budget
rmlx kv-calibrate /path/to/qwen3-snapshot --recipe k_norm_proxy \
    --prompts ./prompts/custom_calibration.json \
    --mass-threshold 0.99 \
    --out ./head_budgets.json
```

Each prompt is tokenised, prefilled into a fresh bf16 KV cache
(`KvQuant::None`), and the per-layer K accumulator is walked on host to
compute per-(kv-head, key-position) K-norm² as a stand-in for softmax mass.
For each causal query position, the smallest top-K covering
`--mass-threshold` of the visible mass is recorded; the per-(layer, head)
budget is the running maximum across all (prompt, q-pos) observations.
GQA-shared KV heads expand to fill the Q-head rows of the output.

Calibration prompt JSON shape:

```json
{
  "version": 1,
  "description": "free-form note",
  "prompts": [
    "first prompt body…",
    "second prompt body…"
  ]
}
```

The prompts file is SHA-256-hashed and the hex digest is recorded in
`calibration.prompt_set_sha256` for provenance. Each prompt is capped at
768 tokens after tokenisation (`HEAD_BUDGET_MAX_TOKENS_PER_PROMPT`); see
`crates/rmlx-cli/src/commands/kv_calibrate.rs`.

---

### `--sparse-attn` (global flag)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--sparse-attn` | enum `auto\|on\|off` | `auto` | Two-phase sparse-attention dispatcher gate. `auto` resolves OFF on every host (warm-TTFT dormant by design). `on` force-sets `RMLX_SPARSE_ATTN=1` before first inference but does NOT cause the kernels to fire on the normal generate flow — that contract is structural (the bf16-K seed shortcut absorbs the decode window). `off` is a hard override: removes `RMLX_SPARSE_ATTN` from the process env so a stale shell-set `=1` cannot latch the OnceLock to `true`. |
| `RMLX_SPARSE_ATTN` | env var | unset | Read once at first `sparse_attn_enabled()` call. `=1` enables the dispatch gate; absent or any other value disables. `--sparse-attn` resolves the env var before any inference runs. |

The two-phase sparse-attention kernels operate over PlanarQuant-K packed
buffers and stay dormant on warm-TTFT decode windows (every quantised codec
routes through the bf16-K seed materialised by `exit_prefill`). They remain
reachable for seedless workloads (synthetic caches, PPL eval, future
prompt-cache hits) via the public production entry point
`rmlx_models::kv_cache::attention_dispatch::sparse_attn_dispatch`.
The dispatch counter aggregator
`rmlx_kv_quant::sparse_attn::sparse_attn_total_dispatch_count` proves
fire-vs-dormant in tests under
`crates/rmlx-models/tests/sparse_attn_dispatch.rs`.

---

### `healthcheck`

Shell-able readiness probe. Emits one JSON line per check, plus a final
aggregate line. The default path never loads the MLX runtime; `--full` loads
it for the smoke probe.

```bash
rmlx healthcheck
rmlx healthcheck --port 8080 --human
rmlx healthcheck --registry /path/to/registry.json --full
```

**Output format** (JSON, default):
```text
{"check":"<name>","status":"green|red|info","detail":"..."}
{"check":"aggregate","status":"green|red","red_checks":["..."]}
```

**Exit codes**: `0` = all green, `1` = any red, `2` = internal error.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--registry` | path | — | Check every registered model's loadability. Mutually exclusive with `--model`. |
| `--model` | path | — | Check a single model snapshot's loadability. Mutually exclusive with `--registry`. |
| `--port` | u16 | — | Also probe a live server on this port (claim file + HTTP `/health` checks). |
| `--db` | path | (env / default) | Path to the metrics SQLite DB. |
| `--min-disk-gb` | u64 | 5 | Minimum free disk space in GiB for `metrics/` and `logs/` directories. |
| `--full` | bool flag | off | Also run the MLX smoke probe per model. Loads MLX — exclusive Metal context. Do not use while another rMLX instance is running. |
| `--human` | bool flag | off | Emit plain `OK`/`FAIL` text instead of JSON lines. |

---

### `metrics`

Manages the metrics SQLite database (`runs.db`). All subcommands operate on
the DB resolved via `--db <path>` or `RMLX_METRICS_DB` (defaults to the
workspace DB at `<RMLX_HOME>/metrics/runs.db`).

```bash
rmlx metrics init
rmlx metrics doctor --fix
rmlx metrics record --file /path/to/run.json
rmlx metrics best --backend rmlx --namespace prism-ml --model Bonsai-8B \
  --weight-quant 2bit --kv-quant k8v8 --metric decode_tps_warm
rmlx metrics export --markdown
rmlx metrics champions --backend rmlx
```

**Global option**: `--db <path>` — override the DB path for any subcommand.

#### `metrics init`

Create the schema and seed `schema_meta`. Refuses if the file already exists.

#### `metrics doctor`

Verify schema version, integrity, FKs, whitelists, units, and directions.

| Flag | Default | Description |
|---|---|---|
| `--fix` | off | Apply detected repairs. |

#### `metrics backup`

WAL-checkpointed copy of the DB.

| Flag | Default | Description |
|---|---|---|
| `--out <path>` | default backups dir | Destination path. |
| `--keep <N>` | — | Keep at most N backups; prune oldest beyond that. |

#### `metrics restore`

Replace the DB from a backup, snapshotting the current DB first.

| Flag | Description |
|---|---|
| `--from <path>` | Source backup file (required). |

#### `metrics record`

Ingest one §8.5 `RunRecord` JSON into the `observations` table. Exactly one
of `--inline`, `--file`, `--stdin`, or `--replay-pending` must be provided.

| Flag | Default | Description |
|---|---|---|
| `--inline <JSON>` | — | Inline JSON object. |
| `--file <path>` | — | Read JSON from file (preferred; follows the §8.4 buffer pattern). |
| `--stdin` | off | Read JSON from stdin. |
| `--dry-run` | off | Validate and show what would be written without committing. |
| `--replay-pending` | off | Walk `metrics/buffer/pending/`, ingest each file, move failures to `failed/`. |

#### `metrics best`

Print the champion row for one `(cell, metric)` tuple.

| Flag | Description |
|---|---|
| `--backend` | Backend name (e.g. `rmlx`). |
| `--namespace` | Model namespace (e.g. `prism-ml`). |
| `--model` | Model name. |
| `--weight-quant` | Weight quantization label. |
| `--kv-quant` | KV quantization label. |
| `--ctx-max` | Server max-ctx at run time (default 8192). |
| `--prompt-id` | Prompt FK into the prompts table. Mutually exclusive with `--prompt-name`. |
| `--prompt-name` | Resolve prompt id by name (latest revision). Mutually exclusive with `--prompt-id`. |
| `--metric` | Metric name (e.g. `decode_tps_warm`). |

#### `metrics rank`

Top-N champions for one metric across all cells.

| Flag | Default | Description |
|---|---|---|
| `--metric` | required | Metric name. |
| `--backend` | — | Filter to one backend. |
| `--limit` | 20 | Number of rows to return. |

#### `metrics compare`

Side-by-side champion comparison for two or more backends (comma-separated).

| Flag | Description |
|---|---|
| `--backends` | Comma-separated backend names, e.g. `rmlx,mlx_lm`. |
| `--metric` | Metric name. |
| `--namespace`, `--model`, `--weight-quant`, `--kv-quant` | Optional filters. |

#### `metrics history`

All observations for one cell, ordered oldest-first.

| Flag | Default | Description |
|---|---|---|
| `--backend`, `--namespace`, `--model`, `--weight-quant`, `--kv-quant` | required | Cell coordinates. |
| `--ctx-max` | 8192 | Max context filter. |
| `--prompt-id` / `--prompt-name` | — | Prompt filter (mutually exclusive). |
| `--metric` | — | Filter to one metric. |
| `--since` | — | ISO-8601 lower bound, e.g. `2026-01-01`. |

#### `metrics timeseries`

Bucketed mean per period for one `(cell, metric)`.

| Flag | Default | Description |
|---|---|---|
| `--bucket` | `day` | Bucket granularity: `day` or `week`. |
| `--since` | — | ISO-8601 lower bound. |
| (cell + metric flags) | — | Same as `history`. |

#### `metrics regress`

Champion-scoped regression gate for one model + metric.

Exit codes: `0` = within tolerance or improvement; `1` = regressed; `125` = no champion found (bisect-safe skip).

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Model name substring. |
| `--metric` | required | Metric name. |
| `--kv` | — | KV quant filter. |
| `--threshold-pct` | 1.0 | Regression threshold percentage. |

#### `metrics deltas`

Regressions and improvements per cell per metric since a git SHA.

| Flag | Default | Description |
|---|---|---|
| `--since-sha` | required | Git SHA to compare against. |
| `--threshold-pct` | 5.0 | Delta threshold percentage. |
| `--exit-code` | `true` | Exit 1 when any regression found. Pass `--exit-code=false` to suppress. |

#### `metrics describe`

Annotate an observation or all observations in a run.

| Flag | Description |
|---|---|
| `--observation-id` | Single observation to annotate. Mutually exclusive with `--run-id`. |
| `--run-id` | Annotate every observation with this run id. Mutually exclusive with `--observation-id`. |
| `--text` | Annotation text (required). |

#### `metrics query`

Run a raw `SELECT` against the DB; outputs TSV. Refuses non-SELECT statements.

```bash
rmlx metrics query "SELECT model, decode_tps_warm FROM bests ORDER BY decode_tps_warm DESC LIMIT 10"
```

#### `metrics open`

Open the DB in an interactive `sqlite3` shell.

| Flag | Default | Description |
|---|---|---|
| `--readonly` | off | Open read-only (`-readonly` flag passed to `sqlite3`). |

#### `metrics export`

Write the `bests` view to a specified format. At least one format flag must be set.

| Flag | Description |
|---|---|
| `--markdown` | Emit `BENCHMARK_CHAMPIONS.md`. |
| `--json` | Emit a compact JSON array. |
| `--csv` | Emit CSV with a header row. |
| `--jsonl` | Emit JSONL (one row per line). |
| `--scope <path>` | Optional `config/scope.toml` for filtering/ordering markdown output. `--markdown` only. |

#### `metrics prompts`

Manage the content-addressed prompt registry.

| Subcommand | Description |
|---|---|
| `list` | List all registered prompts. |
| `get --name <name>` | Print the body of the latest revision for `<name>` to stdout. |
| `add --file <path> [--name <name>] [--notes <text>]` | Register a prompt from a JSON file. |
| `sync` | Sync all `*.json` files under `rMLX/prompts/` into the registry. |

#### `metrics champions`

One row per `(model_namespace, model, weight_quant, kv_quant)` with each
canonical metric as a column.

| Flag | Default | Description |
|---|---|---|
| `--backend` | — | Filter to one backend. |
| `--jsonl` | off | Output JSONL instead of Markdown. |

#### `metrics migrate`

One-shot idempotent ingestion of legacy JSONL/CSV/Markdown into the DB.

| Flag | Description |
|---|---|
| `--rmlx-glob <glob>` | Glob for rMLX JSONL files, e.g. `"metrics/**/*.jsonl"`. |
| `--cbb-csv <path>` | Path to `Cross-Backend-Bench/metrics/summary.csv`. |
| `--records-md <path>` | Path to `BENCHMARK_CHAMPIONS.md` fallback. |
| `--hardware-tag <tag>` | Hardware tag stamped on every migrated observation (default `m5_max_128gb`). |

---

### `eval ppl`

Computes perplexity over a text corpus using sliding-window NLL. Supported
models: Qwen3 family (Bonsai is the smoke target).

Prints one JSON line to stdout:
```text
{"ppl":..,"mean_nll":..,"scored_tokens":..,"windows":..}
```

When `--corpus` is non-empty, also ingests one §8.5 `RunRecord` into
`<RMLX_HOME>/metrics/runs.db` under op `ppl_wikitext2`.

```bash
rmlx eval ppl --model /path/to/snapshot --text-file /path/to/corpus.txt
rmlx eval ppl --model /path/to/snapshot --text-file wiki.txt \
  --corpus wikitext-2 --ctx-window 4096 --stride 2048
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Path to the model snapshot directory. |
| `--text-file` | path | required | Raw UTF-8 corpus file. |
| `--ctx-window` | usize | 4096 | Tokens forwarded per window. |
| `--stride` | usize | 2048 | Stride between consecutive windows. |
| `--corpus` | string | `""` | Corpus identifier. Non-empty triggers metrics ingestion. |
| `--device` | `cpu \| gpu` | `gpu` | Inference device. |
| `--max-tokens` | usize | 0 | Cap on tokens fed to the scorer. `0` = use the whole corpus. |

---

### `profile list`

Lists the names of all defined profiles in `<RMLX_HOME>/profiles.toml`, one
per line. A missing file prints nothing and exits `0`.

```bash
rmlx profile list
```

Profiles are created by editing `<RMLX_HOME>/profiles.toml` directly. The
format is:

```toml
[profile.myrun]
model = "/abs/path/to/snapshot"
port = 9001
host = "0.0.0.0"
kv_quant = "k8v4"
max_ctx = 8192
```

Bindable fields: `model`, `registry`, `port`, `host`, `device`, `kv_quant`,
`max_ctx`, `idle_timeout_secs`, `prompt_cache_slots`, `draft_model`,
`max_tokens_cap`, `max_timeout_secs`, `max_loaded_models`, `max_queue_depth`,
`default_temperature`.

Boolean toggles (`--turbo-flash`, etc.) and per-side `--cache-type-*` codecs
are CLI-only and cannot be set in a profile.

---

## `qwen36_diag` binary

`qwen36_diag` is a low-level diagnostic binary for Qwen3.6 graph verification.
It is not part of the `rmlx` command tree; it is built as a separate binary
from `crates/rmlx-cli/src/bin/qwen36_diag.rs`.

```bash
# Single forward pass — prints argmax token id + max logit
qwen36_diag <model-dir> [cpu|gpu]

# Greedy generation of N tokens from a canonical 20-token prompt
qwen36_diag <model-dir> <device> <N>
```

The canonical prompt is hardcoded (20 tokens). The expected baseline is
`argmax_id=8160` ("Here"), `max_abs_logit=29.75` (from the mlx-lm reference).

---

## Environment variables

### User / operational

These are the variables a typical operator sets. For each variable that has a
matching CLI flag, **the flag wins** — the env var is a convenience for
persistent shell configuration.

| Variable | Flag (if any) | Default | Description |
|---|---|---|---|
| `RMLX_HOME` | — | `<workspace>/.rmlx/` or `$HOME/.rmlx/` | Root directory for all on-disk state (`logs/`, `metrics/`, `cache/`). Resolution order: (1) `$RMLX_HOME` (must be absolute), (2) `<workspace>/.rmlx/` (auto-detected by walking up for `Cargo.lock`), (3) `$HOME/.rmlx/`. |
| `RMLX_LOG_CAP_MB` | `--log-cap-mb` | `100` | Total log directory size cap in megabytes. Oldest `.jsonl` files are deleted at startup until the total is within cap. Flag wins. |
| `RUST_LOG` | `--log` | — | Explicit `tracing` filter directive. When set, overrides `--log`. Example: `RUST_LOG=debug,rmlx=trace`. |
| `RMLX_METRICS_DB` | — | `<RMLX_HOME>/metrics/runs.db` | Override path to `runs.db`. Used by `rmlx metrics` subcommands and any subcommand that opens an `EventRecorder`. |
| `RMLX_WHISPER_MODEL_PATH` | `--whisper-model-path` | — | Path to a Whisper snapshot directory. Required for `/v1/audio/transcriptions` and `/v1/audio/translations`. Flag wins. |
| `RMLX_WHISPER_TOKENIZER_PATH` | `--whisper-tokenizer-path` | — | Path to a directory containing `tokenizer.json` for Whisper. Flag wins. |
| `RMLX_TTS_MODEL_PATH` | `--tts-model-path` | — | Path to a Qwen3-TTS model snapshot directory. Required for `/v1/audio/speech`. Flag wins. |
| `RMLX_TTS_TOKENIZER_PATH` | `--tts-tokenizer-path` | — | Path to the Qwen3-TTS speech tokenizer snapshot directory. Flag wins. |
| `RMLX_MM_CACHE_BYTES` | `--mm-cache-bytes` | `536870912` (512 MiB) | Byte budget for the multimodal encoder-output cache. `0` disables. Flag wins. |
| `RMLX_SESSION_CACHE_MAX_SESSIONS` | `--session-cache-max-sessions` | `8` | Maximum number of prompt-cache sessions held resident. Flag wins. |
| `RMLX_YARN_FACTOR` | `--yarn-factor` | — | For Qwen3-family models that ship without `rope_scaling` in `config.json`, set this to a float `> 1.0` to synthesise a YARN config at model load. Default `beta_fast=32, beta_slow=1` (per the YARN paper). Models that already declare `rope_scaling.rope_type == "yarn"` (Bonsai) ignore this var — config wins. Flag wins. |
| `RMLX_YARN_ORIGINAL_MAX` | `--yarn-original-max` | (model `max_position_embeddings`) | Optional companion to `RMLX_YARN_FACTOR`: the training-time `original_max_position_embeddings`. Flag wins. |
| `RMLX_PROMPTS_DIR` | `--prompts-dir` | `<repo>/prompts/` | Directory containing prompt JSON files used by `rmlx baseline` and bench scripts. Flag wins. |
| `MLX_VLM_DRAFT_KIND` | `--draft-kind` | — | Drafter architecture for speculative decoding. Values: `mtp`, `dflash`, `eagle3`. Flag wins. |
| `MLX_VLM_DRAFT_BLOCK_SIZE` | `--draft-block-size` | `4` | Draft block size (tokens per speculative round). Flag wins. |

### Internal / advanced (not needed for normal use)

These variables are read once via `OnceLock` at first use. They exist as
bridges for the corresponding CLI flags, or as dev / ablation toggles that
have no user-facing flag. **Prefer the matching `--flag` where one exists** —
the env var is only the internal bridge that the flag writes before the first
inference call latches the lock.

| Variable | Flag (if any) | Default | Description |
|---|---|---|---|
| `RMLX_TURBO_FLASH` | `--turbo-flash` | (hardware-gated) | Set to `1` to enable the TurboFlash MSL attention kernel. `--turbo-flash on` force-sets this var; `--turbo-flash off` hard-removes it so a stale `=1` cannot latch the OnceLock. Prefer `--turbo-flash`. |
| `RMLX_TURBO_FLASH_LOCK` | `--turbo-flash-lock` | unset | Set to `1` to enable the TurboFlash lock variant. The CLI flag takes precedence. Prefer `--turbo-flash-lock`. |
| `RMLX_TURBO_FLASH_MIN` | — | `1` | Minimum batch-sequence count below which TurboFlash is bypassed regardless of the `RMLX_TURBO_FLASH` gate. Dev tuning only. |
| `RMLX_PLANAR_FLASH_DECODE` | `--planar-flash-decode` | unset (resolves OFF) | Set to `1` to enable the `planar_flash_decode` MSL kernel. `--planar-flash-decode on` force-sets; `--planar-flash-decode off` hard-removes. Only applies to `KvStorage::PlanarK` caches. Prefer `--planar-flash-decode`. |
| `RMLX_FUSED_QK` | `--fused-qk` | unset (resolves OFF) | Set to `1` to enable the fused-QK MSL kernel for PlanarK caches. `--fused-qk on` force-sets; `--fused-qk off` hard-removes. Prefer `--fused-qk`. |
| `RMLX_FUSED_QK_MIN` | — | `1` | Minimum sequence length threshold for fused-QK dispatch. Dev tuning only. |
| `RMLX_SPARSE_ATTN` | `--sparse-attn` | unset (resolves OFF) | Set to `1` to enable the two-phase sparse-attention dispatcher. `--sparse-attn on` force-sets; `--sparse-attn off` hard-removes. Prefer `--sparse-attn`. |
| `RMLX_ROTOR_QJL` | `--rotor-qjl` | unset (default ON for rotor codecs) | Set to `0` to disable the K-side 1-bit QJL residual for rotor-K codecs. Prefer `--rotor-qjl off`. |
| `RMLX_ROT_K_FUSED` | — | unset (resolves OFF) | Set to `1` to route rot_k decode steps through the fused MSL path. No CLI flag — ablation / bench only. |
| `RMLX_EAGLE3_NO_FCS` | — | unset | Set to any value to disable the FCS (final correction step) in Eagle3 speculative decoding. Ablation only. |
| `RMLX_PREFILL_CHUNK` | — | (per-arch default) | Override the global prefill chunk size (tokens per forward pass). Also accepts per-arch form `RMLX_PREFILL_CHUNK_<ARCH>` (e.g. `RMLX_PREFILL_CHUNK_QWEN3_5_MOE=256`). Per-arch override takes precedence over the global. Dev tuning. |
| `RMLX_KV_MAX_SEQ_HARD_CAP` | — | unset (no cap) | Opt-in hard cap on KV sequence length. When set, the KV cache rejects any extension beyond this token count. `--max-ctx` is the normal gate; this env is a last-resort safety guard. |
| `RMLX_HARDWARE_TAG` | — | `m5_max_128gb` | Hardware tag embedded in `rmlx baseline` and `rmlx eval ppl` result rows. Set to match your machine (e.g. `m4_max_64gb`) when recording bench results for cross-machine comparison. |
| `RMLX_REPO_ROOT` | — | (auto-detected) | Root directory of the rMLX workspace, used by `rmlx metrics export` when resolving the `BENCHMARK_CHAMPIONS.md` output path. Typically set automatically; override when running from a non-standard working directory. |

---

## Claim file enforcement

Apple Silicon exposes a single Metal GPU context per process. rMLX enforces
a single-process invariant via a POSIX lock file at `/tmp/rmlx.<port>.claim`.

- On `rmlx serve --port <N>`: the claim file for port `N` is written and held
  for the server lifetime. Shutdown releases the file. The server installs a
  SIGINT/SIGTERM graceful-shutdown handler, so `Ctrl-C`, `kill`, and `pkill`
  trigger normal teardown and remove the claim proactively.
- On `rmlx chat`, `rmlx baseline`, `rmlx eval ppl`, and
  `rmlx info --probe-{forward,smoke}`: the sentinel port `0xCAFE` (51966) is
  used as the claim port, indicating a CLI-side (non-HTTP) GPU holder.
- `rmlx healthcheck --port <N>` checks for the claim file without holding it.

### Stale-claim auto-reclaim

A claim left behind by a process that died without running its cleanup —
SIGKILL, a crash, or power loss, none of which a signal handler can intercept —
is reclaimed automatically on the next acquisition. When the claim file already
exists, rMLX reads the holder PID and probes it with `kill(pid, 0)`:

- Holder PID **dead** → the claim is reclaimed (a `warn` is logged) and startup
  proceeds. No manual `rm` is needed.
- Holder PID **alive** → the claim is respected; the new process logs an error
  and exits with code `11`. A live claim is **never** stolen — that would put
  two MLX processes on the single Metal context.

If another rMLX process holds the claim (live holder), the conflicting PID is
included in the error message. To stop it and recover manually:

```bash
pkill -f 'rmlx serve' && rm -f /tmp/rmlx.<port>.claim
```

(The trailing `rm` is only needed if you kill the holder with `SIGKILL`/`-9`;
a plain `kill`/`pkill` lets graceful shutdown remove the file, and the next
serve reclaims any stale file regardless.)

CPU-mode runs (`--device cpu`) skip claim acquisition entirely.

---

## Examples

### Serve a model

```bash
# Serve on default port 8080
rmlx serve --model /path/to/mlx-community__Qwen3.6-35B-A3B-8bit

# Serve with K8V4 cache, custom port, LRU multi-model pool
rmlx serve \
  --model /path/to/snapshot \
  --kv-quant k8v4 \
  --port 9000 \
  --max-loaded-models 3

# Serve from a registry with speculative decoding
rmlx serve \
  --registry ./registry.json \
  --draft-model /path/to/draft-snapshot \
  --draft-kind eagle3 \
  --draft-block-size 4
```

### Inspect a snapshot

```bash
# Print arch + quant metadata
rmlx info --model /path/to/snapshot

# List all supported KV cache codecs
rmlx info --list-cache-types

# Offline smoke probe — exits 1 on BrokenPunctLoop / BrokenNan
rmlx info --model /path/to/snapshot --probe-smoke
```

### Run perplexity evaluation

```bash
rmlx eval ppl \
  --model /path/to/Bonsai-8B \
  --text-file /path/to/wiki.test.txt \
  --corpus wikitext-2 \
  --ctx-window 4096 \
  --stride 2048
```

### Record a performance baseline

```bash
# Quick smoke baseline, 32 tokens
rmlx baseline --model /path/to/snapshot --record

# Full 4k-prompt bench run, labelled for a campaign
rmlx baseline \
  --model /path/to/snapshot \
  --prompt-tokens 4096 \
  --max-tokens 128 \
  --kv-quant k8v8 \
  --label "phase-3-gate" \
  --record
```

### Metrics database operations

```bash
# Initialize schema
rmlx metrics init

# Ingest a pending buffer file
rmlx metrics record --file .rmlx/metrics/buffer/pending/run.json

# Export champions to Markdown
rmlx metrics export --markdown

# Regression gate for Bonsai decode TPS (threshold 1%)
rmlx metrics regress \
  --model bonsai \
  --metric decode_tps_warm \
  --kv k8v8 \
  --threshold-pct 1.0

# Compare two backends on decode TPS
rmlx metrics compare \
  --backends rmlx,mlx_lm \
  --metric decode_tps_warm
```

### Healthcheck

```bash
# Default JSON probe — no MLX load
rmlx healthcheck

# Probe a running server on port 8080, plain-text output
rmlx healthcheck --port 8080 --human

# Full probe including smoke test (loads MLX — ensure no other instance is running)
rmlx healthcheck --model /path/to/snapshot --full
```

---

## See also

- `docs/METRICS_DB.md` — full metrics database schema, §8.2 query API contract, §8.5 ingest JSON shape, and operating rules.
- `docs/PROJECTS_CONFIG.md` — `projects.toml` format for per-project SSD prompt-cache caps.
- `docs/PROFILING.md` — `dhat-heap` feature build, samply flamegraph workflow, `release-debug` profile.
- `docs/TESTING.md` — integration test setup, `make model-check`, `make model-check-full`, golden-token fixtures.
- `docs/KV_CACHE.md` — full §D1 codec table, KV quantization families (TurboQuant, PlanarQuant, RotK), and combination rules.
