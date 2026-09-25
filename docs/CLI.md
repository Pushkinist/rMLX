# rMLX CLI Reference

## Overview

`rmlx` is the one binary that `cargo build --release` produces. The flag
definitions live in `crates/rmlx-cli/src/main.rs` (clap). `rmlx <subcommand>
--help` prints the long help for every flag.

```text
rmlx [global flags] <subcommand> [flags]
```

| Subcommand | Purpose |
|---|---|
| `serve` | OpenAI- and Anthropic-compatible HTTP inference server |
| `chat` | Resolves the KV flags for a snapshot, takes the claim and exits; no REPL |
| `transcribe` | Speech-to-text for one audio file |
| `info` | Architecture and quant metadata for a snapshot; optional probes |
| `baseline` | One generation; load time, TTFT, decode TPS, peak RSS; optional `runs.db` row |
| `bench` | Repeated generations of one cell; median and range; writes nothing |
| `kv-calibrate` | Writes `kv_calib.json` or `head_budgets.json` for a snapshot |
| `healthcheck` | Readiness probe; JSON lines or plain text |
| `metrics` | Metrics database: schema, ingest, queries, export |
| `eval ppl` | Perplexity over a text corpus |
| `profile list` | Names of the `serve` profiles in `profiles.toml` |

`qwen36_diag` is a separate diagnostic binary; see its section below.

---

## Global flags

Every global flag works before or after the subcommand name, so
`rmlx --turbo-flash off serve …` and `rmlx serve --turbo-flash off …` agree.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--log` | `info \| debug \| verbose` | `info` | Log preset. `debug` adds per-step phase events; `verbose` adds per-token, per-FFI and per-layer trace events. `RUST_LOG`, when set, overrides it. |
| `--log-cap-mb` | u64 | `100` | Size cap for `<RMLX_HOME>/logs/` in MB. At startup the oldest `.jsonl` files are deleted until the total fits. `0` turns rotation off. Env: `RMLX_LOG_CAP_MB`. |
| `--metrics` | `off \| events \| full` | `full` | `off` writes nothing and never opens or creates `runs.db`. `events` keeps the event stream and records no observations. Reads work in every mode. See [`METRICS_DB.md`](METRICS_DB.md) §10.1.1. |
| `--turbo-flash` | `on \| off \| auto` | `auto` | TurboFlash attention kernel for `k8v4`. `auto` honours `RMLX_TURBO_FLASH=1` and is off otherwise; `off` ignores the variable. See [`KV_CODECS.md`](KV_CODECS.md) § "TurboFlash is off by default". |
| `--turbo-flash-lock` | bool flag | off | TurboFlash lock variant: skips bf16 K/V maintenance once the flash buffers are seeded. No effect unless TurboFlash is on. When the flag is absent, `RMLX_TURBO_FLASH_LOCK=1` turns it on; there is no `off` arm. |
| `--fused-qk` | `on \| off \| auto` | `auto` | Fused-QK kernels over a head-major K shadow for the q8, TurboSym and rotor-asym K codecs. `auto` honours `RMLX_FUSED_QK=1` and is off otherwise. See [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) § "Fused-QK head-major K storage". |
| `--planar-flash-decode` | `on \| off \| auto` | `auto` | `planar_flash_decode` kernel for `planar_k` caches. `auto` honours `RMLX_PLANAR_FLASH_DECODE=1` and is off otherwise. A cache that went through prefill holds a bf16 K seed and never reaches the kernel. See [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) § "`planar_flash_decode`". |
| `--planar-fused-qk` | `on \| off` | `on` | `planar_fused_qk` kernel for `planar_k` decode steps with no bf16 K seed. `off` uses dequant + SDPA. No environment variable. See [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) § "Fused-QK kernels". |
| `--rot-k-fused` | `on \| off \| auto` | `auto` | Fused FWHT + affine-quantize kernel for `rot_k_v<bits>g<group>`. `auto` honours `RMLX_ROT_K_FUSED=1` and is off otherwise. Every other codec ignores it. |
| `--rotor-qjl` | `on \| off` | `off` | 1-bit QJL residual on the K side of `rotor3_sym`, `rotor4_sym`, `k_rotor3`, `k_rotor4` and `rotor_k_{3,4}_asym_*`. `on` has no Metal kernel and runs the rotor K encode and decode on the CPU. The binary always installs this value, so `RMLX_ROTOR_QJL` has no effect on it. |
| `--sparse-attn` | `on \| off \| auto` | `auto` | Sets `DispatchPolicy::sparse_attn`. It changes no output: no production path calls the sparse-attention dispatcher. `auto` honours `RMLX_SPARSE_ATTN=1`. |

**Kernel gates resolve once, in `main`.** `--turbo-flash`,
`--turbo-flash-lock`, `--fused-qk`, `--sparse-attn`, `--planar-flash-decode`
and `--rot-k-fused` fold into one
[`DispatchPolicy`](../crates/rmlx-core/src/dispatch_policy.rs) before the
subcommand runs. So `bench` and `baseline` measure the kernels `serve` runs.
Each KV cache captures the policy when it is built. A library that embeds rMLX
without the CLI gets `DispatchPolicy::from_env()`;
`rmlx_core::set_dispatch_policy` replaces it for caches built later.
`--rotor-qjl` and `--planar-fused-qk` are process-wide `OnceLock`s instead.

---

## KV codec flags

`serve`, `chat`, `info`, `baseline`, `bench` and `eval ppl` take this set.
`--kv-boundary-layers` is on `serve`, `baseline`, `bench` and `eval ppl` only.
What each codec does, and which ones are accepted but change nothing, is in
[`KV_QUANT.md`](KV_QUANT.md) § "Codec disposition". `rmlx info
--list-cache-types` prints each codec's resident KV against bf16.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--kv-quant` | string | `auto` | KV codec. `auto` is unquantised bf16 on every architecture; see [`KV_QUANT.md`](KV_QUANT.md) § "The auto default". `bf16` and `none` are aliases. Conflicts with `--kv-preset`, `--cache-type-*` and `--kv-bits`. |
| `--kv-preset` | name | — | Named preset: `auto`, `fp16`, `q8`, `speed`, `quality`, `planar`, `planar3`, `k_only_planar`. No preset holds less resident KV than `fp16`. Conflicts with `--kv-quant`, `--cache-type-*`, `--kv-bits`. See [`KV_QUANT.md`](KV_QUANT.md) § "Named preset interface". |
| `--cache-type-k` / `--ctk` | tag | — | Per-side K codec, e.g. `q8_g128`, `bf16`. Conflicts with `--kv-quant`, `--kv-preset` and `--kv-bits`. Refused with `--registry` (exit 78). |
| `--cache-type-v` / `--ctv` | tag | — | Per-side V codec, e.g. `q4_g64`, `tq4`, `planar4`. Conflicts with `--kv-quant`, `--kv-preset` and `--kv-bits`. Refused with `--registry` (exit 78). |
| `--kv-bits` | float | — | mlx-lm bit-width alias; see the mapping below. Conflicts with `--kv-quant`, `--kv-preset` and `--cache-type-*`. |
| `--kv-group-size` | usize | `64` | Group size for `--kv-bits`: `32`, `64` or `128`. Requires `--kv-bits`. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | Leading and trailing layers held at the boundary floor. `0,0` turns the promotion off. Windowed and shared-KV consumer layers are unaffected, so the effect depends on the model. A non-default value is recorded in `decode_config`. See [`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md) § "Layer-adaptive overrides". |

A codec the resolver refuses for the loaded architecture exits 78
(`EX_CONFIG`) before the weights load.

**`--kv-bits` mapping.** `(8, 128)` gives `k8v8`. Any other integer gives
`mixed_k8g64_v<bits>g<group>`. A fraction gives
`mixed_k<floor>g<group>_v<ceil>g<group>`, e.g. `3.5` → `mixed_k3g64_v4g64`.
Valid integers are 2, 3, 4, 5, 6 and 8.

---

## Subcommands

### `serve`

Starts the HTTP server. Routes, request fields and error bodies are in
[`SERVER.md`](SERVER.md).

```bash
rmlx serve --model /path/to/snapshot
rmlx serve --registry /path/to/registry.json --port 9000
rmlx serve --profile myrun
```

`--model` and `--registry` are mutually exclusive. With neither, the server
starts with an empty registry. The KV codec flags above apply too.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | — | Model snapshot directory. |
| `--registry` | path | — | JSON registry: `{"models":[{"id":"name","path":"/abs/path"},…]}`. At startup the server loads the first `--max-loaded-models` ids in sorted order; the rest load on first request. |
| `--profile` | name | — | `[profile.<name>]` from `<RMLX_HOME>/profiles.toml`. A flag on the command line wins over the profile. |
| `--port` | u16 | `8080` | TCP port. |
| `--host` | string | `127.0.0.1` | Bind address. |
| `--device` | `cpu \| gpu` | `gpu` | Inference device. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | See "KV codec flags". |
| `--max-ctx` | u32 | `min(capacity, 4096)` | Context ceiling in tokens. The KV ring grows up to it on demand; see [`KV_CACHE.md`](KV_CACHE.md) §4.6. A value above the positional capacity is refused; see "Context ceiling". Must be ≥ 256. |
| `--yarn-factor` | f32 | — | YaRN RoPE factor; raises the positional capacity. Env: `RMLX_YARN_FACTOR`. See "Context ceiling". |
| `--yarn-original-max` | u32 | from `config.json` | Window `--yarn-factor` scales from: `original_max_position_embeddings`, else `max_position_embeddings`. Env: `RMLX_YARN_ORIGINAL_MAX`. |
| `--idle-timeout-secs` | duration | `15m` | Idle time before a model unloads: seconds (`900`) or `30s`, `15m`, `2h`. Negative pins the model; `0` unloads after each response. |
| `--prompt-cache-slots` | usize | `4` | Prompt-cache slots. `0` disables the cache; see [`PROMPT_CACHE.md`](PROMPT_CACHE.md) § "Zero slots". Each active `X-Session-Id` session adds one slot when the value is not `0`. |
| `--session-cache-max-sessions` | usize | `64` | Sessions the session cache holds; the least recently used one loses its slot. Env: `RMLX_SESSION_CACHE_MAX_SESSIONS`. |
| `--prompt-cache-ram-gb` | f64 | `2.0` | RAM cap for the prompt cache, in GiB. |
| `--prefix-index` | `linear \| radix` | `linear` | Longest-prefix lookup: `linear` scans every slot, `radix` walks a positional radix tree. |
| `--kv-ssd-cache-gb` | f64 | `0` | SSD prompt-cache budget per namespace, in GiB. Blocks go to `<RMLX_HOME>/cache/kv/<namespace>/`. See [`SSD_TIER.md`](SSD_TIER.md). |
| `--kv-ssd-global-gb` | f64 | `0` | SSD budget across all namespaces, in GiB. The tier is on when either budget is above `0`. With both set, the tighter one binds a namespace. |
| `--project` | name | model id | SSD namespace. Refused unless `--kv-ssd-cache-gb` is above `0`. |
| `--paged-kv` | bool flag | off | Routes K8V4, K8V8 and Planar caches through paged block storage. Refused (exit 1) when the resolved codec is bf16, so `auto` needs an explicit `--kv-quant`, and with `--cache-type-k rot_k*`. |
| `--paged-kv-page-tokens` | i32 | `32` | Tokens per page; must be positive. Requires `--paged-kv`. |
| `--draft-model` | path | — | Drafter snapshot for speculative decoding: a sidecar head or a smaller model of the verifier's family. Must differ from `--model`. See [`SPECULATIVE.md`](SPECULATIVE.md) § "Which drafter a snapshot is". |
| `--draft-kind` | `mtp \| dflash \| dflash2 \| eagle3 \| two_model` | from the snapshot | Drafter kind for a snapshot whose `config.json` names none. Refused when it contradicts the snapshot. Requires `--draft-model`. Env: `MLX_VLM_DRAFT_KIND`. |
| `--draft-block-size` | usize | `5`, capped by the drafter's declared depth | Tokens the verifier scores per round, its own included. Refused below 2 or above 1024. A DFlash checkpoint's `block_size` also caps it. Env: `MLX_VLM_DRAFT_BLOCK_SIZE`. |
| `--max-tokens-cap` | u32 | `1048576` | Per-request `max_tokens` ceiling; a request above it gets HTTP 400. |
| `--max-timeout-secs` | u64 | `600` | Per-request wall-clock cap in seconds. `X-Request-Timeout-Seconds` can lower it. `0` disables it. |
| `--max-loaded-models` | usize | `1` | Models held resident; the least recently used one is evicted past the cap. |
| `--max-queue-depth` | usize | `64` | Admission queue depth; a request past it gets HTTP 429. `0` is unlimited. |
| `--adaptive-admission` | bool flag | off | Adaptive admission controller: adjusts the queue depth and answers 503 with `Retry-After` when the predicted step time exceeds twice `--step-target-ms`. See [`SERVER.md`](SERVER.md) § "Adaptive admission controller". |
| `--step-target-ms` | u64 | `500` | End-to-end step target. Hidden alias `--ttft-target-ms`. No effect without `--adaptive-admission`. |
| `--itl-target-ms` | u64 | `50` | Inter-token latency target. No effect without `--adaptive-admission`. |
| `--adaptive-prefill-chunk` | bool flag | off | Lets the controller set a process-wide prefill chunk in `[32, 2048]`. That override replaces every per-architecture default, including `RMLX_PREFILL_CHUNK*`, for the rest of the process. No effect without `--adaptive-admission`. |
| `--require-smoke-probe` | bool flag | off | Runs the 8-token smoke probe on every model load; a broken verdict fails the load with HTTP 503. |
| `--default-temperature` | f32 | — | Temperature for a request that omits it; in `[0.0, 2.0]`. Precedence: request, this flag, `generation_config.json`, `1.0`. |
| `--enable-thinking` | bool | — | Default thinking mode for Qwen3-family models. A request's `enable_thinking` wins. |
| `--image-max-tokens` | usize | `max_soft_tokens` from `processor_config.json` | Per-image soft-token budget for Gemma4-unified vision, clamped to 1120. `0` is refused. A request's `image_max_tokens` wins. |
| `--mm-cache-bytes` | usize | `536870912` | Multimodal encoder-output cache budget, keyed on input content and model. `0` disables it. Env: `RMLX_MM_CACHE_BYTES`. |
| `--whisper-model-path` | path | — | Whisper snapshot for `/v1/audio/transcriptions` and `/v1/audio/translations`. Env: `RMLX_WHISPER_MODEL_PATH`. |
| `--whisper-tokenizer-path` | path | — | Directory with the Whisper `tokenizer.json`. Env: `RMLX_WHISPER_TOKENIZER_PATH`. |
| `--tts-model-path` | path | — | Qwen3-TTS snapshot for `/v1/audio/speech`. Env: `RMLX_TTS_MODEL_PATH`. |
| `--tts-tokenizer-path` | path | — | Qwen3-TTS speech-tokenizer snapshot; `/v1/audio/speech` needs both TTS paths. Env: `RMLX_TTS_TOKENIZER_PATH`. |

`--kv-quant` and `--max-ctx` are launch defaults. A request can override both
for itself; see [`SERVER.md`](SERVER.md) § "Per-request KV-config hot-swap".
The SSD budgets are fixed at launch and enforced for the life of the process;
see [`SSD_TIER.md`](SSD_TIER.md) § "Evict-to-budget (runtime)".

**Keep-alive.** A model's idle timer resets after every request and never
unloads a model that is decoding. `POST /v1/models/{id}/load` takes a
`keep_alive` field in the same syntax, which wins over the flag. The OpenAI
and Anthropic routes ignore that field but still reset the timer. See
[`SERVER.md`](SERVER.md) § "Model lifecycle endpoints". An unload frees the
model, not the claim file; only process exit releases it.

### Context ceiling

Every context cap comes from one function,
`rmlx_models::context::resolve_context`. Its callers are the KV ring sizing,
the server's `context_length_exceeded` guard, the per-request `max_ctx` and
the default `--max-prompt-tokens`. `crates/rmlx-models/src/context_tests.rs`
lists them.

**Positional capacity** is what the checkpoint's RoPE can address:

| Situation | Capacity | Behaviour |
|---|---|---|
| plain RoPE | `max_position_embeddings` (mpe) | — |
| `rope_scaling` in `config.json` | `max(mpe, factor × original_max)` | extended silently |
| `--yarn-factor f` (`--yarn-original-max n`) | `max(mpe, f × n)` | extended; every run past the trained window logs a `warn!` |

A `--max-ctx` above the capacity is refused, never clamped. The refusal
happens at model load and is fatal. `baseline` and `bench` exit non-zero.
`serve` aborts during the startup load, before it binds the port. The message
names the request, the capacity, the trained window and the flag that would
lift it.

`--yarn-factor` overrides a `rope_scaling` that the config declares. Only
Qwen3 implements RoPE scaling. On any other architecture the flag is ignored
with a `warn!`, and a refusal does not offer it.

The `slots: model loaded` log line (`effective_max_ctx`, `positional_max`)
and `GET /v1/models` (`max_ctx`, `positional_max`) report the numbers. An
architecture with no `max_position_embeddings` has no capacity to enforce, so
neither field is published for it.

---

### `chat`

Loads `config.json`, resolves the KV flags, takes the claim, prints one line
and exits. It runs no model and reads no input.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Model snapshot directory. |
| `--device` | `cpu \| gpu` | `gpu` | Device. |
| `--max-ctx` | u32 | `min(capacity, 4096)` | Context ceiling; see "Context ceiling". |

Plus the KV codec flags, without `--kv-boundary-layers`.

---

### `transcribe`

Transcribes one audio file with the engine behind `POST
/v1/audio/transcriptions`. The backend is chosen from the snapshot's
`config.json`; Whisper is the only one. Any container Symphonia reads is
decoded and resampled to 16 kHz mono. Decoding is greedy at temperature 0.

```bash
rmlx transcribe meeting.m4a --model <whisper-snapshot> --tokenizer <tok-dir> \
  --format vtt --output meeting.vtt
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `<AUDIO>` | path | required | Input audio file. |
| `--model` | path | required | Whisper snapshot. Env: `RMLX_WHISPER_MODEL_PATH`. |
| `--tokenizer` | path | the model directory | Directory with `tokenizer.json`; Whisper snapshots ship none. Env: `RMLX_WHISPER_TOKENIZER_PATH`. |
| `--format` | `txt \| json \| srt \| vtt` | `txt` | Output format; `json`, `srt` and `vtt` carry per-segment times. |
| `--language` | string | `auto` | Language code, or `auto` to detect it. |
| `--translate` | bool flag | off | Translate to English. |
| `--output` | path | stdout | Output file. |
| `--device` | `cpu \| gpu` | `gpu` | Device. |

---

### `info`

Prints architecture and quantization metadata without running inference.

```bash
rmlx info --model /path/to/snapshot
rmlx info --list-cache-types
rmlx info --model /path/to/snapshot --probe-smoke
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | — | Model snapshot directory. Required unless `--list-cache-types`. |
| `--list-cache-types` | bool flag | off | Prints the KV codec table with each codec's resident KV, then exits. No model load. |
| `--probe-forward` | bool flag | off | One forward pass over token id 2; prints the top-1 token id and the max logit. |
| `--probe-smoke` | bool flag | off | 8-token smoke probe. Exit `0` ok, `1` broken, `3` load failed, `4` inconclusive, `5` unsupported architecture. |
| `--device` | `cpu \| gpu` | `gpu` | Device for the probes. |
| `--max-ctx` | u32 | `min(capacity, 4096)` | Context ceiling; see "Context ceiling". |

Plus the KV codec flags, without `--kv-boundary-layers`. The smoke probe
renders its seed prompt through the snapshot's `chat_template.jinja` when
there is one, so it sees the input a served request sees.

---

### `baseline`

One generation per process, with EOS stop off. Prints one summary line.
`--record` also writes one `runs.db` row.

```bash
rmlx baseline --model /path/to/snapshot
rmlx baseline --model /path/to/snapshot --prompt-tokens 4096 --max-tokens 128 \
  --record
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Model snapshot directory. |
| `--prompt` | path | `crates/rmlx-cli/tests/fixtures/baseline_prompt.txt` | Prompt file; see "Prompt files". Conflicts with `--prompt-tokens`. |
| `--prompt-tokens` | u32 | — | Uses `longctx_<N/1024>k.json` from the prompts directory. |
| `--prompts-dir` | path | the nearest `prompts/` holding `longctx_4k.json`, walking up from the working directory | Prompts directory. Env: `RMLX_PROMPTS_DIR`. |
| `--prompt-label` | string | the file name | Value of the record's `prompt` column. |
| `--device` | `cpu \| gpu` | `gpu` | Device. |
| `--max-tokens` / `--gen-tokens` | u32 | `32` | Tokens to generate. |
| `--max-ctx` / `--ctx-max` | u32 | `min(capacity, 4096)` | Context ceiling; see "Context ceiling". |
| `--yarn-factor`, `--yarn-original-max` | | — | As on `serve`. |
| `--max-prompt-tokens` | usize | the context ceiling | Prompt cap; see "Prompt-length cap". Must be ≥ 1. |
| `--allow-truncate` | bool flag | off | Truncate an over-cap prompt on the GPU instead of refusing it. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | See "KV codec flags". |
| `--label` | string | — | Written to the record's `notes` column. |
| `--record` | bool flag | off | Writes a [`METRICS_DB.md`](METRICS_DB.md) §8.5 record to the buffer and ingests it into `runs.db`. |
| `--git-sha` | string | — | Commit SHA for the record's `git_sha`; the binary never derives one. |
| `--emit-token-ids` | bool flag | off | Adds a `baseline: token_ids=<ids>` line, so an A/B harness can compare token streams. |

Plus the KV codec flags.

```text
baseline: model=<name>  load=<ms>  ttft_ms=<ms>  decode_tps=<n>  overall_tps=<n>
          prefill_tps=<n>  prompt_tokens=<n>  peak_rss=<n>MB
          metal_peak_mb=<n>  metal_gen_alloc_mb=<n>  kv_cache_bytes=<n>
```

- `peak_rss` is host RSS from `ps`.
- `metal_peak_mb` is the Metal allocator's peak during generation, weights
  included. `metal_gen_alloc_mb` is that peak minus what was live at the
  start. Only `metal_gen_alloc_mb` compares between two runs. Both read `0`
  with no Metal allocator.
- `kv_cache_bytes` is the filled prefix of the KV cache
  (`KvCache::resident_bytes`, [`METRICS_DB.md`](METRICS_DB.md) §4), not an
  allocator peak. When the reported count is zero it reads `n/a`, and the
  record omits the column; the timing row stands.
  `scripts/perf_ab.sh` parses it.

**Refusals.** If the model's KV byte counter did not advance, the generation
ended early without an error. `baseline` then prints no summary, writes no
record and exits non-zero.

#### Prompt files

A file that parses as JSON with a non-empty `messages` array is a chat
fixture. Its messages go through the model's `chat_template.jinja` before
tokenizing, as on the HTTP route. So `--prompt-tokens N` measures N content
tokens. Any other file is tokenized as plain text. A chat fixture whose
messages are not `{"role": "<string>", "content": "<string>"}` is an error.

#### Prompt-length cap

The cap defaults to the resolved context ceiling. A prompt over the cap:

- `--device cpu`: truncated, with a `warn!`.
- `--device gpu`, default cap, no `--allow-truncate`: an error. Raise
  `--max-ctx` to measure the whole prompt.
- `--device gpu` with `--allow-truncate` or an explicit
  `--max-prompt-tokens`: truncated, with a `warn!`.

An explicit `--max-prompt-tokens` above the ceiling is refused on either
device.

#### GPU capture (`metal-capture` builds only)

These flags exist only in a binary built with
`--features rmlx-cli/metal-capture`. See [`PROFILING.md`](PROFILING.md) §5.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--gpu-capture` | path | — | Writes a `.gputrace` of a window of decode steps. Conflicts with `--record`, and forces `--metrics off`. |
| `--gpu-capture-skip` | u32 | `4` | Decode steps before the window. Requires `--gpu-capture`. |
| `--gpu-capture-steps` | u32 | `8` | Decode steps in the window. Requires `--gpu-capture`. |

The process needs `MTL_CAPTURE_ENABLED=1` at launch; `scripts/gpu_capture.sh`
(`make profile-gputrace`) sets it. These fail before the model loads: no
capture layer, an occupied destination, a missing parent directory, an empty
window. So does a `--max-tokens` below `skip + steps + 2`.

---

### `bench`

Runs one cell `--warmup` + `--runs` times in one process and prints the median
and range of each metric. It writes nothing to `runs.db`.

```bash
rmlx bench --model /path/to/snapshot --prompt-tokens 4096 --max-tokens 128
rmlx bench --model /path/to/snapshot --prompt-tokens 32768 --max-ctx 40960 \
  --kv-quant k8v4 --runs 5 --json
```

| Metric | Meaning |
|---|---|
| `ttft_ms` | Prefill through the first token. |
| `itl_p50_ms` / `itl_p99_ms` | Nearest-rank percentiles of the gaps between tokens within a run. |
| `decode_tps` | Rate over tokens 2..N. |
| `prefill_tps` | Prompt tokens / TTFT; `n/a` when undefined. |
| `kv_cache_bytes` | Filled-prefix KV bytes after decode. |
| `token_digest` | FNV-1a-64 of the run's token ids. |

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Model snapshot directory. |
| `--prompt`, `--prompt-tokens`, `--prompts-dir` | | | As on `baseline`. |
| `--device` | `cpu \| gpu` | `gpu` | Device. |
| `--max-tokens` / `--gen-tokens` | u32 | `128` | Tokens per run. |
| `--runs` | u32 | `3` | Measured runs; must be ≥ 2. |
| `--warmup` | u32 | `1` | Discarded runs before them. |
| `--max-ctx` / `--ctx-max` | u32 | `min(capacity, 4096)` | Context ceiling; see "Context ceiling". |
| `--max-prompt-tokens`, `--allow-truncate` | | | As on `baseline`. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | See "KV codec flags". |
| `--json` | bool flag | off | One JSON object with every run instead of the table. |
| `--temperature` | f32 | `0.0` | `0` is greedy on the GPU. Above `0` routes through the host sampler. In `[0, 2]`. |
| `--top-p` | f32 | `1.0` | In `(0, 1]`; needs `--temperature` above `0`. |
| `--top-k` | u32 | `0` | `0` is off; needs `--temperature` above `0`. |
| `--repetition-penalty` | f32 | `1.0` | Over the last 20 tokens; must be positive. `1.0` is a no-op. |

Plus the KV codec flags. The sampler seed is fixed and the RNG is fresh per
run, so a sampled cell repeats its token stream. The host sampler's cost is in
[`SAMPLING.md`](SAMPLING.md) § "Cost of the host path".

`bench` clears the prompt cache before every run. A `baseline` TTFT is a
first generation in a fresh process; a `bench` TTFT is a warmed one. Compare
each only with its own kind.

**Refusals.** `bench` exits with an error, and prints no result, when:

- `--runs` is below 2;
- a run hit the prompt cache, or the cache counters are missing;
- `ttft_ms`, `decode_tps`, `itl_p50_ms`, `prefill_tps` or `kv_cache_bytes`
  moved by more than 10% of its median from the first run to the last, on a
  line fit in run order (raise `--warmup`);
- two runs produced different token digests, warmup included;
- the KV byte counter did not advance, or reported zero;
- the token count and the callback count differ;
- fewer than 2 tokens were generated.

**Host load.** `bench` reads the 1-minute load average before and after the
runs. If either is at or above the CPU count, the summary is marked
`CONTENDED` and a `warn!` is logged. An unreadable load average prints `n/a`.

---

### `kv-calibrate`

```bash
rmlx kv-calibrate /path/to/snapshot --recipe turbo3
rmlx kv-calibrate /path/to/qwen3-snapshot --recipe softmax_mass
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `MODEL` | path | required | Model snapshot directory. |
| `--recipe` | see below | `turbo3` | Calibration recipe. |
| `--out` | path | `<MODEL>/kv_calib.json`, or `<MODEL>/head_budgets.json` | Output file. |
| `--prompts` | path | `prompts/calibration_long_context.json` for `softmax_mass`, `prompts/calibration_default.json` otherwise | Calibration prompts; head-budget recipes only. |
| `--mass-threshold` | f32 | `0.95` | Softmax-mass coverage in `[0.50, 1.00]`; head-budget recipes only. |
| `--target-mass-budget-floor` | u32 | `16` | Minimum per-(layer, head) budget; `softmax_mass` only. |

**Weight-norm recipes** (`turbo2`, `turbo2_tcq`, `turbo3`, `turbo3_tcq`,
`turbo4`) run on the CPU with no Metal claim. They need float weights (F32,
BF16 or F16). They write `kv_calib.json` in the `multi-turboquant`
`turboquant_kv.json` v1 schema. `turbo2*` keeps 25% of `head_dim` at high
precision (`turboquant25`); the others keep 50% (`turboquant35`). The count is
rounded to a multiple of 16.

**Head-budget recipes** (`softmax_mass`, `head_budget`, `k_norm_proxy`) load a
`Qwen3ForCausalLM` snapshot on the GPU. Each prompt is capped at 768 tokens.
`softmax_mass` measures softmax mass from real Q·Kᵀ and writes
`head_budgets.json` schema v2. `head_budget` and its alias `k_norm_proxy` use
K-norm² as a stand-in and write schema v1. `serve` loads a
`head_budgets.json` it finds in the snapshot, but no production path reads it
for attention; see
[`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) § "Sparse attention".

---

### `healthcheck`

One JSON line per check, then an aggregate line. Without `--full` it loads no
MLX.

```text
{"check":"<name>","status":"green|red|info","detail":"..."}
{"check":"aggregate","status":"green|red","red_checks":["..."]}
```

Exit `0` all green, `1` any red, `2` internal error.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--registry` | path | — | Checks each registered model. Conflicts with `--model`. |
| `--model` | path | — | Checks one snapshot. Conflicts with `--registry`. |
| `--port` | u16 | — | Also checks the claim file and `/health` of a server on this port. |
| `--db` | path | `RMLX_METRICS_DB`, else `<RMLX_HOME>/metrics/runs.db` | Metrics DB. |
| `--min-disk-gb` | u64 | `5` | Free space required for `metrics/` and `logs/`, in GiB. |
| `--full` | bool flag | off | Also runs the smoke probe per model; loads MLX. |
| `--human` | bool flag | off | Plain `OK` / `FAIL` text. |

---

### `metrics`

Every `metrics` subcommand takes `--db <path>`. Without it the DB is
`RMLX_METRICS_DB`, else `<RMLX_HOME>/metrics/runs.db`. The schema, the record
shape and the rules are in [`METRICS_DB.md`](METRICS_DB.md).

`query`, `best`, `rank`, `compare`, `history`, `timeseries`, `regress`,
`deltas`, `describe`, `export` and `prompts list|get` never migrate. They
refuse a missing DB, and a DB whose `bests` view is stale, naming
`doctor --fix`. `champions` migrates the DB before it reads.

| Subcommand | Flags | Description |
|---|---|---|
| `init` | — | Creates the schema. Refuses an existing file. |
| `doctor` | `--fix` | Checks schema version, integrity, foreign keys, whitelists, units, directions and the [`METRICS_DB.md`](METRICS_DB.md) §4.1 bounds. `--fix` rebuilds a stale `bests` view; it never edits a value. |
| `backup` | `--out <path>`, `--keep <N>` | WAL-checkpointed copy; `--keep` prunes older backups. |
| `restore` | `--from <path>` (required) | Replaces the DB from a backup after snapshotting the current one. |
| `record` | `--inline <json>` \| `--file <path>` \| `--stdin` \| `--replay-pending`; `--dry-run` | Ingests one `METRICS_DB.md` §8.5 record. `--replay-pending` ingests every file in `metrics/buffer/pending/` and moves failures to `failed/`. `--dry-run` writes nothing. |
| `validate` | `--file <path>` \| `--stdin` | Runs `RunRecord::validate` and writes nothing; exit 1 on rejection. `record` checks more: it refuses a `sha256` prompt that is not registered. |
| `identity` | `--json` | Prints `backend`, `backend_version`, `build_profile`, `hardware_tag` for this binary. No `git_sha`; see `METRICS_DB.md` §8.5.1. |
| `best` | `--backend --namespace --model --weight-quant --kv-quant --metric` (required); `--ctx-max` (`8192`), `--decode-config`, `--prompt-id` \| `--prompt-name` | Champion row for one cell and metric. |
| `rank` | `--metric` (required), `--backend`, `--limit` (`20`) | Top champions for one metric. |
| `compare` | `--backends a,b` and `--metric` (required); `--namespace`, `--model`, `--weight-quant`, `--kv-quant` | Champions of several backends side by side. The four filters are accepted and ignored. |
| `history` | the `best` cell flags; `--metric`, `--since <date>` optional | Every observation of one cell, oldest first. |
| `timeseries` | the `best` flags; `--since`, `--bucket day\|week` (`day`) | Mean per bucket for one cell and metric. |
| `regress` | `--model <substring>`, `--metric` (required); `--kv`, `--threshold-pct` (`1.0`) | Latest observation against the champion. Exit `0` within tolerance, `1` regressed, `125` nothing to compare. |
| `deltas` | `--since-sha` (required), `--threshold-pct` (`5.0`), `--exit-code` (`true`) | Changes per cell and metric since a commit. A SHA with no observations is an error (exit 1). With `--exit-code true`: `1` when a row regressed past the threshold, `125` when no row has a baseline, `0` otherwise. |
| `describe` | `--observation-id` \| `--run-id`; `--text` (required) | Sets the `description` of one observation or of every observation in a run. |
| `query` | `<SQL>` | Runs one `SELECT`; TSV output. |
| `open` | `--readonly` | Opens the DB in `sqlite3`. |
| `export` | exactly one of `--markdown`, `--json`, `--csv`, `--jsonl`; `--scope <path>` | Prints the `bests` view to stdout. `--scope` filters and orders `--markdown` and is refused with any other format. |
| `champions` | `--backend`, `--jsonl` | One row per (namespace, model, weight quant, KV quant) with one column per metric. |
| `prompts` | `list`; `get --name`; `add --file [--name] [--notes]`; `sync` | Prompt registry. `sync` registers every `*.json` under `prompts/` in `RMLX_REPO_ROOT`, else the working directory. |
| `migrate` | `--rmlx-glob`, `--cbb-csv`, `--records-md`, `--hardware-tag` (`m5_max_128gb`) | Idempotent import of legacy JSONL, CSV and Markdown. |

`record` and `validate` refuse a record whose `notes` or `description`
contains `synthetic=true`. They also refuse an `rmlx` record whose
`backend_version` is missing or not semver.

---

### `eval ppl`

Perplexity by sliding-window NLL, for Qwen3, Gemma4 and Qwen3.5 (dense and
MoE). Prints one JSON line:

```text
{"ppl":..,"mean_nll":..,"scored_tokens":..,"windows":..}
```

```bash
rmlx eval ppl --model /path/to/snapshot --text-file wiki.txt \
  --corpus wikitext-2 --ctx-window 4096 --stride 2048
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Model snapshot directory. |
| `--text-file` | path | required | UTF-8 corpus. |
| `--ctx-window` | usize | `4096` | Tokens per window. |
| `--stride` | usize | `2048` | Stride between windows. |
| `--corpus` | string | `""` | Corpus name. A non-empty value ingests one `METRICS_DB.md` §8.5 record into `runs.db` |
| `--device` | `cpu \| gpu` | `gpu` | Device. |
| `--max-tokens` | usize | `0` | Tokens scored; `0` is the whole corpus. |
| `--git-sha` | string | — | Commit SHA for the record's `git_sha`. |
| `--kv-quant` | string | — | Codec to score through. Unset means no KV cache. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | Refused unless a KV codec flag is given. |

Plus `--kv-preset`, `--cache-type-*`, `--kv-bits` and `--kv-group-size`.

**Two scorers.** With no KV codec flag, each window is one forward pass with
no cache. With one, each window is teacher-forced through a real per-layer
cache, one decode step per scored token. That is slower, and it is the only
way a codec can move the number. Qwen3.5 refuses a KV codec flag, since its
GatedDeltaNet layers carry state no codec touches.

The record's metric is `ppl_<corpus>` without a cache and `ppl_<corpus>_cached`
with one; hyphens are dropped (`wikitext-2` → `ppl_wikitext2`). They are
different metrics; compare each only with its own kind.

---

### `profile list`

Prints the profile names in `<RMLX_HOME>/profiles.toml`, one per line. A
missing file prints nothing and exits `0`. Profiles apply to `serve` only.

```toml
[profile.myrun]
model = "/abs/path/to/snapshot"
port = 9001
kv_quant = "k8v4"
max_ctx = 8192
```

Keys: `model`, `registry`, `port`, `host`, `device`, `kv_quant`, `max_ctx`,
`idle_timeout_secs` (integer seconds), `prompt_cache_slots`, `draft_model`,
`max_tokens_cap`, `max_timeout_secs`, `max_loaded_models`, `max_queue_depth`,
`default_temperature`. Every other flag is command-line only.

---

## `qwen36_diag` binary

A Qwen3.6 forward-pass check, built from
`crates/rmlx-cli/src/bin/qwen36_diag.rs`. It feeds a fixed 20-token prompt.

```bash
qwen36_diag <model-dir> [cpu|gpu]     # one forward pass: argmax, max |logit|
qwen36_diag <model-dir> <device> <N>  # greedy generation of N tokens
```

The device defaults to `cpu`. The mlx-lm reference is `argmax_id=8160`,
`max_abs_logit=29.75`.

---

## Environment variables

Where a variable has a flag, the flag wins.

| Variable | Flag | Read by | Effect |
|---|---|---|---|
| `RMLX_HOME` | — | `rmlx_core::paths` | Root of all on-disk state. A relative path is ignored with a `warn!`. Else `<workspace>/.rmlx/` (nearest `Cargo.lock` upward), else `$HOME/.rmlx/`. |
| `RUST_BACKTRACE` | — | `main` | Set to `full` at startup when unset. |
| `RUST_LOG` | `--log` | the tracing filter | Overrides `--log` when set, e.g. `RUST_LOG=debug,rmlx=trace`. |
| `RMLX_LOG_CAP_MB` | `--log-cap-mb` | clap | Log directory cap. |
| `RMLX_METRICS_DB` | `--db` | `rmlx metrics`, `rmlx healthcheck` | DB path. The event recorder and in-process ingest always use `<RMLX_HOME>/metrics/runs.db`. |
| `RMLX_HARDWARE_TAG` | — | `rmlx_metrics::identity` | `hardware_tag` of every record this binary emits. Default `m5_max_128gb`. |
| `RMLX_REPO_ROOT` | — | `metrics prompts sync`, `metrics migrate` | Directory holding `prompts/`. Default: the working directory. |
| `RMLX_PROMPTS_DIR` | `--prompts-dir` | clap | Prompts directory for `baseline` and `bench`. |
| `RMLX_YARN_FACTOR`, `RMLX_YARN_ORIGINAL_MAX` | `--yarn-factor`, `--yarn-original-max` | clap | `serve` and `baseline` only. |
| `RMLX_SESSION_CACHE_MAX_SESSIONS` | `--session-cache-max-sessions` | clap | `serve`. |
| `RMLX_MM_CACHE_BYTES` | `--mm-cache-bytes` | clap | `serve`. |
| `RMLX_WHISPER_MODEL_PATH`, `RMLX_WHISPER_TOKENIZER_PATH` | `--whisper-*`, `transcribe --model` / `--tokenizer` | clap | Whisper paths. |
| `RMLX_TTS_MODEL_PATH`, `RMLX_TTS_TOKENIZER_PATH` | `--tts-*` | clap | Qwen3-TTS paths. |
| `MLX_VLM_DRAFT_KIND`, `MLX_VLM_DRAFT_BLOCK_SIZE` | `--draft-kind`, `--draft-block-size` | clap | `serve`. |
| `RMLX_TURBO_FLASH`, `RMLX_FUSED_QK`, `RMLX_SPARSE_ATTN`, `RMLX_PLANAR_FLASH_DECODE`, `RMLX_ROT_K_FUSED` | the matching gate | `DispatchPolicy::from_env` | `=1` turns the gate on under `auto`. |
| `RMLX_TURBO_FLASH_LOCK` | `--turbo-flash-lock` | `DispatchPolicy::from_env` | `=1` turns the lock on when the flag is absent. |
| `RMLX_TURBO_FLASH_MIN` | — | `DispatchPolicy::from_env` | TurboFlash runs only above this `kv_seq`. Default `4096`; a negative value is `0`, an unparseable one warns and keeps the default. |
| `RMLX_FUSED_QK_MIN` | — | `DispatchPolicy::from_env` | Minimum `kv_seq` for fused-QK. Default `512`; an unparseable value warns and keeps it. |
| `RMLX_ROTOR_QJL` | `--rotor-qjl` | `rmlx_kv_quant::rotor_qjl` | Only for an embedder that never installs the flag; `rmlx` always does. `1`, `on`, `true` or `yes` turns QJL on. |
| `RMLX_PREFILL_CHUNK`, `RMLX_PREFILL_CHUNK_<ARCH>` | — | `rmlx_models::prefill_chunk` | Prefill chunk in tokens; the per-architecture form wins. See [`KV_CACHE.md`](KV_CACHE.md) § "Chunked prefill". |
| `RMLX_KV_MAX_SEQ_HARD_CAP` | — | `rmlx_kv_quant::kvcache::update` | Refuses a KV extension past this many tokens. Unset: no cap. |
| `RMLX_EAGLE3_NO_FCS` | — | `speculative::eagle3` | Set to any value to skip the Eagle3 drafter's per-slice `fcs` norms. |
| `MTL_CAPTURE_ENABLED` | — | Metal | Must be `1` at launch for `--gpu-capture`. |

---

## Claim file

Metal allows one GPU context per process. rMLX takes a claim file before it
uses the GPU. The file is `/tmp/rmlx.<port>.claim`, locked with
`flock` and holding the owner's PID. Internals are in
[`SERVER.md`](SERVER.md) § "Claim file".

| Holder | Port |
|---|---|
| `serve` | `--port` |
| `chat`, `transcribe`, `baseline`, `bench`, `eval ppl`, `info --probe-forward`/`--probe-smoke` | `51966` (`0xCAFE`) |
| `kv-calibrate` head-budget recipes, during the model load only | `0` |
| `healthcheck --full` (runs the smoke probe on the GPU) | none |

A claim refuses only a second holder of the same port. That process names the
PID and exits with code `11`. So these pairs can hold the GPU together:

- a `serve` and any one-shot command;
- two `serve` processes on different ports;
- a one-shot command and a `kv-calibrate` model load;
- `healthcheck --full` and anything;
- a `kv-calibrate` measurement, which runs after the claim is released, and
  anything.

Other rules:

- `--device cpu` takes no claim.
- A claim whose PID is dead is reclaimed with a `warn!`; nothing needs
  removing by hand.
- Normal exit removes the file; `serve` also removes it on `SIGINT` and
  `SIGTERM`. `info --probe-smoke` with a non-zero verdict exits without
  removing it, and the next holder reclaims it.
- `rmlx healthcheck --port <N>` checks the file without taking it.

---

## See also

- [`METRICS_DB.md`](METRICS_DB.md): schema, record shape, `metrics`
  subcommands, operating rules.
- [`PROJECTS_CONFIG.md`](PROJECTS_CONFIG.md): per-project SSD caps in
  `projects.toml`.
- [`PROFILING.md`](PROFILING.md): samply, Instruments, dhat and GPU capture.
- [`TESTING.md`](TESTING.md): test setup and the test-model variables.
- [`KV_QUANT.md`](KV_QUANT.md): the KV codecs and the flags that select them.
