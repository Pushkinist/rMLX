# Environment variables

Split out of [`CLI.md`](CLI.md) to keep that doc under the size cap. Where a
variable has a flag, the flag wins.

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
