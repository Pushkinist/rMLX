# Changelog

All notable changes to rMLX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Env-var surface cleanup** (`chore/env-var-cleanup`). Five previously
  env-var-only knobs are now proper `--flag` / `env=` pairs so the flag always
  takes precedence: `--log-cap-mb` (`RMLX_LOG_CAP_MB`), `--yarn-factor`
  (`RMLX_YARN_FACTOR`), `--yarn-original-max` (`RMLX_YARN_ORIGINAL_MAX`),
  `--session-cache-max-sessions` (`RMLX_SESSION_CACHE_MAX_SESSIONS`),
  `--prompts-dir` (`RMLX_PROMPTS_DIR`).
- `docs/CLI.md` env-var section restructured: split into **User / operational**
  and **Internal / advanced** subsections, with flag / default / description
  columns for every entry.
- `docs/TESTING.md`: added `RMLX_KV_TEST_MODEL`, `RMLX_DRAFT_TEST_MODEL`,
  `RMLX_VL_TEST_MODEL`, `RMLX_TEST_MODEL` to the specialised test-model table;
  added a **Test behaviour toggles** table covering `RMLX_SKIP_GPU`,
  `RMLX_REGEN_GOLDENS`, `RMLX_E2E_*`, `RMLX_REGISTRY_TEST`,
  `RMLX_NIAH_KV_QUANT`, and the `*_STRICT` flags.
- `.env.example` expanded to document all user-facing env vars: runtime data
  vars (`RMLX_HOME`, `RMLX_METRICS_DB`), all five newly-promoted flag-envs,
  audio path vars, `RMLX_MM_CACHE_BYTES`, `RMLX_SESSION_CACHE_MAX_SESSIONS`,
  draft compat keys, and prefill chunk tuning.

### Removed

The following env vars no longer have live readers in the Rust codebase.
**This is a breaking change** for any shell config that set them directly —
use the replacement flag instead.

| Removed variable | Replacement |
|---|---|
| `RMLX_KEEP_ALIVE` | `--idle-timeout-secs` |
| `RMLX_PROMPT_CACHE_MAX_BYTES` | `--prompt-cache-ram-gb` |
| `RMLX_PAGED_KV` | `--paged-kv` |
| `RMLX_KV_PAGE_SIZE` | `--paged-kv-page-tokens` |

The following debug / internal vars were dropped with no user-facing
replacement (they had no stable semantics across releases):

- `RMLX_SPEC_K` — undocumented experimental speculative-lookahead override.
  Its only value was the default; lookahead `K` is now fixed at 4. The
  independent `--draft-block-size` flag still controls the draft round size.
- `RMLX_MTP_DUMP`, `RMLX_DFLASH_DEBUG` — folded into `tracing` events; use
  `--log debug` or `RUST_LOG=rmlx=debug` instead.
- `RMLX_GIT_SHA` — was read for the metrics drainer's `git_sha` annotation but
  nothing ever set it (always `None`); the annotation now reuses the same
  `git rev-parse` helper the run ID uses, so it is populated in a git checkout.
- `RMLX_METAL_AVAILABLE`, `RMLX_METAL_CAPTURE` — doc-only, never implemented.
- `RMLX_METRICS_LOCK` — doc-only, never implemented (WAL handles concurrency).
- `RMLX_GPU_RESIDENT_ISO`, `RMLX_SPARSE_V_KERNEL`, `RMLX_SPARSE_V_THRESHOLD` —
  deep perf/kernel toggles, now hardcoded to their proven-best defaults
  (`off`, `on`, `1e-6`); the override env was removed (no perf change).
- `RMLX_OMODELS_DIR` — bench-script alias renamed to the canonical
  `RMLX_O_MODELS_ROOT`.

## [0.1.1] - 2026-06-06

Bug-fix + dependency-maintenance release.

### Added

- `rmlx baseline --max-prompt-tokens <N>` — the prompt-truncation cap (previously
  a hardcoded 65536) is now configurable, enabling ≥128k-context baselines
  (validated `>= 1`). (#11)

### Fixed

- Eagle3 speculative decode crashed mid-generation on Qwen3-MoE
  (`slice_update` zero-length KV dim). The drafter KV cache is now sized to the
  verifier context limit instead of a hardcoded 4096. (#8)
- SSD KV-tier spill failed with `no Stream(gpu, N) in current thread` and skipped
  persisting blocks. KV/lin caches are now materialized on the inference thread
  before the prompt-cache store, so the drain thread's eval is a no-op. Applies
  to qwen3.5-moe, qwen3, and gemma4. (#10)

### Changed

- Dependency bumps: `bindgen` 0.72 (FFI codegen — golden-token-verified
  behaviorally identical), `sha2` 0.11, `actions/checkout` 6, and a minor/patch
  group (`serde_json`, `tokio`, `minijinja`, `chrono`, `uuid`).

## [0.1.0] - 2026-06-06

First release. Native, single-binary [MLX](https://github.com/ml-explore/mlx)
inference + conversion backend for Apple Silicon — no Python at runtime.

### Added

- Text generation — OpenAI `/v1/chat/completions` + `/v1/completions` and an
  Anthropic-compatible surface (temperature, top-k/p, penalties, thinking
  budget, constrained / schema-guided decoding).
- Image input — vision towers (Gemma 4 SigLIP, Qwen3-VL-MoE deepstack) via
  `image_url` content parts.
- Audio input — transcription / translation for audio-capable models.
- Multimodal embeddings — `/v1/embeddings`, including text + image (jina-v4).
- Tool / function calling — OpenAI `tool_calls` + Anthropic `tool_use`,
  multi-turn, multiple emit formats (Qwen XML, Hermes-JSON, Gemma).
- Quantization — affine 2–8 bit, mxfp4 / mxfp8, nvfp4, ParoQuant weights; KV
  quant incl. fp8, TurboQuant, RotorQuant, PlanarQuant, IsoQuant, paged-KV,
  mixed / asymmetric K/V, and an SSD KV tier — including rotation-based KV
  families no other MLX server ships.
- Speculative decoding — MTP, DFlash, and Eagle3 drafters.
- Prompt caching — automatic prefix caching with block hashing.
- Conversion — `rmlx convert` re-quantizes / repacks MLX → MLX.

### Tested

- Golden-token decode gates (temp=0) for Gemma 4
  (`Gemma4ForConditionalGeneration`), Qwen 3.6
  (`Qwen3_5MoeForConditionalGeneration`), Bonsai (`Qwen3ForCausalLM`), and
  BitNet (`BitNetForCausalLM`).
- Multimodal embeddings (`jina-embeddings-v4`).
- Speculative drafters validated against their verifiers: Qwen 3.6 MTP sidecar
  and the Gemma 4 assistant drafter.

[Unreleased]: https://github.com/Pushkinist/rMLX/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Pushkinist/rMLX/releases/tag/v0.1.1
[0.1.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.1.0
