# Changelog

All notable changes to rMLX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-06-17

Correctness + maintenance release. Closes a systemic KV-cache head-scramble
class that affected **every** flat quantized KV codec, hardens the SSD KV tier
and prompt cache, makes the single-MLX Metal claim self-heal after a crashed
holder, and unifies the per-architecture model code onto shared seams (decode
loop, loader, `Architecture` dispatch). Plus a round of dependency bumps. No
breaking changes.

### Fixed

- **Systemic KV head-scramble class.** Every flat quantized KV codec wrote its
  buffer sequence-major but reshaped it head-major on dequant — agreeing only
  when `batch × kv_heads == 1`, and scrambling per-head K/V on any multi-append
  (decode after a multi-token prefill, or after an SSD hydrate) when
  `kv_heads > 1` (grouped-query attention). Fixed family-wide with a canonical
  sequence-major layout (transpose on append + on dequant) and an explicit
  `Array::contiguous` before each custom MSL kernel, which reads its input by
  raw linear index and so cannot honor a lazy transpose. Covers `QuantK`
  (#103), `QuantV` / TurboSym-K / paged-K handoff (#108), the Iso/Rotor
  rotation codecs (#109), and PlanarQuant K/V plus its packed-K decode kernels
  (#110).
- **SSD KV tier.** Spill + restore now carry the bf16 K/V payload for
  `KvQuant::None` layers (#88); SSD-hydrated entries are excluded from the
  exact-hit fast path so a hydrate cannot be mistaken for an exact prompt-cache
  hit (#87); a Gemma 4 entry hydrated with an empty SWA layer degrades to a
  full re-prefill instead of decoding from a hole (#90).
- **Prompt cache unified across architectures.** A single model-agnostic
  `consume` engine replaces the per-arch hydrate/reuse glue and is retrofitted
  onto five architectures, so the SSD-hydrate / prefix-reuse correctness fixes
  above hold identically on every model (#98).
- **GPU default stream on every inference entry.** The image, speculative,
  audio, and embeddings blocking-thread entries now establish the thread-local
  GPU stream the text path already had, fixing intermittent
  `no Stream(gpu, N)` failures off the text path (#104). The adaptive
  prefill-chunk fallback resolves the loaded architecture instead of assuming
  Gemma 4 (#68).
- **Metal claim self-heals.** A stale claim left by a crashed holder is
  auto-reclaimed once the holder PID is proven dead (re-probed under the file
  lock); `SIGTERM`/`SIGINT` now shut the server down gracefully and release the
  claim (#112).
- **`Array::to_bytes` evaluates before reading the data pointer**, closing a
  lazy-eval race in the only reader of the raw MLX array buffer (#101).
- `MetalKernel::new` frees its input vector when output-name conversion fails
  (#60); the Planar3 V codec uses one packing path on CPU and GPU (#102) and
  warms its MSL kernels at precompile (#59); the resident-bytes estimator
  models Iso/Rotor sidebands exactly (#58); `chunked_prefill` exits prefill on
  every cache after a failure (#57); f16 negative subnormals no longer decode
  to `-0.0` (#56); the tensor-view loader distinguishes not-found from I/O /
  parse failures (#4b5ea54 → see history).
- **Gemma 4 loading:** unquantized bf16 and affine-int4 (QAT) snapshots load;
  affine biases pass through the MoE expert `gather_qmm`; the perplexity scorer
  prepends BOS to every sliding window.

### Changed

- **Shared decode loop.** Qwen 3, Qwen 3.5-MoE, Gemma 4, and Gemma 3 now run on
  one decode loop (per-arch copies removed); `ProbeStep` / `SmokeVerdict` live
  in the shared loop.
- **Shared loader seam.** All architecture loaders (Gemma 4 / 3, Qwen 3 /
  3.5-MoE / 3-VL-MoE, Laguna) adopt `load_util::Weights` — an index-first,
  header-truth tensor fetch; AWQ byte-math moved to `rmlx-quant`; a single
  `read_raw_config` helper replaces six per-loader clones.
- **`Architecture` dispatch.** Auto-KV default, KV-byte reporting, and
  prompt-cache stats now dispatch through the `Architecture` trait rather than
  arch-specific branches.
- Shared fused-QK setup scaffold (q8 / turbo-K3 / turbo-K4 / iso dispatchers
  ported onto it); arch modules construct arrays via
  `Array::from_{i32,f32}_slice` per `docs/FFI.md`; `refuses_qwen_moe` renamed to
  `k_below_8bit` (it is a codec property, not an arch rule).

### Dependencies

- `tokenizers` 0.20 → 0.23 (encode/decode add-special / skip-special semantics
  preserved; verified on Gemma 4, Qwen 3.6, and Bonsai tokenizers) (#97).
- `toml` 0.8 → 1.1 (#94), `tikv-jemallocator` 0.6 → 0.7 (#96),
  `criterion` 0.5 → 0.8 (dev / benches) (#95), `uuid` 1.23.2 → 1.23.3 and
  `time` 0.3.47 → 0.3.49 (#93).

### Tested

- Full KV-codec regression re-sweep after the head-scramble fixes: every codec
  class (QuantK/V, Iso/Rotor, Planar including its live fused-QK kernel) is
  within ±5 % of its recorded best decode cell on Bonsai, Gemma 4-e4b, and
  Qwen 3.6 — no decode regression. GPU round-trip tests assert each layout flip
  reconstructs true head-major K/V at quant noise (with pre-fix scramble
  controls).
- Tokenizer correctness re-proven on three tokenizer families (SentencePiece +
  BPE) at temp 0.

## [0.2.0] - 2026-06-10

Gemma 4 decode is now competitive with mlx-lm across the whole family, Gemma 4
speculative decoding (MTP) works end to end, the KV ring grows lazily with
per-request KV / context hot-swap, KV-cache metrics report live sizes, and the
env-var surface is cleaned up — **breaking** for shell configs that set removed
vars directly (see Removed).

### Added

- **Per-request KV-quant + `--max-ctx` hot-swap** on a resident model — switch
  the KV codec or context ceiling per request without reloading the model. (#26)
- **Per-layer KV net-benefit estimator** — warns when a KV codec costs more
  resident bytes than it saves on a given layer mix (general across arches). (#34)
- Five env-var-only knobs promoted to proper `--flag` / `env=` pairs (the flag
  always takes precedence): `--log-cap-mb`, `--yarn-factor`,
  `--yarn-original-max`, `--session-cache-max-sessions`, `--prompts-dir`.

### Fixed

- **Gemma 4 speculative (MTP) functional end to end.** Dispatch routes
  `--draft-kind mtp` by draft arch family and rejects a plain-`gemma4` draft
  cleanly (#23); the assistant SWA mask uses array mode instead of the rejected
  additive mode (#24); a verify-step SWA mask off-by-one in both the producer
  and consumer branches is fixed (#32); and the loader supports both assistant
  LM-head variants — sparse centroid-routed (e2b/e4b) and plain tied-head
  (26b/31b) (#49). All four Gemma 4 sizes load and run coherent under MTP.
- **Gemma 4 decode kept bf16 end to end.** `gelu_tanh` f32 constants plus the
  embed / per-layer scales no longer promote the dense activation stream to f32
  (#44), and the MoE router's strong-f32 root-size scalar no longer leaks f32
  into the routing weights and the downstream KV (#51). Net: e2b/e4b beat mlx-lm
  decode, 26b-a4b MoE closed from −10…−28 % to −4…+1 %, and global `--kv-quant
  none` KV is halved (bf16) on every model.
- **`--max-ctx` is a virtual ceiling** — the KV ring grows on demand, so a high
  ceiling no longer penalizes small-prompt decode. (#25)
- **Rotation / K-only KV codecs** precompile their MSL kernels at load and are
  truthfully classified Metal vs CPU (no silent host-CPU fallback). (#36)
- **Qwen3.6-MoE SSD-hydrated prefix skips prefill** via a hydrated-tail path — a
  cache hit no longer re-runs the full prefill. (#9)
- **Live KV-cache metrics** — `kv_cache_bytes` reports the real resident size
  (was always 0) and counts the filled prefix, not the `--max-ctx` ceiling.
  (#33, #39)

### Performance

- **MoE prefill ~4× faster** on gemma4-26b and Qwen3.5-MoE via sorted-index
  expert gather (contiguous per-expert access in `gather_qmm`) — 26b 128k cold
  TTFT ~403 s → ~117 s. (#46)

### Tested

- Falsified the 6× SWA-KV claim: windowed SWA KV is window-bounded, not
  full-context (#35, #40).
- Full Gemma 4 and Qwen 3.6 KV × context bench matrices (per-model decode /
  TTFT / KV-size across all codecs) recorded under `docs/models/`.

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
- Dependency bumps: `safetensors` 0.4 → 0.7, `symphonia` 0.5 → 0.6.

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

[Unreleased]: https://github.com/Pushkinist/rMLX/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.1
[0.2.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.0
[0.1.1]: https://github.com/Pushkinist/rMLX/releases/tag/v0.1.1
[0.1.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.1.0
