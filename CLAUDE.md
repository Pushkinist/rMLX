# rMLX — agent guide

Rust-native, single-binary MLX inference + conversion backend for Apple
Silicon. Goal: the fastest fully-featured **native, no-Python** backend for
MLX-format models.

## Local-only machine paths

Paths in this file are **relative on purpose** — it is checked in and public.
Concrete absolute machine paths (the model-snapshot root `RMLX_O_MODELS_ROOT`,
the single-MLX claim file under `/tmp`, and local sibling repos) live in a
**gitignored** `LOCAL.md` at the repo root. Use it as a local resolver; never
copy an absolute path from it into this file, a commit, a report, a log, or
any artifact that leaves the machine.

## What this project is

One `cargo build --release` binary that:

1. Loads any MLX-format model (`safetensors`, `mlx-community` layout) with
   **no Python at runtime**.
2. Serves an **OpenAI-compatible HTTP API** — text, plus image and audio
   input for models that support those modalities.
3. Supports the **widest weight × KV quantization matrix** MLX can express,
   including rotation-based KV families no other MLX server ships
   (TurboQuant, IsoQuant, PlanarQuant, ParoQuant).
4. **Converts** models between quant formats / layouts (re-quantize, KV-quant
   repack) — MLX in, MLX out.
5. Multi-model lifecycle (load on demand, unload on idle), but enforces a
   **single MLX process at a time** (Apple Silicon Metal context is exclusive
   per process).

## Documentation map

Subsystem references live under `docs/`. Read these to understand specific
areas before touching code:

| Doc | Topic |
|---|---|
| [`docs/CLI.md`](docs/CLI.md) | rmlx CLI: subcommands, flags, env vars, claim file |
| [`docs/SERVER.md`](docs/SERVER.md) | HTTP server: OpenAI/Anthropic compat, routes, tool calling, retry envelope |
| [`docs/MODELS.md`](docs/MODELS.md) | Per-architecture model reference (Qwen, Gemma, Laguna, Jina, etc.) |
| [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md) | New-arch integration surface: shared seams + per-arch points + verification ritual |
| [`docs/WEIGHT_QUANTS.md`](docs/WEIGHT_QUANTS.md) | Weight quantization formats (mxfp, affine, TurboQuant, PlanarQuant, ParoQuant) |
| [`docs/KV_QUANT.md`](docs/KV_QUANT.md) | KV-cache quantization variants (K8V4, K8V8, Mixed, Planar, Paged, rot_k) |
| [`docs/KV_CACHE.md`](docs/KV_CACHE.md) | KV cache architecture (block alignment, ring buffer, SWA snapshot, chunked prefill) |
| [`docs/SSD_TIER.md`](docs/SSD_TIER.md) | SSD KV tier (layout_key, ssd_index schema, hydrate, spill, cross-namespace LRU) |
| [`docs/SSD_CANARY.md`](docs/SSD_CANARY.md) | SSD KV cross-restart smoke probe |
| [`docs/PROMPT_CACHE.md`](docs/PROMPT_CACHE.md) | Prompt cache + automatic prefix caching (block hashing, ReusePolicy, prefix index) |
| [`docs/SPECULATIVE.md`](docs/SPECULATIVE.md) | Speculative decoding (MTP, DFlash, Eagle3 drafters; round-loop; accept-rate gates) |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Per-token sampling (temperature, top-k/p, penalties, thinking budget, constrained decoding) |
| [`docs/FFI.md`](docs/FFI.md) | rmlx-mlx ↔ mlx-c FFI bridge; MSL kernel surface; unsafe policy |
| [`docs/METRICS_DB.md`](docs/METRICS_DB.md) | Metrics DB: observations / events / bests; ingest, query, export, deltas |
| [`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md) | Recorded decode-TPS anchors per (model, KV quant) cell |
| [`docs/PROFILING.md`](docs/PROFILING.md) | samply / Instruments flamegraph workflow |
| [`docs/PROJECTS_CONFIG.md`](docs/PROJECTS_CONFIG.md) | Per-project cap defaults via `<RMLX_HOME>/projects.toml` |
| [`docs/TESTING.md`](docs/TESTING.md) | RMLX_TEST_MODEL_* env vars + RMLX_O_MODELS_ROOT for test snapshot resolution |
| [`docs/RELEASING.md`](docs/RELEASING.md) | Release flow: single-source version, `make tag` / `release-package` / `tap-sync`, Homebrew formula + tap, `CHANGELOG.md` |

Subdir `docs/superpowers/` holds process artifacts — not a subsystem reference.

## What this project is not

- Not a GGUF runtime — that is `llama.cpp`'s lane.
- Not training / fine-tune / fuse / lora-merge. Conversion is not training.
- Not a Python tool. Native Rust only.

## Status — where we are going

Target **0.1.0**: a fully functional native MLX backend with broad feature
and quantization coverage. Scope:

- **Text** generation, OpenAI-compatible.
- **Image input** for models that accept it (vision towers).
- **Audio input** for models that accept it.
- **Agent integration** — tool / function calling, multi-turn, the full
  agent-driving surface.
- **Models from the `RMLX_O_MODELS_ROOT` folder** served end-to-end.
- **Maximum quantization coverage** — every weight and KV quant we can
  support, including the rotation-based KV families.
- **Conversion** — quant↔quant and layout repack as a first-class command.

Build a fast, native, no-Python backend. Port from and study the sibling
repos rather than reinventing.

## Test targets

Under `RMLX_O_MODELS_ROOT` (the dev checkout uses `../../O-Models/`; public
users set it via `.env`). At minimum these three families must serve
end-to-end at every change:

| Family | Example snapshot | Arch |
|---|---|---|
| Gemma4 | `mlx-community__gemma-4-e4b-it-mxfp8`, `mlx-community__gemma-4-26b-a4b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| Qwen3.6 | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| Bonsai | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |

Other Open Models snapshots (`z-lab__Qwen3.6-27B-PARO`, `medgemma`, the `jina`
embedding/reranker models, `ReaderLM-v2`, …) are in scope as feature
coverage grows.

## Key external repos (GitHub)

- `oxideai/mlx-rs` — community Rust binding over `mlx-c`.
- `ml-explore/mlx-c` — Apple's stable C ABI.
- `huggingface/safetensors` — Rust safetensors crate.
- `z-lab/paroquant`, `ParaMind2025/isoquant` — rotation-KV references.

## Hard rules

1. **Apple Silicon only**. Metal first. No CUDA, no ROCm, no x86 SIMD.
2. **Single binary**. `cargo build --release` is the artifact. No bundled
   Python, no runtime data files (weights + chat templates are model-side).
3. **MLX-format only**. GGUF is out of scope. rMLX can re-quantize / convert
   MLX↔MLX itself; it never reads GGUF.
4. **No training**. No fine-tune / fuse / lora-merge. Quant and format
   conversion is allowed and in scope.
5. **Asymmetric K/V is real**, not a fake single-bit-width flag. See docs/KV_CACHE.md.
6. **Smoke-probe every new snapshot / quant** (short generation, reject
   incoherent output) before adding it to the registry.
7. **Document the truth, not the docstring**. If an upstream algorithm name
   lies, call it out in code + docs.
8. **Single MLX process per Mac**. Hold the claim file; unload competing MLX
   servers before claiming the GPU; never bypass the claim silently.
9. **`make ci-perf` builds + tests under `release-perf` (panic=unwind, debug-assertions off).** A failure there that doesn't reproduce under `dev` → rebuild under `release-debug` (full DWARF) and re-run the failing case to capture symbols. Never rely on the `dev` profile to reproduce a release-mode bug — codegen and inlining differ.

## Coding style

- Workspace `Cargo.toml` with member crates `crates/rmlx-{core,quant,kv-quant,kv-ssd,mlx,loader,metrics,models,runtime,server,cli,audio}`. `rmlx-kv-quant` owns the KV-cache codec layer (storage enums, MSL kernels, per-layer `KvCache`, paged-KV, mixed/rot-K, turbo/planar CPU codecs). `rmlx-kv-ssd` owns the SSD KV tier (index, spill, hydrate, block I/O, layout-key salt, 5 Prometheus hook globals, `SsdHydrate<E>` trait, FNV-1a-64 block-digest helpers); only the per-arch `attach_ssd_tier` dispatcher remains in `rmlx-models::ssd_tier` because the arch-specific `SpillSink<Entry>` / `SsdHydrate<Entry>` impls live in `rmlx-models`. The policy/builder wrappers (`KvCacheBuilder`, `kv_quant_for_layer`, `kv_quant_for_ctx`) stay in `rmlx-models::kv_cache`.
- `thiserror` for library errors, `anyhow` for binary entry-point.
- `tracing` for logging, not `log` or `eprintln`.
- Async only at boundaries (HTTP server, file I/O). Compute is sync.
- Tests in sibling `*_tests.rs` files; see "File-size + inline-test convention" below. Integration tests under `tests/`.
- No unsafe outside `rmlx-core` FFI module unless heavily justified + reviewed.
- Public API surface conservative — no leaking mlx-rs types directly.

## Workspace dep graph

Current member-crate edges (2026-05). `→` means "depends on".

```
rmlx-core    (root — no internal deps)
rmlx-mlx     → rmlx-core, rmlx-loader
rmlx-quant   → rmlx-core
rmlx-loader  → rmlx-core, rmlx-quant
rmlx-kv-quant → rmlx-core, rmlx-mlx
rmlx-metrics → rmlx-core
rmlx-kv-ssd  → rmlx-core, rmlx-mlx, rmlx-kv-quant, rmlx-metrics
rmlx-runtime → rmlx-core, rmlx-mlx, rmlx-loader, rmlx-metrics
rmlx-models  → rmlx-core, rmlx-mlx, rmlx-quant, rmlx-kv-quant, rmlx-kv-ssd, rmlx-loader, rmlx-runtime, rmlx-metrics
rmlx-audio   → rmlx-core, rmlx-loader
rmlx-server  → rmlx-core, rmlx-mlx, rmlx-kv-quant, rmlx-kv-ssd, rmlx-loader, rmlx-metrics, rmlx-models, rmlx-audio
rmlx-cli     → all of the above
```

The **direct** `rmlx-kv-quant` / `rmlx-kv-ssd` edges from `rmlx-server` and
`rmlx-cli` were added after dropping the `rmlx_models::kv_cache::*` re-export
shims. Every caller now imports codec items from `rmlx_kv_quant` and SSD-tier
items from `rmlx_kv_ssd` directly.

Hard rules:

* Codec layer (`rmlx-kv-quant`) must remain a leaf of `rmlx-models` — never
  reach into `rmlx-models` or `rmlx-runtime`. Higher-level policy stays in
  `rmlx-models::kv_cache`.
* SSD tier (`rmlx-kv-ssd`) sits **between** `rmlx-kv-quant` and `rmlx-models`.
  It depends on `rmlx-kv-quant` (consumes `KvStorage`, `KvCache`,
  `LinearAttnCache`, `KvQuant`) but MUST NOT reach back into `rmlx-models`
  or `rmlx-runtime` — that would re-introduce the cycle the codec extraction removed. The
  arch-specific dispatch (Gemma4 / Qwen3 / Qwen3.5-MoE `attach_ssd_tier`)
  lives in `rmlx_models::ssd_tier` and calls `rmlx_kv_ssd::prepare_attach`
  for the per-namespace SSD work.
* `rmlx-quant` and `rmlx-kv-quant` are **sibling** crates: weight-quant
  codecs (`affine`, `awq`, `bf16`, `fp4`, `fp8`, `mxfp`) stay in `rmlx-quant`
  (`awq` is pure byte-math — AWQ→MLX pack/unpack with no `mlx`/`Array` dep);
  KV-side codecs (`turboquant`, `planarquant`, MSL wrappers, storage,
  `KvCache`, paged, mixed/rot-K) live in `rmlx-kv-quant`. New code MUST
  add KV codecs to `rmlx-kv-quant` and weight codecs to `rmlx-quant`,
  never mix them. `rmlx-quant` does NOT depend on `rmlx-kv-quant` (avoids a
  cycle through `rmlx-loader → rmlx-quant`).

## File-size + inline-test convention

- **Soft 1000 LOC guideline** for source files. Files near or above this should
  be examined for natural split lines, but cohesion trumps line count. Files
  that exceed the limit deliberately should carry a `// LOC-exempt: ...`
  comment at the top explaining why.
- **Hard rule: no inline `#[cfg(test)] mod tests { ... }` blocks** outside
  `tests.rs` / `<name>_tests.rs` files. Extract test bodies to a sibling file
  and reference with:
  ```rust
  #[cfg(test)]
  #[path = "<name>_tests.rs"]
  mod <name>_tests;
  ```
  The CI gate `make check-no-inline-tests` enforces this as a hard-fail step
  in `make ci`. All workspace violations have been migrated.
- **Advisory: `make file-size-report`** prints files >1000 LOC. Non-failing.
  Also runs at the end of `make ci` (advisory, non-blocking).

## Comments and identifiers (hard rule)

Code comments, identifiers, log/error/reason strings must be **general** — never
reference task/issue/PR/review numbers (`// #36 review:`, `// fix for #32`).
Ticket traceability lives in git history, commit messages, and PR descriptions,
not in source. A comment must still read correctly and be useful once the ticket
is gone.

## Simplicity rules (hard)

1. **Readability first.** Match existing style. Plain names. No clever macros, no trait towers, no premature generics.
2. **No over-engineering.** Build what task needs. No speculative abstractions, no single-use traits, no "configurable" knobs that have one caller.
3. **Straight-forward core backend.** Inference path is sequential, sync, explicit. Async only at HTTP/file-I/O boundaries (already in coding style above).
4. **Inline beats premature factoring.** Extract to a function/module only when 2+ real callers exist. Three similar lines is better than a wrong abstraction.
5. **No env-gated one-caller knobs.** Prefer a fixed default or a CLI flag with a real second caller. Env vars are invisible config — each is a support and repro burden. Keep the existing env surface minimal; new env vars need explicit justification (and are an "Ask before" item).

## Common commands (Makefile)

Top-level `Makefile` wraps the dev loop. Prefer it over typing cargo flags by
hand — keeps the CI gate and the local gate identical.

| Target | What it runs |
|---|---|
| `make` / `make help` | List targets. |
| `make build` | `cargo build --workspace --release`. |
| `make check` | `cargo check --workspace --all-targets` (fast). |
| `make test` | `cargo test --workspace`. |
| `make fmt` / `make fmt-check` | Write / check `cargo fmt`. |
| `make lint` | `cargo clippy -D warnings`. |
| `make audit` | `cargo audit` with RustSec ignores from `deny.toml`. |
| `make deny` | `cargo deny --all-features check` (licenses, bans, sources, advisories). |
| `make precommit` | `pre-commit run --all-files`. |
| `make hooks` | Install the git `pre-commit` hook. |
| `make ci` | `fmt-check + lint + test + deny + audit` — pre-merge gate. |
| `make tag` | Create annotated `v<version>` tag from `[workspace.package].version` (single source). |
| `make release-package` | Build + bundle `dist/rmlx-v<ver>-aarch64-apple-darwin.tar.gz` (+ `.sha256`). |
| `make release-sha` | Print sha256 of the `v<ver>` GitHub source tarball (`--write` patches the formula). |
| `make tap-sync` | Copy `packaging/homebrew/rmlx.rb` into the `homebrew-rmlx` tap and push. |
| `make clean` | `cargo clean`. |
| `make serve` | Launch `rmlx serve` on `$(MODEL)` (default = primary test model) at `$(PORT)`. |
| `make chat` | Launch `rmlx chat` REPL on `$(MODEL)`. |
| `make info` | Dump arch + quant info for `$(MODEL)`. |
| `make logs-tail` | `tail -f` newest `logs/*.jsonl`. |
| `make metrics-summary` | `cat metrics/summary.csv`. |
| `make model-check` | `cargo test -p rmlx-{models,runtime,quant}` only — no server/cli/metrics; <30 s, no model needed. |
| `make model-check-full MODEL=…` | Same three crates + golden-token integration tests. Pass one model path; each golden reads `config.json` and skips gracefully when arch does not match — matching arch runs+passes, others skip. Target is green for any single test-target model. |

`MODEL` and `PORT` override at the CLI: `make info MODEL=/path/to/snapshot`.

Run `make ci` before push. The per-commit `pre-commit` hook only runs the
fast checks (fmt, clippy, file hygiene) — `cargo audit` and `cargo deny`
fetch the RustSec advisory DB over the network and were stalling on slow
links, so they are gated behind the `manual` stage. Trigger them via:

- `make ci` (full pre-push gate, recommended).
- `make audit` / `make deny` (individual).
- `pre-commit run --hook-stage manual` (runs the manual hooks).

## Runtime data root: `.rmlx/` (hard rule)

All on-disk state — logs, metrics DB, summary CSVs, ingest buffer, model cache, scratch — lives under a single root, resolved at process start by [`rmlx_core::paths::home()`] in this exact order:

1. `$RMLX_HOME` — absolute path, env-var override. Set this in dev shells (`export RMLX_HOME=$PWD/.rmlx`) or production environments where the canonical location is not `$HOME/.rmlx/`.
2. `<workspace>/.rmlx/` — auto-detected by walking up from cwd for `Cargo.lock`. **This is the dev default.** Co-located with the checkout, gitignored, trivially wiped (`rm -rf .rmlx`).
3. `$HOME/.rmlx/` — installed-binary default. Persists across runs.

Standard sub-tree:

```
.rmlx/
  logs/                 per-run JSON logs (rotated by total-size cap)
  metrics/
    runs.db             SQLite metrics DB (source-of-truth)
    summary.csv         rolling CSV mirror
    backups/            VACUUM INTO snapshots
    buffer/pending/     §8.5 universal-shape ingest queue
    legacy/             archived per-run jsonls (read-only)
  cache/                future model/weight cache
  tmp/                  transient; may be wiped at startup
```

**Hard rules:**

- **Never hard-code `"logs"`, `"metrics"`, or `metrics/runs.db` strings.** Always go through `rmlx_core::paths::*`. CWD-relative paths leak files into `crates/rmlx-cli/` when callers run from a sub-directory.
- **Never write outside `.rmlx/`** at runtime. Prompts (`prompts/`) and registry files are checked-in inputs and stay where they are.

## Debug mode + log retention (hard rule)

Development runs at **info level** by default. Logs accumulate as a runtime-behavior knowledge base and rotate only by total-size cap.

- **Log dir**: `<RMLX_HOME>/logs/` (resolved via `rmlx_core::paths::logs_dir()`).
- **Verbosity flag**: `--log {info|debug|verbose}` (CLI-wide). `info` is the default; `debug` enables per-step phase events; `verbose` enables per-token / per-FFI / per-layer trace events.
- **Per-token / per-layer trace events (e.g. `kv_bytes`) default OFF**; opt in with `--log verbose` or `RUST_LOG=...=trace`. This keeps `tracing` overhead out of steady-state decode.
- **EnvFilter precedence**: `RUST_LOG` (if set) > `--log` preset. `RUST_LOG=debug,rmlx=trace` remains the explicit escape hatch.
- **Run-id**: `YYYYMMDD-HHMMSS-<short-git-sha>` (or `-dirty` for uncommitted state). One file per run: `<run-id>.jsonl`.
- **Total-size rotation**: at startup, oldest files are deleted until the directory total is ≤ `RMLX_LOG_CAP_MB` (default 100 MB). The in-flight log file is never a deletion candidate (rotation runs before the appender opens).
- **Never truncate a single file mid-write.** Rotation always deletes whole `.jsonl` files in mtime order, oldest-first.

## Traceability (hard rule)

`tracing` is the only legitimate runtime-event channel inside engine code. The point of this rule is end-to-end debuggability — being able to reconstruct, from a single run's `.jsonl`, what happened to **every token, every model load, every cache op, every FFI call** that mattered.

- **All runtime events go through `tracing`.** No `eprintln!`, no `println!`, no `log::*` outside of: user-facing CLI output (commands that print to stdout/stderr for the operator), `#[cfg(test)]` diagnostics, and `build.rs` scripts.
- **Every critical path has a span or event.** Required coverage: model load (per-stage), tensor mmap + dequant + warmup, every prefill chunk, every decode step (token id + decision branch), every KV-cache shape change, every cache hit/miss, prompt-cache slot ops, the Metal claim acquire/release, every HTTP request lifecycle (in / out / error), and every FFI error path that could otherwise vanish silently.
- **Structured fields, not string-interp.** Use `tracing::field` attributes (`run_id`, `model`, `kv_quant`, `token_id`, `layer_idx`, …) so log search by exact field is cheap.
- **Levels:**
  - `error!` — unrecoverable / aborts an operation. Includes context.
  - `warn!` — recoverable degradation. Note the workaround.
  - `info!` — start/finish of phases, configuration commits, registry changes.
  - `debug!` — per-step inside a phase (per-layer, per-chunk, per-cache-op).
  - `trace!` — per-token / per-FFI-call / per-tensor. Off by default; opt in with `--log verbose` or `RUST_LOG=...=trace`.
- **`#[tracing::instrument]`** preferred over manual spans where lifetimes align with a function. Keep `skip(...)` for large buffers so they do not bloat the log.

## Metrics retention (hard rule)

Real metrics (load time, tok/s, prefill speed, KV-cache size, memory residency, smoke-probe pass/fail) are collected from every run that touches a model and persisted to `<RMLX_HOME>/metrics/runs.db`. The SQLite DB is the **single source-of-truth** — per-event JSON-Lines files are no longer written.

- **Two tables, one DB**:
  - `observations` — bench-grade run records (one row per measurement), schema migration `001_init.sql`. Written by the §8.5 ingest path (`rmlx metrics record --file <buffer-json>`) and by the in-process `Recorder`.
  - `events` — runtime per-event stream (schema migration `002_events.sql`), written by `rmlx_metrics::events::EventRecorder` (replaces the legacy `rmlx_core::metrics::MetricsSink`). One `INSERT` per `record()` call. WAL absorbs concurrent writers.
- **No more per-event jsonls.** The old `metrics/<run-id>.jsonl` + `summary.csv` writers are gone. Any historical `*.jsonl` under `<RMLX_HOME>/metrics/legacy/` is read-only archive material.
- **Append-only.** Do not delete or overwrite rows; regressions across stages and quants are detected by diffing observations and events over time.
- `rmlx metrics …` subcommands run before any `EventRecorder` opens, so they can target an alternate DB (`--db <path>` or `RMLX_METRICS_DB`) without contending on the workspace lock.

## Metrics database (hard rule)

All bench metrics from any backend land in `metrics/runs.db` (SQLite, gitignored). Three user tables: `prompts`, `observations` (append-only, every measurement), `bests` (VIEW over observations). Schema and operating rules: `docs/METRICS_DB.md`.

- DB is source-of-truth from day-1. Old `metrics/*.jsonl` archived under `metrics/legacy/`, never read or extended.
- New runs: bench script writes `metrics/buffer/pending/<ts>-<uuid>.json` → `rmlx metrics record --file <path>` → recorder ingests + deletes.
- `BENCHMARK_CHAMPIONS.md` regenerated via `rmlx metrics export --markdown`. Never hand-edited.
- Cross-backend recording: every backend (rMLX, mlx_lm, paroquant, omlx, ollama) emits the §8.5 universal JSON shape.
- Prompts owned by `rMLX/prompts/*.json` (content-addressed). CBB symlinks.

Do not add tables, hand-edit `BENCHMARK_CHAMPIONS.md`, or write directly to the DB from non-Rust code. See `docs/METRICS_DB.md` §13 for the full operating rules.

## Regression-bench discipline (hard rule)

Every code-touching change runs a regression smoke before declaring done:
the three test-target families (Gemma4, Qwen3.6, Bonsai) **plus any model
the change touches**, each at its best-known KV quant.

- Decode TPS within ±1% of the recorded best for that model at that KV mode.
- Beat a record at any cell → update `BENCHMARK_CHAMPIONS.md` **and** the report.
- Regress >5% → STOP and report — do not commit.
- Bench rows append to `../Cross-Backend-Bench/metrics/summary.csv`.
- Models out of scope (do not bench, do not optimize): Laguna, DR-Venus.

**Perf tooling (feat/cache-type-flags).** The fast pre-commit smoke is
`bash scripts/perf_canary.sh` — 1 warmup + 3 measured baseline calls per
model (Bonsai, Gemma4-e4b, Qwen3.6), prints decode-only TPS, appends one
CSV row per model to `.rmlx/bench/perf_canary.csv`. Phase 3 anchors (the
committed baseline: Bonsai ~110, Gemma4-e4b ~74, Qwen3.6 ~97 TPS) live in
`docs/PERF_BASELINE.md`. For automated gates use `scripts/regression_gate.sh
<model> <baseline_tps> <baseline_stddev>` — pure awk float math, exit 125 =
`git bisect skip`, exit 1 = regression. Two `Cargo.toml` perf profiles are
in play: `release-perf` (`debug-assertions=false`, `overflow-checks=false`,
stripped debug, `panic=unwind` kept for `MetalClaim::Drop` RAII — see Hard
rule 9) is the canary / bench / `make ci-perf` profile; `release-debug`
(full DWARF, `debug=true`) is the samply flamegraph profile. Build targets:
`make build-perf`, `make build-debug`, `make test-perf`, `make ci-perf`.
The build-by-failure rule is in §Hard rules rule 9 — do not duplicate it here.

## Ask before

- Adding a new dependency to `Cargo.toml`.
- Forking a non-trivial upstream lib.
- Removing a smoke-probe / safety check.
- Bypassing the single-process claim file.
- Deleting or truncating anything under `<RMLX_HOME>/metrics/` (the size-cap log rotation in `<RMLX_HOME>/logs/` is automatic and does not require asking).
- Adding a new environment variable or runtime config knob.

## What "0.1.0 done" looks like

| Capability | Criteria |
|---|---|
| Text | All three test-target families serve OpenAI-compatible text at temp=0 with correct output. |
| Image input | Vision-capable Open Models accept image input and produce coherent output. |
| Audio input | Audio-capable Open Models accept audio input and produce coherent output. |
| Agent | Tool / function-calling multi-turn loop drives a real coding agent end-to-end, zero protocol errors. |
| Quant | Maximum weight × KV quant matrix incl. rotation KV families; smoke-probe green on every snapshot. |
| Convert | `rmlx convert` re-quantizes / repacks an MLX model MLX→MLX. |
