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
   (TurboQuant, IsoQuant, PlanarQuant, RotorQuant). ParoQuant is supported
   too, on the **weight** side — it is not a KV method (see
   `docs/WEIGHT_QUANTS.md` §7).
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
| [`docs/SPEC_ANSWER_EQUIVALENCE.md`](docs/SPEC_ANSWER_EQUIVALENCE.md) | Answer equivalence for speculative decoding: the divergence-confidence oracle, why agreement cannot be thresholded |
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
users set it via `.env`, a shell export, or a `make` command-line variable —
in that order of increasing precedence). At minimum these three families must
serve end-to-end at every change:

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
- `z-lab/paroquant` — weight-side pairwise-rotation INT4 reference. Not a KV
  method: the token `kv` does not occur in the repo, and its calibration path
  drops `use_cache`. See `docs/WEIGHT_QUANTS.md` §7.
- `ParaMind2025/isoquant` — SO(4) isoclinic rotation reference. Stage-1
  quantize/dequantize only (5 tracked files, two CUDA kernels); no cache and no
  decode path upstream, so rMLX's `iso*` KV codecs have no counterpart to port.

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
9. **`make ci-perf` builds + tests under `release-perf` (panic=unwind, debug-assertions off), then runs the GPU/Metal suite.** A failure in the `release-perf` half that doesn't reproduce under `dev` → rebuild under `release-debug` (full DWARF) and re-run the failing case to capture symbols. Never rely on the `dev` profile to reproduce a release-mode bug — codegen and inlining differ. The GPU half is the exception and builds under `dev` on purpose: debug assertions are correctness guards and those are correctness tests. Its consequence: **no gate anywhere executes a `Device::Gpu` test under `release-perf`** — `make test` / `make ci` are `dev` with no `--ignored`, `test-perf` is `release-perf` with no `--ignored`, the GPU suite is `dev` with `--ignored`. A GPU-path defect that appears only with debug-assertions off is therefore out of every gate's scope and must be reproduced by hand at that profile.
10. **Every KV-cache codec ships an MSL (Metal) decode kernel.** A codec whose decode falls back to CPU dequant is not shippable — it strands the codec at single-digit TPS (GPU idle) and is a bug, not a valid mode. New KV codecs (and the decode path of existing ones) MUST decode on-GPU, reading the quant store directly (fused flash-decode-over-quant; see `docs/KV_QUANT.md`, `docs/FFI.md`, and #45). Every MSL kernel body a **production** path can dispatch — KV codec or not — lives in a `.metal` file under a gated `src/metal/` directory (the list is `scripts/metal_dirs.sh`: `rmlx-kv-quant`, `rmlx-models`, `rmlx-mlx`), never in a Rust string literal, and carries a **native-compilation test** (`xcrun -sdk macosx metal -c` at `-std=metal3.0` and `-std=metal4.0`, wired as `make check-metal-compiles` in `make ci`) so MSL syntax errors surface at CI, not on first GPU dispatch. The gate also fails on a `.metal` file its directory's `probes/kernels.manifest` does not name — an unchecked body is the same defect wearing a different hat. Throwaway `#[cfg(test)]` bodies are exempt and stay inline (`metal_kernel_tests.rs` holds a trivial `add_one` smoke and a deliberately-invalid source that must never compile). **Know the gate's boundary:** it keys off directory membership, so it enforces this rule for kernels already in those directories but cannot detect a new inline-MSL literal in a fresh module — that part is review's job, not CI's. Kernels stay **model-agnostic** — keyed off codec + shape (`head_dim`, `kv_heads`, `bits`), never an arch name.

## Coding style

- Workspace `Cargo.toml` with member crates `crates/rmlx-{core,quant,kv-quant,kv-ssd,mlx,loader,metrics,models,runtime,server,cli,audio}`. `rmlx-kv-quant` owns the KV-cache codec layer (storage enums, MSL kernels, per-layer `KvCache`, paged-KV, mixed/rot-K, turbo/planar CPU codecs). `rmlx-kv-ssd` owns the SSD KV tier (index, spill, hydrate, block I/O, layout-key salt, 5 Prometheus hook globals, `SsdHydrate<E>` trait, FNV-1a-64 block-digest helpers); only the per-arch `attach_ssd_tier` dispatcher remains in `rmlx-models::ssd_tier` because the arch-specific `SpillSink<Entry>` / `SsdHydrate<Entry>` impls live in `rmlx-models`. The policy/builder wrappers (`KvCacheBuilder`, `kv_quant_for_layer`, `DEFAULT_KV_QUANT`) stay in `rmlx-models::kv_cache`.
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
- **Hard rule: a test that reaches `Device::Gpu` carries `#[ignore]`.** A shared
  Metal context driven from parallel `cargo test` threads aborts the whole test
  binary ("Rust cannot catch foreign exceptions"), taking every other test in
  the crate with it. **This rule is about Metal only — do not widen it to cover
  MLX contact generally.** The CPU side has its own, unrelated hazard (MLX
  0.31.x fills a process-global command-encoder map without synchronisation, so
  parallel test threads used to SIGSEGV the binary with no failing test named);
  that one is contained by `EVAL_LOCK` / `with_eval_lock` in `rmlx-mlx`, which
  serialises every evaluation process-wide — not by ignoring CPU tests, which
  would only stop running them. Two deterministic gates hold it, and they are
  complementary by construction: `make check-eval-lock` fails the build on any
  MLX eval FFI call made outside the lock (25-symbol reach-set, derived from the
  linked dylibs — **not** just the eval-named ones) but is blind to a lock that
  stopped locking, which `with_eval_lock_serialises_concurrent_callers` catches.
  `make eval-lock-stress` is the probabilistic reproducer, deliberately out of
  `make ci`. See `docs/FFI.md`.
  Run GPU tests with **`make gpu-test`** (every member crate,
  serialized; `CRATE=` / `FILTER=` to narrow), or by hand as
  `cargo test -p <crate> --lib -- --ignored <filter> --test-threads=1`.
  `make gpu-test` is the only step that executes them — `make test` passes no
  `--ignored` and the hosted CI has no Metal. The same suite runs as the last
  step of **`make ci-perf`** (invoked directly, so `CRATE=`/`VALIDATE=` cannot
  narrow or disarm the gate), which is why `ci-perf` now requires an idle GPU
  when it previously did not. It is deliberately not in `make ci`, which would
  then need the Metal context to itself on every commit. A guard
  that only exercises a check the dispatcher rejects **before** touching a
  device-parameterized op is not a GPU test: pass `Device::Cpu` and leave it
  un-ignored — ignoring a CPU test silently stops running it. The CI gate
  `make check-gpu-tests-ignored` enforces this across **every workspace member
  crate** (from `Cargo.toml`), scanning `src/**/{*_tests.rs,tests.rs}` and
  `tests/*.rs`; it keys on shape (does the test reach `Device::Gpu`?), never on
  the ignore reason's text. "Test" covers `#[test]` and `#[tokio::test]` (with
  or without arguments). A pure device-*policy* test (passes `Device::Gpu`
  as a plain value, never dispatches Metal) opts out **per fn** with a
  line-leading `// gpu-test-gate: exempt` marker in its own attribute block —
  scoped to that one `#[test]`, not the whole file; **inside a `macro_rules!`
  body that one `#[test]` is every cell the macro generates**, so audit such a
  marker against all its invocations. The **converse is fatal**: an `#[ignore]`
  whose reason claims a Metal context on a test the classifier can reach no
  `Device::Gpu` from runs under no gate at all — ignored by `make test`,
  unclassified by `make gpu-test` — so it fails until one of three dispositions
  is recorded: declare the route with a line-leading
  `// gpu-test-gate: metal-unscanned` marker (the exact inverse of `exempt` —
  dispatches Metal but never names the device: an in-process HTTP handler, or a
  child process), drop the `#[ignore]` and pass `Device::Cpu`, or reword the
  `#[ignore]` so it does not claim Metal. A declared test is enforced but
  deliberately **not** in `--list` — `run_gpu_tests.sh` asserts a Metal
  validation banner per crate and every declared test is snapshot-gated or drives
  a child, so listing one would fail the suite over a missing model. This one
  check keys on the ignore *text*, which is why a Metal-driving test whose reason
  never says "Metal" or "GPU" stays outside it. A **macro-generated** test is
  enforced at
  its `macro_rules!` body (one body governs every cell it emits), and a body the
  scanner cannot read back — an assembled fn name, a whole macro on one line, an
  item whose brace never closes, an attribute whose bracket never closes — is a
  hard failure rather than a clean scan;
  those cells are deliberately excluded from `--list` / `make gpu-test`, which
  every run prints. A proc-macro-generated test, and a `macro_rules!` with a
  non-brace delimiter, remain outside the fail-closed net — neither exists in
  the tree. `make check-gpu-tests-ignored-fixtures` pins the gate's recall in
  both directions, asserting each case's failure *reason* and not just its exit
  code. See `docs/TESTING.md`.
- **Advisory: `make file-size-report`** prints files >1000 LOC. Non-failing.
  Also runs at the end of `make ci` (advisory, non-blocking).
- **Advisory: `make target-size-report`** prints `target/` size and, past a
  50 GB threshold, a hint to run `make target-gc` (the staleness-based
  pruner, see `scripts/target_gc.sh`). `target/` has no size cap; this just
  makes growth visible. Non-failing. Also runs at the end of `make ci`
  (advisory, non-blocking).

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
| `make test` | `cargo test --workspace` — **skips every `#[ignore]` GPU test**. |
| `make gpu-test` | Run the GPU/Metal `#[ignore]` tests, `--test-threads=1` (`CRATE=` / `FILTER=` narrow). Needs exclusive machine access. Under Metal shader validation (`--nocapture`, so the tests' own skip notices reach the scan): the hits it observes are diffed against `scripts/gpu_validation_census.txt`, which pins one count per originating test; the expectation is the sum over the tests that ran, an exact match passes and prints what it accepted, any deviation fails naming the delta. Part of `make ci-perf`, not `make ci`. |
| `make fmt` / `make fmt-check` | Write / check `cargo fmt`. |
| `make lint` | `cargo clippy -D warnings`. |
| `make audit` | `cargo audit` with RustSec ignores from `deny.toml`. |
| `make deny` | `cargo deny --all-features check` (licenses, bans, sources, advisories). |
| `make precommit` | `pre-commit run --all-files`. |
| `make hooks` | Install the git `pre-commit` hook. |
| `make ci` | `fmt-check + lint + test + deny + audit` — pre-merge gate. |
| `make ci-perf` | `test-perf` under `release-perf` + the serialized GPU/Metal suite. Requires an idle GPU; without the snapshots the pinned entries' tests skip and are reported as not enforced in full rather than failing. Run before merging perf-sensitive or codec-layer changes (~21 min). |
| `make check-kv-layer-quants` | CI gate (in `make ci`): the per-layer KV codec vector has one producer (`kv_layer_quants`) — no second `kv_quant_for_layer` loop, and every per-layer cache stack either uses it or declares itself uniform. |
| `make check-kv-codec-disposition` | CI gate (in `make ci`): the `--kv-quant` / `--kv-bits` help and the `docs/KV_QUANT.md` INERT banners agree with each codec's runtime disposition, derived from `ALL_KV_QUANTS` + `decode_reads_packed_store` / `feeds_bf16_{k,v}_at_decode`. |
| `make check-kv-codec-disposition-fixtures` | CI gate (in `make ci`): recall test for the above, 18 synthetic scan roots (one edit each), each asserting which rule fired and exit 2 vs exit 1. |
| `make check-published-samples` | CI gate (in `make ci` and hosted CI): the checked-in published-protocol sample sets under `prompts/published/` agree with the facts pinned in `scripts/published_samples.py` — seed, revision, template, count, licence, and the digest and byte length of each file — and each sample's user message renders from that template and the sample's own recorded copy of the upstream record. A manifest can be re-blessed around an edited file; the pins in the script cannot, so that is where the anchor is. `--sources` holds the samples to the upstream files record for record, and no gate passes it: the upstream files are not checked in. |
| `make check-published-samples-fixtures` | CI gate (in `make ci` and hosted CI): recall test for the above, 30 synthetic sample-set roots (one edit each), each asserting the reason as well as exit 1 vs exit 2. The same rewritten prompt is run with the script's pin left alone and with it moved, so the layering is exercised rather than asserted. |
| `make spec-bench-published-selftest` | CI gate (in `make ci`): mutation check for `scripts/spec_bench_published.sh` against a stub server over a shrunken copy of the checked-in sample sets, 70 cases, each asserting a literal exit code and, for every refusal, the reason. No GPU, no model, no DB. |
| `make published-ingest-selftest` | CI gate (in `make ci`): mutation check for `scripts/ingest/published_ingest.py`, 24 cases over one synthetic published-protocol result on the real checked-in sample sets. One edit each, asserting the literal exit code and the reason: both ways a sample's content address can move, a sample set edited after the run, a rebuilt binary, a binary whose digest agrees but whose marker literals do not, each refusal and its waiver, and the buffer-queue discipline in all three outcomes. Never writes `runs.db`. |
| `make check-kv-byte-model-parity` | CI gate (in `make ci`): `scripts/perf_ceiling.py`'s KV byte model against the engine's, swept from `ALL_KV_QUANTS` across both topologies and two shapes. The engine is the oracle — this does not check either model is right, only that there is effectively one of them. |
| `make check-kv-byte-model-parity-fixtures` | CI gate (in `make ci`): recall test for the above, synthetic scan roots asserting the reason as well as exit 2 vs exit 1. |
| `make gpu-runner-selftest` | CI gate (in `make ci` and hosted CI): the GPU runner reports a shader-validation hit and a crate failure found in the same run, the access mix it prints is the one the diagnostics named, and every census-pin verdict — exact match, new kernel, count above or below the expectation, a silent entry, a hit that moved crate, any store, an entry whose test was not selected or skipped, and each way the pin file itself can be malformed — reaches the report as itself, the tracked pin included. Stub crates, no GPU. |
| `make canary-ab-selftest` | CI gate (in `make ci`): mutation check for `scripts/perf_ab.sh` against stub binaries. Every case declares `--synthetic-arms`, so the machine is not consulted and the outcome cannot depend on host load; the cases that exercise the host gates supply `ps` and `pgrep` shims instead. |
| `make canary-ab-host-gate-fixtures` | CI gate (in `make ci`): recall test for that boundary — the quiescence and Metal-exclusivity gates still fire on a shimmed hostile host, a hostile and a quiet host give `--synthetic-arms` the same verdict, the flag waives no arm-reading guard, and the result file carries no reading taken off this machine. |
| `make llama-ab-selftest` | CI gate (in `make ci`): mutation check for `scripts/bench_llama_ab.sh` against a stub `llama-server`. Same `--synthetic-arms` boundary, shared through `scripts/lib/cpu_snapshot.sh`; every case asserts a literal exit code, and the count of cases that could reach this host must be zero. |
| `make check-kv-boundary-default-parity` | CI gate (in `make ci`): the CLI help, `docs/CLI.md` and the ingest resolver all name the default KV boundary the engine applies. The Rust constants are the oracle; the Python side derives rather than restates. |
| `make check-doc-source-citations` | CI gate (in `make ci`): every `crates/...` path cited in `docs/` resolves. Covers paths only, not identifiers — telling a test name from a variable needs a parser. |
| `make check-eval-lock` | CI gate (in `make ci`): every MLX eval FFI call is made under the process-wide evaluation lock (25-symbol reach-set). |
| `make check-eval-lock-fixtures` | CI gate (in `make ci`): recall test for the above, 26 synthetic scan roots, each asserting which rule fired. |
| `make eval-lock-stress` | Drive the evaluation-lock reproducer across `RUNS` fresh processes (default 60). Not in `make ci` — probabilistic (~8%/run) and costs ~412 threads. |
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
| `make model-check` | `cargo test -p rmlx-{models,runtime,quant,kv-quant}` only — no server/cli/metrics; <30 s, no model needed. |
| `make model-check-full MODEL=…` | `cargo test -p rmlx-{models,runtime,quant}` (note: **not** `rmlx-kv-quant`) + golden-token integration tests. Pass one model path; each golden reads `config.json` and skips gracefully when arch does not match — matching arch runs+passes, others skip. Target is green for any single test-target model. |

`MODEL` and `PORT` override at the CLI: `make info MODEL=/path/to/snapshot`.

Run `make ci` before push, plus `make ci-perf` when the change touches
`rmlx-kv-quant`, a `.metal` kernel, or a KV/decode path — `make ci` runs no GPU
test. The per-commit `pre-commit` hook only runs the
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
- **Per-token / per-layer trace events (e.g. per-token `token_id`, per-FFI dispatch) default OFF**; opt in with `--log verbose` or `RUST_LOG=...=trace`. This keeps `tracing` overhead out of steady-state decode. Level gating hides the *emission* cost, not the cost of *computing* a field: an event whose field is an O(seq) call still makes decode quadratic once the level is enabled. Compute such fields at request boundaries and emit them there (`kv_bytes` is one — it is a per-request `debug!`, not a per-layer `trace!`).
- **EnvFilter precedence**: `RUST_LOG` (if set) > `--log` preset. `RUST_LOG=debug,rmlx=trace` remains the explicit escape hatch.
- **Run-id**: `YYYYMMDD-HHMMSS-<version>`. One file per run: `<run-id>.jsonl`. (The binary does no git of any kind — see `docs/METRICS_DB.md` §8.5.1 — so the discriminator is the backend semver, not a commit SHA.)
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

- **Correctness first, on ≥2 architectures.** Any code-touching change proves
  correct output on real models spanning ≥2 archs — minimum **gemma4-e2b**
  (single-KV-head `kv_h == 1`, shared-KV) **+ Ternary-Bonsai-8B** (`kv_h > 1`,
  dense). A KV/kernel fix that holds at one arch/shape can silently fail at
  another (`kv_h == 1` vs `kv_h > 1`, power-of-two vs non-power-of-two
  `head_dim`). Unit tests alone are not proof; serve the model.
- Decode TPS within ±1% of the recorded best for that model at that KV mode.
- Beat a record at any cell → update `BENCHMARK_CHAMPIONS.md` **and** the report.
- Regress >5% → STOP and report — do not commit.
- Bench rows append to `../Cross-Backend-Bench/metrics/summary.csv`.
- Models out of scope (do not bench, do not optimize): Laguna, DR-Venus.

**Perf tooling (feat/cache-type-flags).** The fast pre-commit smoke is
`bash scripts/perf_canary.sh` — 1 warmup + 3 measured baseline calls per
model (Bonsai, Gemma4-e4b, Qwen3.6), prints decode-only TPS, appends one
CSV row per model to `.rmlx/bench/perf_canary.csv`. Anchors live in
`docs/PERF_BASELINE.md`; the current ones are at the bf16 `auto` default
(Bonsai ~142, Gemma4-e4b ~80, Qwen3.6 ~101 TPS). The older Phase-3 row set
(Bonsai ~110, Gemma4-e4b ~74, Qwen3.6 ~97) was recorded at the retired per-arch
codec defaults and is not comparable to it. For automated gates use
`scripts/regression_gate.sh <model> <baseline_tps> <baseline_stddev>` — pure
awk float math, exit 125 =
`git bisect skip`, exit 1 = regression. Two `Cargo.toml` perf profiles are
in play: `release-perf` (`debug-assertions=false`, `overflow-checks=false`,
stripped debug, `panic=unwind` kept for `MetalClaim::Drop` RAII — see Hard
rule 9) is the canary / bench profile and the profile of `make ci-perf`'s
`test-perf` half — its GPU half runs under `dev`, see Hard rule 9; `release-debug`
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
