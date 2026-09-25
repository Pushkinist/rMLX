# rMLX — agent guide

Rust-native, single-binary MLX inference backend for Apple Silicon.
MLX→MLX conversion is in scope, not yet implemented. Goal: the fastest fully-featured **native, no-Python** backend for
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
   repack) — MLX in, MLX out. A 0.1.0 target, not yet implemented: there is
   no `rmlx convert` today.
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
| [`docs/WEIGHT_QUANTS.md`](docs/WEIGHT_QUANTS.md) | Weight quantization formats (bf16, MXFP, affine, ParoQuant, ternary BitLinear), where they are decoded, adding a format |
| [`docs/KV_QUANT.md`](docs/KV_QUANT.md) | KV quantization contract: CLI flags and presets, the auto default, memory and bit rates, KV byte accounting, per-request hot-swap, storage summary, Metal-vs-CPU hot path, the break-even condition and each codec's disposition, public API and import paths |
| [`docs/KV_LAYER_POLICY.md`](docs/KV_LAYER_POLICY.md) | Which KV codec each layer gets: per-layer net-benefit decision, bf16 at `--kv-quant none`, layer-adaptive overrides, Qwen MoE low-bit K rejection |
| [`docs/KV_CODECS.md`](docs/KV_CODECS.md) | Storage and decode path per `KvStorage` variant (None, K8V8, K8V4, Planar, Mixed, rot_k, turbo, PlanarK, Paged); TurboQuant calibration (`kv_calib.json`) |
| [`docs/KV_ROTATION_CODECS.md`](docs/KV_ROTATION_CODECS.md) | The iso and rotor KV codecs and their K-side variants |
| [`docs/KV_FUSED_KERNELS.md`](docs/KV_FUSED_KERNELS.md) | Fused KV decode kernels: fused-QK, fused flash-decode over rotor / iso / PlanarK, fused-QK head-major K storage, the dispatch axis, sparse attention |
| [`docs/KV_STORE_TRUNCATION.md`](docs/KV_STORE_TRUNCATION.md) | `KvCache::truncate_to` per KV store: the shared planner, block splits, refusals, clamping |
| [`docs/KV_CODEC_FIDELITY.md`](docs/KV_CODEC_FIDELITY.md) | Measured KV codec fidelity: incoherence, the turbo family's missing rotation, rate-distortion |
| [`docs/KV_CACHE.md`](docs/KV_CACHE.md) | KV cache architecture (block alignment, ring buffer, SWA snapshot, chunked prefill) |
| [`docs/KV_UPDATE_PATH.md`](docs/KV_UPDATE_PATH.md) | KV update path: per-family update files, one body per store shape, width-generic stores and their guard, what stays per width, the store-bytes oracle and what it cannot see, what cannot move |
| [`docs/SSD_TIER.md`](docs/SSD_TIER.md) | SSD KV tier (layout_key, ssd_index schema, hydrate, spill, cross-namespace LRU) |
| [`docs/SSD_CANARY.md`](docs/SSD_CANARY.md) | SSD KV cross-restart smoke probe |
| [`docs/PROMPT_CACHE.md`](docs/PROMPT_CACHE.md) | Prompt cache + automatic prefix caching (block hashing, ReusePolicy, prefix index) |
| [`docs/SPECULATIVE.md`](docs/SPECULATIVE.md) | Speculative decoding: the drafters (MTP, DFlash, EAGLE-3, two full models), the round loop, reading a run, judging a draft-side change, CLI |
| [`docs/SPEC_ANSWER_EQUIVALENCE.md`](docs/SPEC_ANSWER_EQUIVALENCE.md) | Answer equivalence for speculative decoding: the divergence-confidence oracle, why agreement cannot be thresholded |
| [`docs/SPEC_ROUND_SKELETON.md`](docs/SPEC_ROUND_SKELETON.md) | The speculative round loop `run_rounds` and the `RoundDrafter` interface: what each drafter declares, what differs per drafter, the two two-model entries kept apart, the oracle, what no runtime check sees |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Per-token sampling (temperature, top-k/p, penalties, thinking budget, constrained decoding) |
| [`docs/FFI.md`](docs/FFI.md) | rmlx-mlx ↔ mlx-c FFI bridge; MSL kernel surface; unsafe policy |
| [`docs/METRICS_DB.md`](docs/METRICS_DB.md) | Metrics DB: schema (observations, events, prompts, the bests view), metric registry, identity rules, ingest, `rmlx metrics` tooling, operating rules |
| [`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md) | The three canary anchors and the bench methods: A/B comparison, per-codec cells, the bandwidth ceiling, cross-backend cells |
| [`docs/PUBLISHED_PROTOCOL.md`](docs/PUBLISHED_PROTOCOL.md) | Generated: published-protocol results (MT-Bench / MATH-500 / HumanEval, fixed prompt) with every number beside the bound it cannot pass |
| [`docs/PROFILING.md`](docs/PROFILING.md) | Profiling runbook: samply, Instruments, Metal GPU capture, memory counters, the prefill-chunk knob |
| [`docs/PROJECTS_CONFIG.md`](docs/PROJECTS_CONFIG.md) | Per-project cap defaults via `<RMLX_HOME>/projects.toml` |
| [`docs/TESTING.md`](docs/TESTING.md) | Test snapshot resolution (RMLX_O_MODELS_ROOT, RMLX_TEST_MODEL_*), test variables, golden-token fixtures, CPU numeric gates, NIAH and codec smoke matrix |
| [`docs/GPU_TESTS.md`](docs/GPU_TESTS.md) | GPU/Metal tests: the `#[ignore]` rule and its gate, `make gpu-test`, stand-downs, halves, `make ci-perf`, shader validation, census pin |
| [`docs/AUDIO.md`](docs/AUDIO.md) | Audio: Whisper transcription, Qwen3-TTS, WAV I/O, Whisper token ids |
| [`docs/E2E_TEST_PLAN.md`](docs/E2E_TEST_PLAN.md) | End-to-end feature-proof harness: modality, tool-calling and speculative-decoding cases |
| [`docs/RELEASING.md`](docs/RELEASING.md) | Release flow: single-source version, `make tag` / `release-package` / `tap-sync`, Homebrew formula + tap, `CHANGELOG.md`, branch model (`next/*`, hotfix, fast-forward release) |

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

Under `RMLX_O_MODELS_ROOT`, set via `.env`, a shell export or a `make`
command-line variable, in that order of increasing precedence. Unset, the
`Makefile` falls back to `models/` in the repo. At minimum these three families
must serve end-to-end at every change:

| Family | Example snapshot | Arch |
|---|---|---|
| Gemma4 | `mlx-community__gemma-4-e4b-it-mxfp8`, `mlx-community__gemma-4-26b-a4b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| Qwen3.6 | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| Bonsai | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |

Other Open Models snapshots (`z-lab__Qwen3.6-27B-PARO`, `medgemma`, the `jina`
embedding/reranker models, `ReaderLM-v2`, …) are in scope as feature
coverage grows.

## Key external repos (GitHub)

- `oxiglade/mlx-rs` — community Rust binding over `mlx-c`.
- `ml-explore/mlx-c` — Apple's stable C ABI.
- `safetensors/safetensors` — Rust safetensors crate.
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
3. **MLX-format only**. GGUF is out of scope. MLX↔MLX re-quantize / convert
   is in scope, not yet implemented; rMLX never reads GGUF.
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
10. **Every KV-cache codec ships an MSL (Metal) decode kernel.** A codec whose decode falls back to CPU dequant is not shippable — it strands the codec at single-digit TPS (GPU idle) and is a bug, not a valid mode. New KV codecs (and the decode path of existing ones) MUST decode on-GPU, reading the quant store directly (fused flash-decode-over-quant; see `docs/KV_FUSED_KERNELS.md` and `docs/FFI.md`). Every MSL kernel body a **production** path can dispatch — KV codec or not — lives in a `.metal` file under a gated `src/metal/` directory (the list is `scripts/metal_dirs.sh`: `rmlx-kv-quant`, `rmlx-models`, `rmlx-mlx`), never in a Rust string literal, and carries a **native-compilation test** (`xcrun -sdk macosx metal -c` at `-std=metal3.1` and `-std=metal4.0`, wired as `make check-metal-compiles` in `make ci`) so MSL syntax errors surface at CI, not on first GPU dispatch. The gate also fails on a `.metal` file its directory's `probes/kernels.manifest` does not name — an unchecked body is the same defect wearing a different hat. Throwaway `#[cfg(test)]` bodies are exempt and stay inline (`metal_kernel_tests.rs` holds a trivial `add_one` smoke and a deliberately-invalid source that must never compile). **Know the gate's boundary:** it keys off directory membership, so it enforces this rule for kernels already in those directories but cannot detect a new inline-MSL literal in a fresh module — that part is review's job, not CI's. Kernels stay **model-agnostic** — keyed off codec + shape (`head_dim`, `kv_heads`, `bits`), never an arch name.

## Coding style

- Workspace `Cargo.toml` with member crates `crates/rmlx-{core,quant,kv-quant,kv-ssd,mlx,loader,metrics,models,runtime,server,cli,audio}`. `rmlx-kv-quant` owns the KV-cache codec layer (storage enums, MSL kernels, per-layer `KvCache`, paged-KV, mixed/rot-K, turbo/planar CPU codecs). `rmlx-kv-ssd` owns the SSD KV tier (index, spill, hydrate, block I/O, layout-key salt, the five hook globals, the `SsdHydrate<E>` and `HydratedEntry` traits with the one blanket `impl<E: HydratedEntry> SsdHydrate<E> for SsdHydrator` that joins them, FNV-1a-64 block-digest helpers). `rmlx-models` keeps the per-arch dispatch (`ssd_tier::attach_at_load`), the one blanket `impl<E: PromptCacheEntry> SpillSink<E> for SsdSpiller` in `prompt_cache.rs` and the per-arch `HydratedEntry` impls. The blanket hydrate impl cannot live there: both the trait and `Self` are foreign to `rmlx-models` and the type parameter is uncovered, so the orphan rule rejects it. The policy/builder wrappers (`KvCacheBuilder`, `kv_quant_for_layer`, `DEFAULT_KV_QUANT`) stay in `rmlx-models::kv_cache`.
- `thiserror` for library errors, `anyhow` for binary entry-point.
- `tracing` for logging, not `log` or `eprintln`.
- Async only at boundaries (HTTP server, file I/O). Compute is sync.
- Tests in sibling `*_tests.rs` files; see "File-size + inline-test convention" below. Integration tests under `tests/`.
- No unsafe outside `rmlx-core` FFI module unless heavily justified + reviewed.
- Public API surface conservative — no leaking mlx-rs types directly.

## Workspace dep graph

Member-crate edges. `→` means "depends on".

```
rmlx-core    (root — no internal deps)
rmlx-mlx     → rmlx-core, rmlx-loader
rmlx-quant   → rmlx-core
rmlx-loader  → rmlx-core, rmlx-quant
rmlx-kv-quant → rmlx-core, rmlx-mlx
rmlx-metrics → rmlx-core
rmlx-kv-ssd  → rmlx-core, rmlx-mlx, rmlx-kv-quant, rmlx-metrics
rmlx-runtime → rmlx-core, rmlx-mlx
rmlx-models  → rmlx-core, rmlx-mlx, rmlx-quant, rmlx-kv-quant, rmlx-kv-ssd, rmlx-loader, rmlx-runtime, rmlx-metrics
rmlx-audio   → rmlx-core, rmlx-mlx, rmlx-loader
rmlx-server  → rmlx-core, rmlx-mlx, rmlx-kv-quant, rmlx-kv-ssd, rmlx-loader, rmlx-metrics, rmlx-models, rmlx-audio
rmlx-cli     → every crate above except rmlx-runtime
```

`rmlx-server` and `rmlx-cli` import codec items from `rmlx_kv_quant` and
SSD-tier items from `rmlx_kv_ssd` directly; `rmlx_models::kv_cache` re-exports
neither.

Hard rules:

* Codec layer (`rmlx-kv-quant`) must remain a leaf of `rmlx-models` — never
  reach into `rmlx-models` or `rmlx-runtime`. Higher-level policy stays in
  `rmlx-models::kv_cache`.
* SSD tier (`rmlx-kv-ssd`) sits **between** `rmlx-kv-quant` and `rmlx-models`.
  It depends on `rmlx-kv-quant` (consumes `KvStorage`, `KvCache`,
  `LinearAttnCache`, `KvQuant`) but MUST NOT reach back into `rmlx-models`
  or `rmlx-runtime`, which would make a dependency cycle. The per-arch
  dispatch (`attach_at_load`, one arm per architecture with a prompt cache)
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
  in `make ci`.
- **Hard rule: a test that reaches `Device::Gpu` carries `#[ignore]`.** A shared
  Metal context driven from parallel `cargo test` threads aborts the whole test
  binary ("Rust cannot catch foreign exceptions"), taking every other test in
  the crate with it. **This rule is about Metal only — do not widen it to cover
  MLX contact generally.** The CPU side has its own, unrelated hazard (MLX
  0.31.x fills a process-global command-encoder map without synchronisation, so
  unserialised parallel test threads SIGSEGV the binary with no failing test
  named);
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
  narrow or disarm the gate), which is why `ci-perf` requires an idle GPU.
  It is deliberately not in `make ci`, which would then need the Metal context
  to itself on every commit. A guard
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
  code. See `docs/GPU_TESTS.md`.
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
6. **No twins, and every change names its removals.** Two types, functions, kernels, or files whose bodies differ only in a compile-time constant (`bits`, `head_dim`, group size, a codebook) or in a component's name are one item and a parameter, not two — a const-generic or a trait-bound blanket impl, never a copy per caller (rule 1's "no premature generics" still governs the single-instantiation case). One near-twin is kept apart on purpose: the two two-model entries, which the seven-path census holds as two paths (`docs/SPEC_ROUND_SKELETON.md` § "The two two-model entries"). A change that adds an item without deleting the twin it replaces is not done — every PR description says what it deletes, and "nothing" is a valid answer only if it is written down.

## Common commands (Makefile)

Top-level `Makefile` wraps the dev loop. Prefer it over typing cargo flags by
hand — keeps the CI gate and the local gate identical.

| Target | What it runs |
|---|---|
| `make` / `make help` | List targets. |
| `make build` | `cargo build --workspace --release`. |
| `make check` | `cargo check --workspace --all-targets` (fast). |
| `make test` | `cargo test --workspace` — **skips every `#[ignore]` GPU test**. |
| `make gpu-test` | Run the GPU/Metal `#[ignore]` tests, `--test-threads=1` (`CRATE=` / `FILTER=` narrow). Needs exclusive machine access. Under Metal shader validation (`--nocapture`, so the tests' own skip notices reach the scan): the hits it observes are diffed against `scripts/gpu_validation_census.txt`, which pins one count per (kernel, kind, crate, originating test) — a test that loads two checkpoints can carry two entries; the expectation is the sum over the tests that ran, an exact match passes and prints what it accepted, any deviation fails naming the delta. A cell that stood down is listed with the reason its `SKIP <test>: <why>` notice gave; one whose notice omitted the test name is counted but not attributable; either way the final line reads INCOMPLETE — a test that could not run is not a test that passed, and libtest reports both as `ok`. A test that notes a missing checkpoint and still asserts on the ones it found prints a `note`, not a `SKIP`, so its entries stay expected. A change that adds a GPU test, or makes one perform a load it did not perform before, derives its own pin entry in the same change (`docs/GPU_TESTS.md`). `HALF=codec\|rest` runs one side of the partition `scripts/gpu_test_halves.sh` computes — that producer is the only place the partition is computed, and a classified test it places in no half is a refusal naming the test, not a note. A half's census expectation is its own slice of the one pin, keyed on (crate, test); a stand-down inside a half still ends it INCOMPLETE; the final line names the half. Part of `make ci-perf`, not `make ci`. |
| `make fmt` / `make fmt-check` | Write / check `cargo fmt`. |
| `make lint` | `cargo clippy -D warnings`. |
| `make audit` | `cargo audit` with RustSec ignores from `deny.toml`. |
| `make deny` | `cargo deny --all-features check` (licenses, bans, sources, advisories). |
| `make precommit` | `pre-commit run --all-files`. |
| `make hooks` | Install the git `pre-commit` hook. |
| `make ci` | Pre-merge gate. The `Makefile` `ci:` recipe is the full list: `fmt-check`, `lint`, `test`, `test-capture`, `deny`, `audit`, `ci-metrics`, then the CI gates, then the advisory reports. Not every gate in it has a row here. |
| `make ci-perf` | `test-perf` under `release-perf` + the serialized GPU/Metal suite. Requires an idle GPU. The run ends `ci-perf INCOMPLETE` rather than `ci-perf ok` whenever a selected GPU test did not run — a missing snapshot, or a cell gated on a variable the host did not set (`RMLX_KV_TEST_MODEL`, `RMLX_VL_TEST_MODEL`, `RMLX_PROMPT_CACHE_TEST_MODEL_*`; see `docs/GPU_TESTS.md`) — or a stand-down notice named no test; pinned entries it could not count are reported as not enforced in full rather than failing. Run before merging perf-sensitive or codec-layer changes. A PR runs `make ci-perf HALF=codec` when its diff touches only crates below the model layer, only `crates/rmlx-models/src/`, or only integration binaries that select a KV codec; `HALF=rest` when it touches only the binaries that do not. Anything else, and every merge to `main`, runs the whole `make ci-perf`. `HALF` must be exactly `codec` or `rest` — an empty one is refused, not waved through — and an accidentally exported one is ignored (`$(origin HALF)`), which closes the stale-export case and nothing more. The defence that does hold is the last line: a half-run reads `ci-perf <half>-half ok — NOT the whole gate` and never the string `ci-perf ok`. What a half cannot catch is a codec change that passes the codec half and breaks a golden or an equivalence pair in the rest half — `main` runs the whole gate, so that is found one merge later than it would have been. |
| `make check-kv-layer-quants` | CI gate (in `make ci`): the per-layer KV codec vector has one producer (`kv_layer_quants`) — no second `kv_quant_for_layer` loop, and every per-layer cache stack either uses it or declares itself uniform. |
| `make check-kv-codec-disposition` | CI gate (in `make ci`): the `--kv-quant` / `--kv-bits` help and the INERT banners agree with each codec's runtime disposition, derived from `ALL_KV_QUANTS` + `decode_reads_packed_store` / `feeds_bf16_{k,v}_at_decode`. The banners are read from a fixed list of docs (`BANNER_DOCS`); a banner in any other doc fails, a listed doc that is missing, an empty list and a doc listed twice are exit 2, and each inert codec is named in exactly one banner. |
| `make check-kv-codec-disposition-fixtures` | CI gate (in `make ci`): recall test for the above, 25 synthetic scan roots (one edit each), each asserting which rule fired and exit 2 vs exit 1. |
| `make check-published-samples` | CI gate (in `make ci` and hosted CI): the checked-in published-protocol sample sets under `prompts/published/` agree with the facts pinned in `scripts/published_samples.py` — seed, revision, template, count, licence, and the digest and byte length of each file — and each sample's user message renders from that template and the sample's own recorded copy of the upstream record. A manifest can be re-blessed around an edited file; the pins in the script cannot, so that is where the anchor is. `--sources` holds the samples to the upstream files record for record, and no gate passes it: the upstream files are not checked in. |
| `make check-published-samples-fixtures` | CI gate (in `make ci` and hosted CI): recall test for the above, 30 synthetic sample-set roots (one edit each), each asserting the reason as well as exit 1 vs exit 2. The same rewritten prompt is run with the script's pin left alone and with it moved, so the layering is exercised rather than asserted. |
| `make spec-bench-published-selftest` | CI gate (in `make ci`): mutation check for `scripts/spec_bench_published.sh` against a stub server over a shrunken copy of the checked-in sample sets, 71 cases, each asserting a literal exit code and, for every refusal, the reason. No GPU, no model, no DB. |
| `make published-ingest-selftest` | CI gate (in `make ci`): mutation check for `scripts/ingest/published_ingest.py`, 25 cases over one synthetic published-protocol result on the real checked-in sample sets. One edit each, asserting the literal exit code and the reason: both ways a sample's content address can move, a sample set edited after the run, a rebuilt binary, a binary whose digest agrees but whose marker literals do not, each refusal and its waiver, and the buffer-queue discipline in all three outcomes. Never writes `runs.db`. |
| `make published-table` | Regenerate `docs/PUBLISHED_PROTOCOL.md` from the published-protocol result files. Every measured figure is rendered beside the bound it cannot exceed — the decode rate against `scripts/perf_ceiling.py`'s autoregressive ceiling, the resident peaks against the weight+KV floor, `tokens_per_round` against the drafter's block — and the pinned protocol choices and the three disclosures are printed in the header. Defaults to the checked-in fixture; `PUBLISHED_TABLE_RESULTS=` / `PUBLISHED_TABLE_MODEL=` render a real run. |
| `make check-published-table` | CI gate (in `make ci`): the committed `docs/PUBLISHED_PROTOCOL.md` is exactly what the emitter renders from its checked-in inputs — it is generated, never hand-written. It reads the fixture directly, so no variable can point it elsewhere; regenerating the doc from a real run without checking that run's result files in beside the fixture turns this red, which is how a published number with unrecorded inputs is found. |
| `make published-table-selftest` | CI gate (in `make ci`): mutation check for `scripts/lib/published_table.py`, 35 cases against a header-only fixture snapshot. The numeric cases re-derive the ceiling and the resident floor from that snapshot's own safetensors header in arithmetic sharing no code with `perf_ceiling.py`; the codec and context ceiling are checked to be passed through, since a wrong bound renders exactly as plausibly as a right one. No GPU, no model, no DB. |
| `make check-kv-byte-model-parity` | CI gate (in `make ci`): `scripts/perf_ceiling.py`'s KV byte model against the engine's, swept from `ALL_KV_QUANTS` across both topologies and two shapes. The engine is the oracle — this does not check either model is right, only that there is effectively one of them. |
| `make check-kv-byte-model-parity-fixtures` | CI gate (in `make ci`): recall test for the above, synthetic scan roots asserting the reason as well as exit 2 vs exit 1. |
| `make gpu-runner-selftest` | CI gate (in `make ci` and hosted CI): the GPU runner reports a shader-validation hit and a crate failure found in the same run, the access mix it prints is the one the diagnostics named, and every census-pin verdict — exact match, new kernel, count above or below the expectation, a silent entry, a hit that moved crate, any store, an entry whose test was not selected or skipped, and each way the pin file itself can be malformed — reaches the report as itself, the tracked pin included. Stub crates, no GPU. |
| `make canary-ab-selftest` | CI gate (in `make ci`): mutation check for `scripts/perf_ab.sh` against stub binaries. Every case declares `--synthetic-arms`, so the machine is not consulted and the outcome cannot depend on host load; the cases that exercise the host gates supply `ps` and `pgrep` shims instead. |
| `make canary-ab-host-gate-fixtures` | CI gate (in `make ci`): recall test for that boundary — the quiescence and Metal-exclusivity gates still fire on a shimmed hostile host, a hostile and a quiet host give `--synthetic-arms` the same verdict, the flag waives no arm-reading guard, and the result file carries no reading taken off this machine. |
| `make llama-ab-selftest` | CI gate (in `make ci`): mutation check for `scripts/bench_llama_ab.sh` against a stub `llama-server`. Same `--synthetic-arms` boundary, shared through `scripts/lib/cpu_snapshot.sh`; every case asserts a literal exit code, and the count of cases that could reach this host must be zero. |
| `make check-kv-boundary-default-parity` | CI gate (in `make ci`): the CLI help, `docs/CLI.md` and the ingest resolver all name the default KV boundary the engine applies. The Rust constants are the oracle; the Python side derives rather than restates. |
| `make check-spec-sampling` | CI gate (in `make ci`): every speculative generation path is handed the request's sampler and draws with it, and every drafter arm of the server's dispatch passes it. The population is `make check-spec-charge`'s, at any visibility — the round loops, the entries that build a loop's `RoundCfg`, and the fns that call one of them, which is the two-model entry guard that routes a request to one of two entries by reading whether the sampler is active. A loop must construct its draw from the sampler — `VerifierDraw::new(sampler_cfg)`, followed to its closing parenthesis so a wrapped argument list reads like a single-line one, and not a mention of the name; an entry must carry it inside the configuration it hands over; a guard must pass it at every route that takes one. The gate names no exception. Every needle reads a line's code with the body of its string literals blanked, through the one reader in `scripts/lib/awk_text.sh`: a commented-out draw beside a greedy one is a loop that decodes greedily and a scan that says it does not. A loop that takes no sampler decodes greedily, does not warn, and returns fluent text, so the caller's temperature is lost with nothing anywhere saying so. This gate cannot tell a right distribution from a wrong one; `crates/rmlx-models/tests/spec_sampled_distribution.rs` does that, on one pair under `make gpu-test`. A scan matching no loop, and a drafter-path count that is not the pinned seven — a migrated drafter's loop body becomes its entry, one for one — is exit 2, not a pass. |
| `make check-spec-sampling-fixtures` | CI gate (in `make ci`): recall test for the above, 30 cases over four tree shapes — seven loop bodies, one where a migrated drafter's path is an entry, one where the two-model greedy path is an entry beside a migrated sidecar with the other four still bodies, and the tree's own census of one shared loop and seven entries — each asserting the reason as well as exit 1 vs exit 2. A loop that takes no sampler, one that takes it and builds no draw from it while still naming it, a guard that drops it on one of its two routes, an entry that leaves it out of the configuration it hands over, an entry that never took one, the shared loop drawing from a greedy default, one dispatch arm dropping it, an eighth path and a lost one — both the census, not the sampler rule — a commented-out draw, a commented-out parameter, a needle inside a string literal, and a dispatch arm that keeps the sampler in a comment or names it inside a literal, a trait of bodiless declarations above the loop that draws — where the drafter-path count is blind, the lost loop's slot being the one the new entry fills — an entry with no forwarded loop to enter, a draw wrapped over two lines and a comment in a parameter list, the last two of which must pass, the greedy entry held to both of the entry rule's conditions — its sampler parameter stripped, and the sampler left out of the configuration it hands over — and a loop that seeds a generator of its own from the request's seed instead of constructing the round's draw. No case edits the gate itself. |
| `make check-spec-charge` | CI gate (in `make ci`): each speculative round loop names one phase-charge decision and names it everywhere the decision is read — the `charge` argument of every `rollback_round` call inside the loop and the `charged:` field of every `RoundReport` and of the `RoundTotals` it hands the one recorder are the same token, and no fn outside the two populations that may name it — the round loops and the entries — writes the field — fn bodies only, so a decision parked in a module-level constant is outside it; the recorder carries the value across as a destructured `charged,`, and `crates/rmlx-models/src/speculative/round_common_tests.rs` is what reads that it arrives. The three populations are derived and none is a name list: the round loops (the driver signature plus a `RoundTotals`, split **by signature** into the classic ones and a forwarded one whose parameter list carries `RoundCfg`, the configuration type that holds the charge field), the entries (the signature, no totals, no rollback, and it constructs `RoundCfg` — the conjunct that keeps the entry guard and the emit helpers out), and the drafter rollbacks (a `rollback_round` outside a loop, whose `charge` is the round context's `charged` where the fn is handed one, and a field of one of its own parameters otherwise). A forwarded loop binds its token to exactly `<cfg>.charged` or names that field at every site, and the configuration it was handed is read-only inside it — structurally, since it is handed over by `&RoundCfg` and a loop taking it by `&mut` or by value is exit 2: the spellings for moving a value are unbounded, the parameter is one thing. The census is over every charge token that is not that field — the classic loops' and the entries' — and is exactly `charge_phases` three times and `false` four: a forwarded loop naming the field it was handed contributes nothing, and the same loop hard-wiring a literal is caught twice, by the rule that names the line and by the census that reads eight sites. Bindings are read per binding, not per fn: a token bound twice is exit 1 for a forwarded loop, where there is something exact to say about the second binding, and exit 2 for a classic loop and for an entry, where nothing in the scan can say which binding governs which site. A loop that charges its phases and records that it did not re-attributes its own work to the drafter with every token and every count identical, and no other observable in the tree can see it: the equivalence pairs read the answer, the accept counters read the aggregate, and the per-round stream reads `charged` as `false` on both sides by construction, because a capture that enabled the switch would be measuring a different, slower run. Every loop also reaches the one round emit `log_round` exactly once, and, inside the engine source it scans, the target that emit writes on is named in `round_stats.rs` alone — read by file and by fn, so a loop that moves into that file does not inherit its exemption. (The literal is restated once outside that scan, in `tests/common/round_stream.rs`, because the constant is private and a capture has to name the target it declines to enable; `the_engine_and_its_readers_state_the_same_target_and_round_fields` reads `round_stats.rs` and holds the two together, and the same test holds the Rust `ROUND_EVENT_FIELDS` to the tuple `spec_round_stream_compare.py` reads a capture back with.) A loop that can name it can write a second round event with the same fields in the same order, and a line that agrees byte for byte is invisible to the pinned digests, to the pairs and to this census — measured, that bypass passes every other check in the tree. Fewer loops than the tree ships, a call whose arguments the scan cannot read back, a value that is neither a bare identifier nor a `<ident>.<ident>` closed by `,`, `}` or the end of the line — so a call, a cast or a negation that resembles the field is refused rather than counted as it — a loop that does not reach the emit exactly once, the phase target named outside its file, a configuration rebound or written to inside the loop it was handed to, or the low-level rollback named anywhere but `round_common.rs` — and, inside it, named by a round loop rather than by the wrapper — is exit 2. |
| `make check-spec-charge-fixtures` | CI gate (in `make ci`): recall test for the above, 76 cases over three tree shapes — seven self-deciding loops, one drafter migrated, and all seven decisions in entries, which is the tree's shape — each carrying the tree's own census and asserting the reason as well as exit 1 vs exit 2. Among them: a forwarded loop that hard-wires, ORs a condition in, shadows its binding, rebinds the configuration or writes its charge field, assigns through its reference, swaps it out with `mem::replace`, or takes the configuration by `&mut` or by value at all, and one that names the field at every site and passes; a loop taking two configurations, which is an ambiguity refused rather than resolved to the last one declared; a low-level rollback behind a comment carrying a brace, run against its own control, so which reading fired is not left to inference; a trait of bodiless declarations above the loop, which must not swallow it, and an entry with no forwarded loop to enter, which is a loop the scan lost and which the census cannot see, since the lost loop's decision leaves it exactly as the entry's decision enters it; an entry that drops its decision, states two, binds the name to a literal, shadows its binding, or stops building the configuration — which must read as a lost decision and not as a clean scan; a drafter rollback carrying what it was handed, deciding for itself, charging off a second parameter beside the round context, and on a charge that cannot be read back; a comment in a parameter list, which is prose and not a declaration; and the end state with its one loop deleted. It also holds the two shapes that drop a loop out of the derived population, which must read as a lost loop and not as a census that moved, the shared seed emit, which carries a driver's signature and is kept out by naming no totals of its own, and the six RULE 7 shapes: a loop writing its own event on its own target, the same bypass on the shared target, a helper that merely holds the target string, a loop closing its round twice, a loop that moved into the target's own file and writes a second event beside its emit, a trailing comment naming the emit on a line that is not a call to it, and a doc comment naming both the target and the emit, which is prose and passes. |
| `make check-doc-source-citations` | CI gate (in `make ci`): every `crates/...` path cited in `docs/` resolves. Covers paths only, not identifiers — telling a test name from a variable needs a parser. |
| `make check-doc-refs` | CI gate (in `make ci`): no doc edit broke or re-pointed a reference into `docs/` — links, anchors, `§` sections, quoted phrases, line citations, bare doc names and the documentation-map rows — since the base. The base is the merge-base with `DOC_REFS_BASE=<ref>` on the command line (never from the environment), or by default the nearest of `origin/main` and `origin/next/*` that does not contain HEAD; every run prints it. A reference already broken at the base is carried, except that a doc the change edits carries nothing in or out: a cut fixes the broken citations of the docs it touches. `CHANGELOG.md` references are printed and never fail. The reader list and the rules live in `scripts/check_doc_refs.py`. `make check-doc-refs-selftest` is its recall test (in `make ci`). |
| `make check-doc-consumers` | Every reader of doc text in one run: `check-doc-refs`, `check-doc-source-citations`, `check-kv-codec-disposition` (the INERT banners), `check-kv-boundary-default-parity` (the `docs/CLI.md` rows) and `check-published-table` (the generated table). A doc deletion is safe when this passes; whether the kept text is true is the reviewer's check. |
| `make debt-report` | Advisory (non-failing): sibling-file/fn similarity ("twins") in `crates/rmlx-kv-quant` and `crates/rmlx-models` — printing the speculative round-loop drivers as a named group regardless of threshold. The group is *discovered*, not a literal name list, and its rule has one producer: it is `scripts/check_spec_charge.sh`'s population (a) — the driver signature plus a constructed `RoundTotals` — read from `check_spec_charge.sh --list-drivers` (`file`, `fn`, `line`) and joined one to one, with a listed fn the report cannot resolve counted and named rather than dropped, and a lost population or a missing directory reported as `unavailable`, never as zero drivers. Today that is one fn, `run_rounds`. `scripts/debt_report.sh --matched-lines <population>` is the one producer of every duplication figure: summed `difflib` matching-block lines over digit-folded bodies, pairwise over the named population. Eleven populations, each a record carrying its own root, its own collector and its own pairing rule — the root is what the label prints, so no figure names a directory it was not measured over. `drivers`, `impls` and `iso-updates` (the round-loop drivers, the `impl RoundDrafter` bodies, every fn of the KV update files carrying the codec token `iso`) are one family and pair every item with every other — the last because no two of those fns share a digit-stripped name, so a width key would report `0` whatever the files held; `rotor-storage`, `iso-storage` and `turbo-storage` (the non-test `quant_rotor_*.rs` / `quant_iso_*.rs` / `quant_k_turbo*.rs` files under `crates/rmlx-kv-quant/src/storage`), `rotor-updates` and `turbo-updates` (the `update_rotor*` fns and the fns carrying the `tsym` token as a whole segment, both over `crates/rmlx-kv-quant/src/kvcache/update*.rs` — the dispatch file plus one file per codec family, a glob so that a body moving between them cannot move a figure — the turbo one keys on the token rather than a prefix because the entry `update_tsym` and the body it enters, `tsym_update`, spell it on opposite sides of the name, and an anchored prefix would measure the entry and never the body) and `turbo-ssd` (the turbo helper fns of `crates/rmlx-kv-ssd/src/block_io.rs`, a population of its own because its root is a file in another crate and a figure prints the root it was measured over) pair only inside a group sharing the name with every digit run removed — same axis, different width — so a collapsed axis reads a measured `0` with the population still found. The seven KV populations share three collectors, given their glob, their directory or their name pattern at the registration site through `functools.partial`, because a `glob` field on the record would be one the others never read. Every pair is measured both ways round and reported as the larger: `difflib.SequenceMatcher` anchors on the longest match in its first argument, so a one-directional figure moves when a body changes file or name, or when a new item re-orders the population, with no line of any body changing. `population_pairs` also name-sorts every population, so the pair list is a function of the population and not of the filesystem walk. `ssd-hydrate` (the non-test fns under `crates/rmlx-models/src` named exactly `hydrate` or exactly `from_hydrated`) is one family again, and carries two names so one command measures both the per-arch `hydrate` bodies and the short `from_hydrated` entry constructors. `update-bodies` (every `update_`-prefixed fn of the KV update files — the per-variant bodies of every family at once plus the shared entries the dispatch reaches, paired every item with every other, because a width key goes blind the moment the widths collapse) is one family too; the prefix is the whole rule, so the label says what it finds rather than claiming a narrower population. Those nine are a glob plus a name rule, never a file or fn list, which is what lets one command measure a tree that carries the twins and the tree that collapsed them; a population resolving to zero members is `unavailable` and exits 1, never a `0` indistinguishable from a collapsed one. Four debt counters (`#[allow(` sites, debt-marker comments, `check-*` Make targets, oversized files without `LOC-exempt`, each stating its own file population), the churn (summed over commits, not a net diff) since the last tag, and `docs/**/*.md` files over 40 KiB (git-ignored files skipped; outside a git work tree the section reads `unavailable`). Also runs at the end of `make ci` (advisory, non-blocking). |
| `make debt-report-selftest` | CI gate (in `make ci` and hosted CI): the above over synthetic fixtures, 133 cases — a planted twin pair is reported and a same-naming-shape pair just under the threshold is not, the round-loop group is the charge gate's population (a) (signature plus `RoundTotals`, one driver; the seven signature-only fns beside it are excluded; removing the one `RoundTotals` reports `unavailable`, not zero; a renamed driver is found under its new name, count unchanged; two same-named drivers in one file count two, not four; a listed driver the scan cannot resolve prints `1 listed, 0 resolved` and its name; a dead producer prints `unavailable` and no count line), `--matched-lines` is asserted over all eleven populations (0 matched over one driver; a planted impl pair at 7 matched lines that drops to 6 under a body edit; five planted rotor storage files at 51 matched lines over 63 and five planted `update_rotor*` fns at 15 over 23, each with its item and pair count. Both populations carry a **three-member group** beside a two-member one, which is what tells every-pair-in-a-group from consecutive-only pairing — with two-member groups only, the two are the same function. One planted fn spells its width as its own segment (`update_rotor_5_sym`) and must still join its family, which a key that left the doubled separator behind would split off. So an all-pairs population, consecutive-only pairing, a widened glob, a widened fn prefix, a separator-sensitive key or a reverted label all read wrong; each rotor population is then measured at `0` with the population still found once the width twin is deleted, and reported `unavailable` — reason and exit code both — separately for a root that is missing and a root that is there with nothing in it. The empty-population rule is a population rule, not a rotor one, and is asserted on `impls` too. The two iso populations carry the same four shapes at two members per group — four planted `quant_iso_*.rs` files at 23 matched lines over 48 and four planted `update_iso*` fns at 7 over 18, each collapsing to a measured `0` with the population found, then `unavailable` for an empty root and for a missing one — and they share their directory with the rotor storage files and their file with the `update_rotor*` fns, so a widened glob or a widened prefix reads 9 item(s) on one of the two. The four update populations are asserted on both layouts of one planted fixture: moving three `update_rotor*` bodies into a codec family's own `update_rotor.rs` leaves `rotor-updates` and `update-bodies` at the figure, the item count and the pair count they read over one file, so a file list or a glob that missed the family files reads wrong. One further pair is planted for the orientation rule — two bodies that share four lines read one way and three read the other, in both arrangements of their two names, which is the only pair in the fixture tree whose figure that rule can move; reverting `matched_lines` to one direction turns one of the two red, and reverting it together with the sort turns the other. `ssd-hydrate` is asserted on both tree shapes from one planted fixture — two per-arch `hydrate` bodies at 13 matched lines over 28, then the same two arches carrying only the short `from_hydrated` constructors at 8 over 16 — so dropping either name from the rule turns one of the two red, the first by count and the second by going `unavailable`; a `hydrate_from_ssd` beside them and a `hydrate` in a `_tests.rs` file are each outside the population, and counting either would read 3 item(s)), two named `fn …: NN.N% shared` lines are asserted from the twin pair, all four debt counters are asserted against planted fixtures (one `#[allow(`, one debt comment in source and one in a `_tests.rs` file, a Makefile with two `check-*:` targets, an oversized file and its `LOC-exempt`-marked twin), and a synthetic two-commit, one-tag git repo (plus a top-level `README.md` alongside a `docs/*.md` file) proves the churn figure and its "*.md" pathspec are both exercised. Asserts what each case found, not just that the tool ran. |
| `make check-doc-size` | CI gate (in `make ci` and hosted CI): every `docs/**/*.md` git does not ignore (`.md` in any case) is at most 40 KiB. It fails naming each doc over the cap, a stale temporary exception and any doc carrying a line-leading `size-exempt:` marker, after any Markdown leader; no doc exempts itself. The one temporary exception, `docs/METRICS_DB.md`, fails if it grows past its recorded size and once its split lands. Exit 2 when it cannot measure. One producer: `scripts/lib/debt_report.py`. Do not raise the cap; split the doc. |
| `make check-doc-size-selftest` | CI gate (in `make ci` and hosted CI): recall test for the above over throwaway git trees, 29 cases, each asserting the exit code and the reason. |
| `make kv-update-census` | Advisory (non-failing): the KV structural figures, all derived from the tree on every run — the `KvStorage` variant shapes and how many carry the two store slots and `max_seq` one update body can serve; every `match` over `KvStorage` or `KvQuant` naming at least half the enum's variants, per file, with the count and the sub-count that force a touch (no catch-all arm); every `update_`-prefixed fn of the update files (`crates/rmlx-kv-quant/src/kvcache/update*.rs`) with the file it sits in and the lines its body holds — the prefix is the whole rule, and `debt-report --matched-lines update-bodies` reads the same population through the same fn scanner (`scripts/lib/debt_report.py`), so the two figures cannot drift. Exits 2 with `unavailable: <reason>` rather than print a `0` for a tree it could not read. |
| `make kv-update-census-selftest` | CI gate (in `make ci`): recall test for the above over planted trees, 39 cases, each asserting the exit code and the figure or reason — a site in a file the producer was never told about, a collapsed site, a catch-all arm, a match under the bar, a `match` inside a comment and inside a string literal, a variant that grows a state field, a bodiless declaration, a body moved into a codec family's own update file, and every way the tree can be unmeasurable. The derived bar is read with no `--threshold`: it must be half the enum, and a fifth variant moves it and drops a two-variant site with it, so replacing the derivation with a constant turns four cases red. A wide `match` planted in a `*_tests.rs` file is counted only under `--include-tests`, so disabling the test-file exclusion turns one case red. |
| `make check-eval-lock` | CI gate (in `make ci`): every MLX eval FFI call is made under the process-wide evaluation lock (25-symbol reach-set). |
| `make check-eval-lock-fixtures` | CI gate (in `make ci`): recall test for the above, 26 synthetic scan roots, each asserting which rule fired. |
| `make eval-lock-stress` | Drive the evaluation-lock reproducer across `RUNS` fresh processes (default 60). Not in `make ci` — probabilistic (~8%/run) and costs ~412 threads. |
| `make tag` | Create annotated `v<version>` tag from `[workspace.package].version` (single source). |
| `make release-package` | Build + bundle `dist/rmlx-v<ver>-aarch64-apple-darwin.tar.gz` (+ `.sha256`). |
| `make release-sha` | Print sha256 of the `v<ver>` GitHub source tarball (`--write` patches the formula). |
| `make tap-sync` | Copy `packaging/homebrew/rmlx.rb` into the `homebrew-rmlx` tap and push. |
| `make clean` | `cargo clean`. |
| `make serve` | Launch `rmlx serve` on `$(MODEL)` (default = primary test model) at `$(PORT)`. |
| `make chat` | `rmlx chat` on `$(MODEL)`. A stub: it resolves the KV flags, takes the claim, prints one line and exits. |
| `make info` | Dump arch + quant info for `$(MODEL)`. |
| `make logs-tail` | `tail -f` the newest `logs/*.jsonl` under the working directory, not under `<RMLX_HOME>/logs/`. |
| `make metrics-summary` | Prints `<RMLX_HOME>/metrics/summary.csv`. Nothing writes that file, so it prints `no metrics yet`. |
| `make model-check` | `cargo test -p rmlx-{models,runtime,quant,kv-quant}` only — no server/cli/metrics; <30 s, no model needed. |
| `make model-check-full MODEL=…` | `cargo test -p rmlx-{models,runtime,quant}` (note: **not** `rmlx-kv-quant`) + golden-token integration tests. Pass one model path; each golden reads `config.json` and skips gracefully when arch does not match — matching arch runs+passes, others skip. Target is green for any single test-target model. |

`MODEL` and `PORT` override at the CLI: `make info MODEL=/path/to/snapshot`.

**A draft-side change is not proven by byte equality.** Greedy verification
emits the verifier's own argmax at every position regardless of what the
drafter proposed, so a byte-identical stream after a drafter change shows that
run's near-ties happened not to move, not that the round loop is unaffected.
Judge it with one of three checks, each blind to something different: an
identity-row CPU test on the drafter's own math (blind to the round loop and
the verifier); the accept stream (`accept_rate`, `tokens_per_round` — a signal
that moves on near-ties, not an oracle; blind to whether a moved rate is a
defect or a legitimate flip); or an equivalence pair judged by the
divergence-confidence oracle in
[`docs/SPEC_ANSWER_EQUIVALENCE.md`](docs/SPEC_ANSWER_EQUIVALENCE.md) (judges
near-tie vs. defect; blind to the two-model stochastic path, which has no
pair — see its Coverage table). Full argument there and in
[`docs/SPECULATIVE.md`](docs/SPECULATIVE.md).

**A cache-resume change is not proven by byte equality either.** A restored
prefix plus a tail forward is one arithmetic in a different chunking from a
single-shot prefill, so its rows agree to bf16 noise and not, in general, bit
for bit, and one exact tie in a wide vocabulary decodes an unrelated stream
from two equally correct states. Judge a resume arm on the tail logits (argmax
at every position plus a per-logit bound), or against a cold baseline forced
to the same chunk split, and assert the consume branch it reached, because a
`Miss` agrees with a cold baseline for free. See
[`docs/PROMPT_CACHE.md`](docs/PROMPT_CACHE.md) "Judging a resume arm".

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

All on-disk state — logs, metrics DB, ingest buffer, SSD KV cache, bench CSVs, scratch — lives under a single root, resolved at process start by [`rmlx_core::paths::home()`] in this exact order:

1. `$RMLX_HOME` — absolute path, env-var override. Set this in dev shells (`export RMLX_HOME=$PWD/.rmlx`) or production environments where the canonical location is not `$HOME/.rmlx/`.
2. `<workspace>/.rmlx/` — auto-detected by walking up from cwd for `Cargo.lock`. **This is the dev default.** Co-located with the checkout, gitignored, trivially wiped (`rm -rf .rmlx`).
3. `$HOME/.rmlx/` — installed-binary default. Persists across runs.

Standard sub-tree:

```
.rmlx/
  logs/                 per-run JSON logs (rotated by total-size cap)
  metrics/
    runs.db             SQLite metrics DB (source of truth)
    baseline.csv        one row per `rmlx baseline` run
    backups/            `rmlx metrics backup` snapshots
    buffer/pending/     ingest queue
    buffer/failed/      records a replay rejected
  cache/kv/<namespace>/ SSD KV tier blocks and index
  bench/                bench CSVs (perf_canary.csv, …)
  tmp/                  transient files
  profiles.toml         optional `rmlx serve --profile` presets
  projects.toml         optional per-project caps
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

## Metrics (hard rule)

Runs that touch a model record their measurements (load time, tok/s, prefill
speed, KV-cache size, memory residency, smoke-probe pass/fail) in
`<RMLX_HOME>/metrics/runs.db`, one SQLite file and the single source of truth.
`--metrics events` or `--metrics off` records less. Schema and operating rules:
`docs/METRICS_DB.md`; §13 is the summary.

- **Tables**: `observations` (every measurement), `events` (the runtime event
  stream, written by `rmlx_metrics::events::EventRecorder`), `prompts`
  (content-addressed bodies) and the `bests` view over `observations`.
- **Append-only.** Do not delete or overwrite rows.
- **Ingest**: a bench script writes a §8.5 record to
  `<RMLX_HOME>/metrics/buffer/pending/`, then `rmlx metrics record --file
  <path>` ingests and deletes it. Every backend emits that shape.
- **`BENCHMARK_CHAMPIONS.md`** is generated by `make metrics-export`, is
  git-ignored and is never hand-edited.
- **Prompts** are owned by `prompts/*.json`, content-addressed.
- `rmlx metrics …` targets another DB with `--db <path>` or `RMLX_METRICS_DB`.

Do not add tables, hand-edit `BENCHMARK_CHAMPIONS.md`, or write directly to the
DB from non-Rust code.

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
- Beat a record at any cell → record it, regenerate `BENCHMARK_CHAMPIONS.md` (`make metrics-export`) **and** name it in the report.
- Regress >5% → STOP and report — do not commit.
- Bench rows append to `../Cross-Backend-Bench/metrics/summary.csv`.
- Models out of scope (do not bench, do not optimize): Laguna, DR-Venus.

**Perf tooling.** The fast pre-commit smoke is `make canary`, which builds
`release-perf` and runs `scripts/perf_canary.sh`: 1 warmup + 3 measured
baseline calls per model (Bonsai, Gemma4-e4b, Qwen3.6). It prints decode-only
TPS, appends one CSV row per model to `<RMLX_HOME>/bench/perf_canary.csv` and
records one further run in `runs.db`. The anchors in `docs/PERF_BASELINE.md`
are at the bf16 `auto` default (Bonsai ~142, Gemma4-e4b ~80, Qwen3.6 ~101
TPS). Its limit: the `canary` target deletes every `/tmp/rmlx.*.claim` file
before it runs, which bypasses the claim (hard rule 8); check for another MLX
process first. For automated gates use `make canary-gate SHA=<sha>` against
`runs.db`, or `scripts/regression_gate.sh <model> <baseline_tps>
<baseline_stddev>`: exit 125 = `git bisect skip`, exit 1 = regression.
`canary-gate` exits 0 when the SHA has no rows, so a clean exit does not prove
the SHA was measured. Two `Cargo.toml` perf profiles are
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
