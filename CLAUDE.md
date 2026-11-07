# rMLX — agent guide

Rust-native, single-binary MLX inference backend for Apple Silicon.
MLX→MLX conversion is in scope, not yet implemented. Goal: the fastest fully-featured **native, no-Python** backend for
MLX-format models.

## Local-only machine paths

Paths in this file are **relative on purpose** — it is checked in and public.
Concrete absolute machine paths (the model-snapshot root `RMLX_O_MODELS_ROOT`
and local sibling repos) live in a **gitignored** `LOCAL.md` at the repo
root. Use it as a local resolver; never copy an absolute path from it into
this file, a commit, a report, a log, or any artifact that leaves the
machine. The Metal claim path, `/var/tmp/rmlx.claim`, is a fixed system path,
not a per-machine one — it is safe to name in checked-in text; see
`docs/CLI.md` § "Claim file".

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
| [`docs/CLI.md`](docs/CLI.md) | rmlx CLI: subcommands, flags, claim file |
| [`docs/CLI_ENV_VARS.md`](docs/CLI_ENV_VARS.md) | Environment variables `CLI.md` reads, split out for size |
| [`docs/CLI_BASELINE.md`](docs/CLI_BASELINE.md) | `rmlx baseline`, split out of `CLI.md` for size: flags, summary line, refusals, prompt files, prompt-length cap, GPU capture flags |
| [`docs/SERVER.md`](docs/SERVER.md) | HTTP server: OpenAI compat, routes, embeddings, retry envelope |
| [`docs/SERVER_ANTHROPIC.md`](docs/SERVER_ANTHROPIC.md) | Anthropic routes of the HTTP server, split out of `SERVER.md` for size: `/v1/messages` fields, message blocks, input bounds, response and stream shape, the `stop_reason` mapping |
| [`docs/SERVER_TOOLS.md`](docs/SERVER_TOOLS.md) | Tool calling in the HTTP server, split out of `SERVER.md` for size: parsers, template probe, multi-turn loop, forced tool calls, EOF recovery |
| [`docs/SERVER_AUDIO.md`](docs/SERVER_AUDIO.md) | Audio routes of the HTTP server, split out of `SERVER.md` for size: speech-to-text (form, upload, admission, cancel, model cache) and text-to-speech (fields, sampling defaults, response headers, errors) |
| [`docs/SERVER_SHUTDOWN.md`](docs/SERVER_SHUTDOWN.md) | Shutdown of the HTTP server, split out of `SERVER.md` for size: the rule after SIGINT or SIGTERM, the two bounds, the 503, what stops and what does not, the Metal claim at the exit, log events |
| [`docs/SERVER_CLAIM.md`](docs/SERVER_CLAIM.md) | The Metal claim file of the HTTP server, split out of `SERVER.md` for size: `try_claim`, the `flock`, the holder record, the legacy per-port claim |
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
| [`docs/SPECULATIVE.md`](docs/SPECULATIVE.md) | Speculative decoding: the drafters (MTP, DFlash, EAGLE-3, two full models), the round loop, judging a draft-side change, CLI |
| [`docs/SPEC_READING_A_RUN.md`](docs/SPEC_READING_A_RUN.md) | Reading a speculative run, split out of `SPECULATIVE.md` for size: the request record, the round event, charged and uncharged phases, benching an arm, and what a round costs (the verify cost of one more fed row, the step at 6 rows, the round cost by weight format) |
| [`docs/SPEC_ANSWER_EQUIVALENCE.md`](docs/SPEC_ANSWER_EQUIVALENCE.md) | Answer equivalence for speculative decoding: the divergence-confidence oracle, why agreement cannot be thresholded |
| [`docs/SPEC_ROUND_SKELETON.md`](docs/SPEC_ROUND_SKELETON.md) | The speculative round loop `run_rounds` and the `RoundDrafter` interface: what each drafter declares, what differs per drafter, the two two-model entries kept apart, the oracle, what no runtime check sees |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Per-token sampling (temperature, top-k/p, penalties, thinking budget, constrained decoding) |
| [`docs/FFI.md`](docs/FFI.md) | rmlx-mlx ↔ mlx-c FFI bridge; MSL kernel surface; unsafe policy |
| [`docs/MLX_PAIR.md`](docs/MLX_PAIR.md) | The pinned MLX / mlx-c pair: what to do on each Mac, the source build, `make mlx-restore-pin`, the pin gate and its verdicts, the two mlx-c C APIs, the attention row rule, moving the pin |
| [`docs/METRICS_DB.md`](docs/METRICS_DB.md) | Metrics DB: path, identity rules, ingest and the run class, `rmlx metrics` tooling, operating rules |
| [`docs/METRICS_EXPORT.md`](docs/METRICS_EXPORT.md) | `BENCHMARK_CHAMPIONS.md` and the exports, split out of `METRICS_DB.md` for size: what `export --markdown` renders, the other formats, the `Updated` column |
| [`docs/METRICS_IMPORT.md`](docs/METRICS_IMPORT.md) | `rmlx metrics import`, split out of `METRICS_DB.md` for size: what is copied, the run class by source version, the skip key, the report |
| [`docs/METRICS_SCHEMA.md`](docs/METRICS_SCHEMA.md) | Metrics DB schema: tables (observations, events, prompts, the bests view), metric registry, plausible-value bounds, known-bad rows |
| [`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md) | The three canary anchors and the bench methods: A/B comparison, per-codec cells, the bandwidth ceiling, cross-backend cells |
| [`docs/PUBLISHED_PROTOCOL.md`](docs/PUBLISHED_PROTOCOL.md) | Generated: published-protocol results (MT-Bench / MATH-500 / HumanEval, fixed prompt) with every number beside the bound it cannot pass |
| [`docs/PROFILING.md`](docs/PROFILING.md) | Profiling runbook: samply, Instruments, Metal GPU capture, memory counters, the prefill-chunk knob |
| [`docs/PROJECTS_CONFIG.md`](docs/PROJECTS_CONFIG.md) | Per-project cap defaults via `<RMLX_HOME>/projects.toml` |
| [`docs/TESTING.md`](docs/TESTING.md) | Test snapshot resolution (RMLX_O_MODELS_ROOT, RMLX_TEST_MODEL_*), test variables, golden-token fixtures, CPU numeric gates, NIAH and codec smoke matrix |
| [`docs/GPU_TESTS.md`](docs/GPU_TESTS.md) | GPU/Metal tests: the `#[ignore]` rule and its gate, `make gpu-test`, stand-downs, halves, `make ci-perf`, shader validation, census pin |
| [`docs/AUDIO.md`](docs/AUDIO.md) | Audio: Whisper transcription (word timestamps, prompt, window decision, HTTP route), Whisper token ids |
| [`docs/AUDIO_WAV.md`](docs/AUDIO_WAV.md) | WAV I/O, split out of `AUDIO.md` for size: the decoder (formats, resample and memory, duration cap, declared length, trim, lost audio, refusals) and `WavEncoder` |
| [`docs/AUDIO_TTS.md`](docs/AUDIO_TTS.md) | Qwen3-TTS: the speech model (loading, talker, code predictor, generation loop, memory of a request), sampling of the codec tokens, the codec decoder |
| [`docs/E2E_TEST_PLAN.md`](docs/E2E_TEST_PLAN.md) | End-to-end feature-proof harness: modality, tool-calling and speculative-decoding cases |
| [`docs/RELEASING.md`](docs/RELEASING.md) | Release flow: single-source version, `make tag` / `release-package` / `tap-sync`, Homebrew formula + tap, `CHANGELOG.md`, branch model (one PR per issue into `main`, squash-merged), post-release retrospective |

Filing or triaging an issue: see the `## Issues` section in
[`CONTRIBUTING.md`](CONTRIBUTING.md) — templates, labels, stages, and the
analysis-comment shape.

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
| Qwen3.8 | `mlx-community__Qwen3.8-27B-4bit` | `Qwen3_5ForConditionalGeneration` |
| Bonsai 2 | `prism-ml__Ternary-Bonsai-2-27B-mlx-2bit` | `prism_hadamard_qwen35` (`model_type`; dispatched as `Qwen3_5ForConditionalGeneration`) |

Other Open Models snapshots (`z-lab__Qwen3.6-27B-PARO`, `medgemma`, the `jina`
embedding/reranker models, `ReaderLM-v2`, …) are in scope as feature
coverage grows. Qwen3.6 and every Bonsai without "2" keep their cells and are
not test targets; medgemma, Laguna and DR-Venus have no test bound to their
snapshots; see `docs/TESTING.md` § "Test targets, kept cells and gaps".

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
8. **Single MLX process per Mac**. A held claim refuses with exit code 11,
   naming the holder; stop the holder by its PID (`kill <PID>`). Never
   delete the claim file. See `docs/CLI.md` § "Claim file".
9. **`make ci-perf` builds + tests under `release-perf` (panic=unwind, debug-assertions off), then runs the GPU/Metal suite.** A failure in the `release-perf` half that doesn't reproduce under `dev` → rebuild under `release-debug` (full DWARF) and re-run the failing case to capture symbols. Never rely on the `dev` profile to reproduce a release-mode bug — codegen and inlining differ. The GPU half is the exception and builds under `dev` on purpose: debug assertions are correctness guards and those are correctness tests. Its consequence: **no gate anywhere executes a `Device::Gpu` test under `release-perf`** — `make test` / `make ci` are `dev` with no `--ignored`, `test-perf` is `release-perf` with no `--ignored`, the GPU suite is `dev` with `--ignored`. A GPU-path defect that appears only with debug-assertions off is therefore out of every gate's scope and must be reproduced by hand at that profile.
10. **Every KV-cache codec ships an MSL (Metal) decode kernel.** A codec whose decode falls back to CPU dequant is not shippable — it strands the codec at single-digit TPS (GPU idle) and is a bug, not a valid mode. New KV codecs (and the decode path of existing ones) MUST decode on-GPU, reading the quant store directly (fused flash-decode-over-quant; see `docs/KV_FUSED_KERNELS.md` and `docs/FFI.md`). Every MSL kernel body a **production** path can dispatch — KV codec or not — lives in a `.metal` file under a gated `src/metal/` directory (the list is `scripts/lib/metal_dirs.sh`: `rmlx-kv-quant`, `rmlx-models`, `rmlx-mlx`), never in a Rust string literal, and carries a **native-compilation test** (`xcrun -sdk macosx metal -c` at `-std=metal3.1` and `-std=metal4.0`, wired as `make check-metal-compiles` in `make ci`) so MSL syntax errors surface at CI, not on first GPU dispatch. The gate also fails on a `.metal` file its directory's `probes/kernels.manifest` does not name — an unchecked body is the same defect wearing a different hat. Throwaway `#[cfg(test)]` bodies are exempt and stay inline (`metal_kernel_tests.rs` holds a trivial `add_one` smoke and a deliberately-invalid source that must never compile). **Know the gate's boundary:** it keys off directory membership, so it enforces this rule for kernels already in those directories but cannot detect a new inline-MSL literal in a fresh module — that part is review's job, not CI's. Kernels stay **model-agnostic** — keyed off codec + shape (`head_dim`, `kv_heads`, `bits`), never an arch name.
    **When to write a kernel.** Beyond KV decode, write a `.metal` kernel when MLX has no op for the job, or when an A/B against the MLX op on two or more architectures (one `kv_h == 1`, one `kv_h > 1`) shows a gain beyond noise; the MLX op is the baseline arm. Prefer porting from sibling backends (llama.cpp `ggml-metal`, MLX) over writing from scratch; the PR names the source file and its licence. Dispatch through MLX `metal_kernel`. MLX's own internal kernels (for example `quantized_matmul`) are not covered by this rule.

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
  MLX contact generally.** The CPU side is held apart differently:
  `EVAL_LOCK` / `with_eval_lock` in `rmlx-mlx` serialises every evaluation
  process-wide, so parallel test threads never evaluate at once — not by
  ignoring CPU tests, which would only stop running them. Every evaluation
  also runs on the one MLX thread that owns the streams every op is built on
  (`docs/FFI.md` § "The MLX thread"). Two deterministic gates hold the lock, and they are
  complementary by construction: `make check-eval-lock` fails the build on any
  MLX eval FFI call made outside the lock (25-symbol reach-set, derived from the
  linked dylibs — **not** just the eval-named ones) but is blind to a lock that
  stopped locking, which `under_eval_lock_excludes_concurrent_callers` and
  `with_eval_lock_holds_the_lock_while_its_job_runs` catch: through the MLX
  thread alone, a lock that does not lock still runs one evaluation at a time.
  `make eval-lock-stress` is the 400-thread end-to-end driver, deliberately out
  of `make ci`. See `docs/FFI.md`.
  Run GPU tests with **`make gpu-test`** (every member crate,
  serialized; `CRATE=` / `FILTER=` to narrow), or by hand as
  `cargo test --no-run -p <crate> --tests` (build, outside any claim) then
  `rmlx claim run -- cargo test -p <crate> --lib -- --ignored <filter>
  --test-threads=1` (run, holding the Metal claim) — see `docs/GPU_TESTS.md`.
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
  `tests/**/*.rs`; it keys on shape (does the test reach `Device::Gpu`?), never on
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
  deliberately **not** in `scripts/check/gpu_tests.sh list` — `suite/run_gpu_tests.sh` asserts a Metal
  validation banner per crate and every declared test is snapshot-gated or drives
  a child, so listing one would fail the suite over a missing model. This one
  check keys on the ignore *text*, which is why a Metal-driving test whose reason
  never says "Metal" or "GPU" stays outside it. A **macro-generated** test is
  enforced at
  its `macro_rules!` body (one body governs every cell it emits), and a body the
  scanner cannot read back — an assembled fn name, a whole macro on one line, an
  item whose brace never closes, an attribute whose bracket never closes, an item
  after an attribute's closing bracket on its line, a qualifier alone on a line after an attribute, a signature that closes a bracket it never opened and does not end in `;`, a string or block comment still open at the end of a file — is a
  hard failure rather than a clean scan;
  those cells are deliberately excluded from `scripts/check/gpu_tests.sh list` / `make gpu-test`, which
  every run prints. A proc-macro-generated test, and a `macro_rules!` with a
  non-brace delimiter, remain outside the fail-closed net — neither exists in
  the tree. A bare `#[ignore]` on an environment-gated test (an `env::var` read or the
  `test_snapshot::snapshot` resolver) that reaches no `Device::Gpu`
  is fatal on the same terms: it runs under no gate and has no reason for that
  check to read.
  `make check-gpu-tests-ignored-fixtures` (in `make ci` and hosted CI: the
  `gpu_tests_ignored` mode of `scripts/check/rust_gates_fixtures.sh`, run
  once under each awk on the machine) pins the gate's recall in both directions,
  asserting each case's failure *reason* and not just its exit code. A directory or
  file the scan cannot read is a refusal that names it, never a tree with no
  findings. See `docs/GPU_TESTS.md`.
- **Advisory: `make file-size-report`** prints files >1000 LOC. Non-failing.
  Also runs at the end of `make ci` (advisory, non-blocking).
- **Advisory: `make target-size-report`** prints `target/` size and, past a
  50 GB threshold, a hint to run `make target-gc` (the staleness-based
  pruner, run by `make target-gc`). `target/` has no size cap; this just
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
7. **Pass model-side values through.** A value whose meaning the model or its chat template defines (a template variable, an effort level, a free-form label) goes to the model side as the client sent it, and the model side's own error is the answer for a bad one. No allow-list, no table per model or per template, and no check whose only job is to tell the client that its value is unusual. A check on a request value stays when it protects the server: memory, time, a crash, a path leak, an injection. One mechanism keyed on data (the template, the config, the tensor shape) replaces a branch per model or template name.

## Common commands (Makefile)

`make help` lists every target with its one-line help. The Makefile is the dev
loop and the CI gate: prefer it over cargo flags typed by hand. Script rules:
[`scripts/README.md`](scripts/README.md). Every script, its caller and its
purpose: [`scripts/INDEX.md`](scripts/INDEX.md) (generated by
`make scripts-index`). What a gate checks is in its script's header.

| Target | What it runs |
|---|---|
| `make build` / `check` / `test` | `cargo build --workspace --release` / `check --workspace --all-targets` / `test --workspace` (no `#[ignore]` GPU test) |
| `make gpu-test` | the GPU/Metal `#[ignore]` tests, serialized, under the Metal claim (`HALF=` `CRATE=` `FILTER=` `VALIDATE=0`); see `docs/GPU_TESTS.md` |
| `make fmt` / `fmt-check` / `lint` | `cargo fmt` / its check / `cargo clippy -D warnings` |
| `make audit` / `deny` | `cargo audit` / `cargo deny check` |
| `make precommit` / `hooks` | run the pre-commit hooks / install them |
| `make ci` | the pre-merge gate: fmt, lint, test, deny, audit, every `check-*` gate and recall test, then the advisory reports |
| `make ci-perf` | `test-perf` under `release-perf` + the GPU suite; needs an idle GPU (`HALF=codec\|rest`); see `docs/GPU_TESTS.md` |
| `make check-<name>` | one tree gate (every one but the aggregate `check-doc-consumers` is in `make ci`); `check-<name>-selftest` / `-fixtures` is its recall test |
| `make check-scripts` / `scripts-index` | the scripts rules / regenerate `scripts/INDEX.md` |
| `make check-doc-consumers` | every reader of doc text: doc refs, source citations, codec banners, boundary default, published table; a doc deletion is safe when it passes, but whether the kept text is true is the reviewer's check |
| `make published-table` | regenerate `docs/PUBLISHED_PROTOCOL.md` (generated, never hand-written) |
| `make debt-report` / `kv-update-census` / `file-size-report` / `target-size-report` / `shellcheck-report` | advisory reports (exit 0) |
| `make canary` / `canary-ab` / `canary-gate SHA=` | decode-TPS canary / interleaved A/B / gate on `runs.db` |
| `make spec-canary` / `ssd-canary` / `schema-constraint-canary` | real-model canaries; `spec-canary-gate` / `ssd-canary-gate` gate them on `runs.db` |
| `make mlx-restore-pin` | restore the pinned MLX pair (`docs/MLX_PAIR.md`) |
| `make eval-lock-stress` / `miri` | eval-lock reproducer across fresh processes / Miri on the MLX-thread tests (not in `make ci`) |
| `make target-gc` / `traces-gc` | report (`APPLY=1` prunes) stale `target/` profiles / `.rmlx/traces` bundles |
| `make tag` / `release-package` / `release-sign` / `release-sha` / `bottle` / `tap-sync` | release flow (`docs/RELEASING.md`) |
| `make metrics-init` / `metrics-doctor` / `metrics-export` / `metrics-backup` | the `runs.db` tooling (`docs/METRICS_DB.md`) |
| `make serve` / `chat` / `info` | `rmlx` on `$(MODEL)` at `$(PORT)` (`chat` is a stub) |
| `make model-check` / `model-check-full MODEL=` | model-logic crates only, no model / + golden tokens (not `rmlx-kv-quant`) |

A PR runs `make ci` and the GPU tests of its blast radius: `make ci-perf
HALF=codec` when its diff touches only crates below the model layer, only
`crates/rmlx-models/src/`, or only integration binaries that select a KV codec;
`HALF=rest` when it touches only the binaries that do not. Anything else, and
every merge to `main`, runs the whole `make ci-perf`. Do not raise the doc-size
cap; split the doc.

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
`--metrics events` or `--metrics off` records less. Schema:
`docs/METRICS_SCHEMA.md`. Operating rules: `docs/METRICS_DB.md`; §13 is the
summary.

- **Tables**: `observations` (every measurement), `events` (the runtime event
  stream, written by `rmlx_metrics::events::EventRecorder`), `prompts`
  (content-addressed bodies) and the `bests` view over `observations`.
- **Append-only.** Do not delete or overwrite rows.
- **Run class.** Every `observations` row is `gate`, `ab_arm` or `diagnostic`
  (`docs/METRICS_DB.md` §8.5). Record every measurement run under its class:
  canaries, release gates and the surviving arm of an A/B as `gate`, every A/B
  slot as `ab_arm`, investigation runs as `diagnostic`. Only `gate` rows reach
  `bests`, the gates and the exports. A record that states no class is refused.
  `rmlx metrics import` brings a scratch DB's rows in
  (`docs/METRICS_IMPORT.md`).
- **Ingest**: a bench script writes a §8.5 record to
  `<RMLX_HOME>/metrics/buffer/pending/`, then `rmlx metrics record --file
  <path>` ingests and deletes it. Every backend emits that shape. A run that
  measured and whose record did not land exits 75; a `gate` caller fails on
  it, and other callers keep the reading and end non-zero
  (`docs/METRICS_DB.md` §13 rule 24).
- **`BENCHMARK_CHAMPIONS.md`** is generated by `make metrics-export`, is
  git-ignored and is never hand-edited.
- **Prompts** are owned by `prompts/*.json`, content-addressed.
- `rmlx metrics …` targets another DB with `--db <path>` or `RMLX_METRICS_DB`.

Do not add tables, hand-edit `BENCHMARK_CHAMPIONS.md`, or write directly to the
DB from non-Rust code.

## Regression-bench discipline (hard rule)

Every code-touching change runs a regression smoke before declaring done:
the three test-target families (Gemma4, Qwen3.8, Bonsai 2) **plus any model
the change touches**, each at its best-known KV quant.

- **Correctness first, on ≥2 architectures.** Any code-touching change proves
  correct output on real models spanning ≥2 archs — minimum **gemma4-e2b**
  (single-KV-head `kv_h == 1`, shared-KV) **+ Qwen3.8-27B-4bit** (`kv_h == 4`,
  dense Qwen3.5 hybrid). No target has the Qwen3 architecture or YaRN, and
  `head_dim` 128 with attention on every layer is held only by cells outside
  the rules. A KV/kernel fix that holds at one arch/shape can silently fail at
  another (`kv_h == 1` vs `kv_h > 1`, power-of-two vs non-power-of-two
  `head_dim`). Unit tests alone are not proof; serve the model.
- Decode TPS within ±1% of the recorded best for that model at that KV mode.
- Beat a record at any cell → record it, regenerate `BENCHMARK_CHAMPIONS.md` (`make metrics-export`) **and** name it in the report.
- Regress >5% → STOP and report — do not commit.
- Bench rows append to `../Cross-Backend-Bench/metrics/summary.csv`.
- Models out of scope (do not bench, do not optimize): Laguna, DR-Venus.

**Perf tooling.** The fast pre-commit smoke is `make canary`, which builds
`release-perf` and runs `scripts/bench/baseline_cells.sh --preset canary`: 1
warmup + 3 measured baseline calls per model (Bonsai, Gemma4-e4b, Qwen3.6). It prints decode-only
TPS, appends one CSV row per model to `<RMLX_HOME>/bench/perf_canary.csv` and
records one further run in `runs.db`. The anchors in `docs/PERF_BASELINE.md`
are at the bf16 `auto` default (Bonsai ~142, Gemma4-e4b ~80, Qwen3.6 ~101
TPS). The `canary` target stops no process and deletes no claim file: when
another process holds the Metal claim, the canary script exits 11 and names the
holder. For automated gates use `make canary-gate SHA=<sha>` against
`runs.db`.
Through `make canary-gate`, a failed recipe line always reports `make`'s own
exit code (2) — a regression, an unknown SHA and the underlying skip
condition all surface as the same 2. Call `rmlx metrics deltas
--since-sha <SHA> --threshold-pct 3 --exit-code true` directly to read its
own exit codes: 1 for a regression or a SHA with no gate observations
(unknown SHA), 125 for the skip condition (no comparable baseline). A known SHA with
no cell measured again after it exits 0 even directly, so a clean exit does
not prove every cell was re-measured. Two `Cargo.toml` perf profiles are
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
