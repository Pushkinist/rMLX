# rMLX Test Environment Variables

Integration and smoke tests skip gracefully when their snapshot is absent.
Set the env vars below to point at local model snapshots and re-run tests to
exercise the model-gated paths.

All paths must be **absolute** paths to existing model snapshot directories.
The model directory must contain at minimum `config.json` and `tokenizer_config.json`.

---

## Model snapshot variables

| Variable | Open Models snapshot | Arch |
|----------|------------------|------|
| `RMLX_TEST_MODEL_GEMMA4_E4B` | `mlx-community__gemma-4-e4b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| `RMLX_TEST_MODEL_GEMMA4_E2B` | `mlx-community__gemma-4-e2b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| `RMLX_TEST_MODEL_GEMMA4_PARO` | `z-lab__gemma-4-31B-it-PARO` | `Gemma4ForConditionalGeneration` |
| `RMLX_TEST_MODEL_GEMMA4_12B` | `mlx-community__gemma-4-12B-it-mxfp8` | `Gemma4UnifiedForConditionalGeneration` |
| `RMLX_TEST_MODEL_MEDGEMMA` | `mlx-community__medgemma-1.5-4b-it-8bit` | `Gemma3ForConditionalGeneration` |
| `RMLX_TEST_MODEL_QWEN36` | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| `RMLX_TEST_MODEL_QWEN36_PARO` | `z-lab__Qwen3.6-27B-PARO` | `Qwen3_5MoeForConditionalGeneration` |
| `RMLX_TEST_MODEL_BONSAI` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |
| `RMLX_TEST_MODEL_DR_VENUS` | `z-lab__DR-Venus-*` | `Qwen3ForCausalLM` |
| `RMLX_TEST_MODEL_JINA_V4` | `jinaai__jina-embeddings-v4` | `JinaVLForEmbedding` |
| `RMLX_TEST_MODEL_LAGUNA` | `z-lab__Laguna-*` | `LagunaForCausalLM` |
| `RMLX_TEST_MODEL_READERLM_V2` | `mlx-community__jinaai-ReaderLM-v2` | `Qwen2ForCausalLM` |
| `RMLX_TEST_MODEL_QWEN3_VL_30B` | `mlx-community__Qwen3-VL-30B-Instruct-*` | `Qwen3VLForConditionalGeneration` |

## Specialised test-model variables

Some integration tests use dedicated snapshot variables instead of the family
variables above:

| Variable | Used by | Purpose |
|----------|---------|---------|
| `RMLX_TEST_MODEL` | `rmlx-server/tests/ssd_cache_restart.rs` | Generic single-model override for the SSD-restart smoke test. |
| `RMLX_KV_TEST_MODEL` | `gemma4_kv_cache_equivalence.rs`, `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs`, `projects_toml_e2e.rs`, `cli_flags_e2e.rs` | Model snapshot for KV-cache equivalence and drafter-alignment tests. Typically set to a Gemma4-e4b path. |
| `RMLX_DRAFT_TEST_MODEL` | `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs` | Draft model snapshot path. Used alongside `RMLX_KV_TEST_MODEL` for speculative-decode alignment tests. |
| `RMLX_VL_TEST_MODEL` | `qwen3_vl_moe_text_parity.rs` | Vision-language model snapshot for VL text-parity tests. |

The Whisper audio integration tests (`crates/rmlx-audio/tests/transcribe.rs`)
deliberately use **no** dedicated env var — they resolve the
`mlx-community__whisper-large-v3-mlx` + `openai__whisper-large-v3-tokenizer`
snapshots directly under `RMLX_O_MODELS_ROOT` (auto-discovery, skip-if-absent) and
scan the gitignored `crates/rmlx-audio/tests/fixtures/` dir for a
`*.{m4a,wav,…}` + sibling `*.transcript.vtt` long-form regression pair. The
former `RMLX_TEST_MODEL_WHISPER` knob was removed.

## Directory root variable

| Variable | Purpose | Default |
|----------|---------|---------|
| `RMLX_O_MODELS_ROOT` | Root directory containing all model snapshots. Used by fixture generators and integration helpers that resolve snapshots by slug. | `./models` (repo-local fallback; set RMLX_O_MODELS_ROOT) |

## E2E harness — data-driven model specs

The E2E harness (`crates/rmlx-cli/tests/e2e/`, `make e2e`) resolves a manifest
`model` field that is a **path**, a snapshot **slug**, or a frozen alias
(`BONSAI`, `GEMMA4_E4B`, `GEMMA4_E2B`, `QWEN36`) — see
`docs/E2E_TEST_PLAN.md` §Model resolution. Adding a model needs no code edit.

Per-spec runtime override: `RMLX_E2E_MODEL_<SPEC>` (or `RMLX_TEST_MODEL_<SPEC>`),
where `<SPEC>` is the spec upper-cased with every non-alphanumeric mapped to
`_`. For the big-Gemma4 rows whose `model` is a raw slug, the override keys are:

| Manifest `model` slug | Override variable |
|---|---|
| `mlx-community__gemma-4-26b-a4b-it-mxfp8` | `RMLX_E2E_MODEL_MLX_COMMUNITY__GEMMA_4_26B_A4B_IT_MXFP8` |
| `mlx-community__gemma-4-31b-it-mxfp8` | `RMLX_E2E_MODEL_MLX_COMMUNITY__GEMMA_4_31B_IT_MXFP8` |

Alias-form rows keep the short keys above (`RMLX_TEST_MODEL_BONSAI`, …).

---

## Usage examples

Set a single model for a targeted test run:

```bash
export RMLX_TEST_MODEL_GEMMA4_E4B=/absolute/path/to/mlx-community__gemma-4-e4b-it-mxfp8
cargo test -p rmlx-server
```

Set all three primary test-target models for the full regression suite:

```bash
export RMLX_TEST_MODEL_GEMMA4_E4B=/absolute/path/to/open-models/mlx-community__gemma-4-e4b-it-mxfp8
export RMLX_TEST_MODEL_QWEN36=/absolute/path/to/open-models/mlx-community__Qwen3.6-35B-A3B-8bit
export RMLX_TEST_MODEL_BONSAI=/absolute/path/to/open-models/prism-ml__Ternary-Bonsai-8B-mlx-2bit
cargo test --workspace
```

Set the Open Models root for fixture generators and roundtrip tests:

```bash
export RMLX_O_MODELS_ROOT=/absolute/path/to/open-models
cargo test --workspace
# or run the fixture generator:
python crates/rmlx-server/tests/chat_template_fixtures/gen_fixtures.py
```

---

## CI behaviour

When env vars are unset, all snapshot-gated tests **skip** with an `[SKIP]` or
`tracing::warn!` message and report success. The test suite is always green on
machines without model snapshots (including CI).

GPU-intensive tests are additionally marked `#[ignore]` and must be run
explicitly:

```bash
cargo test --test embeddings_smoke -- --ignored --test-threads=1
```

---

## Cosine-similarity gate

Every KV-cache codec has a per-codec cosine-similarity quality gate in the
`rmlx-kv-quant` unit-test suite. The gate verifies that a quantize →
dequantize round-trip preserves the directional information in each row vector
to within an empirically derived floor.

All cosine gates use the **same LCG fixture** (seed `TEST_SEED =
0x0000_00C0_FFEE_BEEF`, Knuth LCG) so they are deterministic and require no
model snapshot or GPU.

### Thresholds

| Codec / variant | Test name | `mean` threshold | `min` threshold | Source |
|---|---|---|---|---|
| q8_0 (K8V8 both sides) | `q8_cosine_gate_k8v8` | ≥ 0.9990 | ≥ 0.9970 | empirical floor 2026-05-30 |
| TurboQuant V4 | `turbo_v4_cosine_gate_k8v4` | ≥ 0.9937 | — | mtq README `turbo4`=0.9947 − 0.001 |
| TurboQuant V3 (K8VTurbo3) | `turbo_v3_cosine_gate_k8vturbo3` | ≥ 0.9807 | — | mtq README `turbo3`=0.9817 − 0.001 |
| PlanarQuant V4 | `planar_v4_cosine_gate` | ≥ 0.9942 | — | mtq README `planar4`=0.9952 − 0.001 |
| rot_k Hadamard 8-bit | `rot_k_hadamard_8bit_cosine_gate` | ≥ 0.9970 | ≥ 0.9990 | empirical floor remeasured 2026-05-30 (LCG >> 32 fix; was ≥ 0.9950 on biased fixture) |
| Mixed K8V4 (bits=4, group=64) | `mixed_k8v4_g128_64_cosine_gate` | ≥ 0.9937 | — | same floor as TurboQuant V4 |
| Mixed K8V8 (bits=8, group=128) | `mixed_k8v8_g128_128_cosine_gate` | ≥ 0.9990 | — | same floor as q8_0 |
| Mixed K8V2 (bits=2, group=32) | `mixed_k8v2_g128_32_cosine_gate` | ≥ 0.9000 | — | empirical floor 2026-05-30 |

### Helpers

All helpers live in `crates/rmlx-kv-quant/src/test_utils.rs`:

- `cosine_similarity_per_row` — f64-accumulator cosine per `head_dim`-sized row; returns `CosineStats { mean, min, n_rows }`.
- `lcg_data(n, seed)` — deterministic LCG fixture data in `[-1.0, 1.0]` (upper 32 bits of state, symmetric; a `>> 33` bug that biased output to `[-1.0, ~0.0)` was fixed).
- `fwht_normalize(buf, n)` — CPU Walsh-Hadamard transform (self-inverse when applied twice), used by the rot_k cosine test.
- `TEST_SEED` — pinned seed constant (`0x0000_00C0_FFEE_BEEF`). Never replace with `thread_rng`.

### Running only cosine gates

```bash
cargo test -p rmlx-kv-quant cosine_gate
```

---

## Vectorized-vs-scalar parity

Each rMLX KV codec ships a CPU scalar reference path and a GPU/MSL kernel.
Parity tests verify that the two paths agree within a codec-specific tolerance.

### Helper

`crates/rmlx-kv-quant/src/test_utils.rs` — `pub(crate)`:

```rust
pub(crate) fn vectorized_parity_check<F1, F2>(
    cpu_path: F1,
    msl_path: F2,
    input: &[f32],
    tol: f32,
    name: &str,
) where
    F1: FnOnce(&[f32]) -> Vec<f32>,
    F2: FnOnce(&[f32]) -> Vec<f32>,
```

Runs both paths on `input`, asserts `max-abs-error <= tol`, and prints a
concise diff (first diverging index) on failure.

### `RMLX_SKIP_GPU` env-var opt-out

Set `RMLX_SKIP_GPU=1` to skip GPU parity tests silently, even when
`--include-ignored` is passed. Each parity test body starts with:

```rust
if crate::test_utils::skip_if_no_gpu_env() { return; }
```

`#[ignore]` still gates the default test run (opt-in requires `--include-ignored`).
`RMLX_SKIP_GPU=1` is an additional opt-out for CI environments that have Metal
present but should not exercise the GPU.

Run parity tests in isolation on a Metal machine:

```bash
cargo test -p rmlx-kv-quant -- --include-ignored --test-threads=1
```

Skip them even when running with `--include-ignored`:

```bash
RMLX_SKIP_GPU=1 cargo test -p rmlx-kv-quant -- --include-ignored
```

### Per-codec tolerance policy

| Codec family | Tolerance | Rationale |
|---|---|---|
| Integer / packed codes (bit-level) | exact | GPU layout == CPU bit-pack |
| TurboQuant V4 (codebook lookup) | 5e-3 | f32 rounding in codebook lookup |
| PlanarQuant V4 (codebook + rotation) | 5e-3 | f32 rounding in codebook lookup |
| K8VTurbo3 V (3-bit codebook lookup) | 1e-3 | tighter: 3-bit centroids smaller |
| rot_k FWHT + affine q8 | 0.10 | one 8-bit quant step for D=128 FWHT range |
| q8_0 group-128 affine | 5e-3 | f32 rounding in min/max scan |

Tolerance values are **upper bounds** — tightening them silently may cause
flakes on M-chip generations with different f32 rounding. Change only with
a measured justification.

---

## NIAH long-context harness

`crates/rmlx-models/tests/niah_long_context.rs` — server-free needle-in-a-
haystack test that verifies long-context retrieval at multiple ctx tiers
(8k / 16k / 32k) × multiple needle depths (10/30/50/70/90%) for each of
the three primary test-target models.

Each cell is its own `#[ignore]` `#[test]`, gated on its `RMLX_TEST_MODEL_*`
env var (per the table above). Cells are parametrised by a `FlashKind`
axis:

- **`niah_<model>_*`** (Turbo family): forces `KvQuant::K8V4` so the
  TurboFlash MSL kernel dispatches when enabled. Consults
  `RMLX_TURBO_FLASH`.
- **`niah_pflash_<model>_*`** (planar_flash_decode family): forces
  `KvQuant::PlanarK` so the
  `update_and_sdpa_planar_k_fused` → `planar_flash_decode_sdpa` chain
  activates. Consults `RMLX_PLANAR_FLASH_DECODE`. Bonsai-only Reachable
  arch (Qwen3.6 MoE rejects PlanarK at validate_resolved; Gemma4 routes
  through `update_and_sdpa_returning_kv`).

Neither family sets its env var directly — the kernel gates are
`OnceLock`-latched. To compare OFF vs ON, run the shell driver:

```bash
# Default — TurboFlash cells, OFF then ON
bash scripts/release_e2e/stage6_perf/niah_long_context.sh

# planar_flash_decode cells, defaults to niah_pflash_ filter
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --mode pflash

# Both families
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --mode both

# Pin to one orientation:
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --off-only
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --on-only

# Filter to one cell / model:
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --filter niah_gemma4_32k
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --mode pflash --filter niah_pflash_bonsai_16k
```

The driver runs two fresh `cargo test` processes per pass (OFF then ON),
each with `--ignored --test-threads=1 --nocapture` so per-cell decoded
outputs are logged. Preflight `pkill`/claim-file cleanup honours
CLAUDE.md hard rule 8 (single MLX process). Per-pass logs land in
`/tmp/niah-<family>-<label>.log`.

Required env vars: `RMLX_TEST_MODEL_GEMMA4_E4B`,
`RMLX_TEST_MODEL_QWEN36`, `RMLX_TEST_MODEL_BONSAI`. Unset → skip.

---

## Prompt fixture note

`prompts/ssd_bench/structured_regex_gen.json` contains the path `/home/user/.rmlx/metrics/runs.db-wal` as LLM input content. This is a synthetic placeholder (`/home/user/` is not the developer's home directory) and is intentional — it is content-addressed, so changing it would invalidate the fixture hash.

---

## Sparse-attn calibration runner

The `rmlx kv-calibrate --recipe head_budget` subcommand is a
model-loading calibration pass — not a unit test. Two test surfaces
exist in CI:

* **CLI smoke** (`cargo test -p rmlx-cli kv_calibrate`) — preflight
  checks (missing `config.json`, out-of-range `--mass-threshold`,
  non-Qwen3 architecture in `config.architectures[0]`). No model load,
  no Metal claim.
* **Schema round-trip** (`cargo test -p rmlx-loader head_budgets`) —
  validates `HeadBudgets` / `HeadBudgetCalibration` writer + reader and
  structural validation (shape mismatch, zero-budget rejection).

To run the real calibration end-to-end on a snapshot (Bonsai is the
primary smoke target):

```bash
# Preflight (CLAUDE.md hard rule 8 — single MLX process):
pkill -f "rmlx serve"; pkill -f mlx_lm; \
  rm -f /tmp/rmlx.0.claim /tmp/rmlx.8080.claim

rmlx kv-calibrate \
  /path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit \
  --recipe head_budget \
  --mass-threshold 0.95
```

Default prompt set: `prompts/calibration_default.json` (8 prompts).
Override via `--prompts <path>`. Output: `<MODEL>/head_budgets.json`
per [`rmlx_loader::head_budgets`](../crates/rmlx-loader/src/head_budgets.rs).

For the synthetic GPU parity tests covering the two-phase sparse-attn
MSL kernels (`phase1_score`, `phase2_sparse_attend`, LSE merge), see
the unit tests in `crates/rmlx-kv-quant/src/sparse_attn/`.

---

## Codec smoke + NIAH matrix

End-to-end smoke + long-context retrieval gate over every supported
(codec, model) cell. Driver:
`scripts/release_e2e/stage6_perf/codec_smoke_runner.sh`.

### Manifest schema

`scripts/release_e2e/stage6_perf/kv_codec_matrix.toml` — one `[[entries]]`
table per cell. Primary key: `(codec_name, model)`.

| Field | Type | Purpose |
|---|---|---|
| `codec_name` | string | Display name (`k8v4`, `TurboSym3`, …). |
| `model` | string | Manifest slug — `bonsai-8b`, `gemma4-e4b`, `qwen3.6-moe-8bit`. |
| `context_length` | int | NIAH context size in tokens (32 768 for all 0.1.0 rows). |
| `expected_retrieval_pct` | float | Baseline retrieval rate; `0.0` = not yet recorded. |
| `smoke_probe_prompts` | array<string> | Prompt names from `smoke_prompts.toml`. |
| `skip_reason` | string | Non-empty = row skipped (see conventions below). |
| `cli_args` | string | `rmlx baseline` suffix that selects the codec. |
| `niah_filter` | string | NIAH cargo-test test-name filter (e.g. `niah_bonsai_32k`). |

### Smoke prompt set

`scripts/release_e2e/stage6_perf/smoke_prompts.toml`. Three prompts shared
across every row:

| Name | Purpose | Validation |
|---|---|---|
| `coherence` | "Describe a sunrise in three sentences." | regex `(?i)(?:[a-z]+[\s,.';:!?*#-]+){4}[a-z]+` + printable ratio ≥ 0.95 |
| `instruction` | "List 3 benefits of regular exercise. Number them 1, 2, 3." | regex `(?s)1.*2.*3` |
| `multi_turn` | Short-context colour recall ("red and white lighthouse"). | Must echo `lighthouse`, `red`, or `white` |

All prompts are English-only and arch-independent (no chat-template
markup) so a single fixture works for Bonsai / Gemma4 / Qwen3.6.

**Smoke prompt calibration notes:**

- `coherence` separator class updated from `[ ,.';:!?-]` to `[\s,.';:!?*#-]`
  so thinking-model output (Bonsai / Qwen3.6 emit `<think>` blocks with
  structured markdown and newlines) passes the five-word sequence gate.
- `instruction` regex simplified from `(?s).*1[.)].*2[.)].*3[.)]` to
  `(?s)1.*2.*3`. The word-boundary `\b` variant (`\b1\b`) also fails:
  Rust's Debug formatter escapes `\n` → `\\n` in the tracing field, so
  the char preceding `1` in extracted output is `n` (word-char), and `\b`
  does not fire.
- `multi_turn` prompt replaced from a `User:`/`Assistant:` multi-turn
  format to a single inline colour-recall paragraph. Gemma4 in raw
  text-completion mode does not use plain `User:`/`Assistant:` labels as
  role delimiters and produces off-context responses for the earlier format.

### Runner usage

```bash
# Full matrix.
make smoke-codec-matrix

# Filter to one codec across all three models.
make smoke-codec-matrix CODEC=k8v4

# Filter to one model across all codecs.
make smoke-codec-matrix MATRIX_MODEL=bonsai-8b

# Both filters compose.
make smoke-codec-matrix CODEC=PlanarK MATRIX_MODEL=gemma4-e4b

# Record baselines (writes measured retrieval_pct back into the manifest
# for rows whose `expected_retrieval_pct == 0.0`).
make smoke-codec-matrix RECORD=1
```

The variable is `MATRIX_MODEL` (not `MODEL`) because the top-level
`MODEL ?= …/gemma-4-e4b-it-mxfp8` default would otherwise leak into the
filter for the default `make smoke-codec-matrix` invocation.

Direct shell invocation supports the same flags plus `--manifest <path>`
and `--dry-run`:

```bash
bash scripts/release_e2e/stage6_perf/codec_smoke_runner.sh --dry-run
bash scripts/release_e2e/stage6_perf/codec_smoke_runner.sh \
    --filter codec_name=Iso3Sym --filter model=bonsai-8b --record-baseline
```

Per-run aggregate output: `scripts/release_e2e/stage6_perf/last_run.json`
(gitignored).

### Baseline recording vs gating

* **First run on a fresh row** (`expected_retrieval_pct == 0.0`) with
  `--record-baseline`: the measured retrieval rate is written back into
  the manifest. Re-run without `--record-baseline` to gate.
* **Subsequent runs**: `measured >= expected - 0.02` passes; otherwise the
  row FAILs (and `agg_rc != 0`). The two-percentage-point slack absorbs
  M-chip-generation f32 rounding noise.

The baseline-recording pass populates the bf16 reference rows first, then
sweeps each codec against its bf16 baseline.

### Skip conventions

| `skip_reason` | Meaning |
|---|---|
| `qwen-moe-A.y-rejected` | A.y arch invariant: Qwen3.6-MoE rejects K-side ≤4-bit codecs. Symmetric in the manifest for cross-model inventory; never executed. |
| `production dispatch pending` | Fused-QK / sparse-attn integration HOLD. Production wiring not landed; do not gate on retrieval until the integration ships. Remove this value per row when the integration merges. |
| empty string | Row is live and executes. |

### CI gate

`.github/workflows/codec-matrix.yml`:

* Triggers only on `push` to `develop`.
* Self-hosted Apple Silicon runner; required env: the three `RMLX_TEST_MODEL_*` snapshot paths.
* Pull requests do not trigger the gate (manual / Exec B sweeps only).
* `last_run.json` uploaded as artifact on every run.
* On any row FAIL the workflow posts a one-line commit comment with the
  failed-row count.

The gate honours the single-MLX-process discipline (CLAUDE.md hard
rule 8) via the runner's `preflight` (pkill + claim-file cleanup) before
each row's `rmlx baseline` and NIAH `cargo test` invocations.

---

## Test behaviour toggles

These variables modify test execution without requiring a model snapshot.
They are read only inside test code (`tests/` and `*_tests.rs` files).

| Variable | Values | Description |
|---|---|---|
| `RMLX_SKIP_GPU` | `1` | Skip GPU/Metal parity tests even when `--include-ignored` is passed. |
| `RMLX_REGEN_GOLDENS` | any | Regenerate golden-token fixtures instead of asserting them. |
| `RMLX_E2E_REGEN_GOLDEN` | `1` | Regenerate E2E golden snapshots in the harness runner. |
| `RMLX_E2E_ONLY` | comma-separated spec names | Run only the named E2E specs; skip all others. |
| `RMLX_REGISTRY_TEST` | any | Enable multi-model registry smoke tests (require model snapshots). Unset → skip. |
| `RMLX_NIAH_KV_QUANT` | KV quant name (e.g. `k8v4`) | Override the KV quant used in NIAH long-context harness tests. |
| `RMLX_APPLE10_STRICT` | `1` | Fail (not warn) on Apple10 head-dim=256 cosine gate below floor. |
| `RMLX_FUSED_QK_STRICT` | `1` | Fail (not warn) on fused-QK parity tests. |
| `RMLX_RETURNING_KV_STRICT` | `1` | Fail (not warn) on returning-KV dispatch parity tests. |
| `RMLX_SPARSE_ATTN_STRICT` | `1` | Fail (not warn) on sparse-attn dispatch parity tests. |

---

## In-process tests must not rely on the `paths::home()` `OnceLock`

`rmlx_core::paths::home()` caches its resolved root in a `OnceLock` — fixed
for the lifetime of the process. In-process unit tests share one process, so a
test that does `std::env::set_var("RMLX_HOME", tmp)` and then reads a
`paths::*` path **races every other test in the same binary**: whichever test
resolves `home()` first pins the root, and a later `set_var` is silently
ignored. The path then points at the workspace `.rmlx/` instead of the temp
dir, which both flakes the test and leaks artifacts into the checkout.

For in-process tests, **inject the root explicitly** — pass a temp path to the
routine under test (open SQLite via `SsdKvIndex::open_at(&db_path)`, write
fixture files under the temp dir) rather than going through `paths::home()`.
Setting `RMLX_HOME` is only hermetic for **subprocess** tests
(`Command::new(...).env("RMLX_HOME", tmp)`), where the child gets a fresh
`OnceLock`.
