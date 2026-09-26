# rMLX tests: snapshots, variables and CPU gates

This doc covers how model-gated tests find their snapshots, the variables that
steer tests, and the numeric gates that need no model. The tests that drive
the GPU, and the gates that run them, are in [`GPU_TESTS.md`](GPU_TESTS.md).

## Running the tests

| Target | What it runs |
|---|---|
| `make test` | `cargo test --workspace`, no `--ignored`. A snapshot-gated test skips. |
| `make model-check` | the `rmlx-models`, `rmlx-runtime`, `rmlx-quant` and `rmlx-kv-quant` tests; no model |
| `make model-check-full MODEL=<snapshot>` | `model-check` without `rmlx-kv-quant`, then the five golden-token suites with `MODEL` as `RMLX_KV_TEST_MODEL` |
| `make e2e` | the E2E harness, `crates/rmlx-cli/tests/e2e/` |
| `make gpu-test` | the GPU and Metal tests |

A test that skips reports success, so a machine without snapshots runs the
default suite green.

## Snapshot resolution

### The models root

`RMLX_O_MODELS_ROOT` is the directory that holds every snapshot under its slug
(`mlx-community__gemma-4-e4b-it-mxfp8`). A bare `cargo test` sees it only if
the shell exports it.

Every `make` target exports a root that an operator names. The precedence is:
command-line variable, then environment, then `.env`. With none of them set,
make exports the repo-local `models/` directory only when it exists. The two
commands below mean the same:

```bash
RMLX_O_MODELS_ROOT=/tmp/empty make gpu-test CRATE=rmlx-models
make gpu-test CRATE=rmlx-models RMLX_O_MODELS_ROOT=/tmp/empty
```

### The resolvers

Each test resolves its snapshot by one of these rules:

| Rule | Code | Order | A set but wrong value |
|---|---|---|---|
| golden | `common::model_for` | `RMLX_KV_TEST_MODEL` if it serves the test's architecture, then the slug | fails |
| slug first | `common::slug_or_override` | the slug, then the variable the test names | fails when the slug is absent; ignored when it is present |
| slug first, in `src/` | `rmlx_models::test_snapshot::snapshot` | the slug, then a `RMLX_TEST_MODEL_*` variable | skips |
| variable first | `tests/resolved_arch_class.rs`, `crates/rmlx-cli/src/commands/kv_calibrate_tests.rs` | a `RMLX_TEST_MODEL_*` variable, then the slug | falls through to the slug |
| variable first, drafter | `resolve_drafter` in `tests/spec_sampled_distribution.rs` | `RMLX_DRAFT_TEST_MODEL`, then the slug; a directory with a `config.json` is enough | fails |
| variable only | every other test that names a variable | the variable | skips |

`common` is `crates/rmlx-models/tests/common/mod.rs`. Its two rules share one
probe. It fails when `RMLX_O_MODELS_ROOT` is set but is not a directory, and
when a variable it consults names a path that is not a runnable snapshot. It
skips when nothing is configured, or when the root does not hold the slug.

"Runnable" depends on what the caller opens. The probe takes a `Role`:

| File | `Standalone` | `Sidecar` |
|---|---|---|
| `config.json` | required | required |
| `tokenizer.json` | required | — |
| `model.safetensors.index.json` or `model.safetensors` | required | required |
| at least one `*.safetensors` file | required | required |

`Sidecar` is the drafter role. A drafter is decoded with its verifier's
tokenizer, and mlx-community ships drafter snapshots without one. A download
writes the JSON files before the shards, so a snapshot with no shard is a
half-written one. The probe reads it as absent, and the test skips.

## Model snapshot variables

| Variable | Snapshot | Architecture |
|---|---|---|
| `RMLX_TEST_MODEL_GEMMA4_E4B` | `mlx-community__gemma-4-e4b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| `RMLX_TEST_MODEL_GEMMA4_E2B` | `mlx-community__gemma-4-e2b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| `RMLX_TEST_MODEL_GEMMA4_PARO` | `z-lab__gemma-4-31B-it-PARO` | `Gemma4ForConditionalGeneration` |
| `RMLX_TEST_MODEL_MEDGEMMA` | `mlx-community__medgemma-1.5-4b-it-8bit` | `Gemma3ForConditionalGeneration` |
| `RMLX_TEST_MODEL_QWEN36` | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| `RMLX_TEST_MODEL_QWEN36_PARO` | `z-lab__Qwen3.6-27B-PARO` | `Qwen3_5ForConditionalGeneration` |
| `RMLX_TEST_MODEL_ORNITH_9B` | `sahilchachra__ornith-1.0-9b-mxfp8-mlx` | `Qwen3_5ForConditionalGeneration` |
| `RMLX_TEST_MODEL_BONSAI` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |
| `RMLX_TEST_MODEL_DR_VENUS` | a `z-lab__DR-Venus-*` snapshot | `Qwen3ForCausalLM` |
| `RMLX_TEST_MODEL_JINA_V4` | `jinaai__jina-embeddings-v4` | `JinaEmbeddingsV4Model` |
| `RMLX_TEST_MODEL_LAGUNA` | a `z-lab__Laguna-*` snapshot | `LagunaForCausalLM` |
| `RMLX_TEST_MODEL_READERLM_V2` | `mlx-community__jinaai-ReaderLM-v2` | `Qwen2ForCausalLM` |
| `RMLX_TEST_MODEL_QWEN3_VL_30B` | a `mlx-community__Qwen3-VL-30B-Instruct-*` snapshot | `Qwen3VLMoeForConditionalGeneration` |

The architecture column is the resolved class, `Architecture::arch_class()`. The
Jina encoder has no `Architecture` variant; its column is the declared
`architectures[0]`. For the Qwen3.5 family it follows the checkpoint's tensors,
not its `architectures[0]`. `tests/resolved_arch_class.rs` pins that. It builds
a snapshot that declares dense and ships MoE tensors, with symlinked weights.
The Qwen-MoE K-side codec guard must still reject a codec on it.

`cache_type_tests.rs` covers the guard's table without weights. Whether
`Architecture::generate_greedy`, `generate_image`, the `ArchGenerator` and
`SpeculativeGenerator` constructors and the speculative per-request path call
it, only snapshot-gated tests check. Deleting one of those calls leaves
`make ci` green.

## Other test-model variables

| Variable | Read by | Rule |
|---|---|---|
| `RMLX_KV_TEST_MODEL` | the golden-token suites, `bitnet_logprobs.rs`, `qwen3_5_moe_forward_seq_last_k.rs`, the verifiers of the alignment suites, `spec_conditioning_residual.rs`, `spec_greedy_equivalence.rs` and `spec_sampled_distribution.rs` | golden |
| `RMLX_KV_TEST_MODEL` | `gemma4_kv_cache_equivalence.rs`, `kv_bytes_sample_point.rs`, `cli_flags_e2e.rs`, `projects_toml_e2e.rs` | variable only |
| `RMLX_DRAFT_TEST_MODEL` | the drafters of the alignment suites, `spec_conditioning_residual.rs` and `spec_greedy_equivalence.rs` | slug first |
| `RMLX_DRAFT_TEST_MODEL` | the drafter of `spec_sampled_distribution.rs` | variable first, drafter |
| `RMLX_VL_TEST_MODEL` | `qwen3_vl_moe_text_parity.rs` | variable only |
| `RMLX_PROMPT_CACHE_TEST_MODEL_A`, `_B` | `prompt_cache_cross_model.rs` | variable only |
| `RMLX_TEST_MODEL` | `crates/rmlx-server/tests/ssd_cache_restart.rs` | variable only |

The files above are under `crates/rmlx-models/tests/` unless a path is given.

`prompt_cache_cross_model.rs` needs two snapshots of one architecture, with
the same KV shape and different weights. `mlx-community__gemma-4-e2b-it-mxfp8`
and `mlx-community__gemma-4-E2B-it-qat-4bit` fit. The test refuses a pair
whose outputs are identical, since the comparison would prove nothing.

`crates/rmlx-audio/tests/transcribe.rs` takes no variable. It resolves
`mlx-community__whisper-large-v3-mlx` and `openai__whisper-large-v3-tokenizer`
under `RMLX_O_MODELS_ROOT`. Its long-form case reads audio files with a
sibling `*.transcript.vtt` from the git-ignored
`crates/rmlx-audio/tests/fixtures/`.

## Speculative-decoding suites

Every suite below names its verifier and its drafter by slug, as constants in
the test file. The verifier resolves by the golden rule. The drafter resolves
by the slug-first rule, except in `spec_sampled_distribution.rs`, where the
variable comes first. A machine that holds both snapshots runs the suite with
no variable set.

| File | What it asserts |
|---|---|
| `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs`, `qwen3_5_mtp_drafter_alignment.rs`, `qwen3_5_eagle3_alignment.rs`, `qwen3_5_two_model_alignment.rs` | the drafter loads and aligns with its verifier over a first round or a prefix |
| `spec_greedy_equivalence.rs` | each of its ten pairs produces the verifier's own answer at temperature 0 |
| `spec_sampled_distribution.rs` | the sidecar arm draws from the verifier's distribution above temperature 0 |
| `two_model_stochastic.rs` | the two-model loop samples at temperature 1.0, and one seed reproduces one stream |
| `dflash2_loader.rs` | the DFlash2 checkpoint loads, and its forward matches a committed reference |
| `spec_conditioning_residual.rs` | the DFlash drafters' conditioning path |

A first-round alignment check cannot see a rollback that corrupts the
verifier's state later in a run. `spec_greedy_equivalence.rs` can: it runs 256
tokens per prompt. It judges each pair with the divergence-confidence oracle in
`docs/SPEC_ANSWER_EQUIVALENCE.md`. Before loading, it refuses a drafter of the
wrong kind, a drafter quantized unlike its verifier, and a drafter of the wrong
width. For the two-model pair it compares the two tokenizers id by id
(`vocab_pairing` in `crates/rmlx-models/src/speculative/mod.rs`).

`spec_sampled_distribution.rs` gives each emitted token a surprise under the
distribution the plain path would have drawn from. It reads the stream's total
surprise as a `z` score. A second arm at temperature 0 is a positive control
that the run must refuse. Four CPU cases in the same file need no snapshot.
They pin the statistic's power against a greedy stream, a wrong temperature and
a stream drawn without the request's filters.

`two_model_stochastic.rs` and `dflash2_loader.rs` resolve by slug only, through
`common::slug_snapshot`.

## Golden-token suites: how their snapshot resolves

`crates/rmlx-models/tests/{bonsai,gemma4,qwen3,bitnet,medgemma}_golden_tokens.rs`
each decode 32 tokens at temperature 0 and compare the ids with a committed
fixture under `tests/fixtures/`. Each suite covers one architecture and names
its snapshot by slug. They resolve by the golden rule:

1. `RMLX_KV_TEST_MODEL`, for the golden whose architecture it serves. Named
   at another architecture, it falls through to step 2 and prints
   `NOTE <test>: … using <path> instead`.
2. The golden's slug under `RMLX_O_MODELS_ROOT`.

The fall-through exists because other suites need `RMLX_KV_TEST_MODEL`,
typically at a Gemma4 path. With it exported, a golden of another
architecture still runs from its slug.

The goldens do not read the `RMLX_TEST_MODEL_*` variables. A golden pins the
bytes of one checkpoint, and a same-family substitute would fail as a
regression. To use a snapshot outside the models root, symlink it in under its
slug.

| Configuration | Outcome |
|---|---|
| the snapshot resolves and its architecture matches | run |
| `RMLX_KV_TEST_MODEL` names another architecture | fall through to the slug |
| the same, with `RMLX_REGEN_GOLDENS` set | fail |
| nothing configured, or the root does not hold the slug | skip |
| the slug directory is half-written | skip |
| `RMLX_KV_TEST_MODEL` names a path that is not a runnable snapshot | fail |
| `RMLX_KV_TEST_MODEL` names a snapshot whose `config.json` is unreadable | fail |
| `RMLX_O_MODELS_ROOT` is set but is not a directory | fail |
| the slug is a snapshot of another architecture | fail |

The goldens are `#[ignore]`d, so `make ci` runs none of them. `make gpu-test`
and `make ci-perf` run them.

### Recording a fixture, and the gate on overwriting one

`RMLX_REGEN_GOLDENS` (any value) makes the harness write the fixture instead of
comparing it:

1. With no committed fixture, or with unchanged ids, it writes.
2. When the ids changed, it decodes again with `top_logprobs_k = 2`. It reads
   the top-2 logprob gap at the first differing index.
3. It writes only when that gap is at most `REGEN_MAX_TIE_MARGIN` (0.10).
   Otherwise it panics with `REFUSED`, naming the index, both ids and the
   margin.

A change in token count is refused at any margin. So is a margin it cannot
measure. The margin is printed on the `WROTE` line on stderr; the fixture does
not record it. Passing this gate does not
make the new output correct. It shows only that the flip sat at a tie.

## E2E harness model specs

The E2E harness (`crates/rmlx-cli/tests/e2e/`, `make e2e`) reads a manifest
`model` field that is a path, a snapshot slug, or one of the aliases
`BONSAI`, `GEMMA4_E4B`, `GEMMA4_E2B` and `QWEN36`. See
`docs/E2E_TEST_PLAN.md` §Model resolution.

`RMLX_E2E_MODEL_<SPEC>` or `RMLX_TEST_MODEL_<SPEC>` overrides one spec. `<SPEC>`
is the spec in upper case, with every non-alphanumeric character mapped to
`_`. A slug row such as `mlx-community__gemma-4-31b-it-mxfp8` takes
`RMLX_E2E_MODEL_MLX_COMMUNITY__GEMMA_4_31B_IT_MXFP8`.

## Test behaviour variables

These change how a test runs and need no snapshot. Only test code reads them.

| Variable | Values | Effect |
|---|---|---|
| `RMLX_SKIP_GPU` | `1` | the GPU tests that check it return at once, `--ignored` or not |
| `RMLX_REGEN_GOLDENS` | any | the golden-token suites write their fixtures |
| `RMLX_E2E_REGEN_GOLDEN` | `1` | the E2E harness writes its golden snapshots |
| `RMLX_E2E_ONLY` | case ids, comma-separated | the E2E harness runs only those cases |
| `RMLX_REGISTRY_TEST` | any | arms `crates/rmlx-server/tests/multi_model_smoke.rs` |
| `RMLX_NIAH_KV_QUANT` | a KV quant name | the NIAH cells decode with that codec |
| `RMLX_APPLE10_STRICT` | `1` | `apple10_head_dim_256.rs` fails where it warns |
| `RMLX_FUSED_QK_STRICT` | `1` | `fused_qk_dispatch.rs` fails where it warns |
| `RMLX_SHARED_SOURCE_STRICT` | `1` | `shared_source_dispatch.rs` fails where it warns |
| `RMLX_SPARSE_ATTN_STRICT` | `1` | `sparse_attn_dispatch.rs` fails where it warns |

### `RMLX_SKIP_GPU` opt-out

`test_utils::skip_if_no_gpu_env()` in `rmlx-kv-quant` returns true only for
the value `1`. Most GPU tests in that crate's `src/` call it first and
return. Several integration tests under `crates/rmlx-kv-quant/tests/`, and
`crates/rmlx-models/tests/sparse_attn_dispatch.rs`, read the variable with the
same rule through a private copy. Other GPU tests do not check it.

```bash
RMLX_SKIP_GPU=1 cargo test -p rmlx-kv-quant -- --include-ignored
```

No test writes `RMLX_SKIP_GPU`. The tests read it without the env lock, so
a write could skip a live test. The value rule is the pure function
`skip_value_means_skip()`, which is tested directly.

## Env-backed gates: readers need the lock too

`test_utils::env_lock()` in `rmlx-kv-quant` is a process-global guard for
tests that touch the environment. No gate checks that a test takes it. The
rules:

1. Hold it for the whole test body.
2. Take it to read, as well as to write. A test that reads
   `rotor_qjl_enabled()` races the tests that set `RMLX_ROTOR_QJL`.
3. Set the state you assert. The lock serializes access and resets nothing.

One lock covers the whole environment. `setenv` is undefined behaviour against
a concurrent `getenv` of any key.

`env_lock()` returns an `EnvGuard` that restores `RMLX_ROTOR_QJL` on drop,
also while unwinding from a failed assertion. It is the only key the guard
manages. `RMLX_ROTOR_QJL` is read again at every store construction.

The kernel gates (`RMLX_TURBO_FLASH`, `RMLX_TURBO_FLASH_LOCK`,
`RMLX_TURBO_FLASH_MIN`, `RMLX_FUSED_QK`, `RMLX_FUSED_QK_MIN`,
`RMLX_SPARSE_ATTN`, `RMLX_PLANAR_FLASH_DECODE`, `RMLX_ROT_K_FUSED`) seed a
[`DispatchPolicy`](../crates/rmlx-core/src/dispatch_policy.rs). Each `KvCache`
captures one at construction. A test builds its cache with
`.with_dispatch_policy(…)` and takes no lock. The variable still sets the
default for a whole process, which is what the shell drivers use.

`rmlx-kv-ssd` is a separate test binary and keeps its own lock.

## In-process tests must not rely on the `paths::home()` `OnceLock`

`rmlx_core::paths::home()` caches its root in a `OnceLock` for the life of the
process. Unit tests share one process. So a test that sets `RMLX_HOME` and then
reads a `paths::*` path races every other test in the binary. The first caller
pins the root, and the files land in the workspace `.rmlx/`.

In-process, pass the temp path to the code under test, for example
`SsdKvIndex::open_at(&db_path)`. `RMLX_HOME` is hermetic only for a child
process: `Command::new(…).env("RMLX_HOME", tmp)`.

## Allocation gates (`PeakBracket`)

A numerics test cannot see a change that keeps every output bit and allocates
an extra buffer per dispatch. `rmlx_mlx::PeakBracket` scopes the Metal
allocator's high-water mark to a region:

```rust
let bracket = PeakBracket::open();
let out = op_under_test(&input, Device::Gpu)?;
out.eval()?;                       // MLX is lazy: materialise inside
let reading = bracket.close();

assert!(reading.observed_allocation());               // first
assert!(reading.headroom_bytes() <= 4 * input_bytes); // relative
```

- **Assert `observed_allocation()` before any upper bound.** An `eval()`
  outside the bracket reads as no allocation, and an upper bound then passes.
  The predicate is `headroom_bytes() > 0`.
- **Bound a multiple of the workload's size, never an absolute count.** MLX
  pools buffers, so an absolute figure depends on what ran before.
- **The peak mark is process-global.** These tests reach `Device::Gpu` and run
  serialized like every GPU test.

Reference caller: `q8_msl_roundtrip_allocation_stays_within_budget` in
`crates/rmlx-kv-quant/src/q8_msl_tests.rs`. The accessors are described in
[`docs/PROFILING.md` §9.1](PROFILING.md).

## Cosine-similarity gate

The `*_cosine_gate` tests in `rmlx-kv-quant` round-trip a fixture through a
codec and assert a floor on the per-row cosine. The i.i.d. gates use the LCG
fixture and need no snapshot and no GPU. Each assertion states its floor.

```bash
cargo test -p rmlx-kv-quant cosine_gate
```

The LCG fixture is i.i.d. uniform. A decorrelating rotation cannot improve
it, so an identity rotation passes every i.i.d. gate. The rotation-quality
gates below cover rotation.

### Helpers

`crates/rmlx-kv-quant/src/test_utils.rs` holds:

- `cosine_similarity_per_row`: cosine per `head_dim` row, f64 accumulator;
  returns `CosineStats { mean, min, n_rows }`.
- `lcg_data(n, seed)`: LCG data in `[-1.0, 1.0]` from the upper 32 bits of
  the state.
- `gaussian_data(n, seed)`: standard normal from the same LCG, by Box–Muller.
- `outlier_channel_data(rows, head_dim, channels, ratio, seed)`: Gaussian with
  persistent large channels. `outlier_fixture()` is 256 × 128 with 4 channels
  at 20×.
- `incoherence_per_row`: `mu = sqrt(d)·max|x_i|/||x||_2` per row; returns
  `IncoherenceStats { mean, p99, max, n_rows }`.
- `sqnr_db`, `wasted_bits`, `lloyd_max_anchor_db`,
  `LLOYD_MAX_GAUSSIAN_SQNR_DB`, `DB_PER_BIT`: the rate-distortion reference.
- `fwht_normalize(buf, n)`: CPU Walsh-Hadamard transform.
- `TEST_SEED`: the seed, `0x0000_00C0_FFEE_BEEF`.
- `vectorized_parity_check(cpu_path, msl_path, input, tol, name)`: runs both
  paths and fails when the max absolute error exceeds `tol`, naming the first
  index past it. Each caller passes its codec's tolerance; most compare a
  CPU and an MSL path, and two in `planarquant_tests.rs` compare two CPU
  paths.

## Rotation-quality gates

`crates/rmlx-kv-quant/src/rotation_fidelity_tests.rs` runs on the CPU with no
snapshot, inside `make model-check`. It measures on `outlier_fixture()`. The
measured figures are in `docs/KV_CODEC_FIDELITY.md` § "Codec fidelity —
measured".

```bash
cargo test -p rmlx-kv-quant --lib rotation_fidelity -- --nocapture
```

| Gate | Asserts |
|---|---|
| `hadamard_incoherence_ratio_beats_every_block_local_rotation` | `rot_k` cuts mean `mu` at least 3×; each block-local family stays under its `sqrt(block)` ceiling and under `rot_k` |
| `non_full_dimension_rotations_fail_the_hadamard_incoherence_gate` | a block-4 truncated Hadamard and the iso, rotor and planar transforms all fail the 3× bar |
| `identity_rotation_excluded_by_the_hadamard_incoherence_threshold` | the 3× bar excludes 1.00× |
| `iso_block_rotation_incoherence_gate`, `planar3_…`, `planar4_…` | the reduction is under the `sqrt(block)` ceiling and at least the pinned value minus 0.05 |
| `rotor_block_rotation_incoherence_gate` | the same, over 8 `(layer, head)` draws, pinned to the weakest |
| `rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data` | the Hadamard buys at least 1.5 bits of SQNR on outlier data and loses bits on i.i.d. data |
| `non_full_dimension_rotations_fail_the_rot_k_gain_gate` | a block-4 truncated Hadamard and the iso quaternion fail the 1.5-bit bar |
| `identity_rotation_excluded_by_the_rot_k_gain_threshold` | the 1.5-bit bar excludes 0 bits |
| `<codec>_outlier_cosine_gate` (7) | outlier-fixture cosine floors for `rot_k`, `iso3/4`, `rotor3/4` and `planar3/4` |
| `lossier_codecs_fail_the_outlier_cosine_floors` | each of the seven floors rejects a lossier real codec |
| `wider_codebooks_score_higher_on_the_outlier_fixture` | iso4 beats iso3, and rotor4 beats rotor3 |

The outlier cosine floors are relative to the error: a codec may double
`1 − cos` before its floor fails (`COSINE_ERROR_TOLERANCE`). An absolute slack
cannot work here. `rot_k` scores within 0.001 of 1 with the Hadamard deleted.

## Rate-distortion reference

`crates/rmlx-kv-quant/src/rate_distortion_tests.rs` runs on the CPU with no
snapshot, inside `make model-check`.

```bash
cargo test -p rmlx-kv-quant --lib rate_distortion -- --nocapture
```

It encodes an i.i.d. Gaussian fixture through every scalar-codebook codec at
every shipped width. It reports SQNR against the Lloyd-Max Gaussian anchor for
that width, as wasted bits. Two thresholds, in bits:

- **Per-cell pin**: the measured value plus `PIN_SLACK_BITS` (0.10). This is
  the gate that fails. `pinned_budgets_sit_one_slack_above_the_measurement`
  keeps each pin one slack above its measurement.
- **Absolute**: `MAX_WASTED_BITS` (1.0) against the anchor. Every pin is
  below it, so it labels a failure and adds no coverage.
  `one_bit_short_codec_fails_the_rate_distortion_gate` shows a codec one bit
  short that stays inside 1.0 and fails its pin.

Two equalities are pinned, so a change that moves them fails:
`trellis_coded_quantization_claws_back_nothing` (TCQ equals plain turbo) and
`planar_widths_are_byte_identical_and_the_others_pay_for_their_bits`. The
second asserts that planar costs the same bytes at 3 and 4 bits, and that iso
and rotor pay one bit per stored code for the fourth bit.

## NIAH long-context harness

`crates/rmlx-models/tests/niah_long_context.rs` hides a needle in a filler
text and asserts that a greedy decode recovers it. Each cell is an
`#[ignore]`d test generated by `niah_cell!` or `niah_pflash_cell!`:

- `niah_<model>_<ctx>k_d<depth>`: 45 cells over Bonsai, Gemma4-e4b and
  Qwen3.6, at 8k, 16k and 32k tokens and five depths. They force
  `KvQuant::K8V4`.
- `niah_pflash_<model>_*`: 16 cells. They force `KvQuant::PlanarK`.

A cell reads its model from `RMLX_TEST_MODEL_BONSAI`,
`RMLX_TEST_MODEL_GEMMA4_E4B` or `RMLX_TEST_MODEL_QWEN36`, with no slug
fallback. Unset, the cell skips. `RMLX_NIAH_KV_QUANT` replaces the forced
codec.

A cell that reaches its decode asserts that the decode recovers the needle.
It also reads the kernel's dispatch counter around the decode, under the
process-default dispatch policy. With the kernel on and the cell marked
`Reachable`, a TurboFlash cell asserts that the kernel ran. A planar cell
asserts that it did not: the live bf16 K seed keeps it dormant. Every other
cell asserts no dispatch.

The three `niah_pflash_qwen36_32k_*` cells never reach their asserts.
`validate_resolved` rejects `PlanarK` on Qwen3.6 MoE, so with
`RMLX_TEST_MODEL_QWEN36` set they panic in `generate_greedy`. Unless
`RMLX_NIAH_KV_QUANT` replaces the codec, every driver run that selects them
fails, the default `--mode turbo` run included.

`scripts/release_e2e/stage6_perf/niah_long_context.sh` runs the cells in two
fresh processes, with the kernel off and then on:

```bash
bash scripts/release_e2e/stage6_perf/niah_long_context.sh                # TurboFlash
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --mode pflash  # niah_pflash_ cells
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --mode both
bash scripts/release_e2e/stage6_perf/niah_long_context.sh --on-only --filter niah_gemma4_32k
```

`--mode turbo` runs every cell unless `--filter` narrows it. The driver sets
`RMLX_TURBO_FLASH` or `RMLX_PLANAR_FLASH_DECODE`, builds under `release-perf`
and writes each pass's log to `/tmp/niah-<family>-<label>.log`.

### Why NIAH is not in `make gpu-test`

The `#[ignore]` rule holds on the two macro bodies:
`make check-gpu-tests-ignored` fails if either loses the attribute. The cells
are not in `make gpu-test`. A macro cell has no name in the source, so the gate
cannot derive a filter for it. Each cell also loads a snapshot and prefills up
to 32k tokens. The NIAH driver and `make smoke-codec-matrix` run them.

## Codec smoke + NIAH matrix

`make smoke-codec-matrix` runs
`scripts/release_e2e/stage6_perf/codec_smoke_runner.sh` over every
`(codec, model)` row of `kv_codec_matrix.toml`. For each row it runs the smoke
prompts through `rmlx baseline`, then the row's NIAH cells.

```bash
make smoke-codec-matrix                                  # every row
make smoke-codec-matrix CODEC=k8v4                       # one codec
make smoke-codec-matrix MATRIX_MODEL=bonsai-8b           # one model
make smoke-codec-matrix RECORD=1                         # record baselines
bash scripts/release_e2e/stage6_perf/codec_smoke_runner.sh --dry-run
```

The variable is `MATRIX_MODEL`, because the Makefile's `MODEL` has a default.
The script also takes `--manifest <path>`. It writes
`scripts/release_e2e/stage6_perf/last_run.json`, which git ignores. It exits 0
only when every row that ran passed.

A row reads its model from `RMLX_TEST_MODEL_BONSAI`,
`RMLX_TEST_MODEL_GEMMA4_E4B` or `RMLX_TEST_MODEL_QWEN36`, and skips when the
variable is unset. Each smoke run takes the Metal claim itself, and each NIAH
run runs under `rmlx claim run`. When another process holds the claim, the row
fails with exit 11 and the refusal names the holder.

### Manifest

`scripts/release_e2e/stage6_perf/kv_codec_matrix.toml` holds one `[[entries]]`
table per `(codec_name, model)`:

| Field | Meaning |
|---|---|
| `codec_name` | display name (`k8v4`, `TurboSym3`, …) |
| `model` | `bonsai-8b`, `gemma4-e4b` or `qwen3.6-moe-8bit` |
| `context_length` | NIAH context in tokens |
| `expected_retrieval_pct` | the baseline; `0.0` means not recorded |
| `smoke_probe_prompts` | prompt names from `smoke_prompts.toml` |
| `skip_reason` | non-empty skips the row |
| `cli_args` | the `rmlx baseline` suffix that selects the codec |
| `niah_filter` | the NIAH test-name filter |

A row passes when the measured retrieval is at least the expected value minus
0.02. With `--record-baseline`, a row whose expected value is `0.0` writes the
measured value into the manifest instead. A measured `0.0` is refused.

### Smoke prompts

`scripts/release_e2e/stage6_perf/smoke_prompts.toml` holds three prompts with
no chat-template markup. A row fails if any prompt fails.

| Name | Prompt | Validation |
|---|---|---|
| `coherence` | "Describe a sunrise in three sentences." | five words in a row; printable ratio ≥ 0.95 |
| `instruction` | "List 3 benefits of regular exercise. Number them 1, 2, 3." | `1`, `2` and `3` in order |
| `multi_turn` | a lighthouse painted red and white; "What color was the lighthouse?" | mentions `lighthouse`, `red` or `white` |

## Sparse-attn calibration runner

`rmlx kv-calibrate --recipe head_budget` loads a model and writes
`<model>/head_budgets.json` per
[`rmlx_loader::head_budgets`](../crates/rmlx-loader/src/head_budgets.rs):

```bash
rmlx kv-calibrate /path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit \
  --recipe head_budget --mass-threshold 0.95
```

`--mass-threshold` takes a value in `[0.50, 1.00]`. The default prompt set is
`prompts/calibration_default.json`; `--prompts <path>` replaces it.

Two tests need no model:

- `cargo test -p rmlx-cli kv_calibrate`: the preflight errors, such as a
  missing `config.json` or an out-of-range `--mass-threshold`.
- `cargo test -p rmlx-loader head_budgets`: the `head_budgets.json` writer,
  reader and validation.

The sparse-attention kernels have their own GPU parity tests under
`crates/rmlx-kv-quant/src/sparse_attn/`.

## Prompt fixture note

`prompts/ssd_bench/structured_regex_gen.json` contains the path
`/home/user/.rmlx/metrics/runs.db-wal` as model input. It is a placeholder.
Prompts are content-addressed, so changing it would change the fixture's hash.
