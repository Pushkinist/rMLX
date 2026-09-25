# rMLX tests: snapshots, variables and CPU gates

This doc covers how model-gated tests find their snapshots, the variables that
steer tests, and the numeric gates that need no model.

## Running the tests

| Target | What it runs |
|---|---|
| `make test` | `cargo test --workspace`, no `--ignored`. A snapshot-gated test skips. |
| `make model-check` | the `rmlx-models`, `rmlx-runtime`, `rmlx-quant` and `rmlx-kv-quant` tests; no model |
| `make model-check-full MODEL=<snapshot>` | `model-check` without `rmlx-kv-quant`, then the five golden-token suites with `MODEL` as `RMLX_KV_TEST_MODEL` |
| `make e2e` | the E2E harness, `crates/rmlx-cli/tests/e2e/` |
| `make gpu-test` | the GPU and Metal tests |

A test that skips reports success. A machine without snapshots, hosted CI
included, runs the default suite green.

## Snapshot resolution

### The models root

`RMLX_O_MODELS_ROOT` is the directory that holds every snapshot under its slug
(`mlx-community__gemma-4-e4b-it-mxfp8`). A bare `cargo test` sees it only if
the shell exports it.

Every `make` target exports it. The precedence is: command-line variable,
then environment, then `.env`, then the repo-local `models/` directory. The
`models/` fallback is exported only when it exists. The two commands below
mean the same:

```bash
RMLX_O_MODELS_ROOT=/tmp/empty make gpu-test CRATE=rmlx-models
make gpu-test CRATE=rmlx-models RMLX_O_MODELS_ROOT=/tmp/empty
```

### The resolvers

Each test resolves its snapshot by one of these rules:

| Rule | Code | Order | A set but wrong value |
|---|---|---|---|
| golden | `common::model_for` | `RMLX_KV_TEST_MODEL` if it serves the test's architecture, then the slug | fails |
| slug first | `common::slug_or_override` | the slug, then the variable the test names | fails |
| slug first, in `src/` | `rmlx_models::test_snapshot::snapshot` | the slug, then a `RMLX_TEST_MODEL_*` variable | skips |
| variable first | `tests/resolved_arch_class.rs`, `crates/rmlx-cli/src/commands/kv_calibrate_tests.rs` | a `RMLX_TEST_MODEL_*` variable, then the slug | falls through to the slug |
| variable only | every other test that names a variable | the variable | skips |

`common` is `crates/rmlx-models/tests/common/mod.rs`. Its two rules share one
probe. It fails when `RMLX_O_MODELS_ROOT` is set but is not a directory, and
when a variable names a path that is not a runnable snapshot. It skips when
nothing is configured, or when the root does not hold the slug.

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
| `RMLX_TEST_MODEL_JINA_V4` | `jinaai__jina-embeddings-v4` | `JinaVLForEmbedding` |
| `RMLX_TEST_MODEL_LAGUNA` | a `z-lab__Laguna-*` snapshot | `LagunaForCausalLM` |
| `RMLX_TEST_MODEL_READERLM_V2` | `mlx-community__jinaai-ReaderLM-v2` | `Qwen2ForCausalLM` |
| `RMLX_TEST_MODEL_QWEN3_VL_30B` | a `mlx-community__Qwen3-VL-30B-Instruct-*` snapshot | `Qwen3VLForConditionalGeneration` |

The architecture column is the resolved class, `Architecture::arch_class()`.
For the Qwen3.5 family it follows the checkpoint's tensors, not its
`architectures[0]`. `tests/resolved_arch_class.rs` pins that. It builds a
snapshot that declares dense and ships MoE tensors, with symlinked weights. The
Qwen-MoE K-side codec guard must still reject a codec on it.

`cache_type_tests.rs` covers the guard's table without weights. Whether
`Architecture::generate_greedy`, `generate_image`, the `ArchGenerator` and
`SpeculativeGenerator` constructors and the speculative per-request path call
it, only snapshot-gated tests check. Deleting one of those calls leaves
`make ci` green.

## Other test-model variables

| Variable | Read by | Rule |
|---|---|---|
| `RMLX_KV_TEST_MODEL` | the golden-token suites, `bitnet_logprobs.rs`, the verifiers of the alignment suites and of `spec_greedy_equivalence.rs` and `spec_sampled_distribution.rs` | golden |
| `RMLX_KV_TEST_MODEL` | `gemma4_kv_cache_equivalence.rs`, `kv_bytes_sample_point.rs`, `cli_flags_e2e.rs`, `projects_toml_e2e.rs` | variable only |
| `RMLX_DRAFT_TEST_MODEL` | the drafters of the alignment suites, `spec_conditioning_residual.rs`, `spec_greedy_equivalence.rs` and `spec_sampled_distribution.rs` | slug first |
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
the test file. The verifier resolves by the golden rule and the drafter by
the slug-first rule. A machine that holds both snapshots runs the suite with no
variable set.

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
measure. The written fixture records the margin. Passing this gate does not
make the new output correct. It shows only that the flip sat at a tie.

## Metal-context `#[ignore]` convention (enforced)

A test that drives the GPU carries `#[ignore]` and runs serialized:

```bash
cargo test --test embeddings_smoke -- --ignored --test-threads=1
cargo test -p rmlx-kv-quant --lib -- --ignored <filter> --test-threads=1
```

`cargo test` runs a binary's tests on parallel threads. A shared Metal context
driven from several of them aborts the whole process:

```
fatal runtime error: Rust cannot catch foreign exceptions, aborting
```

Every other test in the binary dies with it. The abort depends on load, and
each test still passes alone, so the rule is mechanical.

**Which tests carry it:** every test that reaches `Device::Gpu`. A guard that
returns before any device-parameterized op is not a GPU test: pass it
`Device::Cpu` and leave it un-ignored, so the default gate keeps running it.

`make check-gpu-tests-ignored` (`scripts/check_gpu_tests_ignored.sh`) enforces
the rule. It runs in `make ci` and in the hosted `source gates` job. Its scope:

- every workspace member, read from `Cargo.toml`;
- `src/**/*_tests.rs`, `src/**/tests.rs` and `tests/*.rs` in each member;
- `#[test]`, `#[tokio::test]` and `#[tokio::test(…)]`.

A test reaches `Device::Gpu` when the name appears in its body, or in a
module-scope `const … = Device::Gpu`, or in a helper it calls. An unqualified
call binds to a helper in the same file. A `module::helper(..)` call binds to
the scanned file of that module name in the same crate. The gate keys on this
shape and never on the ignore reason's wording, with one exception below.

#### Exempting a device-as-value test

A test that passes `Device::Gpu` to a pure function as a plain value, and never
dispatches Metal, opts out with a line-leading `// gpu-test-gate: exempt` in its
own attribute block. The marker covers that one fn; a copy inside a fn body
exempts nothing. Inside a `macro_rules!` body it covers every cell the macro
generates, so audit such a marker against every invocation.

#### An `#[ignore]` that claims Metal and cannot prove it is fatal

An `#[ignore]` whose reason names a Metal context, on a test that reaches no
`Device::Gpu`, runs under no gate: `make test` skips it as ignored, and
`make gpu-test` skips it as unclassified. The gate fails it until one of three
dispositions is recorded:

1. It drives Metal by a route the scanner cannot follow: declare the route
   (below).
2. It does not touch the GPU: drop the `#[ignore]` and pass `Device::Cpu`.
3. It is ignored for another reason: say that reason instead of claiming Metal.

This one check reads the ignore reason's wording. A Metal-driving test whose
reason never says "Metal" or "GPU" is invisible to it.

#### Declaring a Metal route the scanner cannot follow

```rust
// gpu-test-gate: metal-unscanned  <why the scanner cannot see it>
#[ignore = "GPU Metal: …"]
#[tokio::test]
async fn drives_metal_over_http() { … }
```

The marker follows the placement rules of `exempt` and says the inverse: this
test dispatches Metal but never names the device. The test counts as
GPU-touching, so deleting its `#[ignore]` fails the gate. Two shapes fail:
the marker on a test from which `Device::Gpu` is reachable (a stale marker),
and the marker beside `exempt`.

#### The declared routes, and what covers them

| test | file | route | covered by |
|---|---|---|---|
| `valid_single_vector_200_shape` | `crates/rmlx-server/tests/embeddings_smoke.rs` | HTTP → `embeddings()` → `Device::Gpu` | nothing; run by hand |
| `return_multivector_toggles_shape` | same | same | nothing; run by hand |
| `invalid_dimensions_is_400` | same | same; the 400 comes after a full forward | nothing; run by hand |
| `image_single_vector_200_shape` | same | same | nothing; run by hand |
| `image_multivector_toggles_shape` | same | same | nothing; run by hand |
| `ssd_cache_survives_server_restart` | `crates/rmlx-server/tests/ssd_cache_restart.rs` | spawned `rmlx serve` child | `make e2e` phase 2a runs the same spill → restart → hydrate chain |
| `serve_refuses_to_start_above_the_positional_capacity` | `crates/rmlx-cli/tests/serve_context_ceiling.rs` | spawned `rmlx serve` child | nothing; run by hand |
| `paro_kernel_registration` | `crates/rmlx-models/src/paroquant_msl_tests.rs` | `paro_rotate_kernel()` in a non-scanned source file | the `paro_rotate_identity_roundtrip_*` cells in `make gpu-test` dispatch the same kernel |

Run the embeddings cells by hand:

```sh
RMLX_TEST_MODEL_JINA_V4=/abs/path/to/jinaai__jina-embeddings-v4 \
  cargo test -p rmlx-server --test embeddings_smoke -- --ignored --test-threads=1
```

**Three populations.** `--list` is what `make gpu-test` executes. It is a
strict subset of what the gate enforces, and every run prints the difference:

| population | `#[ignore]` enforced | in `--list` | why not listed |
|---|---|---|---|
| `Device::Gpu` reachable | yes | yes | — |
| macro-generated | yes, at the `macro_rules!` body | no | a cell has no name before expansion, so no libtest filter selects it |
| `metal-unscanned` | yes | no | the gate lists none of them; the runner's per-crate banner check would fail on a snapshot-gated or child-driving test on a host without the snapshot |

`paro_kernel_registration` needs no snapshot and drives no child. It is
unlisted only because the gate lists no `metal-unscanned` test.

#### Macro-generated tests

The gate classifies a `macro_rules!` body that declares `#[test]` as one
synthetic test at its definition site. A `#[ignore]` deleted from the body
fails the gate however many invocations exist. Reachability is traced from the
body like any fn.

These shapes fail closed, reported as `U`, because an unreadable shape looks
compliant:

- a body declaring more `#[test]` than the gate can read back as items: a
  name built by `paste!` or `concat_idents!`, or a `#[test]` on its `fn` line;
- a `macro_rules!` written on one line whose body declares `#[test]`;
- an item whose closing brace is never found;
- an attribute whose closing `]` is never found.

Write a generated fn as `fn $name()` on its own line, with its attributes above
it. End a wrapped attribute on a line whose last significant character is `]`.

#### What the scanner cannot see

Each of these passes the gate silently:

- A signature-only fn whose `where` clause pushes the `;` to a later line. The
  latch closes at the next line indented like the fn, so any `#[test]` it
  swallowed goes unclassified. The fixtures `trait_where_signature` and
  `trait_where_signature_open_hole` pin both outcomes.
- An attribute left open by a raw string or a block comment, closed by a later
  line that ends in `]`. The items between are lost with no report.
- A raw string (`r"…"`, `r#"…"#`) or a `/* … */` block comment on an item's
  opening line: neither is tracked.
- An unqualified cross-file call through a glob import (`use m::*; helper()`).
- A helper in a non-scanned source file. The one instance,
  `paro_kernel_registration`, is declared `metal-unscanned`.
- A `macro_rules!` with a non-brace delimiter: its items are classified, but
  the `U` counters do not cover its body.
- A test generated by a proc macro (`#[rstest]`, `#[test_case]`), or by a
  `macro_rules!` defined in a non-scanned file.

The script's `PARSING` header gives the close-test rules in full.

`make check-gpu-tests-ignored-fixtures`
(`scripts/check_gpu_tests_ignored_fixtures.sh`, in `make ci` and hosted CI) is
the recall test. Each fixture under `scripts/fixtures/gpu_tests_ignored/` is a
synthetic workspace driven through the gate's `--root`. Some are violations and
some are legitimate shapes. Each case asserts the exit code, the violation
marker and the label the gate must name, because the gate has other paths that
also exit 1. The suite runs once under each of `awk`, `gawk` and `mawk` that is
installed, and prints a note when only one is.

### Running them: `make gpu-test`

`make test` passes no `--ignored`, and hosted CI has no Metal. `make gpu-test`
(`scripts/run_gpu_tests.sh`) is the step that runs the GPU tests:

```bash
make gpu-test                                   # every member crate
make gpu-test CRATE=rmlx-kv-quant               # one crate
make gpu-test CRATE=rmlx-kv-quant FILTER=rotor_flash
make gpu-test HALF=codec                        # one half, see below
make gpu-test VALIDATE=0                        # without shader validation
```

It runs exactly the tests `check_gpu_tests_ignored.sh --list` names. It does
not run every `#[ignore]` test: many are ignored for network access, a missing
feature or a doc example. Each name becomes a libtest substring filter, not
`--exact`; an over-match runs a test twice and hides nothing.

It refuses to report OK when:

- a crate executed fewer tests than were classified for it;
- the selection matched no test, or the classification is empty;
- `RMLX_SKIP_GPU=1` is set, since every classified test would return before
  touching Metal;
- another MLX process is live (`pgrep -f 'rmlx serve|mlx_lm|paroquant|omlx'`).

A failing test is never on a known-red list; the runner keeps none. Before
blaming a failure on a change, re-run the same crate and filter on a clean
checkout of the base commit and compare.

The runner reports every red it found before it exits: shader-validation hits,
failing tests, under-matched crates and crates with no validation banner.
`make gpu-runner-selftest` (`scripts/run_gpu_tests_selftest.sh`, in `make ci`
and hosted CI) pins each report against stub crates, with no GPU.

#### A cell that stood down is reported, and it is not a pass

A model-gated cell whose snapshot is absent returns before asserting, and
libtest prints `ok` for it. The runner passes `--nocapture` and harvests each
cell's own notice, `SKIP <test>: <why>`:

```
stood down — these selected GPU tests announced they did not run:
  rmlx-models <test>: <why>

OK: … GPU tests passed across … workspace member(s) … — INCOMPLETE: 1 selected
GPU test(s) stood down and 0 further notice(s) named no test; they asserted
nothing (listed above)
```

A notice is listed only when its name is a classified GPU test of that crate.
Any other notice is counted on the final line and listed nowhere. An unset or
missing `RMLX_O_MODELS_ROOT` also marks the final line INCOMPLETE. A stand-down
has no exit code, so a developer without the weights is not blocked.
`make ci-perf` prints `ci-perf INCOMPLETE` instead of `ci-perf ok` when the
runner's output carries the word.

`make check-named-skip-notices` (`scripts/check_named_skip_notices.sh`, in
`make ci`) enforces the notice at the source line, over the classified GPU
tests:

- a notice names its own test fn, written out or as the exact `{test}`
  placeholder; a notice naming no test or another test fails. A helper that
  stands a cell down takes the caller's name and prints `SKIP {test}:`, since
  no libtest filter reaches a helper;
- in the files that declare those tests, a block that reads an environment
  variable and exits by a `return` carrying no value (`return;`, `return`,
  `return None;`, `return Ok(());`) must print a notice. `RMLX_SKIP_GPU` guards
  are exempt.

The notice's shape lives in `scripts/lib/skip_notice_patterns.sh`, which the
gate and the runner both read. `make check-named-skip-notices-fixtures` is the
gate's recall test.

A cell that resolves through `common::slug_or_override`
(`crates/rmlx-models/tests/common/mod.rs`) or `test_snapshot::snapshot`
(`crates/rmlx-models/src/test_snapshot.rs`) looks up its slug under
`RMLX_O_MODELS_ROOT` first, and its variable is only a fallback. An unset
variable therefore does not stand such a cell down. The golden-token suites
rank `RMLX_KV_TEST_MODEL` first when it serves their architecture; see
"Golden-token suites: how their snapshot resolves". In the integration
harness, a root that is set but missing fails, an absent slug skips, and a
fallback that is used prints `NOTE <test>: …`.

Cells that read only a variable still stand down without it. Examples:
`RMLX_KV_TEST_MODEL` in `gemma4_kv_cache_equivalence.rs` and
`kv_bytes_sample_point.rs`, `RMLX_VL_TEST_MODEL` in
`qwen3_vl_moe_text_parity.rs`, the `RMLX_PROMPT_CACHE_TEST_MODEL_*` pair in
`prompt_cache_cross_model.rs`. A run ends INCOMPLETE unless those are set.

`spec_greedy_equivalence.rs` refuses a mis-paired drafter before either model
loads, with a named notice. `declared_kind` checks that the sidecar declares the
kind the pair's loop drives. `declared_quant_mode` checks that the sidecar is
quantized like its verifier. For the two-model pair, `vocab_pairing` in
`crates/rmlx-models/src/speculative/mod.rs` compares the two tokenizers id by
id; `load_speculative` calls the same function.

#### Splitting the GPU suite by what it guards

`scripts/gpu_test_halves.sh` is the one producer of the partition. It prints
`half<TAB>crate<TAB>test` for every classified test:

> A classified GPU test is in the **`codec`** half if and only if its declaring
> file is under a workspace member that `rmlx-models` depends on, or under
> `crates/rmlx-models/src/`, or **selects a KV codec** — names
> `DEFAULT_KV_QUANT`, or a `KvQuant::<V>` whose `<V>` is a variant of
> `ALL_KV_QUANTS` other than `None`. Every other classified GPU test is in the
> **`rest`** half.

- Every fact is read from the tree, never from a test name. A test that moves
  file can move half; a renamed test keeps its half.
- The codec clause reads the file, not the test. It reads code only: comments
  and string bodies are blanked through `scripts/lib/awk_text.sh`.
- The codec names come from `ALL_KV_QUANTS` in
  `crates/rmlx-kv-quant/src/quant.rs` of this checkout, even under `--root`.
  An empty derived set is exit 2.
- A file that names only `KvQuant::None` has pinned the codec off; it is in
  `rest`.

`HALF=codec|rest` on `make gpu-test` and `make ci-perf` passes `--half` to
the runner. The Makefile holds it to three rules:

1. A half-run's last line reads `ci-perf <half>-half ok — NOT the whole gate`,
   never `ci-perf ok`. This is the defence that holds however `HALF` arrived.
2. `HALF` must be exactly `codec` or `rest`; any other value, the empty string
   included, is a Make error.
3. Only a command-line `HALF` counts. An exported one has origin `environment`
   and is ignored. `MAKEFLAGS=HALF=codec` still counts, which rule 1 covers.

Under `--half` the runner holds:

- A classified test in no half, or a producer row naming no classified test,
  is a refusal naming the test.
- The census expectation is the half's own slice of the one pin, keyed on
  `(crate, test)`. An entry of the other half is silent. The pin's validity
  check still reads the whole classification.
- A stand-down inside a half still ends it INCOMPLETE.
- The two halves' selections and census expectations sum to the whole run's.

Each rule is a case under `THE HALVES` in `scripts/run_gpu_tests_selftest.sh`.
No case can see a codec change that passes the codec half and breaks a test in
the rest half. The whole gate on `main` finds it one merge later.

#### Where it runs: `make ci-perf`, not `make ci`

`make ci-perf` is the only shared gate that runs the GPU tests. In order:

1. `run_gpu_tests.sh --preflight` checks the environment and runs no test:
   `RMLX_SKIP_GPU` unset, no competing MLX process, a non-empty
   classification. It fails before the long step.
2. `make test-perf` runs the workspace under `release-perf`.
3. The GPU suite runs, and the last line reports its verdict: `ci-perf ok`, or
   `ci-perf INCOMPLETE` when the runner printed the marker.

It is not in `make ci`: the suite needs the Metal context to itself (CLAUDE.md
hard rule 8) and takes too long for every commit.

`ci-perf` calls the runner directly, not `make gpu-test`. Make passes
command-line variables to sub-makes, so `CRATE=` or `VALIDATE=0` would narrow
or disarm the gate. The coverage check cannot see a narrowed run, because
`--crate` shrinks the classified set with the executed one. `HALF` is the one
variable `ci-perf` takes, under the rules above.

The GPU suite builds under `dev`, with debug assertions live, while
`test-perf` builds under `release-perf`. So no gate runs a `Device::Gpu` test
with debug assertions off (CLAUDE.md hard rule 9). The `dev` build is not
shared with `test-perf`. `scripts/target_gc.sh` protects `release-perf` and
prunes stale profiles, so a run after `make target-gc` can pay a cold `dev`
build.

While iterating, run `make gpu-test` narrowed with `CRATE=` / `FILTER=`.

### Metal shader validation (on by default here)

An out-of-bounds device store from a Metal kernel is dropped. The command
buffer completes, `cb.error` is `nil`, cargo exits 0, and the tests over the
buffer still pass. So `make gpu-test` runs every pipeline under Metal shader
validation and scans the output for a diagnostic:

```
Invalid device store at offset 4000064, executing kernel function: "custom_kernel_rmlx_q8_quantize"
```

- **The exit code is not the signal.** With validation on, cargo still exits 0.
  The runner scans the text, anywhere on a line: the layer writes while libtest
  is mid-line.
- **The runner owns the environment.** It pins every
  `MTL_SHADER_VALIDATION_*` knob it relies on, `REPORT_TO_STDERR=1` included;
  that one defaults to 0 and sends reports to Unified Logging
  (`man MetalValidation`).
- **The banner is asserted per crate.** A crate that never printed
  `Metal GPU Validation Enabled` ran uninstrumented and fails. This usually
  means it did not build.
- **A positive control runs first.**
  `crates/rmlx-kv-quant/src/shader_validation_canary.rs`, behind the
  `shader-validation-canary` feature, stores out of bounds on purpose. The run
  refuses to trust a clean scan unless the canary's diagnostic matched. The
  canary dispatches only while validation is on, and is not in the population
  `make gpu-test` selects.

MLX owns the allocator and reports buffers as `<unnamed>`. The kernel function
name, `custom_kernel_` plus the rMLX kernel name, is the attribution.

Validation costs throughput, so it stays on this target and off every cell
whose numbers are recorded. `VALIDATE=0` opts out. Never draw a conclusion
about model output from a run under Metal shader validation.

A diagnostic names a *load* or a *store*. A store is a dropped write; a load
matters only if the kernel keeps the lanes it filled. The failure banner
prints the access mix per diagnostic. A clean scan does not prove that nothing
read out of bounds. The layer bounds against the `MTLBuffer`, not the array,
and MLX recycles buffers from size buckets.

The pin accepts one diagnostic: MLX's own
`affine_qmm_t_splitk_bfloat16_t_gs_64_b_{4,8}_alN_false`, loads only. On a
host with every snapshot, armed cells report hits the pin does not name, until
the pin is derived again.
`QuantizedBlockLoader::load_safe` in `mlx/backend/metal/kernels/quantized.h`
bounds its row index against the tile's column extent. A transposed quantized
matmul whose `N` is not a multiple of the output tile width then reads past the
packed weight and scales. The header of `scripts/gpu_validation_census.txt`
gives the argument that the loaded lanes never reach the output.

#### The census pin

`scripts/gpu_validation_census.txt` pins the accepted hits, so a standing
diagnostic from a kernel this repo does not compile does not keep every run
red. One entry per `(kernel, kind, crate, test)` carries that test's own count
and the reference to the analysis. The file's header states the fields.

For each `(crate, kind, kernel)` the runner expects the sum of the pinned
counts whose test ran. A test the selection dropped, and a test that printed
its own named `SKIP`, contribute 0. So a narrowed run is compared exactly, and
an entry's test must print a named notice when it skips. A test that misses
one of several checkpoints and asserts on the rest prints `note <test>: …`,
not `SKIP`. Its entries stay expected. The missing checkpoint's kernel reports
`no longer fires` when no other test that ran shares it, and
`count moved down` otherwise.

| observed | verdict |
|---|---|
| the expectation exactly, nothing else | pass: `census matches the pin`, with the accepted entries |
| a kernel the pin does not name | fail: `not pinned: N <kind> "<kernel>" in <crate>` |
| above the expectation | fail: `count moved up: … expected N, observed M` |
| below the expectation | fail: `count moved down: …` |
| nothing where the expectation is positive | fail: `no longer fires: …` |
| any store | fail: `never accepted: …` |
| an entry whose test was not selected, or skipped | pass: `census NOT enforced in full`, naming the entry |

A hit in another crate than its entry names reads as `not pinned` there and
`no longer fires` where it was pinned.

The pin file is checked as it is read. Each defect is a failure:

| pin defect | reason reported |
|---|---|
| not six `\|`-separated fields | `line N: expected 6 fields — …` |
| count not a positive integer | `line N: count '<x>' is not a positive integer` |
| a kind naming a store | `line N: a store is never pinnable — …` |
| a test that is not a classified GPU test of that crate | `line N: <crate> has no classified GPU test '<test>'` |
| the same kernel, kind and test twice | `line N: … is pinned twice …` |
| the file missing | `<path> not found — the census pin is tracked; …` |

`make gpu-runner-selftest` pins every verdict and every pin defect against
stub crates. One case parses the tracked pin against the real classifier, so
a malformed pin or a renamed test fails `make ci`.

**Deriving a count.** Run the test alone
(`make gpu-test CRATE=<crate> FILTER=<test>`). Or attribute a full run's
diagnostics to the test libtest last announced; the suite runs
`--test-threads=1 --nocapture`. Split each output line before matching, and
count the tail of a `test <name> ... ` line too: a test's first diagnostic
often lands there.

**Changing the pin** takes one of two things: an upstream reference showing
the defect is not ours and does not reach our output, or an analysis of the
same standard. That means the symbol's provenance, a load/store census over
every diagnostic, and a proof that the loaded lanes do not reach the output. A
count that came in low is re-derived from a full run, never edited to fit.

**A change derives its own entry, in the same change.** A change that adds a
GPU test, or makes one perform a new load, takes its own count and writes the
entry with it. Hosted CI has no Metal, so a missing entry surfaces only at the
next `make ci-perf`.

To exercise the skipped-entry path on a host with snapshots, point the root at
an empty directory: `make gpu-test CRATE=rmlx-models
RMLX_O_MODELS_ROOT=/tmp/empty`. Every pinned test skips, and each entry is
reported as not enforced.

### `#[ignore]` is not a place to park a broken test

An ignored test runs only when someone asks for it, so a failure can sit
unseen. When a deliberate behaviour change makes an assertion stale, re-point
it at the new contract. Then mutation-check it: revert the change, and the
repaired test must go red. Do not relax it.

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
| `RMLX_SKIP_GPU` | `1` | GPU tests return at once, `--ignored` or not |
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
the value `1`. The GPU tests of that crate call it first and return.
`crates/rmlx-models/tests/sparse_attn_dispatch.rs` has its own copy.

```bash
RMLX_SKIP_GPU=1 cargo test -p rmlx-kv-quant -- --include-ignored
```

No test writes `RMLX_SKIP_GPU`. The GPU tests read it without the env lock, so
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

The kernel gates (`RMLX_TURBO_FLASH`, `RMLX_FUSED_QK`, `RMLX_SPARSE_ATTN`,
`RMLX_PLANAR_FLASH_DECODE`, `RMLX_ROT_K_FUSED`) seed a
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
  index past it. Each GPU parity test passes its codec's tolerance.

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

Every cell asserts that the decode recovers the needle. It also reads the
kernel's dispatch counter around the decode, under the process-default
dispatch policy. With the kernel on and the cell marked `Reachable`, a
TurboFlash cell asserts that the kernel ran. A planar cell asserts that it did
not: the live bf16 K seed keeps it dormant. Every other cell asserts no
dispatch.

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
are not in `make gpu-test`. A macro cell has no name before expansion, so no
libtest filter selects it. Each cell also loads a snapshot and prefills up to
32k tokens. The NIAH driver and `make smoke-codec-matrix` run them.

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
variable is unset. Before each smoke run and each NIAH run, the runner kills
competing MLX processes and deletes the claim files.

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
