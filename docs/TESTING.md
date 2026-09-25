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
| `RMLX_TEST_MODEL_MEDGEMMA` | `mlx-community__medgemma-1.5-4b-it-8bit` | `Gemma3ForConditionalGeneration` |
| `RMLX_TEST_MODEL_QWEN36` | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| `RMLX_TEST_MODEL_QWEN36_PARO` | `z-lab__Qwen3.6-27B-PARO` | `Qwen3_5ForConditionalGeneration` (dense PARO) |
| `RMLX_TEST_MODEL_ORNITH_9B` | `sahilchachra__ornith-1.0-9b-mxfp8-mlx` | `Qwen3_5ForConditionalGeneration` (dense) |
| `RMLX_TEST_MODEL_BONSAI` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |
| `RMLX_TEST_MODEL_DR_VENUS` | `z-lab__DR-Venus-*` | `Qwen3ForCausalLM` |
| `RMLX_TEST_MODEL_JINA_V4` | `jinaai__jina-embeddings-v4` | `JinaVLForEmbedding` |
| `RMLX_TEST_MODEL_LAGUNA` | `z-lab__Laguna-*` | `LagunaForCausalLM` |
| `RMLX_TEST_MODEL_READERLM_V2` | `mlx-community__jinaai-ReaderLM-v2` | `Qwen2ForCausalLM` |
| `RMLX_TEST_MODEL_QWEN3_VL_30B` | `mlx-community__Qwen3-VL-30B-Instruct-*` | `Qwen3VLForConditionalGeneration` |

The `Arch` column is the **resolved** class (`Architecture::arch_class()`), which
for the Qwen3.5 family follows the checkpoint's tensors rather than its
`architectures[0]`. `tests/resolved_arch_class.rs` pins that distinction and
builds a deliberately mislabelled snapshot (dense declaration, MoE tensors) to
prove the Qwen-MoE K-side codec guard still fires. It symlinks the weights, so
the fixture costs no disk; it is `#[ignore]`d only because it loads real
snapshots.

> **Known coverage gap.** The *invariant table* is covered weights-free
> (`cache_type_tests.rs`, including
> `validate_resolved_qwen3_5_dense_and_moe_strings_diverge`, which pins that the
> two Qwen3.5 strings give opposite verdicts). Whether the enforcing call sites
> actually consult it — `Architecture::generate_greedy` / `generate_image`, the
> `ArchGenerator` and `SpeculativeGenerator` constructors, and the speculative
> per-request seam — is exercised **only** by snapshot-gated tests. Deleting one
> of those calls leaves `cargo test --workspace` and `make ci` green. Run the
> `--ignored` suites above before trusting a change to those seams.

## Specialised test-model variables

Some integration tests use dedicated snapshot variables instead of the family
variables above:

| Variable | Used by | Purpose |
|----------|---------|---------|
| `RMLX_TEST_MODEL` | `rmlx-server/tests/ssd_cache_restart.rs` | Generic single-model override for the SSD-restart smoke test. |
| `RMLX_KV_TEST_MODEL` | `gemma4_kv_cache_equivalence.rs`, `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs`, `qwen3_5_mtp_drafter_alignment.rs`, `qwen3_5_eagle3_alignment.rs`, `qwen3_5_two_model_alignment.rs`, `spec_greedy_equivalence.rs`, `projects_toml_e2e.rs`, `cli_flags_e2e.rs`, and as the single-model override for the golden-token suites | Model snapshot for KV-cache equivalence and drafter-alignment tests. Typically set to a Gemma4-e4b path; the Qwen3.5-family alignment tests take a **verifier** here instead (see below). |
| `RMLX_DRAFT_TEST_MODEL` | `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs`, `qwen3_5_mtp_drafter_alignment.rs`, `qwen3_5_eagle3_alignment.rs`, `qwen3_5_two_model_alignment.rs`, `spec_greedy_equivalence.rs`, `spec_sampled_distribution.rs` | Draft model snapshot path. Used alongside `RMLX_KV_TEST_MODEL` for speculative-decode alignment tests. |
| `RMLX_VL_TEST_MODEL` | `qwen3_vl_moe_text_parity.rs` | Vision-language model snapshot for VL text-parity tests. |
| `RMLX_PROMPT_CACHE_TEST_MODEL_A` / `_B` | `rmlx-models/tests/prompt_cache_cross_model.rs` | **Two** snapshots of the same architecture with the same KV shape but different weights — the prompt cache is one static per arch, and this pair is what shows whether its key separates two resident models. `mlx-community__gemma-4-e2b-it-mxfp8` + `mlx-community__gemma-4-E2B-it-qat-4bit` fit (both `Gemma4ForConditionalGeneration`, 35 layers x 1 KV head x head_dim 256). Same-shape matters: a shape mismatch would fail for the wrong reason. Different weights matter: identical outputs make the comparison vacuous, and the test refuses rather than passing. |

The three Qwen3.5-family alignment suites **return silently when their two
variables are unset** (and the EAGLE-3 / two-model ones also when the drafter
handed to them is of the wrong kind), so an unnamed consumer here is a gate that
passes while never running. The pairs their thresholds are calibrated against:

| Test | `RMLX_KV_TEST_MODEL` (verifier) | `RMLX_DRAFT_TEST_MODEL` (drafter) |
|---|---|---|
| `qwen3_5_mtp_drafter_alignment.rs` | `mlx-community__Qwen3.8-27B-mxfp8` | `mlx-community__Qwen3.8-27B-MTP-mxfp8` |
| `qwen3_5_eagle3_alignment.rs` | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Dogacel__specdrift-qwen3.6-35b-a3b-eagle3` |
| `qwen3_5_two_model_alignment.rs` | `mlx-community__Qwen3.8-27B-mxfp8` | `sahilchachra__ornith-1.0-9b-mxfp8-mlx` (a full model, not a drafter head — both halves must be GDN hybrids sharing a vocabulary) |

`two_model_stochastic.rs` is the two-model loop's other acceptance rule: it runs
`spec_generate_greedy` at `temperature 1.0` on `mlx-community__gemma-4-e4b-it-mxfp8`
drafted by `mlx-community__gemma-4-e2b-it-mxfp8`, both resolved by slug from
`RMLX_O_MODELS_ROOT`, and pins that one seed reproduces one sequence while a
second seed and `temperature 0` do not — the Leviathan loop is sampling, and it
is the loop that ran. It is the only gate on that loop; every alignment suite
runs greedy. It resolves by slug, so `make gpu-test` runs it wherever the
snapshots are; under `scripts/run_gpu_tests.sh` with shader validation on it
passes in about 80 s and produces **zero** validation hits, which is why it has
no entry in `scripts/gpu_validation_census.txt` — the runner fails on an
unpinned hit, so a clean pass is the evidence, not the absence of a pin.

Point either at a different pair and re-measure both arms before reading a
failure as a regression.

`dflash2_loader.rs` takes no model variable at all: it resolves
`z-lab__Qwen3.8-27B-DFlash2` by slug from `RMLX_O_MODELS_ROOT`, like
`two_model_stochastic.rs`, so `make gpu-test` runs it wherever the snapshot is.
It goes through `tests/common`'s `slug_snapshot` at `Role::Sidecar` — the role
that asks only for the files a drafter checkpoint carries, since it is decoded
with the verifier's tokenizer and ships none of its own. Each of its tests
passes its own function name, so a stand-down announces
`SKIP <that test>: <why>` and `run_gpu_tests.sh` can attribute it; a notice
naming the file instead is counted as unattributable and listed nowhere. It
loads the weights, asserts the names and shapes, and runs the forward and the
selector against the committed reference. It has no entry in
`scripts/gpu_validation_census.txt` — the runner fails on an unpinned hit, so a
clean pass is the evidence, not the absence of a pin.

`spec_sampled_distribution.rs` is the sidecar half of the same question, above
temperature 0, and it is a distributional one: not what the arm emitted but what
it drew from. Every emitted token carries a surprise under the distribution the
plain path would have drawn from at the same prefix, and if the arm draws from
that distribution the stream's total surprise has a mean and a variance those
distributions fix exactly — so the verdict is a `z` with no threshold measured
on a healthy engine first. It runs
`mlx-community__gemma-4-e2b-it-mxfp8` drafted by
`mlx-community__gemma-4-E2B-it-assistant-bf16`, both resolved by slug, over five
prose questions, and it runs a second arm at temperature 0 as a positive control
that the run is red unless it *refuses*: a verifier that is nearly certain at
every position passes under every acceptance rule, so a prompt set with no
evidence in it must report that rather than a pass. Measured, +0.58 for the
sampled arm and -8.70 for the control. It produces zero shader-validation hits
and so has no census entry.

Its `RMLX_DRAFT_TEST_MODEL` override names the drafter; a named path that is not
a snapshot fails, and a models root that does not hold the slug skips. The four
CPU cases in the same file need no snapshot at all and pin the statistic's power
against a greedy stream, a stream drawn at the wrong temperature and one drawn
without the request's filters — the last of which the surprise test does not
refuse on its own, which is why the file carries a second oracle over the tokens
that carry no target mass.

`spec_greedy_equivalence.rs` asks a different question from the three alignment
suites: not
whether the round loop keeps the verifier's state consistent for a while, but
whether the run produces the answer the verifier produces alone, over 256 tokens
and every prompt the file carries. Its oracle is where the two arms first differ
and how sure the verifier was there — a rank in the reference arm's own margin
distribution, so one ceiling covers two models whose logits are not on the same
scale. It is documented in full in `docs/SPEC_ANSWER_EQUIVALENCE.md`, including
why the obvious oracle (how much of one answer the arms share) cannot be
thresholded at all.

Unlike those suites, its **assistant pair resolves both halves by slug** from
`RMLX_O_MODELS_ROOT`, so `make gpu-test` runs that pair on a machine holding the
snapshots and `run_gpu_tests.sh` reports a machine without them as INCOMPLETE.
The other five pairs are the exceptions and are not gated: their
drafter comes from `RMLX_DRAFT_TEST_MODEL` or the pair does not run, because
their verifiers' quantized matmuls trip the shader-validation census (see the
table below and `docs/SPEC_ANSWER_EQUIVALENCE.md`). That one variable names one
drafter, so a pair whose loop does not drive the kind that snapshot declares
stands down naming both. Its drafter goes through the same
`slug_snapshot` at `Role::Sidecar` that `dflash2_loader.rs` does — one copy of
the rules, and the role a drafter checkpoint can satisfy — and every stand-down
names the test function it happened in, so `run_gpu_tests.sh` can attribute it.
The verifier goes through the golden harness's own resolver
(`common::model_for`); `RMLX_DRAFT_TEST_MODEL` overrides the drafter. Both
`-e2b-` and `-e4b-` assistant snapshots declare the same architecture, so the
harness's arch stand-down cannot separate them: the drafter's
`backbone_hidden_size` is checked against the verifier's width before the drafter
is loaded, and a mismatched pair skips with that reason rather than panicking in
the loader. The two-model pair has the same problem in a different form —
`two_model` is inferred from the architecture registry, which every full model
satisfies — and the declared vocabulary is what separates those.

| Pair | verifier | drafter | selected by |
|---|---|---|---|
| assistant | `mlx-community__gemma-4-e2b-it-mxfp8` | `mlx-community__gemma-4-E2B-it-assistant-bf16` | slug |
| recurrent | `mlx-community__Qwen3.8-27B-mxfp8` | `mlx-community__Qwen3.8-27B-MTP-mxfp8` | `RMLX_DRAFT_TEST_MODEL` only |
| block | `mlx-community__Qwen3.8-27B-4bit` | `z-lab__Qwen3.8-27B-DFlash2` | `RMLX_DRAFT_TEST_MODEL` only |
| adaptive | `mlx-community__Qwen3.6-35B-A3B-8bit` | `z-lab__Qwen3.6-35B-A3B-DFlash` | `RMLX_DRAFT_TEST_MODEL` only |
| restricted-vocabulary | `mlx-community__Qwen3.6-35B-A3B-8bit` | `Dogacel__specdrift-qwen3.6-35b-a3b-eagle3` | `RMLX_DRAFT_TEST_MODEL` only |
| two-model | `mlx-community__Qwen3.8-27B-mxfp8` | `sahilchachra__ornith-1.0-9b-mxfp8-mlx` | `RMLX_DRAFT_TEST_MODEL` only |

The assistant pair produces **zero** Metal shader-validation hits and so has no
entry in `scripts/gpu_validation_census.txt` and needs none. The other pairs'
verifiers drive MLX's mxfp8 or affine quantized matmul and a narrowed run reports
1344 invalid loads from it — the same `load_safe` bound the census already records
for the affine instantiation, in a kernel this repo does not compile. The census
pins one exact count per test and a count from a 256-token generation is not
stable across a prompt change, so those pairs are named rather than slug-resolved
and `make gpu-test` reports them as skipped. See
`docs/SPEC_ANSWER_EQUIVALENCE.md`.

`dflash_drafter_alignment.rs` is **not** one of the alignment suites above and
does not gate the
same property. It asserts that the drafter's round-0 first-block proposal aligns
with the verifier's greedy continuation (`accept > 0`) and that the live loop
emits coherent prose — a round-0 check, taken before any partial-accept rollback
has happened. It cannot see a rollback that corrupts the verifier state part-way
through a run. What does, for DFlash 1 on a GDN hybrid, is
`the_adaptive_round_loop_reproduces_plain_greedy` in `spec_greedy_equivalence.rs`.

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

**Precedence through `make`:** command-line variable > environment > `.env` >
the repo-local `models/` fallback. A makefile assignment normally outranks the
environment, so the `-include`d `.env` used to win over a shell export and the
run would quietly use the `.env` path while looking redirected. The Makefile now
captures the environment value before the include and restores it after, so
these two mean the same thing:

```bash
RMLX_O_MODELS_ROOT=/tmp/empty make gpu-test CRATE=rmlx-models
make gpu-test CRATE=rmlx-models RMLX_O_MODELS_ROOT=/tmp/empty
```

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

When env vars are unset, snapshot-gated tests **skip** with an `[SKIP]` or
`tracing::warn!` message and report success. The test suite is always green on
machines without model snapshots (including CI).

Absence is not the same as a wrong pointer. `RMLX_KV_TEST_MODEL` **naming** a
directory which is not a snapshot — a typo, or a path a snapshot has since moved
out of — is a hard failure in the golden-token suites (below), not a skip.
Skipping there is how a stale export turns into a green run that asserted
nothing.

The mirror image is a step whose result depends on the machine rather than on
the code. `make ci` contains shell gates as well as `cargo test`, and one of
them — `scripts/perf_ab_selftest.sh` — used to inherit `perf_ab.sh`'s
host-quiescence and Metal-exclusivity preconditions while checking a property
that has nothing to do with runtime, so an `rmlx serve` left running failed 27
of the 48 cases it then had. A gate that fails for the environment trains
contributors to re-run it until green, which does the same damage as one that
cannot fail. The
fix was to split the two kinds of precondition rather than to loosen a
threshold; see `docs/PERF_BASELINE.md` "`--synthetic-arms` is not an escape hatch". The same
boundary now covers `scripts/bench_llama_ab_selftest.sh`, whose verdict cases
used to resolve their expected exit code from the run's own output — an
expectation that agrees with whatever happened cannot catch anything. Both
suites count, rather than claim, how many of their cases could reach this
machine, and fail when that count is not zero.

---

## Golden-token suites: how their snapshot resolves

`crates/rmlx-models/tests/{bonsai,gemma4,qwen3,bitnet,medgemma}_golden_tokens.rs`
each pin a 32-token temp=0 decode of one architecture against a committed
fixture under `tests/fixtures/`. Each covers ONE arch and names its own snapshot
by slug. `tests/common/mod.rs` reads exactly **two** variables:

1. `RMLX_KV_TEST_MODEL`, **for the one golden whose architecture it serves**.
   Pointed at another architecture it is not a statement about this golden, and
   resolution falls through to step 2 rather than standing the golden down.
2. the golden's snapshot **slug** under `RMLX_O_MODELS_ROOT`.

Step 2 is what arms these gates by default, and **an operator normally sets
neither**. Every `make` target exports `RMLX_O_MODELS_ROOT` when it resolves, so
on a machine holding the snapshots `make gpu-test` and `make ci-perf` run every
golden whose model is on disk. Before it existed, a golden needed
`RMLX_KV_TEST_MODEL` — which those targets do not set — so all of them returned
before asserting and libtest reported `ok`. A committed fixture that nothing
compares against is a fixture nobody maintains.

The fall-through in step 1 is not a nicety. `RMLX_KV_TEST_MODEL` is not a
golden-only variable: `gemma4_kv_cache_equivalence.rs`, `cli_flags_e2e.rs` and
`projects_toml_e2e.rs` all require it, typically at a Gemma4-e4b path. Were the
override to make non-matching goldens skip, a developer with it exported would
disarm four of the five on every run — the original defect, surviving for
exactly the developer who most needs these gates. Ranking the slug *first*
instead would break the other direction: `RMLX_REGEN_GOLDENS=1
RMLX_KV_TEST_MODEL=<path>` would record the fixture from the slug snapshot and
silently ignore the named one.

Reach for `RMLX_KV_TEST_MODEL` in exactly two situations: recording a fixture
(`RMLX_REGEN_GOLDENS=1`), and comparing one golden against a snapshot that is
not the slug under your models root. Each golden is its own test binary, so
`RMLX_KV_TEST_MODEL=<path> cargo test -p rmlx-models --test bonsai_golden_tokens
-- --ignored` retargets that one deliberately and reaches no other golden.

**The per-architecture `RMLX_TEST_MODEL_*` family is deliberately not consulted
by the goldens.** Those variables mean "a snapshot of this family for the smoke,
template and NIAH suites", and the workflow two sections above exports the three
primary ones persistently for a whole `cargo test --workspace`. A golden is a
byte-exact fixture over ONE checkpoint's weights, so letting a shell export steer
it turns any same-family substitution — a QAT rebuild, a re-quantized sibling —
into a token mismatch indistinguishable from a decode regression, and the
architecture check below cannot separate the two because the substitute passes
it. If a snapshot lives outside your models root, symlink it in under its slug:
one action, and every other slug-addressed consumer (`make e2e`,
`scripts/perf_canary.sh`, the bench scripts) picks it up too.

The run / skip / fail rule:

| configuration | outcome |
|---|---|
| snapshot resolves, arch matches | **run** the assertion |
| `RMLX_KV_TEST_MODEL` names a different architecture | **fall through** to the slug, and say so |
| ...the same, while `RMLX_REGEN_GOLDENS` is set | **fail** — see below |
| nothing configured, or an existing models root that does not hold this slug | **skip** — a developer without the weights cannot run the gate |
| the models root holds a half-written slug directory | **skip** — an interrupted download is an absence, not a wrong pointer |
| `RMLX_KV_TEST_MODEL` names a path that is not a runnable snapshot | **fail** |
| `RMLX_KV_TEST_MODEL` names a snapshot whose `config.json` is unreadable | **fail** — a named directory with a broken config is a broken pointer, not another architecture |
| `RMLX_O_MODELS_ROOT` is set but is not an existing directory | **fail** — one keystroke disarms all five gates |
| the slug under the models root is a snapshot of the wrong arch | **fail** |

"Runnable" means the directory holds every file the caller opens **by name**,
and which files those are depends on what the caller will do with it. The probe
takes that as a `Role`:

| file | opened by | `Standalone` | `Sidecar` |
|---|---|---|---|
| `config.json` | `model_arch`, `arch::load_model`, every `<Kind>Drafter::load` | required | required |
| `tokenizer.json` | `run_golden_test`'s `Tokenizer::from_file` | required | — |
| `model.safetensors.index.json` **or** `model.safetensors` | `rmlx_loader::load_shard_index`, which tries them in that order and errors if neither exists | required | required |

`Sidecar` is the drafter case, and it exists because a drafter has no tokenizer:
it proposes ids for a verifier and is decoded with the verifier's, and
mlx-community ships those snapshots without one. Requiring a `tokenizer.json` of
a drafter turned a checkpoint sitting on disk into an absence — which is a skip,
and a skip in `spec_greedy_equivalence.rs` reads exactly like the equivalence
holding. `two_model_stochastic.rs` resolves both of its models as `Standalone`,
because there both sides are full models and the pair is loaded through
`load_speculative`, which reads a tokenizer from each.

The weight entrypoints are not padding. A download writes the small JSON files
first and the multi-GB shards last, so `config.json` + `tokenizer.json` + no
shards is the *modal* half-written snapshot — and accepting it converted the
intended verdict for a partial download (skip, so a developer without the
weights is not blocked) into a panic several frames deeper.

**Recording is stricter than checking.** With `RMLX_REGEN_GOLDENS` set, an
override pointed at another architecture is a hard failure rather than a
fall-through: writing a committed fixture from a snapshot you did not name,
while the one you did name is discarded, gives that golden untraceable
provenance — and regenerating the whole set under one override would give each
fixture a different origin with nothing said about it. When the override does
serve the golden, recording proceeds normally. On the read path the fall-through
is announced on stderr (`NOTE <test>: … using <path> instead`) rather than
happening silently.

`make ci` runs none of them either way: the goldens are `#[ignore]`d for the
Metal context, and `make ci` passes no `--ignored`. `make gpu-test` /
`make ci-perf` are where they execute — `scripts/check_gpu_tests_ignored.sh`
classifies them as GPU tests through the cross-file `common::run_golden_test`
helper, and `scripts/run_gpu_tests.sh` runs everything that classifier names.

libtest discards a passing test's output, so a golden that *skipped* prints its
reason into a stream a bare `cargo test` does not show. `make gpu-test` and
`make ci-perf` pass `--nocapture` and report every stand-down by name — see *A
cell that stood down is reported* below. Running the golden by hand, add the
flag yourself when you need to see which ones stood down:

```bash
cargo test -p rmlx-models --test bonsai_golden_tokens -- --ignored --nocapture
```

### Recording a fixture, and the gate on overwriting one

`RMLX_REGEN_GOLDENS=1` makes the test write the fixture instead of asserting it.
A golden updated to match whatever the tree produces today gates nothing, so
**overwriting a fixture whose ids changed is itself gated**:

1. The harness decodes as usual and reads the committed fixture.
2. If the ids are unchanged, or there is no committed fixture, it writes.
3. If they changed, it re-decodes once with `top_logprobs_k = 2` and measures
   the top-2 logprob gap at the first differing index.
4. It writes only when that gap is `<= REGEN_MAX_TIE_MARGIN` (0.10) — a step the
   model had no real preference at. Otherwise it **panics with `REFUSED`**,
   naming the index, both ids and the measured margin.

Refusals are deliberate dead ends, not obstacles to route around: a token count
change is refused at any margin, and a margin that cannot be measured — a
missing step, absent logprobs, or a probe run that decodes a different id, i.e.
non-determinism — is refused too. A gate that waves through what it could not
check is the shape this harness exists to remove.

The written fixture's reason line carries the margin, so a regenerated golden
records *why* it moved. That matters because a regenerated golden with no stated
reason is indistinguishable from a hidden regression.

**This gate does not tell you the new output is correct** — only that the flip
sat at a tie the engine's dtype could not resolve. Deciding a fixture is stale
rather than regressed still needs evidence from outside the harness: a bisect to
the commit that moved it, a reference comparison, and coherent decoded text.

### Why this is not the only snapshot resolver

Three other suites resolve snapshots their own way, and the difference is
deliberate rather than drift. What a suite asserts decides what it may accept:

| suite | resolves from | on a set-but-wrong value |
|---|---|---|
| golden-token (`tests/common/mod.rs`) | `RMLX_KV_TEST_MODEL` + slug | **fails** |
| `tests/niah_long_context.rs` | `RMLX_TEST_MODEL_*` only | skips |
| `tests/resolved_arch_class.rs` | `RMLX_TEST_MODEL_*`, then slug | skips |
| `crates/rmlx-cli/src/commands/kv_calibrate_tests.rs` | `RMLX_TEST_MODEL_*`, then slug | falls through to the slug |

The goldens are the strict case because they are the only ones pinning **exact
bytes from one checkpoint**. The other three make semantic assertions — a needle
is retrieved, an architecture resolves to the expected class, prompts clear a
token floor — which any snapshot of the right family satisfies. That is also why
they may read the per-architecture `RMLX_TEST_MODEL_*` variables and the goldens
may not: a same-family substitute is fine for a semantic assertion and fatal for
a byte-exact one.

Two consequences worth knowing rather than discovering:

* A typo'd `RMLX_TEST_MODEL_BONSAI` panics in `rmlx-models` (if it also breaks a
  golden's root) but only skips in `rmlx-cli`. The suites disagree because their
  assertions do.
* **`niah_long_context.rs` has no slug fallback, so every NIAH cell stands down
  unless its variable is set.** That is the same silent-skip shape the goldens
  just left, and it is deliberately not fixed here: arming the resolution would
  change nothing that runs.

  The NIAH cells are macro-generated, and the two populations they belong to are
  now split on purpose (see *Why NIAH is not in `make gpu-test`*). The
  `#[ignore]` rule is **enforced** on the `niah_cell!` / `niah_pflash_cell!`
  bodies, but the cells are **not listed** for execution — a macro cell has no
  name until expansion, so `run_gpu_tests.sh` cannot build a libtest filter for
  one, and ~60 cells each running an 8k–32k-token prefill would turn the
  pre-merge GPU suite into hours. So `run_gpu_tests.sh` never selects NIAH, and
  a resolver that resolved perfectly would still be exercised by nothing.

  Arm it only alongside a decision to move those cells into a gate that executes
  them — the same condition that section records.

---

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
| `metal-unscanned` | yes | no | each is snapshot-gated or drives a child; the runner's per-crate banner check would fail on a host without the snapshot |

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

Each of these passes the gate silently. None occurs in the tree today:

- A signature-only fn whose `where` clause pushes the `;` to a later line. The
  latch closes at the next line indented like the fn, so any `#[test]` it
  swallowed goes unclassified. The fixtures `trait_where_signature` and
  `trait_where_signature_open_hole` pin both outcomes.
- An attribute left open by a raw string or a block comment, closed by a later
  line that ends in `]`. The items between are lost with no report.
- A raw string (`r"…"`, `r#"…"#`) or a `/* … */` block comment on an item's
  opening line: neither is tracked.
- An unqualified cross-file call through a glob import (`use m::*; helper()`).
- A helper in a non-scanned source file.
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
  variable and exits by a `return` carrying no value must print a notice.
  `RMLX_SKIP_GPU` guards are exempt.

The notice's shape lives in `scripts/lib/skip_notice_patterns.sh`, which the
gate and the runner both read. `make check-named-skip-notices-fixtures` is the
gate's recall test.

A cell that names a slug resolves it under `RMLX_O_MODELS_ROOT`, and its
variable is only a fallback. An unset variable therefore does not stand the
cell down. Integration tests resolve through `common::slug_or_override` in
`crates/rmlx-models/tests/common/mod.rs`; lib unit tests through
`crates/rmlx-models/src/test_snapshot.rs`. In the integration harness, a root
that is set but missing fails; an absent slug skips. A fallback that is used
prints `NOTE <test>: …`.

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
whose numbers are recorded. `VALIDATE=0` opts out.

A diagnostic names a *load* or a *store*. A store is a dropped write; a load
matters only if the kernel keeps the lanes it filled. The failure banner
prints the access mix per diagnostic. A clean scan does not prove that nothing
read out of bounds. The layer bounds against the `MTLBuffer`, not the array,
and MLX recycles buffers from size buckets.

The one standing diagnostic is MLX's own
`affine_qmm_t_splitk_bfloat16_t_gs_64_b_{4,8}_alN_false`, loads only.
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
not `SKIP`. Its entries stay expected, so the missing checkpoint's entry
reports `no longer fires`.

| observed | verdict |
|---|---|
| the expectation exactly, nothing else | pass: `census matches the pin`, with the accepted entries |
| a kernel the pin does not name | fail: `not pinned: N <kind> "<kernel>" in <crate>` |
| above the expectation | fail: `count moved up: … expected N, observed M` |
| below the expectation | fail: `count moved down: …` |
| nothing where the expectation is positive | fail: `no longer fires: …` |
| any store | fail |
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

---

## Allocation gates (`PeakBracket`)

A numerics test cannot see a change that leaves every output bit identical but
allocates an extra scratch buffer per dispatch. `rmlx_mlx::PeakBracket` scopes
the Metal allocator's high-water mark to a region so an allocation regression
becomes a test failure — no GPU timing, no model, no tolerance.

```rust
let bracket = PeakBracket::open();
let out = op_under_test(&input, Device::Gpu)?;
out.eval()?;                       // MLX is lazy: materialise INSIDE
let reading = bracket.close();

assert!(reading.observed_allocation());               // anti-vacuous, first
assert!(reading.headroom_bytes() <= 4 * input_bytes); // relative, never absolute
```

Three rules, each of which has a corresponding way to get it wrong:

- **Assert `observed_allocation()` before any upper bound.** An upper bound
  holds trivially against a region that allocated nothing, which is exactly
  what happens if the `eval()` drifts outside the bracket — the reading comes
  back `peak_bytes: 0` and the gate passes while measuring nothing. The
  predicate is `headroom_bytes() > 0`, i.e. this region's live bytes rose above
  where they started; `peak_bytes > 0` would be true in every real process,
  because MLX lifts the mark to the whole live count on the first allocation
  after a reset.
- **Bound a multiple of the workload's own size, never an absolute byte
  count.** MLX pools its buffers, so an absolute figure encodes what ran
  earlier in the test binary as much as what the region did.
- **The peak mark is process-global.** These tests reach `Device::Gpu`, so
  they carry `#[ignore]` and run under `--test-threads=1` like every other
  GPU test here; two brackets on parallel threads would reset each other.

Reference caller: `q8_msl_roundtrip_allocation_stays_within_budget` in
`crates/rmlx-kv-quant/src/q8_msl_tests.rs`. Accessor semantics are tabulated in
[`docs/PROFILING.md` §9.1](PROFILING.md).

---

## Cosine-similarity gate

Every KV-cache codec has a per-codec cosine-similarity quality gate in the
`rmlx-kv-quant` unit-test suite. The gate verifies that a quantize →
dequantize round-trip preserves the directional information in each row vector
to within an empirically derived floor.

The gates below use the **LCG fixture** (seed `TEST_SEED =
0x0000_00C0_FFEE_BEEF`, Knuth LCG) so they are deterministic and require no
model snapshot or GPU.

**What they do not measure.** The LCG fixture is i.i.d. uniform, which is
already close to maximally incoherent, so a decorrelating rotation cannot
improve it — an identity rotation passes every gate in the table below. That
axis is covered separately by the incoherence gates; see "Rotation-quality
gates".

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
- `gaussian_data(n, seed)` — standard normal from the same LCG via Box–Muller.
- `outlier_channel_data(rows, head_dim, channels, ratio, seed)` — Gaussian base with persistent high-magnitude channels; `outlier_fixture()` is the canonical 256 x 128, 4 channels at 20x. The doc comment carries the citations for that shape.
- `incoherence_per_row` — `mu = sqrt(d)·max|x_i|/||x||_2` per row; returns `IncoherenceStats { mean, p99, max, n_rows }`.
- `sqnr_db` / `wasted_bits` / `lloyd_max_anchor_db` / `LLOYD_MAX_GAUSSIAN_SQNR_DB` / `DB_PER_BIT` — rate-distortion reference.
- `fwht_normalize(buf, n)` — CPU Walsh-Hadamard transform (self-inverse when applied twice), used by the rot_k cosine test.
- `TEST_SEED` — pinned seed constant (`0x0000_00C0_FFEE_BEEF`). Never replace with `thread_rng`.

### Running only cosine gates

```bash
cargo test -p rmlx-kv-quant cosine_gate
```

---

## Rotation-quality gates

`crates/rmlx-kv-quant/src/rotation_fidelity_tests.rs`. CPU-only, no snapshot,
inside `make model-check`.

```bash
cargo test -p rmlx-kv-quant --lib rotation_fidelity -- --nocapture
```

Measured on `outlier_fixture()` — i.i.d. Gaussian with 4 of 128 channels at
20x, modelling the persistent per-channel Key outliers the KV-quantization
literature reports. Numbers and their derivation live in `docs/KV_CODEC_FIDELITY.md`
§ "Codec fidelity — measured".

| Gate | Asserts |
|---|---|
| `hadamard_incoherence_ratio_beats_every_block_local_rotation` | `rot_k` reduces mean `mu` ≥ 3x (measured 3.89x); every block-local family stays under its `sqrt(block)` ceiling and under `rot_k`. |
| `non_full_dimension_rotations_fail_the_hadamard_incoherence_gate` | Mutation guard: the same FWHT truncated to block-4, plus the iso / rotor / planar transforms, all fail. Rejection is a theorem — `mu` reduction of `R` needs block ≥ `R²`, so 3.0 needs block ≥ 9. |
| `identity_rotation_excluded_by_the_hadamard_incoherence_threshold` | Pins that the threshold excludes 1.00x. Named for what it is: the ratio is exact by construction, so this is a constant comparison, not a transform mutation. |
| `iso_block_rotation_incoherence_gate`, `planar3_…`, `planar4_…` | Two-sided: under the `sqrt(block)` ceiling (a theorem) and over the pinned floor (the regression guard). |
| `rotor_block_rotation_incoherence_gate` | Same, **swept over 8 `(layer, head)` draws** and pinned to the weakest (1.0815x of 1.0815–1.2089x). Only ~4 rotors of 43 touch outlier channels, so one draw is a four-sample estimate. |
| `rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data` | The Hadamard buys ≥ 1.5 bits of SQNR over the same quantizer without it on outlier data (measured +1.81), and **loses** bits on i.i.d. data (−0.63). |
| `non_full_dimension_rotations_fail_the_rot_k_gain_gate` | Mutation guard: block-4 truncated FWHT (+0.91) and the iso quaternion (+0.47). A block-`b` transform can buy at most `0.5·log2(b)` bits, so 1.5 demands block ≥ 8. |
| `identity_rotation_excluded_by_the_rot_k_gain_threshold` | Pins that the threshold excludes 0.00 bits; exact by construction, as above. |
| `<codec>_outlier_cosine_gate` (7) | Outlier-fixture cosine floors for `rot_k`, `iso3/4`, `rotor3/4`, `planar3/4`. |
| `lossier_codecs_fail_the_outlier_cosine_floors` | Mutation guard for all seven floors: each is shown to reject a genuinely lossier real codec, judged by the same floor function the gates use. |
| `wider_codebooks_score_higher_on_the_outlier_fixture` | iso4 > iso3 and rotor4 > rotor3 — catches a bit-width plumbing fault a per-codec floor cannot. |

**Outlier cosine floors use an error-relative tolerance**, not the `measured −
0.001` convention above: a codec may double `1 − cos` before the floor bites.
The absolute convention cannot work here — `rot_k` scores 0.999989 and 0.999881
with the Hadamard deleted outright, so a 0.001 slack is fifty times wider than
the whole effect and the deletion passes. `lossier_codecs_fail_the_outlier_cosine_floors`
checks all seven against a genuinely lossier real codec.

---

## Rate-distortion reference

`crates/rmlx-kv-quant/src/rate_distortion_tests.rs`. CPU-only, no snapshot,
inside `make model-check`.

```bash
cargo test -p rmlx-kv-quant --lib rate_distortion -- --nocapture
```

Every scalar-codebook codec at every shipped bit width, encoded and decoded on
an i.i.d. Gaussian fixture, reported as SQNR against the fixed-rate Lloyd-Max
Gaussian anchor for that width and converted to wasted bits. The full table is
in `docs/KV_CODEC_FIDELITY.md` § "Codec fidelity — measured".

Two thresholds, both stated in bits:

- **Absolute** — `MAX_WASTED_BITS = 1.0` against the anchor. The escalation
  line: a codec past it gets a filed follow-up with the measured number.
- **Per-cell pinned** — `measured + PIN_SLACK_BITS` (0.10 bits = 0.60 dB).
  This is the gate that fires. The absolute line above cannot fire on its own —
  every budget is ≤ +0.44, so crossing 1.0 implies crossing the budget too; it
  labels *why* a failure matters rather than adding independent coverage. And
  the absolute line alone would not be enough: codecs sit at very different
  offsets from the anchor (`turbo4` is 0.23 bits *ahead*), so one that silently
  loses a full bit can still land inside a 1-bit absolute budget.
  `one_bit_short_codec_fails_the_rate_distortion_gate` demonstrates exactly
  that at `bits = 4`. `pinned_budgets_sit_one_slack_above_the_measurement`
  keeps the pins where they claim to be.

Two measured facts are pinned as equalities so a fix turns them red rather than
passing silently: `trellis_coded_quantization_claws_back_nothing` (TCQ = plain
turbo, 0.000 dB) and `byte_identical_bit_widths_leave_one_width_dominated`
(iso, rotor and planar cost the same bytes at 3 and 4 bits, so each family has
a strictly dominated width).

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
  through `update_and_sdpa_shared_source`).

Neither family sets its env var directly — the harness reads the resolved
process-default policy, which the shell driver sets per process. To compare
OFF vs ON, run:

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

### Why NIAH is not in `make gpu-test`

The cells are macro-generated, so until the classifier learned to read
`macro_rules!` bodies they were absent from `make gpu-test` for no reason anyone
had decided — the detector simply could not see them. The split now in force is
deliberate:

* **Enforced.** The `#[ignore]` rule applies to the `niah_cell!` /
  `niah_pflash_cell!` bodies, and `make check-gpu-tests-ignored` fails if either
  loses the attribute. That is the half that was genuinely unguarded.
* **Not executed by `make gpu-test`.** These cells load a real snapshot and run
  an 8k–32k-token prefill each; ~60 of them would turn the pre-merge GPU suite
  from ~21 minutes into hours and make it depend on model snapshots being
  present. They already have a purpose-built driver — the shell wrapper above,
  plus `make smoke-codec-matrix` — which is where the long-context correctness
  claim is actually made.

So the runner never visits them and the gate never mandates an attribute on a
test the runner visits: those are two different populations here, on purpose.
Move them into `make gpu-test` only alongside a decision to accept model-gated
hours in the pre-merge gate.

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
| `RMLX_SHARED_SOURCE_STRICT` | `1` | Fail (not warn) on shared-KV producer dispatch parity tests. |
| `RMLX_SPARSE_ATTN_STRICT` | `1` | Fail (not warn) on sparse-attn dispatch parity tests. |

---

## Env-backed gates: readers need the lock too

`rmlx-kv-quant` exposes `test_utils::env_lock()`, a process-global guard for
every test in that binary that touches the environment. Three rules:

1. **Hold it for the whole test body**, not just across the mutation.
2. **Readers take it as well as writers.** A test that merely *reads* an
   env-backed gate — `rotor_qjl_enabled()`, a raw
   `std::env::var("RMLX_TURBO_FLASH")`, or anything that calls them, such as
   `KvQuant::cpu_hot_path_reason()` — races the tests that set
   `RMLX_ROTOR_QJL` and fails intermittently. Prefer a value the test owns
   (`.with_dispatch_policy(…)`) over an env read wherever one exists.
3. **Establish the state you assert.** The lock serializes access; it does not
   reset it. A test that asserts "QJL is off" without clearing
   `RMLX_ROTOR_QJL` first fails for anyone who has it exported, with a message
   that blames the test.

The granularity is the whole environment, not one variable: `setenv` is UB
against a concurrent `getenv` of *any* key, so one lock is the correct scope and
a per-variable lock would be unsound.

`env_lock()` returns an `EnvGuard` that **restores the managed keys on drop**,
including while unwinding from a failed assertion. Tests therefore set what they
need and do not clean up. This is not a convenience: every writer is shaped
`set_var` → `assert!` → restore, so before the guard existed a failing assertion
skipped its own restore and leaked the value into every later test, which then
failed with a message about its own precondition and buried the assertion that
actually broke.

The kernel gates (`RMLX_TURBO_FLASH`, `RMLX_FUSED_QK`, `RMLX_SPARSE_ATTN`,
`RMLX_PLANAR_FLASH_DECODE`, `RMLX_ROT_K_FUSED`) are **not** env reads at the
dispatch site: they seed a [`DispatchPolicy`](../crates/rmlx-core/src/dispatch_policy.rs)
that each `KvCache` captures at construction. A test that wants a gate on
should build its cache with `.with_dispatch_policy(…)` and take no env lock at
all — that is both race-free and the only way to have two gate states live in
one binary. Setting the env var still works for a whole process (it is the
`auto` fallback), which is what the shell drivers do.

`RMLX_ROTOR_QJL` is deliberately **not** latched (it is re-read on every
construction), which is what makes it raceable, and it is the only key
`EnvGuard` manages.

`RMLX_SKIP_GPU` is deliberately **never written** by any test. Its reader
`skip_if_no_gpu_env()` runs at the top of every `#[ignore]`d GPU test and none of
those take the lock, so a transient write could silently skip a live GPU test or
un-ignore a Metal one into a parallel run. The membership rule is factored out as
the pure `skip_value_means_skip()` and tested directly instead.

`rmlx-kv-ssd` keeps its own lock: separate crate, separate test binary,
separate process, no shared environment.

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
