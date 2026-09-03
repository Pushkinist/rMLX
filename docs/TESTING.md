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
| `RMLX_KV_TEST_MODEL` | `gemma4_kv_cache_equivalence.rs`, `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs`, `qwen3_5_mtp_drafter_alignment.rs`, `qwen3_5_eagle3_alignment.rs`, `qwen3_5_two_model_alignment.rs`, `projects_toml_e2e.rs`, `cli_flags_e2e.rs`, and as the single-model override for the golden-token suites | Model snapshot for KV-cache equivalence and drafter-alignment tests. Typically set to a Gemma4-e4b path; the Qwen3.5-family alignment tests take a **verifier** here instead (see below). |
| `RMLX_DRAFT_TEST_MODEL` | `dflash_drafter_alignment.rs`, `gemma4_mtp_drafter_alignment.rs`, `qwen3_5_mtp_drafter_alignment.rs`, `qwen3_5_eagle3_alignment.rs`, `qwen3_5_two_model_alignment.rs` | Draft model snapshot path. Used alongside `RMLX_KV_TEST_MODEL` for speculative-decode alignment tests. |
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

Point either at a different pair and re-measure both arms before reading a
failure as a regression.

`dflash_drafter_alignment.rs` is **not** one of those three and does not gate the
same property. It asserts that the drafter's round-0 first-block proposal aligns
with the verifier's greedy continuation (`accept > 0`) and that the live loop
emits coherent prose — a round-0 check, taken before any partial-accept rollback
has happened. It cannot see a rollback that corrupts the verifier state part-way
through a run, which is what the three suites above exist to catch. DFlash on a
GDN hybrid therefore has no greedy-tracking gate.

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
threshold; see `docs/PERF_BASELINE.md` under `--synthetic-arms`. The same
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

"Runnable" means the directory holds every file the harness opens **by name**:

| file | opened by |
|---|---|
| `config.json` | `model_arch`, `arch::load_model` |
| `tokenizer.json` | `run_golden_test`'s `Tokenizer::from_file` |
| `model.safetensors.index.json` **or** `model.safetensors` | `rmlx_loader::load_shard_index`, which tries them in that order and errors if neither exists |

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

**Residual, stated rather than papered over:** libtest discards a passing test's
output, so a golden that *skipped* prints its reason into a stream nothing shows.
The gate cannot report "0 goldens checked" from inside a normal run. Add
`--nocapture` when you need to see which ones stood down:

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
| `rmlx-cli/src/commands/kv_calibrate_tests.rs` | `RMLX_TEST_MODEL_*`, then slug | falls through to the slug |

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

A test that drives the GPU is marked `#[ignore]` and run explicitly:

```bash
cargo test --test embeddings_smoke -- --ignored --test-threads=1
cargo test -p rmlx-kv-quant --lib -- --ignored <filter> --test-threads=1
```

This is a correctness requirement, not a runtime-saving convenience. `cargo
test` runs a binary's tests on parallel threads, and a shared Metal context
driven from several of them aborts the **whole process**:

```
fatal runtime error: Rust cannot catch foreign exceptions, aborting
```

Every other test in the crate dies with it. The failure is load-dependent — a
couple of parallel GPU tests can pass for a long time before enough of them
tip the binary over — and each one still passes when run alone, so a PR's own
targeted run looks green. That is why the rule is mechanical rather than
judgement-based.

**Which tests get the attribute:** those that reach `Device::Gpu`, directly or
through a helper. A guard that only exercises a shape check the dispatcher
rejects before it touches a device-parameterized op is **not** a GPU test —
pass it `Device::Cpu` and leave it un-ignored, so it keeps running in the
default gate. Ignoring a CPU test is how a test silently stops running.

`make check-gpu-tests-ignored` enforces this across **every workspace member
crate** (read from `Cargo.toml`, never hard-coded), scanning both unit-test
roots — `src/**/*_tests.rs` *and* bare `src/**/tests.rs` — and the integration
binaries under `tests/*.rs`. It runs in `make ci` and in the hosted `source
gates` job. It keys on the shape (does the test reach `Device::Gpu`, directly,
through a same-file helper, or via a module-scope `const … = Device::Gpu`?),
never on the ignore reason's wording, which varies across the tree. The one
deliberate exception is the converse check below, which has no shape to key on.

"Test" means `#[test]`, `#[tokio::test]`, and `#[tokio::test(flavor = …)]`. An
async test attribute is a test attribute; matching only the bare spelling left
~107 of them in `rmlx-server` unclassified in both directions.

#### Exempting a device-as-value test

A pure device-*policy* test — one that passes `Device::Gpu` to a non-mlx function
as a plain selector value and never dispatches Metal — opts out **per fn** with a
line-leading `// gpu-test-gate: exempt` marker in its own attribute block. The
exemption is scoped to that one `#[test]`, not the file, so a Metal-driving test
added beside it still trips the gate; a copy of the marker inside a fn body
exempts nothing. A reviewer audits each one.

#### An `#[ignore]` that claims Metal and cannot prove it is fatal

An `#[ignore]` whose reason names a Metal context, on a test from which the
classifier can reach no `Device::Gpu`, runs under **no gate at all**: `make test`
skips it because it is ignored, and `make gpu-test` skips it because it is not
classified. It reads as covered at both.

This was a non-fatal warning, on the reasoning that some ignores are legitimately
non-GPU and that a helper in a non-scanned source file makes a real Metal test
look GPU-free to a source scanner. Both are true, and neither is a reason to
leave the finding advisory. An advisory channel with two valid outcomes and no
way to record which one applies is a channel nothing ever closes; seven tests
accumulated in it. It is now a hard failure with three dispositions:

1. **It drives Metal by a route the scanner cannot follow** — declare it (below).
2. **It does not touch the GPU** — drop the `#[ignore]` and pass `Device::Cpu`,
   so it runs in the default gate again.
3. **It is ignored for some other reason** — say *that* reason in the `#[ignore]`
   text instead of claiming Metal.

(3) is a genuine escape hatch and is stated rather than hidden. This one check
keys on the ignore reason's **wording** — it has to, the absence of a shape to
key on being the whole problem — so a Metal-driving test whose ignore text never
says "Metal" or "GPU" is invisible to it. Nothing else in the gate works that
way.

#### Declaring a Metal route the scanner cannot follow

```rust
// gpu-test-gate: metal-unscanned  <why the scanner cannot see it>
#[ignore = "GPU Metal: …"]
#[tokio::test]
async fn drives_metal_over_http() { … }
```

Line-leading, in the fn's own attribute block, scoped to that one `#[test]` —
the same rules as `// gpu-test-gate: exempt`, of which it is the exact inverse:
`exempt` says *this names the device but never dispatches*, `metal-unscanned`
says *this dispatches but never names the device*.

Effect: the test counts as GPU-touching, so the `#[ignore]` rule bites on it —
deleting the attribute is now a violation, which before the marker it was not.
It is **not** emitted to `--list`; see the population table below.

Two shapes fail closed rather than being resolved in the author's favour: a
marker on a test the reachability pass *can* see through (stale — its claim is
checkable, and keeping it would hold a listable test out of the runner forever),
and a marker paired with `exempt` (the test both does and does not drive Metal).

#### The seven declared routes, and what actually covers them

Two boundaries defeat a source scanner here, and no extension of it would help:
an HTTP request resolves through a routing table rather than a call graph, and a
child process has its own Metal context.

| test | file | route | covered by |
|---|---|---|---|
| `valid_single_vector_200_shape` | `crates/rmlx-server/tests/embeddings_smoke.rs` | HTTP → `embeddings()` → `Device::Gpu` | nothing; run by hand |
| `return_multivector_toggles_shape` | `crates/rmlx-server/tests/embeddings_smoke.rs` | as above | nothing; run by hand |
| `invalid_dimensions_is_400` | `crates/rmlx-server/tests/embeddings_smoke.rs` | as above | nothing; run by hand |
| `image_single_vector_200_shape` | `crates/rmlx-server/tests/embeddings_smoke.rs` | as above | nothing; run by hand |
| `image_multivector_toggles_shape` | `crates/rmlx-server/tests/embeddings_smoke.rs` | as above | nothing; run by hand |
| `ssd_cache_survives_server_restart` | `crates/rmlx-server/tests/ssd_cache_restart.rs` | spawned `rmlx serve` child | `make e2e` phase 2a covers the same spill → restart → hydrate chain |
| `paro_kernel_registration` | `crates/rmlx-models/src/paroquant_msl_tests.rs` | `paro_rotate_kernel()` in a non-scanned source file | nothing; the two `paro_rotate_identity_roundtrip_*` cells in `make gpu-test` compile and dispatch the same kernel |

All seven genuinely need Metal — audited against the code, not against their own
`#[ignore]` text. The five embeddings cells post to `/v1/embeddings`, whose
handler loads the jina encoder and runs the forward under `Device::Gpu` on a
`spawn_blocking` worker of the same process. `invalid_dimensions_is_400` looks
like the file's other request-validation 400s and is not one: the handler defers
`dimensions` to the model's matryoshka set, so the rejection comes out of
`pooling::single_vector` *after* a full GPU forward.

**They are enforced but not executed, and that is a runner property, not a
classifier one.** `run_gpu_tests.sh` asserts per crate that Metal's
shader-validation banner appeared — a crate that created no Metal device proved
nothing. Every declared test is snapshot-gated (`RMLX_TEST_MODEL_JINA_V4`,
`RMLX_TEST_MODEL`) or drives a child, so on a machine without that snapshot the
runner would execute a handful of early returns, see no banner for the crate, and
fail the suite over a missing model rather than a defect. `ssd_cache_restart`
could not be listed on any machine: its Metal is in the child, so the in-process
instrumentation covers nothing, and it additionally needs `cargo build -p
rmlx-cli` first (`cargo test --tests` does not build that binary), `pkill`s every
MLX process, and spends two 180 s readiness waits.

Run the embeddings cells by hand:

```sh
RMLX_TEST_MODEL_JINA_V4=/abs/path/to/jinaai__jina-embeddings-v4 \
  cargo test -p rmlx-server --test embeddings_smoke -- --ignored --test-threads=1
```

`paro_kernel_registration` is the one entry that could be listed as-is —
`rmlx-models` is already in the runner's population and already produces a
banner. It is left declared-but-unlisted because moving it needs a GPU run to
confirm, which is a separate change.

**Three populations, not one.** `--list` (what `make gpu-test` executes) is a
strict subset of what the gate enforces, and every run prints the difference:

| population | `#[ignore]` enforced | in `--list` | why |
|---|---|---|---|
| `Device::Gpu` reachable | yes | yes | the ordinary case |
| macro-generated | yes, at the `macro_rules!` body | no | cell names exist only after expansion, so no libtest filter selects them |
| `metal-unscanned` | yes | no | the runner's per-crate banner assertion; see above |

The other known edge: the reachability seed is file-local for unqualified calls,
so a test that reaches Metal only through a same-named helper in another module
can draw the fatal converse as a false positive. The marker is the disposition
for that too — verify the route before adding it.

**Inside a `macro_rules!` body the exemption's blast radius is every cell.** The
body is one synthetic test, so a single marker line exempts every test that
macro generates — around thirty cells from one comment, at the shape in this
tree. "Scoped to that one `#[test]`" stays literally true and badly understates
it: review a marker inside a macro body against every invocation, not one.

#### Macro-generated tests

A `macro_rules!` body that emits `#[test] fn $name() { .. }` names no readable
fn at its definition site and emits no `fn` line at its invocation sites. A
scanner that only accepts `fn <ident>` therefore misses such a test in **both**
directions at once — never flagged however much Metal it dispatches, and never
listed for the runner either — so the rule holds by author discipline and a
deleted `#[ignore]` is caught by nothing.

The gate classifies the macro **body** as one synthetic test. That is where the
attribute under enforcement lives, and one body governs every cell it generates,
so a `#[ignore]` deleted there fails the gate regardless of how many invocations
exist. Reachability is traced from the body like any other fn's, so a body that
reaches Metal one call deep (the shape in the tree today) counts.

Four things are fail-closed rather than skipped, because a shape the parser
cannot read looks exactly like a compliant one:

* A `macro_rules!` body that declares more `#[test]` than the gate could read
  back as items — the name is assembled rather than written (`paste!` /
  `concat_idents!`), or a `#[test]` shares its `fn`'s line.
* A `macro_rules!` written entirely on **one line** whose body declares
  `#[test]` — no `fn` line ever follows, so nothing is classifiable.
* An item whose closing brace is never found, which means the parser lost the
  file at that point and everything after it went unclassified.
* An **attribute** whose closing `]` is never found, for the same reason: while
  an attribute capture is latched it consumes every line, `fn` lines included.

Write the generated fn as `fn $name()` on its own line with its attributes above
it, close the attribute on a line whose last significant character is `]`, or
extend the classifier.

The first counter is also what stops the original blindness from returning
quietly. A future edit that narrows the fn-name recognition again does not make
macro cells invisible — it makes the declared-vs-readable counts disagree, and
the gate goes red naming the macro. Verified by mutation: reverting the fn
pattern to identifiers-only turns `check-gpu-tests-ignored` red on
`niah_long_context.rs`.

**A self-contained `fn` line must not latch the multi-line capture.** A macro
body may write `fn $name() { .. }` whole on one line. Latching there waits for a
closing brace that never comes, so every later item in the file joins that one
body and stops being classified — which loses violations the gate previously
caught. `make fmt-check` does not keep the shape out: rustfmt refuses to
reformat a `macro_rules!` body containing a `$(..)*` repetition (verified
byte-identical after `rustfmt --edition 2021`).

The classifier decides this from the line's **last significant character** — `}`
ends an inline body, `;` ends a signature-only declaration — and deliberately
**not** from a brace count. Braces inside a string, a char literal or a trailing
comment are not block delimiters, and counting them is wrong in both directions:

* a `}` inside a string literal makes a self-contained line look open, so it
  latches and swallows the rest of the file (measured: `exit 0`, `OK`, on a file
  one character away from a compliant one);
* a `}` inside a trailing comment makes an opening line look closed, so the
  body is never captured and a `Device::Gpu` inside it is invisible — that
  direction was a recall regression against `main`, not just a missed
  extension;
* a signature-only `fn` in a `trait` or `extern` block has no brace at all, and
  a rule keyed on "no brace open" latches it forever and hard-fails the whole
  run blaming a `macro_rules!` the file does not contain.

**Known OPEN blind spot — a `where`-split signature.** The `;` alternative
covers a signature that fits on **one line**. A `where` clause pushes the `;`
onto a later line, so that declaration latches — and the latch is closed by the
first later line that bares to the fn's indent, which in a Rust test file is
`    }`, the close of any nested block. When it closes that way **no error is
emitted**: the swallowed `#[test]` was never registered, so nothing looks
unterminated, and the gate reports `OK` at exit 0 over an un-ignored
`Device::Gpu` test.

This is **fail-open**, not fail-closed. It is unreachable in the tree today (no
scanned file has a where-split signature), and the class is tracked in #386 for
reconciliation against the compiled `cargo test -- --list`, which is what
actually closes it. Two fixtures bracket it: `trait_where_signature` pins the
sub-case where nothing closes the latch (loud), and
`trait_where_signature_open_hole` pins the open answer so the hole is visible in
the corpus rather than only in prose.

It is the only fail-open hole reachable by a shape the parser reads correctly.
The attribute capture below has a second, narrower one that needs a parse hazard
first — see the end of that section.

#### The attribute capture

Attributes are the third latching capture, alongside the fn item and the
`macro_rules!` body, and they carry the same hazard: an attribute that does not
close on its own line latches, and while latched it consumes every following
line — `fn` lines included — so a latch that never ends silently unclassifies
the rest of the file.

The close-test is the same **last significant character** rule the fn arm uses.
Two narrower spellings of it were each wrong on their own:

* keyed on `)]`, the shape a wrapped `#[cfg(..)]` ends in, it never matched the
  wrapped **string** form `"]` — which is how `#[ignore = "GPU Metal context: \`
  wraps, i.e. exactly the remediation this gate's own error message recommends.
  The documented way to comply was the way to blind it;
* read from the **raw** line, a trailing comment (`#[test] // why`) hides the
  closing `]` and latches just as thoroughly.

String state is carried across the line break, so a `//` inside the continued
payload of a wrapped string is not mistaken for a comment.

Neither recognition rule can be complete, so an attribute still open at a file
boundary is reported (`U`) rather than passed over.

**What remains open here**, stated rather than papered over: that backstop only
covers a latch that reaches the end of the file. If any *later, unrelated* line
bares to `]` — a subsequent `#[test]` is exactly that shape — the latch ends
there instead. Classification resumes correctly, but the items swallowed in
between are gone with **no report**. That is fail-open, the same shape as the
`where`-split hole above, and this parser cannot see it; reconciliation against
the compiled `cargo test -- --list` is what would.

So the close-test does not buy immunity, it buys reachability. Getting into that
state now requires one of the two `bare()` hazards below (a raw string, a block
comment); before, the ordinary wrapped-string spelling was enough — and that
spelling is the one the gate's own error message recommends.

Three fixtures pin the section: `attr_multiline_ignore` (the wrapped-string
form closes, and is read as one block rather than merely un-latched),
`attr_trailing_comment` (a comment after the closing `]` does not latch), and
`attr_never_closes` (an unterminated attribute is reported, not swallowed).

The trailing-comment scan is string-aware, so neither a URL in a one-line fn
(`"http://…"`) nor a char literal of any payload form (`b'"'`, `'\x1b'`,
`'\u{FFFD}'`, `'é'` — 18 such literals occur in the scanned tree) derails it;
either would otherwise leave the scanner stuck "inside a string" for the rest of
the line. A non-ASCII payload is one character but two-to-four *bytes*, so where
awk indexes bytes it is stepped by locating the closing quote with `index()`
inside a bounded window rather than by a byte class — `index`, `substr`,
`length` and `RLENGTH` all count in the same units in every awk, so the offset
lands correctly either way. The step additionally requires the payload to begin
with a non-ASCII unit, which is what keeps a lifetime tick followed by a nearby
quote (`<'a>'x'`) from being consumed as though it were a literal. It does
**not** handle raw strings (the `\` in `r"a\"` is not an escape, and `r#"…"#`
hashes are not tracked) or block comments: a `/* … */` spanning an item's
opening line is read literally. The same scan decides the attribute close-test,
so those two hazards bound that rule exactly as they bound this one.

Still out of reach on the macro side: a `macro_rules!` with a **non-brace**
delimiter (`macro_rules! m ( .. );`). Its name is captured so findings are
greppable and its items are still classified — a `$metavar` fn is
macro-generated by its own shape — but its body extent is not tracked, so the
readability counters above do not cover it.

Still out of reach, and deliberately stated rather than assumed away: a test
generated by a **proc macro** (`#[rstest]`, `#[test_case]`), and a
`macro_rules!` defined in a non-scanned source file and invoked in a scanned
one. Nothing in the tree has either shape; adding one needs the classifier
extended, not a review note.

`make check-gpu-tests-ignored-fixtures` pins all of this. Each fixture under
`scripts/fixtures/gpu_tests_ignored/` is a synthetic workspace driven through the
gate's `--root` option. Half are violations it must catch and half are legitimate
shapes it must leave alone — a gate that fails everything is as useless as one
that fails nothing, and only the pair pins it.

**Each case asserts the reason, not just the exit code.** The gate has six
fail-closed paths that also exit 1 (missing `--root` directory, zero parsed
members, members fewer than crate dirs, a member whose `src/` is gone, an
unreadable package name, zero matched test files), so a case checking only
`exit == 1` is satisfied by all of them — deleting a fixture would make its own
case pass. Every case therefore pins the exit code, the violation-class marker,
the specific label the gate must name, and optionally a string that must not
appear; the harness also refuses to run a case whose fixture directory is
missing.

**Every case runs once per awk on the machine.** The gate is an awk program and
awk implementations genuinely disagree, so checking one proves less than it
looks like it does. A bracket range of octal escapes (`[\300-\337]`, the obvious
way to match a UTF-8 continuation byte) is accepted by BSD awk and is a *hard
syntax error* in gawk — a gate written that way is green on a Mac and does not
execute at all on the Linux CI runner, which is worse than the blind spot it
closes. Byte-vs-character indexing differs too: gawk indexes characters under a
UTF-8 locale where BSD awk and mawk index bytes, and `[[:print:]]` calls a
continuation byte printable in BSD awk and mawk under UTF-8 but not under C.
The harness therefore shims `awk` on `PATH` and re-runs itself under each of
`awk`, `gawk` and `mawk` that exists (deduped by inode, so Debian's
`awk` → `mawk` symlink counts once). **When only one is installed it says so** —
a run that quietly checked a single implementation is the same
"gate that cannot fail" shape the reason-assertions above exist to prevent.
Install the others locally with `brew install gawk mawk`.

### Running them: `make gpu-test`

`make test` is `cargo test --workspace` — it passes no `--ignored`, so it skips
every test above. The hosted CI runs no tests at all and has no Metal. For a
long time nothing ran them, and GPU tests went red on `main` and stayed red
while every gate reported green. `make gpu-test` is the step that runs them:

```bash
make gpu-test                                   # every member crate
make gpu-test CRATE=rmlx-kv-quant               # one crate
make gpu-test CRATE=rmlx-kv-quant FILTER=rotor_flash
make gpu-test VALIDATE=0                        # without shader validation
```

It runs exactly the set `check_gpu_tests_ignored.sh --list` emits — the named
`#[test]` fns that reach `Device::Gpu` **and** carry `#[ignore]`. Deriving the
population from the enforcing gate's own classifier is the point: a separate
hand-maintained list would drift, and the rule would end up mandating `#[ignore]`
on tests the runner never visits.

It deliberately does **not** run every `#[ignore]` test. Many are ignored for
reasons unrelated to Metal — live network access, a missing cargo feature,
`ignore`-marked doc-comment pseudo-code — and sweeping those in would keep this
gate permanently red for things it cannot speak to.

**One documented divergence: macro-generated tests are enforced but not listed.**
`--list` feeds this runner, which turns each name into a libtest substring
filter — and a macro cell has no name until the compiler expands it, so listing
`niah_cell!{$name}` would emit a filter matching nothing and trip the runner's
own "executed fewer than classified" check. The enforcing run prints the
excluded set on every invocation, so the divergence is stated rather than
inferred from source comments. The one population it applies to is the NIAH
long-context cells; the reasoning is in the NIAH section below.

It is fail-closed in three ways:

* **Coverage.** Every classified test must actually run. If a crate executes
  fewer tests than were classified for it — a renamed fn, a target no longer
  built, a test compiled out behind a feature — that is an error. Checking only
  for "executed zero" would let 317 of 318 run and still report green.
* **`RMLX_SKIP_GPU=1` is refused outright.** Every classified test opens with
  `if skip_if_no_gpu_env() { return; }`, so with it set the whole suite returns
  before touching Metal and the gate would happily print `OK: 318 GPU tests
  passed` having dispatched nothing. It is a documented setting for Metal-less
  environments, so a stale export in a dev shell would otherwise disarm the one
  step that proves the GPU path works.
* **Exclusive GPU.** It refuses to start while another MLX process holds the
  Metal context (CLAUDE.md hard rule 8).

**No test in this suite is known-red on `main`, and this runner tracks no
known-red list of tests.** (Shader-validation hits are the one thing it does
track a baseline for, and that baseline is exact — see the census pin below. It
covers hits, never a failing test.) It used to claim the opposite, in the failure
banner and here.
That claim is what turns an inherited failure into a waved-through one — either
direction: a real regression read as "the known one", or hours spent on a red
that predates the branch. A failure is attributable only after the same crate
and filter are re-run on a clean checkout of the base commit and the two
outputs are compared; that takes minutes and is the only evidence that
separates the two cases. A list of currently-known failures is deliberately not
kept here — it would rot into exactly the false assurance it replaced, and a
comparison against the base commit is always current.

Note that a failure of this runner is not only a failing test: a Metal
shader-validation hit the census pin does not account for fails it too, on a
crate whose tests all passed. An out-of-bounds device store is dropped rather
than raised, so the tests can pass while the GPU reads or writes memory it does
not own — which is precisely a result no test result can be read for.

Read the diagnostic's own wording before assuming that is what happened. A
*store* is corruption; a *load* is illegal but can only affect the result if the
lanes it fills are ones the kernel keeps, and whether they are is a property of
the kernel, not of the diagnostic. The final banner prints the mix it actually
saw — each hit's own `device load` / `device store` wording, counted per
diagnostic rather than per output line, since the layer writes while libtest is
mid-line and reports routinely share a line — so it is
read off the run rather than assumed; the one standing hit in this tree is
160/160 loads, see the entry below on `affine_qmm_t_splitk`. The converse does
not hold either: a clean scan does not establish that nothing read out of
bounds, for the buffer-versus-array reason recorded with that entry.

**Every red the run found is reported, at both levels.** Shader-validation hits
and crate failures — a failing test, a crate that executed fewer tests than were
classified for it, a crate that produced no validation banner — accumulate
independently across the crate loop, and every one of them is printed before the
runner exits. Within a crate the same holds: a shortfall is recorded and falls
through to the exit-code check rather than skipping the rest, because an aborting
test binary produces both, and a shortfall alone reads as "a filter stopped
matching" and sends the reader after a renamed fn instead of the test that took
the binary down.

That ordering is load-bearing rather than cosmetic: while any standing diagnostic
exists, reporting the validation aggregate and exiting would discard the failing
test names the runner already extracted, and each crate's log is deleted inside
the loop, so nothing would survive to re-read. The failing-test oracle would be
real, working, and starved of execution by an earlier, less-specific exit — the
same "gate that cannot fail" shape as a vacuous oracle or a golden that skips
silently. `scripts/run_gpu_tests_selftest.sh` (`make gpu-runner-selftest`, in
`make ci` and in the hosted `source gates` job) pins it against stub crates, with
no GPU: a canned libtest log per crate carries a validation hit, a failing test,
an under-match, a missing banner and two diagnostics sharing one output line, and
each case asserts the reason that reaches the final report rather than only the
exit code. The census-pin verdicts are pinned in the same file, by the same
means.

#### Where it runs: `make ci-perf`, not `make ci`

`make ci-perf` is three lines, and it is the only shared gate that executes the
GPU tests:

```make
ci-perf:
	@bash scripts/run_gpu_tests.sh --preflight
	$(MAKE) test-perf
	@bash scripts/run_gpu_tests.sh
```

It is deliberately **not** in `make ci`. Two costs rule that out: the suite needs
the Metal context to itself (CLAUDE.md hard rule 8), so `make ci` could no longer
be run alongside a live `rmlx serve`; and it adds minutes to a target that runs
on every commit.

**This is a real new cost for `ci-perf`, not a free one.** Before, `ci-perf` was
exactly `cargo test --workspace --profile release-perf` — one invocation that
runs no `#[ignore]` test and so needs no GPU to itself; it could be, and was, run
next to a live server. It now refuses to start unless the GPU is idle. `ci-perf`
is simply the cheapest place in the tree to pay that: it is already the long,
pre-merge-only target, and the preflight line makes the new precondition fail in
milliseconds rather than after the release-perf half.

That ordering is the point of splitting the runner across two lines.
`--preflight` checks only the environment — `RMLX_SKIP_GPU` unset, no competing
MLX process, a non-empty classification — and runs no tests. Those are the most
likely way this gate fails in daily use, and finding out about a live `rmlx
serve` after `test-perf` has finished throws away the ~16 min it took. The tests
themselves go last, after `test-perf`, because `test-perf` covers the whole
workspace: a compile error anywhere surfaces there, while the GPU run visits five
crates and holds the GPU while it does.

`ci-perf` invokes the runner **directly** rather than through `make gpu-test`.
Make propagates command-line variables into sub-makes, so `make ci-perf
CRATE=rmlx-audio` would have run 7 of 318 tests and still printed `ci-perf ok`,
and `make ci-perf VALIDATE=0` would have disarmed shader validation. Neither is
catchable by the coverage check, because `--crate` narrows the classified
population in lockstep with the executed one. The knobs stay on `make gpu-test`,
where a human asking for a subset means it.

The two halves build under different profiles, also on purpose. The GPU run uses
`dev`, where debug assertions are live — 61 `debug_assert!` sites in
`rmlx-kv-quant` alone — and those are correctness guards on correctness tests.
`test-perf` must be `release-perf`, because that is the codegen a perf-sensitive
change ships under. The consequence is recorded in CLAUDE.md hard rule 9: **no
gate anywhere runs a `Device::Gpu` test with debug-assertions off**, so a GPU
defect that only appears at that profile has to be reproduced by hand.

**What it costs.** Measured on this host, both figures with a warm `target/`:

| | |
|---|---|
| GPU suite alone | **re-measure** — the last figure (264 s, 318 tests over 5 crates) was taken while every model-gated cell returned instantly, and they no longer do |
| Whole `make ci-perf` after a codec-layer edit | **~21 min** (1270 s green, 1358 s on the red run), plus whatever the model-gated cells now cost |

The second is the number that matters, since a `.metal` change invalidates
`rmlx-kv-quant` and everything downstream of it under `release-perf` too.

The golden-token suites and the snapshot-loading `rmlx-models` cells were always
in that 318, but contributed nothing to the runtime because nothing set the
variable they resolved from (see the golden-token section above). They now
resolve by slug under `RMLX_O_MODELS_ROOT`, so a machine holding the snapshots
pays their real cost — several model loads, one of them a 35B MoE. On a machine
with no models root, nothing changes.

**Neither figure includes a cold `dev` build, and that is the case to expect.**
`ci-perf` otherwise touches only `release-perf`, so the GPU half brings a second,
unshared `dev`-profile build of those five crates and their dependency trees with
it — at `opt-level = 0`, and after `test-perf` has just built the same test
binaries under `release-perf`. `target/debug` is also precisely what
`scripts/target_gc.sh` prunes first: it protects `release-perf` and names
`target/debug` as the dominant consumer. So the first `make ci-perf` after the
`make target-gc` that `make ci` advertises pays a full cold `dev` build on top of
the ~21 min.

Sharing `release-perf` with `test-perf` would remove that, and is deliberately
not done — it would run the GPU correctness suite with `debug_assert!` compiled
out, which is the wrong trade for the layer with this repo's documented
silent-corruption class.

While iterating on the codec layer, run `make gpu-test` directly and narrow it
with `CRATE=` / `FILTER=` rather than paying for the whole gate each time.

### Metal shader validation (on by default here)

Running the GPU tests is necessary but not sufficient, because the failure this
layer is prone to **does not fail a test**. An out-of-bounds device store from a
Metal kernel is dropped: the command buffer completes, `cb.error` is `nil`, the
process exits 0, and the assertions downstream of the frozen buffer still pass.
Measured on a deliberately broken `q8_quantize` kernel — both GPU tests over it
reported `ok` and the runner printed `OK: 2 GPU tests passed`, with no output of
any kind about the invalid write.

`make gpu-test` therefore instruments every pipeline with Metal's shader
validation and **scans the output**, failing the run when a diagnostic appears:

```
Invalid device store at offset 4000064, executing kernel function: "custom_kernel_rmlx_q8_quantize"
```

Five things decide how this is wired, each of which would otherwise produce a
gate that runs and can never fire:

* **The exit code is not the signal.** With validation on, cargo *still* exits 0
  and the tests *still* report `ok`. Only the diagnostic text distinguishes the
  broken tree from the clean one.
* **`MTL_SHADER_VALIDATION=1` alone reports nothing to stderr.**
  `MTL_SHADER_VALIDATION_REPORT_TO_STDERR` defaults to `0`, so reports go to
  Unified Logging (`man MetalValidation`). `scripts/run_gpu_tests.sh` owns the
  whole `MTL_SHADER_VALIDATION_*` environment rather than inheriting it —
  `DEFAULT_STATE`, `DISABLE_PIPELINES`, `ENABLE_ERROR_REPORTING`,
  `GLOBAL_MEMORY`, `THREADGROUP_MEMORY`, `FAIL_MODE`, `REPORT_TO_STDERR` are all
  pinned — so no stale export in a dev shell can leave it looking armed while
  blind.
* **The diagnostic is not line-anchored.** The validation layer writes to the
  process's stderr while libtest is mid-line, so a report routinely appears
  appended to a `test some::name ... ` prefix. Matching `^Invalid` catches only
  about half of them.
* **The validation banner is asserted, per crate.** If Metal never prints `Metal
  GPU Validation Enabled` for a crate, that crate ran uninstrumented and its
  silence proves nothing. (A crate reported this way has usually failed to
  *build*: no test binary means no Metal device and therefore no banner.)
* **A positive control runs first.** The banner proves the instrumentation
  loaded; it says nothing about whether the *detector* still matches what the
  layer emits, and the detector is a hand-written pattern over an undocumented,
  version-specific message. So the runner first executes a kernel that stores
  out of bounds on purpose — `crates/rmlx-kv-quant/src/shader_validation_canary.rs`,
  behind the `shader-validation-canary` feature so nothing else builds it — and
  refuses to trust a clean scan unless that produced a diagnostic it matched.
  The canary declines to dispatch unless validation is on, so the deliberate
  out-of-bounds write is never a real one. It is excluded from the population
  `make gpu-test` derives from the ignore-rule classifier, since it is the
  gate's self-test rather than a correctness test.

Buffer labels would make a hit even more direct, but MLX owns the Metal
allocator and mlx-c exposes no labelling API, so KV stores appear as
`buffer: <unnamed>`. The **kernel function name** carries the attribution
instead, and rMLX does control that: `custom_kernel_rmlx_<codec>_<op>` names the
codec and the operation exactly.

Validation costs throughput, so it belongs on this target and **not** on any
cell whose numbers get recorded. `VALIDATE=0` opts out.

**What it costs.** On the `rmlx-kv-quant` GPU suite: **133 s → 157 s, +18%**
(alternating pairs; an independent run measured 131.2 s → 154.6 s, +17.8%, n=3).
A cold first run inflates the uninstrumented baseline, so compare adjacent runs.
On real inference it is far worse — Ternary-Bonsai-8B decode 126.6 → 21.7 TPS,
a **5.8× slowdown** (n=3 each). That is the reason this never goes near a perf
cell.

**Never draw a conclusion about model *output* from a validated run.** The unit
suites are invariant — the same pass/fail result in every repetition of both
modes, with zero invalid-access reports — but real inference is not.
Ternary-Bonsai-8B (Qwen3 dense) intermittently produces a NaN prefill on this
host. The generation now aborts with an `error!` naming `nan_count`,
`max_abs_logit` and `prompt_len`. Under a one-shot command (`baseline`, `chat`,
`info --probe-smoke`) that ends the process with a non-zero exit; under `serve`
it is a failed request — HTTP 503 — and the process keeps running, so the log
event is the only signal. It used to emit a single token (id 0, `"!"`), report
`tps=0.064`, and exit 0 with nothing logged at any level — that silent shape is
gone, but the underlying NaN is not fixed by making it loud.

**What the trigger is not.** A 108-run campaign across four separately-built
binaries put all three degenerate events inside one window where `prefill_tps`
was depressed ~7% (0 of 60 runs at ~266 tok/s; 3 of 10 at ~247), and ruled out
`--log debug`, CPU contention, and prompt length with numbers. Architecture
specificity is **not** established: gemma-4-e2b has been clean, but no
gemma-4-e2b run has been shown to sample that depressed band, and a campaign
that never enters it has not sampled the failure regime. Do not pool runs across
a >5% `prefill_tps` shift, and record the band on every run.

Nor are the two arches equally exposed per run. `enter_prefill` returns early on
a rotating cache, so gemma-4's sliding-window layers never touch the shared
`prefill_raw` append at all and only its `full_attention` layers do — where
every Qwen3-dense layer does. Same number of runs is not the same number of
draws, and comparing the two by run count overstates what a clean gemma-4-e2b
record rules out.

`MTL_SHADER_VALIDATION_FAIL_MODE` was tried as a discriminator, on the theory
that `zerofill` silently dropping a KV store would explain it. **That
experiment settles nothing**: n=3 per arm, Fisher p = 1.0, no power at any
effect size — and its direction was read backwards. `allow` reproducing while
`zerofill` is clean is what a genuine out-of-bounds *access* predicts; the
assignment is inverted only for a different hypothesis ("validation drops a
write the engine needs"). No invalid access is ever reported, and that negative
*is* load-bearing: the detector was mutation-checked by running the OOB canary
under both fail modes, and it reported in both. An out-of-bounds access would
have been caught and was not, which leaves an in-bounds read of stale or
never-written device memory as the next candidate — but not in a buffer this
repo owns:

* **The KV ring's never-written tail is not it.** `update_prefill_raw`
  allocates the ring with `zeros()` on both the lazy-alloc and the grow branch,
  and hands attention `[.., 0..offset, ..]`. The slots past `offset` hold 0.0
  and are not part of any SDPA input at any prompt length. Pinned by
  `prefill_ring_tail_is_zeroed_and_never_returned` (`rmlx-kv-quant`,
  CPU) — including against a NaN payload, which a max-of-abs fold would have
  missed.
* **No rMLX Metal kernel runs before that logit row on this architecture.**
  While `in_prefill` is set, `KvCache::update` routes every non-rotating cache
  to `update_prefill_raw`, which is MLX ops only; the codec dispatch sits behind
  that branch and the fused-QK paths are gated on `q_seq == 1`. Qwen3-dense has
  no rotating/SWA layers, and neither `.metal` kernel in `rmlx-models`
  (`gated_delta_step`, `paroquant_rotate`) is on this arch. Every kernel body
  rMLX dispatches in such a request comes from `exit_prefill`, which runs after
  the logit graph is already built. The prefill logit row is produced by MLX
  ops end to end.
* **"Reproduces at both `k8v8` and `none`" carries no codec information.**
  Those two cells execute the same appends over the same bf16 buffer during
  prefill — pinned by `prefill_append_is_codec_independent` — so the pair
  discriminates nothing. It is consistent with a fault in the shared path and
  says nothing more.

What remains is an MLX-level mechanism: an MLX op, or the Metal runtime
underneath it (MLX allocates with `MTLHazardTrackingModeUntracked` and its
command encoder models read-after-write hazards only — see ml-explore/mlx#3630
and ml-explore/mlx#3461). Still a hypothesis, and tracked outside this
document.

Current state: four of the five crates run clean under validation. That is an
inference from matching totals (a narrowed `CRATE=rmlx-models` run and the
full five-crate run both total 160, and `rmlx-models` is the only crate a
narrowed run can attribute anything to), not a directly observed clean pass
over each of the other four individually — `run_gpu_tests.sh` prints a
per-crate line only for a crate with a nonzero hit count.
`rmlx-models` does not — it reports 160 invalid device **loads**, all in MLX's
own `affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false`. Those 160 are pinned
(next section), so a run that reproduces them exactly is green and prints the
census it accepted. The cause is in
`mlx/backend/metal/kernels/quantized.h`: `QuantizedBlockLoader::load_safe`
bounds its row index against the tile's column extent, so the guard never fires
and a transposed quantized matmul whose `N` is not a multiple of the kernel's
output tile width reads the out-of-range rows of the packed weight and the
scales. That width is **not** always 32: the non-batched `qmm_splitk` path
tiles at 32, the batched and gather `*_nax` paths at 64, so an `N` of 32 is
unaligned there and trips it too. rMLX passes correctly sized operands; the
reads are loads only and are shown bitwise not to reach the output — including
with the out-of-range rows held at NaN under the same kernel instantiation.

Two cautions that generalise beyond this kernel. A diagnostic names a *load* or
a *store* and the two differ in severity, so read the access mix the banner
prints rather than the total. And **absence of a diagnostic is not absence of an
out-of-bounds access**: the validation layer bounds against the MTLBuffer, not
the logical array, and MLX recycles buffers from size buckets, so a read past
an array's end that lands inside a roomier recycled allocation is silent. A
clean scan is evidence about reporting, not about memory safety.

The full investigation, the reproducer, the pinned-MLX `PYTHONPATH` needed to
repeat it, and the reason no caller-side workaround was taken live in
`.rmlx/mlx-qmm-t-tail-row-oob.md`.

#### The census pin

A gate that is red on every run is the inverse of a vacuous one: its exit code
stops carrying information, everyone learns to read `Error 1` as background
noise, and the next real hit arrives inside the standing one. That is what a
diagnostic from a kernel this repo neither compiles nor can fix would otherwise
do to `make gpu-test` and `make ci-perf`.

So the accepted hits are pinned, in `scripts/gpu_validation_census.txt`. One
line per (kernel, access kind), each carrying the exact count over a full run,
the crate and test it is reached from, the model snapshot it needs, and the
reference to the analysis that says it is benign. `run_gpu_tests.sh` diffs the
per-kernel tally it computed against that file:

| observed | verdict |
|---|---|
| every pinned entry at its exact count, nothing else | pass, printing the census it accepted |
| a kernel the pin does not name | fail — `not pinned: N <kind> "<kernel>"` |
| a pinned count that moved **up** | fail — `count moved up: … pinned N, observed M` |
| a pinned count that moved **down** | fail — the pin is stale, re-derive it |
| a pinned kernel that stopped firing | fail — same reason, from the pin's side |
| any store, pinned kernel or not | fail — a dropped write is corruption |

The comparison is a tally diff over the file's contents; no kernel name appears
in the script. A store is refused on the pin side too: there is no analysis that
makes an invalid write acceptable, so the runner rejects an entry that names one
rather than comparing it.

Two structural exemptions, both one-directional — an excess count, an unpinned
kernel and any store fail in every run:

* A **narrowed** run (`CRATE=` / `FILTER=`) visits a subset of the population and
  observes fewer hits by construction, so its shortfall says nothing about the
  pin. Enforcing the pinned total there would make every narrowed iteration red.
* An entry whose **snapshot is not on this machine** cannot fire. The suite's
  contract is already that a model-gated cell skips and counts as passed, so a
  developer without the weights is not told the pin is stale.

Both print `not enforced downward: …` with the reason, so a run that could not
check an entry never looks like one that did.

**Updating the pin** takes one of two things and nothing else: a linked upstream
reference showing the defect is not ours and does not reach our output, or an
analysis of the standard of the one above — the symbol's provenance, a
load/store census over *every* diagnostic rather than a sample, and a
demonstration that the loaded lanes do not reach the output. A passing test
suite is not that demonstration, and neither is a quiet run: the validation layer
bounds the MTLBuffer rather than the array, so absence of a diagnostic is not
absence of an out-of-bounds access. A count that dropped is never edited down to
fit the run in front of you — re-derive it from a full run and record what
changed.

`scripts/run_gpu_tests_selftest.sh` (in `make ci`, no GPU) pins each of those
verdicts against a stub runner, including the two exemptions and the pin-file
refusals, so the mechanism cannot quietly become one that only passes.

### `#[ignore]` is not a place to park a broken test

An ignored test runs only when someone asks for it, so a real failure can sit in
the tree indefinitely — that is the failure mode `make gpu-test` exists to close,
not one it makes impossible. A test edited until it goes green is the same defect
wearing a different hat: when a deliberate behaviour change makes an assertion
stale, re-point the assertion at the new contract and then mutation-check it
(revert the change; the repaired test must go red), rather than relaxing it.

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
literature reports. Numbers and their derivation live in `docs/KV_QUANT.md`
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
in `docs/KV_QUANT.md` § "Codec fidelity — measured".

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
