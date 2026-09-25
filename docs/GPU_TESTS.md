# GPU and Metal tests

This doc covers the tests that drive the GPU: the `#[ignore]` rule and its
gate, `make gpu-test`, `make ci-perf`, Metal shader validation and the census
pin. Snapshot resolution, the test variables and the CPU gates are in
[`TESTING.md`](TESTING.md).

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

### Exempting a device-as-value test

A test that passes `Device::Gpu` to a pure function as a plain value, and never
dispatches Metal, opts out with a line-leading `// gpu-test-gate: exempt` in its
own attribute block. The marker covers that one fn; a copy inside a fn body
exempts nothing. Inside a `macro_rules!` body it covers every cell the macro
generates, so audit such a marker against every invocation.

### An `#[ignore]` that claims Metal and cannot prove it is fatal

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

### Declaring a Metal route the scanner cannot follow

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

### The declared routes, and what covers them

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

### Macro-generated tests

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

### What the scanner cannot see

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

## Running the GPU tests: `make gpu-test`

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
  touching Metal (`TESTING.md` § "`RMLX_SKIP_GPU` opt-out");
- another MLX process is live (`pgrep -f 'rmlx serve|mlx_lm|paroquant|omlx'`).

A failing test is never on a known-red list; the runner keeps none. Before
blaming a failure on a change, re-run the same crate and filter on a clean
checkout of the base commit and compare.

The runner reports every red it found before it exits: shader-validation hits,
failing tests, under-matched crates and crates with no validation banner.
`make gpu-runner-selftest` (`scripts/run_gpu_tests_selftest.sh`, in `make ci`
and hosted CI) pins each report against stub crates, with no GPU.

### A cell that stood down is reported, and it is not a pass

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
`TESTING.md` § "Golden-token suites: how their snapshot resolves". In the
integration harness, a root that is set but missing fails, an absent slug
skips, and a fallback that is used prints `NOTE <test>: …`.

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

### Splitting the GPU suite by what it guards

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

### Where it runs: `make ci-perf`, not `make ci`

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

## Metal shader validation (on by default here)

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

### The census pin

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

## `#[ignore]` is not a place to park a broken test

An ignored test runs only when someone asks for it, so a failure can sit
unseen. When a deliberate behaviour change makes an assertion stale, re-point
it at the new contract. Then mutation-check it: revert the change, and the
repaired test must go red. Do not relax it.
