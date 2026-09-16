# Rotor storage and update twins

The 3-bit and 4-bit rotor storage types and their `update_*` bodies are the
closest twin pair in `rmlx-kv-quant`, and the pilot for the twin rule
(`CLAUDE.md` §Simplicity rules 6). This doc holds the measured premise, the
oracle the collapse is judged by, and the plan the collapse executes.

The oracle itself is
`crates/rmlx-kv-quant/src/kvcache/rotor_store_bytes_tests.rs`.

## 1. Premise, re-measured against the tree

The numbers below are re-derived at the branch point, not restated. Method:
apply the three renames (`rotor3`↔`rotor4`, `ROTOR3_BITS`↔`ROTOR4_BITS`,
`V3`↔`V4`) to both members of a pair, then count unified-diff changed lines at
zero context.

| Pair | Lines now | Differing now | Verdict |
|---|---|---|---|
| `update_rotor3` / `update_rotor4` | 84 / 84 | 4 | MOVED (was 100 / 97, 29) |
| `update_rotor3_sym` / `update_rotor4_sym` | 93 / 89 | 20 | MOVED (was 101 / 104, 35) |
| `update_rotor_k_only_3` / `_4` | 63 / 61 | 20 | new row; the issue's table omits this pair |
| `update_rotor_k_asym_3` / `_4` | 88 / 88 | 8 | MOVED (was 100 / 89, 21) |
| `quant_rotor_v3.rs` / `v4.rs` | 648 / 450 | 294 | MOVED (was 631 / 436, 313) |
| `quant_rotor_k3.rs` / `k4.rs` | 700 / 446 | 408 | new row; same shape as the V pair |
| `rotorquant_dequantize_rotor3.metal` / `rotor4` | 45 / 42 | 11 | MOVED (was 52 / 47, 13) |

Two premise claims do **not** hold.

* **"V3 has `gpu_append`, `rows`, `retain_rows`; V4 has not" — FALSE.**
  `QuantRotorV4` and `QuantRotorK4` both carry `gpu_append` and
  `gpu_packed_view`. What lives only in the 3-bit file is the *shared* payload
  type (`RotorBlocks` / `RotorKBlocks`), its `impl BlockRows` (which is where
  `rows` and `retain_rows` are), and the ring-sync helper
  (`synced_rotor_v_blocks` / `synced_rotor_k_blocks`) — all of which the 4-bit
  file imports and uses. Stripping comments and blank lines and applying the
  renames, `quant_rotor_v3.rs` and `quant_rotor_v4.rs` differ by exactly four
  hunks: the `use` line, the `RotorBlocks` + `BlockRows` definitions, the
  `synced_rotor_v_blocks` definition, and the test-module path. The K pair
  differs by the same four plus one `use`.

  **Consequence for the plan:** there is no behaviour asymmetry between the
  widths, so the unification has **no intended behaviour change**, and the
  issue's proof row "state whether the 4-bit path now takes the GPU ring append
  that only V3 had" resolves to "it already did". That row stays in the proof
  table below, answered rather than dropped.

* **"`rotorquant.rs` branches on a runtime `bits` value (line ~1032)" — HELD,
  MOVED.** The branch is at `rotorquant.rs:796` and `:805`
  (`if bits == ROTOR3_BITS`). The codec math is bit-width-parametric today.

One further fact the issue's list does not carry: `ROTOR4_GROUP_SIZE` is
defined as `ROTOR3_GROUP_SIZE` (`rotorquant.rs:118-122`), so the pair differs
in **one** scalar, the bit width. A single `const BITS: u8` parameter is
sufficient; no second parameter is needed.

## 2. The generic types

```
storage/quant_rotor_v.rs   pub struct QuantRotorV<const BITS: u8>
storage/quant_rotor_k.rs   pub struct QuantRotorK<const BITS: u8>
```

with `pub type QuantRotorV3 = QuantRotorV<3>;` and siblings kept as aliases so
no caller outside `rmlx-kv-quant` changes in this chunk. `RotorBlocks`,
`RotorKBlocks`, their `BlockRows` impls and the two `synced_rotor_*_blocks`
helpers are already width-agnostic and move as-is.

The one width-dependent call inside each body is the encode/decode entry
(`rotor3_encode` / `rotor4_encode`, `rotor3_decode` / `rotor4_decode`). Those
already delegate to `rotor_encode(..., bits)` / `rotor_decode(..., bits)`, so
the generic body calls the parametric form with `BITS` and the four
width-named wrappers stay as the public spelling.

`kvcache/update.rs`: the eight `update_rotor*` bodies become four, each
generic over the store's `BITS`. The `KvStorage` variants stay eight — a
variant is a spelling, and no spelling is removed (issue §Sequencing).

Rule 1 of the simplicity rules still governs: the generic is introduced
because there are two instantiations of every item, not to admit a third.

## 3. What cannot move

Verbatim from the test module doc
(`crates/rmlx-kv-quant/src/kvcache/rotor_store_bytes_tests.rs`):

> Every rotor spelling that exists before the storage/update unification exists
> after it, and for each spelling, at each shape:
>
> * the **packed store bytes** are identical — every `codes` word, every
>   `scales` and `norms` float, the rotor table, the QJL residual planes, the
>   affine/turbo companion plane, the block boundaries and the accumulated
>   shape;
> * the **K and V rows the attention receives** from that store are identical,
>   bit for bit, at the chunk append and at every decode step;
> * `resident_bytes()` is identical;
> * and the served temp-0 token stream is identical.

There is **no named exception**. Per §1, the 4-bit path is not inheriting a
GPU append path it lacked; both widths already have one.

## 4. Why the served digest is not the oracle

`KvQuant::materialises_packed_store()` is false for `Rotor3`, `Rotor4`,
`RotorK3Asym` and `RotorK4Asym`, so `exit_prefill` clears their payload and
decode runs off the bf16 mirror. Measured on the branch point, at
`--max-tokens 200`, those four produce the same token-id digest **and** the
same `kv_cache_bytes` as `--kv-quant none`, on both `gemma-4-e2b` and
Ternary-Bonsai-8B, at both 4k and 32k. A served digest is therefore a gate
that cannot fail on half the population under test.

The four that do read the store at decode — `rotor3_sym`, `rotor4_sym`,
`k_rotor3`, `k_rotor4` — separate from `none` at every cell. Two further
cautions from the same capture: `k_rotor3` and `rotor3_sym` share a digest at
4k on both models (they diverge at 32k), so a 4k-only digest table would
conflate two spellings.

The CPU store-bytes pin is the oracle that covers all eight, plus two extra
asymmetric-V configurations. It is not exhaustive, and §8 lists what it cannot
see: the `exit_prefill` bulk-encode arms (the route production takes for the
four live spellings), `gpu_append` / `gpu_packed_view` and the ring-readback
branch of the two sync helpers, `from_cpu_blocks` / `try_deep_clone`, and six of
the ten legal `rotor_k_*_asym_*` V configurations.

## 5. Proof rows for the code chunk

| # | Row | Who runs it |
|---|---|---|
| 1 | `make ci` green | code chunk |
| 2 | `crates/rmlx-kv-quant/src/kvcache/rotor_store_bytes_tests.rs` green, pins unchanged (no re-baseline) | code chunk |
| 3 | `make ci-perf`: census matches, every GPU test passes, the `INCOMPLETE` line quoted verbatim | code chunk |
| 4 | Served temp-0 digest + `kv_cache_bytes` per cell equal to `.rmlx/analysis/482/baseline/cells.csv`, 36 cells, each exactly 200 ids | code chunk |
| 5 | The 4-bit GPU-ring question, answered: both widths already had `gpu_append`; §1 | answered here |
| 6 | Net line count of `storage/` and `kvcache/update.rs` | code chunk |
| 7 | `Removals` section, §7 below | code chunk |
| 8 | Decode TPS within ±1 % of the recorded anchor per cell | **owner-gated** |

Row 8 is a performance number. It is not measured in the test chunk and is not
measured by the code chunk on its own initiative; it is requested from the
owner as one batched ask alongside the code chunk's proof, together with the
cell list it would cover.

On row 3: the owner has decided to split `make ci-perf` by what it guards, but
that split does not exist yet. The code chunk therefore runs the full
`make ci-perf` — invoked directly, so no variable narrows it — and reports all
three facts. `ci-perf` ends `INCOMPLETE` today on a host holding every
snapshot, for the one-environment-variable reason recorded in `CLAUDE.md`;
that line is quoted rather than paraphrased, so a *new* incompleteness is
visible against it.

## 6. Duplication figure

`scripts/debt_report.sh --matched-lines {drivers,impls}` is the one producer of
the campaign's duplication figure, but both of its populations are hard-coded
to `SPEC_DIR` (`crates/rmlx-models/src/speculative`). **It cannot produce a
before-figure for the rotor population as it stands.**

The advisory half of the same script *does* already reach the rotor file
twins, because the file-pair scan runs over `crates/rmlx-kv-quant`. Its current
output at the branch point:

```
crates/rmlx-kv-quant/src/storage/quant_rotor_k3.rs <-> .../quant_rotor_k4.rs: 70.0% shared
crates/rmlx-kv-quant/src/storage/quant_rotor_v3.rs <-> .../quant_rotor_v4.rs: 75.2% shared
```

What it cannot reach is (a) the summed matched-line count for that population,
and (b) the eight `update_*` bodies, which live in one file and so are invisible
to a file-pair scan.

**Smallest extension that keeps one producer** (design item for the code
chunk): give `--matched-lines` a population argument rather than a second tool
— add a `rotor-storage` and a `rotor-updates` entry to
`MATCHED_LINES_POPULATIONS` in `scripts/lib/debt_report.py`, each a collector
over its own directory/fn set, reusing the existing `normalize()` and
`matched_lines()`. No new normalisation, no new script, no new Make target
beyond what `debt-report-selftest` already covers.

Two constraints on that edit, both of which a naive version gets wrong:

* `matched_lines_report` renders `f"{label} ({SPEC_DIR}): …"` — the directory in
  the label is a module constant, so a rotor population would print a
  speculative path beside a rotor figure. **The label's directory must come from
  the population entry**, alongside its label and collector, not from
  `SPEC_DIR`.
* `debt_report_selftest.sh` asserts what each case found, not that the tool ran.
  **Each new population gets its own selftest case**, asserting a planted
  matched-line figure over a synthetic fixture and the `unavailable` path when
  its directory is missing — the same two directions the existing populations
  are held to.

**Before-figure, by hand this once**, computed by importing that module's own
`normalize()` and `matched_lines()` so the normalisation is not retyped:

| Population | Pairs | Matched lines | Over |
|---|---|---|---|
| rotor storage twins | `v3↔v4`, `k3↔k4` | 814 | 2244 file lines |
| rotor `update_*` twins | the four 3↔4 pairs | 313 | 650 body lines |

Per pair: `v3↔v4` 413 (75.2 %), `k3↔k4` 401 (70.0 %); `update_rotor3↔4` 82
(97.6 %), `_sym` 86 (94.5 %), `_k_only` 57 (91.9 %), `_k_asym` 88 (100.0 %).

## 7. Removals

The code chunk deletes, by file and by name:

* `crates/rmlx-kv-quant/src/storage/quant_rotor_v4.rs` — the whole file; its
  `QuantRotorV4` becomes `QuantRotorV<4>`.
* `crates/rmlx-kv-quant/src/storage/quant_rotor_k4.rs` — likewise for
  `QuantRotorK4`.
* `crates/rmlx-kv-quant/src/storage/quant_rotor_v3.rs` and
  `quant_rotor_k3.rs` — renamed to `quant_rotor_v.rs` / `quant_rotor_k.rs`,
  not deleted; the shared payload types and sync helpers stay.
* `KvCache::update_rotor4`, `update_rotor4_sym`, `update_rotor_k_only_4`,
  `update_rotor_k_asym_4` in `kvcache/update.rs` — four bodies, replaced by the
  generic form of their 3-bit siblings.
* The twin halves of the storage test files:
  `quant_rotor_v4_tests.rs` and `quant_rotor_k4_tests.rs` are folded into
  `quant_rotor_v_tests.rs` / `quant_rotor_k_tests.rs`. What is deleted is the
  duplicated **body**, not the coverage: a test whose body differs from its
  3-bit sibling only in the width token becomes one body with the width as a
  parameter, still run at both widths. The **cell count does not fall.** The PR
  reports `cargo test -p rmlx-kv-quant` passed-test counts before and after, and
  a count that drops is a deleted case, not a collapsed twin.

The deletions above make this section's own `crates/...` paths dangle, which
`make check-doc-source-citations` fails on. The code chunk corrects §7 in the
same commit that deletes the files — it is a generated-from-fact section, not a
historical record.

Not deleted, and stated so the list is not read as covering them:

* No `KvQuant` variant, no `KvStorage` variant, no CLI spelling.
* No `check-*` Make target. None keys on the rotor storage file names; the
  `check-kv-codec-disposition` gate keys on `ALL_KV_QUANTS` plus the
  disposition predicates, which are untouched.
* No `.metal` kernel. `rotorquant_{quantize,dequantize}_rotor{3,4}.metal`
  remain four files; the issue puts them out of scope, and folding their
  constants into Metal function constants is a separate change with its own
  `probes/kernels.manifest` and native-compile consequences.
* No `docs/KV_QUANT.md` section. The rotor sections describe codecs, not
  storage types; the type names they cite are checked by
  `make check-doc-source-citations` (paths only), and any cited path that moves
  is corrected in the same commit rather than appended to.

## 8. Mutations

The oracle in §4 is only as good as what it can turn red. Every mutation below
was applied to the tree, run, and reverted from a pre-mutation snapshot whose
sha256 was re-verified afterwards. "Uncaught" rows are the file's blind spots
and are restated in the test module doc.

| # | Edit | Caught by |
|---|---|---|
| M1 | `storage/quant_rotor_v4.rs` `append` — encode through `rotor3_encode` (a generic instantiated at the wrong width) | `rotor_store_bytes_are_pinned_per_spelling_and_shape`, `rotor_store_geometry_follows_the_codec_bit_width`, `three_and_four_bit_twins_hold_different_stores`, `a_rotor_cell_is_reproducible`; first line `rotor4 decode: code plane holds 312 words for 24 rows, which need 408` |
| M2 | `storage/quant_rotor_v4.rs` `new` — `bits: 3` instead of `ROTOR4_V_BITS` | `rotor_store_geometry_follows_the_codec_bit_width` (bit tag) and the pin |
| M3 | `storage/quant_rotor_k4.rs` `append` — `make_rotor_table(head_idx, layer_idx, …)`, arguments swapped | the pin **only** (`rotor4_sym @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`); geometry stays green |
| M4 | `storage/quant_rotor_v3.rs` `append` — drop `transpose_heads_seq`, store head-major | the pin, **at shape B only** (`rotor3 @ kv_h=4 head_dim=96`) |
| M5 | `storage/quant_rotor_v4.rs` `append` — drop the last `norms` entry | geometry (norm count) and the pin; first line `decoded 2944 elems but shape [1, 1, 24, 128] implies 3072` |
| M6 | `storage/quant_rotor_v4.rs` `dequant` — drop `transpose_chunked_seq_heads`; store untouched | the pin's **rows** assertion only, at shape B (`rotor4 @ kv_h=4 head_dim=96: the K/V rows attention receives moved`) |
| M7 | `storage/quant_rotor_v4.rs` `gpu_append` — `n_groups + 1` | **uncaught** — exit 0, all tests green |
| M8 | `storage/quant_rotor_v4.rs` `truncate_to` — `(n - 1).max(0)` | the pin's **truncate** column (`rotor4 @ kv_h=1 head_dim=128: packed store bytes after truncate_to moved`) |
| M9 | `quant.rs` — a ninth rotor variant (`Rotor5`) added to the enum, to `ALL_KV_QUANTS` and to `Display`, wired through all 26 exhaustive matches | `every_rotor_spelling_is_pinned_at_both_shapes` (`rotor spelling census moved … ["rotor3", "rotor4", "rotor5", …]`) and `rotor_store_bytes_are_pinned_per_spelling_and_shape` (`no pin for 2 cell(s)`) |

M9 is the regression test for the census itself: with a hand-written
`matches!` variant list in place of the `Display`-derived filter, `rotor5` never
enters the population and both tests stay green with nothing pinned.

M3, M4, M6 and M8 are each caught by exactly one assertion, which is why all
four columns and both shapes are load-bearing rather than redundant.

Uncaught by inspection, not measured, same class as M7: `gpu_packed_view`, the
ring-readback branch of `synced_rotor_v_blocks` / `synced_rotor_k_blocks`,
`from_cpu_blocks` and `try_deep_clone`. All are `Device::Gpu` or SSD-hydrate
paths that a CPU drive never enters; `make ci-perf`'s GPU suite is their only
gate.

## 9. Carried over into the unified bodies

* `update.rs`, the `update_rotor4_sym` doc comment, says "no MSL kernel for
  rotor4" while the body a few lines above dispatches
  `rotor4_gpu_append_into_blocks`. The comment is stale. It is **not** edited in
  the test chunk — that file is engine code — but the unified body must not
  carry it forward; the collapse writes one doc comment per unified fn, and it
  states what the fn does now.

### The QJL toggle's full residency term

Adding a long CPU test to `crates/rmlx-kv-quant/src/kvcache/` turned
`warm_ttft_cross_codec_tests::shares_kv_moves_only_the_mixed_machinery` red in
three runs of four: `Rotor4Sym: residency must be byte-identical under both
topologies, left 1107296 right 1000800`. That sweep builds every codec twice and
took no `env_lock`, so a sibling test toggling `RMLX_ROTOR_QJL` between the two
arms built them as two different codecs. It now takes the lock.

The delta is 106 496 B, measured at the sweep's geometry (`kv_h = 8`,
`head_dim = 128`, prefill 256 → 2048 rows) by building the same cache with the
toggle off and on:

| Term | Bytes | Shape |
|---|---|---|
| `qjl_s_matrix` | 65 536 | `[head_dim, head_dim]` f32 — static, one per store |
| `qjl_codes` | 32 768 | 2048 rows × `head_dim / 8` packed sign bytes |
| `qjl_norms` | 8 192 | 2048 rows × one f32 |
| **total** | **106 496** | equals the observed delta exactly |

The per-row planes alone are 40 960 B; the static projection matrix is the other
65 536 B and is the term easy to miss. Both are allocated at the first append
under the **same** `rotor_qjl_enabled()` read, so there is no second term
outside the lock's governance and the fix is complete.

## 10. Baseline

`.rmlx/analysis/482/baseline/` (gitignored) holds `cells.csv` (36 cells),
`commands.txt` (the exact command line per cell, with the model root elided),
the per-cell raw logs, and `MANIFEST.sha256` covering all of them plus the
`rmlx` binary the capture ran on. The code chunk re-runs `capture.sh` against
the same snapshots and diffs `cells.csv`.

**Diff key.** A cell is `(model, prompt_tokens, codec)`. The before/after
comparison is over **every column except `binary_sha256`**, which must differ —
a run whose binary digest did not change did not test the change. The two binary
digests are reported separately, beside the diff, so "identical rows" and "same
binary" cannot be confused. `exit_code` must be `0` and `n_ids` exactly `200` in
both captures; a row missing either is a stop, not a difference.

**What the PR may quote.** `commands.txt` only. The per-cell logs under `raw/`
carry the absolute model-snapshot path and must never be pasted into a commit
message, a PR body, an issue or any other public surface. `commands.txt` is
written with the model root elided for exactly this reason.
