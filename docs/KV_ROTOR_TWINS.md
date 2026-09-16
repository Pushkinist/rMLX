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

The CPU store-bytes pin is the oracle that covers all eight.

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
  `quant_rotor_v_tests.rs` / `quant_rotor_k_tests.rs` as width-parameterised
  cases. Any test whose body differs from its 3-bit sibling only in the width
  token is deleted, not copied.

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

## 8. Baseline

`.rmlx/analysis/482/baseline/` (gitignored) holds `cells.csv` (36 cells),
`commands.txt` (the exact command line per cell, with the model root elided),
the per-cell raw logs, and `MANIFEST.sha256` covering all of them plus the
`rmlx` binary the capture ran on. The code chunk re-runs `capture.sh` against
the same snapshots and diffs `cells.csv`.
