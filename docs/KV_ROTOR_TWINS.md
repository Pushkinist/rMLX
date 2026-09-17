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

`scripts/debt_report.sh --matched-lines <population>` is the one producer of
this figure. It carried two populations, both hard-coded to `SPEC_DIR`
(`crates/rmlx-models/src/speculative`); it now carries four, each with its own
root and its own pairing rule, over the same `normalize()` and
`matched_lines()` the speculative populations use. No second matcher, no
second script, no new Make target.

**The rule**, verbatim from the header of `scripts/lib/debt_report.py`:

> * `rotor-storage` — the non-test `quant_rotor_*.rs` files under
>   `crates/rmlx-kv-quant/src/storage`, paired inside a group sharing the
>   filename stem with every digit run removed (`quant_rotor_v3` and
>   `quant_rotor_v4` -> `quant_rotor_v`): same axis, different width. A group
>   of one contributes an item and no pair, so a collapsed axis reads 0
>   matched lines with the population still found.
> * `rotor-updates` — the `update_rotor*` fns of
>   `crates/rmlx-kv-quant/src/kvcache/update.rs`, paired by the same
>   digit-stripped-name rule (`update_rotor_k_only_3` and `_4` ->
>   `update_rotor_k_only_`).
>
> Neither rotor population is a literal file or fn list: both are a glob plus
> a name rule, so the same command measures a tree that still carries the
> twins and one that does not.

Two properties the figure depends on:

* **The label names the population's own root.** `matched_lines_report`
  rendered every label beside `SPEC_DIR`, so a rotor figure would have printed
  a speculative path. The root is now a field of the population entry.
* **An empty population is `unavailable`, not `0`.** It prints to stderr and
  exits 1. A collapsed population — members present, no same-axis pair left —
  is a measured `0` and exits 0. The two are different answers and only the
  second is a measurement.

### Before, on the pre-collapse tree

Run from this branch with `--root` pointed at a separate worktree of the
branch point, so the tool is one version and the tree is the other:

```
rotor storage twins (crates/rmlx-kv-quant/src/storage): 814 matched lines over 2244 body lines (4 item(s), 2 pair(s))
rotor update twins (crates/rmlx-kv-quant/src/kvcache/update.rs): 293 matched lines over 610 body lines (8 item(s), 4 pair(s))
```

Per pair, storage: `quant_rotor_k3` <-> `quant_rotor_k4` 401 (70.0 %),
`quant_rotor_v3` <-> `quant_rotor_v4` 413 (75.2 %). Updates:
`update_rotor3` <-> `update_rotor4` 77 (97.5 %), `_sym` 81 (94.2 %),
`_k_only` 52 (91.2 %), `_k_asym` 83 (100.0 %).

### After, on the collapsed tree

```
rotor storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 1404 body lines (2 item(s), 0 pair(s))
rotor update twins (crates/rmlx-kv-quant/src/kvcache/update.rs): 69 matched lines over 146 body lines (8 item(s), 4 pair(s))
```

### Reconciliation with the figures computed by hand

| Population | By hand | By the tool | Difference |
|---|---|---|---|
| storage, before | 814 over 2244 | 814 over 2244 | none |
| updates, before | 313 over 650 | 293 over 610 | −20 matched, −40 lines |
| storage, after | 0 over 1394 | 0 over 1404 | −10 lines |
| updates, after | 0 over 234 | **69 over 146** | a different population |

* **Updates, before.** The hand run compared each fn from its `fn NAME(` line
  through its closing brace; the tool compares `FnInfo.body`, which starts at
  the opening brace, the same unit the two speculative populations are measured
  over. Each of the eight fns carries exactly five signature lines above that
  brace, and those five fold identical across a pair, so the hand figure is
  larger by 8 × 5 = 40 lines and 4 × 5 = 20 matched lines. Per pair the
  whole-fn count reproduces the hand numbers exactly (82 / 86 / 57 / 88).
  Nothing else differs.
* **Storage, after.** The hand run recorded 1394 file lines where the tool
  reads 1404. The 814 / 2244 before-figure reproduces to the line, and the
  after-figure's matched count is 0 either way, so the ten lines are a slip in
  the hand count, not a difference in the rule. The tool's number stands.
* **Updates, after.** The hand after-figure was taken over the *generic*
  bodies the entries now delegate to — one per family — which is not the
  population the rule names. Under the rule, the eight `update_rotor{3,4}*`
  entries still exist, and each pair still differs only in the width digit, so
  the population is 8 items and 4 pairs on both trees. What the collapse bought
  is the size of the bodies: 293 matched over 610 lines became 69 over 146.
  The twin is reduced by 76 %, not removed. Closing it means the eight entries
  collapsing onto a width-generic dispatch of their own, which this chunk did
  not do and no proof row claimed.

### The advisory half

`make debt-report`'s file-pair scan reached the storage twins on its own, at
70.0 % and 75.2 % shared. Those rows went with the deleted files. It never
reached the update bodies, which live in one file and so are invisible to a
file-pair scan, nor the summed matched-line count for either population —
which is what `--matched-lines` adds.

For scale, and outside the figure on both trees: the two surviving storage
files share 434 matched lines *across* the K/V axis, and the eight surviving
update entries share 228 across families. Neither is a twin under the rule —
they differ in what they store, not in a constant — and neither was ever in
the before-figure. (§11 recorded 430 for the first of those; the tool reads
434.)

`scripts/debt_report_selftest.sh` holds all of this: 19 of its 64 cases are
the two rotor populations, each asserting a planted figure, a measured 0 with
the population still found, and `unavailable` for a missing root and for a
root that is there and empty — reason and exit code both.

## 7. Removals

The collapse deleted, by file and by name:

* `storage/quant_rotor_v4.rs` — the whole file; its `QuantRotorV4` is now
  `pub type QuantRotorV4 = QuantRotorV<4>`.
* `storage/quant_rotor_k4.rs` — likewise for `QuantRotorK4`.
* `storage/quant_rotor_v3.rs` and `quant_rotor_k3.rs` — renamed to
  `storage/quant_rotor_v.rs` / `storage/quant_rotor_k.rs`, not deleted; the
  shared payload types (`RotorBlocks`, `RotorKBlocks`), their `BlockRows` impls
  and the two `synced_rotor_*_blocks` helpers stay there.
* `KvCache::update_rotor4`, `update_rotor4_sym`, `update_rotor_k_only_4`,
  `update_rotor_k_asym_4` in `kvcache/update.rs` — four bodies. All eight
  entries survive as the storage-variant resolver plus the warm-TTFT shortcut,
  over four shared bodies: `rotor_v_update`, `rotor_sym_update`,
  `rotor_k_only_k_side` and `rotor_k_asym_update`, each generic over the store's
  `BITS`.
* The twin halves of the storage test files: `quant_rotor_v4_tests.rs` and
  `quant_rotor_k4_tests.rs` are folded into `quant_rotor_v_tests.rs` /
  `quant_rotor_k_tests.rs`. What is deleted is the duplicated **body**, not the
  coverage: each such case is one generic body with the width as a parameter
  plus one `#[test]` per width, so the cell count does not fall.

**Deleted beyond the list above, and why it is not scope creep.** Once the
store is one type over both widths, the `update.rs` helpers that take a store
by reference cannot stay per-width — the shared body could not call them. Twelve
further twin pairs therefore became twelve fns: `ensure_rotor_v_table`,
`ensure_rotor_k_table`, `push_rotor_v_block`, `push_rotor_k_block`,
`rotor_gpu_append_into_v_blocks`, `rotor_gpu_append_into_k_blocks`,
`materialize_rotor_v_ring_tail`, `materialize_rotor_k_ring_tail`,
`rotor_v_sync_ring`, `rotor_k_sync_ring`,
`drop_blocks_when_ring_live_rotor_{v,k}`. In `rotorquant.rs` the two `bits ==
ROTOR3_BITS` forks inside the K encode path collapsed onto `rotor_encode` /
`rotor_decode`, and the K side gained the parametric entries the V side already
had (`rotor_k_encode_at`, `rotor_k_decode_at`); the four width-named `rotor{3,4}_k_*`
wrappers stay as the public spelling.

**Kept, though it is the same shape.** Nine twin pairs still exist once per
width. Four are the `pub(super)` fused-append entries
`rotor{3,4}_k_only_gpu_append` / `rotor{3,4}_sym_gpu_append` in `update.rs`:
they resolve a `KvStorage` variant exactly as the `update_*` entries do, and
they now call the generic helpers at their own width. The other five each bind
a per-width `.metal` kernel or its dispatch counter, and the kernels are out of
scope for this chunk:

| Pair | File |
|---|---|
| `rotor{3,4}_quant_kernel` | `crates/rmlx-kv-quant/src/rotorquant_msl.rs` |
| `rotor{3,4}_dequant_kernel` | `crates/rmlx-kv-quant/src/rotorquant_msl.rs` |
| `rotor{3,4}_fused_qk_sdpa` | `crates/rmlx-kv-quant/src/rotor_fused_qk_msl.rs` |
| `rotor{3,4}_fused_qk_dispatch_count` | `crates/rmlx-kv-quant/src/rotor_fused_qk_msl.rs` |
| `rotor{3,4}_flash_decode_dispatch_count` | `crates/rmlx-kv-quant/src/rotor_flash_decode_msl.rs` |

Each of the five names a distinct `.metal` entry point, so collapsing the Rust
side would have to collapse the kernels with it. That is a separate change.

Not deleted, and stated so the list is not read as covering them:

* No `KvQuant` variant, no `KvStorage` variant, no CLI spelling.
* No `check-*` Make target. None keys on the rotor storage file names; the
  `check-kv-codec-disposition` gate keys on `ALL_KV_QUANTS` plus the
  disposition predicates, which are untouched.
* No `.metal` kernel. `rotorquant_{quantize,dequantize}_rotor{3,4}.metal`
  remain four files; the issue puts them out of scope.
* No `docs/KV_QUANT.md` section. The rotor sections describe codecs, not
  storage types; the paths and type names they cite that moved are corrected in
  the same commit rather than appended to.
* `scripts/lib/debt_report.py` is **not** extended with a rotor population —
  that is the next chunk's work (§6 records the design). The after-figure in
  §11 is hand-run through the same module's `normalize()` / `matched_lines()`,
  the way §6's before-figure was.

## 8. Mutations

The oracle in §4 is only as good as what it can turn red. Every mutation below
was applied to the tree, run, and reverted from a pre-mutation snapshot whose
sha256 was re-verified afterwards. The file names are the **pre-collapse**
tree's; §11 records the re-run against the unified bodies. "Uncaught" rows are
the file's blind spots and are restated in the test module doc.

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
gate. §11 measures that claim on `gpu_append`.

## 9. Carried over into the unified bodies

* `update.rs`, the `update_rotor4_sym` doc comment, said "no MSL kernel for
  rotor4" while the body a few lines above dispatched
  `rotor4_gpu_append_into_blocks`. The comment was stale. It was **not** edited
  in the test chunk — that file is engine code — and the unified body did not
  carry it forward: the collapse writes one doc comment per unified fn, stating
  what the fn does now. §11 records that the sentence is gone from the tree.

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

## 11. What landed

The collapse, measured on the branch.

### Net line count

| Population | + | − | net |
|---|---|---|---|
| `crates/rmlx-kv-quant/src/storage/` (source) | 155 | 990 | **−835** |
| `crates/rmlx-kv-quant/src/storage/` (its `*_tests.rs`) | 235 | 668 | **−433** |
| `crates/rmlx-kv-quant/src/kvcache/update.rs` | 524 | 912 | **−388** |
| **total** | **914** | **2570** | **−1656** |

From `git diff --numstat -M` against the branch point, so a renamed file counts
as the edit it carries and not as a whole file added and a whole file deleted.
`rotorquant.rs` is +70 / −20 on top, for the two parametric K entries and the
two collapsed `bits ==` forks. The issue expected roughly −900; the difference
is the twelve helper pairs in §7 and the folded test bodies, neither of which
its estimate covered.

### Duplication, after

The figure is no longer hand-run. `scripts/debt_report.sh --matched-lines
rotor-storage` and `--matched-lines rotor-updates` produce it, over the rule
§6 states; the numbers below are that tool's output on this tree, and §6
reconciles them against what was measured by hand here first.

```
rotor storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 1404 body lines (2 item(s), 0 pair(s))
rotor update twins (crates/rmlx-kv-quant/src/kvcache/update.rs): 69 matched lines over 146 body lines (8 item(s), 4 pair(s))
```

The storage twin is closed: one file per axis, no same-axis pair left, a
measured 0 with the population still found. **The update twin is not.** This
subsection first recorded `0 over 234 body lines` for it, which was measured
over the generic bodies the entries delegate to — one per family — and not
over the `update_rotor*` entries the population names. Those eight entries
survive, each pair still differing only in the width digit, so the population
is 8 items and 4 pairs before and after; what the collapse bought is their
size, 293 matched over 610 lines down to 69 over 146. A 76 % reduction, not a
removal.

`make debt-report`'s file-pair scan no longer reports a rotor row at all; its
`70.0 %` and `75.2 %` entries went with the deleted files.

### Mutations, re-run against the unified bodies

Four of §8's edits re-applied to the collapsed files, each from a snapshot whose
sha256 was re-verified after the revert (`git checkout --` is not used: it would
revert uncommitted work, see the mutation-harness trap).

| # | Edit, on the unified file | Result |
|---|---|---|
| §8 M1 | `quant_rotor_v.rs` `append` — encode at `ROTOR3_BITS` instead of `BITS`, i.e. the generic instantiated at the wrong width | RED, 4 of the 5 pin tests, first line `rotor4 decode: rotor: code plane holds 312 words for 24 rows, which need 408` |
| §8 M3 | `quant_rotor_k.rs` `append` — `make_rotor_table(head_idx, layer_idx, …)`, arguments swapped | RED, the pin **only**: `rotor3_sym @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved` |
| §8 M8 | `quant_rotor_v.rs` `truncate_to` — `(n - 1).max(0)` | RED, the pin's **truncate** column: `rotor3 @ kv_h=1 head_dim=128: packed store bytes after truncate_to moved` |
| §8 M7 | `quant_rotor_v.rs` `gpu_append` — `n_groups + 1` | **GREEN** against the CPU pin, 5 passed, exactly as before the collapse. **RED** under `make gpu-test CRATE=rmlx-kv-quant` — see below |

M1, M3 and M8 name the 3-bit cell where the pre-collapse run named the 4-bit one,
which is the collapse working: one body, so either instantiation surfaces it.

M7 is the ring-side geometry: `gpu_append`, `gpu_packed_view`, the ring-readback
branch of the two sync helpers, `from_cpu_blocks` and `try_deep_clone` are not
reachable from a `Device::Cpu` drive, so the CPU pin cannot see them. The claim
that the GPU suite is the gate over them was measured, not assumed: M7 was
re-applied from a sha256-verified snapshot and `make gpu-test
CRATE=rmlx-kv-quant` was run on an idle GPU. It is **red**, 225 passed and 2
failed, on:

* `kvcache::rotor_flash_dispatch_tests::rotor_sym3_multi_token_append_after_fused_decode_drops_the_ring`
* `kvcache::rotor_flash_dispatch_tests::rotor_sym4_multi_token_append_after_fused_decode_drops_the_ring`

with the decisive line

```
decode update_and_sdpa: Quant("QuantKGpuRing::seed_from_cpu: CPU prefix length mismatch at filled_seq=24 (codes 624 want 624, scales 2064 want 2112, norms 48 want 48)")
```

Both widths fail, which is again the collapse working. The file was restored and
its sha256 re-verified against the snapshot. So the ring-side bodies are covered
— by the GPU suite only, never by a CPU drive.

### The GPU-ring question, closed by reading the diff

§1 predicted "it already did". The diff confirms it: `gpu_append`,
`gpu_packed_view`, `from_cpu_blocks` and `try_deep_clone` were byte-identical
between the two widths modulo the width token, so the unified body is the same
code at both instantiations. The 4-bit path takes no route it did not take
before, and the 3-bit path takes none it did not either.

### The gates, as run

| §5 row | Result |
|---|---|
| 2 | `rotor_store_bytes_tests.rs`: 5 passed, 0 failed. The file is untouched in the branch diff — **no pin was re-baselined**. |
| 4 | The 36-cell served capture re-run and diffed under §10's key: identical in every column except `binary_sha256`, which differs. `exit_code` is `0` and `n_ids` is `200` on both sides. The `none` control is unchanged in all four of its cells. |
| 6 | The table above. |
| 7 | §7, rewritten rather than appended to. |

§9's stale sentence is gone: no `.rs` file in the tree now says "no MSL kernel
for rotor4".

`cargo test -p rmlx-kv-quant` reports 566 passed, 0 failed, 257 ignored before
the collapse and the same after, five runs on each side. The cell count did not
fall: the folded test bodies kept one `#[test]` per width.

The CI gates the code chunk owns are green: `make fmt-check`, `make lint`,
`cargo check --workspace --all-targets`, `make check-doc-source-citations`
(242 cited paths resolve), `make check-no-inline-tests`,
`make check-gpu-tests-ignored`, `make check-kv-codec-disposition` (28 codecs
classified, 17 inert) and `make check-kv-layer-quants`.

### `make ci-perf`

Invoked as `make ci-perf`, in the foreground, on an idle GPU. 88 minutes. No
test failed in either half.

The census verdict:

```
shader validation: census matches the pin (scripts/gpu_validation_census.txt)
```

The GPU suite, per crate, as banner-selected / libtest-passed: `rmlx-audio`
7 / 7, `rmlx-kv-quant` 253 / 253, `rmlx-kv-ssd` 12 / 12, `rmlx-mlx` 10 / 10,
`rmlx-models` 100 / 101 — libtest's filters are substrings, so a name that is a
prefix of another selects both, and the extra cell passed too.

The last two lines, verbatim:

```
OK: 383 GPU tests passed across 5 workspace member(s), shader validation matches the pinned census. — INCOMPLETE: 24 selected GPU test(s) stood down and 9 further notice(s) named no test; they asserted nothing (listed above)
ci-perf INCOMPLETE — the GPU suite did not run every gate it names (see above)
```

Every stand-down names an unset environment variable — `RMLX_TEST_MODEL_QWEN36`,
or the one-variable drafter case `CLAUDE.md` already records. This is the
INCOMPLETE the repository ends on today; it is not new, and no stand-down names
a rotor test. `rmlx-kv-quant`'s 253 are the gate over the ring-side bodies §8's
M7 showed the CPU pin cannot see, and all 253 passed.
