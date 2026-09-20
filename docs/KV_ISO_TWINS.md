# Iso storage and update twins

The 3-bit and 4-bit iso storage types and their `update_*` bodies are the
second width-twin pair in `rmlx-kv-quant`, after the rotor pilot.

**The method is not restated here.** [`docs/KV_ROTOR_TWINS.md`](KV_ROTOR_TWINS.md)
holds it: how a premise is re-measured against the tree, why a served digest is
not the oracle, what a store-bytes pin is, how a mutation run is conducted, and
how the duplication figure is produced. This doc holds only what the iso family
does differently — the drift, the GPU decision the collapse forces, the fidelity
bound, the iso mutation list, and the removals owed.

**Why a second doc rather than a widened one.** The method is about a tenth of
the rotor doc's text; the premise, the mutation list and the removals are the
other nine tenths and are per-family. A merged "width twins" doc would carry two
premises, two mutation lists and two removal lists under one shared method, and
a reader looking for the iso removals would page past the rotor ones. The rotor
doc is already 41 KB against the 200 KB advisory ceiling
(`make debt-report`), so splitting also keeps both files inside it.

The oracle itself is
`crates/rmlx-kv-quant/src/kvcache/iso_store_bytes_tests.rs`.

## 1. Premise, re-measured against the tree

Re-derived at the branch point, not restated. Method: strip the width digit from
every iso token in both members of a pair (`iso3`/`iso4` -> `iso`, `ISO3_`/`ISO4_`
-> `ISO_`, `IsoV3`/`IsoV4` -> `IsoV`, `IsoK3`/`IsoK4` -> `IsoK`), then count
unified-diff changed lines at zero context. The "stripped" column repeats the
count with comment-only and blank lines removed, because the two iso storage
files carry very different amounts of prose.

| Pair | Lines now | Differing now | Stripped | Verdict |
|---|---|---|---|---|
| `quant_iso_v.rs` / `quant_iso_v4.rs` | 1566 / 407 | 1298 | 935 / 271, 728 | MOVED (issue: 1551 / 402, 1351) |
| `quant_iso_k.rs` / `quant_iso_k4.rs` | 618 / 393 | 341 | 341 / 259, 112 | MOVED (issue: 577 / 388, 323) |
| `update_iso3` / `update_iso4` | 131 / 80 | 73 | — | new row; the issue's table omits this pair |
| `update_iso3_sym` / `update_iso4_sym` | 117 / 72 | 87 | — | MOVED (issue: 130 / 86) |
| `update_iso_k_only_3` / `_4` | 92 / 40 | 62 | — | new row; the issue's table omits this pair |
| `isoquant_dequantize_iso3.metal` / `iso4` | 37 / 36 | 2 | 27 / 27, **0** | MOVED (issue: 3) |
| `isoquant_quantize_iso3.metal` / `iso4` | 59 / 62 | 28 | 39 / 39, **2** | new row |

The `update_*` line counts are fn **bodies**, brace to brace, which is the unit
`scripts/lib/debt_report.py` measures and the unit §5 quotes. The issue's
"86 lines against 130" is the whole fn including its signature and doc comment.

Two premise claims do **not** hold as written.

* **"The 4-bit store is CPU-only" — FALSE.** `QuantIsoV4` and `QuantIsoK4` both
  carry `gpu: QuantKGpuRing` and the three ring entries `gpu_append`,
  `gpu_packed_view` and `reconcile_ring`. That ring is what the iso
  flash-decode kernels read, and it is live at both widths. What lives only in
  the 3-bit files is a different mechanism (next bullet) and the shared payload
  type `IsoBlocks` with its `BlockRows` impl and the `synced_iso_v_blocks`
  ring-sync helper, which `quant_iso_v4.rs` imports and uses.

* **"The 3-bit store carries the GPU-resident mirror" — HOLDS, and the mirror
  is dead.** `QuantIsoV3::append_gpu` guards its mirror write on
  `crate::gpu_resident_iso_enabled()`, which is a `false` constant under
  `#[cfg(not(test))]` (`crates/rmlx-kv-quant/src/lib.rs`). The mirror writes no
  byte in production at either width. `docs/KV_QUANT.md` records the bench
  decision behind that: "hardcoded OFF (bench decision — bench showed no
  measurable benefit on the warm-TTFT path where the bf16 seed absorbs the
  dequant)".

### What the 4-bit path reads at decode, today

This is the question the issue's title turns on, and the answer is not the one
the title implies.

| Spelling | Production decode route | Reads |
|---|---|---|
| `iso3`, `iso4` | `update_decode_fp16` | the bf16 mirror, both axes. `decode_reads_packed_store()` is false and `exit_prefill` clears the payload. Identical at both widths. |
| `iso3_sym`, `iso4_sym` | `update_and_sdpa` arm 1d' -> `update_and_sdpa_iso_sym_fused` | the **packed store, on GPU**, through `iso_flash_decode_symv`. `iso_sym_flash_over_store` selects bits 3 or 4 off the live `KvStorage` variant. |
| `k_iso3`, `k_iso4` | `update_and_sdpa` arm 1d -> `update_and_sdpa_iso_k_fused` | the **packed store, on GPU**, through `iso_flash_decode`. `iso_k_flash_over_store` selects bits 3 or 4 the same way. |

Both fused arms match `IsoKOnly3 | IsoKOnly4` and `IsoSym3 | IsoSym4` alike
(`crates/rmlx-kv-quant/src/kvcache/sdpa.rs`), so **the 4-bit decode does not
fall back to CPU dequant on the production route**. Hard rule 10 is satisfied at
both widths today.

`exit_prefill` is symmetric too: its `Iso3Sym`/`Iso4Sym` and
`IsoKOnly3`/`IsoKOnly4` arms are the same CPU bulk `append` at both widths.

### The one live behaviour difference

It is on the **non-fused** route, which runs on a `Device::Cpu` drive, at
`q_seq > 1`, and whenever a fused arm's shape gate (`iso_flash_shape_ok`,
`b == 1`, `head_dim <= 512`) rejects and falls through to `update()`:

| Entry | 3-bit | 4-bit |
|---|---|---|
| `update_iso*` (V axis) | `vs.append_gpu(..)` then `vs.dequant_gpu(device)` on GPU, `vs.dequant_on(device)` on CPU | `iso4_gpu_append_into_v_blocks(..)` then `vs.dequant()` — a host scalar decode of the whole prefix |
| `update_iso*_sym` (K axis) | `ks.dequant_gpu(device)` on GPU | `ks.dequant()` |
| `update_iso_k_only_*` | `ks.dequant_gpu(device)` on GPU | `ks.dequant()` |

`QuantIsoV4` has no `dequant_gpu` and no `dequant_on`; `QuantIsoK4` has neither
either. That is the asymmetry the collapse removes, and it is a real behaviour
change — just not on the route the issue named.

### Three facts the issue's list does not carry

1. **`ISO4_GROUP_SIZE`, `ISO3_GROUP_SIZE`, `ISO_K3_GROUP_SIZE` and
   `ISO_K4_GROUP_SIZE` are all `ISO_QUAT_BLOCK_SIZE` (4).** The pair differs in
   **one** scalar, the bit width. A single `const BITS: u8` parameter is
   sufficient; no second parameter is needed. Same conclusion the rotor pilot
   reached, for the same reason.
2. **`QuantIsoV3` carries no `max_seq` field and `QuantIsoV4` does.** The V3
   omission is deliberate and documented: "a cached copy is a snapshot that goes
   stale the moment the window grows — the same trap `QuantRotorK{3,4}` used to
   carry". `QuantIsoK3` and `QuantIsoK4` both carry one and both call it inert.
   The constructors differ accordingly: `QuantIsoV3::new(init_shape)` against
   `QuantIsoV4::new(init_shape, max_seq)`. The collapse must resolve this, and
   the direction the tree already argues for is to drop the field — see §7.
3. **`crates/rmlx-kv-quant/src/kvcache/sdpa.rs` carries no iso width twins.**
   Its fourteen iso helpers are already width-generic: each reads the width off
   the live `KvStorage` variant. The collapse touches `update.rs` and
   `storage/`, not `sdpa.rs`.

## 2. The GPU decision, stated as a constraint

The issue leaves one decision open: does the 4-bit iso path get the GPU mirror?
§1 splits that question in two, and the two halves have different answers.

**The GPU *decode* entry: yes, and it is forced.** Once `QuantIsoV<BITS>` and
`QuantIsoK<BITS>` are one type each, `dequant_gpu` and `dequant_on` are one body
each and both widths have them. Leaving the 4-bit width without them would mean
keeping two types, which is the thing being removed. This is the behaviour
change, and §"What may move" below states its bound.

**The GPU-resident *mirror*: it comes along, and it changes nothing.** The
mirror write in `append_gpu` is behind a `false` constant in production, so
generic-ing it hands the 4-bit width a path that writes no byte. That is not a
new dead path — it is the existing dead path, now one copy instead of
one-and-a-half. **Deleting the mirror outright is the better change and it is
not this issue's.** It is dead code behind a gate no production build can open,
`docs/KV_QUANT.md` describes it as a forward-compatibility hook for seedless
decode, and removing it would touch `lib.rs`, the gate's test guard and the
`quant_iso_v_tests.rs` cells that exercise it. Recorded here as a follow-up, not
done here.

### (a) What cannot move

* Every **3-bit** cell, byte for byte: packed store bytes after the bulk
  append, after the decode steps and after `truncate_to`; the K/V rows the
  attention receives; `resident_bytes()`.
* Every **K-only** cell at both widths, on the same five columns. `k_iso3` and
  `k_iso4` write one store through one entry and the collapse changes neither
  the encode nor the CPU decode.
* The six `KvQuant` spellings (`Iso3`, `Iso4`, `Iso3Sym`, `Iso4Sym`,
  `IsoKOnly3`, `IsoKOnly4`), their `Display` text (`iso3`, `iso4`, `iso3_sym`,
  `iso4_sym`, `k_iso3`, `k_iso4`), their `FromStr` spellings, the six
  `KvStorage` variants, the CLI names, the layout tags, the four `.metal` kernel
  entry points and `scripts/gpu_validation_census.txt`.
* `make check-kv-codec-disposition`'s verdict per codec. No disposition
  predicate arm changes: the collapse moves no codec between the bf16-mirror
  class and the store-reading class.

### (b) What may move, and how far

The **4-bit cells only**, and only on the route the new `dequant_gpu` entry
serves — `Device::Gpu`, non-fused. Nothing on the CPU route may move, which is
why the CPU pins in §3 are held at both widths.

The bound is documented and the 3-bit path is already held to it. From
`docs/KV_QUANT.md`, on the 3-bit GPU dequant:

> Parity verified by `iso_v3_dequant_gpu_matches_dequant_cpu` and
> `iso_k3_dequant_gpu_matches_dequant_cpu` in
> `crates/rmlx-kv-quant/src/isoquant_msl_tests.rs` (`#[ignore]`-gated).
> Observed `max|cpu-gpu| ≤ 2.4e-7` on the LCG fixture (a few f32 ULPs from
> different summation order between CPU `iso_decode_fast` and the MSL kernel —
> not a real codec divergence). The parity test gates at 5e-3 (codebook
> tolerance) and additionally enforces a strict ≤ 1e-6 bound.

The test itself asserts both: a per-element `diff <= 5e-3` and, after the loop,
`max_abs <= 1e-6`. **The 4-bit path is held to the same pair.** What exists for
it today is weaker — `iso_v4_msl_matches_cpu_within_eps` in
`crates/rmlx-kv-quant/src/isoquant_msl_v4_tests.rs` gates at 5e-3 only, on the
kernel rather than on `dequant_gpu`, because there is no 4-bit `dequant_gpu` to
compare against. The code chunk owes the two 4-bit siblings of the 3-bit tests,
at the same two bounds. §4 names them.

The codec's own quality floors are unchanged by the collapse and are restated
here only so the two kinds of bound are not confused: `iso3_cosine_gate` gates
V-side cosine at mean 0.994 / min 0.993, `iso4_cosine_gate` at mean 0.998638 /
min 0.998092, and the K-side floors are 0.97 for `iso_k_3` and 0.99 for
`iso_k_4` (`docs/KV_QUANT.md`). Those are CPU-codec gates; they say nothing
about CPU-versus-GPU agreement, which is what §(b) is about.

## 3. The oracle

`crates/rmlx-kv-quant/src/kvcache/iso_store_bytes_tests.rs`, six `#[test]` fns,
12 cells (6 spellings x 2 shapes), 5 pinned quantities per cell:
`store_after_chunk`, `store_after_decode`, `store_after_truncate`, `rows`,
`resident_bytes`.

The module doc holds the full statement of what cannot move and what the pin
cannot see; it is not duplicated here. Four points that are specific to the iso
family:

* **Shapes.** The same `(1, 128)` and `(4, 96)` the rotor pin uses, so the two
  files compare like with like. Both satisfy the iso constraint
  `head_dim % 4 == 0` and the q8_0 K side's `B * kv_h * seq * head_dim % 128 == 0`
  at both the 24-token chunk and the one-token step.
* **No ragged group exists on this family.** The rotor pin's shape A is chosen
  for a padded last multivector group. Iso rejects any `head_dim` that is not a
  multiple of 4 outright (`IsoQuantError::HeadDimNotMultipleOf4`), so there is
  no third shape to add for that case.
* **`max_seq` is deliberately not a digest field.** It is inert on all three
  stores that carry one and absent on the fourth; pinning it would force a
  re-baseline whichever way §7's removal goes.
* **The 4-bit rows are the reference.** They are the CPU side the future 4-bit
  `dequant_gpu` is compared against. They are not allowed to move: the new
  entry is on `Device::Gpu` and every cell here drives `Device::Cpu`.

### The census

`iso_spellings()` filters `ALL_KV_QUANTS` by `Display` text containing `iso`.
No hand-written variant list and no `matches!` arm: a list names only the
variants that existed when it was written, so a seventh iso spelling would never
enter the population and the census would stay green with nothing pinned. No
other codec's `Display` text contains `iso`, so the text is the membership test.
`ISO_SPELLING_COUNT` (6) is the anchor beside it, and `PINS.len()` is asserted
to be exactly `2 x` the population.

Measured against `ALL_KV_QUANTS` on this tree: the six spellings the issue names
are exactly the six the list holds. **No iso spelling is missing from the list,
and the list names none the issue omits.** That is where this family differs
from the rotor pilot, whose census found two omitted pairs.

### What the CPU oracle cannot see, and the GPU test owed for each

| Unseen | GPU test owed | Census entry owed |
|---|---|---|
| 4-bit `dequant_gpu` against the CPU reference, V axis | `iso_v4_dequant_gpu_matches_dequant_cpu`, sibling of the 3-bit test in `isoquant_msl_v4_tests.rs`, both bounds (5e-3 per element, `max_abs <= 1e-6`) | yes — a new `#[ignore]` GPU test that loads no checkpoint still derives its `scripts/gpu_validation_census.txt` entry from the kernels it dispatches |
| 4-bit `dequant_gpu` against the CPU reference, K axis | `iso_k4_dequant_gpu_matches_dequant_cpu`, same file, same bounds | yes |
| `gpu_append` / `gpu_packed_view` / `reconcile_ring` at both widths | already covered: `kvcache::iso_flash_dispatch_tests` and `kvcache::resident_ring_tests` | no new entry; re-derive if a cell gains a load |
| The ring-readback branch of `synced_iso_v_blocks` | `kvcache::iso_flash_dispatch_tests` | no |
| `QuantIsoV::append_gpu`'s mirror write | none. The gate is a `false` constant in production; the CPU test `the_gpu_resident_iso_mirror_is_off_unless_a_test_forces_it` is what says so, and it dispatches nothing | no |
| The `exit_prefill` bulk-encode arms | the served capture in §6 | no |

**No GPU test is written in this chunk.** Each of the two owed ones needs a
`scripts/gpu_validation_census.txt` entry derived in the same change and a
`make gpu-test HALF=codec` run to bless it, which is the integration window's
work, not chunk 0's.

## 4. Mutations

The oracle is only as good as what it can turn red. Every mutation below was
applied to the tree, run, and reverted from a snapshot whose sha256 of the
**working** file was compared before and after — `git checkout --` is not used,
it would revert uncommitted work. Each row names the assertion that caught it,
not just that something failed.

<!-- MUTATION-TABLE -->

## 5. Duplication figure

`scripts/debt_report.sh --matched-lines` carries no iso population today. Its
four populations are `drivers`, `impls`, `rotor-storage`, `rotor-updates` and
`ssd-hydrate`; none of them reaches a `quant_iso_*` file or an `update_iso*` fn.

The figures below are a **hand-run** of that module's own `normalize()` and
`matched_lines()` over two populations defined exactly as the rotor ones are —
a glob plus the digit-stripped `width_pair_key`, never a file or fn list. No
second matcher was written: the script was imported and its functions called.

```
iso storage twins (crates/rmlx-kv-quant/src/storage): 651 matched lines over 2984 body lines (4 item(s), 2 pair(s))
    quant_iso_k <-> quant_iso_k4: 338 (86.0 % of the shorter body)
    quant_iso_v <-> quant_iso_v4: 313 (76.9 % of the shorter body)
iso update twins (crates/rmlx-kv-quant/src/kvcache/update.rs): 162 matched lines over 532 body lines (6 item(s), 3 pair(s))
    update_iso3 <-> update_iso4: 69 (86.2 % of the shorter body)
    update_iso_k_only_3 <-> update_iso_k_only_4: 38 (95.0 % of the shorter body)
    update_iso3_sym <-> update_iso4_sym: 55 (76.4 % of the shorter body)
```

### The population design item the code chunk owes

Two entries in `MATCHED_LINES_POPULATIONS`, beside the rotor ones and not in
place of them, each a glob plus a name rule so one command measures the tree
that carries the width twins and the tree that has collapsed them:

* **`iso-storage`** — the non-test `quant_iso_*.rs` files under
  `crates/rmlx-kv-quant/src/storage`, paired inside a group sharing the filename
  stem with every digit run removed (`quant_iso_v` and `quant_iso_v4` ->
  `quant_iso_v`). Same collector shape as `rotor_storage_items`, which differs
  only in its glob — so the two are one collector and a glob parameter, not two
  fns.
* **`iso-updates`** — the `update_iso*` fns of
  `crates/rmlx-kv-quant/src/kvcache/update.rs`, paired by the same
  digit-stripped-name rule. Same collector shape as `rotor_update_items`, which
  differs only in its fn-name prefix.

The two rotor collectors are each a glob-or-prefix plus `width_pair_key`, and
the iso ones are the same collector at a different constant. Adding them as two
more copies would plant the twin the campaign exists to remove: the code chunk
parameterises the existing collectors and registers four entries over them.

`scripts/debt_report_selftest.sh` owes matching cases: a planted figure per
population, a measured `0` with the population still found once a width twin is
deleted, and `unavailable` (reason **and** exit code) for a missing root and for
a root that is there and empty. The rotor cases are the template.

**What an `iso-updates` population cannot see.** Its glob is the `update_iso*`
fns of one file, so the eight further iso width-twin pairs in that same file are
outside it — `iso{3,4}_gpu_append_into_{k,v}_blocks`, `iso{3,4}_k_only_gpu_append`,
`iso{3,4}_sym_gpu_append`, `iso{3,4}_sync_ring`, `iso{3,4}_v_sync_ring`,
`push_iso{3,4}_k_block` and `drop_blocks_when_ring_live_iso_{k,v}{3,4}`. A `0`
from this population says no two `update_iso*` entries differ only in a width
digit; it does not say there is no duplication left in the file. The rotor
population has the same boundary and the rotor doc records it.

## 6. Real-model rows — deferred

Not run in this chunk, and not run by the code chunk on its own initiative. They
belong to the integration run at the end of the integration branch, and the
decode-TPS row is owner-gated.

**The "before" arm is built from the commit before the collapse.** The
implementing chunk records that SHA in its PR body; it is not named here,
because this doc is written before the collapse commit exists.

What the run must show:

| # | Row | Gate |
|---|---|---|
| 1 | Temp-0 served digest at `--max-tokens 200`, six spellings x {`gemma-4-e2b`, Ternary-Bonsai-8B} x {4k, 32k} — 24 cells, each exactly 200 ids | the **six 3-bit-and-K-only cells' digests** byte-identical before/after. `iso3`, `iso3_sym`, `k_iso3` and `k_iso4` must not move at all. `iso4` and `iso4_sym` may move **only** if the change is the new GPU dequant entry taking over, and then only within §2(b)'s bound |
| 2 | `kv_cache_bytes` per cell | identical in all 24 cells, both arms. The collapse changes no store layout |
| 3 | A `none` control in the same capture | present and unchanged, so the capture is shown to separate codecs rather than reporting one stream 24 times |
| 4 | Positive control | `iso3_sym` and `none` carry different digests **and** different `kv_cache_bytes` at every (model, context) pair. Without it a table of identical rows cannot be told from a table of nothing |
| 5 | `exit_code` and `n_ids` | `0` and exactly `200` in both arms, all 24 cells. A row missing either is a stop, not a difference |
| 6 | Binary digest | must **differ** between the arms. A run whose binary digest did not change did not test the change. Reported beside the diff, never folded into it |
| 7 | Decode TPS, 3-bit cells | within ±1 % of the recorded anchor — **owner-gated**, requested as one batched ask with the cell list it covers |
| 8 | Decode TPS, 4-bit cells | reported before/after; expected to improve only on the non-fused route, which a served request does not take, so **no change is the expected result** — §1 |

Row 8 deserves the caveat spelled out. The issue expects the 4-bit cells to get
faster. They will not, on a served request: the production decode route for
`iso4_sym` and `k_iso4` is already the fused GPU arm, and the entry the collapse
adds serves the fall-through route only. A measured TPS gain would mean the
fused arm was rejecting more often than anyone thought, which is a finding, not
a win.

Per-cell raw logs carry the absolute model-snapshot path and must never reach a
commit message, a PR body, an issue or any other public surface. Only a
commands file written with the model root elided may be quoted.

## 7. Removals the code chunk owes

By file and by name.

* `storage/quant_iso_v4.rs` — the whole file; its `QuantIsoV4` becomes
  `pub type QuantIsoV4 = QuantIsoV<4>`.
* `storage/quant_iso_k4.rs` — likewise for `QuantIsoK4`.
* `storage/quant_iso_v.rs` and `quant_iso_k.rs` — kept and made generic, not
  deleted. `IsoBlocks`, its `BlockRows` impl, `synced_iso_v_blocks`,
  `iso_n_groups_for` and `iso_row_words` are already width-agnostic and move
  as-is.
* `QuantIsoV4::max_seq` — the field, and the second parameter of
  `QuantIsoV4::new`. `QuantIsoV3` deliberately carries neither and its own
  comment gives the reason. Resolving the asymmetry the other way would
  re-introduce the stale-window trap that comment names. The `KvStorage`
  variants keep their `max_seq`; that is where the live window is read from.
* `KvCache::update_iso4`, `update_iso4_sym`, `update_iso_k_only_4` in
  `kvcache/update.rs` — three bodies, over three shared bodies generic over the
  store's `BITS`.
* The three entries above them — `update_iso3`, `update_iso3_sym`,
  `update_iso_k_only_3` — become three, each resolving the width from the
  `KvStorage` variant it was dispatched on, the shape `update_rotor_v` and its
  siblings already have.
* The eight further width-twin pairs in the same file, which cannot stay
  per-width once the store is one type — a shared body could not call them:
  `iso{3,4}_gpu_append_into_k_blocks`, `iso{3,4}_gpu_append_into_v_blocks`,
  `iso{3,4}_k_only_gpu_append`, `iso{3,4}_sym_gpu_append`, `iso{3,4}_sync_ring`,
  `iso{3,4}_v_sync_ring`, `push_iso{3,4}_k_block`,
  `drop_blocks_when_ring_live_iso_k{3,4}` and
  `drop_blocks_when_ring_live_iso_v{3,4}`. Sixteen fns become eight. With them
  goes `iso4_v_gpu_append_for_test`, whose only job is to name the private
  4-bit appender for a test.
* `LEGACY_ISO4_V_FEED` — a `const RingFeed` with one width's name on it. The
  unified V appender takes the feed its caller states.
* The twin halves of the storage test files: `quant_iso_v4_tests.rs` and
  `quant_iso_k4_tests.rs` fold into `quant_iso_v_tests.rs` /
  `quant_iso_k_tests.rs`. What is deleted is the duplicated **body**, not the
  coverage: each case becomes one generic body with the width as a parameter
  plus one `#[test]` per width, so the cell count does not fall.
* The stale comment in `QuantIsoV3::append_gpu` that reads "in test mode it uses
  OnceLock latching on first read". The `OnceLock` was removed; `lib.rs` says so
  three lines from the gate. The unified body does not carry it forward.
* The `QuantIsoV4` doc comment "CPU-only. The existing MSL kernel is hard-coded
  for `bits=3`; an iso4 MSL kernel variant is deferred." Both halves are false
  on this tree: the store carries a GPU ring, and
  `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs` dispatches an iso4 kernel pair.
  The same sentence is in the `KvStorage::IsoV4` variant doc and in
  `KvStorage::IsoSym3`/`IsoSym4` ("CPU-only"), and goes with them.

### Kept, and why

* **The four `.metal` kernels.** The issue puts them out of scope and the
  measurement in §1 supports leaving them: after the rename the dequant pair is
  **identical** in code (its 2 differing lines are comments) and the quantize
  pair differs in **one** code line, the codebook-bounds loop count `7u` against
  `15u`, which is `(1 << bits) - 1`. They are twins, and collapsing them means
  collapsing four `.metal` entry points, their two Rust dispatchers
  (`isoquant_msl.rs` / `isoquant_msl_v4.rs`), the injected `ISO3_BOUNDS[7]` /
  `ISO4_BOUNDS[15]` constants and their two captured header snapshots under
  `src/metal/probes/`. That is a separate change with its own
  `make check-metal-compiles` and census work. **The kernel names, the four
  files and the two dispatchers stay.**
* **The GPU-resident mirror.** Dead behind a `false` constant, and deleting it
  is the better change — but it is not this one. §2 states the argument.
* **No `KvQuant` variant, no `KvStorage` variant, no CLI spelling, no layout
  tag, no census entry.** Retirement is a separate session. Every spelling that
  exists before exists after, with the same bytes and the same tokens.
* **No `check-*` Make target.** None keys on the iso storage file names;
  `check-kv-codec-disposition` keys on `ALL_KV_QUANTS` plus the disposition
  predicates, which are untouched.
* **`scripts/lib/debt_report.py`'s existing populations.** The two iso ones are
  added beside them, not in place of them.
