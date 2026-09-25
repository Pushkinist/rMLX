# Iso storage and update twins

The 3-bit and 4-bit iso storage types and their `update_*` bodies were the
second width-twin pair in `rmlx-kv-quant`, after the rotor pilot. They are
collapsed: the stores are `QuantIsoV<BITS>` and `QuantIsoK<BITS>` with the
width-named spellings as type aliases, and the six `update_iso*` entries are
three, each resolving the width from the `KvStorage` variant it was
dispatched on.

This doc is the record of that collapse: the premise it was decided on, the
bound the one behaviour change is held to, the oracle, the mutation run
against the collapsed tree, and what was removed.

**The method is not restated here.** [`docs/KV_ROTOR_TWINS.md`](KV_ROTOR_TWINS.md)
holds it: how a premise is re-measured against the tree, why a served digest is
not the oracle, what a store-bytes pin is, how a mutation run is conducted, and
how the duplication figure is produced. This doc holds only what the iso family
does differently — the drift, the GPU decision the collapse forces, the fidelity
bound, the iso mutation list, and the removals the collapse made.

**Why a second doc rather than a widened one.** The method is about a tenth of
the rotor doc's text; the premise, the mutation list and the removals are the
other nine tenths and are per-family. A merged "width twins" doc would carry two
premises, two mutation lists and two removal lists under one shared method, and
a reader looking for the iso removals would page past the rotor ones. The rotor
doc is already 41 KB against the 200 KB advisory ceiling
(`make debt-report`), so splitting also keeps both files inside it.

The oracle itself is
`crates/rmlx-kv-quant/src/kvcache/iso_store_bytes_tests.rs`.

## 1. Premise, re-measured against the tree — the pre-collapse baseline

**Everything in this section describes the tree before the collapse.** It is
kept because it is the evidence the decision was taken on, and because a
"removed N duplicated lines" claim with no measured baseline is not a claim.

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

Measured after the collapse, by the same tool and the same rule (§5): both
storage pairs and all three `update_iso*` pairs are gone, and the two
populations report a measured `0` with the population still found.

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
  `crate::gpu_resident_iso_enabled()`, whose production body returns the
  constant `GPU_RESIDENT_ISO_PRODUCTION` — `false`
  (`crates/rmlx-kv-quant/src/lib.rs`). The mirror writes no byte in production
  at either width, and §4's M14 is the mutation that holds that claim. `docs/KV_QUANT.md` records the bench
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
(`crates/rmlx-kv-quant/src/kvcache/sdpa.rs`), so on any shape the arms accept,
**the 4-bit decode does not fall back to CPU dequant**.

**That conclusion is conditional, and the condition bites.** `iso_flash_shape_ok`
requires `b == 1`, `kv_h > 0`, `head_dim % 4 == 0`, `head_dim <= 512` **and
`head_dim` a power of two** — the last for the kernel's tree reduction over
`head_dim` threads. A model with `head_dim` 80, 96 or 160 rejects the arm on
**every** decode step, falls through to `update()`, and there the 4-bit width
runs `ks.dequant()` / `vs.dequant()`: a host scalar decode of the whole prefix
per token, with the GPU idle. That is the hard-rule-10 defect, exactly, and it
exists today on the 4-bit width and not on the 3-bit one.

So: hard rule 10 is satisfied at both widths **on the shapes the gate accepts**;
on any other shape the 4-bit width CPU-decodes the prefix every step today, and
the collapse is what fixes it. The two proof models are both gate-accepted —
Ternary-Bonsai-8B is `head_dim = 128` and `gemma-4-e2b` is `head_dim = 256`
(`docs/KV_QUANT.md` § `iso_flash_decode`) — so **no served 4-bit cell in §6 is
expected to move**, and a cell that does is a finding, not the intended change.

`exit_prefill` is symmetric too: its `Iso3Sym`/`Iso4Sym` and
`IsoKOnly3`/`IsoKOnly4` arms are the same CPU bulk `append` at both widths.

### The one live behaviour difference — what the collapse removed

It was on the **non-fused** route, which runs on a `Device::Gpu` drive at
`q_seq > 1`, and whenever `iso_flash_shape_ok` rejects — a batched cache, or a
`head_dim` that is not a power of two — and the step falls through to
`update()`. Before the collapse:

| Entry | 3-bit | 4-bit |
|---|---|---|
| `update_iso*` (V axis) | `vs.append_gpu(..)` then `vs.dequant_gpu(device)` on GPU, `vs.dequant_on(device)` on CPU | `iso4_gpu_append_into_v_blocks(..)` then `vs.dequant()` — a host scalar decode of the whole prefix |
| `update_iso*_sym` (K axis) | `ks.dequant_gpu(device)` on GPU | `ks.dequant()` |
| `update_iso_k_only_*` | `ks.dequant_gpu(device)` on GPU | `ks.dequant()` |

`QuantIsoV4` had no `dequant_gpu` and no `dequant_on`; `QuantIsoK4` had
neither either. That is the asymmetry the collapse removed, and it is a real
behaviour change — just not on the route the issue named. Both widths now
reach the same `dequant_gpu` / `dequant_on` body, and both reach their own
kernel through `crate::isoquant_msl_dispatch`, which is the one place `bits`
selects a kernel module.

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
   `QuantIsoV4::new(init_shape, max_seq)`. The collapse resolved it by dropping
   the field, which is the direction the V3 comment already argued for; the
   unified constructor is `QuantIsoV::<BITS>::new(init_shape)`.
3. **`crates/rmlx-kv-quant/src/kvcache/sdpa.rs` carries no iso width twins.**
   Its fourteen iso helpers are already width-generic: each reads the width off
   the live `KvStorage` variant. The collapse touched `update.rs` and
   `storage/`, plus two lines of `sdpa.rs`: its `iso_k_gpu_append` and
   `iso_sym_gpu_append` wrappers existed only to pick between the two
   width-named entries, so they went with those entries and their one caller
   each now calls the update dispatcher directly.

## 2. The GPU decision, as taken

The issue left one decision open: does the 4-bit iso path get the GPU mirror?
§1 splits that question in two, and the two halves got different answers.

**The GPU *decode* entry: yes, and it was forced.** With `QuantIsoV<BITS>` and
`QuantIsoK<BITS>` one type each, `dequant_gpu` and `dequant_on` are one body
each and both widths have them. Leaving the 4-bit width without them would have
meant keeping two types, which is the thing being removed. This is the
behaviour change, and §"(b) What may move, and how far" below states its bound.

**The GPU-resident *mirror*: it came along, and it changes nothing.** The
mirror write in `append_gpu` is behind a `false` constant in production, so
generic-ing it handed the 4-bit width a path that writes no byte. That is not a
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
  predicate arm changed: the collapse moved no codec between the bf16-mirror
  class and the store-reading class. Measured after: 28 codecs classified, 17
  inert — the same verdict as before.

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
`max_abs <= 1e-6`. **The 4-bit path is held to the same pair**, by
`iso_v4_dequant_gpu_matches_dequant_cpu` and
`iso_k4_dequant_gpu_matches_dequant_cpu` in
`crates/rmlx-kv-quant/src/isoquant_msl_v4_tests.rs`, both `#[ignore]`-gated.
They share one `assert_dequant_parity` body, because writing the two bounds out
twice would plant the twin this change removes. What existed before is weaker
and stays beside them: `iso_v4_msl_matches_cpu_within_eps` gates at 5e-3 only
and compares the kernels, not the store entries a decode step calls.

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
* **The 4-bit rows are the reference.** They are the CPU side the 4-bit
  `dequant_gpu` is compared against, and they did not move — but that is the
  pin's whole contribution on the 4-bit GPU question. Every cell here drives
  `Device::Cpu`, and the module doc lists `append_gpu` among the things a CPU
  drive cannot reach, so **nothing below is evidence that the 4-bit store holds
  the same bytes under `append_gpu`.** What says so is a read of the two
  appenders, not a measurement: `QuantIsoV::append_gpu` and the deleted
  `RingFeed::Skip` call on `iso_gpu_append_into_v_blocks` — the appender itself
  is live, on the fused path, at `RingFeed::MaintainRingOnly` — build their CPU block
  from the same `iso_gpu_outputs_to_cpu` call at the same
  `(n_tokens_total, n_groups, bits)`; `reconcile_ring(device, Drop)` is
  `reconcile_ring(device, Keep)` followed by `gpu.clear()`, which is what the
  `Skip` feed did through `iso_v_sync_ring`; and with
  `GPU_RESIDENT_ISO_PRODUCTION` `false` the mirror write that follows stores no
  byte. The gate over that route is `make gpu-test`, not this file.

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

### What the CPU oracle cannot see, and the GPU test that covers each

| Unseen | GPU test that covers it | Census entry |
|---|---|---|
| 4-bit `dequant_gpu` against the CPU reference, V axis | `iso_v4_dequant_gpu_matches_dequant_cpu` in `isoquant_msl_v4_tests.rs`, both bounds (5e-3 per element, `max_abs <= 1e-6`) — **written** | derived from a run: the runner fails on an unpinned hit, so a clean pass is the evidence and no entry is the answer |
| 4-bit `dequant_gpu` against the CPU reference, K axis | `iso_k4_dequant_gpu_matches_dequant_cpu`, same file, same bounds — **written** | same |
| `gpu_append` / `gpu_packed_view` / `reconcile_ring` at both widths | already covered: `kvcache::iso_flash_dispatch_tests` and `kvcache::resident_ring_tests` | no new entry; re-derive if a cell gains a load |
| The ring-readback branch of `synced_iso_v_blocks` | `kvcache::iso_flash_dispatch_tests` | no |
| `QuantIsoV::append_gpu`'s mirror write | none, and §4's M14 is why that is safe to say: the production value is the constant `GPU_RESIDENT_ISO_PRODUCTION`, `the_production_gpu_resident_iso_mirror_is_off` asserts on that constant, and flipping it turns the assertion red. The test dispatches nothing | no |
| The `exit_prefill` bulk-encode arms | the served capture in §6 | no |

Both tests are written. Their census disposition is derived from a run rather
than asserted: `scripts/gpu_validation_census.txt` records **accepted invalid
accesses**, so a test that produces no shader-validation hit carries no entry,
and that is the same disposition the 3-bit pair has.

## 4. Mutations

The oracle is only as good as what it can turn red. Every mutation below was
applied to the tree, run, and reverted from a snapshot whose sha256 of the
**working** file was compared before and after — `git checkout --` is not used,
it would revert uncommitted work. Each row names the assertion that caught it,
not just that something failed.

**The run below is against the collapsed tree.** Each row states its target in
the collapsed tree; where the collapse merged two bodies, the same edit now
reaches both widths and the row says which cell reports it. The recall is
unchanged: 13 of 14 red, and M11 is still the one it cannot catch.

| # | Edit | Caught by |
|---|---|---|
| M1 | `QuantIsoV::append` — encode at the literal `3` instead of `BITS`, i.e. the width the body reads ignoring the width it was instantiated at | 4 of the 6 tests: the pin, the geometry, the twins control and the reproducibility drive. **Red, not a compile error**: `BITS` and a literal are both `u8`, so nothing in the type system separates them |
| M2 | `QuantIsoV::append` — drop the last code word (`codes.pop()`) | the same 4 |
| M3 | `quant_iso_k.rs` — `ISO_QUAT_BLOCK_SIZE` 4 -> 8, moving the quaternion group boundary. One constant now, where the pre-collapse run edited `ISO3_GROUP_SIZE` | the pin (`iso3 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`) **and** the geometry (scale count 384 against 768) |
| M4 | `isoquant.rs` — `FIXED_QUAT` replaced by `[0.5, 0.5, 0.5, 0.5]`, the codec's one rotation constant | the pin **only** (`iso3 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`); geometry stays green, because the code plane is the same length |
| M5 | `iso_sym_update` — the K store appends `v_f32`, so the symmetric entry writes V's data on both axes | the pin **only**. It now reports at `iso3_sym` rather than `iso4_sym`, because one body serves both widths and the 3-bit cell is driven first |
| M6 | `KvCache::update_iso_k_only` — `new_v` handed to `iso_k_only_k_side` instead of `new_k`, so the K-only entry stores the V stream | the pin **only** (`k_iso3 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`) |
| M7 | `QuantIsoV::truncate_to` — `(n - 1).max(0)` | the pin's **truncate** column **only** (`iso3 @ kv_h=1 head_dim=128: packed store bytes after truncate_to moved`) |
| M8 | `QuantIsoV::dequant_on` — drop `transpose_chunked_seq_heads`; the store is untouched, only the rows the attention receives move | the pin's **rows** column only, at **shape B only** (`iso3 @ kv_h=4 head_dim=96: the K/V rows attention receives moved`) |
| M9 | `quant.rs` — `KvQuant::Iso4` removed from `ALL_KV_QUANTS` | the census (`iso spelling census moved … ["iso3", "iso3_sym", "iso4_sym", "k_iso3", "k_iso4"]`, left 5 right 6) |
| M10 | `quant.rs` — `KvQuant::Iso3` listed twice in `ALL_KV_QUANTS`, standing in for a seventh iso spelling | the census (left 7 right 6). It exercises the count anchor only: a genuinely new variant would also miss `pin_for` and turn the pin test red, which this stand-in cannot show without wiring a variant through every exhaustive match |
| M11 | `QuantIsoV::gpu_append` — `n_groups + 1` into `append_encoded` | **uncaught** — 6 passed, exit 0 |
| M12 | `QuantIsoV::byte_size` — drop the CPU-blocks term | the pin's **resident_bytes** column only (`iso3 @ kv_h=1 head_dim=128: resident_bytes moved`, left 3564 right 22248) |
| M13 | `QuantIsoK::append` — coalesce a one-token block into its predecessor after the push, so the payload is identical and only the block boundaries move | the pin's **store_after_decode** column only (`iso3_sym @ kv_h=1 head_dim=128: packed store bytes after 3 decode steps moved`). The chunk column and the geometry stay green: the first append has no predecessor to merge into, and the summed plane lengths do not change |
| M14 | `lib.rs` — `GPU_RESIDENT_ISO_PRODUCTION` flipped to `true` | `the_production_gpu_resident_iso_mirror_is_off` (`the GPU-resident iso V mirror is on in production`). Its control is below |

**The other half of "width mis-resolved" is a compile error.** `QuantIsoV<5>`
and `QuantIsoK<5>` are rejected by a named associated const —
`WIDTH_IS_A_SHIPPED_ONE: ()`, an `assert!` over `BITS` — in each
`impl<const BITS: u8>` block.

**A guard is only reached where something reads it**, and that is the whole
design of this one. A sibling const does not force another, so `Self::NAME`
reading the width table on its own would leave a store born through a path that
never mentions the guard unprotected. So `NAME` opens with
`let () = Self::WIDTH_IS_A_SHIPPED_ONE;`, both `Debug` impls take their string
from `NAME`, and **both** birth paths — `new` and `from_cpu_blocks`, the second
being the SSD-hydrate route — read the guard directly. Measured, each on its
own throwaway probe:

```
QuantIsoV::<5>::new              -> error[E0080]: evaluation panicked: the iso codec ships 3-bit and 4-bit only
QuantIsoV::<5>::from_cpu_blocks  -> error[E0080]: (same)
QuantIsoV::<5>::NAME             -> error[E0080]: (same)
```

It fires at **monomorphisation**, so `cargo build` and `cargo test` catch it and
`cargo check` / `cargo clippy` do not — the error is post-monomorphisation and a
check-only pass never instantiates the type. Before the assertion the extra
width compiled, was refused only at runtime by the kernel dispatch, and printed
through the `Debug` impl's `_` arm as the 4-bit store.

M4, M5, M6, M7, M8, M12 and M13 are each caught by exactly one assertion, and
M7, M8, M12 and M13 each by exactly one **column** — truncate, rows,
resident_bytes and store_after_decode respectively. With M1 and M2 on the chunk
column, all five columns are load-bearing rather than redundant. M8 fires at one
shape only, which is why both shapes are.

**What the collapse cost the mutation set, and what it did not.** Five rows
(M5, M6, M8, M12, M13) used to name a 4-bit-only body and now name a shared
one, so each reports at the 3-bit cell instead of the 4-bit one. The assertion
that fires, the column it fires on and the count of cells that fail are
otherwise unchanged. What is genuinely lost is the *independence* of the two
widths under those five edits: before, an edit to the 4-bit body could not
move a 3-bit cell, and the pin said so. After, one body serves both, and the
pin's answer to "did the 4-bit width move on its own?" is a question the tree
no longer has a place to ask. That is the intended consequence of having one
body, not a gap in the oracle.

### M14's control, and why the assertion reads a constant

`the_production_gpu_resident_iso_mirror_is_off` used to call
`crate::gpu_resident_iso_enabled()`. **It could not fail.** Under `cargo test`
that name resolves to the `#[cfg(test)]` body, an override flag a sibling test
sets; the production body is not compiled into the test binary at all. Measured
on the pre-fix tree: flipping the `cfg(not(test))` body from `false` to `true`
left all six cells green.

The value is now a plain `pub(crate) const GPU_RESIDENT_ISO_PRODUCTION`, read by
the production body and asserted on by the test. M14 is the recall case and is
red. Two further properties come with the shape: the test reads a compile-time
constant rather than a process-global `AtomicBool`, so its outcome no longer
depends on whether
`quant_iso_v_tests::the_mirror_gate_is_scoped_to_its_guard_and_restores_the_previous_state`
happened to be inside its guard at the time — an ordering hazard neither test
held a lock against — and it needs no lock to be free of it.

### The mutation I could not catch

**M11.** `QuantIsoV::gpu_append` is handed one more quaternion group than the
head dimension has, and every one of the six tests passes.

The reason is structural, not an oversight in the assertions: `gpu_append`
writes the GPU ring, and a `Device::Cpu` drive never calls it. Neither does
anything else this file reaches — `CPU append` clears the ring rather than
feeding it. The same blind spot covers `gpu_packed_view`, `reconcile_ring`, the
ring-readback branch of `synced_iso_v_blocks`, `from_cpu_blocks` and
`try_deep_clone`. It is the same class the rotor pilot recorded and measured:
there, re-applying the equivalent edit and running `make gpu-test
CRATE=rmlx-kv-quant` turned it red, on two named tests. That measurement is
**not repeated here** — it needs an idle GPU and belongs to the integration
window.

Two further things this run cannot see, both stated so the reviewer does not
read the 13-of-14 as wider than it is. The pin drives `Device::Cpu`, so the
new 4-bit `dequant_gpu` entry — the collapse's one behaviour change — is
outside every row above; the two GPU parity tests in §3 are what covers it.
And no row mutates the width dispatch in `isoquant_msl_dispatch`, because the
pin never dispatches a kernel; a width sent to the wrong kernel module is a
`make gpu-test` finding, not a `make ci` one.

So the honest statement of this oracle's power: it covers the CPU encode, the
CPU decode, the block bookkeeping, `truncate_to` and residency at both widths
and both shapes, and it covers **nothing on the ring**. `make gpu-test
HALF=codec` is the only gate over the ring, and the collapse's reviewer must
read its result rather than this file's.


## 5. Duplication figure

`scripts/debt_report.sh --matched-lines` now carries `iso-storage` and
`iso-updates`, beside the rotor pair and not in place of it. One command
measures both tree shapes, which is what makes the before and after
comparable:

```
$ bash scripts/debt_report.sh --matched-lines iso-storage     # before
iso storage twins (crates/rmlx-kv-quant/src/storage): 651 matched lines over 2984 body lines (4 item(s), 2 pair(s))
$ bash scripts/debt_report.sh --matched-lines iso-updates     # before
iso update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs): 162 matched lines over 532 body lines (6 item(s), 3 pair(s))
  (the "before" arm is read with the rule this chunk started from — the prefix
   `update_iso`, paired on width — and the "after" arm with the rule below: the
   codec token, every pair compared. The two are not one series. What the
   before arm says is that the width twins it could see are gone; what the
   after arm says is where the family's duplication now is, which the first
   rule could not have reported at all)

$ bash scripts/debt_report.sh --matched-lines iso-storage     # after
iso storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 2234 body lines (2 item(s), 0 pair(s))
$ bash scripts/debt_report.sh --matched-lines iso-updates     # after
iso update fns (crates/rmlx-kv-quant/src/kvcache/update*.rs): 875 matched lines over 625 body lines (21 item(s), 210 pair(s))
```

The label names the producer's root, which is a property of the producer and
not of the tree: it reads `crates/rmlx-kv-quant/src/kvcache/update*.rs` since
the update path was split by codec family, so both transcripts above are shown
under that name. The `875` is what the producer printed at this commit;
`matched_lines` has since begun measuring each pair both ways round and
reporting the larger, which moves the figure without moving a body. Re-run on
either tree the same population now reads:

```
iso update fns (crates/rmlx-kv-quant/src/kvcache/update*.rs): 878 matched lines over 625 body lines (21 item(s), 210 pair(s))
```

The item count, the pair count and the body-line total are the ones recorded
here. See [`KV_UPDATE_SPLIT.md`](KV_UPDATE_SPLIT.md) §3.

The "before" arm is the tool run with `--root` pointing at a worktree of the
commit before the collapse, so the same code produces both figures. A measured
`0` **with the population still found** is the answer that matters: a deleted
population would print `unavailable` and exit 1, and a population that quietly
resolved to nothing would be indistinguishable from a collapsed one if the
module let it print `0`.

The pre-collapse per-pair breakdown, for the record: `quant_iso_k` against
`quant_iso_k4` 338 lines (86.0 % of the shorter body), `quant_iso_v` against
`quant_iso_v4` 313 (76.9 %); `update_iso3` against `update_iso4` 69 (86.2 %),
`update_iso_k_only_3` against `_4` 38 (95.0 %), `update_iso3_sym` against
`update_iso4_sym` 55 (76.4 %).

### The population design

Two entries in `MATCHED_LINES_POPULATIONS`, beside the rotor ones and not in
place of them, each a glob plus a name rule:

* **`iso-storage`** — the non-test `quant_iso_*.rs` files under
  `crates/rmlx-kv-quant/src/storage`, paired inside a group sharing the filename
  stem with every digit run removed (`quant_iso_v` and `quant_iso_v4` ->
  `quant_iso_v`).
* **`iso-updates`** — every fn of the update files
  (`crates/rmlx-kv-quant/src/kvcache/update*.rs`: the dispatch file plus one
  file per codec family) whose name carries the codec token `iso`, **every
  pair compared**. Two departures from the rotor entry,
  each for a stated reason.

  A name **pattern**, not a prefix: the collapse split the family into entries
  (`update_iso_*`) and the bodies they enter (`iso_v_update`, `iso_sym_update`,
  `iso_k_only_k_side`), which share no prefix, so a prefix population would
  have reported a clean scan over the entries alone. The pattern is
  `(^|_)iso(\d|_|$)` — the token on a segment boundary, with the width digit
  admitted where it is glued to it (`update_iso3`), which is how the rule reads
  the pre-collapse tree as well as this one. The rotor entry keeps prefix
  behaviour by anchoring its own pattern (`^update_rotor`), so its figures are
  unchanged.

  **No `width_pair_key`**, which is the bigger departure. That key groups by
  the digit-stripped name, so it compares a 3-bit body with its 4-bit sibling
  and with nothing else. After the collapse no two iso fns in this file share a
  digit-stripped name, so a width-keyed population would report `0 pair(s)` and
  `0 matched lines` *whatever the file held* — a counter that cannot move. The
  live duplication in this family is between **same-width** bodies of different
  entries, which is exactly what a width key is blind to, so this population
  pairs every item with every other. The width-twin question for the iso family
  is `iso-storage`'s, and it keeps the key.

Three things it gets right, each for a stated reason.

1. **Parameterised at the registration site, not on the record.** `Population`
   already carries `collect` and `root`; the collector is a plain
   `Callable[[Path], list[FnInfo]]`. So `rotor_storage_items` generalised to
   `storage_file_items(root, *, glob)` and all four width-twin entries register
   `functools.partial(...)`; `rotor_update_items` generalised to
   `file_fn_items(root, *, file, prefix)` the same way. A `glob` or `prefix`
   **field** on `Population` would make `drivers`, `impls` and `ssd-hydrate`
   carry a field none of them reads.
2. **One root constant, not two.** `ROTOR_STORAGE_DIR` was already
   `crates/rmlx-kv-quant/src/storage`, which is the iso stores' root too, and
   `ROTOR_UPDATE_FILE` was already the one update file. Both now have two
   callers and named neither, so they are `KV_STORAGE_DIR` and
   `KV_UPDATE_FILE`.
3. **No CLI change was owed.** `--matched-lines`'s `choices` are
   `sorted(MATCHED_LINES_POPULATIONS)`, so registering the two entries is what
   added them to the flag.

Registering two more copies of the rotor collectors would plant the twin the
campaign exists to remove.

`scripts/debt_report_selftest.sh` carries the matching cases, 70 to 89: a
planted figure per population (four `quant_iso_*.rs` files at 23 matched lines
over 48, 2 pairs; six iso fns of the update file at 41 over 28, 15 pairs),
`iso-storage` measured at `0` with the population still found once a width twin
is deleted, and `unavailable` — reason **and** exit code — for a missing root
and for a root that is there and empty.

`iso-updates` is held in both directions a counter can move. Deleting the two
4-bit entries drops it from 41 over 6 items to 17 over 4 — a width collapse
moves it even though it is not keyed on width. Collapsing the planted
**same-width** twin (`iso_v_update` / `iso_sym_update`) drops it from 41 to 27
over 5 items — that is the case a `width_pair_key` population could not have,
because it would read the same figure with the duplication present or gone.

The iso fixtures sit in the same directory and the same file as the rotor ones,
so a widened glob or a widened pattern reads the wrong item count on one of the
two families; and a token pattern that did not admit a glued width digit would
drop `update_iso3` / `update_iso4` and read four items where six are planted.

**The same-width residual, and what the population now reports.** The axis the
collapse does not address is the one within a width: two entries of different
families, at one width, sharing a skeleton. It is the axis a `width_pair_key`
population is blind to by construction, which is why this one does not use it.

There was such a pair, and it was larger than the width twin this chunk
removed. Measured with the module's own `normalize()` and `matched_lines()`:

```
iso_v_update <-> iso_sym_update:  80 matched lines, shorter body 103 (ratio 0.777)
update_iso3  <-> update_iso4:     69 matched lines, shorter body  80 (ratio 0.862)
```

The shared mass was the **whole V axis**: lazy store creation, the
`append_gpu` / `append` split, the `iso_encode` trace, the
`dequant_gpu` / `dequant_on` split, the two decode traces, and
`f32_vec_to_array`. The two bodies differed in the trace message alone
(`"iso hot-path"` against `"iso hot-path (sym V)"`).

It is extracted: `iso_v_encode_decode<BITS>` is that block once, and the
message it differed on is now a `variant` field carrying the storage spelling
the caller resolved, which says more than the parenthesis did. Re-measured on
the same rule:

```
iso_v_update <-> iso_sym_update:  27 matched lines, shorter body 39 (ratio 0.692)
```

80 matched lines over two bodies of 113 and 103 became 27 over 45 and 39. The
pins did not move: `iso_store_bytes` 6/6 and `rotor_store_bytes` 5/5, no
assertion edited. What is left is the K axis of each body, which is genuinely
different — one drives `QuantK` (affine q8_0), the other `QuantIsoK<BITS>` —
plus the shared `array_to_f32_vec` preamble and the `Ok((k_full, v_full))`
tail.

**The pair that is left, and why it stays.** The extraction created the file's
largest iso pair, which the all-pairs population duly reports at the top of its
210:

```
iso_v_encode_decode <-> iso_k_only_k_side:  50 matched lines, bodies 76 and 75 (ratio 0.667)
```

Same lines on both sides: the lazy store creation, the let-else bind, four
trace phases, the GPU early return, `dequant_on`, `f32_vec_to_array`. **They
are not twins under the twin rule, and collapsing them would cost more than it
removes.** The rule names two items whose bodies differ only in a compile-time
constant or in a component's name. These differ in the *operation on the axis*:
the V body encodes through the store's own `QuantIsoV::append_gpu`, which
dispatches the encode kernel, reconciles and drops the ring and pushes the CPU
block; the K body encodes through the `kvcache` ring-aware appender
`iso_gpu_append_into_k_blocks` with an explicit `RingFeed`, which is a
different mechanism with different ring semantics and a different signature.
Their constructors differ too (`new(init_shape)` against
`new(init_shape, max_seq)`).

One body over both would need a store-side trait with an encode-GPU, an
encode-CPU, a constructor, a sequence length and two dequant entries — seven
methods, two implementors, introduced to share a trace-and-branch skeleton.
That is the single-use trait tower the simplicity rules forbid, and it would
hide the one thing a reader of either body needs to see: which appender the
axis goes through. So the pair stays, the figure is recorded here, and the
population that reports it is the one that would show the figure moving if the
two ever did converge onto a constant.

The rotor population is a width-twin detector and the rotor doc records that
boundary; this one is not, and records that instead.

## 6. Real-model rows — deferred

Not run in this chunk, and not run by the code chunk on its own initiative. They
belong to the integration run at the end of the integration branch, and the
decode-TPS row is owner-gated.

**The "before" arm is the commit before the collapse**, which the PR body
records by SHA.

What the run must show:

| # | Row | Gate |
|---|---|---|
| 1 | Temp-0 served digest at `--max-tokens 200`, six spellings x {`gemma-4-e2b`, Ternary-Bonsai-8B} x {4k, 32k} — 24 cells, each exactly 200 ids | **all 24 cells byte-identical before/after.** Both proof models are gate-accepted (§1), so no cell reaches the new entry and none may move. A moved 4-bit cell is admitted only after the capture shows *that cell* rejected the fused arm — the served log names the dispatch path and the kernel it fired — and then §2(b)'s bound applies to it. A moved cell with no such evidence is a stop, not a difference |
| 2 | `kv_cache_bytes` per cell | identical in all 24 cells, both arms. The collapse changes no store layout |
| 3 | A `none` control in the same capture | present and unchanged, so the capture is shown to separate codecs rather than reporting one stream 24 times |
| 4 | Positive control | `iso3_sym` and `none` carry different digests **and** different `kv_cache_bytes` at every (model, context) pair. Without it a table of identical rows cannot be told from a table of nothing |
| 5 | `exit_code` and `n_ids` | `0` and exactly `200` in both arms, all 24 cells. A row missing either is a stop, not a difference |
| 6 | Binary digest | must **differ** between the arms. A run whose binary digest did not change did not test the change. Reported beside the diff, never folded into it |
| 7 | Decode TPS, 3-bit cells | within ±1 % of the recorded anchor — **owner-gated**, requested as one batched ask with the cell list it covers |
| 8 | Decode TPS, 4-bit cells | reported before/after; **no change is the expected result on these two models** — §1 |

Row 8 deserves the caveat spelled out. The issue expects the 4-bit cells to get
faster. They will not, on these two models: both are gate-accepted, so the
production decode route for `iso4_sym` and `k_iso4` is already the fused GPU
arm, and the entry the collapse adds serves the fall-through route only. A
measured TPS gain on a gate-accepted model would mean the fused arm was
rejecting more often than anyone thought, which is a finding, not a win.

**Where the gain is real, and why it is not measured here.** On a model whose
`head_dim` is not a power of two the fused arm rejects every step and the 4-bit
width host-decodes the whole prefix, so the collapse should move that cell by a
large factor. No such model is in the proof set, and adding one is a scope
change the owner decides. It is named so the absent row is a decision rather
than an oversight.

Per-cell raw logs carry the absolute model-snapshot path and must never reach a
commit message, a PR body, an issue or any other public surface. Only a
commands file written with the model root elided may be quoted.

## 7. Removals, done

By file and by name. Every item below is deleted on the collapse commit; the
PR body lists them again with the net line count.

* `storage/quant_iso_v4.rs` — the whole file; `QuantIsoV4` is now
  `pub type QuantIsoV4 = QuantIsoV<4>`.
* `storage/quant_iso_k4.rs` — likewise for `QuantIsoK4`.
* `storage/quant_iso_v.rs` and `quant_iso_k.rs` — kept and made generic, not
  deleted. `IsoBlocks`, its `BlockRows` impl, `synced_iso_v_blocks`,
  `iso_n_groups_for` and `iso_row_words` are already width-agnostic and moved
  as-is. `ISO4_BITS`, `ISO4_GROUP_SIZE`, `ISO_K4_BITS` and `ISO_K4_GROUP_SIZE`
  moved with their types and keep their spellings, because `rmlx-kv-ssd`'s
  tests name them.
* `QuantIsoV4::max_seq` — the field, and the second parameter of
  `QuantIsoV4::new`. `QuantIsoV3` deliberately carried neither and its own
  comment gives the reason. Resolving the asymmetry the other way would have
  re-introduced the stale-window trap that comment names. The `KvStorage`
  variants keep their `max_seq`; that is where the live window is read from.
* `KvCache::update_iso3`, `update_iso4`, `update_iso3_sym`, `update_iso4_sym`,
  `update_iso_k_only_3` and `update_iso_k_only_4` — six entries over three:
  `update_iso_v`, `update_iso_sym` and `update_iso_k_only`, each resolving the
  width from the `KvStorage` variant it was dispatched on, the shape
  `update_rotor_v` and its siblings already had. Their bodies are
  `iso_v_update<BITS>`, `iso_sym_update<BITS>` and `iso_k_only_k_side<BITS>`,
  beside the rotor bodies.
* Nine further width-twin pairs in the same file, which could not stay
  per-width once the store is one type — a shared body could not call them.
  Eighteen fns over nine: `push_iso_k_block`, `iso_gpu_append_into_k_blocks`,
  `iso_gpu_append_into_v_blocks`, `iso_k_sync_ring`, `iso_v_sync_ring`,
  `drop_blocks_when_ring_live_iso_k`, `drop_blocks_when_ring_live_iso_v`,
  `iso_k_only_gpu_append` and `iso_sym_gpu_append` (the last two a dispatcher
  plus an `_at<BITS>` body, the rotor shape). With them went
  `iso4_v_gpu_append_for_test`, whose only job was to name the private 4-bit
  appender for a test; the seam is now `iso_v_gpu_append_for_test<BITS>` and
  states its own feed.
* `sdpa.rs`'s `iso_k_gpu_append` and `iso_sym_gpu_append` — two wrappers whose
  only job was picking between the two width-named entries. Each had one
  caller, which now calls the update dispatcher directly.
* `LEGACY_ISO4_V_FEED` — a `const RingFeed` with one width's name on it. Its
  `kvcache/ring_feed_routing_tests.rs` cell is now
  `a_skip_feed_reaches_the_iso_v_block_path_at_every_shape`, over the feed the
  caller states rather than over a width-named constant.
* The two `bits` match blocks in `iso_gpu_encode_block_retaining` and
  `iso_gpu_encode_ring_only`. Width dispatch has one home,
  `crate::isoquant_msl_dispatch`, shared with the stores' `append_gpu` and
  `dequant_gpu`. Without it the collapse would have planted three more copies
  of that match.
* The twin halves of the storage test files: `quant_iso_v4_tests.rs` and
  `quant_iso_k4_tests.rs` folded into `quant_iso_v_tests.rs` /
  `quant_iso_k_tests.rs`. What is deleted is the duplicated **body**, not the
  coverage: each case is one generic body with the width as a parameter plus
  one `#[test]` per width, so the cell count did not fall —
  `cargo test -p rmlx-kv-quant` reports 572 passed on both sides of the
  collapse.
* `rotor_storage_mismatch` — renamed `storage_mismatch`. The iso V and
  symmetric entries build the same `Error::Mlx`, and a second copy would have
  been the twin this change exists to remove.
* `iso_storage_mismatch` — an `Error::KvStorageMismatch` builder for six call
  sites in a file that constructs that variant inline at sixteen others. One
  shape per file: the iso sites are inline now, like the rotor K-only and asym
  ones.
* The V axis of `iso_v_update` and `iso_sym_update` — one block, twice. It is
  `iso_v_encode_decode<BITS>` once, and the trace message the two copies
  differed on is a `variant` field. §5 carries the measurement.
* The stale comment in `QuantIsoV3::append_gpu` that read "in test mode it uses
  OnceLock latching on first read". The `OnceLock` was removed; `lib.rs` says
  so three lines from the gate.
* The `QuantIsoV4` doc comment "CPU-only. The existing MSL kernel is hard-coded
  for `bits=3`; an iso4 MSL kernel variant is deferred." Both halves were false
  on this tree: the store carries a GPU ring, and
  `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs` dispatches an iso4 kernel pair.
  The sentences in `docs/KV_QUANT.md` and `docs/KV_CACHE.md` that said the same
  went with it, including `docs/KV_CACHE.md` §5.7.3's "`QuantIsoV3` is the one
  GPU-resident member … the remaining seven are CPU-only", which named one
  width and counted eight instantiations. It is `QuantIsoV<BITS>` at both
  widths now, and six.

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
  existed before exists after, with the same bytes and the same tokens.
* **No `check-*` Make target.** None keys on the iso storage file names;
  `check-kv-codec-disposition` keys on `ALL_KV_QUANTS` plus the disposition
  predicates, which are untouched.
* **`scripts/lib/debt_report.py`'s existing populations.** The two iso ones are
  added beside them, not in place of them.

## 8. What each chunk ran

The test chunk ran the oracle, this doc and the first mutation run — no engine
code, no GPU test, no served capture, no performance number. The code chunk
ran the collapse and re-ran everything below.

| Gate | Result |
|---|---|
| `cargo test -p rmlx-kv-quant --lib iso_store_bytes` | 6 passed, 0 failed, **0 pins edited** |
| `cargo test -p rmlx-kv-quant --lib rotor_store_bytes` | 5 passed, 0 failed — the rotor `PINS` table is untouched |
| `cargo test -p rmlx-kv-quant` | 572 passed, 0 failed, 257 ignored before the GPU tests landed; 259 ignored after |
| `cargo fmt` | clean |
| `make lint` | clean, `-D warnings` across the workspace |
| `make check-no-inline-tests` | OK |
| `make check-gpu-tests-ignored` | OK, 354 files across 12 workspace members |
| `make check-gpu-tests-ignored-fixtures` | OK |
| `make check-doc-source-citations` | OK, 310 cited paths resolve |
| `make check-kv-codec-disposition` | OK, 28 codecs classified, 17 inert — the verdict per codec is unchanged |
| `make check-kv-layer-quants` | OK |
| `make check-metal-compiles` | SKIP — this host has Xcode selected without the Metal Toolchain component. The hosted `msl` job is strict |
| `make debt-report-selftest` | OK, 89 cases |
| `make gpu-runner-selftest` | OK |
| `make gpu-test CRATE=rmlx-kv-quant FILTER=iso` | 63 selected tests, second run clean. See below |

`make ci` belongs to the orchestrator and `make ci-perf` to the integration
window, which also owns the served capture in §6.

### The GPU run, and the one red it produced

The two new parity tests ran and passed, and the run produced **no
shader-validation hit**, so neither derives a
`scripts/gpu_validation_census.txt` entry — the runner fails on an unpinned
hit, so a clean pass is the evidence and the 3-bit pair carries no entry
either. The narrowed selection leaves the `rmlx-models` entries reported as
"not enforced in full", which is what a filtered run is supposed to say.

**One test went red on the first validated run and not the second:**
`kvcache::v_mirror_alloc_tests::iso_decode_does_not_copy_the_v_mirror`, at
`688 per mille of a V prefix copy, outside the measured band 500..=660`. It is
a resident-memory growth measurement, and the band it missed is the **clean
floor** — the packed-K view's own per-token materialisation — not the defect
bound. The defect bound is 1500 per mille and held with better than 2x margin
in the failing run, so the V mirror was not being copied at any point.

Measured after: it passes in isolation, it passes on a second
`make gpu-test CRATE=rmlx-kv-quant FILTER=iso`, and it passes three times in a
row under the same crate-and-filter selection without shader validation. One
red in five runs of a byte-counting probe, on the reading that instrumenting
every Metal pipeline perturbs.

**Traced, and there is nothing to change in code.** The `IsoKOnly3` decode loop
allocates nothing this chunk added: V on that codec is the bf16 mirror and
never reaches `append_gpu`, and 688 per mille is the clean-floor term — the
packed-K view's own per-token materialisation — not a V copy. The band
`500..=660` is a single pinned measurement with no allowance for the allocator
perturbation shader validation introduces.

**The base arm, owed at the integration window.** Not one run: at least five
runs of `make gpu-test CRATE=rmlx-kv-quant FILTER=iso` at the commit before the
collapse, under the same validation, with **each run's per-mille value
reported**. If the base lands outside the band even once, the pin is
under-wide and is re-measured across validated runs rather than attributed to
this change. Until that is done this is an unattributed red, not a clean run.

### The helper extraction

The store-byte serialisation the rotor pin carried — `fnv1a64`, `StoreBytes`,
`array_bytes`, `dtype_tag`, `f32_arr` and `push_quant_k` — moved to
`crates/rmlx-kv-quant/src/test_utils.rs` and is shared by both pin files.
Copying it into a second file would have planted a twin in the change whose
purpose is removing one. `rotor_store_bytes_tests.rs` lost those six items and
gained one `use`; its pins, its drive and its assertions are unchanged, which
the 5-passed run above is what says.
