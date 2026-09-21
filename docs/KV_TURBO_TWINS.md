# TurboQuant K storage and update twins

The 3-bit and 4-bit TurboQuant **K-side** storage types and their `update_*`
bodies are the third width-twin pair in `rmlx-kv-quant`, after the rotor pilot
and the iso collapse. This doc is the record the collapse is taken on: the
premise re-measured against the tree, the decisions the four named divergences
force, the oracle, the mutation run, the duplication figure, and the removals
owed.

**The method is not restated here.** [`docs/KV_ROTOR_TWINS.md`](KV_ROTOR_TWINS.md)
holds it: how a premise is re-measured, why a served digest is not the oracle,
what a store-bytes pin is, how a mutation run is conducted, how the duplication
figure is produced. [`docs/KV_ISO_TWINS.md`](KV_ISO_TWINS.md) holds the second
worked example and the argument for a doc per family. This doc holds only what
the turbo family does differently.

**The collapse has landed.** The test chunk wrote the oracle, this doc and the
first mutation run and changed no engine code; the implementing chunk took
every decision below and this doc now records the tree as it is. §1's premise
table is the **branch-point** measurement and is kept as the record of what was
collapsed, not as a description of the tree; §5 carries the after arm beside
the before one. Real-model rows (§6) are still deferred to the integration run.

The oracles are
`crates/rmlx-kv-quant/src/kvcache/turbo_store_bytes_tests.rs` (the CPU store
bytes) and `crates/rmlx-kv-ssd/src/block_io_turbo_hydrate_tests.rs` (the SSD
hydrate, which is where the divergences live).

## 1. Premise, re-measured against the tree

Method: the module's own `normalize()` (fold every digit run touching an
identifier to `N`) plus `matched_lines()`, hand-run before the populations of
§5 existed and re-run through them after. The "differing" column strips `//`
line comments, applies the same digit fold, then counts non-`equal` `difflib`
opcode lines.

Measured at the branch point, before the collapse. The rows are what was
removed, not what the tree holds.

| Pair | Lines | Matched | Differing | Verdict |
|---|---|---|---|---|
| `quant_k_turbo3.rs` / `quant_k_turbo4.rs` | 540 / 472 | 412 | 58 of 362 / 326 | HOLDS exactly |
| `quant_k_turbo3_tests.rs` / `quant_k_turbo4_tests.rs` | 489 / 223 | 154 | 276 of 340 / 182 | HOLDS exactly |
| `update_tsym3` / `update_tsym4` bodies | 74 / 80 | 59 | 30 of 62 / 74 | MOVED — see below |
| `write_quant_k_turbo3` / `_4` | 36 / 36 | 36 | 0 | HOLDS — byte-identical bodies |
| `read_quant_k_turbo3` / `_4` | 16 / 17 | 13 | 7 | HOLDS in substance |
| `read_tsym3` / `read_tsym4` | 9 / 9 | 8 | 2 | new row; the issue's table omits this pair |
| `k_turbo3_shape` / `k_turbo4_shape` | 3 / 3 | 3 | 0 | new row; byte-identical bodies |
| `turbo_k3_fused_qk_msl.rs` / `_k4_` | 241 / 246 | 220 | 2 of 99 / 99 | HOLDS exactly |
| `turbo_k3_fused_qk.metal` / `_k4_` | 73 / 70 | — | 35 raw, 16 folded of 50 / 46 | MOVED — see below |
| `turbo_k3_fused_qk_header.metal` / `_k4_` | 11 / 19 | — | 26 of 10 / 18 | HOLDS — the codebook array |

Three rows moved and each moved for a stated reason.

* **`update_tsym3` / `update_tsym4`, 74 / 80 not 79 / 85.** A counting
  convention, not drift. `scripts/lib/debt_report.py` measures a body brace to
  brace; the issue's figure includes the five signature lines above the opening
  brace. `74 + 5 = 79` and `80 + 5 = 85`, and the differing count (30) is the
  issue's exactly. The bodies have not changed since `rMLX 0.1.0` — one commit
  touches them.
* **The `.metal` kernel pair, 35 raw differing lines not "~40".** The
  conclusion the issue draws from that row is unaffected: the 3-bit kernel
  builds a `ulong window` from up to two consecutive `u32` words, and the 4-bit
  one is nibble-aligned inside one word. The unpacking really does differ by
  width. **Out of scope, as the issue says.**
* **"8 extra tests" in `quant_k_turbo3_tests.rs` is 7.** The file holds ten
  `#[test]` fns against the 4-bit file's three. The seven the issue names by
  name are exactly the seven that exist; the count beside them is wrong.

### Turbo spellings against `ALL_KV_QUANTS`

Six, and the list holds exactly the six the issue names. **None is missing and
none is extra.**

| `KvQuant` | `Display` | K side | V side |
|---|---|---|---|
| `K8VTurbo3` | `k8vturbo3` | `QuantK` (affine q8_0) | `QuantV` bits 3 |
| `K8VTurbo3Tcq` | `k8vturbo3tcq` | `QuantK` | `QuantV` bits 3, `use_tcq` |
| `K8VTurbo2` | `k8vturbo2` | `QuantK` | `QuantV` bits 2 |
| `K8VTurbo2Tcq` | `k8vturbo2tcq` | `QuantK` | `QuantV` bits 2, `use_tcq` |
| `TurboSym3` | `tsym3` | **`QuantKTurbo3`** | `QuantV` bits 3 |
| `TurboSym4` | `tsym4` | **`QuantKTurbo4`** | `QuantV` bits 4 |

**Only `TurboSym3` and `TurboSym4` construct the K twin** — confirmed by a scan
of every non-test reference to the two types, and asserted at runtime off the
live storage variant by
`only_the_symmetric_spellings_build_the_k_side_turbo_store`. The four
construction sites are `kvcache/update.rs` (two `exit_prefill` arms, two decode
arms), plus the SSD hydrate constructors.

**Every turbo spelling is decode-inert, and more strongly than the issue
states.** All six return `false` from `decode_reads_packed_store()`, and all
six also return `true` from `feeds_bf16_k_at_decode(false)` and
`feeds_bf16_v_at_decode(false)`, so `materialises_packed_store()` is `false`
for every one of them. `exit_prefill` does not merely stop reading the payload
— it clears it. §3 states what follows for the oracle.

**The V side really is one runtime-`bits` type**, as the issue says, with one
qualification that matters for divergence 5: `QuantV::append` enters its GPU
branch on `device == Device::Gpu` **with no bit-width guard**, and the first
thing inside that branch is `if self.bits != 4 { return Err(...) }`. So the V
side is collapsed on the CPU and 4-bit-only on the GPU, and the callers carry
the routing.

### `max_seq` on the K stores is inert everywhere but the hydrate

Both stores carry the field. Neither ever reads it: `append` sizes its buffer
from its own `max_seq` **parameter**, and `try_deep_clone` copies the field
across. Nothing else touches it. The live window a decode step reads is the
`KvStorage::TurboSym{3,4}` variant's own `max_seq`, and the SSD geometry is
written from that variant field at both widths. So the divergence in §2 is
observable at the hydrate and nowhere else — which is why the pin that holds it
is in `rmlx-kv-ssd` and the CPU pin deliberately leaves `max_seq` out of its
digest.

## 2. The divergences, as constraints on the collapse

Five, not four. The four the issue names, plus one it does not, which is the
largest of them.

### (1) Constructor shape — the 3-bit form is the reference

`QuantKTurbo3::new(init_shape, max_seq)` exists; `QuantKTurbo4` has no `new`
and both its call sites build a nine-field struct literal. `QuantKTurbo3` also
carries `reset()`, `dequant()` and the exported constant `TURBO3_K_BITS`, none
of which the 4-bit file has — three more 3-bit-only items the issue's list
omits.

**Decision, landed.** One constructor, `QuantKTurbo::<BITS>::new(init_shape,
max_seq)`, used by all four construction sites. `reset()` and `dequant()` are
now available at both widths rather than at 3-bit only, which is the iso
store's shape and costs no behaviour: neither has a non-test caller, and both
are exercised per width by `quant_k_turbo_tests.rs`. `TURBO3_K_BITS` is gone;
`TURBO_K3_BITS` and `TURBO_K4_BITS` replace it, both read by the width guard.

**Observable: the CPU pin's `store_after_chunk` at both widths — over two of
the four sites, not four.** The pin drives `in_prefill` false and so reaches
only the two decode arms. The other two are the `exit_prefill` literals, and
`exit_prefill` returns at its `materialises_packed_store()` gate before every
turbo arm, so **those two execute for no turbo spelling today**. They are kept
deliberately — they are the re-enable path a codec takes when it grows a decode
kernel over its own store — and the guard that pins them unreachable is
`warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`,
which sweeps every variant and fails the moment an arm and its classification
disagree. A collapse that gets those two literals wrong is therefore caught by
nothing in this campaign's oracles; it is caught the day the predicate flips.

`crates/rmlx-kv-quant/src/storage/quant_planar_k.rs` documented its own
GPU-path init as "matching the `QuantKTurbo4` inline-literal signature". The
literal is gone and so is that clause.

### (2) `from_cpu_blocks` and the hydrated window — the 3-bit form is the reference

`QuantKTurbo3::from_cpu_blocks(blocks, shape, bits, max_seq)` takes the window;
`QuantKTurbo4::from_cpu_blocks(blocks, shape, bits)` does not and hard-codes
`max_seq: 0`. `read_tsym4` already reads the geometry's `max_seq` — it needs it
for the `KvStorage` field — and simply does not forward it.

**This divergence changes an observable at one width today**, which the SSD pin
measures: after a spill and a hydrate of the same cache, the 3-bit K store
carries `4096` and the 4-bit one carries `0`.

**Decision, landed: the 3-bit behaviour is the reference.** The collapsed
`from_cpu_blocks(blocks, shape, max_seq)` takes the window at both widths, the
one `read_quant_k_turbo` forwards the value it already has, and **the 4-bit
hydrate pin moved from `0` to the written window** —
`HYDRATED_K_MAX_SEQ_4BIT`, the one expectation constant this campaign changed.
The cell that records it is now
`tsym4_hydrate_restores_the_k_payload_and_the_window`; it was named
`..._but_not_the_window`, which the move made false.

The `bits` parameter went with the change. A const-generic store states its own
width, so a second `bits` argument beside it is a way for a caller to disagree
with the type. `from_cpu_blocks` sets `bits: BITS`.

Why this direction and not the other:

* It is the "restore what was written" direction, which is the SSD tier's job.
  Resolving the other way would delete a parameter whose sibling documents it
  deliberately ("Takes `max_seq` explicitly (do not derive from shape[2])").
* The call site already has the value. This is a one-line completion of a
  helper that reads `max_seq` and drops it, not a new plumbing run.
* The counter-argument is that the field's doc says "the sequence length the
  GPU buffer was sized for", and a hydrated store has no GPU buffer, so `0` is
  the truthful answer. It does not survive: the first GPU `append` after a
  hydrate overwrites the field from its own parameter, so under either choice
  the field is only ever read-visible on a CPU-only hydrated store, where
  nothing reads it at all.

An owner who wants the other direction has one thing to change and one pin to
re-bless; the pin names both numbers so either move is loud.

### (3) The unsafe scale-byte cast — the 3-bit (safe) form is the reference

In `append`'s GPU hydrated-init branch, the 3-bit store builds its scale bytes
with `flat_scales.iter().flat_map(f32::to_le_bytes).collect()` and the 4-bit
store with `unsafe { std::slice::from_raw_parts(...) }`. Same operation.

**Decision, landed.** The safe form, once. **Observable:** not the CPU pin —
that branch needs a `Device::Gpu` append after a hydrate, which no test here
drives. What holds it is that the two forms produce the same bytes by
construction (`f32::to_le_bytes` is the definition of the little-endian layout
the cast reinterprets), plus `make gpu-test`. The collapse removed one `unsafe`
block from the crate and added none.

### (4) `.eval()` before `to_bytes()` — the 4-bit form is the reference

`read_quant_k_turbo4` calls `.eval()` on both loaded tensors;
`read_quant_k_turbo3` does not, and its doc comment argues the call is
unnecessary.

**Measured: neither form changes a byte.** §4's M13 removes the call from the
4-bit path and M16 adds it to the 3-bit path, and both leave all three SSD
cells green. This is a convention decision, not a behaviour one, and the doc
says so rather than dressing it up.

**Decision, landed: keep the call.** Every other reader in the same file —
`read_quant_k`, `read_quant_v_bits`, `read_quant_planar_v` — calls it.
`read_quant_k_turbo3` is the one outlier in the file, and a collapsed helper
that reads differently from its four neighbours costs a future reader a
question for no gain. The 3-bit doc comment's claim is true of the loader, not
of `to_bytes()`, so it is the weaker of the two guarantees.

### (5) The V-axis device split — **both** widths are the reference, and the width decides

The issue does not name this one and it is the one that breaks a blind
width-substitution.

| | `update_tsym3` | `update_tsym4` |
|---|---|---|
| V `append` device | `Device::Cpu`, always | the caller's `device` |
| V `dequantize_choice` device | `Device::Cpu`, always | the caller's `device` |
| what it does with `v_arr_opt` | discards it (`_`) and always rebuilds the rows from the f32 vec | takes the `Array` when the GPU returned one |
| `v_f32` materialised | always | only when the caller's device is CPU |

The third row is the one a reader skips. `dequantize_choice` returns
`(Vec<f32>, Option<Array>)` and fills the `Array` only on the GPU path; the
3-bit body discards it because, on `Device::Cpu`, it is always `None`. A
collapsed body that kept the 3-bit spelling would silently drop the 4-bit GPU
result and round-trip it through the host vector instead — correct output, one
device-to-host copy per decode step. So the width rule governs three sites, not
two.

On a CPU drive the two are the same routing. On a GPU drive they are not, and
the 3-bit forcing is **load-bearing, not a perf preference**: `QuantV::append`
enters its GPU branch on the device alone and then refuses `bits != 4`. A
collapsed body that adopted the 4-bit shape would hand a 3-bit `QuantV` the GPU
device and get `Error::Quant` on every prefill chunk. A collapsed body that
adopted the 3-bit shape would take the 4-bit V axis off the GPU, which changes
its store and its speed.

**Decision, landed.** `tsym_update` resolves the V device from the width: the
caller's `device` at `BITS == TURBO_K4_BITS`, `Device::Cpu` otherwise, `v_f32`
materialised exactly when that resolved device is CPU, and the `Array` the
dequant returns taken when it returns one. That is the one rule that reproduces
both bodies. **Observable: none on CPU.** §4's M15 drops the forcing and every
CPU cell stays green. The gate is the `#[ignore]` pair in
`crates/rmlx-kv-quant/src/kvcache/turbo_v_axis_gpu_tests.rs`, which the
implementing chunk added for exactly this, plus the served capture in §6.

### What cannot move

* Every `TurboSym3` and `TurboSym4` cell on the five pinned columns — packed
  store bytes after the bulk append, after the decode steps and after
  `truncate_to`; the K/V rows attention receives; `resident_bytes()`.
* The four asymmetric spellings' cells on the same five columns. The collapse
  touches no type they use, and they are pinned so that a change which reaches
  further than intended says so.
* The SSD spill bytes and the hydrated payload at both widths — codes, scales,
  per-block original shape and bit tag, accumulated shape, block count — and
  the `KvStorage`-level `max_seq` the decode path actually reads. The one field
  allowed to move is the K store's own `max_seq`, per divergence 2.
* The six `KvQuant` spellings, their `Display` text, their `FromStr` spellings,
  the six `KvStorage` variants, the CLI names, `TURBOSYM3_LAYOUT_TAG` /
  `TURBOSYM4_LAYOUT_TAG`, the layout-key salt, the four `.metal` entry points
  and `scripts/gpu_validation_census.txt`.
* **A layout-key salt bump is a STOP.** Nothing in this collapse changes an
  on-disk layout, so a salt bump would mean the collapse changed the wire
  format, which it must not.
* `make check-kv-codec-disposition`'s verdict per codec. No disposition
  predicate arm changes; the collapse moves no codec between the bf16-mirror
  class and the store-reading class.

## 3. The oracle

### Why a served digest cannot judge this change, at all

Stronger on this family than on any other in the tree. `materialises_packed_store()`
is `false` for all six turbo spellings, so `exit_prefill` **clears** the packed
payload and decode runs off the bf16 mirror on both axes. A served request at
`tsym3` or `tsym4` reads no turbo store byte after prefill. A build with a
correct store and a build with an empty one emit the same tokens, at any
context, on any model.

So a byte-identical served digest is not evidence here — it is the expected
result of doing nothing and also the expected result of destroying the store.
The store-bytes pin is the only oracle, and §6's served rows exist to catch a
change that reached past the store, not to judge the store.

### The CPU pin

`crates/rmlx-kv-quant/src/kvcache/turbo_store_bytes_tests.rs`, eight `#[test]`
fns, 12 cells (6 spellings x 2 shapes), 5 pinned quantities per cell:
`store_after_chunk`, `store_after_decode`, `store_after_truncate`, `rows`,
`resident_bytes`. Shapes `(1, 128)` and `(4, 96)`, the same pair the rotor and
iso pins use.

Two of the twelve pin rows are copies of two others, on purpose — see "the TCQ
finding" below.

### The census needs two tokens, and that is the family's own trap

`turbo_spellings()` filters `ALL_KV_QUANTS` by `Display` text containing
`turbo` **or** `tsym`. A single `contains("turbo")` filter — the obvious
translation of the iso file's `contains("iso")` — finds four of the six and
misses `tsym3` and `tsym4`: **exactly the two spellings that build the store
this whole collapse is about.** The symmetric members spell themselves with a
token that shares no substring with the family name. No other spelling's
`Display` text contains either token, so the two together are the membership
test, with `TURBO_SPELLING_COUNT` (6) as the anchor beside them and
`TURBO_K_TWIN_COUNT` (2) as the scope anchor.

### The TCQ finding

`k8vturbo3` and `k8vturbo3tcq` write **byte-identical** stores, and so do
`k8vturbo2` and `k8vturbo2tcq`. Measured, on both shapes.

The reason is structural. `crate::tcq::build_transition_table` gives every
trellis state an outgoing edge for **every** level — the transition reads
`level & 1` to pick the next state and forbids no level — so the additive
Viterbi path cost is minimised position by position, which is the greedy
nearest-centroid assignment the plain encoder already makes. The trellis
constrains nothing. The flag is set, the Viterbi pass runs, and the codes come
out the same.

This is not the turbo K collapse's business and nothing in this change touches
it. It is recorded because two identical pin rows with no explanation read as a
copy-paste error, and because the property should be a measurement with its
reason attached rather than a coincidence.

A second stale statement, in the same area and also not this issue's to fix:
`decode_reads_packed_store`'s own comment calls the bf16-mirror family's store
one "written once at `exit_prefill` and never read again on a seeded cache".
For every member whose `materialises_packed_store()` is false — which is all
six turbo spellings — `exit_prefill` returns before writing anything and clears
what is there. The store is not written once; it is not written at all. A
removal the implementing chunk owes if it touches that comment's file.
`the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes` asserts the
flag and the identity together, so a change that makes the trellis constrain
the level set turns red and names itself. **Follow-up, not this issue's:
either the trellis is wrong or the two spellings are the same codec.**

### The SSD pin

`crates/rmlx-kv-ssd/src/block_io_turbo_hydrate_tests.rs`, three `#[test]` fns,
one cell per width plus a control. Field by field, not a digest: the
`rmlx-kv-quant` serialisation helpers are `pub(crate)` to that crate, and
copying them across to build a digest would plant the twin this campaign
removes — and a collapse's reviewer wants the field named, not the store.

Both the store and its one block are destructured **exhaustively**, so a field
added to either — or a constructor that starts filling one that used to sit at
its default — fails to compile here rather than passing unread. The hydrated
store is also asserted to claim no GPU buffer and no buffer bookkeeping: a
hydrate builds a CPU-path store, and a capacity it invented would hand the next
append a geometry no buffer backs.

The third test spills and hydrates **both** widths and compares what the two
engines returned. Comparing the two pinned constants instead would execute no
engine code. Before the collapse it asserted the two windows differed; the
collapse made them agree, so it is now
`the_two_widths_hydrate_the_same_window` and asserts two things: that the two
engines return the same window, and that the agreed window is the one the spill
wrote. The second assertion is what stops the pair agreeing on a wrong value.
§4's U1 was the measurement that predicted the move.

### What the CPU oracle cannot see, and the GPU tests the collapse owes

| Unseen | GPU test owed | Census disposition |
|---|---|---|
| The V-axis device split (divergence 5) at `Device::Gpu` — the one defect a blind merge introduces | **landed**: `kvcache/turbo_v_axis_gpu_tests.rs`, a `Device::Gpu` drive of `KvCache::update` with `in_prefill` false and no bf16 seed — the GPU twin of this file's own drive — at `tsym3` and `tsym4`. It asserts the appends succeed, the row shapes, and the V store's own disposition per width: no GPU mirror and one CPU block per append at 3 bits, a GPU mirror and no CPU block at 4. Under the lost width rule the 3-bit cell fails on its first append, because `QuantV::append` enters its GPU branch on the device alone and returns `Error::Quant` for `bits != 4`; under the reverse edit the 4-bit cell fails on its disposition. A hard red, not a tolerance. **Not a served prefill** — see the note below the table | derived from a run: no shader-validation hit, so no census entry |
| The GPU `append` path — buffer allocation, paged growth, the MSL encode dispatch at both widths | covered by `storage::quant_k_turbo_tests::quant_k_turbo3_gpu_two_append_multi_head_roundtrip` and its 4-bit sibling, both `#[ignore]`-gated | no new entry |
| The hydrated-init upload branch, where divergence 3 lives | **not written.** The branch is one safe conversion after the collapse and the two forms it replaced produce the same bytes by construction; a test for it would assert `f32::to_le_bytes`. Recorded as owed rather than claimed | — |
| CPU/MSL parity of the K codec | **landed at both widths**: `cpu_msl_parity` is one generic body with `quant_k_turbo3_cpu_msl_parity` and `quant_k_turbo4_cpu_msl_parity` entering it | derived from a run: no hit |
| The fused-QK decode kernels | `turbo_k3_fused_qk_msl_tests` / `turbo_k4_fused_qk_msl_tests`, unchanged by the collapse | no |

#### Which routes reach the turbo V append at all

This decides what the first row can ask for, and the obvious answer — "serve a
prefill on the GPU" — is wrong.

* **A served prefill does not reach it.** `KvCache::update` with `in_prefill`
  true returns into `update_prefill_raw` and never enters the storage
  dispatch, and `exit_prefill` returns at its `materialises_packed_store()`
  gate, before every arm that would bulk encode. Neither touches
  `update_tsym`.
* **A served decode step does not reach it either.** `update_tsym` opens with
  `if self.decode_fp16_k.is_some() { return self.update_decode_fp16(..) }`,
  and `exit_prefill` is what installs that seed. On a normally-prefilled cache
  every decode step short-circuits.

Two routes do reach it, and one of them is the GPU test the implementing chunk
owes:

1. **A `Device::Gpu` drive of `KvCache::update` with `in_prefill` false and no
   bf16 seed** — the same drive this file's CPU cells use, on the other device.
   This is the one that was written (`kvcache/turbo_v_axis_gpu_tests.rs`): it is
   hermetic, needs no model, and fails on the first append under the lost width
   rule.
2. **An SSD-hydrated `tsym3` cache resuming decode on `Device::Gpu`.** A
   hydrated entry carries a store and no mirror, so `decode_fp16_k` is `None`
   and the dispatch reaches `update_tsym` for real. It is the production route
   that would hit the defect, and it needs a spilled block plus a device, so it
   belongs beside the SSD suite rather than beside the store-bytes pin.

## 4. Mutations

Every mutation below was applied to the tree, run, and reverted from a snapshot
whose sha256 of the **working** file was compared before and after —
`git checkout --` is not used; it would revert uncommitted work. Each row names
the assertion that caught it, not just that something failed.

**The table is the branch-point run**, sixteen of nineteen red. The collapsed
tree was re-run against the same list and §4.1 holds the result.

| # | Edit | Caught by |
|---|---|---|
| M1 | `QuantKTurbo4::append` — encode at the literal `3` instead of `self.bits` | 2 tests: the pin (`tsym4 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`) and the geometry (`the packed code plane must be ceil(values * 4 / 8) bytes`) |
| M2 | `QuantKTurbo3::append` — drop the last code byte (`codes.pop()`) | 4 tests. The driver's own decode-step append fails first, so the panic is at the drive and not at a column |
| M3 | `turboquant.rs` — `GROUP_SIZE` 32 -> 64, moving the group boundary | 5 tests: the pin's **chunk** column and the geometry's scale count (`one scale per group of 32`) |
| M4 | `turboquant.rs` — one `CODEBOOK_3BIT` centroid moved by one ULP | the pin's **rows** column **only** (`k8vturbo3 @ kv_h=1 head_dim=128: the K/V rows attention receives moved`). The code assignment does not change, so the store bytes stay; only what the attention reads back moves |
| M5 | `QuantKTurbo3::truncate_to` — `(n - 1).max(0)` | the pin's **truncate** column **only** (`tsym3 @ kv_h=1 head_dim=128: packed store bytes after truncate_to moved`) |
| M6 | `QuantKTurbo3::dequantize_choice` — drop `transpose_chunked_seq_heads`; the store is untouched, only the rows attention receives move | the pin's **rows** column only, at **shape B only** (`tsym3 @ kv_h=4 head_dim=96`). Shape A has `kv_h = 1`, where the transpose is the identity — which is why both shapes are pinned |
| M7 | `QuantKTurbo3::byte_size` — drop the CPU-blocks term | the pin's **resident_bytes** column only (`tsym3 @ kv_h=1 head_dim=128: resident_bytes moved`) |
| M8 | `QuantKTurbo3::append` — coalesce a one-token block into its predecessor | 3 tests, again at the driver's append rather than at a column: the block bookkeeping the codec relies on breaks outright |
| M8b | `QuantKTurbo3::append` — zero the first scale of the **last** decode block only | the pin's **store_after_decode** column only (`tsym3 @ kv_h=1 head_dim=128: packed store bytes after 3 decode steps moved`). The chunk column cannot see a decode block and the truncate column drops it |
| M9 | `quant.rs` — `KvQuant::TurboSym4` removed from `ALL_KV_QUANTS` | 2 tests: the census (`turbo spelling census moved`) **and** the scope anchor (`the set of spellings backed by the K-side turbo store moved: ["tsym3"]`) |
| M10 | `quant.rs` — `TurboSym3` moved into the store-reading arm of `decode_reads_packed_store` | `every_turbo_spelling_is_decode_inert`. This is the premise the whole oracle rests on, so it is asserted rather than assumed |
| M11 | `update_k8vturbo3_tcq` — `use_tcq: false` | `the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes` (`the Viterbi assignment flag on the V store is not what the spelling selects`) |
| M12 | `read_quant_k_turbo3` — pass `0` instead of the geometry's `max_seq` | the SSD pin's 3-bit window cell. This is divergence 2 resolved the other way, and it is loud |
| M13 | `read_quant_k_turbo4` — remove both `.eval()` calls | **uncaught** — 3 passed, exit 0 |
| M14 | `write_quant_k_turbo3` — scale the serialised scale plane by 2 | the SSD pin's payload (`tsym3: the scale plane changed across the spill/hydrate round trip`) |
| M15 | `update_tsym3` — V axis routed to the caller's `device` instead of `Device::Cpu` | **uncaught** — 8 passed, exit 0 |
| M16 | `read_quant_k_turbo3` — add the two `.eval()` calls | **uncaught** — 3 passed, exit 0 |
| U1 | `QuantKTurbo4::from_cpu_blocks` — `max_seq: 0` -> `4096`, i.e. divergence 2 resolved the recommended way | the SSD pin's 4-bit window cell **and** `the_two_widths_hydrate_a_different_window_today`. The control measures both hydrates rather than comparing two constants, which is what lets it see the widths converge |
| U2 | `QuantKTurbo4::from_cpu_blocks` — `gpu_capacity: 0` -> `7`, a hydrated store claiming a buffer it has none of | the SSD pin's GPU-bookkeeping assertion (`the hydrated store carries GPU buffer bookkeeping for a buffer it does not have`). Both stores are destructured exhaustively, so a field the pin does not read cannot exist |
| U3 | `KvQuant::K8VTurbo2Tcq` — `Display` text changed to a third token (`lloyd2tcq`) **and** its storage redirected to `KvStorage::TurboSym3`: a spelling that builds the K twin and that the census filter does not select | the scope anchor, by name (`lloyd2tcq is backed by the K-side turbo store and the Display filter does not select it`), plus the census count. **Its control is below** |

M4, M5, M6, M7, M8b and M12 are each caught by exactly one assertion, and M5,
M6, M7 and M8b each by exactly one **column** — truncate, rows, resident_bytes
and store_after_decode. With M1 and M3 on the chunk column, **all five columns
are load-bearing** rather than redundant. M6 fires at one shape only, which is
what says both shapes are.

### 4.1 The same list, re-run on the collapsed tree

Seventeen of eighteen runnable rows red. Two rows changed shape, for reasons
that belong to the collapse and are stated rather than folded in.

| # | On the collapsed tree | Result |
|---|---|---|
| M1 | the one CPU append encodes at the literal `3` instead of `self.bits` | RED — the pin and the geometry test |
| M2 | the one CPU append drops the last code byte | RED — 4 tests |
| M3 | `GROUP_SIZE` 32 -> 64 | RED — 5 tests |
| M4 | one `CODEBOOK_3BIT` centroid to the next representable f32 | RED — the pin only |
| M5 | `truncate_to` keeps `(n - 1).max(0)` | RED — the pin only |
| M6 | `dequantize_choice` drops `transpose_chunked_seq_heads` | RED — the pin only |
| M7 | `byte_size` drops the CPU-blocks term | RED — the pin only |
| M8 | the CPU append coalesces a one-token block into its predecessor | RED — the pin only |
| M8b | the CPU append zeroes the first scale of a one-token block | RED — the pin only |
| M9 | `TurboSym4` removed from `ALL_KV_QUANTS` | RED — the census and the scope anchor |
| M10 | `TurboSym3` into the store-reading arm | RED — `every_turbo_spelling_is_decode_inert` |
| M11 | the decode-path `use_tcq: false` in `update_k8vturbo3_tcq` | RED — `the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes` |
| M12 | `read_quant_k_turbo` passes `0` instead of the geometry's `max_seq` | RED — all three SSD cells, at **both** widths now, because there is one reader |
| M13 | `read_quant_k_turbo` drops both `.eval()` calls | **GREEN**, as at the branch point. This is divergence 4's measurement, and it is why that decision is stated as convention |
| M14 | `write_quant_k_turbo` scales the serialised scale plane by 2 | RED — both payload cells |
| M15 | `tsym_update` routes the V axis to the caller's `device` at both widths | **GREEN under every CPU gate.** See §4.2 |
| M16 | not expressible. It added the `.eval()` calls to the reader that lacked them; there is one reader and it calls them. M13 is the live half | — |
| U1 | `from_cpu_blocks` hard-codes `max_seq: 0` again | RED — both window cells **and** the rewritten control's second assertion |
| U2 | `from_cpu_blocks` claims a `gpu_capacity` of 7 | RED — the GPU-bookkeeping assertion at both widths |
| U3 | `K8VTurbo2Tcq` spells itself `lloyd2tcq` and builds the K-side turbo store | RED — the scope anchor by name, the census, and the TCQ identity |

Two harness findings, recorded because each first read as a missed detection:

* The branch-point M4 edit `-1.343_908_5` -> `-1.343_908_6` is a **no-op**. Both
  literals round to the same `f32`: the ULP at that magnitude is ~1.19e-7 and
  the edit is 1e-7. The row above uses the next representable value,
  `-1.343_908_67`.
* An M11 anchor of `use_tcq: true` at the decode arms' indentation also matches
  the `exit_prefill` arms' deeper-indented line as a substring, and a
  first-match replace lands there. Those arms execute for no turbo spelling, so
  the edit was invisible. The row above anchors on the whole `QuantV` literal.

### 4.2 M15, and the drive that does catch it

**M15 is green under every CPU gate**, and after the collapse it is worse than
before: the edit that used to break one width now breaks the shared body, and
every CPU cell is still green. The CPU pin drives `Device::Cpu`, where the
caller's device is the answer the rule forces, so the drive cannot tell "CPU
because the body resolved it from the width" from "CPU because the caller
asked".

`crates/rmlx-kv-quant/src/kvcache/turbo_v_axis_gpu_tests.rs` is the drive that
can, and the pair was run under it, in both directions:

| Edit | Result on Metal |
|---|---|
| M15 — `let v_device = device;` | **RED**, `turbo_sym3_v_axis_stays_on_the_cpu_under_a_metal_drive` |
| M15r — `let v_device = Device::Cpu;` | **RED**, `turbo_sym4_v_axis_follows_the_caller_to_the_gpu` |

Both cells are `#[ignore]`-gated, so **no `make ci` run executes them**. A
reviewer of this collapse must read a `make gpu-test` result, and the one on
record is `OK: 29 GPU tests passed across 1 workspace member(s)` for
`make gpu-test CRATE=rmlx-kv-quant FILTER=turbo`. Neither new cell produced a
shader-validation hit, so neither owes an entry in
`scripts/gpu_validation_census.txt`.

### U3's control, and why the scope anchor sweeps the enum

The anchor used to iterate `turbo_spellings()` — the `Display`-filtered subset
— which made it and the census share one blind spot: a spelling the filter does
not select is invisible to both, so the census pins nothing for it and the
anchor never counts it. Measured, not argued: re-running U3 with the anchor's
loop reverted to `turbo_spellings()` leaves the anchor **green**, exit 0.

It sweeps `ALL_KV_QUANTS` now and asserts that every quant whose storage is a
symmetric turbo variant is a member of `turbo_spellings()`, so the filter is
checked against the thing it claims to select. It reads the variant
`KvStorage::new` sets at construction and drives nothing — the claim does not
rest on an append.

**The width guard is the other half of M1.** The collapsed type refuses
`QuantKTurbo<5>` at monomorphisation the way the iso stores do:
`WIDTH_IS_A_SHIPPED_ONE` is read by `NAME`, by `new`, by `from_cpu_blocks` and
by the two MSL width selectors, so no birth path and no kernel dispatch can
miss it. It is a `const` assert, so there is no run to mutate — a tree that
lost it fails to build at the first `QuantKTurbo<5>`, and a tree that kept it
never reaches one.

### The mutation I could not catch

**M15**, and it is the important one.

`update_tsym3` forces `Device::Cpu` on the V axis. Replace that with the
caller's `device`, the shape `update_tsym4` uses, and every cell of the CPU pin
stays green — because the caller's device on every cell of that pin **is**
`Device::Cpu`. The edit is the exact mistake a blind width-substitution makes,
it takes a 3-bit `QuantV` into a GPU branch that refuses `bits != 4`, and the
oracle this chunk built is structurally unable to see it.

Nothing in the assertions is missing. The blind spot is the drive: a
`Device::Cpu` pin cannot distinguish "CPU because the body forced it" from "CPU
because the caller asked". Widening the pin to drive `Device::Gpu` is not an
option for `make ci` — a Metal-touching test carries `#[ignore]` and no CI gate
runs it. **So the collapse's single largest risk is gated by `make gpu-test`
and by §6's served capture, and by nothing a `make ci` run executes.** The
reviewer must read a GPU result, not this file's.

Two further uncaught edits, both of them benign and both recorded because the
pair is the measurement behind divergence 4. **M13** removes `.eval()` from the
4-bit hydrate and **M16** adds it to the 3-bit one; both leave all three SSD
cells green. That is the evidence that the `.eval()` divergence changes no
byte, taken in both directions so the claim is not one-sided. The decision in
§2(4) is therefore about convention, and the doc says so rather than inventing
a behavioural reason for it.

Two more things this run cannot see, stated so the 16-of-19 is not read as
wider than it is. No row mutates the GPU append path, because the pin never
dispatches a kernel. And no row mutates the `.metal` kernels, which are out of
scope and which no CPU test compiles.

## 5. Duplication figure — before and after

Three populations, registered in `scripts/lib/debt_report.py`'s
`MATCHED_LINES_POPULATIONS` beside the rotor and iso ones and not in place of
them. Measured on the branch point:

```
$ bash scripts/debt_report.sh --matched-lines turbo-storage
turbo storage twins (crates/rmlx-kv-quant/src/storage): 412 matched lines over 1012 body lines (2 item(s), 1 pair(s))
$ bash scripts/debt_report.sh --matched-lines turbo-updates
turbo update twins (crates/rmlx-kv-quant/src/kvcache/update.rs): 59 matched lines over 154 body lines (2 item(s), 1 pair(s))
$ bash scripts/debt_report.sh --matched-lines turbo-ssd
turbo ssd helper twins (crates/rmlx-kv-ssd/src/block_io.rs): 60 matched lines over 129 body lines (8 item(s), 4 pair(s))
```

Per-pair, for the record: `write_quant_k_turbo3` against `_4` 36 lines of 36
(byte-identical bodies), `read_quant_k_turbo3` against `_4` 13 of 16,
`read_tsym3` against `read_tsym4` 8 of 9, `k_turbo3_shape` against
`k_turbo4_shape` 3 of 3.

After the collapse, on the same three commands:

```
turbo storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 608 body lines (1 item(s), 0 pair(s))
turbo update twins (crates/rmlx-kv-quant/src/kvcache/update.rs): 0 matched lines over 91 body lines (2 item(s), 0 pair(s))
turbo ssd helper twins (crates/rmlx-kv-ssd/src/block_io.rs): 0 matched lines over 82 body lines (4 item(s), 0 pair(s))
```

All three read a measured `0` with the population still found, which is the
answer the issue asks for. A deleted population prints `unavailable` and exits
1, and a population that quietly resolved to nothing would be
indistinguishable from a collapsed one if the module let it print `0`.

`turbo-ssd` reads four items, not one: the four helpers keep their turbo token
and lose their width, and `width_pair_key` puts each in a group of its own. The
digit-free pattern is what makes that a `0` instead of an `unavailable`.

**`TURBO_UPDATE_FN_PATTERN` widened**, from `^update_tsym` to
`(^|_)tsym(\d|_|$)`. The collapsed entry is `update_tsym` and the
width-parametric body it enters is `tsym_update`, which spell the token on
opposite sides of the name; an anchored prefix would have measured the 23-line
entry and never the body, so a re-split of that body into two width bodies
would have left this counter at `0`. The token still admits nothing else in the
file — `update_k8vturbo3` and its siblings carry no `tsym` — so the exclusion
§5's design paragraph asks for is unchanged, and the selftest's own base and
collapsed fixtures read the same figures they did. The one selftest case that
moved is the empty-population one, which used to empty the population by moving
the prefix off the front; it now removes the token, the way the iso case
already did.

### The population design

Each is a glob or a name pattern plus a root, parameterised at the registration
site through `functools.partial` — never a file or fn list — so one command
measures the tree that carries the twins and the tree that has collapsed them.

* **`turbo-storage`** — the non-test `quant_k_turbo*.rs` files under
  `crates/rmlx-kv-quant/src/storage`, `width_pair_key`. Reuses
  `storage_file_items` with a third glob.
* **`turbo-updates`** — the fns of the KV update file whose name carries the
  `tsym` token as a whole segment, `width_pair_key`, through `file_fn_items`
  with a third pattern. **Keyed on the symmetric token on purpose.** The same file carries two more turbo
  width pairs — `update_k8vturbo3` against `update_k8vturbo2` (72 matched
  lines) and their TCQ siblings (73) — which a `(turbo|tsym)` pattern would
  fold in, at 204 matched lines over 6 items. Those are a different family's
  twins, this issue does not collapse them, and with them in the population
  this counter could never reach the measured `0` the issue's proof asks for.
  Recorded here so the narrower pattern is a decision: **145 matched lines of
  turbo update duplication stay in that file after this collapse**, and
  widening the pattern is the follow-up that would report them.
* **`turbo-ssd`** — the turbo helper fns of `crates/rmlx-kv-ssd/src/block_io.rs`,
  `width_pair_key`, through the same `file_fn_items`.

**Why `turbo-ssd` is its own population and not a widened `turbo-storage`
glob.** Two reasons, and either alone is sufficient. `storage_file_items` reads
whole files under one directory of one crate, and these are fns inside one file
of another crate, so no glob reaches them without changing what the collector
means. And a `Population` prints the root it was measured over — that is the
rule the label carries — so folding them in would print the kv-quant storage
directory beside a number measured partly in kv-ssd.

**Its pattern carries no width digit**, which is the one subtle thing about it.
The four collapsed helpers keep their names without a width suffix
(`read_quant_k_turbo`, `read_tsym`, …). A digit-bearing pattern would match
nothing, report `unavailable` and exit 1 — the one answer a collapse must not
produce. `matched_lines_turbo_ssd_collapsed_zero` is the selftest case that
pins it, and the live tree now confirms it.

`scripts/debt_report_selftest.sh` carries 25 new cases, red both ways: the
planted figure and its item and pair counts for each population, the collapsed
tree at a measured `0` with the population still found for each, an empty
population, and a missing root — reason **and** exit code in every case. The
fixture plants the two `quant_k_turbo*.rs` files beside the rotor and iso ones
in the same directory, the `update_tsym*` pair beside `update_k8vturbo3` /
`update_k8vturbo2` in the same file, and the eight block_io helpers beside a
`read_quant_k` that carries no turbo token — so a widened glob, a widened
pattern or a shared root reads the wrong item count on one of them. The suite
is 114 cases.

## 6. Real-model rows — deferred

Not run in this chunk, and not run by the code chunk on its own initiative.
They belong to the integration run at the end of the integration branch, and
the decode-TPS row is owner-gated.

**The "before" arm is the commit before the collapse**, which the PR body
records by SHA.

What the run must show:

| # | Row | Gate |
|---|---|---|
| 1 | Temp-0 served digest at `--max-tokens 200`, `tsym3` and `tsym4` x {`gemma-4-e2b`, Ternary-Bonsai-8B} x {4k, 32k} — 8 cells, each exactly 200 ids | all 8 byte-identical before/after. **This row is weak evidence by construction** — §3 says no served capture reads a turbo store byte — so it catches a change that reached past the store, not a change inside it. A moved cell is a stop |
| 2 | `kv_cache_bytes` per cell | identical in all 8 cells, both arms. The collapse changes no store layout |
| 3 | The four asymmetric spellings | untouched Rust; deferred to the integration run on the base branch rather than re-baselined here |
| 4 | A `none` control in the same capture | present and unchanged, so the capture is shown to separate codecs rather than reporting one stream 8 times |
| 5 | Positive control | `tsym3` and `none` carry different digests **and** different `kv_cache_bytes` at every (model, context) pair. Without it a table of identical rows cannot be told from a table of nothing |
| 6 | `exit_code` and `n_ids` | `0` and exactly `200` in both arms, all 8 cells. A row missing either is a stop, not a difference |
| 7 | Binary digest | must **differ** between the arms. A run whose binary digest did not change did not test the change. Reported beside the diff, never folded into it |
| 8 | An **SSD-hydrated `tsym3` cache resuming decode on `Device::Gpu`** | must succeed. This is the served route that covers divergence 5, which §4's M15 shows no CPU test can. A served prefill does **not** cover it — §3's route note says why — so this row is a hydrate-and-resume, not a cold prompt. A collapse that lost the V-axis width rule fails here with `Error::Quant` on the first decode step that reaches the store |
| 9 | Decode TPS, `tsym3` and `tsym4` | within ±1 % of the recorded anchor — **owner-gated**, requested as one batched ask with the cell list it covers. No change is the expected result: the collapse adds no entry to a decode route |

Per-cell raw logs carry the absolute model-snapshot path and must never reach a
commit message, a PR body, an issue or any other public surface. Only a
commands file written with the model root elided may be quoted.

## 7. Removals, as landed

* `storage/quant_k_turbo3.rs` and `storage/quant_k_turbo4.rs` — both gone.
  `storage/quant_k_turbo.rs` holds `QuantKTurbo<BITS>`; the width-named
  spellings stay as type aliases, because `rmlx-kv-ssd` and
  `rmlx-kv-quant::storage::kv_storage` name them. The generic name is exported
  `pub` rather than `pub(crate)`, unlike the iso and rotor stores: the four
  collapsed SSD helpers live in another crate and are width-parametric, so they
  have to name it. The guard against a third width is
  `QuantKTurbo::WIDTH_IS_A_SHIPPED_ONE`, a const assert that **every method
  that reads or writes the store forces** — `new`, `from_cpu_blocks`, `reset`,
  `byte_size`, `truncate_to`, `try_deep_clone`, the two MSL width selectors and
  `NAME`, which `append` and `dequantize_choice` read. State the boundary that
  holds rather than a wider one: the nine fields are `pub` and the file allows
  `clippy::exhaustive_structs`, so a caller can write the struct literal at an
  unshipped width and it builds. That store is inert — no method of it
  compiles, and no `KvStorage` variant can hold one, because the two symmetric
  variants name the two aliases. Measured with a `QuantKTurbo::<7>` literal
  probe: the bare literal builds, and `reset`, `byte_size`, `truncate_to` and
  `try_deep_clone` each fail `E0080` naming the method they were instantiated
  from. `#[non_exhaustive]` is deliberately **not** used: the SSD hydrate pin
  destructures both the store and its block exhaustively, and that is the drift
  guard which catches a constructor that starts filling a field it used to
  leave at its default.
* `TURBO3_K_BITS` — gone, with its `storage/mod.rs` re-export.
  `TURBO_K3_BITS` and `TURBO_K4_BITS` replace it, one per shipped width, both
  read by the width guard and by `NAME`.
* `QuantKTurbo4`'s two inline struct literals in `kvcache/update.rs` — the
  `exit_prefill` arm and the decode arm — replaced by the one constructor.
  **The `exit_prefill` arms are rewritten, not deleted.** They execute for no
  turbo spelling today (`exit_prefill` returns at its
  `materialises_packed_store()` gate first), and they are kept on purpose as
  the re-enable path for a codec that grows a decode kernel over its own store.
  `warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`
  is the guard that holds an arm and its classification together, and it is the
  only thing that will ever execute those two sites.
  `storage/quant_planar_k.rs`'s comment "matching the `QuantKTurbo4`
  inline-literal signature" went with the rewrite.
* `KvCache::update_tsym3` and `update_tsym4` — gone. `KvCache::update_tsym`
  resolves the width from the `KvStorage` variant and enters `tsym_update`,
  the free generic body, the shape `update_iso_sym` already had. That body
  carries the V-axis width rule from §2(5).
* The four `block_io.rs` helper pairs — `k_turbo3_shape` / `k_turbo4_shape`,
  `write_quant_k_turbo3` / `_4`, `read_tsym3` / `read_tsym4`,
  `read_quant_k_turbo3` / `_4` — eight fns over four: `k_turbo_shape`,
  `write_quant_k_turbo` and `read_quant_k_turbo` are const-generic over the
  width, and `read_tsym` takes the width as a runtime `u8` because the two
  widths land in two `KvStorage` variants and only that fn knows which to
  build. It refuses any other width rather than falling through to one of the
  two it has.
* `read_quant_k_turbo3`'s doc comment "Note: tensors loaded from safetensors
  are pre-materialized byte-buffers; `.to_bytes()` is sufficient" — false as a
  justification once §2(4)'s decision keeps the call, and it is the comment
  that argued for the divergence.
* The "Mirrors `X` exactly" doc comments on both `write_quant_k_turbo3` and
  `read_quant_k_turbo3`. The second is **already** false — the bodies differ by
  `max_seq` and by `.eval()` — and neither has anything to mirror once there is
  one body.
* The two Rust MSL wrappers `turbo_k3_fused_qk_msl.rs` /
  `turbo_k4_fused_qk_msl.rs` (220 matched lines of 241 / 246) — **not
  collapsed.** The reason is structural rather than a preference. Each wrapper
  owns a `static AtomicU64` dispatch counter and a `static
  OnceLock<MetalKernel>` holding its own kernel, built from its own `.metal`
  source and header. A `static` declared inside a generic fn in Rust is **one**
  item shared across every instantiation, so a width-generic wrapper cannot
  keep the two counters and the two kernel singletons the tree reads by name in
  `kvcache/fused_qk_dispatch.rs` and in
  `crates/rmlx-kv-quant/tests/kv_decode_dtype_contract.rs`; it would have to
  move them out into per-width tables, which replaces two readable modules with
  one module plus two tables and deletes neither kernel nor counter. The
  width-selection layer the iso family gets from `isoquant_msl_dispatch` the
  turbo family already has, in `kvcache/fused_qk_dispatch.rs`'s `TURBO_K3_FN` /
  `TURBO_K4_FN`. **220 matched lines of wrapper duplication therefore stay**,
  and collapsing them is a kernel-shape decision that belongs with the `.metal`
  pair the issue puts out of scope.
* **The seven 3-bit-only tests — all seven ported to both widths.** Each is
  one generic body plus one `#[test]` per width, the iso precedent, so the cell
  count rose rather than fell.

  | Test | Disposition |
  |---|---|
  | `_new_shapes_correct` | ported; asserts `bits == BITS` instead of the constant |
  | `_roundtrip_cpu_single_step` | ported |
  | `_append_cpu_path` | ported |
  | `_reset_clears_seq` | ported; `reset()` is a both-width method now |
  | `_cosine_empirical_floor_head_dim_128` | ported, floor as a parameter. 3-bit keeps its 0.9807 gate unchanged; 4-bit measured 0.996401 at the same seed and shape, gated at 0.9954 |
  | `_from_cpu_blocks_max_seq_explicit` | ported; follows §2(2), so both widths assert the explicit window |
  | `_cpu_msl_parity` | ported, `#[ignore]` at both widths |

  The 4-bit file's three tests came across with it. Two of them were unprefixed
  (`cpu_two_append_multi_head_roundtrip`, `gpu_two_append_multi_head_roundtrip`)
  and are now `quant_k_turbo4_two_append_multi_head_roundtrip` and
  `quant_k_turbo4_gpu_two_append_multi_head_roundtrip`, because one merged file
  carrying both widths cannot leave one of them unnamed.

### Kept, and why

* **The two `.metal` kernels and their headers.** The issue puts them out of
  scope and §1's measurement supports it: 35 of 73 raw lines differ and the
  difference is the bit-unpacking algorithm, not a constant. The headers differ
  in the codebook array (`CB3[8]` against `CB4[16]`), which is a constant — but
  collapsing a header without its kernel buys nothing.
* **The `update_k8vturbo*` width pairs.** 145 matched lines of duplication in
  the same file, a different family, not this issue's. Recorded in §5.
* **The TCQ trellis.** §3 records that it constrains nothing and that two
  spellings are therefore the same codec. A real finding, and a separate
  change.
* **No `KvQuant` variant, no `KvStorage` variant, no CLI spelling, no layout
  tag, no census entry, no layout-key salt.** Every spelling that existed
  before exists after, with the same bytes and the same tokens.
* **`scripts/lib/debt_report.py`'s existing populations.** The three turbo ones
  sit beside them; the count stays ten.

## 8. What each chunk ran

The test chunk ran the two oracles, this doc, the populations and the first
mutation run — no engine code, no GPU test, no served capture, no performance
number.

| Gate | Result |
|---|---|
| `cargo test -p rmlx-kv-quant --lib turbo_store_bytes` | 8 passed, 0 failed |
| `cargo test -p rmlx-kv-ssd --lib block_io_turbo_hydrate` | 3 passed, 0 failed |
| `make ci` | green, `CI_EXIT=0`, run by the orchestrator in the main tree |
| `cargo fmt` | clean |
| `make lint` | clean, `-D warnings` across the workspace |
| `make check-no-inline-tests` | OK |
| `make check-gpu-tests-ignored` | OK, 356 files across 12 workspace members |
| `make check-doc-source-citations` | OK, 311 cited paths resolve |
| `make check-kv-codec-disposition` | OK — the verdict per codec is unchanged |
| `make debt-report-selftest` | OK, 114 cases |

The implementing chunk ran the collapse, the merged tests, the GPU pair, the
second mutation run (§4.1) and the after arm of §5.

| Gate | Result |
|---|---|
| `cargo test -p rmlx-kv-quant --lib turbo_store_bytes` | 8 passed, no assertion edited |
| `cargo test -p rmlx-kv-ssd --lib block_io_turbo_hydrate` | 3 passed, one expectation constant changed |
| `cargo test -p rmlx-kv-quant --lib iso_store_bytes` | 6 passed, untouched |
| `cargo test -p rmlx-kv-quant --lib rotor_store_bytes` | 5 passed, untouched |
| `cargo test -p rmlx-kv-quant` | 586 passed, 262 ignored |
| `cargo test -p rmlx-kv-ssd` | 121 passed, 12 ignored |
| `cargo fmt` | clean |
| `make lint` | clean, `-D warnings` across the workspace |
| `make check-no-inline-tests` | OK |
| `make check-gpu-tests-ignored` | OK, 356 files across 12 workspace members |
| `make check-gpu-tests-ignored-fixtures` | OK, 39 cases |
| `make check-doc-source-citations` | OK, 320 cited paths resolve |
| `make check-kv-codec-disposition` | OK, 28 codecs (17 inert) — unchanged |
| `make check-kv-layer-quants` | OK |
| `make check-metal-compiles` | SKIP — the Metal Toolchain component is not installed on this host |
| `make debt-report-selftest` | OK, 114 cases |
| `make gpu-runner-selftest` | OK |
| `make gpu-test CRATE=rmlx-kv-quant FILTER=turbo` | OK, 29 GPU tests passed across 1 workspace member |

`make ci` belongs to the orchestrator and `make ci-perf` to the integration
window, which also owns the served capture in §6.
