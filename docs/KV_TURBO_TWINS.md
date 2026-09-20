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

**Written before the collapse.** Everything below describes the tree as it
stands at the branch point. The test chunk wrote the oracle, this doc and the
first mutation run; it changed no engine code. Where a section states a
decision, that decision is owed to the implementing chunk, not already taken.

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

**Constraint.** One constructor, `QuantKTurbo::<BITS>::new(init_shape,
max_seq)`, used by all four construction sites. **Observable:** the CPU pin's
`store_after_chunk` at both widths — a constructor that leaves a field at the
wrong initial value moves it. Note that `crates/rmlx-kv-quant/src/storage/quant_planar_k.rs`
documents its own GPU-path init as "matching the `QuantKTurbo4` inline-literal
signature", so deleting the literal leaves that comment false; §7 lists it.

### (2) `from_cpu_blocks` and the hydrated window — the 3-bit form is the reference

`QuantKTurbo3::from_cpu_blocks(blocks, shape, bits, max_seq)` takes the window;
`QuantKTurbo4::from_cpu_blocks(blocks, shape, bits)` does not and hard-codes
`max_seq: 0`. `read_tsym4` already reads the geometry's `max_seq` — it needs it
for the `KvStorage` field — and simply does not forward it.

**This divergence changes an observable at one width today**, which the SSD pin
measures: after a spill and a hydrate of the same cache, the 3-bit K store
carries `4096` and the 4-bit one carries `0`.

**Decision: the 3-bit behaviour is the reference.** The collapsed
`from_cpu_blocks` takes `max_seq`, `read_quant_k_turbo4`'s caller forwards the
value it already has, and **the 4-bit hydrate pin moves from `0` to the written
window.** That is the collapse's one intended observable change, and
`tsym4_hydrate_restores_the_k_payload_but_not_the_window` is the cell that
records it.

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

**Constraint.** The safe form, once. **Observable:** not the CPU pin — that
branch needs a `Device::Gpu` append after a hydrate, which no test here drives.
What holds it is that the two forms produce the same bytes by construction
(`f32::to_le_bytes` is the definition of the little-endian layout the cast
reinterprets), plus `make gpu-test`. The collapse removes one `unsafe` block
from the crate and adds none.

### (4) `.eval()` before `to_bytes()` — the 4-bit form is the reference

`read_quant_k_turbo4` calls `.eval()` on both loaded tensors;
`read_quant_k_turbo3` does not, and its doc comment argues the call is
unnecessary.

**Measured: neither form changes a byte.** §4's M13 removes the call from the
4-bit path and M16 adds it to the 3-bit path, and both leave all three SSD
cells green. This is a convention decision, not a behaviour one, and the doc
says so rather than dressing it up.

**Decision: keep the call.** Every other reader in the same file —
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
| `v_f32` materialised | always | only when the caller's device is CPU |

On a CPU drive the two are the same routing. On a GPU drive they are not, and
the 3-bit forcing is **load-bearing, not a perf preference**: `QuantV::append`
enters its GPU branch on the device alone and then refuses `bits != 4`. A
collapsed body that adopted the 4-bit shape would hand a 3-bit `QuantV` the GPU
device and get `Error::Quant` on every prefill chunk. A collapsed body that
adopted the 3-bit shape would take the 4-bit V axis off the GPU, which changes
its store and its speed.

**Constraint.** The collapsed body resolves the V device from the width: the
caller's `device` at `BITS == 4`, `Device::Cpu` at `BITS == 3`, and `v_f32` is
materialised exactly when that resolved device is CPU — which is the one rule
that reproduces both bodies. **Observable: none on CPU.** §4's M15 drops the
forcing and every cell stays green. The gates are `make gpu-test` and the
served capture in §6.

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

The third test asserts the two widths answer `max_seq` differently today, so
that the collapse making them agree is a deliberate, visible move rather than
two cells quietly converging.

### What the CPU oracle cannot see, and the GPU tests the collapse owes

| Unseen | GPU test owed | Census disposition |
|---|---|---|
| The V-axis device split (divergence 5) at `Device::Gpu` — the one defect a blind merge introduces | a `Device::Gpu` prefill at `tsym3` and at `tsym4`, asserting the append succeeds and the store bytes match the CPU cells' payload. `tsym3` on the 4-bit routing returns `Error::Quant`, so this is a hard red, not a tolerance | derived from a run: `scripts/gpu_validation_census.txt` pins accepted invalid accesses, so a test producing no shader-validation hit carries no entry. Re-derive if a cell gains a load |
| The GPU `append` path — buffer allocation, paged growth, the MSL encode dispatch at both widths | already covered by `storage::quant_k_turbo3_tests::quant_k_turbo3_gpu_two_append_multi_head_roundtrip` and its 4-bit sibling, both `#[ignore]`-gated | no new entry |
| The hydrated-init upload branch, where divergence 3 lives | a `Device::Gpu` append on a store built by `from_cpu_blocks`, at both widths, asserting the uploaded scales match the CPU blocks | derived from a run |
| CPU/MSL parity of the K codec | `quant_k_turbo3_cpu_msl_parity` exists at 3-bit and has **no 4-bit counterpart** — one of the seven 3-bit-only tests. §7 says what it owes | derived from a run |
| The fused-QK decode kernels | `turbo_k3_fused_qk_msl_tests` / `turbo_k4_fused_qk_msl_tests`, unchanged by the collapse | no |

## 4. Mutations

Every mutation below was applied to the tree by hand, run, and reverted from a
snapshot whose sha256 of the **working** file was compared before and after —
`git checkout --` is not used; it would revert uncommitted work. Each row names
the assertion that caught it, not just that something failed. Thirteen of
sixteen red.

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

M4, M5, M6, M7, M8b and M12 are each caught by exactly one assertion, and M5,
M6, M7 and M8b each by exactly one **column** — truncate, rows, resident_bytes
and store_after_decode. With M1 and M3 on the chunk column, **all five columns
are load-bearing** rather than redundant. M6 fires at one shape only, which is
what says both shapes are.

**The width guard is the other half of M1.** The collapsed type should refuse
`QuantKTurbo<5>` at monomorphisation the way the iso stores do — a named
associated const read by `new`, by `from_cpu_blocks` and by the `Debug` impl,
so no birth path can miss it. That is owed to the implementing chunk; this
tree has two separate types and no such guard to mutate.

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

Two more things this run cannot see, stated so the 13-of-16 is not read as
wider than it is. No row mutates the GPU append path, because the pin never
dispatches a kernel. And no row mutates the `.metal` kernels, which are out of
scope and which no CPU test compiles.

## 5. Duplication figure — the "before" arm

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

**After the collapse all three must read a measured `0` with the population
still found.** A deleted population prints `unavailable` and exits 1, and a
population that quietly resolved to nothing would be indistinguishable from a
collapsed one if the module let it print `0`.

### The population design

Each is a glob or a name pattern plus a root, parameterised at the registration
site through `functools.partial` — never a file or fn list — so one command
measures the tree that carries the twins and the tree that has collapsed them.

* **`turbo-storage`** — the non-test `quant_k_turbo*.rs` files under
  `crates/rmlx-kv-quant/src/storage`, `width_pair_key`. Reuses
  `storage_file_items` with a third glob.
* **`turbo-updates`** — the `update_tsym*` fns of the KV update file,
  `width_pair_key`, through `file_fn_items` with a third pattern. **Anchored on
  the symmetric entries on purpose.** The same file carries two more turbo
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
After the collapse those four helpers keep their names without a width suffix
(`read_quant_k_turbo`, `read_tsym`, …). A digit-bearing pattern would then
match nothing, report `unavailable` and exit 1 — the one answer a collapse must
not produce. `matched_lines_turbo_ssd_collapsed_zero` is the selftest case that
pins it.

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
| 8 | A `Device::Gpu` prefill at `tsym3` | must succeed. This is the one row that covers divergence 5, which §4's M15 shows no CPU test can. A collapse that lost the V-axis width rule fails here with `Error::Quant`, on the first chunk |
| 9 | Decode TPS, `tsym3` and `tsym4` | within ±1 % of the recorded anchor — **owner-gated**, requested as one batched ask with the cell list it covers. No change is the expected result: the collapse adds no entry to a decode route |

Per-cell raw logs carry the absolute model-snapshot path and must never reach a
commit message, a PR body, an issue or any other public surface. Only a
commands file written with the model root elided may be quoted.

## 7. Removals the collapse owes

Written before the collapse; the implementing chunk deletes these and the PR
body lists them again with the net line count.

* `storage/quant_k_turbo3.rs` and `storage/quant_k_turbo4.rs` — one becomes
  `QuantKTurbo<BITS>` and the other goes. The width-named spellings stay as
  type aliases, because `rmlx-kv-ssd` and `rmlx-kv-quant::storage::kv_storage`
  name them.
* `TURBO3_K_BITS` — a constant with one width's name on it, exported from
  `storage/mod.rs`. It has no 4-bit counterpart. Either it becomes
  `QuantKTurbo::<BITS>::BITS` or it goes; `storage/mod.rs`'s re-export line
  goes with it either way.
* `QuantKTurbo4`'s two inline struct literals in `kvcache/update.rs` — the
  `exit_prefill` arm and the decode arm — replaced by the one constructor.
  `storage/quant_planar_k.rs`'s comment "matching the `QuantKTurbo4`
  inline-literal signature" is made false by that and must go or be reworded.
* `KvCache::update_tsym3` and `update_tsym4` — two entries over one,
  resolving the width from the `KvStorage` variant, the shape
  `update_rotor_v` and `update_iso_v` already have. Its body carries the
  V-axis width rule from §2(5).
* The four `block_io.rs` helper pairs — `k_turbo3_shape` / `k_turbo4_shape`,
  `write_quant_k_turbo3` / `_4`, `read_tsym3` / `read_tsym4`,
  `read_quant_k_turbo3` / `_4` — eight fns over four. The write pair's bodies
  are byte-identical, so that one is a pure deletion.
* `read_quant_k_turbo3`'s doc comment "Note: tensors loaded from safetensors
  are pre-materialized byte-buffers; `.to_bytes()` is sufficient" — false as a
  justification once §2(4)'s decision keeps the call, and it is the comment
  that argued for the divergence.
* The "Mirrors `X` exactly" doc comments on both `write_quant_k_turbo3` and
  `read_quant_k_turbo3`. The second is **already** false — the bodies differ by
  `max_seq` and by `.eval()` — and neither has anything to mirror once there is
  one body.
* The two Rust MSL wrappers `turbo_k3_fused_qk_msl.rs` /
  `turbo_k4_fused_qk_msl.rs` (220 matched lines of 241 / 246) — **the issue
  lists them and this doc does not decide them.** They dispatch the two
  `.metal` kernels, which are out of scope, so collapsing the wrappers means
  deciding what a width-generic wrapper over two genuinely different kernels
  looks like. If the implementing chunk collapses them it says so; if it does
  not, it says that, with the figure.
* **The seven 3-bit-only tests.** `quant_k_turbo3_new_shapes_correct`,
  `_roundtrip_cpu_single_step`, `_append_cpu_path`, `_reset_clears_seq`,
  `_cosine_empirical_floor_head_dim_128`, `_from_cpu_blocks_max_seq_explicit`
  and `_cpu_msl_parity` have no 4-bit counterpart. The iso precedent is that
  each case becomes one generic body plus one `#[test]` per width, so the cell
  count does not fall. Two of these need a decision rather than a translation:
  `_cosine_empirical_floor_head_dim_128` carries a 3-bit quality floor that is
  not the 4-bit floor, so it takes a per-width bound or stays 3-bit-only; and
  `_from_cpu_blocks_max_seq_explicit` is the test for divergence 2 and must
  follow whichever way §2(2) is resolved. **The PR states, for each of the
  seven, ported-to-both-widths or stays-3-bit-only.**

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
  are added beside them.

## 8. What this chunk ran

The test chunk ran the two oracles, this doc, the populations and the first
mutation run — no engine code, no GPU test, no served capture, no performance
number.

| Gate | Result |
|---|---|
| `cargo test -p rmlx-kv-quant --lib turbo_store_bytes` | 8 passed, 0 failed |
| `cargo test -p rmlx-kv-ssd --lib block_io_turbo_hydrate` | 3 passed, 0 failed |
| `cargo fmt` | clean |
| `make lint` | clean, `-D warnings` across the workspace |
| `make check-no-inline-tests` | OK |
| `make check-gpu-tests-ignored` | OK, 356 files across 12 workspace members |
| `make check-doc-source-citations` | OK, 311 cited paths resolve |
| `make check-kv-codec-disposition` | OK — the verdict per codec is unchanged |
| `make debt-report-selftest` | OK, 114 cases |

`make ci` belongs to the orchestrator and `make ci-perf` to the integration
window, which also owns the served capture in §6.
