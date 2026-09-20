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

### The one live behaviour difference

It is on the **non-fused** route, which runs on a `Device::Cpu` drive, at
`q_seq > 1`, and whenever `iso_flash_shape_ok` rejects — a batched cache, or a
`head_dim` that is not a power of two — and the step falls through to `update()`:

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
| `QuantIsoV::append_gpu`'s mirror write | none, and §4's M14 is why that is safe to say: the production value is the constant `GPU_RESIDENT_ISO_PRODUCTION`, `the_production_gpu_resident_iso_mirror_is_off` asserts on that constant, and flipping it turns the assertion red. The test dispatches nothing | no |
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

| # | Edit | Caught by |
|---|---|---|
| M1 | `quant_iso_v4.rs` `append` — encode at `3` instead of `ISO4_BITS`, i.e. a generic instantiated at the wrong width | 4 of the 6 tests: the pin, the geometry, the twins control and the reproducibility drive. First line `iso4 decode: isoquant: code plane holds 288 words for 24 rows, which need 384` |
| M2 | `quant_iso_v4.rs` `append` — drop the last code word (`codes.pop()`) | the same 4. First line `code plane holds 383 words for 24 rows, which need 384` |
| M3 | `quant_iso_v.rs` — `ISO3_GROUP_SIZE` 4 -> 8, moving the quaternion group boundary | the pin (`iso3 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`) **and** the geometry (scale count 384 against 768) |
| M4 | `isoquant.rs` — `FIXED_QUAT` replaced by `[0.5, 0.5, 0.5, 0.5]`, the codec's one rotation constant | the pin **only** (`iso3 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`); geometry stays green, because the code plane is the same length |
| M5 | `update_iso4_sym` — the K store appends `v_f32`, so the symmetric entry writes V's data on both axes | the pin **only** (`iso4_sym @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`) |
| M6 | `update_iso_k_only_4` — the K store appends `new_v`, so the K-only entry stores the V stream | the pin **only** (`k_iso4 @ kv_h=1 head_dim=128: packed store bytes after the bulk append moved`) |
| M7 | `quant_iso_v.rs` `truncate_to` — `(n - 1).max(0)` | the pin's **truncate** column **only** (`iso3 @ kv_h=1 head_dim=128: packed store bytes after truncate_to moved`) |
| M8 | `quant_iso_v4.rs` `dequant` — drop `transpose_chunked_seq_heads`; the store is untouched, only the rows the attention receives move | the pin's **rows** column only, at **shape B only** (`iso4 @ kv_h=4 head_dim=96: the K/V rows attention receives moved`). This is the 4-bit CPU decode, and it is the reference the future GPU entry is compared against |
| M9 | `quant.rs` — `KvQuant::Iso4` removed from `ALL_KV_QUANTS` | the census (`iso spelling census moved … ["iso3", "iso3_sym", "iso4_sym", "k_iso3", "k_iso4"]`, left 5 right 6) |
| M10 | `quant.rs` — `KvQuant::Iso3` listed twice in `ALL_KV_QUANTS`, standing in for a seventh iso spelling | the census (left 7 right 6). It exercises the count anchor only: a genuinely new variant would also miss `pin_for` and turn the pin test red, which this stand-in cannot show without wiring a variant through every exhaustive match |
| M11 | `quant_iso_v4.rs` `gpu_append` — `n_groups + 1` into `append_encoded` | **uncaught** — 6 passed, exit 0 |
| M12 | `quant_iso_v4.rs` `byte_size` — drop the CPU-blocks term | the pin's **resident_bytes** column only (`iso4 @ kv_h=1 head_dim=128: resident_bytes moved`, left 3564 right 22680) |
| M13 | `quant_iso_k4.rs` `append` — coalesce a one-token block into its predecessor after the push, so the payload is identical and only the block boundaries move | the pin's **store_after_decode** column only (`iso4_sym @ kv_h=1 head_dim=128: packed store bytes after 3 decode steps moved`). The chunk column and the geometry stay green: the first append has no predecessor to merge into, and the summed plane lengths do not change |
| M14 | `lib.rs` — `GPU_RESIDENT_ISO_PRODUCTION` flipped to `true` | `the_production_gpu_resident_iso_mirror_is_off` (`the GPU-resident iso V mirror is on in production`). Its control is below |

M4, M5, M6, M7, M8, M12 and M13 are each caught by exactly one assertion, and
M7, M8, M12 and M13 each by exactly one **column** — truncate, rows,
resident_bytes and store_after_decode respectively. With M1 and M2 on the chunk
column, all five columns are load-bearing rather than redundant. M8 fires at one
shape only, which is why both shapes are.

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

**M11.** `QuantIsoV4::gpu_append` is handed one more quaternion group than the
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

So the honest statement of this oracle's power: it covers the CPU encode, the
CPU decode, the block bookkeeping, `truncate_to` and residency at both widths
and both shapes, and it covers **nothing on the ring**. `make gpu-test
HALF=codec` is the only gate over the ring, and the collapse's reviewer must
read its result rather than this file's.


## 5. Duplication figure

`scripts/debt_report.sh --matched-lines` carries no iso population today. Its
five populations are `drivers`, `impls`, `rotor-storage`, `rotor-updates` and
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
  `quant_iso_v`).
* **`iso-updates`** — the `update_iso*` fns of
  `crates/rmlx-kv-quant/src/kvcache/update.rs`, paired by the same
  digit-stripped-name rule.

Three things the code chunk should not get wrong, because the record already
carries what is needed.

1. **Parameterise at the registration site, not on the record.** `Population`
   already carries `collect` and `root`; the collector is a plain
   `Callable[[Path], list[FnInfo]]`. So `rotor_storage_items` generalises to a
   `storage_file_items(root, *, glob)` and both entries register
   `functools.partial(storage_file_items, glob=...)`; `rotor_update_items`
   generalises to a `file_fn_items(root, *, file, prefix)` the same way. Adding
   a `glob` or `prefix` **field** to `Population` would make `drivers`, `impls`
   and `ssd-hydrate` carry a field none of them reads.
2. **`iso-storage` needs one new constant, not two.** `ROTOR_STORAGE_DIR` is
   already `crates/rmlx-kv-quant/src/storage`, which is the iso stores' root
   too; only the glob is new. (The name wants widening to `KV_STORAGE_DIR` in
   the same change, since it will then have two callers and name neither.)
3. **No CLI change is owed.** `--matched-lines`'s `choices` are
   `sorted(MATCHED_LINES_POPULATIONS)`, so registering the two entries is what
   adds them to the flag.

Registering two more copies of the rotor collectors would plant the twin the
campaign exists to remove.

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
  unified V appender takes the feed its caller states. It has a live test
  caller: `kvcache/ring_feed_routing_tests.rs` imports it and asserts on it
  twice, so that cell becomes a cell over the feed the caller states rather
  than over a width-named constant.
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

## 8. What the test chunk ran

The oracle, the doc and the mutation run. No engine code, no GPU test, no
served capture, no performance number.

| Gate | Result |
|---|---|
| `cargo test -p rmlx-kv-quant --lib iso_store_bytes` | 6 passed, 0 failed |
| `cargo test -p rmlx-kv-quant --lib rotor_store_bytes` | 5 passed, 0 failed — **no rotor pin was re-baselined**; the rotor file's `PINS` table is untouched |
| `cargo test -p rmlx-kv-quant` | 572 passed, 0 failed, 257 ignored. The six added cells are the whole delta |
| `cargo fmt` | clean |
| `make lint` | clean, `-D warnings` across the workspace |
| `make check-no-inline-tests` | OK |
| `make check-gpu-tests-ignored` | OK, 356 files across 12 workspace members |
| `make check-doc-source-citations` | OK, 310 cited paths resolve |
| `make check-kv-codec-disposition` | OK, 28 codecs classified, 17 inert |
| `make check-kv-layer-quants` | OK |

`make ci` and `make ci-perf` are not run here. `make ci-perf` needs an idle GPU
and belongs to the integration window with the two owed GPU tests and the
served capture.

### The helper extraction

The store-byte serialisation the rotor pin carried — `fnv1a64`, `StoreBytes`,
`array_bytes`, `dtype_tag`, `f32_arr` and `push_quant_k` — moved to
`crates/rmlx-kv-quant/src/test_utils.rs` and is shared by both pin files.
Copying it into a second file would have planted a twin in the change whose
purpose is removing one. `rotor_store_bytes_tests.rs` lost those six items and
gained one `use`; its pins, its drive and its assertions are unchanged, which
the 5-passed run above is what says.
