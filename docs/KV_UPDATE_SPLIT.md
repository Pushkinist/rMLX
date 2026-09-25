# Splitting the KV update path

`crates/rmlx-kv-quant/src/kvcache/update.rs` held every codec's update path in
one file. This document is the plan to split it, the record of what was
measured before the split started, and the record of each chunk as it lands.
Chunks (a), (b1) and (b2) have landed: the per-family decode bodies, then the
per-family prefill bulk-encode bodies, now sit in
`kvcache/update_{rotor,iso,turbo,affine,planar,paged,mixed}.rs` and
`update.rs` is the dispatcher over them; chunk (b2) then wrote one body per
store shape instead of one per spelling. Every figure below that says "today"
was measured before chunk (a); §4 carries the after.

Method comes from the three twin-collapse documents and is not restated here:
[`KV_ROTOR_TWINS.md`](KV_ROTOR_TWINS.md), [`KV_ISO_TWINS.md`](KV_ISO_TWINS.md),
[`KV_TURBO_TWINS.md`](KV_TURBO_TWINS.md). What is new here is the oracle over
**every** spelling, the structural metric the split is judged on, and the
constraints the implementing chunks work under.

---

## 1. The premise, re-measured

Every figure below is derived by a script, never counted by hand:

```bash
make kv-update-census                     # all three figures
python3 scripts/kv_update_census.py match-sites
python3 scripts/kv_update_census.py refs --file crates/rmlx-kv-ssd/src/block_io.rs
python3 scripts/lib/debt_report.py --matched-lines update-bodies
```

The "before the collapses" column is the same command run against a worktree at
the commit before the rotor, iso and turbo twin collapses landed.

| Row | Claimed | Before the collapses | Today | Verdict |
|---|---|---|---|---|
| `update.rs` lines | 8105 | 8116 | 7151 | Confirmed. The collapses took 968 lines; the interim `LOC-exempt` marker added 3. |
| Lines in `update_`-prefixed fn bodies | 2898 | 2539 over 33 bodies | 1440 over 25 bodies | Moved. The claimed figure does not reproduce under the body rule used here (brace to brace, signature excluded). |
| `match` blocks of 26–27 arms in `update.rs` | 6 | 6 | 6 | Confirmed, and none of the six moved. |
| The same in `storage/kv_storage.rs` | 3 | 4 | 4 | Confirmed for `KvStorage` (3); a fourth site there matches over `KvQuant`. |
| The same in `kvcache/helpers.rs` | 2, of 23 arms | 3, of 9 / 8 / 27 arms | 3 | Does not reproduce as stated. Three sites name all 27 variants; two of them do it with or-patterns, so their **arm** count is 9 and 8 while their **variant** count is 27. |
| Variant references in `rmlx-kv-ssd/src/block_io.rs` | 60 | 55 | 55 | Moved. The rule here counts `KvStorage::<Variant>` occurrences; it finds 55 over 27 distinct variants, plus 2 `KvQuant::` ones. |
| `KvStorage` variants sharing the store-slot shape | 24 of 27 | 24 of 27 | 24 of 27 | Confirmed exactly. |
| `update_k8vturbo3` against `update_k8v4` | 52 of 100 lines | — | 60 matched lines, bodies of 78 and 72 | Confirmed in direction, larger in size. Measured with the `debt-report` rule (digit-folded `difflib` matching blocks). |

Two rows moved because the collapses moved them, and one row — the
`helpers.rs` one — never held as stated.

**What the body row counts.** Every fn of the update file whose name starts
`update_`, and nothing else. The prefix admits the shared entries the dispatch
reaches as well as the per-variant bodies — `update_and_sdpa_k8v4_flash` and
its inner and no-lock siblings, `update_decode_fp16` and its two variants,
`update_prefill_raw` — about 470 of the 1440 lines. They are counted on
purpose: any rule that dropped them would be a hand-drawn boundary over which
body is "per-variant" enough, and the shared entries are part of what the
restructure has to shrink. The label says what the prefix finds.

### Variant shapes

27 variants. 24 carry the store slots and `max_seq` that one update body can
serve:

| Shape | Count | Variants |
|---|---|---|
| `k`, `v`, `max_seq` | 16 | `K8V4`, `K8V8`, `K8VTurbo3`, `K8VTurbo3Tcq`, `K8VTurbo2`, `K8VTurbo2Tcq`, `TurboSym3`, `TurboSym4`, `IsoV3`, `IsoV4`, `IsoSym3`, `IsoSym4`, `RotorV3`, `RotorV4`, `RotorSym3`, `RotorSym4` |
| The same plus scalar knobs | 3 | `Planar` (`bits`), `RotorKAsym3` / `RotorKAsym4` (`v_bits`, `v_group_size`) |
| `k`, `max_seq` — V is bf16 on the parent cache | 5 | `PlanarK`, `IsoKOnly3`, `IsoKOnly4`, `RotorKOnly3`, `RotorKOnly4` |

Three do not fit, and each for its own reason:

* **`None { max_seq }`** holds no store at all. Its buffers are
  `KvCache::decode_fp16_k` / `decode_fp16_v` on the parent cache.
* **`Mixed { state, max_seq }`** holds `MixedKvState`, a pair of `mx.quantize`
  3-tuples plus a rotation matrix. It is not a K slot and a V slot, and the
  cache **rejects `update` outright** for it: the only entry that appends to it
  is `update_and_sdpa`.
* **`Paged { quant, k, v_k8, v_planar, max_seq }`** holds a block table, two
  V slots and the quant it was built for. It is selected by a process-global
  the CLI latches, not by a spelling.

The scalar knobs the classifier admits beside the slots are `bits`, `v_bits`
and `v_group_size`. A field outside that set holds state, not a setting, and
takes the variant out of the shared shape — asserted by a planted fixture in
the census selftest, so the classifier cannot quietly widen.

---

## 2. The oracle

This restructure is pure: every `--kv-quant` spelling must write the same bytes
after it as before it. The observable is at the deepest seam a caller reaches
on CPU, and it is five columns per spelling per shape:

| Column | What moves it |
|---|---|
| Store bytes after the bulk append | any change to what the chunk encode writes |
| Store bytes after three decode steps | any change to the per-step append |
| Store bytes after a truncate into the chunk | either half of the truncate plan |
| The K and V rows the attention receives | any change to what the store hands back |
| `resident_bytes()` | any change to what the cell holds |

`crates/rmlx-kv-quant/src/kvcache/store_bytes_tests.rs` is the one file that
pins them.

Two spellings hold the store column constant across all three of its
readings, and the pin table shows the same digest three times for them. That
is the cell, not a gap:

* **`none`** holds no store at all. Its buffers live on the parent `KvCache`,
  so the column is one tag and the rows and `resident_bytes()` carry the cell.
* **`planar_k`** writes its K store once, at the chunk append. The decode
  steps route through the warm-TTFT bf16 K seed and the truncate plan does not
  reach a 24-position store, so neither moves a byte. The bulk-append reading
  carries the store claim; the rows and `resident_bytes()` carry the rest.

### Coverage audit

Before this work, three family files pinned 20 of the 28 spellings in
`ALL_KV_QUANTS`, plus two extra rotor asym parameterisations — 44 cells.

| Spelling | Pinned before | Pinned now |
|---|---|---|
| `rotor3`, `rotor4`, `rotor3_sym`, `rotor4_sym`, `k_rotor3`, `k_rotor4`, `rotor_k_3_asym_v4_g64`, `rotor_k_4_asym_v4_g64` | yes | yes |
| `rotor_k_3_asym_v2_g64`, `rotor_k_4_asym_v2_g64` (beyond `ALL_KV_QUANTS`) | yes | yes |
| `iso3`, `iso4`, `iso3_sym`, `iso4_sym`, `k_iso3`, `k_iso4` | yes | yes |
| `k8vturbo3`, `k8vturbo3tcq`, `k8vturbo2`, `k8vturbo2tcq`, `tsym3`, `tsym4` | yes | yes |
| `none`, `k8v4`, `k8v8`, `planar`, `planar3`, `planar_k` | **no** | yes |
| `mixed_k8g64_v4g64`, `rot_k_v8g64` | **no** | yes |

60 cells now. The eight newly pinned spellings are the oldest bodies in the
update path, and the ones a restructure is most likely to move.

### One oracle, not four

The three family files held one machinery three times. They differed in the
spelling filter, the family name and the per-variant field list; the drive, the
five columns, the census shape and the reproducibility check were identical.
That is the twin shape rule 6 names, so the machinery is now in one file and
sweeps `ALL_KV_QUANTS` rather than a `Display` filter.

**No pin value was re-baselined.** All 44 pre-existing rows are reproduced by
the merged oracle byte for byte. The proof is a diff of the pin literals: parse
the `PINS` tables of the three deleted family tables and of the merged one, key
each row by `(spelling, kv_h, head_dim)`, and compare the five values. The
result was 44 rows present, 0 values moved, 16 rows added.

The three family files keep only what is about their family: the
geometry-follows-bit-width tests, their own width-twin pair lists, the TCQ byte
identity, the K-side scope anchor, and the GPU-resident mirror check. The
width-twin control itself is one fn in the oracle — `assert_width_twins_differ`
— that the three call with their pairs; the three
`*_store_geometry_follows_the_codec_bit_width` tests stay apart, because each
restates a different codec's published layout arithmetic and one body cannot
carry three.

Counted the same way on both sides (`wc -l`, whole file, doc comments
included): test code across the files went from 2611 lines over three files to
2762 over four, while pinned cells went from 44 to 60 store-bytes cells plus 20
prefill cells.

### What the oracle cannot drive, and what covers it

* **`KvStorage::Paged`.** No spelling builds it in a test process: the routing
  reads a process-global the CLI latches once, and a test that set it could not
  unset it for the rest of the binary. `no_spelling_builds_the_paged_storage_in_this_process`
  asserts that, so a change which starts routing a spelling there fails rather
  than leaving a cell unpinned. The paged path's own suite is `crate::paged`.
* **The mixed pair's second shape.** `mixed_k8g64_v4g64` and `rot_k_v8g64`
  group by 64 and `MixedKvState::init_quant` rejects `head_dim = 96`. Their
  second shape is `(4, 128)` — `kv_h > 1` kept, the non-power-of-two head
  dimension given up. The choice is derived from the spelling's own group
  sizes, not from a list.
* **The mixed pair's rows column** is the attention output, not the K and V
  rows, because the cache refuses `update` for them. It moves on the same
  defects, one step further downstream.
* **The `exit_prefill` arms of the 18 decode-inert spellings.** The gate
  returns before them, so no CPU route runs them at all. They are kept as the
  re-enable path for a codec that grows a decode kernel over its own store,
  and the guard named above is what holds them to the predicate. Chunk (b1)
  moved them into the family files with the live ones; being unreachable is
  not a reason to leave a body in the dispatch file.
* **Every GPU path** — MSL encode dispatch, resident rings, fused flash-decode
  arms, the hydrated-init upload branch. `make gpu-test` is the gate.

### Two populations, two drives

`exit_prefill` returns at its `materialises_packed_store()` gate, before every
bulk-encode arm, and clears whatever payload the cache arrived with. That gate
splits the 28 spellings in two, and each half needs a drive of its own. The
split is derived from the predicate and pinned in the oracle
(`the_two_populations_partition_every_spelling`), so a codec crossing it turns
a cell red instead of quietly changing what the real-model run is worth.

**18 spellings report `false`.** A served prefill writes them no packed store
and decode runs off the bf16 seed, so a served temp-0 capture against an empty
store and against a correct store emits the same tokens. **Their store bytes
are observable in the oracle and nowhere else.** Their `exit_prefill` arms are
dead code under every CPU route — the gate returns first — and that the gate
keeps them empty is already pinned by
`warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`,
which asserts a zero store after prefill, a zero store after one decode step,
and a residency equal to the bf16 baseline, for all 18. The five columns for
these spellings come from the decode-dispatch drive, which is the one CPU route
on which they write a store at all.

**10 report `true`** — `mixed_k8g64_v4g64`, `rot_k_v8g64`, `iso3_sym`,
`iso4_sym`, `k_iso3`, `k_iso4`, `rotor3_sym`, `rotor4_sym`, `k_rotor3`,
`k_rotor4`. Their `exit_prefill` arms run, and the bytes those arms bulk-encode
are what a served decode then reads. The decode-dispatch drive never reaches
those arms, and the guard above reads only whether the store is non-empty, so
before this work **nothing in the tree read a byte any live `exit_prefill` arm
wrote**. Measured: scaling `k_f32` by 1.01 inside the `Rotor3Sym` arm left the
whole crate green, 576 tests passed. A second drive closes it —
`drive_prefill` brackets the chunk with `enter_prefill` / `exit_prefill` and
pins, per spelling per shape, the store the arm wrote, `resident_bytes()`, and
the rows of the first decode step after the bracket. 20 cells, the population
derived from the predicate rather than listed. The same mutation now turns
exactly one test red, `rotor3_sym @ kv_h=1 head_dim=128`.

### The oracle, falsified

Seven engine mutations, spanning every observed column and four families. The
control run, with no mutation, is green.

| Mutation | First cell red | Column |
|---|---|---|
| `resident_bytes` over-counts by one byte | `none @ kv_h=1 head_dim=128` | `resident_bytes` |
| `KvStorage::truncate_to` keeps one position too few | `k8v4 @ kv_h=1 head_dim=128` | store bytes after truncate |
| one planar rotation angle off by 0.001 rad | `planar @ kv_h=1 head_dim=128` | store bytes after the bulk append |
| one iso quaternion component off in the seventh digit | `iso3 @ kv_h=1 head_dim=128` | store bytes after the bulk append |
| one TurboQuant 2-bit centroid off by 0.01 | `k8vturbo2 @ kv_h=1 head_dim=128` | store bytes after the bulk append |
| per-block planar V scales scaled by 1.01 in `QuantPlanarV::append` | `planar @ kv_h=1 head_dim=128` | store bytes after the bulk append |
| `k_f32` scaled by 1.01 in the `Rotor3Sym` `exit_prefill` arm | `rotor3_sym @ kv_h=1 head_dim=128` | store bytes after `exit_prefill` |

The rotation-angle mutation is **not** the coverage argument. Measured, it
turns five pre-existing tests red on its own:
`planarquant::tests::cb4_rotation_codebook_bit_exact`,
`planar_flash_decode_msl::tests::hdr_probe_snapshot_matches_builder`, and
`hdr_probe_snapshots_match_builders` in `planar_fused_qk_msl`,
`planarquant_msl` and `sparse_attn`. It moves the codebook the probe snapshots
are built from, so it never needed a store-bytes pin to be caught.
(`rate_distortion_tests::pinned_budgets_sit_one_slack_above_the_measurement`
stays green at 0.001 rad.)

The mutation that does carry the coverage argument is the per-block scale one:
multiplying the per-block scales `planar_quantize` returns by 1.01 inside
`QuantPlanarV::append`. It never touches the rotation codebook, so no probe
snapshot and no codebook test can see it. Measured, it turns **one** test red —
this oracle — and before this work it would have turned none.

The last row is the second drive's: it moves a byte no other column in this
table can reach. Every mutation here was applied to a snapshot, run, and
restored by file copy with a digest check — never by `git checkout`.

---

## 3. The structural metric

The issue's proof asks for the count of `match` sites a new codec must touch,
before and after, with a target of 3 or fewer.

Producer: `scripts/kv_update_census.py match-sites`, indexed in
[`../scripts/INDEX.md`](../scripts/INDEX.md), runnable as
`make kv-update-census`.

**The rule.** A site is a `match` expression whose arm patterns name at least
half the enum's variants. Below half, a `match` is a case analysis over a
subset; at half or more it enumerates the codec surface, which is the
population the restructure has to shrink. The bar is derived from the enum on
every run (14 of 27 for `KvStorage`, 14 of 28 for `KvQuant`), so a codec added
or retired moves it. A second figure counts the sites with no catch-all arm:
those are the ones `wildcard_enum_match_arm = deny` forces a new codec to
touch.

**Before, per file** (26 sites, all 26 forcing):

| File | Sites |
|---|---|
| `crates/rmlx-kv-quant/src/quant.rs` | 9 |
| `crates/rmlx-kv-quant/src/kvcache/update.rs` | 6 |
| `crates/rmlx-kv-quant/src/storage/kv_storage.rs` | 4 |
| `crates/rmlx-kv-quant/src/kvcache/helpers.rs` | 3 |
| `crates/rmlx-kv-ssd/src/block_io.rs` | 1 |
| `crates/rmlx-models/src/kv_cache/cache_type.rs` | 1 |
| `crates/rmlx-models/src/kv_cache/mod.rs` | 1 |
| `crates/rmlx-server/src/engine/helpers.rs` | 1 |

The target of 3 or fewer is tree-wide: every site in every crate counts.
This split does not meet it; the site reduction has its own plan.

**The census undercounts.** The compiler is the ground truth: plant one
variant in each enum and count the `E0004` errors. It reads 29 production
sites, not 26: 25 in `rmlx-kv-quant` and 4 downstream. The three the census
misses are the `Self::` matches in `impl KvStorage` (`reset`, `truncate_to`,
`try_deep_clone`). The census also cannot see an alias or glob import, a match
over an enum a `KvStorage` field holds, a `matches!` subset, or a string
spelling table such as `FromStr` or the SSD `read_layer` tag dispatch.

**Recall.** `scripts/kv_update_census_selftest.sh`, 49 cases over planted
trees, run by `make kv-update-census-selftest` and by `make ci`. Six of them
are pending: each states a figure for one blind spot above that the census
does not report yet, and fails the run the day the census reports it. A site in a
file the producer was never told about is found; a collapsed site drops the
count; a catch-all arm keeps the site and drops the forcing count; a match
under the bar is not a site; `match` inside a comment or a string literal is
not a site; a missing directory, a missing enum, an unreadable file and an
empty population each read `unavailable` with their own reason, never `0`.

Two of those cases hold the rules a threshold override would hide. The bar is
read with no `--threshold` at all, on a four-variant fixture carrying a site
that names exactly two: the bar reads 2 and the site is in. A fifth variant
then moves the bar to 3 and that site drops out, so the bar and the count move
together — replacing the derivation with a constant turns four cases red. A
wide `match` planted in a `*_tests.rs` file is not counted, and is counted
under `--include-tests`, so disabling the test-file exclusion turns one case
red. Both were measured by applying each mutation and re-running.

**After chunk (a):** 26 sites, all 26 forcing, and the same six in `update.rs`.
The split moved bodies, not `match` arms, so no figure in the table above
moved.

**After chunk (b2):** still 26 sites, all 26 forcing, still six in
`update.rs`. That chunk collapsed arms and not sites, and the target of 3 or
fewer is **not met** — §4 names the three accessors whose move would meet it
for `update.rs` while leaving the tree-wide count where it is.

### File size

`make file-size-report`, `rmlx-kv-quant` only, before the split:

| Lines | File | `LOC-exempt` |
|---|---|---|
| 7151 | `kvcache/update.rs` | added by this change, interim |
| 2877 | `kvcache/sdpa.rs` | no |
| 1952 | `quant.rs` | no |
| 1881 | `storage/kv_storage.rs` | yes |
| 1636 | `storage/quant_iso_v.rs` | yes |
| 1066 | `rotorquant.rs` | no |

**The rule the split must satisfy:** every file it produces is at or under 1000
lines, or carries a `LOC-exempt` marker whose text says why. The four other
oversized files in the crate are outside this work.

After chunk (a), the same command over the files the split produced:

| Lines | File | `LOC-exempt` |
|---|---|---|
| 3195 | `kvcache/update.rs` | yes — the dispatch, the capacity bookkeeping and the shared helpers are still over the guideline; chunk (b) is what removes the rest |
| 1242 | `kvcache/update_rotor.rs` | yes — four storage spellings, each with its own encode, ring sync and materialise-tail path |
| 963 | `kvcache/update_iso.rs` | no |
| 857 | `kvcache/update_affine.rs` | no |
| 578 | `kvcache/update_turbo.rs` | no |
| 229 | `kvcache/update_paged.rs` | no |
| 202 | `kvcache/update_planar.rs` | no |

The interim marker on `update.rs` did not go; its text was rewritten to say
what is left, which is what the rule asks of a marker that stays.

After chunk (b1), the same command:

| Lines | File | `LOC-exempt` |
|---|---|---|
| 2193 | `kvcache/update.rs` | yes — rewritten again: `exit_prefill` is down to 276 body lines of 2193, and what is left is the dispatch, the capacity bookkeeping, the bf16 mirror and the GPU-state walks |
| 1614 | `kvcache/update_rotor.rs` | yes — rewritten: the family's eight prefill bulk-encode bodies joined its four decode paths |
| 1260 | `kvcache/update_iso.rs` | yes — new marker: six prefill bodies took the file over the guideline |
| 995 | `kvcache/update_turbo.rs` | no — five lines under the guideline, so the next body added to this family needs a marker or a split |
| 968 | `kvcache/update_affine.rs` | no |
| 313 | `kvcache/update_planar.rs` | no |
| 222 | `kvcache/update_paged.rs` | no |
| 58 | `kvcache/update_mixed.rs` | no |

The chunk takes one file over the guideline and gives it a marker, and takes
`update.rs` 999 lines closer to it. The count of oversized files with no
marker anywhere in the tree is 22 on both sides, measured with
`make debt-report`.

Chunk (b2)'s reading of the same command is in §4, beside the pairs it
deleted. Its largest single move is `update_turbo.rs`, 995 to 530.

### Duplication

`python3 scripts/lib/debt_report.py --matched-lines update-bodies` is an
eleventh `debt-report` population: every `update_`-prefixed fn of the update
files, paired every item with every other. The three family populations each
read one codec's twins and go blind the moment those widths collapse; this one
reads the shape the restructure writes once.

One producer, not two. The census and this population print the same two
numbers, so both read fns through `lib/debt_report.py`'s `extract_fns`;
`lib/rust_scan.py` reads enums and `match` arms only.

**Before:** 3585 matched lines over 1440 body lines, 25 items, 300 pairs.
(Before the three collapses: 12122 over 2539, 33 items, 528 pairs.)

**The root is a glob, and the measure is orientation-free.** Chunk (a)
re-rooted the four update populations from the one file onto
`crates/rmlx-kv-quant/src/kvcache/update*.rs` — the dispatch file plus one file
per codec family — through one collector given its directory, glob and name
pattern at the registration site.

The measure had to change with it. `difflib.SequenceMatcher` anchors on the
longest match it finds in its first argument, so its matching-block sum is not
symmetric: measured one way round, the figure moved by 21 lines when the same
25 bodies were regrouped across seven files with no line of any of them
changing. It would have moved again on a rename, and again when a population
gained an item that re-ordered it. `matched_lines` now measures each pair both
ways round and reports the larger — the count of lines the pair genuinely
shares, which depends on the pair alone. `population_pairs` sorts the
population by name first, so the pair list is a function of the population and
not of the filesystem walk; no figure depends on that order any more.

Under the orientation-free measure the figure is the same on the commit before
chunk (a) and on the tree after it:

Chunk (b1) widened two of the four rules, so the `rotor-updates` and
`turbo-updates` rows below are re-measured on both arms under the widened one.
`iso-updates` and `update-bodies` are unchanged rules.

| Population | Before chunk (b1) | After chunk (b1) |
|---|---|---|
| `rotor-updates` | 0 over 776 (27 items, 0 pairs) | 138 over 1052 (35 items, 4 pairs) |
| `iso-updates` | 878 over 625 (21 items, 210 pairs) | 1545 over 837 (27 items, 351 pairs) |
| `turbo-updates` | 0 over 91 (2 items, 0 pairs) | 37 over 181 (4 items, 1 pair) |
| `update-bodies` | 3610 over 1440 (25 items, 300 pairs) | same |

`update-bodies` does not move, and that is its rule working rather than a
chunk that changed nothing: it is keyed on the `update_` prefix and an
`exit_prefill_*` fn does not carry it. The other three are keyed on a codec
token matched as a whole segment wherever it sits in the name, so each reads
its family's prefill bulk-encode bodies beside the decode ones.

**Two of those rules could not, before this chunk.** `rotor-updates` was
`^update_rotor`, an anchored prefix, and `turbo-updates` named only the `tsym`
spelling while the prefill bodies spell the same token `turbo_sym`. Both read
a measured `0` over a population that had just gained four and one exact pair
respectively — a counter that cannot fail. Measured: deleting
`exit_prefill_rotor4` from the tree left `rotor-updates` reading
`0 matched lines over 105 body lines (4 item(s), 0 pair(s))` before and after
under the anchored rule. Under the widened rule the same deletion reads
`138 over 1052 (35 items, 4 pairs)` before and `98 over 1012 (34 items, 3
pairs)` after, and deleting `exit_prefill_turbo_sym4` takes `turbo-updates`
from `37 over 181 (4 items, 1 pair)` to `0 over 138 (3 items, 0 pairs)`. The
four rotor pairs and the one turbo pair the widened rules now see are the
width twins chunk (b2) collapses; the counter has to be able to move when it
does.

The moved bodies duplicate each other exactly as much inside the family file
as they did inside the `match`. The figures record more items, not new twins —
what changed is that three of the four counters can now see them.

Chunk (b2) is what moved them. §4 carries all four figures on both arms, the
largest remaining pair in each population, and the reason each survivor stays.

Three figures in the tree moved when the measure did, on both arms alike, with
no body changing: `iso-updates` 875 → 878, `update-bodies` 3585 → 3610, and
`impls` 919 → 936 ([`SPEC_ROUND_SKELETON.md`](SPEC_ROUND_SKELETON.md) records
that one). The eight other populations are unchanged; each of them either
forms no pair or holds pairs `difflib` reads the same either way.

Recall: twelve cases in `scripts/debt_report_selftest.sh` — the planted
population held to its figure and its item and pair count, one body left
reading a measured `0` with the population still found, every body renamed out
of the prefix reading `unavailable`, a glob that matches no file named apart
from a root whose files hold no matching fn, three sym bodies moved into a
family file of their own leaving every figure where it was, and one planted
pair that shares four lines read one way and three read the other, in both
arrangements of its two names. That pair is the only one in the fixture tree
whose figure the orientation rule can move — every other planted body is too
short or too uniform for the asymmetry to show, which is why a one-directional
producer passed every case in the file before it was planted. Measured: with
the pair present, reverting `matched_lines` to one direction turns one case
red; reverting it and dropping the sort together turns the other red.

---

## 4. The chunks

One branch, one pull request, chunks as commits. Each chunk names its own
removals.

### Chunk (a) — split by codec family — landed

Each family's update path moved out of `update.rs` into a file of its own:
`kvcache/update_{rotor,iso,turbo,affine,planar,paged}.rs`. 67 fns moved — the
per-variant `update_*` entries, the width-parametric bodies they enter, the GPU
encode, ring-sync, materialise-tail and chunk-append helpers, and the two
rotor feed constants. `update.rs` keeps the `KvStorage` and `KvQuant`
dispatch, the prefill and decode capacity bookkeeping, the bf16 decode mirror,
and every helper with two or more family callers.

**Why `kvcache/` and not `storage/`.** The issue asks for the bodies "next to
the storage type". Most of them are `impl KvCache` methods, and they read
`KvCache` fields that are `pub(super)` in `kvcache::core` — visible inside
`crate::kvcache` and nowhere else. `crate::storage` is not a descendant of
`crate::kvcache`, so that placement would have to widen about twenty private
fields to `pub(crate)`, which is a change to the cache's encapsulation and not
a move. It would also invert the crate's layering, where `storage` is a leaf
of `kvcache`. The family files therefore sit beside `update.rs`, one per
family, matching the flat module layout `kvcache/` already uses.

**Pure move.** Every moved body is byte-identical to its body before the move
once leading whitespace, blank lines, `use` lines and a leading visibility
keyword are normalised away. `update_planar_k` carries the one `target =`
override in the file, and the override travelled with the body, so the three
readers of `warm_ttft_bypass` see no change.

Removed: the moved bodies from `update.rs`; the `use` items and storage-type
imports the move left without a caller there; the `ring_feed_routing_tests`
module declaration, which moved to `update_rotor.rs` with the two constants it
reads.

Kept, with its text rewritten: the `LOC-exempt` marker on `update.rs`, which is
still 3195 lines. `update_rotor.rs` is 1242 and carries a marker of its own.
The five other family files are under the guideline.

Judged by: the oracle green with no re-baselined pin; the match-site count per
file, unchanged at 26 with six in `update.rs`; `make file-size-report`; the
four duplication figures, unchanged under the re-rooted producer.

**The move, falsified.** Four mutations, one per family and one in an
`exit_prefill` arm of a materialising spelling. Each scales the K values a
body hands its store by 1.01, on the CPU route the oracle drives. The control
is the green suite above.

| Mutation | File | First cell red | Column |
|---|---|---|---|
| `tsym_update` K values ×1.01 | `update_turbo.rs` | `tsym3 @ kv_h=1 head_dim=128` | store bytes after the bulk append |
| `iso_v_update` K values ×1.01 | `update_iso.rs` | `iso3 @ kv_h=1 head_dim=128` | store bytes after the bulk append |
| `rotor_sym_update` K values ×1.01 | `update_rotor.rs` | `rotor3_sym @ kv_h=1 head_dim=128` | store bytes after the bulk append, and the rows after `exit_prefill` |
| `Rotor3Sym` `exit_prefill` arm K values ×1.01 | `update.rs` | `rotor3_sym @ kv_h=1 head_dim=128` | store bytes after `exit_prefill` |

The last one is the second drive's, and its arm did not move: the
`exit_prefill` arms stayed with the dispatcher (see below). Every mutation was
applied to a file copy, run, and restored by `cp` with a digest check on both
sides — never by `git checkout`.

### Chunk (b1) — extract the `exit_prefill` arms — landed

Chunk (a) left the `exit_prefill` arms where they were. They read locals the
enclosing fn builds, so moving one is an extraction and not a move. This chunk
does the extraction. 25 of the 26 arms of the one `match self.quant` are now a
fn in their family's file, and the arm that selected them is one call.
`exit_prefill` keeps the rotating and paged early returns, the raw-buffer
slice, the compact bf16 seed, the `KvStorage::None` guard, the
`materialises_packed_store()` gate, the dispatch and the warm-TTFT epilogue:
276 body lines, down from 1258.

| Family | Arms | File |
|---|---|---|
| rotor | 8 | `kvcache/update_rotor.rs` |
| iso | 6 | `kvcache/update_iso.rs` |
| turbo | 6 | `kvcache/update_turbo.rs` |
| affine | 2 | `kvcache/update_affine.rs` |
| planar | 2 | `kvcache/update_planar.rs` |
| mixed | 1 | `kvcache/update_mixed.rs` — new file |

`KvQuant::None` is the one arm that stays, and the reason is structural
before it is editorial. It is the only arm that reads `raw_k` / `raw_v` — the
owned buffers the enclosing fn took out of `self`, which every other arm sees
only through the `k_full` / `v_full` slices — and the only one whose body
carries a successful early `return Ok(())` rather than falling through to the
warm-TTFT epilogue. Both facts are derived from the 26 arm bodies at the base
commit, not asserted: no other arm names either buffer or returns `Ok`. Extracting it would mean handing a fn two more owned buffers and
giving it a way to say "skip the epilogue", which is a change to the
dispatch's shape and not a move. It also holds no codec store: it promotes
those buffers into the bf16 decode mirror that `update.rs` owns, beside
`update_none` and `update_decode_fp16`. There is no `None` family file and
this chunk does not make one. `KvStorage::Paged` has no arm at all —
its seed is an early return above the gate, not a bulk encode — so
`update_paged.rs` is untouched.

`update_mixed.rs` is new because `Mixed` had no family file: its decode path
is not in the update file at all (`KvCache::update` refuses a Mixed cache, and
the per-step append is `update_and_sdpa_mixed` in `sdpa.rs`), so chunk (a) had
nothing to move. Its one prefill body could have gone into the affine file —
`MixedKvState` is affine K8 and affine V plus a rotation — but a file named
for the affine spellings is not where a reader looks for `Mixed`, and the
issue names mixed as a family of its own.

**One fn per arm, not one per family.** A fn per family needs a second
`match self.quant` inside it, with a wildcard arm that every new codec of that
family has to touch — the opposite of what this restructure is measured on.
The one dispatch stays one dispatch: 26 arms, 25 of them a single call, no new
`match` anywhere. A per-arm fn is also the unit chunk (b) deletes, since that
chunk writes one body per store shape and a per-arm fn is what such a body
replaces.

**Naming.** `exit_prefill_<spelling>`, the `KvQuant` variant in snake case.
The prefix is load-bearing for the metrics: `update-bodies`, `rotor-updates`
and `turbo-updates` are keyed on `update_`, `^update_rotor` and the `tsym`
token, so a chunk that adds no update body moves none of them. `iso-updates`
is keyed on the `iso` token wherever it sits in the name, so the six iso
prefill bodies do join it. §3 records the figure.

**The 18 decode-inert spellings' arms moved too.** The gate returns before
them and no CPU route runs them, so they are dead code under every drive but
the oracle's. They are not deleted: they are the re-enable path for a codec
that grows a decode kernel over its own store, and that is what the comment at
the gate says. Deleting them would be a codec retirement wearing a
housekeeping hat.

Two tests hold them, and they hold different things.
`warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`
holds the **classification**: it sweeps every variant and fails when a codec's
arm and its `materialises_packed_store()` answer disagree. It does not read a
byte of the 18 inert bodies, because after the gate there is nothing to read.
What a gate regression trips first is
`store_bytes_tests::the_two_populations_partition_every_spelling`, which
derives the 18/10 split from the predicate itself, so a spelling crossing the
line turns a cell red rather than quietly changing what a served run is worth.
Both are unchanged and green.

**Extraction, proven.** Every extracted arm is statement-identical to the arm
it replaces, and the proof is a script rather than a reading. It takes the old
arm body from the base commit, applies the declared parameter binding, wraps
it in the new fn's own signature inside an `impl KvCache` block, runs
`rustfmt`, and compares the result byte for byte with the fn that now sits in
the family file. Running `rustfmt` on both sides is what removes the reflow a
dedent of eight columns causes, so no rule in the script has to describe that
reflow and no rule can hide a real change behind it. **25 arms compared, 0
differing**, at the per-family counts in the table above.

Three declared changes, and nothing else:

* the parameter binding — `&k_full` becomes `k_full` and `&v_full` becomes
  `v_full`, because the parameter is already a reference;
* the trailing `Ok(())` — the arm fell through to the epilogue, the fn
  returns;
* five comments whose "above" or "below" named a line of the enclosing fn.
  Each names `decode_fp16_pair`, which the caller still builds, so each now
  says "the caller's". The script lists them one by one.

The parameter list is derived per arm from the arm's own body, not chosen:
`k_full` and `device` always, `v_full` where the arm encodes a V axis,
`total_seq` where the arm's `tracing::debug!` names it, and `policy` in the
one arm that hands it to `MixedKvState::bulk_init_from_fp16`.

**Trace targets.** No event in a moved arm carried a `target =` override, so
each one's target was the module path `rmlx_kv_quant::kvcache::update` and is
now its family module's. One of them is read by name: the `iso3_encode` phase
of the `Iso3` arm, which [`PERF_BASELINE.md`](PERF_BASELINE.md) names beside
its target. This chunk takes the option chunk (a) took — correct the reader in
the same change, rather than pin the event with an explicit `target =`, which
would be a statement the old arm did not have and would fail the identity
proof. `iso3_encode` now emits on `rmlx_kv_quant::kvcache::update_iso` beside
the four iso decode phases, one target where the doc used to list two, and its
`site = "exit_prefill"` field is unchanged. A grep of the tree for
`kvcache::update` finds no other reader.

Removed: the 25 arm bodies from `update.rs`; the eight storage-type imports
and the two f32-conversion helper imports the move left without a caller
there; and two of `exit_prefill`'s three `#[allow]` attributes,
`clippy::unreachable` and `clippy::wildcard_enum_match_arm`, which no
statement left in the fn raises. The two were found rather than guessed: every
attribute the chunk touches was written as `#[expect]` first, which reports an
attribute nothing fires under, and only the ones clippy confirmed were kept.
That is also how each moved body's own allow list was cut — 75 candidates, 20
of them unfulfilled, 55 kept.

**The move, falsified.** Six mutations: one live arm of each materialising
family that has one, and two arms of decode-inert spellings. Each scales by
1.01 the K values the body hands its store. The control is the green suite
above.

| Mutation | File | Reachable | First cell red | Column |
|---|---|---|---|---|
| `exit_prefill_rotor3_sym` K values ×1.01 | `update_rotor.rs` | yes | `rotor3_sym @ kv_h=1 head_dim=128` | store bytes after `exit_prefill` |
| `exit_prefill_iso3_sym` K values ×1.01 | `update_iso.rs` | yes | `iso3_sym @ kv_h=1 head_dim=128` | store bytes after `exit_prefill` |
| `exit_prefill_rotor_k_only3` K values ×1.01 | `update_rotor.rs` | yes | `k_rotor3 @ kv_h=1 head_dim=128` | store bytes after `exit_prefill` |
| `exit_prefill_mixed` K array ×1.01 | `update_mixed.rs` | yes | `mixed_k8g64_v4g64 @ kv_h=1 head_dim=128` | store bytes after `exit_prefill` |
| `exit_prefill_turbo_sym3` K values ×1.01 | `update_turbo.rs` | **no** | none — 578 passed | none |
| `exit_prefill_k8v4` K values ×1.01 | `update_affine.rs` | **no** | none — 578 passed | none |

Every red cell is the second drive's, `exit_prefill_bulk_encode_bytes_are_pinned_per_spelling_and_shape`,
and each names its own spelling. The control run is 578 passed, 0 failed —
the lib suite, which is the population every row above is counted against.
The whole crate, lib plus integration binaries, is 584 passed, 0 failed.

The turbo file has no reachable row to offer. All six turbo spellings report
`false` from `decode_reads_packed_store()`, so the gate returns before every
one of their arms; the turbo mutation above is therefore a second dead-arm
row, not a live one.

The last two rows are the point of the exercise. The `k8v4` and `turbo_sym3`
arms are two of the 18 the gate returns before, so no CPU route reaches it and no assertion in the tree
can observe what either writes — the guard pins their store at zero bytes, and
a zero-byte store is what an unreachable arm and a mutated unreachable arm
both produce. **Those two mutations are green, and they must be**: a red cell
there would mean the gate had stopped gating. The evidence for those arms is that they
compile, that the guard holds them at zero, and that the extraction is
statement-identical — not a test that runs them, because none does. Every
mutation was applied to a file copy, run, and restored by `cp` with a digest
check on both sides — never by `git checkout`.

### Chunk (b2) — one body per store shape — landed

24 of the 27 variants carry the same store-slot shape and their bodies are the
same sequence: cast to bf16, ensure the store, append K, append V, dequant or
read the mirror, run SDPA. This chunk writes that sequence once per shape, on
the prefill side for every family and on the decode side for the one family
that still held copies. It also normalises the storage-mismatch spelling
across every prefill body.

#### The mechanism, and why it is not a trait

Three mechanisms were on the table: a store-side trait pair, a const-generic
width, and plain fns taking the K and V stores. Two of the three are used and
the trait is not, and the reason is measured rather than stylistic.

**What varies across the 25 prefill bodies** is four things: the `KvStorage`
variant they destructure, the store *types* in the two slots, the scalar
knobs those stores are built with (`bits`, `use_tcq`, `v_bits`), and the
device rule for each axis. A shared body can absorb the last two — they are
values — and a const-generic can absorb a store type that is already generic
over its width. It cannot absorb the variant, because a `match` arm binding
`Option<QuantRotorV<3>>` and one binding `Option<QuantRotorV<4>>` have
different types and one `match` cannot return both.

So every shape is an **entry plus a body**. The entry destructures its
variants and hands the two store slots to the body; the body is a free fn
taking `&mut Option<KStore>` and `&mut Option<VStore>` and nothing else of the
cache. That is the shape the `update_rotor_*` and `update_iso_*` decode
entries already had, so the prefill side now reads like the decode side.

Where the two bodies of a shape differ only in the code width, the body is
`<const BITS: u8>` and the entry resolves the width — the mechanism the three
storage collapses established. Where they differ only in a scalar, the body is
a plain fn and the scalar is a parameter.

**A store-side trait pair would be needed for one thing only**: the three
bodies whose slots hold different store *types* at the same shape —
`update_k8v4` (`QuantV`), `update_k8v8` (`QuantK`) and `update_planar`
(`QuantPlanarV`), and their prefill counterparts.

Measured over that population alone, the trait is **two methods, three impls
and no constructor**: all three V stores take the same `append(vals, shape,
src, device, max_seq)` and the same `dequantize_choice(device, dtype)`. What
they do not share is how they are *built* — `QuantV` needs `bits` and
`use_tcq`, `QuantPlanarV` needs `bits`, `QuantK` needs neither — and building
is what the entry does. So the reason these three stay apart is not that a
trait would be large: it is that the difference is in the construction, and
the entry is where construction lives.

That makes the two-method trait a **candidate for a later chunk**, not a shape
this one rejects on principle. The figure it would move is 64 matched lines
between `update_k8v4` and `update_k8v8` and 62 between `update_k8v4` and
`update_planar` — both 64 and 62 on the arm before this chunk as well, so
neither is this chunk's to claim. A trait over the *whole* 24-variant shape is
a different proposition and is the one constraint 1 forbids: the five V-side
constructors in play there are `QuantV`, `QuantK`, `QuantPlanarV`,
`QuantRotorV::new(shape, max_seq, layer_idx)` and `QuantIsoV::new(shape)`, and
the rotor and iso `append`s take `(vals, shape)` — a different arity from the
other three.

#### What the chunk deletes

**Prefill: 25 bodies to 14.** Nine shared bodies replace 20 copies, and five
bodies stay as they are.

| Shared body | Deletes |
|---|---|
| `rotor_v_bulk_encode<BITS>` | `exit_prefill_rotor3`, `exit_prefill_rotor4` |
| `rotor_sym_bulk_encode<BITS>` | `exit_prefill_rotor3_sym`, `exit_prefill_rotor4_sym` |
| `rotor_k_only_bulk_encode<BITS>` | `exit_prefill_rotor_k_only3`, `exit_prefill_rotor_k_only4` |
| `rotor_k_asym_bulk_encode<BITS>` | `exit_prefill_rotor_k3_asym`, `exit_prefill_rotor_k4_asym` |
| `iso_v_bulk_encode<BITS>` | `exit_prefill_iso3`, `exit_prefill_iso4` |
| `iso_sym_bulk_encode<BITS>` | `exit_prefill_iso3_sym`, `exit_prefill_iso4_sym` |
| `iso_k_only_bulk_encode<BITS>` | `exit_prefill_iso_k_only3`, `exit_prefill_iso_k_only4` |
| `k8_turbo_v_bulk_encode` | `exit_prefill_k8vturbo3`, `exit_prefill_k8vturbo2`, `exit_prefill_k8vturbo3_tcq`, `exit_prefill_k8vturbo2_tcq` |
| `tsym_bulk_encode<BITS>` | `exit_prefill_turbo_sym3`, `exit_prefill_turbo_sym4` |

Nine entries carry them: `exit_prefill_rotor_v`, `exit_prefill_rotor_sym`,
`exit_prefill_rotor_k_only`, `exit_prefill_rotor_k_asym`,
`exit_prefill_iso_v`, `exit_prefill_iso_sym`, `exit_prefill_iso_k_only`,
`exit_prefill_k8_turbo_v`, `exit_prefill_turbo_sym`.

The symmetric turbo pair was kept apart in the first pass of this chunk, for a
CPU-forced V axis. Review found that reading wrong: the V device is one
expression, and [`tsym_update`] already carries it —
`if BITS == TURBO_K4_BITS { device } else { Device::Cpu }` — while
`arrays_to_f32` is two `array_to_f32_vec` calls, so the f32 rule follows the
resolved device per axis rather than the two widths. The pair collapsed with
no other change. §"What this chunk could not prove" carries the evidence,
which is not a red cell.

**Decode: 1 shared body replaces 4 copies.** `k8_turbo_v_update` deletes
`update_k8vturbo3`, `update_k8vturbo2`, `update_k8vturbo3_tcq` and
`update_k8vturbo2_tcq`; `update_k8_turbo_v` is its entry. Measured on the
arm before this chunk, those four were the tree's largest duplication pairs —
`update_k8vturbo2_tcq` against `update_k8vturbo3_tcq` at 73 matched lines,
`update_k8vturbo2` against `update_k8vturbo3` at 72. The rotor and iso decode
paths were already one body per shape, from the storage collapses.

**Also deleted**: the `use` import of `QuantIsoV3` and `QuantIsoV4` from
`update_iso.rs`, whose shared bodies name `QuantIsoV<BITS>` instead. The two
aliases themselves stay — they are declared in `storage/quant_iso_v.rs` and
are the field types of four `KvStorage` variants. two `clippy::wildcard_enum_match_arm` allows the rotor
per-variant matches needed; eight `clippy::unreachable` allows; two
`clippy::unwrap_used` allows the four deleted turbo decode bodies needed,
because the shared body reads its store back with `let Some(..) else`, the
shape `iso_v_update` and `tsym_update` already use. One allow came back —
`clippy::wildcard_enum_match_arm` on `k8_turbo_v_knobs`, whose `_ => None`
answers for the 23 variants the table is not about. The `#[allow(` count over
`kvcache/update*.rs` goes 110 to 65.

#### The pairs that stay, and why

Two shapes keep two bodies, and both for the same reason: a different V store
*type*, which is the one difference an entry cannot absorb.

| Pair | Matched lines | Why it stays |
|---|---|---|
| `update_k8v4` / `update_k8v8` | 64 | `Option<QuantV>` against `Option<QuantK>`. Needs the two-method trait argued about above. Pre-existing and unmoved: 64 on both arms. |
| `update_k8v4` / `update_planar` | 62 | The same, with `Option<QuantPlanarV>`. Also 62 on both arms. |

A third pair was on this list and is not any more:
`exit_prefill_turbo_sym3` / `exit_prefill_turbo_sym4`, at 45 matched lines.
Its stated reason — a CPU-forced V axis — did not survive review, and the
collapse above took it to a measured `0`.

The largest pair in `iso-updates` after the chunk is `iso_k_only_k_side`
against `iso_v_encode_decode` at 50 matched lines, unchanged on both arms.
Those two are different axes with different ring policies, not a width pair,
and this chunk did not touch either.

#### One storage-mismatch spelling

The prefill bodies said "the storage does not match the quant" three ways,
with three different consequences for the same construction-time defect: a
panic, an untyped `Error::Mlx` string, and the typed
`Error::KvStorageMismatch { expected, got }`. Every prefill body now returns
the typed one. Measured over the `exit_prefill_*` bodies of
`kvcache/update*.rs`: **47 sites in 25 bodies before — 15 `unreachable!`, 26
`Error::Mlx`, 6 structured — and 21 sites in 15 bodies after, all
structured**, no `unreachable!` and no `Error::Mlx` left. All eight
`clippy::unreachable` allows are gone with them. `update_mixed.rs` now
carries no `#[allow]` at all.

One producer, not 21 literals. `storage_mismatch` in `update.rs` is the one
place that builds the error, and every prefill and decode path calls it.
Review found it still building an `Error::Mlx` while the prefill bodies it
sits beside had moved to `Error::KvStorageMismatch` — the same defect in two
retry classes at once.

**That is a retry-envelope move, and it is the point of the change.**
`Error::is_migratable` reports `Error::Mlx` as transient, and
`rmlx-server`'s retry envelope replays a transient failure on another engine.
A storage mismatch is not transient: the cache was built for one `KvQuant`
and is being driven as another, so the replay builds the same wrong cache.
Ten decode callers used to say "transient, replay may succeed" about a
condition the prefill callers already called "permanent, replay futile".
Permanent is the right class for both, and that is what they all report now.

Left alone, and the one place the spelling is still not uniform: the four
decode bodies that say `unreachable!` — `update_k8v4`, `update_k8v8`,
`update_planar` and `update_paged`. They panic where every other path returns.

#### The collapse, falsified

Twenty-one mutations on the collapsed tree, plus a control. Every one was
applied to a file copy, run, and restored by `cp` with a digest check on both
sides — never by `git checkout`. The control is 578 passed, 0 failed, the lib
suite. The first fifteen are the first pass of this chunk; the last six were
added when review reopened the symmetric turbo pair and the knob table.

| Mutation | Result | First cell red |
|---|---|---|
| control, no mutation | green, 578 passed | — |
| `rotor_sym_bulk_encode` K ×1.01 | red | `rotor3_sym @ kv_h=1 head_dim=128` |
| the same, guarded to `BITS == 4` | red | `rotor4_sym @ kv_h=1 head_dim=128` |
| `iso_sym_bulk_encode` K ×1.01 | red | `iso3_sym @ kv_h=1 head_dim=128` |
| `rotor_k_only_bulk_encode` K ×1.01 | red | `k_rotor3 @ kv_h=1 head_dim=128` |
| `iso_k_only_bulk_encode` K ×1.01 | red | `k_iso3 @ kv_h=1 head_dim=128` |
| the same, guarded to `BITS == 4` | red | `k_iso4 @ kv_h=1 head_dim=128` |
| `iso_v_bulk_encode` K ×1.01 (dead arm) | **green**, 578 passed | none |
| `k8_turbo_v_bulk_encode` K ×1.01 (dead arm) | **green**, 578 passed | none |
| `exit_prefill_k8_turbo_v` TCQ flag `true` → `false` (dead arm) | **green**, 578 passed | none |
| `k8_turbo_v_update` K ×1.01 | red | `k8vturbo3 @ kv_h=1 head_dim=128` |
| `iso_v_update` K ×1.01 | red | `iso3 @ kv_h=1 head_dim=128` |
| `tsym_update` K ×1.01 | red | `tsym3 @ kv_h=1 head_dim=128` |
| `exit_prefill_rotor_v` resolves `RotorV3` to `::<4>` | **compile error** `E0308` | — |
| `update_k8_turbo_v` passes `v_bits = 2` for `K8VTurbo3` | red, 4 tests | `k8vturbo3 @ kv_h=1 head_dim=128` |
| `tsym_update` K ×1.01, guarded to `BITS == 4` | red | `tsym4 @ kv_h=1 head_dim=128` |
| `tsym_update` V-device rule inverted | **green**, 578 passed | none |
| `tsym_bulk_encode` K ×1.01 (dead arm) | **green**, 578 passed | none |
| the same, guarded to `BITS == 4` (dead arm) | **green**, 578 passed | none |
| `tsym_bulk_encode` V-device rule inverted (dead arm) | **green**, 578 passed | none |
| `k8_turbo_v_knobs` `K8VTurbo3Tcq` `use_tcq` `true` → `false` | red | `the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes` |
| `k8_turbo_v_knobs` `K8VTurbo3` `v_bits` `3` → `2` | red, 4 tests | `k8vturbo3 @ kv_h=1 head_dim=128` |

Three rows carry the argument.

**A mis-resolved width is a compile error, not a red cell.** The entry hands
the body a slot whose type names the width — `&mut Option<QuantRotorV<3>>`
against a `&mut Option<QuantRotorV<4>>` parameter — so `::<3>` and `::<4>`
are not interchangeable and `rustc` rejects the swap. That holds for all seven
const-generic bodies. It does **not** hold for `k8_turbo_v_update`, whose
width is the runtime `v_bits` field of a `QuantV`: there a wrong argument
compiles, and the last row is the oracle catching it — four cells, not one.

**Both widths of a shared body are driven.** The three guarded mutations are
the converse of the collapse's risk: a shared body that only its 3-bit
instantiation ever reached would leave the 4-bit cell testing nothing. Scaling
K only when `BITS == 4` turns `rotor4_sym`, `k_iso4` and `tsym4` red and
leaves the 3-bit cells green, so both instantiations reach a pinned cell.

**Six of the seven green rows are the gate working, not a gap.** All six turbo
spellings and both plain iso spellings report `false` from
`decode_reads_packed_store()`, so `exit_prefill` returns before their arms and
no CPU route runs them. A red cell there would mean the gate had stopped
gating. §2 records this population and the guard that holds it to the
predicate.

**The seventh green row is a different blindness, and it is the oracle's.**
Inverting the V-device rule in `tsym_update` — the reachable decode body —
is green as well, and the reason is that the rule only distinguishes anything
when `device == Device::Gpu`. Every drive in this oracle is CPU, so
`if BITS == TURBO_K4_BITS { device } else { Device::Cpu }` and its inverse
both evaluate to `Device::Cpu` at every cell. The rule is a GPU-path choice
and no CPU pin can see it, on the decode side or the prefill side. Its gate is
`make gpu-test`, which is owed and not run here. That is also why the
symmetric prefill collapse's evidence below is a construction argument rather
than a mutation: the one line the two widths did not share is the one line
this oracle is blind to.

#### The blind spot this chunk opened, and closed

The first pass of this chunk moved the four turbo spellings'
`(v_bits, use_tcq)` pairs out of four bodies and into **two** call sites, one
in the decode entry and one in the prefill entry. That was a decision table
written twice, and the prefill copy was unobservable: those four arms sit
behind the `materialises_packed_store()` gate, the prefill drive covers only
the ten materialising spellings, and none of them is a turbo one. Measured at
that point: flipping `use_tcq` from `true` to `false` in the `K8VTurbo3Tcq`
prefill arm left the whole crate green, 578 passed.

`k8_turbo_v_knobs` is now the one table and both entries read it. There is
nothing left to mutate at the prefill entry — it carries no constant — and
mutating the table instead is caught by the decode drive, which reads the same
four rows. Measured on the collapsed tree:

| Mutation of the one table | Result | First cell red |
|---|---|---|
| `K8VTurbo3Tcq` `use_tcq` `true` → `false` | red, 1 test | `the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes` |
| `K8VTurbo3` `v_bits` `3` → `2` | red, 4 tests | `k8vturbo3 @ kv_h=1 head_dim=128` |

The first of those is the exact mutation that was green before the table was
merged. Each entry still binds its stores with a four-variant or-pattern, but
that pattern carries no constants: every arm hands the body the same triple,
so a wrong arm cannot mis-configure a store.

#### What this chunk still cannot prove

**The symmetric turbo prefill collapse has no red cell to offer.** All six
turbo spellings report `false` from `decode_reads_packed_store()`, so
`exit_prefill` returns before both `tsym_bulk_encode` instantiations and no
CPU route runs either. Measured: scaling the K values the body hands its store
by 1.01 leaves the crate green at 578 passed, and so does the same mutation
guarded to `BITS == 4`. **Those greens are the gate working** — the same
result §4 already records for `exit_prefill_turbo_sym3` under chunk (b1) — and
a red cell there would mean the gate had stopped gating.

The evidence for that one collapse is therefore not a test. It is three
things:

1. **The width binds at compile time.** The entry hands the body a
   `&mut Option<QuantKTurbo<3>>` or a `&mut Option<QuantKTurbo<4>>`, so
   `::<3>` and `::<4>` are not interchangeable.
2. **The V-device rule is `tsym_update`'s, unchanged, and that body is
   reachable at both widths.** Scaling K in `tsym_update` turns `tsym3` red,
   and the same mutation guarded to `BITS == 4` turns `tsym4` red. The rule
   itself is outside this oracle on both sides — inverting it in either body
   is green, because it selects nothing until `device == Device::Gpu` and
   every drive here is CPU. What the collapse did was reuse the rule the
   decode path already carries, rather than write a second copy of it.
3. **The four cases agree by construction.** `arrays_to_f32(k, v, d)` is
   `(array_to_f32_vec(k, d), array_to_f32_vec(v, d))`, so each old body and
   the new one produce the same three values at every `(BITS, device)`:

| `BITS` | `device` | `k_f32` | `v_f32` | V append device |
|---|---|---|---|---|
| 3 | `Cpu` | `array_to_f32_vec(k, Cpu)` | `array_to_f32_vec(v, Cpu)` | `Cpu` |
| 3 | `Gpu` | empty | `array_to_f32_vec(v, Cpu)` | `Cpu` |
| 4 | `Cpu` | `array_to_f32_vec(k, Cpu)` | `array_to_f32_vec(v, Cpu)` | `Cpu` |
| 4 | `Gpu` | empty | empty | `Gpu` |

The store constructors carry over unchanged: `QuantKTurbo3` and
`QuantKTurbo4` are aliases for `QuantKTurbo<3>` and `QuantKTurbo<4>`, and the
V store's `bits: 3` / `bits: 4` become `bits: BITS`.

**A merged prefill entry no longer refuses a quant/storage width
disagreement.** Before this chunk, `KvQuant::Iso3` on a `KvStorage::IsoV4`
cache reached the `Iso3`-only entry and returned a mismatch. The merged entry
resolves the width from the storage and encodes at 4 bits. That direction is
right — decode has always dispatched on the storage, so the two halves of the
cache now agree where they used to disagree — but the guard is gone, and a
construction-time defect that used to fail loudly would pass silently.
`warn_if_width_disagrees` is what makes it visible: it compares the quant's
own side width, from `KvQuant::approx_code_bits`, against the width the entry
resolved, and warns. Deliberately not an assert, because the storage is the
authority both halves already follow. Nothing asserts on that warning, so a
mutation that inverts the comparison is not caught either. The diagnostic is
reached by 3 of its 17 sites under the CPU oracle today (`iso3_sym`,
`k_iso3`, `k_rotor3`); the other 14 sit behind the
`materialises_packed_store()` gate and are inert. The no-false-warning
conclusion holds by construction, not by measurement.

#### The figures

Match sites: 26 before, 26 after, six of them in `update.rs`, all 26 forcing.
The chunk collapsed `match` **arms**, not sites — the `KvStorage` dispatch
went 19 arms to 16 and the `KvQuant` `exit_prefill` dispatch 26 to 16, each
still naming every variant with no catch-all, so a new codec still has to
touch both. **The issue's target of 3 or fewer is not met and this chunk
cannot meet it.** Its six sites are the two dispatches plus
`storage_has_materialised_payload`, `eval_gpu_state`, `storage_max_seq` and
`set_storage_max_seq`. The last three are `KvStorage` accessors that belong on
`KvStorage` itself; moving them to `storage/kv_storage.rs` would take
`update.rs` to three sites but would **move** sites rather than remove them,
leaving the tree-wide count at 26. That is a different change from writing one
body per shape, and this chunk deliberately did not start it.

Update bodies, `make kv-update-census`: 25 bodies over 1440 lines before, 22
over 1117 after. `file-lines` over the update files 7623 to 6818.

Duplication, `scripts/lib/debt_report.py --matched-lines`:

| Population | Before chunk (b2) | After chunk (b2) |
|---|---|---|
| `update-bodies` | 3610 over 1440 (25 items, 300 pairs) | 1897 over 1117 (22 items, 231 pairs) |
| `rotor-updates` | 138 over 1052 (35 items, 4 pairs) | 0 over 931 (35 items, 0 pairs) |
| `turbo-updates` | 37 over 181 (4 items, 1 pair) | 0 over 149 (4 items, 0 pairs) |
| `iso-updates` | 1545 over 837 (27 items, 351 pairs) | 1261 over 720 (27 items, 351 pairs) |

`rotor-updates` reads a measured `0` with its 35 items still found — the four
width pairs chunk (b1)'s widened rule had just made visible are the four this
chunk collapsed, and the counter moved when it did. That is the shape the
storage collapses left behind too, and it is not `unavailable`: a population
that resolved to nothing would exit 1.

`turbo-updates` reads a measured `0` over its 4 items. It went the other way
first: the error-spelling normalisation took its one pair from 37 to 45,
because both bodies of `exit_prefill_turbo_sym3` / `exit_prefill_turbo_sym4`
gained the same structured-error blocks. Collapsing that pair then took the
figure to zero. Both readings are on the same rule, the one chunk (b1)
widened onto the codec token.

`iso-updates` holds its item count at 27 by coincidence — six prefill bodies
left and three shared bodies plus three entries arrived — while the lines the
population shares fell by 275.

File sizes, `make file-size-report`:

| File | Before | After | `LOC-exempt` |
|---|---|---|---|
| `kvcache/update.rs` | 2193 | 2202 | yes — the dispatch, the capacity bookkeeping, the bf16 mirror and the GPU-state walks |
| `kvcache/update_rotor.rs` | 1614 | 1480 | yes — four storage shapes at two widths, each with its own GPU encode, ring sync and materialise-tail path |
| `kvcache/update_iso.rs` | 1260 | 1121 | yes |
| `kvcache/update_affine.rs` | 968 | 957 | no |
| `kvcache/update_turbo.rs` | 995 | 487 | no — the file had been five lines under the guideline; the collapses took 508 lines out of it |
| `kvcache/update_planar.rs` | 313 | 299 | no |
| `kvcache/update_paged.rs` | 222 | 222 | no |
| `kvcache/update_mixed.rs` | 58 | 50 | no |

Every family file shrank. `update.rs` grew by 9: it gained
`warn_if_width_disagrees`, and `storage_mismatch` is longer as a typed
constructor with a doc comment that states the retry class. No file crosses
the guideline in either direction, and the three markers that were already
there stay.

**Trace targets.** No body changed module in this chunk, so no event changed
target. The one target read by name is still the one
[`PERF_BASELINE.md`](PERF_BASELINE.md) records:
`rmlx_kv_quant::kvcache::update_iso` for the four iso decode phases and the
`iso3_encode` prefill phase. That phase is the one place a width pair is not
exact — the 3-bit V encode is timed and the 4-bit one has no pinned phase — so
the shared body keeps it behind a `BITS == 3` guard rather than renaming it or
adding an `iso4_encode` the doc does not name. The PlanarK `warm_ttft_bypass`
override is in `update_planar.rs` and untouched; its three readers see no
change.

**Nothing retired.** Every `KvQuant` spelling, every `KvStorage` variant, every
`Display` / `FromStr` arm, every CLI name, every `LAYOUT_TAG` and every
`.metal` kernel is untouched: the diff is seven files, all
`crates/rmlx-kv-quant/src/kvcache/update*.rs`, 707 insertions and 1412
deletions. The SSD layout-key salt was not bumped and was not read.

### Constraints

These are the orchestrator's, and they bound chunk (b2) hardest.

1. **Readability first.** No trait tower. No premature generics.
2. **A generic body is justified only where it deletes two or more real
   copies**, and the doc names each copy it deletes. One instantiation is not a
   generic; it is a function.
3. **The runtime `KvStorage` enum stays.** `--kv-quant` dispatch reads it. The
   generic pair lives inside the arm, not instead of it.
4. **No codec retirement.** No spelling, CLI name, `.metal` kernel or census
   entry is retired, renamed or deleted. Every spelling keeps the same bytes
   and the same tokens.

### What cannot move

Grep before touching any of it.

* **Every pin** in `store_bytes_tests.rs`, and the four `LAYOUT_TAG` families
  in `storage/kv_storage.rs` (`ISO_*`, `ROTOR_*`, `TURBOSYM*`, `PLANARK4`,
  `K8VTURBO*_TCQ`). A layout tag is written into the SSD index.
* **The `KvQuant` spellings, their `Display` and `FromStr`**, and the CLI names
  built from them. `make check-kv-codec-disposition` reads the help text and
  the `docs/KV_QUANT.md` banners against the runtime disposition.
* **The `.metal` kernels** and their `probes/kernels.manifest` entries.
  `make check-metal-compiles` fails on a `.metal` file its manifest does not
  name.
* **The GPU shader-validation census**, `scripts/gpu_validation_census.txt`:
  one entry per (kernel, kind, crate, originating test).
* **The trace target the update path emits on.** All but one event in
  `update.rs` carries no `target =` override, so its target is the module path
  — `rmlx_kv_quant::kvcache::update`. [`PERF_BASELINE.md`](PERF_BASELINE.md)
  names that string and the phases under it. Moving a body to another module
  moves its target. A chunk that moves an instrumented body updates that doc
  in the same commit, or keeps the target explicit. Chunk (a) took the first
  option, and so did chunk (b1): the four iso decode phases (`iso_encode`,
  `iso_dequant_gpu`, `iso_dequant_cpu`, `iso_vec_to_array`) and the
  `iso3_encode` prefill phase all emit on
  `rmlx_kv_quant::kvcache::update_iso`, and `PERF_BASELINE.md` names that one
  target. No other moved event's target is read by name anywhere in the tree —
  a grep for `kvcache::update` finds that doc table, this document, and one
  `use` path.
* **The one `target =` override in the file.** The PlanarK warm-TTFT bypass
  emits on `rmlx_kv_quant::warm_ttft` with `path = "warm_ttft_bypass"`. Three
  readers name that string: `crates/rmlx-cli/tests/e2e/runner.rs`,
  `crates/rmlx-models/tests/niah_long_context.rs` and
  [`KV_QUANT.md`](KV_QUANT.md). The override travels with the body, so moving
  it changes nothing — dropping or renaming it breaks all three.
* **The `kv_bytes` request-boundary event.** `scripts/bench/tri_engine_summarize.py`
  reads it out of a run's stderr; the whole cross-engine comparison rests on
  that column.
* **The SSD layout-key salt.** A bump invalidates every spilled block.
  **Bumping it is a STOP**: it is not part of this restructure, and a chunk
  that finds itself needing one has changed a layout rather than moved a body.

---

## 5. The real-model run

Deferred to the integration run, not owed by any single chunk.

It must show, for `gemma-4-e2b` and `Ternary-Bonsai-8B`, at 4k and 32k context,
`--max-tokens 200`, temperature 0:

* a byte-identical token digest for every `--kv-quant` spelling that reaches a
  model, against the same spelling on the "before" arm;
* decode TPS within 1 % per cell.

The **"before" arm is the commit immediately before chunk (a)** — not `main`,
and not the branch point, so that the comparison isolates the restructure from
everything else on the integration branch.

One caveat carries from §2: for the 18 decode-inert spellings a served digest
cannot see a store byte, so a green digest there is evidence about the
dispatch, not about the store. The store evidence for those cells is the
oracle, and only the oracle.
