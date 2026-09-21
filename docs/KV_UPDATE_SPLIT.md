# Splitting the KV update path

`crates/rmlx-kv-quant/src/kvcache/update.rs` holds every codec's update path in
one file. This document is the plan to split it, and the record of what was
measured before the split starts.

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
  and the guard named above is what holds them to the predicate.
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

The target of 3 or fewer is about the six sites in `update.rs`, which the split
owns. The nine in `quant.rs` are the enum's own `Display`, `FromStr` and
disposition predicates; they are not this restructure's to remove.

**Recall.** `scripts/kv_update_census_selftest.sh`, 35 cases over planted
trees, run by `make kv-update-census-selftest` and by `make ci`. A site in a
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
lines, or carries a `LOC-exempt` marker whose text says why. The interim marker
on `update.rs` goes with the split. The four other oversized files in the crate
are outside this work.

### Duplication

`python3 scripts/lib/debt_report.py --matched-lines update-bodies` is an
eleventh `debt-report` population: every `update_`-prefixed fn of the update
file, paired every item with every other. The three family populations each
read one codec's twins and go blind the moment those widths collapse; this one
reads the shape the restructure writes once.

One producer, not two. The census and this population print the same two
numbers, so both read fns through `lib/debt_report.py`'s `extract_fns`;
`lib/rust_scan.py` reads enums and `match` arms only.

**Before:** 3585 matched lines over 1440 body lines, 25 items, 300 pairs.
(Before the three collapses: 12122 over 2539, 33 items, 528 pairs.)

Recall: six cases in `scripts/debt_report_selftest.sh` — the planted population
held to its figure and its item and pair count, one body left reading a
measured `0` with the population still found, and every body renamed out of the
prefix reading `unavailable`.

---

## 4. The chunks

One branch, one pull request, chunks as commits. Each chunk names its own
removals.

### Chunk (a) — split by codec family

Move each family's update path next to its storage type: the rotor bodies with
the rotor storage, the same for iso, turbo, planar, affine and paged. The file
stops being "every codec's update" and becomes a dispatcher over per-family
modules.

Removes: the moved bodies from `update.rs`; the interim `LOC-exempt` marker on
any file that ends at or under 1000 lines; any helper in `update.rs` left with
no caller after the move.

Judged by: the oracle green with no re-baselined pin; the match-site count per
file; `make file-size-report`.

### Chunk (b) — one update body per store shape

24 of the 27 variants carry the same store-slot shape and their bodies are the
same sequence: cast to bf16, ensure the store, append K, append V, dequant or
read the mirror, run SDPA. Write that sequence once.

Removes: every per-variant body the shared one replaces. The chunk's commit
message names each one.

Judged by: the same oracle, plus the `update-bodies` duplication figure, which
must fall and must stay a measured figure rather than an empty population.

### Constraints

These are the orchestrator's, and they bind chunk (b) hardest.

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
  names that string and the phases under it (`iso3_encode`,
  `iso3_dequant_cpu`, `iso3_vec_to_array`). Moving a body to another module
  moves its target. A chunk that moves an instrumented body updates that doc
  in the same commit, or keeps the target explicit.
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
