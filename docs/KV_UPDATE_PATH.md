# The KV update path

How `KvCache` appends to a quantized KV store, how the per-codec bodies are
laid out, which width twins stay apart and why, and the CPU oracle that pins
every store byte. The codecs themselves are in `docs/KV_CODECS.md` and
`docs/KV_ROTATION_CODECS.md`.

## Layout

The update path is in `crates/rmlx-kv-quant/src/kvcache/`:

| File | Holds |
|---|---|
| `update.rs` | `update` and `exit_prefill`, which call the entries `KvStorage::view_mut` names, prefill and decode capacity bookkeeping, the bf16 decode mirror, the GPU-state and residency walks, helpers shared by two or more families, `storage_mismatch`, `warn_if_width_disagrees` |
| `update_rotor.rs` | Rotor decode and prefill bodies, GPU encode, ring sync, materialised tail |
| `update_iso.rs` | Iso bodies, the same parts |
| `update_turbo.rs` | TurboQuant bodies and `k8_turbo_v_knobs` |
| `update_affine.rs` | `k8v4`, `k8v8` |
| `update_planar.rs` | `planar`, `planar_k`, including the warm-TTFT bypass |
| `update_paged.rs` | The paged storage |
| `update_mixed.rs` | The two `Mixed` entries: the prefill body and the decode refusal |

The family files sit in `kvcache/`, not in `storage/`. The bodies are
`impl KvCache` methods and read `KvCache` fields that are `pub(super)` in
`kvcache::core`. `KvStorage::view_mut` in `storage` names the entries as
`pub(crate)` fn pointers and does not read those fields.

## Entries

`KvStorage::view_mut` names two entries for each storage variant, with one
signature each:

- `update`: `fn(&mut KvCache, &Array, &Array, Device) -> Result<(Array, Array)>`
  (`new_k`, `new_v`, `device`).
- `exit_prefill`: `fn(&mut KvCache, &Array, &Array, Device, i32) -> Result<()>`
  (`k_full`, `v_full`, `device`, `total_seq`). An entry ignores an argument it
  does not use. `exit_prefill_mixed` reads the dispatch policy from the cache.

The caller copies the entry out of the view, drops the view, then calls the
entry.

**`update` is keyed on the storage.** It reads the entry of the storage the
cache holds. After an SSD hydrate, an SWA layer holds `KvStorage::None` while
its `quant` is the model's codec, and `Paged` comes from a process-global
switch; both must take their own storage's entry. A decode step pays one view
build (one match, no allocation) and one indirect call.
`entry_routing_tests.rs` drives both shapes.

**`exit_prefill` is keyed on the codec.** It reads the entry of
`KvStorage::new(self.quant)`, the same key its `materialises_packed_store()`
gate reads. Building that storage allocates nothing (every slot is `None`),
and it runs once per layer per prefill. A storage of another family than the
codec reaches a body that returns `KvStorageMismatch`. A width disagreement
inside one family only warns (`warn_if_width_disagrees`).

`exit_prefill` has three guards before the entry. `None` storage returns with
the bf16 seeds as its storage. `Paged` returns with its compact seed. The gate
returns before any bulk encode for a codec whose decode reads only the bf16
mirror, and clears any payload the cache arrived with. The `exit_prefill` entry
of `None` and `Paged` is `exit_prefill_behind_guard`, which refuses. The entries
of the mirror family are unreachable through `exit_prefill` today and stay as
the re-enable path for a codec that grows a decode kernel over its own store.
`every_exit_prefill_entry_builds_a_store_on_its_own_storage` calls each entry
directly. This test fails when an entry and the gate disagree:
`warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`.

After a hydrate, a non-`None` storage has the same storage variant as the one
its codec builds, because each hydrated layer gets the codec the arch builder
gave that layer. `block_io_storage_family_tests.rs` in `rmlx-kv-ssd` holds this
for one codec, comparing the variant only. `ssd_boundary_codec_tests.rs` in
`rmlx-models` holds it for the boundary layers by decoding them.
`a_boundary_layer_that_builds_a_store_keeps_the_base_storage` holds a policy
fact of `kv_layer_quants`, not of the hydrate: a boundary codec that builds a
store builds the base's storage variant.

The `update` entry of `Mixed` is `update_mixed`, which refuses. The `Mixed`
per-step append is `update_and_sdpa_mixed` in `sdpa.rs`.

## In-place update

A step writes its K and V rows into buffers of many positions with
`slice_update`. MLX writes in place only when nothing else owns the old buffer
when MLX evaluates the update (`mlx/backend/common/copy.h:25-48` and
`mlx/array.h:304-306`, MLX v0.32.3). If something owns it, MLX allocates a new
buffer and copies the full buffer, at each step. Two kinds of owner cause this:

- **A handle that the cache keeps.** The cache replaces its handle with the
  result of `slice_update` before it evaluates that result. An evaluation
  before the replacement copies the buffer at each step.
  `crates/rmlx-kv-quant/src/kvcache/in_place_update_tests.rs` holds this on the
  CPU for the bf16 mirror of each codec that decodes from it, on a layer that
  does not share its KV, for the raw prefill buffer and for the ring.
- **GPU work that is not complete.** MLX keeps the buffer of each input of a
  command buffer until the GPU completes that command buffer: the completion
  handler holds them (`mlx/backend/metal/eval.cpp:47-68`), and the queue makes
  command buffers with unretained references (`device.cpp:323`, `:559`). The
  last reader of a K/V buffer in step t is the attention of its layer and, on
  a layer whose K/V later layers share, the attention of each of those layers.
  If the CPU encodes the update of step t+1 before the GPU completes that last
  reader, MLX copies.

The second owner depends on how far the CPU is ahead of the GPU. The decode
loop encodes step t+1 while the GPU runs step t. Metal stops the CPU when a
queue holds 64 command buffers that are not complete
(`mlx/backend/metal/device.cpp:315` makes the queue with the default limit),
and each `async_eval` commits one or more command buffers. Thus the update
stays in place when more than 64 command buffers separate the last reader from
the update of step t+1. The cache evaluates the K buffer and the V buffer with
one `async_eval` each. On Ternary-Bonsai-8B (36 layers) that distance is then
one step, and each update stays in place. With one `async_eval` for both
buffers, MLX copies them on that model: decode is 22 % slower and the Metal
allocation is 497 MB larger (`scripts/perf_ab.sh`, 8 slots).

On gemma-4-e2b the distance is shorter for layers 13 and 14, whose K/V layers
15 to 34 read, and those two layers copy. A per-layer tally of one 100-token
decode at the default KV quant, with one `async_eval` per buffer:

| MLX | Layers 13 and 14 | Layers 0 to 12 |
|---|---|---|
| 0.32.3 | 353 of 392 updates copied | none of 2548 copied |
| 0.31.2 | 392 of 392 copied | none of 2548 copied |

No CPU test sees this owner, because it depends on GPU timing.
`metal_gen_alloc_mb` sees the copies only when the copied buffers are large:
on gemma-4-e2b it shows 1096.1 MB with 2546 copies and with 388.

## One body per store shape

24 of the 27 `KvStorage` variants hold store slots. 19 hold a K
slot and a V slot; `Planar`, `RotorKAsym3` and `RotorKAsym4` also carry a
scalar knob. 5 hold a K slot only, with V as bf16 on the parent cache:
`PlanarK`, `IsoKOnly3`, `IsoKOnly4`, `RotorKOnly3`, `RotorKOnly4`. `None` holds
no store, `Mixed` holds a `MixedKvState`, and `Paged` holds a block table.
`make kv-update-census` prints this classification from the enum.

Each shape is an entry plus a body. The entry destructures its `KvStorage`
variants and hands the store slots to the body. The body is a free function over
`&mut Option<KStore>`, plus `&mut Option<VStore>` where the shape has a V slot,
and nothing else of the cache. Where two variants differ only in code width, the
body is `<const BITS: u8>` and the entry resolves the width from the storage
variant. Where they differ in a scalar, the scalar is a parameter.

| Family | Decode entry → body | Prefill entry → body |
|---|---|---|
| rotor V | `update_rotor_v` → `rotor_v_update` | `exit_prefill_rotor_v` → `rotor_v_bulk_encode` |
| rotor sym | `update_rotor_sym` → `rotor_sym_update` | `exit_prefill_rotor_sym` → `rotor_sym_bulk_encode` |
| rotor K-only | `update_rotor_k_only` → `rotor_k_only_k_side` | `exit_prefill_rotor_k_only` → `rotor_k_only_bulk_encode` |
| rotor K-asym | `update_rotor_k_asym` → `rotor_k_asym_update` | `exit_prefill_rotor_k_asym` → `rotor_k_asym_bulk_encode` |
| iso V | `update_iso_v` → `iso_v_update` | `exit_prefill_iso_v` → `iso_v_bulk_encode` |
| iso sym | `update_iso_sym` → `iso_sym_update` | `exit_prefill_iso_sym` → `iso_sym_bulk_encode` |
| iso K-only | `update_iso_k_only` → `iso_k_only_k_side` | `exit_prefill_iso_k_only` → `iso_k_only_bulk_encode` |
| turbo sym | `update_tsym` → `tsym_update` | `exit_prefill_turbo_sym` → `tsym_bulk_encode` |
| K8 + turbo V | `update_k8_turbo_v` → `k8_turbo_v_update` | `exit_prefill_k8_turbo_v` → `k8_turbo_v_bulk_encode` |

The fused-append entries follow the same shape: `rotor_k_only_gpu_append`,
`rotor_sym_gpu_append`, `iso_k_only_gpu_append` and `iso_sym_gpu_append`, each
over a width-generic body.

**A mis-resolved width does not compile** where the slot's type carries
`BITS`: handing a `QuantRotorV<3>` slot to `rotor_v_update::<4>` is a type
error. A `QuantV` slot carries its width at runtime, in `bits`, so a wrong
width there compiles. That covers the V axis of `tsym_update`, of
`rotor_k_asym_update` and of `k8_turbo_v_update`. The four K8 + turbo V
spellings read `(v_bits, use_tcq)` from one table, `k8_turbo_v_knobs`, which
both entries share.

**The width comes from the storage.** A prefill entry encodes at the storage's
width even when the `KvQuant` spelling names another. Both entries come from
the storage variant, so the two halves agree. `warn_if_width_disagrees` warns when
they differ; it does not refuse.

**Three bodies stay apart.** `update_k8v4`, `update_k8v8` and `update_planar`
hold different V store types (`QuantV`, `QuantK`, `QuantPlanarV`). The three
share `append` and `dequantize_choice` but are built differently, and building
is the entry's job. A two-method trait over those three would fold them.

**A storage mismatch is a permanent error.** Every path that finds the wrong
`KvStorage` variant returns `storage_mismatch`, which builds
`Error::KvStorageMismatch`. The retry envelope does not replay it: the replay
would build the same wrong cache. Four decode bodies still panic instead:
`update_k8v4`, `update_k8v8`, `update_planar` and `update_paged`.

## Width-generic stores

The rotor, iso and TurboQuant K stores are one const-generic type each, with
the width-named spellings as aliases: `QuantRotorV<BITS>`, `QuantRotorK<BITS>`,
`QuantIsoV<BITS>`, `QuantIsoK<BITS>`, `QuantKTurbo<BITS>`. The shipped widths
are 3 and 4.

**The width guard.** `QuantIsoV`, `QuantIsoK` and `QuantKTurbo` carry the
associated const `WIDTH_IS_A_SHIPPED_ONE`, an `assert!` over `BITS`. `NAME`,
`new` and `from_cpu_blocks` read it, and so do `QuantKTurbo`'s two MSL width
selectors and every other width-dependent method of `QuantKTurbo`. A third
width fails at monomorphisation, so `cargo build` and `cargo test`
catch it; `cargo check` and `cargo clippy` do not. The rotor stores carry no
guard. They stay crate-internal under their generic names, and callers outside
`rmlx-kv-quant` name the aliases.

**What stays per width.** Each item below binds a per-width `.metal` entry
point, so folding it means folding the kernels:

- rotor: `rotor{3,4}_quant_kernel` and `rotor{3,4}_dequant_kernel`
  (`rotorquant_msl.rs`), `rotor{3,4}_fused_qk_sdpa` and
  `rotor{3,4}_fused_qk_dispatch_count` (`rotor_fused_qk_msl.rs`),
  `rotor{3,4}_flash_decode_dispatch_count` (`rotor_flash_decode_msl.rs`);
- iso: the four `isoquant_{quantize,dequantize}_iso{3,4}.metal` kernels and
  their dispatchers `isoquant_msl.rs` and `isoquant_msl_v4.rs`. Width dispatch
  has one home, `isoquant_msl_dispatch`;
- turbo: `turbo_k{3,4}_fused_qk.metal`, their headers and
  `turbo_k{3,4}_fused_qk_msl.rs`.

### Iso

`QuantIsoV` carries no `max_seq`: a cached window goes stale when the window
grows. The live window is the `KvStorage` variant's.

Both widths have `dequant_gpu` and `dequant_on`.
`iso_{v,k}{3,4}_dequant_gpu_matches_dequant_cpu` (`isoquant_msl_tests.rs`,
`isoquant_msl_v4_tests.rs`) hold each at 5e-3 per element and 1e-6 overall.

The fused iso decode arms accept a shape only when `iso_flash_shape_ok` does:
`b == 1`, and `head_dim` a multiple of 4, at most 512 and a power of two. On
any other shape the step falls through to `update()` and dequantizes through
`dequant_gpu`.

`QuantIsoV::append_gpu` writes a GPU-resident mirror only when
`gpu_resident_iso_enabled()` is true. In production that reads the constant
`GPU_RESIDENT_ISO_PRODUCTION`, which is `false`, so the mirror writes nothing.
`the_production_gpu_resident_iso_mirror_is_off` asserts on the constant.

### TurboQuant

Only `tsym3` and `tsym4` build the K-side `QuantKTurbo` store;
`only_the_symmetric_spellings_build_the_k_side_turbo_store` asserts it. All six
turbo spellings are decode-inert: a served prefill writes no packed store and
a seeded decode step returns through `update_decode_fp16`. Only an update
with no bf16 seed reaches `update_tsym`: the oracle's drive, a `Device::Gpu`
drive, or an SSD-hydrated cache resuming decode.

**The V device follows the width.** The V side is `QuantV` with a runtime
`bits`. `QuantV::append` enters its GPU branch on the device alone and refuses
`bits != 4` there. So `tsym_update` and `tsym_bulk_encode` hand the V axis the
caller's device at 4 bits and `Device::Cpu` at 3. They keep the `Array` the GPU
dequant returns, and build the f32 vector only on the CPU path. No CPU test
can see this rule. `turbo_v_axis_gpu_tests.rs` drives both widths on Metal:
the 3-bit cell fails on its first append if the rule is lost.

**The hydrated window.** `QuantKTurbo::from_cpu_blocks` takes `max_seq` at both
widths, and the SSD reader forwards the window the spill recorded. The field is
inert: nothing sizes a buffer from it, and the first GPU `append` overwrites
it. `crates/rmlx-kv-ssd/src/block_io_turbo_hydrate_tests.rs` pins the hydrated
payload field by field, and `the_two_widths_hydrate_the_same_window` holds the
window. The reader calls `.eval()` on the loaded tensors, like every other
reader in `block_io.rs`; the call changes no byte.

**The TCQ spellings write the plain bytes.** `k8vturbo3tcq` writes the same
store as `k8vturbo3`, and `k8vturbo2tcq` the same as `k8vturbo2`.
`crate::tcq::build_transition_table` gives every trellis state an edge for
every level, so the Viterbi path is the greedy nearest-centroid assignment.
`the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes` asserts the
flag and the identity together. Either the trellis is wrong or the two
spellings are one codec; that is open.

## The store-bytes oracle

`crates/rmlx-kv-quant/src/kvcache/store_bytes_tests.rs` pins every spelling in
`ALL_KV_QUANTS` at two shapes, `(kv_h, head_dim)`: `(1, 128)` and `(4, 96)`.
A spelling whose own group size does not divide 96 takes `(4, 128)` as its
second shape instead (`shapes_for`); today that is the mixed pair,
`mixed_k8g64_v4g64` and `rot_k_v8g64`. Its module doc is the full statement;
this is the summary. Per spelling and shape it pins:

| Column | What moves it |
|---|---|
| Store bytes after the bulk append | What the chunk encode writes |
| Store bytes after three decode steps | The per-step append |
| Store bytes after a truncate into the chunk | Either half of the truncate plan |
| The K and V rows the attention receives | What the store hands back |
| `resident_bytes()` | What the cell holds |

A restructure that moves a pin is a defect, not a re-baseline.

**Two drives, two populations.** `materialises_packed_store()` splits the
spellings, and `the_two_populations_partition_every_spelling` derives the split
from the predicate. `drive` appends through the decode dispatch for every
spelling; it is the one CPU route on which a decode-inert spelling writes a
store. For the materialising spellings, `drive_prefill` also brackets a chunk
with `enter_prefill` and `exit_prefill`. It pins the store the arm wrote,
`resident_bytes()`, and the rows of the first decode step
(`exit_prefill_bulk_encode_bytes_are_pinned_per_spelling_and_shape`).

The family files `rotor_store_bytes_tests.rs`, `iso_store_bytes_tests.rs` and
`turbo_store_bytes_tests.rs` keep the claims that are about one family: the
store geometry per width, the width-twin pairs checked by
`assert_width_twins_differ`, the TCQ identity, the turbo K scope anchor and the
iso mirror check. Each family census filters `ALL_KV_QUANTS` by `Display` text.
The turbo filter needs two tokens, `turbo` and `tsym`: `tsym3` and `tsym4`
contain no `turbo`.

**Why a served digest is not the oracle.** For a decode-inert spelling, a
build with a correct store and a build with an empty one emit the same tokens.
A served digest there judges the dispatch, never the store.

**What the CPU oracle cannot see:**

- `KvStorage::Paged`: no spelling builds it in a test process.
  `no_spelling_builds_the_paged_storage_in_this_process` fails if one starts
  to. The paged suite is `crate::paged`.
- The mixed pair at a non-power-of-two `head_dim`: its second shape is
  `(4, 128)`, because `MixedKvState` groups by 64.
- The `exit_prefill` arms of the decode-inert spellings: no CPU route reaches
  them.
- Every GPU path: MSL encode dispatch, the resident rings, `gpu_append`,
  `gpu_packed_view`, the ring-readback branch of the sync helpers, the fused
  flash-decode arms, the width dispatch to a kernel, and the turbo V-device
  rule. `make gpu-test` is the gate.
- `from_cpu_blocks` and `try_deep_clone`, pinned in `rmlx-kv-ssd`.
- The `layer_idx` a fused-append entry threads into a new rotor K table. A
  wrong table is self-consistent, and no test drives that entry with an empty
  K slot.

## Figures

The figures are printed by tools, not recorded here:

- `make kv-update-census`: the variant shapes, the `match` sites over
  `KvStorage` or `KvQuant` that name at least half the enum (and those with no
  catch-all arm, which a new codec must touch), and the `update_` bodies.
- `scripts/debt_report.sh --matched-lines <population>`: the duplication
  figure per population. The KV ones are `rotor-storage`, `iso-storage`,
  `turbo-storage`, `rotor-updates`, `iso-updates`, `turbo-updates`,
  `turbo-ssd` and `update-bodies`.

## The sites a new codec must touch

A new `KvQuant` variant (and its `KvStorage` variant) does not compile until
each exhaustive site below names it. Each stays for its reason:

| Site | Why it cannot move |
|---|---|
| `quant_descriptor.rs` `KvQuant::descriptor` | The one place that classifies a codec. The predicates, `Display` and the fieldless `FromStr` spellings read it. |
| `storage/kv_storage.rs` `KvStorage::new` | It builds one storage variant per codec. Data cannot name a variant. |
| `storage/kv_storage.rs` `KvStorage::view` | The one place where the concrete stores become `&dyn KvSlot` for read-only work. |
| `storage/kv_storage.rs` `KvStorage::view_mut` | The same for mutation, plus the `update` and `exit_prefill` entries. One view cannot be derived from the other without a macro. |
| `storage/kv_storage.rs` `KvStorage::try_deep_clone` | Building the twin needs the concrete variant. A trait method would need `dyn Any` downcasts, or a codec the storage cannot always state. |
| `rmlx-kv-ssd` `block_io.rs` `write_layer` | The SSD block format (tensor names, dtypes, trims) belongs to `rmlx-kv-ssd` and needs the concrete store types. |
| `rmlx-models` `kv_cache/cache_type.rs` `decompose_auto` | Per-codec policy (fallback, warning, panic, per-side tag) that no codec fact decides. |

One site compiles with a new variant and still must change: the layout-tag
table of `rmlx-kv-ssd` `block_io.rs` `read_layer`. It is the read side of
`write_layer`, for the same reason. Its keys are SSD layout tags, not codec
spellings (`none_bf16`, `mixed`, `paged`, the `_qjl` rotor tags), so the
descriptor spelling cannot drive it. A new codec without an arm there spills
and then fails to hydrate with `unknown layer tag`.

`FromStr` has no table of its own: it finds a fieldless spelling in
`ALL_KV_QUANTS` through the descriptor, an alias in `KV_QUANT_ALIASES`, and
keeps one structured parser per payload shape (`mixed_*`, `rot_k_v*`,
`rotor_k_{3,4}_asym_*`). The list of valid spellings in the
`KvQuantParseError::Unknown` text is written by hand and pinned by
`unknown_error_text_names_every_fixed_spelling`.

## What cannot move

- Every pin in `store_bytes_tests.rs` and the family files.
- The layout tags in `storage/kv_storage.rs`. A layout tag is written into the
  SSD index. The SSD layout-key salt: a bump invalidates every spilled block.
- The `KvQuant` spellings, their `Display` and `FromStr`, and the CLI names.
  `make check-kv-codec-disposition` reads the help text and the INERT banners
  against the runtime disposition.
- The `.metal` kernels and their `probes/kernels.manifest` entries, and
  `scripts/gpu_validation_census.txt`.
- The iso trace target `rmlx_kv_quant::kvcache::update_iso`, which
  `docs/PERF_BASELINE.md` § "Iso trace phases" names. An event without a
  `target =` override takes its module path, so moving a body moves its target.
- The warm-TTFT bypass event, target `rmlx_kv_quant::warm_ttft`, field
  `path = "warm_ttft_bypass"`. `crates/rmlx-cli/tests/e2e/runner.rs`,
  `crates/rmlx-models/tests/niah_long_context.rs` and
  `docs/KV_FUSED_KERNELS.md` read it.
- The `kv_bytes` request-boundary event, which
  `scripts/bench/tri_engine_summarize.py` reads from a run's stderr.
