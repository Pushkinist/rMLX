# The KV update path

How `KvCache` appends to a quantized KV store, how the per-codec bodies are
laid out, which width twins stay apart and why, and the CPU oracle that pins
every store byte. The codecs themselves are in `docs/KV_CODECS.md` and
`docs/KV_ROTATION_CODECS.md`.

## Layout

The update path is in `crates/rmlx-kv-quant/src/kvcache/`:

| File | Holds |
|---|---|
| `update.rs` | The `KvStorage` and `KvQuant` dispatch, `exit_prefill`, prefill and decode capacity bookkeeping, the bf16 decode mirror, the GPU-state and residency walks, helpers shared by two or more families, `storage_mismatch`, `warn_if_width_disagrees` |
| `update_rotor.rs` | Rotor decode and prefill bodies, GPU encode, ring sync, materialised tail |
| `update_iso.rs` | Iso bodies, the same parts |
| `update_turbo.rs` | TurboQuant bodies and `k8_turbo_v_knobs` |
| `update_affine.rs` | `k8v4`, `k8v8` |
| `update_planar.rs` | `planar`, `planar_k`, including the warm-TTFT bypass |
| `update_paged.rs` | The paged storage |
| `update_mixed.rs` | The `Mixed` prefill body |

The family files sit in `kvcache/`, not in `storage/`. The bodies are
`impl KvCache` methods and read `KvCache` fields that are `pub(super)` in
`kvcache::core`. `storage` is a leaf of `kvcache` and cannot see them.

`exit_prefill` returns at its `materialises_packed_store()` gate, before any
bulk-encode arm. It also clears any payload the cache arrived with. A spelling
that reports `false` gets no packed store from a served prefill; its decode
reads the bf16 mirror. Its `exit_prefill` arm is unreachable today and is kept
as the re-enable path for a codec that grows a decode kernel over its own
store. This test fails when an arm and the predicate disagree:
`warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`.

Two `exit_prefill` cases stay inline. `KvQuant::None` reads the raw buffers
the function owns and returns early, promoting them into the bf16 mirror.
`KvStorage::Paged` has no arm: its seed is an early return above the gate.

`KvCache::update` refuses a `Mixed` cache. The `Mixed` per-step append is
`update_and_sdpa_mixed` in `sdpa.rs`.

## One body per store shape

24 of the 27 `KvStorage` variants hold a K slot, a V slot and `max_seq`.
`None` holds no store, `Mixed` holds a `MixedKvState`, and `Paged` holds a
block table. `make kv-update-census` prints this classification from the enum.

Each shape is an entry plus a body. The entry destructures its `KvStorage`
variants and hands the two store slots to the body. The body is a free function
over `&mut Option<KStore>` and `&mut Option<VStore>` and nothing else of the
cache. Where two variants differ only in code width, the body is
`<const BITS: u8>` and the entry resolves the width from the storage variant.
Where they differ in a scalar, the scalar is a parameter.

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

**A mis-resolved width does not compile.** The slot's type names the width, so
handing a `QuantRotorV<3>` slot to `rotor_v_update::<4>` is a type error. This
holds for every const-generic body. It does not hold for `k8_turbo_v_update`,
whose width is the runtime `v_bits` of a `QuantV`. Its four spellings read
`(v_bits, use_tcq)` from one table, `k8_turbo_v_knobs`, which both entries
share.

**The width comes from the storage.** A prefill entry encodes at the storage's
width even when the `KvQuant` spelling names another. Decode dispatches on the
storage too, so the two halves agree. `warn_if_width_disagrees` warns when
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
selectors.
A third width fails at monomorphisation, so `cargo build` and `cargo test`
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
`ALL_KV_QUANTS` at two shapes: `(kv_h, head_dim)` of `(1, 128)` and `(4, 96)`.
Its module doc is the full statement; this is the summary. Per spelling and
shape it pins:

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
- The mixed pair's second shape is `(4, 128)`: `MixedKvState` groups by 64.
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
