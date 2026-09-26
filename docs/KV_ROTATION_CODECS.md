# iso and rotor KV codecs

This doc gives the storage and the decode path of the iso and rotor KV codecs
and of their K-side variants.

The other KV quantization docs: [`KV_QUANT.md`](KV_QUANT.md) (the contract:
CLI flags, the auto default, bit rates, byte accounting, hot-swap, codec
disposition, public API); [`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md) (which
codec each layer gets); [`KV_CODECS.md`](KV_CODECS.md) (storage per
`KvStorage` variant, TurboQuant calibration);
[`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) (fused-QK, fused flash-decode,
the dispatch axis, sparse attention);
[`KV_STORE_TRUNCATION.md`](KV_STORE_TRUNCATION.md) (`truncate_to` per store);
[`KV_CODEC_FIDELITY.md`](KV_CODEC_FIDELITY.md) (measured codec fidelity).

---

## iso and rotor codecs

### iso3 codec

> **INERT on this build** — `iso3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**Algorithm — quaternion SO(4) isoclinic rotation.** iso3 applies a
left-isoclinic SO(4) rotation to each group of 4 elements, then quantizes to 3
bits:

```
T(v) = q_L * v     (one quaternion per group)
```

`*` is the **Hamilton product**, and `v ∈ ℝ⁴` is a quaternion
`v = (v₀, v₁, v₂, v₃)`. The inverse is `T⁻¹(r) = q̄_L * r`.

**Encode (per token):**

1. Compute the L2 norm of the vector and store it. Divide the vector by it.
2. Split the vector into `head_dim / 4` quaternion groups.
3. Apply `r = q_L * v` with the scalar Hamilton product.
4. Compute a per-group scale: `max(|r_i|) / max_centroid`.
5. Find the nearest centroid of the 3-bit Lloyd-Max codebook.
6. Write the codes into the dense code plane of the row, 3 bits each
   (`crate::code_plane`).

**Decode:** unpack → centroid lookup → rescale → inverse rotation → multiply by
the norm.

**Iso memory truth.** iso spends, per 4-element group, four codes in the row's
dense code plane (`bits` each — see `crate::code_plane`) **and** one scale at
the ring's sideband dtype (`bf16`), plus one `bf16` norm per token. At
head_dim=128 that is 114 B per token per kv_head for iso3 against bf16's 256 B
(**7.125 bits per value, 0.445× bf16**) and 130 B for iso4 (**8.125**, 0.508×).

The rate changes with `head_dim`. The allocation is
`ceil(D·bits/32)·4 B` codes + `(D/4)·2 B` scales + `2 B` norm, thus:

```
iso stored bits/value = bits + 4 + 16/head_dim
```

For iso3 this is 7.125 at D=128, 7.0625 at D=256 and 7.03125 at D=512. It comes
from `QuantKGpuRing::alloc` (`Dtype::U32` codes, `KV_SIDEBAND_DTYPE` scales and
norms, one scale per group and one norm per token per KV head).
`ring_bytes_match_independent_geometry` and `kv_rate_tests` measure it against
the allocation.

**The scale is the largest term after the codes:** 4.000 bits per value,
against 3.000 bits of codes at iso3. The CPU encoders round each scale and norm
to the stored width before they select the codes. Thus the encoder and the
store agree, and a ring seeded from CPU blocks decodes the same bits as those
blocks.

The CPU `IsoBlocks` form adds a 4×f32 quaternion per group and holds its scale
and norm planes at `f32`. This makes ≈692 B per token (≈43.25 bits per value,
2.7× bf16). The quaternion is the constant `FIXED_QUAT` repeated per group, not
data. The GPU ring that the K-only and symmetric codecs decode from does not
hold it. Three cases:

- The V-only `iso3` / `iso4` codecs decode from the bf16 mirror. `exit_prefill`
  builds no store for them, and they measure the same bytes as `none`
  (`docs/KV_QUANT.md` §"Codec disposition", Class 2).
- The store-reading members, `k_iso3/4` and `iso3_sym/4_sym`, hold the block
  form for one window: `exit_prefill` encodes them on the CPU, into blocks. The
  first fused decode step seeds the ring from those blocks and frees them
  (`drop_blocks_when_ring_live_iso_*` in `kvcache/update_iso.rs`).
- After that step, the ring is the only resident copy.

`estimated_resident_bytes_per_layer` sizes the four store-reading members from
the ring. The estimate is low in two cases. The first case is the window
between `exit_prefill` and the first fused decode step. The second case is the
full request on a layer whose shape the fused path rejects: batch > 1, or a
`head_dim` that is not a power of two at most 512. In the second case the ring
is never allocated and the blocks stay.

**Both rings are under bf16 at every head_dim, and iso is under it by more.**
The code plane costs `bits` per value for iso and `bits·⌈D/3⌉·3/D` for rotor:
3.000 and 3.250 at `bits = 3`, `D = 128`. The sideband makes the difference:
one scale per group at `KV_SIDEBAND_DTYPE` is 4.000 bits per value for a group
of 4 values, and 5.375 for a group of 3.
`every_store_family_is_at_or_below_the_bf16_floor_or_exempt` and
`exempt_families_actually_exceed_the_floor`
(`crates/rmlx-kv-quant/src/kv_rate_tests.rs`) pin both signs from real encoder
bytes. The operator sees the same sign in the resolve-time net-negative warn
(`docs/KV_LAYER_POLICY.md` §"Per-layer net-benefit decision + net-negative warn") and in
`rmlx info --list-cache-types`.

**Crate-wide rate ceiling.** `crates/rmlx-kv-quant/src/kv_rate_tests.rs`
reports the bits per value of each store family. It fails a family above
bf16's 16.0 that has no written exemption. It also fails an exempt family that
does not measure above the floor. The `KvQuant` → family map is an exhaustive
`match`, so a new variant does not compile until it names its families. The
list of measured representatives is kept by hand: a variant with no
representative is not measured, and the gate stays green. Review must catch
this. Table at `head_dim = 128`:

| Family | Stored bits / value | Provenance | Verdict |
|---|---|---|---|
| turbo2 / tcq2 | 3.00 | measured | under |
| turbo3 / tcq3 | 4.00 | measured | under |
| turbo4 | 5.00 | measured | under |
| q8 (group 128) | 8.25 | measured | under |
| affine (`CacheType::Q8G32`) | 9.00 | layout formula, measured sideband | under |
| **iso3** (GPU ring) | **7.12** | measured | under |
| **iso4** (GPU ring) | **8.12** | measured | under |
| **rotor3** (GPU ring) | **8.75** | measured | under |
| **rotor4** (GPU ring) | **9.75** | measured | under |
| bf16 | 16.00 | by definition | the floor |
| **planar3 / planar4** | **22.00** | measured | exempt |

Planar is exempt because its scale is per *pair*. The `IsoBlocks` host form
measures 43.25; no served request stays in that form.

"Measured" means the byte total that the store reports for the buffers its own
encoder made from a shared fixture. For iso and rotor that is
`QuantKGpuRing::byte_size`, read from a real seeded ring. bf16 and MLX affine
have no CPU encoder in this crate. bf16 is two bytes per value. MLX affine is
`bits + 32/group`: the code bits plus one scale and one bias per group, each at
the dtype of the KV stream. **That sideband is 32 bits, measured**: the stream
at the store is bf16 (`cast_store_bf16` floors it at the store boundary), and
`mx.quantize(mode = "affine")` writes both scalars at the input dtype.
`affine_sideband_is_thirty_two_bits_per_group`
(`crates/rmlx-kv-quant/src/quant_tests.rs`) reads the figure from a real
`MixedTuple`.

**The affine row covers the whole `mixed_*` grammar.** The row uses the widest
cadence the grid accepts (`q8_g32`, 9.00). `parse_kv_side` output goes through
`validate_mixed_side` (2/3/4/5/6/8 bits, group 32/64/128), so this table has a
rate for every spelling that parses.
`mixed_grammar_admits_only_bounded_affine_rates` pins this.

**`head_dim % 4 == 0` constraint.** iso3 uses groups of 4. The encoder and the
decoder reject a `head_dim` that is not a multiple of 4 with
`IsoQuantError::HeadDimNotMultipleOf4`.

**Fixed quaternion.** The codec applies the golden-ratio unit quaternion
`q = (1, φ, φ−1, 1) / ‖(1, φ, φ−1, 1)‖` (`φ = (1+√5)/2`) to every group
(`FIXED_QUAT`). This is the quaternion of `multi_turboquant/methods/isoquant.py`.
It needs no calibration.

**Codebook — Gaussian Lloyd-Max, not Beta Lloyd-Max.** The Python references
(`rotorquant/turboquant/lloyd_max.py`) derive a Lloyd-Max codebook for the Beta
distribution that a random rotation of a unit vector gives. rMLX uses
`turboquant::lloyd_gaussian_codebook(3)` (Lloyd-Max for N(0,1)), the same
codebook family as TurboQuant and PlanarQuant. For `head_dim ≥ 64`, Beta(d) →
N(0, 1/d), and the per-group scale normalizes to N(0,1) before the centroid
lookup.

**GPU dispatch.** When `device == Device::Gpu`, `update_iso_v`,
`update_iso_sym` and `update_iso_k_only` send the encode and the dequant through
`isoquant_msl_dispatch`, which selects the kernel from the `BITS` of the store.
`QuantIsoV::dequant_gpu` / `QuantIsoK::dequant_gpu` put the CPU block payloads
(codes, scales, quaternions, per-token norms) into single byte buffers, upload
them once with `Array::from_bytes`, run the dequant kernel and reshape the
output to `[B, kv_h, S, D]`. On `Device::Cpu`, the CPU codec runs. The
GPU-resident `QuantIsoV` mirror is off in production:
`gpu_resident_iso_enabled()` returns `false`.

**CPU ↔ GPU parity.** `iso_v3_dequant_gpu_matches_dequant_cpu` and
`iso_k3_dequant_gpu_matches_dequant_cpu` in
`crates/rmlx-kv-quant/src/isoquant_msl_tests.rs` (`#[ignore]`-gated).
Observed `max|cpu-gpu| ≤ 2.4e-7` on the LCG fixture
(a few f32 ULPs from a different summation order between CPU `iso_decode_fast`
and the MSL kernel). The tests assert 5e-3 per element (codebook tolerance)
and a strict ≤ 1e-6 bound.

**Cosine gates.** `iso3_cosine_gate` in
`crates/rmlx-kv-quant/src/isoquant_tests.rs`. `quant_iso_v_roundtrip_dequant`
asserts that the `QuantIsoV3` round-trip matches `iso_decode_fast` within
`max_abs_err < 1e-3`.

**Sequence-major buffer layout (all Iso / Rotor block stores).**
`QuantIsoV<BITS>`, `QuantIsoK<BITS>`, `QuantRotorV<BITS>` and `QuantRotorK<BITS>`
add one `*Blocks` entry per `append` and concatenate them on `dequant`. The
caller reshapes the concatenation head-major `[B, kv_h, S, D]`. Thus each
`append` reorders the head-major chunk heads↔seq (`[B, new_seq, kv_h, D]`)
before it encodes. `dequant` reorders **each block back at its own sequence
offset** (`seq_layout::transpose_chunked_seq_heads`). One reorder over the
whole concatenation gives the same result only at `B == 1`.

The codec is positional per token row, so the sidebands stay with their rows:

- The Iso per-(token, group) scale and norm and the constant `FIXED_QUAT` move
  with the rows.
- The Rotor static rotor table and the QJL projection are keyed by group or
  projection. They do not move.
- The per-token QJL `qjl_codes` / `qjl_norms` move with the rows.

A `.kvb` SSD block stores its token rows in this sequence-major order. See
`docs/KV_CACHE.md` §5.7.3.

---

### iso4 codec

> **INERT on this build** — `iso4` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**Algorithm — quaternion SO(4) isoclinic rotation, 4-bit codebook.**

iso4 is the 4-bit form of [iso3](#iso3-codec). The rotation, the group
geometry and the fixed quaternion are the same. The differences are the
codebook (16 centroids, not 8) and the code width (4 bits, not 3).

| Property | iso3 | iso4 |
|---|---|---|
| Code bits / element | 3 | 4 |
| Delivered bits / element, ring-resident (`k_iso*`, `*_sym`) | **7.125** (114 B/token at head\_dim=128, see "Iso memory truth" in the iso3 section) | **8.125** (130 B/token) |
| Delivered bits / element, CPU-blocks form | ≈43.25 (≈692 B/token at head\_dim=128, with the constant quaternion sideband). The ring-backed members hold it between `exit_prefill` and the first fused decode step. `iso3` / `iso4` build no store (`docs/KV_QUANT.md` §"Codec disposition", Class 2) | ≈44.25 |
| Codebook | `lloyd_gaussian_codebook(3)` (8 centroids) | `lloyd_gaussian_codebook(4)` (16 centroids) |
| Pack density | dense code plane, 3 bits per code | dense code plane, 4 bits per code |
| Rotation | golden-ratio fixed quaternion (`FIXED_QUAT`) | same |
| Group size | 4 elements (one quaternion block) | same |
| `head_dim` constraint | `% 4 == 0` | same |
| MSL kernel | `iso_quantize_v3_gpu` / `iso_dequantize_v3_gpu` in `crates/rmlx-kv-quant/src/isoquant_msl.rs` | `iso_quantize_v4_gpu` / `iso_dequantize_v4_gpu` in `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs` |
| SSD layout tag (V-only / symmetric) | `iso_v_3` / `iso_sym_3` | `iso_v_4_v2` / `iso_sym_4_v2` |

**Codebook** — the same as iso3: Gaussian Lloyd-Max `lloyd_gaussian_codebook(4)`,
not Beta Lloyd-Max.

**MSL kernel.** `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs` dispatches the
4-bit kernel pair `iso_quantize_v4_gpu` / `iso_dequantize_v4_gpu`: one thread
per (token, group), atomic-OR pack with `(idx & 0xF) << shift`, and a boundary
table of 15 mid-points from `lloyd_gaussian_codebook(4)`. The bodies are in
`src/metal/isoquant_quantize_iso4.metal` and
`src/metal/isoquant_dequantize_iso4.metal`. `make check-metal-compiles` compiles
them against the header snapshot `src/metal/probes/isoquant_iso4.hdr.metal`.
The three iso update entries reach this width through `isoquant_msl_dispatch`
when `device == Device::Gpu`. The CPU codec runs on `Device::Cpu`.

**CPU ↔ GPU parity.** Three tests in
`crates/rmlx-kv-quant/src/isoquant_msl_v4_tests.rs`, all `#[ignore]`-gated (run
`cargo test -p rmlx-kv-quant -- --ignored isoquant_msl_v4 --test-threads=1`).
`iso_v4_msl_matches_cpu_within_eps` asserts that the CPU codec
(`iso_encode_fast` + `iso_decode_fast`, `bits=4`) and the MSL kernels agree
within 5e-3 on a 32×128 LCG fixture. `iso_v4_dequant_gpu_matches_dequant_cpu`
and `iso_k4_dequant_gpu_matches_dequant_cpu` assert the same on the store
entries that a decode step calls, at the two bounds of the 3-bit pair: 5e-3 per
element and a strict `max|cpu-gpu| ≤ 1e-6`.

**Cosine gate.** `iso4_cosine_gate` in
`crates/rmlx-kv-quant/src/isoquant_tests.rs`. SSD round-trip: `roundtrip_iso4`
in `crates/rmlx-kv-ssd/src/block_io_tests.rs` asserts that all four V buffers
(codes_packed, scales, quaternions, norms) are bit-identical after hydrate.

**One store type, two widths.** The CPU encode and decode functions take
`bits ∈ {3, 4}` (the shared dense code plane, `crate::code_plane`), and so does
the storage struct: one `QuantIsoV<BITS>` and one `QuantIsoK<BITS>`, with
`QuantIsoV3` / `QuantIsoV4` / `QuantIsoK3` / `QuantIsoK4` as type aliases.
`IsoBlocks` is shared.

---

### rotor3 codec

> **INERT on this build** — `rotor3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**Algorithm — Cl(3,0) Clifford rotor sandwich, 3-bit codebook.**

rotor3 is the 3-bit member of the Clifford rotation family of KV codecs. The
codec puts each `head_dim`-element V vector into Cl(3,0) (the 8-dimensional
multivector algebra of 3D Euclidean space) in groups of 3 grade-1 elements. It
applies a per-(layer, head) static rotor `R_g` as a sandwich, then quantizes to
3 bits with the Lloyd-Max N(0,1) codebook. The codec stores the static rotor
once per layer, for all tokens.

| Property | rotor3 |
|---|---|
| Delivered bits / element | **8.75** at head\_dim=128: 140 B/token/kv\_head against 256 B for bf16 (**0.547× bf16**). The split is **3.25 codes + 5.375 scales + 0.125 norm**. 43 groups cover a 128-element row. Each group stores its three grade-1 codes in the dense code plane of the row. Each group also stores one scale at `KV_SIDEBAND_DTYPE`. The allocation gives `rotor stored bits/value = (32·⌈⌈D/3⌉·3·bits/32⌉ + 16·⌈D/3⌉ + 16) / D`: 8.75 at D=128, 8.5625 at D=256. This comes from `QuantKGpuRing::alloc`. `rotor_rate_splits_into_documented_code_scale_and_norm_bits` measures it against a seeded ring. **The scale is the largest term.** Rotor shares one scale across 3 values, iso across 4. Thus rotor is the wider of the two families at every width and every head\_dim (`iso_and_rotor_k_codecs_are_under_the_floor_at_every_geometry`). |
| Code budget | **Only the 3 grade-1 codes per group are stored.** A rotor sandwich keeps the grade. Thus 3 values put in as the grade-1 part leave the scalar, the three bivector and the pseudoscalar slots at zero on encode. On decode, the inverse sandwich keeps every part that is not grade 1 out of the vector. `clifford_tests::sandwich_of_grade1_in_3d_stays_grade1` (encode side) and `clifford_tests::inverse_sandwich_of_non_grade1_leaks_nothing_into_grade1` (decode side) pin this. |
| Codebook | `lloyd_gaussian_codebook(3)` (8 centroids), one codebook for all 8 multivector components |
| Pack density | dense code plane, 3 bits per code, 3 codes per group |
| Rotation | static per-(layer, head) rotor table `[n_groups, 4]` in `[s, b12, b13, b23]` form, seeded from `ROTORQUANT_GLOBAL_SEED ^ (layer << 32) ^ (head << 16) + group` (`crate::clifford::rotor_seed`) |
| Group size | 3 elements (one Cl(3,0) grade-1 group) |
| `head_dim` constraint | none: the encoder pads a `head_dim % 3 != 0` tail with zeros and the decoder masks it off |
| MSL kernel | `rotorquant_msl.rs`: V-side encode, and K-side when QJL is off |
| SSD layout tag | `rotor_v_3` |

**One codebook.** The Python reference (`rotorquant/turboquant/rotorquant.py`)
uses a grade-aware codebook split (separate `vector` and `trivector` codebooks
at different bit budgets). rMLX uses **one 8-centroid codebook** for all 8
multivector components.

**Per-layer rotor tables.** `KvCache` has a `layer_idx: usize` field. Each arch
builder sets it with `KvCache::with_quant_max_seq(…).with_layer_idx(i)`. The
rotor3/rotor4 codec constructors (`QuantRotorV3::new`, `QuantRotorV4::new`,
`QuantRotorK3::new`, `QuantRotorK4::new`) get `self.layer_idx as u32` at each
`exit_prefill` and decode-time creation site. The `(layer << 32)` term in
`crate::clifford::rotor_seed` gives each layer a different rotor table.

**No QJL residual (V-only codec).** The Python reference has an optional 1-bit
QJL sign-quantization residual on the K side. The V-only rotor3 codec does not
apply it (see § "rotor K-side variants").

**Sign errors in the references.** The Python `clifford.py::geometric_product`
and the `rotor_fused.metal::gp_rotor_mv` kernel have sign errors in the grade-2
and grade-3 component formulas (for example, `e23 * e1 = +e123` in the Cl(3,0)
multiplication table, but the Python formula gives `-e123`). The Rust port
computes a table-driven dense geometric product at compile time from the
algebra rules. The algebra tests in `clifford_tests.rs` (known-answer 90°
rotation, unit rotor identity, sandwich of grade 1 stays grade 1) validate it.

**MSL kernel.** `crates/rmlx-kv-quant/src/rotorquant_msl.rs` has GPU encode and
decode kernels for rotor3 and rotor4. The kernel applies the Cl(3,0) sandwich
as a closed-form 3×3 SO(3) rotation matrix `M(R)` from `R * mv * R̃` (for
grade-1 input the grade-2 and grade-3 components cancel). The per-(layer, head,
group) rotor table is a buffer argument (`rotors_in : f32 [n_groups, 4]`).
`update_rotor_v`, `update_rotor_sym`, `update_rotor_k_only` and
`update_rotor_k_asym` (one entry per family for both code widths) dispatch it
when `device == Device::Gpu`. The CPU encoder runs on `Device::Cpu`.

**K-side QJL.** The K-side rotor codecs can carry a 1-bit QJL residual that
needs the JL projection matrix `S` at dequant. The GPU dequant kernels in
`rotorquant_msl.rs` do not implement QJL. When
`crate::rotor_qjl::rotor_qjl_enabled()` is `true` (`--rotor-qjl on`), the K-side
append and decode use the CPU `rotor3_k_encode` / `rotor3_k_decode` path. With
QJL off (**the default**), the GPU K-side kernel runs.

**CPU ↔ GPU parity tests.** `crates/rmlx-kv-quant/src/rotorquant_msl_tests.rs`
asserts max-abs-error ≤ 5e-3 between the CPU `rotor3_encode` / `rotor4_encode`
round-trip and the MSL round-trip (the same tolerance as iso3 / iso4). The tests
are `#[ignore]`-gated:
`cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1`.

**Cosine gate.** `rotor3_cosine_gate` in
`crates/rmlx-kv-quant/src/rotorquant_tests.rs`.

**SSD round-trip.** `roundtrip_rotor3` in
`crates/rmlx-kv-ssd/src/block_io_tests.rs` asserts that all four V buffers
(`codes_packed`, `scales`, `norms`, `rotors`) are bit-identical after hydrate.
The block stores the rotor table with the per-token payload, so a change to
`ROTORQUANT_GLOBAL_SEED` does not change a hydrated cache.

**Paged-KV routing.** rotor3 does not use the PagedAttention block-table path.
`PagedKStorage` is q8 only and `PagedPlanarVStorage` is PlanarQuant only.

---

### rotor4 codec

> **INERT on this build** — `rotor4` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**Algorithm — Cl(3,0) Clifford rotor sandwich, 4-bit codebook.**

rotor4 is the 4-bit member of the Clifford rotation family. The algebra and the
rotor sandwich are the same as rotor3. Only the codebook and the code width are
different:

| Property | rotor4 |
|---|---|
| Delivered bits / element | **9.75** at head\_dim=128 (156 B/token/kv\_head, **0.609× bf16**): 4.25 codes + 5.375 scales + 0.125 norm |
| Codebook | `lloyd_gaussian_codebook(4)` (16 centroids), one codebook for all 8 multivector components |
| Pack density | dense code plane, 4 bits per code, 3 codes per group |
| Rotation | the same static per-(layer, head) rotor table as rotor3 (`[n_groups, 4]`), from the same `ROTORQUANT_GLOBAL_SEED` formula |
| Group size | 3 elements (the same Cl(3,0) grade-1 group as rotor3) |
| `head_dim` constraint | none: the same tail padding as rotor3 |
| MSL kernel | `rotorquant_msl.rs`, shared with rotor3 through `rotor_quantize_v{3,4}_gpu` / `rotor_dequantize_v{3,4}_gpu` |
| SSD layout tag | `rotor_v_4` |

**One type, two widths.** `QuantRotorV4` and `QuantRotorV3` are aliases of
`QuantRotorV<4>` and `QuantRotorV<3>`: one const-generic store that encodes and
decodes through `rotor_encode` / `rotor_decode` at its own `BITS`. `RotorBlocks`
is shared.

**Tests.** `rotor4_cosine_gate` in
`crates/rmlx-kv-quant/src/rotorquant_tests.rs`;
`rotorquant_msl_tests.rs::rotor_v4_msl_matches_cpu_within_eps`;
`roundtrip_rotor4` in `crates/rmlx-kv-ssd/src/block_io_tests.rs` (all four V
buffers bit-identical after hydrate, rotor table stored with the payload).

**Paged-KV routing:** the same as rotor3. `RotorV4` does not use PagedAttention.

---

### iso K-side variants

Four variants apply the iso3 / iso4 codec to the K axis. The IsoQuant codec
(`iso_encode_fast` / `iso_decode_fast`) is **axis-agnostic**: the encoder takes
a flat `[B, kv_h, S, D]` row buffer and a per-row `head_dim`. Only the role on
the SDPA path and the SSD tensor names (`l{idx}.k.*` against `l{idx}.v.*`) make
it K or V.

| `KvQuant` | K codec | V codec | CacheType pair (`(K, V)`) | SSD layout tag |
|---|---|---|---|---|
| `Iso3Sym` | iso3 (3-bit quaternion SO(4)) | iso3 (3-bit) | `(IsoK3, Iso3)` | `iso_sym_3` |
| `Iso4Sym` | iso4 (4-bit quaternion SO(4)) | iso4 (4-bit) | `(IsoK4, Iso4)` | `iso_sym_4_v2` |
| `IsoKOnly3` | iso3 (3-bit) | **bf16** (parent `decode_fp16_v`) | `(IsoK3, Bf16)` | `iso_k_only_3` |
| `IsoKOnly4` | iso4 (4-bit) | **bf16** | `(IsoK4, Bf16)` | `iso_k_only_4` |

**Qwen MoE arch guard.** `cache_type::validate_resolved` rejects all four
variants on Qwen MoE with `ResolveError::QwenMoeIsoKRejected { variant }`. The
error names the variant, and the process exits with code 78. `auto` never
selects any of the four on any arch.

**IsoKOnly bf16-V layout.** The V buffer is the parent
`KvCache::decode_fp16_v`, as for `KvStorage::None` and `KvStorage::PlanarK`.
The SSD writer writes only the K-side tensors
(`l{idx}.k.codes_packed/scales/quaternions/norms`).

**Status.** GPU-resident on the hot path. `QuantIsoK3` / `QuantIsoK4` each hold
a `QuantKGpuRing`. The K encode writes the packed ring on the GPU, and
`iso_flash_decode` reads that ring (see `docs/KV_FUSED_KERNELS.md` § `iso_flash_decode`).

**Cosine gates.** On the LCG fixture at `head_dim=128, n_rows=16, TEST_SEED`
(`storage/quant_iso_k_tests.rs`): `iso_k_3` gates at 0.97, `iso_k_4` at 0.99
(minimum cosine).

**SSD round-trip tests.** Four tests in
`crates/rmlx-kv-ssd/src/block_io_tests.rs`: `roundtrip_iso_sym_3`,
`roundtrip_iso_sym_4`, `roundtrip_iso_k_only_3` and `roundtrip_iso_k_only_4`.
They assert that the K-side codes are bit-identical after hydrate and that the
K dequant matches within 1e-3.

### rotor K-side variants

> **INERT on this build** — the two asymmetric members,
> `rotor_k_3_asym_v<vb>_g<vg>` and `rotor_k_4_asym_v<vb>_g<vg>`, decode from the
> bf16 mirror on both axes, so `exit_prefill` never builds the packed store
> described below and their codec math never runs. Resident KV and generated
> tokens measure identical to bf16. The two K-only members of this section are
> not in that class. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree
> is for".

Six variants apply the rotor3 / rotor4 codec to the K axis. They can add a
**1-bit QJL residual sideband** (a Johnson–Lindenstrauss sketch of the
post-rotor MSE residual) with `--rotor-qjl on`. QJL is **off** by default. QJL
has no MSL kernel, so QJL on moves the rotor K encode and dequant to the CPU.
`rotor_qjl_enabled()` in `rmlx-kv-quant::rotor_qjl` holds the toggle.

| `KvQuant` | K codec | V codec | CacheType pair (`(K, V)`) | SSD tag (QJL off / on) |
|---|---|---|---|---|
| `Rotor3Sym` | rotor3 (+ QJL) | rotor3 | `(RotorK3, Rotor3)` | `rotor_sym_3` / `rotor_sym_3_qjl` |
| `Rotor4Sym` | rotor4 (+ QJL) | rotor4 | `(RotorK4, Rotor4)` | `rotor_sym_4` / `rotor_sym_4_qjl` |
| `RotorKOnly3` | rotor3 (+ QJL) | **bf16** (parent `decode_fp16_v`) | `(RotorK3, Bf16)` | `rotor_k_only_3` / `rotor_k_only_3_qjl` |
| `RotorKOnly4` | rotor4 (+ QJL) | **bf16** | `(RotorK4, Bf16)` | `rotor_k_only_4` / `rotor_k_only_4_qjl` |
| `RotorK3Asym { v_bits, v_group_size }` | rotor3 (+ QJL) | **TurboQuant V** at `v_bits` (the K8V4 / K8VTurbo3 / K8VTurbo2 V codec) | `(RotorK3, Q*G*)` | `rotor_k_asym_3_v{vb}_g{vg}` / `rotor_k_asym_3_qjl_v{vb}_g{vg}` |
| `RotorK4Asym { v_bits, v_group_size }` | rotor4 (+ QJL) | **TurboQuant V** at `v_bits` | `(RotorK4, Q*G*)` | `rotor_k_asym_4_v{vb}_g{vg}` / `rotor_k_asym_4_qjl_v{vb}_g{vg}` |

**Asymmetric rotor-K variants.** `RotorK{3,4}Asym` carry a TurboQuant V codec at
`v_bits ∈ {2, 3, 4}`. The V slot uses the `QuantV` codec of `K8V4` /
`K8VTurbo3` / `K8VTurbo2` (Lloyd-Max N(0,1) codebook, fixed internal group of
32). The `v_group_size` field goes into the SSD layout key only. The parser and
the composer reject `v_bits = 8`, because TurboQuant has no 8-bit path. For
bf16 V use `--ctv bf16` (`RotorKOnly{3,4}`); for rotor V use
`--ctv rotor_v_{3,4}` (`Rotor{3,4}Sym`).

Display form: `rotor_k_3_asym_v{v_bits}_g{v_group_size}` (and `_4_`).
Compose forms:

- `--ctk k_rotor3 --ctv q4_g64` → `RotorK3Asym { v_bits: 4, v_group_size: 64 }`
- `--ctk k_rotor3 --ctv q4_g128` → `RotorK3Asym { v_bits: 4, v_group_size: 128 }`
- `--ctk k_rotor4 --ctv q3_g64` → `RotorK4Asym { v_bits: 3, v_group_size: 64 }`
- `--ctk k_rotor4 --ctv q2_g64` → `RotorK4Asym { v_bits: 2, v_group_size: 64 }`
- `--ctk k_rotor3 --ctv rotor_v_3` → `Rotor3Sym`
- `--ctk k_rotor3 --ctv bf16` → `RotorKOnly3`

**Qwen MoE arch guard.** `cache_type::validate_resolved` rejects all six
variants on Qwen MoE with `ResolveError::QwenMoeRotorKRejected { variant }`.
The `variant` field holds the full Display form (for example
`rotor_k_3_asym_v4_g64`). The process exits with code 78.

**Decode cost with QJL on.** With `--rotor-qjl on`, the rotor K-side codecs
have no GPU-resident code mirror. Each decode step decodes the full K prefix on
the CPU, applies an O(head_dim²) QJL score correction per cached token, and
uploads the K prefix again. This cost grows with `kv_seq`. With QJL off, the
rotor K encode and `rotor_flash_decode` run on Metal.

**Fused-QK.** Of the rotor codecs, the head-major fused-QK path has kernels
for `RotorK3Asym` and `RotorK4Asym` only. `--fused-qk` is off by default. The
path runs only when all of these conditions are true
(`try_fused_qk_dispatch`):

- `--fused-qk on`;
- the device is GPU;
- the step is a decode step (`q_seq == 1`) and `new_k` has rank 4;
- `head_dim` is 128 or 256;
- `kv_seq` is at least `fused_qk_min_kv_seq` (default 512, env
  `RMLX_FUSED_QK_MIN`);
- the codec has a fused-QK kernel and a GPU K encoder;
- QJL is off (the kernel does not use the QJL residual);
- the bf16 K mirror is live to seed the shadow;
- the storage variant carries a `max_seq`, and the step does not overflow it.

See `docs/KV_FUSED_KERNELS.md` § "Fused-QK head-major K storage".

**QJL residual — storage format.** When QJL is on at the first `append`, the
codec stores one extra sign bit per `head_dim` element per token with the rotor
codes. Wire format: packed `u8`, row-major, shape
`[B, kv_h, max_seq, ceil(head_dim/8)]`. Bit order: LSB = element 0, MSB =
element 7 (the same as the Python `rotorquant/turboquant/rotorquant.py`
reference). The QJL projection matrix `S` (`[head_dim, head_dim]` f32,
row-major) is made once at the first append and written to the SSD block
(`l{idx}.k.qjl_s`). The layout tag (`*_qjl`) tells the reader to hydrate the
projection matrix.

**QJL toggle.** CLI: `--rotor-qjl on|off` (default `off`). The `rmlx` binary
always installs the flag value at startup, so the environment variable
`RMLX_ROTOR_QJL` has no effect on it. `RMLX_ROTOR_QJL=1` (or `on`, `true`,
`yes`) enables QJL only in a process that does not install the flag, for
example a test binary. The toggle is read at each construction (not cached).

**Score-time QJL correction.** `apply_qjl_correction` (called from
`rotor3_k_decode` / `rotor4_k_decode`) applies the correction **at decode time**
as a per-token K-side residual-add:

```
Δk[t, j] = ||r_t|| · sqrt(π/2)/m · sum_i ( S[i, j] · signs[t, i] )
K_corrected[t] = K_rotor[t] + Δk[t]
```

The result `Q · K_corrected` is equal to the score-time `term1 + term2` of the
Python reference (`RotorQuantProd.inner_product` in
`rotorquant/turboquant/rotorquant.py:246-263`), because `term2` is linear in
`Q`. Thus the correction stays inside `rmlx-kv-quant`, and every caller of
`rotor3_k_decode` / `rotor4_k_decode` gets it.

Validation (both in `crates/rmlx-kv-quant/src/rotorquant_tests.rs`, run by
`make ci`):

- `qjl_correction_score_estimator_unbiased_rotor3` and `_rotor4` reproduce the
  Python
  `test_inner_product_unbiased` (n=1024 unit-normalized pairs). They assert
  `|bias| < 0.05` with QJL on and off, through `apply_qjl_correction`.
- `qjl_residual_add_matches_score_time_correction` asserts, per token,
  `Q · K_on == Q · K_off + Python_term2` within 1e-4 absolute. This proves that
  the dequant-side residual-add is the same as the score-time `term2` of the
  reference for the same rotor MSE codes.

Per-K cosine on the LCG fixture is slightly lower with QJL on. This is
expected: the JL sketch gives an unbiased inner product, not a closer K vector.

**SSD round-trip tests.** Eight tests in
`crates/rmlx-kv-ssd/src/block_io_tests.rs`:
`roundtrip_rotor_sym_3_{qjl,no_qjl}`,
`roundtrip_rotor_sym_4_{qjl,no_qjl}`,
`roundtrip_rotor_k_only_{3,4}_{qjl,no_qjl}`. Each asserts that the K codes are
bit-identical after hydrate and that the `use_qjl()` flag matches the tag. The
tests hold `ROTOR_QJL_ENV_LOCK` (a process-wide mutex) to prevent env-var races
under parallel `cargo test`.
