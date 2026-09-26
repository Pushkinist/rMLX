// Promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill — which stay in `rmlx-models`) can still
// reach them across the crate boundary. Doc/visibility warnings on the
// promoted surface are silenced; the API is otherwise unchanged.
#![allow(
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
//! `KvStorage` — top-level KV cache storage enum.
#![allow(clippy::match_same_arms, clippy::too_many_lines)]

use rmlx_core::error::Result;

use super::kv_slot::KvSlot;
use super::{
    QuantIsoK3, QuantIsoK4, QuantIsoV3, QuantIsoV4, QuantK, QuantKTurbo3, QuantKTurbo4,
    QuantPlanarK, QuantPlanarV, QuantRotorK3, QuantRotorK4, QuantRotorV3, QuantRotorV4, QuantV,
};
use crate::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};
use crate::KvQuant;

/// Layout tag for symmetric TurboQuant 3-bit K + turbo3 V.
///
/// Both K and V use the same Lloyd-Max N(0,1) 3-bit codebook (axis-agnostic
/// turbo3 kernel). Distinct from `"k8vturbo3"` (asymmetric K8V turbo3) and
/// from [`TURBOSYM4_LAYOUT_TAG`] (4-bit symmetric) so the SSD reader can
/// dispatch to the correct symmetric 3-bit hydrate path. Format
/// `"<codec>_lloyd_<k_bits>_<v_bits>"`.
pub const TURBOSYM3_LAYOUT_TAG: &str = "tsym3_lloyd_3_3";

/// Layout tag for symmetric TurboQuant 4-bit K + tq4 V.
///
/// Single source of truth for the SSD geometry tag (used by `KvBlockWriter` /
/// `KvBlockReader` and the layout-key tier). Format
/// `"<codec>_lloyd_<k_bits>_<v_bits>"`.
pub const TURBOSYM4_LAYOUT_TAG: &str = "tsym4_lloyd_4_4";

/// Layout tag for K-axis PlanarQuant 4-bit.
///
/// Single source of truth for the SSD geometry tag and the layout-key tier;
/// referenced by `KvBlockWriter`, `KvBlockReader`, and the SSD index.
pub const PLANARK4_LAYOUT_TAG: &str = "planar_k_4";

/// Layout tag for V-axis IsoQuant 3-bit.
///
/// Single source of truth for the SSD geometry tag.
pub const ISOV3_LAYOUT_TAG: &str = "iso_v_3";

/// Layout tag for V-axis IsoQuant 4-bit.
///
/// Single source of truth for the SSD geometry tag.
// v2: the iso4 V GPU-encode append stored its block head-major while `dequant`
// reads sequence-major, so a multi-token `kv_h > 1` chunk spilled head-scrambled
// bytes. The append is fixed, and the tag is bumped because nothing else on disk
// tells the two layouts apart: the block header carries only
// `{tag, max_seq, shape}`, and the SSD index key
// (`FNV_OFFSET ^ layout_key ^ cache_key_salt ^ model_sig`) is derived from the
// arch, geometry and codec name — none of which move when the orientation of the
// bytes inside a block does. With the bump a pre-fix entry hits no read arm and
// fails its hydrate loudly ("unknown layer tag"); without it, it is read back
// with the new orientation and no error. One cold pass after upgrading.
pub const ISOV4_LAYOUT_TAG: &str = "iso_v_4_v2";

/// Layout tag for V-axis rotor3 (Cl(3,0) Clifford sandwich).
///
/// Single source of truth for the SSD geometry tag.
pub const ROTORV3_LAYOUT_TAG: &str = "rotor_v_3";

/// Layout tag for V-axis rotor4 (Cl(3,0) Clifford sandwich, 4-bit).
///
/// Single source of truth for the SSD geometry tag.
pub const ROTORV4_LAYOUT_TAG: &str = "rotor_v_4";

/// Layout tag for symmetric IsoQuant 3-bit K+V.
///
/// Distinct from the V-only `"iso_v_3"` (`ISOV3_LAYOUT_TAG`) tag so the SSD
/// reader can dispatch to the symmetric K+V hydrate path. K-side payload
/// uses identical wire format to V-side (codes_packed/scales/quaternions/norms)
/// under the `l{idx}.k.*` tensor names.
pub const ISO_SYM_3_LAYOUT_TAG: &str = "iso_sym_3";

/// Layout tag for symmetric IsoQuant 4-bit K+V.
// v2 for the same reason as `ISOV4_LAYOUT_TAG` — the V half of this storage
// takes the same append.
pub const ISO_SYM_4_LAYOUT_TAG: &str = "iso_sym_4_v2";

/// Layout tag for K-only IsoQuant 3-bit (V stays bf16).
///
/// Mirrors `PLANARK4_LAYOUT_TAG` shape — only the K side is serialised; the
/// V side lives on the parent `KvCache::decode_fp16_v` and is rebuilt
/// transparently on hydrate.
pub const ISO_K_ONLY_3_LAYOUT_TAG: &str = "iso_k_only_3";

/// Layout tag for K-only IsoQuant 4-bit.
pub const ISO_K_ONLY_4_LAYOUT_TAG: &str = "iso_k_only_4";

/// Symmetric rotor3 K+V layout tag (QJL OFF).
///
/// Distinct from `ROTORV3_LAYOUT_TAG` (`"rotor_v_3"`) so the SSD reader can
/// dispatch to the symmetric K+V hydrate path. K payload uses the K-side
/// rotor3 wire format (codes_packed/scales/norms/rotors under `l{idx}.k.*`)
/// without the optional QJL sideband; V payload is unchanged from `RotorV3`.
pub const ROTOR_SYM_3_LAYOUT_TAG: &str = "rotor_sym_3";

/// Symmetric rotor3 K+V layout tag with QJL residual ON.
///
/// Same as [`ROTOR_SYM_3_LAYOUT_TAG`] but includes the K-side QJL sideband
/// (`l{idx}.k.qjl_codes`, `l{idx}.k.qjl_norms`, `l{idx}.k.qjl_s`). The tag
/// distinction is the load-bearing signal for the reader to hydrate the QJL
/// projection matrix.
pub const ROTOR_SYM_3_QJL_LAYOUT_TAG: &str = "rotor_sym_3_qjl";

/// Symmetric rotor4 K+V layout tag (QJL OFF).
pub const ROTOR_SYM_4_LAYOUT_TAG: &str = "rotor_sym_4";

/// Symmetric rotor4 K+V layout tag with QJL residual ON.
pub const ROTOR_SYM_4_QJL_LAYOUT_TAG: &str = "rotor_sym_4_qjl";

/// K-only rotor3 layout tag (V is bf16; QJL OFF).
pub const ROTOR_K_ONLY_3_LAYOUT_TAG: &str = "rotor_k_only_3";

/// K-only rotor3 layout tag with QJL residual ON.
pub const ROTOR_K_ONLY_3_QJL_LAYOUT_TAG: &str = "rotor_k_only_3_qjl";

/// K-only rotor4 layout tag (QJL OFF).
pub const ROTOR_K_ONLY_4_LAYOUT_TAG: &str = "rotor_k_only_4";

/// K-only rotor4 layout tag with QJL residual ON.
pub const ROTOR_K_ONLY_4_QJL_LAYOUT_TAG: &str = "rotor_k_only_4_qjl";

/// Asymmetric rotor3 K + `QuantV` V layout tag prefix
/// (QJL OFF). The full tag carries the V (bits, group) tuple appended as
/// `_v{v_bits}g{v_group_size}` so the SSD reader can pick the V codec on
/// hydrate. Distinct from [`ROTOR_K_ONLY_3_LAYOUT_TAG`] because the V-side
/// payload differs (`QuantV` store vs bf16-on-parent).
pub const ROTOR_K_ASYM_3_LAYOUT_PREFIX: &str = "rotor_k_asym_3";

/// Asymmetric rotor3 K + `QuantV` V layout tag prefix (QJL residual ON).
pub const ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX: &str = "rotor_k_asym_3_qjl";

/// Asymmetric rotor4 K + `QuantV` V layout tag prefix (QJL OFF).
pub const ROTOR_K_ASYM_4_LAYOUT_PREFIX: &str = "rotor_k_asym_4";

/// Asymmetric rotor4 K + `QuantV` V layout tag prefix (QJL residual ON).
pub const ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX: &str = "rotor_k_asym_4_qjl";

/// Layout tag for K8VTurbo3Tcq (Viterbi trellis 3-bit V).
///
/// Distinct from `"k8vturbo3"` so the SSD layer can refuse cross-codec
/// hydrate: the on-disk byte stream is layout-compatible with plain turbo3
/// (same packing), but the assignment came from Viterbi — re-hydrating a TCQ
/// blob into a plain `K8VTurbo3` cache would silently keep the Viterbi
/// indices and then re-encode any newly appended decode tokens with
/// nearest-centroid, producing a mixed-assignment cache. Hard-tagging the
/// payload prevents this.
pub const K8VTURBO3_TCQ_LAYOUT_TAG: &str = "k8vturbo3tcq";

/// Layout tag for K8VTurbo2Tcq (Viterbi trellis 2-bit V).
///
/// Distinct from `"k8vturbo2"` for the same cross-codec protection reason as
/// [`K8VTURBO3_TCQ_LAYOUT_TAG`]: the pack is byte-for-byte identical to plain
/// turbo2 (2-bit LSB-first, 16 values per u32), but the Viterbi assignment
/// came from the 4-state TCQ trellis. Hard-tagging prevents silent demotion
/// to nearest-centroid on subsequent decode-step encodes.
pub const K8VTURBO2_TCQ_LAYOUT_TAG: &str = "k8vturbo2tcq";

// ── KvStorage ─────────────────────────────────────────────────────────────────

/// Internal storage: quantized byte buffers per K/V scheme.
///
/// `auto` resolves to [`None`](KvStorage::None) (bf16) on every arch; every
/// other variant is opt-in.
pub enum KvStorage {
    /// K = affine q8_0, V = TurboQuant 4-bit.
    K8V4 {
        k: Option<QuantK>,
        v: Option<QuantV>,
    },
    /// K = affine q8_0, V = affine q8_0.
    K8V8 {
        k: Option<QuantK>,
        v: Option<QuantK>,
    },
    /// K = affine q8_0, V = PlanarQuant N-bit.
    ///
    /// `bits ∈ {3, 4}`: 4 = original Planar codec (`KvQuant::Planar`);
    /// 3 = 3.25-bit variant (`KvQuant::Planar3`).
    ///
    /// Opt-in via `--kv-quant planar` (4-bit) or `--kv-quant planar3` (3-bit).
    Planar {
        k: Option<QuantK>,
        v: Option<QuantPlanarV>,
        /// Bit-width for the V codec: 3 (Planar3) or 4 (Planar).
        bits: u8,
    },
    /// Unquantised bf16 cache. The actual `Array` buffers live in
    /// `KvCache::decode_fp16_k` / `decode_fp16_v`, reusing the same machinery
    /// as the warm-TTFT fp16 decode-seed path. The variant holds no field.
    None {},
    /// Mixed-precision K/V via `mx.quantize` 3-tuples.
    ///
    /// State is owned by [`crate::mixed_quant::MixedKvState`]. Its buffers
    /// grow in `STEP=256` increments matching mlx-lm-tq.
    Mixed {
        state: crate::mixed_quant::MixedKvState,
    },
    /// PagedAttention block-table KV storage (paged KV path, `--paged-kv`).
    ///
    /// K is always q8_0 (PagedKStorage). V variant is chosen by quant mode:
    /// - K8V4 / K8V8 → PagedVStorage (TurboQuant V4 for K8V4, q8_0 for K8V8).
    /// - Planar → PagedPlanarVStorage.
    ///
    /// For single-request decoding this degenerates to contiguous behaviour.
    /// The block table is only appended to: no cross-request sharing and no
    /// eviction.
    Paged {
        quant: KvQuant,
        k: Option<PagedKStorage>,
        v_k8: Option<Box<PagedVStorage>>,
        v_planar: Option<Box<PagedPlanarVStorage>>,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = TurboQuant 3-bit Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Structurally identical to [`K8V4`](KvStorage::K8V4) but with `bits=3`
    /// in the [`QuantV`] slot.
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    K8VTurbo3 {
        k: Option<QuantK>,
        v: Option<QuantV>,
    },
    /// Symmetric TurboQuant 3-bit: K = `QuantKTurbo3`, V = `QuantV` (bits=3).
    ///
    /// Both axes use the Lloyd-Max N(0,1) 3-bit codebook (the same
    /// axis-agnostic CPU + MSL kernel as the V-side `K8VTurbo3` path). The K and V
    /// buffers are kept as independent types so the two append paths stay
    /// decoupled. Layout tag: [`TURBOSYM3_LAYOUT_TAG`].
    ///
    /// **Arch guard**: never resolved automatically for Qwen MoE — explicit
    /// `--kv-quant tsym3` on Qwen MoE is rejected by the post-resolve invariant
    /// check in `rmlx_models::kv_cache::validate_resolved_kv_quant` (K below
    /// 8 bits).
    ///
    /// **V-side device**: the 3-bit V axis is pinned to the CPU, because
    /// `QuantV::append`'s GPU branch refuses `bits != 4` (see `tsym_update`).
    /// K side uses the GPU turbo3 MSL kernel when `Device::Gpu` is in effect.
    TurboSym3 {
        k: Option<QuantKTurbo3>,
        v: Option<QuantV>,
    },
    /// Symmetric TurboQuant 4-bit: K = `QuantKTurbo4`, V = `QuantV` (bits=4).
    ///
    /// Both axes use the Lloyd-Max N(0,1) 4-bit codebook (the same
    /// axis-agnostic CPU + MSL kernel as the V-side `K8V4` path). The K and V
    /// buffers are kept as independent types so the two append paths stay
    /// decoupled. Layout tag: [`TURBOSYM4_LAYOUT_TAG`].
    ///
    /// **Arch guard**: never resolved automatically for Qwen MoE — explicit
    /// `--kv-quant tsym4` on Qwen MoE is rejected by the post-resolve invariant
    /// check in `rmlx_models::kv_cache::validate_resolved_kv_quant` (K below
    /// 8 bits).
    TurboSym4 {
        k: Option<QuantKTurbo4>,
        v: Option<QuantV>,
    },
    /// K = PlanarQuant 4-bit, V = unquantised bf16 (mtq `k_only_planar`).
    ///
    /// Opposite of [`Planar`](KvStorage::Planar): Givens-rotation 4-bit codec on
    /// the **K** axis; V stays full-precision bf16 (lives on the parent
    /// `KvCache::decode_fp16_v`, same machinery as [`None`](KvStorage::None)
    /// for V). Qwen MoE rejects PlanarK (K below 8 bits) — see
    /// `cache_type::validate_resolved`. `head_dim % 32 == 0` is required
    /// (PlanarQuant block constraint).
    ///
    /// Opt-in only via `--kv-quant planar_k`. Never an auto default.
    PlanarK { k: Option<QuantPlanarK> },
    /// K = affine q8_0 (group_size=128),
    /// V = TurboQuant **2-bit** Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Structurally identical to [`K8V4`](KvStorage::K8V4) but with `bits=2`
    /// in the [`QuantV`] slot. Native 2.25-bit V codec with no outlier mask.
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    K8VTurbo2 {
        k: Option<QuantK>,
        v: Option<QuantV>,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = IsoQuant 3-bit (quaternion SO(4) rotation + Lloyd-Max codebook).
    ///
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    IsoV3 {
        k: Option<QuantK>,
        v: Option<QuantIsoV3>,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = IsoQuant 4-bit (quaternion SO(4) rotation + Lloyd-Max 4-bit codebook).
    ///
    /// Same machinery as [`IsoV3`](KvStorage::IsoV3) with `bits=4` and the
    /// dense code plane at 4 bits per code.
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    IsoV4 {
        k: Option<QuantK>,
        v: Option<QuantIsoV4>,
    },
    /// Symmetric IsoQuant 3-bit — both K and V use the same
    /// quaternion SO(4) + 3-bit Lloyd-Max codebook (axis-agnostic codec).
    ///
    /// K is stored in `QuantIsoK3`; V in `QuantIsoV3`.
    /// Decode runs the flash kernel over both packed rings
    /// (`docs/KV_FUSED_KERNELS.md`).
    /// Layout tag: [`ISO_SYM_3_LAYOUT_TAG`].
    IsoSym3 {
        k: Option<QuantIsoK3>,
        v: Option<QuantIsoV3>,
    },
    /// Symmetric IsoQuant 4-bit — both K and V use the same
    /// quaternion SO(4) + 4-bit Lloyd-Max codebook, 4 bits per code in the plane.
    ///
    /// Decode runs the flash kernel over both packed rings
    /// (`docs/KV_FUSED_KERNELS.md`).
    /// Layout tag: [`ISO_SYM_4_LAYOUT_TAG`].
    IsoSym4 {
        k: Option<QuantIsoK4>,
        v: Option<QuantIsoV4>,
    },
    /// K-only IsoQuant 3-bit; V is bf16 on the parent
    /// `KvCache::decode_fp16_v` (same machinery as `KvStorage::None` /
    /// `KvStorage::PlanarK` for V).
    ///
    /// Layout tag: [`ISO_K_ONLY_3_LAYOUT_TAG`]. Opt-in only.
    IsoKOnly3 { k: Option<QuantIsoK3> },
    /// K-only IsoQuant 4-bit; V is bf16 on the parent
    /// `KvCache::decode_fp16_v`.
    ///
    /// Layout tag: [`ISO_K_ONLY_4_LAYOUT_TAG`].
    IsoKOnly4 { k: Option<QuantIsoK4> },
    /// Symmetric rotor3 — both K and V use the same Cl(3,0)
    /// Clifford rotor sandwich + 3-bit Lloyd-Max codebook (axis-agnostic
    /// codec). The K side optionally carries a 1-bit QJL residual per
    /// element (`QuantRotorK3::qjl_s_matrix.is_some()`).
    ///
    /// Layout tag: [`ROTOR_SYM_3_LAYOUT_TAG`] or [`ROTOR_SYM_3_QJL_LAYOUT_TAG`].
    /// Decode runs the flash kernel over both packed rings
    /// (`docs/KV_FUSED_KERNELS.md`). A QJL store keeps the CPU dequant path
    /// on both axes.
    RotorSym3 {
        k: Option<QuantRotorK3>,
        v: Option<QuantRotorV3>,
    },
    /// Symmetric rotor4 — same shape as `RotorSym3` with the
    /// 4-bit codebook on both axes.
    ///
    /// Layout tag: [`ROTOR_SYM_4_LAYOUT_TAG`] or [`ROTOR_SYM_4_QJL_LAYOUT_TAG`].
    RotorSym4 {
        k: Option<QuantRotorK4>,
        v: Option<QuantRotorV4>,
    },
    /// K-only rotor3 — K is rotor3; V stays bf16 on the parent
    /// `KvCache::decode_fp16_v` (same machinery as `IsoKOnly3` /
    /// `PlanarK` / `None`).
    ///
    /// Decode reads the packed K store directly via the rotor flash-decode MSL
    /// kernel when the store carries no QJL sideband; a QJL store keeps the CPU
    /// dequant path. See `docs/KV_FUSED_KERNELS.md` § `rotor_flash_decode`.
    ///
    /// Layout tag: [`ROTOR_K_ONLY_3_LAYOUT_TAG`] or
    /// [`ROTOR_K_ONLY_3_QJL_LAYOUT_TAG`].
    RotorKOnly3 { k: Option<QuantRotorK3> },
    /// K-only rotor4 — same shape as `RotorKOnly3` with 4-bit
    /// codes.
    ///
    /// Layout tag: [`ROTOR_K_ONLY_4_LAYOUT_TAG`] or
    /// [`ROTOR_K_ONLY_4_QJL_LAYOUT_TAG`].
    RotorKOnly4 { k: Option<QuantRotorK4> },
    /// Asymmetric rotor3 K + `QuantV` V — K is rotor3 (optional QJL residual
    /// sideband); V is `QuantV`, the TurboQuant N(0,1) Lloyd-Max codec at a
    /// fixed 32-element group, despite the `v_bits` / `v_group_size` names.
    ///
    /// Layout key prefix: [`ROTOR_K_ASYM_3_LAYOUT_PREFIX`] /
    /// [`ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX`]; the full SSD layout key suffixes
    /// `_v{v_bits}g{v_group_size}` so hydrate can pick the V codec.
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    RotorKAsym3 {
        k: Option<QuantRotorK3>,
        v: Option<QuantV>,
        /// V quantization bit-width.
        v_bits: u8,
        /// V group size as the codec spelling carries it.
        v_group_size: u16,
    },
    /// Asymmetric rotor4 K + `QuantV` V — same shape as
    /// `RotorKAsym3` with the dense 4-bit rotor codebook on K.
    ///
    /// Layout key prefix: [`ROTOR_K_ASYM_4_LAYOUT_PREFIX`] /
    /// [`ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX`].
    RotorKAsym4 {
        k: Option<QuantRotorK4>,
        v: Option<QuantV>,
        /// V quantization bit-width.
        v_bits: u8,
        /// V group size as the codec spelling carries it.
        v_group_size: u16,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = rotor3 (Cl(3,0) Clifford rotor sandwich + 3-bit Lloyd-Max codebook).
    ///
    /// Static per-layer rotor table on the V side (lazily generated on first
    /// append; never per-token).
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    RotorV3 {
        k: Option<QuantK>,
        v: Option<QuantRotorV3>,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = rotor4 (Cl(3,0) Clifford rotor sandwich + 4-bit Lloyd-Max codebook).
    ///
    /// 4.25-bit V codec — same Clifford sandwich as rotor3 with the
    /// 16-centroid Lloyd-Max N(0,1) codebook, 4 bits per code in the plane
    /// (iso4 convention). Higher fidelity than rotor3 at the cost of one extra
    /// bit per code: 9.75 bits per value at `head_dim = 128` against rotor3's
    /// 8.75. See `crate::rotorquant` § "Effective bpe".
    ///
    /// Static per-layer rotor table on the V side (lazily generated on first
    /// append).
    /// Decode reads the bf16 mirror, so a cache that went through prefill
    /// builds no store (`docs/KV_QUANT.md` § "Codec disposition", Class 2).
    ///
    /// `--paged-kv` does not route it: it keeps this storage.
    RotorV4 {
        k: Option<QuantK>,
        v: Option<QuantRotorV4>,
    },
    /// K = affine q8_0 (group_size=128), V = TurboQuant 3-bit with
    /// Viterbi trellis (TCQ) assignment over the standard Lloyd-Max codebook.
    ///
    /// Layout is byte-for-byte identical to
    /// [`K8VTurbo3`](KvStorage::K8VTurbo3) (same `QuantV` pack with `bits=3`);
    /// only the encode-side assignment differs. `QuantV::use_tcq` distinguishes
    /// the encode path inside `QuantV::append`. The decoder is shared with
    /// plain turbo3.
    ///
    /// Layout tag: [`K8VTURBO3_TCQ_LAYOUT_TAG`]. Opt-in only via
    /// `--kv-quant k8vturbo3tcq`; never an auto default.
    K8VTurbo3Tcq {
        k: Option<QuantK>,
        v: Option<QuantV>,
    },
    /// K = affine q8_0 (group_size=128), V = TurboQuant 2-bit with
    /// Viterbi trellis (TCQ) assignment over the standard Lloyd-Max 2-bit
    /// codebook (4 centroids).
    ///
    /// Layout is byte-for-byte identical to
    /// [`K8VTurbo2`](KvStorage::K8VTurbo2) (same `QuantV` pack with `bits=2`,
    /// 16 values per u32); only the encode-side assignment differs.
    /// `QuantV::use_tcq` distinguishes the encode path inside `QuantV::append`.
    /// The decoder is shared with plain turbo2 (decoder is assignment-agnostic).
    ///
    /// Layout tag: [`K8VTURBO2_TCQ_LAYOUT_TAG`]. Opt-in only via
    /// `--kv-quant k8vturbo2tcq`; never an auto default.
    K8VTurbo2Tcq {
        k: Option<QuantK>,
        v: Option<QuantV>,
    },
}

impl KvStorage {
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn new(quant: KvQuant) -> Self {
        use crate::paged::paged_kv_enabled;

        // When paged KV is enabled (--paged-kv), route K8V4 / K8V8 / Planar to
        // the block-table paged path. None and Mixed stay on their existing paths (bf16
        // buffers / mx.quantize 3-tuples) — paging does not apply to them.
        if paged_kv_enabled() {
            match quant {
                KvQuant::K8V4 | KvQuant::K8V8 | KvQuant::Planar | KvQuant::Planar3 => {
                    return Self::Paged {
                        quant,
                        k: None,
                        v_k8: None,
                        v_planar: None,
                    };
                }
                _ => {}
            }
        }

        match quant {
            KvQuant::K8V4 => Self::K8V4 { k: None, v: None },
            KvQuant::K8V8 => Self::K8V8 { k: None, v: None },
            KvQuant::Planar => Self::Planar {
                k: None,
                v: None,
                bits: 4,
            },
            // Planar3 routes to Planar storage with bits=3.
            // No new KvStorage variant — the existing Planar variant carries bits.
            KvQuant::Planar3 => Self::Planar {
                k: None,
                v: None,
                bits: 3,
            },
            KvQuant::None => Self::None {},
            KvQuant::Mixed {
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
            } => Self::Mixed {
                state: crate::mixed_quant::MixedKvState::new(
                    i32::from(k_bits),
                    i32::from(v_bits),
                    i32::from(k_group_size),
                    i32::from(v_group_size),
                ),
            },
            // RotK reuses Mixed storage with K-side rotation enabled.
            KvQuant::RotK {
                v_bits,
                v_group_size,
            } => Self::Mixed {
                state: crate::mixed_quant::MixedKvState::new_rotated(
                    i32::from(v_bits),
                    i32::from(v_group_size),
                ),
            },
            // K8VTurbo3 — same layout as K8V4 but QuantV bits=3.
            KvQuant::K8VTurbo3 => Self::K8VTurbo3 { k: None, v: None },
            // TurboSym3 — symmetric 3-bit Lloyd-Max K+V. Never routes through
            // the paged path: PagedKStorage is q8-only and there is no paged
            // TurboQuant-K3 variant. Deviation documented in docs/KV_CODECS.md.
            KvQuant::TurboSym3 => Self::TurboSym3 { k: None, v: None },
            // TurboSym4 — symmetric 4-bit Lloyd-Max K+V. Never routes through
            // the paged path: PagedKStorage is q8-only and adding a TurboQuant-K
            // paged variant is out of scope; deviation documented in
            // docs/KV_CODECS.md.
            KvQuant::TurboSym4 => Self::TurboSym4 { k: None, v: None },
            // PlanarK — K-axis PlanarQuant 4-bit; V is bf16 on the parent KvCache.
            // Never routes through paged: PagedKStorage is q8-only and there is no
            // paged PlanarQuant-K variant; deviation documented in docs/KV_CODECS.md.
            KvQuant::PlanarK => Self::PlanarK { k: None },
            // K8VTurbo2 — same layout as K8V4 but QuantV bits=2.
            KvQuant::K8VTurbo2 => Self::K8VTurbo2 { k: None, v: None },
            // Iso3 — K = affine q8_0, V = IsoQuant 3-bit.
            KvQuant::Iso3 => Self::IsoV3 { k: None, v: None },
            // Iso4 — K = affine q8_0, V = IsoQuant 4-bit.
            KvQuant::Iso4 => Self::IsoV4 { k: None, v: None },
            // Rotor3 — K = affine q8_0, V = rotor3 (Cl(3,0) rotor).
            // Does NOT route through the paged path: PagedVStorage is q8/tq4-only
            // and PagedPlanarVStorage is PlanarQuant-only; a paged RotorV3 would
            // need its own per-token (codes/scales/norms) container plus a static
            // per-layer rotor table inside the paged arena. Deferred per the
            // iso3 / iso4 precedent — opt-in only via --kv-quant rotor3, never
            // an auto baseline.
            KvQuant::Rotor3 => Self::RotorV3 { k: None, v: None },
            // Rotor4 — K = affine q8_0, V = rotor4 (Cl(3,0) rotor, 4-bit).
            // Same paged-KV deferral as Rotor3 — falls through.
            KvQuant::Rotor4 => Self::RotorV4 { k: None, v: None },
            // K8VTurbo3Tcq — same layout as K8VTurbo3 with Viterbi encode-side
            // assignment. The `use_tcq` flag is resolved by `k8_turbo_v_knobs`
            // and written by `k8_turbo_v_update` / `k8_turbo_v_bulk_encode`,
            // all in `kvcache/update_turbo.rs`.
            KvQuant::K8VTurbo3Tcq => Self::K8VTurbo3Tcq { k: None, v: None },
            // K8VTurbo2Tcq — same layout as K8VTurbo2 with Viterbi encode-side
            // assignment. The `use_tcq` flag is resolved by `k8_turbo_v_knobs`
            // and written by `k8_turbo_v_update` / `k8_turbo_v_bulk_encode`,
            // all in `kvcache/update_turbo.rs`.
            KvQuant::K8VTurbo2Tcq => Self::K8VTurbo2Tcq { k: None, v: None },
            // Iso3Sym — K = iso3, V = iso3 (axis-agnostic).
            KvQuant::Iso3Sym => Self::IsoSym3 { k: None, v: None },
            // Iso4Sym — K = iso4, V = iso4.
            KvQuant::Iso4Sym => Self::IsoSym4 { k: None, v: None },
            // IsoKOnly3 — K = iso3; V bf16 lives on the parent
            // `KvCache::decode_fp16_v` (same machinery as PlanarK / None).
            KvQuant::IsoKOnly3 => Self::IsoKOnly3 { k: None },
            // IsoKOnly4 — K = iso4; V bf16 on the parent.
            KvQuant::IsoKOnly4 => Self::IsoKOnly4 { k: None },
            // Rotor3Sym — K = rotor3, V = rotor3 (axis-agnostic).
            KvQuant::Rotor3Sym => Self::RotorSym3 { k: None, v: None },
            // Rotor4Sym — K = rotor4, V = rotor4.
            KvQuant::Rotor4Sym => Self::RotorSym4 { k: None, v: None },
            // RotorKOnly3 — K = rotor3; V bf16 lives on the parent
            // `KvCache::decode_fp16_v` (same machinery as IsoKOnly3 / PlanarK).
            KvQuant::RotorKOnly3 => Self::RotorKOnly3 { k: None },
            // RotorKOnly4 — K = rotor4; V bf16 on the parent.
            KvQuant::RotorKOnly4 => Self::RotorKOnly4 { k: None },
            // Asymmetric rotor3 K + affine V — carry the affine V bit-width /
            // group size on the storage so the codec layer picks the right
            // QuantV pack at first append.
            KvQuant::RotorK3Asym {
                v_bits,
                v_group_size,
            } => {
                // Backstop for callers that built the `KvQuant::RotorK3Asym { .. }`
                // variant directly (pub fields).
                // `validate_rotor_k_asym_v` is the single source-of-truth.
                debug_assert!(
                    crate::quant::validate_rotor_k_asym_v(v_bits, v_group_size).is_ok(),
                    "invalid (v_bits={v_bits}, v_group_size={v_group_size}) — caller bypassed validator",
                );
                Self::RotorKAsym3 {
                    k: None,
                    v: None,
                    v_bits,
                    v_group_size,
                }
            }
            // Asymmetric rotor4 K + affine V.
            KvQuant::RotorK4Asym {
                v_bits,
                v_group_size,
            } => {
                // Backstop for callers that built the `KvQuant::RotorK4Asym { .. }`
                // variant directly (pub fields).
                debug_assert!(
                    crate::quant::validate_rotor_k_asym_v(v_bits, v_group_size).is_ok(),
                    "invalid (v_bits={v_bits}, v_group_size={v_group_size}) — caller bypassed validator",
                );
                Self::RotorKAsym4 {
                    k: None,
                    v: None,
                    v_bits,
                    v_group_size,
                }
            }
        }
    }

    /// Clear the accumulated sequence, keeping the allocations.
    ///
    /// Every slot goes through its store's own `truncate_to(0)` or `reset()`
    /// ([`KvSlot::reset`]), for the same reason `truncate_to` does: zeroing
    /// `shape[2]` alone leaves the CPU-side payload — a block list, or
    /// `QuantK`'s flat `codes`/`scales` — covering the sequence that was just
    /// discarded, so the next `append` stacks on top of it and the dequant
    /// reads the discarded tokens back. The flat stores keep their GPU buffers
    /// in place so the next request reuses the same allocation; the next
    /// `append` overwrites their prefix from offset 0. `None` has no slot: its
    /// bf16 buffers live on `KvCache`, which resets them.
    pub fn reset(&mut self) {
        for slot in self.view_mut().slots.into_iter().flatten() {
            slot.reset();
        }
    }

    /// Truncate the sequence dimension to `n` tokens.
    ///
    /// Every slot delegates to its store's own `truncate_to`, which lowers
    /// `shape[2]` to `n` **and** cuts whatever CPU-side state accumulates
    /// independently of it. The GPU buffers are kept in place (no reallocation)
    /// because `append` uses `slice_update` with a position offset derived from
    /// `shape[2]`; the CPU-side blocks / codes are append-only and have to be
    /// cut, or the next `append` stacks on top of the rejected tokens. `None`
    /// has no slot: `KvCache::truncate_to` handles its bf16 buffers.
    pub fn truncate_to(&mut self, n: i32) {
        // Clamp the negative case once, here, so no store computes from a
        // negative `n`.
        //
        // The upper clamp is NOT uniform, and the divergence is worth naming
        // rather than papering over. The turbo / planar / affine stores clamp
        // `n` down to their own `shape[2]` (`storage::clamp_truncate_target`);
        // the rotor / iso stores deliberately do not, because they abort loudly
        // on an over-long target instead. So for `n > shape[2]` the mixed
        // variants leave the two axes of one codec at different lengths:
        // `IsoV3`, `IsoV4`, `RotorV3`, `RotorV4` (affine K clamps, codec V does
        // not) and `RotorKAsym3` / `RotorKAsym4` (rotor K does not, affine V
        // does). That matters on spill, where the layer geometry is derived from
        // the K shape while the V payload is written raw — the reconciliation
        // guard on the unclamped side is what surfaces it.
        //
        // `Mixed` is a third reading and belongs in the same list: it has no
        // `shape[2]` to clamp, because `state.offset` IS its coverage. Rolling
        // that marker back is the truncation. It keeps its fill on an over-long
        // target and reports one through an error event
        // (`MixedKvState::truncate_to`) — loud like the rotor / iso stores, but
        // at the truncate rather than at the next read, since nothing downstream
        // of it would notice.
        let n = n.max(0);
        for slot in self.view_mut().slots.into_iter().flatten() {
            slot.truncate_to(n);
        }
    }

    #[allow(
        clippy::cognitive_complexity,
        reason = "single match over the closed KvStorage enum — one arm per variant, each is small and self-contained; splitting would hide the 1-to-1 mapping"
    )]
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(match self {
            Self::K8V4 { k, v } => Self::K8V4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            Self::K8V8 { k, v } => Self::K8V8 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            Self::Planar { k, v, bits } => Self::Planar {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                bits: *bits,
            },
            Self::None {} => Self::None {},
            Self::Mixed { state } => Self::Mixed {
                state: state.try_deep_clone()?,
            },
            // Paged: for speculative decoding clone, return a fresh Paged storage.
            // The block-table state is not cloneable efficiently with the page-slab
            // design; callers that need true deep-clone of paged state should
            // re-populate from model forward passes. This returns an empty shell
            // consistent with the existing QuantK::try_deep_clone semantics (which
            // clone GPU buffers but the CPU path is also valid).
            Self::Paged { quant, .. } => Self::Paged {
                quant: *quant,
                k: None,
                v_k8: None,
                v_planar: None,
            },
            // K8VTurbo3 deep-clones like K8V4.
            Self::K8VTurbo3 { k, v } => Self::K8VTurbo3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // TurboSym3 — deep-clone both the 3-bit turbo K store and QuantV.
            Self::TurboSym3 { k, v } => Self::TurboSym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // TurboSym4 — deep-clone both the 4-bit turbo K store and QuantV.
            Self::TurboSym4 { k, v } => Self::TurboSym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // PlanarK — deep-clone K only; V (bf16) on parent.
            Self::PlanarK { k } => Self::PlanarK {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
            },
            // K8VTurbo2 deep-clones like K8V4.
            Self::K8VTurbo2 { k, v } => Self::K8VTurbo2 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // IsoV3 — deep-clone the K (QuantK) + V (QuantIsoV3).
            Self::IsoV3 { k, v } => Self::IsoV3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // IsoV4 — deep-clone the K (QuantK) + V (QuantIsoV4).
            Self::IsoV4 { k, v } => Self::IsoV4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // RotorV3 — deep-clone K (QuantK) + V (QuantRotorV3).
            Self::RotorV3 { k, v } => Self::RotorV3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // RotorV4 — deep-clone K (QuantK) + V (QuantRotorV4).
            Self::RotorV4 { k, v } => Self::RotorV4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // K8VTurbo3Tcq deep-clones like K8VTurbo3.
            Self::K8VTurbo3Tcq { k, v } => Self::K8VTurbo3Tcq {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // K8VTurbo2Tcq deep-clones like K8VTurbo2 / K8VTurbo3Tcq.
            Self::K8VTurbo2Tcq { k, v } => Self::K8VTurbo2Tcq {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // IsoSym3 / IsoSym4 — deep-clone both K + V iso buffers.
            Self::IsoSym3 { k, v } => Self::IsoSym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            Self::IsoSym4 { k, v } => Self::IsoSym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // IsoKOnly3 / IsoKOnly4 — K only.
            Self::IsoKOnly3 { k } => Self::IsoKOnly3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
            },
            Self::IsoKOnly4 { k } => Self::IsoKOnly4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
            },
            // RotorSym3 / RotorSym4 — deep-clone both K + V rotor buffers.
            Self::RotorSym3 { k, v } => Self::RotorSym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            Self::RotorSym4 { k, v } => Self::RotorSym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
            },
            // RotorKOnly3 / RotorKOnly4 — K only.
            Self::RotorKOnly3 { k } => Self::RotorKOnly3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
            },
            Self::RotorKOnly4 { k } => Self::RotorKOnly4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
            },
            // RotorKAsym3 / RotorKAsym4 — deep-clone K rotor + V affine.
            // V codec parameters (bits / group) carry forward unchanged.
            Self::RotorKAsym3 {
                k,
                v,
                v_bits,
                v_group_size,
            } => Self::RotorKAsym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                v_bits: *v_bits,
                v_group_size: *v_group_size,
            },
            Self::RotorKAsym4 {
                k,
                v,
                v_bits,
                v_group_size,
            } => Self::RotorKAsym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                v_bits: *v_bits,
                v_group_size: *v_group_size,
            },
        })
    }

    /// Resident byte footprint of this variant's quantized storage.
    ///
    /// Every slot counts its store's own `byte_size`, which derives the total
    /// from that store's real allocations — CPU blocks, GPU mirrors and GPU
    /// rings alike. There is deliberately **no** per-codec byte formula here:
    /// a second, hand-maintained restatement of what each store holds is what
    /// let a store grow a GPU ring while this total stayed blind to it. The
    /// store owns its own size; this function only sums the slots of
    /// [`Self::view`].
    ///
    /// GPU buffers count their full allocation (rings and mirrors are sized to
    /// capacity, not to the filled prefix) — that is the memory actually held.
    ///
    /// **Excludes paged-pool overhead.** The paged slots count the pages the
    /// slabs hold, not the arena bookkeeping around them. Paged KV
    /// (`--paged-kv`) is default-OFF, so that overhead is 0 on every normal
    /// path; if it is ever flipped default-ON, give `PagedKvArena` its own
    /// `byte_size` and sum it here rather than estimating it.
    ///
    /// `KvStorage::None` returns **0**: its buffers live on the parent
    /// `KvCache::decode_fp16_k/v` and are counted by `KvCache::resident_bytes`,
    /// which also adds the warm-TTFT fp16 decode seeds for quantized variants.
    pub fn resident_bytes(&self) -> u64 {
        self.view()
            .slots
            .iter()
            .flatten()
            .map(|slot| slot.resident_bytes())
            .sum()
    }

    /// Drop every packed payload this variant holds, leaving its geometry
    /// (bit widths, group sizes) intact.
    ///
    /// `exit_prefill` calls this on the path where it decides **not** to build
    /// a store, and that is not housekeeping: `enter_prefill` does not clear
    /// `storage`, so a cache that already carries a payload — an SSD-hydrated
    /// entry that was deep-cloned and tail-extended — would otherwise come out
    /// of the second prefill holding a store of the *old* length beside a
    /// mirror of the new one. The spill writer prefers the store whenever it is
    /// populated, so that block would be written under the full prompt's hash
    /// while holding only the prefix, and hydrate would hand it back with a
    /// shorter `offset` and no error anywhere. Clearing at the source removes
    /// the divergence instead of teaching each reader to detect it.
    ///
    /// [`Self::view_mut`] places every slot, so a new payload slot is cleared
    /// here, or a stale copy of it survives the skip.
    pub fn clear_payload(&mut self) {
        for slot in self.view_mut().slots.into_iter().flatten() {
            slot.clear();
        }
    }

    /// True when this layer holds **no packed payload**, so the only thing
    /// there is to persist about it is its geometry.
    ///
    /// Three situations reach it, and the SSD spill writer treats them alike:
    ///
    /// * a rotating (SWA) layer — its KV lives in the bf16 ring off `storage`,
    ///   which is not serialisable, so the window is re-established on reuse;
    /// * a codec whose decode reads only the bf16 mirror
    ///   ([`crate::KvQuant::materialises_packed_store`] is `false`), so
    ///   `exit_prefill` built no store to write;
    /// * [`KvStorage::None`], which never had one.
    ///
    /// The K-side slot is the indicator throughout: every two-sided variant
    /// populates both slots in the same `exit_prefill` statement, so a `None`
    /// on K means the whole layer is empty. The two variants whose payload is
    /// not an `Option` — `Mixed` and `Paged` — always answer `false`
    /// here: their writers serialise their own empty state, and diverting them
    /// would change what an unfilled layer of theirs round-trips as.
    ///
    /// The answer is the first slot's [`KvSlot::geometry_only`], and `true`
    /// for a variant with no slot. [`Self::view`] places every slot, so a new
    /// variant is classified there: otherwise the writer would stamp a codec
    /// geometry with no tensors behind it and the reader would fail on the
    /// first missing tensor.
    #[must_use]
    pub fn is_geometry_only(&self) -> bool {
        let [first, _, _] = self.view().slots;
        first.is_none_or(KvSlot::geometry_only)
    }

    /// The one read-only match from a variant to its store slots, in K, V,
    /// second-V order. The read-only sites (bytes, graph flush, the SSD
    /// geometry decision, the probes, the variant name) go through it.
    ///
    /// Every arm binds every field and has no `..` rest pattern, so a field
    /// added to a variant does not compile here until it is placed. The KV
    /// census refuses a `..` in this fn.
    pub(crate) fn view(&self) -> StorageView<'_> {
        match self {
            Self::K8V4 { k, v } => StorageView {
                name: "K8V4",
                slots: [Some(k), Some(v), None],
            },
            Self::K8V8 { k, v } => StorageView {
                name: "K8V8",
                slots: [Some(k), Some(v), None],
            },
            Self::Planar { k, v, bits: _ } => StorageView {
                name: "Planar",
                slots: [Some(k), Some(v), None],
            },
            Self::None {} => StorageView {
                name: "None",
                slots: [None, None, None],
            },
            Self::Mixed { state } => StorageView {
                name: "Mixed",
                slots: [Some(state), None, None],
            },
            Self::Paged {
                quant: _,
                k,
                v_k8,
                v_planar,
            } => StorageView {
                name: "Paged",
                slots: [Some(k), Some(v_k8), Some(v_planar)],
            },
            Self::K8VTurbo3 { k, v } => StorageView {
                name: "K8VTurbo3",
                slots: [Some(k), Some(v), None],
            },
            Self::TurboSym3 { k, v } => StorageView {
                name: "TurboSym3",
                slots: [Some(k), Some(v), None],
            },
            Self::TurboSym4 { k, v } => StorageView {
                name: "TurboSym4",
                slots: [Some(k), Some(v), None],
            },
            Self::PlanarK { k } => StorageView {
                name: "PlanarK",
                slots: [Some(k), None, None],
            },
            Self::K8VTurbo2 { k, v } => StorageView {
                name: "K8VTurbo2",
                slots: [Some(k), Some(v), None],
            },
            Self::IsoV3 { k, v } => StorageView {
                name: "IsoV3",
                slots: [Some(k), Some(v), None],
            },
            Self::IsoV4 { k, v } => StorageView {
                name: "IsoV4",
                slots: [Some(k), Some(v), None],
            },
            Self::RotorV3 { k, v } => StorageView {
                name: "RotorV3",
                slots: [Some(k), Some(v), None],
            },
            Self::RotorV4 { k, v } => StorageView {
                name: "RotorV4",
                slots: [Some(k), Some(v), None],
            },
            Self::K8VTurbo3Tcq { k, v } => StorageView {
                name: "K8VTurbo3Tcq",
                slots: [Some(k), Some(v), None],
            },
            Self::K8VTurbo2Tcq { k, v } => StorageView {
                name: "K8VTurbo2Tcq",
                slots: [Some(k), Some(v), None],
            },
            Self::IsoSym3 { k, v } => StorageView {
                name: "IsoSym3",
                slots: [Some(k), Some(v), None],
            },
            Self::IsoSym4 { k, v } => StorageView {
                name: "IsoSym4",
                slots: [Some(k), Some(v), None],
            },
            Self::IsoKOnly3 { k } => StorageView {
                name: "IsoKOnly3",
                slots: [Some(k), None, None],
            },
            Self::IsoKOnly4 { k } => StorageView {
                name: "IsoKOnly4",
                slots: [Some(k), None, None],
            },
            Self::RotorSym3 { k, v } => StorageView {
                name: "RotorSym3",
                slots: [Some(k), Some(v), None],
            },
            Self::RotorSym4 { k, v } => StorageView {
                name: "RotorSym4",
                slots: [Some(k), Some(v), None],
            },
            Self::RotorKOnly3 { k } => StorageView {
                name: "RotorKOnly3",
                slots: [Some(k), None, None],
            },
            Self::RotorKOnly4 { k } => StorageView {
                name: "RotorKOnly4",
                slots: [Some(k), None, None],
            },
            Self::RotorKAsym3 {
                k,
                v,
                v_bits: _,
                v_group_size: _,
            } => StorageView {
                name: "RotorKAsym3",
                slots: [Some(k), Some(v), None],
            },
            Self::RotorKAsym4 {
                k,
                v,
                v_bits: _,
                v_group_size: _,
            } => StorageView {
                name: "RotorKAsym4",
                slots: [Some(k), Some(v), None],
            },
        }
    }

    /// The one mutating match from a variant to its store slots, in the same
    /// K, V, second-V order as [`Self::view`]. `reset`, `truncate_to` and
    /// `clear_payload` go through it.
    ///
    /// The same binding rule as [`Self::view`]: every arm binds every field and
    /// has no `..`, and the KV census refuses a `..` in this fn.
    pub(crate) fn view_mut(&mut self) -> StorageViewMut<'_> {
        match self {
            Self::K8V4 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::K8V8 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::Planar { k, v, bits: _ } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::None {} => StorageViewMut {
                slots: [None, None, None],
            },
            Self::Mixed { state } => StorageViewMut {
                slots: [Some(state), None, None],
            },
            Self::Paged {
                quant: _,
                k,
                v_k8,
                v_planar,
            } => StorageViewMut {
                slots: [Some(k), Some(v_k8), Some(v_planar)],
            },
            Self::K8VTurbo3 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::TurboSym3 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::TurboSym4 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::PlanarK { k } => StorageViewMut {
                slots: [Some(k), None, None],
            },
            Self::K8VTurbo2 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::IsoV3 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::IsoV4 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::RotorV3 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::RotorV4 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::K8VTurbo3Tcq { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::K8VTurbo2Tcq { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::IsoSym3 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::IsoSym4 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::IsoKOnly3 { k } => StorageViewMut {
                slots: [Some(k), None, None],
            },
            Self::IsoKOnly4 { k } => StorageViewMut {
                slots: [Some(k), None, None],
            },
            Self::RotorSym3 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::RotorSym4 { k, v } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::RotorKOnly3 { k } => StorageViewMut {
                slots: [Some(k), None, None],
            },
            Self::RotorKOnly4 { k } => StorageViewMut {
                slots: [Some(k), None, None],
            },
            Self::RotorKAsym3 {
                k,
                v,
                v_bits: _,
                v_group_size: _,
            } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
            Self::RotorKAsym4 {
                k,
                v,
                v_bits: _,
                v_group_size: _,
            } => StorageViewMut {
                slots: [Some(k), Some(v), None],
            },
        }
    }
}

/// Read-only view of one [`KvStorage`], built by [`KvStorage::view`].
pub(crate) struct StorageView<'a> {
    /// The variant name that typed errors print.
    pub(crate) name: &'static str,
    /// The K slot, the V slot and the paged planar V slot; `None` where the
    /// variant has no such slot. The scalar knobs (`bits`, `v_bits`,
    /// `v_group_size`, the paged `quant`) are not slots.
    pub(crate) slots: [Option<&'a dyn KvSlot>; 3],
}

/// Mutable view of one [`KvStorage`], built by [`KvStorage::view_mut`].
pub(crate) struct StorageViewMut<'a> {
    /// The K slot, the V slot and the paged planar V slot; `None` where the
    /// variant has no such slot.
    pub(crate) slots: [Option<&'a mut dyn KvSlot>; 3],
}
