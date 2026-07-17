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
use rmlx_mlx::Array;

use super::{
    QuantIsoK3, QuantIsoK4, QuantIsoV3, QuantIsoV4, QuantK, QuantKTurbo3, QuantKTurbo4,
    QuantPlanarK, QuantPlanarV, QuantRotorK3, QuantRotorK4, QuantRotorV3, QuantRotorV4, QuantV,
};
use crate::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};
use crate::KvQuant;

// ── Byte-accounting helper ────────────────────────────────────────────────────

/// Actual on-device bytes for a single `Array` — shape product × dtype item size.
///
/// Works entirely from array metadata (no FFI eval, no data read). Returns 0
/// for zero-dimensional or empty arrays. Used by [`KvStorage::resident_bytes`].
#[inline]
fn array_nbytes(a: &Array) -> u64 {
    let n: u64 = a.shape().iter().map(|&d| d as u64).product();
    n * a.dtype().itemsize() as u64
}

/// Layout tag for symmetric TurboQuant 3-bit K + turbo3 V.
///
/// Both K and V use the same Lloyd-Max N(0,1) 3-bit codebook (axis-agnostic
/// turbo3 kernel). Distinct from `"k8vturbo3"` (asymmetric K8V turbo3) and
/// from [`TURBOSYM4_LAYOUT_TAG`] (4-bit symmetric) so the SSD reader can
/// dispatch to the correct symmetric 3-bit hydrate path. Format
/// `"<codec>_wht_<k_bits>_<v_bits>"`.
pub const TURBOSYM3_LAYOUT_TAG: &str = "tsym3_wht_3_3";

/// Layout tag for symmetric TurboQuant 4-bit K + tq4 V.
///
/// Single source of truth for the SSD geometry tag (used by `KvBlockWriter` /
/// `KvBlockReader` and the layout-key tier). Format
/// `"<codec>_wht_<k_bits>_<v_bits>"`.
pub const TURBOSYM4_LAYOUT_TAG: &str = "tsym4_wht_4_4";

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
pub const ISOV4_LAYOUT_TAG: &str = "iso_v_4";

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
pub const ISO_SYM_4_LAYOUT_TAG: &str = "iso_sym_4";

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

/// Asymmetric rotor3 K + affine V layout tag prefix
/// (QJL OFF). The full tag carries the V (bits, group) tuple appended as
/// `_v{v_bits}g{v_group_size}` so the SSD reader can pick the affine V
/// codec on hydrate. Distinct from [`ROTOR_K_ONLY_3_LAYOUT_TAG`] because
/// V-side payload differs (affine 3-tuple vs bf16-on-parent).
pub const ROTOR_K_ASYM_3_LAYOUT_PREFIX: &str = "rotor_k_asym_3";

/// Asymmetric rotor3 K + affine V layout tag prefix (QJL residual ON).
pub const ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX: &str = "rotor_k_asym_3_qjl";

/// Asymmetric rotor4 K + affine V layout tag prefix (QJL OFF).
pub const ROTOR_K_ASYM_4_LAYOUT_PREFIX: &str = "rotor_k_asym_4";

/// Asymmetric rotor4 K + affine V layout tag prefix (QJL residual ON).
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
/// The unquantised `Full` variant was removed and later restored as `None`
/// — opt-in only for bf16-KV parity benching.
/// Auto-resolver still picks K8V8 / K8V4 / Planar by default.
pub enum KvStorage {
    /// K = affine q8_0, V = TurboQuant 4-bit.
    K8V4 {
        k: Option<QuantK>,
        v: Option<QuantV>,
        max_seq: i32,
    },
    /// K = affine q8_0, V = affine q8_0.
    K8V8 {
        k: Option<QuantK>,
        v: Option<QuantK>,
        max_seq: i32,
    },
    /// K = affine q8_0, V = PlanarQuant N-bit (S3.4).
    ///
    /// `bits ∈ {3, 4}`: 4 = original Planar codec (`KvQuant::Planar`);
    /// 3 = new 3.25-bit variant (`KvQuant::Planar3`).
    ///
    /// Opt-in via `--kv-quant planar` (4-bit) or `--kv-quant planar3` (3-bit).
    /// Default arch selection unchanged (auto = Planar 4-bit on eligible archs).
    Planar {
        k: Option<QuantK>,
        v: Option<QuantPlanarV>,
        max_seq: i32,
        /// Bit-width for the V codec: 3 (Planar3) or 4 (legacy default).
        bits: u8,
    },
    /// Unquantised bf16 cache. The actual `Array` buffers live in
    /// `KvCache::decode_fp16_k` / `decode_fp16_v`, reusing the same machinery
    /// as the warm-TTFT fp16 decode-seed path. This variant just records the
    /// max_seq so the dispatch is uniform with the other variants.
    None { max_seq: i32 },
    /// Mixed-precision K/V via `mx.quantize` 3-tuples.
    ///
    /// State is owned by [`crate::mixed_quant::MixedKvState`]. `max_seq` is
    /// recorded for symmetry with the other variants but unused by the mixed
    /// path — buffers grow in `STEP=256` increments matching mlx-lm-tq.
    Mixed {
        state: crate::mixed_quant::MixedKvState,
        #[allow(dead_code)]
        max_seq: i32,
    },
    /// PagedAttention block-table KV storage (paged KV path, `--paged-kv`).
    ///
    /// K is always q8_0 (PagedKStorage). V variant is chosen by quant mode:
    /// - K8V4 / K8V8 → PagedVStorage (TurboQuant V4 for K8V4, q8_0 for K8V8).
    /// - Planar → PagedPlanarVStorage.
    ///
    /// For single-request decoding this degenerates to contiguous behaviour.
    /// The block table is monotonically appended (no sharing/eviction) in this
    /// phase. Real cross-request sharing arrives in continuous batching.
    Paged {
        quant: KvQuant,
        k: Option<PagedKStorage>,
        v_k8: Option<Box<PagedVStorage>>,
        v_planar: Option<Box<PagedPlanarVStorage>>,
        max_seq: i32,
    },
    /// RotK K (MLX affine 8-bit, rotated basis) + TurboQuant 4-bit V.
    ///
    /// K-side: [`MixedKvState`](crate::mixed_quant::MixedKvState) with `rotate_k=true`
    /// (same as [`Mixed`](KvStorage::Mixed) / [`KvQuant::RotK`](KvQuant::RotK)).
    /// V-side: [`QuantV`] (TurboFlash MSL tq4, same as [`K8V4`](KvStorage::K8V4) V).
    ///
    /// SDPA: K is dequantized from its affine 3-tuple (codes, scales, biases) to
    /// bf16, Q is pre-rotated, V is dequantized from TurboFlash to bf16, then
    /// standard `scaled_dot_product_attention` runs. This is a dequant-then-SDPA
    /// path rather than the fused `quantized_matmul` path of plain Mixed / RotK.
    RotKTq4V {
        k_state: crate::mixed_quant::MixedKvState,
        v: Option<QuantV>,
        max_seq: i32,
    },
    /// bench prototype: K = affine q8_0 (group_size=128),
    /// V = TurboQuant 3-bit Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Structurally identical to [`K8V4`](KvStorage::K8V4) but with `bits=3`
    /// in the [`QuantV`] slot. No MSL kernel for 3-bit — CPU dequant only.
    K8VTurbo3 {
        k: Option<QuantK>,
        v: Option<QuantV>,
        max_seq: i32,
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
    /// check in `rmlx_models::kv_cache::validate_resolved_kv_quant` (symmetric
    /// 3-bit K is the PPL-disaster path on Qwen MoE, same as 4-bit symmetric).
    ///
    /// **V-side GPU dispatch**: V side is forced CPU path (same as the asymmetric
    /// `K8VTurbo3` precedent — GPU V-side dispatch regressed −2% on K8VTurbo3).
    /// K side uses the GPU turbo3 MSL kernel when `Device::Gpu` is in effect.
    TurboSym3 {
        k: Option<QuantKTurbo3>,
        v: Option<QuantV>,
        max_seq: i32,
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
    /// check in `rmlx_models::kv_cache::validate_resolved_kv_quant`
    /// — symmetric 4-bit K is the PPL-218→8641 disaster path.
    TurboSym4 {
        k: Option<QuantKTurbo4>,
        v: Option<QuantV>,
        max_seq: i32,
    },
    /// K = PlanarQuant 4-bit, V = unquantised bf16 (mtq `k_only_planar`).
    ///
    /// Opposite of [`Planar`](KvStorage::Planar): Givens-rotation 4-bit codec on
    /// the **K** axis; V stays full-precision bf16 (lives on the parent
    /// `KvCache::decode_fp16_v`, same machinery as [`None`](KvStorage::None)
    /// for V). PlanarK is a PPL-disaster on Qwen MoE — see arch guard in
    /// `cache_type::validate_resolved`. `head_dim % 32 == 0` is required
    /// (PlanarQuant block constraint).
    ///
    /// Opt-in only via `--kv-quant planar_k`. Never an auto default.
    PlanarK {
        k: Option<QuantPlanarK>,
        max_seq: i32,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = TurboQuant **2-bit** Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Structurally identical to [`K8V4`](KvStorage::K8V4) but with `bits=2`
    /// in the [`QuantV`] slot. Native 2.25-bit V codec; ships naïve (no
    /// outlier-mask); outlier-mask deferred pending calibration loader.
    /// CPU dequant only on the hot path; MSL kernel exists as future-ref hook.
    K8VTurbo2 {
        k: Option<QuantK>,
        v: Option<QuantV>,
        max_seq: i32,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = IsoQuant 3-bit (quaternion SO(4) rotation + Lloyd-Max codebook).
    ///
    /// CPU-only — the V codec dispatches via [`QuantIsoV3::dequant`] and falls
    /// through to the dequant-then-SDPA legacy path. SSD spill/hydrate and the
    /// MSL kernel are deferred.
    IsoV3 {
        k: Option<QuantK>,
        v: Option<QuantIsoV3>,
        max_seq: i32,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = IsoQuant 4-bit (quaternion SO(4) rotation + Lloyd-Max 4-bit codebook).
    ///
    /// Same machinery as [`IsoV3`](KvStorage::IsoV3) with `bits=4` and the
    /// dense 8-vals-per-u32 pack. CPU-only — the existing MSL kernel is
    /// hard-coded for `bits=3`; an iso4 MSL kernel is deferred.
    IsoV4 {
        k: Option<QuantK>,
        v: Option<QuantIsoV4>,
        max_seq: i32,
    },
    /// Symmetric IsoQuant 3-bit — both K and V use the same
    /// quaternion SO(4) + 3-bit Lloyd-Max codebook (axis-agnostic codec).
    ///
    /// K is stored in `QuantIsoK3`; V in `QuantIsoV3`. SDPA falls through to
    /// the dequant-then-SDPA legacy path (same as `IsoV3`). CPU-only.
    /// Layout tag: [`ISO_SYM_3_LAYOUT_TAG`].
    IsoSym3 {
        k: Option<QuantIsoK3>,
        v: Option<QuantIsoV3>,
        max_seq: i32,
    },
    /// Symmetric IsoQuant 4-bit — both K and V use the same
    /// quaternion SO(4) + 4-bit Lloyd-Max codebook (dense 8-vals-per-u32 pack).
    ///
    /// Layout tag: [`ISO_SYM_4_LAYOUT_TAG`]. CPU-only.
    IsoSym4 {
        k: Option<QuantIsoK4>,
        v: Option<QuantIsoV4>,
        max_seq: i32,
    },
    /// K-only IsoQuant 3-bit; V is bf16 on the parent
    /// `KvCache::decode_fp16_v` (same machinery as `KvStorage::None` /
    /// `KvStorage::PlanarK` for V).
    ///
    /// Layout tag: [`ISO_K_ONLY_3_LAYOUT_TAG`]. Opt-in only.
    IsoKOnly3 { k: Option<QuantIsoK3>, max_seq: i32 },
    /// K-only IsoQuant 4-bit; V is bf16 on the parent
    /// `KvCache::decode_fp16_v`.
    ///
    /// Layout tag: [`ISO_K_ONLY_4_LAYOUT_TAG`].
    IsoKOnly4 { k: Option<QuantIsoK4>, max_seq: i32 },
    /// Symmetric rotor3 — both K and V use the same Cl(3,0)
    /// Clifford rotor sandwich + 3-bit Lloyd-Max codebook (axis-agnostic
    /// codec). The K side optionally carries a 1-bit QJL residual per
    /// element (`QuantRotorK3::qjl_s_matrix.is_some()`).
    ///
    /// Layout tag: [`ROTOR_SYM_3_LAYOUT_TAG`] or [`ROTOR_SYM_3_QJL_LAYOUT_TAG`].
    /// CPU-only on both axes. SDPA falls through the dequant-then-SDPA path.
    RotorSym3 {
        k: Option<QuantRotorK3>,
        v: Option<QuantRotorV3>,
        max_seq: i32,
    },
    /// Symmetric rotor4 — same shape as `RotorSym3` with the
    /// 4-bit codebook on both axes.
    ///
    /// Layout tag: [`ROTOR_SYM_4_LAYOUT_TAG`] or [`ROTOR_SYM_4_QJL_LAYOUT_TAG`].
    RotorSym4 {
        k: Option<QuantRotorK4>,
        v: Option<QuantRotorV4>,
        max_seq: i32,
    },
    /// K-only rotor3 — K is rotor3; V stays bf16 on the parent
    /// `KvCache::decode_fp16_v` (same machinery as `IsoKOnly3` /
    /// `PlanarK` / `None`).
    ///
    /// Decode reads the packed K store directly via the rotor flash-decode MSL
    /// kernel when the store carries no QJL sideband; a QJL store keeps the CPU
    /// dequant path. See `docs/KV_QUANT.md` § `rotor_flash_decode`.
    ///
    /// Layout tag: [`ROTOR_K_ONLY_3_LAYOUT_TAG`] or
    /// [`ROTOR_K_ONLY_3_QJL_LAYOUT_TAG`].
    RotorKOnly3 {
        k: Option<QuantRotorK3>,
        max_seq: i32,
    },
    /// K-only rotor4 — same shape as `RotorKOnly3` with 4-bit
    /// codes.
    ///
    /// Layout tag: [`ROTOR_K_ONLY_4_LAYOUT_TAG`] or
    /// [`ROTOR_K_ONLY_4_QJL_LAYOUT_TAG`].
    RotorKOnly4 {
        k: Option<QuantRotorK4>,
        max_seq: i32,
    },
    /// Asymmetric rotor3 K + affine V — K is rotor3 (CPU; optional
    /// QJL residual sideband); V is MLX-affine `QuantV` at `v_bits` /
    /// `v_group_size` (reuses the existing affine V encode/decode path).
    ///
    /// Layout key prefix: [`ROTOR_K_ASYM_3_LAYOUT_PREFIX`] /
    /// [`ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX`]; the full SSD layout key suffixes
    /// `_v{v_bits}g{v_group_size}` so hydrate can pick the affine V codec.
    /// SDPA falls through to the legacy dequant-then-SDPA path (K rotor3 +
    /// affine V both dequant to bf16 before `scaled_dot_product_attention`).
    RotorKAsym3 {
        k: Option<QuantRotorK3>,
        v: Option<QuantV>,
        max_seq: i32,
        /// V quantization bit-width (affine).
        v_bits: u8,
        /// V affine group size.
        v_group_size: u16,
    },
    /// Asymmetric rotor4 K + affine V — same shape as
    /// `RotorKAsym3` with the dense 4-bit rotor codebook on K.
    ///
    /// Layout key prefix: [`ROTOR_K_ASYM_4_LAYOUT_PREFIX`] /
    /// [`ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX`].
    RotorKAsym4 {
        k: Option<QuantRotorK4>,
        v: Option<QuantV>,
        max_seq: i32,
        /// V quantization bit-width (affine).
        v_bits: u8,
        /// V affine group size.
        v_group_size: u16,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = rotor3 (Cl(3,0) Clifford rotor sandwich + 3-bit Lloyd-Max codebook).
    ///
    /// Static per-layer rotor table on the V side (lazily generated on first
    /// append; never per-token). CPU-only — MSL kernel deferred (single-bit
    /// pack convention matches planar3 / iso3 for future kernel reuse).
    /// SDPA falls through to the dequant-then-SDPA legacy fallback path.
    RotorV3 {
        k: Option<QuantK>,
        v: Option<QuantRotorV3>,
        max_seq: i32,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = rotor4 (Cl(3,0) Clifford rotor sandwich + 4-bit Lloyd-Max codebook).
    ///
    /// 4.25-bit V codec — same Clifford sandwich as rotor3 with the
    /// 16-centroid Lloyd-Max N(0,1) codebook and dense 8-vals-per-u32 pack
    /// (iso4 convention). Higher fidelity than rotor3 at the cost of one extra
    /// bit per value (~10.7 bpe pre-scale at bits=4).
    ///
    /// Static per-layer rotor table on the V side (lazily generated on first
    /// append). CPU-only — MSL kernel deferred per rotor3 / iso4 precedent.
    /// SDPA falls through to the dequant-then-SDPA legacy fallback path.
    ///
    /// Paged route: falls through (same paged-KV gate deferral as rotor3).
    RotorV4 {
        k: Option<QuantK>,
        v: Option<QuantRotorV4>,
        max_seq: i32,
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
        max_seq: i32,
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
        max_seq: i32,
    },
}

impl KvStorage {
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn new(quant: KvQuant, max_seq: i32) -> Self {
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
                        max_seq,
                    };
                }
                _ => {}
            }
        }

        match quant {
            KvQuant::K8V4 => Self::K8V4 {
                k: None,
                v: None,
                max_seq,
            },
            KvQuant::K8V8 => Self::K8V8 {
                k: None,
                v: None,
                max_seq,
            },
            KvQuant::Planar => Self::Planar {
                k: None,
                v: None,
                max_seq,
                bits: 4,
            },
            // Planar3 routes to Planar storage with bits=3.
            // No new KvStorage variant — the existing Planar variant carries bits.
            KvQuant::Planar3 => Self::Planar {
                k: None,
                v: None,
                max_seq,
                bits: 3,
            },
            KvQuant::None => Self::None { max_seq },
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
                max_seq,
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
                max_seq,
            },
            // RotKTq4V — rotated K (MLX affine 8-bit) + TurboQuant 4-bit V.
            // -review MEDIUM 2: use new_k_only_rotated() — V is stored
            // separately as QuantV; the v_bits/v_group_size params were dead.
            KvQuant::RotKTq4V => Self::RotKTq4V {
                k_state: crate::mixed_quant::MixedKvState::new_k_only_rotated(),
                v: None,
                max_seq,
            },
            // K8VTurbo3 — same layout as K8V4 but QuantV bits=3.
            KvQuant::K8VTurbo3 => Self::K8VTurbo3 {
                k: None,
                v: None,
                max_seq,
            },
            // TurboSym3 — symmetric WHT-3 K+V. Never routes through
            // the paged path: PagedKStorage is q8-only and there is no paged
            // TurboQuant-K3 variant. Deviation documented in docs/KV_QUANT.md.
            KvQuant::TurboSym3 => Self::TurboSym3 {
                k: None,
                v: None,
                max_seq,
            },
            // TurboSym4 — symmetric WHT-4 K+V. Never routes through
            // the paged path: PagedKStorage is q8-only and adding a TurboQuant-K
            // paged variant is out of scope; deviation documented in
            // docs/KV_QUANT.md.
            KvQuant::TurboSym4 => Self::TurboSym4 {
                k: None,
                v: None,
                max_seq,
            },
            // PlanarK — K-axis PlanarQuant 4-bit; V is bf16 on the parent KvCache.
            // Never routes through paged: PagedKStorage is q8-only and there is no
            // paged PlanarQuant-K variant; deviation documented in docs/KV_QUANT.md.
            KvQuant::PlanarK => Self::PlanarK { k: None, max_seq },
            // K8VTurbo2 — same layout as K8V4 but QuantV bits=2.
            KvQuant::K8VTurbo2 => Self::K8VTurbo2 {
                k: None,
                v: None,
                max_seq,
            },
            // Iso3 — K = affine q8_0, V = IsoQuant 3-bit.
            KvQuant::Iso3 => Self::IsoV3 {
                k: None,
                v: None,
                max_seq,
            },
            // Iso4 — K = affine q8_0, V = IsoQuant 4-bit.
            KvQuant::Iso4 => Self::IsoV4 {
                k: None,
                v: None,
                max_seq,
            },
            // Rotor3 — K = affine q8_0, V = rotor3 (Cl(3,0) rotor).
            // Does NOT route through the paged path: PagedVStorage is q8/tq4-only
            // and PagedPlanarVStorage is PlanarQuant-only; a paged RotorV3 would
            // need its own per-token (codes/scales/norms) container plus a static
            // per-layer rotor table inside the paged arena. Deferred per the
            // iso3 / iso4 precedent — opt-in only via --kv-quant rotor3, never
            // an auto baseline.
            KvQuant::Rotor3 => Self::RotorV3 {
                k: None,
                v: None,
                max_seq,
            },
            // Rotor4 — K = affine q8_0, V = rotor4 (Cl(3,0) rotor, 4-bit).
            // Same paged-KV deferral as Rotor3 — falls through.
            KvQuant::Rotor4 => Self::RotorV4 {
                k: None,
                v: None,
                max_seq,
            },
            // K8VTurbo3Tcq — same layout as K8VTurbo3 with Viterbi encode-side
            // assignment. The `use_tcq` flag is set on the `QuantV` slot lazily
            // at first `append` (see `update_k8vturbo3_tcq` in `kvcache/update.rs`).
            KvQuant::K8VTurbo3Tcq => Self::K8VTurbo3Tcq {
                k: None,
                v: None,
                max_seq,
            },
            // K8VTurbo2Tcq — same layout as K8VTurbo2 with Viterbi encode-side
            // assignment. The `use_tcq` flag is set on the `QuantV` slot lazily
            // at first `append` (see `update_k8vturbo2_tcq` in `kvcache/update.rs`).
            KvQuant::K8VTurbo2Tcq => Self::K8VTurbo2Tcq {
                k: None,
                v: None,
                max_seq,
            },
            // Iso3Sym — K = iso3, V = iso3 (axis-agnostic).
            KvQuant::Iso3Sym => Self::IsoSym3 {
                k: None,
                v: None,
                max_seq,
            },
            // Iso4Sym — K = iso4, V = iso4.
            KvQuant::Iso4Sym => Self::IsoSym4 {
                k: None,
                v: None,
                max_seq,
            },
            // IsoKOnly3 — K = iso3; V bf16 lives on the parent
            // `KvCache::decode_fp16_v` (same machinery as PlanarK / None).
            KvQuant::IsoKOnly3 => Self::IsoKOnly3 { k: None, max_seq },
            // IsoKOnly4 — K = iso4; V bf16 on the parent.
            KvQuant::IsoKOnly4 => Self::IsoKOnly4 { k: None, max_seq },
            // Rotor3Sym — K = rotor3, V = rotor3 (axis-agnostic).
            KvQuant::Rotor3Sym => Self::RotorSym3 {
                k: None,
                v: None,
                max_seq,
            },
            // Rotor4Sym — K = rotor4, V = rotor4.
            KvQuant::Rotor4Sym => Self::RotorSym4 {
                k: None,
                v: None,
                max_seq,
            },
            // RotorKOnly3 — K = rotor3; V bf16 lives on the parent
            // `KvCache::decode_fp16_v` (same machinery as IsoKOnly3 / PlanarK).
            KvQuant::RotorKOnly3 => Self::RotorKOnly3 { k: None, max_seq },
            // RotorKOnly4 — K = rotor4; V bf16 on the parent.
            KvQuant::RotorKOnly4 => Self::RotorKOnly4 { k: None, max_seq },
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
                    max_seq,
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
                    max_seq,
                    v_bits,
                    v_group_size,
                }
            }
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "long match enumerates all KvStorage variants; splitting would obscure the 1-to-1 mapping"
    )]
    pub fn reset(&mut self) {
        match self {
            // Quant paths: zero the accumulated shape but keep the GPU buffers
            // so the next request reuses the same allocation. Each Quant struct's
            // own append() will overwrite the prefix from offset=0.
            Self::K8V4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            Self::K8V8 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            Self::Planar { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // None: no quant state — the bf16 buffers live on KvCache and are
            // dropped/reset by KvCache::reset directly.
            Self::None { .. } => {}
            Self::Mixed { state, .. } => state.reset(),
            Self::Paged {
                k, v_k8, v_planar, ..
            } => {
                if let Some(pk) = k.as_mut() {
                    pk.reset();
                }
                if let Some(pv) = v_k8.as_mut() {
                    pv.reset();
                }
                if let Some(pv) = v_planar.as_mut() {
                    pv.reset();
                }
            }
            Self::RotKTq4V { k_state, v, .. } => {
                k_state.reset();
                if let Some(qv) = v.as_mut() {
                    qv.shape[2] = 0;
                }
            }
            // K8VTurbo3 resets like K8V4.
            Self::K8VTurbo3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // TurboSym3 — symmetric reset (K3 + V3 shape-zeroing).
            Self::TurboSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // TurboSym4 — symmetric reset (same shape-zeroing).
            Self::TurboSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // PlanarK — K only; V (bf16) lives on parent KvCache.
            Self::PlanarK { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
            }
            // K8VTurbo2 resets like K8V4.
            Self::K8VTurbo2 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // IsoV3 — K is q8_0; V holds CPU IsoBlocks (reset clears them so
            // the next request starts fresh).
            Self::IsoV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // IsoV4 — same shape semantics as IsoV3.
            Self::IsoV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // RotorV3 — K shape zeroes; V codec resets blocks but KEEPS the
            // static rotor table.
            Self::RotorV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // RotorV4 — same semantics as RotorV3.
            Self::RotorV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // K8VTurbo3Tcq resets like K8VTurbo3 / K8V4.
            Self::K8VTurbo3Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // K8VTurbo2Tcq resets like K8VTurbo2 / K8VTurbo3Tcq.
            Self::K8VTurbo2Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = 0;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            // IsoSym3/IsoSym4 reset both K + V iso buffers.
            Self::IsoSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            Self::IsoSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // IsoKOnly3/4 — K iso buffer only; V bf16 lives on parent.
            Self::IsoKOnly3 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
            }
            Self::IsoKOnly4 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
            }
            // RotorSym3 / RotorSym4 — reset both K + V rotor buffers (each has
            // its own per-token blocks; rotor table + QJL matrix are layer-static
            // and kept).
            Self::RotorSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            Self::RotorSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // RotorKOnly3/4 — K only; V bf16 on parent.
            Self::RotorKOnly3 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
            }
            Self::RotorKOnly4 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
            }
            // RotorKAsym3 / RotorKAsym4 — K rotor reset; V affine shape-zero
            // (same as K8V4 V-side).
            Self::RotorKAsym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
            Self::RotorKAsym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = 0;
                }
            }
        }
    }

    /// Truncate the sequence dimension to `n` tokens.
    ///
    /// Sets `shape[2] = n` so the next `append` call overwrites positions
    /// `[n..]`. The GPU buffers are kept in place (no reallocation) because
    /// `append` uses `slice_update` with a position offset derived from
    /// `shape[2]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "long match enumerates all KvStorage variants; splitting would obscure the 1-to-1 mapping"
    )]
    pub fn truncate_to(&mut self, n: i32) {
        match self {
            Self::K8V4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            Self::K8V8 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            Self::Planar { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // None: bf16 buffers are sliced lazily on next read; nothing to
            // truncate here. KvCache::truncate_to drops the buffers itself.
            Self::None { .. } => {}
            // Mixed: drop quant state (mlx-lm-tq's `is_trimmable` returns False;
            // a fresh request resets the cache anyway).
            Self::Mixed { state, .. } => state.reset(),
            Self::Paged {
                k, v_k8, v_planar, ..
            } => {
                if let Some(pk) = k.as_mut() {
                    pk.truncate_to(n);
                }
                if let Some(pv) = v_k8.as_mut() {
                    pv.truncate_to(n);
                }
                if let Some(pv) = v_planar.as_mut() {
                    pv.truncate_to(n);
                }
            }
            // RotKTq4V: same as Mixed — reset state on truncate.
            Self::RotKTq4V { k_state, v, .. } => {
                k_state.reset();
                if let Some(qv) = v.as_mut() {
                    qv.shape[2] = 0;
                }
            }
            // K8VTurbo3 truncates like K8V4.
            Self::K8VTurbo3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // TurboSym3 — symmetric truncate (K3 + V3 shape).
            Self::TurboSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // TurboSym4 — symmetric truncate (same shape semantics).
            Self::TurboSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // PlanarK — truncate K only; V (bf16) sliced lazily.
            Self::PlanarK { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
            }
            // K8VTurbo2 truncates like K8V4.
            Self::K8VTurbo2 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // IsoV3 — K shape truncates; V codec is per-token so dropping
            // trailing blocks is delegated to QuantIsoV3.
            Self::IsoV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // IsoV4 — same shape semantics as IsoV3.
            Self::IsoV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // RotorV3 — K shape truncates; V codec drops trailing blocks
            // (rotor table kept).
            Self::RotorV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // RotorV4 — same semantics as RotorV3.
            Self::RotorV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // K8VTurbo3Tcq truncates like K8VTurbo3 / K8V4.
            Self::K8VTurbo3Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // K8VTurbo2Tcq truncates like K8VTurbo2 / K8VTurbo3Tcq.
            Self::K8VTurbo2Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.shape[2] = n;
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            // IsoSym3 / IsoSym4 — both axes are per-token block codecs;
            // delegate to each side's truncate_to (same as IsoV3).
            Self::IsoSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            Self::IsoSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // IsoKOnly3 / IsoKOnly4 — K only; V bf16 sliced lazily on parent.
            Self::IsoKOnly3 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
            }
            Self::IsoKOnly4 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
            }
            // RotorSym3 / RotorSym4 — both axes are per-token block codecs;
            // delegate to each side's truncate_to.
            Self::RotorSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            Self::RotorSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // RotorKOnly3 / RotorKOnly4 — K only; V bf16 sliced lazily on parent.
            Self::RotorKOnly3 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
            }
            Self::RotorKOnly4 { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
            }
            // RotorKAsym3 / RotorKAsym4 — K rotor truncate; V affine
            // shape-truncate (same as K8V4 V-side).
            Self::RotorKAsym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
            Self::RotorKAsym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.shape[2] = n;
                }
            }
        }
    }

    #[allow(
        clippy::cognitive_complexity,
        reason = "single match over the closed KvStorage enum — one arm per variant, each is small and self-contained; splitting would hide the 1-to-1 mapping"
    )]
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(match self {
            Self::K8V4 { k, v, max_seq } => Self::K8V4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            Self::K8V8 { k, v, max_seq } => Self::K8V8 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            Self::Planar {
                k,
                v,
                max_seq,
                bits,
            } => Self::Planar {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
                bits: *bits,
            },
            Self::None { max_seq } => Self::None { max_seq: *max_seq },
            Self::Mixed { state, max_seq } => Self::Mixed {
                state: state.try_deep_clone()?,
                max_seq: *max_seq,
            },
            // Paged: for speculative decoding clone, return a fresh Paged storage.
            // The block-table state is not cloneable efficiently with the page-slab
            // design; callers that need true deep-clone of paged state should
            // re-populate from model forward passes. This returns an empty shell
            // consistent with the existing QuantK::try_deep_clone semantics (which
            // clone GPU buffers but the CPU path is also valid).
            Self::Paged { quant, max_seq, .. } => Self::Paged {
                quant: *quant,
                k: None,
                v_k8: None,
                v_planar: None,
                max_seq: *max_seq,
            },
            Self::RotKTq4V {
                k_state,
                v,
                max_seq,
            } => Self::RotKTq4V {
                k_state: k_state.try_deep_clone()?,
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // K8VTurbo3 deep-clones like K8V4.
            Self::K8VTurbo3 { k, v, max_seq } => Self::K8VTurbo3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // TurboSym3 — deep-clone both QuantKTurbo3 + QuantV (bits=3).
            Self::TurboSym3 { k, v, max_seq } => Self::TurboSym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // TurboSym4 — deep-clone both QuantKTurbo4 + QuantV.
            Self::TurboSym4 { k, v, max_seq } => Self::TurboSym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // PlanarK — deep-clone K only; V (bf16) on parent.
            Self::PlanarK { k, max_seq } => Self::PlanarK {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // K8VTurbo2 deep-clones like K8V4.
            Self::K8VTurbo2 { k, v, max_seq } => Self::K8VTurbo2 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // IsoV3 — deep-clone the K (QuantK) + V (QuantIsoV3).
            Self::IsoV3 { k, v, max_seq } => Self::IsoV3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // IsoV4 — deep-clone the K (QuantK) + V (QuantIsoV4).
            Self::IsoV4 { k, v, max_seq } => Self::IsoV4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // RotorV3 — deep-clone K (QuantK) + V (QuantRotorV3).
            Self::RotorV3 { k, v, max_seq } => Self::RotorV3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // RotorV4 — deep-clone K (QuantK) + V (QuantRotorV4).
            Self::RotorV4 { k, v, max_seq } => Self::RotorV4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // K8VTurbo3Tcq deep-clones like K8VTurbo3.
            Self::K8VTurbo3Tcq { k, v, max_seq } => Self::K8VTurbo3Tcq {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // K8VTurbo2Tcq deep-clones like K8VTurbo2 / K8VTurbo3Tcq.
            Self::K8VTurbo2Tcq { k, v, max_seq } => Self::K8VTurbo2Tcq {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // IsoSym3 / IsoSym4 — deep-clone both K + V iso buffers.
            Self::IsoSym3 { k, v, max_seq } => Self::IsoSym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            Self::IsoSym4 { k, v, max_seq } => Self::IsoSym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // IsoKOnly3 / IsoKOnly4 — K only.
            Self::IsoKOnly3 { k, max_seq } => Self::IsoKOnly3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            Self::IsoKOnly4 { k, max_seq } => Self::IsoKOnly4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // RotorSym3 / RotorSym4 — deep-clone both K + V rotor buffers.
            Self::RotorSym3 { k, v, max_seq } => Self::RotorSym3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            Self::RotorSym4 { k, v, max_seq } => Self::RotorSym4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                v: match v {
                    Some(qv) => Some(qv.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // RotorKOnly3 / RotorKOnly4 — K only.
            Self::RotorKOnly3 { k, max_seq } => Self::RotorKOnly3 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            Self::RotorKOnly4 { k, max_seq } => Self::RotorKOnly4 {
                k: match k {
                    Some(qk) => Some(qk.try_deep_clone()?),
                    None => None,
                },
                max_seq: *max_seq,
            },
            // RotorKAsym3 / RotorKAsym4 — deep-clone K rotor + V affine.
            // V codec parameters (bits / group) carry forward unchanged.
            Self::RotorKAsym3 {
                k,
                v,
                max_seq,
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
                max_seq: *max_seq,
                v_bits: *v_bits,
                v_group_size: *v_group_size,
            },
            Self::RotorKAsym4 {
                k,
                v,
                max_seq,
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
                max_seq: *max_seq,
                v_bits: *v_bits,
                v_group_size: *v_group_size,
            },
        })
    }

    /// Actual on-device byte footprint of the quantized storage buffers.
    ///
    /// Sums the bytes of every allocated buffer in the storage variant:
    /// - For GPU-backed variants (`QuantK`, `QuantV`, `QuantPlanarV`, …):
    ///   actual `Array` shape × dtype item-size (allocated at `gpu_capacity`,
    ///   not just filled tokens — the allocator uses power-of-two pages).
    /// - For CPU-backed variants (`IsoBlocks`, `RotorBlocks`, …):
    ///   sum of all `Vec` element sizes in bytes.
    /// - `KvStorage::None` returns **0**: its buffers live on the parent
    ///   `KvCache::decode_fp16_k/v` and are counted by `KvCache::resident_bytes`.
    ///
    /// Does **not** count the warm-TTFT fp16 decode-seed buffers
    /// (`KvCache::decode_fp16_k/v`) — those are tallied separately in
    /// [`KvCache::resident_bytes`] for all quantized variants.
    #[allow(
        clippy::too_many_lines,
        reason = "exhaustive match over all KvStorage variants — LOC-exempt: KvStorage has 30 variants; each arm is a one-to-three line byte summation that cannot be factored further without losing explicitness"
    )]
    pub fn resident_bytes(&self) -> u64 {
        match self {
            // ── Unquantised (bf16) ─────────────────────────────────────────────
            // Buffers live on KvCache::decode_fp16_k/v; nothing extra here.
            KvStorage::None { .. } => 0,

            // ── K8V8 (K = q8_0, V = q8_0; V uses QuantK not QuantV) ─────────
            KvStorage::K8V8 { k, v, .. } => quant_k_bytes(k.as_ref()) + quant_k_bytes(v.as_ref()),

            // ── K8V4 / K8VTurbo* (K = q8_0, V = TurboQuant) ─────────────────
            KvStorage::K8V4 { k, v, .. }
            | KvStorage::K8VTurbo3 { k, v, .. }
            | KvStorage::K8VTurbo3Tcq { k, v, .. }
            | KvStorage::K8VTurbo2 { k, v, .. }
            | KvStorage::K8VTurbo2Tcq { k, v, .. } => {
                quant_k_bytes(k.as_ref()) + quant_v_bytes(v.as_ref())
            }

            // ── Planar (K=q8, V=PlanarQuant) ─────────────────────────────────
            KvStorage::Planar { k, v, .. } => {
                quant_k_bytes(k.as_ref()) + quant_planar_v_bytes(v.as_ref())
            }

            // ── PlanarK (K=PlanarQuant, V=bf16 on KvCache) ───────────────────
            KvStorage::PlanarK { k, .. } => quant_planar_k_bytes(k.as_ref()),

            // ── Mixed (MLX mx.quantize 3-tuples, opt. RotK) ───────────────────
            KvStorage::Mixed { state, .. } => mixed_kv_state_bytes(state),

            // ── RotKTq4V (rotated-K 3-tuple + TurboQuant V4) ─────────────────
            KvStorage::RotKTq4V { k_state, v, .. } => {
                mixed_kv_state_bytes(k_state) + quant_v_bytes(v.as_ref())
            }

            // ── Symmetric Turbo (K=TurboK3/4, V=TurboV) ─────────────────────
            KvStorage::TurboSym3 { k, v, .. } => {
                quant_k_turbo3_bytes(k.as_ref()) + quant_v_bytes(v.as_ref())
            }
            KvStorage::TurboSym4 { k, v, .. } => {
                quant_k_turbo4_bytes(k.as_ref()) + quant_v_bytes(v.as_ref())
            }

            // ── IsoQuant V (K=q8, V=Iso3/4) ──────────────────────────────────
            KvStorage::IsoV3 { k, v, .. } => {
                quant_k_bytes(k.as_ref()) + quant_iso_v3_bytes(v.as_ref())
            }
            KvStorage::IsoV4 { k, v, .. } => {
                quant_k_bytes(k.as_ref()) + quant_iso_v4_bytes(v.as_ref())
            }

            // ── IsoQuant Sym (K=IsoK3/4, V=IsoV3/4) ─────────────────────────
            KvStorage::IsoSym3 { k, v, .. } => {
                quant_iso_k3_bytes(k.as_ref()) + quant_iso_v3_bytes(v.as_ref())
            }
            KvStorage::IsoSym4 { k, v, .. } => {
                quant_iso_k4_bytes(k.as_ref()) + quant_iso_v4_bytes(v.as_ref())
            }

            // ── IsoKOnly (K=Iso3/4, V=bf16 on KvCache) ───────────────────────
            KvStorage::IsoKOnly3 { k, .. } => quant_iso_k3_bytes(k.as_ref()),
            KvStorage::IsoKOnly4 { k, .. } => quant_iso_k4_bytes(k.as_ref()),

            // ── RotorV (K=q8, V=Rotor3/4) ────────────────────────────────────
            KvStorage::RotorV3 { k, v, .. } => {
                quant_k_bytes(k.as_ref()) + quant_rotor_v3_bytes(v.as_ref())
            }
            KvStorage::RotorV4 { k, v, .. } => {
                quant_k_bytes(k.as_ref()) + quant_rotor_v4_bytes(v.as_ref())
            }

            // ── RotorSym (K=RotorK3/4, V=RotorV3/4) ─────────────────────────
            KvStorage::RotorSym3 { k, v, .. } => {
                quant_rotor_k3_bytes(k.as_ref()) + quant_rotor_v3_bytes(v.as_ref())
            }
            KvStorage::RotorSym4 { k, v, .. } => {
                quant_rotor_k4_bytes(k.as_ref()) + quant_rotor_v4_bytes(v.as_ref())
            }

            // ── RotorKOnly (K=RotorK3/4, V=bf16 on KvCache) ─────────────────
            KvStorage::RotorKOnly3 { k, .. } => quant_rotor_k3_bytes(k.as_ref()),
            KvStorage::RotorKOnly4 { k, .. } => quant_rotor_k4_bytes(k.as_ref()),

            // ── RotorKAsym (K=RotorK3/4, V=affine QuantV) ────────────────────
            KvStorage::RotorKAsym3 { k, v, .. } => {
                quant_rotor_k3_bytes(k.as_ref()) + quant_v_bytes(v.as_ref())
            }
            KvStorage::RotorKAsym4 { k, v, .. } => {
                quant_rotor_k4_bytes(k.as_ref()) + quant_v_bytes(v.as_ref())
            }

            // ── Paged (block-table KV, --paged-kv path) ───────────────────────
            KvStorage::Paged {
                k, v_k8, v_planar, ..
            } => {
                let k_bytes = k.as_ref().map_or(0, PagedKStorage::resident_bytes);
                // v_k8 / v_planar are Box-wrapped; closure used to deref through Box.
                let vk8_bytes = v_k8.as_ref().map_or(0, |s| s.resident_bytes());
                let vp_bytes = v_planar.as_ref().map_or(0, |s| s.resident_bytes());
                k_bytes + vk8_bytes + vp_bytes
            }
        }
    }
}

// ── Per-type byte-counting helpers ────────────────────────────────────────────
//
// Each function accepts an `Option<&T>` (None = storage not yet populated →
// returns 0) and sums the actual buffer bytes for that codec type.

/// Affine q8_0 K buffer (`QuantK`).
///
/// GPU path: `gpu_codes_buf` (u32) + `gpu_scales_buf` (f32) — sized to
/// `gpu_capacity`, not just `offset`. CPU path: `codes` (u8) + `scales` (f32).
///
/// NOTE: After an SSD-hydrate init the GPU mirror is live (`gpu_codes_buf =
/// Some`) *and* the pre-hydration CPU `codes`/`scales` are still resident in
/// RAM (the hydrate upload path never clears them). Both allocations are
/// counted.
fn quant_k_bytes(qk: Option<&QuantK>) -> u64 {
    let Some(qk) = qk else {
        return 0;
    };
    if let Some(ref codes) = qk.gpu_codes_buf {
        let scales_bytes = qk.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (pre-hydration data not cleared on
        // GPU-mirror init; see `QuantK::append_inner` hydrated-init block).
        let cpu_residual = qk.codes.len() as u64 + qk.scales.len() as u64 * 4;
        array_nbytes(codes) + scales_bytes + cpu_residual
    } else {
        // CPU path: codes = Vec<u8>, scales = Vec<f32>
        qk.codes.len() as u64 + qk.scales.len() as u64 * 4
    }
}

/// TurboQuant V buffer (`QuantV`), used for K8V4 / K8V8 / K8VTurbo* / RotorKAsym.
///
/// GPU path: `gpu_codes_buf` (u32) + `gpu_scales_buf` (f32). CPU path: sum of
/// `TurboBlocks` — each block has `codes: Vec<u8>` (1 B/elem) and
/// `scales: Vec<f32>` (4 B/elem).
///
/// NOTE: After an SSD-hydrate init the GPU mirror is live and the pre-hydration
/// CPU `blocks` are still resident in RAM. Both are counted.
fn quant_v_bytes(qv: Option<&QuantV>) -> u64 {
    let Some(qv) = qv else {
        return 0;
    };
    if let Some(ref codes) = qv.gpu_codes_buf {
        let scales_bytes = qv.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (pre-hydration blocks not cleared on
        // GPU-mirror init; see `QuantV::append_inner` hydrated-init block).
        let cpu_residual: u64 = qv
            .blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4)
            .sum();
        array_nbytes(codes) + scales_bytes + cpu_residual
    } else {
        // CPU path: TurboBlocks — codes: Vec<u8>, scales: Vec<f32>
        qv.blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4)
            .sum()
    }
}

/// PlanarQuant V buffer (`QuantPlanarV`), used for `Planar` variant.
///
/// GPU path: `gpu_codes_buf` (u32) + `gpu_scales_buf` (f32) +
/// `gpu_rotations_buf` (u32). CPU path: sum of `PlanarBlocks` — each has
/// `codes: Vec<u8>` (1 B), `scales: Vec<f32>` (4 B), `rotations: Vec<u8>` (1 B).
///
/// NOTE: After an SSD-hydrate init the GPU mirror is live and the pre-hydration
/// CPU `blocks` are still resident in RAM. Both are counted.
fn quant_planar_v_bytes(qv: Option<&QuantPlanarV>) -> u64 {
    let Some(qv) = qv else {
        return 0;
    };
    if let Some(ref codes) = qv.gpu_codes_buf {
        let scales_bytes = qv.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        let rot_bytes = qv.gpu_rotations_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (pre-hydration blocks not cleared on
        // GPU-mirror init; see `QuantPlanarV::append` hydrated-init block).
        let cpu_residual: u64 = qv
            .blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4 + b.rotations.len() as u64)
            .sum();
        array_nbytes(codes) + scales_bytes + rot_bytes + cpu_residual
    } else {
        qv.blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4 + b.rotations.len() as u64)
            .sum()
    }
}

/// PlanarQuant K buffer (`QuantPlanarK`), used for `PlanarK` variant.
///
/// Same buffer layout as `QuantPlanarV` — GPU or CPU with codes/scales/rotations.
///
/// NOTE: After an SSD-hydrate init the GPU mirror is live and the pre-hydration
/// CPU `blocks` are still resident in RAM. Both are counted.
fn quant_planar_k_bytes(qk: Option<&QuantPlanarK>) -> u64 {
    let Some(qk) = qk else {
        return 0;
    };
    if let Some(ref codes) = qk.gpu_codes_buf {
        let scales_bytes = qk.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        let rot_bytes = qk.gpu_rotations_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (pre-hydration blocks not cleared on
        // GPU-mirror init; see `QuantPlanarK::append` hydrated-init block).
        let cpu_residual: u64 = qk
            .blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4 + b.rotations.len() as u64)
            .sum();
        array_nbytes(codes) + scales_bytes + rot_bytes + cpu_residual
    } else {
        qk.blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4 + b.rotations.len() as u64)
            .sum()
    }
}

/// MLX mx.quantize 3-tuple state (`MixedKvState`).
///
/// Each `MixedTuple` has three `Array`s: codes (u32), scales (bf16/f32), biases
/// (bf16/f32). Also includes the optional per-layer Hadamard rotation matrix
/// (`k_rotation`: `Array`).
fn mixed_kv_state_bytes(state: &crate::mixed_quant::MixedKvState) -> u64 {
    fn tuple_bytes(t: &crate::mixed_quant::MixedTuple) -> u64 {
        array_nbytes(&t.codes) + array_nbytes(&t.scales) + array_nbytes(&t.biases)
    }
    let k_bytes = state.keys.as_ref().map_or(0, tuple_bytes);
    let v_bytes = state.values.as_ref().map_or(0, tuple_bytes);
    let rot_bytes = state.k_rotation.as_ref().map_or(0, array_nbytes);
    k_bytes + v_bytes + rot_bytes
}

/// TurboQuant 3-bit K buffer (`QuantKTurbo3`).
///
/// GPU path: `gpu_codes_buf` (u32) + `gpu_scales_buf` (f32).
/// CPU path: sum of `TurboBlocks`.
///
/// NOTE: After an SSD-hydrate init the GPU mirror is live and the pre-hydration
/// CPU `blocks` are still resident in RAM. Both are counted.
fn quant_k_turbo3_bytes(qk: Option<&QuantKTurbo3>) -> u64 {
    let Some(qk) = qk else {
        return 0;
    };
    if let Some(ref codes) = qk.gpu_codes_buf {
        let scales_bytes = qk.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (pre-hydration blocks not cleared on
        // GPU-mirror init; see `QuantKTurbo3::append` hydrated-init block).
        let cpu_residual: u64 = qk
            .blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4)
            .sum();
        array_nbytes(codes) + scales_bytes + cpu_residual
    } else {
        qk.blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4)
            .sum()
    }
}

/// TurboQuant 4-bit K buffer (`QuantKTurbo4`).
///
/// Same buffer layout as `QuantKTurbo3` — GPU codes/scales or CPU TurboBlocks.
///
/// NOTE: After an SSD-hydrate init the GPU mirror is live and the pre-hydration
/// CPU `blocks` are still resident in RAM. Both are counted.
fn quant_k_turbo4_bytes(qk: Option<&QuantKTurbo4>) -> u64 {
    let Some(qk) = qk else {
        return 0;
    };
    if let Some(ref codes) = qk.gpu_codes_buf {
        let scales_bytes = qk.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (pre-hydration blocks not cleared on
        // GPU-mirror init; see `QuantKTurbo4::append` hydrated-init block).
        let cpu_residual: u64 = qk
            .blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4)
            .sum();
        array_nbytes(codes) + scales_bytes + cpu_residual
    } else {
        qk.blocks
            .iter()
            .map(|b| b.codes.len() as u64 + b.scales.len() as u64 * 4)
            .sum()
    }
}

/// IsoQuant 3-bit V buffer (`QuantIsoV3`).
///
/// GPU path (when `gpu_resident_iso` gate enabled): `gpu_codes_buf` (u32) +
/// `gpu_scales_buf` (f32) + `gpu_norms_buf` (f32). CPU path: sum of
/// `IsoBlocks` — each has `codes: Vec<u32>` (4 B), `scales: Vec<f32>` (4 B),
/// `quaternions: Vec<f32>` (4 B), `norms: Vec<f32>` (4 B).
///
/// NOTE: `QuantIsoV3::gpu_offset` advances independently from `blocks` when
/// CPU-only blocks are also appended, e.g. after an SSD-hydrate fallback
/// (see `quant_iso_v.rs` field doc on `gpu_offset`). Both GPU-mirror and any
/// coexisting CPU `blocks` are resident simultaneously, so both are counted.
fn quant_iso_v3_bytes(qv: Option<&QuantIsoV3>) -> u64 {
    let Some(qv) = qv else {
        return 0;
    };
    // GPU path (crate-private fields accessed from within the same crate).
    if let Some(ref codes) = qv.gpu_codes_buf {
        let scales_bytes = qv.gpu_scales_buf.as_ref().map_or(0, array_nbytes);
        let norms_bytes = qv.gpu_norms_buf.as_ref().map_or(0, array_nbytes);
        // Add any coexisting CPU residual (blocks can be non-empty under a live
        // GPU mirror after an SSD-hydrate fallback — gpu_offset advances
        // independently; CPU blocks remain resident).
        let cpu_residual: u64 = qv
            .blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4
                    + b.scales.len() as u64 * 4
                    + b.quaternions.len() as u64 * 4
                    + b.norms.len() as u64 * 4
            })
            .sum();
        return array_nbytes(codes) + scales_bytes + norms_bytes + cpu_residual;
    }
    // CPU path.
    qv.blocks
        .iter()
        .map(|b| {
            b.codes.len() as u64 * 4
                + b.scales.len() as u64 * 4
                + b.quaternions.len() as u64 * 4
                + b.norms.len() as u64 * 4
        })
        .sum()
}

/// IsoQuant 4-bit V buffer (`QuantIsoV4`, CPU-only).
fn quant_iso_v4_bytes(qv: Option<&QuantIsoV4>) -> u64 {
    qv.map_or(0, |q| {
        q.blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4
                    + b.scales.len() as u64 * 4
                    + b.quaternions.len() as u64 * 4
                    + b.norms.len() as u64 * 4
            })
            .sum()
    })
}

/// IsoQuant 3-bit K buffer (`QuantIsoK3`, CPU-only). Same layout as IsoV3 blocks.
fn quant_iso_k3_bytes(qk: Option<&QuantIsoK3>) -> u64 {
    qk.map_or(0, |q| {
        q.blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4
                    + b.scales.len() as u64 * 4
                    + b.quaternions.len() as u64 * 4
                    + b.norms.len() as u64 * 4
            })
            .sum()
    })
}

/// IsoQuant 4-bit K buffer (`QuantIsoK4`, CPU-only). Same block layout.
fn quant_iso_k4_bytes(qk: Option<&QuantIsoK4>) -> u64 {
    qk.map_or(0, |q| {
        q.blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4
                    + b.scales.len() as u64 * 4
                    + b.quaternions.len() as u64 * 4
                    + b.norms.len() as u64 * 4
            })
            .sum()
    })
}

/// Rotor3 V buffer (`QuantRotorV3`, CPU-only).
///
/// Includes the static per-layer rotor table (`rotors: Vec<f32>`, 4 B each)
/// and the per-token `RotorBlocks` payload: `codes: Vec<u32>` (4 B),
/// `scales: Vec<f32>` (4 B), `norms: Vec<f32>` (4 B).
fn quant_rotor_v3_bytes(qv: Option<&QuantRotorV3>) -> u64 {
    qv.map_or(0, |q| {
        let rotor_bytes = q.rotors.len() as u64 * 4;
        let block_bytes: u64 = q
            .blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4 + b.scales.len() as u64 * 4 + b.norms.len() as u64 * 4
            })
            .sum();
        rotor_bytes + block_bytes
    })
}

/// Rotor4 V buffer (`QuantRotorV4`, CPU-only). Same layout as RotorV3.
fn quant_rotor_v4_bytes(qv: Option<&QuantRotorV4>) -> u64 {
    qv.map_or(0, |q| {
        let rotor_bytes = q.rotors.len() as u64 * 4;
        let block_bytes: u64 = q
            .blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4 + b.scales.len() as u64 * 4 + b.norms.len() as u64 * 4
            })
            .sum();
        rotor_bytes + block_bytes
    })
}

/// Rotor3 K buffer (`QuantRotorK3`).
///
/// Counts the CPU blocks only. The GPU-resident ring (`QuantKGpuRing`) that backs
/// the rotor flash-decode is accounted by `QuantRotorK3::byte_size`.
///
/// Includes: static `rotors: Vec<f32>` (4 B), optional QJL projection matrix
/// `qjl_s_matrix: Vec<f32>` (4 B), and per-token `RotorKBlocks`:
/// `codes: Vec<u32>` (4 B), `scales: Vec<f32>` (4 B), `norms: Vec<f32>` (4 B),
/// `qjl_codes: Vec<u8>` (1 B), `qjl_norms: Vec<f32>` (4 B).
fn quant_rotor_k3_bytes(qk: Option<&QuantRotorK3>) -> u64 {
    qk.map_or(0, |q| {
        let rotor_bytes = q.rotors.len() as u64 * 4;
        let qjl_mat_bytes = q.qjl_s_matrix.as_ref().map_or(0, |m| m.len() as u64 * 4);
        let block_bytes: u64 = q
            .blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4
                    + b.scales.len() as u64 * 4
                    + b.norms.len() as u64 * 4
                    + b.qjl_codes.len() as u64
                    + b.qjl_norms.len() as u64 * 4
            })
            .sum();
        rotor_bytes + qjl_mat_bytes + block_bytes
    })
}

/// Rotor4 K buffer (`QuantRotorK4`). Same layout as RotorK3; same GPU-ring
/// accounting note.
fn quant_rotor_k4_bytes(qk: Option<&QuantRotorK4>) -> u64 {
    qk.map_or(0, |q| {
        let rotor_bytes = q.rotors.len() as u64 * 4;
        let qjl_mat_bytes = q.qjl_s_matrix.as_ref().map_or(0, |m| m.len() as u64 * 4);
        let block_bytes: u64 = q
            .blocks
            .iter()
            .map(|b| {
                b.codes.len() as u64 * 4
                    + b.scales.len() as u64 * 4
                    + b.norms.len() as u64 * 4
                    + b.qjl_codes.len() as u64
                    + b.qjl_norms.len() as u64 * 4
            })
            .sum();
        rotor_bytes + qjl_mat_bytes + block_bytes
    })
}
