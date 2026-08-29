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
    /// bit per value in the codes (~10.7 bpe at bits=4). Storage is unaffected:
    /// both widths spend one `u32` code word plus one
    /// [`crate::storage::KV_SIDEBAND_DTYPE`] scale per group, so rotor3 and
    /// rotor4 occupy byte-identical bytes — 16.25 bits per value at
    /// `head_dim = 128`. See `crate::rotorquant` § "Effective bpe".
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
    /// Clear the accumulated sequence, keeping the allocations.
    ///
    /// Every arm goes through the store's own `truncate_to(0)` / `reset()`, for
    /// the same reason `truncate_to` does: zeroing `shape[2]` alone leaves the
    /// CPU-side payload — a block list, or `QuantK`'s flat `codes`/`scales` —
    /// covering the sequence that was just discarded, so the next `append`
    /// stacks on top of it and the dequant reads the discarded tokens back.
    /// `truncate_to(0)` cuts that payload as well. The GPU buffers are still
    /// kept in place so the next request reuses the same allocation; the next
    /// `append` overwrites their prefix from offset 0.
    pub fn reset(&mut self) {
        match self {
            Self::K8V4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            Self::K8V8 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            Self::Planar { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
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
            // K8VTurbo3 resets like K8V4.
            Self::K8VTurbo3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            // TurboSym3 — symmetric reset (K3 + V3 shape-zeroing).
            Self::TurboSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            // TurboSym4 — symmetric reset (same shape-zeroing).
            Self::TurboSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            // PlanarK — K only; V (bf16) lives on parent KvCache.
            Self::PlanarK { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
            }
            // K8VTurbo2 resets like K8V4.
            Self::K8VTurbo2 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            // IsoV3 — K is q8_0; V holds CPU IsoBlocks (reset clears them so
            // the next request starts fresh).
            Self::IsoV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // IsoV4 — same shape semantics as IsoV3.
            Self::IsoV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // RotorV3 — K shape zeroes; V codec resets blocks but KEEPS the
            // static rotor table.
            Self::RotorV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // RotorV4 — same semantics as RotorV3.
            Self::RotorV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.reset();
                }
            }
            // K8VTurbo3Tcq resets like K8VTurbo3 / K8V4.
            Self::K8VTurbo3Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
            // K8VTurbo2Tcq resets like K8VTurbo2 / K8VTurbo3Tcq.
            Self::K8VTurbo2Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(0);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
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
                    vs.truncate_to(0);
                }
            }
            Self::RotorKAsym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.reset();
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(0);
                }
            }
        }
    }

    /// Truncate the sequence dimension to `n` tokens.
    ///
    /// Every arm delegates to the store's own `truncate_to`, which lowers
    /// `shape[2]` to `n` **and** cuts whatever CPU-side state accumulates
    /// independently of it. The GPU buffers are kept in place (no reallocation)
    /// because `append` uses `slice_update` with a position offset derived from
    /// `shape[2]`; the CPU-side blocks / codes are append-only and have to be
    /// cut, or the next `append` stacks on top of the rejected tokens.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "long match enumerates all KvStorage variants; splitting would obscure the 1-to-1 mapping"
    )]
    pub fn truncate_to(&mut self, n: i32) {
        // Clamp the negative case once, here, so no arm can compute from a
        // negative `n` before delegating.
        //
        // The upper clamp is NOT uniform, and the divergence is worth naming
        // rather than papering over. The turbo / planar / affine stores clamp
        // `n` down to their own `shape[2]` (`storage::clamp_truncate_target`);
        // the rotor / iso stores deliberately do not, because they abort loudly
        // on an over-long target instead. So for `n > shape[2]` the mixed arms
        // leave the two axes of one codec at different lengths: `IsoV3`,
        // `IsoV4`, `RotorV3`, `RotorV4` (affine K clamps, codec V does not) and
        // `RotorKAsym3` / `RotorKAsym4` (rotor K does not, affine V does). That
        // matters on spill, where the layer geometry is derived from the K shape
        // while the V payload is written raw — the reconciliation guard on the
        // unclamped side is what surfaces it.
        let n = n.max(0);
        match self {
            Self::K8V4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            Self::K8V8 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            Self::Planar { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // None: bf16 buffers are sliced lazily on next read; nothing to
            // truncate here. KvCache::truncate_to drops the buffers itself.
            Self::None { .. } => {}
            // Mixed: the store is a capacity buffer with `state.offset` as its
            // fill marker, so rolling that marker back IS the truncation — see
            // `MixedKvState::truncate_to`. Resetting instead dropped the kept
            // prefix too, and `KvCache::truncate_to` then set `offset = n`,
            // leaving a cache that reports `n` positions and holds none.
            Self::Mixed { state, .. } => state.truncate_to(n),
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
            // K8VTurbo3 truncates like K8V4.
            Self::K8VTurbo3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // TurboSym3 — symmetric truncate (K3 + V3 shape).
            Self::TurboSym3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // TurboSym4 — symmetric truncate (same shape semantics).
            Self::TurboSym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // PlanarK — truncate K only; V (bf16) sliced lazily.
            Self::PlanarK { k, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
            }
            // K8VTurbo2 truncates like K8V4.
            Self::K8VTurbo2 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // IsoV3 — K shape truncates; V codec is per-token so dropping
            // trailing blocks is delegated to QuantIsoV3.
            Self::IsoV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // IsoV4 — same shape semantics as IsoV3.
            Self::IsoV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // RotorV3 — K shape truncates; V codec drops trailing blocks
            // (rotor table kept).
            Self::RotorV3 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // RotorV4 — same semantics as RotorV3.
            Self::RotorV4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // K8VTurbo3Tcq truncates like K8VTurbo3 / K8V4.
            Self::K8VTurbo3Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
                }
            }
            // K8VTurbo2Tcq truncates like K8VTurbo2 / K8VTurbo3Tcq.
            Self::K8VTurbo2Tcq { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
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
                    vs.truncate_to(n);
                }
            }
            Self::RotorKAsym4 { k, v, .. } => {
                if let Some(ks) = k.as_mut() {
                    ks.truncate_to(n);
                }
                if let Some(vs) = v.as_mut() {
                    vs.truncate_to(n);
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

    /// Resident byte footprint of this variant's quantized storage.
    ///
    /// Every arm delegates to the owning store's own `byte_size`, which derives
    /// the total from that store's real allocations — CPU blocks, GPU mirrors
    /// and GPU rings alike. There is deliberately **no** per-codec byte formula
    /// here: a second, hand-maintained restatement of what each store holds is
    /// what let a store grow a GPU ring while this total stayed blind to it.
    /// The store owns its own size; this function only routes.
    ///
    /// GPU buffers count their full allocation (rings and mirrors are sized to
    /// capacity, not to the filled prefix) — that is the memory actually held.
    ///
    /// **Excludes paged-pool overhead.** The `Paged` arm counts the pages the
    /// slabs hold, not the arena bookkeeping around them. Paged KV
    /// (`--paged-kv`) is default-OFF, so that overhead is 0 on every normal
    /// path; if it is ever flipped default-ON, give `PagedKvArena` its own
    /// `byte_size` and sum it in the `Paged` arm rather than estimating it here.
    ///
    /// `KvStorage::None` returns **0**: its buffers live on the parent
    /// `KvCache::decode_fp16_k/v` and are counted by `KvCache::resident_bytes`,
    /// which also adds the warm-TTFT fp16 decode seeds for quantized variants.
    ///
    /// The arms bind every field explicitly (no `..` rest-patterns): adding a
    /// buffer to a variant is then a compile error here until it is accounted.
    #[allow(
        clippy::too_many_lines,
        reason = "exhaustive match over all KvStorage variants — LOC-exempt: KvStorage has 30 variants; each arm is a one-to-three line delegation that cannot be factored further without losing explicitness"
    )]
    pub fn resident_bytes(&self) -> u64 {
        match self {
            // ── Unquantised (bf16) ─────────────────────────────────────────────
            // Buffers live on KvCache::decode_fp16_k/v; nothing extra here.
            KvStorage::None { max_seq: _ } => 0,

            // ── K8V8 (K = q8_0, V = q8_0; V uses QuantK not QuantV) ─────────
            KvStorage::K8V8 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantK::byte_size) + opt_bytes(v.as_ref(), QuantK::byte_size)
            }

            // ── K8V4 / K8VTurbo* (K = q8_0, V = TurboQuant) ─────────────────
            KvStorage::K8V4 { k, v, max_seq: _ }
            | KvStorage::K8VTurbo3 { k, v, max_seq: _ }
            | KvStorage::K8VTurbo3Tcq { k, v, max_seq: _ }
            | KvStorage::K8VTurbo2 { k, v, max_seq: _ }
            | KvStorage::K8VTurbo2Tcq { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantK::byte_size) + opt_bytes(v.as_ref(), QuantV::byte_size)
            }

            // ── Planar (K=q8, V=PlanarQuant) ─────────────────────────────────
            KvStorage::Planar {
                k,
                v,
                max_seq: _,
                bits: _,
            } => {
                opt_bytes(k.as_ref(), QuantK::byte_size)
                    + opt_bytes(v.as_ref(), QuantPlanarV::byte_size)
            }

            // ── PlanarK (K=PlanarQuant, V=bf16 on KvCache) ───────────────────
            KvStorage::PlanarK { k, max_seq: _ } => opt_bytes(k.as_ref(), QuantPlanarK::byte_size),

            // ── Mixed (MLX mx.quantize 3-tuples, opt. RotK) ───────────────────
            KvStorage::Mixed { state, max_seq: _ } => state.byte_size(),

            // ── Symmetric Turbo (K=TurboK3/4, V=TurboV) ─────────────────────
            KvStorage::TurboSym3 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantKTurbo3::byte_size)
                    + opt_bytes(v.as_ref(), QuantV::byte_size)
            }
            KvStorage::TurboSym4 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantKTurbo4::byte_size)
                    + opt_bytes(v.as_ref(), QuantV::byte_size)
            }

            // ── IsoQuant V (K=q8, V=Iso3/4) ──────────────────────────────────
            KvStorage::IsoV3 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantK::byte_size)
                    + opt_bytes(v.as_ref(), QuantIsoV3::byte_size)
            }
            KvStorage::IsoV4 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantK::byte_size)
                    + opt_bytes(v.as_ref(), QuantIsoV4::byte_size)
            }

            // ── IsoQuant Sym (K=IsoK3/4, V=IsoV3/4) ─────────────────────────
            KvStorage::IsoSym3 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantIsoK3::byte_size)
                    + opt_bytes(v.as_ref(), QuantIsoV3::byte_size)
            }
            KvStorage::IsoSym4 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantIsoK4::byte_size)
                    + opt_bytes(v.as_ref(), QuantIsoV4::byte_size)
            }

            // ── IsoKOnly (K=Iso3/4, V=bf16 on KvCache) ───────────────────────
            KvStorage::IsoKOnly3 { k, max_seq: _ } => opt_bytes(k.as_ref(), QuantIsoK3::byte_size),
            KvStorage::IsoKOnly4 { k, max_seq: _ } => opt_bytes(k.as_ref(), QuantIsoK4::byte_size),

            // ── RotorV (K=q8, V=Rotor3/4) ────────────────────────────────────
            KvStorage::RotorV3 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantK::byte_size)
                    + opt_bytes(v.as_ref(), QuantRotorV3::byte_size)
            }
            KvStorage::RotorV4 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantK::byte_size)
                    + opt_bytes(v.as_ref(), QuantRotorV4::byte_size)
            }

            // ── RotorSym (K=RotorK3/4, V=RotorV3/4) ─────────────────────────
            KvStorage::RotorSym3 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantRotorK3::byte_size)
                    + opt_bytes(v.as_ref(), QuantRotorV3::byte_size)
            }
            KvStorage::RotorSym4 { k, v, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantRotorK4::byte_size)
                    + opt_bytes(v.as_ref(), QuantRotorV4::byte_size)
            }

            // ── RotorKOnly (K=RotorK3/4, V=bf16 on KvCache) ─────────────────
            KvStorage::RotorKOnly3 { k, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantRotorK3::byte_size)
            }
            KvStorage::RotorKOnly4 { k, max_seq: _ } => {
                opt_bytes(k.as_ref(), QuantRotorK4::byte_size)
            }

            // ── RotorKAsym (K=RotorK3/4, V=affine QuantV) ────────────────────
            KvStorage::RotorKAsym3 {
                k,
                v,
                max_seq: _,
                v_bits: _,
                v_group_size: _,
            } => {
                opt_bytes(k.as_ref(), QuantRotorK3::byte_size)
                    + opt_bytes(v.as_ref(), QuantV::byte_size)
            }
            KvStorage::RotorKAsym4 {
                k,
                v,
                max_seq: _,
                v_bits: _,
                v_group_size: _,
            } => {
                opt_bytes(k.as_ref(), QuantRotorK4::byte_size)
                    + opt_bytes(v.as_ref(), QuantV::byte_size)
            }

            // ── Paged (block-table KV, --paged-kv path) ───────────────────────
            KvStorage::Paged {
                k,
                v_k8,
                v_planar,
                quant: _,
                max_seq: _,
            } => {
                let k_bytes = k.as_ref().map_or(0, PagedKStorage::resident_bytes);
                // v_k8 / v_planar are Box-wrapped; closure used to deref through Box.
                let vk8_bytes = v_k8.as_ref().map_or(0, |s| s.resident_bytes());
                let vp_bytes = v_planar.as_ref().map_or(0, |s| s.resident_bytes());
                k_bytes + vk8_bytes + vp_bytes
            }
        }
    }

    /// Drop every packed payload this variant holds, leaving its geometry
    /// (`max_seq`, bit widths, group sizes) intact.
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
    /// Exhaustive on purpose: a new payload slot on any variant must be
    /// classified here, or a stale copy of it survives the skip.
    pub fn clear_payload(&mut self) {
        match self {
            KvStorage::None { .. } => {}
            KvStorage::K8V4 { k, v, .. }
            | KvStorage::K8VTurbo3 { k, v, .. }
            | KvStorage::K8VTurbo3Tcq { k, v, .. }
            | KvStorage::K8VTurbo2 { k, v, .. }
            | KvStorage::K8VTurbo2Tcq { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::K8V8 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::Planar { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::PlanarK { k, .. } => *k = None,
            KvStorage::TurboSym3 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::TurboSym4 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::IsoV3 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::IsoV4 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::IsoSym3 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::IsoSym4 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::IsoKOnly3 { k, .. } => *k = None,
            KvStorage::IsoKOnly4 { k, .. } => *k = None,
            KvStorage::RotorV3 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::RotorV4 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::RotorSym3 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::RotorSym4 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::RotorKOnly3 { k, .. } => *k = None,
            KvStorage::RotorKOnly4 { k, .. } => *k = None,
            KvStorage::RotorKAsym3 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            KvStorage::RotorKAsym4 { k, v, .. } => {
                *k = None;
                *v = None;
            }
            // Payload is not an `Option`; each owns a `reset`.
            KvStorage::Mixed { state, .. } => state.reset(),
            KvStorage::Paged {
                k, v_k8, v_planar, ..
            } => {
                *k = None;
                *v_k8 = None;
                *v_planar = None;
            }
        }
    }

    /// `Some(max_seq)` when this layer holds **no packed payload**, so the only
    /// thing there is to persist about it is its geometry.
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
    /// not an `Option` — `Mixed` and `Paged` — always answer `None`
    /// here: their writers serialise their own empty state, and diverting them
    /// would change what an unfilled layer of theirs round-trips as.
    ///
    /// Exhaustive on purpose: a new variant must be classified, or the writer
    /// would stamp a codec geometry with no tensors behind it and the reader
    /// would fail on the first missing tensor.
    #[must_use]
    pub fn geometry_only_max_seq(&self) -> Option<i32> {
        // `k.is_none()` is the test in every arm; a `Some` K means the layer
        // carries a real payload and belongs to its codec's writer.
        fn empty<T>(k: Option<&T>, max_seq: i32) -> Option<i32> {
            k.is_none().then_some(max_seq)
        }
        match self {
            KvStorage::None { max_seq } => Some(*max_seq),
            KvStorage::K8V4 { k, max_seq, .. }
            | KvStorage::K8V8 { k, max_seq, .. }
            | KvStorage::Planar { k, max_seq, .. }
            | KvStorage::K8VTurbo3 { k, max_seq, .. }
            | KvStorage::K8VTurbo3Tcq { k, max_seq, .. }
            | KvStorage::K8VTurbo2 { k, max_seq, .. }
            | KvStorage::K8VTurbo2Tcq { k, max_seq, .. }
            | KvStorage::IsoV3 { k, max_seq, .. }
            | KvStorage::IsoV4 { k, max_seq, .. }
            | KvStorage::RotorV3 { k, max_seq, .. }
            | KvStorage::RotorV4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::TurboSym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::TurboSym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::PlanarK { k, max_seq } => empty(k.as_ref(), *max_seq),
            KvStorage::IsoSym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::IsoSym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::IsoKOnly3 { k, max_seq } => empty(k.as_ref(), *max_seq),
            KvStorage::IsoKOnly4 { k, max_seq } => empty(k.as_ref(), *max_seq),
            KvStorage::RotorSym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::RotorSym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::RotorKOnly3 { k, max_seq } => empty(k.as_ref(), *max_seq),
            KvStorage::RotorKOnly4 { k, max_seq } => empty(k.as_ref(), *max_seq),
            KvStorage::RotorKAsym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            KvStorage::RotorKAsym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
            // Payload is not an Option — their own writers handle emptiness.
            KvStorage::Mixed { .. } | KvStorage::Paged { .. } => None,
        }
    }
}

/// Bytes of an optional store slot; an unpopulated slot (`None`) holds nothing.
fn opt_bytes<T>(slot: Option<&T>, byte_size: impl Fn(&T) -> u64) -> u64 {
    slot.map_or(0, byte_size)
}
