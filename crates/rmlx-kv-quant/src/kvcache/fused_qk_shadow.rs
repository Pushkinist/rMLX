// Head-major persistent K storage for fused-QK MSL kernels.
//
// Extended the shadow into a `{per_token, sideband_table}` split per
// the followup design (see
// `docs/research/fused-qk-storage-design.md` §"Update — fix cycle").
//
// The shadow now carries up to **four** GPU-resident arrays per `KvCache`:
//
//   * `k_codes`              — `u32 [B, kv_h, max_seq, codes_per_token]`
//   * `k_scales`             — `f32 [B, kv_h, max_seq, scales_per_token]`
//   * `sideband_norms`       — `f32 [B, kv_h, max_seq, 1]` (rotor only)
//   * `sideband_rotor_table` — `f32 [n_groups * 4]` (rotor only, static-per-layer)
//
// The `FusedQkFn` kernel signature uses 13 args; sidebands are passed as
// **separate `Option<&Array>` arguments** rather than being concatenated
// into the `k_scales` buffer at call time. The pre-fix concat path cost
// ~28 MB of CPU↔GPU marshaling per decode step at Bonsai 8B head_dim=128,
// n_groups=32, layers=26, kv_seq=8192 and swamped the kernel compute
// savings (12.3 TPS vs 63.5 TPS legacy — see commit f42aa0f).
//
// The widened signature delivers:
//   * `k_scales`      — flat `[B * kv_h * kv_seq * scales_per_token]`
//   * `k_norms`       — `Option<&Array>`; rotor passes
//                       `Some(per-token L2 norms)`, q8 / turbo pass `None`.
//   * `k_rotor_table` — `Option<&Array>`; rotor passes
//                       `Some([n_groups * 4])` static-per-layer table,
//                       q8 / turbo pass `None`.
//
// Layout summary by codec family (which sidebands the shim consumes):
//
//   q8 / turbo3 / turbo4: per-token scales only — `has_norm = false`,
//                         `has_rotor_table = false`. Shim ignores
//                         `k_norms` and `k_rotor_table`.
//   rotor-asym 3 / 4    : per-token scales + per-token norm + static rotor
//                         table — `has_norm = true`, `has_rotor_table = true`.
//                         Shim reads both `k_norms` and `k_rotor_table`.
//
// Sibling pattern: the existing TurboFlash flash_* fields on `KvCache`
// (`kvcache/update.rs`).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::q8_msl::Q8_GROUP_SIZE;
use crate::turboquant::GROUP_SIZE as TURBO_GROUP_SIZE;
use crate::KvQuant;

/// Per-token payload geometry for a fused-QK codec.
///
/// Tells the shadow how big the head-major slot at one token is, plus
/// whether the codec carries sideband norms and a static rotor table
/// (rotor-asym only). The dispatch layer reads these flags to
/// decide whether to allocate / concat the optional sidebands at call
/// time.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed per-token + sideband contract — all fields are required to describe a codec's payload geometry"
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedQkLayout {
    /// Number of u32 codes stored per token.
    pub(crate) codes_per_token: i32,
    /// Number of f32 scales stored per token (excludes sidebands).
    pub(crate) scales_per_token: i32,
    /// `true` ⇒ allocate the per-token norm sideband (rotor-asym only).
    pub(crate) has_norm: bool,
    /// `true` ⇒ allocate the static per-layer rotor table sideband
    /// (rotor only — both `Sym` and K-only / asym).
    pub(crate) has_rotor_table: bool,
    /// Number of rotor groups (`ceil(head_dim / 3)` for rotor3 / rotor4).
    /// Zero when `has_rotor_table = false`.
    pub(crate) n_groups: i32,
}

impl FusedQkLayout {
    /// Compute the layout for the given codec + `head_dim`.
    ///
    /// Returns `None` when the codec does not have a fused-QK kernel
    /// (i.e. it is not in `FUSED_QK_TABLE`). Errors when `head_dim` does
    /// not satisfy the codec's group-size invariant.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for codecs without a fused-QK kernel; new KvQuant variants are treated as non-fused-QK by default until added to FUSED_QK_TABLE"
    )]
    pub(crate) fn for_codec(codec: KvQuant, head_dim: i32) -> Result<Option<Self>> {
        if head_dim <= 0 {
            return Err(Error::Quant(format!(
                "FusedQkLayout: non-positive head_dim={head_dim}"
            )));
        }
        let hd = head_dim as usize;
        let layout = match codec {
            // q8 (K8V4/K8V8): codes = head_dim/4 u32, scales = head_dim/128 f32.
            // No sidebands.
            KvQuant::K8V4 | KvQuant::K8V8 => {
                if !hd.is_multiple_of(Q8_GROUP_SIZE) {
                    return Err(Error::Quant(format!(
                        "FusedQkLayout(q8): head_dim={head_dim} must be a multiple of \
                         Q8_GROUP_SIZE={Q8_GROUP_SIZE}"
                    )));
                }
                Self {
                    codes_per_token: head_dim / 4,
                    scales_per_token: head_dim / Q8_GROUP_SIZE as i32,
                    has_norm: false,
                    has_rotor_table: false,
                    n_groups: 0,
                }
            }
            // TurboSym3: codes = head_dim*3/32 u32, scales = head_dim/32 f32.
            KvQuant::TurboSym3 => {
                if !hd.is_multiple_of(TURBO_GROUP_SIZE) {
                    return Err(Error::Quant(format!(
                        "FusedQkLayout(turbo3): head_dim={head_dim} must be a multiple of \
                         TURBO_GROUP_SIZE={TURBO_GROUP_SIZE}"
                    )));
                }
                Self {
                    codes_per_token: head_dim * 3 / TURBO_GROUP_SIZE as i32,
                    scales_per_token: head_dim / TURBO_GROUP_SIZE as i32,
                    has_norm: false,
                    has_rotor_table: false,
                    n_groups: 0,
                }
            }
            // TurboSym4: codes = head_dim/8 u32, scales = head_dim/32 f32.
            KvQuant::TurboSym4 => {
                if !hd.is_multiple_of(TURBO_GROUP_SIZE) {
                    return Err(Error::Quant(format!(
                        "FusedQkLayout(turbo4): head_dim={head_dim} must be a multiple of \
                         TURBO_GROUP_SIZE={TURBO_GROUP_SIZE}"
                    )));
                }
                Self {
                    codes_per_token: head_dim / 8,
                    scales_per_token: head_dim / TURBO_GROUP_SIZE as i32,
                    has_norm: false,
                    has_rotor_table: false,
                    n_groups: 0,
                }
            }
            // Rotor-asym 3 / 4: codes = the row's dense code plane, scales =
            // ceil(head_dim/3) f32, 1 norm per token, plus a static
            // `[n_groups * 4]` rotor table broadcast across tokens.
            //
            // The rotor `Sym` / `KOnly` variants are absent on purpose: they
            // keep no bf16 K mirror, so the shadow can never be seeded for
            // them and their decode runs a flash-decode kernel over the packed
            // ring instead. Same for every iso variant.
            KvQuant::RotorK3Asym { .. } => {
                let n_groups_usize = crate::rotorquant::n_groups_for(hd);
                if n_groups_usize == 0 || n_groups_usize > i32::MAX as usize {
                    return Err(Error::Quant(format!(
                        "FusedQkLayout(rotor3): invalid n_groups={n_groups_usize} for head_dim={head_dim}"
                    )));
                }
                let n_groups = n_groups_usize as i32;
                let code_words = crate::rotorquant::row_words_for(hd, 3) as i32;
                Self {
                    codes_per_token: code_words,
                    scales_per_token: n_groups,
                    has_norm: true,
                    has_rotor_table: true,
                    n_groups,
                }
            }
            KvQuant::RotorK4Asym { .. } => {
                let n_groups_usize = crate::rotorquant::n_groups_for(hd);
                if n_groups_usize == 0 || n_groups_usize > i32::MAX as usize {
                    return Err(Error::Quant(format!(
                        "FusedQkLayout(rotor4): invalid n_groups={n_groups_usize} for head_dim={head_dim}"
                    )));
                }
                let n_groups = n_groups_usize as i32;
                let code_words = crate::rotorquant::row_words_for(hd, 4) as i32;
                Self {
                    codes_per_token: code_words,
                    scales_per_token: n_groups,
                    has_norm: true,
                    has_rotor_table: true,
                    n_groups,
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(layout))
    }
}

/// Head-major persistent fused-QK K-side shadow buffer.
///
/// Per-token arrays:
///   * `k_codes`        — `u32 [B, kv_h, max_seq, codes_per_token]`
///   * `k_scales`       — `f32 [B, kv_h, max_seq, scales_per_token]`
///   * `sideband_norms` — `f32 [B, kv_h, max_seq, 1]` (rotor-asym only)
///
/// Static per-layer sideband:
///   * `sideband_rotor_table` — `f32 [n_groups * 4]` (rotor only)
///
/// The kernel shim reads the per-token arrays as flat 1-D views; the
/// per-token row stride at dispatch time equals `max_seq` (slots
/// `[filled..max_seq)` are zero-init and never read because the kernel
/// iterates only `t < kv_seq = filled`).
#[allow(missing_debug_implementations)]
pub(crate) struct FusedQkShadow {
    pub(super) k_codes: Array,
    pub(super) k_scales: Array,
    /// Per-token norm sideband (rotor-asym codecs). `None` for
    /// q8 / turbo where the shim does not consume norms.
    pub(super) sideband_norms: Option<Array>,
    /// Static per-layer rotor table `[n_groups * 4]` f32.
    /// `None` for non-rotor codecs.
    pub(super) sideband_rotor_table: Option<Array>,
    pub(super) max_seq: i32,
    pub(super) filled: i32,
    #[allow(
        dead_code,
        reason = "kept for diagnostics; future codec-aware lifecycle hooks may read it"
    )]
    pub(super) codec: KvQuant,
    pub(super) layout: FusedQkLayout,
}

impl FusedQkShadow {
    /// Resident bytes of the shadow's head-major K buffers and sidebands.
    ///
    /// Counted at full allocation — the shadow is sized to `max_seq` up front.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    pub(crate) fn byte_size(&self) -> u64 {
        let Self {
            k_codes,
            k_scales,
            sideband_norms,
            sideband_rotor_table,
            // Geometry / tags, not allocations.
            max_seq: _,
            filled: _,
            codec: _,
            layout: _,
        } = self;
        crate::bytes::array_bytes(k_codes)
            + crate::bytes::array_bytes(k_scales)
            + crate::bytes::opt_array_bytes(sideband_norms.as_ref())
            + crate::bytes::opt_array_bytes(sideband_rotor_table.as_ref())
    }

    /// Allocate a zero-init shadow for `[B, kv_h, max_seq, *]` plus any
    /// codec-specific sidebands. The rotor table sideband is allocated
    /// **zero-init** here; the dispatch layer seeds it from
    /// `clifford::make_rotor_table(layer_idx, head_idx=0, n_groups)` on
    /// the first encode-chunk call.
    pub(crate) fn allocate(
        codec: KvQuant,
        b: i32,
        kv_h: i32,
        head_dim: i32,
        max_seq: i32,
        device: Device,
    ) -> Result<Self> {
        let layout = FusedQkLayout::for_codec(codec, head_dim)?.ok_or_else(|| {
            Error::Quant(format!(
                "FusedQkShadow::allocate: codec {codec:?} is not in the fused-QK table"
            ))
        })?;
        if b <= 0 || kv_h <= 0 || max_seq <= 0 {
            return Err(Error::Quant(format!(
                "FusedQkShadow::allocate: invariant b={b}, kv_h={kv_h}, max_seq={max_seq} must all be > 0"
            )));
        }
        let codes_shape = [b, kv_h, max_seq, layout.codes_per_token];
        let scales_shape = [b, kv_h, max_seq, layout.scales_per_token];
        let k_codes = zeros(&codes_shape, Dtype::U32, device)?;
        let k_scales = zeros(&scales_shape, Dtype::F32, device)?;
        let sideband_norms = if layout.has_norm {
            Some(zeros(&[b, kv_h, max_seq, 1], Dtype::F32, device)?)
        } else {
            None
        };
        let sideband_rotor_table = if layout.has_rotor_table {
            if layout.n_groups <= 0 {
                return Err(Error::Quant(format!(
                    "FusedQkShadow::allocate: rotor codec has invalid n_groups={}",
                    layout.n_groups
                )));
            }
            // 4 entries per group: [s, b12, b13, b23].
            Some(zeros(&[layout.n_groups * 4], Dtype::F32, device)?)
        } else {
            None
        };
        Ok(Self {
            k_codes,
            k_scales,
            sideband_norms,
            sideband_rotor_table,
            max_seq,
            filled: 0,
            codec,
            layout,
        })
    }

    /// Truncate the shadow's filled count to `n` (no buffer reallocation).
    pub(crate) fn truncate_to(&mut self, n: i32) {
        if self.filled > n {
            self.filled = n.max(0);
        }
    }

    /// Borrow the layout descriptor.
    pub(crate) fn layout(&self) -> &FusedQkLayout {
        &self.layout
    }
}
