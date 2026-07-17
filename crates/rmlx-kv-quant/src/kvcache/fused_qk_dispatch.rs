// Production fused-QK dispatch using the head-major persistent K shadow.
//
// Enables the 6 rotor variants (Rotor3Sym, Rotor4Sym, RotorKOnly3,
// RotorKOnly4, RotorKAsym{3,4}) and the 4 iso variants (Iso3Sym, Iso4Sym,
// IsoKOnly3, IsoKOnly4) on the fused-QK fast path via the per-token +
// sideband-table array split. Iso carries the per-token norm sideband but no
// rotor table — its rotation is one fixed quaternion baked into the kernel
// header.
//
// Wires the fused-QK MSL kernel families (q8 = K8V*, TurboSym3, TurboSym4,
// Rotor3Sym, Rotor4Sym, …) into the production decode path.
//
// The kernel shims share a uniform `FusedQkFn` 13-arg signature; sidebands
// are passed as separate `Option<&Array>` args (pre-fix the dispatch site
// concatenated them into the `k_scales` buffer per decode step, which cost
// ~28 MB of marshaling at Bonsai 8B kv_seq=8192 and swamped the kernel
// compute savings); we mirror their dispatch table here (no `rmlx-models`
// dependency — the codec layer must stay a leaf per the workspace dep
// graph rule). Per-codec K-side encoders that map a chunk of bf16/f32 K
// into the head-major code + per-token scales (+ per-token norm, + static
// rotor table) layout live in this module.
//
// Wire-in: `KvCache::try_fused_qk_dispatch` (called from `sdpa.rs` just
// before the legacy SDPA fallback). The dispatch is decode-only
// (`q_seq == 1`) and gated by `RMLX_FUSED_QK=1`. Returns `None` (fall
// through) on any codec without a populated encoder or when QJL is
// enabled for rotor (the kernel does not consume the QJL residual).

#![allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::items_after_statements
)]
#![allow(
    clippy::wildcard_enum_match_arm,
    reason = "fall-through arms here are the contract: codecs without a fused-QK kernel or GPU encoder MUST return None / fall back. Exhaustive matches would add a new branch for every future KvQuant variant for no semantic gain."
)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{matmul, softmax_precise, Array, Device, Dtype};

use crate::fused_qk_enabled;
use crate::iso_fused_qk_msl::{iso3_fused_qk_sdpa, iso4_fused_qk_sdpa};
use crate::k8vturbo3_append_msl::turbo_quantize_v3_gpu;
use crate::kvcache::fused_qk_shadow::FusedQkShadow;
use crate::kvcache::KvCache;
use crate::q8_fused_qk_msl::q8_fused_qk_sdpa;
use crate::q8_msl::q8_quantize_gpu;
use crate::rotor_fused_qk_msl::{rotor3_fused_qk_sdpa, rotor4_fused_qk_sdpa};
use crate::rotorquant::{n_groups_for, ROTOR3_BITS, ROTOR4_BITS};
use crate::rotorquant_msl::{rotor_quantize_v3_gpu, rotor_quantize_v4_gpu, rotor_table_to_array};
use crate::storage::{iso_n_groups_for, ISO_K3_BITS, ISO_K4_BITS, ISO_QUAT_BLOCK_SIZE};
use crate::turbo_k3_fused_qk_msl::turbo_k3_fused_qk_sdpa;
use crate::turbo_k4_fused_qk_msl::turbo_k4_fused_qk_sdpa;
use crate::turboquant_msl::turbo_quantize_v4_gpu;
use crate::KvQuant;

/// Production minimum kv_seq for fused-QK to be worthwhile. Below this,
/// per-step encode overhead can dominate. Matches the
/// `TURBO_FLASH_MIN_KV_SEQ` baseline. Override via `RMLX_FUSED_QK_MIN` env var.
const DEFAULT_FUSED_QK_MIN_KV_SEQ: i32 = 512;

fn min_kv_seq() -> i32 {
    use std::sync::OnceLock;
    static V: OnceLock<i32> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("RMLX_FUSED_QK_MIN") {
        Ok(s) => match s.parse::<i32>() {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(
                    value = %s,
                    error = %e,
                    default = DEFAULT_FUSED_QK_MIN_KV_SEQ,
                    "RMLX_FUSED_QK_MIN: parse failed, using default"
                );
                DEFAULT_FUSED_QK_MIN_KV_SEQ
            }
        },
        Err(_) => DEFAULT_FUSED_QK_MIN_KV_SEQ,
    })
}

/// Canonical fused-QK kernel pointer type — mirror of
/// `rmlx_models::kv_cache::attention_dispatch::FusedQkFn`.
type FusedQkFn = fn(
    /* query         */ &Array,
    /* k_codes       */ &Array,
    /* k_scales      */ &Array,
    /* k_norms       */ Option<&Array>,
    /* k_rotor_table */ Option<&Array>,
    /* additive_mask */ Option<&Array>,
    /* b             */ i32,
    /* kv_h          */ i32,
    /* kv_seq        */ i32,
    /* head_dim      */ i32,
    /* heads_per_kv  */ i32,
    /* scale         */ f32,
    /* device        */ Device,
) -> Result<Array>;

/// In-crate mirror of `FUSED_QK_TABLE`. The public table in
/// `rmlx-models::kv_cache::attention_dispatch::FUSED_QK_TABLE` lists the
/// 8 `*Sym` entries (K8V4, K8V8, TurboSym3, TurboSym4, Iso3Sym, Iso4Sym,
/// Rotor3Sym, Rotor4Sym). This in-crate mirror is a **superset**: it
/// additionally maps the `*KOnly*` and rotor-asym codecs to the same
/// kernels as their `*Sym` counterparts because the K-side decode is
/// identical (only the V-side codec differs and the V-side is the SDPA
/// caller's responsibility, not the fused-QK kernel's — see
/// `attention_dispatch.rs:186-188`).
const Q8_FN: FusedQkFn = q8_fused_qk_sdpa;
const TURBO_K3_FN: FusedQkFn = turbo_k3_fused_qk_sdpa;
const TURBO_K4_FN: FusedQkFn = turbo_k4_fused_qk_sdpa;
const ISO3_FN: FusedQkFn = iso3_fused_qk_sdpa;
const ISO4_FN: FusedQkFn = iso4_fused_qk_sdpa;
const ROTOR3_FN: FusedQkFn = rotor3_fused_qk_sdpa;
const ROTOR4_FN: FusedQkFn = rotor4_fused_qk_sdpa;

fn lookup_fused_qk_kernel(q: KvQuant) -> Option<FusedQkFn> {
    match q {
        KvQuant::K8V4 | KvQuant::K8V8 => Some(Q8_FN),
        KvQuant::TurboSym3 => Some(TURBO_K3_FN),
        KvQuant::TurboSym4 => Some(TURBO_K4_FN),
        KvQuant::Iso3Sym | KvQuant::IsoKOnly3 => Some(ISO3_FN),
        KvQuant::Iso4Sym | KvQuant::IsoKOnly4 => Some(ISO4_FN),
        KvQuant::Rotor3Sym | KvQuant::RotorKOnly3 | KvQuant::RotorK3Asym { .. } => Some(ROTOR3_FN),
        KvQuant::Rotor4Sym | KvQuant::RotorKOnly4 | KvQuant::RotorK4Asym { .. } => Some(ROTOR4_FN),
        _ => None,
    }
}

/// Returns the global fused-QK dispatch count, summed across all codec
/// families. Tests use this to prove the production decode path actually
/// fired the kernels (vs silently falling back to bf16 SDPA).
pub fn fused_qk_total_dispatch_count() -> u64 {
    crate::q8_fused_qk_msl::q8_fused_qk_dispatch_count()
        + crate::turbo_k3_fused_qk_msl::turbo_k3_fused_qk_dispatch_count()
        + crate::turbo_k4_fused_qk_msl::turbo_k4_fused_qk_dispatch_count()
        + crate::iso_fused_qk_msl::iso3_fused_qk_dispatch_count()
        + crate::iso_fused_qk_msl::iso4_fused_qk_dispatch_count()
        + crate::rotor_fused_qk_msl::rotor3_fused_qk_dispatch_count()
        + crate::rotor_fused_qk_msl::rotor4_fused_qk_dispatch_count()
}

impl KvCache {
    /// Production fused-QK dispatch.
    ///
    /// Returns `Some(out)` when the head-major K shadow path successfully
    /// dispatches a fused-QK kernel and assembles the SDPA output;
    /// returns `None` to fall through to the legacy dequant + SDPA path.
    ///
    /// Gates (in order): env-var (`RMLX_FUSED_QK=1`), GPU device, decode-
    /// only (`q_seq == 1`), codec is in the fused-QK table, `head_dim` is
    /// in the kernel-supported set (128 or 256), `kv_seq` ≥ minimum
    /// threshold, codec has a GPU encoder available, and (for rotor
    /// codecs) the QJL toggle is OFF (the kernel does not consume the
    /// QJL residual — see `rotor_fused_qk_msl.rs`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_fused_qk_dispatch(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        if !fused_qk_enabled() {
            return Ok(None);
        }
        if device != Device::Gpu {
            return Ok(None);
        }
        // Decode-only path.
        let q_shape = queries.shape();
        if q_shape.len() != 4 || q_shape[2] != 1 {
            return Ok(None);
        }
        if new_k.shape().len() != 4 {
            return Ok(None);
        }
        let head_dim = new_k.shape()[3];
        // Kernel shims are hard-gated on head_dim ∈ {128, 256}.
        if head_dim != 128 && head_dim != 256 {
            return Ok(None);
        }
        let new_seq = new_k.shape()[2];
        let kv_seq_after = self.offset + new_seq;
        if kv_seq_after < min_kv_seq() {
            return Ok(None);
        }
        let codec = self.quant;
        let kernel = match lookup_fused_qk_kernel(codec) {
            Some(k) => k,
            None => return Ok(None),
        };
        // GPU encoder coverage gate.
        if !codec_has_gpu_encoder(codec) {
            return Ok(None);
        }
        // Rotor QJL fallback gate. When QJL is enabled the kernel
        // cannot reproduce the K-side residual; fall back to legacy bf16
        // SDPA. This mirrors the K-side encode discipline
        // (`gpu_k_ok = use_gpu && !rotor_qjl_enabled()`).
        if codec_is_rotor(codec) && crate::rotor_qjl::rotor_qjl_enabled() {
            tracing::trace!(
                ?codec,
                "fused_qk: rotor + QJL enabled → fall back to legacy bf16 SDPA"
            );
            return Ok(None);
        }
        // The bf16 mirror is required to seed the shadow on first dispatch.
        if self.decode_fp16_k.is_none() {
            return Ok(None);
        }
        let max_seq = match self.storage_max_seq_for_fused_qk() {
            Some(m) => m,
            None => return Ok(None),
        };

        let prev_offset = self.offset;
        let (b, kv_h) = {
            let s = new_k.shape();
            (s[0], s[1])
        };

        // Overflow gate. When `prev_offset + new_seq`
        // exceeds the shadow / bf16-mirror capacity the per-step slice in
        // `populate_fused_qk_shadow_from_fp16` would silently clip to a
        // 0-length chunk (MLX slice with stop > dim_size). The encode
        // kernel then returns 0-length codes/scales arrays and the
        // subsequent `reshape(&[b, kv_h, n, codes_per_token])` panics with
        // `Cannot reshape array of size 0 into shape (1,kv_h,n,X)`.
        //
        // The legacy `--fused-qk off` path tolerates this overflow because
        // the bf16 mirror's `slice_update` no-ops out-of-range writes and
        // the post-update slice clamps to the buffer size. The shadow
        // populate path has no analogous clamp (it must round-trip through
        // a per-step GPU encode and a head-major slice_update), so we
        // simply fall through to the legacy SDPA path here. Without this
        // gate Bonsai 8B `--ctk k_rotor3 --ctv rotor_v_3 --fused-qk on`
        // crashes at decode step `max_seq - prefill_len`.
        if prev_offset + new_seq > max_seq {
            warn_kv_overflow_once(codec, prev_offset, new_seq, max_seq);
            return Ok(None);
        }

        // MEDIUM-3 — allocate the shadow + seed BEFORE bumping offset /
        // mirror. If alloc or seed fails, `?` returns Err and the cache is
        // unchanged — no torn state.
        if self.fused_qk_shadow.is_none() {
            self.fused_qk_shadow = Some(FusedQkShadow::allocate(
                codec, b, kv_h, head_dim, max_seq, device,
            )?);
            // Seed the rotor table sideband (if applicable) before any
            // encode-chunk runs.
            if codec_is_rotor(codec) {
                // `seed_rotor_table` hardcodes `head_idx = 0` in its call
                // to `make_rotor_table`. Assert the active storage's
                // `head_idx` agrees so future multi-head-KV work cannot
                // silently desync the shadow's table from the CPU-storage
                // path's table.
                debug_assert_eq!(
                    self.active_storage_rotor_head_idx().unwrap_or(0),
                    0,
                    "seed_rotor_table assumes head_idx=0; multi-head KV needs threading head_idx through FusedQkShadow"
                );
                seed_rotor_table(
                    self.fused_qk_shadow.as_mut().ok_or_else(|| {
                        Error::Mlx("fused_qk: shadow vanished post-allocate".into())
                    })?,
                    head_dim as usize,
                    self.layer_idx,
                )?;
            }
            // Seed the head-major shadow from the bf16 prefill prefix
            // (range `[0, prev_offset)`).
            if prev_offset > 0 {
                tracing::debug!(
                    prefix = prev_offset,
                    "fused_qk: seeding shadow from bf16 prefix"
                );
                self.populate_fused_qk_shadow_from_fp16(0, prev_offset, device)?;
            }
        }

        // Bump offset + mirror the new K/V into the bf16 buffer.
        self.offset = prev_offset + new_seq;
        self.update_decode_fp16(new_k, new_v, max_seq, device)?;

        // Append the new chunk (range `[prev_offset, prev_offset+new_seq)`).
        self.populate_fused_qk_shadow_from_fp16(prev_offset, new_seq, device)?;

        // ── Run the kernel ──
        let kv_seq = self.offset; // == prev_offset + new_seq
        let shadow = if let Some(s) = self.fused_qk_shadow.as_ref() {
            s
        } else {
            tracing::error!(
                "fused_qk: shadow vanished between alloc and dispatch — internal invariant violated"
            );
            return Err(Error::Mlx(
                "fused_qk: shadow vanished between alloc and dispatch".into(),
            ));
        };
        let layout = *shadow.layout();
        // Slice the head-major shadow to the kernel's expected `kv_seq`
        // tile. The kernel reads these as flat 1-D buffers.
        let codes_view = slice_shadow_to_kv_seq(
            &shadow.k_codes,
            b,
            kv_h,
            kv_seq,
            layout.codes_per_token,
            device,
        )?;
        let scales_flat = slice_shadow_to_kv_seq(
            &shadow.k_scales,
            b,
            kv_h,
            kv_seq,
            layout.scales_per_token,
            device,
        )?;
        // Pass scales / norms / rotor_table as separate Array arguments to the
        // kernel shim. The pre-fix dispatch site concatenated
        // `[scales | norms | rotor_table]` into one Array on every decode step;
        // at Bonsai head_dim=128, n_groups=32, layers=26, kv_seq=8192 that was
        // ~28 MB of marshaling per step and swamped the kernel compute savings
        // (12.3 TPS vs 63.5 TPS legacy). The `FusedQkFn` signature takes
        // `k_norms` / `k_rotor_table` as `Option<&Array>` directly; the shim
        // does the per-codec wiring.
        let k_norms_arg = if layout.has_norm {
            if let Some(norms_buf) = shadow.sideband_norms.as_ref() {
                Some(slice_shadow_to_kv_seq(
                    norms_buf, b, kv_h, kv_seq, 1, device,
                )?)
            } else {
                // Internal-invariant violation (shadow construction bug,
                // not a runtime condition). Log loudly so the failure is
                // not lost; still propagate Err for crash-recovery parity
                // with the "shadow vanished" branch above.
                tracing::error!(
                    ?codec,
                    "fused_qk: layout requires norms sideband but shadow is missing it — internal invariant violated"
                );
                return Err(Error::Mlx(
                    "fused_qk: layout requires norms sideband but shadow is missing it".into(),
                ));
            }
        } else {
            None
        };
        let k_rotor_arg = if layout.has_rotor_table {
            if let Some(rt) = shadow.sideband_rotor_table.as_ref() {
                Some(rt.try_clone()?)
            } else {
                // Same rationale as the k_norms_arg branch above:
                // internal invariant, log + Err.
                tracing::error!(
                    ?codec,
                    "fused_qk: rotor codec requires rotor_table sideband but shadow is missing it — internal invariant violated"
                );
                return Err(Error::Mlx(
                    "fused_qk: rotor codec requires rotor_table sideband but shadow is missing it"
                        .into(),
                ));
            }
        } else {
            None
        };

        let n_q_heads = q_shape[1];
        if n_q_heads <= 0 || kv_h <= 0 || n_q_heads % kv_h != 0 {
            return Err(Error::Mlx(format!(
                "fused_qk: n_q_heads={n_q_heads} not divisible by kv_h={kv_h}"
            )));
        }
        let heads_per_kv = n_q_heads / kv_h;

        tracing::debug!(
            codec = ?codec,
            kv_seq,
            head_dim,
            n_q_heads,
            kv_h,
            "fused_qk: dispatching kernel"
        );

        let scores = kernel(
            queries,
            &codes_view,
            &scales_flat,
            k_norms_arg.as_ref(),
            k_rotor_arg.as_ref(),
            additive_mask,
            b,
            kv_h,
            kv_seq,
            head_dim,
            heads_per_kv,
            scale,
            device,
        )?;

        // Softmax + GQA-SV: identical post-fused-QK shape to PlanarK.
        let probs = softmax_precise(&scores, -1, device)?;
        let v_full = if let Some(v) = self.decode_fp16_v.as_ref() {
            v.try_clone()?
        } else {
            tracing::error!(
                "fused_qk: decode_fp16_v missing post-update — internal invariant violated"
            );
            return Err(Error::Mlx(
                "fused_qk: decode_fp16_v missing post-update".into(),
            ));
        };
        let v_sliced = v_full.slice(
            &[0_i32, 0, 0, 0],
            &[b, kv_h, kv_seq, head_dim],
            &[1_i32; 4],
            device,
        )?;
        let probs_g = probs.reshape(&[b, kv_h, heads_per_kv, 1, kv_seq], device)?;
        let v_g = v_sliced.reshape(&[b, kv_h, 1, kv_seq, head_dim], device)?;
        let out_g = matmul(&probs_g, &v_g, device)?;
        let out = out_g.reshape(&[b, n_q_heads, 1, head_dim], device)?;
        let out = if out.dtype() == queries.dtype() {
            out
        } else {
            out.astype(queries.dtype(), device)?
        };
        if let Some(ref mut shadow) = self.fused_qk_shadow {
            shadow.filled = kv_seq;
        }
        Ok(Some(out))
    }

    /// Return the rotor K-storage `head_idx` for the active storage variant,
    /// or `None` when the active codec is not rotor / has no allocated K
    /// storage. Used by the dispatch path's `debug_assert!` to confirm
    /// `seed_rotor_table`'s `head_idx=0` hardcode matches reality.
    fn active_storage_rotor_head_idx(&self) -> Option<u32> {
        use crate::storage::KvStorage;
        match &self.storage {
            KvStorage::RotorSym3 { k: Some(k), .. } => Some(k.head_idx),
            KvStorage::RotorSym4 { k: Some(k), .. } => Some(k.head_idx),
            KvStorage::RotorKOnly3 { k: Some(k), .. } => Some(k.head_idx),
            KvStorage::RotorKOnly4 { k: Some(k), .. } => Some(k.head_idx),
            KvStorage::RotorKAsym3 { k: Some(k), .. } => Some(k.head_idx),
            KvStorage::RotorKAsym4 { k: Some(k), .. } => Some(k.head_idx),
            _ => None,
        }
    }

    /// Look up the `max_seq` the shadow should be sized to, taken from the
    /// active storage variant.
    fn storage_max_seq_for_fused_qk(&self) -> Option<i32> {
        use crate::storage::KvStorage;
        let m = match &self.storage {
            KvStorage::K8V4 { max_seq, .. } => *max_seq,
            KvStorage::K8V8 { max_seq, .. } => *max_seq,
            KvStorage::TurboSym3 { max_seq, .. } => *max_seq,
            KvStorage::TurboSym4 { max_seq, .. } => *max_seq,
            // Rotor storage variants (Sym + K-only + Asym, both 3-bit and 4-bit).
            KvStorage::RotorSym3 { max_seq, .. } => *max_seq,
            KvStorage::RotorSym4 { max_seq, .. } => *max_seq,
            KvStorage::RotorKOnly3 { max_seq, .. } => *max_seq,
            KvStorage::RotorKOnly4 { max_seq, .. } => *max_seq,
            KvStorage::RotorKAsym3 { max_seq, .. } => *max_seq,
            KvStorage::RotorKAsym4 { max_seq, .. } => *max_seq,
            _ => return None,
        };
        if m <= 0 {
            None
        } else {
            Some(m)
        }
    }

    /// Quantise `[B, kv_h, n, D]` from `decode_fp16_k` starting at token
    /// `start` and `slice_update` it head-major into the fused-QK shadow
    /// at `[:, :, start:start+n, :]`. Used for both the prefill seed
    /// (`n = prev_offset`) and the per-decode-token append (`n = 1`).
    fn populate_fused_qk_shadow_from_fp16(
        &mut self,
        start: i32,
        n: i32,
        device: Device,
    ) -> Result<()> {
        if n <= 0 {
            return Ok(());
        }
        let fp16_k = match self.decode_fp16_k.as_ref() {
            Some(a) => a,
            None => {
                return Err(Error::Mlx(
                    "fused_qk populate: decode_fp16_k missing (BUG: caller must guard)".into(),
                ))
            }
        };
        let k_shape: Vec<i32> = fp16_k.shape();
        if k_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "fused_qk populate: decode_fp16_k rank != 4, got {k_shape:?}"
            )));
        }
        let b = k_shape[0];
        let kv_h = k_shape[1];
        let d = k_shape[3];

        // Slice [B, kv_h, n, D] from decode_fp16 starting at `start`.
        let sl_start = [0_i32, 0, start, 0];
        let sl_stop = [b, kv_h, start + n, d];
        let sl_strides = [1_i32; 4];
        let k_chunk = fp16_k.slice(&sl_start, &sl_stop, &sl_strides, device)?;
        let k_f32 = if k_chunk.dtype() == Dtype::F32 {
            k_chunk
        } else {
            k_chunk.astype(Dtype::F32, device)?
        };

        let codec = self.quant;
        let shadow_layout = {
            let s = self.fused_qk_shadow.as_ref().ok_or_else(|| {
                Error::Mlx("fused_qk populate: shadow missing (caller alloc'd first)".into())
            })?;
            *s.layout()
        };

        // For rotor codecs we need the rotor table Array to drive the
        // encoder; pull it from the shadow's sideband (seeded by the
        // dispatch path on first allocate).
        let rotor_table_for_encode = if codec_is_rotor(codec) {
            let s = self.fused_qk_shadow.as_ref().ok_or_else(|| {
                Error::Mlx("fused_qk populate: shadow missing while reading rotor table".into())
            })?;
            Some(
                s.sideband_rotor_table
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Mlx(
                            "fused_qk populate: rotor codec but shadow rotor table not allocated"
                                .into(),
                        )
                    })?
                    .try_clone()?,
            )
        } else {
            None
        };

        let encoded = encode_chunk_to_head_major(
            codec,
            &k_f32,
            b,
            kv_h,
            n,
            d,
            shadow_layout,
            rotor_table_for_encode.as_ref(),
            device,
        )?;

        // 4-D slice_update bounds.
        let max_seq;
        let codes_stop;
        let scales_stop;
        let norms_stop;
        {
            let s = self.fused_qk_shadow.as_ref().ok_or_else(|| {
                Error::Mlx("fused_qk populate: shadow vanished mid-update".into())
            })?;
            max_seq = s.max_seq;
            codes_stop = [b, kv_h, start + n, shadow_layout.codes_per_token];
            scales_stop = [b, kv_h, start + n, shadow_layout.scales_per_token];
            norms_stop = [b, kv_h, start + n, 1];
        }
        if start + n > max_seq {
            return Err(Error::Mlx(format!(
                "fused_qk populate: start+n={} exceeds shadow max_seq={}",
                start + n,
                max_seq
            )));
        }

        // Non-destructive on partial failure. Build every updated Array in
        // a local first; only mutate `self.fused_qk_shadow` after all
        // `slice_update`s succeed. If any `?` returns Err the locals drop
        // and `self.fused_qk_shadow` is never touched, preserving the
        // "no torn state" invariant the MEDIUM-3 comment at the dispatch
        // site promises.
        let shadow_ref = self.fused_qk_shadow.as_ref().ok_or_else(|| {
            Error::Mlx("fused_qk populate: shadow vanished while reading buffers".into())
        })?;
        let codes_new = shadow_ref.k_codes.slice_update(
            &encoded.codes,
            &sl_start,
            &codes_stop,
            &sl_strides,
            device,
        )?;
        let scales_new = shadow_ref.k_scales.slice_update(
            &encoded.scales,
            &sl_start,
            &scales_stop,
            &sl_strides,
            device,
        )?;
        let norms_new = if let (Some(norms_buf), Some(encoded_norms)) =
            (shadow_ref.sideband_norms.as_ref(), encoded.norms)
        {
            Some(norms_buf.slice_update(
                &encoded_norms,
                &sl_start,
                &norms_stop,
                &sl_strides,
                device,
            )?)
        } else {
            None
        };
        // All updates succeeded — commit.
        let shadow = self
            .fused_qk_shadow
            .as_mut()
            .ok_or_else(|| Error::Mlx("fused_qk populate: shadow vanished pre-commit".into()))?;
        shadow.k_codes = codes_new;
        shadow.k_scales = scales_new;
        if let Some(updated_norms) = norms_new {
            shadow.sideband_norms = Some(updated_norms);
        }
        if start + n > shadow.filled {
            shadow.filled = start + n;
        }
        Ok(())
    }
}

/// Slice `[B, kv_h, max_seq, payload]` to a flat view of the first
/// `kv_seq` rows: `[B * kv_h * kv_seq * payload]`. The dim-2 slice is
/// non-contiguous; the trailing reshape forces a per-step materialisation
/// — see the caller note for the cost framing.
fn slice_shadow_to_kv_seq(
    buf: &Array,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    payload: i32,
    device: Device,
) -> Result<Array> {
    let max_seq = buf.shape()[2];
    if kv_seq > max_seq {
        return Err(Error::Mlx(format!(
            "fused_qk slice: kv_seq={kv_seq} exceeds shadow max_seq={max_seq}"
        )));
    }
    let sliced = buf.slice(
        &[0_i32, 0, 0, 0],
        &[b, kv_h, kv_seq, payload],
        &[1_i32; 4],
        device,
    )?;
    let total: i32 = sliced.shape().iter().product();
    sliced.reshape(&[total], device)
}

/// Emit a one-shot `tracing::warn!` when the fused-QK dispatch falls through
/// on KV overflow. The overflow is a recoverable degradation (the legacy bf16
/// SDPA path takes over). Wrapped in `OnceLock` to avoid log-spam from
/// every subsequent decode step in the same run (matches the
/// `warn_iso_hold_once` pattern in `fused_qk_shadow.rs`).
fn warn_kv_overflow_once(codec: KvQuant, prev_offset: i32, new_seq: i32, max_seq: i32) {
    use std::sync::OnceLock;
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            ?codec,
            prev_offset,
            new_seq,
            max_seq,
            "fused_qk: kv overflow (prev_offset + new_seq > max_seq) — falling back to legacy SDPA; \
             this run will not exercise the fused-QK fast path beyond this point"
        );
    });
}

/// Predicate for any rotor-family codec.
fn codec_is_rotor(codec: KvQuant) -> bool {
    matches!(
        codec,
        KvQuant::Rotor3Sym
            | KvQuant::Rotor4Sym
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. }
    )
}

/// Whether the codec has a GPU encoder wired in for the fused-QK shadow path.
///
/// Every codec listed here must have a matching arm in
/// [`encode_chunk_to_head_major`] — the two lists are the same fact stated
/// twice, and a codec in one but not the other either errors at encode or
/// silently sits on the legacy path.
fn codec_has_gpu_encoder(codec: KvQuant) -> bool {
    matches!(
        codec,
        KvQuant::K8V4
            | KvQuant::K8V8
            | KvQuant::TurboSym3
            | KvQuant::TurboSym4
            | KvQuant::Iso3Sym
            | KvQuant::Iso4Sym
            | KvQuant::IsoKOnly3
            | KvQuant::IsoKOnly4
            | KvQuant::Rotor3Sym
            | KvQuant::Rotor4Sym
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. }
    )
}

/// Seed the static rotor-table sideband on a freshly-allocated shadow.
/// Mirrors `ensure_k{3,4}_rotors` in `update.rs` — same input
/// `(layer_idx, head_idx=0, n_groups)` so the kernel sees the identical
/// rotor table the CPU storage path would have produced.
///
/// # head_idx coupling
///
/// The rotor table is built with `head_idx = 0` (constant). Production
/// rotor storage variants (`QuantRotorK3` / `QuantRotorK4`) default
/// `head_idx: 0` and rMLX currently has no multi-head-KV path that bumps
/// it. The `debug_assert!` in the dispatch caller enforces this on debug
/// builds; if multi-head KV is ever wired the assertion fires loudly and
/// this function must be re-shaped to thread `head_idx` from the storage
/// variant through `FusedQkShadow`.
///
/// # GPU residency
///
/// `rotor_table_to_array` produces a CPU-resident Array (built via
/// `Array::from_bytes`). The kernel shim and the per-step encoder
/// (`encode_chunk_rotor`) both consume it on the GPU stream; without an
/// explicit cast MLX marshals it CPU→GPU on every call. We pin it onto
/// the GPU once here with an identity `astype(F32, Gpu)`, evaluated
/// eagerly so the marshaling cost is paid at seed-time, not per decode
/// step.
fn seed_rotor_table(shadow: &mut FusedQkShadow, head_dim: usize, layer_idx: usize) -> Result<()> {
    let layout = *shadow.layout();
    if !layout.has_rotor_table {
        return Ok(());
    }
    if layout.n_groups <= 0 {
        return Err(Error::Mlx(format!(
            "fused_qk seed_rotor_table: invalid n_groups={}",
            layout.n_groups
        )));
    }
    let n_groups = n_groups_for(head_dim);
    if n_groups != layout.n_groups as usize {
        return Err(Error::Mlx(format!(
            "fused_qk seed_rotor_table: layout n_groups={} != head_dim-derived {n_groups}",
            layout.n_groups
        )));
    }
    let layer_u32 = u32::try_from(layer_idx).map_err(|_| {
        Error::Mlx(format!(
            "fused_qk seed_rotor_table: layer_idx={layer_idx} out of u32 range"
        ))
    })?;
    let table = crate::clifford::make_rotor_table(layer_u32, 0, n_groups);
    let arr_cpu = rotor_table_to_array(&table)?;
    // LOW-1: pin onto the GPU. `astype(F32, Gpu)` on an F32 array is an
    // identity cast that schedules on the GPU stream — the result is a
    // GPU-resident Array. We do NOT eagerly `eval()` here: the first
    // kernel dispatch will fold the cast into its own batch, and an
    // eager eval at seed-time forces a sync that splits the rotor-table
    // marshal off the encode-chunk graph. The kernel + encoder both
    // consume `sideband_rotor_table` on the GPU stream, so the cast's
    // residency is honored without extra blocking.
    let arr_gpu = arr_cpu.astype(Dtype::F32, Device::Gpu)?;
    tracing::debug!(
        layer_idx,
        n_groups,
        "fused_qk: rotor_table seeded onto GPU stream"
    );
    shadow.sideband_rotor_table = Some(arr_gpu);
    Ok(())
}

/// One chunk encoded into the codec's per-token layout.
struct ChunkEncoded {
    /// `u32 [B, kv_h, n, codes_per_token]`.
    codes: Array,
    /// `f32 [B, kv_h, n, scales_per_token]`.
    scales: Array,
    /// `f32 [B, kv_h, n, 1]` for iso/rotor; `None` for q8/turbo.
    norms: Option<Array>,
}

/// Encode a single chunk of head-major `[B, kv_h, n, D]` f32 K into the
/// codec-specific per-token slabs that the kernel shim expects.
fn encode_chunk_to_head_major(
    codec: KvQuant,
    k_f32: &Array,
    b: i32,
    kv_h: i32,
    n: i32,
    d: i32,
    layout: crate::kvcache::fused_qk_shadow::FusedQkLayout,
    rotor_table: Option<&Array>,
    device: Device,
) -> Result<ChunkEncoded> {
    use crate::q8_msl::Q8_GROUP_SIZE;
    use crate::turboquant::GROUP_SIZE as TURBO_GROUP;
    match codec {
        KvQuant::K8V4 | KvQuant::K8V8 => {
            let (codes, scales) = q8_quantize_gpu(k_f32, device)?;
            let codes_4d = codes.reshape(&[b, kv_h, n, d / 4], device)?;
            let scales_4d = scales.reshape(&[b, kv_h, n, d / Q8_GROUP_SIZE as i32], device)?;
            assert_layout(layout, d / 4, d / Q8_GROUP_SIZE as i32)?;
            Ok(ChunkEncoded {
                codes: codes_4d,
                scales: scales_4d,
                norms: None,
            })
        }
        KvQuant::TurboSym3 => {
            let (codes, scales) = turbo_quantize_v3_gpu(k_f32, device)?;
            let codes_per_tok = d * 3 / TURBO_GROUP as i32;
            let scales_per_tok = d / TURBO_GROUP as i32;
            let codes_4d = codes.reshape(&[b, kv_h, n, codes_per_tok], device)?;
            let scales_4d = scales.reshape(&[b, kv_h, n, scales_per_tok], device)?;
            assert_layout(layout, codes_per_tok, scales_per_tok)?;
            Ok(ChunkEncoded {
                codes: codes_4d,
                scales: scales_4d,
                norms: None,
            })
        }
        KvQuant::TurboSym4 => {
            let (codes, scales) = turbo_quantize_v4_gpu(k_f32, device)?;
            let codes_per_tok = d / 8;
            let scales_per_tok = d / TURBO_GROUP as i32;
            let codes_4d = codes.reshape(&[b, kv_h, n, codes_per_tok], device)?;
            let scales_4d = scales.reshape(&[b, kv_h, n, scales_per_tok], device)?;
            assert_layout(layout, codes_per_tok, scales_per_tok)?;
            Ok(ChunkEncoded {
                codes: codes_4d,
                scales: scales_4d,
                norms: None,
            })
        }
        KvQuant::Iso3Sym | KvQuant::IsoKOnly3 => {
            encode_chunk_iso(k_f32, b, kv_h, n, d, layout, ISO_K3_BITS, device)
        }
        KvQuant::Iso4Sym | KvQuant::IsoKOnly4 => {
            encode_chunk_iso(k_f32, b, kv_h, n, d, layout, ISO_K4_BITS, device)
        }
        KvQuant::Rotor3Sym | KvQuant::RotorKOnly3 | KvQuant::RotorK3Asym { .. } => {
            encode_chunk_rotor(
                k_f32,
                rotor_table,
                b,
                kv_h,
                n,
                d,
                layout,
                ROTOR3_BITS,
                device,
            )
        }
        KvQuant::Rotor4Sym | KvQuant::RotorKOnly4 | KvQuant::RotorK4Asym { .. } => {
            encode_chunk_rotor(
                k_f32,
                rotor_table,
                b,
                kv_h,
                n,
                d,
                layout,
                ROTOR4_BITS,
                device,
            )
        }
        _ => Err(Error::Mlx(format!(
            "fused_qk encode: codec {codec:?} HOLD — GPU encoder not yet wired"
        ))),
    }
}

/// Iso chunk encode. Calls the K-side GPU quantize kernel and reshapes the flat
/// `[n_tokens * n_groups]` outputs into the per-token head-major slabs the
/// shadow stores. The per-token norm is extracted as the first group's norm
/// slot, the same convention [`encode_chunk_rotor`] uses.
///
/// No rotor-table sideband: iso rotates every group by the same fixed
/// golden-ratio quaternion, which the kernel header bakes in.
///
/// `bits` is selected explicitly by the caller; there is no default arm, so a
/// width with no kernel errors rather than encoding through the other's.
#[allow(clippy::too_many_arguments)]
fn encode_chunk_iso(
    k_f32: &Array,
    b: i32,
    kv_h: i32,
    n: i32,
    d: i32,
    layout: crate::kvcache::fused_qk_shadow::FusedQkLayout,
    bits: u8,
    device: Device,
) -> Result<ChunkEncoded> {
    let d_usize = usize::try_from(d)
        .map_err(|_| Error::Mlx(format!("fused_qk iso encode: negative head_dim={d}")))?;
    if !d_usize.is_multiple_of(ISO_QUAT_BLOCK_SIZE) {
        return Err(Error::Mlx(format!(
            "fused_qk iso encode: head_dim={d} must be a multiple of the quaternion block \
             size {ISO_QUAT_BLOCK_SIZE}"
        )));
    }
    let n_groups = iso_n_groups_for(d_usize);
    let n_groups_i32 = i32::try_from(n_groups).map_err(|_| {
        Error::Mlx(format!(
            "fused_qk iso encode: n_groups {n_groups} overflows"
        ))
    })?;

    let (codes_flat, scales_flat, _quats, norms_flat) = match bits {
        ISO_K3_BITS => crate::isoquant_msl::iso_quantize_v3_gpu(k_f32, d_usize, device)?,
        ISO_K4_BITS => crate::isoquant_msl_v4::iso_quantize_v4_gpu(k_f32, d_usize, device)?,
        other => {
            return Err(Error::Mlx(format!(
                "fused_qk iso encode: unsupported bits={other} (only 3 and 4); refusing to \
                 encode with another width's kernel"
            )))
        }
    };

    // codes_flat / scales_flat shape: [n_tokens * n_groups]. Reshape to
    // head-major — one u32 word per quaternion block at both bit widths.
    let codes_4d = codes_flat.reshape(&[b, kv_h, n, n_groups_i32], device)?;
    let scales_4d = scales_flat.reshape(&[b, kv_h, n, n_groups_i32], device)?;
    // norms_flat is per-group; deduplicate to per-token by slicing the first
    // slot of each token's group-tuple.
    let norms_per_group = norms_flat.reshape(&[b, kv_h, n, n_groups_i32], device)?;
    let norms_4d =
        norms_per_group.slice(&[0_i32, 0, 0, 0], &[b, kv_h, n, 1], &[1_i32; 4], device)?;
    assert_layout(layout, n_groups_i32, n_groups_i32)?;
    Ok(ChunkEncoded {
        codes: codes_4d,
        scales: scales_4d,
        norms: Some(norms_4d),
    })
}

/// Rotor chunk encode. Calls the K-side GPU quantize kernel and reshapes
/// the flat `[n_tokens * n_groups]` outputs into the per-token head-major
/// slabs the shadow stores. The per-token norm is extracted as the first
/// group's norm slot (`rotor_gpu_outputs_to_cpu` convention).
#[allow(clippy::too_many_arguments)]
fn encode_chunk_rotor(
    k_f32: &Array,
    rotor_table: Option<&Array>,
    b: i32,
    kv_h: i32,
    n: i32,
    d: i32,
    layout: crate::kvcache::fused_qk_shadow::FusedQkLayout,
    bits: u8,
    device: Device,
) -> Result<ChunkEncoded> {
    let rotors = rotor_table
        .ok_or_else(|| Error::Mlx("fused_qk rotor encode: rotor_table sideband missing".into()))?;
    let n_groups = n_groups_for(d as usize);
    if n_groups != layout.n_groups as usize {
        return Err(Error::Mlx(format!(
            "fused_qk rotor encode: layout n_groups={} != head_dim-derived {n_groups}",
            layout.n_groups
        )));
    }
    let (codes_flat, scales_flat, norms_flat) = if bits == ROTOR3_BITS {
        rotor_quantize_v3_gpu(k_f32, rotors, d as usize, device)?
    } else {
        rotor_quantize_v4_gpu(k_f32, rotors, d as usize, device)?
    };
    // codes_flat shape: [n_tokens * n_groups]. Reshape to head-major.
    let n_groups_i32 = n_groups as i32;
    let codes_4d = codes_flat.reshape(&[b, kv_h, n, n_groups_i32], device)?;
    let scales_4d = scales_flat.reshape(&[b, kv_h, n, n_groups_i32], device)?;
    // norms_flat is per-group; deduplicate to per-token by slicing the
    // first slot of each token's group-tuple. Reshape to
    // `[B, kv_h, n, n_groups]` then slice the last dim to `0..1`.
    let norms_per_group = norms_flat.reshape(&[b, kv_h, n, n_groups_i32], device)?;
    let norms_4d =
        norms_per_group.slice(&[0_i32, 0, 0, 0], &[b, kv_h, n, 1], &[1_i32; 4], device)?;
    assert_layout(layout, n_groups_i32, n_groups_i32)?;
    Ok(ChunkEncoded {
        codes: codes_4d,
        scales: scales_4d,
        norms: Some(norms_4d),
    })
}

fn assert_layout(
    layout: crate::kvcache::fused_qk_shadow::FusedQkLayout,
    expected_codes: i32,
    expected_scales: i32,
) -> Result<()> {
    if layout.codes_per_token != expected_codes {
        return Err(Error::Mlx(format!(
            "fused_qk encode: layout codes_per_token={} != expected {}",
            layout.codes_per_token, expected_codes
        )));
    }
    if layout.scales_per_token != expected_scales {
        return Err(Error::Mlx(format!(
            "fused_qk encode: layout scales_per_token={} != expected {}",
            layout.scales_per_token, expected_scales
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "fused_qk_dispatch_tests.rs"]
mod fused_qk_dispatch_tests;
