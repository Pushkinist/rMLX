//! TurboQuant KV update path.
//!
//! Holds every update-side body that only the TurboQuant storage types use:
//! the per-variant `update_k8vturbo*` and `update_tsym` decode entries, the
//! width-parametric symmetric body they enter, and the `exit_prefill_*`
//! prefill bulk-encode bodies. The `KvStorage` dispatch and the helpers with
//! more than one family caller stay in [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::{
    KvStorage, QuantK, QuantKTurbo, QuantKTurbo3, QuantKTurbo4, QuantV, TURBO_K4_BITS,
};

use super::helpers::{array_to_f32_vec, arrays_to_f32, f32_vec_to_array, storage_variant_name};
use super::update::storage_mismatch;
use super::KvCache;

/// Decode update for `TurboSym3` / `TurboSym4`: K is [`QuantKTurbo`] at
/// `BITS`, V is [`QuantV`] at the same width.
///
/// # The V-axis device rule
///
/// The K axis always takes the caller's device. The V axis takes it at 4 bits
/// and is pinned to `Device::Cpu` at 3, and that is load-bearing rather than a
/// tuning choice: [`QuantV::append`] enters its GPU branch on the device alone
/// and then refuses `bits != 4`, so handing a 3-bit V store the caller's
/// device returns `Error::Quant` on every GPU append. The 3-bit V GPU dispatch
/// also failed the −2% TPS gate on `K8VTurbo3`. The host vector `v_f32` is
/// materialised exactly when the resolved V device is the CPU, and the GPU
/// `Array` the dequant returns is taken when it returns one — rebuilding those
/// rows from the host vector instead costs one device-to-host copy per step.
///
/// `variant` is the storage spelling the caller resolved, used only in the
/// "buffer absent after init" diagnostics.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
pub(super) fn tsym_update<const BITS: u8>(
    k: &mut Option<QuantKTurbo<BITS>>,
    v: &mut Option<QuantV>,
    max_seq: i32,
    variant: &'static str,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();
    let v_device = if BITS == TURBO_K4_BITS {
        device
    } else {
        Device::Cpu
    };

    let k_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, device)?
    };
    let v_f32 = if v_device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_v, Device::Cpu)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantKTurbo::<BITS>::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
    let k_shape = ks.shape.clone();
    let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
    let k_full = match k_arr_opt {
        Some(arr) => arr,
        None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
    };

    if v.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *v = Some(QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: BITS,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        });
    }
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx(format!("{variant} V buffer absent after init")));
    };
    vs.append(&v_f32, &new_shape, new_v, v_device, max_seq)?;
    let v_shape = vs.shape.clone();
    let (v_recon_f32, v_arr_opt) = vs.dequantize_choice(v_device, new_v.dtype())?;
    let v_full = match v_arr_opt {
        Some(arr) => arr,
        None => f32_vec_to_array(&v_recon_f32, &v_shape)?,
    };

    Ok((k_full, v_full))
}

impl KvCache {
    /// K8VTurbo3 decode update — K = affine q8_0, V = TurboQuant 3-bit
    /// Lloyd-Max (codebook from `turboquant::lloyd_gaussian_codebook(3)`).
    ///
    /// Mirrors `update_k8v4` but with `bits=3` on the V side. added a
    /// Metal 3-bit kernel (`k8vturbo3_append_msl`) but bench showed it
    /// regresses Gemma4-e4b by ~3.5% and Gemma4-26b by ~6.9% vs the
    /// `Mixed{v_bits:3}` affine baseline — both fail the −2% TPS
    /// gate. The GPU dispatch wiring was therefore reverted; the V side
    /// stays on CPU here, exactly as in . The MSL kernel source is
    /// retained in `k8vturbo3_append_msl.rs` as a future-reference hook
    /// (still unit-tested for bit-equivalence) — see
    /// `docs/research/turboquant_v3_vs_affine_v3.md` "Second pass" for the
    /// bench numbers.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn update_k8vturbo3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4).
        // V-side: force CPU for 3-bit (: GPU dispatch failed the −2% gate;
        // dispatch was reverted, kernel source kept as future-reference hook).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantK {
                codes: Vec::new(),
                scales: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                max_seq,
            });
        }
        let ks = k.as_mut().unwrap();
        ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
        let k_shape = ks.shape.clone();
        let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
        let k_full = match k_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
        };

        if v.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 3,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
        }
        let vs = v.as_mut().unwrap();
        // V-side: force CPU path for 3-bit (GPU kernel wired but disabled;
        // see `update_k8vturbo3` doc-comment + research doc for the −2% gate fail).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }
    /// Symmetric TurboQuant decode update, both code widths.
    ///
    /// K is [`QuantKTurbo`] at the width the active storage variant carries
    /// (`TurboSym3` is 3 bits, `TurboSym4` is 4); V is [`QuantV`] at the same
    /// width. The body is [`tsym_update`] — see it for the V-axis device rule,
    /// which is the one thing the two widths do not share.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the dispatch routes only the two symmetric turbo variants here; a future variant belongs in its own entry, and the fall-through names the mismatch rather than adding a branch per KvStorage variant"
    )]
    pub(super) fn update_tsym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let (KvStorage::TurboSym3 { max_seq, .. } | KvStorage::TurboSym4 { max_seq, .. }) =
            &self.storage
        else {
            return Err(storage_mismatch("TurboSym3 | TurboSym4", &self.storage));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        match &mut self.storage {
            KvStorage::TurboSym3 { k, v, .. } => {
                tracing::trace!(quant = "tsym3", "update_tsym: decode step");
                tsym_update::<3>(k, v, max_seq, "TurboSym3", new_k, new_v, device)
            }
            KvStorage::TurboSym4 { k, v, .. } => {
                tracing::trace!(quant = "tsym4", "update_tsym: decode step");
                tsym_update::<4>(k, v, max_seq, "TurboSym4", new_k, new_v, device)
            }
            other => Err(storage_mismatch("TurboSym3 | TurboSym4", other)),
        }
    }
    /// K8VTurbo3Tcq decode update — K = affine q8_0,
    /// V = TurboQuant 3-bit with Viterbi trellis (TCQ) assignment.
    ///
    /// Mirrors [`Self::update_k8vturbo3`] structurally — same K-side q8_0
    /// dispatch, same CPU V-side path; only `QuantV::use_tcq` is set so that
    /// `QuantV::append` calls the Viterbi encoder
    /// ([`crate::tcq::tcq_quantize_v3`]) instead of nearest-centroid. The
    /// decoder is shared with plain turbo3 (TCQ output layout is byte-for-byte
    /// identical), so `dequantize_choice` is unchanged.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn update_k8vturbo3_tcq(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo3Tcq { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo3Tcq, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: force CPU — the TCQ MSL kernel ships as a future-ref hook
        // (see `tcq_v_msl.rs` Dispatch status).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantK {
                codes: Vec::new(),
                scales: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                max_seq,
            });
        }
        let ks = k.as_mut().unwrap();
        ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
        let k_shape = ks.shape.clone();
        let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
        let k_full = match k_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
        };

        if v.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 3,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: true,
            });
        }
        let vs = v.as_mut().unwrap();
        // Force CPU on the TCQ encode path (Viterbi over the trellis).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }
    /// K8VTurbo2Tcq decode update — K = affine q8_0,
    /// V = TurboQuant **2-bit** with Viterbi trellis (TCQ) assignment.
    ///
    /// Structurally identical to [`Self::update_k8vturbo3_tcq`] — same K-side
    /// GPU-capable affine q8_0 path (same as K8V4 / K8VTurbo3), same forced-CPU
    /// V-side. TCQ is an **encode-side** Viterbi trellis assignment only; decode
    /// is bit-identical to plain turbo and reuses the turbo GPU dequant. The
    /// production V-encode runs forced-CPU here: the 2-bit TCQ encode has no
    /// wired MSL kernel (its hook was removed), while the 3-bit encode kernel in
    /// `tcq_v_msl.rs` stays parked. Only the V-side `bits` changes from 3 to 2.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction: if k/v.is_none() block above guarantees Some on both branches"
    )]
    pub(super) fn update_k8vturbo2_tcq(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo2Tcq { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo2Tcq, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: force CPU — the TCQ MSL kernel ships as a future-ref hook.
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantK {
                codes: Vec::new(),
                scales: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                max_seq,
            });
        }
        let ks = k.as_mut().unwrap();
        ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
        let k_shape = ks.shape.clone();
        let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
        let k_full = match k_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
        };

        if v.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 2,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: true,
            });
        }
        let vs = v.as_mut().unwrap();
        // Force CPU on the 2-bit TCQ encode path (Viterbi over the trellis).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }
    /// K8VTurbo2 decode update — K = affine q8_0, V = TurboQuant
    /// 2-bit Lloyd-Max (codebook from `turboquant::lloyd_gaussian_codebook(2)`).
    ///
    /// Mirrors [`update_k8vturbo3`](Self::update_k8vturbo3) byte-for-byte with
    /// `bits=2` on the V side. CPU dequant only on the hot path: the MSL kernel
    /// in `turbo2_v_msl.rs` is wired as a future-reference hook (unit-tested
    /// for bit-exact CPU↔GPU parity) but never dispatched. The naïve Lloyd-Max
    /// 2-bit codebook ships without outlier-mask; outlier-mask deferred pending
    /// calibration loader. See `docs/KV_QUANT.md`
    /// for the gap-vs-mtq quantification.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn update_k8vturbo2(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo2 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo2, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: force CPU for 2-bit (no GPU dispatch on the hot path; the
        // MSL kernel is a future-reference hook in `turbo2_v_msl.rs`).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantK {
                codes: Vec::new(),
                scales: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                max_seq,
            });
        }
        // SAFETY: k is Some by construction — the `if k.is_none()` block above
        // assigns Some on both branches (initial + pre-existing), so this unwrap
        // cannot fail.
        #[allow(
            clippy::unwrap_used,
            reason = "Option is Some by construction: if k.is_none() block above guarantees Some on both branches"
        )]
        let ks = k.as_mut().unwrap();
        ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
        let k_shape = ks.shape.clone();
        let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
        let k_full = match k_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
        };

        if v.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 2,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
        }
        // SAFETY: v is Some by construction — the `if v.is_none()` block above
        // assigns Some on both branches (initial + pre-existing), so this unwrap
        // cannot fail.
        #[allow(
            clippy::unwrap_used,
            reason = "Option is Some by construction: if v.is_none() block above guarantees Some on both branches"
        )]
        let vs = v.as_mut().unwrap();
        // V-side: force CPU path for 2-bit (no GPU dispatch on the hot path;
        // see `update_k8vturbo2` doc-comment + `turbo2_v_msl.rs` "Dispatch status").
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    // K8VTurbo3 — bulk-quantize K (affine q8_0) + V (TurboQuant 3-bit).
    // CPU dequant only: no GPU path for 3-bit V (no MSL kernel this pass).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::unreachable,
        reason = "the storage variant is fixed by the `match self.quant` that selects this arm; a mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_k8vturbo3(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill K8VTurbo3: bulk-quantizing K (q8_0) + V (turbo3)"
        );
        let max_seq = match &self.storage {
            KvStorage::K8VTurbo3 { max_seq, .. } => *max_seq,
            _ => unreachable!("KvQuant::K8VTurbo3 but storage is not K8VTurbo3"),
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;

        let KvStorage::K8VTurbo3 { k, v, .. } = &mut self.storage else {
            unreachable!("KvQuant::K8VTurbo3 but storage is not K8VTurbo3");
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qk = QuantK {
            codes: Vec::new(),
            scales: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape.clone(),
            max_seq,
        };
        let mut qv = QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: 3,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        };
        // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        // V-side: CPU TurboQuant 3-bit (GPU path not yet available for bits=3).
        qv.append(&v_f32, &new_shape, v_full, Device::Cpu, max_seq)?;
        *k = Some(qk);
        *v = Some(qv);
        Ok(())
    }

    // TurboSym3 — symmetric 3-bit Lloyd-Max K + turbo3 V.
    // K side uses the GPU turbo3 MSL kernel (Decision B); V side forced CPU
    // (K8VTurbo3 precedent: GPU V-side dispatch regressed −2% TPS gate).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::unreachable,
        reason = "the storage variant is fixed by the `match self.quant` that selects this arm; a mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_turbo_sym3(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill TurboSym3: bulk-quantizing K (turbo3/GPU) + V (turbo3/CPU)"
        );
        let max_seq = match &self.storage {
            KvStorage::TurboSym3 { max_seq, .. } => *max_seq,
            _ => unreachable!("KvQuant::TurboSym3 but storage is not TurboSym3"),
        };
        let new_shape = k_full.shape();
        // K GPU-capable; V CPU-forced.
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(k_full, device)?
        };
        let v_f32 = array_to_f32_vec(v_full, Device::Cpu)?;

        let KvStorage::TurboSym3 { k, v, .. } = &mut self.storage else {
            unreachable!("KvQuant::TurboSym3 but storage is not TurboSym3");
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qk = QuantKTurbo3::new(init_shape.clone(), max_seq);
        let mut qv = QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: 3,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        };
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        // V-side: force CPU path for 3-bit (GPU kernel wired but disabled;
        // see update_k8vturbo3 doc-comment for the −2% gate fail).
        qv.append(&v_f32, &new_shape, v_full, Device::Cpu, max_seq)?;
        *k = Some(qk);
        *v = Some(qv);
        Ok(())
    }

    // TurboSym4 — symmetric 4-bit Lloyd-Max K + tq4 V. Both axes are
    // bulk-quantized via the same MSL kernel (axis-agnostic).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::unreachable,
        reason = "the storage variant is fixed by the `match self.quant` that selects this arm; a mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_turbo_sym4(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill TurboSym4: bulk-quantizing K (tq4) + V (tq4)"
        );
        let max_seq = match &self.storage {
            KvStorage::TurboSym4 { max_seq, .. } => *max_seq,
            _ => unreachable!("KvQuant::TurboSym4 but storage is not TurboSym4"),
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(k_full, v_full, device)?
        };

        let KvStorage::TurboSym4 { k, v, .. } = &mut self.storage else {
            unreachable!("KvQuant::TurboSym4 but storage is not TurboSym4");
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qk = QuantKTurbo4::new(init_shape.clone(), max_seq);
        let mut qv = QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: 4,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        };
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        qv.append(&v_f32, &new_shape, v_full, device, max_seq)?;
        *k = Some(qk);
        *v = Some(qv);
        Ok(())
    }

    // K8VTurbo2 — same shape as K8VTurbo3 with bits=2.
    // CPU dequant only (the turbo2 MSL kernel is a future-reference
    // hook, mirroring K8VTurbo3; see `turbo2_v_msl.rs`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_k8vturbo2(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill K8VTurbo2: bulk-quantizing K (q8_0) + V (turbo2)"
        );
        let max_seq = match &self.storage {
            KvStorage::K8VTurbo2 { max_seq, .. } => *max_seq,
            _ => return Err(Error::Mlx(
                "K8VTurbo2 exit_prefill: storage mismatch (KvQuant::K8VTurbo2 but storage is not K8VTurbo2)".into()
            )),
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;

        let KvStorage::K8VTurbo2 { k, v, .. } = &mut self.storage else {
            return Err(Error::Mlx(
                "K8VTurbo2 exit_prefill: storage mismatch (KvQuant::K8VTurbo2 but storage is not K8VTurbo2)".into()
            ));
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qk = QuantK {
            codes: Vec::new(),
            scales: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape.clone(),
            max_seq,
        };
        let mut qv = QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: 2,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        };
        // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        // V-side: CPU TurboQuant 2-bit (GPU path not wired on hot path).
        qv.append(&v_f32, &new_shape, v_full, Device::Cpu, max_seq)?;
        *k = Some(qk);
        *v = Some(qv);
        Ok(())
    }

    // K8VTurbo3Tcq — bulk-quantize K (affine q8_0) + V
    // (TurboQuant 3-bit with Viterbi assignment). Mirrors the K8VTurbo3
    // arm with `use_tcq = true` on the V slot so the encode dispatch in
    // QuantV::append picks the Viterbi path.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::unreachable,
        reason = "the storage variant is fixed by the `match self.quant` that selects this arm; a mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_k8vturbo3_tcq(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill K8VTurbo3Tcq: bulk-quantizing K (q8_0) + V (turbo3 + Viterbi)"
        );
        let max_seq = match &self.storage {
            KvStorage::K8VTurbo3Tcq { max_seq, .. } => *max_seq,
            _ => unreachable!("KvQuant::K8VTurbo3Tcq but storage is not K8VTurbo3Tcq"),
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;

        let KvStorage::K8VTurbo3Tcq { k, v, .. } = &mut self.storage else {
            unreachable!("KvQuant::K8VTurbo3Tcq but storage is not K8VTurbo3Tcq");
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qk = QuantK {
            codes: Vec::new(),
            scales: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape.clone(),
            max_seq,
        };
        let mut qv = QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: 3,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: true,
        };
        // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        // V-side: CPU TurboQuant 3-bit with Viterbi assignment.
        qv.append(&v_f32, &new_shape, v_full, Device::Cpu, max_seq)?;
        *k = Some(qk);
        *v = Some(qv);
        Ok(())
    }

    // K8VTurbo2Tcq — bulk-quantize K (affine q8_0) + V
    // (TurboQuant 2-bit with Viterbi assignment). Mirrors K8VTurbo3Tcq
    // with bits=2 and the `turbo2_tcq` max_compression preset.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::unreachable,
        reason = "the storage variant is fixed by the `match self.quant` that selects this arm; a mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_k8vturbo2_tcq(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill K8VTurbo2Tcq: bulk-quantizing K (q8_0) + V (turbo2 + Viterbi)"
        );
        let max_seq = match &self.storage {
            KvStorage::K8VTurbo2Tcq { max_seq, .. } => *max_seq,
            _ => unreachable!("KvQuant::K8VTurbo2Tcq but storage is not K8VTurbo2Tcq"),
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;

        let KvStorage::K8VTurbo2Tcq { k, v, .. } = &mut self.storage else {
            unreachable!("KvQuant::K8VTurbo2Tcq but storage is not K8VTurbo2Tcq");
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qk = QuantK {
            codes: Vec::new(),
            scales: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape.clone(),
            max_seq,
        };
        let mut qv = QuantV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            bits: 2,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: true,
        };
        // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        // V-side: CPU TurboQuant 2-bit with Viterbi assignment.
        qv.append(&v_f32, &new_shape, v_full, Device::Cpu, max_seq)?;
        *k = Some(qk);
        *v = Some(qv);
        Ok(())
    }
}
