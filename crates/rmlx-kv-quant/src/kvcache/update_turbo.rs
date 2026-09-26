//! TurboQuant KV update path.
//!
//! Holds every update-side body that only the TurboQuant storage types use:
//! the two decode entries and the two prefill entries, the width-parametric
//! symmetric bodies they enter, and the affine-K / turbo-V bodies the four
//! `K8VTurbo*` spellings share. The helpers with more than one family
//! caller stay in [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::{KvStorage, QuantK, QuantKTurbo, QuantV, TURBO_K4_BITS};

use super::helpers::{array_to_f32_vec, arrays_to_f32, f32_vec_to_array};
use super::update::{storage_mismatch, warn_if_width_disagrees};
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
/// device returns `Error::Quant` on every GPU append. The host vector `v_f32` is
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
        array_to_f32_vec(new_v, v_device)?
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

/// The four storage spellings [`KvCache::update_k8_turbo_v`] and
/// [`KvCache::exit_prefill_k8_turbo_v`] serve. Named because the mismatch
/// diagnostic and the two entries have to agree on the list.
const K8_TURBO_V_VARIANTS: &str = "K8VTurbo3 | K8VTurbo2 | K8VTurbo3Tcq | K8VTurbo2Tcq";

/// Decode update for the four affine-K / TurboQuant-V spellings —
/// `K8VTurbo3`, `K8VTurbo2` and their two TCQ siblings.
///
/// K is affine q8_0 and GPU-capable, the same path `update_k8v4` takes. V is
/// [`QuantV`] at `v_bits`, and its axis is forced onto the CPU: the GPU branch
/// of [`QuantV::append`] refuses `bits != 4`. The 3-bit kernel in
/// `k8vturbo3_append_msl.rs` serves the 3-bit K store and the TurboSym3
/// fused-QK encode, not this V axis; the 2-bit and TCQ kernels have no
/// production dispatch.
///
/// `use_tcq` selects the encoder inside [`QuantV::append`]: Viterbi over the
/// trellis when set, nearest-centroid otherwise. The decoder is shared —
/// TCQ output layout is byte-for-byte the plain layout — so
/// `dequantize_choice` does not read it.
///
/// `variant` is the storage spelling the caller resolved, used only in the
/// "buffer absent after init" diagnostics.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the decode shape is rank-4 and `init_shape` is its clone"
)]
pub(super) fn k8_turbo_v_update(
    k: &mut Option<QuantK>,
    v: &mut Option<QuantV>,
    max_seq: i32,
    v_bits: u8,
    use_tcq: bool,
    variant: &'static str,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();

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
            bits: v_bits,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq,
        });
    }
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx(format!("{variant} V buffer absent after init")));
    };
    vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
    let v_shape = vs.shape.clone();
    let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
    let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

    Ok((k_full, v_full))
}

/// Prefill bulk encode for the same four spellings: K affine q8_0, V
/// [`QuantV`] at `v_bits` on the CPU axis. `use_tcq` selects the Viterbi
/// encoder, as in [`k8_turbo_v_update`].
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
)]
pub(super) fn k8_turbo_v_bulk_encode(
    k: &mut Option<QuantK>,
    v: &mut Option<QuantV>,
    max_seq: i32,
    v_bits: u8,
    use_tcq: bool,
    k_full: &Array,
    v_full: &Array,
    device: Device,
    total_seq: i32,
) -> Result<()> {
    tracing::debug!(
        total_seq,
        v_bits,
        use_tcq,
        "exit_prefill affine-K / turbo-V: bulk-quantizing K (q8_0) + V (turbo CPU)"
    );
    let new_shape = k_full.shape();
    let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;
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
        bits: v_bits,
        max_seq,
        high_precision_indices: None,
        value_codebook: None,
        value_codebook_gpu: None,
        use_tcq,
    };
    qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
    qv.append(&v_f32, &new_shape, v_full, Device::Cpu, max_seq)?;
    *k = Some(qk);
    *v = Some(qv);
    Ok(())
}

/// Symmetric TurboQuant prefill bulk encode: K is [`QuantKTurbo`] at `BITS`,
/// V is [`QuantV`] at the same width.
///
/// The V-axis device is the one thing the two widths do not share, and the
/// rule is [`tsym_update`]'s: the V axis takes the caller's device at 4 bits
/// and is pinned to `Device::Cpu` at 3, because [`QuantV::append`] enters its
/// GPU branch on the device alone and then refuses `bits != 4`. The host
/// vector for an axis is materialised exactly when that axis resolves to the
/// CPU.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
)]
pub(super) fn tsym_bulk_encode<const BITS: u8>(
    k: &mut Option<QuantKTurbo<BITS>>,
    v: &mut Option<QuantV>,
    max_seq: i32,
    k_full: &Array,
    v_full: &Array,
    device: Device,
    total_seq: i32,
) -> Result<()> {
    let v_device = if BITS == TURBO_K4_BITS {
        device
    } else {
        Device::Cpu
    };
    tracing::debug!(
        total_seq,
        bits = BITS,
        v_device = ?v_device,
        "exit_prefill turbo symmetric: bulk-quantizing K + V (both turbo)"
    );
    let new_shape = k_full.shape();
    let k_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(k_full, device)?
    };
    let v_f32 = if v_device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(v_full, v_device)?
    };
    let mut init_shape = new_shape.clone();
    init_shape[2] = 0;
    let mut qk = QuantKTurbo::<BITS>::new(init_shape.clone(), max_seq);
    let mut qv = QuantV {
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
    };
    qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
    qv.append(&v_f32, &new_shape, v_full, v_device, max_seq)?;
    *k = Some(qk);
    *v = Some(qv);
    Ok(())
}

/// The V code width, the TCQ flag and the spelling for the four affine-K /
/// TurboQuant-V storage variants.
///
/// One table, read by both [`KvCache::update_k8_turbo_v`] and
/// [`KvCache::exit_prefill_k8_turbo_v`]. Written twice, the two entries could
/// disagree about which spelling is which, and the prefill half of such a
/// disagreement is unobservable: its arms sit behind the
/// `materialises_packed_store()` gate, so only the decode half would turn a
/// cell red.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the table answers for four variants and `None` for the rest; a per-variant `None` arm for the other 23 would say nothing the fall-through does not"
)]
fn k8_turbo_v_knobs(storage: &KvStorage) -> Option<(u8, bool, &'static str)> {
    match storage {
        KvStorage::K8VTurbo3 { .. } => Some((3, false, "K8VTurbo3")),
        KvStorage::K8VTurbo2 { .. } => Some((2, false, "K8VTurbo2")),
        KvStorage::K8VTurbo3Tcq { .. } => Some((3, true, "K8VTurbo3Tcq")),
        KvStorage::K8VTurbo2Tcq { .. } => Some((2, true, "K8VTurbo2Tcq")),
        _ => None,
    }
}

impl KvCache {
    /// Decode update for the four affine-K / TurboQuant-V spellings. The body
    /// is [`k8_turbo_v_update`] at the V width and TCQ flag
    /// [`k8_turbo_v_knobs`] resolves; this entry takes the warm-TTFT bf16
    /// shortcut and hands the body its stores.
    pub(crate) fn update_k8_turbo_v(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let Some((v_bits, use_tcq, variant)) = k8_turbo_v_knobs(&self.storage) else {
            return Err(storage_mismatch(K8_TURBO_V_VARIANTS, &self.storage));
        };
        let max_seq = self.max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let (KvStorage::K8VTurbo3 { k, v, .. }
        | KvStorage::K8VTurbo2 { k, v, .. }
        | KvStorage::K8VTurbo3Tcq { k, v, .. }
        | KvStorage::K8VTurbo2Tcq { k, v, .. }) = &mut self.storage
        else {
            return Err(storage_mismatch(K8_TURBO_V_VARIANTS, &self.storage));
        };
        k8_turbo_v_update(
            k, v, max_seq, v_bits, use_tcq, variant, new_k, new_v, device,
        )
    }

    /// Prefill bulk encode for the four affine-K / TurboQuant-V spellings. The
    /// body is [`k8_turbo_v_bulk_encode`] at the V width and TCQ flag
    /// [`k8_turbo_v_knobs`] resolves — the same table the decode entry reads.
    pub(crate) fn exit_prefill_k8_turbo_v(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        let Some((v_bits, use_tcq, _variant)) = k8_turbo_v_knobs(&self.storage) else {
            return Err(storage_mismatch(K8_TURBO_V_VARIANTS, &self.storage));
        };
        warn_if_width_disagrees(self.quant, self.quant.approx_code_bits().1, v_bits);
        let max_seq = self.max_seq;
        let (KvStorage::K8VTurbo3 { k, v, .. }
        | KvStorage::K8VTurbo2 { k, v, .. }
        | KvStorage::K8VTurbo3Tcq { k, v, .. }
        | KvStorage::K8VTurbo2Tcq { k, v, .. }) = &mut self.storage
        else {
            return Err(storage_mismatch(K8_TURBO_V_VARIANTS, &self.storage));
        };
        k8_turbo_v_bulk_encode(
            k, v, max_seq, v_bits, use_tcq, k_full, v_full, device, total_seq,
        )
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
    pub(crate) fn update_tsym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let max_seq = self.max_seq;
        let (KvStorage::TurboSym3 { .. } | KvStorage::TurboSym4 { .. }) = &self.storage else {
            return Err(storage_mismatch("TurboSym3 | TurboSym4", &self.storage));
        };

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
    /// Symmetric TurboQuant prefill bulk encode at the width the active
    /// storage variant carries (`TurboSym3` is 3 bits, `TurboSym4` is 4). The
    /// body is [`tsym_bulk_encode`]; this entry resolves the storage variant.
    pub(crate) fn exit_prefill_turbo_sym(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        let quant_bits = self.quant.approx_code_bits().0;
        let max_seq = self.max_seq;
        if let KvStorage::TurboSym3 { k, v, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 3);
            tsym_bulk_encode::<3>(k, v, max_seq, k_full, v_full, device, total_seq)
        } else if let KvStorage::TurboSym4 { k, v, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 4);
            tsym_bulk_encode::<4>(k, v, max_seq, k_full, v_full, device, total_seq)
        } else {
            Err(storage_mismatch("TurboSym3 | TurboSym4", &self.storage))
        }
    }
}
