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

/// The four storage spellings [`KvCache::update_k8_turbo_v`] and
/// [`KvCache::exit_prefill_k8_turbo_v`] serve. Named because the mismatch
/// diagnostic and the two entries have to agree on the list.
const K8_TURBO_V_VARIANTS: &str = "K8VTurbo3 | K8VTurbo2 | K8VTurbo3Tcq | K8VTurbo2Tcq";

/// Decode update for the four affine-K / TurboQuant-V spellings —
/// `K8VTurbo3`, `K8VTurbo2` and their two TCQ siblings.
///
/// K is affine q8_0 and GPU-capable, the same path `update_k8v4` takes. V is
/// [`QuantV`] at `v_bits`, and its axis is forced onto the CPU: the 3-bit
/// Metal kernel regressed Gemma4-e4b by ~3.5 % and Gemma4-26b by ~6.9 %
/// against the `Mixed{v_bits:3}` affine baseline, so the GPU dispatch was
/// reverted and the kernel source stays a future-reference hook in
/// `k8vturbo3_append_msl.rs` — see `docs/research/turboquant_v3_vs_affine_v3.md`
/// "Second pass". The 2-bit and TCQ kernels never had a hot-path dispatch.
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

impl KvCache {
    /// Decode update for the four affine-K / TurboQuant-V spellings. The body
    /// is [`k8_turbo_v_update`] at the V width and TCQ flag the active storage
    /// variant carries; this entry resolves the variant and takes the
    /// warm-TTFT bf16 shortcut.
    pub(super) fn update_k8_turbo_v(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let (KvStorage::K8VTurbo3 { max_seq, .. }
        | KvStorage::K8VTurbo2 { max_seq, .. }
        | KvStorage::K8VTurbo3Tcq { max_seq, .. }
        | KvStorage::K8VTurbo2Tcq { max_seq, .. }) = &self.storage
        else {
            return Err(Error::KvStorageMismatch {
                expected: K8_TURBO_V_VARIANTS,
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        if let KvStorage::K8VTurbo3 { k, v, .. } = &mut self.storage {
            k8_turbo_v_update(k, v, max_seq, 3, false, "K8VTurbo3", new_k, new_v, device)
        } else if let KvStorage::K8VTurbo2 { k, v, .. } = &mut self.storage {
            k8_turbo_v_update(k, v, max_seq, 2, false, "K8VTurbo2", new_k, new_v, device)
        } else if let KvStorage::K8VTurbo3Tcq { k, v, .. } = &mut self.storage {
            k8_turbo_v_update(k, v, max_seq, 3, true, "K8VTurbo3Tcq", new_k, new_v, device)
        } else if let KvStorage::K8VTurbo2Tcq { k, v, .. } = &mut self.storage {
            k8_turbo_v_update(k, v, max_seq, 2, true, "K8VTurbo2Tcq", new_k, new_v, device)
        } else {
            // Unreachable: the read above accepted no other variant.
            Err(Error::KvStorageMismatch {
                expected: K8_TURBO_V_VARIANTS,
                got: storage_variant_name(&self.storage),
            })
        }
    }

    /// Prefill bulk encode for the four affine-K / TurboQuant-V spellings. The
    /// body is [`k8_turbo_v_bulk_encode`]; this entry resolves the storage
    /// variant.
    pub(super) fn exit_prefill_k8_turbo_v(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        if let KvStorage::K8VTurbo3 { k, v, max_seq } = &mut self.storage {
            k8_turbo_v_bulk_encode(k, v, *max_seq, 3, false, k_full, v_full, device, total_seq)
        } else if let KvStorage::K8VTurbo2 { k, v, max_seq } = &mut self.storage {
            k8_turbo_v_bulk_encode(k, v, *max_seq, 2, false, k_full, v_full, device, total_seq)
        } else if let KvStorage::K8VTurbo3Tcq { k, v, max_seq } = &mut self.storage {
            k8_turbo_v_bulk_encode(k, v, *max_seq, 3, true, k_full, v_full, device, total_seq)
        } else if let KvStorage::K8VTurbo2Tcq { k, v, max_seq } = &mut self.storage {
            k8_turbo_v_bulk_encode(k, v, *max_seq, 2, true, k_full, v_full, device, total_seq)
        } else {
            Err(Error::KvStorageMismatch {
                expected: K8_TURBO_V_VARIANTS,
                got: storage_variant_name(&self.storage),
            })
        }
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
    // TurboSym3 — symmetric 3-bit Lloyd-Max K + turbo3 V.
    // K side uses the GPU turbo3 MSL kernel (Decision B); V side forced CPU
    // (K8VTurbo3 precedent: GPU V-side dispatch regressed −2% TPS gate).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
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
            _ => {
                return Err(Error::KvStorageMismatch {
                    expected: "TurboSym3",
                    got: storage_variant_name(&self.storage),
                })
            }
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
            return Err(Error::KvStorageMismatch {
                expected: "TurboSym3",
                got: storage_variant_name(&self.storage),
            });
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
            _ => {
                return Err(Error::KvStorageMismatch {
                    expected: "TurboSym4",
                    got: storage_variant_name(&self.storage),
                })
            }
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(k_full, v_full, device)?
        };

        let KvStorage::TurboSym4 { k, v, .. } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "TurboSym4",
                got: storage_variant_name(&self.storage),
            });
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
}
