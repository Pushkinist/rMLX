//! PlanarQuant KV update path.
//!
//! Holds every update-side body that only the planar storage types use: the
//! `update_planar` and K-only `update_planar_k` decode entries, the second of
//! which carries the warm-TTFT bf16 bypass, and the `exit_prefill_planar` /
//! `exit_prefill_planar_k` prefill bulk-encode bodies. The `KvStorage`
//! dispatch and the helpers with more than one family caller stay in
//! [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::{KvStorage, QuantK, QuantPlanarK, QuantPlanarV};

use super::helpers::{array_to_f32_vec, arrays_to_f32, f32_vec_to_array};
use super::update::storage_mismatch;
use super::KvCache;

impl KvCache {
    #[allow(
        clippy::unreachable,
        reason = "storage variant is guaranteed by the `match &self.storage` dispatch in \
                  KvCache::update() (KvStorage::Planar arm); \
                  mismatch is a construction-time BUG"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn update_planar(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let max_seq = self.max_seq;
        let KvStorage::Planar { k, v, bits, .. } = &mut self.storage else {
            unreachable!("storage mismatch: expected Planar");
        };
        let v_bits = *bits;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(new_k, new_v, device)?
        };

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
            *v = Some(QuantPlanarV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_rotations_buf: None,
                gpu_codes_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_rotations_words_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                max_seq,
                bits: v_bits,
            });
        }
        let vs = v.as_mut().unwrap();
        vs.append(&v_f32, &new_shape, new_v, device, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, v_arr_opt) = vs.dequantize_choice(device, new_v.dtype())?;
        let v_full = match v_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&v_recon_f32, &v_shape)?,
        };

        Ok((k_full, v_full))
    }
    /// PlanarK decode update. K is quantized through `QuantPlanarK`
    /// (shared MSL kernel with `Planar` V-side — PlanarQuant is axis-agnostic).
    /// V stays bf16 in `decode_fp16_v` (same machinery as `KvStorage::None`).
    ///
    /// Warm-TTFT shortcut: when `decode_fp16_k` is present (set by
    /// `exit_prefill`), route through `update_decode_fp16` so the bf16 K seed
    /// is used for the rest of the request, as every mirror-fed codec does
    /// (`docs/KV_CACHE.md` §9.6). Without it PlanarK would re-encode K through
    /// the lossy 4-bit Lloyd-Max + Givens kernel on every decode step.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn update_planar_k(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let max_seq = self.max_seq;
        let KvStorage::PlanarK { k, .. } = &mut self.storage else {
            return Err(storage_mismatch("PlanarK", &self.storage));
        };

        // Warm-TTFT bf16 K seed path, the shortcut every mirror-fed
        // `update_<codec>` takes.
        if self.decode_fp16_k.is_some() {
            tracing::debug!(
                target: "rmlx_kv_quant::warm_ttft",
                path = "warm_ttft_bypass",
                codec = "PlanarK",
                offset = self.offset,
                "PlanarK update routing through warm-TTFT bf16 K seed; \
                 4-bit codec stays quiescent for this decode step"
            );
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };

        // ── K side: QuantPlanarK encode + dequant for SDPA input ─────────────
        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantPlanarK::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("PlanarK K buffer absent after init".into()));
        };
        ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
        let k_shape = ks.shape.clone();
        let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
        let k_full = match k_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
        };

        // ── V side: bf16 via update_decode_fp16 (same as KvStorage::None V).
        // We pass the original `new_k` shadow for the unused K side; the V
        // returned is the one we keep. K from quant codec is what SDPA sees.
        let (_k_shadow, v_full) = self.update_decode_fp16(new_k, new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(crate) fn exit_prefill_planar(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        _total_seq: i32,
    ) -> Result<()> {
        let (max_seq, v_bits) = match &self.storage {
            KvStorage::Planar { bits, .. } => (self.max_seq, *bits),
            _ => return Err(storage_mismatch("Planar", &self.storage)),
        };
        let new_shape = k_full.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(k_full, v_full, device)?
        };

        let KvStorage::Planar { k, v, .. } = &mut self.storage else {
            return Err(storage_mismatch("Planar", &self.storage));
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
        let mut qpv = QuantPlanarV {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_rotations_buf: None,
            gpu_codes_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_rotations_words_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            max_seq,
            bits: v_bits,
        };
        qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        qpv.append(&v_f32, &new_shape, v_full, device, max_seq)?;
        *k = Some(qk);
        *v = Some(qpv);
        Ok(())
    }

    // PlanarK — bulk-quantize K via QuantPlanarK; V stays
    // bf16 (materialised by the caller's decode_fp16_pair).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(crate) fn exit_prefill_planar_k(
        &mut self,
        k_full: &Array,
        _v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill PlanarK: bulk-quantizing K (planar4); V stays bf16"
        );
        let max_seq = match &self.storage {
            KvStorage::PlanarK { .. } => self.max_seq,
            _ => return Err(storage_mismatch("PlanarK", &self.storage)),
        };
        let new_shape = k_full.shape();
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(k_full, device)?
        };

        let KvStorage::PlanarK { k, .. } = &mut self.storage else {
            return Err(storage_mismatch("PlanarK", &self.storage));
        };
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        let mut qpk = QuantPlanarK::new(init_shape, max_seq);
        qpk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
        *k = Some(qpk);
        // V side: bf16 — materialized by the caller's decode_fp16_pair.
        Ok(())
    }
}
