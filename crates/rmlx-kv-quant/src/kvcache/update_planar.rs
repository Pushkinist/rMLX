//! PlanarQuant KV update path.
//!
//! Holds the two update-side bodies that only the planar storage types use:
//! `update_planar` and the K-only `update_planar_k`, which carries the
//! warm-TTFT bf16 bypass. The `KvStorage` dispatch and the helpers with more
//! than one family caller stay in [`super::update`].
#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::too_many_lines
)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::{KvStorage, QuantK, QuantPlanarK, QuantPlanarV};

use super::helpers::{array_to_f32_vec, arrays_to_f32, f32_vec_to_array, storage_variant_name};
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
    pub(super) fn update_planar(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::Planar {
            k,
            v,
            max_seq,
            bits,
        } = &mut self.storage
        else {
            unreachable!("storage mismatch: expected Planar");
        };
        let max_seq = *max_seq;
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
    /// is used for the rest of the request. This matches every other
    /// `update_<arch>` (K8V4/K8V8/Planar/Mixed/K8VTurbo*/Iso*/Rotor*/
    /// TurboSym*). Before this fix, PlanarK was the **sole** codec that
    /// re-encoded K through the lossy 4-bit Lloyd-Max + Givens kernel on every
    /// decode step while every other variant silently stayed in bf16 K, and
    /// that asymmetry surfaced as the Bonsai PlanarK NIAH retrieval failure.
    ///
    /// See `docs/reports/planar-chunked-prefill-fix.md` § "Followups"
    /// for the open question of whether warm-TTFT-as-default is the intended
    /// steady-state design for the entire quantised-KV surface.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn update_planar_k(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::PlanarK { k, max_seq } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "PlanarK",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        // Warm-TTFT bf16 K seed path. Same shortcut every other
        // `update_<arch>` honours; before this fix PlanarK lacked it and was
        // the only codec exercising the per-decode-step 4-bit K encode.
        //
        // See `docs/reports/planar-chunked-prefill-fix.md` § "Followups"
        // for the open question of whether warm-TTFT-as-default is the intended
        // steady-state design for the entire quantised-KV surface.
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
}
