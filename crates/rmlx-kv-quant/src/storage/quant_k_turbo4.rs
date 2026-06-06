// K-side TurboQuant 4-bit storage, mirror of `QuantV`'s turbo4 layout.
//
// Promoted to `pub` for the SSD modules (block_io / hydrate / spill) under
// the same rationale as the sibling `QuantK` / `QuantV` structs.
#![allow(
    missing_docs,
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
//! Quantized K buffer with **TurboQuant 4-bit** layout: [`QuantKTurbo4`].
//!
//! This is the K-side counterpart to [`super::QuantV`] (V-side TurboQuant 4-bit).
//! The two structs are independent — symmetric WHT-4 K + tq4 V form the new
//! [`super::KvStorage::TurboSym4`] storage variant.
//!
//! The CPU codec ([`crate::turboquant::turbo_quantize_v`] / `turbo_dequantize`)
//! and MSL kernel ([`crate::turboquant_msl::turbo_quantize_v4_gpu`] /
//! `turbo_dequantize_v4_gpu`) are axis-agnostic — they take a flat f32 buffer
//! plus a 4-D shape and produce flat codes/scales. Re-using them for the K side
//! is exact: no kernel fork (Step 0 decision documented in `docs/KV_QUANT.md`).
//!
//! # Layout
//!
//! * GPU codes  — `u32 [B × kv_h × max_seq × D / 8]`  (8 nibble indices / u32)
//! * GPU scales — `f32 [B × kv_h × max_seq × D / 32]` (one f32 / 32-elem group)
//! * CPU blocks — `Vec<TurboBlocks>` (one per decode step), bit-packed indices
//!
//! Identical to [`super::QuantV`] except the contained K vectors are the
//! attention-key projections rather than the value projections.
#![allow(clippy::too_many_lines)]

use rmlx_core::error::Result;
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::turboquant_msl::{turbo_dequantize_v4_gpu, turbo_quantize_v4_gpu};

use crate::turboquant::{turbo_dequantize, turbo_quantize_v, TurboBlocks, GROUP_SIZE};

use super::KV_PAGE_SIZE;

// ── Quantized K (turbo4) storage ─────────────────────────────────────────────

/// Accumulated TurboQuant K4 cache — symmetric counterpart of [`super::QuantV`].
///
/// The struct mirrors `QuantV`'s field layout exactly. The two are kept as
/// independent types (not a renamed wrapper) so the K-side and V-side buffers
/// remain decoupled in [`super::KvStorage::TurboSym4`] dispatch (Step 0
/// requirement from the TurboSym4 symmetric variant).
#[derive(Debug)]
pub struct QuantKTurbo4 {
    // ── CPU path ────────────────────────────────────────────────────────────
    /// Per-decode-step TurboQuant blocks accumulated on the CPU path.
    pub blocks: Vec<TurboBlocks>,
    // ── GPU path ────────────────────────────────────────────────────────────
    /// Pre-allocated codes buffer (`u32`, 4 words per group of GROUP_SIZE=32 elements).
    /// Length: `B * kv_h * max_seq * D / 8`.
    pub gpu_codes_buf: Option<Array>,
    /// Pre-allocated scales buffer (`f32`, one per group of GROUP_SIZE=32).
    /// Length: `B * kv_h * max_seq * D / GROUP_SIZE`.
    pub gpu_scales_buf: Option<Array>,
    /// Number of u32 codes written per single-step.
    pub gpu_words_per_step: i32,
    /// Number of f32 scales written per single-step.
    pub gpu_scales_per_step: i32,
    /// Current allocated capacity in tokens (paged growth).
    pub gpu_capacity: i32,
    // ── Shared ──────────────────────────────────────────────────────────────
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Quantization bit-width (always 4 for this struct).
    pub bits: u8,
    /// Maximum sequence length the GPU buffer was sized for.
    pub max_seq: i32,
}

impl QuantKTurbo4 {
    /// Append a new K slice. CPU path uses scalar Rust; GPU path uses MSL kernel
    /// + pre-allocated 1D buffer with `slice_update`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unreachable,
        reason = "gpu_codes_buf / gpu_scales_buf are set to Some just above in the same block; unreachable! documents the local invariant without introducing a fallible return path"
    )]
    pub fn append(
        &mut self,
        f32_data: &[f32],
        new_shape: &[i32],
        k_arr: &Array,
        device: Device,
        max_seq: i32,
    ) -> Result<()> {
        let prev_seq = self.shape[2];
        self.shape[2] += new_shape[2];

        if device == Device::Gpu {
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let d = new_shape[3] as usize;

            if self.gpu_codes_buf.is_none() {
                // TurboQuant 4-bit: 4 u32 codes per group of GROUP_SIZE=32 elements
                // (same layout as the V-side path — axis-agnostic kernel).
                let words_per_step = b * kv_h * d * 4 / GROUP_SIZE;
                let scales_per_step = b * kv_h * d / GROUP_SIZE;
                self.gpu_words_per_step = words_per_step as i32;
                self.gpu_scales_per_step = scales_per_step as i32;
                self.max_seq = max_seq;
                let init_cap = if prev_seq > 0 {
                    let needed = prev_seq + new_shape[2];
                    let pages = (needed + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
                    (pages * KV_PAGE_SIZE).min(max_seq)
                } else {
                    KV_PAGE_SIZE.min(max_seq)
                };
                self.gpu_capacity = init_cap;
                let codes_buf = zeros(
                    &[(words_per_step * init_cap as usize) as i32],
                    Dtype::U32,
                    device,
                )?;
                let scales_buf = zeros(
                    &[(scales_per_step * init_cap as usize) as i32],
                    Dtype::F32,
                    device,
                )?;
                // Hydrated init: upload CPU TurboBlocks to GPU when prev_seq > 0
                // and self.blocks is non-empty. Same layout policy as QuantV.
                let (codes_buf, scales_buf) = if !self.blocks.is_empty() && prev_seq > 0 {
                    let mut flat_codes: Vec<u8> = Vec::new();
                    let mut flat_scales: Vec<f32> = Vec::new();
                    for blk in &self.blocks {
                        flat_codes.extend_from_slice(&blk.codes);
                        flat_scales.extend_from_slice(&blk.scales);
                    }
                    let cpu_words = flat_codes.len() / 4;
                    let cpu_scales = flat_scales.len();
                    // SAFETY: `flat_scales` is `Vec<f32>`; `f32` is `Copy` with a
                    // fixed 4-byte LE layout. Reinterpreting as `&[u8]` is safe
                    // because `f32` and `u8` have no alignment or validity
                    // requirements beyond what `Vec` guarantees.
                    let scales_bytes = unsafe {
                        std::slice::from_raw_parts(
                            flat_scales.as_ptr().cast::<u8>(),
                            flat_scales.len() * 4,
                        )
                    };
                    let cpu_codes_arr =
                        Array::from_bytes(&flat_codes, &[cpu_words as i32], Dtype::U32)?;
                    let cpu_scales_arr =
                        Array::from_bytes(scales_bytes, &[cpu_scales as i32], Dtype::F32)?;
                    let codes_buf = codes_buf.slice_update(
                        &cpu_codes_arr,
                        &[0],
                        &[cpu_words as i32],
                        &[1],
                        device,
                    )?;
                    let scales_buf = scales_buf.slice_update(
                        &cpu_scales_arr,
                        &[0],
                        &[cpu_scales as i32],
                        &[1],
                        device,
                    )?;
                    tracing::debug!(
                        prev_seq,
                        init_cap,
                        cpu_words,
                        cpu_scales,
                        "QuantKTurbo4 hydrated init: uploaded CPU TurboBlocks -> GPU"
                    );
                    (codes_buf, scales_buf)
                } else {
                    (codes_buf, scales_buf)
                };
                self.gpu_codes_buf = Some(codes_buf);
                self.gpu_scales_buf = Some(scales_buf);
            }

            // ── Grow if needed ───────────────────────────────────────────────
            let needed = self.shape[2];
            if needed > self.gpu_capacity {
                let new_cap = {
                    let pages = (needed + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
                    (pages * KV_PAGE_SIZE).min(max_seq)
                };
                let words_per_step = self.gpu_words_per_step as usize;
                let scales_per_step = self.gpu_scales_per_step as usize;
                let new_codes = zeros(
                    &[(words_per_step * new_cap as usize) as i32],
                    Dtype::U32,
                    device,
                )?;
                let new_scales = zeros(
                    &[(scales_per_step * new_cap as usize) as i32],
                    Dtype::F32,
                    device,
                )?;
                let Some(old_codes) = self.gpu_codes_buf.take() else {
                    unreachable!("gpu_codes_buf set in init block above")
                };
                let Some(old_scales) = self.gpu_scales_buf.take() else {
                    unreachable!("gpu_scales_buf set in init block above")
                };
                let filled_words = prev_seq * self.gpu_words_per_step;
                let filled_scales = prev_seq * self.gpu_scales_per_step;
                let new_codes = if filled_words > 0 {
                    new_codes.slice_update(
                        &old_codes.slice(&[0], &[filled_words], &[1], device)?,
                        &[0],
                        &[filled_words],
                        &[1],
                        device,
                    )?
                } else {
                    new_codes
                };
                let new_scales = if filled_scales > 0 {
                    new_scales.slice_update(
                        &old_scales.slice(&[0], &[filled_scales], &[1], device)?,
                        &[0],
                        &[filled_scales],
                        &[1],
                        device,
                    )?
                } else {
                    new_scales
                };
                self.gpu_codes_buf = Some(new_codes);
                self.gpu_scales_buf = Some(new_scales);
                self.gpu_capacity = new_cap;
            }

            let new_seq = new_shape[2];
            let words_per_seq = self.gpu_words_per_step;
            let scales_per_seq = self.gpu_scales_per_step;

            let (new_codes, new_scales) = turbo_quantize_v4_gpu(k_arr, device)?;

            let codes_start = prev_seq * words_per_seq;
            let codes_stop = (prev_seq + new_seq) * words_per_seq;
            let scales_start = prev_seq * scales_per_seq;
            let scales_stop = (prev_seq + new_seq) * scales_per_seq;

            let Some(codes_buf) = self.gpu_codes_buf.take() else {
                unreachable!("gpu_codes_buf set in init block above")
            };
            let Some(scales_buf) = self.gpu_scales_buf.take() else {
                unreachable!("gpu_scales_buf set in init block above")
            };

            self.gpu_codes_buf = Some(codes_buf.slice_update(
                &new_codes,
                &[codes_start],
                &[codes_stop],
                &[1],
                device,
            )?);
            self.gpu_scales_buf = Some(scales_buf.slice_update(
                &new_scales,
                &[scales_start],
                &[scales_stop],
                &[1],
                device,
            )?);
        } else {
            // CPU path: scalar Rust quantization (axis-agnostic — same codec as
            // V-side; the codebook is N(0,1) Lloyd-Max regardless of which axis
            // the data came from).
            let block = turbo_quantize_v(f32_data, self.bits, new_shape)?;
            self.blocks.push(block);
        }
        Ok(())
    }

    /// Dequantize all accumulated K slices to a flat f32 vec (CPU path) or
    /// directly return the MLX `Array` (GPU path).
    ///
    /// Returns `(flat_f32, None)` for CPU or `(empty, Some(Array))` for GPU.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn dequantize_choice(
        &self,
        device: Device,
        out_dtype: Dtype,
    ) -> Result<(Vec<f32>, Option<Array>)> {
        if device == Device::Gpu {
            if let (Some(codes_buf), Some(scales_buf)) = (&self.gpu_codes_buf, &self.gpu_scales_buf)
            {
                let s = self.shape[2];
                let codes = codes_buf.slice(&[0], &[s * self.gpu_words_per_step], &[1], device)?;
                let scales =
                    scales_buf.slice(&[0], &[s * self.gpu_scales_per_step], &[1], device)?;
                let out = turbo_dequantize_v4_gpu(&codes, &scales, &self.shape, out_dtype, device)?;
                return Ok((Vec::new(), Some(out)));
            }
            return Ok((Vec::new(), None));
        }
        // CPU path.
        let total: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out = Vec::with_capacity(total);
        for block in &self.blocks {
            let slice = turbo_dequantize(block)?;
            out.extend_from_slice(&slice);
        }
        Ok((out, None))
    }

    /// Reconstruct a CPU-path `QuantKTurbo4` from serialized TurboQuant blocks.
    /// GPU buffers stay empty.
    pub fn from_cpu_blocks(blocks: Vec<TurboBlocks>, shape: Vec<i32>, bits: u8) -> Self {
        Self {
            blocks,
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape,
            bits,
            max_seq: 0,
        }
    }

    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            blocks: self.blocks.clone(),
            gpu_codes_buf: match &self.gpu_codes_buf {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            gpu_scales_buf: match &self.gpu_scales_buf {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            gpu_words_per_step: self.gpu_words_per_step,
            gpu_scales_per_step: self.gpu_scales_per_step,
            gpu_capacity: self.gpu_capacity,
            shape: self.shape.clone(),
            bits: self.bits,
            max_seq: self.max_seq,
        })
    }
}
