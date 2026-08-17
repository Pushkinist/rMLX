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
//! `turbo_dequantize_v4_gpu`) are positional — they take a flat f32 buffer plus
//! a 4-D shape, group over the last axis, and produce flat codes/scales without
//! interpreting the head/seq axes. Re-using them for the K side is exact: no
//! kernel fork (decision documented in `docs/KV_QUANT.md`). Because the codec is
//! positional, `append` stores every chunk sequence-major (`[B, S, kv_h, D]`)
//! and `dequantize_choice` reorders back to the logical `[B, kv_h, S, D]`; a
//! flat head-major store would transpose heads across a multi-append GQA cache.
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
    /// Resident bytes held by this store: CPU blocks plus the GPU mirror.
    ///
    /// Both are summed unconditionally — an SSD-hydrate init leaves the
    /// pre-hydration CPU blocks resident under a live GPU mirror, so both are
    /// real memory at once. Unallocated buffers contribute 0 on their own.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            blocks,
            gpu_codes_buf,
            gpu_scales_buf,
            // Geometry / bookkeeping about the buffers above, not allocations.
            gpu_words_per_step: _,
            gpu_scales_per_step: _,
            gpu_capacity: _,
            shape: _,
            bits: _,
            max_seq: _,
        } = self;
        blocks.iter().map(TurboBlocks::byte_size).sum::<u64>()
            + crate::bytes::opt_array_bytes(gpu_codes_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_scales_buf.as_ref())
    }

    /// Truncate the store to `n` sequence positions.
    ///
    /// Drops trailing CPU blocks past `n` and **splits** the block the cut lands
    /// inside (see [`super::truncate_plan`]), then lowers `shape[2]` to `n`. The
    /// GPU buffers need no cut — see [`super::QuantV::truncate_to`].
    pub fn truncate_to(&mut self, n: i32) {
        let n = n.max(0);
        let plan = super::truncate_plan(
            self.blocks
                .iter()
                .map(|blk| super::block_rows(&blk.original_shape)),
            &self.shape,
            n,
        );
        super::apply_truncate_plan(&mut self.blocks, &plan);
        // `get_mut` rather than `shape[2]`: the store shape is rank-4 by
        // construction, and this is the bounds proof rather than a claim.
        if let Some(seq) = self.shape.get_mut(2) {
            *seq = n;
        }
    }

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

            // Reorder the chunk to sequence-major `[B, new_seq, kv_h, D]` before
            // quantizing: the flat buffer accumulates chunks at
            // `prev_seq * words_per_seq`, so a head-major store + head-major
            // reshape on dequant transposes heads across appends when kv_h>1.
            // `transpose` yields a strided view; the TurboQuant MSL kernel reads
            // its input by raw linear offset (ignores MLX strides), so
            // materialize the permutation with `contiguous` first.
            let k_seq_major = k_arr.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
            let (new_codes, new_scales) = turbo_quantize_v4_gpu(&k_seq_major, device)?;

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
            // the data came from). Store the chunk sequence-major so the CPU
            // blocks share one layout with the GPU buffer (spill/hydrate moves
            // codes between them); `f32_data` is head-major, reorder to
            // `[B, new_seq, kv_h, D]` and pass the matching shape.
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let new_seq = new_shape[2] as usize;
            let d = new_shape[3] as usize;
            let seq_major = super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, d);
            let seq_shape = [new_shape[0], new_shape[2], new_shape[1], new_shape[3]];
            let block = turbo_quantize_v(&seq_major, self.bits, &seq_shape)?;
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
                // Flat buffer is sequence-major (see `append`): dequant into
                // `[B, S, kv_h, D]`, then reorder heads↔seq back to the logical
                // `[B, kv_h, S, D]`. `contiguous` after the output transpose so
                // raw byte-readers (SSD spill) see the permuted bytes.
                let seq_major_shape = [self.shape[0], self.shape[2], self.shape[1], self.shape[3]];
                let out =
                    turbo_dequantize_v4_gpu(&codes, &scales, &seq_major_shape, out_dtype, device)?;
                let out = out.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
                return Ok((Vec::new(), Some(out)));
            }
            return Ok((Vec::new(), None));
        }
        // CPU path. Blocks are sequence-major (see `append`); reorder the
        // concatenated decode back to the logical `[B, kv_h, S, D]`.
        let total: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out = Vec::with_capacity(total);
        for block in &self.blocks {
            let slice = turbo_dequantize(block)?;
            out.extend_from_slice(&slice);
        }
        // Blocks must cover `shape[2]` exactly. Silently cutting an over-run
        // back kept the rejected prefix of a speculative partial accept and
        // dropped the appended correction; silently zero-padding a shortfall
        // fabricates a gap. See `super::QuantV::dequantize_choice`.
        if out.len() != total {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantKTurbo4::dequantize_choice: CPU blocks decode to {} elems but shape \
                 {:?} implies {total} — refusing to zero-pad / truncate",
                out.len(),
                self.shape,
            )));
        }
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let d = self.shape[3] as usize;
        let out = super::seq_layout::transpose_seq_heads(&out, b, s, kv_h, d);
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

#[cfg(test)]
#[path = "quant_k_turbo4_tests.rs"]
mod quant_k_turbo4_tests;
