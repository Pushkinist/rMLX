// QuantPlanarK — K-axis PlanarQuant 4-bit storage.
// Mirrors QuantPlanarV layout (identical GPU buffers + CPU PlanarBlocks);
// the only difference is which axis (K vs V) of the KV cache it backs.
#![allow(
    missing_docs,
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
//! Quantized K buffer: `QuantPlanarK` (PlanarQuant 4-bit on the K axis).
//!
//! The MSL kernel is **shared** with the V-side codec
//! ([`planar_quantize_v4_gpu`](crate::planarquant_msl::planar_quantize_v4_gpu)
//! / [`planar_dequantize_v4_gpu`](crate::planarquant_msl::planar_dequantize_v4_gpu))
//! — PlanarQuant operates on flat `[B, kv_h, S, D]` and is axis-agnostic; only
//! the dispatch side (K vs V) differs. See `docs/KV_QUANT.md` §PlanarK.
#![allow(clippy::too_many_lines)]

use rmlx_core::error::Result;
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::planarquant_msl::{planar_dequantize_v4_gpu, planar_quantize_v4_gpu};

use crate::planarquant::{planar_dequantize, planar_quantize, PlanarBlocks};
use crate::turboquant::GROUP_SIZE;

use super::KV_PAGE_SIZE;

// ── PlanarQuant K storage ─────────────────────────────────────────────────────

/// Accumulated PlanarQuant 4-bit cache for the **K axis**.
///
/// Bit-identical buffer layout to [`super::QuantPlanarV`] — PlanarQuant is
/// axis-agnostic at the kernel input
/// level (flat `[B, kv_h, S, D]` with `D % 32 == 0`), so the same MSL kernel
/// services both K and V. This struct exists separately so the storage enum
/// can carry independent K/V buffers without conflating semantics, but the
/// fields, growth policy, and dequant path mirror `QuantPlanarV` exactly.
#[derive(Debug)]
pub struct QuantPlanarK {
    // ── CPU path ────────────────────────────────────────────────────────────
    pub blocks: Vec<PlanarBlocks>,
    // ── GPU path (pre-allocated buffers) ─────────────────────────────────────
    pub gpu_codes_buf: Option<Array>,
    pub gpu_scales_buf: Option<Array>,
    pub gpu_rotations_buf: Option<Array>,
    pub gpu_codes_words_per_step: i32,
    pub gpu_scales_per_step: i32,
    pub gpu_rotations_words_per_step: i32,
    pub gpu_capacity: i32,
    // ── Shared ──────────────────────────────────────────────────────────────
    pub shape: Vec<i32>,
    pub max_seq: i32,
}

impl QuantPlanarK {
    /// Create an empty `QuantPlanarK` with the accumulated shape initialised
    /// to `init_shape` (seq dimension = 0) and GPU buffers unallocated.
    /// Mirrors the `from_cpu_blocks` pattern but sets `max_seq` for later
    /// GPU-path init, matching the `QuantKTurbo4` inline-literal signature.
    pub fn new(init_shape: Vec<i32>, max_seq: i32) -> Self {
        Self {
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
        }
    }

    /// Append a new K slice. CPU path uses scalar Rust; GPU path uses the
    /// shared MSL kernel + pre-allocated 1D buffers with `slice_update`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
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
            let total_per_step = b * kv_h * d;

            if self.gpu_codes_buf.is_none() {
                let codes_words_per_step = total_per_step * 4 / GROUP_SIZE;
                let scales_per_step = total_per_step / 2;
                let rotations_words_per_step = total_per_step * 2 / GROUP_SIZE;
                self.gpu_codes_words_per_step = codes_words_per_step as i32;
                self.gpu_scales_per_step = scales_per_step as i32;
                self.gpu_rotations_words_per_step = rotations_words_per_step as i32;
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
                    &[(codes_words_per_step * init_cap as usize) as i32],
                    Dtype::U32,
                    device,
                )?;
                let scales_buf = zeros(
                    &[(scales_per_step * init_cap as usize) as i32],
                    Dtype::F32,
                    device,
                )?;
                let rot_buf = zeros(
                    &[(rotations_words_per_step * init_cap as usize) as i32],
                    Dtype::U32,
                    device,
                )?;
                // Hydrated init: upload CPU PlanarBlocks → GPU when blocks present.
                let (codes_buf, scales_buf, rot_buf) = if !self.blocks.is_empty() && prev_seq > 0 {
                    let mut flat_codes: Vec<u8> = Vec::new();
                    let mut flat_scales: Vec<f32> = Vec::new();
                    let mut flat_rot: Vec<u8> = Vec::new();
                    for blk in &self.blocks {
                        flat_codes.extend_from_slice(&blk.codes);
                        flat_scales.extend_from_slice(&blk.scales);
                        flat_rot.extend_from_slice(&blk.rotations);
                    }
                    let cpu_codes_words = flat_codes.len() / 4;
                    let cpu_scales = flat_scales.len();
                    let cpu_rot_words = flat_rot.len() / 4;
                    // SAFETY: `flat_scales` is `Vec<f32>`; `f32` is `Copy` with a
                    // fixed 4-byte LE layout. Reinterpreting as `&[u8]` is safe
                    // because `f32` and `u8` have no alignment requirements
                    // beyond what `Vec` guarantees.
                    let scales_bytes = unsafe {
                        std::slice::from_raw_parts(
                            flat_scales.as_ptr().cast::<u8>(),
                            flat_scales.len() * 4,
                        )
                    };
                    let cpu_codes_arr =
                        Array::from_bytes(&flat_codes, &[cpu_codes_words as i32], Dtype::U32)?;
                    let cpu_scales_arr =
                        Array::from_bytes(scales_bytes, &[cpu_scales as i32], Dtype::F32)?;
                    let cpu_rot_arr =
                        Array::from_bytes(&flat_rot, &[cpu_rot_words as i32], Dtype::U32)?;
                    let codes_buf = codes_buf.slice_update(
                        &cpu_codes_arr,
                        &[0],
                        &[cpu_codes_words as i32],
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
                    let rot_buf = rot_buf.slice_update(
                        &cpu_rot_arr,
                        &[0],
                        &[cpu_rot_words as i32],
                        &[1],
                        device,
                    )?;
                    tracing::debug!(
                        prev_seq,
                        init_cap,
                        cpu_codes_words,
                        cpu_scales,
                        cpu_rot_words,
                        "QuantPlanarK hydrated init: uploaded CPU PlanarBlocks → GPU"
                    );
                    (codes_buf, scales_buf, rot_buf)
                } else {
                    (codes_buf, scales_buf, rot_buf)
                };
                self.gpu_codes_buf = Some(codes_buf);
                self.gpu_scales_buf = Some(scales_buf);
                self.gpu_rotations_buf = Some(rot_buf);
            }

            // ── Grow if needed ───────────────────────────────────────────────
            let needed = self.shape[2];
            if needed > self.gpu_capacity {
                let new_cap = {
                    let pages = (needed + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
                    (pages * KV_PAGE_SIZE).min(max_seq)
                };
                let cw = self.gpu_codes_words_per_step as usize;
                let sp = self.gpu_scales_per_step as usize;
                let rw = self.gpu_rotations_words_per_step as usize;
                let new_codes = zeros(&[(cw * new_cap as usize) as i32], Dtype::U32, device)?;
                let new_scales = zeros(&[(sp * new_cap as usize) as i32], Dtype::F32, device)?;
                let new_rotations = zeros(&[(rw * new_cap as usize) as i32], Dtype::U32, device)?;
                let old_codes = self.gpu_codes_buf.take().unwrap();
                let old_scales = self.gpu_scales_buf.take().unwrap();
                let old_rotations = self.gpu_rotations_buf.take().unwrap();
                let filled_codes = prev_seq * self.gpu_codes_words_per_step;
                let filled_scales = prev_seq * self.gpu_scales_per_step;
                let filled_rotations = prev_seq * self.gpu_rotations_words_per_step;
                let new_codes = if filled_codes > 0 {
                    new_codes.slice_update(
                        &old_codes.slice(&[0], &[filled_codes], &[1], device)?,
                        &[0],
                        &[filled_codes],
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
                let new_rotations = if filled_rotations > 0 {
                    new_rotations.slice_update(
                        &old_rotations.slice(&[0], &[filled_rotations], &[1], device)?,
                        &[0],
                        &[filled_rotations],
                        &[1],
                        device,
                    )?
                } else {
                    new_rotations
                };
                self.gpu_codes_buf = Some(new_codes);
                self.gpu_scales_buf = Some(new_scales);
                self.gpu_rotations_buf = Some(new_rotations);
                self.gpu_capacity = new_cap;
            }

            let new_seq = new_shape[2];
            let cw = self.gpu_codes_words_per_step;
            let sp = self.gpu_scales_per_step;
            let rw = self.gpu_rotations_words_per_step;

            // Shared kernel (axis-agnostic — see Step 0 decision).
            let (new_codes, new_scales, new_rotations) = planar_quantize_v4_gpu(k_arr, device)?;

            let codes_buf = self.gpu_codes_buf.take().unwrap();
            let scales_buf = self.gpu_scales_buf.take().unwrap();
            let rotations_buf = self.gpu_rotations_buf.take().unwrap();

            self.gpu_codes_buf = Some(codes_buf.slice_update(
                &new_codes,
                &[prev_seq * cw],
                &[(prev_seq + new_seq) * cw],
                &[1],
                device,
            )?);
            self.gpu_scales_buf = Some(scales_buf.slice_update(
                &new_scales,
                &[prev_seq * sp],
                &[(prev_seq + new_seq) * sp],
                &[1],
                device,
            )?);
            self.gpu_rotations_buf = Some(rotations_buf.slice_update(
                &new_rotations,
                &[prev_seq * rw],
                &[(prev_seq + new_seq) * rw],
                &[1],
                device,
            )?);
        } else {
            // CPU path: scalar Rust PlanarQuant (axis-agnostic — same function).
            let block = planar_quantize(f32_data, GROUP_SIZE, 4, new_shape)?;
            self.blocks.push(block);
        }
        Ok(())
    }

    /// Dequantize all accumulated K slices.
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
            if let (Some(codes_buf), Some(scales_buf), Some(rotations_buf)) = (
                &self.gpu_codes_buf,
                &self.gpu_scales_buf,
                &self.gpu_rotations_buf,
            ) {
                let s = self.shape[2];
                let codes =
                    codes_buf.slice(&[0], &[s * self.gpu_codes_words_per_step], &[1], device)?;
                let scales =
                    scales_buf.slice(&[0], &[s * self.gpu_scales_per_step], &[1], device)?;
                let rotations = rotations_buf.slice(
                    &[0],
                    &[s * self.gpu_rotations_words_per_step],
                    &[1],
                    device,
                )?;
                let out =
                    planar_dequantize_v4_gpu(&codes, &scales, &rotations, &self.shape, device)?;
                let out = if out_dtype == Dtype::F32 {
                    out
                } else {
                    out.astype(out_dtype, device)?
                };
                return Ok((Vec::new(), Some(out)));
            }
            return Ok((Vec::new(), None));
        }
        // CPU path.
        let total: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out = Vec::with_capacity(total);
        for block in &self.blocks {
            let slice = planar_dequantize(block)?;
            out.extend_from_slice(&slice);
        }
        Ok((out, None))
    }

    /// Return the GPU-resident packed K buffers sliced to the accumulated
    /// `S` tokens, **without dequantizing**.  This is the input contract for
    /// the fused-QK MSL kernel.
    ///
    /// Returns `Ok(None)` when GPU buffers are not present (CPU-only path or
    /// uninitialised cache).
    #[allow(
        clippy::indexing_slicing,
        reason = "shape[2] is the S dimension by construction; `init_shape` is built with rank==4"
    )]
    pub fn gpu_packed_view(&self, device: Device) -> Result<Option<(Array, Array, Array)>> {
        if device != Device::Gpu {
            return Ok(None);
        }
        let (Some(codes_buf), Some(scales_buf), Some(rotations_buf)) = (
            self.gpu_codes_buf.as_ref(),
            self.gpu_scales_buf.as_ref(),
            self.gpu_rotations_buf.as_ref(),
        ) else {
            return Ok(None);
        };
        let s = self.shape[2];
        let codes = codes_buf.slice(&[0], &[s * self.gpu_codes_words_per_step], &[1], device)?;
        let scales = scales_buf.slice(&[0], &[s * self.gpu_scales_per_step], &[1], device)?;
        let rotations =
            rotations_buf.slice(&[0], &[s * self.gpu_rotations_words_per_step], &[1], device)?;
        Ok(Some((codes, scales, rotations)))
    }

    /// Reconstruct a CPU-path `QuantPlanarK` from serialized PlanarQuant blocks.
    /// GPU buffers stay empty.
    pub fn from_cpu_blocks(blocks: Vec<PlanarBlocks>, shape: Vec<i32>) -> Self {
        Self {
            blocks,
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_rotations_buf: None,
            gpu_codes_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_rotations_words_per_step: 0,
            gpu_capacity: 0,
            shape,
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
            gpu_rotations_buf: match &self.gpu_rotations_buf {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            gpu_codes_words_per_step: self.gpu_codes_words_per_step,
            gpu_scales_per_step: self.gpu_scales_per_step,
            gpu_rotations_words_per_step: self.gpu_rotations_words_per_step,
            gpu_capacity: self.gpu_capacity,
            shape: self.shape.clone(),
            max_seq: self.max_seq,
        })
    }
}
