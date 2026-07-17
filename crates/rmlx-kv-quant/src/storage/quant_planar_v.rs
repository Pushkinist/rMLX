// promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill) can still reach them across the crate
// boundary. Doc/visibility warnings on the promoted surface are silenced; the
// API is otherwise unchanged.
#![allow(
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
//! Quantized V buffer: `QuantPlanarV` (PlanarQuant 4-bit).
#![allow(clippy::too_many_lines)]

use rmlx_core::error::Result;
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::planarquant_msl::{
    planar_dequantize_v3_gpu, planar_dequantize_v4_gpu, planar_quantize_v3_gpu,
    planar_quantize_v4_gpu,
};

use crate::planarquant::{planar_dequantize, planar_quantize, PlanarBlocks};
use crate::turboquant::GROUP_SIZE;

use super::seq_layout::{transpose_heads_seq, transpose_seq_heads};
use super::KV_PAGE_SIZE;

// ── PlanarQuant V storage ─────────────────────────────────────────────────────

/// Accumulated PlanarQuant V cache (supports 3-bit and 4-bit).
///
/// Supports two backends:
/// - **CPU**: scalar Rust `planar_quantize` / `planar_dequantize`.
/// - **GPU**: MSL kernel `planar_quantize_v4_gpu` / `planar_dequantize_v4_gpu` (4-bit);
///   CPU dequant fallback for 3-bit until the 3-bit MSL kernel is dispatched.
///
/// Per-pair scales (one f32 per 2 elements, vs TurboQuant's per-block) give
/// lower reconstruction error. GPU kernels carry separate codes, scales, and
/// rotation arrays.
///
/// # Bits field
///
/// `bits ∈ {3, 4}`. The 4-bit path uses the existing MSL kernel
/// (`planar_quantize_v4_gpu`). The 3-bit path uses 10 vals/u32 packing
/// (ForgeAttention-compatible 3.25-bit variant) and dispatches through the
/// `planar_quantize_v3_gpu` / `planar_dequantize_v3_gpu` MSL kernels.
pub struct QuantPlanarV {
    // ── CPU path ────────────────────────────────────────────────────────────
    pub blocks: Vec<PlanarBlocks>,
    // ── GPU path (pre-allocated buffers) ─────────────────────────────────────
    /// u32 codes buffer (4 words per group of 32 elements, for both 3-bit and 4-bit).
    ///
    /// 4-bit: 8 vals/word × 4 words = 32 vals. 3-bit: 10 vals/word × 4 words = 40
    /// capacity, 32 used (2 wasted slots per u32 column). Same word count — no
    /// buffer-size difference between 3-bit and 4-bit. Enables fused-QK kernel reuse.
    pub gpu_codes_buf: Option<Array>,
    /// f32 scales buffer (one per pair = 16 per group).
    pub gpu_scales_buf: Option<Array>,
    /// u32 rotations buffer (2 words per group, 8 4-bit rotations per word).
    pub gpu_rotations_buf: Option<Array>,
    /// Per-step counts for slice_update offsets.
    pub gpu_codes_words_per_step: i32,
    pub gpu_scales_per_step: i32,
    pub gpu_rotations_words_per_step: i32,
    /// Current allocated capacity in tokens (paged growth).
    pub gpu_capacity: i32,
    // ── Shared ──────────────────────────────────────────────────────────────
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the GPU buffer was sized for.
    pub max_seq: i32,
    /// Bit-width: 3 (Planar3) or 4 (Planar / legacy default).
    pub bits: u8,
}

impl QuantPlanarV {
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
            gpu_rotations_buf,
            // Geometry / bookkeeping about the buffers above, not allocations.
            gpu_codes_words_per_step: _,
            gpu_scales_per_step: _,
            gpu_rotations_words_per_step: _,
            gpu_capacity: _,
            shape: _,
            max_seq: _,
            bits: _,
        } = self;
        blocks.iter().map(PlanarBlocks::byte_size).sum::<u64>()
            + crate::bytes::opt_array_bytes(gpu_codes_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_scales_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_rotations_buf.as_ref())
    }

    /// Append a new V slice. CPU path uses scalar Rust; GPU path uses MSL kernel
    /// + pre-allocated 1D buffers with `slice_update`.
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
        v_arr: &Array,
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
                // PlanarQuant: 4 u32 codes per group of 32 (both 3-bit and 4-bit use the same
                // word count: 4-bit packs 8 vals/word, 3-bit packs 10 vals/word but
                // ceil(32/10)=4 words/group — same word count for 3-bit and 4-bit).
                // 16 f32 scales per group (one per pair), 2 u32 rotations per group.
                let codes_words_per_step = total_per_step * 4 / GROUP_SIZE;
                let scales_per_step = total_per_step / 2;
                let rotations_words_per_step = total_per_step * 2 / GROUP_SIZE;
                self.gpu_codes_words_per_step = codes_words_per_step as i32;
                self.gpu_scales_per_step = scales_per_step as i32;
                self.gpu_rotations_words_per_step = rotations_words_per_step as i32;
                self.max_seq = max_seq;
                // C2 fix: allocate enough for prev_seq + new chunk when hydrating
                // (prev_seq > 0). Without this the grow path tries to copy
                // prev_seq words from an old_codes buffer sized for only one page
                // (e.g. 256) while prev_seq may be larger (e.g. 300), producing
                // an OOB slice_update → broadcast error / panic.
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
                // Upload CPU PlanarBlocks when initialising a hydrated layer
                // (prev_seq > 0 and self.blocks non-empty).
                //
                // PlanarBlocks CPU layout: `codes: Vec<u8>` — 4 little-endian
                // u32 words (16 bytes) per group of 32 in the shared word
                // convention (`32 / bits` vals/u32), byte-identical to what the
                // GPU dequant kernel reads. This holds for BOTH 3-bit (10
                // vals/u32) and 4-bit (8 vals/u32); a dense 3-bit layout would
                // not round-trip here. `scales: Vec<f32>`, `rotations: Vec<u8>`
                // (4-bit rotation index per pair, 2 pairs per byte — also shared
                // with the GPU). Flatten all blocks, reinterpret as bytes,
                // upload via `Array::from_bytes` + `slice_update`.
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
                    // SAFETY: `flat_scales` is `Vec<f32>`; `f32` is `Copy`
                    // with a fixed 4-byte LE layout. Reinterpreting as `&[u8]`
                    // is safe because `f32` and `u8` have no alignment or
                    // validity requirements beyond what `Vec` guarantees.
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
                    // H7: tracing event for hydrated QuantPlanarV GPU init.
                    tracing::debug!(
                        prev_seq,
                        init_cap,
                        cpu_codes_words,
                        cpu_scales,
                        cpu_rot_words,
                        "QuantPlanarV hydrated init: uploaded CPU PlanarBlocks → GPU"
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
            // Note: PlanarQuant uses atomic OR for rotations — zero-init of the
            // fresh capacity region is guaranteed by `zeros()` on each realloc.
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

            // CRITICAL: planar_quantize_v*_gpu uses atomic OR — output zero-init is required.
            // We write to a slice of a pre-zeroed buffer; the buffer was zero-initialized by `zeros`,
            // and slice_update on subsequent calls writes a fresh zero region only on first use.
            // Since we never re-write the same offset within a sequence, atomic OR over the same
            // bits stays correct.

            // Reorder the chunk to sequence-major `[B, new_seq, kv_h, D]` before
            // quantizing — the flat buffer accumulates chunks at a per-token
            // (all-heads) offset (`prev_seq * words_per_seq`), so the prefix is
            // sequence-major. `dequantize_choice` reads it back with the matching
            // reshape + transpose. For a single token the transpose is identity
            // (decode hot path byte-unchanged); for a single chunk it cancels
            // with the dequant transpose (cold-prefill round-trip exact). The
            // MSL kernel reads by raw linear offset, so materialize the strided
            // transpose into a row-major buffer before dispatch.
            let v_seq_major = v_arr.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
            let (new_codes, new_scales, new_rotations) = if self.bits == 3 {
                planar_quantize_v3_gpu(&v_seq_major, device)?
            } else {
                planar_quantize_v4_gpu(&v_seq_major, device)?
            };

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
            // CPU mirror of the GPU path: store sequence-major so the blocks
            // share one layout with the GPU buffer (spill/hydrate moves codes
            // between the two). `f32_data` arrives head-major
            // (`[B, kv_h, new_seq, D]`); reorder to `[B, new_seq, kv_h, D]` and
            // record that chunk shape so `planar_quantize`'s flat grouping
            // matches the reordered stream.
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let new_seq = new_shape[2] as usize;
            let d = new_shape[3] as usize;
            let seq_major = transpose_heads_seq(f32_data, b, kv_h, new_seq, d);
            let seq_major_shape = [new_shape[0], new_shape[2], new_shape[1], new_shape[3]];
            let block = planar_quantize(&seq_major, GROUP_SIZE, self.bits, &seq_major_shape)?;
            self.blocks.push(block);
        }
        Ok(())
    }

    /// Dequantize all accumulated V slices.
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
                // Flat buffer is sequence-major (see `append`): dequant into
                // `[B, S, kv_h, D]` then transpose heads↔seq back to the logical
                // `[B, kv_h, S, D]`. `contiguous` makes the physical layout match
                // for raw byte-readers (SSD spill / hydrate); this is the
                // post-hydrate / chunked-prefill dequant path, off the decode hot
                // path, so the copy is acceptable.
                let seq_major_shape = [self.shape[0], s, self.shape[1], self.shape[3]];
                let out = if self.bits == 3 {
                    planar_dequantize_v3_gpu(&codes, &scales, &rotations, &seq_major_shape, device)?
                } else {
                    planar_dequantize_v4_gpu(&codes, &scales, &rotations, &seq_major_shape, device)?
                };
                let out = out.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
                let out = if out_dtype == Dtype::F32 {
                    out
                } else {
                    out.astype(out_dtype, device)?
                };
                return Ok((Vec::new(), Some(out)));
            }
            return Ok((Vec::new(), None));
        }
        // CPU path: blocks are sequence-major (`[B, S, kv_h, D]`, see `append`).
        // The caller reshapes the returned flat vector to the logical
        // `[B, kv_h, S, D]`, so reorder heads↔seq back to head-major first.
        let total: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out = Vec::with_capacity(total);
        for block in &self.blocks {
            let slice = planar_dequantize(block)?;
            out.extend_from_slice(&slice);
        }
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let d = self.shape[3] as usize;
        Ok((transpose_seq_heads(&out, b, s, kv_h, d), None))
    }

    /// Reconstruct a CPU-path `QuantPlanarV` from serialized PlanarQuant blocks.
    /// GPU buffers stay empty. `bits` must be 3 or 4.
    pub fn from_cpu_blocks(blocks: Vec<PlanarBlocks>, shape: Vec<i32>, bits: u8) -> Self {
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
            bits,
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
            bits: self.bits,
        })
    }
}
