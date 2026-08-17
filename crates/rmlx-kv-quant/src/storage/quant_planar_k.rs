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

use super::seq_layout::{transpose_chunked_seq_heads, transpose_heads_seq};
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
        } = self;
        blocks.iter().map(PlanarBlocks::byte_size).sum::<u64>()
            + crate::bytes::opt_array_bytes(gpu_codes_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_scales_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_rotations_buf.as_ref())
    }

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

    /// Truncate the store to `n` sequence positions.
    ///
    /// Drops trailing CPU blocks past `n` and **splits** the block the cut lands
    /// inside (see [`super::truncate_plan`]), then lowers `shape[2]` to `n`. The
    /// GPU buffers need no cut — see [`super::QuantV::truncate_to`]. The target
    /// is clamped to the store's current `shape[2]`
    /// ([`super::clamp_truncate_target`]).
    pub fn truncate_to(&mut self, n: i32) {
        super::truncate_block_store(&mut self.blocks, &mut self.shape, n);
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

            // Reorder the chunk to sequence-major `[B, new_seq, kv_h, D]` before
            // quantizing. The flat GPU buffer accumulates chunks back-to-back at
            // `prev_seq * words_per_seq`, where `words_per_seq` counts ONE token
            // across all heads, so the active prefix is sequence-major
            // (`[B, S, kv_h, D]`): for each token, all heads are contiguous.
            // The incoming chunk is head-major (`[B, kv_h, new_seq, D]`);
            // transposing heads↔seq makes the stored layout self-consistent
            // across any number of appends and any `kv_h`. For a single token
            // (`new_seq == 1`) the transpose is the identity, so the decode hot
            // path is byte-unchanged; for a single chunk (`prev_seq == 0`) the
            // dequant transpose is its inverse, so the cold-prefill round-trip
            // is exact. PlanarQuant is layout-agnostic (it processes the flat
            // element stream group-by-group and `D % GROUP_SIZE == 0`, so no
            // group spans a (head, token) boundary), so the per-group scales are
            // identical to the head-major grouping — the reorder is bit-exact,
            // not just within-noise.
            //
            // `transpose` yields a strided view; the PlanarQuant MSL kernel reads
            // its input by raw linear offset (it ignores MLX lazy-transpose
            // strides), so materialize the heads↔seq permutation into a row-major
            // buffer here — otherwise the kernel would read the original
            // head-major bytes and scramble the stored codes.
            let k_seq_major = k_arr.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
            let (new_codes, new_scales, new_rotations) =
                planar_quantize_v4_gpu(&k_seq_major, device)?;

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
            // CPU mirror of the GPU path: store the chunk sequence-major so the
            // accumulated blocks share one layout with the GPU buffer (the
            // spill/hydrate round-trip moves codes between the two). `f32_data`
            // arrives head-major (`[B, kv_h, new_seq, D]`); reorder it to
            // `[B, new_seq, kv_h, D]` before quantizing, then record the chunk
            // shape as `[B, new_seq, kv_h, D]` so `planar_quantize`'s flat
            // grouping matches the reordered stream.
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let new_seq = new_shape[2] as usize;
            let d = new_shape[3] as usize;
            let seq_major = transpose_heads_seq(f32_data, b, kv_h, new_seq, d);
            let seq_major_shape = [new_shape[0], new_shape[2], new_shape[1], new_shape[3]];
            let block = planar_quantize(&seq_major, GROUP_SIZE, 4, &seq_major_shape)?;
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
                // The flat buffer is sequence-major (see `append`): the active
                // prefix lays out as `[B, S, kv_h, D]`. Dequant into that shape,
                // then transpose heads↔seq back to the logical `[B, kv_h, S, D]`
                // callers expect. For a single-chunk cache this transpose is the
                // inverse of the append-side transpose, so the round-trip is
                // exact. `transpose` alone yields a strided view; the raw
                // byte-readers (SSD spill / hydrate) need a row-major
                // `[B, kv_h, S, D]` buffer, so force contiguity. This is the
                // post-hydrate dequant path, off the steady-state decode hot
                // path, so the copy is acceptable.
                // The flat GPU buffer is a run of `[B, S_chunk, kv_h, D]`
                // chunks written at `prev_seq * words_per_step`, and
                // `words_per_step` folds `b` in. Reading the prefix as one
                // `[B, S_total, kv_h, D]` run therefore interleaves batch
                // elements once `B > 1`. Unlike the CPU half this arm has no
                // payload-vs-shape coverage check — the slice is sized *from*
                // `shape[2]` — so `S == 1` is not evidence that the prefix is a
                // single `[B, 1, kv_h, D]` chunk: a mid-chunk truncate at
                // `b > 1` lowers `shape[2]` without touching this buffer. Only
                // the empty store is exempt, because `truncate_to(0)` (which
                // `KvStorage::reset` routes through) must still decode to
                // nothing. Refuse
                // rather than return a scrambled tensor — the CPU half of this
                // reader handles every `B` via
                // `seq_layout::transpose_chunked_seq_heads`, which is what the
                // block list makes possible and this buffer does not.
                if self.shape[0] != 1 && self.shape[2] != 0 {
                    return Err(rmlx_core::error::Error::Quant(format!(
                        "QuantPlanarK::dequantize_choice: the flat GPU buffer is b == 1 only \
                         (its per-step stride does not interleave batch), got shape {:?}",
                        self.shape
                    )));
                }
                let seq_major_shape = [self.shape[0], s, self.shape[1], self.shape[3]];
                let out = planar_dequantize_v4_gpu(
                    &codes,
                    &scales,
                    &rotations,
                    &seq_major_shape,
                    device,
                )?;
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
        // Blocks must cover `shape[2]` exactly. `transpose_chunked_seq_heads`
        // rejects a buffer whose length disagrees with the declared shape in
        // either direction, but the check is kept here so the error names this
        // store and its block list rather than the reorder helper. Before both,
        // an over-run was silently cut back — after a mid-block truncate that
        // kept the *rejected* speculative prefix and dropped the appended
        // correction, with no error anywhere — and a shortfall panicked out of
        // range.
        // `truncate_to` now cuts the blocks; when it cannot (see
        // `super::truncate_plan`) it drops the trailing block whole and leaves
        // the store short on purpose, and that must abort the request here.
        if out.len() != total {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantPlanarK::dequantize_choice: CPU blocks decode to {} elems but shape \
                 {:?} implies {total} — refusing to zero-pad / truncate",
                out.len(),
                self.shape,
            )));
        }
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let d = self.shape[3] as usize;
        // Blocks are sequence-major (see `append`), one per append; reorder each
        // at its own sequence offset back to head-major `[B, kv_h, S, D]`.
        // Reading the concatenation as a single run would interleave batch
        // elements once `B > 1`.
        let out = transpose_chunked_seq_heads(
            &out,
            b,
            s,
            kv_h,
            d,
            self.blocks.iter().map(super::BlockRows::rows),
        )?;
        Ok((out, None))
    }

    /// Return the GPU-resident packed K buffers sliced to the accumulated
    /// `S` tokens, **without dequantizing**.  This is the input contract for
    /// the fused-QK / flash-decode MSL kernels.
    ///
    /// The packed prefix is **sequence-major** (`[B, S, kv_h, D]` element
    /// order — see `append`): per token, all heads are contiguous. The fused-QK
    /// / flash-decode / sparse-attn phase-1 kernels index it with the
    /// sequence-major token base `((b * S + s) * kv_h + h)`, NOT the head-major
    /// `((b * kv_h + h) * S + s)` base. Any new consumer must honour the same
    /// layout.
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
