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
//! Quantized K buffer: `QuantK` (q8_0).
// unsafe_code: f32 slice reinterpret for hydration upload
#![allow(unsafe_code)]
#![allow(clippy::too_many_lines)]

use rmlx_core::error::Result;
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::q8_msl::{q8_dequantize_gpu, q8_quantize_gpu};

use super::KV_PAGE_SIZE;
use crate::q8::{q8_dequantize, q8_quantize, Q8_GROUP_SIZE};

// ── Quantized K storage ───────────────────────────────────────────────────────

/// Accumulated q8_0 K cache (group_size=128).
///
/// Supports two backends:
/// - **CPU**: scalar Rust `q8_quantize` / `q8_dequantize`, storing
///   `Vec<u8>` codes and `Vec<f32>` scales.
/// - **GPU**: MSL kernel `q8_quantize_gpu` / `q8_dequantize_gpu`. Codes and
///   scales accumulate into pre-allocated 1D `Array`s sized to the model's
///   `max_seq`. Each step quantizes the new chunk and `slice_update`s it
///   into the buffer at offset `prev_words..new_words`. This avoids the
///   O(n²) lazy-concat tree that builds up if every step appends with
///   `concatenate`.
///
/// The backend is chosen on the first `append` call based on the `Device`.
/// Subsequent calls must use the same device.
pub struct QuantK {
    // ── CPU path ────────────────────────────────────────────────────────────
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    // ── GPU path ────────────────────────────────────────────────────────────
    /// Pre-allocated u32 codes buffer, flat 1D, length =
    /// `B * kv_h * max_seq * D / 4`. Each step writes a contiguous slice.
    pub gpu_codes_buf: Option<Array>,
    /// Pre-allocated f32 scales buffer, flat 1D, length =
    /// `B * kv_h * max_seq * D / Q8_GROUP_SIZE`.
    pub gpu_scales_buf: Option<Array>,
    /// Number of u32 codes written per single-step (B * kv_h * 1 * D / 4).
    pub gpu_words_per_step: i32,
    /// Number of f32 scales written per single-step (B * kv_h * 1 * D / Q8_GROUP_SIZE).
    pub gpu_scales_per_step: i32,
    /// Current allocated capacity in tokens (paged growth).
    /// Grows in KV_PAGE_SIZE increments up to max_seq.
    pub gpu_capacity: i32,
    // ── Shared ──────────────────────────────────────────────────────────────
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the GPU buffer was sized for (set on first GPU append).
    pub max_seq: i32,
}

use super::seq_layout::{transpose_heads_seq, transpose_seq_heads};

impl QuantK {
    /// Resident bytes held by this store: CPU codes/scales plus the GPU mirror.
    ///
    /// Both are summed unconditionally — an SSD-hydrate init leaves the
    /// pre-hydration CPU `codes`/`scales` resident under a live GPU mirror (the
    /// hydrate upload path never clears them), so both are real memory at once.
    /// Unallocated buffers contribute 0 on their own.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            codes,
            scales,
            gpu_codes_buf,
            gpu_scales_buf,
            // Geometry / bookkeeping about the buffers above, not allocations.
            gpu_words_per_step: _,
            gpu_scales_per_step: _,
            gpu_capacity: _,
            shape: _,
            max_seq: _,
        } = self;
        crate::bytes::vec_bytes(codes)
            + crate::bytes::vec_bytes(scales)
            + crate::bytes::opt_array_bytes(gpu_codes_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_scales_buf.as_ref())
    }

    /// Truncate the store to `n` sequence positions.
    ///
    /// Cuts the CPU-side `codes` / `scales` back to the leading `n` positions
    /// and lowers `shape[2]` to `n`. The GPU buffers need no cut:
    /// `dequantize_choice` slices `[0, shape[2])` and the next `append` writes
    /// at `prev_seq == shape[2]`, so lowering `shape[2]` already makes the
    /// rejected region overwritable. The CPU buffers are append-only, so
    /// without this the next `append` stacks on top of the rejected tokens and
    /// the dequant reads the rejected prefix back instead — wrong attention
    /// with no error, which is exactly what the coverage check in
    /// `dequantize_choice` now refuses to produce.
    ///
    /// The target is clamped to the store's current `shape[2]` — see
    /// [`super::clamp_truncate_target`] for why that is not defensive padding
    /// but the only correct reading for a ring-less store.
    pub fn truncate_to(&mut self, n: i32) {
        let n = super::clamp_truncate_target(&self.shape, n);
        self.retain_cpu_prefix(n);
        // `get_mut` rather than `shape[2]`: the store shape is rank-4 by
        // construction, and this is the bounds proof rather than a claim.
        if let Some(seq) = self.shape.get_mut(2) {
            *seq = n;
        }
    }

    /// Keep the leading `n` sequence positions of the CPU `codes` / `scales`.
    ///
    /// The buffers are sequence-major (`[B, S, kv_h, D]`, see [`Self::append`]),
    /// so at `b == 1` a sequence prefix is a byte prefix and this is a plain
    /// `Vec::truncate` — no allocation, no copy, which matters because it sits
    /// on the speculative rollback path.
    ///
    /// Two cuts are refused, and both leave the store deliberately over-covering
    /// so `dequantize_choice` aborts rather than return a plausible-looking
    /// wrong tensor:
    ///
    /// * `b > 1` — rows run batch-major, so a sequence prefix is not a
    ///   contiguous prefix and cutting anyway would scramble the store silently.
    /// * a target that lands inside a q8 group — one scale covers
    ///   `Q8_GROUP_SIZE` elements, so a cut that is not a whole number of groups
    ///   cannot be expressed without re-quantizing, and the f32 source is gone
    ///   by now.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape rank is checked to be >= 4 before any element is read"
    )]
    fn retain_cpu_prefix(&mut self, n: i32) {
        if self.codes.is_empty() || self.shape.len() < 4 {
            return;
        }
        let b = self.shape[0].max(0) as usize;
        let kv_h = self.shape[1].max(0) as usize;
        let d = self.shape[3].max(0) as usize;
        let keep_elems = (n.max(0) as usize)
            .saturating_mul(b)
            .saturating_mul(kv_h)
            .saturating_mul(d);
        if keep_elems >= self.codes.len() {
            // Nothing accumulated past the target — includes the whole-store
            // and hydrated-prefix cases, so neither warns.
            return;
        }
        if b != 1 {
            tracing::warn!(
                b,
                kv_h,
                target_seq = n,
                "QuantK truncate: refusing to cut CPU codes at b > 1 — the buffer runs \
                 batch-major, so a sequence prefix is not a contiguous prefix and the cut \
                 would be silently scrambled; leaving the store over-covering, which the \
                 next dequant rejects"
            );
            return;
        }
        if !keep_elems.is_multiple_of(Q8_GROUP_SIZE) {
            tracing::warn!(
                keep_elems,
                group_size = Q8_GROUP_SIZE,
                target_seq = n,
                "QuantK truncate: refusing to cut CPU codes inside a q8 group — one scale \
                 covers the whole group and the f32 source is gone; leaving the store \
                 over-covering, which the next dequant rejects"
            );
            return;
        }
        self.codes.truncate(keep_elems);
        self.scales.truncate(keep_elems / Q8_GROUP_SIZE);
    }

    /// Append a new K slice. Picks CPU or GPU path from `device`.
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

            // ── First call: allocate initial page ────────────────────────────
            if self.gpu_codes_buf.is_none() {
                let words_per_step = b * kv_h * d / 4;
                let scales_per_step = b * kv_h * d / Q8_GROUP_SIZE;
                self.gpu_words_per_step = words_per_step as i32;
                self.gpu_scales_per_step = scales_per_step as i32;
                self.max_seq = max_seq;
                // Paged growth: start with one page, not max_seq.
                // Hydration note: if prev_seq > 0, the layer was reconstructed
                // from CPU-side codes (SSD hydration). The initial buffer must
                // cover prev_seq + new_seq so the grow path can copy prev_seq
                // words from an old_codes buffer that actually has that many
                // elements. Round up to page boundary, cap at max_seq.
                let min_cap = if prev_seq > 0 {
                    let pages = (prev_seq + new_shape[2] + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
                    (pages * KV_PAGE_SIZE).min(max_seq)
                } else {
                    KV_PAGE_SIZE.min(max_seq)
                };
                let init_cap = min_cap;
                self.gpu_capacity = init_cap;
                let total_words = words_per_step * init_cap as usize;
                let total_scales = scales_per_step * init_cap as usize;
                let codes_buf = zeros(&[total_words as i32], Dtype::U32, device)?;
                let scales_buf = zeros(&[total_scales as i32], Dtype::F32, device)?;
                // Upload existing CPU codes when initialising GPU buffers for a
                // hydrated layer (prev_seq > 0 and CPU codes are populated).
                //
                // Layout note: CPU `self.codes` is `Vec<u8>` (1 byte per q8
                // quantized element, 4 bytes per u32 GPU word). CPU `self.scales`
                // is `Vec<f32>` (4 bytes per scale). To create GPU arrays:
                // - codes: pass `self.codes` bytes directly, shape=[cpu_words]
                // where cpu_words = codes.len() / 4 (4 u8 per u32 word).
                // - scales: reinterpret f32 Vec as bytes, shape=[cpu_scales].
                // H7: emit tracing event when uploading CPU codes to GPU on hydration.
                let (codes_buf, scales_buf) = if !self.codes.is_empty() && prev_seq > 0 {
                    let cpu_words = self.codes.len() / 4; // u8 codes → u32 words
                    let cpu_scales = self.scales.len();
                    // SAFETY: `self.scales` is `Vec<f32>`; `f32` is `Copy` with a
                    // fixed 4-byte LE layout. Reinterpreting as `&[u8]` is safe
                    // because `f32` and `u8` have no alignment or validity
                    // requirements beyond what `Vec` guarantees.
                    let scales_bytes = unsafe {
                        std::slice::from_raw_parts(
                            self.scales.as_ptr().cast::<u8>(),
                            self.scales.len() * 4,
                        )
                    };
                    let cpu_codes_arr =
                        Array::from_bytes(&self.codes, &[cpu_words as i32], Dtype::U32)?;
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
                        "QuantK hydrated init: uploaded CPU codes → GPU"
                    );
                    (codes_buf, scales_buf)
                } else {
                    (codes_buf, scales_buf)
                };
                self.gpu_codes_buf = Some(codes_buf);
                self.gpu_scales_buf = Some(scales_buf);
            }

            // ── Grow if the new tokens would exceed current capacity ──────────
            let needed = self.shape[2]; // already incremented above
            if needed > self.gpu_capacity {
                let new_cap = {
                    let pages = (needed + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
                    (pages * KV_PAGE_SIZE).min(max_seq)
                };
                let words_per_step = self.gpu_words_per_step as usize;
                let scales_per_step = self.gpu_scales_per_step as usize;
                let new_total_words = (words_per_step * new_cap as usize) as i32;
                let new_total_scales = (scales_per_step * new_cap as usize) as i32;
                // Allocate larger buffer and copy the filled prefix from old.
                let new_codes = zeros(&[new_total_words], Dtype::U32, device)?;
                let new_scales = zeros(&[new_total_scales], Dtype::F32, device)?;
                let old_codes = self.gpu_codes_buf.take().unwrap();
                let old_scales = self.gpu_scales_buf.take().unwrap();
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

            let words_per_seq = self.gpu_words_per_step;
            let scales_per_seq = self.gpu_scales_per_step;
            let new_seq = new_shape[2];

            // Reorder the chunk to sequence-major `[B, new_seq, kv_h, D]` before
            // quantizing. The flat GPU buffer accumulates chunks back-to-back at
            // `prev_seq * words_per_seq`, so the active prefix is sequence-major
            // (`[B, S, kv_h, D]` ordering): for each token, all heads are
            // contiguous. `dequantize_choice` reads it back with the matching
            // `[B, S, kv_h, D]` reshape + transpose. The incoming chunk is
            // head-major (`[B, kv_h, new_seq, D]`); transposing heads↔seq makes
            // the stored layout self-consistent across any number of appends and
            // any `kv_h`. With a single chunk (`prev_seq == 0`, `new_seq == S`)
            // this is the identity for the dequant view, so the cold-prefill
            // path stays logically correct. It is byte-identical to the pre-fix
            // head-major grouping ONLY when `D % 128 == 0` (each q8 group of 128
            // stays inside one head — e.g. linear head_dim=128); for `D` not a
            // multiple of 128 (e.g. head_dim=64) a group spans a (head,token)
            // boundary, so per-group `abs_max` scales differ from the head-major
            // grouping — logically correct and within q8 noise, but not bit-exact.
            // `transpose` yields a strided view. `q8_quantize_gpu` reads its
            // input by raw linear offset (the MSL kernel ignores MLX strides),
            // so materialize the heads↔seq permutation into a row-major buffer
            // here — otherwise the kernel would read the original head-major
            // bytes and scramble the stored codes.
            let k_seq_major = k_arr.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
            let (new_codes, new_scales) = q8_quantize_gpu(&k_seq_major, device)?;

            // Compute offsets in the 1D buffer.
            let codes_start = prev_seq * words_per_seq;
            let codes_stop = (prev_seq + new_seq) * words_per_seq;
            let scales_start = prev_seq * scales_per_seq;
            let scales_stop = (prev_seq + new_seq) * scales_per_seq;

            let codes_buf = self.gpu_codes_buf.take().unwrap();
            let scales_buf = self.gpu_scales_buf.take().unwrap();

            let codes_updated =
                codes_buf.slice_update(&new_codes, &[codes_start], &[codes_stop], &[1], device)?;
            let scales_updated = scales_buf.slice_update(
                &new_scales,
                &[scales_start],
                &[scales_stop],
                &[1],
                device,
            )?;

            self.gpu_codes_buf = Some(codes_updated);
            self.gpu_scales_buf = Some(scales_updated);
        } else {
            // CPU mirror of the GPU path: store the chunk sequence-major so the
            // accumulated `self.codes` share one layout with the GPU buffer (the
            // spill/hydrate round-trip moves codes between the two). `f32_data`
            // arrives head-major (`[B, kv_h, new_seq, D]`); reorder it to
            // `[B, new_seq, kv_h, D]` before quantizing.
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let new_seq = new_shape[2] as usize;
            let d = new_shape[3] as usize;
            let seq_major = transpose_heads_seq(f32_data, b, kv_h, new_seq, d);
            let (new_codes, new_scales) = q8_quantize(&seq_major);
            self.codes.extend_from_slice(&new_codes);
            self.scales.extend_from_slice(&new_scales);
        }
        Ok(())
    }

    /// Dequantize all accumulated K. Returns `(flat_f32, None)` for CPU or
    /// `(empty, Some(Array))` for GPU.
    ///
    /// `out_dtype` is the target dtype for the GPU output array. If `Bf16`,
    /// the f32 dequant result is cast to bf16 on-device so downstream SDPA
    /// can use the bf16-native fast kernel.
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
                // Slice the active prefix.
                let s = self.shape[2];
                let codes = codes_buf.slice(&[0], &[s * self.gpu_words_per_step], &[1], device)?;
                let scales =
                    scales_buf.slice(&[0], &[s * self.gpu_scales_per_step], &[1], device)?;
                // The flat buffer is sequence-major (see `append`): the active
                // prefix lays out as `[B, S, kv_h, D]`. Dequant into that shape,
                // then transpose heads↔seq back to the logical `[B, kv_h, S, D]`
                // that callers expect. For a single-chunk cache this transpose is
                // the inverse of the append-side transpose, so the logical
                // round-trip is exact (within q8 noise). It reproduces the pre-fix
                // head-major output bit-for-bit ONLY when `D % 128 == 0`; for `D`
                // not a multiple of 128 the q8 grouping differs (see `append`), so
                // a debugger comparing decode logits against the base commit will
                // see q8-noise-level deltas, not zero. Pass out_dtype directly:
                // kernel writes bf16/f16/f32 — no follow-up astype.
                let seq_major_shape = [self.shape[0], s, self.shape[1], self.shape[3]];
                let out = q8_dequantize_gpu(&codes, &scales, &seq_major_shape, out_dtype, device)?;
                // Materialize the heads↔seq transpose into a row-major buffer.
                // `transpose` alone yields a strided view: stride-honoring MLX ops
                // (SDPA) read it correctly, but a raw byte read (`to_bytes`, SSD
                // spill, a custom kernel) sees the un-permuted sequence-major
                // bytes and scrambles heads↔seq. Forcing contiguity makes the
                // physical layout match the logical `[B, kv_h, S, D]` shape for
                // every consumer.
                // Required for the raw byte-readers (SSD spill / hydrate
                // round-trip) — this is the post-hydrate dequant path, off the
                // steady-state decode hot path, so the copy is acceptable.
                let out = out.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
                return Ok((Vec::new(), Some(out)));
            }
            return Ok((Vec::new(), None));
        }
        // CPU codes are sequence-major (`[B, S, kv_h, D]`, see `append`). The
        // caller reshapes the returned flat vector to the logical
        // `[B, kv_h, S, D]`, so reorder heads↔seq back to head-major first.
        let flat = q8_dequantize(&self.codes, &self.scales);
        // The codes must cover `shape[2]` exactly. `transpose_seq_heads` reads
        // only the first `b * s * kv_h * d` elements, so an over-run was
        // silently cut back — after a mid-sequence truncate that returned the
        // *rejected* speculative prefix and dropped the appended correction,
        // with no error anywhere — and a shortfall panicked on an out-of-range
        // index. `truncate_to` now cuts the codes; when it cannot (see
        // `Self::retain_cpu_prefix`) it leaves the store over-covering on
        // purpose, and that must abort the request here.
        let total: usize = self.shape.iter().map(|&d| d.max(0) as usize).product();
        if flat.len() != total {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantK::dequantize_choice: CPU codes decode to {} elems but shape {:?} \
                 implies {total} — refusing to zero-pad / truncate",
                flat.len(),
                self.shape,
            )));
        }
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let d = self.shape[3] as usize;
        Ok((transpose_seq_heads(&flat, b, s, kv_h, d), None))
    }

    /// Reconstruct a CPU-path `QuantK` from serialized codes / scales and the
    /// accumulated logical shape. GPU buffers stay
    /// empty; the dequant path uses the CPU codes/scales.
    pub fn from_cpu_parts(codes: Vec<u8>, scales: Vec<f32>, shape: Vec<i32>) -> Self {
        Self {
            codes,
            scales,
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape,
            max_seq: 0,
        }
    }

    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            codes: self.codes.clone(),
            scales: self.scales.clone(),
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
            max_seq: self.max_seq,
        })
    }
}

#[cfg(test)]
#[path = "quant_k_tests.rs"]
mod quant_k_tests;
