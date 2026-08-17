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
//! Quantized V buffer: `QuantV` (TurboQuant 4-bit / q8_0).
#![allow(clippy::too_many_lines)]

use rmlx_core::error::Result;
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::turboquant_msl::{
    turbo_dequantize_v4_codebook_buf_gpu, turbo_dequantize_v4_gpu,
    turbo_quantize_v4_codebook_buf_gpu, turbo_quantize_v4_gpu,
};

use crate::tcq::{tcq_quantize_v2, tcq_quantize_v3, tcq_quantize_v3_with_codebook};
use crate::turboquant::{
    turbo_dequantize, turbo_dequantize_with_codebook, turbo_quantize_v_with_codebook, TurboBlocks,
    GROUP_SIZE,
};

use super::KV_PAGE_SIZE;

// ── Quantized V storage ───────────────────────────────────────────────────────

/// Accumulated TurboQuant V-cache at configurable bit-width (2-bit, 3-bit, or 4-bit Lloyd-Max).
///
/// Two backends:
/// - CPU scalar `turbo_quantize_v` / `turbo_dequantize`
/// - GPU MSL kernel (bits=4 only; dispatched from `QuantV::append`).
///
/// Backend chosen on first `append` call.
pub struct QuantV {
    // ── CPU path ────────────────────────────────────────────────────────────
    pub blocks: Vec<TurboBlocks>,
    // ── GPU path ────────────────────────────────────────────────────────────
    /// Pre-allocated codes buffer (u32, 4 words per group of GROUP_SIZE=32 elements).
    /// Length: `B * kv_h * max_seq * D / 8`.
    pub gpu_codes_buf: Option<Array>,
    /// Pre-allocated scales buffer (f32, one per group).
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
    pub bits: u8,
    /// Maximum sequence length the GPU buffer was sized for.
    pub max_seq: i32,
    // ── Calibration surface ───────────────────────────────────────────────────
    /// Per-KV-head sorted list of high-precision head-dimension indices for the
    /// V projection, sourced from `kv_calib.json` via `KvCacheBuilder`.
    ///
    /// `None` = no calibration attached; codec behavior is unchanged (default).
    /// `Some(indices)` = calibration present; outer vec length = `num_kv_heads`,
    /// inner vec = sorted high-precision indices for that head.
    ///
    /// Stored; not read by any codec yet. Wired by the TCQ / codebook-override
    /// encode paths.
    pub high_precision_indices: Option<Vec<Vec<u32>>>,
    // ── Codebook override surface ─────────────────────────────────────────────
    /// Per-layer V-side codebook override, sourced from `kv_calib.json`
    /// [`CodebookOverride::value`][rmlx_loader::CodebookOverride::value] via
    /// `KvCacheBuilder`.
    ///
    /// `None` = use the built-in Lloyd-Max N(0,1) codebook (default).
    /// `Some(cb)` = use these centroids for the CPU V-encode path. Length must
    /// equal `2^bits`; validated at encode time.
    ///
    /// Stored; CPU encode path reads it when `Some`. An MSL kernel variant
    /// (`turbo_quantize_v4_codebook_buf_gpu`) allows the GPU encode path to
    /// also honor the override at `bits == 4`. See [`Self::value_codebook_gpu`]
    /// for the upload cache.
    ///
    /// # Mutation contract
    ///
    /// Mutating this field after the first GPU encode is **not supported**.
    /// The GPU upload is cached lazily into [`Self::value_codebook_gpu`] on the
    /// first dispatch and reused thereafter; a later in-place swap of the CPU
    /// codebook would silently diverge from the GPU-side centroids until the
    /// cache is rebuilt. Treat this as an init-time field (set before the
    /// first `append`) or, if a layer must be re-keyed, also set
    /// [`Self::value_codebook_gpu`] back to `None` so the next encode re-uploads.
    /// The field stays `pub` to avoid a wide-fanout API break; dispatch sites
    /// `debug_assert!` the cache/source invariant.
    pub value_codebook: Option<Vec<f32>>,
    // ── Per-layer codebook GPU upload cache ──────────────────────────────────
    /// Lazy-uploaded GPU copy of [`Self::value_codebook`]. `None` until the
    /// first GPU `append_inner` call observes `value_codebook.is_some()`,
    /// at which point a `[16]` f32 Array is built once and reused across
    /// every subsequent encode + dequant on that layer.
    ///
    /// The cache lives on `QuantV` (per-layer-stable scope) so a single
    /// override codebook is never uploaded twice. It is preserved by
    /// `try_deep_clone` (clone via `Array::try_clone` — handle refcount, not
    /// buffer copy — so the snapshot retains the GPU upload alongside
    /// `gpu_codes_buf`); it is reset to `None` by `from_cpu_blocks*` because
    /// hydration starts with `value_codebook = None` and the next GPU encode
    /// re-uploads on demand.
    pub value_codebook_gpu: Option<Array>,
    // ── Viterbi trellis (TCQ) assignment ─────────────────────────────────────
    /// When `true`, the CPU encode path uses the Viterbi trellis-coded
    /// assignment from [`crate::tcq`] instead of nearest-centroid. Decoder is
    /// unchanged (TCQ output layout matches plain `turbo_quantize_v`).
    ///
    /// Set by `KvStorage::K8VTurbo3Tcq` (bits=3) and
    /// `KvStorage::K8VTurbo2Tcq` (bits=2) construction sites.
    /// Ignored on any other `bits` value. When `true` and `Device::Gpu`
    /// is requested, the encode path falls back to CPU (`QuantV::append_inner`).
    pub use_tcq: bool,
}

impl QuantV {
    /// Resident bytes held by this store: CPU blocks, the GPU mirror, and the
    /// TCQ sidebands (high-precision indices, value codebook and its GPU copy).
    ///
    /// All are summed unconditionally — an SSD-hydrate init leaves the
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
            high_precision_indices,
            value_codebook,
            value_codebook_gpu,
            // Geometry / bookkeeping about the buffers above, not allocations.
            gpu_words_per_step: _,
            gpu_scales_per_step: _,
            gpu_capacity: _,
            shape: _,
            bits: _,
            max_seq: _,
            use_tcq: _,
        } = self;
        // `high_precision_indices` is a Vec of per-KV-head Vecs: sum the inner
        // payloads, not just the outer Vec's headers.
        let hpi: u64 = high_precision_indices.as_ref().map_or(0, |v| {
            v.iter().map(|inner| crate::bytes::vec_bytes(inner)).sum()
        });
        blocks.iter().map(TurboBlocks::byte_size).sum::<u64>()
            + crate::bytes::opt_array_bytes(gpu_codes_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_scales_buf.as_ref())
            + hpi
            + crate::bytes::opt_vec_bytes(value_codebook.as_ref())
            + crate::bytes::opt_array_bytes(value_codebook_gpu.as_ref())
    }

    /// Zero-state decode-init helper. Returns a fresh `QuantV` with cleared
    /// CPU/GPU buffers, the supplied `shape` / `bits` / `max_seq`, and all
    /// calibration / TCQ fields set to their default-off values.
    ///
    /// Shared by every callsite that initializes a `QuantV` from scratch with
    /// an N(0,1) Lloyd-Max codec (K8V4 V, K8VTurbo* V, RotorK{3,4}Asym V).
    #[must_use]
    pub fn new_affine_decode(shape: Vec<i32>, bits: u8, max_seq: i32) -> Self {
        Self {
            blocks: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape,
            bits,
            max_seq,
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        }
    }

    /// Append a new V slice. CPU path uses scalar Rust; GPU path uses MSL kernel
    /// + pre-allocated 1D buffer with `slice_update`.
    ///
    /// The GPU grow path caps at `max_seq` tokens (defensive backstop for K8V4
    /// and similar callers). RotKTq4V must use [`Self::append_uncapped`] instead
    /// because its decode appends tokens past max_seq after a full-prefill
    /// bulk-init that already fills the buffer (/ -review HIGH 1).
    pub fn append(
        &mut self,
        f32_data: &[f32],
        new_shape: &[i32],
        v_arr: &Array,
        device: Device,
        max_seq: i32,
    ) -> Result<()> {
        self.append_inner(f32_data, new_shape, v_arr, device, Some(max_seq))
    }

    /// Append a new V slice without a max-seq cap on the grow path.
    ///
    /// Used exclusively by RotKTq4V decode, where tokens accumulate beyond the
    /// initial max_seq allocation (/ -review HIGH 1). All other
    /// callers must use [`Self::append`] which retains the defensive cap.
    pub fn append_uncapped(
        &mut self,
        f32_data: &[f32],
        new_shape: &[i32],
        v_arr: &Array,
        device: Device,
    ) -> Result<()> {
        self.append_inner(f32_data, new_shape, v_arr, device, None)
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "adding the TCQ encode + GPU→CPU fallback bumped this fn over the lint threshold; further extraction would push state across boundaries with no readability gain"
    )]
    fn append_inner(
        &mut self,
        f32_data: &[f32],
        new_shape: &[i32],
        v_arr: &Array,
        device: Device,
        max_seq_cap: Option<i32>,
    ) -> Result<()> {
        let prev_seq = self.shape[2];
        self.shape[2] += new_shape[2];

        // The codebook-buffer MSL kernel variant
        // (`turbo_quantize_v4_codebook_buf_gpu`) lets the GPU encode path honor
        // per-layer overrides at `bits == 4`. For `bits != 4` (no GPU codec
        // wired today) the bits-guard below still routes to CPU correctly.
        //
        // GPU upload of the override codebook is lazy: `value_codebook_gpu`
        // caches the [16] f32 GPU Array so it is built once per layer.
        // Note: explicit nested-Option destructure avoids the Rust 2024-only
        // let-chain syntax while still expressing the four-way guard.
        let needs_cb_upload = self.value_codebook.is_some()
            && device == Device::Gpu
            && self.bits == 4
            && self.value_codebook_gpu.is_none();
        if let (true, Some(cb)) = (needs_cb_upload, self.value_codebook.as_ref()) {
            // Validation: length must equal 16 (hardcoded for 4-bit GPU path —
            // the only `bits` value the GPU codec supports today; the bits-guard
            // a few lines down enforces `bits == 4` for the GPU branch).
            let expected_len: usize = 16;
            if cb.len() != expected_len {
                return Err(rmlx_core::error::Error::Quant(format!(
                    "value_codebook.len()={} must equal 16 for 4-bit GPU \
                     dispatch; caller bug",
                    cb.len(),
                )));
            }
            // SAFETY: `cb` is `&[f32]`; `f32` is `Copy` with a fixed 4-byte LE
            // layout. Reinterpreting as `&[u8]` is safe because `f32` and `u8`
            // have no alignment or validity requirements beyond what the slice
            // already provides.
            let cb_bytes =
                unsafe { std::slice::from_raw_parts(cb.as_ptr().cast::<u8>(), cb.len() * 4) };
            let cb_arr = Array::from_bytes(cb_bytes, &[cb.len() as i32], Dtype::F32)?;
            if prev_seq == 0 {
                tracing::info!(
                    bits = self.bits,
                    cb_len = cb.len(),
                    "QuantV: codebook override active — GPU codebook-buffer encode"
                );
            }
            self.value_codebook_gpu = Some(cb_arr);
        }

        // When Viterbi-trellis (TCQ) assignment is requested, the GPU TCQ
        // kernel ships as a future-reference hook. Force CPU on the hot encode
        // path — the Metal kernel regressed −2 % on K8VTurbo3.
        let device = if self.use_tcq && device == Device::Gpu {
            if prev_seq == 0 {
                tracing::info!(
                    bits = self.bits,
                    "QuantV: TCQ Viterbi encode active — CPU encode path (MSL kernel parked as future-ref hook)"
                );
            }
            Device::Cpu
        } else {
            device
        };

        if device == Device::Gpu {
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let d = new_shape[3] as usize;

            if self.gpu_codes_buf.is_none() {
                // Words-per-group is `bits` u32 words per
                // group of GROUP_SIZE=32 elements. Generalised from the prior
                // hard-coded `4` so 2-bit / 3-bit / 4-bit QuantV instances all
                // size their GPU buffer correctly.
                //   * bits=2 → 2*32/32 = 2 u32 per group (64 bits / 32)
                //   * bits=3 → 3*32/32 = 3 u32 per group (96 bits / 32)
                //   * bits=4 → 4*32/32 = 4 u32 per group (128 bits / 32)
                debug_assert!(
                    d.is_multiple_of(GROUP_SIZE),
                    "QuantV: head_dim must be a multiple of GROUP_SIZE={GROUP_SIZE}"
                );
                let words_per_step = b * kv_h * d * self.bits as usize / GROUP_SIZE;
                let scales_per_step = b * kv_h * d / GROUP_SIZE;
                self.gpu_words_per_step = words_per_step as i32;
                self.gpu_scales_per_step = scales_per_step as i32;
                self.max_seq = max_seq_cap.unwrap_or(i32::MAX);
                // Paged growth: start with one page, not max_seq.
                // Hydration note: if prev_seq > 0, allocate enough for the
                // existing data plus the new chunk so the grow path copies from
                // an old_codes buffer that has at least prev_seq * words_per_step
                // elements (avoids out-of-bounds slice → broadcast error).
                let init_cap = if prev_seq > 0 {
                    let needed = prev_seq + new_shape[2];
                    let pages = (needed + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
                    let uncapped = pages * KV_PAGE_SIZE;
                    match max_seq_cap {
                        Some(ms) => uncapped.min(ms),
                        None => uncapped,
                    }
                } else {
                    match max_seq_cap {
                        Some(ms) => KV_PAGE_SIZE.min(ms),
                        None => KV_PAGE_SIZE,
                    }
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
                // C1 fix: upload CPU TurboBlocks to GPU when initialising a
                // hydrated layer (prev_seq > 0 and self.blocks non-empty).
                //
                // TurboBlocks CPU layout: `codes: Vec<u8>` (4-bit packed, 4 bytes
                // per u32 GPU word, same LSB-first layout as the GPU kernel — see
                // QUANTIZE_SOURCE comment in turboquant_msl.rs). `scales: Vec<f32>`
                // (4 bytes per f32). Flatten all blocks, reinterpret as bytes, and
                // upload via `Array::from_bytes` + `slice_update`.
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
                    // H7: tracing event for hydrated QuantV GPU init.
                    tracing::debug!(
                        prev_seq,
                        init_cap,
                        cpu_words,
                        cpu_scales,
                        "QuantV hydrated init: uploaded CPU TurboBlocks → GPU"
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
                    let uncapped = pages * KV_PAGE_SIZE;
                    // Cap at max_seq when the caller provided one (K8V4 and
                    // similar paths); uncapped callers (RotKTq4V) may grow past.
                    match max_seq_cap {
                        Some(ms) => uncapped.min(ms),
                        None => uncapped,
                    }
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

            let new_seq = new_shape[2];
            let words_per_seq = self.gpu_words_per_step;
            let scales_per_seq = self.gpu_scales_per_step;

            // The only GPU codec wired here today is the
            // V4 path. The 2-bit / 3-bit QuantV variants (K8VTurbo2 / K8VTurbo3)
            // force `Device::Cpu` at their update sites and never reach this
            // branch; a fall-through with bits != 4 would silently corrupt the
            // codes buffer because the V4 kernel writes 4 u32 per group.
            if self.bits != 4 {
                return Err(rmlx_core::error::Error::Quant(format!(
                    "QuantV::append: GPU path only supports bits=4 (got bits={}); \
                     callers for 2-bit / 3-bit must dispatch with Device::Cpu",
                    self.bits
                )));
            }
            // Route to the codebook-buffer kernel variant when
            // a per-layer override is present. Cache built above by the
            // value_codebook_gpu lazy-init block. Predicate is `value_codebook`
            // (the user-visible invariant), not `value_codebook_gpu`, so a
            // caller-mutated codebook is surfaced rather than silently routed
            // back to the hardwired-CB kernel.
            debug_assert_eq!(
                self.value_codebook.is_some(),
                self.value_codebook_gpu.is_some(),
                "value_codebook ↔ value_codebook_gpu cache desync — \
                 caller mutated pub value_codebook after first GPU upload"
            );
            // Reorder the chunk to sequence-major `[B, new_seq, kv_h, D]` before
            // quantizing. The flat GPU buffer accumulates chunks back-to-back at
            // `prev_seq * words_per_seq`, so the active prefix is sequence-major;
            // `dequantize_choice` reads it back with the matching reshape +
            // transpose. The incoming chunk is head-major
            // (`[B, kv_h, new_seq, D]`); transposing heads↔seq makes the stored
            // layout self-consistent across any number of appends and any
            // `kv_h`. With a single chunk (`prev_seq == 0`) this is the identity
            // for the dequant view, so the cold-prefill path stays logically
            // correct (byte-identical when `D % GROUP_SIZE == 0`, which the
            // init-block debug_assert enforces). `transpose` yields a strided
            // view; the TurboQuant MSL kernel reads its input by raw linear
            // offset (ignores MLX strides), so materialize the permutation into
            // a row-major buffer with `contiguous` first.
            let v_seq_major = v_arr.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
            let (new_codes, new_scales) = match (
                self.value_codebook.is_some(),
                self.value_codebook_gpu.as_ref(),
            ) {
                (true, Some(cb_gpu)) => {
                    turbo_quantize_v4_codebook_buf_gpu(&v_seq_major, cb_gpu, device)?
                }
                (false, None) => turbo_quantize_v4_gpu(&v_seq_major, device)?,
                (true, None) => {
                    return Err(rmlx_core::error::Error::Quant(
                        "value_codebook set but GPU upload missing — internal \
                         invariant violation"
                            .to_string(),
                    ));
                }
                (false, Some(_)) => {
                    return Err(rmlx_core::error::Error::Quant(
                        "value_codebook_gpu set without value_codebook — internal \
                         invariant violation"
                            .to_string(),
                    ));
                }
            };

            let codes_start = prev_seq * words_per_seq;
            let codes_stop = (prev_seq + new_seq) * words_per_seq;
            let scales_start = prev_seq * scales_per_seq;
            let scales_stop = (prev_seq + new_seq) * scales_per_seq;

            let codes_buf = self.gpu_codes_buf.take().unwrap();
            let scales_buf = self.gpu_scales_buf.take().unwrap();

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
            // CPU path: scalar Rust quantization.
            //
            // CPU mirror of the GPU path: store the chunk sequence-major so the
            // accumulated `self.blocks` share one layout with the GPU buffer
            // (the spill/hydrate round-trip moves codes between the two, and the
            // hydrated GPU init uploads CPU blocks at offset 0). `f32_data`
            // arrives head-major (`[B, kv_h, new_seq, D]`); reorder it to
            // `[B, new_seq, kv_h, D]` and pass the matching shape before
            // quantizing. The codecs group positionally over the last axis `D`,
            // so the seq-major shape is valid and `dequantize_choice` reorders
            // back to the logical `[B, kv_h, S, D]`.
            //
            // When `use_tcq` is set, dispatch to the Viterbi trellis encoder at
            // the matching bit-width. bits==3: 3-bit TCQ; bits==2: 2-bit TCQ.
            // Codebook override is supported for 3-bit only; 2-bit TCQ always
            // uses the built-in Lloyd-Max 2-bit codebook.
            // If no TCQ and a per-layer codebook override is set, use it;
            // otherwise fall back to the built-in Lloyd-Max codebook.
            let b = new_shape[0] as usize;
            let kv_h = new_shape[1] as usize;
            let new_seq = new_shape[2] as usize;
            let d = new_shape[3] as usize;
            let seq_major = super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, d);
            let seq_shape = [new_shape[0], new_shape[2], new_shape[1], new_shape[3]];
            let block = if self.use_tcq && self.bits == 3 {
                match self.value_codebook.as_deref() {
                    Some(cb) => tcq_quantize_v3_with_codebook(&seq_major, &seq_shape, cb)?,
                    None => tcq_quantize_v3(&seq_major, &seq_shape)?,
                }
            } else if self.use_tcq && self.bits == 2 {
                // 2-bit TCQ. Codebook override not wired for 2-bit — TCQ reuses
                // the standard Lloyd-Max 2-bit codebook.
                tcq_quantize_v2(&seq_major, &seq_shape)?
            } else {
                turbo_quantize_v_with_codebook(
                    &seq_major,
                    self.bits,
                    &seq_shape,
                    self.value_codebook.as_deref(),
                )?
            };
            self.blocks.push(block);
        }
        Ok(())
    }

    /// Truncate the store to `n` sequence positions.
    ///
    /// Drops trailing CPU blocks past `n` and **splits** the block the cut lands
    /// inside (see [`super::truncate_plan`]), then lowers `shape[2]` to `n`.
    ///
    /// The GPU buffers are deliberately left alone: `dequantize_choice` slices
    /// `[0, shape[2])` and the next `append` writes at `prev_seq == shape[2]`,
    /// so lowering `shape[2]` already makes the rejected region overwritable.
    /// The CPU blocks are append-only, so without the cut the next `append`
    /// stacks on top of the rejected tokens and the dequant reads the rejected
    /// prefix back.
    ///
    /// The target is clamped to the store's current `shape[2]` — see
    /// [`super::clamp_truncate_target`] for why that is not defensive padding
    /// but the only correct reading for a ring-less store.
    pub fn truncate_to(&mut self, n: i32) {
        super::truncate_block_store(&mut self.blocks, &mut self.shape, n);
    }

    /// Dequantize all accumulated V slices to a flat f32 vec (CPU path) or
    /// directly return the MLX Array (GPU path).
    ///
    /// Returns `(flat_f32, None)` for CPU or `(empty, Some(Array))` for GPU.
    ///
    /// # Errors
    ///
    /// On the CPU path, returns `Error::Quant` when the accumulated blocks do
    /// not decode to exactly `prod(shape)` elements — see the coverage check
    /// in the body.
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
                // Only the V4 GPU dequant kernel is wired at the QuantV GPU
                // dispatch slot today. 2-bit / 3-bit variants round-trip through
                // the CPU path; refuse a GPU dequant request
                // on a non-4-bit QuantV rather than silently producing garbage.
                if self.bits != 4 {
                    return Err(rmlx_core::error::Error::Quant(format!(
                        "QuantV::dequantize_choice: GPU dequant only supports bits=4 \
                         (got bits={}); callers for 2-bit / 3-bit must request Device::Cpu",
                        self.bits
                    )));
                }
                let s = self.shape[2];
                let codes = codes_buf.slice(&[0], &[s * self.gpu_words_per_step], &[1], device)?;
                let scales =
                    scales_buf.slice(&[0], &[s * self.gpu_scales_per_step], &[1], device)?;
                // Route to the codebook-buffer dequant kernel variant when a
                // per-layer override is present, otherwise the hardwired-CB
                // kernel. Both kernels write bf16/f16/f32 in one pass — no
                // follow-up astype. Predicate is `value_codebook` (the
                // user-visible invariant); a typed error surfaces a cache/source
                // desync rather than silently routing to the hardwired-CB kernel.
                debug_assert_eq!(
                    self.value_codebook.is_some(),
                    self.value_codebook_gpu.is_some(),
                    "value_codebook ↔ value_codebook_gpu cache desync — \
                     caller mutated pub value_codebook after first GPU upload"
                );
                // The flat buffer is sequence-major (see `append`): the active
                // prefix lays out as `[B, S, kv_h, D]`. Dequant into that shape,
                // then reorder heads↔seq back to the logical `[B, kv_h, S, D]`
                // callers expect. For a single-chunk cache the transpose inverts
                // the append-side reorder, so the round-trip is exact within
                // quant noise. The dequant kernel reads its inputs by raw linear
                // offset; the seq-major reshape on the OUTPUT then needs a final
                // `contiguous` because raw byte-readers (SSD spill / hydrate)
                // would otherwise see the un-permuted sequence-major bytes.
                let seq_major_shape = [self.shape[0], self.shape[2], self.shape[1], self.shape[3]];
                let out = match (
                    self.value_codebook.is_some(),
                    self.value_codebook_gpu.as_ref(),
                ) {
                    (true, Some(cb_gpu)) => turbo_dequantize_v4_codebook_buf_gpu(
                        &codes,
                        &scales,
                        cb_gpu,
                        &seq_major_shape,
                        out_dtype,
                        device,
                    )?,
                    (false, None) => turbo_dequantize_v4_gpu(
                        &codes,
                        &scales,
                        &seq_major_shape,
                        out_dtype,
                        device,
                    )?,
                    (true, None) => {
                        return Err(rmlx_core::error::Error::Quant(
                            "value_codebook set but GPU upload missing — internal \
                             invariant violation"
                                .to_string(),
                        ));
                    }
                    (false, Some(_)) => {
                        return Err(rmlx_core::error::Error::Quant(
                            "value_codebook_gpu set without value_codebook — internal \
                             invariant violation"
                                .to_string(),
                        ));
                    }
                };
                let out = out.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
                return Ok((Vec::new(), Some(out)));
            }
            return Ok((Vec::new(), None));
        }
        // CPU path. Blocks are stored sequence-major (see `append`): the
        // concatenated decode lays out as `[B, S, kv_h, D]`. Reorder heads↔seq
        // back to the logical `[B, kv_h, S, D]` the caller reshapes to.
        let total: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out = Vec::with_capacity(total);
        let cb_override = self.value_codebook.as_deref();
        for block in &self.blocks {
            let slice = if cb_override.is_some() {
                turbo_dequantize_with_codebook(block, cb_override)?
            } else {
                turbo_dequantize(block)?
            };
            out.extend_from_slice(&slice);
        }
        // The reorder requires exactly the `[B, S, kv_h, D]` buffer the shape
        // declares. Blocks that over-run it used to be silently cut back here,
        // which after a mid-block truncate kept the *rejected* speculative
        // prefix and threw away the appended correction — wrong attention with
        // no error anywhere. Blocks that fall short used to be zero-padded,
        // which fabricates a gap. `truncate_to` now cuts the blocks; when it
        // cannot (see `super::truncate_plan`) it drops the trailing block whole
        // and leaves the store short on purpose, and that must abort the
        // request here rather than decode a plausible-looking wrong tensor.
        if out.len() != total {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantV::dequantize_choice: CPU blocks decode to {} elems but shape {:?} \
                 implies {total} — refusing to zero-pad / truncate",
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
        let out = super::seq_layout::transpose_chunked_seq_heads(
            &out,
            b,
            s,
            kv_h,
            d,
            self.blocks.iter().map(super::BlockRows::rows),
        )?;
        Ok((out, None))
    }

    /// Reconstruct a CPU-path `QuantV` from serialized TurboQuant blocks.
    /// GPU buffers stay empty. `high_precision_indices` and `value_codebook`
    /// are not persisted to SSD; hydrated instances start with `None`
    /// (surface-only wiring; high-precision indices not yet consumed).
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
            high_precision_indices: None,
            value_codebook: None,
            value_codebook_gpu: None,
            use_tcq: false,
        }
    }

    /// Companion of [`Self::from_cpu_blocks`] for TCQ-tagged payloads.
    /// Identical except `use_tcq = true` so any subsequent decode-step
    /// `append` continues using the Viterbi encoder.
    pub fn from_cpu_blocks_tcq(blocks: Vec<TurboBlocks>, shape: Vec<i32>, bits: u8) -> Self {
        let mut q = Self::from_cpu_blocks(blocks, shape, bits);
        q.use_tcq = true;
        q
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
            high_precision_indices: self.high_precision_indices.clone(),
            value_codebook: self.value_codebook.clone(),
            value_codebook_gpu: match &self.value_codebook_gpu {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            use_tcq: self.use_tcq,
        })
    }
}

#[cfg(test)]
#[path = "quant_v_tests.rs"]
mod quant_v_tests;
