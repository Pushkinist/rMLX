// IsoQuant 3-bit V storage struct.
//
// LOC-exempt: the GPU mirror adds ~360 LOC in append_gpu + dequant_gpu;
// planned split to sibling iso_gpu_mirror.rs once the remaining codec
// mirrors land. The split is deferred to keep the review surface contained.
//
// Mirrors the `QuantPlanarV` layout (CPU-side `Vec` buffers); GPU buffers are
// reserved for T11d when the MSL kernel lands. Promoted to `pub` so the SSD
// modules (in `rmlx-kv-ssd`) can reach across the crate boundary once T11c
// wires SSD spill/hydrate.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized V buffer: `QuantIsoV3` (IsoQuant 3-bit V codec).

use rmlx_core::error::Result;
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};

use super::KV_PAGE_SIZE;

/// Bit-width of the iso3 V codec (fixed at 3-bit — see
/// [`crate::isoquant::iso_encode_fast`]).
pub const ISO3_BITS: u8 = 3;

/// Quaternion-block size for the iso3 codec (fixed at 4 elements / group; one
/// quaternion per group in fast mode).
pub const ISO3_GROUP_SIZE: usize = 4;

/// One token's iso3 payload: codes + per-group scales + per-group quaternion +
/// L2 norm.
///
/// Mirrors the [`crate::planarquant::PlanarBlocks`] shape but stores quaternions
/// + per-token norm rather than per-pair rotations. CPU-side only — the GPU
/// path lands in T11d.
#[derive(Debug, Clone)]
pub struct IsoBlocks {
    /// Packed `bits`-bit codes; layout determined by the owning `QuantIsoV*`
    /// struct. Length per token = `n_groups * ceil(group_size / vals_per_word(bits))` u32s
    /// where `vals_per_word(bits) = 32 / bits`.
    pub codes: Vec<u32>,
    /// Per-group scale (one f32 per `(token, group)`).
    pub scales: Vec<f32>,
    /// Per-group quaternion `[w, x, y, z]` — `n_tokens * n_groups * 4` f32s.
    pub quaternions: Vec<f32>,
    /// Per-token L2 norm.
    pub norms: Vec<f32>,
    /// Number of tokens this block represents.
    pub n_tokens: usize,
}

impl IsoBlocks {
    /// Heap bytes this block holds.
    ///
    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile. See [`crate::bytes`].
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            codes,
            scales,
            quaternions,
            norms,
            // Inline metadata (no heap payload).
            n_tokens: _,
        } = self;
        crate::bytes::vec_bytes(codes)
            + crate::bytes::vec_bytes(scales)
            + crate::bytes::vec_bytes(quaternions)
            + crate::bytes::vec_bytes(norms)
    }
}

/// Accumulated IsoQuant V cache (3-bit, quaternion SO(4) fast mode).
///
/// Storage parallels [`crate::storage::QuantPlanarV`] but the per-group
/// rotation is a 4-component quaternion rather than a 4-bit rotation index, and
/// a per-token norm scalar is preserved separately.
///
/// CPU-only. GPU buffers are not yet allocated — the MSL kernel is deferred.
/// The SDPA dispatch falls through to the dequant-then-SDPA legacy path
/// (see `kvcache::sdpa`).
pub struct QuantIsoV3 {
    /// Accumulated per-token blocks (one entry per append call; `dequant`
    /// flattens them).
    pub blocks: Vec<IsoBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    // The provisioned window is deliberately NOT a field: it lives on the
    // `KvStorage` variant, grows as the sequence does, and is passed to
    // `append_gpu` per call. A cached copy is a snapshot that goes stale the
    // moment the window grows — the same trap `QuantRotorK{3,4}` used to carry.
    /// Bit-width tag (always [`ISO3_BITS`] for this codec; kept as a field for
    /// symmetry with the other `Quant*` structs and future-proofing).
    pub bits: u8,
    // ── GPU-resident codec mirror ────────────────────────────────────────────
    //
    // Internal mirror state. Crate-private per CLAUDE.md "Public API surface
    // conservative — no leaking mlx-rs types directly". Consumers go through
    // `append_gpu` / `dequant_gpu` / `byte_size` / `try_deep_clone` /
    // `truncate_to` / `reset`.
    /// Pre-allocated u32 codes buffer on GPU. Length
    /// `B * kv_h * max_seq * n_groups * WORDS_PER_GROUP`. Quaternion buffer is
    /// omitted on purpose — every group uses the same [`FIXED_QUAT`] constant
    /// (the `iso_dequantize_v3_gpu` kernel never reads it; see kernel source
    /// in `isoquant_msl.rs`). `None` until first GPU `append_gpu` call, or
    /// when the [`gate`][crate::gpu_resident_iso_enabled] is disabled.
    pub(crate) gpu_codes_buf: Option<Array>,
    /// Pre-allocated f32 scales buffer on GPU. Length
    /// `B * kv_h * max_seq * n_groups` (one f32 per group).
    pub(crate) gpu_scales_buf: Option<Array>,
    /// Pre-allocated f32 norms buffer on GPU, per-group layout (kernel
    /// contract — one slot per `(token, group)`). Length matches scales.
    pub(crate) gpu_norms_buf: Option<Array>,
    /// Number of u32 code-words written per single token (across `B * kv_h`).
    pub(crate) gpu_words_per_step: i32,
    /// Number of f32 scale/norm slots written per single token (across
    /// `B * kv_h`).
    pub(crate) gpu_groups_per_step: i32,
    /// Currently allocated capacity in tokens (paged growth, multiple of
    /// `KV_PAGE_SIZE`). 0 until first GPU append.
    pub(crate) gpu_capacity: i32,
    /// Number of tokens already written into the GPU mirror (matches
    /// `shape[2]` while the mirror is active; advances independently when
    /// CPU-only blocks are also appended, e.g. after an SSD hydrate fallback).
    pub(crate) gpu_offset: i32,
}

impl std::fmt::Debug for QuantIsoV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // GPU buffers are Option<Array>; logging the full handle is noisy.
        // Summarise mirror presence + sizing instead.
        f.debug_struct("QuantIsoV3")
            .field("n_blocks", &self.blocks.len())
            .field("shape", &self.shape)
            .field("bits", &self.bits)
            .field("gpu_capacity", &self.gpu_capacity)
            .field("gpu_offset", &self.gpu_offset)
            .field("gpu_mirror", &self.gpu_codes_buf.is_some())
            .field("gpu_words_per_step", &self.gpu_words_per_step)
            .field("gpu_groups_per_step", &self.gpu_groups_per_step)
            .field(
                "gpu_scales_buf",
                &self.gpu_scales_buf.as_ref().map(|_| "<Array>"),
            )
            .field(
                "gpu_norms_buf",
                &self.gpu_norms_buf.as_ref().map(|_| "<Array>"),
            )
            .finish()
    }
}

impl QuantIsoV3 {
    /// Construct an empty `QuantIsoV3` for `init_shape = [B, kv_h, 0, D]`.
    ///
    /// The seq dim of `init_shape` should be 0 — the first `append` call
    /// supplies `new_shape` with the actual seq increment.
    #[must_use]
    pub fn new(init_shape: Vec<i32>) -> Self {
        Self {
            blocks: Vec::new(),
            shape: init_shape,
            bits: ISO3_BITS,
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_norms_buf: None,
            gpu_words_per_step: 0,
            gpu_groups_per_step: 0,
            gpu_capacity: 0,
            gpu_offset: 0,
        }
    }

    /// Append one V slice (CPU path).
    ///
    /// `f32_data` is the flat f32 view of the V chunk shape `[B, kv_h,
    /// new_seq, D]`. The codec processes each `(b, kv_h, tok)` row of length
    /// `head_dim = D` independently — same convention as
    /// [`crate::planarquant::planar_quantize`].
    ///
    /// # Errors
    ///
    /// Forwards any [`IsoQuantError`] from [`iso_encode_fast`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoV3::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let b = new_shape[0] as usize;
        let kv_h = new_shape[1] as usize;
        let new_seq = new_shape[2] as usize;
        let head_dim = new_shape[3] as usize;
        let n_tokens_total = b * kv_h * new_seq;

        // The codec is per-token-row positional (one row of length `head_dim`
        // per token). Blocks accumulate one chunk per append; `dequant`
        // concatenates them and the caller reshapes head-major `[B, kv_h, S, D]`.
        // A head-major store transposes heads across a multi-append GQA cache
        // (kv_h>1), so store each chunk sequence-major (`[B, new_seq, kv_h, D]`)
        // — reorder the head-major input heads↔seq before quantizing and
        // `dequant` reorders back to the logical `[B, kv_h, S, D]`.
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, quaternions, norms) =
            iso_encode_fast(&seq_major, head_dim, ISO3_GROUP_SIZE, ISO3_BITS).map_err(
                |e: IsoQuantError| rmlx_core::error::Error::Mlx(format!("iso3 encode: {e}")),
            )?;

        self.blocks.push(IsoBlocks {
            codes,
            scales,
            quaternions,
            norms,
            n_tokens: n_tokens_total,
        });

        // Bookkeeping: advance the seq dim of the accumulated shape.
        // shape[0]/shape[1]/shape[3] must be initialised by the caller (or set on
        // first append).
        if self.shape.len() != 4 || self.shape[0] == 0 {
            self.shape = new_shape.to_vec();
        } else {
            self.shape[2] += new_shape[2];
        }
        Ok(())
    }

    /// Construct a `QuantIsoV3` from pre-computed CPU blocks (SSD hydrate path).
    ///
    /// Mirrors `QuantPlanarV::from_cpu_blocks` — caller supplies the flat
    /// concatenated buffers (one `IsoBlocks` per call from `block_io`) and the
    /// 4-D shape `[B, kv_h, S, D]`. `max_seq` defaults to `shape[2]` (the
    /// hydrated sequence length); callers that need a larger window update it
    /// separately after construction.
    #[must_use]
    pub fn from_cpu_blocks(blocks: Vec<IsoBlocks>, shape: Vec<i32>) -> Self {
        // Caller (SSD hydrate `read_quant_iso_v3`) always provides a 4-element
        // [B, kv_h, S, D] shape. A shorter shape means a coding error upstream.
        debug_assert!(
            shape.len() == 4,
            "QuantIsoV3::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        Self {
            blocks,
            shape,
            bits: ISO3_BITS,
            // Hydrate path leaves the GPU mirror unallocated. The next
            // `dequant_gpu` call falls back to the CPU-staged upload path
            // until the next `append_gpu` lazily re-allocates the mirror.
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_norms_buf: None,
            gpu_words_per_step: 0,
            gpu_groups_per_step: 0,
            gpu_capacity: 0,
            gpu_offset: 0,
        }
    }

    /// Reset the accumulated sequence length to zero. Buffers are kept for
    /// reuse on the next request.
    pub fn reset(&mut self) {
        self.blocks.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = 0;
        }
        // Drop the GPU mirror so the next encode re-allocates from scratch
        // (the buffer is sized off `max_seq`; reset is the only safe rebuild
        // point — `truncate_to` keeps the mirror live).
        self.gpu_codes_buf = None;
        self.gpu_scales_buf = None;
        self.gpu_norms_buf = None;
        self.gpu_capacity = 0;
        self.gpu_offset = 0;
        // gpu_*_per_step are derived from shape/head_dim at allocation time;
        // they are re-derived on the next append, so clear them here too to
        // keep `Debug` honest.
        self.gpu_words_per_step = 0;
        self.gpu_groups_per_step = 0;
    }

    /// Truncate the accumulated sequence to `n` tokens.
    ///
    /// Drops trailing blocks until the cumulative `n_tokens` count is `<= n`
    /// and lowers `shape[2]` to `n`. The codec is per-token so block
    /// boundaries align with token boundaries.
    pub fn truncate_to(&mut self, n: i32) {
        let n_usize = n.max(0) as usize;
        // shape[0] * shape[1] is the per-token row multiplier; iso codec
        // operates on flat rows so n_tokens already counts rows. For decode
        // (B=1, kv_h fixed) the relationship is trivial.
        let mut acc: usize = 0;
        let mut keep = 0usize;
        for (i, blk) in self.blocks.iter().enumerate() {
            if acc + blk.n_tokens <= n_usize {
                acc += blk.n_tokens;
                keep = i + 1;
            } else {
                break;
            }
        }
        self.blocks.truncate(keep);
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
        // Lower the GPU mirror offset; underlying buffer untouched (matches
        // `QuantV::truncate` semantics — the trailing tokens become logically
        // free and the next `append_gpu` overwrites them via `slice_update`).
        let n_clamped = n.max(0);
        if self.gpu_offset > n_clamped {
            self.gpu_offset = n_clamped;
        }
    }

    /// Deep-clone (CPU path is plain `Vec` clones).
    ///
    /// # Errors
    ///
    /// Currently infallible on the CPU path; returns `Result` for parity with
    /// the other `Quant*` structs (their GPU buffers fallibly clone Arrays).
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            blocks: self.blocks.clone(),
            shape: self.shape.clone(),
            bits: self.bits,
            // Handle refcount via `try_clone` (clone of the lazy MLX handle,
            // not a buffer copy — same pattern as `QuantV::try_deep_clone`).
            gpu_codes_buf: match &self.gpu_codes_buf {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            gpu_scales_buf: match &self.gpu_scales_buf {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            gpu_norms_buf: match &self.gpu_norms_buf {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            gpu_words_per_step: self.gpu_words_per_step,
            gpu_groups_per_step: self.gpu_groups_per_step,
            gpu_capacity: self.gpu_capacity,
            gpu_offset: self.gpu_offset,
        })
    }

    /// Resident bytes held by this store: CPU blocks plus the GPU mirror.
    ///
    /// Both are summed unconditionally. `gpu_offset` advances independently of
    /// `blocks`, so after an SSD-hydrate fallback the mirror and the CPU blocks
    /// are resident at the same time; an either/or branch would drop one of
    /// them. Unallocated buffers contribute 0 on their own.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            blocks,
            gpu_codes_buf,
            gpu_scales_buf,
            gpu_norms_buf,
            // Geometry / bookkeeping about the buffers above, not allocations.
            shape: _,
            bits: _,
            gpu_words_per_step: _,
            gpu_groups_per_step: _,
            gpu_capacity: _,
            gpu_offset: _,
        } = self;
        blocks.iter().map(IsoBlocks::byte_size).sum::<u64>()
            + crate::bytes::opt_array_bytes(gpu_codes_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_scales_buf.as_ref())
            + crate::bytes::opt_array_bytes(gpu_norms_buf.as_ref())
    }

    /// Dequantize all accumulated V slices into one flat f32 vector
    /// of length `prod(shape)`.
    ///
    /// Concatenates each block's `iso_decode_fast` output in append order.
    ///
    /// # Errors
    ///
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails for
    /// any block.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoV3::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);
        for blk in &self.blocks {
            let dec = iso_decode_fast(
                &blk.codes,
                &blk.scales,
                &blk.quaternions,
                &blk.norms,
                head_dim,
                ISO3_GROUP_SIZE,
                ISO3_BITS,
            )
            .map_err(|e: IsoQuantError| {
                rmlx_core::error::Error::Mlx(format!("iso3 decode: {e}"))
            })?;
            out.extend_from_slice(&dec);
        }
        // dequant may stop short of total_elems if shape[2] hasn't been
        // advanced past the appended blocks — caller treats this as best-
        // effort. Pad with zeros so the returned vec matches the declared
        // shape if needed.
        if out.len() < total_elems {
            out.resize(total_elems, 0.0);
        } else if out.len() > total_elems {
            out.truncate(total_elems);
        }
        // Blocks are sequence-major (see `append`); reorder the concatenated
        // decode back to the logical head-major `[B, kv_h, S, D]`.
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let out = super::seq_layout::transpose_seq_heads(&out, b, s, kv_h, head_dim);
        Ok(out)
    }

    /// GPU-resident encode + mirror update.
    ///
    /// Dispatches the iso3 MSL encode kernel directly on `v_arr` and:
    ///
    /// 1. Reads the GPU outputs back into CPU `IsoBlocks` (SSD spill stays on
    ///    the CPU-blocks path; on-disk format unchanged).
    /// 2. When [`crate::gpu_resident_iso_enabled`] is `true`, writes the
    ///    just-encoded GPU code/scale/norm slices into a pre-allocated
    ///    per-struct buffer via `slice_update`, so the next `dequant_gpu`
    ///    call can skip the `Array::from_bytes` re-upload.
    ///
    /// Replaces the `iso3_gpu_append_into_blocks` callsite in
    /// `update_iso3` / `update_iso3_sym`.
    ///
    /// # Errors
    ///
    /// - `Error::Mlx` if the MSL encode kernel or `slice_update` fails.
    /// - `Error::Quant` for malformed shapes or capacity overflow.
    #[allow(
        clippy::too_many_lines,
        reason = "the lazy-init / grow / per-chunk-update branches share local state \
                  (per-step sizes, prev_seq, capacity) and pulling any one out into a \
                  helper costs >2 round-trip params without improving readability. \
                  Matches `QuantV::append_inner` which carries the same lint allow for \
                  the same reason."
    )]
    pub fn append_gpu(
        &mut self,
        v_arr: &Array,
        new_shape: &[i32],
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoV3::append_gpu: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let b = new_shape[0] as usize;
        let kv_h = new_shape[1] as usize;
        let s_new = new_shape[2] as usize;
        let head_dim = new_shape[3] as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(ISO3_GROUP_SIZE) {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantIsoV3::append_gpu: head_dim={head_dim} must be a positive multiple of \
                 ISO3_GROUP_SIZE={ISO3_GROUP_SIZE}"
            )));
        }
        let n_groups = head_dim / ISO3_GROUP_SIZE;
        let n_tokens_total = b
            .checked_mul(kv_h)
            .and_then(|v| v.checked_mul(s_new))
            .ok_or_else(|| {
                rmlx_core::error::Error::Quant(
                    "QuantIsoV3::append_gpu: n_tokens_total overflow".to_owned(),
                )
            })?;

        // ── 1. Dispatch the MSL encode kernel. ─────────────────────────────
        // The codec is per-token-row positional; the GPU mirror accumulates
        // each chunk's encode at `prev_seq * words_per_step`, so a head-major
        // store + head-major reshape on `dequant_gpu` transposes heads across
        // multi-append GQA caches (kv_h>1). Reorder the chunk to sequence-major
        // `[B, new_seq, kv_h, D]` before quantizing — `transpose` yields a
        // strided view, and the iso3 MSL kernel reads its input by raw linear
        // offset (ignores MLX strides), so materialize with `contiguous` first.
        // `quats_gpu` is a constant FIXED_QUAT-filled placeholder that the
        // dequant kernel never reads (see `iso_dequantize_v3_gpu`); we forward
        // it to `iso3_gpu_outputs_to_cpu` for ABI parity and then drop it.
        let v_seq_major = v_arr.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
        let (codes_gpu, scales_gpu, quats_gpu, norms_gpu) =
            crate::isoquant_msl::iso_quantize_v3_gpu(&v_seq_major, head_dim, device)?;

        // ── 2. Read back into CPU blocks (SSD spill compatibility). ────────
        // Reuse the shared helper so this path stays bit-identical with the
        // CPU-block layout (matters for SSD round-trip — same byte order as
        // `write_quant_iso_v3`).
        let (codes_cpu, scales_cpu, quats_cpu, norms_cpu) =
            crate::isoquant_msl::iso3_gpu_outputs_to_cpu(
                &codes_gpu,
                &scales_gpu,
                &quats_gpu,
                &norms_gpu,
                n_tokens_total,
                n_groups,
            )?;
        self.blocks.push(IsoBlocks {
            codes: codes_cpu,
            scales: scales_cpu,
            quaternions: quats_cpu,
            norms: norms_cpu,
            n_tokens: n_tokens_total,
        });

        // ── 3. Bookkeeping shared with the CPU `append` path. ──────────────
        let prev_seq = if self.shape.len() == 4 && self.shape[0] != 0 {
            self.shape[2]
        } else {
            0
        };
        if self.shape.len() != 4 || self.shape[0] == 0 {
            self.shape = new_shape.to_vec();
        } else {
            self.shape[2] += new_shape[2];
        }

        // SAFETY: `gpu_resident_iso_enabled()` is hardcoded `false` in
        // production; in test mode it uses OnceLock latching on first read.
        // Either way the gate state cannot toggle between prev_seq capture
        // (above) and this check, so prev_seq is safe to use unconditionally.
        // ── 4. GPU mirror write (gated). ───────────────────────────────────
        if !crate::gpu_resident_iso_enabled() {
            return Ok(());
        }

        // Per-step sizes are constant across appends (per-(B, kv_h, head_dim)
        // shape is fixed at first encode). Derive once, then reuse.
        let words_per_step = b
            .checked_mul(kv_h)
            .and_then(|v| v.checked_mul(n_groups))
            .and_then(|v| v.checked_mul(crate::isoquant_msl::ISO3_WORDS_PER_GROUP))
            .ok_or_else(|| {
                rmlx_core::error::Error::Quant("append_gpu: words_per_step overflow".to_owned())
            })?;
        let groups_per_step = b
            .checked_mul(kv_h)
            .and_then(|v| v.checked_mul(n_groups))
            .ok_or_else(|| {
                rmlx_core::error::Error::Quant("append_gpu: groups_per_step overflow".to_owned())
            })?;

        // HIGH-1 torn-state guard: all GPU work below builds new Arrays in
        // locals first; `self.gpu_*` fields are only mutated after every
        // fallible step succeeds. On any Err in this block we reset the
        // mirror to the lazy-init state (None + zero capacity/offset) and
        // emit a tracing::warn so the dequant_gpu fallback runs against
        // matching CPU-blocks state.
        let new_offset = prev_seq + s_new as i32;
        let mirror_result = self.update_gpu_mirror(
            words_per_step,
            groups_per_step,
            prev_seq,
            s_new,
            &codes_gpu,
            &scales_gpu,
            &norms_gpu,
            max_seq,
            device,
        );
        match mirror_result {
            Ok(()) => {
                tracing::trace!(
                    target: "rmlx::kv_quant::iso3_gpu",
                    prev_seq,
                    s_new,
                    new_offset,
                    "iso3 V GPU mirror update"
                );
                self.gpu_offset = new_offset;
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    target: "rmlx::kv_quant::iso3_gpu",
                    error = %e,
                    prev_seq,
                    s_new,
                    "iso3 V GPU mirror update failed; resetting mirror to fall back to CPU-staged dequant"
                );
                self.gpu_codes_buf = None;
                self.gpu_scales_buf = None;
                self.gpu_norms_buf = None;
                self.gpu_capacity = 0;
                self.gpu_offset = 0;
                Err(e)
            }
        }
    }

    /// Inner GPU mirror updater for [`Self::append_gpu`]. Builds all new
    /// Arrays in locals and only commits to `self.gpu_*` after every fallible
    /// step succeeds. Returns `Err` on any failure without mutating the
    /// codes/scales/norms buffer slots — the caller (`append_gpu`) then
    /// performs the torn-state reset.
    ///
    /// Note `self.gpu_words_per_step`, `gpu_groups_per_step`, and
    /// `gpu_capacity` ARE mutated on success; they are set to the new values
    /// only at the end of this fn (after all `slice_update` succeed).
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "extracted from append_gpu purely to gate the torn-state guard; \
                  inlining would re-introduce the bug. The lazy-init/grow/write \
                  branches share local capacity + per-step state and splitting \
                  further hurts readability."
    )]
    fn update_gpu_mirror(
        &mut self,
        words_per_step: usize,
        groups_per_step: usize,
        prev_seq: i32,
        s_new: usize,
        codes_gpu: &Array,
        scales_gpu: &Array,
        norms_gpu: &Array,
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        // ── Lazy-init the GPU mirror on first encode. ──────────────────────
        let init_cap = if self.gpu_codes_buf.is_none() {
            if max_seq > 0 {
                KV_PAGE_SIZE.min(max_seq)
            } else {
                KV_PAGE_SIZE
            }
        } else {
            self.gpu_capacity
        };

        if self.gpu_codes_buf.is_none() {
            let codes_len = (words_per_step * init_cap as usize) as i32;
            let groups_len = (groups_per_step * init_cap as usize) as i32;
            let codes_buf = zeros(&[codes_len], Dtype::U32, device)?;
            let scales_buf = zeros(&[groups_len], Dtype::F32, device)?;
            let norms_buf = zeros(&[groups_len], Dtype::F32, device)?;
            // Commit locals only after all three allocs succeeded.
            self.gpu_codes_buf = Some(codes_buf);
            self.gpu_scales_buf = Some(scales_buf);
            self.gpu_norms_buf = Some(norms_buf);
            self.gpu_words_per_step = words_per_step as i32;
            self.gpu_groups_per_step = groups_per_step as i32;
            self.gpu_capacity = init_cap;
            self.gpu_offset = 0;
            tracing::debug!(
                target: "rmlx::kv_quant::iso3_gpu",
                init_cap,
                words_per_step,
                groups_per_step,
                "iso3 V GPU mirror init"
            );
        } else {
            debug_assert_eq!(
                self.gpu_words_per_step as usize, words_per_step,
                "append_gpu: words_per_step changed across appends — \
                 (B, kv_h, head_dim) must be fixed for the lifetime of a QuantIsoV3"
            );
            debug_assert_eq!(
                self.gpu_groups_per_step as usize, groups_per_step,
                "append_gpu: groups_per_step changed across appends"
            );
        }

        // Guard: silently dropping the mirror on capacity overflow would
        // diverge from CPU blocks. Grow if we still fit under `max_seq`,
        // else surface a typed error.
        let new_offset = prev_seq + s_new as i32;
        if new_offset > self.gpu_capacity {
            // Grow paged. Cap at max_seq when set. `i32::div_ceil` is
            // unstable on the project toolchain (#88581); compute via
            // `usize::div_ceil` instead.
            let needed = new_offset;
            let pages = (needed as usize).div_ceil(KV_PAGE_SIZE as usize) as i32;
            let mut new_cap = pages * KV_PAGE_SIZE;
            if max_seq > 0 {
                new_cap = new_cap.min(max_seq);
            }
            if new_cap < needed {
                return Err(rmlx_core::error::Error::Quant(format!(
                    "append_gpu: needed={needed} > max_seq={max_seq} — caller exceeded \
                     declared cache capacity"
                )));
            }
            let codes_len = (words_per_step * new_cap as usize) as i32;
            let groups_len = (groups_per_step * new_cap as usize) as i32;
            let new_codes_blank = zeros(&[codes_len], Dtype::U32, device)?;
            let new_scales_blank = zeros(&[groups_len], Dtype::F32, device)?;
            let new_norms_blank = zeros(&[groups_len], Dtype::F32, device)?;

            // Copy the active prefix from the old buffers. Borrow `as_ref()`
            // — `slice_update` returns a new Array, the underlying buffer is
            // not consumed, so we can leave `self.gpu_*_buf` populated until
            // the final commit (below) to keep the struct torn-state-safe on
            // any intermediate Err.
            let filled_words = prev_seq * self.gpu_words_per_step;
            let filled_groups = prev_seq * self.gpu_groups_per_step;
            let (Some(old_codes_ref), Some(old_scales_ref), Some(old_norms_ref)) = (
                self.gpu_codes_buf.as_ref(),
                self.gpu_scales_buf.as_ref(),
                self.gpu_norms_buf.as_ref(),
            ) else {
                return Err(rmlx_core::error::Error::Quant(
                    "append_gpu grow: mirror in inconsistent partial-Some state".to_owned(),
                ));
            };
            let new_codes = if filled_words > 0 {
                new_codes_blank.slice_update(
                    &old_codes_ref.slice(&[0], &[filled_words], &[1], device)?,
                    &[0],
                    &[filled_words],
                    &[1],
                    device,
                )?
            } else {
                new_codes_blank
            };
            let new_scales = if filled_groups > 0 {
                new_scales_blank.slice_update(
                    &old_scales_ref.slice(&[0], &[filled_groups], &[1], device)?,
                    &[0],
                    &[filled_groups],
                    &[1],
                    device,
                )?
            } else {
                new_scales_blank
            };
            let new_norms = if filled_groups > 0 {
                new_norms_blank.slice_update(
                    &old_norms_ref.slice(&[0], &[filled_groups], &[1], device)?,
                    &[0],
                    &[filled_groups],
                    &[1],
                    device,
                )?
            } else {
                new_norms_blank
            };
            // Commit grow: all slice_update calls succeeded.
            self.gpu_codes_buf = Some(new_codes);
            self.gpu_scales_buf = Some(new_scales);
            self.gpu_norms_buf = Some(new_norms);
            self.gpu_capacity = new_cap;
            tracing::debug!(
                target: "rmlx::kv_quant::iso3_gpu",
                new_cap,
                "iso3 V GPU mirror grow"
            );
        }

        // ── Write the per-chunk encode outputs into the mirror. ────────────
        let codes_words = (s_new * words_per_step) as i32;
        let groups_slots = (s_new * groups_per_step) as i32;
        let codes_start = prev_seq * self.gpu_words_per_step;
        let codes_stop = codes_start + codes_words;
        let groups_start = prev_seq * self.gpu_groups_per_step;
        let groups_stop = groups_start + groups_slots;

        // Reshape the encode outputs to 1-D so `slice_update` strides line up.
        let codes_1d = codes_gpu.reshape(&[codes_words], device)?;
        let scales_1d = scales_gpu.reshape(&[groups_slots], device)?;
        let norms_1d = norms_gpu.reshape(&[groups_slots], device)?;

        // Borrow current buffers; produce all three new Arrays into locals,
        // then commit. Any Err leaves `self.gpu_*_buf` untouched (only the
        // capacity / offset were mutated above on the grow path, and those
        // are kept consistent with the just-installed buffers).
        let (Some(codes_buf_ref), Some(scales_buf_ref), Some(norms_buf_ref)) = (
            self.gpu_codes_buf.as_ref(),
            self.gpu_scales_buf.as_ref(),
            self.gpu_norms_buf.as_ref(),
        ) else {
            return Err(rmlx_core::error::Error::Quant(
                "append_gpu write: mirror in inconsistent partial-Some state".to_owned(),
            ));
        };
        let new_codes_buf =
            codes_buf_ref.slice_update(&codes_1d, &[codes_start], &[codes_stop], &[1], device)?;
        let new_scales_buf = scales_buf_ref.slice_update(
            &scales_1d,
            &[groups_start],
            &[groups_stop],
            &[1],
            device,
        )?;
        let new_norms_buf =
            norms_buf_ref.slice_update(&norms_1d, &[groups_start], &[groups_stop], &[1], device)?;
        // Commit: all three writes succeeded.
        self.gpu_codes_buf = Some(new_codes_buf);
        self.gpu_scales_buf = Some(new_scales_buf);
        self.gpu_norms_buf = Some(new_norms_buf);
        // gpu_offset is advanced by the caller (append_gpu) so that the
        // torn-state guard there has a single commit point.
        Ok(())
    }

    /// GPU dequant via on-demand `Array::from_bytes` upload.
    ///
    /// Concatenates the per-block CPU payload (codes / scales / quaternions /
    /// per-token norms) into single flat buffers, uploads them to the GPU
    /// **once** via [`Array::from_bytes`] (no intermediate `Vec<f32>`
    /// materialisation of the reconstructed tensor), dispatches
    /// [`crate::isoquant_msl::iso_dequantize_v3_gpu`], and reshapes the flat
    /// f32 output to `[B, kv_h, S, D]`. The returned Array is f32 — callers
    /// that need a different dtype must `astype` afterwards (matches the
    /// existing `dequant() -> Vec<f32>` then `f32_vec_to_array` contract,
    /// which always lands in f32).
    ///
    /// The norm buffer in storage is per-**token** (`n_blocks * n_tokens`
    /// total); the GPU kernel expects per-**group** slots (one per
    /// `(token, group)` pair). This helper expands `norm_per_token` →
    /// `norm_per_group` via `repeat(n_groups)` while copying bytes.
    ///
    /// # Errors
    ///
    /// - `Error::Mlx` if `Array::from_bytes` / kernel dispatch fails.
    /// - `Error::Quant` if `shape` is malformed (rank ≠ 4) or `head_dim` is
    ///   not a positive multiple of [`ISO3_GROUP_SIZE`].
    pub fn dequant_gpu(&self, device: Device) -> Result<Array> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoV3::dequant_gpu: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(ISO3_GROUP_SIZE) {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantIsoV3::dequant_gpu: head_dim={head_dim} must be a positive multiple of \
                 ISO3_GROUP_SIZE={ISO3_GROUP_SIZE}"
            )));
        }
        let n_groups = head_dim / ISO3_GROUP_SIZE;

        // ── GPU mirror fast path ────────────────────────────────────────────
        // When the GPU mirror is populated, slice the active region directly
        // and dispatch the dequant kernel — no `Array::from_bytes` upload of
        // codes/scales/norms, no per-block byte concatenation. The dequant
        // kernel ignores the quaternion buffer (every group uses constant
        // `FIXED_QUAT`); we reuse `codes_slice` for that slot.
        if let (Some(codes_buf), Some(scales_buf), Some(norms_buf)) = (
            self.gpu_codes_buf.as_ref(),
            self.gpu_scales_buf.as_ref(),
            self.gpu_norms_buf.as_ref(),
        ) {
            // Active region matches `shape[2]` (mirror only lives on the
            // active prefix — `truncate_to` lowers `gpu_offset`).
            let s_active = self.shape[2];
            if s_active != self.gpu_offset {
                // Hard guard: a mismatch means the mirror and the CPU blocks
                // diverged (caller mutated `shape[2]` directly, or mixed
                // `append` and `append_gpu` calls). Surfacing a typed error
                // is safer than silently dequanting a stale region.
                return Err(rmlx_core::error::Error::Quant(format!(
                    "dequant_gpu: shape[2]={s_active} != gpu_offset={gpu_off} — \
                     GPU mirror diverged from accumulated shape",
                    gpu_off = self.gpu_offset
                )));
            }
            // Empty cache — return a zero-length array.
            if s_active == 0 {
                return Array::from_bytes(&[][..], &self.shape, Dtype::F32);
            }
            let codes_active = s_active
                .checked_mul(self.gpu_words_per_step)
                .ok_or_else(|| {
                    rmlx_core::error::Error::Quant("dequant_gpu: codes_active overflow".to_owned())
                })?;
            let groups_active =
                s_active
                    .checked_mul(self.gpu_groups_per_step)
                    .ok_or_else(|| {
                        rmlx_core::error::Error::Quant(
                            "dequant_gpu: groups_active overflow".to_owned(),
                        )
                    })?;
            let codes_slice = codes_buf.slice(&[0], &[codes_active], &[1], device)?;
            let scales_slice = scales_buf.slice(&[0], &[groups_active], &[1], device)?;
            let norms_slice = norms_buf.slice(&[0], &[groups_active], &[1], device)?;

            // The dequant kernel reads `_quaternions` purely as a non-consumed
            // input slot (every group uses constant FIXED_QUAT; the buffer
            // pointer is never dereferenced for codes data). Reuse `codes_slice`
            // for that slot to avoid a per-decode `Array::from_bytes`
            // allocation — cheapest option, kernel ignores it.
            let flat = crate::isoquant_msl::iso_dequantize_v3_gpu(
                &codes_slice,
                &scales_slice,
                &codes_slice,
                &norms_slice,
                head_dim,
                Dtype::F32,
                device,
            )?;
            // Mirror/blocks are sequence-major (see `append_gpu`): reshape the
            // flat decode to `[B, S, kv_h, D]`, then reorder heads↔seq back to
            // the logical `[B, kv_h, S, D]`. `contiguous` after the transpose so
            // raw byte-readers (SSD spill) see the permuted bytes.
            let seq_major_shape = [self.shape[0], self.shape[2], self.shape[1], self.shape[3]];
            let out = flat.reshape(&seq_major_shape, device)?;
            return out.transpose(&[0, 2, 1, 3], device)?.contiguous(device);
        }

        // Concatenate all block buffers. CPU codes/scales/quats are already in
        // GPU-kernel layout; per-token norms must be expanded to per-group.
        let mut codes_bytes: Vec<u8> = Vec::new();
        let mut scales_bytes: Vec<u8> = Vec::new();
        let mut quats_bytes: Vec<u8> = Vec::new();
        let mut norms_bytes: Vec<u8> = Vec::new();
        let mut total_groups: usize = 0;

        for blk in &self.blocks {
            for &c in &blk.codes {
                codes_bytes.extend_from_slice(&c.to_le_bytes());
            }
            for &s in &blk.scales {
                scales_bytes.extend_from_slice(&s.to_le_bytes());
            }
            for &q in &blk.quaternions {
                quats_bytes.extend_from_slice(&q.to_le_bytes());
            }
            // Per-token → per-group expand (kernel reads norms_in[gid] where
            // gid = token * n_groups + grp; all groups in a token share the
            // same norm value).
            for &n in &blk.norms {
                let n_bytes = n.to_le_bytes();
                for _ in 0..n_groups {
                    norms_bytes.extend_from_slice(&n_bytes);
                }
            }
            // Loudly fail on overflow rather than silently saturating; the
            // resulting buffer-vs-shape mismatch would otherwise surface as a
            // confusing kernel-dispatch error downstream.
            let blk_groups = blk.n_tokens.checked_mul(n_groups).ok_or_else(|| {
                rmlx_core::error::Error::Quant(
                    "dequant_gpu: blk.n_tokens * n_groups overflow".to_owned(),
                )
            })?;
            total_groups = total_groups.checked_add(blk_groups).ok_or_else(|| {
                rmlx_core::error::Error::Quant("dequant_gpu: total_groups overflow".to_owned())
            })?;
        }

        // Guard against silent shape divergence between `dequant_gpu`
        // (derives total from concatenated blocks) and `dequant` (pads/truncates
        // to `prod(shape)`). After `truncate_to` / `reset` / post-hydrate edge
        // cases the two totals can diverge — refuse to silently reshape rather
        // than producing a garbage Array.
        let declared_total: usize = self.shape.iter().map(|&d| d as usize).product();
        let actual_total: usize = total_groups.checked_mul(ISO3_GROUP_SIZE).ok_or_else(|| {
            rmlx_core::error::Error::Quant(
                "dequant_gpu: total_groups * ISO3_GROUP_SIZE overflow".to_owned(),
            )
        })?;
        if actual_total != declared_total {
            return Err(rmlx_core::error::Error::Quant(format!(
                "dequant_gpu: actual_total={actual_total} (blocks×groups×group_size) != \
                 declared_total={declared_total} (prod(shape)={:?}); refusing to silently \
                 truncate/pad",
                self.shape
            )));
        }

        if total_groups == 0 {
            // Empty cache — return a zero-length array of the correct rank-4 shape.
            // (Matches the `dequant()` empty-output behaviour with shape[2] = 0.)
            // The MEDIUM-1 guard above ensures `prod(shape) == 0` here, so the
            // empty `from_bytes` call is well-formed.
            return Array::from_bytes(&[][..], &self.shape, Dtype::F32);
        }

        let codes_arr = Array::from_bytes(
            &codes_bytes,
            &[total_groups as i32], // WORDS_PER_GROUP=1 for ISO3_GS=4
            Dtype::U32,
        )?;
        let scales_arr = Array::from_bytes(&scales_bytes, &[total_groups as i32], Dtype::F32)?;
        let quats_arr = Array::from_bytes(&quats_bytes, &[(total_groups * 4) as i32], Dtype::F32)?;
        let norms_arr = Array::from_bytes(&norms_bytes, &[total_groups as i32], Dtype::F32)?;

        let flat = crate::isoquant_msl::iso_dequantize_v3_gpu(
            &codes_arr,
            &scales_arr,
            &quats_arr,
            &norms_arr,
            head_dim,
            Dtype::F32,
            device,
        )?;

        // CPU blocks are sequence-major (see `append` / `append_gpu`): reshape
        // to `[B, S, kv_h, D]`, then reorder heads↔seq back to `[B, kv_h, S, D]`.
        let seq_major_shape = [self.shape[0], self.shape[2], self.shape[1], self.shape[3]];
        let out = flat.reshape(&seq_major_shape, device)?;
        out.transpose(&[0, 2, 1, 3], device)?.contiguous(device)
    }
}

#[cfg(test)]
#[path = "quant_iso_v_tests.rs"]
mod quant_iso_v_tests;
