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

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError, FIXED_QUAT};

use super::{iso_n_groups_for, QuantKGpuRing, KV_PAGE_SIZE};

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

impl super::BlockRows for IsoBlocks {
    fn rows(&self) -> usize {
        self.n_tokens
    }

    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile, which is what stops a
    /// buffer from surviving a mid-block truncation at its full length.
    fn retain_rows(&mut self, rows: usize) -> bool {
        let Self {
            codes,
            scales,
            quaternions,
            norms,
            n_tokens,
        } = self;
        let lengths = [codes.len(), scales.len(), quaternions.len(), norms.len()];
        if !super::rows_split_ok(&lengths, *n_tokens, rows) {
            return false;
        }
        super::retain_rows_in(codes, *n_tokens, rows);
        super::retain_rows_in(scales, *n_tokens, rows);
        super::retain_rows_in(quaternions, *n_tokens, rows);
        super::retain_rows_in(norms, *n_tokens, rows);
        *n_tokens = rows;
        true
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
    /// GPU-resident packed ring for the fused symmetric flash-decode path.
    /// Empty until the first [`Self::gpu_append`]. Distinct from the bespoke
    /// `gpu_*_buf` mirror below (which is gated off in production and serves
    /// `dequant_gpu`): the ring stores per-token norms in the sequence-major
    /// `(codes, scales, norms)` form the flash kernel reads, mirroring the K
    /// store and the rotor V store. When live it is the SOLE resident copy —
    /// the CPU `blocks` are dropped and rebuilt on demand via
    /// [`synced_iso_v_blocks`].
    pub gpu: QuantKGpuRing,
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
            .field("gpu_resident", &self.gpu.is_allocated())
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
            gpu: QuantKGpuRing::default(),
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
            return Err(Error::Mlx(format!(
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
            iso_encode_fast(&seq_major, head_dim, ISO3_GROUP_SIZE, ISO3_BITS)
                .map_err(|e: IsoQuantError| Error::Mlx(format!("iso3 encode: {e}")))?;

        self.blocks.push(IsoBlocks {
            codes,
            scales,
            quaternions,
            norms,
            n_tokens: n_tokens_total,
        });

        // A CPU append does not touch the GPU ring, so any live ring is now a
        // stale prefix. Drop it; the next `gpu_append` re-seeds from `blocks`.
        self.gpu.clear();

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
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: QuantKGpuRing::default(),
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
        // The ring would otherwise keep a prefix the CPU blocks no longer
        // claim. Dropped here; the next `gpu_append` re-seeds from `blocks`.
        self.gpu.clear();
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
    /// Drops trailing blocks past `n` and **splits** the block the cut lands
    /// inside (block `n_tokens` counts rows, not sequence positions — see
    /// [`super::truncate_plan`]), then lowers `shape[2]` to `n`.
    ///
    /// The GPU ring is **kept**, not cleared — mirror of the rotor V store's
    /// `truncate_to`. Lowering `shape[2]` to `n` makes the ring's logical fill
    /// `n`; the stale `[n, prev)` capacity is overwritten by the next append and
    /// never read (`packed_view` slices to `shape[2]`). This preserves any
    /// ring-only decode tail up to `n`, so `dequant` / an SSD spill can still
    /// rebuild it via [`synced_iso_v_blocks`]. Clearing the ring here would
    /// discard the tail (the only copy of `[frozen_prefix, n)`), leaving `blocks`
    /// short of `shape[2]` with no ring — the divergent state `dequant` rejects
    /// loudly, which would abort generation on the speculative-decode rollback
    /// path.
    pub fn truncate_to(&mut self, n: i32) {
        let n = n.max(0);
        let plan = super::truncate_plan(
            self.blocks.iter().map(super::BlockRows::rows),
            &self.shape,
            n,
        );
        super::apply_truncate_plan(&mut self.blocks, &plan);
        // NB: no `self.gpu.clear()` — the ring is the source of truth for a
        // ring-only decode tail; see the doc comment above.
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
        // Lower the GPU mirror offset; underlying buffer untouched (matches
        // `QuantV::truncate` semantics — the trailing tokens become logically
        // free and the next `append_gpu` overwrites them via `slice_update`).
        // `n` is already clamped at the top of this function.
        if self.gpu_offset > n {
            self.gpu_offset = n;
        }
    }

    /// Deep-clone.
    ///
    /// Materialises any ring-only decode tail into complete CPU blocks first:
    /// the clone starts CPU-only (the ring is not cloned), and both the
    /// prompt-cache snapshot and the SSD spill clone route through here, so this
    /// is the single point where a store leaving the live decode loop reconciles
    /// its blocks with the ring. A short-blocks clone with no ring would silently
    /// truncate the store — refused loudly by [`synced_iso_v_blocks`] instead.
    ///
    /// # Errors
    ///
    /// Forwards a [`synced_iso_v_blocks`] reconciliation error (blocks over-run
    /// `shape[2]`, or a ring-only tail exists but the ring is absent / too short).
    pub fn try_deep_clone(&self) -> Result<Self> {
        let blocks =
            synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?.into_owned();
        Ok(Self {
            // The clone starts CPU-only: `blocks` carries the full payload, so
            // the ring re-seeds from them on the clone's first GPU append.
            // Sharing the source's Arrays would alias one ring across two
            // independent caches.
            gpu: QuantKGpuRing::default(),
            blocks,
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

    /// Concatenate the accumulated CPU blocks into flat sequence-major
    /// `(codes, scales, norms)` — the form [`QuantKGpuRing::seed_from_cpu`]
    /// wants. The per-group quaternion table is dropped: it is the fixed
    /// golden-ratio constant on every group, which the ring never carries.
    fn flatten_blocks(&self) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        // Exact capacities — the prefill seed concatenates the whole prefix
        // (millions of entries at long context), so growing from empty would
        // realloc+memcpy repeatedly.
        let (n_codes, n_scales, n_norms) = self.blocks.iter().fold((0, 0, 0), |(c, s, n), blk| {
            (
                c + blk.codes.len(),
                s + blk.scales.len(),
                n + blk.norms.len(),
            )
        });
        let mut codes = Vec::with_capacity(n_codes);
        let mut scales = Vec::with_capacity(n_scales);
        let mut norms = Vec::with_capacity(n_norms);
        for blk in &self.blocks {
            codes.extend_from_slice(&blk.codes);
            scales.extend_from_slice(&blk.scales);
            norms.extend_from_slice(&blk.norms);
        }
        (codes, scales, norms)
    }

    /// Push one GPU-encoded chunk into the GPU ring, seeding the ring from the
    /// accumulated CPU blocks first when it is not yet live.
    ///
    /// `codes` / `scales` / `norms` are the iso encode kernel's GPU outputs for
    /// a **sequence-major** chunk; `norms` must already be per-token. `prev_seq`
    /// is the accumulated sequence length **before** this chunk.
    ///
    /// This only maintains the ring — a ring-only-tail caller then drops the CPU
    /// blocks (`drop_blocks_when_ring_live`), which are rebuilt on demand from
    /// the ring at `dequant()` / SSD-spill boundaries.
    ///
    /// `max_seq` is the window the cache is provisioned for **right now**, read
    /// from the active `KvStorage` variant by the caller. It is a parameter
    /// rather than a field so it cannot go stale as the window grows during
    /// decode — the same contract the K-side ring uses.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Quant`] when `head_dim` yields no quaternion groups, and
    /// forwards [`QuantKGpuRing::seed_from_cpu`] / [`QuantKGpuRing::append_encoded`]
    /// errors.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_append(
        &mut self,
        codes: &Array,
        scales: &Array,
        norms: &Array,
        kv_h: i32,
        head_dim: i32,
        prev_seq: i32,
        new_seq: i32,
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        let n_groups = i32::try_from(iso_n_groups_for(
            usize::try_from(head_dim.max(0)).unwrap_or(0),
        ))
        .map_err(|_| {
            Error::Quant(format!(
                "QuantIsoV3::gpu_append: n_groups for head_dim={head_dim} exceeds i32::MAX"
            ))
        })?;
        if n_groups <= 0 {
            return Err(Error::Quant(format!(
                "QuantIsoV3::gpu_append: head_dim={head_dim} yields no quaternion groups"
            )));
        }
        if !self.gpu.is_allocated() && prev_seq > 0 {
            let (c, s, n) = self.flatten_blocks();
            self.gpu
                .seed_from_cpu(&c, &s, &n, kv_h, n_groups, prev_seq, max_seq, device)?;
        }
        self.gpu.append_encoded(
            codes, scales, norms, kv_h, n_groups, prev_seq, new_seq, max_seq, device,
        )
    }

    /// Rebuild any ring-only prefix into the CPU blocks so `blocks` alone cover
    /// `shape[2]` again, and either keep or drop the ring.
    ///
    /// No-op when the ring was never allocated. Every path that pushes a CPU
    /// block onto a store whose ring may be live goes through here: without it
    /// the pushed block is the *only* block, `blocks` no longer cover
    /// `shape[2]`, and the ring left behind is stale. A *read* of that state is
    /// refused by the ring's fill watermark; an *append* onto it is refused by
    /// [`QuantKGpuRing::append_encoded`]'s `prev_seq > filled` guard, because
    /// the write would otherwise zero `[filled, prev_seq)` and then commit a
    /// watermark that covers it. Both directions have to refuse, or the state
    /// is only unreachable by convention.
    ///
    /// `disposition` is a parameter because it is the one thing the two callers
    /// disagree on, and having them disagree by carrying separate copies of this
    /// body is how the two drift apart. [`Self::append_gpu`] drops the ring — it
    /// does not feed it, so leaving it live is the dangerous state the CPU
    /// [`Self::append`] avoids by clearing. The append helpers in `kvcache`
    /// keep it, because their `sync_ring` decides its fate immediately after.
    ///
    /// # Errors
    ///
    /// Forwards a [`synced_iso_v_blocks`] reconciliation error — a ring that
    /// cannot supply the missing prefix is reported, never zero-padded.
    pub(crate) fn reconcile_ring(
        &mut self,
        device: Device,
        disposition: super::RingDisposition,
    ) -> Result<()> {
        if !self.gpu.is_allocated() {
            return Ok(());
        }
        if let std::borrow::Cow::Owned(full) =
            synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, device)?
        {
            self.blocks = full;
        }
        if disposition == super::RingDisposition::Drop {
            self.gpu.clear();
        }
        Ok(())
    }

    /// GPU packed view of the first `kv_seq` positions, or `None` when the ring
    /// is not live (CPU path — caller falls back to `dequant`).
    ///
    /// # Errors
    ///
    /// Forwards [`QuantKGpuRing::packed_view`] errors.
    pub fn gpu_packed_view(
        &self,
        kv_seq: i32,
        device: Device,
    ) -> Result<Option<(Array, Array, Array)>> {
        self.gpu.packed_view(kv_seq, device)
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
            gpu,
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
            + gpu.byte_size()
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
        // The ring lives on the GPU whenever it is live at all, so that is the
        // stream its readback belongs on. `dequant_on` exists so a caller that
        // knows better — a `Device::Cpu` run, or a test that must not touch a
        // shared Metal context — can say so instead of having this constant
        // imposed on it.
        self.dequant_on(Device::Gpu)
    }

    /// [`Self::dequant`] on an explicit device.
    ///
    /// The device selects the stream for the ring readback that reconciles a
    /// ring-only decode tail; it has no effect on a store whose CPU blocks
    /// already cover `shape[2]`, which is every store on the CPU append path.
    ///
    /// # Errors
    ///
    /// Same as [`Self::dequant`].
    pub fn dequant_on(&self, device: Device) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoV3::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile the CPU blocks with the GPU ring: on the fused symmetric
        // decode path the decode tail lives only in the ring (`blocks` trail
        // `shape[2]`), and this rebuilds it on demand rather than decoding a
        // short prefix and zero-padding the gap. Loud on any unrecoverable
        // disagreement.
        let blocks = synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, device)?;

        if blocks.is_empty() {
            // Empty blocks are valid only when the store genuinely holds no
            // tokens (shape[2] == 0 => total_elems == 0). A non-zero declared
            // shape with no blocks means the ring-only decode tail was lost and
            // `synced_iso_v_blocks` could not rebuild it — reject loudly rather
            // than fabricate a zeroed prefix.
            if total_elems != 0 {
                return Err(Error::Mlx(format!(
                    "QuantIsoV3::dequant: no blocks but shape {:?} implies {total_elems} elems — \
                     refusing to zero-pad a lost decode tail",
                    self.shape
                )));
            }
            return Ok(out);
        }

        for blk in blocks.iter() {
            let dec = iso_decode_fast(
                &blk.codes,
                &blk.scales,
                &blk.quaternions,
                &blk.norms,
                head_dim,
                ISO3_GROUP_SIZE,
                ISO3_BITS,
            )
            .map_err(|e: IsoQuantError| Error::Mlx(format!("iso3 decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // `synced_iso_v_blocks` guarantees the blocks cover `shape[2]`, so a
        // length mismatch here is an internal invariant break — surface it
        // loudly rather than zero-padding or truncating a decoded prefix.
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantIsoV3::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
                 refusing to zero-pad / truncate",
                out.len(),
                self.shape
            )));
        }
        // Blocks are sequence-major (see `append`), one per append; reorder each
        // at its own sequence offset back to the logical head-major
        // `[B, kv_h, S, D]`. Reading the concatenation as a single run would
        // interleave batch elements once `B > 1`.
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let out = super::seq_layout::transpose_chunked_seq_heads(
            &out,
            b,
            s,
            kv_h,
            head_dim,
            blocks.iter().map(super::BlockRows::rows),
        )?;
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
            return Err(Error::Mlx(format!(
                "QuantIsoV3::append_gpu: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let b = new_shape[0] as usize;
        let kv_h = new_shape[1] as usize;
        let s_new = new_shape[2] as usize;
        let head_dim = new_shape[3] as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(ISO3_GROUP_SIZE) {
            return Err(Error::Quant(format!(
                "QuantIsoV3::append_gpu: head_dim={head_dim} must be a positive multiple of \
                 ISO3_GROUP_SIZE={ISO3_GROUP_SIZE}"
            )));
        }
        let n_groups = head_dim / ISO3_GROUP_SIZE;
        let n_tokens_total = b
            .checked_mul(kv_h)
            .and_then(|v| v.checked_mul(s_new))
            .ok_or_else(|| {
                Error::Quant("QuantIsoV3::append_gpu: n_tokens_total overflow".to_owned())
            })?;

        // ── 0. Take the ring's prefix back, then drop the ring. ────────────
        // This append pushes a CPU block, which makes `blocks` the authoritative
        // copy of everything accumulated so far. A live ring holds a prefix
        // `blocks` may no longer carry — the fused decode path drops the blocks
        // once the ring is live — so the prefix has to come back before the push
        // or `dequant` / `dequant_gpu` see a store whose blocks cover only this
        // chunk. Dropping the ring afterwards is the same contract the CPU
        // `append` states: a ring left behind while `blocks` grow is the
        // dangerous state, because the next ring append would write past its
        // filled region and `dequant` would read a zeroed gap. The next
        // `gpu_append` re-seeds it from `blocks`.
        self.reconcile_ring(device, super::RingDisposition::Drop)?;

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
            .ok_or_else(|| Error::Quant("append_gpu: words_per_step overflow".to_owned()))?;
        let groups_per_step = b
            .checked_mul(kv_h)
            .and_then(|v| v.checked_mul(n_groups))
            .ok_or_else(|| Error::Quant("append_gpu: groups_per_step overflow".to_owned()))?;

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
                return Err(Error::Quant(format!(
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
                return Err(Error::Quant(
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
            return Err(Error::Quant(
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
            return Err(Error::Mlx(format!(
                "QuantIsoV3::dequant_gpu: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(ISO3_GROUP_SIZE) {
            return Err(Error::Quant(format!(
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
                return Err(Error::Quant(format!(
                    "dequant_gpu: shape[2]={s_active} != gpu_offset={gpu_off} — \
                     GPU mirror diverged from accumulated shape",
                    gpu_off = self.gpu_offset
                )));
            }
            // Empty cache — return a zero-length array.
            if s_active == 0 {
                return Array::from_bytes(&[][..], &self.shape, Dtype::F32);
            }
            // The mirror is one flat buffer written per chunk at
            // `prev_seq * words_per_step`, and `words_per_step` folds `b` in, so
            // the prefix is a run of `[B, S_chunk, kv_h, D]` chunks. Reading it
            // as one `[B, S, kv_h, D]` run below interleaves batch elements once
            // `B > 1` and more than one chunk landed. Refuse rather than return a
            // scrambled tensor — the block path handles every `B`.
            if self.shape[0] != 1 && self.shape[2] > 1 {
                return Err(Error::Quant(format!(
                    "QuantIsoV3::dequant_gpu: the GPU mirror is b == 1 only (its per-step \
                     stride does not interleave batch), got shape {:?}",
                    self.shape
                )));
            }
            let codes_active = s_active
                .checked_mul(self.gpu_words_per_step)
                .ok_or_else(|| Error::Quant("dequant_gpu: codes_active overflow".to_owned()))?;
            let groups_active = s_active
                .checked_mul(self.gpu_groups_per_step)
                .ok_or_else(|| Error::Quant("dequant_gpu: groups_active overflow".to_owned()))?;
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

        // Reconcile the CPU blocks with the GPU ring first, exactly as
        // `dequant` does. Both element counts below — the one derived from the
        // blocks and the one declared by `shape` — must come from the same
        // source, or a store whose decode tail lives only in the ring is
        // reported as a shape disagreement instead of being read. `blocks` that
        // already cover `shape[2]` are borrowed, so the common path pays
        // nothing.
        let blocks = synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, device)?;

        // Build the kernel inputs with the token rows already in head-major
        // order, so the flat result *is* `[B, kv_h, S, D]` and needs no
        // reshape-plus-transpose. Reshaping the whole concatenated decode as one
        // `[B, S, kv_h, D]` run — what this used to do — is only a valid reading
        // at `B == 1`, because each block covers only `[B, S_block, kv_h, D]`.
        let inputs = iso_kernel_inputs_head_major(
            &blocks,
            &self.shape,
            n_groups,
            ISO3_GROUP_SIZE,
            "QuantIsoV3::dequant_gpu",
        )?;

        if inputs.total_groups == 0 {
            // Empty cache — the accounting above already proved `prod(shape)`
            // is 0, so the empty `from_bytes` is well-formed.
            return Array::from_bytes(&[][..], &self.shape, Dtype::F32);
        }

        let n = inputs.total_groups as i32;
        // WORDS_PER_GROUP = 1 for ISO3_GROUP_SIZE = 4.
        let codes_arr = Array::from_bytes(&inputs.codes, &[n], Dtype::U32)?;
        let scales_arr = Array::from_bytes(&inputs.scales, &[n], Dtype::F32)?;
        let quats_arr = Array::from_bytes(&inputs.quaternions, &[n * 4], Dtype::F32)?;
        let norms_arr = Array::from_bytes(&inputs.norms, &[n], Dtype::F32)?;

        let flat = crate::isoquant_msl::iso_dequantize_v3_gpu(
            &codes_arr,
            &scales_arr,
            &quats_arr,
            &norms_arr,
            head_dim,
            Dtype::F32,
            device,
        )?;
        flat.reshape(&self.shape, device)
    }
}

/// Bytes per little-endian payload slot — every iso payload is 4-byte (`u32`
/// codes, `f32` scales / quaternions / norms).
const ISO_SLOT_BYTES: usize = 4;

/// Flat kernel inputs for an iso store, with token rows in head-major order.
///
/// Field lengths, in `(token, group)` slots: `codes` and `scales` carry one
/// entry per slot, `quaternions` four, `norms` one (the per-token norm expanded
/// to per-group, which is what the kernel indexes).
#[derive(Debug)]
pub(crate) struct IsoKernelInputs {
    /// Packed codes, little-endian `u32`.
    pub codes: Vec<u8>,
    /// Per-group scales, little-endian `f32`.
    pub scales: Vec<u8>,
    /// Per-group quaternions (4 components), little-endian `f32`.
    pub quaternions: Vec<u8>,
    /// Per-group norms, little-endian `f32`.
    pub norms: Vec<u8>,
    /// `(token, group)` slot count — the kernel's flat length.
    pub total_groups: usize,
}

/// Build the iso dequant kernel's flat inputs from a block list, placing every
/// token row at its **head-major** `[B, kv_h, S, D]` position.
///
/// One derivation for both bit widths and both readers. Two things used to be
/// written out per store and drifted apart:
///
/// * The declared-vs-actual element accounting. `prod(shape)` against
///   `blocks × groups × group_size` was copied per reader, so the same store
///   could be accepted by one and rejected by the other.
/// * The row order. Concatenating the blocks and reshaping the kernel's flat
///   output as one `[B, S, kv_h, D]` run is only a valid reading at `B == 1`:
///   each block is `[B, S_block, kv_h, D]`, so at `B > 1` the blocks interleave
///   and the reshape maps a later block's batch-0 rows onto batch-1 slots.
///   Permuting the *input* rows instead is exact at every `B` and costs nothing
///   — the payload is being copied either way, the kernel is per-token
///   positional, and the result needs no reshape-plus-transpose afterwards.
///
/// # Errors
///
/// Returns [`Error::Quant`] on a malformed shape, a block whose payload length
/// disagrees with its row count, chunks that do not partition `shape[2]`, or a
/// blocks-vs-shape element-count disagreement.
pub(crate) fn iso_kernel_inputs_head_major(
    blocks: &[IsoBlocks],
    shape: &[i32],
    n_groups: usize,
    group_size: usize,
    what: &str,
) -> Result<IsoKernelInputs> {
    if shape.len() != 4 {
        return Err(Error::Quant(format!("{what}: malformed shape {shape:?}")));
    }
    let dim = |i: usize| -> usize { shape.get(i).copied().unwrap_or(0).max(0) as usize };
    let (b, kv_h, s_total) = (dim(0), dim(1), dim(2));

    // Accounting, derived once. `declared` is what the shape claims; `actual` is
    // what the blocks carry. A disagreement is refused rather than reshaped.
    let declared: usize = shape.iter().map(|&d| d.max(0) as usize).product();
    let mut total_groups: usize = 0;
    for blk in blocks {
        let blk_groups = super::BlockRows::rows(blk)
            .checked_mul(n_groups)
            .ok_or_else(|| Error::Quant(format!("{what}: rows * n_groups overflow")))?;
        total_groups = total_groups
            .checked_add(blk_groups)
            .ok_or_else(|| Error::Quant(format!("{what}: total_groups overflow")))?;
    }
    let actual = total_groups
        .checked_mul(group_size)
        .ok_or_else(|| Error::Quant(format!("{what}: total_groups * group_size overflow")))?;
    if actual != declared {
        return Err(Error::Quant(format!(
            "{what}: actual_total={actual} (blocks×groups×group_size) != \
             declared_total={declared} (prod(shape)={shape:?}); refusing to silently \
             truncate/pad"
        )));
    }

    let perm = super::seq_layout::head_major_token_order(
        b,
        s_total,
        kv_h,
        blocks.iter().map(super::BlockRows::rows),
    )
    .map_err(|e| Error::Quant(format!("{what}: {e}")))?;

    let mut out = IsoKernelInputs {
        codes: vec![0_u8; total_groups * ISO_SLOT_BYTES],
        scales: vec![0_u8; total_groups * ISO_SLOT_BYTES],
        quaternions: vec![0_u8; total_groups * 4 * ISO_SLOT_BYTES],
        norms: vec![0_u8; total_groups * ISO_SLOT_BYTES],
        total_groups,
    };

    let mut row = 0usize;
    for blk in blocks {
        let rows = super::BlockRows::rows(blk);
        // The scatter below indexes each payload by row; a length that is not a
        // whole number of rows would silently read across a row boundary.
        let want_codes = rows * n_groups;
        if blk.codes.len() != want_codes
            || blk.scales.len() != want_codes
            || blk.quaternions.len() != want_codes * 4
            || blk.norms.len() != rows
        {
            return Err(Error::Quant(format!(
                "{what}: block payload disagrees with its {rows} rows at n_groups={n_groups} \
                 (codes {} scales {} quaternions {} norms {}; want {want_codes} {want_codes} {} \
                 {rows})",
                blk.codes.len(),
                blk.scales.len(),
                blk.quaternions.len(),
                blk.norms.len(),
                want_codes * 4,
            )));
        }
        for r in 0..rows {
            let Some(&dst_token) = perm.get(row + r) else {
                return Err(Error::Quant(format!(
                    "{what}: block rows exceed the permutation length {}",
                    perm.len()
                )));
            };
            copy_f32_slots(
                &mut out.codes,
                dst_token * n_groups,
                &blk.codes[r * n_groups..(r + 1) * n_groups],
                u32::to_le_bytes,
            );
            copy_f32_slots(
                &mut out.scales,
                dst_token * n_groups,
                &blk.scales[r * n_groups..(r + 1) * n_groups],
                f32::to_le_bytes,
            );
            copy_f32_slots(
                &mut out.quaternions,
                dst_token * n_groups * 4,
                &blk.quaternions[r * n_groups * 4..(r + 1) * n_groups * 4],
                f32::to_le_bytes,
            );
            // Per-token → per-group expand: the kernel reads `norms[gid]` with
            // `gid = token * n_groups + grp`, and every group in a token shares
            // the token's norm.
            let n_bytes = blk.norms[r].to_le_bytes();
            let base = dst_token * n_groups * ISO_SLOT_BYTES;
            for g in 0..n_groups {
                let at = base + g * ISO_SLOT_BYTES;
                out.norms[at..at + ISO_SLOT_BYTES].copy_from_slice(&n_bytes);
            }
        }
        row += rows;
    }
    Ok(out)
}

/// Write `src` into `dst` as little-endian 4-byte slots starting at slot
/// `slot_offset`. Shared by the code / scale / quaternion scatters above.
#[allow(
    clippy::indexing_slicing,
    reason = "dst is sized total_groups slots and slot_offset + src.len() is bounded by the \
              permutation's range, which head_major_token_order pins to that same count"
)]
fn copy_f32_slots<T: Copy>(dst: &mut [u8], slot_offset: usize, src: &[T], to_le: fn(T) -> [u8; 4]) {
    for (i, &v) in src.iter().enumerate() {
        let at = (slot_offset + i) * 4;
        dst[at..at + 4].copy_from_slice(&to_le(v));
    }
}

/// Reconcile an iso-V store's CPU `blocks` with its GPU ring so the returned
/// slice covers the full accumulated `shape[2]`.
///
/// On the fused symmetric decode path the per-step CPU block download is skipped
/// — the GPU ring is the source of truth for the decode tail, and `blocks` trail
/// `shape[2]` (a **ring-only tail**). This rebuilds the missing prefix from the
/// ring on demand — the single point where a block consumer (`dequant`, or the
/// SSD spill via `try_deep_clone`) reconciles the two. When `blocks` already
/// cover `shape[2]` (the CPU append and SSD-hydrate paths, and every V-only iso
/// cache, which never feeds the ring) the borrow is returned untouched, so those
/// paths pay no GPU readback.
///
/// The rebuilt block's per-group quaternion table is synthesised as the fixed
/// golden-ratio constant on every group — exactly what `iso_encode_fast` writes
/// and the ring drops (see [`crate::isoquant::FIXED_QUAT`]).
///
/// **Invariant (enforced loudly, never zero-padded):** `blocks` track the ring
/// exactly, or the ring exists and supplies the tail. Any state where the CPU
/// blocks fall short of `shape[2]` and the ring cannot make up the difference is
/// an `Error` — the caller must not fabricate a zeroed gap. The enforcement is
/// [`QuantKGpuRing`]'s fill watermark: the readback is sized from `shape[2]` and
/// refused when that runs past what the ring actually wrote. Without it the
/// ring's page-rounded `capacity` accepted the read and the caller got the
/// allocation's zeros as a K/V tail — length-correct and silently wrong.
///
/// **What it does not cover:** when `blocks` fall short the ring supplies the
/// *whole* prefix and the existing `blocks` are discarded, not merged. That is
/// right for the state this exists for — on the fused decode path the ring is a
/// superset of the blocks — but a caller holding a block the ring never saw
/// would lose it with no error. No such caller exists today: every block push on
/// these stores either feeds the ring or drops it first.
///
/// Shared by [`QuantIsoV3`] / [`super::QuantIsoV4`] and the iso K stores
/// ([`super::QuantIsoK3`] / [`super::QuantIsoK4`]) — the `IsoBlocks` payload and
/// ring layout are identical across both axes and both bit widths (iso carries
/// no K-side sideband, unlike the rotor codec). Mirror of the rotor-side
/// `synced_rotor_v_blocks`, plus the fixed-quaternion synthesis.
///
/// # Errors
///
/// Returns [`Error::Quant`] on a malformed shape, when the blocks over-run
/// `shape[2]`, or when a ring-only tail exists but the ring is absent / too
/// short to cover it.
pub(crate) fn synced_iso_v_blocks<'a>(
    blocks: &'a [IsoBlocks],
    shape: &[i32],
    gpu: &QuantKGpuRing,
    device: Device,
) -> Result<std::borrow::Cow<'a, [IsoBlocks]>> {
    if shape.len() != 4 {
        return Err(Error::Quant(format!(
            "synced_iso_v_blocks: malformed shape {shape:?}"
        )));
    }
    let b = shape.first().copied().unwrap_or(0).max(0) as usize;
    let kv_h = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    let full_seq = shape.get(2).copied().unwrap_or(0).max(0) as usize;
    let head_dim = shape.get(3).copied().unwrap_or(0).max(0) as usize;
    let full_tokens = b * kv_h * full_seq;
    let blocks_tokens: usize = blocks.iter().map(super::BlockRows::rows).sum();

    if blocks_tokens == full_tokens {
        return Ok(std::borrow::Cow::Borrowed(blocks));
    }
    if blocks_tokens > full_tokens {
        return Err(Error::Quant(format!(
            "iso V store: CPU blocks hold {blocks_tokens} tokens but shape[2] implies \
             {full_tokens} — blocks over-run the accumulated shape (internal invariant)"
        )));
    }

    // Ring-only tail: the GPU ring must supply the whole prefix. It is
    // sequence-major and stores per-token norms already, so the readback is one
    // block covering `[0, full_seq)`. Refuse to fabricate a zeroed gap.
    let seq_i32 = i32::try_from(full_seq)
        .map_err(|_| Error::Quant(format!("iso V store: shape[2]={full_seq} exceeds i32::MAX")))?;
    let Some((codes, scales, norms)) = gpu.packed_view_cpu(seq_i32, device)? else {
        return Err(Error::Quant(format!(
            "iso V store: CPU blocks cover {blocks_tokens} tokens but shape[2] needs \
             {full_tokens} and the GPU ring is absent — refusing to zero-pad the decode tail"
        )));
    };
    let n_groups = iso_n_groups_for(head_dim);
    let want_codes = full_tokens * n_groups;
    if codes.len() != want_codes || scales.len() != want_codes || norms.len() != full_tokens {
        return Err(Error::Quant(format!(
            "iso V store: ring readback size mismatch (codes {} scales {} norms {}, \
             want codes/scales {want_codes} norms {full_tokens}) — cannot rebuild blocks",
            codes.len(),
            scales.len(),
            norms.len(),
        )));
    }
    // The ring drops the per-group quaternion table (it is the fixed constant on
    // every group); rebuild it so `iso_decode_fast` reads the same rotation the
    // encoder wrote.
    let mut quaternions = Vec::with_capacity(want_codes * FIXED_QUAT.len());
    for _ in 0..want_codes {
        quaternions.extend_from_slice(&FIXED_QUAT);
    }
    Ok(std::borrow::Cow::Owned(vec![IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens: full_tokens,
    }]))
}

#[cfg(test)]
#[path = "quant_iso_v_tests.rs"]
mod quant_iso_v_tests;
