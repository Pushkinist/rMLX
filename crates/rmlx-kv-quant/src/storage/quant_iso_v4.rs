// IsoQuant 4-bit V storage struct.
//
// Mirrors `quant_iso_v.rs` (`QuantIsoV3`) one-for-one with `bits=4` baked in
// and the dense 8-vals-per-u32 pack from the parameterized
// `iso_encode_fast`/`iso_decode_fast` codec.
//
// Step 1 decision (parameterize vs fork): the encode/decode functions in
// `crate::isoquant` are parameterized over `bits ∈ {3, 4}`; the storage layer
// is forked because `QuantIsoV3` is name-stable across crates (SSD writer/
// reader, helpers, dispatch). Renaming to a generic `QuantIsoV` would cause
// large cross-crate churn for no benefit — bits is fixed per storage variant.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized V buffer: `QuantIsoV4` (IsoQuant 4-bit V codec).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};
use crate::storage::quant_iso_v::{synced_iso_v_blocks, IsoBlocks};

use super::{iso_n_groups_for, QuantKGpuRing};

/// Bit-width of the iso4 V codec (fixed at 4-bit).
pub const ISO4_BITS: u8 = 4;

/// Quaternion-block size for the iso4 codec (fixed at 4 elements / group; one
/// quaternion per group in fast mode — identical to iso3).
pub const ISO4_GROUP_SIZE: usize = 4;

/// Accumulated IsoQuant V cache (4-bit, quaternion SO(4) fast mode).
///
/// Same per-block payload as [`crate::storage::QuantIsoV3`] — `IsoBlocks` is
/// bits-agnostic; only the `bits` tag and the codebook used by
/// `iso_encode_fast`/`iso_decode_fast` differ. The dense pack (8 vals/u32) is
/// handled internally by `iso_encode_fast` when invoked with `bits=4`.
///
/// CPU-only. The existing MSL kernel is hard-coded for `bits=3`; an iso4 MSL
/// kernel variant is deferred.
pub struct QuantIsoV4 {
    /// Accumulated per-token blocks (one entry per append call; `dequant`
    /// flattens them).
    pub blocks: Vec<IsoBlocks>,
    /// GPU-resident packed ring for the fused symmetric flash-decode path.
    /// Empty until the first [`Self::gpu_append`]. When live it is the SOLE
    /// resident copy — the CPU `blocks` are dropped and rebuilt on demand via
    /// [`synced_iso_v_blocks`]. Mirror of [`super::QuantIsoV3::gpu`].
    pub gpu: QuantKGpuRing,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the storage was provisioned for.
    pub max_seq: i32,
    /// Bit-width tag (always [`ISO4_BITS`] for this codec; kept as a field for
    /// symmetry with the other `Quant*` structs).
    pub bits: u8,
}

impl std::fmt::Debug for QuantIsoV4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantIsoV4")
            .field("n_blocks", &self.blocks.len())
            .field("gpu_resident", &self.gpu.is_allocated())
            .field("shape", &self.shape)
            .field("max_seq", &self.max_seq)
            .field("bits", &self.bits)
            .finish()
    }
}

impl QuantIsoV4 {
    /// Construct an empty `QuantIsoV4` for `init_shape = [B, kv_h, 0, D]`.
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32) -> Self {
        Self {
            blocks: Vec::new(),
            gpu: QuantKGpuRing::default(),
            shape: init_shape,
            max_seq,
            bits: ISO4_BITS,
        }
    }

    /// Append one V slice (CPU path).
    ///
    /// # Errors
    ///
    /// Forwards any [`IsoQuantError`] from [`iso_encode_fast`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoV4::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let b = new_shape[0] as usize;
        let kv_h = new_shape[1] as usize;
        let new_seq = new_shape[2] as usize;
        let head_dim = new_shape[3] as usize;
        let n_tokens_total = b * kv_h * new_seq;

        // Store each chunk sequence-major so the per-append blocks share one
        // layout; a head-major store transposes heads across multi-append GQA
        // caches (kv_h>1). See [`super::QuantIsoV3::append`].
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, quaternions, norms) =
            iso_encode_fast(&seq_major, head_dim, ISO4_GROUP_SIZE, ISO4_BITS)
                .map_err(|e: IsoQuantError| Error::Mlx(format!("iso4 encode: {e}")))?;

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

        if self.shape.len() != 4 || self.shape[0] == 0 {
            self.shape = new_shape.to_vec();
        } else {
            self.shape[2] += new_shape[2];
        }
        Ok(())
    }

    /// Construct a `QuantIsoV4` from pre-computed CPU blocks (SSD hydrate path).
    #[must_use]
    pub fn from_cpu_blocks(blocks: Vec<IsoBlocks>, shape: Vec<i32>) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantIsoV4::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        let max_seq = if shape.len() >= 3 { shape[2] } else { 0 };
        Self {
            blocks,
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: QuantKGpuRing::default(),
            shape,
            max_seq,
            bits: ISO4_BITS,
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
    }

    /// Truncate the accumulated sequence to `n` tokens.
    ///
    /// Mirror of [`super::QuantIsoV3::truncate_to`]: the GPU ring is **kept**,
    /// not cleared, so a ring-only decode tail up to `n` survives and `dequant` /
    /// an SSD spill can rebuild it via [`synced_iso_v_blocks`].
    pub fn truncate_to(&mut self, n: i32) {
        let keep =
            super::truncate_keep_count(self.blocks.iter().map(|blk| blk.n_tokens), &self.shape, n);
        self.blocks.truncate(keep);
        // NB: no `self.gpu.clear()` — the ring is the source of truth for a
        // ring-only decode tail; see [`super::QuantIsoV3::truncate_to`].
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
    }

    /// Deep-clone. Materialises any ring-only decode tail into complete blocks
    /// first — mirror of [`super::QuantIsoV3::try_deep_clone`].
    ///
    /// # Errors
    ///
    /// Forwards a [`synced_iso_v_blocks`] reconciliation error.
    pub fn try_deep_clone(&self) -> Result<Self> {
        let blocks =
            synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?.into_owned();
        Ok(Self {
            // The clone starts CPU-only: `blocks` carries the full payload, so
            // the ring re-seeds from them on the clone's first GPU append.
            gpu: QuantKGpuRing::default(),
            blocks,
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            bits: self.bits,
        })
    }

    /// Concatenate the accumulated CPU blocks into flat sequence-major
    /// `(codes, scales, norms)` — mirror of [`super::QuantIsoV3::flatten_blocks`].
    fn flatten_blocks(&self) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
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

    /// Push one GPU-encoded chunk into the GPU ring, seeding it from the
    /// accumulated CPU blocks first when it is not yet live. Mirror of
    /// [`super::QuantIsoV3::gpu_append`].
    ///
    /// # Errors
    ///
    /// Forwards [`QuantKGpuRing::seed_from_cpu`] / [`QuantKGpuRing::append_encoded`]
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
                "QuantIsoV4::gpu_append: n_groups for head_dim={head_dim} exceeds i32::MAX"
            ))
        })?;
        if n_groups <= 0 {
            return Err(Error::Quant(format!(
                "QuantIsoV4::gpu_append: head_dim={head_dim} yields no quaternion groups"
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

    /// GPU packed view of the first `kv_seq` positions, or `None` when the ring
    /// is not live.
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

    /// Resident bytes held by this store: CPU blocks plus the GPU ring.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            blocks,
            gpu,
            // Geometry / tags, not allocations.
            shape: _,
            max_seq: _,
            bits: _,
        } = self;
        blocks.iter().map(IsoBlocks::byte_size).sum::<u64>() + gpu.byte_size()
    }

    /// Dequantize all accumulated V slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// # Errors
    ///
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoV4::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile the ring-only decode tail — see [`super::QuantIsoV3::dequant`].
        let blocks = synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?;

        if blocks.is_empty() {
            // Loud on a lost decode tail — see [`super::QuantIsoV3::dequant`].
            if total_elems != 0 {
                return Err(Error::Mlx(format!(
                    "QuantIsoV4::dequant: no blocks but shape {:?} implies {total_elems} elems — \
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
                ISO4_GROUP_SIZE,
                ISO4_BITS,
            )
            .map_err(|e: IsoQuantError| Error::Mlx(format!("iso4 decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // Loud invariant — see [`super::QuantIsoV3::dequant`].
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantIsoV4::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
                 refusing to zero-pad / truncate",
                out.len(),
                self.shape
            )));
        }
        // Blocks are sequence-major (see `append`); reorder back to head-major
        // `[B, kv_h, S, D]`.
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let out = super::seq_layout::transpose_seq_heads(&out, b, s, kv_h, head_dim);
        Ok(out)
    }
}

#[cfg(test)]
#[path = "quant_iso_v4_tests.rs"]
mod quant_iso_v4_tests;
