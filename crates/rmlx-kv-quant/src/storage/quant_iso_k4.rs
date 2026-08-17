// IsoQuant 4-bit K storage struct.
//
// Mirror of `QuantIsoV4` for the K axis. The codec
// (`crate::isoquant::iso_encode_fast` / `iso_decode_fast`) is axis-agnostic;
// only the role on the SDPA path differs.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized K buffer: `QuantIsoK4` (IsoQuant 4-bit K codec).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};
use crate::storage::quant_iso_k::iso_n_groups_i32;
use crate::storage::quant_iso_v::{synced_iso_v_blocks, IsoBlocks};

use super::QuantKGpuRing;

/// Bit-width of the iso4 K codec (fixed at 4-bit).
pub const ISO_K4_BITS: u8 = 4;

/// Quaternion-block size for the iso4 K codec. Alias of
/// [`crate::storage::ISO_QUAT_BLOCK_SIZE`] — one quaternion per group is fixed
/// by the algebra, not chosen per bit width, and defining it from the shared
/// constant is what stops this store's CPU blocks drifting away from the ring
/// stride its GPU paths derive via `iso_n_groups_for`.
pub const ISO_K4_GROUP_SIZE: usize = crate::storage::ISO_QUAT_BLOCK_SIZE;

/// Accumulated IsoQuant K cache (4-bit, quaternion SO(4) fast mode).
///
/// Same per-block payload as [`crate::storage::QuantIsoK3`] with `bits=4` and
/// the dense 8-vals-per-u32 pack handled internally by `iso_encode_fast`, and
/// the same optional GPU ring — see [`crate::storage::QuantIsoK3`] for the
/// field semantics.
pub struct QuantIsoK4 {
    /// Accumulated per-token blocks.
    pub blocks: Vec<IsoBlocks>,
    /// GPU-resident packed ring. Empty until the first [`Self::gpu_append`].
    pub gpu: QuantKGpuRing,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length provisioned. Inert — see
    /// [`crate::storage::QuantIsoK3::max_seq`]; the ring grows against a
    /// per-call `max_seq`, never this snapshot.
    pub max_seq: i32,
    /// Bit-width tag (always [`ISO_K4_BITS`]).
    pub bits: u8,
}

impl std::fmt::Debug for QuantIsoK4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantIsoK4")
            .field("n_blocks", &self.blocks.len())
            .field("gpu_resident", &self.gpu.is_allocated())
            .field("shape", &self.shape)
            .field("max_seq", &self.max_seq)
            .field("bits", &self.bits)
            .finish()
    }
}

impl QuantIsoK4 {
    /// Construct an empty `QuantIsoK4`.
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32) -> Self {
        Self {
            blocks: Vec::new(),
            gpu: QuantKGpuRing::default(),
            shape: init_shape,
            max_seq,
            bits: ISO_K4_BITS,
        }
    }

    /// Append one K slice (CPU path).
    ///
    /// # Errors
    /// Forwards any [`IsoQuantError`] from [`iso_encode_fast`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoK4::append: expected 4D new_shape, got {new_shape:?}"
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
            iso_encode_fast(&seq_major, head_dim, ISO_K4_GROUP_SIZE, ISO_K4_BITS)
                .map_err(|e: IsoQuantError| Error::Mlx(format!("iso_k4 encode: {e}")))?;

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

    /// Construct from pre-computed CPU blocks (SSD hydrate).
    ///
    /// `max_seq` must be the provisioned model window for this layer, **not**
    /// the accumulated sequence length at spill time (`shape[2]`). Passing
    /// `shape[2]` here would set a stale ceiling equal to the spilled length,
    /// causing the next append after hydration to reject tokens that would
    /// fit within the true model window.
    #[must_use]
    pub fn from_cpu_blocks(blocks: Vec<IsoBlocks>, shape: Vec<i32>, max_seq: i32) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantIsoK4::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        Self {
            blocks,
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: QuantKGpuRing::default(),
            shape,
            max_seq,
            bits: ISO_K4_BITS,
        }
    }

    /// Reset the accumulated sequence length to zero.
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
    /// The GPU ring is **kept**, not cleared — mirror of
    /// [`crate::storage::QuantIsoK3::truncate_to`] (the ring-only-tail
    /// treatment). Clearing it would discard a ring-only decode tail.
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
    }

    /// Deep-clone.
    ///
    /// Materialises any ring-only decode tail into complete CPU blocks first —
    /// mirror of [`crate::storage::QuantIsoK3::try_deep_clone`].
    ///
    /// # Errors
    /// Forwards a [`synced_iso_v_blocks`] reconciliation error.
    pub fn try_deep_clone(&self) -> Result<Self> {
        let blocks =
            synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?.into_owned();
        Ok(Self {
            // The clone starts CPU-only — see
            // [`crate::storage::QuantIsoK3::try_deep_clone`].
            gpu: QuantKGpuRing::default(),
            blocks,
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            bits: self.bits,
        })
    }

    /// Concatenate the accumulated CPU blocks into flat sequence-major
    /// `(codes, scales, norms)`. See
    /// [`crate::storage::QuantIsoK3::flatten_blocks`].
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

    /// Push one GPU-encoded chunk into the GPU ring. Mirror of
    /// [`crate::storage::QuantIsoK3::gpu_append`] — see it for the contract,
    /// including why `max_seq` is a parameter and not a field.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Quant`] when `head_dim` violates the
    /// group-size invariant, and forwards ring errors.
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
        let n_groups = iso_n_groups_i32(head_dim, "QuantIsoK4::gpu_append")?;
        if !self.gpu.is_allocated() && prev_seq > 0 {
            let (c, s, n) = self.flatten_blocks();
            self.gpu
                .seed_from_cpu(&c, &s, &n, kv_h, n_groups, prev_seq, max_seq, device)?;
        }
        self.gpu.append_encoded(
            codes, scales, norms, kv_h, n_groups, prev_seq, new_seq, max_seq, device,
        )
    }

    /// GPU packed view of the first `kv_seq` positions. Mirror of
    /// [`crate::storage::QuantIsoK3::gpu_packed_view`].
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
    /// The ring is real resident memory and is counted at its full allocation.
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

    /// Dequantize all accumulated K slices into one flat f32 vector.
    ///
    /// # Errors
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoK4::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile the ring-only decode tail — see
        // [`crate::storage::QuantIsoK3::dequant`].
        let blocks = synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?;

        if blocks.is_empty() {
            // Loud on a lost decode tail — see [`crate::storage::QuantIsoK3::dequant`].
            if total_elems != 0 {
                return Err(Error::Mlx(format!(
                    "QuantIsoK4::dequant: no blocks but shape {:?} implies {total_elems} elems — \
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
                ISO_K4_GROUP_SIZE,
                ISO_K4_BITS,
            )
            .map_err(|e: IsoQuantError| Error::Mlx(format!("iso_k4 decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // Loud invariant — see [`crate::storage::QuantIsoK3::dequant`].
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantIsoK4::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
                 refusing to zero-pad / truncate",
                out.len(),
                self.shape
            )));
        }
        // Per-block reorder back to head-major `[B, kv_h, S, D]` — see
        // [`crate::storage::QuantIsoK3::dequant`].
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
}

#[cfg(test)]
#[path = "quant_iso_k4_tests.rs"]
mod quant_iso_k4_tests;
