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

use rmlx_core::error::Result;

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};
use crate::storage::quant_iso_v::IsoBlocks;

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
            return Err(rmlx_core::error::Error::Mlx(format!(
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
            iso_encode_fast(&seq_major, head_dim, ISO4_GROUP_SIZE, ISO4_BITS).map_err(
                |e: IsoQuantError| rmlx_core::error::Error::Mlx(format!("iso4 encode: {e}")),
            )?;

        self.blocks.push(IsoBlocks {
            codes,
            scales,
            quaternions,
            norms,
            n_tokens: n_tokens_total,
        });

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
            shape,
            max_seq,
            bits: ISO4_BITS,
        }
    }

    /// Reset the accumulated sequence length to zero. Buffers are kept for
    /// reuse on the next request.
    pub fn reset(&mut self) {
        self.blocks.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = 0;
        }
    }

    /// Truncate the accumulated sequence to `n` tokens.
    pub fn truncate_to(&mut self, n: i32) {
        let n_usize = n.max(0) as usize;
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
    }

    /// Deep-clone (CPU path is plain `Vec` clones).
    ///
    /// # Errors
    ///
    /// Currently infallible on the CPU path; returns `Result` for parity with
    /// the other `Quant*` structs.
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            blocks: self.blocks.clone(),
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            bits: self.bits,
        })
    }

    /// Resident bytes held by this store (CPU blocks; this codec has no GPU
    /// mirror).
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            blocks,
            // Geometry / tags, not allocations.
            shape: _,
            max_seq: _,
            bits: _,
        } = self;
        blocks.iter().map(IsoBlocks::byte_size).sum()
    }

    /// Dequantize all accumulated V slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// # Errors
    ///
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoV4::dequant: malformed shape {:?}",
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
                ISO4_GROUP_SIZE,
                ISO4_BITS,
            )
            .map_err(|e: IsoQuantError| {
                rmlx_core::error::Error::Mlx(format!("iso4 decode: {e}"))
            })?;
            out.extend_from_slice(&dec);
        }
        if out.len() < total_elems {
            out.resize(total_elems, 0.0);
        } else if out.len() > total_elems {
            out.truncate(total_elems);
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
