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

use rmlx_core::error::Result;

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};
use crate::storage::quant_iso_v::IsoBlocks;

/// Bit-width of the iso4 K codec (fixed at 4-bit).
pub const ISO_K4_BITS: u8 = 4;

/// Quaternion-block size for the iso4 K codec (fixed at 4 elements/group).
pub const ISO_K4_GROUP_SIZE: usize = 4;

/// Accumulated IsoQuant K cache (4-bit, quaternion SO(4) fast mode).
///
/// Same per-block payload as [`crate::storage::QuantIsoK3`] with `bits=4` and
/// the dense 8-vals-per-u32 pack handled internally by `iso_encode_fast`.
pub struct QuantIsoK4 {
    /// Accumulated per-token blocks.
    pub blocks: Vec<IsoBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length provisioned.
    pub max_seq: i32,
    /// Bit-width tag (always [`ISO_K4_BITS`]).
    pub bits: u8,
}

impl std::fmt::Debug for QuantIsoK4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantIsoK4")
            .field("n_blocks", &self.blocks.len())
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
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoK4::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let head_dim = new_shape[3] as usize;
        let n_tokens_total =
            (new_shape[0] as usize) * (new_shape[1] as usize) * (new_shape[2] as usize);

        let (codes, scales, quaternions, norms) =
            iso_encode_fast(f32_data, head_dim, ISO_K4_GROUP_SIZE, ISO_K4_BITS).map_err(
                |e: IsoQuantError| rmlx_core::error::Error::Mlx(format!("iso_k4 encode: {e}")),
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
            shape,
            max_seq,
            bits: ISO_K4_BITS,
        }
    }

    /// Reset the accumulated sequence length to zero.
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

    /// Deep-clone.
    ///
    /// # Errors
    /// Currently infallible on the CPU path; returns `Result` for parity.
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            blocks: self.blocks.clone(),
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            bits: self.bits,
        })
    }

    /// Approximate byte footprint.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        let mut total = 0usize;
        for blk in &self.blocks {
            total += blk.codes.len() * size_of::<u32>();
            total += blk.scales.len() * size_of::<f32>();
            total += blk.quaternions.len() * size_of::<f32>();
            total += blk.norms.len() * size_of::<f32>();
        }
        total
    }

    /// Dequantize all accumulated K slices into one flat f32 vector.
    ///
    /// # Errors
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoK4::dequant: malformed shape {:?}",
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
                ISO_K4_GROUP_SIZE,
                ISO_K4_BITS,
            )
            .map_err(|e: IsoQuantError| {
                rmlx_core::error::Error::Mlx(format!("iso_k4 decode: {e}"))
            })?;
            out.extend_from_slice(&dec);
        }
        if out.len() < total_elems {
            out.resize(total_elems, 0.0);
        } else if out.len() > total_elems {
            out.truncate(total_elems);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "quant_iso_k4_tests.rs"]
mod quant_iso_k4_tests;
