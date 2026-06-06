// IsoQuant 3-bit K storage struct.
//
// Mirror of `QuantIsoV3` (see `quant_iso_v.rs`) for the K axis. The IsoQuant
// codec (`crate::isoquant::iso_encode_fast` / `iso_decode_fast`) is axis-
// agnostic — it consumes a flat `[B, kv_h, S, D]` f32 row buffer and a
// per-row `head_dim`. We fork the storage struct (not the codec) for the
// same reason `QuantIsoV4` was forked from `QuantIsoV3`: the name is stable
// across crates (SSD writer/reader, helpers, dispatch), and renaming to a
// generic `QuantIso` would cause large cross-crate churn for no benefit —
// bits is fixed per storage variant.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized K buffer: `QuantIsoK3` (IsoQuant 3-bit K codec).

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};
use crate::storage::quant_iso_v::IsoBlocks;

/// Bit-width of the iso3 K codec (fixed at 3-bit — identical codebook to the
/// V-side [`crate::storage::QuantIsoV3`]).
pub const ISO_K3_BITS: u8 = 3;

/// Quaternion-block size for the iso3 K codec (fixed at 4; one quaternion per
/// group in fast mode).
pub const ISO_K3_GROUP_SIZE: usize = 4;

/// Accumulated IsoQuant K cache (3-bit, quaternion SO(4) fast mode).
///
/// CPU-only — no MSL kernel for K-side iso3 (the V-side iso3 MSL kernel
/// in `isoquant_msl.rs` is structurally K/V-agnostic but the K-side dispatch
/// path falls through to the dequant-then-SDPA legacy fallback). Storage
/// payload layout is identical to [`crate::storage::QuantIsoV3`] — the only
/// distinction at the storage level is the role on the SDPA path.
pub struct QuantIsoK3 {
    /// Accumulated per-token blocks (one entry per append call; `dequant`
    /// flattens them).
    pub blocks: Vec<IsoBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the storage was provisioned for.
    pub max_seq: i32,
    /// Bit-width tag (always [`ISO_K3_BITS`]).
    pub bits: u8,
}

impl std::fmt::Debug for QuantIsoK3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantIsoK3")
            .field("n_blocks", &self.blocks.len())
            .field("shape", &self.shape)
            .field("max_seq", &self.max_seq)
            .field("bits", &self.bits)
            .finish()
    }
}

impl QuantIsoK3 {
    /// Construct an empty `QuantIsoK3` for `init_shape = [B, kv_h, 0, D]`.
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32) -> Self {
        Self {
            blocks: Vec::new(),
            shape: init_shape,
            max_seq,
            bits: ISO_K3_BITS,
        }
    }

    /// Append one K slice (CPU path). Same contract as `QuantIsoV3::append`.
    ///
    /// # Errors
    /// Forwards any [`IsoQuantError`] from [`iso_encode_fast`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoK3::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let head_dim = new_shape[3] as usize;
        let n_tokens_total =
            (new_shape[0] as usize) * (new_shape[1] as usize) * (new_shape[2] as usize);

        let (codes, scales, quaternions, norms) =
            iso_encode_fast(f32_data, head_dim, ISO_K3_GROUP_SIZE, ISO_K3_BITS).map_err(
                |e: IsoQuantError| rmlx_core::error::Error::Mlx(format!("iso_k3 encode: {e}")),
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

    /// Construct from pre-computed CPU blocks (SSD hydrate path).
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
            "QuantIsoK3::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        Self {
            blocks,
            shape,
            max_seq,
            bits: ISO_K3_BITS,
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

    /// Deep-clone (CPU path is plain `Vec` clones).
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

    /// Approximate byte footprint of the accumulated payload.
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

    /// Dequantize all accumulated K slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// # Errors
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoK3::dequant: malformed shape {:?}",
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
                ISO_K3_GROUP_SIZE,
                ISO_K3_BITS,
            )
            .map_err(|e: IsoQuantError| {
                rmlx_core::error::Error::Mlx(format!("iso_k3 decode: {e}"))
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

    /// GPU dequant via on-demand `Array::from_bytes` upload.
    ///
    /// See [`crate::storage::QuantIsoV3::dequant_gpu`] for the algorithm and
    /// rationale. K-side mirror: identical pack layout, same MSL kernel
    /// (axis-agnostic).
    ///
    /// # Errors
    ///
    /// - `Error::Mlx` if `Array::from_bytes` / kernel dispatch fails.
    /// - `Error::Quant` if `shape` is malformed or `head_dim` violates the
    ///   `ISO_K3_GROUP_SIZE` multiple constraint.
    pub fn dequant_gpu(&self, device: Device) -> Result<Array> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantIsoK3::dequant_gpu: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(ISO_K3_GROUP_SIZE) {
            return Err(rmlx_core::error::Error::Quant(format!(
                "QuantIsoK3::dequant_gpu: head_dim={head_dim} must be a positive multiple of \
                 ISO_K3_GROUP_SIZE={ISO_K3_GROUP_SIZE}"
            )));
        }
        let n_groups = head_dim / ISO_K3_GROUP_SIZE;

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
            for &n in &blk.norms {
                let n_bytes = n.to_le_bytes();
                for _ in 0..n_groups {
                    norms_bytes.extend_from_slice(&n_bytes);
                }
            }
            // Checked arithmetic — see V-side mirror.
            let blk_groups = blk.n_tokens.checked_mul(n_groups).ok_or_else(|| {
                rmlx_core::error::Error::Quant(
                    "dequant_gpu: blk.n_tokens * n_groups overflow".to_owned(),
                )
            })?;
            total_groups = total_groups.checked_add(blk_groups).ok_or_else(|| {
                rmlx_core::error::Error::Quant("dequant_gpu: total_groups overflow".to_owned())
            })?;
        }

        // Guard against silent shape divergence — see V-side mirror.
        let declared_total: usize = self.shape.iter().map(|&d| d as usize).product();
        let actual_total: usize = total_groups.checked_mul(ISO_K3_GROUP_SIZE).ok_or_else(|| {
            rmlx_core::error::Error::Quant(
                "dequant_gpu: total_groups * ISO_K3_GROUP_SIZE overflow".to_owned(),
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
            return Array::from_bytes(&[][..], &self.shape, Dtype::F32);
        }

        let codes_arr = Array::from_bytes(&codes_bytes, &[total_groups as i32], Dtype::U32)?;
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

        flat.reshape(&self.shape, device)
    }
}

#[cfg(test)]
#[path = "quant_iso_k_tests.rs"]
mod quant_iso_k_tests;
