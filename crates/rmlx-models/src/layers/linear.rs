//! Linear layer — plain bf16, quantized (affine / mxfp8), or PARO INT4.

use rmlx_core::error::Result;
use rmlx_mlx::{matmul, quantized_matmul, Array, Device};

use super::quant::QuantMode;

// ---------------------------------------------------------------------------
// ParoRotation
// ---------------------------------------------------------------------------

/// Pre-computed rotation parameters for a single PARO linear layer.
///
/// Built once at load time from the raw tensor bytes in the checkpoint.
/// All arrays are GPU-resident F16/I32 arrays ready for `paro_rotate_gpu`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed layer struct — fields are the complete PARO rotation-parameter contract; adding a field requires updating paro_rotate_gpu call sites and Linear::Paro construction"
)]
#[allow(missing_debug_implementations)]
pub struct ParoRotation {
    /// `[krot, hidden/2]` I32 packed pair indices.
    pub packed_pairs: Array,
    /// `[krot, hidden/2]` F16 cosine values.
    pub cos_theta: Array,
    /// `[krot, hidden/2]` F16 sine values.
    pub sin_theta: Array,
    /// `[1, hidden]` F16 per-channel scales.
    pub channel_scales: Array,
    /// Actual krot for this layer.
    pub krot: usize,
    /// Group size used by both the rotation kernel and the INT4 matmul.
    pub group_size: usize,
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Linear layer — plain (bf16), integer/mxfp8 quantized, or PARO INT4.
///
/// `biases` is `Some` for affine-int quantized tensors that carry a zero-point offset
/// (e.g. Gemma4 26B `mlp.*` tensors). `None` for mxfp8 and most affine-int snapshots.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — three linear-layer storage variants (plain/quantized/paro); adding a variant requires updating forward(), try_clone(), and all arch layer-loading sites"
)]
/// Linear layer — plain, quantized, or PARO INT4 (see docs/WEIGHT_QUANTS.md).
#[allow(missing_debug_implementations)]
pub enum Linear {
    /// Plain bf16 weight matrix `[out, in]`.
    Plain {
        /// Weight tensor shape `[out_features, in_features]` in bf16.
        weight: Array, // [out, in] bf16
    },
    /// Affine/mxfp8 quantized (MLX `quantized_matmul` format).
    Quantized {
        /// Packed codes `[out, packed_in]` U32.
        weight: Array, // [out, packed_in] U32
        /// Per-group scales `[out, in/group_size]`.
        scales: Array, // [out, in/group_size]
        /// Affine zero-point biases `None` for mxfp8.
        biases: Option<Array>, // [out, in/group_size] for affine, None for mxfp8
        /// Affine group size.
        group_size: i32,
        /// Quantization bit-width.
        bits: i32,
        /// Quantization mode — 1 B (Copy enum) vs 24 B String.
        mode: QuantMode,
    },
    /// ParoQuant INT4: pre-rotation of input activations + standard INT4 affine matmul.
    ///
    /// Forward:
    /// 1. Reshape x to [batch_flat, hidden].
    /// 2. paro_rotate_gpu(x, rotation params).
    /// 3. quantized_matmul(rotated_x, weight, scales, biases, group_size, 4, "affine", true).
    Paro {
        /// PARO rotation parameters.
        rotation: ParoRotation,
        /// Packed INT4 codes `[out, in*4/32]` U32.
        weight: Array, // [out, in*4/32] U32
        /// Per-group scales `[out, num_groups]` F16.
        scales: Array, // [out, num_groups] F16
        /// Per-group biases (zero-points) derived from `qzeros`.
        biases: Array, // [out, num_groups] F16 — always present in PARO (derived from qzeros)
    },
}

impl Linear {
    /// Shallow-clone the layer. MLX arrays are ref-counted, so this is cheap.
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Linear::Plain { weight } => Ok(Linear::Plain {
                weight: weight.try_clone()?,
            }),
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Ok(Linear::Quantized {
                weight: weight.try_clone()?,
                scales: scales.try_clone()?,
                biases: biases.as_ref().map(Array::try_clone).transpose()?,
                group_size: *group_size,
                bits: *bits,
                mode: *mode, // QuantMode is Copy
            }),
            Linear::Paro {
                rotation,
                weight,
                scales,
                biases,
            } => Ok(Linear::Paro {
                rotation: ParoRotation {
                    packed_pairs: rotation.packed_pairs.try_clone()?,
                    cos_theta: rotation.cos_theta.try_clone()?,
                    sin_theta: rotation.sin_theta.try_clone()?,
                    channel_scales: rotation.channel_scales.try_clone()?,
                    krot: rotation.krot,
                    group_size: rotation.group_size,
                },
                weight: weight.try_clone()?,
                scales: scales.try_clone()?,
                biases: biases.try_clone()?,
            }),
        }
    }

    /// `x` shape: `[batch, seq, in_features]` or `[batch, in_features]`.
    pub fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Linear::Plain { weight } => matmul(x, &weight.transpose(&[1, 0], device)?, device),
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode.as_str(),
                true,
                device,
            ),
            Linear::Paro {
                rotation,
                weight,
                scales,
                biases,
            } => {
                // Flatten to [batch_flat, hidden] for the rotation kernel.
                let shape = x.shape();
                let hidden = *shape.last().unwrap_or(&0) as usize;
                let batch_flat: i32 = shape.iter().product::<i32>() / hidden as i32;
                let x_2d = x.reshape(&[batch_flat, hidden as i32], device)?;

                let rotated = crate::paroquant_msl::paro_rotate_gpu(
                    &x_2d,
                    &rotation.packed_pairs,
                    &rotation.cos_theta,
                    &rotation.sin_theta,
                    &rotation.channel_scales,
                    rotation.krot,
                    rotation.group_size,
                    device,
                )?;

                // Restore original shape before matmul.
                let rotated = if shape.len() > 2 {
                    rotated.reshape(&shape, device)?
                } else {
                    rotated
                };

                quantized_matmul(
                    &rotated,
                    weight,
                    scales,
                    Some(biases),
                    rotation.group_size as i32,
                    4,
                    "affine",
                    true,
                    device,
                )
            }
        }
    }
}
