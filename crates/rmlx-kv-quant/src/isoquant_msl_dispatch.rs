//! Width dispatch for the iso MSL kernels.
//!
//! The iso codec ships two code widths and one kernel pair per width —
//! [`crate::isoquant_msl`] at 3 bits, [`crate::isoquant_msl_v4`] at 4. The iso
//! stores and the `kvcache` appenders are width-parametric, so each of them
//! needs the same `bits -> module` selection. It lives here once: a second copy
//! is how one caller comes to encode at one width and read the result back at
//! the other.
//!
//! `bits` is a plain integer, so no exhaustiveness lint guards a `_` arm. Every
//! entry below rejects an unsupported width instead of falling through to one
//! of the two kernels it has, and `what` is the caller's own name so the
//! refusal says who asked.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use crate::storage::{ISO3_BITS, ISO4_BITS};

/// Encode one `[.., head_dim]` chunk with the kernel for `bits`.
///
/// Returns the kernel's `(codes, scales, quaternions, norms)` GPU arrays; the
/// norms plane is per-group, which is the form the encode kernels emit.
///
/// # Errors
///
/// Returns [`Error::Quant`] when `bits` is neither 3 nor 4, and forwards the
/// selected kernel's own errors.
pub(crate) fn iso_quantize_gpu(
    v_full: &Array,
    head_dim: usize,
    bits: u8,
    what: &str,
    device: Device,
) -> Result<(Array, Array, Array, Array)> {
    match bits {
        ISO3_BITS => crate::isoquant_msl::iso_quantize_v3_gpu(v_full, head_dim, device),
        ISO4_BITS => crate::isoquant_msl_v4::iso_quantize_v4_gpu(v_full, head_dim, device),
        other => Err(unsupported(what, other)),
    }
}

/// Read one encode's GPU outputs back into the CPU vectors
/// [`crate::storage::IsoBlocks`] holds, with the unpacker for `bits`.
///
/// # Errors
///
/// Returns [`Error::Quant`] when `bits` is neither 3 nor 4, and forwards the
/// selected readback's own errors.
pub(crate) fn iso_gpu_outputs_to_cpu(
    codes_gpu: &Array,
    scales_gpu: &Array,
    quaternions_gpu: &Array,
    norms_gpu: &Array,
    n_tokens: usize,
    n_groups: usize,
    bits: u8,
    what: &str,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    match bits {
        ISO3_BITS => crate::isoquant_msl::iso3_gpu_outputs_to_cpu(
            codes_gpu,
            scales_gpu,
            quaternions_gpu,
            norms_gpu,
            n_tokens,
            n_groups,
        ),
        ISO4_BITS => crate::isoquant_msl_v4::iso4_gpu_outputs_to_cpu(
            codes_gpu,
            scales_gpu,
            quaternions_gpu,
            norms_gpu,
            n_tokens,
            n_groups,
        ),
        other => Err(unsupported(what, other)),
    }
}

/// Decode a packed iso plane with the kernel for `bits`.
///
/// The quaternion slot is passed through: both kernels declare it and neither
/// dereferences it, because every group carries the same
/// [`crate::isoquant::FIXED_QUAT`].
///
/// # Errors
///
/// Returns [`Error::Quant`] when `bits` is neither 3 nor 4, and forwards the
/// selected kernel's own errors.
pub(crate) fn iso_dequantize_gpu(
    codes_packed: &Array,
    scales: &Array,
    quaternions: &Array,
    norms: &Array,
    head_dim: usize,
    bits: u8,
    out_dtype: Dtype,
    what: &str,
    device: Device,
) -> Result<Array> {
    match bits {
        ISO3_BITS => crate::isoquant_msl::iso_dequantize_v3_gpu(
            codes_packed,
            scales,
            quaternions,
            norms,
            head_dim,
            out_dtype,
            device,
        ),
        ISO4_BITS => crate::isoquant_msl_v4::iso_dequantize_v4_gpu(
            codes_packed,
            scales,
            quaternions,
            norms,
            head_dim,
            out_dtype,
            device,
        ),
        other => Err(unsupported(what, other)),
    }
}

/// The refusal every entry above returns for a width that has no kernel.
fn unsupported(what: &str, bits: u8) -> Error {
    Error::Quant(format!(
        "{what}: unsupported iso bits={bits} (only 3 and 4); refusing to use another \
         width's kernel"
    ))
}
