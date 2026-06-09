// unsafe_code: all unsafe lives in submodules; this file has no unsafe code.

//! mlx-c high-level ops — arithmetic, matmul, reshape, and indexing.
//!
//! Every function in this module is a thin safe wrapper over a raw mlx-c call.
//! Shapes and dtypes are validated by mlx-c; errors are surfaced through the
//! installed mlx error handler and mapped to [`rmlx_core::error::Error::Mlx`].
//!
//! # Public API (selected)
//!
//! - [`matmul`] / [`quantized_matmul`] / [`gather_qmm`] — matrix multiply variants.
//! - [`dequantize`] / [`quantize`] — affine 4-bit weight quant round-trips.
//! - [`scalar_f32`] — create a scalar `[]`-shaped `f32` Array.
//! - [`argmax`] / [`max_axis`] — reduction ops used in sampling and smoke probes.
//! - [`zeros`] — zero-filled tensor allocation.
//! - [`stack_axis`] / [`expand_dims`] / [`broadcast_to`] — shape manipulation.
//! - [`conv1d`] / [`conv2d`] / [`conv_transpose1d`] — convolution ops (used by audio/vision towers).
//! - [`pad`] / [`arange`] / [`tril`] — utility ops.
//!
//! # See also
//!
//! - [`super::fast_ops`] — fused Metal kernels (RMSNorm, RoPE, SDPA).
//! - [`super::metal_kernel`] — custom Metal kernel dispatch.

mod activation;
mod arith;
mod matmul;
mod shape;

pub use activation::{gelu, gelu_tanh, silu, softmax, softmax_precise, tanh};
pub use arith::{
    add, argmax, argpartition, argsort, broadcast_to, clip, concatenate, divide, exp, expand_dims,
    floor_divide, greater_equal, log, log1p, max_axis, multiply, negative, repeat_axis, scalar_f32,
    scatter_add, sigmoid, softplus, sqrt, stack_axis, subtract, sum_axis, sum_axis_keepdims,
    take_along_axis, topk, where_cond, zeros,
};
pub use matmul::{dequantize, gather_qmm, matmul, quantize, quantize_mode, quantized_matmul};
pub use shape::{arange, conv1d, conv2d, conv_transpose1d, cos, maximum, pad, sin, tril};

// ---------------------------------------------------------------------------
// MLX_QUANTIZED_MM_OK_FOR_MXFP8
// ---------------------------------------------------------------------------

/// True: `mlx_quantized_matmul` with `mode="mxfp8"` is available.
///
/// MLX 0.31 (brew) supports mxfp8 via `mlx_quantized_matmul`. The C-level
/// `mode` parameter was added in mlx-c 0.6. A future version may swap the codepath
/// for a custom Metal kernel; for now we use the upstream C symbol.
pub const MLX_QUANTIZED_MM_OK_FOR_MXFP8: bool = true;

// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(unsafe_code)]
mod tests;
