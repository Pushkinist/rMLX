//! Shape manipulation, convolution, and geometric ops.

#![allow(unsafe_code)]

use rmlx_core::error::Result;

use crate::{check_status, install_error_handler, sys, with_stream, Array, Device, Dtype};

use super::arith::scalar_f32;

// ---------------------------------------------------------------------------
// Ops — convolution + padding + elementwise
// ---------------------------------------------------------------------------

/// 2-D convolution. `input`: `[N, H, W, C_in]` (channel-last), `weight`:
/// `[C_out, kH, kW, C_in]`. Symmetric per-axis stride/padding/dilation.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_conv2d(
                &raw mut res,
                input.inner,
                weight.inner,
                stride.0,
                stride.1,
                padding.0,
                padding.1,
                dilation.0,
                dilation.1,
                groups,
                s,
            )
        })
    };
    unsafe { check_status(status, "conv2d") }?;
    Ok(Array { inner: res })
}

/// Pad `a` along the given `axes` with per-axis `(low, high)` zero padding
/// (constant mode, pad value 0). `low`/`high` line up with `axes` by index.
pub fn pad(a: &Array, axes: &[i32], low: &[i32], high: &[i32], device: Device) -> Result<Array> {
    install_error_handler();
    // f32-ok: MLX casts the pad value to the array's dtype inside `mlx_pad`;
    // `pad` returns bf16 for a bf16 input, pinned by
    // `activations_return_the_input_dtype` in ops/tests.rs.
    let zero = scalar_f32(0.0);
    let mode = crate::mode_to_cstr("constant", "pad")?;
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_pad(
                &raw mut res,
                a.inner,
                axes.as_ptr(),
                axes.len(),
                low.as_ptr(),
                low.len(),
                high.as_ptr(),
                high.len(),
                zero.inner,
                mode.as_ptr(),
                s,
            )
        })
    };
    unsafe { check_status(status, "pad") }?;
    Ok(Array { inner: res })
}

/// Element-wise sine.
pub fn sin(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_sin(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "sin") }?;
    Ok(Array { inner: res })
}

/// Element-wise cosine.
pub fn cos(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_cos(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "cos") }?;
    Ok(Array { inner: res })
}

/// Element-wise maximum of `a` and `b` (broadcasting). `maximum(x, 0)` = relu.
pub fn maximum(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_maximum(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "maximum") }?;
    Ok(Array { inner: res })
}

/// Lower-triangular part of `x` (entries above the `k`-th diagonal zeroed).
pub fn tril(x: &Array, k: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_tril(&raw mut res, x.inner, k, s)) };
    unsafe { check_status(status, "tril") }?;
    Ok(Array { inner: res })
}

/// `arange(start, stop, step)` as an F32 vector.
pub fn arange(start: f64, stop: f64, step: f64, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_arange(&raw mut res, start, stop, step, Dtype::F32.to_sys(), s)
        })
    };
    unsafe { check_status(status, "arange") }?;
    Ok(Array { inner: res })
}

/// 1-D transposed convolution (a.k.a. deconvolution / fractionally-strided conv).
///
/// `input` shape `[B, T, C_in]`, `weight` shape `[C_out, K, C_in/groups]`.
/// Returns `[B, T_out, C_out]` where
/// `T_out = (T - 1) * stride - 2*padding + K + output_padding`.
///
/// Semantics match PyTorch `ConvTranspose1d` with the same weight layout.
#[allow(
    clippy::too_many_arguments,
    reason = "conv_transpose1d has 6 hyperparams; fewer is not possible without a struct wrapper"
)]
pub fn conv_transpose1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    output_padding: i32,
    groups: i32,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_conv_transpose1d(
                &raw mut res,
                input.inner,
                weight.inner,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
                s,
            )
        })
    };
    unsafe { check_status(status, "conv_transpose1d") }?;
    Ok(Array { inner: res })
}

/// 1-D convolution. `input` shape `[B, T, C_in]`, `weight` shape
/// `[C_out, K, C_in/groups]`. Returns `[B, T_out, C_out]` with
/// `T_out = (T + 2*padding - K) / stride + 1`.
///
/// For depthwise convolution use `groups = C_in = C_out`.
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_conv1d(
                &raw mut res,
                input.inner,
                weight.inner,
                stride,
                padding,
                dilation,
                groups,
                s,
            )
        })
    };
    unsafe { check_status(status, "conv1d") }?;
    Ok(Array { inner: res })
}
