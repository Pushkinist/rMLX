//! Activation functions: gelu_tanh, gelu, silu, tanh, softmax, softmax_precise.

#![allow(unsafe_code)]

use std::sync::OnceLock;

use rmlx_core::error::Result;

use crate::{check_status, install_error_handler, sys, with_stream, Array, Device};

use super::arith::{add, multiply, scalar_f32};

// ---------------------------------------------------------------------------
// Cached gelu_tanh constants (ch-18 F2)
// ---------------------------------------------------------------------------
//
// gelu_tanh calls scalar_f32 four times per invocation — once for each of the
// fixed arithmetic constants (coeff, alpha, one, half). Each mlx_array_new_float
// + mlx_array_free pair is a heap allocation + atomic ref-count round-trip.
// With ~26–42 FFN layers per token that is 104–168 scalar-array alloc/free
// pairs per decode step from gelu alone.
//
// Fix: cache each constant as a process-global OnceLock<Array>. The handle is
// allocated once and lives for the process lifetime (never freed). mlx-c says
// arrays are thread-safe to pass as inputs, so sharing across request threads
// is correct.

static GELU_COEFF: OnceLock<Array> = OnceLock::new(); // 0.0356774
static GELU_ALPHA: OnceLock<Array> = OnceLock::new(); // 0.7978845608
static GELU_ONE: OnceLock<Array> = OnceLock::new(); // 1.0
static GELU_HALF: OnceLock<Array> = OnceLock::new(); // 0.5
static GELU_INV_SQRT2: OnceLock<Array> = OnceLock::new(); // 1/sqrt(2) ≈ 0.7071067811865475

#[inline]
fn gelu_coeff() -> &'static Array {
    GELU_COEFF.get_or_init(|| scalar_f32(0.035_677_4_f32))
}

#[inline]
fn gelu_alpha() -> &'static Array {
    GELU_ALPHA.get_or_init(|| scalar_f32(0.797_884_6_f32))
}

#[inline]
fn gelu_one() -> &'static Array {
    GELU_ONE.get_or_init(|| scalar_f32(1.0_f32))
}

#[inline]
fn gelu_half() -> &'static Array {
    GELU_HALF.get_or_init(|| scalar_f32(0.5_f32))
}

#[inline]
fn gelu_inv_sqrt2() -> &'static Array {
    GELU_INV_SQRT2.get_or_init(|| scalar_f32(std::f32::consts::FRAC_1_SQRT_2))
}

// ---------------------------------------------------------------------------
// Ops — activations
// ---------------------------------------------------------------------------

/// GELU with PyTorch `tanh` approximation: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// MLX does not expose gelu as a C op. We implement it via elementary ops.
/// This matches `nn.gelu_approx` in mlx-lm.
pub fn gelu_tanh(x: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    // gelu_approx(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // We use the fast tanh approximation from mlx_tanh.
    // Equivalent to: 0.5 * x * (1 + tanh(x * (0.7978845608 + 0.0356774 * x^2)))
    // Constants: sqrt(2/pi) ≈ 0.7978845608028654, 3/sqrt(2*pi) * 0.044715 ≈ 0.0356774
    // kCoeff = 0.044715, kAlpha = sqrt(2/pi)
    //
    // Constants are cached as process-global OnceLock<Array> (ch-18 F2).
    // Each scalar_f32 call was mlx_array_new_float + mlx_array_free per invocation;
    // caching removes 4 alloc/free pairs per FFN layer per decode step.

    // Step 1: x^2
    let x2 = multiply(x, x, device)?;
    // Step 2: 0.0356774 * x^2 (cached constant)
    let coeff_x2 = multiply(gelu_coeff(), &x2, device)?;
    // Step 3: 0.7978845608 + 0.0356774 * x^2 (cached constant)
    let inner = add(gelu_alpha(), &coeff_x2, device)?;
    // Step 4: x * inner
    let x_inner = multiply(x, &inner, device)?;
    // Step 5: tanh(x * inner)
    let tanh_val = {
        let mut res = unsafe { sys::mlx_array_new() };
        let status =
            unsafe { with_stream(device, |s| sys::mlx_tanh(&raw mut res, x_inner.inner, s)) };
        unsafe { check_status(status, "tanh in gelu_tanh") }?;
        Array { inner: res }
    };
    // Step 6: 1 + tanh(...) (cached constant)
    let one_plus_tanh = add(gelu_one(), &tanh_val, device)?;
    // Step 7: x * (1 + tanh(...))
    let x_times = multiply(x, &one_plus_tanh, device)?;
    // Step 8: 0.5 * x * (1 + tanh(...)) (cached constant)
    multiply(gelu_half(), &x_times, device)
}

/// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// This is the **erf-based** (exact) GELU matching `torch.nn.GELU()` default
/// (`approximate='none'`) and `jax.nn.gelu(approximate=False)`.
///
/// Use this for the jina-v4 vision PatchMerger and any other module that uses
/// `nn.GELU()` without `approximate='tanh'`. Do NOT use [`gelu_tanh`] there —
/// at float32 the outputs diverge by up to ~0.002 near the tails, which is
/// enough to fail a cosine ≥ 0.999 parity gate.
///
/// MLX does not expose a `gelu` C op, so we compose via `mlx_erf`:
/// ```text
/// gelu(x) = x * 0.5 * (1 + erf(x * (1/sqrt(2))))
/// ```
/// Constants `0.5`, `1.0`, and `1/sqrt(2)` are cached as process-global
/// [`OnceLock`] scalars (same pattern as [`gelu_tanh`]).
pub fn gelu(x: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    // Step 1: x * (1/sqrt(2))
    let x_scaled = multiply(gelu_inv_sqrt2(), x, device)?;
    // Step 2: erf(x / sqrt(2))
    let erf_val = {
        let mut res = unsafe { sys::mlx_array_new() };
        let status =
            unsafe { with_stream(device, |s| sys::mlx_erf(&raw mut res, x_scaled.inner, s)) };
        unsafe { check_status(status, "erf in gelu") }?;
        Array { inner: res }
    };
    // Step 3: 1 + erf(...) (cached constant)
    let one_plus_erf = add(gelu_one(), &erf_val, device)?;
    // Step 4: x * (1 + erf(...))
    let x_times = multiply(x, &one_plus_erf, device)?;
    // Step 5: 0.5 * x * (1 + erf(...)) (cached constant)
    multiply(gelu_half(), &x_times, device)
}

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu(x: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    // sigmoid(x)
    let sig = {
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe { with_stream(device, |s| sys::mlx_sigmoid(&raw mut res, x.inner, s)) };
        unsafe { check_status(status, "sigmoid in silu") }?;
        Array { inner: res }
    };
    multiply(x, &sig, device)
}

// ---------------------------------------------------------------------------
// Ops — tanh (standalone, used for logit softcapping)
// ---------------------------------------------------------------------------

/// Element-wise hyperbolic tangent.
pub fn tanh(x: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_tanh(&raw mut res, x.inner, s)) };
    unsafe { check_status(status, "tanh") }?;
    Ok(Array { inner: res })
}

// ---------------------------------------------------------------------------
// Ops — softmax
// ---------------------------------------------------------------------------

/// Softmax along `axis`.
pub fn softmax(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            // `precise=false` — fast (bf16-native). Use true for CPU oracle only.
            sys::mlx_softmax_axis(&raw mut res, a.inner, axis, false, s)
        })
    };
    unsafe { check_status(status, "softmax") }?;
    Ok(Array { inner: res })
}

/// Softmax along `axis` with `precise=true` (f32 internal accumulation).
///
/// Mirrors `mx.softmax(scores, axis=-1, precise=True)` from mlx-lm-turboquant's
/// `mixed_quantized_scaled_dot_product_attention`. The non-precise variant
/// runs in bf16/f16 for speed and accumulates softmax norms in the same dtype;
/// the precise variant promotes to f32 for the reduction. Used in the
/// quantized-SDPA path where the upstream reference uses precise softmax.
pub fn softmax_precise(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_softmax_axis(&raw mut res, a.inner, axis, true, s)
        })
    };
    unsafe { check_status(status, "softmax_precise") }?;
    Ok(Array { inner: res })
}
