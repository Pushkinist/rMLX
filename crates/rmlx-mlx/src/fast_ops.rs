// unsafe_code: mlx-rs FFI bridge — calls mlx_fast_* C API via unsafe blocks
#![allow(unsafe_code)]

//! Wrappers for the `mlx_fast_*` family of fused Metal kernels.
//!
//! These ops bypass the elementwise graph and dispatch directly to optimised
//! Metal kernels exposed through the mlx-c `mlx_fast_*` C API.
//! All calls are synchronous with respect to the MLX compute stream.
//!
//! # Public API
//!
//! - [`rms_norm`] — fused RMSNorm with optional learned weight.
//! - [`rope`] — RoPE with static frequencies (base + scale).
//! - [`rope_dynamic`] — RoPE with dynamic (NTK/YARN) scaling.
//! - [`rope_with_freqs`] — RoPE driven by a pre-computed frequency tensor.
//! - [`rope_with_freqs_dynamic`] — dynamic variant of the above.
//! - [`scaled_dot_product_attention`] — fused SDPA (FlashAttention-style).
//!
//! # See also
//!
//! - [`super::ops`] — non-fused elementwise and matmul ops via the mlx-c graph.

use rmlx_core::error::Result;

use crate::{
    check_status, install_error_handler, mode_to_cstr, null_sentinel, sys, with_stream, Array,
    Device,
};

// ---------------------------------------------------------------------------
// Ops — fast ops (rms_norm, rope, sdpa)
// ---------------------------------------------------------------------------

/// Fused RMSNorm: `x / sqrt(mean(x^2) + eps) * weight`.
///
/// `weight` is the learned gamma (passed directly from the model's norm.weight,
/// which is initialised at 1.0 and grows during training). Pass `None` for
/// `RMSNormNoScale` layers (Gemma4 `v_norm`).
///
/// Wraps `mlx_fast_rms_norm`. Note that `mlx_fast_rms_norm` documentation
/// calls the parameter "weight" and the C header says `/* may be null */`.
/// When weight is None we pass a default-constructed empty handle (ctx=null).
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32, device: Device) -> Result<Array> {
    install_error_handler();
    // When weight is None, pass the cached null sentinel (ch-18 F1).
    // Previously: mlx_array_new() + mlx_array_free() per call.
    let w_arr = match weight {
        Some(w) => w.inner,
        None => null_sentinel(),
    };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_rms_norm(&raw mut res, x.inner, w_arr, eps, s)
        })
    };
    // Do NOT free w_arr when weight.is_none() — it is the cached null sentinel.
    unsafe { check_status(status, "rms_norm") }?;
    Ok(Array { inner: res })
}

/// Rotary Position Embedding (RoPE).
///
/// `x` shape: `[batch, n_heads, seq_len, head_dim]`.
/// `dims`: number of dimensions to rotate (may be `head_dim` for full rotation or
/// `partial_rotary_factor * head_dim` for partial).
/// `traditional`: use the "traditional" (non-interleaved) formulation.
/// `base`: rope theta (e.g. 10000.0 for sliding, 1000000.0 for full attention).
/// `scale`: scaling factor for the frequencies (1.0 unless using scaled RoPE).
/// `offset`: position offset (0 for prefill; cache offset for incremental decode).
///
/// Wraps `mlx_fast_rope` with `freqs = null` (MLX computes freqs from `base`).
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let base_opt = sys::mlx_optional_float {
        value: base,
        has_value: true,
    };
    // freqs = null sentinel (let MLX compute from base).
    // Previously: mlx_array_new() + mlx_array_free() per call (ch-18 F1).
    let freqs_null = null_sentinel();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_rope(
                &raw mut res,
                x.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset,
                freqs_null,
                s,
            )
        })
    };
    // Do NOT free freqs_null — it is the cached null sentinel.
    unsafe { check_status(status, "rope") }?;
    Ok(Array { inner: res })
}

/// Rotary Position Embedding (RoPE) with a **dynamic** offset passed as an MLX array.
///
/// Identical math to [`rope`] but the position offset is an `Array` (typically a
/// 0-D `i32` scalar) instead of a captured `i32`. This is the variant required
/// inside an `mx.compile` closure: capturing an `i32` offset would force the
/// closure to retrace on every step (cache miss); passing the offset as an
/// Array keeps a single compiled program across all decode steps, with the
/// offset value plumbed through at runtime.
///
/// Caller MUST evaluate or pass through the result before the offset Array is
/// dropped — MLX's lazy graph holds a borrow of the offset handle until eval.
///
/// `offset` shape: 0-D scalar (`Array::from_bytes(&offset.to_le_bytes(), &[], Dtype::I32)`).
/// `freqs = null` lets MLX compute frequencies from `base`.
///
/// Wraps `mlx_fast_rope_dynamic` with `freqs = null`.
pub fn rope_dynamic(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: &Array,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let base_opt = sys::mlx_optional_float {
        value: base,
        has_value: true,
    };
    // freqs = null sentinel (ch-18 F1).
    let freqs_null = null_sentinel();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_rope_dynamic(
                &raw mut res,
                x.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset.inner,
                freqs_null,
                s,
            )
        })
    };
    // Do NOT free freqs_null — it is the cached null sentinel.
    unsafe { check_status(status, "rope_dynamic") }?;
    Ok(Array { inner: res })
}

/// RoPE with an explicit per-dimension frequency table AND a dynamic offset Array.
///
/// Combines [`rope_dynamic`] (offset as 0-D i32 Array) with [`rope_with_freqs`]
/// (explicit `freqs` table for ProportionalRoPE / Gemma4 full-attention). Used
/// inside `mx.compile` closures where both the position offset and the freq
/// table must flow through the compiled graph rather than being baked in.
///
/// `dims` must equal the full head dimension. `freqs` shape `[dims/2]`. `base`
/// is ignored when `freqs` is supplied; we pass `has_value=false`.
///
/// Wraps `mlx_fast_rope_dynamic` with the `freqs` argument set.
pub fn rope_with_freqs_dynamic(
    x: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
    offset: &Array,
    freqs: &Array,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let base_opt = sys::mlx_optional_float {
        value: 0.0,
        has_value: false,
    };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_rope_dynamic(
                &raw mut res,
                x.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset.inner,
                freqs.inner,
                s,
            )
        })
    };
    unsafe { check_status(status, "rope_with_freqs_dynamic") }?;
    Ok(Array { inner: res })
}

/// RoPE with an explicit per-dimension frequency table (ProportionalRoPE / NTK variants).
///
/// Used for Gemma4 full-attention layers where the frequency exponent is divided by
/// `global_head_dim` (512) rather than the local `rotated_dims` (128). Passing
/// standard `rope()` with `dims=128` would divide by 128 instead — a ~27 000× error
/// at the highest-frequency rotated dim.
///
/// `freqs` is a 1-D F32 array of length `dims / 2`. The kernel applies
/// `cos(freqs[i] * pos)` / `sin(freqs[i] * pos)` to the i-th rotation pair.
/// To leave a pair untouched, set its frequency to `+inf` (the MLX convention).
/// Note: some mlx-c builds may silently ignore `+inf` entries rather than
/// skipping them; empirically this matches the Python reference behaviour.
///
/// `dims` must equal the full head dimension, not just the rotated prefix.
/// `base` is ignored by the kernel when `freqs` is provided; we pass
/// `has_value = false` to make the intent explicit.
///
/// Wraps `mlx_fast_rope` with the optional `freqs` argument set.
pub fn rope_with_freqs(
    x: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
    offset: i32,
    freqs: &Array,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    // base is unused when freqs is supplied; signal this to the kernel.
    let base_opt = sys::mlx_optional_float {
        value: 0.0,
        has_value: false,
    };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_rope(
                &raw mut res,
                x.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset,
                freqs.inner,
                s,
            )
        })
    };
    unsafe { check_status(status, "rope_with_freqs") }?;
    Ok(Array { inner: res })
}

/// Scaled dot-product attention.
///
/// `q`, `k`, `v` shapes: `[batch, n_heads, seq_len, head_dim]`.
/// `scale`: 1/sqrt(head_dim) or 1.0 (Gemma4 uses 1.0).
/// `mask`: optional causal mask array (bf16 or f32 additive mask) or None.
///
/// `mask_mode`: MLX's string hint for mask type. mlx-c accepts ONLY these
/// values (the Metal kernel rejects anything else, incl. `"additive"`):
/// - `"causal"`: use internal causal masking (fastest).
/// - `"array"`: caller supplies an explicit mask in `mask_arr` — an additive
///   bias (0 = allowed, large-negative = masked) whose dtype promotes with
///   Q/K/V. This is how additive / sliding-window masks are passed.
/// - `""` (empty): no mask.
///
/// When mask_mode is `"causal"` or `""`, `mask_arr` is ignored. When `"array"`,
/// `mask_arr` must be a valid array.
///
/// Wraps `mlx_fast_scaled_dot_product_attention`.
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    let mode_cstr = mode_to_cstr(mask_mode, "scaled_dot_product_attention")?;
    // When mask_arr is None, use the cached null sentinel (ch-18 F1).
    let mask_inner = match mask_arr {
        Some(m) => m.inner,
        None => null_sentinel(),
    };
    // sinks = null sentinel (not used for causal/sliding window in Stage 1).
    let sinks_null = null_sentinel();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_scaled_dot_product_attention(
                &raw mut res,
                q.inner,
                k.inner,
                v.inner,
                scale,
                mode_cstr.as_ptr(),
                mask_inner,
                sinks_null,
                s,
            )
        })
    };
    // Do NOT free mask_inner or sinks_null — both are the cached null sentinel.
    unsafe { check_status(status, "scaled_dot_product_attention") }?;
    Ok(Array { inner: res })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "fast_ops_tests.rs"]
mod tests;
