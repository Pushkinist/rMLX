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

use rmlx_core::error::{Error, Result};

use crate::c_api::{self, CApiVerdict};
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
    // When weight is None, pass the cached null sentinel.
    // A cached sentinel, not an mlx_array_new() + mlx_array_free() per call.
    let w_arr = match weight {
        Some(w) => w.inner,
        None => null_sentinel(),
    };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_fast_rms_norm(&raw mut res, x.inner, w_arr, eps, s)
        })
    }?;
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
    // A cached sentinel, not an mlx_array_new() + mlx_array_free() per call.
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
    }?;
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
    // freqs = null sentinel.
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
    }?;
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
    }?;
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
    }?;
    unsafe { check_status(status, "rope_with_freqs") }?;
    Ok(Array { inner: res })
}

/// The `tracing` target of one event for each attention node handed to mlx-c.
///
/// The event is emitted at graph construction, beside the one call of
/// `mlx_fast_scaled_dot_product_attention`. A node that is built and never
/// evaluated is reported too. At TRACE it carries what selects MLX's attention
/// kernel, apart from the batch size: `q_heads`, `q_rows`, `head_dim`,
/// `kv_heads`, `k_rows`, `v_head_dim`, `dtype`, `mask` (the mask mode string)
/// and `device`. `caller_q_rows` is the query row count the caller gave: it
/// differs from `q_rows` when the node's query rows are padded.
pub const ATTENTION_CALL_TARGET: &str = "rmlx_mlx::attention_call";

fn trace_attention_call(
    q: &Array,
    k: &Array,
    v: &Array,
    mask_mode: &str,
    device: Device,
    caller_query_rows: Option<i32>,
) {
    if !tracing::enabled!(target: ATTENTION_CALL_TARGET, tracing::Level::TRACE) {
        return;
    }
    let (q_shape, k_shape, v_shape) = (q.shape(), k.shape(), v.shape());
    let dim = |shape: &[i32], axis: usize| shape.get(axis).copied().unwrap_or(-1);
    tracing::trace!(
        target: ATTENTION_CALL_TARGET,
        q_heads = dim(&q_shape, 1),
        q_rows = dim(&q_shape, 2),
        caller_q_rows = caller_query_rows.unwrap_or_else(|| dim(&q_shape, 2)),
        head_dim = dim(&q_shape, 3),
        kv_heads = dim(&k_shape, 1),
        k_rows = dim(&k_shape, 2),
        v_head_dim = dim(&v_shape, 3),
        dtype = ?q.dtype(),
        mask = mask_mode,
        device = ?device,
        "attention call"
    );
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
///
/// # Errors
/// `Error::Mlx` without a call into mlx-c when the loaded `libmlxc.dylib` has
/// another C API than the one this crate was compiled against: the argument
/// lists differ and the symbol name does not (`src/c_api.rs`).
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    device: Device,
) -> Result<Array> {
    sdpa_under(
        c_api::verdict(),
        q,
        k,
        v,
        scale,
        mask_mode,
        mask_arr,
        device,
    )
}

/// [`scaled_dot_product_attention`] under a given C API verdict, so a test on
/// a matched pair can show that a mismatch stops the call.
///
/// Every attention node is built here, so the row rule below holds for every
/// caller.
fn sdpa_under(
    verdict: CApiVerdict,
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    verdict.require_match()?;
    let Some((query_rows, padded_rows)) = padded_query_rows(q, v, mask_mode, device) else {
        return attention_node(q, k, v, scale, mask_mode, mask_arr, device, None);
    };
    // The padding is the array's own first rows. It keeps the dtype and every
    // other dim, a boolean mask and a batch included, and each padded row is a
    // real attention row: finite, and never read.
    let padded = |a: &Array, shape: &[i32], axis: usize| -> Result<Array> {
        let mut stop = shape.to_vec();
        if let Some(rows) = stop.get_mut(axis) {
            *rows = padded_rows - query_rows;
        }
        let first_rows = a.slice(&vec![0; stop.len()], &stop, &vec![1; stop.len()], device)?;
        crate::concatenate(&[a, &first_rows], axis as i32, device)
    };
    let q_padded = padded(q, &q.shape(), QUERY_ROW_AXIS)?;
    let mask_padded = match mask_arr {
        Some(mask) => {
            let mask_shape = mask.shape();
            // MLX broadcasts the mask against `[batch, heads, query rows, key
            // rows]`, so its query rows are its second axis from the end.
            match mask_shape.len().checked_sub(2) {
                Some(axis) if mask_shape.get(axis) == Some(&query_rows) => {
                    Some(padded(mask, &mask_shape, axis)?)
                }
                // One row, or no row axis: MLX broadcasts it over the query
                // rows, the padded rows included.
                Some(axis) if mask_shape.get(axis) != Some(&1) => {
                    return Err(Error::Mlx(format!(
                        "scaled_dot_product_attention: the mask has shape {mask_shape:?}, and \
                         its query axis is neither 1 nor the {query_rows} query rows"
                    )));
                }
                _ => None,
            }
        }
        None => None,
    };
    let node = attention_node(
        &q_padded,
        k,
        v,
        scale,
        mask_mode,
        mask_padded.as_ref().or(mask_arr),
        device,
        Some(query_rows),
    )?;
    let mut stop = node.shape();
    if let Some(rows) = stop.get_mut(QUERY_ROW_AXIS) {
        *rows = query_rows;
    }
    node.slice(&vec![0; stop.len()], &stop, &vec![1; stop.len()], device)
}

const QUERY_ROW_AXIS: usize = 2;
const HEAD_DIM_AXIS: usize = 3;

/// `(query rows, padded query rows)` when this call must reach MLX with more
/// query rows than it has.
///
/// This rule is for MLX 0.32.3, the pinned version. MLX 0.32.3 runs an
/// attention call at head dim 256, with an array mask and at least 1024 query
/// rows, on its head-dim-split Metal kernel. That kernel is compiled per call
/// for "the query rows are a multiple of 64" and "the key rows are a multiple
/// of 32". With both false it returns `+inf` rows under Metal device-memory
/// shader validation. The fault is seen only under that validation: its cause
/// is the instrumented compile of the kernel, not the kernel source. No run
/// without the instrument has shown it.
///
/// The rule is on the query rows alone. A call with aligned key rows was
/// measured clean, but a clean cell is a bound on a rate, and the padding
/// costs the same. The call reaches MLX with its query rows padded to the next
/// multiple of 64, and the padded rows are sliced off the output. Query rows
/// are independent in attention, so every real row is what the call asked for,
/// and it stays on the fused kernel.
///
/// When the pin moves, measure the cells of `MEASURED_CELLS` in
/// `crates/rmlx-models/tests/prefill_attention_configuration.rs` on the new
/// MLX under device-memory validation, then keep this rule or delete it
/// (`docs/MLX_PAIR.md`, "Moving the pin"). The rule has no `cfg` on purpose:
/// the mlx-c C API a binary is built against is not the MLX version it runs
/// on, and the padding changes no real row on an MLX without the fault.
///
/// A call under the row floor returns after two dim reads, with no
/// allocation: a decode step has one query row.
fn padded_query_rows(q: &Array, v: &Array, mask_mode: &str, device: Device) -> Option<(i32, i32)> {
    const HEAD_DIM: i32 = 256;
    const MIN_QUERY_ROWS: i32 = 1024;
    const QUERY_BLOCK: i32 = 64;
    if mask_mode != "array" || device != Device::Gpu {
        return None;
    }
    let query_rows = q.dim(QUERY_ROW_AXIS).ok()?;
    if query_rows < MIN_QUERY_ROWS || query_rows % QUERY_BLOCK == 0 {
        return None;
    }
    if q.dim(HEAD_DIM_AXIS).ok()? != HEAD_DIM || v.dim(HEAD_DIM_AXIS).ok()? != HEAD_DIM {
        return None;
    }
    Some((
        query_rows,
        query_rows + QUERY_BLOCK - query_rows % QUERY_BLOCK,
    ))
}

/// Hand one attention node to mlx-c. Only [`sdpa_under`] calls this.
/// `caller_query_rows` is `Some` when the node's query rows are padded.
fn attention_node(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    device: Device,
    caller_query_rows: Option<i32>,
) -> Result<Array> {
    let mode_cstr = mode_to_cstr(mask_mode, "scaled_dot_product_attention")?;
    // When mask_arr is None, use the cached null sentinel.
    let mask_inner = match mask_arr {
        Some(m) => m.inner,
        None => null_sentinel(),
    };
    // sinks = null sentinel (not used for causal/sliding window).
    let sinks_null = null_sentinel();
    trace_attention_call(q, k, v, mask_mode, device, caller_query_rows);
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
                // `force_fused = false` keeps MLX's own choice between its fused
                // kernels and the composite graph.
                #[cfg(mlxc_c_api_0_7)]
                false,
                s,
            )
        })
    }?;
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
