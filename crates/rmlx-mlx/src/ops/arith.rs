//! Arithmetic, reduction, and indexing ops.

#![allow(unsafe_code)]

use rmlx_core::error::Result;

use crate::{check_status, install_error_handler, sys, with_stream, Array, Device, Dtype};

// ---------------------------------------------------------------------------
// Scalar-from-f32 helper (used for constant arrays in masking)
// ---------------------------------------------------------------------------

/// Create a scalar Array from an f32. The resulting array has shape `[]` and dtype F32.
pub fn scalar_f32(v: f32) -> Array {
    install_error_handler();
    // SAFETY: mlx_array_new_float creates a valid scalar array.
    let inner = unsafe { sys::mlx_array_new_float(v) };
    Array { inner }
}

// ---------------------------------------------------------------------------
// Ops — plain arithmetic
// ---------------------------------------------------------------------------

/// Element-wise addition: `a + b`. Broadcasts via MLX semantics.
pub fn add(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status =
        unsafe { with_stream(device, |s| sys::mlx_add(&raw mut res, a.inner, b.inner, s)) };
    unsafe { check_status(status, "add") }?;
    Ok(Array { inner: res })
}

/// Element-wise multiplication: `a * b`. Broadcasts.
pub fn multiply(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_multiply(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "multiply") }?;
    Ok(Array { inner: res })
}

/// Element-wise division: `a / b`. Broadcasts.
pub fn divide(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_divide(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "divide") }?;
    Ok(Array { inner: res })
}

/// Element-wise floor division: `floor(a / b)`. Broadcasts. Integer-exact for
/// integer inputs (unlike `divide`, which promotes to float).
pub fn floor_divide(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_floor_divide(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "floor_divide") }?;
    Ok(Array { inner: res })
}

/// Element-wise subtraction: `a - b`. Broadcasts.
pub fn subtract(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_subtract(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "subtract") }?;
    Ok(Array { inner: res })
}

/// Element-wise negation: `-a`.
pub fn negative(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_negative(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "negative") }?;
    Ok(Array { inner: res })
}

/// Element-wise clamp: `min(max(a, a_min), a_max)`. Broadcasts the bounds.
///
/// Ported for Gemma4 `ClippableLinear` (vision/audio towers store per-tensor
/// `input_min/max` + `output_min/max` clamp buffers). `a_min`/`a_max` are
/// scalar arrays loaded from the checkpoint.
pub fn clip(a: &Array, a_min: &Array, a_max: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_clip(&raw mut res, a.inner, a_min.inner, a_max.inner, s)
        })
    };
    unsafe { check_status(status, "clip") }?;
    Ok(Array { inner: res })
}

/// Element-wise `a >= b`. Returns a bool array (dtype = U8 in MLX).
///
/// Used by the attention-mask builders in `rmlx_models::layers::mask` to turn a
/// pair of position vectors into an allowed/blocked map:
/// `greater_equal(q_pos, k_pos)` is the causal half, and
/// `greater_equal(k_pos, oldest_allowed)` the sliding-window half, each `0` or
/// `1` per element before `where_cond` turns it into an additive bias.
pub fn greater_equal(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_greater_equal(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "greater_equal") }?;
    Ok(Array { inner: res })
}

/// Element-wise ternary: `if cond { x } else { y }`. Broadcasts all three.
///
/// `cond` must be a bool (U8) array; `x` and `y` must share the same dtype.
pub fn where_cond(cond: &Array, x: &Array, y: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_where(&raw mut res, cond.inner, x.inner, y.inner, s)
        })
    };
    unsafe { check_status(status, "where_cond") }?;
    Ok(Array { inner: res })
}

/// Repeat values along a given axis: each element along `axis` is duplicated `repeats` times.
/// Shape: `[..., d, ...]` -> `[..., d * repeats, ...]`.
pub fn repeat_axis(a: &Array, repeats: i32, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_repeat_axis(&raw mut res, a.inner, repeats, axis, s)
        })
    };
    unsafe { check_status(status, "repeat_axis") }?;
    Ok(Array { inner: res })
}

/// Stack arrays along a new axis. All arrays must have the same shape.
pub fn stack_axis(arrays: &[&Array], axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let inner_arrs: Vec<sys::mlx_array> = arrays.iter().map(|a| a.inner).collect();
    // mlx_vector_array_new_data: build a vector from a borrowed slice of mlx_array handles.
    let vec_arr = unsafe { sys::mlx_vector_array_new_data(inner_arrs.as_ptr(), inner_arrs.len()) };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_stack_axis(&raw mut res, vec_arr, axis, s)
        })
    };
    let _ = unsafe { sys::mlx_vector_array_free(vec_arr) };
    unsafe { check_status(status, "stack_axis") }?;
    Ok(Array { inner: res })
}

/// Element-wise log(1 + x). Numerically stable for small x.
pub fn log1p(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_log1p(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "log1p") }?;
    Ok(Array { inner: res })
}

/// Element-wise natural log.
pub fn log(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_log(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "log") }?;
    Ok(Array { inner: res })
}

/// Element-wise exp.
pub fn exp(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_exp(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "exp") }?;
    Ok(Array { inner: res })
}

/// Element-wise square root (`mlx_sqrt`).
///
/// Thin wrapper over the raw `mlx_sqrt` FFI, mirroring [`exp`] / [`log`].
/// For an L2 norm, callers should add a small positive floor before calling
/// (`sqrt(sum(x^2) + 1e-12)`) to match `torch.nn.functional.normalize`'s
/// default `eps` denominator clamp and to avoid `d/dx sqrt(0)` blowups.
pub fn sqrt(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_sqrt(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "sqrt") }?;
    Ok(Array { inner: res })
}

/// Sigmoid: 1 / (1 + exp(-x)).
pub fn sigmoid(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe { with_stream(device, |s| sys::mlx_sigmoid(&raw mut res, a.inner, s)) };
    unsafe { check_status(status, "sigmoid") }?;
    Ok(Array { inner: res })
}

/// Sum along `axis`. keepdims=false.
pub fn sum_axis(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_sum_axis(&raw mut res, a.inner, axis, false, s)
        })
    };
    unsafe { check_status(status, "sum_axis") }?;
    Ok(Array { inner: res })
}

/// Concatenate arrays along `axis`.
pub fn concatenate(arrays: &[&Array], axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    // Collect inner handles into a slice for mlx_vector_array_new_data.
    let inners: Vec<sys::mlx_array> = arrays.iter().map(|a| a.inner).collect();
    let vec_arr = unsafe { sys::mlx_vector_array_new_data(inners.as_ptr(), inners.len()) };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_concatenate_axis(&raw mut res, vec_arr, axis, s)
        })
    };
    unsafe { sys::mlx_vector_array_free(vec_arr) };
    unsafe { check_status(status, "concatenate") }?;
    Ok(Array { inner: res })
}

/// Top-k values (not indices) along last axis. Returns `[..., k]` shape.
pub fn topk(a: &Array, k: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_topk_axis(&raw mut res, a.inner, k, -1, s)
        })
    };
    unsafe { check_status(status, "topk") }?;
    Ok(Array { inner: res })
}

/// Argsort (ascending) along last axis.
pub fn argsort(a: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_argsort_axis(&raw mut res, a.inner, -1, s)
        })
    };
    unsafe { check_status(status, "argsort") }?;
    Ok(Array { inner: res })
}

/// Argpartition along `axis`: returns indices that partition `a` so that the
/// element at position `kth` is in its sorted position; elements before `kth`
/// are not greater than it; elements after are not less. O(N) vs argsort's
/// O(N log N). Use with `kth = -k` to get the indices of the top-k elements
/// in the last `k` positions (matches `mx.argpartition(a, kth=-k, axis=-1)`).
pub fn argpartition(a: &Array, kth: i32, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_argpartition_axis(&raw mut res, a.inner, kth, axis, s)
        })
    };
    unsafe { check_status(status, "argpartition") }?;
    Ok(Array { inner: res })
}

/// Take elements from `a` at positions `indices` along `axis`. Equivalent to
/// `np.take_along_axis(a, indices, axis=axis)`. `a` and `indices` must have
/// the same number of dims; on each non-`axis` dim they must broadcast.
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_take_along_axis(&raw mut res, a.inner, indices.inner, axis, s)
        })
    };
    unsafe { check_status(status, "take_along_axis") }?;
    Ok(Array { inner: res })
}

/// Zeros array with the given shape and dtype.
pub fn zeros(shape: &[i32], dtype: Dtype, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_zeros(&raw mut res, shape.as_ptr(), shape.len(), dtype.to_sys(), s)
        })
    };
    unsafe { check_status(status, "zeros") }?;
    Ok(Array { inner: res })
}

/// Scatter-add: `out[indices] += values` along `axis=0`.
///
/// `out` shape: `[n, hidden]`, `indices` shape: `[k]` I32, `values` shape: `[k, hidden]`.
pub fn scatter_add(out: &Array, indices: &Array, values: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_scatter_add_axis(&raw mut res, out.inner, indices.inner, values.inner, 0, s)
        })
    };
    unsafe { check_status(status, "scatter_add") }?;
    Ok(Array { inner: res })
}

// ---------------------------------------------------------------------------
// Ops — argmax (for forward probe: token with highest logit)
// ---------------------------------------------------------------------------

/// Argmax along `axis`. keepdims=false.
pub fn argmax(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_argmax_axis(&raw mut res, a.inner, axis, false, s)
        })
    };
    unsafe { check_status(status, "argmax") }?;
    Ok(Array { inner: res })
}

/// Max along `axis`. keepdims=false. Used to compute max(|logits|).
pub fn max_axis(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_max_axis(&raw mut res, a.inner, axis, false, s)
        })
    };
    unsafe { check_status(status, "max_axis") }?;
    Ok(Array { inner: res })
}

// ---------------------------------------------------------------------------
// Ops — expand_dims + broadcast (used for mask construction)
// ---------------------------------------------------------------------------

/// Add a dimension of size 1 at `axis`.
pub fn expand_dims(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_expand_dims(&raw mut res, a.inner, axis, s)
        })
    };
    unsafe { check_status(status, "expand_dims") }?;
    Ok(Array { inner: res })
}

/// Broadcast `a` to `shape`.
pub fn broadcast_to(a: &Array, shape: &[i32], device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_broadcast_to(&raw mut res, a.inner, shape.as_ptr(), shape.len(), s)
        })
    };
    unsafe { check_status(status, "broadcast_to") }?;
    Ok(Array { inner: res })
}

/// Sum along `axis` keeping the reduced dimension (size 1).
pub fn sum_axis_keepdims(a: &Array, axis: i32, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_sum_axis(&raw mut res, a.inner, axis, true, s)
        })
    };
    unsafe { check_status(status, "sum_axis_keepdims") }?;
    Ok(Array { inner: res })
}

/// Softplus: `log(1 + exp(x))`, computed as `log1p(exp(x))`.
pub fn softplus(a: &Array, device: Device) -> Result<Array> {
    log1p(&exp(a, device)?, device)
}
