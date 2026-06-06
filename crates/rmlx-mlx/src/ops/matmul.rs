//! Matrix multiplication ops: matmul, quantized_matmul, dequantize, quantize, gather_qmm.

#![allow(unsafe_code)]

use rmlx_core::error::{Error, Result};

use crate::{
    check_status, install_error_handler, mode_to_cstr, null_sentinel, sys, with_stream, Array,
    Device,
};

/// Matrix multiplication: `a @ b`. Standard MLX matmul semantics (batch OK).
pub fn matmul(a: &Array, b: &Array, device: Device) -> Result<Array> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_matmul(&raw mut res, a.inner, b.inner, s)
        })
    };
    unsafe { check_status(status, "matmul") }?;
    Ok(Array { inner: res })
}

/// Integer-affine or mxfp8 quantized matrix multiply.
///
/// `w` is the packed weight (U32 for affine/mxfp8, U8 for raw mxfp8 E4M3).
/// `scales` carries per-group scale factors.
/// `biases` is `None` for mxfp8/mxfp4; `Some` for integer affine.
/// `mode` must match what MLX used to quantize: `"affine"` for integer affine,
/// `"mxfp8"` for OCP mxfp8. This model uses `"mxfp8"`.
/// Note: MLX 0.31+ rejects `"default"` — the correct affine mode string is `"affine"`.
/// `transpose_w=true` when the weight matrix is stored transposed (the common case
/// for linear layers: `w` is `[out_features, packed_in_features]` and the
/// matmul computes `x @ w.T`).
///
/// # Note on mxfp8 vs integer affine
/// Despite the config `mode = "mxfp8"`, MLX stores weights in a packed integer
/// format indistinguishable from integer affine at the byte level. The
/// `mlx_quantized_matmul` C function handles both via the `mode` string; passing
/// `"mxfp8"` selects the correct dequant codepath. The distinction matters:
/// integer affine uses uniform scale * code + bias; mxfp8 uses E8M0 scales
/// and E4M3 element codes.
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    transpose_w: bool,
    device: Device,
) -> Result<Array> {
    install_error_handler();

    // biases is optional — use the cached null sentinel when absent (ch-18 F1).
    // Previously: mlx_array_new() + mlx_array_free() per call.
    let biases_arr = match biases {
        Some(b) => b.inner,
        None => null_sentinel(),
    };

    let mode_cstr = mode_to_cstr(mode, "quantized_matmul")?;

    let gs = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bs = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };

    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_quantized_matmul(
                &raw mut res,
                x.inner,
                w.inner,
                scales.inner,
                biases_arr,
                transpose_w,
                gs,
                bs,
                mode_cstr.as_ptr(),
                s,
            )
        })
    };

    // Do NOT free biases_arr when biases.is_none() — it is the cached null sentinel.

    unsafe { check_status(status, "quantized_matmul") }?;
    Ok(Array { inner: res })
}

/// Dequantize a packed quantized weight tensor back to floating point.
///
/// Mirrors `mx.dequantize(w, scales, biases, group_size, bits, mode)` from
/// mlx-lm's `QuantizedEmbedding.__call__`. Used for the on-device embedding
/// lookup path: `dequantize(weight[ids], scales[ids], biases[ids], …)` so
/// the lookup never leaves the device. The CPU fallback in
/// `qwen_embedding_lookup` was an `eye(seq) @ w` workaround that forces a
/// device round-trip on every decode step — disastrous for the
/// `pending: Option<Array>` async-pipeline pattern.
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    install_error_handler();

    // Null sentinel for absent biases and global_scale (ch-18 F1).
    let biases_arr = match biases {
        Some(b) => b.inner,
        None => null_sentinel(),
    };

    let mode_cstr = mode_to_cstr(mode, "dequantize")?;

    let gs = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bs = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };

    // global_scale and dtype are optional (not used for affine/mxfp8 lookup).
    // Use the cached null sentinel — do NOT free.
    let global_scale = null_sentinel();
    let dtype_opt = sys::mlx_optional_dtype {
        value: sys::mlx_dtype::MLX_BFLOAT16, // unused when has_value=false
        has_value: false,
    };

    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_dequantize(
                &raw mut res,
                w.inner,
                scales.inner,
                biases_arr,
                gs,
                bs,
                mode_cstr.as_ptr(),
                global_scale,
                dtype_opt,
                s,
            )
        })
    };

    // Do NOT free biases_arr or global_scale — both are the cached null sentinel.

    unsafe { check_status(status, "dequantize") }?;
    Ok(Array { inner: res })
}

/// Quantize a floating-point tensor using MLX's affine integer-affine codec.
///
/// Mirrors `mx.quantize(w, group_size=..., bits=..., mode="affine")` from
/// mlx-lm-turboquant's `mixed_quant_cache.py`. Returns the canonical 3-tuple
/// `(codes_u32, scales, biases)` MLX uses for affine-quantized tensors:
/// - `codes`: U32, shape `[..., dim / (32/bits)]` — bit-packed integer codes.
/// - `scales`: same dtype as input (typically bf16/f16/f32), shape
///   `[..., dim / group_size]`.
/// - `biases`: same dtype as input, same shape as `scales`.
///
/// Caller passes the inverse triple to [`quantized_matmul`] / [`dequantize`]
/// or to the mixed-quant SDPA helper.
///
/// # Why this wrapper exists
///
/// The byte-for-byte port of mlx-lm-turboquant's `MixedQuantKVCache` stores
/// K/V as the 3-tuple emitted by `mx.quantize` and feeds the tuple directly
/// into two `mx.quantized_matmul` calls inside SDPA — eliminating the
/// per-decode-step full dequantize that dominates rMLX's current k8v4 hot
/// path.
pub fn quantize(
    x: &Array,
    group_size: i32,
    bits: i32,
    device: Device,
) -> Result<(Array, Array, Array)> {
    install_error_handler();

    let mode_cstr = mode_to_cstr("affine", "quantize")?;

    let gs = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bs = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };

    // global_scale is unused for the affine codec; use the cached null sentinel.
    // Do NOT free — it is the process-global EMPTY_ARRAY_SENTINEL.
    let global_scale = null_sentinel();

    // mlx_quantize returns its 3-tuple via an mlx_vector_array.
    // SAFETY: mlx_vector_array_new returns a default-constructed empty vector.
    let mut vec_res = unsafe { sys::mlx_vector_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_quantize(
                &raw mut vec_res,
                x.inner,
                gs,
                bs,
                mode_cstr.as_ptr(),
                global_scale,
                s,
            )
        })
    };
    // Do NOT free global_scale — it is the cached null sentinel.

    // Check status before reading from vec_res.
    let extract_result = (|| -> Result<(Array, Array, Array)> {
        unsafe { check_status(status, "quantize") }?;

        // Verify the vector has 3 entries (codes, scales, biases).
        // SAFETY: vec_res is a valid mlx_vector_array on success.
        let len = unsafe { sys::mlx_vector_array_size(vec_res) };
        if len != 3 {
            return Err(Error::Mlx(format!(
                "quantize: expected 3-element vector_array, got {len}"
            )));
        }

        let mut codes = unsafe { sys::mlx_array_new() };
        let mut scales = unsafe { sys::mlx_array_new() };
        let mut biases = unsafe { sys::mlx_array_new() };
        // SAFETY: vec_res valid; indices in [0, len). mlx_vector_array_get
        // increments the ref-count of the returned array.
        unsafe {
            check_status(
                sys::mlx_vector_array_get(&raw mut codes, vec_res, 0),
                "quantize: get codes",
            )?;
            check_status(
                sys::mlx_vector_array_get(&raw mut scales, vec_res, 1),
                "quantize: get scales",
            )?;
            check_status(
                sys::mlx_vector_array_get(&raw mut biases, vec_res, 2),
                "quantize: get biases",
            )?;
        }
        Ok((
            Array { inner: codes },
            Array { inner: scales },
            Array { inner: biases },
        ))
    })();

    // Always free the vector wrapper (the per-element handles are independent).
    // SAFETY: vec_res is a valid handle.
    unsafe { sys::mlx_vector_array_free(vec_res) };

    extract_result
}

/// Gather quantized matmul with expert indices.
///
/// Computes `x[lhs_indices] @ W[rhs_indices].T` (dequantized) where W is a
/// 3-D quantized weight `[num_experts, out_features, packed_in]`.
///
/// Used for batched MoE expert dispatch.
///
/// - `lhs_indices`: `[total_tokens]` I32 — which token to use for each row
/// - `rhs_indices`: `[total_tokens]` I32 — which expert weight row to use
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: &Array,
    group_size: i32,
    bits: i32,
    mode: &str,
    sorted_indices: bool,
    device: Device,
) -> Result<Array> {
    install_error_handler();
    // Use cached null sentinel for absent biases and lhs_indices (ch-18 F1).
    let biases_arr = match biases {
        Some(b) => b.inner,
        None => null_sentinel(),
    };
    // lhs_indices is optional ("may be null").
    let lhs_arr = match lhs_indices {
        Some(l) => l.inner,
        None => null_sentinel(),
    };
    let mode_cstr = mode_to_cstr(mode, "gather_qmm")?;
    let gs = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bs = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };
    let mut res = unsafe { sys::mlx_array_new() };
    let status = unsafe {
        with_stream(device, |s| {
            sys::mlx_gather_qmm(
                &raw mut res,
                x.inner,
                w.inner,
                scales.inner,
                biases_arr,
                lhs_arr,
                rhs_indices.inner,
                true, // transpose_w
                gs,
                bs,
                mode_cstr.as_ptr(),
                sorted_indices,
                s,
            )
        })
    };
    // Do NOT free biases_arr or lhs_arr — they are the cached null sentinel.
    unsafe { check_status(status, "gather_qmm") }?;
    Ok(Array { inner: res })
}
