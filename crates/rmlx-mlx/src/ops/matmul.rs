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

/// Width of the K tile the `qmm_t_splitk` kernels step the contracted
/// dimension by.
const SPLITK_K_TILE: i32 = 32;

/// Smallest batch that could reach a tiled `qmm` kernel for this shape.
///
/// MLX picks the vector/tiled crossover in `get_qmv_batch_limit`, keyed off the
/// GPU architecture as well as the shape. mlx-c exposes no architecture query,
/// so take the minimum over every branch of that function for the shape: below
/// it the vector kernel runs on every Apple GPU and split-K never applies.
///
/// Erring low is the safe direction — it can only pad a batch the vector kernel
/// would have handled, never leave a tiled one unguarded. Between this floor
/// and the device's real limit (10-32) the guard does pad batches MLX would
/// have run on the vector kernel; that cost is the price of not knowing the
/// architecture, and the per-shape floor keeps it far narrower than a flat one.
fn qmv_batch_limit_floor(k: i32, n: i32) -> i32 {
    if k <= 2048 && n <= 2048 {
        14
    } else if k <= 4096 && n <= 4096 {
        10
    } else {
        6
    }
}

/// The K partition MLX's `qmm_t_splitk` would use for this shape, or `None`
/// when it falls back to the unsplit `qmm` kernel.
///
/// Mirrors the split-K choice in MLX's Metal `quantized.cpp`.
fn splitk_k_partition(m: i32, n: i32, k: i32, group_size: i32) -> Option<i32> {
    if group_size <= 0 || k < group_size {
        return None;
    }
    let n_tiles = (n + SPLITK_K_TILE - 1) / SPLITK_K_TILE;
    let m_tiles = (m + SPLITK_K_TILE - 1) / SPLITK_K_TILE;
    let tiles = n_tiles.checked_mul(m_tiles)?;
    if tiles <= 0 {
        return None;
    }
    let mut split_k = (512 / tiles).max(1).min(k / group_size);
    while split_k > 1 && k % (split_k * group_size) != 0 {
        split_k -= 1;
    }
    (split_k > 1).then(|| k / split_k)
}

/// Row count to run this quantized matmul at so MLX's split-K partition is a
/// whole number of K tiles, or `None` when `m` already is one.
///
/// MLX aligns each split-K partition to `group_size`, but the `qmm_t_splitk`
/// kernels tile the contracted dimension by [`SPLITK_K_TILE`] and do not bound
/// that loop. A codec whose group is narrower than the tile — nvfp4's 16 — can
/// therefore be handed a partition that is not a whole number of tiles, and the
/// kernel reads past it into the following group's codes and scales, silently
/// corrupting every output element. Growing the batch moves MLX onto a coarser
/// split whose partition divides the tile evenly; appended zero rows cannot
/// change the rows that are kept.
///
/// Upstream fixed this by aligning the partition to `max(group_size, 32)`; the
/// fix is on MLX `main` and in no release this builds against.
fn splitk_safe_rows(m: i32, n: i32, k: i32, group_size: i32) -> Option<i32> {
    if group_size % SPLITK_K_TILE == 0 || m < qmv_batch_limit_floor(k, n) {
        return None;
    }
    let tiles_whole = |rows: i32| {
        splitk_k_partition(rows, n, k, group_size).is_none_or(|p| p % SPLITK_K_TILE == 0)
    };
    if tiles_whole(m) {
        return None;
    }
    // Once the tile grid alone reaches MLX's 512-threadgroup target the split
    // collapses to one partition and the unsplit kernel runs, so the answer is
    // always inside this bound.
    let n_tiles = (n + SPLITK_K_TILE - 1) / SPLITK_K_TILE;
    let unsplit_rows = SPLITK_K_TILE * (512 / n_tiles.max(1) + 1);
    let rows = (m + 1..=m.max(unsplit_rows)).find(|&rows| tiles_whole(rows));
    if rows.is_none() {
        tracing::warn!(
            m,
            n,
            k,
            group_size,
            "no tile-whole row count below the unsplit bound; dispatching a \
             partition MLX will over-read"
        );
    }
    rows
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
    // Split-K only runs on the transposed path, and only a group narrower than
    // the K tile can land on a partition that is not tile-whole. Both are
    // decidable from the arguments, so test them before touching the arrays:
    // this is every projection of every layer of every decode step, and each
    // `Array::dim` costs two FFI calls.
    if !transpose_w || group_size % SPLITK_K_TILE == 0 {
        return quantized_matmul_packed(
            x,
            w,
            scales,
            biases,
            group_size,
            bits,
            mode,
            transpose_w,
            device,
        );
    }

    // MLX reaches the split kernel only when `out.size() / M / N == 1`. A 2-D
    // weight is the sub-case where that holds and MLX's own `M` is `x` flattened
    // against its last axis, which is what `rows` computes. A weight of rank 3
    // with unit leading dims also satisfies MLX's test, but there `M` becomes
    // `x.shape(-2)` instead, so matching on it would need a different `rows`;
    // no caller builds one (every site passes a 2-D `Linear`/`Embedding`
    // weight). `rows` further assumes `x` is row-contiguous, as MLX does; when
    // it is not, MLX stays on the unsplit kernel and this guard can pad a shape
    // that never needed it — wasteful, never wrong.
    let x_ndim = x.ndim();
    let mut padded_rows = None;
    if w.ndim() == 2 && x_ndim >= 2 {
        let k = x.dim(x_ndim - 1)?;
        if k > 0 {
            let mut rows: i32 = 1;
            for axis in 0..x_ndim - 1 {
                rows = rows.saturating_mul(x.dim(axis)?);
            }
            let n = w.dim(0)?;
            padded_rows =
                splitk_safe_rows(rows, n, k, group_size).map(|padded| (rows, k, n, padded));
        }
    }

    let Some((rows, k, n, padded)) = padded_rows else {
        return quantized_matmul_packed(
            x,
            w,
            scales,
            biases,
            group_size,
            bits,
            mode,
            transpose_w,
            device,
        );
    };

    tracing::debug!(
        rows,
        padded,
        n,
        k,
        group_size,
        bits,
        mode,
        "quantized_matmul: padding rows off MLX's misaligned split-K partition"
    );

    let flat = x.reshape(&[rows, k], device)?;
    let grown = crate::ops::pad(&flat, &[0], &[0], &[padded - rows], device)?;
    let out = quantized_matmul_packed(
        &grown,
        w,
        scales,
        biases,
        group_size,
        bits,
        mode,
        transpose_w,
        device,
    )?;
    let kept = out.slice(&[0, 0], &[rows, n], &[1, 1], device)?;

    let mut out_shape = x.shape();
    if let [.., last] = out_shape.as_mut_slice() {
        *last = n;
    }
    kept.reshape(&out_shape, device)
}

fn quantized_matmul_packed(
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
    quantize_mode(x, group_size, bits, "affine", device)
}

/// Quantize with an explicit MLX codec mode (`"affine"`, `"mxfp8"`,
/// `"mxfp4"`, `"nvfp4"`).
///
/// Same 3-tuple as [`quantize`]; the no-bias codecs (`mxfp*`/`nvfp4`) still
/// return a `biases` array (MLX emits a placeholder) — callers that pass
/// `None` biases to [`quantized_matmul`] / [`gather_qmm`] simply ignore it.
/// [`quantize`] is the `"affine"` specialization.
pub fn quantize_mode(
    x: &Array,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<(Array, Array, Array)> {
    install_error_handler();

    let mode_cstr = mode_to_cstr(mode, "quantize")?;

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

        // Affine emits 3 entries (codes, scales, biases); the mxfp*/nvfp4
        // codecs emit 2 (codes, scales) — biases is then left as the
        // default-constructed empty array, which the caller discards.
        // SAFETY: vec_res is a valid mlx_vector_array on success.
        let len = unsafe { sys::mlx_vector_array_size(vec_res) };
        if len != 2 && len != 3 {
            return Err(Error::Mlx(format!(
                "quantize: expected 2- or 3-element vector_array, got {len}"
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
            if len == 3 {
                check_status(
                    sys::mlx_vector_array_get(&raw mut biases, vec_res, 2),
                    "quantize: get biases",
                )?;
            }
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

#[cfg(test)]
#[path = "matmul_tests.rs"]
mod matmul_tests;
