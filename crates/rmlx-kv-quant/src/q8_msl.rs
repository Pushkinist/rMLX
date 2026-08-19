// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! q8_0 affine 8-bit MSL (Metal Shading Language) GPU kernels.
//!
//! # What this is
//!
//! GPU (Metal) versions of the CPU `q8_quantize` / `q8_dequantize` paths in
//! `kv_cache.rs`. Sibling to `turboquant_msl.rs` and `planarquant_msl.rs`.
//!
//! # Algorithm (matches CPU path exactly)
//!
//! For each group of 128 f32 elements:
//! 1. `abs_max = max(|x_i|)`.
//! 2. `scale = abs_max / 127`. If `abs_max == 0`, `scale = 0`.
//! 3. `code = clamp(round(x_i / scale), -128, 127) as i8`.
//! 4. Reconstruction: `recon = scale * (code as f32)`.
//!
//! # Storage layout
//!
//! - `codes`: `u32 [total_elems / 4]`. Each u32 packs 4 i8 codes, byte 0 holds
//!   element with offset 0 in the group of 4, byte 3 holds offset 3, etc.
//!   Native LE byte order, so the same memory can be reinterpreted as i8.
//! - `scales`: `f32 [n_groups]` where `n_groups = total_elems / 128`.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

#![allow(clippy::cloned_instead_of_copied)]
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};
use std::sync::OnceLock;

// q8_0 group size: 128 elements per scale.
/// Number of elements per affine q8_0 quantization group.
pub const Q8_GROUP_SIZE: usize = 128;

// Number of u32 packing words per group: 128 elements / 4 bytes per word = 32.
const WORDS_PER_GROUP: usize = Q8_GROUP_SIZE / 4;

// Threads per group of 128 elements: 32 threads, each handles 4 elements.
const THREADS_PER_GROUP: usize = WORDS_PER_GROUP;

// MSL body: q8_0 quantize.
// Grid: (n_groups * 32, 1, 1). Threadgroup: (32, 1, 1).
const QUANTIZE_SOURCE: &str = include_str!("metal/q8_quantize.metal");

// MSL body: q8_0 dequantize.
// Grid: (n_groups * 128, 1, 1). Threadgroup: (128, 1, 1).
//
// Templated on OutT so we can write directly to bf16 / f16 / f32 without a
// follow-up astype kernel (saves one elementwise pass per layer per step).
const DEQUANTIZE_SOURCE: &str = include_str!("metal/q8_dequantize.metal");

static QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn quant_kernel() -> Result<&'static MetalKernel> {
    QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_q8_quantize",
                "",
                QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("q8_quantize kernel init: {e}")))
}

fn dequant_kernel() -> Result<&'static MetalKernel> {
    DEQUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_q8_dequantize",
                "",
                DEQUANTIZE_SOURCE,
                &["codes", "scales"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("q8_dequantize kernel init: {e}")))
}

/// GPU q8_0 quantize (group_size=128).
///
/// `(codes, scales)`:
/// - `codes` u32 `[total_elems / 4]` — 4 i8 per uint32 (LE byte order).
/// - `scales` f32 `[total_elems / 128]`.
///
/// REQUIRES row-major contiguous input. The custom MSL quant kernel indexes the
/// buffer by raw linear offset and ignores MLX strides, so a caller holding a
/// transposed / sliced view must `.contiguous()` it first — otherwise the
/// stored codes follow the original (wrong) physical order. Per-decode-step
/// callers already pass row-contiguous head-major `[B, kv_h, n, D]` chunks; the
/// transposed-view materialization lives at the one site that needs it
/// (`QuantK::append`).
// f32-out-ok: codec buffers, not activations — codes are u32 and the scale /
// rotation buffers are read back only by this codec's own MSL kernels, which
// declare them `device const float*`. They never become an operand of an MLX
// op, so nothing takes its width from them (an `mx.quantize` 3-tuple does:
// `quantized_matmul` and `dequantize` promote to the scales' dtype).
pub fn q8_quantize_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    let total_elems: usize = x.shape().iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(Q8_GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "q8_quantize_gpu: total elements {total_elems} not a multiple of {Q8_GROUP_SIZE}"
        )));
    }
    let n_groups = total_elems / Q8_GROUP_SIZE;
    let n_words = total_elems / 4;

    let x_flat = if x.ndim() == 1 {
        x.try_clone()?
    } else {
        x.reshape(&[total_elems as i32], device)?
    };
    let x_f32 = if x_flat.dtype() == Dtype::F32 {
        x_flat
    } else {
        x_flat.astype(Dtype::F32, device)?
    };

    let kernel = quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    invoke.add_output_shape(&[n_words as i32], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;
    invoke.set_grid((n_groups * THREADS_PER_GROUP) as i32, 1, 1)?;
    invoke.set_thread_group(THREADS_PER_GROUP as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "q8_quantize_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales))
}

/// GPU q8_0 dequantize. Output dtype is `out_dtype`; the kernel writes
/// directly to that dtype, saving a follow-up astype kernel. Acceptable
/// values: F32, Bf16, F16. Any other dtype falls back via astype.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn q8_dequantize_gpu(
    codes: &Array,
    scales: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    let n_words: usize = codes.shape().iter().map(|&d| d as usize).product();
    let n_groups: usize = scales.shape().iter().map(|&d| d as usize).product();
    let total_elems = n_groups * Q8_GROUP_SIZE;

    if n_words * 4 != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "q8_dequantize_gpu: codes ({n_words} words) and scales ({n_groups} groups) \
             disagree on total_elems"
        )));
    }
    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "q8_dequantize_gpu: original_shape product {shape_product} != expected {total_elems}"
        )));
    }

    let codes_flat = codes.reshape(&[n_words as i32], device)?;
    let scales_flat = scales.reshape(&[n_groups as i32], device)?;

    // Templated kernel writes directly to out_dtype. Restrict to the
    // dtypes that have a sensible static_cast<OutT>(float).
    let kernel_out_dtype = match out_dtype {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => out_dtype,
        _ => Dtype::F32,
    };

    let kernel = dequant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_output_shape(&[total_elems as i32], kernel_out_dtype)?;
    invoke.set_template_dtype("OutT", kernel_out_dtype)?;
    invoke.set_grid(total_elems as i32, 1, 1)?;
    invoke.set_thread_group(Q8_GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "q8_dequantize_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);

    let out_flat = if kernel_out_dtype == out_dtype {
        out_flat
    } else {
        out_flat.astype(out_dtype, device)?
    };

    if original_shape.len() == 1 && original_shape[0] == total_elems as i32 {
        return Ok(out_flat);
    }
    out_flat.reshape(original_shape, device)
}

#[cfg(test)]
#[path = "q8_msl_tests.rs"]
mod tests;
