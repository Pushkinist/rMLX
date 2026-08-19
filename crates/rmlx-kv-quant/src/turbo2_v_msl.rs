// Note: this file contains no unsafe blocks. The original unsafe_code allow was
// pre-emptive for a planned slice::from_raw_parts MSL dispatch path. When that
// path is implemented, add #[allow(unsafe_code, reason = "...")] per-function
// with a SAFETY comment at each unsafe block per the FFI policy in docs/FFI.md.

//! TurboQuant V2 (Lloyd-Max 2-bit) Metal kernels.
//!
//! # What this is
//!
//! GPU (Metal) quantize / dequantize kernels for the 2-bit Lloyd-Max N(0,1)
//! TurboQuant codebook (`crate::turboquant::CODEBOOK_2BIT`), mirroring
//! [`crate::turboquant_msl`]'s V4 path and the
//! [`crate::k8vturbo3_append_msl`] V3 path so that
//! [`KvStorage::K8VTurbo2`](super::storage::KvStorage::K8VTurbo2) can stay on
//! the GPU when the V-side dispatch is eventually re-enabled.
//!
//! # File-name choice
//!
//! Named `turbo2_v_msl.rs`; conceptually these are the
//! same plain quantize / dequantize kernels as `turboquant_msl.rs`, not the
//! fused append-into-flash-buffer kernels of the K8V4 path. K8VTurbo2 does
//! not use TurboFlash; the standard `QuantV::append` / `dequantize_choice`
//! GPU path would call these functions for the `bits == 2` branch — but the
//! The update site keeps the V-side on CPU mirroring K8VTurbo3.
//!
//! # Codebook (Lloyd-Max optimal, N(0,1))
//!
//! Four centroids, bit-exact with `crate::turboquant::CODEBOOK_2BIT`.
//! Maximum centroid `CB2_MAX = 1.51` is the denominator for the per-group
//! scale: `scale = max(|x|) / CB2_MAX`.
//!
//! Three decision boundaries are midpoints between consecutive centroids,
//! exactly what `nearest_centroid` computes on the CPU.
//!
//! # Pack format
//!
//! For a group of `GROUP_SIZE = 32` elements at 2 bits / element:
//!
//! - 32 × 2 = 64 bits = exactly **2 `u32`** words.
//! - Element `e` occupies bits `[e*2 .. e*2+2)` of the concatenated 64-bit
//!   little-endian stream. This matches `pack_index` / `unpack_index` in
//!   `crate::turboquant` for `bits = 2` reinterpreted as 2 LE `u32`s.
//!
//! Two of 32 threads in a threadgroup (`elem ∈ {0, 16}`) act as writer
//! lanes — one per output word. Each writer accumulates the bits from all
//! 32 indices into a 64-bit local register and writes the low 32 bits of
//! its target word. (Unlike the V3 path the V2 packing is word-aligned at
//! every 16th element, so the writer never spans a u32 boundary — kept the
//! same signed-shift accumulator pattern as V3 for code-shape uniformity
//! with `k8vturbo3_append_msl.rs`.)
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Dispatch status
//!
//! The hot K8VTurbo2 V-side update path forces `Device::Cpu` in
//! `update_k8vturbo2` (`kvcache/update.rs`), mirroring the K8VTurbo3
//! decision (Metal 3-bit kernel showed −3.5% to −6.9% TPS, failing the −2%
//! gate). This module ships the kernels with full unit-test coverage of
//! bit-exact CPU↔GPU equivalence so that re-wiring it later (e.g. once a
//! bench shows a TPS win) is a one-line change at the dispatch site.

#![allow(dead_code)] // Future-gated hook, see module-level "Dispatch status".

use crate::turboquant::GROUP_SIZE;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};
use std::sync::OnceLock;

// -- Codebook constants (mirror crate::turboquant::CODEBOOK_2BIT) -------

/// Lloyd-Max optimal N(0,1) codebook for 2-bit (4 entries).
///
/// Kept in Rust as a documentation anchor; the MSL kernel embeds the same
/// values via `as_type<float>(0x...)` so any future change is made in one
/// place first and mirrored.
#[allow(dead_code)]
const CB2: [f32; 4] = [-1.51, -0.453, 0.453, 1.51];

/// Maximum codebook centroid. Denominator for the per-group scale.
#[allow(dead_code)]
const CB2_MAX: f32 = 1.51;

// -- MSL kernel sources ------------------------------------------------------

/// MSL header — embeds the 4 Lloyd-Max N(0,1) 2-bit centroids and their 3
/// midpoint decision boundaries as bit-exact `as_type<float>(0x...)`
/// constants.
///
/// Bit patterns derived from `f32::to_bits` on the values in
/// `crate::turboquant::CODEBOOK_2BIT`. Verified by the `cb2_constants_bit_exact`
/// unit test below.
const V2_KERNEL_HEADER: &str = include_str!("metal/turbo2_v_header.metal");

/// MSL body for `rmlx_tq2_quantize`.
///
/// Grid: `(N_groups * 32, 1, 1)`. Threadgroup: `(32, 1, 1)`.
///
/// Pack format: 2 `u32` per group of 32 elements, LSB-first across the
/// 64-bit per-group stream. Element `e` occupies bits `[e*2 .. e*2+2)`.
///
/// Inputs:
/// - `inp` f32 `[N_groups * 32]` — flat input elements.
///
/// Outputs:
/// - `codes` u32 `[N_groups * 2]`
/// - `scales` f32 `[N_groups]` — per-group scale = max(|x|) / CB2_MAX.
const V2_QUANTIZE_SOURCE: &str = include_str!("metal/turbo2_v_quantize.metal");

/// MSL body for `rmlx_tq2_dequantize`.
///
/// Grid: `(N_groups * 32, 1, 1)`. Threadgroup: `(32, 1, 1)`.
///
/// Each thread decodes one output element. Threads 0..1 load the 2 `u32`
/// words into threadgroup shared memory, then every thread extracts its own
/// 2-bit index via a window shift across the word boundary (V2 windows
/// never straddle, but the cross-word formulation matches the V3 kernel
/// shape).
///
/// Inputs:
/// - `codes` u32 `[N_groups * 2]`
/// - `scales` f32 `[N_groups]`
///
/// Outputs:
/// - `out` OutT `[N_groups * 32]`
const V2_DEQUANTIZE_SOURCE: &str = include_str!("metal/turbo2_v_dequantize.metal");

// -- Kernel singletons -------------------------------------------------------

static V2_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static V2_DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn v2_quant_kernel() -> Result<&'static MetalKernel> {
    V2_QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq2_quantize",
                V2_KERNEL_HEADER,
                V2_QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tq2_quantize kernel init: {e}")))
}

fn v2_dequant_kernel() -> Result<&'static MetalKernel> {
    V2_DEQUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq2_dequantize",
                V2_KERNEL_HEADER,
                V2_DEQUANTIZE_SOURCE,
                &["codes", "scales"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tq2_dequantize kernel init: {e}")))
}

// -- Public API --------------------------------------------------------------

/// GPU TurboQuant V2 quantize.
///
/// Quantize `x` (any shape, total elements must be a multiple of
/// [`GROUP_SIZE`] = 32) using the Lloyd-Max N(0,1) 2-bit codebook.
///
/// Returns `(codes, scales)`:
/// - `codes`: `u32` array of shape `[total_elems * 2 / 32]` — 2 u32 per
///   group of 32 elements, LSB-first across a 64-bit per-group stream.
/// - `scales`: `f32` array of shape `[total_elems / 32]` — one scale per
///   group.
// f32-out-ok: `codes` is u32; the f32 `scales` are read back only by
// `turbo_dequantize_v2_gpu`, an MSL kernel that declares them
// `device const float*` and writes its own output at the caller's dtype. No MLX
// op would take its operand width from them, the way `quantized_matmul` and
// `dequantize` take theirs from an `mx.quantize` 3-tuple.
pub fn turbo_quantize_v2_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    let shape = x.shape();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_quantize_v2_gpu: total elements {total_elems} not a multiple \
             of GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    let n_groups = total_elems / GROUP_SIZE;

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

    let kernel = v2_quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    invoke.add_output_shape(&[(n_groups * 2) as i32], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;

    invoke.set_grid((n_groups * GROUP_SIZE) as i32, 1, 1)?;
    invoke.set_thread_group(GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_quantize_v2_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    tracing::trace!(n_groups, total_elems, "tq2 quantize dispatched");
    Ok((codes, scales))
}

/// GPU TurboQuant V2 dequantize.
///
/// Reconstruct an `out_dtype` tensor from `(codes, scales)` produced by
/// [`turbo_quantize_v2_gpu`].
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard catches non-float dtypes (integers, quantized) and coerces them to f32 output; extending the match exhaustively would require re-testing every new Dtype variant added upstream"
)]
pub fn turbo_dequantize_v2_gpu(
    codes: &Array,
    scales: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    let n_words_total: usize = codes.shape().iter().map(|&d| d as usize).product();
    if !n_words_total.is_multiple_of(2) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v2_gpu: codes len {n_words_total} not a multiple of 2"
        )));
    }
    let n_groups_codes = n_words_total / 2;
    let n_groups_scales: usize = scales.shape().iter().map(|&d| d as usize).product();
    if n_groups_codes != n_groups_scales {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v2_gpu: n_groups from codes ({n_groups_codes}) != \
             from scales ({n_groups_scales})"
        )));
    }
    let n_groups = n_groups_codes;
    let total_elems = n_groups * GROUP_SIZE;

    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v2_gpu: original_shape product {shape_product} != \
             expected {total_elems}"
        )));
    }

    let codes_flat = codes.reshape(&[(n_groups * 2) as i32], device)?;
    let scales_flat = scales.reshape(&[n_groups as i32], device)?;

    let kernel_out_dtype = match out_dtype {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => out_dtype,
        _ => Dtype::F32,
    };

    let kernel = v2_dequant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_output_shape(&[total_elems as i32], kernel_out_dtype)?;
    invoke.set_template_dtype("OutT", kernel_out_dtype)?;

    invoke.set_grid((n_groups * GROUP_SIZE) as i32, 1, 1)?;
    invoke.set_thread_group(GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_dequantize_v2_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);
    let out_flat = if kernel_out_dtype == out_dtype {
        out_flat
    } else {
        out_flat.astype(out_dtype, device)?
    };

    tracing::trace!(n_groups, total_elems, "tq2 dequantize dispatched");

    if original_shape.len() == 1 && original_shape[0] == total_elems as i32 {
        return Ok(out_flat);
    }
    out_flat.reshape(original_shape, device)
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
#[path = "turbo2_v_msl_tests.rs"]
mod tests;
