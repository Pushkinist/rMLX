// Note: this file contains no unsafe blocks. The unsafe_code allow is held
// for future slice::from_raw_parts MSL dispatch paths consistent with the rest
// of this crate's MSL kernels.

//! Viterbi-trellis (TCQ) 3-bit Metal kernel.
//!
//! # What this is
//!
//! GPU (Metal) quantize kernel for `turbo3_tcq` — Viterbi-optimal trellis
//! assignment over the standard Lloyd-Max N(0,1) 3-bit codebook. Mirrors the
//! CPU implementation in [`crate::tcq`] bit-for-bit (the decoder reuses
//! [`crate::k8vturbo3_append_msl::turbo_dequantize_v3_gpu`] — TCQ is encode-side
//! only).
//!
//! # Dispatch
//!
//! - 1D grid `(N_groups, 1, 1)` where `N_groups = total_elems / 32`.
//! - Threadgroup `(1, 1, 1)`: one thread per [`crate::turboquant::GROUP_SIZE`]-block.
//!   The Viterbi forward + back-trace runs entirely in thread registers
//!   (TCQ_NUM_STATES = 4 ⇒ 4 f32 costs + 32×4×2 bytes back-pointer table per
//!   thread — well within Metal's per-thread register budget).
//!
//! # Pack format (output)
//!
//! Identical to plain turbo3: 3 `u32` per group of 32 elements, LSB-first
//! across the concatenated 96-bit per-group stream. Each thread writes its
//! own 3 u32 codes and 1 f32 scale to global memory.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Dispatch status
//!
//! Ships as a future-reference hook (mirrors `turbo2_v_msl.rs` /
//! `k8vturbo3_append_msl.rs`). The hot `K8VTurbo3Tcq` V-side update path forces
//! `Device::Cpu` in [`crate::kvcache::KvCache::update_k8vturbo3_tcq`] — the
//! sequential per-thread Viterbi loop is bandwidth-bound and the prior
//! K8VTurbo3 / K8VTurbo2 MSL hooks both regressed −2 % decode TPS gates. The
//! kernel is parity-tested CPU↔GPU (see `tcq_v_msl_tests.rs`) so re-wiring
//! later is a one-line change at the dispatch site.

#![allow(dead_code)] // Future-gated hook, see module-level "Dispatch status".

use crate::turboquant::GROUP_SIZE;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};
use std::sync::OnceLock;

// Lloyd-Max optimal N(0,1) 3-bit codebook (kept here as a documentation anchor
// — the MSL kernel embeds the same constants via `as_type<float>(0x...)`).
#[allow(dead_code)]
const CB3: [f32; 8] = [
    -2.151_944_9,
    -1.343_908_5,
    -0.756_004_75,
    -0.245_093_99,
    0.245_093_99,
    0.756_004_75,
    1.343_908_5,
    2.151_944_9,
];

/// Maximum codebook centroid. Denominator for the per-group scale.
#[allow(dead_code)]
const CB3_MAX: f32 = 2.151_944_9;

// MSL kernel — Viterbi forward + back-trace + pack. Each thread owns one
// `GROUP_SIZE=32` block. Uses 4-state trellis (TCQ_NUM_STATES = 4 in
// `crate::tcq`). Transition: `next = ((state << 1) | (level & 1)) & 3`.
const TCQ_KERNEL_HEADER: &str = include_str!("metal/tcq_v_header.metal");

const TCQ_QUANTIZE_SOURCE: &str = include_str!("metal/tcq_v_quantize.metal");

// -- Kernel singleton --------------------------------------------------------

static TCQ_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn tcq_quant_kernel() -> Result<&'static MetalKernel> {
    TCQ_QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tcq_quantize",
                TCQ_KERNEL_HEADER,
                TCQ_QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tcq_quantize kernel init: {e}")))
}

// -- Public API --------------------------------------------------------------

/// GPU TurboQuant-TCQ V3 quantize.
///
/// Same output layout as [`crate::k8vturbo3_append_msl::turbo_quantize_v3_gpu`]:
/// `(codes, scales)` with the standard 3-u32-per-group pack. The decoder is
/// identical to plain turbo3.
///
/// # Errors
///
/// Returns `Error::Quant` if `total_elems` is not a multiple of
/// [`GROUP_SIZE`] = 32.
// f32-out-ok: `codes` is u32 and `scales` is f32. This encoder has no
// production caller today — the TCQ V store quantizes on the CPU
// (`tcq::tcq_quantize_v3`) and only the tests drive this kernel. If it is ever
// wired, its reader is that store's MSL decode, which declares the scales
// `device const float*`. Either way no MLX
// op would take its operand width from them, the way `quantized_matmul` and
// `dequantize` take theirs from an `mx.quantize` 3-tuple.
pub fn tcq_quantize_v3_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    let shape = x.shape();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "tcq_quantize_v3_gpu: total elements {total_elems} not a multiple \
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

    let kernel = tcq_quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    invoke.add_output_shape(&[(n_groups * 3) as i32], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;

    // One thread per group; threadgroup = 1.
    invoke.set_grid(n_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "tcq_quantize_v3_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    tracing::trace!(n_groups, total_elems, "tcq3 quantize dispatched");
    Ok((codes, scales))
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
#[path = "tcq_v_msl_tests.rs"]
mod tcq_v_msl_tests;
