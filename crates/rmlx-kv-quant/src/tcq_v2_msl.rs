// Note: this file contains no unsafe blocks. The unsafe_code allow is held
// for future slice::from_raw_parts MSL dispatch paths consistent with the rest
// of this crate's MSL kernels.

//! Viterbi-trellis (TCQ) 2-bit Metal kernel.
//!
//! # What this is
//!
//! GPU (Metal) quantize kernel for `turbo2_tcq` — Viterbi-optimal trellis
//! assignment over the standard Lloyd-Max N(0,1) 2-bit codebook. Mirrors the
//! CPU implementation in [`crate::tcq::tcq_quantize_v2`] bit-for-bit (the
//! decoder reuses [`crate::turboquant::turbo_dequantize`] — TCQ is encode-side
//! only, at both 2-bit and 3-bit).
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
//! Identical to plain turbo2: 2-bit indices packed 16 values per u32.
//! `N_groups` u32 words for codes + `N_groups` f32 words for scales.
//! Each thread writes its own 1 u32 code word and 1 f32 scale to global memory.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Dispatch status
//!
//! Ships as a future-reference hook (mirrors `tcq_v_msl.rs` for 3-bit).
//! The hot `K8VTurbo2Tcq` V-side update path forces `Device::Cpu` in
//! [`crate::kvcache::KvCache::update_k8vturbo2_tcq`] — the sequential
//! per-thread Viterbi loop is bandwidth-bound on current Apple Silicon.
//! The kernel is parity-tested CPU↔GPU (see `tcq_v2_msl_tests.rs`) so
//! re-wiring later is a one-line change at the dispatch site.

#![allow(dead_code)] // Future-gated hook, see module-level "Dispatch status".

use crate::turboquant::GROUP_SIZE;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};
use std::sync::OnceLock;

// Lloyd-Max optimal N(0,1) 2-bit codebook (kept here as a documentation anchor
// — the MSL kernel embeds the same constants via `as_type<float>(0x...)`).
#[allow(dead_code)]
const CB2: [f32; 4] = [-1.51, -0.453, 0.453, 1.51];

/// Maximum codebook centroid. Denominator for the per-group scale.
#[allow(dead_code)]
const CB2_MAX: f32 = 1.51;

// MSL kernel — Viterbi forward + back-trace + pack. Each thread owns one
// `GROUP_SIZE=32` block. Uses 4-state trellis (TCQ_NUM_STATES = 4 in
// `crate::tcq`). Transition: `next = ((state << 1) | (level & 1)) & 3`.
const TCQ2_KERNEL_HEADER: &str = include_str!("metal/tcq_v2_header.metal");

// Pack format for 2-bit: 32 elements × 2 bits = 64 bits = 2 u32 words per group.
// Each thread writes 2 u32 code words + 1 f32 scale.
const TCQ2_QUANTIZE_SOURCE: &str = include_str!("metal/tcq_v2_quantize.metal");

// -- Kernel singleton --------------------------------------------------------

static TCQ2_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn tcq2_quant_kernel() -> Result<&'static MetalKernel> {
    TCQ2_QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tcq2_quantize",
                TCQ2_KERNEL_HEADER,
                TCQ2_QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tcq2_quantize kernel init: {e}")))
}

// -- Public API --------------------------------------------------------------

/// GPU TurboQuant-TCQ V2 quantize.
///
/// Output layout: `(codes, scales)` where `codes` has shape `[N_groups * 2]`
/// (2 u32 words per 32-element group = 64-bit pack), and `scales` has shape
/// `[N_groups]`. The decoder is identical to plain turbo2 (bit-packed 2-bit).
///
/// # Errors
///
/// Returns `Error::Quant` if `total_elems` is not a multiple of
/// [`GROUP_SIZE`] = 32.
pub fn tcq_quantize_v2_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    let shape = x.shape();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "tcq_quantize_v2_gpu: total elements {total_elems} not a multiple \
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

    let kernel = tcq2_quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    // 2 u32 words per group.
    invoke.add_output_shape(&[(n_groups * 2) as i32], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;

    // One thread per group; threadgroup = 1.
    invoke.set_grid(n_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "tcq_quantize_v2_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    tracing::trace!(n_groups, total_elems, "tcq2 quantize dispatched");
    Ok((codes, scales))
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
#[path = "tcq_v2_msl_tests.rs"]
mod tcq_v2_msl_tests;
