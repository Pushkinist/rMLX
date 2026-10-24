//! Signed block Walsh-Hadamard transform in one Metal kernel.
//!
//! The body (`metal/signed_block_hadamard.metal`) and its header
//! (`metal/hadamard_header.metal`) are ported from MLX's `hadamard_n` and
//! `radix_func` (`mlx/backend/metal/kernels/hadamard.h`, ml-explore/mlx
//! v0.32.3, MIT). One threadgroup transforms one block: it reads the input
//! dtype, applies the signs and the butterfly in f32 in registers and
//! threadgroup memory, and rounds once to the input dtype at the store. No f32
//! tensor reaches device memory and no cast runs beside it.
//!
//! The kernel keys on the block, the width and the direction, never on an
//! architecture.

use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device};

use crate::hadamard::Direction;

const HEADER: &str = include_str!("metal/hadamard_header.metal");
const SOURCE: &str = include_str!("metal/signed_block_hadamard.metal");

/// The kernel's radix, as MLX chooses it: a thread holds 16 values, so a block
/// runs on `block / 16` threads. The body derives the same radix from `N`.
const MAX_RADIX: usize = 16;

static KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn signed_block_hadamard_kernel() -> Result<&'static MetalKernel> {
    KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_signed_block_hadamard",
                HEADER,
                SOURCE,
                &["x", "signs", "scale"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("signed block Hadamard kernel init: {e}")))
}

/// `H (D x)` ([`Direction::Forward`]) or `D (H x)` ([`Direction::Inverse`])
/// over every `block` run of the last axis of `x`, in one dispatch. The output
/// has the shape and the dtype of `x`.
///
/// `signs` is the `[width]` f32 diagonal `D`, `width` the last axis of `x`;
/// `scale` is the 0-D f32 `1 / sqrt(block)`. The caller checks the width;
/// `block` is a power of two from 2 to [`crate::hadamard::MAX_BLOCK`] that
/// divides it.
///
/// # Errors
/// `Error::Mlx` when the dispatch fails.
pub(crate) fn signed_block_hadamard_gpu(
    x: &Array,
    signs: &Array,
    scale: &Array,
    block: usize,
    direction: Direction,
    device: Device,
) -> Result<Array> {
    let shape = x.shape();
    let width = shape.last().copied().unwrap_or(0);
    let elements: i64 = shape.iter().map(|&d| i64::from(d)).product();
    let to_i32 = |what: &str, v: i64| {
        i32::try_from(v)
            .map_err(|_| Error::Mlx(format!("signed block Hadamard {what} {v} exceeds i32")))
    };
    let blocks = to_i32("block count", elements / block as i64)?;
    let threads = to_i32("threadgroup size", (block / block.min(MAX_RADIX)) as i64)?;

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(x)?;
    invoke.add_input(signs)?;
    invoke.add_input(scale)?;
    invoke.add_output_shape(&shape, x.dtype())?;
    invoke.set_template_dtype("T", x.dtype())?;
    invoke.set_template_int("N", to_i32("block", block as i64)?)?;
    invoke.set_template_int("WIDTH", width)?;
    invoke.set_template_int("INVERSE", i32::from(direction == Direction::Inverse))?;
    invoke.set_grid(threads, blocks, 1)?;
    invoke.set_thread_group(threads, 1, 1)?;

    let mut outputs = signed_block_hadamard_kernel()?.apply(invoke, device)?;
    tracing::trace!(
        block,
        width,
        blocks,
        ?direction,
        "signed block Hadamard dispatched"
    );
    if outputs.len() != 1 {
        return Err(Error::Mlx(format!(
            "signed block Hadamard: expected 1 output, got {}",
            outputs.len()
        )));
    }
    Ok(outputs.remove(0))
}

#[cfg(test)]
#[path = "hadamard_msl_tests.rs"]
mod hadamard_msl_tests;
