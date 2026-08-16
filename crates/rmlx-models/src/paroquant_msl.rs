// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! ParoQuant pairwise Givens rotation Metal kernel.
//!
//! # What this is
//!
//! GPU (Metal) implementation of the per-layer pairwise Givens rotation step
//! used by `z-lab/paroquant` INT4 checkpoints before the standard affine-INT4
//! dequant + matmul.
//!
//! The rotation is applied to input activations `x` BEFORE the quantized weight
//! matmul — it is a pre-rotation of the activations, not of the weights.
//!
//! # rMLX-first on Apple Silicon
//!
//! `johndpope/llama-cpp-turboquant feature/planarquant-kv-cache` Metal kernels
//! fall back to CPU (issue #7). rMLX is the first working Apple Silicon
//! ParoQuant implementation as of 2026-05.
//!
//! # Kernel algorithm (matches upstream Python reference)
//!
//! Reference: `z-lab/paroquant/paroquant/kernels/metal/rotation.metal`
//!
//! For each group of `group_size` channels and each tile of `ROWS_PER_TILE` rows:
//! 1. Load activation rows into shared tile memory, fusing channel_scales.
//! 2. For each of `krot` rotation rounds:
//! - Load pair indices (i, j) and (cos, sin) from rotation params.
//! - Apply 2D Givens rotation in-place: a' = a*c + b*s, b' = b*c - a*s.
//! 3. Write rotated activations back.
//!
//! # Inputs and outputs
//!
//! `paro_rotate_gpu(x, packed_pairs, cos_theta, sin_theta, channel_scales, krot, group_size)`
//!
//! - `x`: `[batch, hidden]` input activations (F16 or BF16).
//! - `packed_pairs`: `[krot, hidden/2]` I32 packed pair indices
//!   (lo 16 bits = i_local, hi 16 bits = j_local within group).
//! - `cos_theta`: `[krot, hidden/2]` F16 cosine values (pre-computed on load).
//! - `sin_theta`: `[krot, hidden/2]` F16 sine values (pre-computed on load).
//! - `channel_scales`: `[hidden]` F16 per-channel scale factors (flattened from [1, hidden]).
//!
//! Output: `[batch, hidden]` rotated activations in the same dtype as input.
//!
//! # Template parameters (MLX template ints, supplied per dispatch)
//!
//! - `ROWS_PER_TILE`: rows processed per threadgroup tile (4 for batch > 1, 1 for decode).
//! - `MAX_KROT`: maximum krot supported (must be >= actual krot; 16 is safe upper bound).
//! - `MAX_GROUP_SIZE`: maximum group_size (must be >= actual group_size; 256 safe upper bound).
//!
//! MLX instantiates one kernel variant per distinct tuple, so all three are
//! compile-time constants inside the body — which two of them must be, since
//! they size arrays.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── MSL kernel source ─────────────────────────────────────────────────────────

/// MSL kernel source for the pairwise Givens rotation.
///
/// The body lives in a `.metal` file so `make check-metal-compiles` and
/// `make check-metal-format` see it; `include_str!` embeds it at compile time,
/// so the binary still carries no runtime data files.
///
/// Ported from `z-lab/paroquant/paroquant/kernels/metal/rotation.metal`.
/// Changes from upstream:
/// - `cos_theta`/`sin_theta` are separate F16 inputs (pre-computed on load from
///   the raw `theta` F16 array) matching the upstream Python `_cache_rotation`
///   pattern. This keeps the kernel identical to upstream and avoids per-dispatch
///   transcendental math.
/// - `params` is a flat I32 array `[batch, hidden, krot, group_size]` (same as
///   upstream).
/// - `ROWS_PER_TILE` / `MAX_KROT` / `MAX_GROUP_SIZE` are MLX template ints
///   supplied at dispatch, not text substituted into the source. MLX
///   instantiates one variant per distinct tuple, so the emitted constants are
///   the same as text substitution produced, and the Rust consts below stay the
///   single source of truth for the bounds the validation checks use.
const KERNEL_SOURCE: &str = include_str!("metal/paroquant_rotate.metal");

// ── Template instantiation ────────────────────────────────────────────────────

/// MAX_KROT is 16 — safe upper bound for all known PARO checkpoints (Qwen3.5:
/// krot=8). If future checkpoints exceed 16, raise this constant.
const MAX_KROT: usize = 16;

/// MAX_GROUP_SIZE is 256 — safe upper bound. Qwen3.5 uses group_size=128.
const MAX_GROUP_SIZE: usize = 256;

// ── Kernel singleton ──────────────────────────────────────────────────────────

/// One registration; `ROWS_PER_TILE` selects the variant per dispatch (1 for a
/// single-row decode step, 4 for batch / prefill).
static PARO_ROTATE_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn paro_rotate_kernel() -> Result<&'static MetalKernel> {
    PARO_ROTATE_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_paro_rotate",
                "", // no header — constants arrive as template ints
                KERNEL_SOURCE,
                &[
                    "x",
                    "packed_pairs",
                    "cos_theta",
                    "sin_theta",
                    "channel_scales",
                    "params",
                ],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("paro_rotate kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Pack raw I16 pair indices `[krot, hidden]` into I32 packed pairs
/// `[krot, hidden/2]` for the Metal kernel.
///
/// Packing: for each group of `group_size` channels, pairs at positions
/// `2*t` and `2*t+1` within the group are packed into one I32:
/// lo16 = index at position 2*t, hi16 = index at position 2*t+1.
///
/// This matches `_pack_pairs` in `paroquant/inference/backends/mlx/modules.py`.
///
/// # Errors
///
/// Returns `Err` if `group_size` is 0, odd, or `hidden` is not divisible
/// by `group_size`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn pack_pairs_cpu(
    pairs_bytes: &[u8],
    krot: usize,
    hidden: usize,
    group_size: usize,
) -> Result<Vec<i32>> {
    if group_size == 0 || !group_size.is_multiple_of(2) {
        return Err(Error::Quant(format!(
            "pack_pairs_cpu: group_size={group_size} must be even and nonzero"
        )));
    }
    if !hidden.is_multiple_of(group_size) {
        return Err(Error::Quant(format!(
            "pack_pairs_cpu: hidden={hidden} not divisible by group_size={group_size}"
        )));
    }
    let expected_bytes = krot * hidden * 2;
    if pairs_bytes.len() != expected_bytes {
        return Err(Error::Loader(format!(
            "pack_pairs_cpu: expected {expected_bytes} bytes for [krot={krot}, hidden={hidden}] I16, \
             got {}",
            pairs_bytes.len()
        )));
    }

    let half_hidden = hidden / 2;
    let half_gs = group_size / 2;
    let num_groups = hidden / group_size;
    let mut out = vec![0i32; krot * half_hidden];

    for k in 0..krot {
        for g in 0..num_groups {
            for t in 0..half_gs {
                let ch_a = g * group_size + 2 * t;
                let ch_b = g * group_size + 2 * t + 1;
                let byte_a = (k * hidden + ch_a) * 2;
                let byte_b = (k * hidden + ch_b) * 2;
                let raw_a = i16::from_le_bytes([pairs_bytes[byte_a], pairs_bytes[byte_a + 1]]);
                let raw_b = i16::from_le_bytes([pairs_bytes[byte_b], pairs_bytes[byte_b + 1]]);
                let i_local = (raw_a as u32) & 0xFFFF;
                let j_local = (raw_b as u32) & 0xFFFF;
                let packed = (i_local | (j_local << 16)) as i32;
                out[k * half_hidden + g * half_gs + t] = packed;
            }
        }
    }

    Ok(out)
}

/// GPU pairwise Givens rotation for ParoQuant.
///
/// Applies `krot` rounds of pairwise Givens rotations to activations `x`
/// before the INT4 quantized matmul.
///
/// # Arguments
///
/// - `x`: `[batch, hidden]` F16/BF16 input activations.
/// - `packed_pairs`: `[krot, hidden/2]` I32 packed pair indices.
/// - `cos_theta`: `[krot, hidden/2]` F16 cosine values (pre-computed on load).
/// - `sin_theta`: `[krot, hidden/2]` F16 sine values (pre-computed on load).
/// - `channel_scales`: `[1, hidden]` or `[hidden]` F16 per-channel scale factors.
/// - `krot`: actual number of rotation rounds (must be <= MAX_KROT=16).
/// - `group_size`: rotation group size (must be <= MAX_GROUP_SIZE=256).
///
/// # Returns
///
/// `[batch, hidden]` rotated activations in the same dtype as `x`.
///
/// # Errors
///
/// Returns `Error::Mlx` on kernel dispatch failure.
/// Returns `Error::Quant` if `krot > MAX_KROT` or `group_size > MAX_GROUP_SIZE`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn paro_rotate_gpu(
    x: &Array,
    packed_pairs: &Array,
    cos_theta: &Array,
    sin_theta: &Array,
    channel_scales: &Array,
    krot: usize,
    group_size: usize,
    device: Device,
) -> Result<Array> {
    if krot > MAX_KROT {
        return Err(Error::Quant(format!(
            "paro_rotate_gpu: krot={krot} exceeds MAX_KROT={MAX_KROT}"
        )));
    }
    if group_size > MAX_GROUP_SIZE {
        return Err(Error::Quant(format!(
            "paro_rotate_gpu: group_size={group_size} exceeds MAX_GROUP_SIZE={MAX_GROUP_SIZE}"
        )));
    }

    let shape = x.shape();
    if shape.len() != 2 {
        return Err(Error::Quant(format!(
            "paro_rotate_gpu: x must be 2D [batch, hidden], got shape {shape:?}"
        )));
    }
    let batch = shape[0] as usize;
    let hidden = shape[1] as usize;

    if !hidden.is_multiple_of(group_size) {
        return Err(Error::Quant(format!(
            "paro_rotate_gpu: hidden={hidden} not divisible by group_size={group_size}"
        )));
    }

    let num_groups = hidden / group_size;
    let half_gs = group_size / 2;

    // Pick ROWS_PER_TILE: 1 for decode (batch=1), 4 for prefill.
    let rpt: usize = if batch <= 1 { 1 } else { 4 };
    let kernel = paro_rotate_kernel()?;

    // params: [batch, hidden, krot, group_size] I32.
    let params_data: [i32; 4] = [batch as i32, hidden as i32, krot as i32, group_size as i32];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params_data.as_ptr().cast::<u8>(), params_data.len() * 4)
    };
    let params = Array::from_bytes(params_bytes, &[4], Dtype::I32)?;

    // Flatten channel_scales from [1, hidden] or [hidden] to [hidden].
    let cs_flat = if channel_scales.shape().len() > 1 {
        channel_scales.reshape(&[hidden as i32], device)?
    } else {
        channel_scales.try_clone()?
    };

    let out_dtype = x.dtype();
    let out_shape = [batch as i32, hidden as i32];

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(x)?;
    invoke.add_input(packed_pairs)?;
    invoke.add_input(cos_theta)?;
    invoke.add_input(sin_theta)?;
    invoke.add_input(&cs_flat)?;
    invoke.add_input(&params)?;
    invoke.add_output_shape(&out_shape, out_dtype)?;
    invoke.set_template_dtype("InT", out_dtype)?;
    invoke.set_template_int("ROWS_PER_TILE", rpt as i32)?;
    invoke.set_template_int("MAX_KROT", MAX_KROT as i32)?;
    invoke.set_template_int("MAX_GROUP_SIZE", MAX_GROUP_SIZE as i32)?;

    // MLX metal_kernel uses dispatchThreads (total threads, not threadgroup counts).
    // Python: grid = (ceil(batch/rpt) * half_gs, num_groups, 1), threadgroup=(half_gs,1,1).
    // → Metal launches ceil(grid.x / threadgroup.x) = ceil(batch/rpt) threadgroups in X.
    // → tile_idx = threadgroup_position_in_grid.x ranges 0..ceil(batch/rpt)-1.
    // We must pass grid=(tiles*half_gs, num_groups, 1) for tile_idx to be correct.
    let tiles = batch.div_ceil(rpt);
    invoke.set_grid((tiles * half_gs) as i32, num_groups as i32, 1)?;
    invoke.set_thread_group(half_gs as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    tracing::trace!(
        batch,
        hidden,
        krot,
        group_size,
        rows_per_tile = rpt,
        "paro rotate dispatched"
    );
    if outputs.is_empty() {
        return Err(Error::Mlx("paro_rotate_gpu: expected 1 output".to_owned()));
    }
    Ok(outputs.remove(0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "paroquant_msl_tests.rs"]
mod tests;
