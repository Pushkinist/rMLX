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
//! # Template parameters (baked at kernel build time)
//!
//! - `ROWS_PER_TILE`: rows processed per threadgroup tile (4 for batch > 1, 1 for decode).
//! - `MAX_KROT`: maximum krot supported (must be >= actual krot; 16 is safe upper bound).
//! - `MAX_GROUP_SIZE`: maximum group_size (must be >= actual group_size; 256 safe upper bound).
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
/// Template parameters substituted at build time:
/// `{RPT}`, `{MK}`, `{MGS}`.
///
/// Ported from `z-lab/paroquant/paroquant/kernels/metal/rotation.metal`.
/// Changes from upstream:
/// - `cos_theta`/`sin_theta` are separate F16 inputs (pre-computed on load from
///   the raw `theta` F16 array) matching the upstream Python `_cache_rotation`
///   pattern. This keeps the kernel identical to upstream and avoids per-dispatch
///   transcendental math.
/// - `params` is a flat I32 array `[batch, hidden, krot, group_size]` (same as
///   upstream).
/// - Template params are Rust `const` substituted strings (not Python .format).
const KERNEL_SOURCE: &str = r"
    constexpr int ROWS_PER_TILE  = {RPT};
    constexpr int MAX_KROT       = {MK};
    constexpr int MAX_GROUP_SIZE = {MGS};

    const int batch_size  = params[0];
    const int hidden_size = params[1];
    const int krot        = params[2];
    const int group_size  = params[3];

    const int half_gs     = group_size / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int tid       = thread_index_in_threadgroup;

    if (tid >= half_gs) return;

    float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
    int   pair_vals[MAX_KROT];

    for (int k = 0; k < krot; k++) {
        int idx = k * half_hidden + group_idx * half_gs + tid;
        cos_vals[k]  = float(cos_theta[idx]);
        sin_vals[k]  = float(sin_theta[idx]);
        pair_vals[k] = int(packed_pairs[idx]);
    }

    threadgroup float tile[MAX_GROUP_SIZE * ROWS_PER_TILE];

    const int ch_lo = group_idx * group_size + tid;
    const int ch_hi = ch_lo + half_gs;
    float scale_lo = float(channel_scales[ch_lo]);
    float scale_hi = float(channel_scales[ch_hi]);

    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            tile[tid * ROWS_PER_TILE + r]             = float(x[row * hidden_size + ch_lo]) * scale_lo;
            tile[(tid + half_gs) * ROWS_PER_TILE + r] = float(x[row * hidden_size + ch_hi]) * scale_hi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < krot; k++) {
        int i_local = pair_vals[k] & 0xFFFF;
        int j_local = pair_vals[k] >> 16;
        float c = cos_vals[k], s = sin_vals[k];

        for (int m = 0; m < ROWS_PER_TILE; m++) {
            float a = tile[i_local * ROWS_PER_TILE + m];
            float b = tile[j_local * ROWS_PER_TILE + m];
            tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
            tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            out[row * hidden_size + ch_lo] = InT(tile[tid * ROWS_PER_TILE + r]);
            out[row * hidden_size + ch_hi] = InT(tile[(tid + half_gs) * ROWS_PER_TILE + r]);
        }
    }
";

// ── Template instantiation ────────────────────────────────────────────────────

/// MAX_KROT is 16 — safe upper bound for all known PARO checkpoints (Qwen3.5:
/// krot=8). If future checkpoints exceed 16, raise this constant.
const MAX_KROT: usize = 16;

/// MAX_GROUP_SIZE is 256 — safe upper bound. Qwen3.5 uses group_size=128.
const MAX_GROUP_SIZE: usize = 256;

/// Build the MSL kernel source by substituting template parameters.
fn build_kernel_source(rows_per_tile: usize) -> String {
    KERNEL_SOURCE
        .replace("{RPT}", &rows_per_tile.to_string())
        .replace("{MK}", &MAX_KROT.to_string())
        .replace("{MGS}", &MAX_GROUP_SIZE.to_string())
}

// ── Kernel singletons ─────────────────────────────────────────────────────────

/// ROWS_PER_TILE=1: used for single-row decode steps.
static KERNEL_RPT1: OnceLock<Result<MetalKernel>> = OnceLock::new();
/// ROWS_PER_TILE=4: used for batch or prefill steps (rows > 1).
static KERNEL_RPT4: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn build_kernel(rpt: usize) -> Result<MetalKernel> {
    let source = build_kernel_source(rpt);
    MetalKernel::new(
        &format!("rmlx_paro_rotate_r{rpt}"),
        "",
        &source,
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
}

fn kernel_rpt1() -> Result<&'static MetalKernel> {
    KERNEL_RPT1
        .get_or_init(|| build_kernel(1))
        .as_ref()
        .map_err(|e| Error::Mlx(format!("paro_rotate_r1 kernel init: {e}")))
}

fn kernel_rpt4() -> Result<&'static MetalKernel> {
    KERNEL_RPT4
        .get_or_init(|| build_kernel(4))
        .as_ref()
        .map_err(|e| Error::Mlx(format!("paro_rotate_r4 kernel init: {e}")))
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
    let kernel = if rpt == 1 {
        kernel_rpt1()?
    } else {
        kernel_rpt4()?
    };

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

    // MLX metal_kernel uses dispatchThreads (total threads, not threadgroup counts).
    // Python: grid = (ceil(batch/rpt) * half_gs, num_groups, 1), threadgroup=(half_gs,1,1).
    // → Metal launches ceil(grid.x / threadgroup.x) = ceil(batch/rpt) threadgroups in X.
    // → tile_idx = threadgroup_position_in_grid.x ranges 0..ceil(batch/rpt)-1.
    // We must pass grid=(tiles*half_gs, num_groups, 1) for tile_idx to be correct.
    let tiles = batch.div_ceil(rpt);
    invoke.set_grid((tiles * half_gs) as i32, num_groups as i32, 1)?;
    invoke.set_thread_group(half_gs as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx("paro_rotate_gpu: expected 1 output".to_owned()));
    }
    Ok(outputs.remove(0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "paroquant_msl_tests.rs"]
mod tests;
