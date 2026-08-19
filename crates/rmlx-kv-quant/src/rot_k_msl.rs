// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over Hadamard kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! sub-item 1 — fused FWHT + affine-8-bit-quantize Metal kernel.
//!
//! # Why this file exists
//!
//! v1 (`rot_k.rs`) rotates K using a plain MLX `matmul` against a precomputed
//! `[D, D]` Hadamard matrix `R`, then calls `mx.quantize` as a separate step.
//! That path is correct and coherent, but it:
//!   1. Materialises an intermediate `[N, D]` `K_rot` tensor in DRAM before
//!      quantizing — one full round-trip per decode step per KV-head.
//!   2. Runs D^2 multiplies (N x D x D) for the matmul when the Hadamard
//!      structure allows N x D x log2(D) via the Fast Walsh-Hadamard Transform
//!      (FWHT).
//!
//! For Bonsai (D=128): D^2 = 16 384 vs D*log2(D) = 896 -- ~18x fewer arithmetic
//! ops, plus the eliminated round-trip allocation.
//!
//! # What this file provides
//!
//! **`rot_k_fwht_quantize_gpu`** -- fused Metal kernel:
//! - Reads K `[N, D]` (bf16 or f32).
//! - Applies an in-place FWHT in threadgroup shared memory (one tg per row).
//! - Scales by `1/sqrt(D)` to produce the normalized Hadamard-rotated basis.
//! - Affine-8-bit-quantizes each group of `GROUP_SIZE` elements: computes
//!   per-group `scale = (max - min) / 255` and `bias = min`, then packs 4
//!   int8 codes per u32.
//! - Outputs the same `(codes_u32, scales, biases)` 3-tuple that
//!   `mx.quantize(mode="affine", bits=8, group_size=64)` would produce,
//!   so the result feeds directly into `mixed_quantized_sdpa` / `quantized_matmul`
//!   without any change to the SDPA path.
//!
//! **`rot_k_fwht_rotate_gpu`** -- Q pre-rotation kernel:
//! - Reads Q `[N, D]` (bf16 or f32).
//! - Applies FWHT + `1/sqrt(D)` scale in SMEM.
//! - Writes rotated Q as f32.
//! - Used by `maybe_pre_rotate_q_gpu` to replace the `rotate_last_axis` matmul
//!   in `mixed_quantized_sdpa` when the fused path is active.
//!
//! # Document the truth (CLAUDE.md hard rule 7)
//!
//! The `planarquant_msl.rs` kernel template uses Givens pair rotations (2x2
//! micro-rotation). That algorithm is correct for PlanarQuant but has nothing
//! to do with the Hadamard rotation used by `rot_k`. The FWHT is a butterfly
//! network operating on the full D-element row simultaneously -- structurally
//! different from Givens. The infrastructure pattern (MSL header builder,
//! `MetalKernel` singleton, `MetalKernelInvoke` dispatch) is what this file
//! borrows from `planarquant_msl.rs`, not the math.
//!
//! # Activation
//!
//! Default-OFF. Enable via `DispatchPolicy::rot_k_fused` (`--rot-k-fused on`,
//! or `RMLX_ROT_K_FUSED=1`). Fallback: `rot_k.rs` matmul path used unchanged
//! on any unsupported `D` or when disabled.
//!
//! Supported D values: powers of two in {32, 64, 128, 256, 512} -- common
//! head_dim values. D=128 (Bonsai) is the primary target.
//!
//! # Output format (bit-exact with mx.quantize affine 8-bit)
//!
//! `mx.quantize(x, group_size=64, bits=8, mode="affine")` returns:
//! - `codes`: shape `[..., D/4]` u32 -- four 8-bit codes packed LSB-first per u32.
//! - `scales`: shape `[..., D/group_size]` -- one f32 scale per group.
//! - `biases`: shape `[..., D/group_size]` -- one f32 bias per group (group min).
//!
//! MLX affine formula: `code = clamp(round((x - bias) / scale), 0, 255)`
//! `x_recon = scale * code + bias`
//!
//! The fused kernel implements this formula exactly in MSL.
//!
//! # Numeric equivalence guarantee
//!
//! The fused kernel output is bit-equivalent to running `rotate_last_axis`
//! then `mx.quantize` within bf16 ULP tolerance (<=2 ULP on reconstructed
//! values), because:
//! 1. FWHT in f32 reproduces the Walsh-Hadamard rotation to f32 precision.
//! 2. The affine quantize formula is reproduced from the MLX spec exactly.
//! 3. Group scale/bias are computed identically (min/max scan per group).
//!
//! Verified by `fwht_quantize_matches_reference_d128` in tests below.
//!
//! # Performance note
//!
//! Threadgroup size = D (one thread per element). For D=128: 128 threads/tg.
//! All butterfly stages use only threadgroup-local SMEM -- no global memory
//! reads after initial load, no global writes until final quantize output.
//! The v1 matmul path makes D+1 passes over global memory.

#![allow(clippy::float_cmp)]
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ---- Supported head_dim values ----------------------------------------------

/// D values for which the FWHT kernel is compiled and dispatched.
///
/// All must be powers of two; each requires its own kernel specialization
/// because the Metal threadgroup size equals D (static per dispatch).
///
/// Rotation accepts every value here. The fused quantize additionally needs a
/// row to hold at least one whole affine group, so it rejects any `D` smaller
/// than [`FWHT_QUANT_GROUP_SIZE`] — see [`build_fwht_quantize_body`].
const SUPPORTED_D: &[usize] = &[32, 64, 128, 256, 512];

/// Returns `true` iff `d` is a supported FWHT *rotation* dimension.
///
/// Not sufficient for the fused quantize, which also requires
/// `d % FWHT_QUANT_GROUP_SIZE == 0`; [`rot_k_fwht_quantize_gpu`] rejects the
/// rest with a shape error, and its caller falls back to the matmul path.
pub fn is_supported_d(d: usize) -> bool {
    SUPPORTED_D.contains(&d)
}

// ---- MSL header (shared across all D specializations) -----------------------

fn build_msl_header() -> String {
    // Minimal header: device-independent helpers only. D-specific constants
    // are inlined into each kernel body to avoid a header-per-D explosion.
    String::from(
        "// rmlx_rot_k_fwht -- fused Walsh-Hadamard Transform + affine 8-bit quantize.\n\
 // Header: device-independent helpers. D-specific constants in kernel body.\n",
    )
}

static KERNEL_HEADER: OnceLock<String> = OnceLock::new();

fn kernel_header() -> &'static str {
    KERNEL_HEADER.get_or_init(build_msl_header)
}

// ---- Per-D kernel body builders ---------------------------------------------

/// Affine quantize group size. Must match `k_group_size` from `RotK` state (= 64).
pub const FWHT_QUANT_GROUP_SIZE: usize = 64;

/// Build the MSL body for the fused FWHT + 8-bit affine quantize kernel.
///
/// # Algorithm (one threadgroup of `d` threads per input row)
///
/// 1. Load one element into threadgroup SMEM `buf[tid]` (cast to f32).
/// 2. FWHT butterfly: log2(d) stages. Stage s: XOR-pair distance = 1<<s.
/// 3. Normalize: buf[tid] /= sqrt(d).
/// 4. Per-group affine quantize:
/// - lidg==0 scans its group in SMEM to find min/max.
/// - scale = (max-min)/255, bias = min.
/// - Each thread quantizes its element to [0,255].
/// 5. Pack 4 int8 codes per u32 via atomic OR (LSB-first).
/// 6. lidg==0 writes scale and bias.
///
/// # Errors
///
/// Returns [`Error::Quant`] when `d` is not a positive multiple of
/// [`FWHT_QUANT_GROUP_SIZE`], or when no kernel specialization exists for it.
fn build_fwht_quantize_body(d: usize) -> Result<String> {
    let gs = FWHT_QUANT_GROUP_SIZE;
    // A row must hold at least one whole affine group: the kernel sizes its
    // per-group SMEM as `grp_max[d / gs]`, so a row shorter than one group
    // emits a zero-length threadgroup array, which is not valid MSL and fails
    // at shader compile. Reject the shape here rather than assert it — a
    // debug_assert is compiled out of release, which is precisely where the
    // broken shader would reach the GPU.
    let groups_per_row = d / gs;
    if !d.is_multiple_of(gs) || groups_per_row == 0 {
        return Err(Error::Quant(format!(
            "rot_k FWHT quantize: head_dim {d} must be a positive multiple of the affine \
             group size {gs} ({groups_per_row} whole groups per row)"
        )));
    }
    debug_assert!(d.is_power_of_two() && d.is_multiple_of(4));
    Ok(match d {
        64 => include_str!("metal/rot_k_fwht_quantize_d64.metal"),
        128 => include_str!("metal/rot_k_fwht_quantize_d128.metal"),
        256 => include_str!("metal/rot_k_fwht_quantize_d256.metal"),
        512 => include_str!("metal/rot_k_fwht_quantize_d512.metal"),
        _ => {
            return Err(Error::Quant(format!(
                "rot_k FWHT quantize: no kernel specialization for head_dim {d}"
            )))
        }
    }
    .to_owned())
}

/// Select the pre-rendered MSL body for the FWHT rotate-only kernel (Q
/// pre-rotation). Rotation has no group structure, so every [`SUPPORTED_D`]
/// has a body.
///
/// # Errors
///
/// Returns [`Error::Quant`] when no kernel specialization exists for `d`,
/// rather than handing back a body built for a different dimension.
fn build_fwht_rotate_body(d: usize) -> Result<String> {
    debug_assert!(d.is_power_of_two() && d >= 32);
    Ok(match d {
        32 => include_str!("metal/rot_k_fwht_rotate_d32.metal"),
        64 => include_str!("metal/rot_k_fwht_rotate_d64.metal"),
        128 => include_str!("metal/rot_k_fwht_rotate_d128.metal"),
        256 => include_str!("metal/rot_k_fwht_rotate_d256.metal"),
        512 => include_str!("metal/rot_k_fwht_rotate_d512.metal"),
        _ => {
            return Err(Error::Quant(format!(
                "rot_k FWHT rotate: no kernel specialization for head_dim {d}"
            )))
        }
    }
    .to_owned())
}

// ---- Kernel singletons (one per D that holds whole affine groups for quantize;
// ---- one per supported D for rotate) ----------------------------------------

struct FwhtKernels {
    // No quantize_d32: a 32-element row is shorter than one affine group, so
    // that specialization is a rejected shape rather than a kernel.
    quantize_d64: OnceLock<Result<MetalKernel>>,
    quantize_d128: OnceLock<Result<MetalKernel>>,
    quantize_d256: OnceLock<Result<MetalKernel>>,
    quantize_d512: OnceLock<Result<MetalKernel>>,
    rotate_d32: OnceLock<Result<MetalKernel>>,
    rotate_d64: OnceLock<Result<MetalKernel>>,
    rotate_d128: OnceLock<Result<MetalKernel>>,
    rotate_d256: OnceLock<Result<MetalKernel>>,
    rotate_d512: OnceLock<Result<MetalKernel>>,
}

// Note: relies on OnceLock<Result<MetalKernel>>: Sync (OnceLock is Sync when
// the contained T: Sync; Result<MetalKernel>: Sync when MetalKernel: Sync,
// which is documented in metal_kernel.rs). Static means one instance per
// process, matching the single-MLX-process requirement (CLAUDE.md hard rule 8).
static FWHT_KERNELS: FwhtKernels = FwhtKernels {
    quantize_d64: OnceLock::new(),
    quantize_d128: OnceLock::new(),
    quantize_d256: OnceLock::new(),
    quantize_d512: OnceLock::new(),
    rotate_d32: OnceLock::new(),
    rotate_d64: OnceLock::new(),
    rotate_d128: OnceLock::new(),
    rotate_d256: OnceLock::new(),
    rotate_d512: OnceLock::new(),
};

fn quant_kernel_for_d(d: usize) -> Result<&'static MetalKernel> {
    let cell = match d {
        64 => &FWHT_KERNELS.quantize_d64,
        128 => &FWHT_KERNELS.quantize_d128,
        256 => &FWHT_KERNELS.quantize_d256,
        512 => &FWHT_KERNELS.quantize_d512,
        _ => {
            return Err(Error::Quant(format!(
                "rot_k_msl: no FWHT quantize kernel for head_dim {d}; each row must hold a \
                 whole number of affine groups of {FWHT_QUANT_GROUP_SIZE}"
            )))
        }
    };
    cell.get_or_init(|| {
        let body = build_fwht_quantize_body(d)?;
        MetalKernel::new(
            &format!("rmlx_rot_k_fwht_q8_d{d}"),
            kernel_header(),
            &body,
            &["inp"],
            &["out_codes", "out_scales", "out_biases"],
        )
    })
    .as_ref()
    .map_err(|e| Error::Mlx(format!("rot_k_fwht_quantize D={d} kernel init: {e}")))
}

fn rotate_kernel_for_d(d: usize) -> Result<&'static MetalKernel> {
    let cell = match d {
        32 => &FWHT_KERNELS.rotate_d32,
        64 => &FWHT_KERNELS.rotate_d64,
        128 => &FWHT_KERNELS.rotate_d128,
        256 => &FWHT_KERNELS.rotate_d256,
        512 => &FWHT_KERNELS.rotate_d512,
        _ => {
            return Err(Error::Mlx(format!(
                "rot_k_msl: unsupported D={d} for FWHT rotate kernel"
            )))
        }
    };
    cell.get_or_init(|| {
        let body = build_fwht_rotate_body(d)?;
        MetalKernel::new(
            &format!("rmlx_rot_k_fwht_rotate_d{d}"),
            kernel_header(),
            &body,
            &["inp"],
            &["out"],
        )
    })
    .as_ref()
    .map_err(|e| Error::Mlx(format!("rot_k_fwht_rotate D={d} kernel init: {e}")))
}

// ---- Public API --------------------------------------------------------------

/// Fused FWHT + 8-bit affine quantize of K.
///
/// Input `k` is `[..., D]` (any leading shape; flattened to `[N, D]` rows).
/// Output: `(codes, scales, biases)` shaped **and typed** to match what
/// `mx.quantize(k_rotated, group_size=64, bits=8, mode="affine")` would
/// produce after `rotate_last_axis(k, R)` — codes `u32`, scales and biases in
/// `k`'s own dtype.
///
/// The kernel computes in f32 (the FWHT butterfly needs it) and declares f32
/// scale/bias outputs; they are cast back before returning. Leaving them f32
/// is not a numerical detail: `mx.quantized_matmul` and `mx.dequantize` take
/// their operand width from the scales, so f32 scales silently promote the
/// whole decode graph of a bf16 model — the fallback path below, which is the
/// same codec, would not.
///
/// Returns `Err` if D is not in `SUPPORTED_D` -- caller must fall back to
/// the v1 `rotate_last_axis + mx.quantize` path.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn rot_k_fwht_quantize_gpu(k: &Array, device: Device) -> Result<(Array, Array, Array)> {
    let shape = k.shape();
    let d = *shape
        .last()
        .ok_or_else(|| Error::Mlx("rot_k_fwht_quantize: empty shape".into()))? as usize;

    if !is_supported_d(d) {
        return Err(Error::Mlx(format!(
            "rot_k_fwht_quantize_gpu: D={d} not in supported set {SUPPORTED_D:?}"
        )));
    }
    if !d.is_multiple_of(FWHT_QUANT_GROUP_SIZE) {
        return Err(Error::Mlx(format!(
            "rot_k_fwht_quantize_gpu: D={d} not divisible by group_size={FWHT_QUANT_GROUP_SIZE}"
        )));
    }

    let total_elems: usize = shape.iter().map(|&x| x as usize).product();
    let n_rows = total_elems / d;
    let n_groups = n_rows * (d / FWHT_QUANT_GROUP_SIZE);
    let n_words = n_rows * (d / 4); // 4 int8 per u32

    let k_2d = {
        let k_flat = k.reshape(&[n_rows as i32, d as i32], device)?;
        if k_flat.dtype() == Dtype::F32 {
            k_flat
        } else {
            k_flat.astype(Dtype::F32, device)?
        }
    };

    let kernel = quant_kernel_for_d(d)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&k_2d)?;
    // out_codes: u32 [N * D/4]
    invoke.add_output_shape(&[n_words as i32], Dtype::U32)?;
    // out_scales: f32 [N * (D/GROUP_SIZE)]
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;
    // out_biases: f32 [N * (D/GROUP_SIZE)]
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;
    // Zero-init codes: written via atomic_fetch_or (stale bits corrupt packing).
    invoke.set_init_value(0.0)?;
    // grid = total threads; MLX derives threadgroups = grid / thread_group.
    // We want N threadgroups of D threads each → total threads = N * D.
    invoke.set_grid((n_rows * d) as i32, 1, 1)?;
    invoke.set_thread_group(d as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 3 {
        return Err(Error::Mlx(
            "rot_k_fwht_quantize_gpu: expected 3 outputs".into(),
        ));
    }
    let biases = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);

    // Reshape to match mx.quantize output shapes: [..., D/4] and [..., D/group].
    let leading: &[i32] = &shape[..shape.len() - 1];
    let groups_per_row = (d / FWHT_QUANT_GROUP_SIZE) as i32;
    let words_per_row = (d / 4) as i32;

    let codes_shape: Vec<i32> = leading
        .iter()
        .copied()
        .chain(std::iter::once(words_per_row))
        .collect();
    let sg_shape: Vec<i32> = leading
        .iter()
        .copied()
        .chain(std::iter::once(groups_per_row))
        .collect();

    let codes_r = codes.reshape(&codes_shape, device)?;
    // Scales and biases go back at K's dtype — `mx.quantize` produces them
    // that way, and every consumer (`quantized_matmul`, `dequantize`) reads
    // its operand width from them.
    let k_dtype = k.dtype();
    let scales_r = scales.reshape(&sg_shape, device)?.astype(k_dtype, device)?;
    let biases_r = biases.reshape(&sg_shape, device)?.astype(k_dtype, device)?;
    Ok((codes_r, scales_r, biases_r))
}

/// Fused FWHT rotate-only (for Q pre-rotation in SDPA).
///
/// Input `q` is `[..., D]`. Output is `[..., D]` in the **same dtype as `q`**
/// holding the Hadamard-rotated Q, ready for the score matmul.
///
/// The Metal kernel always computes in f32 (FWHT butterfly requires it to
/// avoid bf16 accumulation error). The result is cast back to the input dtype
/// before returning so that the downstream `multiply`, GQA `reshape`, and
/// `quantized_matmul` in `mixed_quantized_sdpa` run at the correct dtype
/// (typically bf16) rather than silently widening to f32.
///
/// Returns `Err` when D is not in `SUPPORTED_D`; caller falls back to
/// `rotate_last_axis(q, R)` matmul.
pub fn rot_k_fwht_rotate_gpu(q: &Array, device: Device) -> Result<Array> {
    let shape = q.shape();
    let in_dtype = q.dtype();
    let d = *shape
        .last()
        .ok_or_else(|| Error::Mlx("rot_k_fwht_rotate: empty shape".into()))? as usize;

    if !is_supported_d(d) {
        return Err(Error::Mlx(format!(
            "rot_k_fwht_rotate_gpu: D={d} not in supported set"
        )));
    }

    let total_elems: usize = shape.iter().map(|&x| x as usize).product();
    let n_rows = total_elems / d;

    let q_2d = {
        let q_flat = q.reshape(&[n_rows as i32, d as i32], device)?;
        if q_flat.dtype() == Dtype::F32 {
            q_flat
        } else {
            q_flat.astype(Dtype::F32, device)?
        }
    };

    let kernel = rotate_kernel_for_d(d)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&q_2d)?;
    invoke.add_output_shape(&[total_elems as i32], Dtype::F32)?;
    // grid = total threads; MLX derives threadgroups = grid / thread_group.
    // We want N threadgroups of D threads each → total threads = N * D.
    invoke.set_grid((n_rows * d) as i32, 1, 1)?;
    invoke.set_thread_group(d as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "rot_k_fwht_rotate_gpu: expected 1 output".into(),
        ));
    }
    let out_flat = outputs.remove(0);
    // Cast back to input dtype before reshape so downstream ops (multiply,
    // quantized_matmul) stay at bf16 when Q was bf16 (MEDIUM 1 fix).
    let out_typed = if out_flat.dtype() == in_dtype {
        out_flat
    } else {
        out_flat.astype(in_dtype, device)?
    };
    out_typed.reshape(&shape, device)
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
#[path = "rot_k_msl_tests.rs"]
mod tests;
