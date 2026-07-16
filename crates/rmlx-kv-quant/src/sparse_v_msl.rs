// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! Sparse-V fused MSL kernel.
//!
//! # What this is
//!
//! A custom Metal/MSL kernel that fuses affine-quantized V dequantization with
//! the softmax-probs weighted sum, skipping V dequant entirely for token
//! positions whose attention probability falls below a configurable threshold.
//!
//! The earlier cheap-path sparse-V (commit `e70901f`) zeroed probs below
//! threshold *before* calling `quantized_matmul`. That saves FLOPs on the
//! output side but does NOT reduce V-dequant bandwidth: MLX's opaque
//! `quantized_matmul` touches all V rows regardless of zero weights.
//!
//! This MSL kernel moves the skip **inside** the dequant loop, bypassing both
//! the dequant memory reads and the multiply-accumulate for negligible-weight
//! tokens. At
//! 32K context 90 %+ of attention weights are < 1e-6 (TheTom measurement,
//! N71 §2.6), so the kernel skips >90 % of V bandwidth — matching TheTom's
//! measured +22.8 % decode speedup at 32K on M5 Max.
//!
//! # Kernel design
//!
//! **Grid**: `(B × n_kv_heads × n_repeats × head_dim, 1, 1)` — one thread per
//! output element.
//!
//! **Each thread**:
//!   1. Derives its `(b, kv_head, repeat_idx, dim_idx)` indices from its flat
//!      `thread_position_in_grid.x`.
//!   2. Loops over all `T_seq` context tokens.
//!   3. Reads the softmax prob for that token. If `prob < EPS`, skips the
//!      remaining steps for that token (no memory read on V codes/scales/biases).
//!   4. Extracts the quantized code for `(kv_head, token, dim_idx)` from the
//!      packed `codes` buffer.
//!   5. Dequantizes: `val = scale * code_float + bias`.
//! 6. Accumulates: `acc += prob * val`.
//! 7. Writes `acc` to the output buffer.
//!
//! # Quantization support
//!
//! General affine quantization: any `(bits, group_size)` combination supported
//! by MLX's `quantize(..., mode="affine")`. Bit widths 4 and 8 are validated
//! in tests; the kernel arithmetic generalises to any power-of-two bits.
//!
//! Affine dequantization formula (identical to MLX's `quantized_matmul` internal
//! path for `mode="affine"`):
//! - `raw = (codes_word >> shift) & mask` (unsigned `bits`-bit integer)
//! - `code_float = (float)raw - midpoint` where `midpoint = 2^(bits-1)`
//! - `val = scale * code_float + bias`
//!
//! This is NOT TurboQuant (which uses a nonlinear codebook). This kernel targets
//! the `MixedKvState` (MLX affine) path used by k8v4, k8v8.
//! PlanarQuant V uses its own MSL kernel (`planarquant_msl`) with Givens-rotation
//! non-affine scheme and is NOT served by this kernel.
//!
//! # Activation
//!
//! Always ON (hardcoded; `RMLX_SPARSE_V_KERNEL` env var removed in PASS 3).
//! The kernel is only invoked for L=1 (single decode step); prefill always
//! uses `quantized_matmul`.
//!
//! Default-ON rationale: VG.2 PASS (2/2 greedy identity), 0/20 cells regressed,
//! 6/20 cells beaten.
//!
//! # Reference
//!
//! TheTom `experimental_decode_speed_tests` branch — `TURBO_SPARSE_V` flag in
//! `ggml-metal.metal:8297-8302`. N71 §2.6 documents +22.8 % at 32K, PPL
//! unchanged, NIAH 9/9. rMLX adaptation: uses `MetalKernel` / `MetalKernelInvoke`
//! (mlx-c-based dispatch) instead of GGML's C++ dispatch.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Kernel gates (hardcoded defaults, PASS 3 cleanup) ────────────────────────

/// Returns `true` when the fused sparse-V MSL kernel is enabled.
///
/// Hardcoded ON (`RMLX_SPARSE_V_KERNEL` env var removed in PASS 3).
/// Default-ON rationale: VG.2 PASS, 0/20 cells regressed,
/// 6/20 cells beaten (Gemma26B k8v8 +6-9%, Gemma26B planar +5-11%).
#[inline]
pub fn sparse_v_kernel_enabled() -> bool {
    true
}

/// Threshold below which a softmax probability is treated as negligible and
/// its V-row dequant is skipped. Shared with the cheap-path in `mixed_quant::sdpa`.
///
/// Hardcoded `1e-6` (`RMLX_SPARSE_V_THRESHOLD` env var removed in PASS 3).
/// Matches TheTom `experimental_decode_speed_tests` `TURBO_SPARSE_V` default.
#[inline]
fn kernel_eps() -> f32 {
    1e-6_f32
}

// ── MSL kernel source ─────────────────────────────────────────────────────────

/// MSL header: embeds the EPS threshold as a compile-time constant.
/// The actual value is baked in at kernel-registration time via the header string.
fn build_kernel_header(eps: f32) -> String {
    // Embed EPS as a hex f32 bit pattern for bit-exact reproducibility.
    let eps_bits = eps.to_bits();
    format!(
        "// sparse-V threshold: EPS = {eps:e} (bits 0x{eps_bits:08X}).\n\
         constant float SPARSE_V_EPS = as_type<float>(0x{eps_bits:08X}u);\n"
    )
}

/// MSL body for `rmlx_sparse_v_weighted_sum`.
///
/// # Input buffer layout (order must match `input_names` in [`kernel()`])
///
/// 0. `probs`: f32 `[B * n_kv_heads * n_repeats * T_seq]`
///
/// 1. `v_codes`: u32 `[B * n_kv_heads * T_seq * codes_d]`
///
/// 2. `v_scales`: f32 `[B * n_kv_heads * T_seq * scales_d]`
/// 3. `v_biases`: f32 `[B * n_kv_heads * T_seq * scales_d]`
///
/// 4. `params`: u32 `[8]` — {B, n_kv_heads, n_repeats, T_seq, head_dim,
///    codes_per_u32, threshold_u32, bits}.
///
/// # Output
///
/// `out`: f32 `[B * n_kv_heads * n_repeats * head_dim]`
const KERNEL_SOURCE: &str = include_str!("metal/sparse_v.metal");

// ── Kernel singleton ──────────────────────────────────────────────────────────

static KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

fn kernel() -> Result<&'static MetalKernel> {
    KERNEL
        .get_or_init(|| {
            let header = build_kernel_header(kernel_eps());
            MetalKernel::new(
                "rmlx_sparse_v_weighted_sum",
                &header,
                KERNEL_SOURCE,
                &["probs", "v_codes", "v_scales", "v_biases", "params"],
                &["out"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("sparse_v_weighted_sum kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Fused sparse-V weighted sum: `out = sum_t(probs[t] * dequant(V[t]))`
/// skipping V dequant when `probs[t] < EPS`.
///
/// This replaces `quantized_matmul(probs, v_codes, v_scales, v_biases, ...)`
/// for the L=1 (single decode step) case, eliminating V bandwidth for
/// negligible-weight token positions.
///
/// # Arguments
///
/// - `probs`: f32 array, shape `[B, n_kv_heads, n_repeats, 1, T_seq]`
///   or `[B, n_kv_heads, 1, T_seq]` when n_repeats == 1.
/// - `v_codes`: u32 array, shape `[B, n_kv_heads, T_seq, head_dim / el_per_int]`
/// - `v_scales`: f32 array, shape `[B, n_kv_heads, T_seq, head_dim / group_size]`
/// - `v_biases`: f32 array, shape `[B, n_kv_heads, T_seq, head_dim / group_size]`
/// - `b`, `n_kv_heads`, `n_repeats`, `t_seq`, `head_dim`, `group_size`, `v_bits`:
///   dimension and quantization parameters.
/// - `out_dtype`: desired output dtype.
/// - `device`: MLX device.
///
/// # Returns
///
/// f32 array of shape `[B, n_kv_heads, n_repeats, 1, head_dim]`.
///
/// # Errors
///
/// Returns `Error::Quant` if `v_bits` is not 4 or 8.
/// Returns `Error::Mlx` if kernel compilation or dispatch fails.
#[allow(clippy::too_many_arguments)]
pub fn sparse_v_weighted_sum(
    probs: &Array,
    v_codes: &Array,
    v_scales: &Array,
    v_biases: &Array,
    b: i32,
    n_kv_heads: i32,
    n_repeats: i32,
    t_seq: i32,
    head_dim: i32,
    group_size: i32,
    v_bits: i32,
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    // Validate bits.
    if v_bits != 4 && v_bits != 8 {
        return Err(Error::Quant(format!(
            "sparse_v_weighted_sum: v_bits={v_bits} not supported (only 4 or 8)"
        )));
    }

    let el_per_int = 32 / v_bits;

    // Validate divisibility.
    if head_dim % el_per_int != 0 {
        return Err(Error::Quant(format!(
            "sparse_v_weighted_sum: head_dim={head_dim} not divisible by el_per_int={el_per_int}"
        )));
    }
    if head_dim % group_size != 0 {
        return Err(Error::Quant(format!(
            "sparse_v_weighted_sum: head_dim={head_dim} not divisible by group_size={group_size}"
        )));
    }

    // Flatten probs to 1-D f32.
    let probs_total = b * n_kv_heads * n_repeats * t_seq;
    let probs_flat = {
        let p = probs.reshape(&[probs_total], device)?;
        if p.dtype() == Dtype::F32 {
            p
        } else {
            p.astype(Dtype::F32, device)?
        }
    };

    // Flatten v_codes to 1-D u32.
    let codes_total = b * n_kv_heads * t_seq * (head_dim / el_per_int);
    let v_codes_flat = v_codes.reshape(&[codes_total], device)?;

    // Flatten v_scales / v_biases to 1-D f32.
    let scales_total = b * n_kv_heads * t_seq * (head_dim / group_size);
    let v_scales_flat = {
        let s = v_scales.reshape(&[scales_total], device)?;
        if s.dtype() == Dtype::F32 {
            s
        } else {
            s.astype(Dtype::F32, device)?
        }
    };
    let v_biases_flat = {
        let s = v_biases.reshape(&[scales_total], device)?;
        if s.dtype() == Dtype::F32 {
            s
        } else {
            s.astype(Dtype::F32, device)?
        }
    };

    // Build params array: 8 × u32.
    let params_data: [u32; 8] = [
        b as u32,
        n_kv_heads as u32,
        n_repeats as u32,
        t_seq as u32,
        head_dim as u32,
        group_size as u32,
        v_bits as u32,
        el_per_int as u32,
    ];
    let params_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(params_data.as_ptr().cast::<u8>(), 8 * 4) };
    let params_arr = Array::from_bytes(params_bytes, &[8], Dtype::U32)
        .map_err(|e| Error::Mlx(format!("sparse_v_weighted_sum: params array: {e}")))?;

    let kern = kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&probs_flat)?;
    invoke.add_input(&v_codes_flat)?;
    invoke.add_input(&v_scales_flat)?;
    invoke.add_input(&v_biases_flat)?;
    invoke.add_input(&params_arr)?;

    // Output: flat [B * n_kv_heads * n_repeats * head_dim].
    let out_total = b * n_kv_heads * n_repeats * head_dim;
    invoke.add_output_shape(&[out_total], Dtype::F32)?;

    // Grid: one thread per output element. Threadgroup=1 (simple, coalesced via thread-per-dim).
    invoke.set_grid(out_total, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kern.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "sparse_v_weighted_sum: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);

    // Reshape to [B, n_kv_heads, n_repeats, 1, head_dim] (L=1 decode).
    let out = out_flat.reshape(&[b, n_kv_heads, n_repeats, 1, head_dim], device)?;

    // Cast to requested dtype if needed.
    if out_dtype == Dtype::F32 {
        Ok(out)
    } else {
        out.astype(out_dtype, device)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "sparse_v_msl_tests.rs"]
mod tests;
