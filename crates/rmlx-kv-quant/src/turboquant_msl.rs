// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! TurboQuant V4 MSL (Metal Shading Language) GPU kernels.
//!
//! # What this is
//!
//! GPU (Metal) versions of the CPU `turbo_quantize_v` / `turbo_dequantize`
//! functions from `turboquant.rs`. Kernels are siblings, not replacements —
//! public API names differ (`_gpu` suffix). CPU path stays as correctness
//! reference.
//!
//! # Codebook — Lloyd-Max, N(0,1)
//!
//! The codebook is **Lloyd-Max optimal centroids
//! for N(0,1)**, derived by `turboquant_plus/turboquant/codebook.py::
//! _lloyds_gaussian(sigma=1.0, n_iter=100)` and hardcoded as `as_type<float>(0x...)`.
//!
//! Prior to the Lloyd-Max codebook switch the codebase used N(0,1) *quantile*
//! centroids from the Python fork's dead `_compute_gaussian_codebook` function. GPU constants are
//! bit-exact with the CPU path in `turboquant.rs::CODEBOOK_*`.
//!
//! Derivation script: `scripts/gen_lloyd_codebook.py`.
//!
//! # Algorithm (matches CPU path)
//!
//! For each group of 32 f32 elements:
//! 1. `scale = max(|x_i|) / max_centroid`. If all zero, `scale = 0`.
//! 2. Normalize: `x_i / scale` (or 0 if scale is 0).
//! 3. Nearest centroid: count boundary crossings (same as CPU linear scan).
//! 4. Pack two 4-bit indices per byte, LSB-first (element 0 in bits [0:3]).
//!
//! Dequantize reverses: `out = CB[code] * scale`.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

use crate::turboquant::GROUP_SIZE;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Codebook constants ────────────────────────────────────────────────────────

// Lloyd-Max optimal N(0,1) codebook for 4-bit (16 entries).
//
// Source: turboquant.rs::CODEBOOK_4BIT — mirrored here for clarity.
// Derived by turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0).
//
// Not used from Rust — the same values are embedded directly in KERNEL_HEADER
// as MSL `constant float` declarations for the GPU. Kept here as the
// authoritative reference so any future change is made in one place first.
#[allow(dead_code)]
const CB: [f32; 16] = [
    -2.717_667,
    -2.052_138,
    -1.600_802_4,
    -1.239_959,
    -0.928_244_7,
    -0.645_875_33,
    -0.381_178_23,
    -0.126_046_94,
    0.126_046_94,
    0.381_178_23,
    0.645_875_33,
    0.928_244_7,
    1.239_959,
    1.600_802_4,
    2.052_138,
    2.717_667,
];

// Maximum codebook centroid (denominator when computing scale).
#[allow(dead_code)]
const CB_MAX: f32 = 2.717_667;

// ── MSL kernel sources ────────────────────────────────────────────────────────
//
// The `source` strings are the **body** of the MSL kernel function.
// MLX generates the surrounding function signature and buffer declarations:
// - inputs arrive as `device const <dtype>* <name> [[buffer(N)]]`
// - outputs arrive as `device <dtype>* <name> [[buffer(M)]]`
// - thread position built-ins are available without any declaration.
//
// The header string is MSL code inserted before the kernel body — used here
// to embed the codebook constants so they are visible inside the body.

/// MSL header shared by all kernels: embeds the 4-bit Lloyd-Max codebook
/// constants, their 15 decision boundaries, and the 256-entry
/// `half2` dual-LUT for fast byte-level dequantization.
///
/// # Bit-exact f32 values
///
/// Both `CB` and `BOUNDARIES` use `as_type<float>(0x...)` to embed the
/// **exact IEEE-754 f32 bit patterns** that the CPU Rust path produces.
/// This is necessary because decimal literals like `-1.0841f` round to a
/// different bit pattern than `(-1.2399590f + -0.9282447f) * 0.5f`, causing
/// quantization index differences on boundary-crossing values.
///
/// Bit patterns derived from `scripts/gen_lloyd_codebook.py` and verified
/// against `turboquant.rs::CODEBOOK_4BIT` via `struct.pack('!f', ...)`.
///
/// # 256-entry dual LUT
///
/// `CB_LUT[b] = half2(CB[b & 0xF], CB[b >> 4])` — decodes both nibbles of
/// a byte in one lookup, used by the word-level dequantize kernel. The `half2`
/// representation is bit-exact with the f32 CB values (half has enough precision
/// for the 7-significant-digit centroids).
const KERNEL_HEADER: &str = include_str!("metal/turboquant_header.metal");

/// MSL body for `rmlx_tq4_quantize`.
///
/// Grid: `(N_groups * 32, 1, 1)`. Threadgroup: `(32, 1, 1)`.
///
/// Pack format: uint32, 8 indices per word (4 bits each, LSB-first).
/// For a group of 32 elements: 4 uint32 words (indices 0..7, 8..15, 16..23, 24..31).
/// This matches the bit-layout of the CPU's `pack_index` / `unpack_index`
/// from `turboquant.rs`, but consolidated to whole 32-bit words for GPU safety.
///
/// Inputs:
/// - `inp` f32 `[N_groups * 32]` — flat input elements
///
/// Outputs:
/// - `codes` u32 `[N_groups * 4]` — 4-bit packed, 8 indices per uint32
/// - `scales` f32 `[N_groups]` — per-group max-abs scale
const QUANTIZE_SOURCE: &str = include_str!("metal/turboquant_quantize.metal");

/// MSL body for `rmlx_tq4_dequantize` — word-level dual-LUT path.
///
/// Grid: `(N_groups * 4, 1, 1)`. Threadgroup: `(4, 1, 1)`.
///
/// Each thread processes one uint32 word (8 nibbles = 8 elements) by reading
/// the word's 4 bytes and decoding each byte with a pair of CB lookups. This
/// halves thread count vs the original element-per-thread path and reduces
/// global-memory atomics by processing 8 outputs per thread.
///
/// The 256-entry "dual LUT" is implemented as two CB lookups per byte:
/// lo = CB[byte & 0xFu], hi = CB[byte >> 4u]
/// Metal's constant-memory CB array is already in L1 cache after the first
/// access — the nibble-pair lookup is equivalent to a 256-entry half2 LUT
/// without the 1 KB constant-memory cost.
///
/// Grid launch (in `turbo_dequantize_v4_gpu`): total = N_groups * 4 threads,
/// threadgroup = 4.
///
/// Inputs:
/// - `codes` u32 `[N_groups * 4]` — 4-bit packed, 8 indices per uint32
/// - `scales` f32 `[N_groups]`
///
/// Outputs:
/// - `out` OutT `[N_groups * 32]`
const DEQUANTIZE_SOURCE: &str = include_str!("metal/turboquant_dequantize.metal");

/// MSL body for `rmlx_tq4_quantize_codebook_buffer`.
///
/// Per-layer codebook-override variant of [`QUANTIZE_SOURCE`]. The kernel reads
/// the 16-entry f32 codebook from a kernel buffer (`cb`) instead of the
/// hardwired `CB[16]` / `BOUNDARIES[15]` constants in [`KERNEL_HEADER`].
///
/// # Why a separate kernel
///
/// The hardwired-codebook path stays as [`QUANTIZE_SOURCE`] (zero buffer-load
/// overhead on the default Lloyd-Max path; decision D1 during codebook-buffer
/// design). This
/// variant only runs when [`crate::storage::quant_v::QuantV::value_codebook`]
/// is `Some`.
///
/// # Algorithm — matches `turbo_quantize_v_with_codebook`
///
/// 1. Each threadgroup loads the 16-entry codebook into threadgroup memory once.
/// 2. Thread 0 computes `cb_max = max(|cb[i]|)` across 16 entries.
/// 3. Per-group: `scale = max(|x|) / cb_max` (0 if all-zero group).
/// 4. Per-thread: 15 boundary comparisons against midpoints
///    `(cb[i] + cb[i+1]) * 0.5f` — computed at runtime, not hardwired.
/// 5. Pack as in [`QUANTIZE_SOURCE`].
///
/// # Inputs
///
/// - `inp` f32 `[N_groups * 32]` — flat input elements.
/// - `cb` f32 `[16]` — per-layer codebook (strictly ascending; caller validates).
///
/// # Outputs
///
/// - `codes` u32 `[N_groups * 4]` — same layout as the hardwired path.
/// - `scales` f32 `[N_groups]`.
const QUANTIZE_CB_BUF_SOURCE: &str = include_str!("metal/turboquant_quantize_cb_buf.metal");

/// MSL body for `rmlx_tq4_dequantize_codebook_buffer`.
///
/// Codebook-buffer companion of [`DEQUANTIZE_SOURCE`]. The hardwired path
/// reads from `constant float CB[16]` in [`KERNEL_HEADER`]; this variant
/// reads centroids from a kernel buffer (`cb`) at runtime so per-layer
/// codebook overrides round-trip correctly on GPU.
///
/// # Grid / threadgroup
///
/// Same as [`DEQUANTIZE_SOURCE`]: one thread per uint32 word (8 elements),
/// threadgroup of 4 (= one full group of 32). Caching the 16-entry codebook
/// into threadgroup memory is cheap (16 floats × 4 bytes = 64 bytes) and
/// removes the per-byte global codebook load.
///
/// # Inputs
///
/// - `codes` u32 `[N_groups * 4]`.
/// - `scales` f32 `[N_groups]`.
/// - `cb` f32 `[16]`.
///
/// # Outputs
///
/// - `out` OutT `[N_groups * 32]`.
const DEQUANTIZE_CB_BUF_SOURCE: &str = include_str!("metal/turboquant_dequantize_cb_buf.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

// Kernels are compiled once per process and reused across calls.
// We use `std::sync::OnceLock` to avoid `unsafe` global mutable state.

use std::sync::OnceLock;

static QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static QUANT_CB_BUF_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static DEQUANT_CB_BUF_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn quant_kernel() -> Result<&'static MetalKernel> {
    QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq4_quantize",
                KERNEL_HEADER,
                QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tq4_quantize kernel init: {e}")))
}

fn dequant_kernel() -> Result<&'static MetalKernel> {
    DEQUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq4_dequantize",
                KERNEL_HEADER,
                DEQUANTIZE_SOURCE,
                &["codes", "scales"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tq4_dequantize kernel init: {e}")))
}

/// Codebook-buffer variant of [`quant_kernel`]. Separate kernel singleton so
/// the hardwired path stays compile-clean (no `cb` buffer arg).
fn quant_cb_buf_kernel() -> Result<&'static MetalKernel> {
    QUANT_CB_BUF_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq4_quantize_codebook_buffer",
                // No header: codebook arrives via `cb` buffer; KERNEL_HEADER constants unused.
                "",
                QUANTIZE_CB_BUF_SOURCE,
                &["inp", "cb"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| {
            rmlx_core::error::Error::Mlx(format!("tq4_quantize_codebook_buffer kernel init: {e}"))
        })
}

/// Codebook-buffer variant of [`dequant_kernel`].
fn dequant_cb_buf_kernel() -> Result<&'static MetalKernel> {
    DEQUANT_CB_BUF_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq4_dequantize_codebook_buffer",
                "",
                DEQUANTIZE_CB_BUF_SOURCE,
                &["codes", "scales", "cb"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| {
            rmlx_core::error::Error::Mlx(format!("tq4_dequantize_codebook_buffer kernel init: {e}"))
        })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU TurboQuant V4 quantize.
///
/// Quantize `x` (any shape, total elements must be a multiple of `GROUP_SIZE=32`)
/// using the Lloyd-Max N(0,1) codebook (see module docs).
///
/// # Returns
///
/// `(codes, scales)` where:
/// - `codes`: `u32` array of shape `[total_elems / 8]` — 8 4-bit indices
///   per uint32, packed LSB-first (element 0 in bits [0:3]).
/// - `scales`: `f32` array of shape `[total_elems / 32]` — one scale per group.
///
/// # Errors
///
/// Returns `Error::Mlx` if the kernel fails to compile or dispatch, or
/// `Error::Quant` if `total_elems` is not a multiple of `GROUP_SIZE`.
// f32-out-ok: `codes` is u32; the f32 `scales` are read back only by MSL
// kernels that declare them `device const float*` — `turbo_dequantize_v4_gpu`,
// the `turbo_k4` fused-QK kernel, and TurboFlash P1 as its V-side scale
// buffer. No MLX
// op would take its operand width from them, the way `quantized_matmul` and
// `dequantize` take theirs from an `mx.quantize` 3-tuple.
pub fn turbo_quantize_v4_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    // Validate total elements.
    let shape = x.shape();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_quantize_v4_gpu: total elements {total_elems} not a multiple \
             of GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    let n_groups = total_elems / GROUP_SIZE;

    // Flatten input to [total_elems] f32.
    let x_flat = if x.ndim() == 1 {
        x.try_clone()?
    } else {
        x.reshape(&[total_elems as i32], device)?
    };
    // Ensure f32.
    let x_f32 = if x_flat.dtype() == Dtype::F32 {
        x_flat
    } else {
        x_flat.astype(Dtype::F32, device)?
    };

    let kernel = quant_kernel()?;

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    // codes: [n_groups * 4] u32 (8 4-bit indices per uint32, 4 words per group of 32)
    invoke.add_output_shape(&[(n_groups * 4) as i32], Dtype::U32)?;
    // scales: [n_groups] f32
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;

    // Grid: (n_groups * 32, 1, 1) — total threads (MLX convention: grid = total threads).
    // With threadgroup=(32,1,1) this gives n_groups threadgroups of 32 threads each.
    invoke.set_grid((n_groups * GROUP_SIZE) as i32, 1, 1)?;
    // Threadgroup: (32, 1, 1) — 32 threads per group (one per element in a block).
    invoke.set_thread_group(GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_quantize_v4_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales))
}

/// GPU TurboQuant V4 quantize with a per-layer codebook override.
///
/// Same I/O contract as [`turbo_quantize_v4_gpu`], but dispatches the
/// `rmlx_tq4_quantize_codebook_buffer` MSL kernel and threads the override
/// codebook through as a kernel buffer argument. Output layout (`codes`,
/// `scales`) is bit-compatible with the hardwired path so the existing
/// `turbo_dequantize_v4_gpu` reconstructs correctly — provided the *same*
/// override codebook is used at dequant time (callers route through
/// [`crate::storage::quant_v::QuantV::dequantize_choice`] which dispatches
/// CPU when an override is present).
///
/// # Arguments
///
/// - `x` — input tensor; total elements must be a multiple of `GROUP_SIZE=32`.
/// - `codebook_gpu` — pre-uploaded f32 codebook GPU Array of shape `[16]`.
///   Caller is responsible for length validation (must equal `2^bits` — 16
///   for 4-bit) and strict-ascending order (mirrors CPU validation in
///   `turbo_quantize_v_with_codebook`). The caller-supplied Array is
///   cached on [`crate::storage::quant_v::QuantV::value_codebook_gpu`] so
///   it is uploaded at most once per layer.
///
/// # Errors
///
/// Returns `Error::Quant` if `total_elems` is not a multiple of `GROUP_SIZE`
/// or if the codebook GPU Array shape is not `[16]` (4-bit fixed). Returns
/// `Error::Mlx` if the kernel fails to compile or dispatch.
// f32-out-ok: same two buffers as `turbo_quantize_v4_gpu` (u32 `codes`, f32
// `scales`) with the Lloyd-Max codebook supplied as an input buffer rather
// than baked into the source; the readers are the same MSL kernels, which
// declare the scales `device const float*`. No MLX
// op would take its operand width from them, the way `quantized_matmul` and
// `dequantize` take theirs from an `mx.quantize` 3-tuple.
pub fn turbo_quantize_v4_codebook_buf_gpu(
    x: &Array,
    codebook_gpu: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    // 4-bit kernel: codebook fixed at 16 entries.
    const EXPECTED_CB_LEN: usize = 16;
    // Validate codebook shape.
    let cb_shape = codebook_gpu.shape();
    let cb_len: usize = cb_shape.iter().map(|&d| d as usize).product();
    if cb_len != EXPECTED_CB_LEN {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_quantize_v4_codebook_buf_gpu: codebook length {cb_len} != \
             2^bits={EXPECTED_CB_LEN} (4-bit hardcoded); caller bug"
        )));
    }
    if codebook_gpu.dtype() != Dtype::F32 {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_quantize_v4_codebook_buf_gpu: codebook dtype must be F32 \
             (got {:?})",
            codebook_gpu.dtype()
        )));
    }

    // Validate total elements (mirror `turbo_quantize_v4_gpu`).
    let shape = x.shape();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_quantize_v4_codebook_buf_gpu: total elements {total_elems} \
             not a multiple of GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    let n_groups = total_elems / GROUP_SIZE;

    // Flatten + f32-cast input (mirror `turbo_quantize_v4_gpu`).
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

    // Codebook arrives flat [16]. Defensive reshape — the storage cache
    // builds it that way, but a future caller may pass a higher-rank array.
    let cb_flat = if codebook_gpu.ndim() == 1 {
        codebook_gpu.try_clone()?
    } else {
        codebook_gpu.reshape(&[EXPECTED_CB_LEN as i32], device)?
    };

    let kernel = quant_cb_buf_kernel()?;

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    invoke.add_input(&cb_flat)?;
    invoke.add_output_shape(&[(n_groups * 4) as i32], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;

    invoke.set_grid((n_groups * GROUP_SIZE) as i32, 1, 1)?;
    invoke.set_thread_group(GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_quantize_v4_codebook_buf_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales))
}

/// GPU TurboQuant V4 dequantize.
///
/// Reconstruct f32 tensor from `(codes, scales)` produced by
/// [`turbo_quantize_v4_gpu`].
///
/// # Arguments
///
/// - `codes`: `u32` array `[n_groups * 4]` — 8 4-bit indices per uint32.
/// - `scales`: `f32` array `[n_groups]`.
/// - `original_shape`: shape to reshape the output into (product must equal
///   `n_groups * 32`).
///
/// # Returns
///
/// `f32` array of shape `original_shape`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn turbo_dequantize_v4_gpu(
    codes: &Array,
    scales: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    // n_groups = total uint32 words / 4 (4 words per group of 32 elements).
    let n_groups_codes = codes.shape().iter().map(|&d| d as usize).product::<usize>() / 4;
    let n_groups_scales = scales
        .shape()
        .iter()
        .map(|&d| d as usize)
        .product::<usize>();

    if n_groups_codes != n_groups_scales {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v4_gpu: n_groups from codes ({n_groups_codes}) != \
             from scales ({n_groups_scales})"
        )));
    }
    let n_groups = n_groups_scales;
    let total_elems = n_groups * GROUP_SIZE;

    // Verify original_shape.
    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v4_gpu: original_shape product {shape_product} != \
             expected {total_elems}"
        )));
    }

    // Flatten inputs.
    let codes_flat = codes.reshape(&[(n_groups * 4) as i32], device)?;
    let scales_flat = scales.reshape(&[n_groups as i32], device)?;

    // Restrict to dtypes that have a sensible static_cast<OutT>(float).
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

    // Word-level dual-LUT kernel — 4 threads per group (one per uint32 word),
    // each thread decodes 8 elements via 4 byte-level CB lookups.
    // Grid: (n_groups * 4, 1, 1) — one thread per word.
    // Threadgroup: (4, 1, 1) — 4 words per group of 32 elements.
    let n_words = n_groups * 4;
    invoke.set_grid(n_words as i32, 1, 1)?;
    invoke.set_thread_group(4, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_dequantize_v4_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);

    let out_flat = if kernel_out_dtype == out_dtype {
        out_flat
    } else {
        out_flat.astype(out_dtype, device)?
    };

    // Reshape to original_shape.
    if original_shape.len() == 1 && original_shape[0] == total_elems as i32 {
        return Ok(out_flat);
    }
    out_flat.reshape(original_shape, device)
}

/// GPU TurboQuant V4 dequantize with a per-layer codebook
/// override. Companion of [`turbo_quantize_v4_codebook_buf_gpu`] — pass the
/// same uploaded codebook buffer that was used at encode time.
///
/// Inputs / outputs match [`turbo_dequantize_v4_gpu`] except for the extra
/// `codebook_gpu` arg.
///
/// # Errors
///
/// Returns `Error::Quant` for the same shape / dtype invariants checked in
/// [`turbo_dequantize_v4_gpu`] plus a codebook-length / dtype guard.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn turbo_dequantize_v4_codebook_buf_gpu(
    codes: &Array,
    scales: &Array,
    codebook_gpu: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    // 4-bit kernel: codebook fixed at 16 entries.
    const EXPECTED_CB_LEN: usize = 16;
    // Codebook guard: length + dtype.
    let cb_len: usize = codebook_gpu.shape().iter().map(|&d| d as usize).product();
    if cb_len != EXPECTED_CB_LEN {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v4_codebook_buf_gpu: codebook length {cb_len} != \
             2^bits={EXPECTED_CB_LEN} (4-bit hardcoded); caller bug"
        )));
    }
    if codebook_gpu.dtype() != Dtype::F32 {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v4_codebook_buf_gpu: codebook dtype must be F32 \
             (got {:?})",
            codebook_gpu.dtype()
        )));
    }

    let n_groups_codes = codes.shape().iter().map(|&d| d as usize).product::<usize>() / 4;
    let n_groups_scales = scales
        .shape()
        .iter()
        .map(|&d| d as usize)
        .product::<usize>();

    if n_groups_codes != n_groups_scales {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v4_codebook_buf_gpu: n_groups from codes \
             ({n_groups_codes}) != from scales ({n_groups_scales})"
        )));
    }
    let n_groups = n_groups_scales;
    let total_elems = n_groups * GROUP_SIZE;

    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v4_codebook_buf_gpu: original_shape product \
             {shape_product} != expected {total_elems}"
        )));
    }

    let codes_flat = codes.reshape(&[(n_groups * 4) as i32], device)?;
    let scales_flat = scales.reshape(&[n_groups as i32], device)?;
    let cb_flat = if codebook_gpu.ndim() == 1 {
        codebook_gpu.try_clone()?
    } else {
        codebook_gpu.reshape(&[EXPECTED_CB_LEN as i32], device)?
    };

    let kernel_out_dtype = match out_dtype {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => out_dtype,
        _ => Dtype::F32,
    };

    let kernel = dequant_cb_buf_kernel()?;

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&cb_flat)?;
    invoke.add_output_shape(&[total_elems as i32], kernel_out_dtype)?;
    invoke.set_template_dtype("OutT", kernel_out_dtype)?;

    let n_words = n_groups * 4;
    invoke.set_grid(n_words as i32, 1, 1)?;
    invoke.set_thread_group(4, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_dequantize_v4_codebook_buf_gpu: expected 1 output".to_owned(),
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "turboquant_msl_tests.rs"]
mod tests;
