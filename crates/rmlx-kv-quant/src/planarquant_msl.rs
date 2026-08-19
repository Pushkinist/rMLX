// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! PlanarQuant V4 MSL (Metal Shading Language) GPU kernels.
//!
//! # What this is
//!
//! GPU (Metal) versions of the CPU `planar_quantize` / `planar_dequantize`
//! functions from `crate::planarquant`. Kernels are siblings, not
//! replacements — public API names differ (`_gpu` suffix). CPU path stays as
//! the correctness reference.
//!
//! # rMLX-first on Apple Silicon
//!
//! `johndpope/llama-cpp-turboquant feature/planarquant-kv-cache` Metal kernels
//! fall back to CPU (issue #7). rMLX is the first working Apple Silicon
//! PlanarQuant implementation as of 2026-05.
//!
//! # Algorithm (matches CPU path)
//!
//! For each group of 32 elements (16 pairs):
//!   1. Per pair `(a, b)`: try all 16 Givens rotations, pick the one minimizing
//!      reconstruction error at pair-local scale.
//!   2. Apply chosen rotation → compute pair scale = `max(|ya|, |yb|) / max_centroid`.
//! 3. 4-bit quantize both rotated elements using the Lloyd-Max N(0,1) codebook.
//!
//! Dequantize reverses: pull rotation index → pair scale → apply R_k^T.
//!
//! # MSL output layout
//!
//! - `codes`: `u32 [n_groups * 4]` — 8 4-bit indices per uint32, LSB-first,
//!   same packing as TurboQuant (elements 0..7 in word 0, 8..15 in word 1, etc.)
//!
//! - `scales`: `f32 [n_pairs]` — one f32 per pair (not per group!).
//!
//! - `rot32`: `u32 [n_groups * 2]` — packed rotation indices, 4-bit per pair,
//!   8 pairs per uint32. Group has 16 pairs = 2 words.
//!
//! # Kernel strategy
//!
//! One thread per pair (2 elements). Grid = `n_pairs` flat threads.
//! Threadgroup size = 1. Each thread independently:
//! - Tries all 16 rotations.
//! - Selects the best.
//! - Writes scale, rotation index, and 4-bit code indices via atomic OR.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md, callers must hold `/tmp/rmlx.<port>.claim` before dispatching.

use std::fmt::Write as _;
use std::sync::OnceLock;

use crate::planarquant::{planar_rotation_codebook, N_ROTATIONS};
use crate::turboquant::GROUP_SIZE;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Pairs per group: `GROUP_SIZE / 2 = 16`.
const PAIRS_PER_GROUP: usize = GROUP_SIZE / 2;

// ── MSL header builder ────────────────────────────────────────────────────────

/// Generate the MSL header string from the CPU rotation codebook.
///
/// Ensures GPU and CPU codebooks are identical — any change to
/// `planar_rotation_codebook()` is automatically reflected in the MSL kernels.
///
/// The header embeds:
/// - 16-entry Givens rotation codebook (4 f32 bit-exact patterns per entry).
/// - 4-bit Lloyd-Max N(0,1) codebook (16 entries) and 15 decision boundaries.
/// - `CB_MAX` and `PAIRS_PER_GROUP` constants.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn build_msl_header() -> String {
    // Rotation codebook from CPU.
    let rot_cb = planar_rotation_codebook();
    assert_eq!(rot_cb.len(), N_ROTATIONS);

    let rot_entries: Vec<String> = rot_cb
        .iter()
        .map(|e| {
            let c = f32::to_bits(e[0]);
            let neg_s = f32::to_bits(e[1]);
            let s = f32::to_bits(e[2]);
            let c2 = f32::to_bits(e[3]);
            format!(
                "    {{as_type<float>(0x{c:08X}u), as_type<float>(0x{neg_s:08X}u), \
                 as_type<float>(0x{s:08X}u), as_type<float>(0x{c2:08X}u)}}"
            )
        })
        .collect();

    // 4-bit Lloyd-Max optimal N(0,1) codebook — shared with TurboQuant.
    //
    // Bit patterns are identical to turboquant_msl.rs::KERNEL_HEADER CB[] and
    // turboquant.rs::CODEBOOK_4BIT (canonical source of truth).
    // Derived by turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100).
    // Previous version used N(0,1) quantile centroids (Gaussian-quantile, NOT Lloyd-Max).
    let cb_entries: [f32; 16] = [
        f32::from_bits(0xC02D_EE42), // -2.7176671
        f32::from_bits(0xC003_563B), // -2.0521381
        f32::from_bits(0xBFCC_E718), // -1.6008024
        f32::from_bits(0xBF9E_B6FA), // -1.2399590
        f32::from_bits(0xBF6D_A172), // -0.9282447
        f32::from_bits(0xBF25_5816), // -0.6458753
        f32::from_bits(0xBEC3_29CB), // -0.3811782
        f32::from_bits(0xBE01_1273), // -0.1260469
        f32::from_bits(0x3E01_1273), //  0.1260469
        f32::from_bits(0x3EC3_29CB), //  0.3811782
        f32::from_bits(0x3F25_5816), //  0.6458753
        f32::from_bits(0x3F6D_A172), //  0.9282447
        f32::from_bits(0x3F9E_B6FA), //  1.2399590
        f32::from_bits(0x3FCC_E718), //  1.6008024
        f32::from_bits(0x4003_563B), //  2.0521381
        f32::from_bits(0x402D_EE42), //  2.7176671
    ];
    let cb_max: f32 = f32::from_bits(0x402D_EE42); // 2.7176671 — Lloyd-Max N(0,1) 4-bit max centroid

    let cb_hex: Vec<String> = cb_entries
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let boundaries: Vec<f32> = cb_entries.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    let bound_hex: Vec<String> = boundaries
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let cb_max_bits = f32::to_bits(cb_max);

    let mut s = String::new();
    // write!(String) is infallible — let _ discards the unit Ok.
    let _ = write!(
        s,
        "\n// PlanarQuant rotation codebook — 16 Givens rotations at theta_k = k*pi/16.\n\
 // Generated from planar_rotation_codebook() — bit-exact with CPU path.\n\
 // Each entry: [cos t, -sin t, sin t, cos t] (row-major 2x2).\n\
         constant float ROT_CB[{N}][4] = {{\n{entries}\n}};\n",
        N = N_ROTATIONS,
        entries = rot_entries.join(",\n")
    );
    let _ = write!(
        s,
        "\n// 4-bit Lloyd-Max N(0,1) codebook — 16 entries (shared with TurboQuant).\n\
 // Bit patterns match turboquant_msl.rs::KERNEL_HEADER CB[] and turboquant.rs::CODEBOOK_4BIT.\n\
 // Derived by turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100).\n\
         constant float CB[16] = {{\n{cb}\n}};\n",
        cb = cb_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\n// 15 midpoint decision boundaries.\n\
         constant float BOUNDARIES[15] = {{\n{b}\n}};\n",
        b = bound_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\nconstant float CB_MAX = as_type<float>(0x{cb_max_bits:08X}u);\n\
         constant uint  PAIRS_PER_GROUP_C = {PAIRS_PER_GROUP}u;\n"
    );
    s
}

// ── MSL kernel sources ────────────────────────────────────────────────────────

// Quantize kernel: one thread per pair, flat grid of n_pairs threads.
//
// Thread pair_id:
// 1. Load pair (a, b) from inp[pair_id*2], inp[pair_id*2+1].
// 2. Try all 16 rotations, pick best (min reconstruction error at pair-local scale).
// 3. Write scale (one f32 per pair) to scales[pair_id].
// 4. Pack rotation index (4-bit) into rot32[] via atomic OR.
// 5. Pack code indices (4-bit each) into codes[] via atomic OR.
//
// Packing:
// codes: u32 [n_groups*4], 8 4-bit indices per word, elements in group-order.
// rot32: u32 [n_groups*2], 8 rotation indices (4-bit) per word.

const QUANTIZE_SOURCE: &str = include_str!("metal/planarquant_quantize.metal");

// Dequantize kernel: one thread per pair, flat grid of n_pairs threads.

const DEQUANTIZE_SOURCE: &str = include_str!("metal/planarquant_dequantize.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static KERNEL_HEADER: OnceLock<String> = OnceLock::new();

fn kernel_header() -> &'static str {
    KERNEL_HEADER.get_or_init(build_msl_header)
}

fn quant_kernel() -> Result<&'static MetalKernel> {
    QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_pq4_quantize",
                kernel_header(),
                QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales", "rot32"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("pq4_quantize kernel init: {e}")))
}

fn dequant_kernel() -> Result<&'static MetalKernel> {
    DEQUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_pq4_dequantize",
                kernel_header(),
                DEQUANTIZE_SOURCE,
                &["codes", "scales", "rot32"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("pq4_dequantize kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU PlanarQuant V4 quantize.
///
/// Quantize `x` (any shape; total elements must be a multiple of `GROUP_SIZE=32`)
/// using per-pair Givens rotation + 4-bit Lloyd-Max N(0,1) codebook.
///
/// # Returns
///
/// `(codes, scales, rot32)` where:
/// - `codes`: `u32 [n_groups * 4]` — 8 4-bit indices per word.
/// - `scales`: `f32 [n_pairs]` — one scale per pair (finer than TurboQuant).
/// - `rot32`: `u32 [n_groups * 2]` — 4-bit rotation index per pair,
///   8 pairs per word, 2 words per group.
///
/// # Errors
///
/// Returns `Error::Quant` if total elements not a multiple of `GROUP_SIZE`.
// f32-out-ok: of the three buffers only `scales` is f32 (`codes` and `rot32`
// are u32), and it is read back solely by MSL kernels that declare it
// `device const float*`: `planar_dequantize_v4_gpu`, `planar_fused_qk`,
// `planar_flash_decode` P1, and the sparse phase-1 / phase-2 pair. No MLX
// op would take its operand width from them, the way `quantized_matmul` and
// `dequantize` take theirs from an `mx.quantize` 3-tuple.
pub fn planar_quantize_v4_gpu(x: &Array, device: Device) -> Result<(Array, Array, Array)> {
    let total_elems: usize = x.shape().iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "planar_quantize_v4_gpu: total elements {total_elems} not a multiple \
             of GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    let n_pairs = total_elems / 2;
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

    let kernel = quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;

    // codes: u32 [n_groups * 4].
    invoke.add_output_shape(&[(n_groups * 4) as i32], Dtype::U32)?;
    // scales: f32 [n_pairs].
    invoke.add_output_shape(&[n_pairs as i32], Dtype::F32)?;
    // rot32: u32 [n_groups * 2] (16 pairs/group × 4bits = 2 words).
    invoke.add_output_shape(&[(n_groups * 2) as i32], Dtype::U32)?;

    // Zero-initialise outputs: codes and rot32 are written via atomic_fetch_or.
    // MLX reuses Metal buffers from a pool; without explicit zero-init, recycled
    // buffers retain stale bits that corrupt the atomic-OR accumulation.
    invoke.set_init_value(0.0)?;

    // Grid: n_pairs threads (one per pair). Threadgroup: 1.
    invoke.set_grid(n_pairs as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 3 {
        return Err(rmlx_core::error::Error::Mlx(
            "planar_quantize_v4_gpu: expected 3 outputs".to_owned(),
        ));
    }
    let rot32 = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales, rot32))
}

/// GPU PlanarQuant V4 dequantize.
///
/// Reconstruct the tensor from `(codes, scales, rot32)` produced by
/// [`planar_quantize_v4_gpu`], in `out_dtype`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn planar_dequantize_v4_gpu(
    codes: &Array,
    scales: &Array,
    rot32: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    let n_pairs: usize = scales.shape().iter().map(|&d| d as usize).product();
    let total_elems = n_pairs * 2;
    let n_groups = total_elems / GROUP_SIZE;

    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "planar_dequantize_v4_gpu: shape product {shape_product} != {total_elems}"
        )));
    }

    let codes_flat = codes.reshape(&[(n_groups * 4) as i32], device)?;
    let scales_flat = scales.reshape(&[n_pairs as i32], device)?;
    let rot32_flat = rot32.reshape(&[(n_groups * 2) as i32], device)?;

    let kernel = dequant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&rot32_flat)?;
    invoke.add_output_shape(&[total_elems as i32], Dtype::F32)?;

    invoke.set_grid(n_pairs as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "planar_dequantize_v4_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);
    // The kernel reconstructs in f32; hand the caller its own dtype back.
    // Dequantized K/V flows straight into SDPA, and an f32 operand there
    // promotes the whole attention op — and the residual stream behind it.
    let out_flat = out_flat.astype(out_dtype, device)?;

    if original_shape.len() == 1 && original_shape[0] == total_elems as i32 {
        return Ok(out_flat);
    }
    out_flat.reshape(original_shape, device)
}

// ── V3 (3-bit) ───────────────────────────────────────────────────────────────
//
// 3-bit Givens rotation + per-pair scale codec.
// Same rotation codebook as V4; 8-centroid 3-bit Lloyd-Max N(0,1) codebook.
//
// Pack format: 10 vals/u32 (3 × 10 = 30 bits, 2 wasted). With GROUP_SIZE=32:
//   ceil(32/10) = 4 u32 words per group — same word count as 4-bit (8 vals/u32 × 4).
//   Element e (0-based within group) lives in:
//     word  = e / 10
//     shift = (e % 10) * 3
//
// Rotation: same rot32 layout as V4 (4-bit index, 8 pairs per word, 2 words/group).

/// Generate the MSL header for the 3-bit kernel.
///
/// Uses the same rotation codebook as V4 but a 3-bit Lloyd-Max N(0,1)
/// codebook (8 centroids, 7 decision boundaries).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::expect_used,
    reason = "lloyd_gaussian_codebook(3) is always Ok (3-bit is a registered codebook); \
              codebook slice is non-empty by construction (8 entries). Both invariants are \
              verified at dev-time by debug_assert_eq! below and by the \
              `turbo_codebook_3bit_has_8_entries_monotonic` + \
              `lloyd_gaussian_codebook_3bit_entries_are_finite` tests in turboquant_tests.rs."
)]
fn build_msl_header_v3() -> String {
    use crate::turboquant::lloyd_gaussian_codebook;

    // Rotation codebook from CPU (same 16-entry codebook as V4).
    let rot_cb = planar_rotation_codebook();
    // dev-time guard: codebook size is a build-time constant; test coverage via
    // `turbo_codebook_3bit_has_8_entries_monotonic` catches drift in release builds.
    debug_assert_eq!(rot_cb.len(), N_ROTATIONS, "rotation codebook size");

    let rot_entries: Vec<String> = rot_cb
        .iter()
        .map(|e| {
            let c = f32::to_bits(e[0]);
            let neg_s = f32::to_bits(e[1]);
            let s = f32::to_bits(e[2]);
            let c2 = f32::to_bits(e[3]);
            format!(
                "    {{as_type<float>(0x{c:08X}u), as_type<float>(0x{neg_s:08X}u), \
                 as_type<float>(0x{s:08X}u), as_type<float>(0x{c2:08X}u)}}"
            )
        })
        .collect();

    // 3-bit Lloyd-Max optimal N(0,1) codebook — 8 entries.
    // From turboquant.rs::CODEBOOK_3BIT (canonical source of truth).
    let cb3 = lloyd_gaussian_codebook(3).expect("3-bit codebook must exist");
    // dev-time guard: 8 entries is a structural invariant of the 3-bit codebook.
    // Release correctness enforced by `turbo_codebook_3bit_has_8_entries_monotonic`
    // and `lloyd_gaussian_codebook_3bit_entries_are_finite` in turboquant_tests.rs.
    debug_assert_eq!(cb3.len(), 8, "3-bit codebook must have 8 entries");

    let cb_max3: f32 = *cb3
        .iter()
        .max_by(|a, b| {
            // SAFETY: lloyd_gaussian_codebook(3) returns finite f32 values; enforced by
            // `lloyd_gaussian_codebook_3bit_entries_are_finite` test in turboquant_tests.rs.
            // NaN-poisoned codebook would produce a silent wrong max — the test gate
            // catches this before it reaches a release build.
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("non-empty");

    let cb_hex: Vec<String> = cb3
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let boundaries: Vec<f32> = cb3.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    // dev-time guard: 8 entries → 7 boundaries; structural invariant of the 3-bit codebook.
    debug_assert_eq!(boundaries.len(), 7, "3-bit: 7 decision boundaries expected");
    let bound_hex: Vec<String> = boundaries
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let cb_max_bits = f32::to_bits(cb_max3);

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// PlanarQuant 3-bit — rotation codebook (same 16 Givens entries as V4).\n\
         constant float ROT_CB3[{N}][4] = {{\n{entries}\n}};\n",
        N = N_ROTATIONS,
        entries = rot_entries.join(",\n")
    );
    let _ = write!(
        s,
        "\n// 3-bit Lloyd-Max N(0,1) codebook — 8 entries.\n\
         constant float CB3[8] = {{\n{cb}\n}};\n",
        cb = cb_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\n// 7 midpoint decision boundaries for 3-bit codebook.\n\
         constant float BOUNDARIES3[7] = {{\n{b}\n}};\n",
        b = bound_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\nconstant float CB_MAX_3 = as_type<float>(0x{cb_max_bits:08X}u);\n\
         constant uint  PAIRS_PER_GROUP_C3 = {PAIRS_PER_GROUP}u;\n"
    );
    s
}

// Quantize kernel (3-bit): one thread per pair.
//
// Pack: 10 vals/u32 → element e in group: word = e/10, shift = (e%10)*3.
// Mask: 0x7u (3 bits).
const QUANTIZE_SOURCE_V3: &str = include_str!("metal/planarquant_quantize_v3.metal");

// Dequantize kernel (3-bit): one thread per pair.
const DEQUANTIZE_SOURCE_V3: &str = include_str!("metal/planarquant_dequantize_v3.metal");

// ── V3 kernel singletons ──────────────────────────────────────────────────────

static QUANT_KERNEL_V3: OnceLock<Result<MetalKernel>> = OnceLock::new();
static DEQUANT_KERNEL_V3: OnceLock<Result<MetalKernel>> = OnceLock::new();
static KERNEL_HEADER_V3: OnceLock<String> = OnceLock::new();

fn kernel_header_v3() -> &'static str {
    KERNEL_HEADER_V3.get_or_init(build_msl_header_v3)
}

fn quant_kernel_v3() -> Result<&'static MetalKernel> {
    QUANT_KERNEL_V3
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_pq3_quantize",
                kernel_header_v3(),
                QUANTIZE_SOURCE_V3,
                &["inp"],
                &["codes", "scales", "rot32"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("pq3_quantize kernel init: {e}")))
}

fn dequant_kernel_v3() -> Result<&'static MetalKernel> {
    DEQUANT_KERNEL_V3
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_pq3_dequantize",
                kernel_header_v3(),
                DEQUANTIZE_SOURCE_V3,
                &["codes", "scales", "rot32"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("pq3_dequantize kernel init: {e}")))
}

/// GPU PlanarQuant V3 quantize.
///
/// Quantize `x` using per-pair Givens rotation + 3-bit Lloyd-Max N(0,1) codebook.
/// Pack format: 10 vals/u32 (3 × 10 = 30 bits, 2 wasted); 4 words/group.
///
/// # Returns
///
/// `(codes, scales, rot32)` where:
/// - `codes`: `u32 [n_groups * 4]` — 10 3-bit indices per word (30 bits used).
/// - `scales`: `f32 [n_pairs]` — one scale per pair.
/// - `rot32`: `u32 [n_groups * 2]` — 4-bit rotation index per pair, 8 per word.
///
/// # Errors
///
/// Returns `Error::Quant` if total elements not a multiple of `GROUP_SIZE`.
// f32-out-ok: only `scales` is f32 here (`codes` and `rot32` are u32), and its
// one reader is `planar_dequantize_v3_gpu`, an MSL kernel that declares it
// `device const float*` — the 3-bit codec has no fused-QK arm. No MLX
// op would take its operand width from them, the way `quantized_matmul` and
// `dequantize` take theirs from an `mx.quantize` 3-tuple.
pub fn planar_quantize_v3_gpu(x: &Array, device: Device) -> Result<(Array, Array, Array)> {
    let total_elems: usize = x.shape().iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "planar_quantize_v3_gpu: total elements {total_elems} not a multiple \
             of GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    let n_pairs = total_elems / 2;
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

    let kernel = quant_kernel_v3()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;

    // codes: u32 [n_groups * 4] (10 vals/u32, same word count as 4-bit).
    invoke.add_output_shape(&[(n_groups * 4) as i32], Dtype::U32)?;
    // scales: f32 [n_pairs].
    invoke.add_output_shape(&[n_pairs as i32], Dtype::F32)?;
    // rot32: u32 [n_groups * 2] (4-bit rotation indices, same as V4).
    invoke.add_output_shape(&[(n_groups * 2) as i32], Dtype::U32)?;

    // Zero-initialise: atomic OR accumulation requires clean buffers.
    invoke.set_init_value(0.0)?;

    // Grid: n_pairs threads (one per pair). Threadgroup: 1.
    invoke.set_grid(n_pairs as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 3 {
        return Err(rmlx_core::error::Error::Mlx(
            "planar_quantize_v3_gpu: expected 3 outputs".to_owned(),
        ));
    }
    let rot32 = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales, rot32))
}

/// GPU PlanarQuant V3 dequantize.
///
/// Reconstruct the tensor from `(codes, scales, rot32)` produced by
/// [`planar_quantize_v3_gpu`], in `out_dtype`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn planar_dequantize_v3_gpu(
    codes: &Array,
    scales: &Array,
    rot32: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    let n_pairs: usize = scales.shape().iter().map(|&d| d as usize).product();
    let total_elems = n_pairs * 2;
    let n_groups = total_elems / GROUP_SIZE;

    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "planar_dequantize_v3_gpu: shape product {shape_product} != {total_elems}"
        )));
    }

    let codes_flat = codes.reshape(&[(n_groups * 4) as i32], device)?;
    let scales_flat = scales.reshape(&[n_pairs as i32], device)?;
    let rot32_flat = rot32.reshape(&[(n_groups * 2) as i32], device)?;

    let kernel = dequant_kernel_v3()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&rot32_flat)?;
    invoke.add_output_shape(&[total_elems as i32], Dtype::F32)?;

    invoke.set_grid(n_pairs as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "planar_dequantize_v3_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);
    // The kernel reconstructs in f32; hand the caller its own dtype back.
    // Dequantized K/V flows straight into SDPA, and an f32 operand there
    // promotes the whole attention op — and the residual stream behind it.
    let out_flat = out_flat.astype(out_dtype, device)?;

    if original_shape.len() == 1 && original_shape[0] == total_elems as i32 {
        return Ok(out_flat);
    }
    out_flat.reshape(original_shape, device)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "planarquant_msl_tests.rs"]
mod tests;
