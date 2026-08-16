//! iso4 MSL (Metal Shading Language) GPU kernels.
//!
//! Sibling of [`crate::isoquant_msl`] (iso3). Same algorithm — quaternion SO(4)
//! rotation + Lloyd-Max scalar quantization — with **4-bit codes** (16-entry
//! codebook, dense 8-vals/u32 pack) instead of iso3's 3-bit (8 entries, 10/u32).
//!
//! # Codebook size — 16 centroids, not 32
//!
//! The codec name "iso4" denotes **4-bit V codes** (`bits = 4` in
//! [`crate::isoquant::iso_encode_fast`]), which means 2⁴ = 16 codebook entries.
//! [`crate::turboquant::lloyd_gaussian_codebook(4)`] returns the canonical
//! [`crate::turboquant`]`::CODEBOOK_4BIT` (16 N(0,1) Lloyd-Max centroids).
//! The quaternion SO(4) rotation is identical to iso3 — both partition the
//! head_dim into 4-element groups (one quaternion block per group). Only the
//! codebook size and the bit-pack differ.
//!
//! # Algorithm (per (token, group) thread)
//!
//! 1. Compute the per-token L2 norm (recomputed redundantly per group).
//! 2. Load 4 elements `(w, x, y, z)`, divide by norm, apply Hamilton product
//!    `r = q_L * v` with the fixed golden-ratio quaternion [`FIXED_QUAT`].
//! 3. Per-group scale = `max|r_i| / ISO4_CB_MAX`.
//! 4. 4-bit quantize each `r_i` into the 16-entry Lloyd-Max codebook.
//! 5. Pack 4-bit codes via atomic OR (8 vals/u32, dense — 32 bits used, 0 wasted).
//!
//! Dequantize reverses: unpack → centroid × scale → conjugate Hamilton product
//! → rescale by per-token norm.
//!
//! # Pack format
//!
//! 8 × 4-bit values per u32 (32 bits used). For `ISO4_GS=4`:
//! `WORDS_PER_GROUP = ceil(4 / 8) = 1` u32 per group. Element `e` maps to
//! `word = e / 8`, `shift = (e % 8) * 4`.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md, callers must hold `/tmp/rmlx.<port>.claim` before dispatching.

use std::fmt::Write as _;
use std::sync::OnceLock;

use crate::isoquant::FIXED_QUAT;
use crate::turboquant::lloyd_gaussian_codebook;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Quaternion-block size: 4 elements per group (one quaternion).
/// Matches [`crate::storage::quant_iso_v4::ISO4_GROUP_SIZE`].
const ISO4_GROUP_SIZE: usize = 4;

/// 4-bit values per u32 word (32 bits used, 0 wasted — dense pack).
const VALS_PER_WORD: usize = 8;

/// Words per group for `ISO4_GROUP_SIZE = 4`: ceil(4 / 8) = 1.
const WORDS_PER_GROUP: usize = ISO4_GROUP_SIZE.div_ceil(VALS_PER_WORD);

const _: () = assert!(
    WORDS_PER_GROUP == 1,
    "iso4 MSL kernel assumes 1 word per group; multi-word groups require pack-loop changes"
);

// ── MSL header builder ────────────────────────────────────────────────────────

/// Generate the MSL header string for the iso4 kernels.
///
/// Embeds:
/// - The fixed golden-ratio unit quaternion components (and conjugate).
/// - The 4-bit Lloyd-Max N(0,1) codebook (16 entries) and 15 decision boundaries.
/// - `ISO4_CB_MAX`, `ISO4_GS`, `ISO4_VPW`, `ISO4_WPG` constants.
///
/// # Errors
///
/// Returns `Error::Mlx` if the 4-bit Lloyd-Max codebook lookup fails or yields
/// an empty slice (would indicate an upstream `turboquant` regression). A
/// panic in the `OnceLock` init closure would brick all future iso4 dispatch,
/// so callers must propagate the error rather than `expect`.
fn build_msl_header_iso4() -> Result<String> {
    // Fixed quaternion [w, x, y, z] — golden-ratio unit quat (same as iso3).
    let [qw, qx, qy, qz] = FIXED_QUAT;
    let qw_bits = f32::to_bits(qw);
    let qx_bits = f32::to_bits(qx);
    let qy_bits = f32::to_bits(qy);
    let qz_bits = f32::to_bits(qz);
    let qcx_bits = f32::to_bits(-qx);
    let qcy_bits = f32::to_bits(-qy);
    let qcz_bits = f32::to_bits(-qz);

    // 4-bit Lloyd-Max N(0,1) codebook — 16 entries.
    // Bit patterns match turboquant.rs::CODEBOOK_4BIT.
    let cb4 = lloyd_gaussian_codebook(4).map_err(|e| {
        rmlx_core::error::Error::Mlx(format!("iso4 MSL header: lloyd_gaussian_codebook(4): {e}"))
    })?;
    debug_assert_eq!(
        cb4.len(),
        16,
        "iso4 MSL: 4-bit codebook must have 16 entries"
    );

    let cb_max: f32 = *cb4
        .iter()
        .max_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| {
            rmlx_core::error::Error::Mlx(
                "iso4 MSL header: 4-bit codebook is empty (expected 16 entries)".to_owned(),
            )
        })?;
    let cb_max_bits = f32::to_bits(cb_max);

    let cb_hex: Vec<String> = cb4
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    #[allow(
        clippy::indexing_slicing,
        reason = "windows(2) always produces slices of length 2; indices 0 and 1 are always in-bounds"
    )]
    let boundaries: Vec<f32> = cb4.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    debug_assert_eq!(
        boundaries.len(),
        15,
        "iso4 MSL: 15 decision boundaries expected"
    );
    let bound_hex: Vec<String> = boundaries
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mut s = String::new();

    let _ = write!(
        s,
        "\n// iso4 fixed quaternion q_L = [w, x, y, z] (golden-ratio unit quat).\n\
         // Source: multi_turboquant/methods/isoquant.py; bit-exact with isoquant.rs::FIXED_QUAT.\n\
         constant float ISO4_QW = as_type<float>(0x{qw_bits:08X}u);\n\
         constant float ISO4_QX = as_type<float>(0x{qx_bits:08X}u);\n\
         constant float ISO4_QY = as_type<float>(0x{qy_bits:08X}u);\n\
         constant float ISO4_QZ = as_type<float>(0x{qz_bits:08X}u);\n\
         // Conjugate q̄_L = (w, -x, -y, -z) for dequantize (inverse rotation).\n\
         constant float ISO4_CX = as_type<float>(0x{qcx_bits:08X}u);\n\
         constant float ISO4_CY = as_type<float>(0x{qcy_bits:08X}u);\n\
         constant float ISO4_CZ = as_type<float>(0x{qcz_bits:08X}u);\n"
    );

    let _ = write!(
        s,
        "\n// 4-bit Lloyd-Max N(0,1) codebook — 16 entries.\n\
         // Bit patterns match turboquant.rs::CODEBOOK_4BIT.\n\
         constant float ISO4_CB[16] = {{\n{cb}\n}};\n",
        cb = cb_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\n// 15 midpoint decision boundaries for the 4-bit codebook.\n\
         constant float ISO4_BOUNDS[15] = {{\n{b}\n}};\n",
        b = bound_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\nconstant float ISO4_CB_MAX = as_type<float>(0x{cb_max_bits:08X}u);\n\
         // Quaternion-block group size (4 elements per group).\n\
         constant uint  ISO4_GS = {ISO4_GROUP_SIZE}u;\n\
         // 4-bit values per u32 word (8 vals, 32 bits used — dense pack).\n\
         constant uint  ISO4_VPW = {VALS_PER_WORD}u;\n\
         // u32 words per group = ceil(ISO4_GS / ISO4_VPW) = 1 for GS=4.\n\
         constant uint  ISO4_WPG = {WORDS_PER_GROUP}u;\n"
    );
    Ok(s)
}

// ── MSL kernel sources ────────────────────────────────────────────────────────
//
// Bodies live in `.metal` files so `make check-metal-compiles` and
// `make check-metal-format` see them; `include_str!` embeds them at compile
// time, so the binary still carries no runtime data files.

const QUANTIZE_SOURCE_ISO4: &str = include_str!("metal/isoquant_quantize_iso4.metal");

const DEQUANTIZE_SOURCE_ISO4: &str = include_str!("metal/isoquant_dequantize_iso4.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static ISO4_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ISO4_DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
// Store `std::result::Result<String, String>` (not `rmlx_core::Result`,
// the alias is single-parameter) because we need an owned, `Send` error
// payload. A failure here must not poison the lock for future dispatches,
// so we forward the stored error message as a fresh `Error::Mlx` on every
// call.
static ISO4_KERNEL_HEADER: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn kernel_header_iso4() -> Result<&'static str> {
    match ISO4_KERNEL_HEADER.get_or_init(|| build_msl_header_iso4().map_err(|e| e.to_string())) {
        Ok(s) => Ok(s.as_str()),
        Err(msg) => Err(rmlx_core::error::Error::Mlx(format!(
            "iso4 MSL header init: {msg}"
        ))),
    }
}

fn iso4_quant_kernel() -> Result<&'static MetalKernel> {
    ISO4_QUANT_KERNEL
        .get_or_init(|| {
            let header = kernel_header_iso4()?;
            MetalKernel::new(
                "rmlx_iso4_quantize",
                header,
                QUANTIZE_SOURCE_ISO4,
                &["inp", "n_groups"],
                &["codes_out", "scales_out", "norms_out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("iso4_quantize kernel init: {e}")))
}

fn iso4_dequant_kernel() -> Result<&'static MetalKernel> {
    ISO4_DEQUANT_KERNEL
        .get_or_init(|| {
            let header = kernel_header_iso4()?;
            MetalKernel::new(
                "rmlx_iso4_dequantize",
                header,
                DEQUANTIZE_SOURCE_ISO4,
                &["codes_in", "scales_in", "norms_in", "n_groups"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("iso4_dequantize kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU iso4 quantize.
///
/// Quantize `v_full` (any shape whose total element count is
/// `n_tokens * head_dim`) using quaternion SO(4) rotation + 4-bit Lloyd-Max
/// codebook.
///
/// `head_dim` must be a positive multiple of [`ISO4_GROUP_SIZE`] (= 4).
///
/// # Returns
///
/// `(codes_packed, scales, quaternions, norms)` where:
/// - `codes_packed`: `u32 [n_tokens * n_groups * WORDS_PER_GROUP]` — 4-bit
///   indices, 8 per word (dense pack).
/// - `scales`: `f32 [n_tokens * n_groups]` — per-group scale.
/// - `quaternions`: `f32 [n_tokens * n_groups * 4]` — per-group unit quaternion.
///   All entries are the fixed [`FIXED_QUAT`] in this implementation.
/// - `norms`: `f32 [n_tokens * n_groups]` — per-group L2 norm slot (all slots
///   in the same token share the same value).
///
/// # Errors
///
/// Returns `Error::Quant` for invalid `head_dim`.
/// Returns `Error::Mlx` if Metal kernel compilation fails.
pub fn iso_quantize_v4_gpu(
    v_full: &Array,
    head_dim: usize,
    device: Device,
) -> Result<(Array, Array, Array, Array)> {
    if head_dim == 0 || !head_dim.is_multiple_of(ISO4_GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_quantize_v4_gpu: head_dim={head_dim} must be a positive multiple of \
             ISO4_GROUP_SIZE={ISO4_GROUP_SIZE}"
        )));
    }

    let total_elems: usize = v_full.shape().iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(head_dim) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_quantize_v4_gpu: total elements {total_elems} not a multiple of \
             head_dim={head_dim}"
        )));
    }
    let n_tokens = total_elems / head_dim;
    let n_groups = head_dim / ISO4_GROUP_SIZE;
    let total_groups = n_tokens * n_groups;

    tracing::debug!(
        target: "rmlx::kv_quant::iso4",
        n_tokens,
        n_groups,
        head_dim,
        "iso4 GPU encode block"
    );

    let v_flat = if v_full.ndim() == 1 {
        v_full.try_clone()?
    } else {
        v_full.reshape(&[total_elems as i32], device)?
    };
    let v_f32 = if v_flat.dtype() == Dtype::F32 {
        v_flat
    } else {
        v_flat.astype(Dtype::F32, device)?
    };

    let n_groups_bytes = (n_groups as u32).to_le_bytes();
    let n_groups_arr = Array::from_bytes(&n_groups_bytes, &[1_i32], Dtype::U32)?;

    let kernel = iso4_quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&v_f32)?;
    invoke.add_input(&n_groups_arr)?;

    invoke.add_output_shape(&[(total_groups * WORDS_PER_GROUP) as i32], Dtype::U32)?;
    invoke.add_output_shape(&[total_groups as i32], Dtype::F32)?;
    invoke.add_output_shape(&[total_groups as i32], Dtype::F32)?;

    // Zero-initialise: codes_out written via atomic OR.
    invoke.set_init_value(0.0)?;

    invoke.set_grid(total_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    tracing::trace!(n_tokens, n_groups, head_dim, "iso4 quantize dispatched");
    if outputs.len() < 3 {
        return Err(rmlx_core::error::Error::Mlx(
            "iso_quantize_v4_gpu: expected 3 outputs".to_owned(),
        ));
    }
    let norms = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);

    // Quaternions: all fixed — emit a constant array (total_groups * 4 entries).
    let quat_len = total_groups * 4;
    let quat_bytes: Vec<u8> = FIXED_QUAT
        .iter()
        .cycle()
        .take(quat_len)
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    let quaternions = Array::from_bytes(&quat_bytes, &[quat_len as i32], Dtype::F32)?;

    Ok((codes, scales, quaternions, norms))
}

/// GPU iso4 dequantize.
///
/// Reconstruct f32 tensor from `(codes_packed, scales, _quaternions, norms)`
/// produced by [`iso_quantize_v4_gpu`].
///
/// `head_dim` must match the encode call. `out_dtype` selects the element type
/// of the returned array.
///
/// # Errors
///
/// Returns `Error::Quant` for shape mismatches.
/// Returns `Error::Mlx` if Metal kernel compilation fails.
pub fn iso_dequantize_v4_gpu(
    codes_packed: &Array,
    scales: &Array,
    _quaternions: &Array,
    norms: &Array,
    head_dim: usize,
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    if head_dim == 0 || !head_dim.is_multiple_of(ISO4_GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v4_gpu: head_dim={head_dim} must be a positive multiple of \
             ISO4_GROUP_SIZE={ISO4_GROUP_SIZE}"
        )));
    }

    let total_groups: usize = norms.shape().iter().map(|&d| d as usize).product();
    let n_groups = head_dim / ISO4_GROUP_SIZE;
    if total_groups == 0 || !total_groups.is_multiple_of(n_groups) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v4_gpu: norms length {total_groups} not a multiple of \
             n_groups={n_groups}"
        )));
    }
    let n_tokens = total_groups / n_groups;
    let total_elems = n_tokens * head_dim;

    let expected_codes = total_groups * WORDS_PER_GROUP;
    let codes_len: usize = codes_packed.shape().iter().map(|&d| d as usize).product();
    if codes_len != expected_codes {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v4_gpu: codes length {codes_len} != expected {expected_codes}"
        )));
    }
    let scales_len: usize = scales.shape().iter().map(|&d| d as usize).product();
    if scales_len != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v4_gpu: scales length {scales_len} != total_groups {total_groups}"
        )));
    }

    let codes_flat = codes_packed.reshape(&[expected_codes as i32], device)?;
    let scales_flat = scales.reshape(&[total_groups as i32], device)?;
    let norms_flat = norms.reshape(&[total_groups as i32], device)?;

    let n_groups_bytes = (n_groups as u32).to_le_bytes();
    let n_groups_arr = Array::from_bytes(&n_groups_bytes, &[1_i32], Dtype::U32)?;

    let kernel = iso4_dequant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&norms_flat)?;
    invoke.add_input(&n_groups_arr)?;
    invoke.add_output_shape(&[total_elems as i32], Dtype::F32)?;

    invoke.set_grid(total_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    tracing::trace!(n_tokens, n_groups, head_dim, "iso4 dequantize dispatched");
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "iso_dequantize_v4_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);

    let out = if out_dtype == Dtype::F32 {
        out_flat
    } else {
        out_flat.astype(out_dtype, device)?
    };
    Ok(out)
}

/// Read the GPU iso4 encode outputs back to host and return the four CPU
/// vectors used by [`crate::storage::quant_iso_v::IsoBlocks`].
///
/// `n_tokens` and `n_groups` describe the (token, group) grid that produced
/// the GPU outputs. `norms_gpu` is written per-group on the GPU (one slot per
/// (token, group)); this helper deduplicates by taking the first slot per
/// token so the resulting `norms` vector matches the CPU
/// [`crate::isoquant::iso_decode_fast`] contract (one norm per token).
///
/// # Errors
///
/// Returns `Error::Mlx` on Array materialise / byte-decode failure or
/// `Error::Quant` if the output shapes do not match the declared grid.
pub fn iso4_gpu_outputs_to_cpu(
    codes_gpu: &Array,
    scales_gpu: &Array,
    _quaternions_gpu: &Array,
    norms_gpu: &Array,
    n_tokens: usize,
    n_groups: usize,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let total_groups = n_tokens.saturating_mul(n_groups);
    if total_groups == 0 {
        return Err(rmlx_core::error::Error::Quant(
            "iso4_gpu_outputs_to_cpu: n_tokens * n_groups must be > 0".to_owned(),
        ));
    }

    // chunks_exact(4) yields slices of exactly 4 bytes by contract, so the
    // try_into never fails; expect documents the invariant for the type check.
    // eval-ok: host readback — the `to_bytes()` below copies this array to
    // CPU, so it has to be materialised first. Not a kernel-input barrier: this
    // runs once per quantize call, off the per-decode-step path.
    codes_gpu.eval()?;
    #[allow(
        clippy::expect_used,
        reason = "chunks_exact(4) yields slices of exactly 4 bytes"
    )]
    let codes: Vec<u32> = codes_gpu
        .to_bytes()?
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("len 4 by chunks_exact contract")))
        .collect();
    // eval-ok: host readback — the `to_bytes()` below copies this array to
    // CPU, so it has to be materialised first. Not a kernel-input barrier: this
    // runs once per quantize call, off the per-decode-step path.
    scales_gpu.eval()?;
    #[allow(
        clippy::expect_used,
        reason = "chunks_exact(4) yields slices of exactly 4 bytes"
    )]
    let scales: Vec<f32> = scales_gpu
        .to_bytes()?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("len 4 by chunks_exact contract")))
        .collect();
    // eval-ok: host readback — the `to_bytes()` below copies this array to
    // CPU, so it has to be materialised first. Not a kernel-input barrier: this
    // runs once per quantize call, off the per-decode-step path.
    norms_gpu.eval()?;
    #[allow(
        clippy::expect_used,
        reason = "chunks_exact(4) yields slices of exactly 4 bytes"
    )]
    let norms_per_group: Vec<f32> = norms_gpu
        .to_bytes()?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("len 4 by chunks_exact contract")))
        .collect();

    // FIXED_QUAT cycle; skip GPU readback — kernel doesn't differentiate per-group
    // (every group's quaternion is the same constant FIXED_QUAT). The
    // `_quaternions_gpu` Array is retained in the encode-tuple return for ABI
    // parity with the iso3 sibling; no caller currently reads it.
    let mut quats = Vec::with_capacity(total_groups * 4);
    for _ in 0..total_groups {
        quats.extend_from_slice(&FIXED_QUAT);
    }

    let expected_codes = total_groups * WORDS_PER_GROUP;
    if codes.len() != expected_codes {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso4_gpu_outputs_to_cpu: codes len {} != expected {expected_codes}",
            codes.len()
        )));
    }
    if scales.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso4_gpu_outputs_to_cpu: scales len {} != expected {total_groups}",
            scales.len()
        )));
    }
    if norms_per_group.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso4_gpu_outputs_to_cpu: norms len {} != expected {total_groups}",
            norms_per_group.len()
        )));
    }

    // Deduplicate per-group norms to per-token (first slot of each token).
    // All n_groups slots within a token are identical by GPU-kernel contract.
    let mut norms = Vec::with_capacity(n_tokens);
    for tok in 0..n_tokens {
        let i = tok * n_groups;
        #[allow(
            clippy::indexing_slicing,
            reason = "tok < n_tokens; i = tok * n_groups < total_groups == norms_per_group.len() (validated above)"
        )]
        norms.push(norms_per_group[i]);
    }

    Ok((codes, scales, quats, norms))
}

#[cfg(test)]
#[path = "isoquant_msl_v4_tests.rs"]
mod isoquant_msl_v4_tests;
