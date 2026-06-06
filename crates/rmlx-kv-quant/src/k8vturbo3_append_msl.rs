// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! — TurboQuant V3 (Lloyd-Max 3-bit) Metal kernels.
//!
//! # What this is
//!
//! GPU (Metal) quantize / dequantize kernels for the 3-bit Lloyd-Max N(0,1)
//! TurboQuant codebook (`crate::turboquant::CODEBOOK_3BIT`), mirroring
//! [`crate::turboquant_msl`]'s V4 path so that
//! [`KvStorage::K8VTurbo3`](super::storage::KvStorage::K8VTurbo3) can stay on
//! the GPU instead of the CPU round-trip that the prototype paid.
//!
//! # File-name choice
//!
//! The file lives under `kv_cache/` and is named `k8vturbo3_append_msl.rs`
//! per the spec; conceptually these are the same plain
//! quantize / dequantize kernels as `turboquant_msl.rs`, not the fused
//! append-into-flash-buffer kernels of `k8v4_append_msl.rs`. K8VTurbo3 does
//! not use TurboFlash; the standard `QuantV::append` / `dequantize_choice`
//! GPU path calls these functions for the `bits == 3` branch.
//!
//! # Codebook (Lloyd-Max optimal, N(0,1))
//!
//! Eight centroids, bit-exact with `crate::turboquant::CODEBOOK_3BIT`.
//! Maximum centroid `CB3_MAX = 2.1519449` is the denominator for the
//! per-group scale: `scale = max(|x|) / CB3_MAX`.
//!
//! Seven decision boundaries are midpoints between consecutive centroids,
//! exactly what `nearest_centroid` computes on the CPU.
//!
//! # Pack format
//!
//! For a group of `GROUP_SIZE = 32` elements at 3 bits / element:
//!
//! - 32 × 3 = 96 bits = exactly **3 `u32`** words.
//! - Element `e` occupies bits `[e*3 .. e*3+3)` of the concatenated 96-bit
//!   little-endian stream. This matches `pack_index` / `unpack_index` in
//!   `crate::turboquant` for `bits = 3` reinterpreted as 3 LE `u32`s.
//!
//! Three out of 32 threads in a threadgroup (`elem ∈ {0, 11, 22}`) act as
//! writer lanes — one per output word. Each writer accumulates the bits
//! from all 32 indices into a 64-bit local register via signed-shift
//! insertion, then writes the low 32 bits of its target word.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Dispatch status
//!
//! The bench (Gemma4-e4b / 26b, ctx ~17k) showed that wiring this
//! GPU kernel into the K8VTurbo3 V-side update path regresses decode TPS
//! by 3.5% (e4b) and 6.9% (26b) vs the `Mixed{v_bits:3}` affine baseline —
//! both fail the −2% gate. The CPU dequant path is therefore kept
//! as the canonical K8VTurbo3 V-side path (`update_k8vturbo3` in
//! `kvcache.rs`). This module is retained as a future-reference hook,
//! with full unit-test coverage of bit-exact CPU↔GPU equivalence so that
//! re-wiring it later (e.g. once Gemma4-arch PPL coverage exists) is a
//! one-line change at the dispatch site. See
//! `docs/research/turboquant_v3_vs_affine_v3.md` "Second pass: Metal
//! 3-bit kernel" for the bench numbers and verdict.

#![allow(dead_code)] // Future-gated hook, see module-level "Dispatch status".

use crate::turboquant::GROUP_SIZE;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};
use std::sync::OnceLock;

// -- Codebook constants (mirror crate::turboquant::CODEBOOK_3BIT) -------

/// Lloyd-Max optimal N(0,1) codebook for 3-bit (8 entries).
///
/// Kept in Rust as a documentation anchor; the MSL kernel embeds the same
/// values via `as_type<float>(0x...)` so any future change is made in one
/// place first and mirrored.
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

// -- MSL kernel sources ------------------------------------------------------

/// MSL header — embeds the 8 Lloyd-Max N(0,1) 3-bit centroids and their 7
/// midpoint decision boundaries as bit-exact `as_type<float>(0x...)`
/// constants.
///
/// Bit patterns derived from `f32::to_bits` on the values in
/// `crates/rmlx-quant/src/turboquant.rs::CODEBOOK_3BIT`. Verified by the
/// `cb3_constants_bit_exact` unit test below.
const V3_KERNEL_HEADER: &str = r"
// 3-bit TurboQuant codebook: 8 Lloyd-Max optimal N(0,1) centroids.
// Bit-exact with crate::turboquant::CODEBOOK_3BIT.
constant float CB3[8] = {
    as_type<float>(0xC009B977u),  // -2.1519449
    as_type<float>(0xBFAC0532u),  // -1.3439085
    as_type<float>(0xBF418987u),  // -0.7560048
    as_type<float>(0xBE7AF9EBu),  // -0.2450940
    as_type<float>(0x3E7AF9EBu),  //  0.2450940
    as_type<float>(0x3F418987u),  //  0.7560048
    as_type<float>(0x3FAC0532u),  //  1.3439085
    as_type<float>(0x4009B977u)   //  2.1519449
};

// 7 decision boundaries: midpoints between consecutive centroids,
// computed as (CB3[i] + CB3[i+1]) * 0.5f in single precision.
constant float BOUNDARIES_3[7] = {
    as_type<float>(0xBFDFBC10u),  // -1.7479267
    as_type<float>(0xBF8664FBu),  // -1.0499567
    as_type<float>(0xBF002401u),  // -0.5005494
    as_type<float>(0x00000000u),  //  0.0000000
    as_type<float>(0x3F002401u),  //  0.5005494
    as_type<float>(0x3F8664FBu),  //  1.0499567
    as_type<float>(0x3FDFBC10u)   //  1.7479267
};
";

/// MSL body for `rmlx_tq3_quantize`.
///
/// Grid: `(N_groups * 32, 1, 1)`. Threadgroup: `(32, 1, 1)`.
///
/// Pack format: 3 `u32` per group of 32 elements, LSB-first across the
/// 96-bit concatenated stream. Element `e` occupies bits `[e*3 .. e*3+3)`.
///
/// Inputs:
/// - `inp` f32 `[N_groups * 32]` — flat input elements.
///
/// Outputs:
/// - `codes` u32 `[N_groups * 3]`
/// - `scales` f32 `[N_groups]` — per-group scale = max(|x|) / CB3_MAX.
const V3_QUANTIZE_SOURCE: &str = r"
    uint group_id = threadgroup_position_in_grid.x;
    uint elem     = thread_position_in_threadgroup.x;

 // -- Step 1: load group into threadgroup shared memory -----------------
    threadgroup float shared_x[32];
    shared_x[elem] = inp[group_id * 32u + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // -- Step 2: thread 0 finds max(|x|) and writes scale ------------------
 // Sequential scan by thread 0 mirrors the CPU path exactly (CPU-parity).
    threadgroup float group_scale[1];
    if (elem == 0u) {
        float abs_max = 0.0f;
        for (uint i = 0u; i < 32u; i++) {
            float a = abs(shared_x[i]);
            if (a > abs_max) { abs_max = a; }
        }
 // Lloyd-Max N(0,1) 3-bit max centroid.
        const float cb3_max = 2.1519449f;
        group_scale[0] = (abs_max > 0.0f) ? (abs_max / cb3_max) : 0.0f;
        scales[group_id] = group_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = group_scale[0];

 // -- Step 3: each thread finds its nearest-centroid index --------------
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    float normalized = shared_x[elem] * inv_scale;

    uint idx = 0u;
    for (uint b = 0u; b < 7u; b++) {
        if (normalized > BOUNDARIES_3[b]) {
            idx++;
        }
    }

 // -- Step 4: pack 32 x 3-bit indices into 3 x u32 words ----------------
 // Element e occupies bits [e*3, e*3+3) of the concatenated 96-bit
 // little-endian stream. Word w (w in 0..3) carries bits [w*32, w*32+32).
 // We use 3 writer threads (elem 0, 11, 22) — each scans the 32 element
 // indices, computes signed bit-shift offsets, and ORs the contribution
 // into a 64-bit accumulator.
    threadgroup uint idx_shared[32];
    idx_shared[elem] = idx & 0x7u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool is_writer = (elem == 0u) || (elem == 11u) || (elem == 22u);
    if (is_writer) {
        uint word_off;
        if (elem == 0u)       { word_off = 0u; }
        else if (elem == 11u) { word_off = 1u; }
        else                  { word_off = 2u; }

 // Accumulate into a 64-bit local register so a 3-bit index that
 // straddles the 32-bit word boundary is captured exactly.
        ulong acc = 0ul;
        for (uint e = 0u; e < 32u; e++) {
            int shift = (int)(e * 3u) - (int)(word_off * 32u);
 // Element e's 3-bit window is [shift, shift+3) in word coords.
 // It touches this word iff that window overlaps [0, 32), i.e.
 // shift in [-2, 31].
            if (shift > -3 && shift < 32) {
                ulong bits3 = (ulong)(idx_shared[e] & 0x7u);
                if (shift >= 0) {
                    acc |= (bits3 << (uint)shift);
                } else {
 // shift in {-1, -2}: the low (3 + shift) bits of the
 // element spill into this word, starting at bit 0.
                    acc |= (bits3 >> (uint)(-shift));
                }
            }
        }
        codes[group_id * 3u + word_off] = (uint)(acc & 0xFFFFFFFFul);
    }
";

/// MSL body for `rmlx_tq3_dequantize`.
///
/// Grid: `(N_groups * 32, 1, 1)`. Threadgroup: `(32, 1, 1)`.
///
/// Each thread decodes one output element. Threads 0..2 load the 3 `u32`
/// words into threadgroup shared memory, then every thread extracts its own
/// 3-bit index via a signed-shift across the word boundary (the same
/// arithmetic the CPU `unpack_index` uses across byte boundaries, reframed
/// to 32-bit words).
///
/// Inputs:
/// - `codes` u32 `[N_groups * 3]`
/// - `scales` f32 `[N_groups]`
///
/// Outputs:
/// - `out` OutT `[N_groups * 32]`
const V3_DEQUANTIZE_SOURCE: &str = r"
    uint group_id = threadgroup_position_in_grid.x;
    uint elem     = thread_position_in_threadgroup.x;

    threadgroup uint words[3];
    if (elem < 3u) {
        words[elem] = codes[group_id * 3u + elem];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // Bits [elem*3, elem*3+3) of the concatenated 96-bit stream.
    uint bit_off  = elem * 3u;
    uint word0_id = bit_off / 32u;       // 0, 1 or 2
    uint shift0   = bit_off - word0_id * 32u;
    ulong window  = (ulong)words[word0_id];
    if (word0_id + 1u < 3u) {
        window |= ((ulong)words[word0_id + 1u]) << 32;
    }
    uint idx = (uint)((window >> shift0) & 0x7ul);

    float scale = scales[group_id];
    float v     = CB3[idx] * scale;
    out[group_id * 32u + elem] = static_cast<OutT>(v);
";

// -- Kernel singletons -------------------------------------------------------

static V3_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static V3_DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn v3_quant_kernel() -> Result<&'static MetalKernel> {
    V3_QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq3_quantize",
                V3_KERNEL_HEADER,
                V3_QUANTIZE_SOURCE,
                &["inp"],
                &["codes", "scales"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tq3_quantize kernel init: {e}")))
}

fn v3_dequant_kernel() -> Result<&'static MetalKernel> {
    V3_DEQUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_tq3_dequantize",
                V3_KERNEL_HEADER,
                V3_DEQUANTIZE_SOURCE,
                &["codes", "scales"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("tq3_dequantize kernel init: {e}")))
}

// -- Public API --------------------------------------------------------------

/// GPU TurboQuant V3 quantize.
///
/// Quantize `x` (any shape, total elements must be a multiple of
/// [`GROUP_SIZE`] = 32) using the Lloyd-Max N(0,1) 3-bit codebook.
///
/// Returns `(codes, scales)`:
/// - `codes`: `u32` array of shape `[total_elems * 3 / 32]` — 3 u32 per
///   group of 32 elements, LSB-first across a 96-bit per-group stream.
/// - `scales`: `f32` array of shape `[total_elems / 32]` — one scale per
///   group.
pub fn turbo_quantize_v3_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    let shape = x.shape();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_quantize_v3_gpu: total elements {total_elems} not a multiple \
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

    let kernel = v3_quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&x_f32)?;
    invoke.add_output_shape(&[(n_groups * 3) as i32], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups as i32], Dtype::F32)?;

    invoke.set_grid((n_groups * GROUP_SIZE) as i32, 1, 1)?;
    invoke.set_thread_group(GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 2 {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_quantize_v3_gpu: expected 2 outputs".to_owned(),
        ));
    }
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    tracing::trace!(n_groups, total_elems, "tq3 quantize dispatched");
    Ok((codes, scales))
}

/// GPU TurboQuant V3 dequantize.
///
/// Reconstruct an `out_dtype` tensor from `(codes, scales)` produced by
/// [`turbo_quantize_v3_gpu`].
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn turbo_dequantize_v3_gpu(
    codes: &Array,
    scales: &Array,
    original_shape: &[i32],
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    let n_words_total: usize = codes.shape().iter().map(|&d| d as usize).product();
    if !n_words_total.is_multiple_of(3) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v3_gpu: codes len {n_words_total} not a multiple of 3"
        )));
    }
    let n_groups_codes = n_words_total / 3;
    let n_groups_scales: usize = scales.shape().iter().map(|&d| d as usize).product();
    if n_groups_codes != n_groups_scales {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v3_gpu: n_groups from codes ({n_groups_codes}) != \
             from scales ({n_groups_scales})"
        )));
    }
    let n_groups = n_groups_codes;
    let total_elems = n_groups * GROUP_SIZE;

    let shape_product: usize = original_shape.iter().map(|&d| d as usize).product();
    if shape_product != total_elems {
        return Err(rmlx_core::error::Error::Quant(format!(
            "turbo_dequantize_v3_gpu: original_shape product {shape_product} != \
             expected {total_elems}"
        )));
    }

    let codes_flat = codes.reshape(&[(n_groups * 3) as i32], device)?;
    let scales_flat = scales.reshape(&[n_groups as i32], device)?;

    let kernel_out_dtype = match out_dtype {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => out_dtype,
        _ => Dtype::F32,
    };

    let kernel = v3_dequant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_output_shape(&[total_elems as i32], kernel_out_dtype)?;
    invoke.set_template_dtype("OutT", kernel_out_dtype)?;

    invoke.set_grid((n_groups * GROUP_SIZE) as i32, 1, 1)?;
    invoke.set_thread_group(GROUP_SIZE as i32, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "turbo_dequantize_v3_gpu: expected 1 output".to_owned(),
        ));
    }
    let out_flat = outputs.remove(0);
    let out_flat = if kernel_out_dtype == out_dtype {
        out_flat
    } else {
        out_flat.astype(out_dtype, device)?
    };

    tracing::trace!(n_groups, total_elems, "tq3 dequantize dispatched");

    if original_shape.len() == 1 && original_shape[0] == total_elems as i32 {
        return Ok(out_flat);
    }
    out_flat.reshape(original_shape, device)
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
#[path = "k8vturbo3_append_msl_tests.rs"]
mod tests;
