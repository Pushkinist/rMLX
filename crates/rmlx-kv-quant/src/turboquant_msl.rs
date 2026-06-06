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
const KERNEL_HEADER: &str = r"
// 4-bit TurboQuant codebook: 16 Lloyd-Max optimal N(0,1) centroids.
// Derived by turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100).
// Replaces prior N(0,1) quantile centroids.
// Bit patterns match turboquant.rs::CODEBOOK_4BIT.
constant float CB[16] = {
    as_type<float>(0xC02DEE42u),  // -2.7176671
    as_type<float>(0xC003563Bu),  // -2.0521381
    as_type<float>(0xBFCCE718u),  // -1.6008024
    as_type<float>(0xBF9EB6FAu),  // -1.2399590
    as_type<float>(0xBF6DA172u),  // -0.9282447
    as_type<float>(0xBF255816u),  // -0.6458753
    as_type<float>(0xBEC329CBu),  // -0.3811782
    as_type<float>(0xBE011273u),  // -0.1260469
    as_type<float>(0x3E011273u),  //  0.1260469
    as_type<float>(0x3EC329CBu),  //  0.3811782
    as_type<float>(0x3F255816u),  //  0.6458753
    as_type<float>(0x3F6DA172u),  //  0.9282447
    as_type<float>(0x3F9EB6FAu),  //  1.2399590
    as_type<float>(0x3FCCE718u),  //  1.6008024
    as_type<float>(0x4003563Bu),  //  2.0521381
    as_type<float>(0x402DEE42u)   //  2.7176671
};

// 15 decision boundaries: midpoints between consecutive Lloyd-Max centroids,
// computed as (CB[i] + CB[i+1]) * 0.5f in single precision.
// Bit patterns match what the CPU turboquant.rs::nearest_centroid computes
// at runtime using the same formula.
constant float BOUNDARIES[15] = {
    as_type<float>(0xC018A23Eu),  // -2.3849025
    as_type<float>(0xBFE9C9C7u),  // -1.8264703
    as_type<float>(0xBFB5CF09u),  // -1.4203807
    as_type<float>(0xBF8AC3DAu),  // -1.0841019
    as_type<float>(0xBF497CC4u),  // -0.7870600
    as_type<float>(0xBF03767Eu),  // -0.5135268
    as_type<float>(0xBE81D982u),  // -0.2536126
    as_type<float>(0x00000000u),  //  0.0000000
    as_type<float>(0x3E81D982u),  //  0.2536126
    as_type<float>(0x3F03767Eu),  //  0.5135268
    as_type<float>(0x3F497CC4u),  //  0.7870600
    as_type<float>(0x3F8AC3DAu),  //  1.0841019
    as_type<float>(0x3FB5CF09u),  //  1.4203807
    as_type<float>(0x3FE9C9C7u),  //  1.8264703
    as_type<float>(0x4018A23Eu)   //  2.3849025
};
";

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
const QUANTIZE_SOURCE: &str = r"
 // Each threadgroup handles one group of 32 elements.
 // threadgroup_position_in_grid.x = group index (0 .. N_groups-1).
 // thread_position_in_threadgroup.x = element within group (0 .. 31).
    uint group_id = threadgroup_position_in_grid.x;
    uint elem     = thread_position_in_threadgroup.x;

 // ── Step 1: load group into threadgroup shared memory ──────────────────
    threadgroup float shared[32];
    shared[elem] = inp[group_id * 32u + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // ── Step 2: thread 0 finds max(|x|) and writes scale ──────────────────
 // Sequential scan by thread 0 mirrors the CPU path exactly.
 // scale = max(|x_i|) / CB_MAX where CB_MAX = 2.7176671 (Lloyd-Max N(0,1) 4-bit).
    threadgroup float group_scale[1];
    if (elem == 0u) {
        float abs_max = 0.0f;
        for (uint i = 0u; i < 32u; i++) {
            float a = abs(shared[i]);
            if (a > abs_max) { abs_max = a; }
        }
 // Lloyd-Max N(0,1) 4-bit max centroid 2.7176671 (replaces prior 2.7326 quantile centroid).
        const float cb_max = 2.7176671f;
        group_scale[0] = (abs_max > 0.0f) ? (abs_max / cb_max) : 0.0f;
        scales[group_id] = group_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = group_scale[0];

 // ── Step 3: each thread finds nearest centroid index ───────────────────
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    float normalized = shared[elem] * inv_scale;

    uint idx = 0u;
    for (uint b = 0u; b < 15u; b++) {
        if (normalized > BOUNDARIES[b]) {
            idx++;
        }
    }

 // ── Step 4: pack 8 indices per uint32 word ─────────────────────────────
 // word 0 holds elements 0..7, word 1 holds elements 8..15, etc.
 // Within each word: element e occupies bits [e_in_word*4 .. e_in_word*4+3].
    threadgroup uint idx_shared[32];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // Threads 0, 8, 16, 24 each pack one 32-bit word (8 elements × 4 bits).
    if (elem % 8u == 0u) {
        uint word_idx = group_id * 4u + (elem / 8u);
        uint word = 0u;
        for (uint i = 0u; i < 8u; i++) {
            word |= (idx_shared[elem + i] & 0xFu) << (i * 4u);
        }
        codes[word_idx] = word;
    }
";

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
const DEQUANTIZE_SOURCE: &str = r"
 // gid = word index (0 .. N_groups*4-1).
    uint gid      = thread_position_in_grid.x;
    uint group_id = gid / 4u;   // which group of 32 elements
    uint word_off = gid % 4u;   // word within group (0..3), 8 elements each

    uint word  = codes[gid];
    float scale = scales[group_id];

 // Base output index for this word's 8 elements.
    uint base_out = group_id * 32u + word_off * 8u;

 // Dual-LUT: decode each of the 4 bytes in the word, 2 nibbles per byte.
 // Equivalent to a 256-entry half2 LUT: CB_LUT[b] = (CB[b & 0xF], CB[b >> 4]).
 // CB is in Metal constant memory — repeated lookups hit L1 after first use.
    for (uint byte_idx = 0u; byte_idx < 4u; byte_idx++) {
        uint b   = (word >> (byte_idx * 8u)) & 0xFFu;
        uint lo  = b & 0xFu;
        uint hi  = b >> 4u;
        float v0 = CB[lo] * scale;
        float v1 = CB[hi] * scale;
        out[base_out + byte_idx * 2u    ] = static_cast<OutT>(v0);
        out[base_out + byte_idx * 2u + 1] = static_cast<OutT>(v1);
    }
";

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
const QUANTIZE_CB_BUF_SOURCE: &str = r"
 // Each threadgroup handles one group of 32 elements.
    uint group_id = threadgroup_position_in_grid.x;
    uint elem     = thread_position_in_threadgroup.x;

 // ── Step 0: cache the 16-entry codebook in threadgroup memory ──────────
 // First 16 threads load one centroid each; the rest no-op the load.
    threadgroup float cb_shared[16];
    if (elem < 16u) {
        cb_shared[elem] = cb[elem];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // ── Step 1: load group into threadgroup shared memory ──────────────────
    threadgroup float shared[32];
    shared[elem] = inp[group_id * 32u + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // ── Step 2: thread 0 finds cb_max + group scale ────────────────────────
 // cb_max = max(|cb[i]|) over the 16 entries; scale = max(|x|) / cb_max.
    threadgroup float group_scale[1];
    if (elem == 0u) {
        float cb_max = 0.0f;
        for (uint i = 0u; i < 16u; i++) {
            float a = abs(cb_shared[i]);
            if (a > cb_max) { cb_max = a; }
        }
        float abs_max = 0.0f;
        for (uint i = 0u; i < 32u; i++) {
            float a = abs(shared[i]);
            if (a > abs_max) { abs_max = a; }
        }
        group_scale[0] = (abs_max > 0.0f && cb_max > 0.0f) ? (abs_max / cb_max) : 0.0f;
        scales[group_id] = group_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = group_scale[0];

 // ── Step 3: each thread finds nearest centroid via 15 runtime boundaries
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    float normalized = shared[elem] * inv_scale;

    uint idx = 0u;
    for (uint b = 0u; b < 15u; b++) {
        float boundary = (cb_shared[b] + cb_shared[b + 1u]) * 0.5f;
        if (normalized > boundary) {
            idx++;
        }
    }

 // ── Step 4: pack 8 indices per uint32 word (same layout as hardwired) ──
    threadgroup uint idx_shared[32];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (elem % 8u == 0u) {
        uint word_idx = group_id * 4u + (elem / 8u);
        uint word = 0u;
        for (uint i = 0u; i < 8u; i++) {
            word |= (idx_shared[elem + i] & 0xFu) << (i * 4u);
        }
        codes[word_idx] = word;
    }
";

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
const DEQUANTIZE_CB_BUF_SOURCE: &str = r"
    uint gid      = thread_position_in_grid.x;
    uint group_id = gid / 4u;
    uint word_off = gid % 4u;
    uint elem_in_grp = thread_position_in_threadgroup.x;

 // Cache the 16-entry codebook in threadgroup memory (4 threads per group:
 // each loads 4 entries to cover [0..16)). Guard on `elem_in_grp < 4u` so
 // any future threadgroup-size bump cannot OOB-write past cb_shared[15]
 // (mirrors the `elem < 16u` guard pattern in QUANTIZE_CB_BUF_SOURCE).
    threadgroup float cb_shared[16];
    if (elem_in_grp < 4u) {
        uint base = elem_in_grp * 4u;
        cb_shared[base + 0u] = cb[base + 0u];
        cb_shared[base + 1u] = cb[base + 1u];
        cb_shared[base + 2u] = cb[base + 2u];
        cb_shared[base + 3u] = cb[base + 3u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint word  = codes[gid];
    float scale = scales[group_id];

    uint base_out = group_id * 32u + word_off * 8u;

    for (uint byte_idx = 0u; byte_idx < 4u; byte_idx++) {
        uint b   = (word >> (byte_idx * 8u)) & 0xFFu;
        uint lo  = b & 0xFu;
        uint hi  = b >> 4u;
        float v0 = cb_shared[lo] * scale;
        float v1 = cb_shared[hi] * scale;
        out[base_out + byte_idx * 2u    ] = static_cast<OutT>(v0);
        out[base_out + byte_idx * 2u + 1] = static_cast<OutT>(v1);
    }
";

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
