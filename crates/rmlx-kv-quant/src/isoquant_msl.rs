//! iso3 MSL (Metal Shading Language) GPU kernels.
//!
//! # Status
//!
//! Kernel hook landed. GPU execution leg is `#[ignore]`-gated. CPU path
//! in `isoquant.rs` remains primary until T11e perf gates are evaluated.
//!
//! # Algorithm
//!
//! For each token × group (one thread per group):
//!   1. Load `ISO3_GS=4` elements.
//!   2. Compute the per-token L2 norm by summing over all groups (redundant per
//!      thread; correct by construction — each thread reads all `head_dim` elems).
//!   3. Apply Hamilton product `r = q_L * v_normalised` where `q_L` is the fixed
//!      golden-ratio unit quaternion [`FIXED_QUAT`].
//!   4. Per-group scale = `max|r_i| / ISO3_CB_MAX`.
//!   5. 3-bit quantize each `r_i` into the Lloyd-Max codebook.
//!   6. Pack 3-bit codes via atomic OR (10 vals/u32, Planar3 pack convention).
//!
//! Dequantize reverses: unpack → centroid × scale → conjugate Hamilton product
//! → rescale by per-token norm.
//!
//! # Group size
//!
//! Fixed at `ISO3_GS = 4` (one quaternion block per group). This matches
//! [`crate::storage::quant_iso_v::ISO3_GROUP_SIZE`] and the CPU path.
//!
//! # Pack format
//!
//! 10 × 3-bit values per u32 (30 bits used, 2 wasted). For `ISO3_GS=4`:
//! 1 u32 per group (4 × 3 = 12 bits ≤ 30). Element `e` within a group maps to
//! `word = e / 10`, `shift = (e % 10) * 3`. Mirrors Planar3 convention.
//!
//! # Codebook source
//!
//! `lloyd_gaussian_codebook(3)` — Lloyd-Max N(0,1), 8 centroids.
//! Bit patterns match `turboquant.rs::CODEBOOK_3BIT`.
//! Derived by `turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100)`.
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
/// Matches [`crate::storage::quant_iso_v::ISO3_GROUP_SIZE`].
const ISO3_GROUP_SIZE: usize = 4;

/// 3-bit values per u32 word (30 bits used, 2 wasted). Planar3 convention.
const VALS_PER_WORD: usize = 10;

/// Words per group for `ISO3_GROUP_SIZE=4`: ceil(4/10) = 1.
const WORDS_PER_GROUP: usize = ISO3_GROUP_SIZE.div_ceil(VALS_PER_WORD);

/// Public re-export of [`WORDS_PER_GROUP`] for sibling modules that size
/// per-step GPU buffers off the iso3 codes layout (e.g. the GPU-resident mirror
/// in `storage::quant_iso_v::QuantIsoV3::append_gpu`).
pub const ISO3_WORDS_PER_GROUP: usize = WORDS_PER_GROUP;

// ── MSL header builder ────────────────────────────────────────────────────────

/// Generate the MSL header string for the iso3 kernels.
///
/// Embeds:
/// - The fixed golden-ratio unit quaternion components (and conjugate).
/// - The 3-bit Lloyd-Max N(0,1) codebook (8 entries) and 7 decision boundaries.
/// - `ISO3_CB_MAX`, `ISO3_GS` (group size), `ISO3_WPG` (words per group) constants.
///
/// Constants are generated at runtime so that any change to `FIXED_QUAT` or
/// `lloyd_gaussian_codebook(3)` is automatically reflected in the MSL kernels.
#[allow(
    clippy::expect_used,
    reason = "lloyd_gaussian_codebook(3) is always Ok (3-bit is a registered codebook); \
              codebook slice is non-empty by construction (8 entries). Both invariants are \
              verified at dev-time by debug_assert_eq! below and by the \
              `turbo_codebook_3bit_has_8_entries_monotonic` test in turboquant_tests.rs."
)]
fn build_msl_header_iso3() -> String {
    // Fixed quaternion [w, x, y, z] — golden-ratio unit quat.
    // Source: multi_turboquant/methods/isoquant.py; bit-exact with isoquant.rs::FIXED_QUAT.
    let [qw, qx, qy, qz] = FIXED_QUAT;
    let qw_bits = f32::to_bits(qw);
    let qx_bits = f32::to_bits(qx);
    let qy_bits = f32::to_bits(qy);
    let qz_bits = f32::to_bits(qz);
    // Conjugate: (w, −x, −y, −z).
    let qcx_bits = f32::to_bits(-qx);
    let qcy_bits = f32::to_bits(-qy);
    let qcz_bits = f32::to_bits(-qz);

    // 3-bit Lloyd-Max N(0,1) codebook — 8 entries.
    // Bit patterns match turboquant.rs::CODEBOOK_3BIT (canonical source of truth).
    // Derived by turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100).
    let cb3 = lloyd_gaussian_codebook(3).expect("3-bit codebook must exist");
    // dev-time guard: 8 entries is a structural invariant of the 3-bit codebook.
    // Release correctness enforced by `turbo_codebook_3bit_has_8_entries_monotonic`
    // and `lloyd_gaussian_codebook_3bit_entries_are_finite` in turboquant_tests.rs.
    debug_assert_eq!(cb3.len(), 8, "iso3 MSL: 3-bit codebook must have 8 entries");

    let cb_max: f32 = *cb3
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
        .expect("non-empty codebook");
    let cb_max_bits = f32::to_bits(cb_max);

    let cb_hex: Vec<String> = cb3
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    #[allow(
        clippy::indexing_slicing,
        reason = "windows(2) always produces slices of length 2; indices 0 and 1 are always in-bounds"
    )]
    let boundaries: Vec<f32> = cb3.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    debug_assert_eq!(
        boundaries.len(),
        7,
        "iso3 MSL: 7 decision boundaries expected"
    );
    let bound_hex: Vec<String> = boundaries
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mut s = String::new();

    // Fixed quaternion q_L and conjugate q̄_L.
    let _ = write!(
        s,
        "\n// iso3 fixed quaternion q_L = [w, x, y, z] (golden-ratio unit quat).\n\
         // Source: multi_turboquant/methods/isoquant.py; bit-exact with isoquant.rs::FIXED_QUAT.\n\
         constant float ISO3_QW = as_type<float>(0x{qw_bits:08X}u);\n\
         constant float ISO3_QX = as_type<float>(0x{qx_bits:08X}u);\n\
         constant float ISO3_QY = as_type<float>(0x{qy_bits:08X}u);\n\
         constant float ISO3_QZ = as_type<float>(0x{qz_bits:08X}u);\n\
         // Conjugate q̄_L = (w, -x, -y, -z) for dequantize (inverse rotation).\n\
         // q̄_L.w == q_L.w, so ISO3_QW is reused.\n\
         constant float ISO3_CX = as_type<float>(0x{qcx_bits:08X}u);\n\
         constant float ISO3_CY = as_type<float>(0x{qcy_bits:08X}u);\n\
         constant float ISO3_CZ = as_type<float>(0x{qcz_bits:08X}u);\n"
    );

    // 3-bit Lloyd-Max codebook and boundaries.
    let _ = write!(
        s,
        "\n// 3-bit Lloyd-Max N(0,1) codebook — 8 entries.\n\
         // Bit patterns match turboquant.rs::CODEBOOK_3BIT.\n\
         constant float ISO3_CB[8] = {{\n{cb}\n}};\n",
        cb = cb_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\n// 7 midpoint decision boundaries for the 3-bit codebook.\n\
         constant float ISO3_BOUNDS[7] = {{\n{b}\n}};\n",
        b = bound_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\nconstant float ISO3_CB_MAX = as_type<float>(0x{cb_max_bits:08X}u);\n\
         // Quaternion-block group size (4 elements per group).\n\
         constant uint  ISO3_GS = {ISO3_GROUP_SIZE}u;\n\
         // 3-bit values per u32 word (10 vals, 30 bits used).\n\
         constant uint  ISO3_VPW = {VALS_PER_WORD}u;\n\
         // u32 words per group = ceil(ISO3_GS / ISO3_VPW) = 1 for GS=4.\n\
         constant uint  ISO3_WPG = {WORDS_PER_GROUP}u;\n"
    );
    s
}

// ── MSL kernel sources ────────────────────────────────────────────────────────
//
// Quantize kernel: one thread per (token, group) pair.
// Grid = n_tokens * n_groups flat threads.
//
// Thread index `gid`:
//   token  = gid / n_groups
//   grp    = gid % n_groups
//   head_dim = n_groups * ISO3_GS
//
// The per-token L2 norm is recomputed redundantly by each group thread in the
// same token (all threads read the full token row). Acceptable for a hook
// implementation; T11e may introduce a two-pass approach if GPU time dominates.
//
// Outputs:
//   codes_out  : u32 [n_tokens * n_groups * ISO3_WPG] — 3-bit codes, 10/u32
//   scales_out : f32 [n_tokens * n_groups]             — per-group scale
//   norms_out  : f32 [n_tokens * n_groups]             — per-group (all same per token)
//
// norms_out is per-group (not per-token) to avoid concurrent writes; the
// dequantize kernel reads norms_out[group_global] instead of norms_out[token].

const QUANTIZE_SOURCE_ISO3: &str = r"
    uint gid     = thread_position_in_grid.x;
    uint n_groups_u = n_groups[0];
    uint token   = gid / n_groups_u;
    uint grp     = gid % n_groups_u;
    uint hd      = n_groups_u * ISO3_GS;    // head_dim

 // ── Compute per-token L2 norm (recomputed per group — hook impl) ──────────
    float norm_sq = 0.0f;
    for (uint i = 0u; i < hd; i++) {
        float vi = inp[token * hd + i];
        norm_sq += vi * vi;
    }
    float norm = sqrt(norm_sq);
    if (norm < 1e-8f) norm = 1e-8f;

    norms_out[gid] = norm;   // store per group slot

 // ── Load 4 elements, normalise, apply Hamilton product r = q_L * v ────────
    uint base = token * hd + grp * ISO3_GS;
    float vw = inp[base    ] / norm;
    float vx = inp[base + 1] / norm;
    float vy = inp[base + 2] / norm;
    float vz = inp[base + 3] / norm;

    // Hamilton product: r = q_L * v, [w,x,y,z] convention.
    float rw = ISO3_QW*vw - ISO3_QX*vx - ISO3_QY*vy - ISO3_QZ*vz;
    float rx = ISO3_QW*vx + ISO3_QX*vw + ISO3_QY*vz - ISO3_QZ*vy;
    float ry = ISO3_QW*vy - ISO3_QX*vz + ISO3_QY*vw + ISO3_QZ*vx;
    float rz = ISO3_QW*vz + ISO3_QX*vy - ISO3_QY*vx + ISO3_QZ*vw;

 // ── Per-group scale ────────────────────────────────────────────────────────
    float abs_max = max(max(abs(rw), abs(rx)), max(abs(ry), abs(rz)));
    float scale   = (abs_max < 1e-12f) ? 1e-12f : (abs_max / ISO3_CB_MAX);
    scales_out[gid] = scale;

 // ── 3-bit quantize (codebook lookup) and pack via atomic OR ───────────────
 // ISO3_GS=4 elements → 1 u32 per group (4*3=12 bits ≤ 30).
    uint code_word = gid * ISO3_WPG;   // = gid * 1 for ISO3_GS=4
    float rots[4];
    rots[0] = rw; rots[1] = rx; rots[2] = ry; rots[3] = rz;

    for (uint e = 0u; e < ISO3_GS; e++) {
        float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
        uint idx = 0u;
        for (uint bi = 0u; bi < 7u; bi++) {
            if (norm_val > ISO3_BOUNDS[bi]) idx++;
        }
        uint word  = code_word + e / ISO3_VPW;
        uint shift = (e % ISO3_VPW) * 3u;
        atomic_fetch_or_explicit((device atomic_uint*)&codes_out[word],
                                 (idx & 0x7u) << shift,
                                 memory_order_relaxed);
    }
";

// Dequantize kernel: one thread per (token, group) pair.
// Reverses the quantize kernel.
//
// Inputs:
//   codes_in  : u32 [n_tokens * n_groups * ISO3_WPG]
//   scales_in : f32 [n_tokens * n_groups]
//   norms_in  : f32 [n_tokens * n_groups]  (per-group slot, same value per token)

const DEQUANTIZE_SOURCE_ISO3: &str = r"
    uint gid        = thread_position_in_grid.x;
    uint n_groups_u = n_groups[0];
    uint token      = gid / n_groups_u;
    uint grp        = gid % n_groups_u;
    uint hd         = n_groups_u * ISO3_GS;

    float scale = scales_in[gid];
    float norm  = norms_in[gid];

    uint code_word = gid * ISO3_WPG;
    uint base_out  = token * hd + grp * ISO3_GS;

 // ── Unpack, dequantize, inverse-rotate, rescale ───────────────────────────
    float rots[4];
    for (uint e = 0u; e < ISO3_GS; e++) {
        uint word  = code_word + e / ISO3_VPW;
        uint shift = (e % ISO3_VPW) * 3u;
        uint idx   = (codes_in[word] >> shift) & 0x7u;
        rots[e]    = ISO3_CB[idx] * scale;
    }

    float rw = rots[0]; float rx = rots[1];
    float ry = rots[2]; float rz = rots[3];

    // Inverse rotation: q̄_L * r — Hamilton product with conjugate.
    // q̄_L = (ISO3_QW, ISO3_CX, ISO3_CY, ISO3_CZ).
    float vw = ISO3_QW*rw - ISO3_CX*rx - ISO3_CY*ry - ISO3_CZ*rz;
    float vx = ISO3_QW*rx + ISO3_CX*rw + ISO3_CY*rz - ISO3_CZ*ry;
    float vy = ISO3_QW*ry - ISO3_CX*rz + ISO3_CY*rw + ISO3_CZ*rx;
    float vz = ISO3_QW*rz + ISO3_CX*ry - ISO3_CY*rx + ISO3_CZ*rw;

    out[base_out    ] = vw * norm;
    out[base_out + 1] = vx * norm;
    out[base_out + 2] = vy * norm;
    out[base_out + 3] = vz * norm;
";

// ── Kernel singletons ─────────────────────────────────────────────────────────

static ISO3_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ISO3_DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ISO3_KERNEL_HEADER: OnceLock<String> = OnceLock::new();

fn kernel_header_iso3() -> &'static str {
    ISO3_KERNEL_HEADER.get_or_init(build_msl_header_iso3)
}

fn iso3_quant_kernel() -> Result<&'static MetalKernel> {
    ISO3_QUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_iso3_quantize",
                kernel_header_iso3(),
                QUANTIZE_SOURCE_ISO3,
                &["inp", "n_groups"],
                &["codes_out", "scales_out", "norms_out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("iso3_quantize kernel init: {e}")))
}

fn iso3_dequant_kernel() -> Result<&'static MetalKernel> {
    ISO3_DEQUANT_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_iso3_dequantize",
                kernel_header_iso3(),
                DEQUANTIZE_SOURCE_ISO3,
                &["codes_in", "scales_in", "norms_in", "n_groups"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("iso3_dequantize kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU iso3 quantize.
///
/// Quantize `v_full` (shape `[n_tokens, head_dim]` or any shape whose total
/// element count is `n_tokens * head_dim`) using quaternion SO(4) rotation +
/// 3-bit Lloyd-Max N(0,1) codebook.
///
/// `head_dim` must be a positive multiple of [`ISO3_GROUP_SIZE`] (= 4).
///
/// # Returns
///
/// `(codes_packed, scales, quaternions, norms)` where:
/// - `codes_packed`: `u32 [n_tokens * n_groups * WORDS_PER_GROUP]` — 3-bit
///   indices, 10 per word (Planar3 pack convention).
/// - `scales`: `f32 [n_tokens * n_groups]` — per-group scale.
/// - `quaternions`: `f32 [n_tokens * n_groups * 4]` — per-group unit quaternion.
///   All entries are the fixed [`FIXED_QUAT`] in this implementation.
/// - `norms`: `f32 [n_tokens * n_groups]` — per-group L2 norm slot (all slots
///   in the same token share the same value; stored per-group to avoid races).
///
/// # Errors
///
/// Returns `Error::Quant` for invalid `head_dim`.
/// Returns `Error::Mlx` if Metal kernel compilation fails.
pub fn iso_quantize_v3_gpu(
    v_full: &Array,
    head_dim: usize,
    device: Device,
) -> Result<(Array, Array, Array, Array)> {
    if head_dim == 0 || !head_dim.is_multiple_of(ISO3_GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_quantize_v3_gpu: head_dim={head_dim} must be a positive multiple of \
             ISO3_GROUP_SIZE={ISO3_GROUP_SIZE}"
        )));
    }

    let total_elems: usize = v_full.shape().iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(head_dim) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_quantize_v3_gpu: total elements {total_elems} not a multiple of \
             head_dim={head_dim}"
        )));
    }
    let n_tokens = total_elems / head_dim;
    let n_groups = head_dim / ISO3_GROUP_SIZE;
    let total_groups = n_tokens * n_groups;

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

    // Pass n_groups as a 1-element u32 array (scalar uniform).
    let n_groups_bytes = (n_groups as u32).to_le_bytes();
    let n_groups_arr = Array::from_bytes(&n_groups_bytes, &[1_i32], Dtype::U32)?;

    let kernel = iso3_quant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&v_f32)?;
    invoke.add_input(&n_groups_arr)?;

    // codes_out: u32 [total_groups * WORDS_PER_GROUP].
    invoke.add_output_shape(&[(total_groups * WORDS_PER_GROUP) as i32], Dtype::U32)?;
    // scales_out: f32 [total_groups].
    invoke.add_output_shape(&[total_groups as i32], Dtype::F32)?;
    // norms_out: f32 [total_groups] (per-group slot; same norm per token).
    invoke.add_output_shape(&[total_groups as i32], Dtype::F32)?;

    // Zero-initialise: codes_out written via atomic OR.
    invoke.set_init_value(0.0)?;

    // Grid: one thread per group. Threadgroup: 1.
    invoke.set_grid(total_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 3 {
        return Err(rmlx_core::error::Error::Mlx(
            "iso_quantize_v3_gpu: expected 3 outputs".to_owned(),
        ));
    }
    let norms = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);

    // Quaternions: all fixed — emit a constant array (total_groups * 4 entries).
    // T11e (per-group optimised quaternions) will replace this with per-group values.
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

/// GPU iso3 dequantize.
///
/// Reconstruct f32 tensor from `(codes_packed, scales, _quaternions, norms)`
/// produced by [`iso_quantize_v3_gpu`].
///
/// `head_dim` must match the encode call. `out_dtype` selects the element type
/// of the returned array.
///
/// # Errors
///
/// Returns `Error::Quant` for shape mismatches.
/// Returns `Error::Mlx` if Metal kernel compilation fails.
pub fn iso_dequantize_v3_gpu(
    codes_packed: &Array,
    scales: &Array,
    _quaternions: &Array,
    norms: &Array,
    head_dim: usize,
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    if head_dim == 0 || !head_dim.is_multiple_of(ISO3_GROUP_SIZE) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v3_gpu: head_dim={head_dim} must be a positive multiple of \
             ISO3_GROUP_SIZE={ISO3_GROUP_SIZE}"
        )));
    }

    // total_groups derived from norms (per-group storage, same as scales).
    let total_groups: usize = norms.shape().iter().map(|&d| d as usize).product();
    let n_groups = head_dim / ISO3_GROUP_SIZE;
    if total_groups == 0 || !total_groups.is_multiple_of(n_groups) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v3_gpu: norms length {total_groups} not a multiple of \
             n_groups={n_groups}"
        )));
    }
    let n_tokens = total_groups / n_groups;
    let total_elems = n_tokens * head_dim;

    let expected_codes = total_groups * WORDS_PER_GROUP;
    let codes_len: usize = codes_packed.shape().iter().map(|&d| d as usize).product();
    if codes_len != expected_codes {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v3_gpu: codes length {codes_len} != expected {expected_codes}"
        )));
    }
    let scales_len: usize = scales.shape().iter().map(|&d| d as usize).product();
    if scales_len != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso_dequantize_v3_gpu: scales length {scales_len} != total_groups {total_groups}"
        )));
    }

    let codes_flat = codes_packed.reshape(&[expected_codes as i32], device)?;
    let scales_flat = scales.reshape(&[total_groups as i32], device)?;
    let norms_flat = norms.reshape(&[total_groups as i32], device)?;

    let n_groups_bytes = (n_groups as u32).to_le_bytes();
    let n_groups_arr = Array::from_bytes(&n_groups_bytes, &[1_i32], Dtype::U32)?;

    let kernel = iso3_dequant_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&norms_flat)?;
    invoke.add_input(&n_groups_arr)?;
    invoke.add_output_shape(&[total_elems as i32], Dtype::F32)?;

    invoke.set_grid(total_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "iso_dequantize_v3_gpu: expected 1 output".to_owned(),
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

/// Read the GPU iso3 encode outputs back to host and return the four CPU
/// vectors used by [`crate::storage::quant_iso_v::IsoBlocks`].
///
/// Mirror of [`crate::isoquant_msl_v4::iso4_gpu_outputs_to_cpu`]; the
/// quaternion buffer is skipped (every group's quaternion is the same
/// [`FIXED_QUAT`] constant — emitted directly on the CPU side).
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
pub fn iso3_gpu_outputs_to_cpu(
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
            "iso3_gpu_outputs_to_cpu: n_tokens * n_groups must be > 0".to_owned(),
        ));
    }

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

    // FIXED_QUAT cycle; skip GPU readback (every group's quaternion is the same
    // constant FIXED_QUAT). The `_quaternions_gpu` Array is retained in the
    // encode-tuple return for ABI parity with the iso3/iso4 GPU functions; no
    // caller currently reads it.
    let mut quats = Vec::with_capacity(total_groups * 4);
    for _ in 0..total_groups {
        quats.extend_from_slice(&FIXED_QUAT);
    }

    let expected_codes = total_groups * WORDS_PER_GROUP;
    if codes.len() != expected_codes {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso3_gpu_outputs_to_cpu: codes len {} != expected {expected_codes}",
            codes.len()
        )));
    }
    if scales.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso3_gpu_outputs_to_cpu: scales len {} != expected {total_groups}",
            scales.len()
        )));
    }
    if norms_per_group.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "iso3_gpu_outputs_to_cpu: norms len {} != expected {total_groups}",
            norms_per_group.len()
        )));
    }

    // Deduplicate per-group norms to per-token (first slot of each token).
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
#[path = "isoquant_msl_tests.rs"]
mod isoquant_msl_tests;
