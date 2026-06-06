// LOC-exempt: ~1080 LOC. Holds four MSL kernel sources (rotor3 + rotor4
// encode/decode) plus the runtime codebook header builder, four kernel
// singletons, four public wrappers, and the GPU-output-to-CPU helper.
// Splitting rotor3 and rotor4 across two files (iso3/iso4 precedent —
// 545 + 637 LOC) would duplicate the codebook header builder + invoke
// scaffolding. The combined file shares those across bit-widths via the
// `bits: u8` parameter; one consolidated file is cheaper to maintain
// than two parallel forks of identical machinery.
//! rotor3 / rotor4 MSL (Metal Shading Language) GPU kernels.
//!
//! Sibling of [`crate::isoquant_msl`] / [`crate::isoquant_msl_v4`] but uses the
//! Cl(3,0) Clifford rotor sandwich `R * MV * R̃` instead of the SO(4) Hamilton
//! quaternion product. The rotor `R = [s, b12, b13, b23]` is **per-(layer,
//! head, group)** and supplied by the caller as an `n_groups * 4` f32 buffer
//! (see [`crate::clifford::make_rotor_table`]). Codebook = Lloyd-Max N(0, 1),
//! 8 entries for rotor3 (3-bit codes), 16 entries for rotor4 (4-bit codes).
//!
//! # Algorithm (per (token, group) thread)
//!
//! 1. Compute the per-token L2 norm by reading all `head_dim` elements of the
//!    token row (redundant per group — same as iso3/iso4 hook).
//! 2. Load 3 grade-1 elements `(v1, v2, v3)` from positions `grp*3 + {0,1,2}`
//!    in the token row, dividing by the norm. Tail-pad with 0 when
//!    `grp*3 + e >= head_dim` (CPU contract; mirrors
//!    [`crate::rotorquant::rotor3_encode`]).
//! 3. Apply the rotor sandwich `R * mv * R̃`. Because the input is grade-1
//!    only, the output is **also grade-1 only** (the scalar, bivector, and
//!    pseudoscalar components vanish identically — verified algebraically;
//!    see module documentation in `rotorquant.rs`). The sandwich reduces to
//!    a 3×3 SO(3) rotation matrix `M(R)` over `(v1, v2, v3)`:
//!
//!    ```text
//!    M[0][0] = s² - b12² - b13² + b23²
//!    M[0][1] = 2·s·b12 - 2·b13·b23
//!    M[0][2] = 2·s·b13 + 2·b12·b23
//!    M[1][0] = -2·s·b12 - 2·b23·b13
//!    M[1][1] = s² - b12² - b23² + b13²
//!    M[1][2] = 2·s·b23 - 2·b12·b13
//!    M[2][0] = -2·s·b13 + 2·b23·b12
//!    M[2][1] = -2·s·b23 - 2·b13·b12
//!    M[2][2] = s² - b13² - b23² + b12²
//!    ```
//!
//!    The full 8-component MV is `[0, R1, R2, R3, 0, 0, 0, 0]`.
//! 4. Per-group scale = `max(|R1|, |R2|, |R3|) / CB_MAX` (the 5 zero slots
//!    don't change the max).
//! 5. Quantize **all 8 components** against the Lloyd codebook to keep
//!    bit-identical layout with the CPU encoder. The 5 zero slots map to the
//!    centroid nearest 0; the 3 rotated slots get their codebook lookups.
//! 6. Pack 8 codes into 1 u32 via atomic OR — 3 bits/code (rotor3, 24 used)
//!    or 4 bits/code (rotor4, 32 used).
//!
//! Dequantize reverses:
//!   1. Unpack 8 codes per group → centroid × scale.
//!   2. Take only the grade-1 components (indices 1, 2, 3). The zero slots
//!      carry quantization noise but are discarded.
//!   3. Apply the inverse rotation `M(R)^T` (rotors are orthogonal in SO(3),
//!      so the transpose is the inverse — equivalent to the sandwich with
//!      the reversed rotor `R̃ = [s, -b12, -b13, -b23]`).
//!   4. Rescale by the stored per-token norm.
//!
//! # Pack convention
//!
//! 8 codes per group; one u32 per group for both bit-widths.
//!
//! | Codec  | bits | mask  | bits used per u32 |
//! |--------|------|-------|-------------------|
//! | rotor3 | 3    | 0x7   | 24 / 32 (8 wasted)|
//! | rotor4 | 4    | 0xF   | 32 / 32 (dense)   |
//!
//! Element `e ∈ 0..8` lands at bits `[e*bits .. e*bits + bits]` of word 0
//! within the group, matching [`crate::rotorquant::pack_group`] /
//! [`crate::rotorquant::pack_group_4bit`].
//!
//! # Per-(layer, head) rotor table
//!
//! Passed as a separate buffer arg `rotors_in : f32 [n_groups, 4]` per kernel
//! invocation. The caller (per-layer cache) owns the table lifetime. The
//! table is NOT hardcoded in MSL source.
//!
//! # K-side QJL fallback
//!
//! The K-side rotor codecs may carry a 1-bit QJL residual
//! correction that needs the JL projection matrix `S` at dequant time. The
//! GPU dequant kernels in this module do NOT implement QJL — when the K-side
//! caller has `qjl_s_matrix.is_some()` the dispatch falls back to the CPU
//! [`crate::rotorquant::rotor3_k_decode`] / [`crate::rotorquant::rotor4_k_decode`]
//! path. The V-side and the QJL-disabled K-side run on GPU.
//!
//! # Scale-norm parity caveat
//!
//! GPU computes `scale = max(|R1|,|R2|,|R3|) / CB_MAX` using the analytic
//! SO(3) shortcut. CPU computes max-abs over all 8 `rotor_sandwich` output
//! components — the 5 zero slots carry f32 noise from the dense GP. Scales
//! and norms agree with CPU within (1e-5 abs + 1e-4 rel) per element; the
//! ULP-level scale difference is enough to flip ~4–5% of sub-codes at
//! codebook boundaries, but every such flip is bounded to ±1 index step.
//! The HIGH-1 parity tests assert ≥95% sub-code agreement, all disagreements
//! being one-step boundary slips, plus tight scale + norm tolerance. A real
//! sign error in `M(R)` or a packing-order bug would produce multi-step
//! jumps and push agreement well below 70%.
//!
//! # GPU dequant kernels — production wiring
//!
//! The `rotor_dequantize_v{3,4}_gpu` entry points are kept public for parity
//! tests and for future GPU dequant wire-up. Production decode currently goes
//! through the per-storage `vs.dequant()` / `ks.dequant()` CPU paths (rotor
//! sandwich + per-token rescale runs at f32 on the CPU). The GPU dequant
//! kernels are intentionally not cfg(test)-gated for future adoption.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md, callers must hold `/tmp/rmlx.<port>.claim` before
//! dispatching.

use std::fmt::Write as _;
use std::sync::OnceLock;

use crate::rotorquant::{
    ROTOR3_BITS, ROTOR3_GROUP_SIZE, ROTOR3_MV_COMPONENTS, ROTOR3_VALS_PER_WORD,
    ROTOR3_WORDS_PER_GROUP, ROTOR4_BITS, ROTOR4_VALS_PER_WORD, ROTOR4_WORDS_PER_GROUP,
};
use crate::turboquant::lloyd_gaussian_codebook;
use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Constants ─────────────────────────────────────────────────────────────────

const _: () = assert!(
    ROTOR3_MV_COMPONENTS == 8,
    "rotor MSL kernel assumes 8 MV components (Cl(3,0) basis size)"
);
const _: () = assert!(
    ROTOR3_GROUP_SIZE == 3,
    "rotor MSL kernel assumes 3 grade-1 components per group"
);
const _: () = assert!(
    ROTOR3_WORDS_PER_GROUP == 1,
    "rotor3 MSL pack: 8 codes × 3 bits = 24 bits ≤ 30 → 1 u32 per group"
);
const _: () = assert!(
    ROTOR4_WORDS_PER_GROUP == 1,
    "rotor4 MSL pack: 8 codes × 4 bits = 32 bits → 1 u32 per group (dense)"
);

// ── MSL header builder ────────────────────────────────────────────────────────

/// Generate the MSL header string for the rotor `bits`-bit kernels.
///
/// Embeds the Lloyd-Max N(0, 1) codebook (`2^bits` entries), `2^bits - 1`
/// decision boundaries, the per-codebook max-abs centroid, and group-size
/// constants. Bit-width-specific constant names (`ROTOR3_*` vs `ROTOR4_*`)
/// keep the two kernel sources independent.
///
/// # Errors
///
/// Returns `Error::Mlx` if the Lloyd-Max codebook lookup fails. A panic in
/// the `OnceLock` init closure would brick all future rotor dispatch, so
/// callers must propagate the error rather than `expect`.
fn build_msl_header(bits: u8) -> Result<String> {
    let prefix = match bits {
        ROTOR3_BITS => "ROTOR3",
        ROTOR4_BITS => "ROTOR4",
        other => {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "rotor MSL header: unsupported bit-width {other} (expected 3 or 4)"
            )))
        }
    };
    let n_centroids: usize = 1usize << bits;
    let n_bounds: usize = n_centroids - 1;
    let vals_per_word: usize = if bits == ROTOR3_BITS {
        ROTOR3_VALS_PER_WORD
    } else {
        ROTOR4_VALS_PER_WORD
    };

    let cb = lloyd_gaussian_codebook(bits).map_err(|e| {
        rmlx_core::error::Error::Mlx(format!(
            "rotor MSL header: lloyd_gaussian_codebook({bits}): {e}"
        ))
    })?;
    if cb.len() != n_centroids {
        return Err(rmlx_core::error::Error::Mlx(format!(
            "rotor MSL header: codebook({bits}) length {} != expected {n_centroids}",
            cb.len()
        )));
    }

    let cb_max: f32 = *cb
        .iter()
        .max_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| {
            rmlx_core::error::Error::Mlx(format!("rotor MSL header: {bits}-bit codebook is empty"))
        })?;
    let cb_max_bits = f32::to_bits(cb_max);

    let cb_hex: Vec<String> = cb
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    // windows(2) yields slices of len 2; both indices are in-bounds by contract.
    let boundaries: Vec<f32> = cb
        .windows(2)
        .map(|w| {
            // SAFETY: windows(2) length-2 invariant; w[0]+w[1] over Lloyd centroids.
            #[allow(
                clippy::indexing_slicing,
                reason = "windows(2) yields slices of length 2; indices 0 and 1 always in-bounds"
            )]
            {
                (w[0] + w[1]) * 0.5
            }
        })
        .collect();
    if boundaries.len() != n_bounds {
        return Err(rmlx_core::error::Error::Mlx(format!(
            "rotor MSL header: {bits}-bit boundaries length {} != expected {n_bounds}",
            boundaries.len()
        )));
    }
    let bound_hex: Vec<String> = boundaries
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mv_components: usize = ROTOR3_MV_COMPONENTS;
    let group_size: usize = ROTOR3_GROUP_SIZE;
    let mut s = String::new();

    let _ = write!(
        s,
        "\n// {bits}-bit Lloyd-Max N(0,1) codebook — {n_centroids} entries.\n\
         // Bit patterns match turboquant.rs::CODEBOOK_{bits}BIT.\n\
         constant float {prefix}_CB[{n_centroids}] = {{\n{cb_v}\n}};\n",
        cb_v = cb_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\n// {n_bounds} midpoint decision boundaries for the {bits}-bit codebook.\n\
         constant float {prefix}_BOUNDS[{n_bounds}] = {{\n{b}\n}};\n",
        b = bound_hex.join(",\n")
    );
    let _ = write!(
        s,
        "\nconstant float {prefix}_CB_MAX = as_type<float>(0x{cb_max_bits:08X}u);\n\
         // Multivector component count (Cl(3,0) basis size).\n\
         constant uint  {prefix}_MV = {mv_components}u;\n\
         // Grade-1 group size (3 grade-1 components per group; one rotor).\n\
         constant uint  {prefix}_GS = {group_size}u;\n\
         // {bits}-bit values per u32 word.\n\
         constant uint  {prefix}_VPW = {vals_per_word}u;\n\
         // u32 words per group = 1 for both rotor3 (24/30 bits) and rotor4 (32/32 dense).\n\
         constant uint  {prefix}_WPG = 1u;\n"
    );
    Ok(s)
}

// ── MSL kernel sources ────────────────────────────────────────────────────────
//
// Quantize kernel: one thread per (token, group). Grid = n_tokens * n_groups.
//
// Inputs:
//   inp       : f32 [n_tokens * head_dim]   — V tensor row-major
//   rotors_in : f32 [n_groups * 4]          — per-group rotor [s, b12, b13, b23]
//   n_groups  : u32 [1]                      — scalar uniform
//   head_dim  : u32 [1]                      — scalar uniform
//
// Outputs:
//   codes_out  : u32 [n_tokens * n_groups]   — packed codes, 1 u32 per group
//   scales_out : f32 [n_tokens * n_groups]   — per-group scale
//   norms_out  : f32 [n_tokens * n_groups]   — per-group L2 norm slot (same per token)
//
// `head_dim` is passed separately (rather than computed from `n_groups * GS`)
// because the CPU encoder supports `head_dim % 3 != 0` by tail-zero-padding;
// the kernel must mask the tail in the same way.

const QUANTIZE_SOURCE_ROTOR3: &str = r"
    uint gid     = thread_position_in_grid.x;
    uint n_grp   = n_groups[0];
    uint hd      = head_dim[0];
    uint token   = gid / n_grp;
    uint grp     = gid % n_grp;

 // ── Per-token L2 norm (recomputed per group, no shared memory) ────────────
    float norm_sq = 0.0f;
    for (uint i = 0u; i < hd; i++) {
        float vi = inp[token * hd + i];
        norm_sq += vi * vi;
    }
    float norm = sqrt(norm_sq);
    if (norm < 1e-8f) norm = 1e-8f;
    norms_out[gid] = norm;

 // ── Load 3 grade-1 components with tail-pad (head_dim % 3 may be != 0) ────
    uint grp_start = grp * ROTOR3_GS;
    float v1 = (grp_start + 0u < hd) ? (inp[token * hd + grp_start + 0u] / norm) : 0.0f;
    float v2 = (grp_start + 1u < hd) ? (inp[token * hd + grp_start + 1u] / norm) : 0.0f;
    float v3 = (grp_start + 2u < hd) ? (inp[token * hd + grp_start + 2u] / norm) : 0.0f;

 // ── Load rotor [s, b12, b13, b23] ─────────────────────────────────────────
    uint r_base = grp * 4u;
    float s   = rotors_in[r_base + 0u];
    float b12 = rotors_in[r_base + 1u];
    float b13 = rotors_in[r_base + 2u];
    float b23 = rotors_in[r_base + 3u];

 // ── Apply 3×3 SO(3) rotation matrix M(R) (derived from R * mv * R̃) ──────
    float s2 = s * s;
    float b12_2 = b12 * b12;
    float b13_2 = b13 * b13;
    float b23_2 = b23 * b23;

    float R1 =
        (s2 - b12_2 - b13_2 + b23_2) * v1
      + (2.0f * s * b12 - 2.0f * b13 * b23) * v2
      + (2.0f * s * b13 + 2.0f * b12 * b23) * v3;
    float R2 =
        (-2.0f * s * b12 - 2.0f * b23 * b13) * v1
      + (s2 - b12_2 - b23_2 + b13_2) * v2
      + (2.0f * s * b23 - 2.0f * b12 * b13) * v3;
    float R3 =
        (-2.0f * s * b13 + 2.0f * b23 * b12) * v1
      + (-2.0f * s * b23 - 2.0f * b13 * b12) * v2
      + (s2 - b13_2 - b23_2 + b12_2) * v3;

 // ── 8-component MV: [0, R1, R2, R3, 0, 0, 0, 0] ──────────────────────────
    float rots[8];
    rots[0] = 0.0f;
    rots[1] = R1;
    rots[2] = R2;
    rots[3] = R3;
    rots[4] = 0.0f;
    rots[5] = 0.0f;
    rots[6] = 0.0f;
    rots[7] = 0.0f;

 // ── Per-group scale = max|R_i| / CB_MAX ──────────────────────────────────
    float abs_max = max(max(abs(R1), abs(R2)), abs(R3));
    float scale = (abs_max < 1e-12f) ? 1e-12f : (abs_max / ROTOR3_CB_MAX);
    scales_out[gid] = scale;

 // ── 3-bit quantize all 8 components and pack into 1 u32 via atomic OR ────
    uint code_word = gid * ROTOR3_WPG;   // = gid (WPG = 1)
    for (uint e = 0u; e < ROTOR3_MV; e++) {
        float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
        uint idx = 0u;
        for (uint bi = 0u; bi < 7u; bi++) {
            if (norm_val > ROTOR3_BOUNDS[bi]) idx++;
        }
        uint shift = e * 3u;
        atomic_fetch_or_explicit((device atomic_uint*)&codes_out[code_word],
                                 (idx & 0x7u) << shift,
                                 memory_order_relaxed);
    }
";

const DEQUANTIZE_SOURCE_ROTOR3: &str = r"
    uint gid     = thread_position_in_grid.x;
    uint n_grp   = n_groups[0];
    uint hd      = head_dim[0];
    uint token   = gid / n_grp;
    uint grp     = gid % n_grp;

    float scale = scales_in[gid];
    float norm  = norms_in[gid];

 // ── Unpack 8 codes from 1 u32 ────────────────────────────────────────────
    uint word = codes_in[gid * ROTOR3_WPG];
    float rots[8];
    for (uint e = 0u; e < ROTOR3_MV; e++) {
        uint shift = e * 3u;
        uint idx   = (word >> shift) & 0x7u;
        rots[e]    = ROTOR3_CB[idx] * scale;
    }

 // ── Extract grade-1 components only (zero slots discarded) ───────────────
    float c1 = rots[1];
    float c2 = rots[2];
    float c3 = rots[3];

 // ── Load rotor and apply inverse rotation M(R)^T = M(R̃) ─────────────────
    uint r_base = grp * 4u;
    float s   = rotors_in[r_base + 0u];
    float b12 = rotors_in[r_base + 1u];
    float b13 = rotors_in[r_base + 2u];
    float b23 = rotors_in[r_base + 3u];

    float s2 = s * s;
    float b12_2 = b12 * b12;
    float b13_2 = b13 * b13;
    float b23_2 = b23 * b23;

    // Transpose of M(R) — equivalent to M(R̃) where R̃ flips bivector signs.
    float v1 =
        (s2 - b12_2 - b13_2 + b23_2) * c1
      + (-2.0f * s * b12 - 2.0f * b23 * b13) * c2
      + (-2.0f * s * b13 + 2.0f * b23 * b12) * c3;
    float v2 =
        (2.0f * s * b12 - 2.0f * b13 * b23) * c1
      + (s2 - b12_2 - b23_2 + b13_2) * c2
      + (-2.0f * s * b23 - 2.0f * b13 * b12) * c3;
    float v3 =
        (2.0f * s * b13 + 2.0f * b12 * b23) * c1
      + (2.0f * s * b23 - 2.0f * b12 * b13) * c2
      + (s2 - b13_2 - b23_2 + b12_2) * c3;

 // ── Write grade-1 components × per-token norm, mask tail-pad ─────────────
    uint grp_start = grp * ROTOR3_GS;
    if (grp_start + 0u < hd) out[token * hd + grp_start + 0u] = v1 * norm;
    if (grp_start + 1u < hd) out[token * hd + grp_start + 1u] = v2 * norm;
    if (grp_start + 2u < hd) out[token * hd + grp_start + 2u] = v3 * norm;
";

const QUANTIZE_SOURCE_ROTOR4: &str = r"
    uint gid     = thread_position_in_grid.x;
    uint n_grp   = n_groups[0];
    uint hd      = head_dim[0];
    uint token   = gid / n_grp;
    uint grp     = gid % n_grp;

    float norm_sq = 0.0f;
    for (uint i = 0u; i < hd; i++) {
        float vi = inp[token * hd + i];
        norm_sq += vi * vi;
    }
    float norm = sqrt(norm_sq);
    if (norm < 1e-8f) norm = 1e-8f;
    norms_out[gid] = norm;

    uint grp_start = grp * ROTOR4_GS;
    float v1 = (grp_start + 0u < hd) ? (inp[token * hd + grp_start + 0u] / norm) : 0.0f;
    float v2 = (grp_start + 1u < hd) ? (inp[token * hd + grp_start + 1u] / norm) : 0.0f;
    float v3 = (grp_start + 2u < hd) ? (inp[token * hd + grp_start + 2u] / norm) : 0.0f;

    uint r_base = grp * 4u;
    float s   = rotors_in[r_base + 0u];
    float b12 = rotors_in[r_base + 1u];
    float b13 = rotors_in[r_base + 2u];
    float b23 = rotors_in[r_base + 3u];

    float s2 = s * s;
    float b12_2 = b12 * b12;
    float b13_2 = b13 * b13;
    float b23_2 = b23 * b23;

    float R1 =
        (s2 - b12_2 - b13_2 + b23_2) * v1
      + (2.0f * s * b12 - 2.0f * b13 * b23) * v2
      + (2.0f * s * b13 + 2.0f * b12 * b23) * v3;
    float R2 =
        (-2.0f * s * b12 - 2.0f * b23 * b13) * v1
      + (s2 - b12_2 - b23_2 + b13_2) * v2
      + (2.0f * s * b23 - 2.0f * b12 * b13) * v3;
    float R3 =
        (-2.0f * s * b13 + 2.0f * b23 * b12) * v1
      + (-2.0f * s * b23 - 2.0f * b13 * b12) * v2
      + (s2 - b13_2 - b23_2 + b12_2) * v3;

    float rots[8];
    rots[0] = 0.0f;
    rots[1] = R1;
    rots[2] = R2;
    rots[3] = R3;
    rots[4] = 0.0f;
    rots[5] = 0.0f;
    rots[6] = 0.0f;
    rots[7] = 0.0f;

    float abs_max = max(max(abs(R1), abs(R2)), abs(R3));
    float scale = (abs_max < 1e-12f) ? 1e-12f : (abs_max / ROTOR4_CB_MAX);
    scales_out[gid] = scale;

    uint code_word = gid * ROTOR4_WPG;
    for (uint e = 0u; e < ROTOR4_MV; e++) {
        float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
        uint idx = 0u;
        for (uint bi = 0u; bi < 15u; bi++) {
            if (norm_val > ROTOR4_BOUNDS[bi]) idx++;
        }
        uint shift = e * 4u;
        atomic_fetch_or_explicit((device atomic_uint*)&codes_out[code_word],
                                 (idx & 0xFu) << shift,
                                 memory_order_relaxed);
    }
";

const DEQUANTIZE_SOURCE_ROTOR4: &str = r"
    uint gid     = thread_position_in_grid.x;
    uint n_grp   = n_groups[0];
    uint hd      = head_dim[0];
    uint token   = gid / n_grp;
    uint grp     = gid % n_grp;

    float scale = scales_in[gid];
    float norm  = norms_in[gid];

    uint word = codes_in[gid * ROTOR4_WPG];
    float rots[8];
    for (uint e = 0u; e < ROTOR4_MV; e++) {
        uint shift = e * 4u;
        uint idx   = (word >> shift) & 0xFu;
        rots[e]    = ROTOR4_CB[idx] * scale;
    }

    float c1 = rots[1];
    float c2 = rots[2];
    float c3 = rots[3];

    uint r_base = grp * 4u;
    float s   = rotors_in[r_base + 0u];
    float b12 = rotors_in[r_base + 1u];
    float b13 = rotors_in[r_base + 2u];
    float b23 = rotors_in[r_base + 3u];

    float s2 = s * s;
    float b12_2 = b12 * b12;
    float b13_2 = b13 * b13;
    float b23_2 = b23 * b23;

    float v1 =
        (s2 - b12_2 - b13_2 + b23_2) * c1
      + (-2.0f * s * b12 - 2.0f * b23 * b13) * c2
      + (-2.0f * s * b13 + 2.0f * b23 * b12) * c3;
    float v2 =
        (2.0f * s * b12 - 2.0f * b13 * b23) * c1
      + (s2 - b12_2 - b23_2 + b13_2) * c2
      + (-2.0f * s * b23 - 2.0f * b13 * b12) * c3;
    float v3 =
        (2.0f * s * b13 + 2.0f * b12 * b23) * c1
      + (2.0f * s * b23 - 2.0f * b12 * b13) * c2
      + (s2 - b13_2 - b23_2 + b12_2) * c3;

    uint grp_start = grp * ROTOR4_GS;
    if (grp_start + 0u < hd) out[token * hd + grp_start + 0u] = v1 * norm;
    if (grp_start + 1u < hd) out[token * hd + grp_start + 1u] = v2 * norm;
    if (grp_start + 2u < hd) out[token * hd + grp_start + 2u] = v3 * norm;
";

// ── Kernel singletons ─────────────────────────────────────────────────────────
//
// Store `Result<String, String>` (Send-able) in the header `OnceLock` so a
// transient failure does not poison the lock — every call re-converts the
// stored error message to a fresh `Error::Mlx`.

static ROTOR3_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ROTOR3_DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ROTOR3_KERNEL_HEADER: OnceLock<std::result::Result<String, String>> = OnceLock::new();

static ROTOR4_QUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ROTOR4_DEQUANT_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ROTOR4_KERNEL_HEADER: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn kernel_header_rotor3() -> Result<&'static str> {
    match ROTOR3_KERNEL_HEADER
        .get_or_init(|| build_msl_header(ROTOR3_BITS).map_err(|e| e.to_string()))
    {
        Ok(s) => Ok(s.as_str()),
        Err(msg) => Err(rmlx_core::error::Error::Mlx(format!(
            "rotor3 MSL header init: {msg}"
        ))),
    }
}

fn kernel_header_rotor4() -> Result<&'static str> {
    match ROTOR4_KERNEL_HEADER
        .get_or_init(|| build_msl_header(ROTOR4_BITS).map_err(|e| e.to_string()))
    {
        Ok(s) => Ok(s.as_str()),
        Err(msg) => Err(rmlx_core::error::Error::Mlx(format!(
            "rotor4 MSL header init: {msg}"
        ))),
    }
}

fn rotor3_quant_kernel() -> Result<&'static MetalKernel> {
    ROTOR3_QUANT_KERNEL
        .get_or_init(|| {
            let header = kernel_header_rotor3()?;
            MetalKernel::new(
                "rmlx_rotor3_quantize",
                header,
                QUANTIZE_SOURCE_ROTOR3,
                &["inp", "rotors_in", "n_groups", "head_dim"],
                &["codes_out", "scales_out", "norms_out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("rotor3_quantize kernel init: {e}")))
}

fn rotor3_dequant_kernel() -> Result<&'static MetalKernel> {
    ROTOR3_DEQUANT_KERNEL
        .get_or_init(|| {
            let header = kernel_header_rotor3()?;
            MetalKernel::new(
                "rmlx_rotor3_dequantize",
                header,
                DEQUANTIZE_SOURCE_ROTOR3,
                &[
                    "codes_in",
                    "scales_in",
                    "norms_in",
                    "rotors_in",
                    "n_groups",
                    "head_dim",
                ],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("rotor3_dequantize kernel init: {e}")))
}

fn rotor4_quant_kernel() -> Result<&'static MetalKernel> {
    ROTOR4_QUANT_KERNEL
        .get_or_init(|| {
            let header = kernel_header_rotor4()?;
            MetalKernel::new(
                "rmlx_rotor4_quantize",
                header,
                QUANTIZE_SOURCE_ROTOR4,
                &["inp", "rotors_in", "n_groups", "head_dim"],
                &["codes_out", "scales_out", "norms_out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("rotor4_quantize kernel init: {e}")))
}

fn rotor4_dequant_kernel() -> Result<&'static MetalKernel> {
    ROTOR4_DEQUANT_KERNEL
        .get_or_init(|| {
            let header = kernel_header_rotor4()?;
            MetalKernel::new(
                "rmlx_rotor4_dequantize",
                header,
                DEQUANTIZE_SOURCE_ROTOR4,
                &[
                    "codes_in",
                    "scales_in",
                    "norms_in",
                    "rotors_in",
                    "n_groups",
                    "head_dim",
                ],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| rmlx_core::error::Error::Mlx(format!("rotor4_dequantize kernel init: {e}")))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn n_groups_for(head_dim: usize) -> usize {
    head_dim.div_ceil(ROTOR3_GROUP_SIZE)
}

fn make_scalar_u32_array(value: u32) -> Result<Array> {
    let bytes = value.to_le_bytes();
    Array::from_bytes(&bytes, &[1_i32], Dtype::U32)
}

fn validate_rotors_len(rotors: &Array, n_groups: usize) -> Result<()> {
    let total: usize = rotors.shape().iter().map(|&d| d as usize).product();
    let expected = n_groups * 4;
    if total != expected {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor MSL: rotors length {total} != expected {expected} (n_groups={n_groups} × 4)"
        )));
    }
    Ok(())
}

// ── Public encode/decode API ──────────────────────────────────────────────────

/// GPU rotor3 V-side quantize.
///
/// Mirrors [`crate::rotorquant::rotor3_encode`]: applies the Cl(3,0) sandwich
/// `R * mv * R̃` per (token, group), packs 8 codes per group into 1 u32 at 3
/// bits each.
///
/// # Arguments
///
/// * `v_full` — input V tensor, any shape whose total element count is
///   `n_tokens * head_dim`.
/// * `rotors` — flat `[n_groups * 4]` f32 rotor table (`[s, b12, b13, b23]`
///   per group). Caller is responsible for generating this via
///   [`crate::clifford::make_rotor_table`].
/// * `head_dim` — must be > 0.
///
/// # Returns
///
/// `(codes, scales, norms)`:
/// * `codes`: `u32 [n_tokens * n_groups]` (1 word per group, 3 bits × 8 codes).
/// * `scales`: `f32 [n_tokens * n_groups]` per-group scale.
/// * `norms`: `f32 [n_tokens * n_groups]` per-group L2 norm slot (caller may
///   deduplicate to per-token via [`rotor_gpu_outputs_to_cpu`]).
///
/// # Errors
///
/// Returns `Error::Quant` for invalid shapes; `Error::Mlx` for Metal kernel
/// compilation / dispatch failures.
pub fn rotor_quantize_v3_gpu(
    v_full: &Array,
    rotors: &Array,
    head_dim: usize,
    device: Device,
) -> Result<(Array, Array, Array)> {
    rotor_quantize_gpu_impl(v_full, rotors, head_dim, device, ROTOR3_BITS)
}

/// GPU rotor4 V-side quantize. Mirrors [`crate::rotorquant::rotor4_encode`]
/// with 4-bit codes (16-centroid Lloyd codebook, dense 32-bit pack).
///
/// See [`rotor_quantize_v3_gpu`] for argument and error semantics.
pub fn rotor_quantize_v4_gpu(
    v_full: &Array,
    rotors: &Array,
    head_dim: usize,
    device: Device,
) -> Result<(Array, Array, Array)> {
    rotor_quantize_gpu_impl(v_full, rotors, head_dim, device, ROTOR4_BITS)
}

/// GPU rotor3 dequantize. Mirrors [`crate::rotorquant::rotor3_decode`].
///
/// Production code uses CPU `vs.dequant()`; GPU dequant kept for future
/// wire-up and parity tests.
///
/// `head_dim` must match the encode call. `out_dtype` selects the returned
/// array element type.
///
/// # Errors
///
/// Returns `Error::Quant` for shape mismatches; `Error::Mlx` for Metal
/// dispatch failures.
pub fn rotor_dequantize_v3_gpu(
    codes_packed: &Array,
    scales: &Array,
    norms: &Array,
    rotors: &Array,
    head_dim: usize,
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    rotor_dequantize_gpu_impl(
        codes_packed,
        scales,
        norms,
        rotors,
        head_dim,
        out_dtype,
        device,
        ROTOR3_BITS,
    )
}

/// GPU rotor4 dequantize. Mirrors [`crate::rotorquant::rotor4_decode`].
///
/// Production code uses CPU `vs.dequant()`; GPU dequant kept for future
/// wire-up and parity tests.
pub fn rotor_dequantize_v4_gpu(
    codes_packed: &Array,
    scales: &Array,
    norms: &Array,
    rotors: &Array,
    head_dim: usize,
    out_dtype: Dtype,
    device: Device,
) -> Result<Array> {
    rotor_dequantize_gpu_impl(
        codes_packed,
        scales,
        norms,
        rotors,
        head_dim,
        out_dtype,
        device,
        ROTOR4_BITS,
    )
}

fn rotor_quantize_gpu_impl(
    v_full: &Array,
    rotors: &Array,
    head_dim: usize,
    device: Device,
    bits: u8,
) -> Result<(Array, Array, Array)> {
    if head_dim == 0 {
        return Err(rmlx_core::error::Error::Quant(
            "rotor_quantize_gpu: head_dim must be > 0".to_owned(),
        ));
    }
    let total_elems: usize = v_full.shape().iter().map(|&d| d as usize).product();
    if !total_elems.is_multiple_of(head_dim) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_quantize_gpu: total elements {total_elems} not a multiple of head_dim={head_dim}"
        )));
    }
    let n_tokens = total_elems / head_dim;
    let n_groups = n_groups_for(head_dim);
    let total_groups = n_tokens * n_groups;
    validate_rotors_len(rotors, n_groups)?;

    if bits == ROTOR3_BITS {
        tracing::debug!(
            target: "rmlx::kv_quant::rotor3",
            n_tokens,
            n_groups,
            head_dim,
            "rotor3 GPU encode block"
        );
    } else {
        tracing::debug!(
            target: "rmlx::kv_quant::rotor4",
            n_tokens,
            n_groups,
            head_dim,
            "rotor4 GPU encode block"
        );
    }

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

    let rotors_flat = if rotors.ndim() == 1 {
        rotors.try_clone()?
    } else {
        rotors.reshape(&[(n_groups * 4) as i32], device)?
    };
    let rotors_f32 = if rotors_flat.dtype() == Dtype::F32 {
        rotors_flat
    } else {
        rotors_flat.astype(Dtype::F32, device)?
    };

    // narrowing usize → u32 is safe: head_dim ≤ 4096 in practice; n_groups
    // bounded by head_dim/3 ≤ 1366, both well within u32::MAX.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "n_groups, head_dim are small (≤ 4096) — fit u32"
    )]
    let n_groups_arr = make_scalar_u32_array(n_groups as u32)?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "head_dim ≤ 4096 — fits u32"
    )]
    let head_dim_arr = make_scalar_u32_array(head_dim as u32)?;

    let kernel = match bits {
        ROTOR3_BITS => rotor3_quant_kernel()?,
        ROTOR4_BITS => rotor4_quant_kernel()?,
        other => {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "rotor_quantize_gpu: unsupported bits {other}"
            )))
        }
    };

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&v_f32)?;
    invoke.add_input(&rotors_f32)?;
    invoke.add_input(&n_groups_arr)?;
    invoke.add_input(&head_dim_arr)?;

    // codes_out: 1 u32 per group.
    invoke.add_output_shape(&[total_groups as i32], Dtype::U32)?;
    invoke.add_output_shape(&[total_groups as i32], Dtype::F32)?;
    invoke.add_output_shape(&[total_groups as i32], Dtype::F32)?;

    // Zero-initialise: codes_out written via atomic OR.
    invoke.set_init_value(0.0)?;

    invoke.set_grid(total_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.len() < 3 {
        return Err(rmlx_core::error::Error::Mlx(
            "rotor_quantize_gpu: expected 3 outputs".to_owned(),
        ));
    }
    let norms = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales, norms))
}

#[allow(clippy::too_many_arguments)]
fn rotor_dequantize_gpu_impl(
    codes_packed: &Array,
    scales: &Array,
    norms: &Array,
    rotors: &Array,
    head_dim: usize,
    out_dtype: Dtype,
    device: Device,
    bits: u8,
) -> Result<Array> {
    if head_dim == 0 {
        return Err(rmlx_core::error::Error::Quant(
            "rotor_dequantize_gpu: head_dim must be > 0".to_owned(),
        ));
    }
    let total_groups: usize = norms.shape().iter().map(|&d| d as usize).product();
    let n_groups = n_groups_for(head_dim);
    if total_groups == 0 || !total_groups.is_multiple_of(n_groups) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_dequantize_gpu: norms length {total_groups} not a multiple of n_groups={n_groups}"
        )));
    }
    let n_tokens = total_groups / n_groups;
    let total_elems = n_tokens * head_dim;
    validate_rotors_len(rotors, n_groups)?;

    let codes_len: usize = codes_packed.shape().iter().map(|&d| d as usize).product();
    if codes_len != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_dequantize_gpu: codes length {codes_len} != total_groups {total_groups}"
        )));
    }
    let scales_len: usize = scales.shape().iter().map(|&d| d as usize).product();
    if scales_len != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_dequantize_gpu: scales length {scales_len} != total_groups {total_groups}"
        )));
    }

    let codes_flat = codes_packed.reshape(&[total_groups as i32], device)?;
    let scales_flat = scales.reshape(&[total_groups as i32], device)?;
    let norms_flat = norms.reshape(&[total_groups as i32], device)?;
    let rotors_flat = if rotors.ndim() == 1 {
        rotors.try_clone()?
    } else {
        rotors.reshape(&[(n_groups * 4) as i32], device)?
    };
    let rotors_f32 = if rotors_flat.dtype() == Dtype::F32 {
        rotors_flat
    } else {
        rotors_flat.astype(Dtype::F32, device)?
    };

    #[allow(
        clippy::cast_possible_truncation,
        reason = "n_groups, head_dim ≤ 4096 — fit u32"
    )]
    let n_groups_arr = make_scalar_u32_array(n_groups as u32)?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "head_dim ≤ 4096 — fits u32"
    )]
    let head_dim_arr = make_scalar_u32_array(head_dim as u32)?;

    let kernel = match bits {
        ROTOR3_BITS => rotor3_dequant_kernel()?,
        ROTOR4_BITS => rotor4_dequant_kernel()?,
        other => {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "rotor_dequantize_gpu: unsupported bits {other}"
            )))
        }
    };

    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&norms_flat)?;
    invoke.add_input(&rotors_f32)?;
    invoke.add_input(&n_groups_arr)?;
    invoke.add_input(&head_dim_arr)?;
    invoke.add_output_shape(&[total_elems as i32], Dtype::F32)?;

    // The decode kernel writes only `head_dim % 3 != 0` valid slots; the tail
    // padding (if any) must be zero-initialised so the unwritten elements are
    // deterministic.
    invoke.set_init_value(0.0)?;

    invoke.set_grid(total_groups as i32, 1, 1)?;
    invoke.set_thread_group(1, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(rmlx_core::error::Error::Mlx(
            "rotor_dequantize_gpu: expected 1 output".to_owned(),
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

// ── Materialise GPU outputs to CPU vectors ────────────────────────────────────

/// Read the GPU rotor encode outputs back to host and return the CPU vectors
/// used by [`crate::storage::quant_rotor_v3::RotorBlocks`] /
/// [`crate::storage::quant_rotor_v4`]. The per-group GPU norm slots are
/// deduplicated to per-token (first slot of each token).
///
/// # Errors
///
/// Returns `Error::Mlx` on Array materialise / byte-decode failure or
/// `Error::Quant` if the output shapes do not match the declared grid.
pub fn rotor_gpu_outputs_to_cpu(
    codes_gpu: &Array,
    scales_gpu: &Array,
    norms_gpu: &Array,
    n_tokens: usize,
    n_groups: usize,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    let total_groups = n_tokens.saturating_mul(n_groups);
    if total_groups == 0 {
        return Err(rmlx_core::error::Error::Quant(
            "rotor_gpu_outputs_to_cpu: n_tokens * n_groups must be > 0".to_owned(),
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

    if codes.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_gpu_outputs_to_cpu: codes len {} != expected {total_groups}",
            codes.len()
        )));
    }
    if scales.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_gpu_outputs_to_cpu: scales len {} != expected {total_groups}",
            scales.len()
        )));
    }
    if norms_per_group.len() != total_groups {
        return Err(rmlx_core::error::Error::Quant(format!(
            "rotor_gpu_outputs_to_cpu: norms len {} != expected {total_groups}",
            norms_per_group.len()
        )));
    }

    let mut norms = Vec::with_capacity(n_tokens);
    for tok in 0..n_tokens {
        let i = tok * n_groups;
        #[allow(
            clippy::indexing_slicing,
            reason = "tok < n_tokens, i = tok * n_groups < total_groups validated above"
        )]
        norms.push(norms_per_group[i]);
    }

    Ok((codes, scales, norms))
}

/// Build an MLX `Array` view of a flat rotor table (`[n_groups * 4]` f32).
///
/// Convenience helper for callers that hold the rotor table as a plain
/// `Vec<f32>` (the storage layer's `rotors` field) and need an `Array` to
/// pass into [`rotor_quantize_v3_gpu`] / [`rotor_dequantize_v3_gpu`].
///
/// # Errors
///
/// Returns `Error::Mlx` if `Array::from_bytes` rejects the byte slice.
pub fn rotor_table_to_array(rotors: &[f32]) -> Result<Array> {
    let bytes: Vec<u8> = rotors.iter().flat_map(|&v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[rotors.len() as i32], Dtype::F32)
}

#[cfg(test)]
#[path = "rotorquant_msl_tests.rs"]
mod rotorquant_msl_tests;
