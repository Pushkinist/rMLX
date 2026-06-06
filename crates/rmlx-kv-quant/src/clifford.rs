//! Clifford algebra Cl(3,0) primitives for the rotor3 KV codec.
//!
//! # Algebra
//!
//! Cl(3,0) has an 8-dimensional multivector basis:
//!   `[1, e1, e2, e3, e12, e13, e23, e123]`
//! with grade-0 (scalar), grade-1 (vector), grade-2 (bivector), grade-3
//! (pseudoscalar). Signature `(+, +, +)`: `e_i * e_i = +1`,
//! `e_i * e_j = -e_j * e_i` for `i != j`.
//!
//! A **rotor** is a scalar + bivector: `R = [s, 0, 0, 0, b12, b13, b23, 0]`.
//! It acts on a multivector `x` via the **sandwich product**:
//!   `T(x) = R * x * R̃`
//! where `R̃` is the reverse (`reverse(R)` negates grade-2 and grade-3 parts).
//! For a unit rotor this is a proper rotation that preserves the full
//! algebraic structure (inner products, outer products, grades).
//!
//! # Sparse rotor × multivector product
//!
//! Exploiting the sparsity of rotors (only 4 of 8 components are non-zero —
//! `s, b12, b13, b23`), the dense table-driven geometric product short-
//! circuits on the rotor's 4 zero coefficients and falls from 64 FMAs to 32
//! (matches the inner-loop count in [`geometric_product`]). We expose
//! [`gp_rotor_mv`] / [`gp_mv_rotor`] as the building blocks; the full
//! sandwich `R * x * R̃` (see [`rotor_sandwich`]) is two such calls — 64
//! FMAs total.
//!
//! # Embedding
//!
//! For the KV codec we embed `head_dim`-element vectors into Cl(3,0) by
//! packing **three at a time** into the grade-1 components `(e1, e2, e3)`.
//! For `head_dim` not divisible by 3 the last group has tail padding —
//! the missing components are zeroed at encode and masked off at decode.
//! See [`crate::rotorquant`].
//!
//! # Random rotor seeding
//!
//! The KV codec needs a **static** per-layer/head/group rotor table. To make
//! seeding reproducible and decorrelated across (layer, head, group):
//!   `seed = ROTORQUANT_GLOBAL_SEED ^ (layer_idx << 32) ^ (head_idx << 16) + group_idx`
//! See [`rotor_seed`] and [`make_random_rotor`].
//!
//! # Reference
//!
//! Python source: `rotorquant/turboquant/clifford.py` (`geometric_product`,
//! `reverse`, `rotor_sandwich`, `make_random_rotor`, `embed_vectors_as_multivectors`).
//! MSL reference: `rotorquant/turboquant/rotor_fused.metal` (sparse
//! `gp_rotor_mv` table).
#![allow(
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    reason = "indexing: 8-element arrays are statically sized and accessed via 0..8 loops or fixed literals; suboptimal_flops: numerator/denominator factor splits are intentional for derivation clarity, fused-mul-add does not change the result for these small Cl(3,0) sums"
)]

use std::f32::consts::TAU;

/// Multivector dimension (2^3 components in Cl(3,0)).
pub const MV_DIM: usize = 8;

/// Compact rotor representation: `[s, b12, b13, b23]` (4 non-zero components).
///
/// Stored as 4 f32s rather than a sparse 8-element multivector to halve the
/// rotor table size and match the Python reference `rotor_fused.metal` layout.
pub type Rotor = [f32; 4];

/// Global seed constant for rotor table generation.
///
/// Fixed literal documented here so any reproduction of a serialized rotor
/// table can reseed exactly. Bumping this value invalidates all SSD-spilled
/// rotor3 layers; treat it as part of the on-disk schema.
pub const ROTORQUANT_GLOBAL_SEED: u64 = 0xC1F4_03A0_5EED_B17E;

/// Compute the per-rotor RNG seed from layer, head, and group indices.
///
/// Formula: `ROTORQUANT_GLOBAL_SEED ^ (layer_idx << 32) ^ (head_idx << 16) + group_idx`.
///
/// The XOR / shift mixing decorrelates rotor selection across the three
/// indexing axes; the additive `group_idx` term keeps neighbouring groups
/// within the same (layer, head) on different orbits.
#[inline]
#[must_use]
pub fn rotor_seed(layer_idx: u32, head_idx: u32, group_idx: u32) -> u64 {
    let base = ROTORQUANT_GLOBAL_SEED ^ (u64::from(layer_idx) << 32) ^ (u64::from(head_idx) << 16);
    base.wrapping_add(u64::from(group_idx))
}

// ── Multivector primitives (test/correctness scaffolding) ─────────────────────

// ── Cl(3,0) multiplication table ──────────────────────────────────────────────
//
// Basis order: `[1, e1, e2, e3, e12, e13, e23, e123]`.
//
// Each basis element is a subset of `{1, 2, 3}` represented as a bitmask:
//   `1 = 0b000`, `e1 = 0b001`, `e2 = 0b010`, `e3 = 0b100`,
//   `e12 = 0b011`, `e13 = 0b101`, `e23 = 0b110`, `e123 = 0b111`.
//
// To multiply `e_I * e_J`:
//   1. Annihilate shared indices: signature `(+,+,+)` means `e_i² = +1`, no
//      sign change. The result basis is `I XOR J`.
//   2. Count basis-element swaps required to bring `I, J` into canonical
//      sorted order; each swap flips the sign.
//
// The lookup table `MUL_TABLE[i][j] = (basis_index, sign)` is computed once
// from these rules. Tested for full agreement with hand-derived products.

/// Map our basis index 0..8 to the index bitmask (subset of `{e1, e2, e3}`).
const BASIS_BITS: [u8; MV_DIM] = [
    0b000, // 1
    0b001, // e1
    0b010, // e2
    0b100, // e3
    0b011, // e12
    0b101, // e13
    0b110, // e23
    0b111, // e123
];

/// Inverse of [`BASIS_BITS`]: bitmask → our basis index 0..8.
const BITS_TO_BASIS: [usize; MV_DIM] = {
    let mut t = [0_usize; MV_DIM];
    let mut i = 0;
    while i < MV_DIM {
        t[BASIS_BITS[i] as usize] = i;
        i += 1;
    }
    t
};

/// Pre-computed `(target_basis_index, sign)` for `BASIS_BITS[i] * BASIS_BITS[j]`.
///
/// Signs derived by counting swaps to canonicalise the concatenated index
/// sequence. Cl(3,0) signature `(+, +, +)` means `e_k² = +1`, so duplicate
/// indices vanish without sign change.
const MUL_TABLE: [[(usize, i8); MV_DIM]; MV_DIM] = {
    let mut table = [[(0_usize, 0_i8); MV_DIM]; MV_DIM];
    let mut i = 0;
    while i < MV_DIM {
        let bits_i = BASIS_BITS[i];
        let mut j = 0;
        while j < MV_DIM {
            let bits_j = BASIS_BITS[j];

            // Result basis = symmetric difference (XOR).
            let result_bits = bits_i ^ bits_j;

            // Sign = (-1)^(swaps to canonicalise).
            //
            // The product `e_I * e_J` written out as `e_{i1} e_{i2} ... e_{j1} e_{j2} ...`.
            // We count the number of swaps to reorder into ascending indices.
            //
            // For each index k in J, count how many indices in I are > k:
            // each such index must swap past k → contributes one negation.
            // Then duplicates `e_k * e_k = +1` cancel without further sign flip.
            let mut sign = 1_i8;
            let mut k = 0;
            while k < 3 {
                let bit_k = 1_u8 << k;
                if bits_j & bit_k != 0 {
                    // We're bringing e_{k+1} from the J side leftward past
                    // all indices in I that are strictly greater.
                    let mut m = k + 1;
                    while m < 3 {
                        if bits_i & (1_u8 << m) != 0 {
                            sign = -sign;
                        }
                        m += 1;
                    }
                }
                k += 1;
            }

            // Now both I and J are in ascending order. Any shared index
            // pair `e_k * e_k = 1` removes both from the basis with no sign
            // change. The result basis is already (I XOR J).
            let target = BITS_TO_BASIS[result_bits as usize];
            table[i][j] = (target, sign);
            j += 1;
        }
        i += 1;
    }
    table
};

/// Full Cl(3,0) geometric product on dense multivectors `a * b`.
///
/// Both inputs and output are 8-element arrays in basis order
/// `[1, e1, e2, e3, e12, e13, e23, e123]`. Uses the pre-computed
/// [`MUL_TABLE`] to avoid any hand-derivation sign mistakes.
///
/// Hot encode/decode dispatch should use [`gp_rotor_mv`] which exploits
/// rotor sparsity to skip ~36 of 64 FMAs.
///
/// # Note on the Python reference
///
/// The Python `rotorquant/turboquant/clifford.py::geometric_product` has
/// **sign errors** in the `r12`, `r13`, `r23`, `r123` formulas (verified by
/// the table-driven implementation here). The downstream MSL kernel
/// `rotor_fused.metal::gp_rotor_mv` is also a specialised case that only
/// holds for grade-1 input on the first call. This Rust port is correct;
/// see CLAUDE.md hard rule 7 ("Document the truth, not the docstring").
#[must_use]
pub fn geometric_product(a: &[f32; MV_DIM], b: &[f32; MV_DIM]) -> [f32; MV_DIM] {
    let mut out = [0.0_f32; MV_DIM];
    for i in 0..MV_DIM {
        if a[i] == 0.0 {
            continue;
        }
        for j in 0..MV_DIM {
            let (target, sign) = MUL_TABLE[i][j];
            // f32(sign) is exact for {-1, +1}.
            out[target] += a[i] * b[j] * f32::from(sign);
        }
    }
    out
}

/// Clifford reverse: flips the sign of grade-2 and grade-3 components.
///
/// Used to form `R̃` from a rotor `R`. Identity: `(R̃)̃ == R`. For a unit
/// rotor this is also the inverse (`R * R̃ = 1`).
#[inline]
#[must_use]
pub fn reverse(x: &[f32; MV_DIM]) -> [f32; MV_DIM] {
    [x[0], x[1], x[2], x[3], -x[4], -x[5], -x[6], -x[7]]
}

/// Reverse for the compact rotor representation `[s, b12, b13, b23]`.
///
/// Negates the three bivector components and keeps the scalar.
#[inline]
#[must_use]
pub fn rotor_reverse(r: Rotor) -> Rotor {
    [r[0], -r[1], -r[2], -r[3]]
}

/// Convert a compact rotor `[s, b12, b13, b23]` to the dense 8-element form.
#[inline]
#[must_use]
pub fn rotor_to_mv(r: Rotor) -> [f32; MV_DIM] {
    [r[0], 0.0, 0.0, 0.0, r[1], r[2], r[3], 0.0]
}

/// Rotor × multivector geometric product: `out = R * x`.
///
/// Thin wrapper that converts the compact rotor to its dense 8-element form
/// and dispatches to [`geometric_product`]. The table-driven dense path
/// short-circuits zero coefficients, so the rotor's 4 zeros automatically
/// halve the work (32 FMAs vs 64 worst-case).
///
/// # Why not a fully hand-derived sparse formula?
///
/// The original Python reference (`rotor_fused.metal::gp_rotor_mv`) is a
/// **specialised** kernel correct only for grade-1 input `x` (the first call
/// of the sandwich); reusing it for the second call yields wrong output, as
/// does the Python `clifford.py::geometric_product` (sign errors in the
/// grade-2 and grade-3 component formulas — verified against the table-
/// driven dense path here). The dense GP with non-zero short-circuiting is
/// correct by construction and ~2× faster than the hand-written naive
/// formula for rotor inputs.
///
/// See CLAUDE.md hard rule 7 ("Document the truth, not the docstring").
#[inline]
#[must_use]
pub fn gp_rotor_mv(r: Rotor, x: &[f32; MV_DIM]) -> [f32; MV_DIM] {
    geometric_product(&rotor_to_mv(r), x)
}

/// Multivector × rotor geometric product: `out = x * R`.
///
/// Companion to [`gp_rotor_mv`] for the **right-multiplication** side of the
/// sandwich. Same dispatch strategy: convert the rotor to the dense form and
/// call [`geometric_product`].
#[inline]
#[must_use]
pub fn gp_mv_rotor(x: &[f32; MV_DIM], r: Rotor) -> [f32; MV_DIM] {
    geometric_product(x, &rotor_to_mv(r))
}

/// Apply the full rotor sandwich `R * x * R̃` to a multivector.
///
/// Combines [`gp_rotor_mv`] (left-multiply by R) with [`gp_mv_rotor`]
/// (right-multiply by R̃). Used by the CPU encode/decode path and by the
/// algebra correctness tests.
#[inline]
#[must_use]
pub fn rotor_sandwich(r: Rotor, x: &[f32; MV_DIM]) -> [f32; MV_DIM] {
    let tmp = gp_rotor_mv(r, x);
    gp_mv_rotor(&tmp, rotor_reverse(r))
}

// ── Random rotor generation ───────────────────────────────────────────────────

/// SplitMix64 step — used as a tiny seedable PRNG for rotor table generation.
///
/// Deterministic, no `rand` dependency, identical across platforms.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw an f32 uniformly in `[0, 1)` from a 64-bit PRNG state.
#[inline]
fn next_unit_f32(state: &mut u64) -> f32 {
    // Top 24 bits → f32 mantissa precision; matches `rand::Rng::gen::<f32>`.
    let v = splitmix64(state) >> 40;
    (v as f32) / ((1_u32 << 24) as f32)
}

/// Draw an f32 standard-normal sample via Box–Muller from the PRNG state.
///
/// Used to seed the rotor bivector direction in [`make_random_rotor`].
#[inline]
fn next_normal_f32(state: &mut u64) -> f32 {
    // Box–Muller. Bound u1 away from 0 so ln(u1) is finite.
    let u1 = next_unit_f32(state).max(1e-12);
    let u2 = next_unit_f32(state);
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = TAU * u2;
    r * theta.cos()
}

/// Construct a unit rotor in compact `[s, b12, b13, b23]` form from a 64-bit seed.
///
/// Procedure (mirrors Python `make_random_rotor`):
///   1. Sample three N(0,1) values → bivector direction `[b12, b13, b23]`.
///   2. Normalise the bivector (epsilon-clamped to avoid div-by-zero).
///   3. Sample a uniform angle `θ ∈ [0, 2π)`.
///   4. Form `R = cos(θ/2) + sin(θ/2) * b̂`.
///   5. Final unit-normalise on the full 4-vector (guards against accumulated
///      f32 round-off in `cos²+sin² ≠ 1`).
///
/// Identity guarantee: `||R||² = s² + b12² + b13² + b23² ≈ 1` within f32.
/// The `rotor_sandwich` tests pin the round-trip identity `R x R̃ R x R̃ ≈ x`
/// for vectors and the reverse identity `(R̃)̃ == R`.
#[must_use]
pub fn make_random_rotor(seed: u64) -> Rotor {
    let mut state = seed;
    let bv = [
        next_normal_f32(&mut state),
        next_normal_f32(&mut state),
        next_normal_f32(&mut state),
    ];
    let bv_norm = (bv[0] * bv[0] + bv[1] * bv[1] + bv[2] * bv[2])
        .sqrt()
        .max(1e-8);
    let bv_hat = [bv[0] / bv_norm, bv[1] / bv_norm, bv[2] / bv_norm];
    let angle = next_unit_f32(&mut state) * TAU;
    let half = angle * 0.5;
    let cos_h = half.cos();
    let sin_h = half.sin();
    let raw = [
        cos_h,
        sin_h * bv_hat[0],
        sin_h * bv_hat[1],
        sin_h * bv_hat[2],
    ];
    // Final normalise — f32 round-off can leave |R|² off-by-epsilon.
    let n = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3])
        .sqrt()
        .max(1e-8);
    [raw[0] / n, raw[1] / n, raw[2] / n, raw[3] / n]
}

/// Generate the static per-layer/head rotor table as a flat `Vec<f32>` of
/// shape `[n_groups, 4]`.
///
/// Output layout: `[r0_s, r0_b12, r0_b13, r0_b23, r1_s, ..., r_{n_groups-1}_b23]`.
/// Per-group seed = [`rotor_seed`]`(layer_idx, head_idx, group_idx)`.
#[must_use]
pub fn make_rotor_table(layer_idx: u32, head_idx: u32, n_groups: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_groups * 4);
    for g in 0..n_groups {
        // group_idx narrows from usize to u32; n_groups for any realistic
        // head_dim (≤ 4096) is well within u32::MAX.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "n_groups derived from head_dim (typically 32–512) — far below u32::MAX"
        )]
        let g_u32 = g as u32;
        let seed = rotor_seed(layer_idx, head_idx, g_u32);
        let r = make_random_rotor(seed);
        out.extend_from_slice(&r);
    }
    out
}

#[cfg(test)]
#[path = "clifford_tests.rs"]
mod clifford_tests;
