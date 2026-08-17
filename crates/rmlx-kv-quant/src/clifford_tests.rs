//! Algebra correctness tests for [`crate::clifford`].
//!
//! Covers:
//!   * Reverse identity `(R̃)̃ == R` — both compact and dense forms.
//!   * Sandwich identity: `R * x * R̃` preserves vector norm for unit rotors.
//!   * Sparse vs dense GP equivalence (`gp_rotor_mv` == `geometric_product`).
//!   * Random rotor is unit: `||R||² ≈ 1`.
//!   * Grade preservation in both directions — the algebraic fact that makes
//!     five of the rotor codec's eight stored codes dead budget.

use crate::clifford::{
    geometric_product, gp_rotor_mv, make_random_rotor, make_rotor_table, reverse, rotor_reverse,
    rotor_sandwich, rotor_seed, rotor_to_mv, Rotor, MV_DIM, ROTORQUANT_GLOBAL_SEED,
};

const TOL: f32 = 1e-5;

fn norm_sq(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < TOL
}

/// `(R̃)̃ == R` for both compact and dense reverse forms.
#[test]
fn reverse_identity() {
    let r: Rotor = make_random_rotor(0xDEAD_BEEF_CAFE_0001);

    let r_back = rotor_reverse(rotor_reverse(r));
    for (i, (&a, &b)) in r.iter().zip(r_back.iter()).enumerate() {
        assert!(
            approx_eq(a, b),
            "rotor_reverse twice should equal original at idx {i}: {a} vs {b}"
        );
    }

    let r_mv = rotor_to_mv(r);
    let r_mv_back = reverse(&reverse(&r_mv));
    for (i, (&a, &b)) in r_mv.iter().zip(r_mv_back.iter()).enumerate() {
        assert!(
            approx_eq(a, b),
            "reverse(reverse(rotor_mv)) should equal original at idx {i}: {a} vs {b}"
        );
    }
}

/// A random rotor must be unit: `s² + b12² + b13² + b23² ≈ 1`.
#[test]
fn random_rotor_is_unit() {
    for seed in 0_u64..32 {
        let r = make_random_rotor(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let n2 = norm_sq(&r);
        assert!(
            (n2 - 1.0).abs() < 1e-5,
            "rotor seed={seed:#x} norm² = {n2}, expected ≈ 1.0"
        );
    }
}

/// Sandwich identity: for a unit rotor `R`, the sandwich `R * v * R̃`
/// preserves the Clifford norm `<x x̃>_0` of any multivector.
///
/// In Cl(3,0) the reverse-induced norm has alternating signs by grade:
///   `<x x̃>_0 = x0² + x1² + x2² + x3² - x4² - x5² - x6² - x7²`
/// (grades 2 and 3 contribute negatively because `e_ij² = -1` and
/// `e_ijk² = -1`). Rotor sandwiches preserve this quadratic form, NOT the
/// Euclidean sum of squared coefficients.
///
/// For a pure grade-1 input the Clifford norm equals the Euclidean norm
/// (since only `x1² + x2² + x3²` is non-zero), so this also pins the
/// rotation-preserves-vector-length intuition.
#[test]
fn sandwich_preserves_clifford_norm() {
    fn clifford_norm_sq(x: &[f32; MV_DIM]) -> f32 {
        x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
            - x[4] * x[4]
            - x[5] * x[5]
            - x[6] * x[6]
            - x[7] * x[7]
    }

    let r = make_random_rotor(0x12345678_9ABCDEF0_u64);
    let v_in: [f32; 3] = [0.1, -0.4, 0.9];
    let mut mv = [0.0_f32; MV_DIM];
    mv[1] = v_in[0];
    mv[2] = v_in[1];
    mv[3] = v_in[2];
    let n_in = clifford_norm_sq(&mv);

    let rotated = rotor_sandwich(r, &mv);
    let n_out = clifford_norm_sq(&rotated);

    assert!(
        (n_in - n_out).abs() < 1e-5,
        "sandwich did not preserve Clifford norm²: in={n_in}, out={n_out}"
    );

    // For grade-1 input the Clifford norm equals the Euclidean norm.
    let euc: f32 = v_in.iter().map(|x| x * x).sum();
    assert!(
        (n_in - euc).abs() < 1e-6,
        "grade-1 input: Clifford norm² should equal Euclidean: {n_in} vs {euc}"
    );
}

/// Known-answer test: 90° rotation in e1-e2 plane should send e1 → ±e2.
///
/// Rotor R = cos(45°) + sin(45°)*e12 = (1/√2)(1 + e12).
/// Sandwich `R * e1 * R̃` analytically = -e2 (see derivation in code comment).
/// Output must be a pure grade-1 vector with the expected sign.
#[test]
fn sandwich_known_answer_90deg_e12_plane() {
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    let r: [f32; 4] = [inv_sqrt2, inv_sqrt2, 0.0, 0.0];

    let mut e1 = [0.0_f32; MV_DIM];
    e1[1] = 1.0;

    let rotated = rotor_sandwich(r, &e1);

    // expected = pure grade-1 vector at e2 only ([0, 0, -1, 0, 0, 0, 0, 0]);
    // grades 0, 2, 3 and the e1, e3 slots are all zero.
    // (basis: [1, e1, e2, e3, e12, e13, e23, e123] — indices 1/2/3 are all
    // grade-1, index 4 is grade-2 e12.)
    let expected = [0.0_f32, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (i, (&got, &want)) in rotated.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "rotated[{i}] = {got}, expected {want}"
        );
    }
}

/// Unit rotor identity: `R * R̃ = 1` (scalar 1, zero everywhere else).
#[test]
fn unit_rotor_times_reverse_is_one() {
    let r = make_random_rotor(0xFEED_F00D_BABE_CAFE);
    let r_mv = rotor_to_mv(r);
    let r_rev_mv = reverse(&r_mv);
    let prod = geometric_product(&r_mv, &r_rev_mv);
    // prod should be ≈ [1, 0, 0, 0, 0, 0, 0, 0]
    assert!(
        (prod[0] - 1.0).abs() < 1e-5,
        "scalar part of R*R̃ should be 1, got {}",
        prod[0]
    );
    for (i, v) in prod.iter().enumerate().skip(1) {
        assert!(
            v.abs() < 1e-5,
            "non-scalar part of R*R̃ should be 0 at idx {i}, got {v}"
        );
    }
}

/// In 3D, a unit-rotor sandwich on a pure grade-1 input produces a pure
/// grade-1 output (no grade-3 contamination beyond f32 roundoff).
///
/// This is the algebraic fact the rotor codec's storage layout rests on. The
/// codec embeds three real values as the grade-1 part and then quantises **all
/// eight** multivector components, so five of the eight stored codes describe
/// slots this test pins at zero. They are dead code budget, not reconstruction
/// error — see [`inverse_sandwich_of_non_grade1_leaks_nothing_into_grade1`] for
/// the decode-side half, and `crate::rotorquant` § "Effective bpe" for what the
/// waste costs.
#[test]
fn sandwich_of_grade1_in_3d_stays_grade1() {
    // Sweep rotors and grade-1 directions: a single (rotor, vector) pair could
    // land on a coincidence, and the storage claim is about every group of every
    // token of every layer.
    for seed in [
        0x9999_AAAA_BBBB_CCCC_u64,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_FFFF_FFFF,
        0x1234_5678_9ABC_DEF0,
        ROTORQUANT_GLOBAL_SEED,
    ] {
        let r = make_random_rotor(seed);
        for v in [
            [0.6_f32, -0.7, 0.4],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -3.25],
            [-0.001, 0.002, 0.003],
        ] {
            let mut v_mv = [0.0_f32; MV_DIM];
            v_mv[1] = v[0];
            v_mv[2] = v[1];
            v_mv[3] = v[2];

            let y = rotor_sandwich(r, &v_mv);
            // Genuinely relative: seeded at 0 so a small input gets a small
            // bound. Seeding the fold at 1.0 would turn this into an absolute
            // 1e-5 floor, which for the 1e-3-scale vector below admits a 0.3%
            // leak and costs that sweep entry all its power.
            let max_abs = v_mv.iter().fold(0.0_f32, |acc, x| acc.max(x.abs()));
            let tol = 1e-5 * max_abs;
            // Indices 0 (scalar), 4/5/6 (bivector) and 7 (pseudoscalar) — the
            // five slots the codec quantises and stores but decode discards.
            for idx in [0_usize, 4, 5, 6, 7] {
                assert!(
                    y[idx].abs() < tol,
                    "seed={seed:#x} v={v:?}: non-grade-1 leakage y[{idx}] = {} exceeds {tol}",
                    y[idx],
                );
            }
        }
    }
}

/// The decode-side half: quantisation noise parked in the five non-grade-1
/// slots contributes **nothing** to the reconstructed vector.
///
/// The codec quantises all eight components against one codebook, and no
/// Lloyd-Max centroid is exactly zero, so the five algebraically-zero slots
/// decode to non-zero values. Decode then applies the inverse sandwich
/// `R̃ * x * R` and reads indices 1, 2, 3. This test feeds the inverse sandwich a
/// multivector that is *only* those five slots and requires the grade-1 output
/// to be zero — i.e. the noise is discarded, not "cancelled back into grade-1".
///
/// Together with [`sandwich_of_grade1_in_3d_stays_grade1`] this is the whole
/// justification for calling those five codes dead budget: nothing goes in and
/// nothing comes out, so the 15 of 24 code bits they occupy at
/// `ROTOR3_BITS = 3` carry no information.
///
/// Mutation check: replace the inverse sandwich with any grade-mixing map (drop
/// the reverse on one side, say) and the grade-1 slots pick up the injected
/// values — RED.
#[test]
fn inverse_sandwich_of_non_grade1_leaks_nothing_into_grade1() {
    for seed in [
        0x9999_AAAA_BBBB_CCCC_u64,
        0x0BAD_C0DE_0BAD_C0DE,
        ROTORQUANT_GLOBAL_SEED,
    ] {
        let r = make_random_rotor(seed);
        // Only the slots the codec wastes: scalar, three bivectors, pseudoscalar.
        let noise = [0.83_f32, 0.0, 0.0, 0.0, -0.41, 1.27, -0.66, 0.19];
        // Decode's inverse rotation: `R̃ * x * R`.
        let back = rotor_sandwich(rotor_reverse(r), &noise);
        // Relative to the injected magnitude — see the sibling test's note on
        // why the fold is seeded at 0 rather than 1.
        let tol = 1e-5 * noise.iter().fold(0.0_f32, |acc, x| acc.max(x.abs()));
        for idx in [1_usize, 2, 3] {
            assert!(
                back[idx].abs() < tol,
                "seed={seed:#x}: non-grade-1 noise reached the reconstructed vector at \
                 index {idx} ({}) — the five wasted code slots would then be carrying \
                 signal after all",
                back[idx],
            );
        }
    }
}

/// `gp_rotor_mv(R, x)` must match `geometric_product(R_dense, x)` for any rotor.
#[test]
fn sparse_matches_dense_gp() {
    let r = make_random_rotor(0xAAAA_BBBB_CCCC_DDDD);
    let x: [f32; MV_DIM] = [0.7, -0.2, 0.5, 0.3, -0.1, 0.4, 0.6, -0.8];

    let dense = geometric_product(&rotor_to_mv(r), &x);
    let sparse = gp_rotor_mv(r, &x);
    for (i, (&a, &b)) in dense.iter().zip(sparse.iter()).enumerate() {
        assert!(
            approx_eq(a, b),
            "sparse vs dense GP mismatch at idx {i}: dense={a}, sparse={b}"
        );
    }
}

/// `gp_mv_rotor(x, R)` must match `geometric_product(x, R_dense)` for any rotor.
#[test]
fn sparse_mv_rotor_matches_dense() {
    use crate::clifford::gp_mv_rotor;
    let r = make_random_rotor(0xCAFE_BABE_DEAD_BEEF);
    let x: [f32; MV_DIM] = [0.7, -0.2, 0.5, 0.3, -0.1, 0.4, 0.6, -0.8];

    let dense = geometric_product(&x, &rotor_to_mv(r));
    let sparse = gp_mv_rotor(&x, r);
    for (i, (&a, &b)) in dense.iter().zip(sparse.iter()).enumerate() {
        assert!(
            approx_eq(a, b),
            "gp_mv_rotor vs dense mismatch at idx {i}: dense={a}, sparse={b}"
        );
    }
}

/// Sandwich via the sparse path equals the dense sandwich `R * x * R̃`.
#[test]
fn sandwich_sparse_matches_dense() {
    let r = make_random_rotor(0x1234_5678_9ABC_DEF0);
    let x: [f32; MV_DIM] = [0.7, -0.2, 0.5, 0.3, -0.1, 0.4, 0.6, -0.8];

    let r_mv = rotor_to_mv(r);
    let r_rev_mv = reverse(&r_mv);
    let tmp = geometric_product(&r_mv, &x);
    let dense = geometric_product(&tmp, &r_rev_mv);

    let sparse = rotor_sandwich(r, &x);
    for (i, (&a, &b)) in dense.iter().zip(sparse.iter()).enumerate() {
        assert!(
            approx_eq(a, b),
            "sandwich sparse vs dense mismatch at idx {i}: dense={a}, sparse={b}"
        );
    }
}

/// `rotor_seed` must be deterministic and decorrelate across (layer, head, group).
#[test]
fn rotor_seed_deterministic_and_decorrelated() {
    // Determinism.
    let a = rotor_seed(3, 5, 7);
    let b = rotor_seed(3, 5, 7);
    assert_eq!(a, b, "rotor_seed must be deterministic for the same inputs");
    // Non-trivial dependency on each axis.
    let a0 = rotor_seed(0, 0, 0);
    let a_layer = rotor_seed(1, 0, 0);
    let a_head = rotor_seed(0, 1, 0);
    let a_group = rotor_seed(0, 0, 1);
    assert_ne!(a0, a_layer);
    assert_ne!(a0, a_head);
    assert_ne!(a0, a_group);
    // Verify the base mixes the global seed.
    assert_eq!(a0, ROTORQUANT_GLOBAL_SEED);
}

/// `make_rotor_table` produces `n_groups * 4` f32s with each 4-tuple unit.
#[test]
fn rotor_table_is_unit_per_group() {
    let n_groups = 16;
    let table = make_rotor_table(2, 3, n_groups);
    assert_eq!(table.len(), n_groups * 4);
    for g in 0..n_groups {
        let r = &table[g * 4..g * 4 + 4];
        let n2 = norm_sq(r);
        assert!(
            (n2 - 1.0).abs() < 1e-5,
            "rotor group {g} norm² = {n2}, expected ≈ 1.0"
        );
    }
}
