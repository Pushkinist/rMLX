//! Algebra correctness tests for [`crate::clifford`].
//!
//! Covers:
//!   * Reverse identity `(R̃)̃ == R` — both compact and dense forms.
//!   * Sandwich identity: `R * x * R̃` preserves vector norm for unit rotors.
//!   * Sparse vs dense GP equivalence (`gp_rotor_mv` == `geometric_product`).
//!   * Random rotor is unit: `||R||² ≈ 1`.

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
/// This is the property the encode/decode roundtrip relies on — quantising
/// the grade-1 and grade-3 slots and then inverse-sandwiching reconstructs
/// the original vector accurately because the grade-3 quantisation noise
/// gets cancelled by the inverse rotation back into grade-1.
#[test]
fn sandwich_of_grade1_in_3d_stays_grade1() {
    let r = make_random_rotor(0x9999_AAAA_BBBB_CCCC);
    let mut v_mv = [0.0_f32; MV_DIM];
    v_mv[1] = 0.6;
    v_mv[2] = -0.7;
    v_mv[3] = 0.4;

    let y = rotor_sandwich(r, &v_mv);
    // Grade-3 component must be ~0 (within f32 roundoff for the table-driven
    // GP — typical magnitude < 1e-6 here).
    assert!(
        y[7].abs() < 1e-5,
        "grade-3 contamination y[7] = {} should be ~0 for grade-1 input",
        y[7]
    );
    // Grades 0, 2 (bivector) must also be ~0.
    assert!(y[0].abs() < 1e-5, "scalar leakage y[0] = {}", y[0]);
    for (i, &v) in y.iter().enumerate().take(7).skip(4) {
        assert!(v.abs() < 1e-5, "bivector leakage y[{i}] = {v}");
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
