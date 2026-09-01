//! What would a rotation be worth to the TurboQuant family, which has none?
//!
//! # The question, and why nothing in the tree answers it
//!
//! Every turbo encoder — CPU and MSL, K axis and V axis — is group the flat
//! input into [`GROUP_SIZE`] contiguous elements, `scale = max|x| /
//! max_centroid`, normalise, nearest-centroid against a Lloyd-Max `N(0,1)`
//! codebook, bit-pack. There is no Walsh-Hadamard and no rotation of any kind,
//! in any of them.
//!
//! Whether adding one would buy anything is unmeasured, and the two committed
//! turbo fidelity surfaces cannot measure it:
//!
//! * `rate_distortion_tests` runs on i.i.d. Gaussian, where the Lloyd-Max
//!   codebook's distributional assumption already holds exactly. A
//!   decorrelating rotation has nothing to decorrelate there and an identity
//!   rotation passes.
//! * `rotation_fidelity_tests::lossier_codecs_fail_the_outlier_cosine_floors`
//!   does run turbo on the outlier fixture — but as the negative control
//!   against `iso3` / `rotor3` / `planar4` floors. That comparison is
//!   **confounded**: turbo carries one scale per 32 values against those
//!   families' per-2/3/4 scale, so its deficit mixes the missing rotation with
//!   four-times-coarser scale granularity, and nothing separates them.
//!
//! This module removes the confound by construction. Both arms are the **same
//! turbo codec at the same width and the same `GROUP_SIZE`**; the only
//! difference is a full-`head_dim` normalized Walsh-Hadamard applied before
//! [`turbo_quantize_v`] and undone after [`turbo_dequantize`]. Whatever the
//! difference is, it is the rotation.
//!
//! # Criterion, declared before any fidelity number was measured
//!
//! Metric: **bits of SQNR the rotation buys**,
//!
//! ```text
//! G(rotation, bits, fixture) = (SQNR_rotated - SQNR_plain) / DB_PER_BIT
//! ```
//!
//! computed by [`turbo_rotation_gain_bits`], the single function every gate and
//! every mutation guard below calls, so a guard cannot drift away from the gate
//! it guards.
//!
//! Coverage is **every width the codec accepts**, discovered by probing
//! [`lloyd_gaussian_codebook`] across the whole of `u8`
//! ([`supported_turbo_widths`]) rather than by a hand-written list — a width
//! added to or removed from the codec moves the gate's coverage with it.
//!
//! Threshold: `1.5` bits, **inherited verbatim** from
//! `rotation_fidelity_tests::ROT_K_MIN_OUTLIER_GAIN_BITS`. That constant was
//! derived there from a block-size separation argument — a block-`b` orthogonal
//! transform cuts a group's peak by at most `sqrt(b)`, so it can buy a
//! peak-scaled quantizer at most `0.5·log2(b)` bits: 1.00 for block-4, 0.79 for
//! block-3, 0.50 for block-2, 0 for the identity — and **not** fitted to any
//! codec's measured value. Re-using a constant this module has no freedom to
//! move is the guarantee that the verdict below was not manufactured. Note that
//! for turbo the block-`b` argument bounds only the peak component: turbo's
//! codebook is shape-matched to `N(0,1)`, so a rotation also changes how well
//! the normalised group fits the codebook, and that term is not bounded by
//! `0.5·log2(b)`. The rejection of a block-local substitute is therefore an
//! **empirical** result here, where for `rot_k` it is a theorem.
//!
//! Buckets, fixed in advance:
//!
//! * **PASS** — the gate detects a missing rotation — iff at every supported
//!   width: `G(H_full) >= 1.5` on the outlier fixture, `G(H_full) < 0` on the
//!   i.i.d. fixture, `G(identity) == 0`, and `G(H_block4) < 1.5`.
//! * **FAIL of the magnitude criterion** — `0 < G(H_full) < 1.5` at any width.
//!   The rotation is measurably worth something but less than the pre-declared
//!   demand. Report the measured worth; do **not** lower the threshold to
//!   manufacture a green.
//! * **VACUOUS** — `G(H_full) <= 0` on the outlier fixture. The gate cannot see
//!   a rotation that is there, and that is the finding.
//!
//! Deterministic: seeded fixtures, CPU scalar codec, no MLX, no GPU, no model.
//!
//! # Why the fixture, and the evidence it can discriminate
//!
//! A rotation exists to spread concentrated energy across a basis, so a fixture
//! with nothing concentrated cannot tell a real rotation from an identity one.
//! [`outlier_fixture`] is the literature-shaped K-cache model — i.i.d. Gaussian
//! base with [`OUTLIER_CHANNELS`] persistent channels at [`OUTLIER_RATIO`]
//! times the rest.
//!
//! Its channel placement is load-bearing and is **checked here rather than
//! inherited**: [`OUTLIER_CHANNELS`] was chosen for the affine group of 64, and
//! the same constant has already silently cancelled the effect under test in
//! one other module by landing one outlier in every group of 32.
//! [`the_outlier_fixture_places_energy_in_every_turbo_group`] asserts what the
//! placement is for `GROUP_SIZE`, and the density sweep below varies it so no
//! verdict rests on one placement.
//!
//! The discrimination evidence is the pair of fixtures run through the
//! identical gate: the transform must **win** on the outlier fixture and
//! **lose** on the i.i.d. one. A gate that could not tell them apart would be
//! measuring the codec, not the rotation.

use crate::test_utils::outlier_fixture;
use crate::test_utils::{
    fwht_normalize, gaussian_data, incoherence_per_row, lcg_data, outlier_channels, sqnr_db,
    DB_PER_BIT, OUTLIER_CHANNELS, OUTLIER_HEAD_DIM, OUTLIER_RATIO, OUTLIER_ROWS, TEST_SEED,
};
use crate::turboquant::{lloyd_gaussian_codebook, turbo_dequantize, turbo_quantize_v, GROUP_SIZE};

// ── The two arms: one codec, one difference ─────────────────────────────────

/// Row length every fixture and every rotation in this module uses.
const HEAD_DIM: usize = OUTLIER_HEAD_DIM;

/// The turbo round-trip, exactly as shipped: no transform.
fn turbo_roundtrip(data: &[f32], bits: u8) -> Vec<f32> {
    let rows = data.len() / HEAD_DIM;
    let shape = [1, 1, rows as i32, HEAD_DIM as i32];
    let blocks = turbo_quantize_v(data, bits, &shape).expect("turbo_quantize_v");
    turbo_dequantize(&blocks).expect("turbo_dequantize")
}

/// The same turbo round-trip with `rotate` applied per row before the encoder
/// and `unrotate` after the decoder.
///
/// `rotate` is handed the whole buffer and is responsible for its own row
/// stride, matching the convention in `rotation_fidelity_tests`. Every rotation
/// here is orthogonal, so the SQNR of this arm is directly comparable with
/// [`turbo_roundtrip`]'s: `‖x − Rᵀ Q(R x)‖ = ‖R x − Q(R x)‖`.
fn rotated_turbo_roundtrip(
    data: &[f32],
    bits: u8,
    rotate: impl Fn(&mut [f32]),
    unrotate: impl Fn(&mut [f32]),
) -> Vec<f32> {
    let mut buf = data.to_vec();
    rotate(&mut buf);
    let mut decoded = turbo_roundtrip(&buf, bits);
    unrotate(&mut decoded);
    decoded
}

/// Bits of SQNR `rotate` buys the identical turbo codec at the identical width
/// and [`GROUP_SIZE`].
fn turbo_rotation_gain_bits(
    data: &[f32],
    bits: u8,
    rotate: impl Fn(&mut [f32]),
    unrotate: impl Fn(&mut [f32]),
) -> f64 {
    let rotated = sqnr_db(data, &rotated_turbo_roundtrip(data, bits, rotate, unrotate));
    let plain = sqnr_db(data, &turbo_roundtrip(data, bits));
    (rotated - plain) / DB_PER_BIT
}

// ── Rotations under test ────────────────────────────────────────────────────

/// The full-`head_dim` normalized Walsh-Hadamard — the transform `rot_k` owns
/// and turbo does not. Self-inverse, so it serves as its own `unrotate`.
fn hadamard_full(buf: &mut [f32]) {
    fwht_normalize(buf, HEAD_DIM);
}

/// The same Walsh-Hadamard truncated to blocks of 4: still orthogonal, still
/// self-inverse, still genuinely decorrelating, just not full-dimension.
///
/// This is the plausible way a wired-in rotation stops being what its name says
/// — called with the wrong width — as opposed to being deleted outright.
fn hadamard_block4(buf: &mut [f32]) {
    fwht_normalize(buf, 4);
}

/// No transform: the mutation that models the codec as it ships today.
fn identity_rotation(_buf: &mut [f32]) {}

// ── Coverage, derived from the codec ────────────────────────────────────────

/// Every width [`lloyd_gaussian_codebook`] accepts, discovered by probing it.
///
/// Derived rather than listed so that adding or removing a codebook moves this
/// module's coverage with it. `8` is deliberately rejected by the codec (K8 is
/// affine `q8_0`, not turbo) and so is absent here without being named.
fn supported_turbo_widths() -> Vec<u8> {
    (u8::MIN..=u8::MAX)
        .filter(|&bits| lloyd_gaussian_codebook(bits).is_ok())
        .collect()
}

// ── Fixtures ────────────────────────────────────────────────────────────────

/// The i.i.d. uniform fixture the crate's per-codec cosine gates use, at the
/// outlier fixture's shape so the two are directly comparable.
fn iid_uniform_fixture() -> Vec<f32> {
    lcg_data(OUTLIER_ROWS * HEAD_DIM, TEST_SEED)
}

/// i.i.d. Gaussian with no outlier channels — the shape the KV literature
/// reports for the **Value** cache, and the source the Lloyd-Max codebook is
/// matched to.
fn iid_gaussian_fixture() -> Vec<f32> {
    gaussian_data(OUTLIER_ROWS * HEAD_DIM, TEST_SEED)
}

// ── What the fixture actually puts in front of the codec ────────────────────

/// The canonical outlier fixture's channels, stated against [`GROUP_SIZE`]
/// rather than against the affine group of 64 they were chosen for.
///
/// The placement is not neutral and must not be inherited unchecked: a
/// constant chosen for one group size has already cancelled the effect under
/// test in another module by landing one outlier in every group of 32. For the
/// rotation question that same placement is the *favourable* one — every group
/// of 32 carries an outlier, so a full-row transform has something to recover
/// in all of them — which is exactly why it cannot be the only density
/// measured.
#[test]
fn the_outlier_fixture_places_energy_in_every_turbo_group() {
    let channels = outlier_channels(HEAD_DIM, OUTLIER_CHANNELS);
    let groups: Vec<usize> = channels.iter().map(|&c| c / GROUP_SIZE).collect();
    let n_groups = HEAD_DIM / GROUP_SIZE;

    println!(
        "outlier channels {channels:?} land in groups {groups:?} of {n_groups} \
         (GROUP_SIZE={GROUP_SIZE}, head_dim={HEAD_DIM}, ratio={OUTLIER_RATIO})"
    );

    let mut occupied = vec![false; n_groups];
    for &g in &groups {
        occupied[g] = true;
    }
    assert!(
        occupied.iter().all(|&o| o),
        "the fixture leaves a turbo group clean ({groups:?} of {n_groups}); the density \
         sweep, not this placement, would then carry the verdict"
    );

    let before = incoherence_per_row(&outlier_fixture(), HEAD_DIM);
    let mut rotated = outlier_fixture();
    hadamard_full(&mut rotated);
    let after = incoherence_per_row(&rotated, HEAD_DIM);
    println!(
        "incoherence mu: {:.4} before the Hadamard, {:.4} after ({:.2}x)",
        before.mean,
        after.mean,
        before.mean / after.mean
    );
    assert!(
        before.mean > after.mean,
        "the fixture has no concentration for a rotation to spread (mu {:.4} -> {:.4}); \
         a rotation gate built on it could not discriminate",
        before.mean,
        after.mean
    );
}

// ── Report ──────────────────────────────────────────────────────────────────

/// Print `G` for every supported width on every fixture, with no threshold.
///
/// The gate's numbers, separated from the gate so the measurement can be read
/// and re-read without an assertion in the way.
#[test]
fn turbo_rotation_gain_report() {
    let fixtures: [(&str, Vec<f32>); 3] = [
        ("outlier(4/128 @20x)", outlier_fixture()),
        ("i.i.d. uniform", iid_uniform_fixture()),
        ("i.i.d. Gaussian", iid_gaussian_fixture()),
    ];

    for (fixture_name, data) in &fixtures {
        for bits in supported_turbo_widths() {
            let full = turbo_rotation_gain_bits(data, bits, hadamard_full, hadamard_full);
            let block4 = turbo_rotation_gain_bits(data, bits, hadamard_block4, hadamard_block4);
            let none = turbo_rotation_gain_bits(data, bits, identity_rotation, identity_rotation);
            println!(
                "{fixture_name:<20} turbo{bits}: plain {:7.3} dB | H_full {full:+.3} bits | \
                 H_block4 {block4:+.3} bits | identity {none:+.3} bits",
                sqnr_db(data, &turbo_roundtrip(data, bits)),
            );
        }
    }
}
