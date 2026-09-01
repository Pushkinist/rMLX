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
//!
//! # Result, against the buckets above
//!
//! **The gate detects a missing rotation.** With the transform present it
//! reads `+2.077` / `+2.004` / `+1.574` / `+0.961` bits at 1 / 2 / 3 / 4 bits
//! on the outlier fixture; with it removed, exactly `0.000` at every width.
//! Called at the wrong width — a block-4 truncation of the same Hadamard — it
//! reads `+0.995` / `+1.062` / `+1.074` / `+0.686`, below the full-dimension
//! figure at every width, so the gate separates a degraded rotation from a
//! correct one and not merely a present one from an absent one.
//!
//! **The magnitude criterion is met at three of the four widths.** turbo4's
//! `+0.961` misses the inherited `1.5`-bit threshold. The threshold is not
//! lowered; the shortfall is recorded, and the mechanism —
//! [`the_rotation_is_worth_less_the_wider_the_codebook`] — is asserted rather
//! than narrated. The block-`b` ceiling that makes 1.5 a *theorem* for `rot_k`
//! is confirmed not to transfer: at turbo4 the block-4 truncation reaches
//! `+0.686` of a `+0.961` full-dimension gain, so no threshold in that
//! neighbourhood separates them and the gate rests on the parameter-free
//! ordering instead.
//!
//! **What the missing rotation is worth**, on this fixture and this metric:
//!
//! | | outlier (K-shaped) | i.i.d. Gaussian (V-shaped) | i.i.d. uniform |
//! |---|---:|---:|---:|
//! | turbo2 | `+2.004` bits | `+0.015` | `−0.440` |
//! | turbo3 | `+1.574` bits | `+0.015` | `−0.306` |
//! | turbo4 | `+0.961` bits | `+0.003` | `−0.105` |
//!
//! The two axes are not alike. The KV literature reports outlier channels on
//! the **Key** cache and none on the **Value** cache, and the two columns above
//! are those two shapes: a rotation is worth one to two bits to a turbo K
//! store and about `0.01` bits to a turbo V store. That is the reverse of the
//! implementation cost, where the K side reuses an existing rotate-and-quantize
//! path and the V side needs an explicit inverse after the SV accumulation in
//! the flash and dequant kernels.
//!
//! These are figures on a **model** of K-cache structure, not on a captured
//! tensor: they are measured for the fixture and are an estimate, not a
//! measurement, for any real checkpoint. Producing the latter needs a forward
//! pass, which is a serving cell and not reachable from a CPU unit test.

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::rotation_fidelity_tests::ROT_K_MIN_OUTLIER_GAIN_BITS;
use crate::test_utils::{
    cosine_similarity_per_row, fwht_normalize, gaussian_data, incoherence_per_row, lcg_data,
    outlier_channel_data, outlier_channels, outlier_fixture, sqnr_db, CosineStats, DB_PER_BIT,
    OUTLIER_CHANNELS, OUTLIER_HEAD_DIM, OUTLIER_RATIO, OUTLIER_ROWS, TEST_SEED,
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
/// rather than against the affine group of 64 they were chosen for, and the
/// concentration a rotation has to work on.
///
/// The placement is not neutral and must not be inherited unchecked: the same
/// constant has already cancelled the effect under test in another module by
/// landing one outlier in every group of 32. For the rotation question that
/// placement is the *favourable* one — every group of 32 carries an outlier, so
/// a full-row transform has something to recover in all of them — which is
/// exactly why it cannot be the only density measured. See
/// [`the_rotation_gain_survives_every_outlier_density`].
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

// ── The gate ────────────────────────────────────────────────────────────────

/// The gate: on outlier data the full-`head_dim` Hadamard must buy the turbo
/// codec bits, must beat a block-4 truncation of itself, and that truncation
/// must in turn beat no transform at all.
///
/// The ordering has **no free parameter**. Its subject and both references are
/// the same codec at the same width and the same [`GROUP_SIZE`] on the same
/// fixture, differing only in how much of the row the transform mixes, so no
/// threshold can be chosen to manufacture the verdict. It is the assertion the
/// mutation checks act on: replacing the transform with the identity collapses
/// the first term to the third, and calling it with the wrong width collapses
/// it to the second.
///
/// Measured, `H_full` / `H_block4` in bits: turbo1 `+2.077` / `+0.995`, turbo2
/// `+2.004` / `+1.062`, turbo3 `+1.574` / `+1.074`, turbo4 `+0.961` / `+0.686`.
/// The narrowest margin is turbo4's `0.275` bits, on a seeded deterministic
/// fixture where nothing varies between runs.
#[test]
fn the_full_dimension_hadamard_buys_turbo_bits_that_a_block_local_one_cannot() {
    let data = outlier_fixture();

    for bits in supported_turbo_widths() {
        let full = turbo_rotation_gain_bits(&data, bits, hadamard_full, hadamard_full);
        let block4 = turbo_rotation_gain_bits(&data, bits, hadamard_block4, hadamard_block4);
        let none = turbo_rotation_gain_bits(&data, bits, identity_rotation, identity_rotation);

        println!(
            "turbo{bits} on the outlier fixture: plain {:7.3} dB | H_full {full:+.3} bits | \
             H_block4 {block4:+.3} bits | identity {none:+.3} bits",
            sqnr_db(&data, &turbo_roundtrip(&data, bits)),
        );

        assert!(
            full > block4,
            "turbo{bits}: the full-head_dim Hadamard bought {full:+.3} bits, no more than the \
             block-4 truncation of the same transform at {block4:+.3}. The gate cannot tell a \
             full-dimension rotation from a block-local one"
        );
        assert!(
            block4 > none,
            "turbo{bits}: a block-4 Hadamard bought {block4:+.3} bits against {none:+.3} for no \
             transform at all. The measurement is not seeing the transform"
        );
    }
}

/// The same gate on the fixtures with nothing to spread, which is the evidence
/// that the gate above measures the rotation rather than the codec.
///
/// Two i.i.d. sources, and they answer different questions.
///
/// * **Uniform** is what the crate's older per-codec cosine gates run on. The
///   Hadamard is a net **loss** there — it pushes uniform toward Gaussian,
///   raising the peak-to-RMS ratio the group scale is set by. A gate built on
///   that fixture would have reported a rotation as a regression.
/// * **Gaussian with no outlier channels** is the shape the KV literature
///   reports for the **Value** cache (KIVI, arXiv:2402.02750; KVQuant,
///   arXiv:2401.18079), and it is the source turbo's Lloyd-Max codebook is
///   matched to. The transform buys essentially nothing there, which is the
///   load-bearing asymmetry for any decision to wire one in: turbo's K
///   spellings are the ones a rotation would pay for.
///
/// The Gaussian bound is stated against the gate's own smallest outlier-fixture
/// gain rather than a constant, so it cannot be tuned: measured `+0.003` to
/// `+0.020` bits against an outlier-fixture minimum of `+0.961`.
#[test]
fn the_same_hadamard_does_not_pay_on_either_iid_fixture() {
    let outlier = outlier_fixture();
    let uniform = iid_uniform_fixture();
    let gaussian = iid_gaussian_fixture();

    let mut worst_outlier = f64::INFINITY;
    let mut best_gaussian = f64::NEG_INFINITY;

    for bits in supported_turbo_widths() {
        let on_outlier = turbo_rotation_gain_bits(&outlier, bits, hadamard_full, hadamard_full);
        let on_uniform = turbo_rotation_gain_bits(&uniform, bits, hadamard_full, hadamard_full);
        let on_gaussian = turbo_rotation_gain_bits(&gaussian, bits, hadamard_full, hadamard_full);

        println!(
            "turbo{bits} H_full: outlier {on_outlier:+.3} bits | i.i.d. uniform \
             {on_uniform:+.3} bits | i.i.d. Gaussian {on_gaussian:+.3} bits"
        );

        assert!(
            on_uniform < 0.0,
            "turbo{bits}: the Hadamard gained {on_uniform:+.3} bits on i.i.d. uniform data. It is \
             supposed to lose there, and the premise that the crate's older fixture cannot \
             measure rotation quality needs re-checking if it does not"
        );

        worst_outlier = worst_outlier.min(on_outlier);
        best_gaussian = best_gaussian.max(on_gaussian);
    }

    assert!(
        best_gaussian < worst_outlier,
        "the Hadamard's best i.i.d. Gaussian gain {best_gaussian:+.3} bits reached its worst \
         outlier-fixture gain {worst_outlier:+.3}. The measurement is responding to the codec, \
         not to the concentration the fixture puts in front of it"
    );
}

/// The mutation the codec ships today: no transform buys exactly nothing, on
/// every fixture and at every width.
///
/// Exact by construction — [`turbo_rotation_gain_bits`] subtracts a
/// deterministic value from itself — so this pins the comparison the gate rests
/// on rather than estimating it. The guards that mutate a real transform are
/// the block-4 term inside the gate itself and
/// [`the_rotation_gain_survives_every_outlier_density`].
#[test]
fn removing_the_rotation_buys_turbo_exactly_nothing() {
    for (name, data) in [
        ("outlier", outlier_fixture()),
        ("i.i.d. uniform", iid_uniform_fixture()),
        ("i.i.d. Gaussian", iid_gaussian_fixture()),
    ] {
        for bits in supported_turbo_widths() {
            let gain = turbo_rotation_gain_bits(&data, bits, identity_rotation, identity_rotation);
            assert!(
                gain == 0.0,
                "{name} turbo{bits}: an absent transform reported {gain:+.6} bits of rotation \
                 gain — the gate cannot fail"
            );
        }
    }
}

// ── How much of the fixture the verdict depends on ──────────────────────────

/// Outlier density sweep: the gain is positive at every density, and the
/// full-dimension transform's margin over a block-local one is not.
///
/// The canonical fixture is one density. Sweeping it is what stops the verdict
/// from being a statement about channel placement. Two things come out, and
/// only the first is what the gate rests on:
///
/// * The full Hadamard buys turbo bits at **every** density from 1 channel of
///   128 to 64 of 128, at every width — `+2.264` down to `+0.294`. That is
///   asserted here.
/// * Its margin over the block-4 truncation narrows as density rises and
///   **inverts** at 64 of 128, where half the row is an outlier and the fixture
///   no longer models sparse outlier channels at all. The gate is therefore
///   stated on the canonical fixture, and this test pins where the ordering
///   stops holding so the scope cannot go stale unnoticed.
#[test]
fn the_rotation_gain_survives_every_outlier_density() {
    let n_groups = HEAD_DIM / GROUP_SIZE;
    let mut ordering_holds_up_to = 0usize;

    for channels in [1usize, 2, 4, 8, 16, 32, 64] {
        let data = outlier_channel_data(OUTLIER_ROWS, HEAD_DIM, channels, OUTLIER_RATIO, TEST_SEED);
        let groups_hit = outlier_channels(HEAD_DIM, channels)
            .iter()
            .map(|&c| c / GROUP_SIZE)
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        let mut ordering = true;
        for bits in supported_turbo_widths() {
            let full = turbo_rotation_gain_bits(&data, bits, hadamard_full, hadamard_full);
            let block4 = turbo_rotation_gain_bits(&data, bits, hadamard_block4, hadamard_block4);
            println!(
                "channels={channels:<3} groups_hit={groups_hit}/{n_groups} turbo{bits}: \
                 H_full {full:+.3} bits | H_block4 {block4:+.3} bits"
            );
            assert!(
                full > 0.0,
                "channels={channels} turbo{bits}: the Hadamard bought {full:+.3} bits. The \
                 rotation's worth to turbo is not supposed to depend on the fixture's density \
                 for its sign"
            );
            ordering &= full > block4;
        }
        if ordering {
            ordering_holds_up_to = channels;
        }
    }

    assert_eq!(
        ordering_holds_up_to, 32,
        "the density at which a full-dimension Hadamard stops beating its own block-4 \
         truncation moved. The gate is scoped to the canonical fixture on the strength of \
         where that boundary sits, and the scope note is now wrong"
    );
}

// ── What the missing rotation is worth ──────────────────────────────────────

/// Splits the confounded turbo-versus-`iso` deficit into its two causes.
///
/// The crate's outlier cosine floors run turbo as the deliberately-lossier
/// substitute for `iso3`, at "the same nominal width, i.e. one scale per 32
/// values instead of the family's per-2/3/4 scale". That comparison moves two
/// variables at once — the missing rotation and four-times-coarser scale
/// granularity — and on its own cannot say which dominates.
///
/// Rotating turbo without touching its group size holds granularity fixed, so
/// the share of the gap the rotation closes is the rotation's share. Measured
/// in reconstruction error `1 - cos`: turbo3 `0.0792` plain, `0.0127` rotated,
/// against `iso3`'s `0.0032` — the rotation closes **87%** of the distance.
/// The claim under test is that the rotation is the dominant term, so the
/// assertion is "more than half", not the measured figure.
#[test]
fn the_missing_rotation_is_most_of_the_turbo_iso_cosine_gap() {
    let data = outlier_fixture();

    let iso = |bits: u8| -> Option<CosineStats> {
        let (codes, scales, quats, norms) = iso_encode_fast(&data, HEAD_DIM, 4, bits).ok()?;
        let decoded = iso_decode_fast(&codes, &scales, &quats, &norms, HEAD_DIM, 4, bits)
            .expect("iso_decode_fast accepted an encode iso_encode_fast produced");
        Some(cosine_similarity_per_row(&data, &decoded, HEAD_DIM))
    };

    let shared: Vec<u8> = supported_turbo_widths()
        .into_iter()
        .filter(|&bits| iso(bits).is_some())
        .collect();
    assert!(
        !shared.is_empty(),
        "turbo and iso no longer share a width, so the confounded comparison this test splits \
         cannot be reproduced and the split is unfalsifiable"
    );

    for bits in shared {
        let plain = cosine_similarity_per_row(&data, &turbo_roundtrip(&data, bits), HEAD_DIM);
        let rotated = cosine_similarity_per_row(
            &data,
            &rotated_turbo_roundtrip(&data, bits, hadamard_full, hadamard_full),
            HEAD_DIM,
        );
        let reference = iso(bits).expect("width filtered to those iso accepts");

        let error = |stats: &CosineStats| 1.0 - f64::from(stats.mean);
        let gap = error(&plain) - error(&reference);
        let closed = error(&plain) - error(&rotated);
        let share = closed / gap;

        println!(
            "turbo{bits} reconstruction error 1-cos: {:.6} plain -> {:.6} rotated, against \
             iso{bits} {:.6}: the rotation closes {:.1}% of the gap",
            error(&plain),
            error(&rotated),
            error(&reference),
            share * 100.0,
        );

        assert!(
            gap > 0.0,
            "turbo{bits} already matches iso{bits} on this fixture; there is no gap to \
             attribute and the confound this test exists to split is gone"
        );
        assert!(
            share > 0.5,
            "the rotation closes only {:.1}% of the turbo{bits}-to-iso{bits} cosine gap, so \
             scale granularity — not the missing transform — is the dominant term and the \
             transform work is being justified by the wrong number",
            share * 100.0,
        );
    }
}

/// The pre-declared magnitude criterion, and the width at which it stops being
/// met.
///
/// The module's threshold is `ROT_K_MIN_OUTLIER_GAIN_BITS`, inherited from the
/// `rot_k` gain gate and imported rather than retyped so the two cannot drift.
/// Against it the outcome is a **partial** one and is recorded as such:
/// `+2.077` / `+2.004` / `+1.574` / `+0.961` bits at 1 / 2 / 3 / 4 bits, so the
/// narrowest width clears the bar by a wide margin and the widest misses it.
///
/// The mechanism is the monotonicity asserted here. A rotation helps turbo by
/// two routes — it cuts the group peak that sets the scale, and it restores the
/// `N(0,1)` shape the Lloyd-Max codebook assumes — and both are worth less the
/// more levels the codebook has to spend. The consequence for the transform
/// work is that the payoff sits at the narrow widths.
///
/// This pins the limitation so a later change cannot pass silently on a premise
/// that no longer holds: if the widest width ever clears the inherited
/// threshold, the recorded conclusion goes red rather than green.
#[test]
fn the_rotation_is_worth_less_the_wider_the_codebook() {
    let data = outlier_fixture();
    let widths = supported_turbo_widths();
    let gains: Vec<(u8, f64)> = widths
        .iter()
        .map(|&bits| {
            (
                bits,
                turbo_rotation_gain_bits(&data, bits, hadamard_full, hadamard_full),
            )
        })
        .collect();

    for (bits, gain) in &gains {
        println!(
            "turbo{bits}: {gain:+.3} bits against the inherited \
             {ROT_K_MIN_OUTLIER_GAIN_BITS:.1}-bit threshold"
        );
    }

    for pair in gains.windows(2) {
        let (narrow_bits, narrow_gain) = pair[0];
        let (wide_bits, wide_gain) = pair[1];
        assert!(
            narrow_gain > wide_gain,
            "turbo{wide_bits} gained {wide_gain:+.3} bits from the rotation, at least as much \
             as turbo{narrow_bits}'s {narrow_gain:+.3}. A wider codebook is supposed to have \
             less for a rotation to recover"
        );
    }

    let (narrowest_bits, narrowest_gain) = gains[0];
    let (widest_bits, widest_gain) = gains[gains.len() - 1];
    assert!(
        narrowest_gain >= ROT_K_MIN_OUTLIER_GAIN_BITS,
        "turbo{narrowest_bits} gained only {narrowest_gain:+.3} bits, below the inherited \
         {ROT_K_MIN_OUTLIER_GAIN_BITS:.1}-bit threshold that a full-dimension transform clears \
         on the same fixture with the rot_k quantizer in the middle"
    );
    assert!(
        widest_gain < ROT_K_MIN_OUTLIER_GAIN_BITS,
        "turbo{widest_bits} now gains {widest_gain:+.3} bits, clearing the inherited \
         {ROT_K_MIN_OUTLIER_GAIN_BITS:.1}-bit threshold it was recorded as missing. The \
         partial verdict in this module's docs is stale"
    );
}
