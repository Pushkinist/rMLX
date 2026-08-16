use super::{
    cosine_similarity_per_row, fwht_normalize, gaussian_data, incoherence_per_row, lcg_data,
    lloyd_max_anchor_db, outlier_channel_data, outlier_channels, outlier_fixture,
    skip_if_no_gpu_env, skip_value_means_skip, sqnr_db, vectorized_parity_check, wasted_bits,
    DB_PER_BIT, LLOYD_MAX_GAUSSIAN_SQNR_DB, OUTLIER_HEAD_DIM, OUTLIER_ROWS, TEST_SEED,
};

/// Identity codec (cpu == msl): must pass with zero error.
#[test]
fn identity_codec_parity_check_passes() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    vectorized_parity_check(<[f32]>::to_vec, <[f32]>::to_vec, &input, 1e-7, "identity");
}

/// Codec with constant offset: must pass when offset ≤ tol.
#[test]
fn parity_check_passes_within_tol() {
    let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let tol = 1e-4_f32;
    let tiny_offset = tol * 0.5;
    vectorized_parity_check(
        <[f32]>::to_vec,
        |x| x.iter().map(|&v| v + tiny_offset).collect(),
        &input,
        tol,
        "tiny-offset",
    );
}

/// Codec with offset exceeding tol: must panic.
#[test]
#[should_panic(expected = "CPU vs MSL max-abs-error")]
fn parity_check_fails_beyond_tol() {
    let input: Vec<f32> = vec![1.0; 16];
    vectorized_parity_check(
        <[f32]>::to_vec,
        |x| x.iter().map(|&v| v + 1.0).collect(),
        &input,
        1e-3,
        "large-offset",
    );
}

/// Length mismatch: must panic.
#[test]
#[should_panic(expected = "different lengths")]
fn parity_check_fails_on_length_mismatch() {
    let input: Vec<f32> = vec![0.5; 8];
    vectorized_parity_check(
        <[f32]>::to_vec,
        |x| x[..4].to_vec(),
        &input,
        1e-3,
        "length-mismatch",
    );
}

/// `RMLX_SKIP_GPU` strict membership: only `"1"` means skip.
///
/// Asserted against the pure `skip_value_means_skip` rather than by writing the
/// process environment. Writing it would be unsound at any parallelism: the
/// reader `skip_if_no_gpu_env()` runs at the top of every `#[ignore]`d GPU test
/// in this binary and none of them take the env lock, so a transient `"1"` can
/// silently skip a live GPU test (a false green) and a transient `"0"` can
/// un-ignore a Metal test into a parallel run. The membership rule is pure, so
/// nothing is lost by testing it as such.
#[test]
fn skip_value_means_skip_is_strict_membership() {
    assert!(
        skip_value_means_skip(Some("1")),
        "RMLX_SKIP_GPU=1 must mean skip"
    );
    assert!(
        !skip_value_means_skip(Some("0")),
        "RMLX_SKIP_GPU=0 must not mean skip"
    );
    assert!(
        !skip_value_means_skip(Some("true")),
        "RMLX_SKIP_GPU=true must not mean skip"
    );
    assert!(
        !skip_value_means_skip(Some("")),
        "RMLX_SKIP_GPU= (empty) must not mean skip"
    );
    assert!(
        !skip_value_means_skip(None),
        "RMLX_SKIP_GPU unset must not mean skip"
    );
}

/// The env reader is wired to the membership rule above, not a second copy of
/// it: whatever the process environment currently holds, the two agree.
#[test]
fn skip_if_no_gpu_env_matches_the_membership_rule() {
    let observed = std::env::var("RMLX_SKIP_GPU").ok();
    assert_eq!(
        skip_if_no_gpu_env(),
        skip_value_means_skip(observed.as_deref()),
        "skip_if_no_gpu_env must delegate to skip_value_means_skip"
    );
}

/// `EnvGuard` restores a managed key even when the holder panics.
///
/// This is the whole reason the guard exists. Every env writer in this suite is
/// shaped `set_var` → `assert!` → restore, so a failing assertion skips its own
/// restore; without a `Drop` restore the dirty value leaks into every later test
/// and the next reader fails about its own precondition, burying the assertion
/// that actually broke.
///
/// The lock is not reentrant, so each phase takes it in its own scope and the
/// panicking closure acquires its own. The deliberate panic also poisons the
/// lock, so this exercises the poison recovery in `env_lock()` at the same time.
///
/// Mutation check: delete the `impl Drop for EnvGuard` body — the key stays at
/// `"1"` after the unwind and the final assertion fails (RED).
#[test]
#[allow(unsafe_code)]
fn env_guard_restores_managed_key_when_the_holder_panics() {
    let before = {
        let _guard = crate::test_utils::env_lock();
        std::env::var("RMLX_ROTOR_QJL").ok()
    };

    let caught = std::panic::catch_unwind(|| {
        let _guard = crate::test_utils::env_lock();
        // SAFETY: env lock held — no concurrent env reader/writer.
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
        panic!("simulated assertion failure while the environment is dirty");
    });
    assert!(caught.is_err(), "the closure was supposed to panic");

    let after = {
        let _guard = crate::test_utils::env_lock();
        std::env::var("RMLX_ROTOR_QJL").ok()
    };
    assert_eq!(
        before, after,
        "EnvGuard must restore RMLX_ROTOR_QJL while unwinding, not leak it"
    );
}

// ── cosine_similarity_per_row self-tests ─────────────────────────────────────

/// cosine_similarity_per_row of a vector with itself must be exactly 1.0.
#[test]
fn cosine_self_similarity_is_one() {
    let data = lcg_data(256, TEST_SEED);
    let stats = cosine_similarity_per_row(&data, &data, 64);
    assert_eq!(stats.n_rows, 4);
    assert!((stats.mean - 1.0).abs() < 1e-6, "mean={}", stats.mean);
    assert!((stats.min - 1.0).abs() < 1e-6, "min={}", stats.min);
}

/// cosine_similarity_per_row on identical zero vectors returns 1.0 (zero-vector special case).
#[test]
fn cosine_zero_vector_returns_one() {
    let reference = vec![0.0f32; 64];
    let decoded = vec![0.0f32; 64];
    let stats = cosine_similarity_per_row(&reference, &decoded, 64);
    assert!((stats.mean - 1.0).abs() < 1e-6, "mean={}", stats.mean);
}

/// cosine_similarity_per_row on orthogonal vectors returns ~0.0.
#[test]
fn cosine_orthogonal_vectors_near_zero() {
    // [1, 0] and [0, 1] are perfectly orthogonal.
    let reference = vec![1.0f32, 0.0];
    let decoded = vec![0.0f32, 1.0];
    let stats = cosine_similarity_per_row(&reference, &decoded, 2);
    assert!(stats.mean.abs() < 1e-6, "mean={}", stats.mean);
}

// ── fwht_normalize self-tests ─────────────────────────────────────────────────

/// FWHT of length-2 [1, 1] → [sqrt(2), 0] (normalized Hadamard).
#[test]
fn fwht_normalize_length_2_known_value() {
    let mut buf = vec![1.0f32, 1.0];
    fwht_normalize(&mut buf, 2);
    let expected = 2.0f32 / (2.0f32).sqrt(); // = sqrt(2)
    assert!(
        (buf[0] - expected).abs() < 1e-5,
        "buf[0]={} expected {expected}",
        buf[0]
    );
    assert!(buf[1].abs() < 1e-5, "buf[1]={} expected 0", buf[1]);
}

/// Applying fwht_normalize twice is the identity (R is self-inverse: R R = I).
#[test]
fn fwht_double_apply_is_identity() {
    let original = lcg_data(64, TEST_SEED);
    let mut buf = original.clone();
    fwht_normalize(&mut buf, 64);
    fwht_normalize(&mut buf, 64);
    for (i, (o, b)) in original.iter().zip(buf.iter()).enumerate() {
        assert!(
            (o - b).abs() < 1e-5,
            "identity broken at index {i}: {o} vs {b}"
        );
    }
}

// ── incoherence_per_row self-tests ───────────────────────────────────────────

/// Both endpoints of the statistic, against hand-computed values.
///
/// A flat vector has `max|x_i| = c` and `||x||_2 = c·sqrt(d)`, so
/// `mu = sqrt(d)·c / (c·sqrt(d)) = 1` — the minimum. A one-hot vector has
/// `max|x_i| = ||x||_2`, so `mu = sqrt(d)` — the maximum. Nothing else the
/// statistic could be would satisfy both.
#[test]
fn incoherence_endpoints_are_one_and_sqrt_d() {
    let d = 64usize;
    let sqrt_d = (d as f64).sqrt();

    let flat = vec![0.375f32; d];
    let flat_stats = incoherence_per_row(&flat, d);
    assert_eq!(flat_stats.n_rows, 1);
    assert!(
        (flat_stats.mean - 1.0).abs() < 1e-9,
        "flat vector mu {} != 1.0",
        flat_stats.mean,
    );

    let mut one_hot = vec![0.0f32; d];
    one_hot[7] = -2.5;
    let one_hot_stats = incoherence_per_row(&one_hot, d);
    assert!(
        (one_hot_stats.mean - sqrt_d).abs() < 1e-6,
        "one-hot mu {} != sqrt({d}) = {sqrt_d}",
        one_hot_stats.mean,
    );
}

/// An all-zero row is flat, so it is defined as `mu = 1.0` rather than `0/0`.
#[test]
fn incoherence_zero_row_is_one() {
    let stats = incoherence_per_row(&[0.0f32; 16], 16);
    assert!((stats.mean - 1.0).abs() < 1e-9, "mean={}", stats.mean);
}

/// `mean < p99 <= max` on a mixed set of rows, and the tail statistics pick out
/// the one-hot rows rather than averaging them away.
///
/// 100 rows with the last 2 one-hot: nearest-rank p99 is the 99th ordered
/// value, which lands inside the one-hot pair, so both `p99` and `max` see
/// `mu = sqrt(16) = 4` while the mean stays near 1.
#[test]
fn incoherence_tail_statistics_track_the_worst_rows() {
    let d = 16usize;
    let mut data = vec![1.0f32; d * 100];
    for row in 98..100 {
        for slot in data.iter_mut().skip(row * d).take(d) {
            *slot = 0.0;
        }
        data[row * d] = 1.0;
    }

    let stats = incoherence_per_row(&data, d);
    assert_eq!(stats.n_rows, 100);
    assert!((stats.max - 4.0).abs() < 1e-6, "max={}", stats.max);
    assert!((stats.p99 - 4.0).abs() < 1e-6, "p99={}", stats.p99);
    assert!(
        stats.mean < stats.p99 && stats.p99 <= stats.max,
        "expected mean < p99 <= max, got {} / {} / {}",
        stats.mean,
        stats.p99,
        stats.max,
    );
}

// ── fixture self-tests ───────────────────────────────────────────────────────

/// `gaussian_data` is seed-deterministic and is actually standard normal.
#[test]
fn gaussian_data_is_deterministic_and_standard_normal() {
    let a = gaussian_data(4096, TEST_SEED);
    let b = gaussian_data(4096, TEST_SEED);
    assert_eq!(a, b, "gaussian_data must be reproducible from its seed");
    assert_ne!(
        a,
        gaussian_data(4096, TEST_SEED ^ 1),
        "a different seed must produce different data"
    );

    let n = a.len() as f64;
    let mean = a.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = a.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n - mean * mean;
    // 4096 samples: the sample mean has sd 1/64 and the sample variance sd
    // sqrt(2/4096) ≈ 0.022, so these bounds are ~4 sigma.
    assert!(mean.abs() < 0.07, "sample mean {mean} is not ~0");
    assert!(
        (var - 1.0).abs() < 0.09,
        "sample variance {var} is not ~1 — Box-Muller is miswired"
    );
}

/// The outlier fixture is adversarial, and the i.i.d. fixtures are not.
///
/// Threshold justification, stated before the measurement: for an i.i.d. row of
/// length `d`, `||x||_2 ≈ sqrt(d)·sigma`, so `mu ≈ max_i|x_i| / sigma` — the
/// expected max of `d` samples in sigma units. At `d = 128` that is ≈ 2.83 for
/// a Gaussian and ≈ 1.71 for uniform-on-[-1,1]. **5.0 is ~1.8x above the
/// Gaussian i.i.d. value, so no i.i.d. fixture can reach it** — which is
/// exactly what the two negative controls below assert. Passing it therefore
/// means the fixture really does concentrate its mass in a few coordinates.
#[test]
fn outlier_fixture_is_adversarial_and_iid_fixtures_are_not() {
    const ADVERSARIAL_FLOOR: f64 = 5.0;

    let outlier = incoherence_per_row(&outlier_fixture(), OUTLIER_HEAD_DIM);
    assert!(
        outlier.mean >= ADVERSARIAL_FLOOR,
        "outlier fixture mean mu {:.4} < {ADVERSARIAL_FLOOR} — fixture is not adversarial",
        outlier.mean,
    );

    // Negative control 1: the same Gaussian base with the outlier channels
    // removed. This isolates the channels as the cause.
    let plain_gaussian = gaussian_data(OUTLIER_ROWS * OUTLIER_HEAD_DIM, TEST_SEED);
    let plain = incoherence_per_row(&plain_gaussian, OUTLIER_HEAD_DIM);
    assert!(
        plain.mean < ADVERSARIAL_FLOOR,
        "i.i.d. Gaussian mean mu {:.4} reached the adversarial floor {ADVERSARIAL_FLOOR} — \
         the floor does not separate outlier data from i.i.d. data",
        plain.mean,
    );

    // Negative control 2: the LCG uniform fixture every existing cosine gate
    // uses. It is the least adversarial of the three.
    let uniform = incoherence_per_row(
        &lcg_data(OUTLIER_ROWS * OUTLIER_HEAD_DIM, TEST_SEED),
        OUTLIER_HEAD_DIM,
    );
    assert!(
        uniform.mean < plain.mean,
        "uniform mean mu {:.4} should sit below Gaussian {:.4}",
        uniform.mean,
        plain.mean,
    );

    println!(
        "incoherence mean mu at head_dim={OUTLIER_HEAD_DIM}: \
         uniform={:.4}  gaussian={:.4}  outlier={:.4} (p99={:.4} max={:.4})",
        uniform.mean, plain.mean, outlier.mean, outlier.p99, outlier.max,
    );
}

/// The channel placement really does spread intra-block offsets, as its doc
/// claims, rather than parking every outlier at offset 0.
///
/// The plain even spread this replaced gave 0 / 32 / 64 / 96 — offset 0 for
/// every block size in play. That is the most aligned placement available, and
/// planar's per-pair Givens search is not symmetric under swapping the pair, so
/// it would bake an unmeasured bias into a pinned number.
#[test]
fn outlier_channels_do_not_share_one_intra_block_offset() {
    let channels = outlier_channels(OUTLIER_HEAD_DIM, 4);
    assert_eq!(channels, vec![0, 37, 74, 111], "placement changed");

    for block in [2usize, 3, 4] {
        let offsets: std::collections::BTreeSet<usize> =
            channels.iter().map(|c| c % block).collect();
        let expected = block.min(channels.len());
        assert!(
            offsets.len() >= expected.min(3),
            "block {block}: outliers occupy only offsets {offsets:?} — a constant intra-block \
             offset biases the block-local rotation gates"
        );
    }

    // Every affine group of 64 must contain an outlier: that is the condition
    // the rot_k gain justification rests on.
    for group in 0..OUTLIER_HEAD_DIM / 64 {
        assert!(
            channels.iter().any(|c| c / 64 == group),
            "affine group {group} of 64 has no outlier — the rot_k gain argument does not hold"
        );
    }
}

/// A larger outlier ratio makes the fixture strictly more adversarial.
#[test]
fn outlier_ratio_monotonically_raises_incoherence() {
    let mut previous = 0.0f64;
    for ratio in [1.0f32, 5.0, 10.0, 20.0] {
        let data = outlier_channel_data(64, OUTLIER_HEAD_DIM, 4, ratio, TEST_SEED);
        let stats = incoherence_per_row(&data, OUTLIER_HEAD_DIM);
        assert!(
            stats.mean > previous,
            "mu did not increase at ratio {ratio}: {:.4} <= {previous:.4}",
            stats.mean,
        );
        previous = stats.mean;
    }
}

/// `mu` against outlier-channel count: sharp rise, an early peak, then decay
/// back to the i.i.d. value once "outlier" describes every channel.
///
/// The count is as load-bearing as the ratio — the `rot_k` gain justification
/// turns on every affine group of 64 containing one — so it gets a sweep rather
/// than a bare assertion. It is **not** monotone, and asserting that it were
/// would have been wrong: `mu = sqrt(d)·max/||x||_2` grows its numerator only as
/// the max of `n` draws while the denominator grows as `sqrt(n)`, so past a
/// handful of channels the denominator wins.
///
/// The endpoint is the strong check. Scaling *every* channel by the same factor
/// is a pure change of units, and `mu` is scale invariant, so `n = head_dim`
/// must reproduce `n = 0` exactly. That validates the statistic and the
/// collision handling in `outlier_channels` at once — a placement that scaled
/// some channel twice would fail it.
#[test]
fn incoherence_against_outlier_channel_count_rises_peaks_then_returns() {
    let measure = |channels: usize| {
        let data = outlier_channel_data(256, OUTLIER_HEAD_DIM, channels, 20.0, TEST_SEED);
        incoherence_per_row(&data, OUTLIER_HEAD_DIM).mean
    };

    let baseline = measure(0);
    let curve: Vec<(usize, f64)> = [1usize, 2, 4, 8, 32]
        .into_iter()
        .map(|n| (n, measure(n)))
        .collect();
    let saturated = measure(OUTLIER_HEAD_DIM);

    println!(
        "mu vs outlier channels at head_dim={OUTLIER_HEAD_DIM}: 0 -> {baseline:.4}, {}, \
         {OUTLIER_HEAD_DIM} -> {saturated:.4}",
        curve
            .iter()
            .map(|(n, m)| format!("{n} -> {m:.4}"))
            .collect::<Vec<_>>()
            .join(", "),
    );

    // A single channel already more than doubles the i.i.d. value.
    let (_, one) = curve[0];
    assert!(
        one > 2.0 * baseline,
        "one outlier channel gave mu {one:.4}, not clear of twice the i.i.d. {baseline:.4}"
    );

    // Past the peak the statistic decays: rare is what makes an outlier.
    let (_, two) = curve[1];
    let (_, eight) = curve[3];
    let (_, thirty_two) = curve[4];
    assert!(
        two > eight && eight > thirty_two,
        "mu did not decay past its peak: 2 -> {two:.4}, 8 -> {eight:.4}, 32 -> {thirty_two:.4}"
    );

    // Scale invariance: every channel scaled is a change of units.
    let drift = (saturated - baseline).abs() / baseline;
    println!("scale-invariance residual: {drift:.3e} relative");
    assert!(
        drift < 1e-5,
        "scaling every channel moved mu by {drift:.3e} relative ({baseline:.6} -> \
         {saturated:.6}); that is far beyond f32 rounding, so either the statistic is not \
         scale invariant or a channel was scaled twice"
    );
}

// ── rate-distortion helper self-tests ────────────────────────────────────────

/// A lossless round-trip has infinite SQNR; a known error power gives the
/// hand-computed dB.
#[test]
fn sqnr_db_matches_hand_computed_values() {
    let reference = lcg_data(1024, TEST_SEED);
    assert!(
        sqnr_db(&reference, &reference).is_infinite(),
        "identity round-trip must report infinite SQNR"
    );

    // Scale every sample by 0.9: error = 0.1·x, so P_err = 0.01·P_sig and
    // SQNR = 10·log10(100) = 20 dB. The tolerance covers the f32 rounding of
    // the `v * 0.9` product itself, not the f64 statistic.
    let scaled: Vec<f32> = reference.iter().map(|&v| v * 0.9).collect();
    let measured = sqnr_db(&reference, &scaled);
    assert!(
        (measured - 20.0).abs() < 1e-4,
        "expected 20 dB, got {measured}"
    );
}

/// The anchors are ordered, sit below the `6.02·b` rate-distortion bound they
/// are explicitly not, and approach one bit per step from below.
#[test]
fn lloyd_max_anchors_are_below_the_rate_distortion_bound() {
    for (i, &anchor) in LLOYD_MAX_GAUSSIAN_SQNR_DB.iter().enumerate() {
        let bits = u8::try_from(i + 1).expect("bits index fits u8");
        assert!(
            (lloyd_max_anchor_db(bits) - anchor).abs() < 1e-12,
            "lloyd_max_anchor_db({bits}) disagrees with the table"
        );
        let bound = DB_PER_BIT * f64::from(bits);
        assert!(
            anchor < bound,
            "anchor {anchor} at bits={bits} is at or above the {bound} dB bound — \
             a fixed-rate Lloyd-Max quantizer cannot reach the bound"
        );
        if i > 0 {
            let step = anchor - LLOYD_MAX_GAUSSIAN_SQNR_DB[i - 1];
            assert!(
                step > 0.0 && step < DB_PER_BIT,
                "step to bits={bits} was {step} dB; expected 0 < step < {DB_PER_BIT}"
            );
        }
    }
}

/// `wasted_bits` is the dB shortfall divided by one bit's worth of dB, signed
/// so that "ahead of the anchor" is negative.
#[test]
fn wasted_bits_converts_db_shortfall_to_bits() {
    assert!((wasted_bits(14.616, 14.616)).abs() < 1e-12);
    assert!((wasted_bits(14.616 - DB_PER_BIT, 14.616) - 1.0).abs() < 1e-9);
    assert!((wasted_bits(14.616 + 2.0 * DB_PER_BIT, 14.616) + 2.0).abs() < 1e-9);
}
