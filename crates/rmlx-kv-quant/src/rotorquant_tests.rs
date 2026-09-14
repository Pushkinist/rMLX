//! Tests for the rotor3 / rotor4 KV codecs.
//!
//! Coverage:
//!   * Roundtrip determinism: encode → decode → encode → bit-identical codes.
//!   * Cosine gate on the LCG fixture: empirical-floor pattern.
//!   * Head-dim tail padding: non-multiple-of-3 head dims decode correctly
//!     in the original head_dim slots (no tail leakage).
//!   * Error guards: head_dim==0, rotor table size mismatch, slice length mismatch.
//!   * rotor4: sandwich still affects output (no-op regression guard).

use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    make_qjl_projection, n_groups_for, rotor3_decode, rotor3_encode, rotor3_k_decode,
    rotor3_k_encode, rotor4_decode, rotor4_encode, rotor4_k_decode, rotor4_k_encode, row_words_for,
    unpack_qjl_signs, RotorQuantError, ROTOR3_GROUP_SIZE,
};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

/// Sentinel: a different rotor table must produce different quantized codes.
///
/// Pins the rotor sandwich correctness — an earlier rev called `gp_rotor_mv`
/// twice (both on the LEFT), which collapsed the sandwich `R * mv * R̃` to a
/// no-op (`R̃ * (R * mv) = mv`). With the no-op, swapping the rotor table
/// had zero effect on the codes; this test would have caught that bug.
#[test]
fn rotor_table_actually_affects_output() {
    let head_dim = 96;
    let n_tokens = 8;
    let n_groups = n_groups_for(head_dim);
    let rotors_a = make_rotor_table(0, 0, n_groups);
    let rotors_b = make_rotor_table(11, 7, n_groups);
    // Sanity: the two tables must differ — otherwise the test is vacuous.
    assert_ne!(rotors_a, rotors_b, "rotor tables A and B must differ");

    let data = lcg_data(n_tokens * head_dim, TEST_SEED);
    let (codes_a, _, _) = rotor3_encode(&data, &rotors_a, head_dim).unwrap();
    let (codes_b, _, _) = rotor3_encode(&data, &rotors_b, head_dim).unwrap();
    assert_ne!(
        codes_a, codes_b,
        "rotor table must change quantized codes — sandwich must be R * mv * R̃, not a no-op"
    );
}

const LAYER_IDX: u32 = 0;
const HEAD_IDX: u32 = 0;

// ── Determinism ───────────────────────────────────────────────────────────────

/// Encode twice on the same input: both passes produce bit-identical codes.
///
/// Verifies the codec is purely deterministic: no hidden RNG, no mutable
/// state. Mirror of `iso_encode_decode_roundtrip_determinism` in `isoquant`.
#[test]
fn rotor3_encode_determinism() {
    let head_dim = 96;
    let n_tokens = 8;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let data = lcg_data(n_tokens * head_dim, TEST_SEED);

    let (codes1, scales1, norms1) = rotor3_encode(&data, &rotors, head_dim).unwrap();
    let (codes2, scales2, norms2) = rotor3_encode(&data, &rotors, head_dim).unwrap();

    assert_eq!(codes1.len(), codes2.len(), "codes length mismatch");
    for (i, (&c1, &c2)) in codes1.iter().zip(codes2.iter()).enumerate() {
        assert_eq!(
            c1, c2,
            "codes differ at word {i}: first pass {c1:#010x} vs second pass {c2:#010x}"
        );
    }
    assert_eq!(scales1, scales2, "scales differ between passes");
    assert_eq!(norms1, norms2, "norms differ between passes");

    let expected_words = n_tokens * row_words_for(head_dim, 3);
    assert_eq!(
        codes1.len(),
        expected_words,
        "expected words count mismatch"
    );
    assert_eq!(scales1.len(), n_tokens * n_groups, "scales length");
    assert_eq!(norms1.len(), n_tokens, "norms length");
}

// ── Cosine gate (empirical floor) ─────────────────────────────────────────────

/// rotor3 cosine-similarity gate on the LCG fixture.
///
/// Setup: 32 tokens × 128-dim, group_size=3, bits=3.
/// LCG seed: [`TEST_SEED`] (pinned; must stay reproducible).
///
/// Threshold: empirical-floor pattern — `measured_mean − 0.001`. The first
/// run prints the measured value to stdout; the assertion uses that minus
/// the regression margin. Update both constants if the rotor table or the
/// codebook changes (bump measured + re-set threshold = new_mean − 0.001).
///
/// rMLX uses Gaussian Lloyd-Max codebook (`lloyd_gaussian_codebook(3)`); the
/// Python reference uses Beta Lloyd-Max. The expected delta is negligible for
/// `head_dim ≥ 64`.
///
/// Published mtq number is 0.9780 (multi-turboquant `rotor3`). The rMLX
/// LCG-fixture measurement is recorded inline via `println!`.
///
/// **History note:** the earlier `gp_rotor_mv` × 2 encode/decode path was a
/// silent no-op (`R̃ * (R * mv) = mv`); the previous 0.9955 / 0.9941 numbers
/// were measuring an unrotated codec. After fixing the sandwich to apply
/// `R * mv * R̃` (via [`crate::clifford::rotor_sandwich`]) the LCG-fixture
/// numbers settled at mean = 0.9956, min = 0.9947 (slightly higher because
/// the rotation decorrelates the residual coordinates).
#[test]
fn rotor3_cosine_gate() {
    let n_tokens = 32;
    let head_dim = 128;
    let n_groups = n_groups_for(head_dim);

    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let data = lcg_data(n_tokens * head_dim, TEST_SEED);

    let (codes, scales, norms) = rotor3_encode(&data, &rotors, head_dim).unwrap();
    let decoded = rotor3_decode(&codes, &scales, &norms, &rotors, head_dim).unwrap();

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);

    // Empirical floor — measured on the LCG fixture (TEST_SEED, n_tokens=32,
    // head_dim=128, group_size=3, bits=3) AFTER fixing the rotor sandwich
    // (the earlier `gp_rotor_mv` × 2 path was a no-op).
    //   Mean = 0.995601, Min = 0.994737.
    // Threshold = measured − 0.001 (regression margin, iso3 / iso4 pattern).
    let threshold_mean = 0.994_6_f32;
    let threshold_min = 0.993_7_f32;

    assert!(
        stats.mean >= threshold_mean,
        "rotor3 cosine mean {:.6} fell below threshold {threshold_mean:.6} (n_rows={})",
        stats.mean,
        stats.n_rows,
    );
    assert!(
        stats.min >= threshold_min,
        "rotor3 cosine min {:.6} fell below threshold {threshold_min:.6} (n_rows={})",
        stats.min,
        stats.n_rows,
    );

    println!(
        "rotor3 cosine: mean={:.6}, min={:.6}, n_rows={} (threshold mean>={threshold_mean:.6}, min>={threshold_min:.6})",
        stats.mean, stats.min, stats.n_rows,
    );
}

// ── Tail padding ──────────────────────────────────────────────────────────────

/// Head_dim not a multiple of 3 must decode correctly within the original
/// `head_dim` slots (no tail leakage from padded slots).
#[test]
fn rotor3_head_dim_not_multiple_of_three() {
    // head_dim = 100 → n_groups = ceil(100/3) = 34, last group has 1 real elem
    // and 2 zero-pad slots.
    let head_dim = 100_usize;
    let n_groups = n_groups_for(head_dim);
    assert_eq!(n_groups, 34, "n_groups computation");
    assert_eq!(n_groups * ROTOR3_GROUP_SIZE, 102, "expect 2 pad slots");

    let n_tokens = 4;
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let data = lcg_data(n_tokens * head_dim, TEST_SEED ^ 0xAA);

    let (codes, scales, norms) = rotor3_encode(&data, &rotors, head_dim).unwrap();
    let decoded = rotor3_decode(&codes, &scales, &norms, &rotors, head_dim).unwrap();

    assert_eq!(
        decoded.len(),
        n_tokens * head_dim,
        "decode length must equal n_tokens * head_dim (no tail leakage)"
    );

    // The cosine on this fixture must remain reasonable — using the same
    // empirical-floor pattern with a slightly looser margin since head_dim
    // is smaller (norm-aware codebook has less signal).
    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    assert!(
        stats.mean > 0.95,
        "rotor3 cosine mean {:.6} too low on tail-padded head_dim={head_dim}",
        stats.mean
    );
}

// ── Error guards ──────────────────────────────────────────────────────────────

/// `head_dim == 0` must error with `HeadDimZero`.
#[test]
fn rotor3_encode_zero_head_dim() {
    let rotors: Vec<f32> = Vec::new();
    let v: Vec<f32> = Vec::new();
    let err = rotor3_encode(&v, &rotors, 0).unwrap_err();
    assert!(
        matches!(err, RotorQuantError::HeadDimZero),
        "expected HeadDimZero, got {err:?}"
    );
}

/// Rotor table size mismatch must error with `RotorTableLen`.
#[test]
fn rotor3_encode_wrong_rotor_table_len() {
    let head_dim = 96;
    let n_groups = n_groups_for(head_dim);
    let expected = n_groups * 4;
    // Pass a table that is one rotor short.
    let rotors = vec![0.0_f32; expected - 4];
    let data = vec![0.5_f32; head_dim];
    let err = rotor3_encode(&data, &rotors, head_dim).unwrap_err();
    match err {
        RotorQuantError::RotorTableLen {
            got,
            expected: e,
            n_groups: ng,
        } => {
            assert_eq!(got, expected - 4);
            assert_eq!(e, expected);
            assert_eq!(ng, n_groups);
        }
        RotorQuantError::HeadDimZero
        | RotorQuantError::LenNotMultipleOfHeadDim { .. }
        | RotorQuantError::CodePlaneLen { .. }
        | RotorQuantError::Codebook(_) => panic!("expected RotorTableLen, got {err:?}"),
    }
}

/// `v.len()` not a multiple of `head_dim` must error with
/// `LenNotMultipleOfHeadDim`.
#[test]
fn rotor3_encode_len_not_multiple() {
    let head_dim = 96;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let v = vec![0.5_f32; head_dim + 7]; // off by 7
    let err = rotor3_encode(&v, &rotors, head_dim).unwrap_err();
    match err {
        RotorQuantError::LenNotMultipleOfHeadDim { len, head_dim: hd } => {
            assert_eq!(len, head_dim + 7);
            assert_eq!(hd, head_dim);
        }
        RotorQuantError::HeadDimZero
        | RotorQuantError::RotorTableLen { .. }
        | RotorQuantError::CodePlaneLen { .. }
        | RotorQuantError::Codebook(_) => panic!("expected LenNotMultipleOfHeadDim, got {err:?}"),
    }
}

// ── rotor4 tests ──────────────────────────────────────────────────────────────

/// Sentinel: a different rotor table must produce different rotor4 codes.
///
/// Mirrors `rotor_table_actually_affects_output` for the 4-bit codec.
/// Guards against a no-op sandwich regression.
#[test]
fn rotor4_table_actually_affects_output() {
    let head_dim = 96;
    let n_tokens = 8;
    let n_groups = n_groups_for(head_dim);
    let rotors_a = make_rotor_table(0, 0, n_groups);
    let rotors_b = make_rotor_table(11, 7, n_groups);
    assert_ne!(rotors_a, rotors_b, "rotor tables A and B must differ");

    let data = lcg_data(n_tokens * head_dim, TEST_SEED ^ 0x0F14);
    let (codes_a, _, _) = rotor4_encode(&data, &rotors_a, head_dim).unwrap();
    let (codes_b, _, _) = rotor4_encode(&data, &rotors_b, head_dim).unwrap();
    assert_ne!(
        codes_a, codes_b,
        "rotor4 table must change quantized codes — sandwich must not be a no-op"
    );
}

/// rotor4 encode is deterministic (no hidden RNG or mutable state).
#[test]
fn rotor4_encode_determinism() {
    let head_dim = 96;
    let n_tokens = 8;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let data = lcg_data(n_tokens * head_dim, TEST_SEED);

    let (codes1, scales1, norms1) = rotor4_encode(&data, &rotors, head_dim).unwrap();
    let (codes2, scales2, norms2) = rotor4_encode(&data, &rotors, head_dim).unwrap();

    assert_eq!(codes1, codes2, "codes must be bit-identical");
    assert_eq!(scales1, scales2, "scales must be identical");
    assert_eq!(norms1, norms2, "norms must be identical");

    let expected_words = n_tokens * row_words_for(head_dim, 4);
    assert_eq!(codes1.len(), expected_words, "codes length");
    assert_eq!(scales1.len(), n_tokens * n_groups, "scales length");
    assert_eq!(norms1.len(), n_tokens, "norms length");
}

/// rotor4 cosine-similarity gate on the LCG fixture.
///
/// Setup: 32 tokens × 128-dim, group_size=3, bits=4.
/// LCG seed: [`TEST_SEED`] (same fixture as rotor3, for apples-to-apples comparison).
///
/// Threshold: empirical-floor pattern — `measured_mean − 0.001`.
///
/// Effective bpe note: ~10.7 bpe pre-scale at bits=4 (8 components × 4 bits
/// = 32 bits per group of 3 real elements, + per-group f32 scale + per-token
/// norm). This is the single-codebook simplification.
#[test]
fn rotor4_cosine_gate() {
    let n_tokens = 32;
    let head_dim = 128;
    let n_groups = n_groups_for(head_dim);

    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let data = lcg_data(n_tokens * head_dim, TEST_SEED);

    let (codes, scales, norms) = rotor4_encode(&data, &rotors, head_dim).unwrap();
    let decoded = rotor4_decode(&codes, &scales, &norms, &rotors, head_dim).unwrap();

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);

    // Empirical floor — measured on the LCG fixture (TEST_SEED,
    // n_tokens=32, head_dim=128, group_size=3, bits=4, single-codebook).
    // Threshold = measured − 0.001 (regression margin; iso4 / rotor3 pattern).
    //   Mean = 0.998873, Min = 0.998465.
    let threshold_mean = 0.997_8_f32;
    let threshold_min = 0.997_4_f32;

    assert!(
        stats.mean >= threshold_mean,
        "rotor4 cosine mean {:.6} fell below threshold {threshold_mean:.6} (n_rows={})",
        stats.mean,
        stats.n_rows,
    );
    assert!(
        stats.min >= threshold_min,
        "rotor4 cosine min {:.6} fell below threshold {threshold_min:.6} (n_rows={})",
        stats.min,
        stats.n_rows,
    );

    println!(
        "rotor4 cosine: mean={:.6}, min={:.6}, n_rows={} \
         (threshold mean>={threshold_mean:.6}, min>={threshold_min:.6})",
        stats.mean, stats.min, stats.n_rows,
    );
}

/// Head_dim not a multiple of 3 decodes correctly for rotor4.
#[test]
fn rotor4_head_dim_not_multiple_of_three() {
    let head_dim = 100_usize;
    let n_groups = n_groups_for(head_dim);
    assert_eq!(n_groups, 34, "n_groups computation");

    let n_tokens = 4;
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let data = lcg_data(n_tokens * head_dim, TEST_SEED ^ 0xF4AA);

    let (codes, scales, norms) = rotor4_encode(&data, &rotors, head_dim).unwrap();
    let decoded = rotor4_decode(&codes, &scales, &norms, &rotors, head_dim).unwrap();

    assert_eq!(
        decoded.len(),
        n_tokens * head_dim,
        "decode length must equal n_tokens * head_dim (no tail leakage)"
    );

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    assert!(
        stats.mean > 0.95,
        "rotor4 cosine mean {:.6} too low on tail-padded head_dim={head_dim}",
        stats.mean
    );
}

// ── QJL score-time correction lift tests ──────────────────────────────────────

/// Inner-product `q · k` in f64 for numerical headroom.
fn dot_f64(q: &[f32], k: &[f32]) -> f64 {
    debug_assert_eq!(q.len(), k.len());
    q.iter()
        .zip(k.iter())
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

/// In-place L2-normalize a flat row-major `[n_rows, d]` buffer.
fn l2_normalize_rows(buf: &mut [f32], d: usize) {
    for row in buf.chunks_exact_mut(d) {
        let mut nsq = 0.0_f64;
        for &v in row.iter() {
            nsq += f64::from(v) * f64::from(v);
        }
        let n = nsq.sqrt() as f32;
        if n > f32::EPSILON {
            for v in row.iter_mut() {
                *v /= n;
            }
        }
    }
}

/// Score estimator is unbiased under QJL correction — rotor3.
///
/// This mirrors the Python reference test `test_inner_product_unbiased` in
/// `rotorquant/tests/test_rotorquant.py:94`: over `n=1024` unit-normalized
/// random pairs, the absolute mean bias of the estimated SDPA score must
/// stay below 0.05 — the same threshold the Python suite uses to certify
/// the QJL inner-product estimator. Crucially, this test exercises the
/// dequant-side `apply_qjl_correction` end-to-end through `rotor3_k_decode`.
///
/// Note on "lift": on unit-normalized LCG fixtures the rotor3 codec is
/// already near-unbiased (off-bias ≈ 1e-4), so a per-test `|bias_on| <
/// |bias_off|` gate is statistically noisy. The real-model output-logit
/// cosine lift is documented in `docs/KV_QUANT.md`.
#[test]
fn qjl_correction_score_estimator_unbiased_rotor3() {
    let head_dim = 128_usize;
    let n_tokens = 1024_usize;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let s_matrix = make_qjl_projection(head_dim);

    let mut k_true = lcg_data(n_tokens * head_dim, TEST_SEED);
    let mut q_buf = lcg_data(n_tokens * head_dim, TEST_SEED ^ 0x9E37_79B1);
    l2_normalize_rows(&mut k_true, head_dim);
    l2_normalize_rows(&mut q_buf, head_dim);

    let true_scores: Vec<f64> = (0..n_tokens)
        .map(|t| {
            let qt = &q_buf[t * head_dim..(t + 1) * head_dim];
            let kt = &k_true[t * head_dim..(t + 1) * head_dim];
            dot_f64(qt, kt)
        })
        .collect();

    // QJL OFF — rotor only.
    let (codes_off, scales_off, norms_off, _, _) =
        rotor3_k_encode(&k_true, &rotors, head_dim, None).unwrap();
    let k_dec_off = rotor3_k_decode(
        &codes_off,
        &scales_off,
        &norms_off,
        &rotors,
        head_dim,
        &[],
        &[],
        None,
    )
    .unwrap();
    let scores_off: Vec<f64> = (0..n_tokens)
        .map(|t| {
            dot_f64(
                &q_buf[t * head_dim..(t + 1) * head_dim],
                &k_dec_off[t * head_dim..(t + 1) * head_dim],
            )
        })
        .collect();

    // QJL ON — rotor + dequant-side residual-add via apply_qjl_correction.
    let (codes_on, scales_on, norms_on, qjl_codes_on, qjl_norms_on) =
        rotor3_k_encode(&k_true, &rotors, head_dim, Some(&s_matrix)).unwrap();
    let k_dec_on = rotor3_k_decode(
        &codes_on,
        &scales_on,
        &norms_on,
        &rotors,
        head_dim,
        &qjl_codes_on,
        &qjl_norms_on,
        Some(&s_matrix),
    )
    .unwrap();
    let scores_on: Vec<f64> = (0..n_tokens)
        .map(|t| {
            dot_f64(
                &q_buf[t * head_dim..(t + 1) * head_dim],
                &k_dec_on[t * head_dim..(t + 1) * head_dim],
            )
        })
        .collect();

    let bias_off: f64 = scores_off
        .iter()
        .zip(true_scores.iter())
        .map(|(e, t)| e - t)
        .sum::<f64>()
        / (n_tokens as f64);
    let bias_on: f64 = scores_on
        .iter()
        .zip(true_scores.iter())
        .map(|(e, t)| e - t)
        .sum::<f64>()
        / (n_tokens as f64);

    println!(
        "rotor3 QJL bias: off={bias_off:+.6} on={bias_on:+.6} \
         |off|={:.6} |on|={:.6} (n_tokens={n_tokens})",
        bias_off.abs(),
        bias_on.abs(),
    );

    // Python ref threshold: |bias| < 0.05. Both off and on must satisfy this —
    // proves the dequant-side residual-add does not introduce a runaway
    // systematic error.
    assert!(
        bias_off.abs() < 0.05,
        "QJL OFF bias {bias_off:.6} exceeds Python-ref threshold 0.05"
    );
    assert!(
        bias_on.abs() < 0.05,
        "QJL ON bias {bias_on:.6} exceeds Python-ref threshold 0.05"
    );
}

/// Same bias gate for rotor4 as the rotor3 QJL correction test.
#[test]
fn qjl_correction_score_estimator_unbiased_rotor4() {
    let head_dim = 128_usize;
    let n_tokens = 1024_usize;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let s_matrix = make_qjl_projection(head_dim);

    let mut k_true = lcg_data(n_tokens * head_dim, TEST_SEED);
    let mut q_buf = lcg_data(n_tokens * head_dim, TEST_SEED ^ 0x9E37_79B1);
    l2_normalize_rows(&mut k_true, head_dim);
    l2_normalize_rows(&mut q_buf, head_dim);

    let true_scores: Vec<f64> = (0..n_tokens)
        .map(|t| {
            dot_f64(
                &q_buf[t * head_dim..(t + 1) * head_dim],
                &k_true[t * head_dim..(t + 1) * head_dim],
            )
        })
        .collect();

    let (codes_off, scales_off, norms_off, _, _) =
        rotor4_k_encode(&k_true, &rotors, head_dim, None).unwrap();
    let k_dec_off = rotor4_k_decode(
        &codes_off,
        &scales_off,
        &norms_off,
        &rotors,
        head_dim,
        &[],
        &[],
        None,
    )
    .unwrap();
    let scores_off: Vec<f64> = (0..n_tokens)
        .map(|t| {
            dot_f64(
                &q_buf[t * head_dim..(t + 1) * head_dim],
                &k_dec_off[t * head_dim..(t + 1) * head_dim],
            )
        })
        .collect();

    let (codes_on, scales_on, norms_on, qjl_codes_on, qjl_norms_on) =
        rotor4_k_encode(&k_true, &rotors, head_dim, Some(&s_matrix)).unwrap();
    let k_dec_on = rotor4_k_decode(
        &codes_on,
        &scales_on,
        &norms_on,
        &rotors,
        head_dim,
        &qjl_codes_on,
        &qjl_norms_on,
        Some(&s_matrix),
    )
    .unwrap();
    let scores_on: Vec<f64> = (0..n_tokens)
        .map(|t| {
            dot_f64(
                &q_buf[t * head_dim..(t + 1) * head_dim],
                &k_dec_on[t * head_dim..(t + 1) * head_dim],
            )
        })
        .collect();

    let bias_off: f64 = scores_off
        .iter()
        .zip(true_scores.iter())
        .map(|(e, t)| e - t)
        .sum::<f64>()
        / (n_tokens as f64);
    let bias_on: f64 = scores_on
        .iter()
        .zip(true_scores.iter())
        .map(|(e, t)| e - t)
        .sum::<f64>()
        / (n_tokens as f64);

    println!(
        "rotor4 QJL bias: off={bias_off:+.6} on={bias_on:+.6} \
         |off|={:.6} |on|={:.6} (n_tokens={n_tokens})",
        bias_off.abs(),
        bias_on.abs(),
    );

    assert!(
        bias_off.abs() < 0.05,
        "QJL OFF bias {bias_off:.6} exceeds Python-ref threshold 0.05"
    );
    assert!(
        bias_on.abs() < 0.05,
        "QJL ON bias {bias_on:.6} exceeds Python-ref threshold 0.05"
    );
}

/// Sanity: when `qjl_packed` is empty, `apply_qjl_correction` stays a no-op
/// (caller passes None for the S matrix or empty packed/norms).
/// Exercises the `--rotor-qjl off` path through the public encode/decode API.
#[test]
fn qjl_off_path_is_no_op() {
    let head_dim = 24_usize;
    let n_tokens = 4_usize;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let k = lcg_data(n_tokens * head_dim, TEST_SEED);

    // Encode + decode with QJL off (None S matrix) — must match plain rotor3.
    let (codes_no_qjl, scales_no_qjl, norms_no_qjl, qjl_codes, qjl_norms) =
        rotor3_k_encode(&k, &rotors, head_dim, None).unwrap();
    assert!(qjl_codes.is_empty(), "qjl_codes empty when S is None");
    assert!(qjl_norms.is_empty(), "qjl_norms empty when S is None");

    let dec_off = rotor3_k_decode(
        &codes_no_qjl,
        &scales_no_qjl,
        &norms_no_qjl,
        &rotors,
        head_dim,
        &[],
        &[],
        None,
    )
    .unwrap();
    let plain = rotor3_decode(
        &codes_no_qjl,
        &scales_no_qjl,
        &norms_no_qjl,
        &rotors,
        head_dim,
    )
    .unwrap();
    assert_eq!(dec_off, plain, "QJL off must equal plain rotor3_decode");
}

/// Linearity-bit-equivalence proof: decode-time residual-add (`Q · K_on`)
/// equals the Python-reference score-time correction (`Q · K_off + term2`).
///
/// By linearity of `Q · K`, adding `Δk[t, j] = ||r_t|| · scale · Σ_i S[i,j] · signs[t,i]`
/// to the decoded K is mathematically equivalent to the Python score-time formula
///
/// ```text
///   term2 = ||r|| · sqrt(π/2)/m · (Q @ S.T · qjl_signs).sum()
/// ```
///
/// from `rotorquant.py:246-263`. If this test ever fails, the residual sign /
/// shape / transpose in `apply_qjl_correction` is wrong — the bias-mean test
/// would only catch large systematic errors, while this test pins the **exact**
/// per-token, per-row equivalence the design choice rests on. Pass band: f32
/// reordering means ~1e-4 absolute, ~1e-5 relative is the tightest the floats
/// will hold for small synthetic inputs.
#[test]
fn qjl_residual_add_matches_score_time_correction() {
    let head_dim = 64_usize;
    let n_tokens = 4_usize;
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(LAYER_IDX, HEAD_IDX, n_groups);
    let s_matrix = make_qjl_projection(head_dim);

    // Synthetic Q, K — keep numerical scale moderate so f32 reorder error
    // stays in the low 1e-5 range.
    let mut k = lcg_data(n_tokens * head_dim, TEST_SEED);
    let mut q = lcg_data(n_tokens * head_dim, TEST_SEED ^ 0x9E37_79B1);
    l2_normalize_rows(&mut k, head_dim);
    l2_normalize_rows(&mut q, head_dim);

    let qjl_dim = head_dim; // make_qjl_projection: qjl_dim == head_dim.
    let bytes_per_tok = qjl_dim.div_ceil(8);
    let correction_scale = (std::f32::consts::PI / 2.0_f32).sqrt() / (qjl_dim as f32);

    // Encode with QJL on — gives us (codes, qjl_packed, qjl_norms).
    let (codes, scales, norms, qjl_packed, qjl_norms) =
        rotor3_k_encode(&k, &rotors, head_dim, Some(&s_matrix)).unwrap();
    assert_eq!(qjl_norms.len(), n_tokens, "one qjl_norm per token");
    assert_eq!(
        qjl_packed.len(),
        n_tokens * bytes_per_tok,
        "qjl_packed = n_tokens * ceil(qjl_dim/8) bytes"
    );

    // K_dec_off = rotor3_decode (no QJL applied).
    let k_dec_off = rotor3_decode(&codes, &scales, &norms, &rotors, head_dim).unwrap();
    // K_dec_on = rotor3_k_decode (residual-add applied via apply_qjl_correction).
    let k_dec_on = rotor3_k_decode(
        &codes,
        &scales,
        &norms,
        &rotors,
        head_dim,
        &qjl_packed,
        &qjl_norms,
        Some(&s_matrix),
    )
    .unwrap();

    let mut max_abs_err: f64 = 0.0;
    let mut max_rel_err: f64 = 0.0;
    for t in 0..n_tokens {
        let qt = &q[t * head_dim..(t + 1) * head_dim];
        let kt_off = &k_dec_off[t * head_dim..(t + 1) * head_dim];
        let kt_on = &k_dec_on[t * head_dim..(t + 1) * head_dim];
        let packed_row = &qjl_packed[t * bytes_per_tok..(t + 1) * bytes_per_tok];
        let signs = unpack_qjl_signs(packed_row, qjl_dim);

        // score_off = Q[t] · K_dec_off[t]
        let score_off = dot_f64(qt, kt_off);
        // score_on  = Q[t] · K_dec_on[t]      (decode-time residual-add)
        let score_on = dot_f64(qt, kt_on);

        // score_ref = score_off + ||r_t|| · scale · Σ_i signs[t,i] · (Q[t] @ S.T)[i]
        //            (the Python score-time term2)
        // (Q @ S.T)[i] = Σ_j Q[t,j] · S[i,j]
        let norm_t = qjl_norms[t];
        let mut term2 = 0.0_f64;
        for i in 0..qjl_dim {
            let mut q_dot_s_row = 0.0_f64;
            for j in 0..head_dim {
                q_dot_s_row += f64::from(qt[j]) * f64::from(s_matrix[i * head_dim + j]);
            }
            term2 += f64::from(signs[i]) * q_dot_s_row;
        }
        term2 *= f64::from(norm_t) * f64::from(correction_scale);
        let score_ref = score_off + term2;

        let abs_err = (score_on - score_ref).abs();
        let denom = score_ref.abs().max(1e-9_f64);
        let rel_err = abs_err / denom;
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);

        // Per-token pass band: f32 dot reorder error on n=64 dot products is
        // typically <5e-5 absolute on unit-normalized vectors. The 1e-4 ceiling
        // is the Python-ref equivalence gate.
        assert!(
            abs_err < 1e-4,
            "QJL linearity gate failed at token {t}: \
             score_on={score_on:+.9} score_ref={score_ref:+.9} \
             abs_err={abs_err:.3e} (>1e-4)"
        );
    }
    println!(
        "QJL linearity-bit-equivalence: max_abs_err={max_abs_err:.3e} \
         max_rel_err={max_rel_err:.3e} (n_tokens={n_tokens}, head_dim={head_dim})"
    );
}
