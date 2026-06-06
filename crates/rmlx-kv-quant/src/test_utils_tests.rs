// unsafe_code: std::env::set_var / remove_var are unsafe in Rust 1.95+;
// used in skip_if_no_gpu_env_strict_membership to pin env state for the test.
#![allow(unsafe_code)]

use super::{
    cosine_similarity_per_row, fwht_normalize, lcg_data, skip_if_no_gpu_env,
    vectorized_parity_check, TEST_SEED,
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

/// `skip_if_no_gpu_env` strict membership test: only `"1"` returns true.
///
/// Sets the env var to each interesting value in turn, asserts the expected
/// result, then restores the prior value.  Replaces the old conditional test
/// that silently no-oped when the var happened to be `"1"` already.
#[test]
fn skip_if_no_gpu_env_strict_membership() {
    let prior = std::env::var("RMLX_SKIP_GPU").ok();
    // SAFETY: single-threaded test; we restore the prior value before exit.
    unsafe { std::env::set_var("RMLX_SKIP_GPU", "0") };
    assert!(!skip_if_no_gpu_env(), "RMLX_SKIP_GPU=0 must return false");
    unsafe { std::env::set_var("RMLX_SKIP_GPU", "true") };
    assert!(
        !skip_if_no_gpu_env(),
        "RMLX_SKIP_GPU=true must return false"
    );
    unsafe { std::env::remove_var("RMLX_SKIP_GPU") };
    assert!(
        !skip_if_no_gpu_env(),
        "RMLX_SKIP_GPU unset must return false"
    );
    unsafe { std::env::set_var("RMLX_SKIP_GPU", "1") };
    assert!(skip_if_no_gpu_env(), "RMLX_SKIP_GPU=1 must return true");
    // restore
    match prior {
        Some(v) => unsafe { std::env::set_var("RMLX_SKIP_GPU", v) },
        None => unsafe { std::env::remove_var("RMLX_SKIP_GPU") },
    }
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
