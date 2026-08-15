use super::{
    cosine_similarity_per_row, fwht_normalize, lcg_data, skip_if_no_gpu_env, skip_value_means_skip,
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
