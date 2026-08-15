use super::*;

#[test]
fn test_turbo_flash_disabled_by_default() {
    // Unless RMLX_TURBO_FLASH=1 is set in the test environment, this should
    // return false. This guards against accidentally flipping default-ON.
    //
    // To test with env var set, run:
    // RMLX_TURBO_FLASH=1 cargo test turbo_flash
    //
    // The `env::var` below is a raw, unlatched read, so it takes the env lock
    // like any other reader: `setenv` is UB against a concurrent `getenv` of ANY
    // key, and other tests in this binary write `RMLX_ROTOR_QJL`.
    let _guard = crate::test_utils::env_lock();
    if std::env::var("RMLX_TURBO_FLASH").as_deref() != Ok("1") {
        assert!(
            !turbo_flash_enabled(),
            "TurboFlash must be default-OFF (RMLX_TURBO_FLASH=1 not set)"
        );
    }
}

#[test]
fn test_smoke_probe_no_corruption() {
    // Clean output — diverse token IDs, no run of 4.
    let tokens = [1u32, 5, 3, 7, 2, 8, 4, 6, 9, 10, 11, 12];
    assert!(!smoke_probe_check(&tokens), "should not detect corruption");
}

#[test]
fn test_smoke_probe_detects_corruption() {
    // Run of 4 identical tokens — corruption signature.
    // Use a temporary AtomicBool state by resetting after test.
    let saved = FORCED_FALLBACK.load(Ordering::Relaxed);
    FORCED_FALLBACK.store(false, Ordering::Relaxed);
    let tokens = [1u32, 5, 106, 106, 106, 106, 3];
    assert!(
        smoke_probe_check(&tokens),
        "should detect corruption (run of 4 identical token 106)"
    );
    // Restore state (tests may run in any order).
    FORCED_FALLBACK.store(saved, Ordering::Relaxed);
}

#[test]
fn test_turbo_flash_should_run_gated() {
    // should_run requires:
    // 1. turbo_flash_enabled() (RMLX_TURBO_FLASH=1)
    // 2. !corrupted
    // 3. q_seq == 1
    // 4. kv_seq > TURBO_FLASH_MIN_KV_SEQ

    // With default env (RMLX_TURBO_FLASH unset or "0"), should never run.
    // Raw unlatched env read — takes the lock, as every reader must.
    let _guard = crate::test_utils::env_lock();
    if std::env::var("RMLX_TURBO_FLASH").as_deref() != Ok("1") {
        assert!(!turbo_flash_should_run(1, 8192));
        assert!(!turbo_flash_should_run(1, 100));
    }
}
