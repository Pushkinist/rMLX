use super::*;

#[test]
fn test_turbo_flash_disabled_by_default() {
    // The default policy selects the generic path. This guards against
    // accidentally flipping default-ON.
    assert!(
        !DispatchPolicy::default().turbo_flash,
        "TurboFlash must be default-OFF"
    );
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
    // 1. policy.turbo_flash
    // 2. !corrupted
    // 3. q_seq == 1
    // 4. kv_seq > policy.turbo_flash_min_kv_seq

    let off = DispatchPolicy::default();
    assert!(!turbo_flash_should_run(&off, 1, 8192));
    assert!(!turbo_flash_should_run(&off, 1, 100));

    // Each remaining condition is checked against a policy that satisfies the
    // gate, so a dropped condition cannot pass by accident.
    let on = DispatchPolicy {
        turbo_flash: true,
        ..DispatchPolicy::default()
    };
    assert!(turbo_flash_should_run(&on, 1, 8192), "gate must open");
    assert!(
        !turbo_flash_should_run(&on, 1, 100),
        "kv_seq below the policy threshold must not run"
    );
    assert!(
        !turbo_flash_should_run(&on, 2, 8192),
        "prefill (q_seq > 1) must not run"
    );
    let low_threshold = DispatchPolicy {
        turbo_flash: true,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    };
    assert!(
        turbo_flash_should_run(&low_threshold, 1, 100),
        "the threshold must come from the policy, not a constant"
    );
}
