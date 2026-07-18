//! Unit tests for the `rotor_qjl_enabled()` toggle.
#![allow(unsafe_code)]

use super::*;

/// Default (no CLI install, no env) is `false`: the rotor Metal fused-decode
/// path is reachable at stock defaults, and QJL is opt-in.
#[test]
fn rotor_qjl_default_is_off() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // Cannot pre-empt OnceLock state in tests reliably; skip when CLI is set.
    // Test the env-fallback path explicitly.
    if ROTOR_QJL_CLI.get().is_some() {
        // Another test in the same process already installed; skip the env
        // assertion. The default-off behavior is also covered by the store-gate
        // dispatch-witness test.
        return;
    }
    // Ensure env is not set in this thread before the read.
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    let prev = std::env::var(ROTOR_QJL_ENV).ok();
    unsafe { std::env::remove_var(ROTOR_QJL_ENV) };
    assert!(!rotor_qjl_enabled(), "default rotor QJL must be OFF");
    if let Some(p) = prev {
        unsafe { std::env::set_var(ROTOR_QJL_ENV, p) };
    }
}
