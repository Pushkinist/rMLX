//! Unit tests for the `rotor_qjl_enabled()` toggle.
#![allow(unsafe_code)]

use super::*;

/// Default (no CLI install, no env) is `true` per spec.
#[test]
fn rotor_qjl_default_is_on() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // Cannot pre-empt OnceLock state in tests reliably; use a unique env var
    // path or skip when CLI is set. Test the env-fallback path explicitly.
    if ROTOR_QJL_CLI.get().is_some() {
        // Another test in the same process already installed; skip the env
        // assertion. The default-on behavior is also verified by the
        // production codec round-trip tests.
        return;
    }
    // Ensure env is not set in this thread before the read.
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    let prev = std::env::var(ROTOR_QJL_ENV).ok();
    unsafe { std::env::remove_var(ROTOR_QJL_ENV) };
    assert!(rotor_qjl_enabled(), "default rotor QJL must be ON");
    if let Some(p) = prev {
        unsafe { std::env::set_var(ROTOR_QJL_ENV, p) };
    }
}
