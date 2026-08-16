//! `DispatchPolicy` value + process-default semantics.
//!
//! Deliberately env-free: `from_env` is covered end-to-end by the `rmlx-cli`
//! subprocess tests (`tests/kernel_gate_flags.rs`), which own a private
//! environment. Mutating the environment from a parallel `cargo test` thread
//! would race every other test in this binary.

use super::{
    dispatch_policy, set_dispatch_policy, DispatchPolicy, DEFAULT_FUSED_QK_MIN_KV_SEQ,
    DEFAULT_TURBO_FLASH_MIN_KV_SEQ,
};

/// Serialises the tests that mutate the process default.
static PROCESS_DEFAULT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every kernel path is off by default, and both thresholds carry their
/// documented constant. A default-constructed policy is the generic path.
#[test]
fn default_selects_every_generic_path() {
    let p = DispatchPolicy::default();
    assert!(!p.fused_qk);
    assert!(!p.sparse_attn);
    assert!(!p.turbo_flash);
    assert!(!p.turbo_flash_lock);
    assert!(!p.planar_flash_decode);
    assert!(!p.rot_k_fused);
    assert_eq!(p.fused_qk_min_kv_seq, DEFAULT_FUSED_QK_MIN_KV_SEQ);
    assert_eq!(p.turbo_flash_min_kv_seq, DEFAULT_TURBO_FLASH_MIN_KV_SEQ);
}

/// The process default is replaceable any number of times. This is the
/// property the old `OnceLock` gates lacked: their first read froze the value
/// for the process, so an A/B driver could never present the second arm.
#[test]
fn process_default_is_replaceable_not_latched() {
    #[allow(
        clippy::unwrap_used,
        reason = "a poisoned lock here means another test in this binary panicked; failing loudly is correct"
    )]
    let _guard = PROCESS_DEFAULT_LOCK.lock().unwrap();
    let restore = dispatch_policy();

    let arm_a = DispatchPolicy {
        fused_qk: true,
        turbo_flash: true,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    };
    let arm_b = DispatchPolicy::default();

    // A → B → A. A latch would pin the first value and fail the second and
    // third assertions.
    set_dispatch_policy(arm_a);
    assert_eq!(dispatch_policy(), arm_a);
    set_dispatch_policy(arm_b);
    assert_eq!(dispatch_policy(), arm_b);
    set_dispatch_policy(arm_a);
    assert_eq!(dispatch_policy(), arm_a);

    set_dispatch_policy(restore);
}

/// A policy captured before the process default changes keeps its own value.
/// This is what lets two arms be live at once rather than merely in sequence.
#[test]
fn a_captured_policy_survives_a_later_process_default_change() {
    #[allow(
        clippy::unwrap_used,
        reason = "a poisoned lock here means another test in this binary panicked; failing loudly is correct"
    )]
    let _guard = PROCESS_DEFAULT_LOCK.lock().unwrap();
    let restore = dispatch_policy();

    set_dispatch_policy(DispatchPolicy {
        rot_k_fused: true,
        ..DispatchPolicy::default()
    });
    let captured = dispatch_policy();

    set_dispatch_policy(DispatchPolicy::default());
    assert!(
        captured.rot_k_fused,
        "a captured policy must not follow the process default"
    );
    assert!(!dispatch_policy().rot_k_fused);

    set_dispatch_policy(restore);
}
