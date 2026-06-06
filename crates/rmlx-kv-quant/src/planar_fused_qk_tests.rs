//! Tests for the `planar_fused_qk_enabled()` toggle.

use super::*;

/// Without an explicit install, the default is ON per spec.
///
/// CLI installs are process-global via OnceLock; if another test in the same
/// binary has already installed a value, skip the env assertion (same pattern
/// as `rotor_qjl_default_is_on`).  Default-on is also exercised by the
/// production parity tests.
#[test]
fn planar_fused_qk_default_is_on() {
    if PLANAR_FUSED_QK_CLI.get().is_some() {
        return;
    }
    assert!(
        planar_fused_qk_enabled(),
        "default planar-fused-qk must be ON"
    );
}
