//! Tests for `rmlx_core::unified_memory`.

use super::unified_memory_gb;

/// On Apple Silicon macOS the sysctl call must return a positive GB count.
///
/// This test verifies that `unified_memory_gb()` returns `Some(x)` where x > 0.0 on this machine.
#[test]
#[cfg(target_os = "macos")]
fn unified_memory_returns_some_positive_on_macos() {
    let gb = unified_memory_gb();
    assert!(
        gb.is_some(),
        "unified_memory_gb() must return Some on macOS Apple Silicon"
    );
    let v = gb.unwrap();
    assert!(
        v > 0.0,
        "unified_memory_gb() returned Some({v}), expected > 0.0"
    );
    // Real Apple Silicon Macs ship with at least 8 GB.
    assert!(
        v >= 8.0,
        "unified_memory_gb() returned {v} GB — less than minimum Apple Silicon RAM (8 GB)"
    );
    // Sanity ceiling: no current Mac ships with > 512 GB.
    assert!(
        v <= 512.0,
        "unified_memory_gb() returned {v} GB — exceeds plausible upper bound (512 GB)"
    );
}

/// On non-macOS targets the function must return `None` (stub path).
#[test]
#[cfg(not(target_os = "macos"))]
fn unified_memory_returns_none_on_non_macos() {
    assert_eq!(unified_memory_gb(), None);
}
