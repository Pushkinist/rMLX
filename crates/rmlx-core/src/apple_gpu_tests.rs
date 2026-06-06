//! Unit tests for the Apple GPU-family probe.
//!
//! These tests exercise the pure parser (`parse_apple_generation`,
//! `family_for_m_number`) without going through sysctl, so they run on any
//! host (including CI Linux). The sysctl path is exercised by the
//! `runs_on_apple_silicon` smoke test, which is `#[ignore]` and only
//! meaningful on a macOS Apple Silicon machine.

use super::{apple_silicon_generation, family_for_m_number, parse_apple_generation};

#[test]
fn family_mapping_m1_to_m5() {
    assert_eq!(family_for_m_number(1), Some(7));
    assert_eq!(family_for_m_number(2), Some(8));
    assert_eq!(family_for_m_number(3), Some(9));
    // M4 stays on Apple9 — Apple kept the same GPU feature set on M4.
    assert_eq!(family_for_m_number(4), Some(9));
    assert_eq!(family_for_m_number(5), Some(10));
}

#[test]
fn family_mapping_future_proof() {
    // M6+ — assume one family bump per generation, conservative side.
    assert_eq!(family_for_m_number(6), Some(11));
    assert_eq!(family_for_m_number(7), Some(12));
}

#[test]
fn family_mapping_zero_is_invalid() {
    assert_eq!(family_for_m_number(0), None);
}

#[test]
fn parse_brand_strings() {
    assert_eq!(parse_apple_generation("Apple M1"), Some(7));
    assert_eq!(parse_apple_generation("Apple M1 Max"), Some(7));
    assert_eq!(parse_apple_generation("Apple M2 Pro"), Some(8));
    assert_eq!(parse_apple_generation("Apple M3 Max"), Some(9));
    assert_eq!(parse_apple_generation("Apple M4"), Some(9));
    assert_eq!(parse_apple_generation("Apple M5 Pro"), Some(10));
    assert_eq!(parse_apple_generation("Apple M5 Max"), Some(10));
}

#[test]
fn parse_handles_trailing_nul_and_whitespace() {
    assert_eq!(parse_apple_generation("Apple M3 Max\0"), Some(9));
    assert_eq!(parse_apple_generation("  Apple M3 Max  "), Some(9));
}

#[test]
fn parse_handles_lower_case() {
    assert_eq!(parse_apple_generation("apple m3"), Some(9));
}

#[test]
fn parse_handles_upper_case() {
    // Docs claim case-insensitive on "apple" but the prefix
    // matcher previously only covered "Apple " and "apple ". Add the
    // uppercase "APPLE " variant.
    assert_eq!(parse_apple_generation("APPLE M3"), Some(9));
}

#[test]
fn parse_rejects_non_apple() {
    // Intel Mac brand strings — out of scope per CLAUDE.md hard rule 1, but
    // the parser still returns None rather than panicking.
    assert_eq!(
        parse_apple_generation("Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz"),
        None
    );
}

#[test]
fn parse_rejects_garbage() {
    assert_eq!(parse_apple_generation(""), None);
    assert_eq!(parse_apple_generation("Apple"), None);
    assert_eq!(parse_apple_generation("Apple M"), None);
    assert_eq!(parse_apple_generation("Apple Mfoo"), None);
}

/// Smoke test: when run on a real Apple Silicon Mac this should return
/// `Some(n)` where `n >= 7`. Marked `#[ignore]` because CI Linux must stay
/// green.
#[test]
#[ignore = "requires Apple Silicon host (skipped on CI Linux); enable with `--include-ignored`"]
fn runs_on_apple_silicon() {
    let gen = apple_silicon_generation();
    // Open-ended lower bound — the parser future-proofs to M6+
    // via `family_for_m_number`, so any upper cap (was `..=12`) would silently
    // start failing once Apple ships an M-series past the cap.
    assert!(
        matches!(gen, Some(7..)),
        "expected Apple Silicon family >= 7, got {gen:?}"
    );
}
