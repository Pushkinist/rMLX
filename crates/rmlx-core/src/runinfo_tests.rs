use super::*;

#[test]
fn run_id_shape() {
    let id = make_run_id();
    // YYYYMMDD-HHMMSS-... length >= 16
    assert!(id.len() >= 16, "got: {id}");
    assert_eq!(&id[8..9], "-");
}

#[test]
fn epoch_anchor() {
    assert_eq!(unix_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
}

#[test]
fn known_date() {
    // 2025-01-01 00:00:00 UTC = 1735689600 (well-known anchor).
    assert_eq!(unix_to_ymdhms(1_735_689_600), (2025, 1, 1, 0, 0, 0));
}

#[test]
fn backend_version_is_workspace_semver() {
    let v = backend_version();
    let parts: Vec<&str> = v.split('.').collect();
    assert_eq!(parts.len(), 3, "expected MAJOR.MINOR.PATCH, got {v:?}");
    for p in parts {
        assert!(
            !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()),
            "non-numeric component in {v:?}"
        );
    }
}

#[test]
fn build_profile_is_stamped_by_build_script() {
    // Under `cargo test` the profile dir is `debug` (the test profile inherits
    // dev). The point of the assertion is that build.rs stamped *something*
    // real — never the empty string, never a `cfg!(debug_assertions)` guess.
    let p = build_profile();
    assert!(!p.is_empty());
    assert_ne!(p, "unknown", "build.rs failed to parse OUT_DIR");
    assert_eq!(
        p, "debug",
        "cargo test runs under the dev-inherited profile"
    );
}
