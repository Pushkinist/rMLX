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
    // Same shape the §8.5 ingest validator accepts
    // (`rmlx_metrics::identity::is_semver`): MAJOR.MINOR.PATCH with an
    // optional -pre / +build suffix, e.g. "0.3.0-rc.1". rmlx-core has no
    // internal workspace deps (see the dep-graph hard rule in CLAUDE.md), so
    // the predicate is duplicated here rather than imported — asserting the
    // stricter "all-numeric, exactly 3 dot-parts" shape used to fail on the
    // very first prerelease version bump.
    let core = v.split(['-', '+']).next().unwrap_or(v);
    let parts: Vec<&str> = core.split('.').collect();
    assert_eq!(parts.len(), 3, "expected MAJOR.MINOR.PATCH core, got {v:?}");
    for p in parts {
        assert!(
            !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()),
            "non-numeric component in {v:?}"
        );
    }
}

#[test]
fn build_profile_is_stamped_by_build_script() {
    // Deliberately does NOT pin to one profile name: CLAUDE.md Hard rule 9
    // mandates `make ci-perf` runs this same test suite under
    // `--profile release-perf`, where the correct value is "release-perf",
    // not "debug". The point of the assertion is that build.rs stamped
    // *something* real for the profile this binary was actually built
    // under — never the empty string, never "unknown" for a build cargo
    // itself understands.
    let p = build_profile();
    assert!(!p.is_empty());
    assert!(
        matches!(p, "debug" | "release" | "release-perf" | "release-debug"),
        "unexpected build_profile: {p:?}"
    );
}
