use super::*;

#[test]
fn plain_debug_profile() {
    assert_eq!(
        profile_from_out_dir("/repo/target/debug/build/rmlx-core-abc123/out"),
        "debug"
    );
}

#[test]
fn plain_release_profile() {
    assert_eq!(
        profile_from_out_dir("/repo/target/release/build/rmlx-core-abc123/out"),
        "release"
    );
}

#[test]
fn custom_profile_with_target_triple() {
    assert_eq!(
        profile_from_out_dir(
            "/repo/target/aarch64-apple-darwin/release-perf/build/rmlx-core-abc123/out"
        ),
        "release-perf"
    );
}

/// Regression test for the anchor bug: `CARGO_TARGET_DIR` (or the checkout
/// path itself) containing an earlier `build` component must not shadow the
/// real profile directory. Reproduces the reviewer's repro:
/// `CARGO_TARGET_DIR=.../scratchpad/build/target cargo test -p rmlx-core`.
#[test]
fn earlier_build_component_in_the_path_is_not_the_anchor() {
    assert_eq!(
        profile_from_out_dir(
            "/home/dev/scratchpad/build/target/release-perf/build/rmlx-core-abc123/out"
        ),
        "release-perf",
        "the LAST `build` component must anchor the profile, not the first"
    );
}

#[test]
fn multiple_earlier_build_components_still_anchor_on_the_last() {
    assert_eq!(
        profile_from_out_dir("/build/build/build/debug/build/pkg-xyz/out"),
        "debug"
    );
}

#[test]
fn no_build_component_is_unknown() {
    assert_eq!(profile_from_out_dir("/repo/target/debug/out"), "unknown");
}

#[test]
fn empty_out_dir_is_unknown() {
    assert_eq!(profile_from_out_dir(""), "unknown");
}

/// `build` as the very first path component (nothing before it) must not
/// panic on the `checked_sub` underflow and must fall back honestly.
#[test]
fn build_component_with_nothing_before_it_is_unknown() {
    assert_eq!(profile_from_out_dir("build/pkg-xyz/out"), "unknown");
}
