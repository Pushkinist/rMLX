//! Stamps the real Cargo profile name into the binary.
//!
//! `cfg!(debug_assertions)` cannot tell `release`, `release-perf` and
//! `release-debug` apart — all three build with debug-assertions off, so it
//! collapses them to a single "release" label. Cargo's `PROFILE` env var is no
//! better: for custom profiles it only ever reports `"debug"` or `"release"`.
//!
//! `OUT_DIR` does carry the truth. It is laid out as
//! `<target-dir>/[<triple>/]<profile-dir>/build/<pkg>-<hash>/out`, and the
//! profile dir is named after the profile. Take the path component directly
//! before the LAST `build` component — using the first would misfire whenever
//! the target dir itself contains an earlier `build` component (a checkout
//! path or `CARGO_TARGET_DIR` under a directory literally named `build/`).
//!
//! This file used to also stamp a compile-time git commit SHA and probe a
//! runtime "is the source tree dirty" signal. Both are gone: the binary
//! cannot honestly know the commit it runs from (the working directory it is
//! launched from is not necessarily its own source checkout, and any
//! discovered `.git` may belong to an entirely different repo), and every
//! attempt to make it "know" — compile-time SHA anchored to a discovered
//! workspace root, then a runtime dirty probe against that same path — added
//! real defects across several review rounds for a value nothing downstream
//! actually needed the binary to guess. `git_sha` is now purely
//! caller-supplied provenance: `RunRecord.git_sha` / `observations.git_sha` /
//! `events.git_sha` are ordinary nullable columns a caller (a bench script
//! that already runs `git rev-parse` in its own repo, or `rmlx baseline
//! --git-sha` / `rmlx eval ppl --git-sha`) may fill in. See
//! `docs/METRICS_DB.md` §8.5.1.

use std::path::Path;

// `profile_from_out_dir` is shared with `src/build_info.rs` via `include!` so
// it can be unit-tested under `cargo test` — `build.rs` is a standalone
// program `cargo test` never runs as a test target, so a copy pasted into
// `src/` would silently drift from what actually ships.
include!("build_support.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `include!`d, not a `mod` — rustc's own dep-info would catch a change
    // here anyway, but that is an undeclared dependency in the one file
    // whose entire justification is drift-resistance. Declare it.
    println!("cargo:rerun-if-changed=build_support.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    println!(
        "cargo:rustc-env=RMLX_BUILD_PROFILE={}",
        profile_from_out_dir(&out_dir)
    );
}
