//! Stamps the real Cargo profile name and build-tree git SHA into the binary.
//!
//! ## Build profile
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
//! ## Git SHA
//!
//! Stamped here rather than read at runtime (`git rev-parse` in the running
//! process's cwd) because an installed `rmlx` is normally launched from
//! *someone else's* project directory — `rmlx serve` from inside a user's repo
//! would otherwise stamp that repo's SHA (plus `-dirty` from their uncommitted
//! files) into every metrics row. Anchored to `CARGO_MANIFEST_DIR` — the source
//! tree that produced *this* binary — never the runtime working directory.
//! Empty (⇒ `None` at runtime) when there is no git checkout, e.g. building
//! from a source tarball.

use std::path::Path;

// `profile_from_out_dir` is shared with `src/build_info.rs` via `include!` so
// it can be unit-tested under `cargo test` — `build.rs` is a standalone
// program `cargo test` never runs as a test target, so a copy pasted into
// `src/` would silently drift from what actually ships.
include!("build_support.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    println!(
        "cargo:rustc-env=RMLX_BUILD_PROFILE={}",
        profile_from_out_dir(&out_dir)
    );

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    watch_git_dir(&manifest_dir);
    println!(
        "cargo:rustc-env=RMLX_GIT_SHA={}",
        git_sha_at_compile_time(&manifest_dir)
    );
}

/// Short git SHA (`-dirty` suffixed if the tree had uncommitted changes at
/// build time) of the checkout at `manifest_dir`, or empty when there is none.
fn git_sha_at_compile_time(manifest_dir: &str) -> String {
    if manifest_dir.is_empty() {
        return String::new();
    }
    let Ok(sha_out) = std::process::Command::new("git")
        .args(["-C", manifest_dir, "rev-parse", "--short=7", "HEAD"])
        .output()
    else {
        return String::new();
    };
    if !sha_out.status.success() {
        return String::new();
    }
    let sha = String::from_utf8_lossy(&sha_out.stdout).trim().to_string();
    if sha.is_empty() {
        return String::new();
    }
    let dirty = std::process::Command::new("git")
        .args(["-C", manifest_dir, "status", "--porcelain"])
        .output()
        .is_ok_and(|o| !o.stdout.is_empty());
    if dirty {
        format!("{sha}-dirty")
    } else {
        sha
    }
}

/// Best-effort: rerun this build script when the checkout's HEAD or staged
/// index changes, so `RMLX_GIT_SHA` tracks new commits without a full `cargo
/// clean`. Uncommitted-but-unstaged edits (dirty state) are not tracked by
/// either file — that half is necessarily best-effort at compile time.
fn watch_git_dir(manifest_dir: &str) {
    if manifest_dir.is_empty() {
        return;
    }
    let Ok(out) = std::process::Command::new("git")
        .args(["-C", manifest_dir, "rev-parse", "--git-dir"])
        .output()
    else {
        return;
    };
    if !out.status.success() {
        return;
    }
    let git_dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if git_dir.is_empty() {
        return;
    }
    let git_dir = if Path::new(&git_dir).is_absolute() {
        git_dir
    } else {
        format!("{manifest_dir}/{git_dir}")
    };
    println!("cargo:rerun-if-changed={git_dir}/HEAD");
    println!("cargo:rerun-if-changed={git_dir}/index");
}
