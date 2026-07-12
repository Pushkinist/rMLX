//! Stamps the real Cargo profile name and build-tree identity into the binary.
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
//! ## Git identity — split in two, on purpose
//!
//! `git_sha` (the commit this binary was compiled from) is baked in HERE, at
//! compile time — never read at runtime via `git rev-parse` in the process's
//! working directory. An installed `rmlx` is normally launched from *someone
//! else's* project directory (`rmlx serve` from inside a user's repo); a
//! runtime lookup would stamp that repo's SHA into every metrics row.
//!
//! Two things this module deliberately does NOT do:
//!
//! - It never bakes in a `-dirty` suffix. Emitting `rerun-if-changed` disables
//!   cargo's default whole-package rebuild-watch; editing an ordinary source
//!   file anywhere in the workspace does not touch any of the paths watched
//!   below, so a build script that computed "dirty" once, here, would freeze
//!   that answer until something explicitly on the watch list changed —
//!   silently mislabelling every WIP build after the first as "clean". Working-tree
//!   cleanliness is instead resolved once at RUNTIME (`rmlx_core::runinfo::source_tree_is_dirty`),
//!   against the compile-time-stamped `RMLX_SOURCE_ROOT` below — that half is
//!   observable only at runtime, so that is where it belongs.
//! - It never trusts a git repo found by walking up past the workspace
//!   boundary. `git -C <dir> rev-parse` climbs parent directories until it
//!   finds ANY `.git`; extracting this source tree inside an unrelated
//!   checkout (a tarball unpacked into someone's existing repo, or `cargo
//!   install --path` run from inside one) would otherwise bake in that
//!   enclosing repo's SHA — the exact "someone else's repo" bug this module
//!   exists to prevent, relocated to build time. [`our_workspace_root`] refuses
//!   to trust a discovered repo whose toplevel isn't exactly this workspace.

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

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let source_root = our_workspace_root(&manifest_dir);

    if let Some(root) = source_root.as_deref() {
        watch_git_refs(root);
    }

    println!(
        "cargo:rustc-env=RMLX_GIT_SHA={}",
        source_root
            .as_deref()
            .and_then(git_head_sha)
            .unwrap_or_default()
    );
    // The path `runinfo::source_tree_is_dirty` runs its one runtime `git
    // status --porcelain` against. Empty when there is no trustworthy source
    // tree — an installed binary then correctly never claims dirty, because
    // (per its own doc) there is nothing on disk to have been edited.
    println!(
        "cargo:rustc-env=RMLX_SOURCE_ROOT={}",
        source_root.unwrap_or_default()
    );
}

/// The workspace root that built this crate, IFF it is a git repo whose
/// toplevel is exactly that root.
///
/// `None` when there is no git checkout, OR the nearest git repo discovered
/// from `manifest_dir` is not this workspace — an enclosing unrelated
/// checkout (this source tree extracted inside someone else's repo) or a
/// stray inner one. In either mismatch case nothing derived from that git
/// repo is trustworthy, so callers get an honest absence rather than a wrong
/// identity.
fn our_workspace_root(manifest_dir: &str) -> Option<String> {
    if manifest_dir.is_empty() {
        return None;
    }
    let expected = find_workspace_root(manifest_dir)?;
    let toplevel = git_output(manifest_dir, &["rev-parse", "--show-toplevel"])?;
    let toplevel_canon = std::fs::canonicalize(&toplevel).ok()?;
    (toplevel_canon.to_string_lossy() == expected).then_some(expected)
}

/// Walk up from `start` looking for the nearest ancestor `Cargo.toml` that
/// declares a `[workspace]` table. Bounded to a handful of levels — this
/// repo's crates sit two levels below the workspace root; the bound is
/// generous headroom, not a promise to search indefinitely.
fn find_workspace_root(start: &str) -> Option<String> {
    let mut dir = Path::new(start).to_path_buf();
    for _ in 0..8 {
        let candidate = dir.join("Cargo.toml");
        if let Ok(contents) = std::fs::read_to_string(&candidate) {
            // Line-exact match, not a substring search: `contains("[workspace]")`
            // would also fire on a comment or a `description = "...[workspace]..."`
            // string value anywhere in the file.
            if contents.lines().any(|l| l.trim() == "[workspace]") {
                return std::fs::canonicalize(&dir)
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned());
            }
        }
        if !dir.pop() {
            return None;
        }
    }
    None
}

/// Run `git -C <dir> <args>`, trimmed. `None` on any failure or empty output.
fn git_output(dir: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Short git SHA of `HEAD` at `root`. NEVER dirty-suffixed — see the module
/// doc comment for why that half is resolved at runtime instead.
fn git_head_sha(root: &str) -> Option<String> {
    git_output(root, &["rev-parse", "--short=7", "HEAD"])
}

/// Rerun this build script when the checkout's HEAD or the ref it points to
/// changes, so `RMLX_GIT_SHA` tracks new commits without a full `cargo clean`.
fn watch_git_refs(root: &str) {
    // Per-worktree HEAD: branch switches, detached-HEAD commit changes. Lives
    // in the per-worktree git-dir, not the common dir (each `git worktree
    // add` checkout has its own HEAD).
    if let Some(git_dir) = git_output(root, &["rev-parse", "--git-dir"]) {
        watch_if_exists(&resolve(root, &git_dir), "HEAD");
    }

    // The ref that actually MOVES on a same-branch commit lives in the
    // COMMON dir (refs are shared across worktrees; only HEAD is
    // per-worktree). `.git/HEAD`'s content ("ref: refs/heads/<branch>") does
    // NOT change across commits on the same branch — watching only HEAD
    // misses every commit that doesn't also happen to rewrite the index
    // (`git reset --soft`, `git branch -f`, `git update-ref`, ...).
    let Some(common_dir) = git_output(root, &["rev-parse", "--git-common-dir"]) else {
        return;
    };
    let common_dir = resolve(root, &common_dir);

    if let Some(ref_path) = git_output(root, &["symbolic-ref", "--quiet", "HEAD"]) {
        watch_if_exists(&common_dir, &ref_path);
    }
    // Branches may be packed (fresh clone, post-`git gc`) instead of living
    // as a loose ref file; watch packed-refs too so either form is caught.
    watch_if_exists(&common_dir, "packed-refs");
}

/// Join `maybe_relative` onto `base` unless it is already absolute.
fn resolve(base: &str, maybe_relative: &str) -> String {
    if Path::new(maybe_relative).is_absolute() {
        maybe_relative.to_string()
    } else {
        format!("{base}/{maybe_relative}")
    }
}

/// Emit `rerun-if-changed` only for paths that exist — cargo reruns the
/// build script on EVERY build for a `rerun-if-changed` path that does not
/// exist, which would defeat the point of declaring a watch set at all.
fn watch_if_exists(dir: &str, rel: &str) {
    let path = format!("{dir}/{rel}");
    if Path::new(&path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}
