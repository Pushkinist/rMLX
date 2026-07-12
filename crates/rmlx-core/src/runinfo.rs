//! Run identity: who this binary is, and which run it is.
//!
//! Run-id format: `YYYYMMDD-HHMMSS-<short-git-sha|nogit>`.
//!
//! [`backend_version`], [`build_profile`] and [`git_short_sha`] are the single
//! source for the identity a metrics row records about the binary that
//! produced it. Nothing else may hand-roll them — see `docs/METRICS_DB.md` §8.5.
//!
//! **Exact semantics of the `<sha>[-dirty]` pair** (CLAUDE.md hard rule 7 —
//! document the truth, not the docstring):
//!
//! `<sha>` is the commit this binary was compiled from, baked in at build
//! time. `-dirty` (applied by [`crate::runinfo`]'s callers, not baked into the
//! sha itself — see [`source_tree_is_dirty`]) means the source tree at that
//! same compile-time path had uncommitted changes when *this process*
//! started — not at build time. An installed binary with no source tree on
//! this machine carries no `-dirty` suffix, ever: there is nothing on disk
//! that could be dirty.

/// Build a stable run-id for the current process.
///
/// Used by both `logs/<run-id>.jsonl` and `metrics/<run-id>.jsonl`. Never
/// dirty-suffixed — see [`source_tree_is_dirty`] for the (deliberately
/// separate, runtime-resolved) working-tree-cleanliness signal that
/// `rmlx_metrics::identity::RunIdentity` folds into `git_sha`. A run-id is
/// just a tracking string (`docs/METRICS_DB.md` §3.2), not a value compared
/// across runs, so it does not need that extra runtime check.
pub fn make_run_id() -> String {
    let ts = chrono_now_compact();
    let sha = git_short_sha().unwrap_or_else(|| "nogit".to_string());
    format!("{ts}-{sha}")
}

/// Semver of this binary — the single source for `observations.backend_version`.
///
/// Every crate in the workspace sets `version.workspace = true`, so this is
/// `[workspace.package].version` verbatim.
pub fn backend_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The Cargo profile this binary was built under, e.g. `release`,
/// `release-perf`, `release-debug`, `debug`.
///
/// Stamped by `build.rs` from `OUT_DIR`. `cfg!(debug_assertions)` cannot be
/// used here: it is off for all three release profiles, so it reports every
/// one of them as plain `"release"` and silently makes cross-profile
/// comparisons look like same-profile ones.
///
/// Falls back to `"unknown"` when the build layout is unrecognised — an honest
/// unknown beats a wrong label.
pub fn build_profile() -> &'static str {
    env!("RMLX_BUILD_PROFILE")
}

fn chrono_now_compact() -> String {
    // Avoid pulling chrono into rmlx-core; format manually.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (y, m, d, hh, mm, ss) = unix_to_ymdhms(secs);
    format!("{y:04}{m:02}{d:02}-{hh:02}{mm:02}{ss:02}")
}

/// Short git SHA of the commit that produced *this binary*, or `None` when
/// there was no git checkout to read at build time (e.g. built from a source
/// tarball), or the checkout found was not this workspace (see `build.rs`).
///
/// **Never `-dirty` suffixed.** Baked in at compile time by `build.rs`,
/// anchored to the workspace root — deliberately NOT read at runtime via `git
/// rev-parse` in the process's working directory. An installed `rmlx` is
/// normally launched from someone else's project (`rmlx serve` inside a
/// user's repo): a runtime `git` invocation would stamp *that* repo's SHA
/// into every metrics row this process produces. The whole point of this
/// function is identity that is actually the binary's, not whatever happens
/// to be in the current directory.
///
/// Working-tree cleanliness is a *runtime* fact (whether the tree has changed
/// since this exact binary was compiled) and cannot be baked in without going
/// stale the moment a source file is edited without touching whatever narrow
/// set of paths a build script watches — see [`source_tree_is_dirty`].
pub fn git_short_sha() -> Option<String> {
    let sha = env!("RMLX_GIT_SHA");
    if sha.is_empty() {
        None
    } else {
        Some(sha.to_string())
    }
}

/// True if the source tree that built this binary currently has uncommitted
/// changes, per one `git status --porcelain` call against the workspace root
/// [`build.rs`](../../build.rs) stamped at compile time (never the runtime
/// working directory).
///
/// **Call this at most once per process** (e.g. inside a cache /
/// `OnceLock` initializer) — it does real subprocess I/O and is not meant for
/// a hot path. `rmlx_metrics::identity::RunIdentity::get()` is the one
/// sanctioned caller.
///
/// Always `false` — never claims dirty — when there is no compile-time source
/// root to check: an installed binary has no source tree on the machine it
/// runs on, so there is nothing that could be dirty. Also `false` when that
/// path no longer exists (the checkout was deleted since this binary was
/// built) or is no longer a git repository.
pub fn source_tree_is_dirty() -> bool {
    let root = env!("RMLX_SOURCE_ROOT");
    if root.is_empty() || !std::path::Path::new(root).is_dir() {
        return false;
    }
    std::process::Command::new("git")
        .args(["-C", root, "status", "--porcelain"])
        .output()
        .is_ok_and(|o| o.status.success() && !o.stdout.is_empty())
}

/// Civil-time conversion for UTC. No leap-second handling — we don't need it.
fn unix_to_ymdhms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;

    // 1970-01-01 is day 0 (Thursday).
    let mut y: i64 = 1970;
    let mut d_left = days;
    loop {
        let leap = is_leap(y);
        let yd = if leap { 366 } else { 365 };
        if d_left < yd {
            break;
        }
        d_left -= yd;
        y += 1;
    }
    let leap = is_leap(y);
    let mdays: [i64; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m: u32 = 1;
    for &nd in &mdays {
        if d_left < nd {
            break;
        }
        d_left -= nd;
        m += 1;
    }
    let d = (d_left as u32) + 1;
    (y as u32, m, d, hh, mm, ss)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

#[cfg(test)]
#[path = "runinfo_tests.rs"]
mod tests;
