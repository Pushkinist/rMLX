//! Run identity: who this binary is, and which run it is.
//!
//! Run-id format: `YYYYMMDD-HHMMSS-<short-git-sha|dirty>`.
//!
//! [`backend_version`], [`build_profile`] and [`git_short_sha`] are the single
//! source for the identity a metrics row records about the binary that
//! produced it. Nothing else may hand-roll them — see `docs/METRICS_DB.md` §8.5.

/// Build a stable run-id for the current process.
///
/// Used by both `logs/<run-id>.jsonl` and `metrics/<run-id>.jsonl`.
pub fn make_run_id() -> String {
    let ts = chrono_now_compact();
    let sha = git_short_sha().unwrap_or_else(|| "dirty".to_string());
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

/// Short git SHA (`-dirty` suffixed if the checkout had uncommitted changes at
/// build time) of the source tree that produced *this binary*, or `None` when
/// there was no git checkout to read (e.g. built from a source tarball).
///
/// Stamped by `build.rs` at compile time, anchored to `CARGO_MANIFEST_DIR` —
/// deliberately NOT read at runtime via `git rev-parse` in the process's
/// working directory. An installed `rmlx` is normally launched from someone
/// else's project (`rmlx serve` inside a user's repo): a runtime `git`
/// invocation would stamp *that* repo's SHA — plus a spurious `-dirty` from
/// their uncommitted files — into every metrics row this process produces.
/// The whole point of this module is identity that is actually the binary's,
/// not whatever happens to be in the current directory.
pub fn git_short_sha() -> Option<String> {
    let sha = env!("RMLX_GIT_SHA");
    if sha.is_empty() {
        None
    } else {
        Some(sha.to_string())
    }
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
