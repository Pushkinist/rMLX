//! Run identity: who this binary is, and which run it is.
//!
//! Run-id format: `YYYYMMDD-HHMMSS-<version>`.
//!
//! [`backend_version`] and [`build_profile`] are the single source for the
//! identity a metrics row records about the binary that produced it. Nothing
//! else may hand-roll them — see `docs/METRICS_DB.md` §8.5.
//!
//! There is deliberately no `git_short_sha` / `source_tree_is_dirty` here.
//! Four review rounds concluded the binary cannot honestly know the commit it
//! runs from: not at runtime (the working directory it is launched from is
//! not necessarily its own source checkout — `rmlx serve` is normally started
//! from a user's project, not this repo), and not at build time either
//! (baking in a compile-time SHA, then trying to detect whether the source
//! tree had since gone "dirty" relative to it, required an entire
//! workspace-root-discovery + rerun-if-changed + runtime-probe apparatus that
//! kept producing new defects — wrong-repo detection, stale-commit detection,
//! untracked-file false positives — without any downstream consumer that
//! actually needed the binary to guess). `git_sha` is now purely
//! caller-supplied provenance, exactly like `hardware_tag` already was: a
//! bench script that runs `git rev-parse` in its own repo, or `rmlx baseline
//! --git-sha <sha>` / `rmlx eval ppl --git-sha <sha>`, supplies it. The
//! recording surface accepts provenance; it does not invent it. See
//! `docs/METRICS_DB.md` §8.5.1.

/// Build a stable run-id for the current process.
///
/// Used by both `logs/<run-id>.jsonl` and `metrics/<run-id>.jsonl`. A run-id
/// is just a tracking string (`docs/METRICS_DB.md` §3.2), not a value
/// compared across runs — the version discriminator is enough to tell apart
/// log files from different builds without claiming a commit the binary
/// cannot actually verify.
pub fn make_run_id() -> String {
    let ts = chrono_now_compact();
    format!("{ts}-{}", backend_version())
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
