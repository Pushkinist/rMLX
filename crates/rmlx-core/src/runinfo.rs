//! Run identity used by logs/ and metrics/ for filenames.
//!
//! Format: `YYYYMMDD-HHMMSS-<short-git-sha|dirty>`.

use std::process::Command;

/// Build a stable run-id for the current process.
///
/// Used by both `logs/<run-id>.jsonl` and `metrics/<run-id>.jsonl`.
pub fn make_run_id() -> String {
    let ts = chrono_now_compact();
    let sha = git_short_sha().unwrap_or_else(|| "dirty".to_string());
    format!("{ts}-{sha}")
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

/// Return the short git SHA of the current HEAD, or `None` when the working
/// directory has no git repo (e.g. an installed binary without a checkout).
pub fn git_short_sha() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        return None;
    }
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|o| !o.stdout.is_empty());
    Some(if dirty { format!("{sha}-dirty") } else { sha })
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
