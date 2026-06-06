//! Shared UTC timestamp helper.
//!
//! Centralises the now_iso8601 implementation that was previously duplicated
//! across migrate.rs and prompts.rs (three call sites total).

use crate::error::{Error, Result};

/// Return the current UTC time as an ISO-8601 string, e.g. `2026-05-10T07:00:00Z`.
pub fn now_iso8601() -> Result<String> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::Schema(format!("system clock error: {e}")))?
        .as_secs();

    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400; // days since 1970-01-01

    let (year, month, day) = days_to_ymd(days);
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z"
    ))
}

/// Convert days-since-epoch (1970-01-01) to (year, month, day).
///
/// Algorithm: civil_from_days — Howard Hinnant, public domain.
fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m as u32, d as u32)
}
