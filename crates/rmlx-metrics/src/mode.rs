//! Process-wide metrics kill switch.
//!
//! Resolved exactly once, at process start, from the global `--metrics` CLI
//! flag. Every writer asks this module instead of carrying its own toggle —
//! the same single-integration-point rule the §8.5 record itself follows.
//!
//! `off` is a no-op at the **producer**: the drainer task is never spawned and
//! the SQLite file is never opened, so nothing is built and thrown away. It
//! disables *writing* only — `rmlx metrics best` / `export` / `query` read the
//! DB regardless, and the explicit `rmlx metrics record` / `migrate` commands
//! are user-invoked writes, not telemetry.
//!
//! # Public API
//!
//! - [`MetricsMode`] — off / events / full.
//! - [`init`] — set the mode once, at process start.
//! - [`current`] — read it. Defaults to [`MetricsMode::Full`] when never set
//!   (library use, tests), preserving historical behaviour.
//! - [`events_enabled`] / [`observations_enabled`] — the two questions writers
//!   actually ask.

use std::sync::OnceLock;

/// How much of the metrics subsystem writes to the DB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed user-facing mode set, 1:1 with the --metrics flag values; a new mode is a CLI contract change"
)]
pub enum MetricsMode {
    /// No DB writes at all. No drainer task, no SQLite file.
    Off,
    /// Runtime `events` only — no bench `observations`.
    Events,
    /// Everything. The default; existing behaviour.
    #[default]
    Full,
}

impl MetricsMode {
    /// Parse the `--metrics` flag value.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "events" => Some(Self::Events),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    /// The flag value that selects this mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Events => "events",
            Self::Full => "full",
        }
    }

    /// True when the `events` table may be written under this mode.
    ///
    /// The predicate lives on the enum, not just as global-reading free
    /// functions below, so it can be unit-tested against an explicit value —
    /// asserting against a value read through the process-global `OnceLock`
    /// would race any other test in the same binary that also touches it.
    pub fn events_enabled(self) -> bool {
        self != Self::Off
    }

    /// True when the `observations` table may be written by telemetry under
    /// this mode.
    pub fn observations_enabled(self) -> bool {
        self == Self::Full
    }
}

static MODE: OnceLock<MetricsMode> = OnceLock::new();

/// Set the process metrics mode. First call wins; later calls are ignored.
///
/// Called once from the CLI entry point after flag parsing.
pub fn init(mode: MetricsMode) {
    let _ = MODE.set(mode);
}

/// The active mode. [`MetricsMode::Full`] when [`init`] was never called.
pub fn current() -> MetricsMode {
    MODE.get().copied().unwrap_or_default()
}

/// True when the `events` table may be written (modes `events`, `full`).
pub fn events_enabled() -> bool {
    current().events_enabled()
}

/// True when the `observations` table may be written by telemetry (mode `full`).
pub fn observations_enabled() -> bool {
    current().observations_enabled()
}

#[cfg(test)]
#[path = "mode_tests.rs"]
mod tests;
