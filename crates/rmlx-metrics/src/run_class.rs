//! Why a run was recorded (`docs/METRICS_SCHEMA.md` §3.2, `run_class`).
//!
//! Every `observations` row carries one class. Only [`RunClass::Gate`] rows
//! reach `bests` and the queries a gate or a published figure reads;
//! [`crate::bests_view::gate_rows_sql`] renders that filter.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The class of a recorded run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed set — the migration's CHECK constraint lists exactly these spellings, so a new class is a schema change"
)]
pub enum RunClass {
    /// A canary, a release gate, or the surviving arm of an A/B recorded once.
    Gate,
    /// One slot of an A/B comparison.
    AbArm,
    /// A profiling, traced, capture or investigation run.
    Diagnostic,
}

impl RunClass {
    /// Every class, in the order the docs list them.
    pub const ALL: [Self; 3] = [Self::Gate, Self::AbArm, Self::Diagnostic];

    /// The spelling stored in `observations.run_class`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::AbArm => "ab_arm",
            Self::Diagnostic => "diagnostic",
        }
    }

    /// The three spellings joined with `, `, for an error message.
    pub fn spellings() -> String {
        Self::ALL.map(Self::as_str).join(", ")
    }
}

impl fmt::Display for RunClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RunClass {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|class| class.as_str() == s)
            .ok_or_else(|| {
                format!(
                    "unknown run class {s:?}; expected one of {}",
                    Self::spellings()
                )
            })
    }
}

impl rusqlite::types::FromSql for RunClass {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        value
            .as_str()?
            .parse()
            .map_err(|e: String| rusqlite::types::FromSqlError::Other(e.into()))
    }
}

#[cfg(test)]
#[path = "run_class_tests.rs"]
mod tests;
