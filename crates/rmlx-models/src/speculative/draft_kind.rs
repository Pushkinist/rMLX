//! `DraftKind` enum — speculative drafter architecture family.

use std::str::FromStr;

/// Speculative drafter architecture family.
///
/// Decides which drafter loader the serve layer builds and which round loop
/// drives the request.
/// Mirrors mlx-vlm `--draft-kind choices=[dflash,eagle3,mtp]`.
///
/// `rmlx-models` carries the plain enum (no clap dep).
/// The `clap::ValueEnum` impl lives in `rmlx-cli::main` (see `DraftKindArg`).
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — three speculative drafter kinds (Mtp/DFlash/Eagle3); adding a kind requires updating SpeculativeDispatcher::new, as_str(), from_str(), and all spec-dispatch match arms"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftKind {
    /// Multi-Token Prediction drafter (Llama / Qwen3 family).
    Mtp,
    /// Draft-Flash (attention-based draft head).
    DFlash,
    /// EAGLE-3 speculative drafter.
    Eagle3,
}

impl DraftKind {
    /// Canonical kebab-case name used in log fields and env vars.
    pub fn as_str(self) -> &'static str {
        match self {
            DraftKind::Mtp => "mtp",
            DraftKind::DFlash => "dflash",
            DraftKind::Eagle3 => "eagle3",
        }
    }
}

impl std::fmt::Display for DraftKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DraftKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mtp" => Ok(DraftKind::Mtp),
            "dflash" => Ok(DraftKind::DFlash),
            "eagle3" => Ok(DraftKind::Eagle3),
            other => Err(format!(
                "unknown draft-kind '{other}'; valid values: mtp, dflash, eagle3"
            )),
        }
    }
}
