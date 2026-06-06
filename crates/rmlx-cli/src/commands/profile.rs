// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Named server profiles persisted at `<RMLX_HOME>/profiles.toml`.
//!
//! A profile is a named preset of `rmlx serve` launch parameters. It is kept
//! deliberately SEPARATE from `RegistryConfig` (which stores model *identity*,
//! not launch params). The file format is:
//!
//! ```toml
//! [profile.myrun]
//! model = "/abs/path/to/snapshot"
//! port = 9001
//! host = "0.0.0.0"
//! kv_quant = "k8v4"
//! max_ctx = 8192
//! ```
//!
//! Resolution: `[profile.<name>]` is loaded from [`rmlx_core::paths::profiles_path`]
//! (never CWD). A `serve --profile <name>` invocation merges the profile under
//! the CLI flags — **any flag the user actually passed on the CLI wins**; the
//! profile only supplies values for flags left at their absence (`None`).
//!
//! Scope is `serve` only. The set of bindable fields mirrors the
//! profile-relevant subset of `Cmd::Serve` flags. Boolean toggles
//! (`--turbo-flash`, …) and the per-side `--cache-type-*` codecs are
//! intentionally CLI-only — a `false`/absent boolean cannot be cleanly
//! distinguished from "not set" at the clap layer, so binding them would make
//! the override semantics ambiguous. Use `--kv-quant` in a profile instead.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level `profiles.toml` document: `[profile.<name>]` tables.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProfilesFile {
    /// Map of profile name → launch preset. Serialises as `[profile.<name>]`.
    #[serde(default)]
    pub profile: BTreeMap<String, ServeProfile>,
}

/// One named `serve` launch preset. Every field is optional — a profile only
/// overrides the flags it names; unset fields fall through to the CLI default.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ServeProfile {
    pub model: Option<PathBuf>,
    pub registry: Option<PathBuf>,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub device: Option<String>,
    pub kv_quant: Option<String>,
    pub max_ctx: Option<u32>,
    pub idle_timeout_secs: Option<u64>,
    pub prompt_cache_slots: Option<usize>,
    pub draft_model: Option<PathBuf>,
    pub max_tokens_cap: Option<u32>,
    pub max_timeout_secs: Option<u64>,
    pub max_loaded_models: Option<usize>,
    pub max_queue_depth: Option<usize>,
    pub default_temperature: Option<f32>,
}

impl ProfilesFile {
    /// Parse a `profiles.toml` string. Round-trips with [`Self::to_toml`].
    pub(crate) fn from_toml(s: &str) -> Result<Self> {
        toml::from_str(s).context("parsing profiles.toml")
    }

    /// Serialise to TOML (used by tests / future `profile set`).
    #[cfg(test)]
    pub(crate) fn to_toml(&self) -> Result<String> {
        toml::to_string(self).context("serialising profiles.toml")
    }

    /// Load from [`rmlx_core::paths::profiles_path`]. A missing file is not an
    /// error — it yields an empty document (no profiles defined).
    pub(crate) fn load() -> Result<Self> {
        let path = rmlx_core::paths::profiles_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::from_toml(&raw)
    }

    /// Look up a profile by name, returning a descriptive error listing the
    /// known names when it is missing.
    pub(crate) fn get(&self, name: &str) -> Result<&ServeProfile> {
        self.profile.get(name).ok_or_else(|| {
            let known: Vec<&str> = self.profile.keys().map(String::as_str).collect();
            anyhow::anyhow!(
                "profile '{name}' not found in {}; known profiles: [{}]",
                rmlx_core::paths::profiles_path().display(),
                known.join(", ")
            )
        })
    }
}

/// `rmlx profile list` — print the names of all defined profiles, one per line.
/// Reads from [`rmlx_core::paths::profiles_path`]; a missing file prints nothing
/// and exits 0.
pub(crate) fn run_profile_list() -> Result<()> {
    let file = ProfilesFile::load()?;
    if file.profile.is_empty() {
        tracing::info!(
            path = %rmlx_core::paths::profiles_path().display(),
            "no server profiles defined"
        );
        return Ok(());
    }
    for name in file.profile.keys() {
        println!("{name}");
    }
    Ok(())
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
