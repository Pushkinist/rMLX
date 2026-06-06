//! — per-project cap defaults from `<RMLX_HOME>/projects.toml`.
//!
//! Coding agents or operators can write a `projects.toml` in the `.rmlx/` data
//! root to set stable per-project SSD and RAM budgets without long CLI lines.
//! `rmlx serve --project NAME` inherits per-project caps from `[project.NAME]`
//! and falls back to `[global]` defaults, then to hard-coded built-ins.
//!
//! ## File shape
//!
//! ```toml
//! [global]
//! ssd_pool_gb = 200.0 # default --kv-ssd-global-gb
//! ram_prompt_cache_gb = 2.0 # default --prompt-cache-ram-gb
//!
//! [project.alpha]
//! ssd_cap_gb = 50.0
//!
//! [project.beta]
//! ssd_cap_gb = 30.0
//! ```
//!
//! ## Resolution path
//!
//! The resolved path is `<RMLX_HOME>/projects.toml` (via
//! [`rmlx_core::paths::projects_toml_path`]). The file is **optional**: a
//! missing file is a silent no-op (built-in defaults apply). A malformed file
//! returns `Err` and the CLI surfaces it as a startup error (exit 2).
//!
//! Edits to the file take effect on the next `rmlx serve` restart (no live
//! reload). The file is rMLX-read-only: `rmlx` never writes it; the operator
//! or a coding agent edits it manually.
//!
//! ## Precedence
//!
//! ```text
//! CLI flag > [project.<name>] > [global] > built-in default
//! ```
//!
//! Unknown `--project` names use global defaults — no warning, no auto-create.

use std::collections::HashMap;

use serde::Deserialize;
use thiserror::Error;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors returned by [`load`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProjectsConfigError {
    /// The file exists but is not valid TOML or does not match the schema.
    #[error("projects.toml parse error: {0}")]
    Parse(#[from] toml::de::Error),
    /// I/O error reading the file (distinct from "file not found", which is Ok).
    #[error("projects.toml read error: {0}")]
    Io(#[from] std::io::Error),
}

// ── Schema types (Deserialize only — rMLX never writes this file) ─────────

/// `[global]` section of `projects.toml`.
#[derive(Debug, Default, Clone, Deserialize)]
#[non_exhaustive]
pub struct GlobalConfig {
    /// Default `--kv-ssd-global-gb` (SSD pool ceiling across all namespaces).
    /// `None` = absent from file (caller uses built-in default).
    pub ssd_pool_gb: Option<f64>,
    /// Default `--prompt-cache-ram-gb`.
    /// `None` = absent from file (caller uses built-in default).
    pub ram_prompt_cache_gb: Option<f64>,
}

/// `[project.<name>]` section of `projects.toml`.
#[derive(Debug, Default, Clone, Deserialize)]
#[non_exhaustive]
pub struct ProjectConfig {
    /// Per-namespace SSD cap (`--kv-ssd-cache-gb` for this project).
    /// `None` = absent from file (falls back to global then built-in).
    pub ssd_cap_gb: Option<f64>,
}

/// Top-level `projects.toml` structure.
#[derive(Debug, Default, Clone, Deserialize)]
#[non_exhaustive]
pub struct ProjectsConfig {
    /// `[global]` section.
    #[serde(default)]
    pub global: GlobalConfig,
    /// `[project.<name>]` sections, keyed by project name.
    #[serde(default)]
    pub project: HashMap<String, ProjectConfig>,
}

// ── Load ─────────────────────────────────────────────────────────────────────

/// Load a `projects.toml` from an explicit path.
///
/// - File absent or empty → `Ok(ProjectsConfig::default())` (silent no-op).
/// - File present + valid → `Ok(config)`.
/// - File present + malformed → `Err(ProjectsConfigError::Parse(...))`.
///
/// This is the testable inner implementation. Production code calls [`load`],
/// which resolves the canonical path via [`rmlx_core::paths::projects_toml_path`].
pub(crate) fn load_from_path(
    path: &std::path::Path,
) -> Result<ProjectsConfig, ProjectsConfigError> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ProjectsConfig::default());
        }
        Err(e) => return Err(ProjectsConfigError::Io(e)),
    };
    // Empty file (or all-whitespace) is treated the same as missing.
    if raw.trim().is_empty() {
        return Ok(ProjectsConfig::default());
    }
    let cfg: ProjectsConfig = toml::from_str(&raw)?;
    Ok(cfg)
}

/// Load `<RMLX_HOME>/projects.toml`.
///
/// - File absent or empty → `Ok(ProjectsConfig::default())` (silent no-op).
/// - File present + valid → `Ok(config)`.
/// - File present + malformed → `Err(ProjectsConfigError::Parse(...))`.
///
/// This function does not log. Callers are responsible for emitting the startup
/// `info!` event if they choose to (e.g. only when sections are actually applied).
pub fn load() -> Result<ProjectsConfig, ProjectsConfigError> {
    load_from_path(&crate::paths::projects_toml_path())
}

// ── Resolved caps ─────────────────────────────────────────────────────────

/// The three cap knobs exposed via CLI and `projects.toml`.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — three cap knobs tied to CLI flags; adding a cap requires updating run_serve, CliCaps, and resolution logic"
)]
pub struct ResolvedCaps {
    /// Effective `--kv-ssd-global-gb` value (0.0 = no global ceiling).
    pub ssd_pool_gb: f64,
    /// Effective `--kv-ssd-cache-gb` value for the active project namespace
    /// (0.0 = SSD tier OFF).
    pub ssd_cap_gb: f64,
    /// Effective `--prompt-cache-ram-gb` (`None` = use built-in default).
    pub ram_prompt_cache_gb: Option<f64>,
}

/// CLI caps passed by the caller. `None` = "not set by user".
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — mirrors the three CLI flags; adding a cap requires updating run_serve and resolution logic"
)]
pub struct CliCaps {
    /// `None` = CLI did not pass a non-zero value; the file `[global].ssd_pool_gb` / built-in default applies.
    /// `Some(v)` = CLI explicitly set `v > 0` and wins over file (precedence: CLI > project > global > default).
    pub ssd_pool_gb: Option<f64>,
    /// `None` = CLI did not pass a non-zero value; the file `[global].ssd_cap_gb` / built-in default applies.
    /// `Some(v)` = CLI explicitly set `v > 0` and wins over file (precedence: CLI > project > global > default).
    pub ssd_cap_gb: Option<f64>,
    /// `None` = CLI did not pass a non-zero value; the file `[global].ram_prompt_cache_gb` / built-in default applies.
    /// `Some(v)` = CLI explicitly set `v > 0` and wins over file (precedence: CLI > project > global > default).
    pub ram_prompt_cache_gb: Option<f64>,
}

/// Resolve effective caps following the precedence chain:
/// `CLI flag > [project.<name>] > [global] > built-in default`.
///
/// - `project`: the `--project NAME` value (or `None` if not passed).
/// - Unknown project names silently fall back to global defaults.
/// - Built-in defaults: `ssd_pool_gb = 0.0`, `ssd_cap_gb = 0.0`,
///   `ram_prompt_cache_gb = None` (which the prompt-cache layer maps to 2 GiB).
pub fn resolve_caps(cli: &CliCaps, config: &ProjectsConfig, project: Option<&str>) -> ResolvedCaps {
    // Look up the [project.<name>] section (absent project name → None section).
    let proj_cfg: Option<&ProjectConfig> = project.and_then(|n| config.project.get(n));

    // ssd_pool_gb: CLI > [global] > built-in 0.0
    let ssd_pool_gb = cli
        .ssd_pool_gb
        .unwrap_or_else(|| config.global.ssd_pool_gb.unwrap_or(0.0));

    // ssd_cap_gb: CLI > [project.<name>].ssd_cap_gb > [global] (no per-ns field in global) > built-in 0.0
    // Note: [global] has no ssd_cap_gb; fallback is built-in 0.0.
    let ssd_cap_gb = cli
        .ssd_cap_gb
        .unwrap_or_else(|| proj_cfg.and_then(|p| p.ssd_cap_gb).unwrap_or(0.0));

    // ram_prompt_cache_gb: CLI > [global] > None (built-in handled downstream)
    let ram_prompt_cache_gb = cli
        .ram_prompt_cache_gb
        .or(config.global.ram_prompt_cache_gb);

    ResolvedCaps {
        ssd_pool_gb,
        ssd_cap_gb,
        ram_prompt_cache_gb,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "projects_config_tests.rs"]
mod tests;
