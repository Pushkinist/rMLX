//! Scope file parsing (`config/scope.toml`).
//!
//! Selects the in-scope models for `BENCHMARK_CHAMPIONS.md` rendering and
//! declares which (model × backend) cells are structurally unsupported
//! (rendered as `N/A` instead of `-`).
//!
//! The DB stays the source of truth — this file is just a filter and
//! display layer. Editing it does not mutate any observation.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

// ── Parsed shape ─────────────────────────────────────────────────────────────

/// Parsed representation of `config/scope.toml` — the list of models in scope for metrics.
#[derive(Debug, Deserialize, Clone)]
#[non_exhaustive]
pub struct ScopeFile {
    /// Models listed in the scope file (`[[model]]` TOML tables).
    #[serde(default, rename = "model")]
    pub models: Vec<ScopeModel>,
}

/// One model entry in the scope file.
#[derive(Debug, Deserialize, Clone)]
#[non_exhaustive]
pub struct ScopeModel {
    /// Model namespace (e.g. `"mlx-community"`).
    pub namespace: String,
    /// Model repository name (e.g. `"gemma-4-e4b-it-mxfp8"`).
    pub name: String,
    /// Architecture family string (e.g. `"Gemma4ForConditionalGeneration"`).
    pub arch: String,
    /// Human-readable weight quant for display (e.g. `"mxfp8"`).
    pub weight_quant_display: String,
    /// Sort order within the scope file (unique, ascending).
    pub order: i64,
    /// Alternative (namespace, name) pairs that resolve to this model.
    #[serde(default)]
    pub aliases: Vec<Alias>,
    /// Backends for which this model is known-unsupported.
    #[serde(default)]
    pub unsupported: Vec<Unsupported>,
}

/// An alternative (namespace, name) identifier for a [`ScopeModel`].
#[derive(Debug, Deserialize, Clone)]
#[non_exhaustive]
pub struct Alias {
    /// Namespace component of the alias.
    pub namespace: String,
    /// Name component of the alias.
    pub name: String,
}

/// A backend that cannot run a given [`ScopeModel`].
#[derive(Debug, Deserialize, Clone)]
#[non_exhaustive]
pub struct Unsupported {
    /// Canonical backend identifier that cannot run this model.
    pub backend: String,
    /// Human-readable reason, if documented.
    #[serde(default)]
    pub reason: Option<String>,
}

// ── Loaders ──────────────────────────────────────────────────────────────────

impl ScopeFile {
    /// Read + parse `config/scope.toml` from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Scope(format!("read {}: {e}", path.display())))?;
        Self::parse(&text).map_err(|e| Error::Scope(format!("parse {}: {e}", path.display())))
    }

    /// Parse from an in-memory string. Returns `String` errors so the
    /// caller can wrap with the source path.
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let parsed: ScopeFile = toml::from_str(text).map_err(|e| e.to_string())?;
        // Validate order is unique + ascending after sort.
        let mut sorted = parsed.models;
        sorted.sort_by_key(|m| m.order);
        let mut seen = std::collections::HashSet::new();
        for m in &sorted {
            if !seen.insert(m.order) {
                return Err(format!(
                    "duplicate order={} on model {}/{}",
                    m.order, m.namespace, m.name
                ));
            }
        }
        Ok(ScopeFile { models: sorted })
    }

    /// True if `(namespace, model)` matches a scope entry or one of its
    /// aliases. Out-of-scope models are filtered from the export.
    pub fn matches(&self, namespace: &str, model: &str) -> Option<&ScopeModel> {
        self.models.iter().find(|m| m.is_match(namespace, model))
    }
}

impl ScopeModel {
    /// Returns `true` if `(namespace, model)` matches this entry's primary or alias identifiers.
    pub fn is_match(&self, namespace: &str, model: &str) -> bool {
        if self.namespace == namespace && self.name == model {
            return true;
        }
        self.aliases
            .iter()
            .any(|a| a.namespace == namespace && a.name == model)
    }

    /// Returns `true` if `backend` is listed in the `unsupported` list for this model.
    pub fn is_backend_unsupported(&self, backend: &str) -> bool {
        self.unsupported.iter().any(|u| u.backend == backend)
    }

    /// Canonical display: `namespace__name` to match historical `__` style.
    pub fn display_id(&self) -> String {
        format!("{}__{}", self.namespace, self.name)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "scope_tests.rs"]
mod tests;
