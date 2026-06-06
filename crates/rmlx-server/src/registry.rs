//! Model registry: a small in-process catalog of known snapshot directories.
//!
//! Stage 1 supports a single `--model` flag; the registry has 0 or 1 entries.
//! Stage 3.5 adds `--registry <PATH>` which reads a JSON file:
//!
//! ```json
//! {
//! "models": [
//! { "id": "gemma-4-e4b", "path": "/path/to/snapshot" },
//! { "id": "qwen3.6-35b", "path": "/path/to/other" }
//! ]
//! }
//! ```
//!
//! The `--model` single-snapshot mode remains; `--registry` adds all entries.

#![allow(clippy::cognitive_complexity, clippy::missing_fields_in_debug)]
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use rmlx_loader::load_config;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::chat_template::{self, ChatTemplate};
use crate::generation_config_io::{self, GenerationConfig};
use crate::tokenizer_io;

// ── Registry config (JSON on-disk format) ─────────────────────────────────────

/// One entry in the JSON registry file.
#[allow(
    clippy::exhaustive_structs,
    reason = "wire DTO — two fields are the complete registry-file entry contract; struct-literal construction by serde; adding a field requires a serde default"
)]
#[derive(Deserialize, Debug, Clone)]
pub struct RegistryConfigEntry {
    /// Logical model ID exposed in the API. If absent, derived from path basename.
    pub id: Option<String>,
    /// Absolute or relative path to the snapshot directory.
    pub path: PathBuf,
}

/// Top-level structure of the JSON registry file loaded via `--registry`.
#[allow(
    clippy::exhaustive_structs,
    reason = "wire DTO — single `models` field is the complete registry-file top-level contract; struct-literal construction by serde"
)]
#[derive(Deserialize, Debug)]
pub struct RegistryConfig {
    /// Ordered list of model snapshot entries from the registry JSON file.
    pub models: Vec<RegistryConfigEntry>,
}

impl RegistryConfig {
    /// Parse a JSON registry file from `path`.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("cannot read registry file {}: {e}", path.display()))?;
        let cfg: Self = serde_json::from_slice(&data)
            .map_err(|e| anyhow::anyhow!("malformed registry JSON {}: {e}", path.display()))?;
        Ok(cfg)
    }
}

// ── Types ────────────────────────────────────────────────────────────────────

/// One known model snapshot.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed registry entry — all fields are the complete per-model metadata contract; adding a field requires updating from_config and all ModelEntry construction sites"
)]
#[derive(Clone)]
pub struct ModelEntry {
    /// Basename of the snapshot directory — used as the model `id` in the API.
    pub id: String,
    /// Absolute filesystem path to the model snapshot directory.
    pub abs_path: PathBuf,
    /// Cached UTF-8 string of `abs_path`. Pre-computed once at registry build so
    /// per-request callers avoid `to_string_lossy()` allocation.
    pub abs_path_str: String,
    /// First architecture string from `config.json`.
    pub arch: String,
    /// Compiled Jinja chat template, if `chat_template.jinja` was found.
    pub chat_template: Option<Arc<ChatTemplate>>,
    /// Raw `chat_template.jinja` source text, retained for tool-call format
    /// detection (the same arch string can emit different tool conventions
    /// depending on the snapshot's template). `None` when the file is absent
    /// or unreadable. Cheap to keep — a few KB per model, registry-lifetime.
    pub chat_template_src: Option<Arc<str>>,
    /// Loaded tokenizer, if `tokenizer.json` was found.
    pub tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    /// BOS token string from `tokenizer_config.json`.
    pub bos_token: Option<String>,
    /// EOS token string from `tokenizer_config.json`.
    pub eos_token: Option<String>,
    /// Optional sampling defaults parsed from `generation_config.json`.
    /// `None` when the file is absent. Per-field `None` when the key is absent.
    pub generation_defaults: Option<Arc<GenerationConfig>>,
    /// True iff the compiled chat template can render a context that contains
    /// at least one tool without returning an error.
    ///
    /// Probed once at registry build by rendering a 1-dummy-tool context.
    /// When `false`, the route handler skips tool injection and proceeds
    /// tool-less (warns instead of 500-ing) — see A9 guard.
    ///
    /// `false` also when `chat_template` is `None` (no template = no tools).
    pub tools_supported: bool,
}

// Manual Serialize — skip the non-serialisable fields; they are internal only.
impl Serialize for ModelEntry {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("ModelEntry", 3)?;
        st.serialize_field("id", &self.id)?;
        st.serialize_field("abs_path", &self.abs_path)?;
        st.serialize_field("arch", &self.arch)?;
        st.end()
    }
}

impl std::fmt::Debug for ModelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelEntry")
            .field("id", &self.id)
            .field("arch", &self.arch)
            .field("has_template", &self.chat_template.is_some())
            .field("has_tokenizer", &self.tokenizer.is_some())
            .field(
                "has_generation_defaults",
                &self.generation_defaults.is_some(),
            )
            .field("tools_supported", &self.tools_supported)
            .finish()
    }
}

// ── Tools-support probe ───────────────────────────────────────────────────────

/// Probe whether `tpl` can render a context that includes one dummy tool.
///
/// Renders a minimal 1-message, 1-tool context. Returns `true` on success,
/// `false` on any error (template has no `{% if tools %}` branch, or the
/// template raises an exception when `tools` is passed).
///
/// This is the A9 runtime guard: called once at registry build, result cached
/// in `ModelEntry::tools_supported`. The render is intentionally cheap
/// (minimal message, minimal tool schema) and does NOT compare to HF output —
/// correctness is tested by the fixture round-trip suite.
fn probe_tools_supported(tpl: &ChatTemplate) -> bool {
    use crate::chat_template::{ChatMessageTpl, RenderOpts};

    let dummy_tool = serde_json::json!({
        "type": "function",
        "function": {
            "name": "probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}}
        }
    });
    let messages = [ChatMessageTpl {
        role: "user",
        content: "test",
        ..Default::default()
    }];
    let opts = RenderOpts {
        bos_token: "",
        eos_token: "",
        add_generation_prompt: false,
        tools: std::slice::from_ref(&dummy_tool),
        enable_thinking: None,
    };
    tpl.render(&messages, &opts).is_ok()
}

/// In-process model catalog.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed catalog struct — private BTreeMap field; public API is from_config(), get(), and iter(); adding a field requires updating from_config() and Default"
)]
#[derive(Debug, Default)]
pub struct ModelRegistry {
    entries: BTreeMap<String, ModelEntry>,
}

impl ModelRegistry {
    /// Build a registry from a [`RegistryConfig`] (JSON file loaded via
    /// `--registry`).
    ///
    /// Each entry's `id` field overrides the basename-derived default.
    pub fn from_config(cfg: &RegistryConfig) -> Self {
        let entries: Vec<(String, PathBuf)> = cfg
            .models
            .iter()
            .map(|e| {
                let id = e.id.clone().unwrap_or_else(|| {
                    e.path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_owned()
                });
                (id, e.path.clone())
            })
            .collect();
        Self::from_id_paths(&entries)
    }

    /// Build a registry by `load_config`-ing every supplied snapshot path.
    ///
    /// Missing / malformed `chat_template.jinja` or `tokenizer.json` are
    /// **best-effort**: a `tracing::warn!` is emitted but the entry is still
    /// added. This lets diagnostics work even for partial snapshots.
    ///
    /// Paths where `config.json` itself fails to load are **skipped** entirely.
    pub fn from_paths(paths: &[PathBuf]) -> Self {
        let pairs: Vec<(String, PathBuf)> = paths
            .iter()
            .map(|p| {
                let id = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_owned();
                (id, p.clone())
            })
            .collect();
        Self::from_id_paths(&pairs)
    }

    /// Internal: build from `(id, path)` pairs.
    fn from_id_paths(pairs: &[(String, PathBuf)]) -> Self {
        let mut reg = ModelRegistry::default();
        for (id, path) in pairs {
            let id = id.clone();
            let abs_path = path.canonicalize().unwrap_or_else(|_| path.clone());

            let arch = match load_config(&abs_path) {
                Ok(cfg) => cfg
                    .architectures
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| "unknown".to_owned()),
                Err(e) => {
                    warn!(path = %abs_path.display(), error = %e, "registry: skipping unreadable snapshot");
                    continue;
                }
            };

            // ── chat_template.jinja (best-effort) ───────────────────────────
            // Retain the raw source for tool-call format detection
            // (`detect_tool_call_format`) — the compiled `ChatTemplate` does
            // not keep its source.
            let mut chat_template_src: Option<Arc<str>> = None;
            let chat_template: Option<Arc<ChatTemplate>> = match chat_template::load_template_source(
                &abs_path,
            ) {
                Err(e) => {
                    warn!(
                        model_id = %id,
                        error = %e,
                        "registry: no chat_template.jinja (prompt pipeline disabled for this model)"
                    );
                    None
                }
                Ok(src) => {
                    chat_template_src = Some(Arc::from(src.as_str()));
                    match ChatTemplate::new(src) {
                        Ok(t) => Some(Arc::new(t)),
                        Err(e) => {
                            warn!(
                                model_id = %id,
                                error = %e,
                                "registry: failed to compile chat_template.jinja"
                            );
                            None
                        }
                    }
                }
            };

            // A9: probe whether tools can be injected into the template.
            let tools_supported = chat_template.as_deref().is_some_and(probe_tools_supported);
            if !tools_supported {
                tracing::debug!(
                    model_id = %id,
                    "registry: template does not support tool injection — tools will be disabled for this model"
                );
            }

            // ── tokenizer_config.json (best-effort) ─────────────────────────
            let (bos_token, eos_token) = match tokenizer_io::load_tokenizer_config(&abs_path) {
                Ok(cfg) => (cfg.bos_token, cfg.eos_token),
                Err(e) => {
                    warn!(
                        model_id = %id,
                        error = %e,
                        "registry: missing tokenizer_config.json; bos/eos tokens unknown"
                    );
                    (None, None)
                }
            };

            // ── tokenizer.json (best-effort) ────────────────────────────────
            let tokenizer: Option<Arc<tokenizers::Tokenizer>> =
                match tokenizer_io::load_tokenizer(&abs_path) {
                    Ok(tk) => Some(Arc::new(tk)),
                    Err(e) => {
                        warn!(
                            model_id = %id,
                            error = %e,
                            "registry: failed to load tokenizer.json (prompt pipeline disabled)"
                        );
                        None
                    }
                };

            // ── generation_config.json (best-effort) ────────────────────────
            let generation_defaults: Option<Arc<GenerationConfig>> =
                match generation_config_io::load_generation_config(&abs_path) {
                    Ok(Some(cfg)) => Some(Arc::new(cfg)),
                    Ok(None) => None,
                    Err(e) => {
                        warn!(
                            model_id = %id,
                            error = %e,
                            "registry: generation_config.json present but failed to parse; ignoring"
                        );
                        None
                    }
                };

            reg.entries.insert(
                id.clone(),
                ModelEntry {
                    id,
                    abs_path_str: abs_path.to_string_lossy().into_owned(),
                    abs_path,
                    arch,
                    chat_template,
                    chat_template_src,
                    tokenizer,
                    bos_token,
                    eos_token,
                    generation_defaults,
                    tools_supported,
                },
            );
        }
        reg
    }

    /// Look up a snapshot by its id (directory basename).
    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.get(id)
    }

    /// All registered snapshots, alphabetical by id.
    pub fn list(&self) -> Vec<&ModelEntry> {
        self.entries.values().collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
