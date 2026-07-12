//! Identity canonicalization per `docs/METRICS_DB.md` §5.
//!
//! Normalises the string fields that form the primary key of an `observations`
//! row (model, backend, kv_quant, metric) before any DB insert or query.
//! A consistent canonical form prevents duplicate rows from superficially
//! different string representations of the same identity.
//!
//! # Public API
//!
//! - [`RunIdentity`] — who produced a metrics row: backend, version, git sha,
//!   build profile, hardware tag. The ONE source for those fields in Rust.
//! - [`canonicalize_kv_quant`] — normalise a KV-quant string to its canonical
//!   lower-kebab form (e.g. `"K8V4"` → `"k8v4"`).
//! - [`canonicalize`] — general field canonicalization against a whitelist.
//! - [`split_model_path`] — split `"org/repo"` or an absolute path into
//!   `(org, repo)` for the `model_org` / `model_repo` DB columns.

use std::path::Path;
use std::sync::OnceLock;

use serde::Serialize;

use crate::error::{Error, Result};

/// Default hardware tag when `RMLX_HARDWARE_TAG` is unset.
const DEFAULT_HARDWARE_TAG: &str = "m5_max_128gb";

// ── Run identity ──────────────────────────────────────────────────────────────

/// The identity of the binary that produced a metrics row.
///
/// Rust emitters (the server drainer, `rmlx baseline`, `rmlx eval`) build one
/// of these instead of hand-rolling the five fields; shell emitters get the
/// same block as JSON from `rmlx metrics identity --json`. Both surfaces
/// therefore agree by construction, and the §8.5 ingest validator rejects any
/// `rmlx` record whose `backend_version` did not come from here.
///
/// See `docs/METRICS_DB.md` §8.5.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed identity block — these five fields are exactly the §8.5 run-identity contract; \
              adding one is a contract change that must update the ingest shape and every emitter"
)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunIdentity {
    /// Canonical backend name — always `"rmlx"` for a binary built from this repo.
    pub backend: String,
    /// Semver of this binary (`[workspace.package].version`).
    pub backend_version: String,
    /// Short git SHA of the checkout, `-dirty` suffixed when the tree is dirty.
    /// `None` for an installed binary with no checkout to interrogate.
    pub git_sha: Option<String>,
    /// Real Cargo profile name: `release`, `release-perf`, `release-debug`, `debug`.
    pub build_profile: String,
    /// Hardware the run happened on. `RMLX_HARDWARE_TAG` overrides the default.
    pub hardware_tag: String,
}

/// Computed once per process, cached in a `OnceLock`. None of the five fields
/// can change while the process runs. `backend_version` / `build_profile` /
/// `git_sha` are read from `build.rs`-stamped compile-time constants (no I/O);
/// only `hardware_tag` touches the environment, via a single `std::env::var`.
static IDENTITY: OnceLock<RunIdentity> = OnceLock::new();

impl RunIdentity {
    /// Identity of the currently-running rMLX binary — borrowed, no allocation
    /// after the first call. Prefer this over [`RunIdentity::rmlx`]; the owned
    /// form exists only for the one call site that must move the fields into
    /// an owned [`crate::ingest::RunRecord`].
    pub fn get() -> &'static RunIdentity {
        IDENTITY.get_or_init(|| Self {
            backend: "rmlx".to_string(),
            backend_version: rmlx_core::runinfo::backend_version().to_string(),
            git_sha: rmlx_core::runinfo::git_short_sha(),
            build_profile: rmlx_core::runinfo::build_profile().to_string(),
            hardware_tag: std::env::var("RMLX_HARDWARE_TAG")
                .unwrap_or_else(|_| DEFAULT_HARDWARE_TAG.to_string()),
        })
    }

    /// Owned clone of [`RunIdentity::get`]. Only for callers that must own a
    /// `RunIdentity` value (moving its `String` fields into another owned
    /// struct) — everything else should borrow via `get()`.
    pub fn rmlx() -> Self {
        Self::get().clone()
    }

    /// The `observations.inserted_by` audit string: `"<tool>@<semver>"`.
    ///
    /// e.g. `identity.inserted_by("rmlx-server")` → `"rmlx-server@0.2.8"`.
    pub fn inserted_by(&self, tool: &str) -> String {
        format!("{tool}@{}", self.backend_version)
    }

    /// Merge this identity into a §8.5 record that is still a `serde_json` object.
    ///
    /// For emitters that assemble the record with `serde_json::json!` rather
    /// than a `RunRecord` struct literal. Overwrites any identity key already
    /// present — the point is that the caller does not get to supply its own.
    pub fn stamp_json(&self, record: &mut serde_json::Value) -> Result<()> {
        let obj = record
            .as_object_mut()
            .ok_or_else(|| Error::InvalidIngestField {
                field: "<record>".to_string(),
                message: "expected a JSON object".to_string(),
            })?;
        obj.insert("backend".to_string(), self.backend.clone().into());
        obj.insert(
            "backend_version".to_string(),
            self.backend_version.clone().into(),
        );
        obj.insert(
            "git_sha".to_string(),
            self.git_sha
                .clone()
                .map_or(serde_json::Value::Null, Into::into),
        );
        obj.insert(
            "build_profile".to_string(),
            self.build_profile.clone().into(),
        );
        obj.insert("hardware_tag".to_string(), self.hardware_tag.clone().into());
        Ok(())
    }
}

// ── Model-id canonicalization ─────────────────────────────────────────────────

/// Split a snapshot id or path into `(model_namespace, model)`.
///
/// Accepts `"<ns>__<model>"` (the on-disk snapshot layout) and bare names.
/// Falls back to the `"local"` namespace — always whitelisted — rather than
/// failing, because a metrics row with an approximate namespace beats no row.
///
/// Lenient by design; [`split_model_path`] is the strict, erroring variant used
/// where the caller can act on a bad path.
pub fn split_model_id(model_id: &str) -> (String, String) {
    // Tolerate a full path: only the final component carries the id.
    let basename = Path::new(model_id)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_id);

    basename.find("__").map_or_else(
        || ("local".to_string(), basename.to_string()),
        |idx| {
            let (ns, rest) = basename.split_at(idx);
            (ns.to_string(), rest.trim_start_matches('_').to_string())
        },
    )
}

/// Infer the on-disk weight quantization from a snapshot name.
///
/// Order matters — the more specific token wins (`mxfp8` before `8bit`).
/// Falls back to `bf16`, which is the whitelist's "unquantized / we don't
/// know" value.
pub fn infer_weight_quant(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    for token in [
        "mxfp8", "mxfp4", "nvfp4", "q4_k_m", "q8_0", "8bit", "4bit", "2bit", "3bit", "5bit",
        "6bit", "fp16", "bf16", "paro",
    ] {
        if lower.contains(token) {
            return token.to_string();
        }
    }
    "bf16".to_string()
}

/// True when `v` is a semver `MAJOR.MINOR.PATCH`, with an optional
/// `-prerelease` / `+build` suffix.
///
/// Deliberately hand-rolled rather than pulling in the `semver` crate: the only
/// question asked is "did this come from a real Cargo version, or is it a git
/// sha / a `head` / a fabricated literal?". A numeric three-part core answers
/// that. The suffix is tolerated so an `0.3.0-rc.1` build does not have its
/// metrics silently rejected.
pub fn is_semver(v: &str) -> bool {
    // "1.2.3-rc.1+meta" → "1.2.3"
    let core = v.split(['-', '+']).next().unwrap_or(v);
    let mut parts = core.split('.');
    let numeric =
        |p: Option<&str>| p.is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
    numeric(parts.next())
        && numeric(parts.next())
        && numeric(parts.next())
        && parts.next().is_none()
}

/// Allowed values for the `backend` identity column (see docs/METRICS_DB.md §5.4).
pub const BACKEND_WHITELIST: &[&str] = &[
    "rmlx",
    "mlx_lm",
    "mlx_lm_tq",
    "paroquant",
    "omlx",
    "ollama",
    "vllm",
    "llama_cpp",
];

/// Allowed values for the `model_namespace` identity column (see docs/METRICS_DB.md §5.1).
pub const NAMESPACE_WHITELIST: &[&str] = &[
    "mlx-community",
    "z-lab",
    "prism-ml",
    "paramind",
    "paro-team",
    "ollama",
    "hf",
    "local",
];

/// Allowed values for the `weight_quant` identity column (see docs/METRICS_DB.md §5.2).
pub const WEIGHT_QUANT_WHITELIST: &[&str] = &[
    "bf16", "fp16", "mxfp8", "mxfp4", "nvfp4", "q8_0", "q4_k_m", "2bit", "3bit", "4bit", "5bit",
    "6bit", "8bit", "paro",
];

/// Validate (and canonicalize) a `kv_quant` column value.
///
/// Replaces the fixed `KV_QUANT_WHITELIST` lookup. Accepts:
///
/// * the canonical Display forms emitted by `<KvQuant as Display>` —
///   `"none"`, `"k8v4"`, `"k8v8"`, `"planar"`, and the long-form
///   `"mixed_k<kb>g<kg>_v<vb>g<vg>"` (e.g. `"mixed_k8g128_v4g64"`).
/// * the aliases `"bf16"` and `"f16"`, which canonicalize to `"none"`.
/// * the legacy historical labels `"turbo4"` and `"turbo8"` (preserved
///   verbatim for backward-compat with legacy-ingested rows).
///
/// Lowercases input before checking. Returns the canonical form on success.
pub fn canonicalize_kv_quant(value: &str) -> Result<String> {
    let lower = value.to_lowercase();
    if matches!(lower.as_str(), "turbo4" | "turbo8") {
        return Ok(lower);
    }
    if is_valid_kv_quant_token(&lower) {
        // Re-emit through the same matcher to canonicalize aliases (bf16/f16 → none).
        return Ok(canonicalize_kv_quant_token(&lower));
    }
    Err(Error::IdentityNotInWhitelist {
        field: "kv_quant".to_string(),
        value: value.to_string(),
        allowed: vec![
            "none".to_string(),
            "k8v4".to_string(),
            "k8v8".to_string(),
            "planar".to_string(),
            "mixed_k<kb>g<kg>_v<vb>g<vg>".to_string(),
            "turbo4".to_string(),
            "turbo8".to_string(),
        ],
    })
}

/// Pure-string validator for a kv_quant token (lower-case). Mirrors the
/// grammar in `<KvQuant as FromStr>` without importing `rmlx-models` (which
/// would pull the MLX-bound types into the metrics crate). Update both in
/// lockstep when the canonical grammar changes.
fn is_valid_kv_quant_token(lower: &str) -> bool {
    match lower {
        "none" | "bf16" | "f16" | "k8v4" | "k8v8" | "planar" => true,
        s if s.starts_with("mixed_") => parse_mixed_token(s).is_some(),
        _ => false,
    }
}

/// Canonicalize a validated lower-case token to its Display form
/// (`bf16`/`f16` → `none`; everything else passes through).
fn canonicalize_kv_quant_token(lower: &str) -> String {
    match lower {
        "bf16" | "f16" => "none".to_string(),
        other => other.to_string(),
    }
}

/// Returns `Some(())` iff `s` matches `mixed_k<u8>g<u16>_v<u8>g<u16>`.
fn parse_mixed_token(s: &str) -> Option<()> {
    let rest = s.strip_prefix("mixed_")?;
    let (k_part, v_part) = rest.split_once('_')?;
    let (kb, kg) = parse_side(k_part, 'k')?;
    let (vb, vg) = parse_side(v_part, 'v')?;
    // Smoke-check ranges (real `KvQuant::Mixed` uses `u8`/`u16`).
    let _ = (kb, kg, vb, vg);
    Some(())
}

fn parse_side(spec: &str, prefix: char) -> Option<(u8, u16)> {
    let rest = spec.strip_prefix(prefix)?;
    let (bits, group) = rest.split_once('g')?;
    let bits: u8 = bits.parse().ok()?;
    let group: u16 = group.parse().ok()?;
    Some((bits, group))
}

/// Normalize well-known backend aliases to their canonical form before
/// whitelist lookup. Only called by `canonicalize` when `field == "backend"`.
///
/// Handles common variant spellings that bench tools emit:
/// - `llama.cpp`, `llama-cpp`, `llamacpp` → `llama_cpp`
fn normalize_backend_alias(lower: &str) -> &str {
    match lower {
        "llama.cpp" | "llama-cpp" | "llamacpp" => "llama_cpp",
        other => other,
    }
}

/// Canonicalize a value against a whitelist. Lowercases the input before
/// matching. For `field == "backend"`, also applies alias normalization before
/// the whitelist check. Returns `Error::IdentityNotInWhitelist` with the field
/// name on a miss.
pub fn canonicalize(field: &str, value: &str, whitelist: &[&str]) -> Result<String> {
    let lower = value.to_lowercase();
    let normalized = if field == "backend" {
        normalize_backend_alias(&lower)
    } else {
        lower.as_str()
    };
    if whitelist.contains(&normalized) {
        Ok(normalized.to_string())
    } else {
        Err(Error::IdentityNotInWhitelist {
            field: field.to_string(),
            value: value.to_string(),
            allowed: whitelist.iter().map(ToString::to_string).collect(),
        })
    }
}

/// Split a filesystem path / HF id / ollama tag into `(namespace, model)`.
///
/// Rule set (docs/METRICS_DB.md §5.1):
///
/// 1. **Filesystem path** — starts with `/`, `~`, or is absolute per
///    [`Path::is_absolute`]. Strip trailing `/`, take the last path segment
///    (basename).
///    - If the basename contains `__`, split on the *first* `__` →
///      `(ns, model)`. Validate `ns` against [`NAMESPACE_WHITELIST`]. Return
///      `Error::IdentityNotInWhitelist` if unknown.
///    - Otherwise → `("local", basename)`.
///
/// 2. **Ollama tag** — no `/`, contains `:` → `("ollama", input)`.
///
/// 3. **HF id** — does NOT start with `/`, contains exactly one `/` →
///    `("hf", input)`. `meta-llama/A/B` (two `/`) falls through to error.
///
/// 4. Anything else → `Error::IdentityModelPath`.
pub fn split_model_path(input: &str) -> Result<(String, String)> {
    // Normalise: trim surrounding whitespace and a single trailing newline.
    let input = input.trim_end_matches('\n').trim();

    let is_filesystem =
        input.starts_with('/') || input.starts_with('~') || Path::new(input).is_absolute();

    if is_filesystem {
        // Strip trailing slash(es) then take the last non-empty segment.
        let trimmed = input.trim_end_matches('/');
        let basename = Path::new(trimmed)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                Error::IdentityModelPath(format!("cannot extract basename from: {input}"))
            })?;

        if let Some(pos) = basename.find("__") {
            let ns = &basename[..pos];
            let model = &basename[pos + 2..];
            // Validate namespace against whitelist.
            if NAMESPACE_WHITELIST.contains(&ns) {
                Ok((ns.to_string(), model.to_string()))
            } else {
                Err(Error::IdentityNotInWhitelist {
                    field: "model_namespace".to_string(),
                    value: ns.to_string(),
                    allowed: NAMESPACE_WHITELIST
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                })
            }
        } else {
            Ok(("local".to_string(), basename.to_string()))
        }
    } else if input.contains(':') && !input.contains('/') {
        // Ollama tag: no slash, has colon.
        Ok(("ollama".to_string(), input.to_string()))
    } else if input.contains('/') {
        // HF id: must contain exactly one slash, must not start with '/'.
        let slash_count = input.chars().filter(|&c| c == '/').count();
        if slash_count == 1 {
            Ok(("hf".to_string(), input.to_string()))
        } else {
            Err(Error::IdentityModelPath(format!(
                "unrecognized model path format (too many slashes): {input}"
            )))
        }
    } else {
        Err(Error::IdentityModelPath(format!(
            "unrecognized model path format: {input}"
        )))
    }
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
