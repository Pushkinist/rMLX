//! Identity canonicalization per `docs/METRICS_DB.md` §5.
//!
//! Normalises the string fields that form the primary key of an `observations`
//! row (model, backend, kv_quant, metric) before any DB insert or query.
//! A consistent canonical form prevents duplicate rows from superficially
//! different string representations of the same identity.
//!
//! # Public API
//!
//! - [`canonicalize_kv_quant`] — normalise a KV-quant string to its canonical
//!   lower-kebab form (e.g. `"K8V4"` → `"k8v4"`).
//! - [`canonicalize`] — general field canonicalization against a whitelist.
//! - [`split_model_path`] — split `"org/repo"` or an absolute path into
//!   `(org, repo)` for the `model_org` / `model_repo` DB columns.

use std::path::Path;

use crate::error::{Error, Result};

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
