//! Sibling-tensor resolver for MLX-quantized safetensors layouts.
//!
//! For each quantized linear weight `<base>`, MLX stores up to three siblings:
//! `<base>.weight` — packed quantized integers (required for quantized tensors)
//! `<base>.scales` — per-group scale factor
//! `<base>.biases` — per-group zero-point/bias (affine only; absent for mxfp*/nvfp4)
//!
//! A tensor is quantized iff `<base>.scales` is present in the weight_map.
//! (Canonical rule from `docs/03-mlx-safetensors-format.md`.)
//!
//! ## ParoQuant extension
//!
//! `z-lab/*-PARO` checkpoints store an additional set of rotation siblings alongside
//! the standard INT4 weight/scales/zeros:
//!
//! `<base>.qweight` — I32 INT4-packed weight matrix
//! `<base>.scales` — F16 per-group scales
//! `<base>.qzeros` — I32 per-group zero-points
//! `<base>.pairs` — I16, shape [krot, in_features]
//! `<base>.theta` — F16, shape [krot, in_features/2] (rotation angles)
//! `<base>.channel_scales` — F16, shape [1, in_features]
//!
//! Detection: `<base>.pairs` is present iff the layer has ParoQuant rotation data.
//! Use `resolve_paro()` to collect these into a `ParoQuantState`.

use std::collections::BTreeMap;
use std::collections::HashMap;

use tracing::debug;

use rmlx_core::{Error, Result};

use crate::shards::{ShardHandle, ShardIndex, ShardSet};

// ── ParoQuantParams / ParoQuantState ─────────────────────────────────────────

/// Per-layer ParoQuant rotation parameters, parsed from the safetensors index.
///
/// Contains only shard filenames — tensor bytes are not loaded at this stage.
/// Pass to `view()` to get the actual bytes when needed.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed loader struct — fields are the complete ParoQuant rotation-parameter contract; adding a field requires updating resolve_paro and all PARO layer constructors"
)]
#[derive(Debug, Clone)]
pub struct ParoQuantParams {
    /// Number of rotation groups for this layer (krot dimension of `pairs` / `theta`).
    pub krot: u32,
    /// Shard holding `<base>.pairs` (I16, shape [krot, in_features]).
    pub pairs_shard: String,
    /// Shard holding `<base>.theta` (F16, shape [krot, in_features/2]).
    pub theta_shard: String,
    /// Shard holding `<base>.channel_scales` (F16, shape [1, in_features]).
    pub channel_scales_shard: String,
    /// Shard holding `<base>.qweight` (I32, INT4 packed).
    pub qweight_shard: String,
    /// Shard holding `<base>.qzeros` (I32, INT4 zero-points).
    pub qzeros_shard: String,
}

/// Model-level ParoQuant state: per-layer params keyed by base tensor name.
///
/// Accessible from the loader API so that Stage-2 graph integration (#103)
/// can wire rotation params into the forward pass without re-parsing.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed loader struct — fields are the complete model-level PARO state contract; adding a field requires updating resolve_paro and all PARO consumers"
)]
#[derive(Debug, Clone, Default)]
pub struct ParoQuantState {
    /// Map from base tensor name (e.g. `"model.language_model.layers.0.mlp.down_proj"`)
    /// to the rotation params for that layer.
    pub layers: HashMap<String, ParoQuantParams>,
}

impl ParoQuantState {
    /// Total count of layers with ParoQuant rotation data.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Maximum `krot` value across all layers (0 if no layers).
    pub fn krot_max(&self) -> u32 {
        self.layers.values().map(|p| p.krot).max().unwrap_or(0)
    }

    /// Total bytes consumed by rotation tensors (`pairs` + `theta` + `channel_scales`)
    /// as recorded in the shard index (requires the index shapes for a real count;
    /// this counts distinct shards as a proxy when shapes are not available).
    ///
    /// Note: This counts logical tensor slots, not actual bytes. For real byte
    /// accounting, call `view()` on each tensor and sum `bytes.len()`.
    pub fn rotation_tensor_slots(&self) -> usize {
        // 3 tensors per layer: pairs, theta, channel_scales
        self.layers.len() * 3
    }
}

// ── TensorKind ───────────────────────────────────────────────────────────────

/// Classification of a resolved tensor group.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — variants map to MLX quant formats; adding a variant requires synchronized changes to resolve(), resolve_paro(), and all tensor-loading dispatch sites"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorKind {
    /// Unquantized (e.g. RMSNorm weights, embedding, norm tensors).
    Plain,
    /// MLX affine quantized: weight + scales + biases siblings.
    Affine,
    /// OCP microscaling (mxfp4/mxfp8): weight + scales (no biases).
    /// Callers may upgrade to `Nvfp4` based on `ModelConfig.quantization.mode == "nvfp4"`.
    Mxfp,
    /// nvfp4: weight + scales (no biases). Set by the CLI when config mode == "nvfp4".
    Nvfp4,
    /// ParoQuant INT4: qweight + scales + qzeros + pairs + theta + channel_scales.
    /// The weight key is `<base>.qweight` (not `.weight`).
    ParoQuant,
    /// Heuristic uncertain — caller decides.
    Unknown,
}

// ── ResolvedTensor ───────────────────────────────────────────────────────────

/// A resolved tensor group: base name + kind + which shard(s) hold each sibling.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed loader struct — fields are the complete resolved-tensor contract; adding a field requires updating resolve() and all tensor consumers"
)]
#[derive(Debug, Clone)]
pub struct ResolvedTensor {
    /// Base name (e.g. `"language_model.model.layers.0.mlp.down_proj"`).
    /// For `.weight`-suffixed raw entries the `.weight` suffix is stripped.
    /// For plain entries this is the raw key.
    pub base_name: String,
    /// Classifies this entry as quantized vs plain (drives sibling-shard resolution).
    pub kind: TensorKind,
    /// Shard filename for the `.weight` (or sole plain) tensor.
    pub weight_shard: String,
    /// Shard filename for the `.scales` sibling, if present.
    pub scales_shard: Option<String>,
    /// Shard filename for the `.biases` sibling, if present.
    pub biases_shard: Option<String>,
}

// ── resolve ──────────────────────────────────────────────────────────────────

/// Walk `idx.weight_map` and group entries into resolved tensors.
///
/// Algorithm:
/// 1. Skip any entry whose name ends with `.scales`, `.biases`, `.qzeros`,
///    `.pairs`, `.theta`, `.channel_scales`, or `.qweight` — collected when
///    the primary key for that group is processed.
/// 2. For each remaining entry `key`:
///    - If `key` ends with `.weight`, strip it to get `base`; the shard for
///      the weight component is `weight_map[key]`.
///    - Otherwise `base = key` and the weight shard is `weight_map[key]`.
/// 3. Check for PARO rotation siblings: if `base + ".pairs"` is in the map,
///    the layer is `ParoQuant` (uses `.qweight` not `.weight`).
/// 4. Otherwise look up `.scales` and `.biases`:
///    - Both present -> `Affine`
///    - Only `.scales` -> `Mxfp` (caller upgrades to `Nvfp4` if config says so)
///    - Neither -> `Plain`
/// 5. An orphaned `.scales` key (no corresponding base) produces `Err`.
///
/// Output is sorted by `base_name` for deterministic iteration.
pub fn resolve(idx: &ShardIndex) -> Result<Vec<ResolvedTensor>> {
    // Pure-sibling suffixes: never a primary key; always collected via their base.
    // NOTE: `.qweight` is intentionally absent here — PARO layers have `.qweight`
    // as their primary weight entry, with `.pairs` as the PARO detection marker.
    const SIBLING_SUFFIXES: &[&str] = &[
        ".scales",
        ".biases",
        // ParoQuant rotation siblings (NOT .qweight — it is the primary weight key):
        ".qzeros",
        ".pairs",
        ".theta",
        ".channel_scales",
    ];

    let wm = &idx.weight_map;
    let is_sibling = |key: &str| SIBLING_SUFFIXES.iter().any(|s| key.ends_with(s));

    // Collect all base names from primary keys.
    // Primary keys: `.weight` (strip suffix), `.qweight` (strip suffix, PARO), plain.
    let mut seen_bases: BTreeMap<String, String> = BTreeMap::new(); // base -> weight_shard

    for (key, shard) in wm {
        if is_sibling(key) {
            continue;
        }
        let base = if key.ends_with(".weight") {
            key[..key.len() - ".weight".len()].to_owned()
        } else if key.ends_with(".qweight") {
            // PARO primary key: strip `.qweight` to get base.
            key[..key.len() - ".qweight".len()].to_owned()
        } else {
            key.clone()
        };
        // Duplicate base names (shouldn't happen in a well-formed snapshot, but guard).
        seen_bases.entry(base).or_insert_with(|| shard.clone());
    }

    // Validate: every `.scales` and `.biases` key must have a known base.
    // PARO bases (those with `.pairs`) are exempt from the `.scales` orphan check
    // because their `.scales` is a valid PARO sibling — it IS expected to be present.
    for key in wm.keys() {
        if key.ends_with(".scales") {
            let base = &key[..key.len() - ".scales".len()];
            if !seen_bases.contains_key(base) {
                return Err(Error::Loader(format!(
                    "orphaned .scales entry '{key}' has no corresponding .weight or plain tensor"
                )));
            }
        }
        if key.ends_with(".biases") {
            let base = &key[..key.len() - ".biases".len()];
            // PARO layers may not have .biases; only validate if base is known.
            if seen_bases.contains_key(base) {
                // biases present — will be validated in the build loop below.
            } else {
                return Err(Error::Loader(format!(
                    "orphaned .biases entry '{key}' has no corresponding .weight or plain tensor"
                )));
            }
        }
    }

    // Build resolved tensors.
    let mut resolved: Vec<ResolvedTensor> = Vec::with_capacity(seen_bases.len());

    for (base, weight_shard) in seen_bases {
        // PARO detection: `.pairs` sibling present means this is a ParoQuant layer.
        // ParoQuant layers use `.qweight` as the packed-weight key.
        let is_paro = wm.contains_key(&format!("{base}.pairs"));

        let (kind, scales_shard, biases_shard) = if is_paro {
            // PARO layers: classification is ParoQuant; scales/biases not surfaced
            // here (they are accessed via resolve_paro() instead).
            (TensorKind::ParoQuant, None, None)
        } else {
            let scales_shard = wm.get(&format!("{base}.scales")).cloned();
            let biases_shard = wm.get(&format!("{base}.biases")).cloned();
            let kind = match (&scales_shard, &biases_shard) {
                (Some(_), Some(_)) => TensorKind::Affine,
                (Some(_), None) => TensorKind::Mxfp,
                (None, None) => TensorKind::Plain,
                (None, Some(_)) => {
                    return Err(Error::Loader(format!(
                        "tensor '{base}' has .biases but no .scales — malformed snapshot"
                    )));
                }
            };
            (kind, scales_shard, biases_shard)
        };

        resolved.push(ResolvedTensor {
            base_name: base,
            kind,
            weight_shard,
            scales_shard,
            biases_shard,
        });
    }

    // Already sorted by BTreeMap insertion order (base_name).
    debug!(
        plain = resolved
            .iter()
            .filter(|t| t.kind == TensorKind::Plain)
            .count(),
        affine = resolved
            .iter()
            .filter(|t| t.kind == TensorKind::Affine)
            .count(),
        mxfp = resolved
            .iter()
            .filter(|t| t.kind == TensorKind::Mxfp)
            .count(),
        paroquant = resolved
            .iter()
            .filter(|t| t.kind == TensorKind::ParoQuant)
            .count(),
        "resolve complete"
    );

    Ok(resolved)
}

// ── resolve_paro ──────────────────────────────────────────────────────────────

/// Walk `idx.weight_map` and collect ParoQuant rotation parameters for each
/// layer that has a `.pairs` sibling.
///
/// Returns a `ParoQuantState` with one entry per PARO linear layer.
/// When no `.pairs` tensors exist, returns an empty state (caller can check
/// `state.layer_count() == 0`).
///
/// Extracts `krot` from the first dimension of the `.pairs` shape recorded in
/// the shard's safetensors header. Since shape is not available from the shard
/// index alone (only tensor→shard mappings), `krot` is passed in as a hint
/// from `ModelConfig.quantization_config.krot` when available.
///
/// If `krot_hint` is `None`, the function opens the first shard that contains a
/// `.pairs` tensor and reads its header to extract `krot` from the shape.
/// Pass `Some(krot)` from `ModelConfig.quantization_config` to avoid that I/O.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "tensor key suffixes like '.pairs' are internal MLX format conventions, not filesystem extensions; case-folding is irrelevant and would obscure intent"
)]
pub fn resolve_paro(
    idx: &ShardIndex,
    model_dir: &std::path::Path,
    krot_hint: Option<u32>,
) -> Result<ParoQuantState> {
    let wm = &idx.weight_map;
    let mut state = ParoQuantState::default();

    // Collect all PARO bases (those with a `.pairs` sibling).
    let paro_bases: Vec<String> = {
        let mut bases: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for key in wm.keys() {
            if key.ends_with(".pairs") {
                let base = key[..key.len() - ".pairs".len()].to_owned();
                bases.insert(base);
            }
        }
        bases.into_iter().collect()
    };

    if paro_bases.is_empty() {
        return Ok(state);
    }

    // Resolve krot: use hint if provided, else read first .pairs shard header.
    let krot = if let Some(k) = krot_hint {
        k
    } else {
        // Invariant: paro_bases is non-empty — the is_empty() guard above returns early if empty.
        let first_base = paro_bases
            .first()
            // Logically unreachable: is_empty() guard at line ~332 short-circuits the empty case.
            // Kept defensively so a future refactor that removes the guard still returns a typed error.
            .ok_or_else(|| Error::Loader("paro_bases unexpectedly empty".to_owned()))?;
        let pairs_key = format!("{first_base}.pairs");
        let pairs_shard_file = wm.get(&pairs_key).ok_or_else(|| {
            Error::Loader(format!("missing .pairs shard entry for '{first_base}'"))
        })?;
        let handle = ShardHandle::open(model_dir, pairs_shard_file)?;
        let st = handle.safetensors()?;
        let t = st
            .tensor(&pairs_key)
            .map_err(|e| Error::Loader(format!("cannot read tensor '{pairs_key}': {e}")))?;
        let shape = t.shape();
        // shape.first() returns None for empty shape; the ok_or_else propagates that as a typed Loader error.
        let krot_from_shape = shape.first().copied().ok_or_else(|| {
            Error::Loader(format!("'{pairs_key}' has empty shape — cannot infer krot"))
        })? as u32;
        krot_from_shape
    };

    for base in paro_bases {
        let get_shard = |suffix: &str| -> Result<String> {
            let key = format!("{base}{suffix}");
            wm.get(&key)
                .cloned()
                .ok_or_else(|| Error::Loader(format!("missing '{key}' in shard index")))
        };

        let params = ParoQuantParams {
            krot,
            pairs_shard: get_shard(".pairs")?,
            theta_shard: get_shard(".theta")?,
            channel_scales_shard: get_shard(".channel_scales")?,
            qweight_shard: get_shard(".qweight")?,
            qzeros_shard: get_shard(".qzeros")?,
        };

        state.layers.insert(base, params);
    }

    debug!(
        paro_layers = state.layer_count(),
        krot, "resolve_paro complete"
    );

    Ok(state)
}

// ── TensorView ───────────────────────────────────────────────────────────────

/// A zero-copy view into one tensor's bytes within a memory-mapped shard.
///
/// `bytes` borrows directly from the memory-mapped shard (lifetime `'a`).
/// `shape` is a small owned `Vec<usize>` copied from the safetensors header.
/// Copying shape is negligible (a few elements per tensor).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed zero-copy view — fields are the complete tensor-view contract; adding a field requires updating view() and all shard consumers"
)]
pub struct TensorView<'a> {
    /// Tensor key in the shard (matches the safetensors header).
    pub name: &'a str,
    /// Element dtype declared in the safetensors header.
    pub dtype: safetensors::Dtype,
    /// Owned copy of the tensor shape (copied from safetensors header Vec).
    pub shape: Vec<usize>,
    /// Zero-copy byte slice backed by the shard's mmap.
    pub bytes: &'a [u8],
}

impl std::fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("bytes_len", &self.bytes.len())
            .finish()
    }
}

// ── try_exact_then_suffix ────────────────────────────────────────────────────

/// Resolve a tensor name in `idx.weight_map` using a two-phase strategy.
///
/// **Phase 1 — exact match:** if `name` is a key in the weight_map, return it
/// unchanged. This is the fast path and covers every checkpoint whose tensor
/// names are already known (the common case).
///
/// **Phase 2 — suffix match:** if the exact lookup misses, iterate
/// `weight_map` and return the first key whose final `suffix_segments`
/// dot-delimited components match the tail of `name`. This handles checkpoints
/// where the model author chose a different prefix namespace
/// (e.g., `model.layers.N.*` vs `transformer.h.N.*` vs bare `layers.N.*`).
///
/// `suffix_segments` controls how many trailing dot-components must match.
/// Recommended value: the number of components that uniquely identify the
/// tensor within a layer (typically 3–5; e.g. `"layers.0.self_attn.q_proj.weight"` →
/// suffix_segments=5 matches that exact sub-path regardless of any leading prefix).
///
/// **Degenerate inputs:**
/// - `suffix_segments == 0`: Phase 2 is disabled (returns `None` on a Phase 1
///   miss). Phase 1 can still succeed if `name` is an exact key.
/// - `name` has fewer dot-segments than `suffix_segments`: Phase 2 returns `None`.
///
/// **Disambiguation:** `weight_map` is a `BTreeMap` (lexicographic order).
/// When multiple keys share the same suffix, the first in lexicographic order
/// is returned. Choose `suffix_segments` large enough to be unique across the
/// checkpoint.
///
/// Returns the resolved (possibly rewritten) name if found, or `None`.
/// The returned `&str` borrows from the `ShardIndex`'s `weight_map` keys.
///
/// # Performance
/// Phase 1 is O(log n_tensors). Phase 2 is O(n_tensors × suffix_len) —
/// called only on a cache miss, which is rare (once per layer type per model load).
pub fn try_exact_then_suffix<'a>(
    idx: &'a ShardIndex,
    name: &str,
    suffix_segments: usize,
) -> Option<&'a str> {
    // Phase 1: exact match (BTreeMap O(log n)).
    if let Some((k, _)) = idx.weight_map.get_key_value(name) {
        return Some(k.as_str());
    }

    // Phase 2: suffix match on the last `suffix_segments` dot-components.
    // Compute the suffix as a byte-slice into `name` — no allocation.
    if suffix_segments == 0 {
        return None;
    }
    // Walk backwards counting dots; we need (suffix_segments - 1) dot
    // separators to isolate `suffix_segments` components.
    let mut dots_seen = 0usize;
    let mut suffix_start = 0usize; // default: entire string is the suffix
    for (i, b) in name.bytes().enumerate().rev() {
        if b == b'.' {
            dots_seen += 1;
            if dots_seen == suffix_segments {
                // The suffix starts immediately after this dot.
                suffix_start = i + 1;
                break;
            }
        }
    }
    // If fewer segments exist in `name` than requested, Phase 2 cannot match.
    if dots_seen < suffix_segments - 1 {
        return None;
    }
    let target_suffix: &str = &name[suffix_start..];

    for key in idx.weight_map.keys() {
        // Fast rejection: the key must be at least as long as our suffix.
        if key.len() < target_suffix.len() {
            continue;
        }
        // Check for suffix match at a dot boundary (or start of string).
        if !key.ends_with(target_suffix) {
            continue;
        }
        // Verify the match is at a component boundary: the character
        // immediately before the suffix must be '.' (or the suffix covers
        // the entire key).
        let prefix_len = key.len() - target_suffix.len();
        // prefix_len <= key.len() is guaranteed: key.len() >= target_suffix.len()
        // is enforced by the key.len() < target_suffix.len() { continue } guard above.
        // When prefix_len == 0 the suffix covers the whole key → boundary holds.
        let at_boundary = prefix_len == 0
            || key.as_bytes().get(..prefix_len).and_then(|s| s.last()) == Some(&b'.');
        if at_boundary {
            debug!(
                requested = name,
                resolved = key.as_str(),
                suffix = target_suffix,
                "tensor suffix-match fallback"
            );
            return Some(key.as_str());
        }
    }

    None
}

/// Look up tensor `name` across all shards and return a zero-copy view.
///
/// Locates the owning shard via `idx.weight_map`, then parses the shard's
/// safetensors header (O(KB)) and returns the tensor's byte slice — no full
/// shard data is loaded into user memory beyond the page-faulted mmap region
/// actually touched.
///
/// `bytes` has lifetime `'a` (borrowed from the shard mmap).
/// `shape` is a small owned copy (Vec<usize>) from the parsed header.
pub fn view<'a>(shards: &'a ShardSet, idx: &ShardIndex, name: &'a str) -> Result<TensorView<'a>> {
    let shard_filename = idx
        .weight_map
        .get(name)
        .ok_or_else(|| Error::Loader(format!("tensor '{name}' not found in shard index")))?;

    let handle: &'a ShardHandle = shards.get(shard_filename).ok_or_else(|| {
        Error::Loader(format!(
            "shard '{shard_filename}' not open (referenced by tensor '{name}')"
        ))
    })?;

    // Parse safetensors header — O(header size), not O(shard size).
    // The deserialize-from-bytes lifetime threads through cleanly: bytes are
    // `&'a [u8]` (borrowed from the mmap), so SafeTensors<'a>, TensorView<'a>,
    // and `t.data() -> &'a [u8]` all carry the same lifetime.
    let st = safetensors::SafeTensors::deserialize(handle.as_bytes()).map_err(|e| {
        Error::Loader(format!(
            "cannot parse safetensors header in {shard_filename}: {e}"
        ))
    })?;

    let t = st.tensor(name).map_err(|e| {
        Error::Loader(format!(
            "tensor '{name}' not found in shard '{shard_filename}': {e}"
        ))
    })?;

    Ok(TensorView {
        name,
        dtype: t.dtype(),
        shape: t.shape().to_vec(),
        bytes: t.data(),
    })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tensors_tests.rs"]
mod tests;
