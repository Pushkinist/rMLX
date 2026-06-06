//! Named KV-cache preset table — `--kv-preset <name>` resolution.
//!
//! Each preset is a `PresetSpec` that bundles a `KvQuant`, an optional sparse-
//! attention spec, and a `requires_calibration` flag.
//!
//! # Adding a new preset
//!
//! Append a row to `PRESETS` in `preset_table()` and a test row in
//! `preset_table_tests.rs`. Future presets extend the table;
//! the resolution path in `parse.rs` is unchanged.
//!
//! # Auto-selector
//!
//! `--kv-preset auto` is handled by [`recommend_preset`] which takes model
//! size, context length, and available unified DRAM and returns the best
//! available preset name.  `lookup_preset("auto")` still returns
//! `Err(PresetError::Reserved)` — `parse_kv_preset` in `parse.rs` intercepts
//! `"auto"` before `lookup_preset` is called.

use rmlx_kv_quant::KvQuant;

/// Sparse-attention spec.
///
/// This may be relocated to `rmlx_kv_quant::sparse_attn`; callers should
/// use `Option<SparseAttnSpec>` from this module until such a relocation lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseAttnSpec;

/// A resolved KV-cache preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresetSpec {
    /// The `KvQuant` variant this preset resolves to.
    ///
    /// `KvQuant::None` means bf16/unquantised on **both** sides — this is the
    /// `KvQuant` variant named `None`, NOT an `Option::None` absent value.
    pub(crate) kv_quant: KvQuant,
    /// Sparse-attention override. `None` = no sparse attn (default for all
    /// starter presets). Populated for eligible presets when sparse-attn is wired.
    pub(crate) sparse_attn: Option<SparseAttnSpec>,
    /// Whether this preset requires per-dataset calibration data to function
    /// correctly (e.g. rotation-KV families). Starter presets are all `false`.
    pub(crate) requires_calibration: bool,
}

/// Error returned by [`lookup_preset`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PresetError {
    /// The name is the reserved `auto` placeholder.
    Reserved,
    /// The name is not in the preset table.
    Unknown,
}

/// Static preset table: name → `PresetSpec`.
///
/// Stored as a slice of `(&str, PresetSpec)` pairs.  We use a linear
/// scan (5 entries) rather than `phf` — the table is tiny and `phf` is not
/// currently in the workspace.  When the table grows past ~20 entries, switch
/// to a `phf::Map`.
///
/// ## Starter presets
///
/// | Name | KvQuant | Notes |
/// |---|---|---|
/// | `fp16` | `KvQuant::None` | bf16 both sides (alias: "fp16", "bf16", "none" via `--kv-quant`) |
/// | `q8` | `KvQuant::K8V8` | symmetric 8-bit K+V |
/// | `speed` | `KvQuant::TurboSym3` | Symmetric WHT-3 K+V; matches mtq `speed` exactly; rejected on Qwen MoE (K-side 3-bit PPL-disaster) |
/// | `quality` | `KvQuant::K8V4` | placeholder — true `quality` = turbo4 symm, not yet ported; **diverges from mtq** |
/// | `planar` | `KvQuant::Planar` | PlanarQuant V-side |
/// | `k_only_planar` | `KvQuant::PlanarK` | PlanarQuant K-side; V=bf16; rejected on Qwen MoE |
///
/// Future presets to append: `balanced`, `max_compression`,
/// `k_only_iso`, `agents_8x16k`, `rot_k_quality`.
static PRESETS: &[(&str, PresetSpec)] = &[
    (
        "fp16",
        PresetSpec {
            // KvQuant::None = bf16/unquantised both sides.
            // This is the KvQuant *variant* named None, NOT an Option::None.
            kv_quant: KvQuant::None,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
    (
        "planar",
        PresetSpec {
            kv_quant: KvQuant::Planar,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
    (
        "q8",
        PresetSpec {
            kv_quant: KvQuant::K8V8,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
    (
        // `quality` resolves to `KvQuant::TurboSym4` (symmetric WHT-4 K + tq4 V),
        // matching mtq's `quality` definition byte-for-byte on Apple Silicon.
        // Arch guard: rejected at resolve-time on Qwen MoE (PPL-218→8641 disaster).
        "quality",
        PresetSpec {
            kv_quant: KvQuant::TurboSym4,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
    (
        // `speed` resolves to TurboSym3 (symmetric WHT-3 K+V), matching mtq's
        // `speed` preset definition. Symmetric turbo3 saves ~4-bit of K storage
        // vs K8VTurbo3; K-side WHT-3 matches the V-side codebook exactly.
        // Cosine gate ≥ 0.9807 (K-side empirical floor).
        // Arch guard (Contract A.y): rejected on Qwen MoE (K-side 3-bit is the
        // PPL-disaster zone).
        "speed",
        PresetSpec {
            kv_quant: KvQuant::TurboSym3,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
    (
        // `k_only_planar` resolves to `KvQuant::PlanarK` (K-axis PlanarQuant
        // 4-bit; V stays bf16). Mirrors mtq's `k_only_planar` preset.
        // Arch guard (Contract A.y): rejected at resolve-time on Qwen MoE
        // (K-side 4-bit is the PPL-218→8641 disaster).
        "k_only_planar",
        PresetSpec {
            kv_quant: KvQuant::PlanarK,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
    (
        // `planar3` preset: K = affine q8_0, V = PlanarQuant 3-bit.
        // 3.25-bit effective V; same Givens-rotation algorithm as `planar` but
        // with 3-bit Lloyd-Max N(0,1) codebook. ForgeAttention-compatible pack
        // format (10 vals/u32, 4 words/group).
        "planar3",
        PresetSpec {
            kv_quant: KvQuant::Planar3,
            sparse_attn: None,
            requires_calibration: false,
        },
    ),
];

/// Comma-separated list of all available preset names, for error hints.
/// Does NOT include `auto` — `auto` is resolved before this list is consulted.
pub(crate) const AVAILABLE_NAMES: &str = "fp16, q8, speed, quality, planar, planar3, k_only_planar";

// ── auto-selector helpers ────────────────────────────────────────────────────

/// Return `true` if `name` is a known preset in the static [`PRESETS`] table.
///
/// Used by the auto-selector fallback chain to detect when a preferred
/// preset has not yet landed.
fn preset_exists(name: &str) -> bool {
    PRESETS.iter().any(|(k, _)| *k == name)
}

/// Select the best available 4-bit-class preset.
///
/// Tries `quality` first (WHT-4, near-lossless), then `q8` (always present).
pub(crate) fn preferred_4bit() -> &'static str {
    for &p in &["quality", "q8"] {
        if preset_exists(p) {
            return p;
        }
    }
    // q8 is a starter preset and must always be present; this path is
    // unreachable unless the table is corrupted.
    "q8"
}

/// Select the best available 2-bit-class (maximum compression) preset.
///
/// Tries `max_compression` → `balanced` → `quality` → `q8` in order.
pub(crate) fn preferred_2bit() -> &'static str {
    for &p in &["max_compression", "balanced", "quality", "q8"] {
        if preset_exists(p) {
            return p;
        }
    }
    "q8"
}

/// Recommend a KV-cache preset name given hardware and model constraints.
///
/// This is the core of the auto-selector.  The decision tree is adapted
/// from the `multi-turboquant` reference (`presets.py:recommend_preset`) with
/// rMLX-specific preset names.
///
/// # Decision tree
///
/// ```text
/// model_bytes   = model_size_b × 2e9          (bf16 weight estimate)
/// kv_bf16_bytes = model_size_b × ctx × 1e6    (rough KV-cache at bf16)
/// total_bf16    = model_bytes + kv_bf16_bytes
/// vram_budget   = available_vram_gb × 1e9 × 0.70  (70% safe utilisation)
///
/// if total_bf16        < budget → "fp16"              (unquantised)
/// if model + kv/2      < budget → "q8"                (8-bit K+V)
/// if model + kv/4      < budget → preferred_4bit()    (quality / q8)
/// if model + kv/8      < budget → preferred_2bit()    (max_compression / balanced / quality / q8)
/// else                          → "max_compression_fallback"
/// ```
///
/// The `1e6` factor in `kv_bf16_bytes` is a deliberately conservative constant
/// (approximately `1 MB × model_size_B` per token). It overestimates actual KV
/// footprint by 10–30x for typical transformer architectures, providing a safe
/// margin that biases the selector toward compression rather than risking OOM.
/// Do NOT use these byte counts for tight memory accounting.
///
/// # Arguments
///
/// - `model_size_b` — estimated parameter count in billions.
/// - `context_tokens` — target context window length in tokens.
/// - `available_vram_gb` — unified DRAM available (SI GB).  Callers use the
///   `sysctl hw.memsize` result or a conservative 8 GB fallback.
///
/// # Return value
///
/// A `&'static str` preset name from [`PRESETS`], or `"max_compression_fallback"`
/// when nothing fits.  The caller must resolve `"max_compression_fallback"` to
/// the least-bad available preset (typically `preferred_2bit()`).
pub(crate) fn recommend_preset(
    model_size_b: f32,
    context_tokens: u32,
    available_vram_gb: f32,
) -> &'static str {
    let model_bytes = f64::from(model_size_b) * 2e9_f64;
    let kv_bf16_bytes = f64::from(model_size_b) * f64::from(context_tokens) * 1e6_f64;
    let total_bf16 = model_bytes + kv_bf16_bytes;
    // 70% safe-utilisation cap — leaves headroom for activations, Metal
    // command buffers, and the OS.
    let vram_budget = f64::from(available_vram_gb) * 1e9_f64 * 0.70_f64;

    if total_bf16 < vram_budget {
        return "fp16";
    }
    if model_bytes + kv_bf16_bytes / 2.0 < vram_budget {
        return "q8";
    }
    let p4 = preferred_4bit();
    if model_bytes + kv_bf16_bytes / 4.0 < vram_budget {
        return p4;
    }
    let p2 = preferred_2bit();
    if model_bytes + kv_bf16_bytes / 8.0 < vram_budget {
        return p2;
    }
    // Nothing in the table fits.  Return p2 as the least-bad choice — the
    // caller logs a warning and the user can override with --kv-quant.
    "max_compression_fallback"
}

/// Look up a preset by name.
///
/// Returns:
/// - `Ok(&'static PresetSpec)` — name found in the table.
/// - `Err(PresetError::Reserved)` — name is `"auto"` (reserved for the auto-selector).
/// - `Err(PresetError::Unknown)` — name not in table.
pub(crate) fn lookup_preset(name: &str) -> Result<&'static PresetSpec, PresetError> {
    if name == "auto" {
        return Err(PresetError::Reserved);
    }
    PRESETS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
        .ok_or(PresetError::Unknown)
}

#[cfg(test)]
#[path = "preset_table_tests.rs"]
mod preset_table_tests;
