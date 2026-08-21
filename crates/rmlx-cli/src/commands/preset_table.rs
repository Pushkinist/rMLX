//! Named KV-cache preset table — `--kv-preset <name>` resolution.
//!
//! Each preset is a `PresetSpec` that bundles a `KvQuant`, an optional sparse-
//! attention spec, and a `requires_calibration` flag.
//!
//! # Adding a new preset
//!
//! A new row belongs here only once its codec's decode reads its own packed
//! store — until then it is another spelling of `fp16`, which
//! `no_preset_is_a_memory_lever` states and pins. If that day comes: append to
//! the `PRESETS` const below and add a test row in `preset_table_tests.rs`; the
//! resolution path in `parse.rs` is unchanged.
//!
//! # `auto`
//!
//! `--kv-preset auto` resolves to `rmlx_models::kv_cache::DEFAULT_KV_QUANT`,
//! the same constant `--kv-quant auto` resolves to. It does not consult this
//! table.
//!
//! It used to run a memory-pressure decision tree that picked a "compressing"
//! preset when the model plus its bf16 KV would not fit in unified memory.
//! Every preset it could return holds resident KV **byte-identical** to
//! `fp16`, so no branch of that tree changed a byte — it answered a memory
//! question with a codec that has no memory effect. The tree is gone rather
//! than warned about, because its own KV estimate was 10–30× off and could not
//! be the basis of a diagnostic either.
//!
//! `lookup_preset("auto")` still returns `Err(PresetError::Reserved)` —
//! `parse_kv_preset` in `parse.rs` intercepts `"auto"` before `lookup_preset`
//! is called.

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
/// scan (7 entries) rather than `phf` — the table is tiny and `phf` is not
/// currently in the workspace.  When the table grows past ~20 entries, switch
/// to a `phf::Map`.
///
/// ## Presets
///
/// | Name | KvQuant | Notes |
/// |---|---|---|
/// | `fp16` | `KvQuant::None` | bf16 both sides (alias: "fp16", "bf16", "none" via `--kv-quant`) |
/// | `q8` | `KvQuant::K8V8` | symmetric 8-bit K+V |
/// | `speed` | `KvQuant::TurboSym3` | Symmetric WHT-3 K+V; matches mtq `speed` exactly; rejected on Qwen MoE (K-side 3-bit PPL-disaster) |
/// | `quality` | `KvQuant::TurboSym4` | symmetric WHT-4 K + tq4 V; rejected on Qwen MoE |
/// | `planar` | `KvQuant::Planar` | PlanarQuant V-side |
/// | `planar3` | `KvQuant::Planar3` | PlanarQuant 3-bit V-side |
/// | `k_only_planar` | `KvQuant::PlanarK` | PlanarQuant K-side; V=bf16; rejected on Qwen MoE |
///
/// **None of the six non-`fp16` rows reduces resident KV.** Every one of them
/// resolves to a codec whose decode reads the bf16 mirror and whose packed
/// store is therefore never built, so a served request holds exactly the bytes
/// `fp16` holds and emits exactly the token ids `fp16` emits. They are kept
/// selectable — the names appear in recorded bench rows and each is the entry
/// point for its codec's re-enable path — but a preset is not a memory lever
/// on this tree. See `docs/KV_QUANT.md` § "Codec disposition".
///
/// A new row belongs here only once its codec's decode reads its own packed
/// store; until then it would be another spelling of `fp16`.
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

/// Look up a preset by name.
///
/// Returns:
/// - `Ok(&'static PresetSpec)` — name found in the table.
/// - `Err(PresetError::Reserved)` — name is `"auto"`, which resolves to
///   `DEFAULT_KV_QUANT` in `parse.rs` and never reaches this table.
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
