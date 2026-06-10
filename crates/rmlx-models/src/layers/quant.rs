//! Quantization mode enum and per-tensor quant parameter resolution.

/// Low-cardinality quantization mode — replaces stringly-typed `mode: String`
/// in `Linear::Quantized` and `Embedding::Quantized`.
///
/// Saves 24 B (String header) + 1 heap allocation per quantized layer at load
/// time. Hundreds of `Linear::Quantized` instances per model × ~24 B = KBs
/// less heap + zero allocator traffic.
///
/// `as_str()` returns the exact string value that MLX / `quantized_matmul`
/// / `dequantize` expect — no serde format change.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — four MLX quantization modes; adding a mode requires synchronized changes to as_str(), From<&str>, and all quantized_matmul call sites"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Integer-affine quantization (MLX default). MLX string: `"affine"`.
    Affine,
    /// OCP microscaling fp8. MLX string: `"mxfp8"`.
    Mxfp8,
    /// OCP microscaling fp4. MLX string: `"mxfp4"`.
    Mxfp4,
    /// Nvidia fp4 (E4M3, signed scale). MLX string: `"nvfp4"`.
    /// Note: MLX bug ml-explore/mlx#2962 — unsigned spec, signed impl.
    Nvfp4,
}

impl QuantMode {
    /// The exact string value passed to MLX / `quantized_matmul` / `dequantize`.
    pub fn as_str(self) -> &'static str {
        match self {
            QuantMode::Affine => "affine",
            QuantMode::Mxfp8 => "mxfp8",
            QuantMode::Mxfp4 => "mxfp4",
            QuantMode::Nvfp4 => "nvfp4",
        }
    }
}

impl std::fmt::Display for QuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parse a quantization mode string into `QuantMode`.
///
/// Unrecognised strings fall back to `QuantMode::Affine` (matches the MLX
/// `mode_or_default()` convention of defaulting to `"affine"`).
///
/// # Panics
///
/// Never panics; unrecognised input returns `Affine` silently.
impl From<&str> for QuantMode {
    fn from(s: &str) -> Self {
        match s {
            "mxfp8" => QuantMode::Mxfp8,
            "mxfp4" => QuantMode::Mxfp4,
            "nvfp4" => QuantMode::Nvfp4,
            _ => QuantMode::Affine, // "affine" + anything else
        }
    }
}

// ---------------------------------------------------------------------------
// QuantParams + resolve_quant
// ---------------------------------------------------------------------------

/// Per-tensor quantization parameters, possibly overriding global defaults.
///
/// Laguna (and potentially future architectures) store per-tensor overrides
/// directly in the `quantization` dict of `config.json`, keyed by tensor name.
/// This helper centralises the lookup so arch modules don't duplicate the logic.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — three quant parameter fields; adding a field requires updating global() constructor and all resolve_quant call sites"
)]
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Affine/mxfp8 quantization group size.
    pub group_size: i32,
    /// Quantization bit-width.
    pub bits: i32,
    /// Quantization mode string.
    pub mode: String,
}

impl QuantParams {
    /// Build from global defaults.
    pub fn global(group_size: i32, bits: i32, mode: impl Into<String>) -> Self {
        QuantParams {
            group_size,
            bits,
            mode: mode.into(),
        }
    }
}

/// Resolve quantization parameters for a given tensor base name, owning the
/// `.biases`-sibling / affine rule shared by every quantized architecture.
///
/// Override lookup is an **exact-key** match: `overrides.get(tensor_name)`.
/// The key must equal the tensor base path exactly (e.g.
/// `"language_model.model.layers.5.router.proj"`); there is no suffix or
/// prefix matching. When an override is present its `group_size`/`bits` win,
/// and its `mode` wins when non-empty (else the global default `mode` is
/// inherited).
///
/// After resolving `(group_size, bits, mode)`, the `.biases` sibling governs
/// the final mode. Only a mode that an override sets **explicitly** can clash;
/// a non-affine *global default* mode inherited by a biased tensor is treated
/// as the common affine-int-checkpoint case (gemma4-26b's ~120 biased tensors
/// inherit the global `mxfp8` default yet are affine) and is forced to affine,
/// matching the pre-unification per-arch resolvers:
/// - `has_biases`, no override-set mode → force `"affine"` (per-group
///   zero-point `.biases` ⇒ integer-affine regardless of the global mode).
/// - `has_biases` + override mode already `"affine"` → keep `"affine"`.
/// - `has_biases` + override sets an explicit **non-affine** mode (`mxfp8`,
///   `nvfp4`, …) → hard error: an affine `.biases` sibling cannot decode under
///   a microscaling/fp4 mode, so the config is internally contradictory.
/// - `!has_biases` → the resolved mode is used as-is.
///
/// The hard-error branch is the one behavior new to the unified resolver: it
/// replaces gemma4's honor-the-explicit-override and laguna/qwen3.5-moe's silent
/// force-affine for the contradictory case. No on-disk snapshot in the registry
/// hits it; reaching it means the `config.json` quant block is malformed.
pub fn resolve_quant(
    tensor_name: &str,
    has_biases: bool,
    defaults: &QuantParams,
    overrides: &std::collections::HashMap<String, QuantParams>,
) -> rmlx_core::error::Result<QuantParams> {
    // Track whether a non-empty mode was set by the override itself; only an
    // override-set mode can contradict the biases sibling. A non-affine *global
    // default* inherited by a biased tensor is the normal affine-checkpoint case.
    let override_entry = overrides.get(tensor_name);
    let mode_set_by_override = override_entry.is_some_and(|ov| !ov.mode.is_empty());

    let (group_size, bits, mode) = if let Some(ov) = override_entry {
        let mode = if ov.mode.is_empty() {
            defaults.mode.clone()
        } else {
            ov.mode.clone()
        };
        (ov.group_size, ov.bits, mode)
    } else {
        (defaults.group_size, defaults.bits, defaults.mode.clone())
    };

    if has_biases {
        if mode_set_by_override && QuantMode::from(mode.as_str()) != QuantMode::Affine {
            // An explicit override mode that is not affine alongside an affine
            // `.biases` sibling is a contradiction MLX cannot decode. Parse the
            // mode first so an unrecognized-but-affine string (which
            // `QuantMode::from` maps to Affine) is not refused — only a genuine
            // microscaling/fp4 mode (`mxfp8`, `nvfp4`, …) errors.
            return Err(rmlx_core::error::Error::Loader(format!(
                "config quant mode '{mode}' for '{tensor_name}' contradicts the \
                 .biases sibling: affine biases cannot decode under {mode}"
            )));
        }
        // Affine-int checkpoint: a per-group zero-point `.biases` tensor is
        // present, so the mode is "affine" regardless of the global mode.
        return Ok(QuantParams {
            group_size,
            bits,
            mode: "affine".to_owned(),
        });
    }

    Ok(QuantParams {
        group_size,
        bits,
        mode,
    })
}
