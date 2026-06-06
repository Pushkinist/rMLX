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

/// Resolve quantization parameters for a given tensor base name.
///
/// Checks `overrides` for an entry whose key is a suffix of `tensor_name`
/// (e.g. override key `"model.layers.5.mlp.gate.proj"` matches tensor name
/// `"model.layers.5.mlp.gate.proj"`). If found, that entry's group_size/bits
/// override the defaults; mode falls back to the global default.
///
/// Returns the resolved `(group_size, bits, mode)` tuple.
pub fn resolve_quant(
    tensor_name: &str,
    defaults: &QuantParams,
    overrides: &std::collections::HashMap<String, QuantParams>,
) -> QuantParams {
    if let Some(ov) = overrides.get(tensor_name) {
        QuantParams {
            group_size: ov.group_size,
            bits: ov.bits,
            // Override mode if specified, else inherit global.
            mode: if ov.mode.is_empty() {
                defaults.mode.clone()
            } else {
                ov.mode.clone()
            },
        }
    } else {
        defaults.clone()
    }
}
