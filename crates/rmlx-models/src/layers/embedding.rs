//! Embedding table — plain or quantized (also usable as tied-weight lm_head).

use rmlx_core::error::Result;
use rmlx_mlx::{dequantize, matmul, quantized_matmul, Array, Device, Dtype};

use super::quant::QuantMode;

/// Embedding table — plain or quantized.
/// Also usable as a tied-weight output projection via `as_linear`.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two embedding storage variants (plain/quantized); adding a variant requires updating forward(), as_linear(), and all arch embedding-loading sites"
)]
/// Embedding table — plain or quantized (see docs/WEIGHT_QUANTS.md).
#[allow(missing_debug_implementations)]
pub enum Embedding {
    /// Plain bf16 embedding table.
    Plain {
        /// Embedding weight matrix `[vocab_size, hidden_size]` in bf16.
        weight: Array,
    },
    /// Affine/mxfp8 quantized embedding table.
    Quantized {
        /// Packed codes.
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Affine zero-point biases. `None` for mxfp8 and standard affine-int without biases.
        /// `Some` for PARO INT4 embeddings (derived from qzeros at load time).
        biases: Option<Array>,
        /// Number of elements per quantization group.
        group_size: i32,
        /// Quantization bit-width.
        bits: i32,
        /// Quantization mode — 1 B (Copy enum) vs 24 B String.
        mode: QuantMode,
    },
}

impl Embedding {
    /// Look up token ids. `ids` shape: `[seq]` I32.
    pub fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => weight.take(ids, 0, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => build_one_hot_and_lookup(
                ids,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode.as_str(),
                device,
            ),
        }
    }

    /// Treat embedding as a linear layer for tied-weights output projection.
    /// Computes x @ weight.T (plain) or quantized_matmul (quant).
    pub fn as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => matmul(x, &weight.transpose(&[1, 0], device)?, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode.as_str(),
                true,
                device,
            ),
        }
    }
}

/// On-device quantized embedding lookup.
///
/// Mirrors `mlx_lm.nn.QuantizedEmbedding.__call__`:
/// `dequantize(weight[ids], scales[ids], biases[ids], …)`
///
/// Earlier versions ran this through `Device::Cpu` with an `eye(seq) @ w`
/// trick, forcing a GPU↔CPU round-trip on every decode step. That round-trip
/// blocks the `pending: Option<Array>` async pipeline and is the
/// dominant per-step cost on dense Gemma4 (e4b mxfp8 35 TPS). The on-device
/// `take + dequantize` path keeps everything on `device`, letting MLX fuse the
/// lookup with subsequent layers.
///
/// Works for both mxfp8 (no biases) and PARO INT4 affine (with biases) — the
/// underlying `dequantize` op handles either via the `mode` argument.
fn build_one_hot_and_lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    let weight_rows = weight.take(ids, 0, device)?;
    let scales_rows = scales.take(ids, 0, device)?;
    let biases_rows = biases.map(|b| b.take(ids, 0, device)).transpose()?;
    let dq = dequantize(
        &weight_rows,
        &scales_rows,
        biases_rows.as_ref(),
        group_size,
        bits,
        mode,
        device,
    )?;
    // Downstream layers (RoPE, attention masks, RmsNorm) expect BF16 activations.
    // mxfp8 scales are typically BF16 already (no-op cast); PARO INT4 scales are
    // F16 — force BF16 to match downstream chunked-prefill mask dtype.
    if dq.dtype() == Dtype::Bf16 {
        Ok(dq)
    } else {
        dq.astype(Dtype::Bf16, device)
    }
}
