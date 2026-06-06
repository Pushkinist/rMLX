//! Per-(layer, head) calibration sink trait.
//!
//! Surgical hook plumbed through `Qwen3Text::forward_seq_with_cache →
//! TransformerBlock::forward → Attention::forward` to capture the last-position
//! query and the full accumulated K tensor at the post-RoPE / pre-SDPA
//! insertion point. Steady-state production callers pass `None`; the optimizer
//! dead-code-eliminates the None branch, making per-token overhead negligible.
//! The trait fires only when a calibration session has installed `Some(sink)`.
//!
//! The trait surface is intentionally minimal — the sink owns its own
//! aggregation strategy (e.g. max-across-prompts for softmax-mass
//! budgeting). Engine code only handles array hand-off.
//!
//! # Insertion point (Qwen3)
//!
//! In `crates/rmlx-models/src/qwen3.rs` `Attention::forward`, immediately after
//! per-head q/k RMSNorm + RoPE and before the SDPA dispatch — at the
//! "post-RoPE, pre-SDPA" boundary identified by the calibration audit.

use rmlx_core::error::Result;
use rmlx_mlx::Array;

/// Per-(layer, head) calibration capture point.
///
/// Implementors receive the last-position Q tensor (shape
/// `[1, n_q_heads, 1, head_dim]` post-RoPE) and the **full accumulated** K
/// tensor (shape `[1, n_kv_heads, S_kv, head_dim]` post-RoPE) for one prompt
/// at one layer, with the head dimension flattened by the impl as required.
///
/// The sink is responsible for any aggregation across prompts (max,
/// per-prompt-stash, etc.). Engine code does **not** loop over heads — that is
/// the sink's job, because the (q, k) hand-off cost is dominated by the cross
/// product per (kv_head, q_pos) pair rather than the head loop overhead.
pub trait CalibrationSink {
    /// Record a captured (q_last, k_full) pair for one decoder layer.
    ///
    /// - `layer_idx` is the 0-based decoder layer index.
    /// - `q_last` is the **last-row** query tensor (post-RoPE),
    ///   shape `[1, n_q_heads, 1, head_dim]`.
    /// - `k_full` is the full per-layer K tensor (post-RoPE),
    ///   shape `[1, n_kv_heads, S_kv, head_dim]` where `S_kv` is the current
    ///   prompt length.
    ///
    /// Returning `Err` aborts the forward pass; sinks that want to swallow a
    /// row should do so internally.
    fn record(&mut self, layer_idx: usize, q_last: &Array, k_full: &Array) -> Result<()>;
}
