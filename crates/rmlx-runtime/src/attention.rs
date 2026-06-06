//! Attention-block helpers shared across architectures.
//!
//! The runtime extraction surfaces only the safest helpers:
//! - `repeat_kv` — GQA expansion. Byte-identical across qwen3, qwen2,
//!   qwen3_5_moe (its `layers.rs`), gemma3, laguna.
//!
//! The full SDPA dispatch (q-norm placement, partial vs full RoPE, gating
//! before `o_proj`, sliding-vs-full mask routing) varies enough across the
//! six archs that a single `attention_step()` would need a wide knob struct.
//! We deliberately keep the per-arch SDPA blocks intact for now and only
//! extract the small pure helpers. The mask builders themselves
//! (`build_chunked_prefill_mask`, `build_swa_*`, `pick_attn_mask_mode`)
//! already live in `rmlx-models::layers` and are reused as-is.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device};

/// Expand K/V from `[B, kv_heads, S, D]` to `[B, q_heads, S, D]` by
/// broadcasting along a new axis. Equivalent to `np.repeat(x, repeat, axis=1)`.
///
/// MLX's fast SDPA kernel accepts un-expanded K/V when
/// `q_heads % kv_heads == 0`, so callers should only invoke this when SDPA
/// requires explicit expansion (e.g. arch tests, custom kernels).
#[allow(
    clippy::indexing_slicing,
    reason = "shape is guaranteed to be 4-D [B, kv_heads, S, D] by all KV-cache callers; indexing [0..3] is invariant-safe"
)]
pub fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    if repeat == 1 {
        return x.try_clone();
    }
    let shape = x.shape();
    let (b, kv_h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    let x5 = rmlx_mlx::expand_dims(x, 2, device)?;
    let bc = rmlx_mlx::broadcast_to(&x5, &[b, kv_h, repeat as i32, s, d], device)?;
    bc.reshape(&[b, kv_h * repeat as i32, s, d], device)
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
