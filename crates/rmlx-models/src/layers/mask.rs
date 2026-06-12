//! Attention mask builders for causal, chunked-prefill, and SWA forward passes.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

/// Pick SDPA mask mode for a forward step.
///
/// `pre_offset` = cache offset BEFORE the current K/V was appended (0 for prefill).
/// `new_seq` = number of new query positions in this call.
///
/// Cases:
/// - offset == 0: standard causal mask over [new_seq, new_seq] — `"causal"`.
/// - offset > 0 + new_seq == 1: single decode step — query attends all keys, no mask (`""`).
/// - offset > 0 + new_seq > 1: chunked prefill — explicit `[new_seq, offset+new_seq]` mask,
///   returned as `"array"`. Caller must call `build_chunked_prefill_mask` and pass the
///   result to `scaled_dot_product_attention`.
pub fn pick_attn_mask_mode(pre_offset: i32, new_seq: i32) -> &'static str {
    if pre_offset == 0 {
        "causal"
    } else if new_seq == 1 {
        ""
    } else {
        // Chunked prefill: needs an explicit [new_seq, offset+new_seq] lower-triangular
        // additive mask. MLX's valid mask_mode values are "", "causal", and "array".
        "array"
    }
}

/// Build the explicit additive mask for a chunked-prefill SDPA call.
///
/// Shape: `[1, 1, new_seq, offset + new_seq]` (broadcast over batch and heads).
///
/// Value convention (matches MLX "array" mode additive mask semantics):
/// - `0.0` — position is allowed (query i may attend to key j).
/// - `-1e30` — position is masked (query i must not attend to key j).
///
/// Lower-triangular rule: query at position `offset + i` may attend to all keys
/// `j` where `j <= offset + i`, i.e. columns `0 .. offset + i + 1`.
///
/// MLX's valid `mask_mode` values: `""`, `"causal"`, `"array"`.
/// `"additive"` is NOT accepted by mlx-c — always use `"array"` with this mask.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn build_chunked_prefill_mask(offset: i32, new_seq: i32, device: Device) -> Result<Array> {
    let rows = new_seq as usize;
    let cols = (offset + new_seq) as usize;
    let mut data = vec![-1e30_f32; rows * cols];

    for i in 0..rows {
        // Query absolute position: offset + i. Allow keys 0..=offset+i.
        let allow_up_to = offset as usize + i + 1; // exclusive upper bound
        for j in 0..allow_up_to.min(cols) {
            data[i * cols + j] = 0.0;
        }
    }

    let mask_f32 = Array::from_f32_slice(&data, &[1, 1, rows as i32, cols as i32])?;
    // Convert to BF16 to match Q/K/V dtype (MLX requires mask dtype to promote to output).
    mask_f32.astype(Dtype::Bf16, device)
}

/// Build the banded-causal prefill mask for a Sliding-Window Attention (SWA) layer.
///
/// SWA restricts each query to attend only to keys within the last `window` positions.
/// For prefill (full sequence), this produces a banded lower-triangular matrix.
///
/// Shape: `[1, 1, new_seq, offset + new_seq]` — same as `build_chunked_prefill_mask`.
///
/// Mask rule (query absolute position `q_abs = offset + i`, key absolute position `j`):
/// - Allow: `j <= q_abs` AND `q_abs - j < window`
///   → keys in range `[q_abs - window + 1, q_abs]` (clipped to `[0, q_abs]`).
/// - Block: `j > q_abs` (future tokens) OR `q_abs - j >= window` (too far in past).
///
/// When `offset == 0` (first prefill chunk, no preceding context), this degenerates to a
/// banded lower-triangular matrix. For chunked prefill chunks (offset > 0), the window
/// is applied relative to each query's absolute position.
///
/// Reference: mlx-lm `gemma3_text.py` Gemma3Model.__call__:
/// `sliding_window_mask = create_attention_mask(h, cache[0], window_size=self.window_size)`
/// where `create_attention_mask` in `base.py` uses:
/// `mask[q, k] = True if (q_pos - k_pos >= window_size) or (k_pos > q_pos)`
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn build_swa_prefill_mask(
    offset: i32,
    new_seq: i32,
    window: usize,
    device: Device,
) -> Result<Array> {
    let rows = new_seq as usize;
    let cols = (offset + new_seq) as usize;
    let window = window as i64;
    let mut data = vec![-1e30_f32; rows * cols];

    for i in 0..rows {
        let q_abs = i64::from(offset) + i as i64; // absolute query position
        for j in 0..cols {
            let k_abs = j as i64; // absolute key position
                                  // Allow: key is not in the future AND within the window.
            if k_abs <= q_abs && q_abs - k_abs < window {
                data[i * cols + j] = 0.0;
            }
        }
    }

    let mask_f32 = Array::from_f32_slice(&data, &[1, 1, rows as i32, cols as i32])?;
    mask_f32.astype(Dtype::Bf16, device)
}

/// Build the decode mask for a Sliding-Window Attention (SWA) layer when
/// `total_kv_len > window`.
///
/// During decode (seq == 1), query position is `offset` (= total prefill + prior decode steps).
/// The key sequence has length `total_kv_len = offset + 1` after `c.update()`.
///
/// If `total_kv_len <= window`, the single query can attend all keys — return `None`
/// (caller uses mask_mode `""`).
///
/// If `total_kv_len > window`, we must mask the oldest `total_kv_len - window` keys.
/// Shape: `[1, 1, 1, total_kv_len]`.
///
/// `total_kv_len`: K length after appending the new decode token (= `offset + 1`).
pub fn build_swa_decode_mask(
    total_kv_len: i32,
    window: usize,
    device: Device,
) -> Result<Option<Array>> {
    if total_kv_len <= window as i32 {
        return Ok(None);
    }
    let cols = total_kv_len as usize;
    let window = window as i32;
    let first_allowed = total_kv_len - window; // oldest allowed key index (absolute)

    let mut data = vec![-1e30_f32; cols];
    for cell in data.iter_mut().skip(first_allowed as usize) {
        *cell = 0.0;
    }

    let mask_f32 = Array::from_f32_slice(&data, &[1, 1, 1, cols as i32])?;
    let mask_bf16 = mask_f32.astype(Dtype::Bf16, device)?;
    Ok(Some(mask_bf16))
}
