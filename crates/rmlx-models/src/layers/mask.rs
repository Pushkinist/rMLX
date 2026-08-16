//! Attention mask builders for causal, chunked-prefill, and SWA forward passes.
//!
//! The two prefill builders are O(`new_seq` × (`offset` + `new_seq`)) in mask
//! elements, which at long context is the largest single buffer a prefill chunk
//! touches. They are therefore built from MLX position vectors — `arange`,
//! a broadcast comparison, `where` — so the buffer is produced where it is
//! consumed and never crosses the host boundary. Building them scalar-wise on
//! the host instead cost three full-size buffers per call (an f32 `Vec`, its
//! upload, and the bf16 cast) and made a 64k-token prefill allocate tens of GB
//! of transient host memory, which on a loaded machine exhausts free RAM and
//! turns repeated in-process generations into a monotonic slowdown.
//!
//! `build_swa_decode_mask` deliberately keeps its host fill. It is the
//! `seq == 1` path, so it is O(`kv_len`) rather than O(`seq` × `kv_len`), and
//! it is unreachable whenever the SWA cache rotates — which every shipped
//! gemma3 / gemma4 / laguna configuration sets. Moving it to the device would
//! add dispatches to a per-token path to save a buffer that is three orders of
//! magnitude smaller than the ones above.
//!
//! **Sharing one mask across the layers of a forward call is not automatically
//! a win; measure it per architecture.** It looks like an obvious saving — a
//! chunk needs at most one full-attention and one banded mask. Two
//! measurements, same binary except for the sharing, `--kv-quant none`,
//! `--warmup 0 --runs 3`, free memory settled before each cell:
//!
//! * gemma-4-e2b (35 layers, 7 full + 28 SWA, `head_dim` 256), 68 898-token
//!   prompt: sharing cost **2x** the prefill time (9.5 s vs 4.8 s) and pushed
//!   the host from 0 to ~40 GiB of compressor growth. Per-layer wins by a lot.
//! * Qwen3.6-35B-A3B (40 layers, 10 full + 30 linear, `head_dim` 256), 34k
//!   prompt: 12.27 s shared vs 12.47 s per-layer — sharing wins by 1.6%, inside
//!   run-to-run spread. No penalty at all.
//!
//! So the gemma-4 result does not generalise, and `qwen3_5_moe` / `qwen3_vl_moe`
//! keep the per-forward `shared_mask` they have always had. What produces the
//! gemma-4 magnitude is not established: one extra live mask is ~140 MB at that
//! cell, two orders of magnitude short of the difference, and sharing strictly
//! *reduces* total bytes allocated. Do not restate a mechanism for it here
//! without measuring one.

use rmlx_core::error::Result;
use rmlx_mlx::{arange, greater_equal, multiply, scalar_f32, where_cond, Array, Device, Dtype};

/// Additive bias applied to a blocked (query, key) pair.
///
/// Large-negative rather than `-inf` so a fully-blocked row still softmaxes to
/// a finite distribution instead of NaN. Representable in bf16, which is the
/// dtype every mask here is produced in.
const BLOCKED_BIAS: f32 = -1e30;

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
pub fn build_chunked_prefill_mask(offset: i32, new_seq: i32, device: Device) -> Result<Array> {
    // Query absolute positions as a column, key absolute positions as a row:
    // the mask is their broadcast comparison.
    let (q_pos, k_pos) = position_axes(offset, new_seq, device)?;
    let allowed = greater_equal(&q_pos, &k_pos, device)?;
    let (open, blocked) = mask_scalars(device)?;
    where_cond(&allowed, &open, &blocked, device)
}

/// Build the `[1, 1, new_seq, offset + new_seq]` query / key position axes the
/// prefill masks compare.
///
/// `q_pos` holds absolute query positions `offset .. offset + new_seq` as a
/// column; `k_pos` holds absolute key positions `0 .. offset + new_seq` as a
/// row. Both are F32: every position a mask can address is an exact integer in
/// F32 far below the 2^24 mantissa limit.
fn position_axes(offset: i32, new_seq: i32, device: Device) -> Result<(Array, Array)> {
    let cols = offset + new_seq;
    let q_pos = position_column(i64::from(offset), new_seq, device)?;
    let k_pos = arange(0.0, f64::from(cols), 1.0, device)?.reshape(&[1, 1, 1, cols], device)?;
    Ok((q_pos, k_pos))
}

/// A `[1, 1, len, 1]` column of consecutive positions `start .. start + len`.
///
/// `start` is `i64` and may be negative: the SWA window edge sits `window - 1`
/// positions behind the query, which is before position 0 until the first
/// window has filled.
fn position_column(start: i64, len: i32, device: Device) -> Result<Array> {
    let stop = start + i64::from(len);
    arange(start as f64, stop as f64, 1.0, device)?.reshape(&[1, 1, len, 1], device)
}

/// The two values a `where` selects between to produce an additive mask:
/// `0.0` where the pair is allowed, [`BLOCKED_BIAS`] where it is not.
///
/// BF16 to match Q/K/V dtype — MLX requires the mask dtype to promote to the
/// SDPA output dtype.
fn mask_scalars(device: Device) -> Result<(Array, Array)> {
    let open = scalar_f32(0.0).astype(Dtype::Bf16, device)?;
    let blocked = scalar_f32(BLOCKED_BIAS).astype(Dtype::Bf16, device)?;
    Ok((open, blocked))
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
pub fn build_swa_prefill_mask(
    offset: i32,
    new_seq: i32,
    window: usize,
    device: Device,
) -> Result<Array> {
    let (q_pos, k_pos) = position_axes(offset, new_seq, device)?;
    // Not in the future: k_abs <= q_abs.
    let causal = greater_equal(&q_pos, &k_pos, device)?;
    // Within the window: q_abs - k_abs < window, i.e. k_abs >= q_abs - window + 1.
    // The window edge is its own position column rather than `q_pos` minus a
    // scalar: same values, one op fewer, and no loose scalar in the graph.
    // The arithmetic is i64 so a zero window (block everything) and a window
    // reaching back past position 0 both stay well-defined instead of wrapping.
    let edge_start = i64::from(offset) - (window as i64 - 1);
    let oldest_allowed = position_column(edge_start, new_seq, device)?;
    let in_window = greater_equal(&k_pos, &oldest_allowed, device)?;

    // Allowed where BOTH hold. Both operands are bool (MLX U8), so the product
    // is their conjunction and stays U8 — one byte per cell. Selecting twice
    // instead would materialise a full-size BF16 intermediate to feed the
    // second selection, which is the traffic this construction exists to avoid.
    let allowed = multiply(&causal, &in_window, device)?;
    let (open, blocked) = mask_scalars(device)?;
    where_cond(&allowed, &open, &blocked, device)
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

    let mut data = vec![BLOCKED_BIAS; cols];
    for cell in data.iter_mut().skip(first_allowed as usize) {
        *cell = 0.0;
    }

    let mask_f32 = Array::from_f32_slice(&data, &[1, 1, 1, cols as i32])?;
    let mask_bf16 = mask_f32.astype(Dtype::Bf16, device)?;
    Ok(Some(mask_bf16))
}
