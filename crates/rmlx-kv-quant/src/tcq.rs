//! TCQ — Trellis Coded Quantization on the standard TurboQuant Lloyd-Max codebook.
//!
//! 3-bit and 2-bit variants.
//!
//! # What this is
//!
//! Replaces TurboQuant's nearest-centroid V-side assignment with **Viterbi
//! optimal** path selection through a fixed trellis. The codebook is the same
//! Lloyd-Max N(0,1) table as plain `turbo3` / `turbo2` — quality comes purely
//! from smarter assignment that exploits inter-dimension dependencies. The
//! decoder is therefore bit-identical to [`crate::turboquant::turbo_dequantize`].
//!
//! Reference: `multi-turboquant/multi_turboquant/methods/tcq.py` (CPU Viterbi).
//!
//! # Trellis
//!
//! A rate-1/2 convolutional code with constraint length 3:
//!
//! - [`TCQ_NUM_STATES`] = 4 trellis states.
//! - Transition: `next = ((state << 1) | (level & 1)) % NUM_STATES`.
//! - At each dim-axis position the encoder picks the `(state, level)` that
//!   minimises cumulative path cost `path_cost[state] + dist(rotated, level)`.
//! - Forward pass fills a back-pointer table of shape `[dim, NUM_STATES]`
//!   (per-block); backward pass traces from the best final state and emits
//!   centroid indices in stream order.
//!
//! # Layout / storage
//!
//! Output is a [`TurboBlocks`] whose `bits` field equals `log2(codebook_len)`
//! (2 for 4-centroid / 2-bit, 3 for 8-centroid / 3-bit), byte-for-byte
//! indistinguishable from the corresponding plain turbo encoder for the
//! decoder's purposes:
//!
//! - `codes`: LSB-first bit-packed indices, `GROUP_SIZE * bits` per block.
//! - `scales`: per-block `max(|x|) / max_centroid`.
//! - `original_shape`: 4-D `[B, kv_h, S, D]`.
//!
//! Per-block layout matches `turbo_quantize_v`; only assignment differs.

use rmlx_core::{Error, Result};

use crate::turboquant::{lloyd_gaussian_codebook, pack_index, TurboBlocks, GROUP_SIZE};

// ── Trellis constants ────────────────────────────────────────────────────────

/// Number of states in the Viterbi trellis (rate-1/2, constraint length 3).
///
/// Fixed at 4 in the initial port. The ticket allows up to 8 if a future MSL
/// SMEM budget warrants it; widening requires recomputing the transition table
/// and re-tuning the back-trace storage.
pub const TCQ_NUM_STATES: usize = 4;

// ── Public entry points ──────────────────────────────────────────────────────

/// Encode `x` at 3-bit using Viterbi trellis assignment over the built-in
/// Lloyd-Max N(0,1) 3-bit codebook.
///
/// # Input
///
/// `x` is a flat row-major f32 slice for a tensor of shape
/// `original_shape = [B, kv_h, S, D]` (4-D, product = `x.len()`). `D` must be
/// a multiple of [`GROUP_SIZE`] = 32 (same as plain turboquant).
///
/// # Algorithm
///
/// 1. Partition elements into groups of `GROUP_SIZE = 32`.
/// 2. Per group: `scale = max(|x_i|) / max_centroid` (identical to plain
///    `turbo_quantize_v`). Zero-scale block emits all-zero indices.
/// 3. Normalise `x_i / scale`, then run a 4-state Viterbi forward + back-trace
///    over the 32 normalised values to pick centroid indices that minimise the
///    cumulative L2 distortion under the trellis transition constraint.
/// 4. Bit-pack indices LSB-first into u8 bytes.
///
/// # Output
///
/// A [`TurboBlocks`] with `bits = 3` that decodes via
/// [`crate::turboquant::turbo_dequantize`] (no TCQ-specific decoder needed).
///
/// # Errors
///
/// Returns `Error::Quant` if `original_shape` is not 4-D, has a non-positive
/// dim, the product mismatches `x.len()`, or `D` is not a multiple of
/// `GROUP_SIZE`.
pub fn tcq_quantize_v3(x: &[f32], original_shape: &[i32]) -> Result<TurboBlocks> {
    let codebook = lloyd_gaussian_codebook(3)?;
    tcq_encode_with_codebook(x, original_shape, codebook)
}

/// Encode `x` at 2-bit using Viterbi trellis assignment over the built-in
/// Lloyd-Max N(0,1) 2-bit codebook.
///
/// # Input
///
/// Same contract as [`tcq_quantize_v3`]: flat row-major f32 slice for a tensor
/// of shape `original_shape = [B, kv_h, S, D]`; `D` must be a multiple of
/// [`GROUP_SIZE`] = 32.
///
/// # Algorithm
///
/// Identical Viterbi forward + back-trace as `tcq_quantize_v3`; only the
/// codebook width differs (4 centroids instead of 8). The trellis state count
/// stays at 4 (`TCQ_NUM_STATES = 4`).
///
/// # Output
///
/// A [`TurboBlocks`] with `bits = 2` that decodes via
/// [`crate::turboquant::turbo_dequantize`] (no TCQ-specific decoder needed).
/// Pack format: 2-bit codes, 16 values per u32 (same as plain `turbo2`).
///
/// # Errors
///
/// Returns `Error::Quant` if `original_shape` is not 4-D, has a non-positive
/// dim, the product mismatches `x.len()`, or `D` is not a multiple of
/// `GROUP_SIZE`.
pub fn tcq_quantize_v2(x: &[f32], original_shape: &[i32]) -> Result<TurboBlocks> {
    let codebook = lloyd_gaussian_codebook(2)?;
    tcq_encode_with_codebook(x, original_shape, codebook)
}

/// Encode `x` at 3-bit with a custom codebook (length must equal 8).
///
/// Used by callers that ship a calibrated codebook override. Decoder must be
/// invoked with the same override (see
/// [`crate::turboquant::turbo_dequantize_with_codebook`]).
///
/// # Errors
///
/// Returns `Error::Quant` for an unsupported codebook length, an invalid
/// shape, or `D` not divisible by `GROUP_SIZE`.
pub fn tcq_quantize_v3_with_codebook(
    x: &[f32],
    original_shape: &[i32],
    codebook: &[f32],
) -> Result<TurboBlocks> {
    if codebook.len() != 8 {
        return Err(Error::Quant(format!(
            "tcq_quantize_v3_with_codebook: codebook.len()={} must equal 8 (3-bit)",
            codebook.len()
        )));
    }
    if !codebook.windows(2).all(|w| matches!(w, [a, b] if a < b)) {
        return Err(Error::Quant(format!(
            "tcq_quantize_v3_with_codebook: codebook must be strictly ascending; got {codebook:?}"
        )));
    }
    tcq_encode_with_codebook(x, original_shape, codebook)
}

// ── Trellis transition table ─────────────────────────────────────────────────

/// Build the trellis transition table for `num_levels` centroids.
///
/// Returns a `[TCQ_NUM_STATES * num_levels]` flat lookup where
/// `tbl[state * num_levels + level]` is the next state.
///
/// Bit pattern: `next = ((state << 1) | (level & 1)) mod NUM_STATES`. Mirrors
/// the reference `_build_trellis_transitions` in `tcq.py`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "tbl is sized exactly TCQ_NUM_STATES * num_levels above; both loop bounds are inside; \
              next is in [0, TCQ_NUM_STATES) so fits u8 (NUM_STATES <= 8 by codec invariant)"
)]
fn build_transition_table(num_levels: usize) -> Vec<u8> {
    let mut tbl = vec![0u8; TCQ_NUM_STATES * num_levels];
    for state in 0..TCQ_NUM_STATES {
        for level in 0..num_levels {
            let next = ((state << 1) | (level & 1)) % TCQ_NUM_STATES;
            tbl[state * num_levels + level] = next as u8;
        }
    }
    tbl
}

// ── Core encoder ─────────────────────────────────────────────────────────────

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    reason = "bounds established by construction: block/elem loops range over validated lengths; \
              transition table sized num_states * num_levels; codebook indexed by u8 in [0, num_levels)"
)]
fn tcq_encode_with_codebook(
    x: &[f32],
    original_shape: &[i32],
    codebook: &[f32],
) -> Result<TurboBlocks> {
    // Validate shape (matches turbo_quantize_v_with_codebook).
    if original_shape.len() != 4 {
        return Err(Error::Quant(format!(
            "tcq: original_shape must have 4 elements [B, kv_h, S, D], got {} element(s)",
            original_shape.len()
        )));
    }
    if original_shape.iter().any(|&d| d <= 0) {
        return Err(Error::Quant(format!(
            "tcq: all shape dimensions must be positive, got {original_shape:?}"
        )));
    }
    let shape4: [i32; 4] = [
        original_shape[0],
        original_shape[1],
        original_shape[2],
        original_shape[3],
    ];
    let total_elems: usize = original_shape.iter().map(|&d| d as usize).product();
    if x.len() != total_elems {
        return Err(Error::Quant(format!(
            "tcq: x.len()={} != product of original_shape={original_shape:?} ({total_elems})",
            x.len()
        )));
    }
    let d = original_shape[3] as usize;
    if !d.is_multiple_of(GROUP_SIZE) {
        return Err(Error::Quant(format!(
            "tcq: last dimension D={d} must be a multiple of GROUP_SIZE={GROUP_SIZE}"
        )));
    }

    let num_levels = codebook.len();
    // bits = log2(num_levels): 4 levels → 2, 8 levels → 3.
    // Codebook length is validated by lloyd_gaussian_codebook / caller.
    let bits: u8 = match num_levels {
        4 => 2,
        8 => 3,
        n => {
            return Err(Error::Quant(format!(
                "tcq_encode_with_codebook: unsupported codebook length {n}; expected 4 (2-bit) or 8 (3-bit)"
            )));
        }
    };
    let max_centroid = codebook
        .iter()
        .copied()
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));

    let transitions = build_transition_table(num_levels);

    let n_blocks = total_elems / GROUP_SIZE;
    let bits_per_block = GROUP_SIZE * bits as usize;
    let bytes_per_block = bits_per_block.div_ceil(8);
    let mut codes_bytes = vec![0u8; n_blocks * bytes_per_block];
    let mut scales = vec![0.0_f32; n_blocks];

    // Reusable scratch buffers (per-block): forward path cost + back-pointer
    // table + chosen indices. Allocating once outside the loop keeps the inner
    // pass branchless and avoids re-malloc per group.
    let mut path_cost = [f32::INFINITY; TCQ_NUM_STATES];
    let mut next_cost = [f32::INFINITY; TCQ_NUM_STATES];
    // back_states[t * NUM_STATES + state] = predecessor state at step t-1.
    // back_levels[t * NUM_STATES + state] = chosen level at step t that landed in `state`.
    let mut back_states = [0u8; GROUP_SIZE * TCQ_NUM_STATES];
    let mut back_levels = [0u8; GROUP_SIZE * TCQ_NUM_STATES];
    let mut indices = [0u8; GROUP_SIZE];

    for (block, scale_slot) in scales.iter_mut().enumerate() {
        let start = block * GROUP_SIZE;
        let group = &x[start..start + GROUP_SIZE];

        // Per-group scale: identical to plain turboquant.
        let abs_max = group
            .iter()
            .copied()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let scale = if abs_max > 0.0 {
            abs_max / max_centroid
        } else {
            0.0
        };
        *scale_slot = scale;

        // Zero-scale block: every centroid is equidistant (all zero); emit
        // all-zero indices (no Viterbi needed). Matches turbo_quantize_v.
        let byte_offset = block * bytes_per_block;
        let block_buf = &mut codes_bytes[byte_offset..byte_offset + bytes_per_block];

        if scale == 0.0 {
            // All zeros (Vec initialised to 0); nothing to pack.
            continue;
        }

        let inv_scale = 1.0_f32 / scale;

        // ── Viterbi forward pass ────────────────────────────────────────────
        //
        // path_cost[state] = minimum cumulative cost to land in `state` after
        // the current step. Start in state 0 (canonical initial state) with
        // cost 0; all other states are unreachable initially (+inf).
        path_cost.fill(f32::INFINITY);
        path_cost[0] = 0.0;

        for t in 0..GROUP_SIZE {
            let normalised = group[t] * inv_scale;

            next_cost.fill(f32::INFINITY);

            // For each (prev_state, level), compute candidate next_state.
            // Keep the (prev_state, level) that minimises arrival cost.
            for prev_state in 0..TCQ_NUM_STATES {
                let prev_cost = path_cost[prev_state];
                if prev_cost.is_infinite() {
                    continue;
                }
                for level in 0..num_levels {
                    let next_state = transitions[prev_state * num_levels + level] as usize;
                    let diff = normalised - codebook[level];
                    let cand = prev_cost + diff * diff;
                    if cand < next_cost[next_state] {
                        next_cost[next_state] = cand;
                        back_states[t * TCQ_NUM_STATES + next_state] = prev_state as u8;
                        back_levels[t * TCQ_NUM_STATES + next_state] = level as u8;
                    }
                }
            }

            path_cost.copy_from_slice(&next_cost);
        }

        // ── Back-trace ──────────────────────────────────────────────────────
        //
        // Best final state minimises total path cost.
        let mut best_state = 0usize;
        let mut best_cost = path_cost[0];
        for (s, &c) in path_cost.iter().enumerate().skip(1) {
            if c < best_cost {
                best_cost = c;
                best_state = s;
            }
        }

        // Walk back from t = GROUP_SIZE-1 to 0.
        let mut cur_state = best_state;
        for t in (0..GROUP_SIZE).rev() {
            let level = back_levels[t * TCQ_NUM_STATES + cur_state];
            indices[t] = level;
            cur_state = back_states[t * TCQ_NUM_STATES + cur_state] as usize;
        }

        // ── Pack indices ────────────────────────────────────────────────────
        for (elem, &idx) in indices.iter().enumerate() {
            pack_index(block_buf, elem, idx, bits);
        }
    }

    Ok(TurboBlocks {
        codes: codes_bytes,
        scales,
        original_shape: shape4,
        bits,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tcq_tests.rs"]
mod tcq_tests;
