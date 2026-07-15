//! Affine `bits ∈ {2,3,4,5,6,8}` × `group_size ∈ {32,64,128}` dequant.
//!
//! Ground truth: `docs/03-mlx-safetensors-format.md` §Dequantization formulas.
//!
//! Dequant formula (additive bias):
//! w_fp = scale * code + bias
//!
//! Bias convention: ADDITIVE. The doc (`docs/03-mlx-safetensors-format.md`,
//! §Dequantization formulas, §Affine) reads `w_fp = s * x_q + b`.
//! The AWQ→MLX note in the same section explains `b = -zero_point * scale`,
//! so the net effect is identical to `s*(x_q - zp)` — but the stored bf16 bias
//! already encodes the sign. rMLX stores and applies it as-is (additive).
//!
//! Stage 1.

use std::sync::OnceLock;

use rmlx_core::{Error, Result};
use tracing::warn;

use crate::bf16::bf16_to_f32;

// ── CodeStorage ──────────────────────────────────────────────────────────────

/// Describes how quantized codes are packed in the on-disk weight tensor.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two affine code storage layouts (U8/U32Le); adding a layout requires updating dequant_group and all CodeStorage match arms"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeStorage {
    /// Codes packed LSB-first into u8 bytes.
    ///
    /// Not observed in production MLX snapshots (all real snapshots use U32Le),
    /// but required by the spec for completeness and round-trip testing.
    U8,
    /// Codes packed LSB-first into u32 little-endian words.
    ///
    /// This is what every real MLX affine snapshot uses.
    /// `per_word = 32 / bits`. For 3-bit elements: 10 per u32, last 2 bits
    /// of each u32 are unused padding.
    U32Le,
}

// ── AffineParams ─────────────────────────────────────────────────────────────

/// Parameters that describe one affine-quantized weight tensor.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed params struct — five fields are the complete affine quant descriptor contract; adding a field requires updating AffineParams construction in the loader and all dequant callers"
)]
#[derive(Debug, Clone)]
pub struct AffineParams {
    /// Quantization bit-width. Must be in `{2, 3, 4, 5, 6, 8}`.
    pub bits: u8,
    /// Group size. Must be in `{32, 64, 128}`.
    pub group_size: u32,
    /// How codes are packed in `packed_codes`.
    pub storage: CodeStorage,
    /// Number of rows (output channels).
    pub rows: usize,
    /// Number of columns (input channels, original, unpacked).
    pub cols: usize,
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Affine bit-widths this build's dequant codec supports — CPU (`dequant_to_f32`
/// below) and GPU (mlx-c `affine_dequantize_*` / `quantized_matmul` kernels)
/// alike. Single source of truth: `validate_params` and any load-time
/// pre-flight check (e.g. `rmlx_models::arch::loader`) must both read this
/// constant rather than re-deriving the set.
pub const SUPPORTED_BITS: [u8; 6] = [2, 3, 4, 5, 6, 8];

/// Cold helper: construct the "bits not supported" error.
///
/// Marked `#[cold]` so LLVM places this large `format!` block away from the hot
/// prelude in `validate_params`. The error arm fires at most once per model load.
#[cold]
fn err_bits(b: u8) -> Error {
    Error::Quant(format!(
        "affine: bits={b} not supported; must be one of {{2,3,4,5,6,8}}"
    ))
}

/// Cold helper: construct the "group_size not supported" error.
#[cold]
fn err_group_size(g: u32) -> Error {
    Error::Quant(format!(
        "affine: group_size={g} not supported; must be one of {{32,64,128}}"
    ))
}

/// Cold helper: construct the "cols not multiple of group_size" error.
#[cold]
fn err_cols_not_multiple(cols: usize, group_size: u32) -> Error {
    Error::Quant(format!(
        "affine: cols={cols} is not a multiple of group_size={group_size}"
    ))
}

/// Cold helper: construct the "packed_codes length mismatch" error.
#[cold]
fn err_packed_len(
    got: usize,
    exp: usize,
    rows: usize,
    cols: usize,
    bits: u8,
    storage: CodeStorage,
) -> Error {
    Error::Quant(format!(
        "affine: packed_codes length {got} != expected {exp} \
         (rows={rows}, cols={cols}, bits={bits}, storage={storage:?})"
    ))
}

/// Cold helper: construct the "scales_bf16 length mismatch" error.
#[cold]
fn err_scales_len(got: usize, exp: usize, rows: usize, cols: usize, group_size: u32) -> Error {
    Error::Quant(format!(
        "affine: scales_bf16 length {got} != expected {exp} \
         (rows={rows}, cols={cols}, group_size={group_size})"
    ))
}

/// Cold helper: construct the "biases_bf16 length mismatch" error.
#[cold]
fn err_biases_len(got: usize, exp: usize, rows: usize, cols: usize, group_size: u32) -> Error {
    Error::Quant(format!(
        "affine: biases_bf16 length {got} != expected {exp} \
         (rows={rows}, cols={cols}, group_size={group_size})"
    ))
}

/// Cold helper: construct the "out length mismatch" error.
#[cold]
fn err_out_len(got: usize, exp: usize, rows: usize, cols: usize) -> Error {
    Error::Quant(format!(
        "affine: out length {got} != expected {exp} (rows={rows}, cols={cols})"
    ))
}

fn validate_params(params: &AffineParams) -> Result<()> {
    if !SUPPORTED_BITS.contains(&params.bits) {
        return Err(err_bits(params.bits));
    }
    match params.group_size {
        32 | 64 | 128 => {}
        g => return Err(err_group_size(g)),
    }
    if !params.cols.is_multiple_of(params.group_size as usize) {
        return Err(err_cols_not_multiple(params.cols, params.group_size));
    }
    Ok(())
}

// ── Packed-code length helpers ────────────────────────────────────────────────

/// Number of `u8` bytes needed to hold `rows * cols * bits` packed bits.
fn packed_u8_len(params: &AffineParams) -> usize {
    let total_bits = params.rows * params.cols * params.bits as usize;
    total_bits.div_ceil(8)
}

/// Number of `u8` bytes needed when using U32Le storage.
fn packed_u32_len_bytes(params: &AffineParams) -> usize {
    // per_word = 32 / bits (floor). Total u32 words = ceil(cols / per_word) * rows.
    let per_word = 32 / params.bits as usize;
    let total_words = params.rows * params.cols.div_ceil(per_word);
    total_words * 4 // each u32 = 4 bytes
}

/// Expected byte length of `packed_codes` for a given storage type.
fn expected_packed_len(params: &AffineParams) -> usize {
    match params.storage {
        CodeStorage::U8 => packed_u8_len(params),
        CodeStorage::U32Le => packed_u32_len_bytes(params),
    }
}

/// Expected byte length of `scales_bf16` (or `biases_bf16`).
///
/// shape `[rows, cols / group_size]` × 2 bytes each.
fn expected_sb_len(params: &AffineParams) -> usize {
    let groups_per_row = params.cols / params.group_size as usize;
    params.rows * groups_per_row * 2
}

// ── Bit-unpacking ─────────────────────────────────────────────────────────────

/// Extract the quantized code at position `(row, col)` from U8-packed storage.
///
/// Codes are packed LSB-first across byte boundaries.
/// The global bit-offset of element `(r, c)` is `(r * cols + c) * bits`.
#[inline]
fn unpack_u8(packed: &[u8], row: usize, col: usize, cols: usize, bits: usize) -> u32 {
    let global_bit = (row * cols + col) * bits;
    let byte_idx = global_bit / 8;
    let bit_shift = global_bit % 8;
    let mask = (1u32 << bits) - 1;

    // We may straddle 1–3 bytes. Read up to 3 bytes (at most 8+8+8=24 bits
    // available, sufficient for bits ≤ 8).
    // byte_idx = (r*cols+c)*bits/8; caller ensures (r,c) is within the packed
    // buffer (rows*cols*bits/8 bytes total), so byte_idx < packed.len().
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_idx < packed.len(): global_bit = (r*cols+c)*bits < rows*cols*bits = packed.len()*8"
    )]
    let b0 = u32::from(packed[byte_idx]);
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_idx+1 < packed.len() is checked by the surrounding if condition"
    )]
    let b1 = if byte_idx + 1 < packed.len() {
        u32::from(packed[byte_idx + 1])
    } else {
        0
    };
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_idx+2 < packed.len() is checked by the surrounding if condition"
    )]
    let b2 = if byte_idx + 2 < packed.len() {
        u32::from(packed[byte_idx + 2])
    } else {
        0
    };

    let window = b0 | (b1 << 8) | (b2 << 16);
    (window >> bit_shift) & mask
}

/// Extract the quantized code at position `(row, col)` from U32Le-packed storage.
///
/// Codes are packed LSB-first within each u32 word.
/// `per_word = 32 / bits`.
/// Word index for column `c` in row `r`: `r * words_per_row + c / per_word`.
#[inline]
fn unpack_u32le(packed: &[u8], row: usize, col: usize, cols: usize, bits: usize) -> u32 {
    let per_word = 32 / bits;
    let words_per_row = cols.div_ceil(per_word);
    let word_idx = row * words_per_row + col / per_word;
    let shift = (col % per_word) * bits;
    let mask = (1u32 << bits) - 1;

    // Read u32 LE from 4 bytes.
    // base = word_idx*4; caller validates packed.len() == rows*words_per_row*4,
    // and word_idx < rows*words_per_row, so base+3 < packed.len().
    #[allow(
        clippy::indexing_slicing,
        reason = "base = word_idx*4 < packed.len() - 3: packed holds rows*words_per_row u32 words; word_idx < rows*words_per_row is proven by (r,c) bounds"
    )]
    let base = word_idx * 4;
    #[allow(
        clippy::indexing_slicing,
        reason = "base+3 < packed.len(): see above — packed.len() == rows*words_per_row*4 and base = word_idx*4 where word_idx < rows*words_per_row"
    )]
    let word = u32::from_le_bytes([
        packed[base],
        packed[base + 1],
        packed[base + 2],
        packed[base + 3],
    ]);
    (word >> shift) & mask
}

// ── Bias-sign warning (fire once per process) ─────────────────────────────────

static BIAS_SIGN_WARNED: OnceLock<()> = OnceLock::new();

/// Cold helper: emit the bias-sign convention warning at most once per process.
///
/// The `OnceLock` gate means this fires at most once across all model loads.
/// `#[cold]` keeps the large `warn!` macro expansion off the hot code path in
/// `dequant_to_f32`.
#[cold]
fn warn_bias_sign_once() {
    BIAS_SIGN_WARNED.get_or_init(|| {
        warn!(
            "affine dequant: using ADDITIVE bias convention (w = scale * code + bias) \
             as specified in docs/03-mlx-safetensors-format.md. \
             Some literature uses subtractive bias; rMLX follows the doc."
        );
    });
}

// ── Core dequant ─────────────────────────────────────────────────────────────

/// Dequantize the entire packed weight into a row-major f32 buffer.
///
/// Dequant formula: `w_fp = scale * code + bias`
///
/// Bias is ADDITIVE per `docs/03-mlx-safetensors-format.md`.
/// The AWQ→MLX conversion stores `bias = -zero_point * scale`, so the stored
/// bf16 bias value already carries the correct sign.
///
/// # Input shapes
/// - `packed_codes`: see `CodeStorage`; total encoded bits = `rows * cols * bits`.
/// - `scales_bf16` / `biases_bf16`: `2 * rows * (cols / group_size)` bytes each,
///   row-major bf16 LE.
///
/// # Output
/// `out`: `rows * cols` f32 elements, row-major.
///
/// # Errors
/// Returns `Error::Quant(_)` on any shape mismatch or unsupported parameter.
pub fn dequant_to_f32(
    params: &AffineParams,
    packed_codes: &[u8],
    scales_bf16: &[u8],
    biases_bf16: &[u8],
    out: &mut [f32],
) -> Result<()> {
    validate_params(params)?;

    let exp_packed = expected_packed_len(params);
    if packed_codes.len() != exp_packed {
        return Err(err_packed_len(
            packed_codes.len(),
            exp_packed,
            params.rows,
            params.cols,
            params.bits,
            params.storage,
        ));
    }

    let exp_sb = expected_sb_len(params);
    if scales_bf16.len() != exp_sb {
        return Err(err_scales_len(
            scales_bf16.len(),
            exp_sb,
            params.rows,
            params.cols,
            params.group_size,
        ));
    }
    if biases_bf16.len() != exp_sb {
        return Err(err_biases_len(
            biases_bf16.len(),
            exp_sb,
            params.rows,
            params.cols,
            params.group_size,
        ));
    }

    let exp_out = params.rows * params.cols;
    if out.len() != exp_out {
        return Err(err_out_len(out.len(), exp_out, params.rows, params.cols));
    }

    warn_bias_sign_once();

    let bits = params.bits as usize;
    let group_size = params.group_size as usize;
    let groups_per_row = params.cols / group_size;

    // Walk rows via chunks_exact_mut; within each row walk groups of group_size.
    // This lifts the `c / group_size` divide and the `r * cols + c` multiply out
    // of the innermost loop — both become pointer advances through chunks_exact_mut.
    // Scale/bias are loaded once per group (as before), not once per element.
    for (r, out_row) in out.chunks_exact_mut(params.cols).enumerate() {
        let sb_row_off = r * groups_per_row * 2;
        // Slice exactly groups_per_row bf16 pairs (2 bytes each) for this row.
        // Validated: scales_bf16.len() == rows * groups_per_row * 2 (checked at fn entry).
        #[allow(
            clippy::indexing_slicing,
            reason = "sb_row_off + groups_per_row*2 <= scales_bf16.len(): scales has rows*groups_per_row*2 bytes; r < rows from chunks_exact_mut"
        )]
        let scales_row = &scales_bf16[sb_row_off..sb_row_off + groups_per_row * 2];
        #[allow(
            clippy::indexing_slicing,
            reason = "sb_row_off + groups_per_row*2 <= biases_bf16.len(): biases has same layout as scales"
        )]
        let biases_row = &biases_bf16[sb_row_off..sb_row_off + groups_per_row * 2];

        for (g, out_group) in out_row.chunks_exact_mut(group_size).enumerate() {
            let sb = g * 2;
            // g < groups_per_row (chunks_exact_mut produces exactly groups_per_row chunks),
            // so sb+1 = g*2+1 < groups_per_row*2 == scales_row.len().
            #[allow(
                clippy::indexing_slicing,
                reason = "sb+1 < scales_row.len(): g < groups_per_row from chunks_exact_mut, so g*2+1 < groups_per_row*2"
            )]
            let scale = bf16_to_f32([scales_row[sb], scales_row[sb + 1]]);
            #[allow(
                clippy::indexing_slicing,
                reason = "sb+1 < biases_row.len(): same invariant as scales_row above"
            )]
            let bias = bf16_to_f32([biases_row[sb], biases_row[sb + 1]]);

            let col_start = g * group_size;
            for (c_local, slot) in out_group.iter_mut().enumerate() {
                let c = col_start + c_local;

                // Unpack code — still uses (r, c) addressing; column arithmetic
                // is now just a constant offset per group rather than a full divide.
                let code = match params.storage {
                    CodeStorage::U8 => unpack_u8(packed_codes, r, c, params.cols, bits),
                    CodeStorage::U32Le => unpack_u32le(packed_codes, r, c, params.cols, bits),
                };

                // Dequant: additive bias (doc ground truth).
                *slot = scale.mul_add(code as f32, bias);
            }
        }
    }

    Ok(())
}

/// Convenience wrapper: allocates and returns a fresh `Vec<f32>`.
///
/// Allocates exactly once (`rows * cols` elements).
pub fn dequant_vec(
    params: &AffineParams,
    packed_codes: &[u8],
    scales_bf16: &[u8],
    biases_bf16: &[u8],
) -> Result<Vec<f32>> {
    let mut out = vec![0.0_f32; params.rows * params.cols];
    dequant_to_f32(params, packed_codes, scales_bf16, biases_bf16, &mut out)?;
    Ok(out)
}

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Pack a slice of `u32` codes into bytes using the given storage scheme.
///
/// `codes` must have exactly `rows * cols` elements.
/// Used only in tests to construct round-trip inputs without real snapshots.
#[cfg(any(test, doc))]
pub fn pack_codes_for_test(codes: &[u32], params: &AffineParams) -> Vec<u8> {
    let bits = params.bits as usize;
    match params.storage {
        CodeStorage::U8 => pack_u8(codes, params.rows, params.cols, bits),
        CodeStorage::U32Le => pack_u32le(codes, params.rows, params.cols, bits),
    }
}

#[cfg(any(test, doc))]
fn pack_u8(codes: &[u32], rows: usize, cols: usize, bits: usize) -> Vec<u8> {
    let total_bits = rows * cols * bits;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    for r in 0..rows {
        for c in 0..cols {
            let code = codes[r * cols + c];
            let global_bit = (r * cols + c) * bits;
            let byte_idx = global_bit / 8;
            let bit_shift = global_bit % 8;
            // Write up to bits bits starting at bit_shift within byte_idx.
            // May span 2–3 bytes.
            let mut remaining = bits;
            let mut shift = bit_shift;
            let mut idx = byte_idx;
            let mut val = code;
            while remaining > 0 {
                let take = remaining.min(8 - shift);
                let mask = ((1u32 << take) - 1) as u8;
                out[idx] |= ((val as u8) & mask) << shift;
                val >>= take;
                remaining -= take;
                shift = 0;
                idx += 1;
            }
        }
    }
    out
}

#[cfg(any(test, doc))]
fn pack_u32le(codes: &[u32], rows: usize, cols: usize, bits: usize) -> Vec<u8> {
    let per_word = 32 / bits;
    let words_per_row = cols.div_ceil(per_word);
    let total_words = rows * words_per_row;
    let mut words = vec![0u32; total_words];
    for r in 0..rows {
        for c in 0..cols {
            let code = codes[r * cols + c];
            let word_idx = r * words_per_row + c / per_word;
            let shift = (c % per_word) * bits;
            words[word_idx] |= code << shift;
        }
    }
    // Convert words to LE bytes.
    let mut out = Vec::with_capacity(total_words * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "affine_tests.rs"]
mod tests;
