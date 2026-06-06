//! TurboQuant KV-cache V primitives — Lloyd-Max codebook.
//!
//! # What this is
//!
//! Scalar (CPU/Rust) quantize + dequantize for the V tensor in
//! TurboQuant KV-cache compression. No MSL kernels yet — that is an S2.5
//! optimization. This module is the correctness reference; the Metal port
//! will produce bit-identical output.
//!
//! # Codebook — Lloyd-Max, N(0,1)
//!
//! The codebase previously used **N(0,1) quantile
//! centroids** — hardcoded values from the Python fork's dead
//! `_compute_gaussian_codebook` function. The current codebook uses the
//! correct **Lloyd-Max optimal centroids for N(0,1)**.
//!
//! Derivation: `turboquant_plus/turboquant/codebook.py::_lloyds_gaussian`
//! with `sigma=1.0` (standard normal), 100 Lloyd iterations, f32 output.
//! Source commit: `turboquant_plus` main branch, function `optimal_centroids`.
//!
//! The values are *pre-computed* at build time and hardcoded here (no runtime
//! Python dependency). The derivation script is at
//! `scripts/gen_lloyd_codebook.py` (call `python3 scripts/gen_lloyd_codebook.py`
//! to regenerate if the target distribution changes).
//!
//! ## Why dimension-independent
//!
//! The quantization algorithm normalises each group by `scale = max(|x_i|) /
//! max_centroid` before nearest-centroid lookup. For a symmetric codebook the
//! optimal assignment is determined by *ratios* between centroids, which are
//! invariant to the global sigma. N(0,1) Lloyd-Max therefore gives the same
//! assignment quality as N(0,1/d) Lloyd-Max once the scale step is applied.
//!
//! Codebook values at bits ∈ {1, 2, 3, 4}. Metal kernels in
//! `turboquant_msl.rs` embed the same constants via `as_type<float>(0x...)` for
//! bit-exact GPU parity.
//!
//! # Supported widths
//!
//! | bits | use |
//! |------|-----|
//! | 1 | experimental |
//! | 2 | experimental |
//! | 3 | production |
//! | 4 | production (primary V path, K8+V4 asymmetric) |
//! | 8 | **not supported here** — use `affine` q8_0 for K8 |
//!
//! # Block layout
//!
//! Input tensor is treated as a flat sequence of elements. Elements are
//! partitioned into non-overlapping groups of `GROUP_SIZE = 32`. Each group
//! gets one f32 scale factor. Indices are bit-packed LSB-first into u8 bytes.
//!
//! This matches the MLX `group_size` convention used throughout the project.
//!
//! # K8 note
//!
//! For the asymmetric K8/V4 cache (the mandatory default for Qwen MoE per
//! CLAUDE.md), K-cache uses standard affine 8-bit quantization with
//! `group_size=128` (implemented in `crates/rmlx-quant/src/affine.rs`). Only
//! V uses this module.

use rmlx_core::{Error, Result};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Block (group) size: number of elements per quantization group.
///
/// 32 matches the MLX `group_size` convention used by weight quantization.
pub const GROUP_SIZE: usize = 32;

// Lloyd-Max optimal centroids for N(0,1) at bits ∈ {1, 2, 3, 4}.
//
// Derived by: `turboquant_plus/turboquant/codebook.py::_lloyds_gaussian`
// with sigma=1.0, n_iter=100. Hardcoded as f32 (no runtime Python dep).
// Regenerate via `python3 scripts/gen_lloyd_codebook.py` if needed.
//
// Prior codebook: N(0,1) *quantile* centroids (dead code path from
// mlx-lm-turboquant `_compute_gaussian_codebook`). Replaced by true Lloyd-Max.

const CODEBOOK_1BIT: [f32; 2] = [-0.797_884_6, 0.797_884_6];

const CODEBOOK_2BIT: [f32; 4] = [-1.51, -0.453, 0.453, 1.51];

const CODEBOOK_3BIT: [f32; 8] = [
    -2.151_944_9,
    -1.343_908_5,
    -0.756_004_75,
    -0.245_093_99,
    0.245_093_99,
    0.756_004_75,
    1.343_908_5,
    2.151_944_9,
];

const CODEBOOK_4BIT: [f32; 16] = [
    -2.717_667,
    -2.052_138,
    -1.600_802_4,
    -1.239_959,
    -0.928_244_7,
    -0.645_875_33,
    -0.381_178_23,
    -0.126_046_94,
    0.126_046_94,
    0.381_178_23,
    0.645_875_33,
    0.928_244_7,
    1.239_959,
    1.600_802_4,
    2.052_138,
    2.717_667,
];

// ── Public API ────────────────────────────────────────────────────────────────

/// Compact representation of a TurboQuant-quantized tensor.
///
/// Blocks are `GROUP_SIZE = 32` element groups along the last axis.
/// Each block has one f32 scale and `GROUP_SIZE * bits` packed bits of indices.
///
/// `codes` is a `Vec<u8>` with indices bit-packed LSB-first:
/// `GROUP_SIZE * bits` bits per block, ceil'd to whole bytes, concatenated.
///
/// `scales` is a flat `Vec<f32>` with one scale per block.
///
/// `original_shape` stores the original `[B, kv_h, S, D]` dimensions so
/// `turbo_dequantize` can restore the exact shape.
///
/// `bits` is in `{1, 2, 3, 4}`. 8-bit is not supported — use affine q8_0.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed quantized-tensor struct — four fields are the complete TurboQuant block-storage contract; adding a field requires updating turbo_quantize and all dequant callers"
)]
#[derive(Debug, Clone)]
pub struct TurboBlocks {
    /// Bit-packed quantized indices (LSB-first, `bits` bits per element,
    /// `GROUP_SIZE` elements per block).
    pub codes: Vec<u8>,
    /// Per-block scale factors: `max(|x_group|) / max_centroid`.
    pub scales: Vec<f32>,
    /// Original tensor shape `[B, kv_h, S, D]` as signed i32 (MLX convention).
    ///
    /// Fixed-size array (always 4-D): 16 B inline vs 24 B stack + 16 B heap for Vec<i32>.
    pub original_shape: [i32; 4],
    /// Bit-width. Must be in `{1, 2, 3, 4}`.
    pub bits: u8,
}

/// Cold helper: "bits=8 not supported — use affine q8_0" error.
///
/// This arm fires at validation time only (loader or parameter-check call),
/// never on the decode steady-state path.
#[cold]
fn err_bits8() -> Error {
    Error::Quant(
        "turboquant: bits=8 is not supported — use affine q8_0 (group_size=128) \
         for K8 quantization. This matches the mandatory asymmetric K8/V4 design."
            .to_owned(),
    )
}

/// Cold helper: "bits not supported" error for arbitrary unsupported widths.
#[cold]
fn err_bits_unsupported(b: u8) -> Error {
    Error::Quant(format!(
        "turboquant: bits={b} not supported; must be one of {{1, 2, 3, 4}}"
    ))
}

/// Return the Lloyd-Max optimal N(0,1) centroid codebook for `bits`.
///
/// # Codebook
///
/// Codebook values are **Lloyd-Max optimal centroids for N(0,1)** — derived by
/// `turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0,
/// n_iter=100)` and hardcoded as f32.
///
/// The prior codebase used N(0,1) *quantile* centroids (the Python
/// fork's `_compute_gaussian_codebook` dead-code path). The current values are
/// MSE-optimal for scalar quantization of a standard normal source.
///
/// Derivation script: `scripts/gen_lloyd_codebook.py`.
///
/// # Performance note
///
/// Returns `&'static [f32]` — a direct reference into the compile-time constant.
/// No heap allocation on the hot per-token quantize/dequantize path.
///
/// # Errors
///
/// Returns `Error::Quant` for `bits == 8` (use `affine::dequant_to_f32` with
/// `group_size=128` for K8 quantization) or any other unsupported width.
pub fn lloyd_gaussian_codebook(bits: u8) -> Result<&'static [f32]> {
    match bits {
        1 => Ok(&CODEBOOK_1BIT),
        2 => Ok(&CODEBOOK_2BIT),
        3 => Ok(&CODEBOOK_3BIT),
        4 => Ok(&CODEBOOK_4BIT),
        8 => Err(err_bits8()),
        b => Err(err_bits_unsupported(b)),
    }
}

/// Cold helper: "original_shape must have 4 elements" error.
#[cold]
fn err_shape_rank(got: usize) -> Error {
    Error::Quant(format!(
        "turboquant: original_shape must have 4 elements [B, kv_h, S, D], \
         got {got} element(s)"
    ))
}

/// Cold helper: "all shape dimensions must be positive" error.
#[cold]
fn err_shape_nonpositive(shape: &[i32]) -> Error {
    Error::Quant(format!(
        "turboquant: all shape dimensions must be positive, got {shape:?}"
    ))
}

/// Cold helper: "x.len() != product of original_shape" error.
#[cold]
fn err_len_mismatch(got: usize, shape: &[i32], expected: usize) -> Error {
    Error::Quant(format!(
        "turboquant: x.len()={got} != product of original_shape={shape:?} ({expected})"
    ))
}

/// Cold helper: "last dimension D must be multiple of GROUP_SIZE" error.
#[cold]
fn err_d_not_multiple(d: usize) -> Error {
    Error::Quant(format!(
        "turboquant: last dimension D={d} must be a multiple of GROUP_SIZE={GROUP_SIZE}"
    ))
}

/// Cold helper: "scales.len() != n_blocks" error from turbo_dequantize.
#[cold]
fn err_scales_count(got: usize, expected: usize) -> Error {
    Error::Quant(format!(
        "turboquant: scales.len()={got} != n_blocks={expected}"
    ))
}

/// Cold helper: "codes.len() != expected" error from turbo_dequantize.
#[cold]
fn err_codes_len(got: usize, expected: usize, n_blocks: usize, bytes_per_block: usize) -> Error {
    Error::Quant(format!(
        "turboquant: codes.len()={got} != expected {expected} \
         (n_blocks={n_blocks}, bytes_per_block={bytes_per_block})"
    ))
}

/// Quantize a V tensor at `bits`-bit resolution using the Lloyd-Max N(0,1) codebook.
///
/// This is the default path (no override). For a per-layer codebook override use
/// [`turbo_quantize_v_with_codebook`].
///
/// # Input
///
/// `x` must be a flat, row-major f32 slice representing a tensor of shape
/// `original_shape = [B, kv_h, S, D]` (4 dimensions, product = `x.len()`).
/// `original_shape` must have exactly 4 elements.
///
/// # Algorithm
///
/// 1. Partition elements into non-overlapping groups of `GROUP_SIZE = 32`.
/// 2. Per group: compute `scale = max(|x_i|) / max_centroid`.
///    If all elements are zero, scale is 0 and all codes are the middle index.
/// 3. Normalize group elements by `scale` (or leave at 0 if scale is 0).
/// 4. Assign each normalized element to the nearest centroid (nearest-neighbor
///    via boundary comparisons — same algorithm as the Metal kernel).
/// 5. Bit-pack indices LSB-first into u8 bytes.
///
/// # K8 note
///
/// Do **not** call this function for K-cache. Use `affine::dequant_to_f32`
/// with `group_size=128, bits=8` (q8_0 affine) instead.
///
/// # Errors
///
/// Returns `Error::Quant` if `bits` is unsupported, `original_shape` does not
/// have 4 elements, `x.len()` does not equal the product of `original_shape`,
/// or `D` (last dimension) is not a multiple of `GROUP_SIZE`.
pub fn turbo_quantize_v(x: &[f32], bits: u8, original_shape: &[i32]) -> Result<TurboBlocks> {
    turbo_quantize_v_with_codebook(x, bits, original_shape, None)
}

/// Quantize a V tensor at `bits`-bit resolution with an optional codebook override.
///
/// When `codebook_override` is `None`, uses the built-in Lloyd-Max N(0,1) codebook
/// (identical to [`turbo_quantize_v`]). When `Some(cb)`, uses `cb` as the centroid
/// table. `cb.len()` must equal `2^bits`; otherwise returns `Error::Quant`.
///
/// # GPU dispatch note
///
/// This function is always CPU scalar. The GPU MSL path (`turbo_quantize_v4_gpu`)
/// has the Lloyd-Max codebook hardwired in MSL source. When `codebook_override` is
/// `Some`, callers **must** route to this function (CPU) for that layer.
/// T19b will add an MSL variant that accepts a codebook buffer arg.
///
/// # Errors
///
/// Returns `Error::Quant` for:
/// - unsupported `bits` (same as `turbo_quantize_v`),
/// - wrong override length (`cb.len() != 2^bits`),
/// - invalid `original_shape` (same as `turbo_quantize_v`).
pub fn turbo_quantize_v_with_codebook(
    x: &[f32],
    bits: u8,
    original_shape: &[i32],
    codebook_override: Option<&[f32]>,
) -> Result<TurboBlocks> {
    // -- Validate bits ---------------------------------------------------------
    let builtin = lloyd_gaussian_codebook(bits)?;
    let codebook: &[f32] = if let Some(cb) = codebook_override {
        let expected_len = 1usize << bits;
        if cb.len() != expected_len {
            return Err(Error::Quant(format!(
                "turbo_quantize_v_with_codebook: codebook_override.len()={} must equal \
                 2^bits={} for bits={}; caller bug",
                cb.len(),
                expected_len,
                bits,
            )));
        }
        if !cb.windows(2).all(|w| matches!(w, [a, b] if a < b)) {
            return Err(Error::Quant(format!(
                "turbo_quantize_v_with_codebook: codebook must be strictly ascending; got {cb:?}"
            )));
        }
        cb
    } else {
        builtin
    };
    let max_centroid = codebook
        .iter()
        .copied()
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));

    // -- Validate shape --------------------------------------------------------
    if original_shape.len() != 4 {
        return Err(err_shape_rank(original_shape.len()));
    }
    if original_shape.iter().any(|&d| d <= 0) {
        return Err(err_shape_nonpositive(original_shape));
    }
    // original_shape.len() == 4 is checked above (err_shape_rank guard).
    #[allow(
        clippy::indexing_slicing,
        reason = "original_shape.len()==4 validated by the err_shape_rank guard above"
    )]
    let shape4: [i32; 4] = [
        original_shape[0],
        original_shape[1],
        original_shape[2],
        original_shape[3],
    ];
    let total_elems: usize = original_shape.iter().map(|&d| d as usize).product();
    if x.len() != total_elems {
        return Err(err_len_mismatch(x.len(), original_shape, total_elems));
    }
    // original_shape.len()==4 verified above; index 3 is valid.
    #[allow(
        clippy::indexing_slicing,
        reason = "original_shape.len()==4 validated by err_shape_rank guard; index 3 is always valid"
    )]
    let d = original_shape[3] as usize;
    if !d.is_multiple_of(GROUP_SIZE) {
        return Err(err_d_not_multiple(d));
    }

    // -- Quantize --------------------------------------------------------------
    let n_blocks = total_elems / GROUP_SIZE;
    let bits_per_block = GROUP_SIZE * bits as usize;
    let bytes_per_block = bits_per_block.div_ceil(8);
    let mut codes_bytes = vec![0u8; n_blocks * bytes_per_block];
    let mut scales = vec![0.0_f32; n_blocks];

    for (block, scale_slot) in scales.iter_mut().enumerate() {
        let start = block * GROUP_SIZE;
        // block < n_blocks == total_elems/GROUP_SIZE; x.len()==total_elems (validated above),
        // so start + GROUP_SIZE <= total_elems == x.len().
        #[allow(
            clippy::indexing_slicing,
            reason = "start+GROUP_SIZE <= x.len(): block < n_blocks=total_elems/GROUP_SIZE; x.len()==total_elems validated above"
        )]
        let group = &x[start..start + GROUP_SIZE];

        // Compute scale: max(|x|) / max_centroid.
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

        // Quantize each element.
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let byte_offset = block * bytes_per_block;
        // byte_offset + bytes_per_block <= codes_bytes.len(): codes_bytes.len()==n_blocks*bytes_per_block; block < n_blocks.
        #[allow(
            clippy::indexing_slicing,
            reason = "byte_offset+bytes_per_block <= codes_bytes.len(): codes_bytes.len()==n_blocks*bytes_per_block; block < n_blocks from iter"
        )]
        let block_buf = &mut codes_bytes[byte_offset..byte_offset + bytes_per_block];

        for (elem, &val) in group.iter().enumerate() {
            let normalized = val * inv_scale;
            let idx = nearest_centroid(normalized, codebook) as u8;
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

/// Dequantize a [`TurboBlocks`] back to f32 using the built-in Lloyd-Max codebook.
///
/// This is a thin delegate to [`turbo_dequantize_with_codebook`] with `None`.
///
/// Output is a flat row-major f32 `Vec` of shape `blocks.original_shape`.
///
/// # Errors
///
/// Returns `Error::Quant` if the blocks are internally inconsistent (wrong
/// `codes` or `scales` length relative to `original_shape` and `bits`).
pub fn turbo_dequantize(blocks: &TurboBlocks) -> Result<Vec<f32>> {
    turbo_dequantize_with_codebook(blocks, None)
}

/// Dequantize a [`TurboBlocks`] back to f32, optionally using a custom codebook.
///
/// When `codebook_override` is `Some`, those centroids are used instead of the
/// built-in Lloyd-Max table. This is the correct inverse of
/// [`turbo_quantize_v_with_codebook`] — callers that encode with an override
/// **must** pass the same override here, or dequantized values will be wrong.
///
/// Output is a flat row-major f32 `Vec` of shape `blocks.original_shape`.
///
/// # Errors
///
/// Returns `Error::Quant` if the blocks are internally inconsistent (wrong
/// `codes` or `scales` length relative to `original_shape` and `bits`).
pub fn turbo_dequantize_with_codebook(
    blocks: &TurboBlocks,
    codebook_override: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let builtin = lloyd_gaussian_codebook(blocks.bits)?;
    let codebook: &[f32] = codebook_override.unwrap_or(builtin);

    // Validate shape consistency.
    // original_shape is [i32; 4] — length-4 check is a compile-time invariant.
    let total_elems: usize = blocks.original_shape.iter().map(|&d| d as usize).product();
    let n_blocks = total_elems / GROUP_SIZE;

    let bits_per_block = GROUP_SIZE * blocks.bits as usize;
    let bytes_per_block = bits_per_block.div_ceil(8);

    if blocks.scales.len() != n_blocks {
        return Err(err_scales_count(blocks.scales.len(), n_blocks));
    }
    let expected_code_bytes = n_blocks * bytes_per_block;
    if blocks.codes.len() != expected_code_bytes {
        return Err(err_codes_len(
            blocks.codes.len(),
            expected_code_bytes,
            n_blocks,
            bytes_per_block,
        ));
    }

    let mut out = vec![0.0_f32; total_elems];

    // Walk blocks via zip of scales + code-byte chunks + out chunks.
    // chunks_exact_mut(GROUP_SIZE) on out lets LLVM prove the inner writes
    // are in-bounds without per-element `out_offset + elem` arithmetic.
    for ((block, &scale), out_block) in blocks
        .scales
        .iter()
        .enumerate()
        .zip(out.chunks_exact_mut(GROUP_SIZE))
    {
        let byte_offset = block * bytes_per_block;
        // byte_offset + bytes_per_block <= blocks.codes.len(): validated above (err_codes_len guard).
        #[allow(
            clippy::indexing_slicing,
            reason = "byte_offset+bytes_per_block <= blocks.codes.len(): codes length validated by err_codes_len guard; block < n_blocks from iter"
        )]
        let block_codes = &blocks.codes[byte_offset..byte_offset + bytes_per_block];

        for (elem, slot) in out_block.iter_mut().enumerate() {
            let idx = unpack_index(block_codes, elem, blocks.bits) as usize;
            // idx < codebook.len(): codebook has 2^bits entries; unpack_index returns ≤ 2^bits-1.
            // For override codebooks, callers must pass the same length — no runtime length check
            // here (the encode path already validated length + monotonicity).
            #[allow(
                clippy::indexing_slicing,
                reason = "idx < codebook.len(): unpack_index returns value ≤ 2^bits-1; lloyd_gaussian_codebook returns 2^bits entries; override callers validated at encode time"
            )]
            let dequant_val = codebook[idx] * scale;
            *slot = dequant_val;
        }
    }

    Ok(out)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Return the index of the nearest centroid using sorted-boundary comparisons.
///
/// Equivalent to counting how many boundaries the normalized value exceeds
/// (the same algorithm as the Metal kernel's linear scan in `turboquant_metal.py`).
#[inline]
fn nearest_centroid(normalized: f32, codebook: &[f32]) -> usize {
    // Build midpoint boundaries on the fly — avoids allocating a separate vec.
    // For a codebook of N centroids there are N-1 midpoint boundaries.
    // idx = count of boundaries exceeded (from left).
    let n = codebook.len();
    let mut idx = 0usize;
    for b in 0..n - 1 {
        // b ∈ [0, n-2] from loop bound 0..n-1; b+1 ≤ n-1 < n == codebook.len().
        #[allow(
            clippy::indexing_slicing,
            reason = "b ∈ [0,n-2] from loop bound 0..n-1; b+1 ≤ n-1 < codebook.len()"
        )]
        let boundary = (codebook[b] + codebook[b + 1]) * 0.5;
        if normalized > boundary {
            idx += 1;
        }
    }
    idx
}

/// Pack `bits`-wide `index` into `block_bytes` at element position `elem`.
///
/// Indices are packed LSB-first; element 0 occupies bits 0..bits-1.
#[inline]
pub(crate) fn pack_index(block_bytes: &mut [u8], elem: usize, index: u8, bits: u8) {
    let bit_offset = elem * bits as usize;
    let mut remaining = bits as usize;
    let mut shift = bit_offset % 8;
    let mut byte_idx = bit_offset / 8;
    let mut val = u32::from(index);

    while remaining > 0 {
        let take = remaining.min(8 - shift);
        let mask = ((1u32 << take) - 1) as u8;
        // byte_idx advances through the byte range occupied by elem; caller passes
        // block_bytes whose length == bytes_per_block, covering all elems in the block.
        #[allow(
            clippy::indexing_slicing,
            reason = "byte_idx < block_bytes.len(): block_bytes covers GROUP_SIZE*bits/8 bytes; byte_idx bounded by bit-packing arithmetic over valid elem/bits"
        )]
        let byte_slot = &mut block_bytes[byte_idx];
        *byte_slot |= ((val as u8) & mask) << shift;
        val >>= take;
        remaining -= take;
        shift = 0;
        byte_idx += 1;
    }
}

/// Unpack a `bits`-wide index from `block_bytes` at element position `elem`.
#[inline]
fn unpack_index(block_bytes: &[u8], elem: usize, bits: u8) -> u8 {
    let bit_offset = elem * bits as usize;
    let byte_start = bit_offset / 8;
    let bit_shift = bit_offset % 8;
    let mask = (1u32 << bits) - 1;

    // Read at most 2 bytes (bits ≤ 8, so at most 1 byte boundary crossing).
    // byte_start = (elem * bits) / 8 < block_bytes.len(): block_bytes covers GROUP_SIZE*bits/8
    // bytes; elem < GROUP_SIZE is upheld by callers (from out_block.iter_mut().enumerate()).
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_start < block_bytes.len(): block_bytes covers bytes_per_block=GROUP_SIZE*bits/8; elem < GROUP_SIZE from out_block iterator"
    )]
    let b0 = u32::from(block_bytes[byte_start]);
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_start+1 < block_bytes.len() is checked by the surrounding if condition"
    )]
    let b1 = if byte_start + 1 < block_bytes.len() {
        u32::from(block_bytes[byte_start + 1])
    } else {
        0
    };
    let window = b0 | (b1 << 8);
    ((window >> bit_shift) & mask) as u8
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "turboquant_tests.rs"]
mod tests;
