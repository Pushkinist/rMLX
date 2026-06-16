//! PlanarQuant KV-cache quantization — Givens-rotation codebook + N-bit quantization.
//!
//! # What this is
//!
//! PlanarQuant rotates each pair of values by a Givens rotation pulled from a
//! 16-entry codebook before quantizing. Compared to TurboQuant V4 (which uses a
//! single per-block scale across 32 elements), PlanarQuant uses a **per-pair scale**
//! (one scale per 2 elements) after rotation. This guarantees equal or lower
//! reconstruction error on any input, at the cost of 4 bits of rotation index and
//! additional scale storage per pair.
//!
//! Reference: `ParaMind2025/isoquant` (CUDA reference implementation). rMLX is
//! the first working Apple Silicon implementation as of 2026-05 —
//! `johndpope/llama-cpp-turboquant feature/planarquant-kv-cache` Metal kernels
//! fall back to CPU (issue #7).
//!
//! # Algorithm
//!
//! Input vector `x ∈ R^D` is processed in pairs `(x_{2i}, x_{2i+1})`:
//!
//! 1. **Rotation codebook**: 16 Givens rotations sampled at angles
//!    `θ_k = k × π/16` for k ∈ 0..15 (evenly covers [0, π) — sufficient since
//!    `R(θ + π) = -R(θ)` and sign is absorbed by the scale).
//!    Each entry stores `[cos θ, -sin θ, sin θ, cos θ]` — the 2×2 rotation matrix
//!    in row-major order.
//!
//! 2. **Rotation selection per pair**: For each pair `(a, b)`, try all 16 rotations,
//!    apply a **pair-local** scale `max(|ya|, |yb|) / max_centroid`, quantize+dequantize,
//!    and pick the rotation `k` that minimizes max absolute reconstruction error.
//!    Store `k` as 4 bits per pair.
//!
//! 3. **Quantize**: Apply chosen rotation, compute per-pair scale, quantize both
//!    elements against that scale using the Lloyd-Max N(0,1) N-bit codebook.
//!
//! 4. **Dequantize**: For each pair, pull rotation index → per-pair scale → apply R_k^T.
//!
//! # Block layout
//!
//! Groups of `GROUP_SIZE = 32` elements per block (16 pairs per block):
//! - `codes`: 4 u32 words (16 bytes) per block. Codes use the shared **word
//!   convention** (`vals_per_word = 32 / bits`): element `e` of a block lives at
//!   `word = e / vals_per_word`, `shift = (e % vals_per_word) * bits`, within a
//!   little-endian u32 word. This is the same convention iso3 / rotor3 and the
//!   PlanarQuant MSL kernels use, so the byte stream is path-independent (CPU
//!   and GPU produce/consume identical bytes). For `bits=4` (8 vals/u32) this
//!   is byte-identical to a dense LSB-first stream; for `bits=3` (10 vals/u32,
//!   30 bits used, 2 wasted per word) it is **not** dense — a dense layout would
//!   corrupt on any CPU↔GPU round-trip (e.g. SSD spill/hydrate).
//! - `scales`: `GROUP_SIZE / 2` f32 values per block — **one scale per pair**.
//! - `rotations`: `GROUP_SIZE / 4` bytes per block — 4-bit rotation index per pair,
//!   2 indices per byte, packed LSB-first. `GROUP_SIZE/2 = 16` pairs per block,
//!   so `8` rotation bytes per block.
//!
//! # Per-pair scale vs TurboQuant per-block scale
//!
//! TurboQuant uses one scale per 32 elements. PlanarQuant uses one scale per 2
//! elements (after rotation). This alone is the primary source of error reduction —
//! finer-grained scale means the quantization range is tighter. The rotation
//! additionally reduces error when pairs are correlated by aligning them to the
//! quantization grid.
//!
//! # Codebook derivation
//!
//! Givens rotations at `θ_k = k × π/16`, k ∈ 0..15. Stored as
//! `[cos θ, -sin θ, sin θ, cos θ]` (row-major 2×2 matrix). Orthogonal by
//! construction: `R_k^T R_k = I`.

use crate::turboquant::{lloyd_gaussian_codebook, GROUP_SIZE};
use rmlx_core::{Error, Result};
use std::f32::consts::PI;

// ── Rotation codebook ─────────────────────────────────────────────────────────

/// Number of rotation codebook entries (4-bit index → 16 entries).
pub const N_ROTATIONS: usize = 16;

/// Return the 16-entry Givens rotation codebook.
///
/// Each entry `[cos θ, -sin θ, sin θ, cos θ]` is a 2×2 rotation matrix in
/// row-major order. Angles θ_k = k × π/16 for k ∈ 0..15.
///
/// The codebook is orthogonal by construction: `R_k^T R_k = I` (within f32 precision).
///
/// # Design
///
/// Evenly sampling [0, π) is sufficient: a rotation by θ and θ+π differ only by
/// sign of the output pair, and the sign is absorbed by the per-pair scale
/// (which is max-abs, always non-negative). Using [0, 2π) would waste half the
/// codebook on sign-equivalent rotations.
pub fn planar_rotation_codebook() -> Vec<[f32; 4]> {
    (0..N_ROTATIONS)
        .map(|k| {
            let theta = (k as f32) * PI / (N_ROTATIONS as f32);
            let c = theta.cos();
            let s = theta.sin();
            // Row-major 2×2: [[c, -s], [s, c]]
            [c, -s, s, c]
        })
        .collect()
}

/// Apply rotation `R = [[c, -s], [s, c]]` to pair `(a, b)`.
#[inline]
fn rotate(a: f32, b: f32, entry: &[f32; 4]) -> (f32, f32) {
    let ya = entry[0].mul_add(a, entry[1] * b);
    let yb = entry[2].mul_add(a, entry[3] * b);
    (ya, yb)
}

/// Apply transpose rotation `R^T = [[c, s], [-s, c]]` to pair `(ya, yb)`.
#[inline]
fn rotate_t(ya: f32, yb: f32, entry: &[f32; 4]) -> (f32, f32) {
    // R^T = [[c, s], [-s, c]] where entry = [c, -s, s, c]
    let a = entry[0].mul_add(ya, entry[2] * yb); // c*ya + s*yb
    let b = entry[1].mul_add(ya, entry[3] * yb); // -s*ya + c*yb
    (a, b)
}

// ── Quantized representation ──────────────────────────────────────────────────

/// Compact representation of a PlanarQuant-quantized tensor.
///
/// Blocks are `GROUP_SIZE = 32` element groups (16 pairs per block).
/// Each block has:
/// - `GROUP_SIZE / 2` f32 scales (one per pair — key difference from TurboQuant)
/// - 4 u32 code words = 16 code bytes (word convention `32 / bits` vals/u32,
///   path-independent CPU↔GPU; dense only for `bits=4`)
/// - `GROUP_SIZE / 4` rotation bytes (4 bits per pair, 2 per byte, 8 bytes/block)
///
/// `original_shape` stores the original `[B, kv_h, S, D]` dimensions.
/// `bits` is in `{1, 2, 3, 4}`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed quantized-tensor struct — five fields are the complete PlanarQuant block-storage contract; adding a field requires updating planar_quantize and all dequant callers"
)]
#[derive(Debug, Clone)]
pub struct PlanarBlocks {
    /// Quantized indices packed in the shared word convention
    /// (`vals_per_word = 32 / bits`, little-endian u32 words, 4 words/block).
    /// Byte-identical to a dense LSB-first stream only for `bits=4`.
    pub codes: Vec<u8>,
    /// Per-pair scale factors (one per 2 elements, not per 32).
    /// Length = `total_elems / 2`.
    pub scales: Vec<f32>,
    /// Per-pair rotation indices, 4 bits each, 2 per byte, packed LSB-first.
    /// Length = `total_elems / 4` bytes (16 pairs/block × 0.5 bytes/pair).
    pub rotations: Vec<u8>,
    /// Original tensor shape `[B, kv_h, S, D]` as signed i32.
    ///
    /// Fixed-size array (always 4-D): 16 B inline vs 24 B stack + 16 B heap for Vec<i32>.
    pub original_shape: [i32; 4],
    /// Bit-width. Must be in `{1, 2, 3, 4}`.
    pub bits: u8,
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Pairs per block: `GROUP_SIZE / 2`.
const PAIRS_PER_BLOCK: usize = GROUP_SIZE / 2;

/// Rotation bytes per block: 4-bit per pair, 2 pairs per byte.
const ROT_BYTES_PER_BLOCK: usize = PAIRS_PER_BLOCK / 2;

/// u32 code words per block under the shared word convention.
///
/// `vals_per_word = 32 / bits`. For both supported widths the word count is the
/// same: `bits=4` → 8 vals/u32 → `ceil(32/8) = 4`; `bits=3` → 10 vals/u32 →
/// `ceil(32/10) = 4`. Matching the PlanarQuant MSL kernels and iso3/rotor3 so
/// the byte stream round-trips across the CPU/GPU boundary unchanged.
const CODE_WORDS_PER_BLOCK: usize = 4;

/// Code bytes per block: 4 u32 words.
const CODE_BYTES_PER_BLOCK: usize = CODE_WORDS_PER_BLOCK * 4;

/// Values packed per u32 word for `bits`: `32 / bits`.
///
/// `bits=3` → 10 (30 bits used, 2 wasted); `bits=4` → 8 (dense). Only 3 and 4
/// are valid PlanarQuant widths; the codebook lookup rejects others upstream.
#[inline]
const fn vals_per_word(bits: u8) -> usize {
    (32 / bits) as usize
}

// ── Quantize ──────────────────────────────────────────────────────────────────

/// Cold helper: "group_size must be GROUP_SIZE" error.
#[cold]
fn err_group_size(got: usize) -> Error {
    Error::Quant(format!(
        "planarquant: group_size must be {GROUP_SIZE}, got {got}"
    ))
}

/// Cold helper: "original_shape must have 4 elements" error.
#[cold]
fn err_shape_rank(got: usize) -> Error {
    Error::Quant(format!(
        "planarquant: original_shape must have 4 elements [B, kv_h, S, D], \
         got {got} element(s)"
    ))
}

/// Cold helper: "all shape dimensions must be positive" error.
#[cold]
fn err_shape_nonpositive(shape: &[i32]) -> Error {
    Error::Quant(format!(
        "planarquant: all shape dimensions must be positive, got {shape:?}"
    ))
}

/// Cold helper: "x.len() != product of original_shape" error.
#[cold]
fn err_len_mismatch(got: usize, shape: &[i32], expected: usize) -> Error {
    Error::Quant(format!(
        "planarquant: x.len()={got} != product of original_shape={shape:?} ({expected})"
    ))
}

/// Cold helper: "last dimension D must be multiple of GROUP_SIZE" error.
#[cold]
fn err_d_not_multiple(d: usize) -> Error {
    Error::Quant(format!(
        "planarquant: last dimension D={d} must be a multiple of GROUP_SIZE={GROUP_SIZE}"
    ))
}

/// Cold helper: "scales.len() != n_pairs" error from planar_dequantize.
#[cold]
fn err_scales_count(got: usize, expected: usize) -> Error {
    Error::Quant(format!(
        "planarquant: scales.len()={got} != n_pairs={expected}"
    ))
}

/// Cold helper: "codes.len() != expected" error from planar_dequantize.
#[cold]
fn err_codes_len(got: usize, expected: usize) -> Error {
    Error::Quant(format!(
        "planarquant: codes.len()={got} != expected {expected}"
    ))
}

/// Cold helper: "rotations.len() != expected" error from planar_dequantize.
#[cold]
fn err_rotations_len(got: usize, expected: usize) -> Error {
    Error::Quant(format!(
        "planarquant: rotations.len()={got} != expected {expected}"
    ))
}

/// Quantize a tensor using PlanarQuant.
///
/// # Input
///
/// `x` flat row-major f32 slice of shape `original_shape = [B, kv_h, S, D]`.
/// `D` must be a multiple of `GROUP_SIZE = 32`.
/// `group_size` must equal `GROUP_SIZE` (32). `bits` must be in `{1, 2, 3, 4}`.
///
/// # Algorithm (per block of 32 elements = 16 pairs)
///
/// For each pair `(a, b)` in the block:
/// 1. Try all 16 Givens rotations. For each, compute pair-local scale
///    `max(|ya|, |yb|) / max_centroid`, quantize+dequantize, measure max error.
/// 2. Choose the rotation minimizing reconstruction error.
/// 3. Apply chosen rotation, store scale and quantized indices.
///
/// # Errors
///
/// Returns `Error::Quant` for unsupported `bits`, wrong shape, or non-multiple-of-32 D.
pub fn planar_quantize(
    x: &[f32],
    group_size: usize,
    bits: u8,
    original_shape: &[i32],
) -> Result<PlanarBlocks> {
    if group_size != GROUP_SIZE {
        return Err(err_group_size(group_size));
    }

    let codebook = lloyd_gaussian_codebook(bits)?;
    let max_centroid = codebook
        .iter()
        .copied()
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));

    if original_shape.len() != 4 {
        return Err(err_shape_rank(original_shape.len()));
    }
    if original_shape.iter().any(|&d| d <= 0) {
        return Err(err_shape_nonpositive(original_shape));
    }
    // original_shape.len() == 4 is checked above (err_shape_rank guard).
    #[allow(
        clippy::indexing_slicing,
        reason = "original_shape.len()==4 is validated by the err_shape_rank guard four lines above"
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

    let rot_cb = planar_rotation_codebook();
    let n_blocks = total_elems / GROUP_SIZE;
    let code_bytes_per_block = CODE_BYTES_PER_BLOCK;
    let n_pairs = total_elems / 2;

    let mut codes = vec![0u8; n_blocks * code_bytes_per_block];
    let mut scales = vec![0.0_f32; n_pairs]; // one per pair
    let mut rotations = vec![0u8; n_blocks * ROT_BYTES_PER_BLOCK];

    for block in 0..n_blocks {
        let block_start = block * GROUP_SIZE;
        // block < n_blocks == total_elems/GROUP_SIZE; x.len()==total_elems (validated above),
        // so block_start + GROUP_SIZE <= total_elems == x.len().
        #[allow(
            clippy::indexing_slicing,
            reason = "block_start+GROUP_SIZE <= x.len(): block < n_blocks=total_elems/GROUP_SIZE; x.len()==total_elems validated above"
        )]
        let group = &x[block_start..block_start + GROUP_SIZE];

        let code_offset = block * code_bytes_per_block;
        let rot_offset = block * ROT_BYTES_PER_BLOCK;
        let scale_offset = block * PAIRS_PER_BLOCK; // one scale per pair

        for pair in 0..PAIRS_PER_BLOCK {
            // pair < PAIRS_PER_BLOCK == GROUP_SIZE/2; group.len()==GROUP_SIZE,
            // so pair*2+1 < GROUP_SIZE == group.len().
            #[allow(
                clippy::indexing_slicing,
                reason = "pair*2+1 < group.len(): pair < PAIRS_PER_BLOCK=GROUP_SIZE/2; group.len()==GROUP_SIZE"
            )]
            let a = group[pair * 2];
            #[allow(
                clippy::indexing_slicing,
                reason = "pair*2+1 < group.len(): pair < PAIRS_PER_BLOCK=GROUP_SIZE/2; group.len()==GROUP_SIZE"
            )]
            let b = group[pair * 2 + 1];

            // ── Choose best rotation (pair-local scale criterion) ─────────────
            let mut best_rot = 0u8;
            let mut best_scale = 0.0_f32;
            let mut best_err = f32::INFINITY;

            for (k, rot_entry) in rot_cb.iter().enumerate() {
                let (ya, yb) = rotate(a, b, rot_entry);

                let abs_max = ya.abs().max(yb.abs());
                let scale = if abs_max > 0.0 {
                    abs_max / max_centroid
                } else {
                    0.0
                };
                let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

                let idx_a = nearest_centroid(ya * inv_scale, codebook);
                let idx_b = nearest_centroid(yb * inv_scale, codebook);
                // idx_a/idx_b returned by nearest_centroid are in [0, codebook.len()-1].
                #[allow(
                    clippy::indexing_slicing,
                    reason = "idx_a < codebook.len(): nearest_centroid returns count of boundaries exceeded, bounded to [0, n-1] where n==codebook.len()"
                )]
                let recon_a = codebook[idx_a] * scale;
                #[allow(
                    clippy::indexing_slicing,
                    reason = "idx_b < codebook.len(): nearest_centroid returns count of boundaries exceeded, bounded to [0, n-1] where n==codebook.len()"
                )]
                let recon_b = codebook[idx_b] * scale;

                let err = (ya - recon_a).abs().max((yb - recon_b).abs());
                if err < best_err {
                    best_err = err;
                    best_rot = k as u8;
                    best_scale = scale;
                }
            }

            // ── Apply chosen rotation and encode ──────────────────────────────
            // best_rot was set from k < rot_cb.len() in the loop above.
            #[allow(
                clippy::indexing_slicing,
                reason = "best_rot as usize < rot_cb.len(): best_rot was assigned from k which is an enumerate index over rot_cb"
            )]
            let rot_entry = &rot_cb[best_rot as usize];
            let (ya, yb) = rotate(a, b, rot_entry);

            // scale_offset + pair < scales.len(): scales.len()==n_pairs==n_blocks*PAIRS_PER_BLOCK;
            // block < n_blocks and pair < PAIRS_PER_BLOCK.
            #[allow(
                clippy::indexing_slicing,
                reason = "scale_offset+pair < scales.len(): scales.len()==n_blocks*PAIRS_PER_BLOCK; block<n_blocks and pair<PAIRS_PER_BLOCK"
            )]
            let scales_slot = &mut scales[scale_offset + pair];
            *scales_slot = best_scale;

            let inv_scale = if best_scale > 0.0 {
                1.0 / best_scale
            } else {
                0.0
            };
            let elem_a = pair * 2;
            let elem_b = pair * 2 + 1;
            // code_offset+code_bytes_per_block <= codes.len(): block<n_blocks; codes.len()==n_blocks*code_bytes_per_block.
            #[allow(
                clippy::indexing_slicing,
                reason = "code_offset+code_bytes_per_block <= codes.len(): codes.len()==n_blocks*code_bytes_per_block; block<n_blocks"
            )]
            let block_buf = &mut codes[code_offset..code_offset + code_bytes_per_block];
            let idx_a = nearest_centroid(ya * inv_scale, codebook) as u8;
            let idx_b = nearest_centroid(yb * inv_scale, codebook) as u8;
            pack_index(block_buf, elem_a, idx_a, bits);
            pack_index(block_buf, elem_b, idx_b, bits);

            // ── Pack rotation index (4-bit, 2 per byte, LSB-first) ────────────
            let rot_byte = pair / 2;
            let rot_shift = (pair % 2) * 4;
            // rot_offset+rot_byte < rotations.len(): rotations.len()==n_blocks*ROT_BYTES_PER_BLOCK;
            // rot_byte = pair/2 < PAIRS_PER_BLOCK/2 == ROT_BYTES_PER_BLOCK; block < n_blocks.
            #[allow(
                clippy::indexing_slicing,
                reason = "rot_offset+rot_byte < rotations.len(): rotations.len()==n_blocks*ROT_BYTES_PER_BLOCK; rot_byte=pair/2<PAIRS_PER_BLOCK/2==ROT_BYTES_PER_BLOCK; block<n_blocks"
            )]
            let rot_slot = &mut rotations[rot_offset + rot_byte];
            *rot_slot |= (best_rot & 0xF) << rot_shift;
        }
    }

    Ok(PlanarBlocks {
        codes,
        scales,
        rotations,
        original_shape: shape4,
        bits,
    })
}

// ── Dequantize ────────────────────────────────────────────────────────────────

/// Dequantize a [`PlanarBlocks`] back to f32.
///
/// Output is a flat row-major f32 `Vec` of shape `blocks.original_shape`.
///
/// # Errors
///
/// Returns `Error::Quant` if the blocks are internally inconsistent.
pub fn planar_dequantize(blocks: &PlanarBlocks) -> Result<Vec<f32>> {
    let codebook = lloyd_gaussian_codebook(blocks.bits)?;

    // original_shape is [i32; 4] — length-4 check is a compile-time invariant.
    let total_elems: usize = blocks.original_shape.iter().map(|&d| d as usize).product();
    let n_blocks = total_elems / GROUP_SIZE;
    let n_pairs = total_elems / 2;
    let code_bytes_per_block = CODE_BYTES_PER_BLOCK;

    if blocks.scales.len() != n_pairs {
        return Err(err_scales_count(blocks.scales.len(), n_pairs));
    }
    if blocks.codes.len() != n_blocks * code_bytes_per_block {
        return Err(err_codes_len(
            blocks.codes.len(),
            n_blocks * code_bytes_per_block,
        ));
    }
    if blocks.rotations.len() != n_blocks * ROT_BYTES_PER_BLOCK {
        return Err(err_rotations_len(
            blocks.rotations.len(),
            n_blocks * ROT_BYTES_PER_BLOCK,
        ));
    }

    let rot_cb = planar_rotation_codebook();
    let mut out = vec![0.0_f32; total_elems];

    // Walk blocks via zip of pre-sliced block buffers + out.chunks_exact_mut(GROUP_SIZE).
    // This replaces out[out_offset + elem_a/b] with sequential pair writes through
    // chunks_exact_mut(2), eliding per-element out-bounds proofs.
    let scale_chunks = blocks.scales.chunks_exact(PAIRS_PER_BLOCK);
    let rot_chunks = blocks.rotations.chunks_exact(ROT_BYTES_PER_BLOCK);
    let code_chunks = blocks.codes.chunks_exact(code_bytes_per_block);
    let out_blocks = out.chunks_exact_mut(GROUP_SIZE);

    for (((scale_block, rot_block), code_block), out_block) in scale_chunks
        .zip(rot_chunks)
        .zip(code_chunks)
        .zip(out_blocks)
    {
        // Pair loop: walk out_block in pairs of 2 elements.
        for (pair, (out_pair, &scale)) in out_block
            .chunks_exact_mut(2)
            .zip(scale_block.iter())
            .enumerate()
        {
            // Unpack rotation index (4 bits, 2 per byte).
            let rot_byte = pair / 2;
            let rot_shift = (pair % 2) * 4;
            // rot_byte = pair/2 < PAIRS_PER_BLOCK/2 == ROT_BYTES_PER_BLOCK == rot_block.len()
            // (rot_block comes from chunks_exact(ROT_BYTES_PER_BLOCK)).
            #[allow(
                clippy::indexing_slicing,
                reason = "rot_byte=pair/2 < PAIRS_PER_BLOCK/2==ROT_BYTES_PER_BLOCK==rot_block.len(); rot_block from chunks_exact(ROT_BYTES_PER_BLOCK)"
            )]
            let rot_idx = ((rot_block[rot_byte] >> rot_shift) & 0xF) as usize;
            // rot_idx ∈ [0,15] from & 0xF; rot_cb has 16 entries (planar_rotation_codebook()).
            #[allow(
                clippy::indexing_slicing,
                reason = "rot_idx = (byte & 0xF) as usize ∈ [0,15]; planar_rotation_codebook() returns a 16-entry table"
            )]
            let rot_entry = &rot_cb[rot_idx];

            // Unpack quantized indices.
            let elem_a = pair * 2;
            let elem_b = pair * 2 + 1;
            let idx_a = unpack_index(code_block, elem_a, blocks.bits) as usize;
            let idx_b = unpack_index(code_block, elem_b, blocks.bits) as usize;

            // Dequantize in rotated space.
            // idx_a/idx_b returned by unpack_index are bounded to [0, 2^bits-1] < codebook.len()
            // (lloyd_gaussian_codebook always returns 2^bits entries, checked at fn entry).
            #[allow(
                clippy::indexing_slicing,
                reason = "idx_a < codebook.len(): unpack_index returns value ≤ 2^bits-1; codebook has 2^bits entries (lloyd_gaussian_codebook)"
            )]
            let ya = codebook[idx_a] * scale;
            #[allow(
                clippy::indexing_slicing,
                reason = "idx_b < codebook.len(): same invariant as idx_a above"
            )]
            let yb = codebook[idx_b] * scale;

            // Undo rotation via R^T — write directly to the two-slot out_pair.
            let (a, b) = rotate_t(ya, yb, rot_entry);
            // out_pair.len()==2 from chunks_exact_mut(2); indices 0 and 1 always valid.
            // Use get_mut to avoid indexing_slicing lint while preserving zero-overhead access.
            if let Some(s0) = out_pair.get_mut(0) {
                *s0 = a;
            }
            if let Some(s1) = out_pair.get_mut(1) {
                *s1 = b;
            }
        }
    }

    Ok(out)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Return the index of the nearest centroid via midpoint boundary comparisons.
#[inline]
fn nearest_centroid(normalized: f32, codebook: &[f32]) -> usize {
    let n = codebook.len();
    let mut idx = 0usize;
    for b in 0..n - 1 {
        // b ∈ [0, n-2], so b+1 ≤ n-1 < n == codebook.len().
        #[allow(
            clippy::indexing_slicing,
            reason = "b ∈ [0,n-2] from the loop bound 0..n-1; b+1 ≤ n-1 < codebook.len()"
        )]
        let boundary = (codebook[b] + codebook[b + 1]) * 0.5;
        if normalized > boundary {
            idx += 1;
        }
    }
    idx
}

/// Pack a `bits`-wide `index` into `block_bytes` at element position `elem`,
/// using the shared word convention (`vals_per_word = 32 / bits`).
///
/// `block_bytes` is one block's code buffer: `CODE_WORDS_PER_BLOCK` little-endian
/// u32 words (`CODE_BYTES_PER_BLOCK` bytes). Element `elem` maps to
/// `word = elem / vals_per_word`, `shift = (elem % vals_per_word) * bits` within
/// that word. A code never straddles a word boundary, so this packs into one
/// word. This is the same layout the PlanarQuant MSL kernels and iso3/rotor3 use.
#[inline]
fn pack_index(block_bytes: &mut [u8], elem: usize, index: u8, bits: u8) {
    let vpw = vals_per_word(bits);
    let word = elem / vpw;
    let shift = (elem % vpw) * bits as usize;
    let mask = (1u32 << bits) - 1;
    let byte_base = word * 4;
    // Read-modify-write the target little-endian u32 word in place.
    // byte_base + 4 <= block_bytes.len(): block_bytes covers CODE_WORDS_PER_BLOCK
    // words; word = elem/vpw < CODE_WORDS_PER_BLOCK for elem < GROUP_SIZE.
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_base+4 <= block_bytes.len(): block_bytes covers CODE_WORDS_PER_BLOCK u32 words; word=elem/vpw < CODE_WORDS_PER_BLOCK for elem < GROUP_SIZE"
    )]
    let word_bytes = &mut block_bytes[byte_base..byte_base + 4];
    let mut w = u32::from_le_bytes(word_bytes.try_into().unwrap_or([0u8; 4]));
    w |= (u32::from(index) & mask) << shift;
    word_bytes.copy_from_slice(&w.to_le_bytes());
}

/// Unpack a `bits`-wide index from `block_bytes` at element position `elem`,
/// using the shared word convention (`vals_per_word = 32 / bits`).
#[inline]
fn unpack_index(block_bytes: &[u8], elem: usize, bits: u8) -> u8 {
    let vpw = vals_per_word(bits);
    let word = elem / vpw;
    let shift = (elem % vpw) * bits as usize;
    let mask = (1u32 << bits) - 1;
    let byte_base = word * 4;
    // byte_base + 4 <= block_bytes.len(): same invariant as pack_index.
    #[allow(
        clippy::indexing_slicing,
        reason = "byte_base+4 <= block_bytes.len(): block_bytes covers CODE_WORDS_PER_BLOCK u32 words; word=elem/vpw < CODE_WORDS_PER_BLOCK for elem < GROUP_SIZE"
    )]
    let word_bytes = &block_bytes[byte_base..byte_base + 4];
    let w = u32::from_le_bytes(word_bytes.try_into().unwrap_or([0u8; 4]));
    ((w >> shift) & mask) as u8
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "planarquant_tests.rs"]
mod tests;
