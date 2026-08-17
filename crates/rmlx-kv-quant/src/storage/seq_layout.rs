//! Shared head↔sequence reorder helpers for the quantized KV storage structs.
//!
//! Every flat-buffer quantized storage (`QuantK`, `QuantV`, `QuantKTurbo*`,
//! `QuantPlanar*`) accumulates one chunk per `append` at a **sequence offset**
//! (`prev_seq * words_per_seq`), so the active prefix is physically
//! sequence-major: for each token, all heads are contiguous (`[B, S, kv_h, D]`
//! ordering). The dequant view, however, reshapes the flat prefix to the
//! logical head-major `[B, kv_h, S, D]`. Those two orderings only coincide when
//! `B * kv_h == 1`. With `kv_h > 1` and more than one chunk, the second chunk's
//! tokens land after *all* heads' prefixes while the head-major reshape maps one
//! head's new-token slot onto another head's prefix — a head transposition that
//! silently corrupts the cache.
//!
//! The fix makes every such buffer canonically sequence-major: `append`
//! reorders the incoming head-major chunk heads↔seq before quantizing, and the
//! dequant path reshapes the prefix as `[B, S, kv_h, D]` and reorders back to
//! `[B, kv_h, S, D]`. For a single chunk (`prev_seq == 0`) the two reorders
//! cancel, so the common cold-prefill path is logically unchanged.
//!
//! These are the CPU-side scalar mirrors of the GPU `transpose(&[0, 2, 1, 3])`;
//! the GPU side additionally materializes the transposed view with
//! `Array::contiguous` before any custom MSL kernel, because such kernels read
//! their input by raw linear index and ignore MLX lazy-transpose strides.
//!
//! # Chunk boundaries matter once `B > 1`
//!
//! A store that accumulates one buffer per `append` decodes by concatenating
//! them. That concatenation is a single `[B, S_total, kv_h, D]` run **only when
//! `B == 1`**: every chunk is itself `[B, S_chunk, kv_h, D]`, so with more than
//! one batch element each chunk restarts at batch 0 and the chunks interleave.
//! Reading the concatenation as one run then maps a later chunk's batch-0 rows
//! onto batch-1 sequence slots — head/batch-scrambled K/V, with no error.
//! [`transpose_chunked_seq_heads`] is the reorder that knows where the chunks
//! end; [`transpose_seq_heads`] stays correct only for a buffer that really is
//! one `[B, S_total, kv_h, D]` run (a single chunk, or a flat store written at
//! sequence offsets).

use rmlx_core::error::{Error, Result};

/// Reorder a flat head-major `[B, kv_h, S, D]` buffer to sequence-major
/// `[B, S, kv_h, D]`. Used by the CPU `append` paths to mirror the GPU
/// buffer's sequence-major layout.
#[allow(
    clippy::indexing_slicing,
    reason = "src/out are sized b*kv_h*s*d; every (bi,h,si) base + d stays in bounds by construction"
)]
pub(super) fn transpose_heads_seq(
    src: &[f32],
    b: usize,
    kv_h: usize,
    s: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * kv_h * s * d];
    for bi in 0..b {
        for h in 0..kv_h {
            for si in 0..s {
                let src_base = ((bi * kv_h + h) * s + si) * d;
                let dst_base = ((bi * s + si) * kv_h + h) * d;
                out[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
            }
        }
    }
    out
}

/// Inverse of [`transpose_heads_seq`]: reorder a flat sequence-major
/// `[B, S, kv_h, D]` buffer back to head-major `[B, kv_h, S, D]`. Used by the
/// CPU dequant paths so callers see the logical `[B, kv_h, S, D]`.
#[allow(
    clippy::indexing_slicing,
    reason = "src/out are sized b*s*kv_h*d; every (bi,si,h) base + d stays in bounds by construction"
)]
pub(super) fn transpose_seq_heads(
    src: &[f32],
    b: usize,
    s: usize,
    kv_h: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * s * kv_h * d];
    for bi in 0..b {
        for si in 0..s {
            for h in 0..kv_h {
                let src_base = ((bi * s + si) * kv_h + h) * d;
                let dst_base = ((bi * kv_h + h) * s + si) * d;
                out[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
            }
        }
    }
    out
}

/// Reorder a **chunked** sequence-major buffer back to head-major
/// `[B, kv_h, S_total, D]`.
///
/// `src` is the concatenation of one `[B, S_chunk, kv_h, D]` buffer per
/// `append`, in append order; `chunk_rows` is each chunk's row count
/// (`B * kv_h * S_chunk`) in the same order. Every chunk is reordered at its own
/// sequence offset, which is what [`transpose_seq_heads`] over the whole
/// concatenation cannot do once `B > 1` (see the module note).
///
/// At `B == 1` this is exactly [`transpose_seq_heads`] over the concatenation —
/// the sequence axis is then the outermost axis of every chunk, so the chunk
/// boundaries carry no information.
///
/// # Errors
///
/// Returns [`Error::Quant`] when the chunks do not partition
/// `[B, S_total, kv_h, D]`: a wrong total length, a chunk whose row count is not
/// a whole number of sequence positions, or a chunk list that over- or
/// under-runs `s_total`.
#[allow(
    clippy::indexing_slicing,
    reason = "every (bi,si,h) base + d is bounded by the length and partition checks above"
)]
pub(super) fn transpose_chunked_seq_heads(
    src: &[f32],
    b: usize,
    s_total: usize,
    kv_h: usize,
    d: usize,
    chunk_rows: impl IntoIterator<Item = usize>,
) -> Result<Vec<f32>> {
    let total = b * kv_h * s_total * d;
    if src.len() != total {
        return Err(Error::Quant(format!(
            "transpose_chunked_seq_heads: buffer holds {} elems but \
             [B={b}, kv_h={kv_h}, S={s_total}, D={d}] implies {total}",
            src.len()
        )));
    }
    let mut out = vec![0.0_f32; total];
    let rows_per_seq = b * kv_h;
    if rows_per_seq == 0 || d == 0 {
        // Nothing to place: `total` is 0, so the empty output is already right.
        return Ok(out);
    }
    let mut s_off = 0usize;
    let mut src_off = 0usize;
    for rows in chunk_rows {
        if !rows.is_multiple_of(rows_per_seq) {
            return Err(Error::Quant(format!(
                "transpose_chunked_seq_heads: chunk holds {rows} rows, not a whole number of \
                 sequence positions at B={b}, kv_h={kv_h}"
            )));
        }
        let s_chunk = rows / rows_per_seq;
        if s_off + s_chunk > s_total {
            return Err(Error::Quant(format!(
                "transpose_chunked_seq_heads: chunks over-run S={s_total} at sequence offset \
                 {s_off} (+{s_chunk})"
            )));
        }
        for bi in 0..b {
            for si in 0..s_chunk {
                for h in 0..kv_h {
                    let src_base = src_off + ((bi * s_chunk + si) * kv_h + h) * d;
                    let dst_base = ((bi * kv_h + h) * s_total + s_off + si) * d;
                    out[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
                }
            }
        }
        s_off += s_chunk;
        src_off += rows * d;
    }
    if s_off != s_total {
        return Err(Error::Quant(format!(
            "transpose_chunked_seq_heads: chunks cover {s_off} sequence positions but S={s_total}"
        )));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "seq_layout_tests.rs"]
mod seq_layout_tests;
