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

#[cfg(test)]
#[path = "seq_layout_tests.rs"]
mod seq_layout_tests;
