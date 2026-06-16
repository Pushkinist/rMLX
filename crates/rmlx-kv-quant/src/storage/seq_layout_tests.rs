//! Deterministic index-math tests for the head↔sequence reorder helpers.
//!
//! These model the flat-buffer ordering shared by every quantized KV storage
//! struct: a chunk stored head-major (`[B, kv_h, new_seq, D]`) versus
//! sequence-major (`[B, new_seq, kv_h, D]`), accumulated at a sequence offset
//! across appends. They prove the pre-fix head-major store + head-major reshape
//! corrupts a multi-append, multi-head cache, while the sequence-major reorder
//! round-trips exactly for every append pattern.

use super::{transpose_heads_seq, transpose_seq_heads};

const B: usize = 1;

/// Distinct value per (head, token, dim) so any transposition is detectable.
fn val(h: usize, s: usize, d: usize) -> f32 {
    (h * 100_000 + s * 100 + d) as f32
}

/// Build the flat buffer the way the GPU `append` path does, for a list of
/// chunk lengths. `head_major_chunk = true` is the pre-fix ordering (chunk
/// stored `[b][h][s][d]`); `false` is the fixed ordering (chunk reordered to
/// `[b][s][h][d]` before quantizing).
fn build_buffer(appends: &[usize], kv_h: usize, d: usize, head_major_chunk: bool) -> Vec<f32> {
    let elems_per_seq = B * kv_h * d;
    let total: usize = appends.iter().sum::<usize>() * elems_per_seq;
    let mut buf = vec![f32::NAN; total];
    let mut prev = 0usize;
    for &new_seq in appends {
        // Source chunk is always head-major (`[B, kv_h, new_seq, D]`) — that is
        // the layout `append` receives. The fixed path reorders it to
        // sequence-major via `transpose_heads_seq` before storing.
        let mut chunk = vec![0.0_f32; elems_per_seq * new_seq];
        let mut i = 0usize;
        for _b in 0..B {
            for h in 0..kv_h {
                for sl in 0..new_seq {
                    for dd in 0..d {
                        chunk[i] = val(h, prev + sl, dd);
                        i += 1;
                    }
                }
            }
        }
        let stored = if head_major_chunk {
            chunk
        } else {
            transpose_heads_seq(&chunk, B, kv_h, new_seq, d)
        };
        let start = prev * elems_per_seq;
        buf[start..start + stored.len()].copy_from_slice(&stored);
        prev += new_seq;
    }
    buf
}

/// Old (buggy) read: reshape the flat prefix directly to `[B, kv_h, S, D]`.
fn read_head_major(buf: &[f32], kv_h: usize, s_total: usize, d: usize) -> f32 {
    let mut m = 0.0_f32;
    for h in 0..kv_h {
        for s in 0..s_total {
            for dd in 0..d {
                let idx = ((h * s_total) + s) * d + dd;
                m = m.max((buf[idx] - val(h, s, dd)).abs());
            }
        }
    }
    m
}

/// New (fixed) read: reshape the flat prefix to `[B, S, kv_h, D]`, then reorder
/// heads↔seq back to `[B, kv_h, S, D]` via `transpose_seq_heads`.
fn read_seq_major(buf: &[f32], kv_h: usize, s_total: usize, d: usize) -> f32 {
    let logical = transpose_seq_heads(buf, B, s_total, kv_h, d);
    let mut m = 0.0_f32;
    for h in 0..kv_h {
        for s in 0..s_total {
            for dd in 0..d {
                let idx = ((h * s_total) + s) * d + dd;
                m = m.max((logical[idx] - val(h, s, dd)).abs());
            }
        }
    }
    m
}

#[test]
fn head_major_store_corrupts_multi_append_multi_head() {
    // Single chunk: head-major store + head-major read agree (cold-prefill ok).
    let buf = build_buffer(&[3], 2, 4, true);
    assert_eq!(
        read_head_major(&buf, 2, 3, 4),
        0.0,
        "single-shot cold prefill exact"
    );
    // kv_h == 1 control: two appends still agree (no head axis to transpose).
    let buf = build_buffer(&[2, 1], 1, 4, true);
    assert_eq!(read_head_major(&buf, 1, 3, 4), 0.0, "kv_h=1 control exact");
    // kv_h > 1 + two appends: head transposition corrupts the read.
    let buf = build_buffer(&[2, 1], 2, 4, true);
    let bug = read_head_major(&buf, 2, 3, 4);
    assert!(
        bug > 0.0,
        "expected multi-head multi-append corruption, got {bug}"
    );
}

#[test]
fn seq_major_reorder_is_exact_for_all_append_patterns() {
    for &(appends, kv_h, d) in &[
        (&[3usize][..], 2usize, 4usize), // single-shot cold prefill
        (&[2, 1][..], 1, 4),             // kv_h = 1 control
        (&[2, 1][..], 2, 4),             // the bug case
        (&[2, 2][..], 3, 4),             // multi-head, even split
        (&[5, 3, 1][..], 4, 8),          // three appends, 4 heads
        (&[1, 1, 1, 1][..], 8, 4),       // per-token decode, 8 heads
    ] {
        let s_total: usize = appends.iter().sum();
        let buf = build_buffer(appends, kv_h, d, false);
        let m = read_seq_major(&buf, kv_h, s_total, d);
        assert_eq!(
            m, 0.0,
            "fixed path exact for appends={appends:?} kv_h={kv_h} d={d}, got {m}"
        );
    }
}

#[test]
fn reorder_round_trips_identity() {
    // transpose_seq_heads ∘ transpose_heads_seq == identity for any shape.
    let (kv_h, s, d) = (3usize, 5usize, 4usize);
    let mut src = vec![0.0_f32; B * kv_h * s * d];
    for (i, v) in src.iter_mut().enumerate() {
        *v = i as f32;
    }
    let seq = transpose_heads_seq(&src, B, kv_h, s, d);
    let back = transpose_seq_heads(&seq, B, s, kv_h, d);
    assert_eq!(src, back, "reorder round-trip must be identity");
}
