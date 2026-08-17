//! Deterministic index-math tests for the head↔sequence reorder helpers.
//!
//! These model the flat-buffer ordering shared by every quantized KV storage
//! struct: a chunk stored head-major (`[B, kv_h, new_seq, D]`) versus
//! sequence-major (`[B, new_seq, kv_h, D]`), accumulated at a sequence offset
//! across appends. They prove the pre-fix head-major store + head-major reshape
//! corrupts a multi-append, multi-head cache, while the sequence-major reorder
//! round-trips exactly for every append pattern.

use super::{
    head_major_token_order, transpose_chunked_seq_heads, transpose_heads_seq, transpose_seq_heads,
};

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

// ── Chunked reorder (`B` > 1) ────────────────────────────────────────────────

/// Distinct value per `(batch, head, token, dim)` — the batch axis the
/// single-`B` helpers above do not exercise.
fn bval(bi: usize, h: usize, s: usize, d: usize) -> f32 {
    (bi * 10_000_000 + h * 100_000 + s * 100 + d) as f32
}

/// Head-major `[B, kv_h, S, D]` reference, built straight from index math — no
/// reorder helper involved, so it is an independent oracle.
fn head_major_reference(b: usize, kv_h: usize, s_total: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * kv_h * s_total * d];
    for bi in 0..b {
        for h in 0..kv_h {
            for s in 0..s_total {
                for dd in 0..d {
                    out[((bi * kv_h + h) * s_total + s) * d + dd] = bval(bi, h, s, dd);
                }
            }
        }
    }
    out
}

/// The buffer a block-accumulating store decodes to: one sequence-major
/// `[B, S_chunk, kv_h, D]` chunk per append, concatenated in append order.
fn chunked_seq_major_buffer(b: usize, kv_h: usize, appends: &[usize], d: usize) -> Vec<f32> {
    let mut buf = Vec::new();
    let mut s0 = 0usize;
    for &n in appends {
        for bi in 0..b {
            for s in 0..n {
                for h in 0..kv_h {
                    for dd in 0..d {
                        buf.push(bval(bi, h, s0 + s, dd));
                    }
                }
            }
        }
        s0 += n;
    }
    buf
}

#[test]
fn chunked_reorder_is_exact_at_every_batch_and_split() {
    for &(b, kv_h, appends, d) in &[
        (1usize, 1usize, &[3usize][..], 4usize), // degenerate
        (1, 4, &[2, 1][..], 8),                  // B = 1 multi-chunk (unchanged behaviour)
        (2, 1, &[5][..], 4),                     // B > 1, single chunk
        (2, 1, &[2, 3][..], 4),                  // B > 1, two chunks — the defect
        (2, 2, &[2, 3][..], 4),                  // B > 1, GQA
        (3, 4, &[1, 1, 1, 2][..], 8),            // per-token decode after a chunk
    ] {
        let s_total: usize = appends.iter().sum();
        let buf = chunked_seq_major_buffer(b, kv_h, appends, d);
        let got = transpose_chunked_seq_heads(
            &buf,
            b,
            s_total,
            kv_h,
            d,
            appends.iter().map(|&n| b * kv_h * n),
        )
        .expect("well-formed chunk partition");
        assert_eq!(
            got,
            head_major_reference(b, kv_h, s_total, d),
            "chunked reorder must be exact for b={b} kv_h={kv_h} appends={appends:?} d={d}"
        );
    }
}

/// The whole-buffer reorder is what the block stores used to call. It agrees
/// with the chunked one at `B == 1` and disagrees the moment `B > 1` meets more
/// than one chunk — this is what makes the chunked helper load-bearing rather
/// than a rename.
#[test]
fn whole_buffer_reorder_only_matches_at_b_1_or_one_chunk() {
    let d = 4;
    let agree = |b: usize, kv_h: usize, appends: &[usize]| -> bool {
        let s_total: usize = appends.iter().sum();
        let buf = chunked_seq_major_buffer(b, kv_h, appends, d);
        let whole = transpose_seq_heads(&buf, b, s_total, kv_h, d);
        let chunked = transpose_chunked_seq_heads(
            &buf,
            b,
            s_total,
            kv_h,
            d,
            appends.iter().map(|&n| b * kv_h * n),
        )
        .expect("well-formed chunk partition");
        whole == chunked
    };
    assert!(agree(1, 2, &[2, 3]), "B = 1 multi-chunk: the two agree");
    assert!(agree(2, 2, &[5]), "single chunk: the two agree at any B");
    assert!(
        !agree(2, 1, &[2, 3]),
        "B > 1 with two chunks: the whole-buffer reorder must disagree — if it does not, \
         this test no longer proves the chunked helper is needed"
    );
    assert!(
        !agree(3, 2, &[1, 1, 1]),
        "B > 1 per-token decode: disagrees"
    );
}

#[test]
fn chunked_reorder_rejects_a_partition_that_does_not_add_up() {
    let (b, kv_h, d) = (2usize, 2usize, 4usize);
    let buf = chunked_seq_major_buffer(b, kv_h, &[2, 3], d);

    // Chunks cover fewer sequence positions than `s_total` declares.
    let err = transpose_chunked_seq_heads(&buf, b, 5, kv_h, d, [b * kv_h * 2, b * kv_h * 2])
        .expect_err("under-running chunks must be rejected");
    assert!(
        err.to_string().contains("cover 4 sequence positions"),
        "expected the coverage error, got: {err}"
    );

    // A chunk whose row count is not a whole number of sequence positions.
    let err = transpose_chunked_seq_heads(&buf, b, 5, kv_h, d, [b * kv_h * 2, b * kv_h * 3 - 1, 1])
        .expect_err("a ragged chunk must be rejected");
    assert!(
        err.to_string()
            .contains("not a whole number of sequence positions"),
        "expected the ragged-chunk error, got: {err}"
    );

    // Buffer length disagrees with the declared shape.
    let err = transpose_chunked_seq_heads(&buf[..buf.len() - d], b, 5, kv_h, d, [b * kv_h * 5])
        .expect_err("a short buffer must be rejected");
    assert!(
        err.to_string().contains("implies"),
        "expected the length error, got: {err}"
    );
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

// ── Head-major token permutation ─────────────────────────────────────────────

/// `head_major_token_order` must agree with `transpose_chunked_seq_heads`: both
/// are the same reorder, one applied to the payload rows on the way in and one
/// to the decoded values on the way out. Permuting rows by `perm` and reading
/// the result as head-major must equal reordering the sequence-major decode.
///
/// Mutation check: drop the `s_off` term from the destination index in
/// `head_major_token_order` and this goes red at every multi-chunk case.
#[test]
fn head_major_token_order_matches_the_output_reorder() {
    for &(b, kv_h, appends, d) in &[
        (1usize, 1usize, &[3usize][..], 4usize),
        (1, 4, &[2, 1][..], 8),
        (2, 1, &[2, 3][..], 4),
        (2, 2, &[2, 3][..], 4),
        (3, 4, &[1, 1, 1, 2][..], 8),
    ] {
        let s_total: usize = appends.iter().sum();
        let seq_major = chunked_seq_major_buffer(b, kv_h, appends, d);
        let rows = appends.iter().map(|&n| b * kv_h * n);
        let perm = head_major_token_order(b, s_total, kv_h, rows.clone())
            .expect("well-formed chunk partition");

        // Route A: reorder the decoded values.
        let via_output = transpose_chunked_seq_heads(&seq_major, b, s_total, kv_h, d, rows)
            .expect("well-formed chunk partition");
        // Route B: place each source row at `perm[row]` and read straight off.
        let mut via_input = vec![0.0_f32; b * kv_h * s_total * d];
        for (row, &dst) in perm.iter().enumerate() {
            via_input[dst * d..(dst + 1) * d].copy_from_slice(&seq_major[row * d..(row + 1) * d]);
        }

        assert_eq!(
            via_input, via_output,
            "input permutation and output reorder must agree for b={b} kv_h={kv_h} \
             appends={appends:?} d={d}"
        );
        assert_eq!(
            via_input,
            head_major_reference(b, kv_h, s_total, d),
            "and both must equal the index-math reference"
        );
    }
}

/// The permutation is a bijection onto `[0, B * kv_h * S)` — a duplicate or a
/// gap would leave one token slot written twice and another holding the
/// allocation's zeros, which is the silent-corruption shape this whole reorder
/// exists to avoid.
#[test]
fn head_major_token_order_is_a_bijection() {
    for &(b, kv_h, appends) in &[
        (2usize, 2usize, &[2usize, 3][..]),
        (3, 1, &[1, 1, 4][..]),
        (1, 5, &[7][..]),
    ] {
        let s_total: usize = appends.iter().sum();
        let perm = head_major_token_order(b, s_total, kv_h, appends.iter().map(|&n| b * kv_h * n))
            .expect("well-formed chunk partition");
        let mut seen = vec![false; b * kv_h * s_total];
        assert_eq!(perm.len(), seen.len(), "one entry per token row");
        for &dst in &perm {
            assert!(!seen[dst], "destination {dst} written twice");
            seen[dst] = true;
        }
        assert!(seen.iter().all(|&s| s), "every destination slot is written");
    }
}

#[test]
fn head_major_token_order_rejects_a_partition_that_does_not_add_up() {
    let err = head_major_token_order(2, 5, 2, [8usize, 8])
        .expect_err("under-running chunks must be rejected");
    assert!(
        err.to_string().contains("cover 4 sequence positions"),
        "expected the coverage error, got: {err}"
    );
    let err = head_major_token_order(2, 5, 2, [8usize, 11, 1])
        .expect_err("a ragged chunk must be rejected");
    assert!(
        err.to_string()
            .contains("not a whole number of sequence positions"),
        "expected the ragged-chunk error, got: {err}"
    );
}
