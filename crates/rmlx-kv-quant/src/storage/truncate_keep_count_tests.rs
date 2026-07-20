//! Unit tests for [`crate::storage::truncate_keep_count`] — the shared
//! row/sequence unit-conversion helper behind every rotor/iso K and V store's
//! `truncate_to` (#284).
//!
//! Root cause: each per-append block's `n_tokens` counts **rows**
//! (`b * kv_h * seq_of_block`), not raw sequence positions. `truncate_to(n)`
//! takes `n` as a **sequence** target, so `n` must be scaled to `n * b *
//! kv_h` before it is compared against the cumulative `n_tokens`. At
//! `b * kv_h == 1` the two units coincide, which is how the bug shipped
//! unnoticed — every `kv_h > 1` store silently undercounted.

use crate::storage::truncate_keep_count;

/// At `b * kv_h == 1` each block's `n_tokens` already equals its sequence
/// length, so rows and sequence positions coincide — the historical
/// (masked) case. No regression here after the fix.
#[test]
fn truncate_keep_count_kv_h_one_is_identity() {
    let shape = [1_i32, 1, 0, 8];
    // Four one-token blocks; n_tokens == 1 each (b * kv_h * 1).
    let blocks = [1_usize, 1, 1, 1];
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 2), 2);
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 0), 0);
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 4), 4);
}

/// At `kv_h > 1` each block's `n_tokens` is inflated by `b * kv_h` — the
/// target must be converted to row units before comparison, or the walk
/// overshoots early and drops blocks that should be kept (#284).
#[test]
fn truncate_keep_count_kv_h_gt_1_converts_to_rows() {
    let kv_h = 4_usize;
    let shape = [1_i32, kv_h as i32, 0, 8];
    // Four one-token blocks; each holds b * kv_h = 4 rows.
    let blocks = [4_usize, 4, 4, 4];

    // Keep 2 of 4 tokens -> row target = 2 * 4 = 8 -> keep 2 blocks.
    assert_eq!(
        truncate_keep_count(blocks.iter().copied(), &shape, 2),
        2,
        "must keep 2 blocks (row target 8), not floor(2/4)=0 blocks"
    );
    // Keep all 4 tokens -> row target = 16 -> keep all 4 blocks.
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 4), 4);
    // Keep 0 tokens -> keep no blocks.
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 0), 0);
}

/// Mutation guard: comparing `n_tokens` against the raw sequence target
/// (the pre-#284-fix behaviour) undercounts at `kv_h > 1`. This test pins
/// down what that regression looks like so it is visible without hand-
/// reverting the source.
#[test]
fn truncate_keep_count_kv_h_gt_1_would_undercount_with_raw_comparison() {
    let n_usize = 2_usize; // raw seq target, NOT converted to rows
    let blocks = [4_usize, 4, 4, 4]; // kv_h=4 rows per one-token block
    let mut acc = 0_usize;
    let mut keep_buggy = 0_usize;
    for (i, tokens) in blocks.iter().copied().enumerate() {
        if acc + tokens <= n_usize {
            acc += tokens;
            keep_buggy = i + 1;
        } else {
            break;
        }
    }
    assert_eq!(
        keep_buggy, 0,
        "the pre-#284-fix raw comparison keeps 0 of 4 blocks for a target of 2 tokens"
    );

    let kv_h = 4_usize;
    let shape = [1_i32, kv_h as i32, 0, 8];
    let fixed = truncate_keep_count(blocks.iter().copied(), &shape, 2);
    assert_eq!(
        fixed, 2,
        "the fixed helper keeps 2 blocks for the same input"
    );
}

/// A degenerate `b * kv_h == 0` shape (nothing appended yet) keeps no blocks.
#[test]
fn truncate_keep_count_zero_factor_keeps_nothing() {
    let shape = [0_i32, 4, 0, 8];
    let blocks = [4_usize, 4];
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 2), 0);
}

/// Multi-batch (`b > 1`) also inflates `n_tokens` and must be converted the
/// same way as `kv_h > 1`.
#[test]
fn truncate_keep_count_batch_gt_1_converts_to_rows() {
    let b = 2_usize;
    let shape = [b as i32, 1, 0, 8];
    // Four one-token blocks; each holds b * kv_h = 2 rows.
    let blocks = [2_usize, 2, 2, 2];
    assert_eq!(truncate_keep_count(blocks.iter().copied(), &shape, 3), 3);
}
