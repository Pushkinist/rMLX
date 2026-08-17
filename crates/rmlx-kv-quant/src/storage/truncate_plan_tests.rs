//! Unit tests for [`crate::storage::truncate_plan`] — the shared block-cut
//! planner behind every rotor/iso K and V store's `truncate_to`.
//!
//! Three separate hazards live here.
//!
//! **Row/sequence units.** Each per-append block's `n_tokens` counts **rows**
//! (`b * kv_h * seq_of_block`), not raw sequence positions. `truncate_to(n)`
//! takes `n` as a **sequence** target, so `n` must be scaled to `n * b * kv_h`
//! before it is compared against the cumulative `n_tokens`. At `b * kv_h == 1`
//! the two units coincide, which is how that bug shipped unnoticed — every
//! `kv_h > 1` store silently undercounted. The fixtures below use `kv_h > 1`
//! wherever the assertion would otherwise hide behind that degeneracy.
//!
//! **Mid-block cuts.** A block spans one whole append, and a speculative-decode
//! partial accept cuts inside the verifier's multi-token chunk. Dropping the
//! whole block throws the accepted prefix away with the rejected tail, leaving
//! `blocks` short of `shape[2]` — a state only a live GPU ring can repair. The
//! planner therefore splits the trailing block instead.
//!
//! **`b > 1` must NOT split.** Every store decodes the concatenation of its
//! blocks as one `[B, S_total, kv_h, D]` run, which is only a valid reading at
//! `b == 1` or with a single block. Splitting at `b > 1` would replace a loud
//! blocks-short-of-`shape[2]` error with a silently scrambled two-block store,
//! so the planner drops the block there and lets the guard report the gap. The
//! store-level proof of the scrambling is
//! `quant_rotor_v3_tests::quant_rotor_v3_truncate_at_b_gt_1_stays_loud`.

use crate::storage::{apply_truncate_plan, retain_rows_in, truncate_plan, BlockRows};

/// A minimal [`BlockRows`] stand-in: one payload buffer at `stride` values per
/// row, filled with consecutive integers so a mis-sliced cut is visible.
#[derive(Debug, PartialEq, Eq)]
struct FakeBlock {
    payload: Vec<u32>,
    rows: usize,
}

impl FakeBlock {
    fn new(first_row: u32, rows: usize, stride: usize) -> Self {
        let payload = (0..rows * stride)
            .map(|i| first_row * 1000 + i as u32)
            .collect();
        Self { payload, rows }
    }
}

impl BlockRows for FakeBlock {
    fn retain_rows(&mut self, rows: usize) -> bool {
        if !crate::storage::rows_split_ok(&[self.payload.len()], self.rows, rows) {
            return false;
        }
        retain_rows_in(&mut self.payload, self.rows, rows);
        self.rows = rows;
        true
    }
}

fn keep_counts(blocks: &[FakeBlock]) -> Vec<usize> {
    blocks.iter().map(|b| b.rows).collect()
}

// ── Row / sequence unit conversion ───────────────────────────────────────────

/// At `b * kv_h == 1` each block's `n_tokens` already equals its sequence
/// length, so rows and sequence positions coincide — the historical (masked)
/// case.
#[test]
fn plan_kv_h_one_is_identity() {
    let shape = [1_i32, 1, 0, 8];
    let blocks = [1_usize, 1, 1, 1];
    assert_eq!(truncate_plan(blocks.iter().copied(), &shape, 2).keep, 2);
    assert_eq!(truncate_plan(blocks.iter().copied(), &shape, 0).keep, 0);
    assert_eq!(truncate_plan(blocks.iter().copied(), &shape, 4).keep, 4);
}

/// At `kv_h > 1` each block's `n_tokens` is inflated by `b * kv_h` — the target
/// must be converted to row units before comparison, or the walk overshoots
/// early and drops blocks that should be kept.
#[test]
fn plan_kv_h_gt_1_converts_to_rows() {
    let kv_h = 4_usize;
    let shape = [1_i32, kv_h as i32, 0, 8];
    let blocks = [4_usize, 4, 4, 4];

    assert_eq!(
        truncate_plan(blocks.iter().copied(), &shape, 2).keep,
        2,
        "must keep 2 blocks (row target 8), not floor(2/4)=0 blocks"
    );
    assert_eq!(truncate_plan(blocks.iter().copied(), &shape, 4).keep, 4);
    assert_eq!(truncate_plan(blocks.iter().copied(), &shape, 0).keep, 0);
}

/// Mutation guard for the unit conversion: comparing `n_tokens` against the raw
/// sequence target undercounts at `kv_h > 1`. Pinned here so the regression is
/// visible without hand-reverting the source.
#[test]
fn plan_kv_h_gt_1_would_undercount_with_raw_comparison() {
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
        "the raw comparison keeps 0 of 4 blocks for a target of 2 tokens"
    );

    let shape = [1_i32, 4, 0, 8];
    assert_eq!(
        truncate_plan(blocks.iter().copied(), &shape, 2).keep,
        2,
        "the planner keeps 2 blocks for the same input"
    );
}

/// A degenerate `b * kv_h == 0` shape (nothing appended yet) keeps no blocks.
#[test]
fn plan_zero_factor_keeps_nothing() {
    let shape = [0_i32, 4, 0, 8];
    let blocks = [4_usize, 4];
    let plan = truncate_plan(blocks.iter().copied(), &shape, 2);
    assert_eq!(plan.keep, 0);
    assert_eq!(plan.partial_rows, None);
}

/// Multi-batch (`b > 1`) also inflates `n_tokens` and must be converted the
/// same way as `kv_h > 1`.
#[test]
fn plan_batch_gt_1_converts_to_rows() {
    let shape = [2_i32, 1, 0, 8];
    let blocks = [2_usize, 2, 2, 2];
    assert_eq!(truncate_plan(blocks.iter().copied(), &shape, 3).keep, 3);
}

// ── Mid-block cuts ───────────────────────────────────────────────────────────

/// A cut landing inside the trailing block splits it: the leading blocks stay
/// whole and the trailing block keeps exactly the accepted prefix.
///
/// The numbers mirror a speculative round: a 46-position prefill block, a
/// 5-position verifier chunk, 4 of the 5 accepted.
#[test]
fn plan_splits_the_trailing_block_at_a_mid_block_cut() {
    let shape = [1_i32, 1, 0, 8];
    let blocks = [46_usize, 5];
    let plan = truncate_plan(blocks.iter().copied(), &shape, 50);
    assert_eq!(plan.keep, 1, "the 46-row prefill block stays whole");
    assert_eq!(
        plan.partial_rows,
        Some(4),
        "4 of the verifier chunk's 5 rows are retained"
    );
}

/// The same cut at `kv_h > 1`: the retained count is `keep_seq * kv_h` rows, not
/// `keep_seq` rows.
#[test]
fn plan_partial_rows_are_in_row_units_at_kv_h_gt_1() {
    let kv_h = 4_usize;
    let shape = [1_i32, kv_h as i32, 0, 8];
    // One 10-position block (40 rows) then one 5-position block (20 rows).
    let blocks = [40_usize, 20];
    let plan = truncate_plan(blocks.iter().copied(), &shape, 13);
    assert_eq!(plan.keep, 1);
    assert_eq!(
        plan.partial_rows,
        Some(12),
        "3 sequence positions x kv_h = 4"
    );
}

/// A cut exactly on a block boundary produces no partial — the historical
/// whole-block path.
#[test]
fn plan_boundary_cut_has_no_partial() {
    let shape = [1_i32, 1, 0, 8];
    let blocks = [46_usize, 5];
    let plan = truncate_plan(blocks.iter().copied(), &shape, 46);
    assert_eq!(plan.keep, 1);
    assert_eq!(plan.partial_rows, None);
}

/// The plan applied to real blocks keeps exactly the retained rows' payload,
/// value for value.
///
/// The oracle is the untruncated block's own payload sliced at the expected
/// boundary — it shares no arithmetic with the planner, which never sees the
/// payload.
#[test]
fn apply_plan_keeps_exactly_the_retained_payload() {
    let shape = [1_i32, 1, 0, 8];
    let stride = 3_usize;
    let mut blocks = vec![FakeBlock::new(0, 46, stride), FakeBlock::new(1, 5, stride)];
    let untouched_tail = blocks[1].payload.clone();

    let plan = truncate_plan(blocks.iter().map(|b| b.rows), &shape, 50);
    apply_truncate_plan(&mut blocks, &plan);

    assert_eq!(keep_counts(&blocks), vec![46, 4]);
    assert_eq!(
        blocks[1].payload,
        untouched_tail[..4 * stride],
        "the split block must keep the first 4 rows byte-for-byte"
    );
    let total_rows: usize = blocks.iter().map(|b| b.rows).sum();
    assert_eq!(total_rows, 50, "blocks must cover the truncation target");
}

/// Mutation guard: the pre-fix whole-block drop loses the accepted prefix.
///
/// Without this the split test above proves only that the new code agrees with
/// itself — this pins what the old behaviour did to the same input, and that it
/// leaves the store 4 rows short of `shape[2]`.
#[test]
fn whole_block_drop_leaves_the_store_short_of_the_target() {
    let shape = [1_i32, 1, 0, 8];
    let blocks = [46_usize, 5];
    let target = 50_usize;

    // The historical planner: keep only blocks that fit whole.
    let mut acc = 0_usize;
    let mut keep_whole_only = 0_usize;
    for (i, rows) in blocks.iter().copied().enumerate() {
        if acc + rows <= target {
            acc += rows;
            keep_whole_only = i + 1;
        } else {
            break;
        }
    }
    assert_eq!(keep_whole_only, 1);
    assert_eq!(
        acc, 46,
        "whole-block truncation covers 46 rows against a target of 50 — the 4-row \
         gap the rotor/iso reconciliation guards abort on"
    );

    let plan = truncate_plan(blocks.iter().copied(), &shape, target as i32);
    let covered: usize = blocks
        .iter()
        .copied()
        .take(plan.keep)
        .sum::<usize>()
        .saturating_add(plan.partial_rows.unwrap_or(0));
    assert_eq!(covered, target, "the planner covers the target exactly");
}

// ── Refusals ─────────────────────────────────────────────────────────────────

/// `b > 1` never splits, however cleanly the cut would divide.
///
/// Not a limitation of the planner — a correctness bound imposed by the decode
/// path. Every store reads its block concatenation as one `[B, S_total, kv_h,
/// D]` run, which interleaves batch elements once more than one block survives.
/// Splitting here would replace the loud blocks-short-of-`shape[2]` error with a
/// silently scrambled store; dropping keeps the error.
#[test]
fn plan_never_splits_at_batch_gt_1() {
    let (b, kv_h) = (2_usize, 2_usize);
    let shape = [b as i32, kv_h as i32, 0, 8];
    // One 4-position block: b * seq * kv_h = 16 rows. A cut at 3 positions is
    // mid-block and divides cleanly in row units — the split is refused anyway.
    let blocks = [16_usize];
    let plan = truncate_plan(blocks.iter().copied(), &shape, 3);
    assert_eq!(plan.keep, 0, "the block is dropped, not kept whole");
    assert_eq!(
        plan.partial_rows, None,
        "no split at b > 1 — the decode path cannot read a multi-block batched store"
    );

    // The same shape at b == 1 does split, so the refusal is keyed on `b` and
    // not on some other property of this fixture.
    let shape_b1 = [1_i32, kv_h as i32, 0, 8];
    let blocks_b1 = [8_usize]; // 4 positions x kv_h = 2
    let plan_b1 = truncate_plan(blocks_b1.iter().copied(), &shape_b1, 3);
    assert_eq!(plan_b1.partial_rows, Some(6));
}

/// A block whose row count is not a whole number of sequence positions stops the
/// walk at the last clean boundary.
///
/// Unreachable in practice — every `append` pushes `b * kv_h * seq` rows — but
/// the branch exists so a malformed block is dropped rather than cut on a
/// guessed layout, and its `break` semantics differ from the old helper's (which
/// would have kept such a block if it fit under the row target). Pinned so that
/// difference is deliberate.
#[test]
fn plan_stops_at_a_block_whose_rows_are_not_whole_positions() {
    let kv_h = 4_usize;
    let shape = [1_i32, kv_h as i32, 0, 8];
    // 8 rows = 2 positions, then 6 rows = 1.5 positions (malformed), then 4.
    let blocks = [8_usize, 6, 4];
    let plan = truncate_plan(blocks.iter().copied(), &shape, 4);
    assert_eq!(
        plan.keep, 1,
        "the walk stops at the malformed block, keeping only the clean prefix"
    );
    assert_eq!(plan.partial_rows, None);

    // The old helper's behaviour on the same input, for contrast: 8 + 6 = 14
    // rows fits under the 16-row target, so it kept the malformed block — and
    // its 1.5 positions then silently shifted every sequence index after it.
    let n_rows = 4 * kv_h;
    let mut acc = 0_usize;
    let mut keep_old = 0_usize;
    for (i, rows) in blocks.iter().copied().enumerate() {
        if acc + rows <= n_rows {
            acc += rows;
            keep_old = i + 1;
        } else {
            break;
        }
    }
    assert_eq!(
        keep_old, 2,
        "the old whole-block helper kept the malformed block; the planner stops before it"
    );
}

/// A cut past the block's own row count is refused before any buffer is
/// touched — the block is dropped whole, never half-cut.
#[test]
fn out_of_range_cut_is_refused_not_partially_applied() {
    let mut block = FakeBlock::new(0, 4, 3);
    let before = block.payload.clone();
    assert!(
        !block.retain_rows(5),
        "a cut past the block's 4 rows must be refused"
    );
    assert_eq!(block.payload, before, "a refused split must not mutate");
    assert_eq!(
        block.rows, 4,
        "a refused split must not change the row count"
    );
}

/// A block whose payload is not a whole number of rows cannot be split; the
/// caller drops it rather than cut it into an inconsistent state.
///
/// Dropping it is safe but **not** free: the store's `truncate_to` lowers
/// `shape[2]` to `n` regardless, so the result is the blocks-short-of-`shape[2]`
/// state that aborts the next `dequant`. This test asserts the size of that gap
/// rather than only the surviving block count, so the cost of the refusal is
/// visible here instead of being discovered at a consumer boundary. The refusal
/// also emits a `warn!` naming the same numbers.
///
/// `kv_h = 2` on purpose: the gap arithmetic below is in **row** units, and a
/// `b * kv_h == 1` fixture would let a sequence-vs-row unit slip pass.
#[test]
fn apply_plan_drops_an_unsplittable_block_and_leaves_a_named_gap() {
    let kv_h = 2_usize;
    let shape = [1_i32, kv_h as i32, 0, 8];
    let target_seq = 3_usize;
    let rows_per_seq = kv_h;
    // 2 positions (4 rows) then 3 positions (6 rows); cut at 3 positions.
    let mut blocks = vec![FakeBlock::new(0, 4, 3), FakeBlock::new(1, 6, 3)];
    // Corrupt the trailing block so its payload length is not a multiple of its
    // row count.
    blocks[1].payload.pop();

    let plan = truncate_plan(blocks.iter().map(|b| b.rows), &shape, target_seq as i32);
    assert_eq!(
        plan.partial_rows,
        Some(rows_per_seq),
        "the plan wanted 1 position x kv_h rows out of the trailing block"
    );
    apply_truncate_plan(&mut blocks, &plan);

    assert_eq!(
        keep_counts(&blocks),
        vec![4],
        "an unsplittable trailing block is dropped whole"
    );
    let covered: usize = blocks.iter().map(|b| b.rows).sum();
    assert_eq!(
        target_seq * rows_per_seq - covered,
        plan.partial_rows.unwrap_or(0),
        "the refusal leaves exactly the wanted rows uncovered — the gap a consumer \
         will abort on, and the number the warn! reports. Both sides are row counts; \
         the target is scaled by b * kv_h to get there"
    );
}

/// [`retain_rows_in`] leaves an empty buffer empty — an inactive sideband (the
/// rotor QJL residual) must not be resized into existence.
#[test]
fn retain_rows_in_leaves_an_empty_sideband_empty() {
    let mut empty: Vec<f32> = Vec::new();
    retain_rows_in(&mut empty, 5, 4);
    assert!(empty.is_empty());
}

/// [`retain_rows_in`] keeps a row prefix at every count, at a stride > 1.
#[test]
fn retain_rows_in_keeps_a_row_prefix_at_every_count() {
    let (rows, stride) = (7_usize, 3_usize);
    for keep in 0..=rows {
        let mut buf: Vec<u32> = (0..(rows * stride) as u32).collect();
        retain_rows_in(&mut buf, rows, keep);
        let expected: Vec<u32> = (0..(keep * stride) as u32).collect();
        assert_eq!(buf, expected, "row prefix wrong at keep={keep}");
    }
}
