//! DFlash drafter unit tests.

use super::*;

// --- dflash_next_block_size schedule ---

#[test]
fn block_size_clamped_to_budget() {
    // No history; budget caps below the requested ceiling.
    assert_eq!(dflash_next_block_size(&[], 16, 4, false), 4);
    assert_eq!(dflash_next_block_size(&[], 16, 100, false), 16);
}

#[test]
fn block_size_prefer_requested_short_circuits() {
    // Even with weak history, prefer_requested returns the (budget-capped) ceiling.
    let weak = [(0usize, 8usize), (0, 8), (0, 8)];
    assert_eq!(dflash_next_block_size(&weak, 16, 100, true), 16);
}

#[test]
fn block_size_one_or_zero_passthrough() {
    assert_eq!(dflash_next_block_size(&[(8, 8)], 1, 100, false), 1);
    assert_eq!(dflash_next_block_size(&[(8, 8)], 16, 1, false), 1);
    assert_eq!(dflash_next_block_size(&[(8, 8)], 16, 0, false), 0);
}

#[test]
fn block_size_backs_off_hard_on_weak_acceptance() {
    // accept_rate < 0.30 -> halve when current >= 8.
    // last drafted 15 -> current = min(16, 16) = 16, >= 8 -> 16/2 = 8.
    let weak = [(2usize, 15usize), (1, 15), (0, 15)];
    let next = dflash_next_block_size(&weak, 16, 100, false);
    assert_eq!(next, 8);
}

#[test]
fn block_size_backs_off_small_when_current_below_8() {
    // current < 8 and weak -> current - 2, floored at min_total (4).
    // last drafted 5 -> current = min(16, 6) = 6; weak -> 6-2 = 4.
    let weak = [(0usize, 5usize), (0, 5)];
    let next = dflash_next_block_size(&weak, 16, 100, false);
    assert_eq!(next, 4);
}

#[test]
fn block_size_grows_on_strong_full_hits() {
    // accept_rate >= 0.85 and full_hit_rate >= 0.75 -> current + 2.
    // all full hits, last drafted 6 -> current = 7 -> 7+2 = 9.
    let strong = [(6usize, 6usize), (6, 6), (6, 6), (6, 6)];
    let next = dflash_next_block_size(&strong, 16, 100, false);
    assert_eq!(next, 9);
}

#[test]
fn block_size_holds_on_moderate_acceptance() {
    // 0.50 <= accept_rate < 0.85 -> hold at current.
    // drafted 10 each, accepted 6 -> rate 0.6; last drafted 10 -> current = 11.
    let mod_hist = [(6usize, 10usize), (6, 10), (6, 10)];
    let next = dflash_next_block_size(&mod_hist, 16, 100, false);
    assert_eq!(next, 11);
}

// --- walk_block_greedy acceptance ---

#[test]
fn walk_all_accepted_emits_bonus() {
    let draft = [10, 11, 12];
    let target = [10, 11, 12, 99]; // n_draft + 1 predictions
    let (acc, emit) = walk_block_greedy(&draft, &target, 8);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11, 12, 99]);
}

#[test]
fn walk_partial_accept_emits_correction() {
    let draft = [10, 11, 12];
    let target = [10, 11, 55, 0]; // diverge at pos 2
    let (acc, emit) = walk_block_greedy(&draft, &target, 8);
    assert_eq!(acc, 2);
    assert_eq!(emit, vec![10, 11, 55]);
}

#[test]
fn walk_zero_accept_emits_only_correction() {
    let draft = [10, 11];
    let target = [42, 0, 0];
    let (acc, emit) = walk_block_greedy(&draft, &target, 8);
    assert_eq!(acc, 0);
    assert_eq!(emit, vec![42]);
}

#[test]
fn walk_respects_budget() {
    let draft = [10, 11, 12];
    let target = [10, 11, 12, 99];
    let (acc, emit) = walk_block_greedy(&draft, &target, 2);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11]);
}

/// Compile-check: the public DFlash surface exists with expected sigs.
#[test]
fn dflash_module_compiles() {
    let _load = DFlashDrafter::load;
    let _bs = dflash_next_block_size;
    let _walk = walk_block_greedy;
    let _ = (_load, _bs, _walk);
}
