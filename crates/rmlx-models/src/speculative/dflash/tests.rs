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

// --- DFlashRoundState rollback round-trip (GDN snapshot/restore) ---

use rmlx_mlx::{Array, Dtype};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn arr(vals: &[f32]) -> Array {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[vals.len() as i32], Dtype::F32).expect("from_bytes")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn read(a: &Array) -> Vec<f32> {
    Array::eval(a).expect("materialise");
    a.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Snapshot a GDN cache before a round, mutate it (simulate a draft round
/// advancing the recurrence), then restore on partial rejection — the
/// cache must come back to the pre-round state exactly.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn round_state_rollback_round_trip() {
    let mut caches = vec![LinearAttnCache {
        conv_state: Some(arr(&[1.0, 2.0, 3.0])),
        delta_state: Some(arr(&[4.0, 5.0])),
    }];
    let round = DFlashRoundState::snapshot(&caches).expect("snapshot");
    assert_eq!(round.len(), 1);
    assert!(!round.is_empty());

    // Simulate the draft round advancing the recurrent state.
    caches[0].conv_state = Some(arr(&[9.0, 9.0, 9.0]));
    caches[0].delta_state = Some(arr(&[8.0, 8.0]));

    // Partial rejection -> restore.
    for (c, snap) in caches.iter_mut().zip(round.into_snapshots()) {
        c.restore_snapshot(snap);
    }
    assert_eq!(
        read(caches[0].conv_state.as_ref().unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        read(caches[0].delta_state.as_ref().unwrap()),
        vec![4.0, 5.0]
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn round_state_empty_for_non_gdn() {
    let caches: Vec<LinearAttnCache> = vec![];
    let round = DFlashRoundState::snapshot(&caches).expect("snapshot");
    assert!(round.is_empty());
    assert_eq!(round.len(), 0);
}

/// Compile-check: the public DFlash surface exists with expected sigs.
#[test]
fn dflash_module_compiles() {
    let _load = DFlashDrafter::load;
    let _bs = dflash_next_block_size;
    let _walk = walk_block_greedy;
    let _snap = DFlashRoundState::snapshot;
    let _ = (_load, _bs, _walk, _snap);
}

// --- unread_tensor_refusal ---

/// A snapshot every tensor of which was read loads; one carrying tensors this
/// loader has no code for is refused, naming them.
///
/// Both directions, because the quiet direction is what keeps the check from
/// decaying into noise a reader learns to skip. Mutation this fails on:
/// `!consumed.contains(name)` -> `consumed.contains(name)`, which refuses the
/// supported checkpoint and admits the unsupported one.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions: panicking on unexpected values is intentional"
)]
fn a_snapshot_this_loader_only_half_reads_is_refused_and_a_whole_one_is_not() {
    let read_by_the_loader = ["fc.weight", "hidden_norm.weight", "norm.weight"];
    let consumed: HashSet<String> = read_by_the_loader.iter().map(|s| (*s).to_owned()).collect();

    // Every tensor consumed — the shape of the DFlash 1 checkpoint.
    let whole: Vec<String> = read_by_the_loader.iter().map(|s| (*s).to_owned()).collect();
    assert!(
        unread_tensor_refusal(&whole, &consumed).is_ok(),
        "a snapshot the loader reads entirely must load"
    );

    // Extra weight families — the shape of the DFlash 2 checkpoint.
    let mut partial = whole;
    partial.push("candidate_selector.successor_codebook".to_owned());
    partial.push("layers.0.attention_conv.base_kernel".to_owned());
    let err = unread_tensor_refusal(&partial, &consumed).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains('2'),
        "refusal must count the unread tensors: {msg}"
    );
    assert!(
        msg.contains("candidate_selector.successor_codebook")
            && msg.contains("layers.0.attention_conv.base_kernel"),
        "refusal must name the unread tensors: {msg}"
    );
    assert!(
        !msg.contains("fc.weight"),
        "a consumed tensor must not be reported unread: {msg}"
    );
}
