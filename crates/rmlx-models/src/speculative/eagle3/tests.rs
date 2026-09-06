use super::*;

#[test]
fn block_size_clamped_to_budget() {
    assert_eq!(eagle3_next_block_size(5, 3), 3);
    assert_eq!(eagle3_next_block_size(5, 100), 5);
}

#[test]
fn block_size_one_or_zero_passthrough() {
    assert_eq!(eagle3_next_block_size(5, 1), 1);
    assert_eq!(eagle3_next_block_size(5, 0), 0);
}

#[test]
fn walk_all_accepted_emits_bonus() {
    let draft = [10, 11, 12];
    let target = [10, 11, 12, 99];
    let (acc, emit) = eagle3_walk(&draft, &target, 8);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11, 12, 99]);
}

#[test]
fn walk_partial_accept_emits_correction() {
    let draft = [10, 11, 12];
    let target = [10, 11, 55, 0];
    let (acc, emit) = eagle3_walk(&draft, &target, 8);
    assert_eq!(acc, 2);
    assert_eq!(emit, vec![10, 11, 55]);
}

#[test]
fn walk_zero_accept_emits_only_correction() {
    let draft = [10, 11];
    let target = [42, 0, 0];
    let (acc, emit) = eagle3_walk(&draft, &target, 8);
    assert_eq!(acc, 0);
    assert_eq!(emit, vec![42]);
}

#[test]
fn walk_respects_budget() {
    let draft = [10, 11, 12];
    let target = [10, 11, 12, 99];
    let (acc, emit) = eagle3_walk(&draft, &target, 2);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11]);
}

#[test]
fn d2t_remap_offsets_draft_to_target() {
    // target = draft + d2t[draft].
    let d2t = vec![0, 5, 100, -3];
    assert_eq!(draft_to_target(0, &d2t), 0);
    assert_eq!(draft_to_target(1, &d2t), 6);
    assert_eq!(draft_to_target(2, &d2t), 102);
    assert_eq!(draft_to_target(3, &d2t), 0); // 3 + (-3)
}

#[test]
fn d2t_empty_passes_through() {
    let d2t: Vec<i32> = Vec::new();
    assert_eq!(draft_to_target(42, &d2t), 42);
    assert_eq!(draft_to_target(248319, &d2t), 248319);
}

#[test]
fn d2t_out_of_range_passes_through() {
    let d2t = vec![1, 2];
    assert_eq!(draft_to_target(99, &d2t), 99);
}

/// Compile-check: the public EAGLE-3 surface exists with expected sigs.
#[test]
fn eagle3_module_compiles() {
    let _load = Eagle3Drafter::load;
    let _bs = eagle3_next_block_size;
    let _walk = eagle3_walk;
    let _d2t = draft_to_target;
    let _ffp = find_full_pos;
    let _gen = eagle3_generate;
    let _ = (_load, _bs, _walk, _d2t, _ffp, _gen);
}

// -----------------------------------------------------------------------
// Forward-call-counter tests (MEDIUM 3, ).
//
// These tests verify the loop invariants of `draft_block` and the token
// sequence fed to `forward_tokens_conditioned` in `accept_and_reseed`
// without loading any model weights. They use inline mock functions that
// replicate the loop logic with injected counters / recorders.
// -----------------------------------------------------------------------

/// Mirror of the `draft_block` loop with injected `forward_fn` / `greedy_fn`.
/// `forward_fn(tok)` → next hidden token id (stub: just returns `tok + 1`).
/// `greedy_fn()` → next draft token id.
/// Returns `(tokens, forward_call_count)`.
fn mock_draft_block(
    carry_tok: u32,
    precomputed_first_tok: Option<u32>,
    block_size: usize,
    mut forward_fn: impl FnMut(u32) -> u32,
    mut greedy_fn: impl FnMut() -> u32,
) -> (Vec<u32>, usize) {
    if block_size <= 1 {
        return (vec![], 0);
    }
    let mut tokens: Vec<u32> = Vec::with_capacity(block_size - 1);
    let mut tok = if let Some(first) = precomputed_first_tok {
        tokens.push(first);
        first
    } else {
        carry_tok
    };
    let n_iters = (block_size - 1).saturating_sub(tokens.len());
    let mut fwd_calls = 0usize;
    for _ in 0..n_iters {
        let next_hidden = forward_fn(tok);
        fwd_calls += 1;
        tok = greedy_fn();
        let _ = next_hidden;
        tokens.push(tok);
    }
    (tokens, fwd_calls)
}

/// Mirror of the `accept_and_reseed` token-sequence construction.
/// Returns `(sequence_fed_to_forward_conditioned, hidden_slice_end)`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn mock_accept_and_reseed_tokens(
    draft_tokens: &[u32],
    correction: u32,
    accepted: usize,
) -> (Vec<u32>, usize) {
    let n = accepted + 1;
    let mut tokens: Vec<u32> = Vec::with_capacity(n);
    tokens.extend_from_slice(&draft_tokens[..accepted]);
    tokens.push(correction);
    // hidden_slice end index (exclusive upper bound on the sequence axis).
    (tokens, n)
}

#[test]
fn draft_block_unseeded_calls_forward_bs_minus_1_times() {
    // Unseeded path: forward_token must be called exactly block_size - 1 times.
    for bs in 2usize..=6 {
        let counter = std::cell::Cell::new(0usize);
        let (toks, fwd) = mock_draft_block(
            42,
            None,
            bs,
            |tok| {
                counter.set(counter.get() + 1);
                tok + 1
            },
            || 7,
        );
        assert_eq!(
            fwd,
            bs - 1,
            "bs={bs}: expected {} fwd calls, got {fwd}",
            bs - 1
        );
        assert_eq!(toks.len(), bs - 1, "bs={bs}: expected {} tokens", bs - 1);
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn draft_block_seeded_calls_forward_bs_minus_2_times_and_first_tok_is_seed() {
    // Seeded path: forward_token called exactly block_size - 2 times;
    // tokens[0] is the precomputed seed token.
    let seed = 99u32;
    for bs in 2usize..=6 {
        let (toks, fwd) = mock_draft_block(42, Some(seed), bs, |tok| tok + 1, || 7);
        let expected_fwd = bs.saturating_sub(2);
        assert_eq!(
            fwd, expected_fwd,
            "bs={bs}: expected {expected_fwd} fwd calls, got {fwd}"
        );
        if !toks.is_empty() {
            assert_eq!(toks[0], seed, "bs={bs}: tokens[0] must be the seed token");
        }
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn accept_and_reseed_sequence_length_is_accepted_plus_one() {
    // The token sequence fed to forward_tokens_conditioned must be
    // `accepted + 1` long (accepted draft prefix + correction).
    let draft = [10u32, 11, 12, 13];
    let correction = 99u32;
    for accepted in 0..=draft.len() {
        let (seq, n) = mock_accept_and_reseed_tokens(&draft, correction, accepted);
        assert_eq!(
            seq.len(),
            accepted + 1,
            "accepted={accepted}: seq.len()={} != {}",
            seq.len(),
            accepted + 1
        );
        assert_eq!(n, accepted + 1);
        // Last token is always the correction.
        assert_eq!(*seq.last().unwrap(), correction);
        // Prefix matches the accepted draft tokens.
        assert_eq!(&seq[..accepted], &draft[..accepted]);
    }
}

#[test]
fn accept_and_reseed_hidden_slice_covers_zero_through_accepted() {
    // The hidden slice end must equal `accepted + 1` (the sequence axis
    // covers positions 0..=accepted, i.e. [b, draft[0], ..., correction]).
    let draft = [1u32, 2, 3];
    let correction = 77u32;
    for accepted in 0..=draft.len() {
        let (_, hidden_end) = mock_accept_and_reseed_tokens(&draft, correction, accepted);
        assert_eq!(
            hidden_end,
            accepted + 1,
            "accepted={accepted}: hidden_end={hidden_end}"
        );
    }
}

// -----------------------------------------------------------------------
// Prefill chunking boundary tests.
//
// These tests verify the chunk-boundary arithmetic used in
// `forward_verify_capture_chunked` without loading any model weights.
// They mirror the while-loop logic inline.
// -----------------------------------------------------------------------

/// Compute the list of (start, end, is_last) chunk windows for a given
/// prompt length and chunk size. Mirrors the while-loop body.
fn chunk_windows(n: usize, chunk_size: usize) -> Vec<(usize, usize, bool)> {
    assert!(chunk_size > 0);
    let mut windows = Vec::new();
    let mut pos = 0;
    while pos < n {
        let end = (pos + chunk_size).min(n);
        let is_last = end == n;
        windows.push((pos, end, is_last));
        pos = end;
    }
    windows
}

#[test]
fn prefill_chunk_windows_cover_full_range() {
    // The chunk windows must partition [0, n) without gaps or overlaps.
    for n in [1, 2, 1023, 1024, 1025, 2048, 4096, 4097] {
        let windows = chunk_windows(n, 1024);
        let mut covered = 0usize;
        for &(s, e, _) in &windows {
            assert!(s < e, "n={n}: empty window [{s},{e})");
            assert_eq!(s, covered, "n={n}: gap at pos {s}, expected {covered}");
            covered = e;
        }
        assert_eq!(covered, n, "n={n}: windows do not cover full range");
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prefill_chunk_windows_exactly_one_last_chunk() {
    // Exactly one chunk must be marked `is_last = true`, and it must be
    // the final window.
    for n in [1, 100, 1024, 1025, 4096] {
        let windows = chunk_windows(n, 1024);
        let last_count = windows.iter().filter(|&&(_, _, il)| il).count();
        assert_eq!(
            last_count, 1,
            "n={n}: expected 1 last chunk, got {last_count}"
        );
        assert!(
            windows.last().unwrap().2,
            "n={n}: final window not marked is_last"
        );
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn prefill_chunk_windows_short_prompt_is_single_chunk() {
    // When n <= chunk_size the short-circuit path fires: a single window
    // covering all n tokens marked is_last.
    for n in 1..=1024 {
        let windows = chunk_windows(n, 1024);
        assert_eq!(windows.len(), 1, "n={n}: expected 1 window");
        assert_eq!(windows[0], (0, n, true), "n={n}: window mismatch");
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn prefill_chunk_windows_count_and_last_chunk_size() {
    // For n=4096, chunk_size=1024: expect 4 windows of exactly 1024.
    let windows = chunk_windows(4096, 1024);
    assert_eq!(windows.len(), 4);
    for (i, &(s, e, _)) in windows.iter().enumerate() {
        assert_eq!(e - s, 1024, "chunk {i}: size != 1024");
    }
    // For n=4097, chunk_size=1024: 4 full chunks + 1 partial.
    let windows = chunk_windows(4097, 1024);
    assert_eq!(windows.len(), 5);
    assert_eq!(windows[4].1 - windows[4].0, 1);
}

// -----------------------------------------------------------------------
// hot-path hot_ids precomputation + restricted-vocab argmax contract.
//
// These tests verify the hot_ids precomputation logic and the contract
// between restricted-vocab argmax and full-vocab argmax without loading
// any model weights.
// -----------------------------------------------------------------------

/// Compute hot_ids from a d2t table: hot_ids[i] = i + d2t[i].
fn compute_hot_ids(d2t: &[i32]) -> Vec<u32> {
    d2t.iter()
        .enumerate()
        .map(|(i, &offset)| (i as i64 + i64::from(offset)) as u32)
        .collect()
}

/// Simulate restricted-vocab argmax: given `full_logits` (length=full_vocab)
/// and `hot_ids` (length=draft_vocab), return the target-vocab token id that
/// the restricted argmax would select.
///
/// Computes argmax over `full_logits[hot_ids[i]]` for i in 0..draft_vocab,
/// then maps back to the target-vocab id via `hot_ids`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn restricted_argmax_target(full_logits: &[f32], hot_ids: &[u32]) -> u32 {
    let best_draft_idx = hot_ids
        .iter()
        .enumerate()
        .max_by(|(_, &a), (_, &b)| {
            full_logits[a as usize]
                .partial_cmp(&full_logits[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i);
    hot_ids[best_draft_idx]
}

#[test]
fn hot_ids_precomputed_matches_draft_to_target() {
    // hot_ids[i] must equal draft_to_target(i, d2t) for every draft position.
    let d2t = vec![0i32, 5, -3, 100, -1, 0];
    let hot = compute_hot_ids(&d2t);
    for (i, &h) in hot.iter().enumerate() {
        let expected = draft_to_target(i as u32, &d2t);
        assert_eq!(
            h, expected,
            "hot_ids[{i}]={h} != draft_to_target({i})={expected}"
        );
    }
}

#[test]
fn hot_ids_empty_when_d2t_empty() {
    // When d2t is empty (same vocab), no hot_ids should be produced.
    let d2t: Vec<i32> = Vec::new();
    let hot = compute_hot_ids(&d2t);
    assert!(hot.is_empty(), "expected empty hot_ids for empty d2t");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn restricted_argmax_matches_full_when_best_in_draft_vocab() {
    // When the full-vocab argmax token IS in the draft vocab, the restricted
    // argmax must produce the same target-vocab token id.
    //
    // Setup: draft_vocab=4, full_vocab=10.
    // d2t maps draft positions to target-vocab ids 2, 5, 7, 9.
    // Set logits so that full argmax = token 5 (in draft vocab at position 1).
    // Use non-negative offsets to keep all ids positive.
    let d2t = vec![2i32, 4, 5, 6]; // hot_ids = [2, 5, 7, 9]
    let hot = compute_hot_ids(&d2t);
    assert_eq!(hot, vec![2, 5, 7, 9]);

    let mut full_logits = vec![0.0f32; 10];
    full_logits[5] = 10.0; // highest logit → best token is 5

    let restricted_result = restricted_argmax_target(&full_logits, &hot);
    let full_argmax: u32 = full_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    assert_eq!(
        full_argmax, 5,
        "full argmax should be 5 (highest logit at index 5)"
    );
    assert_eq!(
        restricted_result, 5,
        "restricted argmax must match full argmax when best token is in draft vocab"
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn restricted_argmax_differs_when_best_not_in_draft_vocab() {
    // When the full-vocab best token is NOT in the draft vocab, the restricted
    // argmax will return a different token (the best within the draft vocab).
    // This is the expected failure mode that the correction-position full-vocab
    // pass handles.
    //
    // Setup: draft_vocab=4, full_vocab=10.
    // hot_ids = [2, 5, 7, 9]. Best full-vocab token is 8 (not in draft vocab).
    let d2t = vec![2i32, 4, 5, 6]; // hot_ids = [2, 5, 7, 9]
    let hot = compute_hot_ids(&d2t);

    let mut full_logits = vec![0.0f32; 10];
    full_logits[8] = 10.0; // best token is 8 — NOT in hot_ids
    full_logits[5] = 5.0; // second best, IS in hot_ids

    let full_argmax: u32 = full_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    let restricted_result = restricted_argmax_target(&full_logits, &hot);

    // Full argmax = 8, not in draft vocab.
    assert_eq!(full_argmax, 8, "full argmax should be 8");
    assert!(
        !hot.contains(&8),
        "token 8 must not be in draft vocab for this test"
    );

    // Restricted argmax must differ from full argmax.
    assert_ne!(
        restricted_result, full_argmax,
        "restricted argmax must NOT match full argmax when best token is outside draft vocab"
    );
    // Restricted argmax must be the best token within the draft vocab subset.
    assert_eq!(
        restricted_result, 5,
        "restricted argmax should return token 5 (best within draft vocab)"
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn hot_ids_arr_correctness() {
    // Verify the hot_ids array construction mirrors the host Vec computation.
    let d2t = vec![0i32, 5, 100, -3];
    let hot = compute_hot_ids(&d2t);
    // hot[0]=0, hot[1]=6, hot[2]=102, hot[3]=0 (3 + (-3))
    assert_eq!(hot[0], 0);
    assert_eq!(hot[1], 6);
    assert_eq!(hot[2], 102);
    assert_eq!(hot[3], 0); // 3 + (-3) = 0
}

/// Simulate the EOS append logic from `load_eagle3`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn append_eos_to_hot(d2t: &[i32], eos_token_ids: &[u32]) -> Vec<u32> {
    let mut hot = compute_hot_ids(d2t);
    let pre_len = hot.len();
    for &eos in eos_token_ids {
        if !hot[..pre_len].contains(&eos) {
            hot.push(eos);
        }
    }
    hot
}

#[test]
fn hot_ids_contains_all_eos_token_ids() {
    // After load, hot_ids must contain every eos_token_id.
    // hot_ids[i] = i + d2t[i]; EOS are appended when not already present.
    let d2t = vec![2i32, 4, 5, 6]; // hot_ids = [2, 5, 7, 9]
    let eos_token_ids = &[151645u32, 151643]; // typical Qwen3 EOS set
    let hot = append_eos_to_hot(&d2t, eos_token_ids);
    for &eos in eos_token_ids {
        assert!(
            hot.contains(&eos),
            "hot_ids must contain EOS token id {eos}"
        );
    }
    // Base d2t-derived ids must still be present.
    for &expected in &[2u32, 5, 7, 9] {
        assert!(
            hot.contains(&expected),
            "hot_ids must still contain d2t-derived id {expected}"
        );
    }
}

#[test]
fn hot_ids_eos_deduplicated_when_already_present() {
    // When an EOS id already appears in the d2t-derived hot_ids, it must
    // not be duplicated.
    let d2t = vec![0i32, 0, 0]; // hot_ids = [0, 1, 2]
    let eos_token_ids = &[1u32]; // id 1 already in hot_ids
    let hot = append_eos_to_hot(&d2t, eos_token_ids);
    // Length must stay 3 (no duplicate appended).
    assert_eq!(
        hot.len(),
        3,
        "EOS id already in hot_ids must not be duplicated"
    );
    assert_eq!(hot.iter().filter(|&&v| v == 1).count(), 1);
}

// -----------------------------------------------------------------------
// review: find_full_pos first-mismatch logic (MEDIUM 2).
//
// `find_full_pos(draft_tokens, tokens)` returns the index of the first
// position in 0..n_draft where `tokens[i] != draft_tokens[i]`, or `n_draft`
// when all draft positions match (bonus position).
// -----------------------------------------------------------------------

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn all_equal_prefix_full_pos_eq_n_draft() {
    // When all tokens[i] for i in 0..n_draft match draft_tokens[i],
    // full_pos must equal n_draft (correction lands on the bonus position).
    let n_draft = 4usize;
    let draft_tokens: Vec<u32> = vec![10, 20, 30, 40];
    // tokens has n_draft+1 entries (including bonus); first n_draft match draft.
    let tokens: Vec<u32> = vec![10, 20, 30, 40, 99];
    assert_eq!(
        find_full_pos(&draft_tokens, &tokens[..n_draft]),
        n_draft,
        "all-match: full_pos must be n_draft"
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn first_mismatch_at_j_full_pos_eq_j() {
    // A mismatch at position j returns j.
    let draft_tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    for j in 0..draft_tokens.len() {
        // tokens matches draft for 0..j, then diverges at j.
        let mut tokens = draft_tokens.clone();
        tokens[j] = 99;
        // Add a bonus slot so len matches n_draft for the slice.
        let result = find_full_pos(&draft_tokens, &tokens[..draft_tokens.len()]);
        assert_eq!(result, j, "mismatch at {j}: full_pos must be {j}");
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn mismatch_only_at_bonus_full_pos_eq_n_draft() {
    // A mismatch at index n_draft (the bonus position) is outside the draft
    // range; the caller only passes tokens[..n_draft] so find_full_pos only
    // sees the draft positions and must return n_draft.
    let n_draft = 3usize;
    let draft_tokens: Vec<u32> = vec![10, 20, 30];
    // tokens matches draft at all n_draft positions; bonus slot differs.
    let tokens_full: Vec<u32> = vec![10, 20, 30, 77]; // index 3 = bonus
                                                      // We pass only tokens[..n_draft] to find_full_pos, same as the hot-path.
    let result = find_full_pos(&draft_tokens, &tokens_full[..n_draft]);
    assert_eq!(
        result, n_draft,
        "bonus mismatch must not influence full_pos (out of range)"
    );
}

// -----------------------------------------------------------------------
// KV-cache sizing regression test.
//
// Guards the fix in `Eagle3Drafter::reset`: the drafter cache must be sized
// to the verifier's context limit (max_seq), not a hardcoded 4096. Prior to
// the fix, once prompt + emitted tokens exceeded 4096 the decode path hit a
// zero-width `slice_update` range (prev_offset >= max_seq with a live
// buffer), producing a broadcast-shape panic.
// -----------------------------------------------------------------------

/// Regression: a `KvCache` built via `KvCache::with_quant_max_seq(KvQuant::None, 8192)`
/// must successfully process decode steps whose cumulative offset surpasses
/// 4096 (the former hardcoded cap).
///
/// The test mirrors what `Eagle3Drafter::reset(max_seq)` does, then drives
/// the same update path that crashed: prefill to 4090, then decode past 4096.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test tensors + KvCache are built by construction in this fn; the unwrapped Results are infallible by setup"
)]
fn eagle3_drafter_cache_sized_from_max_seq_decodes_past_4096() {
    use rmlx_kv_quant::{KvCache, KvQuant};
    use rmlx_mlx::{Array, Device, Dtype};

    const B: i32 = 1;
    const KV_H: i32 = 2;
    const HEAD_DIM: i32 = 8;
    // Prefill to just below 4096; decode steps will push offset above it.
    const PREFILL_SEQ: i32 = 4090;
    // Decode this many single-token steps — step 7 puts offset at 4097.
    const DECODE_STEPS: i32 = 10;

    let device = Device::Cpu;

    // Build the cache the same way reset() does (KvQuant::None, max_seq=8192).
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 8192);

    // Helper: allocate a tiny f32 array for K or V.
    let make_arr = |seq: i32, fill: f32| -> Array {
        let shape = [B, KV_H, seq, HEAD_DIM];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let data = vec![fill; n];
        // SAFETY: f32 is 4-byte LE on Apple Silicon (CLAUDE.md hard rule 1).
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), n * 4) };
        Array::from_bytes(bytes, &shape, Dtype::F32).unwrap()
    };

    // Prefill phase: feed PREFILL_SEQ tokens.
    cache.enter_prefill();
    let k_pre = make_arr(PREFILL_SEQ, 0.1);
    let v_pre = make_arr(PREFILL_SEQ, 0.2);
    cache.update(&k_pre, &v_pre, device).unwrap();
    cache.exit_prefill(device).unwrap();

    assert_eq!(
        cache.offset(),
        PREFILL_SEQ,
        "offset must equal PREFILL_SEQ after prefill"
    );

    // Decode phase: single-token steps. Each step increments offset by 1.
    // Steps 1-6 stay at or below 4096; step 7 reaches 4097.
    // With the old max_seq=4096 hardcode, step 7 panicked here.
    for step in 1..=DECODE_STEPS {
        let k_dec = make_arr(1, 0.01 * step as f32);
        let v_dec = make_arr(1, 0.02 * step as f32);
        cache.update(&k_dec, &v_dec, device).unwrap_or_else(|e| {
            panic!(
                "decode step {step} (offset {}) failed: {e}",
                PREFILL_SEQ + step
            )
        });
        assert_eq!(
            cache.offset(),
            PREFILL_SEQ + step,
            "offset must advance by 1 each decode step"
        );
    }

    // Final offset must be 4100 (4090 + 10), comfortably past 4096.
    assert_eq!(
        cache.offset(),
        PREFILL_SEQ + DECODE_STEPS,
        "cache must decode past 4096 without error"
    );
}
