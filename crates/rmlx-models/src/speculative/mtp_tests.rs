use super::*;

/// Pure mirror of the deferred-greedy acceptance accounting (the on-device
/// `walk_deferred_greedy` differs only by deriving `target` from logits).
/// Locks the semantics: longest greedy-matching prefix + one
/// correction/bonus, capped at `budget`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn walk_logic(target: &[u32], draft: &[u32], budget: usize) -> (usize, Vec<u32>) {
    let n_draft = draft.len();
    let mut accepted = 0usize;
    let mut new_tokens = Vec::new();
    for pos in 0..=n_draft {
        let token = target[pos];
        if pos < n_draft && token == draft[pos] {
            accepted += 1;
            if new_tokens.len() < budget {
                new_tokens.push(token);
            }
            continue;
        }
        if new_tokens.len() < budget {
            new_tokens.push(token);
        }
        break;
    }
    (accepted, new_tokens)
}

#[test]
fn walk_all_accepted_emits_bonus() {
    let draft = [10, 11, 12];
    let target = [10, 11, 12, 99]; // n_draft+1 verifier predictions
    let (acc, emit) = walk_logic(&target, &draft, 8);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11, 12, 99]);
}

#[test]
fn walk_partial_accept_emits_correction() {
    let draft = [10, 11, 12];
    let target = [10, 11, 55, 0]; // diverge at pos 2
    let (acc, emit) = walk_logic(&target, &draft, 8);
    assert_eq!(acc, 2);
    assert_eq!(emit, vec![10, 11, 55]); // 2 accepted + correction
}

#[test]
fn walk_zero_accept_emits_only_correction() {
    let draft = [10, 11];
    let target = [42, 0, 0];
    let (acc, emit) = walk_logic(&target, &draft, 8);
    assert_eq!(acc, 0);
    assert_eq!(emit, vec![42]);
}

#[test]
fn walk_respects_budget() {
    let draft = [10, 11, 12];
    let target = [10, 11, 12, 99];
    let (acc, emit) = walk_logic(&target, &draft, 2);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11]); // capped at budget=2
}

/// Compile-check: the public MTP surface exists with the expected sigs.
#[test]
fn mtp_module_compiles() {
    // Reference the items so the symbols are checked at compile time
    // without spelling out their (clippy-flagged complex) fn types.
    let _load = MtpDrafter::load;
    let _walk = walk_deferred_greedy;
    let _ = (_load, _walk);
}
