//! DFlash drafter unit tests.

use super::*;

// --- dflash_next_block_size schedule ---

#[test]
fn block_size_clamped_to_budget() {
    // No history; the budget caps below the requested ceiling. `remaining`
    // counts tokens still to emit and the block counts the verifier's own
    // token too, so three left is a block of four.
    assert_eq!(dflash_next_block_size(&[], 16, 3), 4);
    assert_eq!(dflash_next_block_size(&[], 16, 99), 16);
}

#[test]
fn block_size_floors_where_the_shared_round_block_does() {
    // Both degenerate inputs land on `round_block`'s floor rather than passing
    // through: a request narrower than a round can run, and a budget with
    // nothing left to emit. Neither is reachable from `dflash_generate`, whose
    // guard gives at least one remaining token and whose ceiling is a resolved
    // block; they are here because the schedule is public.
    assert_eq!(dflash_next_block_size(&[(8, 8)], 1, 99), 2);
    assert_eq!(dflash_next_block_size(&[(8, 8)], 16, 0), 2);
}

#[test]
fn block_size_backs_off_hard_on_weak_acceptance() {
    // accept_rate < 0.30 -> halve when current >= 8.
    // last drafted 15 -> current = min(16, 16) = 16, >= 8 -> 16/2 = 8.
    let weak = [(2usize, 15usize), (1, 15), (0, 15)];
    let next = dflash_next_block_size(&weak, 16, 99);
    assert_eq!(next, 8);
}

#[test]
fn block_size_backs_off_small_when_current_below_8() {
    // current < 8 and weak -> current - 2, floored at min_total (4).
    // last drafted 5 -> current = min(16, 6) = 6; weak -> 6-2 = 4.
    let weak = [(0usize, 5usize), (0, 5)];
    let next = dflash_next_block_size(&weak, 16, 99);
    assert_eq!(next, 4);
}

#[test]
fn block_size_grows_on_strong_full_hits() {
    // accept_rate >= 0.85 and full_hit_rate >= 0.75 -> current + 2.
    // all full hits, last drafted 6 -> current = 7 -> 7+2 = 9.
    let strong = [(6usize, 6usize), (6, 6), (6, 6), (6, 6)];
    let next = dflash_next_block_size(&strong, 16, 99);
    assert_eq!(next, 9);
}

#[test]
fn block_size_holds_on_moderate_acceptance() {
    // 0.50 <= accept_rate < 0.85 -> hold at current.
    // drafted 10 each, accepted 6 -> rate 0.6; last drafted 10 -> current = 11.
    let mod_hist = [(6usize, 10usize), (6, 10), (6, 10)];
    let next = dflash_next_block_size(&mod_hist, 16, 99);
    assert_eq!(next, 11);
}

/// Compile-check: the public DFlash surface exists with expected sigs.
#[test]
fn dflash_module_compiles() {
    let _load = DFlashDrafter::load;
    let _bs: fn(&[(usize, usize)], usize, usize) -> usize = dflash_next_block_size;
    let _ = (_load, _bs);
}

/// The verifier the config-only drafters below are paired with: Qwen3.6-35B-A3B.
const VERIFIER: VerifierTrunk = VerifierTrunk {
    hidden_size: 2048,
    num_hidden_layers: 40,
    vocab_size: 248_320,
};

/// The three facts a DFlash config restates about its base, as the verifier
/// has them.
const MATCHED: [(&str, usize); 3] = [
    ("hidden_size", VERIFIER.hidden_size),
    ("num_target_layers", VERIFIER.num_hidden_layers),
    ("vocab_size", VERIFIER.vocab_size),
];

/// A drafter folder holding nothing but a `config.json` that declares `facts`
/// beside the keys the loader reads first.
#[allow(
    clippy::expect_used,
    reason = "test-only: the tempdir is this process's own, so a failure to write into it names a broken environment"
)]
fn config_only_drafter(facts: &[(&str, usize)]) -> tempfile::TempDir {
    let mut config = serde_json::json!({
        "architectures": ["DFlashDraftModel"],
        "dflash_config": { "mask_token_id": 248_070, "target_layer_ids": [1, 10, 19, 28, 37] },
    });
    let keys = config.as_object_mut().expect("config is an object");
    for &(key, value) in facts {
        keys.insert(key.to_owned(), serde_json::json!(value));
    }
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config.json");
    dir
}

/// The error the loader stops with on a folder that has no weights.
fn load_error(dir: &tempfile::TempDir) -> Error {
    match DFlashDrafter::load(dir.path(), VERIFIER, Device::Cpu) {
        Ok(_) => Error::Other("a config-only drafter loaded".to_owned()),
        Err(e) => e,
    }
}

/// A drafter whose config restates a base other than the verifier is refused
/// before a weight is read, naming the key and both values. One case per fact,
/// so dropping any one of them from the comparison turns its case red.
#[test]
fn a_drafter_built_for_another_base_is_refused_naming_both_values() {
    for (key, foreign) in [
        ("hidden_size", 4096),
        ("num_target_layers", 64),
        ("vocab_size", 262_144),
    ] {
        let facts: Vec<(&str, usize)> = MATCHED
            .iter()
            .map(|&(k, v)| (k, if k == key { foreign } else { v }))
            .collect();
        let held = MATCHED
            .iter()
            .find_map(|&(k, v)| (k == key).then_some(v))
            .unwrap_or_default();
        let err = load_error(&config_only_drafter(&facts));
        let Error::SpeculativePairing { reason } = &err else {
            panic!("{key}={foreign}: expected a pairing refusal, got: {err}");
        };
        assert!(
            reason.contains(key)
                && reason.contains(&foreign.to_string())
                && reason.contains(&held.to_string()),
            "the refusal must name {key}, the drafter's {foreign} and the verifier's {held}: \
             {reason}"
        );
    }
}

/// A drafter that agrees with the verifier on every fact it declares goes on to
/// its weights, as does one that declares only some of them.
#[test]
fn a_drafter_that_agrees_or_declares_some_goes_on_to_its_weights() {
    let hidden_only = [MATCHED[0]];
    for facts in [&MATCHED[..], &hidden_only[..]] {
        let err = load_error(&config_only_drafter(facts));
        let msg = err.to_string();
        assert!(
            !matches!(err, Error::SpeculativePairing { .. }) && msg.contains("shard index"),
            "declared {facts:?}: expected the load to reach the weights, got: {msg}"
        );
    }
}

/// A config with no `hidden_size` is refused rather than read at a default
/// width that a verifier of that width would then pair with.
#[test]
fn a_config_without_hidden_size_is_refused() {
    let err = load_error(&config_only_drafter(&[]));
    assert!(
        matches!(err, Error::Model(_)) && err.to_string().contains("hidden_size"),
        "expected a refusal naming hidden_size, got: {err}"
    );
}
