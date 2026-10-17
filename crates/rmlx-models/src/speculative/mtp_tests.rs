use super::*;

/// Compile-check: the public MTP surface exists with the expected sigs.
#[test]
fn mtp_module_compiles() {
    // Reference the items so the symbols are checked at compile time
    // without spelling out their (clippy-flagged complex) fn types.
    let _load = MtpDrafter::load;
    let _ = _load;
}

/// The sidecar's declared block is the depth it was trained at, and the loop
/// runs the depth the request asks for whatever that declaration says.
///
/// Every shipped Qwen3.5-family sidecar declares `block_size: 3`, so a request
/// clamped to the declaration could never run a deeper block on any checkpoint
/// that exists. The declaration is passed in at every case here for that reason:
/// a resolution that narrowed to it would read as correct against a `None`.
#[test]
fn a_request_deeper_than_the_declared_block_runs_at_the_request() {
    for declared in [None, Some(2), Some(3), Some(8)] {
        for requested in [4, 6, 8, 16] {
            assert_eq!(
                block_from_request(requested, declared),
                requested,
                "declared={declared:?} requested={requested}"
            );
        }
    }
}

/// A request past what one verify forward can score is clamped to it.
#[test]
fn a_request_past_the_verify_ceiling_runs_at_the_ceiling() {
    for declared in [None, Some(3), Some(MAX_BLOCK_SIZE)] {
        assert_eq!(block_from_request(MAX_BLOCK_SIZE, declared), MAX_BLOCK_SIZE);
        assert_eq!(
            block_from_request(MAX_BLOCK_SIZE + 1, declared),
            MAX_BLOCK_SIZE
        );
        assert_eq!(block_from_request(usize::MAX, declared), MAX_BLOCK_SIZE);
    }
}

/// Below two there is a seed and no draft, so there is no round to run.
#[test]
fn a_request_below_two_runs_at_two() {
    for declared in [None, Some(3), Some(8)] {
        assert_eq!(block_from_request(0, declared), 2);
        assert_eq!(block_from_request(1, declared), 2);
        assert_eq!(block_from_request(2, declared), 2);
    }
}

/// The verifier side of every pairing case: the Qwen3.8-27B trunk.
const VERIFIER: VerifierTrunk = VerifierTrunk {
    hidden_size: 5120,
    num_hidden_layers: 64,
    vocab_size: 248_320,
};

/// The three trunk facts as the verifier states them.
const MATCHED: [(&str, usize); 3] = [
    ("hidden_size", VERIFIER.hidden_size),
    ("num_hidden_layers", VERIFIER.num_hidden_layers),
    ("vocab_size", VERIFIER.vocab_size),
];

/// A sidecar folder holding nothing but a `config.json` whose `text_config`
/// declares `facts` plus the head dims the loader reads after the pairing check.
#[allow(
    clippy::expect_used,
    reason = "test-only: the tempdir is this process's own, so a failure to write into it names a broken environment"
)]
fn config_only_sidecar(facts: &[(&str, usize)]) -> tempfile::TempDir {
    let head_dims = [
        ("num_attention_heads", 24),
        ("num_key_value_heads", 4),
        ("head_dim", 256),
    ];
    let tc: serde_json::Map<String, serde_json::Value> = facts
        .iter()
        .chain(head_dims.iter())
        .map(|&(key, value)| (key.to_owned(), serde_json::json!(value)))
        .collect();
    let config = serde_json::json!({ "model_type": "qwen3_5_mtp", "text_config": tc });
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config.json");
    dir
}

/// The error `load_mtp_head` stops with on a folder that has no weights.
fn load_error(dir: &tempfile::TempDir) -> Error {
    match load_mtp_head(dir.path(), VERIFIER) {
        Ok(_) => Error::Other("a config-only sidecar loaded".to_owned()),
        Err(e) => e,
    }
}

/// A sidecar whose `text_config` restates a base other than the verifier is
/// refused before a weight is read, and the refusal names the key and both
/// values. One case per fact, so dropping any one of them from the comparison
/// turns its case red.
#[test]
fn a_sidecar_declaring_another_base_is_refused_naming_both_values() {
    for (key, foreign) in [
        ("hidden_size", 4096),
        ("num_hidden_layers", 40),
        ("vocab_size", 262_144),
    ] {
        let facts: Vec<(&str, usize)> = MATCHED
            .iter()
            .map(|&(k, v)| (k, if k == key { foreign } else { v }))
            .collect();
        let verifier_value = MATCHED
            .iter()
            .find_map(|&(k, v)| (k == key).then_some(v))
            .unwrap_or_default();
        let err = load_error(&config_only_sidecar(&facts));
        let msg = err.to_string();
        assert!(
            matches!(err, Error::SpeculativePairing { .. }),
            "{key}={foreign}: expected a pairing refusal, got: {msg}"
        );
        assert!(
            msg.contains(key)
                && msg.contains(&foreign.to_string())
                && msg.contains(&verifier_value.to_string()),
            "the refusal must name {key}, the sidecar's {foreign} and the verifier's \
             {verifier_value}: {msg}"
        );
    }
}

/// A sidecar that agrees with the verifier on every fact it declares is not
/// refused: the load goes past the pairing check and stops on the missing
/// weights. So does one that declares only some of the facts, or none of them.
#[test]
fn a_sidecar_that_agrees_or_declares_nothing_goes_on_to_its_weights() {
    let partial = [("num_hidden_layers", VERIFIER.num_hidden_layers)];
    for facts in [&MATCHED[..], &partial[..], &[]] {
        let err = load_error(&config_only_sidecar(facts));
        let msg = err.to_string();
        assert!(
            !matches!(err, Error::SpeculativePairing { .. }) && msg.contains("shard index"),
            "text_config facts {facts:?}: expected the load to reach the weights, got: {msg}"
        );
    }
}
