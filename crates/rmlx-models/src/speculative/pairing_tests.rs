use std::fmt::Write as _;
use std::sync::{Arc, Mutex, PoisonError};

use super::*;

/// Where the test drafter lives, as the loaders pass it.
const DRAFT_DIR: &str = "drafts/example";

/// The Qwen3.8-27B trunk.
const VERIFIER: VerifierTrunk = VerifierTrunk {
    hidden_size: 5120,
    num_hidden_layers: 64,
    vocab_size: 248_320,
};

/// A config holding the given integer keys.
fn config(keys: &[(&str, usize)]) -> serde_json::Map<String, serde_json::Value> {
    keys.iter()
        .map(|&(key, value)| (key.to_owned(), serde_json::json!(value)))
        .collect()
}

/// A config that restates the verifier under `layers_key`, with `key` moved to
/// `foreign`.
fn config_with(
    layers_key: &str,
    key: &str,
    foreign: usize,
) -> serde_json::Map<String, serde_json::Value> {
    let facts = [
        ("hidden_size", VERIFIER.hidden_size),
        (layers_key, VERIFIER.num_hidden_layers),
        ("vocab_size", VERIFIER.vocab_size),
    ];
    let moved: Vec<(&str, usize)> = facts
        .iter()
        .map(|&(k, v)| (k, if k == key { foreign } else { v }))
        .collect();
    config(&moved)
}

/// A config that disagrees with the verifier on one fact is refused, and the
/// refusal names the key, the drafter's value and the verifier's. One case per
/// fact and per layers key, so dropping any one comparison turns its case red.
#[test]
fn a_config_declaring_another_base_is_refused_naming_both_values() {
    for layers_key in ["num_hidden_layers", "num_target_layers"] {
        for (key, foreign, verifier_value) in [
            ("hidden_size", 4096, VERIFIER.hidden_size),
            (layers_key, 40, VERIFIER.num_hidden_layers),
            ("vocab_size", 262_144, VERIFIER.vocab_size),
        ] {
            let declared = config_with(layers_key, key, foreign);
            let err = VERIFIER
                .refuse_another_base(
                    Path::new(DRAFT_DIR),
                    "the test drafter's config",
                    &[&declared],
                    TrunkKeys::named(Some(layers_key)),
                )
                .err();
            let Some(Error::SpeculativePairing { reason }) = err else {
                panic!("{layers_key}/{key}={foreign}: expected a pairing refusal, got {err:?}");
            };
            assert!(
                reason.contains("the test drafter's config")
                    && reason.contains(key)
                    && reason.contains(&foreign.to_string())
                    && reason.contains(&verifier_value.to_string()),
                "the refusal must name the source, {key}, the drafter's {foreign} and the \
                 verifier's {verifier_value}: {reason}"
            );
        }
    }
}

/// The events of level WARN emitted on this thread while `f` runs, each as its
/// fields in one string.
///
/// The subscriber is scoped to the thread and answers `always` for every
/// callsite, so a callsite another test hit first is not left disabled for this
/// one.
fn warnings_during(f: impl FnOnce()) -> Vec<String> {
    #[derive(Clone, Default)]
    struct Collect(Arc<Mutex<Vec<String>>>);

    struct Fields(String);

    impl tracing::field::Visit for Fields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            let _ = write!(self.0, "{}={value:?} ", field.name());
        }
    }

    impl tracing::Subscriber for Collect {
        fn register_callsite(
            &self,
            _meta: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::always()
        }
        fn enabled(&self, _md: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut fields = Fields(String::new());
                event.record(&mut fields);
                self.0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(fields.0);
            }
        }
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }

    let collect = Collect::default();
    tracing::subscriber::with_default(collect.clone(), f);
    let seen = collect.0.lock().unwrap_or_else(PoisonError::into_inner);
    seen.clone()
}

/// A config that declares none of the keys loads, with one warning that names
/// the draft directory. A key that is absent, null or not an integer declares
/// nothing. A config that declares any one of them is compared and warns of
/// nothing.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: no case declares a disagreeing fact, so an Err is the assertion failing"
)]
fn a_config_declaring_none_of_the_keys_warns_naming_the_draft() {
    let check = |declared: serde_json::Value| {
        let map = declared.as_object().cloned().unwrap_or_default();
        warnings_during(|| {
            VERIFIER
                .refuse_another_base(
                    Path::new(DRAFT_DIR),
                    "the test drafter's config",
                    &[&map],
                    TrunkKeys::named(Some("num_target_layers")),
                )
                .expect("no disagreeing fact");
        })
    };
    for none in [
        serde_json::json!({}),
        serde_json::json!({ "vocab_size": null }),
        serde_json::json!({ "hidden_size": "5120" }),
        serde_json::json!({ "num_hidden_layers": VERIFIER.num_hidden_layers }),
    ] {
        let warnings = check(none.clone());
        assert_eq!(
            warnings.len(),
            1,
            "{none}: expected one warning, got {warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .all(|w| w.contains(DRAFT_DIR) && w.contains("the test drafter's config")),
            "{none}: the warning must name the draft directory and the source: {warnings:?}"
        );
    }
    for some in [
        serde_json::json!({ "vocab_size": VERIFIER.vocab_size }),
        serde_json::json!({ "num_target_layers": VERIFIER.num_hidden_layers }),
        serde_json::Value::Object(config_with("num_target_layers", "", 0)),
    ] {
        let warnings = check(some.clone());
        assert!(
            warnings.is_empty(),
            "{some}: expected no warning, got {warnings:?}"
        );
    }
}

/// The layers key picks the layer-count key and no other: a drafter whose own
/// `num_hidden_layers` is its depth, not its base's, is not refused for it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the only key present is the drafter's own depth, so an Err is the assertion failing"
)]
fn a_drafters_own_depth_is_not_the_base_layer_count() {
    let own_depth = config(&[("num_hidden_layers", 8)]);
    VERIFIER
        .refuse_another_base(
            Path::new(DRAFT_DIR),
            "the test drafter's config",
            &[&own_depth],
            TrunkKeys::named(Some("num_target_layers")),
        )
        .expect("the drafter's own depth is not compared");
}

/// A config that restates no layer count compares none: its `num_hidden_layers`
/// is the drafter's own depth whatever the verifier has.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: no key disagrees, so an Err is the assertion failing"
)]
fn a_config_with_no_layer_key_compares_no_layer_count() {
    let declared = config(&[
        ("hidden_size", VERIFIER.hidden_size),
        ("num_hidden_layers", 1),
        ("num_target_layers", 8),
        ("vocab_size", VERIFIER.vocab_size),
    ]);
    VERIFIER
        .refuse_another_base(
            Path::new(DRAFT_DIR),
            "the test drafter's config",
            &[&declared],
            TrunkKeys::named(None),
        )
        .expect("no layer count is compared");
}

/// The hidden size is read under the key the config names for it and no other,
/// and a key is looked up in each map in order: a drafter that restates the base
/// width as `backbone_hidden_size` is not refused for the width of its own
/// `hidden_size`, and a vocabulary in a second map is compared.
#[test]
fn keys_and_maps_are_the_ones_the_caller_names() {
    let keys = TrunkKeys {
        hidden_size: "backbone_hidden_size",
        num_hidden_layers: None,
        vocab_size: "vocab_size",
    };
    let refuse = |declared: &[&serde_json::Map<String, serde_json::Value>]| {
        VERIFIER.refuse_another_base(
            Path::new(DRAFT_DIR),
            "the test drafter's config",
            declared,
            keys,
        )
    };
    let top = config(&[
        ("backbone_hidden_size", VERIFIER.hidden_size),
        ("hidden_size", 256),
    ]);
    let nested = config(&[("vocab_size", VERIFIER.vocab_size)]);
    assert!(refuse(&[&top, &nested]).is_ok());

    let foreign = config(&[("vocab_size", 262_144)]);
    let err = refuse(&[&top, &foreign]).err();
    let Some(Error::SpeculativePairing { reason }) = err else {
        panic!("expected a pairing refusal for the vocabulary in the second map, got {err:?}");
    };
    assert!(
        reason.contains("vocab_size = 262144"),
        "the refusal must name the vocabulary: {reason}"
    );

    let wrong_width = config(&[("backbone_hidden_size", 4096)]);
    let err = refuse(&[&wrong_width]).err();
    assert!(
        matches!(&err, Some(Error::SpeculativePairing { reason }) if reason.contains("backbone_hidden_size = 4096")),
        "the refusal must name the backbone key: {err:?}"
    );
}
