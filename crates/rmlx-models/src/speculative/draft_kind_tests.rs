//! The declaration rule that picks a drafter kind from a snapshot's config.

use super::DraftKind;

/// Every shipped drafter snapshot declares itself, and each declaration lands
/// on the kind whose loader can build it. The pairs are the `architectures[0]`
/// / `model_type` fields of the real snapshots.
#[test]
fn shipped_declarations_resolve_to_their_loader() {
    let cases: &[(&str, &str, DraftKind)] = &[
        // Gemma4 assistant sidecar.
        (
            "Gemma4AssistantForCausalLM",
            "gemma4_assistant",
            DraftKind::Mtp,
        ),
        // Qwen3.5-family MTP sidecar: model_type only, no architectures array.
        ("", "qwen3_5_mtp", DraftKind::Mtp),
        // DFlash and DFlash2 declare a plain `qwen3` model type under the head's
        // own architecture name, so the architecture must be read first.
        ("DFlashDraftModel", "qwen3", DraftKind::DFlash),
        ("DFlash2DraftModel", "qwen3", DraftKind::DFlash),
        // EAGLE-3 declares `llama` as its model type for the same reason.
        ("LlamaForCausalLMEagle3", "llama", DraftKind::Eagle3),
        // Full models of every family the two-model loop can run.
        (
            "Gemma4ForConditionalGeneration",
            "gemma4",
            DraftKind::TwoModel,
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            "qwen3_5",
            DraftKind::TwoModel,
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            "qwen3_5_moe",
            DraftKind::TwoModel,
        ),
        ("Qwen3ForCausalLM", "qwen3", DraftKind::TwoModel),
    ];
    for (arch, model_type, want) in cases {
        assert_eq!(
            DraftKind::from_declaration(arch, model_type),
            Some(*want),
            "arch={arch:?} model_type={model_type:?}"
        );
    }
}

/// A declaration no loader can build is `None`, never a guess. An unregistered
/// architecture over a familiar model type is the case that matters: `qwen3`
/// alone must not make a foreign head a two-model draft.
#[test]
fn an_unknown_declaration_is_no_kind() {
    for (arch, model_type) in [
        ("", ""),
        ("SomethingElseForCausalLM", "qwen3"),
        ("", "gemma4"),
    ] {
        assert_eq!(
            DraftKind::from_declaration(arch, model_type),
            None,
            "arch={arch:?} model_type={model_type:?}"
        );
    }
}

/// The CLI spelling and the log spelling are one string per kind, and the
/// parser accepts exactly those.
#[test]
fn every_kind_round_trips_through_its_name() {
    for kind in [
        DraftKind::Mtp,
        DraftKind::DFlash,
        DraftKind::Eagle3,
        DraftKind::TwoModel,
    ] {
        assert_eq!(kind.as_str().parse::<DraftKind>(), Ok(kind));
        assert_eq!(kind.to_string(), kind.as_str());
    }
    let err = "two-model".parse::<DraftKind>().err().unwrap_or_default();
    assert!(
        err.contains("two_model"),
        "the error names the valid spellings: {err}"
    );
}
