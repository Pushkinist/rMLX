//! Unit tests for the load-time drafter routing.
//!
//! `decide_draft_kind` picks the kind from the draft snapshot's declaration and
//! the optional flag; `classify_mtp_draft` / `mtp_reject_reason` are the
//! `mtp` family's second-level routing, which keeps a plain Gemma4 draft from
//! falling through to the Qwen3.5 MTP sidecar loader (which would leak a
//! `text_config missing num_experts` error).

use super::{
    classify_mtp_draft, decide_draft_kind, dflash2_reject_reason, drafted_per_round,
    mtp_reject_reason, round_block, MtpDraftFamily, DEFAULT_DRAFT_BLOCK_SIZE, MIN_DRAFT_BLOCK_SIZE,
};
use rmlx_models::{Declared, DraftKind};

// ── decide_draft_kind ────────────────────────────────────────────────────────

/// A declaration alone selects the kind: this is what makes a bare
/// `--draft-model` run, for every kind, including the two-model one.
#[test]
fn a_declaration_alone_selects_the_kind() {
    for kind in [
        DraftKind::Mtp,
        DraftKind::DFlash,
        DraftKind::DFlash2,
        DraftKind::Eagle3,
    ] {
        let declared = Declared::Sidecar(kind);
        assert_eq!(
            decide_draft_kind(None, declared, "arch", "type").ok(),
            Some(kind)
        );
        assert_eq!(
            decide_draft_kind(Some(kind), declared, "arch", "type").ok(),
            Some(kind),
            "a flag that agrees with the declaration changes nothing"
        );
    }
    assert_eq!(
        decide_draft_kind(None, Declared::FullModel, "arch", "type").ok(),
        Some(DraftKind::TwoModel)
    );
}

/// The flag is for a snapshot that declares nothing.
#[test]
fn the_flag_names_an_undeclared_snapshot() {
    assert_eq!(
        decide_draft_kind(Some(DraftKind::Mtp), Declared::Unknown, "", "").ok(),
        Some(DraftKind::Mtp)
    );
}

/// Nothing named on either side is refused, and the refusal says what the
/// snapshot declared and what would settle it.
#[test]
fn an_undeclared_snapshot_without_a_flag_is_refused() {
    let msg = decide_draft_kind(None, Declared::Unknown, "FooForCausalLM", "foo")
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        msg.contains("FooForCausalLM") && msg.contains("\"foo\""),
        "{msg}"
    );
    assert!(msg.contains("--draft-kind"), "names the way out: {msg}");
}

/// A flag that contradicts a sidecar marker is refused rather than obeyed: no
/// loader can build a snapshot as a kind it is not, and the error it would
/// die with later names neither side. The refusal names both.
#[test]
fn a_flag_that_contradicts_the_declaration_is_refused() {
    let msg = decide_draft_kind(
        Some(DraftKind::TwoModel),
        Declared::Sidecar(DraftKind::Mtp),
        "Gemma4AssistantForCausalLM",
        "gemma4_assistant",
    )
    .err()
    .map_or_else(String::new, |e| e.to_string());
    assert!(
        msg.contains("--draft-kind two_model"),
        "names the flag: {msg}"
    );
    assert!(
        msg.contains("a mtp drafter"),
        "names the declaration: {msg}"
    );
    assert!(
        msg.contains("Gemma4AssistantForCausalLM"),
        "names the snapshot's own words: {msg}"
    );
}

/// The registry's full-model inference yields to an explicit flag. It is not a
/// marker the snapshot carries, and the registry is edited whenever a model is
/// supported, so a refusal there would let an unrelated registry edit turn a
/// working `--draft-kind mtp` run into a hard refusal.
#[test]
fn the_flag_outranks_the_registry_inference() {
    let declared = Declared::from_snapshot("Gemma4ForConditionalGeneration", "gemma4");
    assert_eq!(
        declared,
        Declared::FullModel,
        "precondition: a registered arch"
    );
    assert_eq!(
        decide_draft_kind(
            Some(DraftKind::Mtp),
            declared,
            "Gemma4ForConditionalGeneration",
            "gemma4"
        )
        .ok(),
        Some(DraftKind::Mtp)
    );
}

/// A sidecar flag over a registered full model is refused before a weight is
/// read: the snapshot carries no such head, and the loader it would reach
/// materialises the whole verifier first and then dies on a tensor name that
/// names neither side. The refusal names both, and the two ways out.
#[test]
fn a_sidecar_flag_over_a_full_model_is_refused() {
    let declared = Declared::from_snapshot("Gemma4ForConditionalGeneration", "gemma4");
    for kind in [DraftKind::Eagle3, DraftKind::DFlash, DraftKind::DFlash2] {
        let msg = decide_draft_kind(
            Some(kind),
            declared,
            "Gemma4ForConditionalGeneration",
            "gemma4",
        )
        .err()
        .map_or_else(String::new, |e| e.to_string());
        assert!(
            msg.contains(&format!("--draft-kind {kind}")),
            "names the flag: {msg}"
        );
        assert!(
            msg.contains("Gemma4ForConditionalGeneration") && msg.contains("full model"),
            "names the snapshot and what it is: {msg}"
        );
        assert!(
            msg.contains(&format!("{kind} sidecar")) && msg.contains("two_model"),
            "names both ways out: {msg}"
        );
    }
}

// ── round block ──────────────────────────────────────────────────────────────

/// One flag value is one round block, whichever drafter runs: the sidecars
/// take it whole and the two-model loop drafts one fewer and records it back
/// as `k + 1`. A block too small to hold a draft token is refused by name.
#[test]
fn one_flag_value_is_one_round_block() {
    assert_eq!(round_block(None).ok(), Some(DEFAULT_DRAFT_BLOCK_SIZE));
    for block in [MIN_DRAFT_BLOCK_SIZE, 5, 16] {
        assert_eq!(round_block(Some(block)).ok(), Some(block));
        assert_eq!(
            drafted_per_round(block) + 1,
            block,
            "the two-model loop records k + 1, which must be the block the flag named"
        );
    }
    for block in [0, MIN_DRAFT_BLOCK_SIZE - 1] {
        let msg = round_block(Some(block))
            .err()
            .map_or_else(String::new, |e| e.to_string());
        assert!(
            msg.contains(&format!("block size {block}"))
                && msg.contains(&format!("at least {MIN_DRAFT_BLOCK_SIZE}")),
            "{msg}"
        );
    }
}

// ── classify_mtp_draft ───────────────────────────────────────────────────────

#[test]
fn qwen35_mtp_sidecar_routes_to_mtp_drafter() {
    // mlx-community Qwen3.5 MTP sidecars carry arch == model_type == qwen3_5_mtp.
    assert_eq!(
        classify_mtp_draft("qwen3_5_mtp", "qwen3_5_mtp"),
        MtpDraftFamily::Qwen35Mtp
    );
    // model_type alone is sufficient — real Qwen3.6-35B-A3B-MTP-5bit has no
    // `architectures` array; only `model_type=qwen3_5_mtp` is set.
    assert_eq!(
        classify_mtp_draft("", "qwen3_5_mtp"),
        MtpDraftFamily::Qwen35Mtp
    );
}

#[test]
fn qwen35_mtp_substring_match() {
    // arch substring: tolerate minor variant suffixes in the arch string.
    assert_eq!(
        classify_mtp_draft("qwen3_5_mtp_head_v2", ""),
        MtpDraftFamily::Qwen35Mtp,
        "arch substring containing qwen3_5_mtp should route to Qwen35Mtp"
    );
    // model_type substring: same tolerance on model_type side.
    assert_eq!(
        classify_mtp_draft("", "qwen3_5_mtp_variant"),
        MtpDraftFamily::Qwen35Mtp,
        "model_type substring containing qwen3_5_mtp should route to Qwen35Mtp"
    );
}

#[test]
fn empty_config_falls_through_to_qwen35_mtp() {
    // Both fields absent: legacy blank-config snapshot. The downstream
    // MtpDrafter::load warns and proceeds by tensor names — this must NOT
    // be rejected as Unsupported (regression from issue #23 fix scope).
    assert_eq!(
        classify_mtp_draft("", ""),
        MtpDraftFamily::Qwen35Mtp,
        "blank arch+model_type must fall through to Qwen35Mtp, not Unsupported"
    );
}

#[test]
fn gemma4_assistant_routes_to_assistant_drafter() {
    // Dedicated assistant snapshot: model_type gemma4_assistant.
    assert_eq!(
        classify_mtp_draft("Gemma4Assistant", "gemma4_assistant"),
        MtpDraftFamily::Gemma4Assistant
    );
    // architectures-substring path (some exports set arch, not model_type).
    assert_eq!(
        classify_mtp_draft("Gemma4AssistantForCausalLM", ""),
        MtpDraftFamily::Gemma4Assistant
    );
}

#[test]
fn plain_gemma4_draft_is_unsupported_not_qwen_fallthrough() {
    // The issue #23 case: a plain dense Gemma4 model must NOT classify as the
    // Qwen3.5 MTP sidecar (which would leak `num_experts`); it is Unsupported.
    assert_eq!(
        classify_mtp_draft("Gemma4ForConditionalGeneration", ""),
        MtpDraftFamily::Unsupported
    );
}

#[test]
fn unrelated_family_is_unsupported() {
    assert_eq!(
        classify_mtp_draft("Qwen3ForCausalLM", ""),
        MtpDraftFamily::Unsupported
    );
}

#[test]
fn plain_gemma4_reject_reason_points_at_assistant_snapshot() {
    let reason = mtp_reject_reason("Gemma4ForConditionalGeneration", "");
    // Must name the family and the actionable alternative.
    assert!(
        reason.contains("Gemma4"),
        "reason names the family: {reason}"
    );
    assert!(
        reason.contains("assistant"),
        "reason points at the assistant snapshot: {reason}"
    );
    // Must NOT mention the leaked Qwen3.5 internal error.
    assert!(
        !reason.contains("num_experts"),
        "reason must not leak the Qwen3.5 loader error: {reason}"
    );
}

#[test]
fn non_gemma_reject_reason_is_generic_and_clean() {
    let reason = mtp_reject_reason("Qwen3ForCausalLM", "qwen3");
    assert!(
        reason.contains("Qwen3ForCausalLM"),
        "names the arch: {reason}"
    );
    assert!(
        !reason.contains("num_experts"),
        "reason must not leak the Qwen3.5 loader error: {reason}"
    );
}

// ── DFlash generations ───────────────────────────────────────────────────────

/// The two DFlash checkpoints route to their own loaders, and neither flag
/// takes the other's snapshot.
///
/// `DFlash2DraftModel` contains `DFlash`, so a declaration rule that read the
/// older marker first would send every DFlash 2 snapshot to the DFlash 1
/// loader — which builds a different architecture out of the 58 tensors it
/// recognises and records the accept rate under this checkpoint's name.
#[test]
fn the_two_dflash_generations_do_not_take_each_other() {
    let one = Declared::from_snapshot("DFlashDraftModel", "qwen3");
    let two = Declared::from_snapshot("DFlash2DraftModel", "qwen3");
    assert_eq!(one, Declared::Sidecar(DraftKind::DFlash));
    assert_eq!(two, Declared::Sidecar(DraftKind::DFlash2));

    for (declared, arch, flag) in [
        (two, "DFlash2DraftModel", DraftKind::DFlash),
        (one, "DFlashDraftModel", DraftKind::DFlash2),
    ] {
        let msg = decide_draft_kind(Some(flag), declared, arch, "qwen3")
            .err()
            .map_or_else(String::new, |e| e.to_string());
        assert!(
            msg.contains(&format!("--draft-kind {flag}")),
            "names the flag: {msg}"
        );
        assert!(msg.contains(arch), "names the snapshot's own words: {msg}");
    }
}

/// The DFlash 2 refusal states the gap and closes the route a reader would try
/// next. Naming the checkpoint is not enough: `--draft-kind dflash` would load
/// it, and the row it wrote could not afterwards be told from the real
/// drafter's.
#[test]
fn the_dflash2_reject_reason_closes_the_dflash_1_route() {
    let reason = dflash2_reject_reason("DFlash2DraftModel");
    assert!(
        reason.contains("DFlash2DraftModel"),
        "names the snapshot: {reason}"
    );
    assert!(
        reason.contains("not implemented"),
        "states the gap rather than implying a bad checkpoint: {reason}"
    );
    assert!(
        reason.contains("--draft-kind dflash"),
        "closes the route a reader tries next: {reason}"
    );
    assert!(
        reason.contains("--draft-model"),
        "names the way out: {reason}"
    );
}
