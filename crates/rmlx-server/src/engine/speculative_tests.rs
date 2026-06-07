//! Unit tests for the `--draft-kind mtp` arch-family dispatch (issue #23).
//!
//! These cover the pure classifier `classify_mtp_draft` and the rejection
//! message builder `mtp_reject_reason` — the load-time routing decision that
//! keeps a plain Gemma4 draft from falling through to the Qwen3.5 MTP sidecar
//! loader (which would leak a `text_config missing num_experts` error).

use super::{classify_mtp_draft, mtp_reject_reason, MtpDraftFamily};

#[test]
fn qwen35_mtp_sidecar_routes_to_mtp_drafter() {
    // mlx-community Qwen3.5 MTP sidecars carry arch == model_type == qwen3_5_mtp.
    assert_eq!(
        classify_mtp_draft("qwen3_5_mtp", "qwen3_5_mtp"),
        MtpDraftFamily::Qwen35Mtp
    );
    // model_type alone is sufficient (some exports leave architectures empty).
    assert_eq!(
        classify_mtp_draft("", "qwen3_5_mtp"),
        MtpDraftFamily::Qwen35Mtp
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
