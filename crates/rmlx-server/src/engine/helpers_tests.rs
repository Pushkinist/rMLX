//! Model-free tests for the server's KV-codec resolution.
//!
//! Two seams, both reachable without a snapshot: what `--kv-quant auto`
//! resolves to at load, and what one request ends up running. They are tested
//! together because the whole point of the pair is that they cannot disagree —
//! a second resolver that re-picks per request is how the launch-time answer
//! and the served answer drift apart.
//!
//! What these tests pin is *consistency*, not the codec's identity: they compare
//! against `DEFAULT_KV_QUANT` rather than against `KvQuant::None`, so changing
//! the default does not turn them red. The identity is pinned once, in
//! `rmlx-cli`'s `auto_kv_quant_resolves_to_bf16_for_every_arch_branch`. Split on
//! purpose — a value that appears on both sides of an assertion checks nothing.

use super::{kv_quant_for_request, resolve_kv_quant_for_load};
use rmlx_kv_quant::KvQuant;
use rmlx_models::kv_cache::DEFAULT_KV_QUANT;

/// One synthetic `config.json` per architecture class the server can load.
/// Every one of these used to resolve to a different codec.
fn arch_fixtures() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "qwen3.5-moe",
            r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"]}"#,
        ),
        (
            "qwen3.5-dense-paro",
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],
                "quantization_config":{"quant_method":"paroquant"}}"#,
        ),
        (
            "qwen3-dense-2bit",
            r#"{"architectures":["Qwen3ForCausalLM"],
                "quantization":{"group_size":64,"bits":2}}"#,
        ),
        (
            "gemma3",
            r#"{"architectures":["Gemma3ForConditionalGeneration"]}"#,
        ),
        (
            "gemma4-small",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],
                "text_config":{"hidden_size":1536}}"#,
        ),
        (
            "gemma4-dense",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],
                "text_config":{"hidden_size":5376}}"#,
        ),
        (
            "gemma4-unified-12b",
            r#"{"architectures":["Gemma4UnifiedForConditionalGeneration"],
                "text_config":{"hidden_size":3840}}"#,
        ),
        (
            "qwen3-vl-moe",
            r#"{"architectures":["Qwen3VLMoeForConditionalGeneration"]}"#,
        ),
        (
            "unknown-arch",
            r#"{"architectures":["NoSuchArchForCausalLM"]}"#,
        ),
    ]
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture JSON is a literal in this file; a parse failure is a test bug"
)]
fn auto_resolves_to_the_engine_default_for_every_arch() {
    for (name, body) in arch_fixtures() {
        let cfg: rmlx_loader::ModelConfig = serde_json::from_str(body).unwrap();
        let (resolved, explicit) = resolve_kv_quant_for_load(&cfg, None, name);
        assert_eq!(
            resolved,
            Some(DEFAULT_KV_QUANT),
            "auto load resolution for '{name}' is {resolved:?}"
        );
        assert!(!explicit, "auto mode must not report a user-explicit codec");
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture JSON is a literal in this file; a parse failure is a test bug"
)]
fn an_explicit_launch_codec_survives_every_arch() {
    for (name, body) in arch_fixtures() {
        let cfg: rmlx_loader::ModelConfig = serde_json::from_str(body).unwrap();
        let (resolved, explicit) = resolve_kv_quant_for_load(&cfg, Some(KvQuant::Planar), name);
        assert_eq!(
            resolved,
            Some(KvQuant::Planar),
            "explicit --kv-quant was rewritten on '{name}'"
        );
        assert!(explicit, "an explicit codec must report user_explicit");
    }
}

#[test]
fn a_request_without_its_own_codec_runs_the_launch_codec() {
    // Auto mode: the launch value is the engine default and the request keeps it.
    assert_eq!(
        kv_quant_for_request(None, Some(DEFAULT_KV_QUANT)),
        Some(DEFAULT_KV_QUANT)
    );
    // Explicit launch mode: the request keeps that too. Nothing between the
    // load resolver and the request may substitute a different codec — a
    // per-prompt-length policy sat here and silently did exactly that.
    for launch in [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
    ] {
        assert_eq!(
            kv_quant_for_request(None, Some(launch)),
            Some(launch),
            "launch codec {launch:?} was not what the request ran"
        );
    }
}

#[test]
fn a_per_request_codec_wins_over_the_launch_codec() {
    assert_eq!(
        kv_quant_for_request(Some(KvQuant::K8V8), Some(KvQuant::None)),
        Some(KvQuant::K8V8)
    );
    assert_eq!(
        kv_quant_for_request(Some(KvQuant::None), Some(KvQuant::K8V8)),
        Some(KvQuant::None)
    );
}

#[test]
fn no_launch_codec_and_no_request_codec_stays_unset() {
    // The arch entry points read `Option<KvQuant>`; `None` there means "use the
    // arch's own default", which is the same constant. Manufacturing a codec
    // here would hide a resolver that failed to run.
    assert_eq!(kv_quant_for_request(None, None), None);
}
