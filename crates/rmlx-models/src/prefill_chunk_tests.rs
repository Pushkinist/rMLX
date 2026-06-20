use super::*;

#[test]
fn defaults_match_recommendations() {
    // Lock the per-arch defaults so tuning changes are intentional.
    // Test is racy w.r.t. env vars; we only assert when no override
    // is set so CI runs (which set RMLX_PREFILL_CHUNK=64 for VG.2)
    // don't false-fail.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() {
        return;
    }
    assert_eq!(arch_default("qwen3"), Some(256));
    assert_eq!(arch_default("qwen3_5_moe"), Some(2048));
    assert_eq!(arch_default("qwen3_vl_moe"), Some(512));
    assert_eq!(arch_default("gemma3"), Some(256));
    assert_eq!(arch_default("gemma4"), Some(512));
    assert_eq!(arch_default("qwen2"), Some(256));
    assert_eq!(arch_default("laguna"), Some(256));
    assert_eq!(arch_default("bitnet"), Some(64));
    assert_eq!(arch_default("unknown_arch"), None);
}

#[test]
fn module_key_for_class_maps_supported_classes() {
    // Each supported config `architectures[0]` class maps to the same
    // module-style key its own generate path passes to `prefill_chunk_for`.
    assert_eq!(
        module_key_for_class("Gemma4ForConditionalGeneration"),
        "gemma4"
    );
    assert_eq!(
        module_key_for_class("Gemma4UnifiedForConditionalGeneration"),
        "gemma4"
    );
    assert_eq!(
        module_key_for_class("Gemma3ForConditionalGeneration"),
        "gemma3"
    );
    assert_eq!(module_key_for_class("Qwen2ForCausalLM"), "qwen2");
    assert_eq!(module_key_for_class("Qwen3ForCausalLM"), "qwen3");
    assert_eq!(module_key_for_class("LagunaForCausalLM"), "laguna");
    assert_eq!(
        module_key_for_class("Qwen3_5MoeForConditionalGeneration"),
        "qwen3_5_moe"
    );
    assert_eq!(
        module_key_for_class("Qwen3_5ForConditionalGeneration"),
        "qwen3_5_moe"
    );
    assert_eq!(module_key_for_class("BitNetForCausalLM"), "bitnet");
    // Qwen3-VL-MoE chunks its image prefill (native tiling → thousands of soft
    // tokens would trip the Metal watchdog in one forward).
    assert_eq!(
        module_key_for_class("Qwen3VLMoeForConditionalGeneration"),
        "qwen3_vl_moe"
    );

    // Unknown classes → "" → FALLBACK chunk, never the oversized gemma4 default.
    assert_eq!(module_key_for_class("JinaEmbeddingsV4Model"), "");
    assert_eq!(module_key_for_class("TotallyUnknownArch"), "");
}

#[test]
fn module_key_resolves_to_arch_default_chunk() {
    // The class→key→chunk chain lands on the arch's known default. Racy w.r.t.
    // env vars; only assert when no global override is set.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() {
        return;
    }
    // qwen3_5_moe resolves to its own 2048 default (GDN kernel handles any T),
    // distinct from gemma4's 512.
    let key = module_key_for_class("Qwen3_5MoeForConditionalGeneration");
    assert_eq!(prefill_chunk_for(key), 2048);
    assert_eq!(
        prefill_chunk_for(module_key_for_class("Gemma4ForConditionalGeneration")),
        512
    );
    // Unknown class resolves through "" to FALLBACK, not gemma4's 512.
    assert_eq!(
        prefill_chunk_for(module_key_for_class("TotallyUnknownArch")),
        FALLBACK
    );
}

#[test]
fn unknown_arch_falls_back() {
    // Env-var path may override; only assert pure default behaviour.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() || env::var("RMLX_PREFILL_CHUNK_BOGUS").is_ok() {
        return;
    }
    assert_eq!(prefill_chunk_for("bogus"), FALLBACK);
}
