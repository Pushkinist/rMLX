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
    assert_eq!(arch_default("qwen3_5_moe"), Some(64));
    assert_eq!(arch_default("gemma3"), Some(256));
    assert_eq!(arch_default("gemma4"), Some(512));
    assert_eq!(arch_default("qwen2"), Some(256));
    assert_eq!(arch_default("laguna"), Some(256));
    assert_eq!(arch_default("bitnet"), Some(64));
    assert_eq!(arch_default("unknown_arch"), None);
}

#[test]
fn unknown_arch_falls_back() {
    // Env-var path may override; only assert pure default behaviour.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() || env::var("RMLX_PREFILL_CHUNK_BOGUS").is_ok() {
        return;
    }
    assert_eq!(prefill_chunk_for("bogus"), FALLBACK);
}
