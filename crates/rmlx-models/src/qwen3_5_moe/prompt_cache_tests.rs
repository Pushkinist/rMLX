use super::*;

/// Qwen3.5-MoE's policy is hard-gated `ExactOnly` — the
/// generate-loop's `CacheLookup` match must NEVER take the partial-prefix
/// path because the GDN `lin_caches` cannot be block-truncated.
#[test]
fn arch_policy_is_exact_only() {
    assert_eq!(PROMPT_CACHE.policy(), ReusePolicy::ExactOnly);
}
