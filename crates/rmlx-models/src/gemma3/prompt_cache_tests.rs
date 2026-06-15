//! In-process unit tests for the Gemma3 prompt cache.
//!
//! These exercise the parts that do not need a loaded model or a Metal GPU
//! stream: the arch policy, the `Gemma3Entry` deep_clone round-trip over empty
//! (never-prefilled) KV caches, the SSD-hydrated entry field invariants
//! (placeholder `first_id` + `is_ssd_hydrated` flag) the real-model SSD smoke
//! relies on, and the SWA `is_hydrate_complete` predicate (complete vs
//! synthetically-incomplete). Decoded-token reuse is proven by the real-model
//! smoke, not here.
//!
//! `is_hydrate_complete` is GPU-free: a `KvQuant::K8V8` cache constructs with a
//! non-`None` storage (so `has_persistent_cache()` is `true`) and a
//! `KvQuant::None` cache constructs with `KvStorage::None` and no `decode_fp16`
//! seed (so it is payload-less) — neither path materialises any GPU array.

use super::*;
use rmlx_kv_quant::{KvCache, KvQuant};

/// Build a `Gemma3Entry` with `n_layers` empty (never-prefilled) `K8V8` KV
/// caches.
///
/// An empty cache holds no live GPU arrays, so `try_deep_clone` is a no-op
/// refcount walk that runs without a Metal stream — safe in a plain unit test.
/// `K8V8` storage is non-`None`, so every layer reports `has_persistent_cache()`
/// → the entry is hydrate-complete.
fn mock_ram_entry(n_layers: usize, prompt: &[u32], first_id: u32) -> Gemma3Entry {
    Gemma3Entry {
        prompt_token_ids: prompt.to_vec(),
        block_hashes: vec![0xABCD_u64; n_layers],
        kv_caches: (0..n_layers)
            .map(|_| KvCache::with_quant(KvQuant::K8V8))
            .collect(),
        first_id,
        first_piece: "hello".to_string(),
        kv_quant: Some(KvQuant::K8V8),
        is_ssd_hydrated: false,
    }
}

/// Gemma3's policy is hard-gated `ExactOnly` (SWA-first): the shared consume
/// engine must only ever yield `Exact` or `Miss` for this arch (no partial /
/// strict-prefix reuse — gemma3's ring/mask differs from gemma4).
#[test]
fn arch_policy_is_exact_only() {
    assert_eq!(PROMPT_CACHE.policy(), ReusePolicy::ExactOnly);
}

/// `deep_clone` copies every metadata field and produces an independent entry
/// with the same KV-cache cardinality (SWA ring included). Empty caches carry
/// no GPU data, so the clone is a refcount walk; this pins the field-copy
/// contract that the RAM Exact-hit reuse relies on.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: deep_clone over empty KV caches cannot fail — .expect documents the invariant"
)]
fn entry_deep_clone_preserves_fields() {
    let prompt = [1_u32, 2, 3, 4];
    let entry = mock_ram_entry(3, &prompt, 42);
    let cloned = entry
        .deep_clone()
        .expect("deep_clone of empty-cache entry must succeed");

    assert_eq!(cloned.prompt_token_ids, entry.prompt_token_ids);
    assert_eq!(cloned.block_hashes, entry.block_hashes);
    assert_eq!(cloned.kv_caches.len(), entry.kv_caches.len());
    assert_eq!(cloned.first_id, 42);
    assert_eq!(cloned.first_piece, "hello");
    assert_eq!(cloned.kv_quant, Some(KvQuant::K8V8));
    assert!(!cloned.is_ssd_hydrated);
}

/// The `PromptCacheEntry` accessors expose the entry's fields verbatim. A
/// pure-attention Gemma3 entry has no GDN recurrent state, so `lin_caches()` is
/// empty — this is what makes the blanket `SpillSink` carry KV only.
#[test]
fn entry_accessors_match_fields() {
    let prompt = [7_u32, 8, 9];
    let entry = mock_ram_entry(2, &prompt, 5);

    assert_eq!(entry.prompt_token_ids(), &prompt);
    assert_eq!(entry.block_hashes(), &[0xABCD_u64, 0xABCD_u64]);
    assert_eq!(entry.kv_caches().len(), 2);
    assert_eq!(entry.kv_quant(), Some(KvQuant::K8V8));
    assert!(!entry.is_ssd_hydrated());
    assert!(
        entry.lin_caches().is_empty(),
        "pure-attention arch has no GDN lin state"
    );
}

/// A non-hydrated partial match is never reusable under ExactOnly: the default
/// `is_reusable_prefix_of` hook returns `None` regardless of `matched_blocks`,
/// so the consume engine can only reach `Exact` or `Miss`. Gemma3 keeps the
/// default (no gemma4-style block-truncate / B1 strict-prefix).
#[test]
fn non_hydrated_partial_is_not_reusable() {
    let entry = mock_ram_entry(1, &[1, 2, 3], 0);
    // A longer incoming prompt that the cached one is a prefix of: still None
    // because the entry is not SSD-hydrated and the default hook declines.
    assert!(entry
        .is_reusable_prefix_of(&[1, 2, 3, 4, 5], false, 1)
        .is_none());
}

/// An SSD-hydrated entry carries placeholder first-token fields (`first_id == 0`,
/// empty `first_piece`) and the `is_ssd_hydrated` flag set, mirroring exactly
/// what `SsdHydrate::hydrate` constructs. The consume engine excludes such an
/// entry from the Exact fast path so the generate loop recomputes the real first
/// token via re-prefill. Even as a strict prefix of a longer prompt it is
/// declined here (the default hook returns `None`), so Gemma3 stays Exact-only.
#[test]
fn ssd_hydrated_entry_field_invariants() {
    let hydrated = Gemma3Entry {
        prompt_token_ids: vec![10, 11, 12],
        block_hashes: vec![0xFEED_u64],
        kv_caches: vec![KvCache::with_quant(KvQuant::K8V8)],
        first_id: 0,
        first_piece: String::new(),
        kv_quant: Some(KvQuant::K8V8),
        is_ssd_hydrated: true,
    };

    assert_eq!(hydrated.first_id, 0, "SSD block stores no real first token");
    assert!(hydrated.first_piece.is_empty());
    assert!(hydrated.is_ssd_hydrated());
    // ExactOnly Gemma3 declines even a hydrated strict-prefix reuse (default
    // hook → None), unlike the moe HydratedTail seam.
    assert!(hydrated
        .is_reusable_prefix_of(&[10, 11, 12, 13], true, 0)
        .is_none());
}

/// A fully RAM-resident snapshot (every layer cache `K8V8`, so non-`None`
/// storage) is hydrate-complete: every attended layer carries payload.
#[test]
fn is_hydrate_complete_true_for_complete_entry() {
    let entry = mock_ram_entry(4, &[1, 2, 3, 4], 7);
    assert!(
        entry.is_hydrate_complete(),
        "all-K8V8 caches have persistent storage → complete"
    );
}

/// An entry with a payload-less SWA layer — a `KvQuant::None` cache that carries
/// neither a persistent storage buffer (`KvStorage::None`) nor an off-storage
/// bf16 decode seed — is hydrate-incomplete, exactly the shape an SSD-hydrated
/// gemma3 entry has after its rotating ring was dropped. The engine would
/// exclude such an entry from prefix reuse (and the Exact arm already excludes
/// it via `!is_ssd_hydrated`).
#[test]
fn is_hydrate_complete_false_for_payloadless_swa_layer() {
    // Three persistent (full-attention) layers + one payload-less SWA layer.
    let mut kv_caches: Vec<KvCache> = (0..3).map(|_| KvCache::with_quant(KvQuant::K8V8)).collect();
    // `KvQuant::None` constructs `KvStorage::None`; with no `with_decode_fp16_seed`
    // it has no persistent cache and no decode seed → payload-less.
    kv_caches.push(KvCache::with_quant(KvQuant::None));

    let entry = Gemma3Entry {
        prompt_token_ids: vec![1, 2, 3, 4],
        block_hashes: vec![0xABCD_u64],
        kv_caches,
        first_id: 0,
        first_piece: String::new(),
        kv_quant: Some(KvQuant::None),
        is_ssd_hydrated: true,
    };

    assert!(
        !entry.is_hydrate_complete(),
        "a None-storage SWA layer with no bf16 seed makes the entry incomplete"
    );
}
