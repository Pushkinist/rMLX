use super::*;
use crate::prompt_cache::{PromptCache, FNV_OFFSET};
use rmlx_kv_quant::KvQuant;

fn entry_with(kv_caches: Vec<KvCache>, ids: Vec<u32>) -> Gemma4Entry {
    let block_hashes = rmlx_kv_ssd::chained_block_hashes(&ids);
    Gemma4Entry {
        prompt_token_ids: ids,
        block_hashes,
        kv_caches,
        first_id: 0,
        first_piece: String::new(),
        kv_quant: Some(KvQuant::K8V8),
    }
}

fn entry_with_quant(
    kv_caches: Vec<KvCache>,
    ids: Vec<u32>,
    kv_quant: Option<KvQuant>,
) -> Gemma4Entry {
    let block_hashes = rmlx_kv_ssd::chained_block_hashes(&ids);
    Gemma4Entry {
        prompt_token_ids: ids,
        block_hashes,
        kv_caches,
        first_id: 0,
        first_piece: String::new(),
        kv_quant,
    }
}

/// A freshly built snapshot — every cache at offset 0 — is trimmable, so
/// `can_truncate_to_block` permits the production `CacheLookup::Prefix`
/// path. This is the cold-equal regime (cached prompt within
/// `sliding_window`, SWA ring not wrapped).
#[test]
fn can_truncate_when_no_swa_wrap() {
    let kv = vec![
        KvCache::with_quant_max_seq_window(KvQuant::K8V4, 8192, None), // full
        KvCache::with_quant_max_seq_window(KvQuant::K8V4, 8192, Some(1024)), // SWA
        KvCache::with_quant_max_seq_window(KvQuant::K8V4, 8192, Some(1024)), // SWA
    ];
    for c in &kv {
        assert!(c.is_trimmable(), "fresh cache (offset 0) must be trimmable");
    }
    let e = entry_with(kv, (0..512u32).collect());
    assert!(
        e.can_truncate_to_block(1),
        "all-trimmable snapshot must allow the block-truncate Prefix path"
    );
}

/// A flat (full-attention only) snapshot is always trimmable regardless of
/// length — mirrors mlx-lm `KVCache.is_trimmable()` -> True.
#[test]
fn flat_only_snapshot_always_trimmable() {
    let kv = vec![
        KvCache::with_quant_max_seq_window(KvQuant::K8V8, 8192, None),
        KvCache::with_quant_max_seq_window(KvQuant::Planar, 8192, None),
    ];
    let e = entry_with(kv, (0..256u32).collect());
    assert!(e.can_truncate_to_block(1));
}

/// MISMATCH path: stored `Some(K8V8)`, runtime `K8V4` → evict + miss.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn kv_quant_mismatch_evicts_and_misses() {
    let prompt_ids: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let entry = entry_with_quant(kv, prompt_ids.clone(), Some(KvQuant::K8V8));
    let mut cache: PromptCache<Gemma4Entry> = PromptCache::new(4);
    cache.push(entry);

    let runtime_quant = KvQuant::K8V4;
    let (slot_idx, _) = cache
        .find_best_prefix(&prompt_ids, FNV_OFFSET)
        .expect("block hit");
    let entry_quant = cache.slots[slot_idx].entry.kv_quant;
    assert!(entry_quant != Some(runtime_quant));
    cache.evict_slot(slot_idx);

    assert_eq!(cache.slots.len(), 0);
    assert_eq!(cache.stats().evictions, 1);
    assert!(cache.find_best_prefix(&prompt_ids, FNV_OFFSET).is_none());
}

/// BACKWARD-COMPAT path: stored `None` — treated as MISMATCH.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn kv_quant_legacy_none_evicts_and_misses() {
    let prompt_ids: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let entry = entry_with_quant(kv, prompt_ids.clone(), None);
    let mut cache: PromptCache<Gemma4Entry> = PromptCache::new(4);
    cache.push(entry);

    let runtime_quant = KvQuant::K8V8;
    let (slot_idx, _) = cache
        .find_best_prefix(&prompt_ids, FNV_OFFSET)
        .expect("block hit");
    assert!(cache.slots[slot_idx].entry.kv_quant != Some(runtime_quant));
    cache.evict_slot(slot_idx);
    assert_eq!(cache.slots.len(), 0);
    assert_eq!(cache.stats().evictions, 1);
}

/// MATCH path: stored `Some(K8V8)`, runtime `K8V8` → hit, no eviction.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn kv_quant_match_hits_no_eviction() {
    let prompt_ids: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let entry = entry_with_quant(kv, prompt_ids.clone(), Some(KvQuant::K8V8));
    let mut cache: PromptCache<Gemma4Entry> = PromptCache::new(4);
    cache.push(entry);

    let (slot_idx, _) = cache
        .find_best_prefix(&prompt_ids, FNV_OFFSET)
        .expect("block hit");
    assert_eq!(cache.slots[slot_idx].entry.kv_quant, Some(KvQuant::K8V8));
    assert_eq!(cache.slots.len(), 1);
    assert_eq!(cache.stats().evictions, 0);
}

/// gemma4's policy is `Partial` — the generate-loop is allowed to
/// take the block-aligned partial-prefix path (subject to the per-slot
/// `can_truncate_to_block` SWA gate).
#[test]
fn arch_policy_is_partial() {
    assert_eq!(PROMPT_CACHE.policy(), ReusePolicy::Partial);
}

/// / B1: `is_strict_prefix_of` is the SWA snapshot/restore "hook"
/// (Risk #1 — silently breaking this would forfeit the wrapped-SWA
/// multi-turn flat-prefill path). Preserves token-identity, BLOCK_TOKENS
/// floor, and the strict-extension constraint (>= 1 new token).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn is_strict_prefix_of_swa_snapshot_hook() {
    let cached: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let e = entry_with(kv, cached.clone());

    // Strict extension by 1 token → hit.
    let mut ext = cached.clone();
    ext.push(999_999);
    assert!(
        e.is_strict_prefix_of(&ext),
        "strict extension by >= 1 token must be a hit"
    );

    // Equal length → NOT a strict prefix (Exact path handles this case).
    assert!(
        !e.is_strict_prefix_of(&cached),
        "equal-length prompt is not a strict prefix"
    );

    // Shorter request → NOT a strict prefix (caller must shrink instead).
    let short = &cached[..BLOCK_TOKENS];
    assert!(
        !e.is_strict_prefix_of(short),
        "shorter prompt is not a strict prefix of a longer cached entry"
    );

    // Below the BLOCK_TOKENS floor → NOT a strict prefix (worthwhile gate).
    let kv2 = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let short_cached: Vec<u32> = (0..(BLOCK_TOKENS - 1) as u32).collect();
    let e_short = entry_with(kv2, short_cached.clone());
    let mut ext_short = short_cached;
    ext_short.push(42);
    assert!(
        !e_short.is_strict_prefix_of(&ext_short),
        "cached_len < BLOCK_TOKENS must NOT be a strict prefix (worthwhile floor)"
    );

    // Divergent prompt → NOT a strict prefix.
    let mut diverged = cached.clone();
    diverged[0] = 99;
    diverged.push(123);
    assert!(
        !e.is_strict_prefix_of(&diverged),
        "divergent prompt is not a strict prefix"
    );
}
