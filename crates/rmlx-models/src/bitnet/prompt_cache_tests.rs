//! In-process unit tests for the BitNet prompt cache.
//!
//! These exercise the parts that do not need a loaded model or a Metal GPU
//! stream: the arch policy, the `BitNetEntry` deep_clone round-trip over empty
//! (never-prefilled) KV caches, and the SSD-hydrated entry field invariants
//! (placeholder `first_id` + `is_ssd_hydrated` flag) the real-model SSD smoke
//! relies on. Decoded-token reuse is proven by the real-model smoke, not here.

use super::*;
use rmlx_kv_quant::{KvCache, KvQuant};

/// Build a `BitNetEntry` with `n_layers` empty (never-prefilled) KV caches.
///
/// An empty cache holds no live GPU arrays, so `try_deep_clone` is a no-op
/// refcount walk that runs without a Metal stream — safe in a plain unit test.
fn mock_ram_entry(n_layers: usize, prompt: &[u32], first_id: u32) -> BitNetEntry {
    BitNetEntry {
        prompt_token_ids: prompt.to_vec(),
        block_hashes: vec![0xABCD_u64; n_layers],
        kv_caches: (0..n_layers)
            .map(|_| KvCache::with_quant(KvQuant::None))
            .collect(),
        first_id,
        first_piece: "hello".to_string(),
        kv_quant: Some(KvQuant::None),
        is_ssd_hydrated: false,
    }
}

/// BitNet's policy is hard-gated `ExactOnly`: the shared consume engine must
/// only ever yield `Exact` or `Miss` for this arch (no partial-prefix reuse).
#[test]
fn arch_policy_is_exact_only() {
    assert_eq!(PROMPT_CACHE.policy(), ReusePolicy::ExactOnly);
}

/// `deep_clone` copies every metadata field and produces an independent entry
/// with the same KV-cache cardinality. (Empty caches carry no GPU data, so the
/// clone is a refcount walk; this pins the field-copy contract.)
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
    assert_eq!(cloned.kv_quant, Some(KvQuant::None));
    assert!(!cloned.is_ssd_hydrated);
}

/// The `PromptCacheEntry` accessors expose the entry's fields verbatim. A
/// pure-attention BitNet entry has no GDN recurrent state, so `lin_caches()` is
/// empty — this is what makes the blanket `SpillSink` carry KV only.
#[test]
fn entry_accessors_match_fields() {
    let prompt = [7_u32, 8, 9];
    let entry = mock_ram_entry(2, &prompt, 5);

    assert_eq!(entry.prompt_token_ids(), &prompt);
    assert_eq!(entry.block_hashes(), &[0xABCD_u64, 0xABCD_u64]);
    assert_eq!(entry.kv_caches().len(), 2);
    assert_eq!(entry.kv_quant(), Some(KvQuant::None));
    assert!(!entry.is_ssd_hydrated());
    assert!(
        entry.lin_caches().is_empty(),
        "pure-attention arch has no GDN lin state"
    );
}

/// A non-hydrated partial match is never reusable under ExactOnly: the default
/// `is_reusable_prefix_of` hook returns `None` regardless of `matched_blocks`,
/// so the consume engine can only reach `Exact` or `Miss`.
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
/// what `HydratedEntry::from_hydrated` constructs. The consume engine excludes such an
/// entry from the Exact fast path so the generate loop recomputes the real
/// first token via re-prefill. Even as a strict prefix of a longer prompt it is
/// declined here (the default hook returns `None`), so BitNet stays Exact-only.
#[test]
fn ssd_hydrated_entry_field_invariants() {
    let hydrated = BitNetEntry {
        prompt_token_ids: vec![10, 11, 12],
        block_hashes: vec![0xFEED_u64],
        kv_caches: vec![KvCache::with_quant(KvQuant::None)],
        first_id: 0,
        first_piece: String::new(),
        kv_quant: Some(KvQuant::None),
        is_ssd_hydrated: true,
    };

    assert_eq!(hydrated.first_id, 0, "SSD block stores no real first token");
    assert!(hydrated.first_piece.is_empty());
    assert!(hydrated.is_ssd_hydrated());
    // ExactOnly BitNet declines even a hydrated strict-prefix reuse (default
    // hook → None), unlike the moe HydratedTail seam.
    assert!(hydrated
        .is_reusable_prefix_of(&[10, 11, 12, 13], true, 0)
        .is_none());
}

/// One K8V4 paged layer: an empty storage, or one whose K slot holds a page.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a CPU page allocation must succeed, and the panic names the slab"
)]
fn paged_layer(with_page: bool) -> KvCache {
    use rmlx_kv_quant::paged::PagedKStorage;
    use rmlx_kv_quant::storage::KvStorage;

    let k = with_page.then(|| {
        let mut k = PagedKStorage::new(4096, 16, 4);
        let id = k
            .codes
            .alloc(rmlx_mlx::Device::Cpu)
            .expect("allocate a K page");
        k.scales
            .alloc(rmlx_mlx::Device::Cpu)
            .expect("allocate a K scale page");
        k.block_table.push(id);
        k.total_tokens = 13;
        k.shape = vec![1, 1, 13, 64];
        k
    });
    let storage = KvStorage::Paged {
        quant: KvQuant::K8V4,
        k,
        v_k8: None,
        v_planar: None,
    };
    KvCache::from_storage(
        storage,
        4096,
        KvQuant::K8V4,
        if with_page { 13 } else { 0 },
        0,
        rmlx_core::DispatchPolicy::default(),
        false,
    )
}

/// An exact prompt repeat whose stored entry holds a paged layer with pages is
/// a miss (`DeepCloneErr`), not a hit on an empty clone. The slot stays, so the
/// next repeat is the same miss. An entry whose paged layer is empty is served,
/// so the miss comes from the pages and not from the setup.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test assertion: the cache was installed and the entry pushed just above"
)]
fn exact_repeat_of_a_paged_entry_with_pages_is_a_miss() {
    use crate::prompt_cache::{
        chained_block_hashes_seeded, request_cache_seed, Consumed, MissReason, PromptCache,
        BLOCK_TOKENS,
    };

    const SIG: u64 = 0x0bad_cafe;
    let prompt = (0..BLOCK_TOKENS as u32).collect::<Vec<_>>();
    let seed = request_cache_seed(
        0,
        KvQuant::K8V4,
        1,
        crate::bitnet::SHARES_KV_ACROSS_LAYERS,
        SIG,
    );
    let consume = |with_page: bool| {
        let arch: ArchPromptCache<BitNetEntry> = ArchPromptCache::new(
            "test-paged-clone",
            ReusePolicy::ExactOnly,
            crate::bitnet::SHARES_KV_ACROSS_LAYERS,
        );
        arch.with_inner_mut(|guard| {
            let mut cache = PromptCache::new(4);
            cache.push(BitNetEntry {
                prompt_token_ids: prompt.clone(),
                block_hashes: chained_block_hashes_seeded(&prompt, seed),
                kv_caches: vec![paged_layer(with_page)],
                first_id: 7,
                first_piece: "x".to_string(),
                kv_quant: Some(KvQuant::K8V4),
                is_ssd_hydrated: false,
            });
            *guard = Some(cache);
        });
        let first = arch.consume(&prompt, KvQuant::K8V4, 1, false, SIG);
        let second = arch.consume(&prompt, KvQuant::K8V4, 1, false, SIG);
        let slots = arch.with_inner_mut(|guard| guard.as_ref().map_or(0, |c| c.slots.len()));
        (first, second, slots)
    };

    let (first, second, slots) = consume(true);
    for (label, out) in [("first", first), ("second", second)] {
        assert!(
            matches!(out, Consumed::Miss(MissReason::DeepCloneErr)),
            "{label} repeat: a paged entry with pages must miss on its refused clone"
        );
    }
    assert_eq!(slots, 1, "the refused clone must leave the source slot");

    let (first, _, _) = consume(false);
    assert!(
        matches!(first, Consumed::Exact(_)),
        "an entry whose paged layer holds no page must be served"
    );
}
