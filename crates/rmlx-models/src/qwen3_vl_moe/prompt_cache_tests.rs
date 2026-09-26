//! In-process unit tests for the Qwen3-VL-MoE prompt cache.
//!
//! These exercise the parts that do not need a loaded model or a Metal GPU
//! stream: the arch policy, the `Qwen3VlMoeEntry` deep_clone round-trip over
//! empty (never-prefilled) KV caches, the accessor contract, and the
//! SSD-hydrated entry field invariants (placeholder `first_id` +
//! `is_ssd_hydrated` flag) the real-model SSD smoke relies on. Decoded-token
//! reuse is proven by the real-model smoke, not here.

use super::*;
use rmlx_kv_quant::{KvCache, KvQuant};

/// Build a `Qwen3VlMoeEntry` with `n_layers` empty (never-prefilled) KV caches.
///
/// An empty cache holds no live GPU arrays, so `try_deep_clone` is a no-op
/// refcount walk that runs without a Metal stream — safe in a plain unit test.
fn mock_ram_entry(n_layers: usize, prompt: &[u32], first_id: u32) -> Qwen3VlMoeEntry {
    Qwen3VlMoeEntry {
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

/// Qwen3-VL-MoE's policy is hard-gated `ExactOnly`: the shared consume engine
/// must only ever yield `Exact` or `Miss` for this arch (no partial-prefix
/// reuse — the text path never sees image ids, and the MoE is FFN-only).
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
/// plain-attention Qwen3-VL-MoE entry has no GDN recurrent state, so
/// `lin_caches()` is empty — this is what makes the blanket `SpillSink` carry KV
/// only (the MoE is FFN-only and never produces a KV-side cache).
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
        "plain-attention text decoder has no GDN lin state"
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
/// entry from the Exact fast path so the generate loop recomputes the real first
/// token via re-prefill. Even as a strict prefix of a longer prompt it is
/// declined here (the default hook returns `None`), so Qwen3-VL-MoE stays
/// Exact-only.
#[test]
fn ssd_hydrated_entry_field_invariants() {
    let hydrated = Qwen3VlMoeEntry {
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
    // ExactOnly Qwen3-VL-MoE declines even a hydrated strict-prefix reuse
    // (default hook → None), unlike the moe HydratedTail seam in qwen3_5_moe.
    assert!(hydrated
        .is_reusable_prefix_of(&[10, 11, 12, 13], true, 0)
        .is_none());
}

/// An SSD source that records the per-layer codec vector each probe hands it
/// and reports a miss.
struct LayerQuantsProbe {
    seen: std::sync::Arc<std::sync::Mutex<Vec<Vec<KvQuant>>>>,
}

impl rmlx_kv_ssd::SsdHydrate<Qwen3VlMoeEntry> for LayerQuantsProbe {
    fn hydrate(
        &self,
        _prompt_ids: &[u32],
        _seed: u64,
        _kv_quant: KvQuant,
        layer_quants: &[KvQuant],
        _policy: rmlx_core::DispatchPolicy,
    ) -> Result<Option<Qwen3VlMoeEntry>> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(layer_quants.to_vec());
        Ok(None)
    }
}

/// This arch builds every layer at the requested codec, so an SSD hydrate must
/// build every layer at that codec too. A boundary layer hydrated at the
/// boundary floor would hold a base-width store under a floor codec.
///
/// Reads the arch's own `PROMPT_CACHE`, the static the generate path consumes
/// through, and the vector its consume hands the SSD source. No other test in
/// this binary touches this static.
#[test]
fn a_hydrate_builds_every_layer_at_the_requested_codec() {
    const N_LAYERS: usize = 12;
    let base = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    PROMPT_CACHE.with_inner_mut(|g| {
        let mut cache = crate::prompt_cache::PromptCache::new(4);
        cache.set_ssd_source(Box::new(LayerQuantsProbe {
            seen: std::sync::Arc::clone(&seen),
        }));
        *g = Some(cache);
    });
    let prompt: Vec<u32> = (0..2 * rmlx_kv_ssd::BLOCK_TOKENS as u32).collect();
    let _ = PROMPT_CACHE.consume(&prompt, base, N_LAYERS, false, 0x5eed);
    PROMPT_CACHE.with_inner_mut(|g| *g = None);

    let seen = seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        seen,
        vec![vec![base; N_LAYERS]],
        "one probe, every layer at the requested codec"
    );
}
