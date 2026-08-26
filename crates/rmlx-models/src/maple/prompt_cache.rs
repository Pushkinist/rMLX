//! Maple prompt-cache entry + global.
//!
//! Pure-attention hybrid SWA + full KV (no GatedDeltaNet). Same shape as
//! Gemma3's prompt-cache entry: the layer-cache vector carries the rotating
//! SWA ring so a RAM Exact `deep_clone` is safe. SSD hydrate of SWA rings is
//! incomplete (the ring is not serialised).
//!
//! Hydrate is `SsdHydrator::lookup_seeded` + a struct literal.
//!
//! ## Reuse policy — Exact-only (SWA)
//!
//! Maple uses `ReusePolicy::ExactOnly`. The SWA rotating ring cannot be
//! reconstructed from a block-truncated / SSD-hydrated prefix, so partial
//! reuse is declined (trait-default `is_reusable_prefix_of` → `None`). The
//! only reuse is a RAM Exact hit: full in-memory `deep_clone` of every layer
//! cache, SWA ring included.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;

use crate::prompt_cache::{ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy, SsdHydrate};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Post-prefill snapshot for one Maple request.
pub(crate) struct MapleEntry {
    /// Full prompt token IDs used to fill this slot.
    pub(crate) prompt_token_ids: Vec<u32>,
    /// Chained 256-token block digests of `prompt_token_ids` (trailing
    /// partial block excluded). Computed at construction.
    pub(crate) block_hashes: Vec<u64>,
    /// Post-prefill KV caches (one per decoder layer; SWA + full-attention).
    pub(crate) kv_caches: Vec<KvCache>,
    /// Argmax token from the first decode step after prefill.
    pub(crate) first_id: u32,
    /// Decoded piece for `first_id`.
    pub(crate) first_piece: String,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was written.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier (block-aligned
    /// prefix only; `first_id` / `first_piece` are placeholders). MUST be set
    /// only in `SsdHydrate::hydrate`.
    pub(crate) is_ssd_hydrated: bool,
}

impl MapleEntry {
    /// True iff every layer carries real K/V payload.
    ///
    /// SWA rotating rings are not serialised to SSD; on hydrate they come back
    /// as payload-less `KvStorage::None`. Under ExactOnly such an entry is
    /// never reused (`!is_ssd_hydrated` on the Exact arm); this predicate
    /// documents the SWA truth for a later Partial promotion.
    pub(crate) fn is_hydrate_complete(&self) -> bool {
        self.kv_caches
            .iter()
            .all(|c| c.has_persistent_cache() || c.decode_fp16_kv().is_some())
    }
}

impl PromptCacheEntry for MapleEntry {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }

    fn deep_clone(&self) -> Result<Self> {
        let kv_caches: Result<Vec<_>> = self.kv_caches.iter().map(|c| c.try_deep_clone()).collect();
        Ok(Self {
            prompt_token_ids: self.prompt_token_ids.clone(),
            block_hashes: self.block_hashes.clone(),
            kv_caches: kv_caches?,
            first_id: self.first_id,
            first_piece: self.first_piece.clone(),
            kv_quant: self.kv_quant,
            is_ssd_hydrated: self.is_ssd_hydrated,
        })
    }

    fn kv_caches(&self) -> &[KvCache] {
        &self.kv_caches
    }

    fn kv_caches_mut(&mut self) -> &mut [KvCache] {
        &mut self.kv_caches
    }

    fn kv_quant(&self) -> Option<KvQuant> {
        self.kv_quant
    }

    fn is_ssd_hydrated(&self) -> bool {
        self.is_ssd_hydrated
    }

    fn lin_caches(&self) -> &[LinearAttnCache] {
        &[]
    }

    fn is_hydrate_complete(&self) -> bool {
        MapleEntry::is_hydrate_complete(self)
    }
}

// ---------------------------------------------------------------------------
// SSD-hydrate source — lookup_seeded; SWA rings come back payload-less.
// ---------------------------------------------------------------------------

impl SsdHydrate<MapleEntry> for SsdHydrator {
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> Result<Option<MapleEntry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(prompt_ids, seed, kv_quant, policy)?
        else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches: _,
        } = block;
        Ok(Some(MapleEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            first_id: 0,
            first_piece: String::new(),
            kv_quant: Some(kv_quant),
            is_ssd_hydrated: true,
        }))
    }
}

// ---------------------------------------------------------------------------
// Global cache instance
// ---------------------------------------------------------------------------

/// Exact-only: SWA ring is only safe to reuse via a full RAM `deep_clone`.
pub(crate) static PROMPT_CACHE: ArchPromptCache<MapleEntry> =
    ArchPromptCache::new("MapleForCausalLM", ReusePolicy::ExactOnly);

/// Active SSD-tier `layout_key`, or `0` when the tier is OFF.
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global prompt cache is initialised with exactly `capacity` slots;
/// `0` disables it.
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Maple prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}
