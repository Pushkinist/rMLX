//! Maple prompt-cache entry + global.
//!
//! Pure-attention hybrid SWA + full KV (no GatedDeltaNet). The layer-cache
//! vector carries the rotating SWA ring so a RAM Exact `deep_clone` is safe.
//!
//! Hydrate is `SsdHydrator::lookup_seeded` + a struct literal.
//!
//! ## Reuse policy — Exact-only (SWA)
//!
//! Maple uses `ReusePolicy::ExactOnly`. Partial reuse is declined: the only
//! reusable result is a full-token-equality RAM Exact hit, or a complete SSD
//! hydration whose codec restored every SWA ring and can be safely extended.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;

use crate::prompt_cache::{ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy, SsdHydrate};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{ExactReplayMetadata, HydratedBlock, SsdHydrator};

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
    /// Exact first-token replay persisted with a complete SSD snapshot.
    /// `None` is retained for legacy/incomplete records and must fail closed.
    pub(crate) exact_replay: Option<ExactReplayMetadata>,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was written.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier. Legacy
    /// records may have only a block-aligned identity and placeholder token;
    /// those remain marked hydrated and are rejected by the exact-replay and
    /// completeness gates. MUST be set only in `SsdHydrate::hydrate`.
    pub(crate) is_ssd_hydrated: bool,
}

impl MapleEntry {
    /// True iff a hydrated entry carries real K/V payload for every layer.
    ///
    /// The empty vector is deliberately rejected for hydrated entries: old
    /// SSD payloads could decode to no layers and `Iterator::all` would make
    /// that malformed state vacuously complete. A rotating layer is complete
    /// when the generic codec has restored its live ring; dense layers remain
    /// complete when they carry persistent storage or a decode seed.
    pub(crate) fn is_hydrate_complete(&self) -> bool {
        if !self.is_ssd_hydrated {
            return true;
        }
        !self.kv_caches.is_empty()
            && self.kv_caches.iter().all(|c| {
                // A geometry-only legacy cache can report a storage
                // variant even when it contains no tokens. Require a
                // positive offset as well as a payload marker so a fake
                // empty K8V8 cache cannot pass the hydrate gate.
                c.offset() > 0
                    && (c.has_persistent_cache() || c.is_rotating() || c.decode_fp16_kv().is_some())
            })
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
            exact_replay: self.exact_replay.clone(),
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

    fn exact_replay(&self) -> Option<&ExactReplayMetadata> {
        self.exact_replay.as_ref()
    }

    fn lin_caches(&self) -> &[LinearAttnCache] {
        &[]
    }

    fn is_hydrate_complete(&self) -> bool {
        MapleEntry::is_hydrate_complete(self)
    }
}

// ---------------------------------------------------------------------------
// SSD-hydrate source — lookup_seeded; acceptance remains fail-closed unless the
// complete hybrid SWA/full snapshot and exact replay metadata are restored.
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
            exact_replay,
            kv_caches,
            lin_caches: _,
        } = block;
        let (first_id, first_piece) = exact_replay.as_ref().map_or((0, String::new()), |replay| {
            (replay.id, replay.piece.clone())
        });
        Ok(Some(MapleEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            first_id,
            first_piece,
            exact_replay,
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

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
