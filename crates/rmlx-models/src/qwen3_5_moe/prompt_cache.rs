//! Qwen3.5 MoE prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats
//! readback, last-bytes counter) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Qwen3.5-MoE-specific bits:
//!
//! - `Qwen35MoeEntry`: `Vec<KvCache>` + `Vec<LinearAttnCache>` (hybrid GDN).
//! - `impl PromptCacheEntry for Qwen35MoeEntry`: deep_clone + KV/lin accessors.
//!   The trait-default `truncate_kv_to` trims only the KV caches; `lin_caches`
//!   are NOT truncated — they hold sequence-end recurrent state re-run on the
//!   tail (the default body structurally cannot reach them). The SSD spill path
//!   is the blanket `SpillSink<E> for SsdSpiller` in `crate::prompt_cache`,
//!   which spills both `kv_caches()` and `lin_caches()` (hybrid).
//! - `impl SsdHydrate<Qwen35MoeEntry> for SsdHydrator`: hybrid hydrate.
//!
//! The per-arch shell (`PROMPT_CACHE`, attach/ensure/stats wrappers) is pure
//! delegation to `ArchPromptCache`.
//!
//! ## Reuse policy — Exact-only (hard runtime gate, )
//!
//! Qwen3.5-MoE uses [`ReusePolicy::ExactOnly`]. Block-truncated partial-prefix
//! reuse is **unsafe** here: `truncate_kv_to_block` deliberately leaves
//! `lin_caches` untouched (recurrent state cannot be reconstructed from a
//! block-truncated KV — re-running the GDN over the original prompt's tail and
//! then a different new tail would produce wrong state), so partial hits must
//! degrade to a full re-prefill (Miss). The Exact path is full-token-equality
//! reuse (skips re-prefill entirely), keyed off `prompt_token_ids()`.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;

use crate::prompt_cache::{ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy, SsdHydrate};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Post-prefill snapshot for one Qwen3.5 MoE request.
pub(crate) struct Qwen35MoeEntry {
    /// Full prompt token IDs used to fill this slot.
    pub(crate) prompt_token_ids: Vec<u32>,
    /// Chained 256-token block digests of `prompt_token_ids` (trailing
    /// partial block excluded). Computed at construction.
    pub(crate) block_hashes: Vec<u64>,
    /// Post-prefill KV caches (one per layer; GDN layers have empty `KvCache`).
    pub(crate) kv_caches: Vec<KvCache>,
    /// Post-prefill linear-attention recurrent states (GDN layers populated;
    /// full-attention layers have empty `LinearAttnCache`).
    pub(crate) lin_caches: Vec<LinearAttnCache>,
    /// Argmax token from the first decode step after prefill.
    pub(crate) first_id: u32,
    /// Decoded piece for `first_id`.
    pub(crate) first_piece: String,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was written.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier and therefore
    /// stores only the BLOCK-ALIGNED prefix (`prompt_token_ids.len()` is a
    /// multiple of `BLOCK_TOKENS`). The tail tokens of the original request
    /// were never prefilled into these caches. The generate loop detects this
    /// flag to take the `HydratedTail` path: it re-prefills only the missing
    /// tail on top of the restored KV/lin state, then decodes normally.
    ///
    /// MUST be set only in `SsdHydrate::hydrate`; never set by the normal
    /// RAM-cache push path. Do NOT use the `first_id == 0` heuristic as a
    /// substitute — `<bos>` token id is 0 for some models.
    pub(crate) is_ssd_hydrated: bool,
}

impl PromptCacheEntry for Qwen35MoeEntry {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }

    fn deep_clone(&self) -> Result<Self> {
        let kv_caches: Result<Vec<_>> = self.kv_caches.iter().map(|c| c.try_deep_clone()).collect();
        let lin_caches: Result<Vec<_>> =
            self.lin_caches.iter().map(|c| c.try_deep_clone()).collect();
        Ok(Self {
            prompt_token_ids: self.prompt_token_ids.clone(),
            block_hashes: self.block_hashes.clone(),
            kv_caches: kv_caches?,
            lin_caches: lin_caches?,
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

    // Hybrid GDN arch: real recurrent caches. The default `truncate_kv_to`
    // deliberately cannot reach these (recurrent state is re-run on the tail,
    // never sliced), and the default `kv_bytes` sums their `approx_bytes`.
    fn lin_caches(&self) -> &[LinearAttnCache] {
        &self.lin_caches
    }
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.
}

// ---------------------------------------------------------------------------
// SSD-spill sink — the blanket `impl SpillSink<E> for SsdSpiller` in
// `crate::prompt_cache` covers Qwen3.5-MoE: `kv_caches()` + `lin_caches()`
// both return the real per-layer slices, so the spill job carries both the
// attention KV and the GDN recurrent state.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate source — hybrid: reconstructs both KV and lin caches.
// ---------------------------------------------------------------------------

impl SsdHydrate<Qwen35MoeEntry> for SsdHydrator {
    fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<Qwen35MoeEntry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(prompt_ids)? else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches,
        } = block;
        Ok(Some(Qwen35MoeEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            lin_caches,
            first_id: 0,
            first_piece: String::new(),
            kv_quant: Some(self.kv_quant()),
            // SSD-hydrated entries store only the block-aligned prefix.
            // The generate loop uses this flag to re-prefill the tail tokens
            // before decoding (HydratedTail path). See `Qwen35MoeEntry::is_ssd_hydrated`.
            is_ssd_hydrated: true,
        }))
    }
}

// ---------------------------------------------------------------------------
// Global cache instance — unified ArchPromptCache shell.
// ---------------------------------------------------------------------------

/// per-arch shell with hard-gated `ReusePolicy::ExactOnly`. Partial /
/// block-aligned reuse is unsafe for the hybrid GDN arch (the recurrent
/// `lin_caches` cannot be reconstructed from a block-truncated KV) — the
/// generate-loop's `CacheLookup` match enforces this at runtime by routing
/// any `Some(_)` non-Exact match to `CacheLookup::Miss`.
pub(crate) static PROMPT_CACHE: ArchPromptCache<Qwen35MoeEntry> =
    ArchPromptCache::new("Qwen3_5MoeForConditionalGeneration", ReusePolicy::ExactOnly);

/// Wire the spiller + hydrator onto the Qwen3.5-MoE prompt cache.
pub(crate) fn attach_ssd_tier(
    namespace: &str,
    kv_quant: KvQuant,
    layout_key: u64,
    device: rmlx_mlx::Device,
) {
    PROMPT_CACHE.attach_ssd_tier(namespace, kv_quant, layout_key, device);
}

/// active SSD-tier `layout_key`, or `0` when tier OFF.
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global prompt cache is initialised with at least `capacity` slots.
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Qwen3.5 MoE prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

/// Read the KV-cache bytes (KV + linear-attn) from the last completed Qwen3.5
/// MoE request.
pub fn read_kv_cache_bytes() -> u64 {
    PROMPT_CACHE.read_kv_cache_bytes()
}

/// Record the KV-cache byte total for the just-completed request.
pub(crate) fn store_kv_cache_bytes(n: u64) {
    PROMPT_CACHE.store_kv_cache_bytes(n);
}

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
