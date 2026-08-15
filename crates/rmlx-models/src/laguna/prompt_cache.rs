//! Laguna prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats readback,
//! and the find → SSD-hydrate retry → quant-guard → Exact → Miss consume
//! decision) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Laguna-specific bits:
//!
//! - `LagunaEntry`: `Vec<KvCache>` only (pure-attention sparse-MoE arch; the MoE
//!   routing is a forward-pass detail that does not change KV-cache geometry —
//!   each decoder layer still maintains a standard K/V cache pair, and no GDN
//!   recurrent / linear-attention state exists).
//! - `impl PromptCacheEntry for LagunaEntry`: deep_clone + KV accessors. All
//!   three reuse hooks (`is_hydrate_complete`, `is_reusable_prefix_of`,
//!   `prepare_reuse`) take the trait defaults, which give correct
//!   [`ReusePolicy::ExactOnly`] behaviour: a non-hydrated partial match is never
//!   reusable, and the only reuse the engine permits (a hydrated strict-prefix)
//!   is declined here (`is_reusable_prefix_of` → `None`), so the only reachable
//!   consume outcomes are `Exact` and `Miss`.
//! - `impl SsdHydrate<LagunaEntry> for SsdHydrator`: pure-attention hydrate (the
//!   reconstructed block carries `kv_caches` only; `lin_caches` is discarded).
//!   The SSD spill path is the blanket `SpillSink<E> for SsdSpiller` in
//!   `crate::prompt_cache`, which spills `kv_caches()` only for a pure-attention
//!   entry (`lin_caches()` is `&[]`).
//!
//! ## Reuse policy — Exact-only
//!
//! Laguna uses [`ReusePolicy::ExactOnly`]. It is pure-attention (no recurrent
//! state); the dominant workload is the Exact-hit warm-TTFT (identical prompt
//! repeat skips re-prefill entirely), and keeping the single Exact / Miss arm set
//! is simpler and matches the Qwen2/BitNet/Qwen3 dense policy.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;

use crate::prompt_cache::{ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy, SsdHydrate};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Post-prefill snapshot for one Laguna request.
///
/// Pure-attention sparse-MoE arch: only `kv_caches` needs snapshotting (no GDN
/// recurrent state). MoE routing is a weight-side detail that does not change
/// the KV-cache geometry; the entry shape is identical to Qwen2 and BitNet.
pub(crate) struct LagunaEntry {
    /// Full prompt token IDs used to fill this slot.
    pub(crate) prompt_token_ids: Vec<u32>,
    /// Chained 256-token block digests of `prompt_token_ids` (trailing partial
    /// block excluded). Computed at construction.
    pub(crate) block_hashes: Vec<u64>,
    /// Post-prefill KV caches (one per decoder layer).
    pub(crate) kv_caches: Vec<KvCache>,
    /// Argmax token from the first decode step after prefill.
    pub(crate) first_id: u32,
    /// Decoded piece for `first_id`.
    pub(crate) first_piece: String,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was written.
    /// Read by the engine's quant-mismatch guard and by the blanket spill path.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier and therefore
    /// stores only the block-aligned prefix KV — `first_id` / `first_piece` are
    /// placeholders, not a real decode token. The consume engine excludes such
    /// an entry from the Exact fast path so it falls through to a full
    /// re-prefill that recomputes the real first token.
    ///
    /// MUST be set only in `SsdHydrate::hydrate`; never by the RAM-cache push
    /// path. Do NOT use the `first_id == 0` heuristic as a substitute —
    /// `<bos>` token id is 0 for some models.
    pub(crate) is_ssd_hydrated: bool,
}

impl PromptCacheEntry for LagunaEntry {
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

    // Pure-attention sparse-MoE arch: no GDN linear state.
    fn lin_caches(&self) -> &[LinearAttnCache] {
        &[]
    }
    // is_hydrate_complete / is_reusable_prefix_of / prepare_reuse: trait
    // defaults. The defaults give correct ExactOnly behaviour (complete by
    // construction, no partial reuse, deep_clone on the unreachable reuse path).
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.
}

// ---------------------------------------------------------------------------
// SSD-spill sink — the blanket `impl SpillSink<E> for SsdSpiller` in
// `crate::prompt_cache` covers Laguna (pure-attention: `lin_caches()` is `&[]`,
// so the spill job carries `kv_caches` only).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate source
// ---------------------------------------------------------------------------

/// Hydrate a `LagunaEntry` from the SSD tier on a RAM-cache miss.
///
/// Pure-attention arch: the reconstructed block carries `kv_caches` only (the
/// `lin_caches` from the block are discarded). The matched block-aligned prefix
/// token IDs become the entry's `prompt_token_ids`; block hashes are recomputed
/// and the runtime `kv_quant` recorded. `first_id` / `first_piece` are sentinels
/// (the SSD block stores no first decode token), so the entry is flagged
/// `is_ssd_hydrated = true`; the consume engine excludes it from the Exact fast
/// path and the generate loop recomputes the real first token via re-prefill.
impl SsdHydrate<LagunaEntry> for SsdHydrator {
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
    ) -> Result<Option<LagunaEntry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(prompt_ids, seed, kv_quant)? else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches: _, // pure-attention arch has no GDN state
        } = block;
        Ok(Some(LagunaEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            first_id: 0,
            first_piece: String::new(),
            kv_quant: Some(kv_quant),
            // Block-aligned prefix only; the placeholder first_id must not be
            // replayed — the generate loop re-prefills to recompute it.
            is_ssd_hydrated: true,
        }))
    }
}

// ---------------------------------------------------------------------------
// Global cache instance — unified ArchPromptCache shell.
// ---------------------------------------------------------------------------

/// Per-arch shell with `ReusePolicy::ExactOnly`. Laguna is pure-attention with
/// no recurrent state; the dominant workload is the Exact-hit warm-TTFT, so the
/// single Exact / Miss arm set (no partial reuse) is the policy — matching Qwen2
/// and BitNet.
pub(crate) static PROMPT_CACHE: ArchPromptCache<LagunaEntry> =
    ArchPromptCache::new("LagunaForCausalLM", ReusePolicy::ExactOnly);

/// Active SSD-tier `layout_key` for the Laguna cache, or `0` when the tier is
/// OFF. `FNV_OFFSET ^ 0 == FNV_OFFSET` ⇒ legacy un-salted digests when no SSD
/// tier is attached, preserving byte-identical RAM-only behaviour.
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global prompt cache is initialised with exactly `capacity` slots;
/// `0` disables it (nothing is stored, every request prefills).
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Laguna prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
