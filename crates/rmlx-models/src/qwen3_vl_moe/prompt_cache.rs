//! Qwen3-VL-MoE prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats readback,
//! last-bytes counter, and the find → SSD-hydrate retry → quant-guard → Exact →
//! Miss consume decision) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Qwen3-VL-MoE-specific bits:
//!
//! - `Qwen3VlMoeEntry`: `Vec<KvCache>` only. The Qwen3-VL text decoder is a
//!   plain GQA stack — the MoE is in the FFN and never touches the KV cache, so
//!   the entry is the same pure-attention shape as Qwen2 / Qwen3 dense (no
//!   GatedDeltaNet `lin_caches`, no qwen3-only `first_logprobs`).
//! - `impl PromptCacheEntry for Qwen3VlMoeEntry`: deep_clone + KV accessors. All
//!   three reuse hooks (`is_hydrate_complete`, `is_reusable_prefix_of`,
//!   `prepare_reuse`) take the trait defaults, which give correct
//!   [`ReusePolicy::ExactOnly`] behaviour: a non-hydrated partial match is never
//!   reusable, and the only reuse the engine permits (a hydrated strict-prefix)
//!   is declined here (`is_reusable_prefix_of` → `None`), so the only reachable
//!   consume outcomes are `Exact` and `Miss`.
//! - `impl SsdHydrate<Qwen3VlMoeEntry> for SsdHydrator`: pure-attention hydrate
//!   (the reconstructed block carries `kv_caches` only; `lin_caches` is
//!   discarded). The SSD spill path is the blanket `SpillSink<E> for SsdSpiller`
//!   in `crate::prompt_cache`, which spills `kv_caches()` only for a
//!   pure-attention entry (`lin_caches()` is `&[]`).
//!
//! ## Reuse policy — Exact-only
//!
//! Qwen3-VL-MoE uses [`ReusePolicy::ExactOnly`]. The text path is pure-attention
//! with losslessly trimmable KV, so a partial-prefix path would be technically
//! safe; but the dominant workload is the Exact-hit warm-TTFT (identical-prompt
//! repeat skips re-prefill entirely), and keeping the single Exact / Miss arm
//! set is simpler and matches the Qwen2 / Qwen3 dense policy. Image turns never
//! reach the cache (a separate image generate path, plus the consume/store
//! `has_image` gates) — the token-id key is unsafe across image spans.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;

use crate::prompt_cache::{
    ArchPromptCache, CacheStats, KvBytesSample, PromptCacheEntry, ReusePolicy, SsdHydrate,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Post-prefill snapshot for one Qwen3-VL-MoE text request.
///
/// The text decoder is plain GQA (the MoE lives in the FFN and never touches the
/// KV cache), so only `kv_caches` needs snapshotting (no recurrent state). Same
/// shape as `Qwen2Entry`.
pub(crate) struct Qwen3VlMoeEntry {
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

impl PromptCacheEntry for Qwen3VlMoeEntry {
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

    // Plain-attention text decoder: no GDN linear state.
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
// `crate::prompt_cache` covers Qwen3-VL-MoE (pure-attention: `lin_caches()` is
// `&[]`, so the spill job carries `kv_caches` only).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate source
// ---------------------------------------------------------------------------

/// Hydrate a `Qwen3VlMoeEntry` from the SSD tier on a RAM-cache miss.
///
/// Plain-attention text decoder: the reconstructed block carries `kv_caches`
/// only (the `lin_caches` from the block are discarded). The matched
/// block-aligned prefix token IDs become the entry's `prompt_token_ids`; block
/// hashes are recomputed and the runtime `kv_quant` recorded. `first_id` /
/// `first_piece` are sentinels (the SSD block stores no first decode token), so
/// the entry is flagged `is_ssd_hydrated = true`; the consume engine excludes it
/// from the Exact fast path and the generate loop recomputes the real first
/// token via re-prefill.
impl SsdHydrate<Qwen3VlMoeEntry> for SsdHydrator {
    fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<Qwen3VlMoeEntry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(prompt_ids)? else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches: _, // plain-attention text decoder has no GDN state
        } = block;
        Ok(Some(Qwen3VlMoeEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            first_id: 0,
            first_piece: String::new(),
            kv_quant: Some(self.kv_quant()),
            // Block-aligned prefix only; the placeholder first_id must not be
            // replayed — the generate loop re-prefills to recompute it.
            is_ssd_hydrated: true,
        }))
    }
}

// ---------------------------------------------------------------------------
// Global cache instance — unified ArchPromptCache shell.
// ---------------------------------------------------------------------------

/// per-arch shell with `ReusePolicy::ExactOnly`. The Qwen3-VL text decoder is
/// pure-attention with no recurrent state; the dominant workload is the
/// Exact-hit warm-TTFT, so the single Exact / Miss arm set (no partial reuse) is
/// the policy — matching Qwen2 / Qwen3 dense.
pub(crate) static PROMPT_CACHE: ArchPromptCache<Qwen3VlMoeEntry> =
    ArchPromptCache::new("Qwen3VLMoeForConditionalGeneration", ReusePolicy::ExactOnly);

/// active SSD-tier `layout_key` for the qwen3-vl-moe cache, or `0` when the tier
/// is OFF. `FNV_OFFSET ^ 0 == FNV_OFFSET` ⇒ legacy un-salted digests when no SSD
/// tier is attached, preserving byte-identical RAM-only behaviour.
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global prompt cache is initialised with at least `capacity` slots.
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Qwen3-VL-MoE prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

/// Read the KV-cache bytes from the last completed Qwen3-VL-MoE request.
pub fn read_kv_cache_bytes_sample() -> KvBytesSample {
    PROMPT_CACHE.read_kv_cache_bytes_sample()
}

/// Record the KV-cache byte total for the just-completed request.
pub(crate) fn store_kv_cache_bytes(n: u64, post: crate::decode_loop::PostDecode) {
    PROMPT_CACHE.store_kv_cache_bytes(n, post);
}

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
