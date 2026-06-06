//! Qwen3.5 MoE prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats
//! readback, last-bytes counter) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Qwen3.5-MoE-specific bits:
//!
//! - `Qwen35MoeEntry`: `Vec<KvCache>` + `Vec<LinearAttnCache>` (hybrid GDN).
//! - `impl PromptCacheEntry for Qwen35MoeEntry`: deep_clone trims only KV
//!   caches; `lin_caches` are NOT truncated — they hold sequence-end recurrent
//!   state and are re-run on the tail.
//! - `impl SpillSink<Qwen35MoeEntry> for SsdSpiller`: hybrid spill — both KV
//!   and lin_caches.
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

use crate::prompt_cache::{
    chained_block_hashes_seeded, ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy,
    SpillSink, SsdHydrate, BLOCK_TOKENS, FNV_OFFSET,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SpillJob, SsdHydrator, SsdSpiller};

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
        })
    }

    fn truncate_kv_to(&mut self, prefix_len: usize) {
        for kv in &mut self.kv_caches {
            if kv.offset() > 0 {
                kv.truncate_to(prefix_len as i32);
            }
        }
        // lin_caches are NOT truncated — they are re-run on the tail.
    }

    fn truncate_kv_to_block(&mut self, block_count: usize) {
        // Delegates to truncate_kv_to. ReusePolicy::ExactOnly means the
        // generate-loop never actually calls this (partial hits degrade to
        // Miss); the impl is retained for the trait contract + SSD code paths.
        self.truncate_kv_to(block_count * BLOCK_TOKENS);
    }

    fn kv_bytes(&self) -> u64 {
        let kv: u64 = self.kv_caches.iter().map(|c| c.approx_bytes()).sum();
        let lin: u64 = self.lin_caches.iter().map(|c| c.approx_bytes()).sum();
        kv + lin
    }
}

// ---------------------------------------------------------------------------
// SSD-spill sink — hybrid: spills both `kv_caches` and `lin_caches`.
// ---------------------------------------------------------------------------

impl SpillSink<Qwen35MoeEntry> for SsdSpiller {
    fn spill(&self, entry: &Qwen35MoeEntry) {
        let Some(&hash) = entry.block_hashes.last() else {
            return; // no full block → no stable spill key
        };
        let Some(kv_quant) = entry.kv_quant else {
            return; // unknown quant → cannot tag the block
        };
        let layout_key = self.layout_key();
        let kv_caches: Result<Vec<KvCache>> =
            entry.kv_caches.iter().map(|c| c.try_deep_clone()).collect();
        let lin_caches: Result<Vec<LinearAttnCache>> = entry
            .lin_caches
            .iter()
            .map(|c| c.try_deep_clone())
            .collect();
        let (kv_caches, lin_caches) = match (kv_caches, lin_caches) {
            (Ok(kv), Ok(lin)) => (kv, lin),
            (Err(e), _) | (_, Err(e)) => {
                tracing::warn!(error = %e, "kv-spill: qwen3.5-moe cache clone failed, skipping spill");
                return;
            }
        };
        let materialized = kv_caches
            .iter()
            .try_for_each(|c| c.eval_for_spill())
            .and_then(|()| lin_caches.iter().try_for_each(|c| c.eval_for_spill()));
        if let Err(e) = materialized {
            tracing::warn!(error = %e, "kv-spill: qwen3.5-moe eval-for-spill failed, skipping spill");
            return;
        }
        self.try_spill(SpillJob {
            hash,
            layout_key,
            model_id: self.model_id().to_string(),
            kv_quant,
            kv_caches,
            lin_caches,
        });
    }
}

// ---------------------------------------------------------------------------
// SSD-hydrate source — hybrid: reconstructs both KV and lin caches.
// ---------------------------------------------------------------------------

impl SsdHydrate<Qwen35MoeEntry> for SsdHydrator {
    fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<Qwen35MoeEntry>> {
        let Some(block) = self.lookup(prompt_ids)? else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches,
        } = block;
        let block_hashes = chained_block_hashes_seeded(&prompt_ids, FNV_OFFSET ^ self.layout_key());
        Ok(Some(Qwen35MoeEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            lin_caches,
            first_id: 0,
            first_piece: String::new(),
            kv_quant: Some(self.kv_quant()),
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
