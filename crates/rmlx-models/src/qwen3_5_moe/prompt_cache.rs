//! Qwen3.5 MoE prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats
//! readback) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Qwen3.5-MoE-specific bits:
//!
//! - `Qwen35MoeEntry`: `Vec<KvCache>` + `Vec<LinearAttnCache>` (hybrid GDN),
//!   and the [`Anchor`] it saved inside its prompt.
//! - `impl PromptCacheEntry for Qwen35MoeEntry`: deep_clone + KV/lin accessors,
//!   and the reuse hook. The SSD spill path is the blanket
//!   `SpillSink<E> for SsdSpiller` in `crate::prompt_cache`, which spills both
//!   `kv_caches()` and `lin_caches()` (hybrid).
//! - `impl HydratedEntry for Qwen35MoeEntry`: what an SSD-restored block
//!   becomes here. This is the one entry that keeps the block's `lin_caches`;
//!   the pure-attention entries drop them. The probe itself is the blanket
//!   `SsdHydrate<E> for SsdHydrator` in `rmlx-kv-ssd`.
//!
//! The per-arch shell (`PROMPT_CACHE`, ensure/stats wrappers) is pure
//! delegation to `ArchPromptCache`; SSD attach is invoked directly on the
//! static from `ssd_tier.rs`.
//!
//! ## Reuse
//!
//! A recurrent state has no sequence axis, so it cannot be cut back to a
//! shorter prefix the way K/V can. The entry therefore resumes a request from a
//! state it actually holds:
//!
//! - its own, at the end of its prompt, when the request extends the whole
//!   stored prompt (`StrictPrefix`);
//! - its [`Anchor`], at the last full block of the stored prompt that leaves a
//!   non-empty tail, when the request first differs at or after that block
//!   (`BlockTruncate`; the K/V is cut back to the anchor).
//!
//! Anything else misses. The arch is `Partial`: the engine leaves a RAM entry's
//! reuse to the entry's hook, which holds both rules.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::{Error, Result};

use crate::prompt_cache::{
    Anchor, ArchPromptCache, CacheStats, PromptCacheEntry, ReuseKind, ReusePolicy, BLOCK_TOKENS,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, HydratedEntry};

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
    /// The recurrent state at the last full block of the prompt that leaves a
    /// non-empty tail ([`Anchor::pos_for`]); `None` below one block, and on an
    /// entry restored from the SSD tier.
    pub(crate) anchor: Option<Anchor>,
    /// Argmax token from the first decode step after prefill.
    pub(crate) first_id: u32,
    /// Decoded piece for `first_id`.
    pub(crate) first_piece: String,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was written.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier and therefore
    /// stores only the BLOCK-ALIGNED prefix (`prompt_token_ids.len()` is a
    /// multiple of `BLOCK_TOKENS`). The tail tokens of the original request
    /// were never prefilled into these caches, and `first_id` is a placeholder:
    /// the engine never serves such an entry as an Exact hit, and the reuse hook
    /// resumes it, re-prefilling only the tail on top of the restored KV/lin
    /// state.
    ///
    /// MUST be set only in `HydratedEntry::from_hydrated`; never set by the normal
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
            anchor: self
                .anchor
                .as_ref()
                .map(Anchor::try_deep_clone)
                .transpose()?,
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

    // Hybrid GDN arch: real recurrent caches. The default `truncate_kv_to`
    // deliberately cannot reach these (recurrent state is re-run on the tail,
    // never sliced), and the default `kv_bytes` sums their `resident_bytes`.
    fn lin_caches(&self) -> &[LinearAttnCache] {
        &self.lin_caches
    }
    fn anchor(&self) -> Option<&Anchor> {
        self.anchor.as_ref()
    }
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.

    /// Which state of this entry a request may resume from.
    ///
    /// Both rules compare token ids, so they hold for a RAM entry and an
    /// SSD-hydrated one alike: a hydrated block's caches sit exactly at its
    /// claimed prefix.
    ///
    /// 1. The stored prompt is a STRICT prefix of the request: resume the whole
    ///    entry at `stored.len()`. The strict less-than is load-bearing: it
    ///    leaves the equal-length case to the Exact arm, and a hydrated
    ///    equal-length entry (placeholder first token) to a re-prefill.
    /// 2. Otherwise the [`Anchor`], when the request agrees with the stored
    ///    prompt up to it, leaves a non-empty tail past it, and every K/V cache
    ///    can be cut back to it.
    fn is_reusable_prefix_of(
        &self,
        prompt_ids: &[u32],
        _is_ssd_hydrated: bool,
        _matched_blocks: usize,
    ) -> Option<ReuseKind> {
        let stored = self.prompt_token_ids();
        if stored.len() < prompt_ids.len() && prompt_ids.starts_with(stored) {
            return Some(ReuseKind::StrictPrefix {
                prefix_len: stored.len(),
            });
        }
        let anchor = self.anchor.as_ref()?;
        let effective_blocks = anchor.pos / BLOCK_TOKENS;
        (prompt_ids != stored
            && anchor.pos < prompt_ids.len()
            && prompt_ids.starts_with(stored.get(..anchor.pos)?)
            && self.can_truncate_to_block(effective_blocks))
        .then_some(ReuseKind::BlockTruncate { effective_blocks })
    }

    /// `StrictPrefix` clones the entry as it is. `BlockTruncate` clones it, cuts
    /// the K/V back to the anchor and puts the anchor's state in place of the
    /// end state. The anchor the clone keeps is never used after a
    /// `BlockTruncate`: the resumed request captures the anchor of its own entry
    /// from the resume point on (`anchor_target`), the resume point included.
    fn prepare_reuse(&self, kind: ReuseKind) -> Result<Self> {
        let ReuseKind::BlockTruncate { effective_blocks } = kind else {
            return self.deep_clone();
        };
        let anchor = self
            .anchor
            .as_ref()
            .filter(|a| a.pos == effective_blocks * BLOCK_TOKENS)
            .ok_or_else(|| {
                Error::Model(format!(
                    "qwen3_5moe prompt cache: no anchor at {effective_blocks} blocks to resume from"
                ))
            })?;
        let mut cloned = self.deep_clone()?;
        cloned.truncate_kv_to_block(effective_blocks)?;
        cloned.lin_caches = anchor.try_deep_clone()?.lin_caches;
        Ok(cloned)
    }
}

// ---------------------------------------------------------------------------
// SSD-spill sink — the blanket `impl SpillSink<E> for SsdSpiller` in
// `crate::prompt_cache` covers Qwen3.5-MoE: `kv_caches()` + `lin_caches()`
// both return the real per-layer slices, so the spill job carries both the
// attention KV and the GDN recurrent state.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate entry — hybrid: keeps both the KV and the lin caches.
// ---------------------------------------------------------------------------

impl HydratedEntry for Qwen35MoeEntry {
    const SHARES_KV: bool = crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS;

    fn from_hydrated(block: HydratedBlock, block_hashes: Vec<u64>, kv_quant: KvQuant) -> Self {
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches,
        } = block;
        Self {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            lin_caches,
            anchor: None,
            first_id: 0,
            first_piece: String::new(),
            kv_quant: Some(kv_quant),
            // Block-aligned prefix only; see `Qwen35MoeEntry::is_ssd_hydrated`.
            is_ssd_hydrated: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Global cache instance — unified ArchPromptCache shell.
// ---------------------------------------------------------------------------

/// Per-arch shell. `Partial`: the engine leaves a RAM entry's reuse to
/// [`Qwen35MoeEntry`]'s hook, which only ever offers a state the entry holds.
pub(crate) static PROMPT_CACHE: ArchPromptCache<Qwen35MoeEntry> = ArchPromptCache::new(
    "Qwen3_5MoeForConditionalGeneration",
    ReusePolicy::Partial,
    crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS,
);

/// active SSD-tier `layout_key`, or `0` when tier OFF.
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global prompt cache is initialised with exactly `capacity` slots;
/// `0` disables it (nothing is stored, every request prefills).
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Qwen3.5 MoE prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
