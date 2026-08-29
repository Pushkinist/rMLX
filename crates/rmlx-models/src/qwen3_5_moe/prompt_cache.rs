//! Qwen3.5 MoE prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats
//! readback) is unified in
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
//! The per-arch shell (`PROMPT_CACHE`, ensure/stats wrappers) is pure
//! delegation to `ArchPromptCache`; SSD attach is invoked directly on the
//! static from `ssd_tier.rs`.
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
use rmlx_core::DispatchPolicy;

use crate::prompt_cache::{
    ArchPromptCache, CacheStats, PromptCacheEntry, ReuseKind, ReusePolicy, SsdHydrate,
};
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

    fn is_ssd_hydrated(&self) -> bool {
        self.is_ssd_hydrated
    }

    // Hybrid GDN arch: real recurrent caches. The default `truncate_kv_to`
    // deliberately cannot reach these (recurrent state is re-run on the tail,
    // never sliced), and the default `kv_bytes` sums their `resident_bytes`.
    fn lin_caches(&self) -> &[LinearAttnCache] {
        &self.lin_caches
    }
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.

    /// HydratedTail seam: an SSD-hydrated entry whose stored block-aligned
    /// prefix is a STRICT prefix of the incoming prompt may be reused — the
    /// hydrated KV + GDN `lin_caches` are the recurrent state at exactly
    /// `t = prefix_len` of THIS same prompt, so re-prefilling only
    /// `prompt_ids[prefix_len..]` on top is sequentially correct (identical to
    /// pausing/resuming the original prefill at the block boundary).
    ///
    /// All three guards must hold:
    /// 1. The entry was promoted from the SSD tier (`is_ssd_hydrated`).
    /// 2. Stored ids are STRICTLY shorter than the incoming prompt. The strict
    ///    less-than is load-bearing: it excludes the block-aligned EQUAL-length
    ///    hydrated entry (no tail), which must fall through to Miss so its
    ///    placeholder first token is recomputed rather than replayed.
    /// 3. Stored ids are byte-identical to the matching leading subsequence of
    ///    `prompt_ids` (guarantees the tail is the same prompt's continuation,
    ///    not a divergent one).
    ///
    /// A non-hydrated partial match never reaches this hook (the ExactOnly
    /// policy gates it out in the engine). Unlike gemma4, there is no
    /// `>= BLOCK_TOKENS` floor: a hydrated entry is block-aligned by
    /// construction, so the strict less-than alone is the correct gate.
    fn is_reusable_prefix_of(
        &self,
        prompt_ids: &[u32],
        is_ssd_hydrated: bool,
        _matched_blocks: usize,
    ) -> Option<ReuseKind> {
        let stored = self.prompt_token_ids();
        if is_ssd_hydrated && stored.len() < prompt_ids.len() && prompt_ids.starts_with(stored) {
            Some(ReuseKind::StrictPrefix {
                prefix_len: stored.len(),
            })
        } else {
            None
        }
    }
    // is_hydrate_complete / prepare_reuse: trait defaults. The GDN `lin_caches`
    // recurrent state is carried by `deep_clone` (the default `prepare_reuse`)
    // and is NEVER block-truncated — moe only ever yields `StrictPrefix`, and
    // the default `truncate_kv_to` structurally cannot reach `lin_caches`.
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
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> Result<Option<Qwen35MoeEntry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(
            prompt_ids, seed, kv_quant, policy,
            // No cross-layer KV sharing on this stack: nothing reads a
            // Mixed/RotK bf16 mirror, so a hydrated cache builds none.
            false,
        )?
        else {
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
            kv_quant: Some(kv_quant),
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
/// `lin_caches` cannot be reconstructed from a block-truncated KV) — the shared
/// consume engine enforces this at runtime: under `ExactOnly` a non-hydrated
/// partial match is forbidden, and the only permitted reuse is a hydrated
/// strict-prefix (HydratedTail) via `is_reusable_prefix_of`; everything else
/// degrades to a full re-prefill (Miss).
pub(crate) static PROMPT_CACHE: ArchPromptCache<Qwen35MoeEntry> = ArchPromptCache::new(
    "Qwen3_5MoeForConditionalGeneration",
    ReusePolicy::ExactOnly,
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
