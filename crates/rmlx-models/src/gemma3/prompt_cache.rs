//! Gemma3 prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats readback,
//! and the find → SSD-hydrate retry → quant-guard → Exact → Miss consume
//! decision) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Gemma3-specific bits:
//!
//! - `Gemma3Entry`: `Vec<KvCache>` only (pure-attention arch — no GDN state).
//!   Same shape as `Gemma4Entry`: gemma3 also has sliding-window-attention (SWA)
//!   layers backed by a `RotatingKvCache`, so the entry must carry the full
//!   layer-cache vector (SWA ring included) for a RAM Exact `deep_clone` to be
//!   safe.
//! - `impl PromptCacheEntry for Gemma3Entry`: deep_clone + KV accessors, plus an
//!   `is_hydrate_complete` SWA-completeness override. The two prefix-reuse hooks
//!   (`is_reusable_prefix_of`, `prepare_reuse`) take the trait defaults, which
//!   give correct [`ReusePolicy::ExactOnly`] behaviour: a non-hydrated partial
//!   match is never reusable, and the only reuse the engine would permit (a
//!   hydrated strict-prefix) is declined here (`is_reusable_prefix_of` → `None`),
//!   so the only reachable consume outcomes are `Exact` and `Miss`.
//! - `impl SsdHydrate<Gemma3Entry> for SsdHydrator`: pure-attention hydrate (the
//!   reconstructed block carries `kv_caches` only; ExactOnly still forces any
//!   hydrated entry back through re-prefill before it can decode). The SSD spill path is the blanket
//!   `SpillSink<E> for SsdSpiller` in `crate::prompt_cache`.
//!
//! ## Reuse policy — Exact-only (SWA-first)
//!
//! Gemma3 uses [`ReusePolicy::ExactOnly`]. Gemma3's sliding-window ring and
//! SWA-mask differ from gemma4, so the partial / strict-prefix snapshot-restore
//! path that gemma4 runs under `ReusePolicy::Partial` is NOT enabled here yet —
//! a Partial promotion (reusing a gemma4-style prefix predicate + completeness
//! gate) is a separate follow-up that must first prove a strict-prefix restore
//! on gemma3's ring. Under ExactOnly the only reuse is a RAM `Exact` hit on a
//! non-hydrated entry: that is a full in-memory `deep_clone` of every layer
//! cache (SWA ring included), so it is correct by construction.
//!
//! ## SWA-hydrate completeness
//!
//! An SSD-hydrated entry is never reused under ExactOnly anyway (the Exact arm
//! declines `is_ssd_hydrated`, and the default `is_reusable_prefix_of` returns
//! `None`), so it always falls to a full re-prefill. The
//! `is_hydrate_complete` override below still encodes a conservative payload
//! predicate to future-proof a later Partial promotion.

#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;

use crate::prompt_cache::{ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy, SsdHydrate};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Post-prefill snapshot for one Gemma3 request.
///
/// Pure-attention arch (no GDN recurrent state), but with SWA layers backed by
/// a `RotatingKvCache`. Same shape as `Gemma4Entry`; the layer-cache vector
/// carries the full per-layer KV (SWA ring included) so a RAM Exact `deep_clone`
/// reuses the exact attention context.
pub(crate) struct Gemma3Entry {
    /// Full prompt token IDs used to fill this slot.
    pub(crate) prompt_token_ids: Vec<u32>,
    /// Chained 256-token block digests of `prompt_token_ids` (trailing partial
    /// block excluded). Computed at construction.
    pub(crate) block_hashes: Vec<u64>,
    /// Post-prefill KV caches (one per decoder layer; SWA + full-attention).
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

impl Gemma3Entry {
    /// True iff every layer this architecture attends to carries real K/V
    /// payload — i.e. the snapshot has a complete attention context.
    ///
    /// Gemma3 stays ExactOnly, so hydrated entries are not reused today, but we
    /// still keep a conservative payload predicate here for future Partial
    /// promotion. A layer is treated as complete only when it carries actual
    /// persistent storage or an explicit bf16 seed.
    ///
    /// Under ExactOnly such a hydrated entry is never reused anyway, but the
    /// predicate documents the SWA truth and gates the engine's prefix-reuse
    /// path for a future Partial promotion.
    pub(crate) fn is_hydrate_complete(&self) -> bool {
        self.kv_caches
            .iter()
            .all(|c| c.has_persistent_cache() || c.decode_fp16_kv().is_some())
    }
}

impl PromptCacheEntry for Gemma3Entry {
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

    // Pure-attention arch: no GDN linear state.
    fn lin_caches(&self) -> &[LinearAttnCache] {
        &[]
    }

    /// Delegates to the inherent [`Gemma3Entry::is_hydrate_complete`]: an
    /// SSD-hydrated entry whose payload is incomplete is excluded from any
    /// future prefix reuse.
    /// (Under the current ExactOnly policy it is excluded from the Exact arm by
    /// `!is_ssd_hydrated` regardless; this override documents the SWA truth and
    /// future-proofs a Partial promotion.)
    fn is_hydrate_complete(&self) -> bool {
        Gemma3Entry::is_hydrate_complete(self)
    }
    // is_reusable_prefix_of / prepare_reuse: trait defaults. The defaults give
    // correct ExactOnly behaviour (no partial reuse, deep_clone on the
    // unreachable reuse path). Gemma3's SWA ring/mask differs from gemma4, so a
    // gemma4-style block-truncate / B1 strict-prefix path is deliberately NOT
    // enabled — promotion to Partial is a separate follow-up.
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.
}

// ---------------------------------------------------------------------------
// SSD-spill sink — the blanket `impl SpillSink<E> for SsdSpiller` in
// `crate::prompt_cache` covers Gemma3 (pure-attention: `lin_caches()` is `&[]`,
// so the spill job carries `kv_caches` only).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate source
// ---------------------------------------------------------------------------

/// Hydrate a `Gemma3Entry` from the SSD tier on a RAM-cache miss.
///
/// Pure-attention arch: the reconstructed block carries `kv_caches` only (the
/// `lin_caches` from the block are discarded). The matched block-aligned prefix
/// token IDs become the entry's `prompt_token_ids`; block hashes are recomputed
/// and the runtime `kv_quant` recorded. `first_id` / `first_piece` are
/// sentinels (the SSD block stores no first decode token), so the entry is
/// flagged `is_ssd_hydrated = true`; the consume engine excludes it from the
/// Exact fast path and the generate loop recomputes the real first token via
/// re-prefill.
impl SsdHydrate<Gemma3Entry> for SsdHydrator {
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> Result<Option<Gemma3Entry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(prompt_ids, seed, kv_quant, policy)?
        else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            ..
        } = block;
        Ok(Some(Gemma3Entry {
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

/// per-arch shell with `ReusePolicy::ExactOnly`. Gemma3 has SWA layers, but
/// gemma3's ring / SWA-mask differs from gemma4, so the prefix-reuse path is not
/// wired here — the only reuse is a RAM Exact hit (full `deep_clone`, SWA ring
/// included). Promotion to `Partial` is a separate follow-up.
pub(crate) static PROMPT_CACHE: ArchPromptCache<Gemma3Entry> =
    ArchPromptCache::new("Gemma3ForConditionalGeneration", ReusePolicy::ExactOnly);

/// active SSD-tier `layout_key` for the gemma3 cache, or `0` when the tier is
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

/// Read the current hit/miss/bytes stats for the Gemma3 prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
