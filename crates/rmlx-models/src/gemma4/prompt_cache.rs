//! Gemma4 prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats
//! readback, last-bytes counter) is unified in
//! [`crate::prompt_cache::ArchPromptCache`]. This file keeps only the genuinely
//! Gemma4-specific bits:
//!
//! - `Gemma4Entry`: `Vec<KvCache>` only (pure-attention arch — no GDN state).
//! - `impl PromptCacheEntry for Gemma4Entry`: deep_clone + KV accessors;
//!   truncate / kv_bytes come from the trait defaults (pure-attention, no GDN).
//!   The SSD spill path is the blanket `SpillSink<E> for SsdSpiller` in
//!   `crate::prompt_cache` (pure-attention: `lin_caches()` is `&[]`).
//! - `impl SsdHydrate<Gemma4Entry> for SsdHydrator`: pure-attention hydrate.
//! - SWA helpers `can_truncate_to_block` / `is_strict_prefix_of` consumed by
//!   `gemma4/generate.rs`'s `CacheLookup` match (/ B1 snapshot/restore).
//!
//! The per-arch shell (`PROMPT_CACHE` static, ensure/stats wrappers) is
//! pure delegation to `ArchPromptCache`; SSD attach is invoked directly on the
//! static from `ssd_tier.rs`.
//!
//! ## SWA / RotatingKvCache note
//! Gemma4 has sliding-window attention (SWA) layers backed by a
//! `RotatingKvCache`. Once the ring wraps (offset >= sliding_window) a
//! block-aligned truncation is no longer lossless on the SWA layers; the SWA
//! gate (`can_truncate_to_block`) makes the partial-prefix path correctness-
//! safe by falling back to Miss in that regime. The B1 strict-prefix multi-
//! turn extension path (`is_strict_prefix_of`) sidesteps truncation entirely
//! and is correct for wrapped SWA.

use rmlx_core::error::Result;

use crate::prompt_cache::{
    ArchPromptCache, CacheStats, PromptCacheEntry, ReusePolicy, SsdHydrate, BLOCK_TOKENS,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Post-prefill snapshot for one Gemma4 request.
pub(crate) struct Gemma4Entry {
    /// Full prompt token IDs used to fill this slot.
    pub(crate) prompt_token_ids: Vec<u32>,
    /// Chained 256-token block digests of `prompt_token_ids` (trailing
    /// partial block excluded). Computed at construction.
    pub(crate) block_hashes: Vec<u64>,
    /// Post-prefill KV caches for all decoder layers.
    pub(crate) kv_caches: Vec<KvCache>,
    /// Argmax token from the first decode step after prefill.
    pub(crate) first_id: u32,
    /// Decoded piece for `first_id`.
    pub(crate) first_piece: String,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was written
    /// (Plan §D8 / Task 11.5). See the original commit comments for the
    /// `Option<None>` legacy-sentinel rationale.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier and therefore
    /// stores only the block-aligned prefix KV — `first_id` / `first_piece` are
    /// placeholders, not a real decode token. The generate loop excludes such
    /// an entry from the Exact fast path so it falls through to a re-prefill
    /// that recomputes the real first token.
    ///
    /// MUST be set only in `SsdHydrate::hydrate`; never by the RAM-cache push
    /// path. Do NOT use the `first_id == 0` heuristic as a substitute —
    /// `<bos>` token id is 0 for some models.
    pub(crate) is_ssd_hydrated: bool,
}

impl Gemma4Entry {
    /// True iff a block-aligned truncation to `block_count` blocks would be
    /// lossless on EVERY layer cache (mlx-lm `can_trim_prompt_cache`:
    /// `all(c.is_trimmable())`, `cache.py:88-92`).
    ///
    /// Gemma4 has SWA layers backed by a `RotatingKvCache`. Once that ring
    /// buffer wraps (cached prompt longer than `sliding_window`) it is no
    /// longer trimmable — `truncate_kv_to_block` would silently no-op on the
    /// SWA layers while truncating the full-attention layers, desyncing the
    /// caches and corrupting the re-prefilled tail (this was the original
    /// broken gemma4 Prefix path: longctx 22 -> 0 tokens). When this returns
    /// `false` the caller MUST fall back to a full re-prefill (Miss).
    pub(crate) fn can_truncate_to_block(&self, block_count: usize) -> bool {
        debug_assert!(
            block_count >= 1,
            "block_count must be >= 1 for a prefix hit"
        );
        self.kv_caches.iter().all(KvCache::is_trimmable)
    }

    /// True iff this entry's full token sequence is a STRICT prefix of
    /// `prompt_ids` (i.e. `prompt_ids` extends the cached prompt by ≥1 token).
    ///
    /// / B1: the post-prefill snapshot for `prompt_token_ids` is exactly
    /// the cache state the new prompt needs at absolute position `cached_len` —
    /// for every layer type, including SWA layers whose `RotatingKVCache` has
    /// wrapped. The deep-cloned snapshot already holds the last
    /// `sliding_window` K/V of the cached prefix (mlx-lm `state`/`meta_state`
    /// reuse contract), which is precisely and sufficiently what the tail
    /// attends to under SWA. No truncation → no SWA-wrap desync.
    ///
    /// Requires the shared prefix to be worthwhile (`cached_len >=
    /// BLOCK_TOKENS`) — same 256-token break-even floor as the block path.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn is_strict_prefix_of(&self, prompt_ids: &[u32]) -> bool {
        let cached_len = self.prompt_token_ids.len();
        cached_len >= BLOCK_TOKENS
            && cached_len < prompt_ids.len()
            && self.prompt_token_ids[..] == prompt_ids[..cached_len]
    }

    /// True iff every layer this architecture attends to carries real K/V
    /// payload — i.e. the snapshot is safe to reuse as a prefix (B1
    /// strict-prefix snapshot/restore or block-truncate) without re-prefilling
    /// the prefix.
    ///
    /// Gemma4's sliding-window (SWA) layers run a bf16 `RotatingKvCache` ring
    /// that is NOT serialised to the SSD tier; on hydrate they are
    /// reconstructed as a payload-less `KvStorage::None` (the spill writer
    /// records them geometry-only, see `block_io::write_layer`). Reusing such a
    /// hydrated entry via a prefix arm would re-prefill only the tail on top of
    /// an empty SWA prefix, giving the SWA layers the wrong attention context
    /// and corrupting the output. `KvCache::is_trimmable()` returns `true` for a
    /// `None` layer, so it is the wrong predicate to detect this.
    ///
    /// A layer is payload-less when its storage is `None` AND it carries no
    /// off-storage bf16 seed. This distinguishes a dropped SWA ring (no seed →
    /// incomplete) from a `KvQuant::None` global layer whose bf16 K/V was
    /// spilled and restored via `KvCache::with_decode_fp16_seed` (seed present →
    /// complete). A fully RAM-resident snapshot always passes (no layer is
    /// `None`-without-seed), so only hydrated SWA entries are degraded.
    pub(crate) fn is_hydrate_complete(&self) -> bool {
        self.kv_caches
            .iter()
            .all(|c| c.has_persistent_cache() || c.decode_fp16_kv().is_some())
    }
}

impl PromptCacheEntry for Gemma4Entry {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }

    fn deep_clone(&self) -> Result<Self> {
        let kv_caches: Result<Vec<_>> =
            self.kv_caches.iter().map(KvCache::try_deep_clone).collect();
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
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.
    // SWA correctness is gated upstream by `can_truncate_to_block` /
    // `is_strict_prefix_of`; the trim itself is the byte-identical default.
}

// ---------------------------------------------------------------------------
// SSD-spill sink — the blanket `impl SpillSink<E> for SsdSpiller` in
// `crate::prompt_cache` covers Gemma4 (pure-attention: `lin_caches()` is `&[]`,
// so the spill job carries no recurrent state).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate source — pure-attention: lin_caches from block discarded.
// ---------------------------------------------------------------------------

impl SsdHydrate<Gemma4Entry> for SsdHydrator {
    fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<Gemma4Entry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(prompt_ids)? else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches: _,
        } = block;
        Ok(Some(Gemma4Entry {
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

/// per-arch shell. All boilerplate (static lock, attach params, ensure,
/// stats, last-bytes counter, install_ssd_sinks) lives on `ArchPromptCache`.
/// Gemma4 uses [`ReusePolicy::Partial`] — block-aligned partial-prefix reuse
/// is correct under the SWA `can_truncate_to_block` gate; the B1 strict-prefix
/// path (`is_strict_prefix_of`) is also taken via the generate-loop match.
pub(crate) static PROMPT_CACHE: ArchPromptCache<Gemma4Entry> =
    ArchPromptCache::new("Gemma4ForConditionalGeneration", ReusePolicy::Partial);

/// the active `layout_key` for SSD tier seeding, or `0` when the tier
/// is OFF. See [`ArchPromptCache::active_layout_key`].
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global Gemma4 prompt cache is initialised with at least
/// `capacity` slots.
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Gemma4 prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

/// Read the KV-cache bytes from the last completed Gemma4 request.
pub fn read_kv_cache_bytes() -> u64 {
    PROMPT_CACHE.read_kv_cache_bytes()
}

/// Record the KV-cache byte total for the just-completed request. Called by
/// `gemma4/generate.rs` at request boundary.
pub(crate) fn store_kv_cache_bytes(n: u64) {
    PROMPT_CACHE.store_kv_cache_bytes(n);
}

// ---------------------------------------------------------------------------
// Tests — C1 partial-prefix trimmability gate + KvQuant mismatch eviction.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
