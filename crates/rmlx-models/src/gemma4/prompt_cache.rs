//! Gemma4 prompt-cache entry + global.
//!
//! The heavy lifting (static cache, SSD attach/install, ensure, stats
//! readback) is unified in
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
//!   `gemma4/generate.rs`'s `consume()` dispatch (/ B1 snapshot/restore).
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
use rmlx_core::DispatchPolicy;

use crate::prompt_cache::{
    ArchPromptCache, CacheStats, PromptCacheEntry, ReuseKind, ReusePolicy, SsdHydrate, BLOCK_TOKENS,
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
    /// Gemma4 has SWA layers backed by a `RotatingKvCache`. A ring that has
    /// wrapped can only be rolled back while it still holds the window the
    /// shorter prefix attends over, and a cached prompt long enough to wrap it
    /// never does over a whole block. `truncate_kv_to_block` would then fail on
    /// the SWA layers after truncating the full-attention ones, desyncing the
    /// caches (this was the original broken gemma4 Prefix path: longctx
    /// 22 -> 0 tokens). When this returns `false` the caller MUST fall back to
    /// a full re-prefill (Miss).
    pub(crate) fn can_truncate_to_block(&self, block_count: usize) -> bool {
        debug_assert!(
            block_count >= 1,
            "block_count must be >= 1 for a prefix hit"
        );
        let target = (block_count * BLOCK_TOKENS) as i32;
        // Same population `truncate_kv_to` acts on: a never-filled layer — the
        // KV-shared tail, whose offset stays 0 — is skipped by the truncation
        // and must not veto it either.
        self.kv_caches
            .iter()
            .filter(|c| c.offset() > 0)
            .all(|c| c.can_truncate_to(target))
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
    /// and corrupting the output. `KvCache::can_truncate_to()` returns `true`
    /// for a `None` layer, so it is the wrong predicate to detect this.
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

    /// Delegates to the inherent [`Gemma4Entry::is_hydrate_complete`]: an
    /// SSD-hydrated entry whose SWA rotating ring came back as a payload-less
    /// `KvStorage::None` (the ring is not serialised to the SSD tier) is
    /// incomplete, so the consume engine excludes it from prefix reuse and
    /// degrades it to a full re-prefill.
    fn is_hydrate_complete(&self) -> bool {
        Gemma4Entry::is_hydrate_complete(self)
    }

    /// B1 (strict-prefix SWA snapshot/restore) is checked FIRST; only when it
    /// declines does the block-truncate path run. This precedence preserves the
    /// live arm order: the strict-prefix snapshot supersedes block-truncate for
    /// the wrapped-SWA multi-turn case (where truncation would desync the SWA
    /// ring).
    ///
    /// - Strict prefix: the cached prompt is a token-for-token prefix that the
    ///   request extends by `>= 1` token (`is_strict_prefix_of`, with the
    ///   `cached_len >= BLOCK_TOKENS` worthwhile floor). Reuse the snapshot
    ///   verbatim at `prefix_len == cached_len` (no truncation → no SWA-wrap
    ///   desync). Works whether the entry is RAM-resident or SSD-hydrated.
    /// - Block truncate: otherwise, derive `effective_blocks` by dropping the
    ///   trailing block when the matched blocks would cover the whole
    ///   block-aligned prompt (the re-prefilled tail must be non-empty). When
    ///   `effective_blocks >= 1` and every layer cache is losslessly trimmable
    ///   (`can_truncate_to_block` — false once an SWA ring has wrapped), reuse
    ///   the first `effective_blocks` blocks; otherwise decline.
    fn is_reusable_prefix_of(
        &self,
        prompt_ids: &[u32],
        _is_ssd_hydrated: bool,
        matched_blocks: usize,
    ) -> Option<ReuseKind> {
        // B1 precedence — strict prefix supersedes block-truncate.
        if self.is_strict_prefix_of(prompt_ids) {
            return Some(ReuseKind::StrictPrefix {
                prefix_len: self.prompt_token_ids.len(),
            });
        }
        // Block-truncate path. The tail `[prefix_len..prompt_len)` must be
        // non-empty (its final position is where the next-token logits are
        // read). When the matched blocks already cover the whole block-aligned
        // prompt, drop the last matched block back into the re-prefill tail; a
        // genuine partial divergence (`matched * BLOCK < len`) reuses ALL
        // matched blocks since the tail past the last block boundary is already
        // non-empty.
        let prompt_blocks = prompt_ids.len() / BLOCK_TOKENS;
        let effective_blocks = if matched_blocks * BLOCK_TOKENS >= prompt_ids.len() {
            matched_blocks.min(prompt_blocks).saturating_sub(1)
        } else {
            matched_blocks
        };
        if effective_blocks >= 1 && self.can_truncate_to_block(effective_blocks) {
            Some(ReuseKind::BlockTruncate { effective_blocks })
        } else {
            None
        }
    }

    /// `StrictPrefix` reuses the whole cached prefix verbatim → plain
    /// `deep_clone` (trait default semantics). `BlockTruncate` deep-clones then
    /// trims the KV to the block boundary, asserting the post-trim
    /// `offset == effective_blocks * BLOCK_TOKENS` invariant the
    /// generate-loop tail-forward relies on.
    fn prepare_reuse(&self, kind: ReuseKind) -> Result<Self> {
        match kind {
            ReuseKind::StrictPrefix { .. } => self.deep_clone(),
            ReuseKind::BlockTruncate { effective_blocks } => {
                // The caller (generate-loop tail-forward) uses
                // `prefix_len = effective_blocks * BLOCK_TOKENS` as the absolute
                // position the tail re-prefills from. The block-truncate path is
                // only constructed when `can_truncate_to_block(effective_blocks)`
                // held (every layer cache losslessly trimmable — no wrapped-SWA
                // ring), so re-asserting the precondition here is redundant.
                // Assert the spec POSTcondition instead: after the trim, every
                // KV cache that holds data lands exactly on the block boundary
                // `effective_blocks * BLOCK_TOKENS`. Never-filled layers
                // (`offset() == 0`) are skipped by `truncate_kv_to`, so they are
                // excluded from the check.
                let mut cloned = self.deep_clone()?;
                cloned.truncate_kv_to_block(effective_blocks)?;
                debug_assert!(
                    {
                        let expected = (effective_blocks * BLOCK_TOKENS) as i32;
                        cloned
                            .kv_caches
                            .iter()
                            .filter(|c| c.offset() > 0)
                            .all(|c| c.offset() == expected)
                    },
                    "block-truncate postcondition: every filled KV cache offset \
                     must equal effective_blocks * BLOCK_TOKENS after the trim"
                );
                Ok(cloned)
            }
        }
    }
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
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> Result<Option<Gemma4Entry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(
            prompt_ids,
            seed,
            kv_quant,
            policy,
            crate::gemma4::SHARES_KV_ACROSS_LAYERS,
        )?
        else {
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

/// per-arch shell. All boilerplate (static lock, attach params, ensure,
/// stats, install_ssd_sinks) lives on `ArchPromptCache`.
/// Gemma4 uses [`ReusePolicy::Partial`] — block-aligned partial-prefix reuse
/// is correct under the SWA `can_truncate_to_block` gate; the B1 strict-prefix
/// path (`is_strict_prefix_of`) is also taken via the generate-loop match.
pub(crate) static PROMPT_CACHE: ArchPromptCache<Gemma4Entry> = ArchPromptCache::new(
    "Gemma4ForConditionalGeneration",
    ReusePolicy::Partial,
    crate::gemma4::SHARES_KV_ACROSS_LAYERS,
);

/// the active `layout_key` for SSD tier seeding, or `0` when the tier
/// is OFF. See [`ArchPromptCache::active_layout_key`].
pub(crate) fn active_layout_key() -> u64 {
    PROMPT_CACHE.active_layout_key()
}

/// Ensure the global Gemma4 prompt cache is initialised with exactly `capacity`
/// slots; `0` disables it (nothing is stored, every request prefills).
pub(crate) fn ensure_prompt_cache(capacity: usize) {
    PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Gemma4 prompt cache.
pub fn read_cache_stats() -> Option<CacheStats> {
    PROMPT_CACHE.read_cache_stats()
}

// ---------------------------------------------------------------------------
// Tests — C1 partial-prefix trimmability gate + KvQuant mismatch eviction.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
