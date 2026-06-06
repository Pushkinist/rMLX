//! BlockManager facade.
//!
//! Reference: `dynamo/lib/kvbm-logical/src/manager/mod.rs`.
//!
//! Public API: `allocate_blocks` / `register_blocks` / `match_blocks` /
//! `scan_matches`. The facade owns a [`BlockStore`] and the shared event
//! manager; per-arch glue constructs one BlockManager per loaded
//! model.

#![allow(clippy::missing_fields_in_debug)]
use std::sync::Arc;

use super::events::EventManager;
use super::hash::{chained_block_digest, initial_seed};
use super::overflow::OverflowSink;
use super::store::{
    BlockStore, ImmutableBlock, MatchOutcome, MutableBlock, StoreError, StoreStats,
};
use super::{BlockHash, BlockMetadata};

/// Per-instance configuration.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    /// Number of slots in the store. Today this is dimensioned by the caller
    /// based on RAM budget / page size.
    pub capacity: usize,
    /// Frequency-sketch capacity for the TinyLFU tracker. Default `2^21`
    /// matches the reference `Medium` preset.
    pub tinylfu_capacity: usize,
    /// Block size in tokens (default 256, matching today's `BLOCK_TOKENS`).
    /// Configurable here per the ticket — quantiser group sizes (32 for
    /// `tq4`/`planar4`) may want a smaller block.
    pub block_tokens: usize,
    /// layout key — mixed into the chained block hash seed so caches
    /// keyed under different KV layouts do not collide.
    pub layout_key: u64,
    /// Optional LoRA salt — mixed into the layout key. `None` ⇒ no LoRA. Per
    /// `dynamo/lib/kv-router/src/protocols.rs`, this scopes the cache to a
    /// specific LoRA adapter.
    pub lora_salt: Option<u64>,
    /// Optional multimodal `mm_hash` salt — same rationale: mixing it into
    /// the seed scopes the cache to images/audio paired with the prompt.
    pub mm_hash: Option<u64>,
}

impl Default for BlockManagerConfig {
    /// Default sizing: scales the TinyLFU sketch to the store capacity so a
    /// small store does not over-allocate the frequency table. The reference
    /// `Medium` preset (`1 << 21` counters ≈ 4 MiB) is exposed via
    /// [`BlockManagerConfig::medium`] for callers that want it explicitly.
    fn default() -> Self {
        let capacity = 1024usize;
        Self {
            capacity,
            // ~8 counters per slot, floor of 64. For capacity = 1024 this
            // gives 8192 counters → 2048 u64 entries → ~16 KiB.
            tinylfu_capacity: capacity.saturating_mul(8).max(64),
            block_tokens: 256,
            layout_key: 0,
            lora_salt: None,
            mm_hash: None,
        }
    }
}

impl BlockManagerConfig {
    /// Reference-equivalent "Medium" sizing: 2^21 counters (~4 MiB sketch).
    /// Use this preset when you want the reference's eviction-quality
    /// guarantees on workloads with a wide hash space.
    pub fn medium() -> Self {
        Self {
            tinylfu_capacity: 1 << 21,
            ..Self::default()
        }
    }

    /// Final mixed seed for the chained block hash walk.
    ///
    /// Mixing chain: `layout_key → ⊕ lora → rotl(13) → × FNV_PRIME → ⊕ mm →
    /// rotl(29)`. The FNV multiply (`0x100000001b3`) is non-commutative with
    /// XOR, so every input contributes regardless of order; this avoids the
    /// theoretical collision in a pure-XOR mix where two inputs at the same
    /// position cancel.
    pub fn chained_seed(&self) -> u64 {
        // FNV-1a 64-bit prime, used here as a non-commutative mixing step.
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut s = self.layout_key;
        if let Some(l) = self.lora_salt {
            s = (s ^ l).rotate_left(13).wrapping_mul(FNV_PRIME);
        }
        if let Some(m) = self.mm_hash {
            s = (s.wrapping_mul(FNV_PRIME) ^ m).rotate_left(29);
        }
        initial_seed(s)
    }
}

/// One match result row from [`BlockManager::scan_matches`] — outcome plus
/// the (sub-)prefix length in tokens that matched.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed result struct — two fields are the complete prefix-match contract; adding a field requires updating scan_matches and all callers"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Number of full blocks that matched the query prefix.
    pub n_blocks: usize,
    /// Number of tokens that matched (may be `< block_tokens` for the last partial block).
    pub n_tokens: usize,
}

/// Facade. Cheap to clone (`Arc`-backed).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed manager — fields are private; public API is the BlockManager methods, not struct literal construction"
)]
pub struct BlockManager<T: BlockMetadata> {
    store: BlockStore<T>,
    events: Arc<EventManager>,
    config: BlockManagerConfig,
}

impl<T: BlockMetadata> Clone for BlockManager<T> {
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            events: self.events.clone(),
            config: self.config.clone(),
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for BlockManager<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockManager")
            .field("config", &self.config)
            .finish()
    }
}

impl<T: BlockMetadata> BlockManager<T> {
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    /// Create a new `BlockManager` with the given configuration.
    pub fn new(config: BlockManagerConfig) -> Self {
        let events = Arc::new(EventManager::new());
        let store = BlockStore::new(config.capacity, config.tinylfu_capacity, events.clone());
        Self {
            store,
            events,
            config,
        }
    }

    /// Return a reference to the manager's configuration.
    pub fn config(&self) -> &BlockManagerConfig {
        &self.config
    }

    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    /// Return a clone of the shared event manager.
    pub fn events(&self) -> Arc<EventManager> {
        self.events.clone()
    }

    /// Return a reference to the underlying block store.
    pub fn store(&self) -> &BlockStore<T> {
        &self.store
    }

    /// Bump the TinyLFU frequency for `hash` without altering refcount.
    /// Tests use this to plant explicit bin assignments; callers in the
    /// integration layer use it on hydrate-from-SSD.
    ///
    /// If `hash` is currently sitting in the Inactive set, the underlying
    /// [`BlockStore::touch_frequency`] also re-bins the entry under the new
    /// count (otherwise it would stay in its old tier until the next
    /// `release_ref` or explicit touch).
    pub fn touch_frequency(&self, hash: BlockHash) {
        self.store.touch_frequency(hash);
    }

    /// Return a snapshot of store statistics.
    pub fn stats(&self) -> StoreStats {
        self.store.stats()
    }

    /// Install an overflow sink for blocks evicted beyond capacity.
    pub fn set_overflow_sink(&self, sink: Arc<dyn OverflowSink<T>>) {
        self.store.set_overflow_sink(sink);
    }

    /// Compute chained block-hash digests for `tokens` under this manager's
    /// configured seed. Trailing partial block (len % block_tokens) is
    /// dropped. Mirrors `prompt_cache::chained_block_hashes_seeded`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn chained_block_hashes(&self, tokens: &[u32]) -> Vec<BlockHash> {
        let block = self.config.block_tokens;
        let n_blocks = tokens.len() / block;
        let mut out = Vec::with_capacity(n_blocks);
        let mut prev = self.config.chained_seed();
        for b in 0..n_blocks {
            let d = chained_block_digest(&tokens[b * block..(b + 1) * block], prev);
            out.push(d);
            prev = d;
        }
        out
    }

    /// Allocate `n` mutable blocks.
    pub fn allocate_blocks(&self, n: usize) -> Result<Vec<MutableBlock<T>>, StoreError> {
        self.store.allocate_blocks(n)
    }

    /// Register a single mutable block.
    pub fn register_block(
        &self,
        mutable: MutableBlock<T>,
        hash: BlockHash,
        payload: T,
    ) -> ImmutableBlock<T> {
        mutable.register(&self.store, hash, payload)
    }

    /// Register a batch. Caller supplies one (mutable, hash, payload) per
    /// block. Returns one `ImmutableBlock` per request, in input order.
    pub fn register_blocks(
        &self,
        items: Vec<(MutableBlock<T>, BlockHash, T)>,
    ) -> Vec<ImmutableBlock<T>> {
        items
            .into_iter()
            .map(|(m, h, p)| m.register(&self.store, h, p))
            .collect()
    }

    /// Batched hash → outcome lookup. Each hit carries an `ImmutableBlock`
    /// the caller must consume or drop. The whole batch runs in a single
    /// store-lock critical section.
    pub fn match_blocks(&self, hashes: &[BlockHash]) -> Vec<MatchOutcome<T>> {
        self.store.match_blocks(hashes)
    }

    /// Longest-prefix scan. Returns the longest contiguous run of hits at
    /// the head of `hashes` — equivalent to today's `find_best_prefix`.
    ///
    /// Implementation: a probe-then-confirm split. The probe pass is
    /// state-free (no refcount bumps, no resurrections); we find the prefix
    /// length, then confirm only the accepted prefix. Hits past the first
    /// miss are NOT touched — their tier and refcount stay intact.
    pub fn scan_matches(&self, hashes: &[BlockHash]) -> PrefixMatch {
        let probes = self.store.probe_blocks(hashes);
        // Find first miss.
        let mut n = probes.len();
        for (i, p) in probes.iter().enumerate() {
            if p.is_none() {
                n = i;
                break;
            }
        }
        // Confirm exactly `n` entries — caller does not receive the returned
        // `ImmutableBlock`s here; in production callers will use
        // `match_blocks` directly to take ownership. `scan_matches` is the
        // length-only API.
        let _confirmed = self.store.confirm_prefix(hashes, n);
        PrefixMatch {
            n_blocks: n,
            n_tokens: n * self.config.block_tokens,
        }
    }
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
