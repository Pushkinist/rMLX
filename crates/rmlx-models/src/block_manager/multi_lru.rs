//! 4-tier LRU backend for the inactive pool, keyed by TinyLFU
//! frequency.
//!
//! Reference: `dynamo/lib/kvbm-logical/src/pools/inactive/backends/multi_lru_backend.rs`.
//! Default bin thresholds `[3, 8, 15]` map TinyLFU counts to 4 pools (0 =
//! coldest, 3 = hottest). Eviction drains pool 0 first, then 1, 2, 3. Match
//! lookups walk all 4 pools in order.
//!
//! Implementation uses plain `VecDeque<BlockHash>` per pool — single-threaded
//! by the surrounding store mutex, so a full `lru` crate is unnecessary. The
//! `VecDeque` is FIFO; `touch` moves a block to the back (most-recently used).

use std::collections::VecDeque;
use std::sync::Arc;

use super::tinylfu::{bin_for_count, TinyLfuTracker};
use super::BlockHash;

/// The trait the `BlockStore` calls to evict / match / move blocks in the
/// inactive set. Implementations: [`MultiLruBackend`].
///
/// `touch` and `contains` are part of the surface the radix-tree registry
/// (follow-up) will use; they are exercised by unit tests today.
#[allow(dead_code)]
pub(crate) trait InactiveIndex: Send + Sync + std::fmt::Debug {
    /// Insert a block. Bin is decided from the current TinyLFU count.
    fn insert(&mut self, hash: BlockHash);

    /// Remove a block (resurrection back to active). Returns true if present.
    fn remove(&mut self, hash: BlockHash) -> bool;

    /// Touch a block on lookup hit: re-bin under current count, move to the
    /// back of its pool.
    fn touch(&mut self, hash: BlockHash);

    /// Pop the coldest block (pool 0 first). Returns `None` when empty.
    fn evict(&mut self) -> Option<BlockHash>;

    /// True if `hash` is currently in the inactive set.
    fn contains(&self, hash: BlockHash) -> bool;

    /// Number of inactive blocks (sum across all pools).
    fn len(&self) -> usize;

    /// Per-tier counts (debugging / tests).
    fn tier_lens(&self) -> [usize; 4];
}

/// 4-tier LRU keyed by TinyLFU bin.
#[derive(Debug)]
pub(crate) struct MultiLruBackend {
    pools: [VecDeque<BlockHash>; 4],
    tracker: Arc<TinyLfuTracker>,
}

impl MultiLruBackend {
    pub(crate) fn new(tracker: Arc<TinyLfuTracker>) -> Self {
        Self {
            pools: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            tracker,
        }
    }

    /// Bin a hash by the *current* TinyLFU estimate.
    fn bin_of(&self, hash: BlockHash) -> usize {
        bin_for_count(self.tracker.estimate(hash))
    }

    fn remove_from_any(&mut self, hash: BlockHash) -> bool {
        for pool in &mut self.pools {
            if let Some(idx) = pool.iter().position(|h| *h == hash) {
                pool.remove(idx);
                return true;
            }
        }
        false
    }
}

impl InactiveIndex for MultiLruBackend {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn insert(&mut self, hash: BlockHash) {
        // De-dup defensively — caller may not have removed first.
        self.remove_from_any(hash);
        let bin = self.bin_of(hash);
        self.pools[bin].push_back(hash);
    }

    fn remove(&mut self, hash: BlockHash) -> bool {
        self.remove_from_any(hash)
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn touch(&mut self, hash: BlockHash) {
        if !self.remove_from_any(hash) {
            return;
        }
        let bin = self.bin_of(hash);
        self.pools[bin].push_back(hash);
    }

    fn evict(&mut self) -> Option<BlockHash> {
        for pool in &mut self.pools {
            if let Some(h) = pool.pop_front() {
                return Some(h);
            }
        }
        None
    }

    fn contains(&self, hash: BlockHash) -> bool {
        self.pools.iter().any(|p| p.iter().any(|h| *h == hash))
    }

    fn len(&self) -> usize {
        self.pools.iter().map(VecDeque::len).sum()
    }

    fn tier_lens(&self) -> [usize; 4] {
        [
            self.pools[0].len(),
            self.pools[1].len(),
            self.pools[2].len(),
            self.pools[3].len(),
        ]
    }
}

#[cfg(test)]
#[path = "multi_lru_tests.rs"]
mod tests;
