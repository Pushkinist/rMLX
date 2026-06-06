//! TinyLFU counting Count-Min Sketch with 4-bit saturating counters
//! and halving-decay aging.
//!
//! Port of `dynamo/lib/kvbm-logical/src/tinylfu.rs` adapted to rMLX's stdlib
//! plus the FNV-1a hash family. The reference uses `xxhash-rust` with 4 distinct
//! 192-byte secrets; that crate is not in rMLX's dependency set and adding
//! it requires user sign-off (CLAUDE.md "Ask before adding a new
//! dependency"). Instead we derive four independent stable hash streams from
//! four distinct FNV-1a-64 seeds. Same big-O, same algorithmic semantics
//! (4 independent CMS slots, halving decay, 4-bit ceiling), determinism
//! unchanged. The FNV family is already vetted for stability across the
//! codebase (see `prompt_cache::chained_block_hashes`, required for `.kvb`
//! SSD persistence).
//!
//! Layout (verified against reference):
//!
//! * 4 hash functions
//! * Table: `max(1, capacity / 4)` `u64`s; each `u64` packs 16 4-bit counters
//! * Increment saturates at 15
//! * `decay_threshold = capacity * 10` increments
//! * Decay: `*entry = (*entry >> 1) & 0x7777_7777_7777_7777`
//!
//! Single-threaded use is the intended pattern (KVBM holds the store mutex
//! around all sketch ops). The `Mutex` is wrapped externally by callers — the
//! sketch itself is plain `&mut self` for simplicity.

use std::sync::Mutex;

use super::hash::fnv1a64_seeded;

/// Four stable u64 seeds for the 4-bit CMS hash family. Distinct constants
/// chosen so the high bits diverge from the FNV-1a offset basis used elsewhere
/// in the codebase — keeps these slots independent of the prompt-cache chained
/// hashes.
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3, // pi (binary) — used as a stable nothing-up-my-sleeve constant
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

const RESET_MASK: u64 = 0x7777_7777_7777_7777;
const ONE_MASK: u64 = 0x1111_1111_1111_1111;

/// 4-bit counting Count-Min Sketch with halving decay.
///
/// `capacity` is the desired number of trackable entries; the table is sized
/// to `max(1, capacity / 4)` `u64`s (≈ 4 nibbles per entry).
#[derive(Debug)]
pub(crate) struct TinyLfuSketch {
    table: Vec<u64>,
    size: u64,
    decay_threshold: u64,
    capacity: usize,
}

impl TinyLfuSketch {
    pub(crate) fn new(capacity: usize) -> Self {
        let table_len = (capacity / 4).max(1);
        Self {
            table: vec![0u64; table_len],
            size: 0,
            decay_threshold: (capacity as u64).saturating_mul(10).max(1),
            capacity,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Increment counters for `key`. Saturates at 15 per slot. If any slot
    /// incremented, `size` is bumped and decay fires when `size >=
    /// decay_threshold`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn increment(&mut self, key: u64) {
        let mut any_inc = false;
        let table_len = self.table.len() as u64;
        for &seed in &SEEDS {
            let h = fnv1a64_seeded(&key.to_le_bytes(), seed);
            let table_index = (h % table_len) as usize;
            let counter_index = (h & 15) as u32;
            let shift = counter_index * 4;
            let entry = self.table[table_index];
            let cur = (entry >> shift) & 0xF;
            if cur < 15 {
                self.table[table_index] = (entry & !(0xF << shift)) | ((cur + 1) << shift);
                any_inc = true;
            }
        }
        if any_inc {
            self.size = self.size.saturating_add(1);
            if self.size >= self.decay_threshold {
                self.decay();
            }
        }
    }

    /// Lower-bound frequency estimate — minimum of the 4 slots.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn estimate(&self, key: u64) -> u8 {
        let mut min_val: u8 = 15;
        let table_len = self.table.len() as u64;
        for &seed in &SEEDS {
            let h = fnv1a64_seeded(&key.to_le_bytes(), seed);
            let table_index = (h % table_len) as usize;
            let counter_index = (h & 15) as u32;
            let shift = counter_index * 4;
            let cur = ((self.table[table_index] >> shift) & 0xF) as u8;
            if cur < min_val {
                min_val = cur;
            }
        }
        min_val
    }

    /// Halve every counter. Each odd-valued nibble (1, 3, 5, 7, 9, 11, 13,
    /// 15) loses its low bit on the right-shift; `count_ones(entry &
    /// ONE_MASK)` totals the increments lost across all such nibbles. Each
    /// `add` writes up to 4 nibbles (one per hash function), so dividing the
    /// lost-increments count by 4 gives the lost-adds correction to subtract
    /// from `size / 2`.
    fn decay(&mut self) {
        let mut zeroed = 0u64;
        for entry in &mut self.table {
            zeroed = zeroed.saturating_add(u64::from((*entry & ONE_MASK).count_ones()));
            *entry = (*entry >> 1) & RESET_MASK;
        }
        self.size = (self.size / 2).saturating_sub(zeroed / 4);
    }
}

/// Thread-safe wrapper. Block manager holds a single mutex around store ops,
/// but exposing the sketch as `Mutex<…>` lets us share it across the registry
/// and the MultiLRU backend without re-deriving the lock layer.
#[derive(Debug)]
pub(crate) struct TinyLfuTracker {
    inner: Mutex<TinyLfuSketch>,
}

impl TinyLfuTracker {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(TinyLfuSketch::new(capacity)),
        }
    }

    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn increment(&self, key: u64) {
        self.inner.lock().unwrap().increment(key);
    }

    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn estimate(&self, key: u64) -> u8 {
        self.inner.lock().unwrap().estimate(key)
    }
}

/// Default `MultiLruBackend` bin thresholds — matches the reference.
pub(crate) const MULTI_LRU_THRESHOLDS: [u8; 3] = [3, 8, 15];

/// Map a TinyLFU count to one of 4 priority bins (0 = coldest, 3 = hottest).
pub(crate) fn bin_for_count(count: u8) -> usize {
    if count < MULTI_LRU_THRESHOLDS[0] {
        0
    } else if count < MULTI_LRU_THRESHOLDS[1] {
        1
    } else if count < MULTI_LRU_THRESHOLDS[2] {
        2
    } else {
        3
    }
}

#[cfg(test)]
#[path = "tinylfu_tests.rs"]
mod tests;
