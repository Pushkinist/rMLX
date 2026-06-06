//! Stable hash helpers for the block manager.
//!
//! Block hashing uses FNV-1a-64 — same family as `prompt_cache.rs` — so SSD
//! `.kvb` rows persisted by earlier commits keep working. The layout key is mixed in
//! via `FNV_OFFSET ^ layout_key`.

use crate::prompt_cache::{FNV_OFFSET, FNV_PRIME};

/// FNV-1a-64 over `bytes`, seeded from `seed`. Plain stdlib walk.
#[inline]
pub(crate) fn fnv1a64_seeded(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// chained-block-hash digest for one block of tokens, folding in the
/// previous block's digest as the seed and the layout_key as XOR salt.
///
/// `prev` for the first block must be `FNV_OFFSET ^ layout_key`.
#[inline]
pub(crate) fn chained_block_digest(tokens: &[u32], prev: u64) -> u64 {
    let mut h = prev;
    for &id in tokens {
        for byte in id.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Convenience: derive the chained seed for the first block from a `layout_key`.
#[inline]
pub(crate) fn initial_seed(layout_key: u64) -> u64 {
    FNV_OFFSET ^ layout_key
}

#[cfg(test)]
#[path = "hash_tests.rs"]
mod tests;
