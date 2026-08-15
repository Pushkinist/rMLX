//! Stable hash helpers for the block manager.
//!
//! Block hashing uses FNV-1a-64, the same family as the prompt cache, and for
//! the same reasons: deterministic across runs, cheap, no new dependency.
//!
//! It is **not** the prompt cache's `cache_seed` and does not produce the same
//! digest stream. This module keys a different store with a different identity —
//! `CacheKey::chained_seed` mixes `layout_key` with an optional `lora_salt` and
//! `mm_hash` through a rotate/multiply chain, terms `cache_seed` does not have,
//! and it has no model or codec term. So these digests cannot address `.kvb`
//! rows and are not interchangeable with them. The block manager is not wired
//! into `serve` today; when it is, it needs its own store, not the SSD tier's.

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
