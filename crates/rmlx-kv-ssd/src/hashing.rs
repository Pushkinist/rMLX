//! Chained FNV-1a-64 block-digest helpers + the 256-token block constant.
//!
//! Migrated from `rmlx_models::prompt_cache` so the SSD modules (`spill`,
//! `hydrate`, `ssd_tier`) can use them without a back-edge into `rmlx-models`.
//! The constants and the seeded-FNV walk are byte-identical to the previous
//! definitions; in-crate `crate::prompt_cache::FNV_OFFSET` / `BLOCK_TOKENS` /
//! `chained_block_hashes_seeded` call sites in `rmlx-models` resolve via
//! `pub use rmlx_kv_ssd::*` re-exports in `prompt_cache.rs`.

/// Prefix-match block size, in tokens (oMLX-parity).
///
/// Prefix matching is block-aligned: only whole 256-token blocks are matched,
/// the trailing partial block (`len % BLOCK_TOKENS`) is never stored and is
/// re-prefilled every request. A match of >= 1 block is the worthwhile
/// threshold (256 tokens ≈ the break-even point vs full re-prefill).
pub const BLOCK_TOKENS: usize = 256;

/// FNV-1a-64 standard offset basis. Public so the salted seed math is
/// observable from tests and callers that need the un-salted default.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a-64 standard prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Chained FNV-1a-64 block digests over the full 256-token blocks of `ids`.
///
/// One digest per full block (trailing partial block excluded). Each block's
/// digest folds in the previous block's digest as its seed — so a single
/// digest compare validates the entire prefix up to and including that block
/// (oMLX `prefix_cache.py` chained-hash semantic).
///
/// FNV-1a-64 with the standard fixed offset basis / prime — deterministic and
/// stable across runs (required for SSD persistence; must NOT use the
/// randomly-seeded std `DefaultHasher`).
///
/// Equivalent to `chained_block_hashes_seeded(ids, FNV_OFFSET)`.
pub fn chained_block_hashes(ids: &[u32]) -> Vec<u64> {
    chained_block_hashes_seeded(ids, FNV_OFFSET)
}

/// Explicit-seed variant of [`chained_block_hashes`].
///
/// The chained FNV walk starts from `seed` instead of the hard-coded
/// [`FNV_OFFSET`]; everything else is byte-identical to the un-seeded form.
/// Calling `chained_block_hashes_seeded(ids, FNV_OFFSET)` therefore returns
/// the exact same digests as `chained_block_hashes(ids)`, by construction.
///
/// ## Mixing function
///
/// The layout-key salt lives at the call site, NOT inside this function:
///
/// ```text
/// seed = FNV_OFFSET ^ layout_key
/// digests = chained_block_hashes_seeded(ids, seed)
/// ```
///
/// where `layout_key` is a stable u64 hash over
/// `(arch, n_layers, n_kv_heads, head_dim, kv_quant)` (see
/// [`crate::ssd_tier::compute_layout_key`]). XOR is the documented mixing
/// function — simpler than re-keying the FNV prime and keeps the existing
/// avalanche behaviour intact. When the caller passes the bare `FNV_OFFSET`
/// (layout-key salt absent), digests collapse to the legacy un-salted stream.
/// Different layouts produce disjoint chained-digest streams for the
/// same `ids`, so a prompt cached at one KV layout cannot accidentally collide
/// with the same prompt cached at another layout. Pairs with the
/// `(hash, layout_key)` composite PK on the `kv_blocks` SSD index for
/// defence-in-depth.
#[allow(
    clippy::indexing_slicing,
    reason = "loop bounds: b < n_blocks where n_blocks = ids.len() / BLOCK_TOKENS, \
              so b * BLOCK_TOKENS and (b+1) * BLOCK_TOKENS are both ≤ ids.len()"
)]
pub fn chained_block_hashes_seeded(ids: &[u32], seed: u64) -> Vec<u64> {
    let n_blocks = ids.len() / BLOCK_TOKENS;
    let mut digests = Vec::with_capacity(n_blocks);
    let mut prev = seed;
    for b in 0..n_blocks {
        let mut h = prev;
        for &id in &ids[b * BLOCK_TOKENS..(b + 1) * BLOCK_TOKENS] {
            for byte in id.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        digests.push(h);
        prev = h;
    }
    digests
}
