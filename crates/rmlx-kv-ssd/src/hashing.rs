//! Chained FNV-1a-64 block-digest helpers, the 256-token block constant, and
//! the one prompt-cache seed definition shared by RAM and disk.
//!
//! Migrated from `rmlx_models::prompt_cache` so the SSD modules (`spill`,
//! `hydrate`, `ssd_tier`) can use them without a back-edge into `rmlx-models`.
//! The constants and the seeded-FNV walk are byte-identical to the previous
//! definitions; in-crate `crate::prompt_cache::FNV_OFFSET` / `BLOCK_TOKENS` /
//! `chained_block_hashes_seeded` call sites in `rmlx-models` resolve via
//! `pub use rmlx_kv_ssd::*` re-exports in `prompt_cache.rs`.
//!
//! [`cache_seed`] lives here, below both consumers, for the same reason: the
//! RAM prompt cache (in `rmlx-models`) and the SSD hydrate probe (in this
//! crate) have to seed the same digest stream, and this is the deepest crate
//! both can call.

use rmlx_kv_quant::KvQuant;

#[cfg(test)]
#[path = "hashing_tests.rs"]
mod hashing_tests;

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

/// The block-digest seed that a prompt-cache lookup, its matching push, the
/// SSD spill row and the SSD hydrate probe must all use.
///
/// Three things partition the key, and every one of them is a case where reusing
/// another entry's K/V would be wrong rather than merely unhelpful:
///
/// - `model_sig` — **which model produced this K/V.** The prompt cache is one
///   static per architecture, so two models of the same arch resident at once
///   (the multi-model registry, or a speculative pair) share it. Without this
///   term, model B's identical prompt matches model A's slot, the token-id
///   equality check passes because the tokens *are* equal, and B decodes from
///   A's K/V and A's first token — wrong output, silently. The SSD tier needs
///   it for the same reason and one more: `--project` collapses every loaded
///   model onto one namespace, so the on-disk directory is not a per-model
///   partition either.
/// - `layout_key` — the SSD tier's attach-time shape key, or `0` when the tier
///   is OFF. It is a *shape* identity and carries no model identity, which is
///   why `model_sig` is a separate term rather than something to fold into it.
/// - `kv_quant` — the codec the stored K/V is packed under.
/// - `layer_quants` — the per-layer codec vector **that codec resolves to
///   under the caller's current layer policy**, one entry per decoder layer.
///
/// The last term is why this takes a vector rather than the codec alone. Which
/// codec each layer gets is a policy decision, not a property of the requested
/// codec: the boundary-layer promotion rewrites some entries, and it can change
/// between builds. `layout_key` folds a vector too, but it is fixed at attach
/// from the *launch* codec, and a request may resolve a different one
/// (auto-by-context, or a per-request override) — for those requests the
/// attach-time vector describes a layout they are not running. The seed is
/// where the request's own codec already enters, so it is where the request's
/// own mixture belongs: fold it here and a policy change invalidates every
/// stored block for every codec, not only for requests that happened to run the
/// launch default.
///
/// One function, in the crate below every caller, so no side can compute the
/// seed on its own: a push seeded differently from the query is not a bug that
/// surfaces as a wrong answer, it surfaces as a cache that silently never hits.
/// `rmlx-models` wraps it as `prompt_cache::request_cache_seed` to supply
/// `layer_quants` from the single producer, because the policy lives up there.
pub fn cache_seed(
    layout_key: u64,
    kv_quant: KvQuant,
    layer_quants: &[KvQuant],
    model_sig: u64,
) -> u64 {
    let mut h = FNV_OFFSET ^ layout_key ^ kv_quant.cache_key_salt() ^ model_sig;
    for q in layer_quants {
        h ^= q.cache_key_salt();
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

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
/// The salt lives at the call site, NOT inside this function:
///
/// ```text
/// seed    = cache_seed(layout_key, kv_quant, model_sig)
/// digests = chained_block_hashes_seeded(ids, seed)
/// ```
///
/// Every production caller — RAM push, RAM query, SSD spill key, SSD hydrate
/// probe — builds that seed with [`cache_seed`] and nothing else. XOR is the
/// documented mixing function — simpler than re-keying the FNV prime and keeps
/// the existing avalanche behaviour intact. When the caller passes the bare
/// `FNV_OFFSET` (no salt at all), digests collapse to the legacy un-salted
/// stream, which is what the tests and the un-seeded wrapper use. Different
/// models / layouts / codecs produce disjoint chained-digest streams for the
/// same `ids`, so a prompt cached under one of them cannot collide with the
/// same prompt cached under another. Pairs with the `(hash, layout_key)`
/// composite PK on the `kv_blocks` SSD index for defence-in-depth.
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
