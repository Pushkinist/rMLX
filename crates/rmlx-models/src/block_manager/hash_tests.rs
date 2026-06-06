use super::*;
use crate::prompt_cache::chained_block_hashes_seeded;

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn chained_digest_matches_prompt_cache_walk() {
    // Build a single 256-token block.
    let toks: Vec<u32> = (0u32..256u32).collect();
    let layout_key = 0xa5a5_a5a5_5a5a_5a5au64;
    let seed = initial_seed(layout_key);
    let walk = chained_block_digest(&toks, seed);
    let reference = chained_block_hashes_seeded(&toks, seed);
    assert_eq!(reference.len(), 1);
    assert_eq!(walk, reference[0]);
}

#[test]
fn fnv1a_changes_with_seed() {
    let h1 = fnv1a64_seeded(b"abc", FNV_OFFSET);
    let h2 = fnv1a64_seeded(b"abc", FNV_OFFSET ^ 1);
    assert_ne!(h1, h2);
}
