use super::*;

/// Deterministic LCG so tests are reproducible without an external crate.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_u32_in(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() as u32) % (hi - lo + 1)
    }
    fn next_usize_in(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

// Synthesize a chained-hash vector of `n_blocks` entries seeded by `s`.
// Re-uses simple FNV-1a-64 byte mix so the same `s` always yields the
// same sequence; not bit-identical to `chained_block_hashes` but the
// PrefixIndex contract doesn't care about that — only that the bytes
// compare equal/unequal correctly.
fn synth_chained(seed: u64, n_blocks: usize) -> Vec<u64> {
    let mut prev = seed;
    let mut out = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let mut h = prev;
        for byte in i.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        out.push(h);
        prev = h;
    }
    out
}

// --- Unit tests: RadixTree::match_best primitives ---

#[test]
fn radix_empty_returns_none() {
    let tree = RadixTree::new();
    let q = synth_chained(1, 3);
    assert!(tree.match_best(&q, 0).is_none());
    assert!(tree.is_empty());
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn radix_single_entry_exact_match() {
    let mut tree = RadixTree::new();
    let key = synth_chained(0xAA, 4);
    tree.insert(&key, 7, 42);
    let (slot, n) = tree.match_best(&key, 7).expect("exact match");
    assert_eq!(slot, 42);
    assert_eq!(n, 4);
    // Wrong layout_key → no match.
    assert!(tree.match_best(&key, 8).is_none());
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn radix_partial_block_prefix() {
    let mut tree = RadixTree::new();
    let key = synth_chained(0xBB, 6);
    tree.insert(&key, 0, 99);
    // Query with same first 3 blocks then diverge.
    let mut q = key[..3].to_vec();
    let mut alt = synth_chained(0xCC, 3);
    q.append(&mut alt);
    let (slot, n) = tree.match_best(&q, 0).expect("partial prefix hits");
    assert_eq!(slot, 99);
    assert_eq!(n, 3);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn radix_two_siblings_share_first_two_blocks() {
    let mut tree = RadixTree::new();
    let mut a = synth_chained(0xD0, 2);
    let mut b = a.clone();
    a.extend(synth_chained(0xD1, 3));
    b.extend(synth_chained(0xD2, 3));
    tree.insert(&a, 0, 1);
    tree.insert(&b, 0, 2);
    assert_eq!(tree.len(), 2);

    // Query exactly equals a → deepest match wins (5 blocks, slot 1).
    let (slot, n) = tree.match_best(&a, 0).expect("a hits");
    assert_eq!((slot, n), (1, 5));
    // Query equals b → slot 2 wins.
    let (slot, n) = tree.match_best(&b, 0).expect("b hits");
    assert_eq!((slot, n), (2, 5));
    // Query is the shared 2-block prefix only + a divergent third block
    // not in the tree → both paths share 2 blocks; the third block
    // misses → best = 2 blocks. Slot id is whichever was first traversed,
    // but our test only cares about the depth.
    let mut shared = a[..2].to_vec();
    shared.extend(synth_chained(0xD3, 1));
    let (_slot, n) = tree
        .match_best(&shared, 0)
        .expect("shared prefix is 2 blocks");
    assert_eq!(n, 2);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn radix_eviction_then_reinsert_no_orphans() {
    let mut tree = RadixTree::new();
    let key = synth_chained(0xE0, 5);
    tree.insert(&key, 1, 100);
    assert_eq!(tree.len(), 1);
    tree.remove(&key, 1);
    assert_eq!(tree.len(), 0);
    // After full eviction the lookup must miss.
    assert!(tree.match_best(&key, 1).is_none());
    // Reinsert + lookup again — no stale payload from the first insert.
    tree.insert(&key, 1, 200);
    let (slot, n) = tree.match_best(&key, 1).expect("reinsert hits");
    assert_eq!((slot, n), (200, 5));
    // Reachable root walk has exactly one path of length 5 (pruning + re-creation
    // reuses orphan nodes — node-vec length may grow, but the *reachable*
    // chain stays the same shape).
    let mut cursor = RadixTree::ROOT;
    let mut depth = 0;
    while !tree.nodes[cursor as usize].children.is_empty() {
        // Each node should have exactly one child since only one entry is live.
        assert_eq!(
            tree.nodes[cursor as usize].children.len(),
            1,
            "only one live entry → fanout=1 at every reachable node"
        );
        let c = tree.nodes[cursor as usize].children[0];
        cursor = c;
        depth += 1;
    }
    assert_eq!(depth, 5);
}

// --- Unit tests: layout_key partitioning ---

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn radix_layout_key_partitions_keys() {
    let mut tree = RadixTree::new();
    let key = synth_chained(0xF0, 3);
    tree.insert(&key, 1, 10);
    tree.insert(&key, 2, 20);
    assert_eq!(tree.len(), 2);
    let (s1, _) = tree.match_best(&key, 1).expect("layout=1 hits");
    let (s2, _) = tree.match_best(&key, 2).expect("layout=2 hits");
    assert_eq!(s1, 10);
    assert_eq!(s2, 20);
    // No bleed across layouts.
    assert_eq!(tree.match_best(&key, 99), None);
}

#[test]
fn linear_layout_key_partitions_keys() {
    let mut idx = LinearScan::new();
    let key = synth_chained(0xF0, 3);
    idx.insert(&key, 1, 10);
    idx.insert(&key, 2, 20);
    assert_eq!(idx.len(), 2);
    assert_eq!(idx.match_best(&key, 1), Some((10, 3)));
    assert_eq!(idx.match_best(&key, 2), Some((20, 3)));
    assert_eq!(idx.match_best(&key, 99), None);
}

// --- Differential test: LOAD-BEARING ---

/// 1000 random prompts against a populated cache; LinearScan and
/// RadixTree must agree on Some/None, on `n_matched_blocks`, and (when
/// the matched prefix is unambiguous — only one entry shares it at the
/// matched depth) on `slot_id` too.
///
/// review MEDIUM-1: the original version asserted on
/// `lin.map(|(_, n)| n) == rdx.map(|(_, n)| n)` only — a slot-id divergence
/// at a unique prefix would have silently passed. The strengthened
/// assertions catch that class of bug. Insert-count parity (insert
/// returning different `len()` post-write across impls) is also pinned.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn differential_linear_vs_radix_1000_prompts() {
    let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
    let n_entries = 32usize;
    let max_blocks = 8usize;
    let layout_keys = [0u64, 0xCAFE_F00D, 0x9999_DEAD_BEEF_1111];

    let mut linear = LinearScan::new();
    let mut radix = RadixTree::new();

    // Populate: 32 entries with varying lengths + layouts.
    let mut entries: Vec<(Vec<u64>, u64, u64)> = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let n_blocks = rng.next_usize_in(1, max_blocks);
        let lk = layout_keys[rng.next_usize_in(0, layout_keys.len() - 1)];
        // Bias: half the entries share a common 2-block prefix to
        // exercise sibling branching.
        let seed = if i % 2 == 0 { 0xAAAA } else { rng.next_u64() };
        let chained = synth_chained(seed, n_blocks);
        let slot = i as u64 + 1;
        linear.insert(&chained, lk, slot);
        radix.insert(&chained, lk, slot);
        entries.push((chained, lk, slot));
        // review MEDIUM-1: catch count-divergence at the source.
        assert_eq!(
            linear.len(),
            radix.len(),
            "len() drift after insert #{i}: linear={} radix={}",
            linear.len(),
            radix.len()
        );
    }

    // 1000 random prompts: half "near hits" derived from a populated
    // entry, half pure-random noise.
    for q in 0..1000 {
        let lk = layout_keys[rng.next_usize_in(0, layout_keys.len() - 1)];
        let chained = if q % 2 == 0 {
            // Take an existing entry, optionally truncate or extend it.
            let base = &entries[rng.next_usize_in(0, n_entries - 1)].0;
            let take = rng.next_usize_in(1, base.len());
            let mut prompt = base[..take].to_vec();
            // 25% chance of appending divergent blocks.
            if rng.next_u32_in(0, 3) == 0 {
                let extra = rng.next_usize_in(1, 4);
                prompt.extend(synth_chained(rng.next_u64(), extra));
            }
            prompt
        } else {
            // Pure noise prompt.
            let n_blocks = rng.next_usize_in(1, max_blocks + 2);
            synth_chained(rng.next_u64(), n_blocks)
        };

        let lin = linear.match_best(&chained, lk);
        let rdx = radix.match_best(&chained, lk);

        // review MEDIUM-1: assert Some/None parity first — masks
        // the depth assertion below from silently passing on (None,
        // None).
        assert_eq!(
            lin.is_some(),
            rdx.is_some(),
            "Some/None mismatch at q={q}: linear={lin:?} radix={rdx:?}"
        );

        if let (Some((lin_slot, lin_n)), Some((rdx_slot, rdx_n))) = (lin, rdx) {
            assert_eq!(
                lin_n, rdx_n,
                "n_matched_blocks mismatch at q={q}: linear={lin:?} radix={rdx:?}"
            );

            // Slot identity is asserted only when the matched prefix is
            // unambiguous — i.e. exactly one populated entry shares the
            // first `lin_n` blocks under this layout_key. When two or
            // more entries share the matched prefix, the LinearScan vs
            // Radix tiebreaker is impl-defined and we accept divergence.
            let matched_prefix = &chained[..lin_n];
            let candidates = entries
                .iter()
                .filter(|(c, k, _)| {
                    *k == lk
                        && c.len() >= lin_n
                        && c.iter()
                            .zip(matched_prefix.iter())
                            .take_while(|(a, b)| a == b)
                            .count()
                            == lin_n
                })
                .count();
            if candidates == 1 {
                assert_eq!(
                    lin_slot, rdx_slot,
                    "slot_id mismatch at q={q} with unambiguous prefix: linear={lin:?} radix={rdx:?}"
                );
            }
        }
    }
}

/// review MEDIUM-2: re-inserting at the same
/// `(chained_hashes, layout_key)` key with a *different* slot id must
/// overwrite the prior slot (LinearScan contract). Pre-fix the radix
/// tree appended a second tuple at the leaf and `match_best` returned
/// the older slot via `max_by_key(leaf_depth)` tiebreak.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn radix_insert_same_key_different_slot_id_overwrites() {
    let mut tree = RadixTree::new();
    let key = synth_chained(0xBEEF, 4);
    tree.insert(&key, 0, 100);
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.match_best(&key, 0), Some((100, 4)));

    // Overwrite with a fresh slot id.
    tree.insert(&key, 0, 200);
    assert_eq!(
        tree.len(),
        1,
        "len() must stay at 1 after overwrite at the same key"
    );
    let got = tree
        .match_best(&key, 0)
        .expect("post-overwrite lookup hits");
    assert_eq!(
        got,
        (200, 4),
        "match_best must return the new slot after overwrite"
    );

    // Same on LinearScan for cross-impl parity.
    let mut linear = LinearScan::new();
    linear.insert(&key, 0, 100);
    linear.insert(&key, 0, 200);
    assert_eq!(linear.len(), 1);
    assert_eq!(linear.match_best(&key, 0), Some((200, 4)));
}

// --- Startup-rebuild determinism ---

#[test]
fn radix_rebuild_from_sqlite_is_deterministic() {
    // Construct an arbitrary "SQLite snapshot" — two iterations must
    // produce a byte-identical tree (canonical_hash equality).
    let rows: Vec<RebuildRow> = (0..16u64)
        .map(|i| RebuildRow {
            last_block_hash: i.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            layout_key: i % 3,
            row_id: 1000 + i,
        })
        .collect();
    let (a, _) = rebuild_from_sqlite_rows(rows.iter().copied());
    let (b, _) = rebuild_from_sqlite_rows(rows.iter().copied());
    assert_eq!(a.canonical_hash(), b.canonical_hash());
    assert_eq!(a.len(), b.len());
}

// --- LinearScan basic sanity ---

#[test]
fn linear_empty_returns_none() {
    let idx = LinearScan::new();
    let q = synth_chained(1, 3);
    assert!(idx.match_best(&q, 0).is_none());
    assert!(idx.is_empty());
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn linear_longest_prefix_wins() {
    let mut idx = LinearScan::new();
    let mut a = synth_chained(0x11, 2);
    let mut b = a.clone();
    a.extend(synth_chained(0x22, 4));
    b.extend(synth_chained(0x33, 1));
    idx.insert(&a, 0, 1);
    idx.insert(&b, 0, 2);
    // Query == a → 6 blocks (slot 1) beats 3 blocks (slot 2).
    let (slot, n) = idx.match_best(&a, 0).unwrap();
    assert_eq!((slot, n), (1, 6));
}

// --- CLI parse ---

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prefix_index_kind_parse_round_trip() {
    assert_eq!(
        PrefixIndexKind::from_str("linear").unwrap(),
        PrefixIndexKind::Linear
    );
    assert_eq!(
        PrefixIndexKind::from_str("RADIX").unwrap(),
        PrefixIndexKind::Radix
    );
    assert!(PrefixIndexKind::from_str("unknown").is_err());
}
