use super::*;

#[derive(Debug, Clone)]
struct P(#[allow(dead_code)] u32);
impl BlockMetadata for P {}

fn mgr(cap: usize) -> BlockManager<P> {
    let cfg = BlockManagerConfig {
        capacity: cap,
        tinylfu_capacity: 1024,
        block_tokens: 4, // small for tests
        ..Default::default()
    };
    BlockManager::new(cfg)
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn register(m: &BlockManager<P>, hash: BlockHash) -> ImmutableBlock<P> {
    let mut mu = m.allocate_blocks(1).unwrap();
    let h = mu.pop().unwrap();
    m.register_block(h, hash, P(0))
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn match_blocks_returns_active_hit_via_hashmap() {
    let m = mgr(4);
    let _i = register(&m, 1);
    let outcomes = m.match_blocks(&[1]);
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(&outcomes[0], MatchOutcome::Active(_, _)));
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn match_blocks_returns_resurrected_for_inactive_hit() {
    let m = mgr(4);
    let i = register(&m, 7);
    drop(i);
    let outcomes = m.match_blocks(&[7]);
    assert!(matches!(&outcomes[0], MatchOutcome::Resurrected(_, _)));
}

#[test]
fn scan_matches_longest_prefix_wins() {
    let m = mgr(4);
    let _a = register(&m, 1);
    let _b = register(&m, 2);
    let _c = register(&m, 3);
    // First three hits, fourth miss.
    let res = m.scan_matches(&[1, 2, 3, 999]);
    assert_eq!(res.n_blocks, 3);
    // block_tokens = 4 in this test config
    assert_eq!(res.n_tokens, 12);
}

#[test]
fn scan_matches_returns_zero_on_first_miss() {
    let m = mgr(4);
    let _a = register(&m, 1);
    let res = m.scan_matches(&[999, 1]);
    assert_eq!(res.n_blocks, 0);
    assert_eq!(res.n_tokens, 0);
}

#[test]
fn chained_seed_changes_with_layout_key() {
    let a = BlockManagerConfig {
        layout_key: 1,
        ..Default::default()
    };
    let b = BlockManagerConfig {
        layout_key: 2,
        ..Default::default()
    };
    assert_ne!(a.chained_seed(), b.chained_seed());
}

#[test]
fn chained_seed_changes_with_lora_salt() {
    let base = BlockManagerConfig {
        layout_key: 7,
        ..Default::default()
    };
    let lora = BlockManagerConfig {
        lora_salt: Some(0xfeed_face),
        ..base
    };
    assert_ne!(base.chained_seed(), lora.chained_seed());
}

#[test]
fn chained_seed_changes_with_mm_hash() {
    let base = BlockManagerConfig {
        layout_key: 7,
        ..Default::default()
    };
    let mm = BlockManagerConfig {
        mm_hash: Some(0xdead_beef),
        ..base
    };
    assert_ne!(base.chained_seed(), mm.chained_seed());
}

#[test]
fn chained_block_hashes_match_under_same_seed() {
    let cfg = BlockManagerConfig {
        block_tokens: 4,
        layout_key: 42,
        ..Default::default()
    };
    let m: BlockManager<P> = BlockManager::new(cfg.clone());
    let tokens: Vec<u32> = (0u32..12).collect();
    let h1 = m.chained_block_hashes(&tokens);
    let h2 = m.chained_block_hashes(&tokens);
    assert_eq!(h1, h2);
    // Different layout → different hashes.
    let cfg2 = BlockManagerConfig {
        layout_key: 43,
        ..cfg
    };
    let m2: BlockManager<P> = BlockManager::new(cfg2);
    let h3 = m2.chained_block_hashes(&tokens);
    assert_ne!(h1, h3);
}

/// Property test: insert N blocks with planted TinyLFU counts (via
/// `touch_frequency`, which bumps the sketch without changing refcount),
/// then evict repeatedly and verify eviction order follows the TinyLFU
/// bin (`[3, 8, 15]`) → MultiLRU tier mapping.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn eviction_order_follows_tinylfu_bin_mapping() {
    // Capacity equals the plan size so allocate_blocks must go through
    // the eviction path (no free slots left after the plan registers).
    let m = mgr(12);
    // (hash, extra_increments). Each register also increments the sketch
    // once on commit, so final count = 1 + extra.
    let plan: Vec<(BlockHash, u32)> = vec![
        (101, 0),  // count=1 -> bin 0
        (102, 0),  // count=1 -> bin 0
        (103, 1),  // count=2 -> bin 0
        (104, 3),  // count=4 -> bin 1
        (105, 5),  // count=6 -> bin 1
        (106, 8),  // count=9 -> bin 2
        (107, 9),  // count=10 -> bin 2
        (108, 15), // count saturates at 15 -> bin 3
        (109, 30), // saturates at 15 -> bin 3
        (110, 0),  // bin 0
        (111, 0),  // bin 0
        (112, 4),  // count=5 -> bin 1
    ];
    let mut imms = Vec::new();
    for (h, extra) in &plan {
        let imm = register(&m, *h);
        for _ in 0..*extra {
            m.touch_frequency(*h);
        }
        imms.push(imm);
    }
    // Drop all -> all Inactive (insert re-bins under current count).
    drop(imms);

    // Force 5 evictions: keep each newly-allocated MutableBlock alive so
    // its slot is not immediately recycled back to the free pool. Each
    // allocation past the free-pool-empty point pops the coldest inactive.
    let mut kept = Vec::new();
    for _ in 0..5 {
        kept.extend(m.allocate_blocks(1).unwrap());
    }
    // Hold `kept` until after the asserts so the allocations stay in
    // Mutable state and the eviction work is preserved.
    let _kept_alive = kept;
    // Bin 0 contained: 101, 102, 103, 110, 111 (count <= 2 < threshold 3).
    for h in [101u64, 102, 103, 110, 111] {
        let r = m.match_blocks(&[h]);
        assert!(
            matches!(&r[0], MatchOutcome::Miss),
            "hash {h} (bin 0) expected evicted, got {:?}",
            r[0]
        );
    }
    // Hot bin 3 entries (108, 109) must remain.
    for h in [108u64, 109] {
        let r = m.match_blocks(&[h]);
        assert!(
            matches!(
                &r[0],
                MatchOutcome::Resurrected(_, _) | MatchOutcome::Active(_, _)
            ),
            "hot hash {h} (bin 3) unexpectedly missing"
        );
    }
}
