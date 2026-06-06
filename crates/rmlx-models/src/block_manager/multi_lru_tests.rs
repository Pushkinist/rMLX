use super::*;

fn warmed_tracker(hot_keys: &[(BlockHash, usize)]) -> Arc<TinyLfuTracker> {
    let t = Arc::new(TinyLfuTracker::new(1024));
    for &(k, n) in hot_keys {
        for _ in 0..n {
            t.increment(k);
        }
    }
    t
}

#[test]
fn empty_evict_returns_none() {
    let t = Arc::new(TinyLfuTracker::new(64));
    let mut b = MultiLruBackend::new(t);
    assert!(b.evict().is_none());
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn coldest_pool_evicts_first() {
    // Build a mix: one hot (count >= 15), one mid (count >= 3), one cold.
    let t = warmed_tracker(&[(101, 30), (102, 5)]);
    let mut b = MultiLruBackend::new(t);
    b.insert(101); // hot → bin 3
    b.insert(102); // mid → bin 1
    b.insert(103); // cold → bin 0
    let first = b.evict().unwrap();
    assert_eq!(first, 103, "cold block should evict first");
    // After draining tier 0, the next evict goes to tier 1.
    let second = b.evict().unwrap();
    assert_eq!(second, 102);
    // Then the hot one.
    let third = b.evict().unwrap();
    assert_eq!(third, 101);
    assert!(b.evict().is_none());
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn warmer_tiers_are_not_drained_while_pool_zero_has_entries() {
    let t = warmed_tracker(&[(201, 20)]);
    let mut b = MultiLruBackend::new(t);
    // Three cold blocks, one hot.
    b.insert(1);
    b.insert(2);
    b.insert(3);
    b.insert(201);
    assert_eq!(b.tier_lens(), [3, 0, 0, 1]);
    let popped = b.evict().unwrap();
    // First evicted must come from pool 0 (FIFO inside the pool).
    assert_eq!(popped, 1);
    let popped = b.evict().unwrap();
    assert_eq!(popped, 2);
    // Pool 0 not fully drained yet — hot block (201) must still be there.
    assert!(b.contains(201));
}

#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn touch_repromotes_after_tinylfu_bin_transition() {
    let t = Arc::new(TinyLfuTracker::new(1024));
    let mut b = MultiLruBackend::new(t.clone());
    b.insert(42);
    assert_eq!(b.tier_lens(), [1, 0, 0, 0]);
    // Drive the TinyLFU count up across the first threshold (3).
    for _ in 0..5 {
        t.increment(42);
    }
    // touch() re-bins under current count.
    b.touch(42);
    assert_eq!(b.tier_lens()[0], 0);
    assert!(b.tier_lens()[1] >= 1);
}

#[test]
fn remove_returns_false_when_absent() {
    let t = Arc::new(TinyLfuTracker::new(64));
    let mut b = MultiLruBackend::new(t);
    assert!(!b.remove(999));
    b.insert(7);
    assert!(b.remove(7));
    assert!(!b.contains(7));
}

#[test]
fn insert_dedups_on_reinsert() {
    let t = Arc::new(TinyLfuTracker::new(64));
    let mut b = MultiLruBackend::new(t);
    b.insert(5);
    b.insert(5);
    b.insert(5);
    assert_eq!(b.len(), 1);
}
