use super::*;

#[test]
fn increment_then_estimate_returns_at_least_one() {
    let mut s = TinyLfuSketch::new(1024);
    s.increment(0xdead_beef);
    assert!(s.estimate(0xdead_beef) >= 1);
}

#[test]
fn estimate_saturates_at_fifteen() {
    let mut s = TinyLfuSketch::new(1024);
    for _ in 0..50 {
        s.increment(42);
    }
    assert_eq!(s.estimate(42), 15);
}

#[test]
fn decay_halves_target_slot_and_size_directly() {
    // Direct unit test of the halving step — drive a single saturating
    // key, call decay() explicitly, verify the estimate halves.
    let mut s = TinyLfuSketch::new(4096);
    let key = 0xfeed_face_dead_beefu64;
    for _ in 0..15 {
        s.increment(key);
    }
    assert_eq!(s.estimate(key), 15);
    let size_before = s.size;
    s.decay();
    // 15 >> 1 = 7
    assert_eq!(s.estimate(key), 7);
    // size shrinks (size/2 - zeroed/4). It must not grow.
    assert!(s.size <= size_before / 2 + 1);
}

#[test]
fn decay_fires_when_threshold_crossed() {
    // Drive size past threshold and verify decay was triggered (size
    // dropped from threshold-or-more to below threshold).
    let cap = 64usize;
    let mut s = TinyLfuSketch::new(cap);
    for k in 0u64..(cap as u64 * 12) {
        s.increment(k);
    }
    // After at least one decay, size must be below decay_threshold.
    assert!(
        s.size < s.decay_threshold,
        "size {} should be below decay_threshold {} after triggered halving",
        s.size,
        s.decay_threshold
    );
}

#[test]
fn bin_thresholds_cross_at_3_8_15() {
    assert_eq!(bin_for_count(0), 0);
    assert_eq!(bin_for_count(2), 0);
    assert_eq!(bin_for_count(3), 1);
    assert_eq!(bin_for_count(7), 1);
    assert_eq!(bin_for_count(8), 2);
    assert_eq!(bin_for_count(14), 2);
    assert_eq!(bin_for_count(15), 3);
}

#[test]
fn increment_bumps_bin_through_3_8_15_transitions() {
    let mut s = TinyLfuSketch::new(2048);
    let k = 0x1234_5678_9abc_def0;
    // 0 -> 0
    assert_eq!(bin_for_count(s.estimate(k)), 0);
    // 3 increments → count >= 3 → bin 1
    for _ in 0..3 {
        s.increment(k);
    }
    assert!(bin_for_count(s.estimate(k)) >= 1);
    // up to 8 → bin 2
    for _ in 0..5 {
        s.increment(k);
    }
    assert!(bin_for_count(s.estimate(k)) >= 2);
    // up to 15 → bin 3
    for _ in 0..10 {
        s.increment(k);
    }
    assert_eq!(bin_for_count(s.estimate(k)), 3);
}

#[test]
fn tracker_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TinyLfuTracker>();
}
