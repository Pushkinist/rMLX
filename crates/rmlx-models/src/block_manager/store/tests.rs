use super::*;
use crate::block_manager::events::{EventManager, EventSubscriber, KvCacheEvent};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Tok(u32);
impl crate::block_manager::BlockMetadata for Tok {}

fn store(capacity: usize) -> BlockStore<Tok> {
    let events = Arc::new(EventManager::new());
    BlockStore::new(capacity, 1024, events)
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn allocate_promotes_reset_to_mutable() {
    let s = store(4);
    let mu = s.allocate_blocks(1).unwrap();
    assert_eq!(mu.len(), 1);
    let id = mu[0].id();
    assert_eq!(s.slot_state(id), Some(SlotState::Mutable));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn drop_mutable_returns_slot_to_reset() {
    let s = store(2);
    {
        let _mu = s.allocate_blocks(1).unwrap();
    }
    // After drop, the slot is back in the free pool.
    let stats = s.stats();
    assert_eq!(stats.free, 2);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn register_promotes_mutable_to_primary() {
    let s = store(2);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let id = m.id();
    let imm = m.register(&s, 0xdead, Tok(7));
    assert_eq!(s.slot_state(id), Some(SlotState::Primary));
    assert_eq!(s.slot_refcount(id), Some(1));
    drop(imm);
    assert_eq!(s.slot_state(id), Some(SlotState::Inactive));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn drop_immutable_demotes_to_inactive() {
    let s = store(2);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let id = m.id();
    let imm = m.register(&s, 99, Tok(0));
    drop(imm);
    // Slot still occupies hash 99 in slots[].hash, just sitting Inactive.
    assert_eq!(s.slot_state(id), Some(SlotState::Inactive));
    // active_by_hash purged on release_ref (Primary→Inactive). Verify.
    assert!(!s.active_hashes().contains(&99));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn match_one_hit_active_bumps_refcount() {
    let s = store(2);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let id = m.id();
    let _imm = m.register(&s, 7, Tok(0));
    assert_eq!(s.slot_refcount(id), Some(1));
    let outcome = s.match_one(7);
    assert!(matches!(outcome, MatchOutcome::Active(got_id, _) if got_id == id));
    assert_eq!(s.slot_refcount(id), Some(2));
    // Outcome carries a live ImmutableBlock; dropping it releases the bump.
    drop(outcome);
    assert_eq!(s.slot_refcount(id), Some(1));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn match_one_hit_resurrects_inactive() {
    let s = store(2);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let id = m.id();
    let imm = m.register(&s, 17, Tok(0));
    drop(imm);
    assert_eq!(s.slot_state(id), Some(SlotState::Inactive));
    let outcome = s.match_one(17);
    assert!(matches!(outcome, MatchOutcome::Resurrected(got_id, _) if got_id == id));
    assert_eq!(s.slot_state(id), Some(SlotState::Primary));
    assert_eq!(s.slot_refcount(id), Some(1));
}

#[test]
fn match_one_miss_returns_miss() {
    let s = store(2);
    assert!(matches!(s.match_one(0xdead_beef), MatchOutcome::Miss));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn full_store_evicts_inactive_for_new_alloc() {
    let s = store(2);
    // Register two blocks, then drop both → both Inactive.
    let mut mus = s.allocate_blocks(2).unwrap();
    let m1 = mus.remove(0);
    let m0 = mus.remove(0);
    let i0 = m0.register(&s, 1, Tok(0));
    let i1 = m1.register(&s, 2, Tok(0));
    drop(i0);
    drop(i1);
    assert_eq!(s.stats().inactive, 2);
    // Allocate 1 → evicts oldest inactive.
    let _ = s.allocate_blocks(1).unwrap();
    let st = s.stats();
    assert_eq!(st.inactive, 1);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn full_store_returns_error_when_nothing_evictable() {
    let s = store(1);
    let _mu = s.allocate_blocks(1).unwrap(); // holds it Mutable
    let r = s.allocate_blocks(1);
    assert!(matches!(r, Err(StoreError::Full { .. })));
}

struct Counter {
    creates: AtomicUsize,
    removes: AtomicUsize,
}
impl Counter {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            creates: AtomicUsize::new(0),
            removes: AtomicUsize::new(0),
        })
    }
}
impl EventSubscriber for Counter {
    fn on_event(&self, e: KvCacheEvent) {
        match e {
            KvCacheEvent::Create(_) => {
                self.creates.fetch_add(1, Ordering::Relaxed);
            }
            KvCacheEvent::Remove(_) => {
                self.removes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn lifecycle_create_emitted_on_register() {
    let events = Arc::new(EventManager::new());
    let sub = Counter::new();
    events.subscribe(sub.clone());
    let s = BlockStore::<Tok>::new(2, 1024, events);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let _imm = m.register(&s, 5, Tok(0));
    assert!(sub.creates.load(Ordering::Relaxed) >= 1);
}

#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn lifecycle_remove_emitted_on_terminal_drop() {
    let events = Arc::new(EventManager::new());
    let sub = Counter::new();
    events.subscribe(sub.clone());
    let s = BlockStore::<Tok>::new(2, 1024, events);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let imm = m.register(&s, 5, Tok(0));
    // Drop both immutable and the release-handle's last clone.
    drop(imm);
    // imm dropping reduces refcount → Inactive, but the canonical clone
    // is still held by the slot. Force the slot's clone to drop by
    // evicting (allocate past capacity).
    // Capacity 2, so allocate 2 more blocks (one extra triggers eviction).
    let _force = s.allocate_blocks(2).unwrap();
    assert_eq!(sub.removes.load(Ordering::Relaxed), 1);
}

/// HIGH-1 regression: eviction must not emit a duplicate Remove. With the
/// canonical-handle plumbing in place, only one Remove fires per Primary
/// lifecycle even when eviction resets the slot.
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn eviction_does_not_emit_double_remove() {
    let events = Arc::new(EventManager::new());
    let sub = Counter::new();
    events.subscribe(sub.clone());
    let s = BlockStore::<Tok>::new(1, 1024, events);
    let mut mu = s.allocate_blocks(1).unwrap();
    let m = mu.pop().unwrap();
    let imm = m.register(&s, 0xfeed, Tok(0));
    // Drop the primary handle → slot goes Inactive, but the canonical
    // clone is still in the slot. No Remove yet.
    drop(imm);
    assert_eq!(sub.removes.load(Ordering::Relaxed), 0);
    // Allocate-evict: forces the Inactive slot to evict back to Mutable.
    let _ev = s.allocate_blocks(1).unwrap();
    // Exactly one Remove fired.
    assert_eq!(
        sub.removes.load(Ordering::Relaxed),
        1,
        "expected single Remove on eviction"
    );
}

/// HIGH-3 regression: probing more hashes than `scan_matches` accepts must
/// not leak refcounts on the unaccepted prefix. With the probe/confirm
/// split, the unaccepted tail is never confirmed, so no refcount bumps to
/// undo.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn match_blocks_releases_unaccepted_prefix() {
    let s = store(4);
    // Register 3 blocks at hashes 1,2,3.
    let mut mus = s.allocate_blocks(3).unwrap();
    let m2 = mus.pop().unwrap();
    let m1 = mus.pop().unwrap();
    let m0 = mus.pop().unwrap();
    let i0 = m0.register(&s, 1, Tok(0));
    let i1 = m1.register(&s, 2, Tok(0));
    let i2 = m2.register(&s, 3, Tok(0));
    // baseline refcounts.
    assert_eq!(s.slot_refcount(i0.id()), Some(1));
    assert_eq!(s.slot_refcount(i1.id()), Some(1));
    assert_eq!(s.slot_refcount(i2.id()), Some(1));
    // Probe 5 hashes: prefix [1,2,3] hits, then [999,888] miss. The probe
    // path does NOT bump refcounts on hits past the first miss (and
    // doesn't bump at all — confirm does the bump).
    let probe = s.probe_blocks(&[1, 2, 3, 999, 888]);
    assert_eq!(probe.len(), 5);
    assert!(probe[0].is_some());
    assert!(probe[1].is_some());
    assert!(probe[2].is_some());
    assert!(probe[3].is_none());
    assert!(probe[4].is_none());
    // After probe alone — refcounts unchanged.
    assert_eq!(s.slot_refcount(i0.id()), Some(1));
    assert_eq!(s.slot_refcount(i1.id()), Some(1));
    assert_eq!(s.slot_refcount(i2.id()), Some(1));
    // Now confirm a prefix of 2 (not all 3 hits): only those 2 bump.
    let confirmed = s.confirm_prefix(&[1, 2, 3], 2);
    assert_eq!(confirmed.len(), 2);
    assert_eq!(s.slot_refcount(i0.id()), Some(2));
    assert_eq!(s.slot_refcount(i1.id()), Some(2));
    // The 3rd slot was not confirmed — its refcount stays at 1.
    assert_eq!(s.slot_refcount(i2.id()), Some(1));
    // Drop the confirmed outcomes — slot 0 and 1 return to 1.
    drop(confirmed);
    assert_eq!(s.slot_refcount(i0.id()), Some(1));
    assert_eq!(s.slot_refcount(i1.id()), Some(1));
    // Drop the original ImmutableBlocks — all slots return to 0 (Inactive).
    drop(i0);
    drop(i1);
    drop(i2);
    // All refcounts to 0; no leak.
    let stats = s.stats();
    assert_eq!(stats.active, 0, "no leaked refs in active set");
}
