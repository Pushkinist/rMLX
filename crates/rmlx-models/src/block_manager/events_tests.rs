use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingSub {
    creates: AtomicUsize,
    removes: AtomicUsize,
}
impl CountingSub {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            creates: AtomicUsize::new(0),
            removes: AtomicUsize::new(0),
        })
    }
    fn creates(&self) -> usize {
        self.creates.load(Ordering::Relaxed)
    }
    fn removes(&self) -> usize {
        self.removes.load(Ordering::Relaxed)
    }
}
impl EventSubscriber for CountingSub {
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
fn raii_drop_emits_remove_once() {
    let mgr = Arc::new(EventManager::new());
    let sub = CountingSub::new();
    mgr.subscribe(sub.clone());
    {
        let h1 = EventReleaseHandle::new(mgr, 42);
        let h2 = h1.clone_for_dup();
        assert_eq!(sub.removes(), 0);
        drop(h1);
        assert_eq!(sub.removes(), 0, "no emit until last clone drops");
        drop(h2);
    }
    assert_eq!(sub.removes(), 1);
}

#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn explicit_emit_create_fires_subscribers() {
    let mgr = Arc::new(EventManager::new());
    let sub = CountingSub::new();
    mgr.subscribe(sub.clone());
    mgr.emit(KvCacheEvent::Create(1));
    mgr.emit(KvCacheEvent::Create(2));
    assert_eq!(sub.creates(), 2);
}

#[test]
fn power_of_two_policy_keeps_only_powers() {
    for p in 1..=20 {
        let want = matches!(p, 1 | 2 | 4 | 8 | 16);
        assert_eq!(PowerOfTwoPolicy::keep(p), want, "pos={p}");
    }
}

#[test]
fn power_of_two_filter_returns_powers_of_two_positions() {
    let batch: Vec<u64> = (0..10).collect();
    let kept = PowerOfTwoPolicy::filter(&batch);
    // positions 1,2,4,8 → values [0, 1, 3, 7]
    assert_eq!(kept, vec![0, 1, 3, 7]);
}
