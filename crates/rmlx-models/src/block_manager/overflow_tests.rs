use super::*;
use crate::block_manager::events::EventManager;
use crate::block_manager::store::BlockStore;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Payload(#[allow(dead_code)] u32);
impl BlockMetadata for Payload {}

#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tier_zero_evict_calls_overflow_sink_with_layout_key() {
    let events = Arc::new(EventManager::new());
    let store: BlockStore<Payload> = BlockStore::new(2, 1024, events);
    let layout_key = 0xc0ffee_u64;
    let sink = Arc::new(RecordingSink::<Payload>::new(layout_key));
    store.set_overflow_sink(sink.clone());

    // Register two blocks, drop refs → both Inactive in tier 0 (cold).
    let mut mus = store.allocate_blocks(2).unwrap();
    let m1 = mus.remove(0);
    let m0 = mus.remove(0);
    let i0 = m0.register(&store, 0xabcd, Payload(0));
    let i1 = m1.register(&store, 0xbcde, Payload(1));
    drop(i0);
    drop(i1);

    // Allocate 1 → evicts the coldest, which offers to the sink.
    let _ = store.allocate_blocks(1).unwrap();

    let offers = sink.offers();
    assert_eq!(offers.len(), 1, "exactly one block evicted + offered");
    // First evicted is the first-registered (FIFO inside tier 0).
    assert_eq!(offers[0].0, 0xabcd);
    // Layout key is exposed to the sink.
    assert_eq!(sink.layout_key(), layout_key);
}
