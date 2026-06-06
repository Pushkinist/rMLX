//! Block lifecycle events + RAII release.
//!
//! Reference: `dynamo/lib/kvbm-logical/src/events/protocol.rs`.
//!
//! Two event kinds — `Create` (block newly registered) and `Remove` (block
//! evicted / last reference dropped). The reference uses
//! `tokio::sync::broadcast` for a fan-out channel; rMLX is single-process and
//! the only required consumer is `tracing` plus the metrics-events sink (out
//! of scope here), so we wire to a simple subscriber callback under a mutex
//! plus structured `tracing::info!` calls.
//!
//! ## RAII
//!
//! `EventReleaseHandle` is an `Arc`-backed token. The `Remove` event fires
//! exactly once, when the **last clone** of the handle drops — matching the
//! reference's `Arc<Inner> + Drop` semantics. Multiple `ImmutableBlock`s
//! cloning the same registration share one handle.
//!
//! ## PowerOfTwoPolicy
//!
//! The reference batches events: only blocks at power-of-2 positions in a
//! batch are emitted on the wire. We replicate the policy as a pure function
//! `PowerOfTwoPolicy::keep(position)` so callers can filter before emit.

use std::sync::{Arc, Mutex};

use super::BlockHash;

/// Lifecycle event variants. Matches the reference enum.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed event enum — two lifecycle events (Create/Remove); adding an event requires updating EventManager::emit and all subscribers"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Per-block lifecycle event emitted by the block manager.
pub enum KvCacheEvent {
    /// Block newly registered in the active set.
    Create(BlockHash),
    /// Block evicted or last reference dropped.
    Remove(BlockHash),
}

/// Batch flavour. Sorted by token position (asc for Create, desc for Remove)
/// per the reference. Sorting happens at the call site that builds the
/// batch — the event manager just publishes whatever ordering it gets.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed event-batch enum — three variants (Create/Remove/Shutdown); adding a variant requires updating all EventSubscriber implementations"
)]
#[derive(Debug, Clone)]
/// Batched event group delivered to subscribers.
pub enum KvCacheEvents {
    /// Batch of newly created block hashes.
    Create(Vec<BlockHash>),
    /// Batch of removed block hashes.
    Remove(Vec<BlockHash>),
    /// Signal that the block manager is shutting down.
    Shutdown,
}

/// Trait for an event subscriber. The block manager registers one or more
/// subscribers; emit fans out under the events lock.
pub trait EventSubscriber: Send + Sync {
    /// Called once per event by the `EventManager` fan-out loop.
    fn on_event(&self, event: KvCacheEvent);
}

/// Power-of-2 batch filter. Drops blocks whose position-in-batch is not a
/// power of two — keeps event volume O(log N) in the batch size.
#[allow(
    clippy::exhaustive_structs,
    reason = "unit struct policy — no fields; public API is the keep() and filter() methods"
)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PowerOfTwoPolicy;

impl PowerOfTwoPolicy {
    /// True when `position` is a power of two (including 1). Position is the
    /// 1-indexed offset of the block in the batch (so 1, 2, 4, 8, …).
    pub fn keep(position: usize) -> bool {
        position > 0 && position.is_power_of_two()
    }

    /// Filter `batch` keeping only blocks at power-of-2 positions (1-indexed).
    pub fn filter<T: Copy>(batch: &[T]) -> Vec<T> {
        batch
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if Self::keep(i + 1) { Some(*v) } else { None })
            .collect()
    }
}

/// Manages subscribers + emits events.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed manager — field is private; public API is subscribe() and emit()"
)]
#[derive(Default)]
pub struct EventManager {
    subscribers: Mutex<Vec<Arc<dyn EventSubscriber>>>,
}

impl std::fmt::Debug for EventManager {
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventManager")
            .field("n_subscribers", &self.subscribers.lock().unwrap().len())
            .finish()
    }
}

impl EventManager {
    /// Create a new empty `EventManager` with no subscribers.
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    /// Register a subscriber that will receive all future events.
    pub fn subscribe(&self, sub: Arc<dyn EventSubscriber>) {
        self.subscribers.lock().unwrap().push(sub);
    }

    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    /// Fan out `event` to all registered subscribers.
    pub fn emit(&self, event: KvCacheEvent) {
        // The rich `tracing::info!` for lifecycle events is emitted by the
        // call sites that have full context (block_id, tier, tinylfu_count) —
        // see `BlockStore::commit_block` and the `ReleaseInner::drop` path.
        // `emit` is responsible for subscriber fan-out only.
        //
        // Snapshot subscribers OUT of the lock before fanning out: a
        // subscriber's `on_event` may itself drop an `ImmutableBlock`, which
        // would re-enter `EventManager::emit`. Holding `subscribers.lock()`
        // across that call would deadlock (or panic-poison if the user code
        // panics).
        let subs: Vec<Arc<dyn EventSubscriber>> = self.subscribers.lock().unwrap().clone();
        for s in &subs {
            s.on_event(event);
        }
    }
}

/// Inner shared state for a RAII release handle.
struct ReleaseInner {
    events: Arc<EventManager>,
    hash: BlockHash,
}

impl Drop for ReleaseInner {
    fn drop(&mut self) {
        self.events.emit(KvCacheEvent::Remove(self.hash));
    }
}

/// RAII handle: when the last clone drops, `KvCacheEvent::Remove(hash)` is
/// emitted via the event manager. Equivalent to the reference's
/// `EventReleaseHandle` (`protocol.rs:84-90`).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal RAII handle — field is private Arc; public API is Drop emission"
)]
#[derive(Clone)]
pub struct EventReleaseHandle {
    inner: Arc<ReleaseInner>,
}

impl std::fmt::Debug for EventReleaseHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventReleaseHandle")
            .field("hash", &self.inner.hash)
            .field("ref_count", &Arc::strong_count(&self.inner))
            .finish()
    }
}

impl EventReleaseHandle {
    pub(crate) fn new(events: Arc<EventManager>, hash: BlockHash) -> Self {
        Self {
            inner: Arc::new(ReleaseInner { events, hash }),
        }
    }

    /// Clone for an ImmutableBlock duplication — keeps the same inner Arc so
    /// the Remove event fires once when ALL dup handles drop.
    pub(crate) fn clone_for_dup(&self) -> Self {
        self.clone()
    }

    /// The block hash this handle was minted for.
    pub fn hash(&self) -> BlockHash {
        self.inner.hash
    }
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
