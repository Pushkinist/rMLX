//! RAII block handles: `MutableBlock` and `ImmutableBlock`.

#![allow(clippy::missing_fields_in_debug, clippy::option_as_ref_cloned)]

use std::sync::{Mutex, Weak};

use super::super::events::EventReleaseHandle;
use super::super::{BlockHash, BlockId, BlockMetadata};
use super::core::{BlockStore, StoreInner};
use super::slot::SlotState;

/// RAII handle for a mutable (uncommitted) block. Drop without `register`
/// returns the slot to the reset pool.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal RAII handle — fields are private; public API is register() and the Drop impl"
)]
pub struct MutableBlock<T: BlockMetadata> {
    pub(super) store: Weak<Mutex<StoreInner<T>>>,
    pub(super) id: BlockId,
    pub(super) committed: bool,
}

impl<T: BlockMetadata> std::fmt::Debug for MutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutableBlock")
            .field("id", &self.id)
            .field("committed", &self.committed)
            .finish()
    }
}

impl<T: BlockMetadata> MutableBlock<T> {
    /// Return the block's slot id.
    pub fn id(&self) -> BlockId {
        self.id
    }

    /// Commit this mutable to active under `hash` + payload. Returns an
    /// `ImmutableBlock` carrying RAII release semantics + the lifecycle event
    /// handle.
    pub fn register(
        mut self,
        store: &BlockStore<T>,
        hash: BlockHash,
        payload: T,
    ) -> ImmutableBlock<T> {
        let (id, _created, handle, _events) = store.commit_block(self.id, hash, payload);
        self.committed = true;
        ImmutableBlock {
            store: store.clone(),
            id,
            hash,
            release: Some(handle),
        }
    }
}

impl<T: BlockMetadata> Drop for MutableBlock<T> {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Roll back: slot stays Mutable in current design? Reference resets
        // back to Reset. Mirror that.
        if let Some(inner) = self.store.upgrade() {
            let mut g = inner.lock().unwrap();
            let slot = &mut g.slots[self.id];
            if matches!(slot.state, SlotState::Mutable) {
                slot.state = SlotState::Reset;
                slot.hash = None;
                slot.payload = None;
                slot.refcount = 0;
                slot.release = None;
                g.free.push_back(self.id);
            }
        }
    }
}

/// Immutable handle to a registered block. Cloning increments refcount;
/// dropping decrements (and demotes to Inactive when last drop). Also owns an
/// `EventReleaseHandle` which fires `KvCacheEvent::Remove` on terminal drop.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal RAII handle — fields are private; public API is the Clone and Drop impls plus hash/id accessors"
)]
pub struct ImmutableBlock<T: BlockMetadata> {
    pub(super) store: BlockStore<T>,
    pub(super) id: BlockId,
    pub(super) hash: BlockHash,
    pub(super) release: Option<EventReleaseHandle>,
}

impl<T: BlockMetadata> std::fmt::Debug for ImmutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImmutableBlock")
            .field("id", &self.id)
            .field("hash", &self.hash)
            .finish()
    }
}

impl<T: BlockMetadata> ImmutableBlock<T> {
    /// Return the block's slot id.
    pub fn id(&self) -> BlockId {
        self.id
    }
    /// Return the chained block hash.
    pub fn hash(&self) -> BlockHash {
        self.hash
    }
}

impl<T: BlockMetadata> Clone for ImmutableBlock<T> {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn clone(&self) -> Self {
        // Cloning yields another live ref → bump refcount.
        let mut g = self.store.inner.lock().unwrap();
        g.slots[self.id].refcount += 1;
        drop(g);
        Self {
            store: self.store.clone(),
            id: self.id,
            hash: self.hash,
            // Release-handle is per-handle; emit Remove once when both Arcs
            // drop. Mirror by sharing the same event manager but skipping the
            // emit on clone — emit only fires on the **last** EventReleaseHandle.
            release: self.release.as_ref().map(EventReleaseHandle::clone_for_dup),
        }
    }
}

impl<T: BlockMetadata> Drop for ImmutableBlock<T> {
    fn drop(&mut self) {
        // Release the per-slot refcount.
        self.store.release_ref(self.id);
        // EventReleaseHandle drop emits Remove when the last clone goes.
        // (handled by EventReleaseHandle's Arc-based refcount).
        drop(self.release.take());
    }
}
