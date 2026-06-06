//! `BlockStore` — single-mutex unified store + core state machine.

#![allow(clippy::missing_fields_in_debug, clippy::option_as_ref_cloned)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use super::super::events::{EventManager, EventReleaseHandle, KvCacheEvent};
use super::super::multi_lru::{InactiveIndex, MultiLruBackend};
use super::super::overflow::OverflowSink;
use super::super::tinylfu::TinyLfuTracker;
use super::super::{BlockHash, BlockId, BlockMetadata};
use super::handles::{ImmutableBlock, MutableBlock};
use super::slot::{BlockSlot, SlotState};

/// Errors raised by the store. `thiserror` per CLAUDE.md.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Block store has no free or evictable slot to satisfy an allocation.
    #[error("block store is full: capacity {capacity}, evictable {evictable}")]
    Full {
        /// Total store capacity.
        capacity: usize,
        /// Number of blocks currently evictable.
        evictable: usize,
    },
    /// A `BlockId` was outside the valid range `[0, capacity)`.
    #[error("block id {0} out of range")]
    BadId(BlockId),
    /// A block slot was found in an unexpected `SlotState` for the requested op.
    #[error("block slot {id} in unexpected state {state:?}")]
    BadState {
        /// The slot id that was in the wrong state.
        id: BlockId,
        /// The actual state found.
        state: SlotState,
    },
}

/// Result of a `match_blocks` call — one entry per requested hash. On a hit
/// the variant carries an `ImmutableBlock` that owns the bumped refcount and
/// the cloned event-release handle; dropping it releases the slot.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed result enum — three match outcomes; adding an outcome requires updating all match_blocks call sites"
)]
#[derive(Debug)]
pub enum MatchOutcome<T: BlockMetadata> {
    /// Hash hit in active set (refcount bumped, handle cloned).
    Active(BlockId, ImmutableBlock<T>),
    /// Hash hit in inactive set — resurrected to active (refcount bumped, handle minted).
    Resurrected(BlockId, ImmutableBlock<T>),
    /// Miss.
    Miss,
}

impl<T: BlockMetadata> MatchOutcome<T> {
    /// True when this outcome holds a live `ImmutableBlock`.
    pub fn is_hit(&self) -> bool {
        !matches!(self, MatchOutcome::Miss)
    }
}

/// Inner state guarded by the single store mutex.
pub(super) struct StoreInner<T: BlockMetadata> {
    pub(super) slots: Vec<BlockSlot<T>>,
    /// Reset pool — slots ready for fresh allocation. FIFO.
    pub(super) free: VecDeque<BlockId>,
    /// Active set keyed by hash. Multiple `Primary`s can share a hash only via
    /// `Duplicate` policy = Allow; v2 default is Reject — we reject by
    /// returning the existing id and dropping the caller's mutable.
    pub(super) active_by_hash: HashMap<BlockHash, BlockId>,
    /// Inactive index by hash — O(1) reverse lookup from a hash to its slot id
    /// while the slot is sitting Inactive. Kept in sync with `inactive`.
    pub(super) inactive_by_hash: HashMap<BlockHash, BlockId>,
    /// Inactive index — see [`InactiveIndex`].
    pub(super) inactive: Box<dyn InactiveIndex>,
    pub(super) tinylfu: Arc<TinyLfuTracker>,
    pub(super) events: Arc<EventManager>,
    pub(super) overflow: Option<Arc<dyn OverflowSink<T>>>,
}

/// Public store handle. Cloning is cheap (`Arc`-backed) and threads share
/// access to the same inner mutex.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed store — fields are the complete BlockStore contract; adding a field requires updating BlockStore::new and all callers"
)]
pub struct BlockStore<T: BlockMetadata> {
    pub(super) inner: Arc<Mutex<StoreInner<T>>>,
    pub(super) capacity: usize,
}

impl<T: BlockMetadata> Clone for BlockStore<T> {
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            capacity: self.capacity,
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for BlockStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockStore")
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl<T: BlockMetadata> BlockStore<T> {
    /// Return the maximum number of slots in this store.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Build a new store.
    ///
    /// `capacity` is the maximum number of slots; `tinylfu_capacity` sizes the
    /// frequency sketch (default reference uses `2^21` for Medium). `events`
    /// is shared with the manager facade so subscribers see the same stream.
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    pub fn new(capacity: usize, tinylfu_capacity: usize, events: Arc<EventManager>) -> Self {
        let tinylfu = Arc::new(TinyLfuTracker::new(tinylfu_capacity));
        let inactive: Box<dyn InactiveIndex> = Box::new(MultiLruBackend::new(tinylfu.clone()));

        let mut slots = Vec::with_capacity(capacity);
        let mut free = VecDeque::with_capacity(capacity);
        for i in 0..capacity {
            slots.push(BlockSlot::fresh(i));
            free.push_back(i);
        }

        let inner = StoreInner {
            slots,
            free,
            active_by_hash: HashMap::new(),
            inactive_by_hash: HashMap::new(),
            inactive,
            tinylfu,
            events,
            overflow: None,
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
            capacity,
        }
    }

    /// Install an SSD overflow sink. Called once at startup by the integration
    /// layer once an `SsdSpiller` is available.
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn set_overflow_sink(&self, sink: Arc<dyn OverflowSink<T>>) {
        self.inner.lock().unwrap().overflow = Some(sink);
    }

    /// Snapshot of per-tier counts and total active/inactive — debugging.
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn stats(&self) -> StoreStats {
        let g = self.inner.lock().unwrap();
        let tier_lens = g.inactive.tier_lens();
        StoreStats {
            capacity: self.capacity,
            free: g.free.len(),
            active: g.active_by_hash.len(),
            inactive: g.inactive.len(),
            inactive_tier_lens: tier_lens,
        }
    }

    /// Allocate `n` mutable blocks from the reset pool. Evicts inactive
    /// blocks if the reset pool is empty.
    ///
    /// Returns `MutableBlock` handles whose `drop` releases the slot back to
    /// `Reset` if `commit_block` was never called.
    ///
    /// ## Lock discipline
    ///
    /// We hold the store mutex while choosing the next slot and updating
    /// indices. If eviction needs to offer the about-to-drop payload to the
    /// overflow sink, we **drop the mutex** before calling
    /// `OverflowSink::offer_evicted` — the sink is documented as non-blocking
    /// `try_send`, but the API contract (`block_manager/mod.rs` "Lock order"
    /// section) is that the store mutex is never held across a sink call.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn allocate_blocks(&self, n: usize) -> Result<Vec<MutableBlock<T>>, StoreError> {
        let mut out = Vec::with_capacity(n);
        // Sink calls deferred until after we drop the lock guard.
        let mut deferred_offers: Vec<(BlockHash, T, Arc<dyn OverflowSink<T>>)> = Vec::new();
        {
            let mut g = self.inner.lock().unwrap();
            for _ in 0..n {
                let id = if let Some(id) = g.free.pop_front() {
                    id
                } else {
                    // Try evicting from inactive.
                    let evicted = g.inactive.evict();
                    if let Some(h) = evicted {
                        // Inactive entries: look up via inactive_by_hash
                        // (O(1)) — active_by_hash never holds Inactive
                        // hashes (`release_ref` removes them on the
                        // Primary→Inactive transition).
                        let id = if let Some(id) = g.inactive_by_hash.remove(&h) {
                            id
                        } else {
                            // Defensive: hash was evicted from the
                            // tier index but not present in our
                            // reverse map — should not happen, but
                            // fall back to a linear scan rather
                            // than panicking.
                            let id_opt = g.slots.iter().position(|s| s.hash == Some(h));
                            match id_opt {
                                Some(i) => i,
                                None => {
                                    return Err(StoreError::Full {
                                        capacity: self.capacity,
                                        evictable: 0,
                                    });
                                }
                            }
                        };
                        // Reset the slot in-place; defer any overflow
                        // offer to after the lock guard drops.
                        if let Some(offer) = Self::reset_slot_collect_offer(&mut g, id) {
                            deferred_offers.push(offer);
                        }
                        id
                    } else {
                        return Err(StoreError::Full {
                            capacity: self.capacity,
                            evictable: 0,
                        });
                    }
                };
                // Promote to Mutable.
                let slot = &mut g.slots[id];
                debug_assert!(matches!(slot.state, SlotState::Reset));
                slot.state = SlotState::Mutable;
                slot.refcount = 0;
                slot.hash = None;
                slot.payload = None;
                slot.release = None;
                out.push(MutableBlock {
                    store: Arc::downgrade(&self.inner),
                    id,
                    committed: false,
                });
            }
            // Explicit drop documents the lock release point.
            drop(g);
        }
        // Now safely outside the store mutex — offer evicted payloads.
        for (hash, payload, sink) in deferred_offers {
            let tier = 0; // evicted from tier 0 — coldest, by construction.
            tracing::info!(
                event = "kvbm_block_evict_offered",
                block_hash = hash,
                inactive_tier = tier,
                "kvbm: offering evicted block to overflow sink"
            );
            sink.offer_evicted(hash, payload);
        }
        Ok(out)
    }

    /// Evict the slot in `id` back to Reset. Returns the
    /// `(hash, payload, sink)` triple if the slot had a payload AND an
    /// overflow sink is installed; the caller is expected to invoke
    /// `sink.offer_evicted` AFTER dropping the store mutex.
    ///
    /// Does **not** push `id` into `g.free` — the eviction path immediately
    /// promotes the slot to `Mutable` for re-use, so adding it to the free
    /// queue would cause double-allocation on a later `pop_front`.
    ///
    /// Lifecycle: this path does NOT emit `KvCacheEvent::Remove` explicitly.
    /// Remove is emitted by `ReleaseInner::drop` when the last clone of the
    /// `EventReleaseHandle` is released — which, for an Inactive slot, is
    /// the canonical handle stashed in the slot. Setting `slot.release = None`
    /// drops the canonical clone here; if no other clones exist (the
    /// Primary→Inactive precondition under which we evict), that triggers
    /// Remove exactly once.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn reset_slot_collect_offer(
        g: &mut StoreInner<T>,
        id: BlockId,
    ) -> Option<(BlockHash, T, Arc<dyn OverflowSink<T>>)> {
        let slot = &mut g.slots[id];
        let old_hash = slot.hash;
        let payload = slot.payload.take();
        slot.state = SlotState::Reset;
        slot.hash = None;
        slot.refcount = 0;
        // Drop the canonical release handle. The Remove event fires from
        // `ReleaseInner::drop` once strong-count goes to zero.
        let _ = slot.release.take();

        if let (Some(hash), Some(payload)) = (old_hash, payload) {
            if let Some(sink) = g.overflow.as_ref().cloned() {
                return Some((hash, payload, sink));
            }
        }
        None
    }

    /// Commit a mutable block to active under `hash` + payload.
    ///
    /// Returns `(id, created, handle, events)`:
    ///
    /// - `id` — final slot id (may differ from the input on dedup).
    /// - `created` — `true` when the input slot was promoted to a new Primary
    ///   (caller's `MutableBlock` is now Primary); `false` on dedup hit.
    /// - `handle` — a clone of the canonical `EventReleaseHandle` to attach to
    ///   the resulting `ImmutableBlock`. Cloning (rather than minting fresh)
    ///   on dedup is what keeps "Remove fires exactly once per Primary".
    /// - `events` — `EventManager` for emitting Create. Returned out (rather
    ///   than mutated inline) so the caller can fire the event after dropping
    ///   the store mutex.
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn commit_block(
        &self,
        id: BlockId,
        hash: BlockHash,
        payload: T,
    ) -> (BlockId, bool, EventReleaseHandle, Arc<EventManager>) {
        let mut g = self.inner.lock().unwrap();
        // De-dup: if hash already active, drop ours.
        if let Some(&existing) = g.active_by_hash.get(&hash) {
            // Reset our slot back to free.
            let s = &mut g.slots[id];
            s.state = SlotState::Reset;
            s.hash = None;
            s.payload = None;
            s.refcount = 0;
            s.release = None;
            g.free.push_back(id);
            // Bump refcount on existing primary + clone its canonical handle.
            let prim = &mut g.slots[existing];
            prim.refcount += 1;
            let handle = prim
                .release
                .as_ref()
                .expect("Primary slot must hold a canonical release handle")
                .clone_for_dup();
            g.tinylfu.increment(hash);
            tracing::debug!(
                block_id = existing,
                block_hash = hash,
                "kvbm: register hit existing primary (dedup)"
            );
            let events = g.events.clone();
            return (existing, false, handle, events);
        }

        // Promote to Primary. Mint the canonical release handle (held by the
        // slot) + return one clone to the caller.
        let events = g.events.clone();
        let canonical = EventReleaseHandle::new(events.clone(), hash);
        let handle_for_caller = canonical.clone_for_dup();
        let slot = &mut g.slots[id];
        slot.state = SlotState::Primary;
        slot.hash = Some(hash);
        slot.payload = Some(payload);
        slot.refcount = 1;
        slot.release = Some(canonical);
        g.active_by_hash.insert(hash, id);
        g.tinylfu.increment(hash);

        // Drop guard before emitting Create event to keep the events channel
        // call outside the store lock.
        let tinylfu_count = g.tinylfu.estimate(hash);
        drop(g);
        let tier = 0; // brand new — coldest tier
        tracing::info!(
            event = "kvbm_block_create",
            block_id = id,
            block_hash = hash,
            tinylfu_count = tinylfu_count,
            inactive_tier = tier,
            "kvbm: block create"
        );
        events.emit(KvCacheEvent::Create(hash));
        (id, true, handle_for_caller, events)
    }

    /// Decrement refcount; if zero, demote to Inactive.
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn release_ref(&self, id: BlockId) {
        let mut g = self.inner.lock().unwrap();
        let slot = &mut g.slots[id];
        debug_assert!(
            slot.refcount > 0,
            "double release on slot {id} (state {:?})",
            slot.state
        );
        if slot.refcount == 0 {
            return;
        }
        slot.refcount -= 1;
        if slot.refcount == 0 && matches!(slot.state, SlotState::Primary | SlotState::Duplicate) {
            let hash = slot.hash.expect("primary has hash");
            slot.state = SlotState::Inactive;
            // Pull the hash out of the active map — Inactive entries are
            // looked up via the inactive index, not active_by_hash.
            g.active_by_hash.remove(&hash);
            g.inactive_by_hash.insert(hash, id);
            g.inactive.insert(hash);
            let count = g.tinylfu.estimate(hash);
            tracing::debug!(
                block_id = id,
                block_hash = hash,
                tinylfu_count = count,
                "kvbm: block → inactive"
            );
        }
    }

    /// Probe a single hash WITHOUT mutating state — no refcount bump, no
    /// resurrection, no LRU re-touch. Returns the slot id and its current
    /// state if the hash is known to the store. Used by `match_blocks` to
    /// decide the longest-prefix without prematurely committing.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn probe_one(g: &StoreInner<T>, hash: BlockHash) -> Option<(BlockId, SlotState)> {
        if let Some(&id) = g.active_by_hash.get(&hash) {
            return Some((id, g.slots[id].state));
        }
        if let Some(&id) = g.inactive_by_hash.get(&hash) {
            return Some((id, g.slots[id].state));
        }
        None
    }

    /// Confirm a previously-probed hit — bumps refcount, resurrects if needed,
    /// and returns an `ImmutableBlock` carrying the bumped refcount.
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn confirm_one(
        g: &mut StoreInner<T>,
        store: &BlockStore<T>,
        hash: BlockHash,
    ) -> Option<(BlockId, bool, ImmutableBlock<T>)> {
        if let Some(&id) = g.active_by_hash.get(&hash) {
            let slot = &mut g.slots[id];
            slot.refcount += 1;
            let handle = slot
                .release
                .as_ref()
                .expect("Primary slot must hold canonical handle")
                .clone_for_dup();
            g.tinylfu.increment(hash);
            return Some((
                id,
                false,
                ImmutableBlock {
                    store: store.clone(),
                    id,
                    hash,
                    release: Some(handle),
                },
            ));
        }
        if let Some(id) = g.inactive_by_hash.remove(&hash) {
            // Pull from the LRU index too.
            let _ = g.inactive.remove(hash);
            let events = g.events.clone();
            let canonical = EventReleaseHandle::new(events, hash);
            let caller_handle = canonical.clone_for_dup();
            let slot = &mut g.slots[id];
            slot.state = SlotState::Primary;
            slot.refcount = 1;
            slot.release = Some(canonical);
            g.active_by_hash.insert(hash, id);
            g.tinylfu.increment(hash);
            tracing::debug!(
                block_id = id,
                block_hash = hash,
                "kvbm: resurrected from inactive"
            );
            return Some((
                id,
                true,
                ImmutableBlock {
                    store: store.clone(),
                    id,
                    hash,
                    release: Some(caller_handle),
                },
            ));
        }
        None
    }

    /// Look up by hash. Active hit ⇒ bump refcount, return `Active`. Inactive
    /// hit ⇒ resurrect (state ← Primary, refcount=1, remove from inactive
    /// index), return `Resurrected`.
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn match_one(&self, hash: BlockHash) -> MatchOutcome<T> {
        let mut g = self.inner.lock().unwrap();
        match Self::confirm_one(&mut g, self, hash) {
            Some((id, false, imm)) => MatchOutcome::Active(id, imm),
            Some((id, true, imm)) => MatchOutcome::Resurrected(id, imm),
            None => MatchOutcome::Miss,
        }
    }

    /// Batched match. Returns one outcome per requested hash in input order.
    ///
    /// All work runs under a single store-lock critical section. Hits past a
    /// miss are NOT probed-then-confirmed wastefully: we still scan every
    /// hash since the caller asked for all of them, but each step is a
    /// state-mutating confirm (matches the reference and lets callers consume
    /// the returned `ImmutableBlock`s individually if they want per-hash
    /// fan-out).
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn match_blocks(&self, hashes: &[BlockHash]) -> Vec<MatchOutcome<T>> {
        // Single critical section — one lock acquired for the whole batch.
        let mut g = self.inner.lock().unwrap();
        let mut out = Vec::with_capacity(hashes.len());
        for &h in hashes {
            match Self::confirm_one(&mut g, self, h) {
                Some((id, false, imm)) => out.push(MatchOutcome::Active(id, imm)),
                Some((id, true, imm)) => out.push(MatchOutcome::Resurrected(id, imm)),
                None => out.push(MatchOutcome::Miss),
            }
        }
        out
    }

    /// Probe-only batched match: report which hashes are present (Active or
    /// resurrectable from Inactive) WITHOUT mutating state. Returns
    /// `(slot_id, was_inactive)` per hit, `None` per miss. Used by
    /// `scan_matches` to find the longest prefix before committing.
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn probe_blocks(
        &self,
        hashes: &[BlockHash],
    ) -> Vec<Option<(BlockId, bool /* was_inactive */)>> {
        let g = self.inner.lock().unwrap();
        hashes
            .iter()
            .map(|&h| {
                Self::probe_one(&g, h).map(|(id, state)| (id, matches!(state, SlotState::Inactive)))
            })
            .collect()
    }

    /// Confirm a prefix of `hashes` (`accept_n` items). Returns one
    /// `MatchOutcome` per confirmed entry — never a Miss in the returned
    /// vector. `hashes[..accept_n]` MUST all be hits as returned by a prior
    /// `probe_blocks` call; otherwise this returns Misses for the unexpected
    /// holes (and the caller's invariant is broken).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn confirm_prefix(
        &self,
        hashes: &[BlockHash],
        accept_n: usize,
    ) -> Vec<MatchOutcome<T>> {
        let mut g = self.inner.lock().unwrap();
        let mut out = Vec::with_capacity(accept_n);
        for &h in &hashes[..accept_n] {
            match Self::confirm_one(&mut g, self, h) {
                Some((id, false, imm)) => out.push(MatchOutcome::Active(id, imm)),
                Some((id, true, imm)) => out.push(MatchOutcome::Resurrected(id, imm)),
                None => out.push(MatchOutcome::Miss),
            }
        }
        out
    }

    /// Bump the TinyLFU frequency for `hash` without changing refcount or
    /// state. Used by integration to bias the eviction policy when a
    /// hit is observed outside the active set (e.g. on hydrate-from-SSD).
    ///
    /// If `hash` is currently sitting in the Inactive set, this also calls
    /// `inactive.touch(hash)` so the bin re-binds under the new count
    /// (otherwise the entry would stay in its old tier until the next
    /// `release_ref` / explicit touch).
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn touch_frequency(&self, hash: BlockHash) {
        let mut g = self.inner.lock().unwrap();
        g.tinylfu.increment(hash);
        if g.inactive_by_hash.contains_key(&hash) {
            g.inactive.touch(hash);
        }
    }

    /// Test/diag helper: peek slot state.
    #[cfg(test)]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn slot_state(&self, id: BlockId) -> Option<SlotState> {
        let g = self.inner.lock().unwrap();
        g.slots.get(id).map(|s| s.state)
    }

    /// Test/diag helper: peek slot refcount.
    #[cfg(test)]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn slot_refcount(&self, id: BlockId) -> Option<usize> {
        let g = self.inner.lock().unwrap();
        g.slots.get(id).map(|s| s.refcount)
    }

    /// Test/diag helper: list hashes currently active.
    #[cfg(test)]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn active_hashes(&self) -> Vec<BlockHash> {
        let g = self.inner.lock().unwrap();
        g.active_by_hash.keys().copied().collect()
    }
}

/// Snapshot of store-state. Returned by [`BlockStore::stats`].
#[non_exhaustive]
#[derive(Debug, Clone)]
/// A point-in-time snapshot of block store occupancy.
pub struct StoreStats {
    /// Total store capacity.
    pub capacity: usize,
    /// Number of free (unallocated) slots.
    pub free: usize,
    /// Number of active (in-use) slots.
    pub active: usize,
    /// Number of inactive (cached but not referenced) slots.
    pub inactive: usize,
    /// Occupancy of each TinyLFU inactive tier (4 levels: 0/1/2/3).
    pub inactive_tier_lens: [usize; 4],
}
