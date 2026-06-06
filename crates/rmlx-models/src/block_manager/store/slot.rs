//! `BlockSlot` — per-slot state machine node for the block store.

#![allow(clippy::missing_fields_in_debug, clippy::option_as_ref_cloned)]

use super::super::events::EventReleaseHandle;
use super::super::{BlockHash, BlockId, BlockMetadata};

/// Slot states (see module-level lifecycle diagram).
///
/// `Staged` and `Duplicate` are part of the documented lifecycle but the
/// commit only exercises the Reset ↔ Mutable ↔ Primary ↔ Inactive
/// path. will activate Staged (multi-block atomic commit) and
/// Duplicate (BlockDuplicationPolicy::Allow). Keep them in the enum so the
/// state machine matches the reference and so external arch glue can switch
/// over without API churn.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed state-machine enum — four slot states; adding a state requires updating all transition logic in StoreInner"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SlotState {
    /// Free, in the reset pool.
    Reset,
    /// Allocated, no hash assigned yet — writable.
    Mutable,
    /// Hash assigned but not yet committed to `active_by_hash` — staging.
    Staged,
    /// Committed to `active_by_hash`, live references outstanding.
    Primary,
    /// Same hash as a Primary — a duplicate registration. Lives parallel
    /// until refcount drops, then becomes Inactive.
    Duplicate,
    /// Refcount dropped to 0; sitting in the inactive index, eligible for
    /// resurrection (reuse) or eviction.
    Inactive,
}

/// One block slot. Carries the `BlockMetadata` payload until evicted.
#[derive(Debug)]
pub(crate) struct BlockSlot<T: BlockMetadata> {
    /// Slot id — duplicates the index in `slots`. Kept around for invariant
    /// checks (`debug_assert!(slot.id == index)`) and for `Debug` output.
    #[allow(dead_code)]
    pub(crate) id: BlockId,
    pub(crate) state: SlotState,
    pub(crate) hash: Option<BlockHash>,
    /// Refcount on live `ImmutableBlock`s pointing at this slot.
    pub(crate) refcount: usize,
    pub(crate) payload: Option<T>,
    /// Canonical lifecycle handle for this Primary. Mints when the slot first
    /// transitions to Primary (fresh-commit or resurrect). Cloned via
    /// `clone_for_dup` for every additional reference (dedup, match-hit,
    /// `ImmutableBlock::clone`). When the last clone drops, the inner Arc
    /// strong-count goes to zero and `ReleaseInner::drop` fires Remove.
    ///
    /// `None` while the slot is Reset / Mutable / Inactive.
    pub(crate) release: Option<EventReleaseHandle>,
}

impl<T: BlockMetadata> BlockSlot<T> {
    pub(super) fn fresh(id: BlockId) -> Self {
        Self {
            id,
            state: SlotState::Reset,
            hash: None,
            refcount: 0,
            payload: None,
            release: None,
        }
    }
}
