//! BlockStore — single-mutex unified store + slot state machine.
//!
//! Reference: `dynamo/lib/kvbm-logical/src/pools/store.rs`.
//!
//! ## Single-mutex assumption
//!
//! rMLX decode is single-threaded today; the store guards every mutation with
//! one mutex which is held only for short pure-CPU work. If continuous
//! batching arrives ( follow-ups) the mutex may need finer-grained
//! splitting. Document drift here.
//!
//! ## Lock order
//!
//! `attachments → store, never reverse.` The block manager facade does NOT
//! expose `&BlockStore` through any handle that already holds an attachment
//! lock; instead it exposes thin API methods that take the store lock
//! internally. This is enforced by keeping `inner` private and only handing
//! out copy-out values (e.g. `MatchResult`).
//!
//! ## Slot lifecycle
//!
//! ```text
//! Reset ──allocate──> Mutable ──register──> Staged ──commit──> Primary
//! │
//! └─dup─> Duplicate
//! Primary ──refcount=0──> Inactive ──reuse──> Primary
//! │
//! └──evict──> Reset (offered to SSD)
//! ```

mod core;
mod handles;
mod slot;

pub use core::{BlockStore, MatchOutcome, StoreError, StoreStats};
pub use handles::{ImmutableBlock, MutableBlock};
pub use slot::SlotState;

#[cfg(test)]
mod tests;
