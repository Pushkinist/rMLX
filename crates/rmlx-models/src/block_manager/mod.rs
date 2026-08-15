//! KVBM block manager (logical layer). Per-arch integration handled by the per-arch glue.
//!
//! This module is **unwired** in the initial commit: no production code path
//! (engine, prefill, decode, per-arch `PromptCache`) calls into it yet.
//! The per-arch swap follows once the per-arch glue wires it in.
//!
//! Reference: NVIDIA Dynamo `lib/kvbm-logical/` (Apache-2.0). Local research
//! A local research summary covers the KVBM logical layer design.
//!
//! ## Layered phases
//!
//! 1. [`tinylfu`] — 4-bit Count-Min Sketch + halving decay.
//! 2. [`multi_lru`] — 4-tier LRU keyed by TinyLFU bin.
//! 3. [`store`] — single-mutex `BlockStore` + slot state machine.
//! 4. [`events`] — `KvCacheEvent` + RAII `EventReleaseHandle` +
//!    `PowerOfTwoPolicy`.
//! 5. [`manager`] — public `BlockManager` facade (`allocate_blocks` /
//!    `register_blocks` / `match_blocks` / `scan_matches`).
//! 6. **Skipped here, owned by the per-arch glue** — per-arch `PromptCache` swap. Files
//!    under `crates/rmlx-models/src/{gemma4,qwen3_5_moe,qwen3}/prompt_cache.rs`
//!    are intentionally untouched.
//! 7. [`overflow`] — `OverflowSink` trait for the existing SSD spiller
//!    (`kv_cache::spill::SsdSpiller`). Hydration on lookup-miss is NOT
//!    plumbed here — the per-arch glue owns that.
//!
//! ## Hash family
//!
//! Block hashing keeps the project-wide FNV-1a-64 family
//! (see `prompt_cache::chained_block_hashes_seeded`), but not its key:
//! `layout_key` is mixed via `FNV_OFFSET ^ layout_key` with no model or codec
//! term, so these digests are a separate address space from the prompt cache's
//! `cache_seed` and cannot address `.kvb` rows. The reference uses xxh3 with 4
//! distinct 192-byte secrets for the TinyLFU CMS; we instead derive 4
//! independent FNV streams from 4 stable u64 seeds — same algorithmic
//! properties (4 independent counter slots, 4-bit ceiling, halving decay) and
//! avoids adding `xxhash-rust` as a new workspace dependency.
//!
//! ## Lock order
//!
//! `attachments → store, never reverse.` The store mutex is never held while
//! the manager calls into an `OverflowSink` callback (which itself is
//! documented as non-blocking `try_send`). See [`store::BlockStore`] docs.
//!
//! ## Compile-only today
//!
//! Items are exposed `pub` so can build on top, but the upstream
//! crates (`rmlx-runtime`, `rmlx-server`) do not yet import them. The unit
//! tests in this module exercise the full state machine + eviction policy +
//! event lifecycle.

mod hash;

pub mod events;
pub mod manager;
pub mod multi_lru;
pub mod overflow;
pub mod store;
pub mod tinylfu;

/// Block identifier — a dense index into the store's slot vector. Matches the
/// reference `BlockId = usize`.
pub type BlockId = usize;

/// Content hash of one logical block. The reference uses `SequenceHash = u128`
/// (a `PositionalLineageHash`). rMLX persists FNV-1a-64 to `.kvb` files
/// today, so the block-manager API stays on `u64`; widening is a follow-up
/// in + if the radix-tree registry ports.
pub type BlockHash = u64;

/// Marker trait the per-arch payload must implement.
///
/// Today this is a minimal `Send + Sync + 'static`. The follow-up integration
/// adds `kv_bytes` / `truncate_kv_to_block` per arch as needed.
pub trait BlockMetadata: Send + Sync + 'static {}

pub use events::{
    EventManager, EventReleaseHandle, EventSubscriber, KvCacheEvent, PowerOfTwoPolicy,
};
pub use manager::{BlockManager, BlockManagerConfig, PrefixMatch};
pub use overflow::OverflowSink;
pub use store::{ImmutableBlock, MatchOutcome, MutableBlock, StoreError, StoreStats};
