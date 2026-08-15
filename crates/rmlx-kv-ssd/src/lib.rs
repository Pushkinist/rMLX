#![warn(missing_docs)]
//! SSD-tier KV cache (extracted from `rmlx-models`).
//!
//! Owns the entire SSD KV story:
//!
//! - [`ssd_index`] — SQLite index of on-disk `.kvb` blocks (`SsdKvIndex`,
//!   schema v3, LRU eviction, layout-key column).
//! - [`block_io`] — `KvBlockWriter` / `KvBlockReader` (safetensors record
//!   format per `KvStorage` variant) — the authoritative dispatch for
//!   Contract B (every codec adds one match arm here).
//! - [`spill`] — `SsdSpiller` + `SpillJob` + bounded-channel drain thread.
//! - [`hydrate`] — `SsdHydrator` for on-demand RAM-miss reload.
//! - [`ssd_tier`] — `install_config`, `attach_at_load`, `compute_layout_key`,
//!   pre-release v1 wipe.
//! - [`hooks`] — process-global Prometheus hook setters + SSD event recorder.
//! - [`traits::SsdHydrate`] — single trait arch entries implement; bridges the
//!   prompt cache to the SSD hydrator.
//! - [`hashing`] — chained FNV-1a-64 block-digest helpers + `BLOCK_TOKENS`
//!   constant shared between `prompt_cache` (RAM) and the SSD tier (disk).
//!
//! The crate has no back-edge into `rmlx-models`. The hashing primitives
//! migrated here; `prompt_cache.rs` (in `rmlx-models`) imports them via a
//! `pub(crate) use rmlx_kv_ssd::{…}` named import — keeping all in-crate
//! `crate::prompt_cache::FNV_OFFSET` call sites unchanged.

#![cfg_attr(
    test,
    allow(
        clippy::panic,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::ignore_without_reason,
    )
)]

pub mod block_io;
pub mod hashing;
pub mod hooks;
pub mod hydrate;
pub mod spill;
pub mod ssd_index;
pub mod ssd_tier;
pub mod traits;

// Top-level re-export surface — call sites in `rmlx-server`, `rmlx-cli`,
// `rmlx-models/tests`, etc. import these symbols directly from `rmlx_kv_ssd`
// (the `rmlx_models::kv_cache::*` re-export shim was dropped).

pub use block_io::{write_caches, BlockIoError, KvBlockReader, KvBlockWriter};
pub use hashing::{
    chained_block_hashes, chained_block_hashes_seeded, BLOCK_TOKENS, FNV_OFFSET, FNV_PRIME,
};
pub use hooks::{
    call_ssd_bytes_used_hook, call_ssd_evict_total_hook, call_ssd_hydrate_prom_hook,
    call_ssd_spill_prom_hook, set_ssd_bytes_used_hook, set_ssd_event_recorder,
    set_ssd_evict_total_hook, set_ssd_hydrate_prom_hook, set_ssd_spill_prom_hook,
    ssd_event_recorder, BytesUsedHook, PromHook,
};
pub use hydrate::{HydratedBlock, SsdHydrator};
pub use spill::{SpillJob, SsdSpiller};
pub use ssd_index::{hash_to_hex, SsdKvIndex};
pub use ssd_tier::{
    active as active_ssd_tier_config, compute_layout_key, install_config, prepare_attach,
    AttachInfo, SsdTierConfig,
};
pub use traits::SsdHydrate;
