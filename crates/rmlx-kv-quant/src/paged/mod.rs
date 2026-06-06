//! PagedAttention KV block table + page allocator.
//!
//! # Design overview
//!
//! vLLM-style paged KV allocation. Instead of a single contiguous buffer that
//! grows in 256-token pages (contiguous-growth path in `storage.rs`), this module maintains:
//!
//! 1. A **page pool** — a pre-allocated slab of N fixed-size page arrays. Each
//!    page holds `PAGE_TOKENS` tokens worth of quantized K or V data.
//! 2. A **block table** — a per-sequence `Vec<usize>` mapping logical page index
//!    to physical page ID in the pool.
//! 3. **Scatter / gather** — writes land into `pool[phys_id][token_slot]`; reads
//!    concatenate the active pages in order.
//!
//! For single-request decoding (current rMLX) the block table is monotonically
//! appended (no sharing, no eviction) and degenerates to contiguous
//! behaviour — same peak memory, same TPS. The value is in future N3
//! continuous-batching support: different requests can share a pool, return
//! pages on completion, and avoid per-request max-seq pre-allocation.
//!
//! # Submodules
//!
//! - `config`: feature flag, resolver functions, OnceLock globals.
//! - `alloc`: `PageSlab` — low-level page pool.
//! - `ops`: `PagedKStorage`, `PagedVStorage`, `PagedPlanarVStorage`.

mod alloc;
mod config;
mod ops;

pub use config::{install_paged_kv, resolve_paged_kv, resolve_paged_kv_page_tokens};
pub use config::{paged_kv_enabled, paged_kv_page_tokens};
pub use ops::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};
