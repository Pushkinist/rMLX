//! Trait surface for the SSD-tier hydrate bridge.
//!
//! Migrated from `rmlx_models::prompt_cache` so per-arch entry types in
//! `rmlx-models` can implement it against [`crate::SsdHydrator`] without a
//! back-edge from `rmlx-kv-ssd` into `rmlx-models`. Promoted from
//! `pub(crate)` to `pub` to cross the new crate boundary.

use rmlx_core::error::Result;

/// Source that reconstructs a prompt-cache entry from the SSD tier
/// (`.kvb` + [`crate::SsdKvIndex`]) on a RAM-cache miss.
///
/// Symmetric to the prompt-cache `SpillSink`: where `SpillSink` persists an
/// evicted entry, `SsdHydrate` reads one back. Generic over the entry type so
/// the prompt cache stays arch-agnostic and the source is mockable in tests.
///
/// `hydrate` is given the request's full prompt token IDs. The implementation
/// queries the index for the longest matching block-hash prefix, reads the
/// `.kvb`, verifies its `model_id`/`kv_quant` metadata, and reconstructs the
/// arch entry. It returns:
/// - `Ok(Some(entry))` — an SSD hit; the cache promotes it into RAM.
/// - `Ok(None)` — a true SSD miss (no indexed prefix).
/// - `Err(_)` — never returned by the production impl: corruption
///   (bad read / metadata mismatch / missing file) is handled inside the impl
///   (delete file + index row, `warn!`) and surfaces as `Ok(None)` so the
///   caller falls through to a full prefill. The signature keeps `Result`
///   only so the impl can use `?` on the index calls and map any residual
///   error to `None`.
///
/// Must not panic.
pub trait SsdHydrate<E>: Send {
    /// Attempt to reconstruct an entry for `prompt_ids` from the SSD tier.
    fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<E>>;
}
