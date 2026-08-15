//! Trait surface for the SSD-tier hydrate bridge.
//!
//! Migrated from `rmlx_models::prompt_cache` so per-arch entry types in
//! `rmlx-models` can implement it against [`crate::SsdHydrator`] without a
//! back-edge from `rmlx-kv-ssd` into `rmlx-models`. Promoted from
//! `pub(crate)` to `pub` to cross the new crate boundary.

use rmlx_core::error::Result;
use rmlx_kv_quant::KvQuant;

/// Source that reconstructs a prompt-cache entry from the SSD tier
/// (`.kvb` + [`crate::SsdKvIndex`]) on a RAM-cache miss.
///
/// Symmetric to the prompt-cache `SpillSink`: where `SpillSink` persists an
/// evicted entry, `SsdHydrate` reads one back. Generic over the entry type so
/// the prompt cache stays arch-agnostic and the source is mockable in tests.
///
/// `hydrate` is given the request's full prompt token IDs, plus the two facts
/// that identify what the request is asking for: the `seed` the RAM cache is
/// querying under and the `kv_quant` it is running. **Both are per-request and
/// must be passed, never read off the source.** A hydrate source is installed
/// once per arch and outlives the model that installed it — several models of
/// one architecture can be resident at a time, and the KV codec is
/// per-request — so any value the source remembers from its own construction
/// belongs to whichever model attached last, not to the request in hand.
/// Seeding the probe from such a value is how the tier silently stops hitting.
///
/// The implementation queries the index for the longest matching block-hash
/// prefix, reads the `.kvb`, verifies its `model_id`/`kv_quant` metadata, and
/// reconstructs the arch entry. It returns:
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
    /// Attempt to reconstruct an entry for `prompt_ids` from the SSD tier
    /// under the requesting model's `seed` and the request's `kv_quant`.
    fn hydrate(&self, prompt_ids: &[u32], seed: u64, kv_quant: KvQuant) -> Result<Option<E>>;
}
