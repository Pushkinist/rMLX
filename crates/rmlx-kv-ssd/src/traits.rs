//! Trait surface for the SSD-tier hydrate bridge.
//!
//! Two traits and one impl. [`SsdHydrate`] is what the prompt cache calls on a
//! RAM miss. [`HydratedEntry`] is what an arch entry states about itself, and
//! the blanket `impl<E: HydratedEntry> SsdHydrate<E> for SsdHydrator` is the
//! only implementation of the probe for production entries.
//!
//! Both live here rather than in `rmlx_models::prompt_cache` so per-arch entry
//! types in `rmlx-models` can implement against [`crate::SsdHydrator`] without
//! a back-edge from `rmlx-kv-ssd` into `rmlx-models`.

use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::KvQuant;

use crate::hydrate::{HydratedBlock, SsdHydrator};

/// Source that reconstructs a prompt-cache entry from the SSD tier
/// (`.kvb` + [`crate::SsdKvIndex`]) on a RAM-cache miss.
///
/// Symmetric to the prompt-cache `SpillSink`: where `SpillSink` persists an
/// evicted entry, `SsdHydrate` reads one back. Generic over the entry type so
/// the prompt cache stays arch-agnostic and the source is mockable in tests.
///
/// `hydrate` is given the request's full prompt token IDs, plus the three facts
/// that identify what the request is asking for: the `seed` the RAM cache is
/// querying under, the `kv_quant` it is running, and the `DispatchPolicy` its
/// caches dispatch under. **All three are per-request and must be passed,
/// never read off the source.** A hydrate source is installed
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
/// - `Err(_)` — a caller-contract error: a `layer_quants` whose length is not
///   the block's layer count. The block is kept. Corruption (bad read /
///   metadata mismatch / missing file) is not an `Err`: it is handled inside
///   the impl (delete file + index row, `warn!`) and surfaces as `Ok(None)` so
///   the caller falls through to a full prefill.
///
/// Must not panic.
pub trait SsdHydrate<E>: Send {
    /// Attempt to reconstruct an entry for `prompt_ids` from the SSD tier
    /// under the requesting model's `seed`, the request's `kv_quant`, the codec
    /// the arch builder gives each layer at that `kv_quant` (`layer_quants`),
    /// and the `policy` its caches dispatch under.
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        layer_quants: &[KvQuant],
        policy: DispatchPolicy,
    ) -> Result<Option<E>>;
}

/// A prompt-cache entry the SSD tier can rebuild from one stored block.
///
/// Every arch entry implements this, the hybrid one included, and the blanket
/// [`SsdHydrate`] impl below is the only implementation of the probe. What an
/// arch states here is what its entry does with a block the tier restored, and
/// nothing else: the probe, the corruption handling and the hit/miss branch are
/// the same work for every arch.
///
/// The trait lives here rather than in `rmlx-models` because the blanket impl
/// has to. Both [`SsdHydrate`] and [`SsdHydrator`] are foreign to `rmlx-models`
/// and the type parameter is uncovered there, so the orphan rule rejects the
/// shape (`E0210`). An arch entry is a local type in its own crate, so its impl
/// of this trait is allowed.
///
/// `pub` and unsealed. No member crate of this workspace is published, so an
/// unsealed public blanket carries no semver obligation to an outside
/// implementor. It is an internal bridge, the status [`HydratedBlock`] already
/// records for itself.
pub trait HydratedEntry: Sized {
    /// The arch's cross-layer-KV topology — its `SHARES_KV_ACROSS_LAYERS`.
    ///
    /// It lands on every restored `KvCache`, and a hydrated cache can be
    /// tail-extended, which re-runs the `exit_prefill` gate this flag decides.
    /// A hard-coded `false` drops a bf16 mirror a sharing arch reads; a
    /// hard-coded `true` builds one no other arch reads. The constant is what
    /// keeps the shared body from choosing either.
    const SHARES_KV: bool;

    /// Build the entry from the block the tier restored.
    ///
    /// `block_hashes` is the chained block-digest stream of the block's own
    /// tokens, computed under the seed the probe ran, so the promoted entry is
    /// findable by the query that triggered the hydrate. `kv_quant` is the
    /// codec the request is running.
    ///
    /// An entry built here holds the block-aligned prefix only, so it must be
    /// flagged as SSD-hydrated: its first decode token is a placeholder, and
    /// the consume engine has to exclude it from the exact fast path.
    fn from_hydrated(block: HydratedBlock, block_hashes: Vec<u64>, kv_quant: KvQuant) -> Self;
}

impl<E: HydratedEntry> SsdHydrate<E> for SsdHydrator {
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        layer_quants: &[KvQuant],
        policy: DispatchPolicy,
    ) -> Result<Option<E>> {
        let Some((block, block_hashes)) = self.lookup_seeded(
            prompt_ids,
            seed,
            kv_quant,
            layer_quants,
            policy,
            E::SHARES_KV,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(E::from_hydrated(block, block_hashes, kv_quant)))
    }
}
