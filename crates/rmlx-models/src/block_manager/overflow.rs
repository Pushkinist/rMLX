//! Overflow sink trait.
//!
//! When the inactive index evicts a block from tier 0 (coldest), the store
//! offers the about-to-be-dropped payload to an `OverflowSink` for off-RAM
//! persistence. The intended production implementation wraps the existing
//! [`rmlx_kv_ssd::spill::SsdSpiller`] and translates the `layout_key`
//! into the spill job.
//!
//! Hydration on lookup miss is **out of scope** for (per-arch glue lives
//! in ). Only the offer-hook is provided here.

use super::{BlockHash, BlockMetadata};

/// Trait implemented by the SSD overflow tier glue. The block store calls
/// `offer_evicted` once per tier-0 evict; the implementation is expected to
/// be non-blocking (`try_send` semantics — drops on backpressure).
pub trait OverflowSink<T: BlockMetadata>: Send + Sync {
    /// Layout key threaded through to the SSD index for `(hash, layout_key)`
    /// composite PK. Stable for the lifetime of the sink.
    fn layout_key(&self) -> u64;

    /// Offer `payload` for the evicted `hash`. Implementation must NOT block
    /// the calling decode thread. Drops on backpressure are acceptable.
    fn offer_evicted(&self, hash: BlockHash, payload: T);
}

/// Test/stub sink that records every offer in a `Vec`. Useful for the Phase
/// 7 unit test that verifies the offer hook is invoked.
///
/// Gated behind `#[cfg(test)]` — this type is for unit tests only;
/// integration uses a real `SsdSpiller`-wrapping sink.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct RecordingSink<T: BlockMetadata + Clone> {
    pub(crate) layout_key: u64,
    pub(crate) offers: std::sync::Mutex<Vec<(BlockHash, T)>>,
}

#[cfg(test)]
impl<T: BlockMetadata + Clone> std::fmt::Debug for RecordingSink<T> {
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingSink")
            .field("layout_key", &self.layout_key)
            .field("n_offers", &self.offers.lock().unwrap().len())
            .finish()
    }
}

#[cfg(test)]
impl<T: BlockMetadata + Clone> RecordingSink<T> {
    pub(crate) fn new(layout_key: u64) -> Self {
        Self {
            layout_key,
            offers: std::sync::Mutex::new(Vec::new()),
        }
    }

    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(crate) fn offers(&self) -> Vec<(BlockHash, T)> {
        self.offers.lock().unwrap().clone()
    }
}

#[cfg(test)]
impl<T: BlockMetadata + Clone> OverflowSink<T> for RecordingSink<T> {
    fn layout_key(&self) -> u64 {
        self.layout_key
    }
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn offer_evicted(&self, hash: BlockHash, payload: T) {
        tracing::info!(
            event = "kvbm_overflow_offer",
            block_hash = hash,
            layout_key = self.layout_key,
            "kvbm: overflow sink received evicted block"
        );
        self.offers.lock().unwrap().push((hash, payload));
    }
}

#[cfg(test)]
#[path = "overflow_tests.rs"]
mod tests;
