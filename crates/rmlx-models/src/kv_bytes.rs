//! Per-model-instance KV-cache byte accounting.
//!
//! One model instance owns one [`KvBytesCounter`]. Its generate paths write the
//! resident KV byte total there at the end of every generation, and readers go
//! through [`crate::arch::Architecture::kv_cache_bytes_sample`], which names the
//! instance it is reading.
//!
//! ## Why the counter is per instance
//!
//! It used to be one static per arch *type*. Two models of the same
//! architecture resident at once — the multi-model registry, or a speculative
//! draft/verifier pair that happens to share an arch — then wrote the same
//! location, and a reader bracketing a generation on model A would see model
//! B's store advance the sequence it was watching. The `seq` check would pass
//! and hand back B's byte count labelled as A's. That figure goes into the
//! append-only `events` table, where a wrong row is permanent (see
//! `docs/METRICS_DB.md` §8.1.1).
//!
//! Serialisation of GPU admission hid it rather than fixed it: the server's
//! `gpu_queue` is a one-permit semaphore held across the blocking closure, so
//! two generations do not overlap *today*. Anything that relaxes admission —
//! batching, a second queue, another in-process consumer — brings the
//! cross-attribution straight back. Owning the counter per instance removes the
//! shared location instead of relying on nobody reaching it at the same time.

use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// KvBytesSample
// ---------------------------------------------------------------------------

/// A read of one model instance's last-generation KV-cache byte total, tagged
/// with the store sequence it was written at.
///
/// The bare byte count is ambiguous: `0` is both the never-written initialiser
/// and a legal (if suspicious) reading, and a non-zero value read after a
/// generation that never reported one is the *previous* generation's figure.
/// Callers that record the number as a measurement need to tell those apart, so
/// they sample the pair before and after the generation and require `seq` to
/// have advanced. See [`crate::arch::Architecture::kv_cache_bytes_sample`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvBytesSample {
    /// Byte total recorded by the most recent store on this model instance.
    /// Meaningless unless `seq > 0`.
    pub bytes: u64,
    /// Number of stores on this model instance since it was loaded.
    /// `0` means no generation on it has reported a figure.
    pub seq: u64,
}

/// What a pair of [`KvBytesSample`] reads taken around one generation says
/// about the byte count that generation produced.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed by construction: the sequence either advanced or it did not, and if it \
              did the byte count is either zero or not. There is no fourth state, and a \
              wildcard arm at the call sites would defeat the purpose of the distinction"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBytesVerdict {
    /// The generation reported a non-zero byte count. Usable.
    Reported(u64),
    /// The generation reported nothing: the readable value belongs to an
    /// earlier generation, or is the never-written initialiser. This is the
    /// state that a bare `-> u64` accessor collapses into "0" or, worse, into
    /// the previous run's plausible-looking figure.
    Unreported,
    /// The generation did report, and reported zero bytes. Distinct from
    /// [`KvBytesVerdict::Unreported`]: the plumbing works and the answer is
    /// still not usable as a resident-KV measurement after a real prefill.
    ReportedZero,
}

/// Classify the byte count a generation produced, from the store sequence
/// observed before and after it.
///
/// Detection ("did this generation report?") is decided by the sequence, and
/// only then is the value interpreted. Collapsing the two — treating a zero, or
/// an unchanged sequence, as "no KV" — is what silently records one run's
/// number under another run's label.
///
/// Both samples must come from the **same model instance**; a sequence is only
/// comparable with itself.
///
/// Every caller that *records* or *reports* a KV-byte figure as a measurement
/// goes through this. A caller that only displays the last-known value
/// (`/metrics/cache`) may read the bare count.
#[must_use]
pub const fn classify_kv_bytes(before: KvBytesSample, after: KvBytesSample) -> KvBytesVerdict {
    if after.seq <= before.seq {
        return KvBytesVerdict::Unreported;
    }
    if after.bytes == 0 {
        return KvBytesVerdict::ReportedZero;
    }
    KvBytesVerdict::Reported(after.bytes)
}

// ---------------------------------------------------------------------------
// KvBytesCounter
// ---------------------------------------------------------------------------

/// The resident-KV byte total of the last generation on **one model instance**,
/// paired with a monotonic store sequence.
///
/// Held as a field on each arch's model struct, so a reader that has the model
/// has the counter, and no other model can advance it. `&self` stores keep it
/// usable from the generate paths, which only ever borrow the model.
#[derive(Debug, Default)]
pub struct KvBytesCounter {
    bytes: AtomicU64,
    seq: AtomicU64,
}

impl KvBytesCounter {
    /// A counter that has never been written.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: AtomicU64::new(0),
            seq: AtomicU64::new(0),
        }
    }

    /// Read the byte total paired with the sequence it was written at.
    ///
    /// `seq == 0` means no generation on this model has reported a figure yet,
    /// so `bytes` is the `0` initialiser and not a measurement.
    #[must_use]
    pub fn sample(&self) -> KvBytesSample {
        // Load the sequence FIRST, and with Acquire. The writer stores `bytes`
        // before releasing the sequence, so observing sequence `k` guarantees
        // the byte count for generation `k` is already visible: the byte load
        // that follows returns generation `k`'s figure or a later one. The
        // pairing invariant is therefore "bytes is never *older* than seq",
        // which is the direction that matters — a caller comparing sequences
        // can refuse a reading that was in fact valid, but can never accept a
        // stale byte count as a fresh generation's measurement.
        //
        // The opposite order does not hold: an Acquire load only orders the
        // accesses that follow it, so reading bytes first leaves that load
        // outside the release/acquire edge (and free to sink below it on
        // aarch64). The reader could then pair generation `k-1`'s bytes with
        // sequence `k` — exactly the stale-value-under-a-fresh-label failure
        // `KvBytesSample` exists to make impossible.
        //
        // No seqlock needed: the payload is a single `u64`, which cannot tear.
        let seq = self.seq.load(Ordering::Acquire);
        let bytes = self.bytes.load(Ordering::Relaxed);
        KvBytesSample { bytes, seq }
    }

    /// Record the KV-cache bytes for the generation that just finished on this
    /// model.
    ///
    /// The `PostDecode` witness pins the sample to a single lifecycle point:
    /// after the decode loop, when every resident KV allocation (incl. the
    /// decode-time ring) exists. It is minted only by the decode loops, so a
    /// caller cannot record `kv_cache_bytes` at the prefill snapshot — a
    /// pre-decode number would silently omit the ring on ring-backed codecs.
    ///
    /// Also emits the `kv_bytes` event. This is the one place it is emitted:
    /// `n` is already summed over every cache by the caller, so the event costs
    /// nothing extra. Emitting it per-layer per-decode-step instead would call
    /// `KvCache::resident_bytes` — which walks a block list that grows by one
    /// entry per decode step — making a generation quadratic in context for the
    /// sake of a diagnostic.
    pub(crate) fn store(&self, n: u64, _post: crate::decode_loop::PostDecode) {
        tracing::debug!(kv_bytes = n, "kv cache bytes");
        self.bytes.store(n, Ordering::Relaxed);
        // Release-ordered so a reader that observes the bumped sequence also
        // observes the byte count that goes with it.
        self.seq.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
#[path = "kv_bytes_tests.rs"]
mod kv_bytes_tests;
