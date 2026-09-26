//! Mixed / rotated-K KV update path.
//!
//! Holds the two entries of [`crate::storage::KvStorage::Mixed`]: the prefill
//! bulk-encode into [`crate::mixed_quant::MixedKvState`], and the decode entry
//! that refuses a direct [`super::KvCache::update`]. The per-step append is
//! [`super::KvCache::update_and_sdpa_mixed`] in [`super::sdpa`]. The helpers
//! with more than one family caller stay in [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::KvStorage;

use super::update::storage_mismatch;
use super::KvCache;

impl KvCache {
    /// The decode entry of a Mixed cache. It refuses: the append must go
    /// through `update_and_sdpa`.
    #[allow(
        clippy::unused_self,
        reason = "an update entry: it has the signature every storage variant's entry has"
    )]
    pub(crate) fn update_mixed(
        &mut self,
        _new_k: &Array,
        _new_v: &Array,
        _device: Device,
    ) -> Result<(Array, Array)> {
        Err(Error::Mlx(
            "Contract violation: KvCache::update called on a Mixed cache. \
                 These caches MUST be driven through KvCache::update_and_sdpa (universal \
                 wrapper). Direct update() bypasses the quantized SDPA and leaves the cache \
                 in an inconsistent state."
                .into(),
        ))
    }

    // Mixed cache uses the fp16 prefill_raw scaffolding. Bulk-quantize
    // the accumulated fp16 K/V (total_seq tokens) into the Mixed state
    // directly (skips the zero-alloc + 6×slice_update round-trip that
    // the per-token path pays on large prefill prefixes).
    pub(crate) fn exit_prefill_mixed(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill Mixed/RotK: bulk-quantizing fp16 prefill K/V"
        );
        let policy = self.policy;
        let KvStorage::Mixed { state, .. } = &mut self.storage else {
            return Err(storage_mismatch("Mixed", &self.storage));
        };
        // Reset so bulk_init_from_fp16 starts clean. reset() preserves
        // `k_rotation` so RotK keeps rotating K post-prefill.
        state.reset();
        // Directly quantize and store — no zero-buffer alloc, no
        // write_at, no slice_seq_to. offset is set to total_seq.
        state.bulk_init_from_fp16(k_full, v_full, device, policy)?;
        // The compact fp16 seed for warm-TTFT decode is the same buffer
        // the caller already materialised (decode_fp16_pair).
        Ok(())
    }
}
