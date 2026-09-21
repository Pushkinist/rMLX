//! Mixed / rotated-K KV update path.
//!
//! Holds the update-side body that only [`crate::storage::KvStorage::Mixed`]
//! uses: the prefill bulk-encode into [`crate::mixed_quant::MixedKvState`].
//! There is no decode body here — [`super::KvCache::update`] refuses a Mixed
//! cache outright, and the per-step append is
//! [`super::KvCache::update_and_sdpa_mixed`] in [`super::sdpa`]. The
//! `KvStorage` dispatch and the helpers with more than one family caller stay
//! in [`super::update`].

use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device};

use crate::storage::KvStorage;

use super::KvCache;

impl KvCache {
    // Mixed cache uses the fp16 prefill_raw scaffolding. Bulk-quantize
    // the accumulated fp16 K/V (total_seq tokens) into the Mixed state
    // directly (skips the zero-alloc + 6×slice_update round-trip that
    // the per-token path pays on large prefill prefixes).
    #[allow(
        clippy::unreachable,
        reason = "the storage variant is fixed by the `match self.quant` that selects this arm; a mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the arm reads one storage variant; every other is the same construction-time mismatch and needs no per-variant spelling"
    )]
    pub(super) fn exit_prefill_mixed(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
        policy: DispatchPolicy,
    ) -> Result<()> {
        tracing::debug!(
            total_seq,
            "exit_prefill Mixed/RotK: bulk-quantizing fp16 prefill K/V"
        );
        let state = match &mut self.storage {
            KvStorage::Mixed { state, .. } => state,
            _ => unreachable!("Mixed-path quant but storage is not Mixed"),
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
