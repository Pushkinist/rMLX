//! Two-phase sparse attention MSL kernels.
//!
//! # What this module is
//!
//! Two Metal kernels that implement the multi-turboquant two-phase sparse
//! attention dispatch on top of the rMLX PlanarQuant K/V codec (NOT the mtq
//! `mtq` codec — see CLAUDE.md hard rules).
//!
//! * [`phase1_score_msl::phase1_score`] — Phase 1: per-tile QK scoring over
//!   packed PlanarQuant K.  Each `(B, H, tile)` threadgroup computes raw QK
//!   scores for every token in the tile, writes per-tile top-4 (score, index),
//!   and writes all scores to a flat buffer for Phase 2.
//! * [`phase2_sparse_attend_msl::phase2_sparse_attend`] — Phase 2: per-tile
//!   sparse attend.  Inputs per-head threshold (CPU-computed from Phase 1
//!   tile tops) and uses the all-scores buffer to skip below-threshold
//!   tokens; survivors contribute to an online-softmax + V accumulate.
//!   The kernel emits per-tile partial outputs + LSE state which the
//!   LSE-merge kernel collapses into the final `[B, H, head_dim]`
//!   output.
//!
//! Reference (Python / mtq codec):
//! `multi-turboquant/multi_turboquant/kernels/metal/fused_attention.py`
//! `PHASE1_SCORE_KERNEL` + `PHASE2_SPARSE_ATTEND_KERNEL`.
//!
//! # Why two phases
//!
//! At long contexts only a small fraction of tokens contribute meaningfully
//! to the softmax mass (95% rule).  Two-phase splits the work into:
//!
//! 1. A cheap scoring pass (K decode + QK only) that materialises a
//!    per-tile top-4 summary.
//! 2. A short CPU/MLX reduction over the top-4 to derive a per-head
//!    threshold.
//! 3. A costly attend pass (K + V decode + softmax + SV) that runs only
//!    on tiles with at least one above-threshold token.
//!
//! At top-1024 of 50K tokens the typical (~5/196) tile-level survival rate
//! is enough to recover the dense-attention output to ≥0.99 cosine while
//! cutting V bandwidth by ~95%.
//!
//! # Output contract & LSE merge
//!
//! Phase 2 emits per-tile `partial_o`, `tile_max`, `tile_sum_exp` in the
//! same layout as [`crate::planar_flash_decode_msl`] P1 — the existing
//! P2 LSE-merge kernel collapses them into the final output, so this
//! module does not duplicate the P2 LSE-merge logic.
//!
//! # Single-MLX claim
//!
//! Per CLAUDE.md "Single MLX process per Mac", callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

pub mod phase1_score_msl;
pub mod phase2_sparse_attend_msl;

/// Aggregated dispatch counter across the two-phase sparse-attention kernels.
///
/// Returns the process-lifetime sum of phase-1 ([`phase1_score_msl::phase1_score_dispatch_count`])
/// and phase-2 ([`phase2_sparse_attend_msl::phase2_sparse_attend_dispatch_count`])
/// invocations that reached the Metal enqueue point.
///
/// Used by integration tests and the `rmlx metrics …` surface to prove that a
/// production call path actually routed through the sparse-attention dispatch.
/// On a successful dispatch the aggregator increments by 2 per
/// `sparse_attn_dispatch` call (one P1 + one P2); when phase-2 short-circuits
/// (all tiles below threshold; the seedless-cache common case at top-K of
/// large kv_seq) the counter still grows by 2 because both kernels enqueue
/// before the early-exit shortcut inside P2 fires.
///
/// Mirrors the aggregator pattern from
/// [`crate::kvcache::fused_qk_total_dispatch_count`].
///
/// # Atomic coherence note
///
/// P1 and P2 counters are loaded with `Relaxed` ordering — concurrent
/// dispatches between the two loads may produce a sum that does not
/// correspond to any single moment in time.  Use before/after deltas in
/// a single-threaded test context (where no concurrent dispatch can race
/// the two loads) to get a meaningful measurement.
#[must_use]
pub fn sparse_attn_total_dispatch_count() -> u64 {
    phase1_score_msl::phase1_score_dispatch_count()
        + phase2_sparse_attend_msl::phase2_sparse_attend_dispatch_count()
}

#[cfg(test)]
#[path = "sparse_attn_tests.rs"]
mod tests;
