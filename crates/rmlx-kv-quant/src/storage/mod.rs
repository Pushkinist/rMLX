// Promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill — which stay in `rmlx-models`) can still
// reach them across the crate boundary. Doc/visibility warnings on the
// promoted surface are silenced; the API is otherwise unchanged.
#![allow(missing_docs, missing_debug_implementations, unreachable_pub)]
//! Quantized KV buffer types: `QuantK`, `QuantV`, `QuantPlanarV`, and `KvStorage`.
//!
//! Each struct wraps the on-GPU (or paged-GPU) storage for one axis of the
//! KV cache under a specific quantization scheme. `KvStorage` is the
//! top-level enum that [`KvCache`][super::kvcache::KvCache] holds.
//!
//! # Storage types
//!
//! - [`QuantK`] — quantized K buffers (q8_0 or rot-K affine-8-bit).
//! - [`QuantV`] — quantized V buffers (TurboQuant V4 or q8_0).
//! - [`QuantPlanarV`] — quantized V buffers (PlanarQuant V4).
//! - [`KvStorage`] — enum that selects the active K/V codec and holds the
//!   matching buffer pair.
//!
//! # Paged growth
//!
//! GPU buffers are allocated in multiples of `KV_PAGE_SIZE` (256) tokens.
//! On every `append`, if the filled sequence would exceed the current
//! allocation, the buffer grows by one page (reallocate + prefix copy).
//!
//! # See also
//!
//! - [`super::q8`] — CPU-side q8_0 encode/decode.
//! - `docs/KV_CACHE.md` — subsystem spec and codec matrix.
// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for quantized scale arrays
#![allow(unsafe_code)]

mod kv_storage;
mod quant_iso_k;
mod quant_iso_k4;
mod quant_iso_v;
mod quant_iso_v4;
mod quant_k;
mod quant_k_gpu_ring;
mod quant_k_turbo3;
mod quant_k_turbo4;
mod quant_planar_k;
mod quant_planar_v;
mod quant_rotor_k3;
mod quant_rotor_k4;
mod quant_rotor_v3;
mod quant_rotor_v4;
mod quant_v;
mod seq_layout;

pub use kv_storage::{
    KvStorage, ISOV3_LAYOUT_TAG, ISOV4_LAYOUT_TAG, ISO_K_ONLY_3_LAYOUT_TAG,
    ISO_K_ONLY_4_LAYOUT_TAG, ISO_SYM_3_LAYOUT_TAG, ISO_SYM_4_LAYOUT_TAG, K8VTURBO2_TCQ_LAYOUT_TAG,
    K8VTURBO3_TCQ_LAYOUT_TAG, PLANARK4_LAYOUT_TAG, ROTORV3_LAYOUT_TAG, ROTORV4_LAYOUT_TAG,
    ROTOR_K_ASYM_3_LAYOUT_PREFIX, ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX, ROTOR_K_ASYM_4_LAYOUT_PREFIX,
    ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX, ROTOR_K_ONLY_3_LAYOUT_TAG, ROTOR_K_ONLY_3_QJL_LAYOUT_TAG,
    ROTOR_K_ONLY_4_LAYOUT_TAG, ROTOR_K_ONLY_4_QJL_LAYOUT_TAG, ROTOR_SYM_3_LAYOUT_TAG,
    ROTOR_SYM_3_QJL_LAYOUT_TAG, ROTOR_SYM_4_LAYOUT_TAG, ROTOR_SYM_4_QJL_LAYOUT_TAG,
    TURBOSYM3_LAYOUT_TAG, TURBOSYM4_LAYOUT_TAG,
};
pub use quant_iso_k::{
    iso_n_groups_for, QuantIsoK3, ISO_K3_BITS, ISO_K3_GROUP_SIZE, ISO_QUAT_BLOCK_SIZE,
};
pub use quant_iso_k4::{QuantIsoK4, ISO_K4_BITS, ISO_K4_GROUP_SIZE};
pub(crate) use quant_iso_v::synced_iso_v_blocks;
pub use quant_iso_v::{IsoBlocks, QuantIsoV3, ISO3_BITS, ISO3_GROUP_SIZE};
pub use quant_iso_v4::{QuantIsoV4, ISO4_BITS, ISO4_GROUP_SIZE};
pub use quant_k::QuantK;
pub use quant_k_gpu_ring::QuantKGpuRing;
pub use quant_k_turbo3::{QuantKTurbo3, TURBO3_K_BITS};
pub use quant_k_turbo4::QuantKTurbo4;
pub use quant_planar_k::QuantPlanarK;
pub use quant_planar_v::QuantPlanarV;
pub(crate) use quant_rotor_k3::synced_rotor_k_blocks;
pub use quant_rotor_k3::{QuantRotorK3, RotorKBlocks, ROTOR3_K_BITS, ROTOR3_K_GROUP_SIZE};
pub use quant_rotor_k4::{QuantRotorK4, ROTOR4_K_BITS, ROTOR4_K_GROUP_SIZE};
pub(crate) use quant_rotor_v3::synced_rotor_v_blocks;
pub use quant_rotor_v3::{QuantRotorV3, RotorBlocks, ROTOR3_V_BITS, ROTOR3_V_GROUP_SIZE};
pub use quant_rotor_v4::{QuantRotorV4, ROTOR4_V_BITS, ROTOR4_V_GROUP_SIZE};
pub use quant_v::QuantV;

#[cfg(test)]
#[path = "quant_rotor_k_qjl_tests.rs"]
mod quant_rotor_k_qjl_tests;

// Long-prompt PlanarK chunked-prefill regression tests.
#[cfg(test)]
#[path = "quant_planar_k_tests.rs"]
mod quant_planar_k_tests;

// `truncate_plan` row/sequence unit conversion + mid-block split tests.
#[cfg(test)]
#[path = "truncate_plan_tests.rs"]
mod truncate_plan_tests;

// ── Paged KV growth ───────────────────────────────────────────────────────────
//
// GPU quantized buffers are allocated in multiples of PAGE_SIZE tokens instead
// of sizing to `max_seq` immediately. On every `append`, if the filled
// sequence would exceed the current allocation, we grow by another page block
// (reallocate + copy prefix). This reduces peak-prefill memory at long ctx:
// a 64K max_seq sequence that only uses 8K tokens carries only ~12.5% of the
// original buffer cost.
//
// Growth algorithm: next_capacity = ceil((prev_seq + new_seq) / PAGE_SIZE) × PAGE_SIZE,
// capped at max_seq. At 64K / 256 that is at most 256 reallocations total per
// layer per request — acceptable versus ~40% peak-memory reduction.
pub const KV_PAGE_SIZE: i32 = 256;

// ── Shared truncate-to-sequence helper ───────────────────────────────────────
//
// Every rotor/iso K and V store accumulates per-append blocks whose `n_tokens`
// field counts **rows** (`b * kv_h * seq_of_block`), not raw sequence
// positions — see the dequant-time reconciliation guards (e.g.
// `synced_rotor_v_blocks`, `synced_iso_v_blocks`) that sum `n_tokens` and
// compare against `b * kv_h * shape[2]`. `truncate_to(n)` takes `n` as a
// **sequence** target, so `n` must be converted to the same row units before
// it is compared against the cumulative `n_tokens` — otherwise, at `kv_h > 1`
// (or `b > 1`), each block's `n_tokens` is inflated by that factor and the
// walk overshoots the target early, dropping blocks that should have been
// kept. This was invisible at `b * kv_h == 1` (the row and sequence counts
// coincide there), which is how the bug shipped.
//
// Blocks are also **not** an alignment the truncation target respects. A block
// spans one whole append, and a speculative-decode partial accept cuts inside
// the verifier's multi-token chunk: keeping only whole blocks would throw away
// the accepted prefix along with the rejected tail, leaving `blocks` short of
// `shape[2]`. That state is only recoverable when a GPU ring happens to hold
// the same prefix; on the CPU append path (a QJL-carrying rotor K store, or
// any `Device::Cpu` run) there is no ring and the next `dequant()` /
// `try_deep_clone()` aborts the request rather than fabricate a zeroed gap.
// So the trailing block is **split**, not dropped.

/// How a block-accumulating KV store must cut its blocks to reach `n`
/// sequence positions.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct TruncatePlan {
    /// Number of leading blocks kept whole.
    pub keep: usize,
    /// Row ranges to retain from block index `keep`, in that block's own row
    /// units. Empty when the cut lands on a block boundary, in which case
    /// block `keep` and everything after it is dropped.
    pub partial: Vec<std::ops::Range<usize>>,
    /// Rows block `keep` holds after the partial cut. `0` when `partial` is
    /// empty.
    pub partial_rows: usize,
}

/// Plan the block cut that truncates a store's accumulated sequence to `n`.
///
/// `block_tokens` is each block's `n_tokens` (rows), in append order.
/// `shape` is the store's `[b, kv_h, seq, head_dim]`.
///
/// A degenerate `b * kv_h == 0` shape (nothing appended yet) keeps no blocks.
/// A block whose row count is not a whole multiple of `b * kv_h` cannot be
/// mapped onto sequence positions, so the walk stops at the last clean
/// boundary rather than guess.
pub(crate) fn truncate_plan(
    block_tokens: impl Iterator<Item = usize>,
    shape: &[i32],
    n: i32,
) -> TruncatePlan {
    let b = shape.first().copied().unwrap_or(0).max(0) as usize;
    let kv_h = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    let rows_per_seq = b.saturating_mul(kv_h);
    if rows_per_seq == 0 {
        return TruncatePlan::default();
    }
    let target_seq = n.max(0) as usize;
    let mut acc_seq = 0usize;
    let mut keep = 0usize;
    for rows in block_tokens {
        // A row count that is not a whole number of sequence positions cannot be
        // mapped onto the truncation target at all. Stop at the last clean
        // boundary — dropping the rest is lossy, but cutting a block whose
        // layout is not understood would be silent corruption. (A zero-row block
        // divides cleanly and spans zero positions, so it is kept, not a stop.)
        if !rows.is_multiple_of(rows_per_seq) {
            break;
        }
        let blk_seq = rows / rows_per_seq;
        if acc_seq + blk_seq <= target_seq {
            acc_seq += blk_seq;
            keep += 1;
            continue;
        }
        let keep_seq = target_seq.saturating_sub(acc_seq);
        if keep_seq == 0 {
            break;
        }
        // Rows within a block run batch-major, then sequence, then kv-head
        // (the seq-major reorder every packed store applies before encoding),
        // so a sequence prefix is one contiguous run per batch element.
        let partial = (0..b)
            .map(|bi| {
                let base = bi * blk_seq * kv_h;
                base..base + keep_seq * kv_h
            })
            .collect();
        return TruncatePlan {
            keep,
            partial,
            partial_rows: keep_seq * kv_h * b,
        };
    }
    TruncatePlan {
        keep,
        partial: Vec::new(),
        partial_rows: 0,
    }
}

/// A per-append payload block whose buffers hold one equal-stride row per
/// `(batch, sequence position, kv head)` triple.
pub(crate) trait BlockRows {
    /// Keep only `ranges` (row indices), leaving `rows` rows behind.
    ///
    /// Returns `false` and leaves the block untouched when any payload buffer's
    /// length is not a whole number of rows — an unsplittable block, which the
    /// caller drops rather than cut into an inconsistent state.
    fn retain_rows(&mut self, ranges: &[std::ops::Range<usize>], rows: usize) -> bool;
}

/// Apply a [`truncate_plan`] to a store's block list.
pub(crate) fn apply_truncate_plan<B: BlockRows>(blocks: &mut Vec<B>, plan: &TruncatePlan) {
    if plan.partial.is_empty() {
        blocks.truncate(plan.keep);
        return;
    }
    blocks.truncate(plan.keep.saturating_add(1));
    let split_ok = blocks
        .last_mut()
        .is_some_and(|last| last.retain_rows(&plan.partial, plan.partial_rows));
    if !split_ok {
        blocks.truncate(plan.keep);
    }
}

/// Whether a block can be split as asked: every buffer divides cleanly into
/// `rows` equal strides, and every range lies inside those rows.
///
/// Checked **once, before any buffer is touched**. A per-buffer check would let
/// an out-of-range plan cut some buffers and not others, leaving a block whose
/// codes, scales and norms disagree about how many rows it holds — silent
/// corruption rather than a refused split.
pub(crate) fn rows_split_ok(
    lengths: &[usize],
    rows: usize,
    ranges: &[std::ops::Range<usize>],
) -> bool {
    rows != 0
        && lengths.iter().all(|len| len.is_multiple_of(rows))
        && ranges.iter().all(|r| r.start <= r.end && r.end <= rows)
}

/// Retain `ranges` (row indices) of a payload buffer holding `total_rows`
/// equal-stride rows.
///
/// Caller must have cleared the block through [`rows_split_ok`] first. An empty
/// buffer — an inactive sideband such as the rotor QJL residual — stays empty.
pub(crate) fn retain_rows_in<T: Copy>(
    buf: &mut Vec<T>,
    total_rows: usize,
    ranges: &[std::ops::Range<usize>],
) {
    if buf.is_empty() || total_rows == 0 {
        return;
    }
    let stride = buf.len() / total_rows;
    let kept: usize = ranges.iter().map(std::ops::Range::len).sum();
    let mut out = Vec::with_capacity(kept.saturating_mul(stride));
    for r in ranges {
        if let Some(slice) = buf.get(r.start * stride..r.end * stride) {
            out.extend_from_slice(slice);
        }
    }
    *buf = out;
}
