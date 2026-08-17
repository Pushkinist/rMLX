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
//
// The split is confined to `b == 1`, and that is a correctness bound, not a
// simplification. Every store ends `dequant` with
// `seq_layout::transpose_seq_heads` over the *concatenation* of its blocks,
// which reads the buffer as one `[B, S_total, kv_h, D]` run. Each block is
// only `[B, S_block, kv_h, D]`, so at `b > 1` the concatenation interleaves
// batch elements and the reading is wrong for any store holding more than one
// block. Splitting at `b > 1` would turn a mid-block cut from
// blocks-short-of-`shape[2]` — which the reconciliation guards reject loudly —
// into a two-block store that decodes silently scrambled. So `b > 1` keeps the
// whole-block drop and the loud error. `sdpa::rotor_flash_shape_ok` refuses
// `b != 1` for the same underlying reason, which is also why no `b > 1` store
// ever has a ring to rebuild the gap from.

/// How a block-accumulating KV store must cut its blocks to reach `n`
/// sequence positions.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct TruncatePlan {
    /// Number of leading blocks kept whole.
    pub keep: usize,
    /// Rows to retain from block index `keep`. `None` when the cut lands on a
    /// block boundary, or when the split was refused — in both cases block
    /// `keep` and everything after it is dropped.
    pub partial_rows: Option<usize>,
}

/// Plan the block cut that truncates a store's accumulated sequence to `n`.
///
/// `block_tokens` is each block's `n_tokens` (rows), in append order.
/// `shape` is the store's `[b, kv_h, seq, head_dim]`.
///
/// A degenerate `b * kv_h == 0` shape (nothing appended yet) keeps no blocks.
/// A block whose row count is not a whole multiple of `b * kv_h` cannot be
/// mapped onto sequence positions, so the walk stops at the last clean
/// boundary rather than guess. `b > 1` never splits — see the module note.
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
            // Loud: the caller still lowers `shape[2]` to `n`, so the store is
            // about to enter the blocks-short-of-shape state that aborts the
            // next `dequant`. Without this the trail goes cold here.
            tracing::warn!(
                block_rows = rows,
                rows_per_seq,
                kept_seq = acc_seq,
                target_seq,
                "KV block truncate: block row count is not a whole number of sequence \
                 positions — dropping it and every block after it, leaving the store short \
                 of its truncation target"
            );
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
        if b != 1 {
            // Rows run batch-major, so a sequence prefix is not a row prefix and
            // the resulting two-block store would decode scrambled rather than
            // error. Drop the block and let the reconciliation guard report the
            // gap. See the module note.
            tracing::warn!(
                b,
                kv_h,
                kept_seq = acc_seq,
                target_seq,
                "KV block truncate: refusing to split a block at b > 1 — the decode path \
                 reads the block concatenation as one sequence run, so a split store would \
                 be silently scrambled; dropping the block instead, which leaves the store \
                 short of its truncation target"
            );
            break;
        }
        return TruncatePlan {
            keep,
            partial_rows: Some(keep_seq * kv_h),
        };
    }
    TruncatePlan {
        keep,
        partial_rows: None,
    }
}

/// A per-append payload block whose buffers hold one equal-stride row per
/// `(sequence position, kv head)` pair.
pub(crate) trait BlockRows {
    /// Keep the first `rows` rows, dropping the rest.
    ///
    /// Returns `false` and leaves the block untouched when the cut is not
    /// expressible — any payload buffer whose length is not a whole number of
    /// rows, or a `rows` past the block's own row count. The caller drops such
    /// a block rather than cut it into an inconsistent state.
    fn retain_rows(&mut self, rows: usize) -> bool;
}

/// Apply a [`truncate_plan`] to a store's block list.
pub(crate) fn apply_truncate_plan<B: BlockRows>(blocks: &mut Vec<B>, plan: &TruncatePlan) {
    let Some(rows) = plan.partial_rows else {
        blocks.truncate(plan.keep);
        return;
    };
    blocks.truncate(plan.keep.saturating_add(1));
    let split_ok = blocks.last_mut().is_some_and(|last| last.retain_rows(rows));
    if !split_ok {
        // Loud for the same reason as the planner's refusals: the caller lowers
        // `shape[2]` regardless, so dropping the block leaves `rows` uncovered
        // and the next `dequant` aborts a long way from here.
        tracing::warn!(
            kept_blocks = plan.keep,
            uncovered_rows = rows,
            "KV block truncate: trailing block could not be split — dropping it whole, \
             leaving the store short of its truncation target"
        );
        blocks.truncate(plan.keep);
    }
}

/// Whether a block can be cut to `keep_rows`: every buffer divides cleanly into
/// `total_rows` equal strides, and the target is inside them.
///
/// Checked **once, before any buffer is touched**. A per-buffer check would let
/// an out-of-range cut shorten some buffers and not others, leaving a block
/// whose codes, scales and norms disagree about how many rows it holds — silent
/// corruption rather than a refused split.
pub(crate) fn rows_split_ok(lengths: &[usize], total_rows: usize, keep_rows: usize) -> bool {
    total_rows != 0
        && keep_rows <= total_rows
        && lengths.iter().all(|len| len.is_multiple_of(total_rows))
}

/// Keep the first `keep_rows` rows of a payload buffer holding `total_rows`
/// equal-stride rows.
///
/// Caller must have cleared the block through [`rows_split_ok`] first. An empty
/// buffer — an inactive sideband such as the rotor QJL residual — stays empty.
///
/// A row prefix is a byte prefix (the split is `b == 1` only, see the module
/// note), so this is `Vec::truncate`: no allocation, no copy. That matters
/// because it sits on the speculative rollback path, which runs once per round
/// per store per layer.
pub(crate) fn retain_rows_in<T>(buf: &mut Vec<T>, total_rows: usize, keep_rows: usize) {
    if buf.is_empty() || total_rows == 0 {
        return;
    }
    let stride = buf.len() / total_rows;
    buf.truncate(keep_rows.saturating_mul(stride));
}
