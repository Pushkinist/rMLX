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

// Mid-sequence truncation of the CPU-side turbo / planar / affine stores.
#[cfg(test)]
#[path = "cpu_block_truncate_tests.rs"]
mod cpu_block_truncate_tests;

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
// Every CPU-side K and V store accumulates per-append payload blocks (the
// rotor/iso stores, and the turbo / planar / affine ones) whose row count is
// `b * kv_h * seq_of_block`, not raw sequence positions — see the dequant-time
// reconciliation guards (e.g. `synced_rotor_v_blocks`, `synced_iso_v_blocks`)
// that sum `n_tokens` and compare against `b * kv_h * shape[2]`.
// `truncate_to(n)` takes `n` as a **sequence** target, so `n` must be converted
// to the same row units before it is compared against the cumulative row count
// — otherwise, at `kv_h > 1` (or `b > 1`), each block's row count is inflated
// by that factor and the walk overshoots the target early, dropping blocks that
// should have been kept. This was invisible at `b * kv_h == 1` (the row and
// sequence counts coincide there), which is how the bug shipped.
//
// Blocks are also **not** an alignment the truncation target respects. A block
// spans one whole append, and a speculative-decode partial accept cuts inside
// the verifier's multi-token chunk: keeping only whole blocks would throw away
// the accepted prefix along with the rejected tail, leaving `blocks` short of
// `shape[2]`. That state is only recoverable when a GPU ring happens to hold
// the same prefix; on the CPU append path (a QJL-carrying rotor K store, a
// codec whose encode is forced to CPU such as the 2-/3-bit turbo V side, or
// any `Device::Cpu` run) there is no ring and the next `dequant()` /
// `dequantize_choice()` / `try_deep_clone()` aborts the request rather than
// fabricate a zeroed gap. So the trailing block is **split**, not dropped.
//
// The flat-GPU-buffer half of the turbo / planar / affine stores needs no cut:
// its dequant slices `[0, shape[2])` and the next `append` writes at
// `prev_seq == shape[2]`, so lowering `shape[2]` alone already makes the
// rejected region overwritable. Only the append-only CPU side has to be cut.
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

/// Clamp a truncation target to what the store already covers.
///
/// Truncation is monotone-decreasing by contract, and for the turbo / planar /
/// affine stores that has to be enforced rather than assumed. `n > shape[2]` is
/// reachable: post-`exit_prefill` the codec store is frozen at the prefill
/// length while `KvCache::offset` keeps advancing on the bf16 decode mirror, so
/// a speculative rollback to a position inside the decode window arrives with a
/// target past the store's own fill. Raising `shape[2]` to meet it invents
/// coverage that no payload backs — the CPU dequant then reads past the blocks
/// and the SSD spill persists a store whose header claims more tokens than its
/// bytes hold.
///
/// The rotor / iso stores deliberately do **not** clamp, and the reason is that
/// they do not need to, not that clamping would cost them anything: a ring-only
/// tail spans `[blocks_coverage, shape[2])`, strictly below `shape[2]`, so a
/// `min(n, shape[2])` could never discard it. What makes the asymmetry safe is
/// that those stores already **abort loudly** on an over-long target —
/// `synced_rotor_v_blocks` / `synced_iso_v_blocks` size their ring readback from
/// `shape[2]` and return `Err` when the ring cannot cover it. The clamp would buy
/// them nothing. These stores have no ring and no such guard, so for them the
/// clamp is the only reading that keeps `shape[2] == payload coverage` true.
pub(crate) fn clamp_truncate_target(shape: &[i32], n: i32) -> i32 {
    let covered = shape.get(2).copied().unwrap_or(0).max(0);
    n.max(0).min(covered)
}

/// Rows a `[…, D]`-shaped codec block holds — one row per `head_dim`-wide
/// vector, i.e. the product of the leading three axes.
///
/// [`TurboBlocks`] and [`PlanarBlocks`] carry no explicit row count (unlike the
/// rotor / iso blocks' `n_tokens`), so it has to come from `original_shape` —
/// and that field is **not** a reliable axis map. The CPU append paths record
/// the sequence-major chunk shape `[B, S_block, kv_h, D]` while the SSD hydrate
/// paths record the store's head-major `[B, kv_h, S, D]` over the same
/// sequence-major bytes. Only the product is ever read back (`turbo_dequantize`
/// / `planar_dequantize` use it purely as an element count) and the last axis is
/// the row width in both, so "product of the first three axes" is the one
/// reading both conventions agree on.
///
/// A shape that cannot be read this way (negative dimension, or overflow)
/// yields 0. Such a block holds no decodable payload either — the dequant-side
/// coverage check is what reports it.
pub(crate) fn block_rows(original_shape: &[i32; 4]) -> usize {
    let [d0, d1, d2, _] = *original_shape;
    match (
        usize::try_from(d0),
        usize::try_from(d1),
        usize::try_from(d2),
    ) {
        (Ok(a), Ok(b), Ok(c)) => a.saturating_mul(b).saturating_mul(c),
        _ => 0,
    }
}

/// A per-append payload block whose buffers hold one equal-stride row per
/// `(sequence position, kv head)` pair.
pub(crate) trait BlockRows {
    /// Rows this block currently holds.
    ///
    /// The single way to ask the question — the rotor / iso blocks answer from
    /// their `n_tokens` field, the turbo / planar blocks from
    /// [`block_rows`] over `original_shape`. Keeping it on the trait is what
    /// stops the two conventions being re-derived at each of the thirteen
    /// `truncate_plan` call sites.
    fn rows(&self) -> usize;

    /// Keep the first `rows` rows, dropping the rest.
    ///
    /// Returns `false` and leaves the block untouched when the cut is not
    /// expressible — any payload buffer whose length is not a whole number of
    /// rows, or a `rows` past the block's own row count. The caller drops such
    /// a block rather than cut it into an inconsistent state.
    fn retain_rows(&mut self, rows: usize) -> bool;
}

/// Apply a [`truncate_plan`] to a store's block list.
///
/// The plan must have been built from `blocks`. Nothing in the types enforces
/// that — [`truncate_plan`] takes an iterator of row counts, not the list — so
/// the split addresses block `plan.keep` **by index**. Reaching for the last
/// element instead would, against a shorter or already-mutated list, silently
/// cut a block that should have been kept whole; a missing index degrades to the
/// documented drop-and-warn instead.
pub(crate) fn apply_truncate_plan<B: BlockRows>(blocks: &mut Vec<B>, plan: &TruncatePlan) {
    let Some(rows) = plan.partial_rows else {
        blocks.truncate(plan.keep);
        return;
    };
    blocks.truncate(plan.keep.saturating_add(1));
    let split_ok = blocks
        .get_mut(plan.keep)
        .is_some_and(|target| target.retain_rows(rows));
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

// The two impls below live here rather than beside their structs: `TurboBlocks`
// and `PlanarBlocks` are codec-layer types shared by five different storage
// structs, and how they are cut is a storage-layer contract that belongs with
// `truncate_plan` / `rows_split_ok` / `retain_rows_in`. The rotor / iso blocks
// are declared inside `storage/` and keep their impls beside the struct.

/// Truncate a block-accumulating store to `n` sequence positions.
///
/// The whole cut sequence for the turbo / planar stores, in one place: clamp the
/// target, plan the block walk, apply it, then lower `shape[2]`. Five stores
/// share it (`QuantV`, `QuantKTurbo3`, `QuantKTurbo4`, `QuantPlanarK`,
/// `QuantPlanarV`) and the order is load-bearing — the clamp has to run before
/// the plan, and `shape[2]` has to move after it. Independent copies of that
/// sequence are the same drift surface that let twelve `KvStorage` arms keep a
/// bare `shape[2] = n` while the rest of the layer learned to cut.
///
/// Not used by the rotor / iso stores: they do not clamp (see
/// [`clamp_truncate_target`]) and carry extra ring bookkeeping around the same
/// three steps.
pub(crate) fn truncate_block_store<B: BlockRows>(blocks: &mut Vec<B>, shape: &mut [i32], n: i32) {
    let n = clamp_truncate_target(shape, n);
    let plan = truncate_plan(blocks.iter().map(BlockRows::rows), shape, n);
    apply_truncate_plan(blocks, &plan);
    // `get_mut` rather than `shape[2]`: the store shape is rank-4 by
    // construction, and this is the bounds proof rather than a claim.
    if let Some(seq) = shape.get_mut(2) {
        *seq = n;
    }
}

impl BlockRows for crate::turboquant::TurboBlocks {
    fn rows(&self) -> usize {
        block_rows(&self.original_shape)
    }

    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile, which is what stops a
    /// buffer from surviving a mid-block truncation at its full length.
    fn retain_rows(&mut self, rows: usize) -> bool {
        let Self {
            codes,
            scales,
            original_shape,
            bits: _,
        } = self;
        let total_rows = block_rows(original_shape);
        let [_, _, _, width] = *original_shape;
        // Everything fallible resolves before the first buffer is touched —
        // half-cutting a block is the silent corruption this is here to avoid.
        let Ok(rows_i32) = i32::try_from(rows) else {
            return false;
        };
        if !rows_split_ok(&[codes.len(), scales.len()], total_rows, rows) {
            return false;
        }
        retain_rows_in(codes, total_rows, rows);
        retain_rows_in(scales, total_rows, rows);
        // Record the geometry the cut actually produced — `rows` rows of
        // `width` elements — instead of guessing which axis of the caller's
        // convention held the sequence. Both consumers read this as an element
        // count only (see `block_rows`).
        *original_shape = [1, 1, rows_i32, width];
        true
    }
}

impl BlockRows for crate::planarquant::PlanarBlocks {
    fn rows(&self) -> usize {
        block_rows(&self.original_shape)
    }

    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile, which is what stops a
    /// buffer from surviving a mid-block truncation at its full length.
    fn retain_rows(&mut self, rows: usize) -> bool {
        let Self {
            codes,
            scales,
            rotations,
            original_shape,
            bits: _,
        } = self;
        let total_rows = block_rows(original_shape);
        let [_, _, _, width] = *original_shape;
        let Ok(rows_i32) = i32::try_from(rows) else {
            return false;
        };
        if !rows_split_ok(
            &[codes.len(), scales.len(), rotations.len()],
            total_rows,
            rows,
        ) {
            return false;
        }
        retain_rows_in(codes, total_rows, rows);
        retain_rows_in(scales, total_rows, rows);
        retain_rows_in(rotations, total_rows, rows);
        // See the TurboBlocks impl: the recorded shape is the cut geometry, not
        // a claim about which axis the caller's convention held the sequence on.
        *original_shape = [1, 1, rows_i32, width];
        true
    }
}
