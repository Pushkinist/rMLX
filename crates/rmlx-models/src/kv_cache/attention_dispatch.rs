//! Two-phase sparse-attention dispatch.
//!
//! Owns [`sparse_attn_dispatch_if_enabled`], the gate + shape check in front
//! of the phase-1 / phase-2 sparse-attention MSL kernels.
//!
//! # Where the fused-QK table went
//!
//! This module used to also carry a public `FUSED_QK_TABLE` mirroring the
//! codec layer's codec → kernel map. It had no caller: production dispatch is
//! [`rmlx_kv_quant::kvcache::KvCache::try_fused_qk_dispatch`], which uses its
//! own in-crate table (`lookup_fused_qk_kernel`) because the codec layer
//! cannot depend on this crate per the workspace dep-graph rule. A second copy
//! that nothing read could only ever drift from the one that runs, so it was
//! removed rather than kept in sync by hand. Codec coverage and the
//! reachability rule are documented where the live table is: see
//! `docs/KV_QUANT.md` § "Fused-QK head-major K storage".

use rmlx_core::error::Result;
use rmlx_kv_quant::sparse_attn::phase1_score_msl::{phase1_score, TOP_PER_TILE};
use rmlx_kv_quant::sparse_attn::phase2_sparse_attend_msl::{
    phase2_lse_merge, phase2_sparse_attend,
};
use rmlx_kv_quant::sparse_attn_enabled;
use rmlx_loader::HeadBudgets;
use rmlx_mlx::{Array, Device, Dtype};

// ── Two-phase sparse-attention dispatch ──────────────────────────────────────

/// Per-layer shape + tensor inputs for the two-phase sparse-attention
/// dispatch.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed input pack — every field is required by the downstream kernels"
)]
#[derive(Debug)]
pub struct SparseAttnInputs<'a> {
    /// Q tensor for the new token, `[B, n_q_heads, 1, head_dim]`.
    pub query: &'a Array,
    /// PlanarQuant K codes (u32 packed).
    pub k_codes: &'a Array,
    /// PlanarQuant K per-pair scales (f32).
    pub k_scales: &'a Array,
    /// PlanarQuant K 4-bit rotation indices (u32 packed).
    pub k_rot32: &'a Array,
    /// V tensor (bf16 / f16 / f32), `[B, kv_h, kv_seq, head_dim]`.
    pub v: &'a Array,
    /// Batch size.
    pub b: i32,
    /// Number of KV heads.
    pub kv_h: i32,
    /// KV sequence length.
    pub kv_seq: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Query heads per KV head (`n_q_heads / kv_h`).
    pub heads_per_kv: i32,
    /// Layer index (selects row in `head_budgets.per_layer_per_head_budget`).
    pub layer_idx: usize,
    /// Softmax pre-scale (typically `1/sqrt(head_dim)`).
    pub scale: f32,
    /// MLX device (`Device::Gpu` for production).
    pub device: Device,
}

/// Phase 1 + CPU threshold + Phase 2 + LSE merge.
///
/// Inner dispatcher; [`sparse_attn_dispatch_if_enabled`] layers the
/// `RMLX_SPARSE_ATTN` gate + `head_budgets` presence check on top.
pub fn sparse_attn_dispatch(
    inputs: &SparseAttnInputs<'_>,
    head_budgets: &HeadBudgets,
) -> Result<Array> {
    let p1 = phase1_score(
        inputs.query,
        inputs.k_codes,
        inputs.k_scales,
        inputs.k_rot32,
        inputs.b,
        inputs.kv_h,
        inputs.kv_seq,
        inputs.head_dim,
        inputs.heads_per_kv,
        inputs.scale,
        inputs.device,
    )?;

    let n_q_heads = inputs.kv_h * inputs.heads_per_kv;
    let n_bh = inputs.b * n_q_heads;
    let tts_vec = read_tile_top_scores(&p1.tile_top_scores)?;
    let thr_vec = compute_head_threshold(
        &tts_vec,
        p1.n_tiles as usize,
        n_bh as usize,
        head_budgets,
        inputs.layer_idx,
    )?;
    let head_threshold_arr = build_threshold_array(&thr_vec, n_bh)?;

    let p2 = phase2_sparse_attend(
        inputs.query,
        inputs.k_codes,
        inputs.k_scales,
        inputs.k_rot32,
        inputs.v,
        &p1.all_scores,
        &head_threshold_arr,
        inputs.b,
        inputs.kv_h,
        inputs.kv_seq,
        inputs.head_dim,
        inputs.heads_per_kv,
        p1.n_tiles,
        inputs.scale,
        inputs.device,
    )?;

    phase2_lse_merge(
        &p2.partial_o,
        &p2.tile_lse,
        inputs.b,
        n_q_heads,
        inputs.head_dim,
        p1.n_tiles,
        inputs.device,
    )
}

/// Two-phase sparse-attention dispatch with env-var gate + budget check.
///
/// Returns `Some(Array)` when:
/// 1. [`sparse_attn_enabled`] is `true` (env-var `RMLX_SPARSE_ATTN=1`),
/// 2. `head_budgets` is `Some`, and
/// 3. [`sparse_attn_dispatch`] succeeds.
///
/// Returns `None` when either gate fails OR the inner dispatch errors.
pub fn sparse_attn_dispatch_if_enabled(
    inputs: &SparseAttnInputs<'_>,
    head_budgets: Option<&HeadBudgets>,
) -> Option<Array> {
    if !sparse_attn_enabled() {
        return None;
    }
    let budgets = head_budgets?;
    match sparse_attn_dispatch(inputs, budgets) {
        Ok(out) => Some(out),
        Err(e) => {
            tracing::warn!(error = %e, "sparse_attn inner dispatch errored — falling back to dense");
            None
        }
    }
}

// ── Bridge helpers ───────────────────────────────────────────────────────────

/// Pull `tile_top_scores` `[n_tiles, n_bh, TOP_PER_TILE]` f32 to host.
fn read_tile_top_scores(tile_top_scores: &Array) -> Result<Vec<f32>> {
    tile_top_scores.eval()?;
    let bytes = tile_top_scores.to_bytes()?;
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        // chunks_exact(4) guarantees four elements.  `unwrap_or_default`
        // keeps clippy::indexing_slicing happy without changing semantics
        // (the default `[0; 4]` is unreachable on an exact-sized chunk).
        let arr: [u8; 4] = chunk.try_into().unwrap_or_default();
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// Compute the per-`(B, H)` raw QK threshold from Phase-1 tile tops.
fn compute_head_threshold(
    tile_top_scores: &[f32],
    n_tiles: usize,
    n_bh: usize,
    head_budgets: &HeadBudgets,
    layer_idx: usize,
) -> Result<Vec<f32>> {
    let n_layers = head_budgets.per_layer_per_head_budget.len();
    let row = head_budgets
        .per_layer_per_head_budget
        .get(layer_idx)
        .ok_or_else(|| {
            rmlx_core::error::Error::Quant(format!(
                "sparse_attn: layer_idx={layer_idx} exceeds head_budgets layer rows ({n_layers})"
            ))
        })?;
    let top_per_tile = TOP_PER_TILE as usize;

    let n_q_heads = row.len();
    if n_q_heads == 0 || !n_bh.is_multiple_of(n_q_heads) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "sparse_attn: n_bh={n_bh} not divisible by n_q_heads={n_q_heads}"
        )));
    }

    let mut thresholds = vec![f32::NEG_INFINITY; n_bh];
    for (bh, slot) in thresholds.iter_mut().enumerate().take(n_bh) {
        let hq = bh % n_q_heads;
        let k = (*row.get(hq).ok_or_else(|| {
            rmlx_core::error::Error::Quant(format!(
                "sparse_attn: invariant violation — hq={hq} out of bounds (n_q_heads={n_q_heads})"
            ))
        })?) as usize;
        let mut all: Vec<f32> = Vec::with_capacity(n_tiles * top_per_tile);
        for t in 0..n_tiles {
            let tile_base = (t * n_bh + bh) * top_per_tile;
            let tile_slice = tile_top_scores
                .get(tile_base..tile_base + top_per_tile)
                .ok_or_else(|| {
                    rmlx_core::error::Error::Quant(format!(
                        "sparse_attn: invariant violation — tile_top_scores OOB at tile_base={tile_base} top_per_tile={top_per_tile}"
                    ))
                })?;
            for &v in tile_slice {
                if v.is_finite() {
                    all.push(v);
                }
            }
        }
        all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(all.len()).max(1) - 1;
        *slot = all.get(take).copied().unwrap_or(f32::NEG_INFINITY);
    }
    Ok(thresholds)
}

/// Build a `[n_bh]` f32 mlx `Array` from a flat threshold vec.
fn build_threshold_array(thr: &[f32], n_bh: i32) -> Result<Array> {
    let bytes: Vec<u8> = thr.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[n_bh], Dtype::F32)
}

#[cfg(test)]
#[path = "attention_dispatch_tests.rs"]
mod attention_dispatch_tests;
