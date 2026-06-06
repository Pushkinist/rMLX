#![allow(
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    clippy::wildcard_enum_match_arm,
    reason = "Softmax-mass sink: bounds checked by explicit shape guards above each indexing path; \
              numeric loops over fixed-size shape components are clearer with explicit indices; \
              dtype match has explicit F32/BF16 arms and a bail-out for unsupported variants."
)]

//! True softmax-mass calibration sink.
//!
//! Implements `CalibrationSink` by computing per-(layer, kv_head, q_pos)
//! attention scores from the captured `(q_last, k_full)` pair, applying the
//! Q-row softmax, and selecting the smallest top-`k` that covers
//! `target_mass` cumulative softmax weight. Per-head budgets are aggregated
//! across prompts by **max** (worst-case retrieval-safe).
//!
//! # Aggregation policy
//!
//! Per-prompt: for each Q-head, score the kv keys against that **individual
//! Q-head**'s last-position query, softmax, sort descending, find smallest k
//! such that cumulative >= target. Within a GQA group (q_per_kv Q-heads
//! sharing one KV head), take the **max** budget over Q-heads — sharper
//! per-Q-head distributions can require a wider top-K than the group mean
//! would suggest, and `max` preserves the worst-case retrieval-safe contract
//! at the per-prompt aggregation step.
//!
//! Across prompts: again `max`. The same safety principle applies — covers
//! every prompt's worst case at the cost of higher budgets than `mean`. `max`
//! is mandated for both within-GQA-group and across-prompt aggregation.
//!
//! # Tensors
//!
//! - `q_last`: shape `[1, n_q_heads, S_full_or_last, head_dim]`
//!   (we use the last row only).
//! - `k_full`: shape `[1, n_kv_heads, S_kv, head_dim]` — full accumulated K.
//!
//! Per-prompt host-side cost: O(n_layers * n_kv_heads * S_kv * head_dim) for
//! the Q@K^T pass, dominated by host-mem bandwidth — acceptable for short
//! calibration corpora.

use std::cmp::Ordering;

use rmlx_core::error::Result as RmlxResult;
use rmlx_mlx::{Array, Dtype};
use rmlx_models::calibration_sink::CalibrationSink;

/// Per-prompt -> max-aggregator accumulator.
pub(crate) struct SoftmaxMassSink {
    /// `[num_layers][n_q_heads]` running max-budget table. Per-Q-head budgets
    /// are computed against each Q-head's own last-row query and aggregated
    /// across prompts by `max`.
    pub budgets: Vec<Vec<u32>>,
    /// Per-prompt observed max sequence length (max over layers, prompts).
    pub max_seq_len: u32,
    /// Target cumulative softmax mass (e.g. `0.95`).
    target_mass: f32,
    /// Floor budget per (kv_head).
    floor: u32,
    /// Number of (query) attention heads.
    n_q_heads: usize,
    /// Number of KV heads (must divide `n_q_heads`).
    n_kv_heads: usize,
    /// Attention head dim.
    head_dim: usize,
    /// Scaling factor for attention scores (typically `1 / sqrt(head_dim)`).
    scale: f32,
}

impl SoftmaxMassSink {
    pub(crate) fn new(
        num_layers: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        target_mass: f32,
        floor: u32,
    ) -> Self {
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        Self {
            budgets: vec![vec![floor.max(1); n_q_heads]; num_layers],
            max_seq_len: 0,
            target_mass,
            floor: floor.max(1),
            n_q_heads,
            n_kv_heads,
            head_dim,
            scale,
        }
    }

    /// Return the `[num_layers][n_q_heads]` table for emission. Budgets are
    /// stored per-Q-head, so this is a straight clone; the within-GQA-group
    /// `max` happens at `record` time.
    pub(crate) fn expand_to_q_heads(&self) -> Vec<Vec<u32>> {
        self.budgets.clone()
    }

    /// Compute the per-key softmax distribution + the smallest k covering
    /// `target_mass`. Pure-host pure-CPU helper, exposed for unit testing.
    ///
    /// Returns `(budget, distribution_sorted_desc)`. `distribution_sorted_desc`
    /// is the descending-sorted normalised softmax row.
    pub(crate) fn budget_for_distribution(scores: &[f32], target_mass: f32, floor: u32) -> u32 {
        if scores.is_empty() {
            return floor.max(1);
        }
        // Subtract max for stability.
        let mut max_score = f32::NEG_INFINITY;
        for &s in scores {
            if s > max_score {
                max_score = s;
            }
        }
        if !max_score.is_finite() {
            return floor.max(1);
        }
        let mut probs: Vec<f32> = Vec::with_capacity(scores.len());
        let mut sum = 0.0_f64;
        for &s in scores {
            let e = (s - max_score).exp();
            sum += f64::from(e);
            probs.push(e);
        }
        if sum <= 0.0 || !sum.is_finite() {
            return floor.max(1);
        }
        let inv_sum = (1.0_f64 / sum) as f32;
        for p in &mut probs {
            *p *= inv_sum;
        }
        probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let mut acc = 0.0_f32;
        let mut budget = probs.len() as u32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if acc >= target_mass {
                budget = (i + 1) as u32;
                break;
            }
        }
        budget.max(floor.max(1))
    }
}

impl CalibrationSink for SoftmaxMassSink {
    fn record(&mut self, layer_idx: usize, q_last: &Array, k_full: &Array) -> RmlxResult<()> {
        let q_shape = q_last.shape();
        let k_shape = k_full.shape();
        if q_shape.len() != 4 || k_shape.len() != 4 {
            return Err(rmlx_core::Error::Config(format!(
                "SoftmaxMassSink: unexpected shapes q={q_shape:?} k={k_shape:?}"
            )));
        }
        let q_h = q_shape[1] as usize;
        let q_s = q_shape[2] as usize;
        let q_d = q_shape[3] as usize;
        let kv_h = k_shape[1] as usize;
        let s_kv = k_shape[2] as usize;
        let k_d = k_shape[3] as usize;
        if q_h != self.n_q_heads
            || kv_h != self.n_kv_heads
            || q_d != self.head_dim
            || k_d != self.head_dim
        {
            return Err(rmlx_core::Error::Config(format!(
                "SoftmaxMassSink: shape mismatch — expected n_q_heads={}, n_kv_heads={}, head_dim={}; got q={q_shape:?} k={k_shape:?}",
                self.n_q_heads, self.n_kv_heads, self.head_dim
            )));
        }
        if s_kv == 0 {
            return Ok(());
        }
        if s_kv as u32 > self.max_seq_len {
            self.max_seq_len = s_kv as u32;
        }

        let q_vec = array_to_host_f32(q_last)?;
        let k_vec = array_to_host_f32(k_full)?;

        let q_per_kv = q_h / kv_h;
        let head_dim = self.head_dim;
        let q_last_row_off = (q_s - 1) * head_dim;

        // Compute per-Q-head softmax independently, then take `max` within each
        // GQA group when writing the per-Q-head budget table. This matches the
        // across-prompt `max` aggregation and avoids under-budgeting Q-heads
        // whose distribution is sharper than the group mean would suggest.
        let mut scores = vec![0.0_f32; s_kv];
        for kvh in 0..kv_h {
            let kvh_off = kvh * s_kv * head_dim;
            let q_group_start = kvh * q_per_kv;
            let q_group_end = q_group_start + q_per_kv;

            // Per-Q-head budgets across this GQA group.
            let mut group_budgets: Vec<u32> = Vec::with_capacity(q_per_kv);
            for hq in q_group_start..q_group_end {
                let q_off = hq * q_s * head_dim + q_last_row_off;
                // Scores = (Q_hq . K_kvh,i^T) * scale for each i in 0..s_kv.
                for i in 0..s_kv {
                    let k_row_off = kvh_off + i * head_dim;
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q_vec[q_off + d] * k_vec[k_row_off + d];
                    }
                    scores[i] = dot * self.scale;
                }
                let b = Self::budget_for_distribution(&scores, self.target_mass, self.floor);
                group_budgets.push(b);
            }

            // Within-GQA-group `max`: each Q-head in the group gets the
            // worst-case (largest) per-Q-head budget. This is a stricter form
            // of GQA expansion than the prior mean-Q approach: it covers any
            // sharper-than-mean head individually instead of relying on the
            // averaged-Q approximation. Across-prompt aggregation (also `max`)
            // happens via the `if b_max > existing` write below.
            let group_max = group_budgets.iter().copied().max().unwrap_or(self.floor);
            for hq in q_group_start..q_group_end {
                if group_max > self.budgets[layer_idx][hq] {
                    self.budgets[layer_idx][hq] = group_max;
                }
            }
        }
        Ok(())
    }
}

/// Copy a 4-D Array to a host f32 buffer. Supports F32 / BF16.
fn array_to_host_f32(a: &Array) -> RmlxResult<Vec<f32>> {
    a.eval()
        .map_err(|e| rmlx_core::Error::Mlx(format!("SoftmaxMassSink: array eval: {e}")))?;
    let bytes = a
        .to_bytes()
        .map_err(|e| rmlx_core::Error::Mlx(format!("SoftmaxMassSink: to_bytes: {e}")))?;
    // Guard against int overflow on pathological shapes.
    let mut total: usize = 1;
    for d in a.shape() {
        if d < 0 {
            return Err(rmlx_core::Error::Config(format!(
                "SoftmaxMassSink: negative shape component {d} in {:?}",
                a.shape()
            )));
        }
        total = total.checked_mul(d as usize).ok_or_else(|| {
            rmlx_core::Error::Config(format!(
                "SoftmaxMassSink: shape product overflow on {:?}",
                a.shape()
            ))
        })?;
    }
    match a.dtype() {
        Dtype::F32 => {
            if bytes.len() < total * 4 {
                return Err(rmlx_core::Error::Config(format!(
                    "SoftmaxMassSink: F32 buffer too small: {} < {}",
                    bytes.len(),
                    total * 4
                )));
            }
            // Use chunks_exact(4) to match `k_to_host_f32` style; identical
            // semantics, fewer bounds checks.
            let out: Vec<f32> = bytes[..total * 4]
                .chunks_exact(4)
                .map(|c| {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(c);
                    f32::from_le_bytes(arr)
                })
                .collect();
            Ok(out)
        }
        Dtype::Bf16 => {
            if bytes.len() < total * 2 {
                return Err(rmlx_core::Error::Config(format!(
                    "SoftmaxMassSink: BF16 buffer too small: {} < {}",
                    bytes.len(),
                    total * 2
                )));
            }
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let o = i * 2;
                let mut arr = [0u8; 2];
                arr.copy_from_slice(&bytes[o..o + 2]);
                let raw = u16::from_le_bytes(arr);
                out.push(f32::from_bits(u32::from(raw) << 16));
            }
            Ok(out)
        }
        other => Err(rmlx_core::Error::Config(format!(
            "SoftmaxMassSink: unsupported dtype {other:?}"
        ))),
    }
}

#[cfg(test)]
#[path = "calibration_softmax_tests.rs"]
mod calibration_softmax_tests;
