//! Maple sparse MoE: fp32 `MapleGate` + clamped-SwiGLU `MapleSwitchGLU`.
//!
//! v1 is the portable reference path from `mlx_lm/models/maple.py` (no fused
//! Metal router). Every decoder layer is MoE (`first_k_dense_replace=0`); there
//! are no shared experts.

#![allow(clippy::struct_field_names)]

use rmlx_core::error::Result;
use rmlx_mlx::{
    add, argpartition, argsort, clip, divide, expand_dims, floor_divide, matmul, maximum, multiply,
    negative, scalar_f32, silu, softmax, sum_axis, sum_axis_keepdims, take_along_axis, Array,
    Device, Dtype,
};

use crate::layers::Linear;

/// Flattened `n_tokens * top_k` count at or above which the expert dispatch
/// sorts indices for contiguous per-expert access (prefill). Below it (decode,
/// single token) the simple broadcast path is cheaper. Matches mlx-lm SwitchGLU.
const SORT_DISPATCH_THRESHOLD: i32 = 64;

/// SwiGLU clamp for MoE experts only (the dense MapleMLP is unclamped).
/// Part of the trained forward, not an optional guard.
const MLP_CLAMP: f32 = 7.0;

/// `silu(minimum(gate, 7)) * clip(up, -7, 7)`.
///
/// Scalars are narrowed to the activation dtype so bf16 experts stay bf16
/// (Python uses Python floats for the same reason).
fn clamped_swiglu(gate: &Array, up: &Array, device: Device) -> Result<Array> {
    let seven = scalar_f32(MLP_CLAMP).astype(gate.dtype(), device)?;
    let neg_seven = scalar_f32(-MLP_CLAMP).astype(gate.dtype(), device)?;
    // min(gate, 7) = -max(-gate, -7). Not `clip(gate, -7, 7)`: the lower bound
    // is unbounded in the trained forward.
    let gate_c = negative(
        &maximum(&negative(gate, device)?, &neg_seven, device)?,
        device,
    )?;
    let gate_act = silu(&gate_c, device)?;
    let up_c = clip(up, &neg_seven, &seven, device)?;
    multiply(&gate_act, &up_c, device)
}

/// Combined in float32, rounded once at the end (reference `moe_infer`).
fn aggregate_expert_outputs(
    expert_outputs: &Array,
    scores: &Array,
    device: Device,
) -> Result<Array> {
    let y = expert_outputs.astype(Dtype::F32, device)?;
    let w = expand_dims(scores, -1, device)?;
    let scaled = multiply(&y, &w, device)?;
    let summed = sum_axis(&scaled, 1, device)?;
    summed.astype(expert_outputs.dtype(), device)
}

/// Router: plain (unquantized) `[num_experts, hidden]` weight.
///
/// `gates = x.astype(F32) @ W.T.astype(F32)`, softmax over experts, argpartition
/// top-k, gather scores, renormalize (`sum + 1e-20`).
#[allow(missing_debug_implementations)]
pub(super) struct MapleGate {
    /// `[num_experts, hidden]` bf16/f32 param — never quantized.
    pub(super) weight: Array,
    /// Experts per token (8).
    pub(super) top_k: usize,
}

impl MapleGate {
    /// `x`: `[n_tokens, hidden]`.
    /// Returns `(indices [n, top_k] i32, scores [n, top_k] f32)`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: weight is [num_experts, hidden], x is [n, hidden]"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<(Array, Array)> {
        let n_tokens = x.shape()[0];
        let ne = self.weight.shape()[0];
        let tk = self.top_k as i32;

        // f32-ok: Maple router_dtype is fp32; near-tied top-8 flips in bf16
        let x_f32 = x.astype(Dtype::F32, device)?;
        let w_t = self
            .weight
            .astype(Dtype::F32, device)?
            .transpose(&[1, 0], device)?;
        let gates = matmul(&x_f32, &w_t, device)?;

        let probs = softmax(&gates, -1, device)?;
        let part_idx = argpartition(&probs, -tk, -1, device)?;
        let expert_idx = part_idx.slice(&[0, ne - tk], &[n_tokens, ne], &[1, 1], device)?;
        let expert_idx_i32 = expert_idx.astype(Dtype::I32, device)?;
        let scores = take_along_axis(&probs, &expert_idx_i32, -1, device)?;
        let s_sum = sum_axis_keepdims(&scores, -1, device)?;
        // f32-ok: Maple router_dtype is fp32; near-tied top-8 flips in bf16
        let scores = divide(&scores, &add(&s_sum, &scalar_f32(1e-20), device)?, device)?;
        Ok((expert_idx_i32, scores))
    }
}

/// SwitchGLU with split gate/up/down expert projections (Qwen3.5 layout).
///
/// Checkpoint may ship fused `up_gate_proj`; the loader keeps them split.
#[allow(missing_debug_implementations)]
pub(super) struct MapleSwitchGLU {
    /// Expert gate projection `[num_experts, moe_intermediate, hidden]`.
    pub(super) gate_proj: Linear,
    /// Expert up projection `[num_experts, moe_intermediate, hidden]`.
    pub(super) up_proj: Linear,
    /// Expert down projection `[num_experts, hidden, moe_intermediate]`.
    pub(super) down_proj: Linear,
}

impl MapleSwitchGLU {
    /// `x`: `[n_tokens, hidden]`.
    /// `expert_indices`: `[n_tokens, top_k]` (i32).
    /// Returns unscaled expert outputs `[n_tokens, top_k, hidden]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: index buffers sized from x/indices shapes"
    )]
    fn forward(&self, x: &Array, expert_indices: &Array, device: Device) -> Result<Array> {
        let s = expert_indices.shape();
        let n_tokens = s[0];
        let tk = s[1];
        if n_tokens * tk >= SORT_DISPATCH_THRESHOLD {
            return self.forward_sorted(x, expert_indices, device);
        }
        self.forward_broadcast(x, expert_indices, device)
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: index buffers sized from x/indices shapes"
    )]
    fn forward_broadcast(
        &self,
        x: &Array,
        expert_indices: &Array,
        device: Device,
    ) -> Result<Array> {
        // Expand x: [n, hidden] -> [n, 1, 1, hidden].
        let xe = expand_dims(x, -2, device)?;
        let xe = expand_dims(&xe, -2, device)?;

        let gate_raw = self
            .gate_proj
            .gather_forward(&xe, expert_indices, false, device)?;
        let up_raw = self
            .up_proj
            .gather_forward(&xe, expert_indices, false, device)?;

        let gs = gate_raw.shape();
        let gate_3d = gate_raw.reshape(&[gs[0], gs[1], gs[3]], device)?;
        let us = up_raw.shape();
        let up_3d = up_raw.reshape(&[us[0], us[1], us[3]], device)?;
        let gated = clamped_swiglu(&gate_3d, &up_3d, device)?;

        let gd = expand_dims(&gated, -2, device)?;
        let out_raw = self
            .down_proj
            .gather_forward(&gd, expert_indices, false, device)?;
        let os = out_raw.shape();
        out_raw.reshape(&[os[0], os[1], os[3]], device)
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: index buffers sized from x/indices shapes"
    )]
    fn forward_sorted(&self, x: &Array, expert_indices: &Array, device: Device) -> Result<Array> {
        let s = expert_indices.shape();
        let n_tokens = s[0];
        let tk = s[1];
        let total = n_tokens * tk;
        let hidden = x.shape()[1];

        let flat_idx = expert_indices.reshape(&[total], device)?;
        let order = argsort(&flat_idx, device)?.astype(Dtype::I32, device)?;
        let inv_order = argsort(&order, device)?.astype(Dtype::I32, device)?;
        let sorted_idx = flat_idx.take(&order, 0, device)?;

        // token = order // tk; gather x rows -> [total, hidden].
        let tk_arr = scalar_f32(tk as f32).astype(Dtype::I32, device)?;
        let tok_of_order = floor_divide(&order, &tk_arr, device)?;
        let x_sorted = x.take(&tok_of_order, 0, device)?;

        // x [total, 1, hidden] aligns its leading dim with rhs_indices [total].
        let xe = expand_dims(&x_sorted, -2, device)?;

        let gate_raw = self
            .gate_proj
            .gather_forward(&xe, &sorted_idx, true, device)?;
        let up_raw = self
            .up_proj
            .gather_forward(&xe, &sorted_idx, true, device)?;

        let gs = gate_raw.shape();
        let gate_2d = gate_raw.reshape(&[gs[0], gs[2]], device)?;
        let us = up_raw.shape();
        let up_2d = up_raw.reshape(&[us[0], us[2]], device)?;
        let gated = clamped_swiglu(&gate_2d, &up_2d, device)?;

        let gd = expand_dims(&gated, -2, device)?;
        let out_raw = self
            .down_proj
            .gather_forward(&gd, &sorted_idx, true, device)?;
        let os = out_raw.shape();
        let out_2d = out_raw.reshape(&[os[0], os[2]], device)?;

        let out_unsorted = out_2d.take(&inv_order, 0, device)?;
        out_unsorted.reshape(&[n_tokens, tk, hidden], device)
    }
}

/// Sparse MoE FFN: fp32 top-8 router, 256 clamped-SwiGLU experts, no shared expert.
#[allow(missing_debug_implementations)]
pub(super) struct MapleSparseMoeBlock {
    /// Expert router (`mlp.gate.weight`).
    pub(super) gate: MapleGate,
    /// Per-expert SwitchGLU (`mlp.switch_mlp.{gate,up,down}_proj`).
    pub(super) switch: MapleSwitchGLU,
}

impl MapleSparseMoeBlock {
    /// `x`: `[B, S, hidden]` — flattens to `[n_tokens, hidden]` internally.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: x is [B, S, hidden]"
    )]
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let sh = x.shape();
        let (batch, seq, hidden) = (sh[0], sh[1], sh[2]);
        let n_tokens = batch * seq;
        let ne = self.gate.weight.shape()[0];
        let tk = self.gate.top_k;

        let x_flat = x.reshape(&[n_tokens, hidden], device)?;

        let (expert_idx, scores) = {
            let _router_span = tracing::debug_span!(
                "moe_router",
                num_active_experts = tk,
                routing_topk = tk,
                num_experts = ne,
            )
            .entered();
            self.gate.forward(&x_flat, device)?
        };

        let expert_out = self.switch.forward(&x_flat, &expert_idx, device)?;
        let combined = aggregate_expert_outputs(&expert_out, &scores, device)?;
        combined.reshape(&[batch, seq, hidden], device)
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod moe_tests;
