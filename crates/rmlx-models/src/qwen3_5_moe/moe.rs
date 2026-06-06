//! Sparse MoE block (SparseMoeBlock), shared expert, switch MLP, and dense SwiGLU MLP.
//!
//! SparseMoeBlock:
//! Router: Linear -> softmax -> argpartition top-K -> gather scores -> optional normalize.
//! SwitchMlp: SwiGLU via gather_qmm (batched over experts).
//! SharedExpert: dense SwiGLU + sigmoid gate.
//!
//! DenseMlp: used by the PARO dense variant (Qwen3_5ForConditionalGeneration).

#![allow(clippy::struct_field_names)]
use rmlx_core::error::Result;
use rmlx_mlx::{
    add, argpartition, divide, expand_dims, multiply, sigmoid, silu, softmax, sum_axis,
    take_along_axis, Array, Device, Dtype,
};

use super::layers::Linear;

#[allow(missing_debug_implementations)]
pub(super) struct SwitchMlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl SwitchMlp {
    /// x: [n_tokens, hidden].
    /// expert_indices: [n_tokens, top_k] (i32).
    /// routing_weights: [n_tokens, top_k].
    /// Returns [n_tokens, hidden].
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        expert_indices: &Array,
        routing_weights: &Array,
        device: Device,
    ) -> Result<Array> {
        // Expand x: [n, hidden] -> [n, 1, 1, hidden].
        let xe = expand_dims(x, -2, device)?;
        let xe = expand_dims(&xe, -2, device)?;

        let gate_raw = self.gate_proj.gather_forward(&xe, expert_indices, device)?;
        let up_raw = self.up_proj.gather_forward(&xe, expert_indices, device)?;

        // Squeeze singleton: [n, tk, 1, inter] -> [n, tk, inter].
        let gs = gate_raw.shape();
        let gate_3d = gate_raw.reshape(&[gs[0], gs[1], gs[3]], device)?;
        let us = up_raw.shape();
        let up_3d = up_raw.reshape(&[us[0], us[1], us[3]], device)?;
        let gate = silu(&gate_3d, device)?;
        let gated = multiply(&gate, &up_3d, device)?;

        // Re-expand for down_proj: [n, tk, inter] -> [n, tk, 1, inter].
        let gd = expand_dims(&gated, -2, device)?;
        let out_raw = self.down_proj.gather_forward(&gd, expert_indices, device)?;
        let os = out_raw.shape();
        let out_3d = out_raw.reshape(&[os[0], os[1], os[3]], device)?;

        // Scale by routing_weights: [n, tk, 1] * [n, tk, hidden].
        let rw = expand_dims(routing_weights, -1, device)?;
        let out_scaled = multiply(&out_3d, &rw, device)?;

        sum_axis(&out_scaled, 1, device)
    }
}

#[allow(missing_debug_implementations)]
pub(super) struct SharedExpert {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl SharedExpert {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = self.gate_proj.forward(x, device)?;
        let gate = silu(&gate, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

#[allow(missing_debug_implementations)]
pub(super) struct SparseMoeBlock {
    pub(super) gate: Linear,
    pub(super) switch_mlp: SwitchMlp,
    pub(super) shared_expert: SharedExpert,
    pub(super) shared_expert_gate: Linear,
    pub(super) num_experts: usize,
    pub(super) top_k: usize,
    pub(super) norm_topk_prob: bool,
}

impl SparseMoeBlock {
    /// x: [B, S, hidden] — block forward flattens to [n_tokens, hidden] internally.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let sh = x.shape();
        let (batch, seq, hidden) = (sh[0], sh[1], sh[2]);
        let n_tokens = batch * seq;
        let ne = self.num_experts;
        let tk = self.top_k;

        let x_flat = x.reshape(&[n_tokens, hidden], device)?;

        // Task 10: MoE router debug span — captures per-step routing cost.
        let router_out = {
            let _router_span = tracing::debug_span!(
                "moe_router",
                num_active_experts = tk,
                routing_topk = tk,
                num_experts = ne,
            )
            .entered();

            // Routing: softmax(gate(x)).
            let logits = self.gate.forward(&x_flat, device)?; // [n, num_experts]
            let gates = softmax(&logits, -1, device)?; // [n, num_experts]

            // top-K via argpartition (O(N) vs argsort's O(N log N)) — matches
            // mlx-lm-turboquant qwen3_next.py:338:
            // inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            // scores = mx.take_along_axis(gates, inds, axis=-1)
            let part_idx = argpartition(&gates, -(tk as i32), -1, device)?; // [n, num_experts]
            let expert_idx = part_idx.slice(
                &[0, (ne - tk) as i32],
                &[n_tokens, ne as i32],
                &[1, 1],
                device,
            )?; // [n, tk]
            let expert_idx_i32 = expert_idx.astype(Dtype::I32, device)?;

            // Gather routing scores using take_along_axis — fewer ops than the
            // manual offsets+take dance (single kernel vs reshape+add+take+reshape).
            let mut scores = take_along_axis(&gates, &expert_idx_i32, -1, device)?;

            if self.norm_topk_prob {
                let s_sum = sum_axis(&scores, -1, device)?;
                let s_sum = s_sum.reshape(&[n_tokens, 1], device)?;
                scores = divide(&scores, &s_sum, device)?;
            }

            (expert_idx_i32, scores)
        };
        let (expert_idx_i32, scores) = router_out;

        // Expert dispatch.
        let routed_out = self
            .switch_mlp
            .forward(&x_flat, &expert_idx_i32, &scores, device)?; // [n, hidden]

        // Shared expert with sigmoid gate.
        let shared_out = self.shared_expert.forward(&x_flat, device)?; // [n, hidden]
        let sg = self.shared_expert_gate.forward(&x_flat, device)?; // [n, 1]
        let sg = sigmoid(&sg, device)?;
        let shared_gated = multiply(&sg, &shared_out, device)?;

        let combined = add(&routed_out, &shared_gated, device)?;
        combined.reshape(&[batch, seq, hidden], device)
    }
}

// ---------------------------------------------------------------------------
// Dense SwiGLU MLP (PARO dense variant)
// ---------------------------------------------------------------------------

/// Dense gate-up-down SwiGLU FFN.
///
/// Used by `Qwen3_5ForConditionalGeneration` (e.g. z-lab PARO 27B).
/// Identical feed-forward structure to the shared expert in `SparseMoeBlock`
/// but without the MoE routing overhead.
#[allow(missing_debug_implementations)]
pub(super) struct DenseMlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl DenseMlp {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = self.gate_proj.forward(x, device)?;
        let gate = silu(&gate, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}
