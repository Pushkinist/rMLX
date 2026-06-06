//! Sparse MoE block for the Qwen3-VL text decoder.
//!
//! Plain Qwen3-MoE routing (NO shared expert, unlike the Qwen3-Next block in
//! [`crate::qwen3_5_moe::moe`]):
//! router: `Linear -> softmax -> argpartition top-k -> gather scores ->
//! optional renormalize`; experts: SwiGLU via `gather_qmm`.
//!
//! Faithful to `mlx_vlm/models/qwen3_vl_moe/language.py::Qwen3VLMoESparseMoeBlock`.

#![allow(clippy::struct_field_names)]
use rmlx_core::error::Result;
use rmlx_mlx::{
    argpartition, divide, expand_dims, multiply, silu, softmax, sum_axis, take_along_axis, Array,
    Device, Dtype,
};

use super::layers::Linear;

#[allow(missing_debug_implementations)]
pub(super) struct SwitchMlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl SwitchMlp {
    /// `x`: `[n_tokens, hidden]`, `expert_indices`/`routing_weights`:
    /// `[n_tokens, top_k]`. Returns `[n_tokens, hidden]`. Identical dispatch
    /// math to [`crate::qwen3_5_moe::moe::SwitchMlp::forward`].
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
        let xe = expand_dims(x, -2, device)?;
        let xe = expand_dims(&xe, -2, device)?;

        let gate_raw = self.gate_proj.gather_forward(&xe, expert_indices, device)?;
        let up_raw = self.up_proj.gather_forward(&xe, expert_indices, device)?;

        let gs = gate_raw.shape();
        let gate_3d = gate_raw.reshape(&[gs[0], gs[1], gs[3]], device)?;
        let us = up_raw.shape();
        let up_3d = up_raw.reshape(&[us[0], us[1], us[3]], device)?;
        let gate = silu(&gate_3d, device)?;
        let gated = multiply(&gate, &up_3d, device)?;

        let gd = expand_dims(&gated, -2, device)?;
        let out_raw = self.down_proj.gather_forward(&gd, expert_indices, device)?;
        let os = out_raw.shape();
        let out_3d = out_raw.reshape(&[os[0], os[1], os[3]], device)?;

        let rw = expand_dims(routing_weights, -1, device)?;
        let out_scaled = multiply(&out_3d, &rw, device)?;
        sum_axis(&out_scaled, 1, device)
    }
}

#[allow(missing_debug_implementations)]
pub(super) struct SparseMoeBlock {
    pub(super) gate: Linear,
    pub(super) switch_mlp: SwitchMlp,
    pub(super) num_experts: usize,
    pub(super) top_k: usize,
    pub(super) norm_topk_prob: bool,
}

impl SparseMoeBlock {
    /// `x`: `[B, S, hidden]`. No shared expert.
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

        let logits = self.gate.forward(&x_flat, device)?; // [n, num_experts]
        let gates = softmax(&logits, -1, device)?;

        let part_idx = argpartition(&gates, -(tk as i32), -1, device)?;
        let expert_idx = part_idx.slice(
            &[0, (ne - tk) as i32],
            &[n_tokens, ne as i32],
            &[1, 1],
            device,
        )?;
        let expert_idx_i32 = expert_idx.astype(Dtype::I32, device)?;

        let mut scores = take_along_axis(&gates, &expert_idx_i32, -1, device)?;
        if self.norm_topk_prob {
            let s_sum = sum_axis(&scores, -1, device)?;
            let s_sum = s_sum.reshape(&[n_tokens, 1], device)?;
            scores = divide(&scores, &s_sum, device)?;
        }

        let routed_out = self
            .switch_mlp
            .forward(&x_flat, &expert_idx_i32, &scores, device)?;
        routed_out.reshape(&[batch, seq, hidden], device)
    }
}

/// Dense SwiGLU MLP (used by `mlp_only_layers`, normally empty for the target).
#[allow(missing_debug_implementations)]
pub(super) struct DenseMlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl DenseMlp {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = silu(&self.gate_proj.forward(x, device)?, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}
