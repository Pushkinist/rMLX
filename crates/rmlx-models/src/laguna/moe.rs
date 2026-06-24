//! Laguna sparse MoE block: router, experts, shared expert.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for offset arrays
#![allow(unsafe_code)]
#![allow(clippy::struct_field_names)]
use rmlx_core::error::Result;
use rmlx_mlx::{add, argsort, multiply, sigmoid, silu, sum_axis, Array, Device, Dtype};

use super::layers::{DenseMlp, Linear};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Top-K sigmoid router (LagunaTopKRouter).
#[allow(missing_debug_implementations)]
pub(super) struct Router {
    pub(super) gate_proj: Linear,
    pub(super) e_score_correction_bias: Array,
    pub(super) num_experts: usize,
    pub(super) top_k: usize,
}

impl Router {
    /// x: [n_tokens, hidden].
    /// Returns (routing_weights [n_tokens, top_k], expert_indices [n_tokens, top_k]).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<(Array, Array)> {
        let n_tokens = x.shape()[0] as usize;
        let ne = self.num_experts;
        let tk = self.top_k;

        // Sigmoid scores: [n_tokens, num_experts]
        let logits = self.gate_proj.forward(x, device)?;
        let scores = sigmoid(&logits, device)?;

        // Selection scores: scores + correction_bias.
        let bias = self
            .e_score_correction_bias
            .reshape(&[1, ne as i32], device)?;
        let select_scores = add(&scores, &bias, device)?;

        // Argsort ascending -> last top_k columns are the top-k experts.
        let sorted_idx = argsort(&select_scores, device)?; // [n_tokens, ne]
        let expert_idx = sorted_idx.slice(
            &[0, (ne - tk) as i32],
            &[n_tokens as i32, ne as i32],
            &[1, 1],
            device,
        )?; // [n_tokens, top_k]

        // Gather routing weights at selected expert positions.
        // Build flat linear index: token_i * ne + expert_idx[i, j].
        let mut offset_data = vec![0i32; n_tokens * tk];
        for i in 0..n_tokens {
            for j in 0..tk {
                offset_data[i * tk + j] = (i * ne) as i32;
            }
        }
        let offset_bytes = unsafe {
            std::slice::from_raw_parts(offset_data.as_ptr().cast::<u8>(), offset_data.len() * 4)
        };
        let offsets = Array::from_bytes(offset_bytes, &[(n_tokens * tk) as i32], Dtype::I32)?;

        let expert_idx_flat = expert_idx.reshape(&[(n_tokens * tk) as i32], device)?;
        let expert_idx_i32 = expert_idx_flat.astype(Dtype::I32, device)?;
        let flat_idx = add(&expert_idx_i32, &offsets, device)?;

        let scores_flat = scores.reshape(&[(n_tokens * ne) as i32], device)?;
        let rw_flat = scores_flat.take(&flat_idx, 0, device)?; // [n_tokens*top_k]

        // Normalise per token.
        let rw_mat = rw_flat.reshape(&[n_tokens as i32, tk as i32], device)?;
        let rw_sum = sum_axis(&rw_mat, -1, device)?; // [n_tokens]
        let rw_sum = rw_sum.reshape(&[n_tokens as i32, 1], device)?;
        let rw_norm = rmlx_mlx::divide(&rw_mat, &rw_sum, device)?;

        // expert_idx: [n_tokens, top_k] (2D, matching rw_norm shape)
        let expert_idx_2d = expert_idx_i32.reshape(&[n_tokens as i32, tk as i32], device)?;

        Ok((rw_norm, expert_idx_2d))
    }
}

// ---------------------------------------------------------------------------
// SwitchExperts
// ---------------------------------------------------------------------------

/// Batched experts with 3-D weight tensors [num_experts, out, in_packed].
#[allow(missing_debug_implementations)]
pub(super) struct SwitchExperts {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl SwitchExperts {
    /// Follows the mlx-lm Python pattern from switch_layers.py.
    ///
    /// x: [n_tokens, hidden].
    /// routing_weights: [n_tokens, top_k] (normalised per-token weights).
    /// expert_indices: [n_tokens, top_k] (which experts to dispatch to).
    /// Returns [n_tokens, hidden] accumulated output.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        routing_weights: &Array,
        expert_indices: &Array,
        device: Device,
    ) -> Result<Array> {
        // Expand x: [n, hidden] -> [n, 1, 1, hidden] (as Python: expand_dims(-2,-3))
        let xe = rmlx_mlx::expand_dims(x, -2, device)?; // [n, 1, hidden]
        let xe = rmlx_mlx::expand_dims(&xe, -2, device)?; // [n, 1, 1, hidden]

        // gate_proj output: rhs_indices[n,tk].shape + [xe.shape(-2)=1, inter] = [n, tk, 1, inter]
        let gate_raw = self.gate_proj.gather_forward(&xe, expert_indices, device)?;
        let up_raw = self.up_proj.gather_forward(&xe, expert_indices, device)?;

        // Squeeze the singleton dim -2: [n, tk, 1, inter] -> [n, tk, inter]
        let s = gate_raw.shape();
        let gate_3d = gate_raw.reshape(&[s[0], s[1], s[3]], device)?;
        let s = up_raw.shape();
        let up_3d = up_raw.reshape(&[s[0], s[1], s[3]], device)?;
        let gate = silu(&gate_3d, device)?;
        let gated = multiply(&gate, &up_3d, device)?; // [n, tk, inter]

        // Re-expand gated for down_proj: [n, tk, inter] -> [n, tk, 1, inter]
        // Need exactly 4D so that gd.shape()[:-2] = [n, tk] matches rhs_indices[n,tk].
        let gd = rmlx_mlx::expand_dims(&gated, -2, device)?; // [n, tk, 1, inter]

        // down_proj: rhs_indices[n,tk].shape + [gd.shape(-2)=1, hidden] = [n, tk, 1, hidden]
        let out_raw = self.down_proj.gather_forward(&gd, expert_indices, device)?;
        let s = out_raw.shape();
        let out_3d = out_raw.reshape(&[s[0], s[1], s[3]], device)?; // [n, tk, hidden]

        // Scale by routing weights: [n, tk, 1] * [n, tk, hidden]
        let rw = rmlx_mlx::expand_dims(routing_weights, -1, device)?; // [n, tk, 1]
        let out_scaled = multiply(&out_3d, &rw, device)?; // [n, tk, hidden]

        // Sum over top_k axis: [n, hidden]
        sum_axis(&out_scaled, 1, device)
    }
}

// ---------------------------------------------------------------------------
// SparseMoeBlock
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct SparseMoeBlock {
    pub(super) router: Router,
    pub(super) experts: SwitchExperts,
    pub(super) shared_expert: DenseMlp,
    pub(super) routed_scaling_factor: f32,
}

impl SparseMoeBlock {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let hidden = shape[2];

        let x_flat = x.reshape(&[batch * seq, hidden], device)?;

        let shared_out = self.shared_expert.forward(&x_flat, device)?;

        let (routing_weights, expert_indices) = self.router.forward(&x_flat, device)?;

        let routed_out =
            self.experts
                .forward(&x_flat, &routing_weights, &expert_indices, device)?;

        let scale_arr =
            rmlx_mlx::scalar_f32(self.routed_scaling_factor).astype(routed_out.dtype(), device)?;
        let routed_scaled = multiply(&routed_out, &scale_arr, device)?;
        let combined = add(&routed_scaled, &shared_out, device)?;

        combined.reshape(&[batch, seq, hidden], device)
    }
}
