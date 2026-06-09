//! Gemma4 MoE block: PerLayerInput, Gemma4Router, Gemma4Experts, Gemma4MoeBlock.

// unsafe_code: slice::from_raw_parts byte-reinterpret for Array::from_bytes in gather_forward.
#![allow(unsafe_code)]

use rmlx_core::error::Result;
use rmlx_mlx::{
    argpartition, argsort, expand_dims, floor_divide, multiply, rms_norm, scalar_f32, softmax,
    sum_axis, take_along_axis, Array, Device, Dtype,
};

use crate::layers::{Linear, RmsNorm};

use super::kernels::{geglu_fused, pli_gelu_fused};

/// Flattened `n_tokens * top_k` count at or above which the expert dispatch
/// sorts indices for contiguous per-expert access (prefill). Below it (decode,
/// single token) the simple broadcast path is cheaper. Matches mlx-lm SwitchGLU.
const SORT_DISPATCH_THRESHOLD: i32 = 64;

#[allow(missing_debug_implementations)]
pub(crate) struct PerLayerInput {
    pub(crate) gate: Linear, // [hidden, hidden_per_layer] — projects h to per_layer dim
    pub(crate) projection: Linear, // [hidden_per_layer, hidden] — projects gated back to hidden
    pub(crate) post_norm: RmsNorm,
}

impl PerLayerInput {
    /// Compute gated per-layer contribution (does NOT add residual).
    ///
    /// Returns `gate` only. The caller (DecoderLayer) owns the residual add:
    /// `h = h + gate`
    ///
    /// Reference: gemma4_text.py DecoderLayer.__call__ lines 379-384:
    /// gate = per_layer_input_gate(h); gate = gelu_approx(gate)
    /// gate = gate * per_layer_input; gate = per_layer_projection(gate)
    /// gate = post_per_layer_input_norm(gate)
    /// h = residual + gate ← residual add is outside this function
    ///
    /// Fix: was returning `add(h, &gate)` (built-in residual), causing
    /// DecoderLayer to add residual a second time → 2h + gate.
    /// Verified against mlx-lm/mlx_lm/models/gemma4_text.py:379-384.
    pub(crate) fn forward(&self, h: &Array, per_layer: &Array, device: Device) -> Result<Array> {
        let gate = self.gate.forward(h, device)?;
        // Fuse gelu_tanh(gate) * per_layer into one Metal program.
        // Drops 8 separate pointwise kernel launches (gelu_tanh) + 1 multiply = 9
        // launches → 1 fused launch per PLI call per decode step.
        let gate = pli_gelu_fused(&gate, per_layer, device)?;
        let gate = self.projection.forward(&gate, device)?;
        self.post_norm.forward(&gate, device)
    }
}

// ---------------------------------------------------------------------------
// Gemma4 MoE block (26B / a4b model — enable_moe_block=true)
// ---------------------------------------------------------------------------
//
// Reference: gemma4_text.py DecoderLayer.__call__ MoE path (lines 353-368):
// h1 = post_ffn_norm_1(mlp(pre_ffn_norm(h)))
// top_k_indices, top_k_weights = router(h)
// h2 = post_ffn_norm_2(experts(pre_ffn_norm_2(h), top_k_indices, top_k_weights))
// h = h1 + h2
// h = post_ffn_norm(residual + h)
//
// Router (gemma4_text.py Router.__call__):
// x = rms_norm(x, scale * root_size, eps) # norm by root(hidden)
// expert_scores = proj(x)
// top_k_indices = argpartition(expert_scores, kth=-top_k)[..., -top_k:]
// top_k_weights = softmax(expert_scores[top_k_indices])
// top_k_weights *= per_expert_scale[top_k_indices]
//
// Experts: SwitchGLU — 3D gathered quantized matmul via gather_qmm.
// Tensor names after sanitize:
// experts.switch_glu.{gate,up,down}_proj.{weight,scales}
// Shape: [num_experts, out, packed_in] (mxfp8, no biases).
//
// Differences from Laguna MoE:
// - Laguna: sigmoid router + e_score_correction_bias + argsort topK
// - Gemma4: RMSNorm input → softmax topK + per_expert_scale weighting
// - Laguna: shared dense expert always present
// - Gemma4: no shared expert (a4b model); MoE and dense MLP both run in parallel
// - Laguna: single post_attn + post_mlp norms; Gemma4 MoE has 3 extra norms

/// Gemma4 expert router.
///
/// Applies RMSNorm (with learned scale and 1/sqrt(hidden) factor), projects to
/// num_experts, takes softmax top-K, then scales by per-expert bias weights.
#[allow(missing_debug_implementations)]
pub(crate) struct Gemma4Router {
    pub(crate) proj: Linear,
    /// Learned scale for the input RMSNorm. Shape [hidden].
    pub(crate) scale: Array,
    /// Per-expert post-softmax scale. Shape [num_experts].
    pub(crate) per_expert_scale: Array,
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) root_size: f32, // = hidden^-0.5
    pub(crate) eps: f32,
}

impl Gemma4Router {
    /// Returns (top_k_indices [n_tokens, top_k], top_k_weights [n_tokens, top_k]).
    ///
    /// Byte-for-byte port of mlx-lm `gemma4_text.py::Router.__call__`:
    /// ```python
    /// x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
    /// expert_scores = self.proj(x)
    /// top_k_indices = mx.argpartition(expert_scores, kth=-tk, axis=-1)[..., -tk:]
    /// top_k_weights = mx.take_along_axis(expert_scores, top_k_indices, axis=-1)
    /// top_k_weights = mx.softmax(top_k_weights, axis=-1) # softmax over top_k!
    /// top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
    /// ```
    ///
    /// Prior rMLX impl had two divergences from mlx-lm (now fixed):
    ///
    /// 1. `argsort` (O(N log N) full sort) instead of `argpartition` (O(N)).
    /// 2. `softmax` over the full ne=128 score vector then gather, instead of
    ///
    /// Also drops the per-call CPU-side `vec![0; n*tk]` heap alloc + the
    /// `Array::from_bytes` host→device i32 transfer (about 5 μs
    /// per launch + 3 μs per transfer × 16 MoE layers/step), saving about 7
    /// fewer dispatches + 1 fewer host transfer per MoE layer per step.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn forward(&self, x: &Array, device: Device) -> Result<(Array, Array)> {
        let shape = x.shape();
        let n_tokens = shape[0];
        let ne = self.num_experts as i32;
        let tk = self.top_k as i32;

        // RMSNorm: pre-scaled weight = self.scale * root_size (root_size = hidden^-0.5).
        // mlx-lm: `mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)`.
        // The `scale * root_size` could be baked at load-time (sub-1% gain);
        // keeping per-call to preserve weight tying with checkpoint.
        let rs = scalar_f32(self.root_size);
        let scaled_weight = multiply(&self.scale, &rs, device)?;
        let x_normed = rms_norm(x, Some(&scaled_weight), self.eps, device)?;

        // Project to expert scores: [n_tokens, num_experts].
        let expert_scores = self.proj.forward(&x_normed, device)?;

        // top_k_indices = argpartition(scores, -tk, axis=-1)[..., -tk:]
        // argpartition guarantees `expert_scores[..., -tk:]` are the top-K
        // unsorted; the relative order within the top-K matches mlx-lm
        // (also unsorted, since `mx.argpartition` returns the same partition).
        let part_idx = argpartition(&expert_scores, -tk, -1, device)?; // [n, ne]
        let expert_idx = part_idx.slice(&[0, ne - tk], &[n_tokens, ne], &[1, 1], device)?; // [n, tk]
        let expert_idx_i32 = expert_idx.astype(Dtype::I32, device)?;

        // top_k_weights = take_along_axis(scores, top_k_indices, axis=-1)
        let top_k_scores = take_along_axis(&expert_scores, &expert_idx_i32, -1, device)?; // [n, tk]

        // top_k_weights = softmax(top_k_weights, axis=-1)
        // Softmax over **only the top_k=8 values** (mlx-lm semantics). Crucial:
        // this normalizes the K selected experts to sum to 1, vs the prior
        // rMLX softmax-over-128-then-gather which produced <1 sum across K.
        let top_k_weights = softmax(&top_k_scores, -1, device)?;

        // top_k_weights = top_k_weights * per_expert_scale[top_k_indices]
        // mlx-lm uses fancy indexing `self.per_expert_scale[top_k_indices]`
        // which corresponds to `take(per_expert_scale, indices, axis=0)` since
        // per_expert_scale is 1-D. Result shape matches indices: [n, tk].
        let pes_flat = self.per_expert_scale.take(
            &expert_idx_i32.reshape(&[n_tokens * tk], device)?,
            0,
            device,
        )?;
        let pes = pes_flat.reshape(&[n_tokens, tk], device)?;
        let weights = multiply(&top_k_weights, &pes, device)?;

        Ok((expert_idx_i32, weights))
    }
}

/// Gemma4 sparse expert FFN using SwitchGLU with 3-D gathered matmuls.
///
/// Expert tensors: `[num_experts, out, packed_in]` after sanitize splits gate_up_proj.
#[allow(missing_debug_implementations)]
pub(crate) struct Gemma4Experts {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
}

impl Gemma4Experts {
    /// x: [n_tokens, hidden].
    /// expert_indices: [n_tokens, top_k] I32.
    /// routing_weights: [n_tokens, top_k].
    /// Returns [n_tokens, hidden].
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn forward(
        &self,
        x: &Array,
        expert_indices: &Array,
        routing_weights: &Array,
        device: Device,
    ) -> Result<Array> {
        let s = expert_indices.shape();
        let n_tokens = s[0];
        let tk = s[1];

        // Multi-token (prefill) fast path: sort the flattened expert indices so
        // each expert's rows are contiguous, then dispatch gather_qmm with
        // sorted_indices=true and scatter the outputs back. Contiguous expert
        // access lets the gathered-matmul kernel run each expert as one dense
        // block instead of scattered per-token gathers. Mirrors mlx-lm
        // SwitchGLU (threshold `indices.size >= 64`). Decode (n_tokens=1,
        // tk*1 < 64) keeps the simple broadcast path below.
        if n_tokens * tk >= SORT_DISPATCH_THRESHOLD {
            return self.forward_sorted(x, expert_indices, routing_weights, device);
        }
        self.forward_broadcast(x, expert_indices, routing_weights, device)
    }

    /// Broadcast (per-token gather) expert forward. Used for the decode /
    /// single-token case; also the equivalence reference for `forward_sorted`
    /// (the two produce mathematically identical output for any routing).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: index buffers sized from x/indices shapes"
    )]
    fn forward_broadcast(
        &self,
        x: &Array,
        expert_indices: &Array,
        routing_weights: &Array,
        device: Device,
    ) -> Result<Array> {
        // Expand x: [n, hidden] -> [n, 1, 1, hidden]
        let xe = expand_dims(x, -2, device)?; // [n, 1, hidden]
        let xe = expand_dims(&xe, -2, device)?; // [n, 1, 1, hidden]

        // gate: [n, tk, 1, inter]; up: same.
        let gate_raw = self
            .gate_proj
            .gather_forward(&xe, expert_indices, false, device)?;
        let up_raw = self
            .up_proj
            .gather_forward(&xe, expert_indices, false, device)?;

        let s = gate_raw.shape();
        let gate_3d = gate_raw.reshape(&[s[0], s[1], s[3]], device)?;
        let s = up_raw.shape();
        let up_3d = up_raw.reshape(&[s[0], s[1], s[3]], device)?;

        // GeGLU: gelu_tanh(gate) * up — fused via mx.compile.
        // Reuses the dense-path geglu_fused closure (compile_shapeless ⇒ shape-
        // agnostic, dtype/device-keyed cache shared across DenseMlp + Experts).
        // Fuses 9 pointwise kernels into a single Metal program.
        let gated = geglu_fused(&gate_3d, &up_3d, device)?; // [n, tk, inter]

        // Re-expand for down_proj.
        let gd = expand_dims(&gated, -2, device)?; // [n, tk, 1, inter]
        let out_raw = self
            .down_proj
            .gather_forward(&gd, expert_indices, false, device)?;
        let s = out_raw.shape();
        let out_3d = out_raw.reshape(&[s[0], s[1], s[3]], device)?; // [n, tk, hidden]

        // Scale by routing weights: [n, tk, 1] * [n, tk, hidden].
        let rw = expand_dims(routing_weights, -1, device)?;
        let out_scaled = multiply(&out_3d, &rw, device)?;

        // Sum over top_k: [n, hidden].
        sum_axis(&out_scaled, 1, device)
    }

    /// Sorted-dispatch expert forward for the multi-token (prefill) case.
    ///
    /// Sorts the flattened `[n*tk]` expert indices ascending, gathers the
    /// matching x rows into expert-contiguous order, runs the three gathered
    /// quantized matmuls with `sorted_indices=true`, then scatters the result
    /// back to token order. Math is identical to the broadcast path; only the
    /// memory-access order into the expert weights changes.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: index buffers sized from x/indices shapes"
    )]
    fn forward_sorted(
        &self,
        x: &Array,
        expert_indices: &Array,
        routing_weights: &Array,
        device: Device,
    ) -> Result<Array> {
        let s = expert_indices.shape();
        let n_tokens = s[0];
        let tk = s[1];
        let total = n_tokens * tk;
        let hidden = x.shape()[1];

        // order = argsort(flat_indices); inv_order = argsort(order).
        let flat_idx = expert_indices.reshape(&[total], device)?;
        let order = argsort(&flat_idx, device)?.astype(Dtype::I32, device)?;
        let inv_order = argsort(&order, device)?.astype(Dtype::I32, device)?;

        // Sorted expert ids: [total]. take(flat_idx, order).
        let sorted_idx = flat_idx.take(&order, 0, device)?;

        // Sorted token rows: token = order // tk; gather x rows -> [total, hidden].
        let tk_arr = scalar_f32(tk as f32).astype(Dtype::I32, device)?;
        let tok_of_order = floor_divide(&order, &tk_arr, device)?;
        let x_sorted = x.take(&tok_of_order, 0, device)?; // [total, hidden]

        // Shape for gather_qmm: x [total, 1, hidden] aligns its leading `total`
        // dim with the 1-D `rhs_indices [total]` (element-wise expert per row).
        let xe = expand_dims(&x_sorted, -2, device)?; // [total, 1, hidden]

        let gate_raw = self
            .gate_proj
            .gather_forward(&xe, &sorted_idx, true, device)?; // [total, 1, inter]
        let up_raw = self
            .up_proj
            .gather_forward(&xe, &sorted_idx, true, device)?;

        let s = gate_raw.shape();
        let gate_2d = gate_raw.reshape(&[s[0], s[2]], device)?; // [total, inter]
        let s = up_raw.shape();
        let up_2d = up_raw.reshape(&[s[0], s[2]], device)?;

        let gated = geglu_fused(&gate_2d, &up_2d, device)?; // [total, inter]

        let gd = expand_dims(&gated, -2, device)?; // [total, 1, inter]
        let out_raw = self
            .down_proj
            .gather_forward(&gd, &sorted_idx, true, device)?; // [total, 1, hidden]
        let s = out_raw.shape();
        let out_2d = out_raw.reshape(&[s[0], s[2]], device)?; // [total, hidden]

        // Scatter back to token order, then reshape to [n, tk, hidden].
        let out_unsorted = out_2d.take(&inv_order, 0, device)?; // [total, hidden]
        let out_3d = out_unsorted.reshape(&[n_tokens, tk, hidden], device)?;

        // Scale by routing weights and sum over top_k: [n, hidden].
        let rw = expand_dims(routing_weights, -1, device)?;
        let out_scaled = multiply(&out_3d, &rw, device)?;
        sum_axis(&out_scaled, 1, device)
    }
}

/// Gemma4 Linear with optional gather dispatch for 3-D expert tensors.
impl Linear {
    /// Batched expert dispatch via gather_qmm for 3-D weight tensors.
    ///
    /// Same pattern as Laguna's Linear::gather_forward.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn gather_forward(
        &self,
        x: &Array,
        rhs_indices: &Array,
        sorted_indices: bool,
        device: Device,
    ) -> Result<Array> {
        match self {
            Linear::Plain { weight } => {
                // Plain fallback: gather weight rows indexed by rhs_indices, then matmul.
                let s = x.shape(); // [..., 1, 1, hidden]
                let nd = s.len();
                let n_tokens = s[..nd - 2].iter().product::<i32>();
                let tk = rhs_indices.shape()[rhs_indices.shape().len() - 1];
                let idx_flat = rhs_indices.reshape(&[n_tokens * tk], device)?;
                let w_sel = weight.take(&idx_flat, 0, device)?;
                let x_flat = x.reshape(&[n_tokens, s[nd - 1]], device)?;
                // Build token replication index.
                let mut tok_data = vec![0i32; (n_tokens * tk) as usize];
                for i in 0..n_tokens as usize {
                    for j in 0..tk as usize {
                        tok_data[i * tk as usize + j] = i as i32;
                    }
                }
                let tok_bytes = unsafe {
                    std::slice::from_raw_parts(tok_data.as_ptr().cast::<u8>(), tok_data.len() * 4)
                };
                let tok_idx = Array::from_bytes(tok_bytes, &[n_tokens * tk], Dtype::I32)?;
                let x_sel = x_flat.take(&tok_idx, 0, device)?;
                let out = rmlx_mlx::matmul(&x_sel, &w_sel.transpose(&[0, 2, 1], device)?, device)?;
                let rhs_s = rhs_indices.shape();
                let mut out_s: Vec<i32> = rhs_s.to_vec();
                out_s.push(1);
                out_s.push(out.shape()[1]);
                out.reshape(&out_s, device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases: _,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::gather_qmm(
                x,
                weight,
                scales,
                None, // expert tensors are mxfp8, no biases
                None,
                rhs_indices,
                *group_size,
                *bits,
                mode.as_str(),
                sorted_indices,
                device,
            ),
            // PARO layers do not appear as MoE expert projections in Gemma4 PARO (dense MoE path
            // not used). Fall back to sequential forward: flatten [n, tk, 1, hidden] → matmul.
            Linear::Paro { .. } => {
                let s = x.shape();
                let nd = s.len();
                let hidden_in = s[nd - 1];
                let tk = rhs_indices.shape()[rhs_indices.shape().len() - 1];
                let n_tokens: i32 = s[..nd - 2].iter().product();
                let x_2d = x.reshape(&[n_tokens, hidden_in], device)?;
                let out = self.forward(&x_2d, device)?;
                let out_dim = out.shape()[out.shape().len() - 1];
                out.reshape(&[n_tokens, tk, 1, out_dim], device)
            }
        }
    }
}

/// Gemma4 MoE block: dense MLP + sparse expert block run in parallel.
/// The outputs are separately post-normed and summed.
#[allow(missing_debug_implementations)]
pub(crate) struct Gemma4MoeBlock {
    pub(crate) router: Gemma4Router,
    pub(crate) experts: Gemma4Experts,
    /// Additional norms for the MoE path (post_ffn_norm_1, post_ffn_norm_2, pre_ffn_norm_2).
    pub(crate) post_ffn_norm_1: RmsNorm,
    pub(crate) post_ffn_norm_2: RmsNorm,
    pub(crate) pre_ffn_norm_2: RmsNorm,
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod moe_tests;
