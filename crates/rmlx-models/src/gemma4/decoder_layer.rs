//! Gemma4 decoder layer.

use rmlx_core::error::Result;
use rmlx_mlx::{add, multiply, Array, Device};
use tracing::debug_span;

use crate::layers::{Mlp, RmsNorm};
use rmlx_kv_quant::{KvCache, SharedKvOut};

use super::layers::{geglu_fused, Attention, Gemma4MoeBlock, PerLayerInput};

/// MLP forward with mx.compile-fused GeGLU activation.
///
/// Identical math to `Mlp::forward` for `Activation::GeluTanh`:
/// down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))
///
/// The `gelu_tanh(...) * up` block (9 pointwise ops) is fused into a single
/// compiled Metal program via [`geglu_fused`]. The matmuls for gate/up/down
/// stay un-fused (matmul kernels can't be pointwise-fused).
fn mlp_forward_fused(mlp: &Mlp, x: &Array, device: Device) -> Result<Array> {
    let gate = mlp.gate_proj.forward(x, device)?;
    let up = mlp.up_proj.forward(x, device)?;
    let gated = geglu_fused(&gate, &up, device)?;
    mlp.down_proj.forward(&gated, device)
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct DecoderLayer {
    pub(super) input_norm: RmsNorm,
    pub(super) post_attn_norm: RmsNorm,
    pub(super) pre_ffn_norm: RmsNorm,
    pub(super) post_ffn_norm: RmsNorm,
    pub(super) attn: Attention,
    pub(super) mlp: Mlp,
    /// Present when `enable_moe_block=true` (26B model).
    pub(super) moe_block: Option<Gemma4MoeBlock>,
    pub(super) per_layer: Option<PerLayerInput>,
    pub(super) layer_scalar: Option<Array>, // shape [1] bf16 or f32
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        shared_kv: Option<&SharedKvOut>,
        per_layer_input: Option<&Array>,
        offset: i32,
        cache: Option<&mut KvCache>,
        kv_is_rotating: bool,
        device: Device,
    ) -> Result<(Array, Option<SharedKvOut>)> {
        // Attention sub-layer.
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let (h, new_kv) =
            self.attn
                .forward(&h, shared_kv, offset, cache, kv_is_rotating, device)?;
        let h = self.post_attn_norm.forward(&h, device)?;
        let h = add(&residual, &h, device)?;

        // FFN sub-layer.
        let residual = h.try_clone()?;
        let h = if let Some(moe) = &self.moe_block {
            // MoE path: dense MLP + sparse experts in parallel, summed then normed.
            // Reference: gemma4_text.py DecoderLayer.__call__ lines 353-368.
            //
            // Dense path:
            let h1 = self.pre_ffn_norm.forward(&h, device)?;
            let h1 = mlp_forward_fused(&self.mlp, &h1, device)?;
            let h1 = moe.post_ffn_norm_1.forward(&h1, device)?;
            //
            // Sparse path: router runs on un-normed h, experts on pre_ffn_norm_2(h).
            let shape = h.shape();
            let batch = shape[0];
            let seq = shape[1];
            let hidden = shape[2];
            let h_flat = h.reshape(&[batch * seq, hidden], device)?;
            // Task 10: Gemma4 MoE router debug span.
            let (expert_indices, routing_weights) = {
                let _router_span = debug_span!(
                    "moe_router",
                    num_active_experts = moe.router.top_k,
                    routing_topk = moe.router.top_k,
                    num_experts = moe.router.num_experts,
                    arch = "Gemma4",
                )
                .entered();
                moe.router.forward(&h_flat, device)?
            };
            let h2_in = moe.pre_ffn_norm_2.forward(&h, device)?;
            let h2_in_flat = h2_in.reshape(&[batch * seq, hidden], device)?;
            let h2 = moe
                .experts
                .forward(&h2_in_flat, &expert_indices, &routing_weights, device)?;
            let h2 = h2.reshape(&[batch, seq, hidden], device)?;
            let h2 = moe.post_ffn_norm_2.forward(&h2, device)?;
            //
            // Sum and apply overall post_ffn_norm.
            let h_sum = add(&h1, &h2, device)?;
            let h_sum = self.post_ffn_norm.forward(&h_sum, device)?;
            add(&residual, &h_sum, device)?
        } else {
            // Dense-only path (4B / e4b / 31b model).
            let h = self.pre_ffn_norm.forward(&h, device)?;
            let h = mlp_forward_fused(&self.mlp, &h, device)?;
            let h = self.post_ffn_norm.forward(&h, device)?;
            add(&residual, &h, device)?
        };

        // Per-layer input gating.
        let h = if let (Some(pli), Some(per_layer)) = (&self.per_layer, per_layer_input) {
            let residual = h.try_clone()?;
            let gated = pli.forward(&h, per_layer, device)?;
            add(&residual, &gated, device)?
        } else {
            h
        };

        // Layer scalar (element-wise scale).
        let h = if let Some(scalar) = &self.layer_scalar {
            multiply(&h, scalar, device)?
        } else {
            h
        };

        Ok((h, new_kv))
    }
}
