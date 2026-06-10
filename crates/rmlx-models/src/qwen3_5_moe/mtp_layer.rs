//! Reusable single Qwen3.5-MoE decoder layer for the MTP sidecar drafter.
//!
//! The Qwen3.5 MTP drafter's `layers.0` is a *stock* Qwen3.5-MoE decoder layer
//! (`full_attention_interval = 1`, so the single layer is a full-attention
//! layer + sparse-MoE FFN with identical tensor names to the main model). This
//! module REUSES the existing [`FullAttention`] + [`SparseMoeBlock`] +
//! [`DecoderLayer`] machinery rather than hand-porting a second attention /
//! MoE implementation (CLAUDE.md general-solution mandate).
//!
//! [`MtpLayer`] wraps one loaded [`DecoderLayer`] plus the attention dims the
//! sidecar's round-loop needs (head/kv counts are not exposed by
//! `DecoderLayer`). It is constructed from a sidecar [`ShardSet`] keyed on the
//! `layers.{i}.` prefix; the LM head / embedding stay on the verifier.

use std::collections::HashMap;

use rmlx_core::error::{Error, Result};
use rmlx_loader::ShardSet;
use rmlx_mlx::{Array, Device};

use super::attention::FullAttention;
use super::decoder_layer::{AttnBlock, DecoderLayer, MlpBlock};
use super::layers::{Linear, RmsNorm};
use super::moe::{SharedExpert, SparseMoeBlock, SwitchMlp};
use crate::layers::{resolve_quant, QuantParams};
use rmlx_kv_quant::KvCache;

/// Attention/MoE dimensions for a single reused Qwen3.5-MoE MTP layer.
///
/// All fields are read from the MTP-head `text_config` (which mirrors the
/// verifier's). Kept on [`MtpLayer`] because [`DecoderLayer`] does not expose
/// them and the sidecar round-loop never needs the full [`super::config::Qwen3_5MoeConfig`].
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed dims struct — read once from config and consumed by MtpLayer::load"
)]
#[derive(Debug, Clone)]
pub struct MtpLayerDims {
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Number of rotated dims (`partial_rotary_factor * head_dim`).
    pub rope_dims: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Total MoE experts.
    pub num_experts: usize,
    /// Experts selected per token.
    pub num_experts_per_tok: usize,
    /// Normalize top-k routing weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Quantization group size (sidecar global).
    pub quant_group_size: i32,
    /// Quantization bit-width (sidecar global).
    pub quant_bits: i32,
    /// Quantization mode string (sidecar global).
    pub quant_mode: String,
}

/// One reused Qwen3.5-MoE decoder layer + the attn dims the round-loop needs.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed wrapper — DecoderLayer is the reused unit; adding a field requires updating MtpLayer::load"
)]
#[allow(missing_debug_implementations)]
pub struct MtpLayer {
    layer: DecoderLayer,
}

impl MtpLayer {
    /// Load a single full-attention + sparse-MoE Qwen3.5 decoder layer from a
    /// sidecar shard set, keyed on `prefix` (`layers.0`).
    ///
    /// REUSES [`FullAttention`] + [`SparseMoeBlock`] + [`DecoderLayer`]. The
    /// tensor names match the main Qwen3.5-MoE loader exactly:
    /// `{prefix}.self_attn.{q,k,v,o}_proj`, `{prefix}.self_attn.{q,k}_norm`,
    /// `{prefix}.{input,post_attention}_layernorm`, `{prefix}.mlp.gate`,
    /// `{prefix}.mlp.switch_mlp.{gate,up,down}_proj`,
    /// `{prefix}.mlp.shared_expert.{gate,up,down}_proj`,
    /// `{prefix}.mlp.shared_expert_gate`.
    pub fn load(shards: &ShardSet, prefix: &str, dims: &MtpLayerDims) -> Result<Self> {
        let defaults =
            QuantParams::global(dims.quant_group_size, dims.quant_bits, &dims.quant_mode);
        let overrides: HashMap<String, QuantParams> = HashMap::new();

        let load_array = |name: &str| -> Result<Array> {
            for (_, handle) in shards.iter() {
                let st = handle
                    .safetensors()
                    .map_err(|e| Error::Model(format!("MtpLayer: safetensors: {e}")))?;
                if let Ok(t) = st.tensor(name) {
                    let tv = rmlx_loader::TensorView {
                        name,
                        dtype: t.dtype(),
                        shape: t.shape().to_vec(),
                        bytes: t.data(),
                    };
                    return Array::from_safetensor_view(&tv);
                }
            }
            Err(Error::Model(format!("MtpLayer: tensor '{name}' not found")))
        };
        let has_tensor = |name: &str| -> bool {
            shards
                .iter()
                .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
        };

        let load_linear = |base: &str| -> Result<Linear> {
            let w = load_array(&format!("{base}.weight"))?;
            let s_name = format!("{base}.scales");
            if has_tensor(&s_name) {
                let s = load_array(&s_name)?;
                let biases = if has_tensor(&format!("{base}.biases")) {
                    Some(load_array(&format!("{base}.biases"))?)
                } else {
                    None
                };
                // The shared resolver owns the `.biases`-sibling affine rule.
                let qp = resolve_quant(base, biases.is_some(), &defaults, &overrides)?;
                Ok(Linear::Quantized {
                    weight: w,
                    scales: s,
                    biases,
                    group_size: qp.group_size,
                    bits: qp.bits,
                    mode: qp.mode,
                })
            } else {
                Ok(Linear::Plain { weight: w })
            }
        };
        let load_rms = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm {
                weight: load_array(&format!("{name}.weight"))?,
                eps: dims.rms_norm_eps,
            })
        };

        let sa = format!("{prefix}.self_attn");
        let attn_scale = (dims.head_dim as f32).powf(-0.5);
        let attn = AttnBlock::Full(FullAttention {
            q_proj: load_linear(&format!("{sa}.q_proj"))?,
            k_proj: load_linear(&format!("{sa}.k_proj"))?,
            v_proj: load_linear(&format!("{sa}.v_proj"))?,
            o_proj: load_linear(&format!("{sa}.o_proj"))?,
            q_norm: load_rms(&format!("{sa}.q_norm"))?,
            k_norm: load_rms(&format!("{sa}.k_norm"))?,
            n_heads: dims.num_attention_heads,
            n_kv_heads: dims.num_key_value_heads,
            head_dim: dims.head_dim,
            scale: attn_scale,
            rope_theta: dims.rope_theta,
            rope_dims: dims.rope_dims,
        });

        let m = format!("{prefix}.mlp");
        let mlp = MlpBlock::Moe(Box::new(SparseMoeBlock {
            gate: load_linear(&format!("{m}.gate"))?,
            switch_mlp: SwitchMlp {
                gate_proj: load_linear(&format!("{m}.switch_mlp.gate_proj"))?,
                up_proj: load_linear(&format!("{m}.switch_mlp.up_proj"))?,
                down_proj: load_linear(&format!("{m}.switch_mlp.down_proj"))?,
            },
            shared_expert: SharedExpert {
                gate_proj: load_linear(&format!("{m}.shared_expert.gate_proj"))?,
                up_proj: load_linear(&format!("{m}.shared_expert.up_proj"))?,
                down_proj: load_linear(&format!("{m}.shared_expert.down_proj"))?,
            },
            shared_expert_gate: load_linear(&format!("{m}.shared_expert_gate"))?,
            num_experts: dims.num_experts,
            top_k: dims.num_experts_per_tok,
            norm_topk_prob: dims.norm_topk_prob,
        }));

        let layer = DecoderLayer {
            input_layernorm: load_rms(&format!("{prefix}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{prefix}.post_attention_layernorm"))?,
            attn,
            mlp,
        };
        Ok(Self { layer })
    }

    /// Run the reused decoder layer over the sidecar's own KV cache.
    ///
    /// `x`: `[1, n, H]`. `offset` is the RoPE / KV write position (the sidecar's
    /// `_next_position`). The layer is full-attention (no GDN), so `lin_cache`
    /// is `None`; single-token decode builds its own mask internally
    /// (`prebuilt_mask = None`). Returns `[1, n, H]`.
    pub fn forward(
        &self,
        x: &Array,
        offset: i32,
        kv_cache: &mut KvCache,
        device: Device,
    ) -> Result<Array> {
        self.layer
            .forward(x, offset, Some(kv_cache), None, None, device)
    }
}
