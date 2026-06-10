//! Laguna model loader.

#![allow(clippy::cognitive_complexity, clippy::too_many_lines)]
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::{zeros, Array, Device, Dtype};
use tracing::{debug, info, warn};

use crate::layers::{resolve_quant, QuantParams};

use super::attention::Attention;
use super::config::{LagunaConfig, MlpKind};
use super::decoder_layer::{DecoderLayer, Mlp};
use super::layers::{DenseMlp, Embedding, Linear, RmsNorm};
use super::model::LagunaText;
use super::moe::{Router, SparseMoeBlock, SwitchExperts};

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a Laguna model from a snapshot directory.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn load_from_path(model_dir: &Path) -> Result<LagunaText> {
    let cfg_raw = load_config(model_dir)?;

    // Re-read config.json as raw JSON to extract inline quant overrides before
    // Serde struct-parsing drops unknown keys inside the quantization dict.
    let raw_json: serde_json::Value = {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
        serde_json::from_slice(&data)
            .map_err(|e| Error::Loader(format!("malformed config.json: {e}")))?
    };
    let raw_quant = raw_json.get("quantization");

    let cfg = LagunaConfig::from_model_config(&cfg_raw, raw_quant)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        num_experts = cfg.num_experts,
        quant_mode = %cfg.quant_mode,
        quant_group_size = cfg.quant_group_size,
        quant_overrides = cfg.quant_overrides.len(),
        "Laguna: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    let load_array = |name: &str| -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle.safetensors()?;
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
        Err(Error::Loader(format!(
            "tensor '{name}' not found in any shard"
        )))
    };

    let has_tensor = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };

    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);

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
            // The shared resolver owns the `.biases`-sibling affine rule: when
            // biases are present the tensor is integer-affine regardless of the
            // global mode (MLX rejects "default"; affine biases require "affine").
            let qp = resolve_quant(base, biases.is_some(), &defaults, &cfg.quant_overrides)?;
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
            eps: cfg.rms_norm_eps,
        })
    };

    let pfx = "model";

    // Embedding.
    let embed_tokens = {
        let base = format!("{pfx}.embed_tokens");
        if has_tensor(&format!("{base}.scales")) {
            let w = load_array(&format!("{base}.weight"))?;
            let s = load_array(&format!("{base}.scales"))?;
            let biases = if has_tensor(&format!("{base}.biases")) {
                Some(load_array(&format!("{base}.biases"))?)
            } else {
                None
            };
            let qp = resolve_quant(&base, biases.is_some(), &defaults, &cfg.quant_overrides)?;
            Embedding::Quantized {
                weight: w,
                scales: s,
                biases,
                group_size: qp.group_size,
                bits: qp.bits,
                mode: qp.mode,
            }
        } else {
            Embedding::Plain {
                weight: load_array(&format!("{base}.weight"))?,
            }
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if cfg.tie_word_embeddings {
        info!("Laguna: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let base = if has_tensor("lm_head.weight") {
            "lm_head"
        } else {
            "model.lm_head"
        };
        info!(%base, "Laguna: loading separate lm_head");
        Some(load_linear(base)?)
    };

    // Decoder layers.
    let scale = (cfg.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let a = format!("{base}.self_attn");

        let n_heads = cfg.layer_num_heads[i];
        let layer_kind = cfg.layer_attn_kinds[i];

        let attn = Attention {
            q_proj: load_linear(&format!("{a}.q_proj"))?,
            k_proj: load_linear(&format!("{a}.k_proj"))?,
            v_proj: load_linear(&format!("{a}.v_proj"))?,
            o_proj: load_linear(&format!("{a}.o_proj"))?,
            g_proj: load_linear(&format!("{a}.g_proj"))?,
            q_norm: load_rms(&format!("{a}.q_norm"))?,
            k_norm: load_rms(&format!("{a}.k_norm"))?,
            n_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale,
            layer_kind,
            sliding_window: cfg.sliding_window,
            rope_theta_full: cfg.rope_theta_full,
            rope_theta_sliding: cfg.rope_theta_sliding,
            rope_dims_full: cfg.rope_dims_full,
        };

        let mlp = match cfg.layer_mlp_kinds[i] {
            MlpKind::Dense => {
                let m = format!("{base}.mlp");
                Mlp::Dense(DenseMlp {
                    gate_proj: load_linear(&format!("{m}.gate_proj"))?,
                    up_proj: load_linear(&format!("{m}.up_proj"))?,
                    down_proj: load_linear(&format!("{m}.down_proj"))?,
                })
            }
            MlpKind::Sparse => {
                let m = format!("{base}.mlp");

                let gate_base = format!("{m}.gate.proj");
                let router_proj = load_linear(&gate_base)?;

                let bias_name = format!("{m}.gate.e_score_correction_bias");
                let e_score_bias = if has_tensor(&bias_name) {
                    load_array(&bias_name)?
                } else {
                    warn!(
                        layer = i,
                        "laguna: missing e_score_correction_bias, using zeros"
                    );
                    zeros(&[cfg.num_experts as i32], Dtype::Bf16, Device::Cpu)?
                };

                let sw = format!("{m}.switch_mlp");
                let experts = SwitchExperts {
                    gate_proj: load_linear(&format!("{sw}.gate_proj"))?,
                    up_proj: load_linear(&format!("{sw}.up_proj"))?,
                    down_proj: load_linear(&format!("{sw}.down_proj"))?,
                };

                let se = format!("{m}.shared_expert");
                let shared_expert = DenseMlp {
                    gate_proj: load_linear(&format!("{se}.gate_proj"))?,
                    up_proj: load_linear(&format!("{se}.up_proj"))?,
                    down_proj: load_linear(&format!("{se}.down_proj"))?,
                };

                Mlp::Sparse(Box::new(SparseMoeBlock {
                    router: Router {
                        gate_proj: router_proj,
                        e_score_correction_bias: e_score_bias,
                        num_experts: cfg.num_experts,
                        top_k: cfg.num_experts_per_tok,
                    },
                    experts,
                    shared_expert,
                    routed_scaling_factor: cfg.moe_routed_scaling_factor,
                }))
            }
        };

        layers.push(DecoderLayer {
            input_norm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });

        debug!(layer = i, kind = ?cfg.layer_mlp_kinds[i], "laguna: loaded layer");
    }

    info!(
        total_layers = cfg.num_hidden_layers,
        "Laguna: all layers loaded"
    );
    Ok(LagunaText {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}
