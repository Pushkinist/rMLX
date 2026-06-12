//! Laguna model loader.

#![allow(clippy::cognitive_complexity, clippy::too_many_lines)]
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::{zeros, Device, Dtype};
use tracing::{debug, info, warn};

use crate::layers::{resolve_quant, QuantParams};
use crate::load_util::Weights;

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
    let raw_json = crate::load_util::read_raw_config(model_dir)?;
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
    // Laguna ships an honest mxfp8 index → index-first fetch with header-scan
    // fallback via the shared `Weights` helper. (`ShardSet::open`, not
    // `open_dir`, since the index is trustworthy.)
    let shards = ShardSet::open(model_dir, &idx)?;
    let w = Weights::new(&shards, &idx);

    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);

    // Thin adapter: calls the shared seam, then converts the shared
    // `crate::layers::Linear` (mode: QuantMode) to the arch-local
    // `super::layers::Linear` (mode: String). OOM classification is inside
    // `w.linear` — no `.map_err` needed here. `Paro` is unreachable from
    // `w.linear` (it only builds Plain/Quantized) but the match is exhaustive.
    let lin = |base: &str| -> Result<Linear> {
        use crate::layers::Linear as SharedLinear;
        match w.linear(base, |hb| {
            resolve_quant(base, hb, &defaults, &cfg.quant_overrides)
        })? {
            SharedLinear::Plain { weight } => Ok(Linear::Plain { weight }),
            SharedLinear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Ok(Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode: mode.as_str().to_owned(),
            }),
            SharedLinear::Paro { .. } => Err(Error::Loader(format!(
                "{base}: unexpected Paro variant from w.linear"
            ))),
        }
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: w.array(&format!("{name}.weight"))?,
            eps: cfg.rms_norm_eps,
        })
    };

    let pfx = "model";

    // Embedding table. Thin adapter mirrors `lin`: converts shared Embedding
    // (mode: QuantMode) to the arch-local Embedding (mode: String).
    let embed_tokens = {
        use crate::layers::Embedding as SharedEmbedding;
        let base = format!("{pfx}.embed_tokens");
        match w.embedding(&base, |hb| {
            resolve_quant(&base, hb, &defaults, &cfg.quant_overrides)
        })? {
            SharedEmbedding::Plain { weight } => Embedding::Plain { weight },
            SharedEmbedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode: mode.as_str().to_owned(),
            },
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if cfg.tie_word_embeddings {
        info!("Laguna: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let base = if w.has("lm_head.weight")? {
            "lm_head"
        } else {
            "model.lm_head"
        };
        info!(%base, "Laguna: loading separate lm_head");
        Some(lin(base)?)
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
            q_proj: lin(&format!("{a}.q_proj"))?,
            k_proj: lin(&format!("{a}.k_proj"))?,
            v_proj: lin(&format!("{a}.v_proj"))?,
            o_proj: lin(&format!("{a}.o_proj"))?,
            g_proj: lin(&format!("{a}.g_proj"))?,
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
                    gate_proj: lin(&format!("{m}.gate_proj"))?,
                    up_proj: lin(&format!("{m}.up_proj"))?,
                    down_proj: lin(&format!("{m}.down_proj"))?,
                })
            }
            MlpKind::Sparse => {
                let m = format!("{base}.mlp");

                let gate_base = format!("{m}.gate.proj");
                let router_proj = lin(&gate_base)?;

                let bias_name = format!("{m}.gate.e_score_correction_bias");
                let e_score_bias = if w.has(&bias_name)? {
                    w.array(&bias_name)?
                } else {
                    warn!(
                        layer = i,
                        "laguna: missing e_score_correction_bias, using zeros"
                    );
                    zeros(&[cfg.num_experts as i32], Dtype::Bf16, Device::Cpu)?
                };

                let sw = format!("{m}.switch_mlp");
                let experts = SwitchExperts {
                    gate_proj: lin(&format!("{sw}.gate_proj"))?,
                    up_proj: lin(&format!("{sw}.up_proj"))?,
                    down_proj: lin(&format!("{sw}.down_proj"))?,
                };

                let se = format!("{m}.shared_expert");
                let shared_expert = DenseMlp {
                    gate_proj: lin(&format!("{se}.gate_proj"))?,
                    up_proj: lin(&format!("{se}.up_proj"))?,
                    down_proj: lin(&format!("{se}.down_proj"))?,
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
