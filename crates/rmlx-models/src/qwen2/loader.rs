//! Qwen2 model loader from snapshot directory.

#![allow(
    clippy::cloned_instead_of_copied,
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::Array;
use tracing::{debug, info};

use super::config::Qwen2Config;
use super::model::{Attention, DecoderLayer, Embedding, Linear, Mlp, Qwen2Text, RmsNorm};

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a Qwen2 model from a snapshot directory.
pub fn load_from_path(model_dir: &Path) -> Result<Qwen2Text> {
    let cfg_raw = load_config(model_dir)?;
    let cfg = Qwen2Config::from_model_config(&cfg_raw)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        vocab_size = cfg.vocab_size,
        quant_bits = cfg.quant_bits,
        quant_group_size = cfg.quant_group_size,
        "Qwen2: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    // Scan all shard headers (index may omit siblings — same pattern as gemma3.rs).
    fn load_array(shards: &ShardSet, name: &str) -> Result<Array> {
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
    }

    fn has_tensor(shards: &ShardSet, name: &str) -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    }

    let load_quant = |base: &str| -> Result<Linear> {
        let w = load_array(&shards, &format!("{base}.weight"))?;
        let s_name = format!("{base}.scales");
        if has_tensor(&shards, &s_name) {
            let s = load_array(&shards, &s_name)?;
            let biases = if has_tensor(&shards, &format!("{base}.biases")) {
                Some(load_array(&shards, &format!("{base}.biases"))?)
            } else {
                None
            };
            Ok(Linear::Quantized {
                weight: w,
                scales: s,
                biases,
                group_size: cfg.quant_group_size,
                bits: cfg.quant_bits,
                mode: cfg.quant_mode.clone(),
            })
        } else {
            Ok(Linear::Plain { weight: w })
        }
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: load_array(&shards, &format!("{name}.weight"))?,
            eps: cfg.rms_norm_eps,
        })
    };

    // Optional additive bias (`.bias`, not `.biases`) — used by Qwen2 qkv projections.
    let load_attn_bias = |base: &str| -> Result<Option<Array>> {
        let name = format!("{base}.bias");
        if has_tensor(&shards, &name) {
            Ok(Some(load_array(&shards, &name)?))
        } else {
            Ok(None)
        }
    };

    let pfx = "model";

    // Embedding table.
    let embed_tokens = {
        let base = format!("{pfx}.embed_tokens");
        if has_tensor(&shards, &format!("{base}.scales")) {
            let w = load_array(&shards, &format!("{base}.weight"))?;
            let s = load_array(&shards, &format!("{base}.scales"))?;
            let biases = if has_tensor(&shards, &format!("{base}.biases")) {
                Some(load_array(&shards, &format!("{base}.biases"))?)
            } else {
                None
            };
            Embedding::Quantized {
                weight: w,
                scales: s,
                biases,
                group_size: cfg.quant_group_size,
                bits: cfg.quant_bits,
                mode: cfg.quant_mode.clone(),
            }
        } else {
            Embedding::Plain {
                weight: load_array(&shards, &format!("{base}.weight"))?,
            }
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    // lm_head: separate when not weight-tied.
    let lm_head = if cfg.tie_word_embeddings {
        info!("Qwen2: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        // Try top-level "lm_head" first (Qwen3 2-bit layout), then "model.lm_head".
        let base = if has_tensor(&shards, "lm_head.weight") {
            "lm_head"
        } else {
            "model.lm_head"
        };
        info!(%base, "Qwen2: loading separate lm_head");
        if has_tensor(&shards, &format!("{base}.scales")) {
            let w = load_array(&shards, &format!("{base}.weight"))?;
            let s = load_array(&shards, &format!("{base}.scales"))?;
            let biases = if has_tensor(&shards, &format!("{base}.biases")) {
                Some(load_array(&shards, &format!("{base}.biases"))?)
            } else {
                None
            };
            Some(Linear::Quantized {
                weight: w,
                scales: s,
                biases,
                group_size: cfg.quant_group_size,
                bits: cfg.quant_bits,
                mode: cfg.quant_mode.clone(),
            })
        } else {
            Some(Linear::Plain {
                weight: load_array(&shards, &format!("{base}.weight"))?,
            })
        }
    };

    // Decoder layers.
    let scale = (cfg.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let a = format!("{base}.self_attn");

        let attn = Attention {
            q_proj: load_quant(&format!("{a}.q_proj"))?,
            k_proj: load_quant(&format!("{a}.k_proj"))?,
            v_proj: load_quant(&format!("{a}.v_proj"))?,
            o_proj: load_quant(&format!("{a}.o_proj"))?,
            q_bias: load_attn_bias(&format!("{a}.q_proj"))?,
            k_bias: load_attn_bias(&format!("{a}.k_proj"))?,
            v_bias: load_attn_bias(&format!("{a}.v_proj"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale,
            rope_theta: cfg.rope_theta,
        };

        let mlp = Mlp {
            gate_proj: load_quant(&format!("{base}.mlp.gate_proj"))?,
            up_proj: load_quant(&format!("{base}.mlp.up_proj"))?,
            down_proj: load_quant(&format!("{base}.mlp.down_proj"))?,
        };

        layers.push(DecoderLayer {
            input_norm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });

        debug!(layer = i, "qwen2: loaded layer");
    }

    info!(
        total_layers = cfg.num_hidden_layers,
        "Qwen2: all layers loaded"
    );
    Ok(Qwen2Text {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        kv_bytes: crate::kv_bytes::KvBytesCounter::default(),
        model_sig: crate::prompt_cache::model_cache_sig(model_dir),
    })
}
