//! Loaders for Qwen3_5Moe (MoE standard) and Qwen3_5 (PARO dense) checkpoints.
//!
//! Provides two public entry points: [`load_from_path`] for standard MoE
//! checkpoints (bf16, affine-4bit, mxfp8) and [`load_from_path_paro`] for
//! PARO-quantized dense variants. Also contains AWQ → MLX weight conversion
//! helpers used during checkpoint loading.
//!
//! # Public API
//!
//! - [`load_from_path`] — load a standard Qwen3.5-MoE checkpoint.
//! - [`load_from_path_paro`] — load a PARO-quantized Qwen3.5 checkpoint.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::Array;
use rmlx_quant::awq::{f16_bits_to_f32, f32_to_f16_bits};
use tracing::info;

use crate::layers::{resolve_quant, QuantParams};
use crate::load_util::{bf16_param, load_paro_parts, quantize_embedding_int4, Weights};

use super::attention::FullAttention;
use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttnBlock, DecoderLayer, MlpBlock};
use super::gated_delta_net::GatedDeltaNet;
use super::layers::{Embedding, Linear, ParoRotation, RmsNorm};
use super::model::Qwen3_5MoeText;
use super::moe::{DenseMlp, SharedExpert, SparseMoeBlock, SwitchMlp};

// ---------------------------------------------------------------------------
// load_from_path — standard MoE checkpoint
// ---------------------------------------------------------------------------

/// Load a Qwen3_5Moe model from a snapshot directory.
///
/// Expects `config.architectures[0] == "Qwen3_5MoeForConditionalGeneration"`.
/// Tensor prefix: `language_model.model`.
pub fn load_from_path(model_dir: &Path) -> Result<Qwen3_5MoeText> {
    let cfg_raw = load_config(model_dir)?;

    let raw_json = crate::load_util::read_raw_config(model_dir)?;

    let raw_quant = raw_json.get("quantization");
    let raw_text_config = raw_json.get("text_config");

    let cfg = Qwen3_5MoeConfig::from_model_config(&cfg_raw, raw_quant, raw_text_config)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        num_experts = cfg.num_experts,
        num_experts_per_tok = cfg.num_experts_per_tok,
        quant_mode = %cfg.quant_mode,
        quant_overrides = cfg.quant_overrides.len(),
        "Qwen3_5Moe: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;
    let w = Weights::new(&shards, &idx);

    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);

    // Thin adapter: calls the shared seam, then converts the shared
    // `crate::layers::Linear` (mode: QuantMode) to the arch-local
    // `Linear` (mode: String). OOM classification is inside `w.linear`
    // — no `.map_err` needed here. `Paro` is unreachable from `w.linear`
    // (it only builds Plain/Quantized) but the match is exhaustive.
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
                scales: bf16_param(scales)?,
                biases: biases.map(bf16_param).transpose()?,
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
            weight: bf16_param(w.array(&format!("{name}.weight"))?)?,
            eps: cfg.rms_norm_eps,
        })
    };

    let pfx = "language_model.model";

    // Embedding adapter mirrors `lin`: converts shared Embedding
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
                scales: bf16_param(scales)?,
                biases: biases.map(bf16_param).transpose()?,
                group_size,
                bits,
                mode: mode.as_str().to_owned(),
            },
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if cfg.tie_word_embeddings {
        info!("Qwen3_5Moe: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let candidates = [
            "language_model.lm_head",
            "lm_head",
            &format!("{pfx}.lm_head"),
        ];
        let mut base = "language_model.lm_head";
        for cand in candidates {
            if w.has(&format!("{cand}.weight"))? {
                base = cand;
                break;
            }
        }
        info!(%base, "Qwen3_5Moe: loading lm_head");
        Some(lin(base)?)
    };

    let attn_scale = (cfg.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let is_linear = (i + 1) % cfg.full_attention_interval != 0;

        let attn = if is_linear {
            let la = format!("{base}.linear_attn");
            let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;

            let a_log_raw = w.array(&format!("{la}.A_log"))?;
            let hv = cfg.linear_num_value_heads as i32;
            let a_log_3d = a_log_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            let a_log_f32 = a_log_3d.astype(rmlx_mlx::Dtype::F32, rmlx_mlx::Device::Cpu)?;
            let exp_a_log_f32 = rmlx_mlx::exp(&a_log_f32, rmlx_mlx::Device::Cpu)?;
            exp_a_log_f32.eval()?;

            let dt_bias_raw = w.array(&format!("{la}.dt_bias"))?;
            let dt_bias_3d = dt_bias_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            dt_bias_3d.eval()?;

            let inv_scale = (cfg.linear_key_head_dim as f32).powf(-0.5);
            let inv_scale_sq_arr = rmlx_mlx::scalar_f32(inv_scale * inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_sq_arr.eval()?;
            let inv_scale_arr = rmlx_mlx::scalar_f32(inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_arr.eval()?;

            AttnBlock::Linear(GatedDeltaNet {
                in_proj_qkv: lin(&format!("{la}.in_proj_qkv"))?,
                in_proj_z: lin(&format!("{la}.in_proj_z"))?,
                in_proj_b: lin(&format!("{la}.in_proj_b"))?,
                in_proj_a: lin(&format!("{la}.in_proj_a"))?,
                conv1d_weight: w.array(&format!("{la}.conv1d.weight"))?,
                norm_weight: w.array(&format!("{la}.norm.weight"))?,
                exp_a_log_f32,
                dt_bias_3d,
                inv_scale_sq_arr,
                inv_scale_arr,
                out_proj: lin(&format!("{la}.out_proj"))?,
                num_k_heads: cfg.linear_num_key_heads,
                num_v_heads: cfg.linear_num_value_heads,
                head_k_dim: cfg.linear_key_head_dim,
                head_v_dim: cfg.linear_value_head_dim,
                key_dim,
                value_dim,
                eps: cfg.rms_norm_eps,
            })
        } else {
            let sa = format!("{base}.self_attn");
            AttnBlock::Full(FullAttention {
                q_proj: lin(&format!("{sa}.q_proj"))?,
                k_proj: lin(&format!("{sa}.k_proj"))?,
                v_proj: lin(&format!("{sa}.v_proj"))?,
                o_proj: lin(&format!("{sa}.o_proj"))?,
                q_norm: load_rms(&format!("{sa}.q_norm"))?,
                k_norm: load_rms(&format!("{sa}.k_norm"))?,
                n_heads: cfg.num_attention_heads,
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                scale: attn_scale,
                rope_theta: cfg.rope_theta,
                rope_dims: cfg.rope_dims,
            })
        };

        let m = format!("{base}.mlp");
        let mlp = MlpBlock::Moe(Box::new(SparseMoeBlock {
            gate: lin(&format!("{m}.gate"))?,
            switch_mlp: SwitchMlp {
                gate_proj: lin(&format!("{m}.switch_mlp.gate_proj"))?,
                up_proj: lin(&format!("{m}.switch_mlp.up_proj"))?,
                down_proj: lin(&format!("{m}.switch_mlp.down_proj"))?,
            },
            shared_expert: SharedExpert {
                gate_proj: lin(&format!("{m}.shared_expert.gate_proj"))?,
                up_proj: lin(&format!("{m}.shared_expert.up_proj"))?,
                down_proj: lin(&format!("{m}.shared_expert.down_proj"))?,
            },
            shared_expert_gate: lin(&format!("{m}.shared_expert_gate"))?,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        }));

        layers.push(DecoderLayer {
            input_layernorm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });
    }

    Ok(Qwen3_5MoeText {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        cached_hot_head: std::sync::OnceLock::new(),
    })
}

// ---------------------------------------------------------------------------
// load_from_path_paro — PARO dense checkpoint
// ---------------------------------------------------------------------------

/// Load a Qwen3_5 (dense, PARO) model from a snapshot directory.
///
/// Handles `Qwen3_5ForConditionalGeneration` with ParoQuant INT4 weights.
/// Tensor prefix: `model.language_model` (same as MoE variant).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn load_from_path_paro(model_dir: &Path) -> Result<Qwen3_5MoeText> {
    let cfg_raw = load_config(model_dir)?;

    let raw_json = crate::load_util::read_raw_config(model_dir)?;

    let raw_text_config = raw_json.get("text_config");
    let raw_quant = None;

    let mut cfg = Qwen3_5MoeConfig::from_model_config(&cfg_raw, raw_quant, raw_text_config)
        .or_else(|_| {
            let mut tc = raw_text_config
                .and_then(|v| v.as_object())
                .map(|m| {
                    let mut map = serde_json::Map::new();
                    map.extend(m.clone());
                    map
                })
                .unwrap_or_default();
            let inter = tc
                .get("intermediate_size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(17408);
            tc.entry("num_experts").or_insert(serde_json::json!(1));
            tc.entry("num_experts_per_tok")
                .or_insert(serde_json::json!(1));
            tc.entry("moe_intermediate_size")
                .or_insert(serde_json::json!(inter));
            tc.entry("shared_expert_intermediate_size")
                .or_insert(serde_json::json!(inter));
            tc.entry("norm_topk_prob")
                .or_insert(serde_json::json!(false));
            let patched = serde_json::Value::Object(tc);
            Qwen3_5MoeConfig::from_model_config(&cfg_raw, raw_quant, Some(&patched))
        })?;

    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 1;

    let paro_qc = cfg_raw.quantization_config.as_ref().ok_or_else(|| {
        Error::Config("PARO loader: missing quantization_config in config.json".to_owned())
    })?;
    let paro_bits = paro_qc.bits.unwrap_or(4) as usize;
    let paro_group_size = paro_qc.group_size.unwrap_or(128) as usize;
    let paro_krot = paro_qc.krot.unwrap_or(8) as usize;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        paro_bits,
        paro_group_size,
        paro_krot,
        "Qwen3_5 PARO: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;
    // All tensor fetches go through the shared `Weights` handle. The PARO-specific
    // `load_array` / `load_rms` closures keep their manual byte-math bodies (raw
    // dtype match + f16 `+1.0` RMS shift) but source their bytes from `w.raw`.
    let w = Weights::new(&shards, &idx);

    let load_array = |name: &str| -> Result<Array> {
        let (bytes, shape, dtype) = w.raw(name)?;
        let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mlx_dtype = match dtype {
            safetensors::Dtype::F16 => rmlx_mlx::Dtype::F16,
            safetensors::Dtype::BF16 => rmlx_mlx::Dtype::Bf16,
            safetensors::Dtype::F32 => rmlx_mlx::Dtype::F32,
            safetensors::Dtype::I32 => rmlx_mlx::Dtype::I32,
            safetensors::Dtype::U32 => rmlx_mlx::Dtype::U32,
            other => {
                return Err(Error::Loader(format!(
                    "load_array '{name}': unsupported dtype {other:?}"
                )));
            }
        };
        Array::from_bytes(&bytes, &shape_i32, mlx_dtype)
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        let wname = format!("{name}.weight");
        let (w_bytes, w_shape, _) = w.raw(&wname)?;
        let n = w_shape.iter().product::<usize>();
        let mut shifted = vec![0u8; n * 2];
        for i in 0..n {
            let bits = u16::from_le_bytes([w_bytes[i * 2], w_bytes[i * 2 + 1]]);
            let v = f16_bits_to_f32(bits) + 1.0_f32;
            let out_bits = f32_to_f16_bits(v);
            shifted[i * 2..i * 2 + 2].copy_from_slice(&out_bits.to_le_bytes());
        }
        let shape_i32: Vec<i32> = w_shape.iter().map(|&d| d as i32).collect();
        let weight = Array::from_bytes(&shifted, &shape_i32, rmlx_mlx::Dtype::F16)?;
        Ok(RmsNorm {
            weight,
            eps: cfg.rms_norm_eps,
        })
    };

    let load_plain_linear = |base: &str| -> Result<Linear> {
        let w_name = format!("{base}.weight");
        let (w_bytes, w_shape, _) = w.raw(&w_name)?;
        let w_shape_i32: Vec<i32> = w_shape.iter().map(|&d| d as i32).collect();
        let weight = Array::from_bytes(&w_bytes, &w_shape_i32, rmlx_mlx::Dtype::F16)?;
        Ok(Linear::Plain { weight })
    };

    let load_paro = |base: &str| -> Result<Linear> {
        let p = load_paro_parts(&w, base, paro_group_size)?;
        Ok(Linear::Paro {
            rotation: ParoRotation {
                packed_pairs: p.packed_pairs,
                cos_theta: p.cos_theta,
                sin_theta: p.sin_theta,
                channel_scales: p.channel_scales,
                krot: p.krot,
                group_size: p.group_size,
            },
            weight: p.weight,
            scales: p.scales,
            biases: p.biases,
        })
    };

    let load_auto_linear = |base: &str| -> Result<Linear> {
        if w.has(&format!("{base}.pairs"))? {
            load_paro(base)
        } else {
            load_plain_linear(base)
        }
    };

    let pfx = "model.language_model";

    let embed_tokens = {
        let (weight, scales, biases) =
            quantize_embedding_int4(&w, &format!("{pfx}.embed_tokens"), paro_group_size)?;
        Embedding::Quantized {
            weight,
            scales,
            biases: Some(biases),
            group_size: paro_group_size as i32,
            bits: paro_bits as i32,
            mode: "affine".to_owned(),
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if cfg.tie_word_embeddings {
        info!("Qwen3_5 PARO: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let candidates = ["lm_head", &format!("{pfx}.lm_head")];
        let mut base = "lm_head";
        for cand in candidates {
            if w.has(&format!("{cand}.weight"))? {
                base = cand;
                break;
            }
        }
        info!(%base, "Qwen3_5 PARO: loading lm_head (INT4 quantized to match Python loader)");
        let (lm_w, lm_s, lm_b) = quantize_embedding_int4(&w, base, paro_group_size)?;
        Some(Linear::Quantized {
            weight: lm_w,
            scales: lm_s,
            biases: Some(lm_b),
            group_size: paro_group_size as i32,
            bits: paro_bits as i32,
            mode: "affine".to_owned(),
        })
    };

    let attn_scale = (cfg.head_dim as f32).powf(-0.5);
    let _ = paro_krot;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let is_linear = (i + 1) % cfg.full_attention_interval != 0;

        let attn = if is_linear {
            let la = format!("{base}.linear_attn");
            let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;

            let a_log_raw = load_array(&format!("{la}.A_log"))?;
            let hv = cfg.linear_num_value_heads as i32;
            let a_log_3d = a_log_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            let a_log_f32 = a_log_3d.astype(rmlx_mlx::Dtype::F32, rmlx_mlx::Device::Cpu)?;
            let exp_a_log_f32 = rmlx_mlx::exp(&a_log_f32, rmlx_mlx::Device::Cpu)?;
            exp_a_log_f32.eval()?;

            let dt_bias_raw = load_array(&format!("{la}.dt_bias"))?;
            let dt_bias_3d = dt_bias_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            dt_bias_3d.eval()?;

            let inv_scale = (cfg.linear_key_head_dim as f32).powf(-0.5);
            let inv_scale_sq_arr = rmlx_mlx::scalar_f32(inv_scale * inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_sq_arr.eval()?;
            let inv_scale_arr = rmlx_mlx::scalar_f32(inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_arr.eval()?;

            AttnBlock::Linear(GatedDeltaNet {
                in_proj_qkv: load_auto_linear(&format!("{la}.in_proj_qkv"))?,
                in_proj_z: load_auto_linear(&format!("{la}.in_proj_z"))?,
                in_proj_b: load_plain_linear(&format!("{la}.in_proj_b"))?,
                in_proj_a: load_plain_linear(&format!("{la}.in_proj_a"))?,
                conv1d_weight: {
                    let w = load_array(&format!("{la}.conv1d.weight"))?;
                    let s = w.shape();
                    if s.len() == 3 && s[1] < s[2] {
                        w.transpose(&[0, 2, 1], rmlx_mlx::Device::Cpu)?
                    } else {
                        w
                    }
                },
                norm_weight: load_array(&format!("{la}.norm.weight"))?,
                exp_a_log_f32,
                dt_bias_3d,
                inv_scale_sq_arr,
                inv_scale_arr,
                out_proj: load_auto_linear(&format!("{la}.out_proj"))?,
                num_k_heads: cfg.linear_num_key_heads,
                num_v_heads: cfg.linear_num_value_heads,
                head_k_dim: cfg.linear_key_head_dim,
                head_v_dim: cfg.linear_value_head_dim,
                key_dim,
                value_dim,
                eps: cfg.rms_norm_eps,
            })
        } else {
            let sa = format!("{base}.self_attn");
            AttnBlock::Full(FullAttention {
                q_proj: load_auto_linear(&format!("{sa}.q_proj"))?,
                k_proj: load_auto_linear(&format!("{sa}.k_proj"))?,
                v_proj: load_auto_linear(&format!("{sa}.v_proj"))?,
                o_proj: load_auto_linear(&format!("{sa}.o_proj"))?,
                q_norm: load_rms(&format!("{sa}.q_norm"))?,
                k_norm: load_rms(&format!("{sa}.k_norm"))?,
                n_heads: cfg.num_attention_heads,
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                scale: attn_scale,
                rope_theta: cfg.rope_theta,
                rope_dims: cfg.rope_dims,
            })
        };

        let m = format!("{base}.mlp");
        let mlp = MlpBlock::Dense(Box::new(DenseMlp {
            gate_proj: load_auto_linear(&format!("{m}.gate_proj"))?,
            up_proj: load_auto_linear(&format!("{m}.up_proj"))?,
            down_proj: load_auto_linear(&format!("{m}.down_proj"))?,
        }));

        layers.push(DecoderLayer {
            input_layernorm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });
    }

    Ok(Qwen3_5MoeText {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        cached_hot_head: std::sync::OnceLock::new(),
    })
}
