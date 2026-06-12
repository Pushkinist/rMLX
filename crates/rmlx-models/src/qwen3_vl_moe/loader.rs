//! Loader for the `Qwen3VLMoeForConditionalGeneration` snapshot.
//!
//! Enumerates the on-disk `model*.safetensors` shards directly rather than
//! trusting `model.safetensors.index.json` — the mlx-community 4-bit snapshot
//! ships a **stale index** (it references the pre-sanitize 13-shard layout with
//! packed `gate_up_proj` keys, while the actual 4 on-disk shards carry the
//! sanitized `language_model.model.*` / `vision_tower.*` layout with pre-split
//! `switch_mlp.{gate,up,down}_proj` experts). mlx-lm's loader globs the files
//! the same way; mirroring that keeps us robust to the index mismatch.
//!
//! Quant layout (verified from shard headers): 4-bit affine, `group_size = 64`,
//! `weight` U32 + `scales`/`biases` BF16. `mlp.gate` routers are 8-bit affine
//! (per upstream `quant_predicate`). Tensor prefixes:
//! - LM: `language_model.model.*`, final norm `language_model.model.norm`
//! - lm_head: `language_model.lm_head.*`
//! - vision: `vision_tower.*` (BF16, unquantized) — wired by vision step.

#![allow(clippy::too_many_lines)]
use std::collections::HashMap;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, ShardSet};
use rmlx_mlx::Array;
use tracing::info;

use crate::layers::RmsNorm;
use crate::load_util::Weights;

use super::attention::Attention;
use super::config::Qwen3VlMoeConfig;
use super::layers::{Embedding, Linear};
use super::model::{DecoderLayer, MlpBlock, Qwen3VlMoeText};
use super::moe::{SparseMoeBlock, SwitchMlp};

/// Load the full `Qwen3VlMoeConfig` from `config.json`.
pub fn load_config_qwen3_vl(model_dir: &Path) -> Result<Qwen3VlMoeConfig> {
    let raw_json = crate::load_util::read_raw_config(model_dir)?;
    let raw = raw_json
        .as_object()
        .ok_or_else(|| Error::Config("qwen3_vl_moe: config.json is not an object".into()))?;

    // Global quantization block (group_size/bits/mode).
    let (gs, bits, mode) = raw.get("quantization").and_then(|v| v.as_object()).map_or(
        (64, 4, "affine".to_owned()),
        |q| {
            (
                q.get("group_size")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(64) as i32,
                q.get("bits")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(4) as i32,
                q.get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("affine")
                    .to_owned(),
            )
        },
    );

    Qwen3VlMoeConfig::from_raw(raw, (gs, bits, mode, HashMap::new()))
}

/// Load the full Qwen3-VL-MoE model (text decoder + image/vision metadata).
///
/// The vision tower itself is loaded on demand by the server (mirroring the
/// Gemma4/Gemma3 `VisionBundle` pattern) so text-only requests pay no vision
/// load cost; this returns the text decoder plus the token-id / merge-size
/// metadata the image branch needs.
pub fn load_from_path(model_dir: &Path) -> Result<super::model::Qwen3VlMoe> {
    let cfg = load_config_qwen3_vl(model_dir)?;
    let image_token_id = cfg.image_token_id;
    let spatial_merge_size = cfg.vision.spatial_merge_size;
    let max_position_embeddings = cfg.text.max_position_embeddings;
    let text = load_text_from_path(model_dir)?;
    Ok(super::model::Qwen3VlMoe {
        text,
        image_token_id,
        spatial_merge_size,
        max_position_embeddings,
    })
}

/// Load the Qwen3-VL-MoE **text decoder** from a snapshot directory.
///
/// (The vision tower loads separately via [`super::vision::load_vision_tower`].)
pub fn load_text_from_path(model_dir: &Path) -> Result<Qwen3VlMoeText> {
    // Validate model_type via the standard config (best-effort; we parse the
    // nested config ourselves below).
    let _ = load_config(model_dir);

    let cfg = load_config_qwen3_vl(model_dir)?;
    let tc = cfg.text;

    info!(
        num_hidden_layers = tc.num_hidden_layers,
        hidden_size = tc.hidden_size,
        num_experts = tc.num_experts,
        num_experts_per_tok = tc.num_experts_per_tok,
        quant_bits = tc.quant_bits,
        quant_group_size = tc.quant_group_size,
        "qwen3_vl_moe: loading text decoder"
    );

    // Open every `*.safetensors` shard by directory glob (ignoring the stale
    // index) and fetch tensors via a pure header scan — exactly matching the
    // prior `open_shards` + per-shard lookup behaviour.
    let shards = ShardSet::open_dir(model_dir)?;
    let w = Weights::scan_only(&shards);

    // Per-layer quant params. Default = global (4-bit). `mlp.gate` routers are
    // 8-bit (upstream quant_predicate). We read group_size/bits from the actual
    // scales shape rather than hard-coding, so any cell stays correct.
    let load_linear = |base: &str| -> Result<Linear> {
        let weight = w.array(&format!("{base}.weight"))?;
        let s_name = format!("{base}.scales");
        if w.has(&s_name)? {
            let s = w.array(&s_name)?;
            let biases = if w.has(&format!("{base}.biases"))? {
                Some(w.array(&format!("{base}.biases"))?)
            } else {
                None
            };
            let (group_size, bits) = infer_quant(&weight, &s)?;
            Ok(Linear::Quantized {
                weight,
                scales: s,
                biases,
                group_size,
                bits,
                mode: tc.quant_mode.clone(),
            })
        } else {
            Ok(Linear::Plain { weight })
        }
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Some(w.array(&format!("{name}.weight"))?),
            eps: tc.rms_norm_eps,
        })
    };

    let pfx = "language_model.model";

    let embed_tokens = {
        let base = format!("{pfx}.embed_tokens");
        if w.has(&format!("{base}.scales"))? {
            let weight = w.array(&format!("{base}.weight"))?;
            let s = w.array(&format!("{base}.scales"))?;
            let biases = if w.has(&format!("{base}.biases"))? {
                Some(w.array(&format!("{base}.biases"))?)
            } else {
                None
            };
            let (group_size, bits) = infer_quant(&weight, &s)?;
            Embedding::Quantized {
                weight,
                scales: s,
                biases,
                group_size,
                bits,
                mode: tc.quant_mode.clone(),
            }
        } else {
            Embedding::Plain {
                weight: w.array(&format!("{base}.weight"))?,
            }
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if tc.tie_word_embeddings {
        info!("qwen3_vl_moe: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let candidates = ["language_model.lm_head", "lm_head"];
        let mut base = "language_model.lm_head";
        for b in candidates {
            if w.has(&format!("{b}.weight"))? {
                base = b;
                break;
            }
        }
        info!(%base, "qwen3_vl_moe: loading lm_head");
        Some(load_linear(base)?)
    };

    let attn_scale = (tc.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(tc.num_hidden_layers);

    for i in 0..tc.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let sa = format!("{base}.self_attn");
        let self_attn = Attention {
            q_proj: load_linear(&format!("{sa}.q_proj"))?,
            k_proj: load_linear(&format!("{sa}.k_proj"))?,
            v_proj: load_linear(&format!("{sa}.v_proj"))?,
            o_proj: load_linear(&format!("{sa}.o_proj"))?,
            q_norm: load_rms(&format!("{sa}.q_norm"))?,
            k_norm: load_rms(&format!("{sa}.k_norm"))?,
            n_heads: tc.num_attention_heads,
            n_kv_heads: tc.num_key_value_heads,
            head_dim: tc.head_dim,
            scale: attn_scale,
        };

        let m = format!("{base}.mlp");
        let is_moe = !tc.mlp_only_layers.contains(&i)
            && tc.num_experts > 0
            && (i + 1) % tc.decoder_sparse_step == 0;
        let mlp = if is_moe {
            MlpBlock::Moe(Box::new(SparseMoeBlock {
                gate: load_linear(&format!("{m}.gate"))?,
                switch_mlp: SwitchMlp {
                    gate_proj: load_linear(&format!("{m}.switch_mlp.gate_proj"))?,
                    up_proj: load_linear(&format!("{m}.switch_mlp.up_proj"))?,
                    down_proj: load_linear(&format!("{m}.switch_mlp.down_proj"))?,
                },
                num_experts: tc.num_experts,
                top_k: tc.num_experts_per_tok,
                norm_topk_prob: tc.norm_topk_prob,
            }))
        } else {
            MlpBlock::Dense(super::moe::DenseMlp {
                gate_proj: load_linear(&format!("{m}.gate_proj"))?,
                up_proj: load_linear(&format!("{m}.up_proj"))?,
                down_proj: load_linear(&format!("{m}.down_proj"))?,
            })
        };

        layers.push(DecoderLayer {
            input_layernorm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            self_attn,
            mlp,
        });
    }

    Ok(Qwen3VlMoeText {
        cfg: tc,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}

/// Infer `(group_size, bits)` from a quantized weight + its scales.
///
/// MLX affine packing: `weight` is U32 with `in/(32/bits)` columns; `scales`
/// has `in/group_size` columns where `in` is the logical input width. For a 2D
/// projection `weight=[out, packed_in]`, `scales=[out, n_groups]`. For 3D MoE
/// experts `weight=[E, out, packed_in]`, `scales=[E, out, n_groups]`.
///
/// `bits = packed_in*32 / in`, `group_size = in / n_groups`. We recover `in`
/// from the relationship `packed_in = in*bits/32`. With `bits ∈ {4, 8}` the
/// only consistent solution for the observed shapes is selected.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn infer_quant(weight: &Array, scales: &Array) -> Result<(i32, i32)> {
    let w = weight.shape();
    let s = scales.shape();
    let last = w.len() - 1;
    let packed_in = i64::from(w[last]);
    let n_groups = i64::from(s[last]);
    // Try bits = 4 then 8; pick the one giving an integer group_size that
    // divides `in` cleanly.
    for &bits in &[4i64, 8i64] {
        let elems_per_word = 32 / bits;
        let logical_in = packed_in * elems_per_word;
        if logical_in % n_groups == 0 {
            let gs = logical_in / n_groups;
            // group_size must be a power-of-two-ish typical value (32/64/128).
            if gs == 32 || gs == 64 || gs == 128 || gs == 256 {
                return Ok((gs as i32, bits as i32));
            }
        }
    }
    Err(Error::Loader(format!(
        "qwen3_vl_moe: cannot infer (group_size, bits) from weight {w:?} scales {s:?}"
    )))
}
