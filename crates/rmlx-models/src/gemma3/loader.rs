//! Gemma3 model loader.

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::too_many_lines
)]
use std::path::Path;

use rmlx_core::error::Result;
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::Array;
use tracing::{debug, info};

use crate::layers::{resolve_quant, QuantParams};
use crate::load_util::Weights;

use super::attention::Attention;
use super::config::{Gemma3TextConfig, LayerType};
use super::decoder_layer::DecoderLayer;
use super::layers::{Embedding, Linear, Mlp, RmsNormShifted};
use super::model::Gemma3Text;

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a Gemma3 text model from a snapshot directory.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn load_from_path(model_dir: &Path) -> Result<Gemma3Text> {
    let cfg_raw = load_config(model_dir)?;
    let cfg = Gemma3TextConfig::from_model_config(&cfg_raw)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        vocab_size = cfg.vocab_size,
        quant_mode = %cfg.quant_mode,
        "Gemma3: loading model"
    );

    // KEEP the index: `Weights::new` consumes it for index-first speed on the
    // correctly-pointed majority of tensors. But open EVERY `.safetensors` in
    // the dir via `open_dir` rather than the index-driven `open`: the medgemma
    // index lies in two ways the header-scan fallback must be able to repair —
    // 1. Sibling tensors (`.scales`, `.biases`) are not listed at all.
    // 2. Some plain tensors (e.g. `model.norm.weight`) are assigned to the
    //    wrong shard.
    // An index-driven `open` might not open the shard holding an index-omitted
    // tensor; `open_dir` guarantees every shard is open so the fallback reaches
    // it. Tensor names are unique across shards, so the first header-scan match
    // is the same tensor regardless of where the index pointed.
    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open_dir(model_dir)?;
    let w = Weights::new(&shards, &idx);

    // Global quant defaults. medgemma's quant mode is "affine" with `.biases`
    // siblings, so `resolve_quant`'s has_biases→force-affine rule is a no-op
    // (mode/group_size/bits unchanged). gemma3 carries no per-tensor overrides.
    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);
    let overrides = std::collections::HashMap::new();

    let load_plain = |name: &str| -> Result<Array> { w.array(name) };

    // Load affine-quantized or plain linear layer.
    // Handles `.scales` + `.biases` siblings absent from the index.
    let load_quant = |base: &str| -> Result<Linear> {
        let w_name = format!("{base}.weight");
        let s_name = format!("{base}.scales");
        let b_name = format!("{base}.biases");

        let weight = w.array(&w_name)?;

        if w.has(&s_name)? {
            let s = w.array(&s_name)?;
            let biases = if w.has(&b_name)? {
                Some(w.array(&b_name)?)
            } else {
                None
            };
            let qp = resolve_quant(base, biases.is_some(), &defaults, &overrides)?;
            Ok(Linear::Quantized {
                weight,
                scales: s,
                biases,
                group_size: qp.group_size,
                bits: qp.bits,
                mode: qp.mode,
            })
        } else {
            Ok(Linear::Plain { weight })
        }
    };

    let load_rms_shifted = |name: &str| -> Result<RmsNormShifted> {
        let w = load_plain(&format!("{name}.weight"))?;
        RmsNormShifted::from_weight(&w, cfg.rms_norm_eps)
    };

    let pfx = "language_model.model";

    // Embedding table (affine quantized in medgemma).
    let embed_tokens = {
        let base = format!("{pfx}.embed_tokens");
        let s_name = format!("{base}.scales");
        if w.has(&s_name)? {
            let weight = w.array(&format!("{base}.weight"))?;
            let s = w.array(&s_name)?;
            let b_name = format!("{base}.biases");
            let biases = if w.has(&b_name)? {
                Some(w.array(&b_name)?)
            } else {
                None
            };
            let qp = resolve_quant(&base, biases.is_some(), &defaults, &overrides)?;
            Embedding::Quantized {
                weight,
                scales: s,
                biases,
                group_size: qp.group_size,
                bits: qp.bits,
                mode: qp.mode,
            }
        } else {
            let weight = load_plain(&format!("{base}.weight"))?;
            Embedding::Plain { weight }
        }
    };

    // Final norm (shifted-gamma -- single BF16 weight, no quant).
    let final_norm = load_rms_shifted(&format!("{pfx}.norm"))?;

    // lm_head: separate when not weight-tied.
    let lm_head = if cfg.tie_word_embeddings {
        info!("Gemma3: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let lm_base = "language_model.lm_head";
        let lm = if w.has(&format!("{lm_base}.scales"))? {
            load_quant(lm_base)?
        } else {
            let weight = load_plain(&format!("{lm_base}.weight"))?;
            Linear::Plain { weight }
        };
        info!("Gemma3: loaded separate lm_head");
        Some(lm)
    };

    // Decoder layers.
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    let mut sliding_count = 0usize;
    let mut full_count = 0usize;

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let lt = cfg.layer_types[i];
        match lt {
            LayerType::SlidingAttention => sliding_count += 1,
            LayerType::FullAttention => full_count += 1,
        }

        let rope_theta = match lt {
            LayerType::SlidingAttention => cfg.rope_local_theta,
            LayerType::FullAttention => cfg.rope_global_theta,
        };

        let q_norm_w = load_plain(&format!("{base}.self_attn.q_norm.weight"))?;
        let k_norm_w = load_plain(&format!("{base}.self_attn.k_norm.weight"))?;

        let attn = Attention {
            q_proj: load_quant(&format!("{base}.self_attn.q_proj"))?,
            k_proj: load_quant(&format!("{base}.self_attn.k_proj"))?,
            v_proj: load_quant(&format!("{base}.self_attn.v_proj"))?,
            o_proj: load_quant(&format!("{base}.self_attn.o_proj"))?,
            q_norm: RmsNormShifted::from_weight(&q_norm_w, cfg.rms_norm_eps)?,
            k_norm: RmsNormShifted::from_weight(&k_norm_w, cfg.rms_norm_eps)?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale: cfg.attn_scale,
            rope_theta,
            is_sliding: lt == LayerType::SlidingAttention,
            sliding_window: cfg.sliding_window,
        };

        let mlp = Mlp {
            gate_proj: load_quant(&format!("{base}.mlp.gate_proj"))?,
            up_proj: load_quant(&format!("{base}.mlp.up_proj"))?,
            down_proj: load_quant(&format!("{base}.mlp.down_proj"))?,
        };

        layers.push(DecoderLayer {
            input_norm: load_rms_shifted(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms_shifted(&format!("{base}.post_attention_layernorm"))?,
            pre_ffn_norm: load_rms_shifted(&format!("{base}.pre_feedforward_layernorm"))?,
            post_ffn_norm: load_rms_shifted(&format!("{base}.post_feedforward_layernorm"))?,
            attn,
            mlp,
        });

        debug!(layer = i, layer_type = ?lt, "gemma3: loaded layer");
    }

    info!(
        total_layers = cfg.num_hidden_layers,
        sliding_layers = sliding_count,
        full_attn_layers = full_count,
        "Gemma3: all layers loaded"
    );

    Ok(Gemma3Text {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}
