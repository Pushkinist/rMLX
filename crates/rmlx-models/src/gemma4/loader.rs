//! Gemma4 model loaders: mxfp8 and PARO checkpoint variants.
//!
//! Provides two public entry points that build a [`super::model::Gemma4Text`]
//! from a model directory on disk: [`load_from_path`] for standard mxfp8 /
//! bf16 checkpoints and [`load_from_path_paro`] for PARO-quantized snapshots.
//! Both paths share the same shard loading, config parsing, and weight-mapping
//! logic; they differ in how quantized tensors are reconstructed.
//!
//! # Public API
//!
//! - [`load_from_path`] — load a standard Gemma4 checkpoint.
//! - [`load_from_path_paro`] — load a PARO-quantized Gemma4 checkpoint.
//! - [`probe_forward`] — single forward pass used for smoke-probe validation.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::too_many_lines
)]
use std::collections::HashMap;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::{argmax, max_axis, Array, Device, Dtype};
use tracing::{debug, info, warn};

use crate::layers::{
    resolve_quant, Activation, Embedding, Linear, Mlp, ParoRotation, QuantMode, QuantParams,
    RmsNorm,
};
use crate::load_util::{load_paro_parts, quantize_embedding_int4, Weights};

use super::config::{Gemma4TextConfig, LayerType};
use super::decoder_layer::DecoderLayer;
use super::layers::{
    build_proportional_rope_freqs, Attention, Gemma4Experts, Gemma4MoeBlock, Gemma4Router,
    PerLayerInput,
};
use super::model::Gemma4Text;

// ---------------------------------------------------------------------------
// Model loader (mxfp8)
// ---------------------------------------------------------------------------

/// Load a Gemma4 text model from a snapshot directory.
///
/// Reads config.json, resolves tensors, and loads all weights into MLX arrays.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn load_from_path(model_dir: &Path) -> Result<Gemma4Text> {
    let cfg_raw = load_config(model_dir)?;

    // Re-read config.json as raw JSON to capture per-tensor quant overrides.
    let raw_json = crate::load_util::read_raw_config(model_dir)?;
    let raw_quant = raw_json.get("quantization");

    let cfg = Gemma4TextConfig::from_model_config(&cfg_raw, raw_quant)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        vocab_size = cfg.vocab_size,
        num_kv_shared_layers = cfg.num_kv_shared_layers,
        quant_mode = %cfg.quant_mode,
        "Gemma4: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;
    // gemma4 ships an honest index → index-first `array`/`has` hit `Found`
    // and `has` header-truth, with no header-scan degradation on this snapshot.
    let w = Weights::new(&shards, &idx);

    // Resolve quant params for a tensor via the shared resolver: check
    // per-tensor overrides, fall back to global, and let the `.biases` sibling
    // govern the affine rule (hard-erroring on an affine-vs-non-affine clash).
    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);

    // Thin adapter: delegates to the shared seam (OOM classification is inside
    // `w.linear` now — no `.map_err(classify_load_oom)` needed here).
    let lin = |base: &str| -> Result<Linear> {
        w.linear(base, |hb| {
            resolve_quant(base, hb, &defaults, &cfg.quant_overrides)
        })
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        let w_name = format!("{name}.weight");
        let weight = w.array(&w_name)?;
        Ok(RmsNorm {
            weight: Some(weight),
            eps: cfg.rms_norm_eps,
        })
    };

    let load_rms_no_scale = |_name: &str| -> RmsNorm {
        RmsNorm {
            weight: None,
            eps: cfg.rms_norm_eps,
        }
    };

    // Prefix for the text model in this snapshot.
    let pfx = "language_model.model";

    // Embedding tables.
    let embed_tokens = {
        let base = format!("{pfx}.embed_tokens");
        w.embedding(&base, |hb| {
            resolve_quant(&base, hb, &defaults, &cfg.quant_overrides)
        })?
    };

    // Per-layer embedding (optional).
    let embed_tokens_per_layer = if cfg.hidden_size_per_layer_input > 0 {
        let base = format!("{pfx}.embed_tokens_per_layer");
        Some(w.embedding(&base, |hb| {
            resolve_quant(&base, hb, &defaults, &cfg.quant_overrides)
        })?)
    } else {
        None
    };

    // Per-layer model projection.
    let per_layer_model_proj = if cfg.hidden_size_per_layer_input > 0 {
        // This tensor may be plain BF16 (no .scales in this snapshot); `lin` falls
        // through to `Linear::Plain` when no `.scales` sibling is present.
        Some(lin(&format!("{pfx}.per_layer_model_projection"))?)
    } else {
        None
    };

    let per_layer_proj_norm = if cfg.hidden_size_per_layer_input > 0 {
        Some(load_rms(&format!("{pfx}.per_layer_projection_norm"))?)
    } else {
        None
    };

    // Final norm.
    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    // KV sharing: previous_kvs map.
    let previous_kvs = build_previous_kvs(&cfg);

    // Layers.
    let first_kv_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let lt = cfg.layer_types[i];
        let has_kv = i < first_kv_shared;

        let head_dim = match lt {
            LayerType::SlidingAttention => cfg.head_dim,
            LayerType::FullAttention => cfg.global_head_dim,
        };
        let (rope_dims, rope_theta) = match lt {
            LayerType::SlidingAttention => (head_dim as i32, cfg.rope_sliding_theta),
            // rope_dims unused for FullAttention (proportional_rope_freqs handles rotation),
            // but kept for completeness. rope_theta is stored for reference/logging only.
            LayerType::FullAttention => (cfg.rope_full_dims, cfg.rope_full_theta),
        };

        // ProportionalRoPE freqs for full-attention layers only.
        // Built once here, cached in Attention, used on every forward call.
        // Sliding-attention layers keep None and use the standard rope() path.
        let proportional_rope_freqs = match lt {
            LayerType::SlidingAttention => None,
            LayerType::FullAttention => {
                let freqs = build_proportional_rope_freqs(
                    cfg.global_head_dim,
                    cfg.rope_full_dims as usize, // = partial_rotary_factor * global_head_dim = 128
                    cfg.rope_full_theta,
                )?;
                Some(freqs)
            }
        };

        let q_proj = lin(&format!("{base}.self_attn.q_proj"))?;
        let k_proj = if has_kv {
            Some(lin(&format!("{base}.self_attn.k_proj"))?)
        } else {
            None
        };
        // attention_k_eq_v (26B/31B): full-attention layers share K=V weights.
        // The snapshot stores only k_proj and omits v_proj for those layers.
        // Reference: mlx-lm gemma4_text.py line 197:
        // `self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding`
        // When the flag is set AND the layer is full-attention, reuse k_proj as v_proj.
        let use_k_eq_v = cfg.attention_k_eq_v && lt == LayerType::FullAttention;
        let v_proj = if has_kv {
            let v_key = format!("{base}.self_attn.v_proj");
            if use_k_eq_v && !w.has(&format!("{v_key}.weight"))? {
                // v_proj is absent — reuse k_proj (mlx-c arrays are ref-counted, cheap clone).
                debug!(
                    layer = i,
                    "attention_k_eq_v: v_proj absent, reusing k_proj as v_proj"
                );
                k_proj.as_ref().map(Linear::try_clone).transpose()?
            } else {
                Some(lin(&v_key)?)
            }
        } else {
            None
        };
        let o_proj = lin(&format!("{base}.self_attn.o_proj"))?;

        let q_norm = load_rms(&format!("{base}.self_attn.q_norm"))?;
        let k_norm = if has_kv {
            Some(load_rms(&format!("{base}.self_attn.k_norm"))?)
        } else {
            None
        };
        let v_norm = load_rms_no_scale(&format!("{base}.self_attn.v_norm"));

        // When attention_k_eq_v=true, full-attention layers use num_global_key_value_heads
        // (e.g. 2 for 26B, 4 for 31B) rather than num_key_value_heads (8 and 16 respectively).
        // Sliding-attention layers always use num_key_value_heads.
        let n_kv_heads = if use_k_eq_v {
            cfg.num_global_key_value_heads
        } else {
            cfg.num_key_value_heads
        };

        let attn = Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            layer_type: lt,
            sliding_window: cfg.sliding_window,
            rope_dims,
            rope_theta,
            proportional_rope_freqs,
        };

        let mlp = Mlp {
            gate_proj: lin(&format!("{base}.mlp.gate_proj"))?,
            up_proj: lin(&format!("{base}.mlp.up_proj"))?,
            down_proj: lin(&format!("{base}.mlp.down_proj"))?,
            activation: Activation::GeluTanh, // Gemma4 uses GeGLU (GELU with tanh approx).
        };

        let per_layer = if cfg.hidden_size_per_layer_input > 0 {
            let gate = lin(&format!("{base}.per_layer_input_gate"))?;
            let proj = lin(&format!("{base}.per_layer_projection"))?;
            let post_norm = load_rms(&format!("{base}.post_per_layer_input_norm"))?;
            Some(PerLayerInput {
                gate,
                projection: proj,
                post_norm,
            })
        } else {
            None
        };

        // layer_scalar: shape [1], dtype BF16 or F32.
        let layer_scalar_name = format!("{base}.layer_scalar");
        let layer_scalar = if w.has(&layer_scalar_name)? {
            Some(w.array(&layer_scalar_name)?)
        } else {
            None
        };

        // MoE block (26B model — enable_moe_block=true).
        // Tensor layout after sanitize (gemma4_text.py sanitize splits gate_up_proj):
        // experts.switch_glu.{gate,up,down}_proj.{weight,scales} — 3D, no biases (mxfp8)
        // router.{proj,scale,per_expert_scale} — proj is quantized, others are plain
        // Additional norms: post_ffn_norm_1, post_ffn_norm_2, pre_ffn_norm_2.
        let moe_block = if cfg.enable_moe_block {
            let exp_base = format!("{base}.experts.switch_glu");
            let rot_base = format!("{base}.router");

            // Router proj: per-tensor override at g64 b8.
            let router_proj = lin(&format!("{rot_base}.proj"))?;
            let router_scale = w.array(&format!("{rot_base}.scale"))?;
            let per_expert_scale = w.array(&format!("{rot_base}.per_expert_scale"))?;
            let root_size = (cfg.hidden_size as f32).powf(-0.5);

            let router = Gemma4Router {
                proj: router_proj,
                scale: router_scale,
                per_expert_scale,
                num_experts: cfg.num_experts,
                top_k: cfg.top_k_experts,
                root_size,
                eps: cfg.rms_norm_eps,
            };

            let experts = Gemma4Experts {
                gate_proj: lin(&format!("{exp_base}.gate_proj"))?,
                up_proj: lin(&format!("{exp_base}.up_proj"))?,
                down_proj: lin(&format!("{exp_base}.down_proj"))?,
            };

            Some(Gemma4MoeBlock {
                router,
                experts,
                post_ffn_norm_1: load_rms(&format!("{base}.post_feedforward_layernorm_1"))?,
                post_ffn_norm_2: load_rms(&format!("{base}.post_feedforward_layernorm_2"))?,
                pre_ffn_norm_2: load_rms(&format!("{base}.pre_feedforward_layernorm_2"))?,
            })
        } else {
            None
        };

        layers.push(DecoderLayer {
            input_norm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            pre_ffn_norm: load_rms(&format!("{base}.pre_feedforward_layernorm"))?,
            post_ffn_norm: load_rms(&format!("{base}.post_feedforward_layernorm"))?,
            attn,
            mlp,
            moe_block,
            per_layer,
            layer_scalar,
        });

        debug!(layer = i, layer_type = ?lt, has_kv, enable_moe = cfg.enable_moe_block, "loaded layer");
    }

    Ok(Gemma4Text {
        cfg,
        embed_tokens,
        embed_tokens_per_layer,
        per_layer_model_proj,
        per_layer_proj_norm,
        layers,
        final_norm,
        previous_kvs,
        kv_bytes: crate::kv_bytes::KvBytesCounter::new(),
    })
}

/// Build the `previous_kvs` index: `previous_kvs[i]` = index j such that
/// layer `i` should use KV from layer `j`.
///
/// For non-shared layers: `previous_kvs[i] = i` (use own KV).
/// For shared layers (i >= first_kv_shared): use the last non-shared layer
/// of the same attention type.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn build_previous_kvs(cfg: &Gemma4TextConfig) -> Vec<usize> {
    let n = cfg.num_hidden_layers;
    let first_shared = n - cfg.num_kv_shared_layers;

    let mut prev_kvs: Vec<usize> = (0..n).collect();

    if cfg.num_kv_shared_layers > 0 {
        // Build last non-shared layer index per attention type.
        let mut last_by_type: HashMap<u8, usize> = HashMap::new();
        for i in 0..first_shared {
            let key = match cfg.layer_types[i] {
                LayerType::SlidingAttention => 0u8,
                LayerType::FullAttention => 1u8,
            };
            last_by_type.insert(key, i);
        }

        for (j, kv_slot) in prev_kvs.iter_mut().enumerate().skip(first_shared) {
            let key = match cfg.layer_types[j] {
                LayerType::SlidingAttention => 0u8,
                LayerType::FullAttention => 1u8,
            };
            if let Some(&src) = last_by_type.get(&key) {
                *kv_slot = src;
            }
        }
    }

    prev_kvs
}

// ---------------------------------------------------------------------------
// Public helper: run a single forward pass and return the top token + logits.
// ---------------------------------------------------------------------------

/// Run a single token forward pass, return `(top_token_id, max_logit)`.
///
/// This is the entry point called by the CLI `--probe-forward` subcommand.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn probe_forward(model_dir: &Path, token_id: u32, device: Device) -> Result<(u32, f32)> {
    let model = load_from_path(model_dir)?;

    info!(token_id, "forward probe: running single-token forward pass");

    let logits = model.forward_one(token_id, device)?;

    // Logits shape: [1, 1, vocab_size]. Extract the last token's logits.
    let vocab = model.cfg.vocab_size as i32;
    let logits_flat = logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let top = argmax(&logits_flat, -1, device)?;
    top.eval()?;

    let max_val = max_axis(&logits_flat, -1, device)?;
    max_val.eval()?;

    let top_bytes = top.to_bytes()?;
    let top_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;

    let max_bytes = max_val.to_bytes()?;
    // logits output from quantized_matmul may be bf16; read as u16 and convert.
    let max_f32 = match logits_flat.dtype() {
        Dtype::F32 => f32::from_le_bytes(max_bytes[..4].try_into().unwrap()),
        Dtype::Bf16 => {
            let raw = u16::from_le_bytes(max_bytes[..2].try_into().unwrap());
            let f32_bits = u32::from(raw) << 16;
            f32::from_bits(f32_bits)
        }
        _ => {
            warn!("unexpected logits dtype {:?}", logits_flat.dtype());
            0.0
        }
    };

    Ok((top_id, max_f32))
}

// ---------------------------------------------------------------------------
// PARO loader
// ---------------------------------------------------------------------------

/// Load a Gemma4 text model from a ParoQuant INT4 snapshot directory.
///
/// Handles `Gemma4ForConditionalGeneration` with `quantization_config.quant_method = "paroquant"`.
/// Tensor prefix: `model.language_model` (differs from mxfp8 which uses `language_model.model`).
///
/// Key differences from `load_from_path`:
/// - Projection layers loaded as `Linear::Paro` (rotation + INT4 dequant).
/// - `embed_tokens` stored as F16; quantized to INT4 affine at load time.
/// - `attention_k_eq_v=true`: full-attention layers share K=V (v_proj absent in checkpoint).
/// - No per-layer input gating, no KV sharing, no MoE block.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn load_from_path_paro(model_dir: &Path) -> Result<Gemma4Text> {
    let cfg_raw = load_config(model_dir)?;

    // PARO checkpoints use `quantization_config`, not `quantization`.
    let cfg = Gemma4TextConfig::from_model_config(&cfg_raw, None)?;

    let paro_qc = cfg_raw.quantization_config.as_ref().ok_or_else(|| {
        Error::Config("Gemma4 PARO loader: missing quantization_config in config.json".to_owned())
    })?;
    let paro_bits = paro_qc.bits.unwrap_or(4) as usize;
    let paro_group_size = paro_qc.group_size.unwrap_or(128) as usize;
    let _paro_krot = paro_qc.krot.unwrap_or(8) as usize;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        vocab_size = cfg.vocab_size,
        paro_bits,
        paro_group_size,
        "Gemma4 PARO: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;
    // Shared PARO + embedding-quant assembly fetches via `Weights`; the rest of
    // this loader still uses the local `g4_load_raw` closure (full migration to
    // `Weights` is a later task).
    let w = Weights::new(&shards, &idx);

    // ── Low-level tensor loader ──────────────────────────────────────────────

    /// Load raw tensor bytes + shape from any shard.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn g4_load_raw(shards: &ShardSet, name: &str) -> Result<(Vec<u8>, Vec<usize>)> {
        for (_, handle) in shards.iter() {
            let st = handle.safetensors()?;
            if let Ok(t) = st.tensor(name) {
                return Ok((t.data().to_vec(), t.shape().to_vec()));
            }
        }
        Err(Error::Loader(format!(
            "tensor '{name}' not found in shard index"
        )))
    }

    let has_tensor = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };

    // Gemma4 PARO norm weights are stored as-is (no -1.0 offset convention).
    // The qwen3_5_moe PARO applies +1.0 due to its conv1d layout, but Gemma4
    // PARO norms are regular values (~1-5) that need no adjustment.
    let load_rms_paro = |name: &str| -> Result<RmsNorm> {
        let wname = format!("{name}.weight");
        let (w_bytes, w_shape) = g4_load_raw(&shards, &wname)?;
        let shape_i32: Vec<i32> = w_shape.iter().map(|&d| d as i32).collect();
        let w = Array::from_bytes(&w_bytes, &shape_i32, Dtype::F16)?;
        Ok(RmsNorm {
            weight: Some(w),
            eps: cfg.rms_norm_eps,
        })
    };

    // Load a PARO linear layer (returns crate::layers::Linear::Paro).
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

    // Prefix for the text model in this PARO snapshot.
    let pfx = "model.language_model";

    // Embedding: stored as F16 in the PARO checkpoint; quantize to INT4 affine at load time.
    // Python loader calls mlx_nn.quantize(embed_tokens, group_size, bits=4, mode="affine")
    // — required for correct logits (raw F16 produces flat output).
    let embed_tokens = {
        let (weight, scales, biases) =
            quantize_embedding_int4(&w, &format!("{pfx}.embed_tokens"), paro_group_size)?;
        Embedding::Quantized {
            weight,
            scales,
            biases: Some(biases),
            group_size: paro_group_size as i32,
            bits: paro_bits as i32,
            mode: QuantMode::Affine,
        }
    };

    // Final norm.
    let final_norm = load_rms_paro(&format!("{pfx}.norm"))?;

    // KV sharing map (none for 31B PARO: num_kv_shared_layers=0).
    let previous_kvs = build_previous_kvs(&cfg);

    // Layers.
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let lt = cfg.layer_types[i];

        let head_dim = match lt {
            LayerType::SlidingAttention => cfg.head_dim,
            LayerType::FullAttention => cfg.global_head_dim,
        };
        let (rope_dims, rope_theta) = match lt {
            LayerType::SlidingAttention => (head_dim as i32, cfg.rope_sliding_theta),
            LayerType::FullAttention => (cfg.rope_full_dims, cfg.rope_full_theta),
        };

        let proportional_rope_freqs = match lt {
            LayerType::SlidingAttention => None,
            LayerType::FullAttention => {
                let freqs = build_proportional_rope_freqs(
                    cfg.global_head_dim,
                    cfg.rope_full_dims as usize,
                    cfg.rope_full_theta,
                )?;
                Some(freqs)
            }
        };

        // attention_k_eq_v: full-attention layers share K=V.
        // v_proj tensors are absent in the PARO checkpoint for those layers.
        let use_k_eq_v = cfg.attention_k_eq_v && lt == LayerType::FullAttention;

        let sa = format!("{base}.self_attn");
        let q_proj = load_paro(&format!("{sa}.q_proj"))?;
        let k_proj = load_paro(&format!("{sa}.k_proj"))?;
        let v_proj = if use_k_eq_v && !has_tensor(&format!("{sa}.v_proj.qweight")) {
            debug!(
                layer = i,
                "Gemma4 PARO attention_k_eq_v: v_proj absent, reusing k_proj"
            );
            k_proj.try_clone()?
        } else {
            load_paro(&format!("{sa}.v_proj"))?
        };
        let o_proj = load_paro(&format!("{sa}.o_proj"))?;

        let q_norm = load_rms_paro(&format!("{sa}.q_norm"))?;
        let k_norm = load_rms_paro(&format!("{sa}.k_norm"))?;
        // v_norm: not present in PARO checkpoint; use weight=None (RMSNormNoScale).
        let v_norm = RmsNorm {
            weight: None,
            eps: cfg.rms_norm_eps,
        };

        // PARO 31B: num_global_key_value_heads=4 for full-attention, num_key_value_heads=16 for sliding.
        let n_kv_heads = if use_k_eq_v {
            cfg.num_global_key_value_heads
        } else {
            cfg.num_key_value_heads
        };

        let attn = Attention {
            q_proj,
            k_proj: Some(k_proj),
            v_proj: Some(v_proj),
            o_proj,
            q_norm,
            k_norm: Some(k_norm),
            v_norm,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            layer_type: lt,
            sliding_window: cfg.sliding_window,
            rope_dims,
            rope_theta,
            proportional_rope_freqs,
        };

        let mlp = Mlp {
            gate_proj: load_paro(&format!("{base}.mlp.gate_proj"))?,
            up_proj: load_paro(&format!("{base}.mlp.up_proj"))?,
            down_proj: load_paro(&format!("{base}.mlp.down_proj"))?,
            activation: Activation::GeluTanh,
        };

        // layer_scalar: F16 shape [1].
        let layer_scalar_name = format!("{base}.layer_scalar");
        let layer_scalar = if has_tensor(&layer_scalar_name) {
            let (ls_bytes, _) = g4_load_raw(&shards, &layer_scalar_name)?;
            Some(Array::from_bytes(&ls_bytes, &[1], Dtype::F16)?)
        } else {
            None
        };

        layers.push(DecoderLayer {
            input_norm: load_rms_paro(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms_paro(&format!("{base}.post_attention_layernorm"))?,
            pre_ffn_norm: load_rms_paro(&format!("{base}.pre_feedforward_layernorm"))?,
            post_ffn_norm: load_rms_paro(&format!("{base}.post_feedforward_layernorm"))?,
            attn,
            mlp,
            moe_block: None, // enable_moe_block=false for 31B PARO
            per_layer: None, // hidden_size_per_layer_input=0
            layer_scalar,
        });

        debug!(layer = i, layer_type = ?lt, "Gemma4 PARO: loaded layer");
    }

    // tie_word_embeddings=true: lm_head = embed_tokens (no separate tensor).
    Ok(Gemma4Text {
        cfg,
        embed_tokens,
        embed_tokens_per_layer: None,
        per_layer_model_proj: None,
        per_layer_proj_norm: None,
        layers,
        final_norm,
        previous_kvs,
        kv_bytes: crate::kv_bytes::KvBytesCounter::new(),
    })
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
