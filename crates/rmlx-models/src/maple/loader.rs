//! Maple checkpoint loader (`MapleForCausalLM`).
//!
//! Replicates mlx-lm `Model.sanitize()` enough for v1 (no FlashHead, no QKV
//! concat, no up/gate fuse):
//!
//! 1. Expand sibling `{prefix}.row_alpha` → per-group affine `scales` /
//!    `biases = -scales` (2-bit, group 128) when `{prefix}.scales` is absent.
//! 2. Stack `mlp.experts.{0..E-1}.{gate,up,down}_proj.*` along axis 0 into
//!    `mlp.switch_mlp.*` when the Hugging Face per-expert layout is present.
//! 3. Skip `lm_head_flash.*` (v1 has no FlashHead).
//!
//! # Snapshot tensor names
//!
//! | Role | Names |
//! |---|---|
//! | Embed (affine 4-bit g64) | `model.word_embeddings.{weight,scales,biases}` |
//! | Layer norms | `model.layers.{L}.{input,post_attention}_layernorm.weight` |
//! | Attention (2-bit + `row_alpha`) | `self_attn.{q,k,v,o}_proj.{weight,row_alpha}` |
//! | QK RMS | `self_attn.{q,k}_norm.weight` (bf16) |
//! | Router | `mlp.gate.weight` (bf16, never quantized) |
//! | Experts | `mlp.experts.{E}.{gate,up,down}_proj.{weight,row_alpha}` **or** already-stacked `mlp.switch_mlp.{gate,up,down}_proj.{weight,row_alpha}` |
//! | Final RMS | `model.norm.weight` |
//! | LM head (affine 4-bit g64) | `lm_head.{weight,scales,biases}` |
//!
//! # Bundle → `MapleText`
//!
//! [`load_weights`] returns [`MapleLoadBundle`]. [`load_from_path`] calls
//! [`MapleText::from_bundle`], which wires:
//!
//! - [`MapleAttention::new`] — `{ q,k,v,o }_proj` + `MapleRmsNorm` Q/K norms;
//!   `use_rope` / scale / head counts come from `MapleConfig`.
//! - [`MapleSparseMoeBlock`] — `MapleGate { weight, top_k }` +
//!   `MapleSwitchGLU { gate_proj, up_proj, down_proj }` with 3-D expert
//!   weights `[E, out, packed_in]`.
//! - [`MapleDecoderLayer::new`] — pre-norm attn + pre-norm MoE.
//! - [`MapleText`] — `{ cfg, embed, layers, norm, lm_head, kv_bytes, model_sig }`.

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{broadcast_to, expand_dims, negative, stack_axis, Array, Device};
use tracing::info;

use crate::layers::{resolve_quant, Embedding, Linear, QuantMode, QuantParams, RmsNorm};
use crate::load_util::{bf16_param, bf16_scales, Weights};

use super::attention::{MapleAttention, MapleRmsNorm};
use super::config::MapleConfig;
use super::decoder_layer::MapleDecoderLayer;
use super::model::MapleText;
use super::moe::{MapleGate, MapleSparseMoeBlock, MapleSwitchGLU};

/// 2-bit packing: 32 / 2 = 16 codes per uint32 word.
const CODES_PER_U32_2BIT: i64 = 16;

/// Default affine group size for `row_alpha` expansion and most linears.
const DEFAULT_MOE_GROUP: i32 = 128;

/// Default packed bit-width for Maple linears / experts.
const DEFAULT_LINEAR_BITS: i32 = 2;

/// Embed / lm_head affine parameters (checkpoint `quant_predicate`).
const EMBED_GROUP: i32 = 64;
const EMBED_BITS: i32 = 4;

// ---------------------------------------------------------------------------
// Bundle (the constructor surface `model.rs` consumes)
// ---------------------------------------------------------------------------

/// Fully materialized Maple weights, ready for `MapleText::from_bundle`.
#[allow(
    missing_debug_implementations,
    clippy::exhaustive_structs,
    reason = "load-time bundle of MLX arrays; fields are the Maple checkpoint contract"
)]
pub struct MapleLoadBundle {
    /// Parsed `config.json`.
    pub cfg: MapleConfig,
    /// `model.word_embeddings` (4-bit g64, or plain).
    pub embed: Embedding,
    /// Per-layer norms + attention + MLP.
    pub layers: Vec<MapleLayerWeights>,
    /// `model.norm`.
    pub norm: RmsNorm,
    /// Top-level `lm_head`. `None` when `tie_word_embeddings`.
    pub lm_head: Option<Linear>,
    /// Snapshot identity folded into the prompt-cache seed.
    pub model_sig: u64,
}

/// One decoder layer's loaded tensors.
#[allow(
    missing_debug_implementations,
    clippy::exhaustive_structs,
    reason = "load-time layer bundle; fields match the decoder-layer constructor"
)]
pub struct MapleLayerWeights {
    /// `input_layernorm.weight` (bf16 `[hidden]`).
    pub input_layernorm: Array,
    /// `post_attention_layernorm.weight` (bf16 `[hidden]`).
    pub post_attention_layernorm: Array,
    /// Self-attention projections + QK RMS weights.
    pub attn: MapleAttnWeights,
    /// MoE (every Maple-Preview layer) or dense SwiGLU.
    pub mlp: MapleMlpWeights,
}

/// Attention projections for one layer.
///
/// Field names match `MapleAttention` in `attention.rs`.
#[allow(
    missing_debug_implementations,
    clippy::exhaustive_structs,
    reason = "load-time attention bundle; fields match MapleAttention"
)]
pub struct MapleAttnWeights {
    /// `self_attn.q_proj` (2-bit g128 after `row_alpha` expansion).
    pub q_proj: Linear,
    /// `self_attn.k_proj`.
    pub k_proj: Linear,
    /// `self_attn.v_proj`.
    pub v_proj: Linear,
    /// `self_attn.o_proj`.
    pub o_proj: Linear,
    /// `self_attn.q_norm.weight` (bf16, per-head `[head_dim]`).
    pub q_norm: Array,
    /// `self_attn.k_norm.weight` (bf16, per-head `[head_dim]`).
    pub k_norm: Array,
}

/// MLP block: sparse MoE or dense SwiGLU.
#[allow(
    missing_debug_implementations,
    clippy::exhaustive_enums,
    reason = "two checkpoint MLP layouts (MoE vs dense); adding a variant needs a new tensor layout"
)]
pub enum MapleMlpWeights {
    /// Routed experts + bf16 router.
    Moe {
        /// `mlp.gate` — plain bf16 `[num_experts, hidden]`.
        gate: Linear,
        /// Stacked `gate_proj` `[E, moe_ff, packed_in]`.
        gate_proj: Linear,
        /// Stacked `up_proj` `[E, moe_ff, packed_in]`.
        up_proj: Linear,
        /// Stacked `down_proj` `[E, hidden, packed_ff]`.
        down_proj: Linear,
        /// Expert count (256).
        num_experts: i32,
        /// Top-k (8).
        top_k: i32,
    },
    /// Dense SwiGLU (`first_k_dense_replace` layers, if any).
    Dense {
        /// `mlp.gate_proj`.
        gate_proj: Linear,
        /// `mlp.up_proj`.
        up_proj: Linear,
        /// `mlp.down_proj`.
        down_proj: Linear,
    },
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Load a Maple snapshot and assemble [`MapleText`].
pub fn load_from_path(model_dir: &Path) -> Result<MapleText> {
    MapleText::from_bundle(load_weights(model_dir)?)
}

impl MapleText {
    /// Wire a [`MapleLoadBundle`] into attention / MoE / decoder types.
    pub fn from_bundle(bundle: MapleLoadBundle) -> Result<Self> {
        let MapleLoadBundle {
            cfg,
            embed,
            layers: loaded,
            norm,
            lm_head,
            model_sig,
        } = bundle;
        let eps = cfg.rms_norm_eps;
        let mut layers = Vec::with_capacity(loaded.len());
        for (i, layer) in loaded.into_iter().enumerate() {
            let attn = MapleAttention::new(
                &cfg,
                i,
                layer.attn.q_proj,
                layer.attn.k_proj,
                layer.attn.v_proj,
                layer.attn.o_proj,
                MapleRmsNorm::new(layer.attn.q_norm, eps),
                MapleRmsNorm::new(layer.attn.k_norm, eps),
            );
            let mlp = match layer.mlp {
                MapleMlpWeights::Moe {
                    gate,
                    gate_proj,
                    up_proj,
                    down_proj,
                    top_k,
                    ..
                } => {
                    let top_k = usize::try_from(top_k).map_err(|_| {
                        Error::Loader(format!(
                            "maple: num_experts_per_tok {top_k} does not fit usize"
                        ))
                    })?;
                    MapleSparseMoeBlock {
                        gate: MapleGate {
                            weight: router_weight(gate)?,
                            top_k,
                        },
                        switch: MapleSwitchGLU {
                            gate_proj,
                            up_proj,
                            down_proj,
                        },
                    }
                }
                MapleMlpWeights::Dense { .. } => {
                    return Err(Error::Loader(format!(
                        "maple: layer {i} is dense SwiGLU; v1 decoder only hosts MapleSparseMoeBlock"
                    )));
                }
            };
            layers.push(MapleDecoderLayer::new(
                MapleRmsNorm::new(layer.input_layernorm, eps),
                attn,
                MapleRmsNorm::new(layer.post_attention_layernorm, eps),
                mlp,
            ));
        }
        Ok(Self {
            cfg,
            embed,
            layers,
            norm,
            lm_head,
            kv_bytes: crate::kv_bytes::KvBytesCounter::default(),
            model_sig,
        })
    }
}

/// Load every Maple tensor into a [`MapleLoadBundle`].
///
/// Attention / MoE / decoder types are wired by [`MapleText::from_bundle`].
#[allow(
    clippy::cognitive_complexity,
    reason = "single cohesive load sequence: config + embed/head + per-layer attn/MLP"
)]
pub fn load_weights(model_dir: &Path) -> Result<MapleLoadBundle> {
    let raw_json = crate::load_util::read_raw_config(model_dir)?;
    let cfg: MapleConfig = serde_json::from_value(raw_json.clone())
        .map_err(|e| Error::Config(format!("maple: cannot deserialize config.json: {e}")))?;
    let raw_quant = raw_json.get("quantization");
    let overrides = extract_quant_overrides(raw_quant);
    let lin_defaults = linear_defaults(&cfg);
    let embed_defaults = QuantParams::global(EMBED_GROUP, EMBED_BITS, "affine");

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        num_experts = cfg.num_experts,
        quant_bits = lin_defaults.bits,
        quant_group_size = lin_defaults.group_size,
        quant_overrides = overrides.len(),
        "Maple: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;
    let w = Weights::new(&shards, &idx);

    let pfx = w.resolve_prefix(&["model"], "word_embeddings.weight")?;
    info!(prefix = %pfx, "Maple: resolved tensor prefix");

    let device = Device::Gpu;

    let embed = load_embedding(
        &w,
        &format!("{pfx}.word_embeddings"),
        &embed_defaults,
        &overrides,
    )?;

    let norm = RmsNorm {
        weight: Some(load_rms_weight(&w, &format!("{pfx}.norm"))?),
        eps: cfg.rms_norm_eps,
    };

    let lm_head = if cfg.tie_word_embeddings {
        info!("Maple: tie_word_embeddings=true, skipping lm_head");
        None
    } else {
        let candidates = ["lm_head", "model.lm_head"];
        let mut base = "lm_head";
        let mut found = false;
        for cand in candidates {
            if w.has(&format!("{cand}.weight"))? {
                base = cand;
                found = true;
                break;
            }
        }
        if !found {
            return Err(Error::Loader(
                "maple: no lm_head.weight (and tie_word_embeddings is false)".to_owned(),
            ));
        }
        info!(%base, "Maple: loading lm_head");
        Some(load_linear(&w, base, &embed_defaults, &overrides, device)?)
    };

    let n_layers = usize::try_from(cfg.num_hidden_layers).map_err(|_| {
        Error::Config(format!(
            "maple: num_hidden_layers {} does not fit usize",
            cfg.num_hidden_layers
        ))
    })?;
    let mut layers = Vec::with_capacity(n_layers);

    for i in 0..n_layers {
        let base = format!("{pfx}.layers.{i}");
        let sa = format!("{base}.self_attn");
        let attn = MapleAttnWeights {
            q_proj: load_linear(
                &w,
                &format!("{sa}.q_proj"),
                &lin_defaults,
                &overrides,
                device,
            )?,
            k_proj: load_linear(
                &w,
                &format!("{sa}.k_proj"),
                &lin_defaults,
                &overrides,
                device,
            )?,
            v_proj: load_linear(
                &w,
                &format!("{sa}.v_proj"),
                &lin_defaults,
                &overrides,
                device,
            )?,
            o_proj: load_linear(
                &w,
                &format!("{sa}.o_proj"),
                &lin_defaults,
                &overrides,
                device,
            )?,
            q_norm: load_rms_weight(&w, &format!("{sa}.q_norm"))?,
            k_norm: load_rms_weight(&w, &format!("{sa}.k_norm"))?,
        };
        let mlp = load_mlp(
            &w,
            &format!("{base}.mlp"),
            &cfg,
            &lin_defaults,
            &overrides,
            device,
        )?;
        layers.push(MapleLayerWeights {
            input_layernorm: load_rms_weight(&w, &format!("{base}.input_layernorm"))?,
            post_attention_layernorm: load_rms_weight(
                &w,
                &format!("{base}.post_attention_layernorm"),
            )?,
            attn,
            mlp,
        });
    }

    info!(total_layers = n_layers, "Maple: all layers loaded");
    Ok(MapleLoadBundle {
        cfg,
        embed,
        layers,
        norm,
        lm_head,
        model_sig: crate::prompt_cache::model_cache_sig(model_dir),
    })
}

// ---------------------------------------------------------------------------
// Quant defaults / overrides
// ---------------------------------------------------------------------------

fn linear_defaults(cfg: &MapleConfig) -> QuantParams {
    let (bits, group_size, mode) = match cfg.quantization.as_ref() {
        Some(q) => (i32::from(q.bits), q.group_size, "affine"),
        None => (DEFAULT_LINEAR_BITS, DEFAULT_MOE_GROUP, "affine"),
    };
    QuantParams::global(group_size, bits, mode)
}

/// Inline per-tensor `{bits, group_size, mode}` entries from `quantization`.
///
/// Maple stores embed / lm_head overrides as sibling objects of `bits` /
/// `group_size` (`"lm_head"`, `"model.word_embeddings"`), same shape as
/// Laguna / Gemma4.
fn extract_quant_overrides(raw_quant: Option<&serde_json::Value>) -> HashMap<String, QuantParams> {
    let Some(quant_obj) = raw_quant.and_then(serde_json::Value::as_object) else {
        return HashMap::new();
    };
    let gs_default = quant_obj
        .get("group_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(DEFAULT_MOE_GROUP as u64) as i32;
    let bits_default = quant_obj
        .get("bits")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(DEFAULT_LINEAR_BITS as u64) as i32;
    let mut overrides = HashMap::new();
    for (key, val) in quant_obj {
        if matches!(
            key.as_str(),
            "group_size" | "bits" | "mode" | "tensor_overrides"
        ) {
            continue;
        }
        let Some(obj) = val.as_object() else {
            continue;
        };
        let ov_gs = obj
            .get("group_size")
            .and_then(serde_json::Value::as_u64)
            .map_or(gs_default, |v| v as i32);
        let ov_bits = obj
            .get("bits")
            .and_then(serde_json::Value::as_u64)
            .map_or(bits_default, |v| v as i32);
        let ov_mode = obj
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        overrides.insert(
            key.clone(),
            QuantParams {
                group_size: ov_gs,
                bits: ov_bits,
                mode: ov_mode,
            },
        );
    }
    overrides
}

// ---------------------------------------------------------------------------
// Layer builders
// ---------------------------------------------------------------------------

fn load_rms_weight(w: &Weights<'_>, name: &str) -> Result<Array> {
    bf16_param(w.array(&format!("{name}.weight"))?)
}

fn router_weight(gate: Linear) -> Result<Array> {
    match gate {
        Linear::Plain { weight } => Ok(weight),
        Linear::Quantized { .. } | Linear::Paro { .. } => Err(Error::Loader(
            "maple: mlp.gate must be a plain bf16 matrix".to_owned(),
        )),
    }
}

fn load_embedding(
    w: &Weights<'_>,
    base: &str,
    defaults: &QuantParams,
    overrides: &HashMap<String, QuantParams>,
) -> Result<Embedding> {
    match w.embedding(base, |hb| resolve_quant(base, hb, defaults, overrides))? {
        Embedding::Plain { weight } => Ok(Embedding::Plain {
            weight: bf16_param(weight)?,
        }),
        Embedding::Quantized {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        } => Ok(Embedding::Quantized {
            weight,
            scales: bf16_scales(scales)?,
            biases: biases.map(bf16_param).transpose()?,
            group_size,
            bits,
            mode,
        }),
    }
}

/// Load one linear: group-scales checkpoint, `row_alpha` 2-bit, or plain bf16.
fn load_linear(
    w: &Weights<'_>,
    base: &str,
    defaults: &QuantParams,
    overrides: &HashMap<String, QuantParams>,
    device: Device,
) -> Result<Linear> {
    let scales_name = format!("{base}.scales");
    if w.has(&scales_name)? {
        return shared_linear(w, base, defaults, overrides);
    }
    let alpha_name = format!("{base}.row_alpha");
    if w.has(&alpha_name)? {
        let packed = w.array(&format!("{base}.weight"))?;
        let alpha = w.array(&alpha_name)?;
        let group_size = defaults.group_size;
        let (scales, biases) = expand_row_alpha(alpha, &packed, group_size, device)?;
        return Ok(Linear::Quantized {
            weight: packed,
            scales,
            biases: Some(biases),
            group_size,
            bits: defaults.bits,
            mode: QuantMode::Affine,
        });
    }
    Ok(Linear::Plain {
        weight: bf16_param(w.array(&format!("{base}.weight"))?)?,
    })
}

fn shared_linear(
    w: &Weights<'_>,
    base: &str,
    defaults: &QuantParams,
    overrides: &HashMap<String, QuantParams>,
) -> Result<Linear> {
    match w.linear(base, |hb| resolve_quant(base, hb, defaults, overrides))? {
        Linear::Plain { weight } => Ok(Linear::Plain {
            weight: bf16_param(weight)?,
        }),
        Linear::Quantized {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        } => Ok(Linear::Quantized {
            weight,
            scales: bf16_scales(scales)?,
            biases: biases.map(bf16_param).transpose()?,
            group_size,
            bits,
            mode,
        }),
        Linear::Paro { .. } => Err(Error::Loader(format!(
            "{base}: unexpected Paro variant from w.linear"
        ))),
    }
}

fn load_mlp(
    w: &Weights<'_>,
    m: &str,
    cfg: &MapleConfig,
    defaults: &QuantParams,
    overrides: &HashMap<String, QuantParams>,
    device: Device,
) -> Result<MapleMlpWeights> {
    let expert0 = format!("{m}.experts.0.gate_proj.weight");
    let switch_w = format!("{m}.switch_mlp.gate_proj.weight");
    if w.has(&expert0)? || w.has(&switch_w)? {
        let gate = load_router(w, &format!("{m}.gate"))?;
        let gate_proj = load_expert_proj(
            w,
            m,
            "gate_proj",
            cfg.num_experts,
            defaults,
            overrides,
            device,
        )?;
        let up_proj = load_expert_proj(
            w,
            m,
            "up_proj",
            cfg.num_experts,
            defaults,
            overrides,
            device,
        )?;
        let down_proj = load_expert_proj(
            w,
            m,
            "down_proj",
            cfg.num_experts,
            defaults,
            overrides,
            device,
        )?;
        return Ok(MapleMlpWeights::Moe {
            gate,
            gate_proj,
            up_proj,
            down_proj,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
        });
    }
    Ok(MapleMlpWeights::Dense {
        gate_proj: load_linear(w, &format!("{m}.gate_proj"), defaults, overrides, device)?,
        up_proj: load_linear(w, &format!("{m}.up_proj"), defaults, overrides, device)?,
        down_proj: load_linear(w, &format!("{m}.down_proj"), defaults, overrides, device)?,
    })
}

/// Router is a plain bf16 matrix — never a QuantizedLinear.
fn load_router(w: &Weights<'_>, base: &str) -> Result<Linear> {
    if w.has(&format!("{base}.row_alpha"))? || w.has(&format!("{base}.scales"))? {
        return Err(Error::Loader(format!(
            "{base}: router gate is not quantized; found row_alpha/scales sibling"
        )));
    }
    Ok(Linear::Plain {
        weight: bf16_param(w.array(&format!("{base}.weight"))?)?,
    })
}

/// One SwitchMLP projection: stack per-expert tensors if needed, then expand
/// `row_alpha` (or pass through group-scales).
fn load_expert_proj(
    w: &Weights<'_>,
    mlp: &str,
    proj: &str,
    n_experts: i32,
    defaults: &QuantParams,
    overrides: &HashMap<String, QuantParams>,
    device: Device,
) -> Result<Linear> {
    let expert0 = format!("{mlp}.experts.0.{proj}.weight");
    if w.has(&expert0)? {
        return load_stacked_experts(w, mlp, proj, n_experts, defaults, device);
    }
    load_linear(
        w,
        &format!("{mlp}.switch_mlp.{proj}"),
        defaults,
        overrides,
        device,
    )
}

fn load_stacked_experts(
    w: &Weights<'_>,
    mlp: &str,
    proj: &str,
    n_experts: i32,
    defaults: &QuantParams,
    device: Device,
) -> Result<Linear> {
    let weight = stack_expert_leaf(w, mlp, proj, "weight", n_experts, device)?;
    let scales_leaf = format!("{mlp}.experts.0.{proj}.scales");
    let alpha_leaf = format!("{mlp}.experts.0.{proj}.row_alpha");
    if w.has(&scales_leaf)? {
        let scales = bf16_scales(stack_expert_leaf(
            w, mlp, proj, "scales", n_experts, device,
        )?)?;
        let biases = if w.has(&format!("{mlp}.experts.0.{proj}.biases"))? {
            Some(bf16_param(stack_expert_leaf(
                w, mlp, proj, "biases", n_experts, device,
            )?)?)
        } else {
            None
        };
        return Ok(Linear::Quantized {
            weight,
            scales,
            biases,
            group_size: defaults.group_size,
            bits: defaults.bits,
            mode: QuantMode::Affine,
        });
    }
    if w.has(&alpha_leaf)? {
        let alpha = stack_expert_leaf(w, mlp, proj, "row_alpha", n_experts, device)?;
        let (scales, biases) = expand_row_alpha(alpha, &weight, defaults.group_size, device)?;
        return Ok(Linear::Quantized {
            weight,
            scales,
            biases: Some(biases),
            group_size: defaults.group_size,
            bits: defaults.bits,
            mode: QuantMode::Affine,
        });
    }
    Ok(Linear::Plain {
        weight: bf16_param(weight)?,
    })
}

fn stack_expert_leaf(
    w: &Weights<'_>,
    mlp: &str,
    proj: &str,
    leaf: &str,
    n_experts: i32,
    device: Device,
) -> Result<Array> {
    let n = usize::try_from(n_experts)
        .map_err(|_| Error::Loader(format!("maple: num_experts {n_experts} does not fit usize")))?;
    if n == 0 {
        return Err(Error::Loader(format!(
            "{mlp}.experts.*.{proj}.{leaf}: num_experts is 0"
        )));
    }
    let mut parts = Vec::with_capacity(n);
    for e in 0..n {
        parts.push(w.array(&format!("{mlp}.experts.{e}.{proj}.{leaf}"))?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    let stacked = stack_axis(&refs, 0, device)?;
    stacked.eval()?;
    Ok(stacked)
}

// ---------------------------------------------------------------------------
// row_alpha → per-group scales / biases
// ---------------------------------------------------------------------------

/// Expand a per-row `α` into affine group `scales` / `biases`.
///
/// `packed` is uint32 with 16 2-bit codes per last-dim word.
/// `n_groups = (packed.shape[-1] * 16) / group_size`.
/// `alpha` is `[rows]` or `[rows, 1]` (or `[E, rows]` for stacked experts);
/// a trailing size-1 is squeezed so the result is `[…rows, n_groups]`.
fn expand_row_alpha(
    alpha: Array,
    packed: &Array,
    group_size: i32,
    device: Device,
) -> Result<(Array, Array)> {
    let packed_shape = packed.shape();
    let packed_last = packed_shape
        .last()
        .copied()
        .ok_or_else(|| Error::Loader("maple: packed weight has empty shape".to_owned()))?;
    let n_groups = n_groups_2bit(packed_last, group_size)?;
    let alpha = squeeze_to_row_rank(alpha, packed_shape.len(), device)?;
    let mut scale_shape = alpha.shape();
    scale_shape.push(n_groups);
    let expanded = expand_dims(&alpha, -1, device)?;
    let scales = broadcast_to(&expanded, &scale_shape, device)?;
    let scales = scales.contiguous(device)?;
    let scales = bf16_scales(scales)?;
    let biases = negative(&scales, device)?;
    let biases = bf16_param(biases)?;
    scales.eval()?;
    biases.eval()?;
    Ok((scales, biases))
}

fn n_groups_2bit(packed_last: i32, group_size: i32) -> Result<i32> {
    if group_size <= 0 {
        return Err(Error::Loader(format!(
            "maple: invalid group_size {group_size} for row_alpha expansion"
        )));
    }
    let codes = i64::from(packed_last) * CODES_PER_U32_2BIT;
    let gs = i64::from(group_size);
    if codes % gs != 0 {
        return Err(Error::Loader(format!(
            "maple: packed last dim {packed_last} * {CODES_PER_U32_2BIT} is not divisible by group_size {group_size}"
        )));
    }
    i32::try_from(codes / gs).map_err(|_| {
        Error::Loader(format!(
            "maple: n_groups overflow for packed last dim {packed_last}"
        ))
    })
}

/// Squeeze trailing 1s until `alpha.ndim() == packed.ndim() - 1` (the row axes).
fn squeeze_to_row_rank(mut alpha: Array, packed_ndim: usize, device: Device) -> Result<Array> {
    let row_ndim = packed_ndim.saturating_sub(1);
    while alpha.ndim() > row_ndim {
        let shape = alpha.shape();
        let last = shape.last().copied().unwrap_or(0);
        if last != 1 {
            return Err(Error::Loader(format!(
                "maple: row_alpha shape {shape:?} does not match packed rank {packed_ndim}"
            )));
        }
        let new_len = shape.len().saturating_sub(1);
        let new_shape: Vec<i32> = shape.into_iter().take(new_len).collect();
        alpha = alpha.reshape(&new_shape, device)?;
    }
    if alpha.ndim() != row_ndim {
        return Err(Error::Loader(format!(
            "maple: row_alpha rank {} != packed row rank {row_ndim}",
            alpha.ndim()
        )));
    }
    Ok(alpha)
}
