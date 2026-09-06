//! DFlash 2 drafter — config, weights and loader.
//!
//! DFlash 2 (z-lab / inco.ai) is the successor of the DFlash block drafter in
//! the sibling [`crate::speculative::dflash`] module. It keeps DFlash 1's
//! shape — a small standalone transformer that denoises a masked block in one
//! non-autoregressive pass, conditioned on the verifier's hidden states at
//! `target_layer_ids` projected through `fc` — and adds two weight families:
//!
//! * a **two-tap dynamic depthwise convolution** (`attention_conv`,
//!   `mlp_conv`) around each sublayer, whose per-position kernel is a
//!   per-channel base plus a per-group correction projected from the
//!   sublayer's input;
//! * a **candidate-path selector** (`candidate_selector`), which turns the
//!   block's independent per-position argmaxes into one chain by scoring
//!   adjacent (predecessor, successor) pairs against two rank-`selector_rank`
//!   vocabulary codebooks.
//!
//! # Status — document-the-truth (CLAUDE.md hard rule 7)
//!
//! **This module drafts one block; nothing drives it round after round yet.**
//! Config parsing, weight binding and shape validation against
//! `z-lab/Qwen3.8-27B-DFlash2` are wired and covered,
//! [`DFlash2Drafter::forward_hidden`] returns a block's final hidden states, and
//! [`DFlash2Drafter::select_chain`] turns those states and the verifier's
//! logits over them into one ordered draft chain — both checked against the
//! z-lab MLX reference. The round loop that would call them, verify the chain
//! and roll the caches back is not implemented, so the serve layer still
//! refuses a DFlash 2 draft snapshot before any weight is read rather than
//! running one of the other loops under this checkpoint's name.
//!
//! # Why its own module and its own [`crate::DraftKind`]
//!
//! Both generations declare `model_type = qwen3` and an `architectures[0]` of
//! `DFlash*DraftModel`, and the two checkpoints differ by 23 tensors out of 81.
//! Serving one as the other is a downgrade nothing downstream can see: a run's
//! `decode_config` records the kind and the block size, so an accept rate
//! measured from the wrong architecture is filed under the right name. The kind
//! is what keeps those rows apart.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::layers::{Activation, Linear, Mlp, RmsNorm};

mod forward;
mod selector;

/// The number of convolution sides a `base_kernel` carries: one kernel applied
/// to the sublayer's normed input, one to the sublayer's output.
const CONV_SIDES: usize = 2;

/// Taps this drafter's convolution reads: the current position and the one
/// before it.
const CONV_TAPS: usize = 2;

/// DFlash 2 drafter config, parsed from the draft snapshot's `config.json`.
///
/// Every field is **required**. DFlash 2 keeps `block_size` inside
/// `dflash_config` and its RoPE base inside `rope_parameters`, where DFlash 1
/// keeps both at the top level; a loader that defaults a missing key would read
/// the wrong block size off this checkpoint and record it as the checkpoint's
/// own.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed config struct — the complete DFlash 2 drafter contract, every field required by the checkpoint; adding a field requires updating parse_config and its refusal tests"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct DFlash2Config {
    /// Drafter hidden dimension (must equal the verifier's).
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// MLP inner width.
    pub intermediate_size: usize,
    /// Vocabulary size — the selector codebooks' row count.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency, from `rope_parameters.rope_theta`.
    pub rope_theta: f32,
    /// Attention window, in positions.
    pub sliding_window: usize,
    /// Whether the drafter's own attention is causal. DFlash 2 drafts the block
    /// bidirectionally, so this is `false` on the published checkpoint.
    pub is_causal: bool,
    /// Trained block size, including the seed token.
    pub block_size: usize,
    /// Channels per dynamic-kernel correction group.
    pub conv_group_size: usize,
    /// Convolution taps (2 = the current position and its predecessor).
    pub conv_kernel_size: usize,
    /// Selector codebook rank.
    pub selector_rank: usize,
    /// Candidates kept per block position before the path scoring.
    pub selector_top_k: usize,
    /// Token id filling the masked block positions.
    pub mask_token_id: u32,
    /// Verifier layer indices whose residuals the drafter conditions on.
    pub target_layer_ids: Vec<usize>,
}

/// One sublayer's two-tap dynamic depthwise convolution.
///
/// `base_kernel` is `[sides, taps, hidden]` — a per-channel kernel per side.
/// `kernel_projection` maps the sublayer's input to a per-position, per-group
/// correction added to it: `[sides * taps * (hidden / conv_group_size), hidden]`.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed weight struct — the complete tensor set of one dynamic convolution; adding a field requires updating load_conv"
)]
#[allow(missing_debug_implementations)]
pub struct DFlash2Conv {
    /// `[sides, taps, hidden]` per-channel base kernel.
    pub base_kernel: Array,
    /// `[sides * taps * groups, hidden]` correction projection.
    pub kernel_projection: Linear,
}

/// One DFlash 2 decoder layer: a Qwen3-shaped block with a dynamic convolution
/// wrapped around each of its two sublayers.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed weight struct — the complete tensor set of one decoder layer; adding a field requires updating load_layer"
)]
#[allow(missing_debug_implementations)]
pub struct DFlash2Layer {
    /// Pre-attention RMSNorm.
    pub input_layernorm: RmsNorm,
    /// Pre-MLP RMSNorm.
    pub post_attention_layernorm: RmsNorm,
    /// Query projection.
    pub q_proj: Linear,
    /// Key projection.
    pub k_proj: Linear,
    /// Value projection.
    pub v_proj: Linear,
    /// Attention output projection.
    pub o_proj: Linear,
    /// Per-head query RMSNorm.
    pub q_norm: RmsNorm,
    /// Per-head key RMSNorm.
    pub k_norm: RmsNorm,
    /// SwiGLU feed-forward.
    pub mlp: Mlp,
    /// Convolution around the attention sublayer.
    pub attention_conv: DFlash2Conv,
    /// Convolution around the MLP sublayer.
    pub mlp_conv: DFlash2Conv,
}

/// The candidate-path selector head.
///
/// The codebooks are stored **without a `.weight` suffix** — they are raw
/// parameters in the checkpoint, not `nn.Linear` weights.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed weight struct — the complete tensor set of the selector head; adding a field requires updating load_selector"
)]
#[allow(missing_debug_implementations)]
pub struct DFlash2Selector {
    /// `[selector_rank, hidden]` — projects the drafter's final hidden to the
    /// codebook rank.
    pub hidden_projection: Linear,
    /// `[vocab_size, selector_rank]` — the predecessor side of a pair score.
    pub predecessor_codebook: Array,
    /// `[vocab_size, selector_rank]` — the successor side of a pair score.
    pub successor_codebook: Array,
}

/// Loaded DFlash 2 drafter weights + config.
///
/// `embed_tokens` and `lm_head` are the verifier's, as in DFlash 1: the round
/// loop holds the verifier `Architecture` and threads them in.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed drafter struct — the complete DFlash 2 weight set; adding a field requires updating load_dflash2"
)]
#[allow(missing_debug_implementations)]
pub struct DFlash2Drafter {
    /// `fc`: `[hidden, len(target_layer_ids) * hidden]`, no bias.
    pub fc: Linear,
    /// RMSNorm on the projected conditioning hidden.
    pub hidden_norm: RmsNorm,
    /// Final RMSNorm after the decoder stack.
    pub norm: RmsNorm,
    /// The decoder stack.
    pub layers: Vec<DFlash2Layer>,
    /// The candidate-path selector head.
    pub selector: DFlash2Selector,
    /// Parsed config.
    pub cfg: DFlash2Config,
    /// Device the weights were loaded for.
    pub device: Device,
}

impl DFlash2Drafter {
    /// Load a DFlash 2 drafter from `draft_dir` and validate it against the
    /// verifier's hidden size.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when the snapshot does not declare itself DFlash 2,
    /// when any config key the drafter needs is absent, when a tensor is
    /// missing or carries a shape the config does not predict, or when the
    /// snapshot ships a tensor this loader does not read.
    pub fn load(draft_dir: &Path, hidden_size: usize, device: Device) -> Result<Self> {
        let me = load_dflash2(draft_dir, hidden_size, device)?;
        tracing::info!(
            draft = %draft_dir.display(),
            hidden_size,
            num_layers = me.cfg.num_hidden_layers,
            block_size = me.cfg.block_size,
            conv_group_size = me.cfg.conv_group_size,
            selector_rank = me.cfg.selector_rank,
            selector_top_k = me.cfg.selector_top_k,
            sliding_window = me.cfg.sliding_window,
            target_layer_ids = ?me.cfg.target_layer_ids,
            "DFlash2Drafter: loaded drafter"
        );
        Ok(me)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Read a required unsigned integer, naming its full path when it is absent.
fn req_usize(v: Option<&serde_json::Value>, path: &str, key: &str) -> Result<usize> {
    v.and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| missing(path, key))
}

/// Refusal for a config key the drafter cannot run without.
fn missing(path: &str, key: &str) -> Error {
    Error::Model(format!(
        "DFlash2Drafter: config.json has no {path}.{key}; every key the drafter \
         needs is read from the checkpoint, never defaulted — a default here \
         would be recorded as the checkpoint's own value"
    ))
}

/// Parse the DFlash 2 drafter config, refusing rather than defaulting.
///
/// `verifier_hidden` is the width the drafter must match; the conditioning
/// projection reads the verifier's hidden states directly.
///
/// # Errors
///
/// [`Error::Model`] naming the first key that is absent or inconsistent.
fn parse_config(
    cfg_raw: &rmlx_loader::ModelConfig,
    verifier_hidden: usize,
) -> Result<DFlash2Config> {
    let arch = cfg_raw.architectures.first().map_or("", String::as_str);
    if !arch.contains("DFlash2") {
        return Err(Error::Model(format!(
            "DFlash2Drafter: the snapshot declares architectures[0] {arch:?}, not \
             DFlash2DraftModel; this loader builds the DFlash 2 architecture and no other"
        )));
    }

    let top = &cfg_raw.extras;
    let dflash = cfg_raw
        .extras
        .get("dflash_config")
        .ok_or_else(|| missing("config.json", "dflash_config"))?;

    let rope = cfg_raw.extras.get("rope_parameters");
    let rope_theta = rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| missing("rope_parameters", "rope_theta"))?;
    let rope_type = rope
        .and_then(|r| r.get("rope_type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| missing("rope_parameters", "rope_type"))?;
    if rope_type != "default" {
        return Err(Error::Model(format!(
            "DFlash2Drafter: rope_parameters.rope_type is {rope_type:?}; this loader \
             applies plain RoPE and has no code for a scaled one, which would \
             mis-position every draft rather than fail"
        )));
    }

    let rms_norm_eps = top
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| missing("config.json", "rms_norm_eps"))?;
    let is_causal = top
        .get("is_causal")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| missing("config.json", "is_causal"))?;
    let target_layer_ids: Vec<usize> = dflash
        .get("target_layer_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|v| v as usize))
                .collect::<Vec<usize>>()
        })
        .filter(|ids| !ids.is_empty())
        .ok_or_else(|| missing("dflash_config", "target_layer_ids"))?;
    let mask_token_id = req_usize(
        dflash.get("mask_token_id"),
        "dflash_config",
        "mask_token_id",
    )?;

    let cfg = DFlash2Config {
        hidden_size: req_usize(top.get("hidden_size"), "config.json", "hidden_size")?,
        num_hidden_layers: req_usize(
            top.get("num_hidden_layers"),
            "config.json",
            "num_hidden_layers",
        )?,
        num_attention_heads: req_usize(
            top.get("num_attention_heads"),
            "config.json",
            "num_attention_heads",
        )?,
        num_key_value_heads: req_usize(
            top.get("num_key_value_heads"),
            "config.json",
            "num_key_value_heads",
        )?,
        head_dim: req_usize(top.get("head_dim"), "config.json", "head_dim")?,
        intermediate_size: req_usize(
            top.get("intermediate_size"),
            "config.json",
            "intermediate_size",
        )?,
        vocab_size: req_usize(top.get("vocab_size"), "config.json", "vocab_size")?,
        rms_norm_eps,
        rope_theta,
        sliding_window: req_usize(top.get("sliding_window"), "config.json", "sliding_window")?,
        is_causal,
        block_size: req_usize(dflash.get("block_size"), "dflash_config", "block_size")?,
        conv_group_size: req_usize(
            dflash.get("conv_group_size"),
            "dflash_config",
            "conv_group_size",
        )?,
        conv_kernel_size: req_usize(
            dflash.get("conv_kernel_size"),
            "dflash_config",
            "conv_kernel_size",
        )?,
        selector_rank: req_usize(
            dflash.get("selector_rank"),
            "dflash_config",
            "selector_rank",
        )?,
        selector_top_k: req_usize(
            dflash.get("selector_top_k"),
            "dflash_config",
            "selector_top_k",
        )?,
        mask_token_id: u32::try_from(mask_token_id).map_err(|_| {
            Error::Model("DFlash2Drafter: dflash_config.mask_token_id exceeds u32".into())
        })?,
        target_layer_ids,
    };

    check_config(&cfg, cfg_raw, verifier_hidden)?;
    Ok(cfg)
}

/// Refuse a config the loader can parse but the drafter cannot run.
///
/// Each of these is a property the forward assumes and could not detect: a
/// three-tap kernel convolved with two taps, a group size that does not divide
/// the channels, a full-attention layer masked as a sliding one.
fn check_config(
    cfg: &DFlash2Config,
    cfg_raw: &rmlx_loader::ModelConfig,
    verifier_hidden: usize,
) -> Result<()> {
    if cfg.hidden_size != verifier_hidden {
        return Err(Error::Model(format!(
            "DFlash2Drafter: drafter hidden_size {} != verifier hidden_size {verifier_hidden} \
             (wrong draft model?)",
            cfg.hidden_size
        )));
    }
    if cfg.conv_kernel_size != CONV_TAPS {
        return Err(Error::Model(format!(
            "DFlash2Drafter: dflash_config.conv_kernel_size is {}; this drafter's \
             convolution is two-tap (the position and its predecessor) and would \
             silently drop the rest",
            cfg.conv_kernel_size
        )));
    }
    if !cfg.hidden_size.is_multiple_of(cfg.conv_group_size) {
        return Err(Error::Model(format!(
            "DFlash2Drafter: dflash_config.conv_group_size {} does not divide \
             hidden_size {}; the dynamic kernel has one correction per group of \
             channels and cannot cover a partial group",
            cfg.conv_group_size, cfg.hidden_size
        )));
    }
    if cfg.num_key_value_heads == 0
        || !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
    {
        return Err(Error::Model(format!(
            "DFlash2Drafter: num_attention_heads {} is not a multiple of \
             num_key_value_heads {}; grouped-query attention repeats each KV head \
             a whole number of times and the projections' shapes do not show it",
            cfg.num_attention_heads, cfg.num_key_value_heads
        )));
    }
    if cfg.selector_top_k < 2 {
        return Err(Error::Model(format!(
            "DFlash2Drafter: dflash_config.selector_top_k is {}; the path selector \
             chooses between candidates and has nothing to choose from below two",
            cfg.selector_top_k
        )));
    }
    if cfg.selector_top_k > cfg.vocab_size {
        return Err(Error::Model(format!(
            "DFlash2Drafter: dflash_config.selector_top_k is {}, more candidates \
             than the vocabulary of {} holds; the selector's partition has no \
             {}th-largest logit to keep",
            cfg.selector_top_k, cfg.vocab_size, cfg.selector_top_k
        )));
    }
    if cfg.block_size < 2 {
        return Err(Error::Model(format!(
            "DFlash2Drafter: dflash_config.block_size is {}; a block of one is the \
             seed token alone and drafts nothing",
            cfg.block_size
        )));
    }
    if cfg.sliding_window < 2 {
        return Err(Error::Model(format!(
            "DFlash2Drafter: sliding_window is {}; the window holds the block \
             position and the conditioning rows behind it, and below two it \
             reaches back past no row at all",
            cfg.sliding_window
        )));
    }

    let layer_types: Vec<&str> = cfg_raw
        .extras
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
        .ok_or_else(|| missing("config.json", "layer_types"))?;
    if layer_types.len() != cfg.num_hidden_layers
        || layer_types.iter().any(|t| *t != "sliding_attention")
    {
        return Err(Error::Model(format!(
            "DFlash2Drafter: layer_types {layer_types:?} is not {} sliding_attention \
             layers; every layer of this drafter attends over one window and a \
             full-attention layer would be given the wrong mask",
            cfg.num_hidden_layers
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load the DFlash 2 drafter config + tensors from `draft_dir`.
fn load_dflash2(draft_dir: &Path, hidden_size: usize, device: Device) -> Result<DFlash2Drafter> {
    use rmlx_loader::{load_config, load_shard_index, ShardSet};

    let cfg_raw = load_config(draft_dir)
        .map_err(|e| Error::Model(format!("DFlash2Drafter: load_config: {e}")))?;
    let cfg = parse_config(&cfg_raw, hidden_size)?;

    let idx = load_shard_index(draft_dir)
        .map_err(|e| Error::Model(format!("DFlash2Drafter: shard index: {e}")))?;
    let shards = ShardSet::open(draft_dir, &idx)
        .map_err(|e| Error::Model(format!("DFlash2Drafter: open: {e}")))?;

    // Which tensor names the loader consumed, checked against the snapshot
    // below: a DFlash generation newer than this loader would otherwise be
    // built out of the subset it recognises and recorded as itself.
    let consumed: RefCell<HashSet<String>> = RefCell::new(HashSet::new());

    // Every tensor is loaded with the shape the config predicts, so a config
    // read from the wrong keys fails on the first weight rather than at the
    // first draft.
    let load = |name: &str, want: &[usize]| -> Result<Array> {
        consumed.borrow_mut().insert(name.to_owned());
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("DFlash2Drafter: safetensors: {e}")))?;
            if let Ok(t) = st.tensor(name) {
                if t.shape() != want {
                    return Err(Error::Model(format!(
                        "DFlash2Drafter: tensor '{name}' has shape {:?}, not the {want:?} \
                         its config predicts",
                        t.shape()
                    )));
                }
                let tv = rmlx_loader::TensorView {
                    name,
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    bytes: t.data(),
                };
                return Array::from_safetensor_view(&tv);
            }
        }
        Err(Error::Model(format!(
            "DFlash2Drafter: tensor '{name}' not found"
        )))
    };
    let lin = |name: &str, want: &[usize]| -> Result<Linear> {
        Ok(Linear::Plain {
            weight: load(name, want)?,
        })
    };
    let norm = |name: &str, width: usize| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Some(load(name, &[width])?),
            eps: cfg.rms_norm_eps,
        })
    };

    let h = cfg.hidden_size;
    let q_out = cfg.num_attention_heads * cfg.head_dim;
    let kv_out = cfg.num_key_value_heads * cfg.head_dim;
    let conv_groups = h / cfg.conv_group_size;
    let conv_proj_out = CONV_SIDES * cfg.conv_kernel_size * conv_groups;

    let fc = lin("fc.weight", &[h, cfg.target_layer_ids.len() * h])?;
    let hidden_norm = norm("hidden_norm.weight", h)?;
    let final_norm = norm("norm.weight", h)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        let conv = |which: &str| -> Result<DFlash2Conv> {
            Ok(DFlash2Conv {
                base_kernel: load(
                    &format!("{p}.{which}.base_kernel"),
                    &[CONV_SIDES, cfg.conv_kernel_size, h],
                )?,
                kernel_projection: lin(
                    &format!("{p}.{which}.kernel_projection.weight"),
                    &[conv_proj_out, h],
                )?,
            })
        };
        layers.push(DFlash2Layer {
            input_layernorm: norm(&format!("{p}.input_layernorm.weight"), h)?,
            post_attention_layernorm: norm(&format!("{p}.post_attention_layernorm.weight"), h)?,
            q_proj: lin(&format!("{p}.self_attn.q_proj.weight"), &[q_out, h])?,
            k_proj: lin(&format!("{p}.self_attn.k_proj.weight"), &[kv_out, h])?,
            v_proj: lin(&format!("{p}.self_attn.v_proj.weight"), &[kv_out, h])?,
            o_proj: lin(&format!("{p}.self_attn.o_proj.weight"), &[h, q_out])?,
            q_norm: norm(&format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
            k_norm: norm(&format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
            mlp: Mlp {
                gate_proj: lin(
                    &format!("{p}.mlp.gate_proj.weight"),
                    &[cfg.intermediate_size, h],
                )?,
                up_proj: lin(
                    &format!("{p}.mlp.up_proj.weight"),
                    &[cfg.intermediate_size, h],
                )?,
                down_proj: lin(
                    &format!("{p}.mlp.down_proj.weight"),
                    &[h, cfg.intermediate_size],
                )?,
                activation: Activation::Silu,
            },
            attention_conv: conv("attention_conv")?,
            mlp_conv: conv("mlp_conv")?,
        });
    }

    // The codebooks carry no `.weight` suffix: they are raw parameters, not
    // `nn.Linear` weights, and looking for one finds nothing.
    let selector = DFlash2Selector {
        hidden_projection: lin(
            "candidate_selector.hidden_projection.weight",
            &[cfg.selector_rank, h],
        )?,
        predecessor_codebook: load(
            "candidate_selector.predecessor_codebook",
            &[cfg.vocab_size, cfg.selector_rank],
        )?,
        successor_codebook: load(
            "candidate_selector.successor_codebook",
            &[cfg.vocab_size, cfg.selector_rank],
        )?,
    };

    // A set, not a list: a name carried by two shard files would otherwise be
    // counted and listed twice in the refusal.
    let mut present: HashSet<String> = HashSet::new();
    for (_, handle) in shards.iter() {
        let st = handle
            .safetensors()
            .map_err(|e| Error::Model(format!("DFlash2Drafter: safetensors: {e}")))?;
        present.extend(st.names().into_iter().map(ToOwned::to_owned));
    }
    super::unread_tensor_refusal("DFlash2Drafter", &present, &consumed.borrow())?;

    Ok(DFlash2Drafter {
        fc,
        hidden_norm,
        norm: final_norm,
        layers,
        selector,
        cfg,
        device,
    })
}

#[cfg(test)]
mod tests;
