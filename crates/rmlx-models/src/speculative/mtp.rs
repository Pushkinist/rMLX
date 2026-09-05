//! MTP (Multi-Token Prediction) drafter loader + round-loop.
//!
//! Port of mlx-vlm `mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`
//! (`Qwen3_5MTPDraftModel`) and the round-loop in
//! `mlx_vlm/speculative/mtp.py` (`_mtp_verify_target`,
//! `_mtp_rounds`).
//!
//! # What an MTP drafter is
//!
//! Unlike the two-full-model [`super::SpeculativeDispatcher`] path (verifier +
//! independent draft `Architecture`), an MTP drafter is a *sidecar head* that
//! conditions on the verifier's **penultimate hidden state** (the decoder-trunk
//! output before the final RMSNorm + LM head — captured via
//! [`crate::arch::Architecture::forward_verify_capture`] at the last layer). For
//! each draft step it concatenates the input-token embedding with that hidden
//! state, projects `2H -> H` through its `fc` linear, runs one small decoder
//! layer (its own KV cache), normalises, and re-uses the *target's* LM head to
//! pick the next token.
//!
//! Reference weight layout (qwen3.5 `mtp.*` sidecar, `mtp.` prefix stripped):
//! `fc.weight` (2H->H), `pre_fc_norm_embedding.weight`,
//! `pre_fc_norm_hidden.weight`, `layers.{0..}.*` (Qwen3.5-MoE decoder layer),
//! `norm.weight` (final RMSNorm). See `qwen3_5_mtp/split.py`.
//!
//! # Status — document-the-truth (CLAUDE.md hard rule 7)
//!
//! **Fully wired + live-validated** against the
//! `mlx-community/Qwen3.6-35B-A3B-MTP-5bit` sidecar +
//! `mlx-community/Qwen3.6-35B-A3B-8bit` verifier. The drafter's single decoder
//! layer is the **reused** Qwen3.5-MoE `DecoderLayer` (full-attention + sparse
//! MoE — identical tensor names to the verifier; see
//! [`crate::qwen3_5_moe::MtpLayer`]), so there is no second hand-ported attention
//! / MoE implementation. The verifier embedding accessor reuses
//! [`crate::arch::Architecture::embed_tokens_raw`] (the same seam DFlash uses)
//! and the LM head reuses [`crate::arch::Architecture::logits_from_hidden`]. The
//! conditioning hidden comes from [`crate::arch::Architecture::forward_verify_capture`]
//! capturing the verifier's last decoder layer (penultimate, pre-final-norm).

#![allow(
    clippy::implicit_clone,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::unused_self,
    clippy::used_underscore_binding
)]
// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, concatenate, Array, Device};

use super::{emit_step, DecodeWindow};
use crate::arch::Architecture;
use crate::layers::{Linear, RmsNorm};
use crate::qwen3_5_moe::{MtpLayer, MtpLayerDims};
use rmlx_kv_quant::{KvCache, KvQuant};

/// Loaded MTP-head sidecar weights (Qwen3.5 `mtp.*`, prefix stripped).
///
/// Holds the `fc` projection, the three RMSNorms, and the reused Qwen3.5-MoE
/// decoder layer(s). The per-layer decoder compute runs through the existing
/// [`MtpLayer`] (no second attention/MoE port).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed weight bundle struct — consumed by MtpDrafter; adding a field requires updating load_mtp_head and MtpDrafter::load"
)]
#[allow(missing_debug_implementations)]
pub struct MtpHeadWeights {
    /// `fc`: Linear `2*hidden -> hidden`, no bias.
    pub fc: Linear,
    /// Pre-fc RMSNorm applied to the input-token embedding branch.
    pub pre_fc_norm_embedding: RmsNorm,
    /// Pre-fc RMSNorm applied to the conditioning hidden-state branch.
    pub pre_fc_norm_hidden: RmsNorm,
    /// Final RMSNorm after the decoder layer(s).
    pub norm: RmsNorm,
    /// Reused Qwen3.5-MoE decoder layer(s) (`mtp_num_hidden_layers`, usually 1).
    pub layers: Vec<MtpLayer>,
    /// Model hidden size `H`.
    pub hidden_size: usize,
}

/// MTP drafter: a verifier-conditioned sidecar head with its own KV cache.
///
/// Construct with [`MtpDrafter::load`]. `draft_n` mirrors
/// `Qwen3_5MTPDraftModel.draft_block`: autoregressive K-step drafting from a
/// seed `(token, hidden)` pair, re-using the verifier's input embeddings and LM
/// head (threaded in by the round-loop, which holds the verifier `Architecture`).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed drafter struct — private weight + cache fields; public API is draft_n()/forward; adding a field requires updating MtpDrafter::load"
)]
#[allow(missing_debug_implementations)]
pub struct MtpDrafter {
    weights: MtpHeadWeights,
    /// Per-MTP-layer KV cache (the head's own small cache).
    caches: Vec<KvCache>,
    /// Drafter block size (`block_size` in the sidecar config) — the number of
    /// tokens proposed per round (incl. the seed carry).
    block_size: usize,
    device: Device,
}

impl MtpDrafter {
    /// Load an MTP-head sidecar from `draft_dir` and validate it against the
    /// verifier's hidden size.
    ///
    /// `draft_dir` is the standalone drafter folder produced by
    /// `qwen3_5_mtp/split.py`: a `config.json` (`model_type: "qwen3_5_mtp"`) and
    /// a `model.safetensors` with the `mtp.`-stripped tensors. `hidden_size` is
    /// the verifier's model width — `fc` must be `[hidden, 2*hidden]`.
    pub fn load(draft_dir: &Path, hidden_size: usize, device: Device) -> Result<Self> {
        let (weights, block_size) = load_mtp_head(draft_dir, hidden_size)?;
        let caches = (0..weights.layers.len())
            .map(|_| KvCache::with_quant(KvQuant::None))
            .collect();
        tracing::info!(
            draft = %draft_dir.display(),
            hidden_size,
            num_mtp_layers = weights.layers.len(),
            block_size,
            "MtpDrafter: loaded sidecar head"
        );
        Ok(Self {
            weights,
            caches,
            block_size,
            device,
        })
    }

    /// Reset the head's KV cache between generations.
    pub fn reset(&mut self) {
        for c in &mut self.caches {
            *c = KvCache::with_quant(KvQuant::None);
        }
    }

    /// Configured block size (tokens proposed per round, including the carry).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Current KV write offset of the head's first (full-attention) layer.
    pub fn offset(&self) -> i32 {
        self.caches.first().map_or(0, |c| c.offset())
    }

    /// Roll the head's KV cache back to `target` positions (partial accept).
    ///
    /// A layer holding fewer than `target` positions is skipped rather than
    /// grown — but that is a fill the caller's accounting did not predict, and
    /// silently skipping it lets a slot-vs-position gap open one step at a time
    /// and show up only as a quietly decaying accept rate. Say so.
    pub fn truncate_to(&mut self, target: i32) -> Result<()> {
        for c in &mut self.caches {
            if c.offset() >= target {
                c.truncate_to(target)?;
            } else {
                tracing::warn!(
                    target_positions = target,
                    fill = c.offset(),
                    "MtpDrafter::truncate_to: sidecar layer holds fewer positions than the \
                     rollback target — the sidecar KV is behind the verifier prefix"
                );
            }
        }
        Ok(())
    }

    /// Project one `(token_embed, hidden)` pair through the `fc` + pre-fc norms.
    ///
    /// Mirrors `Qwen3_5MTPDraftModel._forward_hidden` up to (not including) the
    /// decoder layers: `fc(concat[norm_e(embed), norm_h(hidden)])`. `token_embed`
    /// and `hidden` are both `[1, n, H]`; returns `[1, n, H]`.
    pub fn project(&self, token_embed: &Array, hidden: &Array) -> Result<Array> {
        let e = self
            .weights
            .pre_fc_norm_embedding
            .forward(token_embed, self.device)?;
        let h = self
            .weights
            .pre_fc_norm_hidden
            .forward(hidden, self.device)?;
        let cat = concatenate(&[&e, &h], -1, self.device)?;
        self.weights.fc.forward(&cat, self.device)
    }

    /// Autoregressive K-step draft (mirrors `draft_block`).
    ///
    /// Given the seed token id `seed_tok`, its conditioning `hidden` (`[1,1,H]`,
    /// the verifier penultimate state at the seed position), the verifier
    /// `Architecture` (for `embed_tokens_raw` + `logits_from_hidden`), and the
    /// `start_offset` (the sidecar's `_next_position`, = verifier prefix length),
    /// produce up to `block_size - 1` draft token ids. Greedy (temp=0).
    ///
    /// The head's KV cache advances one position per drafted token; the RoPE /
    /// KV write offset is `start_offset + step`. The first step conditions on the
    /// verifier hidden; subsequent steps condition on the head's own previous
    /// output hidden (mirrors `_forward_token` re-feeding `h_prev`).
    ///
    /// The loop produces `block_size - 1` tokens but the last one is never fed
    /// back, so it would get no KV slot: a full-accept round then commits
    /// `block_size` verifier positions against `block_size - 1` sidecar slots
    /// and the two drift apart by one per round, permanently. One extra
    /// `forward_token` at the end closes it — the hidden it returns is
    /// discarded, only the KV write matters — so the sidecar always leaves this
    /// call holding a slot for every token it has seen or proposed.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "the `.unwrap()` sites are infallible `<[u8; 4]>::try_from` on a 4-byte `to_bytes()` slice (argmax id decode); width is fixed by construction"
    )]
    pub fn draft_n(
        &mut self,
        verifier: &Architecture,
        seed_tok: u32,
        hidden: &Array,
        block_size: usize,
        start_offset: i32,
    ) -> Result<Vec<u32>> {
        if block_size <= 1 {
            return Ok(vec![]);
        }
        let mut tok = seed_tok;
        let mut h_prev = hidden.try_clone()?;
        let mut tokens: Vec<u32> = Vec::with_capacity(block_size - 1);

        while tokens.len() < block_size - 1 {
            let offset = start_offset + tokens.len() as i32;
            // Embed the current token via the *target* embeddings (no scale).
            let tok_embed = self.embed_token(verifier, tok)?;
            // Project + run the head decoder layer(s) -> next hidden [1,1,H].
            let h_next = self.forward_token(&tok_embed, &h_prev, offset)?;
            // Re-use the verifier LM head to pick the next draft token (greedy).
            // `forward_token` ends with the sidecar's own final norm, so the
            // verifier's norm must not be applied a second time.
            let logits = verifier.logits_from_final_hidden(&h_next, self.device)?;
            let next = argmax(&logits, -1, self.device)?;
            next.eval()?;
            let id = u32::from_le_bytes(next.to_bytes()?[..4].try_into().unwrap());
            tokens.push(id);
            tok = id;
            h_prev = h_next;
        }
        // Give the last drafted token its slot (see the note above). The loop's
        // last write was at `start_offset + tokens.len() - 1`, so this is both
        // the next free slot and that token's position. The hidden it returns is
        // discarded — the next round re-seeds from the verifier's captured one.
        let last_offset = start_offset + tokens.len() as i32;
        let tok_embed = self.embed_token(verifier, tok)?;
        let _ = self.forward_token(&tok_embed, &h_prev, last_offset)?;
        Ok(tokens)
    }

    /// Embed a single token id through the verifier's input embeddings.
    ///
    /// Reuses [`Architecture::embed_tokens_raw`] (the Qwen3.5-MoE sidecar's
    /// `embed_tokens` is a bare `nn.Embedding`, `embed_scale = 1.0`). Returns
    /// `[1, 1, H]`.
    fn embed_token(&self, verifier: &Architecture, tok: u32) -> Result<Array> {
        verifier.embed_tokens_raw(&[tok as i32], self.device)
    }

    /// One MTP decoder-layer forward over the head's own KV cache.
    ///
    /// Mirrors `Qwen3_5MTPDraftModel._forward_token`: `project` -> reused
    /// Qwen3.5-MoE decoder layer(s) (full-attention GQA + per-head q/k RMSNorm +
    /// partial RoPE over `self.caches` at `offset`) -> final `norm`. Returns
    /// `[1, 1, H]`.
    fn forward_token(&mut self, token_embed: &Array, hidden: &Array, offset: i32) -> Result<Array> {
        let mut h = self.project(token_embed, hidden)?;
        for (layer, cache) in self.weights.layers.iter().zip(self.caches.iter_mut()) {
            h = layer.forward(&h, offset, cache, self.device)?;
        }
        self.weights.norm.forward(&h, self.device)
    }

    /// Hidden size the head was loaded for.
    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }
}

/// Load the MTP-head sidecar tensors from `draft_dir`.
///
/// Reads `model.safetensors` (qwen3.5 split layout) and constructs the `fc`
/// linear + three RMSNorms + the reused Qwen3.5-MoE decoder layer(s). Validates
/// `fc` shape `[hidden, 2*hidden]` against the verifier `hidden_size`. Returns
/// `(weights, block_size)`.
///
/// Norm-weight contract: the qwen3.5 sidecar split (`qwen3_5_mtp.py::sanitize`)
/// adds 1.0 to every 1-D norm weight ONLY when the source is NOT already an
/// mlx-format checkpoint. The `mlx-community` MTP snapshots are mlx-format, so
/// the split stores weights verbatim (no +1) — we load them verbatim and apply
/// a plain `rms_norm` (matching the verifier's own RmsNorm), so no centring
/// shift is added here.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn load_mtp_head(draft_dir: &Path, hidden_size: usize) -> Result<(MtpHeadWeights, usize)> {
    use rmlx_loader::{load_config, load_shard_index, ShardSet};

    let cfg = load_config(draft_dir).map_err(|e| {
        Error::Model(format!(
            "MtpDrafter: load_config({}): {e}",
            draft_dir.display()
        ))
    })?;
    let arch = cfg.architectures.first().map_or("", String::as_str);
    let model_type = cfg
        .extras
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if arch != "qwen3_5_mtp" && model_type != "qwen3_5_mtp" {
        tracing::warn!(
            draft = %draft_dir.display(),
            %arch,
            %model_type,
            "MtpDrafter: expected model_type=qwen3_5_mtp; proceeding by tensor names"
        );
    }

    // The MTP head dims live in `text_config` (mirrors the verifier's). The
    // loader parses `text_config` into a typed field (not `extras`), so re-read
    // the raw config.json to reach the full nested object (incl. `head_dim`,
    // `num_experts`, nested `rope_parameters`). Mirrors the qwen3_5_moe loader.
    let raw_json: serde_json::Value = {
        let path = draft_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Model(format!("MtpDrafter: read {}: {e}", path.display())))?;
        serde_json::from_slice(&data)
            .map_err(|e| Error::Model(format!("MtpDrafter: malformed config.json: {e}")))?
    };
    // `text_config` carries the per-arch dims. It MUST be present — defaulting to
    // 35B-A3B constants would silently mis-shape the drafter for another variant
    // (e.g. Qwen3.5-9B-MTP) and the verifier would mask it as a collapsed
    // accept-rate, not an error. Fail loud instead.
    let tc = raw_json
        .get("text_config")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            Error::Model(
                "MtpDrafter: config.json missing `text_config` — cannot resolve drafter dims"
                    .to_owned(),
            )
        })?;
    // Critical dims are required (no default): a wrong head/expert count is a
    // silent corruption, so absence must error.
    let tc_u64_req = |k: &str| {
        tc.get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| Error::Model(format!("MtpDrafter: text_config missing `{k}`")))
    };
    let tc_bool = |k: &str, d: bool| tc.get(k).and_then(serde_json::Value::as_bool).unwrap_or(d);
    let tc_f64 = |k: &str, d: f64| tc.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    let tc_u64 = |k: &str, d: usize| {
        tc.get(k)
            .and_then(serde_json::Value::as_u64)
            .map_or(d, |v| v as usize)
    };
    let tc_opt_u64 = |k: &str| {
        tc.get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
    };

    let rms_eps = tc_f64("rms_norm_eps", 1e-6) as f32;
    let num_attention_heads = tc_u64_req("num_attention_heads")?;
    let num_key_value_heads = tc_u64_req("num_key_value_heads")?;
    let head_dim = tc_u64_req("head_dim")?;
    // MoE dims are optional for the same reason they are optional in
    // `Qwen3_5MoeConfig`: a sidecar whose `layers.0` FFN is a plain SwiGLU omits
    // them entirely. `num_experts == 0` is the shared "dense, no experts"
    // sentinel; `MtpLayer::load` decides dense-vs-MoE from tensor facts and
    // cross-checks it against this value.
    //
    // `num_experts_per_tok` has no such sentinel — every value it can take is a
    // legal routing width, so a default would turn an omitted key into top-1
    // routing on a top-8 checkpoint and collapse draft quality silently. Carry
    // the absence instead and let the MoE branch of `MtpLayer::load` refuse it;
    // the dense branch never reads it, so an absent key stays legal exactly
    // where a dense sidecar needs it to be.
    let num_experts = tc_u64("num_experts", 0);
    let num_experts_per_tok = tc_opt_u64("num_experts_per_tok");
    let norm_topk_prob = tc_bool("norm_topk_prob", true);

    // RoPE: read rope_theta + partial_rotary_factor from `rope_parameters`
    // (preferred) then top-level text_config. `rope_theta` may be absent — fall
    // back to the Qwen3.5-MoE default (1e7).
    let rope = tc.get("rope_parameters").and_then(|v| v.as_object());
    let rope_f64 = |k: &str, d: f64| {
        rope.and_then(|m| m.get(k))
            .or_else(|| tc.get(k))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(d)
    };
    let rope_theta = rope_f64("rope_theta", 10_000_000.0) as f32;
    let partial_rotary_factor = rope_f64("partial_rotary_factor", 0.25);
    let rope_dims = ((head_dim as f64) * partial_rotary_factor).round() as usize;

    // block_size (tokens proposed per round, incl. carry).
    let block_size = cfg
        .extras
        .get("block_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(3) as usize;

    // Sidecar global quant (group_size / bits / mode).
    let (q_gs, q_bits, q_mode) = match &cfg.quantization {
        Some(q) => (
            q.group_size as i32,
            i32::from(q.bits),
            q.mode_or_default().to_owned(),
        ),
        None => (64, 8, "affine".to_owned()),
    };

    let idx = load_shard_index(draft_dir)
        .map_err(|e| Error::Model(format!("MtpDrafter: shard index: {e}")))?;
    let shards = ShardSet::open(draft_dir, &idx)
        .map_err(|e| Error::Model(format!("MtpDrafter: open: {e}")))?;

    // Resolve a single named tensor from any shard (idiom mirrors qwen3 loader).
    fn load_array(shards: &ShardSet, name: &str) -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("MtpDrafter: safetensors: {e}")))?;
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
        Err(Error::Model(format!(
            "MtpDrafter: tensor '{name}' not found"
        )))
    }
    fn has_tensor(shards: &ShardSet, name: &str) -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn all_tensor_names(shards: &ShardSet) -> Vec<String> {
        let mut names = Vec::new();
        for (_, handle) in shards.iter() {
            if let Ok(st) = handle.safetensors() {
                names.extend(st.names().into_iter().map(|s| s.to_owned()));
            }
        }
        names
    }

    // `fc.weight`: quantized in the 5-bit sidecar — shape `[hidden, 2*hidden*bits/32]`
    // for U32-packed weights, or `[hidden, 2*hidden]` when plain. Validate the
    // output dim (rows) matches the verifier hidden; the input dim is encoded in
    // the scales for quantized weights, so we validate `fc` by building it.
    let fc = {
        let w = load_array(&shards, "fc.weight")?;
        let fc_shape = w.shape().to_vec();
        if fc_shape.len() != 2 || fc_shape[0] as usize != hidden_size {
            return Err(Error::Model(format!(
                "MtpDrafter: fc.weight shape {fc_shape:?} row dim != verifier \
                 hidden_size {hidden_size} (wrong draft model?)"
            )));
        }
        if has_tensor(&shards, "fc.scales") {
            let scales = load_array(&shards, "fc.scales")?;
            let biases = if has_tensor(&shards, "fc.biases") {
                Some(load_array(&shards, "fc.biases")?)
            } else {
                None
            };
            Linear::Quantized {
                weight: w,
                scales,
                biases,
                group_size: q_gs,
                bits: q_bits,
                mode: crate::layers::QuantMode::from(q_mode.as_str()),
            }
        } else {
            // Plain fc must be exactly [hidden, 2*hidden].
            if fc_shape[1] as usize != 2 * hidden_size {
                return Err(Error::Model(format!(
                    "MtpDrafter: plain fc.weight shape {fc_shape:?} != [{hidden_size}, {}]",
                    2 * hidden_size
                )));
            }
            Linear::Plain { weight: w }
        }
    };

    let load_norm = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Some(load_array(&shards, name)?),
            eps: rms_eps,
        })
    };
    let pre_fc_norm_embedding = load_norm("pre_fc_norm_embedding.weight")?;
    let pre_fc_norm_hidden = load_norm("pre_fc_norm_hidden.weight")?;
    let norm = load_norm("norm.weight")?;

    // Count MTP decoder layers by the highest `layers.{i}.` index present.
    let num_mtp_layers = all_tensor_names(&shards)
        .iter()
        .filter_map(|n| n.strip_prefix("layers."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|i| i.parse::<usize>().ok())
        .max()
        .map_or(1, |m| m + 1);

    // Load each MTP decoder layer by REUSING the Qwen3.5-MoE decoder layer.
    let dims = MtpLayerDims {
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rope_dims,
        rope_theta,
        rms_norm_eps: rms_eps,
        num_experts,
        num_experts_per_tok,
        norm_topk_prob,
        quant_group_size: q_gs,
        quant_bits: q_bits,
        quant_mode: q_mode,
    };
    let mut layers = Vec::with_capacity(num_mtp_layers);
    for i in 0..num_mtp_layers {
        layers.push(MtpLayer::load(&shards, &format!("layers.{i}"), &dims)?);
    }

    Ok((
        MtpHeadWeights {
            fc,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            norm,
            layers,
            hidden_size,
        },
        block_size,
    ))
}

// ---------------------------------------------------------------------------
// Round-loop (greedy)
// ---------------------------------------------------------------------------

use crate::decode_loop::ProbeStep;

/// MTP speculative-decoding round-loop (greedy / temp=0).
///
/// Port of `_mtp_rounds` (mlx-vlm). Mirrors [`super::dflash::dflash_generate_greedy`]
/// structurally: prefill the verifier, capture the penultimate hidden + first
/// bonus, then per round draft `block_size - 1` tokens via [`MtpDrafter::draft_n`],
/// verify all `block_size` positions in one cached forward (capturing both
/// logits and the penultimate hidden), accept the greedy prefix the verifier's
/// own argmax agrees with, emit, and roll back the verifier KV (GDN-aware) and
/// the sidecar KV on partial acceptance.
///
/// The verifier is the Qwen3.5/3.6-MoE hybrid (carries GDN linear-attention
/// state); rollback uses the [`super::dflash::DFlashRoundState`] snapshot/restore.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(clippy::too_many_lines)]
pub fn mtp_generate_greedy(
    verifier: &Architecture,
    drafter: &mut MtpDrafter,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    requested_block_total: usize,
    kv_quant_override: Option<KvQuant>,
    max_ctx_override: Option<i32>,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    use super::dflash::DFlashRoundState;
    use rmlx_kv_quant::LinearAttnCache;
    use std::time::Instant;

    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "mtp_generate_greedy: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "mtp_generate_greedy: MTP verifier must be the Qwen3.5/3.6-MoE hybrid \
             (needs GDN lin_caches)"
                .into(),
        ));
    }

    // The MTP sidecar conditions on the verifier's penultimate (pre-final-norm)
    // hidden — the residual-stream output of the LAST decoder layer. Capture it
    // via the single-layer capture id [num_hidden_layers - 1].
    let last_layer = verifier.num_hidden_layers().saturating_sub(1);
    let capture_ids = [last_layer];
    let hidden = verifier.hidden_size() as i32;

    // block_total: drafter config is the ceiling (sidecar `block_size`).
    let block_total = requested_block_total.min(drafter.block_size()).max(2);

    // Same constant the verifier resolves — a spec pair must not run two
    // different caches.
    let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
    // The verifier's limits bound the pair; an over-capacity `--max-ctx` is
    // refused here rather than overflowing a cache mid-round.
    let ctx = crate::speculative::verifier_context(verifier, max_ctx_override)?;
    let max_seq = ctx.ceiling;

    let mut v_caches: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| {
            let window = verifier.layer_sliding_window(i);
            KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                .with_max_seq_ceiling(ctx.ceiling)
                .with_layer_idx(i)
                // The verifier stack decides whether its layers read each
                // other's K/V, and so whether Mixed/RotK keep their bf16
                // mirror. A spec pair must not run two different caches.
                .with_shares_kv(verifier.shares_kv_across_layers())
        })
        .collect();
    let mut v_lin: Vec<LinearAttnCache> = (0..verifier.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect();

    drafter.reset();

    let mut total_draft = 0usize;
    let mut total_accept = 0usize;
    let mut rounds = 0usize;
    // One read of process-global log state per request, at the loop head.
    let charge_phases = super::phases_charged();
    let t_total = Instant::now();
    let mut window = DecodeWindow::new();
    let mut draft_ns: u128 = 0;
    let mut verifier_ns: u128 = 0;

    let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

    // -- Prefill verifier on prompt[..-1]; last token is the round-0 carry. --
    let prefill_slice = &prompt_ids[..prompt_ids.len() - 1];
    let prefill_t0 = Instant::now();
    super::prefill_chunked(
        verifier,
        prefill_slice,
        &mut v_caches,
        Some(&mut v_lin),
        device,
    )?;
    let prefill_ns = prefill_t0.elapsed().as_nanos();

    // The sidecar's drafting position (`_next_position`) is the verifier prefix
    // length consumed so far. After prefill + the round-0 carry forward below it
    // equals the verifier's full prefix length (prompt length).
    let mut draft_pos = prefill_slice.len() as i32;

    // -- Round-0: feed the last prompt token, capture its penultimate hidden +
    //    the first bonus token. --
    // Non-empty by the same guard that makes `prompt_ids[..len-1]` above safe.
    let last_prompt = prompt_ids[prompt_ids.len() - 1];
    let (r0_logits, r0_hidden) = verifier.forward_verify_capture(
        &[last_prompt],
        1,
        &capture_ids,
        &mut v_caches,
        Some(&mut v_lin),
        device,
    )?;
    draft_pos += 1; // verifier consumed the carry token
                    // Conditioning hidden for the first draft round = the carry position hidden.
    super::guard_verifier_prefill_logits(verifier, &r0_logits, prompt_ids.len())?;
    let mut h_cond = r0_hidden;
    let mut b = {
        let am = argmax(&r0_logits, -1, device)?;
        am.eval()?;
        u32::from_le_bytes(am.to_bytes()?[..4].try_into().unwrap())
    };
    emit_step(tokenizer, b, step_fn, &mut emitted, &mut window);
    if eos_ids.contains(&b) {
        // The stop token arrived before a round could run. The request still
        // happened, so it still leaves exactly one record.
        super::RoundStats {
            loop_kind: super::SpecLoop::MtpSidecar,
            block_size: block_total,
            rounds: 0,
            emitted: emitted.len(),
            seed_emitted: emitted.len(),
            emitted_in_rounds: 0,
            total_draft: 0,
            total_accept: 0,
            prefill_ns,
            draft_ns: 0,
            verifier_ns: 0,
            round_loop_ns: 0,
            elapsed_ns: t_total.elapsed().as_nanos(),
            decode_tps: window.tps(),
            charged: charge_phases,
        }
        .log_done();
        return Ok(emitted);
    }

    tracing::info!(
        block_size = block_total,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        capture_layer = last_layer,
        "mtp_generate_greedy: starting (Qwen3.6-MoE verifier + MTP sidecar)"
    );

    let seed_emitted = emitted.len();
    let mut emitted_in_rounds = 0usize;
    let round_loop_t0 = Instant::now();
    while emitted.len() < n_tokens {
        let round_t0 = Instant::now();
        rounds += 1;
        let remaining = n_tokens - emitted.len();
        let bs = block_total.min(remaining + 1).max(2);
        if bs <= 1 {
            break;
        }

        // -- Phase A: drafter proposes bs-1 tokens (autoregressive). The sidecar
        //    KV starts this round at `draft_pos` (verifier prefix length). --
        let draft_start = drafter.offset();
        let t0 = Instant::now();
        let draft_tokens = drafter.draft_n(verifier, b, &h_cond, bs, draft_pos)?;
        let round_draft_ns = t0.elapsed().as_nanos();
        draft_ns += round_draft_ns;
        if draft_tokens.is_empty() {
            break;
        }
        total_draft += draft_tokens.len();

        // -- Phase B: verifier scores [b, draft...] + captures penultimate hidden
        //    in one pass. Snapshot GDN state before the verify forward. --
        let round_snap = DFlashRoundState::snapshot(&v_lin)?;
        let mut v_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        v_input.push(b);
        v_input.extend_from_slice(&draft_tokens);
        let v_k = v_input.len();

        let t0 = Instant::now();
        let (v_logits, v_hidden) = verifier.forward_verify_capture(
            &v_input,
            v_k,
            &capture_ids,
            &mut v_caches,
            Some(&mut v_lin),
            device,
        )?;
        // The verify forward already projected all `v_k` positions through the
        // LM head. Read that back once, here, rather than re-deriving the head
        // one position at a time in the acceptance walk: the head is a separate
        // quantised tensor and each re-derivation is another full read of it
        // plus another pipeline drain. Reading it inside this span is also what
        // makes `verifier_ms` the cost of the verify forward rather than the
        // cost of building its graph — the assistant loop reads its verifier
        // argmax back at the same point, and the two figures are only
        // comparable because of it.
        let v_argmax = argmax(&v_logits, -1, device)?;
        v_argmax.eval()?;
        let vb = v_argmax.to_bytes()?;
        let round_verify_ns = t0.elapsed().as_nanos();
        verifier_ns += round_verify_ns;

        // -- Phase C: greedy acceptance walk over the verifier's own tokens.
        let t0 = Instant::now();
        let v_tokens = super::argmax_tokens(&vb, v_k)?;
        let (accept, new_tokens) = super::accept_prefix(&v_tokens, &draft_tokens, remaining)?;
        let round_walk_ns = t0.elapsed().as_nanos();
        total_accept += accept;

        // -- Emit accepted prefix + 1 correction/bonus. --
        let mut hit_eos = false;
        for &id in &new_tokens {
            if emitted.len() >= n_tokens {
                break;
            }
            emit_step(tokenizer, id, step_fn, &mut emitted, &mut window);
            emitted_in_rounds += 1;
            if eos_ids.contains(&id) {
                hit_eos = true;
                break;
            }
        }
        if hit_eos {
            break;
        }

        // -- Phase D: rollback verifier KV (GDN-aware) + sidecar KV. --
        // The verifier processed `v_k` positions (carry + bs-1 drafts). Committed
        // positions this round = accept (consumed drafts) + 1 carry. Drop the
        // unaccepted draft tail from the FA KV caches.
        let t0 = Instant::now();
        let n_committed = new_tokens.len();
        let v_offset_before = v_caches.iter().map(|c| c.offset()).max().unwrap_or(0);
        let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);
        if v_target < v_offset_before {
            let v_pre_round_offset = v_offset_before - v_k as i32;
            super::rollback_round_caches(
                verifier,
                &mut v_caches,
                Some(&mut v_lin),
                Some(round_snap.into_snapshots()),
                &v_input,
                v_pre_round_offset,
                v_target,
                charge_phases,
                device,
            )?;
        } else {
            drop(round_snap);
        }

        // Sidecar KV rollback: this round the head wrote `bs - 1` slots starting
        // at `draft_start` — slot `draft_start+0` holds the carry seed `b`, then
        // slots `draft_start+1..=draft_start+accept` hold the accepted drafts
        // `draft[0..accept-1]`. Keep the carry + accepted prefix = `accept + 1`
        // slots, mirroring the verifier's `pre + 1(carry) + accept` (Phase D
        // above). Keeping only `accept` would drop the last accepted draft's KV
        // every round, silently degrading draft accept-rate.
        let d_target = draft_start + accept as i32 + 1;
        drafter.truncate_to(d_target)?;
        let round_rollback_ns = t0.elapsed().as_nanos();

        // Next-round conditioning: the verifier hidden at the newly accepted
        // bonus slot (= position `accept` of the captured penultimate hidden).
        h_cond = v_hidden.slice(
            &[0, accept as i32, 0],
            &[1, accept as i32 + 1, hidden],
            &[1, 1, 1],
            device,
        )?;
        b = *new_tokens.last().unwrap_or(&b);
        draft_pos += n_committed as i32;

        let round_ns = round_t0.elapsed().as_nanos();
        tracing::debug!(
            target: super::PHASE_TARGET,
            round = rounds,
            accept,
            num_draft = draft_tokens.len(),
            n_committed,
            emitted_total = emitted.len(),
            v_offset_before,
            v_target,
            draft_pos,
            replayed = v_target < v_offset_before,
            charged = charge_phases,
            round_ms = super::ms(round_ns),
            draft_ms = super::ms(round_draft_ns),
            verify_ms = super::ms(round_verify_ns),
            walk_ms = super::ms(round_walk_ns),
            rollback_ms = super::ms(round_rollback_ns),
            other_ms = super::ms(round_ns)
                - super::ms(round_draft_ns)
                - super::ms(round_verify_ns)
                - super::ms(round_walk_ns)
                - super::ms(round_rollback_ns),
            "mtp round"
        );
    }

    let round_loop_ns = round_loop_t0.elapsed().as_nanos();
    super::RoundStats {
        loop_kind: super::SpecLoop::MtpSidecar,
        block_size: block_total,
        rounds,
        emitted: emitted.len(),
        seed_emitted,
        emitted_in_rounds,
        total_draft,
        total_accept,
        prefill_ns,
        draft_ns,
        verifier_ns,
        round_loop_ns,
        elapsed_ns: t_total.elapsed().as_nanos(),
        decode_tps: window.tps(),
        charged: charge_phases,
    }
    .log_done();

    // Report the verifier's resident KV, so a caller that sampled the verifier
    // arch around this call can attribute the figure to it. This round loop
    // never goes through `Architecture::generate_greedy`, so nothing else
    // writes it.
    verifier.store_kv_cache_bytes(
        crate::speculative::verifier_kv_bytes(&v_caches, Some(&v_lin)),
        crate::decode_loop::PostDecode::seal(),
    );
    Ok(emitted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
