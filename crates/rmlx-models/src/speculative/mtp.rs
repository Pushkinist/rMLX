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
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, concatenate, Array, Device};

use super::MAX_BLOCK_SIZE;
use crate::arch::Architecture;
use crate::layers::{Linear, RmsNorm};
use crate::qwen3_5_moe::{MtpLayer, MtpLayerDims};
use crate::speculative::round_loop::{
    run_rounds, CacheSpan, Conditioning, Prefilled, ReportSkippedBy, RoundCfg, RoundCtx,
    RoundDrafter, RoundOutcome, Verdict, VerifierOffsetBasis,
};
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
    /// The block the sidecar's config declares (`block_size`) — the depth the
    /// head was trained at, not a ceiling on the depth it can be run at.
    /// `None` when the config names none.
    block_size: Option<usize>,
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
            ?block_size,
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

    /// The block this sidecar's config declares, including the seed carry, or
    /// `None` when it declares none.
    ///
    /// The trained depth, which a round is free to exceed: `block_from_request`
    /// in this module takes it and does not narrow to it.
    pub fn block_size(&self) -> Option<usize> {
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
/// `(weights, block_size)`, the second `None` when the config names no
/// `block_size`.
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
fn load_mtp_head(draft_dir: &Path, hidden_size: usize) -> Result<(MtpHeadWeights, Option<usize>)> {
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

    // block_size (tokens proposed per round, incl. carry). Absent is its own
    // answer: a caller deciding a default has to be able to tell a checkpoint
    // that declares a depth from one that says nothing, and a fallback here
    // would hand it 3 either way.
    let block_size = cfg
        .extras
        .get("block_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok());

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
use std::time::Instant;

/// The block a request runs at: what it asked for, bounded by what one verify
/// forward can score, and never below the two positions a seed and one draft
/// need.
///
/// **`declared` is not a ceiling here**, which is the whole reason it is an
/// argument. The head chains on its own output hidden — [`MtpDrafter::draft_n`]
/// feeds each step's `h_next` back as the next step's `h_prev` — so proposing
/// past the depth the checkpoint names is structurally admissible; what decays
/// with depth is the acceptance rate, and that is the request's trade to make.
/// Taking it and not clamping to it is what makes this function, rather than its
/// caller, the one place that decision lives.
fn block_from_request(requested: usize, declared: Option<usize>) -> usize {
    let block = requested.clamp(2, MAX_BLOCK_SIZE);
    if declared.is_some_and(|d| block > d) {
        tracing::debug!(
            block,
            declared,
            "mtp_generate: running a block deeper than the sidecar declares; the head \
             chains on its own hidden past its trained depth and acceptance falls with \
             it"
        );
    }
    block
}

/// One request's MTP-sidecar drafting state.
///
/// The sidecar keeps its own KV cache and its own drafting position, so this
/// carries both across the rounds of one request, beside the single verifier
/// row the next round conditions on.
#[allow(missing_debug_implementations)]
pub(crate) struct SidecarRound<'a> {
    drafter: &'a mut MtpDrafter,
    /// The verifier's last decoder layer, whose pre-final-norm output the
    /// sidecar conditions on.
    capture_ids: [usize; 1],
    /// The verifier's model width, for the conditioning slice.
    hidden: i32,
    /// The verifier row the next round drafts from.
    h_cond: Option<Array>,
    /// The verifier prefix the sidecar has consumed (`_next_position`).
    draft_pos: i32,
    /// Where the sidecar's cache stood when this round began.
    draft_start: i32,
    /// The verifier's capture at every position this round verified.
    scored: Option<Array>,
}

impl<'a> SidecarRound<'a> {
    fn new(drafter: &'a mut MtpDrafter, verifier: &Architecture) -> Self {
        Self {
            drafter,
            capture_ids: [verifier.num_hidden_layers().saturating_sub(1)],
            hidden: verifier.hidden_size() as i32,
            h_cond: None,
            draft_pos: 0,
            draft_start: 0,
            scored: None,
        }
    }
}

/// A round that reached for the conditioning row before one was built.
fn missing_conditioning() -> Error {
    Error::Model(
        "mtp_generate: a round read the hidden it conditions on before the prefill \
         built one"
            .into(),
    )
}

impl RoundDrafter for SidecarRound<'_> {
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy = ReportSkippedBy::TheSeedExit;
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis = VerifierOffsetBasis::AfterTheForward;

    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled> {
        let device = ctx.device;
        let Some((&last_prompt, head)) = prompt.split_last() else {
            return Err(Error::Model(
                "mtp_generate: an empty prompt reached the round loop".into(),
            ));
        };
        self.drafter.reset();
        let prefill_t0 = Instant::now();
        super::prefill_chunked(
            ctx.verifier,
            head,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        let prefill_ns = prefill_t0.elapsed().as_nanos();

        // The sidecar's drafting position is the verifier prefix consumed so
        // far. After the prompt less its last token, plus the carry forward
        // below, it is the whole prompt.
        self.draft_pos = head.len() as i32;

        // Round-0: feed the last prompt token, capture its penultimate hidden
        // and read the first bonus token off the same forward.
        let (r0_logits, r0_hidden) = ctx.verifier.forward_verify_capture(
            &[last_prompt],
            1,
            &self.capture_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        self.draft_pos += 1;
        super::guard_verifier_prefill_logits(ctx.verifier, &r0_logits, prompt.len())?;
        self.h_cond = Some(r0_hidden);
        let seed = ctx.draw.seed_token(&r0_logits, device)?;
        Ok(Prefilled {
            seed,
            prefill_ns,
            // The sidecar slices one verifier row per round and projects
            // nothing, so it accumulates no conditioning to report.
            conditioned_rows: None,
        })
    }

    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>> {
        // The sidecar KV starts this round here, and its rollback counts
        // forward from it over the accepted prefix.
        self.draft_start = self.drafter.offset();
        let h_cond = self.h_cond.as_ref().ok_or_else(missing_conditioning)?;
        self.drafter
            .draft_n(ctx.verifier, carry, h_cond, block, self.draft_pos)
    }

    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict> {
        let device = ctx.device;
        // Arm the GDN round tape before the forward: the refold replays the
        // accepted prefix off it.
        super::arm_lin_tapes(ctx.lin.as_deref_mut());

        let t0 = Instant::now();
        let (v_logits, v_hidden) = ctx.verifier.forward_verify_capture(
            fed,
            fed.len(),
            &self.capture_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        // The forward already projected all of `fed` through the LM head. Read
        // that back inside this span rather than re-deriving the head one
        // position at a time in the walk: the head is a separate quantised
        // tensor and each re-derivation is another full read of it plus another
        // pipeline drain. A sampled request draws here too, so the per-position
        // host softmax lands here and not in the walk.
        let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;
        let verify_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let proposals = fed.get(1..).unwrap_or_default();
        let (accept, commit) = super::accept_prefix(&v_tokens, proposals, remaining)?;
        let walk_ns = t0.elapsed().as_nanos();

        self.scored = Some(v_hidden);
        Ok(Verdict {
            accept,
            commit,
            verify_ns,
            walk_ns,
        })
    }

    fn rollback(
        &mut self,
        _ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        _outcome: RoundOutcome,
    ) -> Result<Option<CacheSpan>> {
        // This round the head wrote `block - 1` slots from `draft_start`: slot
        // `draft_start` holds the carry, then `draft_start+1..=+accept` hold the
        // accepted drafts. Keep the carry plus the accepted prefix, mirroring
        // the verifier's `pre + 1 + accept`. Keeping only `accept` would drop
        // the last accepted draft's K/V every round, silently degrading the
        // accept rate.
        let target = self.draft_start + verdict.accept as i32 + 1;
        self.drafter.truncate_to(target)?;
        Ok(Some(CacheSpan {
            before: self.draft_start,
            target,
        }))
    }

    fn condition(
        &mut self,
        ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>> {
        let device = ctx.device;
        let Some(scored) = self.scored.take() else {
            return Err(Error::Model(
                "mtp_generate: a round conditioned on a verify forward that did not run".into(),
            ));
        };
        // The next round conditions on the verifier hidden at the newly
        // accepted bonus slot.
        let accept = verdict.accept as i32;
        let h_cond = scored.slice(
            &[0, accept, 0],
            &[1, accept + 1, self.hidden],
            &[1, 1, 1],
            device,
        )?;
        if ctx.charged {
            // Reading the verifier's tokens forced the logits and the trunk
            // under them, but the capture hangs off a different output of that
            // forward and this slice off the capture. The next round's drafter
            // is the first thing to read either, so with nothing forcing them
            // here the verifier's capture is billed to the drafter. See
            // `phases_charged`.
            h_cond.eval()?;
        }
        let rows = h_cond.shape().get(1).copied();
        self.h_cond = Some(h_cond);
        self.draft_pos += outcome.committed as i32;
        Ok(Some(Conditioning {
            rows,
            projected: None,
        }))
    }

    fn carry(&self, f: &mut dyn FnMut(&[(&str, &Array)])) -> Result<()> {
        f(&[(
            "h_cond",
            self.h_cond.as_ref().ok_or_else(missing_conditioning)?,
        )]);
        Ok(())
    }
}

/// MTP speculative-decoding round-loop entry.
///
/// Port of `_mtp_rounds` (mlx-vlm): prefill the verifier, capture the
/// penultimate hidden and the first bonus, then per round draft
/// `block_size - 1` tokens via [`MtpDrafter::draft_n`], verify all `block_size`
/// positions in one cached forward capturing both logits and the penultimate
/// hidden, accept the prefix the verifier's own token agrees with, emit, and
/// roll back the verifier KV (GDN-aware) and the sidecar KV on partial
/// acceptance. The rounds themselves run in [`run_rounds`]; what is here is what
/// refuses before a cache stack is built, and the request's block.
///
/// `sampler_cfg` decides what "the verifier's own token" means at each position
/// — its argmax at temperature 0, a draw from its post-sampling distribution
/// above it. The drafter is unaffected either way: it proposes its argmax, and
/// [`super::VerifierDraw`] explains why the walk is still exact.
///
/// The verifier is the Qwen3.5/3.6-MoE hybrid (carries GDN linear-attention
/// state); rollback refolds that state from the round tape.
///
/// Returns the emitted steps and **the widest block any round of this run
/// actually ran**. Not the block resolved before the loop: a caller checking
/// what it asked for against that would be trusting the very step it wanted
/// checked, and every round narrows the block again against the remaining token
/// budget.
///
/// # Errors
/// [`Error::Model`] for a prompt under two tokens or a verifier that carries no
/// recurrent state, and whatever the round loop refuses.
#[allow(clippy::too_many_arguments)]
pub fn mtp_generate(
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
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<(Vec<ProbeStep>, usize)> {
    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "mtp_generate: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "mtp_generate: MTP verifier must be the Qwen3.5/3.6-MoE hybrid \
             (needs GDN lin_caches)"
                .into(),
        ));
    }
    let block_total = block_from_request(requested_block_total, drafter.block_size());
    // One read of process-global log state per request, at the entry.
    let charge_phases = super::phases_charged();
    let mut round = SidecarRound::new(drafter, verifier);
    run_rounds(
        verifier,
        &mut round,
        prompt_ids,
        step_fn,
        &RoundCfg {
            loop_kind: super::SpecLoop::MtpSidecar,
            block_size: block_total,
            n_tokens,
            eos_ids,
            tokenizer,
            sampler_cfg,
            charged: charge_phases,
            kv_quant_override,
            max_ctx_override,
        },
        device,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
