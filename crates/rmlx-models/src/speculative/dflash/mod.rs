//! DFlash drafter loader + round-loop.
//!
//! Port of mlx-vlm `mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py`
//! (`DFlashDraftModel`) and the round-loop in `mlx_vlm/speculative/dflash.py`
//! (`_dflash_next_block_size`, `_dflash_rounds`).
//!
//! # What a DFlash drafter is
//!
//! DFlash (z-lab "Block Diffusion for Flash Speculative Decoding") drafts a
//! whole **block** of `block_size` tokens in one non-autoregressive pass: the
//! seed (bonus) token followed by `block_size - 1` `mask_token_id` positions.
//! A small standalone transformer (here 8 full-attention Qwen3-style layers,
//! hidden 2048) denoises the masked block in parallel, conditioned on the
//! verifier's **concatenated multi-layer hidden states** (`target_layer_ids`,
//! projected `len(ids)*H -> H` through the drafter's `fc`).
//!
//! Three properties make DFlash distinct from the MTP / Gemma4-assistant
//! drafters:
//!
//! 1. **Multi-layer conditioning.** It reads the verifier hidden state at

//! 2. **Adaptive block size.** [`dflash_next_block_size`] grows/shrinks the

//! 3. **GDN-aware rollback.** The Qwen3.6-MoE verifier has linear-attention

//!
//! # Status — document-the-truth (CLAUDE.md hard rule 7)
//!
//! **Fully wired + live-validated against `z-lab/Qwen3.6-35B-A3B-DFlash` +
//! `mlx-community/Qwen3.6-35B-A3B-8bit` verifier.** The loader
//! ([`DFlashDrafter::load`] + [`load_dflash`]) — `fc` (`5H->H`), `hidden_norm`,
//! `norm`, all 8 `DFlashDecoderLayer`s, **YARN RoPE** ([`crate::rope::compute_yarn_freqs`]),
//! the drafter forward [`DFlashDrafter::draft_block`], the block-size schedule
//! [`dflash_next_block_size`], the acceptance walk [`walk_block_greedy`], the
//! [`DFlashRoundState`] GDN rollback, and the full round-loop
//! [`dflash_generate_greedy`]. The three verifier-side seams are wired on
//! [`crate::arch::Architecture`] for the Qwen3.6-MoE verifier:
//!
//! 1. **Multi-layer hidden capture** — [`Architecture::forward_verify_capture`]

//! 2. **GDN rollback** — reuses the `LinearAttnCache` snapshot/restore +

//! 3. **Raw embed accessor** — [`Architecture::embed_tokens_raw`] (the Qwen3.5

//!
//! Numeric alignment: round-0 first-proposal + full-run accept-rate match the
//! mlx-vlm `_dflash_rounds` reference exactly (0.515 on the test prompt). Two
//! findings closed the initial ~0%/14% accept gap: (a) the drafter needs
//! **YARN RoPE** (plain RoPE diverges materially even at small offsets); (b) the
//! drafter conditions on the **accumulated** committed-context hidden across
//! rounds (the Python ref's persistent draft KV cache), not just the current
//! round's committed slice.

#![allow(
    clippy::cognitive_complexity,
    clippy::implicit_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    clippy::used_underscore_binding
)]
// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{
    add, argmax, concatenate, divide, multiply, rope, scalar_f32, scaled_dot_product_attention,
    tanh, Array, Device,
};

use super::{emit_step, DecodeWindow};
use crate::arch::Architecture;
use crate::layers::{Activation, Linear, Mlp, RmsNorm};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

/// Choose the next DFlash verify block size from recent acceptance.
///
/// Pure port of `_dflash_next_block_size`. The trained `block_size` (advertised
/// in the drafter config, usually 16) is the ceiling; back off quickly when
/// deeper positions are mostly rejected, grow back when acceptance is strong.
///
/// - `recent` is the recent `(accepted, drafted)` history (most-recent last);
///   the round-loop keeps the last 8.
/// - `requested_block_total` is the configured/CLI block size.
/// - `remaining_budget` caps the block to the remaining token budget.
/// - `prefer_requested` short-circuits to the requested size (config flag).
///
/// Returns the next block total (including the seed token).
pub fn dflash_next_block_size(
    recent: &[(usize, usize)],
    requested_block_total: usize,
    remaining_budget: usize,
    prefer_requested: bool,
) -> usize {
    let block_total = requested_block_total.min(remaining_budget);
    if block_total <= 1 {
        return block_total;
    }
    if prefer_requested {
        return block_total;
    }

    // Keep only the last 8 rounds that actually drafted.
    let recent: Vec<(usize, usize)> = recent
        .iter()
        .rev()
        .take(8)
        .rev()
        .copied()
        .filter(|&(_, d)| d > 0)
        .collect();
    if recent.is_empty() {
        return block_total;
    }

    let last_drafted = recent.last().map_or(0, |&(_, d)| d);
    let current = block_total.min((last_drafted + 1).max(2));
    let min_total = block_total.min(4);
    let drafted: usize = recent.iter().map(|&(_, d)| d).sum();
    let accepted: usize = recent.iter().map(|&(a, _)| a).sum();
    let accept_rate = accepted as f64 / drafted as f64;
    let mean_accept = accepted as f64 / recent.len() as f64;

    if accept_rate < 0.30 || mean_accept < 2.0 {
        if current >= 8 {
            return min_total.max(block_total.min(current / 2));
        }
        return min_total.max(block_total.min(current.saturating_sub(2)));
    }

    if accept_rate < 0.50 {
        return min_total.max(block_total.min(current.saturating_sub(2)));
    }

    let full_hits = recent.iter().filter(|&&(a, d)| a >= d).count();
    let full_hit_rate = full_hits as f64 / recent.len() as f64;
    if accept_rate >= 0.85 && full_hit_rate >= 0.75 {
        return block_total.min(current + 2);
    }

    block_total.min(current)
}

/// One DFlash decoder layer (Qwen3 shape: GQA + per-head q/k RMSNorm + RoPE).
///
/// Mirrors `DFlashDecoderLayer`: pre-norm self-attention with a context /
/// proposal KV split (only context K/V go in the cache) followed by a SwiGLU
/// MLP, both residual.
#[allow(missing_debug_implementations)]
struct DFlashLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    mlp: Mlp,
}

/// Loaded DFlash drafter weights + config.
///
/// Construct with [`DFlashDrafter::load`]. The drafter is a self-contained
/// transformer; `embed_tokens` and `lm_head` are the verifier's (the round-loop
/// holds the verifier `Architecture` and threads them in), mirroring the Python
/// `bind()`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed drafter struct — private weight fields; public API is draft_step() and update_context(); adding a field requires updating load_dflash and DFlashDrafter::load"
)]
#[allow(missing_debug_implementations)]
pub struct DFlashDrafter {
    /// `fc`: Linear `len(target_layer_ids)*H -> H`, no bias.
    fc: Linear,
    /// RMSNorm applied to the projected conditioning hidden.
    hidden_norm: RmsNorm,
    /// Final RMSNorm after the decoder stack.
    norm: RmsNorm,
    layers: Vec<DFlashLayer>,
    /// Per-layer KV cache (the drafter's own; holds context K/V only).
    caches: Vec<KvCache>,
    /// Precomputed YARN inverse-frequency table (`[head_dim/2]`), when the
    /// drafter config specifies `rope_scaling: {rope_type: yarn}`. `None`
    /// falls back to plain RoPE (`rope(theta)`). The Qwen3.6-35B DFlash drafter
    /// uses YARN (factor 64) — applying plain RoPE corrupts its attention and
    /// collapses accept-rate to ~0 (verified numeric divergence, ).
    rope_freqs: Option<Array>,
    /// YARN attention-magnitude scale (`mscale`) applied to q/k before RoPE.
    /// `1.0` for plain RoPE.
    rope_mscale: f32,
    cfg: DFlashConfig,
    device: Device,
}

/// DFlash drafter config (subset rMLX needs), parsed from `config.json`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete DFlash drafter contract; adding a field requires updating parse_dflash_config and DFlashDrafter::load"
)]
#[derive(Debug, Clone)]
/// Parsed DFlash speculative-drafter config (see docs/SPECULATIVE.md).
pub struct DFlashConfig {
    /// Drafter hidden dimension (must equal verifier hidden size).
    pub hidden_size: usize,
    /// Number of decoder layers in the drafter.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Maximum draft block size (tokens per step).
    pub block_size: usize,
    /// Token id used as the mask/pad token during draft generation.
    pub mask_token_id: u32,
    /// Verifier layer indices whose residuals the drafter conditions on.
    pub target_layer_ids: Vec<usize>,
    /// Optional logit soft-capping (matches the verifier).
    pub final_logit_softcapping: Option<f32>,
}

impl DFlashDrafter {
    /// Load a DFlash drafter from `draft_dir` and validate it against the
    /// verifier's hidden size.
    ///
    /// `draft_dir` is the standalone drafter folder (`config.json` with
    /// `architectures: ["DFlashDraftModel"]` + a `dflash_config` block, and a
    /// `model.safetensors`). `hidden_size` is the verifier's model width — the
    /// drafter `hidden_size` and `fc` input (`len(target_layer_ids)*H`) must
    /// match it.
    pub fn load(draft_dir: &Path, hidden_size: usize, device: Device) -> Result<Self> {
        let mut me = load_dflash(draft_dir, hidden_size, device)?;
        me.caches = (0..me.cfg.num_hidden_layers)
            .map(|_| KvCache::with_quant(KvQuant::None))
            .collect();
        tracing::info!(
            draft = %draft_dir.display(),
            hidden_size,
            num_layers = me.cfg.num_hidden_layers,
            block_size = me.cfg.block_size,
            target_layer_ids = ?me.cfg.target_layer_ids,
            "DFlashDrafter: loaded drafter"
        );
        Ok(me)
    }

    /// Reset the drafter's KV cache between generations.
    pub fn reset(&mut self) {
        for c in &mut self.caches {
            *c = KvCache::with_quant(KvQuant::None);
        }
    }

    /// Trained / configured block size (the adaptive-schedule ceiling).
    pub fn block_size(&self) -> usize {
        self.cfg.block_size
    }

    /// `target_layer_ids` the drafter conditions on.
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.cfg.target_layer_ids
    }

    /// `mask_token_id` used to build the masked draft block.
    pub fn mask_token_id(&self) -> u32 {
        self.cfg.mask_token_id
    }

    /// Hidden size the drafter was loaded for.
    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Project the verifier's concatenated multi-layer hidden states into the
    /// drafter's conditioning width: `hidden_norm(fc(concat_hidden))`.
    ///
    /// `concat_hidden` is `[1, n, len(target_layer_ids)*H]` (the verifier hidden
    /// at each `target_layer_id`, concatenated along the feature axis). Mirrors
    /// `DFlashDraftModel._hidden`'s `h_ctx = hidden_norm(fc(target_hidden))`.
    /// Real: `fc` + `hidden_norm` are loaded and run on-device.
    pub fn project_condition(&self, concat_hidden: &Array) -> Result<Array> {
        let projected = self.fc.forward(concat_hidden, self.device)?;
        self.hidden_norm.forward(&projected, self.device)
    }

    /// Build the conditioning hidden by capturing the verifier hidden states at
    /// `target_layer_ids` and concatenating them.
    ///
    /// **Awaiting verifier-side wiring (module docs).** rMLX's
    /// `Architecture::forward_hidden_states` returns only the penultimate trunk
    /// hidden and is Gemma4-only. DFlash needs per-`target_layer_id`
    /// captures concatenated, which the Qwen3.6-MoE verifier path does not yet
    /// expose. Returns [`Error::Model`] until that lands; the `fc`/`hidden_norm`
    /// projection it would feed ([`project_condition`]) is implemented + tested.
    /// Capture the verifier's concatenated multi-layer hidden over `input_ids`
    /// (advancing the supplied caches) WITHOUT projecting. Returns
    /// `[1, k, len(target_layer_ids)*H]` for the last `k` positions. The
    /// round-loop normally uses [`Self::project_condition`] over the hidden
    /// returned by [`Architecture::forward_verify_capture`] (one combined
    /// forward); this helper exists for the standalone capture-then-project
    /// path (round-0 / tests).
    pub fn condition_from_verifier(
        &self,
        verifier: &Architecture,
        input_ids: &[u32],
        k: usize,
        kv_caches: &mut [KvCache],
        lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        let concat = verifier.forward_hidden_states_multi(
            input_ids,
            k,
            &self.cfg.target_layer_ids,
            kv_caches,
            lin_caches,
            device,
        )?;
        self.project_condition(&concat)
    }

    /// Draft a block of `block_size - 1` tokens in one non-autoregressive pass.
    ///
    /// Mirrors `DFlashDraftModel.draft_block`: build the masked block
    /// `[seed, mask, mask, ...]`, run the decoder stack conditioned on `h_ctx`
    /// (the projected verifier hidden), pick greedy tokens at positions `1..`.
    ///
    /// `seed_tok` is the last accepted (bonus) token, `h_ctx` is the output of
    /// [`project_condition`] (`[1, block_size, H]`, one conditioning row per
    /// block position). Greedy (temp=0) here; the stochastic variant reuses the
    /// host sampler at the round-loop level (mirrors ).
    ///
    /// Real: the embed-block construction, all 8 decoder layers (context/proposal
    /// split attention, q/k norm, RoPE, MLP) and the final `norm` are loaded and
    /// run on-device. The seam is the *verifier* `embed_tokens` + `lm_head`
    /// accessor (threaded by the round-loop) — gated in [`embed_block`] below.
    pub fn draft_block(
        &mut self,
        verifier: &Architecture,
        seed_tok: u32,
        h_ctx: &Array,
        block_size: usize,
    ) -> Result<Vec<u32>> {
        if block_size <= 1 {
            return Ok(vec![]);
        }
        // Build the masked input block: [seed, mask, mask, ...] of length block_size.
        let mut block_ids: Vec<i32> = Vec::with_capacity(block_size);
        block_ids.push(seed_tok as i32);
        for _ in 1..block_size {
            block_ids.push(self.cfg.mask_token_id as i32);
        }
        // Embed via the verifier's input embeddings (seam — see below).
        let h = self.embed_block(verifier, &block_ids)?;
        // Run the decoder stack conditioned on the projected verifier hidden.
        let hidden = self.forward_block(&h, h_ctx)?;
        // Pick greedy tokens at positions 1.. (the denoised mask positions).
        self.greedy_block_tokens(verifier, &hidden, block_size)
    }

    /// Embed the masked block through the verifier's input embeddings.
    ///
    /// Awaiting-verifier-wiring seam: the per-arch raw input-embedding accessor
    /// is not yet exposed on `Architecture` (only the full forward path is). The
    /// DFlash drafter, like MTP, needs the bare embedding of the block tokens.
    /// Returns `[1, block_size, H]`.
    fn embed_block(&self, verifier: &Architecture, block_ids: &[i32]) -> Result<Array> {
        // Qwen3.5 verifier embed_tokens is a bare nn.Embedding (embed_scale=1.0)
        // — the DFlash drafter consumes the unscaled embedding.
        verifier.embed_tokens_raw(block_ids, self.device)
    }

    /// Run the DFlash decoder stack over the masked block.
    ///
    /// Real port of `DFlashDraftModel._hidden` (sans embedding): for each layer,
    /// pre-norm self-attention with a context (`h_ctx`) / proposal (`h`) KV split
    /// followed by a SwiGLU MLP, then the final `norm`. Returns `[1, L, H]`.
    fn forward_block(&mut self, h: &Array, h_ctx: &Array) -> Result<Array> {
        let mut x = h.try_clone()?;
        // The conditioning context is shared across layers (Python passes the
        // same `h_ctx` into every layer's attention as the KV-source prefix).
        let layers = self.layers.len();
        for li in 0..layers {
            x = self.layer_forward(li, &x, h_ctx)?;
        }
        self.norm.forward(&x, self.device)
    }

    /// Apply RoPE to a `[1, n_heads, L, hd]` tensor at `offset`, using the
    /// precomputed YARN frequency table + mscale when present (drafter config
    /// `rope_scaling: yarn`), else plain RoPE at `rope_theta`.
    ///
    /// YARN: scale the rotated dims by `mscale` then rotate with the YARN
    /// inverse-freq table (`rope_with_freqs`, `scale=1.0`) — mirrors mlx-lm
    /// `YarnRoPE.__call__` (the drafter was trained with this; plain RoPE
    /// diverges materially even at small offsets — numeric finding).
    fn apply_rope(&self, x: &Array, hd: i32, offset: i32, device: Device) -> Result<Array> {
        match &self.rope_freqs {
            Some(freqs) => {
                let scaled = if (self.rope_mscale - 1.0).abs() > 1e-6 {
                    multiply(
                        x,
                        &scalar_f32(self.rope_mscale).astype(x.dtype(), device)?,
                        device,
                    )?
                } else {
                    x.try_clone()?
                };
                rmlx_mlx::rope_with_freqs(&scaled, hd, false, 1.0, offset, freqs, device)
            }
            None => rope(x, hd, false, self.cfg.rope_theta, 1.0, offset, device),
        }
    }

    /// One DFlash decoder-layer forward (context/proposal split SDPA + MLP).
    ///
    /// Mirrors `DFlashDecoderLayer.__call__` + `DFlashAttention.__call__`:
    /// queries come from the proposal `x`; keys/values from BOTH the context
    /// prefix `h_ctx` (RoPE at offset 0) and the proposal `x` (RoPE at offset S).
    /// Draft-block self-attention is intentionally non-causal (`mask = None`) —
    /// DFlash denoises the whole proposed block at once.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn layer_forward(&self, li: usize, x: &Array, h_ctx: &Array) -> Result<Array> {
        let layer = &self.layers[li];
        let device = self.device;
        let n_heads = self.cfg.num_attention_heads as i32;
        let n_kv = self.cfg.num_key_value_heads as i32;
        let hd = self.cfg.head_dim as i32;
        let l = x.shape()[1]; // proposal length L
        let s = h_ctx.shape()[1]; // context length S

        let residual = x.try_clone()?;
        let h = layer.input_layernorm.forward(x, device)?;

        // Project queries (proposal) and K/V for both context and proposal.
        let q = layer.q_proj.forward(&h, device)?;
        let ctx_k = layer.k_proj.forward(h_ctx, device)?;
        let ctx_v = layer.v_proj.forward(h_ctx, device)?;
        let prop_k = layer.k_proj.forward(&h, device)?;
        let prop_v = layer.v_proj.forward(&h, device)?;

        // q: [1,L,n_heads,hd] -> q_norm -> [1,n_heads,L,hd]
        let q = q.reshape(&[1, l, n_heads, hd], device)?;
        let q = layer.q_norm.forward(&q, device)?;
        let q = q.transpose(&[0, 2, 1, 3], device)?;
        let q = self.apply_rope(&q, hd, s, device)?;

        // ctx K: [1,S,n_kv,hd] -> k_norm -> [1,n_kv,S,hd], RoPE at offset 0
        let ctx_k = ctx_k.reshape(&[1, s, n_kv, hd], device)?;
        let ctx_k = layer.k_norm.forward(&ctx_k, device)?;
        let ctx_k = ctx_k.transpose(&[0, 2, 1, 3], device)?;
        let ctx_k = self.apply_rope(&ctx_k, hd, 0, device)?;
        let ctx_v = ctx_v
            .reshape(&[1, s, n_kv, hd], device)?
            .transpose(&[0, 2, 1, 3], device)?;

        // proposal K: [1,L,n_kv,hd] -> k_norm -> [1,n_kv,L,hd], RoPE at offset S
        let prop_k = prop_k.reshape(&[1, l, n_kv, hd], device)?;
        let prop_k = layer.k_norm.forward(&prop_k, device)?;
        let prop_k = prop_k.transpose(&[0, 2, 1, 3], device)?;
        let prop_k = self.apply_rope(&prop_k, hd, s, device)?;
        let prop_v = prop_v
            .reshape(&[1, l, n_kv, hd], device)?
            .transpose(&[0, 2, 1, 3], device)?;

        // Concat context + proposal K/V along the sequence axis.
        let keys = concatenate(&[&ctx_k, &prop_k], 2, device)?;
        let values = concatenate(&[&ctx_v, &prop_v], 2, device)?;

        let scale = (hd as f32).powf(-0.5);
        // Non-causal block attention: no mask (Python `mask = None`).
        let attn = scaled_dot_product_attention(&q, &keys, &values, scale, "", None, device)?;
        let attn = attn.transpose(&[0, 2, 1, 3], device)?;
        let attn = attn.reshape(&[1, l, n_heads * hd], device)?;
        let attn = layer.o_proj.forward(&attn, device)?;
        let h = add(&residual, &attn, device)?;

        let residual = h.try_clone()?;
        let f = layer.post_attention_layernorm.forward(&h, device)?;
        let f = layer.mlp.forward(&f, device)?;
        add(&residual, &f, device)
    }

    /// Greedy token ids for the denoised block positions `1..block_size`.
    ///
    /// Re-uses the verifier LM head (`logits_from_hidden`) and applies the
    /// drafter's `final_logit_softcapping` if configured (None for the Qwen3.6
    /// drafter). Returns `block_size - 1` token ids.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn greedy_block_tokens(
        &self,
        verifier: &Architecture,
        hidden: &Array,
        block_size: usize,
    ) -> Result<Vec<u32>> {
        let device = self.device;
        let h = self.cfg.hidden_size as i32;
        let mut tokens = Vec::with_capacity(block_size - 1);
        for pos in 1..block_size {
            let row = hidden.slice(
                &[0, pos as i32, 0],
                &[1, pos as i32 + 1, h],
                &[1, 1, 1],
                device,
            )?;
            // `forward_block` ends with the drafter's own final norm.
            let mut logits = verifier.logits_from_final_hidden(&row, device)?;
            if let Some(cap) = self.cfg.final_logit_softcapping {
                let cap_arr = scalar_f32(cap).astype(logits.dtype(), device)?;
                let scaled = divide(&logits, &cap_arr, device)?;
                let t = tanh(&scaled, device)?;
                logits = multiply(&t, &cap_arr, device)?;
            }
            let am = argmax(&logits, -1, device)?;
            am.eval()?;
            let id = u32::from_le_bytes(am.to_bytes()?[..4].try_into().unwrap());
            tokens.push(id);
        }
        Ok(tokens)
    }
}

/// Per-generation GDN rollback bookkeeping for the DFlash round loop.
///
/// Wraps the verifier's `LinearAttnCache` snapshot/restore round-trip: take a
/// snapshot of every GDN cache before a draft round, restore them on partial
/// acceptance. This is the GDN-aware analogue of the Gemma4 spec path's
/// `KvCache::truncate_to` rollback — GDN recurrent state has no sequence axis,
/// so it cannot be truncated (see `linear_attn.rs`).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed rollback-state struct — private snapshots field; public API is snapshot() and restore_if_partial(); adding a field requires updating snapshot() and restore_if_partial()"
)]
#[allow(missing_debug_implementations)]
pub struct DFlashRoundState {
    /// Snapshot of each GDN cache taken at round start (index-aligned with the
    /// verifier's linear-attention caches).
    snapshots: Vec<LinearAttnCache>,
}

impl DFlashRoundState {
    /// Snapshot all GDN caches before a draft round.
    pub fn snapshot(lin_caches: &[LinearAttnCache]) -> Result<Self> {
        let snapshots = lin_caches
            .iter()
            .map(|c| c.snapshot())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { snapshots })
    }

    /// Number of GDN caches snapshotted.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// True when no GDN cache was snapshotted (non-GDN verifier).
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Consume the round state and hand back the per-layer snapshots.
    ///
    /// Restoring them is only one third of a partial-accept rollback — it also
    /// has to roll the KV caches back and replay the retained prefix through
    /// them — so the snapshots are handed to the shared rollback rather than
    /// restored here (mirrors `lm.rollback_speculative_cache` followed by the
    /// next round's verify in `_dflash_rounds`).
    pub fn into_snapshots(self) -> Vec<LinearAttnCache> {
        self.snapshots
    }
}

/// One greedy DFlash acceptance walk over a drafted block (port of the
/// `_speculative_walk` half of `_dflash_rounds`).
///
/// Accept drafted tokens up to the first mismatch with the verifier's greedy
/// choice, then take the verifier's correction/bonus at that position. Returns
/// `(accepted, new_tokens)` capped at `budget`. `target_tokens` are the
/// verifier's greedy predictions for positions `[seed, d0, d1, ...]` — i.e.
/// `draft_tokens.len() + 1` of them.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn walk_block_greedy(
    draft_tokens: &[u32],
    target_tokens: &[u32],
    budget: usize,
) -> (usize, Vec<u32>) {
    let n_draft = draft_tokens.len();
    let mut accepted = n_draft;
    for (i, (&d, &t)) in draft_tokens.iter().zip(target_tokens.iter()).enumerate() {
        if d != t {
            accepted = i;
            break;
        }
    }
    let mut new_tokens: Vec<u32> = draft_tokens[..accepted].to_vec();
    if accepted < target_tokens.len() {
        new_tokens.push(target_tokens[accepted]);
    }
    new_tokens.truncate(budget);
    (accepted, new_tokens)
}

use crate::decode_loop::ProbeStep;

/// DFlash speculative-decoding round-loop (greedy / temp=0).
///
/// Port of `_dflash_rounds` (mlx-vlm). Wires the three verifier-side
/// seams against the Qwen3.6-MoE verifier:
///
/// 1. **Multi-layer hidden capture** — each verify forward.
/// 2. **GDN rollback** — reuses the `LinearAttnCache` snapshot/restore.
/// 3. **Raw embed accessor** — the drafter embeds its masked block through the
///    verifier's embed layer.
///
/// `step_fn` is invoked once per emitted (verifier-confirmed) token; return
/// `Some(id)` to force the next token (unused here — kept symmetric with the
/// MTP path). Returns the emitted token ids.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub fn dflash_generate_greedy(
    verifier: &Architecture,
    drafter: &mut DFlashDrafter,
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
    use std::time::Instant;

    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "dflash_generate_greedy: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "dflash_generate_greedy: DFlash verifier must be the Qwen3.5/3.6-MoE \
             hybrid (needs GDN lin_caches)"
                .into(),
        ));
    }

    let target_layer_ids = drafter.cfg.target_layer_ids.clone();
    let hidden = drafter.cfg.hidden_size as i32;
    let block_total = requested_block_total.min(drafter.cfg.block_size).max(2);

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

    // Diagnostics.
    let mut total_draft = 0usize;
    let mut total_accept = 0usize;
    let mut rounds = 0usize;
    let mut recent: Vec<(usize, usize)> = Vec::new();
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

    // -- Round-0: feed the last prompt token, capture its hidden + first bonus. --
    let last_prompt = *prompt_ids.last().unwrap();
    let (r0_logits, r0_hidden) = verifier.forward_verify_capture(
        &[last_prompt],
        1,
        &target_layer_ids,
        &mut v_caches,
        Some(&mut v_lin),
        device,
    )?;
    // Accumulated conditioning context (concat of every round's committed
    // verifier hidden along the sequence axis). The drafter conditions on the
    // FULL accumulated context each round — equivalent to the Python ref's
    // persistent draft KV cache (`cache.update_and_fetch`), which accumulates
    // context K/V derived deterministically from these same hiddens.
    super::guard_verifier_prefill_logits(verifier, &r0_logits, prompt_ids.len())?;
    let mut h_ctx_raw = r0_hidden;
    let mut b = {
        let am = argmax(&r0_logits, -1, device)?;
        am.eval()?;
        u32::from_le_bytes(am.to_bytes()?[..4].try_into().unwrap())
    };
    // Emit the first bonus.
    emit_step(tokenizer, b, step_fn, &mut emitted, &mut window);
    if eos_ids.contains(&b) {
        // The stop token arrived before a round could run. The request still
        // happened, so it still leaves exactly one record.
        super::RoundStats {
            loop_kind: super::SpecLoop::DFlash,
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
        }
        .log_done();
        return Ok(emitted);
    }

    tracing::info!(
        block_size = block_total,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        ?target_layer_ids,
        "dflash_generate_greedy: starting (Qwen3.6-MoE verifier + DFlash drafter)"
    );

    let seed_emitted = emitted.len();
    let mut emitted_in_rounds = 0usize;
    let round_loop_t0 = Instant::now();
    while emitted.len() < n_tokens {
        rounds += 1;
        let remaining = n_tokens - emitted.len();
        let bs = dflash_next_block_size(&recent, block_total, remaining + 1, false);
        if bs <= 1 {
            break;
        }

        // -- Project the committed verifier hidden into the conditioning ctx. --
        let h_ctx = drafter.project_condition(&h_ctx_raw)?;

        // -- Phase A: drafter proposes bs-1 tokens (non-autoregressive block). --
        let t0 = Instant::now();
        let draft_tokens = drafter.draft_block(verifier, b, &h_ctx, bs)?;
        draft_ns += t0.elapsed().as_nanos();
        if draft_tokens.is_empty() {
            break;
        }
        total_draft += draft_tokens.len();

        // -- Phase B: verifier scores [b, draft...] + captures hidden in one pass.
        // Snapshot GDN state before the verify forward.
        let round_snap = DFlashRoundState::snapshot(&v_lin)?;
        let mut v_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        v_input.push(b);
        v_input.extend_from_slice(&draft_tokens);
        let v_k = v_input.len();

        let t0 = Instant::now();
        let (v_logits, v_hidden) = verifier.forward_verify_capture(
            &v_input,
            v_k,
            &target_layer_ids,
            &mut v_caches,
            Some(&mut v_lin),
            device,
        )?;
        let v_argmax = argmax(&v_logits, -1, device)?;
        v_argmax.eval()?;
        let vb = v_argmax.to_bytes()?;
        verifier_ns += t0.elapsed().as_nanos();
        let mut v_tokens: Vec<u32> = Vec::with_capacity(v_k);
        for i in 0..v_k {
            v_tokens.push(u32::from_le_bytes(vb[i * 4..i * 4 + 4].try_into().unwrap()));
        }

        // -- Phase C: greedy acceptance walk. --------------------------------
        let (accept, new_tokens) = walk_block_greedy(&draft_tokens, &v_tokens, remaining);
        total_accept += accept;
        recent.push((accept, draft_tokens.len()));

        // -- Emit accepted prefix + 1 correction/bonus. ----------------------
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

        // -- Phase D: rollback + next-round setup. ---------------------------
        // Committed positions this round = new_tokens.len(); the verifier
        // consumed v_k positions. On partial accept (accept < bs-1) the GDN
        // recurrent state ran ahead — restore the snapshot and replay the
        // kept prefix so it matches the truncated KV exactly.
        let n_committed = new_tokens.len();
        // Read the post-verify sequence offset from a FullAttention layer:
        // GDN (linear-attn) layers never advance their KvCache::offset (it
        // stays 0), so `v_caches[0]` (layer 0 is GDN for the Qwen3.5/3.6-MoE
        // hybrid) would report 0 and drive `v_target` negative. The FA layers
        // carry the real sequence length — take the max across all caches.
        let v_offset_before = v_caches.iter().map(|c| c.offset()).max().unwrap_or(0);
        // Valid prefix length after the round = pre-round + accepted + 1
        // (the correction token at v_tokens[accept] is a prediction; the
        // verifier processed it as part of v_input only when it was a draft
        // token — i.e. the committed-position count is `accept` consumed
        // draft slots + the carry b). KV target = pre + accept + 1 carry-rows.
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
                device,
            )?;
        } else {
            // Full accept — GDN already correct; drop the snapshot.
            drop(round_snap);
        }

        // Append this round's committed verifier hidden to the accumulated
        // conditioning context (mlx-vlm: the committed `hidden[:, :accepted+1]`
        // is fed as NEW context into the persistent draft cache, which holds
        // all prior rounds). We accumulate the equivalent hidden buffer.
        let committed_hidden = v_hidden.slice(
            &[0, 0, 0],
            &[
                1,
                n_committed as i32,
                hidden * target_layer_ids.len() as i32,
            ],
            &[1, 1, 1],
            device,
        )?;
        h_ctx_raw = concatenate(&[&h_ctx_raw, &committed_hidden], 1, device)?;
        b = *new_tokens.last().unwrap_or(&b);

        tracing::debug!(
            round = rounds,
            accept,
            num_draft = draft_tokens.len(),
            n_committed,
            emitted_total = emitted.len(),
            v_offset_before,
            v_target,
            "dflash round"
        );
    }

    let round_loop_ns = round_loop_t0.elapsed().as_nanos();
    super::RoundStats {
        loop_kind: super::SpecLoop::DFlash,
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
// Loader
// ---------------------------------------------------------------------------

/// Refuse a snapshot carrying tensors this loader never reads.
///
/// A DFlash checkpoint of a later generation ships weight families this loader
/// has no code for — a candidate selector, per-layer dynamic convolutions.
/// Building the drafter out of the remainder yields **this loader's**
/// architecture wearing the checkpoint's name, and the accept rate measured
/// from it is filed under that name: `decode_config` records `dflash/block=N`
/// either way and cannot tell the two apart, so the row outlives any warning
/// and cannot be re-attributed afterwards. Refusing is what keeps that row from
/// being written; it costs the supported checkpoint nothing, which reads every
/// tensor it ships.
///
/// `Ok(())` means every tensor in the snapshot was consumed. It returns a
/// `Result` rather than an `Option<Error>` so a call site that stops propagating
/// it is an `unused_must_use` warning, which `-D warnings` turns into a build
/// failure — the guard cannot be un-wired quietly.
fn unread_tensor_refusal(present: &HashSet<String>, consumed: &HashSet<String>) -> Result<()> {
    let mut unread: Vec<&str> = present
        .iter()
        .map(String::as_str)
        .filter(|name| !consumed.contains(*name))
        .collect();
    if unread.is_empty() {
        return Ok(());
    }
    unread.sort_unstable();
    Err(Error::Model(format!(
        "DFlashDrafter: the snapshot carries {} tensors this loader does not read \
         ({}); the drafter built from the rest would be this loader's architecture \
         and not the checkpoint's, and any accept rate measured from it would be \
         recorded under the checkpoint's name with nothing in the row to say so. \
         Refusing rather than serving a drafter that is not the one named.",
        unread.len(),
        unread.join(", ")
    )))
}

/// Load the DFlash drafter tensors + config from `draft_dir`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn load_dflash(draft_dir: &Path, hidden_size: usize, device: Device) -> Result<DFlashDrafter> {
    use rmlx_loader::{load_config, load_shard_index, ShardSet};

    let cfg_raw = load_config(draft_dir)
        .map_err(|e| Error::Model(format!("DFlashDrafter: load_config: {e}")))?;
    let arch = cfg_raw.architectures.first().map_or("", String::as_str);
    if !arch.contains("DFlash") {
        tracing::warn!(
            draft = %draft_dir.display(),
            %arch,
            "DFlashDrafter: expected architecture DFlashDraftModel; proceeding by tensor names"
        );
    }

    // Typed-field-then-extras helpers (mirrors gemma4_assistant loader idiom).
    let extras = &cfg_raw.extras;
    let g_usize = |k: &str, default: usize| -> usize {
        extras
            .get(k)
            .and_then(serde_json::Value::as_u64)
            .map_or(default, |v| v as usize)
    };
    let g_f32 = |k: &str, default: f32| -> f32 {
        extras
            .get(k)
            .and_then(serde_json::Value::as_f64)
            .map_or(default, |v| v as f32)
    };

    let dflash_block = extras.get("dflash_config");
    let mask_token_id = dflash_block
        .and_then(|d| d.get("mask_token_id"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    let target_layer_ids: Vec<usize> = dflash_block
        .and_then(|d| d.get("target_layer_ids"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|v| v as usize))
                .collect()
        })
        .unwrap_or_default();
    let final_logit_softcapping = extras
        .get("final_logit_softcapping")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32);

    let cfg = DFlashConfig {
        hidden_size: g_usize("hidden_size", 2048),
        num_hidden_layers: g_usize("num_hidden_layers", 8),
        num_attention_heads: g_usize("num_attention_heads", 32),
        num_key_value_heads: g_usize("num_key_value_heads", 4),
        head_dim: g_usize("head_dim", 128),
        rms_norm_eps: g_f32("rms_norm_eps", 1e-6),
        rope_theta: g_f32("rope_theta", 1.0e7),
        block_size: g_usize("block_size", 16),
        mask_token_id,
        target_layer_ids,
        final_logit_softcapping,
    };

    // Validate against the verifier hidden size.
    if cfg.hidden_size != hidden_size {
        return Err(Error::Model(format!(
            "DFlashDrafter: drafter hidden_size {} != verifier hidden_size {} \
             (wrong draft model?)",
            cfg.hidden_size, hidden_size
        )));
    }
    if cfg.target_layer_ids.is_empty() {
        return Err(Error::Model(
            "DFlashDrafter: config missing dflash_config.target_layer_ids".into(),
        ));
    }

    let idx = load_shard_index(draft_dir)
        .map_err(|e| Error::Model(format!("DFlashDrafter: shard index: {e}")))?;
    let shards = ShardSet::open(draft_dir, &idx)
        .map_err(|e| Error::Model(format!("DFlashDrafter: open: {e}")))?;

    // Which tensor names the loader actually consumed. A drafter checkpoint of
    // a later DFlash generation carries families this loader has no code for
    // (a candidate selector, per-layer dynamic convolutions); loading it then
    // silently yields the earlier architecture built out of the subset it does
    // recognise, running at an accept rate that is not the checkpoint's. The
    // set is compared against the snapshot below so that downgrade is stated.
    let consumed: RefCell<HashSet<String>> = RefCell::new(HashSet::new());

    let load = |name: &str| -> Result<Array> {
        consumed.borrow_mut().insert(name.to_owned());
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("DFlashDrafter: safetensors: {e}")))?;
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
            "DFlashDrafter: tensor '{name}' not found"
        )))
    };
    let lin = |name: &str| -> Result<Linear> {
        Ok(Linear::Plain {
            weight: load(name)?,
        })
    };
    let norm = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Some(load(name)?),
            eps: cfg.rms_norm_eps,
        })
    };

    // `fc.weight`: [hidden, len(target_layer_ids)*hidden] — validates the head
    // matches both the verifier width and the configured target-layer count.
    let fc_w = load("fc.weight")?;
    let fc_shape = fc_w.shape().to_vec();
    let expect_in = cfg.target_layer_ids.len() * cfg.hidden_size;
    if fc_shape.len() != 2
        || fc_shape[0] as usize != cfg.hidden_size
        || fc_shape[1] as usize != expect_in
    {
        return Err(Error::Model(format!(
            "DFlashDrafter: fc.weight shape {fc_shape:?} != [{}, {}] \
             (hidden_size / target_layer_ids mismatch)",
            cfg.hidden_size, expect_in
        )));
    }
    let fc = Linear::Plain { weight: fc_w };
    let hidden_norm = norm("hidden_norm.weight")?;
    let final_norm = norm("norm.weight")?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        layers.push(DFlashLayer {
            input_layernorm: norm(&format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
            q_proj: lin(&format!("{p}.self_attn.q_proj.weight"))?,
            k_proj: lin(&format!("{p}.self_attn.k_proj.weight"))?,
            v_proj: lin(&format!("{p}.self_attn.v_proj.weight"))?,
            o_proj: lin(&format!("{p}.self_attn.o_proj.weight"))?,
            q_norm: norm(&format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: norm(&format!("{p}.self_attn.k_norm.weight"))?,
            mlp: Mlp {
                gate_proj: lin(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: lin(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: lin(&format!("{p}.mlp.down_proj.weight"))?,
                activation: Activation::Silu,
            },
        });
    }

    // A set, not a list: a name carried by two shard files would otherwise be
    // counted and listed twice in the refusal.
    let mut present: HashSet<String> = HashSet::new();
    for (_, handle) in shards.iter() {
        let st = handle
            .safetensors()
            .map_err(|e| Error::Model(format!("DFlashDrafter: safetensors: {e}")))?;
        present.extend(st.names().into_iter().map(ToOwned::to_owned));
    }
    unread_tensor_refusal(&present, &consumed.borrow())?;

    // YARN RoPE: the Qwen3.6 DFlash drafter is trained with rope_scaling
    // {rope_type: yarn}. Precompute its inverse-freq table + mscale so the
    // drafter attention matches the checkpoint (plain RoPE diverges materially
    // even at small offsets — numeric finding).
    let (rope_freqs, rope_mscale) = match extras.get("rope_scaling") {
        Some(rs)
            if rs
                .get("rope_type")
                .or_else(|| rs.get("type"))
                .and_then(|v| v.as_str())
                == Some("yarn") =>
        {
            let f = |k: &str, d: f32| {
                rs.get(k)
                    .and_then(serde_json::Value::as_f64)
                    .map_or(d, |v| v as f32)
            };
            let factor = f("factor", 1.0);
            let original = f("original_max_position_embeddings", 4096.0);
            let beta_fast = f("beta_fast", 32.0);
            let beta_slow = f("beta_slow", 1.0);
            let yarn_cfg = crate::rope::YarnConfig {
                factor,
                original_max_position_embeddings: original,
                beta_fast,
                beta_slow,
            };
            let (freqs, mscale) =
                crate::rope::compute_yarn_freqs(cfg.head_dim, cfg.rope_theta, yarn_cfg)?;
            tracing::info!(
                factor,
                original,
                beta_fast,
                beta_slow,
                mscale,
                "DFlashDrafter: YARN RoPE active"
            );
            (Some(freqs), mscale)
        }
        _ => (None, 1.0),
    };

    Ok(DFlashDrafter {
        fc,
        hidden_norm,
        norm: final_norm,
        layers,
        caches: Vec::new(),
        rope_freqs,
        rope_mscale,
        cfg,
        device,
    })
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
#[cfg(test)]
mod yarn_freq_check;
