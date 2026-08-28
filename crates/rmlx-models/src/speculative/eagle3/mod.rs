// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! EAGLE-3 drafter loader + round-loop.
//!
//! Port of mlx-vlm `mlx_vlm/speculative/drafters/eagle3/eagle3.py`
//! (`Eagle3DraftModel`) and the round-loop in `mlx_vlm/speculative/eagle3.py`
//! (`_eagle3_next_block_size`, `_eagle3_rounds`, `_eagle3_walk`). The
//! authoritative weight layout is the mainline SpecForge
//! `LlamaForCausalLMEagle3` model
//! (`sgl-project/SpecForge:specforge/modeling/draft/llama3_eagle.py`).
//!
//! # What an EAGLE-3 drafter is
//!
//! EAGLE-3 (Li et al. 2025, arXiv:2503.01840) drafts tokens **autoregressively**
//! with a single transformer decoder layer conditioned on the verifier's
//! **multi-layer fused hidden state**. Three properties define it:
//!
//! 1. **Multi-layer feature fusion.** It reads the verifier residual stream at
//!    three auxiliary layers (`eagle_aux_hidden_state_layer_ids`, here
//!    `[3, 19, 35]` for the Qwen3.6-35B-A3B target), concatenates them along the
//!    feature axis (`3*H = 6144`), and projects back to `H = 2048` through a
//!    `fc` Linear (`fc.weight [2048, 6144]`, no bias).
//! 2. **Embed + hidden fusion in the first layer.** The single decoder layer
//!    (`Eagle3FirstLayer`) attends over `concat(input_layernorm(embed),
//!    hidden_norm(fc_out))` — a `2*H = 4096`-wide attention input
//!    (`q/k/v_proj` are `[*, 4096]`). The residual is the un-concatenated
//!    `fc_out` hidden.
//! 3. **Reduced draft vocab + d2t remap.** The drafter's `lm_head`
//!    (`[32000, 2048]`) predicts over a 32 000-token *draft* vocabulary, a
//!    subset of the verifier's 248 320-token vocabulary. A `d2t` buffer
//!    (`[32000] i64`) maps a draft id back to the target id via
//!    `target = draft + d2t[draft]`. (`t2d`, the inverse mask, is unused at
//!    inference and dropped at load.)
//!
//! Drafting is autoregressive (one token per step, advancing the drafter's own
//! KV cache), unlike the DFlash non-autoregressive block. The seed token
//! and seed hidden come from the verifier's last position.
//!
//! # Status — document-the-truth (CLAUDE.md hard rule 7)
//!
//! **reference-alignment pass.** Three structural divergences from
//! mlx-vlm reference identified and patched; live accept-rate
//! measurement pending — see BENCHMARK_CHAMPIONS.md.
//!
//! Per-step trace: `RUST_LOG=rmlx_models::speculative::eagle3=trace`.
//!
//! Reuses the three verifier-side seams against the Qwen3.6-MoE verifier:
//!
//! 1. **Multi-layer hidden capture** — [`Architecture::forward_verify_capture`]
//!    returns last-k logits AND the `aux_layer_ids` residual-stream hidden
//!    concatenated along the feature axis in one cached forward. We capture at
//!    `[id - 1]` for each `eagle_aux_hidden_state_layer_ids` id (mlx-vlm
//!    `capture_layer_ids`), i.e. `[2, 18, 34]`.
//! 2. **GDN rollback** — reuses [`super::dflash::DFlashRoundState`]
//!    snapshot/restore + kept-prefix replay on partial acceptance (GDN recurrent
//!    state has no sequence axis; it cannot be truncated).
//! 3. **Raw embed accessor** — [`Architecture::embed_tokens_raw`] (the verifier
//!    `embed_tokens` is the EAGLE-3 `bind()` target; the checkpoint ships no
//!    `embed_tokens` tensor, so the drafter embeds its tokens through the
//!    verifier's unscaled embedding).
//!
//! The drafter loads, drives the verifier, and produces **coherent** output
//! (the verifier corrects every miss, so quality is the verifier's). The forward
//! mirrors the mlx-vlm `Eagle3DraftModel` + canonical SpecForge
//! `LlamaForCausalLMEagle3` reference (`Eagle3FirstLayer` embed/hidden fusion,
//! autoregressive own-hidden feed, d2t remap) *plus* the speculators-format
//! per-aux `fcs` norm (below).
//!
//! **`fcs.0/1/2` (`[2048]` each) — applied (verified accept-rate win, ).**
//! The Dogacel checkpoint (a *speculators*-format export) carries three extra
//! per-aux RMSNorm weight vectors that neither the mainline SpecForge
//! `LlamaForCausalLMEagle3` model nor the mlx-vlm `Eagle3DraftModel` reference
//! defines (mlx-vlm `sanitize` silently drops them; it cannot load this variant
//! faithfully). The trained behavior is to RMSNorm each of the 3 aux hidden
//! states by `fcs.{0,1,2}` *before* the `fc` concat-projection. Applying them
//! more than doubled the live greedy accept-rate (0.09 -> 0.21). Auto-detected
//! by tensor presence; `RMLX_EAGLE3_NO_FCS=1` forces the raw-concat fallback.
//!
//! **: seed-token double-processing bug — fixed.** Via mlx-vlm reference
//! diff, two related bugs were identified:
//!
//! 1. **Prefill path**: `prefill_from_verifier_hidden` returned the drafter's
//!    hidden at the bonus position. The round-loop then called
//!    `draft_block(bonus, h_seed)` which ran `forward_token(bonus, h_seed)` —
//!    processing bonus through the drafter a SECOND time using its own output as
//!    conditioning (the drafter had already processed bonus in the prefill).
//! 2. **Per-round seeding**: `accept_and_reseed` returned `h_seed` (drafter's
//!    hidden at correction). The round-loop then called
//!    `draft_block(correction, h_seed)` which ran `forward_token(correction, h_seed)` —
//!    again a second forward pass for correction.
//!
//! Root cause: the round-loop was treating `h_seed` as an INPUT conditioning
//! hidden (to be fed with a new token), whereas `h_seed` is the drafter's OUTPUT
//! at the seed position (the drafter has already processed the seed token).
//!
//! Fix: `accept_and_reseed` and `prefill_from_verifier_hidden` now return
//! `(h_seed, seed_tok)` where `seed_tok = greedy_target_token(h_seed)` mirrors
//! mlx-vlm's `_seed_token`. `draft_block` gains `precomputed_first_tok:
//! Option<u32>` — when `Some(t)`, `t` is prepended as the first draft token
//! WITHOUT a forward pass, and the loop runs `block_size - 2` more times.
//! This exactly mirrors mlx-vlm `Eagle3DraftModel.draft_block` with `_seed_token`
//! set. Net effect: each round the drafter makes one more genuine prediction.

#![allow(
    clippy::cognitive_complexity,
    clippy::implicit_clone,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::used_underscore_binding
)]
// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{add, argmax, concatenate, rope, Array, Device};

use super::dflash::DFlashRoundState;
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use crate::layers::{Activation, Linear, Mlp, RmsNorm};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache, KV_MAX_SEQ_DEFAULT};

/// Choose the next EAGLE-3 verify block size.
///
/// Pure port of `_eagle3_next_block_size` (non-adaptive branch). mlx-vlm's
/// `_eagle3_rounds` honors the configured/requested size capped to the remaining
/// budget (the Dogacel drafter advertises no `adaptive_max_block_size`, so the
/// adaptive tier walk never fires). Returns the next block total (including the
/// seed/bonus token).
pub fn eagle3_next_block_size(requested_block_total: usize, remaining_budget: usize) -> usize {
    requested_block_total.min(remaining_budget)
}

/// One greedy EAGLE-3 acceptance walk over a drafted block.
///
/// Pure port of `_eagle3_walk`: accept drafted tokens up to the first mismatch
/// with the verifier's greedy choice, then take the verifier's correction/bonus
/// at that position. Returns `(accepted, new_tokens)` capped at `budget`.
/// `target_tokens` are the verifier's greedy predictions for positions
/// `[b, d0, d1, ...]` — `draft_tokens.len() + 1` of them.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn eagle3_walk(
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

/// Find the first position where the restricted-vocab verifier token differs from the
/// corresponding draft token, over the `n_draft` draft positions.
///
/// Returns the index of the first mismatch in `0..n_draft`, or `n_draft` when all
/// draft positions match (the correction falls on the bonus position).
///
/// This is the "Step 4" of the EAGLE-3 hot-path verify logic. Factored here so
/// it can be unit-tested independently of the Metal forward pass.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn find_full_pos(draft_tokens: &[u32], tokens: &[u32]) -> usize {
    let n_draft = draft_tokens.len().min(tokens.len());
    for i in 0..n_draft {
        if draft_tokens[i] != tokens[i] {
            return i;
        }
    }
    n_draft
}

/// Map an EAGLE-3 draft-vocab id to the target-vocab id.
///
/// Pure port of `Eagle3DraftModel._draft_to_target`: `target = draft + d2t[draft]`.
/// When the drafter shares the verifier's vocabulary (`d2t` empty / absent) the
/// id passes through unchanged.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn draft_to_target(draft_id: u32, d2t: &[i32]) -> u32 {
    if d2t.is_empty() {
        return draft_id;
    }
    let idx = draft_id as usize;
    if idx >= d2t.len() {
        return draft_id;
    }
    (i64::from(draft_id) + i64::from(d2t[idx])) as u32
}

/// EAGLE-3 drafter config (subset rMLX needs), parsed from `config.json`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Eagle3 drafter contract; adding a field requires updating parse_eagle3_config and Eagle3Drafter::load"
)]
#[derive(Debug, Clone)]
/// Parsed Eagle3 speculative-drafter config (see docs/SPECULATIVE.md).
pub struct Eagle3Config {
    /// Drafter hidden dimension (must equal verifier hidden size).
    pub hidden_size: usize,
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
    /// Restricted draft vocabulary size (hot-path; may be smaller than verifier vocab).
    pub draft_vocab_size: usize,
    /// Full verifier vocabulary size (used for fallback / tree verification).
    pub vocab_size: usize,
    /// Verifier residual-stream layer indices the drafter conditions on
    /// (`eagle_aux_hidden_state_layer_ids`). 3 entries for the Dogacel drafter.
    pub aux_layer_ids: Vec<usize>,
    /// Block ceiling (`block_size`). mlx-vlm default 5 (speculative_tokens + 2).
    pub block_size: usize,
}

/// Loaded EAGLE-3 drafter weights + config.
///
/// Construct with [`Eagle3Drafter::load`]. The drafter has a single decoder
/// layer (`Eagle3FirstLayer`); the token embedding and the d2t-mapped target ids
/// are resolved through the verifier (the round-loop threads it in), mirroring
/// the Python `bind()`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed drafter struct — private weight fields; public API is draft_step() and update_context(); adding a field requires updating load_eagle3 and Eagle3Drafter::load"
)]
#[allow(missing_debug_implementations)]
pub struct Eagle3Drafter {
    /// `fc`: Linear `3*H -> H`, no bias (fuses the 3 aux hidden states).
    fc: Linear,
    /// Per-aux RMSNorms (`fcs.{0,1,2}`, `[H]` each) applied to each aux hidden
    /// slice before the `fc` concat-projection. `Some` for the speculators-format
    /// Dogacel checkpoint (auto-detected by tensor presence; doubled the live
    /// accept-rate — module docs), `None` for the canonical SpecForge / mlx-vlm
    /// layout that runs `fc` on the raw concat.
    fcs: Option<Vec<RmsNorm>>,
    /// RMSNorm applied to the projected `fc` output in the first layer.
    hidden_norm: RmsNorm,
    /// RMSNorm applied to the token embedding in the first layer.
    input_layernorm: RmsNorm,
    /// Post-attention RMSNorm in the first layer.
    post_attention_layernorm: RmsNorm,
    /// First-layer attention projections (q/k/v take the `2*H` concat input).
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// First-layer SwiGLU MLP.
    mlp: Mlp,
    /// Final RMSNorm before the draft `lm_head`.
    norm: RmsNorm,
    /// Draft-vocab output head (`[draft_vocab_size, H]`).
    lm_head: Linear,
    /// `d2t[draft_id]` offset table (host-side; `target = draft + d2t[draft]`).
    d2t: Vec<i32>,
    /// Hot-path target-vocab ids for the restricted-vocab verifier matmul.
    ///
    /// `hot_ids_host[i] = i + d2t[i]` — the target-vocab token id corresponding
    /// to each draft-vocab position. Precomputed at load time (host-side Vec).
    /// Empty when `d2t` is empty (same vocabulary — hot-path is inactive).
    hot_ids_host: Vec<u32>,
    /// MLX Array view of `hot_ids_host` (`[draft_vocab_size]` i32), used as
    /// index argument to `Architecture::hot_logits_from_final_hidden`. `None`
    /// when `d2t` is empty.
    hot_ids_arr: Option<Array>,
    /// The drafter's own KV cache (single layer).
    cache: KvCache,
    cfg: Eagle3Config,
    device: Device,
}

impl Eagle3Drafter {
    /// Load an EAGLE-3 drafter from `draft_dir`, validating it against the
    /// verifier's hidden size and vocabulary.
    ///
    /// `draft_dir` is the standalone EAGLE-3 folder (`config.json` with
    /// `architectures: ["LlamaForCausalLMEagle3"]` + `eagle_config`, and a
    /// `model.safetensors`). `hidden_size` is the verifier's model width; the
    /// drafter `hidden_size` and `fc` input (`3*H`) must match it. `vocab_size`
    /// is the verifier vocabulary; the drafter's `vocab_size` must match (the
    /// `d2t`-mapped target ids index it). `eos_token_ids` are the verifier's
    /// EOS token ids; they are appended to `hot_ids` so the restricted-vocab
    /// argmax can select EOS at intermediate positions (mirrors mlx-vlm's
    /// `_eagle3_hot_token_ids` which concatenates `eos_token_ids` onto `hot_ids`).
    pub fn load(
        draft_dir: &Path,
        hidden_size: usize,
        vocab_size: usize,
        eos_token_ids: &[u32],
        device: Device,
    ) -> Result<Self> {
        let me = load_eagle3(draft_dir, hidden_size, vocab_size, eos_token_ids, device)?;
        tracing::info!(
            draft = %draft_dir.display(),
            hidden_size,
            draft_vocab_size = me.cfg.draft_vocab_size,
            vocab_size = me.cfg.vocab_size,
            aux_layer_ids = ?me.cfg.aux_layer_ids,
            block_size = me.cfg.block_size,
            "Eagle3Drafter: loaded drafter"
        );
        Ok(me)
    }

    /// Reset the drafter's KV cache between generations.
    ///
    /// `max_seq` must cover the full generation horizon:
    /// `prompt_len + max_new_tokens`. The drafter cache advances by
    /// `accepted + 1` tokens per round; using `KV_MAX_SEQ_DEFAULT = 4096`
    /// here would overflow on any prompt whose length + target exceeds 4096,
    /// causing a silent MLX `slice_update` broadcast failure at the boundary.
    pub fn reset(&mut self, max_seq: i32) {
        self.cache = KvCache::with_quant_max_seq(KvQuant::None, max_seq);
    }

    /// Trained / configured block ceiling.
    pub fn block_size(&self) -> usize {
        self.cfg.block_size
    }

    /// Verifier residual-stream layer indices the drafter conditions on.
    pub fn aux_layer_ids(&self) -> &[usize] {
        &self.cfg.aux_layer_ids
    }

    /// Hidden size the drafter was loaded for.
    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// The `d2t` offset table (`target = draft + d2t[draft]`).
    pub fn d2t(&self) -> &[i32] {
        &self.d2t
    }

    /// Whether the hot-path is active (draft vocab != verifier vocab and d2t present).
    pub fn hot_path_active(&self) -> bool {
        !self.hot_ids_host.is_empty()
    }

    /// MLX int32 array of hot target-vocab ids (`[draft_vocab_size]`), or `None`
    /// when the hot-path is inactive. Used as index arg to
    /// `Architecture::hot_logits_from_final_hidden`.
    pub fn hot_ids_arr(&self) -> Option<&Array> {
        self.hot_ids_arr.as_ref()
    }

    /// Host-side hot target-vocab ids (`hot_ids[i] = i + d2t[i]`). Empty when
    /// the hot-path is inactive.
    pub fn hot_ids_host(&self) -> &[u32] {
        &self.hot_ids_host
    }

    /// Project the verifier's concatenated 3-aux hidden into the drafter width:
    /// `fc(concat_hidden)`. Mirrors `Eagle3DraftModel._prepare_target_hidden`
    /// (with `input_norm == None`, the default for this checkpoint). `concat_hidden`
    /// is `[1, n, 3*H]`; returns `[1, n, H]`.
    ///
    /// When `fcs` is present (speculators checkpoint + `RMLX_EAGLE3_FCS=1`), each
    /// aux slice is RMSNorm'd by `fcs.{0,1,2}` before re-concatenation — an
    /// in-progress numeric-alignment hypothesis for the Dogacel accept gap (see
    /// module docs). Default OFF (raw concat) matches the mlx-vlm reference.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn project_hidden(&self, concat_hidden: &Array) -> Result<Array> {
        let device = self.device;
        match &self.fcs {
            Some(fcs) => {
                let h = self.cfg.hidden_size as i32;
                let mut slices: Vec<Array> = Vec::with_capacity(fcs.len());
                let dims = concat_hidden.shape();
                let n = dims[1];
                for (i, norm) in fcs.iter().enumerate() {
                    let start = i as i32 * h;
                    let slice = concat_hidden.slice(
                        &[0, 0, start],
                        &[1, n, start + h],
                        &[1, 1, 1],
                        device,
                    )?;
                    slices.push(norm.forward(&slice, device)?);
                }
                let refs: Vec<&Array> = slices.iter().collect();
                let normed = concatenate(&refs, 2, device)?;
                self.fc.forward(&normed, device)
            }
            None => self.fc.forward(concat_hidden, device),
        }
    }

    /// One autoregressive EAGLE-3 draft step over a single token.
    ///
    /// Port of `Eagle3DraftModel._forward_tokens` for `tokens.shape[1] == 1`:
    /// embed the token (verifier embedding), fuse with the conditioning hidden
    /// `h_cond` through the `Eagle3FirstLayer`, advancing the drafter KV cache.
    /// Returns the **un-normed** layer output `[1, 1, H]` — this feeds forward as
    /// the next step's conditioning hidden (mlx-vlm threads `h_prev`); the final
    /// RMSNorm is applied separately by [`Self::greedy_target_token`] for logits.
    fn forward_token(
        &mut self,
        verifier: &Architecture,
        tok: u32,
        h_cond: &Array,
    ) -> Result<Array> {
        let device = self.device;
        let embed = verifier.embed_tokens_raw(&[tok as i32], device)?; // [1,1,H]
        self.first_layer(&embed, h_cond)
    }

    /// The single `Eagle3FirstLayer.__call__` forward (causal, KV-cached).
    ///
    /// Port of mlx-vlm `Eagle3FirstLayer`: attention input is
    /// `concat(input_layernorm(embed), hidden_norm(h_proj))` (`2*H`); the residual
    /// is the raw projected hidden `h_proj` (`norm_before_residual=False`). The
    /// RoPE offset and causal mask come from the drafter's KV cache.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn first_layer(&mut self, embed: &Array, h_proj: &Array) -> Result<Array> {
        let device = self.device;
        let n_heads = self.cfg.num_attention_heads as i32;
        let n_kv = self.cfg.num_key_value_heads as i32;
        let hd = self.cfg.head_dim as i32;
        let l = embed.shape()[1]; // 1 (autoregressive)
        let offset = self.cache.offset();

        let embed_n = self.input_layernorm.forward(embed, device)?;
        let hidden_n = self.hidden_norm.forward(h_proj, device)?;
        // Residual is the un-normed projected hidden (norm_before_residual=False).
        let residual = h_proj.try_clone()?;

        // Attention input: concat(embed_n, hidden_n) -> [1, L, 2H].
        let attn_in = concatenate(&[&embed_n, &hidden_n], 2, device)?;

        let q = self.q_proj.forward(&attn_in, device)?;
        let k = self.k_proj.forward(&attn_in, device)?;
        let v = self.v_proj.forward(&attn_in, device)?;

        // [1,L,heads,hd] -> [1,heads,L,hd], RoPE at the cache offset.
        let q = q
            .reshape(&[1, l, n_heads, hd], device)?
            .transpose(&[0, 2, 1, 3], device)?;
        let q = rope(&q, hd, false, self.cfg.rope_theta, 1.0, offset, device)?;
        let k = k
            .reshape(&[1, l, n_kv, hd], device)?
            .transpose(&[0, 2, 1, 3], device)?;
        let k = rope(&k, hd, false, self.cfg.rope_theta, 1.0, offset, device)?;
        let v = v
            .reshape(&[1, l, n_kv, hd], device)?
            .transpose(&[0, 2, 1, 3], device)?;

        // Causal self-attention over the drafter's own KV cache (advances it).
        let scale = (hd as f32).powf(-0.5);
        let attn = self
            .cache
            .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)?;
        let attn = attn
            .transpose(&[0, 2, 1, 3], device)?
            .reshape(&[1, l, n_heads * hd], device)?;
        let attn = self.o_proj.forward(&attn, device)?;
        let h = add(&residual, &attn, device)?;

        let residual = h.try_clone()?;
        let f = self.post_attention_layernorm.forward(&h, device)?;
        let f = self.mlp.forward(&f, device)?;
        add(&residual, &f, device)
    }

    /// Compute draft logits from an un-normed layer-output hidden, greedy-pick,
    /// then d2t-map to the target vocabulary.
    ///
    /// Port of `Eagle3DraftModel._logits` + `_sample(greedy=True)`:
    /// `lm_head(norm(hidden))`. `hidden` is the **un-normed** `[1, 1, H]` layer
    /// output from [`Self::forward_token`]; this applies the final `norm` before
    /// the head. Returns the target-vocab token id.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn greedy_target_token(&self, hidden: &Array) -> Result<u32> {
        let device = self.device;
        let normed = self.norm.forward(hidden, device)?;
        let logits = self.lm_head.forward(&normed, device)?; // [1,1,draft_vocab]
        let am = argmax(&logits, -1, device)?;
        am.eval()?;
        let draft_id = u32::from_le_bytes(am.to_bytes()?[..4].try_into().unwrap());
        Ok(draft_to_target(draft_id, &self.d2t))
    }

    /// Draft a block of `block_size - 1` tokens autoregressively.
    ///
    /// Port of `Eagle3DraftModel.draft_block`. Two calling conventions:
    ///
    /// **Seeded path** (`precomputed_first_tok = Some(t)`): mirrors mlx-vlm when
    /// `_seed_token` is set. `t` is already the drafter's prediction at the seed
    /// position (computed by `accept_and_reseed` / `prefill_from_verifier_hidden`
    /// via [`Self::greedy_target_token`]); it is prepended as `tokens[0]` WITHOUT
    /// a forward pass. The loop then runs `block_size - 2` times starting from
    /// `forward_token(t, h_seed)`. Total output: `block_size - 1` tokens.
    ///
    /// **Unseeded path** (`precomputed_first_tok = None`): used for the first
    /// round when the drafter has NOT been prefilled (`h_seed` is `fc`-projected
    /// verifier hidden, carry token not yet through the drafter). `carry_tok` is
    /// the unprocessed carry token; the loop runs `block_size - 1` times starting
    /// from `forward_token(carry_tok, h_seed)`. This matches the non-prefill
    /// first-round behaviour in mlx-vlm where `_seed_token` is None.
    pub fn draft_block(
        &mut self,
        verifier: &Architecture,
        carry_tok: u32,
        h_seed: &Array,
        precomputed_first_tok: Option<u32>,
        block_size: usize,
    ) -> Result<Vec<u32>> {
        if block_size <= 1 {
            return Ok(vec![]);
        }
        let mut tokens: Vec<u32> = Vec::with_capacity(block_size - 1);
        let mut h_cond = h_seed.try_clone()?;

        // Seeded path: the first token is already known; prepend it and reduce
        // loop count by 1 so we still produce exactly `block_size - 1` tokens.
        let mut tok = if let Some(first_tok) = precomputed_first_tok {
            tokens.push(first_tok);
            first_tok
        } else {
            carry_tok
        };

        let n_iters = (block_size - 1).saturating_sub(tokens.len());
        for _ in 0..n_iters {
            // forward_token returns the un-normed layer output (H), which
            // becomes the conditioning hidden for the next step.
            h_cond = self.forward_token(verifier, tok, &h_cond)?;
            tok = self.greedy_target_token(&h_cond)?;
            tokens.push(tok);
        }
        Ok(tokens)
    }

    /// Multi-token drafter forward conditioned on verifier 3-aux hiddens.
    ///
    /// Port of `Eagle3DraftModel._forward_tokens` for `L ≥ 1`: embeds all
    /// tokens, applies `project_hidden` (fc + optional per-aux fcs norms) to
    /// the 3-aux verifier hidden concat, then runs the Eagle3FirstLayer on all
    /// positions at once (causal mask, advancing the drafter KV cache by L).
    ///
    /// `tokens` has length `L`; `verifier_aux_hidden` is `[1, L, 3*H]`.
    /// Returns the un-normed layer output `[1, L, H]`.
    fn forward_tokens_conditioned(
        &mut self,
        verifier: &Architecture,
        tokens: &[u32],
        verifier_aux_hidden: &Array,
        device: Device,
    ) -> Result<Array> {
        let tok_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let embed = verifier.embed_tokens_raw(&tok_ids, device)?; // [1, L, H]
        let h_proj = self.project_hidden(verifier_aux_hidden)?; // [1, L, H]
        self.first_layer(&embed, &h_proj)
    }

    /// Accept-and-reseed: the per-round post-verification drafter KV update.
    ///
    /// Port of `Eagle3DraftModel.accept_verified_tokens`. Fixes the
    /// two structural bugs in the implementation:
    ///
    /// 1. **KV cache gap**: the old code truncated the drafter cache incorrectly.
    /// 2. **Seed hidden mismatch**: the old code fed `fc(verifier_aux)` directly
    ///    without the correct input transformation.
    ///
    /// Algorithm:
    /// 1. Roll drafter KV cache back to `pre_round_offset`.
    /// 2. Build the re-run sequence: `draft_tokens[0..accepted] + [correction]`.
    /// 3. Run the drafter forward on this sequence, advancing the KV cache by `block_size`.
    /// 4. Sample the drafter's next-token prediction from the last-position hidden.
    ///
    /// Returns:
    /// - `h_seed`: un-normed layer output at the correction position `[1, 1, H]`
    ///   (mirrors mlx-vlm `_seed_hidden`).
    /// - `seed_tok`: greedy drafter prediction from `h_seed`, d2t-mapped
    ///   (mirrors mlx-vlm `_seed_token`). Pass as `precomputed_first_tok` to
    ///   `draft_block` so the correction position is not re-processed.
    ///
    /// `draft_tokens`: the `bs-1` tokens proposed by `draft_block` this round.
    /// `correction`: `v_tokens[accepted]` — verifier's correction or bonus token.
    /// `v_hidden`: `[1, 1+num_draft, 3H]` — verifier 3-aux hidden for the round
    /// (`v_input = [b, draft[0], ..., draft[num_draft-1]]`).
    /// `accepted`: number of draft tokens accepted (0..num_draft).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn accept_and_reseed(
        &mut self,
        verifier: &Architecture,
        pre_round_offset: i32,
        draft_tokens: &[u32],
        correction: u32,
        v_hidden: &Array,
        accepted: usize,
        device: Device,
    ) -> Result<(Array, u32)> {
        // (1) Roll drafter KV cache back to pre-round state.
        self.truncate_cache(pre_round_offset);

        // (2) Build token sequence: accepted draft prefix + correction.
        let n = accepted + 1;
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        tokens.extend_from_slice(&draft_tokens[..accepted]);
        tokens.push(correction);

        // (3) Conditioning hiddens: v_hidden[:, 0..n, :] (the verifier hidden
        // at positions 0..accepted, which predicts draft[0..accepted-1] and
        // correction respectively).
        let h = self.cfg.hidden_size as i32;
        let hidden_slice = v_hidden.slice(
            &[0, 0, 0],
            &[1, n as i32, v_hidden.shape()[2]],
            &[1, 1, 1],
            device,
        )?;

        let out = self.forward_tokens_conditioned(verifier, &tokens, &hidden_slice, device)?;

        // (4) Extract the last-position hidden and greedily sample the drafter's
        // next-token prediction (mirrors mlx-vlm `_set_seed_from_hidden`).
        // The caller passes `seed_tok` to `draft_block` as `precomputed_first_tok`
        // so the correction position is not run through the drafter a second time.
        let h_seed = out.slice(&[0, n as i32 - 1, 0], &[1, n as i32, h], &[1, 1, 1], device)?;
        let seed_tok = self.greedy_target_token(&h_seed)?;
        Ok((h_seed, seed_tok))
    }

    /// Drafter prefill from verifier hidden states.
    ///
    /// Port of `Eagle3DraftModel.prefill_from_target_hidden`, extended
    /// with chunked execution to avoid Metal GPU timeouts on long prompts.
    ///
    /// The drafter has a single transformer layer whose `update_and_sdpa` kernel
    /// scales quadratically with sequence length; at ~800+ tokens the kernel
    /// exceeds the 4-5 s Metal watchdog on Qwen3.6-MoE. This method processes
    /// `shifted_tokens` in windows of at most `DRAFTER_PREFILL_CHUNK` tokens.
    /// Each chunk advances the drafter KV cache (so later chunks attend to all
    /// prior positions) and the corresponding `verifier_aux_hidden` slice is
    /// passed as conditioning. The last chunk's last-position hidden and the
    /// greedy next token are returned as `(h_seed, seed_tok)`.
    ///
    /// `shifted_tokens`: `prompt_ids[1..] + [first_bonus]` (length `n`).
    /// `verifier_aux_hidden`: `[1, n, 3H]` — verifier 3-aux hidden for all
    /// prompt positions from `forward_verify_capture_chunked`.
    ///
    /// Returns `(h_seed[1,1,H], seed_tok)` — same contract as the pre-
    /// single-shot path. For short prompts (n ≤ DRAFTER_PREFILL_CHUNK) this
    /// is a single `forward_tokens_conditioned` call — no overhead.
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn prefill_from_verifier_hidden(
        &mut self,
        verifier: &Architecture,
        shifted_tokens: &[u32],
        verifier_aux_hidden: &Array,
        device: Device,
    ) -> Result<(Array, u32)> {
        if shifted_tokens.is_empty() {
            return Err(Error::Model(
                "Eagle3Drafter::prefill_from_verifier_hidden: empty token sequence".into(),
            ));
        }

        // Maximum tokens per drafter-prefill chunk. The drafter's single
        // attention layer dispatches one fused kernel over the full chunk;
        // at chunk_size = 512, peak attention grid is 512×512 = 262k entries
        // which completes well within the Metal 4-5 s watchdog on Qwen3.6-MoE.
        const DRAFTER_PREFILL_CHUNK: usize = 512;

        let total = shifted_tokens.len();
        let h = self.cfg.hidden_size as i32;
        let aux_h = verifier_aux_hidden.shape()[2]; // 3 * hidden_size

        let mut pos = 0usize;
        let mut h_seed_out: Option<Array> = None;

        while pos < total {
            let end = (pos + DRAFTER_PREFILL_CHUNK).min(total);
            let chunk_toks = &shifted_tokens[pos..end];
            let chunk_n = chunk_toks.len() as i32;
            let is_last = end == total;

            tracing::debug!(pos, end, chunk_n, is_last, "eagle3 drafter prefill chunk");

            // Slice the verifier aux hidden for this chunk: [1, chunk_n, 3H].
            let hidden_slice = verifier_aux_hidden.slice(
                &[0, pos as i32, 0],
                &[1, pos as i32 + chunk_n, aux_h],
                &[1, 1, 1],
                device,
            )?;

            let out =
                self.forward_tokens_conditioned(verifier, chunk_toks, &hidden_slice, device)?;

            if is_last {
                // Extract the last-position hidden of this final chunk.
                let h_seed =
                    out.slice(&[0, chunk_n - 1, 0], &[1, chunk_n, h], &[1, 1, 1], device)?;
                h_seed_out = Some(h_seed);
            } else {
                // Intermediate chunks: materialise to release Metal intermediates.
                out.eval()?;
            }

            pos = end;
        }

        let h_seed = h_seed_out.expect("loop must have a last chunk");
        let seed_tok = self.greedy_target_token(&h_seed)?;
        Ok((h_seed, seed_tok))
    }

    /// Drafter cache position.
    pub fn cache_offset(&self) -> i32 {
        self.cache.offset()
    }

    /// Truncate the drafter cache to `n` positions (partial-accept rollback).
    pub fn truncate_cache(&mut self, n: i32) {
        if self.cache.offset() >= n {
            self.cache.truncate_to(n);
        }
    }
}

// ---------------------------------------------------------------------------
// Round-loop
// ---------------------------------------------------------------------------

/// Emit a single token through `step_fn` + the running `emitted` buffer.
fn emit_step(
    tokenizer: &tokenizers::Tokenizer,
    id: u32,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    emitted: &mut Vec<ProbeStep>,
) {
    let piece = tokenizer
        .id_to_token(id)
        .unwrap_or_else(|| format!("<unk:{id}>"));
    let step = ProbeStep {
        token_id: id,
        piece: piece.into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    };
    step_fn(&step);
    emitted.push(step);
}

/// EAGLE-3 speculative-decoding round-loop (greedy / temp=0).
///
/// Port of `_eagle3_rounds` (mlx-vlm), now with correctness fixes:
///
/// 1. **Drafter KV prefill** (`prefill_from_verifier_hidden`): the drafter's
///    KV cache is seeded from verifier hidden states.
///
/// 2. **Per-round `accept_and_reseed`**: after each acceptance walk,
///    the drafter KV cache and hidden seed are updated.
///
/// Reuses the three verifier-side seams: multi-layer hidden capture,
/// GDN snapshot/restore rollback, and raw embed accessor.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub fn eagle3_generate_greedy(
    verifier: &Architecture,
    drafter: &mut Eagle3Drafter,
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
            "eagle3_generate_greedy: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "eagle3_generate_greedy: EAGLE-3 verifier must be the Qwen3.5/3.6-MoE \
             hybrid (needs GDN lin_caches + multi-layer hidden capture)"
                .into(),
        ));
    }

    let aux_layer_ids = drafter.cfg.aux_layer_ids.clone();
    let block_total = requested_block_total.min(drafter.cfg.block_size).max(2);

    // Same constant the verifier resolves — a spec pair must not run two
    // different caches.
    let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
    let max_seq = max_ctx_override.unwrap_or_else(|| {
        let v_mpe = verifier.max_position_embeddings();
        if v_mpe <= 0 || v_mpe > KV_MAX_SEQ_DEFAULT {
            KV_MAX_SEQ_DEFAULT
        } else {
            v_mpe
        }
    });

    let mut v_caches: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| {
            let window = verifier.layer_sliding_window(i);
            KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
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

    // Size the drafter KV cache to the verifier context limit
    // (max_position_embeddings, capped to KV_MAX_SEQ_DEFAULT, or --max-ctx).
    // This matches the verifier caches so the drafter cannot overflow before
    // the verifier does — fixing the prior hardcoded 4096 that crashed once
    // prompt + emitted tokens exceeded it (zero-length slice_update range in
    // update_decode_fp16, broadcast-shape panic).
    drafter.reset(max_seq);

    let mut total_draft = 0usize;
    let mut total_accept = 0usize;
    let mut rounds = 0usize;
    let t_total = Instant::now();
    let mut draft_ns: u128 = 0;
    let mut verifier_ns: u128 = 0;

    let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

    // -- Verifier prefill + round-0 bonus + drafter KV prefill. --
    //
    // Run the verifier on the full prompt with multi-aux hidden capture,
    // chunked to stay within the Metal command-buffer budget.
    // `forward_verify_capture_chunked` splits into windows of PREFILL_CHUNK_SIZE
    // tokens, runs non-final chunks with `forward_hidden_states_multi` (no logit
    // materialisation), and runs only the final chunk with `forward_verify_capture`
    // to obtain last-position logits.
    //
    // Returns `(logits[1,1,vocab], hidden[1,n,3H])`:
    // - `logits` covers only the last prompt position (needed for bonus token).
    // - `hidden` covers ALL n positions (needed for the drafter KV prefill,
    // which mirrors mlx-vlm `prefill_from_target_hidden` with no length cutoff).
    //
    // For short prompts (n ≤ PREFILL_CHUNK_SIZE) this falls through to a single
    // forward pass — numerically identical to the pre-path.
    //
    // `d_seed_tok`: the drafter's precomputed first draft token for the upcoming
    // round (mirrors mlx-vlm `_seed_token`). Always `Some` after
    // `prefill_from_verifier_hidden` and after each `accept_and_reseed`.
    const PREFILL_CHUNK_SIZE: usize = 1024;
    let n = prompt_ids.len();
    tracing::debug!(
        prompt_len = n,
        chunk_size = PREFILL_CHUNK_SIZE,
        "eagle3: verifier prefill (chunked)"
    );
    let (bonus_logits, all_hidden) = verifier.forward_verify_capture_chunked(
        prompt_ids,
        &aux_layer_ids,
        &mut v_caches,
        Some(&mut v_lin),
        PREFILL_CHUNK_SIZE,
        device,
    )?;
    // `bonus_logits` is [1,1,vocab] — the last prompt position only.
    super::guard_verifier_prefill_logits(verifier, &bonus_logits, prompt_ids.len())?;
    let am = argmax(&bonus_logits, -1, device)?;
    am.eval()?;
    let bonus = u32::from_le_bytes(am.to_bytes()?[..4].try_into().unwrap());

    // Drafter prefill: shifted tokens = prompt[1..] + [bonus].
    // Conditioned on verifier hidden at positions 0..n-1 (all_hidden).
    // Mirrors mlx-vlm `prefill_from_target_hidden`:
    // shifted = concat([input_ids[:, 1:], bonus], axis=1) → n tokens
    // hidden = hidden[:, :n, :] → [1, n, 3H]
    let mut shifted: Vec<u32> = Vec::with_capacity(n);
    shifted.extend_from_slice(&prompt_ids[1..]);
    shifted.push(bonus);
    let (h_seed_init, seed_tok_init) =
        drafter.prefill_from_verifier_hidden(verifier, &shifted, &all_hidden, device)?;
    tracing::debug!(
        prompt_len = n,
        drafter_cache_offset = drafter.cache_offset(),
        seed_tok = seed_tok_init,
        "eagle3: drafter KV prefill done"
    );

    let (mut b, mut h_seed, mut d_seed_tok) = (bonus, h_seed_init, Some(seed_tok_init));

    emit_step(tokenizer, b, step_fn, &mut emitted);
    if eos_ids.contains(&b) {
        return Ok(emitted);
    }

    tracing::info!(
        block_size = block_total,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        ?aux_layer_ids,
        draft_vocab_size = drafter.cfg.draft_vocab_size,
        "eagle3_generate_greedy: starting (Qwen3.6-MoE verifier + EAGLE-3 drafter)"
    );

    while emitted.len() < n_tokens {
        rounds += 1;
        let remaining = n_tokens - emitted.len();
        let bs = eagle3_next_block_size(block_total, remaining + 1);
        if bs <= 1 {
            break;
        }

        // Track drafter cache offset before draft_block so accept_and_reseed
        // knows where to roll back to.
        let draft_pre_round_offset = drafter.cache_offset();

        // -- Phase A: drafter proposes bs-1 tokens autoregressively. --
        // Pass the precomputed seed token (from the prior accept_and_reseed /
        // prefill_from_verifier_hidden) so the correction position is not
        // re-processed inside draft_block.
        let t0 = Instant::now();
        let draft_tokens = drafter.draft_block(verifier, b, &h_seed, d_seed_tok, bs)?;
        draft_ns += t0.elapsed().as_nanos();
        if draft_tokens.is_empty() {
            break;
        }
        total_draft += draft_tokens.len();

        // -- Phase B: verifier scores [b, draft...] + captures multi-aux hidden. --
        let round_snap = DFlashRoundState::snapshot(&v_lin)?;
        let mut v_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        v_input.push(b);
        v_input.extend_from_slice(&draft_tokens);
        let v_k = v_input.len();

        // hot-path: restricted-vocab logits for intermediate verifier positions,
        // full-vocab only at the correction position.
        //
        // Mirrors Python `_eagle3_verify_target_hot`:
        // 1. Run forward pass capturing final-normed hidden at all k positions.
        // 2. Compute restricted-vocab logits for ALL k positions (draft_vocab=32000
        // rows vs 248320) via hot_logits_from_final_hidden.
        // 3. Map argmax → target-vocab ids via hot_ids_host for all k positions.
        // 4. Find full_pos = first mismatch between draft_tokens and tokens[0..k-2],
        // defaulting to k-1 (all accepted → correction is the bonus).
        // 5. Compute full-vocab logits for ONLY the single position full_pos via
        // logits_from_hidden on a sliced hidden — never materialise [1,k,vocab].
        // 6. Replace tokens[full_pos] with the full-vocab correction token.
        //
        // This reduces logit materialisation by ~7.8× for accepted positions.
        let t0 = Instant::now();
        let (v_tokens, v_hidden) = if drafter.hot_path_active() {
            let (v_hidden, v_final_hidden) = verifier.forward_verify_capture_hot(
                &v_input,
                v_k,
                &aux_layer_ids,
                &mut v_caches,
                Some(&mut v_lin),
                device,
            )?;
            let hot_ids_arr = drafter
                .hot_ids_arr()
                .expect("hot_path_active but no hot_ids_arr");
            let hot_ids_host = drafter.hot_ids_host();
            let hidden_sz = drafter.hidden_size() as i32;

            // Step 2-3: restricted logits for all k positions → target-vocab tokens.
            let hot_logits =
                verifier.hot_logits_from_final_hidden(&v_final_hidden, hot_ids_arr, device)?;
            let hot_am = argmax(&hot_logits, -1, device)?;
            hot_am.eval()?;
            let hot_bytes = hot_am.to_bytes()?;
            let mut tokens: Vec<u32> = (0..v_k)
                .map(|i| {
                    let draft_idx =
                        u32::from_le_bytes(hot_bytes[i * 4..i * 4 + 4].try_into().unwrap())
                            as usize;
                    hot_ids_host
                        .get(draft_idx)
                        .copied()
                        .unwrap_or(draft_idx as u32)
                })
                .collect();

            // Step 4: find full_pos = first mismatch draft_tokens[i] vs tokens[i],
            // for i in 0..k-1 (draft positions). Default to k-1 (all accepted).
            let n_draft = v_k - 1; // = draft_tokens.len()
            let full_pos = find_full_pos(&draft_tokens, &tokens[..n_draft]);

            // Step 5-6: full-vocab token at full_pos, replace tokens[full_pos].
            let h_corr = v_final_hidden.slice(
                &[0, full_pos as i32, 0],
                &[1, full_pos as i32 + 1, hidden_sz],
                &[1, 1, 1],
                device,
            )?;
            let corr_logits = verifier.logits_from_hidden(&h_corr, device)?;
            let corr_am = argmax(&corr_logits, -1, device)?;
            corr_am.eval()?;
            let corr_bytes = corr_am.to_bytes()?;
            tokens[full_pos] = u32::from_le_bytes(corr_bytes[..4].try_into().unwrap());

            verifier_ns += t0.elapsed().as_nanos();
            (tokens, v_hidden)
        } else {
            let (v_logits, v_hidden) = verifier.forward_verify_capture(
                &v_input,
                v_k,
                &aux_layer_ids,
                &mut v_caches,
                Some(&mut v_lin),
                device,
            )?;
            let v_argmax = argmax(&v_logits, -1, device)?;
            v_argmax.eval()?;
            let vb = v_argmax.to_bytes()?;
            verifier_ns += t0.elapsed().as_nanos();
            let mut toks = Vec::with_capacity(v_k);
            for i in 0..v_k {
                toks.push(u32::from_le_bytes(vb[i * 4..i * 4 + 4].try_into().unwrap()));
            }
            (toks, v_hidden)
        };

        // -- Phase C: greedy acceptance walk. --
        let (accept, new_tokens) = eagle3_walk(&draft_tokens, &v_tokens, remaining);
        total_accept += accept;
        let n_committed = new_tokens.len();

        // Per-step trace: enable with RUST_LOG=rmlx_models::speculative::eagle3=trace.
        if tracing::enabled!(
            target: "rmlx_models::speculative::eagle3",
            tracing::Level::TRACE
        ) {
            let running_ar = if total_draft > 0 {
                (total_accept as f64) / (total_draft as f64)
            } else {
                0.0
            };
            for (i, (&dt, &vt)) in draft_tokens.iter().zip(v_tokens.iter()).enumerate() {
                tracing::trace!(
                    target: "rmlx_models::speculative::eagle3",
                    round = rounds,
                    step = i,
                    draft_tok = dt,
                    verifier_tok = vt,
                    accepted = i < accept,
                    cumulative_accept_rate = running_ar,
                    carry_tok = b,
                    seed_tok = ?d_seed_tok,
                    "eagle3 step"
                );
            }
        }

        // -- Emit accepted prefix + 1 correction/bonus. --
        let mut hit_eos = false;
        for &id in &new_tokens {
            if emitted.len() >= n_tokens {
                break;
            }
            emit_step(tokenizer, id, step_fn, &mut emitted);
            if eos_ids.contains(&id) {
                hit_eos = true;
                break;
            }
        }
        if hit_eos {
            break;
        }

        // -- Phase D: roll back verifier KV/GDN caches on partial accept. --
        // The verifier consumed v_k positions; keep the committed prefix
        // (pre-round + accept + 1 carry rows). Roll FA KV caches back, restore +
        // replay the GDN recurrence.
        let v_offset_before = v_caches.iter().map(|c| c.offset()).max().unwrap_or(0);
        let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);
        if v_target < v_offset_before {
            for c in &mut v_caches {
                if c.offset() >= v_target {
                    c.truncate_to(v_target);
                }
            }
            let v_pre_round_offset = v_offset_before - v_k as i32;
            let v_kept = (v_target - v_pre_round_offset).max(0) as usize;
            round_snap.restore(&mut v_lin);
            if v_kept > 0 && v_kept <= v_input.len() {
                let mut scratch: Vec<KvCache> = (0..verifier.num_hidden_layers())
                    .map(|_| KvCache::with_quant(KvQuant::None))
                    .collect();
                let _ = verifier.forward_seq_last_k_with_cache(
                    &v_input[..v_kept],
                    1,
                    &mut scratch,
                    Some(&mut v_lin),
                    device,
                )?;
            }
        } else {
            drop(round_snap);
        }

        // -- Phase E: drafter accept-and-reseed. --
        //
        // (a) Roll drafter KV cache back to pre-round offset.
        // (b) Re-run drafter forward on accepted draft prefix + correction,
        // conditioned on the verifier's 3-aux hiddens at those positions.
        // (c) Sample the drafter's next-token prediction from the correction
        // position hidden → `d_seed_tok` (mirrors mlx-vlm `_seed_token`).
        // (d) Pass `d_seed_tok` to the next `draft_block` as `precomputed_first_tok`
        // so the correction position is NOT re-processed a second time.
        //
        // `v_input` = [b, draft[0], ..., draft[num_draft-1]].
        // `v_hidden` = [1, v_k, 3H] — verifier hidden for those positions.
        // `accept_and_reseed` tokens: draft[0..accepted] + correction.
        // Hiddens: v_hidden[:, 0..=accepted, :].
        let correction = *new_tokens.last().unwrap_or(&b);
        let (new_h_seed, seed_tok) = drafter.accept_and_reseed(
            verifier,
            draft_pre_round_offset,
            &draft_tokens,
            correction,
            &v_hidden,
            accept,
            device,
        )?;
        h_seed = new_h_seed;
        d_seed_tok = Some(seed_tok);
        b = correction;

        tracing::debug!(
            round = rounds,
            accept,
            num_draft = draft_tokens.len(),
            n_committed,
            emitted_total = emitted.len(),
            v_offset_before,
            v_target,
            draft_pre_round_offset,
            draft_cache_after = drafter.cache_offset(),
            "eagle3 round"
        );
    }

    let elapsed_ms = (t_total.elapsed().as_nanos() as f64) / 1.0e6;
    let accept_rate = if total_draft > 0 {
        (total_accept as f64) / (total_draft as f64)
    } else {
        0.0
    };
    let decode_tps = if elapsed_ms > 0.0 {
        (emitted.len() as f64) / (elapsed_ms / 1000.0)
    } else {
        0.0
    };
    tracing::info!(
        rounds,
        emitted = emitted.len(),
        total_draft,
        total_accept,
        accept_rate,
        decode_tps,
        elapsed_ms,
        draft_ms = (draft_ns as f64) / 1.0e6,
        verifier_ms = (verifier_ns as f64) / 1.0e6,
        block_size = block_total,
        "eagle3_generate_greedy: done"
    );

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

/// Load the EAGLE-3 drafter tensors + config from `draft_dir`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn load_eagle3(
    draft_dir: &Path,
    hidden_size: usize,
    vocab_size: usize,
    eos_token_ids: &[u32],
    device: Device,
) -> Result<Eagle3Drafter> {
    use rmlx_loader::{load_config, load_shard_index, ShardSet};

    let cfg_raw = load_config(draft_dir)
        .map_err(|e| Error::Model(format!("Eagle3Drafter: load_config: {e}")))?;
    let arch = cfg_raw.architectures.first().map_or("", String::as_str);
    if !arch.contains("Eagle3") {
        tracing::warn!(
            draft = %draft_dir.display(),
            %arch,
            "Eagle3Drafter: expected architecture LlamaForCausalLMEagle3; proceeding by tensor names"
        );
    }

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

    // rope_theta lives under rope_parameters.rope_theta for this checkpoint,
    // with a top-level rope_theta fallback.
    let rope_theta = extras
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .map_or_else(|| g_f32("rope_theta", 1.0e7), |v| v as f32);

    // `eagle_aux_hidden_state_layer_ids` are the *target* layer ids; the verifier
    // residual-stream capture happens one layer earlier, mirroring mlx-vlm
    // `Eagle3Config.capture_layer_ids = [max(id - 1, 0) for id in target_layer_ids]`
    // (config.py). rMLX's `forward_verify_capture` appends the residual stream
    // AFTER layer i, so we pass the capture ids (id-1) directly. Capturing at the
    // raw target ids [3,19,35] instead of [2,18,34] feeds the drafter the wrong
    // residual slice and collapses accept-rate.
    let aux_layer_ids: Vec<usize> = extras
        .get("eagle_config")
        .and_then(|d| d.get("eagle_aux_hidden_state_layer_ids"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|v| (v as usize).saturating_sub(1)))
                .collect()
        })
        .unwrap_or_default();

    // block_size: speculators_config.proposal_methods[0].speculative_tokens + 2,
    // else mlx-vlm default 5.
    let block_size = extras
        .get("speculators_config")
        .and_then(|s| s.get("proposal_methods"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("speculative_tokens"))
        .and_then(serde_json::Value::as_u64)
        .map_or(5, |v| v as usize + 2);

    let cfg = Eagle3Config {
        hidden_size: g_usize("hidden_size", 2048),
        num_attention_heads: g_usize("num_attention_heads", 16),
        num_key_value_heads: g_usize("num_key_value_heads", 2),
        head_dim: g_usize("head_dim", 256),
        rms_norm_eps: g_f32("rms_norm_eps", 1e-6),
        rope_theta,
        draft_vocab_size: g_usize("draft_vocab_size", 32000),
        vocab_size: g_usize("vocab_size", 248320),
        aux_layer_ids,
        block_size,
    };

    if cfg.hidden_size != hidden_size {
        return Err(Error::Model(format!(
            "Eagle3Drafter: drafter hidden_size {} != verifier hidden_size {} \
             (wrong draft model?)",
            cfg.hidden_size, hidden_size
        )));
    }
    if cfg.vocab_size != vocab_size {
        return Err(Error::Model(format!(
            "Eagle3Drafter: drafter vocab_size {} != verifier vocab_size {} \
             (d2t-mapped target ids would be invalid)",
            cfg.vocab_size, vocab_size
        )));
    }
    if cfg.aux_layer_ids.is_empty() {
        return Err(Error::Model(
            "Eagle3Drafter: config missing eagle_config.eagle_aux_hidden_state_layer_ids".into(),
        ));
    }

    let idx = load_shard_index(draft_dir)
        .map_err(|e| Error::Model(format!("Eagle3Drafter: shard index: {e}")))?;
    let shards = ShardSet::open(draft_dir, &idx)
        .map_err(|e| Error::Model(format!("Eagle3Drafter: open: {e}")))?;

    let load = |name: &str| -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("Eagle3Drafter: safetensors: {e}")))?;
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
            "Eagle3Drafter: tensor '{name}' not found"
        )))
    };
    // Read d2t as raw i64 bytes -> host Vec<i32> (avoids on-device gather).
    let load_d2t = |name: &str| -> Result<Vec<i32>> {
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("Eagle3Drafter: safetensors: {e}")))?;
            if let Ok(t) = st.tensor(name) {
                let bytes = t.data();
                let out: Vec<i32> = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
                    .collect();
                return Ok(out);
            }
        }
        Err(Error::Model("Eagle3Drafter: tensor 'd2t' not found".into()))
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

    // fc.weight: [H, 3*H] — validates head matches both width and aux count.
    let fc_w = load("fc.weight")?;
    let fc_shape = fc_w.shape().to_vec();
    let expect_in = cfg.aux_layer_ids.len() * cfg.hidden_size;
    if fc_shape.len() != 2
        || fc_shape[0] as usize != cfg.hidden_size
        || fc_shape[1] as usize != expect_in
    {
        return Err(Error::Model(format!(
            "Eagle3Drafter: fc.weight shape {fc_shape:?} != [{}, {}] \
             (hidden_size / aux_layer count mismatch)",
            cfg.hidden_size, expect_in
        )));
    }
    let fc = Linear::Plain { weight: fc_w };

    // Per-aux RMSNorms (`fcs.{0,1,2}`). The speculators-format Dogacel checkpoint
    // applies these to each aux hidden slice before the `fc` concat-projection;
    // enabling them more than doubled the live greedy accept-rate (0.09 -> 0.21)
    // — see module docs. Auto-detected by tensor presence: when all
    // `fcs.{i}.weight` exist we use the per-aux path, else fall back to the raw
    // concat (mlx-vlm / canonical SpecForge layout). `RMLX_EAGLE3_NO_FCS=1`
    // forces the raw-concat fallback for A/B comparison.
    let fcs = if std::env::var("RMLX_EAGLE3_NO_FCS").is_ok() {
        None
    } else {
        let mut v = Vec::with_capacity(cfg.aux_layer_ids.len());
        let mut ok = true;
        for i in 0..cfg.aux_layer_ids.len() {
            if let Ok(n) = norm(&format!("fcs.{i}.weight")) {
                v.push(n)
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            tracing::info!(
                count = v.len(),
                "Eagle3Drafter: per-aux fcs norms active (speculators checkpoint)"
            );
            Some(v)
        } else {
            None
        }
    };

    // lm_head.weight: [draft_vocab, H].
    let lm_head_w = load("lm_head.weight")?;
    let lm_shape = lm_head_w.shape().to_vec();
    if lm_shape.len() != 2
        || lm_shape[0] as usize != cfg.draft_vocab_size
        || lm_shape[1] as usize != cfg.hidden_size
    {
        return Err(Error::Model(format!(
            "Eagle3Drafter: lm_head.weight shape {lm_shape:?} != [{}, {}]",
            cfg.draft_vocab_size, cfg.hidden_size
        )));
    }
    let lm_head = Linear::Plain { weight: lm_head_w };

    let d2t = if cfg.draft_vocab_size == cfg.vocab_size {
        Vec::new()
    } else {
        let d = load_d2t("d2t")?;
        if d.len() != cfg.draft_vocab_size {
            return Err(Error::Model(format!(
                "Eagle3Drafter: d2t length {} != draft_vocab_size {}",
                d.len(),
                cfg.draft_vocab_size
            )));
        }
        d
    };

    // Precompute hot_ids for the restricted-vocab hot-path.
    // hot_ids[i] = i + d2t[i] — the target-vocab token id for each draft position.
    // EOS token ids are appended (after deduplication) so the restricted-vocab
    // argmax can select EOS at intermediate positions — mirrors mlx-vlm's
    // `_eagle3_hot_token_ids` which concatenates `eos_token_ids` onto `hot_ids`.
    // Only meaningful when d2t is present (draft vocab != verifier vocab).
    let (hot_ids_host, hot_ids_arr) = if d2t.is_empty() {
        (Vec::new(), None)
    } else {
        let mut hot: Vec<u32> = d2t
            .iter()
            .enumerate()
            .map(|(i, &offset)| (i as i64 + i64::from(offset)) as u32)
            .collect();
        // Append EOS ids that are not already in hot (linear scan; count is small).
        let pre_len = hot.len();
        for &eos in eos_token_ids {
            if !hot[..pre_len].contains(&eos) {
                hot.push(eos);
            }
        }
        if hot.len() > pre_len {
            tracing::info!(
                n_eos_appended = hot.len() - pre_len,
                eos_token_ids = ?eos_token_ids,
                "Eagle3Drafter: appended EOS ids to hot_ids (restricted-vocab argmax can now select EOS)"
            );
        }
        // Build i32 Array for use as MLX gather indices.
        let hot_i32: Vec<i32> = hot.iter().map(|&v| v as i32).collect();
        let n = hot_i32.len();
        let bytes = unsafe { std::slice::from_raw_parts(hot_i32.as_ptr().cast::<u8>(), n * 4) };
        let arr = Array::from_bytes(bytes, &[n as i32], rmlx_mlx::Dtype::I32)?;
        arr.eval()?;
        (hot, Some(arr))
    };

    if !hot_ids_host.is_empty() {
        tracing::info!(
            draft_vocab_size = cfg.draft_vocab_size,
            vocab_size = cfg.vocab_size,
            "Eagle3Drafter: hot-path active — restricted vocab {}→{} rows",
            cfg.vocab_size,
            cfg.draft_vocab_size,
        );
    }

    Ok(Eagle3Drafter {
        fc,
        fcs,
        hidden_norm: norm("layers.0.hidden_norm.weight")?,
        input_layernorm: norm("layers.0.input_layernorm.weight")?,
        post_attention_layernorm: norm("layers.0.post_attention_layernorm.weight")?,
        q_proj: lin("layers.0.self_attn.q_proj.weight")?,
        k_proj: lin("layers.0.self_attn.k_proj.weight")?,
        v_proj: lin("layers.0.self_attn.v_proj.weight")?,
        o_proj: lin("layers.0.self_attn.o_proj.weight")?,
        mlp: Mlp {
            gate_proj: lin("layers.0.mlp.gate_proj.weight")?,
            up_proj: lin("layers.0.mlp.up_proj.weight")?,
            down_proj: lin("layers.0.mlp.down_proj.weight")?,
            activation: Activation::Silu,
        },
        norm: norm("norm.weight")?,
        lm_head,
        d2t,
        hot_ids_host,
        hot_ids_arr,
        cache: KvCache::with_quant(KvQuant::None),
        cfg,
        device,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
