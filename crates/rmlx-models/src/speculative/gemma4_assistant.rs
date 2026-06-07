// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Gemma4-assistant MTP drafter (, live family).
//!
//! Port of mlx-vlm
//! `mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`
//! (`Gemma4AssistantDraftModel`) + the shared-K/V round-loop in
//! `mlx_vlm/speculative/mtp.py`.
//!
//! # Design - shared-K/V, NOT own-cache
//!
//! Unlike the qwen3.5 `mtp.*` sidecar (its own KV cache, see [`super::mtp`]),
//! the Gemma4 "assistant" drafter is a small full Gemma4 decoder stack (4
//! layers for E2B) that reads the verifier's per-layer-type K/V instead of
//! computing its own. Each drafter layer has a `q_proj` + `q_norm` only - no
//! `k_proj`/`v_proj` - because K/V come from the verifier:
//!
//! - every drafter `sliding_attention` layer attends over the verifier's
//!   last sliding-layer K/V,
//! - the drafter's `full_attention` layer attends over the verifier's last
//!   full-layer K/V.
//!
//! (mlx-vlm `_mtp_shared_kv_from_prompt_cache` keys the shared dict by
//! `layer.layer_type`, last-wins - exposed on the rMLX verifier as
//! [`crate::gemma4::Gemma4Text::forward_hidden_states_shared_kv`].)
//!
//! Per draft step: the drafter embeds the previous token through the target's
//! input embeddings (embed_scale 1.0 for Gemma4), concatenates
//! `[target_embed(tok), last_hidden]` (`2 * backbone_hidden`), projects down to
//! the drafter width via `pre_projection`, runs the 4-layer stack against the
//! shared K/V, applies `norm`, then `post_projection` back up to backbone width
//! (the `last_hidden` carried into the next step). The greedy next token comes
//! from the centroid-routed sparse LM head (`masked_embedding`) over the
//! drafter's hidden (drafter `tie_word_embeddings=true`).
//!
//! # MVP scope (document-the-truth, CLAUDE.md hard rule 7)
//!
//! Unbatched, B=1, greedy (temp=0). Stochastic temp>0 is deferred (the host
//! sampler slots in at the round-loop level - ). Full-attention masks
//! are `None` (B=1 no-padding); sliding-attention layers get an additive
//! bidirectional-window bias (passed via mlx-c SDPA `"array"` mode, matching
//! the verifier SWA path — there is no `"additive"` mode) when the verifier KV
//! exceeds the window. The
//! `position_ids` are held constant across draft steps (`query_offset =
//! kv_offset`), mirroring `draft_block`'s single-position multi-token generator.

#![allow(clippy::cognitive_complexity, clippy::too_many_lines)]
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{
    argmax, argpartition, concatenate, matmul, rope, rope_with_freqs, scaled_dot_product_attention,
    Array, Device, Dtype,
};

use crate::arch::Architecture;
use crate::gemma4::LayerType;
use crate::layers::{Embedding, Linear, Mlp, RmsNorm};

/// One drafter decoder layer (Gemma4 shape, Q-only - K/V are shared).
#[allow(missing_debug_implementations)]
struct DraftLayer {
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    q_proj: Linear,
    q_norm: RmsNorm,
    o_proj: Linear,
    mlp: Mlp,
    layer_scalar: Option<Array>,
    layer_type: LayerType,
    /// Per-layer head_dim: `head_dim` (256) for sliding, `global_head_dim`
    /// (512) for full-attention layers (Gemma4 convention — the full-attn
    /// q_proj/q_norm/o_proj are sized to `global_head_dim`).
    head_dim: usize,
    /// Full-attention ProportionalRoPE freqs (`Some` only on full layers).
    rope_full_freqs: Option<Array>,
}

/// Loaded Gemma4-assistant drafter weights + config.
#[allow(missing_debug_implementations)]
pub struct Gemma4AssistantDrafter {
    /// `model.embed_tokens` (drafter's own; also the tied LM-head weight).
    embed_tokens: Embedding,
    /// `pre_projection`: `Linear(2*backbone_hidden -> draft_hidden)`.
    pre_projection: Linear,
    /// `post_projection`: `Linear(draft_hidden -> backbone_hidden)`.
    post_projection: Linear,
    /// `model.norm` (final RMSNorm before post_projection).
    norm: RmsNorm,
    layers: Vec<DraftLayer>,
    /// `masked_embedding.centroids`: `Linear(draft_hidden -> num_centroids)`.
    centroids: Linear,
    /// `masked_embedding.token_ordering`: `[vocab_size]` I32 (centroid->token).
    token_ordering: Array,
    draft_hidden: usize,
    backbone_hidden: usize,
    n_heads: usize,
    sliding_window: i32,
    rope_sliding_theta: f32,
    num_centroids: usize,
    centroid_top_k: usize,
    vocab_per_centroid: usize,
    device: Device,
}

impl Gemma4AssistantDrafter {
    /// Backbone (verifier) hidden width this drafter was built for.
    pub fn backbone_hidden(&self) -> usize {
        self.backbone_hidden
    }

    /// Load a Gemma4-assistant drafter from `draft_dir`, validate vs verifier.
    pub fn load(draft_dir: &Path, backbone_hidden: usize, device: Device) -> Result<Self> {
        load_assistant(draft_dir, backbone_hidden, device)
    }

    /// Autoregressive K-step greedy draft (port of `draft_block`).
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn draft_n(
        &self,
        verifier: &Architecture,
        last_token: u32,
        hidden: &Array,
        sliding_kv: (&Array, &Array),
        full_kv: (&Array, &Array),
        kv_offset: i32,
        block_size: usize,
    ) -> Result<Vec<u32>> {
        if block_size <= 1 {
            return Ok(vec![]);
        }
        let kv_len_sliding = sliding_kv.0.shape()[2];
        let swa_mask = self.build_swa_mask(kv_offset, 1, kv_len_sliding)?;

        // numeric-diff: trace round-0 inputs so --log verbose / RUST_LOG=rmlx=trace
        // surfaces them without requiring an fs dump directory.
        tracing::trace!(
            target: "rmlx::mtp",
            last_token,
            kv_offset,
            hidden_shape = ?hidden.shape(),
            sliding_k_shape = ?sliding_kv.0.shape(),
            sliding_v_shape = ?sliding_kv.1.shape(),
            full_k_shape = ?full_kv.0.shape(),
            full_v_shape = ?full_kv.1.shape(),
            "mtp_draft round-0 inputs"
        );

        let mut tok = last_token;
        let mut h_prev = hidden.try_clone()?; // [1,1,backbone_hidden]
        let mut tokens: Vec<u32> = Vec::with_capacity(block_size - 1);

        while tokens.len() < block_size - 1 {
            let step = tokens.len();
            let tok_embed = verifier.embed_token_raw(tok, self.device)?; // [1,1,backbone]
            let cat = concatenate(&[&tok_embed, &h_prev], -1, self.device)?; // [1,1,2*backbone]
            let h = self.pre_projection.forward(&cat, self.device)?; // [1,1,draft_hidden]

            let mut h = h;
            for (li, layer) in self.layers.iter().enumerate() {
                let (sk, sv, mask) = match layer.layer_type {
                    LayerType::SlidingAttention => (sliding_kv.0, sliding_kv.1, swa_mask.as_ref()),
                    LayerType::FullAttention => (full_kv.0, full_kv.1, None),
                };
                h = self.layer_forward(layer, &h, sk, sv, mask, kv_offset)?;
                tracing::trace!(
                    target: "rmlx::mtp",
                    step,
                    li,
                    out_shape = ?h.shape(),
                    "mtp_draft layer out"
                );
            }
            let h = self.norm.forward(&h, self.device)?; // [1,1,draft_hidden]
            h_prev = self.post_projection.forward(&h, self.device)?; // [1,1,backbone]

            let next = self.masked_argmax(&h)?;
            tracing::trace!(
                target: "rmlx::mtp",
                step,
                argmax = next,
                "mtp_draft step"
            );
            tokens.push(next);
            tok = next;
        }
        Ok(tokens)
    }

    /// One drafter decoder layer over shared K/V. `x`: `[1, 1, draft_hidden]`.
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn layer_forward(
        &self,
        layer: &DraftLayer,
        x: &Array,
        sk: &Array,
        sv: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = layer.input_norm.forward(x, self.device)?;

        let hd = layer.head_dim as i32;
        let q = layer.q_proj.forward(&h, self.device)?;
        let q = q.reshape(&[1, 1, self.n_heads as i32, hd], self.device)?;
        let q = layer.q_norm.forward(&q, self.device)?;
        let q = q.transpose(&[0, 2, 1, 3], self.device)?; // [1,H,1,D]
        let q = match layer.layer_type {
            LayerType::SlidingAttention => rope(
                &q,
                hd,
                false,
                self.rope_sliding_theta,
                1.0,
                offset,
                self.device,
            )?,
            LayerType::FullAttention => {
                let freqs = layer
                    .rope_full_freqs
                    .as_ref()
                    .expect("full-attn drafter layer missing proportional freqs");
                rope_with_freqs(&q, hd, false, 1.0, offset, freqs, self.device)?
            }
        };

        // mlx-c SDPA accepts only "causal" / "array" / "" for mask_mode; the
        // additive window-bias mask is supplied via "array" with the bias as
        // mask_arr (mirrors the verifier SWA path in gemma4::layers, NOT a
        // distinct "additive" mode string — that value is rejected by the
        // Metal kernel, see fast_ops::scaled_dot_product_attention docs).
        let (mode, mask_arr) = swa_sdpa_mode(mask);
        let attn = scaled_dot_product_attention(&q, sk, sv, 1.0, mode, mask_arr, self.device)?;
        let attn = attn.transpose(&[0, 2, 1, 3], self.device)?;
        let attn = attn.reshape(&[1, 1, (self.n_heads * layer.head_dim) as i32], self.device)?;
        let attn = layer.o_proj.forward(&attn, self.device)?;
        let attn = layer.post_attn_norm.forward(&attn, self.device)?;
        let h = rmlx_mlx::add(&residual, &attn, self.device)?;

        let residual = h.try_clone()?;
        let f = layer.pre_ffn_norm.forward(&h, self.device)?;
        let f = layer.mlp.forward(&f, self.device)?;
        let f = layer.post_ffn_norm.forward(&f, self.device)?;
        let h = rmlx_mlx::add(&residual, &f, self.device)?;

        if let Some(scalar) = &layer.layer_scalar {
            rmlx_mlx::multiply(&h, scalar, self.device)
        } else {
            Ok(h)
        }
    }

    /// Greedy token via the centroid-routed sparse LM head (`MaskedEmbedder`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn masked_argmax(&self, h: &Array) -> Result<u32> {
        let cscore = self.centroids.forward(h, self.device)?; // [1,1,num_centroids]
        let nc = self.num_centroids as i32;
        let kth = nc - self.centroid_top_k as i32;
        let part = argpartition(&cscore, kth, -1, self.device)?;
        let topk_idx = part.slice(&[0, 0, kth], &[1, 1, nc], &[1, 1, 1], self.device)?;
        topk_idx.eval()?;
        let topk: Vec<i32> = bytes_to_i32(&topk_idx)?;

        let vpc = self.vocab_per_centroid as i32;
        let ordering = self.token_ordering.reshape(&[nc, vpc], self.device)?;
        let topk_arr = i32_array(&topk)?;
        let sel = ordering.take(&topk_arr, 0, self.device)?; // [top_k, vpc]
        sel.eval()?;
        let cand: Vec<i32> = bytes_to_i32(&sel)?;

        let cand_arr = i32_array(&cand)?;
        let emb_w = self.embed_tokens_weight()?; // [vocab, draft_hidden]
        let sel_emb = emb_w.take(&cand_arr, 0, self.device)?; // [N, draft_hidden]
        let h_flat = h.reshape(&[1, self.draft_hidden as i32], self.device)?;
        let logits = matmul(
            &h_flat,
            &sel_emb.transpose(&[1, 0], self.device)?,
            self.device,
        )?; // [1, N]
        let best = argmax(&logits, -1, self.device)?;
        best.eval()?;
        let best_i = bytes_to_i32(&best)?[0] as usize;
        Ok(cand[best_i] as u32)
    }

    fn embed_tokens_weight(&self) -> Result<Array> {
        match &self.embed_tokens {
            Embedding::Plain { weight } => weight.try_clone(),
            Embedding::Quantized { .. } => Err(Error::Model(
                "Gemma4AssistantDrafter: quantized embed_tokens unsupported (bf16 sidecar)".into(),
            )),
        }
    }

    /// Additive bidirectional sliding-window mask for the draft block, or `None`
    /// when the whole KV fits every query's window. `[1,1,query_len,kv_len]` f32.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn build_swa_mask(&self, kv_offset: i32, query_len: i32, kv_len: i32) -> Result<Option<Array>> {
        let window = self.sliding_window;
        let q_off = kv_offset.min(kv_len);
        if kv_len <= window && q_off < window && (kv_len - (q_off + query_len)) < window {
            return Ok(None);
        }
        // Masked cells use -1e30 (not f32::NEG_INFINITY) to match the verifier
        // SWA mask builders (crate::layers::build_swa_*): a fully-masked softmax
        // row over -inf yields NaN, whereas a large finite penalty stays well-
        // conditioned. The bias is converted to Bf16 below so its dtype promotes
        // with the bf16 Q/K/V (mlx-c SDPA "array" mode requires this).
        let mut bias = vec![0.0f32; (query_len * kv_len) as usize];
        for qi in 0..query_len {
            let q = q_off + qi;
            for ki in 0..kv_len {
                let dist = q - ki;
                if !(dist > -window && dist < window) {
                    bias[(qi * kv_len + ki) as usize] = -1e30;
                }
            }
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(bias.as_ptr().cast::<u8>(), bias.len() * 4) };
        let arr = Array::from_bytes(bytes, &[1, 1, query_len, kv_len], Dtype::F32)?;
        Ok(Some(arr.astype(Dtype::Bf16, self.device)?))
    }
}

/// Select the mlx-c SDPA `(mask_mode, mask_arr)` for a drafter attention layer.
///
/// The additive window-bias mask (when present) goes through the `"array"`
/// mode — mlx-c rejects a bespoke `"additive"` mode string (valid values are
/// `"causal"`, `"array"`, `""`). `None` (full-attention or window-covers-all)
/// uses the empty mode. Verifier-faithful: see `gemma4::layers::build_attn_mask`.
fn swa_sdpa_mode(mask: Option<&Array>) -> (&'static str, Option<&Array>) {
    match mask {
        Some(m) => ("array", Some(m)),
        None => ("", None),
    }
}

fn i32_array(v: &[i32]) -> Result<Array> {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
    Array::from_bytes(bytes, &[v.len() as i32], Dtype::I32)
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn bytes_to_i32(a: &Array) -> Result<Vec<i32>> {
    let b = a.to_bytes()?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Load + validate the Gemma4-assistant sidecar.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn load_assistant(
    draft_dir: &Path,
    backbone_hidden: usize,
    device: Device,
) -> Result<Gemma4AssistantDrafter> {
    use rmlx_loader::{load_config, load_shard_index, ShardSet};

    let cfg = load_config(draft_dir)
        .map_err(|e| Error::Model(format!("Gemma4Assistant: load_config: {e}")))?;
    let backbone_hidden_size = cfg
        .extras
        .get("backbone_hidden_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(backbone_hidden as u64) as usize;
    if backbone_hidden_size != backbone_hidden {
        return Err(Error::Model(format!(
            "Gemma4Assistant: backbone_hidden_size {backbone_hidden_size} != verifier hidden {backbone_hidden}"
        )));
    }
    let num_centroids = cfg
        .extras
        .get("num_centroids")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(2048) as usize;
    let centroid_top_k = cfg
        .extras
        .get("centroid_intermediate_top_k")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32) as usize;

    let tc = cfg
        .text_config
        .as_ref()
        .ok_or_else(|| Error::Model("Gemma4Assistant: missing text_config".into()))?;
    let te = &tc.extras;
    // `g`/`gf` read from `extras`; some keys (hidden_size, num_hidden_layers,
    // num_attention_heads, sliding_window, layer_types) are *typed* fields on
    // `TextConfig` and must be read from `tc.*` (not extras — they are not
    // round-tripped there). Mixing the two was a load bug: the full-attention
    // layer's head_dim silently fell back to the sliding default.
    let g = |k: &str, d: u64| te.get(k).and_then(serde_json::Value::as_u64).unwrap_or(d) as usize;
    let gf = |k: &str, d: f64| te.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d) as f32;
    let draft_hidden = tc.hidden_size.unwrap_or(256) as usize;
    let n_layers = tc.num_hidden_layers.unwrap_or(4) as usize;
    let n_heads = tc.num_attention_heads.unwrap_or(4) as usize;
    let head_dim = g("head_dim", 256);
    let global_head_dim = g("global_head_dim", 512);
    let sliding_window = tc.sliding_window.unwrap_or(512) as i32;
    let vocab_size = g("vocab_size", 262144);
    let rms_eps = gf("rms_norm_eps", 1e-6);
    let vocab_per_centroid = vocab_size / num_centroids;

    let rope_params = te.get("rope_parameters");
    let rope_sliding_theta = rope_params
        .and_then(|r| r.get("sliding_attention"))
        .and_then(|s| s.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10000.0) as f32;
    let (rope_full_theta, partial) =
        rope_params
            .and_then(|r| r.get("full_attention"))
            .map_or((1e6, 0.25), |f| {
                (
                    f.get("rope_theta")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(1e6) as f32,
                    f.get("partial_rotary_factor")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.25) as f32,
                )
            });
    let rope_full_dims = ((partial * global_head_dim as f32) as i32) & !1;

    let layer_types: Vec<LayerType> = tc.layer_types.as_ref().map_or_else(
        || vec![LayerType::SlidingAttention; n_layers],
        |a| {
            a.iter()
                .map(|s| match s.as_str() {
                    "full_attention" => LayerType::FullAttention,
                    _ => LayerType::SlidingAttention,
                })
                .collect()
        },
    );

    let idx = load_shard_index(draft_dir)
        .map_err(|e| Error::Model(format!("Gemma4Assistant: shard index: {e}")))?;
    let shards = ShardSet::open(draft_dir, &idx)
        .map_err(|e| Error::Model(format!("Gemma4Assistant: open: {e}")))?;

    let load = |name: &str| -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("Gemma4Assistant: safetensors: {e}")))?;
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
            "Gemma4Assistant: tensor '{name}' not found"
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
            eps: rms_eps,
        })
    };

    let embed_tokens = Embedding::Plain {
        weight: load("model.embed_tokens.weight")?,
    };
    let pre_projection = lin("pre_projection.weight")?;
    let post_projection = lin("post_projection.weight")?;
    let final_norm = norm("model.norm.weight")?;
    let centroids = lin("masked_embedding.centroids.weight")?;
    // `masked_embedding.token_ordering` is stored as int64 in the checkpoint;
    // MLX safetensors loading rejects I64, so read the raw little-endian i64
    // bytes and narrow to i32 on the host (token ids fit in i32 — vocab 262144).
    let token_ordering = {
        let mut found: Option<Array> = None;
        for (_, handle) in shards.iter() {
            let st = handle
                .safetensors()
                .map_err(|e| Error::Model(format!("Gemma4Assistant: safetensors: {e}")))?;
            if let Ok(t) = st.tensor("masked_embedding.token_ordering") {
                let raw = t.data();
                let n = raw.len() / 8;
                let mut ids = Vec::with_capacity(n);
                for c in raw.chunks_exact(8) {
                    let v = i64::from_le_bytes(c.try_into().unwrap());
                    ids.push(v as i32);
                }
                let bytes =
                    unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4) };
                found = Some(Array::from_bytes(bytes, &[n as i32], Dtype::I32)?);
                break;
            }
        }
        found.ok_or_else(|| {
            Error::Model(
                "Gemma4Assistant: tensor 'masked_embedding.token_ordering' not found".into(),
            )
        })?
    };

    let rope_full_freqs = crate::gemma4::build_proportional_rope_freqs(
        global_head_dim,
        rope_full_dims as usize,
        rope_full_theta,
    )?;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let p = format!("model.layers.{i}");
        let lt = layer_types
            .get(i)
            .copied()
            .unwrap_or(LayerType::SlidingAttention);
        let layer_scalar = load(&format!("{p}.layer_scalar")).ok();
        layers.push(DraftLayer {
            input_norm: norm(&format!("{p}.input_layernorm.weight"))?,
            post_attn_norm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
            pre_ffn_norm: norm(&format!("{p}.pre_feedforward_layernorm.weight"))?,
            post_ffn_norm: norm(&format!("{p}.post_feedforward_layernorm.weight"))?,
            q_proj: lin(&format!("{p}.self_attn.q_proj.weight"))?,
            q_norm: norm(&format!("{p}.self_attn.q_norm.weight"))?,
            o_proj: lin(&format!("{p}.self_attn.o_proj.weight"))?,
            mlp: Mlp {
                gate_proj: lin(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: lin(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: lin(&format!("{p}.mlp.down_proj.weight"))?,
                activation: crate::layers::Activation::GeluTanh,
            },
            layer_scalar,
            layer_type: lt,
            // Gemma4: full-attention layers use global_head_dim (512), sliding
            // use head_dim (256). q_proj/q_norm/o_proj are sized accordingly.
            head_dim: match lt {
                LayerType::FullAttention => global_head_dim,
                LayerType::SlidingAttention => head_dim,
            },
            rope_full_freqs: match lt {
                LayerType::FullAttention => Some(rope_full_freqs.try_clone()?),
                LayerType::SlidingAttention => None,
            },
        });
    }

    tracing::info!(
        draft = %draft_dir.display(),
        draft_hidden,
        backbone_hidden,
        n_layers,
        num_centroids,
        "Gemma4AssistantDrafter: loaded sidecar"
    );

    Ok(Gemma4AssistantDrafter {
        embed_tokens,
        pre_projection,
        post_projection,
        norm: final_norm,
        layers,
        centroids,
        token_ordering,
        draft_hidden,
        backbone_hidden,
        n_heads,
        sliding_window,
        rope_sliding_theta,
        num_centroids,
        centroid_top_k,
        vocab_per_centroid,
        device,
    })
}

// ---------------------------------------------------------------------------
// Round-loop (greedy MTP, port of _mtp_rounds + _mtp_acceptance_walk)
// ---------------------------------------------------------------------------

use crate::gemma4::ProbeStep;
use crate::kv_cache::{KvCacheBuilder, ResolverSignals};
use rmlx_kv_quant::{KvCache, KvQuant, KV_MAX_SEQ_DEFAULT};
use std::time::Instant;

/// Greedy Gemma4-assistant MTP speculative generation.
///
/// Mirrors mlx-vlm `_mtp_rounds` (greedy / temp=0):
/// 1. prefill verifier on `prompt[..-1]`, advancing its persistent KV cache;
/// 2. round 0 seed `b` = argmax of the verifier logit after the last prompt
/// 3. each round: drafter proposes `block-1` tokens conditioned on the verifier
///    hidden states; verifier accepts greedily.
///
/// `step_fn` is invoked once per emitted (verifier-confirmed) token. Returns the
/// emitted `ProbeStep`s. `block_size` is the MTP block (draft proposes
/// `block_size - 1` tokens/round).
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub fn mtp_assistant_generate_greedy(
    verifier: &Architecture,
    drafter: &Gemma4AssistantDrafter,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    block_size: usize,
    kv_quant_override: Option<KvQuant>,
    max_ctx_override: Option<i32>,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "mtp_assistant_generate_greedy: prompt must have >=2 tokens".into(),
        ));
    }
    if block_size < 2 {
        return Err(Error::Model(
            "mtp_assistant_generate_greedy: block_size must be >= 2".into(),
        ));
    }
    let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

    // Migrated from for_arch_default (deprecated) to resolve_default.
    // Signals::default() → hidden_size=None → falls through to K8V8 (Gemma4 unknown-size arm).
    let kv_quant = kv_quant_override.unwrap_or_else(|| {
        KvCacheBuilder::resolve_default(
            "Gemma4ForConditionalGeneration",
            ResolverSignals::default(),
        )
    });
    let max_seq = max_ctx_override.unwrap_or_else(|| {
        let v_mpe = verifier.max_position_embeddings();
        if v_mpe <= 0 || v_mpe > KV_MAX_SEQ_DEFAULT {
            KV_MAX_SEQ_DEFAULT
        } else {
            v_mpe
        }
    });

    let mut caches: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| {
            let window = verifier.layer_sliding_window(i);
            KvCache::with_quant_max_seq_window(kv_quant, max_seq, window).with_layer_idx(i)
        })
        .collect();

    // Diagnostics.
    let mut total_draft: usize = 0;
    let mut total_accept: usize = 0;
    let mut rounds: usize = 0;
    let t_total = Instant::now();
    let mut draft_ns: u128 = 0;
    let mut verifier_ns: u128 = 0;

    // --- Prefill on prompt[..-1]; last token is round-0 carry. ----------
    let prefill_slice = &prompt_ids[..prompt_ids.len() - 1];
    super::prefill_chunked(verifier, prefill_slice, &mut caches, None, device)?;

    // Round-0 seed `b`: feed the last prompt token through the verifier (1-token
    // forward), capture hidden + shared K/V, and argmax for the first bonus.
    let last_prompt = *prompt_ids.last().unwrap();
    let (hidden_raw, mut sliding_kv, mut full_kv, mut kv_offset) =
        verifier.forward_hidden_states_shared_kv(&[last_prompt], 1, &mut caches, device)?;
    let mut b = {
        let logits = verifier.logits_from_hidden(&hidden_raw, device)?;
        let am = argmax(&logits, -1, device)?;
        am.eval()?;
        u32::from_le_bytes(am.to_bytes()?[..4].try_into().unwrap())
    };
    // MTP drafter conditions on the *normed* trunk hidden
    // (speculative_draft_hidden = model.norm(h)), not the raw pre-norm hidden.
    let mut hidden = verifier.apply_final_norm(&hidden_raw, device)?;

    // Emit the first bonus.
    {
        let piece = tokenizer
            .id_to_token(b)
            .unwrap_or_else(|| format!("<unk:{b}>"));
        let step = ProbeStep {
            token_id: b,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        };
        step_fn(&step);
        emitted.push(step);
        if eos_ids.contains(&b) {
            return Ok(emitted);
        }
    }

    tracing::info!(
        block_size,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        "mtp_assistant_generate_greedy: starting (Gemma4-assistant MTP)"
    );

    while emitted.len() < n_tokens {
        rounds += 1;
        let remaining = n_tokens - emitted.len();
        let bs = (remaining + 1).min(block_size).max(2);

        // -- Phase A: drafter proposes bs-1 tokens (conditioned on hidden). --
        let t0 = Instant::now();
        let draft_tokens = drafter.draft_n(
            verifier,
            b,
            &hidden,
            (&sliding_kv.0, &sliding_kv.1),
            (&full_kv.0, &full_kv.1),
            kv_offset,
            bs,
        )?;
        draft_ns += t0.elapsed().as_nanos();
        if draft_tokens.is_empty() {
            break;
        }
        total_draft += draft_tokens.len();

        // -- Phase B: verifier scores [b, draft...] in one cached forward. ---
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        verify_input.push(b);
        verify_input.extend_from_slice(&draft_tokens);
        let v_k = verify_input.len();

        let t0 = Instant::now();
        let (v_hidden, new_sliding, new_full, _off) =
            verifier.forward_hidden_states_shared_kv(&verify_input, v_k, &mut caches, device)?;
        // Greedy verifier tokens from the K+1 hidden positions.
        let v_logits = verifier.logits_from_hidden(&v_hidden, device)?;
        let v_argmax = argmax(&v_logits, -1, device)?;
        v_argmax.eval()?;
        let vb = v_argmax.to_bytes()?;
        verifier_ns += t0.elapsed().as_nanos();
        let mut v_tokens: Vec<u32> = Vec::with_capacity(v_k);
        for i in 0..v_k {
            v_tokens.push(u32::from_le_bytes(vb[i * 4..i * 4 + 4].try_into().unwrap()));
        }

        // -- Phase C: greedy acceptance walk. --------------------------------
        // v_tokens[i] = verifier prediction after verify_input[i].
        // Compare v_tokens[i] vs draft_tokens[i] for i in 0..bs-1.
        let mut accept = 0usize;
        for i in 0..draft_tokens.len() {
            if v_tokens[i] == draft_tokens[i] {
                accept += 1;
            } else {
                break;
            }
        }
        total_accept += accept;
        // Emit accepted prefix + 1 correction/bonus: v_tokens[0..=accept].
        let to_emit = (accept + 1).min(v_tokens.len());
        let mut hit_eos = false;
        for &id in v_tokens.iter().take(to_emit) {
            if emitted.len() >= n_tokens {
                break;
            }
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
            if eos_ids.contains(&id) {
                hit_eos = true;
                break;
            }
        }
        if hit_eos {
            break;
        }

        // -- Phase D: rollback + next-round setup. ---------------------------
        // The verifier consumed v_k positions; valid prefix = prev + accept + 1
        // (the correction v_tokens[accept] is a prediction, not yet processed).
        let v_offset_before = caches[0].offset();
        let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);
        for c in &mut caches {
            // KV-shared layers read another layer's K/V and never advance their
            // own cache (offset stays 0); truncating them would assert n>offset.
            // Only roll back caches that actually accumulated this round's keys.
            if c.offset() >= v_target {
                c.truncate_to(v_target);
            }
        }

        // Next hidden = verifier penultimate at the accepted position, then
        // final-normed (speculative_draft_hidden) for the drafter conditioning.
        let h = verifier.hidden_size() as i32;
        let hidden_slice = v_hidden.slice(
            &[0, accept as i32, 0],
            &[1, accept as i32 + 1, h],
            &[1, 1, 1],
            device,
        )?;
        let hidden_slice = hidden_slice.reshape(&[1, 1, h], device)?;
        hidden = verifier.apply_final_norm(&hidden_slice, device)?;
        b = v_tokens[accept];

        // Shared K/V for the next round: re-read from the just-advanced cache.
        // On full accept the verify-call K/V is valid as-is; on partial accept
        // we truncated, so the verify-call K/V tail is stale — re-derive cheaply
        // by re-running a zero-length probe is overkill; instead the shared K/V
        // is the verifier's last-layer accumulated K/V which the truncate_to
        // already trimmed in the cache. The verify call returned the *pre-trim*
        // K/V; slice it to v_target length to match.
        let kv_keep = v_target; // absolute length of valid keys after trim
        sliding_kv = slice_kv_len(&new_sliding, kv_keep, device)?;
        full_kv = slice_kv_len(&new_full, kv_keep, device)?;
        kv_offset = v_target;

        tracing::debug!(
            round = rounds,
            accept,
            num_draft = draft_tokens.len(),
            emitted_total = emitted.len(),
            v_offset_before,
            v_target,
            "mtp assistant round"
        );
    }

    let elapsed_ms = (t_total.elapsed().as_nanos() as f64) / 1.0e6;
    tracing::info!(
        rounds,
        emitted = emitted.len(),
        total_draft,
        total_accept,
        accept_rate = if total_draft > 0 {
            (total_accept as f64) / (total_draft as f64)
        } else {
            0.0
        },
        elapsed_ms,
        draft_ms = (draft_ns as f64) / 1.0e6,
        verifier_ms = (verifier_ns as f64) / 1.0e6,
        block_size,
        "mtp_assistant_generate_greedy: done"
    );
    Ok(emitted)
}

/// Slice a shared `(K, V)` pair to the first `keep` key positions (axis 2).
/// `[1, n_kv, kv, hd] -> [1, n_kv, keep, hd]`. No-op when already <= keep.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn slice_kv_len(kv: &(Array, Array), keep: i32, device: Device) -> Result<(Array, Array)> {
    let do_slice = |a: &Array| -> Result<Array> {
        let s = a.shape();
        if s[2] <= keep {
            return a.try_clone();
        }
        a.slice(
            &[0, 0, 0, 0],
            &[s[0], s[1], keep, s[3]],
            &[1, 1, 1, 1],
            device,
        )
    };
    Ok((do_slice(&kv.0)?, do_slice(&kv.1)?))
}

#[cfg(test)]
#[path = "gemma4_assistant_tests.rs"]
mod tests;
