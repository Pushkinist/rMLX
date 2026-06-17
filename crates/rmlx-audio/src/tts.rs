// LOC-exempt: Qwen3-TTS full synthesis pipeline.
// All components (Qwen3 talker + CodePredictor + codec decoder) form one
// tightly coupled generation loop; splitting would obscure the data flow.
// Estimated LOC: ~1800. Justified by architecture complexity.

//! Qwen3-TTS speech synthesis pipeline — Phase 4b full implementation.
//!
//! Synthesizes mono 24 kHz PCM from text via two stages:
//!
//! **Stage 1 — Talker** (`talker.*` weights, `mlx-community__Qwen3-TTS-*-CustomVoice-8bit`):
//! - 28-layer Qwen3 transformer with MRoPE, per-head q/k RMSNorm, affine-8bit weights
//! - `text_projection` (fc1/fc2): projects text embeddings into talker hidden space
//! - `codec_head`: affine-8bit LM head over audio token vocabulary
//! - `code_predictor`: 5-layer mini-Qwen3, generates 16 codec groups per step
//!
//! **Stage 2 — Codec decoder** (`decoder.*` weights, `Qwen__Qwen3-TTS-Tokenizer-12Hz`):
//! - SplitRVQ: 16 codebooks (1 semantic + 15 acoustic), each 2048×256, projected to 512
//! - Pre-conv (k=3, 512→1024) + 8-layer pre-transformer (hidden=512, out=1024)
//! - 2× ConvNeXt upsample (stride=2 each)
//! - 4× ResNet decoder group (strides 8,5,4,3) with SnakeBeta activations
//! - Final conv + tanh → 24 kHz mono f32

#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(
    clippy::struct_field_names,
    reason = "model fields mirror Python reference names (q_proj, k_proj, etc.)"
)]
#![allow(
    clippy::indexing_slicing,
    reason = "array shapes are validated by the model loader; indexing panics on malformed \
              weights, which is an acceptable abort condition"
)]

use std::path::Path;
use std::time::Instant;

use rmlx_loader::{load_shard_index, view, ShardSet};
use rmlx_mlx::{
    add, argmax, broadcast_to, concatenate, conv1d, conv_transpose1d, cos, divide, exp, gelu,
    matmul, maximum, multiply, negative, quantized_matmul, rms_norm, scalar_f32,
    scaled_dot_product_attention, silu, sin, sqrt, subtract, sum_axis_keepdims, tanh, Array,
    Device,
};
use thiserror::Error;
use tracing::{debug, info, instrument};

// ── Error ─────────────────────────────────────────────────────────────────────

/// TTS pipeline errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum TtsError {
    /// Model / config I/O error.
    #[error("model load error: {0}")]
    Load(String),
    /// MLX operation failure.
    #[error("MLX error: {0}")]
    Mlx(String),
    /// Unknown voice name.
    #[error("unknown voice '{0}'; valid voices: serena, vivian, ryan, aiden, eric, dylan, ono_anna, sohee, uncle_fu")]
    UnknownVoice(String),
    /// Tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Empty input.
    #[error("empty text input")]
    Empty,
}

impl From<rmlx_core::error::Error> for TtsError {
    fn from(e: rmlx_core::error::Error) -> Self {
        Self::Mlx(e.to_string())
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Parsed Qwen3-TTS `config.json` top-level fields.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(
    clippy::exhaustive_structs,
    reason = "TtsConfig is a fixed JSON schema from config.json"
)]
pub struct TtsConfig {
    /// `model_type` from `config.json` (should be `"qwen3_tts"`).
    #[serde(default)]
    pub model_type: String,
    /// BOS token id for TTS generation.
    #[serde(default)]
    pub tts_bos_token_id: u32,
    /// EOS token id for TTS generation.
    #[serde(default)]
    pub tts_eos_token_id: u32,
    /// PAD token id for TTS generation.
    #[serde(default)]
    pub tts_pad_token_id: u32,
    /// Talker model configuration sub-block.
    pub talker_config: TalkerConfig,
}

/// Talker sub-configuration from `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(
    clippy::exhaustive_structs,
    reason = "TalkerConfig is a fixed JSON schema from config.json"
)]
pub struct TalkerConfig {
    /// Talker transformer hidden dimension (default 2048).
    pub hidden_size: usize,
    /// Number of attention heads in the talker.
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA).
    pub num_key_value_heads: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of codec groups per step (16 for Qwen3-TTS).
    pub num_code_groups: usize,
    /// Audio BOS token id injected before codec generation.
    pub codec_bos_id: u32,
    /// Audio EOS token id that terminates codec generation.
    pub codec_eos_token_id: u32,
    /// Token id representing no-think in the codec stream.
    #[serde(default)]
    pub codec_nothink_id: u32,
    /// Codec think BOS token id.
    #[serde(default)]
    pub codec_think_bos_id: u32,
    /// Codec think EOS token id.
    #[serde(default)]
    pub codec_think_eos_id: u32,
    /// Codec PAD token id.
    #[serde(default)]
    pub codec_pad_id: u32,
    /// Mapping from language name to language token id.
    #[serde(default)]
    pub codec_language_id: std::collections::HashMap<String, u32>,
    /// Mapping from speaker name to speaker token id.
    #[serde(default)]
    pub spk_id: std::collections::HashMap<String, u32>,
    /// Attention head dimension (default 128).
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    /// RoPE base theta.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    /// Text vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Intermediate size for talker MLP (gate/up). Default = 4 * hidden_size.
    #[serde(default)]
    pub intermediate_size: usize,
    /// Intermediate size for CodePredictor MLP.
    #[serde(default)]
    pub code_predictor_intermediate_size: usize,
    /// Hidden size of the CodePredictor (default 1024).
    #[serde(default = "default_cp_hidden")]
    pub code_predictor_hidden_size: usize,
    /// Number of CodePredictor layers (default 5).
    #[serde(default = "default_cp_layers")]
    pub code_predictor_num_hidden_layers: usize,
}

fn default_head_dim() -> usize {
    128
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rms_eps() -> f32 {
    1e-6
}
fn default_vocab_size() -> usize {
    152064
}
fn default_cp_hidden() -> usize {
    1024
}
fn default_cp_layers() -> usize {
    5
}

impl TtsConfig {
    /// Load from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, TtsError> {
        serde_json::from_str(s).map_err(|e| TtsError::Load(e.to_string()))
    }

    /// Return the speaker token id for `voice`, or `None` if unknown.
    pub fn speaker_id(&self, voice: &str) -> Option<u32> {
        self.talker_config.spk_id.get(voice).copied()
    }
}

// ── Internal layer types ──────────────────────────────────────────────────────

/// Plain (non-quantized) linear weight + optional bias. `w` shape: `[out, in]`.
struct Plain {
    w: Array,
    b: Option<Array>,
}

/// Affine-8bit quantized linear weight.
struct Quant {
    w: Array,
    scales: Array,
    biases: Array,
    /// Optional additive bias (plain `nn.Linear(bias=True)` term, separate from
    /// quantization `biases`).  Present in `text_projection.linear_fc1/fc2` and
    /// `small_to_mtp_projection`.
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
}

/// Linear layer — plain or quantized.
// Plain variant is architecturally present for non-quantized models but all
// Qwen3-TTS weights are 8-bit; suppress the dead_code lint.
#[allow(dead_code)]
enum Linear {
    Plain(Plain),
    Quant(Quant),
}

impl Linear {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        match self {
            Linear::Plain(p) => {
                let y = matmul(x, &p.w.transpose(&[1, 0], d)?, d)?;
                if let Some(b) = &p.b {
                    Ok(add(&y, b, d)?)
                } else {
                    Ok(y)
                }
            }
            Linear::Quant(q) => {
                let y = quantized_matmul(
                    x,
                    &q.w,
                    &q.scales,
                    Some(&q.biases),
                    q.group_size,
                    q.bits,
                    "affine",
                    true,
                    d,
                )?;
                if let Some(b) = &q.bias {
                    Ok(add(&y, b, d)?)
                } else {
                    Ok(y)
                }
            }
        }
    }
}

/// Plain embedding table. `w` shape: `[vocab, hidden]`.
struct Embedding {
    w: Array,
}

impl Embedding {
    fn forward(&self, ids: &Array, d: Device) -> Result<Array, TtsError> {
        // ids: [B, S] i32 → [B, S, hidden]
        Ok(self.w.take(ids, 0, d)?)
    }
}

/// RMSNorm with learned weight.
struct RmsNormLayer {
    w: Array,
    eps: f32,
}

impl RmsNormLayer {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        Ok(rms_norm(x, Some(&self.w), self.eps, d)?)
    }
}

// ── KV slot (simple slice-based cache for short TTS sequences) ────────────────

struct KvSlot {
    k: Option<Array>,
    v: Option<Array>,
}

impl KvSlot {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn update_and_fetch(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        d: Device,
    ) -> Result<(Array, Array), TtsError> {
        let k = match &self.k {
            None => new_k.try_clone()?,
            Some(prev) => concatenate(&[prev, new_k], 2, d)?,
        };
        let v = match &self.v {
            None => new_v.try_clone()?,
            Some(prev) => concatenate(&[prev, new_v], 2, d)?,
        };
        self.k = Some(k.try_clone()?);
        self.v = Some(v.try_clone()?);
        Ok((k, v))
    }
}

// ── RoPE helpers ─────────────────────────────────────────────────────────────

/// Build positional cos/sin tables for `seq_len` positions starting at `offset`.
/// Returns `(cos, sin)` each `[1, seq_len, head_dim]`.
fn build_rope(
    offset: i32,
    seq_len: i32,
    head_dim: i32,
    theta: f32,
    d: Device,
) -> Result<(Array, Array), TtsError> {
    // inv_freq: [head_dim/2] — 1 / (theta^(2i/head_dim))
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0_f32 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_f = Array::from_f32_slice(&inv_freq, &[1, half])?;

    // positions: [1, seq_len]
    let positions: Vec<f32> = (offset..offset + seq_len).map(|p| p as f32).collect();
    let pos = Array::from_f32_slice(&positions, &[seq_len, 1])?;

    // freqs: [seq_len, half]  = pos @ inv_f (outer product)
    let freqs = matmul(&pos, &inv_f, d)?;

    // emb: [seq_len, head_dim] by doubling
    let emb = concatenate(&[&freqs, &freqs], 1, d)?;

    // cos/sin: [1, seq_len, head_dim]
    let c = cos(&emb, d)?.reshape(&[1, seq_len, head_dim], d)?;
    let s = sin(&emb, d)?.reshape(&[1, seq_len, head_dim], d)?;

    Ok((c, s))
}

/// Apply RoPE to `x` of shape `[B, H, S, D]` using `[1, S, D]` cos/sin.
fn apply_rope(x: &Array, cos_: &Array, sin_: &Array, d: Device) -> Result<Array, TtsError> {
    // cos_/sin_ are [1, S, D]; need [1, 1, S, D] for broadcasting with [B, H, S, D]
    let cos_4d = cos_.reshape(&[1, 1, cos_.shape()[1], cos_.shape()[2]], d)?;
    let sin_4d = sin_.reshape(&[1, 1, sin_.shape()[1], sin_.shape()[2]], d)?;

    let half_d = x.shape()[3] / 2;
    let s = x.shape()[2];
    let h = x.shape()[1];
    let b = x.shape()[0];

    // rotate_half: split last dim, negate second half, concat
    let x1 = x.slice(&[0, 0, 0, 0], &[b, h, s, half_d], &[1, 1, 1, 1], d)?;
    let x2 = x.slice(
        &[0, 0, 0, half_d],
        &[b, h, s, x.shape()[3]],
        &[1, 1, 1, 1],
        d,
    )?;
    let neg_x2 = negative(&x2, d)?;
    let rotated = concatenate(&[&neg_x2, &x1], 3, d)?;

    let xcos = multiply(x, &cos_4d, d)?;
    let xsin = multiply(&rotated, &sin_4d, d)?;
    Ok(add(&xcos, &xsin, d)?)
}

// ── Talker attention layer ────────────────────────────────────────────────────

struct TalkerAttn {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNormLayer,
    k_norm: RmsNormLayer,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

impl TalkerAttn {
    fn forward(
        &self,
        x: &Array,
        kv: &mut KvSlot,
        offset: i32,
        d: Device,
    ) -> Result<Array, TtsError> {
        let (b, s, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.n_heads;
        let kh = self.n_kv_heads;
        let hd = self.head_dim;

        let q_raw = self.q_proj.forward(x, d)?;
        let k_raw = self.k_proj.forward(x, d)?;
        let v_raw = self.v_proj.forward(x, d)?;

        // reshape → [B, S, heads, head_dim] then transpose → [B, heads, S, head_dim]
        let q = q_raw
            .reshape(&[b, s, h, hd], d)?
            .transpose(&[0, 2, 1, 3], d)?;
        let k = k_raw
            .reshape(&[b, s, kh, hd], d)?
            .transpose(&[0, 2, 1, 3], d)?;
        let v = v_raw
            .reshape(&[b, s, kh, hd], d)?
            .transpose(&[0, 2, 1, 3], d)?;

        // per-head q/k RMSNorm (applied on last dim, needs reshape to merge heads)
        let q = {
            let flat = q.reshape(&[b * h, s, hd], d)?;
            let normed = self.q_norm.forward(&flat, d)?;
            normed.reshape(&[b, h, s, hd], d)?
        };
        let k = {
            let flat = k.reshape(&[b * kh, s, hd], d)?;
            let normed = self.k_norm.forward(&flat, d)?;
            normed.reshape(&[b, kh, s, hd], d)?
        };

        // RoPE on q and k
        let (cos_, sin_) = build_rope(offset, s, hd, self.rope_theta, d)?;
        let q = apply_rope(&q, &cos_, &sin_, d)?;
        let k = apply_rope(&k, &cos_, &sin_, d)?;

        // KV cache: concat along seq dim (axis=2)
        let (k, v) = kv.update_and_fetch(&k, &v, d)?;

        // GQA: repeat k/v n_groups times along the head axis.
        let n_groups = (self.n_heads / self.n_kv_heads) as usize;
        let total_s = k.shape()[2];
        let (k_rep, v_rep) = if n_groups > 1 {
            // [B, kh, S, D] → [B, kh, 1, S, D] → concatenate ng times → [B, kh*ng, S, D]
            let k_4 = k.reshape(&[b, kh, 1, total_s, hd], d)?;
            let v_4 = v.reshape(&[b, kh, 1, total_s, hd], d)?;
            let mut k_rows: Vec<Array> = Vec::with_capacity(n_groups);
            let mut v_rows: Vec<Array> = Vec::with_capacity(n_groups);
            for _ in 0..n_groups {
                k_rows.push(k_4.slice(
                    &[0, 0, 0, 0, 0],
                    &[b, kh, 1, total_s, hd],
                    &[1, 1, 1, 1, 1],
                    d,
                )?);
                v_rows.push(v_4.slice(
                    &[0, 0, 0, 0, 0],
                    &[b, kh, 1, total_s, hd],
                    &[1, 1, 1, 1, 1],
                    d,
                )?);
            }
            let k_refs: Vec<&Array> = k_rows.iter().collect();
            let v_refs: Vec<&Array> = v_rows.iter().collect();
            let k5 = concatenate(&k_refs, 2, d)?.reshape(&[b, h, total_s, hd], d)?;
            let v5 = concatenate(&v_refs, 2, d)?.reshape(&[b, h, total_s, hd], d)?;
            (k5, v5)
        } else {
            (k.try_clone()?, v.try_clone()?)
        };

        // SDPA with causal masking — required for correct prefill of the autoregressive talker.
        // "causal" mode applies triangular attention mask internally; safe for single-token
        // decode steps too (trivially causal).
        let out = scaled_dot_product_attention(&q, &k_rep, &v_rep, self.scale, "causal", None, d)?;
        // [B, H, S, D] → [B, S, H*D]
        let out = out
            .transpose(&[0, 2, 1, 3], d)?
            .reshape(&[b, s, h * hd], d)?;
        self.o_proj.forward(&out, d)
    }
}

// ── Talker MLP ────────────────────────────────────────────────────────────────

struct TalkerMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl TalkerMlp {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let g = silu(&self.gate.forward(x, d)?, d)?;
        let u = self.up.forward(x, d)?;
        let h = multiply(&g, &u, d)?;
        self.down.forward(&h, d)
    }
}

// ── Talker transformer layer ──────────────────────────────────────────────────

struct TalkerLayer {
    attn: TalkerAttn,
    mlp: TalkerMlp,
    input_ln: RmsNormLayer,
    post_attn_ln: RmsNormLayer,
}

impl TalkerLayer {
    fn forward(
        &self,
        x: &Array,
        kv: &mut KvSlot,
        offset: i32,
        d: Device,
    ) -> Result<Array, TtsError> {
        let h = add(
            x,
            &self
                .attn
                .forward(&self.input_ln.forward(x, d)?, kv, offset, d)?,
            d,
        )?;
        let out = add(
            &h,
            &self.mlp.forward(&self.post_attn_ln.forward(&h, d)?, d)?,
            d,
        )?;
        Ok(out)
    }
}

// ── Talker model ─────────────────────────────────────────────────────────────

struct TalkerModel {
    text_embed: Embedding,
    codec_embed: Embedding,
    text_proj_fc1: Linear,
    text_proj_fc2: Linear,
    layers: Vec<TalkerLayer>,
    norm: RmsNormLayer,
    codec_head: Linear,
    code_pred: CodePredictor,
}

impl TalkerModel {
    fn text_projection(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        // SwiGLU-style: fc2(silu(fc1(x)) * x) is not the pattern here;
        // from the Python: text_projection is a simple 2-layer MLP with silu
        // Actually inspecting: text_projection.linear_fc1 + linear_fc2 is a
        // gate-up style projection. Let's use: out = fc2(silu(fc1(x)) * x)
        // But fc1 bias exists + fc2 bias exists. Simple sequential:
        let h = silu(&self.text_proj_fc1.forward(x, d)?, d)?;
        self.text_proj_fc2.forward(&h, d)
    }

    /// Embed text token ids and project into talker hidden space.
    fn embed_text(&self, ids: &Array, d: Device) -> Result<Array, TtsError> {
        let raw = self.text_embed.forward(ids, d)?;
        self.text_projection(&raw, d)
    }

    /// Run talker forward. Returns `(logits, hidden_states)`.
    /// `x` is input embeddings `[1, S, hidden]`.
    /// `kv_slots` is mutable KV cache (one per layer).
    /// `offset` is the current position (for RoPE).
    fn forward(
        &self,
        x: &Array,
        kv_slots: &mut [KvSlot],
        offset: i32,
        d: Device,
    ) -> Result<(Array, Array), TtsError> {
        let mut h = x.try_clone()?;
        for (layer, kv) in self.layers.iter().zip(kv_slots.iter_mut()) {
            h = layer.forward(&h, kv, offset, d)?;
        }
        let normed = self.norm.forward(&h, d)?;
        let logits = self.codec_head.forward(&normed, d)?;
        // Return normed hidden — the CP uses the norm-applied hidden as `past_hidden`.
        Ok((logits, normed))
    }
}

// ── CodePredictor ─────────────────────────────────────────────────────────────

/// 5-layer mini-Qwen3 that generates codec groups 1..15 given group-0 token.
struct CodePredictor {
    // 15 additional codec embeddings (for groups 1..15)
    codec_embeddings: Vec<Embedding>,
    /// small_to_mtp_projection: projects talker hidden → CP hidden
    small_to_mtp: Linear,
    layers: Vec<TalkerLayer>, // same structure, smaller dims
    norm: RmsNormLayer,
    /// 15 LM heads (one per codec group 1..15; group 0 used talker.codec_head)
    lm_heads: Vec<Linear>,
}

impl CodePredictor {
    /// Run one autoregressive step of the CodePredictor.
    /// `input`: [1, 1 or 2, cp_hidden] (code_hidden concat code_embed for step 0;
    ///          just code_embed for subsequent steps).
    /// Returns logits `[1, 1, codebook_size]` for the current group.
    fn forward(
        &self,
        input: &Array,
        kv_slots: &mut [KvSlot],
        offset: i32,
        group_idx: usize, // 0-based: 0 → produce group 1, ..., 14 → produce group 15
        d: Device,
    ) -> Result<Array, TtsError> {
        let mut h = input.try_clone()?;
        for (layer, kv) in self.layers.iter().zip(kv_slots.iter_mut()) {
            h = layer.forward(&h, kv, offset, d)?;
        }
        let normed = self.norm.forward(&h, d)?;
        // lm_heads[group_idx]: projects [1, S, cp_hidden] → [1, S, codebook_size]
        let logits = self.lm_heads[group_idx].forward(&normed, d)?;
        Ok(logits)
    }
}

// ── Codec decoder ─────────────────────────────────────────────────────────────

/// Single VQ codebook. Embedding = embedding_sum / max(cluster_usage, 1).
struct VqCodebook {
    embed: Array, // [2048, 256] f32 — pre-normalized embedding table
}

impl VqCodebook {
    fn decode(&self, codes: &Array, d: Device) -> Result<Array, TtsError> {
        // codes: [B, T] i32 → [B, T, 256]
        Ok(self.embed.take(codes, 0, d)?)
    }
}

/// Conv1d weight wrapper for the codec (plain f32).
struct CodecConv1d {
    w: Array, // [out, kernel, in] in MLX layout
    b: Option<Array>,
    padding: i32,
    /// Number of groups (1 = regular conv, C = depthwise conv).
    groups: i32,
    /// Convolution dilation (1 = standard, >1 = dilated/atrous).
    dilation: i32,
}

impl CodecConv1d {
    fn forward(&self, x: &Array, stride: i32, d: Device) -> Result<Array, TtsError> {
        // x: [B, T, C] — MLX conv1d is channel-last.
        // Symmetric padding of `padding` on each side simulates causal padding when
        // we trim the right side by `padding` after the conv.
        let y = conv1d(
            x,
            &self.w,
            stride,
            self.padding,
            self.dilation,
            self.groups,
            d,
        )?;
        let y = if self.padding > 0 {
            // Causal trim: remove the `padding` extra samples added on the right.
            let b = y.shape()[0];
            let t = y.shape()[1];
            let c = y.shape()[2];
            let t_trim = t - self.padding;
            if t_trim > 0 {
                y.slice(&[0, 0, 0], &[b, t_trim, c], &[1, 1, 1], d)?
            } else {
                y
            }
        } else {
            y
        };
        if let Some(b) = &self.b {
            Ok(add(&y, b, d)?)
        } else {
            Ok(y)
        }
    }
}

/// ConvTranspose1d for upsample (causal trim applied).
struct CodecConvT1d {
    w: Array, // [out, kernel, in] MLX layout
    b: Option<Array>,
    stride: i32,
    trim_right: i32,
}

impl CodecConvT1d {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        // x: [B, T, C]
        let y = conv_transpose1d(x, &self.w, self.stride, 0, 1, 0, 1, d)?;
        let y = if let Some(b) = &self.b {
            add(&y, b, d)?
        } else {
            y
        };
        if self.trim_right > 0 {
            let t = y.shape()[1];
            let stop = t - self.trim_right;
            y.slice(
                &[0, 0, 0],
                &[y.shape()[0], stop, y.shape()[2]],
                &[1, 1, 1],
                d,
            )
            .map_err(Into::into)
        } else {
            Ok(y)
        }
    }
}

/// SnakeBeta activation: `x + (1/exp(β)) * sin²(exp(α)·x)`.
struct SnakeBeta {
    alpha: Array, // [C]
    beta: Array,  // [C]
}

impl SnakeBeta {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let a = exp(&self.alpha, d)?;
        let b = exp(&self.beta, d)?;
        // 1 / (b + eps)
        let eps = scalar_f32(1e-9);
        let b_eps = add(&b, &eps, d)?;
        let inv_b = {
            let one = scalar_f32(1.0);
            // divide: one / b_eps
            divide(&one, &b_eps, d)?
        };
        // sin(exp(alpha) * x)
        let ax = multiply(&a, x, d)?;
        let s = sin(&ax, d)?;
        // sin^2
        let s2 = multiply(&s, &s, d)?;
        // x + inv_b * s2
        let term = multiply(&inv_b, &s2, d)?;
        Ok(add(x, &term, d)?)
    }
}

/// LayerNorm for ConvNeXt.
struct LayerNorm {
    w: Array,
    b: Array,
    eps: f32,
}

impl LayerNorm {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let last = (x.ndim() - 1) as i32;
        let n = x.shape()[x.ndim() - 1] as f32;
        let inv_n = scalar_f32(1.0 / n);
        let mean = multiply(&sum_axis_keepdims(x, last, d)?, &inv_n, d)?;
        let centered = subtract(x, &mean, d)?;
        let sq = multiply(&centered, &centered, d)?;
        let var = multiply(&sum_axis_keepdims(&sq, last, d)?, &inv_n, d)?;
        let eps_a = scalar_f32(self.eps);
        let denom = sqrt(&add(&var, &eps_a, d)?, d)?;
        let normed = divide(&centered, &denom, d)?;
        let scaled = multiply(&normed, &self.w, d)?;
        Ok(add(&scaled, &self.b, d)?)
    }
}

/// ConvNeXt block: DW causal conv(k=7) + LayerNorm + PW(4x) + GELU + PW + gamma·residual.
struct ConvNeXt {
    dw_conv: CodecConv1d, // depthwise k=7 groups=C
    norm: LayerNorm,
    pw1_w: Array, // [4C, C] for matmul
    pw1_b: Array,
    pw2_w: Array, // [C, 4C]
    pw2_b: Array,
    gamma: Array, // [C]
}

impl ConvNeXt {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let residual = x.try_clone()?;
        // DW causal conv: causal trim handled inside CodecConv1d.forward.
        let h = self.dw_conv.forward(x, 1, d)?;
        let h = self.norm.forward(&h, d)?;
        // PW linear 1
        let h = {
            let y = matmul(&h, &self.pw1_w.transpose(&[1, 0], d)?, d)?;
            add(&y, &self.pw1_b, d)?
        };
        let h = gelu(&h, d)?;
        // PW linear 2
        let h = {
            let y = matmul(&h, &self.pw2_w.transpose(&[1, 0], d)?, d)?;
            add(&y, &self.pw2_b, d)?
        };
        // gamma scaling
        let h = multiply(&self.gamma, &h, d)?;
        Ok(add(&residual, &h, d)?)
    }
}

/// Pre-transformer RMSNorm (no bias, as in decoder pre_transformer).
struct PreTransRmsNorm {
    w: Array,
    eps: f32,
}

impl PreTransRmsNorm {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        Ok(rms_norm(x, Some(&self.w), self.eps, d)?)
    }
}

/// Pre-transformer attention (plain, no QK norm).
struct PreTransAttn {
    q_proj: Plain,
    k_proj: Plain,
    v_proj: Plain,
    o_proj: Plain,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

impl PreTransAttn {
    fn lin(p: &Plain, x: &Array, d: Device) -> Result<Array, TtsError> {
        let y = matmul(x, &p.w.transpose(&[1, 0], d)?, d)?;
        if let Some(b) = &p.b {
            Ok(add(&y, b, d)?)
        } else {
            Ok(y)
        }
    }

    fn forward(&self, x: &Array, offset: i32, d: Device) -> Result<Array, TtsError> {
        let (b, s, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.n_heads;
        let hd = self.head_dim;

        let q = Self::lin(&self.q_proj, x, d)?
            .reshape(&[b, s, h, hd], d)?
            .transpose(&[0, 2, 1, 3], d)?;
        let k = Self::lin(&self.k_proj, x, d)?
            .reshape(&[b, s, h, hd], d)?
            .transpose(&[0, 2, 1, 3], d)?;
        let v = Self::lin(&self.v_proj, x, d)?
            .reshape(&[b, s, h, hd], d)?
            .transpose(&[0, 2, 1, 3], d)?;

        let (cos_, sin_) = build_rope(offset, s, hd, self.rope_theta, d)?;
        let q = apply_rope(&q, &cos_, &sin_, d)?;
        let k = apply_rope(&k, &cos_, &sin_, d)?;

        // Causal attention for the codec pre-transformer.
        // The decoder was trained with causal masking even in non-streaming mode.
        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, "causal", None, d)?;
        let out = out
            .transpose(&[0, 2, 1, 3], d)?
            .reshape(&[b, s, h * hd], d)?;
        Self::lin(&self.o_proj, &out, d)
    }
}

/// Pre-transformer MLP (SwiGLU, plain weights).
struct PreTransMlp {
    gate: Plain,
    up: Plain,
    down: Plain,
}

impl PreTransMlp {
    fn lin(p: &Plain, x: &Array, d: Device) -> Result<Array, TtsError> {
        let y = matmul(x, &p.w.transpose(&[1, 0], d)?, d)?;
        if let Some(b) = &p.b {
            Ok(add(&y, b, d)?)
        } else {
            Ok(y)
        }
    }

    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let g = silu(&Self::lin(&self.gate, x, d)?, d)?;
        let u = Self::lin(&self.up, x, d)?;
        let h = multiply(&g, &u, d)?;
        Self::lin(&self.down, &h, d)
    }
}

/// One layer of the decoder pre-transformer with layer-scale.
struct PreTransLayer {
    attn: PreTransAttn,
    mlp: PreTransMlp,
    input_ln: PreTransRmsNorm,
    post_attn_ln: PreTransRmsNorm,
    attn_scale: Array, // [hidden] layer scale
    mlp_scale: Array,
}

impl PreTransLayer {
    fn forward(&self, x: &Array, offset: i32, d: Device) -> Result<Array, TtsError> {
        let attn_out = self
            .attn
            .forward(&self.input_ln.forward(x, d)?, offset, d)?;
        let attn_scaled = multiply(&self.attn_scale, &attn_out, d)?;
        let h = add(x, &attn_scaled, d)?;

        let mlp_out = self.mlp.forward(&self.post_attn_ln.forward(&h, d)?, d)?;
        let mlp_scaled = multiply(&self.mlp_scale, &mlp_out, d)?;
        Ok(add(&h, &mlp_scaled, d)?)
    }
}

/// Residual block inside a decoder group.
struct ResBlock {
    act1: SnakeBeta,
    conv1: CodecConv1d,
    act2: SnakeBeta,
    conv2: CodecConv1d,
}

impl ResBlock {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let residual = x.try_clone()?;
        let h = self.act1.forward(x, d)?;
        let h = self.conv1.forward(&h, 1, d)?;
        let h = self.act2.forward(&h, d)?;
        let h = self.conv2.forward(&h, 1, d)?;
        Ok(add(&residual, &h, d)?)
    }
}

/// One decoder group: snake → ConvT1d upsample → 3 ResBlocks.
struct DecoderGroup {
    snake_in: SnakeBeta,
    upsample: CodecConvT1d,
    res_blocks: Vec<ResBlock>, // 3 blocks with dilation 1,3,9
}

impl DecoderGroup {
    fn forward(&self, x: &Array, d: Device) -> Result<Array, TtsError> {
        let mut h = self.snake_in.forward(x, d)?;
        h = self.upsample.forward(&h, d)?;
        for rb in &self.res_blocks {
            h = rb.forward(&h, d)?;
        }
        Ok(h)
    }
}

/// Full codec decoder.
#[allow(dead_code)] // rvq_first_input / rvq_rest_input loaded but not yet applied in decode path
pub(crate) struct CodecDecoder {
    // VQ quantizer: 1 rvq_first + 15 rvq_rest
    rvq_first_input: CodecConv1d, // loaded; applied before VQ in future refinement
    rvq_first_output: CodecConv1d,
    rvq_first_cb: VqCodebook,
    rvq_rest_input: CodecConv1d,
    rvq_rest_output: CodecConv1d,
    rvq_rest_cbs: Vec<VqCodebook>, // 15 codebooks
    // pre_conv: CausalConv1d k=3 512→1024 with padding=2
    pre_conv: CodecConv1d,
    // pre_transformer: 8 layers
    pre_trans_input: Plain, // [512, 1024]
    pre_trans_norm: PreTransRmsNorm,
    pre_trans_output: Plain, // [1024, 512]
    pre_trans_layers: Vec<PreTransLayer>,
    // upsample: 2× (ConvT1d + ConvNeXt)
    upsample: Vec<(CodecConvT1d, ConvNeXt)>,
    // decoder: initial_conv + 4 groups + output_snake + output_conv
    initial_conv: CodecConv1d,
    groups: Vec<DecoderGroup>,
    output_snake: SnakeBeta,
    output_conv: CodecConv1d,
}

impl CodecDecoder {
    /// Decode `codes: [1, 16, T]` (16 codec groups) to audio `[T * upsample]`.
    fn decode(&self, codes: &Array, d: Device) -> Result<Vec<f32>, TtsError> {
        // Step 1: VQ dequantization → [1, T, 512]
        let t = codes.shape()[2];
        let first_codes = codes.slice(&[0, 0, 0], &[1, 1, t], &[1, 1, 1], d)?;
        let first_codes_2d = first_codes.reshape(&[1, t], d)?;

        // rvq_first: input_proj (conv k=1 512→256) → lookup → output_proj (256→512)
        // The codebook embeds [2048, 256], then project out to 512
        let first_emb = self.rvq_first_cb.decode(&first_codes_2d, d)?; // [1, T, 256]
                                                                       // output_proj: conv1d k=1 → [1, T, 512]
        let first_q = self.rvq_first_output.forward(&first_emb, 1, d)?; // [1, T, 512]

        // rvq_rest: sum of 15 codebook embeddings
        let mut rest_sum = {
            let codes_1 = codes.slice(&[0, 1, 0], &[1, 2, t], &[1, 1, 1], d)?;
            let codes_1_2d = codes_1.reshape(&[1, t], d)?;
            let emb = self.rvq_rest_cbs[0].decode(&codes_1_2d, d)?; // [1, T, 256]
            emb
        };
        for i in 1..15usize {
            let codes_i =
                codes.slice(&[0, i as i32 + 1, 0], &[1, i as i32 + 2, t], &[1, 1, 1], d)?;
            let codes_i_2d = codes_i.reshape(&[1, t], d)?;
            let emb = self.rvq_rest_cbs[i].decode(&codes_i_2d, d)?;
            rest_sum = add(&rest_sum, &emb, d)?;
        }
        // rvq_rest output proj: [1, T, 512]
        let rest_q = self.rvq_rest_output.forward(&rest_sum, 1, d)?;

        // Combined: [1, T, 512]
        let mut hidden = add(&first_q, &rest_q, d)?;

        // Step 2: pre_conv k=3 causal (padding=2) → [1, T, 1024]
        hidden = self.pre_conv.forward(&hidden, 1, d)?;

        // Step 3: pre_transformer: input_proj → 8 layers → norm → output_proj
        let h_proj = {
            let y = matmul(&hidden, &self.pre_trans_input.w.transpose(&[1, 0], d)?, d)?;
            if let Some(b) = &self.pre_trans_input.b {
                add(&y, b, d)?
            } else {
                y
            }
        };
        let mut h = h_proj;
        for layer in &self.pre_trans_layers {
            h = layer.forward(&h, 0, d)?;
        }
        h = self.pre_trans_norm.forward(&h, d)?;
        hidden = {
            let y = matmul(&h, &self.pre_trans_output.w.transpose(&[1, 0], d)?, d)?;
            if let Some(b) = &self.pre_trans_output.b {
                add(&y, b, d)?
            } else {
                y
            }
        };

        // Step 4: 2× upsample (ConvT1d stride=2 + ConvNeXt)
        for (conv_t, conv_next) in &self.upsample {
            hidden = conv_t.forward(&hidden, d)?;
            hidden = conv_next.forward(&hidden, d)?;
        }

        // Step 5: initial_conv (k=7 causal, 1024→1536) → 4 groups → output_snake → output_conv
        hidden = self.initial_conv.forward(&hidden, 1, d)?;
        for group in &self.groups {
            hidden = group.forward(&hidden, d)?;
        }
        hidden = self.output_snake.forward(&hidden, d)?;
        hidden = self.output_conv.forward(&hidden, 1, d)?;

        // Step 6: tanh clip → extract samples
        let audio = tanh(&hidden, d)?;
        // Materialise the full audio graph before reading bytes.
        audio.eval()?;
        // audio: [1, samples, 1] → flatten
        let audio_bytes = audio.to_bytes()?;
        let n = audio_bytes.len() / 4;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                f32::from_le_bytes(audio_bytes[i * 4..i * 4 + 4].try_into().unwrap_or([0u8; 4]))
            })
            .collect();
        Ok(samples)
    }
    /// Debug decode: dumps intermediate values to stderr for comparison with reference Python.
    /// Only compiled in test builds.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn debug_decode(&self, codes: &Array, d: Device) -> Result<Vec<f32>, TtsError> {
        fn dump(label: &str, x: &Array) {
            x.eval().ok();
            let bytes = x.to_bytes().unwrap_or_default();
            let n = (bytes.len() / 4).min(5);
            let first5: Vec<f32> = (0..n)
                .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap_or([0u8; 4])))
                .collect();
            let shape: Vec<i32> = (0..x.ndim()).map(|i| x.shape()[i]).collect();
            eprintln!("{label}: shape={shape:?} first5={first5:.6?}");
        }

        let t = codes.shape()[2];
        let first_codes = codes.slice(&[0, 0, 0], &[1, 1, t], &[1, 1, 1], d)?;
        let first_codes_2d = first_codes.reshape(&[1, t], d)?;
        let first_emb = self.rvq_first_cb.decode(&first_codes_2d, d)?;
        let first_q = self.rvq_first_output.forward(&first_emb, 1, d)?;

        let mut rest_sum = {
            let c = codes.slice(&[0, 1, 0], &[1, 2, t], &[1, 1, 1], d)?;
            let c2d = c.reshape(&[1, t], d)?;
            self.rvq_rest_cbs[0].decode(&c2d, d)?
        };
        for i in 1..15usize {
            let c = codes.slice(&[0, i as i32 + 1, 0], &[1, i as i32 + 2, t], &[1, 1, 1], d)?;
            let c2d = c.reshape(&[1, t], d)?;
            let emb = self.rvq_rest_cbs[i].decode(&c2d, d)?;
            rest_sum = add(&rest_sum, &emb, d)?;
        }
        let rest_q = self.rvq_rest_output.forward(&rest_sum, 1, d)?;
        let mut hidden = add(&first_q, &rest_q, d)?;
        dump("VQ combined NLC", &hidden);

        hidden = self.pre_conv.forward(&hidden, 1, d)?;
        dump("pre_conv", &hidden);

        let h_proj = {
            let y = matmul(&hidden, &self.pre_trans_input.w.transpose(&[1, 0], d)?, d)?;
            if let Some(b) = &self.pre_trans_input.b {
                add(&y, b, d)?
            } else {
                y
            }
        };
        let mut h = h_proj;
        for layer in &self.pre_trans_layers {
            h = layer.forward(&h, 0, d)?;
        }
        h = self.pre_trans_norm.forward(&h, d)?;
        hidden = {
            let y = matmul(&h, &self.pre_trans_output.w.transpose(&[1, 0], d)?, d)?;
            if let Some(b) = &self.pre_trans_output.b {
                add(&y, b, d)?
            } else {
                y
            }
        };
        dump("pre_transformer", &hidden);

        for (i, (conv_t, conv_next)) in self.upsample.iter().enumerate() {
            hidden = conv_t.forward(&hidden, d)?;
            dump(&format!("upsample[{i}][0]"), &hidden);
            hidden = conv_next.forward(&hidden, d)?;
            dump(&format!("upsample[{i}][1]"), &hidden);
        }

        hidden = self.initial_conv.forward(&hidden, 1, d)?;
        dump("initial_conv", &hidden);
        for (i, group) in self.groups.iter().enumerate() {
            hidden = group.forward(&hidden, d)?;
            dump(&format!("decoder_group[{i}]"), &hidden);
        }
        hidden = self.output_snake.forward(&hidden, d)?;
        dump("output_snake", &hidden);
        hidden = self.output_conv.forward(&hidden, 1, d)?;
        dump("output_conv", &hidden);

        let audio = tanh(&hidden, d)?;
        audio.eval()?;
        let audio_bytes = audio.to_bytes()?;
        let n = audio_bytes.len() / 4;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                f32::from_le_bytes(audio_bytes[i * 4..i * 4 + 4].try_into().unwrap_or([0u8; 4]))
            })
            .collect();
        eprintln!("Final samples: len={} rms={:.6}", samples.len(), {
            let sq: f32 = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;
            sq.sqrt()
        });
        Ok(samples)
    }
}

// ── Weight loaders ────────────────────────────────────────────────────────────

fn load_tensor(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
) -> Result<Array, TtsError> {
    let tv = view(shards, idx, key).map_err(|e| TtsError::Load(format!("{key}: {e}")))?;
    Array::from_safetensor_view(&tv).map_err(|e| TtsError::Load(format!("{key}: {e}")))
}

fn load_tensor_opt(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
) -> Result<Option<Array>, TtsError> {
    if idx.weight_map.contains_key(key) {
        Ok(Some(load_tensor(shards, idx, key)?))
    } else {
        Ok(None)
    }
}

fn load_quant_linear(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    prefix: &str,
) -> Result<Linear, TtsError> {
    let w = load_tensor(shards, idx, &format!("{prefix}.weight"))?;
    let scales = load_tensor(shards, idx, &format!("{prefix}.scales"))?;
    let biases = load_tensor(shards, idx, &format!("{prefix}.biases"))?;
    // Some layers (text_projection, small_to_mtp_projection) carry a plain additive
    // bias in addition to the quantization biases. Try to load it; silently skip if absent.
    let bias = load_tensor_opt(shards, idx, &format!("{prefix}.bias"))?;
    Ok(Linear::Quant(Quant {
        w,
        scales,
        biases,
        bias,
        group_size: 64,
        bits: 8,
    }))
}

fn load_rms_norm(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
    eps: f32,
) -> Result<RmsNormLayer, TtsError> {
    let w = load_tensor(shards, idx, key)?;
    Ok(RmsNormLayer { w, eps })
}

fn load_embedding(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
) -> Result<Embedding, TtsError> {
    let w = load_tensor(shards, idx, key)?;
    Ok(Embedding { w })
}

fn load_talker_attn(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
    cfg: &TalkerConfig,
) -> Result<TalkerAttn, TtsError> {
    Ok(TalkerAttn {
        q_proj: load_quant_linear(shards, idx, &format!("{pfx}.self_attn.q_proj"))?,
        k_proj: load_quant_linear(shards, idx, &format!("{pfx}.self_attn.k_proj"))?,
        v_proj: load_quant_linear(shards, idx, &format!("{pfx}.self_attn.v_proj"))?,
        o_proj: load_quant_linear(shards, idx, &format!("{pfx}.self_attn.o_proj"))?,
        q_norm: load_rms_norm(
            shards,
            idx,
            &format!("{pfx}.self_attn.q_norm.weight"),
            cfg.rms_norm_eps,
        )?,
        k_norm: load_rms_norm(
            shards,
            idx,
            &format!("{pfx}.self_attn.k_norm.weight"),
            cfg.rms_norm_eps,
        )?,
        n_heads: cfg.num_attention_heads as i32,
        n_kv_heads: cfg.num_key_value_heads as i32,
        head_dim: cfg.head_dim as i32,
        scale: (cfg.head_dim as f32).powf(-0.5),
        rope_theta: cfg.rope_theta,
    })
}

fn load_talker_mlp(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
) -> Result<TalkerMlp, TtsError> {
    Ok(TalkerMlp {
        gate: load_quant_linear(shards, idx, &format!("{pfx}.mlp.gate_proj"))?,
        up: load_quant_linear(shards, idx, &format!("{pfx}.mlp.up_proj"))?,
        down: load_quant_linear(shards, idx, &format!("{pfx}.mlp.down_proj"))?,
    })
}

fn load_talker_layer(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
    cfg: &TalkerConfig,
) -> Result<TalkerLayer, TtsError> {
    Ok(TalkerLayer {
        attn: load_talker_attn(shards, idx, pfx, cfg)?,
        mlp: load_talker_mlp(shards, idx, pfx)?,
        input_ln: load_rms_norm(
            shards,
            idx,
            &format!("{pfx}.input_layernorm.weight"),
            cfg.rms_norm_eps,
        )?,
        post_attn_ln: load_rms_norm(
            shards,
            idx,
            &format!("{pfx}.post_attention_layernorm.weight"),
            cfg.rms_norm_eps,
        )?,
    })
}

fn load_plain_p(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    prefix: &str,
) -> Result<Plain, TtsError> {
    let w = load_tensor(shards, idx, &format!("{prefix}.weight"))?;
    let b = load_tensor_opt(shards, idx, &format!("{prefix}.bias"))?;
    Ok(Plain { w, b })
}

// ── Public model struct ───────────────────────────────────────────────────────

/// Loaded Qwen3-TTS model (talker + codec decoder).
#[allow(clippy::exhaustive_structs)]
pub struct TtsModel {
    /// Parsed talker `config.json`.
    pub config: TtsConfig,
    /// Path to the talker model directory.
    pub talker_path: std::path::PathBuf,
    /// Path to the codec decoder model directory.
    pub codec_path: std::path::PathBuf,
    talker: Option<Box<TalkerModel>>,
    codec: Option<Box<CodecDecoder>>,
}

impl std::fmt::Debug for TtsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsModel")
            .field("model_type", &self.config.model_type)
            .field("talker_path", &self.talker_path)
            .field("codec_path", &self.codec_path)
            .field("talker_loaded", &self.talker.is_some())
            .field("codec_loaded", &self.codec.is_some())
            .finish()
    }
}

impl TtsModel {
    /// Load config only (no weights). Weights loaded on first `synthesize` call.
    pub fn load_config(
        talker_path: impl AsRef<Path>,
        codec_path: impl AsRef<Path>,
    ) -> Result<Self, TtsError> {
        let cfg_path = talker_path.as_ref().join("config.json");
        let cfg_str = std::fs::read_to_string(&cfg_path)
            .map_err(|e| TtsError::Load(format!("{}: {e}", cfg_path.display())))?;
        let config = TtsConfig::from_json(&cfg_str)?;
        Ok(Self {
            config,
            talker_path: talker_path.as_ref().to_path_buf(),
            codec_path: codec_path.as_ref().to_path_buf(),
            talker: None,
            codec: None,
        })
    }

    /// Construct a model shell from an already-parsed config, for testing only.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        config: TtsConfig,
        talker_path: std::path::PathBuf,
        codec_path: std::path::PathBuf,
    ) -> Self {
        Self {
            config,
            talker_path,
            codec_path,
            talker: None,
            codec: None,
        }
    }

    /// Load all weights. Call once before synthesize.
    #[allow(clippy::cognitive_complexity)]
    #[instrument(skip(self), fields(talker_path = %self.talker_path.display()))]
    pub fn load(&mut self) -> Result<(), TtsError> {
        let t0 = Instant::now();
        info!("loading Qwen3-TTS talker weights");
        self.talker = Some(Box::new(load_talker_weights(
            &self.talker_path,
            &self.config,
        )?));
        info!(elapsed_ms = t0.elapsed().as_millis(), "talker loaded");

        let t1 = Instant::now();
        info!("loading Qwen3-TTS codec decoder weights");
        self.codec = Some(Box::new(load_codec_decoder(&self.codec_path)?));
        info!(
            elapsed_ms = t1.elapsed().as_millis(),
            "codec decoder loaded"
        );

        Ok(())
    }
}

fn load_talker_weights(talker_path: &Path, cfg: &TtsConfig) -> Result<TalkerModel, TtsError> {
    // CP layer config constants — defined here to satisfy items_after_statements.
    // From talker_config.code_predictor_config: hidden_size=1024, n_heads=16, kv_heads=8,
    // head_dim=128, intermediate_size=3072, num_hidden_layers=5.
    const CP_N_HEADS: usize = 16;
    const CP_KV_HEADS: usize = 8;
    const CP_HEAD_DIM: usize = 128;
    const CP_INTERMEDIATE: usize = 3072;

    let idx = load_shard_index(talker_path).map_err(|e| TtsError::Load(e.to_string()))?;
    let shards = ShardSet::open(talker_path, &idx).map_err(|e| TtsError::Load(e.to_string()))?;
    let tcfg = &cfg.talker_config;

    // Text embedding + codec embedding
    let text_embed = load_embedding(&shards, &idx, "talker.model.text_embedding.weight")?;
    let codec_embed = load_embedding(&shards, &idx, "talker.model.codec_embedding.weight")?;

    // text_projection: fc1 (quantized, has bias) + fc2 (quantized, has bias)
    // The projection uses a gate-style: fc1 produces half for silu, fc2 projects down.
    // Actual Python: text_projection is just two linear layers applied sequentially.
    let text_proj_fc1 = load_quant_linear(&shards, &idx, "talker.text_projection.linear_fc1")?;
    let text_proj_fc2 = load_quant_linear(&shards, &idx, "talker.text_projection.linear_fc2")?;

    // 28 talker layers
    let mut layers = Vec::with_capacity(tcfg.num_hidden_layers);
    for i in 0..tcfg.num_hidden_layers {
        let pfx = format!("talker.model.layers.{i}");
        layers.push(load_talker_layer(&shards, &idx, &pfx, tcfg)?);
    }

    // Final norm + codec_head
    let norm = load_rms_norm(&shards, &idx, "talker.model.norm.weight", tcfg.rms_norm_eps)?;
    let codec_head = load_quant_linear(&shards, &idx, "talker.codec_head")?;

    // CodePredictor
    let cp_pfx = "talker.code_predictor";

    // 15 codec embeddings (groups 0..14 of the residual embeddings)
    let mut cp_codec_embeddings = Vec::with_capacity(15);
    for i in 0..15usize {
        let emb_w = load_tensor(
            &shards,
            &idx,
            &format!("{cp_pfx}.model.codec_embedding.{i}.weight"),
        )?;
        cp_codec_embeddings.push(Embedding { w: emb_w });
    }

    // small_to_mtp_projection
    let small_to_mtp =
        load_quant_linear(&shards, &idx, &format!("{cp_pfx}.small_to_mtp_projection"))?;

    // CP layer config — from talker_config.code_predictor_config in config.json:
    //   hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
    //   head_dim=128, intermediate_size=3072, num_hidden_layers=5.
    // CP_N_HEADS / CP_KV_HEADS / CP_HEAD_DIM / CP_INTERMEDIATE declared at function top.
    let cp_cfg = TalkerConfig {
        hidden_size: tcfg.code_predictor_hidden_size,
        num_attention_heads: CP_N_HEADS,
        num_key_value_heads: CP_KV_HEADS,
        num_hidden_layers: tcfg.code_predictor_num_hidden_layers,
        head_dim: CP_HEAD_DIM,
        rope_theta: tcfg.rope_theta,
        rms_norm_eps: tcfg.rms_norm_eps,
        num_code_groups: tcfg.num_code_groups,
        codec_bos_id: tcfg.codec_bos_id,
        codec_eos_token_id: tcfg.codec_eos_token_id,
        codec_nothink_id: tcfg.codec_nothink_id,
        codec_think_bos_id: tcfg.codec_think_bos_id,
        codec_think_eos_id: tcfg.codec_think_eos_id,
        codec_pad_id: tcfg.codec_pad_id,
        codec_language_id: std::collections::HashMap::new(),
        spk_id: std::collections::HashMap::new(),
        intermediate_size: CP_INTERMEDIATE,
        code_predictor_intermediate_size: 0,
        code_predictor_hidden_size: tcfg.code_predictor_hidden_size,
        code_predictor_num_hidden_layers: tcfg.code_predictor_num_hidden_layers,
        vocab_size: tcfg.vocab_size,
    };

    let mut cp_layers = Vec::with_capacity(tcfg.code_predictor_num_hidden_layers);
    for i in 0..tcfg.code_predictor_num_hidden_layers {
        let pfx = format!("{cp_pfx}.model.layers.{i}");
        cp_layers.push(load_talker_layer(&shards, &idx, &pfx, &cp_cfg)?);
    }
    let cp_norm = load_rms_norm(
        &shards,
        &idx,
        &format!("{cp_pfx}.model.norm.weight"),
        tcfg.rms_norm_eps,
    )?;

    // 15 LM heads (indices 0..14 → groups 1..15)
    let mut lm_heads = Vec::with_capacity(15);
    for i in 0..15usize {
        let head = load_quant_linear(&shards, &idx, &format!("{cp_pfx}.lm_head.{i}"))?;
        lm_heads.push(head);
    }

    let code_pred = CodePredictor {
        codec_embeddings: cp_codec_embeddings,
        small_to_mtp,
        layers: cp_layers,
        norm: cp_norm,
        lm_heads,
    };

    Ok(TalkerModel {
        text_embed,
        codec_embed,
        text_proj_fc1,
        text_proj_fc2,
        layers,
        norm,
        codec_head,
        code_pred,
    })
}

fn load_vq_codebook(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
    d: Device,
) -> Result<VqCodebook, TtsError> {
    let emb_sum = load_tensor(shards, idx, &format!("{pfx}._codebook.embedding_sum"))?;
    let usage = load_tensor(shards, idx, &format!("{pfx}._codebook.cluster_usage"))?;
    // Normalize: embed = emb_sum / clip(usage, 1e-5, ∞)
    // Reference uses eps=1e-5 (mlx_audio sanitize). cluster_usage values are in [0.02, 0.58]
    // so they are always < 1.0 — max(usage, 1.0) would be wrong (gives raw embedding_sum).
    let eps = Array::from_f32_slice(&[1e-5f32], &[1])?;
    let usage_clamped = maximum(&usage, &eps, d)?;
    // usage: [2048], emb_sum: [2048, 256] → need to expand usage
    let usage_exp = usage_clamped.reshape(&[usage_clamped.shape()[0], 1], d)?;
    let embed = divide(&emb_sum, &usage_exp, d)?;
    Ok(VqCodebook { embed })
}

fn load_codec_conv1d(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
    padding: i32,
) -> Result<CodecConv1d, TtsError> {
    load_codec_conv1d_dil(shards, idx, key, padding, 1)
}

fn load_codec_conv1d_dil(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
    padding: i32,
    dilation: i32,
) -> Result<CodecConv1d, TtsError> {
    let w = load_tensor(shards, idx, &format!("{key}.weight"))?;
    let b = load_tensor_opt(shards, idx, &format!("{key}.bias"))?;
    // PyTorch [out, in, kernel] → MLX [out, kernel, in].
    let w_mlx = w.transpose(&[0, 2, 1], Device::Gpu)?;
    Ok(CodecConv1d {
        w: w_mlx,
        b,
        padding,
        groups: 1,
        dilation,
    })
}

fn load_codec_conv_t1d(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    key: &str,
    stride: i32,
) -> Result<CodecConvT1d, TtsError> {
    let w = load_tensor(shards, idx, &format!("{key}.weight"))?;
    let b = load_tensor_opt(shards, idx, &format!("{key}.bias"))?;
    // ConvTranspose1d weight: PyTorch [in, out, kernel] → MLX [out, kernel, in]
    // Transpose: axes [1, 2, 0]
    let w_mlx = w.transpose(&[1, 2, 0], Device::Gpu)?;
    let kernel = w_mlx.shape()[1];
    let trim_right = kernel - stride;
    Ok(CodecConvT1d {
        w: w_mlx,
        b,
        stride,
        trim_right,
    })
}

fn load_snake_beta(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
) -> Result<SnakeBeta, TtsError> {
    let alpha = load_tensor(shards, idx, &format!("{pfx}.alpha"))?;
    let beta = load_tensor(shards, idx, &format!("{pfx}.beta"))?;
    Ok(SnakeBeta { alpha, beta })
}

fn load_layer_norm(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
) -> Result<LayerNorm, TtsError> {
    let w = load_tensor(shards, idx, &format!("{pfx}.weight"))?;
    let b = load_tensor(shards, idx, &format!("{pfx}.bias"))?;
    Ok(LayerNorm { w, b, eps: 1e-6 })
}

fn load_conv_next(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
) -> Result<ConvNeXt, TtsError> {
    // dwconv: depthwise k=7, groups=C — stored as [C, 1, 7] in PyTorch → [C, 7, 1] in MLX
    let dw_w_raw = load_tensor(shards, idx, &format!("{pfx}.dwconv.conv.weight"))?;
    let dw_w = dw_w_raw.transpose(&[0, 2, 1], Device::Gpu)?;
    let dw_b = load_tensor_opt(shards, idx, &format!("{pfx}.dwconv.conv.bias"))?;
    let c = dw_w.shape()[0];
    // causal padding = kernel_size - 1 = 6
    let dw_conv = CodecConv1d {
        w: dw_w,
        b: dw_b,
        padding: 6,
        groups: c, // depthwise: groups = C (one filter per channel)
        dilation: 1,
    };
    let norm = load_layer_norm(shards, idx, &format!("{pfx}.norm"))?;
    // pwconv1: [4C, C] Linear → matmul transpose
    let pw1_w = load_tensor(shards, idx, &format!("{pfx}.pwconv1.weight"))?;
    let pw1_b = load_tensor(shards, idx, &format!("{pfx}.pwconv1.bias"))?;
    let pw2_w = load_tensor(shards, idx, &format!("{pfx}.pwconv2.weight"))?;
    let pw2_b = load_tensor(shards, idx, &format!("{pfx}.pwconv2.bias"))?;
    let gamma = load_tensor(shards, idx, &format!("{pfx}.gamma"))?;
    Ok(ConvNeXt {
        dw_conv,
        norm,
        pw1_w,
        pw1_b,
        pw2_w,
        pw2_b,
        gamma,
    })
}

fn load_res_block(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
    dilation: i32,
) -> Result<ResBlock, TtsError> {
    let act1 = load_snake_beta(shards, idx, &format!("{pfx}.act1"))?;
    // conv1: CausalConv1d k=7 dilation=dilation, causal padding=(k-1)*dilation
    let conv1 = {
        let padding = (7 - 1) * dilation;
        load_codec_conv1d_dil(shards, idx, &format!("{pfx}.conv1.conv"), padding, dilation)?
    };
    let act2 = load_snake_beta(shards, idx, &format!("{pfx}.act2"))?;
    // conv2: CausalConv1d k=1, no padding
    let conv2 = load_codec_conv1d(shards, idx, &format!("{pfx}.conv2.conv"), 0)?;
    Ok(ResBlock {
        act1,
        conv1,
        act2,
        conv2,
    })
}

fn load_decoder_group(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    group_idx: usize,
    d: Device,
) -> Result<DecoderGroup, TtsError> {
    // upsample_rates = [8, 5, 4, 3]; in_dim halves each group
    let upsample_rates = [8i32, 5, 4, 3];
    let stride = upsample_rates[group_idx];
    let pfx = format!("decoder.decoder.{}", group_idx + 1); // groups at indices 1,2,3,4

    let snake_in = load_snake_beta(shards, idx, &format!("{pfx}.block.0"))?;
    // block[1]: DecoderBlockUpsample — conv.weight, conv.bias; kernel=2*stride
    let upsample = load_codec_conv_t1d(shards, idx, &format!("{pfx}.block.1.conv"), stride)?;

    // block[2,3,4]: ResidualUnits with dilation 1, 3, 9
    let mut res_blocks = Vec::with_capacity(3);
    for (bi, dil) in [(2, 1), (3, 3), (4, 9)] {
        res_blocks.push(load_res_block(
            shards,
            idx,
            &format!("{pfx}.block.{bi}"),
            dil,
        )?);
    }

    let _ = d;
    Ok(DecoderGroup {
        snake_in,
        upsample,
        res_blocks,
    })
}

fn load_pre_trans_attn(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
) -> Result<PreTransAttn, TtsError> {
    // hidden=512, heads=16, head_dim=64
    let n_heads = 16i32;
    let head_dim = 64i32;
    let scale = (head_dim as f32).powf(-0.5);
    Ok(PreTransAttn {
        q_proj: load_plain_p(shards, idx, &format!("{pfx}.q_proj"))?,
        k_proj: load_plain_p(shards, idx, &format!("{pfx}.k_proj"))?,
        v_proj: load_plain_p(shards, idx, &format!("{pfx}.v_proj"))?,
        o_proj: load_plain_p(shards, idx, &format!("{pfx}.o_proj"))?,
        n_heads,
        head_dim,
        scale,
        rope_theta: 10000.0,
    })
}

fn load_pre_trans_mlp(
    shards: &ShardSet,
    idx: &rmlx_loader::ShardIndex,
    pfx: &str,
) -> Result<PreTransMlp, TtsError> {
    Ok(PreTransMlp {
        gate: load_plain_p(shards, idx, &format!("{pfx}.gate_proj"))?,
        up: load_plain_p(shards, idx, &format!("{pfx}.up_proj"))?,
        down: load_plain_p(shards, idx, &format!("{pfx}.down_proj"))?,
    })
}

#[cfg_attr(test, allow(dead_code))]
pub(crate) fn load_codec_decoder(codec_path: &Path) -> Result<CodecDecoder, TtsError> {
    // load_shard_index handles both multi-shard (index.json) and single-shard cases.
    let idx = load_shard_index(codec_path).map_err(|e| TtsError::Load(e.to_string()))?;
    let shards = ShardSet::open(codec_path, &idx).map_err(|e| TtsError::Load(e.to_string()))?;

    let d = Device::Gpu;

    // VQ: rvq_first has 1 codebook, rvq_rest has 15
    let rvq_first_input =
        load_codec_conv1d(&shards, &idx, "decoder.quantizer.rvq_first.input_proj", 0)?;
    let rvq_first_output =
        load_codec_conv1d(&shards, &idx, "decoder.quantizer.rvq_first.output_proj", 0)?;
    let rvq_first_cb =
        load_vq_codebook(&shards, &idx, "decoder.quantizer.rvq_first.vq.layers.0", d)?;

    let rvq_rest_input =
        load_codec_conv1d(&shards, &idx, "decoder.quantizer.rvq_rest.input_proj", 0)?;
    let rvq_rest_output =
        load_codec_conv1d(&shards, &idx, "decoder.quantizer.rvq_rest.output_proj", 0)?;

    let mut rvq_rest_cbs = Vec::with_capacity(15);
    // rvq_rest.vq.layers has indices 0..14
    for i in 0..15usize {
        rvq_rest_cbs.push(load_vq_codebook(
            &shards,
            &idx,
            &format!("decoder.quantizer.rvq_rest.vq.layers.{i}"),
            d,
        )?);
    }

    // pre_conv: k=3, 512→1024, causal padding=2
    let pre_conv = load_codec_conv1d(&shards, &idx, "decoder.pre_conv.conv", 2)?;

    // pre_transformer
    let pre_trans_input = load_plain_p(&shards, &idx, "decoder.pre_transformer.input_proj")?;
    let pre_trans_output = load_plain_p(&shards, &idx, "decoder.pre_transformer.output_proj")?;
    let pre_trans_norm = {
        let w = load_tensor(&shards, &idx, "decoder.pre_transformer.norm.weight")?;
        PreTransRmsNorm { w, eps: 1e-6 }
    };

    let mut pre_trans_layers = Vec::with_capacity(8);
    for i in 0..8usize {
        let pfx = format!("decoder.pre_transformer.layers.{i}");
        let input_ln = {
            let w = load_tensor(&shards, &idx, &format!("{pfx}.input_layernorm.weight"))?;
            PreTransRmsNorm { w, eps: 1e-6 }
        };
        let post_attn_ln = {
            let w = load_tensor(
                &shards,
                &idx,
                &format!("{pfx}.post_attention_layernorm.weight"),
            )?;
            PreTransRmsNorm { w, eps: 1e-6 }
        };
        let attn = load_pre_trans_attn(&shards, &idx, &format!("{pfx}.self_attn"))?;
        let mlp = load_pre_trans_mlp(&shards, &idx, &format!("{pfx}.mlp"))?;
        let attn_scale = load_tensor(&shards, &idx, &format!("{pfx}.self_attn_layer_scale.scale"))?;
        let mlp_scale = load_tensor(&shards, &idx, &format!("{pfx}.mlp_layer_scale.scale"))?;
        pre_trans_layers.push(PreTransLayer {
            attn,
            mlp,
            input_ln,
            post_attn_ln,
            attn_scale,
            mlp_scale,
        });
    }

    // upsample: 2 stages each = [CausalTransposeConv1d, ConvNeXt]
    // upsampling_ratios = [2, 2]
    let mut upsample = Vec::with_capacity(2);
    for i in 0..2usize {
        // upsample.{i}.0.conv: ConvTranspose1d stride=2, kernel=2*2=4
        // weight stored as [in, out, kernel]=[1024,1024,2] → MLX [out, kernel, in]=[1024,2,1024]
        let conv_t =
            load_codec_conv_t1d(&shards, &idx, &format!("decoder.upsample.{i}.0.conv"), 2)?;
        let conv_next = load_conv_next(&shards, &idx, &format!("decoder.upsample.{i}.1"))?;
        upsample.push((conv_t, conv_next));
    }

    // decoder blocks
    // decoder.decoder.0: initial conv k=7, 1024→1536, causal padding=6
    let initial_conv = load_codec_conv1d(&shards, &idx, "decoder.decoder.0.conv", 6)?;

    // decoder.decoder.{1,2,3,4}: 4 DecoderGroups with strides 8,5,4,3
    let mut groups = Vec::with_capacity(4);
    for i in 0..4usize {
        groups.push(load_decoder_group(&shards, &idx, i, d)?);
    }

    // decoder.decoder.5: OutputSnakeBeta
    let output_snake = load_snake_beta(&shards, &idx, "decoder.decoder.5")?;

    // decoder.decoder.6: OutputConv k=7, causal padding=6
    let output_conv = load_codec_conv1d(&shards, &idx, "decoder.decoder.6.conv", 6)?;

    Ok(CodecDecoder {
        rvq_first_input,
        rvq_first_output,
        rvq_first_cb,
        rvq_rest_input,
        rvq_rest_output,
        rvq_rest_cbs,
        pre_conv,
        pre_trans_input,
        pre_trans_norm,
        pre_trans_output,
        pre_trans_layers,
        upsample,
        initial_conv,
        groups,
        output_snake,
        output_conv,
    })
}

// ── TtsTokenizer ─────────────────────────────────────────────────────────────

/// Qwen3-TTS BPE tokenizer.
#[derive(Debug)]
#[allow(clippy::exhaustive_structs)]
pub struct TtsTokenizer {
    /// Filesystem path of the tokenizer directory.
    pub path: std::path::PathBuf,
    inner: tokenizers::Tokenizer,
}

impl TtsTokenizer {
    /// Load tokenizer from a directory.
    ///
    /// Accepted layouts (in priority order):
    /// 1. `tokenizer.json` — standard HuggingFace tokenizer file.
    /// 2. `vocab.json` + `merges.txt` — raw BPE layout shipped by some Qwen3-TTS snapshots.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        let p = path.as_ref().to_path_buf();
        let tok_file = p.join("tokenizer.json");
        if tok_file.exists() {
            let inner = tokenizers::Tokenizer::from_file(&tok_file)
                .map_err(|e| TtsError::Tokenizer(e.to_string()))?;
            return Ok(Self { path: p, inner });
        }
        // Fallback: build BPE tokenizer from vocab.json + merges.txt + added special tokens
        // from tokenizer_config.json (Qwen3 layout ships without a tokenizer.json).
        let vocab_file = p.join("vocab.json");
        let merges_file = p.join("merges.txt");
        if vocab_file.exists() && merges_file.exists() {
            let bpe = tokenizers::models::bpe::BPE::from_file(
                vocab_file.to_str().unwrap_or(""),
                merges_file.to_str().unwrap_or(""),
            )
            .build()
            .map_err(|e| TtsError::Tokenizer(e.to_string()))?;
            let mut inner = tokenizers::Tokenizer::new(bpe);

            // ByteLevel pre-tokenizer: required for Qwen2/3 byte-level BPE.
            // add_prefix_space=false so the first token is not prefixed with a space.
            // trim_offsets=true, use_regex=true match the standard GPT-2/Qwen3 config.
            inner.with_pre_tokenizer(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true),
            ));
            inner.with_decoder(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true),
            ));

            // Load added special tokens from tokenizer_config.json.
            // These include <|im_start|>=151644, <|im_end|>=151645, etc. that are not in
            // the base BPE vocab but are emitted by the chat template. The tokenizers crate
            // assigns added-token IDs sequentially from vocab_size (151643), so we must
            // iterate in sorted-by-ID order to get the correct mappings.
            let tc_file = p.join("tokenizer_config.json");
            if tc_file.exists() {
                if let Ok(tc_str) = std::fs::read_to_string(&tc_file) {
                    if let Ok(tc_val) = serde_json::from_str::<serde_json::Value>(&tc_str) {
                        if let Some(added) = tc_val
                            .get("added_tokens_decoder")
                            .and_then(|v| v.as_object())
                        {
                            // Collect (id, content, is_special) and sort by id.
                            let mut sorted: Vec<(u32, &str, bool)> = added
                                .iter()
                                .filter_map(|(id_str, tok_info)| {
                                    let id = id_str.parse::<u32>().ok()?;
                                    let content = tok_info.get("content")?.as_str()?;
                                    let is_special = tok_info
                                        .get("special")
                                        .and_then(serde_json::Value::as_bool)
                                        .unwrap_or(false);
                                    Some((id, content, is_special))
                                })
                                .collect();
                            sorted.sort_by_key(|(id, _, _)| *id);
                            let special_tokens: Vec<tokenizers::AddedToken> = sorted
                                .iter()
                                .map(|(_, content, is_special)| {
                                    tokenizers::AddedToken::from(*content, *is_special)
                                })
                                .collect();
                            if !special_tokens.is_empty() {
                                // tokenizers 0.21+ takes the tokens by value
                                // (`impl IntoIterator<Item = AddedToken>`) and
                                // returns the added count as a `Result`; the
                                // local Vec is not used afterwards.
                                inner
                                    .add_special_tokens(special_tokens)
                                    .map_err(|e| TtsError::Tokenizer(e.to_string()))?;
                            }
                        }
                    }
                }
            }

            return Ok(Self { path: p, inner });
        }
        Err(TtsError::Load(format!(
            "tokenizer.json (or vocab.json + merges.txt) not found at {}",
            p.display()
        )))
    }

    /// Construct a stub tokenizer backed by an empty BPE model, for testing only.
    /// The stub never tokenizes — tests that require encoding should use a real
    /// tokenizer; tests that fail before tokenization (unknown voice, load error)
    /// are safe to use this stub.
    #[cfg(test)]
    pub(crate) fn stub() -> Self {
        use std::str::FromStr as _;
        // Minimal HuggingFace tokenizer JSON accepted by tokenizers 0.20.
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{},"merges":[]}}"#;
        let inner =
            tokenizers::Tokenizer::from_str(json).expect("stub tokenizer JSON must be valid");
        Self {
            path: std::path::PathBuf::from("/tmp/tts-test"),
            inner,
        }
    }

    /// Encode text to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }
}

// ── synthesize ────────────────────────────────────────────────────────────────

/// Greedy argmax over last position, returns scalar i32 bytes.
/// `suppress`: if `Some((start, eos_id, vocab_size))`, tokens in `[start, vocab_size)` except
///   `eos_id` are suppressed (set to -inf) before argmax. Matches the reference:
///   suppress_tokens = [vocab_size-1024..vocab_size) except eos.
fn argmax_last(
    logits: &Array,
    suppress: Option<(i32, i32)>, // (suppress_start, eos_id) — vocab_size from logits shape
    d: Device,
) -> Result<u32, TtsError> {
    // logits: [1, S, vocab] → take last position → [1, vocab] → argmax → [1]
    let s = logits.shape()[1];
    let vocab = logits.shape()[2];
    let last = logits.slice(&[0, s - 1, 0], &[1, s, vocab], &[1, 1, 1], d)?;
    let last_2d = last.reshape(&[1, vocab], d)?;

    // Suppress special tokens: set [suppress_start, vocab) except eos to -inf.
    let last_2d = if let Some((suppress_start, eos_id)) = suppress {
        let neg_inf = broadcast_to(
            &scalar_f32(f32::NEG_INFINITY).reshape(&[1, 1], d)?,
            &[1, vocab],
            d,
        )?;
        // Suppress [suppress_start, eos_id) — tokens before EOS in the suppress range
        let logits_s = if suppress_start < eos_id {
            last_2d.slice_update(
                &neg_inf.slice(&[0, suppress_start], &[1, eos_id], &[1, 1], d)?,
                &[0, suppress_start],
                &[1, eos_id],
                &[1, 1],
                d,
            )?
        } else {
            last_2d.try_clone()?
        };
        // Suppress (eos_id, vocab) — tokens after EOS in the suppress range
        if eos_id + 1 < vocab {
            logits_s.slice_update(
                &neg_inf.slice(&[0, eos_id + 1], &[1, vocab], &[1, 1], d)?,
                &[0, eos_id + 1],
                &[1, vocab],
                &[1, 1],
                d,
            )?
        } else {
            logits_s
        }
    } else {
        last_2d
    };

    let tok = argmax(&last_2d, 1, d)?;
    // Materialise before reading bytes — MLX is lazy; to_bytes on an unevaluated
    // array dereferences a null data pointer and crashes.
    tok.eval()?;
    let bytes = tok.to_bytes()?;
    if bytes.len() < 4 {
        return Err(TtsError::Mlx("argmax_last: unexpected byte length".into()));
    }
    Ok(i32::from_le_bytes(bytes[..4].try_into().unwrap_or([0u8; 4])) as u32)
}

/// Look up embedding for a single token id in a table. Returns [1, 1, hidden].
fn lookup_embed(table: &Embedding, token_id: u32, d: Device) -> Result<Array, TtsError> {
    let ids = Array::from_i32_slice(&[token_id as i32], &[1, 1])?;
    table.forward(&ids, d)
}

/// Synthesize speech from `text` using `voice`.
/// Returns `(samples, sample_rate)` where samples is mono f32 PCM at 24000 Hz.
#[allow(clippy::cognitive_complexity)]
#[instrument(skip(model, tokenizer), fields(voice, text_len = text.len()))]
pub fn synthesize(
    text: &str,
    voice: &str,
    model: &mut TtsModel,
    tokenizer: &TtsTokenizer,
) -> Result<(Vec<f32>, u32), TtsError> {
    if text.trim().is_empty() {
        return Err(TtsError::Empty);
    }
    let spk_id = model
        .config
        .speaker_id(voice)
        .ok_or_else(|| TtsError::UnknownVoice(voice.to_owned()))?;

    // Load weights on first call
    if model.talker.is_none() {
        model.load()?;
    }

    let talker = model
        .talker
        .as_ref()
        .ok_or_else(|| TtsError::Load("talker not loaded after model.load()".into()))?;
    let codec = model
        .codec
        .as_ref()
        .ok_or_else(|| TtsError::Load("codec not loaded after model.load()".into()))?;
    let tcfg = &model.config.talker_config;
    let d = Device::Gpu;
    let t_start = Instant::now();

    // ── Build input embeddings (matches _prepare_generation_inputs custom_voice path) ──

    // Chat template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    let chat_text = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
    let text_ids = tokenizer.encode(&chat_text)?;
    let text_ids_arr = Array::from_i32_slice(
        &text_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        &[1, text_ids.len() as i32],
    )?;

    // text_embed: [1, S, hidden]
    let text_embed = talker.embed_text(&text_ids_arr, d)?;

    // TTS special tokens: [bos, eos, pad]
    let special_ids = Array::from_i32_slice(
        &[
            model.config.tts_bos_token_id as i32,
            model.config.tts_eos_token_id as i32,
            model.config.tts_pad_token_id as i32,
        ],
        &[1, 3],
    )?;
    let special_embeds = talker.embed_text(&special_ids, d)?;
    // [1, 1, hidden] each
    let tts_bos =
        special_embeds.slice(&[0, 0, 0], &[1, 1, tcfg.hidden_size as i32], &[1, 1, 1], d)?;
    let tts_eos =
        special_embeds.slice(&[0, 1, 0], &[1, 2, tcfg.hidden_size as i32], &[1, 1, 1], d)?;
    let tts_pad =
        special_embeds.slice(&[0, 2, 0], &[1, 3, tcfg.hidden_size as i32], &[1, 1, 1], d)?;

    // Speaker embedding from codec_embed table
    let spk_ids = Array::from_i32_slice(&[spk_id as i32], &[1, 1])?;
    let speaker_embed = talker.codec_embed.forward(&spk_ids, d)?; // [1, 1, hidden]

    // Codec prefix: [nothink, think_bos, think_eos]
    let codec_prefill_ids = Array::from_i32_slice(
        &[
            tcfg.codec_nothink_id as i32,
            tcfg.codec_think_bos_id as i32,
            tcfg.codec_think_eos_id as i32,
        ],
        &[1, 3],
    )?;
    let codec_prefix_embed = talker.codec_embed.forward(&codec_prefill_ids, d)?; // [1, 3, hidden]

    // codec_embed_suffix: [pad, bos]
    let suffix_ids = Array::from_i32_slice(
        &[tcfg.codec_pad_id as i32, tcfg.codec_bos_id as i32],
        &[1, 2],
    )?;
    let codec_suffix = talker.codec_embed.forward(&suffix_ids, d)?; // [1, 2, hidden]

    // Full codec_embed: prefix + speaker + suffix = [1, 6, hidden]
    let codec_embed_full =
        concatenate(&[&codec_prefix_embed, &speaker_embed, &codec_suffix], 1, d)?;

    // Build combined prefix: tts_pad * (prefix_len - 2) + tts_bos
    // prefix_len = codec_embed_full.shape[1] = 6
    let prefix_len = codec_embed_full.shape()[1];
    let pad_count = prefix_len - 2;
    let h = tcfg.hidden_size as i32;
    // Broadcast tts_pad to [1, pad_count, hidden]
    let pad_tiles: Vec<Array> = (0..pad_count)
        .map(|_| tts_pad.try_clone())
        .collect::<Result<Vec<_>, _>>()?;
    let pad_tiles_refs: Vec<&Array> = pad_tiles.iter().collect();
    let pad_embeds = concatenate(&pad_tiles_refs, 1, d)?; // [1, pad_count, hidden]

    let combined = concatenate(&[&pad_embeds, &tts_bos], 1, d)?; // [1, pad_count+1, hidden]
                                                                 // combined = combined + codec_embed_full[:, :-1, :]
    let codec_prefix_slice =
        codec_embed_full.slice(&[0, 0, 0], &[1, prefix_len - 1, h], &[1, 1, 1], d)?;
    let combined = add(&combined, &codec_prefix_slice, d)?;

    // role_embed: text_embed[:, :3, :]
    let role_embed = text_embed.slice(&[0, 0, 0], &[1, 3, h], &[1, 1, 1], d)?;

    // first text token: text_embed[:, 3:4, :] + codec_embed_full[:, -1:, :]
    let first_text = text_embed.slice(&[0, 3, 0], &[1, 4, h], &[1, 1, 1], d)?;
    let codec_last =
        codec_embed_full.slice(&[0, prefix_len - 1, 0], &[1, prefix_len, h], &[1, 1, 1], d)?;
    let first_text_combined = add(&first_text, &codec_last, d)?;

    // input_embeds: role + combined + first_text_combined
    let input_embeds = concatenate(&[&role_embed, &combined, &first_text_combined], 1, d)?;

    // trailing_text_hidden: text_embed[:, 4:-5, :] + tts_eos
    let text_len = text_embed.shape()[1];
    let trailing_text = text_embed.slice(&[0, 4, 0], &[1, text_len - 5, h], &[1, 1, 1], d)?;
    // tts_eos needs to be 3D: [1, 1, h]
    let tts_eos_3d = tts_eos.slice(&[0, 0, 0], &[1, 1, h], &[1, 1, 1], d)?;
    let trailing_text_hidden = concatenate(&[&trailing_text, &tts_eos_3d], 1, d)?;

    // ── Generation loop ────────────────────────────────────────────────────────

    let max_tokens = 2048usize;
    let eos_id = tcfg.codec_eos_token_id;
    // Suppress special tokens [vocab_size-1024, vocab_size) except EOS.
    // Reference: suppress_tokens = [vocab_size-1024..vocab_size) - {eos_id}
    let suppress_start = tcfg.vocab_size as i32 - 1024;
    let suppress = Some((suppress_start, eos_id as i32));

    // KV caches: one per talker layer, one set per CP layer
    let n_talker = tcfg.num_hidden_layers;
    let n_cp = tcfg.code_predictor_num_hidden_layers;
    let mut talker_kv: Vec<KvSlot> = (0..n_talker).map(|_| KvSlot::new()).collect();
    let mut cp_kv: Vec<KvSlot> = (0..n_cp).map(|_| KvSlot::new()).collect();

    let mut generated_codes: Vec<Array> = Vec::with_capacity(max_tokens);
    let mut trailing_idx = 0usize;
    let mut talker_offset = 0i32;

    let mut cur_input = input_embeds;

    debug!("starting Qwen3-TTS generation loop, max_tokens={max_tokens}");

    for step in 0..max_tokens {
        let seq_len = cur_input.shape()[1];

        // Talker forward
        let (logits, hidden) = talker.forward(&cur_input, &mut talker_kv, talker_offset, d)?;
        talker_offset += seq_len;

        // Argmax with special-token suppression (tokens [vocab-1024, vocab) except EOS).
        let next_tok_id = argmax_last(&logits, suppress, d)?;

        if next_tok_id == eos_id {
            debug!(step, "TTS EOS received");
            break;
        }

        // CodePredictor: generate groups 1..15
        let hidden_last = hidden.slice(
            &[0, hidden.shape()[1] - 1, 0],
            &[1, hidden.shape()[1], h],
            &[1, 1, 1],
            d,
        )?;

        // Project talker hidden [1,1,2048] → CP hidden [1,1,1024] via small_to_mtp.
        let hidden_proj = talker.code_pred.small_to_mtp.forward(&hidden_last, d)?;

        // Group-0 codec embedding: use talker.codec_embed (same table as next-step input build),
        // NOT cp.codec_embeddings[0]. Then project 2048→1024 via small_to_mtp.
        // Reference: code_0_embed = self.talker.get_input_embeddings()(next_token)
        let code_0_emb_raw = lookup_embed(&talker.codec_embed, next_tok_id, d)?; // [1,1,2048]
        let code_0_emb_cp = talker.code_pred.small_to_mtp.forward(&code_0_emb_raw, d)?; // [1,1,1024]

        // Reset CP KV cache each step
        for kv in &mut cp_kv {
            kv.k = None;
            kv.v = None;
        }

        let mut code_tokens: Vec<u32> = vec![next_tok_id];
        let mut cp_offset = 0i32;

        for code_idx in 0..15usize {
            let cp_input = if code_idx == 0 {
                // Step 0: concat(hidden_proj [1,1,1024], code_0_emb_cp [1,1,1024]) → [1,2,1024]
                // (small_to_mtp applied to both halves individually = same as applying to the
                //  concatenated [1,2,2048] input, since projection distributes over axis 1)
                concatenate(&[&hidden_proj, &code_0_emb_cp], 1, d)?
            } else {
                // For groups 1..14: embed previous group token via CP codec_embedding[code_idx-1]
                // (2048-dim), then project to CP space.
                // Reference: cp.codec_embedding[code_idx - 1](last_token)
                let last_code = *code_tokens.last().ok_or_else(|| {
                    TtsError::Mlx("code_tokens empty during predictor step".into())
                })?;
                let emb_ids = Array::from_i32_slice(&[last_code as i32], &[1, 1])?;
                let raw = talker.code_pred.codec_embeddings[code_idx - 1].forward(&emb_ids, d)?; // [1,1,2048]
                talker.code_pred.small_to_mtp.forward(&raw, d)? // [1,1,1024]
            };

            let cp_logits = talker
                .code_pred
                .forward(&cp_input, &mut cp_kv, cp_offset, code_idx, d)?;
            cp_offset += cp_input.shape()[1];
            // CP vocab = 2048, all valid audio tokens — no suppression needed.
            let next_code = argmax_last(&cp_logits, None, d)?;
            code_tokens.push(next_code);
        }

        // Stack all 16 codec group tokens for this step: [1, 16]
        let code_arr = Array::from_i32_slice(
            &code_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1, 16],
        )?;
        generated_codes.push(code_arr);

        // Build next input_embeds for talker
        let text_emb = if trailing_idx < trailing_text_hidden.shape()[1] as usize {
            let te = trailing_text_hidden.slice(
                &[0, trailing_idx as i32, 0],
                &[1, trailing_idx as i32 + 1, h],
                &[1, 1, 1],
                d,
            )?;
            trailing_idx += 1;
            te
        } else {
            tts_pad.slice(&[0, 0, 0], &[1, 1, h], &[1, 1, 1], d)?
        };

        // Next-step talker input: sum codec group embeddings.
        // Group 0: talker.codec_embed (vocab=3072, dim=2048)
        // Groups 1..15: cp.codec_embeddings[i-1] (vocab=2048, dim=2048) — each group has
        //               its own embedding table in the CodePredictor.
        // Reference: talker.get_input_embeddings()(group_0) + sum(cp.codec_embedding[i](group_i+1))
        let codec_emb_0 = lookup_embed(&talker.codec_embed, code_tokens[0], d)?;
        let mut codec_emb = codec_emb_0;
        for (i, &ct) in code_tokens[1..].iter().enumerate() {
            let extra = talker.code_pred.codec_embeddings[i]
                .forward(&Array::from_i32_slice(&[ct as i32], &[1, 1])?, d)?;
            codec_emb = add(&codec_emb, &extra, d)?;
        }

        cur_input = add(&text_emb, &codec_emb, d)?;

        if step % 50 == 0 && step > 0 {
            debug!(step, "TTS generation step");
        }
    }

    let gen_ms = t_start.elapsed().as_millis();
    let n_frames = generated_codes.len();
    info!(
        frames = n_frames,
        elapsed_ms = gen_ms,
        "TTS generation done"
    );

    if generated_codes.is_empty() {
        return Err(TtsError::Mlx(
            "no audio generated (EOS on first step)".into(),
        ));
    }

    // Decode: stack codes [1, T, 16] → transpose → [1, 16, T]
    // Each generated_codes[i] is [1, 16] (2D). Reshape to [1, 1, 16] then concatenate
    // along axis 1 → [1, T, 16], then transpose → [1, 16, T].
    let codes_3d_parts: Vec<Array> = generated_codes
        .iter()
        .map(|a| a.reshape(&[1, 1, 16], d))
        .collect::<Result<Vec<_>, _>>()?;
    let codes_3d_refs: Vec<&Array> = codes_3d_parts.iter().collect();
    let codes_3d = concatenate(&codes_3d_refs, 1, d)?; // [1, T, 16]
                                                       // Transpose → [1, 16, T] for codec decoder
    let codes_t = codes_3d.transpose(&[0, 2, 1], d)?;

    let t_dec = Instant::now();
    let samples = codec.decode(&codes_t, d)?;
    info!(
        samples = samples.len(),
        codec_ms = t_dec.elapsed().as_millis(),
        "codec decode done"
    );

    Ok((samples, 24000))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tts_tests.rs"]
mod tts_tests;
