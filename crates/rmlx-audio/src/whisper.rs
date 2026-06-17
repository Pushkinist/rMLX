// LOC-exempt: encoder + decoder + greedy_decode form one tightly coupled MLX graph;
// the attention + residual + FFN blocks are inseparable (shared KV-cache lifetimes).
// The NPZ loader was extracted to crates/rmlx-audio/src/npz.rs.

//! Whisper encoder + decoder model layers.
//!
//! ## Architecture (large-v3)
//!
//! From `config.json`:
//! ```json
//! { "n_mels": 128, "n_audio_ctx": 1500, "n_audio_state": 1280,
//!   "n_audio_head": 20, "n_audio_layer": 32,
//!   "n_vocab": 51866, "n_text_ctx": 448, "n_text_state": 1280,
//!   "n_text_head": 20, "n_text_layer": 32 }
//! ```
//!
//! ## Weight layout (`weights.npz`)
//!
//! Keys follow the mlx-community naming convention:
//! - `encoder.conv1.{weight,bias}` — Conv1d stem (k=3, stride=1, pad=1)
//! - `encoder.conv2.{weight,bias}` — Conv1d stem (k=3, stride=2, pad=1)
//! - `encoder.blocks.N.attn.{query,key,value,out}.{weight,bias}` — MHA
//! - `encoder.blocks.N.attn_ln.{weight,bias}` — pre-attn LayerNorm
//! - `encoder.blocks.N.mlp{1,2}.{weight,bias}` — FFN
//! - `encoder.blocks.N.mlp_ln.{weight,bias}` — pre-FFN LayerNorm
//! - `encoder.ln_post.{weight,bias}` — post-encoder LayerNorm
//! - `decoder.token_embedding.weight` — token embedding
//! - `decoder.positional_embedding` — positional embedding (no grad key)
//! - `decoder.blocks.N.attn.*`, `decoder.blocks.N.cross_attn.*`
//! - `decoder.blocks.N.attn_ln.*`, `decoder.blocks.N.cross_attn_ln.*`
//! - `decoder.blocks.N.mlp{1,2}.*`, `decoder.blocks.N.mlp_ln.*`
//! - `decoder.ln.{weight,bias}` — final decoder LayerNorm
//!
//! ## Conv1d weight layout
//!
//! The `.npz` stores Conv1d weights as `[out_channels, kernel, in_channels]`.
//! MLX `conv1d` expects `[out, in, kernel]`. We transpose on load:
//! `(out, k, in) → (out, in, k)`.
//!
//! ## Positional embedding
//!
//! The encoder positional embedding is computed as fixed sinusoids (not stored)
//! and added inside the encoder forward pass. The decoder positional embedding
//! IS stored as `decoder.positional_embedding` of shape `[n_text_ctx, n_state]`.
//!
//! ## Inference loop
//!
//! Uses greedy decoding (argmax) by default. Temperature scaling is applied
//! when `temperature > 0`. KV-cache is maintained per-block for O(n) decode.

use std::path::Path;

use rmlx_mlx::{
    add, argmax, concatenate, conv1d, divide, gelu, matmul, multiply, scalar_f32, softmax_precise,
    sqrt, subtract, sum_axis_keepdims, Array, Device,
};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

use crate::npz::WeightMap;
use crate::tokenizer::{
    TOK_EN, TOK_EOT, TOK_LANG_LAST, TOK_NOSPEECH, TOK_NO_TIMESTAMPS, TOK_TIMESTAMP_BEGIN,
};

// ── Errors ────────────────────────────────────────────────────────────────────

/// Whisper model errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum WhisperError {
    /// Weight file could not be read.
    #[error("weight file error: {0}")]
    WeightFile(String),
    /// A required weight key was not found.
    #[error("missing weight key: {0}")]
    MissingWeight(String),
    /// MLX operation failed.
    #[error("MLX op error: {0}")]
    Mlx(String),
    /// Config JSON parse error.
    #[error("config parse error: {0}")]
    Config(String),
    /// No transcription produced.
    #[error("no transcription (model detected silence)")]
    Silence,
}

impl From<rmlx_core::error::Error> for WhisperError {
    fn from(e: rmlx_core::error::Error) -> Self {
        Self::Mlx(e.to_string())
    }
}

impl From<crate::npz::NpzError> for WhisperError {
    fn from(e: crate::npz::NpzError) -> Self {
        Self::WeightFile(e.to_string())
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Parsed Whisper `config.json`.
#[allow(
    clippy::exhaustive_structs,
    reason = "WhisperConfig is a fixed JSON schema; field set mirrors the Whisper config.json spec"
)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WhisperConfig {
    /// Number of mel bins (80 or 128).
    pub n_mels: usize,
    /// Audio context length in tokens (1500 for large-v3).
    pub n_audio_ctx: usize,
    /// Audio hidden state dimension.
    pub n_audio_state: usize,
    /// Number of audio attention heads.
    pub n_audio_head: usize,
    /// Number of audio encoder layers.
    pub n_audio_layer: usize,
    /// Vocabulary size (51 866 for large-v3).
    pub n_vocab: usize,
    /// Text context length (448 for large-v3).
    pub n_text_ctx: usize,
    /// Text hidden state dimension.
    pub n_text_state: usize,
    /// Number of text attention heads.
    pub n_text_head: usize,
    /// Number of text decoder layers.
    pub n_text_layer: usize,
}

impl WhisperConfig {
    /// Load from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, WhisperError> {
        serde_json::from_str(s).map_err(|e| WhisperError::Config(e.to_string()))
    }
}

// ── Weight map ────────────────────────────────────────────────────────────────

/// Load all weights from a `.npz` file into a weight map.
///
/// Whisper snapshots from mlx-community ship as `weights.npz` (NumPy
/// compressed array archive = ZIP of `.npy` entries). ZIP64 archives
/// (≥ 4 GiB, as produced by the large-v3 snapshot) are fully supported via
/// the central-directory parser in `crate::npz`.
pub fn load_npz(path: impl AsRef<Path>) -> Result<WeightMap, WhisperError> {
    crate::npz::load_npz(path).map_err(WhisperError::from)
}

// ── Weight helpers ────────────────────────────────────────────────────────────

fn get_weight(map: &WeightMap, key: &str) -> Result<Array, WhisperError> {
    map.get(key)
        .ok_or_else(|| WhisperError::MissingWeight(key.to_owned()))
        .and_then(|a| a.try_clone().map_err(|e| WhisperError::Mlx(e.to_string())))
}

fn get_weight_opt(map: &WeightMap, key: &str) -> Result<Option<Array>, WhisperError> {
    match map.get(key) {
        Some(a) => Ok(Some(
            a.try_clone()
                .map_err(|e| WhisperError::Mlx(e.to_string()))?,
        )),
        None => Ok(None),
    }
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

#[allow(missing_debug_implementations)]
struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    fn load(map: &WeightMap, prefix: &str) -> Result<Self, WhisperError> {
        Ok(Self {
            weight: get_weight(map, &format!("{prefix}.weight"))?,
            bias: get_weight(map, &format!("{prefix}.bias"))?,
            eps: 1e-5,
        })
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "x.ndim() >= 1 checked implicitly: any meaningful tensor has at least 1 dim"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array, WhisperError> {
        let last = (x.ndim() - 1) as i32;
        let dim = x.shape()[x.ndim() - 1] as f32;
        let inv_n = scalar_f32(1.0 / dim);

        let mean = multiply(&sum_axis_keepdims(x, last, device)?, &inv_n, device)?;
        let centered = subtract(x, &mean, device)?;
        let sq = multiply(&centered, &centered, device)?;
        let var = multiply(&sum_axis_keepdims(&sq, last, device)?, &inv_n, device)?;
        let denom = sqrt(&add(&var, &scalar_f32(self.eps), device)?, device)?;
        let normed = divide(&centered, &denom, device)?;
        let scaled = multiply(&normed, &self.weight, device)?;
        add(&scaled, &self.bias, device).map_err(WhisperError::from)
    }
}

// ── Linear ────────────────────────────────────────────────────────────────────

#[allow(missing_debug_implementations)]
struct Linear {
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    fn load(map: &WeightMap, prefix: &str, with_bias: bool) -> Result<Self, WhisperError> {
        let weight = get_weight(map, &format!("{prefix}.weight"))?;
        let bias = if with_bias {
            get_weight_opt(map, &format!("{prefix}.bias"))?
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Array, device: Device) -> Result<Array, WhisperError> {
        let wt = self.weight.transpose(&[1, 0], device)?;
        let out = matmul(x, &wt, device)?;
        match &self.bias {
            Some(b) => add(&out, b, device).map_err(WhisperError::from),
            None => Ok(out),
        }
    }
}

// ── MultiHeadAttention ────────────────────────────────────────────────────────

#[allow(missing_debug_implementations)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn load(map: &WeightMap, prefix: &str, n_head: usize) -> Result<Self, WhisperError> {
        Ok(Self {
            query: Linear::load(map, &format!("{prefix}.query"), true)?,
            key: Linear::load(map, &format!("{prefix}.key"), false)?,
            value: Linear::load(map, &format!("{prefix}.value"), true)?,
            out: Linear::load(map, &format!("{prefix}.out"), true)?,
            n_head,
        })
    }

    fn forward(
        &self,
        x: &Array,
        xa: Option<&Array>,
        kv_cache: Option<(&Array, &Array)>,
        mask: Option<&Array>,
        device: Device,
    ) -> Result<(Array, (Array, Array)), WhisperError> {
        let q = self.query.forward(x, device)?;

        let (k, v) = if let Some(xa_ref) = xa {
            if let Some((ck, cv)) = kv_cache {
                (ck.try_clone()?, cv.try_clone()?)
            } else {
                (
                    self.key.forward(xa_ref, device)?,
                    self.value.forward(xa_ref, device)?,
                )
            }
        } else {
            let k_new = self.key.forward(x, device)?;
            let v_new = self.value.forward(x, device)?;
            if let Some((ck, cv)) = kv_cache {
                (
                    concatenate(&[ck, &k_new], 1, device)?,
                    concatenate(&[cv, &v_new], 1, device)?,
                )
            } else {
                (k_new, v_new)
            }
        };

        let wv = self.qkv_attention(&q, &k, &v, mask, device)?;
        let out = self.out.forward(&wv, device)?;
        Ok((out, (k, v)))
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "shape dims validated: q is [B, Tq, D]; head_dim = D / n_head, all indices bounded"
    )]
    fn qkv_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        mask: Option<&Array>,
        device: Device,
    ) -> Result<Array, WhisperError> {
        let q_shape = q.shape();
        let n_batch = q_shape[0];
        let n_ctx = q_shape[1];
        let n_state = q_shape[2];
        let head_dim = n_state / self.n_head as i32;
        let scale = scalar_f32((head_dim as f32).powf(-0.25));

        let q = q.reshape(&[n_batch, n_ctx, self.n_head as i32, head_dim], device)?;
        let q = q.transpose(&[0, 2, 1, 3], device)?;
        let q = multiply(&q, &scale, device)?;

        let k_shape = k.shape();
        let n_ctx_k = k_shape[1];
        let k = k.reshape(&[n_batch, n_ctx_k, self.n_head as i32, head_dim], device)?;
        let k = k.transpose(&[0, 2, 3, 1], device)?;
        let k = multiply(&k, &scale, device)?;

        let v = v.reshape(&[n_batch, n_ctx_k, self.n_head as i32, head_dim], device)?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        let mut qk = matmul(&q, &k, device)?;

        if let Some(m) = mask {
            // Slice mask [n_text_ctx, n_text_ctx] → [n_ctx, n_ctx].
            //
            // Match Python: `qk = qk + mask[:n_ctx, :n_ctx]`  (Python multi-head attention).
            // During prefill n_ctx == n_ctx_k so this is the full square causal mask.
            // During single-token incremental decode n_ctx == 1, so we get [[0]] which
            // broadcasts to all key positions — effectively no masking (correct, since a
            // causal model may attend to any prior token when generating one new token).
            // Slicing with n_ctx_k instead of n_ctx would incorrectly mask out previously
            // accumulated keys (causing wrong self-attention and garbage decoder output).
            let m_slice = m.slice(&[0, 0], &[n_ctx, n_ctx], &[1, 1], device)?;
            qk = add(&qk, &m_slice, device)?;
        }

        // Use precise softmax (matching Python's `mx.softmax(qk, axis=-1, precise=True)`).
        let w = softmax_precise(&qk, -1, device)?;
        let out = matmul(&w, &v, device)?;
        let out = out.transpose(&[0, 2, 1, 3], device)?;
        out.reshape(&[n_batch, n_ctx, n_state], device)
            .map_err(WhisperError::from)
    }
}

// ── EncoderBlock ──────────────────────────────────────────────────────────────

#[allow(missing_debug_implementations)]
struct EncoderBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

impl EncoderBlock {
    fn load(map: &WeightMap, prefix: &str, n_head: usize) -> Result<Self, WhisperError> {
        Ok(Self {
            attn: MultiHeadAttention::load(map, &format!("{prefix}.attn"), n_head)?,
            attn_ln: LayerNorm::load(map, &format!("{prefix}.attn_ln"))?,
            mlp1: Linear::load(map, &format!("{prefix}.mlp1"), true)?,
            mlp2: Linear::load(map, &format!("{prefix}.mlp2"), true)?,
            mlp_ln: LayerNorm::load(map, &format!("{prefix}.mlp_ln"))?,
        })
    }

    fn forward(&self, x: &Array, device: Device) -> Result<Array, WhisperError> {
        let norm = self.attn_ln.forward(x, device)?;
        let (attn_out, _) = self.attn.forward(&norm, None, None, None, device)?;
        let x = add(x, &attn_out, device)?;

        let norm2 = self.mlp_ln.forward(&x, device)?;
        let ff = self
            .mlp2
            .forward(&gelu(&self.mlp1.forward(&norm2, device)?, device)?, device)?;
        add(&x, &ff, device).map_err(WhisperError::from)
    }
}

// ── DecoderBlock ──────────────────────────────────────────────────────────────

#[allow(missing_debug_implementations)]
struct DecoderBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: MultiHeadAttention,
    cross_attn_ln: LayerNorm,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

impl DecoderBlock {
    fn load(map: &WeightMap, prefix: &str, n_head: usize) -> Result<Self, WhisperError> {
        Ok(Self {
            attn: MultiHeadAttention::load(map, &format!("{prefix}.attn"), n_head)?,
            attn_ln: LayerNorm::load(map, &format!("{prefix}.attn_ln"))?,
            cross_attn: MultiHeadAttention::load(map, &format!("{prefix}.cross_attn"), n_head)?,
            cross_attn_ln: LayerNorm::load(map, &format!("{prefix}.cross_attn_ln"))?,
            mlp1: Linear::load(map, &format!("{prefix}.mlp1"), true)?,
            mlp2: Linear::load(map, &format!("{prefix}.mlp2"), true)?,
            mlp_ln: LayerNorm::load(map, &format!("{prefix}.mlp_ln"))?,
        })
    }

    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        x: &Array,
        xa: &Array,
        self_kv: Option<(&Array, &Array)>,
        cross_kv: Option<(&Array, &Array)>,
        mask: Option<&Array>,
        device: Device,
    ) -> Result<(Array, (Array, Array), (Array, Array)), WhisperError> {
        let norm = self.attn_ln.forward(x, device)?;
        let (attn_out, new_self_kv) = self.attn.forward(&norm, None, self_kv, mask, device)?;
        let x = add(x, &attn_out, device)?;

        let norm2 = self.cross_attn_ln.forward(&x, device)?;
        let (cross_out, new_cross_kv) =
            self.cross_attn
                .forward(&norm2, Some(xa), cross_kv, None, device)?;
        let x = add(&x, &cross_out, device)?;

        let norm3 = self.mlp_ln.forward(&x, device)?;
        let ff = self
            .mlp2
            .forward(&gelu(&self.mlp1.forward(&norm3, device)?, device)?, device)?;
        let x = add(&x, &ff, device)?;

        Ok((x, new_self_kv, new_cross_kv))
    }
}

// ── AudioEncoder ──────────────────────────────────────────────────────────────

/// Whisper audio encoder (Conv1d stem + sinusoidal position + 32 attention blocks).
#[allow(missing_debug_implementations)]
pub struct AudioEncoder {
    conv1_w: Array,
    conv1_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    blocks: Vec<EncoderBlock>,
    ln_post: LayerNorm,
    positional_embedding: Array,
}

impl AudioEncoder {
    fn load(map: &WeightMap, cfg: &WhisperConfig) -> Result<Self, WhisperError> {
        // Conv1d: npz stores [out, k, in]; MLX conv1d weight layout is [out, k, in].
        // No transpose needed — load directly.
        let conv1_w = get_weight(map, "encoder.conv1.weight")?;
        let conv1_b = get_weight(map, "encoder.conv1.bias")?;
        let conv2_w = get_weight(map, "encoder.conv2.weight")?;
        let conv2_b = get_weight(map, "encoder.conv2.bias")?;

        let blocks: Vec<EncoderBlock> = (0..cfg.n_audio_layer)
            .map(|i| EncoderBlock::load(map, &format!("encoder.blocks.{i}"), cfg.n_audio_head))
            .collect::<Result<_, _>>()?;

        let ln_post = LayerNorm::load(map, "encoder.ln_post")?;
        let positional_embedding = build_sinusoids(cfg.n_audio_ctx, cfg.n_audio_state)?;

        info!(n_audio_layer = cfg.n_audio_layer, "AudioEncoder loaded");
        Ok(Self {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            blocks,
            ln_post,
            positional_embedding,
        })
    }

    /// Run encoder on log-mel `[1, T, n_mels]` → `[1, T/2, n_state]`.
    #[instrument(skip(self, mel), fields(shape = ?mel.shape()), level = "debug")]
    pub fn forward(&self, mel: &Array, device: Device) -> Result<Array, WhisperError> {
        // Conv1 k=3, pad=1, stride=1.
        let x = add(
            &conv1d(mel, &self.conv1_w, 1, 1, 1, 1, device)?,
            &self.conv1_b,
            device,
        )?;
        let x = gelu(&x, device)?;
        // Conv2 k=3, pad=1, stride=2, dilation=1.
        let x = add(
            &conv1d(&x, &self.conv2_w, 2, 1, 1, 1, device)?,
            &self.conv2_b,
            device,
        )?;
        let x = gelu(&x, device)?;

        let x = add(&x, &self.positional_embedding, device)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, device)?;
        }
        self.ln_post.forward(&x, device)
    }
}

/// Build sinusoidal positional embeddings `[1, n_ctx, n_state]` (f32).
///
/// Matches `mlx_whisper.whisper.sinusoids`:
/// ```python
/// log_timescale_increment = log(max_timescale) / (channels // 2 - 1)
/// inv_timescales = exp(-log_timescale_increment * arange(channels // 2))
/// scaled_time = arange(length)[:, None] * inv_timescales[None, :]
/// return cat([sin(scaled_time), cos(scaled_time)], axis=1)
/// ```
fn build_sinusoids(n_ctx: usize, n_state: usize) -> Result<Array, WhisperError> {
    let half = n_state / 2;
    let max_timescale: f64 = 10_000.0;
    let log_timescale_increment = max_timescale.ln() / (half as f64 - 1.0);
    let mut data = vec![0.0_f32; n_ctx * n_state];
    #[allow(
        clippy::indexing_slicing,
        reason = "t in [0, n_ctx), i in [0, half); offset t*n_state+i and t*n_state+half+i both < n_ctx*n_state"
    )]
    for t in 0..n_ctx {
        for i in 0..half {
            let inv_timescale = (-(i as f64) * log_timescale_increment).exp() as f32;
            let angle = t as f32 * inv_timescale;
            data[t * n_state + i] = angle.sin();
            data[t * n_state + half + i] = angle.cos();
        }
    }
    Array::from_f32_slice(&data, &[1, n_ctx as i32, n_state as i32])
        .map_err(|e| WhisperError::Mlx(e.to_string()))
}

// ── TextDecoder ───────────────────────────────────────────────────────────────

/// Whisper text decoder (token embedding + positional + 32 decoder blocks).
#[allow(missing_debug_implementations)]
pub struct TextDecoder {
    token_embedding: Array,
    positional_embedding: Array,
    blocks: Vec<DecoderBlock>,
    ln: LayerNorm,
    causal_mask: Array,
    n_state: usize,
}

impl TextDecoder {
    fn load(map: &WeightMap, cfg: &WhisperConfig) -> Result<Self, WhisperError> {
        let token_embedding = get_weight(map, "decoder.token_embedding.weight")?;
        let positional_embedding = get_weight(map, "decoder.positional_embedding")?;
        let blocks: Vec<DecoderBlock> = (0..cfg.n_text_layer)
            .map(|i| DecoderBlock::load(map, &format!("decoder.blocks.{i}"), cfg.n_text_head))
            .collect::<Result<_, _>>()?;
        let ln = LayerNorm::load(map, "decoder.ln")?;
        let causal_mask = build_causal_mask(cfg.n_text_ctx)?;

        info!(
            n_text_layer = cfg.n_text_layer,
            n_vocab = cfg.n_vocab,
            "TextDecoder loaded"
        );
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            causal_mask,
            n_state: cfg.n_text_state,
        })
    }

    /// Forward one step.
    ///
    /// `tokens [1, T_new]`, `xa [1, T_audio, D]`, `offset` = tokens decoded so far.
    /// Returns `(logits [1, T_new, n_vocab], new_self_kvs, new_cross_kvs)`.
    #[allow(
        clippy::type_complexity,
        clippy::indexing_slicing,
        reason = "tok_shape[1]: tokens is always shaped [1, T] by construction; T is at index 1"
    )]
    fn forward(
        &self,
        tokens: &Array,
        xa: &Array,
        offset: usize,
        self_kvs: &[(Array, Array)],
        cross_kvs: &[(Array, Array)],
        device: Device,
    ) -> Result<(Array, Vec<(Array, Array)>, Vec<(Array, Array)>), WhisperError> {
        let tok_shape = tokens.shape();
        let seq_len = tok_shape[1] as usize;

        // Token embedding: gather rows from token_embedding [n_vocab, n_state].
        // Array::take(indices, axis, device): takes rows of `self` at `indices`.
        let flat_tokens = tokens.reshape(&[-1], device)?;
        let x = self
            .token_embedding
            .take(&flat_tokens, 0, device)?
            .reshape(&[1, seq_len as i32, self.n_state as i32], device)?;

        // Positional embedding: [n_text_ctx, n_state] → slice rows [offset, offset+seq_len).
        let pos = self.positional_embedding.slice(
            &[offset as i32, 0],
            &[(offset + seq_len) as i32, self.n_state as i32],
            &[1, 1],
            device,
        )?;
        let pos = pos.reshape(&[1, seq_len as i32, self.n_state as i32], device)?;
        let mut x = add(&x, &pos, device)?;

        let mut new_self_kvs: Vec<(Array, Array)> = Vec::with_capacity(self.blocks.len());
        let mut new_cross_kvs: Vec<(Array, Array)> = Vec::with_capacity(self.blocks.len());

        for (i, block) in self.blocks.iter().enumerate() {
            let self_kv = self_kvs.get(i).map(|(k, v)| (k, v));
            let cross_kv = cross_kvs.get(i).map(|(k, v)| (k, v));
            let (out, ns, nc) =
                block.forward(&x, xa, self_kv, cross_kv, Some(&self.causal_mask), device)?;
            x = out;
            new_self_kvs.push(ns);
            new_cross_kvs.push(nc);
        }

        let x = self.ln.forward(&x, device)?;
        // Weight-tied projection: logits = x @ token_embedding.T
        let emb_t = self.token_embedding.transpose(&[1, 0], device)?;
        let logits = matmul(&x, &emb_t, device)?;
        Ok((logits, new_self_kvs, new_cross_kvs))
    }
}

/// Build causal mask `[n_ctx, n_ctx]` (upper-triangular -1e9).
fn build_causal_mask(n_ctx: usize) -> Result<Array, WhisperError> {
    let mut data = vec![0.0_f32; n_ctx * n_ctx];
    #[allow(
        clippy::indexing_slicing,
        reason = "i, j in [0, n_ctx); i*n_ctx+j in [0, n_ctx*n_ctx)"
    )]
    for i in 0..n_ctx {
        for j in (i + 1)..n_ctx {
            data[i * n_ctx + j] = -1e9_f32;
        }
    }
    Array::from_f32_slice(&data, &[n_ctx as i32, n_ctx as i32])
        .map_err(|e| WhisperError::Mlx(e.to_string()))
}

// ── WhisperModel ──────────────────────────────────────────────────────────────

/// Loaded Whisper model (encoder + decoder).
#[allow(
    missing_debug_implementations,
    clippy::exhaustive_structs,
    reason = "WhisperModel fields are the encoder, decoder, and config — the complete struct for this model type"
)]
pub struct WhisperModel {
    /// Audio encoder.
    pub encoder: AudioEncoder,
    /// Text decoder.
    pub decoder: TextDecoder,
    /// Model configuration.
    pub cfg: WhisperConfig,
}

impl WhisperModel {
    /// Load a Whisper model from a snapshot directory.
    ///
    /// Expects `config.json` and `weights.npz` in the directory.
    ///
    /// `path` must be an operator-controlled, validated absolute path. It is
    /// set once at server startup via `--whisper-model-path` / `RMLX_WHISPER_MODEL_PATH`
    /// and never derived from request input.
    #[instrument(skip(path), fields(path = %path.as_ref().display()), level = "info")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, WhisperError> {
        let path = path.as_ref();
        let cfg_path = path.join("config.json");
        let cfg_str = std::fs::read_to_string(&cfg_path)
            .map_err(|e| WhisperError::WeightFile(format!("config.json: {e}")))?;
        tracing::debug!(stage = "config", path = %cfg_path.display(), "Whisper config loaded");
        let cfg = WhisperConfig::from_json(&cfg_str)?;

        info!(
            n_mels = cfg.n_mels,
            n_vocab = cfg.n_vocab,
            "loading Whisper"
        );
        let weights_path = path.join("weights.npz");
        let weights_size = std::fs::metadata(&weights_path).map_or(0, |m| m.len());
        tracing::debug!(stage = "weights", path = %weights_path.display(), bytes = weights_size, "loading Whisper weights.npz");
        let weights = load_npz(&weights_path)?;

        let encoder = AudioEncoder::load(&weights, &cfg)?;
        let decoder = TextDecoder::load(&weights, &cfg)?;
        Ok(Self {
            encoder,
            decoder,
            cfg,
        })
    }

    /// Detect the spoken language from encoder output.
    ///
    /// Runs a single SOT decoder step and returns the argmax over the 100
    /// language tokens (`<|en|>` … `<|yue|>`, ids `TOK_EN ..= TOK_LANG_LAST`).
    /// Falls back to English (`TOK_EN`) on error.
    ///
    /// Call after `encode_mel`. The returned token id can be passed directly to
    /// `WhisperTokenizer::sot_sequence_from_lang_tok`.
    pub fn detect_language(
        &self,
        encoder_out: &Array,
        device: Device,
    ) -> Result<u32, WhisperError> {
        use crate::tokenizer::TOK_SOT;
        // Single-token SOT input → decoder forward.
        let sot_arr = Array::from_i32_slice(&[TOK_SOT as i32], &[1, 1])
            .map_err(|e| WhisperError::Mlx(e.to_string()))?;
        let (logits, _, _) = self
            .decoder
            .forward(&sot_arr, encoder_out, 0, &[], &[], device)?;
        // logits: [1, 1, vocab] → slice language range → argmax
        let lang_start: i32 = TOK_EN as i32;
        let lang_end: i32 = TOK_LANG_LAST as i32 + 1; // exclusive — 100 language tokens
        let lang_logits =
            logits.slice(&[0, 0, lang_start], &[1, 1, lang_end], &[1, 1, 1], device)?;
        let best = argmax(
            &lang_logits.reshape(&[1, lang_end - lang_start], device)?,
            1,
            device,
        )?;
        // Materialise before reading bytes — same pattern as sample_next.
        best.eval().map_err(|e| WhisperError::Mlx(e.to_string()))?;
        let bytes = best
            .to_bytes()
            .map_err(|e| WhisperError::Mlx(e.to_string()))?;
        let Some(b4) = bytes.get(..4) else {
            return Ok(TOK_EN); // fallback to English
        };
        let idx = i32::from_le_bytes(b4.try_into().unwrap_or([0u8; 4]));
        Ok((lang_start + idx) as u32)
    }

    /// Encode mel frames → encoder output `[1, n_audio_ctx, n_state]`.
    ///
    /// `mel_frames`: `[T_frames][n_mels]` from `MelExtractor::extract()`.
    pub fn encode_mel(
        &self,
        mel_frames: &[Vec<f32>],
        device: Device,
    ) -> Result<Array, WhisperError> {
        let n_frames = mel_frames.len();
        let n_mels = self.cfg.n_mels;
        let mut data = vec![0.0_f32; n_frames * n_mels];
        #[allow(
            clippy::indexing_slicing,
            reason = "t in [0, n_frames), m in [0, n_mels)"
        )]
        for (t, frame) in mel_frames.iter().enumerate() {
            let m_len = n_mels.min(frame.len());
            for m in 0..m_len {
                data[t * n_mels + m] = frame[m];
            }
        }
        let mel = Array::from_f32_slice(&data, &[1, n_frames as i32, n_mels as i32])
            .map_err(|e| WhisperError::Mlx(e.to_string()))?;
        self.encoder.forward(&mel, device)
    }

    /// Greedy decode loop given encoder output and SOT sequence.
    ///
    /// `sot_sequence`: initial tokens (SOT + lang + task + no_timestamps).
    /// `max_tokens`: cap on generated tokens.
    /// `temperature`: 0 = greedy argmax; > 0 = temperature-scaled softmax + argmax.
    /// `filters`: per-step logit suppression set (`DecodeFilters`).
    ///
    /// Returns the full token sequence **including** any timestamp tokens — the
    /// caller (long-form chunker) needs them for segmentation. Special / non-speech
    /// content tokens are masked out by `filters`, never emitted.
    #[allow(
        clippy::explicit_counter_loop,
        reason = "offset tracks KV-cache position starting from sot_len; not a pure iteration counter"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "greedy_decode is a sequential decode loop with KV-cache state; splitting would obscure the prefill→decode flow"
    )]
    pub fn greedy_decode(
        &self,
        encoder_out: &Array,
        sot_sequence: &[u32],
        max_tokens: usize,
        temperature: f32,
        filters: &DecodeFilters,
        device: Device,
    ) -> Result<Vec<u32>, WhisperError> {
        let mut self_kvs: Vec<(Array, Array)> = Vec::new();
        let mut cross_kvs: Vec<(Array, Array)> = Vec::new();
        // Sampled tokens (the suffix after the SOT prefix) — these are what the
        // timestamp rules and the caller operate on.
        let mut sampled: Vec<u32> = Vec::new();

        // Prefill the SOT sequence.
        let sot_i32: Vec<i32> = sot_sequence.iter().map(|&t| t as i32).collect();
        let sot_arr = Array::from_i32_slice(&sot_i32, &[1, sot_i32.len() as i32])
            .map_err(|e| WhisperError::Mlx(e.to_string()))?;

        let (logits, new_self, new_cross) =
            self.decoder
                .forward(&sot_arr, encoder_out, 0, &self_kvs, &cross_kvs, device)?;
        self_kvs = new_self;
        cross_kvs = new_cross;

        // Pick token from last position of prefill output.
        let sot_len = sot_sequence.len();
        let last_logits = logits.slice(
            &[0, sot_len as i32 - 1, 0],
            &[1, sot_len as i32, self.cfg.n_vocab as i32],
            &[1, 1, 1],
            device,
        )?;
        let mut offset = sot_len;

        // First sampled position: apply SuppressBlank (in addition to the standing
        // suppression set + timestamp rules).
        let mut logit_vec = logits_to_f32(&last_logits, self.cfg.n_vocab, device)?;
        filters.apply(&mut logit_vec, &sampled, true);
        let mut next_tok = argmax_f32(&logit_vec, temperature);
        debug!(
            next_tok,
            tok_eot = TOK_EOT,
            tok_nospeech = TOK_NOSPEECH,
            tok_ts_begin = TOK_TIMESTAMP_BEGIN,
            "prefill first token (after filters)"
        );
        if next_tok == TOK_NOSPEECH {
            return Err(WhisperError::Silence);
        }
        sampled.push(next_tok);

        while sampled.len() < max_tokens {
            if next_tok == TOK_EOT {
                break;
            }
            // Belt-and-suspenders: never request a positional-embedding row
            // `>= n_text_ctx`. The next decode step would slice the positional
            // table at `[offset, offset+1)`; once `offset == n_text_ctx` that row
            // is off the `[n_text_ctx, n_state]` table and `forward` would abort
            // the whole transcription. The caller already bounds `max_tokens`, but
            // this guard makes the decode loop self-contained regardless of caller.
            if offset >= self.cfg.n_text_ctx {
                debug!(
                    offset,
                    n_text_ctx = self.cfg.n_text_ctx,
                    "stopping decode: positional-embedding ceiling reached"
                );
                break;
            }

            let tok_i32 = [next_tok as i32];
            let tok_arr = Array::from_i32_slice(&tok_i32, &[1, 1])
                .map_err(|e| WhisperError::Mlx(e.to_string()))?;

            let (step_logits, new_self, new_cross) = self.decoder.forward(
                &tok_arr,
                encoder_out,
                offset,
                &self_kvs,
                &cross_kvs,
                device,
            )?;
            self_kvs = new_self;
            cross_kvs = new_cross;
            offset += 1;

            let mut lv = logits_to_f32(&step_logits, self.cfg.n_vocab, device)?;
            filters.apply(&mut lv, &sampled, false);
            next_tok = argmax_f32(&lv, temperature);
            sampled.push(next_tok);
        }

        // Drop trailing EOT — it is a stop marker, not content.
        while sampled.last() == Some(&TOK_EOT) {
            sampled.pop();
        }
        debug!(n_tokens = sampled.len(), "greedy decode done");
        Ok(sampled)
    }
}

/// Per-step logit suppression for Whisper decode.
///
/// Holds the standing suppression set (non-speech / special tokens derived from
/// the tokenizer) plus a `timestamps` mode flag. Applied to a host-side f32 logit
/// vector at **every** decode step — this is the openai-whisper / `mlx_whisper`
/// `LogitFilter` chain (`SuppressBlank` + `SuppressTokens` + timestamp handling),
/// the absence of which made rMLX decode halt or emit garbage.
#[derive(Debug, Clone)]
pub struct DecodeFilters {
    /// Standing suppression set (always masked).
    suppress: Vec<u32>,
    /// `true` => timestamp mode (emit timestamp tokens, apply pairing rules).
    /// `false` => `no_timestamps` mode (suppress all timestamp tokens).
    timestamps: bool,
    /// First sampled token of the SOT prefix that begins generation (always 0
    /// here because `sampled` is the post-prefix suffix).
    blank_suppress: Vec<u32>,
}

impl DecodeFilters {
    /// Build the filter chain.
    ///
    /// `suppress` is the tokenizer-derived non-speech/special set
    /// (`WhisperTokenizer::suppress_tokens`). `blank_ids` are the SuppressBlank
    /// ids (EOT + the blank-space token, `tokenizer.encode(" ")`). `timestamps`
    /// selects timestamp vs `no_timestamps` mode.
    #[must_use]
    pub fn new(suppress: Vec<u32>, blank_ids: Vec<u32>, timestamps: bool) -> Self {
        Self {
            suppress,
            timestamps,
            blank_suppress: blank_ids,
        }
    }

    /// Apply the filter chain in-place to a host f32 logit vector of length
    /// `n_vocab`.
    ///
    /// - `sampled`: tokens sampled so far (post-SOT-prefix).
    /// - `first_step`: `true` only for the very first sampled position
    ///   (enables SuppressBlank).
    #[allow(
        clippy::indexing_slicing,
        reason = "all token ids are bounds-checked against logits.len() before indexing"
    )]
    pub fn apply(&self, logits: &mut [f32], sampled: &[u32], first_step: bool) {
        let n = logits.len();
        let mask = |logits: &mut [f32], tok: u32| {
            let t = tok as usize;
            if t < n {
                logits[t] = f32::NEG_INFINITY;
            }
        };

        // SuppressBlank — first sampled position only.
        if first_step {
            for &t in &self.blank_suppress {
                mask(logits, t);
            }
        }

        // SuppressTokens — non-speech / special set, every step.
        for &t in &self.suppress {
            mask(logits, t);
        }

        let ts_begin = TOK_TIMESTAMP_BEGIN as usize;
        if !self.timestamps {
            // no_timestamps mode: suppress every timestamp token.
            for l in logits.iter_mut().skip(ts_begin) {
                *l = f32::NEG_INFINITY;
            }
            return;
        }

        // Timestamp mode: faithful port of openai-whisper `ApplyTimestampRules`.
        // <|notimestamps|> can never be sampled here.
        mask(logits, TOK_NO_TIMESTAMPS);

        let len = sampled.len();
        let last_was_ts = sampled.last().is_some_and(|&t| t as usize >= ts_begin);
        // NOTE: `penultimate_was_timestamp` is TRUE when fewer than 2 tokens have
        // been sampled (`len(seq) < 2`) — this is the reference semantics. Getting
        // this wrong (treating len<2 as false) strands the decoder into EOT right
        // after the opening `<|0.00|>` and yields empty transcripts.
        let penult_was_ts = len < 2
            || sampled
                .get(len.wrapping_sub(2))
                .is_some_and(|&t| t as usize >= ts_begin);

        if last_was_ts {
            if penult_was_ts {
                // Timestamps must pair → the next token has to be NON-timestamp.
                for l in logits.iter_mut().skip(ts_begin) {
                    *l = f32::NEG_INFINITY;
                }
            } else {
                // A single open timestamp → the next token cannot be normal text
                // (only a closing timestamp or EOT). Mask everything below EOT.
                let eot = TOK_EOT as usize;
                for l in logits.iter_mut().take(eot.min(n)) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }

        // Monotonic timestamps: forbid timestamps smaller than the last one, and
        // force nonzero segment length.
        if let Some(&last_ts) = sampled.iter().rev().find(|&&t| t as usize >= ts_begin) {
            let last_ts = last_ts as usize;
            let upper = if last_was_ts && !penult_was_ts {
                last_ts // allow re-emitting the same close timestamp
            } else {
                last_ts + 1
            };
            let upper = upper.min(n);
            for (t, l) in logits.iter_mut().enumerate().take(upper) {
                if t >= ts_begin {
                    *l = f32::NEG_INFINITY;
                }
            }
        }

        // Beginning-of-sequence: the first sampled token must be a timestamp.
        if sampled.is_empty() {
            for l in logits.iter_mut().take(ts_begin.min(n)) {
                *l = f32::NEG_INFINITY;
            }
        }

        // Tie-break: if the cumulative probability mass over all timestamp tokens
        // exceeds the best single text-token probability, suppress text entirely
        // (force a timestamp). Mirrors the `log_softmax` comparison in the
        // reference; computed on the post-mask logits.
        let ts_logsumexp = logsumexp(&logits[ts_begin.min(n)..]);
        let max_text = logits
            .iter()
            .take(ts_begin.min(n))
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if ts_logsumexp > max_text {
            for l in logits.iter_mut().take(ts_begin.min(n)) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// Numerically-stable log-sum-exp over a slice (ignores `-inf` entries).
fn logsumexp(xs: &[f32]) -> f32 {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = xs.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Materialise a `[1, 1, n_vocab]` (or `[1, n_vocab]`) logit tensor (F16/F32/Bf16)
/// into a host-side f32 vector of length `n_vocab`.
fn logits_to_f32(logits: &Array, n_vocab: usize, device: Device) -> Result<Vec<f32>, WhisperError> {
    use rmlx_mlx::Dtype;
    let flat = logits.reshape(&[-1], device)?.astype(Dtype::F32, device)?;
    flat.eval().map_err(WhisperError::from)?;
    let bytes = flat.to_bytes().map_err(WhisperError::from)?;
    let mut out = Vec::with_capacity(n_vocab);
    #[allow(
        clippy::indexing_slicing,
        reason = "chunks_exact(4) yields exactly 4 bytes; take(n_vocab) bounds the count"
    )]
    for chunk in bytes.chunks_exact(4).take(n_vocab) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if out.len() < n_vocab {
        out.resize(n_vocab, f32::NEG_INFINITY);
    }
    Ok(out)
}

/// Greedy argmax over a host f32 logit vector.
///
/// `temperature` is accepted for parity with the API; at temp == 0 (the only
/// deterministic mode rMLX serves) this is a plain argmax. Temperature scaling
/// changes the softmax distribution but not the argmax, so for greedy decode the
/// result is identical — we keep the param to avoid an API break and document
/// that sampling (temp > 0) is not yet a stochastic path.
fn argmax_f32(logits: &[f32], _temperature: f32) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "whisper_tests.rs"]
mod tests;
