// LOC-exempt: encoder + decoder + the decoder pass form one tightly coupled MLX graph;
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
//! ## Weight layout (`weights.safetensors` or `weights.npz`)
//!
//! Tensor names follow the mlx-community naming convention:
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
//! The weight file stores Conv1d weights as `[out, kernel, in]`, the layout MLX
//! `conv1d` reads. The load does not transpose them.
//!
//! ## Positional embedding
//!
//! The encoder positional embedding is computed as fixed sinusoids (not stored)
//! and added inside the encoder forward pass. The decoder positional embedding
//! IS stored as `decoder.positional_embedding` of shape `[n_text_ctx, n_state]`.
//!
//! ## Inference loop
//!
//! [`decode_loop`] holds the host side of a decode: the logit filters, the
//! token pick, which the caller supplies, and the average log-probability.
//! [`TextDecoder::open_window`] is the model side: it projects the
//! cross-attention keys and values of an encoder output one time, and each
//! pass over that window is one forward for each token with a self-attention
//! KV-cache per block.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use rmlx_mlx::{
    add, concatenate, conv1d, divide, gelu, matmul, multiply, scalar_f32, softmax_precise, sqrt,
    subtract, sum_axis_keepdims, Array, Device,
};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

use crate::mel::{check_n_mels, supported_n_mels, N_FRAMES};
use crate::npz::WeightMap;
use crate::tokenizer::{N_VOCAB, TOK_EOT, TOK_NO_TIMESTAMPS, TOK_TIMESTAMP_BEGIN};
use alignment_heads::read_alignment_heads;
use decode_loop::{decode_loop, window_start, writes_at_most, Decoded, WindowStart};
use weight_file::load_weights;

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
    /// `config.json` states a size the decode does not support.
    #[error(
        "config.json: {field} is {found}, the decode supports {expected} only \
         (Whisper large-v3, large-v3-turbo and distil-large-v3)"
    )]
    UnsupportedConfig {
        /// The `config.json` field.
        field: &'static str,
        /// The value it states.
        found: usize,
        /// The value or values the decode supports, as text.
        expected: String,
    },
    /// `config.json` has a `quantization` key: the weights are packed, and the
    /// decode reads them as plain matrices.
    #[error("config.json: the key `quantization` is set, and the decode does not support a quantized snapshot")]
    QuantizedSnapshot,
    /// The requested language is not one Whisper knows.
    #[error(
        "unknown language '{0}'; use a Whisper language code, an English language name or 'auto'"
    )]
    UnknownLanguage(String),
    /// The caller's cancel flag was set; window number `window` (from 0), at
    /// sample `seek` of the 16 kHz audio, was not decoded.
    #[error("transcription cancelled before window {window} (sample {seek})")]
    Cancelled {
        /// The first window not decoded.
        window: usize,
        /// Its start, in 16 kHz samples.
        seek: usize,
    },
}

impl From<rmlx_core::error::Error> for WhisperError {
    fn from(e: rmlx_core::error::Error) -> Self {
        Self::Mlx(e.to_string())
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
    /// Audio context length in encoder positions: 1500, one 30 s window.
    pub n_audio_ctx: usize,
    /// Audio hidden state dimension.
    pub n_audio_state: usize,
    /// Number of audio attention heads.
    pub n_audio_head: usize,
    /// Number of audio encoder layers.
    pub n_audio_layer: usize,
    /// Vocabulary size: [`N_VOCAB`], the vocabulary the decode's token ids are for.
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

/// The `quantization` key of `config.json`, whatever it holds.
#[derive(serde::Deserialize)]
struct QuantizationKey {
    quantization: Option<serde::de::IgnoredAny>,
}

impl WhisperConfig {
    /// Load from a JSON string. Refuse a `quantization` key, an `n_mels` other
    /// than 80 or 128, an `n_vocab` other than [`N_VOCAB`] and an `n_audio_ctx`
    /// other than the encoder positions of one 30 s window.
    pub fn from_json(s: &str) -> Result<Self, WhisperError> {
        let cfg: Self = serde_json::from_str(s).map_err(|e| WhisperError::Config(e.to_string()))?;
        let quantization: QuantizationKey =
            serde_json::from_str(s).map_err(|e| WhisperError::Config(e.to_string()))?;
        if quantization.quantization.is_some() {
            return Err(WhisperError::QuantizedSnapshot);
        }
        if check_n_mels(cfg.n_mels).is_err() {
            return Err(WhisperError::UnsupportedConfig {
                field: "n_mels",
                found: cfg.n_mels,
                expected: supported_n_mels(),
            });
        }
        // The encoder's second conv has stride 2: two mel frames per position.
        let supported = [
            ("n_vocab", cfg.n_vocab, N_VOCAB),
            ("n_audio_ctx", cfg.n_audio_ctx, N_FRAMES / 2),
        ];
        for (field, found, expected) in supported {
            if found != expected {
                return Err(WhisperError::UnsupportedConfig {
                    field,
                    found,
                    expected: expected.to_string(),
                });
            }
        }
        Ok(cfg)
    }
}

// ── Weight map ────────────────────────────────────────────────────────────────

/// Load all weights from a `.npz` file into a weight map.
///
/// Older Whisper snapshots from mlx-community ship as `weights.npz` (NumPy
/// compressed array archive = ZIP of `.npy` entries). ZIP64 archives
/// (≥ 4 GiB, as produced by the large-v3 snapshot) are fully supported via
/// the central-directory parser in `crate::npz`. An error names the file, not
/// its directory: the server sends it to the client.
pub fn load_npz(path: impl AsRef<Path>) -> Result<WeightMap, WhisperError> {
    let path = path.as_ref();
    crate::npz::load_npz(path).map_err(|e| {
        let file = path.file_name().unwrap_or_default().to_string_lossy();
        WhisperError::WeightFile(format!("{file}: {e}"))
    })
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
    /// How many times this attention projected keys and values from a cross
    /// input (an encoder output), and did not read them from a cache.
    cross_projections: AtomicUsize,
}

impl MultiHeadAttention {
    fn load(map: &WeightMap, prefix: &str, n_head: usize) -> Result<Self, WhisperError> {
        Ok(Self {
            query: Linear::load(map, &format!("{prefix}.query"), true)?,
            key: Linear::load(map, &format!("{prefix}.key"), false)?,
            value: Linear::load(map, &format!("{prefix}.value"), true)?,
            out: Linear::load(map, &format!("{prefix}.out"), true)?,
            n_head,
            cross_projections: AtomicUsize::new(0),
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
                self.cross_projections.fetch_add(1, Ordering::Relaxed);
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
    n_vocab: usize,
    n_text_ctx: usize,
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
            n_vocab: cfg.n_vocab,
            n_text_ctx: cfg.n_text_ctx,
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

    /// How many times a decoder block projected the cross-attention keys and
    /// values of an encoder output since the load: one for each block of each
    /// encoder output. A window has one encoder output, or two when its
    /// language is read from a detection input.
    pub(crate) fn cross_projections(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.cross_attn.cross_projections.load(Ordering::Relaxed))
            .sum()
    }

    /// Open an encoded window: read its start with one decoder forward of the
    /// start token alone, and keep the cross-attention keys and values that
    /// forward projected. They depend on the encoder output only, so every
    /// decode of the window reads them.
    pub(crate) fn open_window<'a>(
        &'a self,
        encoder_out: &'a Array,
        device: Device,
    ) -> Result<(WindowStart, WindowDecoder<'a>), WhisperError> {
        let mut pass = DecoderPass {
            decoder: self,
            encoder_out,
            device,
            self_kvs: Vec::new(),
            cross_kvs: Vec::new(),
            fed: 0,
        };
        let start = window_start(|input: &[u32]| pass.forward(input))?;
        let window = WindowDecoder {
            decoder: self,
            encoder_out,
            device,
            cross_kvs: pass.cross_kvs,
        };
        Ok((start, window))
    }
}

/// One decoder pass over an encoded window: the attention state of the tokens
/// fed so far.
struct DecoderPass<'a> {
    decoder: &'a TextDecoder,
    encoder_out: &'a Array,
    device: Device,
    self_kvs: Vec<(Array, Array)>,
    cross_kvs: Vec<(Array, Array)>,
    fed: usize,
}

impl DecoderPass<'_> {
    /// Feed `input` after the tokens fed before it and return the logit row
    /// of its last position.
    fn forward(&mut self, input: &[u32]) -> Result<Vec<f32>, WhisperError> {
        let one;
        let many;
        let ids: &[i32] = if let [token] = input {
            one = [*token as i32];
            &one
        } else {
            many = input.iter().map(|&t| t as i32).collect::<Vec<i32>>();
            &many
        };
        let Some(last) = ids.len().checked_sub(1) else {
            return Err(WhisperError::Mlx(
                "decoder forward with no input token".to_owned(),
            ));
        };
        let ids = Array::from_i32_slice(ids, &[1, ids.len() as i32])
            .map_err(|e| WhisperError::Mlx(e.to_string()))?;
        let (logits, new_self, new_cross) = self.decoder.forward(
            &ids,
            self.encoder_out,
            self.fed,
            &self.self_kvs,
            &self.cross_kvs,
            self.device,
        )?;
        self.self_kvs = new_self;
        self.cross_kvs = new_cross;
        self.fed += input.len();
        let n_vocab = self.decoder.n_vocab;
        let row = logits.slice(
            &[0, last as i32, 0],
            &[1, last as i32 + 1, n_vocab as i32],
            &[1, 1, 1],
            self.device,
        )?;
        logits_to_f32(&row, n_vocab, self.device)
    }
}

/// The decoder side of one encoded window, from [`TextDecoder::open_window`].
#[allow(missing_debug_implementations)]
pub(crate) struct WindowDecoder<'a> {
    decoder: &'a TextDecoder,
    encoder_out: &'a Array,
    device: Device,
    /// The cross-attention keys and values of each block for this window.
    cross_kvs: Vec<(Array, Array)>,
}

impl WindowDecoder<'_> {
    /// A decoder pass over the window that was fed nothing and reads the
    /// window's cross-attention keys and values.
    fn pass(&self) -> Result<DecoderPass<'_>, WhisperError> {
        let cross_kvs = self
            .cross_kvs
            .iter()
            .map(|(k, v)| Ok((k.try_clone()?, v.try_clone()?)))
            .collect::<Result<Vec<_>, WhisperError>>()?;
        Ok(DecoderPass {
            decoder: self.decoder,
            encoder_out: self.encoder_out,
            device: self.device,
            self_kvs: Vec::new(),
            cross_kvs,
            fed: 0,
        })
    }

    /// Whether a decode of the window from `prefix` writes at most
    /// `text_bound` text tokens. It ends when the text passes the bound.
    pub(crate) fn writes_at_most(
        &self,
        text_bound: usize,
        prefix: &[u32],
        max_tokens: usize,
        filters: &DecodeFilters,
        pick: impl FnMut(&[f32]) -> Result<u32, WhisperError>,
    ) -> Result<bool, WhisperError> {
        let mut pass = self.pass()?;
        let within = writes_at_most(
            text_bound,
            prefix,
            max_tokens,
            self.decoder.n_text_ctx,
            filters,
            |input: &[u32]| pass.forward(input),
            pick,
        )?;
        debug!(
            text_bound,
            within,
            decoder_steps = pass.fed.saturating_sub(prefix.len()) + 1,
            "bounded window decode done"
        );
        Ok(within)
    }

    /// Decode the window from `prefix` (an optional previous-text prompt,
    /// then the SOT sequence) for at most `max_tokens` tokens. `pick` chooses
    /// each token from its filtered logit row.
    ///
    /// The tokens hold the timestamp tokens: the long-form chunker needs them
    /// for segmentation. `filters` masks the special and non-speech tokens.
    pub(crate) fn decode(
        &self,
        prefix: &[u32],
        max_tokens: usize,
        filters: &DecodeFilters,
        pick: impl FnMut(&[f32]) -> Result<u32, WhisperError>,
    ) -> Result<Decoded, WhisperError> {
        let mut pass = self.pass()?;
        let decoded = decode_loop(
            prefix,
            max_tokens,
            self.decoder.n_text_ctx,
            filters,
            |input: &[u32]| pass.forward(input),
            pick,
        )?;
        debug!(
            n_tokens = decoded.tokens.len(),
            stop = ?decoded.stop,
            "window decode done"
        );
        Ok(decoded)
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
    reason = "AudioEncoder and TextDecoder do not implement Debug"
)]
pub struct WhisperModel {
    /// Audio encoder.
    pub encoder: AudioEncoder,
    /// Text decoder.
    pub decoder: TextDecoder,
    /// Model configuration.
    pub cfg: WhisperConfig,
    alignment_heads: Option<Vec<(usize, usize)>>,
}

impl WhisperModel {
    /// The decoder cross-attention heads that line text tokens up with audio
    /// frames, as (layer, head) pairs. `None` when the snapshot has no
    /// `alignment_heads` entry, or when the entry is not usable pairs (the load
    /// warns of that). Nothing reads them yet.
    #[must_use]
    pub fn alignment_heads(&self) -> Option<&[(usize, usize)]> {
        self.alignment_heads.as_deref()
    }

    /// Load a Whisper model from a snapshot directory.
    ///
    /// Expects `config.json` and `weights.safetensors` or `weights.npz` in the
    /// directory; `weights.safetensors` is read when both exist. A config that
    /// [`WhisperConfig::from_json`] refuses is refused before the weights are read.
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
        let weights = load_weights(path)?;

        let encoder = AudioEncoder::load(&weights, &cfg)?;
        let decoder = TextDecoder::load(&weights, &cfg)?;
        let alignment_heads = read_alignment_heads(&weights, &cfg)?;
        Ok(Self {
            encoder,
            decoder,
            cfg,
            alignment_heads,
        })
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

pub(crate) mod decode_loop;

mod alignment_heads;

mod weight_file;

#[cfg(test)]
mod snapshot_fixture;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "whisper_tests.rs"]
mod tests;
