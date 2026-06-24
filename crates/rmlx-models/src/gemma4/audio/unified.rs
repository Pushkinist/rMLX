//! Gemma4 **unified** (`gemma4_unified` / `Gemma4UnifiedForConditionalGeneration`)
//! encoder-free audio embedder.
//!
//! Faithful host+MLX port of the HF Transformers `gemma4_unified`
//! `Gemma4UnifiedAudioFeatureExtractor` (waveform → fixed-length frames) +
//! `Gemma4UnifiedMultimodalEmbedder` (`RMSNormNoScale -> embedding_projection`).
//!
//! ## Why this is a different path than the Conformer audio tower
//!
//! The unified 12B has **no audio transformer**. Audio is early-fusion: the raw
//! 16 kHz mono waveform is chunked into fixed-length frames of
//! `audio_samples_per_token` (640) samples — 40 ms each at 16 kHz — and each
//! frame is projected straight into the shared 48-layer LM hidden space via the
//! same `embed_audio.embedding_projection` (`RMSNormNoScale -> Linear`) the
//! standard family reuses for its Conformer output. The 12B snapshot ships only
//! `embed_audio.embedding_projection.{weight,scales}` — there is **no**
//! `audio_tower.*`. The standard `gemma4` family (e4b/26b) keeps the existing
//! [`super::AudioEncoder`] Conformer path.
//!
//! ## Reference pipeline (`Gemma4UnifiedAudioFeatureExtractor._extract_waveform_features`)
//!
//! ```text
//! pad_len  = (-len(waveform)) % audio_samples_per_token   # zero-pad tail
//! num_tok  = len(waveform) // audio_samples_per_token       # ceil(n / 640)
//! features = waveform.reshape(num_tok, audio_samples_per_token)  # [T, 640] f32
//! ```
//!
//! There is **no** mel spectrogram, no windowing, no per-sample normalization or
//! scaling — the raw float waveform is fed directly. Since there is no
//! downsampling, `num_soft_tokens == num_tok == ceil(num_samples / 640)`.
//! `audio_embed_dim == audio_samples_per_token == 640` (each soft token's
//! feature vector is exactly one frame of raw samples). The frames are then
//! projected via [`super::vision::MultimodalEmbedder`] (`embed_audio`) into the
//! 3840-wide text hidden space and scattered at the `<|audio|>` positions,
//! exactly mirroring [`super::vision::unified::build_unified_inputs_embeds`].

use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{multiply, scalar_f32, Array, Device};
use tracing::{debug, info};

use super::super::vision::MultimodalEmbedder;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `audio_config` for `gemma4_unified` (`model_type: gemma4_unified_audio`).
///
/// Distinct from [`super::super::config::Gemma4AudioConfig`] (the Conformer
/// tower): the unified audio config has **no** `num_hidden_layers` / heads /
/// conv params — it is encoder-free. Fields are the complete contract for the
/// embedder.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — fields are the complete gemma4_unified audio-embedder contract; adding a field requires updating from_json"
)]
#[derive(Debug, Clone)]
pub struct UnifiedAudioConfig {
    /// Raw-sample feature dim of one audio frame (640). Equals
    /// `audio_samples_per_token` on the snapshot; the `embed_audio` projection
    /// consumes vectors of this width.
    pub audio_embed_dim: usize,
    /// Raw audio samples grouped into one soft token (640 — 40 ms at 16 kHz).
    pub audio_samples_per_token: usize,
    /// `embed_audio.embedding_projection` input dim (640 = output_proj_dims).
    /// Validated against the loaded projection weight at load time.
    pub output_proj_dims: usize,
    /// RMSNorm epsilon (1e-6).
    pub rms_norm_eps: f32,
    /// `<audio_soft_token>` id (top-level `audio_token_id`, 258881 on the 12B).
    pub audio_token_id: u32,
}

impl UnifiedAudioConfig {
    /// Parse from the `audio_config` JSON object. `audio_token_id` is the
    /// top-level config value. Missing keys fall back to the verified
    /// `gemma-4-12B` `gemma4_unified_audio` values.
    pub fn from_json(v: &serde_json::Value, audio_token_id: u32) -> Self {
        let u = |key: &str, dflt: usize| -> usize {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let f = |key: &str, dflt: f32| -> f32 {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(dflt, |x| x as f32)
        };
        Self {
            audio_embed_dim: u("audio_embed_dim", 640),
            audio_samples_per_token: u("audio_samples_per_token", 640),
            output_proj_dims: u("output_proj_dims", 640),
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            audio_token_id,
        }
    }

    /// Read `audio_config` from a model dir when `architectures[0]` is the
    /// unified arch. Returns `None` if there is no `audio_config` key.
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path).map_err(|e| {
            Error::Config(format!(
                "gemma4_unified: cannot read {}: {e}",
                path.display()
            ))
        })?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "gemma4_unified: malformed config.json at {}: {e}",
                path.display()
            ))
        })?;
        let audio_token_id = v
            .get("audio_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(258_881) as u32;
        Ok(v.get("audio_config")
            .map(|ac| Self::from_json(ac, audio_token_id)))
    }
}

// ---------------------------------------------------------------------------
// Encoder-free audio feature extraction (host)
// ---------------------------------------------------------------------------

/// Chunk a raw 16 kHz mono waveform into fixed-length frames of
/// `audio_samples_per_token` samples. The tail is zero-padded to a multiple of
/// the frame size (within the last frame — no extra frame is created).
///
/// Faithful to `Gemma4UnifiedAudioFeatureExtractor._extract_waveform_features`:
/// no normalization, no windowing, raw float samples. Returns the flat
/// `[num_tokens * audio_samples_per_token]` f32 buffer plus `num_tokens`.
#[must_use]
#[allow(
    clippy::indexing_slicing,
    reason = "frames is sized num_tokens*spt >= samples.len() by construction (ceil-div), so the [..samples.len()] prefix is always in bounds"
)]
pub fn extract_waveform_frames(
    samples: &[f32],
    audio_samples_per_token: usize,
) -> (Vec<f32>, usize) {
    let spt = audio_samples_per_token.max(1);
    // ceil(len / spt); pad the tail with zeros to a full frame.
    let num_tokens = samples.len().div_ceil(spt);
    let mut frames = vec![0.0_f32; num_tokens * spt];
    frames[..samples.len()].copy_from_slice(samples);
    (frames, num_tokens)
}

/// Number of audio soft tokens a clip of `num_samples` raw samples yields:
/// `ceil(num_samples / audio_samples_per_token)`. Must equal the model-frame
/// count produced by [`extract_waveform_frames`].
#[inline]
#[must_use]
pub fn unified_num_audio_soft_tokens(num_samples: usize, cfg: &UnifiedAudioConfig) -> usize {
    num_samples.div_ceil(cfg.audio_samples_per_token.max(1))
}

// ---------------------------------------------------------------------------
// Unified audio embedder
// ---------------------------------------------------------------------------

/// Encoder-free unified audio embedder (`embed_audio.*`).
///
/// Wraps the shared [`MultimodalEmbedder`] (`RMSNormNoScale -> embedding_projection`)
/// — the audio path has no tower, conv front-end, or per-frame state, so the
/// embedder *is* the projection. The host feature front-end
/// ([`extract_waveform_frames`]) runs before [`Self::forward`].
#[allow(missing_debug_implementations)]
pub struct UnifiedAudioEmbedder {
    cfg: UnifiedAudioConfig,
    /// Shared `RMSNormNoScale -> embedding_projection` (`embed_audio.*`).
    embed_audio: MultimodalEmbedder,
}

impl UnifiedAudioEmbedder {
    /// Parsed unified-audio sub-config.
    pub fn config(&self) -> &UnifiedAudioConfig {
        &self.cfg
    }

    /// Project chunked waveform frames into the text hidden space.
    ///
    /// `frames`: flat `[num_tokens * audio_embed_dim]` f32 from
    /// [`extract_waveform_frames`]. Returns `[1, num_tokens, text_hidden]` ready
    /// to scatter into `inputs_embeds`.
    pub fn forward(&self, frames: &[f32], num_tokens: usize, device: Device) -> Result<Array> {
        let dim = self.cfg.audio_embed_dim as i32;
        let frame_arr = Array::from_f32_slice(frames, &[1, num_tokens as i32, dim])?;
        let out = self.embed_audio.forward(&frame_arr, device)?;
        debug!(
            num_tokens,
            audio_embed_dim = self.cfg.audio_embed_dim,
            "gemma4_unified audio: embedder forward"
        );
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load the unified audio embedder (`embed_audio.*`) from a snapshot directory.
/// Errors if the unified-audio weights are absent (caller disables audio input
/// on error).
pub fn load_unified_audio_embedder(
    model_dir: &Path,
    cfg: &UnifiedAudioConfig,
) -> Result<UnifiedAudioEmbedder> {
    // embed_audio: RMSNormNoScale -> embedding_projection (reused loader).
    let embed_audio =
        super::super::vision::load_multimodal_embedder(model_dir, "embed_audio", cfg.rms_norm_eps)?;

    // Validate the parsed `output_proj_dims` / `audio_embed_dim` against the
    // loaded projection's actual input feature dim — making the config field
    // load-bearing (mirrors the unified-vision loader's check). A checkpoint
    // whose `embed_audio.embedding_projection` does not consume
    // `output_proj_dims` features is rejected here, not deep in forward.
    if let Some(proj_in) = embed_audio.projection_input_dim() {
        if proj_in != cfg.output_proj_dims {
            return Err(Error::Loader(format!(
                "gemma4_unified audio: output_proj_dims ({}) != embed_audio.embedding_projection \
                 input dim ({proj_in}) — config/checkpoint mismatch",
                cfg.output_proj_dims
            )));
        }
        if proj_in != cfg.audio_embed_dim {
            return Err(Error::Loader(format!(
                "gemma4_unified audio: audio_embed_dim ({}) != embed_audio.embedding_projection \
                 input dim ({proj_in}) — config/checkpoint mismatch",
                cfg.audio_embed_dim
            )));
        }
    }

    info!(
        audio_embed_dim = cfg.audio_embed_dim,
        audio_samples_per_token = cfg.audio_samples_per_token,
        audio_token_id = cfg.audio_token_id,
        "gemma4_unified audio: embedder loaded (encoder-free)"
    );

    Ok(UnifiedAudioEmbedder {
        cfg: cfg.clone(),
        embed_audio,
    })
}

// ---------------------------------------------------------------------------
// build unified inputs_embeds (text + scattered audio soft tokens)
// ---------------------------------------------------------------------------

/// Build the merged `inputs_embeds` for a unified-arch audio prompt.
///
/// Mirrors [`super::super::vision::unified::build_unified_inputs_embeds`] but
/// routes the encode through the encoder-free [`UnifiedAudioEmbedder`]. The clip
/// contributes `num_tokens` soft tokens scattered at its contiguous run of
/// `audio_token_id` positions.
///
/// `frames`: flat `[num_tokens * audio_embed_dim]` f32 (host feature front-end).
/// Returns `(inputs_embeds [1, seq, hidden], masked_ids [seq])`.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: audio_positions are filtered from input_ids; the run is validated contiguous before slice_update"
)]
pub fn build_unified_audio_inputs_embeds(
    model: &super::super::model::Gemma4Text,
    embedder: &UnifiedAudioEmbedder,
    frames: &[f32],
    num_tokens: usize,
    audio_token_id: u32,
    input_ids: &[u32],
    device: Device,
) -> Result<(Array, Array)> {
    let hidden = model.cfg.hidden_size as i32;
    let seq = input_ids.len();

    // Locate the audio-token positions (one contiguous run per clip).
    let audio_positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == audio_token_id)
        .map(|(i, _)| i)
        .collect();
    if audio_positions.len() != num_tokens {
        return Err(Error::Model(format!(
            "gemma4_unified audio: {} audio-token ({audio_token_id}) positions in prompt != \
             {num_tokens} audio soft tokens — scatter would misalign",
            audio_positions.len()
        )));
    }
    info!(
        audio_tokens = audio_positions.len(),
        num_tokens,
        seq,
        "gemma4_unified audio: building inputs_embeds (token count == soft tokens)"
    );

    // Scaled text embeddings: embed_tokens(ids) * sqrt(hidden).
    let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
    let ids_arr = Array::from_i32_slice(&ids_i32, &[seq as i32])?;
    let h_raw = model.embed_tokens.forward(&ids_arr, device)?;
    let embed_scale =
        scalar_f32((model.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
    let mut embeds = multiply(&h_raw, &embed_scale, device)?;
    embeds = embeds.reshape(&[1, seq as i32, hidden], device)?;
    let embeds_dtype = embeds.dtype();

    // Encode audio frames -> [1, num_tokens, hidden] f32.
    let feats = embedder.forward(frames, num_tokens, device)?;
    let fs = feats.shape();
    if fs.first().copied() != Some(1)
        || fs.get(1).copied() != Some(num_tokens as i32)
        || fs.get(2).copied() != Some(hidden)
    {
        return Err(Error::Model(format!(
            "gemma4_unified audio: audio feature shape {fs:?} != [1, {num_tokens}, {hidden}]"
        )));
    }
    let feats = feats.astype(embeds_dtype, device)?;

    // The audio run must be contiguous (one `<|audio|>` run per clip).
    let first = audio_positions.first().copied().unwrap_or(0);
    let contiguous = audio_positions
        .iter()
        .enumerate()
        .all(|(k, &p)| p == first + k);
    if !contiguous {
        return Err(Error::Model(format!(
            "gemma4_unified audio: audio-token positions are not contiguous (got {audio_positions:?})"
        )));
    }
    embeds = embeds.slice_update(
        &feats,
        &[0, first as i32, 0],
        &[1, (first + num_tokens) as i32, hidden],
        &[1, 1, 1],
        device,
    )?;

    // Mask audio-token ids to 0 for per-layer-input gating (matches vision).
    let mut masked: Vec<i32> = ids_i32;
    for &p in &audio_positions {
        masked[p] = 0;
    }
    let masked_arr = Array::from_i32_slice(&masked, &[seq as i32])?;

    Ok((embeds, masked_arr))
}

// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "unified_tests.rs"]
mod unified_tests;
