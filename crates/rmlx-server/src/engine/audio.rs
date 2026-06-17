//! Audio-prompt construction helpers shared by ArchGenerator.
//!
//! - `AudioBundle` — arch-specific multimodal audio components (loaded once)
//! - `build_audio_prompt` — Gemma4 audio embedding + token splice
//!
//! Mirrors `engine/image.rs` for the vision tower. The Gemma4 Conformer audio
//! tower turns raw `input_audio` (base64 wav/etc) into audio soft tokens that
//! are scattered into the prompt at `<|audio|>` placeholder positions.

use std::path::Path;

use rmlx_core::Error;

// ── Gemma4 audio token ids ────────────────────────────────────────────────

/// Gemma4 `<|audio>` begin-of-audio marker token id.
pub(crate) const GEMMA4_BOA_TOKEN_ID: u32 = 256_000;
/// Gemma4 `<audio|>` end-of-audio marker token id.
pub(crate) const GEMMA4_EOA_TOKEN_ID: u32 = 258_883;

/// Sample rate the Gemma4 USM front-end expects (16 kHz mono).
const GEMMA4_AUDIO_SAMPLE_RATE: u32 = 16_000;

/// bundle of the multimodal components needed to turn `input_audio` bytes into
/// scattered `inputs_embeds`. Loaded once per model when the snapshot ships an
/// `audio_config` + `audio_tower.*` weights. One variant per audio-capable
/// architecture (only Gemma4's Conformer tower today).
#[allow(missing_debug_implementations)]
pub(crate) enum AudioBundle {
    /// Gemma4 Conformer audio tower + multimodal embedder + feature extractor.
    Gemma4 {
        encoder: rmlx_models::gemma4::AudioEncoder,
        embedder: rmlx_models::gemma4::MultimodalEmbedder,
        feature_extractor: rmlx_models::gemma4::Gemma4AudioFeatureExtractor,
        audio_token_id: u32,
    },
}

/// Load the Gemma4 audio bundle (Conformer tower + `embed_audio` projector +
/// USM feature extractor) from a snapshot directory.
///
/// Returns `Ok(None)` for checkpoints without an `audio_config` (text-only or
/// vision-only). Errors propagate to the caller, which logs + disables the
/// audio path (audio input then returns a clear "no audio tower" error).
pub(crate) fn load_gemma4_audio_bundle(model_dir: &Path) -> rmlx_core::Result<Option<AudioBundle>> {
    let Some(acfg) = rmlx_models::gemma4::Gemma4AudioConfig::from_model_dir(model_dir)? else {
        return Ok(None);
    };
    let encoder = rmlx_models::gemma4::load_audio_tower(model_dir, &acfg)?;
    let embedder =
        rmlx_models::gemma4::load_multimodal_embedder(model_dir, "embed_audio", acfg.rms_norm_eps)?;
    let feature_extractor = load_gemma4_feature_extractor(model_dir)?;
    Ok(Some(AudioBundle::Gemma4 {
        encoder,
        embedder,
        feature_extractor,
        audio_token_id: acfg.audio_token_id,
    }))
}

/// Build the Gemma4 USM feature extractor from the snapshot's
/// `processor_config.json` (`feature_extractor` block). Falls back to the
/// documented USM defaults when the file or block is absent — the front-end
/// parameters are fixed for all Gemma 4 checkpoints.
fn load_gemma4_feature_extractor(
    model_dir: &Path,
) -> rmlx_core::Result<rmlx_models::gemma4::Gemma4AudioFeatureExtractor> {
    use rmlx_models::gemma4::{AudioFeatureExtractorConfig, Gemma4AudioFeatureExtractor};

    let path = model_dir.join("processor_config.json");
    match std::fs::read_to_string(&path) {
        Ok(json) => Gemma4AudioFeatureExtractor::from_processor_config_str(&json)
            .map_err(|e| Error::Other(format!("gemma4 audio feature extractor: {e}"))),
        Err(_) => {
            // No processor_config.json — use the fixed USM defaults.
            Ok(Gemma4AudioFeatureExtractor::from_config(
                serde_json::from_str::<AudioFeatureExtractorConfig>("{}")
                    .map_err(|e| Error::Other(format!("gemma4 audio default config: {e}")))?,
            ))
        }
    }
}

/// Decode + preprocess `input_audio` clips, expand the prompt with per-clip
/// audio soft-token blocks, run the Conformer tower, and build the
/// scatter-merged `inputs_embeds`.
///
/// Pipeline (mirrors mlx-vlm `processing_gemma4` + `get_input_embeddings`):
/// 1. base64 → bytes → `WavDecoder` (mono f32, native rate) → resample 16 kHz.
/// 2. `Gemma4AudioFeatureExtractor::extract` → log-mel `[T, 128]`.
/// 3. `encoder.num_output_frames(T)` → `T_sub`; splice the audio block
///    `<|audio>` + `T_sub` × `<|audio|>` + `<audio|>` after the leading token.
/// 4. `gemma4::build_audio_inputs_embeds` runs the Conformer tower + projector
///    and scatters the audio soft tokens at the `<|audio|>` positions.
///
/// Currently serves exactly one audio clip per request; >1 is rejected with a
/// clear error rather than silently mis-scattering.
///
/// Returns `(augmented_ids, inputs_embeds [1, seq, hidden], masked_ids [seq])`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length or validated before call"
)]
pub(crate) fn build_audio_prompt(
    arch: &rmlx_models::arch::Architecture,
    ab: &AudioBundle,
    audio_b64: &[String],
    prompt_tokens: &[u32],
    device: rmlx_mlx::Device,
) -> rmlx_core::Result<(Vec<u32>, rmlx_mlx::Array, rmlx_mlx::Array)> {
    if audio_b64.len() != 1 {
        return Err(Error::Other(format!(
            "audio input currently serves exactly one audio clip per request (got {})",
            audio_b64.len()
        )));
    }

    let AudioBundle::Gemma4 {
        encoder,
        embedder,
        feature_extractor,
        audio_token_id,
    } = ab;

    let model = arch
        .as_gemma4()
        .ok_or_else(|| Error::Other("audio input requires the Gemma4 architecture".to_owned()))?;

    // 1. base64 → bytes → mono f32 @ 16 kHz.
    let raw = audio_b64[0].as_str();
    // Tolerate a `data:audio/...;base64,` prefix even though the OpenAI shape
    // carries the bare base64 in `input_audio.data`.
    let b64 = raw.rsplit_once(',').map_or(raw, |(_, tail)| tail);
    let bytes = crate::image_io::base64_decode(b64)
        .map_err(|e| Error::Other(format!("input_audio base64 decode failed: {e}")))?;
    let (samples, sample_rate) = rmlx_audio::wav::WavDecoder::decode(&bytes)
        .map_err(|e| Error::Other(format!("input_audio decode failed: {e}")))?;
    let samples = rmlx_audio::transcribe::resample_to_16k(&samples, sample_rate);
    let dur_secs = samples.len() as f64 / f64::from(GEMMA4_AUDIO_SAMPLE_RATE);

    // 2. log-mel features [T, 128].
    let mel = feature_extractor
        .extract(&samples)
        .map_err(|e| Error::Other(format!("input_audio mel extraction failed: {e}")))?;
    let t_mel = mel.len();
    if t_mel == 0 {
        return Err(Error::Other(
            "input_audio produced zero mel frames (clip too short)".to_owned(),
        ));
    }
    let feature_size = mel[0].len();

    // 3. number of audio soft tokens this clip yields (encoder output frames).
    let t_sub = encoder.num_output_frames(t_mel);
    if t_sub == 0 {
        return Err(Error::Other(
            "input_audio produced zero audio soft tokens (clip too short)".to_owned(),
        ));
    }

    tracing::info!(
        sample_rate,
        samples = samples.len(),
        duration_secs = dur_secs,
        mel_frames = t_mel,
        feature_size,
        audio_soft_tokens = t_sub,
        "Gemma4 audio preprocessed"
    );

    // Flatten mel into a [1, T, feature_size] f32 array; all-zero padding mask
    // (single clip, no batch padding → every frame valid; 1.0 = invalid).
    let mut flat = Vec::with_capacity(t_mel * feature_size);
    for frame in &mel {
        flat.extend_from_slice(frame);
    }
    let mel_arr = rmlx_mlx::Array::from_f32_slice(&flat, &[1, t_mel as i32, feature_size as i32])?;
    let mask_arr = rmlx_mlx::Array::from_f32_slice(&vec![0.0_f32; t_mel], &[1, t_mel as i32])?;

    // 4. splice the audio block in after the prompt's leading BOS token so the
    // clip conditions the whole turn causally (mirrors the image splice).
    let mut block = Vec::with_capacity(t_sub + 2);
    block.push(GEMMA4_BOA_TOKEN_ID);
    block.extend(std::iter::repeat_n(*audio_token_id, t_sub));
    block.push(GEMMA4_EOA_TOKEN_ID);

    let insert_at = usize::from(!prompt_tokens.is_empty());
    let mut aug_ids = Vec::with_capacity(prompt_tokens.len() + block.len());
    aug_ids.extend_from_slice(&prompt_tokens[..insert_at]);
    aug_ids.extend_from_slice(&block);
    aug_ids.extend_from_slice(&prompt_tokens[insert_at..]);

    let in_prompt = aug_ids.iter().filter(|&&t| t == *audio_token_id).count();
    tracing::info!(
        audio_soft_tokens = t_sub,
        audio_tokens_in_prompt = in_prompt,
        aug_len = aug_ids.len(),
        "built Gemma4 audio prompt"
    );

    let (embeds, masked_ids) = rmlx_models::gemma4::build_audio_inputs_embeds(
        model,
        encoder,
        embedder,
        &mel_arr,
        &mask_arr,
        *audio_token_id,
        &aug_ids,
        device,
    )?;
    Ok((aug_ids, embeds, masked_ids))
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;
