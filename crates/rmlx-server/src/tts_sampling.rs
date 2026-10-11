//! Sampling configuration and token pick for `POST /v1/audio/speech`.
//!
//! A value resolves in this order: the request, the talker snapshot's
//! `generation_config.json`, the Qwen3-TTS reference default. The code
//! predictor reads the `subtalker_*` keys and has no repetition penalty.

use std::path::Path;

use rmlx_audio::tts::{PickHead, TokenPick, TtsError};
use rmlx_mlx::{Array, Device};
use rmlx_models::sampler::{select_token, Pcg32, PenaltyConfig, SamplerConfig};
use serde::Deserialize;

use crate::generation_config_io::{load_generation_config, GenerationConfig};

/// Sampling fields that a speech request can set. The names are those of
/// `/v1/chat/completions`.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub(crate) struct TtsSamplingRequest {
    /// Temperature of both heads. `0` is greedy decoding.
    pub(crate) temperature: Option<f32>,
    /// Top-k cutoff of both heads. `0` disables it.
    pub(crate) top_k: Option<u32>,
    /// Repetition penalty of the talker. The code predictor has none.
    pub(crate) repetition_penalty: Option<f32>,
    /// RNG seed of the request. Absent: the shared sampler's default seed.
    pub(crate) seed: Option<u64>,
}

impl TtsSamplingRequest {
    /// Refuse a `temperature` or a `repetition_penalty` out of bounds. `top_k`
    /// and `seed` are unsigned, so every parsed value is valid.
    pub(crate) fn check(&self) -> Result<(), String> {
        self.temperature
            .map_or(Ok(()), |t| check_temperature("temperature", t))?;
        self.repetition_penalty
            .map_or(Ok(()), |p| check_penalty("repetition_penalty", p))
    }
}

/// The smallest `temperature` above 0, and the smallest `repetition_penalty`.
///
/// The sampler divides a logit by the temperature, and divides or multiplies
/// it by the penalty. At `1e-6` a logit of magnitude `1e30` still gives a
/// finite f32 (the limit is about `3.4e38`), and a talker logit is far below
/// that. A smaller value only moves the result toward infinity, where the
/// sampler refuses the row.
const MIN_SAMPLING_VALUE: f32 = 1e-6;

// The bounds below hold for a request field and for a `generation_config.json`
// key. Their upper ends are those of `/v1/chat/completions`.

/// `0` (greedy; `-0.0` is `0`) or a value in `[1e-6, 2]`.
fn check_temperature(key: &str, temperature: f32) -> Result<(), String> {
    if temperature == 0.0 || (MIN_SAMPLING_VALUE..=2.0).contains(&temperature) {
        Ok(())
    } else {
        Err(format!("`{key}` must be 0 or in [1e-6, 2.0]"))
    }
}

/// Finite and at least `1e-6`.
fn check_penalty(key: &str, penalty: f32) -> Result<(), String> {
    if penalty.is_finite() && penalty >= MIN_SAMPLING_VALUE {
        Ok(())
    } else {
        Err(format!("`{key}` must be finite and at least 1e-6"))
    }
}

/// `[0, 1]`; the shared sampler reads both ends as "no top-p cut".
fn check_top_p(key: &str, top_p: f32) -> Result<(), String> {
    if (0.0..=1.0).contains(&top_p) {
        Ok(())
    } else {
        Err(format!("`{key}` must be in [0.0, 1.0]"))
    }
}

/// Sampler and penalties of one model head.
#[derive(Debug, Clone)]
pub(crate) struct HeadSampling {
    pub(crate) sampler: SamplerConfig,
    pub(crate) penalty: PenaltyConfig,
}

/// Resolved sampling of the talker (codec group 0) and the code predictor
/// (codec groups 1 to 15).
#[derive(Debug, Clone)]
pub(crate) struct TtsSampling {
    pub(crate) talker: HeadSampling,
    pub(crate) code_predictor: HeadSampling,
}

// Qwen3-TTS reference defaults (`hard_defaults` in `qwen3_tts_model.py`) for
// each key that neither the request nor the file sets.
const REFERENCE_TEMPERATURE: f32 = 0.9;
const REFERENCE_TOP_K: u32 = 50;
const REFERENCE_TOP_P: f32 = 1.0;
const REFERENCE_REPETITION_PENALTY: f32 = 1.05;

/// Resolve the sampling of both heads from `<model_dir>/generation_config.json`
/// and the request. With an empty request, this gives the snapshot's defaults.
/// A file value out of bounds is an error; the caller checks the request with
/// [`TtsSamplingRequest::check`].
pub(crate) fn resolve_tts_sampling(
    model_dir: &Path,
    req: &TtsSamplingRequest,
) -> rmlx_core::Result<TtsSampling> {
    let file = load_generation_config(model_dir)?.unwrap_or_default();
    let defaults = TtsSampling::from_file(&file);
    defaults.check_file_values().map_err(|reason| {
        let path = model_dir.join("generation_config.json");
        rmlx_core::Error::Other(format!("{}: {reason}", path.display()))
    })?;
    Ok(defaults.with_request(req))
}

impl TtsSampling {
    fn from_file(file: &GenerationConfig) -> Self {
        Self {
            talker: HeadSampling::from_file(
                file.do_sample,
                file.temperature,
                file.top_k,
                file.top_p,
                file.repetition_penalty
                    .unwrap_or(REFERENCE_REPETITION_PENALTY),
            ),
            code_predictor: HeadSampling::from_file(
                file.subtalker_dosample,
                file.subtalker_temperature,
                file.subtalker_top_k,
                file.subtalker_top_p,
                1.0,
            ),
        }
    }

    /// The request bounds, on the values that the file gave. The error names
    /// the file key.
    fn check_file_values(&self) -> Result<(), String> {
        for (head, prefix) in [(&self.talker, ""), (&self.code_predictor, "subtalker_")] {
            check_temperature(&format!("{prefix}temperature"), head.sampler.temperature)?;
            check_top_p(&format!("{prefix}top_p"), head.sampler.top_p)?;
        }
        check_penalty("repetition_penalty", self.talker.penalty.rep_penalty)
    }

    /// Apply the request fields: `temperature`, `top_k` and `seed` to both
    /// heads, `repetition_penalty` to the talker.
    pub(crate) fn with_request(mut self, req: &TtsSamplingRequest) -> Self {
        for head in [&mut self.talker, &mut self.code_predictor] {
            if let Some(temperature) = req.temperature {
                head.sampler.temperature = temperature;
            }
            if let Some(top_k) = req.top_k {
                head.sampler.top_k = top_k;
            }
            head.sampler.seed = req.seed;
        }
        if let Some(penalty) = req.repetition_penalty {
            self.talker.penalty.rep_penalty = penalty;
        }
        self
    }
}

impl HeadSampling {
    /// `do_sample: false` is greedy decoding; the repetition penalty still applies.
    fn from_file(
        do_sample: Option<bool>,
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        rep_penalty: f32,
    ) -> Self {
        let temperature = if do_sample == Some(false) {
            0.0
        } else {
            temperature.unwrap_or(REFERENCE_TEMPERATURE)
        };
        Self {
            sampler: SamplerConfig {
                temperature,
                top_p: top_p.unwrap_or(REFERENCE_TOP_P),
                top_k: top_k.unwrap_or(REFERENCE_TOP_K),
                min_p: 0.0,
                seed: None,
                top_logprobs_k: 0,
            },
            penalty: PenaltyConfig {
                rep_penalty,
                ..PenaltyConfig::default()
            },
        }
    }
}

/// The speech route's [`TokenPick`]: one RNG stream per request, shared by both
/// heads in pick order.
#[derive(Debug)]
pub(crate) struct TtsPicker {
    sampling: TtsSampling,
    rng: Pcg32,
    device: Device,
}

impl TtsPicker {
    /// Build a picker. The RNG is seeded from the talker's seed, or from the
    /// shared sampler's default seed when the request set none.
    pub(crate) fn new(sampling: TtsSampling, device: Device) -> Self {
        let rng = Pcg32::new(sampling.talker.sampler.seed_or_default());
        Self {
            sampling,
            rng,
            device,
        }
    }
}

impl TokenPick for TtsPicker {
    fn pick(
        &mut self,
        head: PickHead,
        logits: &Array,
        allow: Option<&[bool]>,
        history: &[u32],
    ) -> Result<u32, TtsError> {
        let head = match head {
            PickHead::Talker => &self.sampling.talker,
            PickHead::CodePredictor => &self.sampling.code_predictor,
        };
        let vocab = logits.shape().last().copied().unwrap_or(0);
        let row = logits.reshape(&[1, vocab], self.device)?;
        let id = select_token(
            &row,
            &head.sampler,
            allow,
            &head.penalty,
            history,
            &mut self.rng,
            self.device,
        )?;
        id.eval()?;
        let bytes = id.to_bytes()?;
        let first = bytes
            .first_chunk::<4>()
            .ok_or_else(|| TtsError::Mlx(format!("token id array holds {} bytes", bytes.len())))?;
        Ok(u32::from_le_bytes(*first))
    }
}

#[cfg(test)]
#[path = "tts_sampling_tests.rs"]
mod tts_sampling_tests;
