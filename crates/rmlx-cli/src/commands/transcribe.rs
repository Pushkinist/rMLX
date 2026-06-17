//! `rmlx transcribe <audio> --model <snapshot>` — speech-to-text.
//!
//! Arch-dispatched on the snapshot's `config.json`. Whisper is the first (and
//! currently only) ASR backend, but the dispatch is a clean seam: adding a new
//! ASR architecture means adding a match arm in [`run_transcribe`], not a new
//! subcommand. The `--model` is a user-supplied snapshot directory — this is a
//! general tool, not pinned to one machine's layout.
//!
//! The input container is decoded and resampled to 16 kHz mono internally (via
//! the shared `rmlx-audio` decode + linear-resample path), so a user can point
//! `transcribe` straight at an arbitrary `.m4a` / `.wav` / `.mp3` / `.flac`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{render, OutputFormat, TranscribeOptions, Transcriber};
use rmlx_audio::wav::WavDecoder;
use rmlx_audio::whisper::WhisperModel;
use rmlx_mlx::Device;
use std::sync::Arc;
use tracing::info;

/// Arguments for `rmlx transcribe`.
#[allow(clippy::struct_excessive_bools, reason = "small CLI argument bag")]
pub(crate) struct TranscribeArgs<'a> {
    /// Input audio file (any Symphonia-supported container).
    pub audio: &'a Path,
    /// Model snapshot directory.
    pub model: &'a Path,
    /// Optional companion tokenizer directory (Whisper snapshots ship no
    /// `tokenizer.json`; pass the `openai/whisper-large-v3` tokenizer dir).
    /// When absent, the model dir is tried.
    pub tokenizer: Option<&'a Path>,
    /// Output format: `txt | json | srt | vtt`.
    pub format: &'a str,
    /// Language code (`en`, `fr`, …) or `auto`.
    pub language: &'a str,
    /// `true` => translate to English; `false` => transcribe in source language.
    pub translate: bool,
}

/// Dispatch on the snapshot's architecture and run transcription.
///
/// Returns the rendered output string (caller prints to stdout or writes a file).
pub(crate) fn run_transcribe(args: &TranscribeArgs, device: Device) -> Result<String> {
    let cfg_path = args.model.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("read {}", cfg_path.display()))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&cfg_str).with_context(|| format!("parse {}", cfg_path.display()))?;

    // Arch seam: select the ASR backend from config.json. Whisper is keyed by
    // `model_type == "whisper"`; future ASR arches add an arm here.
    let model_type = cfg
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    match model_type {
        "whisper" => run_whisper(args, device),
        other => Err(anyhow!(
            "unsupported ASR architecture for `rmlx transcribe`: model_type='{other}' \
             (supported: whisper). The snapshot at {} is not a recognised ASR model.",
            args.model.display()
        )),
    }
}

/// Whisper backend for `rmlx transcribe`.
fn run_whisper(args: &TranscribeArgs, device: Device) -> Result<String> {
    let out_fmt = OutputFormat::parse(args.format).map_err(|e| anyhow!(e))?;

    // Resolve the tokenizer directory: explicit flag, else the model dir.
    let tok_dir = args.tokenizer.unwrap_or(args.model);

    info!(
        model = %args.model.display(),
        audio = %args.audio.display(),
        "rmlx transcribe (whisper)"
    );

    if device == Device::Gpu {
        rmlx_mlx::ensure_gpu_default_stream();
    }

    let model = WhisperModel::load(args.model).context("load whisper model")?;
    let tokenizer = WhisperTokenizer::from_path(tok_dir).with_context(|| {
        format!(
            "load whisper tokenizer from {} (Whisper snapshots ship no tokenizer.json; \
             pass --tokenizer pointing at the openai/whisper-large-v3 tokenizer dir)",
            tok_dir.display()
        )
    })?;

    let transcriber = Transcriber::new(Arc::new(model), Arc::new(tokenizer))
        .map_err(|e| anyhow!("transcriber init: {e}"))?;

    // Decode + resample to 16 kHz mono.
    let bytes = std::fs::read(args.audio)
        .with_context(|| format!("read audio {}", args.audio.display()))?;
    let (raw, src_rate) = WavDecoder::decode(&bytes).map_err(|e| anyhow!("audio decode: {e}"))?;
    let samples = rmlx_audio::transcribe::resample_to_16k(&raw, src_rate);

    let task = if args.translate {
        WhisperTask::Translate
    } else {
        WhisperTask::Transcribe
    };
    let opts = TranscribeOptions {
        language: args.language.to_owned(),
        task,
        temperature: 0.0,
        condition_on_previous_text: true,
    };

    let t0 = std::time::Instant::now();
    let transcription = transcriber
        .transcribe(&samples, &opts, device)
        .map_err(|e| anyhow!("transcribe: {e}"))?;
    let elapsed = t0.elapsed().as_secs_f64();
    let rtf = if transcription.duration > 0.0 {
        elapsed / f64::from(transcription.duration)
    } else {
        0.0
    };
    info!(
        segments = transcription.segments.len(),
        duration = transcription.duration,
        elapsed_s = elapsed,
        rtf,
        "transcribe complete"
    );

    Ok(render(&transcription, out_fmt))
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
