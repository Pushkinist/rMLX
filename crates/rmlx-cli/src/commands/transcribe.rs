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
//! `transcribe` straight at a `.m4a`, `.wav`, `.mp3` or `.aac` file.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{
    render, requested_language, OutputFormat, TranscribeOptions, Transcriber,
};
use rmlx_audio::wav::WavDecoder;
use rmlx_audio::whisper::WhisperModel;
use rmlx_mlx::Device;

use crate::commands::parse::ClaimedDevice;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tracing::info;

/// Arguments for `rmlx transcribe`.
#[allow(clippy::struct_excessive_bools, reason = "small CLI argument bag")]
pub(crate) struct TranscribeArgs<'a> {
    /// Input audio file: WAV, MP3, M4A or ADTS AAC.
    pub audio: &'a Path,
    /// Model snapshot directory.
    pub model: &'a Path,
    /// Optional companion tokenizer directory (Whisper snapshots ship no
    /// `tokenizer.json`; pass the `openai/whisper-large-v3` tokenizer dir).
    /// When absent, the model dir is tried.
    pub tokenizer: Option<&'a Path>,
    /// Output format: `txt | json | srt | vtt`.
    pub format: &'a str,
    /// ISO code (`en`), English name (`French`) or `auto`.
    pub language: &'a str,
    /// `true` => translate to English; `false` => transcribe in source language.
    pub translate: bool,
    /// Text that prompts the decoder; empty for none.
    pub prompt: &'a str,
    /// Add the time of each word to the JSON output.
    pub words: bool,
}

/// Dispatch on the snapshot's architecture and run transcription.
///
/// Returns the rendered output string (caller prints to stdout or writes a file).
pub(crate) fn run_transcribe(args: &TranscribeArgs, claimed: &ClaimedDevice) -> Result<String> {
    let device = claimed.device();
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
    if args.words && out_fmt != OutputFormat::Json {
        return Err(anyhow!("--words needs --format json"));
    }
    requested_language(args.language)?;

    // Resolve the tokenizer directory: explicit flag, else the model dir.
    let tok_dir = args.tokenizer.unwrap_or(args.model);

    info!(
        model = %args.model.display(),
        audio = %args.audio.display(),
        "rmlx transcribe (whisper)"
    );

    // A prompt that cannot be used is refused before the audio is read and
    // before the model loads; a blank prompt loads no tokenizer yet.
    let (prompt_tokens, early_tokenizer) = if args.prompt.trim().is_empty() {
        (Vec::new(), None)
    } else {
        let tokenizer = load_tokenizer(tok_dir)?;
        let tokens = tokenizer
            .encode_prompt(args.prompt)
            .map_err(|e| anyhow!("invalid prompt: {e}"))?;
        (tokens, Some(tokenizer))
    };

    // Decode first: a file the decoder refuses costs no model load. The file's
    // bytes are dropped here; only the 16 kHz samples stay for the model.
    let samples = read_audio_16k(args.audio)?;

    // The tokenizer next: one of another vocabulary is refused before the weights load.
    let tokenizer = match early_tokenizer {
        Some(tokenizer) => tokenizer,
        None => load_tokenizer(tok_dir)?,
    };
    let model = WhisperModel::load(args.model).context("load whisper model")?;

    let transcriber = Transcriber::new(Arc::new(model), Arc::new(tokenizer))
        .map_err(|e| anyhow!("transcriber init: {e}"))?;

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
        prompt_tokens,
        words: args.words,
    };

    let t0 = std::time::Instant::now();
    let transcription = rmlx_server::audio::transcribe_request(
        &transcriber,
        &samples,
        &opts,
        device,
        &AtomicBool::new(false),
        &mut |_| Ok(()),
    )
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

    render(&transcription, out_fmt).context("render the transcription")
}

/// The tokenizer in `tok_dir`.
fn load_tokenizer(tok_dir: &Path) -> Result<WhisperTokenizer> {
    WhisperTokenizer::from_path(tok_dir).with_context(|| {
        format!(
            "load whisper tokenizer from {} (Whisper snapshots ship no tokenizer.json; \
             pass --tokenizer pointing at the openai/whisper-large-v3 tokenizer dir)",
            tok_dir.display()
        )
    })
}

/// The 16 kHz mono samples of the audio file at `path`. The operator's own
/// file has no duration cap: the server's cap guards it from its clients.
fn read_audio_16k(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read audio {}", path.display()))?;
    let audio =
        WavDecoder::decode_resampled(&bytes, rmlx_audio::mel::SAMPLE_RATE, u32::MAX, &|| false)
            .map_err(|e| anyhow!("audio decode: {e}"))?;
    Ok(audio.samples)
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
