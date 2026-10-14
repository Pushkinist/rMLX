//! Long-form Whisper transcription engine.
//!
//! This is the single transcription core shared by the HTTP
//! `POST /v1/audio/transcriptions` route and the `rmlx transcribe` CLI. It
//! runs a sliding-window seek loop modelled on `openai-whisper` / `mlx_whisper` `transcribe()`:
//!
//! 1. Decode the input container to 16 kHz mono f32 (caller's responsibility;
//!    see [`crate::wav::WavDecoder::decode_resampled`]).
//! 2. Walk the audio in 30 s windows. For each window, run the decoder in
//!    **timestamp mode** with the full [`crate::whisper::DecodeFilters`] chain.
//!    A window is skipped when its no-speech probability is above 0.6 and
//!    its greedy decode with no prompt writes no more than a stock phrase
//!    (`fallback`). A decode that is a repetition loop or has a low average
//!    log-probability is decoded again, up a temperature ladder (`fallback`).
//! 3. Parse the emitted timestamp tokens into segments with real cumulative
//!    times. Consume the whole window when the tokens end in a single
//!    timestamp; otherwise seek to the end of the last closed segment. Feed the
//!    previous window's text back as a prompt (`<|startofprev|>`).
//! 4. Emit multi-segment output (`vtt` / `srt` / `json` / `txt`).
//!
//! Determinism: the caller supplies the token pick ([`RowPick`]). A pick that
//! draws from one seeded generator for each request gives the same output for
//! the same audio on every run, also when a fallback above temperature 0 ran.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rmlx_mlx::Device;
use tracing::{debug, info, warn};

use crate::mel::{pad_after_content, MelExtractor, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::{
    language_name, language_token, WhisperTask, WhisperTokenizer, TOK_EOT, TOK_SOT_PREV,
    TOK_TIMESTAMP_BEGIN,
};
use crate::whisper::{DecodeFilters, WhisperError, WhisperModel};

mod fallback;
mod seek;

use fallback::{decode_with_fallback, WindowOutcome};
use seek::{plan_window, walk_windows, window_segments, SeekRule, WalkEnd};

/// Chooses one token id from a filtered logit row at a temperature: the
/// largest logit at 0, a draw from `softmax(row / temperature)` above it. A
/// masked id has the logit `-inf`. One pick serves every window of a request,
/// so its generator is seeded once for the request.
pub type RowPick<'a> = &'a mut dyn FnMut(&[f32], f32) -> rmlx_core::error::Result<u32>;

/// Compression ratio above which a window's text is a repetition loop
/// (openai-whisper's `compression_ratio_threshold`).
const COMPRESSION_RATIO_LOOP: f32 = 2.4;

/// Max length of the previous-text prompt fed back as `<|startofprev|>` context,
/// derived from the decoder context length at runtime (no fixed literal).
///
/// Mirrors openai-whisper's `prompt[-(n_text_ctx // 2 - 1):]`: half the context
/// minus one, so the prompt leaves room for the SOT_PREV marker, the SOT prefix,
/// and the per-window generation budget without overrunning `n_text_ctx`.
#[must_use]
fn previous_text_cap(n_text_ctx: usize) -> usize {
    (n_text_ctx / 2).saturating_sub(1)
}

/// Per-window decoder generation budget, derived from `n_text_ctx` at runtime.
///
/// openai-whisper uses `sample_len = n_text_ctx // 2`, but the hard ceiling is
/// that the decoder position must stay `< n_text_ctx`: the positional-embedding
/// slice `[offset, offset+seq)` would otherwise run off the `[n_text_ctx, n_state]`
/// table and abort the transcription. `offset` starts at `prefix_len` and grows by
/// one per generated token, so the largest row requested is
/// `prefix_len + generated - 1`. Bounding `generated <= n_text_ctx - prefix_len`
/// keeps that `< n_text_ctx`. Returns 0 when the prefix already fills the context.
#[must_use]
fn window_token_budget(n_text_ctx: usize, prefix_len: usize) -> usize {
    let headroom = n_text_ctx.saturating_sub(prefix_len);
    (n_text_ctx / 2).min(headroom)
}

/// One transcribed segment with real wall-clock times (seconds from start).
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "stable public segment shape")]
pub struct Segment {
    /// Start of the window that holds the segment, in 10 ms mel frames, as
    /// openai-whisper's `seek`.
    pub seek: usize,
    /// Segment start time in seconds from the beginning of the audio.
    pub start: f32,
    /// Segment end time in seconds.
    pub end: f32,
    /// Decoded text (specials / timestamps stripped, trimmed).
    pub text: String,
    /// The segment's slice of the window's tokens, its timestamps included.
    pub tokens: Vec<u32>,
    /// Temperature of the window's kept decode: above the request temperature
    /// when a fallback ran. Every segment of a window has the same.
    pub temperature: f32,
    /// Compression ratio of the window's kept decode: every segment of a
    /// window has the same, as in openai-whisper.
    pub compression_ratio: f32,
}

/// Full transcription result.
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "stable public result shape")]
pub struct Transcription {
    /// Concatenated full text.
    pub text: String,
    /// Per-segment breakdown with timestamps.
    pub segments: Vec<Segment>,
    /// Language of the decode: the lower-case English name, for example
    /// `english`, whether the request named it or the model detected it.
    pub language: String,
    /// Total audio duration in seconds.
    pub duration: f32,
}

/// Options for a transcription run.
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "small, stable options bag")]
pub struct TranscribeOptions {
    /// Language: an ISO code (`"en"`), an English name (`"French"`), in any
    /// letter case, or `"auto"` for detection.
    pub language: String,
    /// Transcribe (same language) or translate (force English).
    pub task: WhisperTask,
    /// Temperature of the first decode of each window. A decode that is a
    /// repetition loop or is weak is decoded again at this value plus 0.2,
    /// and so on up to 1.0; a value above 1.0 gives one decode.
    pub temperature: f32,
    /// Feed the previous window's text back as a decoder prompt.
    pub condition_on_previous_text: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: "auto".to_owned(),
            task: WhisperTask::Transcribe,
            temperature: 0.0,
            condition_on_previous_text: true,
        }
    }
}

/// Long-form Whisper transcriber. Construct once, reuse across requests.
pub struct Transcriber {
    model: Arc<WhisperModel>,
    tokenizer: Arc<WhisperTokenizer>,
    extractor: MelExtractor,
    /// Tokenizer-derived non-speech / special suppression set.
    suppress: Vec<u32>,
    /// SuppressBlank ids (EOT + blank-space token).
    blank_ids: Vec<u32>,
}

impl std::fmt::Debug for Transcriber {
    /// Print config dims + suppression-set sizes; the model/tokenizer/extractor
    /// are opaque (large MLX buffers), so they are summarised, not dumped.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transcriber")
            .field("n_vocab", &self.model.cfg.n_vocab)
            .field("n_text_ctx", &self.model.cfg.n_text_ctx)
            .field("n_mels", &self.model.cfg.n_mels)
            .field("suppress_len", &self.suppress.len())
            .field("blank_ids_len", &self.blank_ids.len())
            .finish_non_exhaustive()
    }
}

impl Transcriber {
    /// Build a transcriber from a loaded model + tokenizer.
    pub fn new(
        model: Arc<WhisperModel>,
        tokenizer: Arc<WhisperTokenizer>,
    ) -> Result<Self, WhisperError> {
        let extractor = MelExtractor::new(model.cfg.n_mels)
            .map_err(|e| WhisperError::Mlx(format!("mel extractor: {e}")))?;
        let suppress = tokenizer.suppress_tokens();
        // SuppressBlank: EOT + whatever " " encodes to (general, from the tokenizer).
        let mut blank_ids = vec![TOK_EOT];
        if let Ok(space) = tokenizer.encode(" ") {
            blank_ids.extend(space);
        }
        Ok(Self {
            model,
            tokenizer,
            extractor,
            suppress,
            blank_ids,
        })
    }

    /// Transcribe a full 16 kHz mono f32 waveform of any length.
    ///
    /// `pick` chooses every token of the request. `cancel` is read before each
    /// 30 s window. Once it is set, no further window opens and the call
    /// returns [`WhisperError::Cancelled`]; a flag that is never set runs the
    /// whole audio.
    #[allow(
        clippy::too_many_lines,
        reason = "the seek loop + segment accumulation is one cohesive long-form pass"
    )]
    pub fn transcribe(
        &self,
        samples: &[f32],
        opts: &TranscribeOptions,
        pick: RowPick<'_>,
        device: Device,
        cancel: &AtomicBool,
    ) -> Result<Transcription, WhisperError> {
        let total_samples = samples.len();
        let duration = total_samples as f32 / SAMPLE_RATE as f32;

        // A named language is known now; `auto` is detected on the first
        // window that is decoded.
        let mut resolved_lang = requested_language(&opts.language)?;

        let mut segments: Vec<Segment> = Vec::new();
        let mut prompt_tokens: Vec<u32> = Vec::new();
        let mut skipped_windows = 0_usize;

        let filters = DecodeFilters::new(self.suppress.clone(), self.blank_ids.clone(), true);

        let end = walk_windows(
            total_samples,
            cancel,
            |seek, window_samples| -> Result<usize, WhisperError> {
                #[allow(
                    clippy::indexing_slicing,
                    reason = "walk_windows yields seek + window_samples <= total_samples"
                )]
                let window = &samples[seek..seek + window_samples];
                let window_start_time = seek as f32 / SAMPLE_RATE as f32;
                // Real audio length of this (un-padded) window, in seconds.
                let window_dur = window_samples as f32 / SAMPLE_RATE as f32;
                let projected_before = self.model.decoder.cross_projections();

                let mut mel_frames = self
                    .extractor
                    .extract(window)
                    .map_err(|e| WhisperError::Mlx(format!("mel: {e}")))?;
                // A short window has two inputs, as in openai-whisper: the
                // language is read from the mel of the zero-padded audio, and
                // the gate and the decode read the mel padded with 0.0.
                let detected_on_padded_audio =
                    if detection_input_needed(resolved_lang, window_samples) {
                        let padded_audio = self.model.encode_mel(&mel_frames, device)?;
                        let (start, _) = self.model.decoder.open_window(&padded_audio, device)?;
                        Some(start.language)
                    } else {
                        None
                    };
                pad_after_content(&mut mel_frames, window_samples);
                let encoder_out = self.model.encode_mel(&mel_frames, device)?;

                let (start, opened) = self.model.decoder.open_window(&encoder_out, device)?;
                let detected = detected_on_padded_audio.unwrap_or(start.language);
                // The language of the file is that of the first window that is
                // decoded: the bounded decode of the gate reads the language of
                // the window and does not store it.
                let window_lang = resolved_lang.unwrap_or(detected);
                // Timestamp mode: no <|notimestamps|> in the SOT sequence.
                let sot = self
                    .tokenizer
                    .sot_sequence_from_tok(window_lang, opts.task, true);
                let n_text_ctx = self.model.cfg.n_text_ctx;
                // The two closures of the window pick in turn, never at once.
                let pick = RefCell::new(&mut *pick);
                let prompt: &[u32] = if opts.condition_on_previous_text {
                    &prompt_tokens
                } else {
                    &[]
                };
                // The greedy decode with no prompt, until its text passes the bound.
                let writes_at_most = |text_bound: usize| {
                    let max_tokens = window_token_budget(n_text_ctx, sot.len());
                    opened.writes_at_most(
                        text_bound,
                        &sot,
                        max_tokens,
                        &filters,
                        |row: &[f32]| (pick.borrow_mut())(row, 0.0).map_err(WhisperError::from),
                    )
                };
                let decode = |prompt: &[u32], temperature: f32| {
                    resolve_language(&mut resolved_lang, detected);
                    let mut full_prefix: Vec<u32> = Vec::new();
                    if !prompt.is_empty() {
                        full_prefix.push(TOK_SOT_PREV);
                        full_prefix.extend(prompt.iter().copied());
                    }
                    full_prefix.extend(sot.iter().copied());

                    // Bounded so the decoder position stays `< n_text_ctx`.
                    let max_tokens = window_token_budget(n_text_ctx, full_prefix.len());
                    opened.decode(&full_prefix, max_tokens, &filters, |row: &[f32]| {
                        (pick.borrow_mut())(row, temperature).map_err(WhisperError::from)
                    })
                };
                let text_of = |tokens: &[u32]| {
                    let text: Vec<u32> = tokens
                        .iter()
                        .copied()
                        .filter(|&t| t < TOK_TIMESTAMP_BEGIN)
                        .collect();
                    self.tokenizer.decode(&text).unwrap_or_default()
                };

                let outcome = decode_with_fallback(
                    start.no_speech_prob,
                    writes_at_most,
                    prompt,
                    opts.temperature,
                    decode,
                    text_of,
                )?;
                outcome.log(seek, start.no_speech_prob);
                skipped_windows += usize::from(outcome.skipped);

                let plan = plan_window(&outcome.tokens, window_samples);
                if plan.rule == SeekRule::ZeroFallback {
                    warn!(
                        seek,
                        window_samples,
                        "last closed segment ends at the window start; consuming the whole window"
                    );
                }
                let window_segments = outcome.kept().map_or_else(Vec::new, |kept| {
                    window_segments(&plan.segments, seek, total_samples, kept, |tokens| {
                        self.tokenizer.decode(tokens).unwrap_or_default()
                    })
                });

                debug!(
                    seek,
                    window_start_time,
                    window_dur,
                    no_speech_prob = start.no_speech_prob,
                    skipped = outcome.skipped,
                    detection_pass = detected_on_padded_audio.is_some(),
                    decodes = outcome.attempts.len(),
                    cross_projections = self.model.decoder.cross_projections() - projected_before,
                    n_tokens = outcome.tokens.len(),
                    n_segments = window_segments.len(),
                    seek_rule = ?plan.rule,
                    advance_samples = plan.advance_samples,
                    "long-form window done"
                );

                let window_text: String = window_segments
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                prompt_tokens = next_prompt(
                    std::mem::take(&mut prompt_tokens),
                    &outcome,
                    opts.condition_on_previous_text,
                    &window_text,
                    |text| self.tokenizer.encode(text).unwrap_or_default(),
                    previous_text_cap(self.model.cfg.n_text_ctx),
                );

                segments.extend(window_segments);
                Ok(plan.advance_samples)
            },
        )?;
        if let WalkEnd::Cancelled { window, seek } = end {
            info!(
                window,
                seek,
                seek_secs = seek as f32 / SAMPLE_RATE as f32,
                total_samples,
                n_segments = segments.len(),
                "long-form transcription cancelled before the window opened"
            );
            return Err(WhisperError::Cancelled { window, seek });
        }

        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        let language = reported_language(resolved_lang, skipped_windows).to_owned();

        info!(
            n_segments = segments.len(),
            duration, "long-form transcription complete"
        );

        Ok(Transcription {
            text,
            segments,
            language,
            duration,
        })
    }
}

/// The language token a request names, or `None` for `auto`. A language that
/// Whisper does not know is an error that names it.
pub fn requested_language(language: &str) -> Result<Option<u32>, WhisperError> {
    if language.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    language_token(language)
        .map(Some)
        .ok_or_else(|| WhisperError::UnknownLanguage(language.to_owned()))
}

/// Whether a window of `window_samples` samples gets a second encoder pass, on
/// the mel of its zero-padded audio, to read its language. Only a window
/// shorter than 30 s has that second input, and it is read only while the file
/// has no language: `resolved` holds the token of a named language or of an
/// earlier decoded window.
fn detection_input_needed(resolved: Option<u32>, window_samples: usize) -> bool {
    resolved.is_none() && window_samples < N_SAMPLES
}

/// The language token of a window that is decoded. `resolved` holds the token
/// of a named language or of an earlier decoded window; otherwise `detected`,
/// the language of this window, is stored. So the first window that is
/// decoded gives the language of the whole file.
fn resolve_language(resolved: &mut Option<u32>, detected: u32) -> u32 {
    *resolved.get_or_insert_with(|| {
        debug!(lang_tok = detected, "language detected");
        detected
    })
}

/// The language a transcription reports for the token its decode used. With
/// no token, no window was decoded: each window was skipped
/// (`skipped_windows` is their count), or the audio is shorter than one mel
/// hop and no window opened.
fn reported_language(token: Option<u32>, skipped_windows: usize) -> &'static str {
    token.and_then(language_name).unwrap_or_else(|| {
        if skipped_windows > 0 {
            debug!(
                skipped_windows,
                "no window was decoded, so no language was detected; reporting english"
            );
        } else {
            warn!("no window was decoded, so no language was detected; reporting english");
        }
        "english"
    })
}

/// The previous-text prompt for the next window. A window whose `outcome`
/// may not be carried (its kept decode is not accepted, or is accepted above
/// temperature 0.5) clears it. Otherwise, with `condition` set and some text in
/// the window, it is `encode(text)` cut to its last `cap` tokens, as
/// openai-whisper cuts its prompt; else the prompt is kept as it is.
fn next_prompt(
    prompt: Vec<u32>,
    outcome: &WindowOutcome,
    condition: bool,
    window_text: &str,
    encode: impl FnOnce(&str) -> Vec<u32>,
    cap: usize,
) -> Vec<u32> {
    let text = window_text.trim();
    if !outcome.carry_text {
        return Vec::new();
    }
    if !condition || text.is_empty() {
        return prompt;
    }
    let mut tokens = encode(text);
    let start = tokens.len().saturating_sub(cap);
    tokens.split_off(start)
}

/// Compression ratio of `text`, as openai-whisper's `compression_ratio`
/// (`whisper/utils.py`): the same bytes (the stripped text) over a zlib stream.
/// The level 6 encoder is miniz, not zlib, so a short text near 2.4 can fall on
/// the other side of the bound. A repetition loop compresses well.
fn compression_ratio(text: &str) -> f32 {
    let bytes = text.trim().as_bytes();
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(bytes, 6);
    bytes.len() as f32 / compressed.len() as f32
}

// ── Output formatters ───────────────────────────────────────────────────────

/// Output format for a transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "the supported format set is closed"
)]
pub enum OutputFormat {
    /// Plain concatenated text.
    Txt,
    /// `{"text", "language", "duration", "segments":[…]}` JSON.
    Json,
    /// SubRip subtitles.
    Srt,
    /// WebVTT subtitles.
    Vtt,
}

impl OutputFormat {
    /// Parse a format string (`txt|json|srt|vtt`), returning an error message on
    /// unrecognised input.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "txt" | "text" => Ok(Self::Txt),
            "json" => Ok(Self::Json),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            other => Err(format!(
                "unsupported format '{other}'; must be one of: txt, json, srt, vtt"
            )),
        }
    }
}

/// Render a transcription to the requested format string.
#[must_use]
pub fn render(t: &Transcription, fmt: OutputFormat) -> String {
    match fmt {
        OutputFormat::Txt => t.text.clone(),
        OutputFormat::Json => render_json(t),
        OutputFormat::Srt => render_srt(&t.segments),
        OutputFormat::Vtt => render_vtt(&t.segments),
    }
}

fn render_json(t: &Transcription) -> String {
    let segs: Vec<serde_json::Value> = t
        .segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            serde_json::json!({
                "id": i,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            })
        })
        .collect();
    serde_json::json!({
        "text": t.text,
        "language": t.language,
        "duration": t.duration,
        "segments": segs,
    })
    .to_string()
}

/// Format seconds as `HH:MM:SS,mmm` (SRT) or `HH:MM:SS.mmm` (VTT).
fn fmt_time(secs: f32, comma: bool) -> String {
    let total_ms = (secs.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    let sep = if comma { ',' } else { '.' };
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

fn render_srt(segments: &[Segment]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        let _ = write!(
            out,
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            fmt_time(s.start, true),
            fmt_time(s.end, true),
            s.text
        );
    }
    out
}

fn render_vtt(segments: &[Segment]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("WEBVTT\n\n");
    for s in segments {
        let _ = write!(
            out,
            "{} --> {}\n{}\n\n",
            fmt_time(s.start, false),
            fmt_time(s.end, false),
            s.text
        );
    }
    out
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
