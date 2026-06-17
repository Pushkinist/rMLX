// LOC-exempt: the long-form transcription engine (window seek loop + timestamp
// segmentation + previous-text conditioning + subtitle formatting) is one
// cohesive sequential pipeline; splitting the seek loop from the formatters
// would scatter the timestamp arithmetic without cohesion gain.

//! Long-form Whisper transcription engine.
//!
//! This is the single transcription core shared by the HTTP
//! `POST /v1/audio/transcriptions` route and the `rmlx transcribe` CLI. It
//! replaces the old "first 30 s only" behaviour with a sliding-window seek
//! loop modelled on `openai-whisper` / `mlx_whisper` `transcribe()`:
//!
//! 1. Decode the input container to 16 kHz mono f32 (caller's responsibility;
//!    see [`crate::wav::WavDecoder`] + [`resample_to_16k`]).
//! 2. Walk the audio in 30 s windows. For each window, run the decoder in
//!    **timestamp mode** with the full [`crate::whisper::DecodeFilters`] chain.
//! 3. Parse the emitted timestamp tokens into segments with real cumulative
//!    times, advance the seek position by the last consumed timestamp, and feed
//!    the previous window's text back as a prompt (`<|startofprev|>`).
//! 4. Emit multi-segment output (`vtt` / `srt` / `json` / `txt`).
//!
//! Determinism: temperature is fixed at 0 (greedy argmax), so the same audio
//! produces byte-identical output across runs.

use std::sync::Arc;

use rmlx_mlx::Device;
use tracing::{debug, info};

use crate::mel::{MelExtractor, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::{WhisperTask, WhisperTokenizer, TOK_EOT, TOK_SOT_PREV, TOK_TIMESTAMP_BEGIN};
use crate::whisper::{DecodeFilters, WhisperError, WhisperModel};

/// Seconds of audio represented by one timestamp-token step (Whisper uses 0.02 s).
const TIME_PRECISION: f32 = 0.02;

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
    /// Segment start time in seconds from the beginning of the audio.
    pub start: f32,
    /// Segment end time in seconds.
    pub end: f32,
    /// Decoded text (specials / timestamps stripped, trimmed).
    pub text: String,
}

/// Full transcription result.
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "stable public result shape")]
pub struct Transcription {
    /// Concatenated full text.
    pub text: String,
    /// Per-segment breakdown with timestamps.
    pub segments: Vec<Segment>,
    /// Resolved language (BCP-47 code or `lang_tok=N` when auto-detected).
    pub language: String,
    /// Total audio duration in seconds.
    pub duration: f32,
}

/// Options for a transcription run.
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "small, stable options bag")]
pub struct TranscribeOptions {
    /// Language: a BCP-47 code (`"en"`, `"fr"`, …) or `"auto"` for detection.
    pub language: String,
    /// Transcribe (same language) or translate (force English).
    pub task: WhisperTask,
    /// Sampling temperature. 0 = deterministic greedy (the only supported path).
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
    #[allow(
        clippy::too_many_lines,
        reason = "the seek loop + segment accumulation is one cohesive long-form pass"
    )]
    pub fn transcribe(
        &self,
        samples: &[f32],
        opts: &TranscribeOptions,
        device: Device,
    ) -> Result<Transcription, WhisperError> {
        let total_samples = samples.len();
        let duration = total_samples as f32 / SAMPLE_RATE as f32;

        // Resolve language once on the first window (auto-detect needs an encode).
        let mut resolved_lang: Option<(u32, String)> = None;

        let mut segments: Vec<Segment> = Vec::new();
        let mut prompt_tokens: Vec<u32> = Vec::new();
        let mut seek: usize = 0; // sample offset of the current window start

        let filters = DecodeFilters::new(self.suppress.clone(), self.blank_ids.clone(), true);

        while seek < total_samples {
            let window_end = (seek + N_SAMPLES).min(total_samples);
            #[allow(
                clippy::indexing_slicing,
                reason = "seek < total_samples (loop guard) and window_end = min(seek+N, total_samples), so seek..window_end is always in bounds"
            )]
            let window = &samples[seek..window_end];
            let window_start_time = seek as f32 / SAMPLE_RATE as f32;
            // Real audio length of this (un-padded) window, in seconds.
            let window_dur = (window_end - seek) as f32 / SAMPLE_RATE as f32;

            // mel + encode (mel pads to 30 s internally).
            let mel_frames = self
                .extractor
                .extract(window)
                .map_err(|e| WhisperError::Mlx(format!("mel: {e}")))?;
            let encoder_out = self.model.encode_mel(&mel_frames, device)?;

            // Resolve language on first window.
            if resolved_lang.is_none() {
                let (lang_tok, lang_str) = if opts.language == "auto" {
                    let t = self
                        .model
                        .detect_language(&encoder_out, device)
                        .unwrap_or(crate::tokenizer::TOK_EN);
                    (t, format!("lang_tok={t}"))
                } else {
                    (
                        crate::tokenizer::language_token(&opts.language),
                        opts.language.clone(),
                    )
                };
                debug!(lang_tok, "language resolved");
                resolved_lang = Some((lang_tok, lang_str));
            }
            let lang_tok = resolved_lang
                .as_ref()
                .map_or(crate::tokenizer::TOK_EN, |(t, _)| *t);

            // Build the SOT sequence (timestamp mode → no <|notimestamps|>), with
            // optional previous-text prompt.
            let sot = self
                .tokenizer
                .sot_sequence_from_tok(lang_tok, opts.task, true);
            let mut full_prefix: Vec<u32> = Vec::new();
            if opts.condition_on_previous_text && !prompt_tokens.is_empty() {
                full_prefix.push(TOK_SOT_PREV);
                full_prefix.extend(prompt_tokens.iter().copied());
            }
            full_prefix.extend(sot.iter().copied());

            // Per-window generation budget, derived from `n_text_ctx` at runtime
            // (no fixed literal); bounded so the decoder position stays `< n_text_ctx`.
            // greedy_decode additionally refuses any positional row `>= n_text_ctx`
            // as a belt-and-suspenders guard.
            let max_tokens = window_token_budget(self.model.cfg.n_text_ctx, full_prefix.len());

            let tokens = match self.model.greedy_decode(
                &encoder_out,
                &full_prefix,
                max_tokens,
                opts.temperature,
                &filters,
                device,
            ) {
                Ok(t) => t,
                Err(WhisperError::Silence) => {
                    // Whole window is silence — skip ahead a full window.
                    seek += N_SAMPLES;
                    continue;
                }
                Err(e) => return Err(e),
            };

            // Parse timestamp tokens into segments within this window.
            let (window_segments, consumed_time) =
                self.split_window(&tokens, window_start_time, window_dur)?;

            // Advance seek by the time we actually consumed.
            let advance_secs = consumed_time.clamp(0.0, window_dur);
            let advance_samples = (advance_secs * SAMPLE_RATE as f32) as usize;
            // Guarantee forward progress even when no timestamp was emitted.
            let advance_samples = advance_samples.max(1).min(window_end - seek);
            // If we consumed essentially nothing but the window is full-size,
            // jump a whole window to avoid stalling.
            let advance_samples =
                if advance_secs <= TIME_PRECISION && window_end - seek >= N_SAMPLES {
                    N_SAMPLES
                } else {
                    advance_samples
                };

            debug!(
                seek,
                window_start_time,
                window_dur,
                n_tokens = tokens.len(),
                n_segments = window_segments.len(),
                consumed_time,
                advance_secs,
                advance_samples,
                "long-form window done"
            );

            // Update the prompt for the next window from this window's text.
            if opts.condition_on_previous_text {
                let window_text: String = window_segments
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                if !window_text.trim().is_empty() {
                    prompt_tokens = self
                        .tokenizer
                        .encode(window_text.trim())
                        .unwrap_or_default();
                    // Cap the previous-text prompt the way openai-whisper does:
                    // the prefix never overruns n_text_ctx once the SOT_PREV marker
                    // + SOT prefix + generation budget are added.
                    let cap = previous_text_cap(self.model.cfg.n_text_ctx);
                    if prompt_tokens.len() > cap {
                        let start = prompt_tokens.len() - cap;
                        prompt_tokens = prompt_tokens.split_off(start);
                    }
                }
            }

            segments.extend(window_segments);
            seek += advance_samples;
        }

        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        let language = resolved_lang.map_or_else(|| "en".to_owned(), |(_, s)| s);

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

    /// Split one window's token stream into timestamped segments.
    ///
    /// Returns `(segments, consumed_time_secs)` where `consumed_time_secs` is the
    /// last timestamp boundary used to advance the seek (relative to the window
    /// start, 0..30 s).
    #[allow(
        clippy::unnecessary_wraps,
        reason = "returns Result for symmetry with tokenizer.decode error paths"
    )]
    fn split_window(
        &self,
        tokens: &[u32],
        window_start_time: f32,
        window_dur: f32,
    ) -> Result<(Vec<Segment>, f32), WhisperError> {
        let mut segments: Vec<Segment> = Vec::new();
        let mut last_ts_time: Option<f32> = None; // relative seconds within window
        let mut cur_text: Vec<u32> = Vec::new();
        let mut seg_start: Option<f32> = None;
        let mut consumed_time = 0.0_f32;

        let ts_to_secs = |tok: u32| -> f32 { (tok - TOK_TIMESTAMP_BEGIN) as f32 * TIME_PRECISION };

        // A segment whose opening timestamp sits past the real (un-padded) audio
        // length is in the 30 s zero-pad tail — Whisper hallucinates filler there
        // ("you", "thank you", "♪"). Drop those. A small tolerance absorbs the
        // 0.02 s timestamp granularity.
        let speech_limit = window_dur + 0.5;
        let in_speech = |start: f32| start <= speech_limit;

        for &tok in tokens {
            if tok == TOK_EOT {
                break;
            }
            if tok >= TOK_TIMESTAMP_BEGIN {
                let t = ts_to_secs(tok);
                match (seg_start, last_ts_time) {
                    (None, _) => {
                        // Opening timestamp of a segment.
                        seg_start = Some(t);
                    }
                    (Some(start), _) => {
                        // Closing timestamp — flush the accumulated text.
                        let text = self.tokenizer.decode(&cur_text).unwrap_or_default();
                        let trimmed = text.trim();
                        if !trimmed.is_empty() && in_speech(start) {
                            segments.push(Segment {
                                start: window_start_time + start,
                                end: window_start_time + t,
                                text: trimmed.to_owned(),
                            });
                        }
                        cur_text.clear();
                        seg_start = None;
                    }
                }
                last_ts_time = Some(t);
                consumed_time = t;
            } else {
                cur_text.push(tok);
            }
        }

        // Flush a dangling open segment (text with an opening timestamp but no
        // closing one — happens when the window cuts mid-utterance).
        if let Some(start) = seg_start {
            let text = self.tokenizer.decode(&cur_text).unwrap_or_default();
            let trimmed = text.trim();
            if !trimmed.is_empty() && in_speech(start) {
                let end = last_ts_time.unwrap_or(window_dur).max(start);
                segments.push(Segment {
                    start: window_start_time + start,
                    end: window_start_time + end,
                    text: trimmed.to_owned(),
                });
            }
        } else if segments.is_empty() && !cur_text.is_empty() {
            // No timestamps at all — emit the whole window as one segment.
            let text = self.tokenizer.decode(&cur_text).unwrap_or_default();
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                segments.push(Segment {
                    start: window_start_time,
                    end: window_start_time + window_dur,
                    text: trimmed.to_owned(),
                });
            }
        }

        // If no usable timestamp boundary was found, consume the whole window.
        if consumed_time <= TIME_PRECISION {
            consumed_time = window_dur;
        }

        Ok((segments, consumed_time))
    }
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

// ── 48 kHz / stereo → 16 kHz mono resampler ─────────────────────────────────

/// Resample mono f32 `samples` from `src_rate` Hz to 16 kHz using linear
/// interpolation.
///
/// Whisper is forgiving of resampler quality (its mel front-end is robust), so a
/// linear resampler is sufficient and avoids a new dependency. Stereo downmix is
/// already handled by [`crate::wav::WavDecoder`] (channel average → mono).
#[must_use]
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(SAMPLE_RATE) / f64::from(src_rate);
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = samples.len() - 1;
    for i in 0..out_len {
        // Position in the source signal.
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        if idx >= last {
            out.push(*samples.last().unwrap_or(&0.0));
            continue;
        }
        let frac = (src_pos - idx as f64) as f32;
        #[allow(
            clippy::indexing_slicing,
            reason = "idx < last guaranteed by the branch above; idx+1 <= last"
        )]
        let a = samples[idx];
        #[allow(
            clippy::indexing_slicing,
            reason = "idx + 1 <= last (idx < last) is in bounds"
        )]
        let b = samples[idx + 1];
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
