// LOC-exempt: the long-form transcription engine (window seek loop + timestamp
// segmentation + previous-text conditioning + subtitle formatting) is one
// cohesive sequential pipeline; splitting the seek loop from the formatters
// would scatter the timestamp arithmetic without cohesion gain.

//! Long-form Whisper transcription engine.
//!
//! This is the single transcription core shared by the HTTP
//! `POST /v1/audio/transcriptions` route and the `rmlx transcribe` CLI. It
//! runs a sliding-window seek loop modelled on `openai-whisper` / `mlx_whisper` `transcribe()`:
//!
//! 1. Decode the input container to 16 kHz mono f32 (caller's responsibility;
//!    see [`crate::wav::WavDecoder`] + [`resample_to_16k`]).
//! 2. Walk the audio in 30 s windows. For each window, run the decoder in
//!    **timestamp mode** with the full [`crate::whisper::DecodeFilters`] chain.
//! 3. Parse the emitted timestamp tokens into segments with real cumulative
//!    times. Consume the whole window when the tokens end in a single
//!    timestamp; otherwise seek to the end of the last closed segment. Feed the
//!    previous window's text back as a prompt (`<|startofprev|>`). No
//!    temperature fallback: a window that loops gets one greedy retry without
//!    the prompt.
//! 4. Emit multi-segment output (`vtt` / `srt` / `json` / `txt`).
//!
//! Determinism: temperature is fixed at 0 (greedy argmax), so the same audio
//! produces byte-identical output across runs.

use std::sync::Arc;

use rmlx_mlx::Device;
use tracing::{debug, info, warn};

use crate::mel::{MelExtractor, HOP_LENGTH, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::{WhisperTask, WhisperTokenizer, TOK_EOT, TOK_SOT_PREV, TOK_TIMESTAMP_BEGIN};
use crate::whisper::{DecodeFilters, WhisperError, WhisperModel};

/// Seconds of audio represented by one timestamp-token step (Whisper uses 0.02 s).
const TIME_PRECISION: f32 = 0.02;

/// Samples of audio represented by one timestamp-token step.
const SAMPLES_PER_TIMESTAMP: usize = SAMPLE_RATE as usize / 50;

/// How far past the real end of a window a segment may open before it counts as
/// zero-pad filler (0.5 s).
const PAD_TAIL_MARGIN_SAMPLES: usize = SAMPLE_RATE as usize / 2;

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

        let filters = DecodeFilters::new(self.suppress.clone(), self.blank_ids.clone(), true);

        walk_windows(total_samples, |seek, window_samples| {
            #[allow(
                clippy::indexing_slicing,
                reason = "walk_windows yields seek + window_samples <= total_samples"
            )]
            let window = &samples[seek..seek + window_samples];
            let window_start_time = seek as f32 / SAMPLE_RATE as f32;
            // Real audio length of this (un-padded) window, in seconds.
            let window_dur = window_samples as f32 / SAMPLE_RATE as f32;

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
            let prompt: &[u32] = if opts.condition_on_previous_text {
                &prompt_tokens
            } else {
                &[]
            };
            let decode = |prompt: &[u32]| {
                let mut full_prefix: Vec<u32> = Vec::new();
                if !prompt.is_empty() {
                    full_prefix.push(TOK_SOT_PREV);
                    full_prefix.extend(prompt.iter().copied());
                }
                full_prefix.extend(sot.iter().copied());

                // Per-window generation budget, derived from `n_text_ctx` at runtime
                // (no fixed literal); bounded so the decoder position stays `< n_text_ctx`.
                // greedy_decode additionally refuses any positional row `>= n_text_ctx`
                // as a belt-and-suspenders guard.
                let max_tokens = window_token_budget(self.model.cfg.n_text_ctx, full_prefix.len());
                self.model.greedy_decode(
                    &encoder_out,
                    &full_prefix,
                    max_tokens,
                    opts.temperature,
                    &filters,
                    device,
                )
            };
            let text_of = |tokens: &[u32]| {
                let text: Vec<u32> = tokens
                    .iter()
                    .copied()
                    .filter(|&t| t < TOK_TIMESTAMP_BEGIN)
                    .collect();
                self.tokenizer.decode(&text).unwrap_or_default()
            };

            let decoded = match decode_window(prompt, decode, text_of) {
                Ok(d) => d,
                Err(WhisperError::Silence) => {
                    // Whole window is silence — skip ahead a full window.
                    return Ok(N_SAMPLES);
                }
                Err(e) => return Err(e),
            };
            decoded.log(seek, window_start_time);
            let tokens = decoded.tokens;

            let plan = plan_window(&tokens, window_samples);
            if plan.rule == SeekRule::ZeroFallback {
                warn!(
                    seek,
                    window_samples,
                    "last closed segment ends at the window start; consuming the whole window"
                );
            }
            let window_segments: Vec<Segment> = plan
                .segments
                .iter()
                .filter_map(|raw| {
                    let text = self.tokenizer.decode(&raw.tokens).unwrap_or_default();
                    let trimmed = text.trim();
                    (!trimmed.is_empty()).then(|| Segment {
                        start: window_start_time + raw.start,
                        end: window_start_time + raw.end,
                        text: trimmed.to_owned(),
                    })
                })
                .collect();

            debug!(
                seek,
                window_start_time,
                window_dur,
                n_tokens = tokens.len(),
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
                decoded.carry_text,
                opts.condition_on_previous_text,
                &window_text,
                |text| self.tokenizer.encode(text).unwrap_or_default(),
                previous_text_cap(self.model.cfg.n_text_ctx),
            );

            segments.extend(window_segments);
            Ok(plan.advance_samples)
        })?;

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
}

/// The tokens kept for one window, and whether their text may prompt the next
/// window.
#[derive(Debug, Clone, PartialEq)]
struct WindowDecode {
    tokens: Vec<u32>,
    carry_text: bool,
    /// Compression ratio of the first decode's text.
    ratio: f32,
    /// Compression ratio of the kept tokens' text.
    kept_ratio: f32,
    /// A second decode without the prompt ran.
    retried: bool,
}

impl WindowDecode {
    /// Record a repetition loop of the window that starts at sample `seek`.
    fn log(&self, seek: usize, window_start_time: f32) {
        if !self.carry_text {
            warn!(
                seek,
                window_start_time,
                ratio = self.ratio,
                kept_ratio = self.kept_ratio,
                retried = self.retried,
                "repetition loop kept in the transcript; it does not prompt the next window"
            );
        } else if self.retried {
            debug!(
                seek,
                window_start_time,
                ratio = self.ratio,
                kept_ratio = self.kept_ratio,
                "repetition loop under the prompt; decoded again without it"
            );
        }
    }
}

/// Decode one window. `decode(prompt)` runs the decoder with that previous-text
/// prompt (empty for none); `text_of` gives the text of a token stream.
///
/// A decode whose text is a repetition loop under a prompt is decoded again
/// with no prompt, as whisper.cpp does for its fallbacks at temperature 0.5 and
/// above: greedy decode has no temperature fallback here, and a looping prompt
/// makes the next window loop too. Of two looping decodes, the one with the
/// lower ratio is kept; it does not prompt the next window.
fn decode_window<E>(
    prompt: &[u32],
    mut decode: impl FnMut(&[u32]) -> Result<Vec<u32>, E>,
    text_of: impl Fn(&[u32]) -> String,
) -> Result<WindowDecode, E> {
    let ratio_of = |tokens: &[u32]| compression_ratio(&text_of(tokens));
    let tokens = decode(prompt)?;
    let ratio = ratio_of(&tokens);
    if ratio <= COMPRESSION_RATIO_LOOP {
        return Ok(WindowDecode {
            tokens,
            carry_text: true,
            ratio,
            kept_ratio: ratio,
            retried: false,
        });
    }
    let retried = !prompt.is_empty();
    let (tokens, kept_ratio) = if retried {
        let retry = decode(&[])?;
        let retry_ratio = ratio_of(&retry);
        if retry_ratio <= ratio {
            (retry, retry_ratio)
        } else {
            (tokens, ratio)
        }
    } else {
        (tokens, ratio)
    };
    Ok(WindowDecode {
        tokens,
        carry_text: kept_ratio <= COMPRESSION_RATIO_LOOP,
        ratio,
        kept_ratio,
        retried,
    })
}

/// The previous-text prompt for the next window. A window whose text may not
/// be carried clears it. Otherwise, with `condition` set and some text in the
/// window, it is `encode(text)` cut to its last `cap` tokens, as openai-whisper
/// cuts its prompt; else the prompt is kept as it is.
fn next_prompt(
    prompt: Vec<u32>,
    carry_text: bool,
    condition: bool,
    window_text: &str,
    encode: impl FnOnce(&str) -> Vec<u32>,
    cap: usize,
) -> Vec<u32> {
    let text = window_text.trim();
    if !carry_text {
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

/// One segment cut from a window's token stream, before text decode. Times are
/// seconds from the window start.
#[derive(Debug, Clone, PartialEq)]
struct RawSegment {
    start: f32,
    end: f32,
    tokens: Vec<u32>,
}

/// Which rule set the seek advance of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeekRule {
    /// A timestamp pair in the stream and one timestamp at the end: the whole
    /// window.
    SingleEnding,
    /// A timestamp pair and no single ending: the end of the last closed
    /// segment.
    LastClosed { end_samples: usize },
    /// No timestamp pair: one segment and the whole window.
    NoPair,
    /// The last closed segment ends at the window start, so the seek would not
    /// move: the whole window.
    ZeroFallback,
}

/// What one decoded window yields: its segments and how far the seek moves.
#[derive(Debug, Clone, PartialEq)]
struct WindowPlan {
    segments: Vec<RawSegment>,
    advance_samples: usize,
    rule: SeekRule,
}

/// Split one window's sampled tokens into timestamped segments and decide the
/// seek advance. `tokens` are the tokens after the decoder prefix, with the end
/// token stripped. `window_samples` is the real (un-padded) length of the window.
///
/// A timestamp pair (two timestamps in a row) closes one segment and opens the
/// next. With a pair in the stream:
/// - a stream that ends in one timestamp has no speech after it, so the whole
///   window is consumed;
/// - otherwise the unfinished segment is dropped and the seek goes to the end
///   of the last closed segment.
///
/// With no pair the stream is one segment from the window start, and the whole
/// window is consumed.
fn plan_window(tokens: &[u32], window_samples: usize) -> WindowPlan {
    let window_dur = window_samples as f32 / SAMPLE_RATE as f32;
    let is_timestamp = |tok: u32| tok >= TOK_TIMESTAMP_BEGIN;
    let ticks = |tok: u32| tok.saturating_sub(TOK_TIMESTAMP_BEGIN);
    let to_secs = |tok: u32| ticks(tok) as f32 * TIME_PRECISION;
    // A segment that opens more than 0.5 s past the real audio is in the
    // zero-pad tail, where Whisper prints filler ("you", "thank you").
    let opens_in_speech = |tok: u32| {
        ticks(tok) as usize * SAMPLES_PER_TIMESTAMP <= window_samples + PAD_TAIL_MARGIN_SAMPLES
    };
    let text_of = |slice: &[u32]| -> Vec<u32> {
        slice
            .iter()
            .copied()
            .filter(|&t| !is_timestamp(t))
            .collect()
    };

    let ends_in_one_timestamp = matches!(
        tokens,
        [.., before, last] if !is_timestamp(*before) && is_timestamp(*last)
    );
    // Index of the second token of every timestamp pair: where one slice ends
    // and the next begins.
    let mut cuts: Vec<usize> = tokens
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| pair.iter().all(|&t| is_timestamp(t)))
        .map(|(i, _)| i + 1)
        .collect();

    let mut segments: Vec<RawSegment> = Vec::new();
    let rule = if cuts.is_empty() {
        let end = match tokens.iter().rev().find(|&&t| is_timestamp(t)) {
            Some(&last) if last != TOK_TIMESTAMP_BEGIN => to_secs(last),
            _ => window_dur,
        };
        segments.push(RawSegment {
            start: 0.0,
            end,
            tokens: text_of(tokens),
        });
        SeekRule::NoPair
    } else {
        let rule = if ends_in_one_timestamp {
            cuts.push(tokens.len());
            SeekRule::SingleEnding
        } else {
            let end_samples = cuts
                .last()
                .and_then(|&cut| tokens.get(cut - 1))
                .map_or(0, |&tok| ticks(tok) as usize * SAMPLES_PER_TIMESTAMP);
            if end_samples == 0 {
                SeekRule::ZeroFallback
            } else {
                SeekRule::LastClosed { end_samples }
            }
        };
        let mut from = 0;
        for &cut in &cuts {
            let slice = tokens.get(from..cut).unwrap_or_default();
            if let (Some(&first), Some(&last)) = (slice.first(), slice.last()) {
                if opens_in_speech(first) {
                    segments.push(RawSegment {
                        start: to_secs(first),
                        end: to_secs(last),
                        tokens: text_of(slice),
                    });
                }
            }
            from = cut;
        }
        rule
    };
    segments.retain(|s| !s.tokens.is_empty());

    let advance_samples = match rule {
        SeekRule::LastClosed { end_samples } => end_samples.min(window_samples),
        SeekRule::SingleEnding | SeekRule::NoPair | SeekRule::ZeroFallback => window_samples,
    };
    WindowPlan {
        segments,
        advance_samples,
        rule,
    }
}

/// Walk the audio window by window. `on_window(seek, window_samples)` decodes
/// the window that starts at sample `seek` and returns how many samples to
/// advance; an advance of 0 consumes the whole window, so the walk always ends.
/// The walk stops when fewer than one mel hop of audio is left, as the
/// reference does at its 10 ms frame grid.
fn walk_windows<E>(
    total_samples: usize,
    mut on_window: impl FnMut(usize, usize) -> Result<usize, E>,
) -> Result<(), E> {
    let mut seek = 0;
    while total_samples.saturating_sub(seek) >= HOP_LENGTH {
        let window_samples = (total_samples - seek).min(N_SAMPLES);
        seek += match on_window(seek, window_samples)? {
            0 => window_samples,
            advance => advance,
        };
    }
    Ok(())
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
