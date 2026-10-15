//! The stop of a speech request in the two loops of a synthesis: the frame
//! loop and the passes of the codec decode. Scripted steps on the CPU, no
//! weights.
//!
//! The scripted tests read what a loop did after the stop was set and what
//! it returned. They do not refuse an adapter that wraps the step: a step
//! that refuses for itself gives the same counts and the same outcome, and
//! these two loops log nothing for a refused step. For speech, the guards of
//! "the loop reads the stop" are the source rule at the end of this file and
//! the speech case of `tests/gone_request.rs` in the server crate, which needs
//! the weights.

use super::{frame, release_reasons, END};
use crate::tts::{
    decode_in_chunks, run_frames, FrameStep, Frames, TtsError, DECODE_CHUNK_FRAMES as CHUNK,
};
use rmlx_mlx::{Array, Device};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "../../rmlx-models/tests/common/stop_scan.rs"]
mod stop_scan;

use stop_scan::{
    loops_under, not_stopping, sources_under, stop_reads_past, unlisted_and_lost, Loop,
    STOP_OUTCOME,
};

/// Hand `stop` to the frame loop. The test does not wrap `step`; see the head
/// of the file.
fn frames_under(
    stop: &AtomicBool,
    step: impl FnMut(usize) -> Result<(FrameStep, bool), TtsError>,
) -> Result<Frames, TtsError> {
    run_frames(stop, step)
}

/// Hand `stop` to the codec decode, on the terms of [`frames_under`].
fn decode_under(
    codes: &Array,
    stop: &AtomicBool,
    window: impl FnMut(&Array) -> Result<Vec<f32>, TtsError>,
) -> Result<Vec<f32>, TtsError> {
    decode_in_chunks(codes, Device::Cpu, stop, window)
}

/// `true` when `e` is the outcome of a stop: an error variant of its own, so
/// the route can tell a request that is gone from a synthesis that failed.
fn is_the_stop(e: &TtsError) -> bool {
    format!("{e:?}").starts_with(STOP_OUTCOME)
}

/// Counts the events at `ERROR` and hands every span and event on to the
/// subscriber that was current, so a recorder around the count still sees them.
struct CountAndPassOn {
    errors: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    outer: tracing::Dispatch,
}

impl tracing::Subscriber for CountAndPassOn {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        self.outer.new_span(attributes)
    }
    fn record(&self, span: &tracing::span::Id, values: &tracing::span::Record<'_>) {
        self.outer.record(span, values);
    }
    fn record_follows_from(&self, span: &tracing::span::Id, follows: &tracing::span::Id) {
        self.outer.record_follows_from(span, follows);
    }
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::ERROR {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        self.outer.event(event);
    }
    fn enter(&self, span: &tracing::span::Id) {
        self.outer.enter(span);
    }
    fn exit(&self, span: &tracing::span::Id) {
        self.outer.exit(span);
    }
}

/// The number of events at `ERROR` that `f` emits on this thread. Every event
/// also reaches the recorder that was installed before the call.
fn errors_passed_on(f: impl FnOnce()) -> usize {
    let errors = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = CountAndPassOn {
        errors: std::sync::Arc::clone(&errors),
        outer: tracing::dispatcher::get_default(Clone::clone),
    };
    tracing::subscriber::with_default(counter, f);
    errors.load(Ordering::Relaxed)
}

/// What one scripted frame loop did.
struct Looped {
    steps: usize,
    /// Steps that started with the stop set.
    steps_after_stop: usize,
    result: Result<Frames, TtsError>,
    /// Events the loop logged at `ERROR`.
    logged_errors: usize,
}

/// Run a talker that never picks the end token. The request goes away while
/// frame `gone_in` is made; `Some(usize::MAX)` is a request that is gone
/// before the loop.
fn frame_loop(gone_in: Option<usize>) -> Looped {
    let stop = AtomicBool::new(gone_in == Some(usize::MAX));
    let steps = Cell::new(0usize);
    let steps_after_stop = Cell::new(0usize);
    let mut result = None;
    let logged_errors = errors_passed_on(|| {
        result = Some(frames_under(&stop, |index| {
            steps.set(steps.get() + 1);
            if stop.load(Ordering::Relaxed) {
                steps_after_stop.set(steps_after_stop.get() + 1);
            }
            if gone_in == Some(index) {
                stop.store(true, Ordering::Relaxed);
            }
            Ok(frame(index as u32))
        }));
    });
    Looped {
        steps: steps.get(),
        steps_after_stop: steps_after_stop.get(),
        result: result.unwrap(),
        logged_errors,
    }
}

/// A stop that is never set changes nothing: the loop ends on the end token
/// with the frames before it.
#[test]
fn a_frame_loop_that_is_not_stopped_makes_every_frame() {
    let stop = AtomicBool::new(false);
    let frames = frames_under(&stop, |step| Ok(if step == 5 { END } else { frame(7) })).unwrap();
    assert_eq!(frames.codes.len(), 5);
    assert!(!frames.truncated);
}

/// The loop reads the stop before it makes a frame, and the step that sets
/// the stop here is the frame before. So it makes no frame after the stop.
#[test]
fn a_stopped_frame_loop_makes_no_further_frame() {
    let run = frame_loop(Some(3));
    assert_eq!(run.steps_after_stop, 0, "of {} frames", run.steps);
}

/// A request that is gone before its first frame makes none.
#[test]
fn a_frame_loop_stopped_before_its_first_frame_makes_none() {
    let run = frame_loop(Some(usize::MAX));
    assert_eq!(run.steps, 0);
}

/// A stopped loop gives no frames, which would start the codec decode and be
/// reported as a synthesis that ended. It returns the stop outcome, and logs
/// no error: a request that is gone is a normal event.
#[test]
fn a_stopped_frame_loop_returns_the_stop_outcome_and_logs_no_error() {
    let run = frame_loop(Some(3));
    assert!(
        run.result.as_ref().is_err_and(is_the_stop),
        "{:?}",
        run.result.map(|f| f.codes.len())
    );
    assert_eq!(run.logged_errors, 0);
}

/// A stopped loop releases the MLX buffer cache once at its end, as a failed
/// one does: the next synthesis starts from the memory of a loop that ended.
/// The release is a duty after the loop on every outcome.
#[test]
fn a_stopped_frame_loop_releases_the_cache_once() {
    let reasons = release_reasons(|| {
        let run = frame_loop(Some(3));
        assert!(run.steps < 16, "the stop did not end the loop");
    });
    assert_eq!(reasons, ["end of the loop"]);
}

/// `[1, 16, frames]` codes of zeros.
fn codes(frames: usize) -> Array {
    Array::from_i32_slice(&vec![0; 16 * frames], &[1, 16, frames as i32]).unwrap()
}

/// What one scripted codec decode did.
struct Decoded {
    passes: usize,
    /// Passes that started with the stop set.
    passes_after_stop: usize,
    result: Result<Vec<f32>, TtsError>,
}

/// Decode four chunks of frames, one sample for each frame. The request goes
/// away while pass `gone_in` runs.
fn codec_decode(gone_in: Option<usize>) -> Decoded {
    let stop = AtomicBool::new(false);
    let passes = Cell::new(0usize);
    let passes_after_stop = Cell::new(0usize);
    let result = decode_under(&codes(4 * CHUNK), &stop, |window| {
        if stop.load(Ordering::Relaxed) {
            passes_after_stop.set(passes_after_stop.get() + 1);
        }
        if gone_in == Some(passes.get()) {
            stop.store(true, Ordering::Relaxed);
        }
        passes.set(passes.get() + 1);
        Ok(vec![0.0; window.shape()[2] as usize])
    });
    Decoded {
        passes: passes.get(),
        passes_after_stop: passes_after_stop.get(),
        result,
    }
}

/// A stop that is never set changes nothing: every pass runs.
#[test]
fn a_codec_decode_that_is_not_stopped_runs_every_pass() {
    let run = codec_decode(None);
    assert_eq!(run.passes, 4);
    assert!(run.result.is_ok());
}

/// A stopped codec decode starts no further pass and returns the stop
/// outcome, not samples.
#[test]
fn a_stopped_codec_decode_starts_no_further_pass() {
    let run = codec_decode(Some(1));
    assert_eq!(run.passes_after_stop, 0, "{} of 4 passes ran", run.passes);
    assert!(
        run.result.as_ref().is_err_and(is_the_stop),
        "{:?}",
        run.result.map(|s| s.len())
    );
}

// ── Every loop that does work for a request ─────────────────────────────────
//
// A loop is named by its file, its fn and its header line.

/// The loops of a synthesis that a request can keep running. Each one reads
/// the stop by the rule.
const SPEECH_LOOPS: [&str; 2] = [
    "tts.rs::decode_passes: while start < total {",
    "tts.rs::frame_loop: for index in 0..MAX_FRAMES {",
];

/// The walk of a transcription. It reads the stop of its request by a rule of
/// its own, which its own tests hold: it reads the stop before each window and
/// returns the window and the sample it stopped at, as a value and not as an
/// error. The stop is the flag of the request, or a window sink whose reader
/// is gone. The one stop helper is not asked of it.
const OWN_RULE_LOOPS: [&str; 1] = [
    "transcribe/seek.rs::walk_windows: while total_samples.saturating_sub(seek) >= HOP_LENGTH {",
];

/// The loops that do work and do not read the stop, each with its bound.
const BOUNDED_LOOPS: [(&str, &str); 6] = [
    (
        "transcribe/fallback.rs::decode_with_fallback: while let Some(&temperature) = temperatures.peek() {",
        "at most 7 decodes of one window: a ladder of at most 6 temperatures and one prompt drop; the walk reads the stop before each window",
    ),
    (
        "tts.rs::decode_window: for (conv_t, conv_next) in &self.upsample {",
        "the two upsample stages of one codec window; the passes read the stop",
    ),
    (
        "tts.rs::decode_window: for group in &self.groups {",
        "the four decoder groups of one codec window; the passes read the stop",
    ),
    (
        "tts.rs::frame_step: for group in 0..CODE_GROUPS - 1 {",
        "the fifteen code groups of one frame; the frame loop reads the stop",
    ),
    (
        "vad.rs::forward: for layer in &self.enc {",
        "the voice activity model, which no route of the server runs",
    ),
    (
        "whisper/decode_loop.rs::decode_loop: let stop = loop {",
        "the tokens of one window, at most `max_tokens` and the decoder context `n_text_ctx`; the walk reads the stop before each window",
    ),
];

fn crate_src() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn crate_loops() -> Vec<(String, Loop)> {
    loops_under(&crate_src())
}

/// Every loop of the crate that runs a forward, an evaluation or a callback
/// in each pass is on one of the lists, unless its shape bounds it. A new loop
/// is on no list, so this test fails until the loop is given a stop or a
/// bound.
#[test]
fn every_loop_that_does_work_reads_the_stop_or_has_a_stated_bound() {
    let mut listed = [&SPEECH_LOOPS[..], &OWN_RULE_LOOPS[..]].concat();
    listed.extend(BOUNDED_LOOPS.iter().map(|(key, _)| *key));
    let (unlisted, lost) = unlisted_and_lost(&crate_loops(), &listed);
    assert_eq!(unlisted, Vec::<String>::new(), "loops on no list");
    assert_eq!(
        lost,
        Vec::<String>::new(),
        "listed loops that the scan lost"
    );
}

/// The walk of a transcription still reads the flag of its request before
/// each window.
#[test]
fn the_walk_of_a_transcription_reads_the_flag_by_its_own_rule() {
    let walks: Vec<Loop> = crate_loops()
        .into_iter()
        .filter(|(file, l)| OWN_RULE_LOOPS.contains(&l.key(file).as_str()))
        .map(|(_, l)| l)
        .collect();
    assert_eq!(walks.len(), 1, "the walk");
    assert!(walks.iter().all(|l| l.code.contains("cancel.load(")));
}

/// The frame loop and the passes of the codec decode read the stop by the
/// rule: the helper is the first statement of a pass, with `?`. The helper is
/// called nowhere else in the crate.
#[test]
fn the_speech_loops_read_the_stop() {
    let loops = crate_loops();
    assert_eq!(not_stopping(&loops, &SPEECH_LOOPS), Vec::<String>::new());
    let past = stop_reads_past(&sources_under(&crate_src()), &loops, &SPEECH_LOOPS);
    assert_eq!(past, 0);
}
