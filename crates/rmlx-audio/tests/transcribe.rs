// LOC-exempt: the model tests of the transcription are one test binary. They share the
// snapshot loader, the word-error helpers and the `say` clip helpers, and the script of the
// long clip is 110 lines of data.

//! Real-model Whisper transcription integration tests.
//!
//! These tests resolve the Whisper snapshot + tokenizer from `RMLX_O_MODELS_ROOT`
//! auto-discovery (the `make model-check-full` convention) and **skip gracefully**
//! when the model, tokenizer, or fixtures are absent — there is no bespoke env
//! var. Two layers:
//!
//! 1. `say_clip_deterministic` — portable, asset-free: synthesise a known English
//!    sentence with macOS `say` + `ffmpeg`, transcribe it, assert it matches the
//!    sentence (WER 0, case/punct-insensitive) and is byte-identical across two
//!    runs at temp=0. Skips when `say`/`ffmpeg` are unavailable.
//!    `two_sentence_clip_is_two_segments` does the same for two sentences and
//!    asserts two segments with no repeat and no trailing filler.
//!    `long_synthetic_clip_is_transcribed_once_in_order` speaks 28 sentences
//!    with gaps (about 112 s, several windows) and asserts each sentence once,
//!    in order, with segment times inside the clip.
//!    `digital_silence_gives_no_segment`,
//!    `silence_around_speech_adds_no_segment` and
//!    `a_window_with_no_speech_gives_no_segment_and_speech_keeps_its_text`
//!    hold the no-speech gate: a window of zeros, noise or a tone gives no
//!    text, alone and behind the prompt of a spoken window, and a spoken
//!    window keeps its text.
//!    `a_spoken_repetition_above_the_no_speech_bound_is_not_skipped` and
//!    `thirty_seconds_of_one_repeated_word_are_skipped` hold the second
//!    condition of the gate: a window above the probability bound is skipped
//!    only when the model writes no more than a stock phrase for it.
//!    `the_gate_picks_at_temperature_0_for_a_request_above_0` holds that the
//!    gate draws nothing at a request temperature above 0.
//!    `the_language_of_auto_is_that_of_the_first_window_that_is_not_skipped`
//!    holds the language of an `auto` request behind a silent window.
//!    `a_window_that_needs_a_fallback_is_decoded_again_and_repeats` holds the
//!    temperature fallback: a spoken repetition is decoded again above
//!    temperature 0, and a second run gives the same tokens.
//!
//! 2. `long_form_regression` — scans the gitignored fixtures dir
//!    (`crates/rmlx-audio/tests/fixtures/`) for any `*.{m4a,wav,mp3,…}` paired
//!    with a sibling `*.transcript.vtt`; transcribes the FULL file and asserts a
//!    normalized WER ≤ threshold. Generic: any user drops their own audio + VTT.
//!
//! Single-MLX discipline: these load the model in-process. Do not run them
//! concurrently with a live `rmlx serve`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::indexing_slicing,
    clippy::bool_to_int_with_if,
    clippy::map_unwrap_or,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::missing_panics_doc,
    clippy::items_after_statements,
    clippy::needless_range_loop
)]

mod say_voice;

use say_voice::{voice_is_installed, ENGLISH_VOICE};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
use rmlx_audio::wav::WavDecoder;
use rmlx_audio::whisper::{WhisperError, WhisperModel};
use rmlx_mlx::Device;

// ── Snapshot resolution ─────────────────────────────────────────────────────

const WHISPER_SLUG: &str = "mlx-community__whisper-large-v3-mlx";
const TOKENIZER_SLUG: &str = "openai__whisper-large-v3-tokenizer";

fn models_root() -> Option<PathBuf> {
    let root = std::env::var("RMLX_O_MODELS_ROOT").ok()?;
    let pb = PathBuf::from(root);
    pb.exists().then_some(pb)
}

fn whisper_paths() -> Option<(PathBuf, PathBuf)> {
    let root = models_root()?;
    let model = root.join(WHISPER_SLUG);
    let tok = root.join(TOKENIZER_SLUG);
    if model.join("config.json").exists() && tok.join("tokenizer.json").exists() {
        Some((model, tok))
    } else {
        None
    }
}

fn load_transcriber() -> Option<Transcriber> {
    let (model_path, tok_path) = whisper_paths()?;
    let model = WhisperModel::load(&model_path).expect("load whisper model");
    let tokenizer = WhisperTokenizer::from_path(&tok_path).expect("load tokenizer");
    Some(Transcriber::new(Arc::new(model), Arc::new(tokenizer)).expect("transcriber"))
}

// ── WER + normalization ─────────────────────────────────────────────────────

/// Normalize a transcript for WER: lowercase, strip punctuation, collapse
/// whitespace. Used on BOTH hypothesis and reference.
fn normalize(text: &str) -> Vec<String> {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Word-level edit distance (Levenshtein) over token vectors.
fn edit_distance(a: &[String], b: &[String]) -> usize {
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn wer(reference: &[String], hypothesis: &[String]) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance(reference, hypothesis) as f64 / reference.len() as f64
}

/// Parse a WEBVTT file, stripping cue numbers, timestamps, and `Speaker Name: `
/// prefixes; return the concatenated reference text.
fn parse_vtt_reference(path: &Path) -> String {
    let raw = std::fs::read_to_string(path).expect("read vtt");
    let mut out: Vec<String> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty()
            || line == "WEBVTT"
            || line.contains("-->")
            || line.chars().all(|c| c.is_ascii_digit())
        {
            continue;
        }
        // Strip leading "Speaker Name: " — the speaker label ends at the first
        // ": " that precedes alpha content. Only strip when a colon appears
        // reasonably early (a name), not mid-sentence.
        let text = match line.find(": ") {
            Some(idx) if idx < 40 => &line[idx + 2..],
            _ => line,
        };
        out.push(text.to_owned());
    }
    out.join(" ")
}

// ── say-clip helpers ────────────────────────────────────────────────────────

fn have_cmd(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Synthesise a known sentence to a 16 kHz mono WAV via `say` + `ffmpeg`,
/// with [`ENGLISH_VOICE`]: the same samples on each run.
fn synth_say_clip(sentence: &str) -> Option<PathBuf> {
    synth_voice_clip(ENGLISH_VOICE, sentence)
}

/// [`synth_say_clip`] with the `say` voice `voice`. `None` when `say`,
/// `ffmpeg` or the voice is not installed; nothing is spoken then.
fn synth_voice_clip(voice: &str, sentence: &str) -> Option<PathBuf> {
    if !have_cmd("ffmpeg") || !voice_is_installed(voice) {
        return None;
    }
    let dir = std::env::temp_dir();
    let aiff = dir.join("rmlx_say_clip.aiff");
    let wav = dir.join("rmlx_say_clip.wav");
    let say_ok = std::process::Command::new("say")
        .args(["-v", voice])
        .args(["-o", aiff.to_str()?, sentence])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !say_ok {
        return None;
    }
    let ff_ok = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            aiff.to_str()?,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            wav.to_str()?,
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    ff_ok.then_some(wav)
}

fn transcribe_file(
    t: &Transcriber,
    path: &Path,
    language: &str,
) -> rmlx_audio::transcribe::Transcription {
    let bytes = std::fs::read(path).expect("read audio");
    let samples =
        WavDecoder::decode_resampled(&bytes, 16_000, rmlx_audio::wav::MAX_AUDIO_SECS, &|| false)
            .expect("decode audio")
            .samples;
    transcribe_samples(t, &samples, language)
}

/// The first largest logit of `row`: the greedy pick.
fn largest_logit(row: &[f32]) -> u32 {
    let mut best = 0;
    for (id, logit) in row.iter().enumerate() {
        if logit > &row[best] {
            best = id;
        }
    }
    best as u32
}

/// Transcribe with the greedy pick at each temperature. `cancelled` is the
/// state of the cancel flag for the whole call.
fn transcribe_greedy(
    t: &Transcriber,
    samples: &[f32],
    opts: &TranscribeOptions,
    cancelled: bool,
) -> Result<rmlx_audio::transcribe::Transcription, WhisperError> {
    t.transcribe(
        samples,
        opts,
        &mut |row, _| Ok(largest_logit(row)),
        Device::Gpu,
        &AtomicBool::new(cancelled),
    )
}

fn english(temperature: f32) -> TranscribeOptions {
    TranscribeOptions {
        language: "en".to_owned(),
        task: WhisperTask::Transcribe,
        temperature,
        condition_on_previous_text: true,
    }
}

fn transcribe_samples(
    t: &Transcriber,
    samples: &[f32],
    language: &str,
) -> rmlx_audio::transcribe::Transcription {
    let opts = TranscribeOptions {
        language: language.to_owned(),
        ..english(0.0)
    };
    transcribe_greedy(t, samples, &opts, false).expect("transcribe")
}

/// `secs` seconds of digital silence at 16 kHz.
fn silence(secs: usize) -> Vec<f32> {
    vec![0.0; secs * 16_000]
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// A voice that `say` does not have is not available: no clip is spoken, so a
/// test stands down and does not run on the default voice of the host. Needs
/// no model.
#[test]
fn a_voice_that_is_not_installed_gives_no_clip() {
    const NO_VOICE: &str = "NoSuchVoiceXyz";
    if !have_cmd("say") {
        eprintln!("SKIP a_voice_that_is_not_installed_gives_no_clip: say unavailable");
        return;
    }
    assert!(!voice_is_installed(NO_VOICE));
    assert!(!voice_is_installed(""));
    assert!(synth_voice_clip(NO_VOICE, "hello").is_none());
    // The name of a voice is the whole name: a prefix of one is not a voice.
    if voice_is_installed(ENGLISH_VOICE) {
        assert!(!voice_is_installed(&ENGLISH_VOICE[..4]));
    }
}

/// Portable smoke: a known `say` sentence transcribes correctly and is
/// deterministic across two runs at temp=0.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test say_clip -- --ignored --test-threads=1"]
fn say_clip_deterministic() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP say_clip_deterministic: whisper snapshot not present");
        return;
    };
    let sentence = "The quick brown fox jumps over the lazy dog.";
    let Some(wav) = synth_say_clip(sentence) else {
        eprintln!("SKIP say_clip_deterministic: say, its voice or ffmpeg unavailable");
        return;
    };

    let r1 = transcribe_file(&t, &wav, "en");
    let r2 = transcribe_file(&t, &wav, "en");

    assert_eq!(
        r1.language, "english",
        "a named language is reported as its English name"
    );
    // Determinism: identical text across runs at temp=0.
    assert_eq!(
        r1.text, r2.text,
        "transcription not deterministic at temp=0:\n  run1: {}\n  run2: {}",
        r1.text, r2.text
    );

    let reference = normalize(sentence);
    let hyp = normalize(&r1.text);
    let w = wer(&reference, &hyp);
    println!("say-clip text: {:?}  WER={w:.3}", r1.text);
    // The clip is one short sentence: any extra word, such as filler after the
    // speech, fails.
    assert_eq!(w, 0.0, "say-clip WER {w:.3} above 0; got {:?}", r1.text);
}

/// Two spoken sentences in a 5.4 s clip are two segments and one pass over the
/// clip: no repeated sentence and no filler after the speech.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test two_sentence_clip -- --ignored --test-threads=1"]
fn two_sentence_clip_is_two_segments() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP two_sentence_clip_is_two_segments: whisper snapshot not present");
        return;
    };
    let Some(wav) = synth_say_clip(
        "The quick brown fox jumps over the lazy dog. Paper was invented in China about two thousand years ago.",
    ) else {
        eprintln!("SKIP two_sentence_clip_is_two_segments: say, its voice or ffmpeg unavailable");
        return;
    };

    let r = transcribe_file(&t, &wav, "en");
    println!("two-sentence text: {:?}", r.text);
    assert_eq!(r.language, "english");
    assert_eq!(r.segments.len(), 2, "segments: {:?}", r.segments);
    let words = normalize(&r.text);
    assert_eq!(
        words.iter().filter(|w| w.as_str() == "paper").count(),
        1,
        "second sentence repeated: {:?}",
        r.text
    );
    assert!(
        !words.iter().any(|w| w == "you"),
        "filler after the speech: {:?}",
        r.text
    );
}

/// One full window of digital silence gives no segment and no text. The
/// language is named, so no detection runs on the silence.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test digital_silence -- --ignored --test-threads=1"]
fn digital_silence_gives_no_segment() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP digital_silence_gives_no_segment: whisper snapshot not present");
        return;
    };
    let r = transcribe_samples(&t, &silence(30), "en");
    println!("silence text: {:?}", r.text);
    assert!(r.segments.is_empty(), "segments: {:?}", r.segments);
    assert_eq!(r.text, "");
}

/// A silent window before the speech, a silent window after it and a short
/// silent tail give no segment: every segment is in the window that holds the
/// speech, and the text is the spoken text.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test silence_around_speech -- --ignored --test-threads=1"]
fn silence_around_speech_adds_no_segment() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP silence_around_speech_adds_no_segment: whisper snapshot not present");
        return;
    };
    let script = FOX;
    let Some(speech) = say_samples(script) else {
        eprintln!(
            "SKIP silence_around_speech_adds_no_segment: say, its voice or ffmpeg unavailable"
        );
        return;
    };
    let speech_secs = speech.len() as f32 / 16_000.0;
    assert!(speech_secs < 25.0, "the speech must fit one window");
    let mut samples = silence(30);
    samples.extend(speech);
    samples.extend(silence(65));

    let r = transcribe_samples(&t, &samples, "en");
    println!("silence around speech: {:?}", r.text);
    for s in &r.segments {
        assert!(
            s.start >= 30.0 && s.end <= 30.0 + speech_secs + 0.5,
            "segment outside the speech (30.00 to {:.2}): {s:?}",
            30.0 + speech_secs
        );
    }
    let w = wer(&normalize(script), &normalize(&r.text));
    assert_eq!(w, 0.0, "WER {w:.3} above 0; got {:?}", r.text);
}

// ── Synthetic windows with no speech ────────────────────────────────────────

const WINDOW_SAMPLES: usize = 30 * 16_000;
const FOX: &str = "The quick brown fox jumps over the lazy dog.";
const LANTERN: &str = "She carried a brass lantern down the narrow stone stairs.";

/// The RMS of `samples` in dBFS.
fn rms_dbfs(samples: &[f32]) -> f64 {
    let squares: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    10.0 * (squares / samples.len().max(1) as f64).log10()
}

/// `samples` scaled to an RMS of `dbfs`.
fn at_level(mut samples: Vec<f32>, dbfs: f64) -> Vec<f32> {
    let gain = 10f64.powf((dbfs - rms_dbfs(&samples)) / 20.0) as f32;
    for sample in &mut samples {
        *sample *= gain;
    }
    samples
}

/// White noise at an RMS of `dbfs`, from a linear congruential generator with
/// a fixed seed: the same samples on every run.
fn noise(len: usize, dbfs: f64) -> Vec<f32> {
    let mut state: u64 = 0x5EED;
    let uniform = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect();
    at_level(uniform, dbfs)
}

/// A sine tone of `hz` at an RMS of `dbfs`.
fn tone(len: usize, hz: f64, dbfs: f64) -> Vec<f32> {
    let sine = (0..len)
        .map(|i| (std::f64::consts::TAU * hz * i as f64 / 16_000.0).sin() as f32)
        .collect();
    at_level(sine, dbfs)
}

/// `sentence` spoken by `say`, as 16 kHz mono samples.
fn say_samples(sentence: &str) -> Option<Vec<f32>> {
    let wav = synth_say_clip(sentence)?;
    Some(WavDecoder::decode(&std::fs::read(wav).ok()?).ok()?.samples)
}

/// `samples`, then zeros to the end of one window.
fn to_window(samples: &[f32]) -> Vec<f32> {
    let mut window = samples.to_vec();
    window.resize(WINDOW_SAMPLES, 0.0);
    window
}

/// Each window with no speech gives no segment: digital silence, white noise
/// from -80 to -30 dBFS, a hum, a tone, and two tails of 0.1 s. It gives none
/// alone, where it is decoded with no prompt, and none after a spoken window,
/// where the text of that window is the previous-text prompt. Each spoken
/// window keeps its text: quiet speech, speech in noise, speech next to
/// silence, one word, and speech behind a prompt.
///
/// The speech is that of one `say` voice ([`ENGLISH_VOICE`]), and the text of
/// each spoken window is that of large-v3 for that voice: another voice or
/// model can fail a row of this test with no defect in the gate.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test a_window_with_no_speech -- --ignored --test-threads=1"]
fn a_window_with_no_speech_gives_no_segment_and_speech_keeps_its_text() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP a_window_with_no_speech_gives_no_segment_and_speech_keeps_its_text: whisper snapshot not present");
        return;
    };
    let (Some(fox), Some(lantern), Some(word)) = (
        say_samples(FOX),
        say_samples(LANTERN),
        say_samples("Hello."),
    ) else {
        eprintln!("SKIP a_window_with_no_speech_gives_no_segment_and_speech_keeps_its_text: say, its voice or ffmpeg unavailable");
        return;
    };
    let fox_secs = fox.len() as f32 / 16_000.0;
    let fox_window = to_window(&fox);

    let mut no_speech: Vec<(String, Vec<f32>)> =
        vec![("zeros 30 s".to_owned(), vec![0.0; WINDOW_SAMPLES])];
    for dbfs in [-80.0, -70.0, -60.0, -50.0, -40.0, -30.0] {
        no_speech.push((format!("noise {dbfs} dBFS"), noise(WINDOW_SAMPLES, dbfs)));
    }
    no_speech.extend([
        (
            "hum 50 Hz -50 dBFS".to_owned(),
            tone(WINDOW_SAMPLES, 50.0, -50.0),
        ),
        (
            "tone 1 kHz -40 dBFS".to_owned(),
            tone(WINDOW_SAMPLES, 1000.0, -40.0),
        ),
        ("zeros 0.1 s".to_owned(), vec![0.0; 1600]),
        (
            "last 0.1 s of the speech".to_owned(),
            fox[fox.len() - 1600..].to_vec(),
        ),
    ]);
    for (label, window) in &no_speech {
        let alone = transcribe_samples(&t, window, "en");
        assert!(
            alone.segments.is_empty(),
            "{label}, alone: {:?}",
            alone.segments
        );
        let after_speech = transcribe_samples(&t, &[fox_window.as_slice(), window].concat(), "en");
        for s in &after_speech.segments {
            assert!(
                s.end <= fox_secs + 0.5,
                "{label}, behind a prompt: segment outside the speech: {s:?}"
            );
        }
        let w = wer(&normalize(FOX), &normalize(&after_speech.text));
        assert_eq!(w, 0.0, "{label}, behind a prompt: {:?}", after_speech.text);
    }

    // White noise at `below` dB under the level of the speech.
    let mixed = |below: f64| -> Vec<f32> {
        fox.iter()
            .zip(noise(fox.len(), rms_dbfs(&fox) - below))
            .map(|(x, y)| x + y)
            .collect()
    };
    // Each row: the label, the samples, the spoken text and the word errors
    // that the row allows. At 22 dB the window must give its sentence with at
    // most one other word: a skipped window gives no text and fails the row.
    let speech: [(&str, Vec<f32>, String, usize); 10] = [
        ("the sentence", fox.clone(), FOX.to_owned(), 0),
        (
            "at -40 dBFS",
            at_level(fox.clone(), -40.0),
            FOX.to_owned(),
            0,
        ),
        (
            "at -55 dBFS",
            at_level(fox.clone(), -55.0),
            FOX.to_owned(),
            0,
        ),
        ("in noise 32 dB below it", mixed(32.0), FOX.to_owned(), 0),
        ("in noise 26 dB below it", mixed(26.0), FOX.to_owned(), 0),
        ("in noise 22 dB below it", mixed(22.0), FOX.to_owned(), 1),
        ("before zeros to 30 s", fox_window, FOX.to_owned(), 0),
        (
            "after 25 s of zeros",
            [silence(25).as_slice(), fox.as_slice()].concat(),
            FOX.to_owned(),
            0,
        ),
        ("one word", word, "Hello.".to_owned(), 0),
        (
            "behind a prompt",
            [to_window(&lantern).as_slice(), fox.as_slice()].concat(),
            format!("{LANTERN} {FOX}"),
            0,
        ),
    ];
    for (label, samples, script, allowed) in &speech {
        let r = transcribe_samples(&t, samples, "en");
        let words = normalize(&r.text);
        let errors = edit_distance(&normalize(script), &words);
        assert!(
            !words.is_empty() && errors <= *allowed,
            "speech {label}: {errors} word errors, {allowed} allowed: {:?}",
            r.text
        );
    }
}

// ── Speech above the no-speech bound ────────────────────────────────────────

/// Thirty seconds of four spoken words in a loop read a no-speech probability
/// above the bound (0.92 for this voice), and the model transcribes them. The
/// window is not skipped: the reply has the counted words, in the digits that
/// the model writes, from the start of the clip.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test a_spoken_repetition -- --ignored --test-threads=1"]
fn a_spoken_repetition_above_the_no_speech_bound_is_not_skipped() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP a_spoken_repetition_above_the_no_speech_bound_is_not_skipped: whisper snapshot not present");
        return;
    };
    let Some(samples) = say_samples(&"one two three four ".repeat(45)) else {
        eprintln!("SKIP a_spoken_repetition_above_the_no_speech_bound_is_not_skipped: say, its voice or ffmpeg unavailable");
        return;
    };
    assert!(
        samples.len() > WINDOW_SAMPLES,
        "the clip is over one window"
    );
    // A request above temperature 1.0 is one decode for each window, and each
    // pick of it is at that temperature. A pick at temperature 0 is a pick of
    // the bounded decode of the gate, which runs only above the bound.
    let mut temperatures: Vec<f32> = Vec::new();
    let r = t
        .transcribe(
            &samples,
            &english(1.5),
            &mut |row, temperature| {
                temperatures.push(temperature);
                Ok(largest_logit(row))
            },
            Device::Gpu,
            &AtomicBool::new(false),
        )
        .expect("transcribe");
    assert_eq!(
        temperatures.first(),
        Some(&0.0),
        "precondition: the first window of the clip reads above the no-speech bound for this \
         voice, so the gate decodes it"
    );
    let first_window: String = r
        .segments
        .iter()
        .filter(|s| s.start < 30.0)
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    println!("repetition, first window: {first_window:?}");
    let words = normalize(&first_window);
    assert!(
        words.starts_with(&["1", "2", "3", "4", "1", "2", "3", "4"].map(str::to_owned)),
        "the first window has the spoken count: {first_window:?}"
    );
    for digit in ["1", "2", "3", "4"] {
        let count = words.iter().filter(|w| w.as_str() == digit).count();
        assert!(
            count >= 20,
            "{digit} is written {count} times: {first_window:?}"
        );
    }
}

/// The bounded decode of the gate picks at temperature 0 also when the
/// request temperature is above 0, so the gate draws nothing from the
/// generator of the request. A window of zeros comes before the speech: each
/// pick of that window is a pick of the gate, and it is at 0. The picks of the
/// spoken window are at the request temperature.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test the_gate_picks_at_temperature_0 -- --ignored --test-threads=1"]
fn the_gate_picks_at_temperature_0_for_a_request_above_0() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP the_gate_picks_at_temperature_0_for_a_request_above_0: whisper snapshot not present");
        return;
    };
    let Some(speech) = say_samples(FOX) else {
        eprintln!("SKIP the_gate_picks_at_temperature_0_for_a_request_above_0: say, its voice or ffmpeg unavailable");
        return;
    };
    let mut samples = silence(30);
    samples.extend(speech);
    let mut temperatures: Vec<f32> = Vec::new();
    let r = t
        .transcribe(
            &samples,
            &english(0.4),
            &mut |row, temperature| {
                temperatures.push(temperature);
                Ok(largest_logit(row))
            },
            Device::Gpu,
            &AtomicBool::new(false),
        )
        .expect("transcribe");
    assert!(
        r.segments.iter().all(|s| s.start >= 30.0) && !r.segments.is_empty(),
        "the window of zeros is skipped and the speech is decoded: {:?}",
        r.segments
    );
    let of_the_gate = temperatures.iter().take_while(|t| **t == 0.0).count();
    assert!(of_the_gate > 0, "the gate picks at 0: {temperatures:?}");
    assert!(
        temperatures[of_the_gate..].iter().all(|t| *t == 0.4),
        "each pick after the gate is at the request temperature: {temperatures:?}"
    );
    assert!(temperatures.len() > of_the_gate, "the speech is decoded");
}

/// Thirty seconds of ONE repeated word also read above the bound (0.88 for
/// this voice), and for them the greedy decode of the model with no prompt is
/// one word (two text tokens). The gate cannot tell that from a stock phrase:
/// the window is skipped, and the reply loses that one word. This test holds
/// the stated limit of the gate, not a wanted result.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test thirty_seconds_of_one_repeated_word -- --ignored --test-threads=1"]
fn thirty_seconds_of_one_repeated_word_are_skipped() {
    let Some(t) = load_transcriber() else {
        eprintln!(
            "SKIP thirty_seconds_of_one_repeated_word_are_skipped: whisper snapshot not present"
        );
        return;
    };
    let Some(samples) = say_samples(&"hello, ".repeat(60)) else {
        eprintln!("SKIP thirty_seconds_of_one_repeated_word_are_skipped: say, its voice or ffmpeg unavailable");
        return;
    };
    assert!(
        samples.len() > WINDOW_SAMPLES,
        "the clip is over one window"
    );
    let r = transcribe_samples(&t, &samples[..WINDOW_SAMPLES], "en");
    assert!(
        r.segments.is_empty(),
        "this test holds a stated limit of the no-speech gate, not a wanted result: a better \
         rule can give this window a segment. Change the test and `docs/AUDIO.md`, \
         \"Known limits\", with the rule. Segments: {:?}",
        r.segments
    );
}

// ── The language of an `auto` request ───────────────────────────────────────

/// With `language: "auto"`, the language of a file is that of the first window
/// that the gate does not skip: 30 s of zeros before a German sentence do not
/// set it. A file in which each window is skipped has no decoded window, and
/// the reply names `english`.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test the_language_of_auto -- --ignored --test-threads=1"]
fn the_language_of_auto_is_that_of_the_first_window_that_is_not_skipped() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP the_language_of_auto_is_that_of_the_first_window_that_is_not_skipped: whisper snapshot not present");
        return;
    };
    let german = "Der schnelle braune Fuchs springt über den faulen Hund.";
    let Some(wav) = synth_voice_clip("Anna", german) else {
        eprintln!("SKIP the_language_of_auto_is_that_of_the_first_window_that_is_not_skipped: the say voice Anna or ffmpeg is unavailable");
        return;
    };
    let speech = WavDecoder::decode(&std::fs::read(wav).expect("read clip"))
        .expect("decode clip")
        .samples;
    let alone = transcribe_samples(&t, &speech, "auto");
    assert_eq!(
        alone.language, "german",
        "precondition: the model detects the speech of the voice as German; text {:?}",
        alone.text
    );

    let mut samples = silence(30);
    samples.extend(&speech);
    let r = transcribe_samples(&t, &samples, "auto");
    println!("auto after silence: {} {:?}", r.language, r.text);
    assert_eq!(r.language, "german", "text {:?}", r.text);
    assert!(
        r.segments.iter().all(|s| s.start >= 30.0),
        "segments: {:?}",
        r.segments
    );
    assert_eq!(r.text, alone.text);

    let silent = transcribe_samples(&t, &silence(60), "auto");
    assert!(
        silent.segments.is_empty(),
        "segments: {:?}",
        silent.segments
    );
    assert_eq!(silent.language, "english");
}

// ── The temperature fallback ────────────────────────────────────────────────

/// The pick of the fallback test and what it was given. It is not a sampler:
/// it has no generator. At temperature 0 it gives the largest logit. Above 0,
/// each pick whose number is a multiple of `20 / tenths` gives the second
/// largest logit that is not masked, so a decode differs at each temperature.
#[derive(Default)]
struct SteppedPick {
    picks_above_zero: usize,
    /// Each temperature it was given, in tenths, in order, without repeats.
    temperatures: Vec<u32>,
}

impl SteppedPick {
    fn pick(&mut self, row: &[f32], temperature: f32) -> u32 {
        let tenths = (temperature * 10.0).round() as u32;
        if self.temperatures.last() != Some(&tenths) {
            self.temperatures.push(tenths);
        }
        let first = largest_logit(row);
        if tenths == 0 {
            return first;
        }
        self.picks_above_zero += 1;
        if !self.picks_above_zero.is_multiple_of((20 / tenths) as usize) {
            return first;
        }
        let mut second = first;
        for (id, logit) in row.iter().enumerate() {
            let better = second == first || logit > &row[second as usize];
            if id as u32 != first && logit.is_finite() && better {
                second = id as u32;
            }
        }
        second
    }
}

/// Four words spoken 20 times are a true repetition, and the greedy decode of
/// that window is not accepted. The engine decodes the window again above
/// temperature 0 and keeps that decode, and a second run of the request gives
/// the same tokens from the same decodes. A request temperature above 0 is the
/// temperature of the first decode.
///
/// The first assertion is the precondition, not the rule: one greedy decode
/// (a request above temperature 1.0 is one step) must be a loop for this `say`
/// voice and this model.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test a_window_that_needs_a_fallback -- --ignored --test-threads=1"]
fn a_window_that_needs_a_fallback_is_decoded_again_and_repeats() {
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP a_window_that_needs_a_fallback_is_decoded_again_and_repeats: whisper snapshot not present");
        return;
    };
    let Some(samples) = say_samples(&"one two three four ".repeat(20)) else {
        eprintln!("SKIP a_window_that_needs_a_fallback_is_decoded_again_and_repeats: say, its voice or ffmpeg unavailable");
        return;
    };
    let run = |temperature: f32| {
        let mut pick = SteppedPick::default();
        let r = t
            .transcribe(
                &samples,
                &english(temperature),
                &mut |row, temperature| Ok(pick.pick(row, temperature)),
                Device::Gpu,
                &AtomicBool::new(false),
            )
            .expect("transcribe");
        let segments: Vec<(Vec<u32>, f32)> = r
            .segments
            .iter()
            .map(|s| (s.tokens.clone(), s.temperature))
            .collect();
        (segments, pick.temperatures, r.text)
    };
    let greedy = transcribe_greedy(&t, &samples, &english(1.5), false).expect("transcribe");
    assert!(
        greedy.segments.iter().any(|s| s.compression_ratio > 2.4),
        "precondition: the greedy decode of the clip is refused as a loop; its ratios are \
         {:?} (the `say` voice or the greedy text of the model changed)",
        greedy
            .segments
            .iter()
            .map(|s| s.compression_ratio)
            .collect::<Vec<_>>()
    );

    let (segments, temperatures, text) = run(0.0);
    println!(
        "fallback: temperatures of the picks {temperatures:?}, of the segments {:?}, text {text:?}",
        segments.iter().map(|(_, t)| *t).collect::<Vec<_>>()
    );
    assert!(
        temperatures.starts_with(&[0, 2]),
        "the fallback goes up the ladder: {temperatures:?}"
    );
    assert!(
        segments.iter().any(|(_, temperature)| *temperature > 0.0),
        "a decode above temperature 0 is kept"
    );
    let (again, temperatures_again, _) = run(0.0);
    assert_eq!(temperatures_again, temperatures);
    assert_eq!(again, segments, "a second run gives the same tokens");

    let (_, from_the_request, _) = run(0.4);
    assert_eq!(
        from_the_request.first(),
        Some(&4),
        "the request temperature is the first step: {from_the_request:?}"
    );
}

/// Sentences of the synthetic long clip. Each holds one keyword that no other
/// sentence holds.
const LONG_CLIP: [(&str, &str); 28] = [
    (
        "violin",
        "The violin rested on the wooden table near the open window.",
    ),
    (
        "harbor",
        "Small boats drifted slowly across the quiet harbor at dawn.",
    ),
    (
        "lantern",
        "She carried a brass lantern down the narrow stone stairs.",
    ),
    (
        "glacier",
        "The glacier moved a few inches every single winter.",
    ),
    (
        "orchard",
        "Children picked ripe apples in the orchard after school.",
    ),
    (
        "compass",
        "He checked the compass twice before leaving the trail.",
    ),
    (
        "bakery",
        "The bakery on the corner opens long before sunrise.",
    ),
    (
        "telescope",
        "Through the telescope the rings of the planet looked sharp.",
    ),
    (
        "meadow",
        "Wild flowers covered the meadow behind the old farmhouse.",
    ),
    (
        "blanket",
        "A thick wool blanket kept the travelers warm all night.",
    ),
    (
        "canyon",
        "Their voices echoed from the walls of the deep canyon.",
    ),
    (
        "pianist",
        "The pianist bowed once and left the stage in silence.",
    ),
    (
        "volcano",
        "Smoke rose from the volcano for three days in a row.",
    ),
    (
        "library",
        "The library keeps rare maps in a locked room upstairs.",
    ),
    (
        "tortoise",
        "An old tortoise crossed the garden path very slowly.",
    ),
    (
        "lighthouse",
        "The keeper climbed the lighthouse to clean the great lamp.",
    ),
    (
        "carpenter",
        "A carpenter fixed the broken door with a single hinge.",
    ),
    (
        "waterfall",
        "Mist from the waterfall soaked everyone on the bridge.",
    ),
    (
        "chessboard",
        "The chessboard was set up and waiting for the next game.",
    ),
    (
        "satellite",
        "Engineers tracked the satellite as it passed overhead.",
    ),
    (
        "museum",
        "The museum moved its oldest statues to the new wing.",
    ),
    (
        "bicycle",
        "He rode his bicycle along the river every morning.",
    ),
    (
        "thunder",
        "Distant thunder rolled across the hills before the rain.",
    ),
    ("pottery", "She sold her pottery at the market on weekends."),
    (
        "submarine",
        "The submarine surfaced near the coast just after noon.",
    ),
    (
        "parrot",
        "A green parrot repeated every word the sailor said.",
    ),
    ("kettle", "The kettle whistled loudly in the empty kitchen."),
    (
        "vineyard",
        "Workers in the vineyard finished the harvest early this year.",
    ),
];

/// Silence after each sentence of the long clip, in seconds. No gap reaches a
/// whole window, so every window holds speech.
const LONG_CLIP_GAPS: [f32; 28] = [
    0.4, 1.2, 0.3, 2.5, 0.6, 4.0, 0.5, 1.0, 0.3, 3.0, 0.8, 0.4, 5.0, 0.6, 1.5, 0.3, 2.0, 0.7, 0.4,
    3.5, 0.5, 1.1, 0.3, 4.5, 0.6, 0.9, 2.2, 0.0,
];

/// The long clip as 16 kHz mono samples: every sentence of [`LONG_CLIP`]
/// spoken by `say`, each followed by its gap.
fn synth_long_clip() -> Option<Vec<f32>> {
    let mut samples = Vec::new();
    for ((_, sentence), gap) in LONG_CLIP.iter().zip(LONG_CLIP_GAPS) {
        let wav = synth_say_clip(sentence)?;
        let speech = WavDecoder::decode(&std::fs::read(wav).ok()?).ok()?;
        assert_eq!(speech.sample_rate, 16_000, "say clip rate");
        samples.extend(speech.samples);
        samples.extend(std::iter::repeat_n(0.0, (gap * 16_000.0) as usize));
    }
    Some(samples)
}

/// A clip of several windows: every sentence is transcribed once, in order,
/// with segment times that move forward and stay inside the clip. A window
/// that ends inside a sentence must hand that sentence to the next window
/// without losing or repeating it. A cancel flag set before the call opens no
/// window and returns `Cancelled`.
///
/// Not covered: the previous-text prompt. The clean clip decodes the same
/// words with no prompt, so a prompt that is dropped or built from the wrong
/// window passes.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test long_synthetic_clip -- --ignored --test-threads=1"]
fn long_synthetic_clip_is_transcribed_once_in_order() {
    let Some(t) = load_transcriber() else {
        eprintln!(
            "SKIP long_synthetic_clip_is_transcribed_once_in_order: whisper snapshot not present"
        );
        return;
    };
    let Some(samples) = synth_long_clip() else {
        eprintln!("SKIP long_synthetic_clip_is_transcribed_once_in_order: say, its voice or ffmpeg unavailable");
        return;
    };
    let opts = english(0.0);
    let cancelled = transcribe_greedy(&t, &samples, &opts, true);
    assert!(
        matches!(
            cancelled,
            Err(WhisperError::Cancelled { window: 0, seek: 0 })
        ),
        "a flag set before the call opens no window: {cancelled:?}"
    );
    let r = transcribe_greedy(&t, &samples, &opts, false).expect("transcribe");
    println!(
        "long clip: duration={:.2}s segments={} text={:?}",
        r.duration,
        r.segments.len(),
        r.text
    );
    assert!(
        r.duration > 90.0,
        "the clip must span several windows: {:.2}s",
        r.duration
    );
    assert_eq!(r.language, "english");

    let words = normalize(&r.text);
    let counts: Vec<(&str, usize)> = LONG_CLIP
        .iter()
        .map(|(keyword, _)| (*keyword, words.iter().filter(|w| w == keyword).count()))
        .collect();
    assert!(
        counts.iter().all(|&(_, n)| n == 1),
        "every sentence once; keyword counts: {counts:?}"
    );
    let positions: Vec<usize> = LONG_CLIP
        .iter()
        .filter_map(|(keyword, _)| words.iter().position(|w| w == keyword))
        .collect();
    assert!(
        positions.windows(2).all(|p| p[0] < p[1]),
        "sentences out of order: {positions:?}"
    );

    let script: String = LONG_CLIP
        .iter()
        .map(|(_, s)| *s)
        .collect::<Vec<_>>()
        .join(" ");
    let w = wer(&normalize(&script), &words);
    assert!(w <= 0.03, "long clip WER {w:.4} above 0.03");

    for pair in r.segments.windows(2) {
        assert!(
            pair[1].start >= pair[0].start,
            "segment starts go back: {pair:?}"
        );
    }
    for s in &r.segments {
        assert!(
            s.end >= s.start && s.end <= r.duration + 0.02,
            "segment outside the clip ({:.2}s): {s:?}",
            r.duration
        );
    }
}

/// Long-form regression: transcribe every fixture audio with a sibling VTT and
/// assert normalized WER ≤ threshold on the FULL file.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test long_form_regression -- --ignored --test-threads=1"]
fn long_form_regression() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if !fixtures.exists() {
        eprintln!("SKIP long_form_regression: no fixtures dir");
        return;
    }
    let Some(t) = load_transcriber() else {
        eprintln!("SKIP long_form_regression: whisper snapshot not present");
        return;
    };

    const AUDIO_EXTS: &[&str] = &["m4a", "wav", "mp3", "flac", "ogg", "aac"];
    let mut ran = 0usize;
    for entry in std::fs::read_dir(&fixtures).expect("read fixtures") {
        let path = entry.expect("dir entry").path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase)
            .unwrap_or_default();
        if !AUDIO_EXTS.contains(&ext.as_str()) {
            continue;
        }
        // Sibling reference: <stem>.transcript.vtt (stem strips the audio ext).
        let vtt = path.with_extension("transcript.vtt");
        if !vtt.exists() {
            continue;
        }

        let t0 = std::time::Instant::now();
        let result = transcribe_file(&t, &path, "en");
        let elapsed = t0.elapsed().as_secs_f64();
        assert_eq!(result.language, "english");
        let rtf = if result.duration > 0.0 {
            elapsed / f64::from(result.duration)
        } else {
            0.0
        };

        let reference = normalize(&parse_vtt_reference(&vtt));
        let hyp = normalize(&result.text);
        let w = wer(&reference, &hyp);

        println!(
            "== {} ==\n  duration={:.1}s segments={} decode={:.1}s RTF={rtf:.3}\n  ref_words={} hyp_words={} WER={w:.4}",
            path.file_name().unwrap().to_string_lossy(),
            result.duration,
            result.segments.len(),
            elapsed,
            reference.len(),
            hyp.len(),
        );
        // Print a few aligned excerpts.
        for i in [0usize, hyp.len() / 3, 2 * hyp.len() / 3] {
            let r: String = reference
                .iter()
                .skip(i)
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            let h: String = hyp
                .iter()
                .skip(i)
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            println!("  ref@{i}: {r}\n  hyp@{i}: {h}");
        }

        assert!(
            w <= 0.30,
            "{}: normalized WER {w:.4} exceeds 0.30",
            path.file_name().unwrap().to_string_lossy()
        );
        ran += 1;
    }

    if ran == 0 {
        eprintln!("SKIP long_form_regression: no fixture audio+VTT pairs found");
    }
}
