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

/// Synthesise a known sentence to a 16 kHz mono WAV via `say` + `ffmpeg`.
fn synth_say_clip(sentence: &str) -> Option<PathBuf> {
    if !have_cmd("say") || !have_cmd("ffmpeg") {
        return None;
    }
    let dir = std::env::temp_dir();
    let aiff = dir.join("rmlx_say_clip.aiff");
    let wav = dir.join("rmlx_say_clip.wav");
    let say_ok = std::process::Command::new("say")
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
    let opts = TranscribeOptions {
        language: language.to_owned(),
        task: WhisperTask::Transcribe,
        temperature: 0.0,
        condition_on_previous_text: true,
    };
    t.transcribe(&samples, &opts, Device::Gpu, &AtomicBool::new(false))
        .expect("transcribe")
}

// ── Tests ───────────────────────────────────────────────────────────────────

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
        eprintln!("SKIP say_clip_deterministic: say/ffmpeg unavailable");
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
        eprintln!("SKIP two_sentence_clip_is_two_segments: say/ffmpeg unavailable");
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
        eprintln!("SKIP long_synthetic_clip_is_transcribed_once_in_order: say/ffmpeg unavailable");
        return;
    };
    let opts = TranscribeOptions {
        language: "en".to_owned(),
        task: WhisperTask::Transcribe,
        temperature: 0.0,
        condition_on_previous_text: true,
    };
    let cancelled = t.transcribe(&samples, &opts, Device::Gpu, &AtomicBool::new(true));
    assert!(
        matches!(
            cancelled,
            Err(WhisperError::Cancelled { window: 0, seek: 0 })
        ),
        "a flag set before the call opens no window: {cancelled:?}"
    );
    let r = t
        .transcribe(&samples, &opts, Device::Gpu, &AtomicBool::new(false))
        .expect("transcribe");
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
