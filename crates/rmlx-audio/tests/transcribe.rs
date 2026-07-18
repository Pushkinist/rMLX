//! Real-model Whisper transcription integration tests.
//!
//! These tests resolve the Whisper snapshot + tokenizer from `RMLX_O_MODELS_ROOT`
//! auto-discovery (the `make model-check-full` convention) and **skip gracefully**
//! when the model, tokenizer, or fixtures are absent — there is no bespoke env
//! var. Two layers:
//!
//! 1. `say_clip_deterministic` — portable, asset-free: synthesise a known English
//!    sentence with macOS `say` + `ffmpeg`, transcribe it, assert it matches the
//!    sentence (low WER, case/punct-insensitive) and is byte-identical across two
//!    runs at temp=0. Skips when `say`/`ffmpeg` are unavailable.
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
use std::sync::Arc;

use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
use rmlx_audio::wav::WavDecoder;
use rmlx_audio::whisper::WhisperModel;
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
    rmlx_mlx::ensure_gpu_default_stream();
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
    let (raw, rate) = WavDecoder::decode(&bytes).expect("decode audio");
    let samples = rmlx_audio::transcribe::resample_to_16k(&raw, rate);
    let opts = TranscribeOptions {
        language: language.to_owned(),
        task: WhisperTask::Transcribe,
        temperature: 0.0,
        condition_on_previous_text: true,
    };
    t.transcribe(&samples, &opts, Device::Gpu)
        .expect("transcribe")
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Portable smoke: a known `say` sentence transcribes correctly and is
/// deterministic across two runs at temp=0.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test say_clip -- --ignored --test-threads=1"]
fn say_clip_deterministic() {
    let Some(t) = load_transcriber() else {
        eprintln!("skip say_clip_deterministic: whisper snapshot not present");
        return;
    };
    let sentence = "The quick brown fox jumps over the lazy dog.";
    let Some(wav) = synth_say_clip(sentence) else {
        eprintln!("skip say_clip_deterministic: say/ffmpeg unavailable");
        return;
    };

    let r1 = transcribe_file(&t, &wav, "en");
    let r2 = transcribe_file(&t, &wav, "en");

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
    // Threshold 0.25: the sentence content must be correct. Whisper is known to
    // append a single filler token ("you", "thank you") on the trailing-silence
    // boundary of a very short clip; one such word over a 9-word sentence is
    // ~0.11 WER and does not indicate a decode defect. Determinism (above) is the
    // stricter property being asserted.
    assert!(
        w <= 0.25,
        "say-clip WER {w:.3} too high (>0.25); got {:?}",
        r1.text
    );
}

/// Long-form regression: transcribe every fixture audio with a sibling VTT and
/// assert normalized WER ≤ threshold on the FULL file.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test long_form_regression -- --ignored --test-threads=1"]
fn long_form_regression() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if !fixtures.exists() {
        eprintln!("skip long_form_regression: no fixtures dir");
        return;
    }
    let Some(t) = load_transcriber() else {
        eprintln!("skip long_form_regression: whisper snapshot not present");
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
        eprintln!("skip long_form_regression: no fixture audio+VTT pairs found");
    }
}
