//! Thread-boundary tests for the models the server loads inside a request:
//! jina-v4, Whisper and TTS.
//!
//! Each model has two cells. The per-request cell loads the model and serves
//! one request on one thread, and serves the next request on another: the
//! production path when every request succeeds. The load-only cell loads the
//! model on one thread and serves the first request on another: the production
//! path when the request that loaded the model fails after the load, so no
//! forward evaluated what the loader built.
//!
//! The seam, the observable, what cannot move and the mutation list are in the
//! module docs of `thread_boundary/mod.rs`.
//!
//! Run: `make gpu-test CRATE=rmlx-server FILTER=thread`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

mod thread_boundary;

use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
use rmlx_audio::tts::{synthesize, TtsModel, TtsTokenizer};
use rmlx_audio::whisper::WhisperModel;
use rmlx_mlx::Device;
use rmlx_models::jina_v4::{self, JinaV4};
use thread_boundary::{
    f32_bits, hand_over, snapshot, spoken_wav, Split, RED_PNG_BASE64, SPOKEN_TEXT,
};

const JINA_V4: (&str, &str) = ("jinaai__jina-embeddings-v4", "JinaEmbeddingsV4Model");
const WHISPER: (&str, &str) = ("mlx-community__whisper-large-v3-mlx", "whisper");
const WHISPER_TOKENIZER: (&str, &str) = ("openai__whisper-large-v3-tokenizer", "");
const TTS: (&str, &str) = (
    "mlx-community__Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "Qwen3TTSForConditionalGeneration",
);
const TTS_CODEC: (&str, &str) = ("Qwen__Qwen3-TTS-Tokenizer-12Hz", "Qwen3TTSTokenizerV2Model");

/// Run `first`, then `second` on its state: on one thread (the oracle), then
/// across `split`. Both steps must return what the oracle's steps returned.
/// Returns the oracle's second step.
fn assert_steps<S, A, B, F, G>(test: &str, split: Split, first: F, second: G) -> B
where
    S: Send + 'static,
    A: PartialEq + Debug + Send + 'static,
    B: PartialEq + Debug + Send + 'static,
    F: Fn() -> (S, A) + Clone + Send + 'static,
    G: Fn(S) -> Result<B, String> + Clone + Send + 'static,
{
    let (oracle_first, oracle_second) = hand_over(Split::None, first.clone(), second.clone());
    let oracle_second =
        oracle_second.unwrap_or_else(|e| panic!("{test}: the one-thread oracle must succeed: {e}"));
    let (crossed_first, crossed_second) = hand_over(split, first, second);
    let crossed_second = crossed_second.unwrap_or_else(|e| {
        panic!("{test}: the step after the hand-over failed on a thread that did not load the model: {e}")
    });
    assert_eq!(
        crossed_first, oracle_first,
        "{test}: the step before the hand-over differs from the oracle"
    );
    assert_eq!(
        crossed_second, oracle_second,
        "{test}: the step after the hand-over differs from the oracle"
    );
    oracle_second
}

// ── jina-v4 ─────────────────────────────────────────────────────────────────

fn jina_load(dir: &Path) -> (JinaV4, tokenizers::Tokenizer) {
    let model = jina_v4::load_from_path(dir).expect("jina_v4::load_from_path");
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    (model, tokenizer)
}

fn ids(tokenizer: &tokenizers::Tokenizer, text: &str, special: bool) -> Vec<i64> {
    tokenizer
        .encode(text, special)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|&id| i64::from(id))
        .collect()
}

fn jina_text(
    model: &JinaV4,
    tokenizer: &tokenizers::Tokenizer,
    device: Device,
) -> Result<Vec<u32>, String> {
    model
        .embed_single(&ids(tokenizer, SPOKEN_TEXT, true), device, None)
        .map(|v| f32_bits(&v))
        .map_err(|e| e.to_string())
}

fn jina_image(
    model: &JinaV4,
    tokenizer: &tokenizers::Tokenizer,
    dir: &Path,
    device: Device,
) -> Result<Vec<u32>, String> {
    let png = rmlx_server::image_io::load_image(
        &format!("data:image/png;base64,{RED_PNG_BASE64}"),
        rmlx_server::image_io::DEFAULT_HTTP_TIMEOUT,
    )
    .expect("png");
    let pcfg = jina_v4::ImagePreprocessConfig::from_model_dir(dir).expect("preprocessor");
    let pv = jina_v4::preprocess_image_bytes(&png, &pcfg).expect("preprocess");
    let prompt = ids(tokenizer, jina_v4::image_prompt(), false);
    model
        .embed_image_single(&prompt, &pv, device, None, None, 0)
        .map(|v| f32_bits(&v))
        .map_err(|e| e.to_string())
}

/// The first request loads jina-v4 and embeds text on one thread. The second
/// embeds an image on another: the first use of the vision weights.
#[test]
#[ignore = "requires Metal GPU and the jina-v4 snapshot; run with `make gpu-test`"]
fn jina_image_embedding_runs_on_a_thread_that_did_not_load_it() {
    let test = "jina_image_embedding_runs_on_a_thread_that_did_not_load_it";
    let Some(dir) = snapshot(test, JINA_V4.0, JINA_V4.1) else {
        return;
    };
    let device = Device::Gpu;
    let d1 = dir.clone();
    let first = move || {
        let (model, tokenizer) = jina_load(&d1);
        let text = jina_text(&model, &tokenizer, device);
        ((model, tokenizer), text)
    };
    let second = move |(model, tokenizer): (JinaV4, tokenizers::Tokenizer)| {
        jina_image(&model, &tokenizer, &dir, device)
    };
    let image = assert_steps(test, Split::BeforeLastRequest, first, second);
    assert!(!image.is_empty(), "{test}: the oracle embedded nothing");
}

/// jina-v4 loads on one thread; its first text and image embeddings run on
/// another.
#[test]
#[ignore = "requires Metal GPU and the jina-v4 snapshot; run with `make gpu-test`"]
fn jina_first_embedding_runs_on_a_thread_that_did_not_load_it() {
    let test = "jina_first_embedding_runs_on_a_thread_that_did_not_load_it";
    let Some(dir) = snapshot(test, JINA_V4.0, JINA_V4.1) else {
        return;
    };
    let device = Device::Gpu;
    let d1 = dir.clone();
    let first = move || (jina_load(&d1), ());
    let second = move |(model, tokenizer): (JinaV4, tokenizers::Tokenizer)| {
        let text = jina_text(&model, &tokenizer, device)?;
        let image = jina_image(&model, &tokenizer, &dir, device)?;
        Ok((text, image))
    };
    let (text, image) = assert_steps(test, Split::AfterLoad, first, second);
    assert!(
        !text.is_empty() && !image.is_empty(),
        "{test}: the oracle embedded nothing"
    );
}

// ── Whisper ─────────────────────────────────────────────────────────────────

type Whisper = (Arc<WhisperModel>, Arc<WhisperTokenizer>);

fn whisper_load(model_dir: &Path, tok_dir: &Path) -> Whisper {
    let model = WhisperModel::load(model_dir).expect("WhisperModel::load");
    let tokenizer = WhisperTokenizer::from_path(tok_dir).expect("WhisperTokenizer");
    (Arc::new(model), Arc::new(tokenizer))
}

fn transcribe(
    (model, tokenizer): &Whisper,
    samples: &[f32],
    device: Device,
) -> Result<String, String> {
    let opts = TranscribeOptions {
        language: "en".to_owned(),
        task: WhisperTask::Transcribe,
        temperature: 0.0,
        condition_on_previous_text: true,
    };
    Transcriber::new(Arc::clone(model), Arc::clone(tokenizer))
        .and_then(|t| t.transcribe(samples, &opts, device))
        .map(|t| t.text)
        .map_err(|e| e.to_string())
}

/// The Whisper snapshots and the spoken clip, made after the snapshot check.
fn whisper_inputs(test: &str) -> Option<(PathBuf, PathBuf, Arc<[f32]>)> {
    let model_dir = snapshot(test, WHISPER.0, WHISPER.1)?;
    let tok_dir = snapshot(test, WHISPER_TOKENIZER.0, WHISPER_TOKENIZER.1)?;
    let (raw, rate) = rmlx_audio::wav::WavDecoder::decode(&spoken_wav()).expect("decode wav");
    let samples: Arc<[f32]> = rmlx_audio::transcribe::resample_to_16k(&raw, rate).into();
    Some((model_dir, tok_dir, samples))
}

/// The first request loads Whisper and transcribes; the second transcribes on
/// another thread.
#[test]
#[ignore = "requires Metal GPU and the Whisper snapshots; run with `make gpu-test`"]
fn whisper_transcribes_on_a_thread_that_did_not_load_it() {
    let test = "whisper_transcribes_on_a_thread_that_did_not_load_it";
    let Some((model_dir, tok_dir, samples)) = whisper_inputs(test) else {
        return;
    };
    let device = Device::Gpu;
    let s1 = Arc::clone(&samples);
    let first = move || {
        let whisper = whisper_load(&model_dir, &tok_dir);
        let text = transcribe(&whisper, &s1, device);
        (whisper, text)
    };
    let second = move |whisper: Whisper| transcribe(&whisper, &samples, device);
    let text = assert_steps(test, Split::BeforeLastRequest, first, second);
    assert!(
        !text.trim().is_empty(),
        "{test}: the oracle transcribed nothing"
    );
}

/// Whisper loads on one thread; its first transcription runs on another.
#[test]
#[ignore = "requires Metal GPU and the Whisper snapshots; run with `make gpu-test`"]
fn whisper_first_transcription_runs_on_a_thread_that_did_not_load_it() {
    let test = "whisper_first_transcription_runs_on_a_thread_that_did_not_load_it";
    let Some((model_dir, tok_dir, samples)) = whisper_inputs(test) else {
        return;
    };
    let device = Device::Gpu;
    let first = move || (whisper_load(&model_dir, &tok_dir), ());
    let second = move |whisper: Whisper| transcribe(&whisper, &samples, device);
    let text = assert_steps(test, Split::AfterLoad, first, second);
    assert!(
        !text.trim().is_empty(),
        "{test}: the oracle transcribed nothing"
    );
}

// ── TTS ─────────────────────────────────────────────────────────────────────

fn tts_config(model_dir: &Path, codec_dir: &Path, device: Device) -> (TtsModel, TtsTokenizer) {
    let model = TtsModel::load_config(model_dir, codec_dir, device).expect("TtsModel::load_config");
    let tokenizer = TtsTokenizer::from_path(model_dir).expect("TtsTokenizer");
    (model, tokenizer)
}

fn speak(
    (mut model, tokenizer): (TtsModel, TtsTokenizer),
) -> (Result<Vec<u32>, String>, (TtsModel, TtsTokenizer)) {
    let samples = synthesize(SPOKEN_TEXT, "serena", &mut model, &tokenizer)
        .map(|(samples, _rate)| f32_bits(&samples))
        .map_err(|e| e.to_string());
    (samples, (model, tokenizer))
}

/// The first request loads the TTS weights inside `synthesize` and speaks; the
/// second speaks on another thread.
#[test]
#[ignore = "requires Metal GPU and the Qwen3-TTS snapshots; run with `make gpu-test`"]
fn tts_synthesizes_on_a_thread_that_did_not_load_it() {
    let test = "tts_synthesizes_on_a_thread_that_did_not_load_it";
    let Some(model_dir) = snapshot(test, TTS.0, TTS.1) else {
        return;
    };
    let Some(codec_dir) = snapshot(test, TTS_CODEC.0, TTS_CODEC.1) else {
        return;
    };
    let device = Device::Gpu;
    let first = move || {
        let (samples, tts) = speak(tts_config(&model_dir, &codec_dir, device));
        (tts, samples)
    };
    let second = |tts: (TtsModel, TtsTokenizer)| speak(tts).0;
    let samples = assert_steps(test, Split::BeforeLastRequest, first, second);
    assert!(
        !samples.is_empty(),
        "{test}: the oracle synthesized nothing"
    );
}

/// `TtsModel::load` runs on one thread; the first synthesis on another.
#[test]
#[ignore = "requires Metal GPU and the Qwen3-TTS snapshots; run with `make gpu-test`"]
fn tts_first_synthesis_runs_on_a_thread_that_did_not_load_it() {
    let test = "tts_first_synthesis_runs_on_a_thread_that_did_not_load_it";
    let Some(model_dir) = snapshot(test, TTS.0, TTS.1) else {
        return;
    };
    let Some(codec_dir) = snapshot(test, TTS_CODEC.0, TTS_CODEC.1) else {
        return;
    };
    let device = Device::Gpu;
    let first = move || {
        let (mut model, tokenizer) = tts_config(&model_dir, &codec_dir, device);
        model.load().expect("TtsModel::load");
        ((model, tokenizer), ())
    };
    let second = |tts: (TtsModel, TtsTokenizer)| speak(tts).0;
    let samples = assert_steps(test, Split::AfterLoad, first, second);
    assert!(
        !samples.is_empty(),
        "{test}: the oracle synthesized nothing"
    );
}
