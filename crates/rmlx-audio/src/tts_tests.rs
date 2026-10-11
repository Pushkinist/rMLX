//! Qwen3-TTS unit tests.

use super::{
    load_codec_decoder, run_frames, synthesize, talker_allow_mask, FrameStep, PickHead, TokenPick,
    TtsConfig, TtsError, TtsModel, TtsTokenizer, MAX_FRAMES,
};
use rmlx_mlx::Array;

/// A pick for synthesis calls that must fail before the first token.
pub(super) struct NoPick;

impl TokenPick for NoPick {
    fn pick(
        &mut self,
        head: PickHead,
        _logits: &Array,
        _allow: Option<&[bool]>,
        _history: &[u32],
    ) -> Result<u32, TtsError> {
        panic!("{head:?}: this synthesis must fail before its first pick")
    }
}

const TALKER_VOCAB: usize = 3072;
const CODEC_EOS: u32 = 2150;

/// The cap that the analysis measured as the defect's frame count stays the cap.
#[test]
fn max_frames_is_2048() {
    assert_eq!(MAX_FRAMES, 2048);
}

/// The mask blocks `[vocab - 1024, vocab)` except the end token, and nothing else.
#[test]
fn talker_mask_blocks_specials_and_keeps_the_end_token() {
    let allow = talker_allow_mask(TALKER_VOCAB, CODEC_EOS);
    assert_eq!(allow.len(), TALKER_VOCAB);
    for (id, &allowed) in allow.iter().enumerate() {
        let want = id < 2048 || id == CODEC_EOS as usize;
        assert_eq!(allowed, want, "id {id}: allowed={allowed}, want {want}");
    }
}

fn frame(id: u32) -> FrameStep {
    FrameStep::Frame(vec![id; 16])
}

/// A talker that never picks the end token runs to the cap, and the loop says so.
#[test]
fn run_frames_stops_at_the_cap_and_reports_truncation() {
    let mut calls = 0usize;
    let frames = run_frames(|step| {
        calls += 1;
        Ok(frame(step as u32))
    })
    .unwrap();
    assert_eq!(frames.codes.len(), MAX_FRAMES);
    assert_eq!(
        calls, MAX_FRAMES,
        "the loop must not call the talker past the cap"
    );
    assert!(frames.truncated, "a loop that reaches the cap is truncated");
    assert_eq!(frames.codes[MAX_FRAMES - 1][0], (MAX_FRAMES - 1) as u32);
}

/// The end token stops the loop at once and is not a frame.
#[test]
fn run_frames_stops_at_the_end_token() {
    let frames = run_frames(|step| Ok(if step == 5 { FrameStep::End } else { frame(7) })).unwrap();
    assert_eq!(frames.codes.len(), 5);
    assert!(
        !frames.truncated,
        "a loop that saw the end token is not truncated"
    );
}

/// The end token on the last step the cap allows is a finished generation.
#[test]
fn run_frames_end_token_on_the_last_step_is_not_truncated() {
    let frames = run_frames(|step| {
        Ok(if step == MAX_FRAMES - 1 {
            FrameStep::End
        } else {
            frame(1)
        })
    })
    .unwrap();
    assert_eq!(frames.codes.len(), MAX_FRAMES - 1);
    assert!(!frames.truncated);
}

/// A step error ends the loop with that error, not with a short WAV.
#[test]
fn run_frames_propagates_a_step_error() {
    let result = run_frames(|step| {
        if step == 3 {
            Err(TtsError::Mlx("step failed".into()))
        } else {
            Ok(frame(1))
        }
    });
    match result {
        Err(TtsError::Mlx(msg)) => assert_eq!(msg, "step failed"),
        other => panic!("expected the step error, got {other:?}"),
    }
}

/// Config parses correctly for the CustomVoice snapshot.
#[test]
fn tts_config_parses() {
    let json = r#"{
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671,
        "talker_config": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_language_id": {"english": 2050},
            "spk_id": {"serena": 3066, "vivian": 3065}
        }
    }"#;
    let cfg = TtsConfig::from_json(json).expect("config should parse");
    assert_eq!(cfg.model_type, "qwen3_tts");
    assert_eq!(cfg.tts_bos_token_id, 151672);
    assert_eq!(cfg.talker_config.hidden_size, 2048);
    assert_eq!(cfg.talker_config.num_code_groups, 16);
    assert_eq!(cfg.speaker_id("serena"), Some(3066));
    assert_eq!(cfg.speaker_id("vivian"), Some(3065));
    assert!(cfg.speaker_id("unknown_voice").is_none());
}

/// A CPU model whose one voice is `serena`, with no weights behind it.
fn serena_only_model() -> TtsModel {
    let json = r#"{
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671,
        "talker_config": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_language_id": {},
            "spk_id": {"serena": 3066}
        }
    }"#;
    TtsModel::new_for_test(
        TtsConfig::from_json(json).unwrap(),
        std::path::PathBuf::from("/tmp"),
        std::path::PathBuf::from("/tmp"),
        rmlx_mlx::Device::Cpu,
    )
}

/// Synthesize with an unknown voice returns the correct error.
#[test]
fn synthesize_unknown_voice_returns_error() {
    let mut model = serena_only_model();
    let tok = TtsTokenizer::stub();

    let result = synthesize("hello", "nonexistent_voice", &mut model, &tok, &mut NoPick);
    match result {
        Err(TtsError::UnknownVoice(v)) => assert_eq!(v, "nonexistent_voice"),
        other => panic!("expected UnknownVoice, got {other:?}"),
    }
}

/// Synthesize with a valid voice but missing weights returns a load error.
/// This test exercises the lazy-load
/// error path that fires when the talker model directory is invalid.
#[test]
fn synthesize_valid_voice_returns_load_error() {
    let mut model = serena_only_model();
    let tok = TtsTokenizer::stub();

    let result = synthesize("hello", "serena", &mut model, &tok, &mut NoPick);
    assert!(
        matches!(result, Err(TtsError::Load(_))),
        "expected Load error when weights missing, got {result:?}"
    );
}

/// The `(field, value)` pairs of every span and event that this thread opens
/// or emits while it is installed, including a value that a span records later.
#[derive(Default)]
struct Recorded {
    fields: std::sync::Mutex<Vec<(String, String)>>,
}

struct Collect<'a>(&'a mut Vec<(String, String)>);

impl tracing::field::Visit for Collect<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_owned(), value.to_owned()));
    }
}

impl tracing::Subscriber for Recorded {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        attributes.record(&mut Collect(&mut self.fields.lock().unwrap()));
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, values: &tracing::span::Record<'_>) {
        values.record(&mut Collect(&mut self.fields.lock().unwrap()));
    }
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        event.record(&mut Collect(&mut self.fields.lock().unwrap()));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The fields that spans and events record while `f` runs.
fn fields_recorded_by(f: impl FnOnce()) -> Vec<(String, String)> {
    let recorded = std::sync::Arc::new(Recorded::default());
    // A second live dispatcher keeps a callsite that another thread registers
    // now from being cached as always off, which would bypass the scoped one.
    let keep = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let guard = tracing::subscriber::set_default(std::sync::Arc::clone(&recorded));
    f();
    drop(guard);
    drop(keep);
    let fields = recorded.fields.lock().unwrap().clone();
    fields
}

const CLIENT_MARKER: &str = "CLIENT-MARKER-5c1e";

/// The request text and a voice that names no speaker reach no span field and
/// no event; the length of the text does.
#[test]
fn synthesize_logs_neither_the_text_nor_an_unknown_voice() {
    let text = format!("hello {CLIENT_MARKER}");
    let voice = format!("voice-{CLIENT_MARKER}");
    let fields = fields_recorded_by(|| {
        let mut model = serena_only_model();
        let result = synthesize(
            &text,
            &voice,
            &mut model,
            &TtsTokenizer::stub(),
            &mut NoPick,
        );
        assert!(
            matches!(result, Err(TtsError::UnknownVoice(_))),
            "{result:?}"
        );
    });
    assert!(
        fields
            .iter()
            .all(|(_, value)| !value.contains(CLIENT_MARKER)),
        "{fields:?}"
    );
    assert!(
        fields
            .iter()
            .any(|(name, value)| name == "text_len" && *value == text.len().to_string()),
        "{fields:?}"
    );
    assert!(fields.iter().all(|(name, _)| name != "voice"), "{fields:?}");
}

/// A voice that names a speaker is recorded once it resolved, as a string
/// value and not through `Debug`; the text is not.
#[test]
fn synthesize_records_a_known_voice_and_not_the_text() {
    let text = format!("hello {CLIENT_MARKER}");
    let fields = fields_recorded_by(|| {
        let mut model = serena_only_model();
        let result = synthesize(
            &text,
            "serena",
            &mut model,
            &TtsTokenizer::stub(),
            &mut NoPick,
        );
        assert!(matches!(result, Err(TtsError::Load(_))), "{result:?}");
    });
    assert!(
        fields
            .iter()
            .all(|(_, value)| !value.contains(CLIENT_MARKER)),
        "{fields:?}"
    );
    assert!(
        fields
            .iter()
            .any(|(name, value)| name == "voice" && value == "serena"),
        "{fields:?}"
    );
}

/// Debug codec decoder: loads real weights and runs with fixed codes.
/// Prints intermediate values so they can be compared with reference Python.
/// Run with: cargo test -p rmlx-audio codec_decoder_debug -- --nocapture 2>&1 | grep -v "^running\|^test "
#[test]
#[ignore = "requires real model weights at Open Models path; run manually for debugging"]
fn codec_decoder_debug() {
    // `<RMLX_O_MODELS_ROOT>/<slug>`, the join every other model-gated cell makes.
    // The path this used to carry was relative to the process's working
    // directory and resolved under no configuration this repo has, so the cell
    // stood down on every host including the ones holding the snapshot.
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!("SKIP codec_decoder_debug: RMLX_O_MODELS_ROOT is unset");
        return;
    };
    let codec_path = std::path::Path::new(&root)
        .join("mlx-community__Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        .join("speech_tokenizer");
    if !codec_path.exists() {
        eprintln!(
            "SKIP codec_decoder_debug: no codec decoder at {}",
            codec_path.display()
        );
        return;
    }

    let d = rmlx_mlx::Device::Gpu;
    let decoder = load_codec_decoder(&codec_path, d).expect("load codec decoder");

    // codes: [1, 16, 12] — code 100 for semantic, 50 for acoustic (same as Python reference)
    let t = 12i32;
    let mut codes_data = vec![0i32; 16 * 12];
    for item in codes_data.iter_mut().take(12) {
        *item = 100; // semantic code
    }
    for i in 1..16 {
        for j in 0..12 {
            codes_data[i * 12 + j] = 50; // acoustic codes
        }
    }
    let codes = Array::from_i32_slice(&codes_data, &[1, 16, t]).expect("build codes array");

    let samples = decoder.debug_decode(&codes, d).expect("debug decode");
    eprintln!(
        "samples len={} rms={:.6}",
        samples.len(),
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    );
}
