//! Qwen3-TTS unit tests.

use super::{
    run_frames, synthesize, talker_allow_mask, FrameStep, PickHead, TokenPick, TtsConfig, TtsError,
    TtsModel, TtsTokenizer, MAX_FRAMES,
};
use super::{text_rows, TextRows, TEXT_HEAD_TOKENS, TEXT_TAIL_TOKENS};
use crate::test_events::fields_recorded_by;
use rmlx_mlx::Array;
use std::sync::atomic::AtomicBool;

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

/// A step that made a frame and grew no key/value buffer.
fn frame(id: u32) -> (FrameStep, bool) {
    (FrameStep::Frame(vec![id; 16]), false)
}

/// A step in which the talker picked the end token.
const END: (FrameStep, bool) = (FrameStep::End, false);

/// A talker that never picks the end token runs to the cap, and the loop says so.
#[test]
fn run_frames_stops_at_the_cap_and_reports_truncation() {
    let mut calls = 0usize;
    let frames = run_frames(&AtomicBool::new(false), |step| {
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
    let frames = run_frames(&AtomicBool::new(false), |step| {
        Ok(if step == 5 { END } else { frame(7) })
    })
    .unwrap();
    assert_eq!(frames.codes.len(), 5);
    assert!(
        !frames.truncated,
        "a loop that saw the end token is not truncated"
    );
}

/// The end token on the last step the cap allows is a finished generation.
#[test]
fn run_frames_end_token_on_the_last_step_is_not_truncated() {
    let frames = run_frames(&AtomicBool::new(false), |step| {
        Ok(if step == MAX_FRAMES - 1 {
            END
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
    let result = run_frames(&AtomicBool::new(false), |step| {
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

    let result = synthesize(
        "hello",
        "nonexistent_voice",
        &mut model,
        &tok,
        &mut NoPick,
        &AtomicBool::new(false),
    );
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

    let result = synthesize(
        "hello",
        "serena",
        &mut model,
        &tok,
        &mut NoPick,
        &AtomicBool::new(false),
    );
    assert!(
        matches!(result, Err(TtsError::Load(_))),
        "expected Load error when weights missing, got {result:?}"
    );
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
            &AtomicBool::new(false),
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
            &AtomicBool::new(false),
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

/// Why the frame loop released the MLX buffer cache while `f` ran, one entry
/// for each release, in order.
fn release_reasons(f: impl FnOnce()) -> Vec<String> {
    fields_recorded_by(f)
        .into_iter()
        .filter(|(name, _)| name == "reason")
        .map(|(_, reason)| reason)
        .collect()
}

const GROWTH_FRAMES: [usize; 3] = [0, 4, 9];

/// A step that made a frame, and grew a key/value buffer in the frames of
/// [`GROWTH_FRAMES`].
fn frame_with_growth(step: usize) -> (FrameStep, bool) {
    (FrameStep::Frame(vec![0; 16]), GROWTH_FRAMES.contains(&step))
}

/// The loop releases the cache once after each frame that grew a buffer and
/// once when it ends, and at no other time.
#[test]
fn run_frames_releases_the_cache_after_each_growth_frame_and_at_the_end() {
    for frames in [1, 5, 12] {
        let reasons = release_reasons(|| {
            let got = run_frames(&AtomicBool::new(false), |step| {
                Ok(if step == frames {
                    END
                } else {
                    frame_with_growth(step)
                })
            })
            .unwrap();
            assert_eq!(got.codes.len(), frames);
        });
        let growths = GROWTH_FRAMES.iter().filter(|&&g| g < frames).count();
        let mut want = vec!["capacity growth"; growths];
        want.push("end of the loop");
        assert_eq!(reasons, want, "{frames} frames");
    }
}

/// A loop that runs to the cap releases as one that ends on the end token.
#[test]
fn run_frames_at_the_cap_releases_after_each_growth_frame_and_at_the_end() {
    let reasons = release_reasons(|| {
        let got = run_frames(&AtomicBool::new(false), |step| Ok(frame_with_growth(step))).unwrap();
        assert!(got.truncated);
    });
    assert_eq!(
        reasons,
        [
            "capacity growth",
            "capacity growth",
            "capacity growth",
            "end of the loop"
        ]
    );
}

/// The end token in a step that grew a buffer: the release at the end covers it.
#[test]
fn run_frames_releases_once_when_the_last_step_grew_a_buffer() {
    let reasons = release_reasons(|| {
        run_frames(&AtomicBool::new(false), |_| Ok((FrameStep::End, true))).unwrap();
    });
    assert_eq!(reasons, ["end of the loop"]);
}

/// A failed step still releases the cache.
#[test]
fn run_frames_releases_the_cache_when_a_step_fails() {
    let reasons = release_reasons(|| {
        let result = run_frames(&AtomicBool::new(false), |step| {
            if step == 2 {
                Err(TtsError::Mlx("step failed".into()))
            } else {
                Ok(frame_with_growth(step))
            }
        });
        assert!(result.is_err());
    });
    assert_eq!(reasons, ["capacity growth", "end of the loop"]);
}

/// Tokens of a chat text that holds `trailing` trailing text tokens: three role
/// tokens, the first text token, the trailing ones and five tail tokens.
const fn chat_tokens(trailing: usize) -> usize {
    trailing + 9
}

/// A text of one token embeds whole and keeps no text row, only `tts_eos`.
#[test]
fn text_rows_keep_every_row_of_a_short_text() {
    assert_eq!(
        text_rows(chat_tokens(0)),
        TextRows {
            embed: 9,
            trailing: 0,
            eos: true
        }
    );
    assert_eq!(
        text_rows(chat_tokens(100)),
        TextRows {
            embed: 109,
            trailing: 100,
            eos: true
        }
    );
}

/// Frame `i` reads row `i`: one row less than the cap leaves room for `tts_eos`
/// at index `cap - 1`; at the cap `tts_eos` would be row `cap`, which no frame
/// reads.
#[test]
fn text_rows_drop_tts_eos_only_when_no_frame_reaches_it() {
    assert_eq!(
        text_rows(chat_tokens(MAX_FRAMES - 1)),
        TextRows {
            embed: chat_tokens(MAX_FRAMES - 1),
            trailing: MAX_FRAMES - 1,
            eos: true
        }
    );
    let at_cap = TextRows {
        embed: 4 + MAX_FRAMES,
        trailing: MAX_FRAMES,
        eos: false,
    };
    assert_eq!(text_rows(chat_tokens(MAX_FRAMES)), at_cap);
    assert_eq!(text_rows(chat_tokens(MAX_FRAMES + 1)), at_cap);
}

/// The rows kept do not grow with the length of the text.
#[test]
fn text_rows_of_a_very_long_text_stay_at_the_cap() {
    let long = text_rows(chat_tokens(4_000_000));
    assert_eq!(long.embed, 4 + MAX_FRAMES);
    assert_eq!(long.trailing, MAX_FRAMES);
    assert!(!long.eos);
}

/// A chat text of fewer than nine tokens has no trailing text token. The chat
/// template rules it out with the model's tokenizer; another `tokenizer.json`
/// can give it. `text_rows` keeps the whole text and no trailing row, with
/// `tts_eos` as the first row the loop reads. `synthesize` does not refuse
/// such a text: it embeds the ids it was given, and its trailing slice, from
/// row `TEXT_HEAD_TOKENS` on, is empty. `tts_synthesize_on_cpu_is_admitted`
/// runs it on such a text, the stub tokenizer's, up to the embedding.
#[test]
fn text_rows_of_a_text_shorter_than_the_template_keep_it_whole() {
    let d = rmlx_mlx::Device::Cpu;
    let template_tokens = TEXT_HEAD_TOKENS + TEXT_TAIL_TOKENS;
    let head = TEXT_HEAD_TOKENS as i32;
    for tokens in 0..template_tokens {
        let rows = text_rows(tokens);
        assert_eq!(
            rows,
            TextRows {
                embed: tokens,
                trailing: 0,
                eos: true
            },
            "{tokens} tokens"
        );
        let embedded = Array::from_f32_slice(&vec![0.0; tokens], &[1, tokens as i32, 1]).unwrap();
        let trailing_end = head + rows.trailing as i32;
        let kept = embedded.slice(&[0, head, 0], &[1, trailing_end, 1], &[1, 1, 1], d);
        assert_eq!(kept.unwrap().shape(), [1, 0, 1], "{tokens} tokens");
    }
    let stub_chat_text = TtsTokenizer::stub()
        .encode("<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n")
        .unwrap();
    assert!(
        stub_chat_text.len() < template_tokens,
        "{} tokens",
        stub_chat_text.len()
    );
}

/// Where the text row that frame `frame` reads comes from.
#[derive(Debug, PartialEq, Eq)]
enum RowSource {
    Token(usize),
    Eos,
    Pad,
}

/// The loop's read of the full trailing layout: text tokens `4 .. tokens - 5`,
/// then `tts_eos`, then the pad row.
fn row_read_in_full_layout(frame: usize, tokens: usize) -> RowSource {
    let trailing = tokens - 9;
    match frame {
        f if f < trailing => RowSource::Token(4 + f),
        f if f == trailing => RowSource::Eos,
        _ => RowSource::Pad,
    }
}

/// The same read of the kept layout.
fn row_read_in_kept_layout(frame: usize, rows: &TextRows) -> RowSource {
    match frame {
        f if f < rows.trailing => RowSource::Token(4 + f),
        f if f == rows.trailing && rows.eos => RowSource::Eos,
        _ => RowSource::Pad,
    }
}

/// For every length of text, every frame the loop runs reads the same row from
/// the kept layout as from the full one, and the kept layout holds the ids it
/// names.
#[test]
fn text_rows_leave_every_row_a_frame_reads_unchanged() {
    let lengths = (9..=MAX_FRAMES + 40).chain([10_000, 1_000_000]);
    for tokens in lengths {
        let rows = text_rows(tokens);
        assert!(4 + rows.trailing <= rows.embed, "{tokens}: {rows:?}");
        assert!(rows.embed <= tokens, "{tokens}: {rows:?}");
        for frame in 0..MAX_FRAMES {
            assert_eq!(
                row_read_in_kept_layout(frame, &rows),
                row_read_in_full_layout(frame, tokens),
                "{tokens} tokens, frame {frame}"
            );
        }
    }
}

/// A text whose trailing rows and `tts_eos` fit is embedded whole, ids and
/// count unchanged, so its rows are the ones that a full embed gives.
#[test]
fn text_rows_embed_a_text_that_fits_whole() {
    for tokens in 9..(MAX_FRAMES + 9) {
        let rows = text_rows(tokens);
        assert_eq!(rows.embed, tokens);
        assert_eq!(rows.trailing, tokens - 9);
        assert!(rows.eos);
    }
}

#[path = "tts_stop_tests.rs"]
mod tts_stop_tests;
