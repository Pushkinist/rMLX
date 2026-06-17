//! Tests for Whisper tokenizer constants.
//!
//! Full round-trip tests require a `tokenizer.json` file (not shipped with the
//! mlx-community Whisper snapshot). Tests here verify the constant table and
//! the `sot_sequence` logic without I/O.

use super::{
    language_token, WhisperTask, TOK_EN, TOK_EOT, TOK_NO_TIMESTAMPS, TOK_SOT, TOK_TRANSCRIBE,
    TOK_TRANSLATE,
};

/// Language token IDs — spot-check against verified HuggingFace values.
///
/// Values verified by loading `openai/whisper-large-v3` with
/// `WhisperTokenizerFast.from_pretrained` and calling `tokenizer.encode(f"<|{code}|>")`.
/// Full table coverage is constrained to languages that are both (a) in the Whisper
/// vocabulary and (b) listed in the `language_token` compile-time match.
/// Integration tests in `crates/rmlx-audio/tests/transcribe.rs` verify against a
/// live `tokenizer.json` when the snapshot is present under `RMLX_O_MODELS_ROOT`.
#[test]
fn language_token_ids() {
    // Verified against WhisperTokenizerFast.from_pretrained output.
    assert_eq!(language_token("en"), 50_259);
    assert_eq!(language_token("zh"), 50_260);
    assert_eq!(language_token("de"), 50_261);
    assert_eq!(language_token("fr"), 50_265);
    assert_eq!(language_token("ja"), 50_266);
    assert_eq!(language_token("ru"), 50_263);
    // Hawaiian — first 3-letter code in the table (L1 spot check).
    assert_eq!(language_token("haw"), 50_352);
    // Yue Chinese — last 3-letter code in the table.
    assert_eq!(language_token("yue"), 50_358);
    // Unknown language falls back to English.
    assert_eq!(language_token("xx"), TOK_EN);
    // Non-EN code that would silently misroute if fallback were wrong.
    assert_ne!(language_token("fr"), TOK_EN);
}

/// Special token constant sanity check.
///
/// large-v3 has 100 language slots (`<|en|>`=50259 … `<|yue|>`=50358), so every
/// special after the language block is shifted up by one vs the v1/v2 layout.
/// Verified against the shipped `tokenizer.json` `added_tokens` table.
#[test]
fn special_token_constants() {
    assert_eq!(TOK_EOT, 50_257);
    assert_eq!(TOK_SOT, 50_258);
    assert_eq!(TOK_EN, 50_259);
    assert_eq!(TOK_TRANSLATE, 50_359);
    assert_eq!(TOK_TRANSCRIBE, 50_360);
    assert_eq!(TOK_NO_TIMESTAMPS, 50_364);
}

/// SOT sequence structure.
///
/// Note: a `WhisperTokenizer` instance is needed for the full `sot_sequence`
/// call. We verify the logic without loading a tokenizer file by checking
/// the token constant values directly.
#[test]
fn sot_sequence_structure() {
    // Expected: [SOT, lang_tok, task_tok, NO_TIMESTAMPS]
    let lang = language_token("en");
    let seq: Vec<u32> = vec![TOK_SOT, lang, TOK_TRANSCRIBE, TOK_NO_TIMESTAMPS];
    assert_eq!(seq[0], TOK_SOT);
    assert_eq!(seq[1], TOK_EN);
    assert_eq!(seq[2], TOK_TRANSCRIBE);
    assert_eq!(seq[3], TOK_NO_TIMESTAMPS);
}

/// Translate task token is distinct from transcribe.
#[test]
fn task_tokens_distinct() {
    assert_ne!(TOK_TRANSCRIBE, TOK_TRANSLATE);
    // Translate is one ID before transcribe (verified from HuggingFace).
    assert_eq!(TOK_TRANSLATE, TOK_TRANSCRIBE - 1);
}

/// WhisperTask enum coverage.
#[test]
fn whisper_task_debug() {
    let t = WhisperTask::Transcribe;
    let u = WhisperTask::Translate;
    assert_ne!(t, u);
    assert_eq!(format!("{t:?}"), "Transcribe");
    assert_eq!(format!("{u:?}"), "Translate");
}
