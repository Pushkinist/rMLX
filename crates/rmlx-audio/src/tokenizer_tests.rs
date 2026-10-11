//! Tests for the Whisper tokenizer: the special-token ids, the SOT sequence
//! logic, and the vocabulary check at load on `tokenizer.json` files built in
//! the test.

use std::path::Path;

use super::{
    language_name, language_token, WhisperTask, WhisperTokenizer, N_VOCAB, TOK_EN, TOK_EOT,
    TOK_LANG_LAST, TOK_NOSPEECH, TOK_NO_TIMESTAMPS, TOK_SOT, TOK_SOT_LM, TOK_SOT_PREV,
    TOK_TIMESTAMP_BEGIN, TOK_TRANSCRIBE, TOK_TRANSLATE,
};

/// Language token IDs — spot-check against verified HuggingFace values.
///
/// Values verified by loading `openai/whisper-large-v3` with
/// `WhisperTokenizerFast.from_pretrained` and calling `tokenizer.encode(f"<|{code}|>")`.
/// `every_reference_language_maps_to_its_slot` covers all 100 codes.
#[test]
fn language_token_ids() {
    assert_eq!(language_token("en"), Some(50_259));
    assert_eq!(language_token("zh"), Some(50_260));
    assert_eq!(language_token("de"), Some(50_261));
    assert_eq!(language_token("fr"), Some(50_265));
    assert_eq!(language_token("ja"), Some(50_266));
    assert_eq!(language_token("ru"), Some(50_263));
    assert_eq!(language_token("haw"), Some(50_352));
    assert_eq!(language_token("yue"), Some(50_358));
}

/// A language Whisper does not know has no token: it never falls back to
/// English.
#[test]
fn an_unknown_language_has_no_token() {
    for unknown in ["xx", "", "auto", "klingon", "e", "eng", "en-US", " en"] {
        assert_eq!(language_token(unknown), None, "{unknown:?}");
    }
}

/// The code and the English name give the same token, in any letter case, as
/// openai-whisper `get_tokenizer` does.
#[test]
fn a_code_and_a_name_in_any_case_give_one_token() {
    for spelling in ["fr", "FR", "Fr", "french", "French", "FRENCH"] {
        assert_eq!(language_token(spelling), Some(50_265), "{spelling}");
    }
    assert_eq!(language_token("Haitian Creole"), language_token("ht"));
}

/// The names openai-whisper `TO_LANGUAGE_CODE` adds to the names of `LANGUAGES`.
#[test]
fn the_reference_aliases_give_their_code_token() {
    for (alias, code) in [
        ("burmese", "my"),
        ("valencian", "ca"),
        ("flemish", "nl"),
        ("haitian", "ht"),
        ("letzeburgesch", "lb"),
        ("pushto", "ps"),
        ("panjabi", "pa"),
        ("moldavian", "ro"),
        ("moldovan", "ro"),
        ("sinhalese", "si"),
        ("castilian", "es"),
        ("mandarin", "zh"),
    ] {
        assert!(language_token(code).is_some(), "{code}");
        assert_eq!(language_token(alias), language_token(code), "{alias}");
    }
}

/// The name of a language token is the English name of the reference, in
/// lower case, for every one of the 100 slots, and no other id has one.
#[test]
fn every_language_token_has_the_reference_name() {
    let names: Vec<&str> = REFERENCE_NAMES.split(',').collect();
    assert_eq!(names.len(), 100);
    for (i, name) in names.iter().enumerate() {
        assert_eq!(language_name(TOK_EN + i as u32), Some(*name), "slot {i}");
    }
    assert_eq!(language_name(TOK_EN - 1), None);
    assert_eq!(language_name(TOK_LANG_LAST + 1), None);
    assert_eq!(language_name(0), None);
}

/// The name a token reports maps back to the token, so a detected language
/// and a named one are one form.
#[test]
fn a_reported_name_is_a_name_the_route_accepts() {
    for token in TOK_EN..=TOK_LANG_LAST {
        let name = language_name(token).expect("a language token has a name");
        assert_eq!(language_token(name), Some(token), "{name}");
    }
}

/// Every id the decode uses, against the `added_tokens` of the
/// `openai/whisper-large-v3` `tokenizer.json`.
#[test]
fn special_token_constants() {
    assert_eq!(TOK_EOT, 50_257);
    assert_eq!(TOK_SOT, 50_258);
    assert_eq!(TOK_EN, 50_259);
    assert_eq!(TOK_LANG_LAST, 50_358);
    assert_eq!(TOK_TRANSLATE, 50_359);
    assert_eq!(TOK_TRANSCRIBE, 50_360);
    assert_eq!(TOK_SOT_LM, 50_361);
    assert_eq!(TOK_SOT_PREV, 50_362);
    assert_eq!(TOK_NOSPEECH, 50_363);
    assert_eq!(TOK_NO_TIMESTAMPS, 50_364);
    assert_eq!(TOK_TIMESTAMP_BEGIN, 50_365);
    assert_eq!(N_VOCAB, 51_866);
}

/// SOT sequence structure.
///
/// Note: a `WhisperTokenizer` instance is needed for the full
/// `sot_sequence_from_tok` call. We verify the logic without loading a
/// tokenizer file by checking the token constant values directly.
#[test]
fn sot_sequence_structure() {
    // Expected: [SOT, lang_tok, task_tok, NO_TIMESTAMPS]
    let lang = language_token("en").unwrap_or_default();
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

// ── Vocabulary check at load ─────────────────────────────────────────────────

/// The language codes of openai-whisper `whisper/tokenizer.py` `LANGUAGES`, in
/// its order (openai/whisper@86098128). This is the oracle the tests below
/// build a vocabulary from; the code under test holds its own list.
const REFERENCE_LANGUAGES: &str = "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he \
    uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu \
    is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk \
    nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue";

/// The English names of openai-whisper `whisper/tokenizer.py` `LANGUAGES`
/// (openai/whisper@86098128), in the order of `REFERENCE_LANGUAGES`.
const REFERENCE_NAMES: &str =
    "english,chinese,german,spanish,russian,korean,french,japanese,portuguese,turkish,\
    polish,catalan,dutch,arabic,swedish,italian,indonesian,hindi,finnish,vietnamese,\
    hebrew,ukrainian,greek,malay,czech,romanian,danish,hungarian,tamil,norwegian,thai,\
    urdu,croatian,bulgarian,lithuanian,latin,maori,malayalam,welsh,slovak,telugu,\
    persian,latvian,bengali,serbian,azerbaijani,slovenian,kannada,estonian,macedonian,\
    breton,basque,icelandic,armenian,nepali,mongolian,bosnian,kazakh,albanian,swahili,\
    galician,marathi,punjabi,sinhala,khmer,shona,yoruba,somali,afrikaans,occitan,\
    georgian,belarusian,tajik,sindhi,gujarati,amharic,yiddish,lao,uzbek,faroese,\
    haitian creole,pashto,turkmen,nynorsk,maltese,sanskrit,luxembourgish,myanmar,\
    tibetan,tagalog,malagasy,assamese,tatar,hawaiian,lingala,hausa,bashkir,javanese,\
    sundanese,cantonese";

/// The special tokens openai-whisper `get_encoding` (`whisper/tokenizer.py`)
/// adds after a BPE vocabulary of `base` tokens, each with the next id: end of
/// text, start of transcript, the first `num_languages` languages, the task and
/// control tokens, then the 1501 timestamps `<|0.00|>` to `<|30.00|>`.
/// Multilingual vocabularies have `base` 50257 (99 languages for v1 and v2,
/// 100 for v3); English-only ones have `base` 50256 and 99 languages.
fn reference_specials(base: u32, num_languages: usize) -> Vec<(String, u32)> {
    let mut names = vec![
        "<|endoftext|>".to_owned(),
        "<|startoftranscript|>".to_owned(),
    ];
    names.extend(
        REFERENCE_LANGUAGES
            .split_whitespace()
            .take(num_languages)
            .map(|code| format!("<|{code}|>")),
    );
    names.extend(
        [
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
        ]
        .map(str::to_owned),
    );
    names.extend((0..1501).map(|i| format!("<|{:.2}|>", f64::from(i) * 0.02)));
    names.into_iter().zip(base..).collect()
}

/// Write a `tokenizer.json` with a word-level vocabulary of `base` plain tokens
/// and `specials` as its added tokens, and return the directory. The
/// `tokenizers` crate gives added tokens the ids after the vocabulary in file
/// order and only warns on their `id` field, so they are written in id order.
fn write_tokenizer(base: u32, specials: &[(String, u32)]) -> tempfile::TempDir {
    let vocab: serde_json::Map<String, serde_json::Value> =
        (0..base).map(|i| (format!("t{i}"), i.into())).collect();
    let mut specials = specials.to_vec();
    specials.sort_by_key(|&(_, id)| id);
    let added: Vec<serde_json::Value> = specials
        .iter()
        .map(|(content, id)| {
            serde_json::json!({
                "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true,
            })
        })
        .collect();
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "t0" },
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), json.to_string()).unwrap();
    dir
}

/// The text of the error `WhisperTokenizer::from_path` refuses `dir` with. It
/// must name the file and must not hold the directory's path.
fn refusal(dir: &tempfile::TempDir) -> String {
    let err = WhisperTokenizer::from_path(dir.path())
        .expect_err("the tokenizer must be refused")
        .to_string();
    assert!(
        err.contains("tokenizer.json"),
        "the error names no file: {err}"
    );
    let path = dir.path().display().to_string();
    assert!(!err.contains(&path), "the error holds the path: {err}");
    err
}

fn assert_names(err: &str, words: &[&str]) {
    for word in words {
        assert!(err.contains(word), "the error does not name {word}: {err}");
    }
}

#[test]
fn the_reference_large_v3_layout_loads() {
    let dir = write_tokenizer(50_257, &reference_specials(50_257, 100));
    WhisperTokenizer::from_path(dir.path()).expect("the large-v3 layout must load");
}

/// The server sends a load error to the client: it names the file, never the
/// directory the operator configured.
#[test]
fn a_missing_tokenizer_names_the_file_and_no_directory() {
    let dir = tempfile::tempdir().unwrap();
    let err = refusal(&dir);
    assert_names(&err, &["No such file"]);
}

/// v1 and v2 have 99 languages, so `<|yue|>` is not there and every later
/// special sits one id lower.
#[test]
fn a_large_v2_layout_is_refused_at_the_missing_language() {
    let dir = write_tokenizer(50_257, &reference_specials(50_257, 99));
    assert_names(&refusal(&dir), &["<|yue|>", "50358"]);
}

/// English-only vocabularies start their specials one id lower.
#[test]
fn an_english_only_layout_is_refused_at_end_of_text() {
    let dir = write_tokenizer(50_256, &reference_specials(50_256, 99));
    assert_names(&refusal(&dir), &["<|endoftext|>", "50256", "50257"]);
}

#[test]
fn one_moved_special_id_is_refused_naming_it() {
    let mut specials = reference_specials(50_257, 100);
    for (name, id) in &mut specials {
        match name.as_str() {
            "<|nospeech|>" => *id = 50_364,
            "<|notimestamps|>" => *id = 50_363,
            _ => {}
        }
    }
    let dir = write_tokenizer(50_257, &specials);
    assert_names(&refusal(&dir), &["<|nospeech|>", "50364", "50363"]);
}

#[test]
fn one_missing_timestamp_is_refused_naming_it() {
    let mut specials = reference_specials(50_257, 100);
    for (name, _) in &mut specials {
        if name == "<|15.00|>" {
            *name = "<|15.000|>".to_owned();
        }
    }
    let dir = write_tokenizer(50_257, &specials);
    assert_names(&refusal(&dir), &["<|15.00|>", "51115"]);
}

/// The first, the last and an odd timestamp are each checked, not only the
/// one in the middle.
#[test]
fn each_end_of_the_timestamp_range_is_refused_when_missing() {
    for (missing, id) in [
        ("<|0.02|>", "50366"),
        ("<|29.98|>", "51864"),
        ("<|30.00|>", "51865"),
    ] {
        let mut specials = reference_specials(50_257, 100);
        for (name, _) in &mut specials {
            if name == missing {
                name.push('x');
            }
        }
        let dir = write_tokenizer(50_257, &specials);
        assert_names(&refusal(&dir), &[missing, id]);
    }
}

#[test]
fn a_vocabulary_with_one_more_token_is_refused_naming_its_size() {
    let mut specials = reference_specials(50_257, 100);
    specials.push(("<|extra|>".to_owned(), 51_866));
    let dir = write_tokenizer(50_257, &specials);
    assert_names(&refusal(&dir), &["51867", "51866"]);
}

/// Every language code the decode accepts maps to its slot in the reference
/// order: `<|en|>` is 50259 and each next code is one id higher.
#[test]
fn every_reference_language_maps_to_its_slot() {
    for (i, code) in REFERENCE_LANGUAGES.split_whitespace().enumerate() {
        assert_eq!(language_token(code), Some(TOK_EN + i as u32), "{code}");
    }
}

/// The local large-v3 tokenizer passes the check. Runs on the CPU when
/// `RMLX_O_MODELS_ROOT` holds `openai__whisper-large-v3-tokenizer`.
#[test]
fn the_local_large_v3_tokenizer_loads() {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!("SKIP the_local_large_v3_tokenizer_loads: RMLX_O_MODELS_ROOT is unset");
        return;
    };
    let dir = Path::new(&root).join("openai__whisper-large-v3-tokenizer");
    if !dir.join("tokenizer.json").exists() {
        eprintln!(
            "SKIP the_local_large_v3_tokenizer_loads: no openai__whisper-large-v3-tokenizer under the root"
        );
        return;
    }
    WhisperTokenizer::from_path(&dir).expect("the large-v3 tokenizer must load");
}
