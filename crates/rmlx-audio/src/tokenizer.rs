//! Whisper BPE tokenizer — pure Rust, no Python tiktoken dependency.
//!
//! ## Approach
//!
//! The `tokenizers` crate (already a workspace dep — used by rmlx-models for
//! LLM BPE tokenization) can load a Whisper-compatible tokenizer JSON directly
//! from the Hugging Face `tokenizer.json` format. However, the `mlx-community__
//! whisper-large-v3-mlx` snapshot ships only `weights.npz` and `config.json`
//! — no tokenizer files.
//!
//! ## Special token ids (large-v3 vocabulary only)
//!
//! The decode uses the ids of the large-v3 vocabulary: 51 866 tokens, 50 257
//! BPE tokens and 1 609 special tokens, 100 of them languages. v1, v2 and the
//! smaller multilingual models have 99 languages, so every special after
//! `<|su|>` sits one id lower; the English-only models start one id lower
//! still. [`WhisperTokenizer::from_path`] refuses a `tokenizer.json` whose
//! special tokens or size differ, and `WhisperConfig::from_json` refuses an
//! `n_vocab` other than [`N_VOCAB`].
//!
//! | Token | ID |
//! |---|---|
//! | `<|endoftext|>` (eot) | 50 257 |
//! | `<|startoftranscript|>` (sot) | 50 258 |
//! | `<|en|>` | 50 259 |
//! | `<|yue|>` (last language) | 50 358 |
//! | `<|translate|>` | 50 359 |
//! | `<|transcribe|>` | 50 360 |
//! | `<|startoflm|>` | 50 361 |
//! | `<|startofprev|>` | 50 362 |
//! | `<|nospeech|>` | 50 363 |
//! | `<|notimestamps|>` | 50 364 |
//! | `<|0.00|>` (timestamp_begin) | 50 365 |
//! | `<|30.00|>` (last timestamp) | 51 865 |
//!
//! ## Tokenizer loading
//!
//! The tokenizer loads from a directory that holds `tokenizer.json`. The
//! `tokenizers` crate does the BPE encode and decode; this module adds the
//! Whisper special-token protocol on top.

use std::path::Path;

use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, instrument};

// ── Special token ids (large-v3 vocabulary) ──────────────────────────────────
//
// openai-whisper `get_encoding` (`whisper/tokenizer.py`) gives each special
// token the next id after the BPE vocabulary, in the order below.

/// End of text (`<|endoftext|>`): the first id after the 50 257 BPE tokens.
pub const TOK_EOT: u32 = 50_257;
/// Start of transcript (`<|startoftranscript|>`).
pub const TOK_SOT: u32 = TOK_EOT + 1;
/// English language token (`<|en|>`), the first of the [`LANGUAGES`] tokens.
pub const TOK_EN: u32 = TOK_SOT + 1;
/// Last language token (`<|yue|>`).
pub const TOK_LANG_LAST: u32 = TOK_EN + N_LANGUAGES - 1;
/// Translate task token (`<|translate|>`).
pub const TOK_TRANSLATE: u32 = TOK_LANG_LAST + 1;
/// Transcribe task token (`<|transcribe|>`).
pub const TOK_TRANSCRIBE: u32 = TOK_TRANSLATE + 1;
/// Start-of-LM token (`<|startoflm|>`).
pub const TOK_SOT_LM: u32 = TOK_TRANSCRIBE + 1;
/// Start-of-previous-context token (`<|startofprev|>`), for prompt conditioning.
pub const TOK_SOT_PREV: u32 = TOK_SOT_LM + 1;
/// No-speech token (`<|nospeech|>`).
pub const TOK_NOSPEECH: u32 = TOK_SOT_PREV + 1;
/// No-timestamps token (`<|notimestamps|>`).
pub const TOK_NO_TIMESTAMPS: u32 = TOK_NOSPEECH + 1;
/// First timestamp token `<|0.00|>`; one token per 0.02 s step.
pub const TOK_TIMESTAMP_BEGIN: u32 = TOK_NO_TIMESTAMPS + 1;
/// Timestamp tokens, `<|0.00|>` to `<|30.00|>`.
pub const N_TIMESTAMPS: u32 = 1501;
/// Size of the vocabulary these ids are for: the `n_vocab` a `config.json`
/// must state and the token count a `tokenizer.json` must hold.
pub const N_VOCAB: usize = (TOK_TIMESTAMP_BEGIN + N_TIMESTAMPS) as usize;

const N_LANGUAGES: u32 = 100;

/// Each language as (ISO code, lower-case English name), in vocabulary order
/// (openai-whisper `LANGUAGES`): `<|en|>` is [`TOK_EN`] and each next entry has
/// the next id. The loader accepts the large-v3 vocabulary only, so this is
/// the one table.
pub const LANGUAGES: [(&str, &str); N_LANGUAGES as usize] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// Names that openai-whisper `TO_LANGUAGE_CODE` takes beside those of
/// [`LANGUAGES`], with the code each stands for.
const LANGUAGE_ALIASES: [(&str, &str); 12] = [
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
];

/// The language token of a code (`fr`) or an English name (`french`), in any
/// letter case, as openai-whisper `get_tokenizer` reads a language. `None`
/// for any other text.
#[must_use]
pub fn language_token(language: &str) -> Option<u32> {
    let wanted = language.to_ascii_lowercase();
    let wanted = LANGUAGE_ALIASES
        .iter()
        .find_map(|&(alias, code)| (alias == wanted).then_some(code))
        .unwrap_or(&wanted);
    (TOK_EN..)
        .zip(LANGUAGES)
        .find_map(|(id, (code, name))| (code == wanted || name == wanted).then_some(id))
}

/// The lower-case English name of a language token, or `None` for an id that is
/// not one of the [`LANGUAGES`] tokens.
#[must_use]
pub fn language_name(token: u32) -> Option<&'static str> {
    language_entry(token).map(|(_, name)| name)
}

/// The code of a language token (`fr`), or `None` for an id that is not one of
/// the [`LANGUAGES`] tokens.
#[must_use]
pub(crate) fn language_code(token: u32) -> Option<&'static str> {
    language_entry(token).map(|(code, _)| code)
}

fn language_entry(token: u32) -> Option<(&'static str, &'static str)> {
    let slot = usize::try_from(token.checked_sub(TOK_EN)?).ok()?;
    LANGUAGES.get(slot).copied()
}

/// Every special token the ids above name, with its id: openai-whisper
/// `get_encoding`'s list for the large-v3 vocabulary.
fn special_tokens() -> Vec<(String, u32)> {
    let mut tokens = vec![
        ("<|endoftext|>".to_owned(), TOK_EOT),
        ("<|startoftranscript|>".to_owned(), TOK_SOT),
    ];
    tokens.extend(
        (TOK_EN..)
            .zip(LANGUAGES)
            .map(|(id, (code, _))| (format!("<|{code}|>"), id)),
    );
    tokens.extend(
        [
            ("<|translate|>", TOK_TRANSLATE),
            ("<|transcribe|>", TOK_TRANSCRIBE),
            ("<|startoflm|>", TOK_SOT_LM),
            ("<|startofprev|>", TOK_SOT_PREV),
            ("<|nospeech|>", TOK_NOSPEECH),
            ("<|notimestamps|>", TOK_NO_TIMESTAMPS),
        ]
        .map(|(token, id)| (token.to_owned(), id)),
    );
    tokens.extend((0..N_TIMESTAMPS).map(|i| {
        (
            format!("<|{:.2}|>", f64::from(i) * 0.02),
            TOK_TIMESTAMP_BEGIN + i,
        )
    }));
    tokens
}

/// Refuse a tokenizer whose special-token ids or size differ from the ids the
/// decode uses.
fn check_vocabulary(tokenizer: &Tokenizer) -> Result<(), TokenizerError> {
    for (token, expected) in special_tokens() {
        let found = tokenizer.token_to_id(&token);
        if found != Some(expected) {
            return Err(TokenizerError::SpecialToken {
                token,
                found: found.map_or_else(|| "absent".to_owned(), |id| format!("id {id}")),
                expected,
            });
        }
    }
    let size = tokenizer.get_vocab_size(true);
    if size != N_VOCAB {
        return Err(TokenizerError::VocabSize {
            found: size,
            expected: N_VOCAB,
        });
    }
    Ok(())
}

// ── WhisperTokenizer ──────────────────────────────────────────────────────────

/// Whisper tokenizer errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// Failed to load the tokenizer from the path.
    #[error("tokenizer load error: {0}")]
    Load(String),
    /// Encode failure.
    #[error("tokenizer encode error: {0}")]
    Encode(String),
    /// Decode failure.
    #[error("tokenizer decode error: {0}")]
    Decode(String),
    /// A prompt holds the text of a special token, for example
    /// `<|endoftext|>`, which would encode as a control token of the decode.
    #[error("the prompt holds the text of a special token (<|...|>)")]
    SpecialTokenText,
    /// A special token is absent or has another id than the decode uses.
    #[error(
        "tokenizer.json: {token} is {found}, the decode expects id {expected}; \
         only the Whisper large-v3 vocabulary is supported"
    )]
    SpecialToken {
        /// The token, for example `<|transcribe|>`.
        token: String,
        /// `id N`, or `absent`.
        found: String,
        /// The id the decode uses.
        expected: u32,
    },
    /// The vocabulary has another size than the decode's.
    #[error(
        "tokenizer.json holds {found} tokens, the decode expects {expected}; \
         only the Whisper large-v3 vocabulary is supported"
    )]
    VocabSize {
        /// Tokens in the file, added tokens included.
        found: usize,
        /// [`N_VOCAB`].
        expected: usize,
    },
}

/// The most bytes of a prompt that are encoded: its end.
pub const PROMPT_TAIL_BYTES: usize = 16 * 1024;

/// The last [`PROMPT_TAIL_BYTES`] bytes of `text`, cut at a character boundary.
fn prompt_tail(text: &str) -> &str {
    let start = text.ceil_char_boundary(text.len().saturating_sub(PROMPT_TAIL_BYTES));
    text.get(start..).unwrap_or_default()
}

/// Whisper BPE tokenizer wrapper.
///
/// Wraps the `tokenizers` crate `Tokenizer` with Whisper-specific special
/// token handling. Load from a path that contains a `tokenizer.json`.
pub struct WhisperTokenizer {
    inner: Tokenizer,
}

impl std::fmt::Debug for WhisperTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperTokenizer").finish_non_exhaustive()
    }
}

impl WhisperTokenizer {
    /// Load tokenizer from a directory containing `tokenizer.json`.
    ///
    /// The Whisper snapshot from `mlx-community` does not ship a tokenizer;
    /// use the companion `openai/whisper-large-v3` HuggingFace repo path, or
    /// any local directory that contains a Whisper-compatible `tokenizer.json`.
    /// A tokenizer of another vocabulary than large-v3's is refused. An error
    /// names `tokenizer.json`, not its directory: the server sends it to the
    /// client.
    #[instrument(skip(path), fields(path = %path.as_ref().display()), level = "debug")]
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let tok_path = path.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&tok_path)
            .map_err(|e| TokenizerError::Load(format!("tokenizer.json: {e}")))?;
        check_vocabulary(&inner)?;
        debug!("WhisperTokenizer loaded");
        Ok(Self { inner })
    }

    /// Encode plain text into token IDs (no special tokens prepended).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Encode the text of a decoder prompt: a space and the last
    /// [`PROMPT_TAIL_BYTES`] bytes of `prompt.trim()` (Unicode White_Space
    /// characters are trimmed). The decoder keeps the last tokens of a prompt
    /// only, and this tail holds far more of them, so a longer text is never
    /// encoded whole. An empty or white-space text gives no token. The text of
    /// a special token (`<|endoftext|>`, `<|en|>`, `<|0.00|>`, ...) in the
    /// tail is an error: the tokenizer would turn it into a control token. The
    /// text before the tail is not read.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, TokenizerError> {
        let text = prompt_tail(prompt.trim());
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let ids = self.encode(&format!(" {text}"))?;
        if ids.iter().any(|&id| id >= TOK_EOT) {
            return Err(TokenizerError::SpecialTokenText);
        }
        Ok(ids)
    }

    /// Derive the Whisper non-speech / special suppression token set from the
    /// loaded tokenizer.
    ///
    /// This is a faithful port of openai-whisper's `Tokenizer.non_speech_tokens`
    /// plus `_get_suppress_tokens`: the non-speech set is computed by BPE-encoding
    /// a fixed list of punctuation / symbol strings and keeping the ids that
    /// encode to a single token (general — no hardcoded magic id list). On top of
    /// that we always suppress the structural specials (`SOT`, `TRANSCRIBE`,
    /// `TRANSLATE`, `NOSPEECH`) so they can never be emitted as content.
    ///
    /// The returned set is sorted+deduplicated and intended to be applied as a
    /// logit mask at every decode step.
    pub fn suppress_tokens(&self) -> Vec<u32> {
        // openai-whisper symbol list (`whisper/tokenizer.py::non_speech_tokens`).
        // Each char of the first group is a standalone symbol; the second group
        // is space-separated multi-char symbols.
        const SYMBOL_CHARS: &str = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』";
        const SYMBOL_WORDS: &str =
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪";
        // "Miscellaneous" musical symbols: each is forced in even when it
        // BPE-splits into multiple tokens (matches the reference).
        const MISC: &str = "♩♪♫♬♭♮♯";

        let mut set: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

        // Reference seeds the set with the first token of " -" and " '".
        for seed in [" -", " '"] {
            if let Ok(ids) = self.encode(seed) {
                if let Some(&first) = ids.first() {
                    set.insert(first);
                }
            }
        }

        let misc: std::collections::BTreeSet<char> = MISC.chars().collect();
        let symbols: Vec<String> = SYMBOL_CHARS
            .chars()
            .map(|c| c.to_string())
            .chain(SYMBOL_WORDS.split(' ').map(str::to_owned))
            .chain(MISC.chars().map(|c| c.to_string()))
            .collect();

        for symbol in &symbols {
            let is_misc = symbol.chars().count() == 1
                && symbol.chars().next().is_some_and(|c| misc.contains(&c));
            for variant in [symbol.clone(), format!(" {symbol}")] {
                if let Ok(ids) = self.encode(&variant) {
                    if (ids.len() == 1 || is_misc) && !ids.is_empty() {
                        if let Some(&first) = ids.first() {
                            set.insert(first);
                        }
                    }
                }
            }
        }

        // Structural specials that must never appear in content.
        for t in [TOK_SOT, TOK_TRANSCRIBE, TOK_TRANSLATE, TOK_NOSPEECH] {
            set.insert(t);
        }

        set.into_iter().collect()
    }

    /// Decode token IDs back to text, skipping tokens ≥ `TOK_EOT` (specials).
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        // Filter timestamp and special tokens before decode.
        let filtered: Vec<u32> = tokens.iter().copied().filter(|&t| t < TOK_EOT).collect();
        self.inner
            .decode(&filtered, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Build the initial decoder token sequence for a language token (from
    /// [`language_token`] or language detection) and a task.
    ///
    /// Sequence: `[SOT, lang_token, task_token, no_timestamps]`; the last
    /// token is present only without timestamps.
    pub fn sot_sequence_from_tok(
        &self,
        lang_tok: u32,
        task: WhisperTask,
        with_timestamps: bool,
    ) -> Vec<u32> {
        let task_tok = match task {
            WhisperTask::Transcribe => TOK_TRANSCRIBE,
            WhisperTask::Translate => TOK_TRANSLATE,
        };
        let mut seq = vec![TOK_SOT, lang_tok, task_tok];
        if !with_timestamps {
            seq.push(TOK_NO_TIMESTAMPS);
        }
        seq
    }
}

/// Whisper decoder task.
#[allow(
    clippy::exhaustive_enums,
    reason = "Whisper only has two tasks; this enum is intentionally closed"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTask {
    /// Transcribe: output is in the same language as the audio.
    Transcribe,
    /// Translate: output is in English regardless of audio language.
    Translate,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;
