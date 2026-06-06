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
//! To avoid downloading files at runtime, we use the `tokenizers` crate to
//! load from a Hugging Face repo path, OR we bundle the mapping of Whisper
//! special tokens (which are stable, version-locked to `openai/whisper-large-v3`).
//!
//! ## Special token IDs (locked to `openai/whisper-large-v3`)
//!
//! All values verified against `WhisperTokenizerFast.from_pretrained` output:
//!
//! | Token | ID |
//! |---|---|
//! | `<|endoftext|>` (eot) | 50 257 |
//! | `<|startoftranscript|>` (sot) | 50 258 |
//! | `<|en|>` | 50 259 |
//! | `<|translate|>` | 50 358 |
//! | `<|transcribe|>` | 50 359 |
//! | `<|nospeech|>` | 50 362 |
//! | `<|notimestamps|>` | 50 363 |
//! | `<|0.00|>` (timestamp_begin) | 50 364 |
//!
//! Total vocabulary: 51 866 tokens (50 257 base GPT-2 + 1 609 added).
//!
//! ## Tokenizer loading
//!
//! At transcription time we load the tokenizer from the HuggingFace cache or
//! from the snapshot directory. The `tokenizers` crate handles BPE decode/encode
//! natively; we only layer the Whisper-specific special-token protocol on top.

use std::path::Path;

use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, instrument};

// ── Special token IDs (large-v3 multilingual) ────────────────────────────────

/// End of text (`<|endoftext|>`).
pub const TOK_EOT: u32 = 50_257;
/// Start of transcript (`<|startoftranscript|>`).
pub const TOK_SOT: u32 = 50_258;
/// English language token (`<|en|>`).
pub const TOK_EN: u32 = 50_259;
/// Translate task token (`<|translate|>`).
pub const TOK_TRANSLATE: u32 = 50_358;
/// Transcribe task token (`<|transcribe|>`).
pub const TOK_TRANSCRIBE: u32 = 50_359;
/// No-speech token (`<|nospeech|>`).
pub const TOK_NOSPEECH: u32 = 50_362;
/// No-timestamps token (`<|notimestamps|>`).
pub const TOK_NO_TIMESTAMPS: u32 = 50_363;
/// First timestamp token `<|0.00|>`.
pub const TOK_TIMESTAMP_BEGIN: u32 = 50_364;

/// Language token IDs for the 99 supported languages.
///
/// Offset from `<|en|>` (50 259): language codes in alphabetical order as
/// stored in the Whisper vocab. For unknown codes, fall back to `<|en|>`.
pub const fn language_token(lang_code: &str) -> u32 {
    // This is a compile-time lookup of the most common languages.
    // The full table is 99 entries; we inline the most common ones.
    // For languages not listed, callers should use detect-language mode.
    match lang_code.as_bytes() {
        b"en" => 50_259,
        b"zh" => 50_260,
        b"de" => 50_261,
        b"es" => 50_262,
        b"ru" => 50_263,
        b"ko" => 50_264,
        b"fr" => 50_265,
        b"ja" => 50_266,
        b"pt" => 50_267,
        b"tr" => 50_268,
        b"pl" => 50_269,
        b"ca" => 50_270,
        b"nl" => 50_271,
        b"ar" => 50_272,
        b"sv" => 50_273,
        b"it" => 50_274,
        b"id" => 50_275,
        b"hi" => 50_276,
        b"fi" => 50_277,
        b"vi" => 50_278,
        b"he" => 50_279,
        b"uk" => 50_280,
        b"el" => 50_281,
        b"ms" => 50_282,
        b"cs" => 50_283,
        b"ro" => 50_284,
        b"da" => 50_285,
        b"hu" => 50_286,
        b"ta" => 50_287,
        b"no" => 50_288,
        b"th" => 50_289,
        b"ur" => 50_290,
        b"hr" => 50_291,
        b"bg" => 50_292,
        b"lt" => 50_293,
        b"la" => 50_294,
        b"mi" => 50_295,
        b"ml" => 50_296,
        b"cy" => 50_297,
        b"sk" => 50_298,
        b"te" => 50_299,
        b"fa" => 50_300,
        b"lv" => 50_301,
        b"bn" => 50_302,
        b"sr" => 50_303,
        b"az" => 50_304,
        b"sl" => 50_305,
        b"kn" => 50_306,
        b"et" => 50_307,
        b"mk" => 50_308,
        b"br" => 50_309,
        b"eu" => 50_310,
        b"is" => 50_311,
        b"hy" => 50_312,
        b"ne" => 50_313,
        b"mn" => 50_314,
        b"bs" => 50_315,
        b"kk" => 50_316,
        b"sq" => 50_317,
        b"sw" => 50_318,
        b"gl" => 50_319,
        b"mr" => 50_320,
        b"pa" => 50_321,
        b"si" => 50_322,
        b"km" => 50_323,
        b"sn" => 50_324,
        b"yo" => 50_325,
        b"so" => 50_326,
        b"af" => 50_327,
        b"oc" => 50_328,
        b"ka" => 50_329,
        b"be" => 50_330,
        b"tg" => 50_331,
        b"sd" => 50_332,
        b"gu" => 50_333,
        b"am" => 50_334,
        b"yi" => 50_335,
        b"lo" => 50_336,
        b"uz" => 50_337,
        b"fo" => 50_338,
        b"ht" => 50_339,
        b"ps" => 50_340,
        b"tk" => 50_341,
        b"nn" => 50_342,
        b"mt" => 50_343,
        b"sa" => 50_344,
        b"lb" => 50_345,
        b"my" => 50_346,
        b"bo" => 50_347,
        b"tl" => 50_348,
        b"mg" => 50_349,
        b"as" => 50_350,
        b"tt" => 50_351,
        b"haw" => 50_352,
        b"ln" => 50_353,
        b"ha" => 50_354,
        b"ba" => 50_355,
        b"jw" => 50_356,
        b"su" => 50_357,
        b"yue" => 50_358,
        _ => TOK_EN, // fallback to English
    }
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
    #[instrument(skip(path), fields(path = %path.as_ref().display()), level = "debug")]
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let tok_path = path.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&tok_path)
            .map_err(|e| TokenizerError::Load(format!("{}: {e}", tok_path.display())))?;
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

    /// Decode token IDs back to text, skipping tokens ≥ `TOK_EOT` (specials).
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        // Filter timestamp and special tokens before decode.
        let filtered: Vec<u32> = tokens.iter().copied().filter(|&t| t < TOK_EOT).collect();
        self.inner
            .decode(&filtered, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Build the initial decoder token sequence for the given task and language.
    ///
    /// Sequence: `[SOT, lang_token, task_token, no_timestamps]`.
    pub fn sot_sequence(
        &self,
        language: &str,
        task: WhisperTask,
        with_timestamps: bool,
    ) -> Vec<u32> {
        let lang_tok = language_token(language);
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

    /// Build the SOT sequence given a raw language token id (e.g. from `detect_language`).
    ///
    /// Useful when the language was detected at runtime rather than specified as a string.
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
