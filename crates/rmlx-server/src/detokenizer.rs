//! Streaming detokenizer with UTF-8 token-healing (A10).
//!
//! ## Problem
//!
//! The streaming decode sites in `engine.rs` build each SSE delta with a
//! full-prefix decode + byte-diff:
//!
//! ```ignore
//! all_ids.push(new_id);
//! let full = tokenizer.decode(&all_ids, true)?; // HF tokenizers crate
//! let delta = &full[decoded_so_far.len()..]; // byte slice
//! decoded_so_far = full;
//! ```
//!
//! For byte-level BPE tokenizers (Qwen3.6 uses a `ByteLevel` decoder) the HF
//! `tokenizers` crate decodes via `String::from_utf8_lossy`. When a multi-byte
//! UTF-8 codepoint's bytes straddle two token ids, the intermediate
//! `decode(all_ids)` ends in the replacement char U+FFFD. The old code emitted
//! that `�` into the delta AND advanced `decoded_so_far` to the
//! `�`-terminated string. On the next token the real bytes appear, the
//! `starts_with(decoded_so_far)` prefix invariant breaks (`�` != real bytes),
//! the fallback emits an empty string, and the codepoint is lost forever.
//! Net effect: emoji / CJK / accented codepoints that split across tokens are
//! permanently corrupted to `�` in streamed output (reproduced on
//! `Qwen3.6-35B-A3B-8bit`: "日本語のテスト 🎉 café" → "日本語のテスト �
//! café").
//!
//! ## Fix (ported semantics, not code)
//!
//! Two reference detokenizers agree on the cure:
//!
//! - HF `tokenizers` 0.20.4 `DecodeStream::step` (`src/tokenizer/mod.rs`,
//!   `step_decode_stream`): emits a chunk only `if string.len() >
//! prefix.len() && !string.ends_with('�')` — otherwise returns `None` and
//!   keeps accumulating ids until the codepoint completes.
//! - mlx-lm `tokenizer_utils.py` `SPMStreamingDetokenizer._try_flush`
//!   (L135-148): `if not force and text.endswith("�"): return` — hold
//!   the unflushed bytes; only `finalize(force=True)` lossy-flushes the tail.
//!
//! rMLX cannot adopt HF `DecodeStream` directly: commit `1cc775f` documents a
//! cross-request panic from its leaking internal state, which is why the
//! engine uses the panic-safe full-prefix-decode model. So this module ports
//! only the *withholding semantic* into that model: keep full-prefix decode,
//! but never advance past — nor emit — a `�`-terminated boundary. The
//! withheld tail stays implicit in the gap between `decoded` and the true
//! full decode; the next token completes the codepoint and the diff picks it
//! up cleanly. `finalize()` flushes whatever remains, lossy, because true
//! end-of-stream is the only place a `�` is legitimate (genuinely truncated
//! generation).
//!
//! ## SPM leading space (per-arch)
//!
//! mlx-lm classifies tokenizers from `tokenizer.json`'s `decoder` field
//! (`_is_spm_decoder` / `_is_spm_decoder_no_space` / `_is_bpe_decoder`) and
//! only the strict `_is_spm_decoder` variant (decoder ends with
//! `Strip{content:" ",start:1}`) gets `trim_space=True`. The rMLX targets:
//!
//! - Gemma3 / Gemma4: decoder = `Sequence[Replace(▁→' '), ByteFallback,
//! Fuse]` → mlx-lm `_is_spm_decoder_no_space` → `trim_space=False`.
//! - Qwen3 / Qwen3.6: decoder = `ByteLevel` → no `▁`, no leading-space rule.
//!
//! The HF `tokenizers` crate's `Replace`/`ByteLevel` decoders already map
//! `▁`→space faithfully per token piece, and the engine decodes the full
//! growing prefix every step so the decoder always sees the true first token
//! at index 0. Empirically `decode(["Hello"…])` →
//! `"Hello"` (no spurious space) and `decode(["▁The"…])` → `" The"` (space is
//! genuine). Therefore **no leading-space stripping is applied here** for
//! either arch — doing so would corrupt Gemma output where a leading space is
//! real. Only the strict-SPM-with-Strip variant would need it, and no rMLX
//! target model uses that decoder. [`TokenizerKind::from_tokenizer_json`]
//! records the kind for documentation / test assertions; the stripping hook
//! exists for the strict-SPM case but is unreachable for current targets.

/// Tokenizer detokenization family, classified from `tokenizer.json`'s
/// `decoder` node exactly as mlx-lm `tokenizer_utils.py` does.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — four tokenizer detokenization families (SpmStrip/SpmNoStrip/ByteLevel/Other); adding a family requires updating classify() and all TokenizerKind match arms"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// `Sequence[Replace(▁→' '), ByteFallback, Fuse, Strip(" ",1,0)]`.
    /// mlx-lm `SPMStreamingDetokenizer` with `trim_space=True`: strip the
    /// leading space on the FIRST emitted segment only.
    SpmStrip,
    /// `Sequence[Replace(▁→' '), ByteFallback, Fuse]` (Gemma3 / Gemma4).
    /// mlx-lm `SPMStreamingDetokenizer(trim_space=False)`: never strip.
    SpmNoStrip,
    /// `{type: ByteLevel}` (Qwen3 / Qwen3.6). Byte-level BPE: no `▁`.
    ByteLevel,
    /// Anything else. Treated like `ByteLevel` for healing purposes
    /// (no leading-space rule); the UTF-8 withholding applies regardless.
    Other,
}

impl TokenizerKind {
    /// Classify from the parsed `tokenizer.json` root value. Mirrors mlx-lm
    /// `_is_spm_decoder` / `_is_spm_decoder_no_space` / `_is_bpe_decoder`.
    pub fn from_tokenizer_json(root: &serde_json::Value) -> Self {
        let Some(dec) = root.get("decoder") else {
            return Self::Other;
        };
        if dec.get("type").and_then(|t| t.as_str()) == Some("ByteLevel") {
            return Self::ByteLevel;
        }
        if dec.get("type").and_then(|t| t.as_str()) == Some("Sequence") {
            let subs = dec.get("decoders").and_then(|d| d.as_array());
            if let Some(subs) = subs {
                let types: Vec<&str> = subs
                    .iter()
                    .map(|s| s.get("type").and_then(|t| t.as_str()).unwrap_or(""))
                    .collect();
                // Replace(▁→' ') must be the first sub-decoder.
                let first_is_metaspace_replace = subs.first().is_some_and(|s| {
                    s.get("type").and_then(|t| t.as_str()) == Some("Replace")
                        && s.get("pattern")
                            .and_then(|p| p.get("String"))
                            .and_then(|v| v.as_str())
                            == Some("\u{2581}")
                        && s.get("content").and_then(|c| c.as_str()) == Some(" ")
                });
                if first_is_metaspace_replace {
                    if types == ["Replace", "ByteFallback", "Fuse", "Strip"] {
                        return Self::SpmStrip;
                    }
                    if types == ["Replace", "ByteFallback", "Fuse"] {
                        return Self::SpmNoStrip;
                    }
                }
            }
        }
        Self::Other
    }

    /// Whether the leading space on the first emitted segment must be
    /// stripped. Only the strict-SPM-with-`Strip` decoder needs this; the
    /// rMLX target models (Gemma `SpmNoStrip`, Qwen `ByteLevel`) do not.
    fn trim_first_space(self) -> bool {
        matches!(self, Self::SpmStrip)
    }
}

/// Incremental, UTF-8-safe detokenizer over the HF `tokenizers` crate.
///
/// Wraps the engine's full-prefix-decode + byte-diff with the
/// `!ends_with('�')` withholding guard ported from HF `DecodeStream` /
/// mlx-lm `SPMStreamingDetokenizer`. ASCII / already-complete codepoints are
/// emitted immediately and byte-identically to the pre-A10 path (an ASCII
/// byte is its own UTF-8 boundary, so the guard is never triggered for it).
///
/// Usage mirrors mlx-lm's contract:
/// ```ignore
/// let mut dt = StreamingDetokenizer::new(kind);
/// for id in ids { let seg = dt.step(&tk, id)?; emit(seg); }
/// let tail = dt.finalize(&tk, &all_ids)?; // lossy flush at true EOS
/// ```
#[derive(Debug)]
pub struct StreamingDetokenizer {
    kind: TokenizerKind,
    /// Ids accumulated so far (the engine already keeps its own copy for
    /// metrics; this is the detokenizer's authoritative list).
    ids: Vec<u32>,
    /// The longest `�`-free decoded prefix emitted so far. Never advanced to
    /// a `�`-terminated string (that is the bug A10 fixes).
    decoded: String,
    /// True until the first non-empty segment is emitted (for the
    /// strict-SPM first-segment leading-space rule).
    first_segment_pending: bool,
}

impl StreamingDetokenizer {
    /// Create a new detokenizer for the given tokenizer family.
    pub fn new(kind: TokenizerKind) -> Self {
        Self {
            kind,
            ids: Vec::new(),
            decoded: String::new(),
            first_segment_pending: true,
        }
    }

    /// Detokenizer's view of all accepted ids (engine keeps its own for
    /// timing; exposed so callers can avoid a parallel `Vec`).
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Feed one token id; return the segment to emit (possibly empty when the
    /// codepoint is incomplete and emission is withheld).
    ///
    /// `decode_err` callers: on a tokenizer decode error the segment is empty
    /// and the buffer state is preserved (next token retries the full
    /// decode), matching the engine's prior `Err(_) => String::new()` arm.
    pub fn step(
        &mut self,
        tk: &tokenizers::Tokenizer,
        id: u32,
    ) -> Result<String, tokenizers::Error> {
        self.ids.push(id);
        let full = tk.decode(&self.ids, true)?;
        Ok(self.diff_emit(full))
    }

    /// Flush any withheld tail at true end-of-stream. Lossy: a residual `�`
    /// here means generation genuinely stopped mid-codepoint, which is the
    /// only place a replacement char is acceptable (mirrors mlx-lm
    /// `_try_flush(force=True)` / HF `finalize`).
    ///
    /// Decodes the full id list once more (authoritative) and emits whatever
    /// was not yet emitted, replacement chars and all.
    pub fn finalize(&mut self, tk: &tokenizers::Tokenizer) -> Result<String, tokenizers::Error> {
        let full = tk.decode(&self.ids, true)?;
        let seg = self.tail_after_decoded(&full);
        self.decoded = full;
        Ok(self.maybe_trim_first(seg))
    }

    /// Byte-diff with the U+FFFD withholding guard. Emits nothing and does
    /// NOT advance `decoded` when `full` ends in a replacement char (split
    /// multi-byte codepoint mid-stream).
    fn diff_emit(&mut self, full: String) -> String {
        // Withhold: a trailing `�` means the last codepoint is incomplete.
        // Keep `decoded` at the previous clean prefix; the next token's
        // decode will not end in `�` and the diff then includes the
        // now-complete codepoint. ASCII never ends in `�` so this branch is
        // unreachable for pure-ASCII streams → byte-identical to pre-A10.
        if full.ends_with('\u{FFFD}') {
            return String::new();
        }
        let seg = self.tail_after_decoded(&full);
        self.decoded = full;
        self.maybe_trim_first(seg)
    }

    /// Byte-suffix of `full` past the already-emitted `decoded` prefix.
    /// Falls back to empty (not panic) if the prefix invariant breaks, same
    /// as the engine's prior behavior.
    fn tail_after_decoded(&self, full: &str) -> String {
        if full.len() >= self.decoded.len() && full.as_bytes().starts_with(self.decoded.as_bytes())
        {
            full[self.decoded.len()..].to_owned()
        } else {
            String::new()
        }
    }

    /// Strip exactly one leading space from the first non-empty segment, only
    /// for the strict-SPM decoder. No-op for Gemma (`SpmNoStrip`) and Qwen
    /// (`ByteLevel`) — their leading spaces, when present, are genuine.
    fn maybe_trim_first(&mut self, seg: String) -> String {
        if seg.is_empty() {
            return seg;
        }
        if self.first_segment_pending {
            self.first_segment_pending = false;
            if self.kind.trim_first_space() {
                if let Some(stripped) = seg.strip_prefix(' ') {
                    return stripped.to_owned();
                }
            }
        }
        seg
    }
}

#[cfg(test)]
#[path = "detokenizer_tests.rs"]
mod tests;
