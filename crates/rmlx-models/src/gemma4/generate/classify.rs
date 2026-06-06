//! Smoke-probe classifier: [`classify_smoke`] + private helpers.

use super::types::{ProbeStep, SmokeVerdict};

// ---------------------------------------------------------------------------
// Smoke classifier
// ---------------------------------------------------------------------------

/// Number of repeats (consecutive or dominant) within the 8-token window that
/// counts as a degenerate loop. 6 of 8 is a clear majority — well past any
/// legitimate short repetition (e.g. "the the", a doubled digit) yet below the
/// full window so a single non-repeated tail token does not mask a loop.
const LOOP_K: usize = 6;

/// Classify whether a sequence of `ProbeStep`s shows a broken-snapshot pattern.
///
/// Heuristic (from CLAUDE.md "mxfp8 broken-snapshot hazard"; B5b widened):
/// - `BrokenNan`: any step had `nan_count > 0`.
/// - `BrokenPunctLoop` (variant name kept for stable exit-code / HTTP / test
///   mapping — it now covers any degenerate repeat, not only ASCII punct):
/// - `≤ 2` distinct token ids AND the dominant piece is a single-char
///   punctuation token (the original B5 signature), OR
/// - `≥ LOOP_K` consecutive identical token ids anywhere in the window
///   (catches the gemma-4-26b-a4b `로` bare-BOS loop and any word-piece
///   loop, not just punctuation), OR
/// - the dominant piece is a single character of *any* Unicode category
///   (punct / letter / CJK / digit) repeated for `≥ LOOP_K` of the window.
/// - `Inconclusive`: fewer than 2 steps (too short to judge).
/// - `Ok`: everything else.
///
/// No new crates: single-char detection is `chars().count() == 1` after
/// stripping the SentencePiece `▁` marker — category-agnostic.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn classify_smoke(steps: &[ProbeStep]) -> SmokeVerdict {
    // --- NaN check first ---
    for (i, s) in steps.iter().enumerate() {
        if s.nan_count > 0 {
            return SmokeVerdict::BrokenNan { at_step: i };
        }
    }

    // Inconclusive if we have fewer than 2 steps.
    if steps.len() < 2 {
        return SmokeVerdict::Inconclusive {
            reason: format!("only {} step(s) generated", steps.len()),
        };
    }

    // --- (i) Consecutive identical-token-id run ---
    // Any token id repeated LOOP_K times back-to-back is a degenerate loop
    // regardless of what the piece decodes to (word-piece, CJK, anything).
    let mut run_id = steps[0].token_id;
    let mut run_len = 1_usize;
    for s in steps.iter().skip(1) {
        if s.token_id == run_id {
            run_len += 1;
            if run_len >= LOOP_K {
                return SmokeVerdict::BrokenPunctLoop {
                    dominant_piece: s.piece.to_string(),
                    distinct_ids: count_distinct_ids(steps),
                };
            }
        } else {
            run_id = s.token_id;
            run_len = 1;
        }
    }

    // --- (ii) Original B5 rule: ≤ 2 distinct ids + single-char punct piece ---
    let distinct_ids = count_distinct_ids(steps);
    if distinct_ids <= 2 {
        // Find the modal piece.
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for s in steps {
            *counts.entry(s.token_id).or_insert(0) += 1;
        }
        if let Some(dominant_id) = counts.iter().max_by_key(|(_, c)| *c).map(|(id, _)| *id) {
            let dominant_piece = steps
                .iter()
                .find(|s| s.token_id == dominant_id)
                .map_or("", |s| s.piece.as_ref());
            if is_punct_piece(dominant_piece) {
                return SmokeVerdict::BrokenPunctLoop {
                    dominant_piece: dominant_piece.to_string(),
                    distinct_ids,
                };
            }
        }
    }

    // --- (iii) B5b extension: single-char piece (any category) dominant ≥ LOOP_K ---
    // Count how many steps have a single-char piece.
    let mut piece_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for s in steps {
        if is_single_char_piece(&s.piece) {
            *piece_counts.entry(s.piece.to_string()).or_insert(0) += 1;
        }
    }
    if let Some((dominant_piece, &count)) = piece_counts.iter().max_by_key(|(_, c)| *c) {
        if count >= LOOP_K {
            return SmokeVerdict::BrokenPunctLoop {
                dominant_piece: dominant_piece.clone(),
                distinct_ids: count_distinct_ids(steps),
            };
        }
    }

    SmokeVerdict::Ok
}

/// Count distinct token ids across `steps`.
fn count_distinct_ids(steps: &[ProbeStep]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for s in steps {
        seen.insert(s.token_id);
    }
    seen.len()
}

/// True if `piece` is exactly one Unicode scalar value (any category) after
/// stripping the optional SentencePiece `▁` leading-space marker. Category
/// agnostic — punctuation, letter, CJK, digit all qualify.
fn is_single_char_piece(piece: &str) -> bool {
    let stripped = piece.strip_prefix('▁').unwrap_or(piece);
    let mut chars = stripped.chars();
    chars.next().is_some() && chars.next().is_none()
}

/// True if `piece` looks like a single-character punctuation token.
///
/// The broken-snapshot pattern produces tokens that decode to a single ASCII
/// punctuation character (or a common unicode punctuation character).
/// We check that the piece is exactly 1 char (after stripping the `▁` SentencePiece
/// space prefix if present) and belongs to the punctuation set.
fn is_punct_piece(piece: &str) -> bool {
    // Strip optional SentencePiece leading space marker.
    let stripped = piece.strip_prefix('▁').unwrap_or(piece);

    // Must be exactly one Unicode scalar value.
    let mut chars = stripped.chars();
    let ch = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if chars.next().is_some() {
        return false; // more than one char
    }

    // Punctuation set — hardcoded per CLAUDE.md, no unicode-properties crate.
    // '…' = \u{2026}, '–' = \u{2013}, '—' = \u{2014} — using the unicode literals directly.
    matches!(
        ch,
        '!' | '?' | '.' | ',' | '"' | '\'' | ':' | ';' | '-' | '–' | '—' | '…'
    )
}
