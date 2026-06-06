//! Unit tests for the stop-sequence matcher.

use super::{find_stop_match, StopMatcher};

fn ss(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

// ── find_stop_match (non-streaming) ────────────────────────────────────────────

#[test]
fn no_stop_returns_none() {
    assert!(find_stop_match("alpha bravo charlie", &ss(&["zzz"])).is_none());
    assert!(find_stop_match("alpha bravo", &[]).is_none());
}

#[test]
fn single_stop_excludes_string() {
    let hit = find_stop_match("alpha bravo charlie delta", &ss(&["charlie"])).unwrap();
    assert_eq!(hit.offset, "alpha bravo ".len());
    assert_eq!(hit.matched, "charlie");
    // Caller truncation drops the stop string and everything after.
    let text = "alpha bravo charlie delta";
    assert_eq!(&text[..hit.offset], "alpha bravo ");
}

#[test]
fn multi_stop_earliest_offset_wins() {
    // "delta" appears before "bravo" in the text; earliest offset must win
    // regardless of array order.
    let text = "x delta y bravo z";
    let hit = find_stop_match(text, &ss(&["bravo", "delta"])).unwrap();
    assert_eq!(hit.matched, "delta");
    assert_eq!(hit.offset, "x ".len());
}

#[test]
fn multi_stop_tie_first_in_array_wins() {
    // Both match at offset 0; OpenAI "first in the array" precedence.
    let hit = find_stop_match("abc", &ss(&["abc", "ab"])).unwrap();
    assert_eq!(hit.matched, "abc");
    assert_eq!(hit.offset, 0);
}

#[test]
fn empty_stop_string_ignored() {
    // An empty needle must not truncate everything at offset 0.
    let hit = find_stop_match("alpha charlie", &ss(&["", "charlie"])).unwrap();
    assert_eq!(hit.matched, "charlie");
}

// ── StopMatcher (streaming) ─────────────────────────────────────────────────────

/// Drive the matcher with a sequence of pieces, concatenate the emitted text.
fn run(stops: &[&str], pieces: &[&str]) -> (String, bool, Option<String>) {
    let mut m = StopMatcher::new(&ss(stops));
    let mut out = String::new();
    let mut stopped = false;
    let mut matched = None;
    for p in pieces {
        let r = m.push(p);
        out.push_str(&r.emit);
        if r.stopped {
            stopped = true;
            matched = r.matched;
            break;
        }
    }
    if !stopped {
        out.push_str(&m.finalize());
    }
    (out, stopped, matched)
}

#[test]
fn inert_when_no_stops() {
    let mut m = StopMatcher::new(&[]);
    assert!(!m.is_active());
    let r = m.push("anything goes");
    assert_eq!(r.emit, "anything goes");
    assert!(!r.stopped);
}

#[test]
fn streaming_single_token_stop() {
    let (out, stopped, matched) = run(&["charlie"], &["alpha ", "bravo ", "charlie", " delta"]);
    assert!(stopped);
    assert_eq!(matched.as_deref(), Some("charlie"));
    assert_eq!(out, "alpha bravo ");
}

#[test]
fn streaming_token_straddling_stop() {
    // The stop "charlie" is split across three pieces: "char" | "l" | "ie".
    // The matcher must hold back "char" and "charl" until the full needle
    // forms, then emit nothing past the boundary.
    let (out, stopped, matched) = run(
        &["charlie"],
        &["alpha bravo ", "char", "l", "ie", " delta echo"],
    );
    assert!(stopped);
    assert_eq!(matched.as_deref(), Some("charlie"));
    assert_eq!(out, "alpha bravo ");
}

#[test]
fn streaming_partial_prefix_then_diverges() {
    // "char" is a prefix of "charlie" so it is held back, but the next piece
    // is "coal" not "lie" → the held tail was a false alarm and must be
    // emitted in full, nothing lost.
    let (out, stopped, _) = run(&["charlie"], &["abc ", "char", "coal ", "done"]);
    assert!(!stopped);
    assert_eq!(out, "abc charcoal done");
}

#[test]
fn streaming_stop_not_present_emits_all() {
    let (out, stopped, _) = run(&["zzz"], &["alpha ", "bravo ", "charlie"]);
    assert!(!stopped);
    assert_eq!(out, "alpha bravo charlie");
}

#[test]
fn streaming_multi_stop_first_match_wins() {
    let (out, stopped, matched) = run(&["echo", "charlie"], &["a charlie b echo c"]);
    assert!(stopped);
    // "charlie" appears before "echo" in the text → earliest offset wins.
    assert_eq!(matched.as_deref(), Some("charlie"));
    assert_eq!(out, "a ");
}

#[test]
fn streaming_no_post_stop_emission() {
    // After a stop, further pushes must emit nothing (the stream is finished).
    let mut m = StopMatcher::new(&ss(&["stop"]));
    let first = m.push("keep stop drop");
    assert!(first.stopped);
    assert_eq!(first.emit, "keep ");
    let after = m.push(" more tokens");
    assert_eq!(after.emit, "");
    assert!(!after.stopped);
}

#[test]
fn streaming_multibyte_held_tail_is_char_safe() {
    // A multi-byte char at the suffix boundary must not panic on slicing.
    // Stop "→end" with a held tail that includes the 3-byte arrow.
    let (out, stopped, matched) = run(&["\u{2192}end"], &["start ", "\u{2192}", "end after"]);
    assert!(stopped);
    assert_eq!(matched.as_deref(), Some("\u{2192}end"));
    assert_eq!(out, "start ");
}

// ── Boundary arithmetic ────────────────────────────────────────────────────────

#[test]
fn streaming_single_byte_stop_across_chunks() {
    // Stop ["X"] fed across chunks ["ab","cX","de"].
    // The matcher must emit "ab" and "c" then stop on "X",
    // leaving "de" suppressed.
    let (out, stopped, matched) = run(&["X"], &["ab", "cX", "de"]);
    assert!(stopped, "expected stop to fire");
    assert_eq!(matched.as_deref(), Some("X"));
    assert_eq!(out, "abc");
}

#[test]
fn streaming_overlapping_stops_shorter_matches_first() {
    // Stops ["abc","ab"]: both share prefix "ab". In the text "ab xyz" the
    // shorter stop "ab" matches at offset 0 and must win because it actually
    // appears there; "abc" does NOT appear (no following 'c').
    let (out, stopped, matched) = run(&["abc", "ab"], &["ab", " xyz"]);
    assert!(stopped, "expected stop to fire");
    assert_eq!(matched.as_deref(), Some("ab"));
    assert_eq!(out, "");
}
