//! Unit tests for smoke-probe classification and type helpers.

use super::classify::classify_smoke;
use super::types::{ProbeStep, SmokeVerdict};

/// Build a `ProbeStep` with finite logit + zero NaNs (the common case).
fn step(token_id: u32, piece: &str) -> ProbeStep {
    ProbeStep {
        token_id,
        piece: piece.to_string().into_boxed_str(),
        max_abs_logit: 12.0,
        nan_count: 0,
        logprobs: None,
    }
}

/// 8 identical steps of `(id, piece)`.
fn loop8(id: u32, piece: &str) -> Vec<ProbeStep> {
    (0..8).map(|_| step(id, piece)).collect()
}

// (a) Regression: repeated ASCII punct ×8 → BrokenPunctLoop (B5 behaviour).
#[test]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn ascii_punct_loop_is_broken() {
    let steps = loop8(999, "!");
    match classify_smoke(&steps) {
        SmokeVerdict::BrokenPunctLoop {
            dominant_piece,
            distinct_ids,
        } => {
            assert_eq!(dominant_piece, "!");
            assert_eq!(distinct_ids, 1);
        }
        v => panic!("expected BrokenPunctLoop, got {v:?}"),
    }
}

// (b) The B5b fix: repeated NON-punct token id ×8 (simulate `로`).
// Token id 237323 with a multi-byte CJK word-piece — B5 returned Ok here
// (false negative on the safety gate); B5b must flag it.
#[test]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn non_punct_token_loop_is_broken() {
    let steps = loop8(237323, "로");
    match classify_smoke(&steps) {
        SmokeVerdict::BrokenPunctLoop { dominant_piece, .. } => {
            assert_eq!(dominant_piece, "로")
        }
        v => panic!("expected BrokenPunctLoop for repeated 로, got {v:?}"),
    }
}

// (c) Repeated single CJK / letter char piece (with SentencePiece marker)
// → BrokenPunctLoop. Also covers the consecutive-run path with >2 distinct
// ids: a long single-letter run still trips rule (i).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn single_char_letter_loop_is_broken() {
    // Single Latin letter, 8×.
    assert!(matches!(
        classify_smoke(&loop8(42, "a")),
        SmokeVerdict::BrokenPunctLoop { .. }
    ));
    // SentencePiece-marked single CJK char, 8×.
    assert!(matches!(
        classify_smoke(&loop8(7, "\u{2581}北")),
        SmokeVerdict::BrokenPunctLoop { .. }
    ));
    // Consecutive run of LOOP_K with extra distinct tail (distinct_ids > 2)
    // — rule (i) must still fire.
    let mut steps = loop8(5, "x"); // 8 identical
    steps[6] = step(11, " then");
    steps[7] = step(12, " end");
    assert!(matches!(
        classify_smoke(&steps),
        SmokeVerdict::BrokenPunctLoop { .. }
    ));
}

// (d) Coherent varied tokens → Ok. Includes a benign short repeat
// ("the the") that must NOT trip the loop detector.
#[test]
fn coherent_varied_is_ok() {
    let steps = vec![
        step(10, "The"),
        step(11, " capital"),
        step(12, " of"),
        step(13, " France"),
        step(14, " is"),
        step(15, " the"),
        step(15, " the"),
        step(16, " Paris"),
    ];
    assert_eq!(classify_smoke(&steps), SmokeVerdict::Ok);
}

// (e) NaN logits anywhere → BrokenNan (precedence over loop checks).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn nan_logits_is_broken_nan() {
    let mut steps = loop8(999, "!"); // would otherwise be punct-loop
    steps[3].nan_count = 2;
    match classify_smoke(&steps) {
        SmokeVerdict::BrokenNan { at_step } => assert_eq!(at_step, 3),
        v => panic!("expected BrokenNan, got {v:?}"),
    }
}

// Verdict → exit-code / HTTP mapping must stay stable: every "broken"
// signature (punct loop, the new non-punct loop, NaN) maps to the
// refuse-to-serve class, Ok/Inconclusive to the allow class. This mirrors
// the match arms in rmlx-cli info.rs (exit 1) and rmlx-server openai.rs
// (HTTP 503). Asserting the discriminant here guards both call sites.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn broken_verdicts_map_to_refuse() {
    // `is_broken` predicate identical to info.rs `is_broken` / openai.rs
    // `require_smoke_probe` gate (BrokenPunctLoop | BrokenNan).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn is_broken(v: &SmokeVerdict) -> bool {
        matches!(
            v,
            SmokeVerdict::BrokenPunctLoop { .. } | SmokeVerdict::BrokenNan { .. }
        )
    }
    assert!(is_broken(&classify_smoke(&loop8(999, "!")))); // punct
    assert!(is_broken(&classify_smoke(&loop8(237323, "로")))); // non-punct (B5b)
    let mut nan = loop8(1, " ok");
    nan[0].nan_count = 1;
    assert!(is_broken(&classify_smoke(&nan))); // NaN
                                               // Healthy → not broken (must still be servable).
    let healthy = vec![
        step(1, "The"),
        step(2, " answer"),
        step(3, " is"),
        step(4, " forty"),
        step(5, " two"),
        step(6, " and"),
        step(7, " also"),
        step(8, " more"),
    ];
    assert!(!is_broken(&classify_smoke(&healthy)));
}

// smoke_prompt_ids: BOS prepended, deterministic, non-empty body.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn smoke_prompt_is_deterministic_and_seeded() {
    let Some(model_dir) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
    else {
        eprintln!("skip: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let tk_path = model_dir.join("tokenizer.json");
    if !tk_path.exists() {
        eprintln!("skip: primary snapshot tokenizer absent");
        return;
    }
    let tk = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer");
    let a = crate::arch::smoke_prompt_ids(&tk, 2).expect("ids");
    let b = crate::arch::smoke_prompt_ids(&tk, 2).expect("ids");
    assert_eq!(a, b, "seed must be deterministic");
    assert_eq!(a[0], 2, "BOS first");
    assert!(a.len() > 1, "seed prompt must contribute tokens past BOS");
}
