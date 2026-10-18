use super::*;
use serde_json::json;

const LEN: usize = 100;

fn pieces(len: usize) -> Vec<String> {
    (0..len).map(|i| format!("t{i}")).collect()
}

/// A chat-completion body with one record per piece. Record `i` lists two
/// candidates `gaps[i]` apart; a piece with no gap lists one.
fn body(tokens: &[String], gaps: &[Option<f32>]) -> String {
    let records: Vec<_> = tokens
        .iter()
        .zip(gaps)
        .map(|(token, gap)| {
            let mut top = vec![json!({"token": token, "logprob": -0.5})];
            if let Some(gap) = gap {
                top.push(json!({"token": "other", "logprob": -0.5 - gap}));
            }
            json!({"token": token, "logprob": -0.5, "top_logprobs": top})
        })
        .collect();
    json!({
        "choices": [{"logprobs": {"content": records}}],
        "usage": {"completion_tokens": tokens.len()},
    })
    .to_string()
}

/// Margins of 1.0 everywhere but `at`, where the gap is `gap`.
fn gaps_with(len: usize, at: usize, gap: f32) -> Vec<Option<f32>> {
    (0..len)
        .map(|i| Some(if i == at { gap } else { 1.0 }))
        .collect()
}

fn flipped(tokens: &[String], at: usize) -> Vec<String> {
    let mut out = tokens.to_vec();
    if let Some(slot) = out.get_mut(at) {
        *slot = "flip".to_string();
    }
    out
}

fn judge(plain: &str, spec: &str) -> Judgement {
    judge_bodies("plain.json", plain, "spec.json", spec)
}

#[test]
fn a_flip_at_a_tied_decision_is_a_near_tie() {
    let plain = pieces(LEN);
    let spec = flipped(&plain, 51);
    let got = judge(
        &body(&plain, &gaps_with(LEN, 51, 0.0)),
        &body(&spec, &gaps_with(LEN, 51, 0.0)),
    );
    assert_eq!(got.verdict, Verdict::NearTie, "{got:?}");
    assert_eq!(got.position, Some(51));
    assert_eq!(got.margin, Some(0.0));
    assert_eq!(got.rank, Some(0.0));
    assert_eq!(got.verdict.exit_code(), 0);
}

#[test]
fn a_flip_at_a_confident_decision_is_refused_with_its_position_and_rank() {
    let plain = pieces(LEN);
    let spec = flipped(&plain, 3);
    let got = judge(
        &body(&plain, &gaps_with(LEN, 3, 5.0)),
        &body(&spec, &gaps_with(LEN, 3, 5.0)),
    );
    assert_eq!(got.verdict, Verdict::Refused, "{got:?}");
    assert_eq!(got.position, Some(3));
    assert_eq!(got.margin, Some(5.0));
    assert!(got.rank.is_some_and(|r| r > MAX_DIVERGENCE_CONFIDENCE));
    assert!(got.reason.contains("token 3"), "{}", got.reason);
    assert_eq!(got.verdict.exit_code(), 1);
}

/// The margins come from the plain file: the speculative file's own gaps do not
/// move the verdict, so swapping the two arguments is not a no-op.
#[test]
fn the_margins_are_read_from_the_plain_file_only() {
    let plain = pieces(LEN);
    let spec = flipped(&plain, 40);
    let tied_plain = body(&plain, &gaps_with(LEN, 40, 0.0));
    let confident_spec = body(&spec, &gaps_with(LEN, 40, 5.0));
    assert_eq!(
        judge(&tied_plain, &confident_spec).verdict,
        Verdict::NearTie
    );
    assert_eq!(
        judge(&confident_spec, &tied_plain).verdict,
        Verdict::Refused
    );
}

#[test]
fn an_answer_that_runs_past_the_reference_is_refused() {
    let plain = pieces(LEN);
    let spec = pieces(LEN + 5);
    let got = judge(
        &body(&plain, &gaps_with(LEN, 0, 0.0)),
        &body(&spec, &gaps_with(LEN + 5, 0, 0.0)),
    );
    assert_eq!(got.verdict, Verdict::Refused, "{got:?}");
    assert_eq!(got.position, Some(LEN));
    assert_eq!(got.rank, None);
    assert!(got.reason.contains("ran on"), "{}", got.reason);
}

/// A tie at the point the shorter answer stopped would pass the rank, so the
/// ratio is what refuses it.
#[test]
fn an_answer_far_shorter_than_the_reference_is_refused_on_its_length() {
    let plain = pieces(LEN);
    let spec = plain[..LEN / 2].to_vec();
    let got = judge(
        &body(&plain, &gaps_with(LEN, LEN / 2, 0.0)),
        &body(&spec, &gaps_with(LEN / 2, 0, 0.0)),
    );
    assert_eq!(got.verdict, Verdict::Refused, "{got:?}");
    assert_eq!(got.position, Some(LEN / 2));
    assert!(got.reason.contains("ratio"), "{}", got.reason);
}

#[test]
fn a_record_count_that_is_not_the_token_count_cannot_be_judged() {
    let plain = pieces(LEN);
    let spec = flipped(&plain, 7);
    let short = {
        let mut value: serde_json::Value =
            serde_json::from_str(&body(&spec, &gaps_with(LEN, 7, 0.0))).unwrap();
        value["usage"]["completion_tokens"] = json!(LEN + 1);
        value.to_string()
    };
    let got = judge(&body(&plain, &gaps_with(LEN, 7, 0.0)), &short);
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
    assert!(got.reason.starts_with("spec.json:"), "{}", got.reason);
    assert_eq!(got.verdict.exit_code(), 2);
}

#[test]
fn a_plain_record_with_one_candidate_cannot_be_judged() {
    let plain = pieces(LEN);
    let spec = flipped(&plain, 7);
    let mut gaps = gaps_with(LEN, 7, 0.0);
    if let Some(slot) = gaps.get_mut(20) {
        *slot = None;
    }
    let got = judge(&body(&plain, &gaps), &body(&spec, &gaps_with(LEN, 7, 0.0)));
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
    assert!(got.reason.starts_with("plain.json:"), "{}", got.reason);
    assert!(got.reason.contains("record 20"), "{}", got.reason);
}

#[test]
fn a_body_with_no_logprobs_cannot_be_judged() {
    let plain = pieces(LEN);
    let bare = json!({
        "choices": [{"message": {"content": "x"}}],
        "usage": {"completion_tokens": LEN},
    })
    .to_string();
    let got = judge(&body(&plain, &gaps_with(LEN, 0, 0.0)), &bare);
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
    assert!(got.reason.starts_with("spec.json:"), "{}", got.reason);
}

#[test]
fn a_body_with_no_usage_cannot_be_judged() {
    let plain = pieces(LEN);
    let mut value: serde_json::Value =
        serde_json::from_str(&body(&plain, &gaps_with(LEN, 0, 0.0))).unwrap();
    value.as_object_mut().unwrap().remove("usage");
    let got = judge(&value.to_string(), &body(&plain, &gaps_with(LEN, 0, 0.0)));
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
    assert!(got.reason.starts_with("plain.json:"), "{}", got.reason);
}

#[test]
fn text_that_is_not_json_cannot_be_judged() {
    let plain = pieces(LEN);
    let got = judge("not json", &body(&plain, &gaps_with(LEN, 0, 0.0)));
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
    assert!(got.reason.starts_with("plain.json:"), "{}", got.reason);
}

/// Two answers the digests told apart but whose pieces read the same leave the
/// oracle nothing to compare.
#[test]
fn answers_with_the_same_pieces_cannot_be_judged() {
    let plain = pieces(LEN);
    let one = body(&plain, &gaps_with(LEN, 5, 0.0));
    let got = judge(&one, &one);
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
    assert!(got.reason.contains("same token pieces"), "{}", got.reason);
}

#[test]
fn an_empty_reference_cannot_be_judged() {
    let plain: Vec<String> = Vec::new();
    let spec = pieces(3);
    let got = judge(&body(&plain, &[]), &body(&spec, &gaps_with(3, 0, 0.0)));
    assert_eq!(got.verdict, Verdict::Unjudgeable, "{got:?}");
}

#[test]
fn the_output_names_only_the_figures_that_were_computed() {
    let plain = pieces(LEN);
    let spec = flipped(&plain, 51);
    let got = judge(
        &body(&plain, &gaps_with(LEN, 51, 0.25)),
        &body(&spec, &gaps_with(LEN, 51, 0.25)),
    );
    let lines = got.lines();
    assert_eq!(lines.first().map(String::as_str), Some("verdict=near_tie"));
    assert!(lines.contains(&"position=51".to_string()));
    assert!(lines.contains(&"margin=0.2500".to_string()));
    assert!(lines.contains(&"rank=0.0000".to_string()));
    assert!(lines.last().is_some_and(|l| l.starts_with("reason=")));
    assert!(lines.iter().all(|l| !l.contains('\n')));

    let unreadable = judge("not json", "not json").lines();
    assert_eq!(unreadable.len(), 2, "{unreadable:?}");
}

#[test]
fn a_missing_file_cannot_be_judged_and_names_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let present = dir.path().join("plain.json");
    std::fs::write(&present, body(&pieces(3), &gaps_with(3, 0, 0.0))).unwrap();
    let absent = dir.path().join("spec.json");
    assert_eq!(run_spec_answer(&present, &absent), 2);
}
