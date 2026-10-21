//! Fixture lines for the round-loop summary scraper. The summary is the
//! `: done` line each loop closes a request with; the other lines are the
//! per-round record, the accounting error and an unrelated `: done`.

use super::{golden_dir, is_coherent_answer, scrape_spec_accept, Manifest};

/// The message and `loop_kind` of each loop's summary line, one per loop.
const SUMMARY_EVENTS: [(&str, &str); 7] = [
    ("mtp_assistant_generate: done", "MtpAssistant"),
    ("mtp_generate: done", "MtpSidecar"),
    ("dflash_generate: done", "DFlash"),
    ("dflash2_generate: done", "DFlash2"),
    ("eagle3_generate: done", "Eagle3"),
    ("spec_generate_greedy_cached: done", "TwoModelGreedy"),
    (
        "spec_generate_stochastic_cached: done",
        "TwoModelStochastic",
    ),
];

fn summary_line(message: &str, loop_kind: &str, accept_rate: f64, rounds: u64) -> String {
    serde_json::json!({
        "fields": {
            "message": message,
            "loop_kind": loop_kind,
            "accept_rate": accept_rate,
            "rounds": rounds,
        }
    })
    .to_string()
}

fn round_line() -> String {
    serde_json::json!({
        "fields": {"message": "speculative round", "loop_kind": "TwoModelGreedy", "round": 1, "accept": 2}
    })
    .to_string()
}

fn accounting_line() -> String {
    serde_json::json!({
        "fields": {
            "message": "speculative round accounting is inconsistent: x",
            "loop_kind": "TwoModelGreedy",
        }
    })
    .to_string()
}

fn unrelated_done_line() -> String {
    serde_json::json!({"fields": {"message": "gemma4 preprocess: done"}}).to_string()
}

fn scrape(lines: &[String]) -> Result<(f64, u64), String> {
    let home = tempfile::tempdir().expect("create temp home");
    let logs = home.path().join("logs");
    std::fs::create_dir_all(&logs).expect("create logs dir");
    std::fs::write(logs.join("run.jsonl"), lines.join("\n")).expect("write run jsonl");
    scrape_spec_accept(home.path())
}

#[test]
fn every_loop_summary_line_is_read() {
    for (message, loop_kind) in SUMMARY_EVENTS {
        let got = scrape(&[summary_line(message, loop_kind, 0.5, 3)]);
        assert_eq!(got, Ok((0.5, 3)), "{message}");
    }
}

#[test]
fn lines_after_the_summary_are_not_taken_for_it() {
    let got = scrape(&[
        summary_line(
            "spec_generate_greedy_cached: done",
            "TwoModelGreedy",
            0.25,
            2,
        ),
        round_line(),
        accounting_line(),
        unrelated_done_line(),
    ]);
    assert_eq!(got, Ok((0.25, 2)));
}

#[test]
fn the_last_summary_line_wins() {
    let got = scrape(&[
        summary_line(
            "spec_generate_greedy_cached: done",
            "TwoModelGreedy",
            0.25,
            2,
        ),
        summary_line(
            "spec_generate_greedy_cached: done",
            "TwoModelGreedy",
            0.75,
            4,
        ),
    ]);
    assert_eq!(got, Ok((0.75, 4)));
}

#[test]
fn a_run_with_no_summary_line_is_an_error() {
    let got = scrape(&[round_line(), accounting_line(), unrelated_done_line()]);
    assert!(got.is_err(), "{got:?}");
}

const MANIFEST: &str = include_str!("manifest.toml");

fn manifest() -> Manifest {
    toml::from_str(MANIFEST).expect("manifest.toml must parse")
}

#[test]
fn case_ids_are_unique() {
    let mut seen = std::collections::BTreeSet::new();
    let repeated: Vec<String> = manifest()
        .case
        .into_iter()
        .map(|c| c.id)
        .filter(|id| !seen.insert(id.clone()))
        .collect();
    assert!(repeated.is_empty(), "repeated case ids: {repeated:?}");
}

#[test]
fn a_one_word_answer_carrying_the_anchor_is_coherent() {
    assert!(is_coherent_answer("Teal.", Some("teal")));
    assert!(is_coherent_answer("  TEAL\n", Some("teal")));
}

#[test]
fn a_one_word_answer_without_the_anchor_is_not_coherent() {
    assert!(!is_coherent_answer("Blue.", Some("teal")));
    assert!(!is_coherent_answer("Teal.", None));
    assert!(!is_coherent_answer("Teal.", Some("")));
    assert!(!is_coherent_answer("Teal.", Some("!teal")));
}

#[test]
fn degenerate_output_stays_incoherent_with_an_anchor() {
    for text in [
        "",
        "   ",
        ".",
        ". . .",
        "the the the the",
        "a b c",
        "tt",
        "NaN",
        "TealTealTealTeal",
        "Teal.Teal.Teal.",
        "xxtealxx",
    ] {
        assert!(!is_coherent_answer(text, Some("teal")), "{text:?}");
        assert!(!is_coherent_answer(text, Some("the")), "{text:?}");
    }
}

#[test]
fn a_multi_word_answer_is_judged_as_before() {
    assert!(is_coherent_answer("Paris is the capital of France.", None));
    assert!(!is_coherent_answer("teal teal teal teal", Some("teal")));
}

/// A `golden` case with no recorded file records one and passes on its first
/// run, so it asserts nothing until the file is checked in.
#[test]
fn every_golden_case_has_a_recorded_file() {
    let missing: Vec<String> = manifest()
        .case
        .into_iter()
        .filter(|c| c.assert.kind == "golden")
        .map(|c| c.id)
        .filter(|id| !golden_dir().join(format!("{id}.json")).exists())
        .collect();
    let files: Vec<String> = missing
        .iter()
        .map(|id| format!("crates/rmlx-cli/tests/e2e/golden/{id}.json"))
        .collect();
    assert!(
        missing.is_empty(),
        "golden cases without a recorded file: {files:?}; record each with \
         `RMLX_E2E_REGEN_GOLDEN=1 RMLX_E2E_ONLY=<case id> cargo test -p rmlx-cli --test \
         e2e_harness -- --ignored --test-threads=1 --nocapture` and check the file in"
    );
}
