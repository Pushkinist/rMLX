//! Fixture lines for the round-loop summary scraper. The summary is the
//! `: done` line each loop closes a request with; the other lines are the
//! per-round record, the accounting error and an unrelated `: done`.

use super::scrape_spec_accept;

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
