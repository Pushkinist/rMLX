//! The `generate: host categorical sampler active` event.
//!
//! Neither request path runs without a model, so the event is read at the one
//! fn both paths call, and a source rule holds that each path calls it once
//! and that the deleted twin event is gone. The source rule reads text: it
//! cannot see whether the call is reached on every request.
//!
//! Mutations and the case that catches each:
//! - the message differs from the script's marker: `the_message_is_the_marker_the_script_reads`.
//! - a greedy request writes the event: `a_sampled_request_writes_one_info_event_and_a_greedy_one_none`.
//! - the event is written below INFO or drops a field: the same case.
//! - a path stops calling the helper, calls it twice, or brings the twin back:
//!   `each_path_calls_the_helper_once_and_the_twin_is_gone`.

use super::log_sampler_active;
use crate::audio::test_events::{events_of, field};
use rmlx_models::SamplerConfig;

const MESSAGE: &str = "generate: host categorical sampler active";

fn config(temperature: f32) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_p: 0.75,
        top_k: 20,
        min_p: 0.25,
        seed: Some(42),
        top_logprobs_k: 0,
    }
}

fn sampler_events(sampler: &SamplerConfig) -> Vec<Vec<(String, String)>> {
    events_of(|| log_sampler_active("model-x", sampler))
        .into_iter()
        .filter(|event| field(event, "message") == Some(MESSAGE))
        .collect()
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "the length is asserted on the line before"
)]
fn a_sampled_request_writes_one_info_event_and_a_greedy_one_none() {
    let events = sampler_events(&config(0.5));
    assert_eq!(events.len(), 1, "{events:?}");
    let event = &events[0];
    assert_eq!(field(event, "level"), Some("INFO"));
    assert_eq!(field(event, "model_id"), Some("model-x"));
    assert_eq!(field(event, "temperature"), Some("0.5"));
    assert_eq!(field(event, "top_p"), Some("0.75"));
    assert_eq!(field(event, "top_k"), Some("20"));
    assert_eq!(field(event, "min_p"), Some("0.25"));
    assert_eq!(field(event, "seed"), Some("42"));

    assert_eq!(sampler_events(&config(0.0)).len(), 0);
}

#[test]
fn the_message_is_the_marker_the_script_reads() {
    const READER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/lib/server_sampling.py"
    ));
    let marker = READER
        .lines()
        .find_map(|line| line.strip_prefix("MARKER = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or("no MARKER line");
    let events = events_of(|| log_sampler_active("m", &config(0.5)));
    let written = events
        .iter()
        .find_map(|event| field(event, "message"))
        .unwrap_or("no event");
    assert_eq!(written, marker);
}

#[test]
fn each_path_calls_the_helper_once_and_the_twin_is_gone() {
    const PLAIN: &str = include_str!("arch_generator.rs");
    const SPECULATIVE: &str = include_str!("speculative.rs");
    for (path, source) in [("arch_generator", PLAIN), ("speculative", SPECULATIVE)] {
        let calls = source
            .lines()
            .filter(|line| line.trim_start().starts_with("log_sampler_active("))
            .count();
        assert_eq!(calls, 1, "{path}");
        assert!(!source.contains(MESSAGE), "{path} writes the event itself");
        assert!(
            !source.contains("SpeculativeGenerator: sampling configuration"),
            "{path} still names the deleted twin"
        );
    }
}
