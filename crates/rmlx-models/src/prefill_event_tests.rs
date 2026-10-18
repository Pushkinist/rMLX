//! The `generate: prefill` event of a request.
//!
//! Three instruments, and each is blind where the others read:
//!
//! - The event itself, under a scripted report: its fields and what each one
//!   counts. It cannot read which request path wrote it.
//! - The tiny attention-only Qwen3.5 verifier, loaded on the CPU, serves a cold
//!   request, an exact repeat and a longer prompt that resumes from the first.
//!   Each leaves exactly one event with the counts of its path. It reads one
//!   architecture only. The speculative round loop has its own case beside its
//!   scripted drafter, in `speculative/round_loop_tests.rs`.
//! - A source rule lists every fn that shows a request its first token and holds
//!   that it calls `log_prefill` once for each path it serves. It reads text, so
//!   it cannot read where in the path the call sits or what it is given, and a
//!   path moved between two fns of one file is read as it stands.
//!
//! Mutations and the case that catches each:
//! - a path writes no event, or two: `a_cold_an_exact_and_a_resumed_request_write_one_event_each`
//!   on Qwen3.5, `every_path_that_shows_a_first_token_writes_the_event` on the rest.
//! - an exact hit reports the prompt as forwarded: the same Qwen3.5 case.
//! - a resume reports the whole prompt as forwarded or none as reused: the same case.
//! - the event reports forwarded tokens as the prompt less none: `the_event_reports_what_the_prefill_did`.

use super::*;

use crate::arch::Architecture;
use crate::test_warnings::{field, info_events_during, only, EventFields};

const EVENT: &str = "generate: prefill";

/// The `generate: prefill` events `f` writes on this thread.
fn prefill_events(f: impl FnOnce()) -> Vec<EventFields> {
    info_events_during(EVENT, f)
}

/// The counts of an event, in the order prompt, reused, forwarded.
fn counts(fields: &EventFields) -> [&str; 3] {
    ["prompt_tokens", "reused_tokens", "forwarded_tokens"]
        .map(|name| field(fields, name).unwrap_or("absent"))
}

#[test]
fn the_event_reports_what_the_prefill_did() {
    let cases = [
        (
            "cold",
            PrefillReport::cold(100, 2_500_000),
            ["100", "0", "100"],
            "2.5",
        ),
        (
            "resumed",
            PrefillReport::reusing(100, 40, 1_000_000),
            ["100", "40", "60"],
            "1.0",
        ),
        (
            "exact",
            PrefillReport::exact(100),
            ["100", "100", "0"],
            "0.0",
        ),
        (
            "carried",
            PrefillReport::carried(100, 2_000_000).with_draft(500_000),
            ["100", "0", "99"],
            "2.0",
        ),
    ];
    for (path, report, want, want_ms) in cases {
        let event = only(prefill_events(|| log_prefill("Arch", &report)));
        let event = &event;
        assert_eq!(counts(event), want, "{path}");
        assert_eq!(field(event, "prefill_ms"), Some(want_ms), "{path}");
        let want_draft = if path == "carried" { "0.5" } else { "0.0" };
        assert_eq!(field(event, "draft_prefill_ms"), Some(want_draft), "{path}");
        assert_eq!(field(event, "arch"), Some("\"Arch\""), "{path}");
        let mut names: Vec<_> = event.iter().map(|(name, _)| *name).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "arch",
                "draft_prefill_ms",
                "forwarded_tokens",
                "message",
                "prefill_ms",
                "prompt_tokens",
                "reused_tokens"
            ],
            "{path}"
        );
    }
}

/// A prompt of `len` tokens of the tiny vocabulary, which no other test shares.
fn prompt(len: usize) -> Vec<u32> {
    (0..len).map(|i| 1 + (i * 5 % 7) as u32).collect()
}

/// One request of three tokens to the tiny verifier on the CPU, and the events
/// it wrote.
#[allow(
    clippy::expect_used,
    reason = "test scaffolding: a request that fails is a failed test"
)]
fn serve(model: &Architecture, tokenizer: &Tokenizer, prompt: &[u32]) -> Vec<EventFields> {
    let sampler_cfg = greedy_cfg();
    let penalty_cfg = no_penalties();
    let mut rng = Pcg32::new(0);
    let mut history = Vec::new();
    let mut step_fn = |_: &ProbeStep| StepVerdict::Continue;
    prefill_events(|| {
        model
            .generate_greedy(
                tokenizer,
                prompt,
                3,
                Device::Cpu,
                Some(KvQuant::None),
                Some(2048),
                4,
                &[],
                &mut step_fn,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut history,
                &AtomicBool::new(false),
            )
            .expect("the tiny verifier serves the request");
    })
}

/// Qwen3.5 serves a request three ways: from nothing, from an identical
/// prompt, and from a prompt it extends. Each writes one event, and the counts
/// say which way it was served. The prompt cache is process-global, so the case
/// runs in a process of its own.
#[test]
fn a_cold_an_exact_and_a_resumed_request_write_one_event_each() {
    crate::test_child::run_child(
        "decode_loop::decode_loop_tests::prefill_event_tests::\
         a_cold_an_exact_and_a_resumed_request_write_one_event_each_child",
    );
}

#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
#[allow(
    clippy::expect_used,
    reason = "test scaffolding: a verifier that does not load is a failed test"
)]
fn a_cold_an_exact_and_a_resumed_request_write_one_event_each_child() {
    if !crate::test_child::started_by_parent() {
        return;
    }
    rmlx_mlx::forbid_gpu().expect("forbid the GPU");
    let model = crate::qwen3_5_moe::attention_only_architecture().expect("load the tiny verifier");
    let tokenizer = tiny_tokenizer(8);
    model.clear_prompt_cache();

    let base = prompt(300);
    let mut longer = base.clone();
    longer.extend(prompt(20));

    let cold = only(serve(&model, &tokenizer, &base));
    assert_eq!(counts(&cold), ["300", "0", "300"], "cold");
    assert_ne!(
        field(&cold, "prefill_ms"),
        Some("0.0"),
        "a cold prefill takes time"
    );

    let exact = only(serve(&model, &tokenizer, &base));
    assert_eq!(counts(&exact), ["300", "300", "0"], "exact");
    assert_eq!(
        field(&exact, "prefill_ms"),
        Some("0.0"),
        "an exact hit runs no prefill"
    );

    let resumed = only(serve(&model, &tokenizer, &longer));
    assert_eq!(counts(&resumed), ["320", "300", "20"], "resumed");
    assert_ne!(
        field(&resumed, "prefill_ms"),
        Some("0.0"),
        "the tail takes time"
    );
    println!("{}", crate::test_child::CHILD_DONE);
}

/// What shows a request its first token, in the fns that serve one.
const FIRST_TOKEN: [&str; 3] = ["emit_replayed_token", "emit_token", "emit_seed_token"];

/// A fn that starts a request: its file, its name, the calls that show the
/// request its first token, and the number of paths the fn serves.
type RequestStart = (&'static str, &'static str, &'static [&'static str], usize);

/// Every fn that starts a request. Each path calls `log_prefill` once, ahead of
/// the call that shows the token.
const REQUEST_STARTS: [RequestStart; 11] = [
    ("bitnet/generate.rs", "generate_greedy", &FIRST_TOKEN, 2),
    ("gemma3/generate.rs", "exact_hit_decode", &FIRST_TOKEN, 1),
    ("gemma3/generate.rs", "generate_greedy", &FIRST_TOKEN, 1),
    ("gemma4/generate/mod.rs", "generate_greedy", &FIRST_TOKEN, 2),
    ("laguna/generate.rs", "generate_greedy", &FIRST_TOKEN, 2),
    ("qwen2/generate.rs", "generate_greedy", &FIRST_TOKEN, 2),
    ("qwen3.rs", "generate_greedy", &FIRST_TOKEN, 2),
    (
        "qwen3_5_moe/generate.rs",
        "generate_greedy",
        &FIRST_TOKEN,
        3,
    ),
    // The text path hands the first token to the shared decode fn.
    (
        "qwen3_vl_moe/generate.rs",
        "generate_greedy",
        &["decode_from"],
        2,
    ),
    (
        "qwen3_vl_moe/generate.rs",
        "generate_image",
        &FIRST_TOKEN,
        1,
    ),
    ("speculative/round_loop.rs", "run_rounds", &FIRST_TOKEN, 1),
];

/// The fns that show a token and start no request: they carry on a request that
/// a fn above started, or they are the calls themselves.
const CONTINUATIONS: [(&str, &str); 9] = [
    ("bitnet/generate.rs", "decode_loop"),
    ("decode_loop.rs", "emit_replayed_token"),
    ("decode_loop.rs", "pipelined_decode"),
    ("laguna/generate.rs", "decode_loop"),
    ("qwen2/generate.rs", "decode_loop"),
    ("qwen3_vl_moe/generate.rs", "decode_from"),
    ("speculative/mod.rs", "emit_step"),
    ("speculative/round_common.rs", "emit_round_tokens"),
    ("speculative/round_common.rs", "emit_seed_token"),
];

/// The fns that call a fn that starts a request and write no event of their own.
const DISPATCHERS: [(&str, &str); 11] = [
    ("arch/loader.rs", "run_smoke_probe"),
    ("arch/mod.rs", "generate_greedy"),
    ("arch/mod.rs", "generate_image"),
    ("speculative/dflash/round.rs", "dflash_generate"),
    ("speculative/dflash2/round.rs", "dflash2_generate"),
    ("speculative/eagle3/round.rs", "eagle3_generate"),
    ("speculative/gemma4_assistant.rs", "mtp_assistant_generate"),
    ("speculative/mod.rs", "spec_generate_greedy"),
    ("speculative/mod.rs", "spec_generate_greedy_cached"),
    ("speculative/mod.rs", "spec_generate_stochastic_cached"),
    ("speculative/mtp.rs", "mtp_generate"),
];

/// The fns of `sources` that show a token, by the fixpoint of "calls a fn that
/// shows one": it starts from the calls in [`FIRST_TOKEN`] and adds every fn that
/// calls a member, and then every fn that calls that one, by name. A fn that
/// writes the event is in it too. Each is `(file, name, code)`.
fn token_showing_fns(sources: &[(String, String)]) -> Vec<(String, String, String)> {
    use crate::stop_scan::{calls_of, fns_of};

    let fns: Vec<(String, String, String)> = sources
        .iter()
        .flat_map(|(file, code)| {
            fns_of(code)
                .into_iter()
                .map(|(name, text)| (file.clone(), name, text))
        })
        .collect();
    let mut shown: Vec<&str> = FIRST_TOKEN.to_vec();
    let mut members: Vec<usize> = Vec::new();
    loop {
        let before = members.len();
        for (index, (_, name, text)) in fns.iter().enumerate() {
            if !members.contains(&index)
                && shown
                    .iter()
                    .any(|callee| !calls_of(text, callee).is_empty())
            {
                members.push(index);
                shown.push(name);
            }
        }
        if members.len() == before {
            break;
        }
    }
    for (index, (_, _, text)) in fns.iter().enumerate() {
        if !members.contains(&index) && !calls_of(text, "log_prefill").is_empty() {
            members.push(index);
        }
    }
    fns.into_iter()
        .enumerate()
        .filter(|(index, _)| members.contains(index))
        .map(|(_, f)| f)
        .collect()
}

/// What is wrong with how `sources` write the event, one line each.
fn event_problems(
    sources: &[(String, String)],
    starts: &[RequestStart],
    continuations: &[(&str, &str)],
    dispatchers: &[(&str, &str)],
) -> Vec<String> {
    use crate::stop_scan::calls_of;

    let mut problems = Vec::new();
    let members = token_showing_fns(sources);
    for (file, name, text) in &members {
        let key = format!("{file}::{name}");
        let logs: Vec<usize> = calls_of(text, "log_prefill").iter().map(|c| c.at).collect();
        let pair = (file.as_str(), name.as_str());
        if let Some((_, _, callees, paths)) = starts.iter().find(|(f, n, _, _)| (*f, *n) == pair) {
            let mut firsts: Vec<usize> = callees
                .iter()
                .flat_map(|callee| calls_of(text, callee))
                .map(|call| call.at)
                .collect();
            firsts.sort_unstable();
            if logs.len() != *paths || firsts.len() != *paths {
                problems.push(format!(
                    "{key} serves {paths} paths, shows {} first tokens and writes the event {} times",
                    firsts.len(),
                    logs.len()
                ));
            } else if logs.iter().zip(&firsts).any(|(log, first)| log > first) {
                problems.push(format!(
                    "{key} shows a first token before it writes the event"
                ));
            }
        } else if continuations.contains(&pair) || dispatchers.contains(&pair) {
            if !logs.is_empty() {
                problems.push(format!("{key} is not a request start and writes the event"));
            }
        } else {
            problems.push(format!("{key} reaches a token and is in no list"));
        }
    }
    for (file, name) in starts
        .iter()
        .map(|(f, n, _, _)| (*f, *n))
        .chain(continuations.iter().copied())
        .chain(dispatchers.iter().copied())
    {
        if !members
            .iter()
            .any(|(f, n, _)| (f.as_str(), n.as_str()) == (file, name))
        {
            problems.push(format!("{file}::{name} is listed and reaches no token"));
        }
    }
    problems
}

/// Every fn that reaches a token is listed above, and every fn that starts a
/// request writes the event once for each path it serves, before it shows the
/// first token of that path. A new fn that calls a helper which shows a token is
/// found by the fixpoint and must be listed. An entry in `CONTINUATIONS` must
/// continue an existing request: the rule cannot read that, so a reviewer checks
/// each addition.
///
/// The rule reads text. It cannot read that a call is given the counts of its
/// own path, or that its path is reached, and a path moved to another fn is
/// read as it stands. A fn is known by its name: two fns of one name are one.
#[test]
fn every_path_that_shows_a_first_token_writes_the_event() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let sources = crate::stop_scan::sources_under(&src);
    let problems = event_problems(&sources, &REQUEST_STARTS, &CONTINUATIONS, &DISPATCHERS);
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// A new arch fn that calls a helper which shows a token, and writes no event,
/// fails the rule. The helper is a continuation, so the call is not a first-token
/// call by name; only the fixpoint reaches the new fn.
#[test]
fn a_new_caller_of_a_fn_that_shows_a_token_must_be_listed() {
    let source = |file: &str, code: &str| (file.to_owned(), code.to_owned());
    let helper = source(
        "decode_loop.rs",
        "fn emit_token() {}\nfn pipelined_decode() {\n    emit_token();\n}\n",
    );
    // The new fn comes first, so a single pass over the fns would not find it.
    let sources = vec![
        source(
            "newarch/generate.rs",
            "fn generate_greedy() {\n    pipelined_decode();\n}\n",
        ),
        helper.clone(),
    ];
    let continuations = [("decode_loop.rs", "pipelined_decode")];
    let problems = event_problems(&sources, &[], &continuations, &[]);
    assert_eq!(
        problems,
        ["newarch/generate.rs::generate_greedy reaches a token and is in no list"]
    );
    let starts: [RequestStart; 1] = [(
        "newarch/generate.rs",
        "generate_greedy",
        &["pipelined_decode"],
        1,
    )];
    assert_eq!(
        event_problems(&sources, &starts, &continuations, &[]),
        ["newarch/generate.rs::generate_greedy serves 1 paths, shows 1 first tokens and writes the event 0 times"]
    );
    let logged = vec![
        source(
            "newarch/generate.rs",
            "fn generate_greedy() {\n    log_prefill();\n    pipelined_decode();\n}\n",
        ),
        helper,
    ];
    assert_eq!(
        event_problems(&logged, &starts, &continuations, &[]),
        Vec::<String>::new()
    );
}

/// The collector turns on the callsites of the thread that asked for it and of no
/// other. The engine has opt-in switches that ask `tracing::enabled!`, and a test
/// that captures nothing must not find one on.
#[test]
fn the_collector_switches_on_nothing_for_a_thread_that_captures_nothing() {
    use crate::test_warnings::events_during;

    let _ = events_during(|| {});
    assert!(
        !tracing::enabled!(tracing::Level::DEBUG),
        "an unarmed thread"
    );
    assert!(!crate::speculative::phases_charged(), "the phase charge");
    let (armed, other_thread) = {
        let mut armed = false;
        let mut other = true;
        let _ = events_during(|| {
            armed = tracing::enabled!(tracing::Level::DEBUG);
            other = std::thread::scope(|scope| {
                scope
                    .spawn(|| tracing::enabled!(tracing::Level::DEBUG))
                    .join()
                    .unwrap_or(true)
            });
        });
        (armed, other)
    };
    assert!(armed, "the thread that collects is on");
    assert!(!other_thread, "a thread beside it is not");
}

/// The drafters whose `prefill` times the target model's forwards for the event.
const DRAFTER_FILES: [&str; 5] = [
    "speculative/mtp.rs",
    "speculative/dflash/round.rs",
    "speculative/dflash2/round.rs",
    "speculative/eagle3/round.rs",
    "speculative/gemma4_assistant.rs",
];

/// A drafter's `prefill_ns` is the target's forwards alone: the span runs up to
/// the point its logits are forced, which the verifier's guard marks, and ends
/// before the seed is drawn from them. A stop moved past the draw would put the
/// draw into the rate.
///
/// The rule reads the order of three calls in one fn. It cannot read that the
/// stop covers every forward of the prefill, or that a draw hidden in a helper
/// sits after it.
#[test]
fn a_drafters_prefill_span_ends_before_it_draws_its_seed() {
    use crate::stop_scan::{calls_of, fns_of, sources_under};

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let sources = sources_under(&src);
    let mut problems = Vec::new();
    for file in DRAFTER_FILES {
        let Some((_, code)) = sources.iter().find(|(f, _)| f == file) else {
            problems.push(format!("{file} is not a source file"));
            continue;
        };
        let Some((_, text)) = fns_of(code).into_iter().find(|(name, _)| name == "prefill") else {
            problems.push(format!("{file} has no fn prefill"));
            continue;
        };
        let first = |callee: &str| calls_of(&text, callee).iter().map(|c| c.at).min();
        match (
            first("guard_verifier_prefill_logits"),
            first("elapsed"),
            first("seed_token"),
        ) {
            (Some(guard), Some(stop), Some(draw)) if guard < stop && stop < draw => {}
            (guard, stop, draw) => problems.push(format!(
                "{file}: guard at {guard:?}, stop at {stop:?}, draw at {draw:?}; the stop \
                 belongs after the guard and before the draw"
            )),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
