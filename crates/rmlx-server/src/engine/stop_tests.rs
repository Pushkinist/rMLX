//! Source rules for the stop of a request in the blocking jobs of the server,
//! and the outcome of a stop as the retry envelope sees it. No model, no GPU.
//!
//! The property the rules hold: a job that holds the GPU permit runs inside
//! the one job runner, so the flag it is handed is the flag that the drop of
//! its request sets, and the job reads that flag and no other. A file on the
//! path of a job uses the flag in two ways only: it hands it to the stop helper
//! and it hands it on, bare, to the fn that takes it.
//!
//! There is one rule test for each job, so a change that moves one job into
//! the runner turns one test green. These rules read text.
//! `tests/gone_request.rs` reads on real models that a request that is gone
//! frees the permit.

use std::path::Path;
use std::sync::atomic::AtomicBool;

#[path = "../../../rmlx-models/tests/common/stop_scan.rs"]
pub(crate) mod stop_scan;

use stop_scan::{
    calls_a_flag_fn, calls_of, calls_off_the_rule, error_events, events_at_info_or_above,
    failure_arms_without_a_stop_arm, flag_fns_that_disagree, flag_uses_off_the_rule, fn_shapes,
    fns_of, loops_of, loud_events, names, sources_under, stops_by_the_rule, FnShape, Work, FLAG,
    STOP_HELPER, STOP_OUTCOME,
};

use crate::retry::{classify, RetryClass};

use super::types::{end_prompt_preparation, unless_stopped};

/// Read the stop of a request from `flag`, as a loop of a request does: one
/// call of the stop helper, with no error built by hand.
fn stop_read(flag: &AtomicBool) -> rmlx_core::Result<()> {
    rmlx_core::cancel::ensure_not_cancelled(flag)
}

/// A stop is an outcome of its own. A clear flag reads as no stop. A set flag
/// gives an error variant of its own, which the retry envelope does not run
/// again, and nothing is logged as an error: a request that is gone is a
/// normal event.
#[test]
fn a_stop_is_its_own_outcome_and_is_not_replayed() {
    assert!(stop_read(&AtomicBool::new(false)).is_ok());
    let mut read = Ok(());
    let logged_errors = error_events(|| read = stop_read(&AtomicBool::new(true)));
    let Err(stop) = read else {
        panic!("a set flag reads as no stop");
    };
    assert!(
        format!("{stop:?}").starts_with(STOP_OUTCOME),
        "the variant of a stop is `{stop:?}`"
    );
    assert!(!stop.is_migratable(), "a stop is replayed");
    assert!(matches!(classify(&stop), RetryClass::Fatal));
    assert_eq!(logged_errors, 0);
}

/// The fn that takes the GPU permit and runs a blocking job with it.
const RUNNER: &str = "run_admitted";

/// The calls that start work on another thread, or keep a blocking job on
/// this one.
const JOB_STARTS: [&str; 5] = [
    "spawn_blocking",
    "spawn",
    "spawn_local",
    "spawn_scoped",
    "block_in_place",
];

const CHAT_JOB: &str = "engine/arch_generator.rs";
const SPECULATIVE_JOB: &str = "engine/speculative.rs";
const SPEECH_JOB: &str = "audio/speech.rs";

fn crate_code() -> Vec<(String, String)> {
    sources_under(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
}

fn code_of_file(file: &str) -> String {
    crate_code()
        .into_iter()
        .find(|(f, _)| f == file)
        .map_or_else(|| panic!("no source file {file}"), |(_, code)| code)
}

/// The closure, or the future, of each job that `code` starts.
fn started_jobs(code: &str) -> Vec<&str> {
    JOB_STARTS
        .iter()
        .flat_map(|start| calls_of(code, start))
        .map(|call| call.args)
        .collect()
}

/// The reads of the permit of a request in `code` that are not in the
/// argument list of the runner. An assignment to the field is not a read.
fn permit_reads_past_the_runner(code: &str) -> usize {
    let field = ".gpu_admission";
    let runner = calls_of(code, RUNNER);
    code.match_indices(field)
        .filter(|(at, _)| {
            let after = code.get(at + field.len()..).unwrap_or_default();
            let longer_name = after.starts_with(|c: char| c.is_alphanumeric() || c == '_');
            let after = after.trim_start();
            let assigned = after.starts_with('=') && !after.starts_with("==");
            !longer_name && !assigned && !runner.iter().any(|call| call.holds(RUNNER, *at))
        })
        .count()
}

/// The jobs that `code` hands to the runner, each as its closure.
fn runner_jobs(code: &str) -> Vec<&str> {
    calls_of(code, RUNNER)
        .into_iter()
        .map(|call| call.args)
        .collect()
}

/// `true` when `job` binds the flag that the runner hands it to the name
/// `cancel`, which the flag rule reads, and not to `_cancel` or `_`.
fn reads_the_flag_of_its_request(job: &str) -> bool {
    job.split('|').nth(1).unwrap_or_default().trim() == FLAG
}

/// What keeps the job of a generator in `code` out of the one runner.
///
/// A send that fails sets nothing: the drop of the reply stream closes the
/// channel and sets the one flag, and the loop stops through the helper as for
/// every other stop. To ask whether the token channel is closed is not legal:
/// the server shutdown and a reply that nobody reads do not close it.
fn outside_the_runner(code: &str) -> Vec<&'static str> {
    let jobs = runner_jobs(code);
    [
        (
            permit_reads_past_the_runner(code) > 0,
            "it reads the permit of the request past the runner",
        ),
        (!started_jobs(code).is_empty(), "it starts a job for itself"),
        (jobs.is_empty(), "it hands no job to the runner"),
        (
            !jobs.iter().all(|job| reads_the_flag_of_its_request(job)),
            "its job does not read the flag of its request",
        ),
        (
            code.contains(".is_closed()"),
            "it asks whether the token channel is closed",
        ),
    ]
    .into_iter()
    .filter(|(wrong, _)| *wrong)
    .map(|(_, why)| why)
    .collect()
}

/// The job of a plain chat request holds the GPU permit inside the one runner
/// and reads the flag of its request.
#[test]
fn the_chat_job_runs_in_the_job_runner() {
    assert_eq!(
        outside_the_runner(&code_of_file(CHAT_JOB)),
        Vec::<&str>::new()
    );
}

/// The job of a speculative request holds the GPU permit inside the one
/// runner and reads the flag of its request.
#[test]
fn the_speculative_job_runs_in_the_job_runner() {
    assert_eq!(
        outside_the_runner(&code_of_file(SPECULATIVE_JOB)),
        Vec::<&str>::new()
    );
}

/// The speech job is in the runner. It must read the flag it is handed.
#[test]
fn the_speech_job_reads_the_flag_of_its_request() {
    let code = code_of_file(SPEECH_JOB);
    let jobs = runner_jobs(&code);
    assert_eq!(jobs.len(), 1, "the speech job");
    assert!(jobs.iter().all(|job| reads_the_flag_of_its_request(job)));
}

/// The speech job decides the stop from the error of its synthesis, as the
/// generation jobs do from the result of theirs: it does not read the flag
/// again. A synthesis that ran to its end keeps its audio and its rows also
/// when its request went away after it.
#[test]
fn the_speech_job_does_not_read_the_flag_after_its_synthesis() {
    let reads = calls_of(&code_of_file(SPEECH_JOB), STOP_HELPER).len();
    assert_eq!(reads, 0);
}

/// The speech job logs the one stop event through [`log_stopped`] at one place,
/// and does not hand a result to [`OUTCOME`]: the error of its synthesis is a
/// `TtsError`.
#[test]
fn the_speech_job_logs_the_stop_once_and_does_not_use_the_outcome_fn() {
    let code = code_of_file(SPEECH_JOB);
    assert_eq!(calls_of(&code, "log_stopped").len(), 1);
    assert_eq!(calls_of(&code, OUTCOME).len(), 0);
}

/// Outside the three jobs above, the crate already keeps the rules: every job
/// of the runner reads its flag, and no code reads the permit of a request
/// past the runner.
#[test]
fn every_other_file_keeps_the_rules_of_the_runner() {
    let wrong: Vec<String> = crate_code()
        .into_iter()
        .filter(|(file, _)| ![CHAT_JOB, SPECULATIVE_JOB, SPEECH_JOB].contains(&file.as_str()))
        .filter(|(_, code)| {
            permit_reads_past_the_runner(code) > 0
                || !runner_jobs(code)
                    .iter()
                    .all(|job| reads_the_flag_of_its_request(job))
        })
        .map(|(file, _)| file)
        .collect();
    assert_eq!(wrong, Vec::<String>::new());
}

/// The fns of `sources` that take the GPU permit and start a job with it,
/// each as `file::fn`.
fn job_runners(sources: &[(String, String)]) -> Vec<String> {
    sources
        .iter()
        .flat_map(|(file, code)| {
            fns_of(code)
                .into_iter()
                .filter(|(_, text)| {
                    let signature = text.split('{').next().unwrap_or_default();
                    names(signature, "GpuAdmission") && !started_jobs(text).is_empty()
                })
                .map(move |(name, _)| format!("{file}::{name}"))
        })
        .collect()
}

/// One fn of the crate takes the GPU permit and starts a blocking job with it:
/// the runner, which hands each job the flag that the drop of its request
/// sets.
#[test]
fn one_fn_starts_a_blocking_job_with_the_gpu_permit() {
    let runners = job_runners(&crate_code());
    assert_eq!(runners.len(), 1, "{runners:?}");
    assert!(
        runners.iter().all(|r| r.ends_with(&format!("::{RUNNER}"))),
        "{runners:?}"
    );
}

/// `true` when `runner` hands its job a clone of the flag that its drop guard
/// holds, and no other.
fn hands_over_the_flag_of_its_guard(runner: &str) -> bool {
    let calls = calls_of(runner, "job");
    runner.matches("let flag").count() == 1
        && runner.contains("let flag = Arc::clone(&cancel_on_drop.0);")
        && calls.len() == 1
        && calls.iter().all(|call| call.args.trim() == "&flag")
}

/// The runner hands its job the flag that the drop of the reply sets. Every
/// other file on the path of a job gets that flag as a parameter and uses it
/// by the flag rule below.
#[test]
fn the_job_runner_hands_its_job_the_flag_of_its_guard() {
    let runners: Vec<String> = crate_code()
        .iter()
        .flat_map(|(_, code)| fns_of(code))
        .filter(|(name, _)| name == RUNNER)
        .map(|(_, text)| text)
        .collect();
    assert_eq!(runners.len(), 1);
    assert!(hands_over_the_flag_of_its_guard(&runners[0]));
}

/// The source of a sibling crate that a job of the server calls.
fn sibling_sources(name: &str) -> Vec<(String, String)> {
    sources_under(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(name)
            .join("src"),
    )
}

/// The files on the path of a job: every file of the engine and every file that
/// hands a job to the runner.
fn is_on_the_path_of_a_job(file: &str, code: &str) -> bool {
    file.starts_with("engine/") || !runner_jobs(code).is_empty()
}

/// In every file on the path of a job, and in every file that takes the flag or
/// calls a fn that takes it, the name `cancel` is used as the argument of the
/// stop helper and as the bare argument of a call, and in no other way; and
/// each call of a fn that takes the flag hands on the `cancel` of its caller at
/// the place of the flag. So a fresh flag by any maker, a `static`, a binding of
/// the flag to another name and a read of it are all red. The fns that the
/// server calls are its own and those of the model and audio crates.
#[test]
fn the_flag_of_a_request_is_only_handed_on_and_read_by_the_helper() {
    let sources = crate_code();
    let mut shapes: Vec<FnShape> = sources
        .iter()
        .flat_map(|(_, code)| fn_shapes(code))
        .collect();
    for sibling in ["rmlx-models", "rmlx-audio"] {
        shapes.extend(
            sibling_sources(sibling)
                .iter()
                .flat_map(|(_, code)| fn_shapes(code))
                .filter(|shape| shape.flag.is_some()),
        );
    }
    let scope: Vec<&(String, String)> = sources
        .iter()
        .filter(|(file, code)| {
            is_on_the_path_of_a_job(file, code)
                || fn_shapes(code).iter().any(|shape| shape.flag.is_some())
                || calls_a_flag_fn(code, &shapes)
        })
        .collect();
    assert_eq!(flag_fns_that_disagree(&shapes), Vec::<String>::new());
    assert!(scope.len() > 5, "the scan reads {} files", scope.len());
    let mut wrong = Vec::new();
    for (file, code) in scope {
        let lines = flag_uses_off_the_rule(code)
            .into_iter()
            .chain(calls_off_the_rule(code, &shapes));
        wrong.extend(lines.map(|line| format!("{file}: {line}")));
    }
    assert_eq!(wrong, Vec::<String>::new());
}

/// In a job of the server, a `match` arm `Err(e)` that logs `e` at `warn` or
/// `error` has an arm `Err(Error::Cancelled)` before it. A stop is not a
/// failure: it is one `info` event, and it sends no error. The arms are read in
/// the closure of each job that is handed to the runner, so the load of a model
/// is not read. A site that prepares the prompt ends through
/// `end_prompt_preparation`, which does this for every stage.
#[test]
fn a_failure_arm_in_a_job_has_a_stop_arm_before_it() {
    let wrong: Vec<String> = crate_code()
        .into_iter()
        .filter(|(file, code)| is_on_the_path_of_a_job(file, code))
        .flat_map(|(file, code)| {
            runner_jobs(&code)
                .into_iter()
                .flat_map(failure_arms_without_a_stop_arm)
                .map(|line| format!("{file}: {line}"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(wrong, Vec::<String>::new());
}

/// A stop in the preparation of a prompt (the load of an image, the decode of
/// an audio clip) is the one stop event: no `warn`, no `error`, nothing sent to
/// the reply. Any other error is one `warn` event and is sent.
#[test]
fn a_stop_in_the_preparation_of_a_prompt_is_the_stop_event_and_sends_nothing() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let (mut loud, mut events) = (0, 0);
    loud += loud_events(|| end_prompt_preparation(&tx, rmlx_core::Error::Cancelled, "image"));
    events += events_at_info_or_above(|| {
        end_prompt_preparation(&tx, rmlx_core::Error::Cancelled, "image");
    });
    assert_eq!((loud, events), (0, 1));
    assert!(rx.try_recv().is_err(), "a stop sent an item");

    let failure = || rmlx_core::Error::Other("the image is not an image".to_owned());
    assert_eq!(
        loud_events(|| end_prompt_preparation(&tx, failure(), "image")),
        1
    );
    assert!(
        matches!(rx.try_recv(), Ok(Err(_))),
        "a failure was not sent"
    );
}

/// A generation that returns the stop outcome is the one stop event and gives
/// the job nothing to report. Any other result goes to the job as it is and
/// logs nothing here: a generation that ran to its end keeps its rows also when
/// its request went away after it.
#[test]
fn a_job_reports_every_result_of_a_generation_but_the_stop() {
    let mut stopped = Some(Ok(()));
    let (loud, events) = (
        loud_events(|| stopped = unless_stopped(Err(rmlx_core::Error::Cancelled), 3)),
        events_at_info_or_above(|| {
            unless_stopped::<()>(Err(rmlx_core::Error::Cancelled), 3);
        }),
    );
    assert!(stopped.is_none(), "a stop was handed to the job");
    assert_eq!((loud, events), (0, 1));

    let mut ended = None;
    let events = events_at_info_or_above(|| ended = unless_stopped(Ok(7u32), 7));
    assert!(matches!(ended, Some(Ok(7))), "{ended:?}");
    assert_eq!(events, 0);

    let mut failed = None;
    let failure = rmlx_core::Error::Model("a scripted failure".to_owned());
    let events = events_at_info_or_above(|| failed = unless_stopped::<()>(Err(failure), 0));
    assert!(
        matches!(failed, Some(Err(rmlx_core::Error::Model(_)))),
        "{failed:?}"
    );
    assert_eq!(events, 0);
}

/// The fn that turns the result of a generation into what its job reports.
const OUTCOME: &str = "unless_stopped";

/// What keeps the job in `code` from deciding the stop from the result of its
/// generation: it must hand that result to [`OUTCOME`] once, and read the flag
/// only for the decode of an audio clip, which has no result to read.
///
/// This rule holds the chat and speculative jobs. The speech job is the
/// exception: its synthesis fails with a `TtsError`, which [`OUTCOME`] cannot
/// read, so the speech job has its own rule, in
/// `the_speech_job_logs_the_stop_once_and_does_not_use_the_outcome_fn`.
fn reads_the_stop_off_its_result(code: &str) -> Vec<&'static str> {
    let jobs = runner_jobs(code);
    let audio_decodes = calls_of(code, "build_audio_prompt");
    let stray_reads = calls_of(code, STOP_HELPER)
        .iter()
        .filter(|read| {
            !audio_decodes
                .iter()
                .any(|decode| decode.holds("build_audio_prompt", read.at))
        })
        .count();
    [
        (jobs.len() != 1, "it hands the runner other than one job"),
        (
            !jobs.iter().all(|job| calls_of(job, OUTCOME).len() == 1),
            "its job does not hand the result of its generation to the outcome fn once",
        ),
        (
            stray_reads > 0,
            "it reads the flag past the decode of an audio clip",
        ),
        (
            !calls_of(code, "log_stopped").is_empty(),
            "it logs the stop event for itself",
        ),
    ]
    .into_iter()
    .filter(|(wrong, _)| *wrong)
    .map(|(_, why)| why)
    .collect()
}

/// The chat job and the speculative job decide that their request stopped from
/// the result of the generation, and not from the flag: a generation that ran
/// to its end before its request went away is reported as one that ended.
#[test]
fn each_generation_job_reads_the_stop_off_its_result() {
    for file in [CHAT_JOB, SPECULATIVE_JOB] {
        assert_eq!(
            reads_the_stop_off_its_result(&code_of_file(file)),
            Vec::<&str>::new(),
            "{file}"
        );
    }
}

/// A loop of a job that loads one image source per pass reads the stop first,
/// by the one rule. A request that is gone then fetches no further image: a
/// remote image takes up to 20 s. The loops are found from the work they do,
/// so a new loop over images is read without a list.
#[test]
fn every_loop_that_loads_an_image_reads_the_stop() {
    let loads: Vec<(String, stop_scan::Loop)> = crate_code()
        .into_iter()
        .filter(|(file, code)| is_on_the_path_of_a_job(file, code))
        .flat_map(|(file, code)| {
            loops_of(&code)
                .into_iter()
                .filter(|l| l.work.contains(&Work::Load))
                .map(move |l| (file.clone(), l))
        })
        .collect();
    assert!(!loads.is_empty(), "no loop that loads an image");
    let wrong: Vec<String> = loads
        .iter()
        .filter(|(_, l)| !stops_by_the_rule(&l.code))
        .map(|(file, l)| l.key(file))
        .collect();
    assert_eq!(wrong, Vec::<String>::new());
}

/// The jobs of `sources` whose closure or future names `held`, each as its
/// file. A file with two such jobs is listed twice.
fn jobs_that_name(sources: &[(String, String)], held: &str) -> Vec<String> {
    sources
        .iter()
        .flat_map(|(file, code)| {
            started_jobs(code)
                .into_iter()
                .filter(|job| names(job, held))
                .map(|_| file.clone())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// No task holds a call of the runner. The flag of a job is set when the
/// future of the runner is dropped; inside a task of its own that future
/// outlives the request, and the flag is never set.
#[test]
fn no_task_holds_a_job_of_the_runner() {
    assert_eq!(jobs_that_name(&crate_code(), RUNNER), Vec::<String>::new());
}

/// One job locks the GPU gate with no permit and outside the runner: the
/// embeddings job. It has no stop, and its stop is not part of this rule. A
/// second such job, in this file or in another, is a job with no stop.
#[test]
fn one_embeddings_job_locks_the_gpu_gate_outside_the_runner() {
    assert_eq!(jobs_that_name(&crate_code(), "gpu_gate"), ["embeddings.rs"]);
}

/// The error of a tower forward reaches its caller with its own class. A
/// wrapper that formats it into a general error makes a `Model` error of the
/// tower `Other`, which the retry envelope runs again.
#[test]
fn the_error_of_a_tower_forward_is_not_wrapped() {
    let wrapped: Vec<String> = crate_code()
        .into_iter()
        .filter(|(file, _)| file.starts_with("engine/"))
        .filter(|(_, code)| {
            calls_of(code, "forward")
                .iter()
                .any(|call| call.after.trim_start().starts_with(".map_err("))
        })
        .map(|(file, _)| file)
        .collect();
    assert_eq!(wrapped, Vec::<String>::new());
}

#[test]
fn the_job_scan_reads_jobs_and_field_reads_and_signatures() {
    let sources = |code: &str| vec![("a.rs".to_owned(), code.to_owned())];
    let runner = "async fn run_admitted<T, F>(guard: GpuAdmission, job: F) -> T {\n    \
                  tokio::task::spawn_blocking(move || {\n        let _guard = guard;\n    })\n}";
    assert_eq!(job_runners(&sources(runner)), ["a.rs::run_admitted"]);

    // The permit of a request is read only in the argument list of the runner,
    // under any name of a local variable.
    let renamed = "let admission = req.gpu_admission;\n\
                   tokio::task::spawn_blocking(move || {\n    let _held = admission;\n});";
    let through = "run_admitted(req.gpu_admission, move |cancel| decode(cancel))";
    let assigned = "gen_req.gpu_admission = Some(guard);\nif a.gpu_admission == b {}";
    assert_eq!(permit_reads_past_the_runner(renamed), 1);
    assert_eq!(permit_reads_past_the_runner(through), 0);
    assert_eq!(permit_reads_past_the_runner(assigned), 1);
    assert!(outside_the_runner(renamed).contains(&"it starts a job for itself"));
    assert_eq!(outside_the_runner(through), Vec::<&str>::new());

    // A task that holds a call of the runner, and each kind of job start.
    let detached = "tokio::spawn(run_admitted(guard, move |cancel| decode(cancel)));";
    assert_eq!(jobs_that_name(&sources(detached), RUNNER), ["a.rs"]);
    for start in [
        "std::thread::spawn",
        "scope.spawn_scoped",
        "tokio::task::block_in_place",
    ] {
        let job = format!("{start}(move || compute(&gpu_gate));");
        assert_eq!(
            jobs_that_name(&sources(&job), "gpu_gate"),
            ["a.rs"],
            "{start}"
        );
    }
    let two = "spawn_blocking(move || a(&gpu_gate));\nspawn_blocking(move || b(&gpu_gate));";
    assert_eq!(jobs_that_name(&sources(two), "gpu_gate"), ["a.rs", "a.rs"]);

    let reads = |job: &str| {
        runner_jobs(job)
            .iter()
            .all(|j| reads_the_flag_of_its_request(j))
    };
    assert!(reads(
        "run_admitted(guard, move |cancel| synthesize(&input, cancel))"
    ));
    assert!(!reads(
        "run_admitted(guard, move |_cancel| synthesize(&input))"
    ));
    assert!(!reads("run_admitted(guard, move |_| synthesize(&input))"));

    // A runner that hands its job a flag of its own, or a clone of another one.
    let runner = "fn run_admitted() {\n    let cancel_on_drop = CancelOnDrop(Arc::new(f()));\n    \
                  let flag = Arc::clone(&cancel_on_drop.0);\n    spawn_blocking(move || {\n        \
                  job(&flag)\n    })\n}";
    assert!(hands_over_the_flag_of_its_guard(runner));
    for wrong in [
        runner.replace("job(&flag)", "job(&AtomicBool::new(false))"),
        runner.replace("job(&flag)", "job(&flag)\n    job(&flag)"),
        runner.replace(
            "Arc::clone(&cancel_on_drop.0)",
            "Arc::new(AtomicBool::new(false))",
        ),
    ] {
        assert!(!hands_over_the_flag_of_its_guard(&wrong), "{wrong}");
    }

    // An arm that logs a failure has a stop arm before it, or it is red.
    let arms = |code: &str| failure_arms_without_a_stop_arm(code).len();
    let with_stop = "match f() {\n    Ok(x) => x,\n    Err(Error::Cancelled) => {\n        stopped();\n    }\n    \
                     Err(e) => {\n        tracing::warn!(error = %e, \"failed\");\n    }\n}";
    let without = "match f() {\n    Ok(x) => x,\n    Err(e) => {\n        tracing::warn!(error = %e, \"failed\");\n    }\n}";
    let silent = "match f() {\n    Ok(x) => x,\n    Err(e) => {\n        send(e);\n    }\n}";
    let debug = "match f() {\n    Ok(x) => x,\n    Err(e) => tracing::debug!(error = %e),\n}";
    let one_line =
        "match f() {\n    Ok(x) => x,\n    Err(e) => tracing::error!(error = %e, \"x\"),\n}";
    assert_eq!(
        (
            arms(with_stop),
            arms(without),
            arms(silent),
            arms(debug),
            arms(one_line)
        ),
        (0, 1, 0, 0, 1)
    );

    // A loop over images reads the stop first, or it is red; a loop that loads
    // nothing is not read.
    let loop_reading = |body: &str| {
        let code = format!("fn f() {{\n    for src in sources {{\n        {body}\n    }}\n}}");
        loops_of(&code)
            .iter()
            .filter(|l| l.work.contains(&Work::Load))
            .map(|l| stops_by_the_rule(&l.code))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        loop_reading("ensure_not_cancelled(cancel)?;\n        load_image(src)?;"),
        [true]
    );
    assert_eq!(loop_reading("load_image(src)?;"), [false]);
    assert_eq!(
        loop_reading("load_image(src)?;\n        ensure_not_cancelled(cancel)?;"),
        [false]
    );
    assert_eq!(loop_reading("rmlx_server_load_image(src)?;"), [false]);
    assert_eq!(loop_reading("log(src);"), Vec::<bool>::new());
}
