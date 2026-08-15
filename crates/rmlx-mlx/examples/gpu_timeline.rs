//! Summarises a `metal-gpu-intervals` export from `xcrun xctrace export`.
//!
//! An example rather than a subcommand: this is profiling tooling, and the
//! shipped binary stays one artifact with no dev-only surface in it.
//!
//! ```sh
//! cargo run -q -p rmlx-mlx --features metal-capture --example gpu_timeline -- \
//!     --input gpu.xml --process rmlx --csv gpu.csv
//! ```
//!
//! `scripts/mst_capture.sh` drives record + export + this, and is the intended
//! entry point.

// A user-facing CLI tool: stdout IS its output, which CLAUDE.md permits for
// operator-facing commands.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use rmlx_mlx::xctrace::{summarise_gpu_intervals, summary_csv, GpuIntervalSummary, SummaryFilter};
use std::process::ExitCode;

fn usage() -> &'static str {
    "usage: gpu_timeline --input <gpu.xml> [--process <substring>] [--skip-ms N] [--csv <path>]\n\
     \n\
       --input <path>       XML from `xctrace export --xpath ...metal-gpu-intervals...`\n\
       --process <substr>   keep only submissions attributed to a matching process\n\
       --skip-ms N          ignore the first N ms of the matched process's own GPU\n\
                            work, i.e. prefill — a recording can only start at\n\
                            launch (--attach is broken for this template), and\n\
                            weight load submits nothing so it leaves no rows\n\
       --csv <path>         also write the per-channel table as CSV\n"
}

struct Args {
    input: String,
    process: Option<String>,
    skip_ms: u64,
    csv: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut process = None;
    let mut skip_ms = 0u64;
    let mut csv = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--input" => input = Some(value("--input")?),
            "--process" => process = Some(value("--process")?),
            "--skip-ms" => {
                let raw = value("--skip-ms")?;
                skip_ms = raw
                    .parse()
                    .map_err(|_| format!("--skip-ms expects an integer, got {raw:?}"))?;
            }
            "--csv" => csv = Some(value("--csv")?),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Args {
        input: input.ok_or_else(|| "--input is required".to_owned())?,
        process,
        skip_ms,
        csv,
    })
}

/// Nanoseconds as milliseconds, three decimals — the scale a decode step lives at.
fn ms(ns: u64) -> String {
    #[allow(
        clippy::cast_precision_loss,
        reason = "display only; a trace is bounded by --time-limit so ns stays far inside f64's exact integer range"
    )]
    let millis = ns as f64 / 1.0e6;
    format!("{millis:.3}")
}

/// Submissions per second over the matched span. The independent cross-check
/// against a decode rate: at one GPU kick per decode step this tracks TPS, and
/// a wild disagreement means the window is not the window it was thought to be.
fn per_second(count: u64, span_ns: u64) -> String {
    if span_ns == 0 {
        return "n/a".to_owned();
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "display only; counts and a --time-limit-bounded span stay far inside f64's exact integer range"
    )]
    let rate = count as f64 * 1.0e9 / span_ns as f64;
    format!("{rate:.1}")
}

fn report(summary: &GpuIntervalSummary, args: &Args) {
    println!(
        "rows: {} total, {} matched{}{}",
        summary.rows_total,
        summary.rows_matched,
        args.process
            .as_deref()
            .map_or_else(String::new, |p| format!(" (process contains {p:?})")),
        if args.skip_ms > 0 {
            format!(" (first {} ms skipped)", args.skip_ms)
        } else {
            String::new()
        }
    );
    println!(
        "span: {} ms   gpu busy: {} ms (channels overlap)   {} submissions/s",
        ms(summary.span_ns()),
        ms(summary.busy_ns()),
        per_second(summary.rows_matched, summary.span_ns())
    );
    println!();
    println!(
        "{:<10} {:>8} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "channel", "subs", "busy ms", "dur p50", "dur p95", "lat p50", "lat p95", "lat max"
    );
    for c in &summary.channels {
        println!(
            "{:<10} {:>8} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            c.channel,
            c.submissions,
            ms(c.busy_ns),
            ms(c.duration_pct(50)),
            ms(c.duration_pct(95)),
            ms(c.latency_pct(50)),
            ms(c.latency_pct(95)),
            ms(c.latency_pct(100)),
        );
        if c.latency_samples() < usize::try_from(c.submissions).unwrap_or(usize::MAX) {
            // Said out loud rather than folded into the percentiles: a NULL
            // start-latency is "not measured", and averaging it in as zero
            // would understate the very gap this table is read for.
            println!(
                "{:<10} {:>8} {}",
                "",
                "",
                format_args!(
                    "note: {} of {} submissions carry no start-latency",
                    c.submissions.saturating_sub(c.latency_samples() as u64),
                    c.submissions
                )
            );
        }
    }
    println!();
    println!("processes seen:");
    for (name, count) in summary.processes.iter().take(8) {
        println!("  {count:>8}  {name}");
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}");
            }
            eprint!("{}", usage());
            return ExitCode::from(2);
        }
    };

    let xml = match std::fs::read_to_string(&args.input) {
        Ok(xml) => xml,
        Err(err) => {
            eprintln!("error: reading {}: {err}", args.input);
            return ExitCode::FAILURE;
        }
    };

    let filter = SummaryFilter {
        process: args.process.as_deref(),
        skip_ms: args.skip_ms,
    };
    let summary = match summarise_gpu_intervals(&xml, filter) {
        Ok(summary) => summary,
        Err(err) => {
            // The parser refuses a layout it cannot align rather than emitting
            // shifted numbers, so this path is a real answer, not a nuisance.
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    report(&summary, &args);

    if let Some(path) = args.csv.as_deref() {
        if let Err(err) = std::fs::write(path, summary_csv(&summary)) {
            eprintln!("error: writing {path}: {err}");
            return ExitCode::FAILURE;
        }
        println!();
        println!("csv: {path}");
    }
    ExitCode::SUCCESS
}
