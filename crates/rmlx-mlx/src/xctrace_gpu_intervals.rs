//! Summary of the `metal-gpu-intervals` table: per-GPU-submission wall-clock
//! time and the CPU→GPU gap, aggregated per channel.
//!
//! Split from the parser in [`super`], which is table-generic; this half is the
//! only part that knows column names. Re-exported through `xctrace`, so callers
//! see a single module.

use std::collections::HashMap;
use std::fmt::Write as _;

use super::{for_each_row, schema_of, Result, XctraceError};

/// Schema this summary reads. Exported with
/// `--xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'`.
pub const GPU_INTERVALS_SCHEMA: &str = "metal-gpu-intervals";

/// Per-channel aggregate of GPU submissions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "a result record read field-by-field by reporting code"
)]
pub struct ChannelStats {
    /// GPU channel: `Compute`, `Vertex`, `Fragment`.
    pub channel: String,
    /// Number of submissions on this channel.
    pub submissions: u64,
    /// Summed GPU-busy time, nanoseconds.
    pub busy_ns: u64,
    /// Sorted submission durations, nanoseconds.
    durations_ns: Vec<u64>,
    /// Sorted CPU→GPU gaps, nanoseconds. Shorter than `durations_ns` when some
    /// rows carry a NULL `start-latency`.
    latencies_ns: Vec<u64>,
}

impl ChannelStats {
    /// Percentile of submission duration, nanoseconds. `p` in `0..=100`.
    #[must_use]
    pub fn duration_pct(&self, p: u8) -> u64 {
        percentile(&self.durations_ns, p)
    }

    /// Percentile of CPU→GPU gap, nanoseconds. `p` in `0..=100`.
    #[must_use]
    pub fn latency_pct(&self, p: u8) -> u64 {
        percentile(&self.latencies_ns, p)
    }

    /// How many rows carried a non-NULL `start-latency`.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        self.latencies_ns.len()
    }
}

fn percentile(sorted: &[u64], p: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let last = sorted.len() - 1;
    let idx = (usize::from(p.min(100)) * last).div_ceil(100);
    sorted.get(idx).copied().unwrap_or_default()
}

/// Whole-table summary, optionally narrowed to one process.
#[derive(Debug, Clone, Default)]
#[allow(
    clippy::exhaustive_structs,
    reason = "a result record read field-by-field by reporting code"
)]
pub struct GpuIntervalSummary {
    /// Rows in the table, before any process filter.
    pub rows_total: u64,
    /// Rows kept by the process filter.
    pub rows_matched: u64,
    /// Per-channel aggregates, busiest first.
    pub channels: Vec<ChannelStats>,
    /// Process display names seen, with row counts, busiest first.
    pub processes: Vec<(String, u64)>,
    /// Earliest submission start, nanoseconds.
    pub first_start_ns: u64,
    /// Latest submission end, nanoseconds.
    pub last_end_ns: u64,
}

impl GpuIntervalSummary {
    /// Wall-clock span the matched submissions cover, nanoseconds.
    #[must_use]
    pub fn span_ns(&self) -> u64 {
        self.last_end_ns.saturating_sub(self.first_start_ns)
    }

    /// Summed GPU-busy time across channels, nanoseconds. Channels run
    /// concurrently, so this can exceed [`Self::span_ns`].
    #[must_use]
    pub fn busy_ns(&self) -> u64 {
        self.channels.iter().map(|c| c.busy_ns).sum()
    }
}

/// Which submissions a summary counts.
#[derive(Debug, Clone, Copy, Default)]
#[allow(
    clippy::exhaustive_structs,
    reason = "a plain options struct built with literal syntax at every call site"
)]
pub struct SummaryFilter<'a> {
    /// Keep only rows whose `process` display form contains this — `rmlx`
    /// matches `rmlx (12345)`.
    pub process: Option<&'a str>,
    /// Drop submissions starting within this many milliseconds of the
    /// **matched process's own** first submission.
    ///
    /// `xctrace --attach` does not work for this template, so a recording can
    /// only start at process launch and its early GPU work is prefill. Reading
    /// a decode-window CPU→GPU gap without excluding that mixes the two.
    ///
    /// Measured from the matched process rather than from the trace, because
    /// weight load submits nothing to the GPU: its duration varies with page
    /// cache and model size and is simply absent from this table. A
    /// trace-relative skip would therefore have to be tuned per model and per
    /// run, and overshooting it silently discards the decode window.
    pub skip_ms: u64,
}

/// The refusal for "the filter matched nothing", built once so both entry
/// branches say the same thing.
///
/// Only reachable on a table that *has* rows: `for_each_row` refuses an empty
/// export itself, before either caller gets here. So a filter being present is
/// the whole decision — when one is, the rows exist and belong to somebody
/// else, and naming who is what separates "the recording failed" from "this
/// process was not in it".
fn no_rows_error(
    schema: String,
    rows_total: u64,
    process_filter: Option<&str>,
    processes: &[(String, u64)],
    after_skip: bool,
) -> XctraceError {
    let Some(want) = process_filter else {
        return XctraceError::NoRows { schema };
    };
    // Bounded: a recording holds a handful of processes, and the census is
    // read only on this error path.
    let seen = processes
        .iter()
        .map(|(name, n)| format!("{name} ({n} rows)"))
        .collect::<Vec<_>>()
        .join(", ");
    XctraceError::NoRowsForProcess {
        schema,
        rows_total,
        filter: want.to_owned(),
        processes: if seen.is_empty() {
            "<none>".to_owned()
        } else {
            seen
        },
        after_skip: if after_skip {
            " after the requested skip".to_owned()
        } else {
            String::new()
        },
    }
}

/// Parses a `metal-gpu-intervals` export and summarises it.
///
/// # Errors
/// [`XctraceError::WrongSchema`] when the export is a different table, plus any
/// parse error. A filter that matches no row is refused rather than summarised
/// as zeros — an empty summary is indistinguishable from a run that recorded
/// nothing, and reporting zeros for it is how a profiling harness lies. Which
/// refusal says which: [`XctraceError::NoRows`] when the table itself is empty,
/// [`XctraceError::NoRowsForProcess`] when it holds other processes' work and
/// none of this one's.
pub fn summarise_gpu_intervals(xml: &str, filter: SummaryFilter<'_>) -> Result<GpuIntervalSummary> {
    // Checked up front, from the header alone, so the wrong table refuses
    // identically whatever the options say. Left inside the row walk it would
    // be reached only after a full pass, and the `skip_ms > 0` branch — which
    // reads columns this schema may not have — would refuse with
    // `UnknownColumn` instead: same input, two different named failures.
    let schema = schema_of(xml)?;
    if schema.name != GPU_INTERVALS_SCHEMA {
        return Err(XctraceError::WrongSchema {
            expected: GPU_INTERVALS_SCHEMA.to_owned(),
            actual: schema.name,
        });
    }
    if filter.skip_ms == 0 {
        return summarise_from(xml, filter.process, 0);
    }
    // The origin is not known until the table has been read once. Rows are
    // cheap to revisit; buffering them all is the memory this parser streams to
    // avoid.
    let mut earliest = u64::MAX;
    let mut latest = 0u64;
    // Counted on the way past so the refusal below can name what the recording
    // did hold; a second pass purely to answer that would be a second reader of
    // the same table.
    let mut rows_total = 0u64;
    let mut processes: HashMap<String, u64> = HashMap::new();
    for_each_row(xml, |row| {
        rows_total += 1;
        let seen = row.fmt("process")?.unwrap_or("<unattributed>");
        match processes.get_mut(seen) {
            Some(n) => *n += 1,
            None => {
                processes.insert(seen.to_owned(), 1);
            }
        }
        if let Some(want) = filter.process {
            if !seen.contains(want) {
                return Ok(());
            }
        }
        // Required, not defaulted: one NULL `start` would pin `earliest` to 0,
        // and the origin below would collapse to a bare `skip_ms` — a
        // trace-relative floor wearing a process-relative label, with the
        // SkipExceedsSpan guard unable to fire.
        let start = row.u64_required("start")?;
        earliest = earliest.min(start);
        latest = latest.max(start.saturating_add(row.u64_required("duration")?));
        Ok(())
    })?;
    if earliest == u64::MAX {
        return Err(no_rows_error(
            GPU_INTERVALS_SCHEMA.to_owned(),
            rows_total,
            filter.process,
            &sorted_processes(processes),
            false,
        ));
    }
    let origin_ns = earliest.saturating_add(filter.skip_ms.saturating_mul(1_000_000));
    if origin_ns >= latest {
        // Naming the span it had to work with is the difference between a
        // usable error and one that reads as "the run recorded nothing".
        return Err(XctraceError::SkipExceedsSpan {
            skip_ms: filter.skip_ms,
            span_ms: latest.saturating_sub(earliest) / 1_000_000,
        });
    }
    summarise_from(xml, filter.process, origin_ns)
}

fn summarise_from(
    xml: &str,
    process_filter: Option<&str>,
    start_floor_ns: u64,
) -> Result<GpuIntervalSummary> {
    let mut summary = GpuIntervalSummary::default();
    let mut channels: HashMap<String, ChannelStats> = HashMap::new();
    let mut processes: HashMap<String, u64> = HashMap::new();
    let mut first_start = u64::MAX;

    let schema = for_each_row(xml, |row| {
        summary.rows_total += 1;
        // Looked up by borrow and only allocated on first sight: the trip count
        // here is rows in the export — hundreds of thousands — while the key
        // sets are a handful of processes and three channels.
        let process = row.fmt("process")?.unwrap_or("<unattributed>");
        match processes.get_mut(process) {
            Some(n) => *n += 1,
            None => {
                processes.insert(process.to_owned(), 1);
            }
        }
        let keep = process_filter.is_none_or(|want| process.contains(want));
        if !keep {
            return Ok(());
        }
        // See the note in summarise_gpu_intervals: a NULL here is an absence,
        // and read as 0 it would set first_start to 0 and inflate span_ns.
        let start = row.u64_required("start")?;
        if start < start_floor_ns {
            return Ok(());
        }
        summary.rows_matched += 1;

        let duration = row.u64_required("duration")?;
        first_start = first_start.min(start);
        summary.last_end_ns = summary.last_end_ns.max(start.saturating_add(duration));

        let channel = row.cell("channel-name")?.text().unwrap_or("<none>");
        let stats = match channels.get_mut(channel) {
            Some(stats) => stats,
            None => channels.entry(channel.to_owned()).or_insert(ChannelStats {
                channel: channel.to_owned(),
                ..ChannelStats::default()
            }),
        };
        stats.submissions += 1;
        stats.busy_ns = stats.busy_ns.saturating_add(duration);
        stats.durations_ns.push(duration);
        if let Some(latency) = row.u64("start-latency")? {
            stats.latencies_ns.push(latency);
        }
        Ok(())
    })?;

    if summary.rows_matched == 0 {
        return Err(no_rows_error(
            schema.name,
            summary.rows_total,
            process_filter,
            &sorted_processes(processes),
            start_floor_ns > 0,
        ));
    }

    summary.first_start_ns = if first_start == u64::MAX {
        0
    } else {
        first_start
    };
    summary.channels = channels.into_values().collect();
    for stats in &mut summary.channels {
        stats.durations_ns.sort_unstable();
        stats.latencies_ns.sort_unstable();
    }
    summary.channels.sort_by(|a, b| {
        b.busy_ns
            .cmp(&a.busy_ns)
            .then_with(|| a.channel.cmp(&b.channel))
    });
    summary.processes = sorted_processes(processes);
    Ok(summary)
}

/// Process census as a busiest-first list, ties broken by name so the order is
/// stable across runs (the refusal message quotes it).
fn sorted_processes(processes: HashMap<String, u64>) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = processes.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// Renders a summary as CSV — the form a regression script asserts on.
///
/// One row per channel; nanoseconds throughout, so no unit parsing downstream.
#[must_use]
pub fn summary_csv(summary: &GpuIntervalSummary) -> String {
    let mut out = String::from(
        "channel,submissions,busy_ns,dur_p50_ns,dur_p95_ns,\
         latency_samples,latency_p50_ns,latency_p95_ns,latency_max_ns\n",
    );
    for c in &summary.channels {
        // Empty, not 0, when nothing was measured: "no CPU->GPU gap" is the most
        // interesting result this table can report, and a script that forgets to
        // read latency_samples would otherwise record a fabricated best.
        let lat = |p: u8| {
            if c.latency_samples() == 0 {
                String::new()
            } else {
                c.latency_pct(p).to_string()
            }
        };
        // Writing to a String cannot fail; the Result is discarded deliberately.
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{}",
            c.channel,
            c.submissions,
            c.busy_ns,
            c.duration_pct(50),
            c.duration_pct(95),
            c.latency_samples(),
            lat(50),
            lat(95),
            lat(100),
        );
    }
    out
}
