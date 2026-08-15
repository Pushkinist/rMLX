//! Tests for the `xctrace export` XML parser.
//!
//! The fixtures below are cut down from a real `metal-gpu-intervals` export so
//! they carry the encoding's three traps — a `<sentinel/>` in a middle column,
//! `id`/`ref` back-references, and an id minted inside a nested subtree that a
//! later column refers to. Each guard has a paired fixture that must be
//! REFUSED: a parser that accepts them produces plausible, shifted numbers, and
//! that is the failure this module exists to make impossible.

use super::{
    for_each_row, summarise_gpu_intervals, summary_csv, Cell, SummaryFilter, XctraceError,
    GPU_INTERVALS_SCHEMA,
};

/// Everything, unfiltered.
fn all() -> SummaryFilter<'static> {
    SummaryFilter::default()
}

/// Only submissions attributed to a process whose name contains `p`.
fn only(p: &str) -> SummaryFilter<'_> {
    SummaryFilter {
        process: Some(p),
        skip_ms: 0,
    }
}

/// Four columns, enough to place a NULL in the middle and reference across.
const SCHEMA: &str = "\
<schema name=\"metal-gpu-intervals\">\
<col><mnemonic>start</mnemonic><name>Creation</name><engineering-type>start-time</engineering-type></col>\
<col><mnemonic>duration</mnemonic><name>Duration</name><engineering-type>duration</engineering-type></col>\
<col><mnemonic>start-latency</mnemonic><name>CPU to GPU Latency</name><engineering-type>duration</engineering-type></col>\
<col><mnemonic>channel-name</mnemonic><name>Channel Name</name><engineering-type>gpu-channel-name</engineering-type></col>\
<col><mnemonic>process</mnemonic><name>Process</name><engineering-type>process</engineering-type></col>\
</schema>";

fn doc(rows: &str) -> String {
    format!("<?xml version=\"1.0\"?><trace-query-result><node xpath='x'>{SCHEMA}{rows}</node></trace-query-result>")
}

/// Row 1 defines ids; row 2 reaches back to them and puts NULL in the middle.
/// Row 2's `process` cell refers to an id first minted INSIDE row 1's
/// `<process>` subtree, which is the case that breaks a top-level-only index.
const ROWS_HAPPY: &str = "\
<row>\
<start-time id=\"1\" fmt=\"00:00.001\">1000</start-time>\
<duration id=\"2\" fmt=\"10.00 µs\">10000</duration>\
<duration id=\"3\" fmt=\"5.00 µs\">5000</duration>\
<gpu-channel-name id=\"4\" fmt=\"Compute\">Compute</gpu-channel-name>\
<process id=\"5\" fmt=\"rmlx (99)\"><pid id=\"6\" fmt=\"99\">99</pid></process>\
</row>\
<row>\
<start-time id=\"7\" fmt=\"00:00.002\">2000</start-time>\
<duration ref=\"2\"/>\
<sentinel/>\
<gpu-channel-name ref=\"4\"/>\
<process ref=\"5\"/>\
</row>";

fn collect(xml: &str) -> Result<Vec<Vec<Cell>>, XctraceError> {
    let mut out = Vec::new();
    for_each_row(xml, |row| {
        let mut cells = Vec::new();
        for column in [
            "start",
            "duration",
            "start-latency",
            "channel-name",
            "process",
        ] {
            cells.push(row.cell(column)?.clone());
        }
        out.push(cells);
        Ok(())
    })?;
    Ok(out)
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn back_references_and_a_middle_sentinel_keep_every_column_in_place() {
    let rows = collect(&doc(ROWS_HAPPY)).unwrap();
    assert_eq!(rows.len(), 2);

    // Row 2 is the interesting one: a ref, a NULL third column, and a ref into
    // a nested subtree. The duration must be row 1's value, NOT shifted.
    let row2 = rows.get(1).unwrap();
    assert_eq!(row2.first().unwrap().text(), Some("2000"), "start");
    assert_eq!(
        row2.get(1).unwrap().text(),
        Some("10000"),
        "duration resolved through ref"
    );
    assert_eq!(
        row2.get(2).unwrap(),
        &Cell::Null,
        "sentinel must stay NULL, not absorb the next column"
    );
    assert_eq!(
        row2.get(3).unwrap().text(),
        Some("Compute"),
        "channel must not shift left into the sentinel slot"
    );
    assert_eq!(
        row2.get(4).unwrap().fmt(),
        Some("rmlx (99)"),
        "ref into a nested subtree must resolve"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn nested_ids_do_not_consume_column_slots() {
    // Row 1's <process> holds a nested <pid> carrying its own id. If nesting
    // were counted as a column the row would have 6 cells, not 5.
    let rows = collect(&doc(ROWS_HAPPY)).unwrap();
    assert_eq!(rows.first().unwrap().len(), 5);
    assert_eq!(
        rows.first().unwrap().get(4).unwrap().fmt(),
        Some("rmlx (99)")
    );
}

// --- the refusals -----------------------------------------------------------

#[test]
fn a_row_short_of_a_column_is_refused_not_padded() {
    // The sentinel is simply gone — the exact mutation a "skip NULLs" parser
    // performs on itself. Four cells against five columns.
    let rows = "<row><start-time fmt=\"a\">1</start-time><duration fmt=\"b\">2</duration>\
                <gpu-channel-name fmt=\"Compute\">Compute</gpu-channel-name>\
                <process fmt=\"rmlx (1)\"></process></row>";
    let err = collect(&doc(rows)).expect_err("a short row must be refused");
    assert!(
        matches!(
            err,
            XctraceError::ColumnCountMismatch {
                expected: 5,
                actual: 4,
                ..
            }
        ),
        "got {err}"
    );
}

#[test]
fn a_column_shifted_by_one_is_caught_by_its_type() {
    // Five cells, so the count check passes — but the sentinel was dropped and
    // an extra appended, sliding channel-name into the start-latency slot.
    // Only the type invariant can see this, and it is the shape that yields
    // believable wrong numbers.
    let rows = "<row><start-time fmt=\"a\">1</start-time><duration fmt=\"b\">2</duration>\
                <gpu-channel-name fmt=\"Compute\">Compute</gpu-channel-name>\
                <process fmt=\"rmlx (1)\"></process>\
                <duration fmt=\"c\">3</duration></row>";
    let err = collect(&doc(rows)).expect_err("a shifted row must be refused");
    assert!(
        matches!(err, XctraceError::ColumnTypeMismatch { column: 2, .. }),
        "got {err}"
    );
    // Asserted through Display, because the message is what an operator acts
    // on: it has to name the column that moved and both types, or "the layout
    // changed" is untriageable.
    let msg = err.to_string();
    assert!(msg.contains("column 2 (start-latency)"), "got {msg}");
    assert!(
        msg.contains("expected <duration> but found <gpu-channel-name>"),
        "got {msg}"
    );
}

#[test]
fn an_unresolvable_ref_is_refused_not_read_as_empty() {
    let rows = "<row><start-time fmt=\"a\">1</start-time><duration ref=\"404\"/>\
                <sentinel/><gpu-channel-name fmt=\"Compute\">Compute</gpu-channel-name>\
                <process fmt=\"rmlx (1)\"></process></row>";
    let err = collect(&doc(rows)).expect_err("a dangling ref must be refused");
    assert!(
        matches!(err, XctraceError::UnresolvedRef { ref id, .. } if id == "404"),
        "got {err}"
    );
}

#[test]
fn a_ref_to_a_still_open_element_is_refused() {
    // Interning at close is what makes this loud: interning at open would hand
    // back an element with no text yet and read as an empty duration.
    let rows = "<row><start-time id=\"9\" fmt=\"a\">1</start-time>\
                <duration fmt=\"b\">2</duration><sentinel/>\
                <gpu-channel-name fmt=\"Compute\">Compute</gpu-channel-name>\
                <process id=\"5\" fmt=\"rmlx (1)\"><pid ref=\"5\"/></process></row>";
    let err = collect(&doc(rows)).expect_err("a self-referential ref must be refused");
    assert!(
        matches!(err, XctraceError::UnresolvedRef { ref id, .. } if id == "5"),
        "got {err}"
    );
}

#[test]
fn an_export_with_no_schema_is_refused() {
    let err = for_each_row("<trace-query-result></trace-query-result>", |_| Ok(()))
        .expect_err("a schema-less document must be refused");
    assert!(matches!(err, XctraceError::MissingSchema), "got {err}");
}

#[test]
fn a_schema_with_no_rows_is_refused_rather_than_summarised_as_zero() {
    let err = for_each_row(&doc(""), |_| Ok(())).expect_err("an empty table must be refused");
    assert!(matches!(err, XctraceError::NoRows { .. }), "got {err}");
}

#[test]
fn a_non_numeric_duration_is_refused_rather_than_defaulted_to_zero() {
    let rows =
        "<row><start-time fmt=\"a\">1</start-time><duration fmt=\"b\">not-a-number</duration>\
                <sentinel/><gpu-channel-name fmt=\"Compute\">Compute</gpu-channel-name>\
                <process fmt=\"rmlx (1)\"></process></row>";
    let mut seen = None;
    let err = for_each_row(&doc(rows), |row| {
        seen = Some(row.u64("duration")?);
        Ok(())
    })
    .expect_err("a non-numeric duration must be refused");
    assert!(
        matches!(err, XctraceError::NotAnInteger { ref mnemonic, .. } if mnemonic == "duration"),
        "got {err}"
    );
}

#[test]
fn asking_for_a_column_the_schema_lacks_is_an_error_not_a_none() {
    let err = for_each_row(&doc(ROWS_HAPPY), |row| {
        row.cell("occupancy")?;
        Ok(())
    })
    .expect_err("an unknown column must be refused");
    assert!(
        matches!(err, XctraceError::UnknownColumn { ref mnemonic, .. } if mnemonic == "occupancy"),
        "got {err}"
    );
}

#[test]
fn a_different_table_is_refused_by_name() {
    let xml = doc(ROWS_HAPPY).replace("metal-gpu-intervals", "metal-driver-intervals");
    let err = summarise_gpu_intervals(&xml, all()).expect_err("the wrong table must be refused");
    assert!(
        matches!(err, XctraceError::WrongSchema { ref actual, .. } if actual == "metal-driver-intervals"),
        "got {err}"
    );
}

#[test]
fn a_process_filter_matching_nothing_is_an_error_not_an_empty_summary() {
    let err = summarise_gpu_intervals(&doc(ROWS_HAPPY), only("no-such-process"))
        .expect_err("a filter that selects nothing must be refused");
    assert!(matches!(err, XctraceError::NoRows { .. }), "got {err}");
}

// --- the summary ------------------------------------------------------------

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn the_summary_totals_the_rows_it_kept() {
    let summary = summarise_gpu_intervals(&doc(ROWS_HAPPY), only("rmlx")).unwrap();
    assert_eq!(summary.rows_total, 2);
    assert_eq!(summary.rows_matched, 2);
    assert_eq!(summary.channels.len(), 1);

    let compute = summary.channels.first().unwrap();
    assert_eq!(compute.channel, "Compute");
    assert_eq!(compute.submissions, 2);
    assert_eq!(compute.busy_ns, 20000, "10000 inline + 10000 through a ref");
    // Only row 1 carries a start-latency; row 2's is a sentinel. Counting the
    // NULL as a zero would halve the reported CPU->GPU gap.
    assert_eq!(compute.latency_samples(), 1);
    assert_eq!(compute.latency_pct(50), 5000);
    // start 1000, last end 2000+10000.
    assert_eq!(summary.span_ns(), 11000);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn the_csv_is_nanoseconds_and_one_row_per_channel() {
    let summary = summarise_gpu_intervals(&doc(ROWS_HAPPY), all()).unwrap();
    let csv = summary_csv(&summary);
    let mut lines = csv.lines();
    assert_eq!(
        lines.next().unwrap(),
        "channel,submissions,busy_ns,dur_p50_ns,dur_p95_ns,\
         latency_samples,latency_p50_ns,latency_p95_ns,latency_max_ns"
    );
    assert_eq!(
        lines.next().unwrap(),
        "Compute,2,20000,10000,10000,1,5000,5000,5000"
    );
    assert!(lines.next().is_none(), "one channel, one row");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn ampersands_in_a_label_survive_unescaping() {
    // The driver coalesces consecutive compute encoders and names the row
    // "EncA & EncB", which the export escapes. Leaving it escaped corrupts the
    // one field that identifies a submission.
    let rows = "<row><start-time fmt=\"a\">1</start-time><duration fmt=\"b\">2</duration>\
                <sentinel/><gpu-channel-name fmt=\"EncA &amp; EncB\">EncA &amp; EncB</gpu-channel-name>\
                <process fmt=\"rmlx (1)\"></process></row>";
    let rows = collect(&doc(rows)).unwrap();
    assert_eq!(
        rows.first().unwrap().get(3).unwrap().text(),
        Some("EncA & EncB")
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn a_skip_covering_the_whole_span_is_refused_and_says_what_it_had() {
    // Starts are 1000 ns and 2000 ns, so any whole-millisecond skip lands past
    // both. Reported as its own error, not as "no rows": overshooting the skip
    // and recording nothing are different problems with different fixes.
    let err = summarise_gpu_intervals(
        &doc(ROWS_HAPPY),
        SummaryFilter {
            process: None,
            skip_ms: 1,
        },
    )
    .expect_err("a skip past every submission must be refused, not summarised as zero");
    assert!(
        matches!(err, XctraceError::SkipExceedsSpan { skip_ms: 1, .. }),
        "got {err}"
    );

    // A zero skip is the identity, and must not shift the origin.
    let kept = summarise_gpu_intervals(
        &doc(ROWS_HAPPY),
        SummaryFilter {
            process: None,
            skip_ms: 0,
        },
    )
    .unwrap();
    assert_eq!(kept.rows_matched, 2);
}

/// Another process submits at t=0; the process of interest only starts at
/// t=10 ms, as rmlx does — weight load submits nothing to the GPU, so the
/// process's own first row is prefill, not launch.
const ROWS_TWO_PROCESSES: &str = "\
<row>\
<start-time id=\"1\" fmt=\"a\">0</start-time>\
<duration id=\"2\" fmt=\"b\">1000</duration>\
<sentinel/>\
<gpu-channel-name id=\"3\" fmt=\"Compute\">Compute</gpu-channel-name>\
<process id=\"4\" fmt=\"WindowServer (1)\"></process>\
</row>\
<row>\
<start-time id=\"5\" fmt=\"a\">10000000</start-time>\
<duration ref=\"2\"/>\
<sentinel/>\
<gpu-channel-name ref=\"3\"/>\
<process id=\"6\" fmt=\"rmlx (99)\"></process>\
</row>\
<row>\
<start-time id=\"7\" fmt=\"a\">20000000</start-time>\
<duration ref=\"2\"/>\
<sentinel/>\
<gpu-channel-name ref=\"3\"/>\
<process ref=\"6\"/>\
</row>";

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture is a literal in this file; a parse failure here is the test failing"
)]
fn skip_ms_counts_from_the_matched_process_not_from_the_trace() {
    // rmlx's own first submission is at 10 ms, so a 5 ms skip starts at 15 ms
    // and keeps only the 20 ms row. Measured from the TRACE's first submission
    // (t=0, another process) the cutoff would be 5 ms and both rows would
    // survive — which is how a skip sized for one model's load time silently
    // fails to exclude prefill on another.
    let summary = summarise_gpu_intervals(
        &doc(ROWS_TWO_PROCESSES),
        SummaryFilter {
            process: Some("rmlx"),
            skip_ms: 5,
        },
    )
    .unwrap();
    assert_eq!(summary.rows_total, 3);
    assert_eq!(summary.rows_matched, 1, "only the submission after 15 ms");
}

#[test]
fn the_schema_name_constant_matches_the_fixture() {
    assert!(SCHEMA.contains(GPU_INTERVALS_SCHEMA));
}
