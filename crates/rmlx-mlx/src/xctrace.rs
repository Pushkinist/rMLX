//! Parser for the XML that `xcrun xctrace export` writes.
//!
//! # Why this exists
//!
//! A `.gputrace` names the kernels a decode window ran but holds no timing, and
//! its replay has the replay's schedule rather than the run's. The one headless
//! source of real GPU wall-clock on this hardware is the *Metal System Trace*
//! instrument, whose `metal-gpu-intervals` table carries, per GPU submission,
//! `start` and `duration` in nanoseconds and — the signal nothing else
//! exposes — `start-latency`, the gap between the CPU submitting work and the
//! GPU starting it. `xctrace export` emits that table as XML and offers no
//! other format, so reading it needs a parser.
//!
//! # The encoding, and why a naive reader gets plausible wrong numbers
//!
//! Rows are **positional**: the `<schema>` element declares N columns in order,
//! and every `<row>` carries exactly N direct children, one per column, in that
//! order. Three properties make a naive reader misalign silently rather than
//! fail:
//!
//! * **`<sentinel/>` is NULL and still occupies a column slot.** Skipping it
//!   shifts every later column left by one.
//! * **Repeated values are `id`/`ref` back-references.** The first occurrence of
//!   a value carries `id="N"`; later occurrences are `<tag ref="N"/>` with no
//!   content of their own.
//! * **`id`s are minted at every nesting depth, and a `ref` may point into
//!   another column's subtree.** A `<process>` element first appears nested
//!   inside a `<formatted-label>` cell and is then referenced by the `process`
//!   column of the same row. Indexing only top-level elements leaves that
//!   reference unresolvable.
//!
//! Each of those failures reads as a shifted-but-well-formed table — the number
//! in the `duration` slot is a real number, just from the wrong column. So this
//! parser refuses instead of guessing, on three independent invariants:
//!
//! 1. A row's direct-child count must equal the declared column count.
//! 2. Each non-NULL cell's tag must equal its column's declared
//!    `engineering-type`. This is what makes a one-column shift impossible to
//!    miss: shifting almost always lands a tag against the wrong type.
//! 3. Every `ref` must resolve to an `id` already seen.
//!
//! Columns are addressed by **mnemonic**, never by index, so a future reordering
//! of the schema cannot silently change which number a caller reads.
//!
//! # Scope
//!
//! Deliberately not a general XML parser: it handles the subset `xctrace`
//! emits (elements, attributes, text, the standard entities) and rejects
//! anything else rather than guessing. Documents are read whole, so bound a
//! recording with `--time-limit`; an 8-second trace exports tens of MB.

use std::collections::HashMap;

/// Reasons this parser refuses a document. Every variant names the position it
/// gave up at, because "the layout moved" is only actionable with a location.
#[derive(Debug, thiserror::Error)]
#[allow(
    clippy::exhaustive_enums,
    reason = "callers match on these to tell a layout change from a bad filter; a hidden catch-all would defeat that"
)]
pub enum XctraceError {
    /// The document is not the shape `xctrace export` writes.
    #[error("malformed xml at byte {offset}: {reason}")]
    MalformedXml {
        /// Byte offset into the document where the scan gave up.
        offset: usize,
        /// What was expected there.
        reason: String,
    },

    /// No `<schema>` element — usually an `--xpath` that matched nothing.
    #[error("no <schema> element found; did the --xpath match a table?")]
    MissingSchema,

    /// A `<col>` lacked the parts needed to address it.
    #[error("column {index} is missing its <{part}>")]
    IncompleteColumn {
        /// Zero-based column position in the schema.
        index: usize,
        /// The child element that was absent.
        part: &'static str,
    },

    /// A row had a different number of cells than the schema declares.
    #[error("row {row}: schema declares {expected} columns but the row has {actual} cells")]
    ColumnCountMismatch {
        /// Zero-based row number.
        row: usize,
        /// Column count from the `<schema>` element.
        expected: usize,
        /// Direct children counted in the row.
        actual: usize,
    },

    /// A cell's element tag disagrees with the column's declared type.
    #[error(
        "row {row} column {column} ({mnemonic}): expected <{expected}> \
         but found <{actual}> — the columns are misaligned"
    )]
    ColumnTypeMismatch {
        /// Zero-based row number.
        row: usize,
        /// Zero-based column position.
        column: usize,
        /// Declared mnemonic of that column.
        mnemonic: String,
        /// `engineering-type` the schema declares.
        expected: String,
        /// Tag actually found.
        actual: String,
    },

    /// A `ref` pointed at an `id` that has not been seen.
    #[error("row {row} column {column}: ref=\"{id}\" does not resolve to any id")]
    UnresolvedRef {
        /// Zero-based row number.
        row: usize,
        /// Zero-based column position.
        column: usize,
        /// The unresolved id.
        id: String,
    },

    /// A caller asked for a column this schema does not declare.
    #[error("schema '{schema}' has no column '{mnemonic}' (have: {available})")]
    UnknownColumn {
        /// Name of the parsed schema.
        schema: String,
        /// Mnemonic that was requested.
        mnemonic: String,
        /// Comma-separated list of declared mnemonics.
        available: String,
    },

    /// A cell that must hold an integer did not.
    #[error("row {row} column '{mnemonic}': {value:?} is not an integer")]
    NotAnInteger {
        /// Zero-based row number.
        row: usize,
        /// Mnemonic of the offending column.
        mnemonic: String,
        /// The text that failed to parse.
        value: String,
    },

    /// The table parsed but is the wrong one.
    #[error("expected schema '{expected}' but the export holds '{actual}'")]
    WrongSchema {
        /// Schema the caller requires.
        expected: String,
        /// Schema the document declares.
        actual: String,
    },

    /// A well-formed export with no rows. Never silently reported as an empty
    /// result: it means the recording captured nothing, which is a failed run,
    /// not a run with zero GPU work.
    #[error("schema '{schema}' parsed but contains no rows")]
    NoRows {
        /// Name of the parsed schema.
        schema: String,
    },

    /// The requested skip covers everything the matched process did.
    #[error(
        "skip of {skip_ms} ms covers the whole {span_ms} ms of GPU work recorded \
         for this process — lower it, or record for longer"
    )]
    SkipExceedsSpan {
        /// The skip that was asked for.
        skip_ms: u64,
        /// GPU-work span actually available, milliseconds.
        span_ms: u64,
    },
}

/// Convenience alias for this module's fallible operations.
pub type Result<T> = std::result::Result<T, XctraceError>;

/// One declared column of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "a mirror of the <col> element; it gains a field only if the export format does"
)]
pub struct Column {
    /// Short machine name, e.g. `start-latency`. Callers address columns by this.
    pub mnemonic: String,
    /// Element tag every non-NULL cell in this column must carry.
    pub engineering_type: String,
}

/// The `<schema>` header of an exported table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "a mirror of the <schema> element; it gains a field only if the export format does"
)]
pub struct Schema {
    /// Schema name, e.g. `metal-gpu-intervals`.
    pub name: String,
    /// Declared columns, in row order.
    pub columns: Vec<Column>,
}

impl Schema {
    /// Zero-based position of `mnemonic`.
    ///
    /// # Errors
    /// [`XctraceError::UnknownColumn`] when the schema does not declare it —
    /// callers get a named failure rather than a silent `None` that reads as
    /// "this row had no value".
    pub fn column_index(&self, mnemonic: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|c| c.mnemonic == mnemonic)
            .ok_or_else(|| XctraceError::UnknownColumn {
                schema: self.name.clone(),
                mnemonic: mnemonic.to_owned(),
                available: self
                    .columns
                    .iter()
                    .map(|c| c.mnemonic.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }
}

/// One cell. `Null` is the `<sentinel/>` that occupies a slot without a value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "a cell is a value or a sentinel; the export format has no third case"
)]
pub enum Cell {
    /// `<sentinel/>` — no value, but a column slot all the same.
    Null,
    /// A value, either inline or resolved through a `ref`.
    Value {
        /// Element tag; equals the column's `engineering-type`.
        tag: String,
        /// Text content. Raw units — nanoseconds for durations.
        text: String,
        /// The `fmt` attribute: the display form, and the only useful value for
        /// composite cells such as `process`, whose text content is empty.
        fmt: String,
    },
}

impl Cell {
    /// Text content, or `None` for a `<sentinel/>`.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Null => None,
            Self::Value { text, .. } => Some(text),
        }
    }

    /// Display form, or `None` for a `<sentinel/>`.
    #[must_use]
    pub fn fmt(&self) -> Option<&str> {
        match self {
            Self::Null => None,
            Self::Value { fmt, .. } => Some(fmt),
        }
    }
}

/// A row plus the schema needed to address it by name.
#[derive(Debug)]
pub struct RowView<'a> {
    schema: &'a Schema,
    cells: &'a [Cell],
    row: usize,
}

impl RowView<'_> {
    /// Zero-based row number within the table.
    #[must_use]
    pub fn row_number(&self) -> usize {
        self.row
    }

    /// The cell under `mnemonic`.
    ///
    /// # Errors
    /// [`XctraceError::UnknownColumn`] when the schema has no such column.
    pub fn cell(&self, mnemonic: &str) -> Result<&Cell> {
        let idx = self.schema.column_index(mnemonic)?;
        self.cells
            .get(idx)
            .ok_or(XctraceError::ColumnCountMismatch {
                row: self.row,
                expected: self.schema.columns.len(),
                actual: self.cells.len(),
            })
    }

    /// Integer value under `mnemonic`, or `None` when the cell is NULL.
    ///
    /// # Errors
    /// [`XctraceError::NotAnInteger`] when the cell holds something else. A
    /// non-numeric duration is a layout change, not a zero.
    pub fn u64(&self, mnemonic: &str) -> Result<Option<u64>> {
        let Some(text) = self.cell(mnemonic)?.text() else {
            return Ok(None);
        };
        text.trim()
            .parse::<u64>()
            .map(Some)
            .map_err(|_| XctraceError::NotAnInteger {
                row: self.row,
                mnemonic: mnemonic.to_owned(),
                value: text.to_owned(),
            })
    }

    /// Display form under `mnemonic`, or `None` when the cell is NULL.
    ///
    /// # Errors
    /// [`XctraceError::UnknownColumn`] when the schema has no such column.
    pub fn fmt(&self, mnemonic: &str) -> Result<Option<&str>> {
        Ok(self.cell(mnemonic)?.fmt())
    }
}

// ---------------------------------------------------------------------------
// XML scanning
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Event<'a> {
    Open { name: &'a str, attrs: &'a str },
    Empty { name: &'a str, attrs: &'a str },
    Close { name: &'a str },
    Text(&'a str),
}

fn malformed<T>(offset: usize, reason: impl Into<String>) -> Result<T> {
    Err(XctraceError::MalformedXml {
        offset,
        reason: reason.into(),
    })
}

struct Scanner<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn next(&mut self) -> Result<Option<Event<'a>>> {
        let rest = self.src.get(self.pos..).unwrap_or_default();
        if rest.is_empty() {
            return Ok(None);
        }
        if !rest.starts_with('<') {
            // Text run up to the next tag.
            let end = rest.find('<').unwrap_or(rest.len());
            let text = rest.get(..end).unwrap_or_default();
            self.pos += end;
            return Ok(Some(Event::Text(text)));
        }
        // Declarations, comments and CDATA carry nothing this parser needs, but
        // skipping them blindly would also skip a malformed tag, so each is
        // matched by its own terminator and an unterminated one is an error.
        for (open, close) in [("<?", "?>"), ("<!--", "-->"), ("<![CDATA[", "]]>")] {
            if rest.starts_with(open) {
                let Some(end) = rest.find(close) else {
                    return malformed(self.pos, format!("unterminated {open}"));
                };
                self.pos += end + close.len();
                return self.next();
            }
        }
        let Some(close_idx) = rest.find('>') else {
            return malformed(self.pos, "unterminated tag");
        };
        let inner = rest.get(1..close_idx).unwrap_or_default();
        let tag_start = self.pos;
        self.pos += close_idx + 1;

        if let Some(name) = inner.strip_prefix('/') {
            let name = name.trim();
            if name.is_empty() {
                return malformed(tag_start, "empty closing tag");
            }
            return Ok(Some(Event::Close { name }));
        }
        let (inner, empty) = match inner.strip_suffix('/') {
            Some(stripped) => (stripped, true),
            None => (inner, false),
        };
        let inner = inner.trim_end();
        let split = inner
            .find(|c: char| c.is_whitespace())
            .unwrap_or(inner.len());
        let name = inner.get(..split).unwrap_or_default();
        if name.is_empty() {
            return malformed(tag_start, "empty tag name");
        }
        let attrs = inner.get(split..).unwrap_or_default();
        Ok(Some(if empty {
            Event::Empty { name, attrs }
        } else {
            Event::Open { name, attrs }
        }))
    }
}

/// Reads `name="value"` pairs. Values are always double-quoted in `xctrace`
/// output; a single-quoted or unquoted attribute is rejected rather than
/// guessed at.
fn attr(attrs: &str, want: &str) -> Option<String> {
    let mut rest = attrs;
    while let Some(eq) = rest.find('=') {
        let (name_part, after) = rest.split_at(eq);
        let name = name_part.trim();
        let after = after.get(1..)?.trim_start();
        let quoted = after.strip_prefix('"')?;
        let end = quoted.find('"')?;
        let value = quoted.get(..end)?;
        if name == want {
            return Some(unescape(value));
        }
        rest = quoted.get(end + 1..)?;
    }
    None
}

/// The five predefined XML entities. Encoder labels really do contain `&` —
/// the driver coalesces encoders and names the row `EncA & EncB` — so leaving
/// `&amp;` in place would corrupt the one field used to identify a submission.
fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(idx) = rest.find('&') {
        out.push_str(rest.get(..idx).unwrap_or_default());
        let tail = rest.get(idx..).unwrap_or_default();
        let mut matched = false;
        for (entity, ch) in [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ] {
            if tail.starts_with(entity) {
                out.push(ch);
                rest = tail.get(entity.len()..).unwrap_or_default();
                matched = true;
                break;
            }
        }
        if !matched {
            // Numeric or unknown entity: keep it verbatim rather than dropping
            // characters. Nothing in this export has produced one.
            out.push('&');
            rest = tail.get(1..).unwrap_or_default();
        }
    }
    out.push_str(rest);
    out
}

/// What an element contributes once seen: its tag, text and display form. Kept
/// for every element carrying an `id`, at any depth, because a `ref` may point
/// into another column's subtree.
#[derive(Clone)]
struct Interned {
    tag: String,
    text: String,
    fmt: String,
}

/// An element currently open inside a row. `id` is carried so the interned copy
/// can be completed in O(1) when the element closes and its text is known.
struct Frame {
    tag: String,
    text: String,
    fmt: String,
    id: Option<String>,
}

/// Parses an exported table, calling `visit` once per row.
///
/// Streaming rather than collecting: a bounded recording still exports tens of
/// MB, and callers aggregate as they go.
///
/// # Errors
/// Any of [`XctraceError`]. The parser refuses a document it cannot align
/// rather than returning partial or shifted rows.
pub fn for_each_row<F>(xml: &str, mut visit: F) -> Result<Schema>
where
    F: FnMut(&RowView<'_>) -> Result<()>,
{
    let mut scanner = Scanner::new(xml);
    let mut schema: Option<Schema> = None;
    let mut ids: HashMap<String, Interned> = HashMap::new();
    let mut row_index = 0usize;

    while let Some(event) = scanner.next()? {
        match event {
            Event::Open {
                name: "schema",
                attrs,
            } => {
                schema = Some(parse_schema(&mut scanner, attrs)?);
            }
            Event::Open { name: "row", .. } => {
                let Some(schema) = schema.as_ref() else {
                    return Err(XctraceError::MissingSchema);
                };
                let cells = parse_row(&mut scanner, schema, &mut ids, row_index)?;
                let view = RowView {
                    schema,
                    cells: &cells,
                    row: row_index,
                };
                visit(&view)?;
                row_index += 1;
            }
            Event::Open { .. } | Event::Empty { .. } | Event::Close { .. } | Event::Text(_) => {}
        }
    }

    let schema = schema.ok_or(XctraceError::MissingSchema)?;
    if row_index == 0 {
        return Err(XctraceError::NoRows {
            schema: schema.name,
        });
    }
    Ok(schema)
}

fn parse_schema(scanner: &mut Scanner<'_>, attrs: &str) -> Result<Schema> {
    let name = attr(attrs, "name").unwrap_or_default();
    let mut columns: Vec<Column> = Vec::new();
    let mut mnemonic: Option<String> = None;
    let mut engineering_type: Option<String> = None;
    let mut current: Option<&'static str> = None;

    while let Some(event) = scanner.next()? {
        match event {
            Event::Open { name: tag, .. } => {
                current = match tag {
                    "mnemonic" => Some("mnemonic"),
                    "engineering-type" => Some("engineering-type"),
                    _ => None,
                };
            }
            Event::Text(text) => match current {
                Some("mnemonic") => mnemonic = Some(unescape(text)),
                Some("engineering-type") => engineering_type = Some(unescape(text)),
                _ => {}
            },
            Event::Close { name: "col" } => {
                let index = columns.len();
                columns.push(Column {
                    mnemonic: mnemonic.take().ok_or(XctraceError::IncompleteColumn {
                        index,
                        part: "mnemonic",
                    })?,
                    engineering_type: engineering_type.take().ok_or(
                        XctraceError::IncompleteColumn {
                            index,
                            part: "engineering-type",
                        },
                    )?,
                });
                current = None;
            }
            Event::Close { name: "schema" } => {
                return Ok(Schema { name, columns });
            }
            Event::Empty { .. } | Event::Close { .. } => {}
        }
    }
    Err(XctraceError::MissingSchema)
}

/// Consumes one `<row>`, returning exactly one cell per declared column.
///
/// Depth tracking is what keeps nested elements out of the column sequence
/// while still interning their ids: only elements closing back to depth 0
/// inside the row are cells.
fn parse_row(
    scanner: &mut Scanner<'_>,
    schema: &Schema,
    ids: &mut HashMap<String, Interned>,
    row_index: usize,
) -> Result<Vec<Cell>> {
    let mut cells: Vec<Cell> = Vec::with_capacity(schema.columns.len());
    // One frame per open element inside the row. A frame is a cell only when it
    // closes back to an empty stack; everything deeper is nested detail whose
    // ids still have to be interned.
    let mut stack: Vec<Frame> = Vec::new();

    while let Some(event) = scanner.next()? {
        match event {
            Event::Close { name } if name == "row" && stack.is_empty() => {
                return finish_row(cells, schema, row_index);
            }
            Event::Open { name, attrs } => {
                stack.push(Frame {
                    tag: name.to_owned(),
                    text: String::new(),
                    fmt: attr(attrs, "fmt").unwrap_or_default(),
                    id: attr(attrs, "id"),
                });
            }
            Event::Close { name } => {
                let Some(done) = stack.pop() else {
                    return Err(XctraceError::MalformedXml {
                        offset: scanner.pos,
                        reason: format!("closing </{name}> with no matching open inside <row>"),
                    });
                };
                if done.tag != name {
                    return Err(XctraceError::MalformedXml {
                        offset: scanner.pos,
                        reason: format!("closing </{name}> does not match open <{}>", done.tag),
                    });
                }
                // Interned at close, not at open: text is only known now, and a
                // ref that resolved to a still-open element would silently read
                // as empty. Close-time interning turns that into an
                // UnresolvedRef instead.
                if let Some(id) = done.id {
                    ids.insert(
                        id,
                        Interned {
                            tag: done.tag.clone(),
                            text: done.text.clone(),
                            fmt: done.fmt.clone(),
                        },
                    );
                }
                if stack.is_empty() {
                    cells.push(Cell::Value {
                        tag: done.tag,
                        text: done.text,
                        fmt: done.fmt,
                    });
                }
            }
            Event::Empty { name, attrs } => {
                let cell = if name == "sentinel" {
                    Cell::Null
                } else if let Some(id) = attr(attrs, "ref") {
                    let target =
                        ids.get(&id)
                            .cloned()
                            .ok_or_else(|| XctraceError::UnresolvedRef {
                                row: row_index,
                                column: cells.len(),
                                id: id.clone(),
                            })?;
                    Cell::Value {
                        tag: target.tag,
                        text: target.text,
                        fmt: target.fmt,
                    }
                } else {
                    let interned = Interned {
                        tag: name.to_owned(),
                        text: String::new(),
                        fmt: attr(attrs, "fmt").unwrap_or_default(),
                    };
                    if let Some(id) = attr(attrs, "id") {
                        ids.insert(id, interned.clone());
                    }
                    Cell::Value {
                        tag: interned.tag,
                        text: interned.text,
                        fmt: interned.fmt,
                    }
                };
                if stack.is_empty() {
                    cells.push(cell);
                }
            }
            Event::Text(text) => {
                if let Some(top) = stack.last_mut() {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        top.text.push_str(&unescape(trimmed));
                    }
                }
            }
        }
    }
    Err(XctraceError::MalformedXml {
        offset: scanner.pos,
        reason: "document ended inside <row>".to_owned(),
    })
}

/// The two alignment invariants. Checked together so a shifted row reports the
/// column it went wrong at rather than just a count.
fn finish_row(cells: Vec<Cell>, schema: &Schema, row_index: usize) -> Result<Vec<Cell>> {
    if cells.len() != schema.columns.len() {
        return Err(XctraceError::ColumnCountMismatch {
            row: row_index,
            expected: schema.columns.len(),
            actual: cells.len(),
        });
    }
    for (index, (cell, column)) in cells.iter().zip(schema.columns.iter()).enumerate() {
        if let Cell::Value { tag, .. } = cell {
            if tag != &column.engineering_type {
                return Err(XctraceError::ColumnTypeMismatch {
                    row: row_index,
                    column: index,
                    mnemonic: column.mnemonic.clone(),
                    expected: column.engineering_type.clone(),
                    actual: tag.clone(),
                });
            }
        }
    }
    Ok(cells)
}

// The `metal-gpu-intervals` summary lives in a sibling file and is re-exported
// here, so callers still see one `xctrace` module. The split is the natural
// seam: everything above is table-generic and works for any exported schema,
// everything below knows one table's column names.
#[path = "xctrace_gpu_intervals.rs"]
mod gpu_intervals;

pub use gpu_intervals::{
    summarise_gpu_intervals, summary_csv, ChannelStats, GpuIntervalSummary, SummaryFilter,
    GPU_INTERVALS_SCHEMA,
};

#[cfg(test)]
#[path = "xctrace_tests.rs"]
mod xctrace_tests;
