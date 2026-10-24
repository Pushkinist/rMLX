//! DB-to-DB import: bring the `observations` rows of another metrics DB, and
//! the prompts they reference, into this one (`docs/METRICS_DB.md` §11).
//!
//! The source is opened read-only and never migrated. A row is copied as it
//! stands, except `id`, `prompt_id` (remapped by the prompt's `sha256`) and
//! `run_class`. A row already present — same `run_id`, `metric`, `ts_utc` and
//! `inserted_utc` — is skipped, so a second import adds nothing.

use std::path::Path;

use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection, OpenFlags};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::run_class::RunClass;
use crate::schema::MIGRATIONS;

/// The last schema version without `observations.run_class`.
pub const LAST_VERSION_WITHOUT_RUN_CLASS: u32 = 8;

/// The columns that identify a row across DBs. A source row whose key is
/// present in the target is skipped.
const KEY_COLUMNS: [&str; 4] = ["run_id", "metric", "ts_utc", "inserted_utc"];

/// The other `observations` columns copied verbatim. `id`, `prompt_id` and
/// `run_class` are set by the import.
const OTHER_COLUMNS: &[&str] = &[
    "backend",
    "model_namespace",
    "model",
    "weight_quant",
    "kv_quant",
    "ctx_max",
    "value",
    "unit",
    "direction",
    "git_sha",
    "build_profile",
    "backend_version",
    "hardware_tag",
    "prompt_tokens",
    "max_tokens",
    "temperature",
    "seed",
    "n_warmups",
    "n_measure",
    "output_first_64",
    "decode_stddev",
    "notes",
    "description",
    "inserted_by",
    "decode_config",
];

/// What one import read and wrote.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct ImportReport {
    /// `PRAGMA user_version` of the source.
    pub source_version: u32,
    /// The class stamped on every row, or `None` when the rows kept their own.
    pub stated_run_class: Option<RunClass>,
    /// Source `observations` rows read.
    pub read: usize,
    /// Rows inserted.
    pub inserted: usize,
    /// Rows already present, not inserted again.
    pub skipped: usize,
    /// Prompts inserted because no prompt with their `sha256` was present.
    pub prompts_inserted: usize,
}

/// Import every `observations` row of the DB at `source` into `conn`, in one
/// transaction.
///
/// A source at [`LAST_VERSION_WITHOUT_RUN_CLASS`] needs `stated`, which every
/// row then carries. A source at the current schema keeps each row's own class
/// and refuses `stated`. Any other version is refused.
pub fn import_db(
    conn: &mut Connection,
    source: &Path,
    stated: Option<RunClass>,
) -> Result<ImportReport> {
    let src = open_source(source)?;
    let source_version: u32 = src.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let current = MIGRATIONS.last().map_or(0, |&(v, _)| v);

    let carries_class = if source_version == current {
        true
    } else if source_version == LAST_VERSION_WITHOUT_RUN_CLASS {
        false
    } else {
        return Err(Error::Schema(format!(
            "{} is at schema version {source_version}; import reads version \
             {LAST_VERSION_WITHOUT_RUN_CLASS} and version {current} only",
            source.display()
        )));
    };
    match (carries_class, stated) {
        (true, Some(class)) => {
            return Err(Error::Schema(format!(
                "{} is at schema version {current}, so each row states its own run_class; \
                 --run-class {class} would overwrite it. Omit --run-class.",
                source.display()
            )))
        }
        (false, None) => {
            return Err(Error::Schema(format!(
                "{} is at schema version {source_version}, which predates run_class; state \
                 the class of its rows with --run-class ({})",
                source.display(),
                RunClass::spellings()
            )))
        }
        _ => {}
    }

    let tx = conn.transaction()?;
    let prompts_inserted = copy_prompts(&src, &tx)?;
    let (read, inserted, skipped) = copy_observations(&src, &tx, stated)?;
    tx.commit()?;

    Ok(ImportReport {
        source_version,
        stated_run_class: stated,
        read,
        inserted,
        skipped,
        prompts_inserted,
    })
}

/// Copy every source observation whose key is not present. Each row gets
/// `stated`, or its own `run_class` when `stated` is `None`. Returns how many
/// rows were read, inserted and skipped.
fn copy_observations(
    src: &Connection,
    dst: &Connection,
    stated: Option<RunClass>,
) -> Result<(usize, usize, usize)> {
    let columns: Vec<&str> = KEY_COLUMNS.iter().chain(OTHER_COLUMNS).copied().collect();
    let own_class = if stated.is_none() {
        ", o.run_class"
    } else {
        ""
    };
    let mut read_stmt = src.prepare(&format!(
        "SELECT o.{}, p.sha256{own_class} FROM observations o
           JOIN prompts p ON p.id = o.prompt_id
          ORDER BY o.id",
        columns.join(", o."),
    ))?;
    let placeholders = (1..=columns.len() + 2)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut insert = dst.prepare(&format!(
        "INSERT INTO observations ({}, prompt_id, run_class) VALUES ({placeholders})",
        columns.join(", ")
    ))?;
    let mut present = dst.prepare(
        "SELECT 1 FROM observations
          WHERE run_id = ?1 AND metric = ?2 AND ts_utc = ?3 AND inserted_utc = ?4",
    )?;
    let mut prompt_id = dst.prepare("SELECT id FROM prompts WHERE sha256 = ?1")?;

    let n = columns.len();
    let mut rows = read_stmt.query([])?;
    let (mut read, mut inserted, mut skipped) = (0_usize, 0_usize, 0_usize);
    while let Some(row) = rows.next()? {
        read += 1;
        let mut values = (0..n)
            .map(|i| row.get::<_, Value>(i))
            .collect::<rusqlite::Result<Vec<Value>>>()?;
        if present.exists(params_from_iter(values.iter().take(KEY_COLUMNS.len())))? {
            skipped += 1;
            continue;
        }
        let sha: String = row.get(n)?;
        values.push(Value::Integer(prompt_id.query_row([&sha], |r| r.get(0))?));
        values.push(match stated {
            Some(class) => Value::Text(class.as_str().to_owned()),
            None => row.get(n + 1)?,
        });
        insert.execute(params_from_iter(values.iter()))?;
        inserted += 1;
    }
    Ok((read, inserted, skipped))
}

/// Open `path` read-only. A read-only open never writes the file, so it does
/// not take the connection PRAGMAs `schema::open` sets.
fn open_source(path: &Path) -> Result<Connection> {
    if !path.exists() {
        return Err(Error::Schema(format!(
            "no metrics DB at {}",
            path.display()
        )));
    }
    let conn =
        Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
            Error::Schema(format!(
                "cannot open {} read-only ({e}); import a copy taken with \
                 `rmlx metrics --db {} backup --out <copy>`",
                path.display(),
                path.display()
            ))
        })?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    Ok(conn)
}

/// Insert every source prompt an observation references and whose `sha256`
/// is not present. Returns how many were inserted.
fn copy_prompts(src: &Connection, dst: &Connection) -> Result<usize> {
    let mut read = src.prepare(
        "SELECT sha256, name, body, tokens_approx, first_seen_utc, notes FROM prompts
          WHERE id IN (SELECT DISTINCT prompt_id FROM observations)",
    )?;
    let mut present = dst.prepare("SELECT 1 FROM prompts WHERE sha256 = ?1")?;
    let mut insert = dst.prepare(
        "INSERT INTO prompts (sha256, name, body, tokens_approx, first_seen_utc, notes)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    let mut rows = read.query([])?;
    let mut inserted = 0_usize;
    while let Some(row) = rows.next()? {
        let sha: String = row.get(0)?;
        if present.exists([&sha])? {
            continue;
        }
        let values = (0..6)
            .map(|i| row.get::<_, Value>(i))
            .collect::<rusqlite::Result<Vec<Value>>>()?;
        insert.execute(params_from_iter(values.iter()))?;
        inserted += 1;
    }
    Ok(inserted)
}
