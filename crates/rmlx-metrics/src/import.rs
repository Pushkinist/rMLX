//! DB-to-DB import: bring the `observations` rows of another metrics DB, and
//! the prompts they reference, into this one (`docs/METRICS_IMPORT.md`).
//!
//! The source is opened read-only and never migrated, so its rows and schema
//! are not changed; SQLite may still create its `-wal` and `-shm` files beside
//! a WAL-mode source. A row is copied as it stands, except `id`, `prompt_id`
//! (remapped by the prompt's `sha256`) and `run_class`. A row already present
//! — same `run_id`, `metric`, `ts_utc` and `inserted_utc` — is skipped, so a
//! second import adds nothing.

use std::path::Path;

use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::run_class::RunClass;
use crate::schema::MIGRATIONS;

/// The last schema version without `observations.run_class`.
pub const LAST_VERSION_WITHOUT_RUN_CLASS: u32 = 8;

/// The columns that identify a row across DBs. A source row whose key is
/// present in the target is skipped.
const KEY_COLUMNS: [&str; 4] = ["run_id", "metric", "ts_utc", "inserted_utc"];

/// The `observations` columns the import sets itself rather than copying.
const SET_BY_IMPORT: [&str; 3] = ["id", "prompt_id", "run_class"];

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
    /// Rows copied with no class: the source carries the column and the row
    /// states none. They reach no gate here either.
    pub classless: usize,
    /// Prompts inserted because no prompt with their `sha256` was present.
    pub prompts_inserted: usize,
}

/// Import every `observations` row of the DB at `source` into `conn`, in one
/// transaction.
///
/// A source at [`LAST_VERSION_WITHOUT_RUN_CLASS`] needs `stated`, which every
/// row then carries. A later source, up to this binary's last migration, keeps
/// each row's own class and refuses `stated`. Any other version is refused, and
/// so is a source missing a column the target has.
pub fn import_db(
    conn: &mut Connection,
    source: &Path,
    stated: Option<RunClass>,
) -> Result<ImportReport> {
    let src = crate::schema::open_readonly(source)?;
    let source_version: u32 = src.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let current = MIGRATIONS.last().map_or(0, |&(v, _)| v);
    if !(LAST_VERSION_WITHOUT_RUN_CLASS..=current).contains(&source_version) {
        return Err(Error::Schema(format!(
            "{} is at schema version {source_version}; import reads versions \
             {LAST_VERSION_WITHOUT_RUN_CLASS} to {current}",
            source.display()
        )));
    }
    let carries_class = source_version > LAST_VERSION_WITHOUT_RUN_CLASS;
    match (carries_class, stated) {
        (true, Some(class)) => {
            return Err(Error::ImportRunClass(format!(
                "{} is at schema version {source_version}, so each row states its own \
                 run_class; --run-class {class} would overwrite it. Omit --run-class.",
                source.display()
            )))
        }
        (false, None) => {
            return Err(Error::ImportRunClass(format!(
                "{} is at schema version {source_version}, which predates run_class; state \
                 the class of its rows with --run-class ({})",
                source.display(),
                RunClass::spellings()
            )))
        }
        _ => {}
    }
    let columns = copied_columns(conn, &src, carries_class, source)?;
    let prompt_columns = shared_columns(conn, &src, "prompts", &["id"], source)?;

    let tx = conn.transaction()?;
    let prompts_inserted = copy_prompts(&src, &tx, &prompt_columns)?;
    let copied = copy_observations(&src, &tx, &columns, stated)?;
    tx.commit()?;

    tracing::info!(
        source = %source.display(),
        source_version,
        read = copied.read,
        inserted = copied.inserted,
        skipped = copied.skipped,
        classless = copied.classless,
        prompts_inserted,
        "metrics import: copied another DB's observations"
    );
    Ok(ImportReport {
        source_version,
        stated_run_class: stated,
        read: copied.read,
        inserted: copied.inserted,
        skipped: copied.skipped,
        classless: copied.classless,
        prompts_inserted,
    })
}

/// The columns of `table` in `conn`, in table order.
fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT name FROM pragma_table_info(?1) ORDER BY cid")?;
    let names = stmt
        .query_map([table], |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    Ok(names)
}

/// The target's `table` columns except `set_here`, refusing a source that
/// lacks one of them: the import would have to invent its values.
fn shared_columns(
    dst: &Connection,
    src: &Connection,
    table: &str,
    set_here: &[&str],
    source: &Path,
) -> Result<Vec<String>> {
    let source_columns = table_columns(src, table)?;
    let mut columns = table_columns(dst, table)?;
    columns.retain(|c| !set_here.contains(&c.as_str()));
    if let Some(missing) = columns.iter().find(|c| !source_columns.contains(c)) {
        return Err(Error::Schema(format!(
            "{} has no {table}.{missing} column, which this DB has; the import \
             would have to invent its values",
            source.display()
        )));
    }
    Ok(columns)
}

/// The `observations` columns the import copies, key columns first. A source
/// whose version carries `run_class` must have the column too.
fn copied_columns(
    dst: &Connection,
    src: &Connection,
    carries_class: bool,
    source: &Path,
) -> Result<Vec<String>> {
    let set_here: &[&str] = if carries_class {
        &["id", "prompt_id"]
    } else {
        &SET_BY_IMPORT
    };
    let needed = shared_columns(dst, src, "observations", set_here, source)?;
    let mut ordered: Vec<String> = KEY_COLUMNS.iter().map(|&c| c.to_owned()).collect();
    ordered.extend(
        needed
            .into_iter()
            .filter(|c| c != "run_class" && !KEY_COLUMNS.contains(&c.as_str())),
    );
    Ok(ordered)
}

/// What [`copy_observations`] read and wrote.
struct Copied {
    read: usize,
    inserted: usize,
    skipped: usize,
    classless: usize,
}

/// Copy every source observation whose key is not present. Each row gets
/// `stated`, or its own `run_class` when `stated` is `None`.
fn copy_observations(
    src: &Connection,
    dst: &Connection,
    columns: &[String],
    stated: Option<RunClass>,
) -> Result<Copied> {
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
    let key_pred = KEY_COLUMNS
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{c} = ?{}", i + 1))
        .collect::<Vec<_>>()
        .join(" AND ");
    let mut present = dst.prepare(&format!("SELECT 1 FROM observations WHERE {key_pred}"))?;
    let mut prompt_id = dst.prepare("SELECT id FROM prompts WHERE sha256 = ?1")?;

    let n = columns.len();
    let mut rows = read_stmt.query([])?;
    let mut copied = Copied {
        read: 0,
        inserted: 0,
        skipped: 0,
        classless: 0,
    };
    while let Some(row) = rows.next()? {
        copied.read += 1;
        let class = match stated {
            Some(class) => Value::Text(class.as_str().to_owned()),
            None => row.get(n + 1)?,
        };
        let mut values = (0..n)
            .map(|i| row.get::<_, Value>(i))
            .collect::<rusqlite::Result<Vec<Value>>>()?;
        if present.exists(params_from_iter(values.iter().take(KEY_COLUMNS.len())))? {
            copied.skipped += 1;
            continue;
        }
        let sha: String = row.get(n)?;
        values.push(Value::Integer(prompt_id.query_row([&sha], |r| r.get(0))?));
        if class == Value::Null {
            copied.classless += 1;
        }
        values.push(class);
        insert.execute(params_from_iter(values.iter()))?;
        copied.inserted += 1;
    }
    Ok(copied)
}

/// Insert every source prompt an observation references and whose `sha256`
/// is not present, copying every `prompts` column but `id`. Returns how many
/// were inserted.
fn copy_prompts(src: &Connection, dst: &Connection, columns: &[String]) -> Result<usize> {
    let sha_at = columns
        .iter()
        .position(|c| c == "sha256")
        .ok_or_else(|| Error::Schema("this DB's prompts table has no sha256 column".to_owned()))?;
    let list = columns.join(", ");
    let mut read = src.prepare(&format!(
        "SELECT {list} FROM prompts
          WHERE id IN (SELECT DISTINCT prompt_id FROM observations)"
    ))?;
    let mut present = dst.prepare("SELECT 1 FROM prompts WHERE sha256 = ?1")?;
    let placeholders = (1..=columns.len())
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut insert = dst.prepare(&format!(
        "INSERT INTO prompts ({list}) VALUES ({placeholders})"
    ))?;
    let mut rows = read.query([])?;
    let mut inserted = 0_usize;
    while let Some(row) = rows.next()? {
        let sha: String = row.get(sha_at)?;
        if present.exists([&sha])? {
            continue;
        }
        let values = (0..columns.len())
            .map(|i| row.get::<_, Value>(i))
            .collect::<rusqlite::Result<Vec<Value>>>()?;
        insert.execute(params_from_iter(values.iter()))?;
        inserted += 1;
    }
    Ok(inserted)
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;
