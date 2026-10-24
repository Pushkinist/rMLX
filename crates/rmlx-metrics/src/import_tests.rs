use std::path::{Path, PathBuf};

use rusqlite::Connection;

use super::*;

fn last_version() -> u32 {
    MIGRATIONS.last().map_or(0, |&(v, _)| v)
}

/// A DB file migrated as far as `version`, holding one prompt and no rows.
fn source_at(dir: &Path, version: u32) -> PathBuf {
    let path = dir.join(format!("source-v{version}.db"));
    let conn = Connection::open(&path).unwrap();
    for &(v, sql) in MIGRATIONS {
        if v > version {
            break;
        }
        conn.execute_batch(sql).unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {v};"))
            .unwrap();
    }
    conn.execute(
        "INSERT INTO prompts (id, sha256, name, body, tokens_approx, first_seen_utc)
         VALUES (7, 'sha-7', 'p', '\"hi\"', 1, '2026-09-01T00:00:00Z')",
        [],
    )
    .unwrap();
    path
}

/// Insert one row. `class_sql` is the SQL value of `run_class` (`NULL` or a
/// quoted class), or `None` for a source without the column.
fn insert_row(path: &Path, run_id: &str, class_sql: Option<&str>) {
    let conn = Connection::open(path).unwrap();
    let (column, value) = class_sql.map_or((String::new(), String::new()), |v| {
        (", run_class".to_owned(), format!(", {v}"))
    });
    conn.execute(
        &format!(
            "INSERT INTO observations (
                 backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
                 metric, value, unit, direction, run_id, ts_utc, hardware_tag, inserted_utc,
                 inserted_by{column}
             ) VALUES (
                 'rmlx', 'mlx-community', 'm', 'mxfp8', 'none', 8192, 7,
                 'decode_tps_warm', 90.0, 'tps', 'higher_better', ?1, '2026-09-01T00:00:00Z',
                 'm5_max_128gb', '2026-09-01T00:00:01Z', 'test'{value}
             )"
        ),
        [run_id],
    )
    .unwrap();
}

fn target() -> Connection {
    let mut conn = crate::schema::open_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

fn classes(conn: &Connection) -> Vec<Option<String>> {
    let mut stmt = conn
        .prepare("SELECT run_class FROM observations ORDER BY run_id")
        .unwrap();
    let rows = stmt.query_map([], |r| r.get(0)).unwrap();
    rows.map(std::result::Result::unwrap).collect()
}

#[test]
fn a_source_without_the_column_takes_the_stated_class() {
    let td = tempfile::tempdir().unwrap();
    let source = source_at(td.path(), LAST_VERSION_WITHOUT_RUN_CLASS);
    insert_row(&source, "a", None);
    let mut conn = target();

    let report = import_db(&mut conn, &source, Some(RunClass::AbArm)).unwrap();
    assert_eq!((report.read, report.inserted, report.classless), (1, 1, 0));
    assert_eq!(classes(&conn), vec![Some("ab_arm".to_owned())]);

    let err = import_db(&mut conn, &source, None).unwrap_err();
    assert!(matches!(err, Error::ImportRunClass(_)), "{err}");
}

#[test]
fn a_source_with_the_column_keeps_each_row_class_and_counts_the_classless() {
    let td = tempfile::tempdir().unwrap();
    let source = source_at(td.path(), last_version());
    insert_row(&source, "a", Some("'diagnostic'"));
    insert_row(&source, "b", Some("NULL"));
    let mut conn = target();

    let err = import_db(&mut conn, &source, Some(RunClass::Gate)).unwrap_err();
    assert!(matches!(err, Error::ImportRunClass(_)), "{err}");

    let report = import_db(&mut conn, &source, None).unwrap();
    assert_eq!((report.read, report.inserted, report.classless), (2, 2, 1));
    assert_eq!(classes(&conn), vec![Some("diagnostic".to_owned()), None]);

    let again = import_db(&mut conn, &source, None).unwrap();
    assert_eq!(
        (again.inserted, again.skipped, again.classless),
        (0, 2, 0),
        "classless counts rows copied with no class, not rows read"
    );
}

#[test]
fn a_source_missing_a_column_the_target_has_is_refused_naming_it() {
    let td = tempfile::tempdir().unwrap();
    let source = source_at(td.path(), last_version());
    insert_row(&source, "a", Some("'gate'"));
    Connection::open(&source)
        .unwrap()
        .execute_batch("ALTER TABLE observations DROP COLUMN decode_stddev;")
        .unwrap();
    let mut conn = target();

    let err = import_db(&mut conn, &source, None).unwrap_err().to_string();
    assert!(err.contains("decode_stddev"), "{err}");
    assert_eq!(classes(&conn), Vec::<Option<String>>::new());
}

#[test]
fn a_source_past_the_run_class_version_without_the_column_is_refused_naming_it() {
    let td = tempfile::tempdir().unwrap();
    let source = source_at(td.path(), LAST_VERSION_WITHOUT_RUN_CLASS);
    Connection::open(&source)
        .unwrap()
        .execute_batch(&format!(
            "PRAGMA user_version = {};",
            LAST_VERSION_WITHOUT_RUN_CLASS + 1
        ))
        .unwrap();
    let mut conn = target();

    let err = import_db(&mut conn, &source, None).unwrap_err().to_string();
    assert!(err.contains("run_class"), "{err}");
}

#[test]
fn a_source_outside_the_readable_versions_is_refused_naming_both() {
    let td = tempfile::tempdir().unwrap();
    let mut conn = target();

    let older = source_at(td.path(), LAST_VERSION_WITHOUT_RUN_CLASS - 1);
    let err = import_db(&mut conn, &older, Some(RunClass::Gate))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(&format!("version {}", LAST_VERSION_WITHOUT_RUN_CLASS - 1)),
        "{err}"
    );

    let newer = source_at(td.path(), last_version());
    Connection::open(&newer)
        .unwrap()
        .execute_batch(&format!("PRAGMA user_version = {};", last_version() + 1))
        .unwrap();
    let err = import_db(&mut conn, &newer, None).unwrap_err().to_string();
    assert!(
        err.contains(&format!("version {}", last_version() + 1)),
        "{err}"
    );
    assert!(err.contains(&format!("{}", last_version())), "{err}");
}

#[test]
fn a_source_missing_a_prompts_column_the_target_has_is_refused_naming_it() {
    let td = tempfile::tempdir().unwrap();
    let source = source_at(td.path(), last_version());
    insert_row(&source, "a", Some("'gate'"));
    Connection::open(&source)
        .unwrap()
        .execute_batch("ALTER TABLE prompts DROP COLUMN notes;")
        .unwrap();
    let mut conn = target();

    let err = import_db(&mut conn, &source, None).unwrap_err().to_string();
    assert!(err.contains("prompts.notes"), "{err}");
    assert_eq!(classes(&conn), Vec::<Option<String>>::new());
}
