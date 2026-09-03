use rusqlite::Connection;

use crate::schema::MIGRATIONS;

/// A DB migrated only as far as `up_to`, i.e. the schema as it stood before the
/// migration under test ran.
fn conn_at_version(up_to: u32) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    for &(version, sql) in MIGRATIONS {
        if version > up_to {
            break;
        }
        conn.execute_batch(sql).unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {version};"))
            .unwrap();
    }
    conn
}

/// Insert straight into `observations`, the way a pre-005 writer did: no
/// `decode_config`, the drafter recorded only in `notes`.
fn insert_pre_005(conn: &Connection, id: i64, value: f64, notes: &str) {
    conn.execute(
        "INSERT INTO prompts (id, sha256, name, body, tokens_approx, first_seen_utc)
         VALUES (1, 'sha', 'p', 'hi', 4, '2026-05-10T07:30:00Z')
         ON CONFLICT(id) DO NOTHING",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO observations (
             id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction, run_id, ts_utc, hardware_tag, notes,
             inserted_utc, inserted_by
         ) VALUES (
             ?1, 'rmlx', 'mlx-community', 'gemma-4-e4b-it-mxfp8', 'mxfp8', 'none', 16384, 1,
             'decode_tps_warm', ?2, 'tps', 'higher_better', 'run', '2026-05-10T07:30:00Z',
             'm5_max_128gb', ?3, '2026-05-10T07:30:00Z', 'test'
         )",
        rusqlite::params![id, value, notes],
    )
    .unwrap();
}

/// The case the column was added for, on rows that predate it: a speculative
/// row and a plain one in what was one cell must end up in two.
#[test]
fn migrating_a_pre_005_db_separates_the_arms() {
    let mut conn = conn_at_version(4);
    insert_pre_005(
        &conn,
        1,
        83.70,
        "config=base draft_kind=none decode_tps=client-side first-to-last SSE token",
    );
    insert_pre_005(
        &conn,
        2,
        160.32,
        "config=mtp6b draft_kind=mtp block_size=6 decode_tps=client-side first-to-last SSE token",
    );

    // Before: the column does not exist, so the key these rows are ranked under
    // is the one without it — one cell, and the drafter holds it.
    let before: Vec<i64> = {
        let mut stmt = conn
            .prepare(
                "SELECT id FROM (
                     SELECT id, ROW_NUMBER() OVER (
                         PARTITION BY backend, model_namespace, model, weight_quant,
                                      kv_quant, ctx_max, prompt_id, metric
                         ORDER BY value DESC
                     ) AS rn
                     FROM observations
                 ) WHERE rn = 1",
            )
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(before, vec![2], "the speculative row held the one cell");

    crate::migrate::run_pending(&mut conn).unwrap();

    let after: Vec<(i64, f64, Option<String>)> = {
        let mut stmt = conn
            .prepare(
                "SELECT id, value, decode_config FROM bests
                 WHERE metric = 'decode_tps_warm' ORDER BY id",
            )
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(after.len(), 2, "each arm keeps its own champion: {after:?}");
    assert_eq!(after[0].2, None, "the no-drafter row stays ordinary decode");
    assert_eq!(
        after[1].2.as_deref(),
        Some("mtp/block=6"),
        "the drafter row says which arm it is"
    );
}

/// A row whose notes say nothing about a drafter is not guessed at.
#[test]
fn migrating_leaves_an_unclassifiable_row_null() {
    let mut conn = conn_at_version(4);
    insert_pre_005(&conn, 1, 100.0, "label=kvbytes_e2b");
    crate::migrate::run_pending(&mut conn).unwrap();

    let stored: Option<String> = conn
        .query_row(
            "SELECT decode_config FROM observations WHERE id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, None);
}
