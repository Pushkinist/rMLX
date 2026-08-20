use super::*;
use std::fs;
use tempfile::TempDir;

const LK_A: u64 = 0x1111_2222_3333_4444;
const LK_B: u64 = 0xaaaa_bbbb_cccc_dddd;

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn open() -> SsdKvIndex {
    SsdKvIndex::open_memory().expect("open_memory")
}

// ── CRUD ──────────────────────────────────────────────────────────────────

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn crud_record_lookup_touch() {
    let idx = open();
    let hash = hash_to_hex(0xdeadbeef_cafebabe);
    let path = PathBuf::from("/tmp/test.kvb");

    // Initially absent.
    assert!(idx.lookup(&hash, LK_A).unwrap().is_none());

    // Record.
    idx.record(&hash, LK_A, &path, "Arch/snap", "k8v8", 1024)
        .unwrap();

    let row = idx.lookup(&hash, LK_A).unwrap().expect("row after record");
    assert_eq!(row.hash, hash);
    assert_eq!(row.layout_key, LK_A);
    assert_eq!(row.path, path);
    assert_eq!(row.model_id, "Arch/snap");
    assert_eq!(row.kv_quant, "k8v8");
    assert_eq!(row.byte_size, 1024);
    let lu0 = row.last_used;

    // Touch — last_used must not regress.
    idx.touch(&hash, LK_A).unwrap();
    let row2 = idx.lookup(&hash, LK_A).unwrap().expect("row after touch");
    assert!(row2.last_used >= lu0);

    // Update via OR REPLACE (re-record same key).
    idx.record(&hash, LK_A, &path, "Arch/snap", "k8v8", 2048)
        .unwrap();
    let row3 = idx
        .lookup(&hash, LK_A)
        .unwrap()
        .expect("row after re-record");
    assert_eq!(row3.byte_size, 2048);
}

/// unit test #6: same hash, two layout_keys → both rows survive,
/// neither overwrites the other. Composite PK guards collisions.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn collision_two_layout_keys_keeps_both_rows() {
    let idx = open();
    let hash = hash_to_hex(0xface_f00d_0bad_c0de_u64);
    let path_a = PathBuf::from("/tmp/k8v4.kvb");
    let path_b = PathBuf::from("/tmp/k8v8.kvb");

    idx.record(&hash, LK_A, &path_a, "Arch/snap", "k8v4", 100)
        .unwrap();
    idx.record(&hash, LK_B, &path_b, "Arch/snap", "k8v8", 200)
        .unwrap();

    let count: i64 = idx
        .conn
        .query_row(
            "SELECT COUNT(*) FROM kv_blocks WHERE hash = ?1",
            params![hash],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 2, "both rows must persist under composite PK");

    let a = idx.lookup(&hash, LK_A).unwrap().expect("layout A row");
    let b = idx.lookup(&hash, LK_B).unwrap().expect("layout B row");
    assert_eq!(a.kv_quant, "k8v4");
    assert_eq!(b.kv_quant, "k8v8");
    assert_eq!(a.byte_size, 100);
    assert_eq!(b.byte_size, 200);
}

/// unit test #7: lookup under a different layout_key than the row
/// was recorded with returns None — no false hit.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn cross_layout_lookup_is_a_miss() {
    let idx = open();
    let hash = hash_to_hex(0x0123_4567_89ab_cdefu64);
    idx.record(
        &hash,
        LK_A,
        &PathBuf::from("/tmp/a.kvb"),
        "Arch/snap",
        "k8v4",
        42,
    )
    .unwrap();

    assert!(idx.lookup(&hash, LK_A).unwrap().is_some());
    assert!(
        idx.lookup(&hash, LK_B).unwrap().is_none(),
        "different layout_key must surface as a miss"
    );
}

// ── LRU stamp granularity ─────────────────────────────────────────────────

/// Microseconds alone are not a total order: a burst of calls fits inside one,
/// and a clock stepped backwards by NTP would hand out a decreasing value.
/// Eviction has no tiebreak column, so the stamp source itself has to be the
/// total order.
#[test]
fn now_stamp_us_is_strictly_increasing() {
    let stamps: Vec<u64> = (0..10_000).map(|_| now_stamp_us()).collect();
    for pair in stamps.windows(2) {
        if let [a, b] = *pair {
            assert!(b > a, "stamp {b} did not advance past {a}");
        }
    }
}

/// The eviction victim must be the least *recently used* block, not an
/// arbitrary member of whatever set shares a coarse timestamp.
///
/// Everything here — eight `record`s and three `touch`es — happens inside a
/// single wall-clock second, which is the ordinary case for a server under
/// load. At second granularity every row carries the same `last_used`,
/// `ORDER BY last_used ASC` has no tiebreak, and SQLite yields rows in
/// insert order: eviction then takes `b0..b2` (the three blocks that were
/// just touched and are the *most* recently used) and keeps `b3..b5`. The
/// touches are deliberately applied to the earliest-inserted blocks so the
/// two orderings disagree on every row rather than by luck.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "every unwrap is on an in-memory SQLite index this test just opened; a failure is a broken fixture and must abort rather than be reported as a passing eviction"
)]
fn evict_lru_picks_the_least_recently_used_within_one_second() {
    let idx = open();
    let names: Vec<String> = (0..8).map(|i| format!("b{i}")).collect();
    for name in &names {
        idx.record(
            name,
            LK_A,
            &PathBuf::from(format!("/tmp/{name}.kvb")),
            "m",
            "k",
            1000,
        )
        .unwrap();
    }

    // Re-use b0, b1, b2 — they become the three most recently used blocks.
    // LRU order is now b3, b4, b5, b6, b7, b0, b1, b2.
    for name in names.iter().take(3) {
        idx.touch(name, LK_A).unwrap();
    }

    // 8000 bytes down to 5000 evicts exactly the three oldest: b3, b4, b5.
    let evicted = idx.evict_lru_until(5000).unwrap();
    let got: Vec<String> = evicted
        .paths
        .iter()
        .map(|p| p.file_stem().unwrap_or_default().to_string_lossy().into())
        .collect();
    assert_eq!(
        got,
        vec!["b3".to_string(), "b4".to_string(), "b5".to_string()],
        "eviction must take the least recently used blocks, in LRU order"
    );
    for name in ["b0", "b1", "b2", "b6", "b7"] {
        assert!(
            idx.lookup(name, LK_A).unwrap().is_some(),
            "{name} was more recently used than every evicted block and must survive"
        );
    }
}

/// The stamps have to separate blocks written from different threads too —
/// that is where the tie was originally observed, and a coarse clock ties
/// hardest exactly when request rate is highest.
///
/// Two properties, both false at second granularity: every row carries a
/// distinct stamp, and within one thread the stamps follow the order that
/// thread touched its own blocks in.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "temp-dir SQLite opens and writes from the worker threads; a failure inside a worker is a broken fixture and must abort that thread so the join surfaces it"
)]
#[allow(
    clippy::expect_used,
    reason = "thread join plus lookups of rows the workers just wrote; a missing row or a panicked worker is a test failure and must surface rather than be swallowed"
)]
fn concurrent_touches_get_distinct_ordered_stamps() {
    use std::sync::{Arc, Barrier};

    const THREADS: usize = 3;
    const PER_THREAD: usize = 16;

    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");
    let _seed = SsdKvIndex::open_at(&db).unwrap();

    // Every thread has to be running before any is joined, so the spawns are
    // collected up front rather than folded into the join.
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::with_capacity(THREADS);
    for t in 0..THREADS {
        let (db, barrier) = (db.clone(), Arc::clone(&barrier));
        handles.push(std::thread::spawn(move || {
            let idx = SsdKvIndex::open_at(&db).unwrap();
            let names: Vec<String> = (0..PER_THREAD).map(|i| format!("t{t}-{i:02}")).collect();
            barrier.wait();
            for name in &names {
                idx.record(
                    name,
                    LK_A,
                    &PathBuf::from(format!("/tmp/{name}.kvb")),
                    "m",
                    "k",
                    1,
                )
                .unwrap();
            }
            // Touch in a fixed order; the resulting stamps must follow it.
            for name in &names {
                idx.touch(name, LK_A).unwrap();
            }
            names
        }));
    }

    let mut per_thread: Vec<Vec<String>> = Vec::with_capacity(THREADS);
    for h in handles {
        per_thread.push(h.join().expect("worker thread panicked"));
    }

    let idx = SsdKvIndex::open_at(&db).unwrap();
    let stamp = |name: &str| {
        idx.lookup(name, LK_A)
            .unwrap()
            .expect("row present")
            .last_used
    };

    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for names in &per_thread {
        let mut prev: Option<u64> = None;
        for name in names {
            let s = stamp(name);
            assert!(
                seen.insert(s),
                "stamp {s} collides across rows; a shared stamp leaves eviction \
                 without a tiebreak and the victim arbitrary"
            );
            if let Some(p) = prev {
                assert!(
                    s > p,
                    "{name} was touched after the previous block in its thread \
                     but carries stamp {s} <= {p}"
                );
            }
            prev = Some(s);
        }
    }
    assert_eq!(seen.len(), THREADS * PER_THREAD);
}

/// A clamp that only remembers what *this* process issued is not durable. A
/// restart reloads it at zero, so if the wall clock moved backwards while the
/// process was down (NTP step, manual date change, RTC drift across sleep)
/// every freshly written block stamps *below* the rows already on disk and
/// becomes the first eviction victim. That inversion persists for as long as
/// the clock lags — strictly worse than the same-second tie, which at least
/// resolved itself a second later.
///
/// Persisting rows stamped ahead of the current clock and then opening the
/// index fresh is exactly what a restart after a backwards step looks like
/// from the index's point of view.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture setup on a temp-dir SQLite DB this test just created; a failure here is a broken fixture and should abort the test"
)]
#[allow(
    clippy::expect_used,
    reason = "the row was inserted a few lines above, so its absence is a broken fixture rather than a condition under test"
)]
fn a_fresh_open_orders_new_blocks_above_rows_written_before_a_clock_step() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");
    // 10 s beyond the stamp source's current position, i.e. the clock has since
    // stepped back by 10 s relative to what is already persisted.
    let ahead = now_stamp_us() + 10_000_000;
    {
        let idx = SsdKvIndex::open_at(&db).unwrap();
        idx.conn
            .execute(
                "INSERT INTO kv_blocks
                 (hash, layout_key, path, model_id, kv_quant, byte_size, last_used)
                 VALUES ('before', ?1, '/tmp/before.kvb', 'm', 'k', 1000, ?2)",
                params![LK_A as i64, ahead as i64],
            )
            .unwrap();
    }

    // The restart.
    let idx = SsdKvIndex::open_at(&db).unwrap();
    idx.record(
        "after",
        LK_A,
        &PathBuf::from("/tmp/after.kvb"),
        "m",
        "k",
        1000,
    )
    .unwrap();

    let before = idx.lookup("before", LK_A).unwrap().expect("seeded row");
    let after = idx.lookup("after", LK_A).unwrap().expect("recorded row");
    assert!(
        after.last_used > before.last_used,
        "a block written after the restart carries stamp {} but the row already \
         on disk carries {}; the newer block would be evicted first",
        after.last_used,
        before.last_used
    );

    // The evictor has to agree, not just the raw column.
    let evicted = idx.evict_lru_until(1000).unwrap();
    assert_eq!(
        evicted.paths,
        vec![PathBuf::from("/tmp/before.kvb")],
        "eviction must take the older block, not the one written after the restart"
    );
}

// ── evict_lru_until ───────────────────────────────────────────────────────

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn evict_lru_returns_oldest_first_and_leaves_within_budget() {
    let idx = open();
    let path = |n: &str| PathBuf::from(format!("/tmp/{n}.kvb"));

    // Insert three blocks with different last_used values via direct SQL
    // so we control the timestamps. Same layout_key for all three.
    let lk = LK_A as i64;
    idx.conn
        .execute(
            "INSERT INTO kv_blocks
             (hash, layout_key, path, model_id, kv_quant, byte_size, last_used)
             VALUES
             ('aaa', ?1, '/tmp/aaa.kvb', 'm', 'k', 1000, 10),
             ('bbb', ?1, '/tmp/bbb.kvb', 'm', 'k', 1000, 20),
             ('ccc', ?1, '/tmp/ccc.kvb', 'm', 'k', 1000, 30)",
            params![lk],
        )
        .unwrap();

    let evicted = idx.evict_lru_until(1500).unwrap();
    assert_eq!(evicted.paths.len(), 2);
    assert_eq!(evicted.paths[0], path("aaa"));
    assert_eq!(evicted.paths[1], path("bbb"));
    assert_eq!(
        evicted.total_bytes_after, 1000,
        "the reported footprint must be the post-eviction one"
    );

    assert!(idx.lookup("ccc", LK_A).unwrap().is_some());
    assert!(idx.lookup("aaa", LK_A).unwrap().is_none());
    assert!(idx.lookup("bbb", LK_A).unwrap().is_none());

    let evicted2 = idx.evict_lru_until(1500).unwrap();
    assert!(evicted2.paths.is_empty());
    assert_eq!(evicted2.total_bytes_after, 1000);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn total_bytes_sums_all_rows_and_tracks_eviction() {
    let idx = open();
    assert_eq!(idx.total_bytes().unwrap(), 0);
    idx.conn
        .execute(
            "INSERT INTO kv_blocks
             (hash, layout_key, path, model_id, kv_quant, byte_size, last_used)
             VALUES
             ('a', ?1, '/tmp/a.kvb', 'm', 'k', 1000, 10),
             ('b', ?1, '/tmp/b.kvb', 'm', 'k', 2000, 20)",
            params![LK_A as i64],
        )
        .unwrap();
    assert_eq!(idx.total_bytes().unwrap(), 3000);

    idx.evict_lru_until(2500).unwrap();
    assert_eq!(idx.total_bytes().unwrap(), 2000);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn evict_empty_index_is_noop() {
    let idx = open();
    let evicted = idx.evict_lru_until(0).unwrap();
    assert!(evicted.paths.is_empty());
    assert_eq!(evicted.total_bytes_after, 0);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn evict_zero_budget_evicts_all() {
    let idx = open();
    idx.conn
        .execute(
            "INSERT INTO kv_blocks
             (hash, layout_key, path, model_id, kv_quant, byte_size, last_used)
             VALUES
             ('x', ?1, '/tmp/x.kvb', 'm', 'k', 500, 1),
             ('y', ?1, '/tmp/y.kvb', ?2, 'k', 500, 2)",
            params![LK_A as i64, "m"],
        )
        .unwrap();
    let evicted = idx.evict_lru_until(0).unwrap();
    assert_eq!(evicted.paths.len(), 2);
    assert_eq!(evicted.paths[0], PathBuf::from("/tmp/x.kvb"));
    assert_eq!(evicted.paths[1], PathBuf::from("/tmp/y.kvb"));
    assert_eq!(evicted.total_bytes_after, 0);
}

// ── prune_missing ─────────────────────────────────────────────────────────

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prune_missing_drops_absent_files() {
    let tmp = TempDir::new().unwrap();
    let real = tmp.path().join("real.kvb");
    fs::write(&real, b"data").unwrap();

    let idx = open();
    idx.record(&hash_to_hex(1), LK_A, &real, "m", "k", 4)
        .unwrap();
    idx.record(
        &hash_to_hex(2),
        LK_A,
        &PathBuf::from("/nonexistent/ghost.kvb"),
        "m",
        "k",
        99,
    )
    .unwrap();

    let pruned = idx.prune_missing().unwrap();
    assert_eq!(pruned, 1);

    assert!(idx.lookup(&hash_to_hex(1), LK_A).unwrap().is_some());
    assert!(idx.lookup(&hash_to_hex(2), LK_A).unwrap().is_none());
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prune_missing_on_empty_index_returns_zero() {
    let idx = open();
    assert_eq!(idx.prune_missing().unwrap(), 0);
}

// ── open_at + schema_version ─────────────────────────────────────────────

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn open_at_tempfile_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");

    {
        let idx = SsdKvIndex::open_at(&db).unwrap();
        idx.record(
            &hash_to_hex(42),
            LK_A,
            &PathBuf::from("/tmp/blk.kvb"),
            "Arch/snap",
            "planar",
            512,
        )
        .unwrap();
    }

    // Re-open — schema version accepted + row persists.
    let idx2 = SsdKvIndex::open_at(&db).unwrap();
    let row = idx2
        .lookup(&hash_to_hex(42), LK_A)
        .unwrap()
        .expect("persisted");
    assert_eq!(row.byte_size, 512);
    assert_eq!(row.kv_quant, "planar");
    assert_eq!(row.layout_key, LK_A);
}

/// unit test #5: an existing DB tagged with an unknown `schema_version`
/// surfaces `SchemaMismatch`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn schema_mismatch_on_unknown_version() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");

    // Seed a DB that has the schema_version table but holds version 99.
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(SCHEMA_PRAGMAS).unwrap();
        conn.execute_batch(SCHEMA_TABLES).unwrap();
        conn.execute("INSERT INTO schema_version (version) VALUES (99)", [])
            .unwrap();
    }

    match SsdKvIndex::open_at(&db) {
        Err(SsdKvIndexError::SchemaMismatch { found, expected }) => {
            assert_eq!(found, 99);
            assert_eq!(expected, SCHEMA_VERSION);
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

/// A v4 index must be refused, not adopted.
///
/// v4 and v5 have the same table shape and the same `(hash, layout_key)` key,
/// so a v4 block still *hits*. What changed is the dtype of a persisted
/// quantization parameter: the fused `rot_k` quantize kernel used to spill f32
/// scales where `mx.quantize` produces them at K's width, and safetensors hands
/// them back as f32. `quantized_matmul` then takes its operand width from those
/// scales and re-imposes the whole-graph f32 promotion for every request served
/// off that block. Nothing about the shape can catch that — only the version.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn schema_mismatch_on_v4_f32_scale_payload() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");

    {
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(SCHEMA_PRAGMAS).unwrap();
        conn.execute_batch(SCHEMA_TABLES).unwrap();
        conn.execute("INSERT INTO schema_version (version) VALUES (4)", [])
            .unwrap();
    }

    match SsdKvIndex::open_at(&db) {
        Err(SsdKvIndexError::SchemaMismatch { found, expected }) => {
            assert_eq!(found, 4, "a v4 index must be reported as v4");
            assert_eq!(
                expected, SCHEMA_VERSION,
                "the bump is what routes v4 namespaces through the wipe"
            );
            // Compile-time: lowering the version back to 4 would make this
            // test adopt the very blocks it exists to refuse, and the failure
            // should land at build time rather than as a confusing runtime
            // mismatch.
            const {
                assert!(
                    SCHEMA_VERSION > 4,
                    "SCHEMA_VERSION was lowered back to 4: pre-fix blocks carrying \
                     f32 rot_k scales would be adopted again and re-impose the \
                     promotion"
                );
            }
        }
        other => panic!("expected SchemaMismatch for a v4 index, got {other:?}"),
    }
}

/// A v5 index must be refused, not adopted.
///
/// v5 and v6 have the same table shape and the same `(hash, layout_key)` key.
/// What changed is the set of layer storage tags a block may carry: `rot_k_tq4v`
/// was withdrawn, and its blocks now fail the tag match on read. They are
/// unreachable by layout key too, but that guarantee lives in a formula in
/// another crate — the version is the local one.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn schema_mismatch_on_v5_retired_codec_tag() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");

    {
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(SCHEMA_PRAGMAS).unwrap();
        conn.execute_batch(SCHEMA_TABLES).unwrap();
        conn.execute("INSERT INTO schema_version (version) VALUES (5)", [])
            .unwrap();
    }

    match SsdKvIndex::open_at(&db) {
        Err(SsdKvIndexError::SchemaMismatch { found, expected }) => {
            assert_eq!(found, 5, "a v5 index must be reported as v5");
            assert_eq!(
                expected, SCHEMA_VERSION,
                "the bump is what routes v5 namespaces through the wipe"
            );
            const {
                assert!(
                    SCHEMA_VERSION > 5,
                    "SCHEMA_VERSION was lowered back to 5: blocks tagged with a \
                     retired codec would be adopted again and fail on read"
                );
            }
        }
        other => panic!("expected SchemaMismatch for a v5 index, got {other:?}"),
    }
}

/// A DB whose `kv_blocks` table exists but has NO `schema_version` table
/// is the pre-release v1 layout. `install_config` was supposed to wipe
/// such namespaces; if `open_at` is asked to consume one directly it must
/// surface `SchemaMismatch { found: 1, .. }`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn schema_mismatch_on_pre_release_v1() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("index.db");

    // Seed a v1-shaped DB by hand (old column layout, no schema_version).
    {
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(SCHEMA_PRAGMAS).unwrap();
        conn.execute(
            "CREATE TABLE kv_blocks (
                hash       TEXT PRIMARY KEY,
                path       TEXT NOT NULL,
                model_id   TEXT NOT NULL,
                kv_quant   TEXT NOT NULL,
                byte_size  INTEGER NOT NULL,
                last_used  INTEGER NOT NULL
            )",
            [],
        )
        .unwrap();
    }

    match SsdKvIndex::open_at(&db) {
        Err(SsdKvIndexError::SchemaMismatch { found, expected }) => {
            assert_eq!(found, 1);
            assert_eq!(expected, SCHEMA_VERSION);
        }
        other => panic!("expected SchemaMismatch v1, got {other:?}"),
    }
}

// ── hash_to_hex ───────────────────────────────────────────────────────────

#[test]
fn hash_to_hex_format() {
    assert_eq!(hash_to_hex(0), "0000000000000000");
    assert_eq!(hash_to_hex(u64::MAX), "ffffffffffffffff");
    assert_eq!(hash_to_hex(0xdeadbeef_cafebabe), "deadbeefcafebabe");
}

// ── evict_pool_lru_until ─────────────────────────────────────────

/// Seed a namespace with three real `.kvb` files (1000 bytes each) at the
/// given `last_used` timestamps, returning the namespace dir.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn seed_ns_rows(kv_root: &Path, ns: &str, rows: &[(&str, u64 /* last_used */)]) -> PathBuf {
    let ns_dir = kv_root.join(ns);
    fs::create_dir_all(&ns_dir).unwrap();
    let db_path = ns_dir.join("index.db");
    let idx = SsdKvIndex::open_at(&db_path).unwrap();
    for (name, last_used) in rows {
        let p = ns_dir.join(format!("{name}.kvb"));
        fs::write(&p, vec![0u8; 1000]).unwrap();
        idx.conn
            .execute(
                "INSERT INTO kv_blocks
                 (hash, layout_key, path, model_id, kv_quant, byte_size, last_used)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    *name,
                    LK_A as i64,
                    p.to_string_lossy(),
                    ns,
                    "k8v8",
                    1000_i64,
                    *last_used as i64
                ],
            )
            .unwrap();
    }
    ns_dir
}

/// test #1: two namespaces, 3 rows each with staggered `last_used`.
/// Global budget 3500 → require evicting 3 rows. Must be the 3 globally
/// oldest (lowest last_used) across the union; never drain one namespace
/// while the other still has older rows.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn evict_pool_lru_oldest_first_across_namespaces() {
    let tmp = TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();

    // alpha: last_used 10, 50, 90 (rows a1, a2, a3).
    // beta: last_used 20, 60, 80 (rows b1, b2, b3).
    let alpha = seed_ns_rows(&kv_root, "alpha", &[("a1", 10), ("a2", 50), ("a3", 90)]);
    let beta = seed_ns_rows(&kv_root, "beta", &[("b1", 20), ("b2", 60), ("b3", 80)]);

    // Pool starts at 6000 bytes. Budget 3500 → must drop 3 rows (3000
    // freed → pool 3000 ≤ 3500).
    let report = evict_pool_lru_until(&kv_root, 3500).unwrap();
    assert_eq!(report.blocks_evicted, 3, "must evict exactly 3 rows");
    assert_eq!(report.bytes_freed, 3000);
    assert_eq!(
        report.namespaces_touched, 2,
        "both namespaces must contribute to eviction"
    );

    // Oldest-first across union: a1(10), b1(20), a2(50). a3, b2, b3 survive.
    let idx_a = SsdKvIndex::open_at(&alpha.join("index.db")).unwrap();
    let idx_b = SsdKvIndex::open_at(&beta.join("index.db")).unwrap();
    assert!(idx_a.lookup("a1", LK_A).unwrap().is_none(), "a1 evicted");
    assert!(idx_b.lookup("b1", LK_A).unwrap().is_none(), "b1 evicted");
    assert!(idx_a.lookup("a2", LK_A).unwrap().is_none(), "a2 evicted");
    assert!(idx_a.lookup("a3", LK_A).unwrap().is_some(), "a3 survives");
    assert!(idx_b.lookup("b2", LK_A).unwrap().is_some(), "b2 survives");
    assert!(idx_b.lookup("b3", LK_A).unwrap().is_some(), "b3 survives");

    // .kvb files removed for evicted rows, present for survivors.
    assert!(!alpha.join("a1.kvb").exists());
    assert!(!beta.join("b1.kvb").exists());
    assert!(!alpha.join("a2.kvb").exists());
    assert!(alpha.join("a3.kvb").exists());
    assert!(beta.join("b2.kvb").exists());
    assert!(beta.join("b3.kvb").exists());
}

/// test #2: budget 0 (default) is a strict no-op even with multiple
/// namespaces present.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn evict_pool_zero_budget_is_noop() {
    let tmp = TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    seed_ns_rows(&kv_root, "alpha", &[("a1", 1), ("a2", 2)]);
    let report = evict_pool_lru_until(&kv_root, 0).unwrap();
    assert_eq!(report.blocks_evicted, 0);
    assert_eq!(report.bytes_freed, 0);
    assert_eq!(report.namespaces_touched, 0);
}

/// test #3: when pool already within budget, no-op.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn evict_pool_within_budget_is_noop() {
    let tmp = TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    seed_ns_rows(&kv_root, "alpha", &[("a1", 1)]);
    // 1000 bytes pool, budget 10000 — nothing to do.
    let report = evict_pool_lru_until(&kv_root, 10_000).unwrap();
    assert_eq!(report.blocks_evicted, 0);
}

/// test #4: single-namespace + zero global budget = strict no-op,
/// matches the documented "no global ceiling" semantics. The per-namespace
/// eviction path is unaffected.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn single_namespace_no_global_budget_noop() {
    let tmp = TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    let alpha = seed_ns_rows(&kv_root, "solo", &[("x1", 1), ("x2", 2), ("x3", 3)]);
    let report = evict_pool_lru_until(&kv_root, 0).unwrap();
    assert_eq!(report.blocks_evicted, 0);
    let idx = SsdKvIndex::open_at(&alpha.join("index.db")).unwrap();
    assert_eq!(idx.total_bytes().unwrap(), 3000);
}

/// The pool sweep merges rows from independent namespace databases, so its
/// ordering has to come from the stamp itself — no per-database counter could
/// compare `alpha`'s rows against `beta`'s.
///
/// Recorded and touched inside one wall-clock second, as a live pool is. At
/// second granularity all six rows tie and the sweep falls back to its
/// `(namespace, hash)` tiebreak, which evicts `alpha/a0`, `alpha/a1`,
/// `alpha/a2` — alphabetically first, and here the three *most* recently used.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "temp-dir namespace databases and their .kvb files, all created by this test; a failure is a broken fixture and must abort rather than be reported as a passing sweep"
)]
fn evict_pool_lru_orders_across_namespaces_within_one_second() {
    let tmp = TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();

    // Recorded oldest-first across both namespaces, interleaved.
    let alpha = seed_ns_recorded(&kv_root, "alpha", &["a0", "a1", "a2"]);
    let beta = seed_ns_recorded(&kv_root, "beta", &["b0", "b1", "b2"]);

    // Re-use alpha's blocks: they become the three most recently used in the
    // pool. LRU order is now b0, b1, b2, a0, a1, a2.
    {
        let idx_a = SsdKvIndex::open_at(&alpha.join("index.db")).unwrap();
        for name in ["a0", "a1", "a2"] {
            idx_a.touch(name, LK_A).unwrap();
        }
    }

    // 6000 bytes down to 3500 evicts three rows: the beta blocks.
    let report = evict_pool_lru_until(&kv_root, 3500).unwrap();
    assert_eq!(report.blocks_evicted, 3);
    assert_eq!(
        report.namespaces_touched, 1,
        "only beta holds least-recently-used blocks, so only beta may be touched"
    );

    let idx_a = SsdKvIndex::open_at(&alpha.join("index.db")).unwrap();
    let idx_b = SsdKvIndex::open_at(&beta.join("index.db")).unwrap();
    for name in ["a0", "a1", "a2"] {
        assert!(
            idx_a.lookup(name, LK_A).unwrap().is_some(),
            "{name} was used after every beta block and must survive"
        );
        assert!(alpha.join(format!("{name}.kvb")).exists());
    }
    for name in ["b0", "b1", "b2"] {
        assert!(
            idx_b.lookup(name, LK_A).unwrap().is_none(),
            "{name} evicted"
        );
        assert!(!beta.join(format!("{name}.kvb")).exists());
    }
}

/// Like [`seed_ns_rows`] but stamps each row through the public `record` path
/// instead of hand-writing a `last_used` value, so the rows carry whatever
/// granularity the index actually issues.
#[allow(
    clippy::unwrap_used,
    reason = "fixture setup: creates a namespace dir, its index and its .kvb files under a temp dir; a failure here is a broken fixture, not a condition under test"
)]
fn seed_ns_recorded(kv_root: &Path, ns: &str, names: &[&str]) -> PathBuf {
    let ns_dir = kv_root.join(ns);
    fs::create_dir_all(&ns_dir).unwrap();
    let idx = SsdKvIndex::open_at(&ns_dir.join("index.db")).unwrap();
    for name in names {
        let p = ns_dir.join(format!("{name}.kvb"));
        fs::write(&p, vec![0u8; 1000]).unwrap();
        idx.record(name, LK_A, &p, ns, "k8v8", 1000).unwrap();
    }
    ns_dir
}
