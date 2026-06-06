use super::*;
use rmlx_kv_quant::kvcache::KvCache;
use tempfile::TempDir;

const MODEL_ID: &str = "Gemma4ForConditionalGeneration/test-snap";

/// A minimal serializable job: one `None`-quant KvCache layer (geometry
/// only, no tensor data) plus no GDN state. `write_caches` produces a valid
/// geometry-only `.kvb`, which is all the spill-capture path needs to
/// verify the file + index row appear.
/// spill jobs carry a `layout_key`; tests use a deterministic
/// placeholder so the recorded row's `layout_key` column is observable.
const TEST_LAYOUT_KEY: u64 = 0xa55a_5aa5_d00d_d00du64;

fn job(hash: u64) -> SpillJob {
    SpillJob {
        hash,
        layout_key: TEST_LAYOUT_KEY,
        model_id: MODEL_ID.into(),
        kv_quant: KvQuant::None,
        kv_caches: vec![KvCache::with_quant(KvQuant::None)],
        lin_caches: Vec::new(),
    }
}

/// (a): a spilled job lands as a `.kvb` file on disk and a row in the
/// SSD index, keyed by the job's hash. End-to-end through the real drain
/// thread + `write_caches` + `SsdKvIndex::record`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn spill_writes_file_and_index_row() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    let hash = 0xdead_beef_0000_0001u64;
    let (spiller, handle) =
        SsdSpiller::spawn_with_index(MODEL_ID, TEST_LAYOUT_KEY, Device::Cpu, dir.clone(), index);
    spiller.try_spill(job(hash));
    drop(spiller); // close the channel so the drain thread exits
    handle.join().unwrap();

    // File present at <dir>/<hex>.kvb.
    let hex = hash_to_hex(hash);
    let kvb = dir.join(format!("{hex}.kvb"));
    assert!(
        kvb.exists(),
        "spilled .kvb file must exist at {}",
        kvb.display()
    );

    // Index row present with the matching hash + model_id + layout_key.
    let idx = SsdKvIndex::open_at(&db).unwrap();
    let row = idx
        .lookup(&hex, TEST_LAYOUT_KEY)
        .unwrap()
        .expect("index row for spilled block");
    assert_eq!(row.hash, hex);
    assert_eq!(row.layout_key, TEST_LAYOUT_KEY);
    assert_eq!(row.model_id, MODEL_ID);
    assert_eq!(row.kv_quant, KvQuant::None.to_string());
    assert!(
        row.byte_size > 0,
        "recorded byte_size must be the file size"
    );
}

/// (c): a spill into an unwritable directory `warn!`s and drops the
/// job — the drain thread does NOT panic and continues (a subsequent job
/// to a good dir still succeeds). The serialize error is contained.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn spill_failure_does_not_panic_and_drains_on() {
    // Point the drain at a path under a regular *file* so dir.join(...)
    // is unwritable → serialize fails.
    let tmp = TempDir::new().unwrap();
    let not_a_dir = tmp.path().join("iam_a_file");
    std::fs::write(&not_a_dir, b"x").unwrap();
    let bad_dir = not_a_dir.join("nope"); // child of a file → unwritable

    let db = tmp.path().join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();
    let (spiller, handle) =
        SsdSpiller::spawn_with_index(MODEL_ID, TEST_LAYOUT_KEY, Device::Cpu, bad_dir, index);

    // This job must fail to serialize but not crash the thread.
    spiller.try_spill(job(0x1111));
    drop(spiller);
    // join() returning Ok proves the drain thread did not panic.
    handle
        .join()
        .expect("drain thread must not panic on spill failure");

    // No index row was recorded for the failed job.
    let idx = SsdKvIndex::open_at(&db).unwrap();
    assert!(
        idx.lookup(&hash_to_hex(0x1111), TEST_LAYOUT_KEY)
            .unwrap()
            .is_none(),
        "failed spill must not record an index row"
    );
}

/// (b): `try_spill` is non-blocking even when the drain is stalled.
/// We stall the drain on a barrier, flood the bounded channel past its
/// capacity, and assert every `try_spill` returns promptly (no hang) — the
/// overflow jobs are dropped, not awaited. A blocking implementation would
/// deadlock here and the test would time out.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn try_spill_does_not_block_when_drain_stalled() {
    use std::sync::mpsc::channel;

    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let index = SsdKvIndex::open_at(&dir.join("index.db")).unwrap();

    // Build a spiller whose drain thread blocks before processing anything.
    let (tx, rx) = sync_channel::<SpillJob>(SPILL_CHANNEL_CAP);
    let (gate_tx, gate_rx) = channel::<()>();
    let dir2 = dir;
    let handle = thread::spawn(move || {
        // Block until released — drain is fully stalled.
        let _ = gate_rx.recv();
        for j in rx {
            drain_one(&index, &dir2, Device::Cpu, j);
        }
    });
    let spiller = SsdSpiller {
        tx,
        model_id: MODEL_ID.into(),
        layout_key: TEST_LAYOUT_KEY,
    };

    // Flood well past the channel capacity; none of these may block.
    for i in 0..(SPILL_CHANNEL_CAP as u64 * 4) {
        spiller.try_spill(job(0x2000 + i));
    }
    // If we reached here, no try_spill blocked. Release + clean up.
    let _ = gate_tx.send(());
    drop(spiller);
    handle.join().unwrap();
}

// Touch `spawn` (the production constructor) so it isn't flagged unused in
// the test build, and confirm it returns a usable handle whose drain exits
// cleanly when the sender is dropped.
#[test]
fn spawn_returns_handle_that_exits_on_drop() {
    let spiller = SsdSpiller::spawn(format!("{MODEL_ID}-spawn"), TEST_LAYOUT_KEY, Device::Cpu);
    assert_eq!(spiller.model_id(), format!("{MODEL_ID}-spawn"));
    assert_eq!(spiller.layout_key(), TEST_LAYOUT_KEY);
    drop(spiller); // closing the channel lets the production drain thread exit
}

/// SSD-tier observability (step2-A): after a successful `drain_one`, the
/// event recorder sees exactly one `ssd_spill` row with `dur_us > 0`,
/// `bytes > 0`, and the three sub-phase fields (`dur_serialize_us`,
/// `dur_write_us`, `dur_index_us`) present (non-zero is not guaranteed on
/// very fast drives, but the JSON notes must parse and the row must exist).
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn drain_one_emits_ssd_spill_event() {
    use rmlx_metrics::events::EventRecorder;
    use rusqlite::Connection;
    use std::sync::Arc;

    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db_path = dir.join("events.db");
    let index = SsdKvIndex::open_at(&dir.join("index.db")).unwrap();

    // Open an event recorder backed by a hermetic temp DB.
    let rec =
        Arc::new(EventRecorder::open_at(&db_path, "spill-test-run").expect("open event recorder"));

    let hash = 0xabe0_cafe_0000_0001u64;
    let (spiller, handle) = SsdSpiller::spawn_with_index_and_recorder(
        MODEL_ID,
        TEST_LAYOUT_KEY,
        Device::Cpu,
        dir,
        index,
        Arc::clone(&rec),
    );
    spiller.try_spill(job(hash));
    drop(spiller);
    handle.join().unwrap();

    // Verify exactly one `ssd_spill` row in the events table.
    let conn = Connection::open(&db_path).expect("reopen events DB");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE op = 'ssd_spill'",
            [],
            |r| r.get(0),
        )
        .expect("query ssd_spill count");
    assert_eq!(count, 1, "exactly one ssd_spill event row expected");

    // Verify the row has dur_us > 0 and bytes > 0.
    let (value, notes): (f64, String) = conn
        .query_row(
            "SELECT value, notes FROM events WHERE op = 'ssd_spill' LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("query ssd_spill row");
    assert!(value > 0.0, "dur_us must be > 0, got {value}");

    // notes is a JSON object; verify the `bytes` field is present and > 0.
    let parsed: serde_json::Value = serde_json::from_str(&notes).expect("notes must be valid JSON");
    let bytes = parsed["bytes"].as_u64().expect("notes.bytes must be u64");
    assert!(bytes > 0, "spilled bytes must be > 0, got {bytes}");

    // Verify the three sub-phase fields are present (values may be 0 on
    // fast in-memory FS, but the keys must exist).
    assert!(
        parsed.get("dur_serialize_us").is_some(),
        "notes must have dur_serialize_us"
    );
    assert!(
        parsed.get("dur_write_us").is_some(),
        "notes must have dur_write_us"
    );
    assert!(
        parsed.get("dur_index_us").is_some(),
        "notes must have dur_index_us"
    );
}
