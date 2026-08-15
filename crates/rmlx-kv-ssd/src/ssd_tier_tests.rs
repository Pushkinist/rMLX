use std::collections::BTreeMap;

use super::*;

#[test]
fn namespace_uses_project_when_set() {
    let cfg = SsdTierConfig {
        per_namespace_budget_bytes: 1024,
        global_budget_bytes: 0,
        default_namespace: Some("proj".into()),
        per_project_budgets: BTreeMap::default(),
    };
    assert_eq!(namespace_for(&cfg, "some-model"), "proj");
}

#[test]
fn namespace_falls_back_to_model_id() {
    let cfg = SsdTierConfig {
        per_namespace_budget_bytes: 1024,
        global_budget_bytes: 0,
        default_namespace: None,
        per_project_budgets: BTreeMap::default(),
    };
    assert_eq!(namespace_for(&cfg, "some-model"), "some-model");
}

// ── compute_layout_key ────────────────────────────────────────────

/// unit test #1: deterministic across calls for a fixed tuple.
#[test]
fn compute_layout_key_is_deterministic() {
    let a = compute_layout_key("Qwen3ForCausalLM", 32, 4, 128, KvQuant::K8V8);
    let b = compute_layout_key("Qwen3ForCausalLM", 32, 4, 128, KvQuant::K8V8);
    assert_eq!(a, b, "layout_key must be deterministic for a fixed tuple");
}

/// unit test #1 (property): every distinct
/// `(arch, n_layers, n_kv_heads, head_dim, kv_quant)` tuple yields a
/// distinct u64. Enumerate a 2×2×2×2×3 product and assert no collisions.
#[test]
fn compute_layout_key_is_distinct_across_tuples() {
    let archs = ["Qwen3ForCausalLM", "Gemma4ForConditionalGeneration"];
    let layers = [32usize, 40];
    let kv_heads = [4usize, 8];
    let head_dims = [64usize, 128];
    let kv_quants = [KvQuant::K8V4, KvQuant::K8V8, KvQuant::None];

    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for arch in &archs {
        for &nl in &layers {
            for &nk in &kv_heads {
                for &hd in &head_dims {
                    for &q in &kv_quants {
                        let k = compute_layout_key(arch, nl, nk, hd, q);
                        assert!(
                            seen.insert(k),
                            "duplicate layout_key {k:016x} for ({arch}, {nl}, {nk}, {hd}, {q})"
                        );
                    }
                }
            }
        }
    }
}

// ── wipe_stale_schema_namespaces ──────────────────────────────────

/// Seed a v1-shaped namespace dir on disk under `kv_root`: a dummy `.kvb`
/// file plus an `index.db` whose `kv_blocks` table matches the pre-
/// schema (no `schema_version` table). Returns the namespace path.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn seed_v1_namespace(kv_root: &Path, ns: &str) -> std::path::PathBuf {
    let ns_dir = kv_root.join(ns);
    std::fs::create_dir_all(&ns_dir).unwrap();
    std::fs::write(ns_dir.join("dummy.kvb"), b"placeholder").unwrap();
    let db_path = ns_dir.join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE kv_blocks (
            hash       TEXT PRIMARY KEY,
            path       TEXT NOT NULL,
            model_id   TEXT NOT NULL,
            kv_quant   TEXT NOT NULL,
            byte_size  INTEGER NOT NULL,
            last_used  INTEGER NOT NULL
        );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO kv_blocks VALUES ('h', '/tmp/dummy.kvb', 'arch/snap', 'k8v8', 1024, 1)",
        [],
    )
    .unwrap();
    ns_dir
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn seed_current_schema_namespace(kv_root: &Path, ns: &str) -> std::path::PathBuf {
    let ns_dir = kv_root.join(ns);
    std::fs::create_dir_all(&ns_dir).unwrap();
    let db_path = ns_dir.join("index.db");
    // SsdKvIndex::open_at seeds the current schema on a non-existent file.
    let _ = SsdKvIndex::open_at(&db_path).expect("seed current schema");
    ns_dir
}

/// Seed a namespace at the **v2** table shape, tagged with an explicit
/// `schema_version` of `version`, plus a dummy `.kvb`.
///
/// The DDL is written out here rather than reused from the production
/// constant, the same way `seed_v1_namespace` does: a migration fixture has to
/// pin the historical shape it claims to be, or a later schema bump silently
/// turns it into a test of the current shape wearing an old version number.
/// v2 and v3 differ only in the unit stored in `last_used`, so this is also
/// what a real v2 file looks like.
#[allow(
    clippy::unwrap_used,
    reason = "fixture setup under a temp dir this test owns; a failure is a broken fixture, not a condition under test"
)]
fn seed_namespace_at_version(kv_root: &Path, ns: &str, version: i64) -> std::path::PathBuf {
    let ns_dir = kv_root.join(ns);
    std::fs::create_dir_all(&ns_dir).unwrap();
    std::fs::write(ns_dir.join("dummy.kvb"), b"placeholder").unwrap();
    let conn = rusqlite::Connection::open(ns_dir.join("index.db")).unwrap();
    conn.execute_batch(
        "CREATE TABLE kv_blocks (
            hash        TEXT    NOT NULL,
            layout_key  INTEGER NOT NULL,
            path        TEXT    NOT NULL,
            model_id    TEXT    NOT NULL,
            kv_quant    TEXT    NOT NULL,
            byte_size   INTEGER NOT NULL,
            last_used   INTEGER NOT NULL,
            PRIMARY KEY (hash, layout_key)
        );
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY NOT NULL
        );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (?1)",
        rusqlite::params![version],
    )
    .unwrap();
    ns_dir
}

/// unit test #3: wipe-on-upgrade. Seed a v1 namespace, drive
/// `wipe_stale_schema_namespaces`, assert the dir is gone and a fresh
/// `SsdKvIndex::open_at` against the (now-absent) path succeeds.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wipe_removes_pre_release_v1_namespace() {
    let tmp = tempfile::TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    let ns_dir = seed_v1_namespace(&kv_root, "wipe-v1");

    assert!(ns_dir.exists(), "fixture should seed namespace dir");
    wipe_stale_schema_namespaces(&kv_root);
    assert!(!ns_dir.exists(), "v1 namespace must be removed");

    // Re-creating the namespace (as a real model load would) → fresh DB.
    std::fs::create_dir_all(&ns_dir).unwrap();
    let db_path = ns_dir.join("index.db");
    let idx = SsdKvIndex::open_at(&db_path).expect("fresh open");
    // Sanity: empty index.
    assert_eq!(idx.total_bytes().unwrap(), 0);
}

/// A superseded-but-tagged namespace has to be reclaimed too, not only the
/// untagged v1 layout. `last_used` moved from seconds to microseconds without
/// changing the table shape, so a v2 DB opens cleanly at the SQL level and its
/// rows would silently mix two units three orders of magnitude apart.
///
/// Left in place it is dead weight: `SsdKvIndex::open` rejects the version, so
/// the namespace is disabled for the run and its `.kvb` bytes sit there with
/// nothing able to reclaim them. Only versions this binary supersedes are
/// wiped — see `wipe_keeps_a_namespace_from_a_newer_binary`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture setup under a temp dir this test owns; a failure is a broken fixture, not a condition under test"
)]
fn wipe_removes_superseded_schema_namespace() {
    let tmp = tempfile::TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    let stale = seed_namespace_at_version(&kv_root, "wipe-v2", 2);
    let current = seed_current_schema_namespace(&kv_root, "keep-current");

    wipe_stale_schema_namespaces(&kv_root);

    assert!(
        !stale.exists(),
        "superseded-schema namespace must be removed"
    );
    assert!(
        current.exists(),
        "current-schema namespace must survive the same pass"
    );
}

/// A namespace written by a **newer** binary must survive. Two rMLX builds on
/// one machine at different schema versions is the ordinary case here (a
/// tap-installed release beside a dev build), and wiping forward means the
/// older binary silently destroys the newer one's whole KV pool on every
/// alternate boot. Stranded bytes are a reclamation problem; they do not
/// license deleting a cache this binary simply cannot read.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture setup under a temp dir this test owns; a failure is a broken fixture, not a condition under test"
)]
fn wipe_keeps_a_namespace_from_a_newer_binary() {
    let tmp = tempfile::TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    let newer = seed_namespace_at_version(
        &kv_root,
        "from-the-future",
        crate::ssd_index::SCHEMA_VERSION + 1,
    );

    wipe_stale_schema_namespaces(&kv_root);

    assert!(
        newer.exists(),
        "a namespace at a newer schema must be left for the binary that wrote it"
    );
    assert!(
        newer.join("index.db").exists(),
        "its index must be intact, not merely its directory"
    );
}

/// Schema creation must be atomic. `SCHEMA_TABLES` creates `kv_blocks` and
/// `schema_version`; the version *row* lands later. A concurrent
/// `wipe_stale_schema_namespaces` that reads the DB in between sees a
/// `kv_blocks` table and an empty `schema_version`, which
/// `read_schema_version` reports as version 0 — stale — and the pass deletes a
/// namespace another process is in the middle of creating.
///
/// The property is that the half-built state is never observable from another
/// connection, which SQLite guarantees once the three steps share one
/// transaction.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "temp-dir SQLite fixtures created by this test; a failure is a broken fixture, not a condition under test"
)]
#[allow(
    clippy::expect_used,
    reason = "thread join: a panic in the worker is a test failure and must surface, not be swallowed"
)]
fn schema_init_never_exposes_tables_without_their_version_row() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    const ROUNDS: usize = 60;

    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path().to_path_buf();
    let observed_half_built = Arc::new(AtomicU64::new(0));
    let rounds_run = Arc::new(AtomicU64::new(0));

    let watcher = {
        let (root, observed, rounds_run) = (
            root.clone(),
            Arc::clone(&observed_half_built),
            Arc::clone(&rounds_run),
        );
        std::thread::spawn(move || {
            while rounds_run.load(Ordering::Acquire) < ROUNDS as u64 {
                for i in 0..ROUNDS {
                    let db = root.join(format!("ns{i}")).join("index.db");
                    if !db.exists() {
                        continue;
                    }
                    let Ok(conn) = rusqlite::Connection::open(&db) else {
                        continue;
                    };
                    let has_blocks = conn
                        .query_row(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kv_blocks'",
                            [],
                            |_| Ok(true),
                        )
                        .unwrap_or(false);
                    if !has_blocks {
                        continue;
                    }
                    // Exactly what `is_stale_schema` asks. Version 0 means the
                    // table exists but carries no row yet.
                    if matches!(crate::ssd_index::read_schema_version(&conn), Ok(Some(0))) {
                        observed.fetch_add(1, Ordering::AcqRel);
                    }
                }
                std::thread::yield_now();
            }
        })
    };

    for i in 0..ROUNDS {
        let dir = root.join(format!("ns{i}"));
        std::fs::create_dir_all(&dir).unwrap();
        let _idx = SsdKvIndex::open_at(&dir.join("index.db")).unwrap();
        rounds_run.fetch_add(1, Ordering::AcqRel);
    }
    watcher.join().expect("watcher thread panicked");

    assert_eq!(
        observed_half_built.load(Ordering::Acquire),
        0,
        "another process observed kv_blocks without its schema_version row; the \
         wipe pass would have deleted a namespace mid-creation"
    );
}

/// unit test #4: no-op on a namespace already at the current schema.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wipe_is_noop_on_current_schema_namespace() {
    let tmp = tempfile::TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    let ns_dir = seed_current_schema_namespace(&kv_root, "wipe-current");
    let db_path = ns_dir.join("index.db");

    let before_meta = std::fs::metadata(&db_path).unwrap();
    let before_mtime = before_meta.modified().unwrap();
    let before_size = before_meta.len();

    wipe_stale_schema_namespaces(&kv_root);

    assert!(ns_dir.exists(), "namespace must survive the wipe");
    assert!(db_path.exists(), "index.db must survive the wipe");
    let after_meta = std::fs::metadata(&db_path).unwrap();
    // mtime must not regress; size must be unchanged.
    assert_eq!(
        after_meta.len(),
        before_size,
        "index.db size must not change"
    );
    assert!(
        after_meta.modified().unwrap() >= before_mtime,
        "index.db mtime must not regress"
    );
}

/// Startup maintenance: an over-budget namespace index is evicted until the
/// on-disk footprint is within budget, and the evicted `.kvb` files are
/// deleted from disk. Also exercises `prune_missing`.
///
/// Hermetic via an explicit temp dir injected into `SsdKvIndex::open_at` and
/// `startup_maintenance` — the routine takes its opened index by reference, so
/// the test never resolves the process-global `paths::home()` `OnceLock` and is
/// immune to its set-once-per-process ordering hazard under parallel runs.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn startup_maintenance_prunes_and_evicts_to_budget() {
    let tmp = tempfile::TempDir::new().unwrap();
    let ns = "test-ns";
    let dir = tmp.path().join(ns);
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("index.db");
    let lk: u64 = 0xa1b2_c3d4_e5f6_a7b8;

    {
        let idx = SsdKvIndex::open_at(&db_path).unwrap();
        // Three 1000-byte real files + one ghost row (file never written).
        for name in ["aaa", "bbb", "ccc"] {
            let p = dir.join(format!("{name}.kvb"));
            std::fs::write(&p, vec![0u8; 1000]).unwrap();
            idx.record(name, lk, &p, ns, "k8v8", 1000).unwrap();
        }
        idx.record("ghost", lk, &dir.join("ghost.kvb"), ns, "k8v8", 9999)
            .unwrap();
        assert_eq!(idx.total_bytes().unwrap(), 3000 + 9999);
    }

    let idx = SsdKvIndex::open_at(&db_path).unwrap();
    startup_maintenance(&idx, ns, 1500);

    assert!(idx.lookup("ghost", lk).unwrap().is_none());
    assert!(idx.total_bytes().unwrap() <= 1500);
    let surviving_files = ["aaa", "bbb", "ccc"]
        .iter()
        .filter(|n| dir.join(format!("{n}.kvb")).exists())
        .count();
    assert!(surviving_files <= 1);
}

// ── Runtime budget enforcement ────────────────────────────────────────────

/// Seed `dir`'s index with `n` blocks of `size` bytes each, returning the
/// opened index. Every block has a real file on disk so eviction has something
/// to remove.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn seed_blocks(dir: &Path, ns: &str, lk: u64, n: usize, size: usize) -> SsdKvIndex {
    std::fs::create_dir_all(dir).unwrap();
    let idx = SsdKvIndex::open_at(&dir.join("index.db")).unwrap();
    for i in 0..n {
        let name = format!("blk{i:04}");
        let p = dir.join(format!("{name}.kvb"));
        std::fs::write(&p, vec![0u8; size]).unwrap();
        idx.record(&name, lk, &p, ns, "k8v8", size as u64).unwrap();
    }
    idx
}

/// The tier only grew and never shrank between model loads: nothing in the
/// serving path called `evict_lru_until`, so a namespace that spilled past
/// `--kv-ssd-cache-gb` stayed past it until the next attach. This is the pass
/// that closes that, and the property it owns is "the footprint comes back
/// under the ceiling, and the bytes that left disk really left disk".
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn enforce_namespace_budget_brings_footprint_within_budget() {
    let tmp = tempfile::TempDir::new().unwrap();
    let ns = "budget-ns";
    let dir = tmp.path().join(ns);
    let lk: u64 = 0x0f0f_0f0f_0f0f_0f0f;
    let idx = seed_blocks(&dir, ns, lk, 5, 1000);
    assert_eq!(idx.total_bytes().unwrap(), 5000);

    let evicted = enforce_namespace_budget(&idx, ns, 2500);

    assert_eq!(
        evicted, 3,
        "5 x 1000 bytes down to 2500 evicts three blocks"
    );
    assert!(idx.total_bytes().unwrap() <= 2500);
    // The index and the disk must agree: no row may point at a file that the
    // pass deleted, and no deleted row may leave its file behind.
    assert_eq!(idx.prune_missing().unwrap(), 0, "no row lost its file");
    let files_left = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "kvb"))
        .count();
    assert_eq!(files_left, 2, "evicted blocks must be gone from disk");
}

/// Zero is the one budget where the two possible readings — "no ceiling" and
/// "keep nothing" — differ by the whole namespace. The tier-level routine reads
/// it as "no ceiling" for every caller, so no call site has to be wrapped to
/// survive it. (The literal reading still lives on the raw index API, which
/// `evict_zero_budget_evicts_all` pins.)
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn enforce_namespace_budget_ignores_a_zero_budget() {
    let tmp = tempfile::TempDir::new().unwrap();
    let ns = "zero-budget-ns";
    let lk: u64 = 0x1234;

    let dir = tmp.path().join("ns");
    let idx = seed_blocks(&dir, ns, lk, 3, 100);
    assert_eq!(enforce_namespace_budget(&idx, ns, 0), 0);
    assert_eq!(
        idx.total_bytes().unwrap(),
        300,
        "an unconfigured ceiling must not empty the namespace"
    );
}

/// `--kv-ssd-global-gb N --kv-ssd-cache-gb 0` turns the tier on, so the
/// namespace ceiling it resolves to has to be a real ceiling. Resolving it to
/// zero gave the attach path "delete everything" and the runtime path "never
/// enforce" off the same config.
#[test]
fn effective_namespace_budget_never_yields_zero_while_the_tier_is_on() {
    let cfg = |per_ns: u64, global: u64| SsdTierConfig {
        per_namespace_budget_bytes: per_ns,
        global_budget_bytes: global,
        default_namespace: None,
        per_project_budgets: BTreeMap::default(),
    };

    // No per-namespace ceiling: the global pool governs the namespace alone.
    assert_eq!(effective_namespace_budget(&cfg(0, 900)), 900);
    // No global pool: the per-namespace budget stands alone.
    assert_eq!(effective_namespace_budget(&cfg(700, 0)), 700);
    // Both set: the tighter of the two, either way round.
    assert_eq!(effective_namespace_budget(&cfg(700, 900)), 700);
    assert_eq!(effective_namespace_budget(&cfg(900, 700)), 700);
    // Both zero is the tier being off; nothing enforces, and the eviction
    // routine treats the zero as "no ceiling" rather than "keep nothing".
    assert_eq!(effective_namespace_budget(&cfg(0, 0)), 0);
}

/// unit test: double `install_config` returns `Err(SsdTierAlreadyInstalled)`.
///
/// Test process is shared, so this test (and only this test) drives the
/// process-global `CONFIG` `OnceLock`. No other test in this binary calls
/// `install_config`.
#[test]
fn install_config_double_call_returns_err() {
    let cfg = SsdTierConfig {
        per_namespace_budget_bytes: 0,
        global_budget_bytes: 0,
        default_namespace: None,
        per_project_budgets: BTreeMap::default(),
    };
    // First call must succeed (or the OnceLock was already set by a prior test run
    // in the same process — that is fine; what we care about is the second call).
    let _ = install_config(cfg.clone());
    // Second call must return Err(SsdTierAlreadyInstalled).
    match install_config(cfg) {
        Err(Error::SsdTierAlreadyInstalled) => {}
        other => panic!("expected Err(SsdTierAlreadyInstalled), got {other:?}"),
    }
}
