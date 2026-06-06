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

// ── wipe_pre_release_v1_namespaces ────────────────────────────────

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
fn seed_v2_namespace(kv_root: &Path, ns: &str) -> std::path::PathBuf {
    let ns_dir = kv_root.join(ns);
    std::fs::create_dir_all(&ns_dir).unwrap();
    let db_path = ns_dir.join("index.db");
    // SsdKvIndex::open_at seeds a fresh v2 schema on a non-existent file.
    let _ = SsdKvIndex::open_at(&db_path).expect("seed v2");
    ns_dir
}

/// unit test #3: wipe-on-upgrade. Seed a v1 namespace, drive
/// `wipe_pre_release_v1_namespaces`, assert the dir is gone and a fresh
/// `SsdKvIndex::open_at` against the (now-absent) path succeeds with v2.
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
    wipe_pre_release_v1_namespaces(&kv_root);
    assert!(!ns_dir.exists(), "v1 namespace must be removed");

    // Re-creating the namespace (as a real model load would) → fresh v2 DB.
    std::fs::create_dir_all(&ns_dir).unwrap();
    let db_path = ns_dir.join("index.db");
    let idx = SsdKvIndex::open_at(&db_path).expect("fresh v2 open");
    // Sanity: empty index.
    assert_eq!(idx.total_bytes().unwrap(), 0);
}

/// unit test #4: no-op on an already-v2 namespace.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wipe_is_noop_on_v2_namespace() {
    let tmp = tempfile::TempDir::new().unwrap();
    let kv_root = tmp.path().to_path_buf();
    let ns_dir = seed_v2_namespace(&kv_root, "wipe-v2");
    let db_path = ns_dir.join("index.db");

    let before_meta = std::fs::metadata(&db_path).unwrap();
    let before_mtime = before_meta.modified().unwrap();
    let before_size = before_meta.len();

    wipe_pre_release_v1_namespaces(&kv_root);

    assert!(ns_dir.exists(), "v2 namespace must survive the wipe");
    assert!(db_path.exists(), "v2 index.db must survive the wipe");
    let after_meta = std::fs::metadata(&db_path).unwrap();
    // mtime must not regress; size must be unchanged.
    assert_eq!(
        after_meta.len(),
        before_size,
        "v2 index.db size must not change"
    );
    assert!(
        after_meta.modified().unwrap() >= before_mtime,
        "v2 index.db mtime must not regress"
    );
}

/// Startup maintenance: an over-budget namespace index is evicted until the
/// on-disk footprint is within budget, and the evicted `.kvb` files are
/// deleted from disk. Also exercises `prune_missing`. Hermetic via an
/// `RMLX_HOME` tempdir.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn startup_maintenance_prunes_and_evicts_to_budget() {
    let tmp = tempfile::TempDir::new().unwrap();
    // SAFETY: edition 2021; single-threaded test, set before paths::home()
    // is first resolved in this process for this namespace.
    std::env::set_var("RMLX_HOME", tmp.path());

    let ns = "test-ns";
    let dir = rmlx_core::paths::kv_cache_dir(ns);
    // `rmlx_core::paths::home()` is a OnceLock — if a prior test in
    // this process resolved it before the `RMLX_HOME` set above, the dir
    // path resolves into the workspace `.rmlx/` and may carry a stale v1
    // schema DB. Wipe the namespace dir first so the fixture below starts
    // from a clean v2-schema state every time.
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    let lk: u64 = 0xa1b2_c3d4_e5f6_a7b8;

    {
        let idx = SsdKvIndex::open(ns).unwrap();
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

    startup_maintenance(ns, 1500);

    let idx2 = SsdKvIndex::open(ns).unwrap();
    assert!(idx2.lookup("ghost", lk).unwrap().is_none());
    assert!(idx2.total_bytes().unwrap() <= 1500);
    let surviving_files = ["aaa", "bbb", "ccc"]
        .iter()
        .filter(|n| dir.join(format!("{n}.kvb")).exists())
        .count();
    assert!(surviving_files <= 1);

    std::env::remove_var("RMLX_HOME");
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
