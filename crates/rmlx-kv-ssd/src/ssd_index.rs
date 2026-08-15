//! SQLite index of on-disk KV blocks (; layout-key disambiguation).
//!
//! [`SsdKvIndex`] tracks `.kvb` safetensors files written by 's
//! `KvBlockWriter`. Each row records the file path, identity fields, byte
//! size, and a `last_used` unix timestamp so that an LRU-by-size eviction
//! policy can be applied when the on-disk budget is exceeded.
//!
//! # DB location
//!
//! The index lives at `<rmlx_core::paths::kv_cache_dir(namespace)>/index.db`.
//! The path is always resolved through `rmlx_core::paths` — never a
//! hard-coded or CWD-relative string.
//!
//! # Schema (v2 — )
//!
//! ```text
//! kv_blocks (
//! hash TEXT NOT NULL, -- chained FNV-1a-64 digest, hex
//! layout_key INTEGER NOT NULL, -- ssd_tier::compute_layout_key u64
//! path TEXT NOT NULL, -- absolute path to the .kvb file
//! model_id TEXT NOT NULL, -- "<arch>/<snapshot>" identity
//! kv_quant TEXT NOT NULL, -- KvQuant Display string
//! byte_size INTEGER NOT NULL, -- on-disk byte size of the .kvb file
//! last_used INTEGER NOT NULL, -- unix epoch (seconds)
//! PRIMARY KEY (hash, layout_key)
//! )
//!
//! schema_version (
//! version INTEGER PRIMARY KEY NOT NULL -- 2
//! )
//!
//! INDEX kv_blocks_last_used ON kv_blocks (last_used) -- LRU eviction order
//! ```
//!
//! The `last_used` index is created idempotently on every open: it carries no
//! row-format change, so it needs no `schema_version` bump, and LRU eviction
//! runs after every spilled block and would otherwise sort the whole table to
//! find the oldest row.
//!
//! The `(hash, layout_key)` composite PK is defence-in-depth: the chained
//! digests stores under a given `layout_key` are already disjoint from
//! those of any other layout (the salt enters the FNV seed), so the PK only
//! ever guards against an upstream regression that forgets to re-seed. Two
//! distinct `layout_key`s for the same arch/project may share the same prompt
//! but never collide on the same row.
//!
//! # Pre-release schema upgrade
//!
//! rMLX is unreleased, so the v1 → v2 transition is a one-time wipe rather
//! than a row-by-row migration. The cross-namespace wipe pass is driven by
//! [`crate::ssd_tier::install_config`] before any [`SsdKvIndex::open`] runs
//! in-process; per-DB, [`SsdKvIndex::open`] enforces `schema_version = 2` and
//! returns [`SsdKvIndexError::SchemaMismatch`] on any other value.

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::too_many_lines
)]
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors raised by [`SsdKvIndex`].
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
/// Errors raised by [`SsdKvIndex`].
pub enum SsdKvIndexError {
    /// Underlying SQLite error.
    #[error("kv-index sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Underlying I/O error.
    #[error("kv-index io: {0}")]
    Io(#[from] std::io::Error),
    /// the on-disk DB advertises a schema version this binary does not
    /// understand. Pre-release v1 namespaces are wiped by
    /// [`crate::ssd_tier::install_config`] BEFORE `open` runs, so any
    /// mismatch surfacing here is a future schema we cannot interpret.
    #[error("kv-index schema mismatch: DB version {found}, expected {expected}")]
    SchemaMismatch {
        /// Schema version found in the DB.
        found: i64,
        /// Schema version this binary expects.
        expected: i64,
    },
}

type Result<T> = std::result::Result<T, SsdKvIndexError>;

// ── Schema + pragmas ──────────────────────────────────────────────────────────

/// Current `kv_blocks` schema version. Bumped from implicit v1 →
/// explicit v2 when the `layout_key` column + composite PK + `schema_version`
/// table were added.
pub(crate) const SCHEMA_VERSION: i64 = 2;

const SCHEMA_PRAGMAS: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
";

const SCHEMA_TABLES: &str = "
CREATE TABLE IF NOT EXISTS kv_blocks (
    hash        TEXT    NOT NULL,
    layout_key  INTEGER NOT NULL,
    path        TEXT    NOT NULL,
    model_id    TEXT    NOT NULL,
    kv_quant    TEXT    NOT NULL,
    byte_size   INTEGER NOT NULL,
    last_used   INTEGER NOT NULL,
    PRIMARY KEY (hash, layout_key)
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY NOT NULL
);
";

/// LRU eviction runs after every spilled block and reads rows in `last_used`
/// order. Without this index SQLite sorts the whole table before it can yield
/// the first row, so stopping the scan early saves the row decode but not the
/// sort — the dominant cost at a large ceiling.
///
/// Indexes carry no row-format change, so this is created idempotently on every
/// open rather than being tied to a `schema_version` bump (which would wipe
/// existing namespaces for no reason).
const SCHEMA_INDEXES: &str = "
CREATE INDEX IF NOT EXISTS kv_blocks_last_used ON kv_blocks (last_used);
";

// ── Row ───────────────────────────────────────────────────────────────────────

/// A single row from the `kv_blocks` table.
#[non_exhaustive]
#[derive(Debug, Clone)]
/// A single row from the `kv_blocks` table.
pub struct KvBlockRow {
    /// Chained FNV-1a-64 block hash (hex string).
    pub hash: String,
    /// Layout key salt used when writing this block.
    pub layout_key: u64,
    /// Absolute path to the `.kvb` safetensors file.
    pub path: PathBuf,
    /// `"<arch>/<snapshot>"` identity string.
    pub model_id: String,
    /// `KvQuant::Display` string (e.g. `"k8v4"`).
    pub kv_quant: String,
    /// On-disk byte size of the `.kvb` file.
    pub byte_size: u64,
    /// Unix epoch timestamp of last access.
    pub last_used: u64,
}

/// Outcome of one [`SsdKvIndex::evict_lru_until`] pass over a namespace.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct NamespaceEviction {
    /// Absolute paths of the `.kvb` files whose rows this call removed. A row
    /// another evictor took first is absent, so `paths.len()` is the number of
    /// rows genuinely evicted here and every path is one this caller owns.
    pub paths: Vec<PathBuf>,
    /// Indexed footprint in bytes once the deletes committed. Saves the caller
    /// a second `SUM(byte_size)` over the table.
    pub total_bytes_after: u64,
}

// ── Index ─────────────────────────────────────────────────────────────────────

/// SQLite index of on-disk KV blocks with LRU-by-size eviction.
///
/// One `SsdKvIndex` instance per namespace. The index DB is opened once and
/// kept open for the lifetime of the struct. The `namespace` is typically
/// `"<model_id>"` or the `--project P` override; it is only used to derive the
/// DB path via [`rmlx_core::paths::kv_cache_dir`]. The layout-key column lets
/// the SAME namespace hold disjoint per-layout rows.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed accessor — field is private; public API is the open/insert/lookup methods, not struct literal construction"
)]
pub struct SsdKvIndex {
    conn: Connection,
}

impl std::fmt::Debug for SsdKvIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsdKvIndex").finish_non_exhaustive()
    }
}

impl SsdKvIndex {
    /// Open (or create) the index for `namespace`.
    ///
    /// The DB is placed at `<kv_cache_dir(namespace)>/index.db`. Behaviour:
    ///
    /// 1. File absent → create v2 schema + insert `schema_version = 2`.
    /// 2. File present, **no `schema_version` table** → this is a pre-release
    ///    v1 DB. Return `SchemaMismatch`.
    /// 3. File present, `schema_version != 2` → `SchemaMismatch`.
    /// 4. File present, `schema_version == 2` → proceed.
    pub fn open(namespace: &str) -> Result<Self> {
        let dir = rmlx_core::paths::kv_cache_dir(namespace);
        let db_path = dir.join("index.db");
        tracing::debug!(
            namespace,
            db = %db_path.display(),
            "SsdKvIndex open"
        );
        Self::open_at(&db_path)
    }

    /// Open an in-memory index (for tests).
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        init_v2_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Open the index at an explicit path. Performs the same schema-version
    /// check as [`open`]: missing `schema_version` table on an existing DB ⇒
    /// [`SsdKvIndexError::SchemaMismatch`].
    pub fn open_at(db_path: &Path) -> Result<Self> {
        let file_existed = db_path.exists();
        let conn = Connection::open(db_path)?;
        conn.execute_batch(SCHEMA_PRAGMAS)?;

        if file_existed {
            // Inspect schema_version before creating any v2 tables — touching
            // them first would mask a pre-release v1 layout.
            let version = read_schema_version(&conn)?;
            match version {
                Some(v) if v == SCHEMA_VERSION => {
                    conn.execute_batch(SCHEMA_INDEXES)?;
                    Ok(Self { conn })
                }
                Some(v) => Err(SsdKvIndexError::SchemaMismatch {
                    found: v,
                    expected: SCHEMA_VERSION,
                }),
                None => {
                    // Pre-release v1 — the cross-namespace wipe in
                    // ssd_tier::install_config should have removed this DB
                    // before we got here.
                    Err(SsdKvIndexError::SchemaMismatch {
                        found: 1,
                        expected: SCHEMA_VERSION,
                    })
                }
            }
        } else {
            init_v2_schema(&conn)?;
            Ok(Self { conn })
        }
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    /// Look up a KV block by its `(hash, layout_key)` composite PK.
    ///
    /// Returns `None` if the row is not in the index.
    pub fn lookup(&self, hash: &str, layout_key: u64) -> Result<Option<KvBlockRow>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT hash, layout_key, path, model_id, kv_quant, byte_size, last_used
             FROM kv_blocks WHERE hash = ?1 AND layout_key = ?2",
        )?;
        let rows: Vec<KvBlockRow> = stmt
            .query_map(params![hash, layout_key as i64], row_from_row)?
            .collect::<std::result::Result<_, _>>()?;
        Ok(rows.into_iter().next())
    }

    /// Find the longest stored block-aligned prefix for a prompt at this
    /// layout (+ ).
    ///
    /// `chained` is `prompt_cache::chained_block_hashes_seeded(ids, FNV_OFFSET ^
    /// layout_key)` — one chained FNV-1a-64 digest per full 256-token block.
    /// The digest at index `k-1` uniquely identifies the entire `k`-block
    /// prefix under this `layout_key`, so it is exactly the spill key
    /// wrote for a `k`-block snapshot at the same layout. Walks longest-first
    /// and returns the first that hits the index under `(hash, layout_key)`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn lookup_longest_prefix(
        &self,
        chained: &[u64],
        layout_key: u64,
    ) -> Result<Option<(usize, KvBlockRow)>> {
        for k in (1..=chained.len()).rev() {
            let hash = hash_to_hex(chained[k - 1]);
            if let Some(row) = self.lookup(&hash, layout_key)? {
                return Ok(Some((k, row)));
            }
        }
        Ok(None)
    }

    /// Delete a single row by composite `(hash, layout_key)`. No-op if absent.
    pub fn delete(&self, hash: &str, layout_key: u64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM kv_blocks WHERE hash = ?1 AND layout_key = ?2",
            params![hash, layout_key as i64],
        )?;
        Ok(())
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Insert or replace a row for the given block at `(hash, layout_key)`.
    ///
    /// `hash` is the chained FNV-1a-64 digest formatted as a hex string.
    /// `layout_key` is the stable u64 over the arch + KV layout.
    /// `path` must be the absolute path to the `.kvb` file. `last_used` is
    /// set to the current unix epoch.
    pub fn record(
        &self,
        hash: &str,
        layout_key: u64,
        path: &Path,
        model_id: &str,
        kv_quant: &str,
        byte_size: u64,
    ) -> Result<()> {
        let now = unix_now();
        let path_str = path.to_string_lossy();
        self.conn.execute(
            "INSERT OR REPLACE INTO kv_blocks
             (hash, layout_key, path, model_id, kv_quant, byte_size, last_used)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                hash,
                layout_key as i64,
                path_str,
                model_id,
                kv_quant,
                byte_size as i64,
                now as i64
            ],
        )?;
        tracing::debug!(
            hash,
            layout_key,
            path = %path.display(),
            byte_size,
            "kv-index: recorded block"
        );
        Ok(())
    }

    /// Update `last_used` for an existing block (cache hit).
    ///
    /// No-ops silently if `(hash, layout_key)` is not found.
    pub fn touch(&self, hash: &str, layout_key: u64) -> Result<()> {
        let now = unix_now();
        let n = self.conn.execute(
            "UPDATE kv_blocks SET last_used = ?1 WHERE hash = ?2 AND layout_key = ?3",
            params![now as i64, hash, layout_key as i64],
        )?;
        if n > 0 {
            tracing::debug!(hash, layout_key, last_used = now, "kv-index: touched block");
        }
        Ok(())
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    /// Evict the oldest-used blocks until the total `byte_size` in the index is
    /// ≤ `budget_bytes`.
    ///
    /// Rows are deleted in ascending `last_used` order (oldest first), inside a
    /// single transaction, so a failure part-way leaves the index exactly as it
    /// was rather than dropping rows whose files the caller never gets told to
    /// unlink. `budget_bytes == 0` is taken literally here ("keep nothing") —
    /// the tier-level "no ceiling configured" reading lives in
    /// [`crate::ssd_tier::enforce_namespace_budget`].
    ///
    /// The returned [`NamespaceEviction`] carries the **absolute paths** of the
    /// `.kvb` files whose rows this call actually removed, so the caller can
    /// unlink exactly those, plus the resulting footprint.
    ///
    /// Runs after every spilled block, so it must not be O(rows): the scan stops
    /// as soon as the running total is within budget instead of materialising
    /// the whole table, and projects only the four columns eviction needs.
    pub fn evict_lru_until(&self, budget_bytes: u64) -> Result<NamespaceEviction> {
        let total = self.total_bytes()?;
        if total <= budget_bytes {
            return Ok(NamespaceEviction {
                paths: Vec::new(),
                total_bytes_after: total,
            });
        }

        // Oldest-first candidates, taking only as many as it takes to get under
        // the ceiling.
        let mut candidates: Vec<(String, u64, PathBuf, u64)> = Vec::new();
        {
            let mut stmt = self.conn.prepare_cached(
                "SELECT hash, layout_key, path, byte_size
                 FROM kv_blocks ORDER BY last_used ASC",
            )?;
            let mut rows = stmt.query([])?;
            let mut running = total;
            while running > budget_bytes {
                let Some(r) = rows.next()? else { break };
                let byte_size = r.get::<_, i64>(3)?.max(0) as u64;
                running = running.saturating_sub(byte_size);
                candidates.push((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)? as u64,
                    PathBuf::from(r.get::<_, String>(2)?),
                    byte_size,
                ));
            }
        }

        // One transaction: either every selected row goes, or none does, so the
        // returned paths always match what was actually removed.
        let tx = self.conn.unchecked_transaction()?;
        let mut paths: Vec<PathBuf> = Vec::with_capacity(candidates.len());
        let mut freed: u64 = 0;
        for (hash, layout_key, path, byte_size) in candidates {
            let n = tx.execute(
                "DELETE FROM kv_blocks WHERE hash = ?1 AND layout_key = ?2",
                params![hash, layout_key as i64],
            )?;
            // A no-op DELETE means another evictor already took this row; it
            // owns the file, so this call neither counts it nor unlinks it.
            if n > 0 {
                tracing::debug!(
                    hash = %hash,
                    layout_key,
                    path = %path.display(),
                    byte_size,
                    "kv-index: evicting block (LRU)"
                );
                freed = freed.saturating_add(byte_size);
                paths.push(path);
            }
        }
        tx.commit()?;

        Ok(NamespaceEviction {
            paths,
            total_bytes_after: total.saturating_sub(freed),
        })
    }

    /// Total `byte_size` summed over all indexed blocks.
    pub fn total_bytes(&self) -> Result<u64> {
        let total: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(byte_size), 0) FROM kv_blocks",
            [],
            |r| r.get(0),
        )?;
        Ok(total.max(0) as u64)
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Drop rows whose `.kvb` file no longer exists on disk.
    ///
    /// Returns the number of rows pruned.
    pub fn prune_missing(&self) -> Result<usize> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT hash, layout_key, path FROM kv_blocks")?;
        let triples: Vec<(String, i64, String)> = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<_, _>>()?;

        let mut pruned = 0usize;
        for (hash, layout_key, path_str) in triples {
            if !Path::new(&path_str).exists() {
                self.conn.execute(
                    "DELETE FROM kv_blocks WHERE hash = ?1 AND layout_key = ?2",
                    params![hash, layout_key],
                )?;
                tracing::debug!(
                    hash,
                    layout_key = layout_key as u64,
                    path = path_str,
                    "kv-index: pruned missing block"
                );
                pruned += 1;
            }
        }

        if pruned > 0 {
            tracing::info!(pruned, "kv-index: pruned missing blocks");
        }
        Ok(pruned)
    }
}

// ── cross-namespace pool eviction ─────────────────────────────────────

/// Summary report from a cross-namespace LRU sweep.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct EvictionReport {
    /// Total `.kvb` byte_size of all evicted rows (sum of `byte_size`).
    pub bytes_freed: u64,
    /// Number of `kv_blocks` rows evicted across all namespaces.
    pub blocks_evicted: u64,
    /// Number of distinct namespaces that contributed at least one eviction.
    pub namespaces_touched: u64,
}

/// cross-namespace LRU eviction down to `global_budget_bytes`.
///
/// Walks every `<kv_root>/<ns>/index.db` (skipping non-namespace dirs, missing
/// or non-SQLite files), builds a merged oldest-first ordering of every
/// `(last_used, namespace, hash, layout_key, byte_size)` row, and deletes
/// rows oldest-first across the union until the total persisted footprint is
/// ≤ `global_budget_bytes`. Each eviction removes the `.kvb` file from disk
/// and the index row from the owning namespace's DB.
///
/// `global_budget_bytes == 0` is a no-op (matches the OFF convention used by
/// the per-namespace evict path). Best-effort: I/O errors on a single
/// namespace are `warn!`ed and the walk continues.
///
/// Emits exactly one `tracing::info!` summarising `bytes_freed`,
/// `blocks_evicted`, `namespaces_touched`, and the final pool size.
pub fn evict_pool_lru_until(kv_root: &Path, global_budget_bytes: u64) -> Result<EvictionReport> {
    if global_budget_bytes == 0 {
        return Ok(EvictionReport::default());
    }
    if !kv_root.exists() {
        return Ok(EvictionReport::default());
    }

    // ── 1. Discover namespaces + load every row (oldest-first across union) ──
    #[derive(Debug)]
    struct PoolRow {
        last_used: u64,
        namespace: String,
        hash: String,
        layout_key: u64,
        byte_size: u64,
        path: PathBuf,
    }

    let mut all_rows: Vec<PoolRow> = Vec::new();
    let mut pool_bytes: u64 = 0;
    // Hold open one SsdKvIndex per namespace for the whole sweep — single
    // `open_at` per namespace (avoids the schema-version round-trip twice).
    let mut ns_indices: std::collections::HashMap<String, SsdKvIndex> =
        std::collections::HashMap::new();

    let entries = match std::fs::read_dir(kv_root) {
        Ok(it) => it,
        Err(e) => {
            tracing::warn!(
                kv_root = %kv_root.display(),
                error = %e,
                "evict_pool_lru_until: kv_root scan failed; skipping"
            );
            return Ok(EvictionReport::default());
        }
    };
    for entry in entries.flatten() {
        let ns_path = entry.path();
        if !ns_path.is_dir() {
            continue;
        }
        let ns_name = match ns_path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let db_path = ns_path.join("index.db");
        if !db_path.exists() {
            continue;
        }
        // Best-effort per-namespace open; `SchemaMismatch` / I/O → warn + skip.
        let idx = match SsdKvIndex::open_at(&db_path) {
            Ok(idx) => idx,
            Err(e) => {
                tracing::warn!(
                    namespace = %ns_name,
                    db = %db_path.display(),
                    error = %e,
                    "evict_pool_lru_until: open failed; skipping namespace"
                );
                continue;
            }
        };

        {
            let mut stmt = idx
                .conn
                .prepare("SELECT hash, layout_key, path, byte_size, last_used FROM kv_blocks")?;
            let iter = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            })?;
            for row in iter {
                let (hash, lk, path_str, byte_size, last_used) = row?;
                let bs = byte_size.max(0) as u64;
                pool_bytes = pool_bytes.saturating_add(bs);
                all_rows.push(PoolRow {
                    last_used: last_used.max(0) as u64,
                    namespace: ns_name.clone(),
                    hash,
                    layout_key: lk as u64,
                    byte_size: bs,
                    path: PathBuf::from(path_str),
                });
            }
        }

        ns_indices.insert(ns_name, idx);
    }

    tracing::debug!(
        event = "ssd_pool_lru_discovery",
        rows = all_rows.len(),
        namespaces = ns_indices.len(),
        pool_bytes,
        global_budget_bytes,
        "ssd pool LRU discovery scan complete"
    );

    if pool_bytes <= global_budget_bytes {
        tracing::debug!(
            pool_bytes,
            global_budget_bytes,
            "evict_pool_lru_until: within budget, no-op"
        );
        return Ok(EvictionReport::default());
    }

    // ── 2. Sort union oldest-first; pick eviction prefix ────────────────────
    all_rows.sort_by(|a, b| {
        (a.last_used, &a.namespace, &a.hash).cmp(&(b.last_used, &b.namespace, &b.hash))
    });

    let mut running = pool_bytes;
    let mut to_evict: Vec<PoolRow> = Vec::new();
    for row in all_rows {
        if running <= global_budget_bytes {
            break;
        }
        running = running.saturating_sub(row.byte_size);
        to_evict.push(row);
    }

    // ── 3. Per-namespace evict: delete .kvb file + index row ────────────────
    let mut bytes_freed: u64 = 0;
    let mut blocks_evicted: u64 = 0;
    let mut ns_touched: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Group evictions by namespace so each DB is opened once.
    let mut by_ns: std::collections::HashMap<String, Vec<PoolRow>> =
        std::collections::HashMap::new();
    for row in to_evict {
        by_ns.entry(row.namespace.clone()).or_default().push(row);
    }
    for (ns, rows) in by_ns {
        let idx = if let Some(idx) = ns_indices.get(&ns) {
            idx
        } else {
            tracing::warn!(
                namespace = %ns,
                "evict_pool_lru_until: no open index handle; skipping namespace"
            );
            continue;
        };
        for row in rows {
            // Delete .kvb on disk (best-effort).
            if let Err(e) = std::fs::remove_file(&row.path) {
                tracing::warn!(
                    path = %row.path.display(),
                    error = %e,
                    "evict_pool_lru_until: .kvb remove failed (continuing)"
                );
            }
            // Delete the index row.
            if let Err(e) = idx.delete(&row.hash, row.layout_key) {
                tracing::warn!(
                    namespace = %ns,
                    hash = %row.hash,
                    layout_key = row.layout_key,
                    error = %e,
                    "evict_pool_lru_until: index delete failed (continuing)"
                );
                continue;
            }
            bytes_freed = bytes_freed.saturating_add(row.byte_size);
            blocks_evicted += 1;
            ns_touched.insert(ns.clone());
        }
    }

    let final_pool_bytes = pool_bytes.saturating_sub(bytes_freed);
    tracing::info!(
        event = "ssd_pool_lru_eviction",
        bytes_freed,
        blocks_evicted,
        namespaces_touched = ns_touched.len() as u64,
        pool_bytes_before = pool_bytes,
        pool_bytes_after = final_pool_bytes,
        global_budget_bytes,
        "cross-namespace SSD pool LRU eviction complete"
    );
    Ok(EvictionReport {
        bytes_freed,
        blocks_evicted,
        namespaces_touched: ns_touched.len() as u64,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Initialise a fresh v2 schema on a connection: create the tables + insert
/// the version row. Idempotent on a clean DB; safe to call on an
/// already-initialised v2 DB (INSERT OR IGNORE on version row).
fn init_v2_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(SCHEMA_PRAGMAS)?;
    conn.execute_batch(SCHEMA_TABLES)?;
    conn.execute_batch(SCHEMA_INDEXES)?;
    conn.execute(
        "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
        params![SCHEMA_VERSION],
    )?;
    Ok(())
}

/// Read the schema version from an existing connection.
///
/// `Ok(Some(v))` when the `schema_version` table exists and holds a row;
/// `Ok(None)` when the table itself is absent (pre-release v1);
/// `Err(_)` only on a genuine SQLite error.
fn read_schema_version(conn: &Connection) -> Result<Option<i64>> {
    // Detect the table without creating it (DROP-and-recreate would mask v1).
    let table_present: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'",
            [],
            |_| Ok(true),
        )
        .unwrap_or(false);
    if !table_present {
        return Ok(None);
    }
    let version: i64 = match conn.query_row(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
        [],
        |r| r.get(0),
    ) {
        Ok(v) => v,
        Err(rusqlite::Error::QueryReturnedNoRows) => 0,
        Err(e) => return Err(e.into()),
    };
    Ok(Some(version))
}

/// Format a chained FNV-1a-64 hash digest as a hex string for use as the
/// index `hash` column.
pub fn hash_to_hex(digest: u64) -> String {
    format!("{digest:016x}")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn row_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<KvBlockRow> {
    // Column order matches the SELECT statements above: hash, layout_key, path,
    // model_id, kv_quant, byte_size, last_used.
    let path_str: String = r.get(2)?;
    Ok(KvBlockRow {
        hash: r.get(0)?,
        layout_key: r.get::<_, i64>(1)? as u64,
        path: PathBuf::from(path_str),
        model_id: r.get(3)?,
        kv_quant: r.get(4)?,
        byte_size: r.get::<_, i64>(5)? as u64,
        last_used: r.get::<_, i64>(6)? as u64,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "ssd_index_tests.rs"]
mod ssd_index_tests;
