//! SSD prompt-cache tier wiring (; layout-key disambiguation).
//!
//! ..47 built the SSD prompt-cache tier piece by piece:
//!
//! - `block_io` — serialize / deserialize a post-prefill snapshot to a
//!   `.kvb` safetensors file.
//! - `SsdKvIndex` — SQLite index of those blocks with LRU-by-size
//!   eviction, under `paths::kv_cache_dir(<namespace>)`.
//! - `SsdSpiller` — a background drain thread that persists RAM-evicted
//!   prompt-cache entries to the tier (`PromptCache::set_spill_sink`).
//! - `SsdHydrator` — reads the longest cached block-aligned prefix back
//!   on a RAM miss (`PromptCache::set_ssd_source`).
//!
//! This module is the production switch that turns the tier ON. It holds a
//! process-global config set once at serve startup ([`install_config`]) from
//! the CLI flags `--kv-ssd-cache-gb` and `--project`, and an attach entry point
//! ([`attach_at_load`]) the model-load path calls per loaded model.
//!
//! ## : layout-key disambiguation
//!
//! Each loaded model derives a stable `layout_key` over
//! `(arch, n_layers, n_kv_heads, head_dim, kv_quant)`. The key is propagated
//! into the per-arch prompt cache, the spiller (`SpillJob::layout_key` →
//! `kv_blocks.layout_key`), and the hydrator (salts the chained-hash
//! digest stream and pins the composite `(hash, layout_key)` PK lookup). Two
//! snapshots of the SAME arch with identical weights at the SAME kv_quant
//! still collide on `layout_key` and therefore share their cache, which is the
//! intended behaviour — `layout_key` is weight-independent on purpose.
//!
//! ## Pre-release schema wipe
//!
//! rMLX is unreleased, so the index schema bumped from implicit v1 → explicit
//! v2 by wiping rather than migrating. [`install_config`] walks every
//! `<RMLX_HOME>/cache/kv/<ns>/` namespace BEFORE any [`SsdKvIndex::open`]
//! runs: if `<ns>/index.db` exists and lacks the `schema_version` table, the
//! ENTIRE namespace dir is removed (`fs::remove_dir_all`) and dropped bytes
//! are logged with an `ssd_cache_pre_release_wipe` event. The pass is
//! idempotent — a second boot with the v2 schema is a no-op (dropped_bytes=0).

use std::borrow::Cow;
use std::path::Path;
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::Device;

use rmlx_kv_quant::KvQuant;

use crate::hashing::{FNV_OFFSET, FNV_PRIME};
use crate::hooks::{call_ssd_bytes_used_hook, call_ssd_evict_total_hook};
use crate::ssd_index::SsdKvIndex;

/// Process-global SSD-tier config, set once at serve startup.
///
/// `None` (never configured) or a `per_namespace_budget_bytes == 0` (with
/// `global_budget_bytes == 0`) means the tier is OFF.
//
// owned by ; consumers read-only
//
// Field order + types are FROZEN here. (per-project budgets via
// `projects.toml`) will populate `per_project_budgets` and read the other
// fields without mutating them. Do not reorder, rename, or change types
// without bumping the owning ticket.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — four fields are the complete SSD-tier config contract; cross-crate construction in rmlx-cli::serve must list all fields; adding a field requires updating both install_config callers"
)]
#[derive(Debug, Clone)]
pub struct SsdTierConfig {
    /// Per-namespace byte budget (the `--kv-ssd-cache-gb` value in bytes).
    /// name was `budget_bytes` pre-; renamed for symmetry with
    /// `global_budget_bytes`. Kept public for tests + .
    pub per_namespace_budget_bytes: u64,
    /// SSD global pool ceiling across ALL namespaces under
    /// `<RMLX_HOME>/cache/kv/*`. `0` means no global cap (per-namespace only).
    pub global_budget_bytes: u64,
    /// Default namespace name (the `--project` override, or `None` →
    /// `model_id` fallback at `attach_at_load`).
    pub default_namespace: Option<String>,
    /// hook: per-project budget overrides (`project_name -> budget_bytes`).
    /// Always empty in ; populates from `projects.toml`.
    pub per_project_budgets: std::collections::BTreeMap<String, u64>,
}

static CONFIG: OnceLock<Option<SsdTierConfig>> = OnceLock::new();

/// Install the process-global SSD-tier config + run the pre-release
/// schema wipe pass (call once at serve startup, before any model loads).
///
/// Per-namespace budget `0` AND global budget `0` install the OFF state
/// (tier disabled). Idempotent in the sense of `OnceLock`: the first call
/// wins; later calls are ignored (with a `warn!` if they disagree). In debug
/// builds, a second call PANICS to surface the bug. The wipe runs
/// unconditionally on the first call, including when the tier is OFF — that
/// way switching `--kv-ssd-cache-gb` 0 → N between restarts cannot resurrect
/// a v1 namespace.
///
/// callers must pass an `SsdTierConfig` directly. `per_project_budgets`
/// MUST be empty in — populated by from `projects.toml`. After
/// the config is committed, runs [`startup_maintenance_pool`] to evict the
/// cross-namespace pool down to `global_budget_bytes`.
/// Install the process-global SSD-tier config (call once at serve startup).
///
/// Returns `Err(Error::SsdTierAlreadyInstalled)` if called more than once.
/// The first call always wins — the `OnceLock` is set on the first call and
/// subsequent calls return the typed error so the caller can decide whether to
/// propagate or log+ignore.
#[allow(
    clippy::cognitive_complexity,
    reason = "startup wiring: OnceLock set, conditional logging, schema wipe, and \
              cross-namespace LRU eviction are all sequentially necessary; splitting \
              would scatter the single-call semantics across helper fns with no benefit"
)]
#[allow(
    clippy::semicolon_if_nothing_returned,
    reason = "tracing macros expand to expressions in macro position; adding ';' \
              would require rewriting as statement blocks — the existing form is clearer"
)]
pub fn install_config(cfg: SsdTierConfig) -> Result<()> {
    let tier_on = cfg.per_namespace_budget_bytes > 0 || cfg.global_budget_bytes > 0;
    let stored = if tier_on { Some(cfg) } else { None };

    if CONFIG.set(stored).is_err() {
        tracing::error!(
            target: "rmlx::ssd_tier",
            "install_config called more than once — refusing to re-install; \
             keeping the first config"
        );
        return Err(Error::SsdTierAlreadyInstalled);
    }
    if let Some(c) = CONFIG.get().and_then(|v| v.as_ref()) {
        tracing::info!(
            per_namespace_budget_bytes = c.per_namespace_budget_bytes,
            per_namespace_budget_gib =
                (c.per_namespace_budget_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
            global_budget_bytes = c.global_budget_bytes,
            global_budget_gib = (c.global_budget_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
            default_namespace = c.default_namespace.as_deref().unwrap_or("(model_id)"),
            "SSD prompt-cache tier ENABLED"
        )
    } else {
        tracing::debug!("SSD prompt-cache tier OFF")
    }

    // single startup pass; runs BEFORE any namespace SsdKvIndex::open.
    wipe_pre_release_v1_namespaces(&rmlx_core::paths::cache_dir().join("kv"));

    // cross-namespace LRU enforcement. No-op when global_budget == 0.
    if let Some(c) = CONFIG.get().and_then(|v| v.as_ref()) {
        if c.global_budget_bytes > 0 {
            let kv_root = rmlx_core::paths::cache_dir().join("kv");
            match crate::ssd_index::evict_pool_lru_until(&kv_root, c.global_budget_bytes) {
                Ok(_) => {}
                Err(e) => tracing::warn!(
                    error = %e,
                    "cross-namespace LRU eviction failed at startup"
                ),
            }
        }
    }
    Ok(())
}

/// The active SSD-tier config, or `None` if the tier is OFF / unconfigured.
pub fn active() -> Option<SsdTierConfig> {
    CONFIG.get().cloned().flatten()
}

/// The byte ceiling one namespace may occupy under `cfg`.
///
/// The per-namespace budget is implicitly capped by the global pool ceiling
/// (`min(per_ns, global)`); when `global_budget_bytes == 0` the per-namespace
/// budget stands alone. Both the attach-time maintenance pass and the spill
/// drain thread resolve their ceiling through here so they cannot disagree
/// about what the configured budget means.
pub(crate) fn effective_namespace_budget(cfg: &SsdTierConfig) -> u64 {
    if cfg.global_budget_bytes > 0 {
        cfg.per_namespace_budget_bytes.min(cfg.global_budget_bytes)
    } else {
        cfg.per_namespace_budget_bytes
    }
}

/// Resolve the on-disk namespace for `model_id` under the active config.
///
/// Returns a borrowed slice of `cfg.default_namespace` when present, or an
/// owned copy of `model_id` as a fallback — avoiding the clone on the common
/// (non-overridden) path.
fn namespace_for<'a>(cfg: &'a SsdTierConfig, model_id: &'a str) -> Cow<'a, str> {
    match &cfg.default_namespace {
        Some(ns) => Cow::Borrowed(ns.as_str()),
        None => Cow::Owned(model_id.to_string()),
    }
}

/// u64 FNV-1a over the concatenated bytes of
/// `arch.as_str() + format!(":{n_layers}:{n_kv_heads}:{head_dim}:{kv_quant}")`.
///
/// The exact formula is committed to in the documented form: every distinct
/// `(arch, n_layers, n_kv_heads, head_dim, kv_quant)` tuple yields a distinct
/// u64 with overwhelming probability, and the function is deterministic across
/// runs (FNV with the standard offset basis + prime). `kv_quant.to_string()`
/// is taken from the existing stable Display impl on [`KvQuant`].
pub fn compute_layout_key(
    arch: &str,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_quant: KvQuant,
) -> u64 {
    let suffix = format!(":{n_layers}:{n_kv_heads}:{head_dim}:{kv_quant}");
    let mut h: u64 = FNV_OFFSET;
    for byte in arch.as_bytes().iter().chain(suffix.as_bytes().iter()) {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Resolved attach context: per-namespace setup is done; the arch crate now
/// needs to wire the spiller + hydrator onto its per-arch prompt cache.
///
/// Returned by [`prepare_attach`]. The arch-specific dispatch
/// (Gemma4 / Qwen3 / Qwen3.5-MoE `attach_ssd_tier`) lives in
/// `rmlx_models::ssd_tier::attach_at_load`, which crosses the crate boundary
/// by reaching back into `rmlx-models`'s per-arch modules. This crate does
/// not — and must not — depend on `rmlx-models`.
#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "internal closed bridge struct")]
pub struct AttachInfo {
    /// Resolved namespace under `<RMLX_HOME>/cache/kv/<namespace>/`.
    pub namespace: String,
    /// Stable u64 over `(arch, n_layers, n_kv_heads, head_dim, kv_quant)`.
    pub layout_key: u64,
    /// KV quant in effect for this snapshot.
    pub kv_quant: KvQuant,
    /// The device handed in by the caller; threaded back unchanged.
    pub device: Device,
}

/// Run startup maintenance for `model_id` under the active SSD-tier config and
/// return the [`AttachInfo`] the per-arch `attach_ssd_tier` call needs.
///
/// `(n_layers, n_kv_heads, head_dim)` are taken from the loaded model's
/// config and folded into the `layout_key`.
///
/// Returns:
/// - `None` when the SSD tier is OFF (no config installed, or both budgets
///   zero) or when `kv_quant` is `None` (unresolved at the call site — logged
///   as a `warn!`).
/// - `Some(info)` otherwise; the caller then dispatches to the per-arch
///   `attach_ssd_tier` with the contained `(namespace, kv_quant, layout_key,
///   device)`.
///
/// The wrapper retains all the side effects the previous `attach_at_load`
/// performed pre-dispatch: layout-key logging, prune + evict-to-budget pass,
/// Prometheus byte / evict counter updates via the registered hooks.
pub fn prepare_attach(
    arch: &str,
    model_id: &str,
    kv_quant: Option<KvQuant>,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
) -> Option<AttachInfo> {
    let cfg = active()?; // tier OFF — true no-op, spill/hydrate hooks never installed
    let Some(kv_quant) = kv_quant else {
        tracing::warn!(
            arch,
            model_id,
            "SSD tier requested but kv_quant unresolved; skipping attach"
        );
        return None;
    };
    let namespace = namespace_for(&cfg, model_id);

    // derive + log layout key BEFORE opening the index so the operator
    // can correlate per-namespace rows with the active layout in the same run.
    let layout_key = compute_layout_key(arch, n_layers, n_kv_heads, head_dim, kv_quant);
    tracing::info!(
        event = "ssd_layout_key_resolved",
        arch,
        n_layers,
        n_kv_heads,
        head_dim,
        kv_quant = %kv_quant,
        layout_key = format!("{:016x}", layout_key),
        "layout_key resolved for SSD tier"
    );

    // Startup maintenance: drop rows whose .kvb vanished, then evict
    // oldest-first until the persisted footprint is within budget.
    let effective_per_ns = effective_namespace_budget(&cfg);
    // Open the namespace index here — this is the single `paths::home()`
    // resolution on the attach path — then hand the opened handle to the
    // maintenance routine. Keeping the open at the boundary means the
    // maintenance logic itself takes its root explicitly (the index already
    // carries it) and never re-resolves the process-global home, which is
    // what makes it testable against an injected temp dir.
    match SsdKvIndex::open(namespace.as_ref()) {
        Ok(index) => startup_maintenance(&index, namespace.as_ref(), effective_per_ns),
        Err(e) => tracing::warn!(
            namespace = %namespace,
            error = %e,
            "ssd-tier startup index open failed; skipping maintenance"
        ),
    }

    Some(AttachInfo {
        namespace: namespace.into_owned(),
        layout_key,
        kv_quant,
        device,
    })
}

/// Prune missing blocks from an already-opened namespace index, then evict LRU
/// until the persisted footprint is within `budget_bytes`. Best-effort: any
/// error is `warn!`ed and the remaining steps proceed (the spiller opens its
/// own index on its drain thread regardless).
///
/// Takes the opened `index` explicitly rather than resolving it from the
/// process-global home — the caller owns the one `paths::home()` resolution on
/// the attach path. This keeps the routine root-injectable (tests open against
/// a temp dir) and free of the `OnceLock` ordering hazard.
#[allow(
    clippy::semicolon_if_nothing_returned,
    reason = "tracing macros expand to expressions in macro position; the trailing \
              tracing::info! / tracing::warn! in match arms are the last expression, \
              and the surrounding match itself is a statement"
)]
fn startup_maintenance(index: &SsdKvIndex, namespace: &str, budget_bytes: u64) {
    match index.prune_missing() {
        Ok(n) if n > 0 => {
            tracing::info!(
                namespace,
                pruned = n,
                "ssd-tier pruned missing blocks at startup"
            )
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(namespace, error = %e, "ssd-tier prune_missing failed"),
    }
    let before = index.total_bytes().unwrap_or(0);
    let evicted = enforce_namespace_budget(index, namespace, budget_bytes);
    let after = index.total_bytes().unwrap_or(0);
    tracing::info!(
        namespace,
        budget_bytes,
        bytes_before = before,
        bytes_after = after,
        evicted,
        "ssd-tier startup evict-to-budget complete"
    );
}

/// Evict oldest-first until the namespace footprint is within `budget_bytes`,
/// delete each evicted `.kvb`, and republish the two Prometheus hooks. Returns
/// the number of blocks evicted.
///
/// This is the single eviction routine for the tier: the attach-time
/// maintenance pass runs it after `prune_missing`, and the spill drain thread
/// runs it after each block it writes, so the configured budget holds for the
/// whole life of the process instead of only at the moment a model is loaded.
///
/// Best-effort throughout: an index error is `warn!`ed and treated as "evicted
/// nothing"; a file that is already gone is not an error (a second evictor, or
/// an operator's `rm`, reaching the same block first leaves exactly the state
/// this pass wanted).
///
/// **Safe to run against a namespace with in-flight hydrate + spill traffic.**
/// `evict_lru_until` deletes the index row before this fn deletes the file, so
/// a concurrent lookup either does not find the row (a plain miss) or reads a
/// file that has since vanished — which the hydrator already handles as a
/// corrupt block: drop the row, `warn!`, and fall through to a full prefill. No
/// reader can be handed a block other than the one it asked for, because a
/// block is only ever reachable through its own `(hash, layout_key)` row.
pub(crate) fn enforce_namespace_budget(
    index: &SsdKvIndex,
    namespace: &str,
    budget_bytes: u64,
) -> u64 {
    let evicted = match index.evict_lru_until(budget_bytes) {
        Ok(paths) => paths,
        Err(e) => {
            tracing::warn!(namespace, error = %e, "ssd-tier evict_lru_until failed");
            Vec::new()
        }
    };
    for p in &evicted {
        match std::fs::remove_file(p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!(path = %p.display(), error = %e, "ssd-tier evicted-block file remove failed");
            }
        }
    }

    // Publish the eviction count to the Prometheus counter hook.
    // Zero when already within budget — no counter bump needed.
    let evict_count = evicted.len() as u64;
    if evict_count > 0 {
        call_ssd_evict_total_hook(namespace, evict_count);
    }

    // Publish the on-disk footprint to the Prometheus gauge. Running on every
    // spill is what keeps `rmlx_ssd_bytes_used` tracking the tier instead of
    // freezing at the value measured when the model was loaded.
    call_ssd_bytes_used_hook(namespace, index.total_bytes().unwrap_or(0));
    evict_count
}

/// Enforce `budget_bytes` on `namespace` from the spill drain thread.
///
/// Thin gate over [`enforce_namespace_budget`]: a zero budget means "no
/// ceiling configured for this spiller" and must not be handed to
/// `evict_lru_until`, which would read it as "keep nothing" and empty the
/// namespace.
pub(crate) fn enforce_budget_after_spill(index: &SsdKvIndex, namespace: &str, budget_bytes: u64) {
    if budget_bytes == 0 {
        return;
    }
    enforce_namespace_budget(index, namespace, budget_bytes);
}

/// pre-release schema wipe: walk every namespace under `kv_root` and
/// remove any whose `index.db` lacks the v2 `schema_version` table.
///
/// Idempotent — on a clean v2 boot every namespace is skipped, dropped_bytes
/// is 0, and no `fs::remove_dir_all` runs. Emits a single
/// `ssd_cache_pre_release_wipe` event listing the removed namespaces.
///
/// Best-effort: I/O errors on individual entries are `warn!`ed and the walk
/// continues. Public only for the `cfg(test)` harness that exercises the wipe
/// directly without driving [`install_config`].
#[allow(
    clippy::cognitive_complexity,
    reason = "directory walk with per-entry SQLite probe, conditional remove, \
              byte accumulation, and structured tracing — sequential with no \
              natural split that would not fragment the schema-wipe logic"
)]
fn wipe_pre_release_v1_namespaces(kv_root: &Path) {
    if !kv_root.exists() {
        return;
    }
    let entries = match std::fs::read_dir(kv_root) {
        Ok(it) => it,
        Err(e) => {
            tracing::warn!(
                kv_root = %kv_root.display(),
                error = %e,
                "ssd-cache wipe scan failed; skipping"
            );
            return;
        }
    };

    let mut wiped_namespaces: Vec<String> = Vec::new();
    let mut total_dropped_bytes: u64 = 0;

    for entry in entries.flatten() {
        let ns_path = entry.path();
        if !ns_path.is_dir() {
            continue;
        }
        let db_path = ns_path.join("index.db");
        if !db_path.exists() {
            continue; // empty namespace dir — nothing to migrate
        }
        if !is_pre_release_v1(&db_path) {
            continue;
        }

        let dropped = dir_size_bytes(&ns_path);
        let ns_name = ns_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        match std::fs::remove_dir_all(&ns_path) {
            Ok(()) => {
                wiped_namespaces.push(ns_name);
                total_dropped_bytes += dropped;
            }
            Err(e) => tracing::warn!(
                namespace = %ns_path.display(),
                error = %e,
                "ssd-cache wipe failed for namespace"
            ),
        }
    }

    if wiped_namespaces.is_empty() {
        tracing::debug!(
            event = "ssd_cache_pre_release_wipe",
            namespaces = ?Vec::<String>::new(),
            dropped_bytes = 0u64,
            "ssd-cache wipe pass — nothing to drop"
        );
    } else {
        tracing::info!(
            event = "ssd_cache_pre_release_wipe",
            namespaces = ?wiped_namespaces,
            dropped_bytes = total_dropped_bytes,
            reason = "pre-release schema upgrade",
            "dropped pre-release v1 namespaces"
        );
    }
}

/// True iff `index.db` is a SQLite file holding the v1 `kv_blocks` table but
/// lacking the v2 `schema_version` table. Any error opening / inspecting the
/// file is treated as "not v1" (leave it alone) — the in-place `SsdKvIndex::open`
/// will surface a `SchemaMismatch` on a future schema and the operator can
/// investigate, rather than this helper silently wiping unrelated DBs.
#[allow(
    clippy::manual_let_else,
    reason = "early-return on Err(_) is cleaner here: the connection is consumed \
              in a separate query_row call, so the let-else form would need to \
              restructure non-trivially around the borrow"
)]
fn is_pre_release_v1(db_path: &Path) -> bool {
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let has_schema_version: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'",
            [],
            |_| Ok(true),
        )
        .unwrap_or(false);
    if has_schema_version {
        return false;
    }
    let has_kv_blocks: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kv_blocks'",
            [],
            |_| Ok(true),
        )
        .unwrap_or(false);
    has_kv_blocks
}

/// Sum the size of every file inside `dir` (non-recursive enough — namespaces
/// hold a flat list of `.kvb` files + the `index.db`). Best-effort; returns 0
/// on any error.
fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in rd.flatten() {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

#[cfg(test)]
#[path = "ssd_tier_tests.rs"]
mod tests;
