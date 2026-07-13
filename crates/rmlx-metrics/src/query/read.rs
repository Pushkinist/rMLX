//! Read-only query functions against the metrics SQLite database.
//!
//! All functions take a `&Connection` and return plain data structs. No writes
//! are performed here — mutating ops live in [`crate::recorder`].

use std::fmt::Write as _;

use rusqlite::{params_from_iter, Connection, OptionalExtension};

use crate::error::{Error, Result};

use super::types::{
    BestRow, Bucket, Cell, ChampionCell, ChampionRow, CompareRow, DeltaRow, ObservationRow,
    RegressResult, TimeseriesPoint,
};

// ── Query functions ───────────────────────────────────────────────────────────

/// `best(cell, metric)` — single champion row, or None if no observations match.
pub fn best(conn: &Connection, cell: &Cell, metric: &str) -> Result<Option<BestRow>> {
    let mut stmt = conn.prepare(
        "SELECT
             id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction,
             run_id, ts_utc, git_sha, backend_version, hardware_tag,
             description, notes, inserted_by
         FROM bests
         WHERE backend        = ?1
           AND model_namespace = ?2
           AND model           = ?3
           AND weight_quant    = ?4
           AND kv_quant        = ?5
           AND ctx_max         = ?6
           AND prompt_id       = ?7
           AND metric          = ?8",
    )?;

    let row = stmt
        .query_row(
            rusqlite::params![
                cell.backend,
                cell.model_namespace,
                cell.model,
                cell.weight_quant,
                cell.kv_quant,
                cell.ctx_max,
                cell.prompt_id,
                metric,
            ],
            row_to_best,
        )
        .optional()?;

    Ok(row)
}

/// `rank(metric, backend_filter, limit)` — top-N champions for one metric.
///
/// Ordering: `higher_better` metrics descend by value; `lower_better` ascend.
/// The CASE expression normalises both into an ascending order so a single
/// `ORDER BY … ASC` works for both directions.
pub fn rank(
    conn: &Connection,
    metric: &str,
    backend_filter: Option<&str>,
    limit: usize,
) -> Result<Vec<BestRow>> {
    let mut stmt = conn.prepare(
        "SELECT
             id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction,
             run_id, ts_utc, git_sha, backend_version, hardware_tag,
             description, notes, inserted_by
         FROM bests
         WHERE metric = ?1
           AND (?2 IS NULL OR backend = ?2)
         ORDER BY
             CASE direction
                 WHEN 'higher_better' THEN -value
                 ELSE value
             END ASC
         LIMIT ?3",
    )?;

    let rows = stmt
        .query_map(
            rusqlite::params![metric, backend_filter, limit as i64],
            row_to_best,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(rows)
}

/// `compare(backends, metric)` — per-cell champion for each listed backend.
///
/// Returns rows keyed by (model_namespace, model, weight_quant, kv_quant,
/// ctx_max, prompt_id). Each row carries a `per_backend` vec whose entries
/// are ordered to match the `backends` slice; missing backends get `None`.
pub fn compare(conn: &Connection, backends: &[&str], metric: &str) -> Result<Vec<CompareRow>> {
    // Group key: (model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id).
    type CellKey = (String, String, String, String, i64, i64);

    if backends.is_empty() {
        return Ok(vec![]);
    }

    // Build: WHERE metric = ?1 AND backend IN (?, ?, …)
    let placeholders: String = (2..=backends.len() + 1)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        "SELECT
             id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction,
             run_id, ts_utc, git_sha, backend_version, hardware_tag,
             description, notes, inserted_by
         FROM bests
         WHERE metric = ?1
           AND backend IN ({placeholders})
         ORDER BY model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id, backend"
    );

    let mut stmt = conn.prepare(&sql)?;

    let mut param_values: Vec<String> = Vec::with_capacity(backends.len() + 1);
    param_values.push(metric.to_owned());
    for b in backends {
        param_values.push((*b).to_owned());
    }

    let bests: Vec<BestRow> = stmt
        .query_map(
            params_from_iter(param_values.iter().map(String::as_str)),
            row_to_best,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    // Group in Rust by the cell key excluding backend.
    // Use Vec<(key, map)> to preserve first-seen insertion order without indexmap.
    let mut order: Vec<CellKey> = Vec::new();
    let mut groups: std::collections::HashMap<CellKey, std::collections::HashMap<String, BestRow>> =
        std::collections::HashMap::new();

    for best in bests {
        let key: CellKey = (
            best.cell.model_namespace.clone(),
            best.cell.model.clone(),
            best.cell.weight_quant.clone(),
            best.cell.kv_quant.clone(),
            best.cell.ctx_max,
            best.cell.prompt_id,
        );
        if !groups.contains_key(&key) {
            order.push(key.clone());
            groups.insert(key.clone(), std::collections::HashMap::new());
        }
        #[allow(
            clippy::unwrap_used,
            reason = "key was inserted into `groups` on the line above if absent, so get_mut is guaranteed Some"
        )]
        groups
            .get_mut(&key)
            .unwrap()
            .insert(best.cell.backend.clone(), best);
    }

    let result = order
        .into_iter()
        .map(|key| {
            let backend_map = groups.remove(&key).unwrap_or_default();
            let per_backend = backends
                .iter()
                .map(|b| ((*b).to_owned(), backend_map.get(*b).cloned()))
                .collect();
            CompareRow {
                model_namespace: key.0,
                model: key.1,
                weight_quant: key.2,
                kv_quant: key.3,
                ctx_max: key.4,
                prompt_id: key.5,
                per_backend,
            }
        })
        .collect();

    Ok(result)
}

/// `history(cell, metric?, since?)` — every observation for one cell, ordered by `ts_utc ASC`.
pub fn history(
    conn: &Connection,
    cell: &Cell,
    metric: Option<&str>,
    since_iso8601: Option<&str>,
) -> Result<Vec<ObservationRow>> {
    // Build query dynamically depending on optional filters.
    let mut sql = String::from(
        "SELECT id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
                metric, value, ts_utc, git_sha, run_id, description
         FROM observations
         WHERE backend        = ?1
           AND model_namespace = ?2
           AND model           = ?3
           AND weight_quant    = ?4
           AND kv_quant        = ?5
           AND ctx_max         = ?6
           AND prompt_id       = ?7",
    );

    let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![
        Box::new(cell.backend.clone()),
        Box::new(cell.model_namespace.clone()),
        Box::new(cell.model.clone()),
        Box::new(cell.weight_quant.clone()),
        Box::new(cell.kv_quant.clone()),
        Box::new(cell.ctx_max),
        Box::new(cell.prompt_id),
    ];

    if let Some(m) = metric {
        let idx = params.len() + 1;
        // write!(String) is infallible — let _ discards the unit Ok.
        let _ = write!(sql, " AND metric = ?{idx}");
        params.push(Box::new(m.to_owned()));
    }

    if let Some(s) = since_iso8601 {
        let idx = params.len() + 1;
        let _ = write!(sql, " AND ts_utc >= ?{idx}");
        params.push(Box::new(s.to_owned()));
    }

    sql.push_str(" ORDER BY ts_utc ASC");

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params_from_iter(params.iter().map(AsRef::as_ref)), |r| {
            Ok(ObservationRow {
                id: r.get(0)?,
                cell: Cell {
                    backend: r.get(1)?,
                    model_namespace: r.get(2)?,
                    model: r.get(3)?,
                    weight_quant: r.get(4)?,
                    kv_quant: r.get(5)?,
                    ctx_max: r.get(6)?,
                    prompt_id: r.get(7)?,
                },
                metric: r.get(8)?,
                value: r.get(9)?,
                ts_utc: r.get(10)?,
                git_sha: r.get(11)?,
                run_id: r.get(12)?,
                description: r.get(13)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(rows)
}

/// `timeseries(cell, metric, since?, bucket)` — bucketed mean per period.
pub fn timeseries(
    conn: &Connection,
    cell: &Cell,
    metric: &str,
    since_iso8601: Option<&str>,
    bucket: Bucket,
) -> Result<Vec<TimeseriesPoint>> {
    // Bucket expression for SQLite:
    // Day → substr(ts_utc, 1, 10) e.g. "2026-05-10"
    // Week → Monday of that ISO week via date arithmetic
    let bucket_expr = match bucket {
        Bucket::Day => "substr(ts_utc, 1, 10)".to_owned(),
        Bucket::Week => {
            // Land on Monday: subtract (weekday - 1 + 7) % 7 days.
            // strftime('%w') = 0 (Sunday) … 6 (Saturday).
            // We want Monday=0 offset, so: ((strftime('%w') - 1 + 7) % 7) days back.
            "date(ts_utc, '-' || ((strftime('%w', ts_utc) - 1 + 7) % 7) || ' days')".to_owned()
        }
    };

    let mut sql = format!(
        "SELECT {bucket_expr} AS bucket_start,
                AVG(value)              AS mean_value,
                COUNT(*)               AS n
         FROM observations
         WHERE backend        = ?1
           AND model_namespace = ?2
           AND model           = ?3
           AND weight_quant    = ?4
           AND kv_quant        = ?5
           AND ctx_max         = ?6
           AND prompt_id       = ?7
           AND metric          = ?8"
    );

    let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![
        Box::new(cell.backend.clone()),
        Box::new(cell.model_namespace.clone()),
        Box::new(cell.model.clone()),
        Box::new(cell.weight_quant.clone()),
        Box::new(cell.kv_quant.clone()),
        Box::new(cell.ctx_max),
        Box::new(cell.prompt_id),
        Box::new(metric.to_owned()),
    ];

    if let Some(s) = since_iso8601 {
        let idx = params.len() + 1;
        let _ = write!(sql, " AND ts_utc >= ?{idx}");
        params.push(Box::new(s.to_owned()));
    }

    sql.push_str(" GROUP BY bucket_start ORDER BY bucket_start ASC");

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params_from_iter(params.iter().map(AsRef::as_ref)), |r| {
            Ok(TimeseriesPoint {
                bucket_start_utc: r.get(0)?,
                mean_value: r.get(1)?,
                n: r.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(rows)
}

/// `deltas(since_sha, threshold_pct?)` — compare current best vs best-as-of a commit.
///
/// See docs/METRICS_DB.md §8.2 for full spec. Uses the earliest `ts_utc` of any
/// observation carrying `since_sha` (or `since_sha-dirty`) as the baseline cutoff.
/// Returns rows where the absolute delta percentage exceeds `threshold_pct`
/// (default 5.0). All cells+metrics that appear in the current `bests` view are
/// evaluated.
pub fn deltas(
    conn: &Connection,
    since_sha: &str,
    threshold_pct: Option<f64>,
) -> Result<Vec<DeltaRow>> {
    let threshold = threshold_pct.unwrap_or(5.0);

    // Step 1: find the earliest ts_utc carrying that sha (or sha-dirty).
    //
    // The binary itself no longer mints a `-dirty` suffix (`git_sha` is now
    // purely caller-supplied provenance — see `rmlx_core::runinfo`'s module
    // doc). This match arm stays regardless: 100k+ historical rows written
    // before that change carry the suffix, and `--since-sha` needs to keep
    // finding them. Do not remove this as dead code — it is live history,
    // not a live git probe.
    let dirty = format!("{since_sha}-dirty");
    let baseline_ts: Option<String> = conn
        .query_row(
            "SELECT MIN(ts_utc) FROM observations WHERE git_sha = ?1 OR git_sha = ?2",
            rusqlite::params![since_sha, dirty],
            |r| r.get(0),
        )
        .optional()?
        .flatten();

    let baseline_ts = baseline_ts.ok_or_else(|| {
        Error::Query(format!(
            "no observations found with git_sha '{since_sha}' (or '{since_sha}-dirty')"
        ))
    })?;

    // Step 2: collect all distinct (cell, metric, direction) tuples from bests.
    // We need to enumerate all cells+metrics that have observations.
    // Pull the full bests view for current bests.
    let mut stmt = conn.prepare(
        "SELECT
             id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction,
             run_id, ts_utc, git_sha, backend_version, hardware_tag,
             description, notes, inserted_by
         FROM bests",
    )?;
    let current_bests: Vec<BestRow> = stmt
        .query_map([], row_to_best)?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    // Step 3: for each current best, compute "best as of baseline_ts" (ts_utc <= baseline_ts)
    // and "best after baseline_ts" (ts_utc > baseline_ts).
    //
    // We compare POST-baseline best vs PRE-baseline best to detect regressions:
    // if a backend was performing better BEFORE the sha than it is in observations
    // recorded AFTER the sha, that signals a regression. When no post-baseline
    // observations exist, we use the all-time current best as "current".
    let best_in_window_sql = "
        SELECT value FROM (
            SELECT value,
                ROW_NUMBER() OVER (
                    ORDER BY
                        CASE WHEN direction = 'higher_better' THEN value END DESC,
                        CASE WHEN direction = 'lower_better'  THEN value END ASC,
                        ts_utc DESC
                ) AS rn
            FROM observations
            WHERE backend        = ?1
              AND model_namespace = ?2
              AND model           = ?3
              AND weight_quant    = ?4
              AND kv_quant        = ?5
              AND ctx_max         = ?6
              AND prompt_id       = ?7
              AND metric          = ?8
              AND ts_utc          <= ?9
        ) WHERE rn = 1";

    let best_after_sql = "
        SELECT value FROM (
            SELECT value,
                ROW_NUMBER() OVER (
                    ORDER BY
                        CASE WHEN direction = 'higher_better' THEN value END DESC,
                        CASE WHEN direction = 'lower_better'  THEN value END ASC,
                        ts_utc DESC
                ) AS rn
            FROM observations
            WHERE backend        = ?1
              AND model_namespace = ?2
              AND model           = ?3
              AND weight_quant    = ?4
              AND kv_quant        = ?5
              AND ctx_max         = ?6
              AND prompt_id       = ?7
              AND metric          = ?8
              AND ts_utc          > ?9
        ) WHERE rn = 1";

    let mut baseline_stmt = conn.prepare(best_in_window_sql)?;
    let mut after_stmt = conn.prepare(best_after_sql)?;

    let mut result = Vec::new();

    for current in current_bests {
        let baseline_value: Option<f64> = baseline_stmt
            .query_row(
                rusqlite::params![
                    current.cell.backend,
                    current.cell.model_namespace,
                    current.cell.model,
                    current.cell.weight_quant,
                    current.cell.kv_quant,
                    current.cell.ctx_max,
                    current.cell.prompt_id,
                    current.metric,
                    baseline_ts,
                ],
                |r| r.get(0),
            )
            .optional()?;

        // "current" value = best after baseline_ts. Fall back to all-time best if no post-baseline obs.
        let post_baseline_value: Option<f64> = after_stmt
            .query_row(
                rusqlite::params![
                    current.cell.backend,
                    current.cell.model_namespace,
                    current.cell.model,
                    current.cell.weight_quant,
                    current.cell.kv_quant,
                    current.cell.ctx_max,
                    current.cell.prompt_id,
                    current.metric,
                    baseline_ts,
                ],
                |r| r.get(0),
            )
            .optional()?;

        let effective_current = post_baseline_value.unwrap_or(current.value);

        let delta_pct = baseline_value.map(|b| {
            if b == 0.0 {
                0.0
            } else {
                (effective_current - b) / b.abs() * 100.0
            }
        });

        // Regression: for higher_better, delta_pct < -threshold; for lower_better, delta_pct > threshold.
        let regressed = match delta_pct {
            None => false,
            Some(d) => match current.direction.as_str() {
                "higher_better" => d < -threshold,
                "lower_better" => d > threshold,
                _ => false,
            },
        };

        // Only include rows where delta exists and exceeds threshold (either direction).
        let above_threshold = match delta_pct {
            None => false,
            Some(d) => d.abs() > threshold,
        };

        if above_threshold || baseline_value.is_none() {
            result.push(DeltaRow {
                cell: current.cell,
                metric: current.metric,
                direction: current.direction,
                baseline_value,
                current_value: effective_current,
                delta_pct,
                regressed,
            });
        }
    }

    Ok(result)
}

/// Compare the latest observation vs the `bests` champion for one model + metric.
///
/// `model` is matched as a **substring** of `bests.model` / `observations.model`
/// (case-sensitive) so callers can pass a short name like `"bonsai"` and it
/// matches `"prism-ml__Ternary-Bonsai-8B-mlx-2bit"`.
///
/// `kv_quant` is an optional filter; when `None` the best row across all
/// kv_quant values is selected.
///
/// Returns a single [`RegressResult`]. The exit-code idiom:
/// - champion missing → skip (caller exits 125)
/// - latest missing → skip (caller exits 125)
/// - within tolerance → ok (caller exits 0)
/// - regressed → fail (caller exits 1)
pub fn regress(
    conn: &Connection,
    model: &str,
    metric: &str,
    kv_quant: Option<&str>,
    threshold_pct: f64,
) -> Result<RegressResult> {
    use crate::registry;

    // Validate metric and read its direction from the registry.
    let (_, direction) = registry::lookup(metric)?;
    let direction_str = direction.as_str().to_owned();

    // Step 1: find the champion from `bests` for this (model, metric).
    // Match model as a substring; optionally filter by kv_quant.
    let champion_value: Option<f64> = if let Some(kv) = kv_quant {
        conn.query_row(
            "SELECT value FROM bests
              WHERE instr(model, ?1) > 0
                AND metric   = ?2
                AND kv_quant = ?3
              ORDER BY
                  CASE direction
                      WHEN 'higher_better' THEN -value
                      ELSE value
                  END ASC
              LIMIT 1",
            rusqlite::params![model, metric, kv],
            |r| r.get(0),
        )
        .optional()?
    } else {
        conn.query_row(
            "SELECT value FROM bests
              WHERE instr(model, ?1) > 0
                AND metric = ?2
              ORDER BY
                  CASE direction
                      WHEN 'higher_better' THEN -value
                      ELSE value
                  END ASC
              LIMIT 1",
            rusqlite::params![model, metric],
            |r| r.get(0),
        )
        .optional()?
    };

    // Step 2: find the most-recent observation for this (model, metric).
    let latest_value: Option<f64> = if let Some(kv) = kv_quant {
        conn.query_row(
            "SELECT value FROM observations
              WHERE instr(model, ?1) > 0
                AND metric   = ?2
                AND kv_quant = ?3
              ORDER BY ts_utc DESC
              LIMIT 1",
            rusqlite::params![model, metric, kv],
            |r| r.get(0),
        )
        .optional()?
    } else {
        conn.query_row(
            "SELECT value FROM observations
              WHERE instr(model, ?1) > 0
                AND metric = ?2
              ORDER BY ts_utc DESC
              LIMIT 1",
            rusqlite::params![model, metric],
            |r| r.get(0),
        )
        .optional()?
    };

    // Step 3: compute delta and regression flag.
    let delta_pct = match (champion_value, latest_value) {
        (Some(c), Some(l)) if c != 0.0 => Some((l - c) / c.abs() * 100.0),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };

    let regressed = match delta_pct {
        None => false,
        Some(d) => match direction_str.as_str() {
            "higher_better" => d < -threshold_pct,
            "lower_better" => d > threshold_pct,
            _ => false,
        },
    };

    // Step 4: build the human-readable message.
    let message = match (champion_value, latest_value, delta_pct) {
        (None, _, _) => format!("{model} {metric} no champion found — skip (no data)"),
        (_, None, _) => format!("{model} {metric} no observations found — skip (no data)"),
        (Some(c), Some(l), Some(d)) => {
            let status = if regressed { "REGRESSED" } else { "ok" };
            format!(
                "{model} {metric} {l:.1} vs champion {c:.1} ({d:+.1}%, threshold {threshold_pct}%) {status}"
            )
        }
        _ => format!("{model} {metric} no comparison possible — skip"),
    };

    Ok(RegressResult {
        model: model.to_owned(),
        metric: metric.to_owned(),
        direction: direction_str,
        champion_value,
        latest_value,
        delta_pct,
        regressed,
        threshold_pct,
        message,
    })
}

/// One row per (model_namespace, model, weight_quant, kv_quant) with per-metric
/// champions. When `backend_filter` is None, picks the overall champion across all
/// backends; when Some, picks per-backend champion only.
///
/// Rows are sorted by (model_namespace ASC, model ASC, weight_quant ASC, kv_quant ASC).
pub fn champions(conn: &Connection, backend_filter: Option<&str>) -> Result<Vec<ChampionRow>> {
    // Step 1: enumerate all distinct (model_namespace, model, weight_quant, kv_quant)
    // cells that have observations (optionally restricted to one backend).
    // Use a single SQL with an optional backend param via "?1 IS NULL OR backend = ?1".
    type CellKey = (String, String, String, String);

    let distinct_cells: Vec<CellKey> = {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT model_namespace, model, weight_quant, kv_quant
             FROM observations
             WHERE (?1 IS NULL OR backend = ?1)
             ORDER BY model_namespace ASC, model ASC, weight_quant ASC, kv_quant ASC",
        )?;
        let x = stmt
            .query_map(rusqlite::params![backend_filter], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        x
    };

    // Step 2: for each (cell, metric), find the champion observation.
    // For no-filter case: use the `bests` view (all-backends champion).
    // For backend-filter case: query `observations` directly applying direction ordering.

    let bests_no_filter_sql = "SELECT value, unit, backend, run_id, git_sha, ts_utc
         FROM bests
         WHERE model_namespace = ?1
           AND model           = ?2
           AND weight_quant    = ?3
           AND kv_quant        = ?4
           AND metric          = ?5
         ORDER BY
             CASE direction
                 WHEN 'higher_better' THEN -value
                 ELSE value
             END ASC
         LIMIT 1";

    let bests_with_backend_sql = "SELECT value, unit, backend, run_id, git_sha, ts_utc
         FROM observations
         WHERE model_namespace = ?1
           AND model           = ?2
           AND weight_quant    = ?3
           AND kv_quant        = ?4
           AND metric          = ?5
           AND backend         = ?6
         ORDER BY
             CASE direction
                 WHEN 'higher_better' THEN -value
                 ELSE value
             END ASC
         LIMIT 1";

    let mut stmt_no_filter = conn.prepare(bests_no_filter_sql)?;
    let mut stmt_with_backend = conn.prepare(bests_with_backend_sql)?;

    let mut rows: Vec<ChampionRow> = Vec::with_capacity(distinct_cells.len());

    for (ns, model, wq, kq) in &distinct_cells {
        let mut metrics_map = std::collections::BTreeMap::new();

        for (metric_name, _, _) in crate::registry::METRICS {
            let champion: Option<ChampionCell> = if let Some(b) = backend_filter {
                stmt_with_backend
                    .query_row(rusqlite::params![ns, model, wq, kq, metric_name, b], |r| {
                        Ok(ChampionCell {
                            value: r.get(0)?,
                            unit: r.get(1)?,
                            backend: r.get(2)?,
                            run_id: r.get(3)?,
                            git_sha: r.get(4)?,
                            ts_utc: r.get(5)?,
                        })
                    })
                    .optional()?
            } else {
                stmt_no_filter
                    .query_row(rusqlite::params![ns, model, wq, kq, metric_name], |r| {
                        Ok(ChampionCell {
                            value: r.get(0)?,
                            unit: r.get(1)?,
                            backend: r.get(2)?,
                            run_id: r.get(3)?,
                            git_sha: r.get(4)?,
                            ts_utc: r.get(5)?,
                        })
                    })
                    .optional()?
            };

            if let Some(cell) = champion {
                metrics_map.insert(metric_name.to_string(), cell);
            }
        }

        rows.push(ChampionRow {
            model_namespace: ns.clone(),
            model: model.clone(),
            weight_quant: wq.clone(),
            kv_quant: kq.clone(),
            metrics: metrics_map,
        });
    }

    Ok(rows)
}

// ── Private helpers ───────────────────────────────────────────────────────────

pub(super) fn row_to_best(r: &rusqlite::Row<'_>) -> rusqlite::Result<BestRow> {
    Ok(BestRow {
        observation_id: r.get(0)?,
        cell: Cell {
            backend: r.get(1)?,
            model_namespace: r.get(2)?,
            model: r.get(3)?,
            weight_quant: r.get(4)?,
            kv_quant: r.get(5)?,
            ctx_max: r.get(6)?,
            prompt_id: r.get(7)?,
        },
        metric: r.get(8)?,
        value: r.get(9)?,
        unit: r.get(10)?,
        direction: r.get(11)?,
        run_id: r.get(12)?,
        ts_utc: r.get(13)?,
        git_sha: r.get(14)?,
        backend_version: r.get(15)?,
        hardware_tag: r.get(16)?,
        description: r.get(17)?,
        notes: r.get(18)?,
        inserted_by: r.get(19)?,
    })
}
