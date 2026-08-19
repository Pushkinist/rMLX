use super::*;
use crate::migrate::schema_runner::run_pending;
use rusqlite::Connection;
use serde_json::json;
use tempfile::NamedTempFile;

fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    run_pending(&mut conn).unwrap();
    conn
}

fn default_opts_with_prompt_dir(dir: &Path) -> MigrateOptions {
    MigrateOptions {
        rmlx_glob: None,
        cbb_csv: None,
        records_md: None,
        hardware_tag: "m5_max_128gb".to_string(),
        prompts_dir: dir.to_path_buf(),
        inserted_by: "test-migrate@0.0.1".to_string(),
    }
}

/// Write a minimal `longctx_4k.json` to a temp directory and return the dir path.
fn make_prompt_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let prompt_path = dir.path().join("longctx_4k.json");
    let body = json!({
        "name": "longctx_4k",
        "messages": [{"role": "user", "content": "test prompt body"}],
        "tokens_approx": 4096,
        "notes": "test prompt"
    });
    std::fs::write(&prompt_path, serde_json::to_string(&body).unwrap()).unwrap();
    dir
}

// ── Weight quant inference ────────────────────────────────────────────────

#[test]
fn infer_weight_quant_mxfp8() {
    assert_eq!(
        infer_weight_quant_from_model("gemma-4-e2b-it-mxfp8"),
        "mxfp8"
    );
    assert_eq!(
        infer_weight_quant_from_model("Qwen3.6-35B-A3B-MXFP8"),
        "mxfp8"
    );
}

#[test]
fn infer_weight_quant_2bit() {
    assert_eq!(
        infer_weight_quant_from_model("Ternary-Bonsai-8B-mlx-2bit"),
        "2bit"
    );
    assert_eq!(infer_weight_quant_from_model("model-2bit"), "2bit");
}

#[test]
fn infer_weight_quant_4bit() {
    assert_eq!(infer_weight_quant_from_model("Qwen2.5-7B-4bit"), "4bit");
}

#[test]
fn infer_weight_quant_8bit() {
    assert_eq!(
        infer_weight_quant_from_model("medgemma-1.5-4b-it-8bit"),
        "8bit"
    );
    assert_eq!(
        infer_weight_quant_from_model("Qwen3.6-35B-A3B-8bit"),
        "8bit"
    );
}

#[test]
fn infer_weight_quant_paro() {
    assert_eq!(infer_weight_quant_from_model("Qwen3.6-27B-PARO"), "paro");
}

#[test]
fn infer_weight_quant_bf16_default() {
    assert_eq!(infer_weight_quant_from_model("gemma-3-4b-it"), "bf16");
    assert_eq!(infer_weight_quant_from_model("some-model"), "bf16");
}

// ── JSONL ingester ────────────────────────────────────────────────────────

fn jsonl_row_str() -> String {
    json!({
        "run_id": "20260510-030329-a6e7b9d",
        "ts_utc": "2026-05-10T03:03:29Z",
        "model_path": "/opt/open-models/mlx-community__gemma-4-e2b-it-mxfp8",
        "kv_quant": "k8v8",
        "decode_tps_mean": 106.44,
        "decode_tps_stddev": 2.08,
        "step_ms_mean": 9.4,
        "first_32_words": ["##", "The", "Roman", "Empire"],
        "git_sha": "a6e7b9d",
        "build_profile": "release",
        "notes": "step_ms_mean=wall/completion_tokens"
    })
    .to_string()
}

#[test]
fn migrate_one_jsonl_row_inserts_two_metrics() {
    let prompt_dir = make_prompt_dir();
    let mut opts = default_opts_with_prompt_dir(prompt_dir.path());

    // Write a temp JSONL file.
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), jsonl_row_str() + "\n").unwrap();
    opts.rmlx_glob = Some(tmp.path().parent().unwrap().display().to_string() + "/**/*.jsonl");
    // Point directly at the file's dir since our walker will find it.
    opts.rmlx_glob = Some(
        tmp.path()
            .parent()
            .unwrap()
            .join("**/*.jsonl")
            .display()
            .to_string(),
    );

    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut conn = test_conn();

    let inserted = ingest_jsonl_row(
        &mut conn,
        &jsonl_row_str(),
        false,
        &prompt,
        &opts,
        &mut MigrateReport::default(),
    )
    .unwrap();
    assert!(inserted, "first insert must succeed");

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2, "decode_tps_warm + step_ms_mean = 2 rows");
}

#[test]
fn migrate_jsonl_idempotent() {
    let prompt_dir = make_prompt_dir();
    let opts = default_opts_with_prompt_dir(prompt_dir.path());
    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut conn = test_conn();
    let line = jsonl_row_str();

    let first = ingest_jsonl_row(
        &mut conn,
        &line,
        false,
        &prompt,
        &opts,
        &mut MigrateReport::default(),
    )
    .unwrap();
    assert!(first);

    let second = ingest_jsonl_row(
        &mut conn,
        &line,
        false,
        &prompt,
        &opts,
        &mut MigrateReport::default(),
    )
    .unwrap();
    assert!(!second, "second insert must be skipped");

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2, "no duplicates after second run");
}

// ── CSV ingester ──────────────────────────────────────────────────────────

fn minimal_csv() -> String {
    "run_id,timestamp_utc,backend,backend_version,model_id,quant_signature,device,prompt_tokens,max_tokens,ttft_ms,itl_p50_ms,itl_p95_ms,decode_tps,overall_tps,peak_rss_mb,model_disk_gb,model_ram_gb,task_pass_at_1,success,error_class,tps_per_gb_disk,tps_per_gb_ram,output_first_64\n\
    r1,2026-05-07T07:26:32Z,ollama,0.23.0,/x/ollama__llama3.2:3b,bf16/none,gpu,4096,32,54.5,11.9,34.1,71.1,71.0,0.0,37.0,0.0,0.0,True,,1.9,0.0,The Roman Empire\n"
        .to_string()
}

fn csv_with_failed_run() -> String {
    "run_id,timestamp_utc,backend,backend_version,model_id,quant_signature,device,prompt_tokens,max_tokens,ttft_ms,itl_p50_ms,itl_p95_ms,decode_tps,overall_tps,peak_rss_mb,model_disk_gb,model_ram_gb,task_pass_at_1,success,error_class,tps_per_gb_disk,tps_per_gb_ram,output_first_64\n\
    r2,2026-05-07T08:00:00Z,rmlx,0.0.1,/x/mlx-community__gemma-4-e2b-it-mxfp8,mxfp8/k8v8,gpu,4096,32,100.0,10.0,20.0,120.0,115.0,5000.0,10.0,0.0,0.0,False,timeout,12.0,0.0,\n"
        .to_string()
}

#[test]
fn migrate_one_csv_row_inserts_multiple_metrics() {
    let prompt_dir = make_prompt_dir();
    let opts = default_opts_with_prompt_dir(prompt_dir.path());
    let mut conn = test_conn();

    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), minimal_csv()).unwrap();

    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut report = MigrateReport::default();
    migrate_cbb_csv(&mut conn, tmp.path(), &prompt, &opts, &mut report).unwrap();

    assert_eq!(report.cbb_csv_rows_inserted, 1);

    // The row carries 7 parseable metric columns. Two of them are CBB's
    // "not measured" placeholders and neither may become an observation:
    // `task_pass_at_1=0.0` (dropped at the parse site — the value alone is a
    // valid score, only the column convention says otherwise) and
    // `peak_rss_mb=0.0` (dropped by the §4.1 bounds — a live process has RSS).
    // Exact counts, not `>=`: a `>=` here passes whether 0 or 3 entries were
    // dropped, which is the whole thing under test.
    let names: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT metric FROM observations ORDER BY metric")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<String>>>()
            .unwrap()
    };
    assert_eq!(
        names,
        vec![
            "decode_tps_warm",
            "itl_p50_ms",
            "itl_p95_ms",
            "overall_tps",
            "ttft_warm_ms"
        ],
        "unexpected metric set ingested from the CBB CSV row"
    );
    assert_eq!(
        report.metrics_dropped_implausible, 1,
        "only peak_rss_mb is a bounds drop; task_pass_at_1 never reaches the record"
    );
}

/// CBB writes `0.0` in `task_pass_at_1` when it ran no quality probe. The §4.1
/// bounds cannot catch it — `0.0` pass@1 is a legitimate score for a model that
/// failed every task — so the parse site must, or the placeholder wins every
/// all-zero partition in `bests` exactly like the rate zeros do.
#[test]
fn csv_zero_task_pass_is_never_ingested() {
    let prompt_dir = make_prompt_dir();
    let opts = default_opts_with_prompt_dir(prompt_dir.path());
    let mut conn = test_conn();

    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), minimal_csv()).unwrap();

    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut report = MigrateReport::default();
    migrate_cbb_csv(&mut conn, tmp.path(), &prompt, &opts, &mut report).unwrap();

    let zero_scores: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM observations WHERE metric = 'task_pass_at_1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        zero_scores, 0,
        "CBB's unmeasured-probe placeholder was ingested as a pass@1 score"
    );
}

/// A graded run that genuinely scored zero is a different thing from an
/// unmeasured one only in the exporter's convention — but a *non-zero* score
/// must still ingest. Guards the parse-site filter against over-reach.
#[test]
fn csv_nonzero_task_pass_still_ingests() {
    let prompt_dir = make_prompt_dir();
    let opts = default_opts_with_prompt_dir(prompt_dir.path());
    let mut conn = test_conn();

    let tmp = NamedTempFile::new().unwrap();
    // Same row, with a graded (non-zero) score in the task_pass_at_1 column.
    let graded = minimal_csv().replace("0.0,37.0,0.0,0.0,True", "0.0,37.0,0.0,0.75,True");
    std::fs::write(tmp.path(), graded).unwrap();

    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut report = MigrateReport::default();
    migrate_cbb_csv(&mut conn, tmp.path(), &prompt, &opts, &mut report).unwrap();

    let score: f64 = conn
        .query_row(
            "SELECT value FROM observations WHERE metric = 'task_pass_at_1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        (score - 0.75).abs() < 1e-9,
        "graded score {score} was dropped"
    );
}

#[test]
fn migrate_csv_skips_failed_runs() {
    let prompt_dir = make_prompt_dir();
    let opts = default_opts_with_prompt_dir(prompt_dir.path());
    let mut conn = test_conn();

    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), csv_with_failed_run()).unwrap();

    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut report = MigrateReport::default();
    migrate_cbb_csv(&mut conn, tmp.path(), &prompt, &opts, &mut report).unwrap();

    assert_eq!(report.cbb_csv_rows_skipped, 1);
    assert_eq!(report.cbb_csv_rows_inserted, 0);

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn migrate_unknown_namespace_logs_and_skips_row() {
    let prompt_dir = make_prompt_dir();
    let opts = default_opts_with_prompt_dir(prompt_dir.path());
    let prompt = load_prompt_file(&prompt_dir.path().join("longctx_4k.json")).unwrap();
    let mut conn = test_conn();

    // `unknown-ns` is not in the namespace whitelist.
    let line = json!({
        "run_id": "20260510-030329-abc1234",
        "ts_utc": "2026-05-10T03:03:29Z",
        "model_path": "/x/unknown-ns__some-model-mxfp8",
        "kv_quant": "k8v8",
        "decode_tps_mean": 50.0,
        "decode_tps_stddev": 1.0,
        "step_ms_mean": 20.0,
        "git_sha": "abc1234",
        "build_profile": "release"
    })
    .to_string();

    let result = ingest_jsonl_row(
        &mut conn,
        &line,
        false,
        &prompt,
        &opts,
        &mut MigrateReport::default(),
    );
    assert!(result.is_err(), "unknown namespace must return Err");

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0, "no observations added for unknown namespace");
}

#[test]
fn migrate_records_md_fallback_optional() {
    let prompt_dir = make_prompt_dir();
    let mut opts = default_opts_with_prompt_dir(prompt_dir.path());
    let mut conn = test_conn();

    // When records_md is None — no work attempted.
    opts.records_md = None;
    let report = migrate_all(&mut conn, &opts).unwrap();
    assert_eq!(report.records_md_cells_added, 0);

    // Write a minimal MD with one cell.
    let md = "### `mlx-community__gemma-4-e2b-it-mxfp8`\n\
        | Backend | KV-quant | Decode TPS warm | Prefill TPS | TTFT cold (ms) | TTFT warm (ms) | Peak RSS (MB) | Updated |\n\
        |---|---|---:|---:|---:|---:|---:|---|\n\
        | rmlx | k8v8 | 119.14 | 1000 | 500 | 5 | 4000 | 2026-05-10 |\n";
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), md).unwrap();

    opts.records_md = Some(tmp.path().to_path_buf());
    let report2 = migrate_all(&mut conn, &opts).unwrap();
    assert_eq!(report2.records_md_cells_added, 1);

    // Idempotency: re-run adds 0.
    let report3 = migrate_all(&mut conn, &opts).unwrap();
    assert_eq!(report3.records_md_cells_added, 0);
}
