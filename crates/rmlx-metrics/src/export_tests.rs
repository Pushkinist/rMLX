use super::*;
use crate::{
    ingest::{MetricEntry, PromptRef, RunRecord},
    recorder::Recorder,
};
use serde_json::json;

fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

fn seed_one(conn: &mut Connection, value: f64, description: Option<&str>) {
    let mut rec = Recorder::new(conn, "test@0.0.1");
    let run = RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: "rmlx".into(),
        backend_version: Some("0.0.1".into()),
        model_namespace: "mlx-community".into(),
        model: "gemma-4-e2b-it-mxfp8".into(),
        weight_quant: "mxfp8".into(),
        kv_quant: "k8v8".into(),
        ctx_max: 8192,
        prompt: PromptRef::ByBody {
            name: "test_prompt".into(),
            body: json!("the quick brown fox"),
            notes: None,
            tokens_approx: Some(4),
        },
        ts_utc: "2026-05-10T07:30:00Z".into(),
        git_sha: Some("abc1234".into()),
        build_profile: Some("release".into()),
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: Some(4),
        max_tokens: Some(32),
        temperature: Some(0.0),
        seed: Some(0),
        n_warmups: Some(1),
        n_measure: Some(3),
        output_first_64: None,
        notes: None,
        description: description.map(ToOwned::to_owned),
        decode_config: None,
        metrics: vec![MetricEntry {
            name: "decode_tps_warm".into(),
            value: Some(value),
            stddev: None,
        }],
    };
    rec.record_run(&run).unwrap();
}

fn seed_named(
    conn: &mut Connection,
    backend: &str,
    namespace: &str,
    model: &str,
    kv_quant: &str,
    metric: &str,
    value: f64,
) {
    let mut rec = Recorder::new(conn, "test@0.0.1");
    let run = RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: backend.into(),
        backend_version: Some("0.2.8".into()),
        model_namespace: namespace.into(),
        model: model.into(),
        weight_quant: "mxfp8".into(),
        kv_quant: kv_quant.into(),
        ctx_max: 8192,
        prompt: PromptRef::ByBody {
            name: "p".into(),
            body: json!("x"),
            notes: None,
            tokens_approx: Some(1),
        },
        ts_utc: "2026-05-10T10:00:00Z".into(),
        git_sha: None,
        build_profile: None,
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: None,
        max_tokens: None,
        temperature: None,
        seed: None,
        n_warmups: None,
        n_measure: None,
        output_first_64: None,
        notes: None,
        description: None,
        decode_config: None,
        metrics: vec![MetricEntry {
            name: metric.into(),
            value: Some(value),
            stddev: None,
        }],
    };
    rec.record_run(&run).unwrap();
}

/// One drafter arm's round-loop metrics, recorded as a speculative cell.
fn seed_speculative(
    conn: &mut Connection,
    model: &str,
    decode_config: Option<&str>,
    metrics: &[(&str, f64)],
) {
    seed_speculative_at(conn, model, decode_config, 16384, metrics);
}

fn seed_speculative_at(
    conn: &mut Connection,
    model: &str,
    decode_config: Option<&str>,
    ctx_max: i64,
    metrics: &[(&str, f64)],
) {
    let mut rec = Recorder::new(conn, "test@0.0.1");
    let run = RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: "rmlx".into(),
        backend_version: Some("0.4.1".into()),
        model_namespace: "mlx-community".into(),
        model: model.into(),
        weight_quant: "mxfp8".into(),
        kv_quant: "none".into(),
        ctx_max,
        prompt: PromptRef::ByBody {
            name: "p".into(),
            body: json!("x"),
            notes: None,
            tokens_approx: Some(1),
        },
        ts_utc: "2026-09-04T10:00:00Z".into(),
        git_sha: None,
        build_profile: None,
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: None,
        max_tokens: None,
        temperature: Some(0.0),
        seed: None,
        n_warmups: None,
        n_measure: None,
        output_first_64: None,
        notes: None,
        description: None,
        decode_config: decode_config.map(ToOwned::to_owned),
        metrics: metrics
            .iter()
            .map(|(name, value)| MetricEntry {
                name: (*name).to_owned(),
                value: Some(*value),
                stddev: None,
            })
            .collect(),
    };
    rec.record_run(&run).unwrap();
}

/// One speculative run, every cell column addressable.
#[derive(Clone)]
struct SpecRun {
    backend: String,
    namespace: String,
    model: String,
    weight_quant: String,
    kv_quant: String,
    ctx_max: i64,
    prompt: String,
    decode_config: String,
}

impl SpecRun {
    fn baseline() -> Self {
        Self {
            backend: "rmlx".into(),
            namespace: "mlx-community".into(),
            model: "Qwen3.8-27B-mxfp8".into(),
            weight_quant: "mxfp8".into(),
            kv_quant: "none".into(),
            ctx_max: 16384,
            prompt: "p".into(),
            decode_config: "mtp/block=3".into(),
        }
    }
}

/// Two runs alike but for `column`. Every entry of `cell::CELL_COLUMNS` has an
/// arm, so a new column fails to compile here rather than silently rendering
/// one blended row.
fn two_runs_differing_in(column: &str) -> (SpecRun, SpecRun) {
    let a = SpecRun::baseline();
    let mut b = a.clone();
    match column {
        "backend" => b.backend = "mlx_lm".into(),
        "model_namespace" => b.namespace = "hf".into(),
        "model" => b.model = "gemma-4-e2b-it-mxfp8".into(),
        "weight_quant" => b.weight_quant = "4bit".into(),
        "kv_quant" => b.kv_quant = "k8v8".into(),
        "ctx_max" => b.ctx_max = 4096,
        "prompt_id" => b.prompt = "q".into(),
        "decode_config" => b.decode_config = "mtp/block=2".into(),
        other => panic!("{other} is a cell column with no arm here; add one"),
    }
    (a, b)
}

fn record_spec(conn: &mut Connection, run: &SpecRun) {
    let mut rec = Recorder::new(conn, "test@0.0.1");
    let record = RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: run.backend.clone(),
        backend_version: Some("0.4.1".into()),
        model_namespace: run.namespace.clone(),
        model: run.model.clone(),
        weight_quant: run.weight_quant.clone(),
        kv_quant: run.kv_quant.clone(),
        ctx_max: run.ctx_max,
        prompt: PromptRef::ByBody {
            name: run.prompt.clone(),
            body: json!(run.prompt.clone()),
            notes: None,
            tokens_approx: Some(1),
        },
        ts_utc: "2026-09-04T10:00:00Z".into(),
        git_sha: None,
        build_profile: None,
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: None,
        max_tokens: None,
        temperature: Some(0.0),
        seed: None,
        n_warmups: None,
        n_measure: None,
        output_first_64: None,
        notes: None,
        description: None,
        decode_config: Some(run.decode_config.clone()),
        metrics: vec![MetricEntry {
            name: "tokens_per_round".into(),
            value: Some(2.5),
            stddev: None,
        }],
    };
    rec.record_run(&record).unwrap();
}

fn tiny_scope() -> ScopeFile {
    ScopeFile::parse(
        r#"
[[model]]
namespace = "mlx-community"
name = "gemma-4-e2b-it-mxfp8"
arch = "Gemma4 small"
weight_quant_display = "mxfp8 g32"
order = 1
unsupported = [
  { backend = "ollama", reason = "no mxfp8 support" },
]
"#,
    )
    .unwrap()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn export_markdown_empty_db_renders_header_only_no_panic() {
    let conn = test_conn();
    let md = export_markdown(&conn, None).unwrap();
    assert!(md.contains("GENERATED FILE"));
    assert!(md.contains("do not hand-edit"));
    assert!(md.contains("No observations in database"));
    assert!(md.contains("## Provenance"));
}

#[test]
fn export_markdown_one_observation_emits_one_row() {
    let mut conn = test_conn();
    seed_one(&mut conn, 119.14, None);
    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();
    assert!(
        md.contains("119.14"),
        "expected value 119.14 in markdown:\n{md}"
    );
    assert!(md.contains("k8v8"));
    assert!(md.contains("Gemma4 small"));
}

#[test]
fn export_markdown_includes_provenance_footer() {
    let mut conn = test_conn();
    seed_one(&mut conn, 100.0, None);
    let md = export_markdown(&conn, None).unwrap();
    assert!(md.contains("## Provenance"));
    assert!(md.contains("Exported at:"));
    assert!(md.contains("Distinct champions"));
}

#[test]
fn export_markdown_does_not_include_legacy_format_emoji_or_handcurated_styling() {
    let mut conn = test_conn();
    seed_one(&mut conn, 100.0, None);
    let md = export_markdown(&conn, None).unwrap();
    assert!(!md.contains('\u{1F947}')); // 🥇
    assert!(!md.contains('\u{1F3C6}')); // 🏆
}

#[test]
fn unsupported_backend_renders_na_row() {
    let mut conn = test_conn();
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        "decode_tps_warm",
        100.0,
    );
    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();
    // ollama is declared unsupported — should render an `N/A` row even
    // without any observation.
    assert!(
        md.contains("| ollama | – | N/A "),
        "missing N/A row for unsupported backend in:\n{md}"
    );
}

#[test]
fn kv_quant_none_renders_as_bf16_kv() {
    let mut conn = test_conn();
    seed_named(
        &mut conn,
        "mlx_lm",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "none",
        "decode_tps_warm",
        88.0,
    );
    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();
    assert!(
        md.contains("| mlx-lm | bf16 KV "),
        "kv_quant=none should display as 'bf16 KV':\n{md}"
    );
}

#[test]
fn champion_summary_picks_best_decode_tps() {
    let mut conn = test_conn();
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        "decode_tps_warm",
        100.0,
    );
    seed_named(
        &mut conn,
        "mlx_lm",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "none",
        "decode_tps_warm",
        120.0,
    );
    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();
    // Champion row should cite mlx-lm with 120.0, rMLX best 100.0, gap -16.7%.
    let cs = md
        .split("## Champion summary")
        .nth(1)
        .expect("champion summary missing");
    assert!(cs.contains("120.00"), "champion summary missing 120:\n{cs}");
    assert!(
        cs.contains("mlx-lm"),
        "champion summary missing mlx-lm:\n{cs}"
    );
    assert!(
        cs.contains("100.00"),
        "champion summary missing rmlx 100:\n{cs}"
    );
    assert!(cs.contains("-16.7%"), "champion summary missing gap:\n{cs}");
}

#[test]
fn scope_filter_drops_out_of_scope_models() {
    let mut conn = test_conn();
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "Laguna-XS.2-mxfp8",
        "k8v8",
        "decode_tps_warm",
        50.0,
    );
    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();
    // Header mentions Laguna in the out-of-scope list — we only assert
    // that no per-model section was emitted for it.
    assert!(
        !md.contains("### `mlx-community__Laguna"),
        "scope filter must drop out-of-scope model section:\n{md}"
    );
}

#[test]
fn alias_merges_into_canonical_row() {
    // Scope with an alias.
    let scope = ScopeFile::parse(
        r#"
[[model]]
namespace = "z-lab"
name = "Qwen3.6-27B-PARO"
arch = "Qwen3.5MoE"
weight_quant_display = "paroquant int4"
order = 1
aliases = [
  { namespace = "hf", name = "z-lab/Qwen3.6-27B-PARO" },
]
"#,
    )
    .unwrap();
    let mut conn = test_conn();
    seed_named(
        &mut conn,
        "rmlx",
        "z-lab",
        "Qwen3.6-27B-PARO",
        "k8v4",
        "decode_tps_warm",
        28.14,
    );
    seed_named(
        &mut conn,
        "paroquant",
        "hf",
        "z-lab/Qwen3.6-27B-PARO",
        "none",
        "decode_tps_warm",
        28.83,
    );
    let md = export_markdown(&conn, Some(&scope)).unwrap();
    // Both observations should appear under the same `### \`z-lab__Qwen3.6-27B-PARO\` ...` section.
    let section_count = md.matches("### `z-lab__Qwen3.6-27B-PARO`").count();
    assert_eq!(
        section_count, 1,
        "alias did not merge — got {section_count} sections"
    );
    assert!(md.contains("28.14"));
    assert!(md.contains("28.83"));
}

#[test]
fn export_json_round_trip() {
    let mut conn = test_conn();
    seed_one(&mut conn, 77.5, None);
    let json_str = export_json(&conn).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed.is_array());
    assert!(!parsed.as_array().unwrap().is_empty());
    assert_eq!(parsed[0]["value"], 77.5);
}

#[test]
fn export_csv_header_present() {
    let conn = test_conn();
    let csv = export_csv(&conn).unwrap();
    let first_line = csv.lines().next().unwrap_or("");
    assert!(
        first_line.starts_with("backend,model_namespace,"),
        "unexpected CSV header: {first_line}"
    );
}

#[test]
fn export_csv_escapes_commas_in_values() {
    let mut conn = test_conn();
    seed_one(&mut conn, 55.0, Some("a,b description"));
    let csv = export_csv(&conn).unwrap();
    assert!(
        csv.contains("\"a,b description\""),
        "comma in description not escaped in CSV:\n{csv}"
    );
}

#[test]
fn export_jsonl_one_row_per_line() {
    let mut conn = test_conn();
    seed_one(&mut conn, 42.0, None);
    seed_one(&mut conn, 43.0, Some("strictly better"));
    let jsonl = export_jsonl(&conn).unwrap();
    let lines: Vec<&str> = jsonl.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "expected 1 champion line, got {}",
        lines.len()
    );
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line).unwrap();
    }
}

// ── KV GB + reduction_vs_bf16 tests ───────────────────────────────────────

/// Helpers for KV-column tests: seed a kv_cache_bytes observation.
fn seed_kv_bytes(conn: &mut Connection, namespace: &str, model: &str, kv_quant: &str, bytes: f64) {
    seed_named(
        conn,
        "rmlx",
        namespace,
        model,
        kv_quant,
        "kv_cache_bytes",
        bytes,
    );
}

#[test]
fn kv_gb_column_renders_nonzero_value() {
    let mut conn = test_conn();
    // 2 GiB of KV cache.
    let two_gib = 2.0 * 1024.0 * 1024.0 * 1024.0;
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        "decode_tps_warm",
        100.0,
    );
    seed_kv_bytes(
        &mut conn,
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        two_gib,
    );

    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();

    // Should contain "2.000 GB" in the KV GB column.
    assert!(
        md.contains("2.000 GB"),
        "expected '2.000 GB' in markdown KV GB column:\n{md}"
    );
    // Header should contain the new column label.
    assert!(
        md.contains("KV GB"),
        "expected 'KV GB' column header:\n{md}"
    );
}

#[test]
fn reduction_vs_bf16_is_less_than_one_for_quantized_row() {
    let mut conn = test_conn();
    // bf16 baseline: 8 GiB (kv_quant = "none")
    let bf16_bytes = 8.0 * 1024.0 * 1024.0 * 1024.0;
    // k8v8 quantized: 2 GiB → reduction = 2/8 = 0.25x
    let quant_bytes = 2.0 * 1024.0 * 1024.0 * 1024.0;

    // Seed decode_tps_warm so rows appear in the table.
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "none",
        "decode_tps_warm",
        80.0,
    );
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        "decode_tps_warm",
        100.0,
    );
    // Seed KV bytes for both quants.
    seed_kv_bytes(
        &mut conn,
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "none",
        bf16_bytes,
    );
    seed_kv_bytes(
        &mut conn,
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        quant_bytes,
    );

    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();

    // k8v8 row: reduction = 2/8 = 0.25x
    assert!(
        md.contains("0.25x"),
        "expected '0.25x' reduction for k8v8 row:\n{md}"
    );
    // bf16/none row: reduction = 8/8 = 1.00x
    assert!(
        md.contains("1.00x"),
        "expected '1.00x' reduction for bf16/none row:\n{md}"
    );
}

#[test]
fn reduction_blank_when_no_bf16_baseline() {
    let mut conn = test_conn();
    // Only k8v8 row, no "none" baseline.
    let quant_bytes = 2.0 * 1024.0 * 1024.0 * 1024.0;
    seed_named(
        &mut conn,
        "rmlx",
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        "decode_tps_warm",
        100.0,
    );
    seed_kv_bytes(
        &mut conn,
        "mlx-community",
        "gemma-4-e2b-it-mxfp8",
        "k8v8",
        quant_bytes,
    );

    let scope = tiny_scope();
    let md = export_markdown(&conn, Some(&scope)).unwrap();

    // KV GB column has a real value.
    assert!(md.contains("2.000 GB"), "expected kv_gb to show:\n{md}");
    // reduction column must show '-' because no bf16 baseline exists.
    // We verify the row contains "| 2.000 GB | - |" pattern.
    assert!(
        md.contains("| 2.000 GB | - |"),
        "expected reduction to be '-' when no bf16 baseline:\n{md}"
    );
}

#[test]
fn build_kv_bytes_map_keeps_minimum() {
    // Unit test for the helper itself, no DB needed.
    use crate::query::Cell;

    let make_row = |kv_quant: &str, bytes: f64| BestRow {
        observation_id: 1,
        cell: Cell {
            backend: "rmlx".into(),
            model_namespace: "mlx-community".into(),
            model: "test-model".into(),
            weight_quant: "mxfp8".into(),
            kv_quant: kv_quant.into(),
            ctx_max: 8192,
            prompt_id: 1,
            decode_config: None,
        },
        metric: "kv_cache_bytes".into(),
        value: bytes,
        unit: "bytes".into(),
        direction: "lower_better".into(),
        run_id: "run1".into(),
        ts_utc: "2026-05-01T00:00:00Z".into(),
        git_sha: None,
        backend_version: None,
        hardware_tag: "m5_max_128gb".into(),
        description: None,
        notes: None,
        inserted_by: "test".into(),
    };

    // Two rows for k8v8: 4 GiB and 3 GiB. Map should keep 3 GiB (minimum).
    let four_gib = 4.0 * 1024.0_f64.powi(3);
    let three_gib = 3.0 * 1024.0_f64.powi(3);
    let rows = vec![make_row("k8v8", four_gib), make_row("k8v8", three_gib)];
    let map = build_kv_bytes_map(&rows);
    let key = (
        "mlx-community".to_owned(),
        "test-model".to_owned(),
        8192_i64,
        "k8v8".to_owned(),
    );
    assert_eq!(map[&key], three_gib);
}

/// Two arms of one cell are two rows in the published table, each labelled with
/// the configuration it measured. Merging them puts the drafter's number under a
/// label that describes plain decode, and the reader has nothing to notice with.
#[test]
fn the_markdown_table_keeps_the_two_arms_apart() {
    let mut conn = test_conn();
    seed_one(&mut conn, 142.5, None);
    {
        let mut rec = Recorder::new(&mut conn, "test@0.0.1");
        let mut run = RunRecord {
            schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
            backend: "rmlx".into(),
            backend_version: Some("0.0.1".into()),
            model_namespace: "mlx-community".into(),
            model: "gemma-4-e2b-it-mxfp8".into(),
            weight_quant: "mxfp8".into(),
            kv_quant: "k8v8".into(),
            ctx_max: 8192,
            prompt: PromptRef::ByBody {
                name: "test_prompt".into(),
                body: json!("the quick brown fox"),
                notes: None,
                tokens_approx: Some(4),
            },
            ts_utc: "2026-05-10T07:31:00Z".into(),
            git_sha: Some("abc1234".into()),
            build_profile: Some("release".into()),
            hardware_tag: "m5_max_128gb".into(),
            prompt_tokens: Some(4),
            max_tokens: Some(32),
            temperature: Some(0.0),
            seed: Some(0),
            n_warmups: Some(1),
            n_measure: Some(3),
            output_first_64: None,
            notes: None,
            description: None,
            decode_config: Some("mtp/block=5".into()),
            metrics: vec![MetricEntry {
                name: "decode_tps_warm".into(),
                value: Some(275.7),
                stddev: None,
            }],
        };
        run.decode_config = Some("mtp/block=5".into());
        rec.record_run(&run).unwrap();
    }

    let md = export_markdown(&conn, None).unwrap();
    assert!(md.contains("| Decode "), "no Decode column:\n{md}");
    assert!(md.contains("| plain "), "no plain-decode row:\n{md}");
    assert!(md.contains("| mtp/block=5 "), "no speculative row:\n{md}");
    assert!(md.contains("142.5"), "plain value missing:\n{md}");
    assert!(md.contains("275.7"), "speculative value missing:\n{md}");
}

/// The CSV carries the column too, so a consumer reading it can tell the rows
/// apart without re-deriving anything.
#[test]
fn the_csv_carries_the_decode_configuration() {
    let mut conn = test_conn();
    seed_one(&mut conn, 142.5, None);
    let csv = export_csv(&conn).unwrap();
    let header = csv.lines().next().unwrap();
    assert!(header.contains("decode_config"), "{header}");
}

// ── Speculative section ───────────────────────────────────────────────────

/// The round-loop figures reach the export, and the arm they belong to is
/// named beside them.
#[test]
fn the_speculative_section_carries_the_round_loop_figures() {
    let mut conn = test_conn();
    seed_speculative(
        &mut conn,
        "Qwen3.8-27B-mxfp8",
        Some("mtp/block=3"),
        &[
            ("decode_tps_warm", 25.5),
            ("accept_rate", 0.728),
            ("tokens_per_round", 2.46),
            ("accepted_per_step", 1.46),
            ("draft_ms_per_round", 12.5),
            ("verify_ms_per_round", 44.0),
            ("loop_ms_per_round", 8.25),
        ],
    );
    let md = export_markdown(&conn, None).unwrap();

    assert!(md.contains("## Speculative decoding"), "{md}");
    let section = md
        .split("## Speculative decoding")
        .nth(1)
        .expect("section present");

    // The whole row, not `contains` per value: every one of these numbers is
    // present under any column order, so only the row as rendered can tell a
    // reordered table from a correct one.
    let row = section
        .lines()
        .find(|l| l.contains("mtp/block=3"))
        .expect("the arm's row");
    let cells: Vec<&str> = row.trim_matches('|').split('|').map(str::trim).collect();
    assert_eq!(
        &cells[..10],
        &[
            "`mlx-community__Qwen3.8-27B-mxfp8`",
            "mtp/block=3",
            "rmlx",
            "bf16 KV",
            "mxfp8",
            "16384",
            "1",
            "25.50",
            "0.728",
            "2.46",
        ],
        "row rendered as {row}"
    );
    assert_eq!(&cells[10..13], &["1.46", "12.50", "44.00"], "{row}");
    assert_eq!(cells[13], "8.25", "{row}");

    // Header and body come from one array, so the heading order is the value
    // order. Pinned here so a reorder is a diff in two places, not one.
    let header = section
        .lines()
        .find(|l| l.contains("Tokens/round"))
        .expect("the heading row");
    let labels: Vec<&str> = header.trim_matches('|').split('|').map(str::trim).collect();
    assert_eq!(
        &labels[7..],
        &[
            "Decode TPS warm",
            "Accept rate",
            "Tokens/round",
            "Accepted/step",
            "Draft ms/round",
            "Verify ms/round",
            "Loop ms/round",
            "Updated",
        ],
        "heading rendered as {header}"
    );
}

/// One row is one cell. Two contexts are two cells, and merging them would take
/// the higher-better token rate from one and the lower-better milliseconds from
/// the other — under a heading saying the three millisecond columns partition
/// one round.
#[test]
fn two_contexts_are_two_rows_not_one_blended_row() {
    let mut conn = test_conn();
    seed_speculative_at(
        &mut conn,
        "Qwen3.8-27B-mxfp8",
        Some("mtp/block=3"),
        4096,
        &[("tokens_per_round", 2.46), ("verify_ms_per_round", 44.0)],
    );
    seed_speculative_at(
        &mut conn,
        "Qwen3.8-27B-mxfp8",
        Some("mtp/block=3"),
        131_072,
        &[("tokens_per_round", 1.90), ("verify_ms_per_round", 210.0)],
    );
    let md = export_markdown(&conn, None).unwrap();
    let section = md
        .split("## Speculative decoding")
        .nth(1)
        .expect("section present");
    let rows: Vec<&str> = section
        .lines()
        .filter(|l| l.contains("mtp/block=3"))
        .collect();
    assert_eq!(rows.len(), 2, "one row per context: {section}");

    let short = rows
        .iter()
        .find(|l| l.contains("| 4096 "))
        .expect("the 4k row");
    assert!(short.contains("2.46"), "{short}");
    assert!(short.contains("44.00"), "{short}");
    assert!(
        !short.contains("210.00"),
        "the 4k row took a millisecond figure from the 128k run: {short}"
    );
}

/// Every cell column separates rows, driven off `CELL_COLUMNS` itself.
///
/// `SpecRowKey`'s doc claims to carry all of them and nothing enforced it. That
/// column list is the module that exists because `decode_config` reached the
/// view and none of the consumers, and nothing failed; a ninth column would
/// reproduce it in this exact tuple and re-blend the rows this table separates.
/// Driven off the constant, a new column fails here until it has a fixture and
/// a key element.
#[test]
fn every_cell_column_separates_two_speculative_rows() {
    for col in crate::cell::CELL_COLUMNS {
        let mut conn = test_conn();
        let (a, b) = two_runs_differing_in(col.name);
        record_spec(&mut conn, &a);
        record_spec(&mut conn, &b);

        let md = export_markdown(&conn, None).unwrap();
        let Some(section) = md.split("## Speculative decoding").nth(1) else {
            panic!("{}: no speculative section rendered", col.name);
        };
        let rows = section
            .lines()
            .filter(|l| l.starts_with("| `") && l.contains("block="))
            .count();
        assert_eq!(
            rows, 2,
            "{} does not separate two rows; they were blended into one:\n{section}",
            col.name
        );
    }
}

/// The rendered set is the declared set. A speculative metric added to the
/// registry and not to the table would otherwise be recorded and never seen.
#[test]
fn the_table_renders_every_declared_speculative_metric() {
    use crate::registry::{SpecRole, SPEC_METRICS};

    let rendered: Vec<&str> = SPEC_METRIC_COLUMNS.iter().map(|c| c.db_name).collect();
    for (name, role) in SPEC_METRICS {
        match role {
            SpecRole::Derived => assert!(
                rendered.contains(name),
                "{name} is a declared speculative figure and has no column"
            ),
            SpecRole::Counter => assert!(
                !rendered.contains(name),
                "{name} is a cumulative total and says nothing as a column"
            ),
        }
    }
    for col in SPEC_METRIC_COLUMNS {
        assert!(
            col.db_name == "decode_tps_warm"
                || SPEC_METRICS.iter().any(|(name, _)| *name == col.db_name),
            "{} has a column and is not a declared speculative metric",
            col.db_name
        );
        assert!(!col.label.is_empty(), "{} has no heading", col.db_name);
    }
    // Every declared name is a real metric, or the registry's own gate would
    // not see it.
    for (name, _) in SPEC_METRICS {
        assert!(
            crate::registry::lookup(name).is_ok(),
            "{name} is declared speculative and is not in METRICS"
        );
    }
}

/// An adaptive arm and a fixed arm at the same ceiling are two rows, not one:
/// they are two cells, and merging them would publish one loop's figures under
/// the other's label.
#[test]
fn an_adaptive_arm_is_a_row_of_its_own() {
    let mut conn = test_conn();
    seed_speculative(
        &mut conn,
        "Qwen3.6-35B-A3B-8bit",
        Some("dflash/block=16"),
        &[("tokens_per_round", 9.5)],
    );
    seed_speculative(
        &mut conn,
        "Qwen3.6-35B-A3B-8bit",
        Some("dflash/block=16,dflash/depth=accept_rate"),
        &[("tokens_per_round", 2.1)],
    );
    let md = export_markdown(&conn, None).unwrap();
    let section = md
        .split("## Speculative decoding")
        .nth(1)
        .expect("section present");
    assert!(section.contains("9.50"), "{section}");
    assert!(section.contains("2.10"), "{section}");
}

/// A non-drafter `decode_config` is not a speculative arm. Keying the section
/// on "the column is not NULL" would file a prefill-chunk sweep here and report
/// a blank round loop for it.
#[test]
fn a_prefill_chunk_sweep_does_not_reach_the_speculative_section() {
    let mut conn = test_conn();
    seed_speculative(
        &mut conn,
        "gemma-4-e2b-it-mxfp8",
        Some("prefill_chunk=2048"),
        &[("decode_tps_warm", 80.0)],
    );
    let md = export_markdown(&conn, None).unwrap();
    assert!(
        !md.contains("## Speculative decoding"),
        "a prefill-chunk row opened a speculative section: {md}"
    );
}

/// No drafter measured means no heading. An empty table under one reads as a
/// verdict about drafters rather than about the database.
#[test]
fn a_database_with_no_drafter_renders_no_speculative_section() {
    let mut conn = test_conn();
    seed_one(&mut conn, 100.0, None);
    let md = export_markdown(&conn, None).unwrap();
    assert!(!md.contains("## Speculative decoding"), "{md}");
}
