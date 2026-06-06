//! Tests for `head_budgets` (schema + reader).

use super::{load_head_budgets, write_head_budgets, HeadBudgetCalibration, HeadBudgets};

fn sample_calibration() -> HeadBudgetCalibration {
    HeadBudgetCalibration::new(
        "softmax_mass".to_string(),
        "deadbeef".to_string(),
        16,
        4096,
        0.95,
    )
}

fn sample_budgets(num_layers: usize, num_heads: usize) -> HeadBudgets {
    HeadBudgets {
        version: 1,
        model_name: "test-model".to_string(),
        num_layers,
        num_heads,
        calibration: sample_calibration(),
        per_layer_per_head_budget: (0..num_layers).map(|_| vec![64_u32; num_heads]).collect(),
    }
}

// ── load_head_budgets: success cases ──────────────────────────────────────────

#[test]
fn missing_file_returns_ok_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let result = load_head_budgets(&path).unwrap();
    assert!(result.is_none(), "missing file must return Ok(None)");
}

#[test]
fn valid_file_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let original = sample_budgets(4, 8);
    let json = serde_json::to_string_pretty(&original).unwrap();
    std::fs::write(&path, json).unwrap();

    let loaded = load_head_budgets(&path).unwrap().expect("must parse");
    assert_eq!(loaded, original);
}

#[test]
fn schema_v1_minimal_json_parses() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let json = serde_json::json!({
        "version": 1,
        "model_name": "tiny",
        "num_layers": 2,
        "num_heads": 2,
        "calibration": {
            "method": "softmax_mass",
            "prompt_set_sha256": "ab",
            "num_prompts": 1,
            "max_seq_len": 128,
            "mass_threshold": 0.9
        },
        "per_layer_per_head_budget": [[16, 16], [32, 32]]
    });
    std::fs::write(&path, json.to_string()).unwrap();
    let loaded = load_head_budgets(&path).unwrap().expect("must parse");
    assert_eq!(loaded.num_layers, 2);
    assert_eq!(loaded.per_layer_per_head_budget[1][0], 32);
}

// ── load_head_budgets: validation failures ────────────────────────────────────

#[test]
fn unsupported_version_err() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let mut bad = sample_budgets(2, 2);
    bad.version = 99;
    std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
    let result = load_head_budgets(&path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unsupported schema version"));
}

#[test]
fn row_count_mismatch_err() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let mut bad = sample_budgets(4, 2);
    bad.per_layer_per_head_budget.pop();
    std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
    let result = load_head_budgets(&path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("expected num_layers=4"));
}

#[test]
fn column_count_mismatch_err() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let mut bad = sample_budgets(2, 4);
    bad.per_layer_per_head_budget[1].truncate(2);
    std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
    let result = load_head_budgets(&path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("expected num_heads=4"));
}

#[test]
fn zero_budget_err() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let mut bad = sample_budgets(2, 2);
    bad.per_layer_per_head_budget[0][1] = 0;
    std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
    let result = load_head_budgets(&path);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("is zero"), "unexpected error: {msg}");
}

// ── write_head_budgets: round-trip ───────────────────────────────────────────

#[test]
fn write_round_trips_through_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let original = sample_budgets(3, 4);
    write_head_budgets(&path, &original).unwrap();
    let loaded = load_head_budgets(&path).unwrap().expect("must parse");
    assert_eq!(loaded, original);
}

#[test]
fn write_rejects_malformed_table() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    let mut bad = sample_budgets(2, 2);
    bad.per_layer_per_head_budget[0][0] = 0;
    let err = write_head_budgets(&path, &bad).unwrap_err();
    assert!(err.to_string().contains("is zero"), "unexpected: {err}");
}

#[test]
fn malformed_json_err() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets.json");
    std::fs::write(&path, b"not json").unwrap();
    let result = load_head_budgets(&path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("malformed head_budgets.json"));
}
