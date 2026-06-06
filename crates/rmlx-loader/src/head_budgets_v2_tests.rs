//! Schema v2 round-trip + v1 back-compat tests.

use super::{load_head_budgets, write_head_budgets, HeadBudgetCalibration, HeadBudgets};

fn v2_calibration() -> HeadBudgetCalibration {
    HeadBudgetCalibration::new_v2(
        "softmax_mass".to_string(),
        "cafebabe".to_string(),
        15,
        8192,
        0.95,
        "softmax_mass".to_string(),
        0.95,
        16,
        vec!["calibration_long_context.json".to_string()],
    )
}

fn v2_budgets(num_layers: usize, num_heads: usize) -> HeadBudgets {
    HeadBudgets::new_v2(
        "test-model-v2".to_string(),
        num_layers,
        num_heads,
        v2_calibration(),
        (0..num_layers).map(|_| vec![32_u32; num_heads]).collect(),
    )
}

#[test]
fn v2_round_trips_through_write_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets_v2.json");
    let original = v2_budgets(3, 4);
    write_head_budgets(&path, &original).unwrap();
    let loaded = load_head_budgets(&path).unwrap().expect("must parse");
    assert_eq!(loaded.version, 2);
    assert_eq!(loaded, original);
    assert_eq!(loaded.calibration.recipe.as_deref(), Some("softmax_mass"));
    assert_eq!(loaded.calibration.target_mass, Some(0.95));
    assert_eq!(loaded.calibration.target_mass_budget_floor, Some(16));
    assert_eq!(
        loaded.calibration.prompts_provenance.as_deref(),
        Some(&["calibration_long_context.json".to_string()][..])
    );
}

#[test]
fn v1_file_still_loads_after_v2_bump() {
    // A v1 JSON with no v2 fields must still load (back-compat).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets_v1.json");
    let json = serde_json::json!({
        "version": 1,
        "model_name": "legacy-bonsai",
        "num_layers": 2,
        "num_heads": 4,
        "calibration": {
            "method": "softmax_mass",
            "prompt_set_sha256": "abad1dea",
            "num_prompts": 8,
            "max_seq_len": 1024,
            "mass_threshold": 0.95
        },
        "per_layer_per_head_budget": [[64, 64, 64, 64], [128, 128, 128, 128]]
    });
    std::fs::write(&path, json.to_string()).unwrap();
    let loaded = load_head_budgets(&path).unwrap().expect("must parse v1");
    assert_eq!(loaded.version, 1);
    assert!(loaded.calibration.recipe.is_none());
    assert!(loaded.calibration.target_mass.is_none());
    assert!(loaded.calibration.target_mass_budget_floor.is_none());
    assert!(loaded.calibration.prompts_provenance.is_none());
}

#[test]
fn v3_version_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head_budgets_v3.json");
    let mut bad = v2_budgets(2, 2);
    bad.version = 3;
    std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
    let err = load_head_budgets(&path).unwrap_err();
    assert!(
        err.to_string().contains("unsupported schema version 3"),
        "unexpected: {err}"
    );
}

#[test]
fn v2_omits_none_fields_on_serialize() {
    // A v2 record with calibration constructed via v1 `new` (i.e. None v2
    // fields) round-trips cleanly: skip_serializing_if drops the absent fields.
    let cal = HeadBudgetCalibration::new("softmax_mass".to_string(), "ff".to_string(), 1, 128, 0.9);
    let json = serde_json::to_string(&cal).unwrap();
    assert!(
        !json.contains("recipe"),
        "v1 calibration leaked `recipe`: {json}"
    );
    assert!(
        !json.contains("target_mass\""),
        "v1 calibration leaked `target_mass`: {json}"
    );
    assert!(
        !json.contains("prompts_provenance"),
        "v1 calibration leaked `prompts_provenance`: {json}"
    );
}
