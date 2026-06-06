//! Tests for `calibration_writer`.

use std::collections::BTreeMap;

use super::{
    discover_kv_calibration, outlier_count_for, read_kv_calibration, recipe_to_internal,
    write_kv_calibration, CalibrationMeta, CodebookOverride, KvCalibration, LayerCalib,
};

// ── recipe_to_internal ────────────────────────────────────────────────────────

#[test]
fn recipe_mapping_turbo2() {
    assert_eq!(recipe_to_internal("turbo2").unwrap(), "turboquant25");
    assert_eq!(recipe_to_internal("turbo2_tcq").unwrap(), "turboquant25");
}

#[test]
fn recipe_mapping_turbo3_and_turbo4() {
    assert_eq!(recipe_to_internal("turbo3").unwrap(), "turboquant35");
    assert_eq!(recipe_to_internal("turbo3_tcq").unwrap(), "turboquant35");
    assert_eq!(recipe_to_internal("turbo4").unwrap(), "turboquant35");
}

#[test]
fn recipe_mapping_identity_passthrough() {
    // mtq's recipe_map passes internal names through unchanged (lines 92-93).
    assert_eq!(recipe_to_internal("turboquant25").unwrap(), "turboquant25");
    assert_eq!(recipe_to_internal("turboquant35").unwrap(), "turboquant35");
}

#[test]
fn recipe_mapping_unknown_returns_err() {
    assert!(recipe_to_internal("turbo99").is_err());
    assert!(recipe_to_internal("").is_err());
}

// ── outlier_count_for ─────────────────────────────────────────────────────────

#[test]
fn outlier_count_head128_turbo3() {
    // head_dim=128, turbo3 (turboquant35): ratio=0.50, aligned to 16 → 64
    let n = outlier_count_for(128, "turboquant35").unwrap();
    assert_eq!(n, 64);
}

#[test]
fn outlier_count_head128_turbo2() {
    // head_dim=128, turbo2 (turboquant25): ratio=0.25, aligned to 16 → 32
    let n = outlier_count_for(128, "turboquant25").unwrap();
    assert_eq!(n, 32);
}

#[test]
fn outlier_count_head64_turbo3() {
    // head_dim=64, turboquant35: 0.50 * 64 = 32, aligned → 32
    let n = outlier_count_for(64, "turboquant35").unwrap();
    assert_eq!(n, 32);
}

#[test]
fn outlier_count_head256_turbo2() {
    // head_dim=256, turboquant25: 0.25 * 256 = 64, aligned → 64
    let n = outlier_count_for(256, "turboquant25").unwrap();
    assert_eq!(n, 64);
}

#[test]
fn outlier_count_unknown_internal_recipe() {
    assert!(outlier_count_for(128, "turboquant99").is_err());
}

// ── round-trip (serialize → deserialize → structural equality) ────────────────

fn make_calib() -> KvCalibration {
    let mut layers = BTreeMap::new();
    layers.insert(
        "model.layers.0.self_attn".to_string(),
        LayerCalib {
            key_high_precision_indices: vec![vec![0, 3, 7], vec![1, 4, 8]],
            value_high_precision_indices: vec![vec![2, 5, 9], vec![0, 6, 10]],
            codebook: None,
        },
    );
    layers.insert(
        "model.layers.1.self_attn".to_string(),
        LayerCalib {
            key_high_precision_indices: vec![vec![0, 1, 2]],
            value_high_precision_indices: vec![vec![3, 4, 5]],
            codebook: None,
        },
    );
    KvCalibration {
        version: 1,
        recipe: "turboquant35".to_string(),
        head_size: 128,
        model_name: "test-model".to_string(),
        transform_version: "structured_hadamard_v1".to_string(),
        codebook_version: "lloyd_beta_v1".to_string(),
        layers,
        calibration: CalibrationMeta {
            method: "weight_norm".to_string(),
            objective: "l2_norm".to_string(),
            num_prompts: 0,
            max_seq_len: 0,
            batch_size: 0,
            num_observed_tokens: 0,
            dtype: "float16".to_string(),
            device: "cpu".to_string(),
            prompts_sha256: String::new(),
        },
        head_budgets: None,
    }
}

#[test]
fn round_trip_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    let original = make_calib();

    write_kv_calibration(&path, &original).unwrap();
    let loaded = read_kv_calibration(&path).unwrap();

    assert_eq!(original, loaded);
}

#[test]
fn round_trip_preserves_layer_order() {
    // BTreeMap keys are sorted — layer order in JSON must be deterministic.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    let original = make_calib();

    write_kv_calibration(&path, &original).unwrap();
    let json = std::fs::read_to_string(&path).unwrap();

    let first_pos = json.find("model.layers.0").unwrap();
    let second_pos = json.find("model.layers.1").unwrap();
    assert!(
        first_pos < second_pos,
        "layer 0 must precede layer 1 in JSON"
    );
}

#[test]
fn version_field_is_1() {
    let calib = make_calib();
    assert_eq!(calib.version, 1);
}

#[test]
fn write_creates_parent_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("a").join("b").join("kv_calib.json");
    let calib = make_calib();
    write_kv_calibration(&nested, &calib).unwrap();
    assert!(nested.exists());
}

// ── rounding regression (MEDIUM 3) ───────────────────────────────────────────

/// Rust uses round-half-away-from-zero; these cases confirm the Rust result.
/// For standard head_dims (64/128/256) `raw / GROUP_ALIGNMENT` is never at
/// an exact midpoint, so Rust and Python banker's rounding agree for all
/// real models.
#[test]
fn outlier_count_rounding_head80_turbo3() {
    // head_dim=80, turboquant35: 80*0.5=40; 40/16=2.5 → Rust away-from-zero = 3*16 = 48
    // Python banker's round(2.5) = 2 → 32. Divergence pinned here.
    let n = outlier_count_for(80, "turboquant35").unwrap();
    // Rust rounds 2.5 away from zero → 3 → 48.
    assert_eq!(
        n, 48,
        "Rust away-from-zero rounding: head_dim=80, turbo35 → 48"
    );
}

#[test]
fn outlier_count_rounding_head160_turbo2() {
    // head_dim=160, turboquant25: 160*0.25=40; 40/16=2.5 → Rust = 3*16 = 48
    // Python banker's round(2.5) = 2 → 32. Divergence pinned here.
    let n = outlier_count_for(160, "turboquant25").unwrap();
    assert_eq!(
        n, 48,
        "Rust away-from-zero rounding: head_dim=160, turbo25 → 48"
    );
}

// ── schema-parity with mtq (HIGH 2) ─────────────────────────────────────────

/// Serialized `KvCalibration` must contain exactly the same top-level keys
/// as mtq's `turboquant_kv.json` v1 (generate_metadata.py lines 173-191).
/// Expected key set: version, recipe, head_size, model_name,
/// transform_version, codebook_version, layers, calibration.
#[test]
fn schema_top_level_keys_match_mtq() {
    use std::collections::BTreeSet;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    let calib = make_calib();
    write_kv_calibration(&path, &calib).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let keys: BTreeSet<String> = val
        .as_object()
        .expect("top-level must be a JSON object")
        .keys()
        .cloned()
        .collect();

    let expected: BTreeSet<String> = [
        "version",
        "recipe",
        "head_size",
        "model_name",
        "transform_version",
        "codebook_version",
        "layers",
        "calibration",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();

    assert_eq!(
        keys, expected,
        "top-level JSON keys must match mtq v1 schema exactly"
    );
}

/// Reading an mtq-style fixture (no `recipe_user` field) must succeed.
#[test]
fn read_mtq_style_fixture_no_recipe_user() {
    let fixture = r#"{
  "version": 1,
  "recipe": "turboquant35",
  "head_size": 128,
  "model_name": "my-model",
  "transform_version": "structured_hadamard_v1",
  "codebook_version": "lloyd_beta_v1",
  "layers": {
    "model.layers.0.self_attn": {
      "key_high_precision_indices": [[0, 1, 2]],
      "value_high_precision_indices": [[3, 4, 5]]
    }
  },
  "calibration": {
    "method": "weight_norm",
    "objective": "l2_norm",
    "num_prompts": 0,
    "max_seq_len": 0,
    "batch_size": 0,
    "num_observed_tokens": 0,
    "dtype": "float16",
    "device": "cpu",
    "prompts_sha256": ""
  }
}"#;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("turboquant_kv.json");
    std::fs::write(&path, fixture.as_bytes()).unwrap();

    let calib = read_kv_calibration(&path).unwrap();
    assert_eq!(calib.version, 1);
    assert_eq!(calib.recipe, "turboquant35");
    assert_eq!(calib.head_size, 128);
    assert_eq!(calib.layers.len(), 1);
    assert_eq!(calib.calibration.dtype, "float16");
}

// ── discover_kv_calibration ───────────────────────────────────────────────────

/// No kv_calib.json → returns None (no error).
#[test]
fn discover_returns_none_when_file_absent() {
    let dir = tempfile::tempdir().unwrap();
    let result = discover_kv_calibration(dir.path(), 128);
    assert!(
        result.is_none(),
        "should return None when kv_calib.json is absent"
    );
}

/// Valid file + matching head_size → returns Some.
#[test]
fn discover_returns_some_for_valid_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    let calib = make_calib(); // head_size = 128
    write_kv_calibration(&path, &calib).unwrap();

    let result = discover_kv_calibration(dir.path(), 128);
    assert!(
        result.is_some(),
        "should return Some for valid file with matching head_size"
    );
    assert_eq!(result.unwrap().layers.len(), 2);
}

/// Valid file but head_size mismatch → returns None (warns).
#[test]
fn discover_returns_none_on_head_size_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    let calib = make_calib(); // head_size = 128
    write_kv_calibration(&path, &calib).unwrap();

    let result = discover_kv_calibration(dir.path(), 256);
    assert!(
        result.is_none(),
        "should return None when head_size (256) != calib.head_size (128)"
    );
}

/// Invalid schema version → returns None (warns).
#[test]
fn discover_returns_none_on_version_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    let mut calib = make_calib();
    calib.version = 2; // wrong version
    write_kv_calibration(&path, &calib).unwrap();

    let result = discover_kv_calibration(dir.path(), 128);
    assert!(
        result.is_none(),
        "should return None for unsupported schema version"
    );
}

/// Malformed JSON → returns None (warns).
#[test]
fn discover_returns_none_on_malformed_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib.json");
    std::fs::write(&path, b"{ this is not valid json }").unwrap();

    let result = discover_kv_calibration(dir.path(), 128);
    assert!(result.is_none(), "should return None for malformed JSON");
}

// ── CodebookOverride schema tests ─────────────────────────────────────────────

/// Round-trip a KvCalibration with a V-side codebook override — serialize +
/// deserialize must preserve the codebook exactly.
#[test]
fn codebook_override_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib_v11.json");

    let mut layers = BTreeMap::new();
    layers.insert(
        "model.layers.0.self_attn".to_string(),
        LayerCalib {
            key_high_precision_indices: vec![vec![0, 1, 2]],
            value_high_precision_indices: vec![vec![3, 4, 5]],
            codebook: Some(CodebookOverride {
                value: Some(vec![-1.5, -0.5, 0.5, 1.5]),
            }),
        },
    );
    layers.insert(
        "model.layers.1.self_attn".to_string(),
        LayerCalib {
            key_high_precision_indices: vec![vec![6, 7]],
            value_high_precision_indices: vec![vec![8, 9]],
            codebook: None,
        },
    );
    let original = KvCalibration {
        version: 1,
        recipe: "turboquant25".to_string(),
        head_size: 128,
        model_name: "test-model".to_string(),
        transform_version: "structured_hadamard_v1".to_string(),
        codebook_version: "custom_v1".to_string(),
        layers,
        calibration: CalibrationMeta {
            method: "weight_norm".to_string(),
            objective: "l2_norm".to_string(),
            num_prompts: 0,
            max_seq_len: 0,
            batch_size: 0,
            num_observed_tokens: 0,
            dtype: "bfloat16".to_string(),
            device: "cpu".to_string(),
            prompts_sha256: String::new(),
        },
        head_budgets: None,
    };

    write_kv_calibration(&path, &original).unwrap();
    let loaded = read_kv_calibration(&path).unwrap();

    assert_eq!(original, loaded, "round-trip must be structurally equal");

    // Verify the codebook was preserved.
    let l0 = loaded.layers.get("model.layers.0.self_attn").unwrap();
    let cb = l0.codebook.as_ref().unwrap();
    assert_eq!(
        cb.value.as_deref(),
        Some(&[-1.5_f32, -0.5, 0.5, 1.5][..]),
        "value codebook must round-trip exactly"
    );
    // K-side key field was dropped in calibration review — only `value` is present.

    // Layer 1 has no codebook.
    let l1 = loaded.layers.get("model.layers.1.self_attn").unwrap();
    assert!(l1.codebook.is_none(), "layer 1 codebook must be None");
}

/// v1 fixture (no `codebook` field) must parse to `codebook = None` on each layer.
///
/// This is the backwards-compatibility regression test: mtq v1 files (without
/// the rMLX v1.1 `codebook` extension) must still parse cleanly.
#[test]
fn v1_fixture_no_codebook_field_parses_to_none() {
    let fixture = r#"{
  "version": 1,
  "recipe": "turboquant35",
  "head_size": 128,
  "model_name": "my-model",
  "transform_version": "structured_hadamard_v1",
  "codebook_version": "lloyd_beta_v1",
  "layers": {
    "model.layers.0.self_attn": {
      "key_high_precision_indices": [[0, 1]],
      "value_high_precision_indices": [[2, 3]]
    },
    "model.layers.1.self_attn": {
      "key_high_precision_indices": [[4]],
      "value_high_precision_indices": [[5]]
    }
  },
  "calibration": {
    "method": "weight_norm",
    "objective": "l2_norm",
    "num_prompts": 0,
    "max_seq_len": 0,
    "batch_size": 0,
    "num_observed_tokens": 0,
    "dtype": "bfloat16",
    "device": "cpu",
    "prompts_sha256": ""
  }
}"#;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib_v1.json");
    std::fs::write(&path, fixture.as_bytes()).unwrap();

    let calib = read_kv_calibration(&path).unwrap();
    assert_eq!(calib.version, 1);
    assert_eq!(calib.layers.len(), 2);

    for (k, layer) in &calib.layers {
        assert!(
            layer.codebook.is_none(),
            "v1 fixture layer {k}: codebook must be None (absent field → serde(default))"
        );
    }
}

/// A v1.1 file with `codebook` present must not be readable as a `codebook = None`
/// fixture — i.e. the field is not dropped. Verify by writing v1.1 and reading
/// back confirming `Some`.
#[test]
fn v11_file_codebook_survives_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_calib_v11.json");

    let mut layers = BTreeMap::new();
    layers.insert(
        "model.layers.0.self_attn".to_string(),
        LayerCalib {
            key_high_precision_indices: vec![vec![0]],
            value_high_precision_indices: vec![vec![1]],
            codebook: Some(CodebookOverride {
                value: Some(vec![
                    -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0,
                ]),
            }),
        },
    );
    let calib = KvCalibration {
        version: 1,
        recipe: "turboquant35".to_string(),
        head_size: 128,
        model_name: "test".to_string(),
        transform_version: "v1".to_string(),
        codebook_version: "custom".to_string(),
        layers,
        calibration: CalibrationMeta {
            method: "weight_norm".to_string(),
            objective: "l2_norm".to_string(),
            num_prompts: 0,
            max_seq_len: 0,
            batch_size: 0,
            num_observed_tokens: 0,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            prompts_sha256: String::new(),
        },
        head_budgets: None,
    };
    write_kv_calibration(&path, &calib).unwrap();
    let loaded = read_kv_calibration(&path).unwrap();

    let l0 = loaded.layers.get("model.layers.0.self_attn").unwrap();
    let cb = l0
        .codebook
        .as_ref()
        .expect("codebook must be Some after round-trip");
    assert_eq!(
        cb.value.as_ref().map(Vec::len),
        Some(16),
        "16-entry V codebook must survive round-trip"
    );
}
