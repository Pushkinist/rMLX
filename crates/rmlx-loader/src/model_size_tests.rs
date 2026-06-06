//! Tests for `rmlx_loader::model_size`.

use serde_json::json;

use super::estimate_params_billions;
use crate::config::ModelConfig;

/// Build a minimal ModelConfig from a JSON value for testing.
fn config_from_json(v: serde_json::Value) -> ModelConfig {
    serde_json::from_value(v).expect("test config must deserialize")
}

// ── Flat-layout (Qwen3/Bonsai-style): hidden_size + num_hidden_layers at root ──

#[test]
fn flat_layout_7b_estimate() {
    // Qwen3-7B: hidden_size=4096, num_hidden_layers=32
    // Expected: 4096² * 32 * 12 / 1e9 ≈ 6.44 B (heuristic)
    let cfg = config_from_json(json!({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    }));
    let b = estimate_params_billions(&cfg).expect("must estimate for flat-layout config");
    // Allow generous ±2x on the heuristic — it's intentionally rough.
    assert!(b > 2.0, "7B flat: got {b} B");
    assert!(b < 15.0, "7B flat: got {b} B — suspiciously large");
}

#[test]
fn flat_layout_72b_estimate() {
    // Qwen3-72B: hidden_size=8192, num_hidden_layers=80
    // Expected: 8192² * 80 * 12 / 1e9 ≈ 64.4 B (heuristic)
    let cfg = config_from_json(json!({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 8192,
        "num_hidden_layers": 80,
    }));
    let b = estimate_params_billions(&cfg).expect("must estimate for 72B flat config");
    // Real 72B is ~72B params; heuristic is order-of-magnitude.
    assert!(b > 20.0, "72B flat: got {b} B");
    assert!(b < 200.0, "72B flat: got {b} B — suspiciously large");
}

// ── Nested-layout (Gemma4 / multimodal): hidden_size + num_hidden_layers in text_config ──

#[test]
fn nested_layout_gemma4_estimate() {
    // Gemma4-27B: text_config.hidden_size=5120, text_config.num_hidden_layers=62
    // Expected: 5120² * 62 * 12 / 1e9 ≈ 19.3 B (heuristic)
    let cfg = config_from_json(json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 62,
        }
    }));
    let b = estimate_params_billions(&cfg).expect("must estimate for nested Gemma4 config");
    assert!(b > 5.0, "Gemma4 nested: got {b} B");
    assert!(b < 60.0, "Gemma4 nested: got {b} B — suspiciously large");
}

// ── Missing fields → None ────────────────────────────────────────────────────

#[test]
fn missing_hidden_size_returns_none() {
    let cfg = config_from_json(json!({
        "architectures": ["SomeModel"],
        "num_hidden_layers": 32,
    }));
    assert!(
        estimate_params_billions(&cfg).is_none(),
        "missing hidden_size must return None"
    );
}

#[test]
fn missing_num_layers_returns_none() {
    let cfg = config_from_json(json!({
        "architectures": ["SomeModel"],
        "hidden_size": 4096,
    }));
    assert!(
        estimate_params_billions(&cfg).is_none(),
        "missing num_hidden_layers must return None"
    );
}

#[test]
fn fully_missing_fields_returns_none() {
    let cfg = config_from_json(json!({
        "architectures": ["SomeModel"],
    }));
    assert!(
        estimate_params_billions(&cfg).is_none(),
        "no arch fields must return None"
    );
}

// ── Nested path preferred over flat when both present ───────────────────────

#[test]
fn nested_wins_over_flat_when_both_present() {
    // text_config fields (nested) should be preferred per resolution order.
    let cfg = config_from_json(json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "hidden_size": 1024,        // flat — should be ignored
        "num_hidden_layers": 10,    // flat — should be ignored
        "text_config": {
            "hidden_size": 5120,    // nested — used
            "num_hidden_layers": 62,
        }
    }));
    let b = estimate_params_billions(&cfg).expect("must estimate");
    // 5120² * 62 * 12 / 1e9 ≈ 19.3 B (not 1024² * 10 * 12 / 1e9 ≈ 0.126 B)
    assert!(b > 5.0, "nested must win: got {b} B");
}
