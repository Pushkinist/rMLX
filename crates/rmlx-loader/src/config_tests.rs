use super::*;

const MINIMAL_JSON: &str = r#"{
    "architectures": ["TestArch"],
    "dtype": "bfloat16",
    "quantization": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine"
    },
    "text_config": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "sliding_window": null,
        "max_position_embeddings": 32768
    }
}"#;

#[test]
fn parses_minimal_config() {
    let cfg: ModelConfig = serde_json::from_str(MINIMAL_JSON).unwrap();
    assert_eq!(cfg.architectures, ["TestArch"]);
    assert_eq!(cfg.dtype.as_deref(), Some("bfloat16"));
    let q = cfg.quantization.unwrap();
    assert_eq!(q.group_size, 64);
    assert_eq!(q.bits, 4);
    assert_eq!(q.mode.as_deref(), Some("affine"));
    assert_eq!(q.mode_or_default(), "affine");
    assert!(q.tensor_overrides.is_none());
    let tc = cfg.text_config.unwrap();
    assert_eq!(tc.num_hidden_layers, Some(32));
    assert_eq!(tc.num_key_value_heads, Some(8));
    assert!(tc.sliding_window.is_none());
}

const WITH_OVERRIDES_JSON: &str = r#"{
    "architectures": ["AnotherArch"],
    "quantization": {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
        "tensor_overrides": {
            "mlp.gate": {
                "group_size": 64,
                "bits": 8,
                "mode": "mxfp8"
            }
        }
    }
}"#;

#[test]
fn parses_tensor_overrides() {
    let cfg: ModelConfig = serde_json::from_str(WITH_OVERRIDES_JSON).unwrap();
    let q = cfg.quantization.unwrap();
    assert_eq!(q.mode.as_deref(), Some("mxfp8"));
    assert_eq!(q.mode_or_default(), "mxfp8");
    let overrides = q.tensor_overrides.unwrap();
    let gate = overrides.get("mlp.gate").unwrap();
    assert_eq!(gate.group_size, 64);
}

#[test]
fn unknown_top_level_keys_round_trip() {
    let json = r#"{"architectures":[],"some_future_key":"hello"}"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(
        cfg.extras.get("some_future_key").and_then(|v| v.as_str()),
        Some("hello")
    );
}

/// Snapshots that omit `mode` (e.g. prism-ml Ternary-Bonsai) must parse
/// successfully. `mode` is `None`; `mode_or_default()` yields `"affine"`.
#[test]
fn parses_quant_without_mode() {
    let json = r#"{"architectures":[],"quantization":{"group_size":128,"bits":2}}"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    let q = cfg.quantization.unwrap();
    assert_eq!(q.group_size, 128);
    assert_eq!(q.bits, 2);
    assert!(
        q.mode.is_none(),
        "mode should be None when absent from JSON"
    );
    assert_eq!(q.mode_or_default(), "affine");
}

/// PARO checkpoint config: `quantization_config` with `quant_method = "paroquant"`.
#[test]
fn parses_paroquant_config() {
    let json = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 128,
            "krot": 8
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.quantization.is_none());
    assert!(cfg.is_paroquant());
    let qc = cfg.quantization_config.as_ref().unwrap();
    assert_eq!(qc.quant_method.as_deref(), Some("paroquant"));
    assert_eq!(qc.bits, Some(4));
    assert_eq!(qc.group_size, Some(128));
    assert_eq!(qc.krot, Some(8));
}

/// A non-PARO config with a `quantization_config` field (e.g. gemma mxfp8)
/// must parse successfully and must not trigger `is_paroquant()`.
#[test]
fn non_paro_quantization_config_parses_ok() {
    let json = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "quantization_config": {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8"
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.quantization_config.is_some());
    assert!(
        !cfg.is_paroquant(),
        "mxfp8 config must not trigger is_paroquant"
    );
}

/// Standard MLX config (no `quantization_config`) must not trigger `is_paroquant()`.
#[test]
fn non_paro_config_is_not_paroquant() {
    let cfg: ModelConfig = serde_json::from_str(MINIMAL_JSON).unwrap();
    assert!(!cfg.is_paroquant());
}

// ----- head_dim() resolver ------------------------------------------------

/// Bonsai (Qwen3): explicit `head_dim` field in `text_config`. Helper returns it.
#[test]
fn head_dim_qwen3_textconfig_explicit() {
    let json = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "text_config": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "head_dim": 128
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), Some(128));
}

/// Gemma4 with both `head_dim` (SWA) and `global_head_dim` (FA).
/// Helper MUST return the FA value (`global_head_dim`).
#[test]
fn head_dim_gemma4_prefers_global() {
    let json = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 2560,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 128
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), Some(128));
}

/// Gemma4 missing `global_head_dim`: fall through to `head_dim`.
#[test]
fn head_dim_gemma4_falls_back_to_head_dim_when_global_absent() {
    let json = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 2560,
            "num_attention_heads": 8,
            "head_dim": 256
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), Some(256));
}

/// Qwen2-style: only `hidden_size` + `num_attention_heads`, no explicit head_dim.
/// Divide-fallback returns the quotient.
#[test]
fn head_dim_qwen2_divide_fallback() {
    let json = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "text_config": {
            "hidden_size": 2048,
            "num_attention_heads": 32
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), Some(64));
}

/// `text_config = None`, no top-level head_dim or sizing fields → `None`.
#[test]
fn head_dim_no_text_config_no_top_level_returns_none() {
    let json = r#"{
        "architectures": ["MysteryArch"]
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.text_config.is_none());
    assert_eq!(cfg.head_dim(), None);
}

/// `text_config` present but no head_dim, no resolvable hidden/heads.
#[test]
fn head_dim_textconfig_without_resolvable_fields_returns_none() {
    let json = r#"{
        "architectures": ["MysteryArch"],
        "text_config": {
            "num_key_value_heads": 8
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), None);
}

/// Bonsai-shaped real config: no `text_config`, all fields at the root.
/// Helper must read top-level `head_dim`.
#[test]
fn head_dim_top_level_qwen3_no_text_config() {
    let json = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "head_dim": 128
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.text_config.is_none());
    assert_eq!(cfg.head_dim(), Some(128));
}

/// Top-level divide-fallback when there's no `text_config` and no explicit head_dim.
#[test]
fn head_dim_top_level_divide_fallback() {
    let json = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 64
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), Some(64));
}

// tensor_overrides at MAX_OVERRIDES limit must pass; over limit must fail.
#[test]
fn load_config_rejects_oversized_tensor_overrides() {
    use std::io::Write;
    // Build a config.json with MAX_OVERRIDES + 1 tensor_overrides entries.
    let n = MAX_OVERRIDES + 1;
    let overrides: Map<String, serde_json::Value> = (0..n)
        .map(|i| {
            (
                format!("tensor_{i}"),
                serde_json::json!({"group_size": 64, "bits": 4, "mode": "affine"}),
            )
        })
        .collect();
    let json = serde_json::json!({
        "architectures": ["TestArch"],
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "tensor_overrides": overrides
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(json.to_string().as_bytes())
        .unwrap();
    let result = load_config(dir.path());
    assert!(result.is_err(), "expected Err for {n} tensor_overrides");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("MAX_OVERRIDES") || msg.contains(&MAX_OVERRIDES.to_string()),
        "error message should mention the limit: {msg}"
    );
}

#[test]
fn load_config_accepts_normal_tensor_overrides() {
    use std::io::Write;
    let json = serde_json::json!({
        "architectures": ["TestArch"],
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "tensor_overrides": {
                "mlp.gate": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
                "mlp.down": {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            }
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(json.to_string().as_bytes())
        .unwrap();
    let result = load_config(dir.path());
    assert!(
        result.is_ok(),
        "expected Ok for 2 tensor_overrides: {:?}",
        result.err()
    );
    let overrides = result
        .unwrap()
        .quantization
        .unwrap()
        .tensor_overrides
        .unwrap();
    assert_eq!(overrides.len(), 2);
}

/// Divide-fallback rejects inexact division (no guessing).
#[test]
fn head_dim_divide_rejects_inexact() {
    let json = r#"{
        "architectures": ["MysteryArch"],
        "text_config": {
            "hidden_size": 100,
            "num_attention_heads": 7
        }
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.head_dim(), None);
}

// ----- eos_token_ids() ---------------------------------------------------

/// scalar `eos_token_id` (Llama-2, Bonsai) — must return a single-element Vec.
#[test]
fn eos_scalar() {
    let json = r#"{"architectures":["LlamaForCausalLM"],"eos_token_id":2}"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.eos_token_ids(), vec![2u32]);
}

/// array `eos_token_id` (Llama-3 / SmolLM-3) — must return all ids in order.
#[test]
fn eos_array() {
    let json = r#"{"architectures":["LlamaForCausalLM"],"eos_token_id":[128001,128008,128009]}"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.eos_token_ids(), vec![128001u32, 128008, 128009]);
}

/// missing `eos_token_id` — must return an empty Vec (caller treats as
/// "no EOS stop, run to max_tokens").
#[test]
fn eos_missing() {
    let json = r#"{"architectures":["SomeArch"]}"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(
        cfg.eos_token_ids().is_empty(),
        "absent eos_token_id must yield empty Vec"
    );
}

/// explicit `null` top-level eos (Qwen3-VL-MoE style) falls back to
/// `text_config.eos_token_id` when that value is a scalar.
#[test]
fn eos_null_top_level_falls_back_to_text_config_scalar() {
    let json = r#"{
        "architectures": ["Qwen3VLMoeForConditionalGeneration"],
        "eos_token_id": null,
        "text_config": {"eos_token_id": 151645}
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.eos_token_ids(), vec![151645u32]);
}

/// explicit `null` top-level eos falls back to an array in `text_config`.
#[test]
fn eos_null_top_level_falls_back_to_text_config_array() {
    let json = r#"{
        "architectures": ["SomeVLMArch"],
        "eos_token_id": null,
        "text_config": {"eos_token_id": [248046, 248044]}
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.eos_token_ids(), vec![248046u32, 248044]);
}

/// both top-level and text_config `eos_token_id` are null → empty Vec.
#[test]
fn eos_null_top_and_null_text_config_returns_empty() {
    let json = r#"{
        "architectures":["X"],
        "eos_token_id":null,
        "text_config":{"eos_token_id":null}
    }"#;
    let cfg: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.eos_token_ids().is_empty());
}
