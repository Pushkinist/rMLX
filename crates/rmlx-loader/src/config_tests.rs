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
    assert_eq!(cfg.eos_token_ids(), Vec::<u32>::new());
}

// ----- stop_token_ids() --------------------------------------------------

/// A snapshot directory holding `config` as `config.json` and, when given,
/// `generation` as `generation_config.json`.
fn snapshot_files(config: &str, generation: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    if let Some(generation) = generation {
        std::fs::write(dir.path().join("generation_config.json"), generation).unwrap();
    }
    dir
}

/// The stop set is the `config.json` ids in their order, then each
/// `generation_config.json` id the config does not name.
#[test]
fn the_stop_set_is_the_config_ids_then_the_generation_config_ids() {
    let cases: [(&str, &str, Option<&str>, &[u32]); 12] = [
        (
            "config scalar, no generation_config",
            r#"{"eos_token_id": 2}"#,
            None,
            &[2],
        ),
        (
            "the same array in both files keeps its order",
            r#"{"eos_token_id": [1, 106, 50]}"#,
            Some(r#"{"eos_token_id": [1, 106, 50]}"#),
            &[1, 106, 50],
        ),
        (
            "nested config id, end of turn only in generation_config",
            r#"{"text_config": {"eos_token_id": 248044}}"#,
            Some(r#"{"bos_token_id": 248044, "eos_token_id": 248046}"#),
            &[248044, 248046],
        ),
        (
            "generation_config order does not reorder the config ids",
            r#"{"eos_token_id": [248044]}"#,
            Some(r#"{"eos_token_id": [248046, 248044]}"#),
            &[248044, 248046],
        ),
        (
            "both ids in both files, config order wins",
            r#"{"eos_token_id": [248044, 248046]}"#,
            Some(r#"{"eos_token_id": [248046, 248044]}"#),
            &[248044, 248046],
        ),
        (
            "null top level falls back to text_config",
            r#"{"eos_token_id": null, "text_config": {"eos_token_id": 151645}}"#,
            Some(r#"{"eos_token_id": [151645, 151643]}"#),
            &[151645, 151643],
        ),
        (
            "generation_config names fewer ids than the config",
            r#"{"eos_token_id": [1, 106]}"#,
            Some(r#"{"eos_token_id": [1]}"#),
            &[1, 106],
        ),
        ("no config id", r"{}", Some(r#"{"eos_token_id": 5}"#), &[5]),
        (
            "generation_config without an end id",
            r#"{"eos_token_id": 7}"#,
            Some(r#"{"temperature": 0.6}"#),
            &[7],
        ),
        (
            "null generation_config id",
            r#"{"eos_token_id": 7}"#,
            Some(r#"{"eos_token_id": null}"#),
            &[7],
        ),
        (
            "an entry that is not a token id is skipped in either file",
            r#"{"eos_token_id": [7, "x"]}"#,
            Some(r#"{"eos_token_id": [-1, 8, 4294967296]}"#),
            &[7, 8],
        ),
        ("neither file names one", r"{}", None, &[]),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|(what, config, generation, want)| {
            let dir = snapshot_files(config, *generation);
            let got = stop_token_ids(dir.path());
            (got.as_deref().ok() != Some(*want))
                .then(|| format!("{what}: want {want:?}, got {got:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A `generation_config.json` that does not parse is an error that names the
/// file, not a stop set that silently lacks its ids. A key of the wrong type
/// does not parse either, also a key that only the speech route reads.
#[test]
fn a_generation_config_that_does_not_parse_is_an_error_naming_it() {
    for generation in ["{ not json", r#"{"do_sample": 1}"#] {
        let dir = snapshot_files(r#"{"eos_token_id": 2}"#, Some(generation));
        let err = stop_token_ids(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("generation_config.json"),
            "{generation}: {err}"
        );
    }
}

/// The sampling keys and the end ids of `generation_config.json`; an absent
/// key is `None`, an unknown key is ignored, an absent file is `Ok(None)`.
#[test]
fn generation_config_reads_its_keys_and_ignores_the_rest() {
    let none = snapshot_files("{}", None);
    assert!(load_generation_config(none.path()).unwrap().is_none());

    let some = snapshot_files(
        "{}",
        Some(
            r#"{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.1,
                "max_new_tokens": 512, "eos_token_id": [1, 106], "bos_token_id": 2,
                "do_sample": false, "subtalker_dosample": true,
                "subtalker_temperature": 0.8, "subtalker_top_p": 0.9, "subtalker_top_k": 40}"#,
        ),
    );
    let cfg = load_generation_config(some.path()).unwrap().unwrap();
    assert_eq!(cfg.temperature, Some(1.0));
    assert_eq!(cfg.top_p, Some(0.95));
    assert_eq!(cfg.top_k, Some(20));
    assert_eq!(cfg.repetition_penalty, Some(1.1));
    assert_eq!(cfg.max_new_tokens, Some(512));
    assert_eq!(cfg.eos_token_ids, [1, 106]);
    assert_eq!(cfg.do_sample, Some(false));
    assert_eq!(cfg.subtalker_dosample, Some(true));
    assert_eq!(cfg.subtalker_temperature, Some(0.8));
    assert_eq!(cfg.subtalker_top_p, Some(0.9));
    assert_eq!(cfg.subtalker_top_k, Some(40));

    let partial = snapshot_files("{}", Some(r#"{"temperature": 0.7}"#));
    let cfg = load_generation_config(partial.path()).unwrap().unwrap();
    assert_eq!(cfg.temperature, Some(0.7));
    assert!(cfg.top_p.is_none() && cfg.top_k.is_none() && cfg.eos_token_ids.is_empty());
    assert!(cfg.do_sample.is_none() && cfg.subtalker_dosample.is_none());
    assert!(cfg.subtalker_temperature.is_none() && cfg.subtalker_top_p.is_none());
    assert!(cfg.subtalker_top_k.is_none());
}

/// A `generation_config.json` that exists and cannot be read (here a
/// directory) is an error that names it. Only an absent file names no ids.
#[test]
fn an_unreadable_generation_config_is_an_error_naming_it() {
    let dir = snapshot_files(r#"{"eos_token_id": 2}"#, None);
    std::fs::create_dir(dir.path().join("generation_config.json")).unwrap();
    let err = stop_token_ids(dir.path()).unwrap_err().to_string();
    assert!(err.contains("generation_config.json"), "{err}");
}

/// Every snapshot under `RMLX_O_MODELS_ROOT` stops at the ids its own files
/// declare, read here from the raw JSON: the `config.json` ids first and in
/// their order (the stop set rMLX served before it read
/// `generation_config.json`), then each `generation_config.json` id they lack.
#[test]
fn every_local_snapshot_stops_at_its_config_ids_then_its_generation_config_ids() {
    const TEST: &str =
        "every_local_snapshot_stops_at_its_config_ids_then_its_generation_config_ids";
    let Some(root) = std::env::var("RMLX_O_MODELS_ROOT")
        .ok()
        .filter(|r| !r.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|r| r.is_dir())
    else {
        println!("SKIP {TEST}: RMLX_O_MODELS_ROOT names no directory");
        return;
    };
    let raw = |path: std::path::PathBuf| -> Option<serde_json::Value> {
        Some(serde_json::from_slice(&std::fs::read(path).ok()?).unwrap())
    };
    let id = |v: &serde_json::Value| v.as_u64().and_then(|u| u32::try_from(u).ok());
    let ids = |v: Option<&serde_json::Value>| -> Vec<u32> {
        match v {
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(id).collect(),
            Some(v) => id(v).into_iter().collect(),
            None => Vec::new(),
        }
    };
    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|dir| dir.join("config.json").is_file())
        .collect();
    dirs.sort();
    let mut wrong = Vec::new();
    let mut widened = Vec::new();
    for dir in &dirs {
        let config = raw(dir.join("config.json")).unwrap();
        let top = config.get("eos_token_id").filter(|v| !v.is_null());
        let mut want = ids(top.or_else(|| config.pointer("/text_config/eos_token_id")));
        let served_before = want.len();
        let generation = raw(dir.join("generation_config.json"));
        for id in ids(generation.as_ref().and_then(|g| g.get("eos_token_id"))) {
            if !want.contains(&id) {
                want.push(id);
            }
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        if want.len() > served_before {
            widened.push(format!("{name} {:?} -> {want:?}", &want[..served_before]));
        }
        match stop_token_ids(dir) {
            Ok(got) if got == want => {}
            got => wrong.push(format!("{name}: want {want:?}, got {got:?}")),
        }
    }
    println!(
        "{TEST}: {} snapshot configs, {} widened by generation_config.json:\n  {}",
        dirs.len(),
        widened.len(),
        widened.join("\n  ")
    );
    assert!(
        !dirs.is_empty(),
        "{} holds no snapshot config",
        root.display()
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The class each config dispatches on. `architectures[0]` wins whenever the
/// config names one. With none, a Prism Hadamard pack (`hadamard_config`)
/// takes the class of its `base_model_type`; any other config has none.
#[test]
fn dispatch_arch_is_the_declared_class_or_the_backbone_of_a_hadamard_pack() {
    let cases = [
        (
            "declared",
            r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
            Some("Qwen3ForCausalLM"),
        ),
        (
            "declared, beside a Hadamard contract",
            r#"{"architectures": ["Qwen3ForCausalLM"], "hadamard_config": "hadamard.json",
                "base_model_type": "qwen3_5"}"#,
            Some("Qwen3ForCausalLM"),
        ),
        (
            "an empty declared class stays empty",
            r#"{"architectures": [""], "hadamard_config": "hadamard.json",
                "base_model_type": "qwen3_5"}"#,
            Some(""),
        ),
        (
            "Hadamard pack on a Qwen3.5 backbone",
            r#"{"model_type": "prism_hadamard_qwen35", "hadamard_config": "hadamard.json",
                "base_model_type": "qwen3_5"}"#,
            Some("Qwen3_5ForConditionalGeneration"),
        ),
        (
            "Hadamard pack, empty architectures array",
            r#"{"architectures": [], "hadamard_config": "hadamard.json",
                "base_model_type": "qwen3_5"}"#,
            Some("Qwen3_5ForConditionalGeneration"),
        ),
        (
            "Hadamard pack on a backbone with no class",
            r#"{"hadamard_config": "hadamard.json", "base_model_type": "llama"}"#,
            None,
        ),
        (
            "Hadamard pack on a Qwen3.5 MoE backbone, which has no class here",
            r#"{"hadamard_config": "hadamard.json", "base_model_type": "qwen3_5_moe"}"#,
            None,
        ),
        (
            "Hadamard pack with no base_model_type",
            r#"{"hadamard_config": "hadamard.json"}"#,
            None,
        ),
        (
            "Hadamard pack with a base_model_type that is not a string",
            r#"{"hadamard_config": "hadamard.json", "base_model_type": 5}"#,
            None,
        ),
        (
            "a base_model_type without a Hadamard contract",
            r#"{"model_type": "qwen3_5", "base_model_type": "qwen3_5"}"#,
            None,
        ),
        ("an MTP sidecar", r#"{"model_type": "qwen3_5_mtp"}"#, None),
        ("nothing", r"{}", None),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|(what, json, want)| {
            let cfg: ModelConfig = serde_json::from_str(json).unwrap();
            let got = cfg.dispatch_arch();
            (got != *want).then(|| format!("{what}: want {want:?}, got {got:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Every snapshot config under `RMLX_O_MODELS_ROOT` dispatches on the class
/// its own file names. The expected class is read from the raw JSON, not
/// through `ModelConfig`: `architectures[0]`, or, for a Prism Hadamard pack,
/// the one class that every other snapshot whose `model_type` is the pack's
/// `base_model_type` declares. A config with neither has no class.
#[test]
fn every_local_snapshot_dispatches_on_the_class_its_config_names() {
    const TEST: &str = "every_local_snapshot_dispatches_on_the_class_its_config_names";
    let Some(root) = std::env::var("RMLX_O_MODELS_ROOT")
        .ok()
        .filter(|r| !r.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|r| r.is_dir())
    else {
        println!("SKIP {TEST}: RMLX_O_MODELS_ROOT names no directory");
        return;
    };
    let mut raws: Vec<(std::path::PathBuf, serde_json::Value)> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|dir| dir.join("config.json").is_file())
        .map(|dir| {
            let bytes = std::fs::read(dir.join("config.json")).unwrap();
            (dir, serde_json::from_slice(&bytes).unwrap())
        })
        .collect();
    raws.sort_by(|a, b| a.0.cmp(&b.0));
    let declared = |raw: &serde_json::Value| {
        raw.pointer("/architectures/0")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    let class_of_model_type = |model_type: &str| {
        raws.iter()
            .filter(|(_, raw)| raw.get("model_type").and_then(|m| m.as_str()) == Some(model_type))
            .filter_map(|(_, raw)| declared(raw))
            .collect::<std::collections::BTreeSet<_>>()
    };

    let mut wrong = Vec::new();
    let mut packs = 0;
    for (dir, raw) in &raws {
        let want = match declared(raw) {
            Some(class) => Some(class),
            None if raw.get("hadamard_config").is_some() => {
                packs += 1;
                let base = raw.get("base_model_type").and_then(|b| b.as_str());
                let classes = base.map(class_of_model_type).unwrap_or_default();
                if classes.len() != 1 {
                    println!(
                        "SKIP {TEST} for {}: the snapshots of base_model_type {base:?} \
                         declare {classes:?}, not one class",
                        dir.display()
                    );
                    continue;
                }
                classes.into_iter().next()
            }
            None => None,
        };
        match load_config(dir) {
            Ok(cfg) if cfg.dispatch_arch() == want.as_deref() => {}
            Ok(cfg) => wrong.push(format!(
                "{}: want {want:?}, got {:?}",
                dir.display(),
                cfg.dispatch_arch()
            )),
            Err(e) => wrong.push(format!("{}: load_config: {e}", dir.display())),
        }
    }
    println!(
        "{TEST}: {} snapshot configs, {packs} Hadamard pack(s)",
        raws.len()
    );
    assert!(
        !raws.is_empty(),
        "{} holds no snapshot config",
        root.display()
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The one table both label producers are held to: this one, and
/// `weight_quant` in `scripts/lib/snapshot_identity.py`, which
/// `scripts/run_class_selftest.sh` runs over the same file.
#[test]
fn weight_quant_label_matches_the_shared_fixture_table() {
    let table: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../tests/fixtures/weight_quant_labels.json")).unwrap();
    assert!(table.len() >= 15, "the table lost cases: {}", table.len());
    for case in &table {
        let cfg: ModelConfig = serde_json::from_value(case["config"].clone()).unwrap();
        let want = case["label"].as_str().map(str::to_owned);
        assert_eq!(cfg.weight_quant_label(), want, "{}", case["name"]);
    }
}
