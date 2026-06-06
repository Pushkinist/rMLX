//! Tests for `kv_calibrate` CLI command (integration-level).

use std::io::Write as _;

use rmlx_loader::read_kv_calibration;

use super::run_kv_calibrate;

// ── end-to-end synthetic fixture ──────────────────────────────────────────────

/// Full pipeline test: synthetic safetensors → run_kv_calibrate → verify JSON.
#[test]
fn run_kv_calibrate_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();

    let num_kv_heads: usize = 2;
    let head_dim: usize = 32; // outlier_k = round(32*0.5/16)*16 = 16
    let in_dim: usize = 4;
    let out_dim = num_kv_heads * head_dim;

    let config = serde_json::json!({
        "architectures": ["TestModelForCausalLM"],
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "head_dim": 32,
        "torch_dtype": "float32"
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    // Inflate dims 0..16 in head=0 so they land in top-16.
    let mut k_weight = vec![1.0_f32; out_dim * in_dim];
    for dim in 0..16_usize {
        let row_base = dim * in_dim;
        for c in 0..in_dim {
            k_weight[row_base + c] = 100.0 + dim as f32;
        }
    }
    let v_weight = k_weight.clone();

    let weight_bytes =
        |w: &Vec<f32>| -> Vec<u8> { w.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let mut tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for layer in 0..2_usize {
        tensors.push((
            format!("model.layers.{layer}.self_attn.k_proj.weight"),
            "F32",
            vec![out_dim, in_dim],
            weight_bytes(&k_weight),
        ));
        tensors.push((
            format!("model.layers.{layer}.self_attn.v_proj.weight"),
            "F32",
            vec![out_dim, in_dim],
            weight_bytes(&v_weight),
        ));
    }

    write_synthetic_safetensors(&model_dir.join("model.safetensors"), &tensors);

    let out_path = model_dir.join("kv_calib.json");
    run_kv_calibrate(model_dir, "turbo3", Some(&out_path), None, None, 16).unwrap();

    let calib = read_kv_calibration(&out_path).unwrap();

    assert_eq!(calib.version, 1);
    assert_eq!(calib.recipe, "turboquant35");
    assert_eq!(calib.head_size, 32);
    assert_eq!(calib.layers.len(), 2);

    let expected_k = 16_usize;
    for layer in calib.layers.values() {
        assert_eq!(layer.key_high_precision_indices.len(), num_kv_heads);
        assert_eq!(layer.value_high_precision_indices.len(), num_kv_heads);
        for idxs in &layer.key_high_precision_indices {
            assert_eq!(idxs.len(), expected_k);
            let mut sorted = idxs.clone();
            sorted.sort_unstable();
            assert_eq!(idxs, &sorted, "indices must be sorted ascending");
        }
    }
}

#[test]
fn run_kv_calibrate_default_output_path() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();

    let num_kv_heads: usize = 2;
    let head_dim: usize = 32;
    let in_dim: usize = 4;
    let out_dim = num_kv_heads * head_dim;

    let config = serde_json::json!({
        "architectures": ["TestModelForCausalLM"],
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "head_dim": 32,
        "torch_dtype": "float32"
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    let weight = vec![1.0_f32; out_dim * in_dim];
    let weight_bytes: Vec<u8> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();

    let tensors = vec![
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            "F32",
            vec![out_dim, in_dim],
            weight_bytes.clone(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            "F32",
            vec![out_dim, in_dim],
            weight_bytes,
        ),
    ];
    write_synthetic_safetensors(&model_dir.join("model.safetensors"), &tensors);

    // No --out path → defaults to <model>/kv_calib.json
    run_kv_calibrate(model_dir, "turbo2", None, None, None, 16).unwrap();

    let default_path = model_dir.join("kv_calib.json");
    assert!(
        default_path.exists(),
        "kv_calib.json should be written to model dir by default"
    );

    let calib = read_kv_calibration(&default_path).unwrap();
    assert_eq!(calib.recipe, "turboquant25");
}

// ── head_budget recipe CLI smoke ──────────────────────────────────────────────

/// Missing config.json → distinct preflight error (no model load attempted).
#[test]
fn run_head_budget_missing_config_err() {
    let dir = tempfile::tempdir().unwrap();
    let result = run_kv_calibrate(dir.path(), "head_budget", None, None, None, 16);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("missing config.json"),
        "unexpected error: {msg}"
    );
}

/// Out-of-range `--mass-threshold` is rejected before any model load.
#[test]
fn run_head_budget_invalid_mass_threshold_err() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();
    std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
    let result = run_kv_calibrate(model_dir, "head_budget", None, None, Some(0.0), 16);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("mass-threshold must be in"),
        "unexpected error: {msg}"
    );
}

/// Non-Qwen3 architecture in config.json → arch-gate error (no model load).
#[test]
fn run_head_budget_non_qwen3_arch_err() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();
    let cfg = serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "head_dim": 32,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec_pretty(&cfg).unwrap(),
    )
    .unwrap();
    // Need a tokenizer.json and prompts file so the only failure mode is
    // the arch-gate (which fires before tokenisation).
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
    let prompts = serde_json::json!({
        "version": 1,
        "description": "test",
        "prompts": ["hello world"]
    });
    let prompts_path = model_dir.join("calibration_default.json");
    std::fs::write(&prompts_path, serde_json::to_vec_pretty(&prompts).unwrap()).unwrap();

    let result = run_kv_calibrate(
        model_dir,
        "head_budget",
        None,
        Some(&prompts_path),
        Some(0.95),
        16,
    );
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Qwen3ForCausalLM only") || msg.contains("not yet wired"),
        "unexpected error: {msg}"
    );
}

#[test]
fn run_kv_calibrate_bad_recipe_err() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();
    std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
    let result = run_kv_calibrate(model_dir, "turbo99", None, None, None, 16);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("unknown recipe"), "unexpected error: {msg}");
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build and write a single-file safetensors.
///
/// `tensors`: list of (name, dtype_str, shape, data_bytes).
/// dtype_str is one of "F32", "F16", "BF16".
fn write_synthetic_safetensors(
    path: &std::path::Path,
    tensors: &[(String, &str, Vec<usize>, Vec<u8>)],
) {
    let mut offset: u64 = 0;
    let mut metadata_map = serde_json::Map::new();
    let mut data_sections: Vec<&[u8]> = Vec::new();

    // Sort by name for determinism.
    let mut sorted: Vec<&(String, &str, Vec<usize>, Vec<u8>)> = tensors.iter().collect();
    sorted.sort_by_key(|(name, _, _, _)| name.as_str());

    for (name, dtype_str, shape, data) in &sorted {
        let end = offset + data.len() as u64;
        metadata_map.insert(
            (*name).clone(),
            serde_json::json!({
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
        data_sections.push(data.as_slice());
    }

    let header_json = serde_json::to_vec(&metadata_map).unwrap();
    let header_len = header_json.len() as u64;

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    for section in data_sections {
        file.write_all(section).unwrap();
    }
}

// ── long-context prompts verification ────────────────────────────────────────

/// Walks up from the test's compile-time manifest dir to find
/// `<repo>/prompts/calibration_long_context.json`. Returns `None` if not
/// found (graceful skip on out-of-tree builds).
fn find_long_context_prompts() -> Option<std::path::PathBuf> {
    let mut probe = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..6 {
        let candidate = probe.join("prompts").join("calibration_long_context.json");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !probe.pop() {
            break;
        }
    }
    None
}

/// Locate the Bonsai tokenizer from `RMLX_TEST_MODEL_BONSAI` or
/// `RMLX_O_MODELS_ROOT`. Returns `None` if neither is set or the file is
/// missing — the test then skips gracefully.
fn find_bonsai_tokenizer() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("RMLX_TEST_MODEL_BONSAI") {
        let candidate = std::path::PathBuf::from(p).join("tokenizer.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Ok(root) = std::env::var("RMLX_O_MODELS_ROOT") {
        let candidate = std::path::PathBuf::from(root)
            .join("prism-ml__Ternary-Bonsai-8B-mlx-2bit")
            .join("tokenizer.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Every prompt in `calibration_long_context.json` must tokenize to >= 4096
/// tokens on Bonsai's (Qwen3 family) tokenizer.
///
/// Verifies the v2 calibration ran on true long-context inputs rather than the
/// prior short-context corpus that gave misleadingly small budget estimates.
#[test]
fn long_context_prompts_meet_4096_token_floor() {
    let Some(prompts_pb) = find_long_context_prompts() else {
        eprintln!("SKIP: calibration_long_context.json not found");
        return;
    };
    let Some(tk_pb) = find_bonsai_tokenizer() else {
        eprintln!(
            "SKIP: Bonsai tokenizer not found; set RMLX_TEST_MODEL_BONSAI or RMLX_O_MODELS_ROOT"
        );
        return;
    };

    let tokenizer = tokenizers::Tokenizer::from_file(&tk_pb).expect("load Bonsai tokenizer.json");
    let bytes = std::fs::read(&prompts_pb).expect("read prompts JSON");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse prompts JSON");
    let arr = json["prompts"].as_array().expect("prompts array");
    assert!(!arr.is_empty(), "prompts array is empty");

    let mut min_tok = usize::MAX;
    let mut max_tok = 0_usize;
    for (i, p) in arr.iter().enumerate() {
        let s = p.as_str().expect("prompt is string");
        let enc = tokenizer.encode(s, true).expect("tokenize prompt");
        let n = enc.get_ids().len();
        assert!(
            n >= 4096,
            "long-context: prompt {i} only {n} tokens; floor is 4096"
        );
        if n < min_tok {
            min_tok = n;
        }
        if n > max_tok {
            max_tok = n;
        }
    }
    eprintln!(
        "long-context prompts: n={} min={} max={} (>=4096)",
        arr.len(),
        min_tok,
        max_tok
    );
}
