//! Tests for `calibration` (algorithm).

use std::collections::BTreeMap;
use std::io::Write as _;

use super::{
    bytes_to_f32, calibrate_model, detect_kv_weight_pattern, effective_quant, f16_to_f32,
    f32_to_bf16_le, layer_key_from_pattern, scales_sibling_name,
};
use crate::config::QuantConfig;

// ── f16_to_f32 ────────────────────────────────────────────────────────────────

#[test]
fn f16_zero_pos() {
    assert_eq!(f16_to_f32(0x0000), 0.0_f32);
}

#[test]
fn f16_zero_neg() {
    let v = f16_to_f32(0x8000);
    assert!(v == -0.0_f32 || v == 0.0_f32);
}

#[test]
fn f16_one() {
    let v = f16_to_f32(0x3C00);
    assert!((v - 1.0_f32).abs() < 1e-5);
}

#[test]
fn f16_minus_two() {
    let v = f16_to_f32(0xC000);
    assert!((v - (-2.0_f32)).abs() < 1e-5);
}

#[test]
fn f16_inf_pos() {
    let v = f16_to_f32(0x7C00);
    assert!(v.is_infinite() && v > 0.0);
}

#[test]
fn f16_inf_neg() {
    let v = f16_to_f32(0xFC00);
    assert!(v.is_infinite() && v < 0.0);
}

// ── bytes_to_f32 ──────────────────────────────────────────────────────────────

#[test]
fn f32_roundtrip() {
    let vals: [f32; 3] = [1.5, -2.0, 0.0];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = bytes_to_f32(&bytes, safetensors::Dtype::F32).unwrap();
    for (a, b) in vals.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn bf16_one() {
    // 1.0 in BF16 = 0x3F80
    let bytes = 0x3F80_u16.to_le_bytes();
    let out = bytes_to_f32(&bytes, safetensors::Dtype::BF16).unwrap();
    assert!((out[0] - 1.0_f32).abs() < 1e-3);
}

#[test]
fn f16_one_bytes() {
    let bytes = 0x3C00_u16.to_le_bytes();
    let out = bytes_to_f32(&bytes, safetensors::Dtype::F16).unwrap();
    assert!((out[0] - 1.0_f32).abs() < 1e-3);
}

#[test]
fn unsupported_dtype_err() {
    let bytes = [0u8; 4];
    assert!(bytes_to_f32(&bytes, safetensors::Dtype::I32).is_err());
}

// ── detect_kv_weight_pattern ──────────────────────────────────────────────────

#[test]
fn detect_self_attn() {
    let mut wm = BTreeMap::new();
    wm.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "s.safetensors".to_string(),
    );
    let p = detect_kv_weight_pattern(&wm, "k_proj").unwrap();
    assert_eq!(p, "model.layers.{}.self_attn.k_proj.weight");
}

#[test]
fn detect_transformer_h() {
    let mut wm = BTreeMap::new();
    wm.insert(
        "transformer.h.0.attn.k_proj.weight".to_string(),
        "s.safetensors".to_string(),
    );
    let p = detect_kv_weight_pattern(&wm, "k_proj").unwrap();
    assert_eq!(p, "transformer.h.{}.attn.k_proj.weight");
}

#[test]
fn detect_attention_prefix() {
    let mut wm = BTreeMap::new();
    wm.insert(
        "model.layers.0.attention.k_proj.weight".to_string(),
        "s.safetensors".to_string(),
    );
    let p = detect_kv_weight_pattern(&wm, "k_proj").unwrap();
    assert_eq!(p, "model.layers.{}.attention.k_proj.weight");
}

#[test]
fn detect_none_when_absent() {
    let wm: BTreeMap<String, String> = BTreeMap::new();
    assert!(detect_kv_weight_pattern(&wm, "k_proj").is_none());
}

#[test]
fn detect_language_model_prefix() {
    let mut wm = BTreeMap::new();
    wm.insert(
        "language_model.model.layers.0.self_attn.k_proj.weight".to_string(),
        "s.safetensors".to_string(),
    );
    let p = detect_kv_weight_pattern(&wm, "k_proj").unwrap();
    assert_eq!(p, "language_model.model.layers.{}.self_attn.k_proj.weight");
}

#[test]
fn detect_skips_audio_tower_scalar() {
    // The audio tower has a same-named `k_proj.input_max` scalar at layer 0,
    // plus a real text k_proj weight at layer 3 (sparse-attention layout).
    let mut wm = BTreeMap::new();
    wm.insert(
        "audio_tower.layers.0.self_attn.k_proj.input_max".to_string(),
        "s.safetensors".to_string(),
    );
    wm.insert(
        "language_model.model.layers.3.self_attn.k_proj.weight".to_string(),
        "s.safetensors".to_string(),
    );
    let p = detect_kv_weight_pattern(&wm, "k_proj").unwrap();
    assert_eq!(p, "language_model.model.layers.{}.self_attn.k_proj.weight");
}

// ── layer_key_from_pattern ────────────────────────────────────────────────────

#[test]
fn layer_key_self_attn() {
    let k = layer_key_from_pattern("model.layers.{}.self_attn.k_proj.weight", 3);
    assert_eq!(k, "model.layers.3.self_attn");
}

#[test]
fn layer_key_attn() {
    let k = layer_key_from_pattern("transformer.h.{}.attn.k_proj.weight", 0);
    assert_eq!(k, "transformer.h.0.attn");
}

#[test]
fn layer_key_attention() {
    let k = layer_key_from_pattern("model.layers.{}.attention.k_proj.weight", 7);
    assert_eq!(k, "model.layers.7.attention");
}

// ── calibrate_model end-to-end ────────────────────────────────────────────────

/// Build a minimal single-shard safetensors, run calibrate_model, verify output.
#[test]
fn calibrate_model_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();

    let num_kv_heads: usize = 2;
    let head_dim: usize = 32; // valid: outlier_k = round(32*0.5/16)*16 = 16
    let in_dim: usize = 4;
    let out_dim = num_kv_heads * head_dim; // 64

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

    // Give first 16 dims in head=0 large norms to ensure they rank top-16.
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

    let calib = calibrate_model(model_dir, "turbo3", "turboquant35").unwrap();

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

// ── scales_sibling_name ───────────────────────────────────────────────────────

#[test]
fn scales_sibling_for_weight() {
    let s = scales_sibling_name("model.layers.0.self_attn.k_proj.weight");
    assert_eq!(s.as_deref(), Some("model.layers.0.self_attn.k_proj.scales"));
}

#[test]
fn scales_sibling_none_for_non_weight() {
    assert!(scales_sibling_name("model.layers.0.self_attn.k_proj.scales").is_none());
}

// ── f32_to_bf16_le ────────────────────────────────────────────────────────────

#[test]
fn bf16_le_one() {
    // 1.0 f32 = 0x3F800000 → bf16 0x3F80, LE bytes [0x80, 0x3F].
    assert_eq!(f32_to_bf16_le(1.0), [0x80, 0x3F]);
}

#[test]
fn bf16_le_roundtrip_close() {
    for v in [0.5_f32, -2.25, 3.5, 100.0, -0.001] {
        let le = f32_to_bf16_le(v);
        let back = bytes_to_f32(&le, safetensors::Dtype::BF16).unwrap()[0];
        // bf16 has 8 bits of relative precision; allow ~1% error.
        let rel = ((back - v) / v).abs();
        assert!(rel < 0.01, "v={v} back={back} rel={rel}");
    }
}

// ── calibrate_model on an affine-quantized snapshot ───────────────────────────

/// Build a synthetic 8-bit affine snapshot (U32-packed weight + BF16 scales +
/// BF16 biases) and verify `calibrate_model` dequantizes and produces
/// non-degenerate per-head index lists of the right length.
#[test]
fn calibrate_model_affine_quantized() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path();

    let num_kv_heads: usize = 2;
    let head_dim: usize = 32; // outlier_k = 16
    let in_dim: usize = 32; // == group_size → 1 group per row
    let out_dim = num_kv_heads * head_dim; // 64
    let bits: u8 = 8;

    let config = serde_json::json!({
        "architectures": ["TestModelForCausalLM"],
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "head_dim": 32,
        "torch_dtype": "float16",
        "quantization": { "group_size": 32, "bits": 8, "mode": "affine" }
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    // Codes: give the first 16 dims of each head large magnitude → large L2 norm.
    // scale=1.0, bias=0.0 → dequant value == code.
    let mut codes = vec![1_u32; out_dim * in_dim];
    for head in 0..num_kv_heads {
        for dim in 0..16_usize {
            let row = head * head_dim + dim;
            for c in 0..in_dim {
                codes[row * in_dim + c] = 200; // large 8-bit code
            }
        }
    }

    // Pack 8-bit codes LSB-first into u32 LE words (4 codes per word, U32Le).
    let per_word = 32 / bits as usize; // 4
    let words_per_row = in_dim / per_word; // 8
    let mut packed = Vec::with_capacity(out_dim * words_per_row * 4);
    for row in 0..out_dim {
        for w in 0..words_per_row {
            let mut word: u32 = 0;
            for slot in 0..per_word {
                let code = codes[row * in_dim + w * per_word + slot] & 0xFF;
                word |= code << (slot * bits as usize);
            }
            packed.extend_from_slice(&word.to_le_bytes());
        }
    }

    // scales = 1.0, biases = 0.0, one group per row → out_dim values each.
    let scales_bytes: Vec<u8> = (0..out_dim).flat_map(|_| f32_to_bf16_le(1.0)).collect();
    let biases_bytes: Vec<u8> = (0..out_dim).flat_map(|_| f32_to_bf16_le(0.0)).collect();

    let mut tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for proj in ["k_proj", "v_proj"] {
        let base = format!("model.layers.0.self_attn.{proj}");
        tensors.push((
            format!("{base}.weight"),
            "U32",
            vec![out_dim, in_dim / (32 / bits as usize)],
            packed.clone(),
        ));
        tensors.push((
            format!("{base}.scales"),
            "BF16",
            vec![out_dim, 1],
            scales_bytes.clone(),
        ));
        tensors.push((
            format!("{base}.biases"),
            "BF16",
            vec![out_dim, 1],
            biases_bytes.clone(),
        ));
    }

    write_synthetic_safetensors(&model_dir.join("model.safetensors"), &tensors);

    let calib = calibrate_model(model_dir, "turbo3", "turboquant35").unwrap();

    assert_eq!(calib.layers.len(), 1);
    let layer = calib.layers.values().next().unwrap();
    assert_eq!(layer.key_high_precision_indices.len(), num_kv_heads);
    for idxs in &layer.key_high_precision_indices {
        assert_eq!(idxs.len(), 16, "outlier_k=16 per head");
        // Non-degenerate: the top-16 are exactly the large-magnitude dims 0..16.
        let mut sorted = idxs.clone();
        sorted.sort_unstable();
        assert_eq!(idxs, &sorted, "indices sorted ascending");
        assert_eq!(
            *idxs,
            (0..16_u32).collect::<Vec<u32>>(),
            "high-norm dims 0..16 must rank top — proves dequant ranking is on float values"
        );
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build and write a minimal single-file safetensors.
///
/// `tensors`: list of (name, dtype_str, shape, data_bytes).
fn write_synthetic_safetensors(
    path: &std::path::Path,
    tensors: &[(String, &str, Vec<usize>, Vec<u8>)],
) {
    let mut offset: u64 = 0;
    let mut metadata_map = serde_json::Map::new();
    let mut data_sections: Vec<&[u8]> = Vec::new();

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

// ── effective_quant (per-tensor override resolution) ─────────────────────────

fn qc(bits: u8, group_size: u32, mode: &str) -> QuantConfig {
    QuantConfig {
        group_size,
        bits,
        mode: Some(mode.to_string()),
        tensor_overrides: None,
    }
}

#[test]
fn effective_quant_no_overrides_returns_toplevel() {
    let top = qc(8, 64, "affine");
    let got = effective_quant(&top, "model.layers.0.self_attn.k_proj.weight");
    assert_eq!(got.bits, 8);
    assert_eq!(got.group_size, 64);
}

#[test]
fn effective_quant_longest_prefix_override_wins() {
    let mut overrides = std::collections::HashMap::new();
    // A broad prefix and a more-specific one; the specific must win.
    overrides.insert("model.layers.0".to_string(), qc(4, 32, "affine"));
    overrides.insert(
        "model.layers.0.self_attn.k_proj".to_string(),
        qc(2, 16, "affine"),
    );
    let mut top = qc(8, 64, "affine");
    top.tensor_overrides = Some(overrides);

    let k = effective_quant(&top, "model.layers.0.self_attn.k_proj.weight");
    assert_eq!(
        (k.bits, k.group_size),
        (2, 16),
        "longest-prefix override wins"
    );

    // A tensor matched only by the broad prefix gets that one.
    let mlp = effective_quant(&top, "model.layers.0.mlp.gate_proj.weight");
    assert_eq!((mlp.bits, mlp.group_size), (4, 32));

    // A tensor matched by no override falls back to top-level.
    let other = effective_quant(&top, "model.layers.9.self_attn.k_proj.weight");
    assert_eq!((other.bits, other.group_size), (8, 64));
}
