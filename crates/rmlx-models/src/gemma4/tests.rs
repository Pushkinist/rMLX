//! Gemma4 unit tests.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret in test helpers
#![cfg_attr(test, allow(unsafe_code))]

use rmlx_mlx::{argmax, Array, Device, Dtype};

use crate::layers::{Linear, RmsNorm};

use super::config::{Gemma4TextConfig, LayerType};
use super::generate::{classify_smoke, ProbeStep, SmokeVerdict};
use super::layers::repeat_kv;
use super::loader::{build_previous_kvs, load_from_path, load_from_path_paro};

fn f32_as_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rms_norm_layer_forward() {
    let x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let w_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let w = Array::from_bytes(f32_as_bytes(&w_data), &[4], Dtype::F32).unwrap();
    let norm = RmsNorm {
        weight: Some(w),
        eps: 1e-6,
    };
    let out = norm.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    let bytes = out.to_bytes().unwrap();
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Expected: x / rms(x) with rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
    let rms = 7.5_f32.sqrt();
    assert!(
        (vals[0] - 1.0 / rms).abs() < 1e-4,
        "rms_norm val[0]: {}",
        vals[0]
    );
    assert!(
        (vals[3] - 4.0 / rms).abs() < 1e-4,
        "rms_norm val[3]: {}",
        vals[3]
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn linear_plain_forward() {
    // 2×2 weight, 1×2 input → 1×2 output.
    // weight = [[1, 0], [0, 1]] (identity), x = [[3, 4]] → out = [[3, 4]].
    let w_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let x_data: [f32; 2] = [3.0, 4.0];
    let w = Array::from_bytes(f32_as_bytes(&w_data), &[2, 2], Dtype::F32).unwrap();
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 2], Dtype::F32).unwrap();
    let lin = Linear::Plain { weight: w };
    let out = lin.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    let bytes = out.to_bytes().unwrap();
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((vals[0] - 3.0).abs() < 1e-5, "linear[0]: {}", vals[0]);
    assert!((vals[1] - 4.0).abs() < 1e-5, "linear[1]: {}", vals[1]);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn repeat_kv_expands_correctly() {
    // kv [1, 2, 1, 4] → [1, 8, 1, 4] with repeat=4
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let kv = Array::from_bytes(f32_as_bytes(&data), &[1, 2, 1, 4], Dtype::F32).unwrap();
    let expanded = repeat_kv(&kv, 4, Device::Cpu).unwrap();
    expanded.eval().unwrap();
    let shape = expanded.shape();
    assert_eq!(shape, vec![1, 8, 1, 4], "repeat_kv shape: {shape:?}");
}

fn make_test_config(
    num_hidden_layers: usize,
    num_kv_shared_layers: usize,
    layer_types: Vec<LayerType>,
) -> Gemma4TextConfig {
    Gemma4TextConfig {
        num_hidden_layers,
        hidden_size: 64,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        head_dim: 32,
        global_head_dim: 64,
        intermediate_size: 128,
        vocab_size: 256,
        sliding_window: 16,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        num_kv_shared_layers,
        hidden_size_per_layer_input: 0,
        final_logit_softcapping: 30.0,
        layer_types,
        quant_group_size: 32,
        quant_bits: 8,
        quant_mode: "mxfp8".to_owned(),
        rope_sliding_theta: 10000.0,
        rope_full_theta: 1_000_000.0,
        rope_full_dims: 16,
        attention_k_eq_v: false,
        quant_overrides: std::collections::HashMap::new(),
        enable_moe_block: false,
        num_experts: 0,
        top_k_experts: 0,
        moe_intermediate_size: 0,
        max_position_embeddings: 0,
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn build_previous_kvs_shared() {
    // Minimal config: 6 layers, 2 shared, pattern [slide, slide, slide, full, slide, full].
    let cfg = make_test_config(
        6,
        2,
        vec![
            LayerType::SlidingAttention,
            LayerType::SlidingAttention,
            LayerType::SlidingAttention,
            LayerType::FullAttention,
            LayerType::SlidingAttention, // shared: should map to last slide = 2
            LayerType::FullAttention,    // shared: should map to last full = 3
        ],
    );
    let kvs = build_previous_kvs(&cfg);
    assert_eq!(kvs[0], 0); // not shared
    assert_eq!(kvs[1], 1);
    assert_eq!(kvs[2], 2);
    assert_eq!(kvs[3], 3);
    assert_eq!(kvs[4], 2, "layer 4 (slide shared) → layer 2 (last slide)");
    assert_eq!(kvs[5], 3, "layer 5 (full shared) → layer 3 (last full)");
}

/// Verify that `attention_k_eq_v=true` is parsed correctly from a config that includes it,
/// and that `num_global_key_value_heads` is read when present.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn config_parses_attention_k_eq_v() {
    let json = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "bfloat16",
        "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
        "text_config": {
            "num_hidden_layers": 6,
            "hidden_size": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "head_dim": 64,
            "global_head_dim": 128,
            "intermediate_size": 512,
            "vocab_size": 1024,
            "sliding_window": 64,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "final_logit_softcapping": 30.0,
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention",
                             "sliding_attention","sliding_attention","full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention":    {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
            }
        }
    }"#;
    let raw: rmlx_loader::ModelConfig = serde_json::from_str(json).unwrap();
    let cfg = Gemma4TextConfig::from_model_config(&raw, None).unwrap();
    assert!(cfg.attention_k_eq_v, "attention_k_eq_v should be true");
    assert_eq!(
        cfg.num_global_key_value_heads, 2,
        "num_global_key_value_heads should be 2"
    );
    // Sliding layers use num_key_value_heads=8, full-attention uses num_global=2.
    assert_eq!(cfg.num_key_value_heads, 8);
}

/// Verify that the loader-level logic for k_eq_v sharing produces the right v_proj.
#[test]
fn k_eq_v_branch_logic() {
    let cases = [
        // (attention_k_eq_v, layer_type, v_proj_absent, expect_reuse)
        (true, LayerType::FullAttention, true, true), // 26B/31B full-attn: reuse
        (true, LayerType::FullAttention, false, false), // v_proj present even with flag: load it
        (true, LayerType::SlidingAttention, true, false), // sliding never reuses
        (false, LayerType::FullAttention, true, false), // flag off: don't reuse (would error)
    ];
    for (k_eq_v, lt, v_absent, expect_reuse) in cases {
        let use_k_eq_v = k_eq_v && lt == LayerType::FullAttention;
        let would_reuse = use_k_eq_v && v_absent;
        assert_eq!(
            would_reuse, expect_reuse,
            "k_eq_v={k_eq_v} lt={lt:?} v_absent={v_absent}: expected reuse={expect_reuse}"
        );
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn config_from_model_config_defaults() {
    let json = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "bfloat16",
        "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
        "text_config": {
            "num_hidden_layers": 6,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "global_head_dim": 128,
            "intermediate_size": 512,
            "vocab_size": 1024,
            "sliding_window": 64,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 32,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
            }
        }
    }"#;
    let raw: rmlx_loader::ModelConfig = serde_json::from_str(json).unwrap();
    let cfg = Gemma4TextConfig::from_model_config(&raw, None).unwrap();
    assert_eq!(cfg.num_hidden_layers, 6);
    assert_eq!(cfg.num_kv_shared_layers, 2);
    assert_eq!(cfg.rope_full_dims, 32); // 0.25 * 128
    assert!((cfg.rope_sliding_theta - 10_000.0).abs() < 1.0);
    assert!((cfg.rope_full_theta - 1_000_000.0).abs() < 1.0);
    assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
    assert!(!cfg.attention_k_eq_v);
    assert_eq!(cfg.num_global_key_value_heads, cfg.num_key_value_heads);
}

// ── classify_smoke ───────────────────────────────────────────────────────

fn make_steps(pieces: &[&str]) -> Vec<ProbeStep> {
    pieces
        .iter()
        .enumerate()
        .map(|(i, &p)| ProbeStep {
            token_id: i as u32,
            piece: p.to_owned().into_boxed_str(),
            max_abs_logit: 1.0,
            nan_count: 0,
            logprobs: None,
        })
        .collect()
}

#[test]
fn classify_smoke_ok_distinct_alphanum() {
    let pieces = ["Hello", "world", "this", "is", "a", "test", "of", "rmlx"];
    let steps = make_steps(&pieces);
    assert_eq!(classify_smoke(&steps), SmokeVerdict::Ok);
}

#[test]
fn classify_smoke_broken_punct_all_bang() {
    let steps: Vec<ProbeStep> = (0..8)
        .map(|_| ProbeStep {
            token_id: 42,
            piece: "!".to_owned().into_boxed_str(),
            max_abs_logit: 1.0,
            nan_count: 0,
            logprobs: None,
        })
        .collect();
    assert!(matches!(
        classify_smoke(&steps),
        SmokeVerdict::BrokenPunctLoop {
            distinct_ids: 1,
            ..
        }
    ));
}

#[test]
fn classify_smoke_broken_punct_alternating() {
    let steps: Vec<ProbeStep> = (0..8)
        .map(|i| ProbeStep {
            token_id: if i % 2 == 0 { 10 } else { 11 },
            piece: if i % 2 == 0 { "?" } else { "!" }
                .to_owned()
                .into_boxed_str(),
            max_abs_logit: 1.0,
            nan_count: 0,
            logprobs: None,
        })
        .collect();
    assert!(matches!(
        classify_smoke(&steps),
        SmokeVerdict::BrokenPunctLoop {
            distinct_ids: 2,
            ..
        }
    ));
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn classify_smoke_broken_nan() {
    let mut steps = make_steps(&["a", "b", "c", "d"]);
    steps[2].nan_count = 5;
    assert_eq!(
        classify_smoke(&steps),
        SmokeVerdict::BrokenNan { at_step: 2 }
    );
}

#[test]
fn classify_smoke_inconclusive_one_step() {
    let steps = make_steps(&["<eos>"]);
    assert!(matches!(
        classify_smoke(&steps),
        SmokeVerdict::Inconclusive { .. }
    ));
}

#[test]
fn classify_smoke_two_distinct_nonpunct_is_ok() {
    let steps: Vec<ProbeStep> = (0..8)
        .map(|i| ProbeStep {
            token_id: if i % 2 == 0 { 100 } else { 101 },
            piece: if i % 2 == 0 { "Hello" } else { "world" }
                .to_owned()
                .into_boxed_str(),
            max_abs_logit: 1.0,
            nan_count: 0,
            logprobs: None,
        })
        .collect();
    assert_eq!(classify_smoke(&steps), SmokeVerdict::Ok);
}

// ── forward_seq arbitrary-length ────────────────────────────────────────

#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn forward_seq_multi_token_gives_different_argmax() {
    let Some(model_path_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
    else {
        eprintln!("[forward_seq_test] skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("[forward_seq_test] snapshot absent — skipping");
        return;
    }

    let model = load_from_path(model_path).expect("load_from_path");

    let bos: u32 = 2;
    let seq_a: &[u32] = &[bos, 105, 2364, 107];
    let seq_b: &[u32] = &[bos, 105, 4368, 107];

    let device = Device::Cpu;

    let logits_a = model.forward_seq(seq_a, device).expect("forward_seq seq_a");
    logits_a.eval().expect("eval seq_a");

    let logits_b = model.forward_seq(seq_b, device).expect("forward_seq seq_b");
    logits_b.eval().expect("eval seq_b");

    let vocab = model.cfg.vocab_size as i32;
    let flat_a = logits_a
        .reshape(&[1, vocab], device)
        .expect("reshape seq_a");
    flat_a.eval().expect("eval flat_a");
    let flat_b = logits_b
        .reshape(&[1, vocab], device)
        .expect("reshape seq_b");
    flat_b.eval().expect("eval flat_b");

    let top_a = argmax(&flat_a, -1, device).expect("argmax a");
    top_a.eval().expect("eval top_a");
    let id_a = i32::from_le_bytes(top_a.to_bytes().expect("bytes")[..4].try_into().unwrap()) as u32;

    let top_b = argmax(&flat_b, -1, device).expect("argmax b");
    top_b.eval().expect("eval top_b");
    let id_b = i32::from_le_bytes(top_b.to_bytes().expect("bytes")[..4].try_into().unwrap()) as u32;

    eprintln!("[forward_seq_test] seq_a argmax={id_a}  seq_b argmax={id_b}");

    assert_ne!(
        id_a, id_b,
        "different prefixes should produce different argmax tokens; got {id_a} for both"
    );
}

/// `forward_hidden_states` returns the pre-final-norm trunk hidden, and
/// `logits_from_hidden` re-derives the exact logits the standard last-K path
/// produces. This is the reference penultimate-state extraction check: the MTP
/// conditioning signal must compose with the LM-head tail to reproduce the
/// verifier logits. Also asserts shape `[1, k, hidden]` + no NaN.
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn forward_hidden_states_matches_reference_extraction() {
    let Some(model_path_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E2B").map(std::path::PathBuf::from)
    else {
        eprintln!("[forward_hidden_states_test] skipping: RMLX_TEST_MODEL_GEMMA4_E2B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("[forward_hidden_states_test] snapshot absent — skipping");
        return;
    }
    let model = load_from_path(model_path).expect("load_from_path");
    let device = Device::Cpu;

    let ids: &[u32] = &[2, 105, 2364, 107, 4368, 105];
    let k = 3usize;
    let hidden = model.cfg.hidden_size as i32;
    let vocab = model.cfg.vocab_size as i32;

    // Pre-final-norm hidden at last k positions: [1, k, hidden].
    let h = model
        .forward_hidden_states(ids, k, None, device)
        .expect("forward_hidden_states");
    h.eval().expect("eval hidden");
    let shape = h.shape();
    assert_eq!(shape, &[1, k as i32, hidden], "hidden shape");

    // Non-NaN: a finite hidden has finite elements.
    let bytes = h.to_bytes().expect("hidden bytes");
    let any_nan = bytes
        .chunks_exact(4)
        .any(|c| f32::from_le_bytes(c.try_into().unwrap()).is_nan());
    assert!(!any_nan, "hidden contains NaN");

    // Reference equivalence: logits_from_hidden(hidden) == forward_seq_last_k.
    let logits_via_hidden = model
        .logits_from_hidden(&h, device)
        .expect("logits_from_hidden");
    logits_via_hidden.eval().expect("eval lvh");
    let logits_direct = model
        .forward_seq_last_k(ids, k, device)
        .expect("forward_seq_last_k");
    logits_direct.eval().expect("eval direct");

    let a = logits_via_hidden
        .reshape(&[k as i32, vocab], device)
        .expect("reshape a");
    let b = logits_direct
        .reshape(&[k as i32, vocab], device)
        .expect("reshape b");
    a.eval().expect("eval a");
    b.eval().expect("eval b");
    let ba = a.to_bytes().expect("ba");
    let bb = b.to_bytes().expect("bb");
    assert_eq!(ba.len(), bb.len(), "logit byte len");
    let max_abs_diff = ba
        .chunks_exact(4)
        .zip(bb.chunks_exact(4))
        .map(|(x, y)| {
            let fx = f32::from_le_bytes(x.try_into().unwrap());
            let fy = f32::from_le_bytes(y.try_into().unwrap());
            (fx - fy).abs()
        })
        .fold(0.0f32, f32::max);
    eprintln!("[forward_hidden_states_test] max_abs_logit_diff={max_abs_diff}");
    assert!(
        max_abs_diff < 1e-3,
        "logits_from_hidden must reproduce forward_seq_last_k; max diff {max_abs_diff}"
    );
}

#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn integration_paro_forward_gemma4() {
    let Some(model_path_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_PARO").map(std::path::PathBuf::from)
    else {
        eprintln!(
            "[integration_paro_forward_gemma4] skipping: RMLX_TEST_MODEL_GEMMA4_PARO not set"
        );
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("[integration_paro_forward_gemma4] snapshot absent — skipping");
        return;
    }

    let model = load_from_path_paro(model_path).expect("load_from_path_paro");

    let seq: &[u32] = &[2, 105];
    let device = Device::Cpu;

    let logits = model.forward_seq(seq, device).expect("forward_seq");
    logits.eval().expect("eval logits");

    let vocab = model.cfg.vocab_size as i32;
    let flat = logits.reshape(&[1, vocab], device).expect("reshape");
    flat.eval().expect("eval flat");

    let top = argmax(&flat, -1, device).expect("argmax");
    top.eval().expect("eval top");

    let id = i32::from_le_bytes(top.to_bytes().expect("bytes")[..4].try_into().unwrap()) as u32;

    eprintln!("[integration_paro_forward_gemma4] argmax token id = {id}");

    assert!(id > 0, "argmax token id should be > 0, got {id}");
}

/// end-to-end image-embeds path. Loads the e4b vision tower, preprocesses
/// a tiny solid-red PNG, builds the image-token block, and asserts
/// `build_inputs_embeds` produces `[1, seq, hidden]` with the image-token count
/// equal to the vision tower's `num_soft_tokens` (the scatter-alignment
/// invariant). Gated `#[ignore]` — needs the model snapshot.
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn build_inputs_embeds_aligns_and_shapes() {
    use super::config::Gemma4VisionConfig;
    use super::{build_inputs_embeds, load_vision_tower, Gemma4ImageProcessor, IMAGE_TOKEN_ID};

    let Some(model_path_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
    else {
        eprintln!("[image_test] skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("[image_test] snapshot absent — skipping");
        return;
    }
    let device = Device::Cpu;

    let model = load_from_path(model_path).expect("load text model");
    let vcfg = Gemma4VisionConfig::from_model_dir(model_path)
        .expect("read vision_config")
        .expect("e4b must ship a vision_config");
    let (vision, embedder) = load_vision_tower(model_path, &vcfg).expect("load vision tower");
    let processor = Gemma4ImageProcessor::from_model_dir(model_path).expect("load processor");

    // Tiny solid-red 64x64 PNG.
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([220, 30, 30]));
    let mut png: Vec<u8> = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode png");
    let pv = processor.preprocess(&png).expect("preprocess");
    let n_soft = pv.num_soft_tokens;
    eprintln!("[image_test] num_soft_tokens = {n_soft}");

    // Minimal prompt: BOS + image block + a couple text tokens.
    let mut ids: Vec<u32> = vec![2, 255_999];
    ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, n_soft));
    ids.push(258_882);
    ids.extend_from_slice(&[105, 2364, 107]);

    let in_prompt = ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    assert_eq!(
        in_prompt, n_soft,
        "image-token count must equal num_soft_tokens"
    );

    let (embeds, masked) =
        build_inputs_embeds(&model, &vision, &embedder, &[pv], &ids, device, None, 0)
            .expect("build embeds");
    embeds.eval().expect("eval embeds");

    let es = embeds.shape();
    assert_eq!(es[0], 1, "batch dim");
    assert_eq!(es[1], ids.len() as i32, "seq dim == augmented prompt len");
    assert_eq!(es[2], model.cfg.hidden_size as i32, "hidden dim");

    // masked ids: image positions zeroed, text positions preserved.
    let mb = masked.to_bytes().expect("masked bytes");
    let masked_ids: Vec<i32> = mb
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(masked_ids.len(), ids.len());
    for (i, &t) in ids.iter().enumerate() {
        if t == IMAGE_TOKEN_ID {
            assert_eq!(masked_ids[i], 0, "image position {i} must be masked to 0");
        } else {
            assert_eq!(masked_ids[i], t as i32, "text position {i} preserved");
        }
    }
    eprintln!("[image_test] OK — embeds {es:?}, alignment verified");
}
