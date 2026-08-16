use super::*;
use rmlx_mlx::{Array, Device, Dtype};

/// Build a small dummy `SparseAttnInputs` for the gate-OFF unit tests.
///
/// The gate check (`!sparse_attn_enabled()`) fires before any input
/// validation or kernel dispatch — so these arrays only need to be
/// constructible, not coherent.  We make them tiny f32 arrays sized for
/// `b=1, kv_h=1, heads_per_kv=1, kv_seq=2, head_dim=64`.
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Array build is correct"
)]
fn make_dummy_inputs() -> (Array, Array, Array, Array, Array) {
    let head_dim = 64usize;
    let kv_seq = 2usize;
    let q_bytes: Vec<u8> = vec![0u8; head_dim * 4];
    let q = Array::from_bytes(&q_bytes, &[1, 1, 1, head_dim as i32], Dtype::F32).expect("dummy Q");
    // PlanarQuant K codes: (head_dim / 32) * 4 = 8 u32 words per token, times 2 tokens = 16.
    let codes_per_tok = (head_dim / 32) * 4;
    let codes_total = kv_seq * codes_per_tok;
    let codes_bytes: Vec<u8> = vec![0u8; codes_total * 4];
    let k_codes =
        Array::from_bytes(&codes_bytes, &[codes_total as i32], Dtype::U32).expect("dummy K codes");
    let scales_total = kv_seq * head_dim / 2;
    let scales_bytes: Vec<u8> = vec![0u8; scales_total * 4];
    let k_scales = Array::from_bytes(&scales_bytes, &[scales_total as i32], Dtype::F32)
        .expect("dummy K scales");
    let rot_total = kv_seq * head_dim / 16;
    let rot_bytes: Vec<u8> = vec![0u8; rot_total * 4];
    let k_rot32 =
        Array::from_bytes(&rot_bytes, &[rot_total as i32], Dtype::U32).expect("dummy K rot32");
    let v_total = kv_seq * head_dim;
    let v_bytes: Vec<u8> = vec![0u8; v_total * 4];
    let v = Array::from_bytes(
        &v_bytes,
        &[1, 1, kv_seq as i32, head_dim as i32],
        Dtype::F32,
    )
    .expect("dummy V");
    (q, k_codes, k_scales, k_rot32, v)
}

fn make_dummy_sparse_inputs<'a>(
    q: &'a Array,
    k_codes: &'a Array,
    k_scales: &'a Array,
    k_rot32: &'a Array,
    v: &'a Array,
) -> SparseAttnInputs<'a> {
    SparseAttnInputs {
        query: q,
        k_codes,
        k_scales,
        k_rot32,
        v,
        b: 1,
        kv_h: 1,
        kv_seq: 2,
        head_dim: 64,
        heads_per_kv: 1,
        layer_idx: 0,
        scale: 0.125,
        device: Device::Cpu,
    }
}

// ── sparse_attn_dispatch_if_enabled stub tests ────────────────────────────────
//
// All four cases return None in Exec A — the dispatch site is a placeholder
// until Exec B wires in real phase-1 / phase-2 MSL kernels.
//
// OnceLock note: `sparse_attn_enabled()` latches its env read on first call.
// In a fresh test process the env-var is unset → returns false → these
// assertions are stable regardless of test ordering. The CLI env-setter
// tests live in `rmlx-cli::commands::serve_tests` (no overlap).

/// Build a `HeadBudgets` via JSON round-trip (the struct is
/// `#[non_exhaustive]` so direct construction is banned outside
/// `rmlx-loader`).
#[allow(
    clippy::expect_used,
    reason = "test helper: JSON is a compile-time constant; panic on parse failure is the correct behavior"
)]
fn make_test_head_budgets() -> HeadBudgets {
    let json = r#"{
      "version": 1,
      "model_name": "test",
      "num_layers": 1,
      "num_heads": 2,
      "calibration": {
        "method": "softmax_mass",
        "prompt_set_sha256": "ab",
        "num_prompts": 1,
        "max_seq_len": 128,
        "mass_threshold": 0.95
      },
      "per_layer_per_head_budget": [[16, 16]]
    }"#;
    serde_json::from_str(json).expect("parse test head_budgets fixture")
}

#[test]
fn sparse_attn_dispatch_none_when_gate_off_and_budgets_absent() {
    // Default state in a fresh test process: env-var unset, no budgets.
    // The function must return None because the gate short-circuits before
    // touching the inputs (Exec B: gate ON path runs the kernels).
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, None).is_none());
}

#[test]
fn sparse_attn_dispatch_none_when_gate_off_with_budgets_present() {
    // Even with budgets present, the gate is off (OnceLock latches OFF
    // in this test process) → must return None.
    let budgets = make_test_head_budgets();
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, Some(&budgets)).is_none());
}

#[test]
fn sparse_attn_dispatch_short_circuits_on_missing_budgets() {
    // Even if a future test in the same process latched the OnceLock to
    // true, missing budgets must still produce None (correctness invariant).
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, None).is_none());
}
