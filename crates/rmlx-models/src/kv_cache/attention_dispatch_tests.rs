use super::*;
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

/// Build a small dummy `SparseAttnInputs` for the gate-OFF unit tests.
///
/// The gate check (`!policy.sparse_attn`) fires before any input
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

// ── sparse_attn_dispatch_if_enabled gate tests ────────────────────────────────
//
// The gate is a policy field the caller supplies, so each case names the
// policy it exercises instead of depending on process state or test ordering.

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
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, None, DispatchPolicy::default()).is_none());
}

#[test]
fn sparse_attn_dispatch_none_when_gate_off_with_budgets_present() {
    // Budgets present but the policy selects the dense path → None.
    let budgets = make_test_head_budgets();
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(
        sparse_attn_dispatch_if_enabled(&inputs, Some(&budgets), DispatchPolicy::default())
            .is_none()
    );
}

#[test]
fn sparse_attn_dispatch_short_circuits_on_missing_budgets() {
    // Gate open, budgets missing: the budgets check must still produce None.
    // Under the old process-global gate this case was unreachable from a
    // test, because the gate latched OFF for the whole binary.
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    let gate_open = DispatchPolicy {
        sparse_attn: true,
        ..DispatchPolicy::default()
    };
    assert!(sparse_attn_dispatch_if_enabled(&inputs, None, gate_open).is_none());
}

// ── dtype contract at the call site ──────────────────────────────────────────

/// `sparse_attn_dispatch` must hand back the query's dtype.
///
/// The phase-2 merge takes its output dtype from this call site
/// (`inputs.query.dtype()`), and nothing else checks it: the source gate scans
/// dispatchers rather than call sites, so a `Dtype::F32` written here would
/// pass it, and the cache-level sweep cannot reach this path at all — sparse
/// attention is driven from this crate with a `HeadBudgets`, which no
/// `KvCache` sweep supplies. An f32 attention output promotes the residual
/// stream and every downstream op in the layer.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-models --lib sparse_attn_dispatch_returns -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: a panic on Array build is the correct failure"
)]
fn sparse_attn_dispatch_returns_the_query_dtype() {
    use rmlx_kv_quant::planarquant_msl::planar_quantize_v4_gpu;

    let device = Device::Gpu;
    let (b, kv_h, heads_per_kv, kv_seq, head_dim) = (1_i32, 1_i32, 2_i32, 128_i32, 64_i32);
    let n_q_heads = kv_h * heads_per_kv;

    // Deterministic fixture data; the numbers do not matter, the dtype does.
    let lcg = |n: usize, mut st: u64| -> Vec<f32> {
        (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((st >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let f32_arr = |data: &[f32], shape: &[i32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
    };

    let kv_n = (b * kv_h * kv_seq * head_dim) as usize;
    let k_arr = f32_arr(&lcg(kv_n, 0x5A5A_0001), &[b, kv_h, kv_seq, head_dim]);
    let k_seq = k_arr
        .transpose(&[0, 2, 1, 3], device)
        .expect("k seq-major")
        .contiguous(device)
        .expect("k contiguous");
    let (k_codes, k_scales, k_rot32) = planar_quantize_v4_gpu(&k_seq, device).expect("quantize K");
    let v = f32_arr(&lcg(kv_n, 0x5A5A_0002), &[b, kv_h, kv_seq, head_dim])
        .astype(Dtype::Bf16, device)
        .expect("V bf16");
    let query = f32_arr(
        &lcg((b * n_q_heads * head_dim) as usize, 0x5A5A_0003),
        &[b, n_q_heads, 1, head_dim],
    )
    .astype(Dtype::Bf16, device)
    .expect("Q bf16");

    let inputs = SparseAttnInputs {
        query: &query,
        k_codes: &k_codes,
        k_scales: &k_scales,
        k_rot32: &k_rot32,
        v: &v,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        layer_idx: 0,
        scale: 1.0 / (head_dim as f32).sqrt(),
        device,
    };
    let budgets = make_test_head_budgets();

    let out = sparse_attn_dispatch(&inputs, &budgets).expect("sparse_attn_dispatch");
    assert_eq!(
        out.dtype(),
        query.dtype(),
        "sparse attention returned {:?} for a {:?} query — an f32 attention output \
         promotes the residual stream and the whole layer behind it",
        out.dtype(),
        query.dtype()
    );
}
