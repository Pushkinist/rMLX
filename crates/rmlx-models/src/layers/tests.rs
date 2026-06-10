use super::*;
use rmlx_mlx::{Array, Device, Dtype};
use std::collections::HashMap;
use std::mem::size_of;

/// `QuantMode` is a Copy enum — 1 byte, no heap.
///
/// Before perf(types): `mode: String` was 24 B + 1 heap alloc per layer.
/// After: 1 B discriminant, zero heap. On aarch64 the enum fits in a u8.
#[test]
fn quant_mode_is_one_byte() {
    assert_eq!(
        size_of::<QuantMode>(),
        1,
        "QuantMode must be 1 byte (fits in u8)"
    );
    // Verify round-trip via as_str / From<&str>.
    assert_eq!(QuantMode::from("affine"), QuantMode::Affine);
    assert_eq!(QuantMode::from("mxfp8"), QuantMode::Mxfp8);
    assert_eq!(QuantMode::from("mxfp4"), QuantMode::Mxfp4);
    assert_eq!(QuantMode::from("nvfp4"), QuantMode::Nvfp4);
    // Unknown strings fall back to Affine.
    assert_eq!(QuantMode::from("unknown_mode"), QuantMode::Affine);
    // as_str returns the MLX-expected strings.
    assert_eq!(QuantMode::Affine.as_str(), "affine");
    assert_eq!(QuantMode::Mxfp8.as_str(), "mxfp8");
    assert_eq!(QuantMode::Mxfp4.as_str(), "mxfp4");
    assert_eq!(QuantMode::Nvfp4.as_str(), "nvfp4");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "non-biased override path is infallible; unwrap asserts the Ok branch"
)]
fn resolve_quant_override_wins() {
    let defaults = QuantParams::global(32, 8, "mxfp8");
    let mut overrides = HashMap::new();
    overrides.insert(
        "model.layers.5.mlp.gate.proj".to_owned(),
        QuantParams {
            group_size: 64,
            bits: 4,
            mode: "mxfp4".to_owned(),
        },
    );
    // Override group_size/bits/mode all applied (no biases → mode kept as-is).
    let resolved =
        resolve_quant("model.layers.5.mlp.gate.proj", false, &defaults, &overrides).unwrap();
    assert_eq!(resolved.group_size, 64, "override group_size");
    assert_eq!(resolved.bits, 4, "override bits");
    assert_eq!(resolved.mode, "mxfp4", "override mode");

    // Empty override mode inherits the global default mode.
    let mut overrides_empty_mode = HashMap::new();
    overrides_empty_mode.insert(
        "model.layers.5.mlp.gate.proj".to_owned(),
        QuantParams {
            group_size: 64,
            bits: 8,
            mode: String::new(),
        },
    );
    let inherited = resolve_quant(
        "model.layers.5.mlp.gate.proj",
        false,
        &defaults,
        &overrides_empty_mode,
    )
    .unwrap();
    assert_eq!(inherited.group_size, 64);
    assert_eq!(
        inherited.mode, "mxfp8",
        "empty override mode inherits global"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "non-biased defaults path is infallible; unwrap asserts the Ok branch"
)]
fn resolve_quant_defaults_apply() {
    let defaults = QuantParams::global(32, 8, "mxfp8");
    let overrides = HashMap::new();
    let resolved =
        resolve_quant("model.layers.0.mlp.gate_proj", false, &defaults, &overrides).unwrap();
    assert_eq!(resolved.group_size, 32);
    assert_eq!(resolved.bits, 8);
    assert_eq!(resolved.mode, "mxfp8");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "biased mode-absent path forces affine and returns Ok; unwrap asserts it"
)]
fn resolve_quant_biases_force_affine_when_mode_absent() {
    let defaults = QuantParams::global(32, 8, "mxfp8");
    let mut overrides = HashMap::new();
    // Override present but mode absent (empty) + a .biases sibling → affine.
    overrides.insert(
        "model.layers.0.router.proj".to_owned(),
        QuantParams {
            group_size: 32,
            bits: 4,
            mode: String::new(),
        },
    );
    let resolved =
        resolve_quant("model.layers.0.router.proj", true, &defaults, &overrides).unwrap();
    assert_eq!(
        resolved.mode, "affine",
        "biases + absent mode forces affine"
    );
    assert_eq!(resolved.group_size, 32);
    assert_eq!(resolved.bits, 4);

    // No override at all, default mode mxfp8, but biases present → affine.
    let empty = HashMap::new();
    let no_ov = resolve_quant("model.layers.0.q_proj", true, &defaults, &empty).unwrap();
    assert_eq!(no_ov.mode, "affine", "biases + no override forces affine");
}

#[test]
fn resolve_quant_biases_with_explicit_nonaffine_is_err() {
    let defaults = QuantParams::global(32, 8, "mxfp8");
    let mut overrides = HashMap::new();
    // Explicit non-affine override mode (`mxfp8` parses to a real microscaling
    // mode, not the affine fallback) alongside a .biases sibling → hard error.
    overrides.insert(
        "model.layers.0.q_proj".to_owned(),
        QuantParams {
            group_size: 32,
            bits: 4,
            mode: "mxfp8".to_owned(),
        },
    );
    let err = resolve_quant("model.layers.0.q_proj", true, &defaults, &overrides);
    assert!(
        err.is_err(),
        "explicit non-affine mode + biases must hard-error"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "unrecognized-but-affine override + biases returns Ok(affine); unwrap asserts it"
)]
fn resolve_quant_biases_with_unrecognized_override_mode_is_affine() {
    // An override-set mode string that `QuantMode::from` maps to Affine (its
    // documented fallback for unrecognized strings) must NOT hard-error when a
    // .biases sibling is present — both pre-unification resolvers decoded such a
    // string as affine. Only a genuine microscaling/fp4 mode contradicts biases.
    let defaults = QuantParams::global(32, 8, "mxfp8");
    let mut overrides = HashMap::new();
    overrides.insert(
        "model.layers.0.q_proj".to_owned(),
        QuantParams {
            group_size: 32,
            bits: 4,
            mode: "int4".to_owned(), // unrecognized → QuantMode::from → Affine
        },
    );
    let resolved = resolve_quant("model.layers.0.q_proj", true, &defaults, &overrides).unwrap();
    assert_eq!(
        resolved.mode, "affine",
        "an unrecognized-but-affine override mode + biases stays affine"
    );
}

fn f32_as_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) }
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    assert!(b.len().is_multiple_of(4));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
fn rms_norm_plain_gamma() {
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
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    let rms = 7.5_f32.sqrt();
    assert!(
        (vals[0] - 1.0 / rms).abs() < 1e-4,
        "rms_norm[0]: {}",
        vals[0]
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
fn rms_norm_no_weight() {
    let x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let norm = RmsNorm {
        weight: None,
        eps: 1e-6,
    };
    let out = norm.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    let rms = 7.5_f32.sqrt();
    assert!(
        (vals[3] - 4.0 / rms).abs() < 1e-4,
        "rms_norm_no_weight[3]: {}",
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
fn linear_plain_identity() {
    let w_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let x_data: [f32; 2] = [3.0, 4.0];
    let w = Array::from_bytes(f32_as_bytes(&w_data), &[2, 2], Dtype::F32).unwrap();
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 2], Dtype::F32).unwrap();
    let lin = Linear::Plain { weight: w };
    let out = lin.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    assert!((vals[0] - 3.0).abs() < 1e-5, "linear[0]: {}", vals[0]);
    assert!((vals[1] - 4.0).abs() < 1e-5, "linear[1]: {}", vals[1]);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn mlp_gelu_tanh_shape() {
    let mk_w = |rows: usize, cols: usize| -> Array {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
        Array::from_bytes(f32_as_bytes(&data), &[rows as i32, cols as i32], Dtype::F32).unwrap()
    };
    let mlp = Mlp {
        gate_proj: Linear::Plain { weight: mk_w(8, 4) },
        up_proj: Linear::Plain { weight: mk_w(8, 4) },
        down_proj: Linear::Plain { weight: mk_w(4, 8) },
        activation: Activation::GeluTanh,
    };
    let x_data: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let out = mlp.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    assert_eq!(out.shape(), vec![1, 4], "mlp output shape");
}

// ---------------------------------------------------------------------------
// SWA mask builder tests
// ---------------------------------------------------------------------------

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn mask_bf16_to_f32(arr: &Array) -> Vec<f32> {
    let arr_f32 = arr.astype(Dtype::F32, Device::Cpu).unwrap();
    arr_f32.eval().unwrap();
    bytes_to_f32(&arr_f32.to_bytes().unwrap())
}

/// 4-token single-chunk prefill with window=2.
/// Expected banded lower-triangular:
/// row 0 (q=0): [0, -inf, -inf, -inf] window=[0]
/// row 1 (q=1): [0, 0, -inf, -inf] window=[0,1]
/// row 2 (q=2): [-inf, 0, 0, -inf] window=[1,2]
/// row 3 (q=3): [-inf, -inf, 0, 0] window=[2,3]
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn swa_prefill_mask_banded() {
    let mask = build_swa_prefill_mask(0, 4, 2, Device::Cpu).unwrap();
    mask.eval().unwrap();
    let vals = mask_bf16_to_f32(&mask);
    assert_eq!(vals.len(), 16, "shape mismatch (expected 1*1*4*4=16 cells)");

    let allowed = |row: usize, col: usize| -> bool { vals[row * 4 + col] == 0.0 };
    // row 0: only col 0
    assert!(allowed(0, 0), "row0,col0 should be allowed");
    assert!(!allowed(0, 1), "row0,col1 should be masked");
    // row 1: cols 0,1
    assert!(allowed(1, 0), "row1,col0 should be allowed");
    assert!(allowed(1, 1), "row1,col1 should be allowed");
    assert!(!allowed(1, 2), "row1,col2 should be masked");
    // row 2: cols 1,2 (col 0 is outside window)
    assert!(
        !allowed(2, 0),
        "row2,col0 should be masked (outside window)"
    );
    assert!(allowed(2, 1), "row2,col1 should be allowed");
    assert!(allowed(2, 2), "row2,col2 should be allowed");
    assert!(!allowed(2, 3), "row2,col3 should be masked (future)");
    // row 3: cols 2,3
    assert!(
        !allowed(3, 1),
        "row3,col1 should be masked (outside window)"
    );
    assert!(allowed(3, 2), "row3,col2 should be allowed");
    assert!(allowed(3, 3), "row3,col3 should be allowed");
}

/// Full-attention chunked-prefill still allows all past keys (no window).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn chunked_prefill_mask_lower_triangular() {
    // offset=2, new_seq=3 → total cols=5
    let mask = build_chunked_prefill_mask(2, 3, Device::Cpu).unwrap();
    mask.eval().unwrap();
    let vals = mask_bf16_to_f32(&mask);
    assert_eq!(vals.len(), 15, "expected 1*1*3*5=15 cells");
    // row 0 (q_abs=2): cols 0,1,2 allowed; 3,4 blocked
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    assert!(vals[3] < -1e10);
    assert!(vals[4] < -1e10);
    // row 2 (q_abs=4): all 5 cols allowed
    assert_eq!(vals[10], 0.0);
    assert_eq!(vals[14], 0.0);
}

/// SWA decode mask: window=1024, total_kv_len=512 → None (all attend).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn swa_decode_mask_no_mask_when_within_window() {
    let result = build_swa_decode_mask(512, 1024, Device::Cpu).unwrap();
    assert!(
        result.is_none(),
        "should return None when total_kv_len <= window"
    );
}

/// SWA decode mask: window=4, total_kv_len=7 → first 3 blocked, last 4 allowed.
#[test]
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
fn swa_decode_mask_masks_old_keys() {
    let result = build_swa_decode_mask(7, 4, Device::Cpu).unwrap();
    let mask = result.expect("should return Some when total_kv_len > window");
    mask.eval().unwrap();
    let vals = mask_bf16_to_f32(&mask);
    assert_eq!(vals.len(), 7, "shape should be [1,1,1,7]");
    // first 3 are masked
    for &v in &vals[..3] {
        assert!(v < -1e10, "old key should be masked: {v}");
    }
    // last 4 are allowed
    for &v in &vals[3..] {
        assert_eq!(v, 0.0, "recent key should be allowed: {v}");
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn mlp_silu_shape() {
    let mk_w = |rows: usize, cols: usize| -> Array {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
        Array::from_bytes(f32_as_bytes(&data), &[rows as i32, cols as i32], Dtype::F32).unwrap()
    };
    let mlp = Mlp {
        gate_proj: Linear::Plain { weight: mk_w(8, 4) },
        up_proj: Linear::Plain { weight: mk_w(8, 4) },
        down_proj: Linear::Plain { weight: mk_w(4, 8) },
        activation: Activation::Silu,
    };
    let x_data: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let out = mlp.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    assert_eq!(out.shape(), vec![1, 4], "mlp silu output shape");
}
