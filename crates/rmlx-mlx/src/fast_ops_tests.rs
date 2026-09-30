use super::*;
use crate::c_api::{self, CApi};
use crate::Dtype;

/// Verify rope_dynamic (offset as 0-D i32 array) produces matching output
/// to rope (offset as captured i32) for a representative shape.
///
/// Foundational invariant for using rope_dynamic inside an mx.compile
/// closure: the dynamic-offset path MUST match the static-offset path at
/// every step — otherwise the model's positional embeddings drift.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rope_dynamic_matches_static -- --ignored --test-threads=1"]
fn rope_dynamic_matches_static() {
    // Shape [B=1, H=2, S=1, D=8] — single-token decode step.
    let b = 1_i32;
    let h = 2_i32;
    let s = 1_i32;
    let d = 8_i32;
    let n = (b * h * s * d) as usize;
    // Deterministic data.
    let data: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.05, -0.4)).collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let x = Array::from_bytes(bytes, &[b, h, s, d], Dtype::F32).expect("from_bytes x");

    let offset_val: i32 = 17;
    let base = 10000.0_f32;
    let scale = 1.0_f32;

    // Static-offset reference.
    let y_static = rope(&x, d, false, base, scale, offset_val, Device::Gpu).expect("rope static");
    Array::eval(&y_static).expect("materialize static");
    let bytes_static = y_static.to_bytes().expect("to_bytes static");

    // Dynamic-offset path.
    let off_bytes = offset_val.to_le_bytes();
    let off_arr = Array::from_bytes(&off_bytes, &[], Dtype::I32).expect("from_bytes offset");
    let y_dyn =
        rope_dynamic(&x, d, false, base, scale, &off_arr, Device::Gpu).expect("rope_dynamic");
    Array::eval(&y_dyn).expect("materialize dynamic");
    let bytes_dyn = y_dyn.to_bytes().expect("to_bytes dynamic");

    assert_eq!(bytes_static.len(), bytes_dyn.len(), "byte-len mismatch");
    let sf: Vec<f32> = bytes_static
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let df: Vec<f32> = bytes_dyn
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (sv, dv)) in sf.iter().zip(df.iter()).enumerate() {
        let diff = (sv - dv).abs();
        assert!(
            diff < 1e-5,
            "rope_dynamic vs rope mismatch at idx {i}: static={sv} dyn={dv} diff={diff}"
        );
    }
}

fn f32_array(values: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

/// One query, two keys, head_dim 2, scale 1: the scores are [1, 0], so the
/// output is `softmax([1, 0]) @ V` = [1 + 2w, 2 + 2w] with w = 1 / (1 + e).
fn one_head_attention_inputs() -> (Array, Array, Array) {
    let q = f32_array(&[1.0, 0.0], &[1, 1, 1, 2]);
    let k = f32_array(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
    let v = f32_array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    (q, k, v)
}

fn one_head_attention_reference() -> Vec<f32> {
    let w = 1.0 / (1.0 + std::f32::consts::E);
    vec![2.0f32.mul_add(w, 1.0), 2.0f32.mul_add(w, 2.0)]
}

/// The CPU stream runs MLX's composite SDPA graph, so this needs no Metal.
/// A wrong argument list in the call does not compile, and `force_fused =
/// true` fails here: the CPU stream has no fused kernel to force.
///
/// On a pair whose `libmlxc.dylib` has another C API than the compiled one,
/// the call must be refused instead, with nothing passed to mlx-c.
#[test]
fn scaled_dot_product_attention_on_cpu_follows_the_c_api_verdict() {
    let (q, k, v) = one_head_attention_inputs();
    let out = scaled_dot_product_attention(&q, &k, &v, 1.0, "", None, Device::Cpu);
    match c_api::verdict() {
        CApiVerdict::Match(_) => {
            let got: Vec<f32> = out
                .expect("sdpa on the CPU stream")
                .to_bytes()
                .expect("to_bytes")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let want = one_head_attention_reference();
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-6, "sdpa {got:?} != reference {want:?}");
            }
            assert_eq!(got.len(), want.len());
        }
        CApiVerdict::Mismatch { .. } => {
            let err = out.expect_err("a C API mismatch must refuse the call");
            assert!(err.to_string().contains("mlx-c C API mismatch"), "{err}");
        }
    }
}

/// A mismatch verdict stops the call before mlx-c sees it. On a matched pair
/// the mlx-c call itself would succeed, so an `Ok` here means the check is
/// gone or comes after the call.
#[test]
fn a_c_api_mismatch_refuses_sdpa_before_the_mlx_c_call() {
    let (q, k, v) = one_head_attention_inputs();
    let mismatch = CApiVerdict::Mismatch {
        compiled: CApi::COMPILED,
        loaded: CApi::COMPILED.other(),
    };
    let err = sdpa_under(mismatch, &q, &k, &v, 1.0, "", None, Device::Cpu)
        .expect_err("a C API mismatch must refuse the call");
    assert!(err.to_string().contains("mlx-c C API mismatch"), "{err}");
}
