use super::*;
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
