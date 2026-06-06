use super::*;

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `RmsNormShifted` with weight=0 must equal plain RMSNorm with weight=1.
#[test]
fn shifted_zero_weight_matches_plain_unit_weight() {
    let x_data = [1.0_f32, 2.0, 3.0, 4.0];
    let zero = [0.0_f32; 4];

    let x = Array::from_bytes(&f32_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let w_zero = Array::from_bytes(&f32_bytes(&zero), &[4], Dtype::F32).unwrap();

    let shifted = RmsNormShifted::from_weight(&w_zero, 1e-6).unwrap();
    let out = shifted.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());

    // E[x^2] = (1+4+9+16)/4 = 7.5. Output[0] = 1 / sqrt(7.5).
    let expected = 1.0_f32 / 7.5_f32.sqrt();
    assert!(
        (vals[0] - expected).abs() < 1e-4,
        "shifted[0]={}, expected≈{expected}",
        vals[0]
    );
}

/// `from_weight` with weight=k should produce `shifted_weight = k + 1`.
#[test]
fn from_weight_adds_one() {
    let w_data = [0.5_f32, 1.5, -0.5, 2.0];
    let w = Array::from_bytes(&f32_bytes(&w_data), &[4], Dtype::F32).unwrap();
    let shifted = RmsNormShifted::from_weight(&w, 1e-6).unwrap();
    shifted.shifted_weight.eval().unwrap();
    let out = bytes_to_f32(&shifted.shifted_weight.to_bytes().unwrap());
    let expected = [1.5_f32, 2.5, 0.5, 3.0];
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5, "got={got}, want={want}");
    }
}
