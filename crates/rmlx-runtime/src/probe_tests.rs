use super::*;

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        // Round-to-nearest-even: take top 16 bits with rounding bias.
        let bits = v.to_bits();
        let rounding_bias = 0x0000_8000 + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding_bias);
        let bf16 = (rounded >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

#[test]
fn count_nan_f32_zero() {
    let bytes = f32_bytes(&[1.0, 2.0, 3.0]);
    assert_eq!(count_nan_in_bytes(&bytes, Dtype::F32), 0);
}

#[test]
fn count_nan_f32_some() {
    let bytes = f32_bytes(&[1.0, f32::NAN, 3.0, f32::NAN]);
    assert_eq!(count_nan_in_bytes(&bytes, Dtype::F32), 2);
}

#[test]
fn count_nan_bf16() {
    let bytes = bf16_bytes(&[1.0, f32::NAN, 3.0]);
    assert_eq!(count_nan_in_bytes(&bytes, Dtype::Bf16), 1);
}

#[test]
fn max_abs_f32() {
    let bytes = f32_bytes(&[1.0, -3.5, 2.0]);
    let m = max_abs_from_bytes(&bytes, Dtype::F32);
    assert!((m - 3.5).abs() < 1e-6, "expected 3.5, got {m}");
}

#[test]
fn max_abs_bf16() {
    let bytes = bf16_bytes(&[1.0, -2.0, 0.5]);
    let m = max_abs_from_bytes(&bytes, Dtype::Bf16);
    assert!((m - 2.0).abs() < 1e-2, "expected ~2.0, got {m}");
}

#[test]
fn unsupported_dtype_returns_zero_count() {
    // I32 buffer should hit the `_ => 0` branch.
    let bytes = vec![0u8; 16];
    assert_eq!(count_nan_in_bytes(&bytes, Dtype::I32), 0);
    assert_eq!(max_abs_from_bytes(&bytes, Dtype::I32), 0.0);
}
