//! Gemma3 unit tests.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret in test helpers
#![cfg_attr(test, allow(unsafe_code))]

use super::layers::RmsNormShifted;
use rmlx_mlx::{rms_norm, Array, Device, Dtype};

fn f32_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) }
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn bytes_f32(b: &[u8]) -> Vec<f32> {
    assert!(b.len().is_multiple_of(4));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// RmsNormShifted should equal rms_norm(x, gamma+1).
///
/// Verifies the gamma+1 convention without involving any model snapshot.
/// Uses a 4-element gamma with non-trivial values and a 2x4 input tensor.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rms_norm_shifted_matches_explicit_plus_one() {
    // gamma = [0.5, -0.5, 1.0, 0.0] -> shifted = [1.5, 0.5, 2.0, 1.0]
    let gamma_data: [f32; 4] = [0.5, -0.5, 1.0, 0.0];
    let x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];

    let gamma = Array::from_bytes(f32_bytes(&gamma_data), &[4], Dtype::F32).unwrap();
    let x = Array::from_bytes(f32_bytes(&x_data), &[2, 4], Dtype::F32).unwrap();

    let shifted = RmsNormShifted::from_weight(&gamma, 1e-6).unwrap();
    let out = shifted.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    let got = bytes_f32(&out.to_bytes().unwrap());

    // Reference: explicit shifted gamma.
    let shifted_gamma_data: [f32; 4] = [1.5, 0.5, 2.0, 1.0];
    let sg = Array::from_bytes(f32_bytes(&shifted_gamma_data), &[4], Dtype::F32).unwrap();
    let ref_out = rms_norm(&x, Some(&sg), 1e-6, Device::Cpu).unwrap();
    ref_out.eval().unwrap();
    let expected = bytes_f32(&ref_out.to_bytes().unwrap());

    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "rms_norm_shifted[{i}]: got {g}, expected {e}"
        );
    }
}
