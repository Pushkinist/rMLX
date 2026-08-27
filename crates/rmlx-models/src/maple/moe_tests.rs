//! CPU tests for Maple clamped-SwiGLU (trained forward, not an optional guard).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use super::clamped_swiglu;
use rmlx_mlx::{Array, Device, Dtype};

fn to_f32_vec(a: &Array) -> Vec<f32> {
    let a_f32 = a.astype(Dtype::F32, Device::Cpu).unwrap();
    a_f32.eval().unwrap();
    a_f32
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn clamped_swiglu_caps_gate_above_seven_and_clips_up() {
    let gate = Array::from_f32_slice(&[8.0, -10.0], &[1, 2]).unwrap();
    let up = Array::from_f32_slice(&[1.0, 10.0], &[1, 2]).unwrap();
    let out = clamped_swiglu(&gate, &up, Device::Cpu).unwrap();
    let got = to_f32_vec(&out);
    // gate 8 → silu(7)*1; gate -10 stays unbounded → silu(-10)*clip(10,7)
    let want = [silu(7.0), silu(-10.0) * 7.0];
    assert!(
        (got[0] - want[0]).abs() < 1e-5,
        "got {} want {}",
        got[0],
        want[0]
    );
    assert!(
        (got[1] - want[1]).abs() < 1e-5,
        "got {} want {}",
        got[1],
        want[1]
    );
}

#[test]
fn clamped_swiglu_preserves_bf16_dtype() {
    let gate = Array::from_f32_slice(&[1.0, 2.0], &[1, 2])
        .unwrap()
        .astype(Dtype::Bf16, Device::Cpu)
        .unwrap();
    let up = Array::from_f32_slice(&[1.0, 2.0], &[1, 2])
        .unwrap()
        .astype(Dtype::Bf16, Device::Cpu)
        .unwrap();
    let out = clamped_swiglu(&gate, &up, Device::Cpu).unwrap();
    out.eval().unwrap();
    assert_eq!(out.dtype(), Dtype::Bf16);
}
