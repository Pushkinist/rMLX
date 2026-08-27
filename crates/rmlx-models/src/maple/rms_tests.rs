//! CPU tests for MapleRMSNorm dtype and load-time f32 weight.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::MapleRmsNorm;
use rmlx_mlx::{Array, Device, Dtype};

#[test]
fn maple_rms_norm_returns_input_dtype() {
    let x = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
    let x = x.astype(Dtype::Bf16, Device::Cpu).unwrap();
    let w = Array::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let w = w.astype(Dtype::Bf16, Device::Cpu).unwrap();
    let n = MapleRmsNorm::new(w, 1e-6, Device::Cpu).unwrap();
    let out = n.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    assert_eq!(out.dtype(), Dtype::Bf16);
    assert_eq!(out.shape(), vec![1, 1, 1, 4]);
    assert_eq!(n.weight.dtype(), Dtype::F32);
}
