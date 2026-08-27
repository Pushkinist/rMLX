//! CPU tests for Maple `row_alpha` → affine group scales/biases expansion.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use super::{concat_linears, expand_row_alpha, n_groups_2bit, squeeze_to_row_rank};
use crate::layers::Linear;
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

#[test]
fn n_groups_2bit_from_packed_last_dim() {
    // 8 u32 words × 16 codes/word = 128 codes → 1 group of 128.
    assert_eq!(n_groups_2bit(8, 128).unwrap(), 1);
    assert_eq!(n_groups_2bit(16, 128).unwrap(), 2);
    assert!(n_groups_2bit(8, 0).is_err());
    assert!(n_groups_2bit(7, 128).is_err());
}

#[test]
fn squeeze_trailing_ones_to_row_rank() {
    let alpha = Array::from_f32_slice(&[0.5, 1.5], &[2, 1]).unwrap();
    let squeezed = squeeze_to_row_rank(alpha, 2, Device::Cpu).unwrap();
    assert_eq!(squeezed.shape(), vec![2]);
}

#[test]
fn expand_row_alpha_broadcasts_and_negates_as_biases() {
    // packed [2, 8] uint32: 8 words × 16 codes = 128 codes/row, group 128 → 1 group.
    let packed = Array::from_bytes(&[0u8; 2 * 8 * 4], &[2, 8], Dtype::U32).unwrap();
    let alpha = Array::from_f32_slice(&[0.5, 1.5], &[2]).unwrap();
    let (scales, biases) = expand_row_alpha(alpha, &packed, 128, Device::Cpu).unwrap();
    assert_eq!(scales.shape(), vec![2, 1]);
    assert_eq!(biases.shape(), vec![2, 1]);
    let s = to_f32_vec(&scales);
    let b = to_f32_vec(&biases);
    assert!((s[0] - 0.5).abs() < 1e-3, "scale[0]={}", s[0]);
    assert!((s[1] - 1.5).abs() < 1e-3, "scale[1]={}", s[1]);
    assert!((b[0] + s[0]).abs() < 1e-3, "bias is -scale");
    assert!((b[1] + s[1]).abs() < 1e-3, "bias is -scale");
}

fn plain(weight: Array) -> Linear {
    Linear::Plain { weight }
}

#[test]
fn concat_linears_axis0_stacks_qkv_output_rows() {
    let q = plain(Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
    let k = plain(Array::from_f32_slice(&[5.0, 6.0], &[1, 2]).unwrap());
    let v = plain(Array::from_f32_slice(&[7.0, 8.0], &[1, 2]).unwrap());
    let fused = concat_linears(&[q, k, v], 0, Device::Cpu).unwrap();
    let Linear::Plain { weight } = fused else {
        panic!("expected plain");
    };
    assert_eq!(weight.shape(), vec![4, 2]);
    let got = to_f32_vec(&weight);
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn concat_linears_axis1_stacks_expert_up_then_gate() {
    // [E=2, out=2, in=2] each.
    let up = plain(
        Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap(),
    );
    let gate = plain(
        Array::from_f32_slice(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[2, 2, 2])
            .unwrap(),
    );
    let fused = concat_linears(&[up, gate], 1, Device::Cpu).unwrap();
    let Linear::Plain { weight } = fused else {
        panic!("expected plain");
    };
    assert_eq!(weight.shape(), vec![2, 4, 2]);
}
