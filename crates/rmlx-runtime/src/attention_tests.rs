use super::*;
use rmlx_mlx::Dtype;

fn bf16_zeros(shape: &[i32]) -> Array {
    let n = shape.iter().product::<i32>() as usize;
    let bytes = vec![0u8; n * 2];
    Array::from_bytes(&bytes, shape, Dtype::Bf16).unwrap()
}

#[test]
fn repeat_kv_passthrough_when_repeat_one() {
    let x = bf16_zeros(&[1, 2, 4, 8]);
    let y = repeat_kv(&x, 1, Device::Cpu).unwrap();
    assert_eq!(y.shape(), x.shape());
}

#[test]
fn repeat_kv_expands_kv_axis() {
    let x = bf16_zeros(&[1, 2, 4, 8]);
    let y = repeat_kv(&x, 4, Device::Cpu).unwrap();
    assert_eq!(y.shape(), vec![1, 8, 4, 8]);
}
