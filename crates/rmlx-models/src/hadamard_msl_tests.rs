//! The signed block Hadamard kernel against the MLX op chain, both on the GPU.
//!
//! The bound: per element, the kernel's bf16 output equals the chain's or is
//! one bf16 ulp from it. Both run the butterfly in f32 and round once, so they
//! can differ only where the two f32 sums are ordered differently.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use super::signed_block_hadamard_gpu;
use crate::hadamard::hadamard_tests::{signs, uniform};
use crate::hadamard::{Direction, SignedBlockHadamard};

const BLOCK: usize = 1024;
/// The input widths of the Ternary-Bonsai-2-27B pack.
const PACK_WIDTHS: [usize; 3] = [5120, 6144, 17408];
const MAX_ULP: u32 = 1;

fn bf16_bits(a: &Array) -> Result<Vec<u16>> {
    if a.dtype() != Dtype::Bf16 {
        return Err(Error::Model(format!("want bf16, got {:?}", a.dtype())));
    }
    a.eval()?;
    Ok(a.to_bytes()?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect())
}

/// The bf16 bit pattern on a line where adjacent values are one apart and
/// both zeros meet.
fn ordered(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7fff);
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Max ulp distance and the share of elements that are not equal.
fn ulp_stats(got: &[u16], want: &[u16]) -> (u32, f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let distances: Vec<u32> = got
        .iter()
        .zip(want)
        .map(|(g, w)| ordered(*g).abs_diff(ordered(*w)))
        .collect();
    let max = distances.iter().copied().max().unwrap_or(0);
    let unequal = distances.iter().filter(|d| **d != 0).count();
    (max, unequal as f64 / distances.len() as f64)
}

fn other(direction: Direction) -> Direction {
    match direction {
        Direction::Forward => Direction::Inverse,
        Direction::Inverse => Direction::Forward,
    }
}

/// At each pack width, block 1024, rows 1 and 64, both directions: the kernel
/// is within one bf16 ulp of the op chain at every element, and
/// `forward` / `inverse` on the GPU return the kernel's bits. The control, the
/// kernel in the other direction, misses the bound on most elements.
#[test]
#[ignore = "Metal: dispatches the signed block Hadamard kernel on Device::Gpu; needs the Metal claim (make gpu-test)"]
fn the_signed_block_hadamard_kernel_is_within_one_bf16_ulp_of_the_op_chain_on_gpu() -> Result<()> {
    let device = Device::Gpu;
    let scale = Array::from_f32_slice(&[1.0 / 32.0], &[])?;
    for (n, &width) in PACK_WIDTHS.iter().enumerate() {
        let s = signs(1100 + n as u64, width);
        let t = SignedBlockHadamard::new(BLOCK, &s)?;
        let sign_input = Array::from_f32_slice(&s, &[width as i32])?;
        for rows in [1, 64] {
            let x = Array::from_f32_slice(
                &uniform(1200 + n as u64, rows * width),
                &[rows as i32, width as i32],
            )?
            .astype(Dtype::Bf16, device)?;
            for direction in [Direction::Forward, Direction::Inverse] {
                let label = format!("width {width} rows {rows} {direction:?}");
                let kernel =
                    signed_block_hadamard_gpu(&x, &sign_input, &scale, BLOCK, direction, device)?;
                assert_eq!(kernel.shape(), x.shape(), "{label}: shape");
                let got = bf16_bits(&kernel)?;
                let chain = bf16_bits(&t.op_chain(direction, &x, device)?)?;
                let (max_ulp, unequal) = ulp_stats(&got, &chain);
                println!("{label}: max ulp {max_ulp}, unequal share {unequal:.6}");
                assert!(
                    max_ulp <= MAX_ULP,
                    "{label}: the kernel is {max_ulp} bf16 ulp from the op chain (bound {MAX_ULP})"
                );

                let public = match direction {
                    Direction::Forward => t.forward(&x, device)?,
                    Direction::Inverse => t.inverse(&x, device)?,
                };
                assert_eq!(
                    bf16_bits(&public)?,
                    got,
                    "{label}: the GPU path is not the kernel"
                );

                let control = bf16_bits(&signed_block_hadamard_gpu(
                    &x,
                    &sign_input,
                    &scale,
                    BLOCK,
                    other(direction),
                    device,
                )?)?;
                let (control_ulp, control_unequal) = ulp_stats(&control, &chain);
                assert!(
                    control_ulp > MAX_ULP && control_unequal > 0.5,
                    "{label}: the other direction is only {control_ulp} ulp from the chain on \
                     {control_unequal:.4} of elements; the bound cannot reject it"
                );
            }
        }
    }
    Ok(())
}
