//! The signed block Hadamard kernel against the MLX op chain, both on the GPU.
//!
//! The bound: per element, the kernel's output equals the chain's or is one
//! ulp of the output dtype from it. Both run the butterfly in f32 and round
//! once, so they can differ only where the two f32 sums are ordered
//! differently.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use super::signed_block_hadamard_gpu;
use crate::hadamard::hadamard_tests::{signs, uniform};
use crate::hadamard::{Direction, SignedBlockHadamard, MAX_BLOCK};

const BLOCK: usize = 1024;
/// The input widths of the Ternary-Bonsai-2-27B pack.
const PACK_WIDTHS: [usize; 3] = [5120, 6144, 17408];
const MAX_ULP: u64 = 1;

/// Each element's bit pattern, widened to `u32`, and the dtype's sign bit.
fn bits(a: &Array) -> Result<(Vec<u32>, u32)> {
    a.eval()?;
    let bytes = a.to_bytes()?;
    match a.dtype() {
        Dtype::Bf16 => Ok((
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u32::from(u16::from_le_bytes(*c)))
                .collect(),
            0x8000,
        )),
        Dtype::F32 => Ok((
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect(),
            0x8000_0000,
        )),
        other @ (Dtype::F16 | Dtype::U8 | Dtype::U32 | Dtype::I32) => {
            Err(Error::Model(format!("want bf16 or f32, got {other:?}")))
        }
    }
}

/// A bit pattern on a line where adjacent values of its dtype are one apart
/// and both zeros meet.
fn ordered(bits: u32, sign: u32) -> i64 {
    let magnitude = i64::from(bits & (sign - 1));
    if bits & sign == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Max ulp distance and the share of elements that are not equal.
fn ulp_stats(got: &[u32], want: &[u32], sign: u32) -> (u64, f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let distances: Vec<u64> = got
        .iter()
        .zip(want)
        .map(|(g, w)| ordered(*g, sign).abs_diff(ordered(*w, sign)))
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
                let (got, sign) = bits(&kernel)?;
                let (chain, _) = bits(&t.op_chain(direction, &x, device)?)?;
                let (max_ulp, unequal) = ulp_stats(&got, &chain, sign);
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
                    bits(&public)?.0,
                    got,
                    "{label}: the GPU path is not the kernel"
                );

                let (control, _) = bits(&signed_block_hadamard_gpu(
                    &x,
                    &sign_input,
                    &scale,
                    BLOCK,
                    other(direction),
                    device,
                )?)?;
                let (control_ulp, control_unequal) = ulp_stats(&control, &chain, sign);
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

/// Every block the kernel takes, 2 to [`MAX_BLOCK`], f32 and bf16, both
/// directions, over 3 rows of width `2 * block`, so the sign offset wraps once
/// per row: the kernel is within one ulp of the op chain at every element, and
/// the GPU `forward` / `inverse` return the kernel's bits.
#[test]
#[ignore = "Metal: dispatches the signed block Hadamard kernel on Device::Gpu; needs the Metal claim (make gpu-test)"]
fn the_signed_block_hadamard_kernel_is_within_one_ulp_of_the_op_chain_at_every_block_on_gpu(
) -> Result<()> {
    const ROWS: usize = 3;
    let device = Device::Gpu;
    let blocks: Vec<usize> = (1..=MAX_BLOCK.trailing_zeros())
        .map(|k| 1usize << k)
        .collect();
    assert_eq!(
        (blocks.len(), blocks.last()),
        (12, Some(&MAX_BLOCK)),
        "the sweep must cover every block from 2 to MAX_BLOCK"
    );
    for block in blocks {
        let width = 2 * block;
        let s = signs(1300 + block as u64, width);
        let t = SignedBlockHadamard::new(block, &s)?;
        let sign_input = Array::from_f32_slice(&s, &[width as i32])?;
        let scale = Array::from_f32_slice(&[(block as f32).sqrt().recip()], &[])?;
        let input = Array::from_f32_slice(
            &uniform(1400 + block as u64, ROWS * width),
            &[ROWS as i32, width as i32],
        )?;
        for dtype in [Dtype::F32, Dtype::Bf16] {
            let x = input.astype(dtype, device)?;
            for direction in [Direction::Forward, Direction::Inverse] {
                let label = format!("block {block} {dtype:?} {direction:?}");
                let kernel =
                    signed_block_hadamard_gpu(&x, &sign_input, &scale, block, direction, device)?;
                assert_eq!(
                    (kernel.shape(), kernel.dtype()),
                    (x.shape(), dtype),
                    "{label}: shape and dtype"
                );
                let (got, sign) = bits(&kernel)?;
                let (chain, _) = bits(&t.op_chain(direction, &x, device)?)?;
                let (max_ulp, unequal) = ulp_stats(&got, &chain, sign);
                println!("{label}: max ulp {max_ulp}, unequal share {unequal:.6}");
                assert!(
                    max_ulp <= MAX_ULP,
                    "{label}: the kernel is {max_ulp} ulp from the op chain (bound {MAX_ULP})"
                );

                let public = match direction {
                    Direction::Forward => t.forward(&x, device)?,
                    Direction::Inverse => t.inverse(&x, device)?,
                };
                assert_eq!(
                    bits(&public)?.0,
                    got,
                    "{label}: the GPU path is not the kernel"
                );
            }
        }
    }
    Ok(())
}
