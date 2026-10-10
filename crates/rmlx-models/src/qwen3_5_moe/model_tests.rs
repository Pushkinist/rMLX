//! The restricted-vocab head refuses a Hadamard head on `Device::Cpu`.
//!
//! Its gathered rows are folded (`W' = W D H`), so a matmul on the
//! untransformed hidden state gives wrong logits with no error.

use std::sync::Arc;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use super::build_cached_hot_head;
use crate::hadamard::hadamard_tests::signs;
use crate::hadamard::SignedBlockHadamard;
use crate::qwen3_5_moe::layers::{Embedding, Linear};

const WIDTH: usize = 512;
const ROWS: i32 = 8;

/// An affine 2-bit table of `ROWS` rows: weight, scales, biases.
fn table() -> Result<(Array, Array, Array)> {
    let groups = (WIDTH / 128) as i32;
    Ok((
        Array::from_bytes(
            &vec![0x5a_u8; ROWS as usize * WIDTH / 4],
            &[ROWS, (WIDTH / 16) as i32],
            Dtype::U32,
        )?,
        Array::from_f32_slice(&vec![0.5; (ROWS * groups) as usize], &[ROWS, groups])?,
        Array::from_f32_slice(&vec![-0.5; (ROWS * groups) as usize], &[ROWS, groups])?,
    ))
}

fn linear() -> Result<Linear> {
    let (weight, scales, biases) = table()?;
    Ok(Linear::Quantized {
        weight,
        scales,
        biases: Some(biases),
        group_size: 128,
        bits: 2,
        mode: "affine".to_owned(),
    })
}

fn embedding() -> Result<Embedding> {
    let (weight, scales, biases) = table()?;
    Ok(Embedding::Quantized {
        weight,
        scales,
        biases: Some(biases),
        group_size: 128,
        bits: 2,
        mode: "affine".to_owned(),
    })
}

fn transform() -> Result<Arc<SignedBlockHadamard>> {
    Ok(Arc::new(SignedBlockHadamard::new(
        WIDTH,
        &signs(4600, WIDTH),
    )?))
}

#[test]
fn the_hot_vocab_head_refuses_a_hadamard_head() -> Result<()> {
    let hot_ids = Array::from_i32_slice(&[1, 4, 6], &[3])?;
    let refused = |r: &Result<super::CachedHotHead>| matches!(r, Err(Error::Model(m)) if m.contains("Hadamard"));

    let head = Some(Linear::hadamard(transform()?, linear()?)?);
    let r = build_cached_hot_head(&head, &embedding()?, &hot_ids, Device::Cpu);
    assert!(refused(&r), "a Hadamard lm_head");

    let tied = Embedding::hadamard(transform()?, embedding()?)?;
    let r = build_cached_hot_head(&None, &tied, &hot_ids, Device::Cpu);
    assert!(refused(&r), "a Hadamard embedding as the tied head");

    let r = build_cached_hot_head(&Some(linear()?), &embedding()?, &hot_ids, Device::Cpu);
    assert!(
        r.is_ok(),
        "control: the quantized lm_head of the same table"
    );
    let r = build_cached_hot_head(&None, &embedding()?, &hot_ids, Device::Cpu);
    assert!(r.is_ok(), "control: the quantized tied embedding");
    Ok(())
}
