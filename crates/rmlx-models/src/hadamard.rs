//! Signed block Walsh-Hadamard transform of a Prism Hadamard pack.
//!
//! A pack stores each folded weight as `W D H`: `D` is the diagonal of a
//! `±1` sign vector over the full input width and `H` is the block-diagonal
//! normalised Sylvester Hadamard matrix (`1 / sqrt(block)` per block). The
//! runtime computes `y = W' . H (D x)` before a folded linear layer
//! ([`SignedBlockHadamard::forward`]) and `e = D H e'` after a folded
//! embedding lookup ([`SignedBlockHadamard::inverse`]). Both directions compute
//! in f32 and return the input dtype.
//!
//! The GPU runs each transform as one Metal kernel (`hadamard_msl`). The CPU
//! runs it as a chain of MLX ops (`SignedBlockHadamard::op_chain`): MLX's
//! custom kernels have no CPU backend, and the chain is the oracle the
//! kernel's GPU test holds it to.
//!
//! The transform keys on `(block, width, signs)` only and names no
//! architecture.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{hadamard_transform, multiply, Array, Device, Dtype};

use crate::hadamard_msl::signed_block_hadamard_gpu;

/// The largest block the kernel is checked at, and the largest a pack
/// contract allows.
pub const MAX_BLOCK: usize = 4096;

/// Which side of the transform the signs are applied on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    /// `H (D x)`, before a folded linear layer.
    Forward,
    /// `D (H x)`, after a folded embedding lookup.
    Inverse,
}

#[cfg(test)]
thread_local! {
    /// [`SignedBlockHadamard::forward`] calls made on this thread: what a test
    /// reads to count the transforms one forward pass builds.
    pub(crate) static FORWARD_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One signed block Hadamard transform: a block size and one sign vector that
/// covers the full input width.
#[derive(Debug)]
pub struct SignedBlockHadamard {
    block: usize,
    width: usize,
    /// `[width]` f32, every entry `+1` or `-1`.
    signs: Array,
    /// 0-D f32 `1 / sqrt(block)`, the kernel's scale input.
    scale: Array,
}

impl SignedBlockHadamard {
    /// Build the transform for `signs.len()` input columns.
    ///
    /// # Errors
    /// `Error::Model` when `block` is not a power of two from 2 to
    /// [`MAX_BLOCK`], when `signs` is empty or its length is not a multiple of
    /// `block`, or when a sign is not `+1` or `-1`.
    pub fn new(block: usize, signs: &[f32]) -> Result<Self> {
        if !block.is_power_of_two() || !(2..=MAX_BLOCK).contains(&block) {
            return Err(Error::Model(format!(
                "Hadamard block {block} is not a power of two from 2 to {MAX_BLOCK}"
            )));
        }
        let width = signs.len();
        if width == 0 || !width.is_multiple_of(block) {
            return Err(Error::Model(format!(
                "Hadamard sign width {width} is not a positive multiple of block {block}"
            )));
        }
        let is_sign =
            |v: f32| v.to_bits() == 1.0f32.to_bits() || v.to_bits() == (-1.0f32).to_bits();
        if let Some((at, v)) = signs.iter().enumerate().find(|(_, v)| !is_sign(**v)) {
            return Err(Error::Model(format!(
                "Hadamard sign {at} of {width} is {v}, not +1 or -1"
            )));
        }
        let shape = i32::try_from(width)
            .map_err(|_| Error::Model(format!("Hadamard sign width {width} exceeds i32")))?;
        Ok(Self {
            block,
            width,
            signs: Array::from_f32_slice(signs, &[shape])?,
            scale: Array::from_f32_slice(&[normalisation(block)], &[])?,
        })
    }

    /// The input width: the length of the sign vector.
    pub fn width(&self) -> usize {
        self.width
    }

    /// `H (D x)` over the last axis of `x`, in f32, returned in `x`'s dtype.
    ///
    /// # Errors
    /// `Error::Model` when the last axis of `x` is not the sign width;
    /// `Error::Mlx` when the GPU dispatch fails.
    pub fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        #[cfg(test)]
        FORWARD_CALLS.with(|n| n.set(n.get() + 1));
        self.apply(Direction::Forward, x, device)
    }

    /// `D (H x)` over the last axis of `x`, in f32, returned in `x`'s dtype.
    ///
    /// # Errors
    /// `Error::Model` when the last axis of `x` is not the sign width;
    /// `Error::Mlx` when the GPU dispatch fails.
    pub fn inverse(&self, x: &Array, device: Device) -> Result<Array> {
        self.apply(Direction::Inverse, x, device)
    }

    fn apply(&self, direction: Direction, x: &Array, device: Device) -> Result<Array> {
        match device {
            Device::Gpu => {
                self.input_shape(x)?;
                signed_block_hadamard_gpu(
                    x,
                    &self.signs,
                    &self.scale,
                    self.block,
                    direction,
                    device,
                )
            }
            Device::Cpu => self.op_chain(direction, x, device),
        }
    }

    /// The transform as MLX ops: widen to f32, apply the signs and
    /// `hadamard_transform`, round to `x`'s dtype: four kernels where the
    /// Metal kernel is one.
    ///
    /// # Errors
    /// `Error::Model` when the last axis of `x` is not the sign width.
    pub(crate) fn op_chain(
        &self,
        direction: Direction,
        x: &Array,
        device: Device,
    ) -> Result<Array> {
        let shape = self.input_shape(x)?;
        // Explicit on purpose: the f32 signs would promote `x` anyway.
        let wide = x.astype(Dtype::F32, device)?;
        let transformed = match direction {
            Direction::Forward => {
                self.blockwise(&multiply(&wide, &self.signs, device)?, &shape, device)?
            }
            Direction::Inverse => {
                multiply(&self.blockwise(&wide, &shape, device)?, &self.signs, device)?
            }
        };
        transformed.astype(x.dtype(), device)
    }

    fn input_shape(&self, x: &Array) -> Result<Vec<i32>> {
        let shape = x.shape();
        let last = shape.last().copied().unwrap_or(0);
        if usize::try_from(last).ok() != Some(self.width) {
            return Err(Error::Model(format!(
                "Hadamard input width {last} is not the sign width {}",
                self.width
            )));
        }
        Ok(shape)
    }

    /// The normalised transform of every `block` run of the last axis.
    fn blockwise(&self, x: &Array, shape: &[i32], device: Device) -> Result<Array> {
        let block = self.block as i32;
        let blocks = x.reshape(&[-1, block], device)?;
        hadamard_transform(&blocks, Some(normalisation(self.block)), device)?.reshape(shape, device)
    }
}

/// `1 / sqrt(block)`, the scale of the normalised Sylvester matrix.
fn normalisation(block: usize) -> f32 {
    (block as f32).sqrt().recip()
}

#[cfg(test)]
#[path = "hadamard_tests.rs"]
pub(crate) mod hadamard_tests;
