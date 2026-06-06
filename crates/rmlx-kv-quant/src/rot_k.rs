// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! K-side rotation codec — the "pre-rotate-Q" trick.
//!
//! # The math
//!
//! Attention scores are `Q · Kᵀ`. Insert an orthogonal rotation `R`
//! (`Rᵀ R = I`, `R` square `[D, D]`) into the K basis and pre-rotate Q by the
//! *same* `R`:
//!
//! ```text
//! (Q Rᵀ) · (K Rᵀ)ᵀ = (Q Rᵀ) · (R Kᵀ) = Q (Rᵀ R) Kᵀ = Q Kᵀ
//! ```
//!
//! So if we store **rotated** K (`K_rot = K Rᵀ`) in the cache and **pre-rotate**
//! the queries (`Q_rot = Q Rᵀ`) before the score matmul, the two rotations
//! cancel and the attention scores are *identical* to the unrotated
//! computation — up to the quantization error on `K_rot`.
//!
//! The win: a Hadamard rotation decorrelates the K channels and equalizes their
//! dynamic range, so affine quantization in the rotated basis incurs lower PPL
//! than affine quantization of raw K (the RotorQuant / TurboQuant / PlanarQuant
//! insight — see `../../rotorquant/README.md`).
//!
//! **K is NEVER inverse-rotated.** Unlike V-side rotation (where the attention
//! output must be un-rotated back to the value basis), the K rotation is
//! cancelled algebraically by the pre-rotated Q. K stays quantized in the
//! rotated basis for the entire cache lifetime.
//!
//! # v1 implementation (mx-ops path)
//!
//! The rotation is a single `matmul` against a precomputed orthogonal matrix
//! `R` of shape `[D, D]`. Applied to K just before `mx.quantize` (encode) and
//! to Q just before the score `quantized_matmul` (decode/prefill). Both are
//! plain MLX array ops — correct and coherent.
//!
//! # sub-item 1: fused FWHT Metal kernel
//!
//! `rot_k_msl.rs` (same directory) implements the fused path: Fast Walsh-Hadamard
//! Transform (FWHT) + affine 8-bit quantize in one Metal kernel pass. The FWHT
//! is O(D log₂D) vs the matmul's O(D²), and eliminates the intermediate `K_rot`
//! DRAM allocation. Opt-in via `RMLX_ROT_K_FUSED=1`; falls back to this
//! matmul path on unsupported D or error. The helper functions below
//! (`hadamard_rotation`, `rotate_last_axis`) remain the v1 reference path
//! and the fallback — they are not removed.
//!
//! `R` is a **normalized Walsh–Hadamard matrix** (`H_D / sqrt(D)`), which is
//! orthogonal and exists for any power-of-two `D` (Bonsai head_dim = 128). It
//! is the standard decorrelating transform of the rotation-KV family.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{matmul, Array, Device, Dtype};

/// Build the normalized Walsh–Hadamard rotation matrix `R = H_D / sqrt(D)`.
///
/// `R` is symmetric and orthogonal (`R = Rᵀ`, `R R = I`), shape `[d, d]`,
/// dtype `dtype` on `device`. Requires `d` to be a power of two (the Sylvester
/// construction); returns an error otherwise so the caller can reject rather
/// than silently producing a non-orthogonal matrix.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn hadamard_rotation(d: usize, dtype: Dtype, device: Device) -> Result<Array> {
    if d == 0 || !d.is_power_of_two() {
        return Err(Error::Mlx(format!(
            "rot_k: head_dim={d} is not a power of two; the Walsh-Hadamard \
             rotation requires a power-of-two dimension. Use an affine K codec \
             (--ctk q8_g128) for this head_dim."
        )));
    }
    // Sylvester construction of H_d (entries ±1), then scale by 1/sqrt(d) to
    // make it orthonormal. Built on CPU as an f32 buffer, then uploaded.
    let mut h = vec![1.0f32; d * d];
    let mut m = 1usize;
    while m < d {
        for i in 0..m {
            for j in 0..m {
                let a = h[i * d + j];
                h[i * d + (j + m)] = a;
                h[(i + m) * d + j] = a;
                h[(i + m) * d + (j + m)] = -a;
            }
        }
        m *= 2;
    }
    let inv_sqrt = 1.0f32 / (d as f32).sqrt();
    for v in &mut h {
        *v *= inv_sqrt;
    }
    let bytes = unsafe { std::slice::from_raw_parts(h.as_ptr().cast::<u8>(), h.len() * 4) };
    let r = Array::from_bytes(bytes, &[d as i32, d as i32], Dtype::F32)?;
    if dtype == Dtype::F32 {
        Ok(r)
    } else {
        r.astype(dtype, device)
    }
}

/// Apply the rotation `R` to the last axis of a `[..., D]` tensor: `x @ R`.
///
/// Since `R` is symmetric (`R = Rᵀ`), the same matrix rotates both K (`K Rᵀ`)
/// and Q (`Q Rᵀ`). `x` is flattened to 2-D `[N, D]` for the matmul, then
/// reshaped back — keeps the contraction dimension explicit and the op cheap.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(super) fn rotate_last_axis(x: &Array, r: &Array, device: Device) -> Result<Array> {
    let shape = x.shape();
    let d = *shape.last().expect("rotate_last_axis: empty shape");
    let n: i32 = shape.iter().product::<i32>() / d;
    let r2 = if r.dtype() == x.dtype() {
        r.try_clone()?
    } else {
        r.astype(x.dtype(), device)?
    };
    let flat = x.reshape(&[n, d], device)?;
    let rotated = matmul(&flat, &r2, device)?;
    rotated.reshape(&shape, device)
}

#[cfg(test)]
#[path = "rot_k_tests.rs"]
mod tests;
