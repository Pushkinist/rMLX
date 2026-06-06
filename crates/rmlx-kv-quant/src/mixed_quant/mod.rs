// unsafe_code: mlx-rs Array zero-copy view
#![allow(unsafe_code)]

//! Mixed-precision quantized KV cache.
//!
//! Byte-for-byte port of `mlx_lm/models/mixed_quant_cache.py::MixedQuantKVCache`
//! from `mlx-lm-turboquant`. Stores K and V as the canonical 3-tuple
//! `(codes_u32, scales, biases)` produced by `mx.quantize(..., mode="affine")`,
//! at independent bit widths and group sizes (default K=8 / V=4 / group=64 each).
//!
//! # Reference
//!
//! - `mlx_lm/models/mixed_quant_cache.py:67-111` (`update_and_fetch`).
//! - `mlx_lm/models/base.py:108-157` (`mixed_quantized_scaled_dot_product_attention`).

mod sdpa;
mod state;

pub use sdpa::{mixed_quantized_sdpa, rot_k_tq4v_sdpa};
pub use state::{MixedKvState, MixedTuple};
