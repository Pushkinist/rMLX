// unsafe_code: mask.rs uses Array::from_bytes via slice::from_raw_parts byte-reinterpretation.
// The allow is scoped to mask.rs; this file does not contain unsafe code.

//! Shared layer primitives reused across architecture modules.
//!
//! Types here are genuinely reused (or will be reused by the next architecture).
//! Anything Gemma4-specific stays in `gemma4.rs`.
//!
//! Included:
//! - `RmsNorm` — plain-gamma convention (Gemma4, Llama, Qwen).
//! - `Linear` — plain bf16 or quantized (affine-int / mxfp8).
//! - `Embedding` — plain or quantized; also usable as tied-weight lm_head.
//! - `Mlp` — dense gate-up-down FFN with pluggable activation.
//! - `MoeBlock` — stub (Stage 2 / Qwen3 territory).

#![allow(clippy::float_cmp, clippy::implicit_hasher)]

mod embedding;
mod linear;
mod mask;
mod mlp;
mod norm;
mod quant;

pub use embedding::Embedding;
pub use linear::{Linear, ParoRotation};
pub use mask::{
    build_chunked_prefill_mask, build_swa_decode_mask, build_swa_prefill_mask, pick_attn_mask_mode,
};
pub use mlp::{Activation, Mlp, MoeBlock};
pub use norm::RmsNorm;
pub use quant::{resolve_quant, QuantMode, QuantParams};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(unsafe_code)]
mod tests;
