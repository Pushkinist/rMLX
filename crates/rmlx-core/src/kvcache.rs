//! KV-cache abstraction.
//!
//! KV-cache abstraction: dispatched as enum, not boxed trait, because per-layer
//! cache type is fixed at model-load time. See docs/04-rust-stack-options.md
//! and the mistral.rs `KvCache` enum for the same pattern.

/// What gets used at each transformer layer for K/V storage.
#[derive(Debug)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed enum — KV storage mode is a registry decision; adding a variant requires synchronized changes to the per-layer dispatch table"
)]
pub enum KvKind {
    /// Stock unquantized cache (bf16 / fp16 / fp32).
    Plain,
    /// FP8 (E4M3) blockwise.
    Fp8,
    /// TurboQuant rotation-KV with explicit per-side bit widths.
    Turbo {
        /// Quantisation bit-width applied to the K tensors.
        k_bits: u8,
        /// Quantisation bit-width applied to the V tensors.
        v_bits: u8,
    },
    /// 2D Givens rotation (PlanarQuant) with per-side bit widths.
    Planar {
        /// Quantisation bit-width applied to the K tensors.
        k_bits: u8,
        /// Quantisation bit-width applied to the V tensors.
        v_bits: u8,
    },
}
