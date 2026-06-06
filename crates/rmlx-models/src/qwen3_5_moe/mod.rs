//! Qwen3.5 MoE model — module root.
//!
//! Public surface (unchanged from the original flat file):
//! - [`Qwen3_5MoeConfig`] — architecture config
//! - [`Qwen3_5MoeText`] — model struct with `forward_seq*` / `forward_arr`
//! - [`generate_greedy`] — greedy decode with KV cache
//! - [`load_from_path`] — load MoE snapshot
//! - [`load_from_path_paro`] — load PARO (dense) snapshot
//!
//! `pub(crate)` helpers re-exported for `gemma4.rs`:
//! AWQ/F16 conversion functions from `loader`.

pub(super) mod attention;
pub(super) mod config;
pub(super) mod decoder_layer;
pub(super) mod gated_delta_net;
pub(super) mod generate;
pub(super) mod layers;
pub(super) mod loader;
pub(super) mod model;
pub(super) mod moe;
pub(crate) mod mtp_layer;
pub(super) mod prompt_cache;

// ---------------------------------------------------------------------------
// Public re-exports (preserve crate::qwen3_5_moe::* paths)
// ---------------------------------------------------------------------------

pub use config::Qwen3_5MoeConfig;
pub use generate::generate_greedy;
pub use loader::{load_from_path, load_from_path_paro};
pub use model::Qwen3_5MoeText;
pub(crate) use mtp_layer::{MtpLayer, MtpLayerDims};
pub(crate) use prompt_cache::attach_ssd_tier;
pub use prompt_cache::read_cache_stats as qwen3_5_moe_cache_stats;
pub use prompt_cache::read_kv_cache_bytes as qwen3_5_moe_kv_cache_bytes;

// pub(crate) re-exports — used by gemma4.rs and crate tests.
pub(crate) use loader::{
    convert_awq_qweight, convert_awq_qzeros_to_biases, f16_bits_to_f32, f32_to_f16_bits,
    quantize_f16_affine_int4,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
