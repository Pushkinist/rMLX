//! Qwen3.5 MoE model — module root.
//!
//! Public surface (unchanged from the original flat file):
//! - [`Qwen3_5MoeConfig`] — architecture config
//! - [`Qwen3_5MoeText`] — model struct with `forward_seq*` / `forward_arr`
//! - [`generate_greedy`] — greedy decode with KV cache
//! - [`load_from_path`] — load MoE snapshot
//! - [`load_from_path_paro`] — load PARO (dense) snapshot
//!
//! The AWQ/F16 byte-math conversion functions now live in
//! [`rmlx_quant::awq`]; the PARO/embedding `Array`-side assembly lives in
//! [`crate::load_util`].

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
pub(crate) mod prompt_cache;

// ---------------------------------------------------------------------------
// Public re-exports (preserve crate::qwen3_5_moe::* paths)
// ---------------------------------------------------------------------------

pub use config::Qwen3_5MoeConfig;
pub use generate::generate_greedy;
pub use loader::{load_from_path, load_from_path_paro};
pub use model::Qwen3_5MoeText;
pub(crate) use mtp_layer::{MtpLayer, MtpLayerDims};
pub use prompt_cache::read_cache_stats as qwen3_5_moe_cache_stats;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;

/// Qwen3.5-MoE's decoder layers each project their own K/V — no cross-layer-KV
/// topology. This is the single producer of that fact for this arch: it is what
/// [`crate::kv_cache::kv_layer_quants`] resolves the boundary-layer codec
/// against, what the prompt-cache seed folds, and what
/// `Architecture::shares_kv_across_layers` reports. It is `false`, which is also
/// `KvCache`'s constructor default, but it is named rather than spelled at each
/// site because the value now selects a codec: a boundary layer of a `Mixed` /
/// `RotK` base is promoted in-family only on a stack that keeps no bf16 mirror,
/// so a flipped literal would change decoded output, not just residency.
pub(crate) const SHARES_KV_ACROSS_LAYERS: bool = false;
