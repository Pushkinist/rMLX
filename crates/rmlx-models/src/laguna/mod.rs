//! Laguna (LagunaForCausalLM) forward pass.
//!
//! Architecture overview:
//! - 40 decoder layers. Layer 0 is dense; layers 1-39 are sparse MoE.
//! - Attention: GQA (num_heads varies per layer), per-head q/k RMSNorm,
//!   partial RoPE (factor 0.5 for full_attention, 1.0 for sliding), gating
//!   via softplus(g_proj(x)).
//! - Dense MLP: standard SwiGLU (gate_proj, up_proj, down_proj).
//! - Sparse MoE: router (gate.proj) + top-8 expert dispatch (switch_mlp,
//!   3-D batched) + shared dense expert (shared_expert).
//!
//! Reference snapshot: mlx-community__Laguna-XS.2-mxfp8
//! - 40 layers, hidden=2048, num_experts=256, moe_intermediate=512
//! - Global quant: mxfp8 g32 b8
//! - Per-tensor override: model.layers.N.mlp.gate.proj -> g64 b8
//!   The router weight also has .biases -> affine quant (mode="default").

#![allow(clippy::too_many_arguments)]

mod attention;
pub mod config;
mod decoder_layer;
mod generate;
mod layers;
pub mod loader;
mod model;
mod moe;
pub(crate) mod prompt_cache;
#[cfg(test)]
mod tests;

pub use config::{LagunaConfig, LayerKind, MlpKind};
pub use generate::generate_greedy;
pub use loader::load_from_path;
pub use model::LagunaText;
pub use prompt_cache::read_cache_stats as laguna_cache_stats;

/// Laguna's decoder layers each project their own K/V — no cross-layer-KV
/// topology. This is the single producer of that fact for this arch: it is what
/// [`crate::kv_cache::kv_layer_quants`] resolves the boundary-layer codec
/// against, what the prompt-cache seed folds, and what
/// `Architecture::shares_kv_across_layers` reports. It is `false`, which is also
/// `KvCache`'s constructor default, but it is named rather than spelled at each
/// site because the value now selects a codec: a boundary layer of a `Mixed` /
/// `RotK` base is promoted in-family only on a stack that keeps no bf16 mirror,
/// so a flipped literal would change decoded output, not just residency.
pub(crate) const SHARES_KV_ACROSS_LAYERS: bool = false;
