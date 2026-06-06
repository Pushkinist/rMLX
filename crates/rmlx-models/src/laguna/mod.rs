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
#[cfg(test)]
mod tests;

pub use config::{LagunaConfig, LayerKind, MlpKind};
pub use generate::generate_greedy;
pub use loader::load_from_path;
pub use model::LagunaText;
