//! Gemma 3 text-only forward pass.
//!
//! Reference: `mlx-community__medgemma-1.5-4b-it-8bit`
//! (34 layers, hidden=2560, heads=8/4, head_dim=256, vocab=262208).
//! Audio + vision tensors in the snapshot are silently ignored.
//!
//! # Key differences from Gemma4
//!
//! | Feature | Gemma3 (this file) | Gemma4 |
//! |---------------------------|---------------------------------|---------------------|
//! | RMSNorm gamma | `gamma + 1` (weight+1 offset) | plain gamma |
//! | Attention scale | `query_pre_attn_scalar^-0.5` | 1.0 |
//! | RoPE | full rotation, `dims=head_dim` | ProportionalRoPE |
//! | KV sharing | none | num_kv_shared_layers|
//! | Per-layer input gating | none | yes |
//! | MoE | none | optional |
//! | Final logit softcapping | optional (null in medgemma) | always 30.0 |
//! | lm_head | separate tensor or tied | tied |
//!
//! # Gamma + 1 convention (RmsNormShifted)
//!
//! Gemma3 stores norm weights initialised at 0.0, so the effective scale
//! is `1.0 + weight`. Applied via `RmsNormShifted` defined in layers.rs;
//! not added to shared `layers.rs` (Gemma3-specific detail).
//! Reference: `gemma3_text.py` class `RMSNorm.__call__` line 111:
//! `return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)`
//!
//! # Snapshot quirk: sibling tensors not in the shard index
//!
//! `mlx-community__medgemma-1.5-4b-it-8bit` stores `.scales` and `.biases`
//! inside the shard file headers but does NOT list them in
//! `model.safetensors.index.json`. The standard `view()` function fails for
//! those names. We use `view_any_shard()` (defined in loader.rs) which scans
//! all open shards directly.

#![allow(clippy::too_many_arguments)]

mod attention;
mod config;
mod decoder_layer;
mod generate;
mod layers;
pub mod loader;
mod model;
pub(crate) mod prompt_cache;
#[cfg(test)]
mod tests;
mod vision;

pub use config::{Gemma3TextConfig, Gemma3VisionConfig, LayerType};
pub use generate::{generate_greedy, probe_forward};
pub use loader::load_from_path;
pub use model::Gemma3Text;
pub use prompt_cache::read_cache_stats as gemma3_cache_stats;
pub use vision::{
    build_inputs_embeds, load_vision_tower, Gemma3ImageProcessor, Gemma3PixelValues,
    MultiModalProjector, VisionModel, BOI_TOKEN_ID, EOI_TOKEN_ID, IMAGE_TOKEN_ID,
};
