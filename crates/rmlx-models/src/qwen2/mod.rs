// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(trivial_numeric_casts, trivial_casts)]

//! Qwen2 text-only forward pass.
//!
//! Architecture: `Qwen2ForCausalLM` (dense, no MoE).
//!
//! Reference snapshot: `mlx-community__jinaai-ReaderLM-v2`
//! (28 layers, hidden=1536, heads=12/2, head_dim=128, vocab=151936, g64 b4 affine).
//!
//! # Key Qwen2 properties
//! - Plain RMSNorm (plain-gamma, no +1 shift).
//! - Attention projections have **additive bias** (q/k/v_proj carry a `.bias` tensor).
//! - No per-head q/k norms (those are Qwen3).
//! - Full RoPE over the whole head_dim.
//! - SwiGLU MLP.
//! - Config fields live at the **root** of config.json (no `text_config` nesting).

pub(crate) mod config;
pub(crate) mod generate;
pub(crate) mod loader;
pub(crate) mod model;
pub(crate) mod prompt_cache;

#[cfg(test)]
mod tests;

pub use config::Qwen2Config;
pub use generate::generate_greedy;
pub use loader::load_from_path;
pub use model::Qwen2Text;
pub use prompt_cache::read_cache_stats as qwen2_cache_stats;

/// Qwen2's decoder layers each project their own K/V — no cross-layer-KV
/// topology. This is the single producer of that fact for this arch: it is what
/// [`crate::kv_cache::kv_layer_quants`] resolves the boundary-layer codec
/// against, what the prompt-cache seed folds, and what
/// `Architecture::shares_kv_across_layers` reports. It is `false`, which is also
/// `KvCache`'s constructor default, but it is named rather than spelled at each
/// site because the value now selects a codec: a boundary layer of a `Mixed` /
/// `RotK` base is promoted in-family only on a stack that keeps no bf16 mirror,
/// so a flipped literal would change decoded output, not just residency.
pub(crate) const SHARES_KV_ACROSS_LAYERS: bool = false;
