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
pub use prompt_cache::read_kv_cache_bytes as qwen2_kv_cache_bytes;
