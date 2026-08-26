//! Maple (`MapleForCausalLM`) — DeepGrove 20B-A1B ternary MoE.
//!
//! Reference: mlx-lm-deepgrove `mlx_lm/models/maple.py` and the
//! `maple-2bit-mlx` snapshot. v1 is the portable reference forward
//! (no FlashHead, no fused Metal kernels).

pub(crate) mod attention;
pub(crate) mod config;
pub(crate) mod decoder_layer;
pub(crate) mod generate;
pub(crate) mod loader;
pub(crate) mod model;
pub(crate) mod moe;
pub(crate) mod prompt_cache;

pub use config::MapleConfig;
pub use generate::generate_greedy;
pub use loader::load_from_path;
pub use model::MapleText;
pub use prompt_cache::read_cache_stats as maple_cache_stats;
