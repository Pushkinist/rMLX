//! BitNet b1.58 text-only forward pass.
//!
//! Architecture: `BitNetForCausalLM` (ternary-weight, no MoE).
//!
//! Reference snapshot: `mlx-community__bitnet-b1.58-2B-4T`
//! (30 layers, hidden=2560, heads=20/5, head_dim=128, vocab=128256).
//!
//! # Key design points
//!
//! 1. **Ternary weights (int2 packed as U8)**. Each linear weight is stored as
//!    U8 with 4 trits/byte (`N//4` rows). Alongside each weight tensor is a
//!    single BF16 scalar `weight_scale`. At load time the U8 is dequantized to
//!    BF16 once — the dequant path produces `{-1, 0, +1} * weight_scale * row_scale`,
//!    where row_scale is derived from `activation_quant` (see comment in model.rs).
//!    This keeps the forward pass as a plain BF16 matmul (no custom Metal kernel
//!    required for correctness, performance can be improved later).
//!
//! 2. **Sub-norms (attn_sub_norm / ffn_sub_norm)**. Inserted after the attention
//!    output but before `o_proj`, and after `relu2(gate)*up` but before `down_proj`.
//!    These are plain RMSNorm layers (eps=1e-5) with weights named
//!    `self_attn.attn_sub_norm.weight` and `mlp.ffn_sub_norm.weight`.
//!
//! 3. **Relu2 activation** (`max(x, 0)^2`). Registered in `layers::Activation::Relu2`.
//!
//! 4. **Tied LM head** (`tie_word_embeddings=true`). The embedding weight is shared
//!    as LM head.
//!
//! # Tensor prefix
//!
//! All model weights use the prefix `model.` (e.g. `model.layers.N.self_attn.q_proj.weight`).

#![allow(clippy::too_many_arguments)]

mod config;
mod generate;
mod loader;
mod model;
pub(crate) mod prompt_cache;

pub use config::BitNetConfig;
pub use generate::generate_greedy;
pub use loader::load_from_path;
pub use model::BitNetText;
pub use prompt_cache::read_cache_stats as bitnet_cache_stats;
pub use prompt_cache::read_kv_cache_bytes as bitnet_kv_cache_bytes;

#[cfg(test)]
#[path = "tests.rs"]
mod bitnet_tests;
