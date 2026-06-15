//! Qwen3-VL-MoE (`Qwen3VLMoeForConditionalGeneration`, model_type `qwen3_vl_moe`).
//!
//! A vision-language model whose **text decoder is a plain Qwen3-MoE GQA**
//! stack (NOT the Qwen3-Next GatedDeltaNet hybrid in [`crate::qwen3_5_moe`]):
//! 48 layers, GQA with per-head q_norm/k_norm, full rotary, MoE every layer
//! (128 experts, top-8, no shared expert), `rope_theta = 5e6`. The vision tower
//! is the Qwen3-VL ViT (LayerNorm blocks, GELU-tanh MLP, learned pos-embed
//! interpolation, deepstack mergers) — architecturally distinct from the
//! Qwen2.5-VL ViT reused by jina-v4 ([`crate::jina_v4::vision`]), so it is built
//! fresh here rather than forked.
//!
//! ## Module layout
//!
//! - [`config`] — nested `text_config` / `vision_config` parser.
//! - [`mrope`] — **interleaved** 3D M-RoPE (`apply_interleaved_mrope` +
//!   `get_rope_index`). The interleaved section layout differs from the chunked
//!   layout in [`crate::jina_v4::image`]; see the module docs for the exact
//!   per-channel selection.
//! - [`image`] — vision-feature scatter at `image_token_id` + deepstack
//!   additive injection.
//!
//! ## Status
//!
//! **Text decoder: DONE + validated.** [`config`], [`mrope`], [`image`] (host
//! helpers) plus the weight-bearing text path — [`layers`] (quantized Linear /
//! Embedding with `gather_qmm` expert dispatch), [`attention`] (plain GQA with
//! 3D interleaved M-RoPE applied via precomputed cos/sin), [`moe`] (plain
//! SparseMoeBlock, no shared expert), [`model::Qwen3VlMoeText`], and
//! [`loader::load_text_from_path`] (enumerates on-disk shards, ignoring the
//! stale 13-shard index).
//!
//! Verified against `mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit`: greedy
//! decode of a text-only chat prompt reproduces the mlx-vlm 0.5.0 reference
//! token ids exactly (`[59604, 151645, 198, …]` = "Paris" + im_end). See
//! `tests/qwen3_vl_moe_text_parity.rs`.
//!
//! **Pending (next increment):** vision tower forward (Conv3d patch
//! embed, learned pos-embed interpolation, vision RoPE, 27 SDPA blocks, the
//! merger + 3 deepstack mergers), the image-branch generator (ViT → scatter at
//! image_token_id → deepstack inject into decoder layers 8/16/24), `arch.rs`
//! enum registration, and the server multipart image path. The vision-feature
//! scatter + deepstack-inject + 3D `get_rope_index` host helpers in [`image`]
//! and [`mrope`] are already implemented and unit-tested for that work.

pub(super) mod attention;
pub mod config;
pub mod generate;
pub(crate) mod image;
pub mod image_preprocess;
pub(super) mod layers;
pub mod loader;
pub mod model;
pub(super) mod moe;
pub(crate) mod mrope;
pub(crate) mod prompt_cache;
pub mod vision;

pub use config::{Qwen3VlMoeConfig, Qwen3VlMoeTextConfig, Qwen3VlMoeVisionConfig};
pub use generate::{generate_greedy, generate_image};
pub use image_preprocess::{preprocess, Qwen3VlImageConfig, Qwen3VlPixelValues};
pub use loader::{load_config_qwen3_vl, load_from_path, load_text_from_path};
pub use model::Qwen3VlMoe;
pub use model::Qwen3VlMoeText;
pub use prompt_cache::read_cache_stats as qwen3_vl_moe_cache_stats;
pub use prompt_cache::read_kv_cache_bytes as qwen3_vl_moe_kv_cache_bytes;
pub use vision::{load_vision_tower, Qwen3VlMoeVision, VisionOutput};
