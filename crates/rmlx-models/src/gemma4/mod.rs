//! Gemma 4 text-only forward pass.
//!
//! Reference: `mlx-community__gemma-4-e4b-it-mxfp8` (42 layers, 4B params).
//! The text forward is the core path; vision (SigLIP) and audio (Conformer)
//! towers are additive modules feeding the multimodal embedder.
//!
//! Architecture summary
//! --------------------
//! - 42 decoder layers; layer type pattern: 5×sliding + 1×full (repeating).
//! - Sliding attention: head_dim=256, rope_theta=10000, full sliding window.
//! - Full attention: head_dim=512, rope_theta=1_000_000, partial rope (dims=128).
//! - GQA 4:1 (q_heads=8, kv_heads=2).
//! - KV sharing: layers 24-41 reuse KV from the last non-shared layer of the
//!   same attention type. Those layers still have on-disk k_proj/v_proj weights
//!   (not dropped by sanitize on this snapshot), but we skip loading them and
//!   pass shared KV forward.
//! - Per-layer input gating: present on all 42 layers.
//! - Weight-tied embeddings (no separate lm_head tensor).
//! - Final logit softcapping: tanh(logits / 30) * 30.
//! - MoE block: disabled (enable_moe_block=false for 4B model).
//!
//! Vision (SigLIP tower) and audio (Conformer tower) branches are additive
//! modules used only by the multimodal embedder path; the text forward is
//! unchanged.

pub mod audio;
pub mod audio_feature_extractor;
mod config;
mod decoder_layer;
mod generate;
mod layers;
mod loader;
mod model;
pub mod preprocessor;
pub(crate) mod prompt_cache;
#[cfg(test)]
mod tests;
mod vision;

pub use audio::unified::{
    build_unified_audio_inputs_embeds, extract_waveform_frames, load_unified_audio_embedder,
    unified_num_audio_soft_tokens, UnifiedAudioConfig, UnifiedAudioEmbedder,
};
pub use audio::{build_audio_inputs_embeds, load_audio_tower, AudioEncoder};
pub use audio_feature_extractor::{
    AudioFeatError, AudioFeatureExtractorConfig, Gemma4AudioFeatureExtractor,
};
pub use config::{Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig, LayerType};
pub use generate::{classify_smoke, generate_greedy, ProbeStep, SmokeVerdict};
pub(crate) use layers::build_proportional_rope_freqs;
pub use loader::{load_from_path, load_from_path_paro, probe_forward};
pub use model::Gemma4Text;
pub use preprocessor::{
    aspect_ratio_preserving_resize, resolve_max_soft_tokens, Gemma4ImageProcessor,
    Gemma4ImageProcessorConfig, Gemma4PixelValues, MAX_SUPPORTED_SOFT_TOKENS,
};
pub use prompt_cache::read_cache_stats as gemma4_cache_stats;
pub use vision::unified::{
    build_unified_inputs_embeds, is_unified_arch, load_unified_vision_embedder,
    unified_image_processor_config, unified_num_soft_tokens, UnifiedVisionConfig,
    UnifiedVisionEmbedder,
};
pub use vision::{
    build_inputs_embeds, load_multimodal_embedder, load_vision_tower, MultimodalEmbedder,
    VisionModel, BOI_TOKEN_ID, EOI_TOKEN_ID, IMAGE_TOKEN_ID,
};
