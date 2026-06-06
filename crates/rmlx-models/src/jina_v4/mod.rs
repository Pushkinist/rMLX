//! jina-embeddings-v4 multimodal embedding encoder.
//!
//! `JinaEmbeddingsV4Model` (jinaai/jina-embeddings-v4) is a Qwen2.5-VL-3B
//! backbone repurposed as a dense embedding model. It is:
//!
//! - **Pure bf16, unquantized** — no dequant kernel required.
//! - **Standalone** — NOT part of the `Architecture` enum. Every enum method
//!   assumes a causal LM (logits, vocab, KV-cache, `generate`) which do not
//!   apply to an encoder.
//! - **Text path**: single-vector + multi-vector text embeddings with runtime
//!   LoRA task selection. Vision (image path) is gated on explicit user
//!   sign-off for the `image` crate dependency.
//!
//! ## Module layout
//!
//! ```text
//! jina_v4/
//! mod.rs — this file; public API surface
//! config.rs — JinaV4Config parser
//! model.rs — text-tower forward + multi_vector_projector
//! lora.rs — multi-task LoRA: decoder + projector
//! pooling.rs — pooling + L2-norm + matryoshka
//! preprocess.rs— Qwen2.5-VL image front-end
//! vision.rs — Qwen2.5-VL ViT tower
//! (future) — M-RoPE merge / image-span pooling
//! ```
//!
//! ## Key design decisions (from docs/jina-v4-recon.md)
//!
//! - LoRA key prefix: `base_model.model.model.language_model.layers.{N}.<proj>.lora_{A,B}.<task>.weight`
//!   (note the `.language_model.` segment). Always enumerate from the actual
//!   safetensors header — never trust docs.
//! - Vision MLP + attn.proj have `bias=True` (jina porting trap; stock
//!   mlx_vlm uses `bias=False`).
//! - `single_vector_pool_strategy = "mean"` over the full attention mask.
//! - Matryoshka: `emb[:dim]` then re-L2-normalize.

mod config;
mod image;
mod lora;
mod model;
mod pooling;
mod preprocess;
mod vision;

pub use config::{JinaV4Config, JinaV4TextConfig, JinaV4VisionConfig};
pub use lora::{AdapterConfig, JinaV4Adapters, JinaV4Task};
pub use model::{JinaV4Text, LoraDelta};
pub use preprocess::{
    preprocess_image_bytes, preprocess_image_path, ImageGridThw, ImagePreprocessConfig, PixelValues,
};
pub use vision::JinaV4Vision;

/// The fixed prompt jina's `process_images` wraps an image in (single
/// `<|image_pad|>` placeholder). The server tokenizes this (no BOS, no chat
/// template — same as the text path) and passes the resulting ids to
/// [`JinaV4::embed_image_single`] / [`JinaV4::embed_image_multi`], which
/// expand the placeholder per `image_grid_thw`. Exposed so the route layer
/// owns tokenization (it holds the registry tokenizer), keeping `jina_v4`
/// tokenizer-free, exactly like the text path.
pub fn image_prompt() -> &'static str {
    image::IMAGE_PROMPT
}

use std::path::Path;

use rmlx_core::error::Result;

// ---------------------------------------------------------------------------
// Thin model shell (grows in later subtasks)
// ---------------------------------------------------------------------------

/// Handle to a loaded jina-v4 model.
///
/// Holds the parsed config, the bf16 text tower, and the parsed multi-task
/// LoRA adapter bundle. The currently-applied task's deltas are live in the
/// tower's `Linear` seams; the default task is `retrieval` (jina convention).
/// Pooling / projector are added in subsequent subtasks.
#[allow(missing_debug_implementations)]
/// jina-embeddings-v4 full model (text tower + vision tower + projector + LoRA).
pub struct JinaV4 {
    /// Parsed top-level model configuration.
    pub config: JinaV4Config,
    /// bf16 Qwen2 text tower exposing `forward_hidden`.
    pub text: JinaV4Text,
    /// bf16 Qwen2.5-VL vision tower (`visual.*`). Loaded by
    /// [`load_from_path`]; produces merged image embeddings
    /// `[num_merged_tokens, out_hidden=2048]`. The image-feature merge /
    /// M-RoPE / image-span pooling that consume this are not yet implemented.
    pub vision: JinaV4Vision,
    /// bf16 `multi_vector_projector` (2048 -> 128, bias=true) with its own
    /// per-task LoRA seam — kept consistent with `active_task`.
    projector: model::MultiVectorProjector,
    /// All three task adapters, parsed once at load.
    pub adapters: JinaV4Adapters,
    /// The task whose LoRA deltas are currently live in `text` + `projector`.
    active_task: JinaV4Task,
}

impl JinaV4 {
    /// Swap the live LoRA set to `task`'s adapters (clean replace — no residue
    /// of the prior task), on **both** the decoder tower and the
    /// `multi_vector_projector` (jina applies the same `task_label` to both —
    /// ref `modeling_jina_embeddings_v4.py:262`). `forward_hidden`,
    /// `embed_single`, and `embed_multi` thereafter run with `task`'s deltas
    /// applied. Idempotent re-application is allowed.
    pub fn apply_task(&mut self, task: JinaV4Task) -> Result<()> {
        self.adapters.apply_task(&mut self.text, task)?;
        self.adapters.apply_projector(&mut self.projector, task)?;
        self.active_task = task;
        Ok(())
    }

    /// The task whose LoRA deltas are currently live.
    pub fn active_task(&self) -> JinaV4Task {
        self.active_task
    }

    /// Full-sequence forward over `input_ids`, returning post-final-norm
    /// hidden states `[1, seq, hidden]` — with the active task's LoRA live.
    pub fn forward_hidden(
        &self,
        input_ids: &[i64],
        device: rmlx_mlx::Device,
    ) -> Result<rmlx_mlx::Array> {
        self.text.forward_hidden(input_ids, device)
    }

    /// Single-vector text embedding: `forward_hidden` → mean-pool over the
    /// sequence → L2-normalize, with the active task's LoRA live. When
    /// `truncate_dim` is `Some`, the (validated, matryoshka-allowed) prefix is
    /// sliced and re-L2-normalized. Output length == 2048 (hidden) or
    /// `truncate_dim`. Port: `modeling_jina_embeddings_v4.py:217-251,349-353`.
    pub fn embed_single(
        &self,
        input_ids: &[i64],
        device: rmlx_mlx::Device,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let hidden = self.text.forward_hidden(input_ids, device)?;
        pooling::single_vector(&hidden, &self.config.matryoshka_dims, truncate_dim, device)
    }

    /// Multi-vector text embedding: `forward_hidden` →
    /// `multi_vector_projector` (with the active task's LoRA) → per-token
    /// L2-normalize. Output shape `[seq][multi_vector_projector_dim]` (128).
    /// Matryoshka does **not** apply to multi-vector (matches the reference —
    /// only `single_vec_emb` is truncated in `_process_batches`). Port:
    /// `modeling_jina_embeddings_v4.py:253-266`.
    pub fn embed_multi(
        &self,
        input_ids: &[i64],
        device: rmlx_mlx::Device,
    ) -> Result<Vec<Vec<f32>>> {
        let hidden = self.text.forward_hidden(input_ids, device)?;
        let projected = self.projector.forward(&hidden, device)?;
        pooling::multi_vector(&projected, device)
    }

    /// Run the Qwen2.5-VL vision tower over one preprocessed image, returning
    /// the merged image embeddings `[num_merged_tokens, out_hidden=2048]`
    /// (`num_merged_tokens = pixel_values.num_patches / spatial_merge_size^2`).
    ///
    /// This is the ViT output **in isolation** — scattering it at the
    /// `<|image_pad|>` positions, M-RoPE, and image-span pooling are subtask
    /// 10. No LoRA is applied (jina excludes vision from its adapters).
    pub fn vision_embed(
        &self,
        pixel_values: &PixelValues,
        device: rmlx_mlx::Device,
    ) -> Result<rmlx_mlx::Array> {
        self.vision.forward(pixel_values, device)
    }

    /// Run the full image-embedding forward and return the post-final-norm
    /// hidden states `[1, seq, hidden]` plus the expanded `input_ids`.
    ///
    /// Pipeline (faithful to `modeling_jina_embeddings_v4.py` image path):
    /// 1. expand the single `<|image_pad|>` in `prompt_ids` to
    /// 2. `embed_tokens(input_ids)` -> text embeddings;
    /// 3. `vision_embed(pixel_values)` -> `[num_merged, hidden]`;
    /// 4. scatter the vision features at the `<|image_pad|>` positions;
    /// 5. compute 3D M-RoPE position ids (`get_rope_index`) + per-token
    /// 6. run the decoder with the M-RoPE-aware forward (active task LoRA
    ///
    /// The committed text-only path (1D RoPE `forward_hidden`) is never
    /// touched here — this is a separate forward gated by image presence.
    fn image_hidden(
        &self,
        prompt_ids: &[i64],
        pixel_values: &PixelValues,
        device: rmlx_mlx::Device,
        mm_cache: Option<&crate::multimodal_cache::MultimodalCache>,
    ) -> Result<(rmlx_mlx::Array, Vec<i64>)> {
        let image_token_id = i64::from(self.config.image_token_id);
        let merge = self.config.vision_config.spatial_merge_size;
        let g = pixel_values.grid;
        let num_merged = (g.t * g.h * g.w) / (merge * merge);

        let input_ids = image::expand_image_pad(prompt_ids, image_token_id, num_merged)?;

        // text embeddings, then scatter the ViT features in-place.
        let inputs_embeds = self.text.embed_ids(&input_ids, device)?;
        // Short-circuit on a cache hit; the cached entry is the
        // merged ViT output `[num_merged, out_hidden]`.
        let key_bytes = crate::multimodal_cache::pixel_f32_bytes(&pixel_values.data);
        // Derive (H, W) from the patch grid so the header disambiguates
        // identical pixel byte runs at different geometries.
        let key = crate::multimodal_cache::MmCacheKey::image_key(
            key_bytes,
            u16::try_from(g.h).unwrap_or(u16::MAX),
            u16::try_from(g.w).unwrap_or(u16::MAX),
            3,
            crate::multimodal_cache::MmDtype::F32,
        );
        let vision_feats = crate::multimodal_cache::get_or_compute(mm_cache, key, || {
            self.vision.forward(pixel_values, device)
        })?;
        let inputs_embeds = image::scatter_vision_features(
            &inputs_embeds,
            &vision_feats,
            &input_ids,
            image_token_id,
            device,
        )?;

        // 3D M-RoPE position ids -> per-token cos/sin tables.
        let rope_idx = image::get_rope_index(&input_ids, image_token_id, g.t, g.h, g.w, merge)?;
        let (head_dim, rope_theta, mrope_section) = image::mrope_params(&self.config);
        let (cos, sin) =
            image::build_mrope_tables(&rope_idx, head_dim, rope_theta, &mrope_section)?;
        let seq = input_ids.len() as i32;
        let cos = image::upload_bf16(&cos, &[seq, head_dim as i32], device)?;
        let sin = image::upload_bf16(&sin, &[seq, head_dim as i32], device)?;

        let hidden =
            self.text
                .forward_hidden_from_embeds_mrope(&inputs_embeds, &cos, &sin, device)?;
        Ok((hidden, input_ids))
    }

    /// Single-vector **image** embedding: image forward → mean-pool hidden
    /// over the `[<|vision_start|>, <|vision_end|>]` span (inclusive) →
    /// L2-normalize, with the active task's LoRA live. When `truncate_dim`
    /// is `Some`, the (matryoshka-validated) prefix is sliced + re-L2-normed.
    /// Port: `modeling_jina_embeddings_v4.py:226-251,349-353`.
    pub fn embed_image_single(
        &self,
        prompt_ids: &[i64],
        pixel_values: &PixelValues,
        device: rmlx_mlx::Device,
        truncate_dim: Option<usize>,
        mm_cache: Option<&crate::multimodal_cache::MultimodalCache>,
    ) -> Result<Vec<f32>> {
        let (hidden, input_ids) = self.image_hidden(prompt_ids, pixel_values, device, mm_cache)?;
        let (start, end) = image::vision_span(
            &input_ids,
            i64::from(self.config.vision_start_token_id),
            i64::from(self.config.vision_end_token_id),
        )?;
        pooling::single_vector_image_span(
            &hidden,
            start,
            end,
            &self.config.matryoshka_dims,
            truncate_dim,
            device,
        )
    }

    /// Multi-vector **image** embedding: image forward →
    /// `multi_vector_projector` (with the active task's LoRA) → per-token
    /// L2-normalize. Output `[seq][multi_vector_projector_dim]` (128). No
    /// matryoshka (matches the reference — only single-vec truncates). Port:
    /// `modeling_jina_embeddings_v4.py:253-266`.
    pub fn embed_image_multi(
        &self,
        prompt_ids: &[i64],
        pixel_values: &PixelValues,
        device: rmlx_mlx::Device,
        mm_cache: Option<&crate::multimodal_cache::MultimodalCache>,
    ) -> Result<Vec<Vec<f32>>> {
        let (hidden, _input_ids) = self.image_hidden(prompt_ids, pixel_values, device, mm_cache)?;
        let projected = self.projector.forward(&hidden, device)?;
        pooling::multi_vector(&projected, device)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Load a jina-v4 model from `model_dir`.
///
/// Parses `config.json`, loads the pure-bf16 Qwen2 text tower
/// (`embed_tokens` + 36 decoder layers + final RMSNorm), the Qwen2.5-VL
/// vision tower (`visual.*` — bias=true, no LoRA), and the base
/// `multi_vector_projector` (2048→128, bias=true), parses the multi-task LoRA
/// bundle from `adapters/`, and applies the default `retrieval` task to
/// **both** the decoder and the projector. Returns a `JinaV4` exposing
/// `forward_hidden`, `embed_single`, `embed_multi`, and `vision_embed` with
/// that task's deltas live. No Metal context is claimed at load time (device
/// is chosen per forward call). The text-only path is unchanged — vision is
/// loaded eagerly but only touched via `vision_embed`.
///
/// Not yet implemented:
/// - image-feature merge / M-RoPE / image-span pooling
pub fn load_from_path(model_dir: &Path) -> Result<JinaV4> {
    let config_path = model_dir.join("config.json");
    let config = JinaV4Config::from_file(&config_path)?;
    let mut text = model::load_text_tower(model_dir, &config.text_config)?;
    let vision = vision::load_vision_tower(model_dir, &config.vision_config)?;
    let mut projector = model::load_multi_vector_projector(model_dir)?;
    let adapters = JinaV4Adapters::load(model_dir, config.text_config.num_hidden_layers)?;
    // jina default task is `retrieval` — apply it (decoder + projector) so the
    // model is usable out of the box; callers switch via `JinaV4::apply_task`.
    let active_task = JinaV4Task::DEFAULT;
    adapters.apply_task(&mut text, active_task)?;
    adapters.apply_projector(&mut projector, active_task)?;
    Ok(JinaV4 {
        config,
        text,
        vision,
        projector,
        adapters,
        active_task,
    })
}
