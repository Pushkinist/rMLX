//! Gemma4 **unified** (`gemma4_unified` / `Gemma4UnifiedForConditionalGeneration`)
//! encoder-free vision embedder.
//!
//! Faithful host+MLX port of the HF Transformers `gemma4_unified`
//! `Gemma4UnifiedVisionEmbedder` (+ image processor `patches_merge` /
//! `convert_image_to_patches`) and `Gemma4UnifiedMultimodalEmbedder`.
//!
//! ## Why this is a different path than the SigLIP tower
//!
//! The unified 12B has **no vision transformer**. Vision is early-fusion: raw
//! pixel patches are projected straight into the shared 48-layer LM hidden
//! space via a single Dense + LayerNorms + factorized 2D positional embedding,
//! producing `num_soft_tokens` (280) soft tokens. The weights are
//! `vision_embedder.*` + `embed_vision.embedding_projection.*` — there is no
//! `vision_tower.*`. The standard `gemma4` family (e4b/26b/31b) keeps the
//! existing [`super::VisionModel`] SigLIP path.
//!
//! ## Reference pipeline (verbatim from `Gemma4UnifiedVisionEmbedder.forward`)
//!
//! ```text
//! hidden = patch_ln1(pixel_values)          # LayerNorm over 6912 raw-patch dims
//! hidden = patch_dense(hidden)              # Linear 6912 -> mm_embed_dim (3840), +bias
//! hidden = patch_ln2(hidden)                # LayerNorm over 3840
//! pos    = pos_embedding[x, 0, :] + pos_embedding[y, 1, :]   # factorized 2D, padding -> 0
//! hidden = pos_norm(hidden + pos)           # LayerNorm over 3840
//! hidden = embed_vision(hidden)             # RMSNormNoScale -> embedding_projection (3840 -> text_hidden)
//! ```
//!
//! Note: `patch_ln1`/`patch_ln2`/`pos_norm` are true **LayerNorm**
//! (mean-subtraction, learned weight **and** bias) — NOT RMSNorm. `embed_vision`
//! is the same `RMSNormNoScale -> Linear` [`super::MultimodalEmbedder`] reused
//! from the tower path.
//!
//! ## Image processing (host)
//!
//! Reference `Gemma4UnifiedImageProcessor`:
//! 1. aspect-ratio-preserving resize to a patch budget (reused from
//!    [`super::super::preprocessor`]; identical for both Gemma4 families).
//! 2. rescale to `[0,1]` (`do_normalize=false`, mean=0/std=1 on the snapshot).
//! 3. patchify into 16px teacher patches `[n_teacher, 16*16*3=768]`, layout
//!    `[patch_h, patch_w, channel]` (`convert_image_to_patches`).
//! 4. `patches_merge`: group `k=pooling_kernel_size` (3) × k teacher patches
//!    into one 48×48 model patch `[n_model, 48*48*3=6912]`; model-patch position
//!    = `(min teacher_x // k, min teacher_y // k)`.
//! 5. The model patch interior is laid out `[k*16, k*16, 3]` (kernel rows ×
//!    kernel cols × teacher-patch pixels), matching the reference reshape
//!    `(length, k, k, 16, 16, 3) -> permute -> (length, 48, 48, 3)`.
//!
//! Step 1 here introduces the `2*(x-0.5)` rescale used by the tower's patch
//! embedder. The unified embedder does **not** apply that; it consumes raw
//! `[0,1]` pixels (patch_ln1 normalizes). So this path uses its own patchify.

use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{add, multiply, scalar_f32, subtract, sum_axis_keepdims, Array, Device, Dtype};
use tracing::{debug, info};

use super::{f32_bytes, i32_bytes, load_raw, read_quant_params, MultimodalEmbedder};
use crate::gemma4::preprocessor::{Gemma4ImageProcessorConfig, Gemma4PixelValues};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `vision_config` for `gemma4_unified` (`model_type: gemma4_unified_vision`).
///
/// Distinct from [`super::super::config::Gemma4VisionConfig`] (the SigLIP
/// tower): the unified vision config has **no** `num_hidden_layers` / heads —
/// it is encoder-free. Fields are the complete contract for the embedder.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — fields are the complete gemma4_unified vision-embedder contract; adding a field requires updating from_json"
)]
#[derive(Debug, Clone)]
pub struct UnifiedVisionConfig {
    /// Multimodal embed dim = LM hidden (3840). `patch_dense` out, LayerNorm dim.
    pub mm_embed_dim: usize,
    /// Factorized positional-embedding table length per axis (1120).
    pub mm_posemb_size: usize,
    /// Model patch side after pooling (48). `patch_dim = model_patch_size^2 * 3`.
    pub model_patch_size: usize,
    /// Teacher patch side (16).
    pub patch_size: usize,
    /// Pooling kernel (`k`, 3) merging `k*k` teacher patches into one model patch.
    pub pooling_kernel_size: usize,
    /// Soft tokens produced per image (280). Padding budget.
    pub num_soft_tokens: usize,
    /// `embed_vision.embedding_projection` input dim (3840 = output_proj_dims).
    pub output_proj_dims: usize,
    /// LayerNorm / RMSNorm epsilon (1e-6).
    pub rms_norm_eps: f32,
}

impl UnifiedVisionConfig {
    /// `patch_dim = model_patch_size^2 * 3` — the raw-patch feature length
    /// (`48*48*3 = 6912`) fed into `patch_ln1` / `patch_dense`.
    #[inline]
    pub fn patch_dim(&self) -> usize {
        self.model_patch_size * self.model_patch_size * 3
    }

    /// Parse from the `vision_config` JSON object. Missing keys fall back to the
    /// verified `gemma-4-12B` `gemma4_unified_vision` values.
    pub fn from_json(v: &serde_json::Value) -> Self {
        let u = |key: &str, dflt: usize| -> usize {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let f = |key: &str, dflt: f32| -> f32 {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(dflt, |x| x as f32)
        };
        Self {
            mm_embed_dim: u("mm_embed_dim", 3840),
            mm_posemb_size: u("mm_posemb_size", 1120),
            model_patch_size: u("model_patch_size", 48),
            patch_size: u("patch_size", 16),
            pooling_kernel_size: u("pooling_kernel_size", 3),
            num_soft_tokens: u("num_soft_tokens", 280),
            output_proj_dims: u("output_proj_dims", 3840),
            rms_norm_eps: f("rms_norm_eps", 1e-6),
        }
    }

    /// Read `vision_config` from a model dir when `architectures[0]` is the
    /// unified arch. Returns `None` if there is no `vision_config` key.
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path).map_err(|e| {
            Error::Config(format!(
                "gemma4_unified: cannot read {}: {e}",
                path.display()
            ))
        })?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "gemma4_unified: malformed config.json at {}: {e}",
                path.display()
            ))
        })?;
        Ok(v.get("vision_config").map(Self::from_json))
    }
}

/// Returns true when `architectures[0] == "Gemma4UnifiedForConditionalGeneration"`.
///
/// The unified 12B loads through the same [`crate::arch::Architecture::Gemma4`]
/// text path as the tower family; this string check is the dispatch that routes
/// the vision/audio front-end to the encoder-free embedder instead of the
/// SigLIP `vision_tower` loader.
pub fn is_unified_arch(model_dir: &Path) -> bool {
    let path = model_dir.join("config.json");
    let Ok(data) = std::fs::read(&path) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return false;
    };
    v.get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|a| a.first())
        .and_then(serde_json::Value::as_str)
        == Some("Gemma4UnifiedForConditionalGeneration")
}

// ---------------------------------------------------------------------------
// LayerNorm (weight + bias, mean-subtraction) — composed from primitives
// ---------------------------------------------------------------------------

/// Standard LayerNorm over the last axis: `(x - mean) / sqrt(var + eps) * w + b`.
///
/// Distinct from [`crate::layers::RmsNorm`] (no mean-subtraction). The unified
/// embedder's `patch_ln1` / `patch_ln2` / `pos_norm` are PyTorch `nn.LayerNorm`
/// (the upstream class names say "ln", and the weights carry both `.weight` and
/// `.bias`). Computed in f32 for stability — the whole embedder runs once per
/// image, off the decode loop.
#[allow(missing_debug_implementations)]
struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    /// `x`: `[..., dim]`. Returns same shape.
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let x = x.astype(Dtype::F32, device)?;
        let shape = x.shape();
        if shape.is_empty() {
            return Err(Error::Model(
                "gemma4_unified LayerNorm: rank-0 input has no last axis to normalize".to_owned(),
            ));
        }
        let axis = (shape.len() as i32) - 1;
        // SAFETY: shape non-empty (checked above) → last() is Some.
        let dim = *shape.last().unwrap_or(&1) as f32;
        // mean over last axis (keepdims for broadcast).
        let sum = sum_axis_keepdims(&x, axis, device)?;
        let mean = multiply(&sum, &scalar_f32(1.0 / dim), device)?; // f32-ok: x is cast to f32 on entry (line above)
        let centered = subtract(&x, &mean, device)?;
        // var = mean(centered^2)
        let sq = multiply(&centered, &centered, device)?;
        let sq_sum = sum_axis_keepdims(&sq, axis, device)?;
        let var = multiply(&sq_sum, &scalar_f32(1.0 / dim), device)?; // f32-ok: x is cast to f32 on entry
        let denom = rmlx_mlx::sqrt(&add(&var, &scalar_f32(self.eps), device)?, device)?; // f32-ok: x is cast to f32 on entry
        let normed = rmlx_mlx::divide(&centered, &denom, device)?;
        let scaled = multiply(&normed, &self.weight, device)?;
        add(&scaled, &self.bias, device)
    }
}

// ---------------------------------------------------------------------------
// Unified vision embedder
// ---------------------------------------------------------------------------

/// Encoder-free unified vision embedder (`vision_embedder.*` + `embed_vision.*`).
#[allow(missing_debug_implementations)]
pub struct UnifiedVisionEmbedder {
    cfg: UnifiedVisionConfig,
    patch_ln1: LayerNorm,
    /// `patch_dense`: `[mm_embed_dim, patch_dim]` (possibly quantized) + bias.
    patch_dense: crate::layers::Linear,
    patch_dense_bias: Array,
    patch_ln2: LayerNorm,
    /// `[mm_posemb_size, 2, mm_embed_dim]` factorized 2D table (f32).
    pos_embedding: Array,
    pos_norm: LayerNorm,
    /// Shared `RMSNormNoScale -> embedding_projection` (`embed_vision.*`).
    embed_vision: MultimodalEmbedder,
}

impl UnifiedVisionEmbedder {
    /// Parsed unified-vision sub-config.
    pub fn config(&self) -> &UnifiedVisionConfig {
        &self.cfg
    }

    /// Run the full embedder over one preprocessed image. Returns
    /// `[1, num_soft_tokens, text_hidden]` ready to scatter into `inputs_embeds`.
    ///
    /// `pv` is the resized/rescaled `[1, 3, H, W]` CHW buffer from the shared
    /// Gemma4 preprocessor; `H`/`W` are multiples of `model_patch_size`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: patch loop indices bounded by host-computed patch grid"
    )]
    pub fn forward(&self, pv: &Gemma4PixelValues, device: Device) -> Result<Array> {
        let (patches, x_idx, y_idx) = self.patchify_and_merge(pv)?;
        let n_model = x_idx.len();
        let patch_dim = self.cfg.patch_dim() as i32;
        let mm = self.cfg.mm_embed_dim as i32;

        // Raw merged patches -> device [n_model, patch_dim].
        let patch_arr = Array::from_bytes(
            f32_bytes(&patches),
            &[n_model as i32, patch_dim],
            Dtype::F32,
        )?;

        // patch_ln1 -> patch_dense (+bias) -> patch_ln2
        let mut h = self.patch_ln1.forward(&patch_arr, device)?;
        h = self.patch_dense.forward(&h, device)?;
        h = add(&h, &self.patch_dense_bias, device)?;
        h = self.patch_ln2.forward(&h, device)?;

        // Factorized 2D positional embedding (host gather of two axis rows).
        let pos = self.gather_pos_embedding(&x_idx, &y_idx, device)?; // [n_model, mm]
        h = add(&h, &pos, device)?;
        h = self.pos_norm.forward(&h, device)?;

        // embed_vision: RMSNormNoScale -> embedding_projection -> [n_model, text_hidden].
        h = h.reshape(&[1, n_model as i32, mm], device)?;
        let out = self.embed_vision.forward(&h, device)?;

        debug!(
            n_model_patches = n_model,
            mm_embed_dim = self.cfg.mm_embed_dim,
            "gemma4_unified vision: embedder forward"
        );
        Ok(out)
    }

    /// Host patchify (16px teacher patches) + `patches_merge` (k×k -> model
    /// patch) producing flat `[n_model * patch_dim]` f32 plus per-model-patch
    /// `(x, y)` positions.
    ///
    /// Faithful to `convert_image_to_patches` (layout `[patch_h, patch_w, ch]`)
    /// and `patches_merge` (model-patch interior `[k, k, 16, 16, 3]` ->
    /// `[48, 48, 3]`, position = top-left teacher position // k).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: all indices derived from the host-computed patch grid"
    )]
    fn patchify_and_merge(&self, pv: &Gemma4PixelValues) -> Result<(Vec<f32>, Vec<i32>, Vec<i32>)> {
        patchify_and_merge_impl(&self.cfg, pv)
    }

    /// Gather `pos_embedding[x, 0, :] + pos_embedding[y, 1, :]` per model patch.
    ///
    /// `pos_embedding` is `[mm_posemb_size, 2, mm_embed_dim]`; axis-0 slice is
    /// the X table, axis-1 slice is the Y table. Positions are in-range by
    /// construction (no padding patches on the single-image path).
    fn gather_pos_embedding(&self, x_idx: &[i32], y_idx: &[i32], device: Device) -> Result<Array> {
        let n = x_idx.len() as i32;
        let posemb = self.cfg.mm_posemb_size as i32;
        let mm = self.cfg.mm_embed_dim as i32;
        // table_x = pos_embedding[:, 0, :], table_y = pos_embedding[:, 1, :].
        let table_x = self
            .pos_embedding
            .slice(&[0, 0, 0], &[posemb, 1, mm], &[1, 1, 1], device)?
            .reshape(&[posemb, mm], device)?;
        let table_y = self
            .pos_embedding
            .slice(&[0, 1, 0], &[posemb, 2, mm], &[1, 1, 1], device)?
            .reshape(&[posemb, mm], device)?;
        let x_arr = Array::from_bytes(i32_bytes(x_idx), &[n], Dtype::I32)?;
        let y_arr = Array::from_bytes(i32_bytes(y_idx), &[n], Dtype::I32)?;
        let pe_x = table_x.take(&x_arr, 0, device)?;
        let pe_y = table_y.take(&y_arr, 0, device)?;
        add(&pe_x, &pe_y, device)
    }
}

/// Host patchify (16px teacher patches) + `patches_merge` (k×k -> model patch)
/// core, factored out of [`UnifiedVisionEmbedder::patchify_and_merge`] so the
/// channel/value layout is covered by a model-free numerical test.
///
/// Faithful to `convert_image_to_patches` (teacher-patch interior `[ry, rx, ch]`)
/// and `patches_merge` (model-patch interior `[ky, ry, kx, rx, ch]` over dims
/// `[k, p, k, p, 3]`; position = top-left teacher position // k). Returns the
/// flat `[n_model * patch_dim]` f32 plus per-model-patch `(x, y)` positions.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: all indices derived from the host-computed patch grid"
)]
fn patchify_and_merge_impl(
    cfg: &UnifiedVisionConfig,
    pv: &Gemma4PixelValues,
) -> Result<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    let p = cfg.patch_size; // 16
    let k = cfg.pooling_kernel_size; // 3
    let h = pv.height;
    let w = pv.width;
    if !h.is_multiple_of(p * k) || !w.is_multiple_of(p * k) {
        return Err(Error::Model(format!(
            "gemma4_unified vision: image {h}x{w} not divisible by model_patch_size {}",
            p * k
        )));
    }
    let p_h = h / p; // teacher rows
    let p_w = w / p; // teacher cols
    let m_h = p_h / k; // model rows
    let m_w = p_w / k; // model cols
    let n_model = m_h * m_w;
    let model_patch = p * k; // 48
    let patch_dim = model_patch * model_patch * 3; // 6912
    let n_pixels = h * w;

    // Build merged patches directly in the reference target layout. The upstream
    // `patches_merge` reshapes the k×k kernel group to `(length, k, k, p, p, 3)`
    // then permutes to `(length, k, p, k, p, 3)` and flattens — i.e. the
    // 6912-vector interior is ordered **`[ky, ry, kx, rx, ch]`**, making the
    // model patch a *contiguous* (k*p)×(k*p) image: full row = `ky*p + ry`, full
    // col = `kx*p + rx`. (Ordering `[ky, kx, ry, rx, ch]` would tile 3×3 blocks
    // instead and scramble fine detail — OCR fails, colour survives.)
    let mut merged = vec![0.0_f32; n_model * patch_dim];
    let mut x_idx = vec![0i32; n_model];
    let mut y_idx = vec![0i32; n_model];
    for my in 0..m_h {
        for mx in 0..m_w {
            let model_i = my * m_w + mx;
            // model-patch position = (min teacher_x // k, min teacher_y // k)
            // = (mx, my) since teacher cols/rows in a kernel are contiguous.
            x_idx[model_i] = mx as i32;
            y_idx[model_i] = my as i32;
            let dst = model_i * patch_dim;
            for ky in 0..k {
                for ry in 0..p {
                    for kx in 0..k {
                        for rx in 0..p {
                            let y = (my * k + ky) * p + ry; // my*48 + ky*16 + ry
                            let x = (mx * k + kx) * p + rx; // mx*48 + kx*16 + rx
                            for ch in 0..3 {
                                let src = ch * n_pixels + y * w + x;
                                // interior index: [ky, ry, kx, rx, ch] over dims [k, p, k, p, 3]
                                let off = ((((ky * p + ry) * k + kx) * p + rx) * 3) + ch;
                                merged[dst + off] = pv.pixel_values[src];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((merged, x_idx, y_idx))
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load the unified vision embedder (`vision_embedder.*` + `embed_vision.*`)
/// from a snapshot directory. Errors if the unified-vision weights are absent
/// (caller disables image input on error).
pub fn load_unified_vision_embedder(
    model_dir: &Path,
    cfg: &UnifiedVisionConfig,
) -> Result<UnifiedVisionEmbedder> {
    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    fn load_f32(shards: &ShardSet, name: &str) -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle.safetensors()?;
            if let Ok(t) = st.tensor(name) {
                let tv = rmlx_loader::TensorView {
                    name,
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    bytes: t.data(),
                };
                let a = Array::from_safetensor_view(&tv)?;
                return a.astype(Dtype::F32, Device::Cpu);
            }
        }
        Err(Error::Loader(format!(
            "gemma4_unified vision: tensor '{name}' not found in any shard"
        )))
    }
    let has = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };

    // patch_ln1 / patch_ln2 / pos_norm are PyTorch `nn.LayerNorm` constructed
    // with the default eps (1e-5) in the reference embedder — NOT the model's
    // `rms_norm_eps` (1e-6, which only governs the `embed_vision` RMSNorm).
    let layer_norm = |prefix: &str| -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: load_f32(&shards, &format!("{prefix}.weight"))?,
            bias: load_f32(&shards, &format!("{prefix}.bias"))?,
            eps: 1e-5,
        })
    };

    // patch_dense: quantized linear (mxfp8 on the snapshot) + additive bias.
    let dense_base = "vision_embedder.patch_dense";
    let patch_dense = if has(&format!("{dense_base}.scales")) {
        let weight = load_raw(&shards, &format!("{dense_base}.weight"))?;
        let scales = load_raw(&shards, &format!("{dense_base}.scales"))?;
        let biases = if has(&format!("{dense_base}.biases")) {
            Some(load_raw(&shards, &format!("{dense_base}.biases"))?)
        } else {
            None
        };
        let (gs, bits, mode) = read_quant_params(model_dir)?;
        crate::layers::Linear::Quantized {
            weight,
            scales,
            biases,
            group_size: gs,
            bits,
            mode,
        }
    } else {
        crate::layers::Linear::Plain {
            weight: load_f32(&shards, &format!("{dense_base}.weight"))?,
        }
    };
    let patch_dense_bias = load_f32(&shards, &format!("{dense_base}.bias"))?;

    let pos_embedding = load_f32(&shards, "vision_embedder.pos_embedding")?;

    // embed_vision: RMSNormNoScale -> embedding_projection (reused loader).
    let embed_vision =
        super::load_multimodal_embedder(model_dir, "embed_vision", cfg.rms_norm_eps)?;

    // Validate the parsed `output_proj_dims` against the loaded projection's
    // actual input feature dim. This makes the config field load-bearing: a
    // checkpoint whose `embed_vision.embedding_projection` does not consume
    // `output_proj_dims` features is rejected here instead of failing with an
    // opaque shape error inside the forward pass.
    if let Some(proj_in) = embed_vision.projection_input_dim() {
        if proj_in != cfg.output_proj_dims {
            return Err(Error::Loader(format!(
                "gemma4_unified vision: output_proj_dims ({}) != embed_vision.embedding_projection \
                 input dim ({proj_in}) — config/checkpoint mismatch",
                cfg.output_proj_dims
            )));
        }
    }

    info!(
        mm_embed_dim = cfg.mm_embed_dim,
        num_soft_tokens = cfg.num_soft_tokens,
        model_patch_size = cfg.model_patch_size,
        "gemma4_unified vision: embedder loaded (encoder-free)"
    );

    Ok(UnifiedVisionEmbedder {
        cfg: cfg.clone(),
        patch_ln1: layer_norm("vision_embedder.patch_ln1")?,
        patch_dense,
        patch_dense_bias,
        patch_ln2: layer_norm("vision_embedder.patch_ln2")?,
        pos_embedding,
        pos_norm: layer_norm("vision_embedder.pos_norm")?,
        embed_vision,
    })
}

// ---------------------------------------------------------------------------
// Image processor for the unified path
// ---------------------------------------------------------------------------

/// Number of soft tokens an image of size `(h, w)` will consume after the
/// unified embedder's `patches_merge`:
/// `(h / model_patch_size) * (w / model_patch_size)`.
///
/// This is the count used to size the image-token block in the prompt. It must
/// equal the model-patch count produced by [`UnifiedVisionEmbedder::forward`].
#[inline]
pub fn unified_num_soft_tokens(h: usize, w: usize, cfg: &UnifiedVisionConfig) -> usize {
    let mp = cfg.model_patch_size;
    (h / mp) * (w / mp)
}

/// Build the [`Gemma4ImageProcessorConfig`] for the unified path from the
/// unified vision config. The shared preprocessor (resize + rescale) is
/// identical to the tower path; only the post-resize patchify differs (done in
/// [`UnifiedVisionEmbedder::forward`]). The processor's reported
/// `num_soft_tokens` is corrected to the model-patch count by
/// [`unified_num_soft_tokens`] at prompt-build time.
pub fn unified_image_processor_config(cfg: &UnifiedVisionConfig) -> Gemma4ImageProcessorConfig {
    let d = Gemma4ImageProcessorConfig::default();
    Gemma4ImageProcessorConfig {
        patch_size: cfg.patch_size,
        max_soft_tokens: cfg.num_soft_tokens,
        pooling_kernel_size: cfg.pooling_kernel_size,
        ..d
    }
}

// ---------------------------------------------------------------------------
// build unified inputs_embeds (text + scattered vision soft tokens)
// ---------------------------------------------------------------------------

/// Build the merged `inputs_embeds` for a unified-arch image prompt.
///
/// Mirrors [`super::build_inputs_embeds`] but routes the encode through the
/// encoder-free [`UnifiedVisionEmbedder`]. Each image contributes
/// `unified_num_soft_tokens` soft tokens scattered at its contiguous run of
/// [`super::IMAGE_TOKEN_ID`] positions.
///
/// Returns `(inputs_embeds [1, seq, hidden], masked_ids [seq])`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: img_positions are filtered from input_ids; per-image runs validated contiguous before slice_update"
)]
pub fn build_unified_inputs_embeds(
    model: &super::super::model::Gemma4Text,
    embedder: &UnifiedVisionEmbedder,
    images: &[Gemma4PixelValues],
    input_ids: &[u32],
    device: Device,
    mm_cache: Option<&crate::multimodal_cache::MultimodalCache>,
    model_sig: u64,
) -> Result<(Array, Array)> {
    let hidden = model.cfg.hidden_size as i32;
    let seq = input_ids.len();

    let img_positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == super::IMAGE_TOKEN_ID)
        .map(|(i, _)| i)
        .collect();
    // Each image's soft-token count = model-patch count for its resized size.
    let per_image: Vec<usize> = images
        .iter()
        .map(|pv| unified_num_soft_tokens(pv.height, pv.width, embedder.config()))
        .collect();
    let expected: usize = per_image.iter().sum();
    if img_positions.len() != expected {
        return Err(Error::Model(format!(
            "gemma4_unified image: {} image-token ({}) positions in prompt != \
             {expected} vision soft tokens ({} image(s)) — scatter would misalign",
            img_positions.len(),
            super::IMAGE_TOKEN_ID,
            images.len()
        )));
    }
    info!(
        image_tokens = img_positions.len(),
        images = images.len(),
        seq,
        "gemma4_unified image: building inputs_embeds (token count == soft tokens)"
    );

    // Scaled text embeddings: embed_tokens(ids) * sqrt(hidden).
    let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
    let ids_arr = Array::from_bytes(i32_bytes(&ids_i32), &[seq as i32], Dtype::I32)?;
    let h_raw = model.embed_tokens.forward(&ids_arr, device)?;
    let embed_scale =
        scalar_f32((model.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
    let mut embeds = multiply(&h_raw, &embed_scale, device)?;
    embeds = embeds.reshape(&[1, seq as i32, hidden], device)?;
    let embeds_dtype = embeds.dtype();

    let mut cursor = 0usize;
    for (img_idx, (pv, &n_soft)) in images.iter().zip(per_image.iter()).enumerate() {
        let key_bytes = crate::multimodal_cache::pixel_f32_bytes(&pv.pixel_values);
        let key = crate::multimodal_cache::MmCacheKey::image_key(
            key_bytes,
            u16::try_from(pv.height).unwrap_or(u16::MAX),
            u16::try_from(pv.width).unwrap_or(u16::MAX),
            3,
            crate::multimodal_cache::MmDtype::F32,
            model_sig,
        );
        let feats = crate::multimodal_cache::get_or_compute(mm_cache, key, || {
            embedder.forward(pv, device)
        })?;
        let fs = feats.shape();
        if fs.first().copied() != Some(1)
            || fs.get(1).copied() != Some(n_soft as i32)
            || fs.get(2).copied() != Some(hidden)
        {
            return Err(Error::Model(format!(
                "gemma4_unified image: vision feature shape {fs:?} != [1, {n_soft}, {hidden}] \
                 for image {img_idx}"
            )));
        }
        let feats = feats.astype(embeds_dtype, device)?;

        let run = &img_positions[cursor..cursor + n_soft];
        let first = run[0];
        let contiguous = run.iter().enumerate().all(|(k, &p)| p == first + k);
        if !contiguous {
            return Err(Error::Model(format!(
                "gemma4_unified image: image-token positions for image {img_idx} are not \
                 contiguous (got {run:?})"
            )));
        }
        embeds = embeds.slice_update(
            &feats,
            &[0, first as i32, 0],
            &[1, (first + n_soft) as i32, hidden],
            &[1, 1, 1],
            device,
        )?;
        cursor += n_soft;
    }

    // Mask image-token ids to 0 for per-layer-input gating (matches the tower path).
    let mut masked: Vec<i32> = ids_i32;
    for &p in &img_positions {
        masked[p] = 0;
    }
    let masked_arr = Array::from_bytes(i32_bytes(&masked), &[seq as i32], Dtype::I32)?;

    Ok((embeds, masked_arr))
}

// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "unified_tests.rs"]
mod unified_tests;
