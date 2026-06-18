// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Gemma3 standard SigLIP vision tower (ViT) + `Gemma3MultiModalProjector`.
//!
//! Faithful port of `mlx_vlm/models/gemma3/vision.py` (`SigLipVisionModel`,
//! `VisionEmbeddings`, `Attention`, `EncoderLayer`, `MLP`) and the
//! `Gemma3MultiModalProjector` from `gemma3.py`.
//!
//! ## How this differs from the Gemma4 SigLIP tower (`gemma4/vision.rs`)
//!
//! Gemma3's SigLIP is the **standard** SigLIP — NOT Gemma4's custom variant.
//! It is therefore a separate implementation, not a reuse of :
//!
//! | Piece | Gemma3 (this file) | Gemma4 (gemma4/vision.rs) |
//! |------------------|-----------------------------------|----------------------------------|
//! | Patch embed | Conv2d (stride=patch) + bias | flatten patches -> `input_proj` |
//! | Patch rescale | none (done in preprocessor) | `2*(x-0.5)` in-tower |
//! | Position embed | learned 1D `[num_patches, hidden]`| one-hot 2-axis table (summed) |
//! | Block norms | **LayerNorm** (weight + bias) | RMSNorm (no bias) |
//! | Attention norms | none (plain MHA, q/k/v/o + bias) | q_norm/k_norm RMSNorm + 2D RoPE |
//! | Positional info | additive learned pos-embed only | 2D multidimensional RoPE on q/k |
//! | Linears | plain `nn.Linear` (+ bias) | `ClippableLinear` (clamp bufs) |
//! | MLP activation | `gelu` precise | `gelu_tanh` |
//! | Pooling | none in tower; AvgPool2d in proj | `VisionPooler` 3x3 in-tower |
//! | Final norm | `post_layernorm` (LayerNorm) | none (pooler scales by sqrt) |
//!
//! ## Projector divergences (the Gemma3-specific additions)
//!
//! `Gemma3MultiModalProjector`:
//! 1. **AvgPool2d** downsample: vision out `[1, 4096, 1152]` reshaped to a
//!    64x64 spatial grid, average-pooled with kernel=stride=4 -> 16x16=256 soft
//!    tokens -> `[1, 256, 1152]`.
//! 2. **Scale-bearing RMSNorm** `mm_soft_emb_norm` (NOT RMSNormNoScale).
//! 3. **einsum** `btm,md->btd` against `mm_input_projection_weight`
//!    `[vision_hidden, text_hidden]` (NOT `nn.Linear`) -> `[1, 256, 2560]`.
//!
//! Scatter (`build_inputs_embeds`): scale features by `1/sqrt(text_hidden)`,
//! masked_scatter at `image_token_index = 262144`.
//!
//! The tower runs in **float32** (weights upcast at load), matching the
//! gemma4/jina precedent: a one-shot encoder, no decode loop, f32 cost is
//! negligible against the bf16 numerical-drift risk over 27 blocks.

#![allow(clippy::items_after_statements)]
use std::mem::size_of_val;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{
    add, divide, gelu, matmul, multiply, rms_norm, scalar_f32, scaled_dot_product_attention, sqrt,
    subtract, sum_axis, Array, Device, Dtype,
};
use tracing::{debug, info};

use super::config::Gemma3VisionConfig;

// ---------------------------------------------------------------------------
// Linear (plain, with bias) — standard SigLIP nn.Linear
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Linear {
    /// `[out, in]` f32 weight (HF convention; transposed at matmul time).
    weight: Array,
    bias: Array,
}

impl Linear {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let y = matmul(x, &self.weight.transpose(&[1, 0], device)?, device)?;
        add(&y, &self.bias, device)
    }
}

// ---------------------------------------------------------------------------
// LayerNorm (weight + bias, over the last axis) — std SigLIP nn.LayerNorm
// ---------------------------------------------------------------------------
//
// rmlx-mlx exposes no `layer_norm` op, so this is computed with the wrapped
// reduction/elementwise ops. The vision tower runs once per image (not in the
// decode hot loop), so the extra dispatches are immaterial.
//
// mean = sum(x, -1) / D
// xc = x - mean
// var = sum(xc^2, -1) / D
// out = xc / sqrt(var + eps) * weight + bias

#[allow(missing_debug_implementations)]
struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let mut keep = x.shape();
        let d = *keep.last().expect("layer_norm: empty shape");
        let inv_d = scalar_f32(1.0 / d as f32);

        // keepdims shape: replace last dim with 1.
        *keep.last_mut().unwrap() = 1;

        let mean = multiply(
            &sum_axis(x, -1, device)?.reshape(&keep, device)?,
            &inv_d,
            device,
        )?;
        let xc = subtract(x, &mean, device)?;
        let var = multiply(
            &sum_axis(&multiply(&xc, &xc, device)?, -1, device)?.reshape(&keep, device)?,
            &inv_d,
            device,
        )?;
        let denom = sqrt(&add(&var, &scalar_f32(self.eps), device)?, device)?;
        let normed = divide(&xc, &denom, device)?;
        let scaled = multiply(&normed, &self.weight, device)?;
        add(&scaled, &self.bias, device)
    }
}

// ---------------------------------------------------------------------------
// MLP: fc1 -> gelu(precise) -> fc2
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let h = self.fc1.forward(x, device)?;
        let h = gelu(&h, device)?; // "precise" GELU (nn.GELU(approx="precise"))
        self.fc2.forward(&h, device)
    }
}

// ---------------------------------------------------------------------------
// Attention — standard MHA, full bidirectional (mask=None), bias on q/k/v/out
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    /// `x`: `[1, seq, hidden]`. Full bidirectional attention (no mask).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let s = x.shape();
        let (b, seq) = (s[0], s[1]);
        let nh = self.num_heads as i32;
        let d = self.head_dim as i32;

        let to_bhsd = |a: Array| -> Result<Array> {
            a.reshape(&[b, seq, nh, d], device)?
                .transpose(&[0, 2, 1, 3], device)
        };
        let q = to_bhsd(self.q_proj.forward(x, device)?)?;
        let k = to_bhsd(self.k_proj.forward(x, device)?)?;
        let v = to_bhsd(self.v_proj.forward(x, device)?)?;

        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, "", None, device)?;
        let out = out
            .transpose(&[0, 2, 1, 3], device)?
            .reshape(&[b, seq, nh * d], device)?;
        self.out_proj.forward(&out, device)
    }
}

// ---------------------------------------------------------------------------
// EncoderLayer — pre-norm: LN1 -> attn -> +res ; LN2 -> mlp -> +res
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct EncoderLayer {
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    attn: Attention,
    mlp: Mlp,
}

impl EncoderLayer {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let r = self
            .attn
            .forward(&self.layer_norm1.forward(x, device)?, device)?;
        let h = add(x, &r, device)?;
        let r = self
            .mlp
            .forward(&self.layer_norm2.forward(&h, device)?, device)?;
        add(&h, &r, device)
    }
}

// ---------------------------------------------------------------------------
// Projector — Gemma3MultiModalProjector (AvgPool2d + RMSNorm + einsum)
// ---------------------------------------------------------------------------

/// `Gemma3MultiModalProjector`: AvgPool2d 4096->256 + scale-bearing RMSNorm
/// (`mm_soft_emb_norm`) + einsum projection (`mm_input_projection_weight`).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed layer struct — private weight fields; public API is forward(); adding a weight requires updating load_weights and the Gemma3 loader"
)]
#[allow(missing_debug_implementations)]
pub struct MultiModalProjector {
    /// `mm_soft_emb_norm.weight` `[vision_hidden]`, pre-shifted `weight + 1.0`
    /// at load (gemma3's RMSNorm uses the `1.0 + weight` gamma convention —
    /// gemma3.py imports `RMSNorm` from `language.py`).
    soft_emb_norm_w: Array,
    /// `mm_input_projection_weight` `[vision_hidden, text_hidden]`.
    input_projection_w: Array,
    norm_eps: f32,
    patches_per_side: usize,
    tokens_per_side: usize,
    pool_kernel: usize,
}

impl MultiModalProjector {
    /// `vision_out`: `[1, num_patches, vision_hidden]` (e.g. `[1, 4096, 1152]`).
    /// Returns `[1, mm_tokens_per_image, text_hidden]` (e.g. `[1, 256, 2560]`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward(&self, vision_out: &Array, device: Device) -> Result<Array> {
        let s = vision_out.shape();
        let (b, num_patches, vh) = (s[0], s[1], s[2]);
        let pps = self.patches_per_side as i32; // 64
        let tps = self.tokens_per_side as i32; // 16
        let k = self.pool_kernel as i32; // 4

        if num_patches != pps * pps {
            return Err(Error::Model(format!(
                "gemma3 projector: {num_patches} patches != {pps}^2 (patches_per_side)"
            )));
        }

        // mlx-vlm:
        // x.transpose(0,2,1) -> [b, vh, num_patches]
        // .reshape(b, vh, pps, pps) -> [b, vh, 64, 64]
        // .transpose(0,2,3,1) -> [b, 64, 64, vh]
        // avg_pool(kernel=4, stride=4) -> [b, 16, 16, vh]
        // .transpose(0,3,1,2).flatten(2) -> [b, vh, 256]
        // .transpose(0,2,1) -> [b, 256, vh]
        let g = vision_out
            .transpose(&[0, 2, 1], device)?
            .reshape(&[b, vh, pps, pps], device)?;

        // AvgPool2d kernel=stride=k over the (h, w) spatial axes. Built as a
        // reshape into [b, vh, tps, k, tps, k] then mean over the two k axes.
        // (Non-overlapping pool == block average == reshape + reduce.)
        let blocked = g.reshape(&[b, vh, tps, k, tps, k], device)?;
        let inv_k = scalar_f32(1.0 / k as f32);
        let m1 = multiply(&sum_axis(&blocked, 5, device)?, &inv_k, device)?; // [b,vh,tps,k,tps]
        let pooled = multiply(&sum_axis(&m1, 3, device)?, &inv_k, device)?; // [b,vh,tps,tps]

        // -> [b, num_soft, vh]
        let num_soft = tps * tps;
        let pooled = pooled
            .reshape(&[b, vh, num_soft], device)?
            .transpose(&[0, 2, 1], device)?;

        // mm_soft_emb_norm (scale-bearing RMSNorm over vision_hidden).
        let normed = rms_norm(&pooled, Some(&self.soft_emb_norm_w), self.norm_eps, device)?;

        // einsum "btm,md->btd": [b, num_soft, vh] @ [vh, text_hidden].
        matmul(&normed, &self.input_projection_w, device)
    }
}

// ---------------------------------------------------------------------------
// VisionModel — SigLIP tower (patch embed -> encoder -> post_layernorm)
// ---------------------------------------------------------------------------

/// Gemma3 SigLIP-style vision encoder (`vision_tower.vision_model.*`).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed layer struct — private weight fields; public API is forward(); adding a weight requires updating load_weights and the Gemma3 loader"
)]
#[allow(missing_debug_implementations)]
pub struct VisionModel {
    cfg: Gemma3VisionConfig,
    /// Conv2d patch embedding weight `[hidden, kH, kW, in_channels]` (MLX layout).
    patch_embedding_w: Array,
    patch_embedding_b: Array,
    /// Learned 1D position embedding `[num_patches, hidden]`.
    position_embedding: Array,
    encoder: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
}

impl VisionModel {
    /// Return a reference to the vision tower configuration.
    pub fn config(&self) -> &Gemma3VisionConfig {
        &self.cfg
    }

    /// Run the SigLIP tower over one preprocessed image. `pv` is CHW f32 in
    /// `[-1, 1]` at `image_size x image_size`. Returns `[1, num_patches, hidden]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward(&self, pv: &Gemma3PixelValues, device: Device) -> Result<Array> {
        let p = self.cfg.patch_size;
        let img = self.cfg.image_size;
        if pv.height != img || pv.width != img {
            return Err(Error::Model(format!(
                "gemma3 vision: image {}x{} != expected {img}x{img}",
                pv.height, pv.width
            )));
        }
        let pps = img / p; // patches per side (64)
        let num_patches = pps * pps; // 4096
        let hidden = self.cfg.hidden_size;
        let in_ch = self.cfg.num_channels;

        // ---- patchify on host: CHW f32 -> [num_patches, kH*kW*C] -------------
        // Conv2d weight is [hidden, kH, kW, C] (MLX layout). The matmul flattens
        // it to [hidden, kH*kW*C], so the patch feature order must be
        // (kH, kW, C) — row-major over the patch window.
        let feat_len = p * p * in_ch;
        let mut patches = vec![0.0_f32; num_patches * feat_len];
        let n_pixels = img * img;
        for ph in 0..pps {
            for pw in 0..pps {
                let dst = (ph * pps + pw) * feat_len;
                for r in 0..p {
                    for c in 0..p {
                        let y = ph * p + r;
                        let x = pw * p + c;
                        for ch in 0..in_ch {
                            let src = ch * n_pixels + y * img + x;
                            let off = (r * p + c) * in_ch + ch;
                            patches[dst + off] = pv.pixel_values[src];
                        }
                    }
                }
            }
        }
        let np = num_patches as i32;
        let fl = feat_len as i32;
        let patch_arr = Array::from_bytes(f32_bytes(&patches), &[np, fl], Dtype::F32)?;

        // patch_embedding: [hidden, feat_len] -> patches @ W^T + bias.
        let w_flat = self
            .patch_embedding_w
            .reshape(&[hidden as i32, fl], device)?;
        let mut h = matmul(&patch_arr, &w_flat.transpose(&[1, 0], device)?, device)?;
        h = add(&h, &self.patch_embedding_b, device)?;

        // ---- learned 1D position embeddings (sequential patch index) --------
        // position_embedding is [num_patches, hidden], indexed 0..num_patches.
        h = add(&h, &self.position_embedding, device)?;

        // [num_patches, hidden] -> [1, num_patches, hidden]
        h = h.reshape(&[1, np, hidden as i32], device)?;

        debug!(
            num_patches,
            pps,
            blocks = self.encoder.len(),
            "gemma3 vision: encoder forward (full bidirectional)"
        );
        for blk in &self.encoder {
            h = blk.forward(&h, device)?;
        }

        self.post_layernorm.forward(&h, device)
    }
}

#[inline]
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes; from_bytes copies immediately.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

#[inline]
fn i32_bytes(v: &[i32]) -> &[u8] {
    // SAFETY: i32 is 4 bytes; from_bytes copies immediately.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// Preprocessor — fixed 896x896 resize + rescale + normalize(0.5/0.5)
// ---------------------------------------------------------------------------

/// Gemma3 preprocessed pixel values: CHW f32 in `[-1, 1]`, `image_size^2`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — four fields are the complete preprocessed-image contract passed to the Gemma3 vision encoder; adding a field requires updating all Gemma3ImageProcessor::process callers"
)]
#[derive(Debug, Clone)]
pub struct Gemma3PixelValues {
    /// Flat `[3, H, W]` f32, channels-first.
    pub pixel_values: Vec<f32>,
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
    /// Soft tokens this image consumes (`mm_tokens_per_image`, 256).
    pub num_soft_tokens: usize,
}

/// Gemma3 image preprocessor — fixed-size resize (no aspect-ratio budgeting).
///
/// Matches `Gemma3ImageProcessor` (`preprocessor_config.json`): resize to a
/// fixed `size.height x size.width` (896x896) with `resample=2` (BILINEAR),
/// rescale `1/255`, normalize per-channel `mean=std=0.5` -> maps `[0,1] -> [-1,1]`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed preprocessor — private fields; public API is process(); adding a field requires updating new() and from_config()"
)]
#[derive(Debug, Clone)]
pub struct Gemma3ImageProcessor {
    image_size: usize,
    rescale_factor: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    mm_tokens_per_image: usize,
}

impl Gemma3ImageProcessor {
    /// Read `preprocessor_config.json` (+ `config.json` `mm_tokens_per_image`).
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Config(format!("gemma3: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!("gemma3: malformed preprocessor_config.json: {e}"))
        })?;
        let image_size = v
            .get("size")
            .and_then(|s| s.get("height"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(896) as usize;
        let rescale_factor = v
            .get("rescale_factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0 / 255.0) as f32;
        let read3 = |key: &str, dflt: f32| -> [f32; 3] {
            v.get(key)
                .and_then(|a| a.as_array())
                .map_or([dflt; 3], |a| {
                    let g = |i: usize| {
                        a.get(i)
                            .and_then(serde_json::Value::as_f64)
                            .unwrap_or(f64::from(dflt)) as f32
                    };
                    [g(0), g(1), g(2)]
                })
        };
        let image_mean = read3("image_mean", 0.5);
        let image_std = read3("image_std", 0.5);

        // mm_tokens_per_image lives in the top-level config.json.
        let mm_tokens_per_image =
            Gemma3VisionConfig::from_model_dir(model_dir)?.map_or(256, |c| c.mm_tokens_per_image);

        Ok(Gemma3ImageProcessor {
            image_size,
            rescale_factor,
            image_mean,
            image_std,
            mm_tokens_per_image,
        })
    }

    /// Decode image bytes -> resize to `image_size^2` -> rescale -> normalize.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn preprocess(&self, bytes: &[u8]) -> Result<Gemma3PixelValues> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| Error::Model(format!("gemma3: image decode failed: {e}")))?;
        let sz = self.image_size as u32;
        // resample=2 (PIL BILINEAR) ~= image crate Triangle filter.
        let resized = img.resize_exact(sz, sz, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let s = self.image_size;
        let n_pixels = s * s;
        let mut pv = vec![0.0_f32; 3 * n_pixels];
        for y in 0..s {
            for x in 0..s {
                let px = rgb.get_pixel(x as u32, y as u32);
                for ch in 0..3 {
                    let v01 = f32::from(px[ch]) * self.rescale_factor;
                    let norm = (v01 - self.image_mean[ch]) / self.image_std[ch];
                    pv[ch * n_pixels + y * s + x] = norm;
                }
            }
        }
        Ok(Gemma3PixelValues {
            pixel_values: pv,
            height: s,
            width: s,
            num_soft_tokens: self.mm_tokens_per_image,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma3 image token ids (from medgemma config.json)
// ---------------------------------------------------------------------------

/// `image_token_index` — the soft-token id scattered with vision features.
pub const IMAGE_TOKEN_ID: u32 = 262_144;
/// `boi_token_index` — begin-of-image marker.
pub const BOI_TOKEN_ID: u32 = 255_999;
/// `eoi_token_index` — end-of-image marker.
pub const EOI_TOKEN_ID: u32 = 256_000;

// ---------------------------------------------------------------------------
// build_inputs_embeds — scale vision features + masked_scatter at IMAGE_TOKEN
// ---------------------------------------------------------------------------

/// Build the merged `inputs_embeds` for a Gemma3 image prompt.
///
/// Faithful host port of mlx-vlm `gemma3.py::Model.get_input_embeddings` +
/// `prepare_inputs_for_multimodal`:
/// 1. `inputs_embeds = embed_tokens(input_ids) * sqrt(hidden)` (scaled text).
/// 2. `image_features = projector(vision_tower(pixels))` -> `[1, n_soft, hidden]`.
/// 3. `scaled = image_features / sqrt(hidden_size)`.
/// 4. masked_scatter `scaled` into the run of [`IMAGE_TOKEN_ID`] positions.
///
/// Returns `(inputs_embeds [1, seq, hidden], ids [seq])`. The ids array is the
/// plain token ids (Gemma3 has no per-layer-input gating, so no masking is
/// needed — it is forwarded only for the trunk's positional accounting).
///
/// Errors if the total `IMAGE_TOKEN_ID` count != sum of images' `num_soft_tokens`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn build_inputs_embeds(
    model: &super::model::Gemma3Text,
    vision: &VisionModel,
    projector: &MultiModalProjector,
    images: &[Gemma3PixelValues],
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
        .filter(|(_, &t)| t == IMAGE_TOKEN_ID)
        .map(|(i, _)| i)
        .collect();
    let expected: usize = images.iter().map(|pv| pv.num_soft_tokens).sum();
    if img_positions.len() != expected {
        return Err(Error::Model(format!(
            "gemma3 image: {} image-token ({IMAGE_TOKEN_ID}) positions in prompt != \
             {expected} vision soft tokens ({} image(s)) — scatter would misalign",
            img_positions.len(),
            images.len()
        )));
    }
    info!(
        image_tokens = img_positions.len(),
        images = images.len(),
        seq,
        "gemma3 image: building inputs_embeds (token count == soft tokens)"
    );

    // ---- scaled text embeddings: embed_tokens(ids) * sqrt(hidden) ----------
    let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
    let ids_arr = Array::from_bytes(i32_bytes(&ids_i32), &[seq as i32], Dtype::I32)?;
    let h_raw = model.embed_tokens.forward(&ids_arr, device)?;
    let embed_scale = scalar_f32((model.cfg.hidden_size as f32).sqrt());
    let mut embeds = multiply(&h_raw, &embed_scale, device)?;
    embeds = embeds.reshape(&[1, seq as i32, hidden], device)?;
    let embeds_dtype = embeds.dtype();

    // Net image-feature scale = 1.0.
    //
    // mlx-vlm applies the embed scale `sqrt(hidden)` to the *whole*
    // `inputs_embeds` (text + scattered image) inside the language model
    // (`gemma3/language.py`: `h *= hidden_size**0.5`), and scales the image
    // features by `1/sqrt(hidden)` in `prepare_inputs_for_multimodal` *before*
    // scatter — the two cancel for image positions, net 1.0.
    //
    // rMLX bakes the `sqrt(hidden)` text scale into `embeds` here (the trunk's
    // `forward_arr_embeds` does NOT re-scale precomputed embeds), so image
    // features are scattered at their raw projector value (the `1/sqrt(hidden)`
    // and the trunk `sqrt(hidden)` are folded out).

    let mut cursor = 0usize;
    for (img_idx, pv) in images.iter().enumerate() {
        let n_soft = pv.num_soft_tokens;
        // Vision tower (f32) -> projector (f32) -> [1, n_soft, hidden].
        // Short-circuit on a cache hit for the post-projector array.
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
            projector.forward(&vision.forward(pv, device)?, device)
        })?;
        let fs = feats.shape();
        if fs.first().copied() != Some(1)
            || fs.get(1).copied() != Some(n_soft as i32)
            || fs.get(2).copied() != Some(hidden)
        {
            return Err(Error::Model(format!(
                "gemma3 image: vision feature shape {fs:?} != [1, {n_soft}, {hidden}] \
                 for image {img_idx}"
            )));
        }
        // Net scale 1.0 — scatter the raw projector output. Cast to embed dtype.
        let feats = feats.astype(embeds_dtype, device)?;

        let run = &img_positions[cursor..cursor + n_soft];
        let first = run[0];
        let contiguous = run.iter().enumerate().all(|(k, &p)| p == first + k);
        if !contiguous {
            return Err(Error::Model(format!(
                "gemma3 image: image-token positions for image {img_idx} are not \
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

    Ok((embeds, ids_arr))
}

// ---------------------------------------------------------------------------
// Loader — vision_tower.vision_model.* + multi_modal_projector.* (all bf16).
// ---------------------------------------------------------------------------

/// Load the Gemma3 SigLIP tower + `MultiModalProjector` from a snapshot dir.
/// Vision/projector tensors are unquantized BF16 in medgemma; loaded as f32.
pub fn load_vision_tower(
    model_dir: &Path,
    cfg: &Gemma3VisionConfig,
) -> Result<(VisionModel, MultiModalProjector)> {
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
            "gemma3 vision: tensor '{name}' not found in any shard"
        )))
    }

    let load_linear = |base: &str| -> Result<Linear> {
        Ok(Linear {
            weight: load_f32(&shards, &format!("{base}.weight"))?,
            bias: load_f32(&shards, &format!("{base}.bias"))?,
        })
    };
    let load_ln = |base: &str| -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: load_f32(&shards, &format!("{base}.weight"))?,
            bias: load_f32(&shards, &format!("{base}.bias"))?,
            eps: cfg.layer_norm_eps,
        })
    };

    info!(
        layers = cfg.num_hidden_layers,
        hidden = cfg.hidden_size,
        heads = cfg.num_attention_heads,
        patch = cfg.patch_size,
        image = cfg.image_size,
        "gemma3: loading SigLIP vision tower (f32)"
    );

    let vm = "vision_tower.vision_model";
    let patch_embedding_w = load_f32(&shards, &format!("{vm}.embeddings.patch_embedding.weight"))?;
    let patch_embedding_b = load_f32(&shards, &format!("{vm}.embeddings.patch_embedding.bias"))?;
    let position_embedding = load_f32(
        &shards,
        &format!("{vm}.embeddings.position_embedding.weight"),
    )?;

    let scale = (cfg.head_dim as f32).powf(-0.5);
    let mut encoder = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let b = format!("{vm}.encoder.layers.{i}");
        let sa = format!("{b}.self_attn");
        encoder.push(EncoderLayer {
            layer_norm1: load_ln(&format!("{b}.layer_norm1"))?,
            layer_norm2: load_ln(&format!("{b}.layer_norm2"))?,
            attn: Attention {
                q_proj: load_linear(&format!("{sa}.q_proj"))?,
                k_proj: load_linear(&format!("{sa}.k_proj"))?,
                v_proj: load_linear(&format!("{sa}.v_proj"))?,
                out_proj: load_linear(&format!("{sa}.out_proj"))?,
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim,
                scale,
            },
            mlp: Mlp {
                fc1: load_linear(&format!("{b}.mlp.fc1"))?,
                fc2: load_linear(&format!("{b}.mlp.fc2"))?,
            },
        });
    }
    let post_layernorm = load_ln(&format!("{vm}.post_layernorm"))?;

    let vision = VisionModel {
        cfg: cfg.clone(),
        patch_embedding_w,
        patch_embedding_b,
        position_embedding,
        encoder,
        post_layernorm,
    };

    // Projector (two tensors, no quant).
    //
    // `mm_soft_emb_norm` is the **gamma+1** RMSNorm (gemma3.py imports
    // `RMSNorm` from language.py: `rms_norm(x, 1.0 + weight, eps)`). rMLX's
    // `rms_norm` uses the weight as-is, so pre-shift `weight + 1.0` at load.
    let soft_emb_norm_w_raw = load_f32(&shards, "multi_modal_projector.mm_soft_emb_norm.weight")?;
    let soft_emb_norm_w = add(&soft_emb_norm_w_raw, &scalar_f32(1.0), Device::Cpu)?;
    let projector = MultiModalProjector {
        soft_emb_norm_w,
        input_projection_w: load_f32(&shards, "multi_modal_projector.mm_input_projection_weight")?,
        norm_eps: cfg.mm_norm_eps,
        patches_per_side: cfg.patches_per_side(),
        tokens_per_side: cfg.tokens_per_side(),
        pool_kernel: cfg.pool_kernel(),
    };

    info!(
        layers = cfg.num_hidden_layers,
        "gemma3: vision tower + multimodal projector loaded"
    );
    Ok((vision, projector))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;
