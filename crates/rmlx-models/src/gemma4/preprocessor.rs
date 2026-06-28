//! Gemma4 image preprocessor — aspect-ratio-preserving resize to patch budget.
//!
//! Faithful CPU port of `mlx_vlm.models.gemma4.processing_gemma4`
//! (`Gemma4ImageProcessor` / `aspect_ratio_preserving_resize`).
//!
//! ## Algorithm (matches mlx-vlm exactly)
//!
//! 1. Decode bytes → RGB u8 HWC.
//! 2. `aspect_ratio_preserving_resize`: find the largest `(target_h, target_w)`
//!    that satisfies:
//!    - `target_h` and `target_w` are multiples of `side_mult = pooling_kernel_size * patch_size`
//! - `target_h * target_w / patch_size^2 ≤ max_patches` where
//!   `max_patches = max_soft_tokens * pooling_kernel_size^2`
//!
//! using `factor = sqrt(target_px / (h * w))` then `floor(factor * dim / side_mult) * side_mult`.
//! Edge cases: if either result is 0, clamp the zero axis to `side_mult` and
//! cap the other at `max_side_length = (max_patches // pooling_kernel_size^2) * side_mult`.
//! Resize with PIL-faithful **BICUBIC** (mlx-vlm: `Image.BICUBIC`, `resample=3`).
//! 3. Rescale: `pixel * rescale_factor` (1/255 by default) → `f32 ∈ [0, 1]`.
//! 4. Optionally normalize per channel `(v - mean[c]) / std[c]` (off in e4b snapshot).
//! 5. Transpose HWC → CHW (channels-first).
//! 6. Wrap in `[1, C, H, W]` batch array.
//!
//! ## Config defaults (from `processor_config.json` `image_processor` block)
//!
//! The e4b snapshot (`mlx-community__gemma-4-e4b-it-mxfp8/processor_config.json`)
//! has these values under the `image_processor` key:
//!
//! | Field | e4b value | Constructor default |
//! |---------------------|------------|---------------------|
//! | `patch_size` | 16 | 16 |
//! | `max_soft_tokens` | 280 | 280 |
//! | `pooling_kernel_size` | 3 | 3 |
//! | `rescale_factor` | 1/255 | 1/255 |
//! | `do_rescale` | true | true |
//! | `do_normalize` | false | false |
//! | `image_mean` | [0,0,0] | [0,0,0] |
//! | `image_std` | [1,1,1] | [1,1,1] |
//!
//! The config's `size: {height: 224, width: 224}` is the minimum fallback
//! (not used by `aspect_ratio_preserving_resize` — it uses `max_soft_tokens`).
//!
//! ## Resize filter
//!
//! mlx-vlm calls `Image.BICUBIC` (PIL resample=3) which is the standard Keys
//! cubic kernel (a=-0.5). We reuse the same PIL-faithful separable bicubic
//! implementation already in jina_v4/preprocess.rs, inlined here to keep the
//! module self-contained and avoid cross-crate coupling at this layer.
//!
//! ## Output: `Gemma4PixelValues`
//!
//! `pixel_values: Vec<f32>` — flat row-major `[1 * C * H * W]` in channels-first
//! order (C=3). `shape: [1, 3, height, width]`. Consumer
//! indexes it as `[batch, channel, row, col]`.

use std::path::Path;

use rmlx_core::error::{Error, Result};

/// Largest image-token budget the Gemma4 processor will honour for an
/// override. Mirrors the upstream `_SUPPORTED_SOFT_TOKENS = (70, 140, 280,
/// 560, 1120)` ceiling in `transformers`/`mlx_vlm` `processing_gemma4`: the
/// reference processor only accepts a `max_soft_tokens` from that set, with
/// `1120` the maximum. A caller-supplied budget override is clamped to this
/// value so a request can never size the resize target beyond what the
/// reference vision front-end was validated for. The lower bound is `1`
/// (a zero budget would degenerate the resize).
pub const MAX_SUPPORTED_SOFT_TOKENS: usize = 1120;

/// Resolve the effective image-token budget for one preprocess call.
///
/// Precedence is `override > config default`: when `override_tokens` is
/// `Some(n)` the budget is `n` clamped to `[1, MAX_SUPPORTED_SOFT_TOKENS]`;
/// when `None` the model's configured `config_default` (the snapshot's
/// `processor_config.json` `max_soft_tokens`, typically 280) is used
/// unchanged. The result is itself clamped to the safe upper bound so a
/// stale config default can never exceed the validated ceiling either.
#[inline]
pub fn resolve_max_soft_tokens(override_tokens: Option<usize>, config_default: usize) -> usize {
    override_tokens
        .unwrap_or(config_default)
        .clamp(1, MAX_SUPPORTED_SOFT_TOKENS)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Gemma4 image preprocessing parameters, parsed from `processor_config.json`
/// (`image_processor` sub-object) or falling back to documented defaults.
///
/// Defaults are the verified values for all released Gemma 4 checkpoints
/// (e4b, 12b, 27b). If the config file is absent or a key is missing the
/// corresponding default is used and the caller can document the assumption.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — fields are the complete Gemma4 image-processor config contract; adding a field requires updating Default and from_processor_config_json"
)]
#[derive(Debug, Clone)]
pub struct Gemma4ImageProcessorConfig {
    /// Spatial patch size of the vision encoder (16 for all Gemma 4 variants).
    pub patch_size: usize,
    /// Maximum number of soft tokens the image may produce (280 for e4b).
    pub max_soft_tokens: usize,
    /// Spatial pooling kernel applied after patchification (3 for all Gemma 4).
    pub pooling_kernel_size: usize,
    /// Pixel rescale factor applied to u8 values (`1/255` by default).
    pub rescale_factor: f32,
    /// Whether to apply per-channel mean/std normalization after rescaling.
    /// False for e4b (and all current Gemma 4 checkpoints).
    pub do_normalize: bool,
    /// Per-channel mean (RGB). Only used when `do_normalize = true`.
    pub image_mean: [f32; 3],
    /// Per-channel std (RGB). Only used when `do_normalize = true`.
    pub image_std: [f32; 3],
}

impl Default for Gemma4ImageProcessorConfig {
    fn default() -> Self {
        // Verified against mlx-community__gemma-4-e4b-it-mxfp8/processor_config.json
        // image_processor block. do_normalize=false per config (image_mean/std are
        // present but do_normalize is explicitly false, so they are unused).
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            rescale_factor: 1.0 / 255.0,
            do_normalize: false,
            image_mean: [0.0, 0.0, 0.0],
            image_std: [1.0, 1.0, 1.0],
        }
    }
}

impl Gemma4ImageProcessorConfig {
    /// Parse from the `image_processor` sub-object inside `processor_config.json`
    /// (the Gemma4 snapshot layout). Missing keys fall back to [`Default`].
    ///
    /// Pass the **inner** `image_processor` JSON value, not the outer
    /// `processor_config.json` root. See [`from_processor_config_json`] for the
    /// full-file variant.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn from_image_processor_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
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
        let b = |key: &str, dflt: bool| -> bool {
            v.get(key)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(dflt)
        };
        let arr3 = |key: &str, dflt: [f32; 3]| -> [f32; 3] {
            v.get(key)
                .and_then(serde_json::Value::as_array)
                .filter(|a| a.len() == 3)
                .and_then(|a| {
                    let mut out = [0.0_f32; 3];
                    for (i, e) in a.iter().enumerate() {
                        out[i] = e.as_f64()? as f32;
                    }
                    Some(out)
                })
                .unwrap_or(dflt)
        };

        Self {
            patch_size: u("patch_size", d.patch_size),
            max_soft_tokens: u("max_soft_tokens", d.max_soft_tokens),
            pooling_kernel_size: u("pooling_kernel_size", d.pooling_kernel_size),
            rescale_factor: f("rescale_factor", d.rescale_factor),
            do_normalize: b("do_normalize", d.do_normalize),
            image_mean: arr3("image_mean", d.image_mean),
            image_std: arr3("image_std", d.image_std),
        }
    }

    /// Parse from the root of `processor_config.json`. Extracts the nested
    /// `image_processor` object if present; falls back to [`Default`] if absent.
    pub fn from_processor_config_json(v: &serde_json::Value) -> Self {
        match v.get("image_processor") {
            Some(ip) => Self::from_image_processor_json(ip),
            None => Self::default(),
        }
    }

    /// Load from a model directory's `processor_config.json`. If the file is
    /// absent or the `image_processor` key is missing, returns [`Default`]
    /// (the verified Gemma4 e4b values).
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("processor_config.json");
        if !path.exists() {
            tracing::debug!(
                path = %path.display(),
                "gemma4: processor_config.json absent, using preprocessor defaults"
            );
            return Ok(Self::default());
        }
        let data = std::fs::read(&path)
            .map_err(|e| Error::Config(format!("gemma4: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "gemma4: malformed processor_config.json at {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self::from_processor_config_json(&v))
    }

    /// `side_mult = pooling_kernel_size * patch_size` — both target dims are
    /// multiples of this value.
    #[inline]
    pub fn side_mult(&self) -> usize {
        self.pooling_kernel_size * self.patch_size
    }

    /// Total patch budget: `max_soft_tokens * pooling_kernel_size^2`.
    /// Target pixel count is `max_patches * patch_size^2`.
    #[inline]
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// Result of Gemma4 image preprocessing.
///
/// `pixel_values` is flat row-major `[1 * 3 * H * W]` f32 in **channels-first**
/// order: `[batch=1, C=3, height, width]`, indexed as
/// `pixel_values[((0 * 3 + c) * height + y) * width + x]`.
///
/// `num_soft_tokens` is the number of soft (image) tokens this image will
/// consume in the LLM: `(H // patch_size) * (W // patch_size) // pooling_kernel_size^2`.
/// It is always a member of `_SUPPORTED_SOFT_TOKENS = {70, 140, 280, 560, 1120}`
/// when the config `max_soft_tokens` is one of those values (the e4b default is 280).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — four fields are the complete preprocessed-image contract passed to the vision encoder; adding a field requires updating all Gemma4ImageProcessor::process callers"
)]
#[derive(Debug, Clone)]
pub struct Gemma4PixelValues {
    /// Flat `[1, 3, height, width]` f32, channels-first.
    pub pixel_values: Vec<f32>,
    /// Resized image height (multiple of `side_mult`).
    pub height: usize,
    /// Resized image width (multiple of `side_mult`).
    pub width: usize,
    /// LLM soft token count for this image.
    pub num_soft_tokens: usize,
}

// ---------------------------------------------------------------------------
// Preprocessor
// ---------------------------------------------------------------------------

/// Gemma4 image preprocessor.
///
/// Stateless after construction. Thread-safe (all methods take `&self`).
/// Construct from [`Gemma4ImageProcessorConfig`] or use [`Gemma4ImageProcessor::default()`].
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed preprocessor — field is private config; public API is process() and process_batch(); adding a field requires updating new() and Default"
)]
#[derive(Debug, Clone)]
pub struct Gemma4ImageProcessor {
    cfg: Gemma4ImageProcessorConfig,
}

impl Gemma4ImageProcessor {
    /// Construct from explicit config.
    pub fn new(cfg: Gemma4ImageProcessorConfig) -> Self {
        Self { cfg }
    }

    /// Construct with defaults (verified Gemma4 e4b values).
    pub fn with_defaults() -> Self {
        Self::new(Gemma4ImageProcessorConfig::default())
    }

    /// Load from a model directory (reads `processor_config.json`).
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let cfg = Gemma4ImageProcessorConfig::from_model_dir(model_dir)?;
        tracing::info!(
            patch_size = cfg.patch_size,
            max_soft_tokens = cfg.max_soft_tokens,
            pooling_kernel_size = cfg.pooling_kernel_size,
            do_normalize = cfg.do_normalize,
            "gemma4 image processor config loaded"
        );
        Ok(Self::new(cfg))
    }

    /// Preprocess image bytes (PNG, JPEG, or any format the `image` crate handles)
    /// into Gemma4 pixel values.
    ///
    /// Pipeline:
    /// 1. Decode → RGB u8 HWC.
    /// 2. `aspect_ratio_preserving_resize` to patch-budget target dims.
    /// 3. Rescale `u8 * rescale_factor` → f32.
    /// 4. If `do_normalize`: apply `(v - mean[c]) / std[c]` per channel.
    /// 5. Transpose HWC → CHW.
    /// 6. Wrap into `[1, 3, H, W]` output.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn preprocess(&self, bytes: &[u8]) -> Result<Gemma4PixelValues> {
        let (rgb_hwc, h, w) = decode_rgb(bytes)?;

        let (target_h, target_w) = aspect_ratio_preserving_resize(h, w, &self.cfg)?;

        tracing::debug!(
            src_h = h,
            src_w = w,
            target_h,
            target_w,
            "gemma4 preprocess: resize target"
        );

        // Resize using PIL-faithful BICUBIC (matches mlx-vlm Image.BICUBIC).
        let resized = pil_resize_rgb8(&rgb_hwc, w, h, target_w, target_h);

        // Build CHW f32 output.
        let n_pixels = target_h * target_w;
        let mut chw = vec![0.0_f32; 3 * n_pixels];
        let rf = self.cfg.rescale_factor;

        if self.cfg.do_normalize {
            let mean = self.cfg.image_mean;
            let std = self.cfg.image_std;
            for i in 0..n_pixels {
                for c in 0..3 {
                    let v = f32::from(resized[i * 3 + c]) * rf;
                    // CHW index: channel c, pixel i
                    chw[c * n_pixels + i] = (v - mean[c]) / std[c];
                }
            }
        } else {
            for i in 0..n_pixels {
                for c in 0..3 {
                    let v = f32::from(resized[i * 3 + c]) * rf;
                    chw[c * n_pixels + i] = v;
                }
            }
        }

        // Add batch dim: [1, 3, H, W] — prepend one copy (already correct, just
        // document that the consumer treats the flat vec as batch-1).
        let num_soft_tokens = (target_h / self.cfg.patch_size) * (target_w / self.cfg.patch_size)
            / (self.cfg.pooling_kernel_size * self.cfg.pooling_kernel_size);

        tracing::debug!(
            target_h,
            target_w,
            num_soft_tokens,
            "gemma4 preprocess: done"
        );

        Ok(Gemma4PixelValues {
            pixel_values: chw,
            height: target_h,
            width: target_w,
            num_soft_tokens,
        })
    }

    /// Convenience: preprocess image from a filesystem path.
    pub fn preprocess_path(&self, path: &Path) -> Result<Gemma4PixelValues> {
        let bytes = std::fs::read(path).map_err(|e| {
            Error::Model(format!("gemma4: cannot read image {}: {e}", path.display()))
        })?;
        self.preprocess(&bytes)
    }

    /// Borrow the config (for callers that need patch_size / soft-token info).
    pub fn config(&self) -> &Gemma4ImageProcessorConfig {
        &self.cfg
    }

    /// Clone this processor with `max_soft_tokens` replaced by `n` (clamped to
    /// the safe upper bound via [`resolve_max_soft_tokens`]).
    ///
    /// Used to apply a per-request image-token budget override without
    /// mutating the shared, load-time processor. A larger budget raises
    /// `max_patches`, so the aspect-ratio resize keeps more pixels and the
    /// resulting [`Gemma4PixelValues::num_soft_tokens`] grows for the same
    /// image. The resize math already caps the per-side length, so no extra
    /// guard is needed beyond the budget clamp.
    #[must_use]
    pub fn with_max_soft_tokens(&self, n: usize) -> Self {
        let mut cfg = self.cfg.clone();
        cfg.max_soft_tokens = resolve_max_soft_tokens(Some(n), self.cfg.max_soft_tokens);
        Self::new(cfg)
    }
}

// ---------------------------------------------------------------------------
// aspect_ratio_preserving_resize
// ---------------------------------------------------------------------------

/// Compute `(target_h, target_w)` for Gemma4's patch-budget resize.
///
/// Faithful port of `Gemma4ImageProcessor.aspect_ratio_preserving_resize`
/// in `mlx_vlm/models/gemma4/processing_gemma4.py`.
///
/// Target constraints:
/// - Both dims are multiples of `side_mult = pooling_kernel_size * patch_size`.
/// - Patch count `(target_h / patch_size) * (target_w / patch_size)` ≤ `max_patches`
///   where `max_patches = max_soft_tokens * pooling_kernel_size^2`.
/// - Aspect ratio is preserved as closely as possible via the scale factor
///   `sqrt(target_px / (h * w))` where `target_px = max_patches * patch_size^2`.
///
/// Does NOT perform the resize itself — only returns the target dimensions.
/// Returns an error if both computed dims are zero (degenerate 0×0 image).
pub fn aspect_ratio_preserving_resize(
    height: usize,
    width: usize,
    cfg: &Gemma4ImageProcessorConfig,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(Error::Model(format!(
            "gemma4 preprocess: zero-size image ({height}x{width})"
        )));
    }

    let max_patches = cfg.max_patches();
    let patch_size = cfg.patch_size;
    let side_mult = cfg.side_mult();

    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / (height as f64 * width as f64)).sqrt();

    let mut target_h = (factor * height as f64 / side_mult as f64).floor() as usize * side_mult;
    let mut target_w = (factor * width as f64 / side_mult as f64).floor() as usize * side_mult;

    // Edge case: both zero → error (matches Python ValueError).
    if target_h == 0 && target_w == 0 {
        return Err(Error::Model(format!(
            "gemma4 preprocess: aspect_ratio_preserving_resize would produce 0x0 \
             image from {height}x{width} with max_soft_tokens={}, patch_size={patch_size}",
            cfg.max_soft_tokens
        )));
    }

    // max_side_length: the longest a single side can be while fitting the patch budget.
    // Python: `(max_patches // pooling_kernel_size**2) * side_mult`
    let pk2 = cfg.pooling_kernel_size * cfg.pooling_kernel_size;
    let max_side_length = (max_patches / pk2) * side_mult;

    // One axis zero → clamp to side_mult; cap the other at max_side_length.
    if target_h == 0 {
        target_h = side_mult;
        target_w = ((width / height) * side_mult).min(max_side_length);
        // Clamp target_w to at least side_mult so it's a valid multiple.
        if target_w == 0 {
            target_w = side_mult;
        }
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = ((height / width) * side_mult).min(max_side_length);
        if target_h == 0 {
            target_h = side_mult;
        }
    }

    Ok((target_h, target_w))
}

// ---------------------------------------------------------------------------
// Image decode
// ---------------------------------------------------------------------------

/// Decode bytes (PNG/JPEG/etc.) to RGB u8 HWC via the `image` crate.
/// Returns `(rgb_hwc, height, width)` with `rgb_hwc` row-major `[h * w * 3]`.
fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| Error::Model(format!("gemma4: image decode failed: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err(Error::Model("gemma4: zero-size image".into()));
    }
    Ok((rgb.into_raw(), h, w))
}

// ---------------------------------------------------------------------------
// PIL-faithful BICUBIC resize (Keys a=-0.5)
//
// This is the same algorithm as jina_v4/preprocess.rs — reproduced here so
// gemma4's preprocessor is self-contained and does not create a cross-module
// dependency. The implementation is verified to be bit-faithful to PIL's
// BICUBIC resampler (Image.BICUBIC == resample=3), which is what mlx-vlm calls.
// ---------------------------------------------------------------------------

/// Keys cubic convolution kernel, `a = -0.5`.
///
/// PIL `BICUBIC` (`Resample.c::bicubic_filter`) uses this kernel.
/// mlx-vlm calls `Image.BICUBIC` (resample=3), so this is the correct filter.
#[allow(clippy::suboptimal_flops)]
fn pil_bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Precompute per-output-pixel PIL resample coefficients along one axis.
/// Faithful port of `Pillow Resample.c::precompute_coeffs`.
#[allow(clippy::type_complexity)]
fn precompute_coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0_f64 * filterscale;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5).floor() as isize;
        if xmin < 0 {
            xmin = 0;
        }
        let xmin = xmin as usize;
        let mut xmax = (center + support + 0.5).floor() as isize;
        if xmax > in_size as isize {
            xmax = in_size as isize;
        }
        let n = (xmax as usize).saturating_sub(xmin);
        let mut w = Vec::with_capacity(n);
        let mut sum = 0.0_f64;
        for k in 0..n {
            let arg = ((k + xmin) as f64 - center + 0.5) / filterscale;
            let val = pil_bicubic(arg);
            w.push(val);
            sum += val;
        }
        if sum != 0.0 {
            for v in &mut w {
                *v /= sum;
            }
        }
        bounds.push((xmin, n));
        weights.push(w);
    }
    (bounds, weights)
}

/// Fixed-point precision for the separable uint8 resampler (PIL value = 22).
///
/// mlx-vlm uses `Image.BICUBIC` (the PIL slow-path processor), so PIL's
/// `PRECISION_BITS = 22` is the correct value here — unlike jina_v4 which
/// uses the torchvision fast-path (`PRECISION_BITS = 14`). However, since
/// the Gemma4 vision encoder is not bit-sensitivity-tested the same way, and
/// the actual values post-rescale are f32, a ±1 LSB difference in the uint8
/// intermediate has negligible effect on the normalized float. We use 22 here
/// for PIL fidelity.
const PRECISION_BITS: u32 = 22;

fn clip8_fixed(acc: i64) -> u8 {
    let v = acc >> PRECISION_BITS;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

/// Quantize coefficients to fixed-point (PIL precision).
#[allow(clippy::suboptimal_flops)]
fn quantize_coeffs(weights: &[Vec<f64>]) -> Vec<Vec<i32>> {
    let scale = f64::from(1u32 << PRECISION_BITS);
    weights
        .iter()
        .map(|w| {
            w.iter()
                .map(|&c| {
                    if c < 0.0 {
                        (-0.5 + c * scale) as i32
                    } else {
                        (0.5 + c * scale) as i32
                    }
                })
                .collect()
        })
        .collect()
}

/// PIL-faithful BICUBIC separable resize for an RGB8 HWC buffer.
///
/// Matches `Image.resize((target_w, target_h), resample=Image.BICUBIC)` as
/// called in mlx-vlm `aspect_ratio_preserving_resize`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn pil_resize_rgb8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let seed: i64 = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: (sh × sw) → (sh × dw).
    let (hb, hw_f) = precompute_coeffs(sw, dw);
    let hw = quantize_coeffs(&hw_f);
    let mut tmp = vec![0u8; sh * dw * 3];
    for y in 0..sh {
        let srow = y * sw * 3;
        for (xx, ((xmin, n), wts)) in hb.iter().zip(hw.iter()).enumerate() {
            let mut acc = [seed; 3];
            for (k, &wk) in wts.iter().take(*n).enumerate() {
                let sp = srow + (xmin + k) * 3;
                let wk = i64::from(wk);
                acc[0] += i64::from(src[sp]) * wk;
                acc[1] += i64::from(src[sp + 1]) * wk;
                acc[2] += i64::from(src[sp + 2]) * wk;
            }
            let dp = (y * dw + xx) * 3;
            tmp[dp] = clip8_fixed(acc[0]);
            tmp[dp + 1] = clip8_fixed(acc[1]);
            tmp[dp + 2] = clip8_fixed(acc[2]);
        }
    }

    // Vertical pass: (sh × dw) → (dh × dw).
    let (vb, vw_f) = precompute_coeffs(sh, dh);
    let vw = quantize_coeffs(&vw_f);
    let mut out = vec![0u8; dh * dw * 3];
    for (yy, ((ymin, n), wts)) in vb.iter().zip(vw.iter()).enumerate() {
        for x in 0..dw {
            let mut acc = [seed; 3];
            for (k, &wk) in wts.iter().take(*n).enumerate() {
                let sp = ((ymin + k) * dw + x) * 3;
                let wk = i64::from(wk);
                acc[0] += i64::from(tmp[sp]) * wk;
                acc[1] += i64::from(tmp[sp + 1]) * wk;
                acc[2] += i64::from(tmp[sp + 2]) * wk;
            }
            let dp = (yy * dw + x) * 3;
            out[dp] = clip8_fixed(acc[0]);
            out[dp + 1] = clip8_fixed(acc[1]);
            out[dp + 2] = clip8_fixed(acc[2]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "preprocessor_tests.rs"]
mod tests;
