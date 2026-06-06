//! Qwen3-VL image preprocessing — smart-resize patchifier.
//!
//! Faithful port of `mlx-vlm/.../qwen3_vl/processing_qwen3_vl.py`:
//! `Qwen3VLImageProcessor._process_one` + `_smart_resize_image`. Produces the
//! `pixel_values` tensor `[num_patches, in_ch * tps * ps * ps]` and the
//! `image_grid_thw = (grid_t, grid_h, grid_w)` consumed by the vision tower and
//! the 3D M-RoPE index.
//!
//! Pipeline:
//! 1. Decode bytes -> RGB u8 HWC.
//! 2. `smart_resize(h, w, factor = ps*merge, min_pixels, max_pixels)`.
//! 3. Bicubic resize to `(resized_h, resized_w)`.
//! 4. Rescale `u8 * 1/255`, normalize `(v - mean) / std` per channel.
//! 5. Duplicate along the temporal axis (`temporal_patch_size` frames).
//! 6. Reshape/transpose into patch order

//!
//! Defaults (image processor): `patch_size=16`, `temporal_patch_size=2`,
//! `merge_size=2`, `image_mean=image_std=[0.5,0.5,0.5]`,
//! `min_pixels=56*56`, `max_pixels=14*14*4*1280`. These match the
//! `mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit` preprocessor_config.json.

use rmlx_core::error::{Error, Result};

/// Preprocessor config (subset of `preprocessor_config.json` we use).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — fields are the complete Qwen3-VL image preprocessor contract; adding a field requires updating Default and from_model_dir"
)]
#[derive(Debug, Clone)]
pub struct Qwen3VlImageConfig {
    /// Spatial patch size in pixels.
    pub patch_size: usize,
    /// Temporal patch size for video inputs.
    pub temporal_patch_size: usize,
    /// Spatial merge factor (2 = 2×2 → 1 token).
    pub merge_size: usize,
    /// Minimum pixel budget for smart resize.
    pub min_pixels: usize,
    /// Maximum pixel budget for smart resize.
    pub max_pixels: usize,
    /// Pixel value rescaling factor (1/255).
    pub rescale_factor: f32,
    /// Per-channel mean for normalization `(v - mean) / std`.
    pub image_mean: [f32; 3],
    /// Per-channel std for normalization.
    pub image_std: [f32; 3],
}

impl Default for Qwen3VlImageConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 56 * 56,
            max_pixels: 14 * 14 * 4 * 1280,
            rescale_factor: 1.0 / 255.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }
}

impl Qwen3VlImageConfig {
    /// Load from `<model_dir>/preprocessor_config.json` (best-effort; falls back
    /// to defaults for any missing field).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        let mut cfg = Self::default();
        let Ok(data) = std::fs::read(&path) else {
            return Ok(cfg);
        };
        let raw: serde_json::Value = serde_json::from_slice(&data)
            .map_err(|e| Error::Loader(format!("malformed preprocessor_config.json: {e}")))?;
        let o = raw.as_object();
        if let Some(o) = o {
            if let Some(v) = o.get("patch_size").and_then(serde_json::Value::as_u64) {
                cfg.patch_size = v as usize;
            }
            if let Some(v) = o
                .get("temporal_patch_size")
                .and_then(serde_json::Value::as_u64)
            {
                cfg.temporal_patch_size = v as usize;
            }
            if let Some(v) = o.get("merge_size").and_then(serde_json::Value::as_u64) {
                cfg.merge_size = v as usize;
            }
            if let Some(v) = o.get("min_pixels").and_then(serde_json::Value::as_u64) {
                cfg.min_pixels = v as usize;
            }
            if let Some(v) = o.get("max_pixels").and_then(serde_json::Value::as_u64) {
                cfg.max_pixels = v as usize;
            }
            // size: { shortest_edge / longest_edge } override.
            if let Some(size) = o.get("size").and_then(|v| v.as_object()) {
                if let Some(v) = size
                    .get("shortest_edge")
                    .and_then(serde_json::Value::as_u64)
                {
                    cfg.min_pixels = v as usize;
                }
                if let Some(v) = size.get("longest_edge").and_then(serde_json::Value::as_u64) {
                    cfg.max_pixels = v as usize;
                }
            }
            if let Some(v) = o.get("rescale_factor").and_then(serde_json::Value::as_f64) {
                cfg.rescale_factor = v as f32;
            }
            if let Some(arr) = o.get("image_mean").and_then(|v| v.as_array()) {
                for (i, x) in arr.iter().take(3).enumerate() {
                    if let Some(f) = x.as_f64() {
                        cfg.image_mean[i] = f as f32;
                    }
                }
            }
            if let Some(arr) = o.get("image_std").and_then(|v| v.as_array()) {
                for (i, x) in arr.iter().take(3).enumerate() {
                    if let Some(f) = x.as_f64() {
                        cfg.image_std[i] = f as f32;
                    }
                }
            }
        }
        Ok(cfg)
    }
}

/// Preprocessed image: pixel values + the patch grid.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — three fields are the complete preprocessed-image contract passed to the Qwen3-VL vision encoder; adding a field requires updating all preprocess() callers"
)]
#[derive(Debug, Clone)]
pub struct Qwen3VlPixelValues {
    /// Row-major `[grid_t*grid_h*grid_w, in_ch*tps*ps*ps]` f32.
    pub pixel_values: Vec<f32>,
    /// `(grid_t, grid_h, grid_w)` in patch units.
    pub grid_thw: (usize, usize, usize),
    /// Number of merged (LM-visible) image tokens =
    /// `grid_t*grid_h*grid_w / merge_size^2`.
    pub num_soft_tokens: usize,
}

/// `smart_resize` (image variant) — ports HF qwen2_vl `smart_resize`.
fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    let ratio = h.max(w) / h.min(w);
    if ratio > 200.0 {
        return Err(Error::Model(format!(
            "qwen3_vl preprocess: aspect ratio {ratio:.1} exceeds 200"
        )));
    }
    let f = factor as f64;
    let round_by = |x: f64| -> usize { ((x / f).round() * f) as usize };
    let floor_by = |x: f64| -> usize { (factor).max(((x / f).floor() * f) as usize) };
    let ceil_by = |x: f64| -> usize { ((x / f).ceil() * f) as usize };

    let mut h_bar = round_by(h);
    let mut w_bar = round_by(w);
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = floor_by(h / beta);
        w_bar = floor_by(w / beta);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ceil_by(h * beta);
        w_bar = ceil_by(w * beta);
    }
    Ok((h_bar, w_bar))
}

/// Decode bytes to RGB u8 HWC. Returns `(rgb_hwc, height, width)`.
fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| Error::Model(format!("qwen3_vl: image decode failed: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err(Error::Model("qwen3_vl: zero-size image".into()));
    }
    Ok((rgb.into_raw(), h, w))
}

/// Bicubic resize via the `image` crate (CatmullRom ≈ PIL BICUBIC). For solid /
/// large-region test images the filter choice is immaterial; CatmullRom is the
/// closest bicubic the crate ships. Returns row-major `[dh*dw*3]` u8 HWC.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn resize_rgb8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let buf = image::RgbImage::from_raw(sw as u32, sh as u32, src.to_vec())
        .expect("rgb buffer size mismatch");
    let dynimg = image::DynamicImage::ImageRgb8(buf);
    let resized = dynimg.resize_exact(
        dw as u32,
        dh as u32,
        image::imageops::FilterType::CatmullRom,
    );
    resized.to_rgb8().into_raw()
}

/// Preprocess image bytes into Qwen3-VL pixel values + grid.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn preprocess(bytes: &[u8], cfg: &Qwen3VlImageConfig) -> Result<Qwen3VlPixelValues> {
    let (rgb_hwc, h, w) = decode_rgb(bytes)?;
    let factor = cfg.patch_size * cfg.merge_size;
    let (rh, rw) = smart_resize(h, w, factor, cfg.min_pixels, cfg.max_pixels)?;
    let resized = resize_rgb8(&rgb_hwc, w, h, rw, rh);

    // Rescale + normalize into CHW f32 [3, rh, rw].
    let n_pixels = rh * rw;
    let mut chw = vec![0.0_f32; 3 * n_pixels];
    let rf = cfg.rescale_factor;
    for i in 0..n_pixels {
        for c in 0..3 {
            let v = f32::from(resized[i * 3 + c]) * rf;
            chw[c * n_pixels + i] = (v - cfg.image_mean[c]) / cfg.image_std[c];
        }
    }

    let ps = cfg.patch_size;
    let tps = cfg.temporal_patch_size;
    let ms = cfg.merge_size;
    let grid_t = 1usize;
    let grid_h = rh / ps;
    let grid_w = rw / ps;
    let c = 3usize;

    // The reference builds:
    // patches = repeat(chw[None,None], tps, axis=1) # [1, tps, 3, rh, rw]
    // reshape (1, grid_t, tps, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
    // transpose (0,1,4,7,5,8,3,2,6,9)
    // reshape (grid_t*grid_h*grid_w, C*tps*ps*ps)
    //
    // We compute the flattened output directly. For a single (t,t-dup) frame the
    // temporal axis is just `chw` duplicated tps times.
    //
    // Output row index for patch (gh_b, gw_b, m_h, m_w):
    // row = ((gh_b*ms + m_h_outer?) ...) — actually after transpose the patch
    // order is (grid_h//ms, grid_w//ms, ms, ms) i.e. merge-grouped, matching
    // the vision tower's expected merge order.
    // Within a row the feature layout is (C, tps, ps, ps).
    let hb = grid_h / ms;
    let wb = grid_w / ms;
    let feat_len = c * tps * ps * ps;
    let num_patches = grid_t * grid_h * grid_w;
    let mut out = vec![0.0_f32; num_patches * feat_len];

    let mut row = 0usize;
    for a in 0..hb {
        for b in 0..wb {
            for mh in 0..ms {
                for mw in 0..ms {
                    // patch top-left in the resized image.
                    let patch_h0 = (a * ms + mh) * ps;
                    let patch_w0 = (b * ms + mw) * ps;
                    let row_base = row * feat_len;
                    // feature layout (C, tps, ps, ps).
                    let mut fi = 0usize;
                    for ch in 0..c {
                        let ch_base = ch * n_pixels;
                        for _t in 0..tps {
                            for py in 0..ps {
                                let img_row = (patch_h0 + py) * rw;
                                for px in 0..ps {
                                    let pix = ch_base + img_row + (patch_w0 + px);
                                    out[row_base + fi] = chw[pix];
                                    fi += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    debug_assert_eq!(row, num_patches);

    let num_soft_tokens = num_patches / (ms * ms);
    Ok(Qwen3VlPixelValues {
        pixel_values: out,
        grid_thw: (grid_t, grid_h, grid_w),
        num_soft_tokens,
    })
}

#[cfg(test)]
#[path = "image_preprocess_tests.rs"]
mod tests;
