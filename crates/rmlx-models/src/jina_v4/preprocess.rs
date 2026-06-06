//! jina-v4 image front-end: Qwen2.5-VL image preprocessing.
//!
//! Faithful CPU port of HF `Qwen2VLImageProcessor` (`transformers`
//! `models/qwen2_vl/image_processing_qwen2_vl.py`) — `smart_resize` +
//! `_preprocess` — exactly as jina-embeddings-v4 invokes it
//! (`custom_st.py:79-105`: `Image.open(...).convert("RGB")` →
//! `processor.process_images`).
//!
//! Scope is the image → tensor front-end **only**. It produces the flat
//! `pixel_values` buffer + `image_grid_thw` triple in the exact memory
//! layout the vision tower's PatchEmbed will consume. The ViT
//! itself is NOT implemented here. Pure CPU, no MLX `Array`, no GPU.
//!
//! ## Fidelity notes
//!
//! - **Resample filter (bit-exact).** jina-embeddings-v4
//!   loads its processor with `use_fast=True`
//!   (`modeling_jina_embeddings_v4.py:145`), so the authoritative reference
//!   is `Qwen2VLImageProcessorFast` → torchvision
//!   `interpolate(mode="bicubic", antialias=True)`, **not** PIL. The `image`
//!   crate's `CatmullRom` differs from it by ±1 LSB on ~10% of pixels — a
//!   delta the Qwen2.5-VL ViT amplifies on small/smooth images (vision
//!   cosine → ~0.86, far below the 0.999 gate). [`pil_resize_rgb8`] is a
//!   from-scratch faithful port of torchvision/aten's separable uint8
//!   antialias bicubic (Keys kernel a=-0.5, `support = 2·max(scale,1)`,
//!   `PRECISION_BITS = 14` fixed point); **verified bit-identical** to
//!   `torchvision.transforms.v2.functional.resize(..., BICUBIC,
//! antialias=True)` (0 of N pixels differ). No deviation remains.
//! - **Rounding.** `smart_resize` replicates Python `round` (banker's /
//!   round-half-to-even, via [`round_half_even`]), `math.floor`, and
//!   `math.ceil` precisely — these decide the integer grid and must match
//!   HF bit-for-bit.
//! - **Channel order.** HF normalizes in HWC then `to_channel_dimension_format`
//!   → CHW (`data_format=ChannelDimension.FIRST`). We resize/rescale/normalize
//!   in HWC and index as CHW directly in the patchify gather (no physical
//!   transpose needed — see [`patchify`]).

#![allow(clippy::float_cmp)]
use std::path::Path;

use rmlx_core::error::{Error, Result};

/// CLIP-style per-channel normalization + Qwen2-VL patch geometry, sourced
/// from the model's own `preprocessor_config.json` (the model's values win
/// over the recon doc — recon truncated the mean/std).
///
/// Defaults are the verified `Qwen2VLImageProcessor` values and are only used
/// as a fallback when a key is absent from the JSON.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — fields are the complete Qwen2VLImageProcessor config contract; adding a field requires updating Default and from_preprocessor_json"
)]
#[derive(Debug, Clone)]
pub struct ImagePreprocessConfig {
    /// Spatial patch size of the vision encoder (14).
    pub patch_size: usize,
    /// LLM-side merge size (2). `factor = patch_size * merge_size`.
    pub merge_size: usize,
    /// Temporal patch size (2). A single still image is duplicated to fill
    /// this many "frames" so `grid_t = 1`.
    pub temporal_patch_size: usize,
    /// Lower bound on resized pixel count (`size.shortest_edge`, 3136).
    pub min_pixels: usize,
    /// Upper bound on resized pixel count (`size.longest_edge`, 602112).
    pub max_pixels: usize,
    /// `1/255` rescale factor.
    pub rescale_factor: f32,
    /// Per-channel mean (RGB), CLIP values.
    pub image_mean: [f32; 3],
    /// Per-channel std (RGB), CLIP values.
    pub image_std: [f32; 3],
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        // Verified against
        // open-models/jinaai__jina-embeddings-v4/preprocessor_config.json.
        Self {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 3136,
            max_pixels: 602_112,
            rescale_factor: 1.0 / 255.0,
            // CLIP mean/std from preprocessor_config.json
            // ([0.48145466, 0.4578275, 0.40821073] /
            // [0.26862954, 0.26130258, 0.27577711]). Written at exact f32
            // precision — these are the identical IEEE-754 single values the
            // Python float32 image pipeline uses, so this is the faithful
            // (not lossy) literal.
            image_mean: [0.481_454_67, 0.457_827_5, 0.408_210_72],
            image_std: [0.268_629_55, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl ImagePreprocessConfig {
    /// `factor = patch_size * merge_size` — both resized dims are divisible
    /// by this (HF `smart_resize(factor=patch_size*merge_size)`).
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Parse `preprocessor_config.json` (Qwen2VLImageProcessor schema).
    /// Missing keys fall back to [`Default`]. `min_pixels` / `max_pixels`
    /// follow HF precedence: explicit `size.shortest_edge` /
    /// `size.longest_edge` win, else top-level `min_pixels` / `max_pixels`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn from_preprocessor_json(v: &serde_json::Value) -> Result<Self> {
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

        // HF: size.shortest_edge / size.longest_edge take precedence over the
        // flat min_pixels / max_pixels keys (image_processing_qwen2_vl.py
        // __init__ + preprocess()).
        let size = v.get("size");
        let shortest = size
            .and_then(|s| s.get("shortest_edge"))
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize);
        let longest = size
            .and_then(|s| s.get("longest_edge"))
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize);

        Ok(Self {
            patch_size: u("patch_size", d.patch_size),
            merge_size: u("merge_size", d.merge_size),
            temporal_patch_size: u("temporal_patch_size", d.temporal_patch_size),
            min_pixels: shortest.unwrap_or_else(|| u("min_pixels", d.min_pixels)),
            max_pixels: longest.unwrap_or_else(|| u("max_pixels", d.max_pixels)),
            rescale_factor: f("rescale_factor", d.rescale_factor),
            image_mean: arr3("image_mean", d.image_mean),
            image_std: arr3("image_std", d.image_std),
        })
    }

    /// Parse from a model directory's `preprocessor_config.json`. If the file
    /// is absent, returns [`Default`] (the verified jina-v4 values) so the
    /// front-end still works on a bare snapshot.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = std::fs::read(&path)
            .map_err(|e| Error::Config(format!("jina-v4: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "jina-v4: malformed preprocessor_config.json at {}: {e}",
                path.display()
            ))
        })?;
        Self::from_preprocessor_json(&v)
    }
}

/// `(grid_t, grid_h, grid_w)` — the temporal/height/width patch grid. HF
/// returns this as `image_grid_thw`; fed to the vision tower's RoPE /
/// window-index and the M-RoPE merge.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed grid struct — three fields (t/h/w) are the complete patch-grid contract; adding a field requires updating all ImageGridThw construction sites and vision-tower RoPE callers"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGridThw {
    /// Temporal grid dimension (usually 1 for still images).
    pub t: usize,
    /// Height grid dimension (number of patch rows).
    pub h: usize,
    /// Width grid dimension (number of patch columns).
    pub w: usize,
}

impl ImageGridThw {
    /// Number of patch rows of `pixel_values` (`grid_t*grid_h*grid_w`).
    pub fn num_patches(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// Flat `pixel_values` + its grid. `data` is row-major
/// `[num_patches, feature_len]` where
/// `feature_len = 3 * temporal_patch_size * patch_size * patch_size`
/// and `num_patches = grid.num_patches()` — the exact layout HF's
/// `flatten_patches` produces and the vision tower's PatchEmbed consumes.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — four fields are the complete preprocessed-patch contract passed to the vision encoder; adding a field requires updating all preprocess() callers"
)]
#[derive(Debug, Clone)]
pub struct PixelValues {
    /// Row-major `[num_patches * feature_len]` f32 (post rescale+normalize).
    pub data: Vec<f32>,
    /// `3 * temporal_patch_size * patch_size * patch_size`.
    pub feature_len: usize,
    /// `grid_t * grid_h * grid_w`.
    pub num_patches: usize,
    /// The patch grid.
    pub grid: ImageGridThw,
}

/// Python `round` semantics: round-half-to-even (banker's rounding).
///
/// `math.floor`/`math.ceil` map to `f64::floor`/`ceil`, but `round(x)` in
/// Python 3 is half-to-even (`round(0.5) == 0`, `round(1.5) == 2`,
/// `round(2.5) == 2`) — Rust's `f64::round` is half-away-from-zero and would
/// diverge from HF's `smart_resize` at exact `.5` boundaries. We replicate
/// Python precisely so the integer grid matches bit-for-bit.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // Exactly .5 → round to even. `floor` is a non-negative integral
        // f64 here (called only on positive dim/factor ratios).
        let f_int = floor as i64;
        if f_int.rem_euclid(2) == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Faithful port of HF `smart_resize` (image_processing_qwen2_vl.py:76-102).
///
/// Rescales `(height, width)` so both are multiples of `factor` and the pixel
/// count is within `[min_pixels, max_pixels]`, preserving aspect as closely as
/// possible. Returns `(h_bar, w_bar)`.
///
/// Errors on the same `aspect_ratio > 200` guard HF raises.
///
/// `pub(crate)` — the M-RoPE / window-index code needs the resized
/// grid geometry; not part of the crate's external API.
pub(crate) fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(Error::Model(format!(
            "jina-v4 smart_resize: zero dimension ({height}x{width})"
        )));
    }
    let (hi, lo) = if height >= width {
        (height, width)
    } else {
        (width, height)
    };
    if (hi as f64) / (lo as f64) > 200.0 {
        return Err(Error::Model(format!(
            "jina-v4 smart_resize: absolute aspect ratio must be < 200, got {}",
            (hi as f64) / (lo as f64)
        )));
    }

    let factor_f = factor as f64;
    let h = height as f64;
    let w = width as f64;

    let mut h_bar = round_half_even(h / factor_f) * factor_f;
    let mut w_bar = round_half_even(w / factor_f) * factor_f;

    let max_p = max_pixels as f64;
    let min_p = min_pixels as f64;

    if h_bar * w_bar > max_p {
        let beta = ((h * w) / max_p).sqrt();
        h_bar = factor_f.max((h / beta / factor_f).floor() * factor_f);
        w_bar = factor_f.max((w / beta / factor_f).floor() * factor_f);
    } else if h_bar * w_bar < min_p {
        let beta = (min_p / (h * w)).sqrt();
        h_bar = (h * beta / factor_f).ceil() * factor_f;
        w_bar = (w * beta / factor_f).ceil() * factor_f;
    }

    Ok((h_bar as usize, w_bar as usize))
}

/// Decode `bytes` (PNG or JPEG; pure-Rust codecs, no system libs) to RGB8,
/// returning `(rgb_hwc, height, width)` with `rgb_hwc` row-major
/// `[h * w * 3]` u8 — equivalent to HF's
/// `Image.open(...).convert("RGB")` + `to_numpy_array` (HWC u8).
fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| Error::Model(format!("jina-v4 image decode failed: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err(Error::Model("jina-v4 image: zero-size image".into()));
    }
    Ok((rgb.into_raw(), h, w))
}

/// Patchify a normalized, resized image into HF `flatten_patches` layout.
///
/// `norm_hwc` is the resized+rescaled+normalized image, row-major HWC
/// `[rh * rw * 3]` f32. HF's reference does (conceptually):
///
/// 1. stack frames → `[t, C, rh, rw]` then temporal-pad to a multiple of
///    `temporal_patch_size`.
/// 2. `reshape(grid_t, tps, C, gh/ms, ms, ps, gw/ms, ms, ps)`
/// 3. `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)`
/// 4. `reshape(grid_t*grid_h*grid_w, C*tps*ps*ps)`
///
/// We compute the destination element directly from `(grid)` indices and
/// gather from `norm_hwc` — no intermediate transpose buffers. The two
/// temporal frames are identical (still image), so the temporal axis just
/// duplicates the spatial value.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn patchify(
    norm_hwc: &[f32],
    rh: usize,
    rw: usize,
    cfg: &ImagePreprocessConfig,
) -> Result<PixelValues> {
    let ps = cfg.patch_size;
    let ms = cfg.merge_size;
    let tps = cfg.temporal_patch_size;
    let channel = 3usize;

    let grid_t = 1usize; // single still image (post temporal-pad: tps frames / tps)
    let grid_h = rh / ps;
    let grid_w = rw / ps;
    if grid_h == 0 || grid_w == 0 || !grid_h.is_multiple_of(ms) || !grid_w.is_multiple_of(ms) {
        return Err(Error::Model(format!(
            "jina-v4 patchify: bad grid ({grid_h}x{grid_w}) for ps={ps} ms={ms} \
             from resized {rh}x{rw}"
        )));
    }

    let num_patches = grid_t * grid_h * grid_w;
    let feature_len = channel * tps * ps * ps;
    let mut out = vec![0.0_f32; num_patches * feature_len];

    // HF post-transpose axis order is:
    // (grid_t, gh_blk, gw_blk, ms_h, ms_w, C, tps, ps_h, ps_w)
    // flattened to row index = ((gt*gh_blk + a)*gw_blk + b)*ms*ms ... — but
    // the final reshape collapses dims 0..5 into `num_patches` and 5..9 into
    // `feature_len`. We iterate the destination directly:
    //
    // patch_row index space = grid_t × (gh/ms) × (gw/ms) × ms × ms
    // feature index space = C × tps × ps × ps
    //
    // The source pixel for (gh_blk,a, gw_blk,b, c, ph, pw):
    // image row = (gh_blk*ms + a)*ps + ph
    // image col = (gw_blk*ms + b)*ps + pw
    // Temporal frames are identical (still image) ⇒ value independent of `f`.
    let gh_blk = grid_h / ms;
    let gw_blk = grid_w / ms;

    let mut row = 0usize; // 0..num_patches
    for _gt in 0..grid_t {
        for hb in 0..gh_blk {
            for wb in 0..gw_blk {
                for a in 0..ms {
                    for b in 0..ms {
                        // feature layout: C, tps, ps_h, ps_w
                        let base = row * feature_len;
                        for c in 0..channel {
                            for fr in 0..tps {
                                for ph in 0..ps {
                                    let img_row = (hb * ms + a) * ps + ph;
                                    for pw in 0..ps {
                                        let img_col = (wb * ms + b) * ps + pw;
                                        // norm_hwc is HWC: [r*rw*3 + col*3 + c]
                                        let src = (img_row * rw + img_col) * channel + c;
                                        let dst = base + ((c * tps + fr) * ps + ph) * ps + pw;
                                        out[dst] = norm_hwc[src];
                                    }
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
    }
    debug_assert_eq!(row, num_patches);

    Ok(PixelValues {
        data: out,
        feature_len,
        num_patches,
        grid: ImageGridThw {
            t: grid_t,
            h: grid_h,
            w: grid_w,
        },
    })
}

/// Keys bicubic convolution kernel, `a = -0.5`.
///
/// This is the kernel **both** PIL `BICUBIC` (`Resample.c::bicubic_filter`)
/// **and** torchvision/aten `interpolate(mode="bicubic", antialias=True)`
/// use. jina-embeddings-v4 loads its processor with `use_fast=True`, so the
/// authoritative reference is the torchvision-AA path; verified bit-exact
/// against `torchvision.transforms.v2.functional.resize(..., BICUBIC,
/// antialias=True)` (impulse-probed aten weights match to 0.0, and the full
/// uint8 result matches 0/N pixels at `PRECISION_BITS = 14`). The antialias
/// window/normalization is identical to PIL's `precompute_coeffs`
/// (`support = 2 * max(scale,1)`, `arg = (k + xmin - center + 0.5) /
/// filterscale`) — the only aten-vs-PIL delta is the fixed-point precision
/// (aten 14, PIL 22), captured by `PRECISION_BITS` below.
// Horner form kept literal (NOT `mul_add`): the resampled u8 output is
// verified *bit-identical* to torchvision, and FMA fuses the mul+add at a
// different rounding than the reference's separate ops — `mul_add` would
// perturb the LSB and break that bit-exactness. Correctness over the
// micro-optimisation here.
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
///
/// Faithful port of `Pillow Resample.c::precompute_coeffs`: `filterscale =
/// max(in/out, 1)`, `support = 2.0 * filterscale`, `center = (xx+0.5)*scale`,
/// window `[xmin, xmin+n)` clamped to `[0, in_size)`, weights
/// `filter((x + xmin - center + 0.5) / filterscale)` then normalized to sum
/// 1. Returns `(bounds[out] = (xmin, n), weights[out][k])`.
#[allow(clippy::type_complexity)]
fn precompute_coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0_f64 * filterscale; // bicubic support == 2.0
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // PIL: xmin = (int)(center - support + 0.5), clamped >= 0.
        let mut xmin = (center - support + 0.5).floor() as isize;
        if xmin < 0 {
            xmin = 0;
        }
        let xmin = xmin as usize;
        // xmax = (int)(center + support + 0.5), clamped <= in_size; n = xmax-xmin.
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

/// Fixed-point precision for the 8-bit separable resample accumulator.
///
/// **14** — torchvision/aten's uint8 antialias resampler precision
/// (`aten _compute_weights_aa` for uint8). jina-v4 uses the fast
/// (torchvision) image processor, so this is the authoritative value:
/// verified the full uint8 resize is **bit-identical** to
/// `torchvision ... resize(BICUBIC, antialias=True)` at `PRECISION_BITS=14`
/// (0 of N pixels differ) and NOT at PIL's 22 (the slow PIL processor, which
/// jina does not use). The coefficient window/kernel are shared (see
/// [`pil_bicubic`] / [`precompute_coeffs`]).
const PRECISION_BITS: u32 = 14;

/// PIL `_clip8` over a fixed-point accumulator (`Resample.c::clip8`): shift
/// down `PRECISION_BITS` with clamp to `[0, 255]`. The `+ (1 <<
/// (PRECISION_BITS-1))` rounding bias is folded into the accumulator seed by
/// the caller (exactly as Pillow does).
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

/// Quantize one axis's normalized f64 coefficients to the `int` fixed point
/// (`(int) round(c * (1 << PRECISION_BITS))`, round-half-away-from-zero).
///
/// `(±0.5 + c*scale) as i32` kept literal (NOT `mul_add`): this is the exact
/// integer-coefficient quantization torchvision/aten performs; FMA fuses at a
/// different rounding and would perturb a coefficient by 1 ulp, breaking the
/// verified bit-exact resampled output.
#[allow(clippy::suboptimal_flops)]
fn quantize_coeffs(weights: &[Vec<f64>]) -> Vec<Vec<i32>> {
    let scale = f64::from(1u32 << PRECISION_BITS);
    weights
        .iter()
        .map(|w| {
            w.iter()
                .map(|&c| {
                    // PIL: c < 0 ? (int)(-0.5 + c*scale) : (int)(0.5 + c*scale)
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

/// Faithful PIL `Image.resize(BICUBIC)` for an RGB8 HWC buffer.
///
/// Bit-exact port of Pillow's 8-bpc separable resampler
/// (`src/libImaging/Resample.c`): per-axis coefficients precomputed in f64,
/// **normalized, then quantized to `int` fixed point** (`PRECISION_BITS`),
/// accumulated as integers with a `1 << (PRECISION_BITS-1)` rounding seed,
/// shifted + clamped via `clip8`. Two u8→u8 passes (horizontal then
/// vertical). The `image` crate's CatmullRom differs from this by ±1 LSB on
/// ~10% of pixels — a delta the Qwen2.5-VL ViT amplifies (small smooth
/// images → vision cosine ~0.86, far below the 0.999 gate). Verified
/// bit-identical to Pillow on the parity images.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn pil_resize_rgb8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let seed: i64 = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: (sh x sw) -> (sh x dw), u8 -> u8 (fixed point).
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
    // Vertical pass: (sh x dw) -> (dh x dw), u8 -> u8 (fixed point).
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

/// Preprocess one image (PNG/JPEG bytes) into Qwen2.5-VL `pixel_values` +
/// `image_grid_thw`, faithfully porting HF `Qwen2VLImageProcessor._preprocess`
/// for the single still-image path jina-embeddings-v4 uses.
///
/// Pipeline: decode→RGB → `smart_resize` → **PIL-faithful bicubic** resize
/// (bit-exact vs Pillow — see [`pil_resize_rgb8`]) → `* rescale_factor` →
/// `(x-mean)/std` per channel → patchify into the flat HF layout. Pure CPU;
/// output is plain f32.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn preprocess_image_bytes(bytes: &[u8], cfg: &ImagePreprocessConfig) -> Result<PixelValues> {
    let (rgb_hwc, h, w) = decode_rgb(bytes)?;

    let (rh, rw) = smart_resize(h, w, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;

    // Bit-exact PIL BICUBIC (HF uses PIL `Image.resize`, resample=3).
    let resized = pil_resize_rgb8(&rgb_hwc, w, h, rw, rh);

    // rescale (*1/255) then per-channel normalize (x-mean)/std, in HWC.
    let mut norm = vec![0.0_f32; rh * rw * 3];
    let rf = cfg.rescale_factor;
    let mean = cfg.image_mean;
    let std = cfg.image_std;
    for i in 0..(rh * rw) {
        for c in 0..3 {
            let v = f32::from(resized[i * 3 + c]) * rf;
            norm[i * 3 + c] = (v - mean[c]) / std[c];
        }
    }

    patchify(&norm, rh, rw, cfg)
}

/// Preprocess an image from a filesystem path (PNG/JPEG). Thin wrapper over
/// [`preprocess_image_bytes`] — reads the file then delegates.
pub fn preprocess_image_path(path: &Path, cfg: &ImagePreprocessConfig) -> Result<PixelValues> {
    let bytes = std::fs::read(path).map_err(|e| {
        Error::Model(format!(
            "jina-v4: cannot read image {}: {e}",
            path.display()
        ))
    })?;
    preprocess_image_bytes(&bytes, cfg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "preprocess_tests.rs"]
mod tests;
