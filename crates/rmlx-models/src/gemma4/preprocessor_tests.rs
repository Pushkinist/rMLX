use super::*;

// ---- aspect_ratio_preserving_resize -----------------------------------

/// Verify target dims for a 640×480 image against hand-computed values.
///
/// With default config: patch_size=16, max_soft_tokens=280,
/// pooling_kernel_size=3.
///
/// max_patches = 280 * 9 = 2520
/// target_px = 2520 * 256 = 645120
/// factor = sqrt(645120 / (640*480)) = sqrt(645120 / 307200)
/// = sqrt(2.0999...) ≈ 1.44914...
/// side_mult = 3 * 16 = 48
///
/// target_h = floor(1.44914 * 480 / 48) * 48 = floor(14.491) * 48 = 14 * 48 = 672
/// target_w = floor(1.44914 * 640 / 48) * 48 = floor(19.321) * 48 = 19 * 48 = 912
///
/// soft_tokens = (672/16) * (912/16) / 9 = 42 * 57 / 9 = 2394 / 9 = 266
/// 266 ≤ 280 ✓
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn resize_640x480_expected_dims() {
    let cfg = Gemma4ImageProcessorConfig::default();
    let (th, tw) = aspect_ratio_preserving_resize(480, 640, &cfg).unwrap();
    assert_eq!(th, 672, "height");
    assert_eq!(tw, 912, "width");

    // Both dims are multiples of side_mult.
    let side_mult = cfg.side_mult();
    assert_eq!(th % side_mult, 0);
    assert_eq!(tw % side_mult, 0);

    // Soft token count is within the budget.
    let soft = (th / cfg.patch_size) * (tw / cfg.patch_size)
        / (cfg.pooling_kernel_size * cfg.pooling_kernel_size);
    assert!(
        soft <= cfg.max_soft_tokens,
        "soft={soft} > max={}",
        cfg.max_soft_tokens
    );
}

/// Square image: target dims should be equal (aspect preserved).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn resize_square_preserves_aspect() {
    let cfg = Gemma4ImageProcessorConfig::default();
    let (th, tw) = aspect_ratio_preserving_resize(512, 512, &cfg).unwrap();
    assert_eq!(th, tw, "square image must produce equal dims");
    assert_eq!(th % cfg.side_mult(), 0);
}

/// Tall portrait (2:1 aspect).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn resize_portrait_aspect() {
    let cfg = Gemma4ImageProcessorConfig::default();
    let (th, tw) = aspect_ratio_preserving_resize(1024, 512, &cfg).unwrap();
    // Both must be multiples of side_mult.
    assert_eq!(th % cfg.side_mult(), 0);
    assert_eq!(tw % cfg.side_mult(), 0);
    // Soft token count within budget.
    let soft = (th / cfg.patch_size) * (tw / cfg.patch_size)
        / (cfg.pooling_kernel_size * cfg.pooling_kernel_size);
    assert!(soft <= cfg.max_soft_tokens);
    // Height should be at least as large as width (portrait aspect preserved).
    assert!(th >= tw, "portrait: height {th} should be >= width {tw}");
}

// ---- gemma4_preprocessor_shape (DoD test) ----------------------------

/// Main DoD test: 640×480 synthetic PNG → `[1, 3, H', W']`,
/// `num_soft_tokens ≤ max_soft_tokens`, pixel values in `[0, 1]`.
///
/// Note on `_SUPPORTED_SOFT_TOKENS = {70, 140, 280, 560, 1120}`: that
/// constant in mlx-vlm is used only by `Gemma4VideoProcessor.__init__`
/// to validate the per-frame budget. `Gemma4ImageProcessor` does NOT snap
/// the output token count to this set — it produces whatever the
/// aspect-ratio resize yields, which for 640×480 with max_soft_tokens=280
/// is 266 (≤ 280 budget, valid). This test asserts the correct invariant:
/// `num_soft_tokens ≤ max_soft_tokens`.
///
/// This is the primary acceptance test.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma4_preprocessor_shape() {
    let proc = Gemma4ImageProcessor::with_defaults();

    // Synthesize a 640×480 RGB gradient PNG.
    let (iw, ih) = (640u32, 480u32);
    let mut img = image::RgbImage::new(iw, ih);
    for y in 0..ih {
        for x in 0..iw {
            img.put_pixel(
                x,
                y,
                image::Rgb([
                    (x * 255 / iw) as u8,
                    (y * 255 / ih) as u8,
                    ((x + y) % 256) as u8,
                ]),
            );
        }
    }
    let mut png_bytes: Vec<u8> = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png_bytes)
            .write_image(img.as_raw(), iw, ih, image::ExtendedColorType::Rgb8)
            .expect("encode test PNG");
    }

    let pv = proc.preprocess(&png_bytes).expect("preprocess");

    // Shape: [1, 3, H', W'] → flat length = 3 * H' * W'.
    let expected_len = 3 * pv.height * pv.width;
    assert_eq!(pv.pixel_values.len(), expected_len, "flat buffer length");

    // Height and width are multiples of side_mult.
    let side_mult = proc.config().side_mult();
    assert_eq!(pv.height % side_mult, 0, "height % side_mult");
    assert_eq!(pv.width % side_mult, 0, "width % side_mult");

    // Soft token count is within the patch budget.
    assert!(
        pv.num_soft_tokens <= proc.config().max_soft_tokens,
        "num_soft_tokens={} > max={}",
        pv.num_soft_tokens,
        proc.config().max_soft_tokens
    );

    // Pixel values in [0, 1] (do_normalize=false by default).
    for &v in &pv.pixel_values {
        assert!(v.is_finite(), "non-finite pixel value");
        assert!((0.0..=1.0).contains(&v), "pixel {v} outside [0,1]");
    }

    // Expected dims (verified by hand above): 672×912.
    assert_eq!(pv.height, 672, "expected resized height 672");
    assert_eq!(pv.width, 912, "expected resized width 912");
    // 266 soft tokens: (672/16) * (912/16) / 9 = 42 * 57 / 9 = 266.
    assert_eq!(
        pv.num_soft_tokens, 266,
        "expected 266 soft tokens for 640×480"
    );
}

/// Verify solid-color image: all pixels must be the rescaled value (1/255 * u8).
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn gemma4_preprocessor_solid_color_rescale() {
    let proc = Gemma4ImageProcessor::with_defaults();

    let (iw, ih) = (192u32, 192u32); // multiple of side_mult=48
    let img = image::RgbImage::from_pixel(iw, ih, image::Rgb([128u8, 64u8, 32u8]));
    let mut png: Vec<u8> = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), iw, ih, image::ExtendedColorType::Rgb8)
            .expect("encode");
    }

    let pv = proc.preprocess(&png).expect("preprocess");
    let rf = proc.config().rescale_factor;

    // A constant-color image stays constant after rescaling.
    // Channel layout: [3, H*W] → offsets [0..n), [n..2n), [2n..3n).
    let n = pv.height * pv.width;
    let expected_c0 = 128.0_f32 * rf;
    let expected_c1 = 64.0_f32 * rf;
    let expected_c2 = 32.0_f32 * rf;

    for i in 0..n {
        assert!(
            (pv.pixel_values[i] - expected_c0).abs() < 2e-3,
            "c0 pixel {i}: got {}, expected {expected_c0}",
            pv.pixel_values[i]
        );
        assert!(
            (pv.pixel_values[n + i] - expected_c1).abs() < 2e-3,
            "c1 pixel {i}: got {}, expected {expected_c1}",
            pv.pixel_values[n + i]
        );
        assert!(
            (pv.pixel_values[2 * n + i] - expected_c2).abs() < 2e-3,
            "c2 pixel {i}: got {}, expected {expected_c2}",
            pv.pixel_values[2 * n + i]
        );
    }
}

// ---- image-token budget override --------------------------------------

/// `resolve_max_soft_tokens`: precedence + clamp.
#[test]
fn resolve_budget_precedence_and_clamp() {
    // None → config default, unchanged (when default within bounds).
    assert_eq!(resolve_max_soft_tokens(None, 280), 280);
    // Some(n) within bounds → n (override wins over config default).
    assert_eq!(resolve_max_soft_tokens(Some(560), 280), 560);
    // Override above the safe upper bound is clamped.
    assert_eq!(
        resolve_max_soft_tokens(Some(100_000), 280),
        MAX_SUPPORTED_SOFT_TOKENS
    );
    assert_eq!(resolve_max_soft_tokens(Some(1120), 280), 1120);
    // Zero override is clamped up to 1 (never degenerate the resize).
    assert_eq!(resolve_max_soft_tokens(Some(0), 280), 1);
    // A stale config default above the ceiling is also clamped when used.
    assert_eq!(
        resolve_max_soft_tokens(None, 100_000),
        MAX_SUPPORTED_SOFT_TOKENS
    );
}

/// `with_max_soft_tokens`: produces a processor whose config carries the
/// clamped override; the shared processor is left untouched.
#[test]
fn with_max_soft_tokens_replaces_budget() {
    let proc = Gemma4ImageProcessor::with_defaults();
    assert_eq!(proc.config().max_soft_tokens, 280);

    let raised = proc.with_max_soft_tokens(560);
    assert_eq!(raised.config().max_soft_tokens, 560);
    // Source processor unchanged.
    assert_eq!(proc.config().max_soft_tokens, 280);

    // Above-ceiling override clamps.
    let clamped = proc.with_max_soft_tokens(5000);
    assert_eq!(clamped.config().max_soft_tokens, MAX_SUPPORTED_SOFT_TOKENS);
}

/// A raised budget must yield MORE soft tokens for the same image — the core
/// budget-override invariant. The resize keeps more pixels under a larger
/// `max_patches`, so `num_soft_tokens` strictly increases (the image here is
/// large enough that the default budget is the binding constraint).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn raised_budget_increases_soft_tokens() {
    // 2000×1500 source: comfortably above the 280-token budget's resize target.
    let (h, w) = (1500usize, 2000usize);

    let soft_for = |max_soft: usize| -> usize {
        let cfg = Gemma4ImageProcessorConfig {
            max_soft_tokens: max_soft,
            ..Gemma4ImageProcessorConfig::default()
        };
        let (th, tw) = aspect_ratio_preserving_resize(h, w, &cfg).unwrap();
        (th / cfg.patch_size) * (tw / cfg.patch_size)
            / (cfg.pooling_kernel_size * cfg.pooling_kernel_size)
    };

    let soft_default = soft_for(280);
    let soft_raised = soft_for(1120);
    assert!(
        soft_raised > soft_default,
        "raised budget must yield more soft tokens: default={soft_default} raised={soft_raised}"
    );
    // Each stays within its own budget (resize never exceeds max_soft_tokens).
    assert!(soft_default <= 280, "default soft={soft_default} > 280");
    assert!(soft_raised <= 1120, "raised soft={soft_raised} > 1120");
}

/// Verify config parsing from a JSON object matching the e4b snapshot layout.
#[test]
fn config_parses_image_processor_block() {
    let v: serde_json::Value = serde_json::json!({
        "image_processor": {
            "do_convert_rgb": true,
            "do_normalize": false,
            "do_rescale": true,
            "do_resize": true,
            "image_mean": [0.0, 0.0, 0.0],
            "image_processor_type": "Gemma4ImageProcessor",
            "image_seq_length": 280,
            "image_std": [1.0, 1.0, 1.0],
            "max_soft_tokens": 280,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": { "height": 224, "width": 224 }
        }
    });
    let cfg = Gemma4ImageProcessorConfig::from_processor_config_json(&v);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.max_soft_tokens, 280);
    assert_eq!(cfg.pooling_kernel_size, 3);
    assert!(!cfg.do_normalize);
    assert!((cfg.rescale_factor - 1.0 / 255.0).abs() < 1e-9);
    assert_eq!(cfg.side_mult(), 48);
    assert_eq!(cfg.max_patches(), 2520);
}

/// Live parse test: skipped when model not present.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn config_parses_e4b_snapshot_if_present() {
    let Some(dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
    else {
        eprintln!("SKIP: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let dir = dir_buf.as_path();
    if !dir.join("processor_config.json").exists() {
        eprintln!("SKIP: e4b snapshot not present");
        return;
    }
    let cfg = Gemma4ImageProcessorConfig::from_model_dir(dir).unwrap();
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.max_soft_tokens, 280);
    assert_eq!(cfg.pooling_kernel_size, 3);
    assert!(!cfg.do_normalize);
}
