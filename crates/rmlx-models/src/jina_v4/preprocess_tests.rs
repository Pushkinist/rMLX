use super::*;

// ----- smart_resize: hand-computed vs the HF algorithm -----------------
//
// Each case derived by running the HF smart_resize formula by hand with
// factor=28, min_pixels=3136, max_pixels=602112 (the jina-v4
// preprocessor_config values). Asserts exact integer output.

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn smart_resize_in_range_rounds_to_factor() {
    // 100x200: round(100/28)=round(3.571)=4 → 112; round(200/28)=
    // round(7.142)=7 → 196. 112*196=21952 ∈ [3136,602112] → unchanged.
    let (h, w) = smart_resize(100, 200, 28, 3136, 602_112).unwrap();
    assert_eq!((h, w), (112, 196));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn smart_resize_scales_down_when_too_large() {
    // 4000x3000: h_bar=round(142.857)*28=143*28=4004,
    // w_bar=round(107.142)*28=107*28=2996; 4004*2996=11,995,984 >
    // 602112. beta=sqrt(4000*3000/602112)=4.464285714285714.
    // h/beta/factor = 4000/4.46428.../28 = 32.0 → floor 32 → 32*28=896
    // w/beta/factor = 3000/4.46428.../28 = 24.0 → floor 24 → 24*28=672
    // (cross-checked against the Python HF smart_resize.)
    let (h, w) = smart_resize(4000, 3000, 28, 3136, 602_112).unwrap();
    assert_eq!((h, w), (896, 672));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn smart_resize_scales_up_when_too_small() {
    // 20x20: h_bar=round(20/28)*28=round(0.714)*28=1*28=28,
    // w_bar=28; 28*28=784 < 3136. beta=sqrt(3136/(20*20))
    // =sqrt(7.84)=2.8.
    // h_bar=ceil(20*2.8/28)*28=ceil(2.0)*28=2*28=56, w_bar=56.
    // 56*56=3136 == min_pixels (the >= boundary holds).
    let (h, w) = smart_resize(20, 20, 28, 3136, 602_112).unwrap();
    assert_eq!((h, w), (56, 56));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn smart_resize_clamps_to_factor_floor() {
    // Degenerate strip: height=28 width=4000 (aspect 4000/28=142.8
    // < 200, allowed). h_bar=round(28/28)*28=28,
    // w_bar=round(4000/28)*28=round(142.857)*28=143*28=4004;
    // 28*4004=112112 ∈ [3136,602112] → unchanged. (cross-checked
    // against the Python HF smart_resize.)
    let (h, w) = smart_resize(28, 4000, 28, 3136, 602_112).unwrap();
    assert_eq!((h, w), (28, 4004));
}

#[test]
fn smart_resize_aspect_guard() {
    // aspect 5000/20 = 250 > 200 → error (matches HF ValueError).
    assert!(smart_resize(5000, 20, 28, 3136, 602_112).is_err());
}

#[test]
fn round_half_even_matches_python() {
    // Python 3 round(): half-to-even.
    assert_eq!(round_half_even(0.5), 0.0);
    assert_eq!(round_half_even(1.5), 2.0);
    assert_eq!(round_half_even(2.5), 2.0);
    assert_eq!(round_half_even(3.5), 4.0);
    assert_eq!(round_half_even(0.4999), 0.0);
    assert_eq!(round_half_even(0.5001), 1.0);
    assert_eq!(round_half_even(7.0), 7.0);
}

// ----- config parsing --------------------------------------------------

#[test]
fn config_defaults_match_preprocessor_values() {
    let c = ImagePreprocessConfig::default();
    assert_eq!(c.patch_size, 14);
    assert_eq!(c.merge_size, 2);
    assert_eq!(c.temporal_patch_size, 2);
    assert_eq!(c.min_pixels, 3136);
    assert_eq!(c.max_pixels, 602_112);
    assert_eq!(c.factor(), 28);
    assert!((c.rescale_factor - 1.0 / 255.0).abs() < 1e-9);
    assert!((c.image_mean[0] - 0.481_454_5).abs() < 1e-5);
    assert!((c.image_std[2] - 0.275_777_1).abs() < 1e-5);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn config_parses_size_edges_over_flat_keys() {
    // HF precedence: size.shortest_edge / longest_edge win.
    let v: serde_json::Value = serde_json::json!({
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "min_pixels": 9999,
        "max_pixels": 8888,
        "size": { "shortest_edge": 3136, "longest_edge": 602112 },
        "rescale_factor": 0.00392156862745098,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std":  [0.26862954, 0.26130258, 0.27577711]
    });
    let c = ImagePreprocessConfig::from_preprocessor_json(&v).unwrap();
    assert_eq!(c.min_pixels, 3136, "size.shortest_edge must win");
    assert_eq!(c.max_pixels, 602_112, "size.longest_edge must win");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn config_parses_real_preprocessor_json_if_present() {
    let Some(dir_buf) = std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from)
    else {
        eprintln!("SKIP: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let dir = dir_buf.as_path();
    if !dir.join("preprocessor_config.json").exists() {
        eprintln!("SKIP: jina-v4 model not present; skipping live parse");
        return;
    }
    let c = ImagePreprocessConfig::from_model_dir(dir).unwrap();
    assert_eq!(c.patch_size, 14);
    assert_eq!(c.merge_size, 2);
    assert_eq!(c.temporal_patch_size, 2);
    assert_eq!(c.min_pixels, 3136);
    assert_eq!(c.max_pixels, 602_112);
    assert!((c.image_mean[0] - 0.481_454_5).abs() < 1e-5);
    assert!((c.image_std[1] - 0.261_302_6).abs() < 1e-5);
}

// ----- end-to-end structural ------------------------------------------

/// Synthesize a small RGB gradient PNG entirely in-test (no external
/// asset), run the full pipeline, and assert structural invariants.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn preprocess_gradient_png_structural() {
    let cfg = ImagePreprocessConfig::default();

    // 96x64 gradient (h=64, w=96). smart_resize(64,96,28,..):
    // round(64/28)=round(2.285)=2 → 56; round(96/28)=round(3.428)=3
    // → 84. 56*84=4704 ∈ [3136,602112] → resized 56x84.
    let (iw, ih) = (96u32, 64u32);
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
    let mut png: Vec<u8> = Vec::new();
    {
        use image::ImageEncoder;
        let enc = image::codecs::png::PngEncoder::new(&mut png);
        enc.write_image(img.as_raw(), iw, ih, image::ExtendedColorType::Rgb8)
            .expect("encode test PNG");
    }

    let pv = preprocess_image_bytes(&png, &cfg).expect("preprocess");

    // Grid: 56/14=4, 84/14=6, t=1. factor-consistent: 56%28==0,
    // 84%28==0; grid divisible by merge_size.
    assert_eq!(pv.grid.t, 1);
    assert_eq!(pv.grid.h, 4);
    assert_eq!(pv.grid.w, 6);
    assert_eq!(56 % cfg.factor(), 0);
    assert_eq!(84 % cfg.factor(), 0);
    assert_eq!(pv.grid.h % cfg.merge_size, 0);
    assert_eq!(pv.grid.w % cfg.merge_size, 0);

    // feature_len = 3 * tps(2) * 14 * 14 = 1176.
    let expect_feat = 3 * cfg.temporal_patch_size * 14 * 14;
    assert_eq!(pv.feature_len, expect_feat);
    let expect_np = pv.grid.t * pv.grid.h * pv.grid.w;
    assert_eq!(pv.num_patches, expect_np);
    assert_eq!(pv.grid.num_patches(), expect_np);

    // Exact total length == grid_t*grid_h*grid_w * 3*tps*14*14.
    assert_eq!(pv.data.len(), expect_np * expect_feat);

    // All finite, in a sane post-normalize band. Worst case
    // (0 or 255)→ (0-mean)/std .. (1-mean)/std ⇒ roughly [-2,3].
    for &x in &pv.data {
        assert!(x.is_finite(), "non-finite pixel value");
        assert!(
            (-5.0..=5.0).contains(&x),
            "pixel value {x} outside sane post-normalize range"
        );
    }

    // Temporal duplication: the two frames must be identical for a
    // still image. For patch row 0, channel 0: feature offset of
    // frame 0 (fr=0) vs frame 1 (fr=1) for the same (ph,pw)=(0,0)
    // is 0 vs (ps*ps)=196.
    let ps = cfg.patch_size;
    assert_eq!(
        pv.data[0],
        pv.data[ps * ps],
        "temporal frames must be identical for a still image"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn preprocess_solid_color_normalized_constant() {
    let cfg = ImagePreprocessConfig::default();
    // Solid mid-gray 120x120 → after resize still constant per channel,
    // so every pixel of channel c == (120/255 - mean[c]) / std[c].
    let (iw, ih) = (120u32, 120u32);
    let img = image::RgbImage::from_pixel(iw, ih, image::Rgb([128, 128, 128]));
    let mut png: Vec<u8> = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), iw, ih, image::ExtendedColorType::Rgb8)
            .expect("encode");
    }
    let pv = preprocess_image_bytes(&png, &cfg).expect("preprocess");

    // Expected per-channel constant.
    let v = 128.0_f32 * cfg.rescale_factor;
    let exp = [
        (v - cfg.image_mean[0]) / cfg.image_std[0],
        (v - cfg.image_mean[1]) / cfg.image_std[1],
        (v - cfg.image_mean[2]) / cfg.image_std[2],
    ];
    // Channel 0 occupies feature offsets [0, tps*ps*ps); spot-check the
    // first element of each channel block in patch row 0.
    let ps = cfg.patch_size;
    let tps = cfg.temporal_patch_size;
    for (c, &expected) in exp.iter().enumerate() {
        let off = c * tps * ps * ps;
        assert!(
            (pv.data[off] - expected).abs() < 2e-3,
            "channel {c}: got {} expected {expected}",
            pv.data[off],
        );
    }
}

#[test]
fn pil_bicubic_kernel_is_keys_a_minus_half() {
    // Keys cubic, a = -0.5: f(0)=1, f(1)=0, f(0.5)=0.5625, f(1.5)=-0.0625,
    // f(>=2)=0. (Shared by PIL BICUBIC and torchvision AA bicubic.)
    assert!((pil_bicubic(0.0) - 1.0).abs() < 1e-12);
    assert!(pil_bicubic(1.0).abs() < 1e-12);
    assert!((pil_bicubic(0.5) - 0.5625).abs() < 1e-12);
    assert!((pil_bicubic(1.5) - (-0.0625)).abs() < 1e-12);
    assert!(pil_bicubic(2.0).abs() < 1e-12);
    assert!(pil_bicubic(3.7).abs() < 1e-12);
    // Even kernel.
    assert!((pil_bicubic(0.3) - pil_bicubic(-0.3)).abs() < 1e-12);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn precompute_coeffs_matches_torchvision_aa() {
    // Ground truth captured via impulse-probing
    // `torchvision.transforms.v2.functional.resize(..., BICUBIC,
    // antialias=True)` for a 64->56 downsample. The window/normalized
    // weights must match aten exactly (the fixed-point precision split
    // PIL=22 vs aten=14 is covered by PRECISION_BITS / the e2e parity).
    let (b, w) = precompute_coeffs(64, 56);
    assert_eq!(b.len(), 56);
    // out 10: symmetric 4-tap, window [10,14).
    assert_eq!(b[10], (10, 4));
    let w10 = &w[10];
    let exp10 = [-0.064417_f64, 0.564417, 0.564417, -0.064417];
    for (got, e) in w10.iter().zip(exp10) {
        assert!((got - e).abs() < 1e-5, "o=10 weight {got} != {e}");
    }
    // out 0: edge window [0,3), 3 taps; aten = [0.89146,0.13875,-0.03021].
    assert_eq!(b[0], (0, 3));
    let exp0 = [0.891_46_f64, 0.138_75, -0.030_21];
    for (got, e) in w[0].iter().zip(exp0) {
        assert!((got - e).abs() < 1e-4, "o=0 weight {got} != {e}");
    }
    // every output row's weights sum to 1 (normalized).
    for (i, row) in w.iter().enumerate() {
        let s: f64 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "row {i} sum {s} != 1");
    }
}

#[test]
fn precision_bits_is_torchvision_uint8() {
    // jina-v4 uses the fast (torchvision) processor; its uint8 resampler
    // precision is 14 (verified bit-exact e2e). NOT PIL's 22.
    assert_eq!(PRECISION_BITS, 14);
}
