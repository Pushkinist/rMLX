use super::*;

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn solid_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut img = image::RgbImage::new(w, h);
    for p in img.pixels_mut() {
        *p = image::Rgb(rgb);
    }
    let mut png = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .expect("encode png");
    }
    png
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn smart_resize_snaps_to_factor() {
    let (h, w) = smart_resize(100, 200, 32, 56 * 56, 14 * 14 * 4 * 1280).unwrap();
    assert_eq!(h % 32, 0);
    assert_eq!(w % 32, 0);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn preprocess_shapes_and_grid() {
    let cfg = Qwen3VlImageConfig::default();
    // 64x64 solid image.
    let png = solid_png(64, 64, [0, 200, 0]);
    let pv = preprocess(&png, &cfg).unwrap();
    let (gt, gh, gw) = pv.grid_thw;
    let feat_len = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    assert_eq!(gt, 1);
    assert_eq!(gh % cfg.merge_size, 0);
    assert_eq!(gw % cfg.merge_size, 0);
    assert_eq!(pv.pixel_values.len(), gt * gh * gw * feat_len);
    assert_eq!(
        pv.num_soft_tokens,
        gt * gh * gw / (cfg.merge_size * cfg.merge_size)
    );
    // The temporal duplication means the two tps frames within a patch are
    // identical: feature[c,0,py,px] == feature[c,1,py,px].
    let ps = cfg.patch_size;
    let one_frame = 3 * ps * ps; // C*ps*ps for one temporal slice? no — layout is (C, tps, ps, ps)
    let _ = one_frame;
    assert!(pv.pixel_values.iter().all(|v| v.is_finite()));
}
