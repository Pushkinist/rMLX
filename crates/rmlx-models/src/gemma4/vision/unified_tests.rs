//! Model-free unit coverage for the gemma4_unified vision embedder front-end:
//! config parse, soft-token math, and the patchify+merge shape/position plumbing.

use std::path::Path;

use super::*;

fn cfg_12b() -> UnifiedVisionConfig {
    // Verified gemma-4-12B `gemma4_unified_vision` values.
    UnifiedVisionConfig {
        mm_embed_dim: 3840,
        mm_posemb_size: 1120,
        model_patch_size: 48,
        patch_size: 16,
        pooling_kernel_size: 3,
        num_soft_tokens: 280,
        output_proj_dims: 3840,
        rms_norm_eps: 1e-6,
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts JSON parse succeeds on a known-good literal"
)]
fn config_parse_matches_snapshot() {
    let v: serde_json::Value = serde_json::from_str(
        r#"{
          "mm_embed_dim": 3840, "mm_posemb_size": 1120, "model_patch_size": 48,
          "patch_size": 16, "pooling_kernel_size": 3, "num_soft_tokens": 280,
          "output_proj_dims": 3840, "rms_norm_eps": 1e-06
        }"#,
    )
    .unwrap();
    let cfg = UnifiedVisionConfig::from_json(&v);
    assert_eq!(cfg.mm_embed_dim, 3840);
    assert_eq!(cfg.mm_posemb_size, 1120);
    assert_eq!(cfg.model_patch_size, 48);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.pooling_kernel_size, 3);
    assert_eq!(cfg.num_soft_tokens, 280);
    assert_eq!(cfg.output_proj_dims, 3840);
    // patch_dim = 48*48*3 = 6912 (input width of patch_dense / patch_ln1).
    assert_eq!(cfg.patch_dim(), 6912);
}

#[test]
fn patch_dim_is_model_patch_squared_times_three() {
    let cfg = cfg_12b();
    assert_eq!(cfg.patch_dim(), 48 * 48 * 3);
}

#[test]
fn soft_token_count_is_model_patch_grid() {
    let cfg = cfg_12b();
    // A 144x96 image: 3x2 model patches (model_patch_size=48) -> 6 soft tokens.
    assert_eq!(unified_num_soft_tokens(144, 96, &cfg), 6);
    // Square 672x672 -> 14x14 = 196 model patches.
    assert_eq!(unified_num_soft_tokens(672, 672, &cfg), 196);
    // 768x720 -> 16x15 = 240.
    assert_eq!(unified_num_soft_tokens(768, 720, &cfg), 240);
}

#[test]
fn image_processor_config_carries_unified_params() {
    let cfg = cfg_12b();
    let pc = unified_image_processor_config(&cfg);
    assert_eq!(pc.patch_size, 16);
    assert_eq!(pc.max_soft_tokens, 280);
    assert_eq!(pc.pooling_kernel_size, 3);
}

/// Validate that the public soft-token count and the documented model-patch
/// grid agree for a non-square image, and that all position ids fall inside the
/// factorized positional-embedding table (`mm_posemb_size`).
#[test]
fn model_patch_grid_and_position_bounds() {
    let cfg = cfg_12b();
    // 96x144 image (h=96, w=144): model patches = 2 rows x 3 cols = 6.
    let h = 96usize;
    let w = 144usize;
    let m_h = h / cfg.model_patch_size; // 2
    let m_w = w / cfg.model_patch_size; // 3
    assert_eq!(m_h * m_w, unified_num_soft_tokens(h, w, &cfg));
    assert_eq!(unified_num_soft_tokens(h, w, &cfg), 6);

    // Position ids span (mx, my) ∈ [0, m_w) × [0, m_h) — all in-table.
    assert!(m_w <= cfg.mm_posemb_size);
    assert!(m_h <= cfg.mm_posemb_size);
}

#[test]
fn is_unified_arch_false_for_missing_dir() {
    // Non-existent dir -> false (no panic).
    let p = Path::new("/nonexistent/gemma4-unified-test-dir");
    assert!(!is_unified_arch(p));
}

/// Build a solid-colour CHW `[1, 3, H, W]` pixel buffer (the shared
/// preprocessor output: rescaled `[0,1]`, channels-first, RGB).
#[allow(
    clippy::indexing_slicing,
    reason = "buffer sized 3*n at allocation; c*n+i bounded by c<3, i<n"
)]
fn solid_chw(h: usize, w: usize, rgb: [f32; 3]) -> Gemma4PixelValues {
    let n = h * w;
    let mut pixel_values = vec![0.0_f32; 3 * n];
    for (c, &v) in rgb.iter().enumerate() {
        for i in 0..n {
            pixel_values[c * n + i] = v;
        }
    }
    Gemma4PixelValues {
        pixel_values,
        height: h,
        width: w,
        num_soft_tokens: 0,
    }
}

/// Model-free numerical guard for the unified patchify front-end — the test that
/// would have caught a channel-order / value-scaling defect (#127's first
/// suspect). For a solid RGB input every merged-patch slot must equal the source
/// channel value exactly, and the three channels must remain in RGB order
/// (interior index `% 3` selects R/G/B). A pure-green input must therefore yield
/// nonzero values only in the green-derived (`off % 3 == 1`) slots.
#[test]
#[allow(
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::expect_used,
    reason = "fixed-size solid-colour buffer indexed by construction; rescaled pixel values are exact 0.0/1.0"
)]
fn patchify_preserves_channel_values_and_order() {
    let cfg = cfg_12b();
    // One 48x48 model patch (k*p = 48): smallest valid image.
    let (h, w) = (48usize, 48usize);

    for (name, rgb) in [
        ("red", [1.0, 0.0, 0.0]),
        ("green", [0.0, 1.0, 0.0]),
        ("blue", [0.0, 0.0, 1.0]),
        ("white", [1.0, 1.0, 1.0]),
        ("yellow", [1.0, 1.0, 0.0]),
    ] {
        let pv = solid_chw(h, w, rgb);
        let (merged, x_idx, y_idx) =
            patchify_and_merge_impl(&cfg, &pv).expect("patchify must succeed");
        assert_eq!(x_idx, vec![0]);
        assert_eq!(y_idx, vec![0]);
        let patch_dim = cfg.patch_dim();
        assert_eq!(merged.len(), patch_dim);

        // Every interior slot equals the source channel value (no scaling,
        // inversion, or channel swap); channel = interior index % 3.
        for (off, &val) in merged.iter().enumerate() {
            let ch = off % 3;
            assert!(
                (val - rgb[ch]).abs() < 1e-6,
                "{name}: slot {off} (ch {ch}) = {val}, expected {}",
                rgb[ch]
            );
        }
        // Channel-order sanity: a channel that is 0 in the source must be 0 in
        // every derived slot (e.g. pure green leaves R- and B-slots at 0).
        for (ch, &src_val) in rgb.iter().enumerate() {
            if src_val == 0.0 {
                let any_nonzero = merged.iter().skip(ch).step_by(3).any(|&v| v != 0.0);
                assert!(
                    !any_nonzero,
                    "{name}: channel {ch} is 0 in source but nonzero after patchify"
                );
            }
        }
    }
}

/// Channel routing within a single 48x48 patch must be spatially faithful: a
/// half-red / half-blue image (left columns red, right columns blue) lands red
/// in the left half of the contiguous model-patch image and blue in the right
/// half — proving the `[ky, ry, kx, rx, ch]` interior layout reconstructs the
/// original pixel grid (not a scrambled 3×3 tiling).
#[test]
#[allow(
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::expect_used,
    reason = "fixed-size half-red/half-blue buffer indexed by construction; pixel values are exact 0.0/1.0"
)]
fn patchify_interior_layout_is_contiguous_image() {
    let cfg = cfg_12b();
    let (h, w) = (48usize, 48usize);
    let n = h * w;
    let mut pixel_values = vec![0.0_f32; 3 * n];
    // CHW: R channel = 1 on left half, B channel = 1 on right half.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x < w / 2 {
                pixel_values[i] = 1.0; // R
            } else {
                pixel_values[2 * n + i] = 1.0; // B
            }
        }
    }
    let pv = Gemma4PixelValues {
        pixel_values,
        height: h,
        width: w,
        num_soft_tokens: 0,
    };
    let (merged, _, _) = patchify_and_merge_impl(&cfg, &pv).expect("patchify must succeed");

    // The merged 6912-vector is a contiguous 48x48x3 image: index
    // (row*48 + col)*3 + ch. Left columns must be red, right columns blue.
    let side = cfg.model_patch_size; // 48
    for row in 0..side {
        for col in 0..side {
            let base = (row * side + col) * 3;
            let (r, g, b) = (merged[base], merged[base + 1], merged[base + 2]);
            assert_eq!(g, 0.0, "no green expected at ({row},{col})");
            if col < side / 2 {
                assert_eq!((r, b), (1.0, 0.0), "left half must be red at ({row},{col})");
            } else {
                assert_eq!(
                    (r, b),
                    (0.0, 1.0),
                    "right half must be blue at ({row},{col})"
                );
            }
        }
    }
}
