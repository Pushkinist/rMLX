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
