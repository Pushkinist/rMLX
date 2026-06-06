//! YARN math + config parse tests.
//!
//! The numeric reference values come from `mlx_lm.models.rope_utils.YarnRoPE`
//! (`speculative::dflash::yarn_freq_check` pins the same numbers for the
//! Qwen3.6 DFlash drafter — `head_dim=128, theta=1e7, factor=64, original=4096`).
//! Adding a separate Bonsai-shaped test here documents the dense-Qwen3 cell
//! (`head_dim=128, theta=1e6, factor=4, original=16384`).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    reason = "YARN math tests pin numeric reference values; .expect() messages document invariants and slice indices are bounded by literal head_dim/2."
)]

use super::{compute_yarn_freqs, compute_yarn_mscale, YarnConfig};

#[test]
fn mscale_default_factor_2() {
    // 0.1 * ln(2) + 1.0 = 1.0693147…
    let m = compute_yarn_mscale(2.0);
    assert!((m - 1.069_314_7).abs() < 1e-5, "mscale={m}");
}

#[test]
fn mscale_factor_4_bonsai() {
    // 0.1 * ln(4) + 1.0 = 1.1386294…
    let m = compute_yarn_mscale(4.0);
    assert!((m - 1.138_629_4).abs() < 1e-5, "mscale={m}");
}

#[test]
fn mscale_no_extension_returns_one() {
    assert!((compute_yarn_mscale(1.0) - 1.0).abs() < 1e-7);
    assert!((compute_yarn_mscale(0.5) - 1.0).abs() < 1e-7);
}

#[test]
fn yarn_freqs_dflash_reference_pin() {
    // Same pin as `speculative::dflash::yarn_freq_check` to prove the
    // shared helper preserves the numerics of the original DFlash port.
    let cfg = YarnConfig::new(64.0, 4096.0);
    let (freqs, mscale) = compute_yarn_freqs(128, 1.0e7, cfg).expect("compute");
    freqs.eval().expect("graph evaluate");
    let b = freqs.to_bytes().expect("to_bytes");
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    assert!((mscale - 1.415_888_3).abs() < 1e-4, "mscale {mscale}");
    assert!((v[0] - 1.0).abs() < 1e-3);
    assert!((v[1] - 1.286_397).abs() < 1e-2);
    assert!((v[63] - 4.975_136_3e8).abs() / 4.975_136_3e8 < 1e-3);
}

#[test]
fn yarn_freqs_bonsai_shape() {
    // Bonsai-shaped cell: factor=4, original=16384, theta=1e6, hd=128.
    let cfg = YarnConfig::new(4.0, 16384.0);
    let (freqs, mscale) = compute_yarn_freqs(128, 1.0e6, cfg).expect("compute");
    freqs.eval().expect("graph evaluate");
    let b = freqs.to_bytes().expect("to_bytes");
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    assert_eq!(v.len(), 64);
    // mscale = 0.1 * ln(4) + 1.0 = 1.1386294…
    assert!((mscale - 1.138_629_4).abs() < 1e-5, "mscale={mscale}");
    // First freq (highest-freq dim, fully extrapolation): stays at 1.0
    // (= theta^0 = wavelength of dim 0).
    assert!((v[0] - 1.0).abs() < 1e-3, "v[0]={}", v[0]);
    // The YARN `_freqs` table is wavelength-domain, not frequency-domain
    // (mlx_lm convention): MLX's `mlx_fast_rope(freqs=...)` divides position
    // by `freqs[i]` rather than multiplying. For dims fully inside the
    // interpolation band (mask=0), `freqs[i] = freq_inter = factor * freq_extra`
    // (= `factor * theta^(2i/d)`). For dims fully outside (mask=1),
    // `freqs[i] = freq_extra` (the un-scaled wavelength).
    //
    // For Bonsai (theta=1e6, dims=128, original=16384, factor=4) the
    // correction band is roughly [low=20, high=37]; i=63 is well past
    // `high`, so mask=0 and:
    //   freqs[63] = factor * theta^(126/128) = 4 * 1e6^0.984375 ≈ 3.22e6.
    let expected_last_unscaled: f64 = 1.0e6_f64.powf(126.0 / 128.0);
    let expected_last = 4.0 * expected_last_unscaled;
    let got = f64::from(v[63]);
    assert!(
        (got - expected_last).abs() / expected_last < 1e-3,
        "last freq {got} vs expected {expected_last}"
    );
}

#[test]
fn yarn_config_parses_bonsai_rope_scaling() {
    let json = serde_json::json!({
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 16384,
    });
    let cfg = YarnConfig::from_extras(&json).expect("parse YarnConfig");
    assert!((cfg.factor - 4.0).abs() < 1e-6);
    assert!((cfg.original_max_position_embeddings - 16384.0).abs() < 1e-3);
    assert!((cfg.beta_fast - 32.0).abs() < 1e-6);
    assert!((cfg.beta_slow - 1.0).abs() < 1e-6);
}

#[test]
fn yarn_config_parses_legacy_type_field() {
    // Some HF configs use `"type": "yarn"` instead of `"rope_type"`.
    let json = serde_json::json!({
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 16.0,
        "beta_slow": 2.0,
    });
    let cfg = YarnConfig::from_extras(&json).expect("parse legacy");
    assert!((cfg.factor - 2.0).abs() < 1e-6);
    assert!((cfg.beta_fast - 16.0).abs() < 1e-6);
    assert!((cfg.beta_slow - 2.0).abs() < 1e-6);
}

#[test]
fn yarn_config_rejects_non_yarn() {
    let json = serde_json::json!({
        "rope_type": "linear",
        "factor": 2.0,
    });
    assert!(YarnConfig::from_extras(&json).is_none());
}

#[test]
fn yarn_config_rejects_missing_factor() {
    let json = serde_json::json!({
        "rope_type": "yarn",
        "original_max_position_embeddings": 4096,
    });
    assert!(YarnConfig::from_extras(&json).is_none());
}
