//! YARN frequency numeric-alignment tests.

use crate::rope::{compute_yarn_freqs, YarnConfig};

/// Numeric-alignment pin: the Rust YARN inverse-freq table + mscale
/// must match `mlx_lm.models.rope_utils.YarnRoPE` for the Qwen3.6 DFlash
/// drafter (head_dim=128, theta=1e7, factor=64, original=4096, beta 32/1).
/// Reference values dumped from mlx-lm: mscale=1.4158883, freqs[0]=1.0,
/// freqs[1]=1.286397, freqs[63]=4.9751363e8. Applying plain RoPE instead
/// (these wrong) collapses the drafter accept-rate to ~0.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn yarn_freqs_match_reference() {
    let cfg = YarnConfig {
        factor: 64.0,
        original_max_position_embeddings: 4096.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let (freqs, mscale) = compute_yarn_freqs(128, 1.0e7, cfg).unwrap();
    freqs.eval().unwrap();
    let b = freqs.to_bytes().unwrap();
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((mscale - 1.4158883).abs() < 1e-4, "mscale {mscale}");
    assert!((v[0] - 1.0).abs() < 1e-3);
    assert!((v[1] - 1.286397).abs() < 1e-2);
    assert!((v[63] - 4.9751363e8).abs() / 4.9751363e8 < 1e-3);
}
