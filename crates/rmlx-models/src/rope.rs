//! YARN (Yet Another RoPE extensioN) — runtime RoPE frequency rescaling.
//!
//! Reference: <https://arxiv.org/abs/2309.00071> (Peng et al., "YaRN: Efficient
//! Context Window Extension of Large Language Models").
//!
//! YARN is a *runtime config change*: it re-scales the RoPE inverse-frequency
//! table so a model trained at `original_max_position_embeddings` (e.g. 16384)
//! can attend over `factor × original_max_position_embeddings` tokens (e.g.
//! 65536 at `factor = 4`) without re-training. The frequency mixing follows
//! "NTK-by-parts": low-frequency dims (long wavelengths) are scaled down by
//! `1/factor` (positional interpolation), high-frequency dims (short
//! wavelengths) are left untouched (extrapolation), and the band in between is
//! linearly ramped. An attention-score correction `mscale = 0.1 * ln(factor) +
//! 1.0` keeps the softmax dynamic range stable across the longer context.
//!
//! Hard rule 4 (CLAUDE.md): this is a frequency table mutation, NOT training.
//! No gradient computation, no weight modification. Pure CPU precomputation
//! of the `[head_dim / 2]` `freqs` array, fed to `mlx_fast_rope` via
//! [`rmlx_mlx::rope_with_freqs`].
//!
//! # History
//!
//! The numeric port of `mlx_lm.models.rope_utils.YarnRoPE` was first written
//! for the Qwen3.6 DFlash drafter (a private `compute_yarn_freqs` in
//! `speculative::dflash`), pinned against the Python reference at
//! `(head_dim=128, theta=1e7, factor=64, original=4096, beta_fast=32,
//! beta_slow=1)` via `dflash::yarn_freq_check`. The routine was lifted
//! here so the dense Qwen3 path (Bonsai) can consume the same numerics.
//! DFlash was migrated to delegate to [`compute_yarn_freqs`] /
//! [`compute_yarn_mscale`], making this a true shared abstraction (2+ callers).
//! `dflash::yarn_freq_check` retains the numeric pin via the shared helper.

use rmlx_core::error::Result;
use rmlx_mlx::Array;

/// YARN configuration parsed from `config.json` `rope_scaling` or synthesised
/// from a CLI `--max-context` override.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct YarnConfig {
    /// Context-extension factor. `new_max_pos = factor *
    /// original_max_position_embeddings`. Bonsai ships `factor = 4.0`
    /// (16k -> 64k).
    pub factor: f32,
    /// Training-time max-position; YARN mixes the un-scaled and scaled
    /// frequencies around this anchor.
    pub original_max_position_embeddings: f32,
    /// High-frequency cutoff (default 32.0 per the paper). Dims above this
    /// number of rotations across `original_max_position_embeddings` are
    /// considered "extrapolation" and left at the original freq.
    pub beta_fast: f32,
    /// Low-frequency cutoff (default 1.0 per the paper). Dims below this
    /// number of rotations are "interpolation" and scaled by `1/factor`.
    pub beta_slow: f32,
}

impl YarnConfig {
    /// Defaults per YARN paper §3.2: `beta_fast=32`, `beta_slow=1`.
    #[must_use]
    pub fn new(factor: f32, original_max_position_embeddings: f32) -> Self {
        Self {
            factor,
            original_max_position_embeddings,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// Parse `rope_scaling` from a config.json extras blob. Returns `None`
    /// unless `rope_type == "yarn"` (or `type == "yarn"` for HF legacy shape).
    ///
    /// Fields: `factor` (required), `original_max_position_embeddings`
    /// (required), `beta_fast` (default 32), `beta_slow` (default 1).
    #[must_use]
    pub fn from_extras(rope_scaling: &serde_json::Value) -> Option<Self> {
        let rope_type = rope_scaling
            .get("rope_type")
            .or_else(|| rope_scaling.get("type"))
            .and_then(serde_json::Value::as_str)?;
        if rope_type != "yarn" {
            return None;
        }
        let f = |k: &str| {
            rope_scaling
                .get(k)
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32)
        };
        // Warn on missing required fields so a malformed config.json is
        // visible in the trace rather than silently disabling YARN.
        let Some(factor) = f("factor") else {
            tracing::warn!("rope.YarnConfig: rope_type=yarn but 'factor' is missing or not a number; YARN disabled");
            return None;
        };
        let Some(original_max_position_embeddings) = f("original_max_position_embeddings") else {
            tracing::warn!("rope.YarnConfig: rope_type=yarn but 'original_max_position_embeddings' is missing or not a number; YARN disabled");
            return None;
        };
        Some(Self {
            factor,
            original_max_position_embeddings,
            beta_fast: f("beta_fast").unwrap_or(32.0),
            beta_slow: f("beta_slow").unwrap_or(1.0),
        })
    }
}

/// YARN attention-score multiplier `mscale = 0.1 * ln(factor) + 1.0`.
///
/// Returned as `1.0` when `factor <= 1.0`. This covers both the identity
/// case (`factor == 1.0`, no extension) and compression (`factor < 1.0`,
/// which would give a value < 1.0 from the formula — treated as identity here
/// to match mlx_lm behaviour; compression is not a YARN use-case in practice).
/// The paper applies this to query/key tensors BEFORE the RoPE rotation;
/// mathematically equivalent to multiplying the SDPA scale by `mscale²`. We
/// follow the mlx_lm convention (multiply `q` and `k` by `mscale`) so the
/// SDPA `1 / sqrt(head_dim)` scale stays untouched and existing fused kernels
/// keep working.
#[must_use]
pub fn compute_yarn_mscale(factor: f32) -> f32 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1_f32.mul_add(factor.ln(), 1.0)
    }
}

/// Precompute the YARN inverse-frequency table for `mlx_fast_rope(freqs=…)`.
///
/// Pure CPU port of `mlx_lm.models.rope_utils.YarnRoPE.__init__`. Returns the
/// `[head_dim / 2]` `_freqs` array (`Dtype::F32`) and the attention `mscale`
/// (applied to `q`/`k` before rotation; see [`compute_yarn_mscale`]).
///
/// Arguments:
/// - `head_dim`: full attention head dim (e.g. 128 for Bonsai/Qwen3).
/// - `base`: `rope_theta` from config.json (e.g. 1e6 for Qwen3, 1e7 for
///   Qwen3.6 DFlash).
/// - `cfg`: [`YarnConfig`] parsed from config.json or synthesised from CLI.
///
/// # Errors
/// Propagates `Array::from_bytes` failures.
pub fn compute_yarn_freqs(head_dim: usize, base: f32, cfg: YarnConfig) -> Result<(Array, f32)> {
    let base = f64::from(base);
    let dims_f = head_dim as f64;
    let factor = f64::from(cfg.factor);
    let original = f64::from(cfg.original_max_position_embeddings);
    let beta_fast = f64::from(cfg.beta_fast);
    let beta_slow = f64::from(cfg.beta_slow);

    // find_correction_dim(num_rotations) — paper Eq. (15).
    let find_correction_dim = |num_rotations: f64| -> f64 {
        (dims_f * (original / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    let low = find_correction_dim(beta_fast).floor().max(0.0);
    let high = find_correction_dim(beta_slow).ceil().min(dims_f - 1.0);

    let half = head_dim / 2;
    let mut freqs = vec![0f32; half];
    let ramp = |i: f64| -> f64 {
        let denom = if (high - low).abs() < 1e-9 {
            0.001
        } else {
            high - low
        };
        ((i - low) / denom).clamp(0.0, 1.0)
    };
    for (i, f) in freqs.iter_mut().enumerate() {
        // freq_extra = base^( (2i)/dims ); freq_inter = factor * freq_extra
        let freq_extra = base.powf((2 * i) as f64 / dims_f);
        let freq_inter = factor * freq_extra;
        let freq_mask = 1.0 - ramp(i as f64);
        // NTK-by-parts blend, paper Eq. (18).
        let val =
            (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask));
        *f = val as f32;
    }
    let arr = Array::from_f32_slice(&freqs, &[half as i32])?;
    Ok((arr, compute_yarn_mscale(cfg.factor)))
}

#[cfg(test)]
#[path = "rope_tests.rs"]
mod rope_tests;
