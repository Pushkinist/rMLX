// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes (seed state)
#![allow(unsafe_code)]

//! Sampler module — A6.2 masked argmax + A7.2/A7.3 host categorical sampler.
//!
//! # Final sampling pipeline (A7.4)
//!
//! The per-arch decode loops select the next token through one of two paths:
//!
//! ## Fast path: pure greedy (no host work)
//!
//! ```text
//! condition: temp == 0 AND !penalties_active() AND logit_bias.is_empty()
//! AND no constraint mask (response_format == Text)
//! action: GPU argmax(&logits_flat, -1, device) — untouched, byte-identical
//! ```
//!
//! When only a constraint mask is present (A6, no temperature):
//! `apply_mask_argmax` keeps the argmax on-GPU with an additive -Inf bias.
//!
//! ## Host path: everything else
//!
//! One GPU→host transfer per step, then in order:
//!
//! ```text
//! 1. constraint mask (A6) — forbidden ids → NEG_INF
//! 2. logit_bias (additive) — logit[id] += bias
//! 3. repetition_penalty — sign-aware multiplicative, last-20 window
//! if logit < 0: logit *= p; else: logit /= p
//! 4. presence_penalty — subtract once per unique id in last-20
//! 5. frequency_penalty — subtract penalty * count in last-20
//! 6. temperature scale + softmax — numerically stable (subtract max before exp)
//! 7. top_p (nucleus) — ascending cumsum, keep where cum > 1-p
//! → min_p — keep where prob >= max_prob * min_p
//! → top_k — keep top-k, zero rest
//! 8. inverse-CDF sample — seeded PCG32; or argmax if temp==0 + penalties
//! ```
//!
//! Steps 2–5 are [`apply_penalties`] (mlx-lm `make_logits_processors` L106-126
//! order). Steps 7 are [`filter_top_p`] → [`filter_min_p`] → [`filter_top_k`]
//! (mlx-lm `sample_utils.py` `make_sampler` L46-69 order).
//!
//! ### mlx-lm parity notes
//!
//! - **Application order** is exact mlx-lm parity: bias → rep → presence → freq,
//!   then top_p → min_p → top_k. This differs from the original A7 task-doc
//!   draft (which listed `penalties → top-k → top-p → min-p`); the code follows
//!   mlx-lm, not the draft.
//! - **Window size** is last-20 generated tokens (`context_size=20` in
//!   mlx-lm), **not** full-context OpenAI semantics. Documented divergence:
//!   long repeated passages beyond the 20-token window are NOT penalised.
//! - **Speculative path**: temperature > 0 is rejected for speculative decoding
//!   (spec candidates must be greedy). The server returns a 400 if a caller
//!   combines `temperature > 0` with speculative parameters.
//!
//! ### GPU masked argmax (greedy + constraint)
//!
//! When `response_format ∈ {json_object, json_schema}` and `temp == 0`:
//!
//! 1. Build a F32 bias buffer on the host: `0.0` for allowed tokens,

//! 2. Wrap as `[1, vocab]` F32 MLX array.
//! 3. `add(logits, bias)` — GPU op; result is F32 regardless of input dtype.
//! 4. `argmax` along axis -1 — produces `[1] I32`.
//!
//! Total overhead vs unconstrained: ~0.05 ms for the host-side bias fill
//! (O(vocab) writes for 262K-token vocab).

#![allow(clippy::float_cmp)]
use rmlx_core::error::{Error, Result};
use rmlx_mlx::{add, argmax, Array, Device, Dtype};

// ===========================================================================
// A7.2 — host-side categorical sampler
// ===========================================================================
//
// `temperature == 0` keeps the existing GPU `argmax` / `apply_mask_argmax`
// greedy path byte-for-byte (the per-arch decode loops branch on
// `sp.temperature <= 0.0` and run the *unchanged* argmax block). Only
// `temperature > 0` enters this module.
//
// On the `temp > 0` path the work is:
//
// 1. GPU→host transfer of the `[1, vocab]` logits (single transfer/step).
// 2. (optional) constraint mask: forbidden ids → -inf, exactly as
// `apply_mask_argmax` does for the greedy branch, so sampling composes
// with A6 grammars.
// 3. Scale logits by `1/temperature`.
// 4. Numerically-stable softmax → probability `Vec<f32>`.
// 5. Pure-Rust filters in mlx-lm `sample_utils.py` order:
// `top_p → min_p → top_k`, then renormalise.
// 6. Inverse-CDF sample with a per-request seeded `Pcg32`.
// 7. Return the chosen id as a `[1] I32` Array — same shape contract as
// `argmax(&logits_flat, -1, device)`, so every downstream call site
// (`to_bytes` → `i32::from_le_bytes`) is unchanged.
//
// mlx-lm parity references (`mlx-lm/mlx_lm/sample_utils.py`):
// - `make_sampler` L46-69 — `temp == 0 ⇒ argmax`; else chain
// `top_p → min_p → (xtc) → top_k → categorical`. XTC is A7.x, skipped.
// - `apply_top_p` L205-237 — ascending sort, **inclusive** cumsum,
// keep where `cumprob > 1 - top_p` (replicated exactly below).
// - `apply_min_p` L155-201 — keep where `prob >= max_prob * min_p`
// (`min_tokens_to_keep` default 1; the max token always survives so the
// floor-of-1 needs no special case).
// - `apply_top_k` L130-151 — keep the `top_k` highest, mask the rest.
// - `categorical_sampling` L278-279 — `mx.random.categorical(logits/temp)`,
// i.e. sample from `softmax(logits/temp)`. We scale before softmax (host)
// which is mathematically identical.

// ===========================================================================
// A7.3 — penalty configuration (non-Copy: contains Vec for logit_bias)
// ===========================================================================
//
// Design: kept separate from `SamplerConfig` so `SamplerConfig` stays `Copy`
// (zero churn on ~30 existing call sites). `PenaltyConfig` is passed
// alongside `SamplerConfig` and `Pcg32` at every arch `generate_greedy`
// boundary; default values are exact no-ops so the common path pays only one
// `penalties_active()` discriminant check.
//
// mlx-lm parity (`mlx_lm/sample_utils.py`):
// - Application order (L106-126): logit_bias → repetition → presence → freq.
// - Window: last 20 generated tokens (`context_size=20`).
// - Composition with A6 mask: mask is applied BEFORE penalties so a
// NEG_INF logit stays NEG_INF; penalising it is harmless but we keep
// mask-first order for semantic clarity.
//
// Hot-path rule:
// temp=0 AND !penalties_active → existing pure-GPU `argmax` / `apply_mask_argmax`
// path untouched, byte-identical (the common case, zero overhead).

/// Per-request logit-penalty configuration (A7.3).
///
/// Passed alongside [`SamplerConfig`] and [`Pcg32`] into every arch
/// `generate_greedy` function. All fields default to no-ops so callers that
/// have no penalties can pass `&PenaltyConfig::default()` with zero cost.
///
/// The struct is **not** `Copy` because `logit_bias` owns a `Vec`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — four logit-penalty fields; adding a penalty requires updating penalties_active(), apply_penalties(), and all arch generate_greedy call sites"
)]
#[derive(Debug, Clone)]
pub struct PenaltyConfig {
    /// Multiplicative sign-aware repetition penalty (`make_repetition_penalty`
    /// in mlx-lm). `1.0` is the exact no-op. For every token id in the
    /// last-20 window: if `logit < 0` → `logit *= penalty`; else → `logit /= penalty`.
    pub rep_penalty: f32,
    /// Additive presence penalty. `0.0` is the no-op. Subtracts `penalty`
    /// once for every token id that appears at least once in the last-20 window.
    pub presence_penalty: f32,
    /// Additive frequency penalty. `0.0` is the no-op. Subtracts
    /// `penalty * count` where `count` is the number of times the token id
    /// appears in the last-20 window.
    pub frequency_penalty: f32,
    /// `(token_id, bias)` pairs applied additively to logits before the
    /// repetition/presence/frequency steps. Out-of-vocab ids are skipped
    /// silently (A7.1 already validated and stored them). Empty = no-op.
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        PenaltyConfig {
            rep_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
        }
    }
}

impl PenaltyConfig {
    /// `true` when at least one penalty is non-trivial.
    ///
    /// The hot path checks this once; all four fields are at their identity
    /// values → returns `false` → the caller skips the GPU→host transfer and
    /// the `apply_penalties` call entirely.
    #[inline]
    pub fn penalties_active(&self) -> bool {
        self.rep_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || !self.logit_bias.is_empty()
    }
}

/// Apply all A7.3 logit processors to `logits` in place, in mlx-lm order:
/// `logit_bias → repetition → presence → frequency`.
///
/// `recent_tokens` must be the **already-trimmed** trailing window (the caller
/// passes `token_history.get(token_history.len().saturating_sub(20)..).unwrap_or(&[])`).
/// This function does NOT trim; it trusts the caller's slice.
///
/// # Application order (mlx-lm `make_logits_processors` L106-126)
///
/// 1. **logit_bias**: `logit[id] += bias` for each `(id, bias)` pair.
/// 2. **repetition_penalty** (`make_repetition_penalty`, sign-aware).
/// 3. **presence_penalty** (`make_presence_penalty`).
/// 4. **frequency_penalty** (`make_frequency_penalty`).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn apply_penalties(
    logits: &mut [f32],
    recent_tokens: &[u32],
    rep_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    logit_bias: &[(u32, f32)],
) {
    let n = logits.len();

    // 1. logit_bias (additive, unconditional).
    for &(id, bias) in logit_bias {
        if (id as usize) < n {
            logits[id as usize] += bias;
        }
    }

    // Early-out when no token-history penalties are needed.
    if recent_tokens.is_empty()
        || (rep_penalty == 1.0 && presence_penalty == 0.0 && frequency_penalty == 0.0)
    {
        return;
    }

    // 2+3+4: single pass through the window; build a count array only when
    // freq_penalty != 0. For presence_penalty we only need presence (0 or 1).
    if frequency_penalty == 0.0 {
        // No frequency penalty — only need unique ids for rep + presence.
        // Dedup without extra allocation: sort + dedup in the tight 20-element window.
        let mut ids: Vec<u32> = recent_tokens.to_vec();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let id_usize = id as usize;
            if id_usize >= n {
                continue;
            }
            if rep_penalty != 1.0 {
                if logits[id_usize] < 0.0 {
                    logits[id_usize] *= rep_penalty;
                } else {
                    logits[id_usize] /= rep_penalty;
                }
            }
            if presence_penalty != 0.0 {
                logits[id_usize] -= presence_penalty;
            }
        }
    } else {
        // Need per-id counts → use a small stack-side HashMap via two passes
        // (sort-based counting avoids allocation for small windows).
        // Window is at most 20 elements; sorting is essentially free.
        let mut ids: Vec<u32> = recent_tokens.to_vec();
        ids.sort_unstable();
        let mut i = 0;
        while i < ids.len() {
            let id = ids[i];
            let id_usize = id as usize;
            if id_usize >= n {
                i += 1;
                continue;
            }
            // Count consecutive equal ids.
            let mut count = 0u32;
            while i < ids.len() && ids[i] == id {
                count += 1;
                i += 1;
            }
            // 2. repetition_penalty (sign-aware, per-id).
            if rep_penalty != 1.0 {
                if logits[id_usize] < 0.0 {
                    logits[id_usize] *= rep_penalty;
                } else {
                    logits[id_usize] /= rep_penalty;
                }
            }
            // 3. presence_penalty (once per unique id).
            if presence_penalty != 0.0 {
                logits[id_usize] -= presence_penalty;
            }
            // 4. frequency_penalty (proportional to count).
            logits[id_usize] -= frequency_penalty * count as f32;
        }
    }
}

/// Return a `[1] I32` Array containing the host-argmax of `logits_flat` after
/// applying penalties. Used for the `temp == 0 AND penalties_active` path.
///
/// One GPU→host transfer per call; the penalty + argmax are done on the CPU.
/// The result has the exact same shape/dtype as `argmax(&logits_flat, -1, device)`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn argmax_with_penalties(
    logits_flat: &Array,
    mask: Option<&[bool]>,
    penalty_cfg: &PenaltyConfig,
    recent_tokens: &[u32],
    device: Device,
) -> Result<Array> {
    let _ = device; // kept for call-site symmetry with sample_token_array
    let vocab = match logits_flat.shape().last() {
        Some(&v) if v > 0 => v as usize,
        _ => {
            return Err(Error::Other(
                "argmax_with_penalties: logits_flat must be [1, vocab]".to_owned(),
            ))
        }
    };

    // GPU→host transfer.
    let mut logits = logits_to_host_f32(logits_flat, vocab)?;

    // Constraint mask: forbidden ids → -inf (same order as sample_token_array).
    if let Some(m) = mask {
        if m.len() != vocab {
            return Err(Error::Other(format!(
                "argmax_with_penalties: mask len {} != vocab {}",
                m.len(),
                vocab
            )));
        }
        for (i, &allowed) in m.iter().enumerate() {
            if !allowed {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    // Apply penalties.
    apply_penalties(
        &mut logits,
        recent_tokens,
        penalty_cfg.rep_penalty,
        penalty_cfg.presence_penalty,
        penalty_cfg.frequency_penalty,
        &penalty_cfg.logit_bias,
    );

    // Host argmax, under the same tie rule as the device reduction.
    let chosen = host_argmax(&logits) as i32;

    Array::from_bytes(&chosen.to_le_bytes(), &[1], Dtype::I32)
        .map_err(|e| Error::Other(format!("argmax_with_penalties: id array build failed: {e}")))
}

/// Index of the maximum of `logits`, resolving ties to the **lowest** index.
///
/// This is the host mirror of the MLX `argmax` reduction that every greedy
/// decode path dispatches on the device. It accumulates with a strict `>` from
/// a `-inf` seed, which gives three properties the device reduction also has:
/// equal values never displace the earlier index, a `NaN` never displaces a
/// real maximum, and a fully constraint-masked (all `-inf`) row yields `0`.
///
/// Greedy selection must not depend on which side of the FFI boundary it runs,
/// so host greedy goes through here rather than `Iterator::max_by` — which
/// returns the *last* maximum, and which lets a `NaN` reset the running best
/// because `partial_cmp` is `None` on it.
fn host_argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_val {
            best_val = l;
            best_idx = i;
        }
    }
    best_idx
}

// ===========================================================================

/// Sampling knobs mirrored from `rmlx_server::engine::SamplingParams`.
///
/// `rmlx-models` must not depend on `rmlx-server`, so the server constructs
/// this small struct from its `SamplingParams` and passes it down through
/// `Architecture::generate_greedy`. Only the fields the host sampler needs
/// are mirrored; penalties + logit_bias live in [`PenaltyConfig`] (A7.3).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — six sampler knob fields; adding a knob requires updating sampling_active(), sampling_distribution(), and all arch generate_greedy call sites"
)]
#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    /// `<= 0.0` ⇒ greedy (the untouched GPU argmax path). `> 0.0` ⇒ host
    /// categorical sampling.
    pub temperature: f32,
    /// Nucleus threshold. `>= 1.0` or `<= 0.0` disables top-p (matches
    /// mlx-lm `make_sampler`, which only applies it when `0 < top_p < 1`).
    pub top_p: f32,
    /// Top-k cutoff. `0` disables.
    pub top_k: u32,
    /// Min-p floor (scaled by the max prob). `0.0` disables.
    pub min_p: f32,
    /// RNG seed. `None` ⇒ a fixed default so absent-seed runs stay
    /// deterministic per process (documented in `seed_or_default`).
    pub seed: Option<u64>,
    /// number of top per-token logprobs to capture. `0` (the default)
    /// disables logprob capture entirely — no log-softmax, no top-k, no alloc
    /// on the decode hot path. `> 0` makes the decode loop call
    /// [`compute_top_logprobs`] once per emitted token.
    pub top_logprobs_k: u32,
}

impl SamplerConfig {
    /// True when the host sampler must run (temperature strictly positive).
    /// `false` keeps the existing greedy GPU argmax path untouched.
    #[inline]
    pub fn sampling_active(&self) -> bool {
        self.temperature > 0.0
    }

    /// Resolve the effective RNG seed. `seed.unwrap_or(0xA7A7)` — absent-seed
    /// runs are deterministic per process. Documented contract; do NOT swap
    /// to entropy without updating callers/tests that assert reproducibility.
    #[inline]
    pub fn seed_or_default(&self) -> u64 {
        self.seed.unwrap_or(0xA7A7)
    }
}

/// Minimal PCG32 (O'Neill, 2014) — hand-rolled, no `rand` dependency
/// (CLAUDE.md forbids casual deps). One instance per request, threaded
/// through every decode step so the random stream is contiguous.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed RNG struct — fields are the complete PCG32 state; any structural change would break the golden-sequence test"
)]
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Seed the generator. Fixed `inc` (must be odd); the seed perturbs the
    /// state between two warm-up steps so nearby seeds diverge immediately.
    pub fn new(seed: u64) -> Self {
        let mut p = Pcg32 {
            state: 0,
            inc: (0x9E37_79B9_7F4A_7C15u64 << 1) | 1,
        };
        p.next_u32();
        p.state = p.state.wrapping_add(seed);
        p.next_u32();
        p
    }

    /// Next 32-bit output (PCG-XSH-RR).
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform `f32` in `[0, 1)` (24-bit mantissa precision).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Read a `[1, vocab]` logits array into a host `Vec<f32>`.
///
/// Handles the two dtypes the model graphs produce (`F32`, `BF16`). This is
/// the single GPU→host transfer the temp>0 path is allowed per step.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn logits_to_host_f32(logits_flat: &Array, vocab: usize) -> Result<Vec<f32>> {
    logits_flat.eval()?;
    let bytes = logits_flat.to_bytes()?;
    match logits_flat.dtype() {
        Dtype::F32 => {
            if bytes.len() < vocab * 4 {
                return Err(Error::Other(format!(
                    "sample_token: F32 logit buffer too small ({} < {})",
                    bytes.len(),
                    vocab * 4
                )));
            }
            Ok((0..vocab)
                .map(|i| {
                    let o = i * 4;
                    f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
                })
                .collect())
        }
        Dtype::Bf16 => {
            if bytes.len() < vocab * 2 {
                return Err(Error::Other(format!(
                    "sample_token: BF16 logit buffer too small ({} < {})",
                    bytes.len(),
                    vocab * 2
                )));
            }
            Ok((0..vocab)
                .map(|i| {
                    let o = i * 2;
                    let raw = u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap());
                    f32::from_bits(u32::from(raw) << 16)
                })
                .collect())
        }
        other => Err(Error::Other(format!(
            "sample_token: unsupported logit dtype {other:?}; expected F32 or BF16"
        ))),
    }
}

/// Numerically-stable softmax of temperature-scaled logits, in place into a
/// fresh `Vec<f32>`. `inv_temp = 1.0 / temperature`. Forbidden positions
/// (already `-inf` from the constraint mask) softmax to exactly `0.0`.
fn softmax_scaled(logits: &[f32], inv_temp: f32) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        let s = l * inv_temp;
        if s > max {
            max = s;
        }
    }
    // All-(-inf) guard: should not happen (constraint masks must keep ≥1
    // token), but stay finite rather than produce NaNs.
    if !max.is_finite() {
        max = 0.0;
    }
    let mut sum = 0.0f32;
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| {
            // Two explicit steps (scale, then subtract-max) — NOT a fused
            // multiply-add. Standard numerically-stable softmax keeps the
            // `exp(x - max)` shift exact; `mul_add` would change rounding.
            let scaled = l * inv_temp;
            let e = (scaled - max).exp();
            sum += e;
            e
        })
        .collect();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in &mut probs {
            *p *= inv;
        }
    }
    probs
}

/// mlx-lm `apply_top_k` (sample_utils.py L130-151): keep the `k` highest
/// probabilities, zero the rest. `k == 0` or `k >= len` ⇒ no-op.
///
/// Equal probabilities are ranked by ascending token id, so when the cut falls
/// inside a tied group the **lowest ids** survive. That is what makes `k == 1`
/// the argmax on every row and not only on rows with a unique maximum — the
/// same lowest-id-wins rule the device reduction uses. mlx-lm's `argpartition`
/// leaves the tied order unspecified; leaving it to the sort's internals here
/// would make which of two equally-likely tokens survives an artefact of
/// pdqsort's pivot choice.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn filter_top_k(probs: &mut [f32], k: usize) {
    let n = probs.len();
    if k == 0 || k >= n {
        return;
    }
    // Build an index array sorted by prob descending, ties by ascending id;
    // everything from rank k on is zeroed.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for &i in &idx[k..] {
        probs[i] = 0.0;
    }
}

/// mlx-lm `apply_top_p` (sample_utils.py L205-237).
///
/// Exact replication of the boundary: sort **ascending**, take an
/// **inclusive** cumulative sum in that order, then a token survives iff its
/// inclusive cumulative probability is **strictly greater than** `1 - top_p`.
/// Survivors keep their prob; the rest go to `0.0`. No-op unless
/// `0 < top_p < 1` (mlx-lm `make_sampler` gate).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn filter_top_p(probs: &mut [f32], top_p: f32) {
    if !(top_p > 0.0 && top_p < 1.0) {
        return;
    }
    let n = probs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // Ascending sort by probability.
    idx.sort_unstable_by(|&a, &b| {
        probs[a]
            .partial_cmp(&probs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = 1.0 - top_p;
    let mut cum = 0.0f32;
    // Walk ascending; cumulative is inclusive of the current element. Keep
    // where cum > threshold (mlx-lm: `mx.where(cumprob > 1 - top_p, ...)`).
    for &i in &idx {
        cum += probs[i];
        if cum <= threshold {
            probs[i] = 0.0;
        }
    }
}

/// mlx-lm `apply_min_p` (sample_utils.py L155-201): remove tokens whose
/// probability is `< max_prob * min_p`. `min_tokens_to_keep` is 1 in our
/// resolved params and the max-prob token always satisfies the threshold
/// (`min_p <= 1`), so no explicit floor handling is needed. `min_p == 0.0`
/// ⇒ no-op.
fn filter_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let max_p = probs.iter().copied().fold(0.0f32, f32::max);
    if max_p <= 0.0 {
        return;
    }
    let cutoff = max_p * min_p;
    for p in probs.iter_mut() {
        if *p < cutoff {
            *p = 0.0;
        }
    }
}

/// Inverse-CDF sample. `r ∈ [0, 1)`; `probs` need not be normalised (uses
/// the running sum). Returns the first index where the cumulative sum
/// strictly exceeds `r * total`. Monotone in `r`: `r→0` ⇒ first nonzero,
/// `r→1` ⇒ last nonzero. Falls back to the last nonzero index on fp drift.
fn sample_inverse_cdf(probs: &[f32], r: f32) -> usize {
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        // Degenerate (all filtered). Return the argmax-equivalent: first
        // index. Callers guarantee ≥1 nonzero in practice (softmax of a
        // finite logit), so this is a safety net only.
        return 0;
    }
    let target = r * total;
    let mut cum = 0.0f32;
    let mut last_nonzero = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_nonzero = i;
            cum += p;
            if cum > target {
                return i;
            }
        }
    }
    last_nonzero
}

/// Host categorical sample for the `temperature > 0` path.
///
/// Returns the chosen token id as a `[1] I32` Array — the exact shape
/// contract of `argmax(&logits_flat, -1, device)` so every decode-loop call
/// site stays unchanged downstream.
///
/// `mask`: optional constraint allow-mask (length `vocab`). When `Some`,
/// `mask[i] == false` ids get `-inf` logits **before** penalties and softmax —
/// so A6 mask composes correctly: mask first, then penalties, then softmax.
///
/// `penalty_cfg`: A7.3 penalty configuration. Applied after the constraint
/// mask, before softmax, in mlx-lm order: logit_bias → rep → presence → freq.
/// `penalty_cfg.penalties_active() == false` skips the `apply_penalties` call.
///
/// `recent_tokens`: trailing window (at most 20 ids) for penalty context.
/// Caller must provide the trimmed slice; this function does not trim.
///
/// `rng`: the per-request `Pcg32`; advanced by exactly one `next_f32()` here
/// so the random stream is contiguous across decode steps.
#[inline(never)]
pub fn sample_token_array(
    logits_flat: &Array,
    sp: &SamplerConfig,
    mask: Option<&[bool]>,
    penalty_cfg: &PenaltyConfig,
    recent_tokens: &[u32],
    rng: &mut Pcg32,
    device: Device,
) -> Result<Array> {
    // `device` is part of the signature for parity with
    // `apply_mask_argmax` (call sites pass the same arg), but the host
    // sampler does all work on the CPU after a single GPU→host transfer,
    // so the MLX stream is not used here.
    let _ = device;
    let vocab = match logits_flat.shape().last() {
        Some(&v) if v > 0 => v as usize,
        _ => {
            return Err(Error::Other(
                "sample_token: logits_flat must be [1, vocab] with vocab > 0".to_owned(),
            ))
        }
    };

    let _ = vocab;
    // 1–5. Build the post-sampling distribution (single GPU→host transfer;
    // mask → penalties → softmax → top_p → min_p → top_k → renormalise).
    // Shared with the speculative `p`/`q` builder so the distribution a
    // caller samples from is byte-identical to the one acceptance scores.
    let probs = sampling_distribution(logits_flat, sp, mask, penalty_cfg, recent_tokens)?;

    // 6. Inverse-CDF sample (sample_inverse_cdf normalises via the running
    // sum, so an explicit renormalise pass is unnecessary).
    let r = rng.next_f32();
    let chosen = sample_inverse_cdf(&probs, r) as i32;

    // 7. Wrap as [1] I32 — argmax shape contract.
    Array::from_bytes(&chosen.to_le_bytes(), &[1], Dtype::I32)
        .map_err(|e| Error::Other(format!("sample_token: id array build failed: {e}")))
}

// ===========================================================================
// — speculative stochastic acceptance (Leviathan 2023, §2.3)
// ===========================================================================
//
// For temperature > 0 speculative decoding the verifier cannot just argmax —
// the algorithm must accept/reject each draft token against the verifier's
// post-sampling distribution so the *output* distribution equals sampling
// from the verifier directly (Leviathan Thm 1). The two distributions must be
// the SAME post-temperature / post-top-p / post-top-k / post-min-p / post-
// penalty distributions the host sampler would produce, or acceptance is
// biased.
//
// accept x with prob min(1, p(x)/q(x)) vs Uniform[0,1]
// on first reject: resample from residual normalize((p - q)+), stop
//
// `q` = draft's sampling distribution at the position that proposed x.
// `p` = verifier's sampling distribution at the same position.

/// Build the host-side post-sampling probability distribution over the vocab
/// from a `[1, vocab]` logits array, applying the SAME pipeline the host
/// categorical sampler uses: optional constraint mask → penalties → softmax
/// (temperature) → top_p → min_p → top_k, then renormalise to sum 1.
///
/// Returned vector sums to ~1.0 (renormalised). Used by both the draft (`q`)
/// and the verifier (`p`) in stochastic speculative acceptance so the two
/// distributions are matched — a hard correctness requirement (Leviathan).
///
/// Mirrors the body of [`sample_token_array`] up to (but not including) the
/// inverse-CDF draw, so the distribution a draft samples from is exactly the
/// distribution the verifier scores acceptance against.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn sampling_distribution(
    logits_flat: &Array,
    sp: &SamplerConfig,
    mask: Option<&[bool]>,
    penalty_cfg: &PenaltyConfig,
    recent_tokens: &[u32],
) -> Result<Vec<f32>> {
    let vocab = match logits_flat.shape().last() {
        Some(&v) if v > 0 => v as usize,
        _ => {
            return Err(Error::Other(
                "sampling_distribution: logits_flat must be [1, vocab] with vocab > 0".to_owned(),
            ))
        }
    };

    let mut logits = logits_to_host_f32(logits_flat, vocab)?;

    if let Some(m) = mask {
        if m.len() != vocab {
            return Err(Error::Other(format!(
                "sampling_distribution: mask len {} != vocab {}",
                m.len(),
                vocab
            )));
        }
        for (i, &allowed) in m.iter().enumerate() {
            if !allowed {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    if penalty_cfg.penalties_active() {
        apply_penalties(
            &mut logits,
            recent_tokens,
            penalty_cfg.rep_penalty,
            penalty_cfg.presence_penalty,
            penalty_cfg.frequency_penalty,
            &penalty_cfg.logit_bias,
        );
    }

    // temperature > 0 guaranteed by the caller's stochastic branch.
    let inv_temp = 1.0 / sp.temperature;
    let mut probs = softmax_scaled(&logits, inv_temp);
    filter_top_p(&mut probs, sp.top_p);
    filter_min_p(&mut probs, sp.min_p);
    filter_top_k(&mut probs, sp.top_k as usize);
    renormalise(&mut probs);
    Ok(probs)
}

/// Renormalise a probability vector to sum 1.0 in place. No-op if the sum is
/// non-positive (degenerate; callers guarantee ≥1 nonzero in practice).
fn renormalise(probs: &mut [f32]) {
    let total: f32 = probs.iter().sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

/// Draw an index from `probs` (need not be normalised) using one `rng` draw.
/// Thin wrapper over [`sample_inverse_cdf`] so the speculative path shares the
/// exact same inverse-CDF semantics (and RNG advance) as the standard sampler.
pub fn sample_index(probs: &[f32], rng: &mut Pcg32) -> usize {
    sample_inverse_cdf(probs, rng.next_f32())
}

/// One Leviathan stochastic-acceptance decision for a single proposed token.
///
/// Given the draft's distribution `q` and the verifier's distribution `p`
/// (both length `vocab`, both already post-sampling and renormalised) and the
/// draft-proposed token id `x`, returns:
///
/// - `Accept` (with prob `min(1, p(x)/q(x))` vs a `rng` draw), meaning emit
///   `x` and continue to the next position.
/// - `Reject(corr)` where `corr` is sampled from the residual
///   `normalize((p - q)+)`, meaning emit `corr` and STOP this round.
///
/// Consumes exactly one `rng.next_f32()` for the accept test; on reject it
/// consumes one more for the residual draw (via [`sample_index`]). The RNG is
/// threaded so tests are deterministic.
///
/// Edge cases:
/// - `q(x) == 0` (draft proposed a token its own distribution gives ~0 mass,
///   e.g. argmax-style): treat acceptance ratio as ≥1 ⇒ accept. This cannot
///   happen when the draft actually sampled from `q`, but is safe.
/// - residual all-zero (`p == q`, accepted branch never reached, or fp
///   drift): fall back to sampling from `p`.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed enum — exactly two speculative-acceptance outcomes; adding an outcome requires updating stochastic_accept and all speculative-round consumers"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptDecision {
    /// Accept the draft token; caller emits `x` and advances.
    Accept,
    /// Reject; caller emits this correction token and stops the round.
    Reject(u32),
}

/// Decide acceptance for proposed token `x` under draft `q` and verifier `p`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn stochastic_accept(p: &[f32], q: &[f32], x: u32, rng: &mut Pcg32) -> Result<AcceptDecision> {
    let xi = x as usize;
    if xi >= p.len() || p.len() != q.len() {
        return Err(Error::Other(format!(
            "stochastic_accept: token id {x} out of range or p/q length mismatch ({} vs {})",
            p.len(),
            q.len()
        )));
    }
    let qx = q[xi];
    let px = p[xi];
    // Acceptance ratio min(1, p(x)/q(x)). q(x)==0 ⇒ accept (ratio ≥ 1).
    let ratio = if qx > 0.0 { (px / qx).min(1.0) } else { 1.0 };
    let u = rng.next_f32();
    if u < ratio {
        return Ok(AcceptDecision::Accept);
    }
    // Reject: resample from residual normalize((p - q)+).
    let mut residual: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).max(0.0))
        .collect();
    let total: f32 = residual.iter().sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        for r in &mut residual {
            *r *= inv;
        }
        let corr = sample_index(&residual, rng) as u32;
        Ok(AcceptDecision::Reject(corr))
    } else {
        // Degenerate residual (p == q after filtering): sample from p.
        let corr = sample_index(p, rng) as u32;
        Ok(AcceptDecision::Reject(corr))
    }
}

// ===========================================================================
// — per-token logprob capture (OpenAI `logprobs` / `top_logprobs`)
// ===========================================================================

/// Per-token log-probability payload for one decode step.
///
/// Computed only when `SamplerConfig::top_logprobs_k > 0`. The disabled path
/// (`k == 0`) never constructs this — no log-softmax, no top-k, no alloc — so
/// the default greedy/sampling decode is byte-identical to before .
///
/// Logprobs are natural-log probabilities of the **raw** model logits
/// (temperature / penalties / nucleus filters are NOT applied — OpenAI reports
/// the model's own per-token log-likelihood, not the post-sampling
/// distribution). Mirrors mlx-vlm `BatchGenerator.compute_logprobs`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — three logprob payload fields; adding a field requires updating compute_top_logprobs and all OpenAI logprobs consumers"
)]
#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    /// The sampled / argmax token id this step.
    pub token_id: u32,
    /// `ln P(token_id)` under the raw-logit softmax. Always `<= 0`.
    pub token_logprob: f32,
    /// The `k` highest-probability `(token_id, logprob)` pairs, descending by
    /// logprob. May or may not include `token_id` itself (OpenAI semantics:
    /// `top_logprobs` is the top-k regardless of which token was chosen).
    pub top: Vec<(u32, f32)>,
}

/// Compute the top-`k` logprobs plus the chosen token's own logprob from the
/// raw `[1, vocab]` logits.
///
/// # Zero-overhead contract
///
/// This function is **only** called from a decode loop behind an explicit
/// `if top_logprobs_k > 0` guard, so callers pay nothing when logprobs are
/// disabled. One GPU→host transfer per step (the same single transfer the
/// `temp > 0` path already performs); the log-softmax + partial top-k select
/// are pure host work.
///
/// `chosen_id` is the already-decided token (argmax or sampled). The natural-log
/// softmax is numerically stable (subtract max before `exp`). Top-k uses a
/// partial selection sort over a `(prob, id)` vector — `O(vocab * k)`, fine for
/// the `k <= 20` OpenAI cap.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn compute_top_logprobs(
    logits_flat: &Array,
    chosen_id: u32,
    k: usize,
) -> Result<TokenLogprobs> {
    let vocab = match logits_flat.shape().last() {
        Some(&v) if v > 0 => v as usize,
        _ => {
            return Err(Error::Other(
                "compute_top_logprobs: logits_flat must be [1, vocab] with vocab > 0".to_owned(),
            ))
        }
    };
    let logits = logits_to_host_f32(logits_flat, vocab)?;

    // Numerically-stable log-softmax: lse = max + ln(sum(exp(l - max))).
    // logprob(i) = l[i] - lse.
    let mut max = f32::NEG_INFINITY;
    for &l in &logits {
        if l > max {
            max = l;
        }
    }
    let mut sum_exp = 0.0f64;
    for &l in &logits {
        sum_exp += f64::from(l - max).exp();
    }
    let lse = f64::from(max) + sum_exp.ln();

    let logprob_of = |id: usize| -> f32 { (f64::from(logits[id]) - lse) as f32 };

    let chosen = chosen_id as usize;
    let token_logprob = if chosen < vocab {
        logprob_of(chosen)
    } else {
        f32::NEG_INFINITY
    };

    // Partial selection of the k highest logits (== k highest logprobs, since
    // log-softmax is monotone in the logit). k is clamped to vocab.
    let k = k.min(vocab);
    let mut idx: Vec<usize> = (0..vocab).collect();
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    for slot in 0..k {
        let mut best = slot;
        for j in (slot + 1)..vocab {
            if logits[idx[j]] > logits[idx[best]] {
                best = j;
            }
        }
        idx.swap(slot, best);
        let id = idx[slot];
        top.push((id as u32, logprob_of(id)));
    }

    Ok(TokenLogprobs {
        token_id: chosen_id,
        token_logprob,
        top,
    })
}

/// F32 negative infinity as a bit pattern. Used to fill forbidden positions.
/// `0xFF800000` is the IEEE 754 bit pattern for F32 -Infinity.
const NEG_INF_F32_BITS: u32 = 0xFF80_0000u32;

/// Apply a boolean allow-mask to `logits` and return the argmax token id as
/// a `[1]` I32 Array (same shape contract as the existing
/// `argmax(&logits_flat, -1, device)` call site).
///
/// # Implementation
///
/// Builds a F32 bias array (0.0 = allowed, -Inf = forbidden), adds it to the
/// logits (casting to F32 if needed), then calls `argmax`. This keeps the
/// heavy work on the GPU and avoids materialising the full logit buffer on
/// the host.
///
/// # Contract
///
/// `logits_flat`: `[1, vocab]` array of F32 or BF16. Other dtypes return Err.
///
/// `mask`: `vocab`-length boolean slice. `mask[i] == true` means token `i`
/// is allowed. At least one entry must be `true`; if all are `false`, the
/// argmax will return an arbitrary forbidden token (callers must not produce
/// all-false masks).
///
/// The `device` parameter selects the MLX stream for the GPU ops.
#[inline(never)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn apply_mask_argmax(logits_flat: &Array, mask: &[bool], device: Device) -> Result<Array> {
    let vocab = mask.len();

    // ── 1. Build the F32 bias buffer on the host ────────────────────────────
    // Allocate once; fill in one pass. Two hot values: 0.0 and -Inf.
    let neg_inf_bytes = NEG_INF_F32_BITS.to_le_bytes();
    let zero_bytes = 0f32.to_le_bytes();

    let mut bias_bytes: Vec<u8> = vec![0u8; vocab * 4];
    for (i, &allowed) in mask.iter().enumerate() {
        if !allowed {
            let off = i * 4;
            bias_bytes[off..off + 4].copy_from_slice(&neg_inf_bytes);
        }
    }
    // Allowed entries are already 0.0 (zero_bytes) from vec initialisation.
    let _ = zero_bytes; // suppress unused warning

    // ── 2. Wrap as [1, vocab] F32 MLX array ────────────────────────────────
    let bias = Array::from_bytes(&bias_bytes, &[1, vocab as i32], Dtype::F32)?;

    // ── 3. Validate dtype ──────────────────────────────────────────────────
    let dtype = logits_flat.dtype();
    if dtype != Dtype::F32 && dtype != Dtype::Bf16 {
        return Err(Error::Other(format!(
            "apply_mask_argmax: unsupported logit dtype {dtype:?}; expected F32 or BF16"
        )));
    }

    // ── 4. Add bias: forbidden tokens get -Inf, allowed stay unchanged ──────
    // MLX promotes BF16+F32 to F32 automatically; the bias is F32 so the
    // result of add is F32 regardless of input dtype.
    let biased = add(logits_flat, &bias, device)
        .map_err(|e| Error::Other(format!("apply_mask_argmax: add failed: {e}")))?;

    // ── 5. Argmax along last axis ───────────────────────────────────────────
    // axis = -1 selects the vocab dimension of [1, vocab], producing [1] I32.
    argmax(&biased, -1, device)
        .map_err(|e| Error::Other(format!("apply_mask_argmax: argmax failed: {e}")))
}

// ───── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "sampler_tests.rs"]
mod sampler_tests;
