//! Smoke-probe types: [`ProbeStep`], [`SmokeVerdict`], byte-level logit helpers.

use rmlx_mlx::{Array, Dtype};

use crate::sampler::TokenLogprobs;

// ---------------------------------------------------------------------------
// Smoke probe types
// ---------------------------------------------------------------------------

/// One record per autoregressive step produced by `generate_greedy`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed per-step probe record — five fields are the complete decode-step observability contract; adding a field requires updating all generate_greedy call sites"
)]
#[derive(Debug, Clone)]
pub struct ProbeStep {
    /// Token id selected by greedy argmax.
    pub token_id: u32,
    /// Single-piece string from `tokenizer.id_to_token(token_id)` — used for
    /// display and the broken-punct-loop heuristic.
    ///
    /// `Box<str>` saves 8 B per step vs `String`: 2 words (ptr + len) vs 3 words
    /// (ptr + len + capacity). For 4096-token sequences: ~32 KB saved + 4096 fewer
    /// allocator capacity slots. Construction: `piece.into_boxed_str()`.
    pub piece: Box<str>,
    /// `max(|logits|)` at this step. Finite normally; NaN/Inf signals a hazard.
    pub max_abs_logit: f32,
    /// Number of NaN cells in the logit vector at this step.
    pub nan_count: usize,
    /// per-token top-k logprobs. `None` unless the request set
    /// `top_logprobs_k > 0` (the zero-overhead default leaves this `None` and
    /// never runs the log-softmax / top-k path).
    pub logprobs: Option<TokenLogprobs>,
}

/// Verdict returned by `classify_smoke`.
///
/// Heuristic: after the seeded 8-token probe, the snapshot is flagged broken on
/// a degenerate repeat — see `classify_smoke` for the exact (B5b-widened) rule.
/// Mirrors the Qwen3.6-35B-A3B-mxfp8 pattern in CLAUDE.md "mxfp8 broken-snapshot
/// hazard" and the gemma-4-26b-a4b single-CJK-token loop (B5b audit).
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — four smoke outcomes (Ok/BrokenPunctLoop/BrokenNan/Inconclusive); adding an outcome requires updating classify_smoke and all SmokeVerdict match arms in the serve path"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Smoke-probe outcome for a Gemma4 forward pass.
pub enum SmokeVerdict {
    /// Generation succeeded and output looks coherent.
    Ok,
    /// A degenerate repeat loop: ≤ 2 distinct ids + single-char punct piece, OR
    /// ≥ `LOOP_K` consecutive identical ids, OR a single-char piece (any
    /// category) dominating ≥ `LOOP_K` of the window. Name kept (not just
    /// punct) to keep exit-code / HTTP-503 / test mapping stable.
    BrokenPunctLoop {
        /// The punctuation piece that dominated the output.
        dominant_piece: String,
        /// Number of distinct token ids in the sample window.
        distinct_ids: usize,
    },
    /// A NaN appeared in the logit vector.
    BrokenNan {
        /// Decode step at which the NaN was detected.
        at_step: usize,
    },
    /// Generation stopped early but neither hazard fired (e.g. EOS at step 1).
    Inconclusive {
        /// Human-readable explanation of why the verdict is inconclusive.
        reason: String,
    },
}

/// capture top-`k` logprobs for an already-chosen token.
///
/// Called ONLY behind a `k > 0` guard at every decode-loop call site, so the
/// default decode path (no logprobs requested) never touches log-softmax /
/// top-k and stays byte-identical. `chosen` is the sampled/argmax token Array
/// (`[1] I32`); it is materialised here so we can pin the chosen token's own
/// logprob to the same logits the rest of `top` is computed from. Errors are
/// downgraded to `None` — logprob capture must never abort generation.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(super) fn capture_logprobs(
    logits_flat: &Array,
    chosen: &Array,
    k: usize,
) -> Option<TokenLogprobs> {
    let top_bytes = match chosen.to_bytes() {
        Ok(b) if b.len() >= 4 => b,
        _ => return None,
    };
    let chosen_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    crate::sampler::compute_top_logprobs(logits_flat, chosen_id, k).ok()
}

// ---------------------------------------------------------------------------
// Byte-level logit helpers
// ---------------------------------------------------------------------------

/// Count NaN values in a byte buffer of floats (F32 or Bf16).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub(super) fn count_nan_in_bytes(bytes: &[u8], dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .filter(|c| f32::from_le_bytes((*c).try_into().unwrap()).is_nan())
            .count(),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .filter(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                let f32_bits = u32::from(raw) << 16;
                f32::from_bits(f32_bits).is_nan()
            })
            .count(),
        _ => 0,
    }
}

/// Compute max(|logit|) from a byte buffer. Returns 0.0 on empty or unknown dtype.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub(super) fn max_abs_from_bytes(bytes: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes((*c).try_into().unwrap()).abs())
            .fold(0.0_f32, f32::max),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                let f32_bits = u32::from(raw) << 16;
                f32::from_bits(f32_bits).abs()
            })
            .fold(0.0_f32, f32::max),
        _ => 0.0,
    }
}
