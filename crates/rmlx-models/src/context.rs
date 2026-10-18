//! Effective context ceiling — the single resolution every context cap reads.
//!
//! One question is asked in several places: *how many tokens may this run
//! address?* The KV ring sizes itself to the answer, the server's admission
//! guard rejects prompts above it, `rmlx baseline` / `rmlx bench` default their
//! prompt cap to it, and `/v1/models` reports it. [`resolve_context`] is the
//! only place that computes it.
//!
//! # Contract
//!
//! A checkpoint's **positional capacity** is what its RoPE can address:
//! `max_position_embeddings`, extended by an active RoPE scaling
//! ([`ContextScaling`]) to `factor * original_max`. Scaling comes from one of
//! two places, and the source decides how loud the engine is about it:
//!
//! | Source | Behaviour |
//! |---|---|
//! | `rope_scaling` in `config.json` | extend silently — the checkpoint author declared the window |
//! | `--yarn-factor` / `--yarn-original-max` | extend, and `warn!` naming the trained window — the operator is taking the risk |
//! | neither | capacity is the trained window |
//!
//! A requested context above the capacity is **refused**, never clamped. RoPE
//! extrapolated past the trained window without scaling does not degrade
//! gracefully — it produces incoherent output — so silently serving a shorter
//! window (the previous behaviour) hid both the truncation and its cause. The
//! refusal names the request, the capacity, the trained window and the flag
//! that would lift it.
//!
//! # Relation to other backends
//!
//! `mlx-lm` applies no cap and extrapolates. `llama.cpp` warns
//! (`possible training context overflow`) and proceeds, and lets its
//! `--rope-scaling` / `--yarn-*` flags override what the model file declares.
//! vLLM refuses a `max_model_len` above the derived window and names the
//! override in the error. rMLX takes vLLM's refusal (a wrong answer is worse
//! than a clear error) with llama.cpp's flag precedence: an explicit
//! `--yarn-factor` overrides the checkpoint's own `rope_scaling`, so the
//! operator always has a way through.

use rmlx_core::error::{Error, Result};
use rmlx_kv_quant::KvCache;

/// Where an active RoPE scaling came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed set — a scaling is declared by the checkpoint or requested by the operator; a third source would change the warn/silent contract"
)]
pub enum ScalingSource {
    /// `rope_scaling` in the checkpoint's `config.json`.
    Config,
    /// An operator flag (`--yarn-factor` / `--yarn-original-max`).
    Operator,
}

/// An active RoPE scaling and the window it reaches.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "three fields are the complete scaling contract; adding one requires updating every construction site and the refusal wording"
)]
pub struct ContextScaling {
    /// Extension factor (`> 1.0` to extend).
    pub factor: f32,
    /// Pre-extension context size the scaling interpolates from.
    pub original_max: f32,
    /// Which of the two mechanisms produced this scaling.
    pub source: ScalingSource,
}

impl ContextScaling {
    /// Largest position this scaling addresses: `factor * original_max`,
    /// saturated into `i32`. Returns `0` for a factor that does not extend.
    #[must_use]
    pub fn extended_max(&self) -> i32 {
        if self.factor <= 1.0 || self.original_max <= 0.0 {
            return 0;
        }
        let extended = f64::from(self.factor) * f64::from(self.original_max);
        if extended >= f64::from(i32::MAX) {
            i32::MAX
        } else {
            extended as i32
        }
    }
}

/// What a checkpoint can address, before `--max-ctx` narrows it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "three fields are the complete per-checkpoint context contract; every architecture builds one and adding a field requires revisiting all of them"
)]
pub struct ContextLimits {
    /// `max_position_embeddings` from `config.json`. `0` means the
    /// architecture does not expose it — treat the capacity as unknown.
    pub trained_max: i32,
    /// Active RoPE scaling, or `None` when the run uses plain RoPE.
    pub scaling: Option<ContextScaling>,
    /// Whether this architecture implements RoPE scaling at all. Decides
    /// whether a refusal may name `--yarn-factor` as a way out.
    pub scaling_supported: bool,
}

impl ContextLimits {
    /// Limits for an architecture that implements no RoPE scaling: the
    /// trained window is the whole story.
    #[must_use]
    pub fn trained_only(trained_max: i32) -> Self {
        Self {
            trained_max,
            scaling: None,
            scaling_supported: false,
        }
    }

    /// Largest position the checkpoint can address. `0` when unknown.
    #[must_use]
    pub fn positional_max(&self) -> i32 {
        self.trained_max
            .max(self.scaling.map_or(0, |s| s.extended_max()))
    }

    /// Largest position the checkpoint can address, as a token count.
    ///
    /// `None` when the architecture does not expose `max_position_embeddings`
    /// — the resolver then accepts any `--max-ctx`, so publishing a number
    /// here would state the opposite of the behaviour.
    #[must_use]
    pub fn positional_tokens(&self) -> Option<usize> {
        usize::try_from(self.positional_max())
            .ok()
            .filter(|n| *n > 0)
    }

    /// The half of a refusal that tells the operator what to do next.
    fn lift_hint(&self, positional_max: i32) -> String {
        if let Some(s) = self.scaling {
            let declared = match s.source {
                ScalingSource::Config => "config.json declares YaRN scaling",
                ScalingSource::Operator => "--yarn-factor requested YaRN scaling",
            };
            return format!(
                "{declared} factor {f} over original {o}, reaching {positional_max}; raise \
                 --yarn-factor, or lower the requested context to {positional_max}",
                f = s.factor,
                o = s.original_max,
            );
        }
        if self.scaling_supported {
            format!(
                "the checkpoint declares no rope_scaling; extend the window with \
                 --yarn-factor <f> (and --yarn-original-max <n>, default {trained}) so that \
                 f * n covers the request, or lower the requested context to {positional_max}",
                trained = self.trained_max,
            )
        } else {
            format!(
                "this architecture has no RoPE-scaling support in rMLX, so the trained window \
                 is the hard limit; lower the requested context to {positional_max}"
            )
        }
    }
}

/// The resolved context bounds for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "the three numbers every consumer of the ceiling reads; a fourth would need a new consumer to justify it"
)]
pub struct ResolvedContext {
    /// Largest position the checkpoint can address (`0` when unknown).
    pub positional_max: i32,
    /// The context bound in force for this run: what the KV ring may grow to
    /// and what the admission guard compares prompts against.
    pub ceiling: i32,
    /// Size the per-layer KV ring is first allocated at. The ring grows from
    /// here toward `ceiling` as the prompt fills.
    pub initial_max_seq: i32,
}

impl ResolvedContext {
    /// [`Self::ceiling`] as a token count.
    ///
    /// [`resolve_context`] never produces a non-positive ceiling — every
    /// branch yields either a positive `--max-ctx` or a positive fallback — so
    /// callers converting for an admission check or a report do not clamp.
    #[must_use]
    pub fn ceiling_tokens(&self) -> usize {
        usize::try_from(self.ceiling).unwrap_or(0)
    }

    /// [`Self::positional_max`] as a token count, `None` when unknown.
    #[must_use]
    pub fn positional_tokens(&self) -> Option<usize> {
        usize::try_from(self.positional_max).ok().filter(|n| *n > 0)
    }

    /// How many tokens a request may generate: `n_tokens`, cut to the room the
    /// ceiling leaves after the prompt.
    ///
    /// The cut is `ceiling - prompt_len` tokens. The plain decode loop needs one
    /// position less than that. A speculative round writes its carry token and
    /// every draft in one forward, and the widest round of a request ends on
    /// position `prompt_len + n_tokens`, so this is the largest budget that
    /// keeps every writer under the ceiling.
    ///
    /// A request for no tokens always gets `Ok(0)`.
    ///
    /// # Errors
    ///
    /// [`Error::KvCeilingExceeded`] when the request asks for tokens and the
    /// prompt leaves no room for the first one. Its `requested` is the length
    /// that one token needs.
    pub fn decode_budget(&self, prompt_len: usize, n_tokens: usize) -> Result<usize> {
        let room = self.ceiling_tokens().saturating_sub(prompt_len);
        if room == 0 && n_tokens > 0 {
            let requested = i32::try_from(prompt_len.saturating_add(1)).unwrap_or(i32::MAX);
            tracing::warn!(
                requested,
                ceiling = self.ceiling,
                "KV max-ctx ceiling exceeded — the prompt leaves no room to generate"
            );
            return Err(Error::KvCeilingExceeded {
                requested,
                ceiling: self.ceiling,
            });
        }
        Ok(n_tokens.min(room))
    }
}

/// Resolve the context bounds for one run, and warn of an override or a
/// scaling that the operator should know about.
///
/// `max_ctx_override` is the operator's `--max-ctx` (or a per-request
/// `max_ctx` field). `None`, or a non-positive value, falls back to
/// `min(positional capacity, KV_MAX_SEQ_DEFAULT)`.
///
/// # Errors
///
/// [`Error::ContextCeilingExceeded`] when `max_ctx_override` is above the
/// checkpoint's positional capacity. The engine never clamps: a request it
/// cannot honour is refused with the numbers and the lifting flag named.
pub fn resolve_context(
    limits: &ContextLimits,
    max_ctx_override: Option<i32>,
) -> Result<ResolvedContext> {
    let resolved = resolve_ceiling(limits, max_ctx_override)?;
    if let Some(n) = max_ctx_override.filter(|n| *n <= 0) {
        tracing::warn!(
            max_ctx_override = n,
            "max_ctx_override <= 0 treated as unset; using arch default"
        );
    }
    if let Some(s) = limits.scaling {
        if s.source == ScalingSource::Operator && resolved.ceiling > limits.trained_max {
            tracing::warn!(
                ceiling = resolved.ceiling,
                trained_max = limits.trained_max,
                factor = s.factor,
                original_max = s.original_max,
                "context extends past the checkpoint's trained window on operator-requested \
                 YaRN scaling; output quality past the trained window is the operator's risk"
            );
        }
    }
    Ok(resolved)
}

/// The bounds [`resolve_context`] resolves, without its warnings.
///
/// A caller that reads the ceiling of a request a second time, to cut its
/// generation to the room under it, calls this one, so the request logs each
/// warning once.
///
/// # Errors
///
/// As [`resolve_context`].
pub(crate) fn resolve_ceiling(
    limits: &ContextLimits,
    max_ctx_override: Option<i32>,
) -> Result<ResolvedContext> {
    let default = rmlx_kv_quant::KV_MAX_SEQ_DEFAULT;
    let positional_max = limits.positional_max();
    let fallback = if positional_max > 0 {
        positional_max.min(default)
    } else {
        default
    };

    let ceiling = match max_ctx_override {
        Some(n) if n > 0 => {
            if positional_max > 0 && n > positional_max {
                return Err(Error::ContextCeilingExceeded {
                    requested: n,
                    positional_max,
                    trained_max: limits.trained_max,
                    lift: limits.lift_hint(positional_max),
                });
            }
            n
        }
        Some(_) | None => fallback,
    };

    Ok(ResolvedContext {
        positional_max,
        ceiling,
        initial_max_seq: default.min(ceiling),
    })
}

/// Bind the K/V of a reused prompt-cache prefix to the context ceiling of the
/// request that reuses it.
///
/// A stored entry's caches carry the ceiling of the request that filled them.
/// A request under another ceiling has to succeed or fail as a cold run of the
/// same prompt would under its own: a lower ceiling refuses a prompt above it,
/// and a higher one lets the tail and the decode grow past the stored ceiling.
/// A request is never turned into a miss by a differing ceiling.
///
/// # Errors
///
/// [`Error::KvCeilingExceeded`] when `prompt_len` is above `ceiling`, the typed
/// refusal a cold prefill of that prompt raises. Its `requested` is the prompt
/// length, where a cold prefill reports the length at the chunk that crossed.
pub(crate) fn rebind_to_ceiling(
    kv_caches: Vec<KvCache>,
    ceiling: i32,
    prompt_len: usize,
) -> Result<Vec<KvCache>> {
    if usize::try_from(ceiling).is_ok_and(|c| prompt_len > c) {
        let requested = i32::try_from(prompt_len).unwrap_or(i32::MAX);
        tracing::warn!(
            requested,
            ceiling,
            "KV max-ctx ceiling exceeded — rejecting reused prefix"
        );
        return Err(Error::KvCeilingExceeded { requested, ceiling });
    }
    Ok(kv_caches
        .into_iter()
        .map(|cache| cache.with_max_seq_ceiling(ceiling))
        .collect())
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod context_tests;
