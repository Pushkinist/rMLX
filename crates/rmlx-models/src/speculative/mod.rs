//! Speculative decoding.
//!
//! Wraps a (verifier, draft) pair of `Architecture` instances.
//!
//! - `spec_forward(input_ids, k)` runs the verifier on `input_ids` and
//!   returns logits for the last `k` positions.
//! - `spec_generate_greedy_cached` runs greedy speculative decoding with
//!   persistent verifier + draft KV caches and `KvCache::truncate_to`-based
//!   rollback on partial acceptance. Mirrors mlx-lm's
//!   `speculative_generate_step`. Per-round verifier cost is O(K), not
//!   O(prompt_len) — 24+ TPS on `gemma-4-31b-mxfp8` at 4k context, against a
//!   0.45 TPS structural ceiling for per-round full re-prefill.
//!
//! An architecture whose `forward_seq_last_k_with_cache` is unwired surfaces
//! that error; there is no re-prefill fallback path.
//!
//! Design and measurement reports are in `docs/reports/`.

#![allow(
    clippy::cognitive_complexity,
    clippy::manual_let_else,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::used_underscore_items
)]
pub mod dflash;
pub mod eagle3;
pub mod gemma4_assistant;
pub mod mtp;

pub(crate) mod draft_kind;
pub(crate) mod round_stats;

// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};

use crate::arch::{load_model, Architecture, LoadOpts};
use crate::decode_loop::ProbeStep;
pub use draft_kind::{Declared, DraftKind};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
pub(crate) use round_stats::{ms, phases_charged, RoundStats, SpecLoop, PHASE_TARGET};

/// Resolve the context bounds a speculative pair runs under.
///
/// The verifier owns the KV geometry — the drafter inherits its cache sizing
/// and its positional limit — so the verifier's [`crate::context::ContextLimits`]
/// are what bound the round loop. Routing through
/// [`crate::context::resolve_context`] keeps the speculative path on the one
/// resolution every other context cap reads, and gives it the same refusal:
/// a `--max-ctx` above the verifier's positional capacity used to be taken
/// verbatim here and only surfaced as a cache overflow mid-round.
///
/// # Errors
///
/// [`rmlx_core::error::Error::ContextCeilingExceeded`] when `max_ctx_override`
/// is above the verifier's positional capacity.
pub(crate) fn verifier_context(
    verifier: &Architecture,
    max_ctx_override: Option<i32>,
) -> Result<crate::context::ResolvedContext> {
    crate::context::resolve_context(&verifier.context_limits(), max_ctx_override)
}

/// Guard the one verifier logit row a speculative driver selects from at
/// prefill.
///
/// Every driver in this module family prefills the verifier, argmaxes exactly
/// one logit row for the first bonus token, and emits it through `step_fn`.
/// That is the same position the non-speculative arches guard and it carries
/// the same failure: greedy selection over a NaN row returns index 0 whatever
/// the model computed, so under `--draft-model` a NaN prefill used to produce a
/// full-length garbage run with no guard, no verdict and exit 0.
///
/// Costs one host readback of the vocab row per request, at TTFT.
///
/// The **per-round** verify logits are deliberately not guarded. Those run
/// `n_tokens / block_size` times per request, so a readback there is a
/// throughput cost on the hot path — the same reason the shared decode loop
/// computes no per-step count. Guarding the prefill row is what makes the
/// speculative path match the ordinary path; per-step detection is a separate
/// open question for both.
pub(crate) fn guard_verifier_prefill_logits(
    verifier: &Architecture,
    logits: &Array,
    prompt_len: usize,
) -> Result<()> {
    Array::eval(logits)?;
    let bytes = logits.to_bytes()?;
    let dtype = logits.dtype();
    let nan_count = count_nan_in_bytes(&bytes, dtype);
    let max_abs_logit = max_abs_from_bytes(&bytes, dtype);
    crate::decode_loop::reject_nan_prefill(
        verifier.arch_class(),
        dtype,
        nan_count,
        max_abs_logit,
        prompt_len,
    )
}

/// Resident KV bytes held by a verifier's own caches.
///
/// Same basis as the per-arch `generate_greedy` byte total: the attention
/// caches plus, on hybrid archs, the recurrent linear-attention state. Only the
/// verifier's caches count — the draft's are an implementation detail of the
/// accelerator, and including them would make a speculative row incomparable
/// with the ordinary row for the same model and context.
pub(crate) fn verifier_kv_bytes(kv: &[KvCache], lin: Option<&[LinearAttnCache]>) -> u64 {
    kv.iter().map(KvCache::resident_bytes).sum::<u64>()
        + lin.map_or(0, |l| l.iter().map(LinearAttnCache::resident_bytes).sum())
}

/// Whether two snapshot paths name the same directory.
///
/// Compares canonical paths so `.`-relative and symlinked spellings of one
/// snapshot still match; falls back to the literal paths when either side
/// cannot be canonicalised (e.g. it does not exist).
fn same_snapshot(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}

/// How many trailing ids one tokenizer may carry that the other does not.
///
/// Snapshots of one family ship the same vocabulary with a different tail of
/// special tokens — an audio or TTS release appends a few, a base release omits
/// them. Every id both sides carry must still name the same piece; only the
/// tail is allowed to differ, and only by this much. It is llama.cpp's
/// `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`, so the two engines admit the same pairs.
///
/// An id in that tail is one the verifier can emit and the draft's tokenizer
/// never named; it is fed back to the draft as context and indexes an
/// embedding row there. That row exists — untrained, not out of bounds — only
/// because [`SpeculativeDispatcher::new`] pins the two `vocab_size` values
/// equal. The tolerance depends on that check.
const VOCAB_TAIL_TOLERANCE: usize = 128;

/// The largest token id a tokenizer may carry and still be compared here.
///
/// The comparison walks every id up to the smaller side's last one, so a
/// stray sentinel at an enormous id would turn model load into a spin over
/// the whole span. No tokenizer this backend loads comes within an order of
/// magnitude of this.
const VOCAB_ID_CEILING: u32 = 1 << 22;

/// The vocabulary a snapshot's `tokenizer.json` declares, added tokens included.
fn snapshot_vocab(dir: &Path) -> Result<HashMap<String, u32>> {
    let path = dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| Error::Model(format!("load tokenizer {}: {e}", path.display())))?;
    Ok(tokenizer.get_vocab(true))
}

/// `vocab` inverted to id order, refusing an id two pieces claim.
///
/// `get_vocab(true)` merges the added tokens into the model vocabulary by
/// piece, and nothing there promises the result is injective. Letting the
/// `collect` pick a winner would make the verdict depend on hash order.
fn vocab_by_id<'a>(side: &str, vocab: &'a HashMap<String, u32>) -> Result<BTreeMap<u32, &'a str>> {
    let mut by_id: BTreeMap<u32, &str> = BTreeMap::new();
    for (piece, id) in vocab {
        if let Some(other) = by_id.insert(*id, piece.as_str()) {
            return Err(Error::SpeculativePairing {
                reason: format!(
                    "the {side} tokenizer names token id {id} twice, as {other:?} and \
                     {piece:?} — a draft proposal of that id has no single meaning"
                ),
            });
        }
    }
    Ok(by_id)
}

/// Whether the draft's tokenizer can stand in for the verifier's.
///
/// A draft proposes token *ids*, and the verifier scores them as indices into
/// its own vocabulary. If the two tokenizers disagree on what an id means, the
/// pair does not fail — it serves garbage, at a low accept rate, with no error.
/// Comparing `vocab_size` cannot see that: Gemma 3 and Gemma 4 both declare
/// 262144 and share no vocabulary. So this compares the pieces, id by id, over
/// every id both sides carry, and tolerates a short tail of ids only one side
/// has (see [`VOCAB_TAIL_TOLERANCE`]).
///
/// The stop ids are deliberately not compared: the prompt is tokenized and the
/// stop decided by the verifier alone, and the draft only ever sees ids.
///
/// # Errors
/// [`Error::SpeculativePairing`], naming the first id whose piece differs, the
/// size of a tail the tolerance does not cover, an id two pieces claim, or an
/// id past [`VOCAB_ID_CEILING`].
pub(crate) fn vocab_pairing_verdict(
    verifier: &HashMap<String, u32>,
    draft: &HashMap<String, u32>,
) -> Result<()> {
    let v = vocab_by_id("verifier", verifier)?;
    let d = vocab_by_id("draft", draft)?;
    let (Some(v_last), Some(d_last)) = (v.keys().next_back(), d.keys().next_back()) else {
        return Err(Error::SpeculativePairing {
            reason: "a tokenizer.json on one side declares no vocabulary".to_owned(),
        });
    };
    let shared_end = (*v_last).min(*d_last);
    if shared_end >= VOCAB_ID_CEILING {
        return Err(Error::SpeculativePairing {
            reason: format!(
                "both tokenizers carry token id {shared_end}, past the {VOCAB_ID_CEILING} this \
                 comparison walks — not a vocabulary this backend recognises"
            ),
        });
    }
    for id in 0..=shared_end {
        let (vp, dp) = (v.get(&id), d.get(&id));
        if vp != dp {
            return Err(Error::SpeculativePairing {
                reason: format!(
                    "draft tokenizer is not the verifier's: token id {id} is {} in the \
                     verifier and {} in the draft — a draft can only propose ids the \
                     verifier reads the same way",
                    vp.map_or("absent".to_owned(), |p| format!("{p:?}")),
                    dp.map_or("absent".to_owned(), |p| format!("{p:?}")),
                ),
            });
        }
    }
    let tail_start = shared_end.saturating_add(1);
    let tail = v.range(tail_start..).count() + d.range(tail_start..).count();
    if tail > VOCAB_TAIL_TOLERANCE {
        return Err(Error::SpeculativePairing {
            reason: format!(
                "draft tokenizer is not the verifier's: the two agree up to id {shared_end} \
                 and then one side carries {tail} more ids, above the {VOCAB_TAIL_TOLERANCE} \
                 a trailing run of special tokens is allowed to differ by"
            ),
        });
    }
    Ok(())
}

/// Holds a verifier and, for the two-model path, a draft `Architecture`.
///
/// When a draft is present:
/// - `verifier.vocab_size() == draft.vocab_size()` (asserted in `new`).
/// - The two tokenizers name the same piece at every id both carry (enforced
///   in `load_speculative`, see [`vocab_pairing_verdict`]).
/// - The two architectures come from distinct snapshot dirs (enforced in
///   `load_speculative`).
/// - Both share the same `Device` at construction time.
#[allow(missing_debug_implementations)]
pub struct SpeculativeDispatcher {
    /// The full verifier model that scores and accepts/rejects draft tokens.
    pub verifier: Architecture,
    /// The lightweight draft model that proposes candidate tokens.
    ///
    /// `None` for a sidecar drafter (MTP / EAGLE-3 / DFlash): those are small
    /// heads owned by the serve layer and driven by their own round loops,
    /// which read only the verifier. A sidecar dispatcher has no second full
    /// model to hold, and `spec_generate_*` refuses to run on one. Private so
    /// the empty slot cannot be filled from outside the constructors.
    draft: Option<Architecture>,
    device: Device,
}

impl SpeculativeDispatcher {
    /// Construct a dispatcher from two pre-loaded `Architecture` values.
    ///
    /// Asserts that the two logit rows are the same width. The greedy loop
    /// only compares argmax ids, but the stochastic loop takes `p` and `q` as
    /// whole distributions and the acceptance test indexes both by one id, so
    /// a draft whose head is padded to a different width has no `q` to hand it.
    /// Whether the ids *mean* the same thing is the tokenizer's business, and
    /// `load_speculative` settles that before either model is loaded.
    pub fn new(verifier: Architecture, draft: Architecture, device: Device) -> Result<Self> {
        if verifier.vocab_size() != draft.vocab_size() {
            return Err(Error::Model(format!(
                "speculative: logit width mismatch — verifier vocab_size={} draft vocab_size={}; \
                 the stochastic acceptance test needs one distribution per id on both sides",
                verifier.vocab_size(),
                draft.vocab_size()
            )));
        }
        Ok(Self {
            verifier,
            draft: Some(draft),
            device,
        })
    }

    /// Load a verifier from a snapshot directory, with no draft model.
    ///
    /// The sidecar-drafter counterpart to [`Self::load_speculative`]: MTP /
    /// EAGLE-3 / DFlash drafters are small heads the serve layer loads and
    /// drives itself, so there is no second full model. One `load_model` call,
    /// one resident copy of the weights, and `spec_generate_*` is unavailable
    /// on the result.
    ///
    /// # Errors
    /// Propagates any [`load_model`] failure.
    pub fn load_verifier_only(verifier_dir: &Path, device: Device) -> Result<Self> {
        tracing::info!(
            verifier = %verifier_dir.display(),
            "speculative: load_verifier_only — loading verifier (no draft model)"
        );
        let verifier = load_model(verifier_dir, device, &LoadOpts::default())?;
        tracing::info!(
            verifier_summary = %verifier.config_summary(),
            "speculative: load_verifier_only — loaded"
        );
        Ok(Self {
            verifier,
            draft: None,
            device,
        })
    }

    /// Load both verifier and draft from snapshot directories.
    ///
    /// The two `load_model` calls run sequentially under the single Apple
    /// Silicon Metal context.
    ///
    /// # Errors
    /// Returns `Error::Model` when both sides name the same directory: that
    /// materialises the weights twice for no benefit — the draft would cost
    /// exactly as much to run as the verifier it is meant to outrun. A caller
    /// wanting one model wants [`Self::load_verifier_only`].
    ///
    /// Returns [`Error::SpeculativePairing`] when the draft's tokenizer is not
    /// the verifier's — see [`vocab_pairing_verdict`]. Both checks run before
    /// any weight is read.
    pub fn load_speculative(verifier_dir: &Path, draft_dir: &Path, device: Device) -> Result<Self> {
        if same_snapshot(verifier_dir, draft_dir) {
            return Err(Error::Model(format!(
                "load_speculative: verifier and draft name the same snapshot directory ({}) — \
                 that loads the weights twice for no speedup. Use load_verifier_only for a \
                 sidecar drafter, or point --draft-model at a smaller model.",
                verifier_dir.display()
            )));
        }
        vocab_pairing_verdict(&snapshot_vocab(verifier_dir)?, &snapshot_vocab(draft_dir)?)?;
        tracing::info!(
            verifier = %verifier_dir.display(),
            draft = %draft_dir.display(),
            "speculative: load_speculative — loading verifier"
        );
        let verifier = load_model(verifier_dir, device, &LoadOpts::default())?;
        tracing::info!(
            verifier = %verifier_dir.display(),
            "speculative: load_speculative — loading draft"
        );
        let draft = load_model(draft_dir, device, &LoadOpts::default())?;
        tracing::info!(
            verifier_summary = %verifier.config_summary(),
            draft_summary = %draft.config_summary(),
            "speculative: load_speculative — both loaded"
        );
        Self::new(verifier, draft, device)
    }

    /// Speculative forward step — verifier-only routing.
    ///
    /// Runs the verifier on `input_ids` (length L) and returns logits
    /// for the last `k` positions: shape `[1, k, vocab_size]`.
    /// The verifier's existing prefill path produces all positions'
    /// logits internally; this method routes the last-`k` slice
    /// instead of the last-1 slice. It proposes and accepts nothing —
    /// `spec_generate_greedy` is the full round loop.
    pub fn spec_forward(&self, input_ids: &[u32], k: usize) -> Result<Array> {
        if input_ids.is_empty() {
            return Err(Error::Model("spec_forward: empty input_ids".to_owned()));
        }
        if k == 0 || k > input_ids.len() {
            return Err(Error::Model(format!(
                "spec_forward: k={k} out of range for L={}",
                input_ids.len()
            )));
        }
        self.verifier.forward_seq_last_k(input_ids, k, self.device)
    }

    /// Vocabulary size shared by both models (asserted at construction).
    pub fn vocab_size(&self) -> usize {
        self.verifier.vocab_size()
    }

    /// The draft model, or an error when this dispatcher holds only a verifier.
    fn draft_model(&self) -> Result<&Architecture> {
        self.draft.as_ref().ok_or_else(|| {
            Error::Model(
                "speculative: two-model generation needs a draft model, but this dispatcher \
                 holds only a verifier — a sidecar drafter runs its own round loop instead"
                    .to_owned(),
            )
        })
    }

    /// The compute device both models were loaded on (— the assistant
    /// MTP round-loop needs it to issue verifier + drafter forwards).
    pub fn device(&self) -> Device {
        self.device
    }

    /// Greedy speculative decoding over persistent verifier + draft caches.
    ///
    /// Algorithm (Leviathan 2023, greedy variant). The verifier holds a
    /// persistent KV cache; each round it re-feeds only the K new draft
    /// tokens through its cache, advancing offset by K. Per-round verifier
    /// compute = K-token forward + (0 if all-accept; 1 single-token forward
    /// otherwise to recompute next-round T_carry past correction).
    ///
    /// ```text
    /// init: prefill verifier on prompt → cache offset = L; T_carry = argmax(last logit)
    /// loop:
    /// draft_tokens = draft.greedy_decode_K(prefix) # K serial draft steps
    /// v_logits = verifier.forward(draft_tokens, cache=Some(...)) # K logits, cache offset += K
    /// # v_logits[i] predicts after [prefix + d[..i+1]] (compares to d[i+1] for i<K-1)
    /// compare T_carry vs d[0]; v_logits[i-1] vs d[i] for i in 1..K # K comparisons
    /// accept = longest-matching-prefix
    /// if accept == K:
    /// emit d[0..K]; T_carry := argmax(v_logits[K-1])
    /// # cache already at L+K = correct; no truncation
    /// else:
    /// emit d[..accept] + correction(=T_carry if accept==0 else argmax(v_logits[accept-1]))
    /// truncate verifier cache to L+accept; feed correction (1-token forward) → new T_carry
    /// L = new prefix length
    /// ```
    ///
    /// The draft keeps its own persistent cache, rolled back alongside the
    /// verifier's on partial acceptance, so per-round draft cost is K decode
    /// steps at draft-model speed (≈10× verifier speed) rather than a
    /// re-prefill; for 31b+e2b that is ~1/4 of verifier cost in practice.
    ///
    /// `step_fn` is called once per emitted token (verifier-confirmed) so
    /// the SSE consumer can stream output.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn spec_generate_greedy(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        prompt_ids: &[u32],
        n_tokens: usize,
        k: usize,
        kv_quant_override: Option<KvQuant>,
        max_ctx_override: Option<i32>,
        prompt_cache_slots: usize,
        eos_ids: &[u32],
        step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
        // A6.2: speculative does not yet integrate the sampler constraint
        // engine. Per-round verifier argmax produces K+1 tokens in one
        // dispatch, but `ConstraintEngine::step_mask` returns one mask for
        // one position — there is no acceptance-aware mask threading yet.
        // Calls with `Some(_)` are rejected with `Error::Model` so route
        // handlers that mix `response_format` with speculative decoding fail
        // fast rather than silently ignoring the constraint. The standalone
        // arch path (`Architecture::generate_greedy`) handles the constraint
        // correctly; only the SpeculativeGenerator route is gated.
        constraint: Option<&mut dyn crate::ConstraintEngine>,
        // `temperature == 0` runs the greedy cached path (byte-identical
        // to before). `temperature > 0` runs Leviathan stochastic acceptance:
        // the draft samples from its post-sampling distribution `q`, the
        // verifier scores each position's post-sampling distribution `p`, and
        // each draft token is accepted with prob `min(1, p(x)/q(x))` vs a
        // uniform draw; on first reject a correction is sampled from the
        // residual `normalize((p−q)+)`. This preserves the verifier's output
        // distribution exactly (Leviathan 2023 Thm 1).
        sampler_cfg: &crate::sampler::SamplerConfig,
    ) -> Result<Vec<ProbeStep>> {
        if k == 0 {
            return Err(Error::Model("spec_generate_greedy: k must be >= 1".into()));
        }
        if n_tokens == 0 {
            return Ok(vec![]);
        }
        if prompt_ids.is_empty() {
            return Err(Error::Model(
                "spec_generate_greedy: empty prompt_ids".into(),
            ));
        }
        if constraint.is_some() {
            return Err(Error::Model(
                "spec_generate_greedy: A6.2 — sampler constraint engine not \
                 supported on the speculative-decoding path. Use the \
                 single-arch path (ArchGenerator) for response_format \
                 requests, or wait for A6.3."
                    .into(),
            ));
        }
        // Persistent verifier + draft KV caches with truncate_to rollback on
        // partial acceptance. There is no no-cache fallback — an architecture
        // whose `forward_seq_last_k_with_cache` is unwired surfaces
        // `Error::Model` from the cached path.
        let _ = prompt_cache_slots;
        if sampler_cfg.sampling_active() {
            // stochastic acceptance (temperature > 0).
            self.spec_generate_stochastic_cached(
                tokenizer,
                prompt_ids,
                n_tokens,
                k,
                kv_quant_override,
                max_ctx_override,
                eos_ids,
                step_fn,
                sampler_cfg,
            )
        } else {
            // Greedy (temperature == 0) — byte-identical to before .
            self.spec_generate_greedy_cached(
                tokenizer,
                prompt_ids,
                n_tokens,
                k,
                kv_quant_override,
                max_ctx_override,
                eos_ids,
                step_fn,
            )
        }
    }

    /// Greedy spec generation with persistent verifier + draft KV
    /// caches and `truncate_to`-based rollback.
    ///
    /// Algorithm (mirrors mlx-lm `speculative_generate_step`):
    ///
    /// ```text
    /// # one-time prefill on prompt[..-1]; carry-token y = prompt[-1]
    /// y = prompt[-1]
    /// loop:
    /// draft_tokens = K serial decode steps through draft cache
    /// v_tokens = verifier.forward([y, draft_tokens]) → K+1 logits
    /// accept = longest matching prefix of v_tokens vs draft_tokens
    /// emit v_tokens[..=accept] # accept matched + 1 correction
    /// y = v_tokens[accept] # next round's carry token
    /// verifier.cache.truncate_to(L + accept + 1) # drop K+1-(accept+1)=K-accept
    /// draft.cache.truncate_to(L + accept) # drop K-(accept+1)=K-accept-1
    /// if accept == K: feed draft_tokens[-1] before y on next round
    /// L = new prefix length
    /// ```
    ///
    /// Wires Gemma4 only (verifier + draft both Gemma4Text). An architecture
    /// whose `forward_seq_last_k_with_cache` is unwired returns `Error::Model`
    /// from here; there is no fallback path.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn spec_generate_greedy_cached(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        prompt_ids: &[u32],
        n_tokens: usize,
        k: usize,
        kv_quant_override: Option<KvQuant>,
        max_ctx_override: Option<i32>,
        eos_ids: &[u32],
        step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    ) -> Result<Vec<ProbeStep>> {
        let draft = self.draft_model()?;
        let device = self.device;
        let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

        if prompt_ids.len() < 2 {
            return Err(Error::Model(
                "spec_generate_greedy_cached: prompt must have ≥2 tokens".into(),
            ));
        }

        // Diagnostic counters.
        let mut total_draft_tokens: usize = 0;
        let mut total_accept_count: usize = 0;
        let mut rounds: usize = 0;
        let t_total = Instant::now();
        let mut window = DecodeWindow::new();
        let mut draft_ns: u128 = 0;
        let mut verifier_ns: u128 = 0;

        // Resolve KV quant — same value for verifier and draft. The drafter
        // stack resolves its own default, so it must read the same constant the
        // verifier does or a spec pair runs two different caches.
        let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
        // The verifier's limits bound the pair; an over-capacity `--max-ctx`
        // is refused here rather than overflowing a cache mid-round.
        let ctx = verifier_context(&self.verifier, max_ctx_override)?;
        let max_seq = ctx.ceiling;

        tracing::info!(
            k,
            prompt_len = prompt_ids.len(),
            n_tokens,
            ?kv_quant,
            max_seq,
            "spec_generate_greedy_cached: starting — persistent caches + truncate_to"
        );

        // --- Allocate per-layer caches for verifier and draft. ---------
        // A layer that reports a sliding window gets the RotatingKvCache port
        // whatever codec it is handed — the branch is `window > 0` alone
        // (`KvCache::with_quant_max_seq_window`), so an SWA layer here is bf16
        // at `sliding_window` tokens under every `kv_quant`, and only the
        // full-attention layers quantize. A block-verify write leaves that ring
        // holding its window plus the block, which is what lets the rollback
        // below drop the rejected tail out of it losslessly.
        let mut verifier_caches: Vec<KvCache> = (0..self.verifier.num_hidden_layers())
            .map(|i| {
                let window = self.verifier.layer_sliding_window(i);
                KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                    .with_max_seq_ceiling(ctx.ceiling)
                    .with_layer_idx(i)
                    // The stack decides whether its layers read each other's
                    // K/V, and so whether Mixed/RotK keep their bf16 mirror.
                    .with_shares_kv(self.verifier.shares_kv_across_layers())
            })
            .collect();
        let mut draft_caches: Vec<KvCache> = (0..draft.num_hidden_layers())
            .map(|i| {
                let window = draft.layer_sliding_window(i);
                KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                    .with_max_seq_ceiling(ctx.ceiling)
                    .with_layer_idx(i)
                    .with_shares_kv(draft.shares_kv_across_layers())
            })
            .collect();

        // --- Recurrent (GatedDeltaNet) caches for hybrid archs. --------
        // Only Qwen3.5MoE needs these; Gemma4 leaves them None and the
        // forward path ignores the parameter. The GDN recurrent state has
        // NO sequence axis, so spec rollback uses snapshot/restore (below)
        // rather than KvCache::truncate_to.
        let mut verifier_lin: Option<Vec<LinearAttnCache>> = if self.verifier.needs_lin_caches() {
            Some(
                (0..self.verifier.num_hidden_layers())
                    .map(|_| LinearAttnCache::new())
                    .collect(),
            )
        } else {
            None
        };
        let mut draft_lin: Option<Vec<LinearAttnCache>> = if draft.needs_lin_caches() {
            Some(
                (0..draft.num_hidden_layers())
                    .map(|_| LinearAttnCache::new())
                    .collect(),
            )
        } else {
            None
        };

        // --- Initial prefill on prompt[..-1] (mirrors mlx-lm _prefill). -
        // Last token becomes the carry-token `y` fed into round 1.
        let prefill_t0 = Instant::now();
        if prompt_ids.len() <= 1 {
            return Err(Error::Model(
                "spec_generate_greedy_cached: prompt too short".into(),
            ));
        }
        let prefill_slice = &prompt_ids[..prompt_ids.len() - 1];
        prefill_chunked(
            &self.verifier,
            prefill_slice,
            &mut verifier_caches,
            verifier_lin.as_deref_mut(),
            device,
        )?;
        prefill_chunked(
            draft,
            prefill_slice,
            &mut draft_caches,
            draft_lin.as_deref_mut(),
            device,
        )?;
        let prefill_ns: u128 = prefill_t0.elapsed().as_nanos();

        // Carry-tokens: per mlx-lm reference, verifier and draft each
        // need their own "input seed" for the next round. They diverge
        // when all draft tokens are accepted: the verifier consumed
        // d1..dK in the K+1 forward, but the draft cache stopped one
        // step earlier (only consumed d1..d_{K-1}). To resync, the next
        // draft round must feed [dK, correction] (2 tokens) before
        // generating new drafts; the verifier still feeds just
        // [correction] as its carry.
        let last_prompt = *prompt_ids.last().unwrap();
        let mut v_carry: Vec<u32> = vec![last_prompt];
        let mut d_seed: Vec<u32> = vec![last_prompt];

        // --- Spec loop. ------------------------------------------------
        let seed_emitted = emitted.len();
        let mut emitted_in_rounds = 0usize;
        let round_loop_t0 = Instant::now();
        while emitted.len() < n_tokens {
            rounds += 1;
            let remaining = n_tokens - emitted.len();
            // Mirror mlx-lm: num_draft = min(remaining, K). Always ≥ 1
            // since loop guard ensures `remaining ≥ 1`.
            let num_draft = remaining.min(k).max(1);

            // -- GDN rollback prep. ------------------------------------
            // The GatedDeltaNet recurrent state has NO sequence axis, so
            // `KvCache::truncate_to` cannot roll it back to an intermediate
            // position on partial acceptance. We snapshot the pre-round GDN
            // state for both models here. On partial acceptance the state is
            // restored from the snapshot and then re-advanced ("replay") over
            // the kept tokens with a single forward, leaving the GDN state
            // exactly consistent with the truncated KvCache. On full
            // acceptance no rollback is needed (state is already correct).
            // Snapshots are deep clones of the small fixed-shape conv/delta
            // tensors — cheap relative to a verifier forward.
            let verifier_lin_snap = snapshot_lin(verifier_lin.as_deref())?;
            let draft_lin_snap = snapshot_lin(draft_lin.as_deref())?;

            // -- Phase A: draft generates `num_draft` tokens via cache. -
            let t0 = Instant::now();
            let draft_tokens = draft_decode_n(
                draft,
                &d_seed,
                num_draft,
                &mut draft_caches,
                draft_lin.as_deref_mut(),
                device,
            )?;
            draft_ns += t0.elapsed().as_nanos();
            total_draft_tokens += draft_tokens.len();

            // -- Phase B: verifier scores K+1 logits in one cached call.
            // Input = v_carry + draft_tokens. v_carry is 1 token: either
            // the last prompt token (round 1) or the previous round's
            // emitted correction/bonus.
            let mut v_input: Vec<u32> = Vec::with_capacity(v_carry.len() + draft_tokens.len());
            v_input.extend_from_slice(&v_carry);
            v_input.extend_from_slice(&draft_tokens);
            let v_k = v_input.len(); // = num_draft + 1
            if v_k < 2 {
                return Err(Error::Model(format!(
                    "spec_generate_greedy_cached: v_k={v_k} too small"
                )));
            }

            let t0 = Instant::now();
            // Hybrid verifier (Qwen3.5MoE) advances its GDN lin caches here;
            // Gemma4 passes None.
            let v_logits = self.verifier.forward_seq_last_k_with_cache(
                &v_input,
                v_k,
                &mut verifier_caches,
                verifier_lin.as_deref_mut(),
                device,
            )?;
            let v_argmax = argmax(&v_logits, -1, device)?;
            v_argmax.eval()?;
            let bytes = v_argmax.to_bytes()?;
            verifier_ns += t0.elapsed().as_nanos();

            let v_tokens = argmax_tokens(&bytes, v_k)?;

            // -- Phase C: greedy acceptance. ---------------------------
            // v_tokens[i] is the verifier's prediction after the i-th
            // input token (positions 0..K). Compare v_tokens[0..num_draft]
            // against draft_tokens[0..num_draft]. Longest matching prefix
            // → emit accept tokens; emit v_tokens[accept] as correction
            // (or bonus when accept == num_draft).
            let mut accept = 0usize;
            for i in 0..draft_tokens.len() {
                if v_tokens[i] == draft_tokens[i] {
                    accept += 1;
                } else {
                    break;
                }
            }
            total_accept_count += accept;

            // Emit accept + 1 tokens: v_tokens[0..=accept].
            let to_emit = (accept + 1).min(v_tokens.len());
            let mut hit_eos = false;
            for &id in v_tokens.iter().take(to_emit) {
                if emitted.len() >= n_tokens {
                    break;
                }
                emit_step(tokenizer, id, step_fn, &mut emitted, &mut window);
                emitted_in_rounds += 1;
                if eos_ids.contains(&id) {
                    hit_eos = true;
                    break;
                }
            }
            if hit_eos {
                RoundStats {
                    loop_kind: SpecLoop::TwoModelGreedy,
                    block_size: k + 1,
                    rounds,
                    emitted: emitted.len(),
                    seed_emitted,
                    emitted_in_rounds,
                    total_draft: total_draft_tokens,
                    total_accept: total_accept_count,
                    prefill_ns,
                    draft_ns,
                    verifier_ns,
                    round_loop_ns: round_loop_t0.elapsed().as_nanos(),
                    elapsed_ns: t_total.elapsed().as_nanos(),
                    decode_tps: window.tps(),
                    charged: false,
                }
                .log_done();
                return Ok(emitted);
            }

            // -- Phase D: setup next round. ----------------------------
            // y = correction token (or bonus). It's already in v_tokens[accept].
            let next_y_token = v_tokens[accept];

            // Verifier cache: it processed `v_k = num_draft + 1` tokens
            // ending at logical position L+v_k. We accepted (accept+1)
            // emitted tokens, so its valid prefix length is L+accept+1
            // (the last emitted = correction = v_tokens[accept], which is
            // a *prediction* — verifier hasn't actually processed it
            // yet). Trim by (v_k - (accept+1)) = num_draft - accept.
            // `max()` and not `[0]`: on a GDN hybrid the recurrent layers'
            // KvCache never advances, so layer 0 may sit at 0 while the
            // full-attention layers carry the round.
            let v_offset_before = verifier_caches
                .iter()
                .map(KvCache::offset)
                .max()
                .unwrap_or(0);
            let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);
            // On a PARTIAL accept the KV keeps `v_target` positions and the GDN
            // recurrent state — which advanced by `v_k` and cannot be sliced —
            // is rebuilt from the pre-round snapshot by replaying the retained
            // prefix through the real caches. On a FULL accept nothing was
            // dropped and the snapshot is discarded.
            if v_target < v_offset_before {
                rollback_round_caches(
                    &self.verifier,
                    &mut verifier_caches,
                    verifier_lin.as_deref_mut(),
                    verifier_lin_snap,
                    &v_input,
                    v_offset_before - v_k as i32,
                    v_target,
                    // This loop times no phases, so it never charges one.
                    false,
                    device,
                )?;
            } else {
                drop(verifier_lin_snap);
            }

            // Draft cache: it processed num_draft tokens (1 carry + K-1
            // intermediates each producing the next, total cache advance
            // = num_draft). Need to keep accept of those + the carry. So
            // truncate to L_initial + 1 + accept = original_offset_before
            // - num_draft + accept + 1. Per mlx-lm:
            // trim_prompt_cache(draft_cache, max(num_draft - accept - 1, 0))
            let d_offset_before = draft_caches.iter().map(KvCache::offset).max().unwrap_or(0);
            let d_drop = (draft_tokens.len() as i32 - accept as i32 - 1).max(0);
            let d_target = d_offset_before - d_drop;
            // `draft_decode_n` fed `d_seed ++ draft_tokens[..num_draft-1]`
            // (each step's input is the prior step's output; the last output is
            // never fed back), so that is the token sequence the rollback
            // replays the retained prefix of.
            if d_target < d_offset_before {
                let mut d_fed: Vec<u32> = Vec::with_capacity(d_seed.len() + draft_tokens.len());
                d_fed.extend_from_slice(&d_seed);
                if draft_tokens.len() > 1 {
                    d_fed.extend_from_slice(&draft_tokens[..draft_tokens.len() - 1]);
                }
                let d_pre_round_offset = d_offset_before - d_fed.len() as i32;
                rollback_round_caches(
                    draft,
                    &mut draft_caches,
                    draft_lin.as_deref_mut(),
                    draft_lin_snap,
                    &d_fed,
                    d_pre_round_offset,
                    d_target,
                    // This loop times no phases, so it never charges one.
                    false,
                    device,
                )?;
            } else {
                drop(draft_lin_snap);
            }

            // Setup next round's carry tokens. Verifier carry is always
            // 1 token (= correction or bonus). Draft seed prepends the
            // last draft token when all-accepted, since the draft cache
            // hasn't yet consumed it.
            v_carry = vec![next_y_token];
            if accept == draft_tokens.len() {
                let last_draft = *draft_tokens.last().unwrap();
                d_seed = vec![last_draft, next_y_token];
            } else {
                d_seed = vec![next_y_token];
            }

            tracing::debug!(
                round = rounds,
                accept,
                num_draft = draft_tokens.len(),
                emitted_round = to_emit,
                emitted_total = emitted.len(),
                v_offset_before,
                v_target,
                d_offset_before,
                d_target,
                "spec round (cached)"
            );
        }

        RoundStats {
            loop_kind: SpecLoop::TwoModelGreedy,
            block_size: k + 1,
            rounds,
            emitted: emitted.len(),
            seed_emitted,
            emitted_in_rounds,
            total_draft: total_draft_tokens,
            total_accept: total_accept_count,
            prefill_ns,
            draft_ns,
            verifier_ns,
            round_loop_ns: round_loop_t0.elapsed().as_nanos(),
            elapsed_ns: t_total.elapsed().as_nanos(),
            decode_tps: window.tps(),
            charged: false,
        }
        .log_done();

        // Report the verifier's resident KV, so a caller that sampled the
        // verifier arch around this call can attribute the figure to it. This
        // path never goes through `Architecture::generate_greedy`, so nothing
        // else writes it.
        self.verifier.store_kv_cache_bytes(
            verifier_kv_bytes(&verifier_caches, verifier_lin.as_deref()),
            crate::decode_loop::PostDecode::seal(),
        );

        Ok(emitted)
    }

    /// Stochastic speculative decoding for `temperature > 0`.
    ///
    /// Identical cache structure / rollback to `spec_generate_greedy_cached`,
    /// but acceptance is the Leviathan (2023, §2.3) stochastic rule instead of
    /// argmax-prefix matching:
    ///
    /// ```text
    /// loop:
    /// draft proposes num_draft tokens; for each, record its post-sampling
    /// distribution q_i and sample x_i ~ q_i
    /// verifier scores num_draft+1 positions → post-sampling p_0..p_{num_draft}
    /// for i in 0..num_draft:
    /// accept x_i with prob min(1, p_i(x_i)/q_i(x_i)) vs Uniform[0,1]
    /// on first reject: emit corr ~ normalize((p_i − q_i)+); stop round
    /// if all accepted: emit a bonus token ~ p_{num_draft}
    /// ```
    ///
    /// `p` and `q` are built with [`crate::sampler::sampling_distribution`] so
    /// they are the SAME post-temperature / post-top-p / post-top-k / post-min-p
    /// distributions the host sampler uses — a hard correctness requirement
    /// (mismatched p/q biases the output; see Leviathan Thm 1).
    ///
    /// The per-request `Pcg32` is seeded from `sampler_cfg.seed_or_default()`
    /// so draws are reproducible (tests rely on this).
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn spec_generate_stochastic_cached(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        prompt_ids: &[u32],
        n_tokens: usize,
        k: usize,
        kv_quant_override: Option<KvQuant>,
        max_ctx_override: Option<i32>,
        eos_ids: &[u32],
        step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
        sampler_cfg: &crate::sampler::SamplerConfig,
    ) -> Result<Vec<ProbeStep>> {
        use crate::sampler::{
            sample_index, sampling_distribution, stochastic_accept, AcceptDecision, Pcg32,
        };

        let draft = self.draft_model()?;
        let device = self.device;
        let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

        if prompt_ids.len() < 2 {
            return Err(Error::Model(
                "spec_generate_stochastic_cached: prompt must have ≥2 tokens".into(),
            ));
        }

        // No penalties / constraint on the spec path (rejected upstream); the
        // distribution builder still needs a no-op config + empty window.
        let penalty_cfg = crate::sampler::PenaltyConfig::default();
        let recent: &[u32] = &[];
        // One RNG threaded through the whole generation so the draw stream is
        // contiguous and reproducible (draft samples + accept tests + residual
        // resamples all advance it in a fixed order).
        let mut rng = Pcg32::new(sampler_cfg.seed_or_default());

        // Diagnostic counters.
        let mut total_draft_tokens: usize = 0;
        let mut total_accept_count: usize = 0;
        let mut rounds: usize = 0;
        let t_total = Instant::now();
        let mut window = DecodeWindow::new();
        let mut draft_ns: u128 = 0;
        let mut verifier_ns: u128 = 0;

        // Same constant the verifier resolves — a spec pair must not run two
        // different caches.
        let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
        let ctx = verifier_context(&self.verifier, max_ctx_override)?;
        let max_seq = ctx.ceiling;

        tracing::info!(
            k,
            prompt_len = prompt_ids.len(),
            n_tokens,
            ?kv_quant,
            max_seq,
            temperature = sampler_cfg.temperature,
            top_p = sampler_cfg.top_p,
            top_k = sampler_cfg.top_k,
            seed = sampler_cfg.seed_or_default(),
            "spec_generate_stochastic_cached: starting (Leviathan stochastic acceptance)"
        );

        let mut verifier_caches: Vec<KvCache> = (0..self.verifier.num_hidden_layers())
            .map(|i| {
                let window = self.verifier.layer_sliding_window(i);
                KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                    .with_max_seq_ceiling(ctx.ceiling)
                    .with_layer_idx(i)
                    // The stack decides whether its layers read each other's
                    // K/V, and so whether Mixed/RotK keep their bf16 mirror.
                    .with_shares_kv(self.verifier.shares_kv_across_layers())
            })
            .collect();
        let mut draft_caches: Vec<KvCache> = (0..draft.num_hidden_layers())
            .map(|i| {
                let window = draft.layer_sliding_window(i);
                KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                    .with_max_seq_ceiling(ctx.ceiling)
                    .with_layer_idx(i)
                    .with_shares_kv(draft.shares_kv_across_layers())
            })
            .collect();

        let mut verifier_lin: Option<Vec<LinearAttnCache>> = if self.verifier.needs_lin_caches() {
            Some(
                (0..self.verifier.num_hidden_layers())
                    .map(|_| LinearAttnCache::new())
                    .collect(),
            )
        } else {
            None
        };
        let mut draft_lin: Option<Vec<LinearAttnCache>> = if draft.needs_lin_caches() {
            Some(
                (0..draft.num_hidden_layers())
                    .map(|_| LinearAttnCache::new())
                    .collect(),
            )
        } else {
            None
        };

        // Initial prefill on prompt[..-1]; last prompt token is round 1's carry.
        let prefill_t0 = Instant::now();
        let prefill_slice = &prompt_ids[..prompt_ids.len() - 1];
        prefill_chunked(
            &self.verifier,
            prefill_slice,
            &mut verifier_caches,
            verifier_lin.as_deref_mut(),
            device,
        )?;
        prefill_chunked(
            draft,
            prefill_slice,
            &mut draft_caches,
            draft_lin.as_deref_mut(),
            device,
        )?;
        let prefill_ns: u128 = prefill_t0.elapsed().as_nanos();

        let last_prompt = *prompt_ids.last().unwrap();
        let mut v_carry: Vec<u32> = vec![last_prompt];
        let mut d_seed: Vec<u32> = vec![last_prompt];

        let seed_emitted = emitted.len();
        let mut emitted_in_rounds = 0usize;
        let round_loop_t0 = Instant::now();
        while emitted.len() < n_tokens {
            rounds += 1;
            let remaining = n_tokens - emitted.len();
            let num_draft = remaining.min(k).max(1);

            let verifier_lin_snap = snapshot_lin(verifier_lin.as_deref())?;
            let draft_lin_snap = snapshot_lin(draft_lin.as_deref())?;

            // -- Phase A: draft samples `num_draft` tokens, recording q_i. ---
            let t0 = Instant::now();
            let (draft_tokens, draft_q) = draft_decode_n_stochastic(
                draft,
                &d_seed,
                num_draft,
                &mut draft_caches,
                draft_lin.as_deref_mut(),
                sampler_cfg,
                &penalty_cfg,
                recent,
                &mut rng,
                device,
            )?;
            draft_ns += t0.elapsed().as_nanos();
            total_draft_tokens += draft_tokens.len();

            // -- Phase B: verifier scores num_draft+1 positions. -------------
            let mut v_input: Vec<u32> = Vec::with_capacity(v_carry.len() + draft_tokens.len());
            v_input.extend_from_slice(&v_carry);
            v_input.extend_from_slice(&draft_tokens);
            let v_k = v_input.len();
            if v_k < 2 {
                return Err(Error::Model(format!(
                    "spec_generate_stochastic_cached: v_k={v_k} too small"
                )));
            }

            let t0 = Instant::now();
            let v_logits = self.verifier.forward_seq_last_k_with_cache(
                &v_input,
                v_k,
                &mut verifier_caches,
                verifier_lin.as_deref_mut(),
                device,
            )?;
            // v_logits: [1, v_k, vocab]. Build p_i for each of the v_k
            // positions via the SAME post-sampling pipeline as q.
            let vocab = self.vocab_size() as i32;
            let mut p_dists: Vec<Vec<f32>> = Vec::with_capacity(v_k);
            for i in 0..v_k {
                // Slice position i → [1, 1, vocab] → reshape [1, vocab].
                let row = v_logits.slice(
                    &[0, i as i32, 0],
                    &[1, i as i32 + 1, vocab],
                    &[1, 1, 1],
                    device,
                )?;
                let row = row.reshape(&[1, vocab], device)?;
                p_dists.push(sampling_distribution(
                    &row,
                    sampler_cfg,
                    None,
                    &penalty_cfg,
                    recent,
                )?);
            }
            verifier_ns += t0.elapsed().as_nanos();

            // -- Phase C: Leviathan stochastic acceptance. -------------------
            // p_dists[i] is the verifier's distribution AT position i (predicts
            // the token after v_input[i]); compare against draft x_i = the
            // draft token proposed at that position, drawn from q_i.
            let mut accept = 0usize;
            let mut correction: Option<u32> = None;
            for i in 0..draft_tokens.len() {
                let x = draft_tokens[i];
                match stochastic_accept(&p_dists[i], &draft_q[i], x, &mut rng)? {
                    AcceptDecision::Accept => {
                        accept += 1;
                    }
                    AcceptDecision::Reject(corr) => {
                        correction = Some(corr);
                        break;
                    }
                }
            }
            total_accept_count += accept;

            // Determine the emitted tokens this round: accepted draft prefix
            // plus one extra (correction on reject, bonus from p_{num_draft}
            // on full accept).
            let mut round_tokens: Vec<u32> = Vec::with_capacity(accept + 1);
            round_tokens.extend_from_slice(&draft_tokens[..accept]);
            let extra = match correction {
                Some(corr) => corr,
                None => {
                    // All accepted ⇒ bonus token from the verifier's last
                    // distribution p_{num_draft} (index v_k - 1).
                    sample_index(&p_dists[v_k - 1], &mut rng) as u32
                }
            };
            round_tokens.push(extra);

            let mut hit_eos = false;
            for &id in &round_tokens {
                if emitted.len() >= n_tokens {
                    break;
                }
                emit_step(tokenizer, id, step_fn, &mut emitted, &mut window);
                emitted_in_rounds += 1;
                if eos_ids.contains(&id) {
                    hit_eos = true;
                    break;
                }
            }
            if hit_eos {
                RoundStats {
                    loop_kind: SpecLoop::TwoModelStochastic,
                    block_size: k + 1,
                    rounds,
                    emitted: emitted.len(),
                    seed_emitted,
                    emitted_in_rounds,
                    total_draft: total_draft_tokens,
                    total_accept: total_accept_count,
                    prefill_ns,
                    draft_ns,
                    verifier_ns,
                    round_loop_ns: round_loop_t0.elapsed().as_nanos(),
                    elapsed_ns: t_total.elapsed().as_nanos(),
                    decode_tps: window.tps(),
                    charged: false,
                }
                .log_done();
                return Ok(emitted);
            }

            // -- Phase D: cache rollback (identical to the greedy path). -----
            // Verifier processed v_k positions; we keep accept+1 (the accepted
            // prefix + the extra, which the verifier has NOT yet processed as
            // input — `extra` is a prediction). Trim v_k - (accept+1).
            let next_y_token = extra;
            let v_offset_before = verifier_caches
                .iter()
                .map(KvCache::offset)
                .max()
                .unwrap_or(0);
            let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);
            if v_target < v_offset_before {
                rollback_round_caches(
                    &self.verifier,
                    &mut verifier_caches,
                    verifier_lin.as_deref_mut(),
                    verifier_lin_snap,
                    &v_input,
                    v_offset_before - v_k as i32,
                    v_target,
                    // This loop times no phases, so it never charges one.
                    false,
                    device,
                )?;
            } else {
                drop(verifier_lin_snap);
            }

            let d_offset_before = draft_caches.iter().map(KvCache::offset).max().unwrap_or(0);
            let d_drop = (draft_tokens.len() as i32 - accept as i32 - 1).max(0);
            let d_target = d_offset_before - d_drop;
            if d_target < d_offset_before {
                let mut d_fed: Vec<u32> = Vec::with_capacity(d_seed.len() + draft_tokens.len());
                d_fed.extend_from_slice(&d_seed);
                if draft_tokens.len() > 1 {
                    d_fed.extend_from_slice(&draft_tokens[..draft_tokens.len() - 1]);
                }
                let d_pre_round_offset = d_offset_before - d_fed.len() as i32;
                rollback_round_caches(
                    draft,
                    &mut draft_caches,
                    draft_lin.as_deref_mut(),
                    draft_lin_snap,
                    &d_fed,
                    d_pre_round_offset,
                    d_target,
                    // This loop times no phases, so it never charges one.
                    false,
                    device,
                )?;
            } else {
                drop(draft_lin_snap);
            }

            v_carry = vec![next_y_token];
            if accept == draft_tokens.len() {
                let last_draft = *draft_tokens.last().unwrap();
                d_seed = vec![last_draft, next_y_token];
            } else {
                d_seed = vec![next_y_token];
            }

            tracing::debug!(
                round = rounds,
                accept,
                num_draft = draft_tokens.len(),
                rejected = correction.is_some(),
                emitted_total = emitted.len(),
                v_offset_before,
                v_target,
                d_offset_before,
                d_target,
                "spec round (stochastic)"
            );
        }

        RoundStats {
            loop_kind: SpecLoop::TwoModelStochastic,
            block_size: k + 1,
            rounds,
            emitted: emitted.len(),
            seed_emitted,
            emitted_in_rounds,
            total_draft: total_draft_tokens,
            total_accept: total_accept_count,
            prefill_ns,
            draft_ns,
            verifier_ns,
            round_loop_ns: round_loop_t0.elapsed().as_nanos(),
            elapsed_ns: t_total.elapsed().as_nanos(),
            decode_tps: window.tps(),
            charged: false,
        }
        .log_done();

        // See the greedy path: the verifier's own resident KV, reported so the
        // caller can attribute it to this call.
        self.verifier.store_kv_cache_bytes(
            verifier_kv_bytes(&verifier_caches, verifier_lin.as_deref()),
            crate::decode_loop::PostDecode::seal(),
        );

        Ok(emitted)
    }
}

// ---------------------------------------------------------------------------
// Cached round-loop helpers
// ---------------------------------------------------------------------------

/// The wall-clock window a round loop spends decoding, first emitted token to
/// last.
///
/// A round loop's total elapsed time also covers prompt prefill, so
/// `emitted / elapsed` shrinks as the prompt grows and cannot be compared with
/// the non-speculative `decode_tps` that `rmlx baseline` records. This measures
/// the same window that one does — `(marks - 1) / (last - first)`.
///
/// The window counts its own marks rather than trusting a caller-supplied
/// token total: the two can only agree if every emitted token went through
/// [`emit_step`], and a count passed in from outside would let a loop that
/// emits without marking report a rate faster than it ran.
///
/// One divergence from `rmlx baseline` remains, and it is deliberate: where
/// that path falls back to an overall (prefill-inclusive) rate when it has
/// fewer than two tokens to work with, this returns `None`. There is no
/// second rate here that would be honest to substitute.
#[derive(Debug, Default)]
pub(crate) struct DecodeWindow {
    first: Option<Instant>,
    last: Option<Instant>,
    marks: usize,
}

impl DecodeWindow {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record a token about to be handed to the sink.
    ///
    /// Both endpoints are taken *before* the `step_fn` callback, so the sink
    /// cost of tokens `1..N-1` (tokenizer decode, think-splitter, channel
    /// backpressure) falls inside the window and the Nth token's does not —
    /// the same convention as the first/last callback stamps `rmlx baseline`
    /// measures between.
    fn mark(&mut self) {
        self.mark_at(Instant::now());
    }

    /// [`Self::mark`] at a caller-supplied instant, so a test can drive the
    /// window without sleeping.
    fn mark_at(&mut self, at: Instant) {
        self.first.get_or_insert(at);
        self.last = Some(at);
        self.marks += 1;
    }

    /// How many tokens this window has seen. Test accessor: production code
    /// reads the count only through [`Self::tps`].
    #[cfg(test)]
    fn marks(&self) -> usize {
        self.marks
    }

    /// Tokens per second over the window, or `None` when fewer than two tokens
    /// were emitted and there is no interval to measure.
    ///
    /// `None` rather than `0.0`: a zero in this slot prints, averages and wins
    /// a champion cell exactly like a real throughput of zero.
    pub(crate) fn tps(&self) -> Option<f64> {
        let (Some(first), Some(last)) = (self.first, self.last) else {
            return None;
        };
        let secs = last.duration_since(first).as_secs_f64();
        (self.marks >= 2 && secs > 0.0).then(|| ((self.marks - 1) as f64) / secs)
    }
}

/// Emit a single token through `step_fn` + the running `emitted` buffer.
///
/// Every speculative round loop emits through here, which is what keeps
/// [`DecodeWindow::tps`] honest — a loop that pushed to `emitted` directly
/// would leave the window short and the rate wrong.
///
/// The `Option<u32>` force-next signal `step_fn` may return is **discarded**:
/// the ordinary decode loop folds it into `forced_next` to close an
/// over-budget thinking block, so that force-close is inert on every
/// speculative path.
pub(crate) fn emit_step(
    tokenizer: &tokenizers::Tokenizer,
    id: u32,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    emitted: &mut Vec<ProbeStep>,
    window: &mut DecodeWindow,
) {
    let piece = tokenizer
        .id_to_token(id)
        .unwrap_or_else(|| format!("<unk:{id}>"));
    let step = ProbeStep {
        token_id: id,
        piece: piece.into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    };
    window.mark();
    step_fn(&step);
    emitted.push(step);
}

/// Chunked prefill of `tokens` into `caches`, mirroring the gemma4 generate
/// path (enter_prefill / exit_prefill brackets, per-arch chunk size).
///
/// Used at the top of `spec_generate_greedy_cached` once per model.
/// `pub` for the MTP drafter-alignment integration test (it replays
/// round-0 prefill explicitly to assert drafter↔verifier first-token match).
pub fn prefill_chunked(
    arch: &Architecture,
    tokens: &[u32],
    caches: &mut [KvCache],
    mut lin_caches: Option<&mut [LinearAttnCache]>,
    device: Device,
) -> Result<()> {
    prefill_chunked_for_class(
        arch.arch_class(),
        tokens,
        caches,
        device,
        |chunk, caches| {
            // Single-position last_k=1 forward — we only need cache update, not
            // logits. The lazy graph drops the lm_head matmul on non-final chunks;
            // on the final chunk we discard the returned Array. For GDN-bearing
            // archs the recurrent lin_caches advance alongside kv_caches; Gemma4
            // passes None. `as_deref_mut` reborrows per chunk.
            arch.forward_seq_last_k_with_cache(chunk, 1, caches, lin_caches.as_deref_mut(), device)
                .map(|_| ())
        },
    )
}

/// [`prefill_chunked`] with the architecture reduced to its class name.
///
/// This is where the chunk is chosen, and it is separate from
/// `prefill_chunked` so a test can drive the choice with an injected forward
/// and read back the slices the prompt was actually cut into — building an
/// `Architecture` needs a snapshot, so a chunk selected inside
/// `prefill_chunked` would be observable only on a machine with the weights.
fn prefill_chunked_for_class(
    arch_class: &str,
    tokens: &[u32],
    caches: &mut [KvCache],
    device: Device,
    forward: impl FnMut(&[u32], &mut [KvCache]) -> Result<()>,
) -> Result<()> {
    let (chunk_size, chunk_source) =
        crate::prefill_chunk::resolve(crate::prefill_chunk::module_key_for_class(arch_class));
    tracing::debug!(
        arch = arch_class,
        prefill_chunk = chunk_size,
        prefill_chunk_source = chunk_source,
        prompt_len = tokens.len(),
        n_chunks = tokens.len().div_ceil(chunk_size.max(1)),
        "prefill: chunking prompt"
    );
    prefill_chunked_with(tokens, caches, chunk_size, device, forward)
}

/// Bracket-and-sweep engine behind [`prefill_chunked`]: `enter_prefill` on every
/// cache, run `forward` per chunk, then `exit_prefill` on every cache.
///
/// The `exit_prefill` sweep is **mandatory** and runs on the failure path too.
/// On a chunk forward / eval failure this captures the first cause, breaks the
/// chunk loop, then sweeps `exit_prefill` over **all** caches unconditionally
/// (no early `?`, no `break` that skips a cache) before returning that first
/// cause. A cache left mid-prefill keeps un-finalized state (no decode seed /
/// un-quantized storage), and the next decode on it errors or corrupts KV — so
/// stranding even one cache poisons any later reuse of this slice.
///
/// The forward is injected so the invariant is unit-testable without a live
/// model: a test drives a failing forward and asserts no cache is left
/// `in_prefill`.
fn prefill_chunked_with(
    tokens: &[u32],
    caches: &mut [KvCache],
    prefill_chunk: usize,
    device: Device,
    mut forward: impl FnMut(&[u32], &mut [KvCache]) -> Result<()>,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    for c in caches.iter_mut() {
        c.enter_prefill();
    }
    let mut first_err: Option<Error> = None;
    let n_chunks = tokens.len().div_ceil(prefill_chunk);
    'chunks: for (chunk_idx, chunk) in tokens.chunks(prefill_chunk).enumerate() {
        let is_last = chunk_idx + 1 == n_chunks;
        if let Err(e) = forward(chunk, caches) {
            tracing::error!(error = %e, "spec prefill chunk forward failed, aborting generation");
            first_err = Some(e);
            break 'chunks;
        }
        // Flush command buffer between chunks via cache eval.
        if !is_last {
            for c in caches.iter() {
                if let Err(e) = c.eval_prefill_state() {
                    tracing::error!(error = %e, "spec prefill chunk cache eval failed, aborting generation");
                    first_err = Some(e);
                    break 'chunks;
                }
            }
        }
    }
    // Mandatory cleanup: every cache entered prefill above and must run
    // exit_prefill, even after a failure — no break, no early `?`. Skipping it
    // strands the remaining caches with un-finalized prefill state that
    // corrupts any later reuse of this slice. The first cause wins; a secondary
    // exit failure is logged (so it does not vanish) but does not overwrite it.
    for c in caches.iter_mut() {
        if let Err(e) = c.exit_prefill(device) {
            tracing::error!(error = %e, "spec prefill: exit_prefill failed during cleanup sweep");
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(())
}

/// Deep-clone snapshot of a per-layer GDN recurrent-state slice, if present.
///
/// Returns `None` for FullAttention-only archs (no `lin_caches`). The clone
/// is a fixed-shape conv/delta tensor pair per layer — cheap relative to a
/// forward. Used by the speculative loop to capture the pre-round GDN state
/// before draft + verifier forwards, so partial-acceptance rollback can
/// restore + replay (the recurrent state has no sequence axis to truncate).
fn snapshot_lin(lin: Option<&[LinearAttnCache]>) -> Result<Option<Vec<LinearAttnCache>>> {
    match lin {
        None => Ok(None),
        Some(caches) => {
            let mut snap = Vec::with_capacity(caches.len());
            for c in caches {
                snap.push(c.snapshot()?);
            }
            Ok(Some(snap))
        }
    }
}

/// Read a verify forward's `argmax` result back as `k` token ids.
///
/// The buffer is checked once, against the position count the caller verified,
/// before any of it is read. A round loop does this every round, so an
/// unguarded index here is a per-round panic on an invariant no type carries:
/// the argmax comes back from the device, and "the device returned fewer bytes
/// than the block has positions" is a state to name, not to abort on.
///
/// Extra trailing bytes are not an error — `k` is what the caller verified and
/// what it walks.
pub(crate) fn argmax_tokens(bytes: &[u8], k: usize) -> Result<Vec<u32>> {
    let want = k * 4;
    if bytes.len() < want {
        return Err(Error::Model(format!(
            "argmax_tokens: the verifier's argmax came back as {} bytes for {k} verified \
             positions, which needs {want}",
            bytes.len()
        )));
    }
    #[allow(
        clippy::indexing_slicing,
        reason = "chunks_exact(4) yields slices of exactly 4, so these four indices are \
                  in bounds by the iterator's own contract"
    )]
    Ok(bytes
        .chunks_exact(4)
        .take(k)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The greedy acceptance walk over one verified block.
///
/// `verifier_tokens[i]` is the verifier's own greedy continuation after
/// position `i` of the verify input, so the draft proposed for that position is
/// accepted exactly when the two agree. The walk stops at the first
/// disagreement.
///
/// Returns the number of accepted proposals and the tokens to emit: the agreed
/// prefix followed by one token the verifier stands behind — its correction at
/// the disagreement, or, when every proposal held, its bonus token past the last
/// draft. `budget` caps the emission, not the acceptance: a round that runs out
/// of token budget still committed the KV it committed, and reporting fewer
/// accepts than the caches hold is how the two disagree.
///
/// `verifier_tokens` carries one position more than `draft_tokens` — the bonus
/// slot. A shorter one simply ends the walk early, which is the same answer as
/// disagreeing there.
pub(crate) fn accept_prefix(
    verifier_tokens: &[u32],
    draft_tokens: &[u32],
    budget: usize,
) -> (usize, Vec<u32>) {
    let mut accepted = 0usize;
    let mut emit: Vec<u32> = Vec::with_capacity(verifier_tokens.len());
    for (pos, &token) in verifier_tokens.iter().enumerate() {
        let agreed = draft_tokens.get(pos) == Some(&token);
        if agreed {
            accepted += 1;
        }
        if emit.len() < budget {
            emit.push(token);
        }
        if !agreed {
            break;
        }
    }
    (accepted, emit)
}

/// Roll one speculative round's caches back to `target_offset` after a partial
/// acceptance — both the full-attention `kv` stack and, when the arch has one,
/// the GDN recurrent state in `lin`.
///
/// `pre_round_offset` is the KV offset before this round's verify forward ran;
/// `round_tokens` are the tokens that forward consumed, in order, so that
/// `round_tokens[..target_offset - pre_round_offset]` is exactly the retained
/// prefix.
///
/// **Full-attention arch** (`lin` empty or absent): every layer's KvCache
/// carries the whole round, so dropping the rejected tail is the entire
/// rollback — `kv` is truncated straight to `target_offset`.
///
/// **GDN hybrid**: the recurrent state has no sequence axis (see
/// `LinearAttnCache::truncate_to`), so it cannot be sliced to an intermediate
/// position. It is restored from the pre-round `snapshot` and replayed over the
/// retained prefix instead. That replay runs the WHOLE layer stack, and in this
/// hybrid the full-attention layers are interleaved between the GDN layers:
/// their output is the residual a later GDN layer consumes. So the replay must
/// see the real KV caches at their real sequence offsets — replaying through a
/// fresh scratch stack makes those FA layers attend a `kept`-token prefix at
/// positions `0..kept`, and every downstream GDN layer then advances on a wrong
/// hidden. The rollback therefore truncates `kv` to `pre_round_offset` and
/// replays into it; the caches land on `target_offset` exactly as the direct
/// truncation would have, and the GDN state lands byte-consistent with them.
///
/// Truncation is guarded by `offset() >= n` because a GDN layer's KvCache never
/// advances (it stays at 0); truncating it to a positive `n` would leave it
/// reporting positions it does not hold.
///
/// Call this only when the round actually dropped positions
/// (`target_offset < offset_before`); on a full accept there is nothing to roll
/// back and the snapshot is simply dropped.
///
/// `charge` is the calling loop's per-request answer from
/// [`phases_charged`], not a decision this function makes. Six loops share it
/// and two of them time their phases; reading the switch here would change the
/// schedule of the other four with nothing on their records saying so.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn rollback_round_caches(
    arch: &Architecture,
    kv: &mut [KvCache],
    lin: Option<&mut [LinearAttnCache]>,
    snapshot: Option<Vec<LinearAttnCache>>,
    round_tokens: &[u32],
    pre_round_offset: i32,
    target_offset: i32,
    charge: bool,
    device: Device,
) -> Result<()> {
    let Some((lin, snapshot)) = lin.zip(snapshot).filter(|(l, _)| !l.is_empty()) else {
        return truncate_kv_to(kv, target_offset);
    };

    let kept = (target_offset - pre_round_offset).max(0) as usize;
    if kept > round_tokens.len() {
        return Err(Error::Model(format!(
            "rollback_round_caches: retained prefix {kept} exceeds the {} tokens the \
             round consumed (pre_round_offset={pre_round_offset}, \
             target_offset={target_offset}) — the caller's offsets do not describe \
             this round",
            round_tokens.len(),
        )));
    }

    // The whole verify block, not the rejected tail: the recurrent state has no
    // sequence axis to truncate, so it is restored from the pre-round snapshot
    // and the accepted prefix replayed into KV caches that must start where the
    // snapshot does. A sliding-window ring cannot serve that past its wrap — it
    // can give back a rejected tail but not the block that established its
    // window — so this branch and a sliding layer are mutually exclusive. No
    // architecture wired today pairs the two (`Architecture::layer_sliding_window`
    // names only the Gemma families, and neither carries recurrent state); the
    // first one that does lands here rather than anywhere subtler.
    truncate_kv_to(kv, pre_round_offset).map_err(|e| {
        Error::Model(format!(
            "rollback_round_caches: a recurrent verifier rolls the whole verify block off \
             and replays the accepted prefix, which a sliding-window layer cannot do past \
             its wrap — this architecture pairs recurrent state with a windowed KV layer \
             and the round loop has no rollback for that combination: {e}"
        ))
    })?;
    for (c, snap) in lin.iter_mut().zip(snapshot) {
        c.restore_snapshot(snap);
    }
    if kept == 0 {
        // Nothing retained beyond the pre-round state — the snapshot is
        // already the answer and the caches are already at `target_offset`.
        return Ok(());
    }
    let replayed =
        arch.forward_seq_last_k_with_cache(&round_tokens[..kept], 1, kv, Some(lin), device)?;
    if charge {
        // Nothing reads this replay until the next round's verify forward, so
        // with nothing forcing it here the whole second weight read is billed
        // to that round's verify span. See `phases_charged`.
        replayed.eval()?;
    }
    Ok(())
}

/// Truncate every KV cache in `kv` that actually holds `n` or more positions.
///
/// A GDN layer's KvCache never advances past 0, so an unguarded truncate would
/// set it to a positive offset over an empty store.
///
/// **All or nothing.** Every layer is asked whether it can reach `n` before any
/// is moved, because a stack left half rolled back is the defect this function
/// exists to prevent, not a milder version of it: an SWA ring that kept the
/// rejected drafts while the full-attention layers dropped them is how a
/// speculative arm stops reproducing plain greedy at long context, and a
/// failure part-way through the loop produces exactly that state with no way
/// back. `KvCache::can_truncate_to` decides reachability on exactly the ground
/// `truncate_to` refuses on — a sliding-window ring's order past its wrap — so
/// on that question the gate and the operation cannot disagree.
///
/// It does not model a fault in the write itself: a ring admitted with no
/// recorded stream, or a buffer that is not 4-D. Both are structural invariants
/// rather than states a caller can reach, and either would still return
/// mid-stack.
fn truncate_kv_to(kv: &mut [KvCache], n: i32) -> Result<()> {
    if let Some((idx, c)) = kv
        .iter()
        .enumerate()
        .find(|(_, c)| c.offset() >= n && !c.can_truncate_to(n))
    {
        return Err(Error::Model(format!(
            "truncate_kv_to: layer {idx} holds {} positions and cannot be rolled back to \
             {n}, so no layer was, and the stack is still where the round left it",
            c.offset(),
        )));
    }
    for c in kv.iter_mut() {
        if c.offset() >= n {
            c.truncate_to(n)?;
        }
    }
    Ok(())
}

/// Run `n` greedy decode steps through `model` with persistent `caches`.
/// Returns the `n` token ids generated. Each step feeds the prior step's
/// argmax via an MLX Array (no CPU readback between steps; final
/// `to_bytes()` materialises all `n` ids in one sync).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn draft_decode_n(
    arch: &Architecture,
    seed: &[u32],
    n: usize,
    caches: &mut [KvCache],
    mut lin_caches: Option<&mut [LinearAttnCache]>,
    device: Device,
) -> Result<Vec<u32>> {
    if n == 0 {
        return Ok(vec![]);
    }
    if seed.is_empty() {
        return Err(Error::Model("draft_decode_n: empty seed".into()));
    }

    // Step 0: feed all seed tokens at once (typical seed.len() = 1 or 2).
    let seed_i32: Vec<i32> = seed.iter().map(|&x| x as i32).collect();
    let mut y_arr = Array::from_i32_slice(&seed_i32, &[seed.len() as i32])?;

    let mut emitted_arrays: Vec<Array> = Vec::with_capacity(n);
    for _step_idx in 0..n {
        // For GDN-bearing drafters (Qwen3.5MoE) the recurrent lin_caches are
        // advanced alongside kv_caches every step. `as_deref_mut` reborrows
        // the Option<&mut [..]> across loop iterations. Gemma4 passes None.
        let logits = arch.forward_arr_with_cache(
            &y_arr,
            y_arr.shape()[0],
            caches,
            lin_caches.as_deref_mut(),
            device,
        )?;
        // logits shape: [1, 1, vocab] (forward_arr returns last-position only).
        // argmax(axis=-1) over [1,1,vocab] → [1,1]; reshape to [1] for next input.
        let next = argmax(&logits, -1, device)?;
        let _ = next.async_eval();
        emitted_arrays.push(next.try_clone()?);
        y_arr = next.reshape(&[1], device)?;
    }

    // Materialise all n argmax arrays in one sync.
    let mut tokens: Vec<u32> = Vec::with_capacity(n);
    for arr in emitted_arrays {
        arr.eval()?;
        let bytes = arr.to_bytes()?;
        if bytes.len() < 4 {
            return Err(Error::Model("draft_decode_n: argmax bytes empty".into()));
        }
        let id = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        tokens.push(id);
    }
    Ok(tokens)
}

/// Stochastic variant of [`draft_decode_n`]: run `n` decode steps,
/// sampling each token from the draft's post-sampling distribution `q_i` and
/// returning both the sampled token ids and the per-step `q_i` distributions.
///
/// Unlike the greedy `draft_decode_n` (which batches argmax and syncs once),
/// each step must read back the full last-position logits to build `q_i` and
/// draw `x_i ~ q_i` before feeding `x_i` into the next step — so this path has
/// one GPU→host transfer per draft step (the same per-token transfer the
/// standard `temp > 0` decode already pays).
///
/// `q_i` is built with [`crate::sampler::sampling_distribution`] using the same
/// `SamplerConfig` / `PenaltyConfig` as the verifier's `p_i`, so acceptance is
/// unbiased (Leviathan: p and q must be the matched post-sampling distributions).
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn draft_decode_n_stochastic(
    arch: &Architecture,
    seed: &[u32],
    n: usize,
    caches: &mut [KvCache],
    mut lin_caches: Option<&mut [LinearAttnCache]>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    penalty_cfg: &crate::sampler::PenaltyConfig,
    recent: &[u32],
    rng: &mut crate::sampler::Pcg32,
    device: Device,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    use crate::sampler::{sample_index, sampling_distribution};

    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if seed.is_empty() {
        return Err(Error::Model("draft_decode_n_stochastic: empty seed".into()));
    }

    let seed_i32: Vec<i32> = seed.iter().map(|&x| x as i32).collect();
    let mut y_arr = Array::from_i32_slice(&seed_i32, &[seed.len() as i32])?;

    let mut tokens: Vec<u32> = Vec::with_capacity(n);
    let mut q_dists: Vec<Vec<f32>> = Vec::with_capacity(n);

    for _step in 0..n {
        let logits = arch.forward_arr_with_cache(
            &y_arr,
            y_arr.shape()[0],
            caches,
            lin_caches.as_deref_mut(),
            device,
        )?;
        // logits shape: [1, 1, vocab]. sampling_distribution reads vocab from
        // the last axis, so the [1,1,vocab] shape is accepted directly.
        let q = sampling_distribution(&logits, sampler_cfg, None, penalty_cfg, recent)?;
        let id = sample_index(&q, rng) as u32;
        q_dists.push(q);
        tokens.push(id);
        // Feed the sampled token into the next step.
        let id_i32 = id as i32;
        y_arr = Array::from_bytes(&id_i32.to_le_bytes(), &[1], Dtype::I32)?;
    }

    Ok((tokens, q_dists))
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests;
