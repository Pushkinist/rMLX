// LOC-exempt. Three things live here and the first two cannot be separated:
// `SpeculativeDispatcher`, which owns the pair and the two two-model entries,
// and what every drafter path shares — prefill chunking, the acceptance walk,
// the request's draw, the block arithmetic, the guards a round is refused by,
// the emit site. Splitting those by drafter duplicates them; splitting them by
// phase separates a guard from the step it guards; and the entries are where a
// request is refused before a cache stack is built, which is one statement away
// from the dispatcher that holds the models.
//
// The third could leave: the snapshot and vocabulary pairing block —
// `vocab_pairing_verdict`, `unread_tensor_refusal` and the vocabulary readers
// under them — is a loader-side check with no round-loop caller, and it is the
// natural first split of this file. It has not moved because nothing this
// change does touches it, and a move is a diff across every caller of a check
// whose whole value is that it refuses before a model is built.
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
pub mod dflash2;
pub mod eagle3;
pub mod gemma4_assistant;
pub mod mtp;

pub(crate) mod draft_kind;
pub(crate) mod round_common;
pub(crate) mod round_loop;
pub(crate) mod round_stats;
pub(crate) mod two_model;

#[cfg(test)]
pub(crate) mod text_scan;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, subtract, Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};

use crate::arch::{load_model, Architecture, LoadOpts};
use crate::decode_loop::ProbeStep;
pub use draft_kind::{Declared, DraftKind};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use round_loop::{run_rounds, RoundCfg};
pub(crate) use round_stats::{
    log_round, phases_charged, RoundPhases, RoundReport, RoundStats, SpecLoop,
};

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

/// Largest block any round loop here can verify, and so the largest a
/// checkpoint may declare or a request ask for.
///
/// **The block is scored in one un-chunked forward.** A round calls
/// `Architecture::forward_verify_capture` once over the carry token and every
/// proposal, and that forward materialises `block_size * vocab_size` logits in a
/// single Metal command buffer — the round loop needs all of them, because it
/// argmaxes every position to walk the acceptance. There is no chunked variant
/// it could fall back on: `forward_verify_capture_chunked` exists precisely
/// because that stops working, and it buys its headroom by materialising the
/// *last* position's logits only, which a verify pass cannot do.
///
/// The number is the one that path already records as measured: a `[1, n, vocab]`
/// logit tensor in one command buffer times the GPU out above roughly a thousand
/// positions on this verifier's family, and a 4096-position single shot exceeds
/// the Metal watchdog on logits alone. So a block above this describes a round
/// that cannot be run rather than one that would be slow — and the block is what
/// sizes the round's token buffer, its verify input, and on the loops that have
/// one, the selector chain and a mask quadratic in it.
///
/// Real blocks are single digits; the published DFlash 2 checkpoint declares 8
/// and its own guidance recommends 5 against a quantized pair. This is a
/// structural ceiling with two orders of magnitude of headroom over anything
/// that drafts, not a tuning knob.
pub const MAX_BLOCK_SIZE: usize = 1024;

/// The round block a request that named none runs at, before a drafter's own
/// declaration narrows it: the verifier's own token plus four drafted.
///
/// One producer for the whole workspace. The serve layer resolves the served
/// block from this and the drafter's declared depth, and the equivalence gate
/// drives a pair that names no block at the same number, so the width that gate
/// judges is a width an operator is actually served. A second copy of the value
/// anywhere makes those two silently different runs.
///
/// It is not derived from any checkpoint. Which block each drafter should
/// default to is a throughput question and belongs to a sweep; this is the
/// number that stands until one answers it.
pub const DEFAULT_BLOCK_SIZE: usize = 5;

/// The block a request that named none runs at, given whatever depth the
/// drafter's checkpoint declares.
///
/// [`DEFAULT_BLOCK_SIZE`] capped by the declaration: a checkpoint is not asked
/// for more depth than it was trained at unless someone asks, and a deeper
/// declaration does not move what an operator is served, because that is a
/// throughput choice and belongs to a sweep. A drafter that declares nothing
/// takes the constant.
///
/// One producer. The serve layer resolves the served block with this, and the
/// two test harnesses that drive a loop the way a no-flag request would resolve
/// it the same way — so a pair or an alignment cell covers the configuration an
/// operator gets rather than one that agreed with it when it was written.
#[must_use]
pub fn default_block_for(declared: Option<usize>) -> usize {
    declared.map_or(DEFAULT_BLOCK_SIZE, |d| DEFAULT_BLOCK_SIZE.min(d))
}

/// The block a round runs at when the drafter's declared depth is a real
/// constraint: what the request asked for, what the checkpoint was trained at,
/// and what one verify forward can score — whichever is smallest, and never
/// below the two positions a seed and one draft need.
///
/// Three loops narrow this way and share this. DFlash 1 and DFlash 2 denoise a
/// block whose width *is* the drafter's input shape, and EAGLE-3's head is
/// defined over its own block; none of the three can propose past what its
/// checkpoint names. The MTP sidecar can, which is why it does not call this.
///
/// **The [`MAX_BLOCK_SIZE`] clamp is not the loaders' guarantee restated.** The
/// drafter structs are public with public fields, so one reaching a round loop
/// need not have come through a loader and its config need not have been
/// checked — the tests build them directly. The block sizes the round's token
/// buffer and its verify input, so each loop bounds it on its own behalf rather
/// than on a promise its argument did not have to make.
pub(crate) fn block_capped_by_checkpoint(requested: usize, declared: usize) -> usize {
    requested.min(declared).clamp(2, MAX_BLOCK_SIZE)
}

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

/// Whether the draft snapshot's tokenizer can stand in for the verifier's,
/// read from the two snapshot directories.
///
/// [`SpeculativeDispatcher::load_speculative`] applies this before any weight is
/// read. It is public because the gate that judges a two-model pair has to apply
/// the same check on the same pair before it builds the dispatcher directly, and
/// two readings of "is this a pair" that can disagree are worse than one.
///
/// # Errors
/// [`Error::SpeculativePairing`] with the verdict's reason; [`Error::Model`]
/// when either `tokenizer.json` cannot be read.
pub fn vocab_pairing(verifier_dir: &Path, draft_dir: &Path) -> Result<()> {
    vocab_pairing_verdict(&snapshot_vocab(verifier_dir)?, &snapshot_vocab(draft_dir)?)
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
        vocab_pairing(verifier_dir, draft_dir)?;
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
    ///
    /// Returns the emitted steps and **the widest block any round of this run
    /// actually ran**, the verifier's own token included. Not the `k + 1` that
    /// was asked for: this loop narrows its draft count per round against the
    /// remaining token budget, and a caller checking what it asked for against
    /// its own argument would be checking nothing.
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
    ) -> Result<(Vec<ProbeStep>, usize)> {
        if k == 0 {
            return Err(Error::Model("spec_generate_greedy: k must be >= 1".into()));
        }
        let k = two_model_drafts_per_round(k);
        if n_tokens == 0 {
            return Ok((vec![], k + 1));
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
                sampler_cfg,
            )
        }
        // The inner loops count drafts; every other loop here counts the block
        // that holds them, so this reports the block the widest round ran.
        .map(|(emitted, widest_draft)| (emitted, widest_draft + 1))
    }

    /// Speculative decoding over two complete models, greedy by routing.
    ///
    /// Nothing below decides to be greedy: the rounds draw every token through
    /// the request's own `VerifierDraw`. What makes this path greedy is
    /// [`Self::spec_generate_greedy`]'s branch above it, which routes a request
    /// whose sampler is active to the stochastic loop instead — so the draw this
    /// path reads is the device argmax on every request that reaches it.
    ///
    /// Mirrors mlx-lm `speculative_generate_step`: one prefill of the prompt
    /// less its last token on each model, then rounds of draft / verify /
    /// accept, with both KV stacks rolled back to the accepted prefix. The draft
    /// model keeps its own persistent cache, so a round costs `k` decode steps
    /// at draft-model speed rather than a re-prefill.
    ///
    /// The rounds run in [`run_rounds`]; what is here is the request's block and
    /// the refusal that runs before a cache stack is built. The drafter is
    /// [`two_model::TwoModelRound`].
    ///
    /// `k` is a **proposal** count where every sidecar loop takes a block, so
    /// the block the loop is configured with is `k + 1` and what this returns is
    /// the widest proposal count any round ran — its caller adds the verifier's
    /// own token back.
    ///
    /// An architecture whose `forward_seq_last_k_with_cache` is unwired returns
    /// `Error::Model` from the round loop; there is no fallback path.
    #[allow(clippy::too_many_arguments)]
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
        sampler_cfg: &crate::sampler::SamplerConfig,
    ) -> Result<(Vec<ProbeStep>, usize)> {
        if prompt_ids.len() < 2 {
            return Err(Error::Model(
                "spec_generate_greedy_cached: prompt must have \u{2265}2 tokens".into(),
            ));
        }
        let mut round =
            two_model::TwoModelRound::new(self.draft_model()?, two_model::Acceptance::Prefix);
        let (emitted, widest_block) = run_rounds(
            &self.verifier,
            &mut round,
            prompt_ids,
            step_fn,
            &RoundCfg {
                loop_kind: SpecLoop::TwoModelGreedy,
                block_size: k + 1,
                n_tokens,
                eos_ids,
                tokenizer,
                sampler_cfg,
                // This request times no phase, so it charges none.
                charged: false,
                kv_quant_override,
                max_ctx_override,
            },
            None,
            self.device,
        )?;
        Ok((emitted, drafts_per_round(widest_block)))
    }

    /// Speculative decoding over two complete models, sampled by routing.
    ///
    /// The same pair, the same caches and the same rollback as
    /// [`Self::spec_generate_greedy_cached`]; what differs is the rule a round
    /// accepts by. Above temperature 0 the draft model draws each proposal from
    /// its own post-sampling distribution `q_i`, the verifier scores the block's
    /// own `p_i`, and each proposal is accepted with probability
    /// `min(1, p_i(x_i)/q_i(x_i))` — on the first rejection the round commits a
    /// correction drawn from the residual `normalize((p_i - q_i)+)`, and a round
    /// that rejected nothing commits the verifier's own draw past the last
    /// proposal. That preserves the verifier's output distribution exactly
    /// (Leviathan 2023 §2.3, Thm 1), which matching sampled proposals against an
    /// argmax would not.
    ///
    /// `p` and `q` are drawn through the request's one
    /// [`VerifierDraw`](super::speculative::VerifierDraw), so they are the same
    /// post-temperature / post-top-p / post-top-k / post-min-p distributions the
    /// ordinary decode path builds — a hard correctness requirement, since
    /// mismatched `p` and `q` bias the output — and they advance one seeded
    /// stream, which is what
    /// `crates/rmlx-models/tests/two_model_stochastic.rs` reproduces.
    ///
    /// The rounds run in [`run_rounds`]; what is here is the request's block and
    /// the refusal that runs before a cache stack is built. The drafter is
    /// [`two_model::TwoModelRound`] under [`two_model::Acceptance::Stochastic`].
    ///
    /// `k` is a **proposal** count where every sidecar loop takes a block, so
    /// the block the loop is configured with is `k + 1` and what this returns is
    /// the widest proposal count any round ran — its caller adds the verifier's
    /// own token back.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(Vec<ProbeStep>, usize)> {
        if prompt_ids.len() < 2 {
            return Err(Error::Model(
                "spec_generate_stochastic_cached: prompt must have \u{2265}2 tokens".into(),
            ));
        }
        let mut round = two_model::TwoModelRound::new(
            self.draft_model()?,
            two_model::Acceptance::Stochastic(Vec::new()),
        );
        let (emitted, widest_block) = run_rounds(
            &self.verifier,
            &mut round,
            prompt_ids,
            step_fn,
            &RoundCfg {
                loop_kind: SpecLoop::TwoModelStochastic,
                block_size: k + 1,
                n_tokens,
                eos_ids,
                tokenizer,
                sampler_cfg,
                // This request times no phase, so it charges none.
                charged: false,
                kv_quant_override,
                max_ctx_override,
            },
            None,
            self.device,
        )?;
        Ok((emitted, drafts_per_round(widest_block)))
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

/// Refill a round's token buffer in place.
///
/// One allocation per request rather than per round, and one place where the
/// clear happens, so a caller cannot add a round that extends a buffer the
/// previous round left full.
fn fill_fed(buf: &mut Vec<u32>, head: &[u32], tail: &[u32]) {
    buf.clear();
    buf.extend_from_slice(head);
    buf.extend_from_slice(tail);
}

/// Arm a round tape on every recurrent cache in `lin`, discarding whatever the
/// previous round left on it.
///
/// A no-op for full-attention archs, which have no recurrent caches. Call it
/// once per round, before the forwards that round takes: every GDN forward
/// through an armed cache records its recurrence inputs, and
/// `round_common`'s refold rebuilds them when the round is partly rejected.
fn arm_lin_tapes(lin: Option<&mut [LinearAttnCache]>) {
    for c in lin.into_iter().flatten() {
        c.arm_tape();
    }
}

/// Drop the round tapes on every recurrent cache in `lin`.
///
/// A fully accepted round keeps the state its forwards produced and has nothing
/// to refold, so its tape is dead the moment the round ends. Holding it to the
/// next round's arming would keep a block's worth of activations per layer alive
/// for no reader.
fn disarm_lin_tapes(lin: Option<&mut [LinearAttnCache]>) {
    for c in lin.into_iter().flatten() {
        let _ = c.take_tape();
    }
}

/// How a round loop reads the verifier's own token out of a block of logits.
///
/// At temperature 0 that is the device argmax. Above it, it is a draw from the
/// verifier's post-sampling distribution at each position, taken through the
/// same host pipeline (`sampling_distribution`) the ordinary decode path uses,
/// so a speculative arm and a plain arm sample from the same distribution.
///
/// Feeding those draws to the acceptance walk unchanged is a distributionally
/// exact sampled speculative step, and needs nothing from the drafter. The walk
/// emits the verifier's own token at every position it reaches, and it only
/// reaches position `i` when the proposals so far all agreed — so the emitted
/// prefix is the prefix the verifier scored, and each emitted token is a draw
/// from the verifier's distribution at exactly that prefix. The proposal decides
/// how far the walk gets; it never decides what comes out. This is the
/// acceptance rule with a point-mass proposal, which
/// `a_point_mass_proposal_emits_the_target_and_matches_sample_and_match` pins
/// against the residual form.
///
/// Holds its own RNG, one per request, so the draw stream is contiguous across
/// rounds and a seeded request reproduces byte for byte. It is the request's
/// **one** sampling stream: a drafter that samples its own proposals draws them
/// here too, through [`Self::proposal`], rather than seeding a second generator
/// from the same seed — two streams off one seed are correlated, and only one
/// of them is the stream a reproducibility pin describes.
pub(crate) struct VerifierDraw {
    cfg: crate::sampler::SamplerConfig,
    penalties: crate::sampler::PenaltyConfig,
    rng: crate::sampler::Pcg32,
}

impl VerifierDraw {
    pub(crate) fn new(cfg: &crate::sampler::SamplerConfig) -> Self {
        Self {
            cfg: *cfg,
            // The speculative path refuses penalties and constrained decoding at
            // the request boundary, so the distribution builder runs with a
            // no-op configuration and an empty history window. A round would
            // need the penalty window rebuilt per drafted position, which no
            // loop carries.
            penalties: crate::sampler::PenaltyConfig::default(),
            rng: crate::sampler::Pcg32::new(cfg.seed_or_default()),
        }
    }

    /// Whether this request samples. `false` keeps every loop on the argmax
    /// path it had.
    pub(crate) fn sampling(&self) -> bool {
        self.cfg.sampling_active()
    }

    /// One token from a single-position logits array (`[1, 1, vocab]` or
    /// `[1, vocab]`) — the seed a loop emits straight after prefill.
    pub(crate) fn seed_token(&mut self, logits: &Array, device: Device) -> Result<u32> {
        if !self.sampling() {
            let am = argmax(logits, -1, device)?;
            am.eval()?;
            let bytes = am.to_bytes()?;
            return argmax_tokens(&bytes, 1)?.first().copied().ok_or_else(|| {
                Error::Model("seed_token: the verifier's argmax carried no id".into())
            });
        }
        let vocab = vocab_axis(logits)?;
        Ok(self.draw_row(&logits.reshape(&[1, vocab], device)?)? as u32)
    }

    /// The verifier's token at each of the `v_k` positions of one verified
    /// block, from its `[1, v_k, vocab]` logits.
    pub(crate) fn block_tokens(
        &mut self,
        logits: &Array,
        v_k: usize,
        device: Device,
    ) -> Result<Vec<u32>> {
        if !self.sampling() {
            let am = argmax(logits, -1, device)?;
            am.eval()?;
            let bytes = am.to_bytes()?;
            return argmax_tokens(&bytes, v_k);
        }
        let vocab = vocab_axis(logits)?;
        let mut tokens = Vec::with_capacity(v_k);
        for i in 0..v_k {
            let row = block_row(logits, i, vocab, device)?;
            tokens.push(self.draw_row(&row)? as u32);
        }
        Ok(tokens)
    }

    /// The verifier's whole post-sampling distribution at each of the `v_k`
    /// positions of one verified block.
    ///
    /// What an acceptance rule that compares distributions needs, where
    /// [`Self::block_tokens`] hands back the draw alone. Both build the same
    /// rows through the same pipeline; this one keeps them.
    pub(crate) fn block_distributions(
        &self,
        logits: &Array,
        v_k: usize,
        device: Device,
    ) -> Result<Vec<Vec<f32>>> {
        let vocab = vocab_axis(logits)?;
        let mut dists = Vec::with_capacity(v_k);
        for i in 0..v_k {
            dists.push(self.row_dist(&block_row(logits, i, vocab, device)?)?);
        }
        Ok(dists)
    }

    /// One token drawn from a drafting step's own logits, with the distribution
    /// it came from.
    ///
    /// The distribution is the request's, built exactly as the verifier's is,
    /// which is what makes a stochastic acceptance test unbiased — the rule
    /// compares two post-sampling distributions and is wrong if they are built
    /// differently.
    pub(crate) fn proposal(&mut self, logits: &Array) -> Result<(u32, Vec<f32>)> {
        let q = self.row_dist(logits)?;
        let id = crate::sampler::sample_index(&q, &mut self.rng) as u32;
        Ok((id, q))
    }

    /// The request's draw stream, for an acceptance rule that draws its own
    /// coins.
    ///
    /// Handed out rather than wrapped because the rule that reads it is the
    /// drafter's and belongs there; what may not move is the stream, which is
    /// one per request.
    pub(crate) fn rng(&mut self) -> &mut crate::sampler::Pcg32 {
        &mut self.rng
    }

    /// One row's post-sampling distribution, from a `[1, vocab]` or
    /// `[1, 1, vocab]` logits array.
    fn row_dist(&self, row: &Array) -> Result<Vec<f32>> {
        crate::sampler::sampling_distribution(row, &self.cfg, None, &self.penalties, &[])
    }

    fn draw_row(&mut self, row: &Array) -> Result<usize> {
        let probs = self.row_dist(row)?;
        Ok(crate::sampler::sample_index(&probs, &mut self.rng))
    }
}

/// Position `i` of a `[1, v_k, vocab]` block of logits, as a `[1, vocab]` row.
///
/// `vocab` is the caller's, read once per block: a block is one array and its
/// vocabulary axis does not change between its positions, where reading it here
/// would allocate a shape per verified position.
fn block_row(logits: &Array, i: usize, vocab: i32, device: Device) -> Result<Array> {
    let i = i as i32;
    logits
        .slice(&[0, i, 0], &[1, i + 1, vocab], &[1, 1, 1], device)?
        .reshape(&[1, vocab], device)
}

/// The vocabulary extent of a logits array — its last axis.
fn vocab_axis(logits: &Array) -> Result<i32> {
    match logits.shape().last() {
        Some(&v) if v > 0 => Ok(v),
        _ => Err(Error::Model(format!(
            "the verifier's logits came back shaped {:?}, which has no vocabulary axis",
            logits.shape()
        ))),
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

/// How many tokens a round of `block` tokens drafts: the block less the
/// verifier's own token.
///
/// The two-model loop is the only one that takes a draft count where the others
/// take a block, so it is the only place the two units meet — and they are the
/// same number one apart, which is exactly the shape a unit error hides in. One
/// producer, so the serve layer and the equivalence harness cannot drift into
/// asking that loop for different widths under the same name.
#[must_use]
pub const fn drafts_per_round(block: usize) -> usize {
    block.saturating_sub(1)
}

/// How many tokens a two-model round drafts, bounded by what one verify forward
/// can score.
///
/// This loop takes a draft count where the sidecar loops take a block; the round
/// verifies the carry token and every draft in one un-chunked pass, so the two
/// are the same quantity offset by one and take the same ceiling. Clamped rather
/// than refused, because the serve layer refuses an over-wide request at parse
/// time and this is the loop's own guard against a caller that is not it.
pub(crate) fn two_model_drafts_per_round(k: usize) -> usize {
    k.min(MAX_BLOCK_SIZE - 1)
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
/// `verifier_tokens` carries exactly one position more than `draft_tokens` —
/// the bonus slot — at every call site, and that is checked rather than
/// assumed. The two arguments are same-typed slices whose order carries the
/// whole meaning, so a swapped call compiles and still returns a plausible
/// accept count; the count then drives a KV rollback. Reversed, the lengths are
/// wrong by two, which is what this refuses.
pub(crate) fn accept_prefix(
    verifier_tokens: &[u32],
    draft_tokens: &[u32],
    budget: usize,
) -> Result<(usize, Vec<u32>)> {
    if verifier_tokens.len() != draft_tokens.len() + 1 {
        return Err(Error::Model(format!(
            "accept_prefix: {} verifier tokens against {} proposals — a verified block \
             is the proposals plus one bonus slot, and swapping the two arguments is \
             how these arrive the wrong way round",
            verifier_tokens.len(),
            draft_tokens.len()
        )));
    }
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
    Ok((accepted, emit))
}

/// The block a round runs, narrowed against what is left of the token budget.
///
/// `block_total` counts the verifier's own token, so a round with `remaining`
/// tokens still to emit can run at most `remaining + 1` of it.
///
/// The floor is a no-op for every input a round loop reaches — every block
/// resolver returns at least 2 and every loop's guard gives at least one
/// remaining token, so `min` alone already answers at least 2 — and it is here
/// for a caller that arrives at `remaining` some other way. A round of one
/// verifies the carry token and drafts nothing, which is plain decode wearing a
/// round's costs.
#[must_use]
pub(crate) fn round_block(block_total: usize, remaining: usize) -> usize {
    block_total.min(remaining + 1).max(2)
}

/// The verifier KV offset a round rolls back to, counted from where the round
/// started.
///
/// The forward consumed the carry token and every proposal, so keeping the carry
/// and the accepted prefix is what drops the rejected tail. The correction the
/// round emits past them is a prediction the verifier has not processed, and is
/// not one of the retained positions.
///
/// Counted from the post-forward offset the same position is
/// `v_offset_before - (proposals - accept)`, and the two agree whenever the
/// forward consumed `1 + proposals` positions, which is what every round's
/// verify input holds. That is the invariant a round line reporting the
/// post-forward read under `VERIFIER_OFFSET_BASIS::AfterTheForward` rests on —
/// it names this position under another number — and
/// `the_rollback_target_retains_the_carry_and_the_accepted_prefix` is what
/// holds the two together. The target itself is computed here and only here:
/// the pre-round read is the position the rollback returns to, where the
/// post-forward one is a function of how far each layer happened to advance.
#[must_use]
pub(crate) fn rollback_target_from_head(pre_round_offset: i32, accept: usize) -> i32 {
    pre_round_offset + accept as i32 + 1
}

/// Rows the two-model loop drops from the draft cache on a partial acceptance.
///
/// The drafting pass feeds its seed and every proposal but the last — the last
/// output is never fed back — so the draft cache advanced by `proposals` and
/// keeps the carry plus the accepted prefix. That is one row fewer dropped than
/// the verifier's, and dropping the extra one would discard the last accepted
/// draft's K/V every round, degrading the accept rate with nothing saying so.
#[must_use]
pub(crate) fn draft_rows_to_drop(proposals: usize, accept: usize) -> i32 {
    (proposals as i32 - accept as i32 - 1).max(0)
}

/// Largest difference allowed between a conditioning projection carried across
/// rounds and the same rows projected in one call, in the host-side fixtures.
///
/// It is not zero and cannot be: `fc` is a matmul, and MLX's kernel for it
/// accumulates differently at different row counts, so a row projected in a call
/// of 3 rows and the same row projected in a call of 40 land one to four `f32`
/// units in the last place apart — 1.2e-7 to 4.8e-7 at the magnitudes these
/// fixtures reach. What the bound has to separate that from is a projection of
/// the wrong rows, which differs by order 1, and it is twenty times the largest
/// rounding difference observed and five orders under that.
///
/// It is an `f32` figure taken on `f32` fixtures and **does not carry to a
/// checkpoint at its own dtype**, where the same two projections are dispatched
/// at different matmul heights over fewer mantissa bits. That residual is
/// reported per request by [`report_conditioning_residual`], not bounded here.
#[cfg(test)]
pub(crate) const PROJECTION_TOL: f32 = 1e-5;

/// Report the largest element-wise gap between a carried conditioning buffer's
/// tail and a fresh projection of the rows that tail was built from.
///
/// `carried` is the buffer the round left, `fresh` the seed row beside this
/// round's commit projected in one call, `projected` the rows the round
/// projected and `hidden` the projection's width. The tail read is the last
/// `1 + projected` rows of `carried`, which is what `fresh` holds.
///
/// Both block drafters take this probe once per request, on the round that
/// first extends their buffer, and `loop_kind` is what names the drafter on the
/// line. It is the whole of what differed between their two copies of it.
///
/// The two are the same rows through the same row-wise projection, so they agree
/// exactly in exact arithmetic. They are **not** required to agree bit for bit
/// at a checkpoint's dtype: a matmul's kernel and reduction order are chosen by
/// shape, and the carried tail was projected in two calls where the comparison
/// takes one. This reports that gap rather than assuming it away, which is why
/// it is a measurement and not an assertion — a drafter conditioned on a
/// last-place-different row proposes a different token only at a near-tie, and
/// the verifier then accepts a different number of them, which is where it shows
/// up first. It does not stop there: a different accept split changes the next
/// verify block's composition, so the verifier's own logits move in their last
/// place too and a near-tie of its own can resolve the other way. Equivalence is
/// judged by the oracle in `docs/SPEC_ANSWER_EQUIVALENCE.md`, not by byte
/// equality.
///
/// **What it does and does not reach.** Both arguments are `[1, rows, hidden]`
/// and bounded by one block, so this is one small pass taken once per request —
/// and so the heights it compares are the ones a round actually projects at, a
/// few rows against a few more. It says nothing about a projection taken at the
/// height of a whole generation, which is what a loop that re-projected its
/// accumulated buffer every round would have used. Reaching that height would
/// mean holding the raw capture for the whole request, which is the cost the
/// carried projection exists to remove.
///
/// # Errors
///
/// From the tail slice, the subtraction, or from reading the result back.
pub(crate) fn report_conditioning_residual(
    loop_kind: SpecLoop,
    carried: &Array,
    fresh: &Array,
    projected: i32,
    hidden: i32,
    device: Device,
) -> Result<()> {
    let rows = 1 + projected;
    let tail = carried.shape().get(1).copied().unwrap_or(0) - rows;
    let carried_tail =
        carried.slice(&[0, tail, 0], &[1, tail + rows, hidden], &[1, 1, 1], device)?;
    let gap = subtract(&carried_tail, fresh, device)?;
    let dtype = gap.dtype();
    tracing::debug!(
        ?loop_kind,
        rows,
        residual = max_abs_from_bytes(&gap.to_bytes()?, dtype),
        "conditioning: carried projection against a fresh one"
    );
    Ok(())
}

/// A round that reached for the conditioning it drafts from before the prefill
/// built one.
///
/// Three drafters carry a buffer across rounds — a projection for the two block
/// drafters, one sliced verifier row for the sidecar — and each refuses a round
/// that read it early the same way. `loop_kind` is what names the request.
pub(crate) fn missing_conditioning(loop_kind: SpecLoop) -> Error {
    Error::Model(format!(
        "{loop_kind:?}: a round read the conditioning it drafts from before the prefill \
         built one"
    ))
}

/// Refuse a round whose reduced-vocabulary prefix disagrees with what the
/// request declared or with what it accepted.
///
/// Two bounds, and they see different things.
///
/// `declared` is the request's own `Prefilled::restricted_read_back`, read a
/// frame above the round. A request that scores every position over the
/// verifier's whole vocabulary has no reduced prefix at all, so any prefix on
/// such a round is a round reporting a read-back the request did not take —
/// which is what a per-round flag made of free constants can say while every
/// text reading of the branch that produced it still passes.
///
/// The second bound is the acceptance. A round commits the accepted prefix and
/// one correction, and the correction is the verifier's own token over its whole
/// vocabulary whatever the positions before it were scored over. The bound is
/// therefore the acceptance and not the committed count, which a request's
/// remaining budget can cut below the acceptance while the prefix stays where it
/// was.
///
/// Either way round the report makes the declared boundary in
/// `docs/SPEC_ANSWER_EQUIVALENCE.md` waive a position that boundary exists to
/// judge, and nothing in an answer, a round line or an accept counter reports
/// it.
///
/// # Errors
///
/// [`Error::Model`] when a request that declared no reduced read-back reports a
/// prefix, or when `restricted` is above `accept`.
fn guard_restricted_prefix(
    loop_kind: SpecLoop,
    round: usize,
    declared: bool,
    restricted: usize,
    accept: usize,
) -> Result<()> {
    if !declared && restricted > 0 {
        return Err(Error::Model(format!(
            "{loop_kind:?}: round {round} attributed {restricted} tokens to a reduced \
             vocabulary on a request that declared it takes no reduced read-back; one of \
             the two is wrong and the equivalence boundary reads the round"
        )));
    }
    if restricted > accept {
        return Err(Error::Model(format!(
            "{loop_kind:?}: round {round} attributed {restricted} tokens to a reduced \
             vocabulary over an acceptance of {accept}; the correction past the accepted \
             prefix is the verifier's own token over its whole vocabulary, and waiving it \
             at the equivalence boundary hides the position that boundary judges"
        )));
    }
    Ok(())
}

/// Refuse a round that conditioned on a different number of rows than it
/// committed.
///
/// `projected` is read back from the array the projection returned; `committed`
/// is the round's own count of what it kept. They are the same number by
/// construction and nothing downstream reads both, which is the problem: a loop
/// that hands the projection one row too few conditions every later round on a
/// buffer missing its carry tokens, and greedy verification still emits the
/// verifier's own tokens, so the request succeeds and only the accept rate
/// falls.
///
/// # Errors
///
/// [`Error::Model`] when the two disagree.
fn guard_round_conditioning(round: usize, projected: i32, committed: usize) -> Result<()> {
    if projected < 0 || projected as usize != committed {
        return Err(Error::Model(format!(
            "speculative round {round} projected {projected} conditioning rows but \
             committed {committed}: the rows a round conditions the next one on are the \
             rows it kept, and nothing in an answer reports them diverging"
        )));
    }
    Ok(())
}

/// The rows a round commits out of its verify pass's capture: the **first**
/// `rows` positions, the carry token followed by the tokens the walk kept.
///
/// Which end this takes is the whole of it. The verify pass scored the carry
/// token, the accepted proposals and the rejected ones in one forward, and the
/// caches keep only the first two — so a slice from the other end conditions the
/// next round on drafts the verifier threw away. It is the same shape and the
/// same row count either way, and greedy verification emits the verifier's own
/// tokens whatever the drafter was conditioned on, so what moves first is the
/// accept rate rather than the text — far enough along, a changed accept split
/// reshapes the verify blocks and the text can move too, at a near-tie of the
/// verifier's own.
///
/// Both DFlash loops commit through this, and they count their rows
/// differently: one takes the accepted proposals plus the carry token, the other
/// the tokens it actually emitted, which the request's remaining budget can cut
/// short. That is why the count is the caller's and the end is not.
///
/// # Errors
///
/// [`Error::Model`] when the capture is not `[1, positions, width]`, when it
/// holds fewer positions than the round commits, when the round commits none, or
/// from the slice.
#[allow(
    clippy::indexing_slicing,
    reason = "each axis is read only after the rank has been compared against 3"
)]
fn committed_rows(v_hidden: &Array, rows: usize, width: i32, device: Device) -> Result<Array> {
    let shape = v_hidden.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[2] != width {
        return Err(Error::Model(format!(
            "committed_rows: the verify capture has shape {shape:?}, not the \
             [1, positions, {width}] this drafter's conditioning reads"
        )));
    }
    if rows == 0 {
        return Err(Error::Model(
            "committed_rows: a round commits no positions — every round keeps at least \
             the carry token the verifier scored, so an empty commit is a miscounted \
             round and not a round that kept nothing"
                .to_owned(),
        ));
    }
    let rows = rows as i32;
    let have = shape[1];
    if rows > have {
        return Err(Error::Model(format!(
            "committed_rows: the round commits {rows} positions but its verify \
             capture holds {have} positions"
        )));
    }
    v_hidden.slice(&[0, 0, 0], &[1, rows, width], &[1, 1, 1], device)
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
/// `q_i` is built through the request's own [`VerifierDraw`], which is the same
/// pipeline and the same RNG stream the verifier's `p_i` and every other draw of
/// the request go through — Leviathan needs `p` and `q` to be the matched
/// post-sampling distributions, and one stream is what a seeded run reproduces.
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
    draw: &mut VerifierDraw,
    device: Device,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
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
        // logits shape: [1, 1, vocab]. The distribution builder reads vocab from
        // the last axis, so the [1,1,vocab] shape is accepted directly.
        let (id, q) = draw.proposal(&logits)?;
        q_dists.push(q);
        tokens.push(id);
        // Feed the sampled token into the next step.
        let id_i32 = id as i32;
        y_arr = Array::from_bytes(&id_i32.to_le_bytes(), &[1], Dtype::I32)?;
    }

    Ok((tokens, q_dists))
}

/// Refuse a draft snapshot carrying tensors its loader never reads.
///
/// A drafter checkpoint of a generation newer than the loader ships weight
/// families that loader has no code for. Building the drafter out of the
/// remainder yields **the loader's** architecture wearing the checkpoint's
/// name, and the accept rate measured from it is filed under that name:
/// `decode_config` records `<kind>/block=N` either way and cannot tell the two
/// apart, so the row outlives any warning and cannot be re-attributed
/// afterwards. Refusing is what keeps that row from being written; it costs a
/// supported checkpoint nothing, which reads every tensor it ships.
///
/// `drafter` names the loader in the message, so a snapshot handed to the wrong
/// generation's loader says which one refused it.
///
/// `Ok(())` means every tensor in the snapshot was consumed. It returns a
/// `Result` rather than an `Option<Error>` so a call site that stops propagating
/// it is an `unused_must_use` warning, which `-D warnings` turns into a build
/// failure — the guard cannot be un-wired quietly.
pub(crate) fn unread_tensor_refusal(
    drafter: &str,
    present: &std::collections::HashSet<String>,
    consumed: &std::collections::HashSet<String>,
) -> Result<()> {
    let mut unread: Vec<&str> = present
        .iter()
        .map(String::as_str)
        .filter(|name| !consumed.contains(*name))
        .collect();
    if unread.is_empty() {
        return Ok(());
    }
    unread.sort_unstable();
    Err(Error::Model(format!(
        "{drafter}: the snapshot carries {} tensors this loader does not read \
         ({}); the drafter built from the rest would be this loader's architecture \
         and not the checkpoint's, and any accept rate measured from it would be \
         recorded under the checkpoint's name with nothing in the row to say so. \
         Refusing rather than serving a drafter that is not the one named.",
        unread.len(),
        unread.join(", ")
    )))
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "round_skeleton_tests.rs"]
mod round_skeleton_tests;
