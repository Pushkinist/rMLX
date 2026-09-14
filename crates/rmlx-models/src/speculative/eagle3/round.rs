//! The EAGLE-3 drafter's side of the shared round loop.
//!
//! Port of the round-loop half of mlx-vlm `mlx_vlm/speculative/eagle3.py`
//! (`_eagle3_rounds`). Verifier-side it is the shape every sidecar drafter in
//! this module family has — prefill, a bonus token out of a round-0 capture
//! forward, then rounds of draft / verify / accept / roll back — and that shape
//! is [`run_rounds`]. What is here is what the loop cannot supply: the drafter's
//! own KV prefill, the autoregressive block, the restricted-vocabulary verify
//! pass, and the accept-and-reseed that rolls the drafter's cache back by
//! re-running it.
//!
//! # The restricted vocabulary, and where it is decided
//!
//! The verify pass takes the argmax over the drafter's reduced target ids at
//! every position it may accept, and over the verifier's whole vocabulary at
//! the round's correction alone. A restricted argmax equals the true one
//! exactly when the true one is in the set, so an accepted position carries an
//! inexactness the correction does not. That is not recoverable from the tokens
//! afterwards, so the round reports how long its restricted prefix is
//! ([`Verdict::restricted`]) and the loop attributes one [`DecidedBy`] per
//! emitted token from it — one per token the request *emitted*, which its
//! budget can cut below what the round committed. `docs/SPEC_ANSWER_EQUIVALENCE.md`
//! is where that attribution is read and what it costs.
//!
//! A sampled request cannot take the read-back at all: the restricted row is a
//! logit vector over the drafter's ids and a distribution needs the whole row's
//! normalising constant, so softmaxing the subset spreads the missing mass over
//! the ids that survived and top-p and top-k then cut a different set. An argmax
//! is indifferent to that, which is why the reduction is sound at temperature 0
//! and only there. `verify` therefore reads the round's draw before it chooses a
//! read-back, and a sampled request pays the full-vocabulary projection at every
//! verified position.
//!
//! # The rollback re-runs rather than truncates
//!
//! [`Eagle3Drafter::accept_and_reseed`] is this drafter's whole cross-round
//! state change: it rolls the drafter cache back to the offset the round opened
//! at, re-runs the drafter over the accepted prefix plus the correction
//! conditioned on the verifier's capture at those positions, and reads the
//! drafter's own next-token prediction off the last position. All of it is
//! `rollback`, `condition` returns `None`, and the drafter-side target on the
//! round line is read back off the cache afterwards rather than computed — so it
//! is a cross-check on the verifier's target rather than a restatement of it.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, Array, Device};

use super::{find_full_pos, step_trace_enabled, DecidedBy, Eagle3Drafter, STEP_TARGET};
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use crate::speculative::round_loop::{
    run_rounds, CacheSpan, Conditioning, Prefilled, ReportSkippedBy, RoundCfg, RoundCtx,
    RoundDrafter, RoundOutcome, Verdict, VerifierOffsetBasis,
};
use crate::speculative::{
    accept_prefix, argmax_tokens, block_capped_by_checkpoint, guard_verifier_prefill_logits,
    SpecLoop,
};
use rmlx_kv_quant::KvQuant;

/// Verifier prefill window. The capture forward materialises one hidden row per
/// prompt position, and a whole long prompt in one command buffer exceeds the
/// Metal watchdog.
const PREFILL_CHUNK_SIZE: usize = 1024;

/// One request's EAGLE-3 drafting state.
///
/// The drafter keeps a KV cache of its own, so what crosses a round boundary is
/// that cache plus the two values the next block forward opens from: the hidden
/// at the correction position and the drafter's own prediction from it.
#[allow(missing_debug_implementations)]
pub(crate) struct Eagle3Round<'a> {
    drafter: &'a mut Eagle3Drafter,
    /// The verifier layers whose residual stream the drafter conditions on.
    aux_layer_ids: Vec<usize>,
    /// The hidden the next block forward conditions on, at the position the
    /// last round's correction sits.
    h_seed: Option<Array>,
    /// The drafter's own prediction at that position, handed to the next block
    /// so the correction is not run through the drafter twice.
    d_seed_tok: Option<u32>,
    /// The carry token this round opened on, kept for the round's own trace and
    /// for a commit the request's budget emptied.
    carry_tok: u32,
    /// The drafter cache offset this round opened at, which the rollback
    /// re-runs from.
    d_offset_before: i32,
    /// This round's proposals, kept because the rollback re-runs their accepted
    /// prefix rather than truncating it away.
    proposed: Vec<u32>,
    /// The verifier's capture at every position this round verified. A round
    /// that stopped on an EOS leaves it set — the loop breaks before `rollback`
    /// — and the request drops this struct immediately after.
    scored: Option<Array>,
    /// Whether this request scores its accepted positions over the drafter's
    /// reduced vocabulary. The drafter's offer and the request's draw together,
    /// decided once in `prefill` and read by `verify` and by the loop's own
    /// per-request line.
    restricted_read_back: bool,
    /// Rounds, proposals and acceptances this request has run, for the
    /// per-position step trace alone. The request's own totals are the loop's;
    /// these are read by nothing else, and by nothing at all unless
    /// [`step_trace_enabled`] is on.
    rounds: usize,
    total_draft: usize,
    total_accept: usize,
}

impl<'a> Eagle3Round<'a> {
    fn new(drafter: &'a mut Eagle3Drafter) -> Self {
        let aux_layer_ids = drafter.cfg.aux_layer_ids.clone();
        Self {
            drafter,
            aux_layer_ids,
            h_seed: None,
            d_seed_tok: None,
            carry_tok: 0,
            d_offset_before: 0,
            proposed: Vec::new(),
            scored: None,
            restricted_read_back: false,
            rounds: 0,
            total_draft: 0,
            total_accept: 0,
        }
    }

    /// The hidden the next block forward opens from, or the refusal for a round
    /// that reached for it before the prefill built one.
    fn seed_hidden(&self) -> Result<&Array> {
        self.h_seed.as_ref().ok_or_else(|| {
            Error::Model(
                "eagle3_generate: a round read the drafter's seed hidden before the \
                 prefill produced one"
                    .into(),
            )
        })
    }

    /// The verify pass over the drafter's reduced vocabulary, with one
    /// full-vocabulary correction.
    ///
    /// Restricted logits for all `v_k` positions (32000 draft rows against the
    /// verifier's whole vocabulary), their argmaxes mapped back to target ids,
    /// then full-vocabulary logits at exactly one position: the first the draft
    /// missed, or the bonus when it missed none.
    #[allow(
        clippy::indexing_slicing,
        reason = "both sites are bounded by the proposal count: the head slice is the proposals' own length, and the correction index is `find_full_pos` over that slice"
    )]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: this arm is taken only where the drafter offers its reduced ids, which is what the array's presence decides"
    )]
    fn verify_restricted(
        &mut self,
        ctx: &mut RoundCtx<'_>,
        fed: &[u32],
        proposals: &[u32],
    ) -> Result<(Vec<u32>, Array)> {
        let device = ctx.device;
        let v_k = fed.len();
        let (v_hidden, v_final_hidden) = ctx.verifier.forward_verify_capture_hot(
            fed,
            v_k,
            &self.aux_layer_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        let hot_ids_arr = self
            .drafter
            .hot_ids_arr()
            .expect("hot_path_active but no hot_ids_arr");
        let hot_ids_host = self.drafter.hot_ids_host();
        let hidden_sz = self.drafter.hidden_size() as i32;

        let hot_logits =
            ctx.verifier
                .hot_logits_from_final_hidden(&v_final_hidden, hot_ids_arr, device)?;
        let hot_am = argmax(&hot_logits, -1, device)?;
        hot_am.eval()?;
        let hot_bytes = hot_am.to_bytes()?;
        // Restricted vocabulary: the argmax is an index into `hot_ids`, not a
        // token id, but the read-back is the same device buffer and the same
        // guard applies to it.
        let mut tokens: Vec<u32> = argmax_tokens(&hot_bytes, v_k)?
            .into_iter()
            .map(|draft_idx| {
                let draft_idx = draft_idx as usize;
                hot_ids_host
                    .get(draft_idx)
                    .copied()
                    .unwrap_or(draft_idx as u32)
            })
            .collect();

        let full_pos = find_full_pos(proposals, &tokens[..proposals.len()]);
        let h_corr = v_final_hidden.slice(
            &[0, full_pos as i32, 0],
            &[1, full_pos as i32 + 1, hidden_sz],
            &[1, 1, 1],
            device,
        )?;
        // `v_final_hidden` is final-normed by `forward_verify_capture_hot`.
        let corr_logits = ctx.verifier.logits_from_final_hidden(&h_corr, device)?;
        let corr_am = argmax(&corr_logits, -1, device)?;
        corr_am.eval()?;
        // Read back through the same guarded reader as the reduced row above: a
        // one-token read-back is still a device buffer whose length is the
        // array's and not this function's to assume.
        let correction = argmax_tokens(&corr_am.to_bytes()?, 1)?
            .first()
            .copied()
            .ok_or_else(|| {
                Error::Model("eagle3_generate: the correction read-back returned no token".into())
            })?;
        tokens[full_pos] = correction;
        Ok((tokens, v_hidden))
    }

    /// One trace event per verified position, when the step switch is on.
    fn trace_steps(&self, v_tokens: &[u32], accept: usize) {
        if !step_trace_enabled() {
            return;
        }
        let running_ar = if self.total_draft > 0 {
            (self.total_accept as f64) / (self.total_draft as f64)
        } else {
            0.0
        };
        for (i, (&dt, &vt)) in self.proposed.iter().zip(v_tokens.iter()).enumerate() {
            tracing::trace!(
                target: STEP_TARGET,
                round = self.rounds,
                step = i,
                draft_tok = dt,
                verifier_tok = vt,
                accepted = i < accept,
                cumulative_accept_rate = running_ar,
                carry_tok = self.carry_tok,
                seed_tok = ?self.d_seed_tok,
                "eagle3 step"
            );
        }
    }
}

impl RoundDrafter for Eagle3Round<'_> {
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy = ReportSkippedBy::TheSeedExit;
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis = VerifierOffsetBasis::AfterTheForward;

    /// Run the verifier over the whole prompt with multi-aux hidden capture,
    /// draw the bonus token from its last position, then prefill the drafter's
    /// own KV cache on the prompt shifted by one with that bonus appended,
    /// conditioned on the capture.
    ///
    /// The span covers all three: the drafter's prefill is conditioned on the
    /// verifier's capture and is part of what this request paid before its first
    /// round.
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled> {
        let device = ctx.device;
        // A sampled request cannot take the reduced read-back — see this module's
        // header — so it is the request's draw and not the drafter alone that
        // decides it, and it is decided once, here, for the round line and for
        // every round's verify pass.
        self.restricted_read_back = self.drafter.hot_path_active() && !ctx.draw.sampling();
        // The drafter advances one row per committed token beside the verifier,
        // so its cache is sized to the ceiling the verifier's stack was built
        // at and cannot overflow first.
        self.drafter.reset(ctx.max_seq);

        let n = prompt.len();
        tracing::debug!(
            prompt_len = n,
            chunk_size = PREFILL_CHUNK_SIZE,
            "eagle3: verifier prefill (chunked)"
        );
        let prefill_t0 = Instant::now();
        let (bonus_logits, all_hidden) = ctx.verifier.forward_verify_capture_chunked(
            prompt,
            &self.aux_layer_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            PREFILL_CHUNK_SIZE,
            // This drafter's own KV prefill conditions on every prompt position,
            // so no capture row can be released early.
            None,
            device,
        )?;
        guard_verifier_prefill_logits(ctx.verifier, &bonus_logits, n)?;
        let bonus = ctx.draw.seed_token(&bonus_logits, device)?;

        // Mirrors mlx-vlm `prefill_from_target_hidden`: the drafter reads the
        // prompt shifted by one with the bonus appended, against the verifier's
        // capture at positions `0..n`.
        let mut shifted: Vec<u32> = Vec::with_capacity(n);
        shifted.extend_from_slice(prompt.get(1..).unwrap_or_default());
        shifted.push(bonus);
        let (h_seed, seed_tok) = self.drafter.prefill_from_verifier_hidden(
            ctx.verifier,
            &shifted,
            &all_hidden,
            device,
        )?;
        tracing::debug!(
            prompt_len = n,
            drafter_cache_offset = self.drafter.cache_offset(),
            seed_tok,
            "eagle3: drafter KV prefill done"
        );
        self.h_seed = Some(h_seed);
        self.d_seed_tok = Some(seed_tok);
        Ok(Prefilled {
            seed: bonus,
            prefill_ns: prefill_t0.elapsed().as_nanos(),
            restricted_read_back: self.restricted_read_back,
            // It carries no conditioning buffer: what crosses a round is its own
            // KV cache and one hidden row, and it projects no rows to report.
            projects_conditioning: false,
        })
    }

    /// Draft `block - 1` tokens autoregressively, opening from the prediction
    /// the last rollback left.
    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>> {
        self.carry_tok = carry;
        self.d_offset_before = self.drafter.cache_offset();
        let h_seed = self.seed_hidden()?.try_clone()?;
        let proposed =
            self.drafter
                .draft_block(ctx.verifier, carry, &h_seed, self.d_seed_tok, block)?;
        self.proposed.clone_from(&proposed);
        Ok(proposed)
    }

    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict> {
        let device = ctx.device;
        let proposals = fed.get(1..).unwrap_or_default();

        // The arm that runs is what says how long the round's reduced prefix is,
        // so the flag cannot name a read-back the round did not take.
        let t0 = Instant::now();
        let (v_tokens, v_hidden, scored_over_reduced_vocab) = if self.restricted_read_back {
            let (v_tokens, v_hidden) = self.verify_restricted(ctx, fed, proposals)?;
            (v_tokens, v_hidden, true)
        } else {
            let (v_logits, v_hidden) = ctx.verifier.forward_verify_capture(
                fed,
                fed.len(),
                &self.aux_layer_ids,
                &mut ctx.kv,
                ctx.lin.as_deref_mut(),
                device,
            )?;
            let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;
            (v_tokens, v_hidden, false)
        };
        let verify_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let (accept, commit) = accept_prefix(&v_tokens, proposals, remaining)?;
        let walk_ns = t0.elapsed().as_nanos();

        self.rounds += 1;
        self.total_draft += proposals.len();
        self.total_accept += accept;
        self.trace_steps(&v_tokens, accept);

        self.scored = Some(v_hidden);
        Ok(Verdict {
            accept,
            commit,
            verify_ns,
            walk_ns,
            // The accepted prefix carries the drafter's own tokens, which the
            // reduced argmax confirmed; the correction past them is the
            // verifier's over its whole vocabulary. A round that took the full
            // read-back restricted nothing.
            restricted: if scored_over_reduced_vocab { accept } else { 0 },
        })
    }

    /// Roll the drafter's cache back to the offset the round opened at by
    /// re-running it over the accepted prefix and the correction, and read the
    /// next round's seed off the last position.
    fn rollback(
        &mut self,
        ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        _outcome: RoundOutcome,
    ) -> Result<Option<CacheSpan>> {
        let Some(scored) = self.scored.take() else {
            return Err(Error::Model(
                "eagle3_generate: a round rolled back over a verify forward that did not run"
                    .into(),
            ));
        };
        // The verifier's own token at the accepted position. It parts from the
        // acceptance only on a commit the request's budget truncated, where the
        // request has emitted its last token and no later round reads this.
        let correction = verdict.commit.last().copied().unwrap_or(self.carry_tok);
        let (h_seed, seed_tok) = self.drafter.accept_and_reseed(
            ctx.verifier,
            self.d_offset_before,
            &self.proposed,
            correction,
            &scored,
            verdict.accept,
            ctx.device,
        )?;
        self.h_seed = Some(h_seed);
        self.d_seed_tok = Some(seed_tok);
        Ok(Some(CacheSpan {
            before: self.d_offset_before,
            // Read back off the cache the re-run left rather than computed from
            // the acceptance, which is what makes it a cross-check on the
            // verifier's target and not a restatement of it.
            target: self.drafter.cache_offset(),
        }))
    }

    /// Nothing: this drafter's whole cross-round state moves in `rollback`, and
    /// it carries no conditioning buffer for the loop to report rows off.
    fn condition(
        &mut self,
        _ctx: &RoundCtx<'_>,
        _verdict: &Verdict,
        _outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>> {
        Ok(None)
    }

    fn carry(&self, f: &mut dyn FnMut(&[(&str, &Array)])) -> Result<()> {
        f(&[("h_seed", self.seed_hidden()?)]);
        Ok(())
    }
}

/// Drive an EAGLE-3 drafter against its verifier.
///
/// `requested_block_total` is the round block including the verifier's own
/// token; it is clamped to the block the drafter's checkpoint declares.
///
/// `decided_by` is filled with one [`DecidedBy`] per emitted token, in emission
/// order, and is cleared first. A caller that only wants the tokens passes a
/// scratch vector; the answer-equivalence gate reads it, because whether the
/// restriction could have changed a token is a property of the round the token
/// came out of and not of the token.
///
/// `sampler_cfg` decides what "the verifier's own token" means at each position
/// — its argmax at temperature 0, a draw from its post-sampling distribution
/// above it. It also decides which read-back the verify pass may take: the
/// reduced one is an argmax over a subset of the row and a distribution needs
/// the whole row.
///
/// `step_fn` is called once per emitted token. Its `Option<u32>` return — the
/// forced-token contract the plain decode loop uses — is discarded here, as it
/// is on every speculative loop: a round's tokens are already the verifier's.
///
/// The rounds themselves run in [`run_rounds`]; what is here is what refuses
/// before a cache stack is built, and the request's block.
///
/// Returns the emitted steps and **the widest block any round of this run
/// actually ran**. Not the block resolved before the loop: a caller checking
/// what it asked for against that would be trusting the very step it wanted
/// checked, and every loop here narrows the block again per round against the
/// remaining token budget.
///
/// # Errors
///
/// [`Error::Model`] when the prompt is too short to seed a round, when the
/// verifier carries no recurrent state, and whatever the round loop refuses.
#[allow(clippy::too_many_arguments)]
pub fn eagle3_generate(
    verifier: &Architecture,
    drafter: &mut Eagle3Drafter,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    requested_block_total: usize,
    kv_quant_override: Option<KvQuant>,
    max_ctx_override: Option<i32>,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    decided_by: &mut Vec<DecidedBy>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<(Vec<ProbeStep>, usize)> {
    decided_by.clear();

    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "eagle3_generate: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "eagle3_generate: EAGLE-3 verifier must be the Qwen3.5/3.6-MoE \
             hybrid (needs GDN lin_caches + multi-layer hidden capture)"
                .into(),
        ));
    }
    let block_total = block_capped_by_checkpoint(requested_block_total, drafter.cfg.block_size);
    let mut round = Eagle3Round::new(drafter);
    run_rounds(
        verifier,
        &mut round,
        prompt_ids,
        step_fn,
        &RoundCfg {
            loop_kind: SpecLoop::Eagle3,
            block_size: block_total,
            n_tokens,
            eos_ids,
            tokenizer,
            sampler_cfg,
            // This request does not charge its phases for the work they issue:
            // its rounds leave the seed hidden for the next round's drafter to
            // force, and the timings say so.
            charged: false,
            kv_quant_override,
            max_ctx_override,
        },
        Some(decided_by),
        device,
    )
}

#[cfg(test)]
#[path = "round_tests.rs"]
mod round_tests;
