//! The two-model drafter's side of the shared round loop.
//!
//! Mirrors mlx-lm `speculative_generate_step`: a second complete model of the
//! verifier's family, with its own KV cache and its own recurrent state, drafts
//! by stepping through the block one token at a time. Verifier-side the shape is
//! the one every drafter in this module family has and it is
//! [`run_rounds`](super::round_loop::run_rounds); what is here is what the loop
//! cannot supply — a second model's prefill and cache stack, an autoregressive
//! drafting pass, and a rollback that rolls that second stack back beside the
//! verifier's.
//!
//! # Why the draft side keeps one row more than the verifier drops
//!
//! The drafting pass feeds its seed and then every proposal but the last: each
//! step's input is the previous step's output, and the last output is never fed
//! back. So a round advances the draft cache by one row per proposal where it
//! advances the verifier's by the carry plus every proposal, and a partial
//! acceptance drops one row fewer from the draft — [`draft_rows_to_drop`] is
//! that arithmetic. The same asymmetry is what the resync below pays for: after
//! a round that accepted every proposal the draft cache is one token behind, so
//! the next drafting pass feeds that token ahead of the correction.
//!
//! # The two acceptance rules
//!
//! A two-model request runs one of two rules, and it is the only thing that
//! differs between the pair's two paths: the greedy prefix walk at temperature
//! 0, and Leviathan's stochastic test above it. [`Acceptance`] is that choice,
//! carried as a field of this one drafter rather than as a second drafter —
//! everything either rule does with a cache, a tape or a seed is the same, and
//! the rule is a statement about `propose` and `verify` alone.
//!
//! Neither rule decides its own temperature: every token of either comes off
//! the round's own [`VerifierDraw`](super::VerifierDraw), which is the request's
//! one sampling stream. What routes a request to one rule or the other is the
//! entry guard above both.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::Array;

use super::round_common::{cache_stack, lin_cache_stack, rollback_round};
use super::round_loop::{
    CacheSpan, Conditioning, Prefilled, ReportSkippedBy, RoundCtx, RoundDrafter, RoundOutcome,
    Seed, Verdict, VerifierOffsetBasis,
};
use super::{
    accept_prefix, arm_lin_tapes, draft_decode_n, draft_decode_n_stochastic, draft_rows_to_drop,
    drafts_per_round, fill_fed, prefill_chunked,
};
use crate::arch::Architecture;
use crate::sampler::{sample_index, stochastic_accept, AcceptDecision, Pcg32};
use rmlx_kv_quant::{KvCache, LinearAttnCache};

/// How a round decides which of its proposals the verifier stands behind.
///
/// The one thing the two two-model paths do not share; see
/// `docs/SPEC_ROUND_SKELETON.md` § "The acceptance rule". The greedy walk
/// compares tokens, and the stochastic one compares the two distributions each
/// position was drawn from. Both commit the same shape — an accepted prefix and
/// one token the verifier stands behind — which is why one
/// [`Verdict`] carries either.
#[allow(missing_debug_implementations)]
pub(crate) enum Acceptance {
    /// The verifier's own token at each position against the proposal made for
    /// it, accepting the agreed prefix. Exact at any temperature: the walk emits
    /// the verifier's draw at every position it reaches, and it only reaches a
    /// position whose proposals all agreed.
    Prefix,
    /// Leviathan's test: accept a proposal with probability `min(1, p(x)/q(x))`,
    /// and on the first rejection commit a correction drawn from the residual
    /// `normalize((p - q)+)`. Carries the distributions this round's proposals
    /// were drawn from — `q` — which only this rule draws and only this rule
    /// reads.
    Stochastic(Vec<Vec<f32>>),
}

/// One request's two-model drafting state: the draft model and everything of
/// this round that its own rollback reads.
#[allow(missing_debug_implementations)]
pub(crate) struct TwoModelRound<'a> {
    draft: &'a Architecture,
    /// The draft model's own per-layer caches, built at the verifier's codec and
    /// ceiling — the verifier owns a pair's KV geometry.
    kv: Vec<KvCache>,
    /// Its recurrent state, on a draft model that keeps one.
    lin: Option<Vec<LinearAttnCache>>,
    /// What the next drafting pass feeds before drafting: the round's
    /// correction, with the last proposal ahead of it on a round that accepted
    /// every one.
    seed: Vec<u32>,
    /// What this round's drafting pass fed, in order. The rollback keeps a
    /// prefix of it, and a recurrent refold is refused unless its tape holds
    /// exactly this many positions.
    fed: Vec<u32>,
    /// How many tokens this round proposed. The rollback's retention and the
    /// resync both read it, and neither reads the chain.
    proposals: usize,
    /// The last of them, which the drafting pass never fed back — the token the
    /// resync hands the next pass on a round that accepted every proposal.
    last_proposal: Option<u32>,
    /// Which rule this request's rounds accept by, and whatever state that rule
    /// carries across a round's two halves.
    accept: Acceptance,
}

impl<'a> TwoModelRound<'a> {
    pub(crate) fn new(draft: &'a Architecture, accept: Acceptance) -> Self {
        Self {
            draft,
            kv: Vec::new(),
            lin: None,
            seed: Vec::new(),
            fed: Vec::new(),
            proposals: 0,
            last_proposal: None,
            accept,
        }
    }

    /// The draft cache's logical position, over the layers that carry one: a
    /// recurrent layer's `KvCache` never advances, so the maximum is the round's
    /// and a layer-zero read is not.
    fn cache_offset(&self) -> i32 {
        self.kv.iter().map(KvCache::offset).max().unwrap_or(0)
    }
}

impl RoundDrafter for TwoModelRound<'_> {
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy = ReportSkippedBy::TheInRoundExit;
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis = VerifierOffsetBasis::AfterTheForward;

    /// Prefill both models on the prompt less its last token, and carry that
    /// token into the first round.
    ///
    /// Neither model has scored it, so it is what the first round's verify input
    /// opens with and what the first drafting pass feeds — and the request emits
    /// nothing until a round commits a token of its own.
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled> {
        let device = ctx.device;
        let Some((&last, head)) = prompt.split_last() else {
            return Err(Error::Model(
                "the two-model prefill: the prompt has no last token for the first \
                 round to carry"
                    .into(),
            ));
        };
        self.kv = cache_stack(self.draft, ctx.kv_quant, ctx.max_seq);
        self.lin = self
            .draft
            .needs_lin_caches()
            .then(|| lin_cache_stack(self.draft));

        let prefill_t0 = Instant::now();
        prefill_chunked(
            ctx.verifier,
            head,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        prefill_chunked(
            self.draft,
            head,
            &mut self.kv,
            self.lin.as_deref_mut(),
            device,
        )?;
        self.seed = vec![last];
        Ok(Prefilled {
            seed: Seed::Carried(last),
            prefill_ns: prefill_t0.elapsed().as_nanos(),
            // It scores every position over the verifier's whole vocabulary.
            restricted_read_back: false,
            // It carries no conditioning buffer: what crosses a round is the
            // draft model's own caches.
            projects_conditioning: false,
        })
    }

    /// Draft `block - 1` tokens by stepping the draft model through its own
    /// cache, greedily or by drawing each one, as this request's rule asks.
    ///
    /// The carry is the loop's and reaches the draft model inside `self.seed`,
    /// which the last round's resync built: on a full acceptance it is two
    /// tokens, and this parameter is only the second of them.
    fn propose(&mut self, ctx: &mut RoundCtx<'_>, _carry: u32, block: usize) -> Result<Vec<u32>> {
        let Self {
            draft,
            kv,
            lin,
            seed,
            fed,
            proposals,
            last_proposal,
            accept,
        } = self;
        // The draft model's own round tape, armed before the forwards that write
        // it. The pass takes one forward per proposal, so the tape accumulates
        // across the whole round and the rollback replays it as one prefix.
        arm_lin_tapes(lin.as_deref_mut());
        let n = drafts_per_round(block);
        let proposed = match accept {
            Acceptance::Prefix => {
                draft_decode_n(draft, seed, n, kv, lin.as_deref_mut(), ctx.device)?
            }
            // The distributions the proposals were drawn from are half of what
            // the rule tests, so the pass that drew them is what keeps them.
            Acceptance::Stochastic(q) => {
                let (proposed, drawn) = draft_decode_n_stochastic(
                    draft,
                    seed,
                    n,
                    kv,
                    lin.as_deref_mut(),
                    &mut ctx.draw,
                    ctx.device,
                )?;
                *q = drawn;
                proposed
            }
        };
        // What the pass actually fed, which is what its tape recorded and what
        // the rollback keeps a prefix of.
        fill_fed(
            fed,
            seed,
            proposed.split_last().map_or(&[], |(_, head)| head),
        );
        *proposals = proposed.len();
        *last_proposal = proposed.last().copied();
        Ok(proposed)
    }

    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict> {
        let device = ctx.device;
        let t0 = Instant::now();
        let v_logits = ctx.verifier.forward_seq_last_k_with_cache(
            fed,
            fed.len(),
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        let proposed = fed.get(1..).unwrap_or_default();
        // Each rule reads the verifier's block back the way it judges it — one
        // token per position, or one whole distribution — so the read-back is
        // inside the arm and its cost is billed to the verify span either way.
        let (accept, commit, verify_ns, walk_ns) = match &mut self.accept {
            Acceptance::Prefix => {
                let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;
                let verify_ns = t0.elapsed().as_nanos();
                let t0 = Instant::now();
                let (accept, commit) = accept_prefix(&v_tokens, proposed, remaining)?;
                (accept, commit, verify_ns, t0.elapsed().as_nanos())
            }
            // Taken, not borrowed. The distributions belong to the drafting pass
            // that drew them and to the one verify that judges it, so a second
            // read finds nothing: a `verify` that ran twice, or one whose
            // `propose` returned before drawing, would otherwise test this
            // round's `p` against the last round's `q`. Only a shape change is
            // refused below, and a stale round of the same width has the same
            // shape — which is a wrong answer with nothing to say so.
            Acceptance::Stochastic(q) => {
                let q = std::mem::take(q);
                let p = ctx.draw.block_distributions(&v_logits, fed.len(), device)?;
                let verify_ns = t0.elapsed().as_nanos();
                let t0 = Instant::now();
                let (accept, commit) = stochastic_prefix(&p, &q, proposed, ctx.draw.rng())?;
                (accept, commit, verify_ns, t0.elapsed().as_nanos())
            }
        };
        Ok(Verdict {
            accept,
            commit,
            verify_ns,
            walk_ns,
            // It scores every position over the verifier's whole vocabulary.
            restricted: 0,
        })
    }

    /// Drop the rejected tail from the draft model's own caches, and set up what
    /// the next drafting pass feeds.
    ///
    /// The resync is here and not in `propose` because it is a function of this
    /// round's acceptance, which `propose` runs before: a drafting pass that
    /// worked out for itself whether the previous round accepted everything
    /// would be keeping a second copy of a fact this round already has.
    fn rollback(
        &mut self,
        ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        outcome: RoundOutcome,
    ) -> Result<Option<CacheSpan>> {
        let before = self.cache_offset();
        // The drafter's own arm, whose answer is not the round's: `refolded` is
        // reported beside the verifier's target and reads the verifier.
        let _ = rollback_round(
            &mut self.kv,
            self.lin.as_deref_mut(),
            &self.fed,
            before - self.fed.len() as i32,
            before - draft_rows_to_drop(self.proposals, verdict.accept),
            ctx.charged,
            ctx.device,
        )?;

        // The token the next round carries, taken from the round rather than
        // re-derived off the same `Verdict`: one producer, so the drafting seed
        // and the loop's verify input cannot part.
        let correction = outcome.carry;
        self.seed.clear();
        if verdict.accept == self.proposals {
            self.seed.extend(self.last_proposal);
        }
        self.seed.push(correction);

        Ok(Some(CacheSpan {
            before,
            // Read back off the cache the rollback left rather than restated
            // from the figure it was handed, so the line cross-checks the
            // verifier's target instead of repeating this side's arithmetic.
            target: self.cache_offset(),
        }))
    }

    /// Nothing: this drafter's whole cross-round state is the draft model's own
    /// caches, and it carries no conditioning buffer for the loop to report rows
    /// off.
    fn condition(
        &mut self,
        _ctx: &RoundCtx<'_>,
        _verdict: &Verdict,
        _outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>> {
        Ok(None)
    }

    /// Nothing: what crosses a round here is two cache stacks, not an array the
    /// next round's drafter reads.
    fn carry(&self, f: &mut dyn FnMut(&[(&str, &Array)])) -> Result<()> {
        f(&[]);
        Ok(())
    }
}

/// The Leviathan acceptance walk over one verified block.
///
/// `p` is the verifier's post-sampling distribution at each of the block's
/// positions and `q` the distribution each proposal was drawn from, so `p` holds
/// exactly one position more than `q` — the bonus slot past the last proposal.
/// Each proposal is accepted with probability `min(1, p(x)/q(x))` against one
/// draw of `rng`; the first rejection commits a correction drawn from the
/// residual instead, and a round that rejected nothing commits the verifier's
/// own draw past the last proposal. Either way the request's output
/// distribution is the verifier's (Leviathan 2023, Thm 1).
///
/// Returns the accepted count and the tokens the round commits. The request's
/// budget is not read here: it caps the emission in the loop, and a round that
/// ran out of budget still committed the KV it committed.
///
/// The three slices' lengths carry the whole meaning and two of them are
/// same-typed, so a swapped or short argument list returns a plausible accept
/// count that then drives a KV rollback. That is what the refusal below is for.
fn stochastic_prefix(
    p: &[Vec<f32>],
    q: &[Vec<f32>],
    proposals: &[u32],
    rng: &mut Pcg32,
) -> Result<(usize, Vec<u32>)> {
    if p.len() != proposals.len() + 1 || q.len() != proposals.len() {
        return Err(Error::Model(format!(
            "stochastic_prefix: {} verifier distributions and {} proposal distributions \
             against {} proposals — a verified block is the proposals plus one bonus \
             slot, and one distribution per proposal",
            p.len(),
            q.len(),
            proposals.len()
        )));
    }
    let mut accept = 0usize;
    let mut correction: Option<u32> = None;
    for ((p_i, q_i), &x) in p.iter().zip(q.iter()).zip(proposals.iter()) {
        match stochastic_accept(p_i, q_i, x, rng)? {
            AcceptDecision::Accept => accept += 1,
            AcceptDecision::Reject(corr) => {
                correction = Some(corr);
                break;
            }
        }
    }
    let mut commit: Vec<u32> = Vec::with_capacity(accept + 1);
    commit.extend(proposals.iter().take(accept));
    // The one token the verifier stands behind: the correction where a proposal
    // was rejected, its own draw at the bonus slot where none was. The length
    // refusal above is what says there is such a slot.
    let closing = if let Some(corr) = correction {
        corr
    } else {
        let bonus = p.last().ok_or_else(|| {
            Error::Model("stochastic_prefix: the verified block has no bonus slot".into())
        })?;
        sample_index(bonus, rng) as u32
    };
    commit.push(closing);
    Ok((accept, commit))
}

#[cfg(test)]
#[path = "two_model_tests.rs"]
mod two_model_tests;
