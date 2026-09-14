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

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::Array;

use super::round_common::{cache_stack, lin_cache_stack, rollback_round};
use super::round_loop::{
    CacheSpan, Conditioning, Prefilled, ReportSkippedBy, RoundCtx, RoundDrafter, RoundOutcome,
    Seed, Verdict, VerifierOffsetBasis,
};
use super::{
    accept_prefix, arm_lin_tapes, draft_decode_n, draft_rows_to_drop, drafts_per_round, fill_fed,
    prefill_chunked,
};
use crate::arch::Architecture;
use rmlx_kv_quant::{KvCache, LinearAttnCache};

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
    /// This round's proposals, kept because the rollback's retention and the
    /// resync are both read off them.
    proposed: Vec<u32>,
    /// The carry token this round opened on, for a commit the request's budget
    /// emptied.
    carry_tok: u32,
}

impl<'a> TwoModelRound<'a> {
    pub(crate) fn new(draft: &'a Architecture) -> Self {
        Self {
            draft,
            kv: Vec::new(),
            lin: None,
            seed: Vec::new(),
            fed: Vec::new(),
            proposed: Vec::new(),
            carry_tok: 0,
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
                "spec_generate_greedy_cached: the prompt has no last token for the first \
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
    /// cache.
    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>> {
        self.carry_tok = carry;
        // The draft model's own round tape, armed before the forwards that write
        // it. The pass takes one forward per proposal, so the tape accumulates
        // across the whole round and the rollback replays it as one prefix.
        arm_lin_tapes(self.lin.as_deref_mut());
        let proposed = draft_decode_n(
            self.draft,
            &self.seed,
            drafts_per_round(block),
            &mut self.kv,
            self.lin.as_deref_mut(),
            ctx.device,
        )?;
        // What the pass actually fed, which is what its tape recorded and what
        // the rollback keeps a prefix of.
        fill_fed(
            &mut self.fed,
            &self.seed,
            proposed.split_last().map_or(&[], |(_, head)| head),
        );
        self.proposed.clone_from(&proposed);
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
        let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;
        let verify_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let (accept, commit) =
            accept_prefix(&v_tokens, fed.get(1..).unwrap_or_default(), remaining)?;
        let walk_ns = t0.elapsed().as_nanos();
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
        _outcome: RoundOutcome,
    ) -> Result<Option<CacheSpan>> {
        let before = self.cache_offset();
        // The drafter's own arm, whose answer is not the round's: `refolded` is
        // reported beside the verifier's target and reads the verifier.
        let _ = rollback_round(
            &mut self.kv,
            self.lin.as_deref_mut(),
            &self.fed,
            before - self.fed.len() as i32,
            before - draft_rows_to_drop(self.proposed.len(), verdict.accept),
            ctx.charged,
            ctx.device,
        )?;

        // The verifier's own token at the accepted position. It parts from the
        // acceptance only on a commit the request's budget truncated, where the
        // request has emitted its last token and no later round reads this.
        let correction = verdict.commit.last().copied().unwrap_or(self.carry_tok);
        self.seed.clear();
        if verdict.accept == self.proposed.len() {
            self.seed.extend(self.proposed.last().copied());
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

#[cfg(test)]
#[path = "two_model_tests.rs"]
mod two_model_tests;
