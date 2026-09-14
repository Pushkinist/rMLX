//! The DFlash 1 drafter's side of the shared round loop.
//!
//! Port of the round-loop half of mlx-vlm `mlx_vlm/speculative/dflash.py`
//! (`_dflash_rounds`). Verifier-side it is the shape every sidecar drafter in
//! this module family has — prefill, a bonus token out of a round-0 capture
//! forward, then rounds of draft / verify / accept / roll back — and that shape
//! is [`run_rounds`]. What is here is what the loop cannot supply: the block
//! schedule, the non-autoregressive block the drafter denoises, the acceptance
//! walk over the verify pass, and the conditioning buffer it grows.
//!
//! # The block is the drafter's, not the loop's
//!
//! This is the one drafter whose verify width is not fixed:
//! [`super::dflash_next_block_size`] opens with the loop's own narrowing and
//! then moves the result by the accept rate of the recent rounds. That history
//! is this struct's, written where the round's acceptance is known and read at
//! the head of the next round, so the schedule is the `block` method and the
//! narrowing stays one function.
//!
//! # Conditioning: the buffer grows and is never bounded
//!
//! The reference gives its drafter a persistent draft K/V cache and feeds it
//! each round's newly committed rows. This drafter rebuilds that K/V from the
//! conditioning rows on every call instead, and carries the **projection** of
//! those rows rather than the capture they came from — `fc` and `hidden_norm`
//! are row-wise, so a row projected once is the row every later round would
//! have re-derived. Unlike the DFlash 2 drafter's sliding equivalent the buffer
//! is never trimmed: this drafter declares no window, its layers are
//! full-attention, and its block queries read the whole context unmasked.
//!
//! # Greedy only
//!
//! The drafter picks its block's tokens by argmax and returns no candidate
//! distribution to sample against. Like every other sidecar drafter here this
//! one is greedy, and the serve layer routes a sidecar request to it whatever
//! the request's temperature.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{concatenate, Array, Device};

use super::{dflash_next_block_size, DFlashDrafter};
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use crate::speculative::round_loop::{
    run_rounds, Conditioning, Prefilled, ReportSkippedBy, RoundCfg, RoundCtx, RoundDrafter,
    RoundOutcome, Verdict, VerifierOffsetBasis,
};
use crate::speculative::{
    accept_prefix, block_capped_by_checkpoint, committed_rows, guard_round_conditioning,
    guard_verifier_prefill_logits, missing_conditioning, prefill_chunked,
    report_conditioning_residual, SpecLoop,
};
use rmlx_kv_quant::KvQuant;

/// One request's DFlash 1 drafting state.
///
/// The drafter keeps no cache of its own — the block forward re-reads the
/// carried projection every round — so what crosses a round boundary is that
/// projection, the verify pass's capture the next one is extended from, and the
/// acceptance history the block schedule reads.
#[allow(missing_debug_implementations)]
pub(crate) struct AdaptiveRound<'a> {
    drafter: &'a DFlashDrafter,
    /// The verifier layers whose hidden the conditioning is projected from.
    target_layer_ids: Vec<usize>,
    /// One capture row's width, `len(target_layer_ids) * hidden_size`.
    condition_width: i32,
    /// The projection the next round's block forward conditions on.
    h_ctx: Option<Array>,
    /// The round-0 capture, kept until the first round has extended the buffer
    /// and then dropped: re-projecting it beside that round's commit is what
    /// says how far the carried projection sits from a fresh one at this
    /// checkpoint's dtype. One row, one round, and the probe releases it. See
    /// `conditioning_residual`.
    probe_seed: Option<Array>,
    /// The verifier's capture at every position this round verified. A round
    /// that stopped on an EOS leaves it set — the loop breaks before
    /// `condition` — and the request drops this struct immediately after, so it
    /// is released there rather than cleared on that path.
    scored: Option<Array>,
    /// `(accepted, drafted)` per round, most-recent last: the history the block
    /// schedule adapts on.
    recent: Vec<(usize, usize)>,
}

impl<'a> AdaptiveRound<'a> {
    fn new(drafter: &'a DFlashDrafter) -> Self {
        let target_layer_ids = drafter.cfg.target_layer_ids.clone();
        let condition_width = drafter.cfg.hidden_size as i32 * target_layer_ids.len() as i32;
        Self {
            drafter,
            target_layer_ids,
            condition_width,
            h_ctx: None,
            probe_seed: None,
            scored: None,
            recent: Vec::new(),
        }
    }

    /// The projection a round conditions on, or the refusal for a round that
    /// reached for it before the prefill built one.
    fn conditioning(&self) -> Result<&Array> {
        self.h_ctx
            .as_ref()
            .ok_or_else(|| missing_conditioning(SpecLoop::DFlash))
    }
}

impl RoundDrafter for AdaptiveRound<'_> {
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy = ReportSkippedBy::TheSeedExit;
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis = VerifierOffsetBasis::AfterTheForward;

    /// Prefill the prompt less its last token, then feed that token on its own:
    /// one forward gives both the first bonus token and the capture the first
    /// round conditions on.
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled> {
        let device = ctx.device;
        let Some((&last_prompt, head)) = prompt.split_last() else {
            return Err(Error::Model(
                "dflash_generate: an empty prompt reached the round loop".into(),
            ));
        };
        let prefill_t0 = Instant::now();
        prefill_chunked(
            ctx.verifier,
            head,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        let prefill_ns = prefill_t0.elapsed().as_nanos();

        let (r0_logits, r0_hidden) = ctx.verifier.forward_verify_capture(
            &[last_prompt],
            1,
            &self.target_layer_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        guard_verifier_prefill_logits(ctx.verifier, &r0_logits, prompt.len())?;
        self.h_ctx = Some(self.drafter.project_condition(&r0_hidden)?);
        self.probe_seed = Some(r0_hidden);

        let seed = ctx.draw.seed_token(&r0_logits, device)?;
        Ok(Prefilled {
            seed,
            prefill_ns,
            // It projects its committed rows every round and the loop counts
            // them. The round-0 row is the starting buffer, not a row a round
            // projected, so the count still opens at zero.
            projects_conditioning: true,
        })
    }

    /// The loop's narrowing, moved by the accept rate of the recent rounds.
    fn block(&self, block_total: usize, remaining: usize) -> usize {
        dflash_next_block_size(&self.recent, block_total, remaining, false)
    }

    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>> {
        self.drafter
            .draft_block(ctx.verifier, carry, self.conditioning()?, block)
    }

    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict> {
        let device = ctx.device;
        // The verifier scores the carry token and every proposal in one pass,
        // capturing the conditioning hidden for the same positions.
        let t0 = Instant::now();
        let (v_logits, v_hidden) = ctx.verifier.forward_verify_capture(
            fed,
            fed.len(),
            &self.target_layer_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            device,
        )?;
        let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;
        let verify_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let proposals = fed.get(1..).unwrap_or_default();
        let (accept, commit) = accept_prefix(&v_tokens, proposals, remaining)?;
        let walk_ns = t0.elapsed().as_nanos();
        // What the next round's block is chosen from. Written here, where the
        // round's acceptance is known, and read by `block` at the head of the
        // round after it.
        self.recent.push((accept, proposals.len()));

        self.scored = Some(v_hidden);
        Ok(Verdict {
            accept,
            commit,
            verify_ns,
            walk_ns,
            // Every position scored over the verifier's whole vocabulary.
            restricted: 0,
        })
    }

    /// Extend the carried projection by this round's committed capture rows.
    ///
    /// The round conditions on what it **committed**, which the request's
    /// remaining budget can cut below the acceptance — the count is the
    /// caller's and `committed_rows` takes it from the head of the capture.
    fn condition(
        &mut self,
        ctx: &RoundCtx<'_>,
        _verdict: &Verdict,
        outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>> {
        let device = ctx.device;
        let Some(scored) = self.scored.take() else {
            return Err(Error::Model(
                "dflash_generate: a round conditioned on a verify forward that did not run".into(),
            ));
        };
        let committed_hidden =
            committed_rows(&scored, outcome.committed, self.condition_width, device)?;
        let (h_ctx, projected_rows) =
            self.drafter
                .grow_conditioning(self.conditioning()?, &committed_hidden, device)?;
        guard_round_conditioning(outcome.round, projected_rows, outcome.committed)?;
        if let Some(seed) = self.probe_seed.take() {
            let raw = concatenate(&[&seed, &committed_hidden], 1, device)?;
            let fresh = self.drafter.project_condition(&raw)?;
            report_conditioning_residual(
                SpecLoop::DFlash,
                &h_ctx,
                &fresh,
                projected_rows,
                self.drafter.cfg.hidden_size as i32,
                device,
            )?;
        }
        let rows = h_ctx.shape().get(1).copied();
        self.h_ctx = Some(h_ctx);
        Ok(Some(Conditioning {
            rows,
            projected: Some(projected_rows),
        }))
    }

    fn carry(&self, f: &mut dyn FnMut(&[(&str, &Array)])) -> Result<()> {
        f(&[("h_ctx", self.conditioning()?)]);
        Ok(())
    }
}

/// Drive a DFlash 1 drafter against its verifier.
///
/// `requested_block_total` is the round block including the verifier's own
/// token; it is clamped to the block the drafter was trained at, which is the
/// ceiling the adaptive schedule grows back to.
///
/// `sampler_cfg` decides what "the verifier's own token" means at each position
/// — its argmax at temperature 0, a draw from its post-sampling distribution
/// above it. The drafter's block is unaffected: it denoises the same masked
/// positions either way, and [`crate::speculative::VerifierDraw`] explains why
/// the walk is still exact.
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
/// checked, and this loop's schedule moves the width within a run.
///
/// # Errors
///
/// [`Error::Model`] when the prompt is too short to seed a round, when the
/// verifier carries no recurrent state, and whatever the round loop refuses.
#[allow(clippy::too_many_arguments)]
pub fn dflash_generate(
    verifier: &Architecture,
    drafter: &DFlashDrafter,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    requested_block_total: usize,
    kv_quant_override: Option<KvQuant>,
    max_ctx_override: Option<i32>,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<(Vec<ProbeStep>, usize)> {
    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "dflash_generate: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "dflash_generate: DFlash verifier must be the Qwen3.5/3.6-MoE \
             hybrid (needs GDN lin_caches)"
                .into(),
        ));
    }
    let block_total = block_capped_by_checkpoint(requested_block_total, drafter.cfg.block_size);
    let mut round = AdaptiveRound::new(drafter);
    run_rounds(
        verifier,
        &mut round,
        prompt_ids,
        step_fn,
        &RoundCfg {
            loop_kind: SpecLoop::DFlash,
            block_size: block_total,
            n_tokens,
            eos_ids,
            tokenizer,
            sampler_cfg,
            // This request does not charge its phases for the work they issue:
            // its rounds leave the conditioning they project for the next
            // round's drafter to force, and the timings say so.
            charged: false,
            kv_quant_override,
            max_ctx_override,
        },
        // This drafter's verify pass scores every position over the verifier's
        // whole vocabulary, so its tokens need no per-token attribution.
        None,
        device,
    )
}

#[cfg(test)]
#[path = "round_tests.rs"]
mod round_tests;
