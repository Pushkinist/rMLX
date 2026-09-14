//! The DFlash 2 drafter's side of the shared round loop.
//!
//! Ported from the z-lab MLX reference `_stream_generate`. Verifier-side it is
//! the shape every sidecar drafter in this module family has — prefill, a bonus
//! token out of the prefill forward, then rounds of draft / verify / accept /
//! roll back — and that shape is [`run_rounds`]. What is here is the four
//! methods the loop cannot supply: the whole-prompt capture its prefill takes,
//! the block its denoiser drafts, the acceptance walk over the verify pass, and
//! the conditioning window it slides.
//!
//! # Conditioning: the K/V is recomputed, the projection is carried
//!
//! The reference gives its drafter a per-layer rotating K/V cache and feeds it
//! only each round's newly committed rows. This drafter rebuilds that K/V from
//! the conditioning rows on every call instead. The two are the same answer —
//! the cached rows are a deterministic function of those rows — and the
//! recomputing form is what makes the drafter forward invariant to a uniform
//! shift of every position, which is why it needs no absolute offset. Adopting
//! the cache would mean cached rows carry their own absolute RoPE, and that
//! invariance, and the proof that rests on it, would have to be rebuilt.
//!
//! One step before that K/V is [`DFlash2Drafter::project_conditioning`], which
//! is row-wise and carries no position at all, and that is what this drafter
//! keeps: it projects each round's committed capture rows as it commits them
//! and carries the projection. Re-projecting the whole window every round would
//! give the same rows for work proportional to the window.
//!
//! The buffer is bounded, not accumulated: the drafter attends over one sliding
//! window, so rows older than it can never be read and are dropped as they fall
//! out. Each carried row is `hidden` wide — 10 KiB on the published pair, where
//! the capture it was projected from is 50 KiB.
//!
//! # Greedy only
//!
//! The reference's sampled arm is rejection sampling restricted to the
//! selector's own candidate set, which is a different acceptance rule from the
//! full-vocabulary one this crate's two-model loop implements, and
//! [`DFlash2Drafter::select_chain`] traces a greedy chain and returns no
//! candidate distribution to sample against. Like every other sidecar drafter
//! here this one is greedy, and the serve layer routes a sidecar request to it
//! whatever the request's temperature.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{concatenate, Array, Device};

use super::DFlash2Drafter;
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use crate::speculative::round_loop::{
    run_rounds, Conditioning, Prefilled, ReportSkippedBy, RoundCfg, RoundCtx, RoundDrafter,
    RoundOutcome, Verdict, VerifierOffsetBasis,
};
use crate::speculative::{
    accept_prefix, block_capped_by_checkpoint, committed_rows, conditioning_residual,
    guard_round_conditioning, guard_verifier_prefill_logits, phases_charged, SpecLoop,
};
use rmlx_kv_quant::KvQuant;

/// Prompt positions per verifier prefill pass.
///
/// The capture returns one hidden row per prompt position, so a single-shot
/// prefill of a long prompt would put the whole capture and a full-vocabulary
/// logit tensor in one Metal command buffer.
pub(super) const PREFILL_CHUNK_SIZE: usize = 1024;

/// One request's DFlash 2 drafting state.
///
/// The drafter keeps no cache of its own — the denoiser re-reads the carried
/// projection every round — so what crosses a round boundary is that projection
/// and the verify pass's capture the next one is built from.
#[allow(missing_debug_implementations)]
pub(crate) struct BlockRound<'a> {
    drafter: &'a DFlash2Drafter,
    /// The verifier layers whose hidden the conditioning is projected from.
    target_layer_ids: Vec<usize>,
    /// One capture row's width, `len(target_layer_ids) * hidden_size`.
    condition_width: i32,
    /// The projection the next round's denoiser conditions on.
    h_ctx: Option<Array>,
    /// The prompt's last raw row, kept until the first round has extended the
    /// buffer and then dropped: re-projecting it beside that round's commit is
    /// what says how far the carried projection sits from a fresh one at this
    /// checkpoint's dtype. One row, one round, and the probe releases it. See
    /// `conditioning_residual`.
    probe_seed: Option<Array>,
    /// The verifier's capture at every position this round verified. A round
    /// that stopped on an EOS leaves it set — the loop breaks before
    /// `condition` — and the request drops this struct immediately after, so it
    /// is released there rather than cleared on that path.
    scored: Option<Array>,
}

impl<'a> BlockRound<'a> {
    fn new(drafter: &'a DFlash2Drafter) -> Self {
        let target_layer_ids = drafter.cfg.target_layer_ids.clone();
        let condition_width = (drafter.cfg.hidden_size * target_layer_ids.len()) as i32;
        Self {
            drafter,
            target_layer_ids,
            condition_width,
            h_ctx: None,
            probe_seed: None,
            scored: None,
        }
    }

    /// The projection a round conditions on, or the refusal for a round that
    /// reached for it before the prefill built one.
    fn conditioning(&self) -> Result<&Array> {
        self.h_ctx.as_ref().ok_or_else(|| {
            Error::Model(
                "dflash2_generate: a round read the projection it conditions on before \
                 the prefill built one"
                    .into(),
            )
        })
    }
}

impl RoundDrafter for BlockRound<'_> {
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy = ReportSkippedBy::TheSeedExit;
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis = VerifierOffsetBasis::AfterTheForward;

    /// Prefill the whole prompt, keeping the conditioning hidden of as many of
    /// its positions as the drafter's window reaches back over — the depth the
    /// reference conditions its first round on, not the last token alone. The
    /// rows before that can never be read, so the capture releases them as the
    /// prefill walks the prompt rather than holding one row per prompt token at
    /// `len(target_layer_ids) * hidden_size` to the end of it.
    #[allow(
        clippy::indexing_slicing,
        reason = "every index is axis 1 of a rank-3 buffer whose rank the seam that produced it checked: the trim and the row it takes the probe from"
    )]
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled> {
        let device = ctx.device;
        let keep_rows = self.drafter.conditioning_rows();
        // Defence in depth: `check_config` refuses a window under 2 or wider
        // than an array axis, so this conversion cannot fail on a drafter that
        // loaded.
        let keep = usize::try_from(keep_rows).map_err(|_| {
            Error::Model(format!(
                "dflash2_generate: this drafter reaches back over {keep_rows} rows, \
                 which is not a row count"
            ))
        })?;
        let prefill_t0 = Instant::now();
        let (bonus_logits, prompt_hidden) = ctx.verifier.forward_verify_capture_chunked(
            prompt,
            &self.target_layer_ids,
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            PREFILL_CHUNK_SIZE,
            Some(keep),
            device,
        )?;
        guard_verifier_prefill_logits(ctx.verifier, &bonus_logits, prompt.len())?;
        let prompt_ctx = self.drafter.trim_conditioning(&prompt_hidden)?;
        // What the capture was asked to keep is a number at a call site, and
        // the trim above accepts a shorter buffer without a word. A drafter
        // conditioned on less than its window still proposes, and greedy
        // verification then emits the verifier's own tokens whatever the
        // proposals were — so what falls is the accept rate, and nothing here
        // would say so.
        let want_rows = keep_rows.min(i32::try_from(prompt.len()).unwrap_or(i32::MAX));
        if prompt_ctx.shape()[1] != want_rows {
            return Err(Error::Model(format!(
                "dflash2_generate: the prompt's capture kept {} conditioning rows where this \
                 drafter reaches back over {want_rows} of a {}-token prompt",
                prompt_ctx.shape()[1],
                prompt.len()
            )));
        }
        // The projection is carried, not the capture it came from: `fc` and
        // `hidden_norm` are row-wise, so a row projected once here is the row
        // every later round would have re-derived. The prompt's rows are
        // projected in this one call and each round then projects only what it
        // commits.
        let h_ctx = self.drafter.project_conditioning(&prompt_ctx)?;
        self.probe_seed = Some(committed_rows(
            &prompt_ctx.slice(
                &[0, prompt_ctx.shape()[1] - 1, 0],
                &[1, prompt_ctx.shape()[1], self.condition_width],
                &[1, 1, 1],
                device,
            )?,
            1,
            self.condition_width,
            device,
        )?);
        if ctx.charged {
            // The guard above forced the logits, and so the whole prompt
            // forward, but not the capture: it hangs off a different output of
            // that forward. Joining the kept chunks, trimming to the window and
            // projecting it is a pass over one row per kept position — and with
            // nothing forcing it here the first round's drafter call pays for
            // all of it. See `phases_charged`.
            h_ctx.eval()?;
        }
        let prefill_ns = prefill_t0.elapsed().as_nanos();
        self.h_ctx = Some(h_ctx);

        let seed = ctx.draw.seed_token(&bonus_logits, device)?;
        Ok(Prefilled {
            seed,
            prefill_ns,
            // It projects its committed rows every round and the loop counts
            // them. The prompt's rows are the starting window, not rows a round
            // projected, so the count still opens at zero.
            projects_conditioning: true,
        })
    }

    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>> {
        // The block never resizes: the drafter denoises the block it was
        // trained at, and only the token budget shortens it — which is the
        // narrowing the loop already applied.
        draft_block(
            ctx.verifier,
            self.drafter,
            carry,
            self.conditioning()?,
            block,
            ctx.device,
        )
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
        if ctx.charged {
            // Reading the verifier's tokens forced the logits and the trunk
            // under them, but the capture is a different output of the same
            // forward and stays lazy. The next round's drafter reads it, so
            // with nothing forcing it here the verifier's own capture is billed
            // to the drafter — and this drafter's capture is
            // `len(target_layer_ids)` times as wide as the sidecar loops',
            // which is the comparison that would mislead. See `phases_charged`.
            v_hidden.eval()?;
        }
        let verify_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let proposals = fed.get(1..).unwrap_or_default();
        let (accept, commit) = accept_prefix(&v_tokens, proposals, remaining)?;
        let walk_ns = t0.elapsed().as_nanos();

        self.scored = Some(v_hidden);
        Ok(Verdict {
            accept,
            commit,
            verify_ns,
            walk_ns,
        })
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "axis 1 of a rank-3 buffer whose rank slide_conditioning established, and the width the drafter's fc reads"
    )]
    fn condition(
        &mut self,
        ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>> {
        let device = ctx.device;
        let Some(scored) = self.scored.take() else {
            return Err(Error::Model(
                "dflash2_generate: a round conditioned on a verify forward that did not run".into(),
            ));
        };
        // The carry token and the accepted proposals.
        let committed_hidden =
            committed_rows(&scored, verdict.accept + 1, self.condition_width, device)?;
        let (h_ctx, projected_rows) = self
            .drafter
            .slide_conditioning(self.conditioning()?, &committed_hidden)?;
        guard_round_conditioning(outcome.round, projected_rows, verdict.accept + 1)?;
        if let Some(seed) = self.probe_seed.take() {
            let raw = concatenate(&[&seed, &committed_hidden], 1, device)?;
            let fresh = self.drafter.project_conditioning(&raw)?;
            let hidden = self.drafter.cfg.hidden_size as i32;
            let tail = h_ctx.shape()[1] - (1 + projected_rows);
            let carried_tail = h_ctx.slice(
                &[0, tail, 0],
                &[1, tail + 1 + projected_rows, hidden],
                &[1, 1, 1],
                device,
            )?;
            tracing::debug!(
                rows = 1 + projected_rows,
                residual = conditioning_residual(&carried_tail, &fresh, device)?,
                "dflash2 conditioning: carried projection against a fresh one"
            );
        }
        if ctx.charged {
            // Projecting this round's committed rows and copying the window
            // they extend is work the next round's drafter is the first thing
            // to read. Forced here it lands in `rollback_ms`, which the loop
            // closes after this call returns; left lazy it lands in the
            // drafter. See `phases_charged`.
            h_ctx.eval()?;
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

/// Drive a DFlash 2 drafter against its verifier.
///
/// `requested_block_total` is the round block including the verifier's own
/// token; it is clamped to the block the drafter was trained at, which is the
/// widest one its selector chain is defined over.
///
/// `sampler_cfg` decides what "the verifier's own token" means at each position
/// — its argmax at temperature 0, a draw from its post-sampling distribution
/// above it. The drafter's selector chain is unaffected: it proposes ids either
/// way, and [`crate::speculative::VerifierDraw`] explains why the walk is still
/// exact.
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
/// checked, and every round narrows the block again against the remaining token
/// budget.
///
/// # Errors
///
/// [`Error::Model`] when the prompt is too short to seed a round, when the
/// verifier is not one this drafter's seams are wired for, and whatever the
/// round loop refuses.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_generate(
    verifier: &Architecture,
    drafter: &DFlash2Drafter,
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
            "dflash2_generate: prompt must have >=2 tokens".into(),
        ));
    }
    // The recurrent state is a proxy, and an exact one today: the three seams
    // this drafter needs — the multi-layer hidden capture, the raw embedding and
    // the LM head over a final-normed hidden — are wired for the Qwen3.5 family
    // and nothing else, and that is the family that carries recurrent state.
    // Each of those seams refuses an architecture it is not wired for, so an
    // arch that gained the state without the seams would still fail rather than
    // draft — at the first block, after a whole prompt has been prefilled, which
    // is the only thing this check buys over letting them refuse.
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "dflash2_generate: this verifier carries no recurrent state, so it \
             is not the Qwen3.5/3.6 hybrid the drafter's seams are wired for — its \
             multi-layer hidden capture, its raw embedding and its LM head over a \
             final-normed hidden"
                .into(),
        ));
    }
    let block_total = block_capped_by_checkpoint(requested_block_total, drafter.cfg.block_size);
    // One read of process-global log state per request, at the entry.
    let charge_phases = phases_charged();
    let mut round = BlockRound::new(drafter);
    run_rounds(
        verifier,
        &mut round,
        prompt_ids,
        step_fn,
        &RoundCfg {
            loop_kind: SpecLoop::DFlash2,
            block_size: block_total,
            n_tokens,
            eos_ids,
            tokenizer,
            sampler_cfg,
            charged: charge_phases,
            kv_quant_override,
            max_ctx_override,
        },
        device,
    )
}

/// One block of `bs - 1` proposals: mask the block behind the carry token,
/// denoise it, and trace the chain the selector picks out of it.
///
/// The block's embeddings come from the **verifier's** input embedding and its
/// unary logits from the verifier's LM head; the drafter has neither of its own.
fn draft_block(
    verifier: &Architecture,
    drafter: &DFlash2Drafter,
    seed: u32,
    h_ctx: &Array,
    bs: usize,
    device: Device,
) -> Result<Vec<u32>> {
    let hidden_size = drafter.cfg.hidden_size as i32;
    // Position 0 of the block *is* the last verified token, which is what the
    // first drafted position's convolution reads back over.
    let mut block_ids: Vec<i32> = Vec::with_capacity(bs);
    block_ids.push(seed as i32);
    block_ids.resize(bs, drafter.cfg.mask_token_id as i32);

    let block = verifier.embed_tokens_raw(&block_ids, device)?;
    let hidden = drafter.forward_hidden_conditioned(&block, h_ctx)?;
    // Row 0 is the seed, which is not drafted.
    let drafted = hidden.slice(&[0, 1, 0], &[1, bs as i32, hidden_size], &[1, 1, 1], device)?;
    let logits = verifier.logits_from_final_hidden(&drafted, device)?;
    drafter.select_chain(&drafted, &logits, seed)
}

#[cfg(test)]
#[path = "round_tests.rs"]
mod round_tests;
