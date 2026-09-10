//! The one speculative round loop, and the interface a drafter meets it at.
//!
//! Every round loop in this directory runs one algorithm: refuse a prompt under
//! two tokens, resolve the block, build the verifier's caches, prefill, emit a
//! seed or not, then per round narrow the block against the remaining budget,
//! propose, verify, walk the acceptance, emit what the round committed, roll the
//! verifier's caches back to the accepted prefix, roll the drafter's own state
//! back, condition the next round and log one round event; then one request
//! record, one report of the verifier's resident KV, and a block figure the
//! caller reads back.
//!
//! What differs between drafters is how one proposes, what it conditions on, how
//! it rolls its own state back and which of the verifier's outputs it captures.
//! Those are [`RoundDrafter`]'s methods; everything above is [`run_rounds`].
//!
//! See `docs/SPEC_ROUND_SKELETON.md` for the interface's rationale and for the
//! per-loop behaviour it is required to preserve.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use super::round_common::{
    emit_round_tokens, emit_seed_token, lin_cache_stack, log_request_record,
    report_verifier_kv_bytes, rollback_round, verifier_cache_stack, RoundTotals,
};
use super::{DecodeWindow, SpecLoop, VerifierDraw};
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

/// Which of the loop's two exits skips the report of the verifier's resident KV.
///
/// The report has one writer — a speculative request never goes through
/// `Architecture::generate_greedy` — so a request that skips it leaves the
/// previous request's figure readable. Which exit skips it is the drafter's own
/// fact and the one thing the shared loop cannot derive, so each drafter
/// declares it and the loop reads the declaration at both exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReportSkippedBy {
    /// The seed EOS: the drafter emits a token out of its prefill forward, and a
    /// request whose whole output is that token returns before any round.
    TheSeedExit,
    /// The in-round EOS: the drafter emits nothing before its first round, so
    /// the only early exit it has is inside one.
    #[allow(
        dead_code,
        reason = "declared by the two two-model loops, which have not migrated onto this \
                  loop yet; the seven-row disposition table in `round_skeleton_tests.rs` \
                  states it for them until they do"
    )]
    TheInRoundExit,
}

/// What one request runs at, decided by the drafter's own entry function.
///
/// The charge decision travels here as a value rather than being made in the
/// loop: the entry names it once, beside the record that has to report the same
/// one. `make check-spec-charge` reads both ends.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoundCfg<'a> {
    /// Which drafter's request this is, for the round line and the record.
    pub(crate) loop_kind: SpecLoop,
    /// The block the request was configured with, verifier token included.
    pub(crate) block_size: usize,
    /// Tokens the request may emit.
    pub(crate) n_tokens: usize,
    /// Token ids that end the request.
    pub(crate) eos_ids: &'a [u32],
    /// Detokenizer for the emitted steps.
    pub(crate) tokenizer: &'a tokenizers::Tokenizer,
    /// What "the verifier's own token" means — its argmax at temperature 0, a
    /// draw from its post-sampling distribution above it.
    pub(crate) sampler_cfg: &'a crate::sampler::SamplerConfig,
    /// Whether the request's phases are charged for the work they issue.
    pub(crate) charged: bool,
    /// The request's KV codec, or the verifier's own default.
    pub(crate) kv_quant_override: Option<KvQuant>,
    /// The request's context ceiling, or the verifier's own.
    pub(crate) max_ctx_override: Option<i32>,
}

/// The verifier, its caches and the draw a round reads them through.
///
/// [`RoundDrafter::rollback`] and [`RoundDrafter::condition`] take this
/// immutably, so only `prefill` and `verify` can advance the caches or the draw.
#[allow(missing_debug_implementations)]
pub(crate) struct RoundCtx<'a> {
    /// The model a round verifies against.
    pub(crate) verifier: &'a Architecture,
    /// Its per-layer attention caches.
    pub(crate) kv: Vec<KvCache>,
    /// Its recurrent linear-attention state, on an architecture that keeps one.
    pub(crate) lin: Option<Vec<LinearAttnCache>>,
    /// The request's draw over the verifier's logits.
    pub(crate) draw: VerifierDraw,
    /// The loop's phase-charge decision, for a drafter that forces the arrays it
    /// carries into the next round.
    pub(crate) charged: bool,
    pub(crate) device: Device,
}

/// What a drafter's prefill left the request with.
#[allow(missing_debug_implementations)]
pub(crate) struct Prefilled {
    /// The token the request emits before its first round.
    pub(crate) seed: u32,
    /// How long the prefill took, timed by the drafter: what the span covers
    /// differs per drafter, and a loop that timed the call would move the figure
    /// on records no gate reads.
    pub(crate) prefill_ns: u128,
    /// Conditioning rows the drafter projects, or `None` for a drafter that
    /// projects none. A statement made before any round has run, which is what
    /// the record of a request that stopped on its seed reports.
    pub(crate) conditioned_rows: Option<usize>,
}

/// What a round's verify forward decided.
#[allow(missing_debug_implementations)]
pub(crate) struct Verdict {
    /// Proposals the verifier accepted.
    pub(crate) accept: usize,
    /// The tokens the round commits: the agreed prefix and the one token the
    /// verifier stands behind, less anything the request's budget cut.
    pub(crate) commit: Vec<u32>,
    /// Wall clock of the verify forward and its read-back.
    pub(crate) verify_ns: u128,
    /// Wall clock of the acceptance walk.
    pub(crate) walk_ns: u128,
}

/// What a drafter does that the shared loop cannot.
pub(crate) trait RoundDrafter {
    /// Which of the loop's two exits skips the report of the verifier's
    /// resident KV.
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy;

    /// Prefill the prompt and produce the token the request emits before its
    /// first round.
    ///
    /// # Errors
    /// Whatever the verifier's prefill or the seed draw refuses.
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled>;

    /// The block this round runs, narrowed against the remaining budget.
    fn block(&self, block_total: usize, remaining: usize) -> usize {
        super::round_block(block_total, remaining)
    }

    /// This round's proposals. An empty chain is the loop's refusal, not this
    /// one's.
    ///
    /// # Errors
    /// Whatever the drafting forward refuses.
    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>>;

    /// Score the round and say what it commits.
    ///
    /// # Errors
    /// Whatever the verify forward or the acceptance walk refuses.
    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict>;

    /// Return the drafter's own state to the accepted prefix. The default is a
    /// drafter that keeps nothing across rounds.
    ///
    /// # Errors
    /// Whatever the drafter's own rollback refuses.
    fn rollback(
        &mut self,
        _ctx: &RoundCtx<'_>,
        _verdict: &Verdict,
        _verifier_target: i32,
    ) -> Result<()> {
        Ok(())
    }

    /// Condition the next round on what this one committed.
    ///
    /// # Errors
    /// Whatever building the next round's conditioning refuses.
    fn condition(
        &mut self,
        ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        verifier_target: i32,
    ) -> Result<()>;

    /// Every array this round leaves for the next round's drafter to read, each
    /// under the name this drafter calls it. On a charged round they must all be
    /// forced by the time the round's line is written.
    fn carry(&self) -> Vec<(&str, &Array)>;
}

/// Run one speculative request's rounds.
///
/// Returns the emitted steps and the widest block any round of this run actually
/// ran — not the block resolved before the loop, which a caller checking what it
/// asked for would be trusting the very step it wanted checked. The seed exit is
/// the exception and returns the resolved block: no round ran there.
///
/// # Errors
///
/// [`Error::Model`] when a drafter proposes nothing, and whatever the verifier's
/// forwards, the acceptance walk or the rollback refuse.
pub(crate) fn run_rounds<D: RoundDrafter>(
    verifier: &Architecture,
    drafter: &mut D,
    prompt_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    cfg: &RoundCfg<'_>,
    device: Device,
) -> Result<(Vec<ProbeStep>, usize)> {
    let charge = cfg.charged;
    let n_tokens = cfg.n_tokens;
    let (kv_quant, _, kv) =
        verifier_cache_stack(verifier, cfg.kv_quant_override, cfg.max_ctx_override)?;
    let lin = verifier
        .needs_lin_caches()
        .then(|| lin_cache_stack(verifier));
    let mut ctx = RoundCtx {
        verifier,
        kv,
        lin,
        draw: VerifierDraw::new(cfg.sampler_cfg),
        charged: charge,
        device,
    };

    let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);
    let mut window = DecodeWindow::new();
    let t_total = Instant::now();
    let mut rounds = 0usize;
    let mut total_draft = 0usize;
    let mut total_accept = 0usize;
    let mut draft_ns: u128 = 0;
    let mut verifier_ns: u128 = 0;

    let prefilled = drafter.prefill(&mut ctx, prompt_ids)?;
    let prefill_ns = prefilled.prefill_ns;
    let conditioned_rows = prefilled.conditioned_rows;
    let mut carry = prefilled.seed;

    if emit_seed_token(
        cfg.tokenizer,
        carry,
        step_fn,
        &mut emitted,
        &mut window,
        cfg.eos_ids,
        &RoundTotals {
            loop_kind: cfg.loop_kind,
            block_size: cfg.block_size,
            conditioned_rows,
            charged: charge,
            // No round ran.
            rounds: 0,
            emitted_in_rounds: 0,
            total_draft: 0,
            total_accept: 0,
            prefill_ns,
            draft_ns: 0,
            verifier_ns: 0,
            round_loop_ns: 0,
            t_total,
        },
    ) {
        if !matches!(D::KV_REPORT_SKIPPED_BY, ReportSkippedBy::TheSeedExit) {
            report_verifier_kv_bytes(verifier, &ctx.kv, ctx.lin.as_deref());
        }
        return Ok((emitted, cfg.block_size));
    }

    tracing::info!(
        ?cfg.loop_kind,
        block_size = cfg.block_size,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        temperature = cfg.sampler_cfg.temperature,
        "speculative round loop: starting"
    );

    let seed_emitted = emitted.len();
    let mut emitted_in_rounds = 0usize;
    let mut widest_bs = 0usize;
    let mut stopped_in_round = false;
    let round_loop_t0 = Instant::now();
    while emitted.len() < n_tokens {
        let round_t0 = Instant::now();
        rounds += 1;
        let remaining = n_tokens - emitted.len();
        let bs = drafter.block(cfg.block_size, remaining);
        widest_bs = widest_bs.max(bs);

        let t0 = Instant::now();
        let draft_tokens = drafter.propose(&mut ctx, carry, bs)?;
        let round_draft_ns = t0.elapsed().as_nanos();
        draft_ns += round_draft_ns;
        if draft_tokens.is_empty() {
            return Err(Error::Model(format!(
                "{:?}: the drafter proposed nothing at block {bs}; a drafter returns \
                 block - 1 ids for any block of two or more, so an empty chain is a \
                 broken drafter and not the end of the request",
                cfg.loop_kind
            )));
        }
        total_draft += draft_tokens.len();

        let mut fed: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        fed.push(carry);
        fed.extend_from_slice(&draft_tokens);

        // Read before the forward: it is the position the rollback returns to,
        // and reading it afterwards makes it a function of how far each layer
        // happened to advance.
        let pre_round_offset = ctx.kv.iter().map(KvCache::offset).max().unwrap_or(0);

        let verdict = drafter.verify(&mut ctx, &fed, remaining)?;
        verifier_ns += verdict.verify_ns;
        total_accept += verdict.accept;

        let emit = emit_round_tokens(
            cfg.tokenizer,
            &verdict.commit,
            n_tokens,
            cfg.eos_ids,
            step_fn,
            &mut emitted,
            &mut emitted_in_rounds,
            &mut window,
            None,
        );
        if emit.hit_eos {
            stopped_in_round = true;
            break;
        }

        let t0 = Instant::now();
        let v_target = super::rollback_target_from_head(pre_round_offset, verdict.accept);
        let refolded = rollback_round(
            &mut ctx.kv,
            ctx.lin.as_deref_mut(),
            &fed,
            pre_round_offset,
            v_target,
            charge,
            device,
        )?;
        drafter.rollback(&ctx, &verdict, v_target)?;
        drafter.condition(&ctx, &verdict, v_target)?;
        carry = verdict.commit.last().copied().unwrap_or(carry);
        let round_rollback_ns = t0.elapsed().as_nanos();

        super::log_round(
            &super::RoundReport {
                loop_kind: cfg.loop_kind,
                round: rounds,
                accept: verdict.accept,
                num_draft: draft_tokens.len(),
                n_committed: emit.committed,
                emitted_total: emitted.len(),
                condition_rows: None,
                projected_rows: None,
                v_offset_before: pre_round_offset,
                v_target,
                d_offset_before: None,
                d_target: None,
                refolded,
                charged: charge,
                phases: Some(super::RoundPhases {
                    round_ns: round_t0.elapsed().as_nanos(),
                    draft_ns: round_draft_ns,
                    verify_ns: verdict.verify_ns,
                    walk_ns: verdict.walk_ns,
                    rollback_ns: round_rollback_ns,
                }),
            },
            &drafter.carry(),
        );
    }

    let round_loop_ns = round_loop_t0.elapsed().as_nanos();
    log_request_record(
        &RoundTotals {
            loop_kind: cfg.loop_kind,
            block_size: cfg.block_size,
            conditioned_rows,
            charged: charge,
            rounds,
            emitted_in_rounds,
            total_draft,
            total_accept,
            prefill_ns,
            draft_ns,
            verifier_ns,
            round_loop_ns,
            t_total,
        },
        &emitted,
        seed_emitted,
        &window,
    );
    if !(stopped_in_round && matches!(D::KV_REPORT_SKIPPED_BY, ReportSkippedBy::TheInRoundExit)) {
        report_verifier_kv_bytes(verifier, &ctx.kv, ctx.lin.as_deref());
    }
    Ok((emitted, widest_bs))
}
