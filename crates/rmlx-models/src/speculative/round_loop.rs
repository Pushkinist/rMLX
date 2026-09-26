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
//! Those are [`RoundDrafter`]'s methods; the rest is [`run_rounds`], with two
//! exceptions at the front. The prompt refusal and the block resolution stay in
//! each drafter's own entry, which runs before the loop: both refuse before a
//! cache stack is built, and the loop is handed a block that is already
//! resolved.
//!
//! See `docs/SPEC_ROUND_SKELETON.md` for the interface's rationale and for the
//! per-loop behaviour it is required to preserve.
//!
//! [`run_rounds`] cannot have a unit test, for the reason `round_common.rs`
//! gives about its own cache-stack functions: it takes an [`Architecture`],
//! which is only reachable by loading weights. Four readings hold it instead,
//! each blind to something different — the equivalence pair
//! `the_assistant_round_loop_reproduces_plain_greedy` in
//! `crates/rmlx-models/tests/spec_greedy_equivalence.rs`, the per-round event
//! stream that run writes against
//! `crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256`, the
//! edge-marker reading of this file in `round_skeleton_tests.rs`, and the text
//! scans `make check-spec-charge` and `make check-spec-sampling`.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use super::round_common::{
    emit_round_tokens, emit_seed_token, lin_cache_stack, log_request_record,
    report_verifier_kv_bytes, rollback_round, verifier_cache_stack, RoundTotals,
};
use super::{guard_restricted_prefix, DecodeWindow, SpecLoop, VerifierDraw};
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

/// Which vocabulary decided one emitted token.
///
/// A drafter whose verify pass scores some positions over a vocabulary smaller
/// than the verifier's emits tokens the verifier's own argmax would not always
/// have chosen: a reduced argmax equals the true one exactly when the true one
/// is in the reduced set. This is the only thing that says whether the
/// reduction could have changed a token, and the token stream cannot express
/// it. Produced here — the loop writes one entry per token it emits, from the
/// round's [`Verdict::restricted`] — and read by the answer-equivalence gate.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed two-valued distinction: a verify position is scored over a reduced vocabulary or over the verifier's, and there is no third"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecidedBy {
    /// A reduced vocabulary — an accepted draft token. It differs from the
    /// verifier's own argmax exactly when that argmax is a token the reduced
    /// set cannot name.
    RestrictedVocab,
    /// The verifier's whole vocabulary — a round's correction, the prefill
    /// seed, or any position of a request that takes no reduced read-back at
    /// all. The reduction cannot have changed this token.
    FullVocab,
}

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
    TheInRoundExit,
}

/// Which read of the verifier's offset a round reports as the one its rollback
/// target was computed from.
///
/// The two reads name the same position under two different numbers: a drafter
/// that counts its rollback back from the tail reads the offset after its
/// verify forward, one that counts forward over the accepted prefix reads it
/// before. Both numbers are on the pinned round line, so which one a drafter
/// reports is its own fact and it declares it; the loop computes the rollback
/// from the head spelling whichever is declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerifierOffsetBasis {
    /// The offset before the verify forward, which is where the rollback
    /// returns to.
    BeforeTheForward,
    /// The offset after it, from which the rejected tail is counted back.
    AfterTheForward,
}

/// Where a drafter's own cache stood at the head of a round, and where the
/// round's drafter-side rollback left it.
///
/// Built by the drafter and reported by the loop. That is what keeps a field
/// added here a compile error at every drafter that keeps a cache, and no error
/// at all in the ones that keep none — they answer `None` and owe no value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CacheSpan {
    /// The drafter cache offset the rollback was read against.
    pub(crate) before: i32,
    /// Where the rollback left that cache. Computed on the drafter's own path,
    /// so it is a cross-check on the verifier's target rather than a
    /// restatement of it.
    pub(crate) target: i32,
}

/// What a round left for the next round's drafter to condition on.
///
/// Paired with [`CacheSpan`] for the same reason: the drafter that builds the
/// buffer is the one that can say how many rows it holds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conditioning {
    /// Rows the buffer holds leaving this round, read off the buffer rather
    /// than off what the round meant to put in it.
    pub(crate) rows: Option<i32>,
    /// Rows this round's projection returned, or `None` for a drafter that
    /// slices a row rather than projecting one.
    pub(crate) projected: Option<i32>,
}

/// What the round committed, and where the loop's own rollback left the
/// verifier.
///
/// Handed to the drafter's rollback and conditioning because both run after the
/// emission and the loop is the one place each of these is known: the committed
/// count is what the request's budget left of the acceptance, the verifier
/// target is the loop's own rollback, and the round index is the loop's counter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoundOutcome {
    /// This round's index, counting from one.
    ///
    /// The loop's own counter rather than a second one kept per drafter: a
    /// drafter that counted its own rounds could drift from the loop's, and the
    /// only thing that reads this is the refusal message a conditioning guard
    /// writes, where a wrong index is a reader sent to the wrong round with
    /// nothing to catch it.
    pub(crate) round: usize,
    /// Tokens the round handed the sink — the accepted prefix and the
    /// verifier's own token, less anything the budget cut. A drafter that keeps
    /// a drafting position advances it by this.
    pub(crate) committed: usize,
    /// Where the loop's rollback left the verifier's caches.
    pub(crate) verifier_target: i32,
    /// The token the next round carries into its verify input: the verifier's
    /// own token at the accepted position, read off the commit.
    ///
    /// The loop's, and the loop's alone. A drafter that opens its next round on
    /// this token can derive the same value from the same `Verdict`, and two
    /// producers of one token drift into a wrong drafting seed — which moves the
    /// accept rate and nothing else, so no equivalence pair and no pinned cell
    /// can see it.
    pub(crate) carry: u32,
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
/// immutably, so only `prefill`, `propose` and `verify` can advance the caches
/// or the draw.
#[allow(missing_debug_implementations)]
pub(crate) struct RoundCtx<'a> {
    /// The model a round verifies against.
    pub(crate) verifier: &'a Architecture,
    /// Its per-layer attention caches.
    pub(crate) kv: Vec<KvCache>,
    /// Its recurrent linear-attention state, on an architecture that keeps one.
    pub(crate) lin: Option<Vec<LinearAttnCache>>,
    /// The context ceiling the caches above were built at. A drafter that
    /// keeps a cache of its own sizes it from here, so it cannot overflow
    /// before the verifier does.
    pub(crate) max_seq: i32,
    /// The codec the caches above were built at. A drafter that builds a second
    /// stack of its own reads it here rather than resolving the request's
    /// override against the default a second time: a pair must run one codec,
    /// and a second resolution is a second answer waiting to differ.
    pub(crate) kv_quant: KvQuant,
    /// The request's draw over the verifier's logits.
    pub(crate) draw: VerifierDraw,
    /// The loop's phase-charge decision, for a drafter that forces the arrays it
    /// carries into the next round.
    pub(crate) charged: bool,
    pub(crate) device: Device,
}

/// The token a request's first round carries, and whether the request emits it
/// before that round runs.
///
/// Two statements in one value rather than a token beside a flag: every round
/// opens on a token, and only some of them come out of a prefill forward the
/// request is entitled to emit. A drafter that stated the two separately could
/// state them of different tokens.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Seed {
    /// A token the prefill forward drew past the whole prompt. The request emits
    /// it before its first round, and that round carries it.
    Emitted(u32),
    /// The prompt's own last token, which the prefill stopped short of. The
    /// first round carries it and the request emits nothing until a round
    /// commits.
    Carried(u32),
}

/// What a drafter's prefill left the request with.
#[allow(missing_debug_implementations)]
pub(crate) struct Prefilled {
    /// The token the first round carries, and whether the request emits it.
    pub(crate) seed: Seed,
    /// How long the prefill took, timed by the drafter: what the span covers
    /// differs per drafter, and a loop that timed the call would move the figure
    /// on records no gate reads.
    pub(crate) prefill_ns: u128,
    /// Whether this request scores any verified position over a vocabulary
    /// smaller than the verifier's.
    ///
    /// A request-level fact, not a per-drafter one: it is the drafter's offer
    /// and the request's sampler together, so a drafter that offers a reduced
    /// read-back still answers `false` above temperature 0. The loop reports it
    /// on the one line it writes per request, because whether a request's
    /// accepted positions carry the verifier's own argmax is recoverable from
    /// no other field of the run's log.
    pub(crate) restricted_read_back: bool,
    /// Whether this drafter projects a conditioning buffer and reports the rows
    /// it accumulates.
    ///
    /// A statement about the drafter, made before any round has run. The rows
    /// themselves are the loop's: it opens the count at zero for a drafter that
    /// declares `true` and adds each round's projection to it, so the figure on
    /// every record is the rows the request's rounds have projected so far —
    /// zero before any round, and the drafter's starting window is not counted.
    /// A drafter cannot supply an opening value, so a request cannot open at a
    /// count no round projected.
    pub(crate) projects_conditioning: bool,
}

/// What a round's verify forward decided.
#[allow(missing_debug_implementations)]
pub(crate) struct Verdict {
    /// Proposals the verifier accepted.
    pub(crate) accept: usize,
    /// The tokens the round commits: the agreed prefix and the one token the
    /// verifier stands behind. A rule may cut it to the request's remaining
    /// budget and the greedy walk does; one that does not is not reporting more
    /// than it committed, since the emission caps at the budget either way.
    pub(crate) commit: Vec<u32>,
    /// Wall clock of the verify forward and its read-back.
    pub(crate) verify_ns: u128,
    /// Wall clock of the acceptance walk.
    pub(crate) walk_ns: u128,
    /// How many of `commit`'s opening tokens this round decided over a reduced
    /// vocabulary rather than the verifier's whole one. Zero for a drafter that
    /// scores every position over the whole vocabulary, which is all of them
    /// but EAGLE-3 and EAGLE-3 itself on a sampled request.
    ///
    /// Never above `accept` — the correction past the accepted prefix is the
    /// verifier's own token over its whole vocabulary — and zero on a request
    /// whose [`Prefilled::restricted_read_back`] is `false`. The loop refuses a
    /// round that says otherwise; see [`super::guard_restricted_prefix`].
    pub(crate) restricted: usize,
}

/// What a drafter does that the shared loop cannot.
pub(crate) trait RoundDrafter {
    /// Which of the loop's two exits skips the report of the verifier's
    /// resident KV.
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy;

    /// Which read of the verifier's offset this drafter's round line reports.
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis;

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

    /// Return the drafter's own state to the accepted prefix, and say what span
    /// of its cache that moved. The default is a drafter that keeps nothing
    /// across rounds.
    ///
    /// # Errors
    /// Whatever the drafter's own rollback refuses.
    fn rollback(
        &mut self,
        _ctx: &RoundCtx<'_>,
        _verdict: &Verdict,
        _outcome: RoundOutcome,
    ) -> Result<Option<CacheSpan>> {
        Ok(None)
    }

    /// Condition the next round on what this one committed. `None` is a drafter
    /// that carries no conditioning buffer.
    ///
    /// # Errors
    /// Whatever building the next round's conditioning refuses.
    fn condition(
        &mut self,
        ctx: &RoundCtx<'_>,
        verdict: &Verdict,
        outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>>;

    /// Every array this round leaves for the next round's drafter to read, each
    /// under the name this drafter calls it, handed to `f` as one borrowed
    /// slice. On a charged round they must all be forced by the time the round's
    /// line is written.
    ///
    /// A callback rather than a returned collection: the list has one consumer,
    /// [`super::log_round`]'s charged arm, and that arm is off on the default
    /// path — a returned `Vec` would be allocated once per round of every
    /// request and dropped unread.
    ///
    /// `f` must be invoked — it is where the round's line is written, and the
    /// loop refuses a round that returns from here without it. A drafter that
    /// carries nothing by design invokes it with an empty slice.
    ///
    /// # Errors
    /// Whatever a drafter that has *lost* conditioning it should be holding
    /// refuses. Answering that state with an empty slice instead would let the
    /// charged round's forcing check pass on exactly the drafter it is for.
    fn carry(&self, f: &mut dyn FnMut(&[(&str, &Array)])) -> Result<()>;
}

/// Run one speculative request's rounds.
///
/// Returns the emitted steps and the widest block any round of this run actually
/// ran — not the block resolved before the loop, which a caller checking what it
/// asked for would be trusting the very step it wanted checked. The seed exit is
/// the exception and returns the resolved block: no round ran there.
///
/// `decided_by` is the per-token attribution buffer of a drafter whose verify
/// pass scores some positions over a reduced vocabulary: one entry per emitted
/// token, in emission order. The loop is what holds it because the emission is
/// the loop's and the request's budget can cut a round's tokens; the prefix
/// length is [`Verdict::restricted`]. A drafter that scores every position over
/// the verifier's whole vocabulary passes `None`. The buffer is not cleared
/// here: the entry clears it before its own refusals, which is what its public
/// signature promises a caller.
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
    mut decided_by: Option<&mut Vec<DecidedBy>>,
    device: Device,
) -> Result<(Vec<ProbeStep>, usize)> {
    let charge = cfg.charged;
    let n_tokens = cfg.n_tokens;
    let (kv_quant, max_seq, kv) =
        verifier_cache_stack(verifier, cfg.kv_quant_override, cfg.max_ctx_override)?;
    let lin = verifier
        .needs_lin_caches()
        .then(|| lin_cache_stack(verifier));
    let mut ctx = RoundCtx {
        verifier,
        kv,
        lin,
        max_seq,
        kv_quant,
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
    let mut conditioned_rows = prefilled.projects_conditioning.then_some(0usize);
    let restricted_read_back = prefilled.restricted_read_back;
    let mut carry = match prefilled.seed {
        Seed::Emitted(token) | Seed::Carried(token) => token,
    };

    // A pair that emits nothing before its first round has no seed to emit, to
    // attribute or to stop on, so all three are this arm's.
    if let Seed::Emitted(seed) = prefilled.seed {
        // A loop that attributes its tokens attributes the seed to the whole
        // vocabulary: it is drawn off the verifier's own prefill logits, and no
        // reduced read-back reaches it.
        if let Some(buf) = decided_by.as_deref_mut() {
            buf.push(DecidedBy::FullVocab);
        }
        if emit_seed_token(
            cfg.tokenizer,
            seed,
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
                report_verifier_kv_bytes(ctx.verifier, &ctx.kv, ctx.lin.as_deref());
            }
            return Ok((emitted, cfg.block_size));
        }
    }

    tracing::info!(
        loop_kind = ?cfg.loop_kind,
        block_size = cfg.block_size,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        temperature = cfg.sampler_cfg.temperature,
        restricted_read_back,
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

        // The rollback returns to the position the round started at, so the
        // target is computed from this read and never from the one after the
        // forward, which is a function of how far each layer happened to
        // advance.
        let pre_round_offset = ctx.kv.iter().map(KvCache::offset).max().unwrap_or(0);

        // Armed here rather than in each `verify`: the loop owns the recurrent
        // stack and is the tape's only consumer, through the rollback below. A
        // stack with no recurrent layer arms nothing.
        super::arm_lin_tapes(ctx.lin.as_deref_mut());
        let verdict = drafter.verify(&mut ctx, &fed, remaining)?;
        // Read again for the round line alone: a drafter that counts its
        // rollback back from the tail reports this number, and it is the same
        // position as the one above.
        let post_round_offset = ctx.kv.iter().map(KvCache::offset).max().unwrap_or(0);
        verifier_ns += verdict.verify_ns;
        total_accept += verdict.accept;

        guard_restricted_prefix(
            cfg.loop_kind,
            rounds,
            restricted_read_back,
            verdict.restricted,
            verdict.accept,
        )?;
        let emit = emit_round_tokens(
            cfg.tokenizer,
            &verdict.commit,
            n_tokens,
            cfg.eos_ids,
            step_fn,
            &mut emitted,
            &mut emitted_in_rounds,
            &mut window,
            decided_by
                .as_deref_mut()
                .map(|buf| (buf, verdict.restricted)),
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
            ctx.device,
        )?;
        // The verifier's own token at the accepted position, read off the
        // commit. The two part only on a full acceptance the request's budget
        // truncated, where the request has emitted its last token and the loop
        // leaves before any round reads what was stored here.
        carry = verdict.commit.last().copied().unwrap_or(carry);
        let outcome = RoundOutcome {
            round: rounds,
            committed: emit.committed,
            verifier_target: v_target,
            carry,
        };
        let d_span = drafter.rollback(&ctx, &verdict, outcome)?;
        let conditioning = drafter.condition(&ctx, &verdict, outcome)?;
        // Read back from what the projection returned rather than from what the
        // round meant to hand it, which is the reading the conditioning guard
        // takes too.
        let projected = conditioning.and_then(|c| c.projected);
        // The declaration's second reader. Without one, a drafter that projects
        // rows and declares it does not leaves the record reporting `None`, and
        // `RoundStats::conditioning_violation` opens by returning on that
        // `None` — so the bound over `emitted_in_rounds` is off for the whole
        // request and every other observable reads clean.
        if projected.is_some() && conditioned_rows.is_none() {
            return Err(Error::Model(format!(
                "{:?}: the drafter projected conditioning rows but declared it projects \
                 none, so the request record reports `None` and the conditioning check \
                 over `emitted_in_rounds` is guarded off for the whole request",
                cfg.loop_kind
            )));
        }
        if let (Some(total), Some(projected)) = (conditioned_rows.as_mut(), projected) {
            // `guard_round_conditioning` refuses a negative projection inside
            // `condition`, so this clamp cannot fire on any drafter on this
            // loop. It is here for one that reports without guarding, where the
            // alternative is a `usize` cast of a negative.
            *total += projected.max(0) as usize;
        }
        let round_rollback_ns = t0.elapsed().as_nanos();

        let report = super::RoundReport {
            loop_kind: cfg.loop_kind,
            round: rounds,
            accept: verdict.accept,
            num_draft: draft_tokens.len(),
            n_committed: emit.committed,
            emitted_total: emitted.len(),
            condition_rows: conditioning.and_then(|c| c.rows),
            projected_rows: conditioning.and_then(|c| c.projected),
            v_offset_before: match D::VERIFIER_OFFSET_BASIS {
                VerifierOffsetBasis::BeforeTheForward => pre_round_offset,
                VerifierOffsetBasis::AfterTheForward => post_round_offset,
            },
            v_target,
            d_offset_before: d_span.map(|s| s.before),
            d_target: d_span.map(|s| s.target),
            refolded,
            charged: charge,
            phases: Some(super::RoundPhases {
                round_ns: round_t0.elapsed().as_nanos(),
                draft_ns: round_draft_ns,
                verify_ns: verdict.verify_ns,
                walk_ns: verdict.walk_ns,
                rollback_ns: round_rollback_ns,
            }),
        };
        // The emit is inside the callback, so a drafter that returns without
        // invoking it deletes its own per-round stream and nothing else notices:
        // the text gates read that the call is written, the marker reading reads
        // where, and greedy verification answers the same either way. Only the
        // snapshot-gated baseline would see it, and only on a machine holding
        // the pair. A runtime refusal is what covers the rest.
        let mut lined = false;
        drafter.carry(&mut |carried| {
            lined = true;
            super::log_round(&report, carried);
        })?;
        if !lined {
            return Err(Error::Model(format!(
                "{:?}: the drafter's `carry` returned without handing its list \
                 over, so the round's line was never written",
                cfg.loop_kind
            )));
        }
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
        report_verifier_kv_bytes(ctx.verifier, &ctx.kv, ctx.lin.as_deref());
    }
    Ok((emitted, widest_bs))
}
