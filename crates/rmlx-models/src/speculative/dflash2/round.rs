//! The DFlash 2 round loop: draft a block, verify it, keep the agreed prefix.
//!
//! Ported from the z-lab MLX reference `_stream_generate`. Verifier-side it is
//! the shape every sidecar loop in this module family has — prefill, a bonus
//! token out of the prefill forward, then rounds of draft / verify / accept /
//! roll back — so it shares [`crate::speculative::accept_prefix`],
//! `rollback_round_caches` and [`crate::speculative::VerifierDraw`] with them
//! rather than restating any of it.
//!
//! # Conditioning: the K/V is recomputed, the projection is carried
//!
//! The reference gives its drafter a per-layer rotating K/V cache and feeds it
//! only each round's newly committed rows. This loop rebuilds that K/V from the
//! conditioning rows on every call instead. The two are the same answer — the
//! cached rows are a deterministic function of those rows — and the recomputing
//! form is what makes the drafter forward invariant to a uniform shift of every
//! position, which is why it needs no absolute offset. Adopting the cache would
//! mean cached rows carry their own absolute RoPE, and that invariance, and the
//! proof that rests on it, would have to be rebuilt.
//!
//! One step before that K/V is [`DFlash2Drafter::project_conditioning`], which
//! is row-wise and carries no position at all, and that is what this loop keeps:
//! it projects each round's committed capture rows as it commits them and
//! carries the projection. Re-projecting the whole window every round would give
//! the same rows for work proportional to the window.
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
//! candidate distribution to sample against. Like every other sidecar loop here
//! this one is greedy, and the serve layer routes a sidecar request to it
//! whatever the request's temperature.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{concatenate, Array, Device};

use super::DFlash2Drafter;
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use crate::speculative::round_common::{
    emit_round_tokens, emit_seed_token, log_request_record, report_verifier_kv_bytes,
    verifier_cache_stack, RoundTotals,
};
use crate::speculative::{
    accept_prefix, arm_lin_tapes, block_capped_by_checkpoint, committed_rows,
    conditioning_residual, disarm_lin_tapes, guard_round_conditioning,
    guard_verifier_prefill_logits, phases_charged, rollback_round_caches,
    rollback_target_from_tail, round_block, DecodeWindow, RoundPhases, SpecLoop, VerifierDraw,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

/// Prompt positions per verifier prefill pass.
///
/// The capture returns one hidden row per prompt position, so a single-shot
/// prefill of a long prompt would put the whole capture and a full-vocabulary
/// logit tensor in one Metal command buffer.
pub(super) const PREFILL_CHUNK_SIZE: usize = 1024;

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
/// Returns the emitted steps and **the widest block any round of this run
/// actually ran**. Not the block resolved before the loop: a caller checking
/// what it asked for against that would be trusting the very step it wanted
/// checked, and every loop here narrows the block again per round against the
/// remaining token budget.
///
/// # Errors
///
/// [`Error::Model`] when the prompt is too short to seed a round, when the
/// verifier is not one this drafter's seams are wired for, or from any forward,
/// acceptance or rollback below.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "every index is axis 1 of a rank-3 buffer whose rank the seam that produced it checked: the trim, the projection and the carried buffer they extend"
)]
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
    // this loop needs — the multi-layer hidden capture, the raw embedding and
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

    let target_layer_ids = drafter.cfg.target_layer_ids.clone();
    let condition_width = (drafter.cfg.hidden_size * target_layer_ids.len()) as i32;
    let block_total = block_capped_by_checkpoint(requested_block_total, drafter.cfg.block_size);

    let (kv_quant, _, mut v_caches) =
        verifier_cache_stack(verifier, kv_quant_override, max_ctx_override)?;
    let mut v_lin: Vec<LinearAttnCache> = (0..verifier.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect();

    let mut draw = VerifierDraw::new(sampler_cfg);

    let mut total_draft = 0usize;
    let mut total_accept = 0usize;
    let mut rounds = 0usize;
    // One read of process-global log state per request, at the loop head.
    let charge_phases = phases_charged();
    let t_total = Instant::now();
    let mut window = DecodeWindow::new();
    let mut draft_ns: u128 = 0;
    let mut verifier_ns: u128 = 0;

    let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

    // Prefill the whole prompt, keeping the conditioning hidden of as many of
    // its positions as the drafter's window reaches back over — the depth the
    // reference conditions its first round on, not the last token alone. The
    // rows before that can never be read, so the capture releases them as the
    // prefill walks the prompt rather than holding one row per prompt token at
    // `len(target_layer_ids) * hidden_size` to the end of it.
    let keep_rows = drafter.conditioning_rows();
    // Defence in depth: `check_config` refuses a window under 2 or wider than an
    // array axis, so this conversion cannot fail on a drafter that loaded.
    let keep = usize::try_from(keep_rows).map_err(|_| {
        Error::Model(format!(
            "dflash2_generate: this drafter reaches back over {keep_rows} rows, \
             which is not a row count"
        ))
    })?;
    let prefill_t0 = Instant::now();
    let (bonus_logits, prompt_hidden) = verifier.forward_verify_capture_chunked(
        prompt_ids,
        &target_layer_ids,
        &mut v_caches,
        Some(&mut v_lin),
        PREFILL_CHUNK_SIZE,
        Some(keep),
        device,
    )?;
    guard_verifier_prefill_logits(verifier, &bonus_logits, prompt_ids.len())?;
    let prompt_ctx = drafter.trim_conditioning(&prompt_hidden)?;
    // What the capture was asked to keep is a number at a call site, and the
    // trim above accepts a shorter buffer without a word. A drafter conditioned
    // on less than its window still proposes, and greedy verification then emits
    // the verifier's own tokens whatever the proposals were — so what falls is
    // the accept rate, and nothing here would say so.
    let want_rows = keep_rows.min(i32::try_from(prompt_ids.len()).unwrap_or(i32::MAX));
    if prompt_ctx.shape()[1] != want_rows {
        return Err(Error::Model(format!(
            "dflash2_generate: the prompt's capture kept {} conditioning rows where this \
             drafter reaches back over {want_rows} of a {}-token prompt",
            prompt_ctx.shape()[1],
            prompt_ids.len()
        )));
    }
    // The loop carries the projection, not the capture it came from: `fc` and
    // `hidden_norm` are row-wise, so a row projected once here is the row every
    // later round would have re-derived. The prompt's rows are projected in this
    // one call and each round then projects only what it commits.
    let mut h_ctx = drafter.project_conditioning(&prompt_ctx)?;
    // The prompt's last raw row, kept until the first round has extended the
    // buffer and then dropped: re-projecting it beside that round's commit is
    // what says how far the carried projection sits from a fresh one at this
    // checkpoint's dtype. One row, one round, and the probe releases it. See
    // `conditioning_residual`.
    let mut probe_seed = Some(committed_rows(
        &prompt_ctx.slice(
            &[0, prompt_ctx.shape()[1] - 1, 0],
            &[1, prompt_ctx.shape()[1], condition_width],
            &[1, 1, 1],
            device,
        )?,
        1,
        condition_width,
        device,
    )?);
    if charge_phases {
        // The guard above forced the logits, and so the whole prompt forward,
        // but not the capture: it hangs off a different output of that forward.
        // Joining the kept chunks, trimming to the window and projecting it is a
        // pass over one row per kept position — and with nothing forcing it here
        // the first round's drafter call pays for all of it. See
        // `phases_charged`.
        h_ctx.eval()?;
    }
    let prefill_ns = prefill_t0.elapsed().as_nanos();

    let mut b = draw.seed_token(&bonus_logits, device)?;
    if emit_seed_token(
        tokenizer,
        b,
        step_fn,
        &mut emitted,
        &mut window,
        eos_ids,
        &RoundTotals {
            loop_kind: SpecLoop::DFlash2,
            block_size: block_total,
            conditioned_rows: Some(0),
            charged: charge_phases,
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
        return Ok((emitted, block_total));
    }

    tracing::info!(
        block_size = block_total,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        ?target_layer_ids,
        condition_rows = h_ctx.shape()[1],
        temperature = sampler_cfg.temperature,
        "dflash2_generate: starting"
    );

    let seed_emitted = emitted.len();
    let mut emitted_in_rounds = 0usize;
    // Conditioning rows the rounds projected, read back from the projection
    // rather than from what the loop meant to hand it. See
    // `guard_round_conditioning`.
    let mut conditioned_rows = 0usize;
    let mut widest_bs = 0usize;
    let round_loop_t0 = Instant::now();
    while emitted.len() < n_tokens {
        rounds += 1;
        let round_t0 = Instant::now();
        let remaining = n_tokens - emitted.len();
        // The block never resizes: the drafter denoises the block it was
        // trained at, and only the token budget shortens it.
        let bs = round_block(block_total, remaining);
        widest_bs = widest_bs.max(bs);

        let t0 = Instant::now();
        let draft_tokens = draft_block(verifier, drafter, b, &h_ctx, bs, device)?;
        let round_draft_ns = t0.elapsed().as_nanos();
        draft_ns += round_draft_ns;
        if draft_tokens.is_empty() {
            return Err(Error::Model(format!(
                "dflash2_generate: the drafter denoised nothing at block {bs}; a block \
                 of two or more yields block - 1 proposals, so an empty block is a \
                 broken drafter and not the end of the request"
            )));
        }
        total_draft += draft_tokens.len();

        // The verifier scores the carry token and every proposal in one pass,
        // capturing the conditioning hidden for the same positions.
        arm_lin_tapes(Some(&mut v_lin));
        let mut v_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        v_input.push(b);
        v_input.extend_from_slice(&draft_tokens);
        let v_k = v_input.len();

        let t0 = Instant::now();
        let (v_logits, v_hidden) = verifier.forward_verify_capture(
            &v_input,
            v_k,
            &target_layer_ids,
            &mut v_caches,
            Some(&mut v_lin),
            device,
        )?;
        let v_tokens = draw.block_tokens(&v_logits, v_k, device)?;
        if charge_phases {
            // Reading the verifier's tokens forced the logits and the trunk
            // under them, but the
            // capture is a different output of the same forward and stays lazy.
            // The next round's drafter reads it, so with nothing forcing it here
            // the verifier's own capture is billed to the drafter — and this
            // drafter's capture is `len(target_layer_ids)` times as wide as the
            // sidecar loops', which is the comparison that would mislead. See
            // `phases_charged`.
            v_hidden.eval()?;
        }
        let round_verify_ns = t0.elapsed().as_nanos();
        verifier_ns += round_verify_ns;

        let t0 = Instant::now();
        let (accept, new_tokens) = accept_prefix(&v_tokens, &draft_tokens, remaining)?;
        let round_walk_ns = t0.elapsed().as_nanos();
        total_accept += accept;

        let hit_eos = emit_round_tokens(
            tokenizer,
            &new_tokens,
            n_tokens,
            eos_ids,
            step_fn,
            &mut emitted,
            &mut emitted_in_rounds,
            &mut window,
            None,
        );
        if hit_eos {
            break;
        }

        // The verifier consumed `v_k` positions and keeps the carry token plus
        // the accepted proposals. Read the offset from the deepest cache: a
        // recurrent layer's KvCache never advances, so layer 0 would report 0.
        let t0 = Instant::now();
        let v_offset_before = v_caches.iter().map(KvCache::offset).max().unwrap_or(0);
        let v_target = rollback_target_from_tail(v_offset_before, draft_tokens.len(), accept);
        let refolded = v_target < v_offset_before;
        if refolded {
            let v_pre_round_offset = v_offset_before - v_k as i32;
            rollback_round_caches(
                &mut v_caches,
                Some(&mut v_lin),
                &v_input,
                v_pre_round_offset,
                v_target,
                charge_phases,
                device,
            )?;
        } else {
            disarm_lin_tapes(Some(&mut v_lin));
        }
        let round_rollback_ns = t0.elapsed().as_nanos();

        // The carry token and the accepted proposals.
        let committed_hidden = committed_rows(&v_hidden, accept + 1, condition_width, device)?;
        let projected_rows;
        (h_ctx, projected_rows) = drafter.slide_conditioning(&h_ctx, &committed_hidden)?;
        guard_round_conditioning(rounds, projected_rows, accept + 1)?;
        conditioned_rows += projected_rows.max(0) as usize;
        if let Some(seed) = probe_seed.take() {
            let raw = concatenate(&[&seed, &committed_hidden], 1, device)?;
            let fresh = drafter.project_conditioning(&raw)?;
            let hidden = drafter.cfg.hidden_size as i32;
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
        if charge_phases {
            // Projecting this round's committed rows and copying the window they
            // extend is work the next round's drafter is the first thing to read.
            // Forced here it lands in the round's unclaimed time, which is where
            // slicing and bookkeeping belong; left lazy it lands in the drafter.
            // See `phases_charged`.
            h_ctx.eval()?;
        }
        b = *new_tokens.last().unwrap_or(&b);

        tracing::debug!(
            round = rounds,
            accept,
            num_draft = draft_tokens.len(),
            n_committed = new_tokens.len(),
            emitted_total = emitted.len(),
            condition_rows = h_ctx.shape()[1],
            projected_rows,
            v_offset_before,
            v_target,
            "dflash2 round"
        );

        RoundPhases {
            round_ns: round_t0.elapsed().as_nanos(),
            draft_ns: round_draft_ns,
            verify_ns: round_verify_ns,
            walk_ns: round_walk_ns,
            rollback_ns: round_rollback_ns,
            refolded,
            charged: charge_phases,
        }
        .log(
            SpecLoop::DFlash2,
            rounds,
            accept,
            draft_tokens.len(),
            &[("h_ctx", &h_ctx)],
        );
    }

    let round_loop_ns = round_loop_t0.elapsed().as_nanos();
    log_request_record(
        &RoundTotals {
            loop_kind: SpecLoop::DFlash2,
            block_size: block_total,
            conditioned_rows: Some(conditioned_rows),
            charged: charge_phases,
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

    report_verifier_kv_bytes(verifier, &v_caches, Some(&v_lin));
    Ok((emitted, widest_bs))
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
