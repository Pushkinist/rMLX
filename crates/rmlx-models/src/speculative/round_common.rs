//! What every speculative round loop sets up, and closes out, the same way.
//!
//! The two cache-stack functions cannot have a unit test: an [`Architecture`]
//! is only reachable by loading weights. What gates them is a round-loop run
//! against a real pair — `the_assistant_round_loop_reproduces_plain_greedy` in
//! `crates/rmlx-models/tests/spec_greedy_equivalence.rs` is the cheapest — and
//! the per-round event stream that run writes.
//!
//! [`round_stats`] is the exception and is read on the CPU: it takes no model,
//! no device and no cache, and `round_common_tests.rs` is what reads it.

// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_kv_quant::{GdnTape, GdnTapeSegment, KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::{concatenate, Array, Device};

use super::eagle3::DecidedBy;
use super::{DecodeWindow, RoundStats, SpecLoop};
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;

/// The KV codec, context ceiling and per-layer cache stack a round loop runs
/// its verifier on.
///
/// The codec is the request's override or the same constant the verifier's own
/// decode path resolves — a spec pair must not run two different caches.
///
/// The ceiling is the verifier's: it owns the KV geometry, the drafter inherits
/// its cache sizing and its positional limit, so the verifier's
/// [`crate::context::ContextLimits`] are what bound the round loop. Routing
/// through [`crate::context::resolve_context`] keeps the speculative path on
/// the one resolution every other context cap reads, and gives it the same
/// refusal: a `--max-ctx` above the verifier's positional capacity used to be
/// taken verbatim here and only surfaced as a cache overflow mid-round.
///
/// # Errors
///
/// [`rmlx_core::error::Error::ContextCeilingExceeded`] when `max_ctx_override`
/// is above the verifier's positional capacity.
pub(crate) fn verifier_cache_stack(
    verifier: &Architecture,
    kv_quant_override: Option<KvQuant>,
    max_ctx_override: Option<i32>,
) -> Result<(KvQuant, i32, Vec<KvCache>)> {
    let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
    let max_seq =
        crate::context::resolve_context(&verifier.context_limits(), max_ctx_override)?.ceiling;
    Ok((kv_quant, max_seq, cache_stack(verifier, kv_quant, max_seq)))
}

/// One model's per-layer cache stack for one speculative request.
///
/// Each layer's cache carries that layer's sliding window, the ceiling, and
/// whether the stack's layers read each other's K/V — which is what decides
/// whether Mixed/RotK keep their bf16 mirror.
///
/// A layer that reports a sliding window gets the RotatingKvCache port whatever
/// codec it is handed — the branch is `window > 0` alone — so an SWA layer is
/// bf16 at `sliding_window` tokens under every `kv_quant` and only the
/// full-attention layers quantize. A block-verify write leaves that ring
/// holding its window plus the block, which is what lets a round's rollback
/// drop the rejected tail out of it losslessly.
///
/// The two-model loops build a second one of these for the draft model, at the
/// verifier's codec and the verifier's ceiling: the verifier owns the KV
/// geometry of a pair, so the draft's stack is the same stack against a
/// different layer count.
pub(crate) fn cache_stack(arch: &Architecture, kv_quant: KvQuant, max_seq: i32) -> Vec<KvCache> {
    (0..arch.num_hidden_layers())
        .map(|i| {
            let window = arch.layer_sliding_window(i);
            KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                .with_max_seq_ceiling(max_seq)
                .with_layer_idx(i)
                .with_shares_kv(arch.shares_kv_across_layers())
        })
        .collect()
}

/// One model's recurrent linear-attention state, one entry per layer.
///
/// Empty at the start of a request and grown by the forward: the state has no
/// sequence axis, so a round's rollback restores it from the tape rather than
/// slicing it.
///
/// The gate is the caller's: a sidecar loop refuses an architecture without
/// recurrent layers outright and always has a stack, while the two-model loops
/// serve both kinds and hold `None` for a verifier or a draft model that reads
/// the parameter and ignores it.
#[must_use]
pub(crate) fn lin_cache_stack(arch: &Architecture) -> Vec<LinearAttnCache> {
    (0..arch.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect()
}

/// What a round loop counted and timed, handed to the one place that records it.
///
/// [`RoundStats`] is this plus the figures a loop does not carry as a counter,
/// and it is assembled in [`round_stats`] and nowhere else. A loop names its
/// own numbers here; a field added to the record is added once.
#[derive(Debug)]
pub(crate) struct RoundTotals {
    /// Which loop produced the tokens.
    pub(crate) loop_kind: SpecLoop,
    /// The block the request was configured with, verifier token included.
    pub(crate) block_size: usize,
    /// Conditioning rows the rounds projected, or `None` for a loop that
    /// carries no conditioning buffer between rounds.
    pub(crate) conditioned_rows: Option<usize>,
    /// Whether the request ran with its phases charged — the same token the
    /// loop's [`rollback_round`] calls take, and the one decision a loop makes
    /// about how its own work is attributed.
    pub(crate) charged: bool,
    /// Rounds the loop entered.
    pub(crate) rounds: usize,
    /// Tokens the round loop itself emitted.
    pub(crate) emitted_in_rounds: usize,
    /// Tokens the drafter proposed, over all rounds.
    pub(crate) total_draft: usize,
    /// Proposed tokens the verifier accepted, over all rounds.
    pub(crate) total_accept: usize,
    /// Prompt prefill span.
    pub(crate) prefill_ns: u128,
    /// Wall-clock inside the drafter call, over all rounds.
    pub(crate) draft_ns: u128,
    /// Wall-clock inside the verify forward, over all rounds.
    pub(crate) verifier_ns: u128,
    /// Wall-clock of the whole round loop.
    pub(crate) round_loop_ns: u128,
    /// When the request started. The elapsed figure is read from it as the
    /// record is written, so a loop cannot report a window that closed before
    /// its last token.
    pub(crate) t_total: Instant,
}

/// The record a round loop's totals describe.
///
/// Three figures are read here rather than carried: the tokens the loop handed
/// the sink, the request's wall-clock, and the decode-window rate.
///
/// `seed_emitted` is the fourth, and it is an argument rather than a field
/// because its source differs between the two callers — a loop that ran rounds
/// reads it off its own local, and a request that stopped on its seed reads it
/// off the buffer. Whether a loop emits a token before its rounds is a
/// measurement either way, never a per-loop constant.
///
/// Separate from [`log_request_record`] because a mapping that returns its
/// record can be read without a model, and nothing else in the tree can read
/// it: a record is a `done` line, so a field written from the wrong place is a
/// plausible row and no failure. `round_common_tests.rs` is what reads it.
///
/// `RoundTotals` is destructured rather than read field by field, so a field
/// added to it and not carried across is a compile error and not a value left
/// quietly behind.
pub(crate) fn round_stats(
    totals: &RoundTotals,
    emitted: &[ProbeStep],
    seed_emitted: usize,
    window: &DecodeWindow,
) -> RoundStats {
    let &RoundTotals {
        loop_kind,
        block_size,
        conditioned_rows,
        charged,
        rounds,
        emitted_in_rounds,
        total_draft,
        total_accept,
        prefill_ns,
        draft_ns,
        verifier_ns,
        round_loop_ns,
        t_total,
    } = totals;
    RoundStats {
        loop_kind,
        block_size,
        rounds,
        emitted: emitted.len(),
        emitted_in_rounds,
        seed_emitted,
        conditioned_rows,
        total_draft,
        total_accept,
        prefill_ns,
        draft_ns,
        verifier_ns,
        round_loop_ns,
        elapsed_ns: t_total.elapsed().as_nanos(),
        decode_tps: window.tps(),
        charged,
    }
}

/// Record one speculative request, once.
pub(crate) fn log_request_record(
    totals: &RoundTotals,
    emitted: &[ProbeStep],
    seed_emitted: usize,
    window: &DecodeWindow,
) {
    round_stats(totals, emitted, seed_emitted, window).log_done();
}

/// Emit the tokens a round committed, and say whether one of them stopped the
/// request.
///
/// The budget is the request's and not the round's: a round emits only what is
/// still owed, so a block that overruns the last token leaves the surplus
/// unemitted and leaves the acceptance — which the rollback reads — alone.
///
/// `decided_by` is EAGLE-3's per-token attribution: the buffer, and the length
/// of this round's restricted-vocabulary prefix. Tokens before it were the
/// drafter's own, confirmed by a restricted argmax; the rest were taken over
/// the whole vocabulary. One entry per emitted token, and every other loop
/// passes `None`.
#[must_use = "the stop signal is the caller's: a round whose token ended the request \
                  must not run another one"]
pub(crate) fn emit_round_tokens(
    tokenizer: &tokenizers::Tokenizer,
    round_tokens: &[u32],
    n_tokens: usize,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    emitted: &mut Vec<ProbeStep>,
    emitted_in_rounds: &mut usize,
    window: &mut DecodeWindow,
    decided_by: Option<(&mut Vec<DecidedBy>, usize)>,
) -> RoundEmit {
    let (mut decided_by, restricted) = match decided_by {
        Some((buf, restricted)) => (Some(buf), restricted),
        None => (None, 0),
    };
    let mut committed = 0usize;
    for (i, &id) in round_tokens.iter().enumerate() {
        if emitted.len() >= n_tokens {
            break;
        }
        super::emit_step(tokenizer, id, step_fn, emitted, window);
        if let Some(buf) = decided_by.as_mut() {
            buf.push(if i < restricted {
                DecidedBy::RestrictedVocab
            } else {
                DecidedBy::FullVocab
            });
        }
        *emitted_in_rounds += 1;
        committed += 1;
        if eos_ids.contains(&id) {
            return RoundEmit {
                committed,
                hit_eos: true,
            };
        }
    }
    RoundEmit {
        committed,
        hit_eos: false,
    }
}

/// What [`emit_round_tokens`] did with a round's tokens.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoundEmit {
    /// How many of them reached the sink.
    ///
    /// The round's committed count, and the one producer of it: a loop that
    /// reports the length of what it *handed* this function reports the tokens
    /// the request's budget cut as committed, and the rollback beside it does
    /// not.
    pub(crate) committed: usize,
    /// Whether one of them ended the request.
    pub(crate) hit_eos: bool,
}

/// Length of `a`'s sequence axis, which every taped recurrence input carries at
/// axis 1.
fn seq_len(a: &Array) -> Result<i32> {
    a.shape().get(1).copied().ok_or_else(|| {
        Error::Model(format!(
            "seq_len: a taped recurrence input carries its positions on axis 1, \
             and this one has shape {:?}",
            a.shape()
        ))
    })
}

/// `a[:, from..to, ...]` — a range of a taped recurrence input's positions.
fn seq_range(a: &Array, from: i32, to: i32, device: Device) -> Result<Array> {
    let len = seq_len(a)?;
    if from < 0 || from > to || to > len {
        return Err(Error::Model(format!(
            "seq_range: positions {from}..{to} are not inside a taped input of \
             length {len}"
        )));
    }
    if from == 0 && to == len {
        return a.try_clone();
    }
    let shape = a.shape();
    let start: Vec<i32> = shape
        .iter()
        .enumerate()
        .map(|(axis, _)| if axis == 1 { from } else { 0 })
        .collect();
    let stop: Vec<i32> = shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| if axis == 1 { to } else { dim })
        .collect();
    a.slice(&start, &stop, &vec![1i32; shape.len()], device)
}

/// Rebuild every recurrent layer's state at `kept` positions into this round
/// from the tape its forwards recorded.
///
/// `round_len` is how many positions the round fed. Each recurrent layer's tape
/// must hold exactly that many, and this refuses the round rather than refolding
/// when one does not: a tape that is short recorded fewer forwards than the
/// round took — armed late, or a forward that ran with recording off — and
/// refolding it would leave the recurrent state describing a different prefix
/// from the K/V stack beside it, which no later call can detect and which shows
/// up only as wrong tokens.
///
/// `lin` carries one slot per decoder layer, and on a hybrid most of them belong
/// to full-attention layers that never touch a recurrence. Those record nothing
/// and hold no state, and are skipped. Holding no state is what separates them
/// from a recurrent layer whose forward failed to record: that one has a state
/// this round advanced, and an empty tape for it is the defect above.
fn refold_lin_tapes(
    lin: &mut [LinearAttnCache],
    round_len: usize,
    kept: usize,
    charge: bool,
    device: Device,
) -> Result<()> {
    let mut refolded: Vec<Array> = Vec::new();
    for (idx, cache) in lin.iter_mut().enumerate() {
        let Some(tape) = cache.take_tape() else {
            return Err(Error::Model(format!(
                "refold_lin_tapes: recurrent layer {idx} has no round tape, so the \
                 {round_len} positions this round fed through it were never recorded \
                 and its state cannot be rolled back to {kept}"
            )));
        };
        let taped = tape.positions();
        if taped == 0 && cache.conv_state.is_none() && cache.delta_state.is_none() {
            continue;
        }
        if taped != round_len {
            return Err(Error::Model(format!(
                "refold_lin_tapes: recurrent layer {idx} taped {taped} positions over \
                 {} forwards but the round fed {round_len} — the tape does not describe \
                 this round",
                tape.segments().len(),
            )));
        }
        let Some(state_in) = tape.state_in() else {
            return Err(Error::Model(format!(
                "refold_lin_tapes: recurrent layer {idx} taped {taped} positions with no \
                 pre-round state"
            )));
        };
        let conv = concat_tape_conv_input(&tape, device)?;
        // What the conv1d carries into the next call is the `kernel - 1`
        // positions before its next input, and the tape's conv input opens with
        // exactly those, so the round's own carry sits at `kept`.
        let pad = seq_len(&conv)? - taped as i32;
        cache.conv_state = Some(seq_range(&conv, kept as i32, kept as i32 + pad, device)?);
        cache.delta_state = Some(if kept == 0 {
            state_in.try_clone()?
        } else {
            let (_y, state_out) = crate::gated_delta_msl::gated_delta_step_gpu(
                &tape_prefix(&tape, |s| &s.q, kept, device)?,
                &tape_prefix(&tape, |s| &s.k, kept, device)?,
                &tape_prefix(&tape, |s| &s.v, kept, device)?,
                &tape_prefix(&tape, |s| &s.g, kept, device)?,
                &tape_prefix(&tape, |s| &s.beta, kept, device)?,
                state_in,
                device,
            )?;
            state_out
        });
        if charge {
            // Nothing reads the refolded state until the next round's forward,
            // so with nothing forcing it here the whole refold is billed to that
            // round. See `phases_charged`. Issued for every layer first and
            // waited on afterwards: draining each layer in turn would price the
            // rollback at the cost of serialising it.
            for a in [&cache.conv_state, &cache.delta_state]
                .into_iter()
                .flatten()
            {
                a.async_eval()?;
                refolded.push(a.try_clone()?);
            }
        }
    }
    for a in &refolded {
        a.eval()?;
    }
    Ok(())
}

/// The round's conv1d input across every taped forward, carried prefix included.
///
/// Each forward's input opens with the `kernel - 1` positions it carried in from
/// its predecessor, and those are already in that predecessor's segment, so only
/// the first segment contributes its prefix.
fn concat_tape_conv_input(tape: &GdnTape, device: Device) -> Result<Array> {
    let Some((first, rest)) = tape.segments().split_first() else {
        return Err(Error::Model(
            "concat_tape_conv_input: empty round tape".into(),
        ));
    };
    if rest.is_empty() {
        return first.conv_input.try_clone();
    }
    let pad = seq_len(&first.conv_input)? - first.len as i32;
    let mut parts: Vec<Array> = Vec::with_capacity(rest.len() + 1);
    parts.push(first.conv_input.try_clone()?);
    for seg in rest {
        let len = seq_len(&seg.conv_input)?;
        parts.push(seq_range(&seg.conv_input, pad, len, device)?);
    }
    concatenate(&parts.iter().collect::<Vec<_>>(), 1, device)
}

/// One recurrence input over the round's first `kept` positions, joined across
/// whatever forwards produced them.
///
/// `kept` is at most the tape's position count, checked by the caller.
fn tape_prefix(
    tape: &GdnTape,
    field: fn(&GdnTapeSegment) -> &Array,
    kept: usize,
    device: Device,
) -> Result<Array> {
    let mut parts: Vec<Array> = Vec::new();
    let mut taken = 0usize;
    for seg in tape.segments() {
        if taken >= kept {
            break;
        }
        let want = (kept - taken).min(seg.len);
        parts.push(seq_range(field(seg), 0, want as i32, device)?);
        taken += want;
    }
    match parts.len() {
        1 => parts
            .pop()
            .ok_or_else(|| Error::Model("tape_prefix: a one-part join lost its part".into())),
        _ => concatenate(&parts.iter().collect::<Vec<_>>(), 1, device),
    }
}

/// Roll one speculative round's caches back to `target_offset` after a partial
/// acceptance — both the full-attention `kv` stack and, when the arch has one,
/// the GDN recurrent state in `lin`.
///
/// `pre_round_offset` is the KV offset before this round's forwards ran;
/// `round_tokens` are the tokens those forwards consumed, in order, so that
/// `round_tokens[..target_offset - pre_round_offset]` is exactly the retained
/// prefix.
///
/// **Full-attention arch** (`lin` empty or absent): every layer's KvCache
/// carries the whole round, so dropping the rejected tail is the entire
/// rollback.
///
/// **GDN hybrid**: the recurrent state has no sequence axis (see
/// `LinearAttnCache`), so it cannot be sliced to an intermediate position. It is
/// rebuilt instead, from the round tape the loop armed before its forwards:
/// the recurrence inputs at the retained positions are the ones the forward
/// already computed, so the state is refolded by the recurrence kernel over
/// those alone. That reads no weights and takes no second forward, which is
/// what makes a partly-accepted round cost the same as a fully accepted one.
///
/// The K/V stack is truncated straight to `target_offset` on both arms. A
/// windowed layer therefore only ever has to give back this round's rejected
/// tail, which is inside any ring's reach.
///
/// Private to this module, and reached only through [`rollback_round`], which
/// is what decides the arm: this is the partial-accept side, and on a full
/// accept there is nothing to roll back and the tapes are dropped instead. A
/// loop that could name this could make a second charge decision beside the one
/// it declares at its own call site, and nothing would read it.
///
/// `charge` is the calling loop's per-request answer from
/// [`super::phases_charged`], not a decision this function makes. Seven loops share it
/// and three of them time their phases; reading the switch here would change the
/// schedule of the other four with nothing on their records saying so.
fn rollback_round_caches(
    kv: &mut [KvCache],
    lin: Option<&mut [LinearAttnCache]>,
    round_tokens: &[u32],
    pre_round_offset: i32,
    target_offset: i32,
    charge: bool,
    device: Device,
) -> Result<bool> {
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
    super::truncate_kv_to(kv, target_offset)?;
    let Some(lin) = lin.filter(|l| !l.is_empty()) else {
        return Ok(false);
    };
    refold_lin_tapes(lin, round_tokens.len(), kept, charge, device)?;
    Ok(true)
}

/// Return a round's caches to the prefix the verifier kept, or drop the round
/// tape when it kept all of it.
///
/// `pre_round_offset` is where the caches stood before they consumed
/// `round_tokens`, so `pre_round_offset + round_tokens.len()` is where they
/// stand now and a `target_offset` below it is a partial accept. On a partial
/// accept the attention caches are truncated and the recurrent state — which
/// has no sequence axis to slice — is refolded from the tape over the retained
/// prefix; on a full accept nothing was dropped and the tape is discarded.
///
/// `charge` is the loop's own phase-charge decision, forwarded rather than
/// made here: it belongs at the call site, beside the record that has to name
/// the same one.
///
/// Returns whether a recurrent state was refolded — true only on the partial
/// arm of a stack that carries one. This is the one producer of that fact:
/// every loop reports it, and a loop re-deriving it from its own offsets says
/// `true` for a full-attention or dense verifier that refolded nothing.
///
/// # Errors
///
/// [`rmlx_core::error::Error::Model`] when the refold cannot replay the
/// retained prefix. The low-level rollback also refuses a prefix longer than
/// the round fed, which cannot be reached from here: this arm is taken only
/// when `target_offset` is inside the round.
pub(crate) fn rollback_round(
    kv: &mut [KvCache],
    lin: Option<&mut [LinearAttnCache]>,
    round_tokens: &[u32],
    pre_round_offset: i32,
    target_offset: i32,
    charge: bool,
    device: Device,
) -> Result<bool> {
    if target_offset < pre_round_offset + round_tokens.len() as i32 {
        return rollback_round_caches(
            kv,
            lin,
            round_tokens,
            pre_round_offset,
            target_offset,
            charge,
            device,
        );
    }
    super::disarm_lin_tapes(lin);
    Ok(false)
}

/// Emit a sidecar loop's seed token, and say whether it ended the request.
///
/// Every sidecar loop argmaxes one bonus token out of its prefill forward and
/// emits it before the first round. When that token is a stop token no round
/// ever runs, and the request still leaves exactly one record — of a request
/// whose whole output is its seed. Returns whether the caller should return now;
/// it owns `emitted` and the block it reports, so the return itself stays with
/// it.
///
/// The two-model loops do not call this: they emit nothing before a round.
#[must_use = "the stop signal is the caller's: a seed that ended the request must not \
                  be followed by a round"]
pub(crate) fn emit_seed_token(
    tokenizer: &tokenizers::Tokenizer,
    seed: u32,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    emitted: &mut Vec<ProbeStep>,
    window: &mut DecodeWindow,
    eos_ids: &[u32],
    totals: &RoundTotals,
) -> bool {
    super::emit_step(tokenizer, seed, step_fn, emitted, window);
    if !eos_ids.contains(&seed) {
        return false;
    }
    log_request_record(totals, emitted, emitted.len(), window);
    true
}

/// Report a verifier's resident KV after one speculative request.
///
/// A speculative round loop never goes through `Architecture::generate_greedy`,
/// so nothing else writes this and a caller that sampled the verifier around the
/// call would otherwise read whatever the previous request left.
///
/// Only the verifier's caches count, on the same basis as the per-arch
/// `generate_greedy` byte total — the attention caches plus, on hybrid archs,
/// the recurrent linear-attention state. The draft model's are an
/// implementation detail of the accelerator, and including them would make a
/// speculative row incomparable with the ordinary row for the same model and
/// context.
pub(crate) fn report_verifier_kv_bytes(
    verifier: &Architecture,
    kv: &[KvCache],
    lin: Option<&[LinearAttnCache]>,
) {
    let bytes = kv.iter().map(KvCache::resident_bytes).sum::<u64>()
        + lin.map_or(0, |l| l.iter().map(LinearAttnCache::resident_bytes).sum());
    verifier.store_kv_cache_bytes(bytes, crate::decode_loop::PostDecode::seal());
}

#[cfg(test)]
#[path = "round_common_tests.rs"]
mod round_common_tests;
