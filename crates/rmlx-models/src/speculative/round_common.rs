//! What every speculative round loop sets up, and closes out, the same way.
//!
//! Nothing here has a unit test, and cannot: an [`Architecture`] is only
//! reachable by loading weights. What gates it is a round-loop run against a
//! real pair — `the_assistant_round_loop_reproduces_plain_greedy` in
//! `crates/rmlx-models/tests/spec_greedy_equivalence.rs` is the cheapest — and
//! the per-round event stream that run writes.

// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::time::Instant;

use rmlx_core::error::Result;
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

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

/// What a round loop counted and timed, handed to the one place that records it.
///
/// [`RoundStats`] is this plus the three figures a loop does not carry as a
/// counter, and it is assembled in [`log_request_record`] and nowhere else. A
/// loop names its own numbers here; a field added to the record is added once.
pub(crate) struct RoundTotals {
    /// Which loop produced the tokens.
    pub(crate) loop_kind: SpecLoop,
    /// The block the request was configured with, verifier token included.
    pub(crate) block_size: usize,
    /// Conditioning rows the rounds projected, or `None` for a loop that
    /// carries no conditioning buffer between rounds.
    pub(crate) conditioned_rows: Option<usize>,
    /// Whether the request ran with its phases charged — the same token the
    /// loop's `rollback_round_caches` calls take, and the one decision a loop
    /// makes about how its own work is attributed.
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

/// Record one speculative request, once.
///
/// Three figures are read here rather than carried: the tokens the loop handed
/// the sink, the request's wall-clock, and the decode-window rate.
///
/// `seed_emitted` is the fourth, and it is an argument rather than a field
/// because its source differs between the two callers — a loop that ran rounds
/// reads it off its own local, and a request that stopped on its seed reads it
/// off the buffer. Whether a loop emits a token before its rounds is a
/// measurement either way, never a per-loop constant.
pub(crate) fn log_request_record(
    totals: &RoundTotals,
    emitted: &[ProbeStep],
    seed_emitted: usize,
    window: &DecodeWindow,
) {
    RoundStats {
        loop_kind: totals.loop_kind,
        block_size: totals.block_size,
        rounds: totals.rounds,
        emitted: emitted.len(),
        emitted_in_rounds: totals.emitted_in_rounds,
        seed_emitted,
        conditioned_rows: totals.conditioned_rows,
        total_draft: totals.total_draft,
        total_accept: totals.total_accept,
        prefill_ns: totals.prefill_ns,
        draft_ns: totals.draft_ns,
        verifier_ns: totals.verifier_ns,
        round_loop_ns: totals.round_loop_ns,
        elapsed_ns: totals.t_total.elapsed().as_nanos(),
        decode_tps: window.tps(),
        charged: totals.charged,
    }
    .log_done();
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
