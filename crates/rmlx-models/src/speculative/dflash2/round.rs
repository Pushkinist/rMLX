//! The DFlash 2 round loop: draft a block, verify it, keep the agreed prefix.
//!
//! Ported from the z-lab MLX reference `_stream_generate` at
//! `temperature == 0`. Verifier-side it is the shape every sidecar loop in this
//! module family has — prefill, a bonus token out of the prefill forward, then
//! rounds of draft / verify / accept / roll back — so it shares
//! [`crate::speculative::accept_prefix`], `rollback_round_caches` and
//! [`crate::speculative::argmax_tokens`] with them rather than restating any of
//! it.
//!
//! # Conditioning: recomputed, not cached
//!
//! The reference gives its drafter a per-layer rotating K/V cache and feeds it
//! only each round's newly committed rows. This loop keeps the committed
//! **hidden states** instead and lets [`DFlash2Drafter::forward_hidden`]
//! re-derive the conditioning K/V from them every round. The two are the same
//! answer — the cached rows are a deterministic function of those hidden states
//! — and the recomputing form is what makes the drafter forward invariant to a
//! uniform shift of every position, which is why it needs no absolute offset.
//! Adopting the cache would mean cached rows carry their own absolute RoPE, and
//! that invariance, and the proof that rests on it, would have to be rebuilt.
//!
//! The buffer is bounded, not accumulated: the drafter attends over one sliding
//! window, so rows older than it can never be read and are dropped as they fall
//! out. Each row is `len(target_layer_ids) * hidden` wide, so an unbounded one
//! would grow by 50 KiB per emitted token on the published pair.
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

// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, Array, Device};

use super::DFlash2Drafter;
use crate::arch::Architecture;
use crate::decode_loop::ProbeStep;
use crate::speculative::dflash::DFlashRoundState;
use crate::speculative::{
    accept_prefix, argmax_tokens, emit_step, guard_verifier_prefill_logits, rollback_round_caches,
    verifier_context, verifier_kv_bytes, DecodeWindow, RoundStats, SpecLoop,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

/// Prompt positions per verifier prefill pass.
///
/// The capture returns one hidden row per prompt position, so a single-shot
/// prefill of a long prompt would put the whole capture and a full-vocabulary
/// logit tensor in one Metal command buffer.
const PREFILL_CHUNK_SIZE: usize = 1024;

/// Drive a DFlash 2 drafter against its verifier, greedily.
///
/// `requested_block_total` is the round block including the verifier's own
/// token; it is clamped to the block the drafter was trained at, which is the
/// widest one its selector chain is defined over.
///
/// `step_fn` is called once per emitted token. Its `Option<u32>` return — the
/// forced-token contract the plain decode loop uses — is discarded here, as it
/// is on every speculative loop: a round's tokens are already the verifier's.
///
/// # Errors
///
/// [`Error::Model`] when the prompt is too short to seed a round, when the
/// verifier is not one this drafter's seams are wired for, or from any forward,
/// acceptance or rollback below.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "the one index is axis 1 of the conditioning buffer, whose rank keep_last_rows checks before returning it"
)]
pub fn dflash2_generate_greedy(
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
    device: Device,
) -> Result<Vec<ProbeStep>> {
    if prompt_ids.len() < 2 {
        return Err(Error::Model(
            "dflash2_generate_greedy: prompt must have >=2 tokens".into(),
        ));
    }
    if !verifier.needs_lin_caches() {
        return Err(Error::Model(
            "dflash2_generate_greedy: the DFlash 2 verifier must be the Qwen3.5/3.6 \
             hybrid — the drafter reads its multi-layer hidden capture, its raw \
             embedding and its LM head, and no other architecture exposes them"
                .into(),
        ));
    }

    let target_layer_ids = drafter.cfg.target_layer_ids.clone();
    let condition_width = (drafter.cfg.hidden_size * target_layer_ids.len()) as i32;
    let block_total = requested_block_total.min(drafter.cfg.block_size).max(2);

    // Same constant the verifier resolves — a spec pair must not run two
    // different caches.
    let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
    // The verifier's limits bound the pair; an over-capacity `--max-ctx` is
    // refused here rather than overflowing a cache mid-round.
    let ctx = verifier_context(verifier, max_ctx_override)?;
    let max_seq = ctx.ceiling;

    let mut v_caches: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| {
            let window = verifier.layer_sliding_window(i);
            KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                .with_max_seq_ceiling(ctx.ceiling)
                .with_layer_idx(i)
                // The verifier stack decides whether its layers read each
                // other's K/V, and so whether Mixed/RotK keep their bf16
                // mirror. A spec pair must not run two different caches.
                .with_shares_kv(verifier.shares_kv_across_layers())
        })
        .collect();
    let mut v_lin: Vec<LinearAttnCache> = (0..verifier.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect();

    let mut total_draft = 0usize;
    let mut total_accept = 0usize;
    let mut rounds = 0usize;
    let t_total = Instant::now();
    let mut window = DecodeWindow::new();
    let mut draft_ns: u128 = 0;
    let mut verifier_ns: u128 = 0;

    let mut emitted: Vec<ProbeStep> = Vec::with_capacity(n_tokens);

    // Prefill the whole prompt, capturing every position's conditioning hidden.
    // The reference conditions its first round on as much of the prompt as the
    // drafter's window reaches back over, not on the last token alone.
    let prefill_t0 = Instant::now();
    let (bonus_logits, prompt_hidden) = verifier.forward_verify_capture_chunked(
        prompt_ids,
        &target_layer_ids,
        &mut v_caches,
        Some(&mut v_lin),
        PREFILL_CHUNK_SIZE,
        device,
    )?;
    guard_verifier_prefill_logits(verifier, &bonus_logits, prompt_ids.len())?;
    let mut h_ctx = drafter.trim_conditioning(&prompt_hidden)?;
    let prefill_ns = prefill_t0.elapsed().as_nanos();

    let mut b = read_argmax(&bonus_logits, device)?;
    emit_step(tokenizer, b, step_fn, &mut emitted, &mut window);
    if eos_ids.contains(&b) {
        // The stop token arrived before a round could run. The request still
        // happened, so it still leaves exactly one record.
        RoundStats {
            loop_kind: SpecLoop::DFlash2,
            block_size: block_total,
            rounds: 0,
            emitted: emitted.len(),
            seed_emitted: emitted.len(),
            emitted_in_rounds: 0,
            total_draft: 0,
            total_accept: 0,
            prefill_ns,
            draft_ns: 0,
            verifier_ns: 0,
            round_loop_ns: 0,
            elapsed_ns: t_total.elapsed().as_nanos(),
            decode_tps: window.tps(),
            charged: false,
        }
        .log_done();
        return Ok(emitted);
    }

    tracing::info!(
        block_size = block_total,
        prompt_len = prompt_ids.len(),
        n_tokens,
        ?kv_quant,
        ?target_layer_ids,
        condition_rows = h_ctx.shape()[1],
        "dflash2_generate_greedy: starting"
    );

    let seed_emitted = emitted.len();
    let mut emitted_in_rounds = 0usize;
    let round_loop_t0 = Instant::now();
    while emitted.len() < n_tokens {
        rounds += 1;
        let remaining = n_tokens - emitted.len();
        // The block never resizes: the drafter denoises the block it was
        // trained at, and only the token budget shortens it.
        let bs = block_total.min(remaining + 1);
        if bs <= 1 {
            break;
        }

        let t0 = Instant::now();
        let draft_tokens = draft_block(verifier, drafter, b, &h_ctx, bs, device)?;
        draft_ns += t0.elapsed().as_nanos();
        if draft_tokens.is_empty() {
            break;
        }
        total_draft += draft_tokens.len();

        // The verifier scores the carry token and every proposal in one pass,
        // capturing the conditioning hidden for the same positions.
        let round_snap = DFlashRoundState::snapshot(&v_lin)?;
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
        let v_argmax = argmax(&v_logits, -1, device)?;
        v_argmax.eval()?;
        let vb = v_argmax.to_bytes()?;
        verifier_ns += t0.elapsed().as_nanos();
        let v_tokens = argmax_tokens(&vb, v_k)?;

        let (accept, new_tokens) = accept_prefix(&v_tokens, &draft_tokens, remaining)?;
        total_accept += accept;

        let mut hit_eos = false;
        for &id in &new_tokens {
            if emitted.len() >= n_tokens {
                break;
            }
            emit_step(tokenizer, id, step_fn, &mut emitted, &mut window);
            emitted_in_rounds += 1;
            if eos_ids.contains(&id) {
                hit_eos = true;
                break;
            }
        }
        if hit_eos {
            break;
        }

        // The verifier consumed `v_k` positions and keeps the carry token plus
        // the accepted proposals. Read the offset from the deepest cache: a
        // recurrent layer's KvCache never advances, so layer 0 would report 0.
        let v_offset_before = v_caches.iter().map(KvCache::offset).max().unwrap_or(0);
        let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);
        if v_target < v_offset_before {
            let v_pre_round_offset = v_offset_before - v_k as i32;
            rollback_round_caches(
                verifier,
                &mut v_caches,
                Some(&mut v_lin),
                Some(round_snap.into_snapshots()),
                &v_input,
                v_pre_round_offset,
                v_target,
                // This loop times no phases, so it never charges one.
                false,
                device,
            )?;
        } else {
            drop(round_snap);
        }

        // The conditioning rows are exactly the positions the caches kept: the
        // carry token and the accepted proposals.
        let committed = accept as i32 + 1;
        let committed_hidden = v_hidden.slice(
            &[0, 0, 0],
            &[1, committed, condition_width],
            &[1, 1, 1],
            device,
        )?;
        h_ctx = drafter.extend_conditioning(&h_ctx, &committed_hidden)?;
        b = *new_tokens.last().unwrap_or(&b);

        tracing::debug!(
            round = rounds,
            accept,
            num_draft = draft_tokens.len(),
            n_committed = new_tokens.len(),
            emitted_total = emitted.len(),
            condition_rows = h_ctx.shape()[1],
            v_offset_before,
            v_target,
            "dflash2 round"
        );
    }

    let round_loop_ns = round_loop_t0.elapsed().as_nanos();
    RoundStats {
        loop_kind: SpecLoop::DFlash2,
        block_size: block_total,
        rounds,
        emitted: emitted.len(),
        seed_emitted,
        emitted_in_rounds,
        total_draft,
        total_accept,
        prefill_ns,
        draft_ns,
        verifier_ns,
        round_loop_ns,
        elapsed_ns: t_total.elapsed().as_nanos(),
        decode_tps: window.tps(),
        charged: false,
    }
    .log_done();

    // Report the verifier's resident KV, so a caller that sampled the verifier
    // arch around this call can attribute the figure to it. This round loop
    // never goes through `Architecture::generate_greedy`, so nothing else
    // writes it.
    verifier.store_kv_cache_bytes(
        verifier_kv_bytes(&v_caches, Some(&v_lin)),
        crate::decode_loop::PostDecode::seal(),
    );
    Ok(emitted)
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
    let hidden = drafter.forward_hidden(&block, h_ctx)?;
    // Row 0 is the seed, which is not drafted.
    let drafted = hidden.slice(&[0, 1, 0], &[1, bs as i32, hidden_size], &[1, 1, 1], device)?;
    let logits = verifier.logits_from_final_hidden(&drafted, device)?;
    drafter.select_chain(&drafted, &logits, seed)
}

/// One token id off a `[.., 1, vocab]` logit row.
#[allow(
    clippy::indexing_slicing,
    reason = "the four bytes are indexed only after the buffer's length has been checked against them"
)]
fn read_argmax(logits: &Array, device: Device) -> Result<u32> {
    let am = argmax(logits, -1, device)?;
    am.eval()?;
    let bytes = am.to_bytes()?;
    if bytes.len() < 4 {
        return Err(Error::Model(format!(
            "dflash2_generate_greedy: the verifier's argmax came back as {} bytes, which \
             is not one token id",
            bytes.len()
        )));
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
