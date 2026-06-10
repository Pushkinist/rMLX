//! Gemma4 greedy generation loop and smoke-probe classifier.
//!
//! [`generate_greedy`] drives the full autoregressive decode pipeline for
//! Gemma4 models: chunked prefill, KV-cache management, per-token sampling,
//! thinking-budget enforcement, speculative-draft integration, and streaming
//! token delivery. [`classify_smoke`] post-processes a short probe sequence
//! to produce a pass/fail [`SmokeVerdict`] used by `rmlx info --probe` and
//! the startup smoke check.
//!
//! # Public API
//!
//! - [`generate_greedy`] — full autoregressive generation entry point.
//! - [`classify_smoke`] — classify a [`ProbeStep`] trace as pass/warn/fail.
//! - [`ProbeStep`] — one decode step's token id + log-probability, used by
//!   the smoke classifier.
//! - [`SmokeVerdict`] — pass / warn / fail verdict with a human-readable reason.

#![allow(
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines
)]
use std::time::Instant;

pub(super) mod classify;
#[cfg(test)]
mod tests;
pub(super) mod types;

pub use classify::classify_smoke;
use types::{count_nan_in_bytes, max_abs_from_bytes};
pub use types::{ProbeStep, SmokeVerdict};

use rmlx_mlx::{Array, Device};
use tracing::info_span;

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, DecodeCtx,
};
use crate::kv_cache::{
    kv_max_seq_and_ceiling, kv_quant_for_layer, warn_if_kv_codec_net_negative, KvLayerShape,
    LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N,
};
use crate::prompt_cache::PromptCacheEntry as _;
use rmlx_kv_quant::{KvCache, KvQuant};

use super::model::Gemma4Text;
use super::prompt_cache::{ensure_prompt_cache, store_kv_cache_bytes, Gemma4Entry, PROMPT_CACHE};

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// One prefill call encodes the full prompt and populates all layer caches.
/// Each subsequent decode step passes a single token and reads from the cache,
/// reducing cost from O(N²) to O(N + T) where T is the number of new tokens.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Recommended: 4. Set to 0 to disable caching (treated as 1 by the
/// underlying `PromptCache::new(capacity.max(1))`).
///
/// See CLAUDE.md "mxfp8 broken-snapshot hazard" for why NaN detection exists.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub fn generate_greedy<'a>(
    model: &Gemma4Text,
    tokenizer: &'a tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &'a [u32],
    // The shared `DecodeCtx` bundles every per-request borrow under one
    // lifetime, so these references share `'a` (a `&mut dyn` trait-object
    // reborrow is invariant and cannot be re-unified once split).
    step_fn: &'a mut dyn FnMut(&ProbeStep) -> Option<u32>,
    // A6.2: optional sampler constraint. `None` = unmasked argmax (the hot
    // path; identical to pre-A6.2 behaviour). `Some(_)` enables the masked
    // branch at every argmax call site below.
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. `temperature <= 0.0` keeps the
    // untouched greedy argmax path (`sampler_cfg.sampling_active() == false`).
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    // `penalty_cfg.penalties_active() == false` AND `!sampler_cfg.sampling_active()`
    // keeps the temp=0 pure-GPU argmax path byte-for-byte untouched.
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
    // optional precomputed multimodal prefill. `Some((embeds,
    // masked_ids))` carries the scatter-merged `inputs_embeds` `[1, seq,
    // hidden]` (text + vision features at image-token positions) and the
    // per-layer-input ids `[seq]` (image positions masked to 0), built by
    // `gemma4::build_inputs_embeds`. When present the prompt cache is bypassed
    // (image prompts are one-shot) and prefill runs from the embeds in one
    // forward. `None` is the text path — byte-identical to pre-.
    image_prefill: Option<(Array, Array)>,
) -> rmlx_core::error::Result<Vec<ProbeStep>> {
    tracing::info!(
        arch = "Gemma4ForConditionalGeneration",
        ?kv_quant,
        ?max_ctx_override,
        prompt_cache_slots,
        has_image = image_prefill.is_some(),
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    // capture before `image_prefill` is moved into the prefill branch.
    // Image prompts must not be saved to the token-id-keyed prompt cache (the
    // K/V depends on scattered vision features, not just the token ids).
    let has_image = image_prefill.is_some();

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

    ensure_prompt_cache(prompt_cache_slots);

    // ------------------------------------------------------------------
    // Prompt cache lookup
    // ------------------------------------------------------------------
    // C1: three outcomes. `Exact` = cached prompt token-for-token equal to
    // this prompt (full reuse, no truncate, no re-prefill). `Prefix` = cached
    // prompt shares >= 1 full 256-token block then diverges AND every layer
    // cache is losslessly trimmable (mlx-lm `can_trim_prompt_cache`) — the KV
    // is block-truncated and only the tail `[prefix_len..prompt_len)` is
    // re-prefilled, at correct absolute positions. `Miss` = no usable prefix
    // OR an SWA layer wrapped (ring buffer not trimmable, see
    // `Gemma4Entry::can_truncate_to_block`) so a partial reuse cannot be made
    // cold-equal — fall back to a full re-prefill.
    enum CacheLookup {
        Exact {
            kv_caches: Vec<KvCache>,
            last_id: u32,
            piece: String,
        },
        Prefix {
            kv_caches: Vec<KvCache>,
            prefix_len: usize,
        },
        Miss,
    }

    // image prompts bypass the prompt cache. The cached K/V is keyed by
    // token ids only; an image prompt's K/V depends on the scattered vision
    // features, so it must never be reused for (or saved under) a token-id key.
    // Issue #26: codec-partitioned prompt-cache key. The query digest stream is
    // salted by `FNV_OFFSET ^ layout_key ^ codec_salt(kv_quant)` — identical to
    // the push seed below — so a slot stored under a different KV codec never
    // matches this request (no cross-codec serve), and codecs coexist as
    // distinct slots rather than thrash-evicting one another. On a RAM-only run
    // (layout_key=0) the push and query seeds are equal (`FNV_OFFSET ^
    // codec_salt`), so same-codec hits still land; digests differ from the
    // pre-#26 stream by the constant codec salt, harmless because the cache is
    // rebuilt per process.
    let cache_seed = crate::prompt_cache::FNV_OFFSET
        ^ crate::gemma4::prompt_cache::active_layout_key()
        ^ kv_quant.cache_key_salt();
    let lookup: CacheLookup = if image_prefill.is_some() {
        CacheLookup::Miss
    } else {
        PROMPT_CACHE.with_inner_mut(|guard| {
            let cache = guard.as_mut().unwrap();
            let mut raw_match = cache.find_best_prefix(prompt_ids, cache_seed);
            // on a RAM miss, try to serve the longest cached block-aligned
            // prefix from the SSD tier (no-op when no SSD source is attached —
            // tier OFF). A hydrate hit promotes the block into RAM; re-run
            // find_best_prefix so the promoted slot is matched (and quant-checked)
            // by the normal path below.
            if raw_match.is_none() && cache.hydrate_from_ssd(prompt_ids).is_some() {
                raw_match = cache.find_best_prefix(prompt_ids, cache_seed);
            }
            // Plan §D8 / Task 11.5: the snapshot is only safe to reuse when the
            // stored `KvQuant` discriminant matches the runtime quant for this
            // request. A mismatch (or a legacy `None`) means the cached
            // K/V layout / dtype / packing is incompatible with the model wired
            // up for this request — evict the slot, log a warn, and fall through
            // to a Miss (full re-prefill is always safe).
            let safe_match = match raw_match {
                Some((slot_idx, block_count)) => {
                    let stored = cache.slots[slot_idx].entry.kv_quant;
                    if stored == Some(kv_quant) {
                        Some((slot_idx, block_count))
                    } else {
                        tracing::warn!(
                            stored = ?stored,
                            runtime = ?kv_quant,
                            prompt_len = prompt_ids.len(),
                            "prompt cache KV quant mismatch — evicting entry, \
                             degrading to re-prefill"
                        );
                        cache.evict_slot(slot_idx);
                        None
                    }
                }
                None => None,
            };
            match safe_match {
                // Exact match: the cached entry's FULL token id list equals the
                // incoming prompt. Verified at token granularity, NOT by
                // `block_count * BLOCK_TOKENS == len` — the block-floored test is
                // essentially never true (only when len % 256 == 0) and was
                // misrouting identical-prompt repeats into the block-truncate +
                // tail-reprefill Prefix path, which produced wrong output here too
                // (22 -> 0 tokens for an identical re-request). Full-equality
                // reuse sidesteps truncation entirely (pre-C1 behaviour).
                Some((slot_idx, _block_count))
                    if cache.slots[slot_idx].entry.prompt_token_ids() == prompt_ids =>
                {
                    let slot = &cache.slots[slot_idx].entry;
                    match slot.deep_clone() {
                        Ok(cloned) => {
                            tracing::debug!(
                                prompt_len = prompt_ids.len(),
                                token_id = cloned.first_id,
                                "gemma4 generate_greedy: prompt cache EXACT HIT"
                            );
                            CacheLookup::Exact {
                                kv_caches: cloned.kv_caches,
                                last_id: cloned.first_id,
                                piece: cloned.first_piece,
                            }
                        }
                        Err(_) => CacheLookup::Miss,
                    }
                }
                // B1: strict-prefix multi-turn extension. The cached entry's FULL
                // token sequence is a token-for-token prefix of this prompt and
                // the new prompt extends it by >= 1 token. The post-prefill
                // snapshot is EXACTLY the state the new prompt needs at absolute
                // position `cached_len` for every layer type — including SWA
                // layers whose RotatingKvCache has WRAPPED. The deep-cloned ring
                // already holds the last `sliding_window` K/V of the cached
                // prefix (mlx-lm `state`/`meta_state` reuse contract), which is
                // precisely and sufficiently what the tail attends to under SWA.
                // We restore the snapshot as-is (NO truncation — `prefix_len ==
                // cached_len`, both full-attn `KVCache.offset` and SWA
                // `RotatingKVCache.offset` are `cached_len` post-prefill) and
                // forward only `prompt_ids[cached_len..]` at positions
                // `[cached_len, new_len)`. Because nothing is trimmed, the wrapped
                // SWA `22 -> 0` desync cannot arise. This supersedes the
                // block-aligned + `can_truncate_to_block` path for the multi-turn
                // case (which forced a full re-prefill Miss on every wrapped-SWA
                // turn — bug B1: prefill_ms grew with conversation length).
                Some((slot_idx, _block_count))
                    if cache.slots[slot_idx].entry.is_strict_prefix_of(prompt_ids) =>
                {
                    let slot = &cache.slots[slot_idx].entry;
                    let prefix_len = slot.prompt_token_ids().len();
                    match slot.deep_clone() {
                        Ok(cloned) => {
                            tracing::debug!(
                                prompt_len = prompt_ids.len(),
                                prefix_len,
                                tail_len = prompt_ids.len() - prefix_len,
                                "gemma4 generate_greedy: prompt cache PREFIX-EXACT HIT \
                             (B1 SWA snapshot/restore — restore + tail-only re-prefill)"
                            );
                            CacheLookup::Prefix {
                                kv_caches: cloned.kv_caches,
                                prefix_len,
                            }
                        }
                        Err(_) => CacheLookup::Miss,
                    }
                }
                // Genuine partial block-prefix hit: cached prompt shares the
                // first `block_count` full 256-token blocks then diverges.
                // gemma4 is pure-KV (no recurrent state), so block-truncating
                // the KV to `block_count*256` and re-prefilling the tail at
                // absolute positions `[prefix_len..prompt_len)` is cold-equal —
                // BUT only when every layer cache is losslessly trimmable
                // (mlx-lm `can_trim_prompt_cache`: `all(c.is_trimmable())`).
                // gemma4 SWA layers use a RotatingKvCache; once the cached
                // prompt exceeds `sliding_window` that ring buffer wraps and
                // `truncate_kv_to_block` would silently no-op on the SWA layers
                // while truncating the full-attention layers — desyncing the
                // caches and corrupting the tail (the original broken path:
                // longctx 22 -> 0 tokens). When `can_truncate_to_block` is
                // false we fall back to a full re-prefill (Miss); correctness
                // over speed for the wrapped-SWA case.
                Some((slot_idx, block_count)) if block_count >= 1 => {
                    // The tail `[prefix_len..prompt_len)` must be non-empty: the
                    // re-prefilled tail's final position is where we read the
                    // next-token logits from. If the matched blocks cover the
                    // whole (block-aligned) prompt, drop the last matched block
                    // back into the re-prefill tail (oMLX `prefix_cache.py`
                    // 421-437 keeps a re-prefill remainder past the last block
                    // boundary). If that leaves zero blocks, it is a Miss.
                    let prompt_blocks = prompt_ids.len() / crate::prompt_cache::BLOCK_TOKENS;
                    let effective_blocks =
                        if block_count * crate::prompt_cache::BLOCK_TOKENS >= prompt_ids.len() {
                            block_count.min(prompt_blocks).saturating_sub(1)
                        } else {
                            block_count
                        };
                    let slot = &cache.slots[slot_idx].entry;
                    if effective_blocks >= 1 && slot.can_truncate_to_block(effective_blocks) {
                        match slot.deep_clone() {
                            Ok(mut cloned) => {
                                cloned.truncate_kv_to_block(effective_blocks);
                                let prefix_len =
                                    effective_blocks * crate::prompt_cache::BLOCK_TOKENS;
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    block_count,
                                    effective_blocks,
                                    prefix_len,
                                    "gemma4 generate_greedy: prompt cache PREFIX HIT \
                                 (block-truncate + tail re-prefill)"
                                );
                                CacheLookup::Prefix {
                                    kv_caches: cloned.kv_caches,
                                    prefix_len,
                                }
                            }
                            Err(_) => CacheLookup::Miss,
                        }
                    } else {
                        tracing::debug!(
                            prompt_len = prompt_ids.len(),
                            block_count,
                            effective_blocks,
                            "gemma4 generate_greedy: partial prefix hit not usable \
                         (no tail blocks left, or an SWA layer wrapped / not \
                         trimmable) — falling back to Miss"
                        );
                        CacheLookup::Miss
                    }
                }
                Some(_) => CacheLookup::Miss,
                None => CacheLookup::Miss,
            }
        })
    };

    // Derive the initial ring size and the virtual ceiling (issue #25):
    // `--max-ctx` is a ceiling the ring grows lazily up to, not an eager
    // allocation. `initial_max_seq` is the small lazy start; `max_seq_ceiling`
    // caps growth and rejects over-long prompts.
    let (initial_max_seq, max_seq_ceiling) =
        kv_max_seq_and_ceiling(max_ctx_override, model.cfg.max_position_embeddings as i32);

    // ------------------------------------------------------------------
    // Path A: exact cache hit — skip prefill entirely. The exact-hit funnel
    // is just "caches + last_id → decode"; `pipelined_decode` is the call
    // site. GDN linear state does not exist for gemma4 (pure KV), so the
    // closure threads only `kv_caches`.
    // ------------------------------------------------------------------
    if let CacheLookup::Exact {
        mut kv_caches,
        last_id,
        piece,
    } = lookup
    {
        step_fn(steps.push_mut(ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        }));
        // exact-hit token into history.
        token_history.push(last_id);

        // EOS-stop. If the cached first token is an EOS, no decode steps —
        // return before the loop, with no kv-bytes store and no decode_profile
        // emission (the cached snapshot already accounts for its own bytes).
        if eos_ids.contains(&last_id) {
            return Ok(steps);
        }

        let stats = {
            let mut ctx = DecodeCtx {
                tokenizer,
                vocab,
                n_tokens,
                device,
                eos_ids,
                step_fn,
                constraint: constraint.take(),
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
                arch: "Gemma4ForConditionalGeneration",
                resolve_pieces: true,
            };
            pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
                model.forward_arr(y, 1, Some(&mut kv_caches), device)
            })?
        };

        let prefill_ms = 0.0_f64;
        let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
        let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
        let step_ms = (stats.step_total_ns as f64) / 1.0e6;
        let n = f64::from(stats.decode_steps.max(1));
        tracing::info!(
            target: "decode_profile",
            arch = "Gemma4ForConditionalGeneration",
            n_steps = stats.decode_steps,
            prefill_ms,
            forward_total_ms = forward_ms,
            eval_total_ms = eval_ms,
            step_total_ms = step_ms,
            forward_per_step_ms = forward_ms / n,
            eval_per_step_ms = eval_ms / n,
            cache_path = "exact",
            "decode_profile"
        );
        // store KV-cache bytes for the /metrics/cache endpoint.
        store_kv_cache_bytes(kv_caches.iter().map(|c| c.resident_bytes()).sum());
        return Ok(steps);
    }

    // ------------------------------------------------------------------
    // Path B (prefix hit) + Path C (miss): both end in the same chunked
    // prefill loop + decode. They differ only in (a) where `caches` comes
    // from and (b) which tokens get prefilled:
    //
    // - Prefix hit: `caches` = the block-truncated clone of a cached
    // snapshot (offset already == prefix_len). Only the tail
    // `prompt_ids[prefix_len..]` is fed through the loop, in
    // decode-mode `update` (NOT enter_prefill/exit_prefill — the clone
    // is already post-prefill quantized state; the tail appends at the
    // correct absolute positions because `forward_arr` derives its
    // RoPE/`base_offset` from `caches.first().offset()` == prefix_len).
    // This is the same cache machinery the Exact path's decode uses, so
    // it is token-for-token cold-equal: identical prefix K/V (produced
    // by a cold prefill of a byte-identical prefix) + tail K/V at the
    // right positions.
    // - Miss: `caches` = freshly allocated, the FULL prompt is prefilled
    // with the enter_prefill/exit_prefill bracketing (raw-BF16 →
    // one-shot quantize).
    //
    // Chunk size is per-arch; default 512 for gemma4 (large vocab + dense
    // SWA layers benefit from amortized FFI/flush overhead). Override via
    // `RMLX_PREFILL_CHUNK` (global) or `RMLX_PREFILL_CHUNK_GEMMA4`
    // (per-arch).
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("gemma4");
    let prefill_t0 = Instant::now();

    use super::config::LayerType;
    let sliding_window_i32 = model.cfg.sliding_window as i32;
    let n_layers = model.cfg.num_hidden_layers;

    // Issue #34: advise once if the resolved codec is estimated to increase
    // resident KV vs bf16 on this windowed+global layer mix. Windowed (SWA)
    // layers already run the bf16 rotating ring and are a no-op for the codec;
    // the warn fires when the per-global-layer warm-TTFT bf16 seed + codec
    // scales exceed the bytes the codec saves at the effective context. Keyed
    // on geometry only (head_dim / kv_heads / window), no arch branch.
    {
        let window_u64 = model.cfg.sliding_window as u64;
        let layer_shapes: Vec<KvLayerShape> = (0..n_layers)
            .map(|i| match model.cfg.layer_types[i] {
                LayerType::SlidingAttention => KvLayerShape {
                    head_dim: model.cfg.head_dim as u64,
                    kv_heads: model.cfg.num_key_value_heads as u64,
                    window: Some(window_u64),
                },
                LayerType::FullAttention => KvLayerShape {
                    head_dim: model.cfg.global_head_dim as u64,
                    kv_heads: model.cfg.num_global_key_value_heads as u64,
                    window: None,
                },
            })
            .collect();
        // Effective context the global layers will hold: the resolved ceiling
        // (or the prompt length if longer — either gives the correct sign).
        let eff_seq = (max_seq_ceiling.max(0) as u64).max(prompt_ids.len() as u64);
        warn_if_kv_codec_net_negative(kv_quant, &layer_shapes, eff_seq);
    }

    // `is_prefix` gates the enter_prefill/exit_prefill bracketing below: the
    // truncated clone is already post-prefill quantized, so the tail must
    // append via decode-mode `update`, exactly like the Exact-hit decode.
    let (mut caches, prefill_ids, is_prefix): (Vec<KvCache>, &[u32], bool) =
        if let CacheLookup::Prefix {
            kv_caches,
            prefix_len,
        } = lookup
        {
            tracing::debug!(
                prompt_len = prompt_ids.len(),
                prefix_len,
                tail_len = prompt_ids.len() - prefix_len,
                "gemma4 generate_greedy: PREFIX path — tail re-prefill only"
            );
            (kv_caches, &prompt_ids[prefix_len..], true)
        } else {
            // CacheLookup::Exact is handled by Path A above and returns early;
            // Miss falls through to a full fresh-cache prefill.
            // Allocate one KvCache per decoder layer using the selected
            // quant mode.
            //
            // SWA layers in bf16-KV mode (`KvQuant::None`) use the
            // byte-for-byte RotatingKvCache port (mirrors mlx-lm
            // `gemma4_text.py::Model.make_cache` line 686:
            // `RotatingKVCache(max_size=sliding_window)` for sliding layers,
            // `KVCache()` for full-attention). Quantized KV codecs stay on
            // the existing full-size path (pending follow-up).
            //
            // Force K8V8 for boundary layers (first head_n + last tail_n) to
            // protect output quality when base_quant uses aggressive V
            // compression. SWA layers keep their window regardless of the
            // quant override.
            let fresh: Vec<KvCache> = (0..n_layers)
                .map(|i| {
                    let window = match model.cfg.layer_types[i] {
                        LayerType::SlidingAttention => Some(sliding_window_i32),
                        LayerType::FullAttention => None,
                    };
                    let q = kv_quant_for_layer(
                        i,
                        n_layers,
                        kv_quant,
                        LAYER_ADAPTIVE_TAIL_N,
                        LAYER_ADAPTIVE_HEAD_N,
                    );
                    KvCache::with_quant_max_seq_window(q, initial_max_seq, window)
                        .with_max_seq_ceiling(max_seq_ceiling)
                        .with_layer_idx(i)
                })
                .collect();
            (fresh, prompt_ids, false)
        };

    // ------------------------------------------------------------------
    // Prefill: encode the prompt in `prefill_chunk` token chunks (default
    // 512 for gemma4; see binding at top of path-B section). Per chunk we
    // eval the KV-cache prefill_raw buffers (not the logits) to flush the
    // Metal command buffer under the ~10s watchdog while letting MLX skip
    // the wasted lm_head matmul for non-final chunks via lazy graph
    // pruning.
    //
    // An earlier version of the loop called `logits.eval()` per chunk, forcing the
    // full lm_head projection on every chunk's last position only to
    // discard it. On Gemma4 (vocab=262K) this dominated prefill cost.
    //
    // Each chunk advances the cache offset, so subsequent chunks
    // automatically use the "chunked prefill" SDPA mask path
    // (pick_attn_mask_mode returns "array" for offset>0+seq>1).
    //
    // enter_prefill() / exit_prefill() bracket the loop so K/V are stored
    // as raw BF16 during chunked prefill instead of being
    // quantize-dequantized on every chunk. exit_prefill() quantizes the
    // whole sequence in one shot when the loop completes.
    //
    // Non-MoE arch — chunk well below the gated_delta ts>=256 threshold
    // that only Qwen3.5MoE worries about (see `gated_delta_prefill_ops`).
    // ------------------------------------------------------------------
    // Phase span: prefill. Entered once per generate_greedy call, not per token.
    // Visible in samply/Instruments as a single region covering the full prefill.
    let _prefill_span = info_span!("prefill", prompt_len = prompt_ids.len()).entered();
    // Three prefill flush protocols, kept per-arch:
    //   - image  : one forward over the scatter-merged embeds (pre-hook),
    //              fresh caches enter_prefill → forward → exit_prefill.
    //   - prefix : B1 SWA snapshot/restore — the truncated clone is already
    //              post-prefill quantized, so the tail appends via decode-mode
    //              `update` with NO enter/exit brackets, flushing the
    //              quantized/decode_fp16 buffers via `eval_gpu_state`.
    //   - miss   : Fresh full re-prefill via the shared `chunked_prefill`
    //              (enter_prefill → per-chunk `eval_prefill_state` → exit).
    let prefill_logits = if let Some((embeds, masked_ids)) = image_prefill {
        // image_prefill is only ever Some on the Miss path (lookup forced to
        // Miss above), so `caches` is fresh. No chunking: an image prompt is
        // short (soft tokens + text) and fits one forward comfortably.
        for c in &mut caches {
            c.enter_prefill();
        }
        let seq_i32 = prompt_ids.len() as i32;
        let logits =
            match model.forward_arr_embeds(embeds, &masked_ids, seq_i32, Some(&mut caches), device)
            {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(error = %e, "generate_greedy: image prefill forward failed");
                    return Ok(steps);
                }
            };
        for c in &mut caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::warn!(error = %e, "generate_greedy: image exit_prefill failed");
                return Ok(steps);
            }
        }
        logits
    } else if is_prefix {
        // Prefix (B1): per-arch tail re-prefill. The cloned snapshot is already
        // post-prefill quantized (offset == prefix_len), so the tail appends
        // via decode-mode `update` — do NOT enter the raw-BF16 prefill
        // scaffolding (it would route the tail into the empty prefill_raw
        // buffer and exit_prefill would re-quantize only the tail). Non-final
        // chunks flush the quantized/decode_fp16 buffers via `eval_gpu_state`
        // (NOT `eval_prefill_state`); there is nothing to quantize on exit.
        let mut last_logits: Option<Array> = None;
        let mut prefill_ok = true;
        let n_chunks = prefill_ids.len().div_ceil(prefill_chunk);
        'prefill: for (chunk_idx, chunk) in prefill_ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            match model.forward_seq_with_cache(chunk, Some(&mut caches), device) {
                Ok(logits) => {
                    if is_last {
                        // Final chunk: keep the logits Array; the post-loop
                        // logits.eval() will materialize the lm_head matmul once.
                        last_logits = Some(logits);
                    } else {
                        // Non-final chunk: discard logits (lazy graph drops
                        // lm_head matmul) and flush only the decode-mode state.
                        for c in &caches {
                            if let Err(e) = c.eval_gpu_state() {
                                tracing::warn!(
                                    error = %e,
                                    chunk_len = chunk.len(),
                                    "generate_greedy: prefix tail chunk cache eval failed"
                                );
                                prefill_ok = false;
                                break 'prefill;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        prompt_len = prompt_ids.len(),
                        "generate_greedy: prefix tail chunk failed, returning empty"
                    );
                    prefill_ok = false;
                    break;
                }
            }
        }
        if !prefill_ok || last_logits.is_none() {
            return Ok(steps);
        }
        last_logits.unwrap()
    } else {
        // Miss: Fresh full re-prefill via the shared chunked_prefill helper. It
        // brackets the loop with enter_prefill() / exit_prefill(), evals only
        // the cache state on non-final chunks, and returns None on rejection.
        let Some(logits) = chunked_prefill(
            &mut caches,
            prompt_ids,
            prefill_chunk,
            device,
            "Gemma4ForConditionalGeneration",
            |chunk, caches| model.forward_seq_with_cache(chunk, Some(caches), device),
        )?
        else {
            return Ok(steps);
        };
        logits
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());

    // top-k logprob capture (0 = disabled, hot-loop zero-overhead).
    let lp_k = sampler_cfg.top_logprobs_k as usize;

    // Build the shared decode context ONCE for the prefill-tail selection AND
    // the decode loop (Prefix + Miss + image all reach here), so the
    // per-request state is borrowed a single time (a `&mut dyn` trait-object
    // reborrow is invariant in its lifetime — two separate `DecodeCtx` over the
    // same params would not compile). Intervening prefill-tail ops route
    // through `ctx.constraint` / `ctx.token_history` / `ctx.step_fn`.
    let mut ctx = DecodeCtx {
        tokenizer,
        vocab,
        n_tokens,
        device,
        eos_ids,
        step_fn,
        constraint: constraint.take(),
        sampler_cfg,
        rng,
        penalty_cfg,
        token_history,
        arch: "Gemma4ForConditionalGeneration",
        resolve_pieces: true,
    };

    // Prefill-tail token selection via the shared sampling fork. Gate the mask
    // ONCE here, before the post-selection `advance()` below — matching the old
    // prefill-tail timing (wants_mask can flip on engagement).
    let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
    let top = choose_token(&mut ctx, &logits_flat, mask_active)?;
    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    // A6.3: advance constraint regardless of mask state (warm-up scans).
    if let Some(c) = ctx.constraint.as_mut() {
        c.advance(last_id);
    }
    // A7.3: push prefill token into history.
    ctx.token_history.push(last_id);
    let prefill_total_ns = prefill_t0.elapsed().as_nanos();

    let piece = tokenizer
        .id_to_token(last_id)
        .unwrap_or_else(|| format!("<unk:{last_id}>"));

    tracing::debug!(
        step = 0,
        token_id = last_id,
        piece = %piece,
        max_abs_logit,
        nan_count,
        prompt_len = prompt_ids.len(),
        "generate_greedy prefill"
    );

    // Save snapshot after full prefill (: never for image prompts —
    // their K/V is not reconstructible from the token-id cache key).
    if !has_image {
        let kv_snap: rmlx_core::error::Result<Vec<_>> =
            caches.iter().map(|c| c.try_deep_clone()).collect();
        if let Ok(kvs) = kv_snap {
            // Materialize GPU arrays on the current inference thread before
            // storing in the prompt cache.  Each spawn_blocking request runs
            // on its own tokio thread with its own Metal GPU stream.  If these
            // lazy arrays are evicted on a *different* inference thread later,
            // that thread's eval_for_spill would fail with "There is no
            // Stream(gpu, N) in current thread".  Pre-eval here makes eval()
            // a no-op from any future thread.
            // The drain thread's spill() refcount-clones these arrays (no new graph) and evals
            // the clone, which shares the same buffers — so materializing here makes that eval a no-op.
            match kvs.iter().try_for_each(|c| c.eval_for_spill()) {
                Ok(()) => {
                    PROMPT_CACHE.with_inner_mut(|guard| {
                        if let Some(cache) = guard.as_mut() {
                            // salt the chained walk with the active layout_key so
                            // an entry that later spills lands under the same `(hash,
                            // layout_key)` row the hydrator will reconstruct. When
                            // the SSD tier is OFF, `active_layout_key()` returns 0 and the
                            // seed collapses to `FNV_OFFSET` — legacy un-salted digests.
                            // Issue #26: salt the stored digest stream by the
                            // active KV codec too, so the push seed matches the
                            // codec-partitioned query seed in `find_best_prefix`
                            // above. Stacks with `layout_key` (XOR) exactly like
                            // the SSD-tier salt.
                            let lk = crate::gemma4::prompt_cache::active_layout_key();
                            cache.push(Gemma4Entry {
                                prompt_token_ids: prompt_ids.to_vec(),
                                block_hashes: crate::prompt_cache::chained_block_hashes_seeded(
                                    prompt_ids,
                                    crate::prompt_cache::FNV_OFFSET
                                        ^ lk
                                        ^ kv_quant.cache_key_salt(),
                                ),
                                kv_caches: kvs,
                                first_id: last_id,
                                first_piece: piece.clone(),
                                kv_quant: Some(kv_quant),
                            });
                            tracing::debug!(
                                prompt_len = prompt_ids.len(),
                                token_id = last_id,
                                n_slots = cache.slots.len(),
                                cache_path = if is_prefix { "prefix" } else { "miss" },
                                "gemma4 generate_greedy: full-prompt snapshot saved"
                            );
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "gemma4 generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
            }
        }
    }

    // prefill is non-pipelined (last_id already materialised), so the
    // prefill token's logprobs come straight from this step's logits.
    let prefill_logprobs = if lp_k > 0 {
        capture_logprobs(&logits_flat, &top, lp_k)
    } else {
        None
    };
    (ctx.step_fn)(steps.push_mut(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: prefill_logprobs,
    }));
    // A7.3: prefill first token into history.
    ctx.token_history.push(last_id);

    if nan_count > 0 {
        return Ok(steps);
    }

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    // Decode: shared pipelined async loop, reusing the prefill-tail `ctx`.
    // Prefix + Miss + image all funnel here as "caches + last_id → decode";
    // the loop is the call site. The pipeline ordering (choose_token →
    // async_eval → drain previous pending → feed) overlaps host sampling with
    // the in-flight GPU forward; see decode_loop.rs. gemma4 is pure-KV, so the
    // closure threads only `caches`.
    let stats = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
        model.forward_arr(y, 1, Some(&mut caches), device)
    })?;

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
           target: "decode_profile",
           arch = "Gemma4ForConditionalGeneration",
           n_steps = stats.decode_steps,
           prefill_ms,
           forward_total_ms = forward_ms,
           eval_total_ms = eval_ms,
           step_total_ms = step_ms,
           forward_per_step_ms = forward_ms / n,
           eval_per_step_ms = eval_ms / n,
    // Path B (prefix hit, incl. B1 SWA snapshot/restore) and Path C
    // (miss) both reach this shared decode-profile emission. Report the
    // path that actually ran so the e2e gate can confirm the B1
    // restore-and-tail path fired (flat prefill_ms across a multi-turn
    // session is the prefix path; a cold full re-prefill is a miss).
           cache_path = if is_prefix { "prefix" } else { "miss" },
           "decode_profile"
       );

    // N16: store KV-cache bytes for the /metrics/cache endpoint.
    let kv_bytes: u64 = caches.iter().map(|c| c.resident_bytes()).sum();
    store_kv_cache_bytes(kv_bytes);

    Ok(steps)
}
