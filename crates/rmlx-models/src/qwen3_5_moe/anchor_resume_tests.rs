//! A request resumed from a RAM entry's saved state, judged on a dense Qwen3.5
//! checkpoint.
//!
//! Each arm stores a prompt through the production cold path (which captures the
//! anchor mid-chunk), then asks for a longer or diverging one. The resume is
//! judged on its tail logits, not on the decoded stream: the argmax at every
//! tail position first, then the per-logit bound, against a cold prefill split
//! where the resume splits. The engine's branch is asserted, because a Miss
//! agrees with a cold baseline for free.
//!
//! "Where the resume splits" is read, not assumed. Each run records the
//! attention calls MLX receives, whose `(query rows, key rows)` are the forward
//! boundaries. The warm side is the pieces that built the resumed state plus its
//! tail; the cold side is the pieces the baseline forwarded. The two lists must
//! be equal before anything is compared, and then the logits must be equal.

use super::attention_calls::{assert_the_recorder_is_live, recorded, AttentionCall};
use super::*;
use crate::prompt_cache::{Anchor, Consumed};
use rmlx_core::error::{Error, Result};
use rmlx_kv_quant::{KvCache, LinearAttnCache};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

const DENSE_SLUG: &str = "mlx-community__Qwen3.8-27B-4bit";
const DENSE_ARCHS: [&str; 1] = ["Qwen3_5ForConditionalGeneration"];

const KV_QUANT: rmlx_kv_quant::KvQuant = rmlx_kv_quant::KvQuant::None;
const MAX_SEQ: i32 = 8192;
const N_DECODE: usize = 6;

/// Deterministic synthetic ids in `1..9999`, never 0.
fn synthetic_ids(len: usize, salt: u32) -> Vec<u32> {
    (1u32..=len as u32)
        .map(|i| (i.wrapping_mul(7).wrapping_add(salt) % 9999).max(1))
        .collect()
}

/// The first `n` ids.
fn head(ids: &[u32], n: usize) -> Vec<u32> {
    ids.iter().copied().take(n).collect()
}

fn fail(what: String) -> Error {
    Error::Model(what)
}

fn reset_cache() {
    prompt_cache::ensure_prompt_cache(4);
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });
}

/// Where the stored entry for a prompt of `len` tokens holds its anchor.
fn stored_anchor_pos(len: usize) -> Option<usize> {
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        let cache = guard.as_ref()?;
        let slot = cache
            .slots
            .iter()
            .find(|s| s.entry.prompt_token_ids.len() == len)?;
        slot.entry.anchor.as_ref().map(|a| a.pos)
    })
}

/// One forward as the attention call shows it: query rows, key rows.
type Cut = (i64, i64);

/// The distinct forwards of a run, in order, ending at or before `up_to` keys.
/// A decode step ends past the prompt, so `up_to = prompt length` leaves it out.
fn cuts(calls: &[AttentionCall], up_to: usize) -> Vec<Cut> {
    let mut out: Vec<Cut> = Vec::new();
    for call in calls
        .iter()
        .filter(|c| usize::try_from(c.k_rows).is_ok_and(|k| k <= up_to))
    {
        let cut = (call.caller_q_rows, call.k_rows);
        if out.last() != Some(&cut) {
            out.push(cut);
        }
    }
    out
}

struct Gate<'a> {
    model: &'a Qwen3_5MoeText,
    tokenizer: &'a tokenizers::Tokenizer,
}

impl Gate<'_> {
    fn generate(&self, ids: &[u32]) -> Result<Vec<u32>> {
        self.generate_under(ids, MAX_SEQ)
    }

    /// `generate` under a `--max-ctx` of `ceiling` tokens.
    fn generate_under(&self, ids: &[u32], ceiling: i32) -> Result<Vec<u32>> {
        let sampler_cfg = crate::sampler::SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: Some(0),
            top_logprobs_k: 0,
        };
        let penalty_cfg = crate::sampler::PenaltyConfig::default();
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            self.model,
            self.tokenizer,
            ids,
            N_DECODE,
            Device::Gpu,
            KV_QUANT,
            Some(ceiling),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
            &AtomicBool::new(false),
        )?;
        Ok(steps.into_iter().map(|s| s.token_id).collect())
    }

    /// Tail logits of the resume the engine would take for `request`: the cloned
    /// entry's caches, the tail forwarded on top. Also the forwards of that tail.
    fn resumed_rows(&self, request: &[u32]) -> Result<(usize, Vec<Vec<f32>>, Vec<Cut>)> {
        let consumed = prompt_cache::PROMPT_CACHE.consume(
            request,
            KV_QUANT,
            self.model.cfg.num_hidden_layers,
            false,
            self.model.model_sig,
        );
        let Consumed::Reuse { entry, kind } = consumed else {
            return Err(fail("the request did not reach a resume".to_owned()));
        };
        let prefix_len = match kind {
            crate::prompt_cache::ReuseKind::StrictPrefix { prefix_len } => prefix_len,
            crate::prompt_cache::ReuseKind::BlockTruncate { effective_blocks } => {
                effective_blocks * BLOCK_TOKENS
            }
        };
        let Qwen35MoeEntry {
            mut kv_caches,
            mut lin_caches,
            ..
        } = entry;
        let (rows, calls) =
            recorded(|| self.tail_rows(request, prefix_len, &mut kv_caches, &mut lin_caches));
        Ok((prefix_len, rows?, cuts(&calls, usize::MAX)))
    }

    fn tail_rows(
        &self,
        request: &[u32],
        prefix_len: usize,
        kv_caches: &mut [KvCache],
        lin_caches: &mut [LinearAttnCache],
    ) -> Result<Vec<Vec<f32>>> {
        let tail = request
            .get(prefix_len..)
            .ok_or_else(|| fail("prefix past the request".to_owned()))?;
        let logits = self.model.forward_seq_last_k_with_cache(
            tail,
            tail.len(),
            kv_caches,
            Some(lin_caches),
            Device::Gpu,
        )?;
        Ok(logit_rows(&logits, tail.len(), Device::Gpu))
    }

    /// Tail logits of `request` forwarded in the pieces that end at `ends`, then
    /// from `prefix_len`. The pieces up to `cold_len` are a prefill; the rest
    /// are tails on top of its caches. Also the forwards the baseline made.
    fn cold_split_rows(
        &self,
        request: &[u32],
        ends: &[usize],
        cold_len: usize,
        prefix_len: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Cut>)> {
        let (prefill_ends, hops): (Vec<usize>, Vec<usize>) =
            ends.iter().copied().partition(|&end| end <= cold_len);
        let reached = hops.last().copied().unwrap_or(cold_len);
        if prefill_ends.last().copied() != Some(cold_len) || reached != prefix_len {
            return Err(fail(format!(
                "the pieces {ends:?} do not run from a prefill of {cold_len} to {prefix_len}"
            )));
        }
        let (rows, calls) = recorded(|| -> Result<Vec<Vec<f32>>> {
            let (mut kv, mut lin) = prefilled_in_pieces(
                self.model,
                &head(request, cold_len),
                &prefill_ends,
                KV_QUANT,
                MAX_SEQ,
            );
            let mut at = cold_len;
            for &hop in &hops {
                let piece = request
                    .get(at..hop)
                    .ok_or_else(|| fail("hop outside the request".to_owned()))?;
                self.model.forward_seq_with_cache(
                    piece,
                    Some(&mut kv),
                    Some(&mut lin),
                    Device::Gpu,
                )?;
                kv.iter().try_for_each(KvCache::eval_gpu_state)?;
                lin.iter().try_for_each(LinearAttnCache::eval_for_spill)?;
                at = hop;
            }
            self.tail_rows(request, prefix_len, &mut kv, &mut lin)
        });
        Ok((rows?, cuts(&calls, usize::MAX)))
    }

    /// Tail logits of one forward over the whole `request`.
    fn single_shot_rows(&self, request: &[u32], tail_len: usize) -> Result<Vec<Vec<f32>>> {
        let n_layers = self.model.cfg.num_hidden_layers;
        let mut kv: Vec<KvCache> = crate::kv_cache::kv_layer_quants(n_layers, KV_QUANT, false)
            .into_iter()
            .enumerate()
            .map(|(i, q)| KvCache::with_quant_max_seq(q, MAX_SEQ).with_layer_idx(i))
            .collect();
        let mut lin: Vec<LinearAttnCache> = (0..n_layers).map(|_| LinearAttnCache::new()).collect();
        let logits = self.model.forward_seq_last_k_with_cache(
            request,
            tail_len,
            &mut kv,
            Some(&mut lin),
            Device::Gpu,
        )?;
        Ok(logit_rows(&logits, tail_len, Device::Gpu))
    }

    /// `generate` and the forwards of its prompt.
    fn generate_cuts(&self, ids: &[u32]) -> Result<(Vec<u32>, Vec<Cut>)> {
        self.generate_cuts_under(ids, MAX_SEQ)
    }

    /// The same under a `--max-ctx` of `ceiling` tokens.
    fn generate_cuts_under(&self, ids: &[u32], ceiling: i32) -> Result<(Vec<u32>, Vec<Cut>)> {
        let (tokens, calls) = recorded(|| self.generate_under(ids, ceiling));
        Ok((tokens?, cuts(&calls, ids.len())))
    }

    /// One arm: store `stored` cold, resume `request` from it, judge the resume
    /// on its logits and on its stream.
    fn judge(&self, arm: &str, stored: &[u32], request: &[u32], branch: &str) -> Result<()> {
        reset_cache();
        let (_, built) = self.generate_cuts(stored)?;
        let history = History {
            stored,
            built,
            stages: vec![(0, stored.len())],
            stream: true,
        };
        self.judge_stored(arm, request, branch, &history)
    }

    /// The same, for a cache that already holds the entry `history` names.
    fn judge_stored(
        &self,
        arm: &str,
        request: &[u32],
        branch: &str,
        history: &History,
    ) -> Result<()> {
        let stored = history.stored;
        assert_eq!(
            stored_anchor_pos(stored.len()),
            Anchor::pos_for(stored.len()),
            "({arm}) the cold prefill must have saved its anchor"
        );
        assert_eq!(
            moe_consume_branch(self.model, request, KV_QUANT),
            branch,
            "({arm}) the engine must reach the resume this arm is named for — a Miss \
             re-prefills the same prompt and agrees with a cold baseline while exercising nothing"
        );

        let (prefix_len, warm_rows, warm_tail) = self.resumed_rows(request)?;
        // The forwards that built the resumed state: those that end at or before
        // the resume point. A piece past it belongs to the stored entry's end
        // state, which the resume does not use.
        let used: Vec<Cut> = history
            .built
            .iter()
            .copied()
            .filter(|c| usize::try_from(c.1).is_ok_and(|k| k <= prefix_len))
            .collect();
        let warm_cuts = [used.clone(), warm_tail].concat();
        assert!(
            warm_cuts.last().map(|c| c.1) == i64::try_from(request.len()).ok(),
            "({arm}) the forwards must end at the last token of the request: {warm_cuts:?}"
        );
        // The pieces the chunking rules give, independent of what was recorded.
        let ends: Vec<usize> = predicted_ends(&history.stages)
            .into_iter()
            .filter(|&end| end <= prefix_len)
            .collect();
        let recorded_ends: Vec<usize> = used
            .iter()
            .filter_map(|c| usize::try_from(c.1).ok())
            .collect();
        assert_eq!(
            ends, recorded_ends,
            "({arm}) the chunk size and the anchor position must predict the forwards the warm \
             state was built in"
        );
        let cold_len = history.stages.first().map_or(0, |s| s.1).min(prefix_len);
        let (cold_rows, cold_cuts) = self.cold_split_rows(request, &ends, cold_len, prefix_len)?;
        assert_eq!(
            warm_cuts, cold_cuts,
            "({arm}) the cold baseline must be forwarded in the pieces the warm state was \
             built in (query rows, key rows)"
        );
        let tail_len = warm_rows.len();
        let single_rows = self.single_shot_rows(request, tail_len)?;

        let mut argmax_match = 0usize;
        let mut max_diff = 0.0f32;
        let mut max_diff_single = 0.0f32;
        for (pos, ((warm, cold), single)) in warm_rows
            .iter()
            .zip(cold_rows.iter())
            .zip(single_rows.iter())
            .enumerate()
        {
            assert!(
                !warm.iter().any(|x| x.is_nan()) && !cold.iter().any(|x| x.is_nan()),
                "({arm}) NaN in the tail logits at tail position {pos}"
            );
            if argmax_row(warm) == argmax_row(cold) {
                argmax_match += 1;
            }
            max_diff = max_diff.max(max_abs_diff(warm, cold));
            max_diff_single = max_diff_single.max(max_abs_diff(warm, single));
        }
        println!(
            "GATE {arm}: branch={branch} cuts_equal=true cuts={warm_cuts:?} prefix_len={prefix_len} \
             tail_len={tail_len} argmax_match={argmax_match}/{tail_len} \
             max_abs_diff_same_split={max_diff} max_abs_diff_single_shot={max_diff_single} \
             (bound {TAIL_LOGIT_NOISE_BOUND})"
        );
        assert_eq!(
            argmax_match,
            tail_len,
            "({arm}) the resume's argmax differs from the cold prefill split at the same \
             boundaries at {} of {tail_len} tail positions",
            tail_len - argmax_match
        );
        assert!(
            max_diff == 0.0,
            "({arm}) the same forwards in the same order must give the same logits: \
             max_abs_diff={max_diff}"
        );
        assert!(
            max_diff_single < TAIL_LOGIT_NOISE_BOUND,
            "({arm}) tail logits left the noise bound against one forward over the whole \
             prompt: {max_diff_single}"
        );

        if !history.stream {
            return Ok(());
        }
        let (warm_tokens, warm_gen) = self.generate_cuts(request)?;
        let warm_stream_cuts = [used, warm_gen].concat();
        // The override clamps to the largest chunk. Past it the cold prefill
        // splits at the anchor on its own, at the resume point.
        crate::prefill_chunk::set_prefill_chunk(
            prefix_len.min(crate::prefill_chunk::PREFILL_CHUNK_MAX),
        );
        reset_cache();
        let cold = self.generate_cuts(request);
        crate::prefill_chunk::set_prefill_chunk(0);
        let (cold_tokens, cold_stream_cuts) = cold?;
        println!(
            "GATE {arm}: stream cuts_equal={} warm={warm_tokens:?} cold_same_split={cold_tokens:?}",
            warm_stream_cuts == cold_stream_cuts
        );
        assert_eq!(
            warm_stream_cuts, cold_stream_cuts,
            "({arm}) the cold generation must be forwarded in the pieces the warm one was"
        );
        assert_eq!(
            warm_tokens, cold_tokens,
            "({arm}) the resumed stream must equal a cold prefill split at the same boundary, \
             token for token"
        );
        Ok(())
    }
}

/// The ends of the forwards the chunking rules give a state built in `stages`:
/// token ranges, the first a prefill and each later one a tail on top of the
/// last. Each stage is cut every chunk from its start and where the anchor of
/// its end falls inside it.
fn predicted_ends(stages: &[(usize, usize)]) -> Vec<usize> {
    let mut ends = Vec::new();
    for (i, &(from, to)) in stages.iter().enumerate() {
        let chunk = if i == 0 {
            crate::prefill_chunk::resolve("qwen3_5_moe").0
        } else {
            crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe")
        };
        let mut stage: Vec<usize> = (from + chunk..to).step_by(chunk).collect();
        stage.push(to);
        stage.extend(Anchor::pos_for(to).filter(|&pos| pos > from));
        stage.sort_unstable();
        stage.dedup();
        ends.extend(stage);
    }
    ends
}

/// How the stored entry's state was reached.
struct History<'a> {
    stored: &'a [u32],
    /// The forwards that built it, in order, as recorded.
    built: Vec<Cut>,
    /// The token ranges it was built in: a prefill, then any tails on top.
    stages: Vec<(usize, usize)>,
    /// Whether the stream can be held to a cold generation. Not when the stored
    /// entry was itself resumed: its tail ran in another mode than a prefill.
    stream: bool,
}

fn dense_snapshot(test: &str) -> Option<PathBuf> {
    crate::test_snapshot::snapshot_by_slug(test, DENSE_SLUG, &DENSE_ARCHS)
}

/// Strict prefix, anchor, and the entries a resume stores, on the dense Qwen3.8
/// checkpoint.
#[test]
#[ignore = "loads a model and drives the Metal GPU"]
fn qwen3_5_anchor_resume_gate() -> Result<()> {
    let name = "qwen3_5_anchor_resume_gate";
    let Some(model_dir) = dense_snapshot(name) else {
        return Ok(());
    };
    assert_the_recorder_is_live(Device::Gpu);
    let model = load_from_path(&model_dir)?;
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| fail(format!("tokenizer: {e}")))?;
    let gate = Gate {
        model: &model,
        tokenizer: &tokenizer,
    };

    // The measured shape: the new prompt differs at the stored prompt's last
    // token (the previous turn's generation prompt rendered differently).
    let stored = synthetic_ids(700, 0);
    let diverged = [head(&stored, 699), synthetic_ids(60, 41)].concat();
    gate.judge("anchor", &stored, &diverged, "Anchor{prefix_len=512}")?;

    // The new prompt extends the stored one.
    let extended = [stored.clone(), synthetic_ids(40, 43)].concat();
    gate.judge(
        "strict_prefix",
        &stored,
        &extended,
        "StrictPrefix{prefix_len=700}",
    )?;

    // A stored prompt that spans two chunks, so the anchor is captured inside
    // the second one. The request is short enough that its own anchor is the
    // resume point, so a cold prefill splits where the resume does.
    let long = synthetic_ids(3000, 3);
    let long_diverged = [head(&long, 2999), synthetic_ids(20, 47)].concat();
    gate.judge(
        "anchor_second_chunk",
        &long,
        &long_diverged,
        "Anchor{prefix_len=2816}",
    )?;

    // What a resume stores: the anchor it inherits when the tail crosses no
    // block, and the one it captures when the tail crosses one.
    reset_cache();
    let (_, built) = gate.generate_cuts(&stored)?;
    gate.generate(&extended)?;
    assert_eq!(
        stored_anchor_pos(extended.len()),
        Some(512),
        "a strict-prefix resume whose tail crosses no block inherits the stored anchor"
    );
    let crossing = [stored.clone(), synthetic_ids(600, 53)].concat();
    assert_eq!(
        moe_consume_branch(&model, &crossing, KV_QUANT),
        "StrictPrefix{prefix_len=700}",
        "the entry the last arm judges is built by a strict-prefix resume of the stored prompt"
    );
    let (_, resumed) = gate.generate_cuts(&crossing)?;
    assert_eq!(
        stored_anchor_pos(crossing.len()),
        Anchor::pos_for(crossing.len()),
        "a strict-prefix resume whose tail crosses a block captures the new anchor mid-tail"
    );
    assert_eq!(stored_anchor_pos(crossing.len()), Some(1280));
    let crossing_diverged = [head(&crossing, crossing.len() - 1), synthetic_ids(70, 59)].concat();
    gate.judge_stored(
        "anchor_captured_in_a_tail",
        &crossing_diverged,
        "Anchor{prefix_len=1280}",
        &History {
            stored: &crossing,
            built: [built, resumed].concat(),
            stages: vec![(0, stored.len()), (stored.len(), crossing.len())],
            stream: false,
        },
    )?;

    // A reused prefix runs under the ceiling of the request that reuses it, not
    // the one it was stored under, whichever way they differ. A lower ceiling
    // refuses a prompt above it as a cold run does; a higher one lets the tail
    // grow past the stored one.
    reset_cache();
    gate.generate(&stored)?;
    assert_eq!(
        moe_consume_branch(&model, &extended, KV_QUANT),
        "StrictPrefix{prefix_len=700}",
        "the lower-ceiling arm must reach the resume, or the refusal below is a cold one"
    );
    let warm = gate.generate_under(&extended, 600);
    reset_cache();
    let cold = gate.generate_under(&extended, 600);
    assert!(
        matches!(warm, Err(Error::KvCeilingExceeded { ceiling: 600, .. }))
            && matches!(cold, Err(Error::KvCeilingExceeded { ceiling: 600, .. })),
        "a prompt above the request's ceiling is refused with the cold run's error: \
         resumed {warm:?}, cold {cold:?}"
    );

    reset_cache();
    let (_, built) = gate.generate_cuts_under(&stored, 720)?;
    assert_eq!(
        moe_consume_branch(&model, &extended, KV_QUANT),
        "StrictPrefix{prefix_len=700}",
        "the higher-ceiling arm must reach the resume"
    );
    let (warm, warm_gen) = gate.generate_cuts_under(&extended, MAX_SEQ)?;
    let warm_cuts = [built, warm_gen].concat();
    crate::prefill_chunk::set_prefill_chunk(stored.len());
    reset_cache();
    let cold = gate.generate_cuts_under(&extended, MAX_SEQ);
    crate::prefill_chunk::set_prefill_chunk(0);
    let (cold, cold_cuts) = cold?;
    assert_eq!(
        warm_cuts, cold_cuts,
        "the cold generation must be forwarded in the pieces the resumed one was"
    );
    println!(
        "GATE ceiling: stored under 720, resumed under {MAX_SEQ}: warm={warm:?} cold={cold:?}"
    );
    assert_eq!(
        warm, cold,
        "a request under a higher ceiling than the stored one grows past it, as a cold run does"
    );
    Ok(())
}
