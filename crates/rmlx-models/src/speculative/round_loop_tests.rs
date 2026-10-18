//! `run_rounds` at a context ceiling, against a drafter that scripts every
//! round.
//!
//! The verifier is a one-layer attention-only Qwen3.5 of a few synthetic
//! weights, loaded on the CPU. The drafter never runs its forward: its prefill
//! and its verify write positions into the loop's real KV caches, so a round
//! that writes past the ceiling fails with the cache's own
//! [`Error::KvCeilingExceeded`]. Every proposal is accepted, the widest round a
//! request can run.
//!
//! Mutations and the case that catches each:
//! - `round_block` narrows to `remaining + 2`:
//!   `a_request_cut_to_the_room_under_the_ceiling_keeps_its_output`, on both
//!   seeds.
//! - `run_rounds` reads the request's token count and not the budget: the same
//!   test.
//! - the budget is read after the prefill: `a_prompt_with_no_room_is_refused_before_any_prefill`.

#![allow(
    clippy::expect_used,
    reason = "unit-test scaffolding: a panic is how an assertion failure surfaces"
)]

use std::sync::atomic::AtomicBool;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use super::super::tests::tiny_tokenizer;
use super::super::SpecLoop;
use super::{
    run_rounds, Conditioning, Prefilled, ReportSkippedBy, RoundCfg, RoundCtx, RoundDrafter,
    RoundOutcome, Seed, Verdict, VerifierOffsetBasis,
};
use crate::arch::Architecture;
use crate::sampler::SamplerConfig;

/// Tokens of the prompt.
const PROMPT: usize = 10;

/// Tokens a request asks for.
const ASKED: usize = 40;

/// Tokens the ceiling of the short cases leaves after the prompt.
const ROOM: usize = 8;

/// The block a request is configured with, the verifier's token included.
const BLOCK: usize = 5;

/// A token that is no stop token of the tiny vocabulary.
const FILLER: u32 = 3;

/// Which prefill the drafter scripts.
#[derive(Clone, Copy, Debug)]
enum Prefill {
    /// The prefill covers the whole prompt and draws the seed the request emits.
    SeedEmitted,
    /// The prefill stops short of the prompt's last token, which the first
    /// round carries.
    SeedCarried,
}

/// A drafter whose rounds all accept, writing straight into the verifier caches.
struct ScriptedRounds {
    prefill: Prefill,
    prefills: usize,
}

/// Append `n` positions to every layer of the verifier's cache stack.
fn write_positions(ctx: &mut RoundCtx<'_>, n: usize) -> Result<()> {
    let n = i32::try_from(n).expect("a few positions");
    let data = vec![0.0_f32; usize::try_from(n).expect("non-negative") * 4];
    let kv = Array::from_f32_slice(&data, &[1, 1, n, 4])?;
    for layer in &mut ctx.kv {
        layer.update(&kv, &kv, Device::Cpu)?;
    }
    Ok(())
}

impl RoundDrafter for ScriptedRounds {
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy = ReportSkippedBy::TheSeedExit;
    const VERIFIER_OFFSET_BASIS: VerifierOffsetBasis = VerifierOffsetBasis::BeforeTheForward;

    fn prefill(
        &mut self,
        ctx: &mut RoundCtx<'_>,
        prompt: &[u32],
        _cancel: &AtomicBool,
    ) -> Result<Prefilled> {
        self.prefills += 1;
        let (seed, written) = match self.prefill {
            Prefill::SeedEmitted => (Seed::Emitted(FILLER), prompt.len()),
            Prefill::SeedCarried => (Seed::Carried(FILLER), prompt.len() - 1),
        };
        write_positions(ctx, written)?;
        Ok(Prefilled {
            seed,
            prefill_ns: 0,
            restricted_read_back: false,
            projects_conditioning: false,
        })
    }

    fn propose(&mut self, _ctx: &mut RoundCtx<'_>, _carry: u32, block: usize) -> Result<Vec<u32>> {
        Ok(vec![FILLER; block - 1])
    }

    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict> {
        write_positions(ctx, fed.len())?;
        Ok(Verdict {
            accept: fed.len() - 1,
            commit: vec![FILLER; fed.len().min(remaining)],
            verify_ns: 0,
            walk_ns: 0,
            restricted: 0,
        })
    }

    fn condition(
        &mut self,
        _ctx: &RoundCtx<'_>,
        _verdict: &Verdict,
        _outcome: RoundOutcome,
    ) -> Result<Option<Conditioning>> {
        Ok(None)
    }

    fn carry(&self, f: &mut dyn FnMut(&[(&str, &Array)])) -> Result<()> {
        f(&[]);
        Ok(())
    }
}

/// Run one request of [`ASKED`] tokens under `max_ctx` and return the number of
/// tokens it emitted and the number of prefills the drafter ran.
fn run(prefill: Prefill, max_ctx: i32) -> (Result<usize>, usize) {
    let verifier: Architecture =
        crate::qwen3_5_moe::attention_only_architecture().expect("load the tiny verifier");
    let tokenizer = tiny_tokenizer();
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut drafter = ScriptedRounds {
        prefill,
        prefills: 0,
    };
    let prompt = vec![FILLER; PROMPT];
    let steps = run_rounds(
        &verifier,
        &mut drafter,
        &prompt,
        &mut |_| None,
        &RoundCfg {
            loop_kind: SpecLoop::MtpSidecar,
            block_size: BLOCK,
            n_tokens: ASKED,
            eos_ids: &[],
            tokenizer: &tokenizer,
            sampler_cfg: &sampler_cfg,
            charged: false,
            kv_quant_override: None,
            max_ctx_override: Some(max_ctx),
        },
        None,
        Device::Cpu,
        &AtomicBool::new(false),
    )
    .map(|(steps, _widest_block)| steps.len());
    (steps, drafter.prefills)
}

fn ceiling(room: usize) -> i32 {
    i32::try_from(PROMPT + room).expect("a few positions")
}

/// A request that asks for more than the ceiling leaves ends at the room, with
/// its output, whichever seed the drafter draws. The last round is the one that
/// reaches the ceiling exactly. A ceiling with room for the whole request is
/// the control.
#[test]
fn a_request_cut_to_the_room_under_the_ceiling_keeps_its_output() {
    for prefill in [Prefill::SeedEmitted, Prefill::SeedCarried] {
        let (cut, _) = run(prefill, ceiling(ROOM));
        assert_eq!(
            cut.expect("a request at the ceiling keeps its output"),
            ROOM,
            "{prefill:?}: the ceiling leaves {ROOM} tokens after the prompt"
        );
        let (control, _) = run(prefill, ceiling(ASKED));
        assert_eq!(
            control.expect("a ceiling with room for the whole request"),
            ASKED,
            "{prefill:?}: a ceiling with room must not cut the request"
        );
    }
}

/// A prompt that fills the ceiling is refused with its values before the
/// drafter prefills.
#[test]
fn a_prompt_with_no_room_is_refused_before_any_prefill() {
    for prefill in [Prefill::SeedEmitted, Prefill::SeedCarried] {
        let (result, prefills) = run(prefill, ceiling(0));
        assert!(
            matches!(
                result,
                Err(Error::KvCeilingExceeded { requested, ceiling: c })
                    if requested == ceiling(1) && c == ceiling(0)
            ),
            "{prefill:?}: {result:?}"
        );
        assert_eq!(prefills, 0, "{prefill:?}: the prompt was prefilled");
    }
}
