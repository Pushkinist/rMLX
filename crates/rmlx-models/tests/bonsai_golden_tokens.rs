//! Per-arch golden-token decode gate for Bonsai (dense Qwen3).
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Covers the dense
//! `Qwen3ForCausalLM` backbone on its production Mixed KV path.
//!
//! Model: `prism-ml__Ternary-Bonsai-8B-mlx-2bit`.
//! KV quant: Mixed{ K=8, V=4, group=64 } — the resolver default for Qwen3
//! dense 2-bit, feeding the quantized 3-tuple straight into SDPA.
//!
//! The snapshot resolves from `RMLX_O_MODELS_ROOT` by slug, so no per-run
//! variable is needed on a machine holding it (see `tests/common/mod.rs`).
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/Ternary-Bonsai-8B-mlx-2bit \
//! cargo test -p rmlx-models --test bonsai_golden_tokens -- --ignored
//! Then gate:
//! cargo test -p rmlx-models --test bonsai_golden_tokens -- --ignored

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

mod common;

use rmlx_kv_quant::KvQuant;

/// The snapshot these tests cover, and the architectures they were recorded
/// against.
const MODEL: common::GoldenModel = common::GoldenModel {
    slug: "prism-ml__Ternary-Bonsai-8B-mlx-2bit",
    archs: &["Qwen3ForCausalLM"],
};

/// The production Mixed stack, at the codec every one of its layers runs.
///
/// The fixture tag carries `_floor8` because the ids are set by the *effective*
/// per-layer vector, not by the requested base: `kv_layer_quants` promotes the
/// first 2 and last 8 of Bonsai's 36 layers to `mixed_k8g64_v8g64`. The
/// previous fixture, `bonsai_8b_mixed_k8g64_v4g64`, was recorded when that
/// promotion landed on `K8V8` — a codec that materialises no store and decodes
/// bf16 — so its ids came from a stack with 10 unquantized layers under a name
/// that said `mixed_k8g64_v4g64`. It is retired rather than refreshed: its ids
/// diverge at index 18 (320 -> 1075) at a top-2 margin of 0.1875, well above
/// `REGEN_MAX_TIE_MARGIN`, and the regen guard refuses them. Widening that
/// bound to admit a numerics change would retire the guard for every change it
/// exists for.
///
/// What the old fixture actually pinned — that this codec agreed with bf16 on
/// this prompt — is kept, under the name of the thing it was measuring, by
/// [`bonsai_golden_tokens_none`].
#[ignore]
#[test]
fn bonsai_golden_tokens_mixed() {
    let Some(model_path) = common::model_for(&MODEL, "bonsai_golden_tokens_mixed") else {
        return;
    };
    common::run_golden_test(
        "bonsai_8b_mixed_k8g64_v4g64_floor8",
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
        &model_path,
    );
}

/// The bf16 control: `--kv-quant none`, every layer unquantized.
///
/// Exempt from the boundary promotion by construction (a base that quantizes
/// neither side has no loss to buy back), so this fixture pins one codec on all
/// 36 layers and is the reference any quantized stack is read against. It is
/// also the gate the Mixed fixture used to be doubling as: its ids are the ones
/// `bonsai_8b_mixed_k8g64_v4g64` held, which that codec could only produce
/// while 10 of its 36 layers were secretly decoding bf16.
#[ignore]
#[test]
fn bonsai_golden_tokens_none() {
    let Some(model_path) = common::model_for(&MODEL, "bonsai_golden_tokens_none") else {
        return;
    };
    common::run_golden_test("bonsai_8b_none", KvQuant::None, &model_path);
}

/// thinking_budget forced injection on the Exact-hit decode path (Qwen3ForCausalLM).
///
/// Run the same prompt twice so the second call hits the prompt cache Exact path.
/// On the second call a step_fn fires `Some(FORCED_ID)` after seeing N_THINKING tokens
/// from the decode loop. The token at position N_THINKING+1 in the output sequence
/// must equal FORCED_ID, proving the Exact-hit loop honours the forced injection.
///
/// `FORCED_ID = 151648` is the `</think>` token in the Qwen3 vocabulary.
/// `N_THINKING = 4` is small enough to land comfortably within `N_TOKENS` decode steps.
///
/// Run:
/// cargo test -p rmlx-models --test bonsai_golden_tokens \
/// thinking_budget_exact_hit_qwen3 -- --ignored --nocapture
#[ignore]
#[test]
fn thinking_budget_exact_hit_qwen3() {
    use rmlx_mlx::Device;
    use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

    let Some(model_path) = common::model_for(&MODEL, "thinking_budget_exact_hit_qwen3") else {
        return;
    };

    const FORCED_ID: u32 = 151648; // </think> in Qwen3 vocabulary
    const N_THINKING: usize = 4; // fire forced injection after this many decode steps
    const N_TOKENS: usize = 16; // total decode budget; must be > N_THINKING + 1

    let device = Device::Gpu;
    let model =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load model");

    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer");

    let prompt_ids: Vec<u32> = tokenizer
        .encode(common::GOLDEN_PROMPT, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();

    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = PenaltyConfig::default();

    // --- First call: warm the prompt cache (Miss path) ---
    {
        let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        model
            .generate_greedy(
                &tokenizer,
                &prompt_ids,
                N_TOKENS,
                device,
                Some(KvQuant::K8V8),
                None,
                1, // single-slot cache
                &[],
                &mut |_| None,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut token_history,
            )
            .expect("warm call");
    }

    // --- Second call: Exact-hit path with thinking budget ---
    let mut decode_step_count = 0usize;
    let mut forced_at: Option<usize> = None;
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let mut token_history: Vec<u32> = Vec::new();

    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            N_TOKENS,
            device,
            Some(KvQuant::K8V8),
            None,
            1,
            &[],
            &mut |_step| {
                decode_step_count += 1;
                // Fire forced injection after the N_THINKING-th decode token
                // (step 0 is the prefill / cached first token; decode steps start at 1).
                if decode_step_count == N_THINKING + 1 && forced_at.is_none() {
                    forced_at = Some(decode_step_count);
                    Some(FORCED_ID)
                } else {
                    None
                }
            },
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("exact-hit call with budget");

    // The forced token must appear at position N_THINKING + 1 in the output steps
    // (0-indexed: position 0 is the cached first token, positions 1..N_THINKING are
    // normal decode tokens, position N_THINKING+1 is the injected </think>).
    assert!(
        forced_at.is_some(),
        "step_fn never fired — N_THINKING={N_THINKING} but only {} steps emitted",
        steps.len()
    );
    let forced_pos = N_THINKING + 1;
    assert!(
        steps.len() > forced_pos,
        "not enough steps emitted: need >{forced_pos}, got {}",
        steps.len()
    );
    assert_eq!(
        steps[forced_pos].token_id, FORCED_ID,
        "exact-hit path did not inject forced token at step {forced_pos}; \
         got {} (expected {FORCED_ID})",
        steps[forced_pos].token_id
    );
}

/// logprobs invariant: forced token must carry `logprobs: None` even when
/// `top_logprobs_k > 0`.
///
/// Exercises the Exact-hit decode path (second call after cache warm) with
/// `top_logprobs_k = 3`. Non-forced steps accumulate real logprobs; the injected
/// `</think>` step is an external override and must not carry logprobs from the
/// concurrently sampled (and discarded) `next_y` candidate.
///
/// Run:
/// cargo test -p rmlx-models --test bonsai_golden_tokens \
/// thinking_budget_forced_token_no_logprobs -- --ignored --nocapture
#[ignore]
#[test]
fn thinking_budget_forced_token_no_logprobs() {
    use rmlx_mlx::Device;
    use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

    let Some(model_path) = common::model_for(&MODEL, "thinking_budget_forced_token_no_logprobs")
    else {
        return;
    };

    const FORCED_ID: u32 = 151648; // </think> in Qwen3 vocabulary
    const N_THINKING: usize = 4;
    const N_TOKENS: usize = 16;
    const LP_K: u32 = 3; // request top-3 logprobs

    let device = Device::Gpu;
    let model =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load model");

    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer");

    let prompt_ids: Vec<u32> = tokenizer
        .encode(common::GOLDEN_PROMPT, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();

    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: LP_K,
    };
    let penalty_cfg = PenaltyConfig::default();

    // --- First call: warm the prompt cache (Miss path) ---
    {
        let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        model
            .generate_greedy(
                &tokenizer,
                &prompt_ids,
                N_TOKENS,
                device,
                Some(KvQuant::K8V8),
                None,
                1,
                &[],
                &mut |_| None,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut token_history,
            )
            .expect("warm call");
    }

    // --- Second call: Exact-hit path with logprobs enabled ---
    let mut decode_step_count = 0usize;
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let mut token_history: Vec<u32> = Vec::new();

    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            N_TOKENS,
            device,
            Some(KvQuant::K8V8),
            None,
            1,
            &[],
            &mut |_step| {
                decode_step_count += 1;
                if decode_step_count == N_THINKING + 1 {
                    Some(FORCED_ID)
                } else {
                    None
                }
            },
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("exact-hit call with logprobs + budget");

    let forced_pos = N_THINKING + 1;
    assert!(
        steps.len() > forced_pos,
        "not enough steps emitted: need >{forced_pos}, got {}",
        steps.len()
    );
    assert_eq!(
        steps[forced_pos].token_id, FORCED_ID,
        "forced token not at expected position {forced_pos}",
    );
    assert!(
        steps[forced_pos].logprobs.is_none(),
        "forced token at step {forced_pos} must have logprobs=None \
         (is an external override, not a sampled token); got Some(_)",
    );
    // Verify non-forced steps do carry logprobs when lp_k > 0.
    // Step 0 (the cached prefill token on the exact-hit path) now also
    // carries its stored prefill-token logprobs — every non-forced step in
    // [0, forced_pos) must be Some.
    for (i, step) in steps.iter().enumerate().take(forced_pos) {
        assert!(
            step.logprobs.is_some(),
            "step {i} (non-forced, lp_k={LP_K}) should have logprobs but got None",
        );
    }
}

/// Per-token logprob count must be identical on the cache-MISS and the
/// exact-HIT path, and the first emitted token must carry the SAME logprob on
/// both paths.
///
/// The exact-hit path replays the cached first token with the stored
/// prefill-token logprobs alongside `first_id`, truncated to the request's
/// `top_logprobs_k`, so the streams are length-equal and the first
/// token's `token_logprob` is byte-equal across paths. Without this, the
/// hit path would return N-1 logprob entries where the miss returns N —
/// an OpenAI contract violation.
///
/// No forced injection here — plain greedy decode both times so EVERY step
/// (step 0 included) is a real, logprob-bearing token.
///
/// Run:
/// cargo test -p rmlx-models --test bonsai_golden_tokens \
/// cache_hit_first_token_logprob_parity -- --ignored --nocapture
#[ignore]
#[test]
fn cache_hit_first_token_logprob_parity() {
    use rmlx_mlx::Device;
    use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

    let Some(model_path) = common::model_for(&MODEL, "cache_hit_first_token_logprob_parity") else {
        return;
    };

    const N_TOKENS: usize = 16;
    const LP_K: u32 = 5;

    let device = Device::Gpu;
    let model =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load model");

    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer");

    let prompt_ids: Vec<u32> = tokenizer
        .encode(common::GOLDEN_PROMPT, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();

    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: LP_K,
    };
    let penalty_cfg = PenaltyConfig::default();

    // single-slot cache; first call = Miss (warm + store), second = exact HIT.
    let do_run = || {
        let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        model
            .generate_greedy(
                &tokenizer,
                &prompt_ids,
                N_TOKENS,
                device,
                Some(KvQuant::K8V8),
                None,
                1,
                &[],
                &mut |_| None,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut token_history,
            )
            .expect("generate")
    };

    let miss_steps = do_run();
    let hit_steps = do_run();

    // Same prompt at temp=0 ⇒ identical token-id sequences (generation must be
    // byte-identical between paths — the logprob fix is logprobs-only).
    let miss_ids: Vec<u32> = miss_steps.iter().map(|s| s.token_id).collect();
    let hit_ids: Vec<u32> = hit_steps.iter().map(|s| s.token_id).collect();
    assert_eq!(
        miss_ids, hit_ids,
        "token-id sequences diverged between miss and hit — generation must be identical",
    );

    // The contract fix: equal count of logprob-bearing steps on both paths.
    let miss_lp_count = miss_steps.iter().filter(|s| s.logprobs.is_some()).count();
    let hit_lp_count = hit_steps.iter().filter(|s| s.logprobs.is_some()).count();
    assert_eq!(
        miss_lp_count,
        miss_steps.len(),
        "miss path: every step must carry logprobs at lp_k={LP_K}",
    );
    assert_eq!(
        hit_lp_count, miss_lp_count,
        "hit path logprob count ({hit_lp_count}) != miss path ({miss_lp_count}) \
         — the cached first token's logprob was dropped",
    );

    // The first emitted token's logprob must be present AND equal on both paths.
    let miss_first = miss_steps[0]
        .logprobs
        .as_ref()
        .expect("miss: first-token logprobs present");
    let hit_first = hit_steps[0]
        .logprobs
        .as_ref()
        .expect("hit first-token logprobs must be present");
    assert_eq!(
        hit_first.token_id, miss_first.token_id,
        "first-token id mismatch between miss and hit logprob records",
    );
    assert_eq!(
        hit_first.token_logprob, miss_first.token_logprob,
        "hit first-token logprob ({}) != miss first-token logprob ({}) \
         — replayed value is not the true miss-path logprob",
        hit_first.token_logprob, miss_first.token_logprob,
    );
    // top-k alternatives must match too (same width, same values, descending).
    assert_eq!(
        hit_first.top, miss_first.top,
        "hit first-token top-{LP_K} logprobs differ from miss path",
    );
}
