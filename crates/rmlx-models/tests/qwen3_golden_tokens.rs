//! Per-arch golden-token decode gate for the Qwen3.5-MoE path.
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Covers the MoE + GatedDeltaNet
//! hybrid backbone (`Qwen3_5MoeForConditionalGeneration`).
//!
//! Model: `mlx-community__Qwen3.6-35B-A3B-8bit`.
//! KV quant: K8V8 (the resolver default for Qwen3.5 MoE; K8V4 on the FA layers
//! regressed decode).
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/Qwen3.6-35B-A3B-8bit \
//! cargo test -p rmlx-models --test qwen3_golden_tokens -- --ignored
//! Then gate:
//! RMLX_KV_TEST_MODEL=/path/to/Qwen3.6-35B-A3B-8bit \
//! cargo test -p rmlx-models --test qwen3_golden_tokens -- --ignored

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

/// Architectures this golden was recorded against. Any other arch is skipped.
const EXPECTED_ARCHS: &[&str] = &[
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
];

#[ignore]
#[test]
fn qwen3_moe_golden_tokens_k8v8() {
    let Some(model_path) = common::model_path_from_env() else {
        return;
    };
    if common::skip_if_arch_mismatch(&model_path, "qwen3_moe_golden_tokens_k8v8", EXPECTED_ARCHS) {
        return;
    }
    common::run_golden_test("qwen3_moe_35b_k8v8", KvQuant::K8V8);
}

/// thinking_budget forced injection on the Exact-hit decode path (Qwen3_5Moe).
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
/// RMLX_KV_TEST_MODEL=/path/to/Qwen3.6-35B-A3B-8bit \
/// cargo test -p rmlx-models --test qwen3_golden_tokens \
/// thinking_budget_exact_hit_qwen3_5_moe -- --ignored --nocapture
#[ignore]
#[test]
fn thinking_budget_exact_hit_qwen3_5_moe() {
    use rmlx_mlx::Device;
    use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

    let Some(model_path) = common::model_path_from_env() else {
        return;
    };
    if common::skip_if_arch_mismatch(
        &model_path,
        "thinking_budget_exact_hit_qwen3_5_moe",
        EXPECTED_ARCHS,
    ) {
        return;
    }

    const FORCED_ID: u32 = 151648; // </think> in Qwen3 vocabulary
    const N_THINKING: usize = 4; // fire forced injection after this many decode steps
    const N_TOKENS: usize = 16; // total decode budget; must be > N_THINKING + 1

    let device = Device::Gpu;
    let model = arch::load_model(&model_path, device).expect("load model");

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
                // Fire forced injection after the N_THINKING-th decode token.
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
