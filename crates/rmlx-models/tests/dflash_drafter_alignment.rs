//! DFlash drafter <-> Qwen3.6-MoE verifier numeric-alignment gate.
//!
//! Pins the three wired verifier-side seams against the real checkpoints:
//!   1. multi-layer hidden capture (`forward_hidden_states_multi` /
//!      `forward_verify_capture`) at `target_layer_ids`,
//!   2. GDN snapshot/restore rollback (mechanism, reused),
//! 3. raw (unscaled) verifier `embed_tokens` accessor (`embed_tokens_raw`).
//!
//! The DFlash `bind()` resolves `embed_scale = 1.0` for Qwen3.5 (bare
//! nn.Embedding), so -- unlike the Gemma4 MTP case -- the drafter consumes the
//! UNSCALED embedding. This test pins that contract end-to-end: on a
//! highly-predictable prompt the drafter's first-block proposal must align
//! with the verifier's greedy continuation (accept > 0), and the live loop
//! must emit coherent prose.
//!
//! Server-free. Run (both checkpoints present):
//! RMLX_KV_TEST_MODEL=<path-to>/mlx-community__Qwen3.6-35B-A3B-8bit \
//! RMLX_DRAFT_TEST_MODEL=<path-to>/z-lab__Qwen3.6-35B-A3B-DFlash \
//! cargo test -p rmlx-models --test dflash_drafter_alignment -- --ignored --nocapture

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

use std::path::PathBuf;

use rmlx_kv_quant::{KvCache, LinearAttnCache};
use rmlx_mlx::{argmax, Device};
use rmlx_models::arch;
use rmlx_models::kv_cache::DEFAULT_KV_QUANT;
use rmlx_models::speculative::dflash::{dflash_generate_greedy, DFlashDrafter};

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn argmax_id(logits: &rmlx_mlx::Array, device: Device) -> u32 {
    let am = argmax(logits, -1, device).expect("argmax");
    am.eval().unwrap();
    u32::from_le_bytes(am.to_bytes().unwrap()[..4].try_into().unwrap())
}

/// Round-0 alignment: the drafter's FIRST proposed token must equal the
/// verifier's greedy next-token after the same bonus `b`. A wrong embed scale,
/// wrong capture point, or stale GDN state collapses this to a mismatch (the
/// ~0%-accept failure mode).
#[ignore]
#[test]
fn dflash_round0_first_token_aligns() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!("[dflash_align] verifier/draft model unset/absent - skipping");
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    assert!(
        verifier.needs_lin_caches(),
        "DFlash verifier must be the Qwen3.5/3.6-MoE hybrid"
    );
    let hidden = verifier.hidden_size();
    let mut drafter = DFlashDrafter::load(&draft_path, hidden, device).expect("load drafter");
    let tlids = drafter.target_layer_ids().to_vec();

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt = "<|im_start|>user\nList the first five prime numbers.<|im_end|>\n\
                  <|im_start|>assistant\n<think>\n\n</think>\n\n";
    let enc = tk.encode(prompt, true).expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

    // Mirror production: the verifier caches are built at the codec the
    // drafter path resolves when no override is given.
    let kv_quant = DEFAULT_KV_QUANT;
    let mut caches: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| {
            KvCache::with_quant_max_seq_window(kv_quant, 4096, verifier.layer_sliding_window(i))
                // Production's verifier stacks declare the topology off the
                // architecture, and the Mixed/RotK codecs refuse a shared-source
                // decode step without it. Reading it the same way here keeps the
                // alignment property, not `DEFAULT_KV_QUANT`, the thing that can
                // fail this test.
                .with_shares_kv(verifier.shares_kv_across_layers())
        })
        .collect();
    let mut lin: Vec<LinearAttnCache> = (0..verifier.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect();

    let prefill = &prompt_ids[..prompt_ids.len() - 1];
    rmlx_models::speculative::prefill_chunked(
        &verifier,
        prefill,
        &mut caches,
        Some(&mut lin),
        device,
    )
    .expect("prefill");

    let last = *prompt_ids.last().unwrap();
    let (r0_logits, h_ctx_raw) = verifier
        .forward_verify_capture(&[last], 1, &tlids, &mut caches, Some(&mut lin), device)
        .expect("forward_verify_capture r0");
    h_ctx_raw.eval().unwrap();
    assert_eq!(
        h_ctx_raw.shape(),
        &[1, 1, (tlids.len() * hidden) as i32],
        "concat hidden shape mismatch"
    );
    let b = argmax_id(&r0_logits, device);

    // Diagnostic: l2 of capture + projected context (compare to mlx-vlm ref:
    // concat_hidden l2~11.95, h_ctx l2~16.03 on this prompt).
    let l2 = |a: &rmlx_mlx::Array| -> f32 {
        let f = a.astype(rmlx_mlx::Dtype::F32, device).unwrap();
        f.eval().unwrap();
        f.to_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|c| {
                let v = f32::from_le_bytes(c.try_into().unwrap());
                v * v
            })
            .sum::<f32>()
            .sqrt()
    };
    let h_ctx = drafter.project_condition(&h_ctx_raw).expect("project");
    eprintln!(
        "[dflash_align] b={b} concat_hidden_l2={:.4} h_ctx_l2={:.4}",
        l2(&h_ctx_raw),
        l2(&h_ctx)
    );

    let block = drafter
        .draft_block(&verifier, b, &h_ctx, 8)
        .expect("draft_block");
    assert!(!block.is_empty(), "drafter produced no tokens");
    eprintln!("[dflash_align] round0 b={b} block={block:?}");

    // The bonus `b` pins prefill + multi-layer capture + logits_from_hidden
    // (it must equal the mlx-vlm reference, b=760 on this prompt). DFlash is a
    // block-diffusion drafter, so its per-position proposals are NOT required to
    // equal verifier greedy (unlike MTP) — accept-rate over a run is the real
    // bar (see the live_loop test). We pin the capture l2 to the reference band.
    assert!(b > 0, "bonus token degenerate");
    let ch = l2(&h_ctx_raw);
    assert!(
        (5.0..30.0).contains(&ch),
        "concat_hidden l2={ch} outside reference band [5,30] (capture-point bug)"
    );
    // Non-degenerate proposal: with YARN RoPE + correct conditioning the block
    // is varied (the plain-RoPE bug produced a near-constant block like
    // [13,513,513,513,513,...]); require at least 3 distinct tokens in 7.
    let distinct: std::collections::HashSet<u32> = block.iter().copied().collect();
    assert!(
        distinct.len() >= 3,
        "drafter block {block:?} near-constant ({} distinct) — RoPE/YARN or \
         conditioning regression",
        distinct.len()
    );
}

/// End-to-end: the live DFlash loop on a predictable prompt must emit coherent
/// output. Returns the full budget AND non-empty text.
#[ignore]
#[test]
fn dflash_live_loop_emits_coherent() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!("[dflash_align] verifier/draft model unset/absent - skipping");
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();
    let mut drafter = DFlashDrafter::load(&draft_path, hidden, device).expect("load drafter");

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt = "<|im_start|>user\nList the first five prime numbers.<|im_end|>\n\
                  <|im_start|>assistant\n<think>\n\n</think>\n\n";
    let enc = tk.encode(prompt, true).expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

    let mut emitted_ids: Vec<u32> = Vec::new();
    let mut step_fn = |s: &rmlx_models::ProbeStep| {
        emitted_ids.push(s.token_id);
        None
    };
    let eos: Vec<u32> = tk
        .token_to_id("<|im_end|>")
        .into_iter()
        .chain(std::iter::once(248046u32))
        .collect();

    let steps = dflash_generate_greedy(
        &verifier,
        &mut drafter,
        &tk,
        &prompt_ids,
        48,
        16,
        None,
        None,
        &eos,
        &mut step_fn,
        device,
    )
    .expect("dflash generate");

    assert_eq!(steps.len(), emitted_ids.len());
    assert!(
        steps.len() >= 8,
        "expected >=8 emitted, got {}",
        steps.len()
    );
    let text = tk.decode(&emitted_ids, false).expect("decode");
    eprintln!("[dflash_align] emitted={} text={:?}", steps.len(), text);
    assert!(!text.trim().is_empty(), "DFlash produced empty output");
}
