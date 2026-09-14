//! How far a conditioning projection carried across rounds sits from the same
//! rows projected in one call, on a real checkpoint at its own dtype.
//!
//! Both DFlash loops project each round's committed rows as the round commits
//! them and carry the result, rather than re-projecting the whole accumulated
//! buffer every round. `fc` is a bias-free linear and `hidden_norm` an RMSNorm,
//! so the two are the same rows through the same row-wise arithmetic and agree
//! exactly in exact arithmetic. They are not required to agree bit for bit: a
//! matmul's kernel and reduction order are chosen by shape, and the carried form
//! projects a handful of rows per call where the re-projecting form projected
//! the whole history.
//!
//! What the host-side fixtures bound is that gap in `f32` at fixture
//! magnitudes. This measures it where it matters — the shipped weights, the
//! shipped dtype, a real generation's rows, and heights spanning a round's
//! commit against the whole of it.
//!
//! It reports rather than asserts a bound. The only thing it fails on is a gap
//! that is not a rounding difference at all: the assertion is against the row's
//! own scale, so a projection that lost or reordered rows is caught and a
//! last-place difference is not.
//!
//! Server-free, one snapshot pair each. Both cases are selected by the same two
//! variables and neither runs without them; the DFlash 2 case resolves its own
//! pair by slug from `RMLX_O_MODELS_ROOT`, for the reason [`slug_path`] gives.
//! Run:
//! RMLX_KV_TEST_MODEL=<path-to>/mlx-community__Qwen3.6-35B-A3B-8bit \
//! RMLX_DRAFT_TEST_MODEL=<path-to>/z-lab__Qwen3.6-35B-A3B-DFlash \
//! cargo test -p rmlx-models --test spec_conditioning_residual -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "integration test: a snapshot that will not load or an array that will not \
              read back is the assertion failing, and the numbers it takes are its output"
)]

use std::path::PathBuf;

use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::{concatenate, Array, Device};
use rmlx_models::arch;
use rmlx_models::speculative::dflash::{dflash_generate, DFlashDrafter};
use rmlx_models::speculative::dflash2::{dflash2_generate, DFlash2Drafter};

/// Tokens the generation runs for before its rows are measured.
///
/// The height the whole-history projection then runs at, which is the number the
/// re-projecting form used and the carried form never does.
const GENERATE: usize = 256;

/// Commit sizes the chunked projection cycles through.
///
/// A DFlash 1 round commits its carry token and the proposals the verifier kept,
/// so one to a block's worth. These are the heights the carried form projects
/// at; every one of them is compared against the same rows inside one call of
/// all of them.
const COMMITS: [usize; 5] = [1, 2, 3, 4, 5];

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
}

/// A snapshot resolved by slug under `RMLX_O_MODELS_ROOT`.
///
/// The variable selects and the slug resolves, which is what the equivalence
/// pairs do and for the same two reasons. One pair of variables cannot name two
/// pairs, so the DFlash 2 case cannot take whatever `RMLX_DRAFT_TEST_MODEL`
/// holds — a DFlash 1 drafter loads against a verifier of the same width without
/// complaint. And a case that resolves itself entirely by slug runs whenever the
/// snapshots happen to be on the machine, including under `make gpu-test`, where
/// this pair's verifier drives an MLX quantized matmul whose invalid loads the
/// shader-validation census would then have to pin — a count that moves with
/// every generation. So this runs when an operator asks and not otherwise.
fn slug_path(slug: &str) -> Option<PathBuf> {
    std::env::var("RMLX_O_MODELS_ROOT")
        .ok()
        .map(|root| PathBuf::from(root).join(slug))
        .filter(|p| p.is_dir())
}

/// `[1, rows, width]` split into consecutive groups whose sizes cycle through
/// `COMMITS`, each projected on its own and the results concatenated — the
/// buffer the round loop carries.
fn carried_projection(drafter: &DFlashDrafter, raw: &Array, device: Device) -> Array {
    let rows = raw.shape()[1];
    let width = raw.shape()[2];
    let mut pieces: Vec<Array> = Vec::new();
    let mut at = 0;
    for size in COMMITS.iter().copied().cycle() {
        if at >= rows {
            break;
        }
        let take = (size as i32).min(rows - at);
        let chunk = raw
            .slice(&[0, at, 0], &[1, at + take, width], &[1, 1, 1], device)
            .expect("slice a commit");
        pieces.push(drafter.project_condition(&chunk).expect("project a commit"));
        at += take;
    }
    let refs: Vec<&Array> = pieces.iter().collect();
    concatenate(&refs, 1, device).expect("join the carried pieces")
}

/// Per-row `(max |a - b|, that gap over the row's RMS)`, and how many rows carry
/// a gap wider than one unit in the last place of `bf16` at their own scale.
///
/// `bf16` keeps 8 significand bits, so a value's neighbours are `2^-8` apart
/// relative to its own magnitude; a row whose worst element sits inside that of
/// its RMS differs by rounding and nothing else.
fn row_gaps(a: &Array, b: &Array, hidden: usize) -> (f32, f32, usize) {
    const BF16_ULP: f32 = 1.0 / 256.0;
    let (x, y) = (to_f32(a), to_f32(b));
    assert_eq!(
        x.len(),
        y.len(),
        "compared buffers must have the same shape"
    );
    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut rows_over = 0usize;
    for (p, q) in x.chunks_exact(hidden).zip(y.chunks_exact(hidden)) {
        let gap = p
            .iter()
            .zip(q.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0f32, f32::max);
        let rms = (p.iter().map(|u| u * u).sum::<f32>() / hidden as f32).sqrt();
        let rel = if rms > 0.0 { gap / rms } else { 0.0 };
        worst_abs = worst_abs.max(gap);
        worst_rel = worst_rel.max(rel);
        if rel > BF16_ULP {
            rows_over += 1;
        }
    }
    (worst_abs, worst_rel, rows_over)
}

fn to_f32(a: &Array) -> Vec<f32> {
    let f32_view = a
        .astype(rmlx_mlx::Dtype::F32, Device::Cpu)
        .expect("cast for readback");
    f32_view
        .to_bytes()
        .expect("read array bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes is an f32")))
        .collect()
}

/// The carried projection against one call at the height a whole generation
/// reaches, on the shipped DFlash 1 pair.
///
/// The rows are a real generation's: the pair drafts and verifies
/// [`GENERATE`] tokens, and the positions the rounds committed are exactly the
/// prompt plus what was emitted, so one capture forward over those ids is the
/// conditioning history the loop accumulated.
#[ignore = "needs a Qwen3.6-MoE verifier + DFlash drafter snapshot and the Metal context"]
#[test]
fn dflash1_carried_projection_against_one_call_at_full_height() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!(
            "SKIP dflash1_carried_projection_against_one_call_at_full_height: \
             RMLX_KV_TEST_MODEL and RMLX_DRAFT_TEST_MODEL must both name an existing \
             snapshot directory"
        );
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();
    let drafter = DFlashDrafter::load(&draft_path, hidden, device).expect("load drafter");
    let target_layer_ids = drafter.target_layer_ids().to_vec();

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt = "<|im_start|>user\nExplain how a B-tree keeps its height balanced \
                  during insertion.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let prompt_ids: Vec<u32> = tk.encode(prompt, true).expect("encode").get_ids().to_vec();
    let eos: Vec<u32> = tk.token_to_id("<|im_end|>").into_iter().collect();

    let mut emitted_ids: Vec<u32> = Vec::new();
    let mut step_fn = |s: &rmlx_models::ProbeStep| {
        emitted_ids.push(s.token_id);
        None
    };
    dflash_generate(
        &verifier,
        &drafter,
        &tk,
        &prompt_ids,
        GENERATE,
        16,
        None,
        None,
        &eos,
        &mut step_fn,
        &rmlx_models::sampler::SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: Some(0),
            top_logprobs_k: 0,
        },
        device,
    )
    .expect("dflash generate");
    assert!(
        emitted_ids.len() >= 128,
        "the pair emitted {} tokens, too few to measure a tall projection against",
        emitted_ids.len()
    );

    // The positions the rounds conditioned on: the prompt and everything the
    // verifier committed. One capture forward over them is that history.
    let mut ids = prompt_ids.clone();
    ids.extend_from_slice(&emitted_ids);
    let mut kv: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| KvCache::with_quant(KvQuant::None).with_layer_idx(i))
        .collect();
    let mut lin: Vec<LinearAttnCache> = (0..verifier.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect();
    let (_logits, raw) = verifier
        .forward_verify_capture(
            &ids,
            ids.len(),
            &target_layer_ids,
            &mut kv,
            Some(&mut lin),
            device,
        )
        .expect("capture the committed history");

    let whole = drafter
        .project_condition(&raw)
        .expect("project the history in one call");
    let carried = carried_projection(&drafter, &raw, device);
    assert_eq!(
        carried.shape(),
        whole.shape(),
        "the carried pieces do not tile the history"
    );

    let (abs, rel, rows_over) = row_gaps(&carried, &whole, hidden);
    let rows = whole.shape()[1];
    println!(
        "[dflash1 residual] rows={rows} dtype={:?} max_abs={abs:e} \
         max_rel_to_row_rms={rel:e} rows_over_1_bf16_ulp={rows_over}",
        whole.dtype()
    );
    assert!(
        rel < 0.05,
        "the carried projection differs from one call by {rel:e} of a row's own scale \
         over {rows} rows — that is not a rounding difference, and the two forms are \
         not computing the same rows"
    );
}

/// The same measurement on the shipped DFlash 2 pair, whose loop bounds its
/// buffer to the drafter's declared window.
///
/// The generation here is shorter than that window, so the trim is inert and
/// what is compared is the same thing as above: the rows a round projects
/// against all of them in one call.
#[ignore = "needs a Qwen3.8-27B verifier + DFlash2 drafter snapshot and the Metal context"]
#[test]
fn dflash2_carried_projection_against_one_call_at_full_height() {
    if env_path("RMLX_KV_TEST_MODEL").is_none() || env_path("RMLX_DRAFT_TEST_MODEL").is_none() {
        eprintln!(
            "SKIP dflash2_carried_projection_against_one_call_at_full_height: \
             RMLX_KV_TEST_MODEL and RMLX_DRAFT_TEST_MODEL must both name an existing \
             snapshot directory — they select this case, and the slugs below resolve it"
        );
        return;
    }
    let (Some(model_path), Some(draft_path)) = (
        slug_path("mlx-community__Qwen3.8-27B-4bit"),
        slug_path("z-lab__Qwen3.8-27B-DFlash2"),
    ) else {
        eprintln!(
            "SKIP dflash2_carried_projection_against_one_call_at_full_height: \
             RMLX_O_MODELS_ROOT must hold mlx-community__Qwen3.8-27B-4bit and \
             z-lab__Qwen3.8-27B-DFlash2"
        );
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();
    let drafter = DFlash2Drafter::load(&draft_path, hidden, device).expect("load drafter");
    let target_layer_ids = drafter.cfg.target_layer_ids.clone();

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt = "<|im_start|>user\nExplain how a B-tree keeps its height balanced \
                  during insertion.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let prompt_ids: Vec<u32> = tk.encode(prompt, true).expect("encode").get_ids().to_vec();
    let eos: Vec<u32> = tk.token_to_id("<|im_end|>").into_iter().collect();

    let mut emitted_ids: Vec<u32> = Vec::new();
    let mut step_fn = |s: &rmlx_models::ProbeStep| {
        emitted_ids.push(s.token_id);
        None
    };
    dflash2_generate(
        &verifier,
        &drafter,
        &tk,
        &prompt_ids,
        GENERATE,
        8,
        None,
        None,
        &eos,
        &mut step_fn,
        &rmlx_models::sampler::SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: Some(0),
            top_logprobs_k: 0,
        },
        device,
    )
    .expect("dflash2 generate");
    assert!(
        emitted_ids.len() >= 128,
        "the pair emitted {} tokens, too few to measure a tall projection against",
        emitted_ids.len()
    );

    let mut ids = prompt_ids.clone();
    ids.extend_from_slice(&emitted_ids);
    let mut kv: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| KvCache::with_quant(KvQuant::None).with_layer_idx(i))
        .collect();
    let mut lin: Vec<LinearAttnCache> = (0..verifier.num_hidden_layers())
        .map(|_| LinearAttnCache::new())
        .collect();
    let (_logits, raw) = verifier
        .forward_verify_capture(
            &ids,
            ids.len(),
            &target_layer_ids,
            &mut kv,
            Some(&mut lin),
            device,
        )
        .expect("capture the committed history");
    assert!(
        raw.shape()[1] <= drafter.conditioning_rows(),
        "the history reached the drafter's window, so this is measuring a trim as well \
         as a projection"
    );

    let whole = drafter
        .project_conditioning(&raw)
        .expect("project the history in one call");
    let carried = carried_projection2(&drafter, &raw, device);
    assert_eq!(
        carried.shape(),
        whole.shape(),
        "the carried pieces do not tile the history"
    );

    let (abs, rel, rows_over) = row_gaps(&carried, &whole, hidden);
    let rows = whole.shape()[1];
    println!(
        "[dflash2 residual] rows={rows} dtype={:?} max_abs={abs:e} \
         max_rel_to_row_rms={rel:e} rows_over_1_bf16_ulp={rows_over}",
        whole.dtype()
    );
    assert!(
        rel < 0.05,
        "the carried projection differs from one call by {rel:e} of a row\'s own scale \
         over {rows} rows — that is not a rounding difference, and the two forms are \
         not computing the same rows"
    );
}

/// [`carried_projection`] for the DFlash 2 drafter, whose projection is its own
/// method on its own type.
fn carried_projection2(drafter: &DFlash2Drafter, raw: &Array, device: Device) -> Array {
    let rows = raw.shape()[1];
    let width = raw.shape()[2];
    let mut pieces: Vec<Array> = Vec::new();
    let mut at = 0;
    for size in COMMITS.iter().copied().cycle() {
        if at >= rows {
            break;
        }
        let take = (size as i32).min(rows - at);
        let chunk = raw
            .slice(&[0, at, 0], &[1, at + take, width], &[1, 1, 1], device)
            .expect("slice a commit");
        pieces.push(
            drafter
                .project_conditioning(&chunk)
                .expect("project a commit"),
        );
        at += take;
    }
    let refs: Vec<&Array> = pieces.iter().collect();
    concatenate(&refs, 1, device).expect("join the carried pieces")
}
