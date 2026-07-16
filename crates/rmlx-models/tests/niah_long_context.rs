//! TurboFlash VG.2 NIAH (Needle-In-A-Haystack) validation harness.
//!
//! Server-free long-context retrieval test: builds a synthetic "haystack" of
//! filler text with a unique alphanumeric "needle" embedded at a known depth,
//! tokenises it with the model's tokenizer, asks the model to recover the
//! needle via greedy temp=0 decode, and asserts the needle string appears in
//! the decoded output.
//!
//! # Models
//!
//! Each model is gated by its `RMLX_TEST_MODEL_*` env var (see
//! `docs/TESTING.md`). When the env var is unset, that model's tests skip
//! silently — the suite is green on machines without snapshots.
//!
//! Covered models (one `#[test]` per (model, ctx_len, depth) cell):
//!
//! - `RMLX_TEST_MODEL_GEMMA4_E4B`  → `Gemma4ForConditionalGeneration`
//! - `RMLX_TEST_MODEL_QWEN36`      → `Qwen3_5MoeForConditionalGeneration`
//! - `RMLX_TEST_MODEL_BONSAI`      → `Qwen3ForCausalLM`
//!
//! # Modes
//!
//! The test consults `RMLX_TURBO_FLASH` to decide whether TurboFlash is on or
//! off — it does NOT set the env var itself (the kernel uses an `OnceLock`
//! and the value is latched on first read, so flipping mid-process is not
//! supported). The shell driver
//! `scripts/release_e2e/stage6_perf/niah_long_context.sh` runs the suite
//! twice, once with `RMLX_TURBO_FLASH=0` and once with `RMLX_TURBO_FLASH=1`,
//! each in a fresh process.
//!
//! A parallel **planar_flash_decode** family of cells (`niah_pflash_*`) consult
//! `RMLX_PLANAR_FLASH_DECODE` and force `KvQuant::PlanarK`. Bonsai
//! (`Qwen3ForCausalLM`) reaches `update_and_sdpa_planar_k_fused` and dispatches;
//! Qwen3.6 MoE rejects `PlanarK` outright at `validate_resolved`
//! (`QwenMoePlanarKRejected`); Gemma4 reaches the same fused arm via
//! `update_and_sdpa_shared_source` but stays dormant behind the warm-TTFT
//! bf16-K-seed gate. See [`FlashRouting`] — dormancy is always a gate, never a
//! property of cross-layer KV sharing.
//!
//! # KV mode
//!
//! For the TurboFlash cells (`niah_<model>_*`): forced to `K8V4` so the
//! TurboFlash MSL kernel actually dispatches when enabled (it gates on
//! `KvQuant::K8V4` plus `kv_seq > 4096`).
//!
//! For the planar_flash_decode cells (`niah_pflash_<model>_*`): forced to
//! `KvQuant::PlanarK` so the `planar_fused_qk` / `planar_flash_decode`
//! K-side path activates. With `RMLX_PLANAR_FLASH_DECODE=1` the per-decode
//! call uses the single-pass planar_flash_decode kernel; with the env unset it
//! falls through to the fused QK → softmax → SV chain.
//!
//! # Why integration test (`tests/`), not unit (`src/*_tests.rs`)
//!
//! `generate_greedy` requires `arch::load_model` which touches Metal +
//! `safetensors` mmap — only reachable from an integration test binary with a
//! real snapshot. Mirrors the pattern in `gemma4_golden_tokens.rs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value,
    clippy::format_push_string,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod common;

use std::path::{Path, PathBuf};

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

// ── Per-model env-var resolution ──────────────────────────────────────────

/// Architectures recognised by this harness. Other archs trigger a skip
/// (see `skip_if_arch_mismatch`).
const EXPECTED_ARCHS_GEMMA4: &[&str] = &["Gemma4ForConditionalGeneration"];
const EXPECTED_ARCHS_QWEN36: &[&str] = &["Qwen3_5MoeForConditionalGeneration"];
const EXPECTED_ARCHS_BONSAI: &[&str] = &["Qwen3ForCausalLM"];

/// Resolve a model path from an env var. Returns `None` and prints a skip
/// note when the var is unset or the path does not exist.
fn model_path(var: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(var) {
        let pb = PathBuf::from(&p);
        if pb.exists() {
            Some(pb)
        } else {
            eprintln!("[niah] SKIP: {var}={p} — path does not exist");
            None
        }
    } else {
        eprintln!("[niah] SKIP: {var} not set");
        None
    }
}

// ── Needle + haystack construction ────────────────────────────────────────

/// Fixed needle string. Chosen to be unambiguous (mixed case + digits) so
/// substring search has no false positives in the filler text.
const NEEDLE: &str = "AX7-PURPLE-FOX-9421";

/// Single filler sentence — neutral, contentful enough that the model does
/// not collapse into a token-loop, ASCII-only so token-counts are roughly
/// stable across tokenizers.
const FILLER_SENTENCE: &str = "The grass is green and the sun is yellow. \
    Mountains rise tall above the silent valley below. \
    Rivers flow steadily toward the open sea. ";

/// Prompt template assembled around the haystack. We use a plain
/// instruction-style prompt (NOT a chat template) so the harness is
/// arch-agnostic — every base model recognises the pattern.
fn build_prompt(haystack: &str) -> String {
    format!(
        "You are given a long document. Read it carefully and find the \
         secret code mentioned somewhere inside. \
         The secret code is a unique alphanumeric token. \
         When you have found it, repeat it exactly.\n\n\
         Document:\n{haystack}\n\n\
         Question: What is the secret code from the document above?\n\
         Answer: The secret code is "
    )
}

/// Tokenise + measure length of one filler sentence (cheap; one call).
fn filler_token_len(tk: &tokenizers::Tokenizer) -> usize {
    tk.encode(FILLER_SENTENCE, false)
        .expect("tokenize filler")
        .get_ids()
        .len()
}

/// Construct a haystack of approximately `target_ctx_tokens` tokens with the
/// needle embedded at `depth_frac` (0.0 = beginning, 1.0 = end). Returns the
/// final tokenised prompt id sequence (BOS prepended when resolvable).
fn build_haystack_prompt(
    tk: &tokenizers::Tokenizer,
    target_ctx_tokens: usize,
    depth_frac: f32,
    bos: Option<u32>,
) -> Vec<u32> {
    let per_sentence = filler_token_len(tk).max(1);
    // Reserve ~300 tokens for the instruction wrapper + needle sentence +
    // tail prompt — generous; the prompt template above is shorter, but
    // overshooting wastes a few tokens, never destroys correctness.
    let reserve = 300usize;
    let filler_budget = target_ctx_tokens.saturating_sub(reserve);
    let total_sentences = filler_budget / per_sentence;
    let needle_idx = ((total_sentences as f32) * depth_frac.clamp(0.0, 1.0)) as usize;
    let needle_sentence =
        format!("Important note: the secret code is {NEEDLE}. Remember this code. ");

    let mut haystack = String::with_capacity(filler_budget * 8);
    for i in 0..total_sentences {
        if i == needle_idx {
            haystack.push_str(&needle_sentence);
        }
        haystack.push_str(FILLER_SENTENCE);
    }
    // Edge case: needle_idx == total_sentences → push at the very end.
    if needle_idx >= total_sentences {
        haystack.push_str(&needle_sentence);
    }

    let prompt = build_prompt(&haystack);
    let enc = tk.encode(prompt, false).expect("tokenize prompt");
    let body = enc.get_ids();
    let mut ids = Vec::with_capacity(body.len() + 1);
    if let Some(b) = bos {
        ids.push(b);
    }
    ids.extend_from_slice(body);
    ids
}

/// BOS resolution mirrors `common::run_golden_test`. Best-effort: returns
/// `None` if no BOS token is configured (the tokenizer will fall back on its
/// own `add_special_tokens` if we encode with `true`, but we encode with
/// `false` here to avoid the tokenizer doubling the BOS).
fn resolve_bos(model_dir: &Path, tk: &tokenizers::Tokenizer) -> Option<u32> {
    let cfg_path = model_dir.join("tokenizer_config.json");
    let v: serde_json::Value = std::fs::read(&cfg_path)
        .ok()
        .and_then(|d| serde_json::from_slice(&d).ok())
        .unwrap_or(serde_json::Value::Null);
    let extract = |key: &str| -> Option<String> {
        match v.get(key) {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Object(map)) => map
                .get("content")
                .and_then(|c| c.as_str())
                .map(str::to_owned),
            _ => None,
        }
    };
    let candidates: Vec<String> = [
        extract("bos_token"),
        Some("<bos>".to_owned()),
        Some("<|im_start|>".to_owned()),
        extract("eos_token"),
        Some("<|endoftext|>".to_owned()),
    ]
    .into_iter()
    .flatten()
    .collect();
    candidates.iter().find_map(|c| tk.token_to_id(c))
}

// ── Test runner ───────────────────────────────────────────────────────────

/// Number of decode tokens to emit. Short answers (the needle is 19 chars,
/// roughly 8–12 tokens) — keep this modest so 32k-ctx tests stay tractable
/// (each decode step is one full attention pass over the full KV cache).
const N_DECODE_TOKENS: usize = 64;

/// Verdict for one cell.
#[derive(Debug)]
struct NiahResult {
    needle_found: bool,
    decoded: String,
    prompt_len: usize,
}

/// Which flash family this cell exercises.
///
/// `Turbo` — TurboFlash MSL kernel, gated on `KvQuant::K8V4` plus
/// `kv_seq > 4096`. Driven by `RMLX_TURBO_FLASH`.
///
/// `Pflash` — planar_flash_decode kernel, gated on `KvStorage::PlanarK` plus
/// power-of-two `head_dim`. Driven by `RMLX_PLANAR_FLASH_DECODE`.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum FlashKind {
    Turbo,
    Pflash,
}

fn run_one(
    model_path: &Path,
    target_ctx_tokens: usize,
    depth_frac: f32,
    kind: FlashKind,
) -> NiahResult {
    let device = Device::Gpu;
    let model =
        arch::load_model(model_path, device, &arch::LoadOpts::default()).expect("arch::load_model");
    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer.json");
    let bos = resolve_bos(model_path, &tokenizer);

    let prompt_ids = build_haystack_prompt(&tokenizer, target_ctx_tokens, depth_frac, bos);
    let prompt_len = prompt_ids.len();

    // Force the KV codec that activates the kernel under test. Turbo cells
    // need K8V4 (TurboFlash MSL gate). Planar-flash cells need PlanarK so the
    // `update_and_sdpa_planar_k_fused` chain is reached; without that, the
    // planar_flash_decode kernel can never dispatch (Auto resolution returns
    // a non-PlanarK codec on every test arch).
    //
    // When the smoke matrix runner injects `RMLX_NIAH_KV_QUANT=<str>`, parse
    // it via `KvQuant::FromStr` and use that instead. A parse failure logs a
    // warning and falls back to the per-FlashKind default so an unmapped
    // codec never silently changes the gate. Bad values cannot reroute kernel
    // dispatch silently.
    let default_kv_quant = match kind {
        FlashKind::Turbo => KvQuant::K8V4,
        FlashKind::Pflash => KvQuant::PlanarK,
    };
    let kv_quant = match std::env::var("RMLX_NIAH_KV_QUANT") {
        Ok(raw) if !raw.is_empty() => match raw.parse::<KvQuant>() {
            Ok(parsed) => {
                eprintln!("[niah] RMLX_NIAH_KV_QUANT={raw} -> {parsed}");
                parsed
            }
            Err(err) => {
                eprintln!(
                    "[niah] WARN: RMLX_NIAH_KV_QUANT={raw} parse failed ({err}); \
                     falling back to default {default_kv_quant}"
                );
                default_kv_quant
            }
        },
        _ => default_kv_quant,
    };
    // max_ctx must comfortably exceed prompt_len + N_DECODE_TOKENS.
    let max_ctx = ((prompt_len + N_DECODE_TOKENS) as i32 + 1024).max(8192);

    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            N_DECODE_TOKENS,
            device,
            Some(kv_quant),
            Some(max_ctx),
            1,
            &[],
            &mut |_| None,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy");

    let token_ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    let decoded: String = tokenizer
        .decode(&token_ids, false)
        .unwrap_or_else(|_| steps.iter().map(|s| s.piece.as_ref()).collect());

    let needle_found = decoded.contains(NEEDLE);
    NiahResult {
        needle_found,
        decoded,
        prompt_len,
    }
}

/// Expected kernel-dispatch outcome for a cell, in ON mode.
///
/// `Reachable` — the MSL kernel MUST fire; delta > 0.
///
/// `Dormant` — the kernel must NOT fire; delta == 0, so the ON run measures the
/// legacy fallback and is equivalent to OFF.
///
/// **Dormancy is a gate/config fact, per cell — never a structural property of
/// cross-layer KV sharing.** An earlier version of this comment claimed the
/// shared-KV producer path "structurally never invokes" the kernel because it
/// must return accumulated bf16 K/V alongside the SDPA output. Both halves are
/// false: `update_and_sdpa_shared_source` reaches the TurboFlash, fused-QK,
/// planar and rotor arms exactly as `update_and_sdpa` does, and a producer that
/// runs a fused arm hands consumers `SharedKv::Store` rather than materialising
/// bf16 (see `docs/KV_QUANT.md`). The measured reason per `Dormant` cell:
///
/// * **Gemma4 + Turbo** — TurboFlash requires `head_dim ∈ {128, 256}`; Gemma4's
///   global layers are `head_dim=512`, so the kernel's own shape gate rejects
///   them (`update_and_sdpa_k8v4_flash_inner`). Nothing to do with KV sharing.
/// * **Gemma4 + Pflash** — the warm-TTFT bf16-K-seed gate keeps the PlanarK
///   kernels dormant while the seed is live (`sdpa.rs`, `warm_ttft_bypass`).
/// * **Qwen3.6 + Pflash** — the arch rejects `KvQuant::PlanarK` at
///   `validate_resolved`, so the codec never runs at all.
///
/// Each of those is a gate that can move. Re-check the affected cells whenever
/// the TurboFlash shape gate, the warm-TTFT gate, or a `validate_resolved`
/// rejection changes — a cell going `delta > 0` means re-classify as
/// `Reachable`, not that the kernel misfired.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum FlashRouting {
    Reachable,
    Dormant,
}

fn run_cell(model_path: &Path, ctx: usize, depth: f32, routing: FlashRouting) {
    run_cell_kind(model_path, ctx, depth, routing, FlashKind::Turbo);
}

/// Phase 4 entry point for planar_flash_decode cells. Same retrieval +
/// dispatch-counter contract as the TurboFlash cells, but consults the
/// `planar_flash_decode` OnceLock + counter and forces `KvQuant::PlanarK`.
fn run_pflash_cell(model_path: &Path, ctx: usize, depth: f32, routing: FlashRouting) {
    run_cell_kind(model_path, ctx, depth, routing, FlashKind::Pflash);
}

fn run_cell_kind(
    model_path: &Path,
    ctx: usize,
    depth: f32,
    routing: FlashRouting,
    kind: FlashKind,
) {
    // The mode label reflects the production OnceLock gate, not just the raw
    // env var. The kernel gates read the env exactly once per process, so
    // this is the same boolean the decode path observes — eliminating drift
    // between the harness label and the gate.
    let enabled = match kind {
        FlashKind::Turbo => rmlx_kv_quant::turbo_flash_msl::turbo_flash_enabled(),
        FlashKind::Pflash => rmlx_kv_quant::planar_flash_decode_msl::planar_flash_decode_enabled(),
    };
    let mode = if enabled { "ON" } else { "OFF" };
    let family = match kind {
        FlashKind::Turbo => "turbo_flash",
        FlashKind::Pflash => "planar_flash_decode",
    };

    // Snapshot the dispatch counter around the generate call. Delta > 0
    // proves the MSL kernel actually fired; delta == 0 means the call
    // silently fell back to the legacy path (mixed_quantized_sdpa for Turbo,
    // fused-QK dequant+softmax+SV for Pflash).
    let read_count = || match kind {
        FlashKind::Turbo => rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count(),
        FlashKind::Pflash => {
            rmlx_kv_quant::planar_flash_decode_msl::planar_flash_decode_dispatch_count()
        }
    };
    let dispatch_before = read_count();
    let res = run_one(model_path, ctx, depth, kind);
    let dispatch_after = read_count();
    let dispatch_delta = dispatch_after - dispatch_before;

    println!(
        "[niah] family={family} model={} mode={mode} routing={routing:?} ctx={ctx} \
         depth={depth:.2} prompt_len={} needle_found={} dispatch_delta={dispatch_delta} \
         decoded={:?}",
        model_path.display(),
        res.prompt_len,
        res.needle_found,
        res.decoded
    );

    // Prove the kernel really dispatched (ON) or really did not (OFF).
    // Without this check, the prior 80-cell "≥95% retrieval" could have
    // measured bf16 fallback the whole time.
    //
    // PlanarK has a warm-TTFT bf16-K shortcut (`update_and_sdpa` skips the
    // fused-QK fast paths when `decode_fp16_k.is_some()`, falling through
    // to the legacy bf16 SDPA). When the cache holds the live bf16 K seed
    // for the entire decode window the Pflash kernel intentionally never
    // fires — needle retrieval is what we care about, not kernel dispatch
    // on every post-prefill cache. The Pflash+Reachable+ON contract is
    // therefore relaxed: dispatch_delta can be 0 as long as the needle is
    // recovered. Turbo keeps the strict contract because TurboFlash
    // dispatches BUILD a head-major K8V4 buffer from the bf16 seed and
    // remain reachable through warm-TTFT.
    match (mode, routing, kind) {
        ("ON", FlashRouting::Reachable, FlashKind::Turbo) => {
            assert!(
                dispatch_delta > 0,
                "[niah] FAIL: family={family} ON, routing=Reachable, but kernel \
                 never dispatched (model={} ctx={ctx} depth={depth:.2}). Likely gate \
                 not met (Turbo: kv_seq > 4096, head_dim ∈ {{128, 256}}, \
                 KvQuant::K8V4, decode q_seq == 1). The recorded ON run \
                 measured the legacy fallback, not the MSL kernel.",
                model_path.display(),
            );
        }
        ("ON", FlashRouting::Reachable, FlashKind::Pflash) => {
            // Warm-TTFT bf16-K shortcut intentionally bypasses the PlanarK
            // fused-QK / flash-decode kernels when the prefill bf16 K seed is
            // live. The kernels stay reachable for the
            // `--planar-fused-qk on` / `RMLX_PLANAR_FLASH_DECODE=1` decode
            // path on a cache without an active seed, but the NIAH harness
            // unconditionally runs through a post-prefill greedy decode where
            // the seed IS live. dispatch_delta = 0 is the ONLY correct outcome
            // here; a non-zero delta means the kernel re-fired on the seeded
            // path, which is a regression (the warm-TTFT gate has been lost).
            assert_eq!(
                dispatch_delta,
                0,
                "warm-TTFT contract: PlanarK Pflash kernel MUST stay \
                 dormant when bf16 K seed is live; got dispatch_delta={dispatch_delta} \
                 (family={family} model={} ctx={ctx} depth={depth:.2})",
                model_path.display(),
            );
        }
        ("ON", FlashRouting::Dormant, _) => {
            // A gate — not the shared-KV topology — keeps this cell's kernel
            // dormant; see `FlashRouting` for the measured per-cell reason.
            // The ON run is therefore byte-identical to OFF.
            assert_eq!(
                dispatch_delta,
                0,
                "[niah] FAIL: family={family} routing=Dormant but kernel still \
                 dispatched (model={} ctx={ctx} depth={depth:.2}). A gate moved: \
                 confirm which one, then re-classify this cell as Reachable.",
                model_path.display(),
            );
            eprintln!(
                "[niah] NOTE: family={family} model={} routing=Dormant — ON \
                 run measures the legacy fallback, NOT the MSL kernel.",
                model_path.display(),
            );
        }
        ("OFF", FlashRouting::Reachable | FlashRouting::Dormant, _) => {
            assert_eq!(
                dispatch_delta,
                0,
                "[niah] FAIL: family={family} OFF but kernel dispatched \
                 (model={} ctx={ctx} depth={depth:.2}). The OFF path must never \
                 enqueue the MSL kernel — investigate the gate.",
                model_path.display(),
            );
        }
        _ => unreachable!("mode is exactly ON or OFF"),
    }

    assert!(
        res.needle_found,
        "[niah] FAIL: needle {NEEDLE:?} not in decoded output \
         (family={family} mode={mode} ctx={ctx} depth={depth:.2} \
         dispatch_delta={dispatch_delta})\n  decoded={:?}",
        res.decoded
    );
}

// ── #[test] cells: per-model × per-ctx × per-depth ────────────────────────

// Context lengths and depths: 8k / 16k / 32k × 5 depths. Each cell is its
// own `#[test]` so a single failure is localised in the report.

macro_rules! niah_cell {
    ($name:ident, $env:literal, $expected_archs:expr, $routing:expr, $ctx:expr, $depth:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            let Some(p) = model_path($env) else { return };
            if common::skip_if_arch_mismatch(&p, stringify!($name), $expected_archs) {
                return;
            }
            run_cell(&p, $ctx, $depth, $routing);
        }
    };
}

// ── Bonsai (Qwen3) ────────────────────────────────────────────────────────
// Bonsai max-pos = 65k via YARN (config.json ships factor=4 on a 16k
// training base). 8k / 16k cells run at the un-scaled band; 32k cells
// exercise the YARN extension path.

niah_cell!(
    niah_bonsai_8k_d10,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.10
);
niah_cell!(
    niah_bonsai_8k_d30,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.30
);
niah_cell!(
    niah_bonsai_8k_d50,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.50
);
niah_cell!(
    niah_bonsai_8k_d70,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.70
);
niah_cell!(
    niah_bonsai_8k_d90,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.90
);

niah_cell!(
    niah_bonsai_16k_d10,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.10
);
niah_cell!(
    niah_bonsai_16k_d30,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.30
);
niah_cell!(
    niah_bonsai_16k_d50,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.50
);
niah_cell!(
    niah_bonsai_16k_d70,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.70
);
niah_cell!(
    niah_bonsai_16k_d90,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.90
);

// Bonsai 32k cells: YARN RoPE (config.json ships `rope_scaling: yarn,
// factor: 4.0, original_max_position_embeddings: 16384`, extending
// max_position_embeddings = 65536). Without YARN the model re-uses
// training-time RoPE freqs past 16k and produces garbage. 5 depths at 32k
// mirror the 16k cells.
niah_cell!(
    niah_bonsai_32k_d10,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    32_768,
    0.10
);
niah_cell!(
    niah_bonsai_32k_d30,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    32_768,
    0.30
);
niah_cell!(
    niah_bonsai_32k_d50,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    32_768,
    0.50
);
niah_cell!(
    niah_bonsai_32k_d70,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    32_768,
    0.70
);
niah_cell!(
    niah_bonsai_32k_d90,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    32_768,
    0.90
);

// ── Gemma4 e4b ───────────────────────────────────────────────────────────

niah_cell!(
    niah_gemma4_8k_d10,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    8_192,
    0.10
);
niah_cell!(
    niah_gemma4_8k_d30,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    8_192,
    0.30
);
niah_cell!(
    niah_gemma4_8k_d50,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    8_192,
    0.50
);
niah_cell!(
    niah_gemma4_8k_d70,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    8_192,
    0.70
);
niah_cell!(
    niah_gemma4_8k_d90,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    8_192,
    0.90
);

niah_cell!(
    niah_gemma4_16k_d10,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    16_384,
    0.10
);
niah_cell!(
    niah_gemma4_16k_d30,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    16_384,
    0.30
);
niah_cell!(
    niah_gemma4_16k_d50,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    16_384,
    0.50
);
niah_cell!(
    niah_gemma4_16k_d70,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    16_384,
    0.70
);
niah_cell!(
    niah_gemma4_16k_d90,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    16_384,
    0.90
);

niah_cell!(
    niah_gemma4_32k_d10,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.10
);
niah_cell!(
    niah_gemma4_32k_d30,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.30
);
niah_cell!(
    niah_gemma4_32k_d50,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.50
);
niah_cell!(
    niah_gemma4_32k_d70,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.70
);
niah_cell!(
    niah_gemma4_32k_d90,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.90
);

// ── Qwen3.6 35B-A3B ──────────────────────────────────────────────────────

niah_cell!(
    niah_qwen36_8k_d10,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    8_192,
    0.10
);
niah_cell!(
    niah_qwen36_8k_d30,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    8_192,
    0.30
);
niah_cell!(
    niah_qwen36_8k_d50,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    8_192,
    0.50
);
niah_cell!(
    niah_qwen36_8k_d70,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    8_192,
    0.70
);
niah_cell!(
    niah_qwen36_8k_d90,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    8_192,
    0.90
);

niah_cell!(
    niah_qwen36_16k_d10,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    16_384,
    0.10
);
niah_cell!(
    niah_qwen36_16k_d30,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    16_384,
    0.30
);
niah_cell!(
    niah_qwen36_16k_d50,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    16_384,
    0.50
);
niah_cell!(
    niah_qwen36_16k_d70,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    16_384,
    0.70
);
niah_cell!(
    niah_qwen36_16k_d90,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    16_384,
    0.90
);

niah_cell!(
    niah_qwen36_32k_d10,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    32_768,
    0.10
);
niah_cell!(
    niah_qwen36_32k_d30,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    32_768,
    0.30
);
niah_cell!(
    niah_qwen36_32k_d50,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    32_768,
    0.50
);
niah_cell!(
    niah_qwen36_32k_d70,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    32_768,
    0.70
);
niah_cell!(
    niah_qwen36_32k_d90,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Reachable,
    32_768,
    0.90
);

// ── planar_flash_decode NIAH cells ───────────────────────────────────────
//
// Mirrors the TurboFlash cells above but exercises the `planar_flash_decode`
// MSL kernel + the planar_fused_qk PlanarK chain. Two-step macro because the
// cells use `KvQuant::PlanarK` (not K8V4) and consult a different env-var /
// dispatch counter — see `FlashKind::Pflash`.
//
// Routing:
// - Bonsai (`Qwen3ForCausalLM`)              → Reachable. Routes through
//   `KvCache::update_and_sdpa` → `sdpa_dispatch` → `update_and_sdpa_planar_k_fused`.
// - Qwen3.6 (`Qwen3_5MoeForConditionalGeneration`) → Dormant. `validate_resolved`
//   rejects `KvQuant::PlanarK` outright with `QwenMoePlanarKRejected`. The
//   cache is never built; the kernel can never dispatch. Cells are kept so
//   the assertion enforces the routing contract.
// - Gemma4 (`Gemma4ForConditionalGeneration`) → Dormant. It DOES reach the same
//   fused arm via `update_and_sdpa_shared_source`, but the warm-TTFT bf16-K-seed
//   gate keeps the kernel dormant. Re-check if that gate changes.
//
// Bonsai max-pos = 16k (no YARN), so cells cap at 16k — matches the existing
// TurboFlash Bonsai cells.

macro_rules! niah_pflash_cell {
    ($name:ident, $env:literal, $expected_archs:expr, $routing:expr, $ctx:expr, $depth:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            let Some(p) = model_path($env) else { return };
            if common::skip_if_arch_mismatch(&p, stringify!($name), $expected_archs) {
                return;
            }
            run_pflash_cell(&p, $ctx, $depth, $routing);
        }
    };
}

// ── Bonsai (Qwen3) — Reachable. 8k + 16k × 5 depths. ─────────────────────

niah_pflash_cell!(
    niah_pflash_bonsai_8k_d10,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.10
);
niah_pflash_cell!(
    niah_pflash_bonsai_8k_d30,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.30
);
niah_pflash_cell!(
    niah_pflash_bonsai_8k_d50,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.50
);
niah_pflash_cell!(
    niah_pflash_bonsai_8k_d70,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.70
);
niah_pflash_cell!(
    niah_pflash_bonsai_8k_d90,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    8_192,
    0.90
);

niah_pflash_cell!(
    niah_pflash_bonsai_16k_d10,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.10
);
niah_pflash_cell!(
    niah_pflash_bonsai_16k_d30,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.30
);
niah_pflash_cell!(
    niah_pflash_bonsai_16k_d50,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.50
);
niah_pflash_cell!(
    niah_pflash_bonsai_16k_d70,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.70
);
niah_pflash_cell!(
    niah_pflash_bonsai_16k_d90,
    "RMLX_TEST_MODEL_BONSAI",
    EXPECTED_ARCHS_BONSAI,
    FlashRouting::Reachable,
    16_384,
    0.90
);

// ── Qwen3.6 35B-A3B — Dormant. PlanarK rejected at validate_resolved
//   (`QwenMoePlanarKRejected`).  Cells exist so the dispatch-counter
//   assertion enforces "Dormant means delta == 0 even with ON". 32k per
//   ticket DoD ceiling. ─────────────────────────────────────────────────

niah_pflash_cell!(
    niah_pflash_qwen36_32k_d10,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Dormant,
    32_768,
    0.10
);
niah_pflash_cell!(
    niah_pflash_qwen36_32k_d50,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Dormant,
    32_768,
    0.50
);
niah_pflash_cell!(
    niah_pflash_qwen36_32k_d90,
    "RMLX_TEST_MODEL_QWEN36",
    EXPECTED_ARCHS_QWEN36,
    FlashRouting::Dormant,
    32_768,
    0.90
);

// ── Gemma4 e4b — Dormant. The fused arm IS reached via
//   `update_and_sdpa_shared_source`; the warm-TTFT bf16-K-seed gate is what
//   keeps the kernel dormant. ──────────────────────────────────────────

niah_pflash_cell!(
    niah_pflash_gemma4_32k_d10,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.10
);
niah_pflash_cell!(
    niah_pflash_gemma4_32k_d50,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.50
);
niah_pflash_cell!(
    niah_pflash_gemma4_32k_d90,
    "RMLX_TEST_MODEL_GEMMA4_E4B",
    EXPECTED_ARCHS_GEMMA4,
    FlashRouting::Dormant,
    32_768,
    0.90
);
