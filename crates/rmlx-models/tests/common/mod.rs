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
    clippy::format_push_string,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]
// Shared test helper compiled into each golden-token test binary. Each binary
// uses only a subset of the items, so dead_code / unreachable_pub fire per
// binary — allow them here (standard `tests/common` pattern).
#![allow(dead_code, unreachable_pub)]

//! Shared harness for the per-arch golden-token decode tests.
//!
//! A deterministic, SERVER-FREE correctness gate: temp=0 greedy decode of a
//! fixed prompt must reproduce a checked-in golden token-id sequence exactly.
//! This isolates genuine model-run regressions from server/metrics noise — the
//! harness touches ONLY `rmlx-models` (`arch::load_model` + `generate_greedy`),
//! never HTTP, EventRecorder, the metrics DB, or any CSV.
//!
//! Goldens are recorded once with `RMLX_REGEN_GOLDENS=1` (the test writes the
//! fixture instead of asserting) and committed. Re-running without the env var
//! asserts byte-for-byte token-id equality against the committed fixture.

use std::path::{Path, PathBuf};

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

/// Read the first entry of the `architectures` array from `<model_dir>/config.json`.
/// Returns an empty string on any parse failure so callers can treat it as a mismatch.
pub fn model_arch(model_dir: &Path) -> String {
    let cfg_path = model_dir.join("config.json");
    let Ok(data) = std::fs::read(&cfg_path) else {
        return String::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return String::new();
    };
    v.get("architectures")
        .and_then(|a| a.get(0))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_owned()
}

/// Return `true` (and print a SKIP notice) when the model at `model_dir` does
/// not match any of the `expected` architecture strings. Return `false` when it
/// matches — the caller should then run the golden assertion.
pub fn skip_if_arch_mismatch(model_dir: &Path, test_name: &str, expected: &[&str]) -> bool {
    let arch = model_arch(model_dir);
    if expected.contains(&arch.as_str()) {
        return false;
    }
    eprintln!("SKIP {test_name}: model arch \"{arch}\" != expected {expected:?}");
    true
}

/// Number of decode tokens to compare. 32 is enough to catch a divergence
/// while staying fast at temp=0 greedy.
pub const N_GOLDEN_TOKENS: usize = 32;

/// Fixed 32-token-ish coherent English prompt. Coherent prose (not random ids)
/// keeps every arch in a stable, non-degenerate decode regime so token-identity
/// is meaningful (random ids make even a correct model loop on special tokens).
/// The exact token count after tokenization is logged in the golden header.
pub const GOLDEN_PROMPT: &str = "The capital of France is Paris. The capital of Japan is Tokyo. \
     Explain in one short sentence why the sky appears blue during the day.";

/// Resolve the model path from `RMLX_KV_TEST_MODEL`. Returns `None` (and prints
/// a skip note) when unset, mirroring the existing equivalence test gate.
pub fn model_path_from_env() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") {
        Some(PathBuf::from(p))
    } else {
        eprintln!("RMLX_KV_TEST_MODEL not set — skipping golden-token test");
        None
    }
}

/// Resolve the BOS token id, mirroring `arch::run_smoke_probe`'s fallback chain
/// at the test level (the production resolver is private). Falls back to the
/// tokenizer's own `add_special_tokens` if no explicit BOS token is found.
fn resolve_bos_id(model_dir: &Path, tk: &tokenizers::Tokenizer) -> Option<u32> {
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

/// Run the golden-token harness for one (arch, kv_quant) pair.
///
/// `fixture_tag` names the committed golden file
/// (`tests/fixtures/<fixture_tag>.golden.txt`). `kv_quant` pins the KV cache
/// mode so the golden is reproducible regardless of the auto-resolver default.
///
/// Behaviour:
/// * `RMLX_REGEN_GOLDENS=1` set → decode + WRITE the golden fixture, no assert.
/// * otherwise → decode + ASSERT exact token-id equality.
pub fn run_golden_test(fixture_tag: &str, kv_quant: KvQuant) {
    let Some(model_path) = model_path_from_env() else {
        return;
    };

    let device = Device::Gpu;
    let model = arch::load_model(&model_path, device, &arch::LoadOpts::default())
        .expect("arch::load_model");

    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer.json");

    // Build the prompt: explicit BOS (when resolvable) + prompt body tokenized
    // with add_special_tokens=false. Falls back to the tokenizer's own special
    // handling when no BOS token is configured.
    let bos = resolve_bos_id(&model_path, &tokenizer);
    let prompt_ids: Vec<u32> = match bos {
        Some(bos_id) => {
            let enc = tokenizer
                .encode(GOLDEN_PROMPT, false)
                .expect("tokenize prompt");
            let mut ids = Vec::with_capacity(1 + enc.get_ids().len());
            ids.push(bos_id);
            ids.extend_from_slice(enc.get_ids());
            ids
        }
        None => tokenizer
            .encode(GOLDEN_PROMPT, true)
            .expect("tokenize prompt")
            .get_ids()
            .to_vec(),
    };

    // temp=0 greedy + fixed seed = fully deterministic (matches run_smoke_probe).
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
            N_GOLDEN_TOKENS,
            device,
            Some(kv_quant),
            None, // max_ctx: use arch default
            1,    // single-slot cache
            &[],  // no EOS stop — force the full N tokens
            &mut |_| None,
            None, // no sampler constraint
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy");

    let token_ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    // Decoded text for the smoke-coherence note (printed, not asserted). Use the
    // tokenizer's real decoder (handles byte-BPE merges) rather than naive
    // per-piece concatenation, which renders control bytes (Ġ/Ċ) unreadably.
    let decoded: String = tokenizer
        .decode(&token_ids, false)
        .unwrap_or_else(|_| steps.iter().map(|s| s.piece.as_ref()).collect());

    let fixture_path = fixtures_dir().join(format!("{fixture_tag}.golden.txt"));

    if std::env::var("RMLX_REGEN_GOLDENS").is_ok() {
        write_golden(&fixture_path, fixture_tag, &kv_quant, &token_ids, &decoded);
        eprintln!(
            "[{fixture_tag}] WROTE golden ({} ids, kv={kv_quant}) -> {}\n  decoded: {decoded:?}",
            token_ids.len(),
            fixture_path.display()
        );
        return;
    }

    let golden = read_golden(&fixture_path).unwrap_or_else(|| {
        panic!(
            "[{fixture_tag}] golden fixture missing: {}\n  re-run once with RMLX_REGEN_GOLDENS=1 to record it",
            fixture_path.display()
        )
    });

    assert_eq!(
        token_ids, golden,
        "[{fixture_tag}] golden-token mismatch (kv={kv_quant}) — decode regression.\n  \
         got    = {token_ids:?}\n  golden = {golden:?}\n  decoded(got) = {decoded:?}"
    );
}

/// Absolute path to the checked-in fixtures directory.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Write the golden file: a header comment block (tag, kv_quant, decoded text)
/// followed by one token id per line. The header is for human inspection; only
/// the numeric lines are parsed back by `read_golden`.
fn write_golden(path: &Path, tag: &str, kv: &KvQuant, ids: &[u32], decoded: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).expect("create fixtures dir");
    let mut out = String::new();
    out.push_str(&format!("# golden tokens — tag={tag} kv_quant={kv}\n"));
    out.push_str(&format!("# n_tokens={}\n", ids.len()));
    out.push_str(&format!("# decoded: {decoded:?}\n"));
    for id in ids {
        out.push_str(&format!("{id}\n"));
    }
    std::fs::write(path, out).expect("write golden fixture");
}

/// Parse a golden file into its token-id sequence (ignores `#` comment lines
/// and blank lines). Returns `None` if the file does not exist.
fn read_golden(path: &Path) -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                l.parse::<u32>()
                    .expect("golden line must be a u32 token id")
            })
            .collect(),
    )
}
