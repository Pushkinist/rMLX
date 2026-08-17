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
//!
//! ## Snapshot resolution
//!
//! Each golden covers ONE architecture and names its own snapshot by slug.
//! Two sources, in order:
//!
//! 1. `RMLX_KV_TEST_MODEL` — one model for a whole run. Goldens for the other
//!    architectures skip, which is what makes it usable at all.
//! 2. the golden's snapshot slug under `RMLX_O_MODELS_ROOT`.
//!
//! Step 2 is what arms these gates by default: every `make` target exports
//! `RMLX_O_MODELS_ROOT`, so a machine holding the snapshots runs every golden
//! whose model is on disk, instead of at most the one `RMLX_KV_TEST_MODEL`
//! happens to name.
//!
//! The per-architecture `RMLX_TEST_MODEL_*` family is deliberately NOT consulted
//! here, even though several of these snapshots have one. Those variables mean
//! "a snapshot of this family for the smoke and template suites", and the
//! documented workflow exports all three primary ones for a whole
//! `cargo test --workspace` run. A golden is a byte-exact fixture over ONE
//! checkpoint's weights, so letting a persistent shell export steer it turns any
//! same-family substitution — a QAT rebuild, a re-quantized sibling — into a
//! token mismatch that looks exactly like a decode regression, and the
//! architecture check cannot tell the two apart because the substitute passes
//! it. Retargeting one golden is still possible without that hazard: each is its
//! own test binary, so `RMLX_KV_TEST_MODEL=<path> cargo test --test
//! <arch>_golden_tokens` names one deliberately, per invocation, and reaches no
//! other golden.
//!
//! A variable that NAMES a directory which is not this golden's snapshot is a
//! hard failure, not a skip. Configuration is present and wrong there, and a
//! skip would report success without asserting anything — the same shape as a
//! gate that cannot fail. Absence (nothing set, or a models root that does not
//! hold this slug) stays a skip: a developer without the weights cannot run the
//! gate, and must not be blocked by it.

use std::path::{Path, PathBuf};

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod snapshot_tests;

/// Single-model override: names ONE snapshot for a whole test run.
pub const SINGLE_MODEL_VAR: &str = "RMLX_KV_TEST_MODEL";
/// Root directory holding every snapshot, addressed by slug.
pub const MODELS_ROOT_VAR: &str = "RMLX_O_MODELS_ROOT";

/// The snapshot one golden covers: the slug it lives under in the models root,
/// and the architectures it was recorded against.
pub struct GoldenModel {
    pub slug: &'static str,
    pub archs: &'static [&'static str],
}

/// Which configuration produced a snapshot path. This decides whether an arch
/// mismatch is benign: `SINGLE_MODEL_VAR` names one model for a run of
/// per-arch goldens, so most of them are expected not to match it. The slug
/// names THIS golden's snapshot, so a mismatch there is a wrong pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    SingleModelOverride,
    ModelsRoot,
}

/// Outcome of the snapshot lookup.
#[derive(Debug, PartialEq, Eq)]
pub enum Snapshot {
    Found {
        path: PathBuf,
        from: Source,
    },
    /// Nothing on this machine points at the snapshot.
    Absent(String),
    /// A variable names a directory that is not a snapshot.
    Misconfigured(String),
}

/// What the caller must do with a lookup result.
#[derive(Debug, PartialEq, Eq)]
pub enum Gate {
    Run(PathBuf),
    Skip(String),
    Fail(String),
}

/// A directory is a snapshot when it carries the config every loader reads.
fn is_snapshot_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file()
}

/// Resolve one golden's snapshot from the two configuration values, without
/// reading the environment — the caller passes them in so this stays a pure
/// function over a directory tree.
pub fn pick_snapshot(
    single_model: Option<&str>,
    models_root: Option<&str>,
    slug: &str,
) -> Snapshot {
    // An exported-but-empty variable is how a shell spells "unset"; treating it
    // as a path yields a nonsense lookup at the filesystem root.
    if let Some(p) = single_model.filter(|p| !p.is_empty()) {
        let path = PathBuf::from(p);
        if is_snapshot_dir(&path) {
            return Snapshot::Found {
                path,
                from: Source::SingleModelOverride,
            };
        }
        return Snapshot::Misconfigured(format!(
            "{SINGLE_MODEL_VAR}={p} does not name a snapshot directory (no config.json in it)"
        ));
    }

    if let Some(root) = models_root.filter(|r| !r.is_empty()) {
        let path = PathBuf::from(root).join(slug);
        if is_snapshot_dir(&path) {
            return Snapshot::Found {
                path,
                from: Source::ModelsRoot,
            };
        }
        // A models root is a bulk convenience — nobody holds every snapshot,
        // and the Makefile points it at a repo-local `models/` dir that need
        // not exist. Absence under it is not a misconfiguration.
        return Snapshot::Absent(format!(
            "{slug} is not under {MODELS_ROOT_VAR}={root}; put the snapshot (or a \
             symlink to it) there, or name it with {SINGLE_MODEL_VAR} for a single run"
        ));
    }

    Snapshot::Absent(format!(
        "no snapshot configured — set {MODELS_ROOT_VAR} (holding {slug}) or {SINGLE_MODEL_VAR}"
    ))
}

/// Turn a lookup result plus the resolved snapshot's `architectures[0]` into a
/// run / skip / fail decision. `arch` is only consulted for [`Snapshot::Found`].
pub fn gate(snapshot: Snapshot, arch: &str, model: &GoldenModel) -> Gate {
    match snapshot {
        Snapshot::Found { path, from } => {
            if model.archs.contains(&arch) {
                return Gate::Run(path);
            }
            let expected = model.archs;
            let shown = path.display();
            match from {
                // One model for a whole run of per-arch goldens: the goldens
                // for the other architectures are meant to stand down.
                Source::SingleModelOverride => Gate::Skip(format!(
                    "{SINGLE_MODEL_VAR} names arch \"{arch}\", this golden covers {expected:?}"
                )),
                Source::ModelsRoot => Gate::Fail(format!(
                    "{shown} (slug {}) has arch \"{arch}\", not one of {expected:?}",
                    model.slug
                )),
            }
        }
        Snapshot::Absent(why) => Gate::Skip(why),
        Snapshot::Misconfigured(why) => Gate::Fail(why),
    }
}

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
/// not match any of the `expected` architecture strings.
///
/// For a golden, prefer [`model_for`] — it resolves the snapshot and gates the
/// arch in one place. This remains for suites that resolve their own per-arch
/// path and only need the arch check (`niah_long_context.rs`).
pub fn skip_if_arch_mismatch(model_dir: &Path, test_name: &str, expected: &[&str]) -> bool {
    let arch = model_arch(model_dir);
    if expected.contains(&arch.as_str()) {
        return false;
    }
    eprintln!("SKIP {test_name}: model arch \"{arch}\" != expected {expected:?}");
    true
}

/// Resolve the snapshot for one golden, or `None` when this machine has none.
///
/// Panics on a configuration that is present and wrong — a variable pointing at
/// a directory that is not a snapshot, or at a snapshot of the wrong
/// architecture. Returning `None` there would let a typo report a green run
/// that asserted nothing.
pub fn model_for(model: &GoldenModel, test_name: &str) -> Option<PathBuf> {
    let single = std::env::var(SINGLE_MODEL_VAR).ok();
    let root = std::env::var(MODELS_ROOT_VAR).ok();
    let snapshot = pick_snapshot(single.as_deref(), root.as_deref(), model.slug);
    let arch = match &snapshot {
        Snapshot::Found { path, .. } => model_arch(path),
        _ => String::new(),
    };
    match gate(snapshot, &arch, model) {
        Gate::Run(path) => Some(path),
        Gate::Skip(why) => {
            eprintln!("SKIP {test_name}: {why}");
            None
        }
        Gate::Fail(why) => panic!("{test_name}: {why}"),
    }
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
/// `model_path` comes from [`model_for`] — resolution happens once, at the
/// caller, so there is exactly one place that decides whether a golden runs.
///
/// Behaviour:
/// * `RMLX_REGEN_GOLDENS=1` set → decode + WRITE the golden fixture, no assert.
/// * otherwise → decode + ASSERT exact token-id equality.
pub fn run_golden_test(fixture_tag: &str, kv_quant: KvQuant, model_path: &Path) {
    let device = Device::Gpu;
    let model =
        arch::load_model(model_path, device, &arch::LoadOpts::default()).expect("arch::load_model");

    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer.json");

    // Build the prompt: explicit BOS (when resolvable) + prompt body tokenized
    // with add_special_tokens=false. Falls back to the tokenizer's own special
    // handling when no BOS token is configured.
    let bos = resolve_bos_id(model_path, &tokenizer);
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
