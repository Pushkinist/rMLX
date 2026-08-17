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
//! Two sources:
//!
//! 1. `RMLX_KV_TEST_MODEL`, **but only for the golden whose architecture it
//!    serves**. Pointed at another architecture it is not a statement about
//!    this golden, and resolution falls through to the slug rather than
//!    standing the golden down.
//! 2. the golden's snapshot slug under `RMLX_O_MODELS_ROOT`.
//!
//! Step 2 is what arms these gates by default: every `make` target exports
//! `RMLX_O_MODELS_ROOT` when it resolves, so a machine holding the snapshots
//! runs every golden whose model is on disk, instead of at most the one
//! `RMLX_KV_TEST_MODEL` happens to name.
//!
//! The fall-through in step 1 matters because `RMLX_KV_TEST_MODEL` is not a
//! golden-only variable: `gemma4_kv_cache_equivalence.rs`, `cli_flags_e2e.rs`
//! and `projects_toml_e2e.rs` all require it. A developer who exports it for
//! those — typically at a Gemma4 path — would otherwise disarm four of the five
//! goldens on every run, which is the original defect surviving for exactly the
//! developer who most needs these gates.
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

/// Outcome of probing one source. A `Found` carries the architecture read from
/// the snapshot's `config.json`, so the decision below is a pure function of
/// data rather than of two loose parallel strings a caller could transpose.
#[derive(Debug, PartialEq, Eq)]
pub enum Snapshot {
    Found {
        path: PathBuf,
        arch: String,
    },
    /// Nothing on this machine points at the snapshot.
    Absent(String),
    /// Configuration is present and wrong.
    Misconfigured(String),
}

/// What the caller must do with a resolution.
#[derive(Debug, PartialEq, Eq)]
pub enum Gate {
    /// `note` carries anything the operator should know about HOW this path was
    /// reached — chiefly that an override they named was stood down.
    Run {
        path: PathBuf,
        note: Option<String>,
    },
    Skip(String),
    Fail(String),
}

/// `RMLX_REGEN_GOLDENS` — writes the fixture instead of asserting it.
pub const REGEN_VAR: &str = "RMLX_REGEN_GOLDENS";

/// True when the harness is recording fixtures rather than checking them.
pub fn regen_requested() -> bool {
    std::env::var(REGEN_VAR).is_ok()
}

/// Largest top-2 logprob margin at a diverging step that still counts as a tie
/// this engine's float dtype cannot resolve.
///
/// A regenerated golden with no recorded reason is indistinguishable from a
/// hidden regression, so overwriting a fixture whose ids changed is only
/// defensible when the model had no real preference at the step that moved.
/// Above this the divergence is a decision the model made confidently, which is
/// a behaviour change to explain, not a fixture to refresh.
///
/// **Where 0.10 comes from.** The top-2 *logprob* gap equals the top-2 *logit*
/// gap — the log-sum-exp normaliser is common to both terms and cancels — and
/// after the load-time bf16 cast those logits are bf16. bf16 carries 7 explicit
/// mantissa bits, so the smallest representable gap is one ULP at the top
/// logit's magnitude: ~0.0625 for |logit| in [8, 16), ~0.125 in [16, 32). 0.10
/// therefore admits an exact tie at every magnitude and a one-ULP gap only in
/// the lower octave.
///
/// **One** measured case exists — bonsai, index 18, margin exactly 0.00000000 —
/// so nothing in the tree distinguishes 0.10 from 0.01. A single exact tie is
/// no evidence for any particular bound above zero. If a case ever lands
/// between, **tighten rather than widen**: a genuine tie stays a tie under any
/// smaller bound, while raising the bound buys nothing except the ability to
/// absorb a real preference.
pub const REGEN_MAX_TIE_MARGIN: f32 = 0.10;

/// Index of the first differing token id. `None` when the sequences match.
/// A length change is reported at the first index the shorter one lacks.
pub fn first_divergence(new_ids: &[u32], committed: &[u32]) -> Option<usize> {
    if let Some(i) = new_ids.iter().zip(committed).position(|(a, b)| a != b) {
        return Some(i);
    }
    if new_ids.len() == committed.len() {
        None
    } else {
        Some(new_ids.len().min(committed.len()))
    }
}

/// Whether a regeneration may overwrite the committed fixture.
#[derive(Debug, PartialEq, Eq)]
pub enum Regen {
    Write(String),
    Refuse(String),
}

/// Adjudicate a regeneration.
///
/// `margin` is the top-2 logprob gap at [`first_divergence`], and is required
/// whenever the ids changed — an unmeasurable margin refuses, because a gate
/// that waves through what it could not check is the shape this whole harness
/// exists to remove.
pub fn regen_verdict(
    new_ids: &[u32],
    committed: Option<&[u32]>,
    margin: Option<f32>,
    max_margin: f32,
) -> Regen {
    let Some(committed) = committed else {
        return Regen::Write("no committed fixture — recording for the first time".to_owned());
    };
    let Some(i) = first_divergence(new_ids, committed) else {
        return Regen::Write("ids unchanged".to_owned());
    };
    if new_ids.len() != committed.len() {
        return Regen::Refuse(format!(
            "token count changed ({} -> {}); a length change is not a tie",
            committed.len(),
            new_ids.len()
        ));
    }
    let Some(margin) = margin else {
        return Regen::Refuse(format!(
            "ids diverge at index {i} but the top-2 margin there could not be measured"
        ));
    };
    if margin <= max_margin {
        Regen::Write(format!(
            "divergence at index {i} ({} -> {}) sits at a top-2 margin of {margin:.8}, \
             at or below the {max_margin} tie floor",
            committed[i], new_ids[i]
        ))
    } else {
        Regen::Refuse(format!(
            "ids diverge at index {i} ({} -> {}) at a top-2 margin of {margin:.8}, above the \
             {max_margin} tie floor — the model chose this token confidently, so something \
             other than a near-tie moved. Investigate before regenerating.",
            committed[i], new_ids[i]
        ))
    }
}

/// Files every snapshot must carry. Both are opened BY NAME: `config.json` by
/// [`model_arch`] and `arch::load_model`, `tokenizer.json` by
/// [`run_golden_test`]'s `Tokenizer::from_file`.
const REQUIRED_FILES: [&str; 2] = ["config.json", "tokenizer.json"];

/// Weight entrypoints. `rmlx_loader::load_shard_index` tries these two in this
/// order and errors if neither exists.
///
/// The entrypoint alone is NOT enough to call a snapshot runnable. A download
/// writes the small JSON files first and the multi-GB shards last, and
/// `model.safetensors.index.json` is itself one of those small JSON files — so
/// an interrupted **sharded** transfer leaves config + tokenizer + index and
/// zero `model-0000N-of-*.safetensors`. `load_shard_index` parses that index
/// happily and the failure surfaces later, when `ShardSet::open` cannot find
/// the files it names. Every test-target snapshot above a few GB is sharded, so
/// that is the majority shape, not an edge.
const WEIGHT_ENTRYPOINTS: [&str; 2] = ["model.safetensors.index.json", "model.safetensors"];

/// True when at least one `*.safetensors` file exists in `dir`.
///
/// `model.safetensors` satisfies this as itself; behind an index it is the
/// shards. Deliberately *presence of any*, not *all the index names*: checking
/// every entry means parsing the index and reimplementing the loader's own
/// `weight_map` walk here, which would drift. This closes the zero-shard case —
/// the one a download actually produces — and a partially-transferred shard set
/// still reaches the loader, which reports the missing file by name.
fn has_any_shard(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|e| {
        e.path()
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
    })
}

/// A directory is a snapshot when every file the harness opens by name is
/// there: both JSONs, an entrypoint `load_shard_index` accepts, and at least
/// one actual shard behind it.
fn is_snapshot_dir(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|f| dir.join(f).is_file())
        && WEIGHT_ENTRYPOINTS.iter().any(|f| dir.join(f).exists())
        && has_any_shard(dir)
}

/// Name what is missing, so a diagnosis says what to look at rather than just
/// "not a snapshot".
fn missing_file(dir: &Path) -> String {
    if let Some(f) = REQUIRED_FILES.iter().find(|f| !dir.join(f).is_file()) {
        return (*f).to_owned();
    }
    if !WEIGHT_ENTRYPOINTS.iter().any(|f| dir.join(f).exists()) {
        return "model.safetensors[.index.json]".to_owned();
    }
    if !has_any_shard(dir) {
        return "any *.safetensors shard (the index is present but names no file that exists)"
            .to_owned();
    }
    "(nothing)".to_owned()
}

/// Probe `RMLX_KV_TEST_MODEL`. `None` when it is unset — an exported-but-empty
/// variable is how a shell spells that, and treating it as a path yields a
/// nonsense lookup at the filesystem root.
///
/// A value that does not name a runnable snapshot is [`Snapshot::Misconfigured`]:
/// the operator named this path, so a typo or a moved snapshot must break the
/// run rather than skip it.
pub fn override_snapshot(single_model: Option<&str>) -> Option<Snapshot> {
    let p = single_model.filter(|p| !p.is_empty())?;
    let path = PathBuf::from(p);
    if is_snapshot_dir(&path) {
        let arch = model_arch(&path);
        return Some(Snapshot::Found { path, arch });
    }
    Some(Snapshot::Misconfigured(format!(
        "{SINGLE_MODEL_VAR}={p} is not a runnable snapshot directory (no {} in it)",
        missing_file(&path)
    )))
}

/// Probe `<RMLX_O_MODELS_ROOT>/<slug>`.
///
/// A root that is set but is not an existing directory is
/// [`Snapshot::Misconfigured`] — one keystroke there disarms every golden at
/// once, which is the widest blast radius in this harness and the last thing
/// that should report success by skipping. An existing root that simply does
/// not hold this slug is [`Snapshot::Absent`]: nobody holds every snapshot.
pub fn slug_snapshot(models_root: Option<&str>, slug: &str) -> Snapshot {
    let Some(root) = models_root.filter(|r| !r.is_empty()) else {
        return Snapshot::Absent(format!(
            "no snapshot configured — set {MODELS_ROOT_VAR} (holding {slug}) or {SINGLE_MODEL_VAR}"
        ));
    };
    let root_path = Path::new(root);
    if !root_path.is_dir() {
        return Snapshot::Misconfigured(format!(
            "{MODELS_ROOT_VAR}={root} is not an existing directory"
        ));
    }
    let path = root_path.join(slug);
    if is_snapshot_dir(&path) {
        let arch = model_arch(&path);
        return Snapshot::Found { path, arch };
    }
    Snapshot::Absent(format!(
        "{MODELS_ROOT_VAR}={root} does not hold a runnable {slug} (no {} in it); put the \
         snapshot, or a symlink to it, there",
        missing_file(&path)
    ))
}

/// Decide from both probes. `regen` is whether the harness is about to WRITE a
/// fixture rather than check one, which makes the override rules stricter.
///
/// The override wins **only for the golden it can serve**. Pointed at another
/// architecture it is not a statement about this golden, so resolution falls
/// through to the slug instead of standing down: `RMLX_KV_TEST_MODEL` is
/// required by the KV-equivalence, CLI-flag and projects-toml suites, and a
/// developer with it exported would otherwise silently disarm every golden but
/// one — reinstating the defect this harness exists to close, for exactly the
/// developer who most needs these gates.
///
/// Ranking the slug first instead would be wrong in the other direction: it
/// would make an override the operator DID name lose to the slug even when it
/// serves this golden.
///
/// Two rules keep the fall-through from becoming its own silent hazard:
///
/// * Under `regen` a stood-down override is a hard failure. Writing a committed
///   fixture from a snapshot the operator did not name, while the one they did
///   name is discarded, is how a golden acquires untraceable provenance — and
///   running the whole set with one override would give each fixture a
///   different origin with nothing said about it.
/// * An override whose `config.json` is present but unreadable (empty `arch`)
///   fails rather than falling through. Only a *legible, different*
///   architecture is a statement about another golden; an unparseable config in
///   a directory the operator named is a broken pointer, and the slug branch
///   already treats the same empty string that way.
pub fn choose(over: Option<Snapshot>, slug: Snapshot, model: &GoldenModel, regen: bool) -> Gate {
    let expected = model.archs;
    let mut stood_down = None;
    match over {
        Some(Snapshot::Found { path, arch }) => {
            if expected.contains(&arch.as_str()) {
                return Gate::Run { path, note: None };
            }
            if arch.is_empty() {
                return Gate::Fail(format!(
                    "{SINGLE_MODEL_VAR}={} has no readable architectures[0] in its config.json",
                    path.display()
                ));
            }
            if regen {
                return Gate::Fail(format!(
                    "{SINGLE_MODEL_VAR}={} is arch \"{arch}\", which this golden does not cover, \
                     and {REGEN_VAR} is set. Refusing to write the {} fixture from a snapshot you \
                     did not name — re-run the regen against the golden this model serves.",
                    path.display(),
                    model.slug
                ));
            }
            stood_down = Some(format!(
                "{SINGLE_MODEL_VAR} names arch \"{arch}\", which this golden does not cover"
            ));
        }
        // `override_snapshot` reports unset as `None`, so either other outcome
        // means the operator named something the harness cannot run.
        Some(Snapshot::Misconfigured(why) | Snapshot::Absent(why)) => return Gate::Fail(why),
        None => {}
    }

    match slug {
        Snapshot::Found { path, arch } => {
            if expected.contains(&arch.as_str()) {
                return Gate::Run {
                    path,
                    note: stood_down,
                };
            }
            Gate::Fail(format!(
                "{} (slug {}) has arch \"{arch}\", not one of {expected:?}",
                path.display(),
                model.slug
            ))
        }
        Snapshot::Misconfigured(why) => Gate::Fail(why),
        Snapshot::Absent(why) => Gate::Skip(match stood_down {
            Some(note) => format!("{why}; {note}"),
            None => why,
        }),
    }
}

/// Turn a decision into what [`model_for`] returns. Split out so the
/// `Fail` → `panic!` edge is covered by a test rather than only by a hand-run
/// invocation: it is the step that makes a wrong pointer visible at all.
pub fn apply(gate: Gate, test_name: &str) -> Option<PathBuf> {
    match gate {
        Gate::Run { path, note } => {
            // A resolution that quietly ignored something the operator named is
            // exactly what must not pass in silence.
            if let Some(note) = note {
                eprintln!("NOTE {test_name}: {note}; using {} instead", path.display());
            }
            Some(path)
        }
        Gate::Skip(why) => {
            eprintln!("SKIP {test_name}: {why}");
            None
        }
        Gate::Fail(why) => panic!("{test_name}: {why}"),
    }
}

/// Read the first entry of the `architectures` array from `<model_dir>/config.json`.
///
/// Returns an empty string when the file is missing or unparseable, which no
/// golden's expected-arch list contains, so callers read it as a mismatch. The
/// two callers then diverge on purpose, and the split is worth knowing:
/// [`choose`] turns a mismatch on a path THIS golden named into a hard failure,
/// while [`skip_if_arch_mismatch`] — used by suites that resolve their own
/// per-architecture path — reports a skip. A golden pins one checkpoint's bytes
/// and an unreadable config there means its own snapshot is broken; those suites
/// are handed a path chosen for another architecture as a matter of course.
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
    let over = override_snapshot(single.as_deref());
    let slug = slug_snapshot(root.as_deref(), model.slug);
    apply(choose(over, slug, model, regen_requested()), test_name)
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

    if regen_requested() {
        let committed = read_golden(&fixture_path);
        // Only pay for the margin measurement when the ids actually moved.
        // Decide the length case BEFORE measuring. `first_divergence` reports a
        // pure length change at the shorter sequence's end, which is out of
        // bounds for the shorter of the two — a raw index panic instead of the
        // designed refusal. `regen_verdict` refuses a length change at any
        // margin anyway, so measuring one is also a wasted GPU decode.
        let margin = committed
            .as_deref()
            .filter(|c| c.len() == token_ids.len())
            .and_then(|c| first_divergence(&token_ids, c))
            .and_then(|i| {
                measure_top2_margin(
                    &model,
                    &tokenizer,
                    &prompt_ids,
                    device,
                    kv_quant,
                    &penalty_cfg,
                    i,
                    &token_ids,
                )
            });
        match regen_verdict(
            &token_ids,
            committed.as_deref(),
            margin,
            REGEN_MAX_TIE_MARGIN,
        ) {
            Regen::Write(why) => {
                write_golden(&fixture_path, fixture_tag, &kv_quant, &token_ids, &decoded);
                eprintln!(
                    "[{fixture_tag}] WROTE golden ({} ids, kv={kv_quant}) -> {}\n  reason: {why}\n  decoded: {decoded:?}",
                    token_ids.len(),
                    fixture_path.display()
                );
            }
            Regen::Refuse(why) => panic!(
                "[{fixture_tag}] REFUSED to regenerate: {why}\n  got    = {token_ids:?}\n  \
                 decoded(got) = {decoded:?}"
            ),
        }
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

/// Re-decode the same prompt asking for top-2 logprobs, and return the gap
/// between the two best candidates at step `index`.
///
/// A second decode rather than raising `top_logprobs_k` on the first: the ids a
/// golden pins must come from the same sampler configuration they were recorded
/// under, and this path only runs while regenerating a fixture whose ids
/// already moved.
///
/// Returns `None` — which [`regen_verdict`] treats as a refusal — when the step
/// is missing, carries no logprobs, has fewer than two candidates, or the probe
/// run's ids differ from `first_run_ids` anywhere in `[..=index]`.
///
/// The prefix, not just `index`: a margin describes the distribution at a step
/// **given everything decoded before it**, so a probe that diverged at step 5
/// and happened to re-converge at step 18 measured a different continuation
/// than the one being written. Comparing one index would accept exactly that.
/// The prompt-cache exact hit makes the two runs agree in practice, which is
/// what would make the flaw silent rather than visible.
fn measure_top2_margin(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    device: Device,
    kv_quant: KvQuant,
    penalty_cfg: &PenaltyConfig,
    index: usize,
    first_run_ids: &[u32],
) -> Option<f32> {
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 2,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let mut token_history: Vec<u32> = Vec::new();
    let steps = model
        .generate_greedy(
            tokenizer,
            prompt_ids,
            N_GOLDEN_TOKENS,
            device,
            Some(kv_quant),
            None,
            1,
            &[],
            &mut |_| None,
            None,
            &sampler_cfg,
            &mut rng,
            penalty_cfg,
            &mut token_history,
        )
        .ok()?;

    let probe_ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    let prefix = first_run_ids.get(..=index)?;
    if probe_ids.get(..=index) != Some(prefix) {
        eprintln!(
            "  margin probe diverged from the first run within [0..={index}] — \
             non-deterministic, refusing to adjudicate\n    first = {prefix:?}\n    probe = {:?}",
            probe_ids.get(..=index)
        );
        return None;
    }

    let step = steps.get(index)?;
    let top = &step.logprobs.as_ref()?.top;
    let (_, best) = *top.first()?;
    let (_, second) = *top.get(1)?;
    Some(best - second)
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
