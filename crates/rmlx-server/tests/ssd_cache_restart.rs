//! — end-to-end smoke test: SSD KV cache survives a server restart.
//!
//! This is the final integration test of the Tiered-KV-Cache feature
//! (..48). It proves the full spill→restart→hydrate chain works against a
//! real `rmlx serve` process:
//!
//! 1. Phase 1 (populate + spill): start `rmlx serve <model> --kv-ssd-cache-gb 1
//! --prompt-cache-slots 1` on a free port. Send a long prompt A (records

//! 2. Phase 2 (restart + hydrate): KILL the server, clear the Metal claim,

//!
//! ## Gating
//!
//! `#[ignore]` + env-gated on `RMLX_TEST_MODEL` (an absolute snapshot path).
//! When unset the test prints a skip note and returns Ok, so the default
//! `cargo test -p rmlx-server` is unaffected. Each run uses a fresh `RMLX_HOME`
//! tempdir so the SSD cache + index are hermetic and repeatable.
//!
//! ```sh
//! RMLX_TEST_MODEL=/abs/path/to/snapshot \
//! cargo test -p rmlx-server --test ssd_cache_restart -- --ignored --nocapture
//! ```
//!
//! ## Single-MLX-process discipline
//!
//! This test never searches for or kills unrelated processes. It owns the
//! children it starts and only terminates those children. Run it with
//! `--test-threads=1` on a host where no other MLX workload is active.

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
    clippy::duration_suboptimal_units
)]

use std::collections::HashSet;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Test parameters ───────────────────────────────────────────────────────────

/// Must be long enough to form at least one full **256-token block** so the
/// prompt-cache entry has a stable spill key (gemma4 `spill_evicted` skips
/// entries with no full block — see `gemma4/prompt_cache.rs`). At ~1.3
/// tokens/word for English this needs well over ~200 words; the text below is
/// ~500 words → ≳600 tokens → 2+ full 256-token blocks, which also makes the
/// hydrate-vs-cold-prefill TTFT gap clearly measurable.
const PROMPT_A: &str = "You are a meticulous senior systems engineer reviewing a design \
specification. Read the following description carefully and then produce a single concise \
summary paragraph that faithfully preserves every stated constraint. The system under review \
is a Rust-native MLX inference backend built exclusively for Apple Silicon hardware. It loads \
models in the safetensors format using the mlx-community layout, and it does so with no Python \
interpreter present at runtime, which is a hard architectural requirement rather than a \
preference. The single compiled binary serves an OpenAI-compatible HTTP application programming \
interface, accepting text input for every supported model, and additionally accepting image \
input for vision-capable models and audio input for audio-capable models. The backend supports \
the widest possible matrix of weight quantization combined with key-value cache quantization \
that the MLX framework can express, including rotation-based key-value families such as \
TurboQuant, IsoQuant, PlanarQuant, and ParoQuant, none of which any other MLX server currently \
ships. Beyond serving, the backend can convert models between quantization formats and between \
on-disk layouts, performing re-quantization and key-value repacking entirely from MLX input to \
MLX output, never reading or writing the GGUF format because that lane belongs to other tools. \
The system manages a multi-model lifecycle in which models load on demand when first requested \
and unload automatically after a configurable idle timeout, yet it strictly enforces that only \
a single MLX process runs on a given Mac at any moment, because the Apple Silicon Metal context \
is exclusive per process and contention would corrupt inference. To enforce that invariant the \
process acquires an on-disk claim file before touching the graphics processor and releases it \
on shutdown, and competing MLX servers must be unloaded before the claim is taken. All runtime \
state, including structured logs, the metrics database, the rolling summary, and the on-disk \
key-value cache, lives under one root directory resolved at startup, and nothing is ever \
written outside that root during normal operation. Logging flows exclusively through structured \
tracing spans and events so that an operator can reconstruct, from a single run, the fate of \
every token, every model load, every cache operation, and every relevant foreign-function-\
interface call. Be precise, be concise, remain strictly faithful to every constraint above, \
and do not introduce any capability that the specification does not explicitly grant.";

/// A different prompt with no shared prefix with A → its insertion evicts A
/// from the single-slot RAM prompt cache, triggering A's spill.
const PROMPT_B: &str = "Translate the following sentence into formal French and \
then explain, in two sentences, one subtle grammatical choice you made and why \
it preserves the original register: 'The quiet engineer reviewed the proposal \
twice before approving the deployment to the production cluster on Friday.'";

/// Warm TTFT must be below this fraction of cold TTFT to count as a material
/// drop. SSD hydrate still does real work (read `.kvb`, deserialize tensors to
/// the device, re-seed the cache) so we do NOT expect a near-zero TTFT — but a
/// hydrated long prefix should clearly beat a full cold prefill. 0.85 is a
/// deliberately tolerant, honest threshold: the assertion is "warm is clearly
/// faster", not "warm is instant".
const TTFT_DROP_THRESHOLD: f64 = 0.85;
const TTFT_REPEATS: usize = 5;
const QUALIFICATION_LENGTHS: [usize; 3] = [1_024, 4_096, 8_192];
const QUALIFICATION_MIN_TRIALS: usize = 10;
const QUALIFICATION_BOOTSTRAP_SAMPLES: usize = 10_000;

fn median(samples: &mut [f64]) -> f64 {
    assert!(!samples.is_empty(), "median requires at least one sample");
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn median_of(samples: &[f64]) -> f64 {
    let mut owned = samples.to_vec();
    median(&mut owned)
}

/// A canary is accepted only when the median warm run beats the median cold
/// run and the robust spread does not make the result ambiguous. The 15%
/// margin is intentionally conservative for a loaded laptop.
fn uplift_accepted(cold: &[f64], warm: &[f64]) -> bool {
    if cold.len() < 3 || warm.len() < 3 {
        return false;
    }
    let mut c = cold.to_vec();
    let mut w = warm.to_vec();
    let cold_median = median(&mut c);
    let warm_median = median(&mut w);
    warm_median < cold_median * TTFT_DROP_THRESHOLD
        && warm.iter().all(|v| v.is_finite() && *v > 0.0)
        && cold.iter().all(|v| v.is_finite() && *v > 0.0)
}

fn percentile_nearest_rank(samples: &[f64], percentile: f64) -> f64 {
    assert!(
        !samples.is_empty(),
        "percentile requires at least one sample"
    );
    assert!((0.0..=1.0).contains(&percentile));
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

/// Deterministic paired bootstrap of the gate statistic
/// (`1 - median(treatment) / median(cold)`).
///
/// The qualification gate is stated on the cell medians, so the bootstrap must
/// resample paired trial indices and recompute those medians for each
/// resample. Bootstrapping the median of per-trial ratios is a different
/// statistic and can accept or reject a cell the gate itself would not.
///
/// Determinism makes the qualification result reproducible from the printed
/// samples instead of depending on host RNG.
fn paired_bootstrap_lower_bound(cold: &[f64], treatment: &[f64]) -> f64 {
    assert_eq!(cold.len(), treatment.len(), "paired samples must align");
    assert!(!cold.is_empty(), "bootstrap requires at least one pair");
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut bootstrap = Vec::with_capacity(QUALIFICATION_BOOTSTRAP_SAMPLES);
    let mut cold_resample = Vec::with_capacity(cold.len());
    let mut treatment_resample = Vec::with_capacity(treatment.len());
    for _ in 0..QUALIFICATION_BOOTSTRAP_SAMPLES {
        cold_resample.clear();
        treatment_resample.clear();
        for _ in 0..cold.len() {
            // SplitMix64: small, deterministic, and sufficient for bootstrap
            // index selection. Statistical cryptographic quality is irrelevant.
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let idx = z as usize % cold.len();
            cold_resample.push(cold[idx]);
            treatment_resample.push(treatment[idx]);
        }
        let cold_median = median(&mut cold_resample);
        let treatment_median = median(&mut treatment_resample);
        bootstrap.push(1.0 - (treatment_median / cold_median));
    }
    percentile_nearest_rank(&bootstrap, 0.025)
}

#[test]
fn qualification_statistics_enforce_paired_lower_bound_and_tail_gate() {
    let cold = vec![100.0; 10];
    let treatment = vec![70.0; 10];
    assert!(paired_bootstrap_lower_bound(&cold, &treatment) >= 0.10);
    assert!((percentile_nearest_rank(&treatment, 0.95) - 70.0).abs() < f64::EPSILON);

    let noisy = vec![70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 115.0, 115.0];
    assert!(percentile_nearest_rank(&noisy, 0.95) > percentile_nearest_rank(&cold, 0.95) * 1.10);
}

#[test]
fn paired_bootstrap_matches_median_gate_statistic() {
    let cold = vec![100.0, 100.0, 200.0];
    let treatment = vec![70.0, 70.0, 190.0];
    let direct = 1.0 - (median_of(&treatment) / median_of(&cold));
    assert!(
        (direct - 0.30).abs() < 1e-9,
        "direct median-based uplift should be 30%, got {direct}"
    );
    assert!(
        paired_bootstrap_lower_bound(&cold, &treatment) <= direct,
        "bootstrap lower bound must stay anchored to the gate statistic's \
         median-based uplift, not a different per-trial ratio summary"
    );
}

fn qualification_lengths() -> Vec<usize> {
    let Some(raw) = std::env::var_os("RMLX_SSD_QUAL_LENGTHS") else {
        return QUALIFICATION_LENGTHS.to_vec();
    };
    let mut lengths = Vec::new();
    for piece in raw.to_string_lossy().split(',') {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed
            .parse::<usize>()
            .expect("RMLX_SSD_QUAL_LENGTHS must be a comma-separated list of integers");
        assert!(
            QUALIFICATION_LENGTHS.contains(&value),
            "RMLX_SSD_QUAL_LENGTHS may only select from {QUALIFICATION_LENGTHS:?}, got {value}"
        );
        if !lengths.contains(&value) {
            lengths.push(value);
        }
    }
    assert!(
        !lengths.is_empty(),
        "RMLX_SSD_QUAL_LENGTHS selected no qualification cells"
    );
    lengths
}

struct ExactPromptBuilder {
    tokenizer: tokenizers::Tokenizer,
    template: rmlx_server::chat_template::ChatTemplate,
    bos_token: String,
    eos_token: String,
}

impl ExactPromptBuilder {
    fn load(model_dir: &Path) -> Self {
        let tokenizer = rmlx_server::tokenizer_io::load_tokenizer(model_dir)
            .expect("qualification model tokenizer.json must load");
        let template_source = rmlx_server::chat_template::load_template_source(model_dir)
            .expect("qualification model chat_template.jinja must load");
        let template = rmlx_server::chat_template::ChatTemplate::new(template_source)
            .expect("qualification chat template must compile");
        let config = rmlx_server::tokenizer_io::load_tokenizer_config(model_dir)
            .expect("qualification tokenizer_config.json must load");
        Self {
            tokenizer,
            template,
            bos_token: config.bos_token.unwrap_or_default(),
            eos_token: config.eos_token.unwrap_or_default(),
        }
    }

    fn rendered_token_count(&self, content: &str) -> usize {
        let messages = [rmlx_server::chat_template::ChatMessageTpl {
            role: "user",
            content,
            ..rmlx_server::chat_template::ChatMessageTpl::default()
        }];
        let rendered = self
            .template
            .render(
                &messages,
                &rmlx_server::chat_template::RenderOpts {
                    bos_token: &self.bos_token,
                    eos_token: &self.eos_token,
                    add_generation_prompt: true,
                    tools: &[],
                    enable_thinking: None,
                },
            )
            .expect("qualification prompt template must render");
        rmlx_server::tokenizer_io::encode(&self.tokenizer, &rendered.text)
            .expect("qualification rendered prompt must tokenize")
            .len()
    }

    /// Find content whose complete production-rendered request is exactly the
    /// requested tokenizer length. Candidate units are intentionally simple;
    /// the count is always verified through the real snapshot template and
    /// tokenizer, never estimated from bytes or words.
    fn build(&self, target_tokens: usize) -> String {
        const UNITS: [&str; 8] = [
            "qualification ",
            "data ",
            "the ",
            "x ",
            "0 ",
            "test\n",
            "a\n",
            ". ",
        ];
        for unit in UNITS {
            let mut low = 0usize;
            let mut high = target_tokens.saturating_mul(2).max(1);
            while low <= high {
                let mid = low + (high - low) / 2;
                let content = unit.repeat(mid);
                match self.rendered_token_count(&content).cmp(&target_tokens) {
                    std::cmp::Ordering::Equal => return content,
                    std::cmp::Ordering::Less => low = mid + 1,
                    std::cmp::Ordering::Greater => high = mid.saturating_sub(1),
                }
            }
        }
        panic!(
            "could not construct an exact {target_tokens}-token rendered prompt with the model tokenizer"
        );
    }
}

// ── Process / claim helpers ─────────────────────────────────────────────────

/// Resolve the built `rmlx` binary by walking up from the test executable's
/// own path to the `target/<profile>/` directory it lives under
/// (`target/<profile>/deps/<test>-<hash>` → `target/<profile>/rmlx`).
fn rmlx_binary() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    // .../target/<profile>/deps/<testbin> → pop deps, pop testbin
    let profile_dir = exe
        .parent()
        .and_then(|p| {
            if p.file_name().is_some_and(|n| n == "deps") {
                p.parent()
            } else {
                Some(p)
            }
        })
        .expect("resolve target/<profile> dir");
    let bin = profile_dir.join("rmlx");
    assert!(
        bin.exists(),
        "rmlx binary not found at {} — run `cargo build -p rmlx-cli` first",
        bin.display()
    );
    bin
}

/// Spawn `rmlx serve` with the SSD tier on, a single RAM prompt-cache slot, and
/// the given hermetic `RMLX_HOME`. Returns the child handle.
fn spawn_serve(bin: &Path, model: &str, port: u16, rmlx_home: &Path, ssd_gb: &str) -> Child {
    spawn_serve_with_max_ctx(bin, model, port, rmlx_home, ssd_gb, None)
}

fn spawn_serve_with_max_ctx(
    bin: &Path,
    model: &str,
    port: u16,
    rmlx_home: &Path,
    ssd_gb: &str,
    max_ctx: Option<usize>,
) -> Child {
    let mut command = Command::new(bin);
    command
        .arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .arg("--kv-ssd-cache-gb")
        .arg(ssd_gb)
        .arg("--prompt-cache-slots")
        .arg("1")
        .env("RMLX_HOME", rmlx_home);
    if let Some(max_ctx) = max_ctx {
        command.arg("--max-ctx").arg(max_ctx.to_string());
    }
    command
        // info baseline + debug on the SSD spill/hydrate modules so a
        // `--nocapture` run shows the spill drain + hydrate events live.
        .env(
            "RUST_LOG",
            "rmlx=info,rmlx_models::prompt_cache=debug,rmlx_kv_ssd::spill=debug,rmlx_kv_ssd::ssd_tier=debug",
        )
        // Inherit stderr so a `--nocapture` run shows the server's tracing
        // output live (model load, spill drain, hydrate) for debugging.
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn rmlx serve")
}

/// Best-effort kill + reap of a serve child, then clear the owned claim file.
fn teardown(mut child: Child, port: u16) {
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(format!("/tmp/rmlx.{port}.claim"));
}

/// Reserve an ephemeral localhost port for a short-lived child process.
///
/// The listener is dropped before the child starts, so this is not a hard
/// reservation, but it avoids the fixed-port collisions that let an unrelated
/// server satisfy `/health` for this hermetic test.
fn reserve_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("read ephemeral port")
        .port()
}

// ── HTTP helpers (raw HTTP/1.1 over TcpStream — mirrors http_smoke.rs) ───────

/// Send a raw HTTP/1.1 request; return (status_code, body_string).
async fn http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).await?;
    let content_header = if let Some(b) = &body {
        format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        )
    } else {
        String::new()
    };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n{content_header}\r\n{}",
        body.unwrap_or("")
    );
    stream.write_all(request.as_bytes()).await?;
    stream.flush().await?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await?;
    let text = String::from_utf8_lossy(&raw).into_owned();

    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok((status, body))
}

/// Poll `/health` (or any 200-returning route) until the server answers or the
/// deadline passes. Returns true once ready.
async fn wait_ready(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok((status, _)) = http(port, "GET", "/health", None).await {
            if status == 200 {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    false
}

/// POST a chat-completion for `prompt`, return (status, body, TTFT-ms).
///
/// TTFT here is wall-clock from request send to full (non-streamed) response.
/// For a temp=0, max_tokens=1 request the dominant cost is prefill, so this is
/// a faithful proxy for time-to-first-token: cold = full prefill, warm =
/// hydrate-then-resume.
async fn chat_ttft(port: u16, model: &str, prompt: &str) -> std::io::Result<(u16, String, f64)> {
    let payload = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
    })
    .to_string();
    let t0 = Instant::now();
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(&payload)).await?;
    let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((status, body, ttft_ms))
}

fn response_text(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v["choices"][0]["message"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.to_owned())
}

/// Scrape `/metrics` (Prometheus text) and sum every
/// `rmlx_prompt_cache_ssd_hits_total{...}` line's value.
async fn scrape_ssd_hits(port: u16) -> std::io::Result<u64> {
    let (status, body) = http(port, "GET", "/metrics", None).await?;
    assert_eq!(status, 200, "GET /metrics must return 200, body: {body}");
    let mut total = 0u64;
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with("rmlx_prompt_cache_ssd_hits_total") {
            if let Some(val) = line.split_whitespace().last() {
                total += val.parse::<u64>().unwrap_or(0);
            }
        }
    }
    Ok(total)
}

fn prometheus_counter(body: &str, metric: &str) -> u64 {
    body.lines()
        .find_map(|line| {
            let line = line.trim();
            (line.starts_with(metric) && !line.starts_with('#'))
                .then(|| line.split_whitespace().last()?.parse::<u64>().ok())
                .flatten()
        })
        .unwrap_or(0)
}

async fn scrape_hydrate_sum_count(port: u16) -> std::io::Result<(u64, u64)> {
    let (status, body) = http(port, "GET", "/metrics", None).await?;
    assert_eq!(status, 200, "GET /metrics must return 200, body: {body}");
    Ok((
        prometheus_counter(&body, "rmlx_ssd_hydrate_us_sum"),
        prometheus_counter(&body, "rmlx_ssd_hydrate_us_count"),
    ))
}

async fn wait_for_one_hydrate(port: u16, before: (u64, u64)) -> f64 {
    for _ in 0..40 {
        let after = scrape_hydrate_sum_count(port)
            .await
            .expect("scrape hydrate histogram after treatment");
        if after.1 > before.1 {
            assert_eq!(
                after.1 - before.1,
                1,
                "one treatment request must record exactly one SSD hydrate"
            );
            return (after.0 - before.0) as f64 / 1_000.0;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("SSD hydrate histogram did not advance after treatment request");
}

fn prompt_tokens_from_response(body: &str) -> Option<usize> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .pointer("/usage/prompt_tokens")?
        .as_u64()
        .map(|n| n as usize)
}

// ── SSD-index disk verification (rusqlite, dev-dep) ──────────────────────────

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct SsdDiskState {
    kvb_files: usize,
    index_rows: usize,
    indexed_bytes: u64,
    file_bytes: u64,
    seq_lens: Vec<i32>,
    errors: Vec<String>,
}

impl SsdDiskState {
    fn is_exactly_reconciled(&self) -> bool {
        self.index_rows > 0
            && self.kvb_files == self.index_rows
            && self.seq_lens.len() == self.index_rows
            && self.indexed_bytes == self.file_bytes
            && self.errors.is_empty()
    }
}

fn read_persisted_seq_len(path: &Path) -> Result<i32, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| {
            format!(
                "{} is shorter than the 8-byte header prefix",
                path.display()
            )
        })?
        .try_into()
        .expect("eight-byte slice");
    let header_len = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| format!("{} header length does not fit usize", path.display()))?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| format!("{} header length overflow", path.display()))?;
    let header_bytes = bytes.get(8..header_end).ok_or_else(|| {
        format!(
            "{} declares a {header_len}-byte header beyond its file length",
            path.display()
        )
    })?;
    let header: serde_json::Value = serde_json::from_slice(header_bytes)
        .map_err(|e| format!("parse {} safetensors header: {e}", path.display()))?;
    let seq_text = header
        .pointer("/__metadata__/seq_len")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("{} header has no metadata seq_len", path.display()))?;
    seq_text
        .parse::<i32>()
        .map_err(|e| format!("parse {} seq_len {seq_text:?}: {e}", path.display()))
}

/// Reconcile every direct `.kvb` child against every SQLite row, including
/// exact path membership and byte totals. A spill writes the final file before
/// inserting its row, so callers poll this snapshot until it is fully settled.
#[allow(
    clippy::too_many_lines,
    reason = "qualification evidence is easier to audit when the row/file reconciliation protocol remains contiguous"
)]
fn ssd_disk_state(rmlx_home: &Path) -> SsdDiskState {
    let kv_root = rmlx_home.join("cache").join("kv");
    let mut state = SsdDiskState::default();
    let Ok(namespaces) = std::fs::read_dir(&kv_root) else {
        return state;
    };
    for ns_result in namespaces {
        let ns = match ns_result {
            Ok(ns) => ns,
            Err(e) => {
                state.errors.push(format!("read namespace entry: {e}"));
                continue;
            }
        };
        let ns_dir = ns.path();
        if !ns_dir.is_dir() {
            continue;
        }
        let canonical_ns = match std::fs::canonicalize(&ns_dir) {
            Ok(path) => path,
            Err(e) => {
                state
                    .errors
                    .push(format!("canonicalize namespace {}: {e}", ns_dir.display()));
                continue;
            }
        };
        let mut indexed_paths = HashSet::new();
        let db = ns_dir.join("index.db");
        if db.exists() {
            match rusqlite::Connection::open(&db).and_then(|conn| {
                let mut statement = conn.prepare(
                    "SELECT hash, layout_key, path, byte_size FROM kv_blocks ORDER BY hash, layout_key",
                )?;
                let rows = statement.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            }) {
                Ok(rows) => {
                    for (hash, layout_key, path_text, byte_size) in rows {
                        state.index_rows += 1;
                        if byte_size < 0 {
                            state.errors.push(format!(
                                "negative byte_size {byte_size} for {hash}/{layout_key}"
                            ));
                            continue;
                        }
                        let row_path = PathBuf::from(&path_text);
                        if !row_path.is_absolute() || row_path.extension().is_none_or(|e| e != "kvb") {
                            state.errors.push(format!(
                                "indexed path must be absolute .kvb: {}",
                                row_path.display()
                            ));
                            continue;
                        }
                        let canonical_path = match std::fs::canonicalize(&row_path) {
                            Ok(path) => path,
                            Err(e) => {
                                state.errors.push(format!(
                                    "indexed path missing/unreadable {}: {e}",
                                    row_path.display()
                                ));
                                continue;
                            }
                        };
                        if canonical_path.parent() != Some(canonical_ns.as_path()) {
                            state.errors.push(format!(
                                "indexed path is not a direct namespace child: {}",
                                canonical_path.display()
                            ));
                        }
                        if canonical_path.file_stem().and_then(|s| s.to_str()) != Some(hash.as_str()) {
                            state.errors.push(format!(
                                "indexed filename stem does not match hash {hash}: {}",
                                canonical_path.display()
                            ));
                        }
                        if !indexed_paths.insert(canonical_path.clone()) {
                            state.errors.push(format!(
                                "duplicate indexed path: {}",
                                canonical_path.display()
                            ));
                        }
                        let metadata_len = match std::fs::metadata(&canonical_path) {
                            Ok(metadata) => metadata.len(),
                            Err(e) => {
                                state.errors.push(format!(
                                    "metadata failed for {}: {e}",
                                    canonical_path.display()
                                ));
                                continue;
                            }
                        };
                        let indexed_len = byte_size as u64;
                        state.indexed_bytes = state.indexed_bytes.checked_add(indexed_len).expect("indexed SSD byte total overflow");
                        if metadata_len != indexed_len {
                            state.errors.push(format!(
                                "byte_size mismatch for {}: index={indexed_len}, file={metadata_len}",
                                canonical_path.display()
                            ));
                        }
                        match read_persisted_seq_len(&canonical_path) {
                            Ok(seq_len) => state.seq_lens.push(seq_len),
                            Err(e) => state.errors.push(e),
                        }
                    }
                }
                Err(e) => state
                    .errors
                    .push(format!("query SSD index {}: {e}", db.display())),
            }
        }

        match std::fs::read_dir(&ns_dir) {
            Ok(files) => {
                for file_result in files {
                    let file = match file_result {
                        Ok(file) => file,
                        Err(e) => {
                            state.errors.push(format!("read cache file entry: {e}"));
                            continue;
                        }
                    };
                    let path = file.path();
                    if path.extension().is_none_or(|e| e != "kvb") {
                        continue;
                    }
                    state.kvb_files += 1;
                    match std::fs::canonicalize(&path) {
                        Ok(canonical_path) => {
                            if !indexed_paths.contains(&canonical_path) {
                                state.errors.push(format!(
                                    "orphan .kvb without index row: {}",
                                    canonical_path.display()
                                ));
                            }
                            match std::fs::metadata(&canonical_path) {
                                Ok(metadata) => {
                                    state.file_bytes = state
                                        .file_bytes
                                        .checked_add(metadata.len())
                                        .expect("SSD file byte total overflow");
                                }
                                Err(e) => state.errors.push(format!(
                                    "metadata failed for scanned file {}: {e}",
                                    canonical_path.display()
                                )),
                            }
                        }
                        Err(e) => state
                            .errors
                            .push(format!("canonicalize scanned file {}: {e}", path.display())),
                    }
                }
            }
            Err(e) => state
                .errors
                .push(format!("read namespace {}: {e}", ns_dir.display())),
        }
    }
    state.seq_lens.sort_unstable();
    state
}

// ── The test ─────────────────────────────────────────────────────────────────

// The Metal context this test needs belongs to the `rmlx serve` CHILD, not to
// the test binary — this process only speaks HTTP to it — so no source shape
// here can express the device and the `#[ignore]` gate is told rather than
// shown. It stays out of `scripts/run_gpu_tests.sh` on top of that: that runner
// instruments the test process, which dispatches no Metal at all; the test also
// needs `cargo build -p rmlx-cli` first (`cargo test --tests` does not build the
// binary), and spends two 180 s readiness waits. The same spill -> restart -> hydrate chain, over the same two
// prompts, is covered by `make e2e` phase 2a
// (`crates/rmlx-cli/tests/e2e/runner.rs`).
// gpu-test-gate: metal-unscanned  Metal belongs to the spawned serve process.
#[ignore = "integration: requires RMLX_TEST_MODEL + a real rmlx serve process (Metal)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[allow(
    clippy::too_many_lines,
    reason = "one serial end-to-end protocol owns cold baseline, spill, restart, hydrate, parity, metrics, and uplift assertions"
)]
async fn ssd_cache_survives_server_restart() {
    let Ok(model) = std::env::var("RMLX_TEST_MODEL") else {
        eprintln!("RMLX_TEST_MODEL not set — skipping ssd_cache_survives_server_restart");
        return;
    };
    if !PathBuf::from(&model).exists() {
        eprintln!("RMLX_TEST_MODEL={model} does not exist — skipping");
        return;
    }
    // Derive the served model id (snapshot basename) for the chat payload and
    // the namespace glob is path-agnostic anyway.
    let model_id = PathBuf::from(&model)
        .file_name()
        .map_or_else(|| model.clone(), |n| n.to_string_lossy().into_owned());

    let bin = rmlx_binary();
    let home_tmp = tempfile::TempDir::new().expect("tempdir");
    let rmlx_home = home_tmp.path().to_path_buf();

    // Reserve fresh ports for this run so `/health` cannot accidentally attach
    // to a stray server from another session. Phase 1 and 2 reuse the same
    // port because they share one `RMLX_HOME` and the same restart chain.
    let port = reserve_port();

    // Cold baseline: a separate SSD-disabled child, with the same B→A
    // eviction pattern used for each treatment sample. This prevents the
    // baseline from accidentally measuring an SSD hit or a RAM hit.
    let cold_home_tmp = tempfile::TempDir::new().expect("cold tempdir");
    let cold_port = reserve_port();
    let cold_child = spawn_serve(&bin, &model, cold_port, cold_home_tmp.path(), "0");
    assert!(
        wait_ready(cold_port, Duration::from_secs(180)).await,
        "cold baseline server did not become ready within 180s"
    );
    let mut cold_samples = Vec::with_capacity(3);
    for sample in 0..3 {
        let (status_b, _, _) = chat_ttft(cold_port, &model_id, PROMPT_B)
            .await
            .expect("cold B request");
        assert_eq!(
            status_b, 200,
            "cold warm-up B request {sample} must return 200"
        );
        let (status_a, _, ttft) = chat_ttft(cold_port, &model_id, PROMPT_A)
            .await
            .expect("cold A request");
        assert_eq!(status_a, 200, "cold A request {sample} must return 200");
        cold_samples.push(ttft);
    }
    teardown(cold_child, cold_port);
    let cold_median = median(&mut cold_samples);

    // ── Phase 1: populate + spill ───────────────────────────────────────────
    let child = spawn_serve(&bin, &model, port, &rmlx_home, "1");
    assert!(
        wait_ready(port, Duration::from_secs(180)).await,
        "phase-1 server did not become ready within 180s"
    );

    // Populate A, then evict it with B so the persisted row is a full-prefix
    // candidate for the restart treatment.
    let (sa, body_a, ttft_cold) = chat_ttft(port, &model_id, PROMPT_A)
        .await
        .expect("phase-1 prompt A request");
    assert_eq!(sa, 200, "phase-1 prompt A must return 200");
    // Prompt B — different prefix → evicts A from the single RAM slot → A spills.
    let (sb, _bb, _ttft_b) = chat_ttft(port, &model_id, PROMPT_B)
        .await
        .expect("phase-1 prompt B request");
    assert_eq!(sb, 200, "phase-1 prompt B must return 200");

    // The spill is a fire-and-forget drain-thread write; give it time to land.
    // Poll the disk rather than sleep-and-pray.
    let mut disk = SsdDiskState::default();
    for _ in 0..40 {
        disk = ssd_disk_state(&rmlx_home);
        if disk.is_exactly_reconciled() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    eprintln!(
        "phase-1: ttft_cold={ttft_cold:.1}ms  kvb_files={}  index_rows={}",
        disk.kvb_files, disk.index_rows
    );
    assert!(
        disk.is_exactly_reconciled(),
        "SSD files/index must reconcile exactly after eviction under {}/cache/kv: {disk:?}",
        rmlx_home.display(),
    );
    assert!(
        disk.seq_lens.iter().all(|seq| *seq >= 256),
        "persisted seq_len must contain at least one complete indexed block, got {:?}",
        disk.seq_lens
    );

    // ── Restart boundary: kill, clear claim, restart same model + RMLX_HOME ──
    teardown(child, port);

    // ── Phase 2: restart + hydrate ──────────────────────────────────────────
    let child2 = spawn_serve(&bin, &model, port, &rmlx_home, "1");
    if !wait_ready(port, Duration::from_secs(180)).await {
        teardown(child2, port);
        panic!("phase-2 server did not become ready within 180s (see inherited stderr above)");
    }

    // Spilled row must have survived startup prune+evict.
    let after_restart = ssd_disk_state(&rmlx_home);
    assert!(
        after_restart.is_exactly_reconciled(),
        "SSD files/index did not survive restart exactly reconciled: {after_restart:?}"
    );
    assert_eq!(
        after_restart.seq_lens, disk.seq_lens,
        "restart must preserve the exact persisted prefix offset, not a trailing partial"
    );

    // Prompt A again — RAM is empty (fresh process) so this is a RAM miss that
    // must hydrate from the surviving .kvb. Record warm TTFT.
    let mut warm_samples = Vec::with_capacity(TTFT_REPEATS);
    let mut warm_hit_deltas = Vec::with_capacity(TTFT_REPEATS);
    let mut body_a2 = String::new();
    for sample in 0..TTFT_REPEATS {
        // Every treatment sample starts from a RAM miss: B evicts A, then A
        // must be accepted from SSD. Do not let hot-RAM requests enter the
        // uplift median.
        let before = scrape_ssd_hits(port)
            .await
            .expect("scrape SSD hits before sample");
        let (status_b, _, _) = chat_ttft(port, &model_id, PROMPT_B)
            .await
            .expect("warm B eviction request");
        assert_eq!(status_b, 200, "warm-up B request {sample} must return 200");
        let (status, body, ttft) = chat_ttft(port, &model_id, PROMPT_A)
            .await
            .expect("phase-2 prompt A request");
        assert_eq!(
            status, 200,
            "phase-2 prompt A request {sample} must return 200"
        );
        if sample == 0 {
            body_a2 = body;
        }
        let after = scrape_ssd_hits(port)
            .await
            .expect("scrape SSD hits after sample");
        warm_hit_deltas.push(after.saturating_sub(before));
        warm_samples.push(ttft);
    }
    let warm_median = median(&mut warm_samples.clone());
    let ssd_hits = scrape_ssd_hits(port).await.expect("scrape /metrics");

    // Part 4 (step 2 carry-over): scrape /metrics and collect the new
    // SSD-tier Prometheus data. We collect the body here and defer assertions
    // until AFTER teardown so the GPU is freed even on assertion failure.
    let metrics_body = http(port, "GET", "/metrics", None)
        .await
        .expect("GET /metrics for SSD assertions")
        .1;

    eprintln!(
        "phase-2: warm_median={warm_median:.1}ms  ssd_hits={ssd_hits}  \
         (cold_median={cold_median:.1}ms, threshold warm<{:.0}%*cold)",
        TTFT_DROP_THRESHOLD * 100.0
    );

    // Print the SSD section for diagnostics.
    eprintln!("--- /metrics (SSD section) ---");
    for line in metrics_body.lines() {
        let l = line.trim();
        if l.contains("ssd") || l.contains("SSD") {
            eprintln!("  {l}");
        }
    }
    eprintln!("--- end SSD section ---");

    // Teardown BEFORE the asserts so the GPU is freed regardless of outcome.
    teardown(child2, port);

    // ── Cross-restart assertions ────────────────────────────────────────────
    assert!(
        ssd_hits >= TTFT_REPEATS as u64,
        "expected at least one SSD hit per treatment sample, got {ssd_hits}"
    );
    assert_eq!(
        response_text(&body_a),
        response_text(&body_a2),
        "cross-restart output parity must hold for deterministic temperature=0 request"
    );
    assert!(
        warm_hit_deltas.iter().all(|delta| *delta > 0),
        "every measured treatment sample must be an accepted SSD reuse, deltas={warm_hit_deltas:?}"
    );
    assert!(
        uplift_accepted(&cold_samples, &warm_samples),
        "median warm TTFT ({warm_median:.1}ms) must be < {:.0}% of cold ({cold_median:.1}ms)",
        TTFT_DROP_THRESHOLD * 100.0
    );

    // ── Part 4: SSD-observability Prometheus assertions ───────
    // All assertions use `metrics_body` collected before teardown above.

    // rmlx_ssd_bytes_used > 0: startup_maintenance fires call_ssd_bytes_used_hook
    // which populates the gauge from the on-disk footprint when the model loads.
    let bytes_used: u64 = metrics_body
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| l.trim_start().starts_with("rmlx_ssd_bytes_used"))
        .filter_map(|l| l.split_whitespace().last()?.parse::<u64>().ok())
        .sum();
    assert!(
        bytes_used > 0,
        "rmlx_ssd_bytes_used must be > 0 after a spill, got {bytes_used}"
    );

    // rmlx_ssd_spill_us TYPE line: the spill happened on Phase 1's server.
    // Phase 2 only hydrates, so count may be 0. Assert TYPE presence (always
    // emitted when the SSD tier is active) to prove histogram infra is wired.
    assert!(
        metrics_body.contains("# TYPE rmlx_ssd_spill_us histogram"),
        "expected '# TYPE rmlx_ssd_spill_us histogram' in /metrics (always emitted when tier is active)"
    );

    // rmlx_ssd_hydrate_us_bucket: Phase 2 hydrated Prompt A — at least one
    // bucket must have count >= 1.
    let hydrate_bucket_present = metrics_body
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .any(|l| {
            let l = l.trim_start();
            if !l.starts_with("rmlx_ssd_hydrate_us_bucket") {
                return false;
            }
            l.split_whitespace()
                .last()
                .and_then(|v| v.parse::<u64>().ok())
                .is_some_and(|n| n >= 1)
        });
    assert!(
        hydrate_bucket_present,
        "expected at least one rmlx_ssd_hydrate_us_bucket with count >= 1 in /metrics"
    );

    // rmlx_ssd_evict_total TYPE line: always emitted when tier is active;
    // per-namespace rows have count=0 when no evictions fired on this instance.
    assert!(
        metrics_body.contains("# TYPE rmlx_ssd_evict_total counter"),
        "rmlx_ssd_evict_total TYPE line must be present in /metrics when SSD tier is active"
    );
}

// gpu-test-gate: metal-unscanned  Metal belongs to the spawned serve process.
#[ignore = "qualification: requires RMLX_TEST_MODEL + long real-model Metal runs"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[allow(
    clippy::too_many_lines,
    reason = "one serial qualification protocol owns exact token cells, paired trials, restart hydrate, statistics, and disk reconciliation"
)]
async fn ssd_cache_restart_qualification_matrix() {
    let Ok(model) = std::env::var("RMLX_TEST_MODEL") else {
        eprintln!("RMLX_TEST_MODEL not set — skipping SSD qualification matrix");
        return;
    };
    let model_path = PathBuf::from(&model);
    if !model_path.exists() {
        eprintln!("RMLX_TEST_MODEL={model} does not exist — skipping");
        return;
    }
    let trials =
        std::env::var("RMLX_SSD_QUAL_TRIALS")
            .ok()
            .map_or(QUALIFICATION_MIN_TRIALS, |value| {
                value
                    .parse::<usize>()
                    .expect("RMLX_SSD_QUAL_TRIALS must be an integer")
            });
    assert!(
        trials >= QUALIFICATION_MIN_TRIALS,
        "SSD qualification requires at least {QUALIFICATION_MIN_TRIALS} paired trials per length"
    );

    let model_id = model_path
        .file_name()
        .map_or_else(|| model.clone(), |name| name.to_string_lossy().into_owned());
    let prompt_builder = ExactPromptBuilder::load(&model_path);
    let bin = rmlx_binary();
    let report_dir = std::env::var_os("RMLX_SSD_QUAL_REPORT_DIR")
        .map_or_else(|| PathBuf::from("target/ssd-qualification"), PathBuf::from);
    std::fs::create_dir_all(&report_dir).expect("create SSD qualification report directory");

    for target_tokens in qualification_lengths() {
        let prompt_a = prompt_builder.build(target_tokens);
        assert_eq!(
            prompt_builder.rendered_token_count(&prompt_a),
            target_tokens,
            "qualification prompt builder must hit the exact rendered tokenizer length"
        );

        let mut cold_samples = Vec::with_capacity(trials);
        let mut treatment_samples = Vec::with_capacity(trials);
        let mut hydrate_samples = Vec::with_capacity(trials);
        let mut hit_deltas = Vec::with_capacity(trials);
        let mut expected_text: Option<String> = None;
        let mut final_disk = SsdDiskState::default();

        for trial in 0..trials {
            // One true pair = one cold arm followed immediately by one
            // treatment arm, each in its own fresh hermetic HOME. The pair
            // index is therefore time-aligned and the paired bootstrap really
            // resamples paired observations rather than two time-separated
            // blocks.
            let cold_home = tempfile::TempDir::new().expect("cold qualification tempdir");
            let cold_port = reserve_port();
            let cold_child = spawn_serve_with_max_ctx(
                &bin,
                &model,
                cold_port,
                cold_home.path(),
                "0",
                Some(target_tokens),
            );
            assert!(
                wait_ready(cold_port, Duration::from_secs(180)).await,
                "{target_tokens}-token cold server did not become ready on trial {trial}"
            );
            let (status_b, _, _) = chat_ttft(cold_port, &model_id, PROMPT_B)
                .await
                .expect("cold qualification B request");
            assert_eq!(status_b, 200, "cold B trial {trial} must return 200");
            let (status_a, body_a, ttft) = chat_ttft(cold_port, &model_id, &prompt_a)
                .await
                .expect("cold qualification A request");
            assert_eq!(status_a, 200, "cold A trial {trial} must return 200");
            assert_eq!(
                prompt_tokens_from_response(&body_a),
                Some(target_tokens),
                "server usage must confirm the exact tokenizer-token cell"
            );
            let text = response_text(&body_a);
            if let Some(expected) = &expected_text {
                assert_eq!(&text, expected, "cold output changed at trial {trial}");
            } else {
                expected_text = Some(text);
            }
            cold_samples.push(ttft);
            teardown(cold_child, cold_port);

            let treatment_home = tempfile::TempDir::new().expect("treatment qualification tempdir");
            let populate_port = reserve_port();
            let populate_child = spawn_serve_with_max_ctx(
                &bin,
                &model,
                populate_port,
                treatment_home.path(),
                "1",
                Some(target_tokens),
            );
            assert!(
                wait_ready(populate_port, Duration::from_secs(180)).await,
                "{target_tokens}-token populate server did not become ready on trial {trial}"
            );
            let (populate_status, populate_body, _) =
                chat_ttft(populate_port, &model_id, &prompt_a)
                    .await
                    .expect("qualification populate A request");
            assert_eq!(populate_status, 200, "populate A must return 200");
            assert_eq!(
                prompt_tokens_from_response(&populate_body),
                Some(target_tokens),
                "populate request must remain in the exact tokenizer-token cell"
            );
            assert_eq!(
                response_text(&populate_body),
                expected_text.as_deref().unwrap_or_default(),
                "populate output must match the SSD-disabled cold output"
            );
            let (evict_status, _, _) = chat_ttft(populate_port, &model_id, PROMPT_B)
                .await
                .expect("qualification populate B eviction request");
            assert_eq!(evict_status, 200, "populate B must return 200");

            let mut populated_disk = SsdDiskState::default();
            for _ in 0..80 {
                populated_disk = ssd_disk_state(treatment_home.path());
                if populated_disk.is_exactly_reconciled() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            assert!(
                populated_disk.is_exactly_reconciled(),
                "{target_tokens}-token populate disk/index did not reconcile on trial {trial}: \
                 {populated_disk:?}"
            );
            assert!(
                populated_disk
                    .seq_lens
                    .iter()
                    .any(|seq_len| *seq_len as usize == target_tokens),
                "persisted metadata must contain the exact {target_tokens}-token prompt, got {:?}",
                populated_disk.seq_lens
            );
            teardown(populate_child, populate_port);

            let treatment_port = reserve_port();
            let treatment_child = spawn_serve_with_max_ctx(
                &bin,
                &model,
                treatment_port,
                treatment_home.path(),
                "1",
                Some(target_tokens),
            );
            assert!(
                wait_ready(treatment_port, Duration::from_secs(180)).await,
                "{target_tokens}-token treatment server did not become ready on trial {trial}"
            );
            let restarted_disk = ssd_disk_state(treatment_home.path());
            assert!(
                restarted_disk.is_exactly_reconciled(),
                "{target_tokens}-token restart disk/index did not reconcile on trial {trial}: \
                 {restarted_disk:?}"
            );

            let (status_b, _, _) = chat_ttft(treatment_port, &model_id, PROMPT_B)
                .await
                .expect("qualification treatment B request");
            assert_eq!(status_b, 200, "treatment B trial {trial} must return 200");
            let hits_before = scrape_ssd_hits(treatment_port)
                .await
                .expect("scrape SSD hits before qualification treatment");
            let hydrate_before = scrape_hydrate_sum_count(treatment_port)
                .await
                .expect("scrape hydrate histogram before qualification treatment");
            let (status_a, body_a, ttft) = chat_ttft(treatment_port, &model_id, &prompt_a)
                .await
                .expect("qualification treatment A request");
            assert_eq!(status_a, 200, "treatment A trial {trial} must return 200");
            assert_eq!(
                prompt_tokens_from_response(&body_a),
                Some(target_tokens),
                "treatment request must remain in the exact tokenizer-token cell"
            );
            assert_eq!(
                response_text(&body_a),
                expected_text.as_deref().unwrap_or_default(),
                "SSD treatment output changed at trial {trial}"
            );
            let hits_after = scrape_ssd_hits(treatment_port)
                .await
                .expect("scrape SSD hits after qualification treatment");
            hit_deltas.push(hits_after.saturating_sub(hits_before));
            hydrate_samples.push(wait_for_one_hydrate(treatment_port, hydrate_before).await);
            treatment_samples.push(ttft);

            // Allow the evictor spill to settle, then require exact path and
            // byte reconciliation with no orphan files for this trial's HOME.
            final_disk = SsdDiskState::default();
            for _ in 0..80 {
                final_disk = ssd_disk_state(treatment_home.path());
                if final_disk.is_exactly_reconciled() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            assert!(
                final_disk.is_exactly_reconciled(),
                "{target_tokens}-token final disk/index must have exact bytes and no orphan \
                 blocks on trial {trial}: {final_disk:?}"
            );
            teardown(treatment_child, treatment_port);
        }

        let cold_median = median(&mut cold_samples.clone());
        let treatment_median = median(&mut treatment_samples.clone());
        let cold_p95 = percentile_nearest_rank(&cold_samples, 0.95);
        let treatment_p95 = percentile_nearest_rank(&treatment_samples, 0.95);
        let hydrate_p95 = percentile_nearest_rank(&hydrate_samples, 0.95);
        let ci_lower = paired_bootstrap_lower_bound(&cold_samples, &treatment_samples);
        eprintln!(
            "SSD qualification {target_tokens} tokens ({trials} pairs): cold={cold_samples:?} treatment={treatment_samples:?} hydrate_ms={hydrate_samples:?}"
        );
        eprintln!(
            "SSD qualification {target_tokens}: cold_median={cold_median:.1}ms treatment_median={treatment_median:.1}ms cold_p95={cold_p95:.1}ms treatment_p95={treatment_p95:.1}ms hydrate_p95={hydrate_p95:.1}ms bootstrap_ci95_lower_uplift={:.1}%",
            ci_lower * 100.0
        );

        let accepted_hits_every_trial = hit_deltas.iter().all(|delta| *delta > 0);
        let median_pass = treatment_median <= cold_median * 0.80;
        let ci_pass = ci_lower >= 0.10;
        let treatment_p95_pass = treatment_p95 <= cold_p95 * 1.10;
        let hydrate_p95_pass = hydrate_p95 < cold_median;
        let disk_pass = final_disk.is_exactly_reconciled();
        let report = serde_json::json!({
            "model": &model_id,
            "target_prompt_tokens": target_tokens,
            "paired_trials": trials,
            "cold_ttft_ms": &cold_samples,
            "treatment_ttft_ms": &treatment_samples,
            "hydrate_ms": &hydrate_samples,
            "ssd_hit_deltas": &hit_deltas,
            "cold_median_ms": cold_median,
            "treatment_median_ms": treatment_median,
            "cold_p95_ms": cold_p95,
            "treatment_p95_ms": treatment_p95,
            "hydrate_p95_ms": hydrate_p95,
            "paired_bootstrap_ci95_lower_uplift": ci_lower,
            "gates": {
                "identical_output": true,
                "accepted_ssd_hit_every_treatment": accepted_hits_every_trial,
                "treatment_median_le_80pct_cold": median_pass,
                "bootstrap_ci95_lower_ge_10pct": ci_pass,
                "treatment_p95_le_110pct_cold_p95": treatment_p95_pass,
                "hydrate_p95_below_cold_median": hydrate_p95_pass,
                "disk_index_exactly_reconciled": disk_pass,
            },
            "disk": {
                "kvb_files": final_disk.kvb_files,
                "index_rows": final_disk.index_rows,
                "indexed_bytes": final_disk.indexed_bytes,
                "file_bytes": final_disk.file_bytes,
                "seq_lens": &final_disk.seq_lens,
                "errors": &final_disk.errors,
            },
        });
        let report_path = report_dir.join(format!("{model_id}-{target_tokens}.json"));
        std::fs::write(
            &report_path,
            serde_json::to_vec_pretty(&report).expect("serialize SSD qualification report"),
        )
        .unwrap_or_else(|e| panic!("write {}: {e}", report_path.display()));
        eprintln!("SSD qualification report: {}", report_path.display());

        assert!(
            accepted_hits_every_trial,
            "every {target_tokens}-token treatment must be an accepted SSD hit: {hit_deltas:?}"
        );
        assert!(
            median_pass,
            "{target_tokens}-token treatment median must be <=80% of cold"
        );
        assert!(
            ci_pass,
            "{target_tokens}-token paired-bootstrap 95% CI lower uplift must be >=10%, got {:.1}%",
            ci_lower * 100.0
        );
        assert!(
            treatment_p95_pass,
            "{target_tokens}-token treatment p95 must be <=110% of cold p95"
        );
        assert!(
            hydrate_p95_pass,
            "{target_tokens}-token hydrate p95 must be below cold-prefill median"
        );
        assert!(
            disk_pass,
            "{target_tokens}-token final disk/index must have exact bytes and no orphan blocks: {final_disk:?}"
        );
    }
}
