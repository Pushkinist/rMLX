//! E2E harness runner: parse the manifest, drive the real `rmlx` binary, assert.
//!
//! Two execution modes per `[[case]]`:
//!
//! * **CLI mode** (`cli = [...]`): subprocess `CARGO_BIN_EXE_rmlx` with the
//!   given args, assert on exit code / stdout. No HTTP, no claim needed for the
//!   low-GPU surface checks; the `--probe-smoke` / `healthcheck --full` cases
//!   load MLX and are claim-gated.
//! * **Serve mode** (`serve_flags = [...]` + `request`): spawn `rmlx serve`,
//!   wait for `/health`, send a named request fixture over raw HTTP/1.1, assert
//!   on the response (golden / coherent / niah / cosine / thinking).
//!
//! Single-MLX discipline (hard rule 8): every model-touching case runs a
//! `pkill`/claim-rm preflight first and tears down the spawned `rmlx serve`
//! child after. The entry point pins `--test-threads=1`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::duration_suboptimal_units,
    clippy::format_push_string,
    clippy::match_same_arms
)]
// Each item is used by only a subset of the harness; the module is included via
// `#[path]` into one test binary, so unreachable_pub / dead_code fire per the
// standard `tests/common` pattern. Allow them here.
#![allow(dead_code, unreachable_pub)]

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::report::{CaseResult, Report, Verdict};

// ── Manifest model ──────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Manifest {
    #[serde(default)]
    pub case: Vec<Case>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Case {
    pub id: String,
    pub feature: String,
    pub subfeature: String,
    /// Env-var KEY (e.g. `BONSAI`) resolving to a snapshot path. Absent for
    /// model-free CLI surface cases.
    #[serde(default)]
    pub model: Option<String>,
    /// CLI subcommand + flags. Mutually exclusive with `serve_flags`.
    #[serde(default)]
    pub cli: Option<Vec<String>>,
    /// `rmlx serve` flags. Pairs with `request`.
    #[serde(default)]
    pub serve_flags: Option<Vec<String>>,
    /// Named request fixture (resolved by `request_fixture`).
    #[serde(default)]
    pub request: Option<String>,
    /// KV preset string passed to `--kv-quant` for serve cases.
    #[serde(default)]
    pub kv_quant: Option<String>,
    /// Per-side K codec tag for `--cache-type-k` (compose form). Used for codecs
    /// whose Display form does not round-trip through `--kv-quant` FromStr (e.g.
    /// RotK → `--ctk rot_k --ctv q4_g64`). Mutually exclusive with `kv_quant`.
    #[serde(default)]
    pub ctk: Option<String>,
    /// Per-side V codec tag for `--cache-type-v`.
    #[serde(default)]
    pub ctv: Option<String>,
    /// Speculative-decoding drafter spec (path/slug/alias, resolved like
    /// `model`). Pairs with `draft_kind`. Used by the `spec_decode` assert.
    #[serde(default)]
    pub draft_model: Option<String>,
    /// Drafter architecture family for `--draft-kind` (`mtp` | `dflash` | `eagle3`).
    #[serde(default)]
    pub draft_kind: Option<String>,
    pub assert: Assert,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Assert {
    pub kind: String,
    /// Free-form expectation; meaning depends on `kind`.
    #[serde(default)]
    pub expect: Option<String>,
    /// Cosine threshold for `cosine_vs_bf16` (default 0.99).
    #[serde(default)]
    pub min_cosine: Option<f64>,
}

// ── Model resolution ────────────────────────────────────────────────────────

/// Resolve a manifest `model` spec to a snapshot directory.
///
/// A spec is data, not code — adding a model NEVER requires editing this
/// function. A spec may be any of:
///
/// 1. **Path** — contains `/` and exists on disk → used verbatim (absolute or
///    relative to cwd). Lets a manifest row or ad-hoc run point at any snapshot.
/// 2. **Slug** — a snapshot directory name (e.g. `mlx-community__…-31b…`) →
///    joined under `RMLX_O_MODELS_ROOT` (default `./models`).
/// 3. **Alias** — a short CLAUDE.md test-target shorthand (`BONSAI`,
///    `GEMMA4_E4B`, …) → mapped to its slug. The alias table is FROZEN to the
///    canonical test targets; new models come in as a slug/path, not a new arm.
///
/// Runtime override (any spec, zero file edit): set
/// `RMLX_E2E_MODEL_<SPEC>` or `RMLX_TEST_MODEL_<SPEC>` (spec upper-cased,
/// non-alphanumerics → `_`) to an absolute path. This is how CI / a dev redirects
/// a model — including one with no manifest row — without touching the repo.
///
/// Returns `None` (case skips with a clear reason) when nothing resolves to an
/// existing directory.
fn resolve_model(spec: &str) -> Option<PathBuf> {
    // A spec must carry at least one alphanumeric: a blank or punctuation-only
    // spec would sanitize to a degenerate env key (`RMLX_E2E_MODEL_` or
    // `..._`) on which distinct junk specs collide. Reject up front.
    if !spec.chars().any(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    // 1. Per-spec env override — runtime redirect, wins over everything.
    let env_key: String = spec
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    for var in [
        format!("RMLX_E2E_MODEL_{env_key}"),
        format!("RMLX_TEST_MODEL_{env_key}"),
    ] {
        if let Ok(p) = std::env::var(&var) {
            let pb = PathBuf::from(&p);
            if pb.exists() {
                return Some(pb);
            }
        }
    }
    // 2. Spec is already a path → use as-is when it exists.
    if spec.contains('/') {
        let pb = PathBuf::from(spec);
        if pb.exists() {
            return Some(pb);
        }
    }
    // 3. Resolve a slug under the Open Models root. Canonical aliases expand to
    //    their slug; any other spec is treated as a literal snapshot dir name.
    let slug = canonical_alias(spec).unwrap_or(spec);
    let root = std::env::var("RMLX_O_MODELS_ROOT").unwrap_or_else(|_| "models".to_owned());
    let pb = PathBuf::from(root).join(slug);
    pb.exists().then_some(pb)
}

/// Short shorthands for the CLAUDE.md test-target snapshots, so the common rows
/// stay readable (`model = "BONSAI"`). FROZEN: new models reference their slug
/// or path directly in the manifest — do not grow this for coverage.
fn canonical_alias(spec: &str) -> Option<&'static str> {
    Some(match spec.to_uppercase().as_str() {
        "BONSAI" => "prism-ml__Ternary-Bonsai-8B-mlx-2bit",
        "GEMMA4_E2B" => "mlx-community__gemma-4-e2b-it-mxfp8",
        "GEMMA4_E4B" => "mlx-community__gemma-4-e4b-it-mxfp8",
        "QWEN36" => "mlx-community__Qwen3.6-35B-A3B-8bit",
        _ => return None,
    })
}

fn rmlx_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rmlx"))
}

/// Process-scoped temp HOME for the whole run.
///
/// INTENTIONAL: this is a per-process temp dir under `std::env::temp_dir()`,
/// NOT the checked-out workspace `.rmlx/`. The harness spawns the real `rmlx`
/// binary with `RMLX_HOME` pinned here so all runtime state (logs, metrics)
/// stays out of the repo. The `e2e/` report subtree under it is the one
/// artifact the entry point preserves; the transient runtime subtrees are
/// best-effort removed at end of the run by `cleanup_e2e_home`.
fn e2e_home() -> PathBuf {
    let home = std::env::temp_dir().join(format!("rmlx_e2e_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&home);
    home
}

/// Best-effort removal of the transient runtime subtrees under the temp HOME
/// (`logs/`, `metrics/`, `cache/`, `tmp/`) at end of the run. The `e2e/` report
/// subtree is deliberately preserved — it is the harness artifact. Errors are
/// ignored; the OS reaps the rest of the temp dir eventually.
fn cleanup_e2e_home() {
    let home = std::env::temp_dir().join(format!("rmlx_e2e_{}", std::process::id()));
    for sub in ["logs", "metrics", "cache", "tmp"] {
        let _ = std::fs::remove_dir_all(home.join(sub));
    }
}

// ── Single-MLX claim discipline (hard rule 8) ───────────────────────────────

/// Kill any stray `rmlx serve` / mlx processes and remove the claim file.
/// Best-effort; failures are logged, not fatal (the spawn will fail loudly if
/// the GPU is genuinely held).
fn claim_preflight() {
    for pat in ["rmlx serve", "mlx_lm", "paroquant", "omlx"] {
        let _ = Command::new("pkill").args(["-f", pat]).output();
    }
    std::thread::sleep(Duration::from_millis(800));
    // Glob-remove EVERY stale claim. Serve binds in 18000..22000 (quant) and
    // 17000..17500 (bf16 ref), so the old fixed `[62265, 8080]` list left a
    // stale claim at the actual port alive. Best-effort: ignore read/remove
    // errors (the spawn fails loudly if the GPU is genuinely held).
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("rmlx.") && name.ends_with(".claim") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

// ── HTTP (raw, std-only — mirrors http_smoke.rs over std::net) ───────────────

fn http_post(port: u16, path: &str, body: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(600)))?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp)?;
    let text = String::from_utf8_lossy(&resp).into_owned();
    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    Ok((status, text[body_start..].to_owned()))
}

fn http_get(port: u16, path: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes())?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp)?;
    let text = String::from_utf8_lossy(&resp).into_owned();
    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    Ok((status, text[body_start..].to_owned()))
}

/// A spawned `rmlx serve` child + its port. `Drop` kills the child.
struct ServeGuard {
    child: Child,
    port: u16,
    home: PathBuf,
}

impl Drop for ServeGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(format!("/tmp/rmlx.{}.claim", self.port));
    }
}

/// Spawn `rmlx serve --model <path> --port <port> <extra serve_flags>` and
/// block until `/health` is green (or timeout). Returns the guard (with the
/// model loaded lazily on first request).
fn spawn_serve(
    model: &std::path::Path,
    port: u16,
    kv_quant: Option<&str>,
    extra: &[String],
    home: &std::path::Path,
) -> Result<ServeGuard, String> {
    let mut cmd = Command::new(rmlx_bin());
    cmd.arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .env("RMLX_HOME", home)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(kv) = kv_quant {
        cmd.arg("--kv-quant").arg(kv);
    }
    for f in extra {
        cmd.arg(f);
    }
    let child = cmd.spawn().map_err(|e| format!("spawn rmlx serve: {e}"))?;
    let mut guard = ServeGuard {
        child,
        port,
        home: home.to_path_buf(),
    };

    // Wait for /health. Server binds before model load, so health goes green
    // quickly; we then send a tiny warmup request to force the lazy load.
    //
    // Capture the LAST non-green observation so the timeout error distinguishes
    // "never responded" (connection refused) from "responding but broken"
    // (e.g. 500, or 200 with a malformed body that parses to status 0).
    let deadline = Instant::now() + Duration::from_secs(120);
    // Reassigned on every poll before the timeout branch reads it; the initial
    // value only applies if the first poll panics, which cannot happen here.
    #[allow(unused_assignments)]
    let mut last_seen = "no response yet".to_owned();
    loop {
        if let Some(status) = guard.child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!("rmlx serve exited early: {status}"));
        }
        match http_get(port, "/health") {
            Ok((200, body)) => {
                if body.contains("\"ok\":true") || body.contains("\"ok\": true") {
                    return Ok(guard);
                }
                last_seen = format!("status 200 but body not ok: {}", trunc(&body));
            }
            Ok((status, body)) => {
                last_seen = format!("status {status}: {}", trunc(&body));
            }
            Err(e) => {
                last_seen = format!("connect/read error: {e}");
            }
        }
        if Instant::now() > deadline {
            return Err(format!(
                "rmlx serve /health never went green within 120s (last: {last_seen})"
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

// ── Request fixtures ────────────────────────────────────────────────────────

/// The model id the server registers is the snapshot dir basename.
fn model_id(model: &std::path::Path) -> String {
    model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_owned()
}

/// Build the JSON body for a named request fixture. All fixtures pin temp=0 +
/// seed=0 for determinism.
///
/// Bonsai is a thinking model: with `enable_thinking` on (its default) the
/// final-answer `content` stays empty until the thinking budget is spent. The
/// deterministic text / quant fixtures therefore set `enable_thinking=false`
/// so the model emits a direct answer (the dedicated `thinking` fixture keeps
/// it on and asserts on `reasoning_content`). See MEMORY: Bonsai/Qwen3
/// thinking-model output lives in `reasoning_content`.
/// File-path URL to the bundled 224×224 solid-red PNG test fixture. The server's
/// `image_io::load_image` accepts an absolute file path; a committed binary asset
/// avoids any base64 transport corruption. Proves the vision path decodes a real
/// image and the model reads its dominant colour.
const RED_PNG_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/e2e/fixtures/vtest_red.png"
);

fn request_fixture(name: &str, model_id: &str) -> serde_json::Value {
    let base_msg = "What is the capital of France? Answer in one short sentence.";
    match name {
        // Vision: a solid-red image + a one-word color question. PASS = the
        // model reads the image and answers with the colour (see assert_image).
        "image_color" => serde_json::json!({
            "model": model_id,
            "max_tokens": 40, "temperature": 0.0, "seed": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is the main color of this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": RED_PNG_PATH}}
            ]}]
        }),
        // Tool-calling: a weather tool + a prompt that should trigger it. PASS =
        // finish_reason "tool_calls" + a get_weather call (see assert_tool_call).
        "tool_weather" => serde_json::json!({
            "model": model_id,
            "max_tokens": 400, "temperature": 0.0, "seed": 0,
            "messages": [{"role": "user", "content":
                "What is the weather in Paris right now? Use the tool."}],
            "tool_choice": "auto",
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {"type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]}
            }}]
        }),
        "chat_basic" => serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": base_msg}],
            "max_tokens": 32, "temperature": 0.0, "seed": 0, "enable_thinking": false
        }),
        "chat_stream" => serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": base_msg}],
            "max_tokens": 32, "temperature": 0.0, "seed": 0, "stream": true,
            "enable_thinking": false
        }),
        "chat_logprobs" => serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": base_msg}],
            "max_tokens": 16, "temperature": 0.0, "seed": 0,
            "logprobs": true, "top_logprobs": 20, "enable_thinking": false
        }),
        "anthropic_basic" => serde_json::json!({
            "model": model_id, "max_tokens": 32,
            "messages": [{"role": "user", "content": base_msg}],
            "temperature": 0.0
        }),
        "multi_turn" => serde_json::json!({
            "model": model_id, "max_tokens": 32, "temperature": 0.0, "seed": 0,
            "enable_thinking": false,
            "messages": [
                {"role": "user", "content": "My favorite color is teal. Remember it."},
                {"role": "assistant", "content": "Got it, your favorite color is teal."},
                {"role": "user", "content": "What is my favorite color? One word."}
            ]
        }),
        "thinking" => serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content":
                "What is 17 times 4? Think step by step, then give the answer."}],
            "max_tokens": 256, "temperature": 0.0, "seed": 0,
            "enable_thinking": true, "thinking_budget": 64
        }),
        "niah" => niah_request(model_id),
        _ => serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": base_msg}],
            "max_tokens": 32, "temperature": 0.0, "seed": 0, "enable_thinking": false
        }),
    }
}

/// 8k-ish needle-in-a-haystack request. Mirrors the algorithm in
/// `niah_long_context.rs` but as a chat request over HTTP.
const NIAH_NEEDLE: &str = "AX7-PURPLE-FOX-9421";

fn niah_request(model_id: &str) -> serde_json::Value {
    let filler = "The grass is green and the sun is yellow. Mountains rise tall above \
        the silent valley below. Rivers flow steadily toward the open sea. ";
    // ~8k tokens: filler is ~30 tokens; 250 reps ≈ 7500 tokens. Needle at ~50%.
    let mut hay = String::with_capacity(filler.len() * 260);
    for i in 0..260 {
        if i == 130 {
            hay.push_str(&format!(
                "Important note: the secret code is {NIAH_NEEDLE}. Remember this code. "
            ));
        }
        hay.push_str(filler);
    }
    let prompt = format!(
        "Read the document and find the secret alphanumeric code, then repeat it exactly.\n\n\
         Document:\n{hay}\n\nQuestion: What is the secret code? Answer with only the code."
    );
    serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32, "temperature": 0.0, "seed": 0, "enable_thinking": false
    })
}

// ── Response extraction ──────────────────────────────────────────────────────

fn openai_content(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_owned)
}

fn openai_reasoning(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .map(str::to_owned)
}

fn anthropic_text(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["content"][0]["text"].as_str().map(str::to_owned)
}

/// Collect the concatenated `content` deltas from an SSE stream body.
fn sse_collect_content(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
                out.push_str(s);
            }
        }
    }
    out
}

/// One decoded position from an OpenAI logprobs response: the chosen token's
/// byte-identity (used to detect token-id divergence between two runs) plus the
/// `top_logprobs` distribution as `(surface, logprob)` pairs.
#[derive(Clone, Debug)]
struct LogprobStep {
    /// Byte sequence of the chosen token surface — the per-position token
    /// identity. Two runs agree at a position iff these bytes match.
    chosen_bytes: Vec<u8>,
    /// `top_logprobs` distribution: token surface → logprob.
    top: std::collections::BTreeMap<String, f64>,
}

/// Parse the per-token logprob steps from an OpenAI chat-completions response.
/// Requires `top_logprobs` to be present (the cosine fixture requests it).
fn openai_logprob_steps(body: &str) -> Option<Vec<LogprobStep>> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let content = v["choices"][0]["logprobs"]["content"].as_array()?;
    let mut out = Vec::with_capacity(content.len());
    for tok in content {
        // Prefer the explicit `bytes` array; fall back to the token surface.
        let chosen_bytes = tok["bytes"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| b.as_u64().map(|n| n as u8))
                    .collect()
            })
            .or_else(|| tok["token"].as_str().map(|s| s.as_bytes().to_vec()))
            .unwrap_or_default();
        let mut top = std::collections::BTreeMap::new();
        if let Some(alts) = tok["top_logprobs"].as_array() {
            for alt in alts {
                if let (Some(t), Some(lp)) = (alt["token"].as_str(), alt["logprob"].as_f64()) {
                    top.insert(t.to_owned(), lp);
                }
            }
        }
        out.push(LogprobStep { chosen_bytes, top });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Outcome of the strengthened cosine comparison.
enum CosineOutcome {
    /// Token-id sequences agreed for the compared positions; mean cosine of the
    /// per-position top-k logprob distributions.
    Ok { mean_cos: f64, positions: usize },
    /// The chosen token-id diverged at `pos` (0-based) within the first
    /// `divergence_window` tokens — the real fidelity-failure signal.
    Diverged { pos: usize },
    /// No comparable positions (empty / no overlapping top-k).
    Empty,
}

/// Number of leading tokens within which a chosen-token-id divergence is
/// treated as the real cosine failure signal.
const COSINE_DIVERGENCE_WINDOW: usize = 8;

/// Compare two logprob-step sequences (bf16 reference vs quant codec).
///
/// Token-id divergence within the first `COSINE_DIVERGENCE_WINDOW` positions is
/// the real failure signal — after a divergence the two runs describe DIFFERENT
/// tokens, so comparing their scalars is meaningless. While the chosen token
/// ids agree, score the per-position top-k logprob DISTRIBUTIONS by cosine over
/// the union of surfaces (missing surfaces contribute 0), and average.
fn compare_logprob_steps(reference: &[LogprobStep], quant: &[LogprobStep]) -> CosineOutcome {
    let n = reference.len().min(quant.len());
    if n == 0 {
        return CosineOutcome::Empty;
    }
    let mut cos_sum = 0.0f64;
    let mut compared = 0usize;
    for i in 0..n {
        if reference[i].chosen_bytes != quant[i].chosen_bytes {
            if i < COSINE_DIVERGENCE_WINDOW {
                return CosineOutcome::Diverged { pos: i };
            }
            // Divergence past the window: stop comparing (different tokens
            // onward) but keep the score accumulated so far.
            break;
        }
        if let Some(c) = topk_cosine(&reference[i].top, &quant[i].top) {
            cos_sum += c;
            compared += 1;
        }
    }
    if compared == 0 {
        CosineOutcome::Empty
    } else {
        CosineOutcome::Ok {
            mean_cos: cos_sum / compared as f64,
            positions: compared,
        }
    }
}

/// Cosine of two top-k logprob distributions over the union of token surfaces.
/// A surface absent from one side contributes 0 on that side. Returns `None`
/// when either distribution is empty.
fn topk_cosine(
    a: &std::collections::BTreeMap<String, f64>,
    b: &std::collections::BTreeMap<String, f64>,
) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut keys: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    keys.extend(a.keys());
    keys.extend(b.keys());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for k in keys {
        let x = a.get(k).copied().unwrap_or(0.0);
        let y = b.get(k).copied().unwrap_or(0.0);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return Some(0.0);
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

/// Coherence gate: non-empty, real words, not a single-token/punct loop, not
/// NaN-ish. Tuned to reject degenerate output ("the the the", ". . .", a single
/// repeated token) while passing genuine short answers ("Paris is the capital
/// of France.").
fn is_coherent(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    if t.to_lowercase().contains("nan") && t.len() < 8 {
        return false;
    }
    // Require ≥3 distinct alphanumeric characters (rejects "aaaa", "1 1 1").
    let distinct: std::collections::HashSet<char> =
        t.chars().filter(|c| c.is_alphanumeric()).collect();
    if distinct.len() < 3 {
        return false;
    }
    // Require ≥2 whitespace-separated words whose mean length is > 1 char.
    // Rejects "a b c" punctuation-noise and single-word non-answers, but a
    // legitimate one-word answer is handled by the caller's substring anchor.
    let words: Vec<&str> = t.split_whitespace().filter(|w| !w.is_empty()).collect();
    if words.len() < 2 {
        return false;
    }
    let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_len = total_chars as f64 / words.len() as f64;
    if mean_len <= 1.0 {
        return false;
    }
    // Reject a degenerate single-token loop: if one normalized word makes up
    // more than 60% of the words AND there are ≥3 words, it is a repeat loop
    // ("the the the the"). Below 3 words this is too aggressive, so skip it.
    if words.len() >= 3 {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for w in &words {
            let key: String = w
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();
            if !key.is_empty() {
                *counts.entry(key.to_lowercase()).or_insert(0) += 1;
            }
        }
        if let Some(max) = counts.values().copied().max() {
            #[allow(clippy::cast_precision_loss)]
            let ratio = max as f64 / words.len() as f64;
            if ratio > 0.60 {
                return false;
            }
        }
    }
    true
}

// ── Case execution ──────────────────────────────────────────────────────────

/// The reference (bf16) logprob-step cache, keyed by `(model, fixture)` so the
/// cosine cases compute the bf16 baseline once per (model, fixture) and a
/// second model cannot reuse another model's bf16 reference (finding 6).
struct RefCache {
    bf16_logprobs: std::collections::HashMap<String, Vec<LogprobStep>>,
}

pub fn run_manifest(manifest_toml: &str) -> Report {
    let manifest: Manifest = toml::from_str(manifest_toml).expect("manifest.toml must parse");
    let mut report = Report::new();
    let mut refc = RefCache {
        bf16_logprobs: std::collections::HashMap::new(),
    };
    let home = e2e_home();

    // Optional case-id allowlist: `RMLX_E2E_ONLY=id1,id2` runs only the named
    // cases (others are omitted from the run entirely). Empty/unset → run all.
    // General selector for targeted reruns; no effect on the default full run.
    let only: Vec<String> = std::env::var("RMLX_E2E_ONLY")
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_owned())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();

    for case in &manifest.case {
        if !only.is_empty() && !only.iter().any(|id| id == &case.id) {
            continue;
        }
        // Phase 2 rows are declarative-only: record PENDING, never execute.
        if case.tags.iter().any(|t| t == "phase2") {
            report.push(CaseResult {
                id: case.id.clone(),
                feature: case.feature.clone(),
                subfeature: case.subfeature.clone(),
                verdict: Verdict::Pending,
                detail: "Phase 2 — not executed".to_owned(),
                tags: case.tags.clone(),
            });
            continue;
        }
        let mut result = run_case(case, &mut refc, &home);
        // `xfail`-tagged cases document a KNOWN product gap. Downgrade a FAIL to
        // XFAIL ONLY when the failure detail matches the EXPECTED failure mode —
        // never blanket. Infra failures (http error, non-200 status, spawn,
        // parse) keep `Fail` so they still trip `any_failed()` instead of being
        // masked as "known gap". A PASS on an xfail case means the gap was fixed
        // — left as PASS so the grid surfaces that the tag can be dropped.
        if result.verdict == Verdict::Fail
            && case.tags.iter().any(|t| t == "xfail")
            && is_expected_xfail(case, &result.detail)
        {
            result.verdict = Verdict::XFail;
        }
        eprintln!(
            "[e2e] {:<10} {} / {} — {}",
            result.verdict.as_str(),
            result.feature,
            result.subfeature,
            result.detail
        );
        report.push(result);
    }

    // Best-effort: drop transient runtime subtrees under the temp HOME. The
    // `e2e/` report subtree is preserved for the caller to write into.
    cleanup_e2e_home();
    report
}

/// Does a FAIL detail match the case's EXPECTED (documented) failure mode?
/// Only such matches are downgraded to XFAIL — infra failures stay FAIL.
///
/// Keyed on the assert kind so the marker is tied to the documented gap:
/// * `stop_halts` → the stop-length assertion fired ("stop did not shorten").
fn is_expected_xfail(case: &Case, detail: &str) -> bool {
    match case.assert.kind.as_str() {
        "stop_halts" => detail.starts_with("stop did not shorten output"),
        // No other xfail-able expected modes today. Add explicit markers here
        // as new documented gaps are tagged `xfail`; never blanket-downgrade.
        _ => false,
    }
}

fn run_case(case: &Case, refc: &mut RefCache, home: &std::path::Path) -> CaseResult {
    let mk = |verdict: Verdict, detail: String| CaseResult {
        id: case.id.clone(),
        feature: case.feature.clone(),
        subfeature: case.subfeature.clone(),
        verdict,
        detail,
        tags: case.tags.clone(),
    };

    // Resolve the model (if the case needs one).
    let model = match &case.model {
        Some(key) => match resolve_model(key) {
            Some(p) => Some(p),
            None => return mk(Verdict::Skip, format!("model {key} unresolved")),
        },
        None => None,
    };

    // CLI mode.
    if let Some(cli_args) = &case.cli {
        return run_cli_case(case, cli_args, model.as_deref(), home, &mk);
    }

    // Serve mode. The `serve_refused` arch-guard cases carry only a codec
    // (`kv_quant` / `ctk` / `ctv`) + the assert — no `request`/`serve_flags`,
    // since the server is expected to exit at resolve time before any request —
    // so route on the assert kind too.
    if case.serve_flags.is_some()
        || case.request.is_some()
        || case.assert.kind == "serve_refused"
        || case.assert.kind == "spec_decode"
    {
        let Some(model) = model.as_deref() else {
            return mk(Verdict::Skip, "serve case missing model".to_owned());
        };
        return run_serve_case(case, model, refc, home, &mk);
    }

    mk(
        Verdict::Fail,
        "case has neither `cli` nor `serve`/`request`".to_owned(),
    )
}

fn run_cli_case(
    case: &Case,
    cli_args: &[String],
    model: Option<&std::path::Path>,
    home: &std::path::Path,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    // GPU-loading CLI cases (probe-smoke, healthcheck --full) are claim-gated.
    let loads_mlx = cli_args
        .iter()
        .any(|a| a == "--probe-smoke" || a == "--full");
    if loads_mlx {
        claim_preflight();
    }

    let mut cmd = Command::new(rmlx_bin());
    cmd.env("RMLX_HOME", home).env("RUST_LOG", "warn");
    // Absolute fixtures dir so CWD-relative defaults in the spawned binary never
    // misresolve (cargo runs the test with CWD = crate dir, not workspace root).
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    for a in cli_args {
        if a == "$MODEL" {
            match model {
                Some(p) => {
                    cmd.arg(p);
                }
                None => return mk(Verdict::Skip, "CLI case wants $MODEL but none".to_owned()),
            }
        } else if let Some(rel) = a.strip_prefix("$FIXTURES/") {
            cmd.arg(fixtures.join(rel));
        } else {
            cmd.arg(a);
        }
    }
    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => return mk(Verdict::Fail, format!("spawn failed: {e}")),
    };
    let exit = out.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    match case.assert.kind.as_str() {
        "exit_code" => {
            let want: i32 = case
                .assert
                .expect
                .as_deref()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            if exit == want {
                mk(Verdict::Pass, format!("exit {exit}"))
            } else {
                mk(
                    Verdict::Fail,
                    format!("exit {exit} != {want}; stderr: {}", trunc(&stderr)),
                )
            }
        }
        "metric_present" => {
            // exit 0 + a REQUIRED non-empty needle present in stdout. An empty
            // `expect` would collapse to a bare exit-0 check (no signal), so a
            // missing needle is a manifest error, not a pass.
            let Some(needle) = case.assert.expect.as_deref().filter(|n| !n.is_empty()) else {
                return mk(
                    Verdict::Fail,
                    "metric_present requires a non-empty `expect` needle".to_owned(),
                );
            };
            if exit == 0 && stdout.contains(needle) {
                mk(Verdict::Pass, format!("exit 0, stdout has {needle:?}"))
            } else {
                mk(
                    Verdict::Fail,
                    format!("exit {exit}, stdout missing {needle:?}: {}", trunc(&stdout)),
                )
            }
        }
        other => mk(
            Verdict::Fail,
            format!("CLI case unsupported assert kind {other:?}"),
        ),
    }
}

fn run_serve_case(
    case: &Case,
    model: &std::path::Path,
    refc: &mut RefCache,
    home: &std::path::Path,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    claim_preflight();
    let port = pick_port(case);
    let id = model_id(model);
    let fixture = case.request.as_deref().unwrap_or("chat_basic");
    let kind = case.assert.kind.as_str();

    // General serve-flag injection per fixture/kind:
    //  - NIAH needs an 8k+ KV window (default max_ctx is 4096).
    //  - Anthropic ignores per-request enable_thinking, so suppress the
    //    thinking block server-wide for the deterministic anthropic golden.
    let mut serve_extra = case.serve_flags.clone().unwrap_or_default();
    if kind == "niah_retrieval" || fixture == "niah" {
        serve_extra.extend(["--max-ctx".to_owned(), "16384".to_owned()]);
    }
    if fixture.starts_with("anthropic") {
        serve_extra.extend(["--enable-thinking".to_owned(), "false".to_owned()]);
    }
    // Compose-form codecs: pass --ctk/--ctv instead of --kv-quant. Used for
    // codecs whose Display form does not round-trip through `--kv-quant`
    // (e.g. RotK → rot_k / q4_g64).
    if let Some(ctk) = &case.ctk {
        serve_extra.extend(["--cache-type-k".to_owned(), ctk.clone()]);
    }
    if let Some(ctv) = &case.ctv {
        serve_extra.extend(["--cache-type-v".to_owned(), ctv.clone()]);
    }
    // --ctk/--ctv conflict with --kv-quant at the clap layer; never pass both.
    let kv_quant = if case.ctk.is_some() || case.ctv.is_some() {
        None
    } else {
        case.kv_quant.as_deref()
    };

    // SSD cross-restart is special: it owns TWO serve phases (populate→kill→
    // restart→hydrate) under single-MLX discipline, with a dedicated hermetic
    // RMLX_HOME so the spilled `.kvb` survives the process boundary. It must NOT
    // share the long-lived `home` (other cases scrub it) and must NOT spawn the
    // shared single guard below. Hand off before any spawn.
    if kind == "byte_identical_restart" {
        return assert_byte_identical_restart(case, model, &id, port, mk);
    }

    // Multi-model lifecycle owns a registry-mode serve (two model entries) +
    // a deliberate second-process claim-enforcement probe. It must NOT spawn
    // the shared single `--model` guard below. Hand off before any spawn.
    if kind == "model_lifecycle" {
        return assert_model_lifecycle(model, &id, port, mk);
    }

    // Attention dispatch_fired owns a verbose-logging serve so it can scrape the
    // per-dispatch `path` span field from the run jsonl. Hand off before the
    // shared spawn (the shared one pins RUST_LOG=warn, which would suppress the
    // verbose trace spans the scrape depends on).
    if kind == "dispatch_fired" {
        return assert_dispatch_fired(case, model, &id, port, mk);
    }

    // Speculative decoding owns a verbose-logging serve with the drafter flags
    // (`--draft-model`/`--draft-kind`) so it can scrape the round-loop's
    // `<kind>_generate_greedy: done` accept_rate from the run jsonl. Hand off
    // before the shared spawn (which pins no draft model and RUST_LOG=warn).
    if kind == "spec_decode" {
        return assert_spec_decode(case, model, &id, port, mk);
    }

    // Arch-guard refusal: the case asserts that an ILLEGAL KV codec for this
    // arch is rejected at serve *resolve time* (before /health binds) with a
    // documented non-zero exit. `resolve_model_flags` loads config.json,
    // resolves the codec, and `std::process::exit(78)` (EX_CONFIG) on an arch
    // invariant failure. So the shared `spawn_serve` (which waits for /health)
    // would just time out — `serve_refused` instead spawns and WAITS for the
    // early exit, asserting the code. Hand off before any /health-gated spawn.
    if kind == "serve_refused" {
        return assert_serve_refused(case, model, port, mk);
    }

    let guard = match spawn_serve(model, port, kv_quant, &serve_extra, home) {
        Ok(g) => g,
        Err(e) => return mk(Verdict::Fail, format!("serve spawn: {e}")),
    };

    // Cosine is special: it must compare the quant logprobs against a bf16
    // reference, but single-MLX (hard rule 8) forbids two concurrent serve
    // processes. So fetch the quant vector here, DROP the quant server, then
    // spawn the bf16 reference server. Everything else asserts in-place.
    if kind == "cosine_vs_bf16" {
        let quant_steps = match fetch_logprobs(port, &id) {
            Ok(v) => v,
            Err(e) => {
                drop(guard);
                return mk(Verdict::Fail, e);
            }
        };
        drop(guard); // release the quant server BEFORE the bf16 spawn.
                     // HARD single-MLX: settle the quant Metal context (kill + ~800ms sleep
                     // + claim glob-remove) BEFORE finish_cosine spawns the bf16 server. No
                     // two `rmlx serve` may overlap. `Drop` already killed the child, but
                     // claim_preflight guarantees the process is gone and the claim cleared.
        claim_preflight();
        return finish_cosine(case, &id, &quant_steps, refc, mk);
    }

    let verdict = match kind {
        "golden" => assert_golden(port, fixture, &id, mk, case),
        "contains_coherent" => assert_contains_coherent(port, fixture, &id, mk, case),
        "coherent" => assert_coherent(port, fixture, &id, mk, case),
        "niah_retrieval" => assert_niah(port, &id, mk),
        "thinking" => assert_thinking(port, &id, mk),
        "stop_halts" => assert_stop_halts(port, &id, mk),
        "cache_hit_equivalence" => assert_cache_hit_equivalence(port, &id, mk),
        "image" => assert_image(port, fixture, &id, mk, case),
        "tool_call" => assert_tool_call(port, fixture, &id, mk, case),
        other => mk(
            Verdict::Fail,
            format!("serve case unsupported assert {other:?}"),
        ),
    };
    drop(guard);
    verdict
}

/// Directory holding recorded golden token sequences, checked into the repo
/// alongside the manifest so the golden travels with the test.
fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("e2e")
        .join("golden")
}

/// Deterministic port per case id so parallel-but-serialized runs don't clash.
fn pick_port(case: &Case) -> u16 {
    let mut h: u32 = 2166136261;
    for b in case.id.bytes() {
        h = h.wrapping_mul(16777619) ^ u32::from(b);
    }
    18000 + (h % 4000) as u16
}

/// REAL golden: capture the per-token chosen-token-id sequence (byte-for-byte,
/// via the OpenAI logprobs `bytes` field) at temp=0 greedy and compare it to a
/// recorded golden file under `tests/e2e/golden/<case_id>.json`.
///
/// * Golden file ABSENT (and not in regen mode) → `Skip` with "no golden
///   recorded" — we do NOT silently downgrade to a substring check.
/// * `RMLX_E2E_REGEN_GOLDEN=1` or first run with the file absent → WRITE the
///   golden and `Pass`.
/// * Golden present → byte-for-byte compare; any divergence is `Fail` with the
///   first mismatching position.
///
/// Golden requires per-token bytes, which only the OpenAI non-stream logprobs
/// path exposes. Streaming / anthropic fixtures use `contains_coherent`.
fn assert_golden(
    port: u16,
    fixture: &str,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
    case: &Case,
) -> CaseResult {
    // Force per-token logprobs on so we get the byte-identity sequence.
    let mut req = request_fixture(fixture, id);
    if let Some(obj) = req.as_object_mut() {
        obj.insert("logprobs".to_owned(), serde_json::json!(true));
        obj.insert("top_logprobs".to_owned(), serde_json::json!(1));
        obj.insert("stream".to_owned(), serde_json::json!(false));
    }
    let body = req.to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let Some(steps) = openai_logprob_steps(&resp) else {
        return mk(
            Verdict::Fail,
            format!("no per-token logprobs in response: {}", trunc(&resp)),
        );
    };
    // Golden = the ordered list of chosen-token byte arrays.
    let actual: Vec<Vec<u8>> = steps.iter().map(|s| s.chosen_bytes.clone()).collect();

    let golden_path = golden_dir().join(format!("{}.json", case.id));
    let regen = std::env::var("RMLX_E2E_REGEN_GOLDEN").is_ok_and(|v| v == "1");
    let exists = golden_path.exists();

    if regen || !exists {
        if let Err(e) = std::fs::create_dir_all(golden_dir()) {
            return mk(Verdict::Fail, format!("create golden dir: {e}"));
        }
        // Trailing newline so the recorded file is end-of-file-fixer clean.
        let serial =
            serde_json::to_string_pretty(&actual).unwrap_or_else(|_| "[]".to_owned()) + "\n";
        match std::fs::write(&golden_path, serial) {
            Ok(()) => {
                let why = if regen { "regen" } else { "first run" };
                return mk(
                    Verdict::Pass,
                    format!(
                        "golden recorded ({why}): {} tokens → {}",
                        actual.len(),
                        golden_path.display()
                    ),
                );
            }
            Err(e) => return mk(Verdict::Fail, format!("write golden: {e}")),
        }
    }

    // Golden present: byte-for-byte compare.
    let raw = match std::fs::read_to_string(&golden_path) {
        Ok(s) => s,
        Err(e) => return mk(Verdict::Fail, format!("read golden: {e}")),
    };
    let expected: Vec<Vec<u8>> = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return mk(Verdict::Fail, format!("parse golden: {e}")),
    };
    if expected == actual {
        mk(
            Verdict::Pass,
            format!("golden match: {} tokens byte-identical", actual.len()),
        )
    } else {
        let pos = expected
            .iter()
            .zip(actual.iter())
            .position(|(e, a)| e != a)
            .unwrap_or_else(|| expected.len().min(actual.len()));
        mk(
            Verdict::Fail,
            format!(
                "golden MISMATCH at token {pos}: expected {} tokens, got {} (golden {})",
                expected.len(),
                actual.len(),
                golden_path.display()
            ),
        )
    }
}

/// Substring + coherence check (the honest name for the old `golden` behavior):
/// deterministic temp=0 output must be coherent AND contain the expected
/// substring. Used for fixtures that cannot expose per-token byte identity
/// over their wire shape (streaming SSE, anthropic `/v1/messages`).
fn assert_contains_coherent(
    port: u16,
    fixture: &str,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
    case: &Case,
) -> CaseResult {
    let body = request_fixture(fixture, id).to_string();
    let path = if fixture.starts_with("anthropic") {
        "/v1/messages"
    } else {
        "/v1/chat/completions"
    };
    let (status, resp) = match http_post(port, path, &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let text = if fixture.starts_with("anthropic") {
        anthropic_text(&resp)
    } else if fixture.contains("stream") {
        Some(sse_collect_content(&resp))
    } else {
        openai_content(&resp)
    };
    let Some(text) = text else {
        return mk(Verdict::Fail, format!("no content: {}", trunc(&resp)));
    };
    let want = case.assert.expect.as_deref().unwrap_or("Paris");
    if is_coherent(&text) && text.contains(want) {
        mk(
            Verdict::Pass,
            format!("temp=0 output contains {want:?}: {:?}", trunc(&text)),
        )
    } else {
        mk(
            Verdict::Fail,
            format!("expected {want:?} in coherent output: {:?}", trunc(&text)),
        )
    }
}

fn assert_coherent(
    port: u16,
    fixture: &str,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
    case: &Case,
) -> CaseResult {
    let body = request_fixture(fixture, id).to_string();
    let path = if fixture.starts_with("anthropic") {
        "/v1/messages"
    } else {
        "/v1/chat/completions"
    };
    let (status, resp) = match http_post(port, path, &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let text = if fixture.contains("stream") {
        sse_collect_content(&resp)
    } else {
        openai_content(&resp).unwrap_or_default()
    };
    if !is_coherent(&text) {
        return mk(Verdict::Fail, format!("incoherent: {:?}", trunc(&text)));
    }
    // Optional substring anchor (e.g. multi_turn → "teal", stop → must NOT
    // contain the post-stop token).
    if let Some(expect) = &case.assert.expect {
        if let Some(neg) = expect.strip_prefix('!') {
            if text.contains(neg) {
                return mk(
                    Verdict::Fail,
                    format!("must not contain {neg:?}: {:?}", trunc(&text)),
                );
            }
        } else if !text.to_lowercase().contains(&expect.to_lowercase()) {
            return mk(
                Verdict::Fail,
                format!("missing {expect:?}: {:?}", trunc(&text)),
            );
        }
    }
    mk(Verdict::Pass, format!("coherent: {:?}", trunc(&text)))
}

fn assert_niah(port: u16, id: &str, mk: &dyn Fn(Verdict, String) -> CaseResult) -> CaseResult {
    let body = niah_request(id).to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let text = openai_content(&resp).unwrap_or_default();
    if text.contains(NIAH_NEEDLE) {
        mk(
            Verdict::Pass,
            format!("needle recovered: {:?}", trunc(&text)),
        )
    } else {
        mk(
            Verdict::Fail,
            format!("needle {NIAH_NEEDLE} not in: {:?}", trunc(&text)),
        )
    }
}

/// Fetch the `chat_logprobs` per-token logprob steps (chosen-token bytes +
/// top-k distribution) from a live server.
fn fetch_logprobs(port: u16, id: &str) -> Result<Vec<LogprobStep>, String> {
    let body = request_fixture("chat_logprobs", id).to_string();
    let (status, resp) =
        http_post(port, "/v1/chat/completions", &body).map_err(|e| format!("http: {e}"))?;
    if status != 200 {
        return Err(format!("status {status}: {}", trunc(&resp)));
    }
    openai_logprob_steps(&resp).ok_or_else(|| "no logprobs in response".to_owned())
}

/// Complete the cosine assertion after the quant server has been dropped:
/// (lazily) compute the bf16 reference logprob steps once per (model, fixture),
/// cache them, and compare against the quant codec's steps. The bf16 server is
/// spawned here under single-MLX discipline — the quant server is already torn
/// down (and the GPU settled via claim_preflight) by the caller.
///
/// Fidelity signal: while the chosen token-ids agree, score the per-position
/// top-k logprob DISTRIBUTIONS by cosine. A chosen-token-id divergence within
/// the first `COSINE_DIVERGENCE_WINDOW` tokens is the real failure (the codec
/// changed the argmax early) and is reported as such — NOT smuggled into a
/// scalar over mismatched tokens.
fn finish_cosine(
    case: &Case,
    id: &str,
    quant_steps: &[LogprobStep],
    refc: &mut RefCache,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    // Key on (model, fixture) so a second model cannot reuse the wrong bf16 ref.
    let model_key = case.model.as_deref().unwrap_or("?");
    let ref_key = format!("{model_key}::chat_logprobs");
    if !refc.bf16_logprobs.contains_key(&ref_key) {
        let Some(model) = case.model.as_deref().and_then(resolve_model) else {
            return mk(Verdict::Fail, "model unresolved for bf16 ref".to_owned());
        };
        // Caller already ran claim_preflight after dropping the quant server.
        let home = e2e_home();
        let port = 17000 + (case.id.len() as u16 % 500);
        let guard = match spawn_serve(&model, port, Some("none"), &[], &home) {
            Ok(g) => g,
            Err(e) => return mk(Verdict::Fail, format!("bf16 ref spawn: {e}")),
        };
        let fetched = fetch_logprobs(port, id);
        drop(guard);
        match fetched {
            Ok(v) => {
                refc.bf16_logprobs.insert(ref_key.clone(), v);
            }
            Err(e) => return mk(Verdict::Fail, format!("bf16 ref: {e}")),
        }
    }
    let ref_steps = &refc.bf16_logprobs[&ref_key];
    let thr = case.assert.min_cosine.unwrap_or(0.99);
    match compare_logprob_steps(ref_steps, quant_steps) {
        CosineOutcome::Ok {
            mean_cos,
            positions,
        } => {
            if mean_cos >= thr {
                mk(
                    Verdict::Pass,
                    format!("top-k cosine {mean_cos:.4} >= {thr} over {positions} matched tokens"),
                )
            } else {
                mk(
                    Verdict::Fail,
                    format!("top-k cosine {mean_cos:.4} < {thr} over {positions} matched tokens"),
                )
            }
        }
        CosineOutcome::Diverged { pos } => mk(
            Verdict::Fail,
            format!(
                "chosen token-id diverged from bf16 at position {pos} (within first \
                 {COSINE_DIVERGENCE_WINDOW}) — codec changed the argmax early"
            ),
        ),
        CosineOutcome::Empty => mk(
            Verdict::Fail,
            "no comparable top-k positions (empty logprobs)".to_owned(),
        ),
    }
}

/// Prove the `stop` parameter halts generation: request the same prompt with
/// and without `stop`, assert the stopped completion is strictly shorter and
/// does not contain the stop string. Both at temp=0 for determinism.
fn assert_stop_halts(
    port: u16,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    let prompt = "Repeat exactly, nothing else: alpha bravo charlie delta echo";
    let no_stop = serde_json::json!({
        "model": id, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 48, "temperature": 0.0, "seed": 0, "enable_thinking": false
    })
    .to_string();
    let with_stop = serde_json::json!({
        "model": id, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 48, "temperature": 0.0, "seed": 0, "enable_thinking": false,
        "stop": ["charlie"]
    })
    .to_string();

    let full = match http_post(port, "/v1/chat/completions", &no_stop) {
        Ok((200, b)) => openai_content(&b).unwrap_or_default(),
        Ok((s, b)) => return mk(Verdict::Fail, format!("no-stop status {s}: {}", trunc(&b))),
        Err(e) => return mk(Verdict::Fail, format!("no-stop http: {e}")),
    };
    let stopped = match http_post(port, "/v1/chat/completions", &with_stop) {
        Ok((200, b)) => openai_content(&b).unwrap_or_default(),
        Ok((s, b)) => return mk(Verdict::Fail, format!("stop status {s}: {}", trunc(&b))),
        Err(e) => return mk(Verdict::Fail, format!("stop http: {e}")),
    };

    // Proof of the stop FEATURE: the stopped completion must be strictly
    // shorter than the unstopped one (generation halted at the stop boundary).
    // We do NOT require the stop string to be byte-excluded — stop matching is
    // token-sequence based, so the detokenised text may still contain the word
    // when it falls inside a larger token; the length delta is the load-bearing
    // signal that the parameter took effect.
    if stopped.len() < full.len() {
        mk(
            Verdict::Pass,
            format!(
                "stop halted generation: stopped {} < full {} chars; stopped={:?}",
                stopped.len(),
                full.len(),
                trunc(&stopped)
            ),
        )
    } else {
        mk(
            Verdict::Fail,
            format!(
                "stop did not shorten output: stopped={:?} full={:?}",
                trunc(&stopped),
                trunc(&full)
            ),
        )
    }
}

fn assert_thinking(port: u16, id: &str, mk: &dyn Fn(Verdict, String) -> CaseResult) -> CaseResult {
    let body = request_fixture("thinking", id).to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let reasoning = openai_reasoning(&resp).unwrap_or_default();
    let content = openai_content(&resp).unwrap_or_default();
    if reasoning.trim().is_empty() {
        return mk(
            Verdict::Fail,
            format!("reasoning_content empty; content={:?}", trunc(&content)),
        );
    }
    // budget enforced: thinking_budget=64 → reasoning should not be runaway.
    // answer correct: 17*4 = 68 should appear in content or reasoning.
    let answered = content.contains("68") || reasoning.contains("68");
    if answered {
        mk(
            Verdict::Pass,
            format!(
                "reasoning populated ({} chars), answer 68 found",
                reasoning.len()
            ),
        )
    } else {
        mk(
            Verdict::Fail,
            format!(
                "answer 68 missing; content={:?} reasoning_len={}",
                trunc(&content),
                reasoning.len()
            ),
        )
    }
}

// ── Phase 2a: SSD cross-restart + prompt-cache reuse ─────────────────────────

/// A long, prefix-stable prompt that forms at least one full 256-token
/// prompt-cache block so both the SSD spill (whole-block-only) and the
/// ExactOnly prompt-cache hit (Bonsai/Qwen3) have a stable cache key. ~520
/// words of English ≈ ≳600 tokens ≈ 2+ full 256-token blocks. Mirrors the
/// PROMPT_A used by `crates/rmlx-server/tests/ssd_cache_restart.rs`.
const CACHE_PROMPT_LONG: &str = "You are a meticulous senior systems engineer reviewing a \
design specification. Read the following description carefully and then produce a single concise \
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
written outside that root during normal operation.";

/// A second prompt with no shared prefix with `CACHE_PROMPT_LONG`. Issuing it
/// evicts the long prompt from a single-slot RAM cache, triggering its spill to
/// the SSD tier. Mirrors PROMPT_B in `ssd_cache_restart.rs`.
const CACHE_PROMPT_EVICTOR: &str = "Translate the following sentence into formal French and \
then explain, in two sentences, one subtle grammatical choice you made and why it preserves \
the original register: 'The quiet engineer reviewed the proposal twice before approving the \
deployment to the production cluster on Friday.'";

/// Build a deterministic temp=0 chat-completion body for an explicit prompt
/// string, with per-token logprob bytes ON so the caller can capture the
/// byte-for-byte chosen-token-id sequence. `enable_thinking=false` so Bonsai
/// emits the final answer directly (not buried in `reasoning_content`).
fn long_logprob_body(id: &str, prompt: &str, max_tokens: u32) -> String {
    serde_json::json!({
        "model": id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
        "enable_thinking": false,
        "logprobs": true, "top_logprobs": 1, "stream": false
    })
    .to_string()
}

/// One completion captured for the state-feature proofs: the per-token chosen-
/// byte sequence (the byte-for-byte token-id identity, used for the equivalence
/// comparison) plus the detokenized `content` (used only for the coherence
/// smoke-probe). The two are distinct: the logprobs `bytes` field carries the
/// raw token-surface bytes (BPE byte-level form, with `Ġ` for spaces), which is
/// the correct identity signal but NOT human-readable; the `content` field is
/// the detokenized answer the coherence gate must judge.
#[derive(Clone, Debug)]
struct Completion {
    chosen: Vec<Vec<u8>>,
    content: String,
}

/// POST an explicit-prompt logprob request; return the per-token chosen-byte
/// sequence + the detokenized content.
fn fetch_completion(
    port: u16,
    id: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<Completion, String> {
    let body = long_logprob_body(id, prompt, max_tokens);
    let (status, resp) =
        http_post(port, "/v1/chat/completions", &body).map_err(|e| format!("http: {e}"))?;
    if status != 200 {
        return Err(format!("status {status}: {}", trunc(&resp)));
    }
    let steps = openai_logprob_steps(&resp)
        .ok_or_else(|| format!("no per-token logprobs in response: {}", trunc(&resp)))?;
    Ok(Completion {
        chosen: steps.into_iter().map(|s| s.chosen_bytes).collect(),
        content: openai_content(&resp).unwrap_or_default(),
    })
}

/// Per-model prompt-cache counters scraped from `GET /metrics/cache` (JSON),
/// summed across all model slots. Used to prove a cache HIT actually fired.
#[derive(Clone, Copy, Default, Debug)]
struct CacheCounters {
    hits: u64,
    block_hits: u64,
    misses: u64,
    ssd_hits: u64,
}

/// Scrape `/metrics/cache` and sum the prompt-cache counters across all model
/// slots. Returns `Err` when the endpoint is unreachable or the body does not
/// parse — a metrics-infra failure must FAIL the case, not silently read 0.
fn fetch_cache_counters(port: u16) -> Result<CacheCounters, String> {
    let (status, body) = http_get(port, "/metrics/cache").map_err(|e| format!("http: {e}"))?;
    if status != 200 {
        return Err(format!("metrics status {status}: {}", trunc(&body)));
    }
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("metrics parse: {e}: {}", trunc(&body)))?;
    let models = v["models"]
        .as_array()
        .ok_or_else(|| format!("metrics has no models array: {}", trunc(&body)))?;
    let mut c = CacheCounters::default();
    // Track whether at least one model entry carried each counter key.  An
    // absent key (cache_stats() was None server-side) would silently coerce to
    // 0 and be indistinguishable from a real zero — that would let a broken
    // metrics path pass as "no hits". A present key with value 0 is a
    // legitimate zero and does NOT trigger this guard.
    let mut saw_block_hits = false;
    let mut saw_ssd_hits = false;
    for m in models {
        if m["block_hits"].is_u64() {
            saw_block_hits = true;
        }
        if m["ssd_hits"].is_u64() {
            saw_ssd_hits = true;
        }
        c.hits += m["hits"].as_u64().unwrap_or(0);
        c.block_hits += m["block_hits"].as_u64().unwrap_or(0);
        c.misses += m["misses"].as_u64().unwrap_or(0);
        c.ssd_hits += m["ssd_hits"].as_u64().unwrap_or(0);
    }
    if !models.is_empty() && (!saw_block_hits || !saw_ssd_hits) {
        return Err(format!(
            "metrics/cache present but missing counter fields \
             (saw_block_hits={saw_block_hits}, saw_ssd_hits={saw_ssd_hits}) \
             — cache_stats() was likely None server-side"
        ));
    }
    Ok(c)
}

/// Walk `<RMLX_HOME>/cache/kv/*/` and count (.kvb files, index.db rows) summed
/// across every namespace. Used to prove the long prompt actually spilled to
/// the SSD tier before the restart. Std-only: parses the SQLite `index.db`
/// header is overkill, so we count `.kvb` files as the spill witness and treat
/// a present, non-empty `index.db` as the row witness via a cheap size check.
fn ssd_kvb_count(home: &std::path::Path) -> usize {
    let kv_root = home.join("cache").join("kv");
    let mut kvb_files = 0usize;
    let Ok(namespaces) = std::fs::read_dir(&kv_root) else {
        return 0;
    };
    for ns in namespaces.flatten() {
        let ns_dir = ns.path();
        if !ns_dir.is_dir() {
            continue;
        }
        if let Ok(files) = std::fs::read_dir(&ns_dir) {
            for f in files.flatten() {
                if f.path().extension().is_some_and(|e| e == "kvb") {
                    kvb_files += 1;
                }
            }
        }
    }
    kvb_files
}

/// **REAL** SSD cross-restart proof (`byte_identical_restart`).
///
/// Owns two serve phases under single-MLX discipline, with a dedicated
/// hermetic `RMLX_HOME` so the spilled `.kvb` survives the process boundary:
///
/// * **Phase 1 (populate + spill):** serve with `--kv-ssd-cache-gb 1
///   --prompt-cache-slots 1`. Generate from the long prompt at temp=0 greedy,
///   capturing its per-token chosen-byte sequence. Issue a second, prefix-
///   disjoint prompt → evicts the long prompt from the single RAM slot →
///   spills it to SSD. Poll the disk until a `.kvb` lands (fire-and-forget
///   drain).
/// * **Restart boundary:** kill the server, `claim_preflight` (settle Metal +
///   clear claim), restart with the SAME `RMLX_HOME` + same SSD flag.
/// * **Phase 2 (restart + hydrate):** RAM is empty (fresh process), so the
///   same long prompt is a RAM miss that MUST hydrate the rehydrated KV blocks
///   from the `.kvb` spill. Re-capture the per-token chosen-byte sequence.
///
/// PASS requires BOTH: (a) the Phase-2 completion is **byte-identical** to
/// Phase 1 (rehydrated blocks reproduce the decode exactly), and (b)
/// `/metrics/cache` reports `ssd_hits >= 1` (the RAM miss was served from the
/// SSD tier, not a silent cold re-prefill that happens to match). A wrong
/// hydrate — different bytes OR zero ssd_hits — FAILS.
fn assert_byte_identical_restart(
    case: &Case,
    model: &std::path::Path,
    id: &str,
    port: u16,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    // Dedicated hermetic HOME — the SSD `.kvb` + index must survive the restart
    // and NOT be scrubbed by the shared-run cleanup. Per-case id keeps it
    // unique within the process.
    let ssd_home =
        std::env::temp_dir().join(format!("rmlx_e2e_ssd_{}_{}", std::process::id(), case.id));
    let _ = std::fs::remove_dir_all(&ssd_home); // fresh each run (hermetic)
    if let Err(e) = std::fs::create_dir_all(&ssd_home) {
        return mk(Verdict::Fail, format!("create ssd home: {e}"));
    }
    let ssd_flags = [
        "--kv-ssd-cache-gb".to_owned(),
        "1".to_owned(),
        "--prompt-cache-slots".to_owned(),
        "1".to_owned(),
    ];
    let max_tokens = 24u32;
    // Phase-2 uses a distinct port: the just-killed Phase-1 listener can sit in
    // TIME_WAIT, causing EADDRINUSE on an immediate re-bind. Cross-restart
    // identity is keyed by RMLX_HOME + .kvb files, not the port number, so a
    // different port is safe. Stay inside the 18000..22000 band used by pick_port.
    let port2 = 18000 + ((port - 18000 + 1000) % 4000);

    // ── Phase 1: populate + spill ────────────────────────────────────────────
    claim_preflight();
    let phase1 = match spawn_serve(model, port, None, &ssd_flags, &ssd_home) {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&ssd_home);
            return mk(Verdict::Fail, format!("phase-1 serve spawn: {e}"));
        }
    };
    let phase1_out = match fetch_completion(port, id, CACHE_PROMPT_LONG, max_tokens) {
        Ok(b) => b,
        Err(e) => {
            drop(phase1);
            let _ = std::fs::remove_dir_all(&ssd_home);
            return mk(Verdict::Fail, format!("phase-1 long prompt: {e}"));
        }
    };
    let bytes_phase1 = phase1_out.chosen;
    // Smoke-probe coherence on the DETOKENIZED content (not the BPE token-
    // surface bytes), before we trust the byte sequence as the identity ref.
    if !is_coherent(&phase1_out.content) {
        drop(phase1);
        let _ = std::fs::remove_dir_all(&ssd_home);
        return mk(
            Verdict::Fail,
            format!(
                "phase-1 output not coherent: {:?}",
                trunc(&phase1_out.content)
            ),
        );
    }
    // Evictor prompt: prefix-disjoint → evicts the long prompt from the single
    // RAM slot → triggers its spill to SSD.
    if let Err(e) = fetch_completion(port, id, CACHE_PROMPT_EVICTOR, 8) {
        drop(phase1);
        let _ = std::fs::remove_dir_all(&ssd_home);
        return mk(Verdict::Fail, format!("phase-1 evictor prompt: {e}"));
    }
    // Spill is a fire-and-forget drain-thread write — poll the disk, don't sleep.
    let mut kvb = 0usize;
    for _ in 0..40 {
        kvb = ssd_kvb_count(&ssd_home);
        if kvb >= 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    if kvb == 0 {
        drop(phase1);
        let _ = std::fs::remove_dir_all(&ssd_home);
        return mk(
            Verdict::Fail,
            "no .kvb spilled (any namespace) after eviction (spill path did not fire)".to_owned(),
        );
    }

    // ── Restart boundary: kill phase 1 BEFORE phase 2 spawns (single-MLX) ────
    drop(phase1);
    claim_preflight();

    // ── Phase 2: restart + hydrate ───────────────────────────────────────────
    let phase2 = match spawn_serve(model, port2, None, &ssd_flags, &ssd_home) {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&ssd_home);
            return mk(Verdict::Fail, format!("phase-2 serve spawn: {e}"));
        }
    };
    let bytes_phase2 = match fetch_completion(port2, id, CACHE_PROMPT_LONG, max_tokens) {
        Ok(b) => b.chosen,
        Err(e) => {
            drop(phase2);
            let _ = std::fs::remove_dir_all(&ssd_home);
            return mk(Verdict::Fail, format!("phase-2 long prompt: {e}"));
        }
    };
    // Scrape the cross-restart ssd_hits BEFORE teardown.
    let counters = fetch_cache_counters(port2);
    drop(phase2);
    claim_preflight();
    let _ = std::fs::remove_dir_all(&ssd_home);

    // ── Assertions ───────────────────────────────────────────────────────────
    // (a) byte-identical completion across the restart.
    if bytes_phase1 != bytes_phase2 {
        let pos = bytes_phase1
            .iter()
            .zip(bytes_phase2.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| bytes_phase1.len().min(bytes_phase2.len()));
        return mk(
            Verdict::Fail,
            format!(
                "restart NOT byte-identical: diverged at token {pos} \
                 (phase1 {} tokens, phase2 {} tokens) — hydrated blocks changed decode",
                bytes_phase1.len(),
                bytes_phase2.len()
            ),
        );
    }
    // (b) the RAM miss was served from the SSD tier (not a silent cold prefill).
    let counters = match counters {
        Ok(c) => c,
        Err(e) => return mk(Verdict::Fail, format!("phase-2 /metrics/cache: {e}")),
    };
    if counters.ssd_hits == 0 {
        return mk(
            Verdict::Fail,
            format!(
                "restart bytes matched but ssd_hits=0 — the long prompt was NOT served \
                 from the SSD tier (cold re-prefill that happened to match). kvb_spilled={kvb}"
            ),
        );
    }
    mk(
        Verdict::Pass,
        format!(
            "byte-identical across restart ({} tokens) AND served from SSD \
             (ssd_hits={}, kvb_spilled={kvb})",
            bytes_phase1.len(),
            counters.ssd_hits
        ),
    )
}

/// **REAL** prompt-cache prefix-reuse proof (`cache_hit_equivalence`).
///
/// Bonsai (`Qwen3ForCausalLM`) is `ReusePolicy::ExactOnly`, so the only legal
/// reuse scenario is a full-token-equality exact prefix hit — i.e. re-issuing
/// the SAME long prompt. The proof, on a single server with the prompt cache
/// ON:
///
/// 1. Issue the long prompt (request A) — a cache MISS that warms a slot with a
///    post-prefill KV snapshot. Capture its detokenized `content` (the
///    user-visible output) as the no-cache reference (this very request was
///    uncached).
/// 2. Snapshot `/metrics/cache` block_hits.
/// 3. Re-issue the SAME long prompt (request B) — an EXACT prefix hit. Capture
///    its `content`.
///
/// PASS requires BOTH: (a) request B's `content` is **byte-identical** to
/// request A's (cache reuse must NOT change the produced output) AND (b) the
/// prompt-cache hit counter **incremented** (`block_hits` rose between A and B
/// — the second request genuinely took the cache path, not a re-prefill). A
/// reuse that changes the produced text FAILS; a "hit" that never increments
/// the counter FAILS (no tautology — both legs must hold).
///
/// **Equivalence axes = `content` AND logprobs-stream length.** Run-2a
/// surfaced a separate reporting finding (cached `first_id` replayed without a
/// logprob, so the hit's `logprobs.content` carried N-1 entries vs the miss's
/// N) — RESOLVED: prefill-token logprobs are stored alongside `first_id` and
/// replayed on hit. With the fix the hit's logprob stream is length-equal to
/// the miss's, so this proof now HARD-ASSERTS that parity
/// (`a.chosen.len() == b.chosen.len()`) in addition to `content` byte-equality
/// and the hit-counter increment.
fn assert_cache_hit_equivalence(
    port: u16,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    let max_tokens = 24u32;
    // Request A — cold (uncached) reference.
    let a = match fetch_completion(port, id, CACHE_PROMPT_LONG, max_tokens) {
        Ok(b) => b,
        Err(e) => return mk(Verdict::Fail, format!("request A: {e}")),
    };
    // Smoke-probe on the detokenized content, not the BPE token-surface bytes.
    if !is_coherent(&a.content) {
        return mk(
            Verdict::Fail,
            format!("request A not coherent: {:?}", trunc(&a.content)),
        );
    }
    let before = match fetch_cache_counters(port) {
        Ok(c) => c,
        Err(e) => return mk(Verdict::Fail, format!("pre-hit /metrics/cache: {e}")),
    };
    // Request B — identical prompt → must be an exact prefix hit.
    let b = match fetch_completion(port, id, CACHE_PROMPT_LONG, max_tokens) {
        Ok(b) => b,
        Err(e) => return mk(Verdict::Fail, format!("request B: {e}")),
    };
    let after = match fetch_cache_counters(port) {
        Ok(c) => c,
        Err(e) => return mk(Verdict::Fail, format!("post-hit /metrics/cache: {e}")),
    };

    // (a) output equivalence — cache reuse must not perturb the produced text.
    if a.content != b.content {
        return mk(
            Verdict::Fail,
            format!(
                "cache reuse CHANGED output: A={:?} B={:?}",
                trunc(&a.content),
                trunc(&b.content)
            ),
        );
    }
    // (b) the hit counter incremented — B genuinely took the cache path.
    let block_hit_delta = after.block_hits.saturating_sub(before.block_hits);
    if block_hit_delta == 0 {
        return mk(
            Verdict::Fail,
            format!(
                "output matched but prompt-cache block_hits did NOT increment \
                 (before={}, after={}) — request B did not hit the cache",
                before.block_hits, after.block_hits
            ),
        );
    }
    // (c) logprobs-stream length parity. The hit path must emit exactly as many
    // per-token logprob entries as the miss path (the cached first token now
    // carries its stored prefill logprob). A skew here means the first-token
    // logprob was dropped — regression.
    let (len_a, len_b) = (a.chosen.len(), b.chosen.len());
    if len_a != len_b {
        return mk(
            Verdict::Fail,
            format!(
                "logprobs stream length skew on cache hit \
                 (miss A {len_a} vs hit B {len_b} entries) — first-token logprob \
                 dropped on the exact-hit path",
            ),
        );
    }
    mk(
        Verdict::Pass,
        format!(
            "exact-prefix reuse: content byte-identical, logprobs length-equal \
             ({len_a} entries) AND block_hits +{} ({}→{})",
            block_hit_delta, before.block_hits, after.block_hits
        ),
    )
}

// ── Phase 2b: multi-model lifecycle ──────────────────────────────────────────

/// Spawn `rmlx serve --registry <json> --max-loaded-models <cap> --port <port>`
/// and block until `/health` is green. Mirrors `spawn_serve` but uses the
/// registry path (multiple model entries) instead of a single `--model`, and
/// leaves `RUST_LOG=warn` (the lifecycle proof reads the HTTP API, not logs).
///
/// Registry mode eagerly pre-loads every entry at startup, bounded by the slot
/// LRU at `cap` — so on a green `/health` the resident set is already the
/// `cap`-survivor of the eager preload (see serve.rs "Eager model preload").
fn spawn_serve_registry(
    registry_json: &std::path::Path,
    port: u16,
    max_loaded: usize,
    home: &std::path::Path,
) -> Result<ServeGuard, String> {
    let mut cmd = Command::new(rmlx_bin());
    cmd.arg("serve")
        .arg("--registry")
        .arg(registry_json)
        .arg("--max-loaded-models")
        .arg(max_loaded.to_string())
        .arg("--port")
        .arg(port.to_string())
        .env("RMLX_HOME", home)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd
        .spawn()
        .map_err(|e| format!("spawn rmlx serve --registry: {e}"))?;
    let mut guard = ServeGuard {
        child,
        port,
        home: home.to_path_buf(),
    };
    let deadline = Instant::now() + Duration::from_secs(180);
    #[allow(unused_assignments)]
    let mut last_seen = "no response yet".to_owned();
    loop {
        if let Some(status) = guard.child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!("rmlx serve --registry exited early: {status}"));
        }
        match http_get(port, "/health") {
            Ok((200, body)) => {
                if body.contains("\"ok\":true") || body.contains("\"ok\": true") {
                    return Ok(guard);
                }
                last_seen = format!("status 200 but body not ok: {}", trunc(&body));
            }
            Ok((status, body)) => last_seen = format!("status {status}: {}", trunc(&body)),
            Err(e) => last_seen = format!("connect/read error: {e}"),
        }
        if Instant::now() > deadline {
            return Err(format!(
                "rmlx serve --registry /health never green within 180s (last: {last_seen})"
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Parse the boolean `loaded` field from a `/v1/models/{id}/status` body.
fn status_loaded(body: &str) -> Option<bool> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["loaded"].as_bool()
}

/// GET `/v1/models/{id}/status` → `Ok(loaded_bool)` on 200, else an `Err`
/// describing the wire failure. An unparseable / non-200 response is an infra
/// failure that must FAIL the case, never silently read "not loaded".
fn model_loaded(port: u16, model_id: &str) -> Result<bool, String> {
    let path = format!("/v1/models/{model_id}/status");
    let (status, body) = http_get(port, &path).map_err(|e| format!("status http: {e}"))?;
    if status != 200 {
        return Err(format!("status {status}: {}", trunc(&body)));
    }
    status_loaded(&body).ok_or_else(|| format!("no `loaded` field: {}", trunc(&body)))
}

/// **REAL** multi-model lifecycle proof (`model_lifecycle`).
///
/// Resolves a second model (`GEMMA4_E2B`) distinct from model A (the case's
/// `model`). When the 2nd model is absent the 2-model legs (b)/(c) record SKIP
/// inline and only the single-model legs run; when present, the full transition
/// chain is proven. Single-MLX discipline: exactly one `rmlx serve` is resident
/// for the lifecycle legs; the claim-enforcement leg deliberately starts a
/// SECOND `rmlx serve` on the SAME port and asserts it is REJECTED (exit 11),
/// never reaching a competing Metal context.
///
/// Legs (each a genuinely falsifiable transition; a wrong status FAILS):
///   (a) load A → A `loaded:true`.
///   (b) cap=2, both A+B registered → after eager preload both `loaded:true`.
///   (c) cap=1 → loading B evicts A (LRU): B `loaded:true`, A flips to `false`.
///   (d) explicit unload B → B `loaded:false`; a 2nd unload → 404.
///   (e) claim enforcement → a 2nd `rmlx serve` on the held port exits non-zero
///       (code 11) and does NOT serve `/health`.
///
/// A status that does not flip as required, a 2nd-process that wrongly starts,
/// or an unparseable status body all FAIL.
#[allow(clippy::too_many_lines)]
fn assert_model_lifecycle(
    model_a: &std::path::Path,
    id_a: &str,
    port: u16,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    let model_b = resolve_model("GEMMA4_E2B");
    let id_b = model_b.as_deref().map(model_id);

    // Dedicated hermetic HOME — registry JSON + runtime state out of the repo.
    let lc_home = std::env::temp_dir().join(format!("rmlx_e2e_lifecycle_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&lc_home);
    if let Err(e) = std::fs::create_dir_all(&lc_home) {
        return mk(Verdict::Fail, format!("create lifecycle home: {e}"));
    }
    // Track passing leg descriptions to roll up into the PASS detail.
    let mut legs: Vec<String> = Vec::new();

    // Helper to write a registry JSON with the given (id,path) entries.
    let write_registry = |entries: &[(&str, &std::path::Path)]| -> Result<PathBuf, String> {
        let models: Vec<serde_json::Value> = entries
            .iter()
            .map(|(id, p)| serde_json::json!({"id": id, "path": p.to_string_lossy()}))
            .collect();
        let cfg = serde_json::json!({ "models": models });
        let path = lc_home.join(format!("registry_{}.json", entries.len()));
        std::fs::write(&path, cfg.to_string()).map_err(|e| format!("write registry: {e}"))?;
        Ok(path)
    };

    // ── Leg (b): cap=2 registry → BOTH A and B resident. ─────────────────────
    // When B is present, prove the cap actually admits two models at once: a
    // cap=2 registry [A,B] eager-preloads both, so both report loaded:true.
    // Single-MLX discipline: this serve is fully torn down before the cap=1
    // serve below spawns (only one `rmlx serve` resident at a time).
    if let (Some(pb), Some(idb)) = (&model_b, &id_b) {
        let reg2 = match write_registry(&[(id_a, model_a), (idb.as_str(), pb.as_path())]) {
            Ok(p) => p,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&lc_home);
                return mk(Verdict::Fail, e);
            }
        };
        // Distinct port from the cap=1 phase to avoid TIME_WAIT re-bind races.
        let port2 = 18000 + ((port - 18000 + 1500) % 4000);
        claim_preflight();
        let cap2_guard = match spawn_serve_registry(&reg2, port2, 2, &lc_home) {
            Ok(g) => g,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&lc_home);
                return mk(Verdict::Fail, format!("cap=2 registry serve: {e}"));
            }
        };
        let a2 = model_loaded(port2, id_a);
        let b2 = model_loaded(port2, idb);
        // Tear down the cap=2 serve BEFORE any cap=1 spawn (single-MLX).
        drop(cap2_guard);
        claim_preflight();
        match (a2, b2) {
            (Ok(true), Ok(true)) => {
                legs.push(format!("(b) cap=2: both {id_a} + {idb} resident"));
            }
            (Ok(a), Ok(b)) => {
                let _ = std::fs::remove_dir_all(&lc_home);
                return mk(
                    Verdict::Fail,
                    format!(
                        "cap=2 did not keep both resident: A loaded={a}, B loaded={b} \
                         (expected both true)"
                    ),
                );
            }
            (Err(e), _) | (_, Err(e)) => {
                let _ = std::fs::remove_dir_all(&lc_home);
                return mk(Verdict::Fail, format!("cap=2 status read: {e}"));
            }
        }
    }

    // ── Legs (a)+(c)+(d): cap=1 registry with A (+B when present). ───────────
    // With cap=1 + eager preload, the LAST registry entry survives the preload.
    // Order [A, B] → B survives → A evicted. That proves leg (c) LRU eviction
    // directly out of the eager preload. With only A, A is resident (leg a).
    let entries_1: Vec<(&str, &std::path::Path)> = match (&model_b, &id_b) {
        (Some(pb), Some(idb)) => vec![(id_a, model_a), (idb.as_str(), pb.as_path())],
        _ => vec![(id_a, model_a)],
    };
    let reg1 = match write_registry(&entries_1) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&lc_home);
            return mk(Verdict::Fail, e);
        }
    };

    claim_preflight();
    let cap1_guard = match spawn_serve_registry(&reg1, port, 1, &lc_home) {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&lc_home);
            return mk(Verdict::Fail, format!("cap=1 registry serve: {e}"));
        }
    };

    // Two-model legs (a)/(c)/(d) when B is present; single-model leg (a) only
    // otherwise.
    if let Some(idb) = id_b.clone() {
        // Eager preload of [A,B] at cap=1 → B resident, A evicted: leg (c).
        let a_loaded = match model_loaded(port, id_a) {
            Ok(v) => v,
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("status A (cap1): {e}")),
        };
        let b_loaded = match model_loaded(port, &idb) {
            Ok(v) => v,
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("status B (cap1): {e}")),
        };
        // Defensive: if the eager order left A resident instead, force the swap
        // by explicitly loading B and re-checking — the LRU evict must fire.
        if a_loaded && !b_loaded {
            let lp = format!("/v1/models/{idb}/load");
            match http_post(port, &lp, "{}") {
                Ok((200, _)) => {}
                Ok((s, b)) => {
                    return fail_lc(
                        &cap1_guard,
                        &lc_home,
                        mk,
                        format!("explicit load B status {s}: {}", trunc(&b)),
                    )
                }
                Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("load B http: {e}")),
            }
        }
        // Re-read both: invariant is exactly-one-resident at cap=1, and it is B.
        let a_now = match model_loaded(port, id_a) {
            Ok(v) => v,
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("re-status A: {e}")),
        };
        let b_now = match model_loaded(port, &idb) {
            Ok(v) => v,
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("re-status B: {e}")),
        };
        if !b_now || a_now {
            return fail_lc(
                &cap1_guard,
                &lc_home,
                mk,
                format!(
                    "cap=1 LRU invariant violated: A loaded={a_now}, B loaded={b_now} \
                     (expected A=false, B=true)"
                ),
            );
        }
        legs.push(format!(
            "(a/c) cap=1 LRU: B={idb} resident, A={id_a} evicted"
        ));

        // Leg (d): explicit unload B → not loaded; 2nd unload → 404.
        let up = format!("/v1/models/{idb}/unload");
        match http_post(port, &up, "{}") {
            Ok((200, _)) => {}
            Ok((s, b)) => {
                return fail_lc(
                    &cap1_guard,
                    &lc_home,
                    mk,
                    format!("unload B status {s}: {}", trunc(&b)),
                )
            }
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("unload B http: {e}")),
        }
        match model_loaded(port, &idb) {
            Ok(false) => {}
            Ok(true) => {
                return fail_lc(
                    &cap1_guard,
                    &lc_home,
                    mk,
                    "unload B succeeded but status still loaded:true".to_owned(),
                )
            }
            Err(e) => {
                return fail_lc(
                    &cap1_guard,
                    &lc_home,
                    mk,
                    format!("post-unload status: {e}"),
                )
            }
        }
        // 2nd unload of an already-unloaded model → 404 (idempotent-evict
        // contract: the slot is empty, so the unload route reports not-loaded).
        match http_post(port, &up, "{}") {
            Ok((404, _)) => {}
            Ok((s, b)) => {
                return fail_lc(
                    &cap1_guard,
                    &lc_home,
                    mk,
                    format!("2nd unload B expected 404, got {s}: {}", trunc(&b)),
                )
            }
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("2nd unload http: {e}")),
        }
        legs.push(format!("(d) unload {idb}→loaded:false, 2nd unload→404"));
    } else {
        // Single-model subset: leg (a) only — A resident after preload.
        match model_loaded(port, id_a) {
            Ok(true) => legs.push(format!("(a) single-model: {id_a} resident")),
            Ok(false) => {
                return fail_lc(
                    &cap1_guard,
                    &lc_home,
                    mk,
                    format!("model A {id_a} not loaded after eager preload"),
                )
            }
            Err(e) => return fail_lc(&cap1_guard, &lc_home, mk, format!("status A: {e}")),
        }
    }

    // ── Leg (e): claim enforcement — 2nd serve on the HELD port is rejected. ──
    // cap1_guard still holds the claim for `port`. Start a SECOND `rmlx serve`
    // on the SAME port; it must hit ClaimError::AlreadyHeld and exit 11 WITHOUT
    // ever binding a competing Metal context. We do NOT claim_preflight here —
    // that would kill the holder and defeat the test. Use `.output()` so the
    // child is reaped; a non-zero exit is the asserted outcome.
    let mut rival = Command::new(rmlx_bin());
    rival
        .arg("serve")
        .arg("--model")
        .arg(model_a)
        .arg("--port")
        .arg(port.to_string())
        .env("RMLX_HOME", &lc_home)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let rival_out = rival.output();
    let claim_leg = match rival_out {
        Ok(out) => {
            let code = out.status.code();
            let stderr = String::from_utf8_lossy(&out.stderr);
            // Contract: ClaimError::AlreadyHeld → std::process::exit(11).
            // Any other exit code — including nonzero — means claim enforcement
            // did NOT fire correctly (e.g. wrong error path, wrong exit code, or
            // rival succeeded with exit 0).  Capture stderr only as context.
            if code == Some(11) {
                Ok(format!(
                    "(e) rival serve on held port {port} rejected (exit 11, claim error); \
                     stderr: {}",
                    trunc(&stderr)
                ))
            } else {
                Err(format!(
                    "rival serve on held port {port} exited {code:?} (expected 11 for \
                     ClaimError::AlreadyHeld) — claim enforcement broken; stderr: {}",
                    trunc(&stderr)
                ))
            }
        }
        Err(e) => Err(format!("spawn rival serve: {e}")),
    };
    // The claim holder is no longer needed; drop it (frees the claim + slot).
    drop(cap1_guard);
    claim_preflight();

    let claim_desc = match claim_leg {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&lc_home);
            return mk(Verdict::Fail, e);
        }
    };
    legs.push(claim_desc);

    let _ = std::fs::remove_dir_all(&lc_home);
    let skip_note = if id_b.is_none() {
        " [2-model legs SKIPPED: GEMMA4_E2B unresolved]"
    } else {
        ""
    };
    mk(
        Verdict::Pass,
        format!(
            "multi-model lifecycle proven: {}{skip_note}",
            legs.join("; ")
        ),
    )
}

/// Tear down a lifecycle serve guard + hermetic home, then build a FAIL result.
fn fail_lc(
    guard: &ServeGuard,
    home: &std::path::Path,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
    detail: String,
) -> CaseResult {
    // Guard is borrowed; the caller's `cap1_guard` Drop kills the child when the
    // borrow ends at function return. Settle the claim here so a follow-on case
    // is not blocked.
    let _ = std::fs::remove_dir_all(home);
    let _ = guard.port; // touch to keep the borrow explicit (no early drop)
    mk(Verdict::Fail, detail)
}

// ── Phase 2b: attention dispatch_fired (log scrape) ──────────────────────────

/// Spawn `rmlx serve --model <path> --kv-quant <kv> --log verbose --port <port>`
/// WITHOUT pinning `RUST_LOG` (so `--log verbose` drives the EnvFilter and the
/// per-dispatch `update_and_sdpa` trace spans reach the run jsonl). Blocks until
/// `/health` is green. `env -u RUST_LOG` semantics: we simply never set it.
fn spawn_serve_verbose(
    model: &std::path::Path,
    port: u16,
    kv_quant: &str,
    home: &std::path::Path,
) -> Result<ServeGuard, String> {
    spawn_serve_verbose_flags(
        model,
        port,
        home,
        &["--kv-quant".to_owned(), kv_quant.to_owned()],
    )
}

/// `spawn_serve_verbose` with caller-supplied extra serve flags (drafter flags,
/// `--max-ctx`, …). Verbose-logging serve, blocks until `/health` is green.
fn spawn_serve_verbose_flags(
    model: &std::path::Path,
    port: u16,
    home: &std::path::Path,
    extra: &[String],
) -> Result<ServeGuard, String> {
    let mut cmd = Command::new(rmlx_bin());
    cmd.arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .args(extra)
        .arg("--log")
        .arg("verbose")
        .env("RMLX_HOME", home)
        .env_remove("RUST_LOG") // verbose preset must win — no warn override
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd
        .spawn()
        .map_err(|e| format!("spawn rmlx serve --log verbose: {e}"))?;
    let mut guard = ServeGuard {
        child,
        port,
        home: home.to_path_buf(),
    };
    let deadline = Instant::now() + Duration::from_secs(120);
    #[allow(unused_assignments)]
    let mut last_seen = "no response yet".to_owned();
    loop {
        if let Some(status) = guard.child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!("rmlx serve --log verbose exited early: {status}"));
        }
        match http_get(port, "/health") {
            Ok((200, body)) if body.contains("\"ok\":true") || body.contains("\"ok\": true") => {
                return Ok(guard);
            }
            Ok((status, body)) => last_seen = format!("status {status}: {}", trunc(&body)),
            Err(e) => last_seen = format!("connect/read error: {e}"),
        }
        if Instant::now() > deadline {
            return Err(format!(
                "verbose serve /health never green within 120s (last: {last_seen})"
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Count `update_and_sdpa` attention-dispatch spans in the newest run jsonl
/// under `<home>/logs/`, tallied by the `path` span field. The jsonl line shape
/// (verified empirically) carries `span.path` + `span.name == "update_and_sdpa"`
/// on every trace event emitted inside that span. We count distinct events that
/// carry a `path` and group by its value.
///
/// **Honest path coverage:** on the warm-TTFT flow (Bonsai/Qwen3, head_dim=128)
/// dispatches resolve to `path="legacy"` or `path="flash"` (TurboFlash eligible).
/// The specialised fused-QK and planar kernels stay dormant on normal generate —
/// their dispatch counters are process-internal and are covered by in-crate tests,
/// not E2E.  PASS requires ≥1 span with *any* resolved `path` value.
///
/// **Log-flush caveat:** the scrape happens after a SIGKILL (no graceful shutdown).
/// A 200 ms settle is inserted before the call.  This relies on the generation
/// having enough decode steps that the interesting spans precede the SIGKILL-
/// truncated tail; a very short generation on a fast machine may yield 0 spans.
fn scrape_dispatch_paths(
    home: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, u64>, String> {
    let logs = home.join("logs");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir(&logs).map_err(|e| format!("read logs dir: {e}"))?;
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "jsonl") {
            if let Ok(meta) = e.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                        newest = Some((mtime, p));
                    }
                }
            }
        }
    }
    let Some((_, jsonl)) = newest else {
        return Err("no .jsonl under logs/".to_owned());
    };
    let body = std::fs::read_to_string(&jsonl).map_err(|e| format!("read jsonl: {e}"))?;
    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for line in body.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        // Count only events whose enclosing span is `update_and_sdpa` and carries
        // a `path` — that is the attention-dispatch decision point.
        let span = &v["span"];
        if span["name"].as_str() == Some("update_and_sdpa") {
            if let Some(p) = span["path"].as_str() {
                *counts.entry(p.to_owned()).or_insert(0) += 1;
            }
        }
    }
    Ok(counts)
}

/// **REAL** attention-dispatch proof (`dispatch_fired`).
///
/// HONEST observable signal (empirically established on Bonsai/Qwen3,
/// head_dim=128): the only externally-observable per-dispatch signal when
/// driving the real binary is the `path` field on the `update_and_sdpa` trace
/// span in the `--log verbose` run jsonl. The specialised attention KERNELS
/// (TurboFlash `flash`, generalised `fused_qk`, `planar_k_fused`) stay DORMANT
/// on the normal generate flow — warm-TTFT routes every decode step
/// through the bf16-K seed, so on Bonsai every dispatch records `path=legacy`
/// (plus PlanarK's `warm_ttft_bypass`). Their dispatch counters are
/// process-internal atomics with NO HTTP/metrics surface, covered by the
/// in-crate counter tests (`sparse_attn_dispatch.rs`,
/// `iso_fused_qk_msl_tests.rs`) — see `docs/E2E_TEST_PLAN.md`.
///
/// What THIS row proves externally: the attention-dispatch instrumentation
/// FIRES (≥ 1 `update_and_sdpa` span with a resolved `path`) for the chosen
/// codec AND the generation stays coherent. FAILS if no dispatch span appears
/// (instrumentation/decode broke) or the output is incoherent.
fn assert_dispatch_fired(
    case: &Case,
    model: &std::path::Path,
    id: &str,
    port: u16,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    let kv = case.kv_quant.as_deref().unwrap_or("k8v4");
    let home = std::env::temp_dir().join(format!("rmlx_e2e_dispatch_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    if let Err(e) = std::fs::create_dir_all(&home) {
        return mk(Verdict::Fail, format!("create dispatch home: {e}"));
    }

    claim_preflight();
    let guard = match spawn_serve_verbose(model, port, kv, &home) {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&home);
            return mk(Verdict::Fail, format!("verbose serve: {e}"));
        }
    };

    // Drive a short generation; coherence is the smoke-probe gate.
    let body = request_fixture("chat_basic", id).to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => {
            drop(guard);
            let _ = std::fs::remove_dir_all(&home);
            return mk(Verdict::Fail, format!("http: {e}"));
        }
    };
    if status != 200 {
        drop(guard);
        let _ = std::fs::remove_dir_all(&home);
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let text = openai_content(&resp).unwrap_or_default();
    let coherent = is_coherent(&text);

    // Drop the server (SIGKILL via ServeGuard::drop) and then wait briefly
    // before scraping the jsonl.  tracing_appender::non_blocking uses a
    // background thread that may not have flushed its in-flight buffer before
    // the process is killed.  There is no graceful /shutdown route and the
    // server does not handle SIGTERM, so we cannot flush cleanly.  A 200 ms
    // settle gives the OS time to flush kernel-buffered I/O after process exit.
    // NOTE: this relies on the generation being long enough (multi-hundred
    // decode steps) that the spans of interest precede the tail that could be
    // lost on SIGKILL; a very short generation on a fast machine may still lose
    // tail spans — acceptable because the assertion only requires ≥1 span.
    drop(guard);
    std::thread::sleep(Duration::from_millis(200));
    claim_preflight();

    let paths = scrape_dispatch_paths(&home);
    let _ = std::fs::remove_dir_all(&home);

    if !coherent {
        return mk(
            Verdict::Fail,
            format!("dispatch ran but output incoherent: {:?}", trunc(&text)),
        );
    }
    match paths {
        Ok(counts) if counts.values().copied().sum::<u64>() >= 1 => {
            let total: u64 = counts.values().copied().sum();
            let summary = counts
                .iter()
                .map(|(p, n)| format!("{p}={n}"))
                .collect::<Vec<_>>()
                .join(", ");
            mk(
                Verdict::Pass,
                format!(
                    "kv={kv}: {total} update_and_sdpa dispatches fired ({summary}); \
                     output coherent ({:?})",
                    trunc(&text)
                ),
            )
        }
        Ok(_) => mk(
            Verdict::Fail,
            format!(
                "no update_and_sdpa dispatch span in verbose jsonl (kv={kv}) — \
                 attention-dispatch instrumentation did not fire"
            ),
        ),
        Err(e) => mk(Verdict::Fail, format!("dispatch scrape: {e}")),
    }
}

/// **REAL** image-input proof (`image`).
///
/// Serves a vision-capable model, sends the `image_color` fixture (a solid-red
/// PNG data-URI + a one-word colour question), and asserts the model READ the
/// image: the answer must name the colour (`expect`, default "red",
/// case-insensitive). Proves the vision tower → soft-token scatter → generate
/// path end-to-end, not just that weights load. FAILS on non-200, an empty
/// answer, or a wrong/absent colour (text-only fallthrough).
fn assert_image(
    port: u16,
    fixture: &str,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
    case: &Case,
) -> CaseResult {
    let want = case
        .assert
        .expect
        .as_deref()
        .unwrap_or("red")
        .to_lowercase();
    let body = request_fixture(fixture, id).to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let text = openai_content(&resp).unwrap_or_default();
    if text.to_lowercase().contains(&want) {
        mk(
            Verdict::Pass,
            format!("vision read image colour {want:?}: {:?}", trunc(&text)),
        )
    } else {
        mk(
            Verdict::Fail,
            format!("image answer missing colour {want:?}: {:?}", trunc(&text)),
        )
    }
}

/// **REAL** tool-calling proof (`tool_call`).
///
/// Serves a tools-capable model, sends the `tool_weather` fixture (a get_weather
/// tool + a prompt that should trigger it), and asserts the model EMITTED a tool
/// call: `finish_reason == "tool_calls"` AND a `tool_calls[]` entry whose
/// function name matches `expect` (default "get_weather"). Proves the full
/// request → template → parse → emit surface, not just that `tools` is accepted.
fn assert_tool_call(
    port: u16,
    fixture: &str,
    id: &str,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
    case: &Case,
) -> CaseResult {
    let want = case.assert.expect.as_deref().unwrap_or("get_weather");
    let body = request_fixture(fixture, id).to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => return mk(Verdict::Fail, format!("http: {e}")),
    };
    if status != 200 {
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&resp) else {
        return mk(
            Verdict::Fail,
            format!("unparseable response: {}", trunc(&resp)),
        );
    };
    let choice = &v["choices"][0];
    let finish = choice["finish_reason"].as_str().unwrap_or("");
    let calls = &choice["message"]["tool_calls"];
    let names: Vec<&str> = calls
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| c["function"]["name"].as_str())
                .collect()
        })
        .unwrap_or_default();
    if finish == "tool_calls" && names.contains(&want) {
        let args = calls[0]["function"]["arguments"].as_str().unwrap_or("");
        mk(
            Verdict::Pass,
            format!("emitted tool_call {want:?} (finish=tool_calls), args={args}"),
        )
    } else {
        mk(
            Verdict::Fail,
            format!(
                "no {want:?} tool_call: finish={finish:?} names={names:?} content={:?}",
                trunc(choice["message"]["content"].as_str().unwrap_or_default())
            ),
        )
    }
}

/// **REAL** speculative-decoding proof (`spec_decode`).
///
/// Serves the verifier (`model`) with a real drafter (`--draft-model` resolved
/// from `case.draft_model`, `--draft-kind` from `case.draft_kind`) under verbose
/// logging, drives one generation, and scrapes the round-loop summary
/// (`<kind>_generate_greedy: done`) from the run jsonl. PASS = the round-loop
/// fired with `accept_rate > 0` AND the output is coherent (mentions the
/// expected token in `content` or, for a thinking model, `reasoning_content`).
/// Proves the drafter actually proposes accepted tokens end-to-end, not just
/// that it loads. SKIPs (clear reason) when the drafter snapshot is absent.
fn assert_spec_decode(
    case: &Case,
    model: &std::path::Path,
    id: &str,
    port: u16,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    let Some(spec) = case.draft_model.as_deref() else {
        return mk(
            Verdict::Fail,
            "spec_decode case missing `draft_model`".to_owned(),
        );
    };
    let Some(draft) = resolve_model(spec) else {
        return mk(
            Verdict::Skip,
            format!("drafter `{spec}` unresolved — snapshot absent"),
        );
    };
    let kind = case.draft_kind.as_deref().unwrap_or("mtp");
    // The expected coherence token (default "Paris") may land in the thinking
    // block on a reasoning model, so a generous budget lets it close the block.
    let want = case.assert.expect.as_deref().unwrap_or("Paris");

    let home = std::env::temp_dir().join(format!("rmlx_e2e_spec_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    if let Err(e) = std::fs::create_dir_all(&home) {
        return mk(Verdict::Fail, format!("create spec home: {e}"));
    }

    claim_preflight();
    let extra = vec![
        "--draft-model".to_owned(),
        draft.to_string_lossy().into_owned(),
        "--draft-kind".to_owned(),
        kind.to_owned(),
        "--max-ctx".to_owned(),
        "16384".to_owned(),
    ];
    let guard = match spawn_serve_verbose_flags(model, port, &home, &extra) {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&home);
            return mk(Verdict::Fail, format!("spec serve ({kind}): {e}"));
        }
    };

    // 300-token budget so a thinking model closes the reasoning block.
    let body = serde_json::json!({
        "model": id,
        "messages": [{"role": "user", "content": "In one sentence, what is the capital of France?"}],
        "max_tokens": 300, "temperature": 0.0, "seed": 0
    })
    .to_string();
    let (status, resp) = match http_post(port, "/v1/chat/completions", &body) {
        Ok(r) => r,
        Err(e) => {
            drop(guard);
            let _ = std::fs::remove_dir_all(&home);
            return mk(Verdict::Fail, format!("http: {e}"));
        }
    };
    if status != 200 {
        drop(guard);
        let _ = std::fs::remove_dir_all(&home);
        return mk(Verdict::Fail, format!("status {status}: {}", trunc(&resp)));
    }
    let content = openai_content(&resp).unwrap_or_default();
    let reasoning = openai_reasoning(&resp).unwrap_or_default();
    let mentions = content.contains(want) || reasoning.contains(want);
    let coherent = is_coherent(&content) || is_coherent(&reasoning);

    drop(guard);
    std::thread::sleep(Duration::from_millis(200));
    claim_preflight();

    let scraped = scrape_spec_accept(&home);
    let _ = std::fs::remove_dir_all(&home);

    if !coherent || !mentions {
        return mk(
            Verdict::Fail,
            format!(
                "spec_decode ({kind}) ran but output not coherent / missing {want:?}: \
                 content={:?} reasoning_tail={:?}",
                trunc(&content),
                trunc(&reasoning)
            ),
        );
    }
    match scraped {
        Ok((ar, rounds)) if ar > 0.0 => mk(
            Verdict::Pass,
            format!(
                "{kind} round-loop fired: accept_rate={ar:.3} over {rounds} rounds; \
                 output coherent (mentions {want:?})"
            ),
        ),
        Ok((ar, rounds)) => mk(
            Verdict::Fail,
            format!("{kind} round-loop fired but accept_rate={ar:.3} (0) over {rounds} rounds"),
        ),
        Err(e) => mk(
            Verdict::Fail,
            format!("{kind} round-loop summary not found in verbose jsonl: {e}"),
        ),
    }
}

/// Scrape the speculative round-loop summary (`<kind>_generate_greedy: done`)
/// from the newest run jsonl under `<home>/logs/`. Returns the LAST summary's
/// `(accept_rate, rounds)`. The fields shape (verified empirically):
/// `fields.message == "<kind>_generate_greedy: done"`, `fields.accept_rate`,
/// `fields.rounds`.
fn scrape_spec_accept(home: &std::path::Path) -> Result<(f64, u64), String> {
    let logs = home.join("logs");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir(&logs).map_err(|e| format!("read logs dir: {e}"))?;
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "jsonl") {
            if let Ok(meta) = e.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                        newest = Some((mtime, p));
                    }
                }
            }
        }
    }
    let Some((_, jsonl)) = newest else {
        return Err("no .jsonl under logs/".to_owned());
    };
    let body = std::fs::read_to_string(&jsonl).map_err(|e| format!("read jsonl: {e}"))?;
    let mut found: Option<(f64, u64)> = None;
    for line in body.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let fields = &v["fields"];
        let msg = fields["message"].as_str().unwrap_or_default();
        if msg.ends_with("_generate_greedy: done") {
            let ar = fields["accept_rate"].as_f64().unwrap_or(-1.0);
            let rounds = fields["rounds"].as_u64().unwrap_or(0);
            found = Some((ar, rounds));
        }
    }
    found.ok_or_else(|| "no `<kind>_generate_greedy: done` summary line".to_owned())
}

/// **REAL** arch-guard refusal proof (`serve_refused`).
///
/// Proves the per-arch KV invariant is a *working* feature: an illegal codec
/// for this architecture must be rejected at serve *resolve time* with the
/// documented non-zero exit, BEFORE the server binds `/health`. The codec is
/// supplied exactly like a normal serve case (`kv_quant` preset, or
/// `ctk`/`ctv` compose form), but here we expect the process to exit early
/// rather than go green.
///
/// `resolve_model_flags` (run between `load_config` and `load_model` on every
/// model command) loads `config.json`, resolves the codec against the arch
/// invariants, and `std::process::exit(78)` (EX_CONFIG) on a `ResolveError`
/// (e.g. `QwenMoeKBitsTooLow` for a K<8 codec, `QwenMoeTurboKRejected` for a
/// symmetric-K turbo codec on Qwen MoE). PASS requires the process to exit
/// with the expected code (default 78) within a short deadline; a server that
/// binds `/health` and stays up (the guard silently failed to fire) FAILS, as
/// does a wrong exit code.
fn assert_serve_refused(
    case: &Case,
    model: &std::path::Path,
    port: u16,
    mk: &dyn Fn(Verdict, String) -> CaseResult,
) -> CaseResult {
    let want: i32 = case
        .assert
        .expect
        .as_deref()
        .unwrap_or("78")
        .parse()
        .unwrap_or(78);

    let home = std::env::temp_dir().join(format!(
        "rmlx_e2e_refused_{}_{}",
        std::process::id(),
        case.id
    ));
    let _ = std::fs::remove_dir_all(&home);
    if let Err(e) = std::fs::create_dir_all(&home) {
        return mk(Verdict::Fail, format!("create refused home: {e}"));
    }

    claim_preflight();

    // Build the serve invocation the same way `run_serve_case` would, but
    // EXPECTING an early exit. --ctk/--ctv conflict with --kv-quant at the clap
    // layer (exit 2), so pass at most one form — exactly like the live cases.
    let mut cmd = Command::new(rmlx_bin());
    cmd.arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .env("RMLX_HOME", &home)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if case.ctk.is_some() || case.ctv.is_some() {
        if let Some(ctk) = &case.ctk {
            cmd.arg("--cache-type-k").arg(ctk);
        }
        if let Some(ctv) = &case.ctv {
            cmd.arg("--cache-type-v").arg(ctv);
        }
    } else if let Some(kv) = &case.kv_quant {
        cmd.arg("--kv-quant").arg(kv);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&home);
            return mk(Verdict::Fail, format!("spawn rmlx serve: {e}"));
        }
    };

    // The guard fires fast (config load + resolve, no model weights), so a few
    // seconds is plenty. Poll for early exit; if the process is still alive at
    // the deadline the guard did NOT fire — that is the failure we are proving
    // against. Kill + clear claim on every exit path so single-MLX stays clean.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().unwrap_or(-1);
                let _ = std::fs::remove_file(format!("/tmp/rmlx.{port}.claim"));
                let _ = std::fs::remove_dir_all(&home);
                claim_preflight();
                return if code == want {
                    mk(
                        Verdict::Pass,
                        format!("illegal codec rejected at resolve time: exit {code} (== {want})"),
                    )
                } else {
                    mk(
                        Verdict::Fail,
                        format!(
                            "illegal codec exited {code} != expected {want} \
                             — wrong exit (guard fired but with the wrong code, \
                             or a different early failure)"
                        ),
                    )
                };
            }
            Ok(None) => {
                if Instant::now() > deadline {
                    // Still alive → the guard did NOT fire. Kill it and FAIL.
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(format!("/tmp/rmlx.{port}.claim"));
                    let _ = std::fs::remove_dir_all(&home);
                    claim_preflight();
                    return mk(
                        Verdict::Fail,
                        "serve stayed alive past 30s — arch guard did NOT reject the \
                         illegal codec (resolve-time refusal feature broken)"
                            .to_owned(),
                    );
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_dir_all(&home);
                claim_preflight();
                return mk(Verdict::Fail, format!("try_wait: {e}"));
            }
        }
    }
}

fn trunc(s: &str) -> String {
    let s = s.trim();
    if s.len() <= 160 {
        return s.to_owned();
    }
    // Truncate on a char boundary at or below 160 bytes — never slice inside a
    // multibyte UTF-8 sequence (e.g. the em-dash in the cache-type table).
    let mut end = 160;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// The output dir the entry point reports.
pub fn report_dir() -> PathBuf {
    e2e_home().join("e2e")
}
