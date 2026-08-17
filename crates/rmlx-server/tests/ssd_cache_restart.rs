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
//! ## Single-MLX-process discipline (CLAUDE.md hard rule 8)
//!
//! Before each `rmlx serve` spawn the test runs the standard preflight
//! (`pkill -f "rmlx serve"`, clear `/tmp/rmlx.*.claim`) and on teardown kills
//! its own child + clears the claim. Only one MLX process is alive at a time.

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

// ── Process / claim helpers ─────────────────────────────────────────────────

/// CLAUDE.md hard rule 8 preflight: ensure no competing MLX process holds the
/// Metal context, and clear any stale claim file before we spawn our own.
fn preflight_claim() {
    for pat in ["rmlx serve", "mlx_lm", "paroquant"] {
        let _ = Command::new("pkill").arg("-f").arg(pat).status();
    }
    std::thread::sleep(Duration::from_secs(5));
    // rm -f /tmp/rmlx.*.claim
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("rmlx.") && name.ends_with(".claim") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

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
fn spawn_serve(bin: &Path, model: &str, port: u16, rmlx_home: &Path) -> Child {
    Command::new(bin)
        .arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .arg("--kv-ssd-cache-gb")
        .arg("1")
        .arg("--prompt-cache-slots")
        .arg("1")
        .env("RMLX_HOME", rmlx_home)
        // info baseline + debug on the SSD spill/hydrate modules so a
        // `--nocapture` run shows the spill drain + hydrate events live.
        .env(
            "RUST_LOG",
            "rmlx=info,rmlx_kv_ssd::spill=debug,rmlx_kv_ssd::ssd_tier=debug",
        )
        // Inherit stderr so a `--nocapture` run shows the server's tracing
        // output live (model load, spill drain, hydrate) for debugging.
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn rmlx serve")
}

/// Best-effort kill + reap of a serve child, then clear the claim.
fn teardown(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
    preflight_claim();
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

// ── SSD-index disk verification (rusqlite, dev-dep) ──────────────────────────

/// Walk `<RMLX_HOME>/cache/kv/*/` and return (kvb_file_count, index_row_count)
/// summed across every namespace directory found. Used to prove A spilled.
fn ssd_disk_state(rmlx_home: &Path) -> (usize, usize) {
    let kv_root = rmlx_home.join("cache").join("kv");
    let mut kvb_files = 0usize;
    let mut index_rows = 0usize;
    let Ok(namespaces) = std::fs::read_dir(&kv_root) else {
        return (0, 0);
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
        let db = ns_dir.join("index.db");
        if db.exists() {
            if let Ok(conn) = rusqlite::Connection::open(&db) {
                if let Ok(n) =
                    conn.query_row("SELECT COUNT(*) FROM kv_blocks", [], |r| r.get::<_, i64>(0))
                {
                    index_rows += n as usize;
                }
            }
        }
    }
    (kvb_files, index_rows)
}

// ── The test ─────────────────────────────────────────────────────────────────

// The Metal context this test needs belongs to the `rmlx serve` CHILD, not to
// the test binary — this process only speaks HTTP to it — so no source shape
// here can express the device and the `#[ignore]` gate is told rather than
// shown. It stays out of `scripts/run_gpu_tests.sh` on top of that: that runner
// instruments the test process, which dispatches no Metal at all; the test also
// needs `cargo build -p rmlx-cli` first (`cargo test --tests` does not build the
// binary), `pkill`s every MLX process on the machine, and spends two 180 s
// readiness waits. The same spill -> restart -> hydrate chain, over the same two
// prompts, is covered by `make e2e` phase 2a
// (`crates/rmlx-cli/tests/e2e/runner.rs`).
// gpu-test-gate: metal-unscanned  Metal belongs to the spawned serve process.
#[ignore = "integration: requires RMLX_TEST_MODEL + a real rmlx serve process (Metal)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
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

    // A fixed, unlikely-to-collide port for both phases (same RMLX_HOME, single
    // process at a time, so reuse is fine).
    let port: u16 = 8731;

    // ── Phase 1: populate + spill ───────────────────────────────────────────
    preflight_claim();
    let mut child = spawn_serve(&bin, &model, port, &rmlx_home);
    assert!(
        wait_ready(port, Duration::from_secs(180)).await,
        "phase-1 server did not become ready within 180s"
    );

    // Prompt A — cold prefill, record TTFT.
    let (sa, _ba, ttft_cold) = chat_ttft(port, &model_id, PROMPT_A)
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
    let mut disk = (0usize, 0usize);
    for _ in 0..40 {
        disk = ssd_disk_state(&rmlx_home);
        if disk.0 >= 1 && disk.1 >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    eprintln!(
        "phase-1: ttft_cold={ttft_cold:.1}ms  kvb_files={}  index_rows={}",
        disk.0, disk.1
    );
    assert!(
        disk.0 >= 1,
        "expected >=1 spilled .kvb file under {}/cache/kv after eviction, found {}",
        rmlx_home.display(),
        disk.0
    );
    assert!(
        disk.1 >= 1,
        "expected >=1 SSD index row after eviction, found {}",
        disk.1
    );

    // ── Restart boundary: kill, clear claim, restart same model + RMLX_HOME ──
    let _ = child.kill();
    let _ = child.wait();
    preflight_claim();

    // ── Phase 2: restart + hydrate ──────────────────────────────────────────
    let child2 = spawn_serve(&bin, &model, port, &rmlx_home);
    if !wait_ready(port, Duration::from_secs(180)).await {
        teardown(child2);
        panic!("phase-2 server did not become ready within 180s (see inherited stderr above)");
    }

    // Spilled row must have survived startup prune+evict.
    let after_restart = ssd_disk_state(&rmlx_home);
    assert!(
        after_restart.1 >= 1,
        "SSD index row did not survive restart: rows={}",
        after_restart.1
    );

    // Prompt A again — RAM is empty (fresh process) so this is a RAM miss that
    // must hydrate from the surviving .kvb. Record warm TTFT.
    let (sa2, _ba2, ttft_warm) = chat_ttft(port, &model_id, PROMPT_A)
        .await
        .expect("phase-2 prompt A request");
    assert_eq!(sa2, 200, "phase-2 prompt A must return 200");

    let ssd_hits = scrape_ssd_hits(port).await.expect("scrape /metrics");

    // Part 4 (step 2 carry-over): scrape /metrics and collect the new
    // SSD-tier Prometheus data. We collect the body here and defer assertions
    // until AFTER teardown so the GPU is freed even on assertion failure.
    let metrics_body = http(port, "GET", "/metrics", None)
        .await
        .expect("GET /metrics for SSD assertions")
        .1;

    eprintln!(
        "phase-2: ttft_warm={ttft_warm:.1}ms  ssd_hits={ssd_hits}  \
         (cold={ttft_cold:.1}ms, threshold warm<{:.0}%*cold)",
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
    teardown(child2);

    // ── Cross-restart assertions ────────────────────────────────────────────
    assert!(
        ssd_hits >= 1,
        "expected cross-restart prompt_cache_ssd_hits >= 1 (RAM miss served from SSD), got {ssd_hits}"
    );
    assert!(
        ttft_warm < ttft_cold * TTFT_DROP_THRESHOLD,
        "expected warm TTFT ({ttft_warm:.1}ms) materially below cold ({ttft_cold:.1}ms): \
         warm must be < {:.0}% of cold ({:.1}ms)",
        TTFT_DROP_THRESHOLD * 100.0,
        ttft_cold * TTFT_DROP_THRESHOLD
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
