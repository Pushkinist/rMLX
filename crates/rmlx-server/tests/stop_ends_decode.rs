//! A stop-string match ends the decode of a real model, through the HTTP route.
//!
//! Three cells, each with its control in the same run: gemma-4-e2b
//! (`kv_h == 1`), Qwen3.8-27B-4bit (`kv_h == 4`, a hybrid recurrent model) and
//! gemma-4-e2b with its assistant drafter (the speculative round loop). The
//! prompt makes the model count to 1000 at temperature 0, so a reply that is
//! not stopped runs its whole budget of [`MAX_TOKENS`].
//!
//! The observables, two counts of the same steps from two sides:
//!
//! - `usage.completion_tokens`, which the route counts from the tokens it read;
//! - the `step_count` of the last `ItlSample`, which the engine writes from the
//!   steps its decode ran. The engine writes it before it sends the terminal
//!   item (`arch_generator.rs` and `speculative.rs` both push the sample, and
//!   then send the terminal item), so a read after the reply sees it with no
//!   wait.
//!
//! `k` is the step at which the reply first holds [`STOP`]: the smallest
//! `max_tokens` whose reply holds the string, found by bisection. The cell
//! asserts that the request at `k` runs exactly `k` engine steps by the
//! engine's own `step_count`, so `k` is the engine's step and not a count of
//! chunks of text, which would miss a step that adds no text. The ring of
//! samples is cleared before each request, so the one sample after it is that
//! request's.
//!
//! The stopped request (`stop: [STOP]`) must give `finish_reason == "stop"`, the
//! control's text up to the match, and both counts in `k..=k + SLACK`. The
//! control must give `finish_reason == "length"` and a count of [`MAX_TOKENS`]:
//! it runs its whole budget, so a stopped request that ran to the end would
//! fail the bound.
//!
//! The slack. At temperature 0 the request goes through two channels of 4 in
//! series. The engine sends each token on a channel of 4
//! (`engine/arch_generator.rs:573`, `engine/speculative.rs:671`), and the
//! replay envelope forwards on another channel of 4 (`retry.rs:414`). When the
//! route reads token `k` and sets the end flag, up to 4 tokens wait in the
//! replay channel and 1 waits to enter it (`k + 1..=k + 5`), up to 4 wait in
//! the engine channel (`k + 6..=k + 9`), and the engine callback is blocked in
//! the send of one more (`k + 10`). That callback is the one the verdict is
//! returned on. [`SLACK`] is 10: 2 * (4 + 1). The lag of the pipelined loop
//! (a token is emitted in the pass after the one that picked it) adds no step:
//! only an emitted token is a step.
//!
//! The prompt cache. The cache stores and matches whole blocks of 256 tokens
//! (`docs/PROMPT_CACHE.md`), so a prompt shorter than a block is never a hit, for
//! any request. The cell pads the user message past [`BLOCK_TOKENS`] tokens, puts
//! the counting instruction last, and asserts the prompt length the server
//! reports. The first request of each plain cell is the stopped one, so it is
//! the request that stores its prompt. The plain decode stores the
//! entry after the prefill and before the first token reaches the callback
//! (`gemma4/generate/mod.rs` stores `kv_caches` as the prefill left them), so
//! the length of the decode, and an end inside it, cannot change the entry. The
//! same request sent again must be a hit, with the same text and count. The
//! speculative loop stores no entry (`speculative/mod.rs`: the prompt-cache slot
//! count is dropped, and every request builds its own scratch cache stack): a
//! hit there would mean a stored entry. After an end inside a block
//! (`round_loop.rs`, the break on `emit.ended` comes before the rollback), the
//! verifier cache stays at the offset it had after the block was fed, past the
//! tokens the request emitted. Nothing stores or reuses that stack.
//!
//! What this cannot see: other architectures' loops (Laguna, BitNet, Qwen2 and
//! Qwen3-VL have no cell here), and the speculative drafters other than the
//! assistant. The source gate of `rmlx-models` holds that every loop calls the
//! one helper.
//!
//! Run: `make gpu-test CRATE=rmlx-server FILTER=stop_ends`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

mod stub_server;
mod thread_boundary;

use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use rmlx_mlx::Device;
use rmlx_server::{ArchGenerator, Generator, ItlStore, LoadedModel, ModelRegistry};
use serde_json::{json, Value};
use thread_boundary::{
    load_config, snapshot, speculative_generator, GEMMA4_E2B, GEMMA4_E2B_ASSISTANT, QWEN38_4BIT,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const PROMPT: &str = "Count from 1 to 1000, separated by commas.";

/// Filler that lifts the prompt past one block of the prompt cache.
const NOTES: &str = "Background notes for the task. ";

/// Tokens in one block of the prompt cache.
const BLOCK_TOKENS: u64 = 256;

/// The user message: the filler, then the instruction.
fn message() -> String {
    format!("{}\n\n{PROMPT}", NOTES.repeat(80))
}
const STOP: &str = "10,";
const MAX_TOKENS: u64 = 256;

/// Steps past the match that the stopped request may run: two channels of 4,
/// each with one send waiting. See the module doc.
const SLACK: u64 = 2 * (4 + 1);

/// Steps by which the control must outrun the bound.
const POWER: u64 = 64;

async fn post(port: u16, body: &Value) -> Value {
    let body = body.to_string();
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let text = String::from_utf8_lossy(&response).into_owned();
    assert!(text.starts_with("HTTP/1.1 200"), "reply: {text}");
    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    serde_json::from_str(&text[body_start..]).unwrap()
}

/// A server over one loaded generator.
struct Served {
    port: u16,
    model: String,
    generator: Arc<dyn Generator>,
    itl: ItlStore,
}

async fn serve(dir: &Path, generator: Arc<dyn Generator>) -> Served {
    let model = dir.file_name().unwrap().to_string_lossy().into_owned();
    let mut state = stub_server::not_ready_state(ModelRegistry::from_paths(&[dir.to_path_buf()]));
    state.device = Device::Gpu;
    let itl = Arc::clone(&state.itl_store);
    let now = Instant::now();
    state.slots.write().push(LoadedModel {
        id: model.clone(),
        model: Arc::clone(&generator),
        loaded_at: now,
        last_used: now,
        effective_max_ctx: usize::MAX,
        context_limits: None,
        decode_lease: Arc::new(AtomicUsize::new(0)),
        unload_handle: Arc::new(parking_lot::Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    let port = stub_server::start_server_with_state(state).await;
    Served {
        port,
        model,
        generator,
        itl,
    }
}

/// What one request showed from the route and from the engine.
struct Reply {
    text: String,
    finish: String,
    /// `usage.completion_tokens`: the tokens the route read.
    usage: u64,
    /// The `step_count` of the sample the engine wrote for this request.
    engine_steps: u64,
    /// `usage.prompt_tokens`.
    prompt_tokens: u64,
}

impl Served {
    async fn ask(&self, max_tokens: u64, stop: Option<&str>) -> Reply {
        let mut body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": message()}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        if let Some(stop) = stop {
            body["stop"] = json!([stop]);
        }
        self.itl.lock().clear();
        let reply = post(self.port, &body).await;
        let choice = &reply["choices"][0];
        let samples = self.itl.lock();
        assert_eq!(
            samples.len(),
            1,
            "the engine wrote one sample for the request"
        );
        let engine_steps = samples.back().map(|s| s.step_count).unwrap() as u64;
        drop(samples);
        Reply {
            text: choice["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            finish: choice["finish_reason"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            usage: reply["usage"]["completion_tokens"].as_u64().unwrap(),
            engine_steps,
            prompt_tokens: reply["usage"]["prompt_tokens"].as_u64().unwrap(),
        }
    }

    fn cache(&self) -> (u64, u64) {
        self.generator
            .cache_stats()
            .map_or((0, 0), |s| (s.hits, s.misses))
    }

    /// The smallest `max_tokens` whose reply holds [`STOP`]: the step at which
    /// the engine's text first holds it. `ceiling` is a count whose reply does.
    async fn step_of_the_match(&self, ceiling: u64) -> u64 {
        let (mut low, mut high) = (1, ceiling);
        while low < high {
            let mid = (low + high) / 2;
            if self.ask(mid, None).await.text.contains(STOP) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }
}

/// One cell: the stopped request first, so it stores its prompt, then the same
/// request again, then the control, then the step of the match.
async fn assert_a_stop_string_ends_the_decode(
    test: &str,
    dir: &Path,
    generator: Arc<dyn Generator>,
    stores_an_entry: bool,
) {
    let served = serve(dir, generator).await;

    let before = served.cache();
    let stopped = served.ask(MAX_TOKENS, Some(STOP)).await;
    let after_first = served.cache();
    let again = served.ask(MAX_TOKENS, Some(STOP)).await;
    let after_second = served.cache();

    let control = served.ask(MAX_TOKENS, None).await;
    assert_eq!(control.finish, "length", "{test}: the control ended early");
    assert_eq!(
        (control.usage, control.engine_steps),
        (MAX_TOKENS, MAX_TOKENS),
        "{test}: the control did not run its whole budget"
    );
    assert!(
        control.text.contains(STOP),
        "{test}: the control never said {STOP:?}"
    );

    let k = served.step_of_the_match(MAX_TOKENS).await;
    let at_k = served.ask(k, None).await;
    assert!(
        at_k.text.contains(STOP),
        "{test}: the reply at {k} steps lacks {STOP:?}"
    );
    assert_eq!(
        at_k.engine_steps, k,
        "{test}: the request at k ran another number of engine steps"
    );
    assert!(
        control.usage >= k + SLACK + POWER,
        "{test}: no power: the control ran {} steps and the match is at step {k}",
        control.usage
    );
    let cut = control.text.find(STOP).unwrap();
    for (what, reply) in [("stopped", &stopped), ("stopped again", &again)] {
        assert_eq!(reply.finish, "stop", "{test}: {what}: finish reason");
        assert_eq!(
            reply.text,
            control.text[..cut],
            "{test}: {what}: the text is the control's up to the match"
        );
        assert!(
            (k..=k + SLACK).contains(&reply.usage),
            "{test}: {what}: the route read {} steps; the match is at step {k}",
            reply.usage
        );
        assert!(
            (k..=k + SLACK).contains(&reply.engine_steps),
            "{test}: {what}: the engine ran {} steps; the match is at step {k}",
            reply.engine_steps
        );
        assert_eq!(
            reply.usage, reply.engine_steps,
            "{test}: {what}: the route and the engine count the same steps"
        );
    }
    assert_eq!(
        (again.text.as_str(), again.usage),
        (stopped.text.as_str(), stopped.usage),
        "{test}: the request sent again gave another text or count"
    );

    let lookups = |from: (u64, u64), to: (u64, u64)| (to.0 - from.0, to.1 - from.1);
    if stores_an_entry {
        assert!(
            stopped.prompt_tokens > BLOCK_TOKENS,
            "{test}: the prompt is {} tokens, under one block of the prompt cache, so no \
             request for it can hit",
            stopped.prompt_tokens
        );
        assert_eq!(
            lookups(after_first, after_second),
            (1, 0),
            "{test}: the stopped request sent again was not a hit"
        );
        assert_eq!(
            lookups(before, after_first),
            (0, 1),
            "{test}: the first stopped request was not a cold miss"
        );
    } else {
        assert_eq!(
            (before, after_first, after_second),
            (before, before, before),
            "{test}: a generator that keeps no entry counted a lookup"
        );
    }
}

fn plain(dir: &Path) -> Arc<dyn Generator> {
    let mut config = load_config(Device::Gpu, None);
    config.prompt_cache_slots = 1;
    Arc::new(
        ArchGenerator::from_snapshot(dir, &config, Arc::new(parking_lot::Mutex::new(())))
            .expect("ArchGenerator::from_snapshot"),
    )
}

#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_string_ends_the_decode_of_a_real_model() {
    let test = "a_stop_string_ends_the_decode_of_a_real_model";
    let Some(dir) = snapshot(test, GEMMA4_E2B.0, GEMMA4_E2B.1) else {
        return;
    };
    assert_a_stop_string_ends_the_decode(test, &dir, plain(&dir), true).await;
}

#[ignore = "requires Metal GPU and the Qwen3.8-27B-4bit snapshot; run with `make gpu-test`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_string_ends_the_decode_of_a_hybrid_model() {
    let test = "a_stop_string_ends_the_decode_of_a_hybrid_model";
    let Some(dir) = snapshot(test, QWEN38_4BIT.0, QWEN38_4BIT.1) else {
        return;
    };
    assert_a_stop_string_ends_the_decode(test, &dir, plain(&dir), true).await;
}

#[ignore = "requires Metal GPU and the gemma-4-e2b and assistant snapshots; run with `make gpu-test`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_string_ends_a_drafted_decode() {
    let test = "a_stop_string_ends_a_drafted_decode";
    let Some(verifier) = snapshot(test, GEMMA4_E2B.0, GEMMA4_E2B.1) else {
        return;
    };
    let Some(drafter) = snapshot(test, GEMMA4_E2B_ASSISTANT.0, GEMMA4_E2B_ASSISTANT.1) else {
        return;
    };
    let generator = (speculative_generator(verifier.clone(), drafter, Device::Gpu))();
    assert_a_stop_string_ends_the_decode(test, &verifier, generator, false).await;
}
