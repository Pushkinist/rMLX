//! A request that is gone frees the GPU, on real models through the server's
//! generators and the speech route.
//!
//! The seam of a chat request is `Generator::generate`. The test drops the
//! reply stream, as the 408 of the timeout layer, a closed connection and the
//! server shutdown drop the request future. It holds the one GPU permit of a
//! server in the request, and reads how long the permit stays held after the
//! drop.
//!
//! Each case has its control in the same run: the same request with no drop.
//! The permit may stay held for a quarter of the time the control runs on,
//! inside [`FLOOR`] and [`CAP`]. A control that is shorter than two such
//! bounds cannot tell a job that stopped from a job that ran to its end, and
//! the case fails on that.
//!
//! What every stop holds, in a decode, in a prefill and in a synthesis:
//!
//! - the permit is free inside the bound after the drop;
//! - the stopped request writes no metrics row;
//! - the job logs one stop event, which is one tracing event at `INFO` or
//!   `DEBUG` with the message [`STOP_EVENT`], and no event at `ERROR`;
//! - a chat request writes no request record, and on a speculative pair its
//!   round lines are whole rounds in sequence;
//! - the request after the stopped one gives what a run before the stop gave.
//!
//! The tokens of a request that is not stopped are held by the golden-token
//! and equivalence tests of `rmlx-models`, not here: two runs of one binary
//! cannot show that a change left them alone.
//!
//! The decode cells of Qwen2, BitNet and Qwen3-VL run with one prompt-cache
//! slot, and the image cell with an encoder cache too. Each cell reads the
//! branch of the prompt cache that every request took, from the counters of the
//! cache, and asserts it: the cold run is a miss, and the control, the stopped
//! request and the request after it take the exact-hit decode, so the request
//! after the stop reads what the stopped one touched. A last request with no
//! slot is stopped in the decode that follows a prefill. An image prompt
//! bypasses the prompt cache, and the encoder cache is a miss and then three
//! hits.
//!
//! The inline prefills that a case stops: Qwen2 (the prefill case) and Qwen3-VL
//! (the image prefill, before its first token).
//!
//! Not covered here, so only the CPU tests of `rmlx-models` and the source
//! rules read them: the count of steps a loop ran after the stop; the tail
//! prefill of an entry that the SSD tier restored (this harness attaches no SSD
//! tier); the Laguna decode loop and prefill (no snapshot is named for them);
//! the inline prefill of BitNet; the stop between two blocks of the Qwen3-VL
//! vision tower (a drop lands after the tower of the test image); and the stop
//! between two images of Gemma 4 and Gemma 3.
//!
//! Run: `make gpu-test CRATE=rmlx-server FILTER=gone`.

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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use rmlx_metrics::events::EventRecorder;
use rmlx_mlx::Device;
use rmlx_models::multimodal_cache::MultimodalCache;
use rmlx_server::engine::{admit_request, Admission};
use rmlx_server::metrics_drainer::spawn_drainer;
use rmlx_server::{ArchGenerator, DrainerHandle, GenerationRequest, Generator, ModelRegistry};
use thread_boundary::round_stream::{round_events, CapturedEvent, RoundStreamRecorder};
use thread_boundary::{
    image_request, load_config, long_prompt, one_thread_pool, record_on, release_cached_buffers,
    snapshot, speculative_generator, text_request, Load, BITNET, BONSAI_8B, GEMMA4_E2B,
    GEMMA4_E2B_ASSISTANT, N_TOKENS, ORNITH_9B, QWEN36_DFLASH, QWEN36_EAGLE3, QWEN36_MOE,
    QWEN38_4BIT, QWEN38_DFLASH2, QWEN38_MTP_4BIT, QWEN38_MXFP8, QWEN3_VL, READERLM_V2, STEP_LIMIT,
    TERNARY_BONSAI_2_27B, TTS, TTS_CODEC,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Semaphore;

/// The least bound on the time the permit stays held after its request is
/// gone: the return of the job and the wake of the next waiter.
const FLOOR: Duration = Duration::from_millis(500);

/// The most that bound can be, whatever the control took.
const CAP: Duration = Duration::from_secs(60);

/// The token budget of a request that is dropped in its decode.
const LONG_REPLY: u32 = 1024;

/// Tokens read from the reply before the drop, so the drop is in the decode.
const READ_BEFORE_THE_DROP: usize = 4;

/// The message of the one event a stopped job logs. It carries the field
/// `emitted`: the tokens, or the frames, that the job had made.
const STOP_EVENT: &str = "the request is gone; the generation stopped";

/// The message of the event that the image prefill and the image decode of
/// Qwen3-VL log when a request enters them.
const IMAGE_PATH_EVENT: &str = "qwen3_vl_moe::generate_image";

/// How long the drainer and the recorder have to write the rows of a request.
const ROWS_WAIT: Duration = Duration::from_secs(3);

/// How long the permit may stay held after the drop, for a request whose
/// control ran on for `control`.
fn bound(control: Duration) -> Duration {
    (control / 4).clamp(FLOOR, CAP)
}

/// Fail when `control` cannot tell a job that stopped from one that ended.
fn assert_power(test: &str, what: &str, control: Duration) {
    assert!(
        control >= 2 * bound(control),
        "{test}: no power: {what} took {control:?} with no drop, so a job that runs to its \
         end frees the permit inside the bound of {:?}",
        bound(control)
    );
}

/// The first turn of a prefill case, in tokens. The turn doubles until its
/// cold prefill takes [`PREFILL_CONTROL`].
const FIRST_TURN_TOKENS: usize = 12_288;

/// The longest turn a prefill case doubles to. The MLX buffer cache keeps a
/// buffer for each prompt size it saw, so a turn with no bound grows the host
/// memory by tens of GiB on a host fast enough to need it.
const MOST_TURN_TOKENS: usize = 4 * FIRST_TURN_TOKENS;

/// The least time a cold prefill takes in a prefill case: four floors, so the
/// bound of [`bound`] is a quarter of it and the case has power.
const PREFILL_CONTROL: Duration = FLOOR.saturating_mul(4);

/// Tokens that a prefill case keeps free for a reply after its two turns.
const REPLY_ROOM: usize = 4096;

/// The byte budget of the encoder cache of a case that has one.
const ENCODER_CACHE_BYTES: usize = 256 << 20;

/// The rows of the metrics DBs at `dbs`.
fn rows(dbs: &[&Path]) -> i64 {
    dbs.iter()
        .filter_map(|db| rusqlite::Connection::open(db).ok())
        .flat_map(|conn| {
            ["observations", "events"].map(|table| {
                conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                    .unwrap_or(0)
            })
        })
        .sum()
}

/// The rows once more than `before` are written, or after [`ROWS_WAIT`].
async fn rows_after(dbs: &[&Path], before: i64) -> i64 {
    let waited = Instant::now();
    while rows(dbs) <= before && waited.elapsed() < ROWS_WAIT {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    rows(dbs)
}

/// The request records among `events`: the one line a speculative request
/// writes when it ends.
fn request_records(events: &[CapturedEvent]) -> usize {
    events
        .iter()
        .filter(|e| e.field("decode_config").is_some())
        .count()
}

fn number(event: &CapturedEvent, field: &str) -> u64 {
    event
        .field(field)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("a round line with no `{field}`: {event:?}"))
}

/// What the events of a stopped request must say.
fn assert_the_events_of_a_stop(test: &str, events: &[CapturedEvent], drafted: bool) {
    let stops: Vec<&CapturedEvent> = events.iter().filter(|e| e.message == STOP_EVENT).collect();
    assert_eq!(stops.len(), 1, "{test}: the stop events of a stopped job");
    for stop in stops {
        assert!(
            [tracing::Level::INFO, tracing::Level::DEBUG].contains(&stop.level),
            "{test}: the stop event is at {}",
            stop.level
        );
        assert!(
            stop.field("emitted").is_some(),
            "{test}: the stop event does not say how far the job got"
        );
    }
    let errors: Vec<&str> = events
        .iter()
        .filter(|e| e.level == tracing::Level::ERROR)
        .map(|e| e.message.as_str())
        .collect();
    assert!(
        errors.is_empty(),
        "{test}: a stop logged as an error: {errors:?}"
    );
    assert_eq!(
        request_records(events),
        0,
        "{test}: a stopped request wrote the record of a request that ended"
    );
    if !drafted {
        return;
    }
    let mut emitted = 0;
    for (at, round) in round_events(events).iter().enumerate() {
        assert_eq!(
            number(round, "round"),
            at as u64 + 1,
            "{test}: the round lines of a stopped request are not a sequence"
        );
        assert!(
            number(round, "accept") <= number(round, "num_draft"),
            "{test}: {round:?}"
        );
        let total = number(round, "emitted_total");
        assert!(total > emitted, "{test}: a round line that emitted nothing");
        emitted = total;
    }
}

/// The tokens of one reply that ran to its end, and its times.
struct Reply {
    ids: Vec<u32>,
    after_the_first_tokens: Duration,
    to_the_first_token: Duration,
}

/// Run `req` to its end. Each item of the reply has [`STEP_LIMIT`] to come.
async fn whole(test: &str, generator: &Arc<dyn Generator>, req: GenerationRequest) -> Reply {
    let started = Instant::now();
    let mut stream = generator.generate(req);
    let mut ids = Vec::new();
    let mut first = None;
    let mut read = None;
    loop {
        let item = tokio::time::timeout(STEP_LIMIT, stream.next())
            .await
            .unwrap_or_else(|_| panic!("{test}: the reply stalled after {} tokens", ids.len()));
        let Some(item) = item else {
            break;
        };
        let token = item.unwrap_or_else(|e| panic!("{test}: the request failed: {e}"));
        if token.done {
            continue;
        }
        ids.push(token.token_id);
        first.get_or_insert_with(|| started.elapsed());
        if ids.len() == READ_BEFORE_THE_DROP {
            read = Some(Instant::now());
        }
    }
    Reply {
        ids,
        after_the_first_tokens: read.map_or(Duration::ZERO, |at| at.elapsed()),
        to_the_first_token: first.unwrap_or_default(),
    }
}

/// Start `req` under the one GPU permit, drop its reply after `read` tokens,
/// or after `wait` when `read` is 0, and return how long the permit stayed
/// held after the drop.
async fn gone(
    test: &str,
    generator: &Arc<dyn Generator>,
    mut req: GenerationRequest,
    read: usize,
    wait: Duration,
) -> Duration {
    let queue = Arc::new(Semaphore::new(1));
    let pending = Arc::new(AtomicUsize::new(0));
    let Admission::Admitted { guard, .. } = admit_request(&queue, &pending, 64).await else {
        panic!("{test}: an empty queue admits");
    };
    req.gpu_admission = Some(guard);
    let mut stream = generator.generate(req);
    for _ in 0..read {
        let item = tokio::time::timeout(STEP_LIMIT, stream.next()).await;
        assert!(
            matches!(item, Ok(Some(Ok(_)))),
            "{test}: the reply ended before the drop"
        );
    }
    if read == 0 {
        tokio::time::sleep(wait).await;
    }
    assert_eq!(
        queue.available_permits(),
        0,
        "{test}: the job ended before the drop"
    );
    drop(stream);
    let dropped = Instant::now();
    let permit = tokio::time::timeout(STEP_LIMIT, queue.acquire()).await;
    assert!(
        matches!(permit, Ok(Ok(_))),
        "{test}: the permit never came back"
    );
    assert_eq!(pending.load(Ordering::Acquire), 0, "{test}");
    dropped.elapsed()
}

/// One loaded model on a pool of one blocking thread, with every event of that
/// thread and the metrics rows of its requests.
struct Served {
    pool: tokio::runtime::Runtime,
    generator: Arc<dyn Generator>,
    events: Arc<RoundStreamRecorder>,
    drainer: DrainerHandle,
    recorder: Arc<EventRecorder>,
    dbs: [PathBuf; 2],
    _dir: tempfile::TempDir,
}

impl Served {
    /// Load a model. The buffers of the models that earlier tests of this
    /// binary dropped are freed first.
    fn new(load: &Load) -> Self {
        release_cached_buffers();
        let pool = one_thread_pool();
        let load = Arc::clone(load);
        let generator = pool
            .block_on(pool.spawn_blocking(move || load()))
            .expect("the load does not panic");
        let events = RoundStreamRecorder::new();
        record_on(&pool, &events);
        let dir = tempfile::tempdir().expect("a temporary directory");
        let dbs = [dir.path().join("runs.db"), dir.path().join("events.db")];
        let drainer = {
            let _in_pool = pool.enter();
            spawn_drainer(dbs[0].clone(), std::collections::BTreeMap::new())
        };
        let recorder = Arc::new(EventRecorder::open_at(&dbs[1], "gone").expect("an events DB"));
        Self {
            pool,
            generator,
            events,
            drainer,
            recorder,
            dbs,
            _dir: dir,
        }
    }

    /// `req` with the metrics sinks of this server.
    fn measured(&self, mut req: GenerationRequest) -> GenerationRequest {
        req.metrics_drainer = Some(self.drainer.clone());
        req.event_recorder = Some(Arc::clone(&self.recorder));
        req
    }

    fn dbs(&self) -> [&Path; 2] {
        [self.dbs[0].as_path(), self.dbs[1].as_path()]
    }

    /// Run `req` to its end. A request that ends must write a metrics row, or
    /// the check that a stopped one writes none has no power.
    async fn ended(&self, test: &str, req: GenerationRequest) -> Reply {
        let before = rows(&self.dbs());
        let reply = whole(test, &self.generator, self.measured(req)).await;
        assert!(
            rows_after(&self.dbs(), before).await > before,
            "{test}: no power: a request that ended wrote no metrics row"
        );
        reply
    }

    /// Stop `req` as [`gone`] does, and hold what its events and its metrics
    /// rows must say. Returns how long the permit stayed held.
    async fn stopped(
        &self,
        test: &str,
        req: GenerationRequest,
        read: usize,
        wait: Duration,
        drafted: bool,
    ) -> Duration {
        let (mark, before) = (self.events.events().len(), rows(&self.dbs()));
        let held = gone(test, &self.generator, self.measured(req), read, wait).await;
        tokio::time::sleep(ROWS_WAIT).await;
        assert_eq!(
            rows(&self.dbs()),
            before,
            "{test}: a stopped request wrote a metrics row"
        );
        assert_the_events_of_a_stop(test, &self.events.events().split_off(mark), drafted);
        held
    }
}

/// The counters of the prompt cache of a model: hits, misses and partial hits.
/// A cache that is replaced, when a request names another number of slots,
/// starts again at 0.
type Counters = (u64, u64, u64);

fn counters(generator: &Arc<dyn Generator>) -> Counters {
    generator
        .cache_stats()
        .map_or((0, 0, 0), |s| (s.hits, s.misses, s.partial_hits))
}

/// What one request did with the prompt cache of its model, read from the
/// counters of the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Branch {
    /// The lookup found no entry: the request prefilled the whole prompt.
    Miss,
    /// The lookup found the whole prompt: no prefill, and the stored first
    /// token is replayed.
    Exact,
    /// The lookup found a prefix of the prompt.
    Prefix,
    /// No lookup: the prompt cache was not consulted.
    Bypass,
}

fn branch_between(before: Counters, now: Counters) -> Branch {
    let delta = (
        now.0.checked_sub(before.0),
        now.1.checked_sub(before.1),
        now.2.checked_sub(before.2),
    );
    match delta {
        (Some(0), Some(0), Some(0)) => Branch::Bypass,
        (Some(0), Some(1), Some(0)) => Branch::Miss,
        (Some(1), Some(0), Some(0)) => Branch::Exact,
        (Some(1), Some(0), Some(1)) => Branch::Prefix,
        _ => panic!("the counters of the prompt cache went from {before:?} to {now:?}"),
    }
}

/// How a decode case uses the prompt cache of its model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reuse {
    /// The model has no slot, so every request prefills.
    NoSlot,
    /// The model has one slot. A request with no slot first empties the cache,
    /// so the cold run is a miss and stores its prompt; the control, the
    /// stopped request and the request after it find that whole prompt and run
    /// the decode that follows an exact hit. A last request with no slot is
    /// stopped in the decode that follows a prefill.
    OneSlot,
    /// The requests carry an image, and an image prompt bypasses the prompt
    /// cache.
    Bypass,
}

/// The branch of each request of a case, in order: the cold run, the control,
/// the stopped request and the request after it.
fn branches_of(reuse: Reuse) -> Option<[Branch; 4]> {
    match reuse {
        Reuse::NoSlot => None,
        Reuse::OneSlot => Some([Branch::Miss, Branch::Exact, Branch::Exact, Branch::Exact]),
        Reuse::Bypass => Some([Branch::Bypass; 4]),
    }
}

/// The prompt-cache counters of one case, read after each request.
struct CacheWatch<'a> {
    test: &'a str,
    generator: &'a Arc<dyn Generator>,
    reuse: Reuse,
    base: Counters,
}

impl<'a> CacheWatch<'a> {
    fn new(test: &'a str, generator: &'a Arc<dyn Generator>, reuse: Reuse) -> Self {
        Self {
            test,
            generator,
            reuse,
            base: counters(generator),
        }
    }

    /// The next request runs on a cache that was just replaced.
    fn restart(&mut self) {
        self.base = (0, 0, 0);
    }

    /// Read what the request before did with the prompt cache and hold it to
    /// `want`.
    fn expect(&mut self, request: &str, want: Branch) {
        let now = counters(self.generator);
        let seen = branch_between(self.base, now);
        assert_eq!(
            seen, want,
            "{}: the {request} took the {seen:?} branch of the prompt cache",
            self.test
        );
        self.base = now;
    }

    /// The request number `at` of the four of a case.
    fn expect_request(&mut self, at: usize, request: &str) {
        if let Some(branches) = branches_of(self.reuse) {
            self.expect(request, branches[at]);
        }
    }
}

/// A generator over `dir` with `slots` prompt-cache slots and, if given, an
/// encoder cache.
fn generator_with(dir: PathBuf, slots: usize, mm_cache: Option<Arc<MultimodalCache>>) -> Load {
    Arc::new(move || {
        let mut config = load_config(Device::Gpu, mm_cache.clone());
        config.prompt_cache_slots = slots;
        let generator: Arc<dyn Generator> = Arc::new(
            ArchGenerator::from_snapshot(&dir, &config, Arc::new(parking_lot::Mutex::new(())))
                .expect("ArchGenerator::from_snapshot"),
        );
        generator
    })
}

/// The decode of a request that is gone stops inside the bound, and the model
/// then serves the next request as it served one before the stop. Returns the
/// events of the requests of the case.
///
/// With a slot (see [`Reuse`]) the request after the stop reads the entry that
/// the stopped request touched, so the equality with the cold run holds for a
/// cache that was shared and not for a model that shared nothing.
fn assert_a_request_gone_in_its_decode_frees_the_gpu(
    test: &str,
    load: &Load,
    request: &dyn Fn(u32) -> GenerationRequest,
    drafted: bool,
    reuse: Reuse,
) -> Vec<CapturedEvent> {
    let served = Served::new(load);
    served.pool.block_on(async {
        let mut cache = CacheWatch::new(test, &served.generator, reuse);
        let mut reference = None;
        if reuse == Reuse::OneSlot {
            let mut no_slot = request(N_TOKENS);
            no_slot.effective_prompt_cache_slots = Some(0);
            reference = Some(whole(test, &served.generator, no_slot).await.ids);
            cache.restart();
        }
        let cold = whole(test, &served.generator, request(N_TOKENS)).await.ids;
        assert!(!cold.is_empty(), "{test}: the model decoded nothing");
        cache.expect_request(0, "cold run");
        if let Some(reference) = reference {
            assert_eq!(
                cold, reference,
                "{test}: a run with a slot gave other tokens than a run with none"
            );
        }

        let mark = served.events.events().len();
        let control = served.ended(test, request(LONG_REPLY)).await;
        cache.expect_request(1, "control");
        assert_power(
            test,
            "the reply after its first tokens",
            control.after_the_first_tokens,
        );
        if drafted {
            let control = served.events.events().split_off(mark);
            assert_eq!(request_records(&control), 1, "{test}: the control record");
            assert!(!round_events(&control).is_empty(), "{test}: no round ran");
        }

        let held = served
            .stopped(
                test,
                request(LONG_REPLY),
                READ_BEFORE_THE_DROP,
                Duration::ZERO,
                drafted,
            )
            .await;
        cache.expect_request(2, "stopped request");
        assert!(
            held <= bound(control.after_the_first_tokens),
            "{test}: the permit stayed held {held:?} after the request was gone; the reply \
             with no drop runs on for {:?}",
            control.after_the_first_tokens
        );

        let after = whole(test, &served.generator, request(N_TOKENS)).await.ids;
        cache.expect_request(3, "request after the stop");
        assert_eq!(
            after, cold,
            "{test}: the request after a stopped one emitted other tokens than the cold run"
        );

        if reuse == Reuse::OneSlot {
            let mut no_slot = request(LONG_REPLY);
            no_slot.effective_prompt_cache_slots = Some(0);
            let held = served
                .stopped(test, no_slot, READ_BEFORE_THE_DROP, Duration::ZERO, false)
                .await;
            cache.restart();
            cache.expect("stopped request with no slot", Branch::Miss);
            assert!(
                held <= bound(control.after_the_first_tokens),
                "{test}: the permit stayed held {held:?} after a request with no slot was gone"
            );
            let last = whole(test, &served.generator, request(N_TOKENS)).await.ids;
            cache.restart();
            cache.expect("last run", Branch::Miss);
            assert_eq!(
                last, cold,
                "{test}: the request after a stop with no slot emitted other tokens than the cold run"
            );
        }
    });
    served.events.events()
}

/// A request for `max_tokens` after a prompt that the model continues with no
/// end token: a sentence said many times, with no chat template. The seed is
/// fixed, so a `temperature` above 0 gives the same tokens each time.
fn continued(dir: &Path, max_tokens: u32, temperature: f32) -> GenerationRequest {
    let mut req = text_request(dir);
    req.prompt_tokens = long_prompt(dir);
    req.max_tokens = max_tokens;
    req.sampling.temperature = temperature;
    req
}

fn assert_plain(test: &str, model: (&str, &str), reuse: Reuse) {
    let Some(dir) = snapshot(test, model.0, model.1) else {
        return;
    };
    let slots = usize::from(reuse == Reuse::OneSlot);
    let load = generator_with(dir.clone(), slots, None);
    let request = |max_tokens: u32| continued(&dir, max_tokens, 0.0);
    assert_a_request_gone_in_its_decode_frees_the_gpu(test, &load, &request, false, reuse);
}

fn assert_drafted(test: &str, verifier: (&str, &str), drafter: (&str, &str), temperature: f32) {
    let Some(v) = snapshot(test, verifier.0, verifier.1) else {
        return;
    };
    let Some(d) = snapshot(test, drafter.0, drafter.1) else {
        return;
    };
    let load = speculative_generator(v.clone(), d, Device::Gpu);
    let request = |max_tokens: u32| continued(&v, max_tokens, temperature);
    assert_a_request_gone_in_its_decode_frees_the_gpu(test, &load, &request, true, Reuse::NoSlot);
}

// ── The shared decode loop ──────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU and the Bonsai-8B snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_shared_loop_with_several_kv_heads() {
    assert_plain(
        "a_gone_request_frees_the_gpu_on_the_shared_loop_with_several_kv_heads",
        BONSAI_8B,
        Reuse::NoSlot,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_shared_loop_with_one_kv_head() {
    assert_plain(
        "a_gone_request_frees_the_gpu_on_the_shared_loop_with_one_kv_head",
        GEMMA4_E2B,
        Reuse::NoSlot,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Ternary-Bonsai-2-27B snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_shared_loop_with_recurrent_layers() {
    assert_plain(
        "a_gone_request_frees_the_gpu_on_the_shared_loop_with_recurrent_layers",
        TERNARY_BONSAI_2_27B,
        Reuse::NoSlot,
    );
}

// ── The architectures with a decode loop of their own ───────────────────────

#[test]
#[ignore = "requires Metal GPU and the ReaderLM-v2 snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_qwen2_loop() {
    assert_plain(
        "a_gone_request_frees_the_gpu_on_the_qwen2_loop",
        READERLM_V2,
        Reuse::OneSlot,
    );
}

#[test]
#[ignore = "requires Metal GPU and the bitnet-b1.58 snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_bitnet_loop() {
    assert_plain(
        "a_gone_request_frees_the_gpu_on_the_bitnet_loop",
        BITNET,
        Reuse::OneSlot,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3-VL-30B snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_qwen3_vl_text_loop() {
    assert_plain(
        "a_gone_request_frees_the_gpu_on_the_qwen3_vl_text_loop",
        QWEN3_VL,
        Reuse::OneSlot,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3-VL-30B snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_qwen3_vl_image_loop() {
    let test = "a_gone_request_frees_the_gpu_on_the_qwen3_vl_image_loop";
    let Some(dir) = snapshot(test, QWEN3_VL.0, QWEN3_VL.1) else {
        return;
    };
    let encoder_cache = Arc::new(MultimodalCache::new(ENCODER_CACHE_BYTES));
    let load = generator_with(dir.clone(), 1, Some(Arc::clone(&encoder_cache)));
    // The image, then the prompt of `continued`: the reply to the chat prompt
    // ends at its end token after a few tokens, and a drop cannot land in it.
    let request = |max_tokens: u32| {
        let mut req = image_request(&dir);
        req.prompt_tokens = long_prompt(&dir);
        req.max_tokens = max_tokens;
        req
    };
    let events = assert_a_request_gone_in_its_decode_frees_the_gpu(
        test,
        &load,
        &request,
        false,
        Reuse::Bypass,
    );
    let on_the_image_path = events
        .iter()
        .filter(|e| e.message == IMAGE_PATH_EVENT)
        .count();
    assert_eq!(
        on_the_image_path, 4,
        "{test}: each of the four requests of the case takes the image path"
    );
    // The cold run computes the tower and stores its output; the control, the
    // stopped request and the request after it read that output.
    let stats = encoder_cache.stats();
    assert_eq!(
        (stats.misses, stats.hits, stats.entries),
        (1, 3, 1),
        "{test}: the encoder cache"
    );
}

/// An image request on Qwen3-VL that is gone in its image prefill, before its
/// first token, frees the GPU, and the model then serves the next request as
/// it served the cold one. The model has an encoder cache, so the stopped
/// request and the one after it read the tower output of the first. The prompt is as long as the context allows, so the
/// prefill takes long enough to tell a job that stopped from one that ran on.
#[test]
#[ignore = "requires Metal GPU and the Qwen3-VL-30B snapshot; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_in_the_qwen3_vl_image_prefill() {
    let test = "a_gone_request_frees_the_gpu_in_the_qwen3_vl_image_prefill";
    let Some(dir) = snapshot(test, QWEN3_VL.0, QWEN3_VL.1) else {
        return;
    };
    let encoder_cache = Arc::new(MultimodalCache::new(ENCODER_CACHE_BYTES));
    let load = generator_with(dir.clone(), 0, Some(Arc::clone(&encoder_cache)));
    let served = Served::new(&load);
    let generator = &served.generator;
    let block = long_prompt(&dir);
    let position_limit = generator
        .context_limits()
        .and_then(|limits| limits.positional_tokens())
        .unwrap_or(usize::MAX);
    let request = |tokens: usize, max_tokens: u32| {
        let mut req = image_request(&dir);
        req.prompt_tokens = block.iter().copied().cycle().take(tokens).collect();
        req.max_tokens = max_tokens;
        req.max_ctx_override = Some((tokens + REPLY_ROOM) as i32);
        req
    };
    served.pool.block_on(async {
        whole(test, generator, request(FIRST_TURN_TOKENS, N_TOKENS)).await;
        release_cached_buffers();
        let mut tokens = FIRST_TURN_TOKENS;
        let mut last_control = None;
        let cold = loop {
            assert!(
                tokens + REPLY_ROOM <= position_limit && tokens <= MOST_TURN_TOKENS,
                "{test}: no power: a prompt of {tokens} tokens passes the {MOST_TURN_TOKENS} \
                 tokens of a prefill case, or with a reply the {position_limit} tokens the \
                 model addresses; the prefill of the prompt before took {last_control:?}, \
                 under {PREFILL_CONTROL:?}"
            );
            let cold = whole(test, generator, request(tokens, N_TOKENS)).await;
            release_cached_buffers();
            if cold.to_the_first_token >= PREFILL_CONTROL {
                break cold;
            }
            last_control = Some(cold.to_the_first_token);
            tokens *= 2;
        };
        assert_power(test, "the image prefill", cold.to_the_first_token);
        let wait = cold.to_the_first_token / 4;
        let mark = served.events.events().len();
        let held = served
            .stopped(test, request(tokens, N_TOKENS), 0, wait, false)
            .await;
        release_cached_buffers();
        let events = served.events.events().split_off(mark);
        let entered = events.iter().filter(|e| e.message == IMAGE_PATH_EVENT);
        assert_eq!(
            entered.count(),
            1,
            "{test}: the request was gone before its tower ended"
        );
        let emitted = events
            .iter()
            .find(|e| e.message == STOP_EVENT)
            .and_then(|stop| stop.field("emitted"));
        assert_eq!(
            emitted,
            Some("0"),
            "{test}: the request was gone after its first token"
        );
        assert!(
            held <= bound(cold.to_the_first_token),
            "{test}: the permit stayed held {held:?} after the request was gone; the image \
             prefill with no drop takes {:?}",
            cold.to_the_first_token
        );
        let after = whole(test, generator, request(tokens, N_TOKENS)).await;
        release_cached_buffers();
        assert_eq!(
            after.ids, cold.ids,
            "{test}: the request after a stopped image prefill emitted other tokens than the cold run"
        );
    });
    // The first request computes the tower and stores its output; every later
    // request, the stopped one included, reads it.
    let stats = encoder_cache.stats();
    assert_eq!(
        (stats.misses, stats.entries),
        (1, 1),
        "{test}: the encoder cache"
    );
}

// ── The speculative round loop ──────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b + assistant snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_the_assistant() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_the_assistant",
        GEMMA4_E2B,
        GEMMA4_E2B_ASSISTANT,
        0.0,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + MTP snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_the_sidecar() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_the_sidecar",
        QWEN38_4BIT,
        QWEN38_MTP_4BIT,
        0.0,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B + DFlash snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_dflash() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_dflash",
        QWEN36_MOE,
        QWEN36_DFLASH,
        0.0,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + DFlash2 snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_dflash2() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_dflash2",
        QWEN38_4BIT,
        QWEN38_DFLASH2,
        0.0,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B + EAGLE-3 snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_eagle3() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_eagle3",
        QWEN36_MOE,
        QWEN36_EAGLE3,
        0.0,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B mxfp8 + ornith-9b snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_a_draft_model() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_a_draft_model",
        QWEN38_MXFP8,
        ORNITH_9B,
        0.0,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B mxfp8 + ornith-9b snapshots; run with `make gpu-test`"]
fn a_gone_request_frees_the_gpu_on_the_round_loop_with_a_sampled_draft_model() {
    assert_drafted(
        "a_gone_request_frees_the_gpu_on_the_round_loop_with_a_sampled_draft_model",
        QWEN38_MXFP8,
        ORNITH_9B,
        0.7,
    );
}

// ── The prefill, and what the prompt cache keeps of a stopped request ───────

fn hits(generator: &Arc<dyn Generator>) -> u64 {
    generator.cache_stats().map_or(0, |stats| stats.hits)
}

/// A request that is gone in its prefill frees the GPU and leaves no prompt
/// in the cache; a request that is gone in its decode leaves the prompt that
/// its whole prefill stored.
///
/// Every reference is taken before the stop it is compared with. A cold
/// reference is a run with no cache slot, which also empties the cache; the
/// time of a prefill is taken from the second cold run, not from the first
/// request after the load. The reference of the second turn is a run on the
/// same path as the run it is compared with, the stored first turn and a
/// tail: a tail prefill agrees with a prefill of the whole prompt to rounding
/// noise only.
fn assert_the_prompt_cache_keeps_no_stopped_prefill(test: &str, model: (&str, &str)) {
    let Some(dir) = snapshot(test, model.0, model.1) else {
        return;
    };
    let load = generator_with(dir.clone(), 4, None);
    let served = Served::new(&load);
    let generator = &served.generator;

    let block = long_prompt(&dir);
    let position_limit = generator
        .context_limits()
        .and_then(|limits| limits.positional_tokens())
        .unwrap_or(usize::MAX);
    let turn_of =
        |tokens: usize| -> Vec<u32> { block.iter().copied().cycle().take(tokens).collect() };
    let request_in = |prompt: &[u32], max_tokens: u32, slots: Option<usize>, max_ctx: usize| {
        let mut req = text_request(&dir);
        req.prompt_tokens = prompt.to_vec();
        req.max_tokens = max_tokens;
        req.max_ctx_override = Some(max_ctx as i32);
        req.effective_prompt_cache_slots = slots;
        req
    };
    served.pool.block_on(async {
        // The first turn, stopped in its prefill. The turn doubles until its
        // cold prefill takes the control time, and the case stops for want of
        // power past [`MOST_TURN_TOKENS`], or when two such turns and a reply
        // pass what the model addresses. The buffer cache is released after
        // each request, so the buffers of one prompt size do not stay under
        // the next.
        let first_turn = turn_of(FIRST_TURN_TOKENS);
        let warm_ctx = 2 * FIRST_TURN_TOKENS + REPLY_ROOM;
        whole(
            test,
            generator,
            request_in(&first_turn, N_TOKENS, Some(0), warm_ctx),
        )
        .await;
        release_cached_buffers();
        let mut turn_tokens = FIRST_TURN_TOKENS;
        let mut last_control = None;
        let (turn, max_ctx, cold) = loop {
            let max_ctx = 2 * turn_tokens + REPLY_ROOM;
            assert!(
                max_ctx <= position_limit && turn_tokens <= MOST_TURN_TOKENS,
                "{test}: no power: a turn of {turn_tokens} tokens passes the {MOST_TURN_TOKENS} \
                 tokens of a prefill case, or two turns and a reply pass the {position_limit} \
                 tokens the model addresses; the prefill of the turn before took \
                 {last_control:?}, under {PREFILL_CONTROL:?}"
            );
            let turn = turn_of(turn_tokens);
            let cold = whole(
                test,
                generator,
                request_in(&turn, N_TOKENS, Some(0), max_ctx),
            )
            .await;
            release_cached_buffers();
            if cold.to_the_first_token >= PREFILL_CONTROL {
                break (turn, max_ctx, cold);
            }
            last_control = Some(cold.to_the_first_token);
            turn_tokens *= 2;
        };
        let two_turns = [turn.as_slice(), turn.as_slice()].concat();
        let request = |prompt: &[u32], max_tokens: u32, slots: Option<usize>| {
            request_in(prompt, max_tokens, slots, max_ctx)
        };
        assert_power(test, "the prefill", cold.to_the_first_token);
        let wait = cold.to_the_first_token / 4;
        let held = served
            .stopped(test, request(&turn, N_TOKENS, None), 0, wait, false)
            .await;
        release_cached_buffers();
        assert!(
            held <= bound(cold.to_the_first_token),
            "{test}: the permit stayed held {held:?} after the request was gone; the prefill \
             with no drop takes {:?}",
            cold.to_the_first_token
        );
        let before = hits(generator);
        let prefilled = served.ended(test, request(&turn, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_eq!(
            hits(generator),
            before,
            "{test}: the prompt of a request stopped in its prefill was found in the cache"
        );
        assert_eq!(prefilled.ids, cold.ids, "{test}: after a stopped prefill");
        let hit = whole(test, generator, request(&turn, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_eq!(
            hits(generator),
            before + 1,
            "{test}: a whole prefill is stored"
        );
        assert_eq!(hit.ids, cold.ids, "{test}: the hit of a whole prefill");

        // The second turn, stopped in the prefill of its tail.
        let second = whole(test, generator, request(&two_turns, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_power(
            test,
            "the prefill of the second turn",
            second.to_the_first_token,
        );
        let cold_short = whole(test, generator, request(&block, N_TOKENS, Some(0))).await;
        release_cached_buffers();
        let first = whole(test, generator, request(&turn, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_eq!(first.ids, cold.ids, "{test}: the first turn, stored again");
        let wait = second.to_the_first_token / 4;
        let held = served
            .stopped(test, request(&two_turns, N_TOKENS, None), 0, wait, false)
            .await;
        release_cached_buffers();
        assert!(
            held <= bound(second.to_the_first_token),
            "{test}: the permit stayed held {held:?} after the second turn was gone; its \
             prefill with no drop takes {:?}",
            second.to_the_first_token
        );
        let again = whole(test, generator, request(&two_turns, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_eq!(
            again.ids, second.ids,
            "{test}: the second turn after its stop"
        );
        let first = whole(test, generator, request(&turn, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_eq!(
            first.ids, cold.ids,
            "{test}: the first turn after that stop"
        );

        // A request stopped in its decode leaves the prompt it prefilled: the
        // entry is the state after the whole prompt, which the stop did not cut.
        // A run with no cache slot rebuilds the cache, and the stop rebuilds it
        // again with its slots, so the stop starts from a cache that holds
        // nothing and the count of hits is 0. The lookup of the next request is
        // then the one hit, if the stop stored its entry.
        whole(test, generator, request(&block, N_TOKENS, Some(0))).await;
        release_cached_buffers();
        let stop = request(&block, LONG_REPLY, None);
        served
            .stopped(test, stop, READ_BEFORE_THE_DROP, Duration::ZERO, false)
            .await;
        release_cached_buffers();
        let before = hits(generator);
        assert_eq!(
            before, 0,
            "{test}: the stop did not start from an empty cache"
        );
        let short = whole(test, generator, request(&block, N_TOKENS, None)).await;
        release_cached_buffers();
        assert_eq!(
            hits(generator),
            before + 1,
            "{test}: the entry of a stopped decode"
        );
        assert_eq!(
            short.ids, cold_short.ids,
            "{test}: the hit after a stopped decode"
        );
    });
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit snapshot; run with `make gpu-test`"]
fn the_prompt_cache_keeps_no_stopped_prefill_on_qwen3_8() {
    assert_the_prompt_cache_keeps_no_stopped_prefill(
        "the_prompt_cache_keeps_no_stopped_prefill_on_qwen3_8",
        QWEN38_4BIT,
    );
}

#[test]
#[ignore = "requires Metal GPU and the ReaderLM-v2 snapshot; run with `make gpu-test`"]
fn the_prompt_cache_keeps_no_stopped_prefill_on_qwen2() {
    assert_the_prompt_cache_keeps_no_stopped_prefill(
        "the_prompt_cache_keeps_no_stopped_prefill_on_qwen2",
        READERLM_V2,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn the_prompt_cache_keeps_no_stopped_prefill_on_a_cache_that_reuses_a_prefix() {
    assert_the_prompt_cache_keeps_no_stopped_prefill(
        "the_prompt_cache_keeps_no_stopped_prefill_on_a_cache_that_reuses_a_prefix",
        GEMMA4_E2B,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Bonsai-8B snapshot; run with `make gpu-test`"]
fn the_prompt_cache_keeps_no_stopped_prefill_on_qwen3() {
    assert_the_prompt_cache_keeps_no_stopped_prefill(
        "the_prompt_cache_keeps_no_stopped_prefill_on_qwen3",
        BONSAI_8B,
    );
}

// ── The speech route ────────────────────────────────────────────────────────

/// POST a speech request and return the status and the body of the reply.
async fn speak(port: u16, input: &str) -> (u16, Vec<u8>) {
    let mut client = speech_request(port, input).await;
    let mut reply = Vec::new();
    tokio::time::timeout(STEP_LIMIT, client.read_to_end(&mut reply))
        .await
        .expect("the synthesis ends")
        .expect("read the reply");
    let head_end = reply
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a reply with a head");
    let status = String::from_utf8_lossy(&reply[..head_end])
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("a status");
    (status, reply.split_off(head_end + 4))
}

/// Send a speech request and return its connection, with no reply read.
async fn speech_request(port: u16, input: &str) -> tokio::net::TcpStream {
    let body = serde_json::json!({
        "model": "tts",
        "input": input,
        "voice": "serena",
        "response_format": "pcm",
        "seed": 7,
    })
    .to_string();
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect");
    let head = format!(
        "POST /v1/audio/speech HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    client
        .write_all(format!("{head}{body}").as_bytes())
        .await
        .expect("send the request");
    client
}

/// A synthesis whose client is gone stops inside the bound, writes no metrics
/// row and logs one stop event, and the model then speaks the audio it spoke
/// before the stop, for the same input and seed.
///
/// The speech job runs on a thread of the server, so this case reads the
/// events of every thread: it installs the recorder for the whole process.
#[test]
#[ignore = "requires Metal GPU and the Qwen3-TTS snapshots; run with `make gpu-test`"]
fn a_gone_speech_request_frees_the_gpu_and_the_next_one_speaks_the_same_audio() {
    let test = "a_gone_speech_request_frees_the_gpu_and_the_next_one_speaks_the_same_audio";
    let Some(model_dir) = snapshot(test, TTS.0, TTS.1) else {
        return;
    };
    let Some(codec_dir) = snapshot(test, TTS_CODEC.0, TTS_CODEC.1) else {
        return;
    };
    release_cached_buffers();
    let events = RoundStreamRecorder::new();
    tracing::subscriber::set_global_default(Arc::clone(&events))
        .expect("no other test of this binary installs a recorder for the process");
    let dir = tempfile::tempdir().expect("a temporary directory");
    let db = dir.path().join("events.db");
    let mut state = stub_server::not_ready_state(ModelRegistry::default());
    state.device = Device::Gpu;
    state.tts_model_path = Some(model_dir);
    state.tts_tokenizer_path = Some(codec_dir);
    state.metrics = Some(Arc::new(
        EventRecorder::open_at(&db, "gone").expect("an events DB"),
    ));
    let queue = Arc::clone(&state.gpu_queue);
    let short = thread_boundary::SPOKEN_TEXT;
    let long = [short; 24].join(" ");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async {
        let port = stub_server::start_server_with_state(state).await;
        let (status, before) = speak(port, short).await;
        assert_eq!(status, 200, "{test}: the synthesis before the stop");
        assert!(!before.is_empty(), "{test}: no audio");

        let written = rows(&[&db]);
        let started = Instant::now();
        let (status, _) = speak(port, &long).await;
        assert_eq!(status, 200, "{test}: the control synthesis");
        let control = started.elapsed();
        assert_power(test, "the synthesis", control);
        assert!(
            rows_after(&[&db], written).await > written,
            "{test}: no power: a synthesis that ended wrote no metrics row"
        );

        let (mark, written) = (events.events().len(), rows(&[&db]));
        let client = speech_request(port, &long).await;
        tokio::time::sleep(control / 4).await;
        assert_eq!(
            queue.available_permits(),
            0,
            "{test}: the synthesis ended before the client went away"
        );
        drop(client);
        let dropped = Instant::now();
        let permit = tokio::time::timeout(STEP_LIMIT, queue.acquire()).await;
        assert!(
            matches!(permit, Ok(Ok(_))),
            "{test}: the permit never came back"
        );
        let held = dropped.elapsed();
        drop(permit);
        assert!(
            held <= bound(control),
            "{test}: the permit stayed held {held:?} after the client was gone; the synthesis \
             with no drop takes {control:?}"
        );
        tokio::time::sleep(ROWS_WAIT).await;
        assert_eq!(
            rows(&[&db]),
            written,
            "{test}: a stopped synthesis wrote a metrics row"
        );
        assert_the_events_of_a_stop(test, &events.events().split_off(mark), false);

        let (status, after) = speak(port, short).await;
        assert_eq!(status, 200, "{test}: the synthesis after the stop");
        assert!(
            after == before,
            "{test}: the synthesis after a stopped one spoke other audio: {} bytes, not {}",
            after.len(),
            before.len()
        );
    });
}
