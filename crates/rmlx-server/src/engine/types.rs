//! Public request / response / admission types for the engine surface.
//!
//! - Phase, NormalizedTool, NormalizedToolChoice, normalized_to_jinja_tool
//! - ModelLoadConfig
//! - NormalizedResponseFormat
//! - SamplingParams, GenerationRequest
//! - GpuAdmission, Admission, admit_request
//! - GenerationToken

use std::sync::Arc;
use std::time::Instant;

use rmlx_metrics::events::EventRecorder;

use crate::metrics_drainer::DrainerHandle;
use crate::openai::ItlStore;

// ── per-request generation phase ───────────────────────────────────────

/// Tracing-field label vocabulary for the TTFT prefill→decode boundary. Not a
/// state machine — emitted as a hardcoded variant at the appropriate emit sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(
    clippy::exhaustive_enums,
    reason = "internal closed enum — exactly two generation phases; adding a phase requires reviewing all emit sites"
)]
pub enum Phase {
    /// Prompt processing phase (evaluating input tokens).
    Prefill,
    /// Autoregressive decode phase (generating new tokens).
    Decode,
}

// ── A5.1: route-agnostic tool normalisation ───────────────────────────────────

/// A single tool in a normalised, route-agnostic form.
///
/// OpenAI's `function.parameters` and Anthropic's `input_schema` both hold a
/// JSON Schema object and map verbatim onto `schema` — they are semantically
/// identical.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — normalized form shared between OpenAI and Anthropic routes; field set is the complete normalization contract"
)]
pub struct NormalizedTool {
    /// Tool function name matching the upstream spec.
    pub name: String,
    /// Optional human-readable description of the tool's purpose.
    pub description: Option<String>,
    /// JSON Schema for the tool's arguments.
    pub schema: serde_json::Value,
}

/// Route-agnostic representation of the `tool_choice` preference.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_enums,
    reason = "internal closed enum — four tool_choice modes from OpenAI/Anthropic specs; adding a mode requires reviewing all route normalization sites"
)]
pub enum NormalizedToolChoice {
    /// Let the model decide whether to call a tool (`"auto"` / default).
    Auto,
    /// Never call a tool (`"none"`).
    None,
    /// Always call some tool (`"required"` / Anthropic `"any"`).
    Required,
    /// Call a specific named tool.
    Named(String),
}

/// A5.2: Convert a `NormalizedTool` to the OpenAI-shaped `serde_json::Value`
/// that chat templates expect in their `tools` context variable.
///
/// Shape:
/// ```json
/// {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
/// ```
///
/// This is the least-common-denominator understood by Qwen3, Llama 3.1+,
/// Mistral-Instruct, and Hermes-2-Pro chat templates.
pub fn normalized_to_jinja_tool(t: &NormalizedTool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema,
        }
    })
}

// ── shared model-load configuration ──────────────────────────────────

/// Shared startup configuration handed to every generator constructor.
///
/// Bundles the model-load args that `ArchGenerator` and
/// `SpeculativeGenerator` carried identically (`device`, `kv_quant`,
/// `max_ctx`, `prompt_cache_slots`). Built once in `run_serve` from the
/// CLI-resolved flags and passed by reference into each constructor.
///
/// `gpu_gate` is intentionally *not* a field here: it is a shared runtime
/// resource handle (`Arc<Mutex<()>>`) cloned per construction, not a plain
/// load-config value, so it stays a separate constructor argument.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed startup config — field set is the complete model-load contract; adding a field requires updating run_serve and all generator constructors"
)]
pub struct ModelLoadConfig {
    /// Compute device (`Cpu` / `Gpu`).
    pub device: rmlx_mlx::Device,
    /// Startup KV-quant selection. `None` = `--kv-quant=auto`; resolved
    /// against the model's `config.json` at load time (registry mode passes
    /// `None`; single-`--model` mode passes the CLI-resolved value).
    pub kv_quant: Option<rmlx_kv_quant::KvQuant>,
    /// `--max-ctx` override. `None` = derive from mpe (capped at 4096).
    pub max_ctx: Option<i32>,
    /// Number of prompt-cache slots for multi-slot prefix matching. Default 4.
    pub prompt_cache_slots: usize,
    /// Shared multimodal encoder-output cache. `None` disables the
    /// cache for this load (text-only generators do not need it; vision
    /// generators receive a populated `Arc` from `AppState.mm_cache`).
    pub mm_cache: Option<Arc<rmlx_models::multimodal_cache::MultimodalCache>>,
    /// Optional KV calibration discovered from `kv_calib.json`
    /// next to the model snapshot. `None` = no calibration file found or
    /// validation failed (version/head_size mismatch). When `Some`, the
    /// per-arch model builder forwards it to `KvCacheBuilder` so the codec
    /// storage layer can attach per-layer high-precision indices.
    ///
    /// Discovery is automatic and transparent — missing JSON = unchanged
    /// behavior. No CLI flag is exposed; the field is populated by the loader
    /// closure in `run_serve` via `rmlx_loader::discover_kv_calibration`.
    pub calibration: Option<rmlx_loader::KvCalibration>,
    /// Runtime YARN RoPE override for Qwen3 models that lack `rope_scaling`
    /// in `config.json`. `None` = no override (default, byte-identical to
    /// pre-flag behaviour). Set via `--yarn-factor` / `--yarn-original-max`
    /// CLI flags on `rmlx serve` / `rmlx baseline`.
    pub yarn: Option<rmlx_models::qwen3::YarnOverride>,
    /// Server-startup default image-token budget for Gemma4-unified vision
    /// (`--image-max-tokens`). `None` = use the snapshot's
    /// `processor_config.json` `max_soft_tokens` (typically 280). A per-request
    /// `image_max_tokens` field overrides this. A no-op for non-vision models.
    pub image_max_tokens: Option<usize>,
}

// ── A6.1: route-agnostic response-format normalisation ───────────────────────

/// Route-agnostic representation of the requested output format.
///
/// OpenAI sends this via the `response_format` field. Anthropic JSON mode is
/// done via prompt + `stop_sequences` and does not set this field (A6 is
/// OpenAI-only). The field is currently a no-op — the generator ignores it
/// and the model relies on the prompt to comply. A6.2..A6.5 will wire logit
/// masking and grammar enforcement.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_enums,
    reason = "internal closed enum — three response format modes from OpenAI spec; adding a mode requires reviewing A6.x constraint wiring"
)]
pub enum NormalizedResponseFormat {
    /// Client explicitly requested plain text (`{"type":"text"}`).
    /// Semantically identical to `None` on `GenerationRequest`; kept as a
    /// distinct variant to allow logging "client asked for text mode" without
    /// ambiguity.
    Text,
    /// Any valid JSON object (`{"type":"json_object"}`).
    JsonObject,
    /// JSON conforming to the supplied JSON Schema (`{"type":"json_schema"}`).
    JsonSchema {
        /// Schema name from the client request.
        name: String,
        /// When `true`, disallow extra properties not in the schema.
        strict: bool,
        /// The JSON Schema value to enforce.
        schema: serde_json::Value,
    },
}

// ── Public types ─────────────────────────────────────────────────────────────

// ── A7.1: SamplingParams ──────────────────────────────────────────────────────

/// Fully resolved sampling parameters for one generation request.
///
/// All fields carry their effective value after the three-tier fallback:
/// **request > model `generation_defaults` (A4) > hard-coded default**.
///
/// Resolution is performed by `resolve_sampling_params` in `openai.rs`.
///
/// **Greedy no-op (A7.1):** The decode loop in every architecture still calls
/// `generate_greedy`, which ignores all sampling fields. Real sampling
/// (temperature, top_k / top_p / min_p nucleus, penalties, logit_bias) lands
/// in A7.2 (core) and A7.3 (penalties + logit_bias).
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — complete A7.x sampling contract; adding a field requires reviewing resolve_sampling_params and all generation sites"
)]
pub struct SamplingParams {
    /// Resolved temperature. Hard-coded default: `1.0`.
    pub temperature: f32,
    /// Resolved top_p. Hard-coded default: `1.0` (disabled).
    pub top_p: f32,
    /// Top-k cutoff. `0` = disabled. Hard-coded default: `0`.
    pub top_k: u32,
    /// Minimum token probability (nucleus floor). `0.0` = disabled.
    pub min_p: f32,
    /// Multiplicative repetition penalty. `1.0` = no-op (mlx-lm / HF convention).
    pub repetition_penalty: f32,
    /// Additive frequency penalty (per-token count, OpenAI convention). `0.0` = disabled.
    pub frequency_penalty: f32,
    /// Additive presence penalty (presence indicator, OpenAI convention). `0.0` = disabled.
    pub presence_penalty: f32,
    /// Token-id → logit bias pairs. Empty = no biases.
    ///
    /// Keys are pre-parsed from the JSON string-keyed map (`"1234"` → `1234u32`).
    /// Out-of-vocab ids are kept here; A7.3 will clamp/skip at apply time.
    pub logit_bias: Vec<(u32, f32)>,
    /// Optional RNG seed. `None` = entropy-seeded (model default).
    pub seed: Option<u64>,
    /// number of top per-token logprobs to capture (OpenAI `top_logprobs`).
    /// `0` = logprob capture disabled (the default; hot-loop zero-overhead).
    /// Resolved in the route handler from `logprobs` / `top_logprobs`:
    /// `logprobs:true` with no `top_logprobs` ⇒ `1`; `top_logprobs:N` ⇒ `N`.
    pub top_logprobs_k: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
            seed: None,
            top_logprobs_k: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// One token-generation request, fully parsed/validated.
///
/// A6.2: dropped the `Clone` derive because `constraint` carries a
/// `Box<dyn ConstraintEngine>` trait object that has no clone impl.
/// No production callsite ever clones a `GenerationRequest`; the route
/// handlers move the value into `Generator::generate` and the engine
/// either consumes it or extracts the fields it needs.
#[derive(Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — complete generation contract; adding a field requires updating both route handlers and all generator implementations"
)]
pub struct GenerationRequest {
    /// Registry model id identifying which generator to dispatch to.
    pub model_id: String,
    /// Pre-tokenized prompt. Empty until Stage 1.7 wires tokenization.
    pub prompt_tokens: Vec<u32>,
    /// Maximum new tokens to generate before stopping.
    pub max_tokens: u32,
    /// A7.1: fully resolved sampling parameters.
    ///
    /// Decode stays greedy (A7.1 schema-only). Real sampling lands in A7.2/A7.3.
    pub sampling: SamplingParams,
    /// Stop strings; generation halts on the first match.
    pub stop: Vec<String>,
    /// When `true`, the response is delivered as an SSE stream.
    pub stream: bool,
    /// Optional system prompt.
    ///
    /// OpenAI format carries system as a `role = "system"` message in the
    /// `messages` array; the OpenAI route extracts it here and strips it from
    /// the prompt-message list so the engine receives one canonical form.
    /// Anthropic format has `system` as a top-level field, which maps directly.
    pub system: Option<String>,
    /// Optional session identifier from `X-Session-Id` header (N2).
    ///
    /// When present, the engine uses this to reserve a PromptCache slot for
    /// the session so FIFO eviction does not clobber it between turns.
    /// Absence falls back to the N1 prompt-cache path with no reservation.
    pub session_id: Option<String>,
    /// Effective prompt-cache slot count computed by the route handler (N2).
    ///
    /// Set to `base_slots + session_cache.active_count()` when a session ID
    /// is present. `None` means use the generator's default `prompt_cache_slots`.
    pub effective_prompt_cache_slots: Option<usize>,
    /// F6/L18: SPSC drainer handle for per-request SQLite metric emission.
    ///
    /// Injected by the route handler from `AppState::metrics_drainer`. `None`
    /// in unit-test paths that do not wire the drainer. The blocking thread
    /// calls `try_emit` (non-blocking) after each post-generation stat read.
    pub metrics_drainer: Option<DrainerHandle>,
    /// M30: ring-buffer for ITL aggregate samples.
    ///
    /// Injected by the route handler from `AppState::itl_store`. The blocking
    /// thread writes one `ItlSample` after all decode steps complete so the
    /// `/metrics/cache` handler can read `last_itl` without blocking on SQLite.
    /// `None` in unit-test paths that do not wire the store.
    pub itl_store: Option<ItlStore>,

    /// per-event DB recorder injected from `AppState::metrics`.
    ///
    /// The blocking thread writes ITL percentiles and kv_cache_bytes into
    /// the `events` table at request completion. TTFT is written by the
    /// HTTP handler layer (off-runtime via `spawn_blocking`), not here.
    /// `None` in unit-test paths that do not wire the recorder.
    pub event_recorder: Option<Arc<EventRecorder>>,

    // A5.1: tool-calling fields (parsed + normalised; not yet consumed).
    // A5.2 (chat-template injection), A5.3 (output parser), A5.4/A5.5
    // (response emission) will read these. The decode loop currently ignores them.
    /// Normalised tools from `tools` array. `None` when the request omitted
    /// `tools` or supplied an empty array.
    pub tools: Option<Vec<NormalizedTool>>,
    /// Normalised tool-choice preference. `None` when neither `tools` nor
    /// `tool_choice` was present in the request.
    pub tool_choice: Option<NormalizedToolChoice>,

    /// A6.1: response-format (parsed + normalised; not yet consumed).
    ///
    /// A6.2..A6.5 will wire logit masking / grammar enforcement.
    /// `None` is equivalent to `Text` — plain text output, no constraint.
    pub response_format: Option<NormalizedResponseFormat>,

    /// A6.2: optional sampler constraint engine, instantiated by the route
    /// handler when `response_format ∈ {JsonObject, JsonSchema}`. Threaded
    /// into the per-arch decode loops via `Architecture::generate_greedy`.
    /// `None` for plain-text requests — the decode loop pays only an
    /// `Option::as_mut()` discriminant check on the hot path.
    pub constraint: Option<Box<dyn rmlx_models::ConstraintEngine>>,

    /// A6.3: shared `is_thinking` flag, updated by the route's step_fn
    /// after the think-splitter classifies each emitted token. The
    /// `JsonObjectConstraint` reads this on every `advance` to defer
    /// engagement while the model is in its reasoning channel. `None`
    /// when no constraint is in use OR the constraint doesn't expose a
    /// handle (e.g. NoOp).
    pub is_thinking_handle: Option<Arc<std::sync::atomic::AtomicBool>>,

    /// per-request thinking-token budget. `Some(n)` caps the
    /// reasoning channel at `n` pieces; once exceeded the engine injects
    /// `</think>` and resumes answer generation. `None` (the default)
    /// disables enforcement — the decode loop pays only the
    /// `Option`-discriminant check the `ThinkSplitter` already performs,
    /// so the budget-unset request is the zero-overhead hot path.
    pub thinking_budget: Option<u32>,

    /// token id of `</think>`, resolved ONCE at request-build time by
    /// encoding the literal with `add_special_tokens=false`. The decode
    /// loop forces this id as the next input when the budget is exceeded,
    /// avoiding any per-token tokenizer work. `None` when no budget is set
    /// or the literal could not be encoded (budget then degrades to a
    /// soft no-op — reasoning runs to `max_tokens`).
    pub thinking_end_token_id: Option<u32>,

    /// Whether the rendered prompt leaves the assistant turn inside an open
    /// `<think>` block — read off the prompt text by
    /// `engine::think::prompt_leaves_think_open`, not inferred from the
    /// architecture or from `enable_thinking` (a chat template is free to
    /// ignore that flag, and some do). Drives the `ThinkSplitter`'s initial
    /// channel and, through it, the constraint engine's `is_thinking` gate.
    pub prompt_think_open: bool,

    /// A5.6: reconstruct tool-protocol special-token markers into the
    /// decoded piece stream so the response tool-call parser can see them.
    ///
    /// The Gemma-4 tool markers (`<|tool_call>`, `<tool_call|>`, `<|"|>`,
    /// `<|channel>`, …) are registered as *special tokens* and are stripped
    /// by `tokenizer.decode(skip_special=true)` — the parser working on
    /// decoded text would never see them. When this is `true` the decode
    /// loop appends the raw single-token surface form (`id_to_token`) of any
    /// fully-suppressed special token whose surface looks like an
    /// angle-pipe protocol marker, so the parser receives it.
    ///
    /// Set by the route handler iff a tool-call parser format is active
    /// (`tools_enabled`). Qwen markers are `special=false` and survive
    /// `skip_special` already, so reconstruction is a no-op for them. When
    /// `false` (the common case — no tools) the decode loop pays only one
    /// bool check per token; visible output is unchanged (markers stay
    /// suppressed).
    pub emit_tool_markers: bool,

    /// per-request override for the thinking-block open delimiter.
    ///
    /// `None` (the default) leaves the `ThinkSplitter` using `"<think>"`.
    /// Set by the route handler from `ChatCompletionsRequest::thinking_start_token`
    /// when the caller supplies a non-default string.
    pub thinking_start_token: Option<String>,

    /// per-request override for the thinking-block close delimiter.
    ///
    /// `None` (the default) leaves the `ThinkSplitter` using `"</think>"`.
    /// When set, the forced-injection also resolves the end-token id from
    /// this string (not the literal `"</think>"`) so custom delimiters still cap
    /// the budget correctly.
    pub thinking_end_token: Option<String>,

    /// C5 Slice A: the FIFO admission guard for this request.
    ///
    /// The route handler acquires the single `AppState::gpu_queue` permit
    /// (FIFO order) and an RAII pending-count decrement, packs both into a
    /// [`GpuAdmission`], and moves it here. `Generator::generate` moves the
    /// whole `GenerationRequest` into its `spawn_blocking` closure, so the
    /// permit lives for the entire decode and is released — and
    /// `gpu_pending` decremented — exactly when the blocking work finishes
    /// (success, error, or task completion after stream drop). `None` in
    /// unit-test / non-route paths that never went through admission.
    pub gpu_admission: Option<GpuAdmission>,

    /// Issue #26: per-request KV-quant codec override. `Some(q)` switches the
    /// per-request cache builder to codec `q` on the resident model (no weight
    /// reload); `None` falls through to the generator's launch default
    /// (`--kv-quant`, or the auto per-ctx policy). The prefix/prompt cache key
    /// is namespaced by codec, so a `none`-codec cached prefix never serves a
    /// quantized-KV request and vice-versa.
    pub kv_quant_override: Option<rmlx_kv_quant::KvQuant>,

    /// Issue #26: per-request max-context ceiling override. `Some(n)` re-sizes
    /// the KV-ring virtual ceiling for this request only (the ring still grows
    /// lazily, #25); `None` uses the generator's launch `--max-ctx`. No weight
    /// touch — a ring realloc only.
    pub max_ctx_override: Option<i32>,

    // multimodal content parts extracted from `user` message Parts.
    // will pass these into the vision/audio towers. The decode loop
    // currently ignores them — text-only requests stay byte-identical.
    /// Image URLs (or base64 data-URLs) from `image_url` / `input_image`
    /// content parts in the last user message. Empty for text-only requests.
    pub images: Vec<String>,
    /// Base64-encoded audio from `input_audio` content parts.
    /// Empty for text-only requests.
    pub audio_b64: Vec<String>,

    /// Per-request image-token budget override for Gemma4-unified vision.
    /// `Some(n)` raises the soft-token budget for dense images (e.g. tables)
    /// so more vision resolution is preserved; clamped to the model's safe
    /// upper bound by the preprocessor. Resolved request > `--image-max-tokens`
    /// CLI flag; `None` falls through to the generator's launch default (CLI
    /// flag or the snapshot's `processor_config.json` `max_soft_tokens`). A
    /// no-op for non-vision requests and non-Gemma4 archs.
    pub image_max_tokens: Option<usize>,
}

/// C5 Slice A: RAII admission guard moved into [`GenerationRequest`].
///
/// Holds the single owned FIFO permit from `AppState::gpu_queue` plus a
/// clone of `AppState::gpu_pending`. Dropping it (when the spawned decode
/// task finishes, OR if the request is dropped before generation starts)
/// releases the permit so the next FIFO waiter is admitted AND decrements
/// the in-flight gauge — so both the success and the error/timeout/drop
/// paths balance the `fetch_add` performed at admission.
#[derive(Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal RAII type — field set is the complete admission contract; adding a field requires reviewing admit_request and Drop"
)]
pub struct GpuAdmission {
    _permit: tokio::sync::OwnedSemaphorePermit,
    pending: Arc<std::sync::atomic::AtomicUsize>,
}

impl GpuAdmission {
    /// Wrap an acquired owned permit + the shared pending counter. The
    /// caller must have already `fetch_add`ed `pending` (at the depth check)
    /// before constructing this; `Drop` performs the matching `fetch_sub`.
    pub fn new(
        permit: tokio::sync::OwnedSemaphorePermit,
        pending: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self {
            _permit: permit,
            pending,
        }
    }
}

/// C5 Slice A: result of an admission attempt.
///
/// `Admitted` carries the FIFO guard the caller must move into the
/// `GenerationRequest`. `QueueFull` means the depth check rejected the
/// request before enqueueing — the caller maps it to its own HTTP 429
/// error shape (OpenAI vs Anthropic error bodies differ).
#[derive(Debug)]
#[allow(
    clippy::exhaustive_enums,
    reason = "internal closed enum — exactly two admission outcomes; adding an outcome requires reviewing both route handler call sites"
)]
pub enum Admission {
    /// Request was admitted; the caller must move `guard` into `GenerationRequest`.
    Admitted {
        /// RAII guard holding the GPU semaphore permit for the duration of decode.
        guard: GpuAdmission,
        /// `gpu_pending` observed at admission (this request inclusive) —
        /// the `queue_depth` metric value.
        depth: u64,
        /// Milliseconds spent waiting in the FIFO queue for the permit.
        wait_ms: u64,
    },
    /// Request was rejected; `gpu_pending >= max_queue_depth`.
    QueueFull,
}

/// C5 Slice A: bounded-depth FIFO admission over the single-GPU permit.
///
/// 1. If `max_queue_depth > 0` and `gpu_pending >= max_queue_depth`, reject
///    the request immediately.
/// 2. Otherwise `fetch_add(1)` the in-flight gauge, then `acquire_owned`
///    the GPU semaphore.
///
/// The semaphore is never closed in normal operation; `acquire_owned`
/// only errors if it were closed, in which case we treat it as `QueueFull`
/// (the decrement is handled by an inline guard so the gauge stays
/// balanced).
pub async fn admit_request(
    queue: &Arc<tokio::sync::Semaphore>,
    pending: &Arc<std::sync::atomic::AtomicUsize>,
    max_queue_depth: usize,
) -> Admission {
    use std::sync::atomic::Ordering;
    if max_queue_depth > 0 && pending.load(Ordering::Acquire) >= max_queue_depth {
        tracing::warn!(
            max_queue_depth,
            "C5: admission rejected — server queue full (429)"
        );
        return Admission::QueueFull;
    }
    // Reserve a slot in the in-flight gauge BEFORE awaiting the permit so
    // concurrent depth checks see this request. `depth` = post-increment
    // value (this request inclusive).
    let depth = pending.fetch_add(1, Ordering::AcqRel) as u64 + 1;
    let t_wait = Instant::now();
    if let Ok(permit) = Arc::clone(queue).acquire_owned().await {
        let wait_ms = t_wait.elapsed().as_millis() as u64;
        Admission::Admitted {
            guard: GpuAdmission::new(permit, Arc::clone(pending)),
            depth,
            wait_ms,
        }
    } else {
        // Semaphore closed (shutdown). Balance the fetch_add and reject.
        pending.fetch_sub(1, Ordering::AcqRel);
        tracing::warn!("C5: gpu_queue semaphore closed during acquire — rejecting");
        Admission::QueueFull
    }
}

impl Drop for GpuAdmission {
    fn drop(&mut self) {
        // saturating: never wrap below zero even if a future refactor
        // double-drops. fetch_sub on usize would wrap; guard explicitly.
        let prev = self
            .pending
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |v| Some(v.saturating_sub(1)),
            )
            .unwrap_or(0);
        tracing::trace!(
            gpu_pending_after = prev.saturating_sub(1),
            "C5: GpuAdmission dropped — permit released, pending decremented"
        );
    }
}

/// One produced token with completion state.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed token type — field set is the complete per-token generation contract; adding a field requires updating all Generator implementations and route consumers"
)]
pub struct GenerationToken {
    /// Vocabulary token id produced by the model.
    pub token_id: u32,
    /// Decoded text fragment for this token (may be empty for special tokens).
    pub piece: String,
    /// `true` on the final token of the stream; subsequent tokens must not be consumed.
    pub done: bool,
    /// `"stop"` | `"length"` | `None` (only set when `done == true`)
    pub finish_reason: Option<String>,
    /// A3: whether the visible `piece` text was emitted from inside a
    /// `<think>...</think>` reasoning block. Always `false` for
    /// architectures whose `Architecture::supports_thinking()` returns
    /// `false`. Also `false` for the terminal `done` token (empty piece).
    pub is_thinking: bool,
    /// OpenAI-shaped per-token logprob record for this token. `None`
    /// unless the request set `logprobs:true`. Resolved from the decode loop's
    /// raw `ProbeStep` logprobs (token-id + logprob pairs) into token-surface
    /// strings inside the engine `step_fn`, where the tokenizer is in scope.
    pub logprobs: Option<crate::openai::ChatLogprobContent>,
}
