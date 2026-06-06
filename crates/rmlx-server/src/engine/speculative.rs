//! `SpeculativeGenerator` — greedy speculative decoding.
//!
//! Generator backed by a (verifier, draft) pair under
//! `rmlx_models::SpeculativeDispatcher`.

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use futures::stream::{self, Stream};
use rmlx_core::Error;
use rmlx_metrics::events::Measurement;

use crate::openai::ItlSample;

use super::generator::Generator;
use super::helpers::{
    compute_itl_stats, is_reconstructible_tool_marker, kv_quant_label, record_itl_percentiles,
    resolve_kv_quant_for_load, spsc_ts,
};
use super::think::ThinkSplitter;
use super::types::{GenerationRequest, GenerationToken, ModelLoadConfig};

// ── SpeculativeGenerator ──────────────────────────────────────────────────────

/// Generator backed by a (verifier, draft) pair under
/// `rmlx_models::SpeculativeDispatcher`.
///
/// Algorithm: greedy speculative decoding. Draft proposes K
/// tokens serially per round; verifier evaluates them in one
/// prefill-style call and emits the longest matching prefix plus one
/// bonus/correction token.
///
/// Phase 2 invariants:
/// - K is fixed at construction time (default 4; tunable via env
///   `RMLX_SPEC_K`).
/// - Verifier and draft both re-prefill on every round (no KV
///   reconciliation — Phase 3).
/// - Greedy only — no sampling. Temperature/seed in the request are
///   ignored with a `tracing::warn!`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal generator implementation — field set is coupled to the speculative-decoding lifecycle; adding a field requires updating from_snapshot and all constructors"
)]
pub struct SpeculativeGenerator {
    dispatcher: Arc<rmlx_models::SpeculativeDispatcher>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: rmlx_mlx::Device,
    model_id: String,
    _lock: Arc<Mutex<()>>,
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    /// True when `--kv-quant <explicit>` was given (not "auto").
    /// When `false`, per-request `kv_quant_for_ctx` applies.
    kv_quant_user_explicit: bool,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: Arc<Vec<u32>>,
    /// Draft K (number of speculative tokens proposed per round).
    k: usize,
    /// A2: effective max prompt-context length for the per-request guard.
    /// Derived from verifier's max_position_embeddings clamped by --max-ctx.
    effective_max_ctx: usize,
    /// A10: detokenization family from the verifier's `tokenizer.json`.
    tokenizer_kind: crate::detokenizer::TokenizerKind,
    /// drafter architecture family (`--draft-kind`).
    /// `None` = legacy path (plain SpeculativeDispatcher, no kind metadata).
    /// /14/15 loaders branch on this to select the correct drafter.
    pub draft_kind: Option<rmlx_models::DraftKind>,
    /// draft block size (`--draft-block-size`).
    /// `None` = use default (current: `k` field set from `RMLX_SPEC_K`).
    pub draft_block_size: Option<usize>,
    /// MTP sidecar drafter, `Some` only when `draft_kind == Mtp`.
    ///
    /// Loaded from the `--draft-model` MTP-head folder and validated against the
    /// verifier's hidden size at construction. The MTP round-loop
    /// (forward_verify_capture + draft_n + walk_deferred_greedy + GDN-aware
    /// rollback) is fully wired in `speculative::mtp::mtp_generate_greedy`
    /// against the Qwen3.5/3.6-MoE verifier (the sidecar reuses the verifier's
    /// embedding + LM head + the Qwen3.5-MoE decoder layer).
    /// Wrapped in `Arc<Mutex>` because `MtpDrafter::draft_n` mutates its KV
    /// cache, the `Generator` trait borrows `&self`, and the handle moves into
    /// the blocking decode task (mirrors `dflash_drafter`).
    mtp_drafter: Option<Arc<Mutex<rmlx_models::speculative::mtp::MtpDrafter>>>,
    /// Gemma4-assistant shared-K/V MTP drafter, `Some` when
    /// `draft_kind == Mtp` AND the `--draft-model` is a `gemma4_assistant`
    /// sidecar (the live, runnable family). `draft_n` borrows `&self` only, so
    /// no `Mutex` is needed — wrapped in `Arc` for the blocking decode task.
    mtp_assistant: Option<Arc<rmlx_models::speculative::gemma4_assistant::Gemma4AssistantDrafter>>,
    /// DFlash drafter, `Some` only when `draft_kind == DFlash`.
    ///
    /// Loaded from the `--draft-model` DFlash folder and validated against the
    /// verifier's hidden size + `target_layer_ids` at construction. The DFlash
    /// round-loop (block-size schedule + draft_block + walk + GDN
    /// snapshot/restore rollback + multi-layer hidden capture + raw embed) is
    /// fully wired in `speculative::dflash::dflash_generate_greedy` against the
    /// Qwen3.6-MoE verifier. `Arc<Mutex>` because `draft_block`
    /// mutates the drafter's own KV cache, the `Generator` trait borrows
    /// `&self`, and the handle moves into the blocking decode task.
    dflash_drafter: Option<Arc<Mutex<rmlx_models::speculative::dflash::DFlashDrafter>>>,
    /// EAGLE-3 drafter, `Some` only when `draft_kind == Eagle3`.
    ///
    /// Loaded from the `--draft-model` EAGLE-3 folder and validated against the
    /// verifier's hidden size + vocabulary at construction. The EAGLE-3
    /// round-loop (autoregressive draft + multi-layer hidden capture + d2t
    /// remap + GDN snapshot/restore rollback + raw embed) is wired in
    /// `speculative::eagle3::eagle3_generate_greedy` against the Qwen3.6-MoE
    /// verifier. `Arc<Mutex>` because `draft_block` mutates the drafter's own KV
    /// cache, the `Generator` trait borrows `&self`, and the handle moves into
    /// the blocking decode task.
    eagle3_drafter: Option<Arc<Mutex<rmlx_models::speculative::eagle3::Eagle3Drafter>>>,
}

impl std::fmt::Debug for SpeculativeGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeGenerator")
            .field("model_id", &self.model_id)
            .field("device", &self.device)
            .field("k", &self.k)
            .field("draft_kind", &self.draft_kind)
            .field("draft_block_size", &self.draft_block_size)
            .finish()
    }
}

impl SpeculativeGenerator {
    /// Load a verifier + draft pair from disk and build the speculative
    /// dispatcher.
    ///
    /// `verifier_dir` is the primary `--model` snapshot (its basename
    /// becomes the OpenAI `model_id`). `draft_dir` is the smaller draft.
    /// Vocab match is enforced inside `SpeculativeDispatcher::new`.
    pub fn from_snapshots(
        verifier_dir: &Path,
        draft_dir: &Path,
        cfg: &ModelLoadConfig,
        gpu_gate: Arc<Mutex<()>>,
    ) -> rmlx_core::Result<Self> {
        Self::from_snapshots_with_id(verifier_dir, draft_dir, None, cfg, gpu_gate, None, None)
    }

    /// Like [`from_snapshots`] but accepts an explicit `model_id` override and
    /// draft metadata (`draft_kind`, `draft_block_size`).
    ///
    /// When `model_id_override` is `None` the id is derived from `verifier_dir`
    /// basename (existing behaviour). `draft_kind` / `draft_block_size` are
    /// stored on the generator for /14/15 loaders that branch on kind.
    pub fn from_snapshots_with_id(
        verifier_dir: &Path,
        draft_dir: &Path,
        model_id_override: Option<&str>,
        load_cfg: &ModelLoadConfig,
        gpu_gate: Arc<Mutex<()>>,
        draft_kind: Option<rmlx_models::DraftKind>,
        draft_block_size: Option<usize>,
    ) -> rmlx_core::Result<Self> {
        let device = load_cfg.device;
        let max_ctx_override = load_cfg.max_ctx;
        let prompt_cache_slots = load_cfg.prompt_cache_slots;

        let model_id = model_id_override.map_or_else(
            || {
                verifier_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_owned()
            },
            ToOwned::to_owned,
        );

        tracing::info!(
            model_id = %model_id,
            verifier = %verifier_dir.display(),
            draft = %draft_dir.display(),
            ?device,
            "SpeculativeGenerator: loading verifier+draft"
        );

        // Resolve kv_quant from verifier's config when --kv-quant=auto.
        // Single shared resolver (user-explicit tracking included).
        let cfg = rmlx_loader::load_config(verifier_dir)
            .map_err(|e| Error::Other(format!("load_config (verifier): {e}")))?;
        let (kv_quant_resolved, kv_quant_user_explicit) =
            resolve_kv_quant_for_load(&cfg, load_cfg.kv_quant, &model_id);

        let eos_ids = cfg.eos_token_ids();
        tracing::info!(
            model_id = %model_id,
            ?eos_ids,
            "SpeculativeGenerator: parsed EOS token ids from verifier config"
        );

        // branch on draft kind. For MTP the `--draft-model` folder is a
        // sidecar HEAD (not a full model), so it cannot go through
        // `load_speculative` (which `load_model`s both sides). We load the
        // verifier once for the dispatcher, the MTP head separately, and — until
        // the on-device MTP decode loop's decoder-layer step is wired against a
        // real checkpoint — reuse the verifier snapshot for the dispatcher's
        // draft slot so the struct is valid and the legacy two-model spec path
        // still serves. The MTP loop, when enabled, ignores that draft slot.
        let (dispatcher, mtp_drafter, mtp_assistant, dflash_drafter, eagle3_drafter) =
            if matches!(draft_kind, Some(rmlx_models::DraftKind::Eagle3)) {
                // the `--draft-model` is a standalone EAGLE-3 drafter (not
                // a full model), so it cannot go through `load_speculative`. Load
                // the verifier once for the dispatcher, the EAGLE-3 drafter
                // separately, and reuse the verifier snapshot for the dispatcher's
                // draft slot so the struct is valid (the EAGLE-3 round-loop ignores
                // it).
                tracing::info!(
                    draft = %draft_dir.display(),
                    "SpeculativeGenerator: EAGLE-3 drafter — loading drafter"
                );
                let dispatcher = rmlx_models::SpeculativeDispatcher::load_speculative(
                    verifier_dir,
                    verifier_dir,
                    device,
                )?;
                let hidden_size = dispatcher.verifier.hidden_size();
                let vocab_size = dispatcher.verifier.vocab_size();
                let drafter = rmlx_models::speculative::eagle3::Eagle3Drafter::load(
                    draft_dir,
                    hidden_size,
                    vocab_size,
                    &eos_ids,
                    device,
                )?;
                (
                    dispatcher,
                    None,
                    None,
                    None,
                    Some(Arc::new(Mutex::new(drafter))),
                )
            } else if matches!(draft_kind, Some(rmlx_models::DraftKind::DFlash)) {
                // the `--draft-model` is a standalone DFlash drafter (not a
                // full model), so it cannot go through `load_speculative`. Load the
                // verifier once for the dispatcher, the DFlash drafter separately,
                // and reuse the verifier snapshot for the dispatcher's draft slot
                // so the struct is valid (the DFlash round-loop ignores it).
                tracing::info!(
                    draft = %draft_dir.display(),
                    "SpeculativeGenerator: DFlash drafter — loading drafter"
                );
                let dispatcher = rmlx_models::SpeculativeDispatcher::load_speculative(
                    verifier_dir,
                    verifier_dir,
                    device,
                )?;
                let hidden_size = dispatcher.verifier.hidden_size();
                let drafter = rmlx_models::speculative::dflash::DFlashDrafter::load(
                    draft_dir,
                    hidden_size,
                    device,
                )?;
                (
                    dispatcher,
                    None,
                    None,
                    Some(Arc::new(Mutex::new(drafter))),
                    None,
                )
            } else if matches!(draft_kind, Some(rmlx_models::DraftKind::Mtp)) {
                tracing::info!(
                    draft = %draft_dir.display(),
                    "SpeculativeGenerator: MTP drafter — loading sidecar head"
                );
                let dispatcher = rmlx_models::SpeculativeDispatcher::load_speculative(
                    verifier_dir,
                    verifier_dir,
                    device,
                )?;
                let hidden_size = dispatcher.verifier.hidden_size();
                // Detect the MTP sidecar family from the draft config.json.
                let draft_cfg = rmlx_loader::load_config(draft_dir)
                    .map_err(|e| Error::Other(format!("load_config (draft): {e}")))?;
                let draft_model_type = draft_cfg
                    .extras
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let is_assistant = draft_model_type == "gemma4_assistant"
                    || draft_cfg
                        .architectures
                        .iter()
                        .any(|a| a.contains("Gemma4Assistant"));
                if is_assistant {
                    let drafter =
                        rmlx_models::speculative::gemma4_assistant::Gemma4AssistantDrafter::load(
                            draft_dir,
                            hidden_size,
                            device,
                        )?;
                    (dispatcher, None, Some(Arc::new(drafter)), None, None)
                } else {
                    let drafter = rmlx_models::speculative::mtp::MtpDrafter::load(
                        draft_dir,
                        hidden_size,
                        device,
                    )?;
                    (
                        dispatcher,
                        Some(Arc::new(Mutex::new(drafter))),
                        None,
                        None,
                        None,
                    )
                }
            } else {
                let dispatcher = rmlx_models::SpeculativeDispatcher::load_speculative(
                    verifier_dir,
                    draft_dir,
                    device,
                )?;
                (dispatcher, None, None, None, None)
            };

        let tk_path = verifier_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
            .map_err(|e| Error::Other(format!("load tokenizer: {e}")))?;

        // A10: classify detokenizer family from the verifier tokenizer.json.
        let tokenizer_kind = match std::fs::read(&tk_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        {
            Some(root) => crate::detokenizer::TokenizerKind::from_tokenizer_json(&root),
            None => crate::detokenizer::TokenizerKind::Other,
        };

        // K from env (RMLX_SPEC_K). Default 4.
        let k: usize = std::env::var("RMLX_SPEC_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        // A2: derive effective_max_ctx from the verifier's positional limit.
        // Same formula as Gemma4Generator above. Speculative inherits the
        // verifier's KV-cache sizing, so its positional bound is what matters.
        let mpe_raw = dispatcher.verifier.max_position_embeddings();
        let mpe: usize = if mpe_raw <= 0 { 4096 } else { mpe_raw as usize };
        let effective_max_ctx: usize = match max_ctx_override {
            Some(n) if n > 0 => (n as usize).min(mpe),
            _ => mpe.min(4096),
        };

        tracing::info!(
            model_id = %model_id,
            k,
            ?kv_quant_resolved,
            ?max_ctx_override,
            effective_max_ctx,
            ?draft_kind,
            ?draft_block_size,
            "SpeculativeGenerator: ready"
        );

        Ok(Self {
            dispatcher: Arc::new(dispatcher),
            tokenizer: Arc::new(tokenizer),
            device,
            model_id,
            // C4: shared process-wide GPU gate (see Gemma4Generator above).
            _lock: gpu_gate,
            kv_quant_override: kv_quant_resolved,
            kv_quant_user_explicit,
            max_ctx_override,
            prompt_cache_slots,
            eos_ids: Arc::new(eos_ids),
            k,
            effective_max_ctx,
            tokenizer_kind,
            // drafter metadata for /14/15 loaders.
            draft_kind,
            draft_block_size,
            // MTP sidecar drafter (Some only when draft_kind == Mtp).
            mtp_drafter,
            // Gemma4-assistant shared-K/V drafter (the live MTP family).
            mtp_assistant,
            // DFlash drafter (Some only when draft_kind == DFlash).
            dflash_drafter,
            // EAGLE-3 drafter (Some only when draft_kind == Eagle3).
            eagle3_drafter,
        })
    }

    /// The model id this generator serves (verifier basename).
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl Generator for SpeculativeGenerator {
    fn effective_max_ctx(&self) -> usize {
        self.effective_max_ctx
    }

    fn cache_stats(&self) -> Option<rmlx_models::CacheStats> {
        // SpeculativeDispatcher wraps a Gemma4 verifier — reads its static.
        rmlx_models::gemma4::gemma4_cache_stats()
    }

    fn kv_cache_bytes(&self) -> u64 {
        // SpeculativeDispatcher wraps a Gemma4 verifier — reads its static.
        rmlx_models::gemma4::gemma4_kv_cache_bytes()
    }

    fn load_phases(&self) -> Option<rmlx_models::LoadPhases> {
        rmlx_models::read_load_phases()
    }

    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn generate(
        &self,
        req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        if req.model_id != self.model_id {
            let served = self.model_id.clone();
            let asked = req.model_id;
            return Box::pin(stream::once(async move {
                Err(Error::Other(format!(
                    "model id mismatch: generator serves '{served}', got '{asked}'"
                )))
            }));
        }

        if req.prompt_tokens.is_empty() {
            return Box::pin(stream::once(async {
                Err(Error::Other("empty prompt".to_owned()))
            }));
        }

        // the DFlash drafter round-loop is wired against the Qwen3.6-MoE
        // verifier (multi-layer hidden capture + GDN snapshot/restore rollback +
        // raw embed). Selected below in the blocking task via `dflash_drafter`.

        // The MTP sidecar drafter round-loop is wired against the Qwen3.5/3.6-MoE
        // verifier (forward_verify_capture penultimate-hidden + autoregressive
        // draft_n over the reused Qwen3.5-MoE decoder layer + deferred greedy
        // acceptance + GDN snapshot/restore rollback). Selected below in the
        // blocking task via `mtp_drafter`.

        let (tx, rx) = tokio::sync::mpsc::channel::<rmlx_core::Result<GenerationToken>>(4);

        let dispatcher = Arc::clone(&self.dispatcher);
        // clone the Gemma4-assistant drafter handle into the blocking
        // task (Arc — `draft_n` borrows `&self`). `Some` selects the assistant
        // MTP round-loop over the two-model spec path.
        let mtp_assistant = self.mtp_assistant.clone();
        // clone the DFlash drafter handle into the blocking task. `Some`
        // selects the DFlash round-loop over the MTP / two-model spec path.
        let dflash_drafter = self.dflash_drafter.clone();
        // clone the MTP sidecar drafter handle into the blocking task. `Some`
        // selects the MTP round-loop over the two-model spec path.
        let mtp_drafter = self.mtp_drafter.clone();
        // clone the EAGLE-3 drafter handle into the blocking task. `Some`
        // selects the EAGLE-3 round-loop over the DFlash / MTP / two-model paths.
        let eagle3_drafter = self.eagle3_drafter.clone();
        let block_size = self.draft_block_size.unwrap_or(self.k + 1).max(2);
        let tokenizer = Arc::clone(&self.tokenizer);
        // A10: detokenizer family for the streaming UTF-8 token-healer.
        let tokenizer_kind = self.tokenizer_kind;
        let prompt_tokens = req.prompt_tokens.clone();
        let n_tokens = req.max_tokens as usize;
        let lock = Arc::clone(&self._lock);
        // Same ctx-based auto selection as Gemma4Generator.
        let kv_quant_override = if self.kv_quant_user_explicit {
            self.kv_quant_override
        } else {
            let ctx_quant = rmlx_models::kv_cache::kv_quant_for_ctx(prompt_tokens.len());
            tracing::info!(
                prompt_len = prompt_tokens.len(),
                ?ctx_quant,
                "speculative generate: auto-KV-by-ctx selected quant"
            );
            Some(ctx_quant)
        };
        let max_ctx_override = self.max_ctx_override;
        // F2: capture effective_max_ctx for drainer MetricEvent.ctx_max field.
        let effective_max_ctx_val = self.effective_max_ctx as i64;
        // N2: use effective_prompt_cache_slots override if set by route handler.
        let prompt_cache_slots = req
            .effective_prompt_cache_slots
            .unwrap_or(self.prompt_cache_slots);
        let eos_ids = Arc::clone(&self.eos_ids);
        let k = self.k;
        let model_id_for_log = self.model_id.clone();
        // F6/L18: drainer handle for non-blocking SQLite metric emission.
        let metrics_drainer = req.metrics_drainer;
        // M30: ITL ring-buffer handle for per-request latency aggregates.
        let itl_store = req.itl_store;
        // per-event DB recorder (TTFT is written by the HTTP handler
        // layer off-runtime; only ITL/kv_cache_bytes are written here).
        let event_recorder = req.event_recorder;
        // C5 Slice A: hold the FIFO admission guard for the lifetime of the
        // blocking decode (mirrors Gemma4Generator). Released on completion.
        let gpu_admission = req.gpu_admission;
        // A6.3 Option SK: speculative decode cannot honor a stateful
        // grammar — the K+1 verifier argmax has no per-token mask hook
        // that aligns with `ConstraintEngine::step_mask`. Rather than
        // silently dropping the engine (the A6.2 behaviour, safe only
        // for NoOp), refuse the request with a clear 503 so callers can
        // either drop `response_format` or use the single-arch generator.
        // Future work: sequential-mask integration (Option SQ in the
        // A6.3 spec) would re-evaluate each accepted draft token
        // through the constraint with rollback on rejection.
        if req.constraint.is_some() {
            tracing::warn!(
                model_id = %req.model_id,
                "SpeculativeGenerator: refusing request — response_format \
                 + speculative decode not supported (A6.3 Option SK)"
            );
            return Box::pin(stream::once(async {
                Err(Error::Other(
                    "speculative_decode_with_response_format_unsupported: \
                     `response_format` (json_object / json_schema) is not \
                     supported on the speculative-decoding path. Retry the \
                     request without speculative draft, or drop \
                     response_format."
                        .to_owned(),
                ))
            }));
        }
        let _ = req.constraint;
        // speculative decoding now supports temperature > 0 via Leviathan
        // stochastic acceptance (was A7.2-rejected). `temperature == 0` keeps the
        // byte-identical greedy cached path. Mirror the resolved sampling knobs
        // into the rmlx-models `SamplerConfig` exactly as the standard path does;
        // the dispatcher branches greedy vs stochastic on `sampling_active()`.
        let spec_sampler_cfg = rmlx_models::SamplerConfig {
            temperature: req.sampling.temperature,
            top_p: req.sampling.top_p,
            top_k: req.sampling.top_k,
            min_p: req.sampling.min_p,
            seed: req.sampling.seed,
            // Speculative path does not capture per-token logprobs (verifier
            // emits ProbeStep without logprobs); keep disabled.
            top_logprobs_k: 0,
        };
        if spec_sampler_cfg.sampling_active() {
            tracing::debug!(
                model_id = %req.model_id,
                temperature = spec_sampler_cfg.temperature,
                top_p = spec_sampler_cfg.top_p,
                top_k = spec_sampler_cfg.top_k,
                min_p = spec_sampler_cfg.min_p,
                seed = spec_sampler_cfg.seed_or_default(),
                "SpeculativeGenerator: stochastic acceptance active (Leviathan)"
            );
        }
        // A3: think_splitter mirrors Gemma4Generator. Speculative wraps a
        // verifier `Architecture` so we can ask it directly. Today the
        // verifier is Gemma4 in all production paths (Qwen3 speculative
        // is unimplemented per L36 N48), so this is always `None`, but
        // the wiring keeps the code symmetric for when L36 lands.
        // thread budget + thinking-end id + effective enable_thinking
        // through the same `new_for_request` constructor as the standard
        // path, so the symmetry holds if a thinking verifier ever lands.
        // also thread per-request delimiter overrides.
        let thinking_budget = req.thinking_budget;
        let thinking_end_token_id = req.thinking_end_token_id;
        let splitter_open = req.enable_thinking != Some(false);
        let think_splitter: Option<ThinkSplitter> = if self.dispatcher.verifier.supports_thinking()
        {
            Some(ThinkSplitter::new_for_request(
                splitter_open,
                thinking_budget,
                req.thinking_start_token.clone(),
                req.thinking_end_token.clone(),
            ))
        } else {
            None
        };
        // A5.6: reconstruct suppressed tool-protocol markers (see
        // Gemma4Generator site for rationale).
        let emit_tool_markers = req.emit_tool_markers;

        // Use spawn_blocking instead of std::thread::spawn (same rationale
        // as Gemma4Generator above).
        tokio::task::spawn_blocking(move || {
            let _guard = {
                let try_result = lock.try_lock();
                if let Some(g) = try_result {
                    g
                } else {
                    tracing::warn!(
                        model_id = %model_id_for_log,
                        "SpeculativeGenerator: concurrent generation — waiting for lock"
                    );
                    lock.lock()
                }
            };

            // C5 Slice A: hold the FIFO admission guard for the whole decode
            // (mirrors Gemma4Generator). Released on closure exit.
            let _gpu_admission = gpu_admission;

            tracing::debug!(model_id = %model_id_for_log, k, "spec generate: blocking thread started");

            // Full-prefix decode → byte-diff per token (mirrors
            // Gemma4Generator). Speculative emits in bursts of accept+1
            // tokens per round; the diff-emit pattern handles bursts
            // transparently — each ProbeStep call appends one id and
            // re-decodes the full prefix.
            //
            // A10: owned by `StreamingDetokenizer` (UTF-8 token-healing —
            // see Gemma4Generator site). A multi-byte codepoint split by a
            // speculative round boundary is held until the next ProbeStep
            // completes it.
            let mut detok = crate::detokenizer::StreamingDetokenizer::new(tokenizer_kind);
            // M30: pre-allocated per-step timestamps for ITL computation.
            let mut step_timestamps: Vec<Instant> = Vec::with_capacity(n_tokens);
            let mut cancelled = false;
            // A3: same shape as Gemma4Generator — `None` for non-reasoning archs.
            let mut think_splitter = think_splitter;
            let tx_ref = &tx;
            let cancelled_ref = &mut cancelled;
            let detok_ref = &mut detok;
            let timestamps_ref = &mut step_timestamps;
            let think_splitter_ref = &mut think_splitter;
            let tokenizer_ref = tokenizer.clone();

            // same `Option<u32>` forced-token contract as the
            // standard path. The production verifier is Gemma4 (non-thinking)
            // so `think_splitter` is `None` here and this always returns
            // `None`, but the wiring is symmetric for a future thinking
            // verifier (L36).
            let mut step_fn = |s: &rmlx_models::gemma4::ProbeStep| -> Option<u32> {
                // M30: record step arrival time for ITL computation.
                timestamps_ref.push(Instant::now());
                if *cancelled_ref {
                    return None;
                }
                let mut text = match detok_ref.step(&tokenizer_ref, s.token_id) {
                    Ok(seg) => seg,
                    Err(e) => {
                        tracing::debug!(
                            token_id = s.token_id,
                            error = ?e,
                            "tokenizer.decode error, using empty string"
                        );
                        String::new()
                    }
                };
                // A5.6: reconstruct suppressed Gemma tool markers (see
                // Gemma4Generator site for the full rationale).
                if emit_tool_markers && text.is_empty() {
                    if let Some(surface) = tokenizer_ref.id_to_token(s.token_id) {
                        if is_reconstructible_tool_marker(&surface) {
                            text = surface;
                        }
                    }
                }
                // A3: route through the think-splitter when present.
                let (visible, is_thinking) = match think_splitter_ref.as_mut() {
                    Some(sm) => sm.step(&text),
                    None => (text, false),
                };
                // resolve raw decode logprobs into the OpenAI wire shape
                // (tokenizer in scope). `None` on the disabled path and on the
                // speculative verifier (Gemma4 captures no logprobs).
                let logprobs = s.logprobs.as_ref().map(|lp| {
                    let chosen_surface = tokenizer_ref
                        .id_to_token(s.token_id)
                        .unwrap_or_else(|| format!("<unk:{}>", s.token_id));
                    crate::openai::resolve_logprobs(lp, &chosen_surface, &tokenizer_ref)
                });
                let tok = GenerationToken {
                    token_id: s.token_id,
                    piece: visible,
                    done: false,
                    finish_reason: None,
                    is_thinking,
                    logprobs,
                };
                if tx_ref.blocking_send(Ok(tok)).is_err() {
                    *cancelled_ref = true;
                }
                // forced-close on budget overflow (see standard path).
                if let Some(sm) = think_splitter_ref.as_mut() {
                    if sm.take_force_close() {
                        if let Some(end_id) = thinking_end_token_id {
                            return Some(end_id);
                        }
                    }
                }
                None
            };

            let result = if let Some(drafter_arc) = eagle3_drafter.as_ref() {
                // EAGLE-3 round-loop (greedy). Autoregressive draft +
                // multi-layer hidden capture + d2t remap + GDN snapshot/restore
                // rollback + raw embed against the Qwen3.6-MoE verifier.
                // temp>0 stochastic deferred.
                let mut drafter = drafter_arc.lock();
                rmlx_models::speculative::eagle3::eagle3_generate_greedy(
                    &dispatcher.verifier,
                    &mut drafter,
                    &tokenizer,
                    &prompt_tokens,
                    n_tokens,
                    block_size,
                    kv_quant_override,
                    max_ctx_override,
                    &eos_ids,
                    &mut step_fn,
                    dispatcher.device(),
                )
            } else if let Some(drafter_arc) = dflash_drafter.as_ref() {
                // DFlash round-loop (greedy). Multi-layer hidden capture +
                // GDN snapshot/restore rollback + raw embed against the
                // Qwen3.6-MoE verifier. temp>0 stochastic deferred.
                let mut drafter = drafter_arc.lock();
                rmlx_models::speculative::dflash::dflash_generate_greedy(
                    &dispatcher.verifier,
                    &mut drafter,
                    &tokenizer,
                    &prompt_tokens,
                    n_tokens,
                    block_size,
                    kv_quant_override,
                    max_ctx_override,
                    &eos_ids,
                    &mut step_fn,
                    dispatcher.device(),
                )
            } else if let Some(assistant) = mtp_assistant.as_ref() {
                // Gemma4-assistant shared-K/V MTP round-loop (greedy).
                // temp>0 stochastic is deferred (greedy-only first cut).
                rmlx_models::speculative::gemma4_assistant::mtp_assistant_generate_greedy(
                    &dispatcher.verifier,
                    assistant,
                    &tokenizer,
                    &prompt_tokens,
                    n_tokens,
                    block_size,
                    kv_quant_override,
                    max_ctx_override,
                    &eos_ids,
                    &mut step_fn,
                    dispatcher.device(),
                )
            } else if let Some(drafter_arc) = mtp_drafter.as_ref() {
                // MTP sidecar round-loop (greedy). Penultimate-hidden capture +
                // autoregressive draft_n over the reused Qwen3.5-MoE decoder
                // layer + deferred greedy acceptance + GDN snapshot/restore
                // rollback against the Qwen3.5/3.6-MoE verifier. temp>0
                // stochastic deferred.
                let mut drafter = drafter_arc.lock();
                rmlx_models::speculative::mtp::mtp_generate_greedy(
                    &dispatcher.verifier,
                    &mut drafter,
                    &tokenizer,
                    &prompt_tokens,
                    n_tokens,
                    block_size,
                    kv_quant_override,
                    max_ctx_override,
                    &eos_ids,
                    &mut step_fn,
                    dispatcher.device(),
                )
            } else {
                dispatcher.spec_generate_greedy(
                    &tokenizer,
                    &prompt_tokens,
                    n_tokens,
                    k,
                    kv_quant_override,
                    max_ctx_override,
                    prompt_cache_slots,
                    &eos_ids,
                    &mut step_fn,
                    // A6.2: see speculative-constraint warning above.
                    None,
                    // greedy when temp==0, Leviathan stochastic when temp>0.
                    &spec_sampler_cfg,
                )
            };

            if cancelled {
                return;
            }

            // M30: compute ITL stats from step timestamps and emit (same as Gemma4Generator).
            {
                let itl_opt = compute_itl_stats(&step_timestamps);
                if let Some((p50, p95, p99, mean, spikes)) = itl_opt {
                    let step_count = step_timestamps.len();
                    tracing::debug!(
                        model_id = %model_id_for_log,
                        step_count,
                        p50_ms = p50,
                        p95_ms = p95,
                        p99_ms = p99,
                        mean_ms = mean,
                        itl_spikes = spikes,
                        "spec generate: ITL stats (M30)"
                    );
                    if let Some(ref store) = itl_store {
                        let mut ring = store.lock();
                        if ring.len() >= crate::openai::ITL_RING_CAPACITY {
                            ring.pop_front();
                        }
                        ring.push_back(ItlSample {
                            model_id: model_id_for_log.clone(),
                            p50_ms: p50,
                            p95_ms: p95,
                            mean_ms: mean,
                            step_count,
                        });
                    }
                    if let Some(ref drainer) = metrics_drainer {
                        use crate::metrics_drainer::{MetricEvent, MetricKind};
                        let ts = spsc_ts();
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_quant_label(kv_quant_override),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::ItlStats {
                                p50_ms: p50,
                                p95_ms: p95,
                                mean_ms: mean,
                                step_count,
                            },
                        });
                        // F9: p99 and spike count as separate metric events.
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_quant_label(kv_quant_override),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::ItlP99Ms(p99),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_quant_label(kv_quant_override),
                            ts_utc: ts,
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::ItlSpikes(spikes),
                        });
                    }
                    // write ITL percentiles to the events table.
                    // only emit on success (mirrors kv_cache_bytes gate).
                    if result.is_ok() {
                        if let Some(ref rec) = event_recorder {
                            let quant_label = kv_quant_label(kv_quant_override);
                            record_itl_percentiles(
                                rec,
                                &model_id_for_log,
                                &quant_label,
                                p50,
                                p95,
                                p99,
                            );
                        }
                    }
                }
            }

            match result {
                Err(e) => {
                    tracing::debug!(model_id = %model_id_for_log, error = %e, "spec generate: error");
                    let _ = tx.blocking_send(Err(e));
                }
                Ok(steps) if steps.is_empty() => {
                    let _ = tx.blocking_send(Err(Error::Other(
                        "speculative generation produced zero tokens".to_owned(),
                    )));
                }
                Ok(steps) => {
                    let last_id = steps.last().map_or(0, |s| s.token_id);
                    let finish_reason = if eos_ids.contains(&last_id) {
                        "stop".to_owned()
                    } else {
                        "length".to_owned()
                    };
                    // F6/L18: emit KV-cache bytes to SPSC drainer (SpeculativeGenerator
                    // always wraps a Gemma4 verifier, so the global static applies).
                    {
                        let kv_bytes = rmlx_models::gemma4::gemma4_kv_cache_bytes();
                        if kv_bytes > 0 {
                            if let Some(ref drainer) = metrics_drainer {
                                use crate::metrics_drainer::{MetricEvent, MetricKind};
                                drainer.try_emit(MetricEvent {
                                    model_id: model_id_for_log.clone(),
                                    kv_quant: kv_quant_label(kv_quant_override),
                                    ts_utc: spsc_ts(),
                                    ctx_max: effective_max_ctx_val,
                                    kind: MetricKind::KvCacheBytes(kv_bytes),
                                });
                            }
                            // also write to events table.
                            // Already inside Ok(steps) success branch — always emit.
                            if let Some(ref rec) = event_recorder {
                                let quant_label = kv_quant_label(kv_quant_override);
                                if let Err(e) = rec.record(&Measurement {
                                    model_path: &model_id_for_log,
                                    quant_mode: &quant_label,
                                    stage: "request",
                                    op: "kv_cache_bytes",
                                    value_unit: "bytes",
                                    value: kv_bytes as f64,
                                    notes: "",
                                }) {
                                    tracing::warn!(error = %e, op = "kv_cache_bytes", "events-table write failed");
                                }
                            }
                        }
                    }
                    // C7: emit Metal allocator high-water at the same boundary.
                    if let Some(peak_bytes) = rmlx_mlx::mlx_peak_memory_bytes() {
                        let peak_mb = peak_bytes / 1_048_576;
                        tracing::info!(
                            model_id = %model_id_for_log,
                            metal_peak_alloc_mb = peak_mb,
                            "generate: metal peak alloc speculative (C7)"
                        );
                        if let Some(ref drainer) = metrics_drainer {
                            use crate::metrics_drainer::{MetricEvent, MetricKind};
                            drainer.try_emit(MetricEvent {
                                model_id: model_id_for_log.clone(),
                                kv_quant: kv_quant_label(kv_quant_override),
                                ts_utc: spsc_ts(),
                                ctx_max: effective_max_ctx_val,
                                kind: MetricKind::MetalPeakAllocMb(peak_mb),
                            });
                        }
                    }
                    // A10: flush any withheld multi-byte tail (see
                    // Gemma4Generator site for rationale).
                    if !cancelled {
                        match detok.finalize(&tokenizer) {
                            Ok(tail) if !tail.is_empty() => {
                                let _ = tx.blocking_send(Ok(GenerationToken {
                                    token_id: last_id,
                                    piece: tail,
                                    done: false,
                                    finish_reason: None,
                                    is_thinking: false,
                                    logprobs: None,
                                }));
                            }
                            Ok(_) => {}
                            Err(e) => tracing::debug!(
                                error = ?e,
                                "A10 detok.finalize error, dropping tail"
                            ),
                        }
                    }
                    let done_tok = GenerationToken {
                        token_id: last_id,
                        piece: String::new(),
                        done: true,
                        finish_reason: Some(finish_reason),
                        // A3: see Gemma4Generator done-token comment.
                        is_thinking: false,
                        logprobs: None,
                    };
                    let _ = tx.blocking_send(Ok(done_tok));
                    tracing::debug!(model_id = %model_id_for_log, "spec generate: blocking thread done");
                }
            }
        });

        let token_stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(token_stream)
    }
}
