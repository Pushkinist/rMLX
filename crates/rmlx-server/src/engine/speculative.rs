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

// ── MTP draft dispatch ────────────────────────────────────────────────────────

/// Which drafter loader an `--draft-kind mtp` draft model routes to, decided by
/// the draft's detected architecture family (never a substring leak).
///
/// `--draft-kind mtp` historically fronted two structurally different loaders:
/// the Qwen3.5-MoE MTP sidecar head (`MtpDrafter`) and the Gemma4 assistant
/// drafter (`Gemma4AssistantDrafter`). A draft whose family backs neither must
/// be rejected at load — see issue #23: a plain `Gemma4ForConditionalGeneration`
/// snapshot used to fall through to the Qwen3.5 sidecar loader and leak a
/// confusing `text_config missing num_experts` error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpDraftFamily {
    /// Dedicated Gemma4 assistant drafter snapshot (`gemma4_assistant`).
    Gemma4Assistant,
    /// Qwen3.5-MoE MTP sidecar head (`qwen3_5_mtp`).
    Qwen35Mtp,
    /// Family that cannot back an MTP draft — reject with a typed error.
    Unsupported,
}

/// Classify an `--draft-kind mtp` draft by its `architectures[0]` and
/// `model_type`. Both fields are checked because mlx-community snapshots set the
/// family on one or the other depending on the export tool.
///
/// Empty-config fall-through: when BOTH `arch` AND `model_type` are absent/empty
/// we treat the snapshot as a Qwen3.5 MTP sidecar. Real-world Qwen3.6 MTP
/// sidecars (e.g. `mlx-community/Qwen3.6-35B-A3B-MTP-5bit`) carry
/// `model_type=qwen3_5_mtp` but an absent `architectures` array — the downstream
/// `MtpDrafter::load` (mtp.rs:~288) only warns on a mismatch and proceeds by
/// tensor names. The issue #23 fix targets *populated* foreign families
/// (e.g. `Gemma4ForConditionalGeneration`), not blanks.
fn classify_mtp_draft(arch: &str, model_type: &str) -> MtpDraftFamily {
    if model_type == "gemma4_assistant" || arch.contains("Gemma4Assistant") {
        MtpDraftFamily::Gemma4Assistant
    } else if arch.contains("qwen3_5_mtp") || model_type.contains("qwen3_5_mtp") {
        // `arch` checked via substring to tolerate minor variant suffixes; same
        // token (`qwen3_5_mtp`) is what real Qwen3.6 MTP sidecars carry in
        // `model_type` (arch field absent in those snapshots).
        MtpDraftFamily::Qwen35Mtp
    } else if arch.is_empty() && model_type.is_empty() {
        // Both fields absent: legacy fall-through to Qwen3.5 MTP sidecar loader,
        // which warns and proceeds by tensor names. A populated foreign family
        // (e.g. plain Gemma4) is caught by the Unsupported arm below.
        MtpDraftFamily::Qwen35Mtp
    } else {
        MtpDraftFamily::Unsupported
    }
}

/// Build the actionable rejection message for an unsupported MTP draft family.
/// A plain Gemma4 model gets a family-specific hint pointing at the assistant
/// snapshot; everything else gets the generic "no MTP sidecar for this family"
/// message. Never leaks the Qwen3.5 loader's internal error.
fn mtp_reject_reason(arch: &str, model_type: &str) -> String {
    if arch.contains("Gemma4") || model_type.contains("gemma4") {
        format!(
            "draft model architecture '{arch}' is a plain Gemma4 model, which has no MTP sidecar \
             head and cannot be used with --draft-kind mtp; a Gemma4 MTP draft requires the \
             dedicated Gemma4 assistant snapshot (e.g. *-it-assistant-bf16), not a plain \
             Gemma4ForConditionalGeneration checkpoint"
        )
    } else {
        format!(
            "draft model architecture '{arch}' (model_type '{model_type}') is not a supported \
             --draft-kind mtp family; MTP supports the Qwen3.5-MoE sidecar head (qwen3_5_mtp) and \
             the Gemma4 assistant drafter (gemma4_assistant)"
        )
    }
}

// ── Drafter kind and round block ──────────────────────────────────────────────

/// The round block when `--draft-block-size` is absent: the verifier's own
/// token plus four drafted.
pub const DEFAULT_DRAFT_BLOCK_SIZE: usize = 5;

/// The smallest round block with room for a draft token.
pub const MIN_DRAFT_BLOCK_SIZE: usize = 2;

/// The round block a run is under: tokens the verifier scores per round, its
/// own token included.
///
/// One meaning for every drafter. The sidecar loops take this number as their
/// block and draft one fewer; the two-model loop takes [`drafted_per_round`]
/// of it and records the block back as `k + 1`. Either way `RoundStats.block_size`
/// — the field `decode_config` files a row under — is this value, so one flag
/// value is one cell whichever drafter runs.
///
/// # Errors
/// `Error::Other` for a block below [`MIN_DRAFT_BLOCK_SIZE`]. The CLI refuses
/// that at parse time; this covers a caller that is not the CLI.
fn round_block(flag: Option<usize>) -> rmlx_core::Result<usize> {
    match flag {
        None => Ok(DEFAULT_DRAFT_BLOCK_SIZE),
        Some(block) if block >= MIN_DRAFT_BLOCK_SIZE => Ok(block),
        Some(block) => Err(Error::Other(format!(
            "draft block size {block} leaves no room for a draft token; it must be at \
             least {MIN_DRAFT_BLOCK_SIZE}"
        ))),
    }
}

/// How many tokens the two-model loop drafts per round of `block` tokens.
const fn drafted_per_round(block: usize) -> usize {
    block - 1
}

/// The drafter kind a run is under.
///
/// `declared` is what the draft snapshot says it is
/// ([`rmlx_models::Declared::from_snapshot`]); `flag` is `--draft-kind`. A
/// declaration alone is enough, which is what lets a bare `--draft-model` run.
/// The flag is for a snapshot that declares nothing — an MTP sidecar exported
/// with an empty `config.json` — and is refused when it contradicts a sidecar
/// marker: no loader can build a snapshot as a kind it is not, and the
/// tensor-name error it would die with later names neither side. It is not
/// refused against the registry's full-model inference, which is not a marker
/// the snapshot carries.
///
/// # Errors
/// [`Error::SpeculativePairing`] when neither side names a kind, or the flag
/// contradicts a sidecar marker.
fn decide_draft_kind(
    flag: Option<rmlx_models::DraftKind>,
    declared: rmlx_models::Declared,
    arch: &str,
    model_type: &str,
) -> rmlx_core::Result<rmlx_models::DraftKind> {
    match (flag, declared) {
        (Some(f), rmlx_models::Declared::Sidecar(d)) if f != d => Err(Error::SpeculativePairing {
            reason: format!(
                "--draft-kind {f} contradicts the draft snapshot, which declares itself \
                     a {d} drafter (architectures[0] {arch:?}, model_type {model_type:?}); \
                     drop the flag or point --draft-model at a {f} snapshot"
            ),
        }),
        (Some(f), _) => Ok(f),
        (None, declared) => declared.kind().ok_or_else(|| Error::SpeculativePairing {
            reason: format!(
                "the draft snapshot's config.json identifies no drafter (architectures[0] \
                 {arch:?}, model_type {model_type:?}) — it is neither a sidecar head \
                 (mtp / dflash / eagle3) nor a registered generative architecture; pass \
                 --draft-kind to name it"
            ),
        }),
    }
}

/// The drafter a generator holds, one variant per round loop.
///
/// The handle lives inside the kind rather than in a slot beside it, so
/// dispatch is a `match` the compiler checks: a fifth kind does not compile
/// until it names its loop. Each sidecar handle is an `Arc` because the
/// `Generator` trait borrows `&self` and the handle moves into the blocking
/// decode task; the ones whose draft step mutates a KV cache carry a `Mutex`.
#[derive(Clone)]
enum Drafter {
    /// EAGLE-3 drafter beside a verifier-only dispatcher.
    Eagle3(Arc<Mutex<rmlx_models::speculative::eagle3::Eagle3Drafter>>),
    /// DFlash drafter beside a verifier-only dispatcher.
    DFlash(Arc<Mutex<rmlx_models::speculative::dflash::DFlashDrafter>>),
    /// Gemma4 assistant, the shared-K/V `mtp` family; `draft_n` borrows `&self`.
    MtpAssistant(Arc<rmlx_models::speculative::gemma4_assistant::Gemma4AssistantDrafter>),
    /// Qwen3.5-family MTP sidecar head.
    MtpSidecar(Arc<Mutex<rmlx_models::speculative::mtp::MtpDrafter>>),
    /// A full draft model, held by the dispatcher itself.
    TwoModel,
}

impl Drafter {
    fn kind(&self) -> rmlx_models::DraftKind {
        match self {
            Drafter::Eagle3(_) => rmlx_models::DraftKind::Eagle3,
            Drafter::DFlash(_) => rmlx_models::DraftKind::DFlash,
            Drafter::MtpAssistant(_) | Drafter::MtpSidecar(_) => rmlx_models::DraftKind::Mtp,
            Drafter::TwoModel => rmlx_models::DraftKind::TwoModel,
        }
    }
}

// ── SpeculativeGenerator ──────────────────────────────────────────────────────

/// Generator backed by a verifier and a drafter of one [`rmlx_models::DraftKind`].
///
/// Each round the drafter proposes up to `k` tokens; the verifier scores them
/// in one cached forward and emits the accepted prefix plus its own next
/// token. Which drafter runs is decided at construction from the draft
/// snapshot's own `config.json`, or by an explicit `--draft-kind`.
///
/// `--draft-block-size` is the round block, the verifier's token included
/// (default 5), so every loop drafts one fewer. The two-model loops accept
/// greedily at `temperature == 0` and by Leviathan stochastic acceptance above
/// it; the sidecar loops are greedy only.
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
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: Arc<Vec<u32>>,
    /// Round block: tokens the verifier scores per round, its own included.
    block_size: usize,
    /// Effective max prompt-context length for the per-request guard — the
    /// ceiling `rmlx_models::context::resolve_context` produced at load from
    /// the verifier's limits.
    effective_max_ctx: usize,
    /// The verifier's context limits. A per-request `max_ctx` override is
    /// resolved against these by the route layer.
    context_limits: rmlx_models::context::ContextLimits,
    /// A10: detokenization family from the verifier's `tokenizer.json`.
    tokenizer_kind: crate::detokenizer::TokenizerKind,
    /// The drafter, and with it the round loop. Never implicit: inferred from
    /// the draft snapshot's declaration, or named by `--draft-kind`.
    drafter: Drafter,
}

impl std::fmt::Debug for SpeculativeGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeGenerator")
            .field("model_id", &self.model_id)
            .field("device", &self.device)
            .field("block_size", &self.block_size)
            .field("draft_kind", &self.drafter.kind())
            .finish()
    }
}

impl SpeculativeGenerator {
    /// Load a verifier + draft pair from disk and build the generator.
    ///
    /// `verifier_dir` is the primary `--model` snapshot; its basename becomes
    /// the OpenAI `model_id` unless `model_id_override` names one. `draft_dir`
    /// is the drafter: a full model or a sidecar head, decided by
    /// [`decide_draft_kind`] from its `config.json` and the optional
    /// `draft_kind` flag.
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
        // Single shared resolver. The user-explicit flag it also returns is
        // only consumed by the image branch, which this generator has none of.
        let cfg = rmlx_loader::load_config(verifier_dir)
            .map_err(|e| Error::Other(format!("load_config (verifier): {e}")))?;
        let (kv_quant_resolved, _kv_quant_user_explicit) =
            resolve_kv_quant_for_load(&cfg, load_cfg.kv_quant, &model_id);

        let eos_ids = cfg.eos_token_ids();
        tracing::info!(
            model_id = %model_id,
            ?eos_ids,
            "SpeculativeGenerator: parsed EOS token ids from verifier config"
        );

        let draft_cfg = rmlx_loader::load_config(draft_dir)
            .map_err(|e| Error::Other(format!("load_config (draft): {e}")))?;
        let draft_arch = draft_cfg.architectures.first().map_or("", String::as_str);
        let draft_model_type = draft_cfg
            .extras
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let declared = rmlx_models::Declared::from_snapshot(draft_arch, draft_model_type);
        let draft_kind = decide_draft_kind(draft_kind, declared, draft_arch, draft_model_type)?;
        let block_size = round_block(draft_block_size)?;
        tracing::info!(
            draft = %draft_dir.display(),
            arch = draft_arch,
            model_type = draft_model_type,
            ?declared,
            %draft_kind,
            "SpeculativeGenerator: drafter kind"
        );

        // For a sidecar head the `--draft-model` folder is not a full model:
        // the dispatcher holds the verifier alone (`load_verifier_only`) and
        // the head is loaded beside it, driven by its own round loop. Only the
        // two-model kind has two models to load.
        let (dispatcher, drafter) = match draft_kind {
            rmlx_models::DraftKind::Eagle3 => {
                let dispatcher =
                    rmlx_models::SpeculativeDispatcher::load_verifier_only(verifier_dir, device)?;
                let hidden_size = dispatcher.verifier.hidden_size();
                let vocab_size = dispatcher.verifier.vocab_size();
                let drafter = rmlx_models::speculative::eagle3::Eagle3Drafter::load(
                    draft_dir,
                    hidden_size,
                    vocab_size,
                    &eos_ids,
                    device,
                )?;
                (dispatcher, Drafter::Eagle3(Arc::new(Mutex::new(drafter))))
            }
            rmlx_models::DraftKind::DFlash => {
                let dispatcher =
                    rmlx_models::SpeculativeDispatcher::load_verifier_only(verifier_dir, device)?;
                let hidden_size = dispatcher.verifier.hidden_size();
                let drafter = rmlx_models::speculative::dflash::DFlashDrafter::load(
                    draft_dir,
                    hidden_size,
                    device,
                )?;
                (dispatcher, Drafter::DFlash(Arc::new(Mutex::new(drafter))))
            }
            rmlx_models::DraftKind::Mtp => {
                let dispatcher =
                    rmlx_models::SpeculativeDispatcher::load_verifier_only(verifier_dir, device)?;
                let hidden_size = dispatcher.verifier.hidden_size();
                // `mtp` fronts two loaders, told apart by the draft's family; a
                // third family is refused here rather than handed to the Qwen3.5
                // loader, whose failure would name a missing MoE config instead
                // of the real mismatch.
                match classify_mtp_draft(draft_arch, draft_model_type) {
                    MtpDraftFamily::Gemma4Assistant => {
                        let drafter =
                            rmlx_models::speculative::gemma4_assistant::Gemma4AssistantDrafter::load(
                                draft_dir,
                                hidden_size,
                                device,
                            )?;
                        (dispatcher, Drafter::MtpAssistant(Arc::new(drafter)))
                    }
                    MtpDraftFamily::Qwen35Mtp => {
                        let drafter = rmlx_models::speculative::mtp::MtpDrafter::load(
                            draft_dir,
                            hidden_size,
                            device,
                        )?;
                        (
                            dispatcher,
                            Drafter::MtpSidecar(Arc::new(Mutex::new(drafter))),
                        )
                    }
                    MtpDraftFamily::Unsupported => {
                        let reason = mtp_reject_reason(draft_arch, draft_model_type);
                        tracing::error!(
                            draft = %draft_dir.display(),
                            arch = draft_arch,
                            model_type = draft_model_type,
                            "SpeculativeGenerator: MTP dispatch — rejecting unsupported draft family"
                        );
                        return Err(Error::SpeculativePairing { reason });
                    }
                }
            }
            rmlx_models::DraftKind::TwoModel => {
                let dispatcher = rmlx_models::SpeculativeDispatcher::load_speculative(
                    verifier_dir,
                    draft_dir,
                    device,
                )?;
                (dispatcher, Drafter::TwoModel)
            }
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

        // Speculative inherits the verifier's KV-cache sizing, so the
        // verifier's positional capacity is what the shared resolver bounds
        // this generator by.
        let resolved_ctx = rmlx_models::context::resolve_context(
            &dispatcher.verifier.context_limits(),
            max_ctx_override,
        )?;
        let effective_max_ctx: usize = resolved_ctx.ceiling_tokens();
        let context_limits = dispatcher.verifier.context_limits();

        // Fail fast on a codec the verifier's resolved architecture refuses.
        // The round loop builds the verifier's KV caches itself rather than
        // going through `Architecture::generate_greedy`, so without this the
        // only enforcing check would run per request — every request failing
        // after a fully successful startup. The verifier is the model whose
        // caches carry the codec, so it is the one to ask.
        if let Some(kq) = kv_quant_resolved {
            dispatcher.verifier.validate_kv_quant(kq)?;
        }

        tracing::info!(
            model_id = %model_id,
            block_size,
            ?kv_quant_resolved,
            ?max_ctx_override,
            effective_max_ctx,
            positional_max = resolved_ctx.positional_max,
            %draft_kind,
            "SpeculativeGenerator: ready"
        );

        Ok(Self {
            dispatcher: Arc::new(dispatcher),
            tokenizer: Arc::new(tokenizer),
            device,
            model_id,
            // C4: shared process-wide GPU gate (see ArchGenerator above).
            _lock: gpu_gate,
            kv_quant_override: kv_quant_resolved,
            max_ctx_override,
            prompt_cache_slots,
            eos_ids: Arc::new(eos_ids),
            block_size,
            effective_max_ctx,
            context_limits,
            tokenizer_kind,
            drafter,
        })
    }

    /// The model id this generator serves (verifier basename).
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Which drafter this generator runs.
    pub fn draft_kind(&self) -> rmlx_models::DraftKind {
        self.drafter.kind()
    }
}

impl Generator for SpeculativeGenerator {
    fn effective_max_ctx(&self) -> usize {
        self.effective_max_ctx
    }

    fn context_limits(&self) -> Option<rmlx_models::context::ContextLimits> {
        Some(self.context_limits)
    }

    fn cache_stats(&self) -> Option<rmlx_models::CacheStats> {
        // These stats belong to the prompt cache the verifier actually uses, so
        // dispatch on the verifier. The cache is one static per architecture,
        // and the verifier's architecture varies with the drafter — eagle3 /
        // dflash / mtp all require a Qwen3.5-MoE verifier — so naming a fixed
        // arch here reports some other architecture's cache for most pairs.
        self.dispatcher.verifier.cache_stats()
    }

    fn kv_cache_bytes(&self) -> u64 {
        // The KV this figure describes is the verifier's, and the counter lives
        // on the verifier instance — so read it there rather than naming an
        // arch. Verifier arch varies with the drafter (eagle3 / dflash / mtp
        // require a Qwen3.5-MoE verifier), so a hard-coded arch reads another
        // model's number for most pairs.
        self.dispatcher.verifier.kv_cache_bytes()
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

        let (tx, rx) = tokio::sync::mpsc::channel::<rmlx_core::Result<GenerationToken>>(4);

        let dispatcher = Arc::clone(&self.dispatcher);
        // The drafter moves into the blocking task and selects the round loop.
        let drafter = self.drafter.clone();
        let block_size = self.block_size;
        let tokenizer = Arc::clone(&self.tokenizer);
        // A10: detokenizer family for the streaming UTF-8 token-healer.
        let tokenizer_kind = self.tokenizer_kind;
        let prompt_tokens = req.prompt_tokens.clone();
        let n_tokens = req.max_tokens as usize;
        let lock = Arc::clone(&self._lock);
        // A per-request `kv_quant` override wins over the launch codec,
        // explicit or auto, scoped to this request only. Same resolution as
        // ArchGenerator, from the same producer.
        let kv_quant_override =
            super::helpers::kv_quant_for_request(req.kv_quant_override, self.kv_quant_override);
        tracing::info!(
            prompt_len = prompt_tokens.len(),
            per_request = req.kv_quant_override.is_some(),
            ?kv_quant_override,
            "speculative generate: KV codec for this request"
        );
        // The round loop below builds the verifier's KV caches directly, so
        // this is the enforcing check for the speculative path — the seam in
        // `Architecture::generate_greedy` is never reached from here. It covers
        // the per-request override, which arrives after startup and is
        // otherwise unvalidated.
        if let Some(kq) = kv_quant_override {
            if let Err(e) = self.dispatcher.verifier.validate_kv_quant(kq) {
                return Box::pin(stream::once(async move { Err(e) }));
            }
        }
        // Per-request max-ctx ceiling override (the lazy-grow path).
        let max_ctx_override = req.max_ctx_override.or(self.max_ctx_override);
        // F2: capture effective_max_ctx for drainer MetricEvent.ctx_max field.
        let effective_max_ctx_val = self.effective_max_ctx as i64;
        // N2: use effective_prompt_cache_slots override if set by route handler.
        let prompt_cache_slots = req
            .effective_prompt_cache_slots
            .unwrap_or(self.prompt_cache_slots);
        let eos_ids = Arc::clone(&self.eos_ids);
        let model_id_for_log = self.model_id.clone();
        // F6/L18: drainer handle for non-blocking SQLite metric emission.
        let metrics_drainer = req.metrics_drainer;
        // M30: ITL ring-buffer handle for per-request latency aggregates.
        let itl_store = req.itl_store;
        // per-event DB recorder (TTFT is written by the HTTP handler
        // layer off-runtime; only ITL/kv_cache_bytes are written here).
        let event_recorder = req.event_recorder;
        // C5 Slice A: hold the FIFO admission guard for the lifetime of the
        // blocking decode (mirrors ArchGenerator). Released on completion.
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
        // Think-splitter mirrors ArchGenerator: budget, thinking-end id, the
        // prompt-derived initial channel and per-request delimiters go through
        // the same `new_for_request` constructor as the standard path.
        let thinking_budget = req.thinking_budget;
        let thinking_end_token_id = req.thinking_end_token_id;
        let splitter_open = req.prompt_think_open;
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
        // ArchGenerator site for rationale).
        let emit_tool_markers = req.emit_tool_markers;

        // Use spawn_blocking instead of std::thread::spawn (same rationale
        // as ArchGenerator above).
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
            // (mirrors ArchGenerator). Released on closure exit.
            let _gpu_admission = gpu_admission;

            // Ensure the GPU default stream is registered for the calling
            // thread. tokio blocking-pool threads start with no GPU stream
            // context; MLX's array materialisation then fails with "There is
            // no Stream(gpu, 0) in current thread". The single-arch text path
            // establishes this in `arch::generate_greedy`, but the speculative
            // drafter round-loops dispatch into the verifier directly without
            // going through that entry, so register it here once per thread
            // entry — covers every drafter variant below with no ML-semantic
            // effect. The CPU stream is registered unconditionally (thread-local
            // since MLX 0.31/0.32) so a CPU-scheduled op does not fault here.
            rmlx_mlx::ensure_cpu_default_stream();
            if dispatcher.device() == rmlx_mlx::Device::Gpu {
                rmlx_mlx::ensure_gpu_default_stream();
            }

            tracing::debug!(model_id = %model_id_for_log, block_size, "spec generate: blocking thread started");

            // Full-prefix decode → byte-diff per token (mirrors
            // ArchGenerator). Speculative emits in bursts of accept+1
            // tokens per round; the diff-emit pattern handles bursts
            // transparently — each ProbeStep call appends one id and
            // re-decodes the full prefix.
            //
            // A10: owned by `StreamingDetokenizer` (UTF-8 token-healing —
            // see ArchGenerator site). A multi-byte codepoint split by a
            // speculative round boundary is held until the next ProbeStep
            // completes it.
            let mut detok = crate::detokenizer::StreamingDetokenizer::new(tokenizer_kind);
            // M30: pre-allocated per-step timestamps for ITL computation.
            let mut step_timestamps: Vec<Instant> = Vec::with_capacity(n_tokens);
            let mut cancelled = false;
            // A3: same shape as ArchGenerator — `None` for non-reasoning archs.
            let mut think_splitter = think_splitter;
            let tx_ref = &tx;
            let cancelled_ref = &mut cancelled;
            let detok_ref = &mut detok;
            let timestamps_ref = &mut step_timestamps;
            let think_splitter_ref = &mut think_splitter;
            let tokenizer_ref = tokenizer.clone();

            // Same `Option<u32>` forced-token contract as the standard path.
            // Every speculative loop discards it (see `emit_step`), so a
            // thinking budget's force-close is inert here.
            let mut step_fn = |s: &rmlx_models::ProbeStep| -> Option<u32> {
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
                // ArchGenerator site for the full rationale).
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

            // Sampled before the generation so the byte count emitted below can
            // be attributed to *this* one: the store sequence must advance
            // across the call, or the readable figure belongs to an earlier
            // generation. The rows below land in the append-only events table,
            // where a wrong value cannot be taken back.
            //
            // Read off the **verifier instance** — the model whose KV this
            // figure describes. The counter is a field on the model, not an
            // arch-keyed location, so this is exact even when draft and
            // verifier share an architecture.
            let kv_before = dispatcher.verifier.kv_cache_bytes_sample();

            let result = match &drafter {
                Drafter::Eagle3(drafter_arc) => {
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
                }
                Drafter::DFlash(drafter_arc) => {
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
                }
                Drafter::MtpAssistant(assistant) => {
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
                }
                Drafter::MtpSidecar(drafter_arc) => {
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
                }
                // Greedy at temperature 0, Leviathan stochastic above it; the
                // constraint is refused above, so `None` here.
                Drafter::TwoModel => dispatcher.spec_generate_greedy(
                    &tokenizer,
                    &prompt_tokens,
                    n_tokens,
                    drafted_per_round(block_size),
                    kv_quant_override,
                    max_ctx_override,
                    prompt_cache_slots,
                    &eos_ids,
                    &mut step_fn,
                    None,
                    &spec_sampler_cfg,
                ),
            };

            if cancelled {
                return;
            }

            // M30: compute ITL stats from step timestamps and emit (same as ArchGenerator).
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
                    // F6/L18: emit KV-cache bytes to the SPSC drainer, read off
                    // the verifier instance's own counter (see `kv_before`).
                    {
                        // Attribute the byte count to this generation before
                        // recording it: an unchanged store sequence means the
                        // readable figure is an earlier generation's, and a
                        // reported zero means the accounting is wrong. Neither
                        // is recordable — skip the row and say why.
                        let kv_bytes = match rmlx_models::classify_kv_bytes(
                            kv_before,
                            dispatcher.verifier.kv_cache_bytes_sample(),
                        ) {
                            rmlx_models::KvBytesVerdict::Reported(n) => n,
                            rmlx_models::KvBytesVerdict::Unreported => {
                                // Every round loop reports at the end of its
                                // decode phase, so this is now reachable only
                                // for a generation that returned before one —
                                // an immediate EOS on the prefill bonus token.
                                tracing::warn!(
                                    model_id = %model_id_for_log,
                                    arch = dispatcher.verifier.arch_class(),
                                    "speculative generation reported no KV-cache byte count \
                                     (store sequence did not advance, so it ended before its \
                                     decode phase); skipping the kv_cache_bytes row rather \
                                     than recording an earlier generation's figure"
                                );
                                0
                            }
                            rmlx_models::KvBytesVerdict::ReportedZero => {
                                tracing::warn!(
                                    model_id = %model_id_for_log,
                                    "speculative generation reported a KV cache of 0 bytes \
                                     after a real prefill — the byte accounting is wrong, not \
                                     the cache; skipping the kv_cache_bytes row"
                                );
                                0
                            }
                        };
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
                    // ArchGenerator site for rationale).
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
                        // A3: see ArchGenerator done-token comment.
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

#[cfg(test)]
#[path = "speculative_tests.rs"]
mod speculative_tests;
