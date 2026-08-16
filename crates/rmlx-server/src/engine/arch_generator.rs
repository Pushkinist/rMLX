//! `ArchGenerator` — architecture-agnostic HTTP generation engine backed by `rmlx_models::arch`.
//!
//! EXEMPTION: This file exceeds 1000 LOC (≈1300). The ArchGenerator::generate
//! method is a single large decode loop (text + image, streaming + blocking,
//! all arch variants) that cannot be split without introducing artificial
//! indirection. All other engine submodules are within the 1000 LOC limit.

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use futures::stream::{self, Stream};
use rmlx_core::Error;
use rmlx_metrics::events::Measurement;

use crate::openai::ItlSample;

use super::audio::{build_audio_prompt, AudioBundle};
use super::generator::Generator;
use super::helpers::{
    compute_itl_stats, is_reconstructible_tool_marker, kv_quant_label, record_itl_percentiles,
    resolve_kv_quant_for_load, spsc_ts,
};
use super::image::{build_image_prompt, run_qwen3vl_image, VisionBundle};
use super::think::ThinkSplitter;
use super::types::{GenerationRequest, GenerationToken, ModelLoadConfig};

/// Reject the unsupported image + audio combination in a single request.
///
/// The fused `generate_image` embeds entry carries a single `(aug_ids, embeds,
/// masked_ids)` triple, so only one modality block can be scattered per
/// request. Returns a CLEAR request-level error when BOTH an image and an audio
/// clip are present (never a silent drop of the audio), and `None` otherwise so
/// the audio-only / image-only / text-only paths route unchanged.
fn reject_combined_image_audio(has_image: bool, has_audio: bool) -> Option<Error> {
    if has_image && has_audio {
        Some(Error::Other(
            "combined image + audio input in one request is not supported; \
             send them in separate turns"
                .to_owned(),
        ))
    } else {
        None
    }
}

// ── ArchGenerator ─────────────────────────────────────────────────────────────

/// Architecture-agnostic HTTP generation engine. Loads any registry arch via
/// `rmlx_models::arch::load_model` and dispatches through the `Architecture` enum.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal generator implementation — field set is coupled to the model-load lifecycle; adding a field requires updating from_snapshot and all constructors"
)]
pub struct ArchGenerator {
    model: Arc<rmlx_models::arch::Architecture>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: rmlx_mlx::Device,
    model_id: String,
    /// Serialises concurrent calls. A second request blocks until the first
    /// finishes. `tracing::warn!` fires so the operator can see contention.
    /// Serialisation covers the CPU path too: the Metal context is exclusive
    /// per process and MLX allocates compute threads per stream, so concurrent
    /// generation is unsafe regardless of whether a request runs on GPU or CPU.
    /// Wrapped in `Arc` so a clone of the handle can be moved into `spawn_blocking`.
    _lock: Arc<Mutex<()>>,
    /// Server-startup-time KV quantization override. `None` = arch default.
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    /// True when `--kv-quant <explicit>` was given at startup (not "auto").
    ///
    /// When `false` (auto mode), each request selects its KV mode via
    /// `kv_quant_for_ctx(prompt_len)`. When `true`, the
    /// user-specified quant in `kv_quant_override` is used for every request
    /// regardless of context length.
    kv_quant_user_explicit: bool,
    /// Server-startup-time max context length override. `None` = derive from mpe (capped at 4096).
    max_ctx_override: Option<i32>,
    /// Number of prompt-cache slots for multi-slot prefix matching. Default 4.
    prompt_cache_slots: usize,
    /// EOS token ids parsed from config.json at load time. Empty when the
    /// field is missing — generate_greedy then runs to max_tokens.
    eos_ids: Arc<Vec<u32>>,
    /// A2: effective max prompt-context length used by the per-request guard.
    /// `min(max_ctx_override, max_position_embeddings, KV_MAX_SEQ_DEFAULT=4096)`.
    effective_max_ctx: usize,
    /// A10: detokenization family classified from `tokenizer.json`'s
    /// `decoder` node (mirrors mlx-lm). Drives the per-arch leading-space
    /// rule in the streaming UTF-8 token-healer. The byte-level withholding
    /// guard applies regardless of kind.
    tokenizer_kind: crate::detokenizer::TokenizerKind,
    /// Gemma4 vision tower + multimodal embedder + image processor,
    /// loaded once at startup when the snapshot ships a `vision_config`.
    /// `None` for text-only checkpoints (the image path is then rejected with
    /// a clear error). Wrapped in `Arc` so a clone moves into `spawn_blocking`.
    vision: Option<Arc<VisionBundle>>,
    /// Gemma4 Conformer audio tower + multimodal embedder + USM feature
    /// extractor, loaded once at startup when the snapshot ships an
    /// `audio_config` + `audio_tower.*` weights. `None` for checkpoints without
    /// an audio path (the `input_audio` path is then rejected with a clear
    /// error). Wrapped in `Arc` so a clone moves into `spawn_blocking`.
    audio: Option<Arc<AudioBundle>>,
    /// Shared multimodal encoder-output cache. `None` disables
    /// caching for this generator (e.g. unit-test stubs); production passes
    /// the `AppState.mm_cache` clone via `ModelLoadConfig`.
    mm_cache: Option<Arc<rmlx_models::multimodal_cache::MultimodalCache>>,
    /// Server-startup default image-token budget for Gemma4-unified vision
    /// (`--image-max-tokens`). `None` = use the snapshot's
    /// `processor_config.json` `max_soft_tokens`. A per-request
    /// `image_max_tokens` field takes precedence over this. A no-op for
    /// text-only / non-Gemma4 generators.
    image_max_tokens: Option<usize>,
}

impl std::fmt::Debug for ArchGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchGenerator")
            .field("model_id", &self.model_id)
            .field("device", &self.device)
            .finish()
    }
}

impl ArchGenerator {
    /// Load weights, tokenizer, and derive `model_id` from the snapshot
    /// directory basename.
    ///
    /// `device = Device::Cpu` is the safe default until the S1.8
    /// thread-exhaustion bug on Metal is resolved (Stage 2).
    pub fn from_snapshot(
        model_dir: &Path,
        cfg: &ModelLoadConfig,
        gpu_gate: Arc<Mutex<()>>,
    ) -> rmlx_core::Result<Self> {
        Self::from_snapshot_with_id(model_dir, None, cfg, gpu_gate)
    }

    /// Like [`from_snapshot`] but accepts an explicit `model_id` override.
    ///
    /// When `model_id` is `None` the id is derived from the snapshot directory
    /// basename (existing behaviour). When `Some`, the supplied string is used
    /// verbatim — this lets the registry expose a custom logical id (e.g. the
    /// full model path for pi's full-path requests) without the generator
    /// re-deriving a different basename.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
    )]
    pub fn from_snapshot_with_id(
        model_dir: &Path,
        model_id_override: Option<&str>,
        load_cfg: &ModelLoadConfig,
        gpu_gate: Arc<Mutex<()>>,
    ) -> rmlx_core::Result<Self> {
        let device = load_cfg.device;
        let max_ctx_override = load_cfg.max_ctx;
        let prompt_cache_slots = load_cfg.prompt_cache_slots;

        let model_id = model_id_override.map_or_else(
            || {
                model_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_owned()
            },
            ToOwned::to_owned,
        );

        tracing::info!(model_id = %model_id, ?device, "ArchGenerator: loading model via arch dispatch");

        // Load config.json once for both kv-quant resolution and EOS-id
        // extraction.
        let cfg = rmlx_loader::load_config(model_dir)
            .map_err(|e| Error::Other(format!("load_config: {e}")))?;

        // Single shared resolver. Explicit override wins;
        // `None` (auto) falls through to the per-arch default table, with
        // `user_explicit=false` so per-request `kv_quant_for_ctx` can override.
        let (kv_quant_resolved, kv_quant_user_explicit) =
            resolve_kv_quant_for_load(&cfg, load_cfg.kv_quant, &model_id);

        // Extract EOS token ids from config.json. HuggingFace stores
        // `eos_token_id` as either a single int or an array of ints (Gemma4
        // uses [1, 106, 50] for <eos>, <end_of_turn>, ...). Empty Vec disables
        // EOS-stop in generate_greedy.
        let eos_ids = cfg.eos_token_ids();
        tracing::info!(
            model_id = %model_id,
            ?eos_ids,
            "ArchGenerator: parsed EOS token ids from config.json"
        );

        let model = rmlx_models::arch::load_model(
            model_dir,
            device,
            &rmlx_models::arch::LoadOpts {
                yarn: load_cfg.yarn,
            },
        )?;

        // Deterministically warm the resolved KV codec's MSL kernels during
        // this load (preload) window so the first user request does not pay a
        // shader cold-compile. General per-codec (keyed off
        // `KvQuant::carries_msl`); a no-op for `none`/affine and for the
        // CPU-hot-path iso/rotor families (nothing to warm). Best-effort: a warm
        // failure logs and proceeds (the kernel compiles lazily on first use).
        if let Some(kq) = kv_quant_resolved {
            // head_dim drives the kernel template / group alignment; a single
            // KV head is enough to force every pipeline to compile (the warm
            // buffer is `[1, 1, tokens, head_dim]`).
            let head_dim = cfg.head_dim().unwrap_or(0);
            if let Err(e) =
                rmlx_kv_quant::precompile::precompile_kv_codec_msl(kq, head_dim, 1, device)
            {
                tracing::warn!(error = %e, kv_quant = %kq, "ArchGenerator: KV codec MSL precompile failed (non-fatal)");
            }
        }

        let tk_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
            .map_err(|e| Error::Other(format!("load tokenizer: {e}")))?;

        // A10: classify the detokenization family from tokenizer.json's
        // `decoder` node (same heuristic mlx-lm uses). Used only for the
        // per-arch leading-space rule; the UTF-8 withholding guard is
        // unconditional. A parse failure is non-fatal — fall back to
        // `Other` (ByteLevel-like: no leading-space rule, healing still on).
        let tokenizer_kind = match std::fs::read(&tk_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        {
            Some(root) => crate::detokenizer::TokenizerKind::from_tokenizer_json(&root),
            None => crate::detokenizer::TokenizerKind::Other,
        };
        tracing::debug!(
            model_id = %model_id,
            ?tokenizer_kind,
            "ArchGenerator: classified detokenizer family (A10)"
        );

        // A2: derive effective_max_ctx for the per-request prompt-length guard.
        // Mirrors the KV-cache allocation logic in gemma4/generate.rs::max_seq:
        // - If --max-ctx given, take min(override, mpe) so the guard never
        // exceeds the model's positional capacity even when the operator
        // over-sized the KV buffer.
        // - Otherwise take min(mpe, 4096) — same fallback chain the cache uses.
        // Archs that don't expose mpe (Gemma3/Qwen2/Qwen3/Laguna) report 0 from
        // `max_position_embeddings()`; treat 0 as "unknown" and fall back to 4096.
        let mpe_raw = model.max_position_embeddings();
        let mpe: usize = if mpe_raw <= 0 { 4096 } else { mpe_raw as usize };
        let effective_max_ctx: usize = match max_ctx_override {
            Some(n) if n > 0 => (n as usize).min(mpe),
            _ => mpe.min(4096),
        };

        // load the Gemma4 vision tower once when the snapshot ships a
        // `vision_config` (multimodal checkpoint). Text-only models return
        // `None` here and the image-input path is rejected at request time.
        // Only the Gemma4 architecture has a vision tower today.
        let vision: Option<Arc<VisionBundle>> = match &model {
            // Gemma4 **unified** (12B): encoder-free vision embedder, no SigLIP
            // tower. Distinguished from the tower family by `architectures[0]`.
            rmlx_models::arch::Architecture::Gemma4(_)
                if rmlx_models::gemma4::is_unified_arch(model_dir) =>
            {
                match rmlx_models::gemma4::UnifiedVisionConfig::from_model_dir(model_dir) {
                    Ok(Some(vcfg)) => {
                        match rmlx_models::gemma4::load_unified_vision_embedder(model_dir, &vcfg) {
                            Ok(embedder) => {
                                let pc = rmlx_models::gemma4::unified_image_processor_config(&vcfg);
                                let processor = rmlx_models::gemma4::Gemma4ImageProcessor::new(pc);
                                tracing::info!(model_id = %model_id, "Gemma4-unified vision embedder loaded (encoder-free, multimodal)");
                                Some(Arc::new(VisionBundle::Gemma4Unified {
                                    embedder,
                                    processor,
                                }))
                            }
                            Err(e) => {
                                tracing::warn!(model_id = %model_id, error = %e, "unified vision embedder load failed — image input disabled");
                                None
                            }
                        }
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(model_id = %model_id, error = %e, "unified vision_config parse failed — image input disabled");
                        None
                    }
                }
            }
            rmlx_models::arch::Architecture::Gemma4(_) => {
                match rmlx_models::gemma4::Gemma4VisionConfig::from_model_dir(model_dir) {
                    Ok(Some(vcfg)) => {
                        match rmlx_models::gemma4::load_vision_tower(model_dir, &vcfg) {
                            Ok((vtower, embedder)) => {
                                match rmlx_models::gemma4::Gemma4ImageProcessor::from_model_dir(
                                    model_dir,
                                ) {
                                    Ok(processor) => {
                                        tracing::info!(model_id = %model_id, "Gemma4 vision tower loaded (multimodal)");
                                        Some(Arc::new(VisionBundle::Gemma4 {
                                            vision: vtower,
                                            embedder,
                                            processor,
                                        }))
                                    }
                                    Err(e) => {
                                        tracing::warn!(model_id = %model_id, error = %e, "image processor load failed — image input disabled");
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(model_id = %model_id, error = %e, "vision tower load failed — image input disabled");
                                None
                            }
                        }
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(model_id = %model_id, error = %e, "vision_config parse failed — image input disabled");
                        None
                    }
                }
            }
            rmlx_models::arch::Architecture::Gemma3(_) => {
                match rmlx_models::gemma3::Gemma3VisionConfig::from_model_dir(model_dir) {
                    Ok(Some(vcfg)) => {
                        match rmlx_models::gemma3::load_vision_tower(model_dir, &vcfg) {
                            Ok((vtower, projector)) => {
                                match rmlx_models::gemma3::Gemma3ImageProcessor::from_model_dir(
                                    model_dir,
                                ) {
                                    Ok(processor) => {
                                        tracing::info!(model_id = %model_id, "Gemma3 vision tower loaded (multimodal)");
                                        Some(Arc::new(VisionBundle::Gemma3 {
                                            vision: vtower,
                                            projector,
                                            processor,
                                        }))
                                    }
                                    Err(e) => {
                                        tracing::warn!(model_id = %model_id, error = %e, "image processor load failed — image input disabled");
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(model_id = %model_id, error = %e, "vision tower load failed — image input disabled");
                                None
                            }
                        }
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(model_id = %model_id, error = %e, "vision_config parse failed — image input disabled");
                        None
                    }
                }
            }
            rmlx_models::arch::Architecture::Qwen3VlMoe(_) => {
                match rmlx_models::qwen3_vl_moe::load_config_qwen3_vl(model_dir) {
                    Ok(cfg) => {
                        match rmlx_models::qwen3_vl_moe::load_vision_tower(model_dir, &cfg.vision) {
                            Ok(vtower) => {
                                let processor =
                                    rmlx_models::qwen3_vl_moe::Qwen3VlImageConfig::from_model_dir(
                                        model_dir,
                                    )
                                    .unwrap_or_default();
                                tracing::info!(model_id = %model_id, "Qwen3-VL-MoE vision tower loaded (multimodal)");
                                Some(Arc::new(VisionBundle::Qwen3VlMoe {
                                    vision: vtower,
                                    processor,
                                }))
                            }
                            Err(e) => {
                                tracing::warn!(model_id = %model_id, error = %e, "vision tower load failed — image input disabled");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(model_id = %model_id, error = %e, "config parse failed — image input disabled");
                        None
                    }
                }
            }
            _ => None,
        };

        // load the Gemma4 audio tower once when the snapshot ships an
        // `audio_config` (+ `audio_tower.*` weights). Text-only / vision-only
        // models return `None` here and the `input_audio` path is rejected at
        // request time with a clear error. Only the Gemma4 architecture has a
        // native audio tower today.
        let audio: Option<Arc<AudioBundle>> = match &model {
            rmlx_models::arch::Architecture::Gemma4(_) => {
                match super::audio::load_gemma4_audio_bundle(model_dir) {
                    Ok(Some(bundle)) => {
                        tracing::info!(model_id = %model_id, "Gemma4 audio tower loaded (multimodal)");
                        Some(Arc::new(bundle))
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(model_id = %model_id, error = %e, "audio tower load failed — audio input disabled");
                        None
                    }
                }
            }
            _ => None,
        };

        // turn the SSD prompt-cache tier ON for this model if configured
        // (`--kv-ssd-cache-gb > 0`). No-op when the tier is OFF (the default):
        // the spiller / hydrator are never installed and decode is
        // byte-identical to the RAM-only path. The arch name selects the right
        // per-arch PROMPT_CACHE; archs without a spill/hydrate impl are skipped.
        // Resolved, not declared: the per-arch PROMPT_CACHE and the layout-key
        // salt must describe the model that was built. A snapshot whose
        // declaration disagrees would otherwise select the wrong cache (or
        // none) while decode runs the other architecture's layout.
        let arch_name = model.arch_class();
        // layout-key inputs come straight from the loaded model so the
        // SSD tier salts every row with `(arch, n_layers, n_kv_heads, head_dim,
        // kv_quant)`. `attach_at_load` is a no-op when the tier is OFF, so
        // these reads are cheap on the RAM-only fast path.
        let n_layers = model.num_hidden_layers();
        let n_kv_heads = model.num_key_value_heads();
        let head_dim = model.head_dim();
        rmlx_models::ssd_tier::attach_at_load(
            arch_name,
            &model_id,
            kv_quant_resolved,
            n_layers,
            n_kv_heads,
            head_dim,
            device,
        );

        tracing::debug!(
            model_id = %model_id,
            ?kv_quant_resolved,
            ?max_ctx_override,
            prompt_cache_slots,
            effective_max_ctx,
            has_vision = vision.is_some(),
            "ArchGenerator: ready"
        );

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
            model_id,
            // C4: shared process-wide GPU gate injected by the loader so the
            // serialisation critical section in `generate` blocks across ALL
            // resident models (single Metal context per process).
            _lock: gpu_gate,
            kv_quant_override: kv_quant_resolved,
            kv_quant_user_explicit,
            max_ctx_override,
            prompt_cache_slots,
            eos_ids: Arc::new(eos_ids),
            effective_max_ctx,
            tokenizer_kind,
            vision,
            audio,
            mm_cache: load_cfg.mm_cache.clone(),
            image_max_tokens: load_cfg.image_max_tokens,
        })
    }

    /// The model id this generator serves (matches the snapshot's basename).
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl Generator for ArchGenerator {
    fn effective_max_ctx(&self) -> usize {
        self.effective_max_ctx
    }

    fn cache_stats(&self) -> Option<rmlx_models::CacheStats> {
        self.model.cache_stats()
    }

    fn kv_cache_bytes(&self) -> u64 {
        self.model.kv_cache_bytes()
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
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
    )]
    fn generate(
        &self,
        req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        // Defensive: caller (registry lookup) already validated this, but
        // mismatches are programmer errors worth surfacing clearly.
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

        // Per-request channel: the blocking generation thread sends one
        // GenerationToken at a time; the async side yields them to SSE.
        //
        // Channel bound=4 to overlap GPU decode with HTTP/SSE serialise+flush.
        // With bound=1 the producer (decode loop) blocked on every token until
        // axum had drained the previous SSE event — adding ~10–12 ms/token of
        // synchronous wall-clock that did not overlap with the next decode step.
        // 26b-a4b bench TPS rose 36 → ~60 with no Qwen35B regression.
        // 4 was chosen because 26b-a4b decode step is ~17 ms internally and
        // worst-case SSE flush is ~12 ms — 2 in-flight tokens fully cover the
        // overlap; we keep a little headroom for jittery HTTP scheduling.
        let (tx, rx) = tokio::sync::mpsc::channel::<rmlx_core::Result<GenerationToken>>(4);

        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        // A10: copied into the blocking decode thread for the streaming
        // UTF-8 token-healer (drives the per-arch leading-space rule).
        let tokenizer_kind = self.tokenizer_kind;
        let device = self.device;
        let prompt_tokens = req.prompt_tokens.clone();
        let n_tokens = req.max_tokens as usize;
        // image sources + the loaded vision bundle (clone of the Arc).
        // `req_images` is empty for text-only requests (zero extra work).
        let req_images = req.images.clone();
        let vision = self.vision.clone();
        // image-token budget override: request value wins over the
        // server-startup `--image-max-tokens` default. `None` falls through to
        // the snapshot's `processor_config.json` budget inside build_image_prompt.
        // This is the sole resolution point for the Anthropic path. For OpenAI
        // the request value was already folded into `req.image_max_tokens` in
        // chat.rs — the `.or(self.image_max_tokens)` here is a harmless safety
        // net that keeps both paths correct if call sites diverge in future.
        let image_max_tokens = req.image_max_tokens.or(self.image_max_tokens);
        // base64 `input_audio` clips + the loaded audio bundle (clone of the Arc).
        // `req_audio` is empty for non-audio requests (zero extra work).
        let req_audio = req.audio_b64.clone();
        let audio = self.audio.clone();
        let mm_cache = self.mm_cache.clone();
        // Issue #26: a per-request `kv_quant` override hot-swaps the KV codec on
        // the resident model. When present it takes precedence over the launch
        // `--kv-quant` (explicit or auto) and over the per-ctx auto policy —
        // exactly like a startup-explicit flag, but scoped to this one request.
        // The prefix/prompt cache key is namespaced by codec downstream, so a
        // codec switch never serves mismatched cached KV.
        let req_kv_quant_override = req.kv_quant_override;
        let kv_quant_user_explicit = self.kv_quant_user_explicit || req_kv_quant_override.is_some();
        let lock = Arc::clone(&self._lock);
        // Per-request ctx-based KV mode selection in auto mode.
        // When the user gave an explicit `--kv-quant <mode>` flag at startup,
        // honour it for every request. In auto mode, override the arch-resolved
        // default with `kv_quant_for_ctx(prompt_len)` so long-ctx requests
        // automatically use a quant mode suited to that context length.
        let kv_quant_override = if let Some(rq) = req_kv_quant_override {
            tracing::info!(
                ?rq,
                "generate: per-request KV-quant override active (issue #26)"
            );
            Some(rq)
        } else if self.kv_quant_user_explicit {
            self.kv_quant_override
        } else if let Some(pin) = self.model.preferred_auto_kv() {
            tracing::info!(?pin, "generate: arch pinned auto-KV default");
            Some(pin)
        } else {
            let ctx_quant = rmlx_models::kv_cache::kv_quant_for_ctx(prompt_tokens.len());
            tracing::info!(
                prompt_len = prompt_tokens.len(),
                ?ctx_quant,
                "generate: auto-KV-by-ctx selected quant"
            );
            Some(ctx_quant)
        };
        // Issue #26: a per-request `max_ctx` re-sizes the KV-ring virtual
        // ceiling for this request only (#25 lazy-grow path); `None` keeps the
        // launch `--max-ctx`. No weight touch — a ring realloc only.
        let max_ctx_override = req.max_ctx_override.or(self.max_ctx_override);
        if req.max_ctx_override.is_some() {
            tracing::info!(
                max_ctx = ?req.max_ctx_override,
                "generate: per-request max-ctx override active (issue #26)"
            );
        }
        // F2: capture effective_max_ctx for drainer MetricEvent.ctx_max field.
        let effective_max_ctx_val = self.effective_max_ctx as i64;
        // N2: route handler sets effective_prompt_cache_slots = base + active_sessions
        // when an X-Session-Id header is present. Use the override if supplied.
        let prompt_cache_slots = req
            .effective_prompt_cache_slots
            .unwrap_or(self.prompt_cache_slots);
        let eos_ids = Arc::clone(&self.eos_ids);
        // F6/L18: drainer handle for non-blocking SQLite metric emission.
        let metrics_drainer = req.metrics_drainer;
        // M30: ITL ring-buffer handle for per-request latency aggregates.
        let itl_store = req.itl_store;
        // per-event DB recorder for events-table ITL/kv_cache writes.
        // TTFT is written by the HTTP handler layer off-runtime.
        let event_recorder = req.event_recorder;
        // C5 Slice A: hold the FIFO admission guard (permit + pending-count
        // RAII) for the lifetime of the blocking decode. Moving it into the
        // spawn_blocking closure releases the permit — and decrements
        // `gpu_pending` — exactly when generation finishes (success/error).
        // `None` for non-route callers (unit tests, internal probes).
        let gpu_admission = req.gpu_admission;
        // A6.2: optional sampler constraint engine (NoOp in A6.2, real
        // json_object grammar in A6.3+). `None` means the hot decode path
        // is identical to pre-A6.2 — see `generate_greedy` decode loops.
        let mut constraint = req.constraint;
        if constraint.is_some() {
            tracing::debug!(model_id = %req.model_id, "generate: constraint engine active (A6.2)");
        }
        // A7.2: mirror the resolved sampling knobs into the rmlx-models
        // `SamplerConfig` (rmlx-models must not depend on rmlx-server). The
        // per-request `Pcg32` is created ONCE here and threaded through every
        // decode step so the random stream is contiguous and reproducible.
        // `temperature <= 0.0` keeps the untouched GPU argmax greedy path.
        let sampler_cfg = rmlx_models::SamplerConfig {
            temperature: req.sampling.temperature,
            top_p: req.sampling.top_p,
            top_k: req.sampling.top_k,
            min_p: req.sampling.min_p,
            seed: req.sampling.seed,
            // 0 = logprob capture disabled (hot-loop zero-overhead).
            top_logprobs_k: req.sampling.top_logprobs_k,
        };
        let mut sampler_rng = rmlx_models::Pcg32::new(sampler_cfg.seed_or_default());
        if sampler_cfg.sampling_active() {
            tracing::debug!(
                model_id = %req.model_id,
                temperature = sampler_cfg.temperature,
                top_p = sampler_cfg.top_p,
                top_k = sampler_cfg.top_k,
                min_p = sampler_cfg.min_p,
                seed = sampler_cfg.seed_or_default(),
                "generate: host categorical sampler active (A7.2)"
            );
        }
        // A7.3: logit-penalty config from per-request sampling params.
        // `token_history` is per-request; starts empty each generation.
        let penalty_cfg = rmlx_models::PenaltyConfig {
            rep_penalty: req.sampling.repetition_penalty,
            presence_penalty: req.sampling.presence_penalty,
            frequency_penalty: req.sampling.frequency_penalty,
            logit_bias: req.sampling.logit_bias.clone(),
        };
        let mut token_history: Vec<u32> = Vec::new();
        if penalty_cfg.penalties_active() {
            tracing::debug!(
                model_id = %req.model_id,
                rep_penalty = penalty_cfg.rep_penalty,
                presence_penalty = penalty_cfg.presence_penalty,
                frequency_penalty = penalty_cfg.frequency_penalty,
                logit_bias_len = penalty_cfg.logit_bias.len(),
                "generate: logit penalties active (A7.3)"
            );
        }
        // A6.3: handle for the step_fn closure to push `is_thinking` into
        // the constraint after each emitted token.
        let is_thinking_handle = req.is_thinking_handle;
        // A5.6: reconstruct suppressed tool-protocol special-token markers
        // into the decoded stream so the response parser can see them.
        let emit_tool_markers = req.emit_tool_markers;
        // per-request thinking budget + pre-resolved thinking-end-token id.
        let thinking_budget = req.thinking_budget;
        let thinking_end_token_id = req.thinking_end_token_id;
        // per-request delimiter overrides (default to None → ThinkSplitter
        // falls back to "<think>"/"</think>").
        let thinking_start_token = req.thinking_start_token.clone();
        let thinking_end_token = req.thinking_end_token.clone();
        // / PART 2: prefills a CLOSED `<think></think>` when the
        // request set `enable_thinking == Some(false)`, so the model answers
        // directly. `splitter_open` reflects that: thinking enabled (the
        // default) → start open (model reasons until the end delimiter); thinking
        // disabled → start closed (route output straight to `content`).
        let splitter_open = req.enable_thinking != Some(false);
        // A3: build the thinking-block splitter for reasoning-capable archs.
        // Non-reasoning archs get `None` here and bypass the matcher in step_fn.
        let supports_thinking = self.model.supports_thinking();
        let think_splitter: Option<ThinkSplitter> = if supports_thinking {
            Some(ThinkSplitter::new_for_request(
                splitter_open,
                thinking_budget,
                thinking_start_token,
                thinking_end_token,
            ))
        } else {
            None
        };
        // A6.5: for reasoning models whose chat template prefills an open
        // `<think>` block into the assistant turn, the constraint's
        // `is_thinking` handle must start as `true` so that `EngagePolicy::
        // Immediate` waits for `</think>` before engaging. Without this,
        // the very first `step_mask` fires with `is_thinking=false`, the
        // scalar constraint engages immediately, and all emitted tokens
        // (the forced `"medium"` etc.) are still inside the prefilled
        // `<think>` block from the engine's perspective — the ThinkSplitter
        // routes them to `reasoning_text` instead of `text`, producing an
        // empty `content` field in the response.
        //
        // / PART 2: when disabled thinking the splitter starts
        // closed, so the handle must start `false` to match — otherwise the
        // constraint would defer engagement waiting for a `</think>` that
        // never comes.
        if supports_thinking {
            if let Some(h) = is_thinking_handle.as_ref() {
                h.store(splitter_open, std::sync::atomic::Ordering::Release);
            }
        }

        // Run generation in tokio's blocking-task pool so the async runtime is not
        // stalled for the full generate_greedy wall-clock time.
        //
        // Use tokio::task::spawn_blocking instead of std::thread::spawn.
        // Raw std::thread::spawn bypassed tokio's blocking-thread pool — under
        // request bursts threads pile up waiting for the serialisation lock, each
        // consuming 2 MiB stack. tokio caps the pool at 512 threads and reuses
        // them across requests, eliminating per-request thread creation overhead
        // and integrating with tokio shutdown.
        //
        // The thread:
        // 1. Acquires the serialisation lock (warns on contention).
        // 2. Calls generate_greedy (returns Vec<ProbeStep> — all tokens at once
        // but with real compute between each one at decode time).
        // 3. Post-decodes each token_id with tokenizer.decode_stream() to get
        // proper UTF-8 instead of raw BPE pieces.
        // 4. Sends each GenerationToken through the channel; the bounded channel
        // (cap=4) lets the decode loop run 1-3 tokens ahead of
        // the SSE consumer so GPU work overlaps HTTP serialise+flush.
        //
        // kv_quant_override and max_ctx_override are set at server-startup time via
        // from_snapshot and passed here; None means use arch default.
        let model_id_for_log = self.model_id.clone();
        // Per-loaded-model signature folded into every multimodal-cache key so
        // a shared (multi-model `--registry`) encoder-output cache never serves
        // one model's vision/audio features to another for the same input. The
        // model id is the stable identity; same id ⇒ same sig ⇒ cache still
        // hits for repeat same-model requests.
        let model_sig = rmlx_models::multimodal_cache::model_sig(&self.model_id);
        tokio::task::spawn_blocking(move || {
            // Acquire the serialisation lock.
            let _guard = {
                let try_result = lock.try_lock();
                if let Some(g) = try_result {
                    g
                } else {
                    tracing::warn!(
                        model_id = %model_id_for_log,
                        "ArchGenerator: concurrent generation — waiting for lock \
                         (Stage 1 allows only one inflight generation at a time)"
                    );
                    lock.lock()
                }
            };

            // C5 Slice A: keep the FIFO admission guard alive for the whole
            // blocking decode. Dropping it here (closure exit) releases the
            // semaphore permit to the next FIFO waiter AND decrements
            // `gpu_pending`, on every exit path (return, panic-unwind, normal
            // completion). With the 1-permit semaphore upstream only one
            // request is ever in this closure, so the `try_lock` above always
            // succeeds — the gpu_gate stays as C4 cross-model defense.
            let _gpu_admission = gpu_admission;

            tracing::debug!(model_id = %model_id_for_log, "generate: blocking thread started");

            // Decode strategy: full-prefix decode → diff-emit per token.
            //
            // Single-token decode (perf-attempt 55994c4) was O(1) per step but
            // dropped BPE post-processing for tokens that need prefix context
            // (multi-byte UTF-8 fragments, repeated whitespace pieces, certain
            // Gemma special tokens). Empirically: on `gemma-4-26b-a4b-it-mxfp8`
            // with the longctx_4k bench prompt, ~11/32 generated tokens decoded
            // to "" — which the bench client filters out, halving measured TPS.
            //
            // Full-prefix decode: keep all generated ids, decode the
            // full prefix each step, emit the byte-level diff vs previously
            // emitted text. Cost is O(N²) bytes-decoded but in practice <0.5ms
            // per step at N≤512, well below the 16+ ms forward step.
            //
            // Avoids DecodeStream because of the cross-request panic
            // documented in 1cc775f (state was leaking between requests).
            //
            // A10: the full-prefix decode + byte-diff is now owned by
            // `StreamingDetokenizer`, which adds the UTF-8 token-healing
            // withholding guard (never emit / advance past a `�`-terminated
            // boundary — a multi-byte codepoint split across two token ids)
            // ported from HF `DecodeStream` / mlx-lm `_try_flush`. ASCII /
            // already-complete codepoints stay byte-identical to pre-A10.
            let mut detok = crate::detokenizer::StreamingDetokenizer::new(tokenizer_kind);

            // M30: pre-allocated timestamp vec for per-token step timing.
            // Capacity = n_tokens so no realloc during decode (constraint: pre-allocate).
            // Timestamps are collected in step_fn and converted to intervals post-decode.
            let mut step_timestamps: Vec<Instant> = Vec::with_capacity(n_tokens);

            // cancelled: set by the step callback when the SSE receiver has gone
            // away; generate_greedy will still push to `steps` but the sends are
            // skipped by the early-return below.
            let mut cancelled = false;
            // A3: think_splitter is `None` for non-reasoning archs; the
            // step_fn closure pattern-matches on the ref and short-circuits.
            let mut think_splitter = think_splitter;
            let tx_ref = &tx;
            let cancelled_ref = &mut cancelled;
            let detok_ref = &mut detok;
            let timestamps_ref = &mut step_timestamps;
            let think_splitter_ref = &mut think_splitter;

            // step_fn is called by generate_greedy inside the decode loop, once
            // per produced token. Each call sends one GenerationToken with
            // done=false to the SSE consumer. Channel cap=4 lets the
            // decode loop run a small number of tokens ahead of the SSE consumer
            // so GPU work and HTTP serialise+flush overlap; cap=1 was the prior
            // setting and forced strict per-token serialisation, costing ~10ms/token.
            let tokenizer_ref = tokenizer.clone();
            // A10: full-prefix decode → byte-diff with UTF-8 token-healing,
            // owned by `StreamingDetokenizer`. `step` returns "" when the
            // current token leaves a multi-byte codepoint incomplete; the
            // tail is flushed once at end-of-stream into the `done` sentinel.
            // `step_fn` returns `Option<u32>` — the mlx-vlm
            // `ThinkingBudgetCriteria.__call__` contract. `Some(id)` asks
            // the decode loop to force `id` as the NEXT decode input,
            // returning the model to the answer channel after the thinking
            // budget was exceeded. `None` is the unconditional default —
            // the budget-unset path always returns `None` after a single
            // `Option`-discriminant + bool check, so the hot loop is
            // unchanged when no budget is set.
            let mut step_fn = |s: &rmlx_models::ProbeStep| -> Option<u32> {
                // M30: record step arrival time for ITL computation.
                // Instant::now() on macOS is a single rdtsc-equivalent syscall,
                // ~3 ns; negligible vs the 8–15 ms decode step.
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
                // A5.6: the Gemma-4 tool markers (`<|tool_call>`,
                // `<tool_call|>`, `<|"|>`) are special tokens stripped by
                // `decode(skip_special=true)`, so `text` is empty for them.
                // When a tool parser is active, reconstruct the marker
                // surface from the raw token id so the parser sees it.
                // Restricted to exactly these three — they never occur
                // outside a tool call, so visible output is unaffected;
                // other Gemma specials (`<turn|>`, `<|channel>`) stay
                // suppressed as before.
                if emit_tool_markers && text.is_empty() {
                    if let Some(surface) = tokenizer_ref.id_to_token(s.token_id) {
                        if is_reconstructible_tool_marker(&surface) {
                            text = surface;
                        }
                    }
                }
                // A3: route the visible piece through the think-splitter
                // state machine when the model architecture supports
                // reasoning tokens. Non-reasoning archs bypass the
                // matcher entirely and emit `is_thinking = false`.
                let (visible, is_thinking) = match think_splitter_ref.as_mut() {
                    Some(sm) => sm.step(&text),
                    None => (text, false),
                };
                // A6.3: propagate is_thinking into the constraint engine
                // BEFORE the next decode step calls `advance`. The
                // engine state is consulted on each `advance` to decide
                // whether to scan for `{` engagement; while thinking we
                // suppress that scan to avoid locking onto example JSON
                // inside the chain of thought.
                if let Some(h) = is_thinking_handle.as_ref() {
                    h.store(is_thinking, std::sync::atomic::Ordering::Relaxed);
                }
                // resolve raw (id, logprob) decode data into the OpenAI
                // wire shape here, where the tokenizer is in scope. `None` on
                // the disabled path. The chosen token's surface uses the raw
                // tokenizer piece so it matches the alternative surfaces.
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
                tracing::trace!(
                    token_id = tok.token_id,
                    piece = %tok.piece,
                    is_thinking = tok.is_thinking,
                    "generate: step_fn sending token"
                );
                if tx_ref.blocking_send(Ok(tok)).is_err() {
                    tracing::debug!("generate: receiver dropped, cancelling further sends");
                    *cancelled_ref = true;
                }
                // if the budget was just exceeded, ask the decode loop
                // to force `</think>` next. `take_force_close()` returns
                // `false` (and `thinking_end_token_id` is unused) on every
                // request without a budget — zero extra work on the hot path.
                if let Some(sm) = think_splitter_ref.as_mut() {
                    if sm.take_force_close() {
                        if let Some(end_id) = thinking_end_token_id {
                            tracing::debug!(
                                end_id,
                                "generate: thinking budget exceeded — forcing </think>"
                            );
                            return Some(end_id);
                        }
                    }
                }
                None
            };

            // A6.2: thread the per-request constraint into the arch-dispatched
            // decode. We materialise `Option<&mut dyn ConstraintEngine>` from
            // the boxed value with an explicit `as_mut().map(...)` to avoid the
            // `as_deref_mut` lifetime-inference issue (the trait object's
            // implicit `'static` bound interacts oddly with the closure's
            // capture lifetime). The borrow is consumed by `generate_greedy`
            // before the call returns; ownership of the box stays here.
            let constraint_arg: Option<&mut dyn rmlx_models::ConstraintEngine> = constraint
                .as_mut()
                .map(|b| &mut **b as &mut dyn rmlx_models::ConstraintEngine);

            // Qwen3-VL-MoE image path is handled in a dedicated branch
            // (its vision tower output + 3D M-RoPE scatter does not fit the
            // Gemma `(aug_ids, embeds, masked_ids)` triple). When this fires it
            // produces `steps_result` directly and skips the Gemma path below.
            let qvl_image = !req_images.is_empty()
                && matches!(vision.as_deref(), Some(VisionBundle::Qwen3VlMoe { .. }));

            // image path — preprocess images, expand the prompt with the
            // per-image soft-token block (`<|image>` + N×`<|image|>` +
            // `<image|>`), encode the vision tower, build the scatter-merged
            // `inputs_embeds`, and decode from embeds. Text-only requests
            // (`req_images` empty or no vision tower) take the plain path below
            // and are byte-identical to pre-.
            let image_inputs: Option<(Vec<u32>, rmlx_mlx::Array, rmlx_mlx::Array)> =
                if !req_images.is_empty() && !qvl_image {
                    if let Some(vb) = vision.as_ref() {
                        match build_image_prompt(
                            model.as_ref(),
                            vb,
                            &req_images,
                            &prompt_tokens,
                            device,
                            mm_cache.as_deref(),
                            model_sig,
                            image_max_tokens,
                        ) {
                            Ok(triple) => Some(triple),
                            Err(e) => {
                                tracing::warn!(error = %e, "image preprocessing failed");
                                let _ = tx.blocking_send(Err(e));
                                return;
                            }
                        }
                    } else {
                        let _ = tx.blocking_send(Err(Error::Other(
                            "this model does not accept image input (no vision tower)".to_owned(),
                        )));
                        return;
                    }
                } else {
                    None
                };

            // Combined image + audio in one request is not supported. Reject
            // with a CLEAR request-level error rather than silently dropping the
            // audio (the exact class of bug this audio wiring exists to
            // eliminate) — surfaced as a proper HTTP error through the same
            // channel as the other request-rejection paths in this function.
            if let Some(err) =
                reject_combined_image_audio(!req_images.is_empty(), !req_audio.is_empty())
            {
                tracing::warn!(
                    model_id = %model_id_for_log,
                    "rejecting combined image + audio input in one request"
                );
                let _ = tx.blocking_send(Err(err));
                return;
            }

            // audio path — decode the `input_audio` clip, expand the prompt with
            // the audio soft-token block (`<|audio>` + T_sub×`<|audio|>` +
            // `<audio|>`), run the Conformer tower, build the scatter-merged
            // `inputs_embeds`, and decode from embeds. Routed through the same
            // `generate_image` fused-embeds entry as vision (both carry the
            // `(aug_ids, embeds, masked_ids)` triple). Submitting `input_audio`
            // to a model without an audio tower returns a CLEAR error here
            // (mirroring vision's no-tower rejection) — never a silent drop.
            // Image + audio in one request was already rejected above, so here
            // `image_inputs` is `None` whenever `req_audio` is non-empty.
            let audio_inputs: Option<(Vec<u32>, rmlx_mlx::Array, rmlx_mlx::Array)> = if !req_audio
                .is_empty()
                && image_inputs.is_none()
                && !qvl_image
            {
                if let Some(ab) = audio.as_ref() {
                    match build_audio_prompt(model.as_ref(), ab, &req_audio, &prompt_tokens, device)
                    {
                        Ok(triple) => Some(triple),
                        Err(e) => {
                            tracing::warn!(error = %e, "audio preprocessing failed");
                            let _ = tx.blocking_send(Err(e));
                            return;
                        }
                    }
                } else {
                    let _ = tx.blocking_send(Err(Error::Other(
                        "this model does not accept audio input (no audio tower)".to_owned(),
                    )));
                    return;
                }
            } else {
                None
            };

            // Image and audio are mutually exclusive per request (combined input
            // rejected above), so at most one of these is `Some`.
            let multimodal_inputs = image_inputs.or(audio_inputs);

            // Sampled before the generation so the byte count recorded below can
            // be attributed to *this* one. The counter is per model instance, so
            // this closes the "which model" question — but not "which
            // generation": a run that returns before its store (immediate EOS,
            // NaN prefill, any early-out) leaves the previous run's figure
            // readable on this very instance, and the rows below land in the
            // append-only observations/events tables where a wrong value cannot
            // be taken back.
            let kv_before = model.kv_cache_bytes_sample();

            let steps_result = if qvl_image {
                // SAFETY: qvl_image is true only when vision.as_deref() matched
                // Some(VisionBundle::Qwen3VlMoe{..}) two lines above, so this
                // branch is only reached when vision is Some.
                let Some(vb) = vision.as_deref() else {
                    tracing::error!(
                        model_id = %model_id_for_log,
                        "qvl_image=true but vision bundle is None — internal state error"
                    );
                    let _ = tx.blocking_send(Err(Error::Other(
                        "internal error: qvl_image set but no vision bundle".to_owned(),
                    )));
                    return;
                };
                run_qwen3vl_image(
                    model.as_ref(),
                    vb,
                    &req_images,
                    &prompt_tokens,
                    n_tokens,
                    device,
                    // Auto mode: pass None so the image branch picks its correct
                    // bf16 KV default. Honor the operator's explicit --kv-quant.
                    if kv_quant_user_explicit {
                        kv_quant_override
                    } else {
                        None
                    },
                    // Effective `--max-ctx` (launch flag or per-request override)
                    // so the image-path KV ring is sized to fit a large
                    // multi-thousand-soft-token prompt and an over-cap prompt is
                    // rejected cleanly rather than overflowing the 4096 default.
                    max_ctx_override,
                    &eos_ids,
                    &tokenizer,
                    &mut step_fn,
                    constraint_arg,
                    &sampler_cfg,
                    &mut sampler_rng,
                    &penalty_cfg,
                    &mut token_history,
                    mm_cache.as_deref(),
                    model_sig,
                )
            } else {
                match multimodal_inputs {
                    Some((aug_ids, embeds, masked_ids)) => model.generate_image(
                        &tokenizer,
                        &aug_ids,
                        embeds,
                        masked_ids,
                        n_tokens,
                        device,
                        kv_quant_override,
                        max_ctx_override,
                        prompt_cache_slots,
                        &eos_ids,
                        &mut step_fn,
                        constraint_arg,
                        &sampler_cfg,
                        &mut sampler_rng,
                        &penalty_cfg,
                        &mut token_history,
                    ),
                    None => model.generate_greedy(
                        &tokenizer,
                        &prompt_tokens,
                        n_tokens,
                        device,
                        kv_quant_override,
                        max_ctx_override,
                        prompt_cache_slots,
                        &eos_ids,
                        &mut step_fn,
                        constraint_arg,
                        &sampler_cfg,
                        &mut sampler_rng,
                        &penalty_cfg,
                        &mut token_history,
                    ),
                }
            };

            if cancelled {
                // Receiver gone — no point sending the done sentinel.
                return;
            }

            // N19: emit per-request prompt-cache stats to tracing after generation.
            // Reads the arch-specific global static (no lock contention with inference
            // since we are outside generate_greedy at this point).
            // F6/L18: also emit via SPSC drainer for SQLite persistence.
            {
                let cs_opt = model.cache_stats();
                if let Some(cs) = cs_opt {
                    let total = cs.hits + cs.misses;
                    let hit_rate = if total == 0 {
                        0.0_f64
                    } else {
                        cs.hits as f64 / total as f64
                    };
                    tracing::info!(
                        model_id = %model_id_for_log,
                        prompt_cache_hits = cs.hits,
                        prompt_cache_misses = cs.misses,
                        prompt_cache_bytes = cs.bytes,
                        prompt_cache_hit_rate = hit_rate,
                        "generate: prompt-cache stats (N19)"
                    );
                    // F6/L18: route prompt-cache stats to SQLite via SPSC drainer.
                    if let Some(ref drainer) = metrics_drainer {
                        let kv_label = kv_quant_label(kv_quant_override);
                        let ts = spsc_ts();
                        use crate::metrics_drainer::{MetricEvent, MetricKind};
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::PromptCacheHits(cs.hits),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::PromptCacheMisses(cs.misses),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::PromptCacheBytes(cs.bytes),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::BlockHits(cs.block_hits),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::BlockMisses(cs.block_misses),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::PartialHits(cs.partial_hits),
                        });
                        // surface the cross-request hot (in-RAM LRU)
                        // prompt-cache hit + eviction counters under their own
                        // registry names so cross-session reuse and LRU churn
                        // are queryable independently of the raw hit/miss split.
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::HotCacheHits(cs.hits),
                        });
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label.clone(),
                            ts_utc: ts.clone(),
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::HotCacheEvictions(cs.evictions),
                        });
                        // SSD-tier hydrate hits (RAM misses served from
                        // the on-disk .kvb tier).
                        drainer.try_emit(MetricEvent {
                            model_id: model_id_for_log.clone(),
                            kv_quant: kv_label,
                            ts_utc: ts,
                            ctx_max: effective_max_ctx_val,
                            kind: MetricKind::SsdHits(cs.ssd_hits),
                        });
                    }
                }
            }

            // N16: emit per-request KV-cache bytes to tracing.
            // Reads the counter on this model instance, written by
            // generate_greedy at request boundary — no lock contention with
            // inference at this point.
            // F6/L18: also emit via SPSC drainer for SQLite persistence.
            {
                // Attribute the byte count to this generation before recording
                // it: an unchanged store sequence means the readable figure is
                // an earlier generation's, and a reported zero means the
                // accounting is wrong rather than the cache empty. Neither is
                // recordable — skip the row and say why. `0` here falls through
                // the `kv_bytes > 0` gate below, which is the skip.
                let kv_bytes = match rmlx_models::classify_kv_bytes(
                    kv_before,
                    model.kv_cache_bytes_sample(),
                ) {
                    rmlx_models::KvBytesVerdict::Reported(n) => n,
                    rmlx_models::KvBytesVerdict::Unreported => {
                        tracing::warn!(
                            model_id = %model_id_for_log,
                            arch = model.arch_class(),
                            "generation reported no KV-cache byte count (store sequence did \
                             not advance, so it ended before its decode phase); skipping the \
                             kv_cache_bytes row rather than recording an earlier \
                             generation's figure"
                        );
                        0
                    }
                    rmlx_models::KvBytesVerdict::ReportedZero => {
                        tracing::warn!(
                            model_id = %model_id_for_log,
                            "generation reported a KV cache of 0 bytes after a real prefill — \
                             the byte accounting is wrong, not the cache; skipping the \
                             kv_cache_bytes row"
                        );
                        0
                    }
                };
                // Reuse `kv_quant_label` so payload-bearing
                // variants (RotorK*Asym, Mixed, RotK) render with their full
                // tag (e.g. `rotor_k_3_asym_v8_g128`). Previously this match
                // was inlined and missed payload variants.
                let quant_mode_owned = kv_quant_label(kv_quant_override);
                if kv_bytes > 0 {
                    tracing::info!(
                        model_id = %model_id_for_log,
                        quant_mode = quant_mode_owned.as_str(),
                        kv_cache_bytes = kv_bytes,
                        "generate: kv-cache bytes (N16)"
                    );
                    // F6/L18: route kv_cache_bytes to SQLite via SPSC drainer.
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
                    // also write to the events table for admission
                    // controller and real-time KV budget tracking.
                    // only emit on successful generation (steps_result.is_ok()).
                    // On error the engine sends Err and returns; the successful
                    // attempt is always the unique emitter — no flag needed.
                    if steps_result.is_ok() {
                        if let Some(ref rec) = event_recorder {
                            if let Err(e) = rec.record(&Measurement {
                                model_path: &model_id_for_log,
                                quant_mode: quant_mode_owned.as_str(),
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
                // C7: emit Metal allocator high-water at the same request boundary.
                if let Some(peak_bytes) = rmlx_mlx::mlx_peak_memory_bytes() {
                    let peak_mb = peak_bytes / 1_048_576;
                    tracing::info!(
                        model_id = %model_id_for_log,
                        metal_peak_alloc_mb = peak_mb,
                        "generate: metal peak alloc (C7)"
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
            }

            // M30: compute ITL p50/p95/mean from per-step timestamps and emit.
            //
            // Computed once, at request end — never inside the hot decode loop.
            // step_timestamps contains one Instant per produced token; intervals
            // are the gaps between consecutive arrivals. Minimum 2 tokens needed
            // for at least one interval; single-token responses skip ITL silently.
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
                        "generate: ITL stats (M30)"
                    );
                    // Write to the ITL ring buffer for /metrics/cache.
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
                    // Emit SPSC event carrying all five aggregates (p50/p95/mean in
                    // ItlStats; p99 and spikes as separate F9 events at same boundary).
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
                    // only emit on success (mirrors kv_cache_bytes gate above).
                    if steps_result.is_ok() {
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

            match steps_result {
                Err(e) => {
                    tracing::debug!(model_id = %model_id_for_log, error = %e, "generate: greedy error");
                    let _ = tx.blocking_send(Err(e));
                }
                Ok(steps) if steps.is_empty() => {
                    let _ = tx.blocking_send(Err(Error::Other(
                        "generation produced zero tokens".to_owned(),
                    )));
                }
                Ok(steps) => {
                    // finish_reason = "stop" when generation halted on
                    // EOS (last emitted token id is in the configured eos set);
                    // "length" when we hit max_tokens. Stop-string matching is
                    // still Stage 2.
                    let last_id = steps.last().map_or(0, |s| s.token_id);
                    let finish_reason = if eos_ids.contains(&last_id) {
                        "stop".to_owned()
                    } else {
                        "length".to_owned()
                    };
                    // A10: flush any withheld multi-byte tail before the
                    // sentinel. Non-empty only when generation genuinely
                    // ended mid-codepoint (lossy-replace per mlx-lm
                    // `_try_flush(force=True)`); normally empty. Sent as a
                    // regular content delta so the SSE/Anthropic handlers
                    // include it; `is_thinking=false` is correct here
                    // because the rMLX reasoning archs split on the
                    // `</think>` tag, never mid-codepoint.
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
                    // Send a done=true sentinel so the SSE consumer can set
                    // finish_reason. piece is empty — no visible delta content.
                    let done_tok = GenerationToken {
                        token_id: last_id,
                        piece: String::new(),
                        done: true,
                        finish_reason: Some(finish_reason),
                        // A3: terminal sentinel always lives on the
                        // content channel; SSE handlers skip empty pieces
                        // anyway and only honour `finish_reason` here.
                        is_thinking: false,
                        logprobs: None,
                    };
                    let _ = tx.blocking_send(Ok(done_tok));
                    tracing::debug!(model_id = %model_id_for_log, "generate: blocking thread done");
                }
            }
        });

        // Convert the mpsc receiver into a futures::Stream.
        // stream::unfold drives the receiver: each poll awaits exactly one token,
        // giving axum SSE the opportunity to flush between tokens.
        let token_stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(token_stream)
    }
}

#[cfg(test)]
#[path = "arch_generator_tests.rs"]
mod arch_generator_tests;
