// ---------------------------------------------------------------------------
// load_model + smoke probe
// ---------------------------------------------------------------------------

use std::path::Path;
use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::{Array, Device};

use super::phases::LAST_LOAD_PHASES;
use super::registry::is_arch_supported;
use super::{Architecture, LoadPhases};

/// Load a model snapshot from `model_dir`.
///
/// Reads `config.json`, dispatches on `architectures[0]`:
/// - `"Gemma4ForConditionalGeneration"` -> Architecture::Gemma4(...)
/// - Anything else -> Error::Model("architecture '...' not yet supported (v0.0.1)")
///
/// Captures per-phase timing (`mmap_ms`, `dequant_ms`, `gpu_residency_ms`,
/// `first_kernel_ready_ms`, `total_load_ms`) into `LAST_LOAD_PHASES`.
/// Read after load via `read_load_phases()`.
///
/// # Errors
/// Returns `Error::Config` if `config.json` cannot be read or parsed.
/// Returns `Error::Model` if the architecture is not yet supported.
/// Returns `Error::Loader` / `Error::Mlx` if weight loading fails.
#[tracing::instrument(skip_all, fields(model_dir = %model_dir.display()))]
pub fn load_model(model_dir: &Path, _device: Device) -> Result<Architecture> {
    let t_total_start = Instant::now();

    let cfg = load_config(model_dir)?;

    let arch_str = cfg.architectures.first().map_or("(empty)", String::as_str);

    tracing::info!(
        model_dir = %model_dir.display(),
        arch = arch_str,
        "arch::load_model: dispatching"
    );

    // Defense-in-depth: reject unknown architectures before I/O.
    // The module-level `KNOWN_ARCHS` / `is_arch_supported` are checked first
    // at serve startup; this guard fires if the caller skips that path.
    if !is_arch_supported(arch_str) {
        tracing::error!(
            arch = arch_str,
            model_dir = %model_dir.display(),
            "arch::load_model: architecture not yet supported in v0.0.1"
        );
        return Err(Error::Model(format!(
            "architecture '{arch_str}' not yet supported in v0.0.1; see arch.rs for how to add it"
        )));
    }

    // -- Phase 1: mmap -------------------------------------------------------
    let t_mmap_start = Instant::now();
    match load_shard_index(model_dir).and_then(|idx| ShardSet::open(model_dir, &idx)) {
        Ok(_) => {}
        Err(e) => {
            tracing::debug!(error = %e, "arch::load_model: mmap pre-open skipped (stale/absent index; arch loader globs shards)");
        }
    }
    let mmap_ms = t_mmap_start.elapsed().as_millis() as u64;
    tracing::debug!(mmap_ms, "arch::load_model: mmap phase complete");

    // -- Phase 2: dequant (tensor decode) ------------------------------------
    let t_dequant_start = Instant::now();

    let arch = match arch_str {
        "Gemma4ForConditionalGeneration" => {
            let model = if cfg.is_paroquant() {
                tracing::info!(
                    "arch::load_model: Gemma4 PARO checkpoint detected, using PARO loader"
                );
                crate::gemma4::load_from_path_paro(model_dir)?
            } else {
                crate::gemma4::load_from_path(model_dir)?
            };
            let a = Architecture::Gemma4(model);
            tracing::info!(summary = %a.config_summary(), "arch::load_model: loaded Gemma4");
            a
        }
        "Gemma3ForConditionalGeneration" => {
            let model = crate::gemma3::load_from_path(model_dir)?;
            let a = Architecture::Gemma3(model);
            tracing::info!(summary = %a.config_summary(), "arch::load_model: loaded Gemma3");
            a
        }
        "Qwen2ForCausalLM" => {
            let model = crate::qwen2::load_from_path(model_dir)?;
            let a = Architecture::Qwen2(model);
            tracing::info!(summary = %a.config_summary(), "arch::load_model: loaded Qwen2");
            a
        }
        "Qwen3ForCausalLM" => {
            let model = crate::qwen3::load_from_path(model_dir)?;
            let a = Architecture::Qwen3(model);
            tracing::info!(summary = %a.config_summary(), "arch::load_model: loaded Qwen3");
            a
        }
        "LagunaForCausalLM" => {
            let model = crate::laguna::load_from_path(model_dir)?;
            let a = Architecture::Laguna(model);
            tracing::info!(summary = %a.config_summary(), "arch::load_model: loaded Laguna");
            a
        }
        "Qwen3_5MoeForConditionalGeneration" => {
            let model = crate::qwen3_5_moe::load_from_path(model_dir)?;
            let a = Architecture::Qwen3_5Moe(model);
            tracing::info!(
                summary = %a.config_summary(),
                "arch::load_model: loaded Qwen3_5Moe"
            );
            a
        }
        // Dense Qwen3.5 variant (ParoQuant INT4 checkpoints from z-lab).
        // Shares the same GatedDeltaNet + FullAttention hybrid backbone as the MoE
        // variant but uses a dense SwiGLU MLP instead of the sparse MoE block.
        "Qwen3_5ForConditionalGeneration" => {
            let model = crate::qwen3_5_moe::load_from_path_paro(model_dir)?;
            let a = Architecture::Qwen3_5Moe(model);
            tracing::info!(
                summary = %a.config_summary(),
                "arch::load_model: loaded Qwen3_5 (PARO dense)"
            );
            a
        }
        "Qwen3VLMoeForConditionalGeneration" => {
            let model = crate::qwen3_vl_moe::load_from_path(model_dir)?;
            let a = Architecture::Qwen3VlMoe(model);
            tracing::info!(
                summary = %a.config_summary(),
                "arch::load_model: loaded Qwen3VlMoe"
            );
            a
        }
        "BitNetForCausalLM" => {
            let model = crate::bitnet::load_from_path(model_dir)?;
            let a = Architecture::BitNet(model);
            tracing::info!(summary = %a.config_summary(), "arch::load_model: loaded BitNet");
            a
        }
        // jina-embeddings-v4 is an encoder served via `/v1/embeddings`, not
        // the generative path. It is in KNOWN_ARCHS so the registry accepts
        // it, but it has no `Architecture` variant — refuse load_model here
        // with a clear pointer to the embedding route (no panic).
        "JinaEmbeddingsV4Model" => {
            return Err(Error::Model(
                "architecture 'JinaEmbeddingsV4Model' is an embedding model — \
                 use POST /v1/embeddings, not the generative load/chat path"
                    .to_owned(),
            ));
        }
        // The early KNOWN_ARCHS check above guarantees this branch is unreachable
        // at runtime. If it fires, a new arch was added to KNOWN_ARCHS without a
        // corresponding implementation arm — surface as a typed error rather than
        // an abrupt halt so the caller can report the BUG cleanly.
        _ => {
            tracing::error!(
                arch = arch_str,
                "arch passed KNOWN_ARCHS guard but has no implementation arm — \
                 update load_model match when adding a new architecture"
            );
            return Err(Error::ArchUnsupported {
                arch: arch_str.to_owned(),
            });
        }
    };

    let dequant_ms = t_dequant_start.elapsed().as_millis() as u64;
    tracing::debug!(dequant_ms, "arch::load_model: dequant phase complete");

    // -- Phase 3: gpu_residency -----------------------------------------------
    // MLX dispatches lazily -- arrays are not pushed to GPU during load.
    // gpu_residency_ms is documented as 0 (see LoadPhases doc).
    let gpu_residency_ms: u64 = 0;

    // -- Phase 4: first_kernel_ready ------------------------------------------
    let t_warmup_start = Instant::now();
    {
        use rmlx_mlx::Dtype;
        match Array::from_bytes(&[], &[0i32], Dtype::F32) {
            Ok(arr) => {
                let _ = arr.to_bytes(); // forces sync evaluation of the empty array
                tracing::debug!("arch::load_model: warmup kernel dispatched");
            }
            Err(e) => {
                tracing::debug!(error = %e, "arch::load_model: warmup array creation failed, first_kernel_ready_ms=0");
            }
        }
    }
    let first_kernel_ready_ms = t_warmup_start.elapsed().as_millis() as u64;
    tracing::debug!(
        first_kernel_ready_ms,
        "arch::load_model: warmup phase complete"
    );

    // -- Phase 5: GDN compile-cache pre-warm ----------------------------------
    // For Qwen3_5Moe models: pre-trace `gated_delta_prefill_ops` at the
    // qwen3_5_moe prefill chunk size so the compile-cache is hot before the
    // first real request.
    if _device == Device::Gpu {
        if let Architecture::Qwen3_5Moe(ref m) = arch {
            let t_gdn_warm = Instant::now();
            let b: i32 = 1;
            let gdn_warmup_t: i32 = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe") as i32;
            let hk = m.cfg.linear_num_key_heads as i32;
            let hv = m.cfg.linear_num_value_heads as i32;
            let dk = m.cfg.linear_key_head_dim as i32;
            let dv = m.cfg.linear_value_head_dim as i32;
            tracing::info!(
                b,
                T = gdn_warmup_t,
                hk,
                hv,
                dk,
                dv,
                "arch::load_model: pre-warm gated_delta_prefill_ops"
            );
            let warmup_result = gdn_warmup(b, gdn_warmup_t, hk, hv, dk, dv);
            match warmup_result {
                Ok(()) => tracing::info!(
                    warmup_ms = t_gdn_warm.elapsed().as_millis(),
                    "arch::load_model: GDN compile-warmup complete"
                ),
                Err(e) => tracing::warn!(
                    error = %e,
                    "arch::load_model: GDN compile-warmup failed (cold TTFT not mitigated)"
                ),
            }
        }
    }

    // -- Store phases + emit tracing event ------------------------------------
    let total_load_ms = t_total_start.elapsed().as_millis() as u64;
    let phases = LoadPhases {
        mmap_ms,
        dequant_ms,
        gpu_residency_ms,
        first_kernel_ready_ms,
        total_load_ms,
    };
    tracing::info!(
        mmap_ms,
        dequant_ms,
        gpu_residency_ms,
        first_kernel_ready_ms,
        total_load_ms,
        arch = arch_str,
        "arch::load_model: load-time phases (N17)"
    );
    {
        let mut guard = LAST_LOAD_PHASES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = phases;
    }

    Ok(arch)
}

/// Pre-warm the GDN compile-cache by running one traced call of
/// `gated_delta_prefill_ops` at the production chunk shape.
///
/// Extracted from `load_model` to keep that function below 200 LOC.
fn gdn_warmup(b: i32, t: i32, hk: i32, hv: i32, dk: i32, dv: i32) -> Result<()> {
    let zeros_f32 = |shape: &[i32]| {
        let n = shape.iter().map(|&x| x as usize).product::<usize>();
        let bytes = vec![0u8; n * 4];
        Array::from_bytes(&bytes, shape, rmlx_mlx::Dtype::F32)
    };
    let zeros_bf16 = |shape: &[i32]| {
        let n = shape.iter().map(|&x| x as usize).product::<usize>();
        let bytes = vec![0u8; n * 2];
        Array::from_bytes(&bytes, shape, rmlx_mlx::Dtype::Bf16)
    };
    let q = zeros_bf16(&[b, t, hk, dk])?;
    let k = zeros_bf16(&[b, t, hk, dk])?;
    let v = zeros_bf16(&[b, t, hv, dv])?;
    let g = zeros_f32(&[b, t, hv])?;
    let beta = zeros_bf16(&[b, t, hv])?;
    let state_in = zeros_f32(&[b, hv, dv, dk])?;
    let (y_out, s_out) = crate::gated_delta_msl::gated_delta_prefill_ops(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &state_in,
        Device::Gpu,
    )?;
    // Force evaluation to complete Metal compilation.
    y_out.eval()?;
    s_out.eval()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Smoke-probe seed prompt
// ---------------------------------------------------------------------------

/// Fixed, deterministic seed prompt for the smoke probe (B5b).
///
/// B5 fed the probe bare BOS (no prompt). That produced *false* degeneration on
/// some healthy snapshots: `gemma-4-26b-a4b-it-mxfp8` answers
/// "The capital of France is Paris." for a real prompt yet loops the Korean
/// token `로` from bare BOS. Seeding with a tiny fixed instruction makes healthy
/// instruction-tuned models generate real text, so a degenerate window is
/// genuine evidence of a broken snapshot rather than a bare-BOS artifact.
///
/// Plain string tokenized with `add_special_tokens=false` (BOS is prepended
/// explicitly by `smoke_prompt_ids`). No chat-template dependency is pulled
/// into `rmlx-models` -- the template engine lives in `rmlx-server`, and the
/// probe must stay usable from `rmlx-cli` without that dep.
pub const SMOKE_PROMPT: &str = "What is the capital of France?";

/// Build the deterministic smoke-probe input: `[bos_id]` followed by
/// `SMOKE_PROMPT` tokenized with `add_special_tokens=false`.
///
/// Single source of truth for the seed so both callers (`run_smoke_probe` and
/// `rmlx info --probe-smoke`) feed byte-identical input. Greedy/temp=0
/// generation on top of this is fully deterministic across models and calls.
pub fn smoke_prompt_ids(tokenizer: &tokenizers::Tokenizer, bos_id: u32) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(SMOKE_PROMPT, false)
        .map_err(|e| Error::Model(format!("smoke_prompt_ids: encode failed: {e}")))?;
    let mut ids = Vec::with_capacity(1 + enc.get_ids().len());
    ids.push(bos_id);
    ids.extend_from_slice(enc.get_ids());
    Ok(ids)
}

// ---------------------------------------------------------------------------
// run_smoke_probe
// ---------------------------------------------------------------------------

/// Run the 8-token smoke probe on a model snapshot and return the verdict.
///
/// Loads the model via `load_model`, resolves the BOS token from
/// `tokenizer_config.json`, runs greedy generation for 8 steps, and returns
/// a `SmokeVerdict`. Used by the server's `--require-smoke-probe` gate (B5).
///
/// Returns `Err` only for hard load/tokenizer failures. `Ok(verdict)` where
/// `verdict != SmokeVerdict::Ok` means the snapshot is broken but loadable.
pub fn run_smoke_probe(
    model_dir: &Path,
    device: Device,
    kv_quant: Option<rmlx_kv_quant::KvQuant>,
    max_ctx_override: Option<i32>,
) -> Result<crate::gemma4::SmokeVerdict> {
    let model = load_model(model_dir, device)?;

    // Resolve BOS token id from tokenizer_config.json.
    let bos_id = resolve_bos_id(model_dir)?;
    tracing::info!(bos_id, "run_smoke_probe: resolved BOS token id");

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::Model(format!("run_smoke_probe: load tokenizer: {e}")))?;

    // B5b: seed with a fixed deterministic prompt instead of bare BOS so
    // healthy instruction-tuned snapshots generate real text.
    let prompt_ids = smoke_prompt_ids(&tokenizer, bos_id)?;
    tracing::info!(
        prompt_len = prompt_ids.len(),
        seed = SMOKE_PROMPT,
        "run_smoke_probe: seeded smoke prompt"
    );

    let sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = crate::sampler::PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    let steps = model.generate_greedy(
        &tokenizer,
        &prompt_ids,
        8,
        device,
        kv_quant, // None = use arch default
        max_ctx_override,
        1,   // single-slot -- no multi-slot needed for smoke probe
        &[], // no EOS stop -- force full 8 steps
        &mut |_| None,
        None, // no sampler constraint
        &sampler_cfg,
        &mut rng,
        &penalty_cfg,
        &mut token_history,
    )?;

    Ok(Architecture::classify_smoke(&steps))
}

/// Resolve BOS token id from `tokenizer_config.json` + `tokenizer.json`.
///
/// Fallback chain: `bos_token` -> `<bos>` -> `<|im_start|>` -> `eos_token` ->
/// `<|endoftext|>`. Returns `Error::Model` if nothing resolves.
fn resolve_bos_id(model_dir: &Path) -> Result<u32> {
    let cfg_path = model_dir.join("tokenizer_config.json");
    let data = std::fs::read(&cfg_path)
        .map_err(|e| Error::Model(format!("cannot read tokenizer_config.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|e| Error::Model(format!("malformed tokenizer_config.json: {e}")))?;

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

    let tk_path = model_dir.join("tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| Error::Model(format!("cannot load tokenizer.json: {e}")))?;

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

    for cand in &candidates {
        if let Some(id) = tk.token_to_id(cand) {
            tracing::debug!(token = %cand, id, "resolve_bos_id: resolved");
            return Ok(id);
        }
    }
    Err(Error::Model(format!(
        "cannot resolve BOS token in {}: tried {candidates:?}",
        model_dir.display()
    )))
}
