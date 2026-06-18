//! Image-prompt construction helpers shared by ArchGenerator.
//!
//! - `VisionBundle` — arch-specific multimodal components (loaded once)
//! - `build_image_prompt` — Gemma4/Gemma3 image embedding + token splice
//! - `run_qwen3vl_image` — Qwen3-VL-MoE full image decode
//! - `rmlx_server_load_image` — thin wrapper mapping load_image errors

use rmlx_core::Error;

// ── image-prompt construction ─────────────────────────────────────────

/// Gemma4 begin-/end-of-image marker token ids. Re-exported from the single
/// source of truth in `rmlx_models::gemma4` so the server and the model agree
/// on these correctness-critical ids.
pub(crate) use rmlx_models::gemma4::BOI_TOKEN_ID as GEMMA4_BOI_TOKEN_ID;
pub(crate) use rmlx_models::gemma4::EOI_TOKEN_ID as GEMMA4_EOI_TOKEN_ID;

/// bundle of the multimodal components needed to turn image bytes
/// into scattered `inputs_embeds`. Loaded once per model. One variant per
/// vision-capable architecture (Gemma4's custom SigLIP, Gemma3's standard SigLIP).
#[allow(missing_debug_implementations)]
pub(crate) enum VisionBundle {
    /// Gemma4 custom SigLIP tower + multimodal embedder + processor.
    Gemma4 {
        vision: rmlx_models::gemma4::VisionModel,
        embedder: rmlx_models::gemma4::MultimodalEmbedder,
        processor: rmlx_models::gemma4::Gemma4ImageProcessor,
    },
    /// Gemma4 **unified** (`Gemma4UnifiedForConditionalGeneration`, 12B):
    /// encoder-free vision embedder (no SigLIP tower) + processor.
    Gemma4Unified {
        embedder: rmlx_models::gemma4::UnifiedVisionEmbedder,
        processor: rmlx_models::gemma4::Gemma4ImageProcessor,
    },
    /// Gemma3 standard SigLIP tower + multimodal projector + processor.
    Gemma3 {
        vision: rmlx_models::gemma3::VisionModel,
        projector: rmlx_models::gemma3::MultiModalProjector,
        processor: rmlx_models::gemma3::Gemma3ImageProcessor,
    },
    /// Qwen3-VL-MoE ViT + smart-resize image processor.
    Qwen3VlMoe {
        vision: rmlx_models::qwen3_vl_moe::Qwen3VlMoeVision,
        processor: rmlx_models::qwen3_vl_moe::Qwen3VlImageConfig,
    },
}

/// load + preprocess images, expand the prompt with per-image soft-token
/// blocks, and build the scatter-merged `inputs_embeds`.
///
/// Pipeline (mirrors mlx-vlm `processing_gemma4` + `get_input_embeddings`):
/// 1. `image_io::load_image` → bytes (HTTP / data-URL / file).
/// 2. `processor.preprocess` → `Gemma4PixelValues` (+ `num_soft_tokens`).
/// 3. Build the image block per image: `<|image>` + `num_soft` × image-token
///    placeholders.
/// 4. `gemma4::build_inputs_embeds` runs the vision tower + multimodal
///    projector and returns the fused embedding tensor.
///
/// Returns `(augmented_ids, inputs_embeds [1, seq, hidden], masked_ids [seq])`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
pub(crate) fn build_image_prompt(
    arch: &rmlx_models::arch::Architecture,
    vb: &VisionBundle,
    sources: &[String],
    prompt_tokens: &[u32],
    device: rmlx_mlx::Device,
    mm_cache: Option<&rmlx_models::multimodal_cache::MultimodalCache>,
    model_sig: u64,
) -> rmlx_core::Result<(Vec<u32>, rmlx_mlx::Array, rmlx_mlx::Array)> {
    // Splice per-image token blocks (`<boi>` + N×image-token + `<eoi>`) in after
    // the prompt's leading BOS token so the image conditions the whole turn
    // causally. Shared between archs; only the token ids + soft counts differ.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn splice(prompt_tokens: &[u32], blocks: &[Vec<u32>]) -> Vec<u32> {
        let total: usize = blocks.iter().map(Vec::len).sum();
        let insert_at = usize::from(!prompt_tokens.is_empty());
        let mut aug = Vec::with_capacity(prompt_tokens.len() + total);
        aug.extend_from_slice(&prompt_tokens[..insert_at]);
        for b in blocks {
            aug.extend_from_slice(b);
        }
        aug.extend_from_slice(&prompt_tokens[insert_at..]);
        aug
    }

    match vb {
        VisionBundle::Gemma4 {
            vision,
            embedder,
            processor,
        } => {
            let model = arch.as_gemma4().ok_or_else(|| {
                Error::Other("image input requires the Gemma4 architecture".to_owned())
            })?;
            let image_token_id = rmlx_models::gemma4::IMAGE_TOKEN_ID;

            let mut pixels = Vec::with_capacity(sources.len());
            for (i, src) in sources.iter().enumerate() {
                let bytes = rmlx_server_load_image(src)?;
                let pv = processor
                    .preprocess(&bytes)
                    .map_err(|e| Error::Other(format!("image {i} preprocess failed: {e}")))?;
                tracing::info!(
                    image_idx = i,
                    width = pv.width,
                    height = pv.height,
                    num_soft_tokens = pv.num_soft_tokens,
                    "Gemma4 image preprocessed"
                );
                pixels.push(pv);
            }

            let blocks: Vec<Vec<u32>> = pixels
                .iter()
                .map(|pv| {
                    let mut b = Vec::with_capacity(pv.num_soft_tokens + 2);
                    b.push(GEMMA4_BOI_TOKEN_ID);
                    b.extend(std::iter::repeat_n(image_token_id, pv.num_soft_tokens));
                    b.push(GEMMA4_EOI_TOKEN_ID);
                    b
                })
                .collect();
            let aug_ids = splice(prompt_tokens, &blocks);

            let total_soft: usize = pixels.iter().map(|p| p.num_soft_tokens).sum();
            let in_prompt = aug_ids.iter().filter(|&&t| t == image_token_id).count();
            tracing::info!(
                images = pixels.len(),
                soft_tokens = total_soft,
                image_tokens_in_prompt = in_prompt,
                aug_len = aug_ids.len(),
                "built Gemma4 image prompt"
            );

            let (embeds, masked_ids) = rmlx_models::gemma4::build_inputs_embeds(
                model, vision, embedder, &pixels, &aug_ids, device, mm_cache, model_sig,
            )?;
            Ok((aug_ids, embeds, masked_ids))
        }
        VisionBundle::Gemma4Unified {
            embedder,
            processor,
        } => {
            // The unified 12B loads through the Gemma4 text architecture; the
            // encoder-free embedder replaces the SigLIP tower.
            let model = arch.as_gemma4().ok_or_else(|| {
                Error::Other("image input requires the Gemma4 architecture".to_owned())
            })?;
            let image_token_id = rmlx_models::gemma4::IMAGE_TOKEN_ID;

            let mut pixels = Vec::with_capacity(sources.len());
            for (i, src) in sources.iter().enumerate() {
                let bytes = rmlx_server_load_image(src)?;
                let pv = processor
                    .preprocess(&bytes)
                    .map_err(|e| Error::Other(format!("image {i} preprocess failed: {e}")))?;
                let n_soft = rmlx_models::gemma4::unified_num_soft_tokens(
                    pv.height,
                    pv.width,
                    embedder.config(),
                );
                tracing::info!(
                    image_idx = i,
                    width = pv.width,
                    height = pv.height,
                    num_soft_tokens = n_soft,
                    "Gemma4-unified image preprocessed"
                );
                pixels.push((pv, n_soft));
            }

            let blocks: Vec<Vec<u32>> = pixels
                .iter()
                .map(|(_, n_soft)| {
                    let mut b = Vec::with_capacity(n_soft + 2);
                    b.push(GEMMA4_BOI_TOKEN_ID);
                    b.extend(std::iter::repeat_n(image_token_id, *n_soft));
                    b.push(GEMMA4_EOI_TOKEN_ID);
                    b
                })
                .collect();
            let aug_ids = splice(prompt_tokens, &blocks);

            let total_soft: usize = pixels.iter().map(|(_, n)| *n).sum();
            let in_prompt = aug_ids.iter().filter(|&&t| t == image_token_id).count();
            tracing::info!(
                images = pixels.len(),
                soft_tokens = total_soft,
                image_tokens_in_prompt = in_prompt,
                aug_len = aug_ids.len(),
                "built Gemma4-unified image prompt"
            );

            let pv_only: Vec<rmlx_models::gemma4::Gemma4PixelValues> =
                pixels.into_iter().map(|(pv, _)| pv).collect();
            let (embeds, masked_ids) = rmlx_models::gemma4::build_unified_inputs_embeds(
                model, embedder, &pv_only, &aug_ids, device, mm_cache, model_sig,
            )?;
            Ok((aug_ids, embeds, masked_ids))
        }
        VisionBundle::Gemma3 {
            vision,
            projector,
            processor,
        } => {
            let model = arch.as_gemma3().ok_or_else(|| {
                Error::Other("image input requires the Gemma3 architecture".to_owned())
            })?;
            let image_token_id = rmlx_models::gemma3::IMAGE_TOKEN_ID;

            let mut pixels = Vec::with_capacity(sources.len());
            for (i, src) in sources.iter().enumerate() {
                let bytes = rmlx_server_load_image(src)?;
                let pv = processor
                    .preprocess(&bytes)
                    .map_err(|e| Error::Other(format!("image {i} preprocess failed: {e}")))?;
                tracing::info!(
                    image_idx = i,
                    width = pv.width,
                    height = pv.height,
                    num_soft_tokens = pv.num_soft_tokens,
                    "Gemma3 image preprocessed"
                );
                pixels.push(pv);
            }

            let blocks: Vec<Vec<u32>> = pixels
                .iter()
                .map(|pv| {
                    let mut b = Vec::with_capacity(pv.num_soft_tokens + 2);
                    b.push(rmlx_models::gemma3::BOI_TOKEN_ID);
                    b.extend(std::iter::repeat_n(image_token_id, pv.num_soft_tokens));
                    b.push(rmlx_models::gemma3::EOI_TOKEN_ID);
                    b
                })
                .collect();
            let aug_ids = splice(prompt_tokens, &blocks);

            let total_soft: usize = pixels.iter().map(|p| p.num_soft_tokens).sum();
            let in_prompt = aug_ids.iter().filter(|&&t| t == image_token_id).count();
            tracing::info!(
                images = pixels.len(),
                soft_tokens = total_soft,
                image_tokens_in_prompt = in_prompt,
                aug_len = aug_ids.len(),
                "built Gemma3 image prompt"
            );

            let (embeds, ids) = rmlx_models::gemma3::build_inputs_embeds(
                model, vision, projector, &pixels, &aug_ids, device, mm_cache, model_sig,
            )?;
            // Gemma3 has no masked-ids concept; pass the plain ids array through
            // the same (Vec, Array, Array) shape so the caller is arch-agnostic.
            Ok((aug_ids, embeds, ids))
        }
        // Qwen3-VL-MoE is handled by `run_qwen3vl_image` before this function is
        // reached (its vision output does not fit the embeds-triple shape).
        VisionBundle::Qwen3VlMoe { .. } => Err(Error::Other(
            "internal: Qwen3-VL image path must route through run_qwen3vl_image".to_owned(),
        )),
    }
}

/// Qwen3-VL-MoE `<|vision_start|>` token id.
pub(crate) const QWEN3VL_VISION_START_ID: u32 = 151_652;
/// Qwen3-VL-MoE `<|vision_end|>` token id.
pub(crate) const QWEN3VL_VISION_END_ID: u32 = 151_653;
/// Qwen3-VL-MoE `<|image_pad|>` token id (the scatter target).
pub(crate) const QWEN3VL_IMAGE_PAD_ID: u32 = 151_655;

/// full Qwen3-VL-MoE image request: preprocess images, run the ViT,
/// splice the per-image vision block (`<|vision_start|>` + N×`<|image_pad|>` +
/// `<|vision_end|>`) after the prompt's leading token, then decode via the
/// model's image branch (scatter at image_pad positions + 3D M-RoPE +
/// deepstack injection).
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
pub(crate) fn run_qwen3vl_image(
    arch: &rmlx_models::arch::Architecture,
    vb: &VisionBundle,
    sources: &[String],
    prompt_tokens: &[u32],
    n_tokens: usize,
    device: rmlx_mlx::Device,
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    eos_ids: &[u32],
    tokenizer: &tokenizers::Tokenizer,
    step_fn: &mut dyn FnMut(&rmlx_models::ProbeStep) -> Option<u32>,
    constraint: Option<&mut dyn rmlx_models::ConstraintEngine>,
    sampler_cfg: &rmlx_models::SamplerConfig,
    rng: &mut rmlx_models::Pcg32,
    penalty_cfg: &rmlx_models::PenaltyConfig,
    token_history: &mut Vec<u32>,
    mm_cache: Option<&rmlx_models::multimodal_cache::MultimodalCache>,
    model_sig: u64,
) -> rmlx_core::Result<Vec<rmlx_models::ProbeStep>> {
    // Registering a thread-local GPU stream + CommandEncoder once per thread entry point.
    // tokio blocking-pool threads start with no GPU stream context; MLX's array
    // materialisation then fails with "There is no Stream(gpu, 0) in current thread".
    // The ViT pass and decode below both materialise arrays on this thread; zero ML-semantic effect.
    if device == rmlx_mlx::Device::Gpu {
        rmlx_mlx::ensure_gpu_default_stream();
    }

    let (vision, processor) = match vb {
        VisionBundle::Qwen3VlMoe { vision, processor } => (vision, processor),
        _ => {
            return Err(Error::Other(
                "run_qwen3vl_image called with non-Qwen3VL vision bundle".to_owned(),
            ))
        }
    };
    let model = arch.as_qwen3vl_moe().ok_or_else(|| {
        Error::Other("image input requires the Qwen3-VL-MoE architecture".to_owned())
    })?;

    // serves a single image per request (the headline validation). Multi-
    // image is a straightforward extension (concat ViT outputs + grids) but is
    // out of scope here; reject >1 with a clear error rather than silently
    // mis-scattering.
    if sources.len() != 1 {
        return Err(Error::Other(format!(
            "Qwen3-VL image path currently serves exactly one image per request (got {})",
            sources.len()
        )));
    }

    // Qwen3-VL image-conditioned attention is sensitive to KV quantization:
    // K8V8 (the global `for_arch_default`) measurably degraded the image
    // description (incoherent output) while bf16 reproduces the mlx-vlm
    // reference answer exactly. Default the image branch to unquantised bf16
    // KV unless the operator explicitly overrode `--kv-quant`.
    let kv_quant = kv_quant_override.unwrap_or(rmlx_kv_quant::KvQuant::None);

    let bytes = rmlx_server_load_image(&sources[0])?;
    let pv = rmlx_models::qwen3_vl_moe::preprocess(&bytes, processor)
        .map_err(|e| Error::Other(format!("Qwen3-VL image preprocess failed: {e}")))?;
    tracing::info!(
        grid_thw = ?pv.grid_thw,
        num_soft_tokens = pv.num_soft_tokens,
        "Qwen3-VL image preprocessed"
    );

    // Run the ViT once.
    // Short-circuit on a cache hit. Qwen3-VL returns a multi-array
    // VisionOutput (image_embeds + deepstack_embeds[]); cache them together
    // via `put_many` so a hit skips the entire ViT pass.
    let forward = || -> Result<rmlx_models::qwen3_vl_moe::VisionOutput, Error> {
        vision
            .forward(&pv.pixel_values, pv.grid_thw, device)
            .map_err(|e| Error::Other(format!("Qwen3-VL vision tower failed: {e}")))
    };
    let vout = if let Some(cache) = mm_cache {
        let key_bytes = rmlx_models::multimodal_cache::pixel_f32_bytes(&pv.pixel_values);
        let (_gt, gh, gw) = pv.grid_thw;
        let key = rmlx_models::multimodal_cache::MmCacheKey::image_key(
            key_bytes,
            u16::try_from(gh).unwrap_or(u16::MAX),
            u16::try_from(gw).unwrap_or(u16::MAX),
            3,
            rmlx_models::multimodal_cache::MmDtype::F32,
            model_sig,
        );
        if let Some(mut arrays) = cache.get_many(&key) {
            // First array is the merged image_embeds, the rest are deepstack.
            // The cache only inserts non-empty multi-array snapshots
            // (`put_many` is only called below with `1 + deepstack_embeds.len()`
            // arrays). An empty Vec means an internal invariant broke.
            debug_assert!(!arrays.is_empty(), "mm_cache: qwen3vl entry empty");
            if arrays.is_empty() {
                tracing::warn!(
                    "mm_cache: qwen3vl entry unexpectedly empty; falling back to forward without caching"
                );
                forward()?
            } else {
                let image_embeds = arrays.remove(0);
                let deepstack_embeds = arrays;
                rmlx_models::qwen3_vl_moe::VisionOutput {
                    image_embeds,
                    deepstack_embeds,
                }
            }
        } else {
            let computed = forward()?;
            // Build a multi-array snapshot to insert.
            let expected = 1 + computed.deepstack_embeds.len();
            let mut snapshot: Vec<rmlx_mlx::Array> = Vec::with_capacity(expected);
            let mut total_bytes = 0usize;
            let mut clone_ok = true;
            match computed.image_embeds.try_clone() {
                Ok(c) => match rmlx_models::multimodal_cache::array_byte_size(&c) {
                    Ok(sz) => {
                        total_bytes += sz;
                        snapshot.push(c);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "mm_cache: qwen3vl image_embeds byte_size failed; not caching");
                        clone_ok = false;
                    }
                },
                Err(e) => {
                    tracing::warn!(error = ?e, "mm_cache: qwen3vl image_embeds clone failed; not caching");
                    clone_ok = false;
                }
            }
            if clone_ok {
                for (idx, d) in computed.deepstack_embeds.iter().enumerate() {
                    match d.try_clone() {
                        Ok(c) => match rmlx_models::multimodal_cache::array_byte_size(&c) {
                            Ok(sz) => {
                                total_bytes += sz;
                                snapshot.push(c);
                            }
                            Err(e) => {
                                tracing::warn!(deepstack_idx = idx, error = %e, "mm_cache: qwen3vl deepstack byte_size failed; not caching");
                                clone_ok = false;
                                break;
                            }
                        },
                        Err(e) => {
                            tracing::warn!(deepstack_idx = idx, error = ?e, "mm_cache: qwen3vl deepstack clone failed; not caching");
                            clone_ok = false;
                            break;
                        }
                    }
                }
            }
            // Only persist if the clones succeeded for the full set; otherwise
            // drop (a partial entry would corrupt the deepstack count).
            if clone_ok && snapshot.len() == expected {
                cache.put_many(key, snapshot, total_bytes);
            }
            computed
        }
    } else {
        forward()?
    };

    // Splice the vision block `<|vision_start|>` + N×`<|image_pad|>` +
    // `<|vision_end|>` immediately after the user-turn opener `<|im_start|>user\n`
    // (token ids 151644, 872, 198) and before the user text — matching the
    // mlx-vlm Qwen3-VL prompt format
    // `<|im_start|>user\n<|vision_start|>...<|vision_end|><text><|im_end|>`. If
    // that opener is not found (non-standard template), fall back to inserting
    // after the leading token.
    let mut block = Vec::with_capacity(pv.num_soft_tokens + 2);
    block.push(QWEN3VL_VISION_START_ID);
    block.extend(std::iter::repeat_n(
        QWEN3VL_IMAGE_PAD_ID,
        pv.num_soft_tokens,
    ));
    block.push(QWEN3VL_VISION_END_ID);

    const IM_START: u32 = 151_644;
    const USER: u32 = 872;
    const NL: u32 = 198;
    let insert_at = prompt_tokens
        .windows(3)
        .position(|w| w == [IM_START, USER, NL])
        .map_or(usize::from(!prompt_tokens.is_empty()), |p| p + 3);
    let mut aug_ids = Vec::with_capacity(prompt_tokens.len() + block.len());
    aug_ids.extend_from_slice(&prompt_tokens[..insert_at]);
    aug_ids.extend_from_slice(&block);
    aug_ids.extend_from_slice(&prompt_tokens[insert_at..]);

    let in_prompt = aug_ids
        .iter()
        .filter(|&&t| t == QWEN3VL_IMAGE_PAD_ID)
        .count();
    tracing::info!(
        soft_tokens = pv.num_soft_tokens,
        image_pads_in_prompt = in_prompt,
        aug_len = aug_ids.len(),
        "built Qwen3-VL image prompt"
    );

    rmlx_models::qwen3_vl_moe::generate_image(
        &model.text,
        tokenizer,
        &aug_ids,
        &vout,
        &[pv.grid_thw],
        model.image_token_id,
        model.spatial_merge_size,
        n_tokens,
        device,
        kv_quant,
        eos_ids,
        step_fn,
        constraint,
        sampler_cfg,
        rng,
        penalty_cfg,
        token_history,
    )
}

/// thin wrapper mapping `image_io::load_image`'s `String` error into the
/// crate `Error`, with a fixed 20s fetch timeout for remote sources.
pub(crate) fn rmlx_server_load_image(source: &str) -> rmlx_core::Result<Vec<u8>> {
    crate::image_io::load_image(source, std::time::Duration::from_secs(20))
        .map_err(|e| Error::Other(format!("image load failed: {e}")))
}
